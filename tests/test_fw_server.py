"""crosspad_hil.fw_server against a real socket on localhost. No board, no gh."""
from __future__ import annotations

import http.client
import json
import subprocess
import urllib.error
import urllib.request
from pathlib import Path

import pytest

from crosspad_hil import fw_server

TAG = "v1.3.0-rc1"
DATA = bytes(range(256)) * 64          # 16 kB


def served(tmp_path: Path, kit: str = "full") -> fw_server.ServedVersion:
    names = fw_server.release_names(TAG, "v2", kit)
    files = {}
    for key, name in names.items():
        if key == "icon":
            continue
        p = tmp_path / name
        p.write_bytes(DATA if key == "bin" else key.encode() * 10)
        files[key] = p
    return fw_server.ServedVersion(kit=kit, version=TAG.lstrip("v"), tag=TAG, rev="v2", channel="beta",
                                   tier="official", files=files)


def get(srv, path, headers=None):
    conn = http.client.HTTPConnection("127.0.0.1", srv.port, timeout=5)
    conn.request("GET", path, headers=headers or {})
    res = conn.getresponse()
    try:
        body = res.read()
    except http.client.IncompleteRead as e:
        body = e.partial
    conn.close()
    return res, body


@pytest.fixture
def srv(tmp_path):
    s = fw_server.FwServer([served(tmp_path)], host="127.0.0.1")
    with s:
        yield s


def test_ping_and_catalog(srv):
    res, body = get(srv, "/api/device/ping")
    assert res.status == 200 and json.loads(body) == {"ok": True}
    res, body = get(srv, "/api/device/catalog?rev=v2")
    cat = json.loads(body)
    assert res.getheader("Date")
    assert cat["v"] == 1 and cat["kits"][0]["kit"] == "full"
    v = cat["kits"][0]["versions"][0]
    assert v["version"] == "1.3.0-rc1" and v["channel"] == "beta"
    assert v["bin"] == {"name": f"CrossPad-{TAG}-v2-full.bin", "size": len(DATA)}
    assert json.loads(get(srv, "/api/device/catalog?rev=v1")[1])["kits"] == []


def test_ranges(srv):
    name = f"CrossPad-{TAG}-v2-full.bin"
    res, body = get(srv, f"/api/device/file/{TAG}/{name}")
    assert res.status == 200 and body == DATA
    res, body = get(srv, f"/api/device/file/{TAG}/{name}", {"Range": "bytes=100-"})
    assert res.status == 206 and body == DATA[100:]
    assert res.getheader("Content-Range") == f"bytes 100-{len(DATA) - 1}/{len(DATA)}"
    res, _ = get(srv, f"/api/device/file/{TAG}/{name}", {"Range": f"bytes={len(DATA)}-"})
    assert res.status == 416 and res.getheader("Content-Range") == f"bytes */{len(DATA)}"
    assert get(srv, f"/api/device/file/{TAG}/nope.bin")[0].status == 404
    assert [e["range"] for e in srv.requests_for(name)] == [None, "bytes=100-", f"bytes={len(DATA)}-"]


def test_drop_once_flip_and_ignore_range(srv):
    name = f"CrossPad-{TAG}-v2-full.bin"
    srv.faults.drop_once_at[name] = 1000
    _, body = get(srv, f"/api/device/file/{TAG}/{name}")
    assert body == DATA[:1000] and srv.log[-1]["cut"]
    _, body = get(srv, f"/api/device/file/{TAG}/{name}", {"Range": "bytes=1000-"})
    assert body == DATA[1000:]                                  # dropped once only
    srv.faults.flip_at[name] = 5
    _, body = get(srv, f"/api/device/file/{TAG}/{name}")
    assert body[5] == DATA[5] ^ 0xFF and body[:5] == DATA[:5]
    srv.faults.reset()
    srv.faults.ignore_range.add(name)
    res, body = get(srv, f"/api/device/file/{TAG}/{name}", {"Range": "bytes=10-"})
    assert res.status == 200 and body == DATA


def test_alias_serves_real_files_under_another_entry(tmp_path):
    (tmp_path / "r").mkdir()
    real = served(tmp_path / "r")
    bad = fw_server.alias(real, "zzsig", "0.0.1", tmp_path / "w", break_sig=True)
    assert bad.files["manifest"].read_bytes() == real.files["manifest"].read_bytes()
    assert bad.files["sig"].read_bytes() != real.files["sig"].read_bytes()
    assert bad.files["bin"].name == f"CrossPad-{TAG}-v2-zzsig.bin"
    with fw_server.FwServer([real, bad], host="127.0.0.1") as s:
        kits = {k["kit"] for k in json.loads(get(s, "/api/device/catalog?rev=v2")[1])["kits"]}
        assert kits == {"full", "zzsig"}


@pytest.mark.parametrize("with_icon", [True, False])
def test_fetch_release_downloads_once_with_gh(tmp_path, monkeypatch, with_icon):
    src = tmp_path / "release"
    src.mkdir()
    for key, name in fw_server.release_names(TAG, "v2", "full").items():
        if key != "icon" or with_icon:
            (src / name).write_bytes(b"x")
    (src / f"CrossPad-{TAG}-kits.json").write_text(json.dumps(
        {"v": 1, "tag": TAG, "kits": [{"kit": "full", "tier": "gold", "revs": ["v2"]}]}))
    calls = []

    def fake_run(cmd, **kw):
        calls.append(cmd)
        if cmd[2] == "view":
            return subprocess.CompletedProcess(cmd, 0, json.dumps({"body": "- fix: x", "assets": [
                {"name": n} for n in [*fw_server.release_names(TAG, "v2", "full").values(), f"CrossPad-{TAG}-kits.json"]
                if "icon" not in n or with_icon]}), "")
        dest = Path(cmd[cmd.index("-D") + 1])
        for i, tok in enumerate(cmd):
            if tok == "-p" and (src / cmd[i + 1]).is_file():
                (dest / cmd[i + 1]).write_bytes((src / cmd[i + 1]).read_bytes())
        return subprocess.CompletedProcess(cmd, 0, "", "")

    monkeypatch.setattr(fw_server.subprocess, "run", fake_run)
    v = fw_server.fetch_release(TAG, "v2", "full", cache=tmp_path / "cache")
    assert v.version == "1.3.0-rc1" and v.channel == "beta" and v.tier == "official" and v.notes == "- fix: x"
    fw_server.fetch_release(TAG, "v2", "full", cache=tmp_path / "cache")
    packages = [c for c in calls if c[2] == "download" and f"CrossPad-{TAG}-v2-full.bin" in c]
    assert len(packages) == 1 and ("icon" in v.files) is with_icon
    assert not any(c[2] == "download" and any("icon" in t for t in c) for c in calls if not with_icon)


TAG_RULES = json.loads((Path(__file__).parent / "fixtures" / "fw_tag_rules.json").read_text())


@pytest.mark.parametrize("tag,channel", sorted(TAG_RULES["channel"].items()))
def test_channel_is_the_shared_tag_rule(tag, channel):
    """A verbatim copy of platform-idf tools/tests/fw_tag_rules.json, as in the Worker."""
    assert fw_server.tag_channel(tag) == channel


def test_tag_limit_is_the_shared_one():
    assert fw_server.TAG_MAX == TAG_RULES["tag_max"]
    assert all(fw_server.tag_ok(t) for t in TAG_RULES["accepted"])
    assert not any(fw_server.tag_ok(t) for t in TAG_RULES["refused"])


def test_newest_kit_release_skips_a_tag_too_long_for_a_board(monkeypatch):
    long = "v1.0.0-abcdefghijklmnopq"
    viewed = []

    def fake_run(cmd, **kw):
        if cmd[1:3] == ["release", "list"]:
            return subprocess.CompletedProcess(cmd, 0, json.dumps([{"tagName": long}, {"tagName": TAG}]), "")
        viewed.append(cmd[3])
        assets = [{"name": f"CrossPad-{cmd[3]}-kits.json"}]
        return subprocess.CompletedProcess(cmd, 0, json.dumps({"assets": assets}), "")

    monkeypatch.setattr(fw_server.subprocess, "run", fake_run)
    assert fw_server.newest_kit_release() == TAG and viewed == [TAG]


def test_newest_kit_release_with_icons_skips_a_release_without_them(monkeypatch):
    newer = "v1.1.0-fwstore.5"

    def fake_run(cmd, **kw):
        if cmd[1:3] == ["release", "list"]:
            return subprocess.CompletedProcess(cmd, 0, json.dumps([{"tagName": newer}, {"tagName": TAG}]), "")
        assets = [{"name": f"CrossPad-{cmd[3]}-kits.json"}]
        if cmd[3] == TAG:
            assets.append({"name": f"CrossPad-{TAG}-v2-full-icon.bin"})
        return subprocess.CompletedProcess(cmd, 0, json.dumps({"assets": assets}), "")

    monkeypatch.setattr(fw_server.subprocess, "run", fake_run)
    assert fw_server.newest_kit_release() == newer
    assert fw_server.newest_kit_release(icons=True) == TAG


# Cases mirror the Worker's test/unit/device-api.test.ts "Range" table.
RANGE_CASES = [
    (None, 100, "full"),
    ("bytes=10-", 100, (10, 99)),
    ("bytes=10-19", 100, (10, 19)),
    ("bytes=90-200", 100, (90, 99)),
    ("bytes=-10", 100, (90, 99)),
    ("bytes=-500", 100, (0, 99)),
    ("bytes=100-", 100, "unsatisfiable"),
    ("bytes=-0", 100, "unsatisfiable"),
    ("bytes=-", 100, "full"),
    ("bytes=0-1,5-6", 100, "full"),
    ("items=0-1", 100, "full"),
    ("bytes=20-10", 100, "full"),
    ("bytes=-5", 0, "unsatisfiable"),
    ("bytes=0-", 0, "unsatisfiable"),
    (None, 0, "full"),
]


@pytest.mark.parametrize("header,size,want", RANGE_CASES)
def test_parse_range_matches_the_worker(header, size, want):
    assert fw_server.parse_range(header, size) == want


def test_suffix_reversed_and_etag_over_http(srv):
    import hashlib
    name = f"CrossPad-{TAG}-v2-full.bin"
    url = f"/api/device/file/{TAG}/{name}"
    etag = '"' + hashlib.sha256(DATA).hexdigest() + '"'
    res, body = get(srv, url, {"Range": "bytes=-4"})
    assert res.status == 206 and body == DATA[-4:]
    assert res.getheader("Content-Range") == f"bytes {len(DATA) - 4}-{len(DATA) - 1}/{len(DATA)}"
    assert res.getheader("ETag") == etag and res.getheader("Accept-Ranges") == "bytes"
    res, body = get(srv, url, {"Range": "bytes=20-10"})
    assert res.status == 200 and body == DATA
    res, _ = get(srv, url, {"Range": f"bytes={len(DATA)}-"})
    assert res.status == 416 and res.getheader("Accept-Ranges") == "bytes" and res.getheader("ETag") == etag
    srv.faults.flip_at[name] = 5
    res, _ = get(srv, url)
    assert res.getheader("ETag") == etag                         # the flip is not in the hash
    srv.faults.flip_at[name] = len(DATA) + 10
    assert get(srv, url)[1] == DATA                              # out of range: ignored


def test_empty_file_answers_416_to_any_range(tmp_path):
    v = served(tmp_path)
    v.files["assets"].write_bytes(b"")
    name = v.files["assets"].name
    with fw_server.FwServer([v], host="127.0.0.1") as s:
        res, _ = get(s, f"/api/device/file/{TAG}/{name}", {"Range": "bytes=-5"})
        assert res.status == 416 and res.getheader("Content-Range") == "bytes */0"
        assert get(s, f"/api/device/file/{TAG}/{name}")[0].status == 200


def test_stop_without_start_and_twice(tmp_path):
    s = fw_server.FwServer([served(tmp_path)], host="127.0.0.1")
    s.stop()
    s2 = fw_server.FwServer([served(tmp_path)], host="127.0.0.1")
    s2.start()
    s2.stop()
    s2.stop()


@pytest.mark.parametrize("tag,rev,kit", [
    ("../x", "v2", "full"), ("v1.3", "v2", "full"), ("v1.3.0-" + "a" * 40, "v2", "full"),
    ("v1.0.0-abcdefghijklmnopq", "v2", "full"),
    (TAG, "v2", "Full"), (TAG, "v2", "v2"), (TAG, "v2", "a" * 16), (TAG, "v2", "x*"),
    (TAG, "2", "full"), (TAG, "../v2", "full"),
])
def test_fetch_release_validates_ids_before_any_path(tmp_path, monkeypatch, tag, rev, kit):
    monkeypatch.setattr(fw_server.subprocess, "run", lambda *a, **k: pytest.fail("gh must not run"))
    with pytest.raises(fw_server.HilError) as ei:
        fw_server.fetch_release(tag, rev, kit, cache=tmp_path / "cache")
    assert ei.value.code == "BAD_ARGS" and not (tmp_path / "cache").exists()


def test_fetch_release_malformed_kits_json_is_a_hil_error(tmp_path, monkeypatch):
    dest = tmp_path / "cache" / TAG
    dest.mkdir(parents=True)
    for key, name in fw_server.release_names(TAG, "v2", "full").items():
        if key != "icon":
            (dest / name).write_bytes(b"x")
    (dest / f"CrossPad-{TAG}-kits.json").write_text("{not json")
    listed = [*fw_server.release_names(TAG, "v2", "full").values(), f"CrossPad-{TAG}-kits.json"]

    def fake_run(cmd, **kw):
        # Every file is cached, so only the view runs: no gh needed, and the
        # error has to come from the kits file itself.
        assert cmd[2] == "view", cmd
        return subprocess.CompletedProcess(cmd, 0, json.dumps(
            {"body": "", "assets": [{"name": n} for n in listed if "icon" not in n]}), "")

    monkeypatch.setattr(fw_server.subprocess, "run", fake_run)
    with pytest.raises(fw_server.HilError) as ei:
        fw_server.fetch_release(TAG, "v2", "full", cache=tmp_path / "cache")
    assert ei.value.code == "ENV" and "is not a kits file" in str(ei.value)


def test_fetch_release_without_gh_is_a_hil_error(tmp_path, monkeypatch):
    def no_gh(cmd, **kw):
        raise FileNotFoundError(2, "No such file or directory", cmd[0])

    monkeypatch.setattr(fw_server.subprocess, "run", no_gh)
    with pytest.raises(fw_server.HilError) as ei:
        fw_server.fetch_release(TAG, "v2", "full", cache=tmp_path / "cache")
    assert ei.value.code == "ENV" and "gh" in str(ei.value)


def make_version(tmp_path, tag, kit="full"):
    d = tmp_path / f"{kit}-{tag}"
    d.mkdir(parents=True, exist_ok=True)
    files = {}
    for key, name in fw_server.release_names(tag, "v2", kit).items():
        if key == "icon":
            continue
        files[key] = d / name
        files[key].write_bytes(b"x")
    return fw_server.ServedVersion(kit=kit, version=tag.lstrip("v"), tag=tag, rev="v2",
                                   channel="beta" if "-" in tag else "stable", tier="official", files=files)


def test_catalog_is_newest_first_and_capped(tmp_path):
    tags = ["v1.3.0-rc9", "v1.3.0", "v1.3.0-rc10", "v1.2.9", "v1.10.0", "v1.3.0-beta"]
    s = fw_server.FwServer([make_version(tmp_path, t) for t in tags], host="127.0.0.1")
    got = [v["version"] for v in s.catalog("v2")["kits"][0]["versions"]]
    assert got == ["1.10.0", "1.3.0", "1.3.0-rc10", "1.3.0-rc9", "1.3.0-beta", "1.2.9"]
    s.stop()
    many = [make_version(tmp_path, f"v1.0.{i}", f"k{k}") for k in range(18) for i in range(10)]
    cat = fw_server.FwServer(many, host="127.0.0.1").catalog("v2")
    assert len(cat["kits"]) == 16 and all(len(k["versions"]) == 8 for k in cat["kits"])
    assert cat["kits"][0]["versions"][0]["version"] == "1.0.9"


def test_catalog_delay_holds_the_answer_and_reset_clears_it(srv):
    import time
    srv.faults.catalog_delay_s = 0.3
    # perf_counter, with a little slack: time.monotonic() ticks every ~15.6 ms on
    # Windows and read this 0.3 s hold as 0.297.
    t0 = time.perf_counter()
    res, _ = get(srv, "/api/device/catalog?rev=v2")
    assert res.status == 200 and time.perf_counter() - t0 >= 0.28
    srv.faults.reset()
    assert srv.faults.catalog_delay_s == 0.0


def _with_meta(tmp_path):
    v = served(tmp_path)
    icon = tmp_path / "icon.bin"
    icon.write_bytes(b"I" * 6924)
    v.files["icon"] = icon
    v.meta = {"display_name": "CrossPad", "description": "All apps.", "author": "", "apps": ["sampler"]}
    v.notes = "Faster kit loads"
    return v


def test_catalog_carries_kit_names_and_icons(tmp_path):
    v = _with_meta(tmp_path)
    with fw_server.FwServer([v]) as srv:
        kit = srv.catalog(v.rev)["kits"][0]
    assert (kit["display_name"], kit["author"]) == ("CrossPad", "")
    assert kit["versions"][0]["icon"] == {"name": "icon.bin", "size": 6924}
    assert "notes" not in kit["versions"][0]


def test_catalog_without_icon_falls_back_like_the_worker(tmp_path):
    v = served(tmp_path)
    with fw_server.FwServer([v]) as srv:
        kit = srv.catalog(v.rev)["kits"][0]
    assert kit["display_name"] == v.kit and "icon" not in kit["versions"][0]


def test_kit_route_serves_the_kit_text(tmp_path):
    v = _with_meta(tmp_path)
    with fw_server.FwServer([v], host="127.0.0.1") as srv:
        with urllib.request.urlopen(f"http://127.0.0.1:{srv.port}/api/device/kit/{v.kit}?rev={v.rev}") as r:
            doc = json.loads(r.read())
        other = srv.kit_details("nope", v.rev)
    assert doc == {"v": 1, "kit": v.kit, "versions": [
        {"version": v.version, "tier": v.tier, "display_name": "CrossPad", "description": "All apps.",
         "author": "", "apps": ["sampler"], "notes": "Faster kit loads"}]}
    assert other["versions"] == []


@pytest.mark.parametrize("path", ["/api/device/kit/full", "/api/device/kit/full?rev=2",
                                  "/api/device/kit/Full?rev=v2", "/api/device/kit/v2?rev=v2",
                                  "/api/device/kit/a%2e%2e?rev=v2", "/api/device/kit/?rev=v2"])
def test_kit_route_refuses_bad_ids_with_400(tmp_path, path):
    with fw_server.FwServer([served(tmp_path)], host="127.0.0.1") as srv:
        res, body = get(srv, path)
    assert res.status == 400 and json.loads(body)["success"] is False


def test_kit_route_unknown_kit_is_an_empty_200(tmp_path):
    with fw_server.FwServer([served(tmp_path)], host="127.0.0.1") as srv:
        res, body = get(srv, "/api/device/kit/nope?rev=v2")
    assert res.status == 200 and json.loads(body) == {"v": 1, "kit": "nope", "versions": []}


def test_icon_over_the_device_limit_is_left_out(tmp_path):
    v = _with_meta(tmp_path)
    v.files["icon"].write_bytes(b"I" * (fw_server.ICON_BYTES_MAX + 1))
    assert "icon" not in fw_server.FwServer([v]).catalog(v.rev)["kits"][0]["versions"][0]


def test_release_notes_match_the_worker():
    body = ("## What's Changed\n* fix: x by @a in https://github.com/x/pull/1\n\n"
            "**Full Changelog**: https://github.com/x/compare")
    assert fw_server.release_notes(body) == "What's Changed\n- fix: x by @a"
    assert fw_server.release_notes("<!-- t -->\n### **Bold** [link](https://a.b) `code`\n---") == "Bold link code"


def test_release_notes_keep_six_lines_of_at_most_64_characters():
    lines = fw_server.release_notes("\n".join(f"- change {i} {'x' * 80}" for i in range(9))).split("\n")
    assert len(lines) == 6 and len(lines[0]) == 64 and lines[0].endswith("...")


def test_release_notes_strip_only_the_in_a_removed_url_left():
    assert fw_server.release_notes("- fix pad in") == "- fix pad in"
    assert fw_server.release_notes("- fix by @a in https://x.y/pull/1") == "- fix by @a"


def test_release_notes_are_bounded_to_the_device_buffer():
    assert len(fw_server.release_notes("\n".join(["\u017b" * 64] * 6)).encode()) <= fw_server.NOTES_BYTES


def _kit_text(tmp_path, **meta):
    v = served(tmp_path)
    v.meta = meta
    srv = fw_server.FwServer([v], host="127.0.0.1")
    srv.stop()
    return srv.kit_details(v.kit, v.rev)["versions"][0]


def test_text_drops_control_characters_and_lone_surrogates(tmp_path):
    assert _kit_text(tmp_path, display_name="A\x07\x7fB\ud800Z")["display_name"] == "ABZ"


def test_text_keeps_a_newline_only_in_the_description(tmp_path):
    t = _kit_text(tmp_path, display_name="a\nb", description="a\n\tb", author="c\nd")
    assert [t["display_name"], t["description"], t["author"]] == ["ab", "a\nb", "cd"]


def test_text_is_cut_by_utf8_bytes_at_a_character_boundary(tmp_path):
    t = _kit_text(tmp_path, display_name="\u017b" * 24, author="\u017b" * 31, description="\u017b" * 160)
    assert t["display_name"] == "\u017b" * 15 and t["author"] == "\u017b" * 15
    assert len(t["description"].encode()) <= 191 and set(t["description"]) == {"\u017b"}


def test_text_apps_are_filtered_and_capped(tmp_path):
    t = _kit_text(tmp_path, apps=["sampler", "pad-mixer", "../evil", 7] + [f"a{i}" for i in range(20)])
    assert t["apps"][:2] == ["sampler", "pad-mixer"] and len(t["apps"]) == 16


def test_kit_name_and_author_come_from_the_newest_release(tmp_path):
    old, new = make_version(tmp_path, "v1.2.0"), make_version(tmp_path, "v1.3.0")
    old.meta, new.meta = {"display_name": "Old"}, {"display_name": "New", "author": "Ola"}
    kit = fw_server.FwServer([old, new]).catalog("v2")["kits"][0]
    assert (kit["display_name"], kit["author"]) == ("New", "Ola")


def test_details_only_for_what_the_catalog_serves(tmp_path):
    srv = fw_server.FwServer([make_version(tmp_path, f"v1.0.{i}", f"k{k:02d}") for k in range(17)
                              for i in range(10)])
    assert [v["version"] for v in srv.kit_details("k15", "v2")["versions"]] == [f"1.0.{i}" for i in range(9, 1, -1)]
    assert srv.kit_details("k16", "v2")["versions"] == []


def test_catalog_stays_inside_the_device_buffer_at_the_worst_case(tmp_path):
    vs = []
    for k in range(fw_server.CATALOG_MAX_KITS):
        for i in range(fw_server.CATALOG_MAX_VERSIONS):
            v = make_version(tmp_path, f"v10.12.3-beta.{10 + i}", f"kit_{k:02d}_abcdefgh")
            v.meta = {"display_name": "\u017b" * 15, "author": "\u017b" * 15}
            icon = tmp_path / f"{k}-{i}-icon.bin"
            icon.write_bytes(b"I" * 100)
            v.files["icon"] = icon
            vs.append(v)
    srv = fw_server.FwServer(vs)
    cat = srv.catalog("v2")
    assert cat["kits"] and all(len(k["versions"]) <= fw_server.CATALOG_MAX_VERSIONS for k in cat["kits"])
    size = len(json.dumps(cat, separators=(",", ":"), ensure_ascii=False).encode())
    assert size <= fw_server.CATALOG_MAX_BYTES


def test_trimming_takes_the_oldest_version_of_the_largest_kit_first(tmp_path, monkeypatch):
    vs = [make_version(tmp_path, f"v1.0.{i}", "big") for i in range(3)] + \
         [make_version(tmp_path, f"v1.0.{i}", "small") for i in range(2)]
    srv = fw_server.FwServer(vs)
    full = len(json.dumps(srv.catalog("v2"), separators=(",", ":")).encode())
    monkeypatch.setattr(fw_server, "CATALOG_MAX_BYTES", full - 1)
    kits = {k["kit"]: [v["version"] for v in k["versions"]] for k in srv.catalog("v2")["kits"]}
    assert kits == {"big": ["1.0.2", "1.0.1"], "small": ["1.0.1", "1.0.0"]}
    assert [v["version"] for v in srv.kit_details("big", "v2")["versions"]] == ["1.0.2", "1.0.1"]
    monkeypatch.setattr(fw_server, "CATALOG_MAX_BYTES", 10)
    assert srv.catalog("v2")["kits"] == []


def test_text_drops_astral_characters():
    assert fw_server.cap_text("a\U0001f600b", 24, 31) == "ab"


def test_apps_with_a_trailing_newline_are_refused(tmp_path):
    assert _kit_text(tmp_path, apps=["sampler\n", "ok"])["apps"] == ["ok"]


@pytest.mark.parametrize("path", ["/api/device/catalog", "/api/device/catalog?rev=2", "/api/device/catalog?rev=v"])
def test_catalog_refuses_a_bad_rev_with_400(tmp_path, path):
    with fw_server.FwServer([served(tmp_path)], host="127.0.0.1") as srv:
        res, body = get(srv, path)
    assert res.status == 400 and json.loads(body)["success"] is False


def test_served_bytes_are_the_compact_utf8_the_ceiling_measures(tmp_path):
    v = served(tmp_path)
    v.meta = {"display_name": "\u017b\u00f3\u0142\u0107"}
    with fw_server.FwServer([v], host="127.0.0.1") as srv:
        _, body = get(srv, "/api/device/catalog?rev=v2")
        assert body == fw_server.compact_json(srv.catalog("v2")) and "Żółć".encode() in body
        assert b", " not in body and b": " not in body
