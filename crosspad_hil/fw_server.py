"""A bench stand-in for crosspad.app's /api/device/* (platform-idf
docs/subsystems/firmware-updates.md), serving real CI-signed packages from a
GitHub release, with faults on demand: cut a file once, flip a byte in it,
ignore Range, or serve the real files under a made-up catalog entry (a
broken signature, a manifest that names another kit).

    python -m crosspad_hil.fw_server [--tag v1.1.0-fwstore.1] [--rev v2] [--kit full] [--port 8765]
                                     [--fault drop|flip|ignore_range ...]

Every request is printed as one JSON line (path, range, status, bytes sent, cut).

The board reaches it over the bench LAN: `FW_UPD_URL http://<this machine>:<port>`,
and back with `FW_UPD_URL DEFAULT`. Drafts work too (gh downloads them), so a
release does not have to be published for the bench.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import socket
import subprocess
import threading
import time
from dataclasses import dataclass, field
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlparse

from crosspad_hil.errors import BAD_ARGS, ENV, HilError

FIRMWARE_REPO = "CrossPad/platform-idf"
CHUNK = 4096
RELEASE_SCAN = 20
_RANGE_RE = re.compile(r"^bytes=([0-9]*)-([0-9]*)$")
# crosspad-core Semver::parse's grammar, which fwVersionValid() applies to a
# manifest's version (the tag with its 'v'); platform-idf make_fw_package.tag_valid().
_TAG_HEAD_RE = re.compile(r"v([0-9]+)\.([0-9]+)\.([0-9]+)(?:-([^+]*))?(?:\+(.*))?", re.S)
# non-empty dot-separated identifiers
_IDENTS_RE = re.compile(r"[0-9A-Za-z-]+(?:\.[0-9A-Za-z-]+)*")
# a git-describe tail, stripped
_DESCRIBE_RE = re.compile(r"(?:(.*)-)?([0-9]{1,5})-g[0-9a-f]{4,}", re.S)
_DIRTY = "-dirty"
_SEMVER_PART_MAX = 65535
_KIT_RE = re.compile(r"^[a-z][a-z0-9_]{0,14}$")
_REV_RE = re.compile(r"^v\d+$")
# A board holds a version of kFwVersionMax - 1 characters (crosspad-core
# kFwVersionMax = 24 counts the NUL), and the signed manifest's version is the
# tag with its 'v'; CI refuses a longer tag and the Worker leaves it out.
FW_VERSION_MAX = 24
TAG_MAX = FW_VERSION_MAX - 1
CATALOG_MAX_KITS = 16
CATALOG_MAX_VERSIONS = 8
FILE_ROUTE = "/api/device/file/"
KIT_ROUTE = "/api/device/kit/"
# Text bounds of the device's buffers (core crosspad/fw/FwStore.hpp), as the Worker
# enforces them in server/lib/firmware/device.ts: characters, then UTF-8 bytes.
DISPLAY_NAME_MAX, DISPLAY_NAME_BYTES = 24, 31
DESCRIPTION_MAX, DESCRIPTION_BYTES = 160, 191
AUTHOR_MAX, AUTHOR_BYTES = 31, 31
NOTES_BYTES = 399
NOTES_MAX_LINES = 6
NOTES_MAX_CHARS = 64
APPS_MAX = 16
CATALOG_MAX_BYTES = 60 * 1024           # the device parses the catalog from one 64 kB buffer
ICON_BYTES_MAX = 16 * 1024
META_KEYS = ("display_name", "description", "author", "apps")
_APP_RE = re.compile(r"^[a-z0-9][a-z0-9_-]{0,22}$")
_CONTROLS = re.compile(r"[\x00-\x1f\x7f]")
_CONTROLS_KEEPING_NEWLINE = re.compile(r"[\x00-\x09\x0b-\x1f\x7f]")
# Lone surrogates, and everything above the BMP: the device's fonts hold no emoji.
_UNDRAWABLE = re.compile(r"[\ud800-\udfff\U00010000-\U0010ffff]")


def cache_root() -> Path:
    base = os.environ.get("XDG_CACHE_HOME") or os.path.join(os.path.expanduser("~"), ".cache")
    return Path(base) / "crosspad-hil" / "fw_releases"


def tag_channel(tag: str) -> str:
    """The channel CI signs for a tag (platform-idf make_fw_package.tag_channel) and the
    Worker lists: a '-' is beta. Never GitHub's pre-release tick."""
    return "beta" if "-" in tag else "stable"


def tag_ok(tag: str) -> bool:
    """CI's tag rule, at most TAG_MAX characters: what the Worker lists and a board's
    fwVersionValid() accepts -- numbers at most 65535, pre-release and build of
    non-empty dot-separated [0-9A-Za-z-] identifiers."""
    m = _TAG_HEAD_RE.fullmatch(tag)
    if len(tag) > TAG_MAX or not m or any(int(x) > _SEMVER_PART_MAX for x in m.group(1, 2, 3)):
        return False
    pre, build = m.group(4), m.group(5)
    if build is not None and not _IDENTS_RE.fullmatch(build):
        return False
    if pre is None or pre == _DIRTY[1:]:
        return True
    if len(pre) > len(_DIRTY) and pre.endswith(_DIRTY):
        pre = pre[:-len(_DIRTY)]
    d = _DESCRIBE_RE.fullmatch(pre)
    if d and int(d.group(2)) <= _SEMVER_PART_MAX:
        return not d.group(1) or bool(_IDENTS_RE.fullmatch(d.group(1)))
    return bool(_IDENTS_RE.fullmatch(pre))


def check_ids(tag: str, rev: str, kit: str) -> None:
    """The same shapes the Worker and the board accept; tag and kit become path parts and gh
    globs."""
    if not tag_ok(tag):
        raise HilError(BAD_ARGS, f"tag {tag!r} is not v<major>.<minor>.<patch>[-pre][+build] "
                                 f"of non-empty identifiers, at most {TAG_MAX} characters")
    if not _KIT_RE.fullmatch(kit) or _REV_RE.fullmatch(kit):
        raise HilError(BAD_ARGS, f"kit {kit!r} is not a kit id (^[a-z][a-z0-9_]{{0,14}}$, "
                                 "never v<digits>)")
    if not _REV_RE.fullmatch(rev):
        raise HilError(BAD_ARGS, f"rev {rev!r} is not v<digits>")


def version_key(version: str) -> tuple:
    """Semver precedence with one deliberate departure: a trailing number in a pre-release
    identifier compares by value (rc9 < rc10), as the board's own ordering does. A release
    sorts above its pre-releases."""
    core, _, pre = version.partition("-")
    nums = tuple(int(x) if x.isdigit() else 0 for x in core.split("."))
    if not pre:
        return nums, 1, ()
    ids = []
    for ident in pre.split("."):
        m = re.fullmatch(r"(\D*)(\d*)", ident)
        ids.append((0, "", int(ident)) if ident.isdigit()
                   else (1, m.group(1), int(m.group(2) or 0)))
    return nums, 0, tuple(ids)


def release_names(tag: str, rev: str, kit: str) -> dict[str, str]:
    base = f"CrossPad-{tag}-{rev}-{kit}"
    return {"bin": f"{base}.bin", "assets": f"{base}-assets.bin",
            "manifest": f"{base}-manifest.json", "sig": f"{base}-manifest.sig",
            "icon": f"{base}-icon.bin"}


@dataclass
class ServedVersion:
    """One catalog entry and the files behind it (keys: bin, assets, manifest, sig; icon when
    the release has one)."""
    kit: str
    version: str
    tag: str
    rev: str
    channel: str
    tier: str
    files: dict[str, Path]
    # display_name, description, author, apps (kits.json)
    meta: dict[str, Any] = field(default_factory=dict)
    notes: str = ""                                       # release_notes() of the release body


def compact_json(doc: dict[str, Any]) -> bytes:
    """The bytes the Worker sends (JSON.stringify, UTF-8): the 60 kB ceiling is measured on
    these."""
    return json.dumps(doc, separators=(",", ":"), ensure_ascii=False).encode("utf-8")


def cap_text(value: Any, max_chars: float, max_bytes: int, keep_newline: bool = False) -> str:
    """The Worker's capText: whole code points only, so a cut never splits a multi-byte
    sequence; control characters, lone surrogates and astral characters are dropped before
    counting, and a newline survives only where the device keeps one (description, notes)."""
    if not isinstance(value, str):
        return ""
    clean = (_CONTROLS_KEEPING_NEWLINE if keep_newline else _CONTROLS).sub("", value)
    out: list[str] = []
    nbytes = 0
    for ch in clean:
        if _UNDRAWABLE.match(ch):
            continue
        nbytes += len(ch.encode("utf-8"))
        if len(out) + 1 > max_chars or nbytes > max_bytes:
            break
        out.append(ch)
    return "".join(out)


def release_notes(body: str) -> str:
    """The Worker's releaseNotes(): the first lines of a release body as plain text --
    headings, bullets, links and emphasis reduced to words, GitHub's "Full Changelog" dropped."""
    lines: list[str] = []
    for raw in re.split(r"\r?\n", body or ""):
        line = _CONTROLS.sub("", raw).strip()
        if not line or line.startswith("<!--") or re.fullmatch(r"-{3,}", line):
            continue
        had_url = re.search(r"https?://", line) is not None
        line = re.sub(r"^#{1,6}\s+", "", line)
        line = re.sub(r"^[*+-]\s+", "- ", line)
        line = re.sub(r"\[([^\]]*)\]\([^)]*\)", r"\1", line)
        line = re.sub(r"[*_`]", "", line)
        line = re.sub(r"https?://\S+", "", line).strip()
        # "by @user in <pull request url>" leaves a dangling "in" once the URL is gone.
        if had_url:
            line = re.sub(r"\s+in\Z", "", line)
        if not line or re.match(r"full changelog", line, re.IGNORECASE):
            continue
        if len(line) > NOTES_MAX_CHARS:
            line = line[:NOTES_MAX_CHARS - 3] + "..."
        lines.append(line)
        if len(lines) >= NOTES_MAX_LINES:
            break
    return cap_text("\n".join(lines), float("inf"), NOTES_BYTES, True)


def _name_of(v: ServedVersion) -> str:
    return cap_text(v.meta.get("display_name"), DISPLAY_NAME_MAX, DISPLAY_NAME_BYTES) or v.kit


def _author_of(v: ServedVersion) -> str:
    return cap_text(v.meta.get("author"), AUTHOR_MAX, AUTHOR_BYTES)


def _apps_of(v: ServedVersion) -> list[str]:
    apps = v.meta.get("apps")
    if not isinstance(apps, list):
        return []
    return [a for a in apps if isinstance(a, str) and _APP_RE.fullmatch(a)][:APPS_MAX]


def fetch_release(tag: str, rev: str, kit: str, *, cache: Path | None = None, gh: str = "gh",
                  repo: str = FIRMWARE_REPO) -> ServedVersion:
    """The package for `kit` on board `rev` from release `tag`, downloaded once with gh."""
    check_ids(tag, rev, kit)
    dest = (cache or cache_root()) / tag
    dest.mkdir(parents=True, exist_ok=True)
    names = release_names(tag, rev, kit)
    kits_name = f"CrossPad-{tag}-kits.json"
    try:
        view = subprocess.run([gh, "release", "view", tag, "-R", repo, "--json", "body,assets"],
                              capture_output=True, text=True)
    except FileNotFoundError as e:
        raise HilError(ENV, f"{gh} is not installed", hint="install the GitHub CLI (gh)") from e
    try:
        if view.returncode != 0:
            raise ValueError((view.stderr or view.stdout).strip())
        info = json.loads(view.stdout)
        body = info.get("body") or ""
        listed = {a.get("name") for a in info.get("assets", [])}
    except (ValueError, AttributeError, TypeError) as e:
        raise HilError(ENV, f"gh release view {tag}: {e}",
                       hint="gh auth status; is the release there?") from e
    required = [n for k, n in names.items() if k != "icon"] + [kits_name]
    # Releases from before kit icons list none, and gh fails on a pattern nothing matches.
    wanted = required + ([names["icon"]] if names["icon"] in listed else [])
    missing = [n for n in wanted if not (dest / n).is_file()]
    if missing:
        cmd = [gh, "release", "download", tag, "-R", repo, "-D", str(dest), "--skip-existing"]
        for n in missing:
            cmd += ["-p", n]
        r = subprocess.run(cmd, capture_output=True, text=True)
        if r.returncode != 0:
            raise HilError(ENV, f"gh release download {tag}: {(r.stderr or r.stdout).strip()}",
                           hint="gh auth status; the release must carry kit packages")
    for n in required:
        if not (dest / n).is_file():
            raise HilError(ENV, f"release {tag} has no {n}")
    try:
        kits = json.loads((dest / kits_name).read_text())
        entry = next((k for k in kits["kits"] if k.get("kit") == kit), None)
    except (ValueError, KeyError, TypeError, AttributeError) as e:
        raise HilError(ENV, f"{kits_name} is not a kits file: {e}") from e
    if entry is None:
        raise HilError(ENV, f"release {tag} does not list kit {kit!r}")
    tier = "community" if entry.get("tier") == "community" else "official"
    return ServedVersion(kit=kit, version=tag.lstrip("v"), tag=tag, rev=rev,
                         channel=tag_channel(tag), tier=tier,
                         files={k: dest / n for k, n in names.items() if (dest / n).is_file()},
                         meta={k: entry[k] for k in META_KEYS if k in entry},
                         notes=release_notes(body))


def newest_kit_release(*, icons: bool = False, gh: str = "gh", repo: str = FIRMWARE_REPO) -> str:
    """Tag of the newest release (drafts included) that carries CrossPad-<tag>-kits.json
    (and with icons=True at least one kit icon, *-icon.bin); a tag the Worker
    would not list (tag_ok) is skipped."""
    r = subprocess.run([gh, "release", "list", "-R", repo, "--json", "tagName", "--limit",
                        str(RELEASE_SCAN)],
                       capture_output=True, text=True)
    if r.returncode != 0:
        raise HilError(ENV, f"gh release list: {(r.stderr or r.stdout).strip()}")
    for rel in json.loads(r.stdout or "[]"):
        tag = rel["tagName"]
        if not tag_ok(tag):
            continue
        v = subprocess.run([gh, "release", "view", tag, "-R", repo, "--json", "assets"],
                           capture_output=True, text=True)
        if v.returncode != 0:
            continue
        names = [a.get("name", "") for a in json.loads(v.stdout).get("assets", [])]
        has_icon = any(n.endswith("-icon.bin") for n in names)
        if f"CrossPad-{tag}-kits.json" in names and (not icons or has_icon):
            return tag
    raise HilError(ENV, "no release with kit packages and icons yet" if icons
                   else "no release with kit packages yet",
                   hint="a maintainer tags platform-idf after the kit CI landed (plan Task 4)")


def alias(real: ServedVersion, kit: str, version: str, workdir: Path, *,
          break_sig: bool = False) -> ServedVersion:
    """The real package's bytes under another catalog entry: its manifest still
    names the real kit (a mismatch on the board), or -- break_sig -- its
    signature has one byte flipped (a refusal before anything else is read)."""
    check_ids(real.tag, real.rev, kit)
    workdir.mkdir(parents=True, exist_ok=True)
    names = release_names(real.tag, real.rev, kit)
    files = {}
    for key, src in real.files.items():
        dst = workdir / names[key]
        shutil.copyfile(src, dst)
        files[key] = dst
    if break_sig:
        sig = bytearray(files["sig"].read_bytes())
        sig[len(sig) // 2] ^= 0xFF
        files["sig"].write_bytes(bytes(sig))
    return ServedVersion(kit=kit, version=version, tag=real.tag, rev=real.rev, channel=real.channel,
                         tier=real.tier, files=files, meta=dict(real.meta), notes=real.notes)


def parse_range(header: str | None, size: int) -> str | tuple[int, int]:
    """The Worker's parseRange (server/lib/firmware/device.ts): "full", "unsatisfiable"
    or the inclusive (start, end) of a partial answer; one range only, a malformed
    or reversed one is served whole."""
    if not header:
        return "full"
    m = _RANGE_RE.match(header.strip())
    if not m or (m.group(1) == "" and m.group(2) == ""):
        return "full"
    if size == 0:
        return "unsatisfiable"
    if m.group(1) == "":
        suffix = int(m.group(2))
        if suffix == 0:
            return "unsatisfiable"
        return max(0, size - suffix), size - 1
    start = int(m.group(1))
    if start >= size:
        return "unsatisfiable"
    end = size - 1 if m.group(2) == "" else min(int(m.group(2)), size - 1)
    if end < start:
        return "full"
    return start, end


def lan_ip() -> str:
    """This machine's address on the route out -- what a board on the same LAN dials."""
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        s.connect(("192.0.2.1", 9))     # TEST-NET-1; a UDP connect sends nothing
        return s.getsockname()[0]
    finally:
        s.close()


def local_ip_for(peer: str) -> str:
    """This machine's address on the route to `peer` -- what that board dials back."""
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        s.connect((peer, 9))
        return s.getsockname()[0]
    finally:
        s.close()


@dataclass
class Faults:
    # file -> cut its first response after N bytes
    drop_once_at: dict[str, int] = field(default_factory=dict)
    # file -> invert this byte in every response
    flip_at: dict[str, int] = field(default_factory=dict)
    # file -> answer a Range request with 200 + all
    ignore_range: set[str] = field(default_factory=set)
    dropped: set[str] = field(default_factory=set)
    # hold every catalog answer this long
    catalog_delay_s: float = 0.0

    def reset(self) -> None:
        self.drop_once_at.clear()
        self.flip_at.clear()
        self.ignore_range.clear()
        self.dropped.clear()
        self.catalog_delay_s = 0.0


class FwServer:
    def __init__(self, versions: list[ServedVersion], *, host: str = "0.0.0.0", port: int = 0,
                 verbose: bool = False) -> None:
        self.versions = list(versions)
        self.verbose = verbose
        self.faults = Faults()
        self.log: list[dict[str, Any]] = []
        self._lock = threading.Lock()
        server = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, fmt: str, *args: Any) -> None:
                return None

            def do_GET(self) -> None:
                server._handle(self)

        self._httpd = ThreadingHTTPServer((host, port), Handler)
        self._thread: threading.Thread | None = None

    @property
    def port(self) -> int:
        return self._httpd.server_address[1]

    def url(self, host_ip: str | None = None) -> str:
        return f"http://{host_ip or lan_ip()}:{self.port}"

    def start(self) -> None:
        self._thread = threading.Thread(target=self._httpd.serve_forever, name="fw_server",
                                        daemon=True)
        self._thread.start()

    def stop(self) -> None:
        # shutdown() waits for serve_forever, so it is only safe once start() ran.
        thread, self._thread = self._thread, None
        if thread is not None:
            self._httpd.shutdown()
            thread.join()
        self._httpd.server_close()

    def __enter__(self) -> FwServer:
        self.start()
        return self

    def __exit__(self, *exc: Any) -> None:
        self.stop()

    def _plan(self, rev: str) -> list[tuple[dict[str, Any], list[ServedVersion]]]:
        """The catalog's kits for `rev`, each with the versions behind its entries in the same
        order: the catalog and the kit details walk one list, so the board sees the same
        versions in both."""
        by_kit: dict[str, list[ServedVersion]] = {}
        for v in self.versions:
            if v.rev == rev:
                by_kit.setdefault(v.kit, []).append(v)
        plan: list[tuple[dict[str, Any], list[ServedVersion]]] = []
        # The Worker caps the list, newest first; the sort stands in for its release index order.
        for kit, vs in list(by_kit.items())[:CATALOG_MAX_KITS]:
            vs = sorted(vs, key=lambda v: version_key(v.version),
                        reverse=True)[:CATALOG_MAX_VERSIONS]
            entries = [self._entry(v) for v in vs]
            plan.append(({"kit": kit, "display_name": _name_of(vs[0]), "author": _author_of(vs[0]),
                          "versions": entries}, vs))
        self._fit_device(plan)
        return plan

    @staticmethod
    def _entry(v: ServedVersion) -> dict[str, Any]:
        entry: dict[str, Any] = {
            "version": v.version, "tag": v.tag, "channel": v.channel, "tier": v.tier,
            "bin": {"name": v.files["bin"].name, "size": v.files["bin"].stat().st_size},
            "assets": {"name": v.files["assets"].name, "size": v.files["assets"].stat().st_size},
            "manifest": v.files["manifest"].name, "sig": v.files["sig"].name}
        # An icon over the device's limit would be refused there; leaving it out costs the
        # tile its picture.
        icon = v.files.get("icon")
        if icon is not None and 0 < icon.stat().st_size <= ICON_BYTES_MAX:
            entry["icon"] = {"name": icon.name, "size": icon.stat().st_size}
        return entry

    @staticmethod
    def _fit_device(plan: list[tuple[dict[str, Any], list[ServedVersion]]]) -> None:
        """The Worker's fitDevice: oldest versions go first (from the kit with the most), then
        the last kit, so what is served always parses in the device's buffer."""
        def size() -> int:
            doc = {"v": 1, "generated": "bench", "kits": [k for k, _ in plan]}
            return len(compact_json(doc))

        while size() > CATALOG_MAX_BYTES and plan:
            longest = plan[0]
            for cand in plan[1:]:
                if len(cand[0]["versions"]) >= len(longest[0]["versions"]):
                    longest = cand
            if len(longest[0]["versions"]) > 1:
                longest[0]["versions"].pop()
                longest[1].pop()
            else:
                plan.pop()

    def catalog(self, rev: str) -> dict[str, Any]:
        return {"v": 1, "generated": "bench", "kits": [k for k, _ in self._plan(rev)]}

    def kit_details(self, kit: str, rev: str) -> dict[str, Any]:
        """A kit card's text per version -- only for kits and versions the catalog serves."""
        served = next((vs for k, vs in self._plan(rev) if k["kit"] == kit), [])
        return {"v": 1, "kit": kit, "versions": [
            {"version": v.version, "tier": v.tier, "display_name": _name_of(v),
             "description": cap_text(v.meta.get("description"), DESCRIPTION_MAX,
                                     DESCRIPTION_BYTES, True),
             "author": _author_of(v), "apps": _apps_of(v), "notes": v.notes}
            for v in served]}

    def requests_for(self, name: str) -> list[dict[str, Any]]:
        with self._lock:
            return [e for e in self.log if e["path"].endswith("/" + name)]

    def _find(self, tag: str, name: str) -> Path | None:
        for v in self.versions:
            if v.tag == tag:
                for p in v.files.values():
                    if p.name == name:
                        return p
        return None

    def _handle(self, h: BaseHTTPRequestHandler) -> None:
        u = urlparse(h.path)
        entry: dict[str, Any] = {"path": u.path, "range": h.headers.get("Range"), "status": 0,
                                 "sent": 0, "cut": False}
        # Logged before the response: a client that saw the connection cut must
        # find this entry already there, filled in up to the cut.
        with self._lock:
            self.log.append(entry)
        try:
            if u.path == "/api/device/ping":
                self._json(h, {"ok": True}, entry)
            elif u.path == "/api/device/catalog":
                if self.faults.catalog_delay_s:
                    time.sleep(self.faults.catalog_delay_s)
                rev = parse_qs(u.query).get("rev", [""])[0]
                if not _REV_RE.fullmatch(rev):
                    self._json(h, {"success": False, "error": "rev=v<N> is required."}, entry, 400)
                else:
                    self._json(h, self.catalog(rev), entry)
            elif u.path.startswith(KIT_ROUTE):
                kit = u.path[len(KIT_ROUTE):]
                rev = parse_qs(u.query).get("rev", [""])[0]
                if (not _REV_RE.fullmatch(rev) or not _KIT_RE.fullmatch(kit)
                        or _REV_RE.fullmatch(kit)):
                    self._json(h, {"success": False,
                                   "error": "rev=v<N> and a kit id are required."}, entry, 400)
                else:
                    self._json(h, self.kit_details(kit, rev), entry)
            elif u.path.startswith(FILE_ROUTE):
                parts = u.path[len(FILE_ROUTE):].split("/")
                path = self._find(parts[0], parts[1]) if len(parts) == 2 else None
                if path is None:
                    self._empty(h, 404, entry)
                else:
                    self._file(h, path, entry)
            else:
                self._empty(h, 404, entry)
        except (BrokenPipeError, ConnectionResetError):
            entry["aborted"] = True
        if self.verbose:
            print(json.dumps(entry), flush=True)

    @staticmethod
    def _json(h: BaseHTTPRequestHandler, doc: dict[str, Any], entry: dict[str, Any],
              status: int = 200) -> None:
        body = compact_json(doc)
        h.send_response(status)
        h.send_header("Content-Type", "application/json")
        h.send_header("Content-Length", str(len(body)))
        h.end_headers()
        h.wfile.write(body)
        entry.update(status=status, sent=len(body))

    @staticmethod
    def _empty(h: BaseHTTPRequestHandler, status: int, entry: dict[str, Any],
               **headers: str) -> None:
        h.send_response(status)
        for k, v in headers.items():
            h.send_header(k.replace("_", "-"), v)
        h.send_header("Content-Length", "0")
        h.end_headers()
        entry["status"] = status

    def _file(self, h: BaseHTTPRequestHandler, path: Path, entry: dict[str, Any]) -> None:
        data = bytearray(path.read_bytes())
        name = path.name
        # of the real bytes: a flip stays invisible to it
        etag = f'"{hashlib.sha256(data).hexdigest()}"'
        flip = self.faults.flip_at.get(name)
        if flip is not None and 0 <= flip < len(data):
            data[flip] ^= 0xFF
        size = len(data)
        start, end, status = 0, size - 1, 200
        if name not in self.faults.ignore_range:
            rng = parse_range(entry["range"], size)
            if rng == "unsatisfiable":
                self._empty(h, 416, entry, Content_Range=f"bytes */{size}", Accept_Ranges="bytes",
                            ETag=etag)
                return
            if rng != "full":
                start, end = rng
                status = 206
        body = bytes(data[start:end + 1])
        cut = None
        with self._lock:
            first_drop = name in self.faults.drop_once_at and name not in self.faults.dropped
            if first_drop:
                self.faults.dropped.add(name)
        if first_drop:
            cut = max(0, self.faults.drop_once_at[name] - start)
        h.send_response(status)
        h.send_header("Content-Type", "application/octet-stream")
        h.send_header("Content-Length", str(len(body)))
        h.send_header("Accept-Ranges", "bytes")
        h.send_header("ETag", etag)
        if status == 206:
            h.send_header("Content-Range", f"bytes {start}-{end}/{size}")
        h.end_headers()
        entry["status"] = status
        payload = body if cut is None else body[:cut]
        for off in range(0, len(payload), CHUNK):
            h.wfile.write(payload[off:off + CHUNK])
        entry["sent"] = len(payload)
        if cut is not None:
            entry["cut"] = True
            h.wfile.flush()
            h.close_connection = True
            h.connection.shutdown(socket.SHUT_RDWR)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tag", default=None,
                    help="release to serve (default: the newest with kit packages)")
    ap.add_argument("--rev", default="v2")
    ap.add_argument("--kit", default="full")
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--host-ip", default=None,
                    help="the address the board should dial (default: auto)")
    ap.add_argument("--fault", action="append", default=[],
                    choices=("drop", "flip", "ignore_range"),
                    help="on CrossPad.bin: drop = cut the first response half way, flip = invert "
                         "one byte every time, ignore_range = answer a resume with the whole file")
    a = ap.parse_args(argv)
    tag = a.tag or newest_kit_release()
    real = fetch_release(tag, a.rev, a.kit)
    srv = FwServer([real], port=a.port, verbose=True)
    bin_name, bin_size = real.files["bin"].name, real.files["bin"].stat().st_size
    if "drop" in a.fault:
        srv.faults.drop_once_at[bin_name] = bin_size // 2
    if "flip" in a.fault:
        srv.faults.flip_at[bin_name] = bin_size // 3
    if "ignore_range" in a.fault:
        srv.faults.ignore_range.add(bin_name)
    with srv:
        print(f"serving {tag} {a.kit} for {a.rev}: FW_UPD_URL {srv.url(a.host_ip)}", flush=True)
        try:
            threading.Event().wait()
        except KeyboardInterrupt:
            pass
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
