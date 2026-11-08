"""Scenario fw_update's bench checks and helpers, without a board."""
from __future__ import annotations

import threading
from dataclasses import dataclass

import pytest

from crosspad_hil.errors import NOT_SUPPORTED, HilError
from crosspad_hil.scenarios import base, fw_update, get
from tests.test_scenario_fw_switch import make_device


@dataclass
class _Ctx(base.Context):
    def open_cdc(self):  # every verb that would use it is stubbed
        return object()


def test_no_saved_network_is_a_bench_problem(tmp_path, monkeypatch):
    monkeypatch.setattr(fw_update.verbs, "net_mode_status", lambda link: {"active": False, "tab": None})
    monkeypatch.setattr(fw_update.verbs, "wifi_status", lambda link: {"enabled": 1})
    monkeypatch.setattr(fw_update.verbs, "wifi_list", lambda link: [])
    monkeypatch.delenv("CROSSPAD_WIFI_SSID", raising=False)
    ctx = _Ctx(device=make_device(), workdir=tmp_path, cancelled=threading.Event(), log=lambda s: None)
    report = get("fw_update").run(ctx, fw_update.Params(), base.Progress())
    assert report.exit_code == 2
    assert "WiFi" in report.summary


def _scenario(tmp_path, link=None):
    ctx = _Ctx(device=make_device(), workdir=tmp_path, cancelled=threading.Event(), log=lambda s: None)
    sc = fw_update.FwUpdateScenario()
    sc.ctx, sc.p, sc.data, sc.link = ctx, fw_update.Params(), {"steps": {}}, link or object()
    return sc


@pytest.mark.parametrize("token", ["no_mem", "no_mem:catalog", "no_mem:job_task", "no_mem:install_task"])
def test_every_no_mem_token_is_a_memory_refusal(tmp_path, monkeypatch, token):
    def refuse(link):
        raise HilError(NOT_SUPPORTED, f"FW_CHECK: {token}")
    monkeypatch.setattr(fw_update.verbs, "fw_check", refuse)
    with pytest.raises(fw_update._Stop, match="memory"):
        _scenario(tmp_path)._start_check()


def test_another_refusal_is_not_taken_for_memory(tmp_path, monkeypatch):
    def refuse(link):
        raise HilError(NOT_SUPPORTED, "FW_CHECK: busy")
    monkeypatch.setattr(fw_update.verbs, "fw_check", refuse)
    with pytest.raises(HilError):
        _scenario(tmp_path)._start_check()


def _status(**over):
    return {"state": "failed", "stack_free": 1800, "inst_stack_free": 0, "progress": 0, **over}


def test_tls_step_points_at_crosspad_app_and_records_the_stack(tmp_path, monkeypatch):
    calls = []
    monkeypatch.setattr(fw_update.verbs, "fw_upd_url", lambda link, url: calls.append(("url", url)))
    monkeypatch.setattr(fw_update.verbs, "fw_check", lambda link: calls.append(("check",)))
    monkeypatch.setattr(fw_update.verbs, "fw_upd_status", lambda link: _status())
    sc = _scenario(tmp_path)
    sc._tls_stack()
    assert calls == [("url", None), ("check",)]
    assert sc.data["stack_free"] == 1800


def test_tls_step_fails_when_the_stack_is_used_up(tmp_path, monkeypatch):
    monkeypatch.setattr(fw_update.verbs, "fw_upd_url", lambda link, url: None)
    monkeypatch.setattr(fw_update.verbs, "fw_check", lambda link: None)
    monkeypatch.setattr(fw_update.verbs, "fw_upd_status", lambda link: _status(stack_free=800))
    with pytest.raises(fw_update._Stop, match="stack"):
        _scenario(tmp_path)._tls_stack()


def test_library_bytes_counts_packages_and_loose_images(monkeypatch):
    listing = {
        fw_update.LIBRARY: [{"name": "full-1.3.0", "dir": True, "size": 0},
                            {"name": "stray.bin", "dir": False, "size": 7},
                            {"name": ".catalog.json", "dir": False, "size": 900},
                            {"name": "notes.txt", "dir": False, "size": 100}],
        f"{fw_update.LIBRARY}/full-1.3.0": [{"name": "CrossPad.bin", "dir": False, "size": 10},
                                            {"name": ".pinned", "dir": False, "size": 1}],
    }
    monkeypatch.setattr(fw_update.verbs, "fs_list", lambda link, path: {"path": path, "entries": listing[path]})
    assert fw_update.library_bytes(object()) == 18      # the device counts packages and loose *.bin only


def test_tls_step_without_internet_is_a_bench_problem(tmp_path, monkeypatch):
    monkeypatch.setattr(fw_update.verbs, "fw_upd_url", lambda link, url: None)
    monkeypatch.setattr(fw_update.verbs, "fw_check", lambda link: None)
    monkeypatch.setattr(fw_update.verbs, "fw_upd_status",
                        lambda link: _status(err="catalog:network", stack_free=0))
    with pytest.raises(fw_update._Stop) as stop:
        _scenario(tmp_path)._tls_stack()
    assert stop.value.exit_code == 2


def test_a_catalog_404_still_counts_as_a_handshake(tmp_path, monkeypatch):
    monkeypatch.setattr(fw_update.verbs, "fw_upd_url", lambda link, url: None)
    monkeypatch.setattr(fw_update.verbs, "fw_check", lambda link: None)
    monkeypatch.setattr(fw_update.verbs, "fw_upd_status", lambda link: _status(err="catalog:http"))
    _scenario(tmp_path)._tls_stack()


def test_restore_runs_every_step_after_a_cancel(tmp_path, monkeypatch):
    calls = []
    sc = _scenario(tmp_path)
    sc.ctx.cancelled.set()
    sc.data["kit"] = "full"
    sc.original = {"channel": "stable", "wifi": 0, "kit": 3, "app": None}
    sc.saved_ssid = "bench"
    v = fw_update.verbs
    monkeypatch.setattr(fw_update, "_live_link", lambda ctx, link: link)
    monkeypatch.setattr(v, "net_mode_status", lambda link: {"active": True, "tab": "updates"})
    monkeypatch.setattr(sc, "_mode", lambda on, tab="updates": (_ for _ in ()).throw(HilError(NOT_SUPPORTED, "x")))
    monkeypatch.setattr(v, "fw_upd_url", lambda link, url: calls.append(("url", url)))
    monkeypatch.setattr(v, "fw_upd_channel", lambda link, kit, ch: calls.append(("channel", ch)))
    monkeypatch.setattr(v, "fw_upd_cfg", lambda link, **kw: calls.append(("cfg", kw)))
    monkeypatch.setattr(v, "wifi_forget", lambda link, ssid: calls.append(("forget", ssid)))
    monkeypatch.setattr(v, "wifi_enable", lambda link, on: calls.append(("wifi", on)))
    monkeypatch.setattr(v, "kit_status", lambda link: {"current": 3})
    monkeypatch.setattr(sc, "_unpin", lambda: calls.append(("unpin",)))
    sc._restore()
    assert not sc.ctx.cancelled.is_set()
    assert [c[0] for c in calls] == ["url", "channel", "cfg", "unpin", "forget", "wifi"]


def test_server_that_cannot_bind_is_a_bench_problem(tmp_path, monkeypatch):
    def refuse(*a, **k):
        raise OSError("address in use")
    monkeypatch.setattr(fw_update.fw_server, "FwServer", refuse)
    ctx = _Ctx(device=make_device(), workdir=tmp_path, cancelled=threading.Event(), log=lambda s: None)
    monkeypatch.setattr(fw_update.FwUpdateScenario, "_tls_stack", lambda self: None)
    monkeypatch.setattr(fw_update.fw_server, "fetch_release", lambda *a: _Real())
    monkeypatch.setattr(fw_update.fw_server, "alias", lambda *a, **k: _Real())
    monkeypatch.setattr(fw_update.verbs, "net_mode_status", lambda link: {"active": False, "tab": None})
    monkeypatch.setattr(fw_update.verbs, "wifi_list", lambda link: ["bench"])
    monkeypatch.setattr(fw_update.verbs, "wifi_status", lambda link: {"enabled": 1})
    monkeypatch.setattr(fw_update.verbs, "wifi_enable", lambda link, on: None)
    monkeypatch.setattr(fw_update, "_to_launcher", lambda ctx, link: None)
    monkeypatch.setattr(fw_update.verbs, "kit_status", lambda link: {"current": 1})
    monkeypatch.setattr(fw_update.verbs, "fw_upd_status", lambda link: {"kit": "full", "running": "1", "channel": "stable"})
    monkeypatch.setattr(fw_update.verbs, "app_versions", lambda link: {"board": {"fw": "v2"}})
    report = fw_update.FwUpdateScenario().run(ctx, fw_update.Params(host_ip="10.42.0.1", tag="v1"), base.Progress())
    assert report.exit_code == 2


class _Real:
    kit, version, channel = "full", "1.0.0", "stable"


@pytest.mark.parametrize("running,release,want", [
    ("v1.3.0", "1.1.0-fwstore.3", "up_to_date"),
    ("1.1.0-fwstore.3", "1.1.0-fwstore.3", "up_to_date"),
    ("v1.1.0-rc9", "1.1.0-rc10", "available"),
    ("v1.1.0-rc10", "1.1.0-rc9", "up_to_date"),
    ("v1.0.0", "1.1.0", "available"),
    ("v1.1.0-rc2", "1.1.0", "available"),
    ("v1.3.0-4-gabc1234-dirty", "1.1.0", "up_to_date"),
    ("v1.0.0-4-gabc1234", "1.1.0", "available"),
    ("v1.1.0-4-gabc1234", "1.1.0", None),
])
def test_expected_check_follows_the_running_version(running, release, want):
    assert fw_update.expected_check(running, release) == want


class _Srv:
    def __init__(self):
        self.faults = fw_update.fw_server.Faults()


def _catalog_stubs(monkeypatch, sc, *, second, st):
    v = fw_update.verbs
    seen = {}
    monkeypatch.setattr(sc, "_settle", lambda t: st)
    monkeypatch.setattr(sc, "_start_check", lambda: seen.update(delay=sc._srv.faults.catalog_delay_s))
    monkeypatch.setattr(v, "fw_check", second)
    monkeypatch.setattr(v, "fw_catalog", lambda link, kit: {"versions": [{"version": "1.1.0"}]})
    return seen


def _real():
    return type("R", (), {"version": "1.1.0", "kit": "full"})()


def _busy(link):
    raise HilError(NOT_SUPPORTED, "FW_CHECK: ERR busy")


def test_catalog_holds_the_first_check_so_the_second_is_busy(tmp_path, monkeypatch):
    sc, srv = _scenario(tmp_path), _Srv()
    sc._srv = srv
    st = _status(state="available", running="v1.0.0", latest="1.1.0")
    seen = _catalog_stubs(monkeypatch, sc, second=_busy, st=st)
    sc._catalog(_real(), srv)
    assert seen["delay"] == fw_update.BUSY_HOLD_S
    assert srv.faults.catalog_delay_s == 0.0          # released afterwards


def test_catalog_second_check_that_is_not_busy_fails(tmp_path, monkeypatch):
    sc, srv = _scenario(tmp_path), _Srv()
    sc._srv = srv
    st = _status(state="available", running="v1.0.0", latest="1.1.0")
    _catalog_stubs(monkeypatch, sc, second=lambda link: None, st=st)
    with pytest.raises(fw_update._Stop, match="busy"):
        sc._catalog(_real(), srv)
    assert srv.faults.catalog_delay_s == 0.0


def test_catalog_newer_running_is_up_to_date_with_no_latest(tmp_path, monkeypatch):
    sc, srv = _scenario(tmp_path), _Srv()
    sc._srv = srv
    st = _status(state="up_to_date", running="v1.3.0", latest=None)
    _catalog_stubs(monkeypatch, sc, second=_busy, st=st)
    sc._catalog(_real(), srv)


def test_catalog_older_running_must_end_available(tmp_path, monkeypatch):
    sc, srv = _scenario(tmp_path), _Srv()
    sc._srv = srv
    st = _status(state="up_to_date", running="v1.0.0", latest=None)
    _catalog_stubs(monkeypatch, sc, second=_busy, st=st)
    with pytest.raises(fw_update._Stop, match="should end available"):
        sc._catalog(_real(), srv)


def test_catalog_available_must_name_the_release(tmp_path, monkeypatch):
    sc, srv = _scenario(tmp_path), _Srv()
    sc._srv = srv
    st = _status(state="available", running="v1.0.0", latest=None)
    _catalog_stubs(monkeypatch, sc, second=_busy, st=st)
    with pytest.raises(fw_update._Stop, match="latest"):
        sc._catalog(_real(), srv)


def test_card_full_is_probed_with_the_real_package(tmp_path, monkeypatch):
    sc = _scenario(tmp_path)
    v = fw_update.verbs
    calls = []
    monkeypatch.setattr(v, "fs_roots", lambda link: {"roots": {}})
    monkeypatch.setattr(v, "fs_list", lambda link, path: {"entries": []})
    monkeypatch.setattr(v, "fs_mkdir", lambda link, path: None)
    monkeypatch.setattr(v, "fw_upd_cfg", lambda link, **kw: calls.append(("cfg", kw)))
    monkeypatch.setattr(fw_update, "library_bytes", lambda link: 0)
    monkeypatch.setattr(sc, "_remove", lambda name: calls.append(("remove", name)))
    monkeypatch.setattr(sc, "_push", lambda pairs: None)
    monkeypatch.setattr(sc, "_exists", lambda path: False)
    monkeypatch.setattr(sc, "_download_ok", lambda *a: calls.append(("ok",)))
    monkeypatch.setattr(sc, "_refused_download", lambda *a: calls.append(("refused", a)))
    monkeypatch.setattr(sc, "_unpin", lambda: None)
    sc._cleanup("full", _real(), "full-1.1.0", 10)
    assert calls[-4:-1] == [("remove", "full-1.1.0"), ("cfg", {"gc_mb": 1}),
                          ("refused", ("full", "1.1.0", "card_full"))]
    assert calls[::-1].index(("remove", "full-1.1.0")) < calls[::-1].index(("ok",))
    assert not any(c[0] == "refused" and c[1][0] == "zzmis" for c in calls)
