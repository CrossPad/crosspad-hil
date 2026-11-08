"""Scenario firmware_app's bench checks and helpers, without a board."""
from __future__ import annotations

import hashlib
import json
import threading
from dataclasses import dataclass

import pytest

from crosspad_hil.errors import NOT_SUPPORTED, HilError
from crosspad_hil.scenarios import base, firmware_app, get
from tests.test_scenario_fw_switch import make_device


@dataclass
class _Ctx(base.Context):
    def open_cdc(self):  # every verb that would use it is stubbed; the rest is echoed
        from crosspad_hil.cdc import CdcLink
        from tests.fakes import FakeSerial
        link = CdcLink("fake-fwapp", serial_factory=lambda path, **kw: FakeSerial(timeout=0.01))
        link.open()
        return link


def _scenario(tmp_path):
    ctx = _Ctx(device=make_device(), workdir=tmp_path, cancelled=threading.Event(), log=lambda s: None)
    sc = firmware_app.FirmwareAppScenario()
    sc.ctx, sc.p, sc.data, sc.link = ctx, firmware_app.Params(), {"steps": {}}, object()
    return sc


def test_no_saved_network_is_a_bench_problem(tmp_path, monkeypatch):
    monkeypatch.setattr(firmware_app.verbs, "net_mode_status", lambda link: {"active": False, "tab": None})
    monkeypatch.setattr(firmware_app.verbs, "wifi_list", lambda link: [])
    ctx = _Ctx(device=make_device(), workdir=tmp_path, cancelled=threading.Event(), log=lambda s: None)
    report = get("firmware_app").run(ctx, firmware_app.Params(), base.Progress())
    assert report.exit_code == 2
    assert "WiFi" in report.summary


def test_test_packages_carry_a_valid_icon_and_a_name(tmp_path):
    files = firmware_app.make_test_packages(tmp_path, 3)
    assert len(files) == 6
    icons = [local for local, remote in files if remote.endswith("/icon.bin")]
    data = icons[0].read_bytes()
    assert len(data) == firmware_app.ICON_BYTES == 6924
    assert data[:2] == bytes([0x19, 0x14])
    assert data[2:4] == b"\x00\x00"                      # the device rejects non-zero icon flags
    assert icons[0].read_bytes() != icons[1].read_bytes()   # different icons: no sharing in the budget
    manifest = next(local for local, remote in files if remote == f"{firmware_app.LIBRARY}/zzui00-1.0.0/manifest.json")
    doc = json.loads(manifest.read_text())
    assert doc["kit"] == "zzui00" and doc["display_name"] == "Test 00"
    # What core's parseFwManifest requires, so the Library names the kit and version without an image.
    assert doc["name"] == "zzui00-1.0.0" and doc["version"] == "1.0.0" and doc["board_rev"] >= 1
    image = doc["files"]["CrossPad.bin"]
    assert image["size"] > 0 and len(image["sha256"]) == 64
    assert doc["files"]["icon.bin"] == {"size": len(data), "sha256": hashlib.sha256(data).hexdigest()}
    assert not any(remote.endswith("/CrossPad.bin") for _, remote in files)   # listed, never pushed


def test_the_budget_counts_the_radio():
    assert firmware_app.Params().budget == 131072


def test_loss_per_visit_reads_app_churns_windows():
    assert firmware_app.loss_per_visit([5000, 4000, 4000, 4000, 4000, 4000]) == 0   # the first visit is not a leak
    assert firmware_app.loss_per_visit([10000 - 100 * i for i in range(8)]) == 100


def test_a_short_series_is_not_a_leak():
    assert firmware_app.loss_per_visit([5000, 4000]) == 0


class _Ring:
    """The encoder as the device moves it: only the members it can focus, in a wrapping ring."""

    def __init__(self, labels):
        self.labels, self.at, self.turns = labels, 0, 0

    def focus(self, link):
        return {"index": self.at, "label": self.labels[self.at], "ptr": f"0x{self.at:x}"}

    def rotate(self, link, delta):
        self.turns += 1
        self.at = (self.at + delta) % len(self.labels)
        return {}


def _walkable(sc, monkeypatch, labels, start=0):
    ring = _Ring(labels)
    ring.at = start
    monkeypatch.setattr(firmware_app.verbs, "enc_focus", ring.focus)
    monkeypatch.setattr(firmware_app.verbs, "enc_rotate", ring.rotate)

    def no_enc_group(link):
        raise AssertionError("ENC_GROUP lists hidden buttons; the walk decides what is offered")

    monkeypatch.setattr(firmware_app.verbs, "enc_group", no_enc_group)
    return ring


def test_a_pinned_card_must_not_offer_delete(tmp_path, monkeypatch):
    sc = _scenario(tmp_path)
    ring = _walkable(sc, monkeypatch, ["Back", "Install and run", "Unpin"], start=1)
    assert sc._has_control("Unpin") and not sc._has_control("Delete from card")
    assert ring.at == 1                                  # a whole turn: the focus is back where it was


def test_a_rolled_back_slot_offers_both_labels(tmp_path, monkeypatch):
    sc = _scenario(tmp_path)
    _walkable(sc, monkeypatch, ["Back", "Try this slot again", "Reinstall"])
    assert sc._has_control("Try this slot again") and sc._has_control("Reinstall")


def test_a_lone_control_ends_the_walk(tmp_path, monkeypatch):
    sc = _scenario(tmp_path)
    ring = _walkable(sc, monkeypatch, ["Back"])
    monkeypatch.setattr(firmware_app, "FOCUS_STEP_S", 0)
    assert sc._reachable() == ["Back"] and ring.turns == 1


def _card_bench(sc, monkeypatch, frees):
    """mem() answers `frees` in turn; the card opens and closes without a board."""
    it = iter(frees)
    monkeypatch.setattr(firmware_app.verbs, "mem", lambda link: {"psram_free": next(it)})
    monkeypatch.setattr(firmware_app.verbs, "fw_ui", lambda link: {"busy": 0})
    monkeypatch.setattr(firmware_app, "SETTLE_S", 0)
    calls = []
    monkeypatch.setattr(sc, "_open_card", lambda key: calls.append(("open", key)))
    monkeypatch.setattr(sc, "_close_card", lambda: calls.append(("close",)))
    return calls


def test_the_card_is_measured_on_its_own(tmp_path, monkeypatch):
    sc = _scenario(tmp_path)
    calls = _card_bench(sc, monkeypatch, [400000, 352568, 400580])   # the bench: +47432, then -580
    sc._card_cost("full@1.1.0", "store")
    assert calls == [("open", "full@1.1.0"), ("close",)]
    assert sc.data["steps"]["card_store"] == {"opened": 47432, "kept_after_close": -580}


def test_a_card_over_its_budget_fails(tmp_path, monkeypatch):
    sc = _scenario(tmp_path)
    _card_bench(sc, monkeypatch, [400000, 400000 - firmware_app.CARD_BUDGET_BYTES - 1, 400000])
    with pytest.raises(firmware_app._Stop, match="card budget"):
        sc._card_cost("k@1", "store")


def test_a_card_that_keeps_memory_after_close_fails(tmp_path, monkeypatch):
    sc = _scenario(tmp_path)
    _card_bench(sc, monkeypatch, [400000, 360000, 400000 - firmware_app.CARD_TOLERANCE_BYTES - 1])
    with pytest.raises(firmware_app._Stop, match="kept"):
        sc._card_cost("k@1", "store")


def test_the_store_budget_is_taken_before_the_card_opens(tmp_path, monkeypatch):
    sc = _scenario(tmp_path)
    order = []
    monkeypatch.setattr(firmware_app.verbs, "fw_ui", lambda link: {
        "host": "app", "tab": "store", "store": "ok", "items": 1, "busy": 0})
    monkeypatch.setattr(sc, "_press", lambda label: order.append(("press", label)))
    monkeypatch.setattr(sc, "_budget", lambda step, base: order.append(("budget", step)))
    monkeypatch.setattr(sc, "_card_cost", lambda key, step: order.append(("card", key)))
    monkeypatch.setattr(sc, "_open_card", lambda key: order.append(("open", key)))
    monkeypatch.setattr(firmware_app.verbs, "fw_upd_status", lambda link: {"state": "idle", "err": None})
    monkeypatch.setattr(firmware_app.verbs, "fw_catalog", lambda link, kit: {
        "versions": [{"version": "1", "on_card": True}]})
    real = type("R", (), {"kit": "k", "version": "1"})()
    sc._store_download(real, "k-1", "store", base_free=500000)
    assert order == [("press", "Store"), ("budget", "store"), ("card", "k@1"), ("open", "k@1"),
                     ("press", "Download")]


def test_restart_waits_for_the_board_to_go(tmp_path, monkeypatch):
    sc = _scenario(tmp_path)
    monkeypatch.setattr(firmware_app, "POLL_S", 0)
    answers = iter([{"slots": []}, {"slots": []}])

    def fw_slots(link):
        try:
            return next(answers)                         # still verifying the image
        except StopIteration:
            raise HilError(firmware_app.TIMEOUT, "no bytes at all arrived") from None

    monkeypatch.setattr(firmware_app.verbs, "fw_slots", fw_slots)
    sc._wait_gone()
    with pytest.raises(StopIteration):
        next(answers)                                    # both answers were read before it counted as gone


def test_restart_ends_when_the_port_node_goes(tmp_path, monkeypatch):
    sc = _scenario(tmp_path)
    sc.link = type("L", (), {"port": str(tmp_path / "ttyACM-gone")})()
    monkeypatch.setattr(firmware_app.verbs, "fw_slots", lambda link: pytest.fail("the node is gone already"))
    sc._wait_gone()


def test_a_board_that_never_restarts_fails(tmp_path, monkeypatch):
    sc = _scenario(tmp_path)
    monkeypatch.setattr(firmware_app, "POLL_S", 0)
    monkeypatch.setattr(firmware_app, "RESTART_GONE_S", 0.05)
    monkeypatch.setattr(firmware_app.verbs, "fw_slots", lambda link: {"slots": []})
    with pytest.raises(firmware_app._Stop, match="still answers"):
        sc._wait_gone()


@pytest.mark.parametrize("token", ["no_mem", "no_mem:job_task", "no_mem:install_task"])
def test_every_no_mem_token_ends_a_download_as_a_memory_refusal(tmp_path, monkeypatch, token):
    sc = _scenario(tmp_path)
    st = {"state": "failed", "err": token}
    monkeypatch.setattr(firmware_app.verbs, "fw_upd_status", lambda link: st)
    monkeypatch.setattr(firmware_app.verbs, "fw_ui", lambda link: {
        "host": "net", "tab": "store", "store": "ok", "items": 1, "busy": 0, "selkey": "k@1", "card": "k@1"})
    monkeypatch.setattr(firmware_app.verbs, "enc_focus", lambda link: {"label": "Download"})
    monkeypatch.setattr(firmware_app.verbs, "enc_press", lambda link: {})
    monkeypatch.setattr(firmware_app, "SETTLE_S", 0)           # no real waits in the unit test
    monkeypatch.setattr(sc, "_card_cost", lambda key, step: None)
    real = type("R", (), {"kit": "k", "version": "1"})()
    with pytest.raises(firmware_app._Stop, match="memory"):
        sc._store_download(real, "k-1", "store", host="net")


def test_the_scenario_is_listed_after_fw_update():
    # Grouped with the firmware scenarios, after fw_update; others of that
    # group (release_check) may sit in between -- firmware_app checks the
    # catalog live and puts back what it changes, so it needs no neighbour.
    from crosspad_hil.scenarios import BUILTIN_MODULES
    assert BUILTIN_MODULES.index("firmware_app") > BUILTIN_MODULES.index("fw_update")


class _Card:
    """FS_STAT / FS_LIST / FS_DELETE over a dict of paths (a directory maps to None)."""

    def __init__(self, paths):
        self.paths = dict(paths)

    def stat(self, link, path):
        if path not in self.paths:
            raise HilError(NOT_SUPPORTED, "no such file")
        return {"path": path, "size": 0, "dir": self.paths[path] is None}

    def list(self, link, path):
        kids = [p for p in self.paths if p.rsplit("/", 1)[0] == path]
        return {"path": path, "entries": [{"name": p.rsplit("/", 1)[1], "dir": self.paths[p] is None, "size": 0}
                                          for p in kids]}

    def delete(self, link, path):
        assert not any(p.rsplit("/", 1)[0] == path for p in self.paths), f"{path} still has files"
        del self.paths[path]


def _card(sc, monkeypatch, paths):
    card = _Card(paths)
    monkeypatch.setattr(firmware_app.verbs, "fs_stat", card.stat)
    monkeypatch.setattr(firmware_app.verbs, "fs_list", card.list)
    monkeypatch.setattr(firmware_app.verbs, "fs_delete", card.delete)
    moved = []
    monkeypatch.setattr(firmware_app.fs, "pull", lambda dev, remote, local: (
        moved.append(("pull", remote)), local.write_bytes(card.paths[remote])))
    monkeypatch.setattr(firmware_app.fs, "push", lambda dev, local, remote: (
        moved.append(("push", remote)), card.paths.__setitem__(remote, local.read_bytes())))
    link = type("L", (), {"close": lambda self: None, "wait_ready": lambda self: None})()
    sc.link = link
    sc.ctx.open_cdc = lambda: link
    sc.original = {}
    return card, moved


def test_the_bench_catalog_leaves_the_card_and_the_cards_own_cache_comes_back(tmp_path, monkeypatch):
    lib = firmware_app.LIBRARY
    sc = _scenario(tmp_path)
    card, moved = _card(sc, monkeypatch, {
        lib: None, f"{lib}/.catalog.json": b"real catalog", f"{lib}/.community_ok": b"drums\n",
        f"{lib}/full-1.3.0": None, f"{lib}/full-1.3.0/manifest.json": b"{}"})
    sc._snapshot_card_cache()
    assert sc.original["card_cache"] == [".catalog.json", ".community_ok"]
    # What the run's checks against the bench server write.
    card.paths.update({f"{lib}/.catalog.json": b"bench catalog", f"{lib}/.catalog.json.tmp": b"x",
                       f"{lib}/.icons": None, f"{lib}/.icons/full-1.3.0.bin": b"i",
                       f"{lib}/.details": None, f"{lib}/.details/full.json": b"{}"})
    sc._restore_card_cache()
    assert card.paths == {lib: None, f"{lib}/.catalog.json": b"real catalog", f"{lib}/.community_ok": b"drums\n",
                          f"{lib}/full-1.3.0": None, f"{lib}/full-1.3.0/manifest.json": b"{}"}
    assert ("push", f"{lib}/.catalog.json") in moved


def test_a_card_without_a_cache_is_left_without_one(tmp_path, monkeypatch):
    lib = firmware_app.LIBRARY
    sc = _scenario(tmp_path)
    card, moved = _card(sc, monkeypatch, {lib: None})
    sc._snapshot_card_cache()
    assert sc.original["card_cache"] == [] and moved == []
    card.paths.update({f"{lib}/.catalog.json": b"bench", f"{lib}/.community_ok": b"zz\n",
                       f"{lib}/.icons": None, f"{lib}/.icons/a.bin": b"i"})
    sc._restore_card_cache()
    assert card.paths == {lib: None} and moved == []


def test_restore_touches_no_cache_the_run_never_saved(tmp_path, monkeypatch):
    sc = _scenario(tmp_path)
    sc.original, sc.kit = {}, ""
    called = []
    monkeypatch.setattr(sc, "_restore_card_cache", lambda: called.append(True))
    monkeypatch.setattr(firmware_app, "_live_link", lambda ctx, link: link)
    monkeypatch.setattr(firmware_app.verbs, "net_mode_status", lambda link: {"active": False})
    monkeypatch.setattr(firmware_app.verbs, "app_stop", lambda link: None)
    monkeypatch.setattr(firmware_app.verbs, "fw_upd_url", lambda link, url: None)
    monkeypatch.setattr(sc, "_remove_test_packages", lambda: None)
    sc._restore()
    assert called == []


def test_restore_puts_the_mixer_and_codec_state_back_last(tmp_path, monkeypatch):
    from tests.fakes import AudioDesk
    from tests.test_audio_state import link_for
    desk = AudioDesk()
    link = link_for(desk)
    sc = _scenario(tmp_path)
    sc.original, sc.kit, sc.link = {}, "", link
    sc.audio_snap = firmware_app.audio_state.snapshot(link)
    desk.vol, desk.adc = [28, 50], [1, 2]            # a walk through the cards found the overlay
    logged = []
    sc.ctx.log = logged.append
    monkeypatch.setattr(firmware_app, "_live_link", lambda ctx, link: link)
    monkeypatch.setattr(firmware_app.verbs, "net_mode_status", lambda link: {"active": False})
    monkeypatch.setattr(firmware_app.verbs, "app_stop", lambda link: None)
    monkeypatch.setattr(firmware_app.verbs, "fw_upd_url", lambda link, url: None)
    monkeypatch.setattr(sc, "_remove_test_packages", lambda: None)
    sc._restore()
    assert desk.vol == [64, 64] and desk.adc == [2, 1]
    assert len(sc.data["audio_state_restored"]) == 4
    assert any("audio_state: restored codec volume codec 1: 50 -> 64" in line for line in logged)
