"""The performer's plan, putting the board back, and the live monitor."""
from __future__ import annotations

import threading
from pathlib import Path
from typing import Any

from crosspad_hil.jam import restore as RS
from crosspad_hil.jam import script as J
from crosspad_hil.jam.actions import ActionLog
from crosspad_hil.jam.control import UacLink
from crosspad_hil.jam.monitor import LiveMonitor
from crosspad_hil.jam.performer import Performer
from crosspad_hil.jam.pilot import PilotSource
from crosspad_hil.jam.spec import PilotSpec


class NullIO:
    def __init__(self) -> None:
        self.sent: list[list[int]] = []

    def send_sysex(self, frame: bytes) -> None:
        self.sent.append(list(frame))

    def send_note(self, on: bool, note: int, vel: int = 100) -> None:
        self.sent.append([0x90 if on else 0x80, note, vel])

    def send_realtime(self, status: int) -> None:
        self.sent.append([status])

    def receive(self, timeout_s: float = 0.0) -> None:
        return None


def test_the_plan_plays_every_step_in_time_order_and_logs_the_real_send() -> None:
    script = J.build_script(1, minutes=1.0)
    block = script.block("B2")
    t = [100.0]
    log = ActionLog(None, clock=lambda: t[0])
    io = NullIO()
    link = UacLink(io, log, clock=lambda: t[0])
    perf = Performer(link, log, kits={"HIL JAM A": 20, "HIL JAM B": 21, "HIL JAM FX": 22,
                                      "HIL JAM P": 23},
                     scenes={"S1": 2, "S2": 3, "S3": 4}, clock=lambda: t[0],
                     sleep=lambda s: t.__setitem__(0, t[0] + s), cancelled=threading.Event())
    events = perf.plan(block, 100.0)
    assert [e.t for e in events] == sorted(e.t for e in events)
    run = perf.run(events)
    assert run.hung is None and run.sent == len(events) and run.failed == 0
    kinds = {a.kind for a in log.actions}
    assert {"step", "pad", "release", "kit", "transport", "clock_tick", "app",
            "bounce"} <= kinds
    assert [s["name"] for s in run.swaps][:2] == ["HIL JAM FX", "HIL JAM A"]
    assert any(m[:4] == [0xF0, 0x7D, 0x1D, 0x0C] for m in io.sent)          # the bounce
    assert any(m[:3] == [0xF0, 0x7D, 0x1F] for m in io.sent)                # companion
    for a in log.actions:
        assert a.t_mono >= 100.0


def test_a_watchdog_answer_stops_the_block() -> None:
    script = J.build_script(1, minutes=1.0)
    t = [0.0]
    log = ActionLog(None, clock=lambda: t[0])
    perf = Performer(UacLink(NullIO(), log, clock=lambda: t[0]), log, kits={}, scenes={},
                     clock=lambda: t[0], sleep=lambda s: t.__setitem__(0, t[0] + s),
                     watchdog=lambda: "capture stopped: no frames for 2 s" if t[0] > 3 else None)
    run = perf.run(perf.plan(script.block("B1"), 0.0))
    assert run.hung == "capture stopped: no frames for 2 s" and 3.0 <= run.t_hung < 3.6
    assert log.actions[-1].kind == "hung"


class Board:
    def __init__(self) -> None:
        self.calls: list[str] = []
        self.scenes = 5


def test_put_back_never_deletes_a_scene_that_was_there(monkeypatch) -> None:
    from crosspad_hil import verbs

    b = Board()

    def seq(link: Any, action: str | None = None, timeout_s: float | None = None) -> dict:
        if action:
            b.calls.append(f"SEQ {action}")
            if action.startswith("DEL"):
                b.scenes -= 1
        return {"scenes": b.scenes, "dirty": 0}

    monkeypatch.setattr(verbs, "seq", seq)
    for name in ("asong", "song_text", "mix_preset", "click"):
        monkeypatch.setattr(verbs, name, lambda link, *a, n=name, **k: b.calls.append(n) or {})
    monkeypatch.setattr(verbs, "app_list", lambda link: {"running": None})
    monkeypatch.setattr(verbs, "kit_status", lambda link: {"current": 7})
    home = RS.HomeState(seq={"scenes": 2, "bpm": 100.0}, scenes0=2, kit={"id": 7, "name": "K"},
                        mix_dir="/sdcard/crosspad/mixer")
    left = RS.put_back_ram(object(), home, jam_scenes=[2, 3, 4, 1],
                           kit_id_by_name=lambda n: 7)
    dels = [c for c in b.calls if c.startswith("SEQ DEL")]
    assert dels == ["SEQ DEL 4", "SEQ DEL 3", "SEQ DEL 2"]          # never 1: it was the user's
    assert left["scene_1"] == "unchanged" and left["kit"] == "unchanged"
    assert "SEQ BPM 100" in b.calls


def test_put_back_files_restores_copies_and_removes_what_was_not_there(tmp_path: Path) -> None:
    keep = tmp_path / "seq.json"
    keep.write_text("{}")
    home = RS.HomeState(files={"/a/sequences.json": str(keep), "/a/preset_4.json": None})
    pushed, deleted = [], []
    left = RS.put_back_files(home, lambda loc, r: pushed.append((loc, r)), deleted.append)
    assert pushed == [(keep, "/a/sequences.json")] and deleted == ["/a/preset_4.json"]
    assert left["/a/preset_4.json"].startswith("removed")


def test_the_monitor_hears_a_pilot_drop_and_a_click() -> None:
    sr = 44_100
    x = PilotSource(PilotSpec(), sr, 1).samples(0, 3 * sr)
    x[int(1.6 * sr):int(1.62 * sr)] = 0.0
    x[int(2.4 * sr)] += 0.5
    mon = LiveMonitor(sr)
    events = []
    for k in range(6):
        events += mon.feed(x[k * sr // 2:(k + 1) * sr // 2], k * 0.5)
    kinds = [(e["event"], e["t"]) for e in events]
    assert any(k == "pilot_drop" and 1.55 <= t <= 1.65 for k, t in kinds)
    assert any(k == "quiet_band_energy" and t == 2.0 for k, t in kinds)


def test_a_failed_seq_read_stops_the_snapshot(monkeypatch: Any) -> None:
    """Guessing scenes0=0 would make putting back delete the user's scenes."""
    from crosspad_hil import verbs
    from crosspad_hil.errors import TIMEOUT, HilError

    def seq(link: Any, *a: Any, **k: Any) -> dict[str, Any]:
        raise HilError(TIMEOUT, "SEQ: no reply")

    monkeypatch.setattr(verbs, "seq", seq)
    try:
        RS.snapshot(object())
    except HilError as e:
        assert e.code == TIMEOUT
    else:
        raise AssertionError("snapshot must not guess the sequencer")


def test_only_a_missing_file_is_recorded_as_absent(tmp_path: Path) -> None:
    from crosspad_hil.errors import FLASH_FAILED, HilError

    home = RS.HomeState(mix_dir="/m", seq_file="/s/sequences.json")

    def pull(remote: str, local: Path) -> None:
        if remote.endswith("preset_4.json"):
            raise HilError(FLASH_FAILED, "device reported FS_ERR stat_2", reply="FS_ERR stat_2")
        local.write_text("x")

    RS.fetch_files(home, pull, tmp_path, slots=(4, 5))
    assert home.files["/m/preset_4.json"] is None
    assert home.files["/s/sequences.json"] is not None

    def stalled(remote: str, local: Path) -> None:
        raise HilError(FLASH_FAILED, "pull stalled at 10/400 bytes")

    try:
        RS.fetch_files(RS.HomeState(seq_file="/s/sequences.json"), stalled, tmp_path)
    except HilError:
        pass
    else:
        raise AssertionError("a stalled pull is not proof the file is absent")


def test_no_scene_is_deleted_without_the_home_sequencer(monkeypatch: Any) -> None:
    from crosspad_hil import verbs

    calls: list[str] = []
    monkeypatch.setattr(verbs, "seq", lambda link, action=None, **k: calls.append(
        f"SEQ {action}") or {"scenes": 8, "dirty": 0})
    for name in ("asong", "song_text", "mix_preset", "click"):
        monkeypatch.setattr(verbs, name, lambda link, *a, **k: {})
    monkeypatch.setattr(verbs, "app_list", lambda link: {"running": None})
    RS.put_back_ram(object(), RS.HomeState(), jam_scenes=[0, 1, 2, 3])
    assert not [c for c in calls if c.startswith("SEQ DEL")]


def test_an_incomplete_put_back_keeps_the_journal(tmp_path: Path, monkeypatch: Any) -> None:
    from crosspad_hil import pins, verbs
    from crosspad_hil.errors import FLASH_FAILED, HilError

    jpath = tmp_path / "pins.json"
    home = RS.HomeState(seq={"scenes": 2}, scenes0=2, files={"/s/sequences.json": str(tmp_path)})
    RS.journal("dev", home, jpath)
    monkeypatch.setattr(verbs, "seq", lambda link, action=None, **k: {"scenes": 2, "dirty": 0})
    for name in ("asong", "song_text", "mix_preset", "click"):
        monkeypatch.setattr(verbs, name, lambda link, *a, **k: {})
    monkeypatch.setattr(verbs, "app_list", lambda link: {"running": None})

    def push(local: Path, remote: str) -> None:
        raise HilError(FLASH_FAILED, "push stalled")

    left = RS.restore_pending(object(), "dev", lambda m: None, push=push,
                              delete=lambda r: None, path=jpath)
    assert left is not None and left["/s/sequences.json"].startswith("FLASH_FAILED")
    assert RS.PIN_KEY in pins.pending("dev", jpath)


def test_no_snapshot_before_the_sequencer_has_read_the_card(monkeypatch: Any) -> None:
    """7.10 on dev_1ede: a snapshot right after a boot saw 1 scene and no file; the
    card held 8 scenes, so putting back would have deleted scenes 1-3 of the user's."""
    from crosspad_hil import verbs
    from crosspad_hil.errors import ENV, HilError

    replies = [{"scenes": 1, "dirty": 1, "file": ""}] * 3 + [
        {"scenes": 8, "dirty": 0, "file": "/sdcard/crosspad/sequences.json"}]
    monkeypatch.setattr(verbs, "seq", lambda link, *a, **k: replies[0])
    try:
        RS.snapshot(object())
    except HilError as e:
        assert e.code == ENV
    else:
        raise AssertionError("an unloaded sequencer is not the user's home state")

    seen = iter(replies)
    monkeypatch.setattr(verbs, "seq", lambda link, *a, **k: next(seen))
    t = [0.0]
    st = RS.wait_seq_store(object(), sleep=lambda s: t.__setitem__(0, t[0] + s),
                           clock=lambda: t[0])
    assert st["scenes"] == 8 and t[0] == 3.0


def test_the_end_check_reads_an_unloaded_sequencer_as_nothing_to_compare(monkeypatch: Any) -> None:
    """With no home kit the store does not read the card after the reboot: no
    comparison of scenes then (the file on the card is compared instead)."""
    from crosspad_hil import verbs

    monkeypatch.setattr(verbs, "seq", lambda link, *a, **k: {"scenes": 1, "file": ""})
    for name in ("song_text", "click", "kit_status", "app_list", "mix_preset", "mix_lvl",
                 "asong", "usb_guard"):
        monkeypatch.setattr(verbs, name, lambda link, *a, **k: {})
    now = RS.snapshot(object(), need_file=False)
    assert now.seq == {}
    home = RS.HomeState(seq={"scenes": 1, "bpm": 120.0})
    assert "seq.scenes" not in RS.board_diff(home, now)


def test_lost_midi_replies_with_a_live_console_are_not_a_hang() -> None:
    """7.10 run 3: three echoes went unanswered while the board kept logging and
    taking SysEx; the old watchdog reset it. Now it is counted, unless the console
    shows a hang signature."""
    from types import SimpleNamespace

    from crosspad_hil.scenarios import live_jam

    events: list[tuple[str, dict[str, Any]]] = []

    class Con:
        seq = 10
        signs: list[str] = []

        def read(self, since_seq: Any = None, match: Any = None, limit: int = 0) -> Any:
            return SimpleNamespace(lines=list(self.signs))

    con = Con()
    link = SimpleNamespace(echo_sent={1: -10.0, 2: -10.0, 3: -10.0}, companion_misses=0)
    link.echo_pending_over = lambda s: sum(1 for t in link.echo_sent.values() if t < -5)
    st = SimpleNamespace(con=con, log=SimpleNamespace(
        emit=lambda kind, **kw: events.append((kind, kw))))
    check = live_jam.LiveJamScenario._watchdog(None, st, None, link)
    assert check() is None and events[0][0] == "midi_in_lost" and not link.echo_sent

    link.echo_sent.update({4: -10.0, 5: -10.0, 6: -10.0})
    con.signs = ["E (1) task_wdt: Task watchdog got triggered"]
    assert check() == "3 echoes unanswered"
