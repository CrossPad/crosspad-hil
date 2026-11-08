"""Scenario `release_gate`: the per-step watchdog and how it interacts with --keep-going.

Every step is substituted with a fake (via monkeypatching release_gate.get), so these
tests exercise only the gate's own orchestration — timeout, device recheck, stop vs.
keep-going — never a real step's logic. That mirrors the real bug this covers: a
release_gate run stuck 4.5h in usb_mode_cycle with no console output and no
report.json, because nothing ever bounded a composed step's run() call.
"""
from __future__ import annotations

import json
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest

from crosspad_hil.devices import Device, Ports, UsbMode
from crosspad_hil.scenarios import base, release_gate

DEV = Device(id="dev_ab12", serial="CP-1", usb_mode=UsbMode.DEFAULT, ports=Ports())


@dataclass
class FakeStepParams:
    """One shape that satisfies every branch of release_gate._step_params."""

    timeout: float = 0.0
    rounds: int = 0
    rapid: float = 0.0
    kit: int | None = None
    scan_timeout: float = 0.0
    connect_timeout: float = 0.0
    duration_hours: float | None = None
    silence_timeout: float = 0.0
    settle_s: float = 0.0
    max_wait_s: float = 0.0
    swaps: int = 0
    max_cdc_ms: float = 0.0
    max_uac_ratio: float = 0.0
    depth: str = "quick"
    max_init_ms: float = 0.0
    stream: bool = True


class FakeScenario:
    def __init__(self, name: str,
                 behavior: Callable[[base.Context, Any, base.Progress], base.Report]) -> None:
        self.name = name
        self.Params = FakeStepParams
        self.description = f"fake {name}"
        self._behavior = behavior

    def run(self, ctx: base.Context, params: Any, progress: base.Progress) -> base.Report:
        return self._behavior(ctx, params, progress)


def pass_report(ctx: base.Context, params: Any, progress: base.Progress) -> base.Report:
    return base.Report(passed=True, summary="ok", data={}, artifacts=[], exit_code=0)


def hang_forever(ctx: base.Context, params: Any, progress: base.Progress) -> base.Report:
    time.sleep(3600)  # never returns inside the shrunk test timeout below
    return base.Report(passed=True, summary="unreachable", data={}, artifacts=[], exit_code=0)


def make_ctx(tmp_path: Path) -> base.Context:
    return base.Context(device=DEV, workdir=tmp_path, cancelled=threading.Event(),
                        log=lambda s: None)


def install_fakes(monkeypatch: pytest.MonkeyPatch, behaviors: dict[str, Callable]) -> None:
    scenarios = {name: FakeScenario(name, behaviors.get(name, pass_report))
                for name in release_gate.STEP_ORDER}
    monkeypatch.setattr(release_gate, "get", lambda name: scenarios[name])
    # Real floor is 120s; a hung fake step only needs to outlast this to prove the
    # watchdog fires — shrinking it is what keeps the test fast.
    monkeypatch.setattr(release_gate, "STEP_TIMEOUT_MIN_S", 0.05)
    # These tests exercise the orchestration, not the budgets: the estimate is
    # zeroed so the floor above is every step's budget, and a timed-out fake gets
    # a short cancel grace (hang_forever ignores it, which is the abandoned case).
    monkeypatch.setattr(release_gate, "_expected_step_seconds", lambda step, sp: 0.0)
    monkeypatch.setattr(release_gate, "STEP_CANCEL_GRACE_S", 0.1)
    # The profile guard re-discovers the board before every step; the fakes have
    # no bus to discover on, so the board is simply where the test says it is.
    monkeypatch.setattr(release_gate, "set_mode",
                        lambda d, m: (_ for _ in ()).throw(AssertionError("set_mode")))


def test_step_timeout_marks_step_timeout_and_stops_without_keep_going(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    install_fakes(monkeypatch, {"kit_churn": hang_forever})
    monkeypatch.setattr(release_gate, "discover", lambda: [DEV])
    rep = release_gate.SCENARIO.run(make_ctx(tmp_path), release_gate.Params(), base.Progress())

    assert not rep.passed
    assert rep.exit_code == 1
    steps = {s["step"]: s for s in rep.data["steps"]}
    assert steps["smoke"]["result"] == "PASS"
    assert steps["app_churn"]["result"] == "PASS"
    assert steps["kit_churn"]["result"] == "TIMEOUT"
    assert steps["kit_churn"]["timed_out"] is True
    assert "TIMEOUT" in steps["kit_churn"]["summary"]
    # everything after the stuck step was skipped, not attempted
    assert steps["usb_mode_cycle"]["result"] == "SKIP"
    assert steps["stability"]["result"] == "SKIP"

    # a report.json exists for the timed-out step even though its real work never
    # finished — this is the exact gap the bug report found empty
    report_path = tmp_path / "kit_churn" / "report.json"
    assert report_path.exists()
    doc = json.loads(report_path.read_text())
    assert doc["passed"] is False
    assert doc["data"]["timeout"] is True


def test_step_timeout_continues_with_keep_going_when_device_survives(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    install_fakes(monkeypatch, {"kit_churn": hang_forever})
    monkeypatch.setattr(release_gate, "discover", lambda: [DEV])  # still there afterwards
    rep = release_gate.SCENARIO.run(make_ctx(tmp_path), release_gate.Params(keep_going=True),
                                    base.Progress())

    assert not rep.passed
    assert rep.exit_code == 1
    steps = {s["step"]: s for s in rep.data["steps"]}
    assert steps["kit_churn"]["result"] == "TIMEOUT"
    # --keep-going ran every step after the timeout, since the device recheck found it
    assert steps["usb_mode_cycle"]["result"] == "PASS"
    assert steps["ble_midi"]["result"] == "PASS"
    assert steps["audio_loopback"]["result"] == "PASS"
    assert steps["stability"]["result"] == "PASS"
    assert rep.data["stopped_at"] is None


def test_step_timeout_with_device_lost_stops_even_with_keep_going(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    install_fakes(monkeypatch, {"kit_churn": hang_forever})
    monkeypatch.setattr(release_gate, "discover", lambda: [])  # gone after the timeout
    rep = release_gate.SCENARIO.run(make_ctx(tmp_path), release_gate.Params(keep_going=True),
                                    base.Progress())

    assert not rep.passed
    steps = {s["step"]: s for s in rep.data["steps"]}
    assert steps["kit_churn"]["result"] == "TIMEOUT"
    # a lost device stops the gate regardless of --keep-going: nothing downstream
    # has a device to run against
    assert steps["usb_mode_cycle"]["result"] == "SKIP"
    assert steps["stability"]["result"] == "SKIP"
    assert rep.data["stopped_at"] == "kit_churn"
    assert "unreachable" in steps["usb_mode_cycle"]["summary"]


def test_no_timeout_when_step_finishes_quickly(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    install_fakes(monkeypatch, {})
    monkeypatch.setattr(release_gate, "discover", lambda: [DEV])
    rep = release_gate.SCENARIO.run(make_ctx(tmp_path), release_gate.Params(), base.Progress())

    assert rep.passed
    assert rep.exit_code == 0
    for s in rep.data["steps"]:
        assert s["result"] == "PASS"
        assert s.get("timed_out") is False


# ── budgets ──────────────────────────────────────────────────────────────────
def test_churn_budgets_scale_with_rounds() -> None:
    """Ten rounds of app_churn are ~50 visits at ~3.5 s: a flat two-minute floor cut
    that off at round seven and left the abandoned step holding the ports."""
    import crosspad_hil.scenarios.app_churn  # noqa: F401 - registers the real Params
    import crosspad_hil.scenarios.kit_churn  # noqa: F401
    import crosspad_hil.scenarios.usb_mode_cycle  # noqa: F401
    p = release_gate.Params()
    app = release_gate._step_timeout_s(
        "app_churn", release_gate._step_params("app_churn", p), p.step_timeout_factor)
    assert app >= 10 * 5 * 3.5  # at least the measured pace, before the factor
    kit = release_gate._step_timeout_s(
        "kit_churn", release_gate._step_params("kit_churn", p), p.step_timeout_factor)
    assert kit >= 20 * 4.0
    # Eight rounds of back_churn measured 132-139 s each on the bench (ten targets,
    # depth 3); the 17-minute budget that came from a 60 s guess timed out a run
    # that was passing.
    import crosspad_hil.scenarios.back_churn  # noqa: F401
    back = release_gate._step_timeout_s(
        "back_churn", release_gate._step_params("back_churn", p), p.step_timeout_factor)
    assert back >= 8 * 139.0 * p.step_timeout_factor
    # A slow card: the gate passes load_timeout through and budgets the worst
    # case, a swap that takes the whole timeout before kit_churn calls it stuck.
    slow = release_gate.Params(kit_churn_load_timeout=30.0)
    slow_sp = release_gate._step_params("kit_churn", slow)
    assert slow_sp.load_timeout == 30.0
    assert release_gate._step_timeout_s("kit_churn", slow_sp, slow.step_timeout_factor) \
        >= 20 * 30.0 * slow.step_timeout_factor
    usb = release_gate._step_timeout_s(
        "usb_mode_cycle", release_gate._step_params("usb_mode_cycle", p), p.step_timeout_factor)
    assert usb >= 5 * 40.0
    quick = release_gate.Params(quick=True)
    assert release_gate._step_timeout_s(
        "app_churn", release_gate._step_params("app_churn", quick), quick.step_timeout_factor,
    ) < app


def test_timed_out_step_that_honours_cancel_is_not_abandoned(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    def hang_until_cancelled(ctx: base.Context, params: Any, progress: base.Progress) -> base.Report:
        ctx.cancelled.wait(3600)
        ctx.check_cancelled()
        return base.Report(passed=True, summary="unreachable", data={}, artifacts=[], exit_code=0)

    install_fakes(monkeypatch, {"kit_churn": hang_until_cancelled})
    monkeypatch.setattr(release_gate, "STEP_CANCEL_GRACE_S", 5.0)
    monkeypatch.setattr(release_gate, "discover", lambda: [DEV])
    ctx = make_ctx(tmp_path)
    rep = release_gate.SCENARIO.run(ctx, release_gate.Params(keep_going=True), base.Progress())
    steps = {s["step"]: s for s in rep.data["steps"]}
    assert steps["kit_churn"]["result"] == "TIMEOUT"
    doc = json.loads((tmp_path / "kit_churn" / "report.json").read_text())
    assert doc["data"]["abandoned"] is False
    assert "released its ports" in doc["summary"]
    # the step's private cancel did not cancel the gate itself
    assert not ctx.cancelled.is_set()
    assert steps["stability"]["result"] == "PASS"


# ── USB profile guard ────────────────────────────────────────────────────────
def test_board_left_in_audio_profile_is_switched_back_before_the_first_step(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A run killed inside audio_loopback leaves MIDI+UAC2 behind; without this every
    CDC step reported NO_CDC_IN_AUDIO_MODE as ENV and the gate said nothing."""
    install_fakes(monkeypatch, {})
    audio_dev = Device(id=DEV.id, serial=DEV.serial, usb_mode=UsbMode.AUDIO, ports=Ports())
    seen: list[UsbMode] = []

    def fake_set_mode(device: Device, mode: UsbMode) -> Device:
        seen.append(mode)
        state["dev"] = DEV
        return DEV

    state = {"dev": audio_dev}
    monkeypatch.setattr(release_gate, "discover", lambda: [state["dev"]])
    monkeypatch.setattr(release_gate, "set_mode", fake_set_mode)
    rep = release_gate.SCENARIO.run(make_ctx(tmp_path), release_gate.Params(), base.Progress())
    assert rep.passed
    assert seen == [UsbMode.DEFAULT]  # once, before smoke; never again once it is default


# ── stall watch ──────────────────────────────────────────────────────────────
def test_stall_watch_dumps_every_thread_when_the_gate_stops_reporting(
    tmp_path: Path,
) -> None:
    lines: list[str] = []
    watch = release_gate._StallWatch(tmp_path, lines.append)
    watch.arm("usb_mode_cycle", 0.05)
    deadline = time.monotonic() + 5.0
    dump = tmp_path / "stall_usb_mode_cycle.txt"
    # _fire writes the dump, then logs: wait for both, or the log line can lose the race.
    while (not dump.exists() or not any("STALL" in line for line in lines)) \
            and time.monotonic() < deadline:
        time.sleep(0.02)
    watch.disarm()
    assert dump.exists()
    text = dump.read_text()
    assert "step=usb_mode_cycle" in text
    assert "Thread" in text or "Current thread" in text  # faulthandler's per-thread header
    assert any("STALL" in line for line in lines)


def test_stall_watch_disarmed_in_time_leaves_nothing(tmp_path: Path) -> None:
    watch = release_gate._StallWatch(tmp_path, lambda s: None)
    watch.arm("smoke", 0.2)
    watch.disarm()
    time.sleep(0.4)
    assert not list(tmp_path.glob("stall_*.txt"))


# ── a step that dies mid-way keeps what it had measured ──────────────────────


def partial_then_hang(ctx: base.Context, params: Any, progress: base.Progress) -> base.Report:
    ctx.partial = {"rounds_done": 0}
    ctx.partial["rounds_done"] = 4          # by reference, after the assignment
    time.sleep(3600)
    return base.Report(passed=True, summary="unreachable", data={}, artifacts=[], exit_code=0)


def partial_then_raise(ctx: base.Context, params: Any, progress: base.Progress) -> base.Report:
    from crosspad_hil.errors import HilError

    ctx.partial = {"rounds_done": 4}
    raise HilError("TIMEOUT", "no MEM:/ERR reply to 'MEM' within 2s")


def test_timed_out_step_keeps_what_it_had_measured(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An abandoned step's numbers are the only evidence there will ever be."""
    install_fakes(monkeypatch, {"kit_churn": partial_then_hang})
    monkeypatch.setattr(release_gate, "discover", lambda: [DEV])
    release_gate.SCENARIO.run(make_ctx(tmp_path), release_gate.Params(), base.Progress())

    doc = json.loads((tmp_path / "kit_churn" / "report.json").read_text())
    assert doc["data"]["timeout"] is True
    assert doc["data"]["rounds_done"] == 4


def test_failed_step_keeps_what_it_had_measured(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    install_fakes(monkeypatch, {"kit_churn": partial_then_raise})
    monkeypatch.setattr(release_gate, "discover", lambda: [DEV])
    release_gate.SCENARIO.run(make_ctx(tmp_path), release_gate.Params(), base.Progress())

    doc = json.loads((tmp_path / "kit_churn" / "report.json").read_text())
    assert doc["data"]["rounds_done"] == 4
    assert doc["data"]["error"]["code"] == "TIMEOUT"


# ── what each step left on the board ─────────────────────────────────────────

CLEAN = {"ble_running": False, "ble_mode": "server", "usb_guard": False, "wifi_enabled": False,
         "audio_route": [1, 1, 1], "ingain": {"IN1": 24, "IN2": 0}, "app": "launcher", "kit": 2,
         "int_free": 120000, "psram_free": 4000000}


def stage_states(monkeypatch: pytest.MonkeyPatch, leaves: dict[str, dict]) -> None:
    """Every step reads CLEAN before; the step named in `leaves` reads CLEAN + that after
    (and the board stays like that for the steps after it, as a real leftover does)."""
    board = dict(CLEAN)
    calls: dict[str, int] = {}

    def read_state(step_ctx: base.Context) -> dict:
        step = step_ctx.workdir.name
        calls[step] = calls.get(step, 0) + 1
        if calls[step] == 2 and step in leaves:      # the reading after the step
            board.update(leaves[step])
        return dict(board)

    monkeypatch.setattr(release_gate, "_read_state", read_state)


def test_a_step_that_leaves_ble_up_is_named_as_a_warning(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """ble_midi once ended every run with BLE up: 63 kB internal free, quiet pads
    four hours later. The gate names the step that left it."""
    install_fakes(monkeypatch, {})
    monkeypatch.setattr(release_gate, "discover", lambda: [DEV])
    stage_states(monkeypatch, {"ble_midi": {"ble_running": True, "int_free": 63000,
                                            "app": "Sampler"}})
    rep = release_gate.SCENARIO.run(make_ctx(tmp_path), release_gate.Params(), base.Progress())
    assert rep.passed and rep.exit_code == 0
    steps = {s["step"]: s for s in rep.data["steps"]}
    assert steps["ble_midi"]["state_left"] == [
        {"key": "ble_running", "before": False, "after": True, "persistent": True}]
    assert [c["key"] for c in steps["ble_midi"]["state_changes"]] == ["ble_running", "app"]
    assert "WARN ble_midi: left the board changed: ble_running False -> True" in rep.summary
    # the steps after it found it that way and changed nothing themselves
    assert steps["audio_loopback"]["state_left"] == []


def test_strict_state_makes_the_leftover_a_fail(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    install_fakes(monkeypatch, {})
    monkeypatch.setattr(release_gate, "discover", lambda: [DEV])
    stage_states(monkeypatch, {"back_churn": {"usb_guard": True}})
    rep = release_gate.SCENARIO.run(make_ctx(tmp_path), release_gate.Params(strict_state=True),
                                    base.Progress())
    assert not rep.passed and rep.exit_code == 1
    steps = {s["step"]: s for s in rep.data["steps"]}
    assert steps["back_churn"]["result"] == "FAIL"
    assert steps["back_churn"]["summary"] == "left the board changed: usb_guard False -> True"
    assert rep.data["stopped_at"] == "back_churn"


def test_an_informational_change_is_not_a_warning(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    install_fakes(monkeypatch, {})
    monkeypatch.setattr(release_gate, "discover", lambda: [DEV])
    stage_states(monkeypatch, {"kit_churn": {"kit": 5, "app": "Sampler"}})
    rep = release_gate.SCENARIO.run(make_ctx(tmp_path), release_gate.Params(strict_state=True),
                                    base.Progress())
    assert rep.passed, rep.summary
    assert "WARN" not in rep.summary
