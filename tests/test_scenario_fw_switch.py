"""Scenario `fw_switch`'s decision logic, against a scripted FW_* fake. No hardware.

`FwDevice` stands in for main/fw_slots.cpp + main/fw_library.cpp closely enough
to drive the scenario: FW_SWITCH flips the running slot and starts it
"trial", a configurable number of later FW_SLOTS polls turn it "confirmed"
(the 10-16 s confirm timer on real hardware), and FW_INSTALL can be made to
take a few bare polls before landing on "done" or "failed".
"""
from __future__ import annotations

import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest

from crosspad_hil.cdc import CdcLink
from crosspad_hil.devices import Device, Ports, SerialPortInfo, UsbMode
from crosspad_hil.errors import HilError
from crosspad_hil.scenarios import base, fw_switch, get
from tests.fakes import FakeSerial


def make_device() -> Device:
    return Device(
        id="dev_ab12", serial="CP-1", usb_mode=UsbMode.DEFAULT,
        ports=Ports(
            cdc=SerialPortInfo(path="/dev/ttyACM0", vid=0x303A, pid=0x3456,
                               serial="CP-1", product="CrossPad", location="1-2.1"),
            console=SerialPortInfo(path="/dev/ttyACM1", vid=0x0483, pid=0x5740,
                                   serial="STM-1", product="CrossPad", location="1-2.2"),
        ),
    )


class FwDevice(FakeSerial):
    def __init__(self, *, running: str = "A", sha: str = "a" * 64,
                confirm_after_polls: int = 2, rollback: bool = True,
                running_state: str = "confirmed") -> None:
        super().__init__()
        self.slots: dict[str, dict[str, Any]] = {
            "A": {"running": running == "A",
                  "state": running_state if running == "A" else "empty",
                  "pkg": "bench-current" if running == "A" else None,
                  "ver": "1.0.0" if running == "A" else None,
                  "kit": "full" if running == "A" else None},
            "B": {"running": running == "B",
                  "state": running_state if running == "B" else "empty",
                  "pkg": "bench-current" if running == "B" else None,
                  "ver": "1.0.0" if running == "B" else None,
                  "kit": "full" if running == "B" else None},
        }
        self.sha = sha
        self.rollback = rollback
        self.confirm_after_polls = confirm_after_polls
        self._trial_polls: dict[str, int] = {}
        self.packages: list[dict[str, Any]] = [
            {"name": "bench-current", "kit": "full", "ver": "1.0.0", "rev": 1,
             "size": 2_300_000, "assets": 1, "settings": 0, "verdict": "ok"},
        ]
        self.install_refuse: str | None = None
        self.install_warn = False
        self.install_steps = 0            # bare polls reporting "writing" before landing
        self.install_result = "done"      # "done" | "failed"
        self.install_fail_reason = "checksum"
        self._install_phase = "idle"
        self._install_percent = 0
        self._install_error = "-"
        self._install_pending: tuple[str, dict[str, Any]] | None = None
        self.switch_refusal: str | None = None
        self.switches = 0

    def _other(self, label: str) -> str:
        return "B" if label == "A" else "A"

    def _running_label(self) -> str:
        return next(label for label, s in self.slots.items() if s["running"])

    def _slots_lines(self) -> list[str]:
        lines = []
        for label in ("A", "B"):
            s = self.slots[label]
            lines.append(
                f"FWSLOT: slot={label} running={1 if s['running'] else 0} "
                f"state={s['state']} pkg={s['pkg'] or '-'} ver={s['ver'] or '-'} "
                f"kit={s['kit'] or '-'}")
        lines.append(f"FWSLOT: rollback={1 if self.rollback else 0}")
        return lines

    def _fwpkg_lines(self) -> list[str]:
        lines = [
            f"FWPKG: name={p['name']} kit={p['kit']} ver={p['ver']} rev={p['rev']} "
            f"size={p['size']} assets={p['assets']} settings={p['settings']} "
            f"verdict={p['verdict']}"
            for p in self.packages
        ]
        lines.append(f"FWPKG: count={len(self.packages)}")
        return lines

    def _apply_install_result(self) -> None:
        if self.install_result == "done" and self._install_pending is not None:
            other, pkg = self._install_pending
            self.slots[other].update(pkg=pkg["name"], ver=pkg["ver"], kit=pkg["kit"],
                                     state="installed")
            self._install_percent = 100
        elif self.install_result == "failed":
            self._install_error = self.install_fail_reason

    def write(self, data: bytes) -> int:
        for raw in data.decode("utf-8", "replace").split("\n"):
            cmd = raw.strip()
            if cmd:
                self.written.append(cmd)
                self._handle(cmd)
        return len(data)

    def _handle(self, cmd: str) -> None:
        if cmd == "APP_LIST":
            self.feed(["APPS: Firmware,Sampler running=-"])
        elif cmd == "FW_SLOTS":
            running = self._running_label()
            s = self.slots[running]
            if s["state"] == "trial":
                left = self._trial_polls.get(running, self.confirm_after_polls) - 1
                if left <= 0:
                    s["state"] = "confirmed"
                    self._trial_polls.pop(running, None)
                else:
                    self._trial_polls[running] = left
            self.feed(self._slots_lines())
        elif cmd == "APP_SHA":
            self.feed([f"APPSHA: {self.sha}"])
        elif cmd == "FW_LIST":
            self.feed(self._fwpkg_lines())
        elif cmd == "FW_SWITCH":
            self.switches += 1
            if self.switch_refusal:
                self.feed([f"ERR {self.switch_refusal}"])
                return
            running = self._running_label()
            other = self._other(running)
            self.feed(["OK reboot=1"])
            self.slots[running]["running"] = False
            self.slots[other]["running"] = True
            if self.slots[other]["state"] != "empty":
                self.slots[other]["state"] = "trial"
            self._trial_polls[other] = self.confirm_after_polls
        elif cmd.startswith("FW_INSTALL"):
            rest = cmd[len("FW_INSTALL"):].strip()
            if not rest:
                if self._install_phase == "writing":
                    self.install_steps -= 1
                    self._install_percent = min(90, self._install_percent + 30)
                    if self.install_steps <= 0:
                        self._install_phase = self.install_result
                        self._apply_install_result()
                self.feed([f"FWINST: phase={self._install_phase} "
                          f"percent={self._install_percent} error={self._install_error}"])
                return
            name = rest.split()[0]
            if self.install_refuse:
                self.feed([f"ERR {self.install_refuse}"])
                return
            pkg = next((p for p in self.packages if p["name"] == name), None)
            if pkg is None:
                self.feed(["ERR no such package"])
                return
            other = self._other(self._running_label())
            self._install_pending = (other, pkg)
            self._install_percent = 0
            self._install_error = "-"
            if self.install_steps > 0:
                self._install_phase = "writing"
            else:
                self._install_phase = self.install_result
                self._apply_install_result()
            self.feed(["OK warn=stm_proto_older" if self.install_warn else "OK"])
        elif cmd == "FW_CONFIRM":
            self.feed(["OK"])
        else:
            self._rx.append((cmd + "\r\n").encode())


@dataclass
class FakeContext(base.Context):
    fw_fake: FwDevice = None  # type: ignore[assignment]
    opens: int = 0

    def open_cdc(self) -> CdcLink:
        self.opens += 1
        link = CdcLink("/dev/ttyACM0", serial_factory=lambda path, **kw: self.fw_fake)
        link.open()
        self.opened.append(link)
        return link


def make_ctx(tmp_path: Path, fw: FwDevice) -> FakeContext:
    return FakeContext(device=make_device(), workdir=tmp_path, cancelled=threading.Event(),
                       log=lambda s: None, fw_fake=fw)


@pytest.fixture(autouse=True)
def fast_polling(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(fw_switch, "POLL_S", 0.0)
    monkeypatch.setattr(fw_switch, "INSTALL_POLL_S", 0.0)
    monkeypatch.setattr(fw_switch, "REBOOT_SETTLE_S", 0.0)


def run(ctx: FakeContext, **params: object) -> base.Report:
    return fw_switch.SCENARIO.run(ctx, fw_switch.Params(**params), base.Progress())  # type: ignore[arg-type]


def test_registered_and_defaults() -> None:
    assert get("fw_switch") is fw_switch.SCENARIO
    p = fw_switch.Params()
    assert (p.package, p.install_timeout, p.confirm_timeout, p.list_timeout) == \
        (None, 120.0, 30.0, 15.0)


def test_running_slot_not_confirmed_is_a_bench_problem(tmp_path: Path) -> None:
    fw = FwDevice(running_state="trial")
    ctx = make_ctx(tmp_path, fw)
    rep = run(ctx)
    assert not rep.passed and rep.exit_code == 2
    assert "not confirmed" in rep.summary


def test_no_rollback_support_is_a_bench_problem(tmp_path: Path) -> None:
    fw = FwDevice(rollback=False)
    ctx = make_ctx(tmp_path, fw)
    rep = run(ctx)
    assert not rep.passed and rep.exit_code == 2
    assert "rollback" in rep.summary


def test_no_matching_package_is_a_bench_problem(tmp_path: Path) -> None:
    fw = FwDevice()
    fw.packages[0]["ver"] = "0.1.0"          # nothing matches the running 1.0.0
    ctx = make_ctx(tmp_path, fw)
    rep = run(ctx)
    assert not rep.passed and rep.exit_code == 2
    assert "make_fw_package.py" in rep.summary
    assert "fs_transfer.py push" in rep.summary


def test_explicit_package_not_on_card_is_a_bench_problem(tmp_path: Path) -> None:
    fw = FwDevice()
    ctx = make_ctx(tmp_path, fw)
    rep = run(ctx, package="does-not-exist")
    assert not rep.passed and rep.exit_code == 2
    assert "does-not-exist" in rep.summary
    assert "bench-current" in rep.summary


def test_full_round_trip_passes(tmp_path: Path) -> None:
    fw = FwDevice(confirm_after_polls=1)
    ctx = make_ctx(tmp_path, fw)
    rep = run(ctx)
    assert rep.passed, rep.summary
    assert rep.exit_code == 0
    assert rep.data["label_before"] == "A"
    assert rep.data["label_after"] == "B"
    assert rep.data["package"] == "bench-current"
    assert rep.data["sha_before"] == rep.data["sha_after"] == fw.sha
    assert fw.switches == 2
    # ended back on the original slot, confirmed
    assert fw.slots["A"]["running"] is True and fw.slots["A"]["state"] == "confirmed"
    for phase in ("preflight", "fw_list", "install", "switch1_reboot", "confirm1",
                  "verify", "switch2_reboot", "confirm2"):
        assert phase in rep.data["phases_s"], phase


def test_explicit_package_is_honoured(tmp_path: Path) -> None:
    fw = FwDevice(confirm_after_polls=1)
    fw.packages.append({"name": "other-build", "kit": "full", "ver": "9.9.9", "rev": 1,
                        "size": 1, "assets": 0, "settings": 0, "verdict": "ok"})
    ctx = make_ctx(tmp_path, fw)
    rep = run(ctx, package="other-build")
    assert rep.passed, rep.summary
    assert rep.data["package"] == "other-build"


def test_install_failure_fails_the_run(tmp_path: Path) -> None:
    fw = FwDevice()
    fw.install_result = "failed"
    fw.install_fail_reason = "checksum"
    ctx = make_ctx(tmp_path, fw)
    rep = run(ctx)
    assert not rep.passed and rep.exit_code == 1
    assert "checksum" in rep.summary
    assert fw.switches == 0     # never got as far as switching


def test_install_timeout_fails_the_run(tmp_path: Path) -> None:
    fw = FwDevice()
    fw.install_steps = 10**9    # never reaches done/failed within the budget
    ctx = make_ctx(tmp_path, fw)
    rep = run(ctx, install_timeout=0.05)
    assert not rep.passed and rep.exit_code == 1
    assert "still writing" in rep.summary or "FW_INSTALL" in rep.summary


def test_switch_refused_raises(tmp_path: Path) -> None:
    fw = FwDevice()
    fw.switch_refusal = "running slot not confirmed yet"
    ctx = make_ctx(tmp_path, fw)
    with pytest.raises(HilError) as ei:
        run(ctx)
    assert ei.value.code == "NOT_SUPPORTED"
    assert "not confirmed" in ei.value.message


def test_confirm_timeout_fails_the_run(tmp_path: Path) -> None:
    # A trial that never confirms within the budget: label changed but the
    # state never reaches "confirmed".
    fw = FwDevice(confirm_after_polls=10**9)
    ctx = make_ctx(tmp_path, fw)
    rep = run(ctx, confirm_timeout=0.05)
    assert not rep.passed and rep.exit_code == 1
    assert "confirmed" in rep.summary
    assert rep.data["label_after"] == "B"


def test_sha_mismatch_fails_the_run(tmp_path: Path) -> None:
    fw = FwDevice(confirm_after_polls=1)
    ctx = make_ctx(tmp_path, fw)

    real_handle = fw._handle
    calls = {"n": 0}

    def flaky_sha(cmd: str) -> None:
        if cmd == "APP_SHA":
            calls["n"] += 1
            if calls["n"] == 2:
                fw.feed([f"APPSHA: {'f' * 64}"])
                return
        real_handle(cmd)

    fw._handle = flaky_sha  # type: ignore[assignment]
    rep = run(ctx)
    assert not rep.passed and rep.exit_code == 1
    assert "APP_SHA changed" in rep.summary
