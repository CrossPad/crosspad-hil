"""Scenario `smoke` against a scripted console (no hardware)."""
from __future__ import annotations

import threading
from pathlib import Path
from typing import Any

import pytest

from crosspad_hil.devices import Device, Ports, SerialPortInfo, UsbMode
from crosspad_hil.scenarios import base, get, smoke
from tests.fakes import FakeSerial

CDC_PATH = "/dev/ttyACM0"
CONSOLE_PATH = "/dev/ttyACM1"

# A healthy boot as the STM VCP shows it — the seven REQUIRED_MARKERS of
# hil_smoke.py in the order the firmware prints them, plus one allow-listed
# E-line (a missing STARTUP.wav is cosmetic: ERROR_ALLOWLIST r"file not found").
HEALTHY_BOOT = [
    "ESP-ROM:esp32s3-20210327",
    "I (312) main_task: Started on CPU0",
    "I (900) CrosspadPlatform: Platform fully initialized",
    "I (950) STM32: STM32 ident: r20 fw 1.4",
    "E (1000) SD: file not found: STARTUP.wav",
    "I (1200) main: Crosspad initialization complete",
    "I (1210) main: All systems operational",
    "I (1500) display: LVGL setup done successfully",
    "I (1600) gui: App registry initialized",
    "I (1700) gui: LoadMainScreen completed successfully",
]


class BootingSerial(FakeSerial):
    """STM VCP stand-in: the esptool pulse (RTS high then low) makes the ESP boot.

    Lines are delivered 0.3 s after RTS falls, i.e. after Console.reset() has
    finished `reset_pulse()` (0.1 s) and `parser.reset_boot_tracking()`, so the
    boot is counted in the fresh boot window exactly as on hardware.
    """

    def __init__(self, boot_lines: list[str]) -> None:
        super().__init__(timeout=0.01)
        self.boot_lines = list(boot_lines)
        self.boots = 0

    @property
    def rts(self) -> bool:
        return FakeSerial.rts.fget(self)  # type: ignore[attr-defined]

    @rts.setter
    def rts(self, value: bool) -> None:
        was = FakeSerial.rts.fget(self)  # type: ignore[attr-defined]
        FakeSerial.rts.fset(self, value)  # type: ignore[attr-defined]
        if was and not value:
            self.boots += 1
            threading.Timer(0.3, self.feed, [list(self.boot_lines)]).start()


def make_device() -> Device:
    return Device(
        id="dev_ab12",
        serial="CP-1",
        usb_mode=UsbMode.DEFAULT,
        ports=Ports(
            cdc=SerialPortInfo(path=CDC_PATH, vid=0x303A, pid=0x3456,
                               serial="CP-1", product="Crosspad", location="1-2.1"),
            console=SerialPortInfo(path=CONSOLE_PATH, vid=0x0483, pid=0x5740,
                                   serial="STM-1", product="CrossPad MIDI+Serial",
                                   location="1-2.2"),
        ),
    )


def make_ctx(tmp_path: Path, lines: list[str]) -> tuple[base.Context, BootingSerial]:
    fake = BootingSerial(lines)
    opened: list[tuple[str, Any]] = []

    def factory(path: str, **kw: Any) -> FakeSerial:
        opened.append((path, kw.get("reset")))
        assert path == CONSOLE_PATH, path
        return fake

    ctx = base.Context(device=make_device(), workdir=tmp_path,
                       cancelled=threading.Event(), log=lambda s: None,
                       serial_factory=factory)
    return ctx, fake


@pytest.fixture(autouse=True)
def fast_settle(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(smoke, "SETTLE_S", 0.2)


@pytest.fixture(autouse=True)
def lock_dir(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv("XDG_RUNTIME_DIR", str(tmp_path / "run"))


def test_registered() -> None:
    assert get("smoke") is smoke.SCENARIO
    assert smoke.SCENARIO.name == "smoke"
    assert smoke.Params().timeout == 25
    assert smoke.Params().flash is None


def test_healthy_boot_passes(tmp_path: Path) -> None:
    ctx, fake = make_ctx(tmp_path, HEALTHY_BOOT)
    report = smoke.SCENARIO.run(ctx, smoke.Params(timeout=5), base.Progress())
    assert report.passed is True
    assert report.exit_code == 0
    assert report.data["pass"] is True
    assert report.data["missing"] == []
    assert report.data["errors"] == []          # "file not found" is allow-listed
    assert report.data["fatal"] == []
    assert report.data["bootloops"] == 0
    assert "SD Card mounted successfully" in report.data["optional_missing"]
    assert {"pass", "missing", "optional_missing", "errors", "fatal", "bootloops",
            "seconds", "port"} <= set(report.data)
    assert report.data["port"] == CONSOLE_PATH
    assert [a.role for a in report.artifacts] == ["console"]
    assert report.artifacts[0].path == str(tmp_path / "console.log")
    assert (tmp_path / "console.log").read_text(encoding="utf-8").count("\n") >= 10
    assert "HIL SMOKE: PASS" in report.summary
    # The device was reset once, esptool-style, and RTS was released afterwards.
    assert fake.boots == 1
    rts = [v for attr, v in fake.control_history if attr == "rts"]
    assert True in rts and rts[-1] is False
    assert all(v is False for attr, v in fake.control_history if attr == "dtr")


def test_missing_marker_fails(tmp_path: Path) -> None:
    lines = [ln for ln in HEALTHY_BOOT if "STM32 ident:" not in ln]
    ctx, _fake = make_ctx(tmp_path, lines)
    report = smoke.SCENARIO.run(ctx, smoke.Params(timeout=2), base.Progress())
    assert report.passed is False
    assert report.exit_code == 1
    assert report.data["pass"] is False
    assert report.data["missing"] == ["STM32 ident:"]
    assert "MISSING MARKER: STM32 ident:" in report.summary
    assert "HIL SMOKE: FAIL" in report.summary


def test_fatal_fails(tmp_path: Path) -> None:
    lines = list(HEALTHY_BOOT[:4])
    lines.append("Guru Meditation Error: Core  0 panic'ed (LoadProhibited).")
    ctx, _fake = make_ctx(tmp_path, lines)
    report = smoke.SCENARIO.run(ctx, smoke.Params(timeout=2), base.Progress())
    assert report.passed is False
    assert report.exit_code == 1
    assert report.data["fatal"] == ["Guru Meditation Error: Core  0 panic'ed (LoadProhibited)."]
    assert "FATAL: Guru Meditation" in report.summary
    assert len(report.data["missing"]) == 5      # everything after the panic is gone


def test_unallowed_error_line_fails(tmp_path: Path) -> None:
    lines = list(HEALTHY_BOOT)
    lines.insert(5, "E (1100) i2c: timeout waiting for ack")
    ctx, _fake = make_ctx(tmp_path, lines)
    report = smoke.SCENARIO.run(ctx, smoke.Params(timeout=5), base.Progress())
    assert report.passed is False
    assert report.exit_code == 1
    assert report.data["errors"] == ["E (1100) i2c: timeout waiting for ack"]
    assert "ERROR LINE: E (1100) i2c: timeout waiting for ack" in report.summary


def test_boot_loop_fails(tmp_path: Path) -> None:
    # hil_smoke.evaluate: boots = max(count per BOOT_MARKER); bootloops = boots - 1
    lines = list(HEALTHY_BOOT[:2]) + list(HEALTHY_BOOT)
    ctx, _fake = make_ctx(tmp_path, lines)
    report = smoke.SCENARIO.run(ctx, smoke.Params(timeout=5), base.Progress())
    assert report.passed is False
    assert report.data["bootloops"] == 1
    assert "BOOT LOOP: device reset 1 extra time(s)" in report.summary


def test_no_banner_is_env_error(tmp_path: Path) -> None:
    ctx, _fake = make_ctx(tmp_path, [])
    report = smoke.SCENARIO.run(ctx, smoke.Params(timeout=1), base.Progress())
    assert report.passed is False
    assert report.exit_code == 2
    assert report.data["pass"] is False
    assert "No boot banner" in report.summary


def test_no_flash_when_param_is_none(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[tuple[Any, ...]] = []
    monkeypatch.setattr(smoke, "flash", lambda *a, **k: calls.append((a, k)))
    ctx, _fake = make_ctx(tmp_path, HEALTHY_BOOT)
    smoke.SCENARIO.run(ctx, smoke.Params(timeout=5), base.Progress())
    assert calls == []


def test_flash_called_with_firmware(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[tuple[str, Path, bool | None]] = []

    def fake_flash(device: Device, firmware: Path, **kw: Any) -> dict[str, Any]:
        calls.append((device.id, Path(firmware), kw.get("wait_boot")))
        return {"flash": {"bytes": 1, "seconds": 0.1, "kbps": 1.0,
                          "version": "x", "mode": "full"}, "boot": None}

    monkeypatch.setattr(smoke, "flash", fake_flash)
    monkeypatch.setattr(smoke, "POST_FLASH_S", 0.0)
    ctx, _fake = make_ctx(tmp_path, HEALTHY_BOOT)
    report = smoke.SCENARIO.run(
        ctx, smoke.Params(flash="build/CrossPad.bin", timeout=5), base.Progress())
    assert calls == [("dev_ab12", Path("build/CrossPad.bin"), False)]
    assert report.passed is True
    assert report.data["flash"]["mode"] == "full"


def test_flash_failure_is_env_error(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from crosspad_hil.errors import FLASH_FAILED, HilError

    def failing_flash(device: Device, firmware: Path, **kw: Any) -> dict[str, Any]:
        raise HilError(FLASH_FAILED, "OTA_ERROR 3", hint="re-run")

    monkeypatch.setattr(smoke, "flash", failing_flash)
    ctx, fake = make_ctx(tmp_path, HEALTHY_BOOT)
    report = smoke.SCENARIO.run(
        ctx, smoke.Params(flash="build/CrossPad.bin", timeout=5), base.Progress())
    assert report.passed is False
    assert report.exit_code == 2
    assert report.summary == "FLASH FAILED: OTA_ERROR 3"
    assert report.data["flash_error"]["code"] == FLASH_FAILED
    assert fake.boots == 0                        # never got as far as the console
