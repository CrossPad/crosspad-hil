"""Scenario `led_state` against a scripted CDC link (no hardware)."""
from __future__ import annotations

import threading
from pathlib import Path
from typing import Any

import pytest

from crosspad_hil.devices import Device, Ports, SerialPortInfo, UsbMode
from crosspad_hil.scenarios import base, get, led_state
from tests.fakes import FakeSerial

CDC_PATH = "/dev/ttyACM0"
CONSOLE_PATH = "/dev/ttyACM1"

# Reply format from platform-idf/main/hil_control.cpp handle_led_state():
# "LEDS: bri=%u anim=%d coalesce=%d cfgbri=%u pwr=0x%02X pwrN=%u txfail=%u colors=…"
COLORS = ["FF0000", "00FF00", "0000FF", "FFFFFF",
          "101010", "202020", "303030", "404040",
          "505050", "606060", "707070", "808080",
          "AA0000", "00AA00", "0000AA", "AAAAAA"]
LEDS_REPLY = ("LEDS: bri=200 anim=0 coalesce=1 cfgbri=200 pwr=0x03 pwrN=2 txfail=0 "
              "colors=" + ",".join(COLORS))
BLACK_REPLY = ("LEDS: bri=0 anim=1 coalesce=0 cfgbri=200 pwr=0x04 pwrN=5 txfail=3 "
               "colors=" + ",".join(["000000"] * 16))


def make_device() -> Device:
    return Device(
        id="dev_ab12", serial="CP-1", usb_mode=UsbMode.DEFAULT,
        ports=Ports(
            cdc=SerialPortInfo(path=CDC_PATH, vid=0x303A, pid=0x3456,
                               serial="CP-1", product="Crosspad", location="1-2.1"),
            console=SerialPortInfo(path=CONSOLE_PATH, vid=0x0483, pid=0x5740,
                                   serial="STM-1", product="CrossPad MIDI+Serial",
                                   location="1-2.2"),
        ),
    )


def make_ctx(tmp_path: Path, reply: str) -> tuple[base.Context, FakeSerial]:
    fake = FakeSerial(script=[("LED_STATE", reply)], timeout=0.01)

    def factory(path: str, **kw: Any) -> FakeSerial:
        assert path == CDC_PATH, path
        return fake

    ctx = base.Context(device=make_device(), workdir=tmp_path,
                       cancelled=threading.Event(), log=lambda s: None,
                       serial_factory=factory)
    return ctx, fake


@pytest.fixture(autouse=True)
def lock_dir(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv("XDG_RUNTIME_DIR", str(tmp_path / "run"))


def test_registered() -> None:
    assert get("led_state") is led_state.SCENARIO
    assert led_state.SCENARIO.name == "led_state"
    assert led_state.Params().watch is False


def test_single_query_renders_grid(tmp_path: Path) -> None:
    ctx, fake = make_ctx(tmp_path, LEDS_REPLY)
    report = led_state.SCENARIO.run(ctx, led_state.Params(), base.Progress())
    assert report.passed is True
    assert report.exit_code == 0
    assert report.data["samples"] == 1
    assert report.data["all_black"] is False
    assert report.data["leds"]["brightness"] == 200
    assert report.data["leds"]["colors"] == COLORS
    rows = [ln for ln in report.summary.splitlines() if ln.strip().startswith("12:")]
    assert rows, report.summary
    # Top row is pads 12..15, as they sit under your hands.
    assert rows[0].split() == ["12:AA0000", "13:00AA00", "14:0000AA", "15:AAAAAA"]
    bottom = [ln for ln in report.summary.splitlines() if ln.strip().startswith("0:")]
    assert bottom[0].split() == ["0:FF0000", "1:00FF00", "2:0000FF", "3:FFFFFF"]
    assert "brightness 200   settings 200   animating 0   coalesced 1" in report.summary
    assert "last power state 0x03   notifications 2" in report.summary
    assert "never reached the strip" not in report.summary
    assert "every pad is black" not in report.summary
    assert fake.written == ["LED_STATE"]
    assert report.artifacts == []


def test_black_model_is_called_out(tmp_path: Path) -> None:
    ctx, _fake = make_ctx(tmp_path, BLACK_REPLY)
    report = led_state.SCENARIO.run(ctx, led_state.Params(), base.Progress())
    assert report.passed is True
    assert report.data["all_black"] is True
    assert "every pad is black in the model — nothing repainted them" in report.summary
    assert "3 frame(s) never reached the strip" in report.summary
    assert "last power state 0x04   notifications 5" in report.summary


def test_render_grid_pads_missing_colors() -> None:
    state = {"brightness": 1, "anim": False, "coalesce": True, "cfgbri": 1, "pwr": 0,
             "pwr_count": 0, "txfail": 0, "colors": ["112233"]}
    text = led_state.render_grid(state)
    assert " 0:112233" in text
    assert "15:??????" in text
    assert "animating 0   coalesced 1" in text
    assert "every pad is black" not in text      # one non-black pad known


def test_watch_stops_on_cancel(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(led_state, "WATCH_PERIOD_S", 0.05)
    ctx, fake = make_ctx(tmp_path, LEDS_REPLY)
    seen: list[tuple[int, int | None, str]] = []

    class Counting(base.Progress):
        def __call__(self, progress: int, total: int | None, message: str) -> None:
            seen.append((progress, total, message))
            if progress >= 3:
                ctx.cancelled.set()

    report = led_state.SCENARIO.run(ctx, led_state.Params(watch=True), Counting())
    assert report.passed is True                  # a cancelled watch is not a failure
    assert report.exit_code == 0
    assert report.data["samples"] == 3
    assert fake.written == ["LED_STATE"] * 3
    assert [p for p, _t, _m in seen] == [1, 2, 3]
    assert all(t is None for _p, t, _m in seen)
    assert "12:AA0000" in seen[0][2]
