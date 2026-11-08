"""Scenario `daw_control` against a fake CDC device + a fake ESP MIDI port.

Two fakes, one for each side of the protocol: `DawDevice` (a `FakeSerial`)
answers APP_LIST/APP_START/APP_STOP/LED_STATE the way `hil_control.cpp` does
(pattern from `tests/scenarios/test_pitched_kit.py` and
`test_cli.py::test_led_state_grid`'s LED_STATE line shape); `DawMidiBus`
answers the SysEx protocol on the ESP MIDI port — hello, and pad press/release
turning into NoteOn/NoteOff — the way the DAW Control app would.

`daw_control.MidiIO` is monkeypatched to hand out the fake port (same trick
`test_velocity.py`'s `EchoBus`/`FakeMidi` uses), so nothing here opens a real
rtmidi port.
"""
from __future__ import annotations

import threading
from collections import deque
from dataclasses import dataclass
from pathlib import Path

import pytest

from crosspad_hil.cdc import CdcLink
from crosspad_hil.devices import Device, MidiPortInfo, Ports, SerialPortInfo, UsbMode
from crosspad_hil.errors import ENV, HilError
from crosspad_hil.midi import MidiRole
from crosspad_hil.scenarios import base, daw_control, get
from tests.fakes import FakeSerial

CDC = SerialPortInfo(path="/dev/ttyACM0", vid=0x303A, pid=0x3456, serial="CP-1",
                     product="CrossPad", location="1-2.1")
ESP_MIDI = MidiPortInfo(name="Crosspad MIDI 1", rtmidi_out=0, rtmidi_in=0,
                        alsa_hw="hw:4,0,0", rawmidi="/dev/snd/midiC4D0")


def make_device(*, esp_midi: MidiPortInfo | None = ESP_MIDI) -> Device:
    return Device(id="dev_ab12", serial="CP-1", usb_mode=UsbMode.DEFAULT,
                  ports=Ports(cdc=CDC, esp_midi=esp_midi))


# --------------------------------------------------------------------------- CDC fake


class DawDevice(FakeSerial):
    """APP_LIST / APP_START / APP_STOP / LED_STATE, hil_control.cpp-shaped."""

    def __init__(self) -> None:
        super().__init__()
        self.apps = ["Sampler", daw_control.APP_NAME]
        self.running: str | None = "Sampler"
        self.led_colors = ["000000"] * 16

    def write(self, data: bytes) -> int:
        for raw in data.decode("utf-8", "replace").split("\n"):
            cmd = raw.strip()
            if not cmd:
                continue
            self.written.append(cmd)
            if cmd == "APP_LIST":
                self.feed([f"APPS: {','.join(self.apps)} running={self.running or '-'}"])
            elif cmd.startswith("APP_START "):
                self.running = cmd.split(" ", 1)[1]
                self.feed(["OK"])
            elif cmd == "APP_STOP":
                self.running = None
                self.feed(["OK"])
            elif cmd == "LED_STATE":
                colors = ",".join(self.led_colors)
                self.feed([f"LEDS: bri=80 anim=0 coalesce=0 cfgbri=80 pwr=0x00 pwrN=1 "
                           f"txfail=0 colors={colors}"])
            else:
                self._rx.append((cmd + "\r\n").encode())
        return len(data)


@dataclass
class FakeContext(base.Context):
    cdc_fake: DawDevice = None  # type: ignore[assignment]

    def open_cdc(self) -> CdcLink:
        fake = self.cdc_fake
        link = CdcLink(self.device.ports.cdc.path, serial_factory=lambda path, **kw: fake)
        link.open()
        self.opened.append(link)
        return link


def make_ctx(tmp_path: Path, *, esp_midi: MidiPortInfo | None = ESP_MIDI) -> tuple[FakeContext, DawDevice]:
    dev = DawDevice()
    ctx = FakeContext(device=make_device(esp_midi=esp_midi), workdir=tmp_path,
                      cancelled=threading.Event(), log=lambda s: None, cdc_fake=dev)
    return ctx, dev


# --------------------------------------------------------------------------- MIDI fake


class FakeMidi:
    """`MidiIO`'s shape the scenario uses: send_sysex/receive/receive_note/drain."""

    def __init__(self, bus: DawMidiBus, role: MidiRole) -> None:
        self._bus = bus
        self.role = role
        self.sysex_q: deque[bytes] = deque()
        self.note_q: deque[list[int]] = deque()
        self.opened = False
        self.closed = False

    def open(self) -> None:
        self.opened = True

    def close(self) -> None:
        self.closed = True

    def drain(self) -> int:
        n = len(self.sysex_q)
        self.sysex_q.clear()
        return n

    def send_sysex(self, frame: bytes) -> None:
        self._bus.sent.append(bytes(frame))
        self._bus.handle(bytes(frame), self)

    def receive(self, timeout_s: float = 1.0) -> bytes | None:
        return self.sysex_q.popleft() if self.sysex_q else None

    def receive_note(self, timeout_s: float = 1.0) -> list[int] | None:
        return self.note_q.popleft() if self.note_q else None


class DawMidiBus:
    """Answers hello and turns the pad-press SysEx stimulus into NoteOn/NoteOff.

    `answer_hello` / `answer_pads` let a test simulate the two ENV-worthy
    firmware failures: no hello reply at all, and a pad press that never
    reaches the note bridge.
    """

    def __init__(self, *, pads: int = 16, caps: int = 0b1111,
                answer_hello: bool = True, answer_pads: bool = True) -> None:
        self.sent: list[bytes] = []
        self.io: FakeMidi | None = None
        self.pads = pads
        self.caps = caps
        self.answer_hello = answer_hello
        self.answer_pads = answer_pads

    def port(self, role: MidiRole) -> FakeMidi:
        io = FakeMidi(self, role)
        self.io = io
        return io

    def handle(self, frame: bytes, io: FakeMidi) -> None:
        if frame[:4] == bytes([0xF0, 0x7D, 0x1E, 0x01]):        # hello, host->dev
            if self.answer_hello:
                io.sysex_q.append(bytes([0xF0, 0x7D, 0x1E, 0x01, 1, self.pads, 3,
                                         self.caps, 0xF7]))
        elif frame[:4] == bytes([0xF0, 0x7D, 0x1D, 0x07]):       # pad press
            if self.answer_pads:
                pad, vel = frame[4], frame[5]
                note = daw_control.BASE_NOTE + pad
                io.note_q.append([0x90 | daw_control.CHANNEL, note, vel])
        elif frame[:4] == bytes([0xF0, 0x7D, 0x1D, 0x08]):       # pad release
            if self.answer_pads:
                pad = frame[4]
                note = daw_control.BASE_NOTE + pad
                io.note_q.append([0x80 | daw_control.CHANNEL, note, 0])
        # led_all / text: no reply needed for this scenario


@pytest.fixture(autouse=True)
def fast(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(daw_control, "RX_SETTLE_S", 0.0)
    monkeypatch.setattr(daw_control, "HELLO_TIMEOUT_S", 0.2)
    monkeypatch.setattr(daw_control, "NOTE_TIMEOUT_S", 0.2)


def wire(monkeypatch: pytest.MonkeyPatch, bus: DawMidiBus) -> None:
    monkeypatch.setattr(daw_control, "MidiIO", lambda device, role: bus.port(role))


def expected_led_colors() -> list[str]:
    pattern = daw_control._led_pattern()
    return daw_control._expected_led_colors(pattern)


def sync_led_state(dev: DawDevice) -> None:
    """Fake the effect `led_all` would have on the device's LED model."""
    dev.led_colors = expected_led_colors()


# --------------------------------------------------------------------------- tests


def test_full_run_passes(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    ctx, dev = make_ctx(tmp_path)
    bus = DawMidiBus()
    wire(monkeypatch, bus)

    # The scenario sends led_all and then reads LED_STATE back; only a real
    # device would connect the two, so the fake CDC device is told what the
    # firmware would now report.
    real_send_sysex = FakeMidi.send_sysex

    def send_sysex_and_sync(self: FakeMidi, frame: bytes) -> None:
        real_send_sysex(self, frame)
        if bytes(frame)[:4] == bytes([0xF0, 0x7D, 0x1E, 0x03]):
            sync_led_state(dev)

    monkeypatch.setattr(FakeMidi, "send_sysex", send_sysex_and_sync)

    scenario = get("daw_control")
    report = scenario.run(ctx, scenario.Params(), base.Progress())

    assert report.passed is True
    assert report.exit_code == 0
    assert report.data["hello"]["pads"] == 16
    assert report.data["led_actual"] == report.data["led_expected"]
    assert report.data["note_on"][:2] == [0x90 | daw_control.CHANNEL,
                                          daw_control.BASE_NOTE + daw_control.TEST_PAD]
    assert report.data["note_off"][:2] == [0x80 | daw_control.CHANNEL,
                                           daw_control.BASE_NOTE + daw_control.TEST_PAD]
    assert report.data["running_after_text"] == daw_control.APP_NAME
    # the fake hello carries no `control` cap: the mode round trip is skipped
    assert report.data["control_mode"] == "skipped: firmware without the control cap"
    # the app was started (it was not the running app) and stopped afterwards
    assert any(w.startswith("APP_START") for w in dev.written)
    assert dev.written[-1] == "APP_STOP" or "APP_STOP" in dev.written


def test_app_already_running_is_not_restarted(tmp_path: Path,
                                              monkeypatch: pytest.MonkeyPatch) -> None:
    ctx, dev = make_ctx(tmp_path)
    dev.running = daw_control.APP_NAME
    bus = DawMidiBus()
    wire(monkeypatch, bus)
    sync_led_state(dev)     # so the led check passes without an app-start round trip

    scenario = get("daw_control")
    report = scenario.run(ctx, scenario.Params(), base.Progress())
    assert report.passed is True
    assert not any(w.startswith("APP_START") for w in dev.written)


def test_no_esp_midi_port_is_env(tmp_path: Path) -> None:
    ctx, _dev = make_ctx(tmp_path, esp_midi=None)
    scenario = get("daw_control")
    with pytest.raises(HilError) as ei:
        scenario.run(ctx, scenario.Params(), base.Progress())
    assert ei.value.code == ENV


def test_app_not_installed_is_env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    ctx, dev = make_ctx(tmp_path)
    dev.apps = ["Sampler"]           # DAW Control is not installed
    bus = DawMidiBus()
    wire(monkeypatch, bus)
    scenario = get("daw_control")
    with pytest.raises(HilError) as ei:
        scenario.run(ctx, scenario.Params(), base.Progress())
    assert ei.value.code == ENV


def test_no_hello_reply_is_a_firmware_failure(tmp_path: Path,
                                              monkeypatch: pytest.MonkeyPatch) -> None:
    ctx, dev = make_ctx(tmp_path)
    bus = DawMidiBus(answer_hello=False)
    wire(monkeypatch, bus)
    scenario = get("daw_control")
    report = scenario.run(ctx, scenario.Params(), base.Progress())
    assert report.passed is False
    assert report.exit_code == 1
    assert "hello" in report.summary


def test_wrong_pad_count_fails(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    ctx, dev = make_ctx(tmp_path)
    bus = DawMidiBus(pads=8)
    wire(monkeypatch, bus)
    scenario = get("daw_control")
    report = scenario.run(ctx, scenario.Params(), base.Progress())
    assert report.passed is False
    assert report.exit_code == 1
    assert "pads=8" in report.summary


def test_led_mismatch_fails(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    ctx, dev = make_ctx(tmp_path)
    bus = DawMidiBus()
    wire(monkeypatch, bus)
    # the fake CDC device never learns the new LED pattern -> stays all-black
    scenario = get("daw_control")
    report = scenario.run(ctx, scenario.Params(), base.Progress())
    assert report.passed is False
    assert report.exit_code == 1
    assert "LED_STATE" in report.summary


def test_no_note_on_pad_press_fails(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    ctx, dev = make_ctx(tmp_path)
    bus = DawMidiBus(answer_pads=False)
    wire(monkeypatch, bus)
    sync_led_state(dev)
    scenario = get("daw_control")
    report = scenario.run(ctx, scenario.Params(), base.Progress())
    assert report.passed is False
    assert report.exit_code == 1
    assert "NoteOn" in report.summary


def test_app_stop_is_called_even_when_the_run_fails(tmp_path: Path,
                                                    monkeypatch: pytest.MonkeyPatch) -> None:
    ctx, dev = make_ctx(tmp_path)
    bus = DawMidiBus(answer_hello=False)
    wire(monkeypatch, bus)
    scenario = get("daw_control")
    scenario.run(ctx, scenario.Params(), base.Progress())
    assert "APP_STOP" in dev.written
