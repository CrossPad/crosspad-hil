from __future__ import annotations

import threading
import time
from dataclasses import replace

import pytest

from crosspad_hil import usbmode
from crosspad_hil.devices import (
    AudioCardInfo,
    Device,
    MidiPortInfo,
    Ports,
    SerialPortInfo,
    UsbMode,
)
from crosspad_hil.errors import BAD_ARGS, NO_DEVICE, TIMEOUT, HilError
from crosspad_hil.midi import MidiIO
from crosspad_hil.usbmode import audio_mode, set_mode
from tests.fakes import FakeMidiBus

CDC = SerialPortInfo(path="/dev/ttyACM0", vid=0x303A, pid=0x3456, serial="ABCD1234",
                     product="Crosspad", location="1-2:1.0")
UAC = AudioCardInfo(name="Crosspad Audio", sounddevice_index=3, alsa_id="hw:3")
MIDI = MidiPortInfo(name="Crosspad MIDI 1", rtmidi_out=0, rtmidi_in=0, alsa_hw=None,
                    rawmidi=None)


def dev_default() -> Device:
    return Device(id="dev_1234", serial="ABCD1234", usb_mode=UsbMode.DEFAULT,
                  ports=Ports(cdc=CDC, esp_midi=MIDI))


def dev_audio() -> Device:
    return Device(id="dev_1234", serial="ABCD1234", usb_mode=UsbMode.AUDIO,
                  ports=Ports(uac2=UAC, esp_midi=MIDI))


def dev_gone() -> list[Device]:
    return []


class FakeDiscover:
    """Returns the scripted inventories in order, repeating the last one forever."""

    def __init__(self, sequence: list[list[Device]]) -> None:
        self.sequence = list(sequence)
        self.calls = 0

    def __call__(self):
        self.calls += 1
        idx = min(self.calls - 1, len(self.sequence) - 1)
        return list(self.sequence[idx])


def midi_factory_for(bus: FakeMidiBus):
    def factory(device: Device, role=None, **kw) -> MidiIO:
        return MidiIO(device, role or "esp", out_factory=bus.out_factory,
                      in_factory=bus.in_factory)
    return factory


def test_set_mode_audio_polls_until_uac2_present() -> None:
    bus = FakeMidiBus()
    disc = FakeDiscover([dev_gone(), dev_gone(), [dev_audio()]])
    slept: list[float] = []
    d = set_mode(dev_default(), UsbMode.AUDIO, discover_fn=disc,
                 midi_factory=midi_factory_for(bus), sleep=slept.append)
    assert bus.sent == [[0xF0, 0x7D, 0x1B, 0x02, 0xF7]]
    assert bus.outs[0].closed
    assert d.usb_mode is UsbMode.AUDIO and d.ports.uac2 is UAC
    assert disc.calls == 3
    assert slept == [0.5, 0.5]


def test_set_mode_default_accepts_cdc_presence_even_if_mode_unknown() -> None:
    bus = FakeMidiBus()
    seen = replace(dev_default(), usb_mode=UsbMode.UNKNOWN)
    disc = FakeDiscover([[seen]])
    d = set_mode(dev_audio(), UsbMode.DEFAULT, discover_fn=disc,
                 midi_factory=midi_factory_for(bus), sleep=lambda s: None)
    assert bus.sent == [[0xF0, 0x7D, 0x1B, 0x01, 0xF7]]
    assert d is seen


def test_set_mode_ignores_other_devices_and_times_out() -> None:
    bus = FakeMidiBus()
    other = replace(dev_audio(), id="dev_9999", serial="ZZZZ")
    disc = FakeDiscover([[other]])
    clock = iter([0.0, 0.0, 5.0, 10.0, 15.0, 20.0, 25.0, 30.0])
    with pytest.raises(HilError) as e:
        set_mode(dev_default(), UsbMode.AUDIO, timeout_s=20.0, discover_fn=disc,
                 midi_factory=midi_factory_for(bus), sleep=lambda s: None,
                 clock=lambda: next(clock))
    assert e.value.code == TIMEOUT
    assert e.value.details["device"] == "dev_1234"
    assert e.value.details["mode"] == "audio"


def test_set_mode_no_wait_and_arg_errors() -> None:
    bus = FakeMidiBus()
    disc = FakeDiscover([dev_gone()])
    d = set_mode(dev_default(), UsbMode.AUDIO, wait=False, discover_fn=disc,
                 midi_factory=midi_factory_for(bus), sleep=lambda s: None)
    assert d.usb_mode is UsbMode.AUDIO and disc.calls == 0
    with pytest.raises(HilError) as e:
        set_mode(dev_default(), UsbMode.BOOTLOADER, discover_fn=disc,
                 midi_factory=midi_factory_for(bus))
    assert e.value.code == BAD_ARGS
    no_midi = replace(dev_default(), ports=Ports(cdc=CDC))
    with pytest.raises(HilError) as e:
        set_mode(no_midi, UsbMode.AUDIO, discover_fn=disc,
                 midi_factory=midi_factory_for(bus))
    assert e.value.code == NO_DEVICE
    assert bus.sent == [[0xF0, 0x7D, 0x1B, 0x02, 0xF7]]


def test_audio_mode_restores_default_unless_keep() -> None:
    bus = FakeMidiBus()
    disc = FakeDiscover([[dev_audio()], [dev_default()]])
    with audio_mode(dev_default(), discover_fn=disc, midi_factory=midi_factory_for(bus),
                    sleep=lambda s: None) as d:
        assert d.usb_mode is UsbMode.AUDIO
        assert bus.sent == [[0xF0, 0x7D, 0x1B, 0x02, 0xF7]]
    assert bus.sent == [[0xF0, 0x7D, 0x1B, 0x02, 0xF7], [0xF0, 0x7D, 0x1B, 0x01, 0xF7]]

    bus2 = FakeMidiBus()
    disc2 = FakeDiscover([[dev_audio()]])
    with audio_mode(dev_default(), keep=True, discover_fn=disc2,
                    midi_factory=midi_factory_for(bus2), sleep=lambda s: None):
        pass
    assert bus2.sent == [[0xF0, 0x7D, 0x1B, 0x02, 0xF7]]


class _WedgedMidiIO:
    """Stands in for a real MidiIO whose underlying rtmidi/ALSA call never returns —
    e.g. a USB MIDI endpoint the firmware left mid-teardown. open()/send()/close() are
    C-extension calls with no timeout of their own; this simulates that by blocking on
    an Event that is never set."""

    def __enter__(self) -> _WedgedMidiIO:
        return self

    def __exit__(self, *exc: object) -> None:
        return None

    def send_sysex(self, frame: bytes) -> None:
        threading.Event().wait()  # never returns


def test_set_mode_send_wedge_times_out_instead_of_hanging_forever(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A hang in opening/sending the mode-switch frame must not block set_mode forever
    (this is the real-world failure behind a release_gate run stuck 4.5h in
    usb_mode_cycle with no console output and no report.json)."""
    monkeypatch.setattr(usbmode, "SEND_TIMEOUT_S", 0.05)
    disc = FakeDiscover([dev_gone()])
    t0 = time.monotonic()
    with pytest.raises(HilError) as e:
        set_mode(dev_default(), UsbMode.AUDIO, discover_fn=disc,
                 midi_factory=lambda device, role=None, **kw: _WedgedMidiIO(),
                 sleep=lambda s: None)
    elapsed = time.monotonic() - t0
    assert e.value.code == TIMEOUT
    assert "wedged" in e.value.message
    assert elapsed < 2.0, f"set_mode blocked for {elapsed:.2f}s instead of bailing out"


def test_audio_mode_restores_on_exception() -> None:
    bus = FakeMidiBus()
    disc = FakeDiscover([[dev_audio()], [dev_default()]])
    with pytest.raises(RuntimeError):
        with audio_mode(dev_default(), discover_fn=disc, midi_factory=midi_factory_for(bus),
                        sleep=lambda s: None):
            raise RuntimeError("scenario blew up")
    assert bus.sent[-1] == [0xF0, 0x7D, 0x1B, 0x01, 0xF7]


def test_set_mode_audio_holds_the_desktop_defaults(_no_real_audio_hold) -> None:
    """A test's switch into MIDI+UAC2 is not a card the user plugged in."""
    bus = FakeMidiBus()
    set_mode(dev_default(), UsbMode.AUDIO, discover_fn=FakeDiscover([[dev_audio()]]),
             midi_factory=midi_factory_for(bus), sleep=lambda s: None)
    assert len(_no_real_audio_hold) == 1


def test_set_mode_default_takes_no_hold(_no_real_audio_hold) -> None:
    bus = FakeMidiBus()
    set_mode(dev_audio(), UsbMode.DEFAULT, discover_fn=FakeDiscover([[dev_default()]]),
             midi_factory=midi_factory_for(bus), sleep=lambda s: None)
    assert _no_real_audio_hold == []
