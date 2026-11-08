"""Shared fixtures. Nothing here touches hardware."""
from __future__ import annotations

from collections.abc import Callable, Sequence

import pytest

from tests.fakes import FakeMidiBus, FakeSerial


@pytest.fixture(autouse=True)
def _private_pins(tmp_path_factory, monkeypatch):
    """Every test gets a pin journal of its own, never the machine's real one."""
    path = tmp_path_factory.mktemp("pins") / "pins.json"
    monkeypatch.setenv("CROSSPAD_HIL_PINS", str(path))
    return path


@pytest.fixture(autouse=True)
def _no_hub_by_default(monkeypatch, request):
    """Every test talks to its fakes directly unless it is marked `uses_hub`:
    without this, an unmarked test would autostart a real hub."""
    if "uses_hub" not in request.keywords:
        monkeypatch.setenv("CROSSPAD_HIL_NO_HUB", "1")


@pytest.fixture(autouse=True)
def _private_bench(tmp_path_factory, monkeypatch):
    """Every test gets a bench lease file of its own: never the machine's real one,
    where a session's live claim would refuse the board ops the tests make."""
    path = tmp_path_factory.mktemp("bench") / "bench.json"
    monkeypatch.setenv("CROSSPAD_HIL_BENCH", str(path))
    monkeypatch.delenv("CROSSPAD_BENCH_MD", raising=False)
    monkeypatch.delenv("CROSSPAD_BENCH_HOLDER", raising=False)
    monkeypatch.delenv("CROSSPAD_BENCH_MODE", raising=False)
    return path


@pytest.fixture(autouse=True)
def _no_real_services(monkeypatch):
    """No test may pause the machine's real audio-nodes service: systemctl is
    answered here, "inactive", unless a test installs its own."""
    import subprocess

    from crosspad_hil import card_hold

    def fake(argv):
        return subprocess.CompletedProcess(argv, 3, "inactive\n", "")

    monkeypatch.setattr(card_hold, "_run", fake)


@pytest.fixture(autouse=True)
def _no_real_audio_hold(monkeypatch):
    """No test may write a hold into the machine's lock dir or set its PipeWire
    defaults: a USB-mode switch to audio records the call here instead."""
    from crosspad_hil import usbmode

    calls: list[tuple] = []
    monkeypatch.setattr(usbmode, "_hold_defaults", lambda *a, **kw: calls.append((a, kw)))
    return calls


@pytest.fixture
def fake_serial() -> Callable[..., FakeSerial]:
    def make(
        script: Sequence[tuple[str, str | list[str]]] = (), timeout: float = 0.01
    ) -> FakeSerial:
        return FakeSerial(script, timeout=timeout)

    return make


@pytest.fixture
def fake_midi() -> FakeMidiBus:
    """One wired out/in rtmidi pair; pass its factories to MidiIO."""
    return FakeMidiBus()


@pytest.fixture
def knowledge() -> dict:
    from crosspad_hil.knowledge import load

    return {"markers": load("markers"), "sysex": load("sysex"), "cdc": load("cdc")}


@pytest.fixture
def fake_backends():
    """Backends with one CrossPad in default mode (ESP CDC + STM VCP + ESP MIDI + STM MIDI).
    Needs crosspad_hil.devices (Plan A chunk 2); skips until it exists."""
    try:
        from crosspad_hil.devices import (
            AudioCardInfo,
            Backends,
            MidiPortInfo,
            SerialPortInfo,
        )
    except ImportError:
        pytest.skip("crosspad_hil.devices not implemented yet")

    serial_ports = [
        SerialPortInfo(path="/dev/ttyACM0", vid=0x303A, pid=0x3456, serial="AABBCC",
                       product="Crosspad", location="1-2.3:1.0"),
        SerialPortInfo(path="/dev/ttyACM1", vid=0x0483, pid=0x5740, serial="STM001",
                       product="CrossPad MIDI+Serial", location="1-2.4:1.2"),
    ]
    midi_ports = [
        MidiPortInfo(name="Crosspad MIDI 1", rtmidi_out=0, rtmidi_in=0, alsa_hw="hw:4,0,0",
                     rawmidi="/dev/snd/midiC4D0"),
        MidiPortInfo(name="CrossPad MIDI+Serial 2", rtmidi_out=1, rtmidi_in=1,
                     alsa_hw="hw:5,0,0", rawmidi="/dev/snd/midiC5D0"),
    ]
    audio_cards: list[AudioCardInfo] = []
    return Backends(
        list_serial=lambda: list(serial_ports),
        list_midi=lambda: list(midi_ports),
        list_audio=lambda: list(audio_cards),
    )


@pytest.fixture(autouse=True)
def isolated_port_locks(tmp_path_factory: pytest.TempPathFactory,
                        monkeypatch: pytest.MonkeyPatch) -> None:
    """Give every test its own port-lock directory.

    Port locks are cross-process by design, so without this a test that leaves a
    Console or CdcLink open makes the next test fail with PORT_BUSY -- and the
    failure lands in whichever test happens to run next, not the leaky one.
    """
    monkeypatch.setenv("XDG_RUNTIME_DIR", str(tmp_path_factory.mktemp("locks")))


@pytest.fixture
def exclusive_log(monkeypatch) -> list[tuple[str, str]]:
    """Record every exclusive op instead of asking a hub: ("take"|"release", purpose)."""
    from crosspad_hil import hub_link

    log: list[tuple[str, str]] = []

    class _Hold:
        def __init__(self, purpose: str) -> None:
            self.purpose = purpose
            self.released = False

        def release(self) -> None:
            if not self.released:
                self.released = True
                log.append(("release", self.purpose))

    def hold(purpose: str, **_: object) -> _Hold:
        log.append(("take", purpose))
        return _Hold(purpose)

    monkeypatch.setattr(hub_link, "hold_exclusive", hold)
    return log
