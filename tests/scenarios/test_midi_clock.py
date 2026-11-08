"""Scenario `midi_clock` against a fake board that models the clock from what it is sent."""
from __future__ import annotations

import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest

from crosspad_hil.cdc import CdcLink
from crosspad_hil.devices import Device, MidiPortInfo, Ports, SerialPortInfo, UsbMode
from crosspad_hil.scenarios import base, midi_clock
from tests.fakes import FakeSerial

CDC = SerialPortInfo(path="/dev/ttyACM0", vid=0x303A, pid=0x3456, serial="CP-1",
                     product="CrossPad", location="1-2.1")
ESP_MIDI = MidiPortInfo(name="Crosspad MIDI 1", rtmidi_out=0, rtmidi_in=0,
                        alsa_hw="hw:4,0,0", rawmidi="/dev/snd/midiC4D0")
# The least a sleep moves the virtual clock: the sender spins on its deadline
# with sleep(0) once it is within 2 ms, and a sleep that moved nothing would
# spin forever.
SLEEP_QUANTUM_S = 1e-6
# Virtual time one CLOCK_STATUS read takes. The hand-back loop polls until
# release_s has passed on the scenario's clock, so a read has to move it.
STATUS_READ_S = 0.02
# How long the link's reader sleeps on the fake's empty buffer. FakeSerial's
# 0.2 s default is a 0.2 s wall-clock wait for every one of the ~60 reads a
# hand-back at STATUS_READ_S takes.
READ_POLL_S = 0.005


class _VirtualTime:
    """`time` for the scenario under test: perf_counter and monotonic read one
    virtual clock that only sleep() and the fake board's status reads move.

    The board stamps its ticks on the same clock, so the tempo it reads back is
    the tempo the sender scheduled. On the wall clock it was whatever the host
    let the sender keep: Windows sleeps in ~15.6 ms steps, and a loaded Linux
    runner delivered 240 BPM as 240.95.
    """

    def __init__(self) -> None:
        self.now = 0.0

    def __getattr__(self, name: str) -> Any:
        return getattr(time, name)

    def perf_counter(self) -> float:
        return self.now

    def monotonic(self) -> float:
        return self.now

    def sleep(self, seconds: float) -> None:
        self.now += max(seconds, SLEEP_QUANTUM_S)


class ClockBoard(FakeSerial):
    """CLOCK_STATUS from the realtime bytes the fake port delivered, timed on
    ``self.time``, the virtual clock the scenario runs on."""

    def __init__(self, bpm_error: float = 0.0, release_s: float = 1.05) -> None:
        super().__init__(timeout=READ_POLL_S)
        self.bpm_error = bpm_error
        self.release_s = release_s
        self.time = _VirtualTime()
        self.ticks: list[float] = []
        self.running = False
        self.ticks_out = 0

    def realtime(self, status: int) -> None:
        if status == 0xF8:
            self.ticks.append(self.time.perf_counter())
        elif status in (0xFA, 0xFB):
            self.running = True
        elif status == 0xFC:
            self.running = False

    def _status(self) -> str:
        now = self.time.perf_counter()
        host = bool(self.ticks) and now - self.ticks[-1] <= self.release_s
        recent = self.ticks[-97:]
        bpm = 120.0
        if len(recent) >= 2:
            # Least-squares period over every tick, not last-minus-first: one
            # tick a busy CI VM sent late at either end read 240 BPM as 240.96.
            n = len(recent)
            mean_i, mean_t = (n - 1) / 2.0, sum(recent) / n
            num = sum((i - mean_i) * (t - mean_t) for i, t in enumerate(recent))
            den = sum((i - mean_i) ** 2 for i in range(n))
            bpm = 60.0 / (24 * (num / den))
        bpm *= 1.0 + self.bpm_error
        self.ticks_out += 5   # the internal clock keeps ticking between reads
        return (f"CLOCK: src={'host' if host else 'int'} bpm={bpm:.2f} "
                f"running={int(self.running)} ticks_in={len(self.ticks)} "
                f"ticks_out={self.ticks_out} last_gap_us=0 max_jitter_us=300 max_late_us=120")

    def write(self, data: bytes) -> int:
        for raw in data.decode("utf-8", "replace").split("\n"):
            cmd = raw.strip()
            if not cmd:
                continue
            self.written.append(cmd)
            if cmd.startswith("CLOCK_STATUS"):
                self.time.sleep(STATUS_READ_S)
                self.feed([self._status()])
            else:
                self._rx.append((cmd + "\r\n").encode())
        return len(data)


class FakePort:
    def __init__(self, board: ClockBoard) -> None:
        self.board = board
        self.sent: list[int] = []

    def open(self) -> None: ...
    def close(self) -> None: ...

    def send_realtime(self, status: int) -> None:
        self.sent.append(status)
        self.board.realtime(status)


@dataclass
class FakeContext(base.Context):
    board: ClockBoard = None  # type: ignore[assignment]

    def open_cdc(self) -> CdcLink:
        link = CdcLink(self.device.ports.cdc.path, serial_factory=lambda path, **kw: self.board)
        link.open()
        self.opened.append(link)
        return link


def run(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, board: ClockBoard,
        params: midi_clock.Params) -> tuple[base.Report, FakePort]:
    port = FakePort(board)
    monkeypatch.setattr(midi_clock, "MidiIO", lambda device, role: port)
    monkeypatch.setattr(midi_clock, "time", board.time)
    device = Device(id="dev_ab12", serial="CP-1", usb_mode=UsbMode.DEFAULT,
                    ports=Ports(cdc=CDC, esp_midi=ESP_MIDI))
    ctx = FakeContext(device=device, workdir=tmp_path, cancelled=threading.Event(),
                      log=lambda s: None, board=board)
    return midi_clock.SCENARIO.run(ctx, params, base.Progress()), port


def test_follows_the_tempo_and_hands_back(tmp_path, monkeypatch) -> None:
    report, port = run(tmp_path, monkeypatch, ClockBoard(),
                       midi_clock.Params(bpms="240", beats=5, release_s=1.5))
    assert report.passed, report.summary
    assert report.exit_code == 0
    assert port.sent[0] == 0xFA and port.sent[-1] == 0xFC
    assert port.sent.count(0xF8) == 5 * 24


def test_a_tempo_half_a_percent_off_fails(tmp_path, monkeypatch) -> None:
    report, _ = run(tmp_path, monkeypatch, ClockBoard(bpm_error=0.005),
                    midi_clock.Params(bpms="240", beats=5, release_s=1.5))
    assert not report.passed
    assert report.exit_code == 1
    assert "board says" in report.summary


def test_a_board_that_never_hands_back_fails(tmp_path, monkeypatch) -> None:
    report, _ = run(tmp_path, monkeypatch, ClockBoard(release_s=99.0),
                    midi_clock.Params(bpms="240", beats=5, release_s=1.5))
    assert not report.passed
    assert "hand-back" in report.summary
