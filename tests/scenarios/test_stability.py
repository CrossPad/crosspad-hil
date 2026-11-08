"""Scenario `stability` against a scripted console — no board, and no hours.

A soak is a loop around a console reader, so the console is a fake serial port
this file feeds and every duration is measured in tenths of a second. What is
being pinned down is the bookkeeping: which banner counts as a reset and which
one is the harness's own, that a half-finished boot is its own failure, that a
silent port is reported once and not once per poll, that E-lines are counted
without failing the run, and that a cancelled soak still hands back what it saw.
"""
from __future__ import annotations

import json
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest

from crosspad_hil.console import Console
from crosspad_hil.devices import Device, MidiPortInfo, Ports, SerialPortInfo, UsbMode
from crosspad_hil.errors import HilError
from crosspad_hil.knowledge import load as load_knowledge
from crosspad_hil.scenarios import base, get, stability
from tests.fakes import AudioDesk, FakeSerial

CONSOLE = SerialPortInfo(path="/dev/ttyACM1", vid=0x0483, pid=0x5740, serial="STM-1",
                         product="CrossPad MIDI+Serial", location="1-2.2")
ESP_MIDI = MidiPortInfo(name="Crosspad MIDI 1", rtmidi_out=0, rtmidi_in=0,
                        alsa_hw="hw:4,0,0", rawmidi=None)
BANNER = "ESP-ROM:esp32s3-20210327"
RST_LINE = "rst:0x3 (RTC_SW_SYS_RST),boot:0x8 (SPI_FAST_FLASH_BOOT)"
PANIC = "Guru Meditation Error: Core  1 panic'ed (LoadProhibited)."
MARKERS = list(load_knowledge("markers")["required_stability"])
HEAP_BLOCK = ["I (100) PerfMon: Heap Statistics",
              "I (100) PerfMon:   Free: 120000 bytes",
              "I (100) PerfMon:   Free: 4000000 bytes",
              "I (100) PerfMon: Total tasks: 21"]


CDC = SerialPortInfo(path="/dev/ttyACM0", vid=0x303A, pid=0x3456, serial="CP-1",
                     product="Crosspad", location="1-2.1")


def make_device(cdc: bool = False) -> Device:
    return Device(id="dev_ab12", serial="CP-1", usb_mode=UsbMode.DEFAULT,
                  ports=Ports(console=CONSOLE, esp_midi=ESP_MIDI, cdc=CDC if cdc else None))


class MemDevice(FakeSerial):
    """A CDC endpoint that answers MEM -- or, once ``dead``, nothing at all."""

    def __init__(self) -> None:
        super().__init__(timeout=0.01)
        self.dead = False
        self.probes = 0

    def write(self, data: bytes) -> int:
        cmd = data.decode("utf-8", "replace").strip()
        self.written.append(cmd)
        if self.dead:
            return len(data)
        if cmd == "MEM":
            self.probes += 1
            self.feed([f"MEM: int_free={40000 - self.probes} int_largest=10000 int_min=30000 "
                       "int_blocks=500 psram_free=390000 psram_largest=270000 psram_blocks=100"])
        else:
            self.feed(["OK"])
        return len(data)


def boot_lines(banner: bool = True, complete: bool = True) -> list[str]:
    lines = [BANNER, RST_LINE] if banner else []
    return lines + [f"I (900) app: {m} ok" for m in (MARKERS if complete else MARKERS[:2])]


@dataclass
class FakeContext(base.Context):
    """A console that is a fake serial port, opened without the reset dance."""

    console_fake: FakeSerial = None  # type: ignore[assignment]
    logs: list[str] = None  # type: ignore[assignment]
    cdc_fake: FakeSerial | None = None

    def open_console(self, reset: bool = False) -> Console:
        fake = self.console_fake
        con = Console(self.device.ports.console.path,
                      log_path=self.workdir / "console.log",
                      serial_factory=lambda path, **kw: fake)
        con.open(reset=reset)
        self.opened.append(con)
        return con


def make_ctx(tmp_path: Path, cdc: FakeSerial | None = None) -> tuple[FakeContext, FakeSerial]:
    fake = FakeSerial(timeout=0.01)
    logs: list[str] = []
    ctx = FakeContext(device=make_device(cdc=cdc is not None), workdir=tmp_path,
                      cancelled=threading.Event(), log=logs.append, console_fake=fake,
                      logs=logs, cdc_fake=cdc, serial_factory=lambda path, **kw: cdc)
    return ctx, fake


class Recorder(base.Progress):
    """Progress sink that keeps every call, so liveness can be asserted."""

    def __init__(self) -> None:
        self.calls: list[tuple[int, int | None, str]] = []

    def __call__(self, progress: int, total: int | None, message: str) -> None:
        self.calls.append((progress, total, message))


def run(ctx: FakeContext, progress: base.Progress | None = None, **params: Any) -> base.Report:
    scenario = stability.SCENARIO
    return scenario.run(ctx, stability.Params(**params), progress or base.Progress())  # type: ignore[arg-type]


def events(ctx: FakeContext) -> list[dict]:
    path = ctx.workdir / "events.jsonl"
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line]


def feed_later(fake: FakeSerial, lines: list[str], after_s: float) -> threading.Timer:
    """Push console lines into a soak that is already running."""
    timer = threading.Timer(after_s, fake.feed, args=(lines,))
    timer.daemon = True
    timer.start()
    return timer


@pytest.fixture(autouse=True)
def fast(monkeypatch: pytest.MonkeyPatch) -> None:
    """Collapse the intervals a multi-hour soak is built around."""
    monkeypatch.setattr(stability, "POLL_MS", 20)
    monkeypatch.setattr(stability, "PROGRESS_EVERY_S", 0.0)
    monkeypatch.setattr(stability, "BOOT_WINDOW_S", 0.3)
    # A burst must never reach a real port list.
    monkeypatch.setattr(stability, "discover", lambda: [])
    monkeypatch.setattr(stability, "select", lambda devices, ident: make_device())


def test_registered_and_defaults() -> None:
    assert get("stability") is stability.SCENARIO
    p = stability.Params()
    assert (p.duration_hours, p.silence_timeout, p.status_interval,
            p.stim_midi, p.stim_interval) == (0.0, 90.0, 300.0, False, 20.0)


def test_quiet_soak_passes_and_reports_the_heap_trend(tmp_path: Path) -> None:
    ctx, fake = make_ctx(tmp_path)
    fake.feed(boot_lines())
    # Fed while the soak runs: what the console printed before the run started
    # watching is not this run's evidence, and the scenario baselines it away.
    feed_later(fake, HEAP_BLOCK, 0.05)
    feed_later(fake, HEAP_BLOCK, 0.15)
    rep = run(ctx, duration_hours=0.4 / 3600, silence_timeout=10.0)
    assert rep.passed, rep.summary
    assert rep.exit_code == 0
    s = rep.data["stats"]
    assert (s["resets"], s["fatals"], s["stalls"], s["incomplete_boots"]) == (0, 0, 0, 0)
    assert s["lines"] > 0 and s["start"] and s["end"]
    assert rep.data["cancelled"] is False
    # two PerfMon blocks, two heap slots, reported first/last/min/delta each
    assert [h["slot"] for h in rep.data["heap_trend"]] == [0, 1]
    assert rep.data["heap_trend"][0]["first"] == 120000
    assert rep.data["heap_trend"][0]["min"] == 120000
    roles = {a.role for a in rep.artifacts}
    assert {"console", "events"} <= roles
    # the ROM banner that opening the port causes is not a device fault
    assert not any(e["kind"] == "reset" for e in events(ctx))


def test_a_second_banner_is_a_reset_with_its_reason_and_context(tmp_path: Path) -> None:
    ctx, fake = make_ctx(tmp_path)
    fake.feed(boot_lines())
    feed_later(fake, boot_lines(), 0.1)
    rep = run(ctx, duration_hours=0.6 / 3600, silence_timeout=10.0)
    assert not rep.passed
    assert rep.exit_code == 1
    assert rep.data["stats"]["resets"] == 1
    reset = rep.data["reset_events"][0]
    assert reset["uptime_s"] >= 0
    assert reset["reason"] == "0x3 (RTC_SW_SYS_RST)"
    assert Path(reset["context"]).exists()
    assert "reset @" in rep.summary
    # the context dump is an artifact of the run, not just a file on disk
    assert any(a.role == "context" for a in rep.artifacts)
    kinds = [e["kind"] for e in events(ctx)]
    assert kinds.count("reset") == 1 and "reset_reason" in kinds


def test_a_boot_that_never_completes_is_its_own_failure(tmp_path: Path) -> None:
    ctx, fake = make_ctx(tmp_path)
    fake.feed(boot_lines())
    # A banner, then a device that comes back only half way up.
    feed_later(fake, boot_lines(complete=False), 0.1)
    rep = run(ctx, duration_hours=0.8 / 3600, silence_timeout=10.0)
    assert not rep.passed
    s = rep.data["stats"]
    assert s["resets"] == 1
    assert s["incomplete_boots"] == 1
    missing = [e for e in events(ctx) if e["kind"] == "incomplete_boot"][0]["missing"]
    assert set(missing) == set(MARKERS[2:])


def test_silence_is_a_stall_reported_once(tmp_path: Path) -> None:
    ctx, fake = make_ctx(tmp_path)
    fake.feed(boot_lines())
    rep = run(ctx, duration_hours=0.6 / 3600, silence_timeout=0.05)
    assert not rep.passed
    assert rep.exit_code == 1
    # many polls happen inside the silent stretch; one stall comes out of it
    assert rep.data["stats"]["stalls"] == 1
    stall = [e for e in events(ctx) if e["kind"] == "stall"][0]
    assert stall["silent_s"] >= 0.05
    assert Path(stall["context"]).exists()


def test_a_stall_ends_when_the_board_speaks_again(tmp_path: Path) -> None:
    ctx, fake = make_ctx(tmp_path)
    fake.feed(boot_lines())
    # feed_later's Timer starts counting here, before run() below has even opened
    # the console — load_knowledge() and ctx.open_console() run first and only
    # then does _soak() start its own silence clock. On a loaded Windows CI
    # runner that setup gap alone can run past 500ms, which ate the previous
    # (tighter) margins here whole and collapsed the first silent stretch to
    # ~0s — never enough to cross silence_timeout, leaving only the second
    # stall to fire. Both numbers below are sized to survive several seconds
    # of that gap, not just scheduling jitter once already inside the loop.
    feed_later(fake, ["I (2000) app: still here"], 2.0)
    rep = run(ctx, duration_hours=4.0 / 3600, silence_timeout=0.3)
    # two silent stretches — before that line and after it — so two stalls
    assert rep.data["stats"]["stalls"] == 2


def test_a_panic_line_fails_the_soak(tmp_path: Path) -> None:
    ctx, fake = make_ctx(tmp_path)
    fake.feed(boot_lines())
    feed_later(fake, [PANIC], 0.1)
    rep = run(ctx, duration_hours=0.5 / 3600, silence_timeout=10.0)
    assert not rep.passed
    assert rep.data["stats"]["fatals"] == 1
    fatal = [e for e in events(ctx) if e["kind"] == "fatal"][0]
    assert "Guru Meditation" in fatal["line"]
    assert Path(fatal["context"]).exists()


def test_error_lines_are_counted_but_do_not_fail_the_run(tmp_path: Path) -> None:
    """hil_stability's verdict never looked at E-lines, and neither does this."""
    ctx, fake = make_ctx(tmp_path)
    fake.feed(boot_lines())
    feed_later(fake, ["E (1200) sdmmc: read failed",
                      "E (1300) audio: STARTUP.wav file not found"], 0.05)
    rep = run(ctx, duration_hours=0.4 / 3600, silence_timeout=10.0)
    assert rep.passed, rep.summary
    # the allow-listed "file not found" is not one of them
    assert rep.data["stats"]["errors"] == 1


def test_cancellation_ends_the_soak_with_a_report(tmp_path: Path) -> None:
    ctx, fake = make_ctx(tmp_path)
    fake.feed(boot_lines())
    recorder = Recorder()
    timer = threading.Timer(0.2, ctx.cancelled.set)
    timer.daemon = True
    timer.start()
    # duration_hours=0 is the "until stopped" soak: only the cancel ends it.
    rep = run(ctx, recorder, duration_hours=0.0, silence_timeout=10.0)
    assert rep.passed, rep.summary
    assert rep.data["cancelled"] is True
    assert rep.data["seconds"] >= 0.2
    assert "cancelled" in rep.summary
    # a soak with no end has nothing to report progress against
    assert recorder.calls and all(total is None for _p, total, _m in recorder.calls)
    assert recorder.calls[-1][2] == "soak finished"


def test_progress_keeps_reporting_while_the_soak_runs(tmp_path: Path) -> None:
    ctx, fake = make_ctx(tmp_path)
    fake.feed(boot_lines())
    recorder = Recorder()
    rep = run(ctx, recorder, duration_hours=0.5 / 3600, silence_timeout=10.0)
    assert rep.passed
    assert len(recorder.calls) > 2                       # alive, not just start and end
    assert all(total == 1 for _p, total, _m in recorder.calls)   # int(0.5 s), floored to 1
    assert [p for p, _t, _m in recorder.calls] == sorted(p for p, _t, _m in recorder.calls)
    assert "up 0s" in recorder.calls[0][2]


def test_midi_stimulus_bursts_and_stops_with_the_run(tmp_path: Path,
                                                     monkeypatch: pytest.MonkeyPatch) -> None:
    sent: list[tuple[bool, int, int]] = []

    class FakeMidi:
        def __init__(self, device: Device, role: Any) -> None:
            self.device = device

        def __enter__(self) -> Any:
            return self

        def __exit__(self, *exc: object) -> None:
            return None

        def send_note(self, on: bool, note: int, vel: int = 100) -> None:
            sent.append((on, note, vel))

    monkeypatch.setattr(stability, "MidiIO", lambda device, role: FakeMidi(device, role))
    monkeypatch.setattr(stability, "STIM_GAP_S", 0.0)
    ctx, fake = make_ctx(tmp_path)
    fake.feed(boot_lines())
    rep = run(ctx, duration_hours=0.5 / 3600, silence_timeout=10.0,
              stim_midi=True, stim_interval=0.05)
    assert rep.passed, rep.summary
    assert rep.data["stim_bursts"] >= 1
    assert rep.data["stim_errors"] == 0
    # one burst is every note on, then every note off
    assert sent[:4] == [(True, n, stability.STIM_VELOCITY) for n in stability.STIM_NOTES]
    assert sent[4:8] == [(False, n, 100) for n in stability.STIM_NOTES]
    time.sleep(0.05)
    assert not any(t.name == "stability_midi_stim" for t in threading.enumerate())


def test_a_silent_board_that_answers_mem_is_alive(tmp_path: Path) -> None:
    """No console chatter for the whole soak; the CDC probe says it is up."""
    cdc = MemDevice()
    ctx, fake = make_ctx(tmp_path, cdc=cdc)
    fake.feed(boot_lines())
    rep = run(ctx, duration_hours=0.6 / 3600, silence_timeout=0.05, probe_interval=0.05)
    assert rep.passed is True
    assert rep.data["liveness"] == "cdc"
    assert rep.data["stats"]["stalls"] == 0
    assert cdc.probes >= 2
    assert [h["slot"] for h in rep.data["heap_trend"]] == [0, 1]
    assert "heap slots: [0] internal free, [1] PSRAM free" in rep.summary


def test_an_unanswered_probe_is_the_stall(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    cdc = MemDevice()
    ctx, fake = make_ctx(tmp_path, cdc=cdc)
    fake.feed(boot_lines())
    cdc.dead = True

    def no_reply(link):
        raise HilError("TIMEOUT", "no MEM reply")

    monkeypatch.setattr(stability.verbs, "mem", no_reply)
    rep = run(ctx, duration_hours=0.4 / 3600, silence_timeout=10.0, probe_interval=0.05)
    assert rep.passed is False
    assert rep.data["stats"]["stalls"] == 1
    stall = [e for e in events(ctx) if e["kind"] == "stall"][0]
    assert stall["probe"] == "TIMEOUT"


class DeskMemDevice(MemDevice):
    """MemDevice plus the mixer/codec verbs; the first MEM probe finds the desk moved."""

    def __init__(self, desk: AudioDesk) -> None:
        super().__init__()
        self.desk = desk

    def write(self, data: bytes) -> int:
        cmd = data.decode("utf-8", "replace").strip()
        lines = self.desk.reply(cmd)
        if lines is not None:
            self.written.append(cmd)
            self.feed(lines)
            return len(data)
        if cmd == "MEM":
            self.desk.vol = [28, 64]
            self.desk.adc = [1, 2]
        return super().write(data)


def test_audio_state_the_soak_leaves_the_desk_as_found(tmp_path: Path) -> None:
    desk = AudioDesk()
    home = desk.state()
    ctx, fake = make_ctx(tmp_path, cdc=DeskMemDevice(desk))
    fake.feed(boot_lines())
    rep = run(ctx, duration_hours=0.4 / 3600, silence_timeout=10.0, probe_interval=0.05)
    assert rep.passed is True
    assert desk.state() == home
    assert rep.data["audio_state_restored"] == [
        "ADC input codec 0: 1 -> 2", "ADC input codec 1: 2 -> 1", "codec volume codec 0: 28 -> 64"]
    assert any("audio_state: restored codec volume codec 0" in line for line in ctx.logs)
