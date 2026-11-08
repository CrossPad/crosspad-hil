"""Scenario `rt_glitch` with the audio layer and the pads faked — no board, no card.

What is under test is the shape of the experiment: the CDC prologue and its one
retry, the three phases in order, the restore on the way out, and the verdict
rule that judges the hammer phase against the sustain phase rather than against
zero. The capture is a WAV this file writes, so "clicks appear under LED load"
is something a test can actually stage.
"""
from __future__ import annotations

import threading
import wave
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from crosspad_hil.cdc import CdcLink
from crosspad_hil.devices import (
    AudioCardInfo,
    Device,
    MidiPortInfo,
    Ports,
    SerialPortInfo,
    UsbMode,
)
from crosspad_hil.errors import CANCELLED, ENV, HilError
from crosspad_hil.scenarios import base, get, rt_glitch
from tests.fakes import AudioDesk, FakeSerial

SR = 48000
CDC_PORT = SerialPortInfo(path="/dev/ttyACM0", vid=0x303A, pid=0x3456, serial="CP-1",
                          product="Crosspad", location="1-2.1")
CONSOLE = SerialPortInfo(path="/dev/ttyACM1", vid=0x0483, pid=0x5740, serial="STM-1",
                         product="CrossPad MIDI+Serial", location="1-2.2")
ESP_MIDI = MidiPortInfo(name="Crosspad MIDI 1", rtmidi_out=0, rtmidi_in=0,
                        alsa_hw="hw:4,0,0", rawmidi=None)
STM_MIDI = MidiPortInfo(name="CrossPad MIDI+Serial 2", rtmidi_out=1, rtmidi_in=1,
                        alsa_hw="hw:5,0,0", rawmidi=None)
UAC2 = AudioCardInfo(name="Crosspad Audio", sounddevice_index=3, alsa_id="Audio")

# Short enough to run in a couple of seconds, long enough that every analysis
# window the scenario cuts (B starts 0.5 s in, both ends trimmed) still exists.
FAST = {"silence_s": 0.6, "sustain_s": 1.0, "hammer_s": 1.0, "hammer_rate": 20.0}


def make_device(mode: UsbMode = UsbMode.DEFAULT) -> Device:
    ports = (Ports(uac2=UAC2, console=CONSOLE, esp_midi=ESP_MIDI, stm_midi=STM_MIDI)
             if mode is UsbMode.AUDIO
             else Ports(cdc=CDC_PORT, console=CONSOLE, esp_midi=ESP_MIDI, stm_midi=STM_MIDI))
    return Device(id="dev_ab12", serial="CP-1", usb_mode=mode, ports=ports)


class RtDevice(FakeSerial):
    """The CDC side of hil_control, only as far as this scenario reaches into it.

    `kit_err` refuses KIT_LOAD (an old firmware without hil_control answers the
    same way); `sampler_starts_on` is the attempt number on which APP_START is
    finally honoured, so the crash-reboot retry can be staged.
    """

    def __init__(self, *, kit_err: bool = False, sampler_starts_on: int = 1,
                 engine_kit: int | None = None, perf_mode: int = 0,
                 perf_button: bool = True, desk: AudioDesk | None = None) -> None:
        super().__init__(timeout=0.01)
        self.desk = desk                      # the mixer/codec verbs; None: a firmware without them
        self.kit_err = kit_err
        self.sampler_starts_on = sampler_starts_on
        self.engine_kit = engine_kit     # None: the engine takes every kit loaded
        self.kit = -1
        self.running: str | None = None
        self.start_attempts = 0
        # The Sampler panel: 0 off, 1 "16 LEVELS: PAD n", 2 "16 PITCH: PAD n" (RAM state).
        # Entry 2 is hidden: the encoder skips it, so a turn count worked out from
        # the group indices overshoots by one -- `ui focus eN` does not land.
        self.perf_mode = perf_mode
        self.perf_button = perf_button
        self.focus_at = 0                     # position among the visible entries
        self.presses = 0
        self.rotations: list[int] = []

    def _panel(self) -> list[tuple[int, str, bool]]:
        """(index, label, hidden) of the encoder group."""
        label = ["16 LEVELS", "16 LEVELS: PAD 6", "16 PITCH: PAD 6"][self.perf_mode]
        rows = [(0, "Back", False), (1, "Kit: DRUMS", False), (2, "Advanced", True),
                (3, "Edit", False)]
        if self.perf_button:
            rows.append((4, label, False))
        return rows

    def _visible(self) -> list[tuple[int, str, bool]]:
        return [r for r in self._panel() if not r[2]]

    def write(self, data: bytes) -> int:
        for raw in data.decode("utf-8", "replace").split("\n"):
            cmd = raw.strip()
            if not cmd:
                continue
            self.written.append(cmd)
            if cmd.startswith("KIT_LOAD "):
                if self.kit_err:
                    self.feed(["ERR bad kit id"])
                else:
                    self.kit = int(cmd.split(" ", 1)[1])
                    self.feed(["OK"])
            elif cmd == "KIT_STATUS":
                name = f"KIT{self.kit}" if self.kit >= 0 else "-"
                self.feed([f"KITSTATUS: current={self.kit} loading=0 pending=-1 name={name}"])
            elif cmd.startswith("APP_START "):
                self.start_attempts += 1
                if self.start_attempts >= self.sampler_starts_on:
                    self.running = cmd.split(" ", 1)[1]
                self.feed(["OK"])
            elif cmd == "SMPL_ENGINE":
                held = self.kit if self.engine_kit is None else self.engine_kit
                self.feed([f"SMPLENG: engine={'native' if held >= 0 else 'none'} state=2 why=- "
                           f"kit={held} pitch=0,0,0 slots=1 complete=1 engines=native"])
            elif cmd == "UI_STATE":
                self.feed(["UI: display=yes touch=yes drawer=0 lcd=80 rgb=60 theme=2 "
                           f"app={self.running or '-'}"])
            elif cmd == "ENC_GROUP":
                rows = self._panel()
                self.feed([f"ENCGROUP: count={len(rows)}",
                           *(f"  [{i}] 0x3fcb{i:04x} {label}" for i, label, _h in rows)])
            elif cmd == "ENC_FOCUS":
                vis = self._visible()
                i, label, _h = vis[self.focus_at % len(vis)]
                self.feed([f"ENCFOCUS: obj=0x3fcb{i:04x} idx={i} text={label}"])
            elif cmd.startswith("ENC_ROTATE "):
                delta = int(cmd.split(" ", 1)[1])
                self.rotations.append(delta)
                self.focus_at = (self.focus_at + delta) % len(self._visible())
                self.feed(["OK"])
            elif cmd.startswith("ENC_PRESS"):
                vis = self._visible()
                if "16 " in vis[self.focus_at % len(vis)][1]:
                    self.perf_mode = (self.perf_mode + 1) % 3
                    self.presses += 1
                self.feed(["OK"])
            elif cmd == "APP_LIST":
                self.feed([f"APPS: Sampler,Mixer running={self.running or '-'}"])
            elif self.desk is not None and (lines := self.desk.reply(cmd)) is not None:
                self.feed(lines)
            else:
                self._rx.append((cmd + "\r\n").encode())
        return len(data)


@dataclass
class FakeResult:
    wav: str
    seconds: float = 3.6
    peak_dbfs: float = -12.0
    rms_dbfs: float = -26.0
    silent: bool = False
    hint: str | None = None

    def to_dict(self) -> dict:
        return {"wav": self.wav, "seconds": self.seconds, "peak_dbfs": self.peak_dbfs,
                "rms_dbfs": self.rms_dbfs, "silent": self.silent,
                "hint": self.hint}


class Bench:
    """Everything the scenario reaches for outside itself, recorded."""

    def __init__(self) -> None:
        self.sessions: list[FakeSession] = []
        self.pads: list[FakePads] = []
        self.sysex: list[bytes] = []
        self.modes: list[UsbMode] = []
        self.start_error: HilError | None = None
        # Sample offsets (in seconds) that get a one-sample pop in the take.
        self.click_times: list[float] = []
        # From when the pads sound in the take (phase B on); None: never.
        self.pads_from_s: float | None = FAST["silence_s"]
        # Until when they sound (the end of B); None: through the end of the take.
        self.pads_until_s: float | None = None
        self.on_press: Any = None


class FakeSession:
    """capture.CaptureSession's contract: start(), wait(), stop() -> a result."""

    def __init__(self, bench: Bench, device: Device, *, seconds: float, preset: str,
                 out: Path, resume_audio_tasks: bool, keep_mode: bool, log: Any) -> None:
        self.bench = bench
        self.device = make_device(UsbMode.AUDIO)
        self.asked_device = device
        self.seconds = seconds
        self.preset = preset
        self.out = Path(out)
        self.resume_audio_tasks = resume_audio_tasks
        self.keep_mode = keep_mode
        self.started = False
        self.waited = False
        self.stopped = False
        self.closed = False
        bench.sessions.append(self)

    def start(self) -> dict:
        if self.bench.start_error is not None:
            raise self.bench.start_error
        self.started = True
        return {"device": self.device.id, "wav": str(self.out), "preset": self.preset}

    def wait(self, timeout_s: float | None = None) -> bool:
        self.waited = True
        return True

    def stop(self) -> FakeResult:
        self.stopped = True
        self._write()
        return FakeResult(wav=str(self.out), seconds=self.seconds)

    def close(self) -> None:
        self.closed = True

    def _write(self) -> None:
        """A faint floor, a 220 Hz tone once the pads sound, and a pop at every
        time the bench asked for."""
        n = int(self.seconds * SR)
        t = np.arange(n) / SR
        x = 0.001 * np.sin(2 * np.pi * 3000 * t)
        if self.bench.pads_from_s is not None:
            on = t >= self.bench.pads_from_s
            if self.bench.pads_until_s is not None:
                on = on & (t < self.bench.pads_until_s)
            x = x + 0.05 * np.sin(2 * np.pi * 220 * t) * on
        for at in self.bench.click_times:
            idx = int(at * SR)
            if 0 <= idx < n:
                x[idx] += 0.8
        pcm = (np.clip(np.column_stack([x, x]), -1.0, 0.999969) * 32768.0).astype("<i2")
        self.out.parent.mkdir(parents=True, exist_ok=True)
        with wave.open(str(self.out), "wb") as w:
            w.setnchannels(2)
            w.setsampwidth(2)
            w.setframerate(SR)
            w.writeframes(pcm.tobytes())


class FakePads:
    """stim.PadTransport: press / release / close, all recorded."""

    name = "sysex_stm"

    def __init__(self, bench: Bench, device: Device) -> None:
        self.bench = bench
        self.device = device
        self.presses: list[tuple[int, int]] = []
        self.releases: list[int] = []
        self.closed = False
        bench.pads.append(self)

    def press(self, pad: int, vel: int) -> None:
        self.presses.append((pad, vel))
        if self.bench.on_press is not None:
            self.bench.on_press()

    def release(self, pad: int) -> None:
        self.releases.append(pad)

    def close(self) -> None:
        self.closed = True


class FakeMidi:
    """MidiIO as a context manager; only send_sysex is exercised."""

    def __init__(self, bench: Bench, device: Device, role: Any) -> None:
        self.bench = bench
        self.role = role

    def __enter__(self) -> FakeMidi:
        return self

    def __exit__(self, *exc: object) -> None:
        return None

    def send_sysex(self, frame: bytes) -> None:
        self.bench.sysex.append(bytes(frame))


@dataclass
class FakeContext(base.Context):
    cdc_fake: RtDevice = None  # type: ignore[assignment]

    def open_cdc(self) -> CdcLink:
        fake = self.cdc_fake
        link = CdcLink(self.device.ports.cdc.path, serial_factory=lambda path, **kw: fake)
        link.open()
        self.opened.append(link)
        return link


@pytest.fixture(autouse=True)
def bench(monkeypatch: pytest.MonkeyPatch) -> Bench:
    """Swap the audio, pad and MIDI layers, and collapse the human-scale waits."""
    b = Bench()
    monkeypatch.setattr(rt_glitch, "CaptureSession",
                        lambda device, **kw: FakeSession(b, device, **kw))
    monkeypatch.setattr(rt_glitch, "open_transport",
                        lambda device, kind: FakePads(b, device))
    monkeypatch.setattr(rt_glitch, "MidiIO",
                        lambda device, role: FakeMidi(b, device, role))

    def set_mode(device: Device, mode: UsbMode, **kw: Any) -> Device:
        b.modes.append(mode)
        return make_device(mode)

    monkeypatch.setattr(rt_glitch, "set_mode", set_mode)
    monkeypatch.setattr(rt_glitch, "KIT_SETTLE_S", 0.0)
    monkeypatch.setattr(rt_glitch, "APP_START_SETTLE_S", 0.0)
    monkeypatch.setattr(rt_glitch, "RETRY_SETTLE_S", 0.0)
    monkeypatch.setattr(rt_glitch, "ROUTE_SETTLE_S", 0.0)
    monkeypatch.setattr(rt_glitch, "SYSEX_GAP_S", 0.0)
    monkeypatch.setattr(rt_glitch, "UI_SETTLE_S", 0.0)
    return b


def make_ctx(tmp_path: Path, mode: UsbMode = UsbMode.DEFAULT, **dev_kw: Any) -> FakeContext:
    return FakeContext(device=make_device(mode), workdir=tmp_path,
                       cancelled=threading.Event(), log=lambda s: None,
                       cdc_fake=RtDevice(**dev_kw))


def run(ctx: FakeContext, **params: Any) -> base.Report:
    scenario = rt_glitch.SCENARIO
    return scenario.run(ctx, rt_glitch.Params(**params), base.Progress())  # type: ignore[arg-type]


def test_registered_and_defaults() -> None:
    assert get("rt_glitch") is rt_glitch.SCENARIO
    p = rt_glitch.Params()
    assert (p.kit, p.pad, p.silence_s, p.sustain_s, p.hammer_s, p.hammer_rate,
            p.keep_mode) == (0, 0, 5.0, 10.0, 15.0, 12.0, False)


def test_clean_baseline_passes_and_restores(tmp_path: Path, bench: Bench) -> None:
    ctx = make_ctx(tmp_path)
    rep = run(ctx, kit=2, **FAST)
    assert rep.passed, rep.summary
    assert rep.exit_code == 0
    d = rep.data
    assert d["verdict"] == "CLEAN"
    assert d["failures"] == []
    assert set(d["phases"]) == {"A_silence", "B_sustain", "C_hammer"}
    assert all(ph["clicks"] == 0 for ph in d["phases"].values())
    assert d["phases"]["B_sustain"]["seconds"] > 0
    assert d["phases"]["C_hammer"]["seconds"] > 0

    # the prologue actually ran, on the CDC link, before the profile switch
    assert ctx.cdc_fake.written[0] == "KIT_LOAD 2"
    assert "APP_START Sampler" in ctx.cdc_fake.written
    assert d["prologue_attempts"] == 1

    # one held pad in B, then a stream of hits in C, all released
    pads = bench.pads[0]
    assert pads.presses[0] == (0, rt_glitch.HAMMER_VELOCITY)
    assert len(pads.presses) > 1
    assert len(pads.releases) == len(pads.presses)
    assert {p for p, _v in pads.presses} <= {0, 1, 2, 3}
    assert pads.closed
    assert d["hits"] == len(pads.presses)

    # the take is an artifact, and the session was stopped (which restores the mode)
    assert [a.role for a in rep.artifacts] == ["capture"]
    assert Path(rep.artifacts[0].path).exists()
    session = bench.sessions[0]
    assert session.preset == rt_glitch.CAPTURE_PRESET and session.resume_audio_tasks
    assert session.seconds == pytest.approx(0.6 + 1.0 + 1.0 + rt_glitch.CAPTURE_TAIL_S)
    assert session.waited and session.stopped

    # routing put back: mixer suspended, ADC to DIFF, mic source codec0, DAC to ALL
    assert [frame.hex(" ").upper() for frame in bench.sysex] == [
        "F0 7D 1D 06 00 F7", "F0 7D 1D 01 01 00 F7",
        "F0 7D 1D 02 00 F7", "F0 7D 1D 03 01 03 F7",
    ]


def test_clicks_under_led_load_fail(tmp_path: Path, bench: Bench) -> None:
    # Phase C runs from silence+sustain to silence+sustain+hammer; put pops in
    # the middle of it, where the scenario's trimmed window is.
    start = FAST["silence_s"] + FAST["sustain_s"]
    bench.click_times = [start + 0.3 + 0.1 * i for i in range(5)]
    ctx = make_ctx(tmp_path)
    rep = run(ctx, **FAST)
    assert not rep.passed
    assert rep.exit_code == 1
    d = rep.data
    assert d["verdict"] == "CLICKS-UNDER-LED-LOAD"
    assert d["phases"]["C_hammer"]["clicks"] > 0
    assert d["phases"]["B_sustain"]["clicks"] == 0
    assert d["phases"]["C_hammer"]["clicks_per_s"] > rt_glitch.CLEAN_MARGIN_PER_S
    assert d["failures"] and "LED path" in d["failures"][0]
    # a failed measurement is still a restored device
    assert bench.sessions[0].stopped and bench.pads[0].closed


def test_clicks_the_sustain_phase_also_has_do_not_fail(tmp_path: Path, bench: Bench) -> None:
    """B is the control, not zero: a take that pops everywhere is not this bug."""
    b_start = FAST["silence_s"]
    c_start = FAST["silence_s"] + FAST["sustain_s"]
    bench.click_times = ([b_start + 0.6 + 0.1 * i for i in range(3)]
                         + [c_start + 0.3 + 0.1 * i for i in range(3)])
    rep = run(make_ctx(tmp_path), **FAST)
    assert rep.data["phases"]["B_sustain"]["clicks"] > 0
    assert rep.data["phases"]["C_hammer"]["clicks"] > 0
    assert rep.passed, rep.summary
    assert rep.data["verdict"] == "CLEAN"


def test_kit_load_refused_is_an_environment_failure(tmp_path: Path, bench: Bench) -> None:
    ctx = make_ctx(tmp_path, kit_err=True)
    rep = run(ctx, **FAST)
    assert rep.exit_code == 2 and not rep.passed
    assert "KIT_LOAD" in rep.summary and "hil_control" in rep.summary
    assert rep.data["prologue_attempts"] == 1     # a refusal is not retried
    assert bench.sessions == []                   # no capture was ever started


def test_sampler_that_never_starts_is_retried_once_then_env(tmp_path: Path,
                                                            bench: Bench) -> None:
    ctx = make_ctx(tmp_path, sampler_starts_on=99)
    rep = run(ctx, **FAST)
    assert rep.exit_code == 2 and not rep.passed
    assert "Sampler not running" in rep.summary
    assert rep.data["prologue_attempts"] == rt_glitch.PROLOGUE_ATTEMPTS
    assert ctx.cdc_fake.written.count("KIT_LOAD 0") == rt_glitch.PROLOGUE_ATTEMPTS


def test_sampler_that_starts_on_the_retry_goes_on(tmp_path: Path, bench: Bench) -> None:
    """The known crash-reboot during sample preload costs an attempt, not the run."""
    ctx = make_ctx(tmp_path, sampler_starts_on=2)
    rep = run(ctx, **FAST)
    assert rep.passed, rep.summary
    assert rep.data["prologue_attempts"] == 2


def test_capture_that_never_starts_is_environment(tmp_path: Path, bench: Bench) -> None:
    bench.start_error = HilError(ENV, "no UAC2 capture endpoint appeared")
    rep = run(make_ctx(tmp_path), **FAST)
    assert rep.exit_code == 2 and not rep.passed
    assert "could not start the capture" in rep.summary
    assert bench.sessions[0].closed        # the session let go of the mode it took
    assert bench.pads == []


def test_a_board_in_audio_mode_is_put_back_first(tmp_path: Path, bench: Bench) -> None:
    ctx = make_ctx(tmp_path, mode=UsbMode.AUDIO)
    ctx.device = make_device(UsbMode.AUDIO)
    rep = run(ctx, **FAST)
    assert rep.passed, rep.summary
    assert bench.modes[0] is UsbMode.DEFAULT
    assert ctx.cdc_fake.written[0] == "KIT_LOAD 0"


def test_cancel_before_the_run_raises_and_touches_nothing(tmp_path: Path,
                                                          bench: Bench) -> None:
    ctx = make_ctx(tmp_path)
    ctx.cancelled.set()
    with pytest.raises(HilError) as ei:
        run(ctx, **FAST)
    assert ei.value.code == CANCELLED
    assert bench.sessions == [] and bench.pads == []


def test_cancel_mid_phase_still_restores_everything(tmp_path: Path, bench: Bench) -> None:
    ctx = make_ctx(tmp_path)
    bench.on_press = ctx.cancelled.set          # cancel on the first pad of phase B
    with pytest.raises(HilError) as ei:
        run(ctx, **FAST)
    assert ei.value.code == CANCELLED
    pads = bench.pads[0]
    assert pads.releases == [0]                 # the held pad was let go
    assert pads.closed
    assert bench.sessions[0].stopped            # profile restored
    assert len(bench.sysex) == 4                # routing restored


def test_a_take_the_host_lost_seconds_of_is_not_scored(tmp_path: Path, bench: Bench,
                                                       monkeypatch: pytest.MonkeyPatch) -> None:
    stop = FakeSession.stop

    def lossy(self: FakeSession) -> FakeResult:
        result = stop(self)
        result.health = {"ok": False, "missed_s": 0.0, "max_gap_ms": 900.0, "overruns": 2}
        return result

    monkeypatch.setattr(FakeSession, "stop", lossy)
    start = FAST["silence_s"] + FAST["sustain_s"]
    bench.click_times = [start + 0.3 + 0.1 * i for i in range(5)]   # holes, not the LED path
    rep = run(make_ctx(tmp_path), **FAST)
    assert not rep.passed and rep.exit_code == 2
    assert "900 ms gap" in rep.summary and "2 host overrun(s)" in rep.summary
    assert rep.data["verdict"] is None and rep.data["phases"] == {}
    assert bench.sessions[0].stopped


def test_a_take_where_the_pads_are_never_heard_is_silent(tmp_path: Path,
                                                         bench: Bench) -> None:
    """2026-10-09: A, B and C all at -40.4 dBFS and the verdict was CLEAN. A
    held pad that is not above the floor means the take recorded something else."""
    bench.pads_from_s = None
    ctx = make_ctx(tmp_path)
    rep = run(ctx, **FAST)
    assert not rep.passed
    assert rep.exit_code == 2
    assert rep.data["verdict"] == "SILENT"
    assert "SILENT" in rep.summary and "B (sustain)" in rep.summary
    assert rep.data["failures"]


def test_a_capture_flagged_silent_is_silent_even_when_the_levels_look_fine(
        tmp_path: Path, bench: Bench, monkeypatch: pytest.MonkeyPatch) -> None:
    stop = FakeSession.stop

    def silent(self: FakeSession) -> FakeResult:
        result = stop(self)
        result.silent = True
        result.hint = "nothing on the capture endpoint"
        return result

    monkeypatch.setattr(FakeSession, "stop", silent)
    rep = run(make_ctx(tmp_path), **FAST)
    assert not rep.passed and rep.exit_code == 2
    assert rep.data["verdict"] == "SILENT"
    assert "the capture reports silence" in rep.summary
    assert "nothing on the capture endpoint" in rep.summary


def test_a_hammer_phase_that_is_not_heard_is_silent(tmp_path: Path, bench: Bench) -> None:
    """B sounds, C does not: the storm is what has to be heard for its clicks to count."""
    bench.pads_until_s = FAST["silence_s"] + FAST["sustain_s"]
    rep = run(make_ctx(tmp_path), **FAST)
    assert not rep.passed and rep.exit_code == 2
    assert rep.data["verdict"] == "SILENT"
    assert "C (hammer)" in rep.summary and "B (sustain)" not in rep.summary


A = -60.0
OVER = rt_glitch.MIN_PHASE_ABOVE_FLOOR_DB


@pytest.mark.parametrize("silent, b, c, ok", [
    (False, A + OVER, A + OVER + 5, True),          # exactly at the margin passes
    (False, A + OVER - 0.1, A + OVER + 5, False),   # B just under
    (False, A + OVER + 5, A + OVER - 0.1, False),   # C just under
    (False, A, A, False),                           # 2026-10-09: all equal
    (True, A + 40, A + 40, False),                  # the capture's own silent flag
])
def test_level_problem(silent: bool, b: float, c: float, ok: bool) -> None:
    why = rt_glitch.level_problem(silent, A, b, c)
    assert (why is None) is ok, why


def test_level_problem_names_every_failing_phase() -> None:
    why = rt_glitch.level_problem(True, A, A, A)
    assert why is not None
    assert "capture reports silence" in why and "B (sustain)" in why and "C (hammer)" in why


# --- Sampler performance modes ----------------------------------------------

@pytest.mark.parametrize("label, on", [
    ("16 LEVELS", False), ("16 LEVELS: PAD 6", True), ("16 PITCH: PAD 12", True),
    ("16 PITCH", False), ("16 levels: pad 1", True),
])
def test_perf_button_reads_the_label(label: str, on: bool) -> None:
    btn = rt_glitch.perf_button({"group": [{"ref": "e0", "label": "Back"},
                                           {"ref": "e1", "label": label}]})
    assert btn is not None and btn["ref"] == "e1" and btn["on"] is on


def test_perf_button_absent() -> None:
    assert rt_glitch.perf_button({"group": [{"ref": "e0", "label": "Back"}]}) is None
    assert rt_glitch.perf_button(None) is None


@pytest.mark.parametrize("mode, presses", [(1, 2), (2, 1)])
def test_a_performance_mode_left_on_is_switched_off_before_the_take(
        tmp_path: Path, bench: Bench, mode: int, presses: int) -> None:
    logs: list[str] = []
    ctx = make_ctx(tmp_path, perf_mode=mode)
    ctx.log = logs.append
    rep = run(ctx, **FAST)
    assert rep.passed, rep.summary
    fake = ctx.cdc_fake
    assert fake.perf_mode == 0
    assert fake.presses == presses
    assert rep.data["performance_mode"]["presses"] == presses
    assert rep.data["performance_mode"]["after"] == "16 LEVELS"
    assert any("performance mode was on" in line and "16" in line for line in logs)
    # the hidden entry makes the first guess overshoot; the walk still got there
    assert fake.rotations[0] == 4 and len(fake.rotations) > 1
    assert bench.sessions[0].started


def test_performance_modes_already_off_are_left_alone(tmp_path: Path, bench: Bench) -> None:
    ctx = make_ctx(tmp_path)
    rep = run(ctx, **FAST)
    assert rep.passed
    assert ctx.cdc_fake.presses == 0
    assert not any(c.startswith("ENC_PRESS") for c in ctx.cdc_fake.written)
    assert rep.data["performance_mode"] == {"found": True, "presses": 0,
                                            "before": "16 LEVELS", "after": "16 LEVELS"}


def test_a_missing_performance_button_warns_and_goes_on(tmp_path: Path, bench: Bench) -> None:
    logs: list[str] = []
    ctx = make_ctx(tmp_path, perf_button=False)
    ctx.log = logs.append
    rep = run(ctx, **FAST)
    assert rep.passed, rep.summary
    assert rep.data["performance_mode"]["found"] is False
    assert any("WARNING" in line and "performance" in line for line in logs)
    assert not any(c.startswith("ENC_PRESS") for c in ctx.cdc_fake.written)


def test_a_sampler_whose_engine_holds_another_kit_is_retried_then_env(tmp_path: Path,
                                                                      bench: Bench) -> None:
    ctx = make_ctx(tmp_path, engine_kit=-1)
    rep = run(ctx, kit=2, **FAST)
    assert not rep.passed
    assert rep.exit_code == 2
    assert rep.data["prologue_attempts"] == rt_glitch.PROLOGUE_ATTEMPTS
    assert rep.data["engine_kit"]["kit"] == -1
    assert bench.sessions == []          # nothing was recorded


def test_the_prologue_checks_the_engine_holds_the_kit(tmp_path: Path, bench: Bench) -> None:
    ctx = make_ctx(tmp_path)
    rep = run(ctx, kit=2, **FAST)
    assert rep.passed, rep.summary
    assert rep.data["engine_kit"]["kit"] == 2
    assert "SMPL_ENGINE" in ctx.cdc_fake.written


# --- mixer / codec state ----------------------------------------------------------

def test_a_healthy_desk_is_left_alone(tmp_path: Path, bench: Bench) -> None:
    desk = AudioDesk()
    rep = run(make_ctx(tmp_path, desk=desk), **FAST)
    assert rep.passed, rep.summary
    assert rep.data["audio_state_fixed"] == [] and desk.commands == []


def test_a_board_that_will_not_say_is_measured_as_before(tmp_path: Path, bench: Bench) -> None:
    rep = run(make_ctx(tmp_path), **FAST)                 # no desk: MIX_LVL / AUDIO_STATUS echo
    assert rep.passed, rep.summary
    assert rep.data["audio_state_fixed"] == []
    assert rep.data["audio_state_unread"] == ["mixer (MIX_LVL)", "codec volumes (AUDIO_STATUS)"]


def test_a_low_codec_volume_and_a_dead_send_are_fixed_and_recorded(
        tmp_path: Path, bench: Bench) -> None:
    desk = AudioDesk()
    desk.vol = [28, 50]
    desk.channels[0]["route"] = [0, 0, 0]
    logged: list[str] = []
    ctx = make_ctx(tmp_path, desk=desk)
    ctx.log = logged.append
    rep = run(ctx, **FAST)
    assert rep.passed, rep.summary
    assert desk.vol == [64, 50] and desk.channels[0]["route"] == [1, 1, 1]
    fixed = rep.data["audio_state_fixed"]
    assert len(fixed) == 4 and any("codec 0 output volume 28" in f for f in fixed)
    assert any(line.startswith("audio_state: fixed: ") for line in logged)
    assert bench.sessions


def test_a_muted_sampler_is_unmuted_and_measured(tmp_path: Path, bench: Bench) -> None:
    desk = AudioDesk()
    desk.channels[0]["mute"] = 1
    desk.vol = [28, 64]
    logged: list[str] = []
    ctx = make_ctx(tmp_path, desk=desk)
    ctx.log = logged.append
    rep = run(ctx, **FAST)
    assert rep.passed, rep.summary
    assert desk.channels[0]["mute"] == 0                  # MIX_MUTE 0 0
    assert "MIX_MUTE 0 0" in desk.commands
    fixed = rep.data["audio_state_fixed"]
    assert any("'Sampler' was muted" in f for f in fixed) and len(fixed) == 2
    assert any(line.startswith("audio_state: fixed: mixer channel 0 'Sampler' was muted") for line in logged)
    assert bench.sessions


def test_a_mute_the_firmware_cannot_lift_is_misconfigured(tmp_path: Path, bench: Bench) -> None:
    """A firmware before MIX_MUTE echoes it: the board is known mute, so not measured."""
    desk = AudioDesk(no_mix_verbs=True)
    desk.channels[0]["mute"] = 1
    rep = run(make_ctx(tmp_path, desk=desk), **FAST)
    assert not rep.passed and rep.exit_code == 2
    assert rep.data["verdict"] == "MISCONFIGURED"
    assert "MISCONFIGURED" in rep.summary and "'Sampler' was muted" in rep.summary
    assert rep.data["audio_state_unfixable"]
    assert bench.sessions == []
    assert desk.channels[0]["mute"] == 1


def test_a_mute_that_does_not_stick_is_misconfigured(tmp_path: Path, bench: Bench) -> None:
    desk = AudioDesk()
    desk.channels[0]["mute"] = 1
    real = desk.reply

    def sticky(cmd: str):
        out = real(cmd)
        if cmd.startswith("MIX_MUTE"):
            desk.channels[0]["mute"] = 1                  # the board answers OK, keeps it muted
        return out
    desk.reply = sticky  # type: ignore[method-assign]
    rep = run(make_ctx(tmp_path, desk=desk), **FAST)
    assert rep.exit_code == 2 and rep.data["verdict"] == "MISCONFIGURED"
    assert any("still wrong after MIX_MUTE 0 0" in u for u in rep.data["audio_state_unfixable"])
    assert bench.sessions == []


def test_a_fix_the_board_refuses_is_misconfigured(tmp_path: Path, bench: Bench) -> None:
    desk = AudioDesk(refuse=("AUDIO_VOL",))
    desk.vol = [28, 64]
    rep = run(make_ctx(tmp_path, desk=desk), **FAST)
    assert rep.exit_code == 2 and rep.data["verdict"] == "MISCONFIGURED"
    assert bench.sessions == []
