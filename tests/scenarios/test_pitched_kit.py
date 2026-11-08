"""Scenario `pitched_kit`: the pure kernels, and the run against fakes.

Companion to `sampler_record`'s test file and built the same way: the fake CDC
device models the parts of hil_control.cpp the setup leans on (kits, the
launcher, `PITCHED_STATUS`); the capture session, the pad transport and the
frequency analysis are replaced at their module seams
(`open_capture`/`open_pad_transport`/`analyze_pitch`/`analyze_sine`/
`read_wav_ch`), so what a fake-device test exercises is the scenario's own
logic — the order of the setup (everything over CDC *before* the profile
switch), the offset re-basing onto the WAV's own t=0, the capture-health
short-circuit, and the promise that the capture is stopped (and the profile
restored) whatever goes wrong.

Below that: the frequency-analysis kernel (`analyze.analyze_pitch`) against a
synthetic WAV built the same way the scenario slices one out of a real
capture, the `PITCHED:` status-line parser (covered in full in
`test_parsers_cdc.py`; only the shape this scenario reads is re-asserted
here), the chord-scoring floor rule, and the verdict logic (`pitch_verdict`)
against fabricated result sets — none of which need a board.
"""
from __future__ import annotations

import math
import threading
import time
import wave
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from crosspad_hil import cdc as cdc_mod
from crosspad_hil.cdc import CdcLink
from crosspad_hil.devices import AudioCardInfo, Device, MidiPortInfo, Ports, SerialPortInfo, UsbMode
from crosspad_hil.errors import ENV, HilError
from crosspad_hil.parsers import parse_cdc_reply
from crosspad_hil.scenarios import base, get, pitched_kit
from tests.fakes import FakeSerial
from tests.test_analyze import SR, tone, write_wav

KITS = ["Basic", "HIL PITCHED", "Perc"]
CDC_PORT = SerialPortInfo(path="/dev/ttyACM0", vid=0x303A, pid=0x3456,
                          serial="CP-1", product="CrossPad", location="1-2.1")
CONSOLE_PORT = SerialPortInfo(path="/dev/ttyACM1", vid=0x0483, pid=0x5740,
                              serial="STM-1", product="CrossPad", location="1-2.2")
ESP_MIDI = MidiPortInfo(name="Crosspad MIDI 1", rtmidi_out=2, rtmidi_in=2,
                        alsa_hw="hw:4,0,0", rawmidi="/dev/snd/midiC4D0")
UAC2 = AudioCardInfo(name="Crosspad Audio", sounddevice_index=5, alsa_id="hw:5")

GOOD_PITCHED_LINE = ("PITCHED: active=1 zones=2 roots=57.00,69.00 voices=2/24 "
                    "bank_kb=180 rt_max_us=800 steals=0")


# --------------------------------------------------------------------------- fakes


class PitchedDevice(FakeSerial):
    """CDC endpoint: kits, the launcher, and PITCHED_STATUS."""

    def __init__(self) -> None:
        super().__init__(timeout=0.01)
        self.current = 0
        self.apps = ["Sampler", "Mixer"]
        self.running: str | None = "Sampler"
        self.status_line = GOOD_PITCHED_LINE
        self.ingain = {"IN1": 24, "IN2": 0}    # the jack at 0 dB, as dcfac16 brings it up

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
            elif cmd == "KIT_LIST":
                # A batch line plus the "KITS: end" trailer, matching the real
                # multi-line shape — a single untrailed line leaves
                # transact_multi's waiter without a "done", which times out
                # at 3s per call instead of returning immediately.
                body = ",".join(f"{i}:{n}" for i, n in enumerate(KITS))
                self.feed([f"KITS: {body}",
                           f"KITS: end count={len(KITS)} current={self.current}"])
            elif cmd == "KIT_STATUS":
                name = KITS[self.current] if self.current >= 0 else "-"
                self.feed([f"KITSTATUS: current={self.current} loading=0 pending=-1 name={name}"])
            elif cmd.startswith("KIT_LOAD "):
                self.current = int(cmd.split(" ", 1)[1])
                self.feed(["OK"])
            elif cmd == "PITCHED_STATUS":
                self.feed([self.status_line])
            elif cmd.startswith("INGAIN"):
                parts = cmd.split()
                if len(parts) == 3:
                    self.ingain[f"IN{int(parts[1]) + 1}"] = int(parts[2])
                self.feed([f"INGAIN: IN1={self.ingain['IN1']} IN2={self.ingain['IN2']}"])
            else:
                self._rx.append((cmd + "\r\n").encode())
        return len(data)


def device_in(mode: UsbMode) -> Device:
    if mode is UsbMode.AUDIO:
        ports = Ports(cdc=None, console=CONSOLE_PORT, esp_midi=ESP_MIDI, uac2=UAC2)
    else:
        ports = Ports(cdc=CDC_PORT, console=CONSOLE_PORT, esp_midi=ESP_MIDI, uac2=None)
    return Device(id="dev_ab12", serial="CP-1", usb_mode=mode, ports=ports)


@dataclass
class FakeContext(base.Context):
    cdc_fake: PitchedDevice = None  # type: ignore[assignment]

    def open_cdc(self) -> CdcLink:
        fake = self.cdc_fake
        link = CdcLink(self.device.ports.cdc.path if self.device.ports.cdc else "fake",
                       serial_factory=lambda path, **kw: fake)
        link.open()
        self.opened.append(link)
        return link


@dataclass
class FakeResult:
    """What crosspad_hil.capture.CaptureResult gives the scenario back."""

    wav: str
    seconds: float = 2.0
    peak_dbfs: float = -6.0
    silent: bool = False
    hint: str | None = None
    health: dict = field(default_factory=lambda: {"ok": True, "detail": "clean"})

    def to_dict(self) -> dict:
        return {"wav": self.wav, "seconds": self.seconds, "peak_dbfs": self.peak_dbfs,
                "silent": self.silent, "hint": self.hint}


WAV_SECONDS = 3.0   # comfortably longer than any schedule the tests below build


class FakeCapture:
    """Stands in for CaptureSession: the profile switch, a real (silent) WAV, the restore."""

    def __init__(self, device: Device, **kw: Any) -> None:
        self.device = device
        self.kw = kw
        self.started = False
        self.stopped = False
        self.started_at = 0.0
        self.start_error: HilError | None = None
        self.result_kw: dict[str, Any] = {}

    def start(self) -> dict:
        if self.start_error is not None:
            raise self.start_error
        self.started = True
        self.started_at = time.monotonic()
        path = Path(self.kw["out"])
        path.parent.mkdir(parents=True, exist_ok=True)
        with wave.open(str(path), "wb") as w:
            w.setnchannels(1)
            w.setsampwidth(2)
            w.setframerate(SR)
            w.writeframes(b"\x00\x00" * int(SR * WAV_SECONDS))
        self.device = device_in(UsbMode.AUDIO)
        return {"wav": str(path), "preset": self.kw["preset"], "backend": "fake"}

    def stop(self) -> FakeResult:
        self.stopped = True
        self.device = device_in(UsbMode.DEFAULT)
        return FakeResult(wav=str(self.kw["out"]), **self.result_kw)


class FakeTransport:
    """A pad transport that records what the schedule asked for."""

    name = "fake"

    def __init__(self, fail_on_press: bool = False) -> None:
        self.presses: list[tuple[int, int]] = []
        self.releases: list[int] = []
        self.closed = False
        self.fail_on_press = fail_on_press

    def press(self, pad: int, vel: int) -> None:
        if self.fail_on_press:
            raise OSError("the device left the bus")
        self.presses.append((pad, vel))

    def release(self, pad: int) -> None:
        self.releases.append(pad)

    def close(self) -> None:
        self.closed = True


def _pad_from_hz(hz: float, base_note: int) -> int:
    """Invert `expected_hz`: the fake pitch kernel gets a frequency, not a pad index."""
    note = 69 + 12 * math.log2(hz / 440.0)
    return round(note - base_note)


class FakeAnalyze:
    """Stands in for `analyze_pitch`/`analyze_sine`/`read_wav_ch`: no real DSP.

    `bad_pads` maps a pad index to the `cents_vs` its per-pad score should
    report (default 0.0 — dead in tune). `bad_chord_pads` does the same for
    chord-peak level, expressed as dB *below* the healthy default, so a test
    can push one pad's peak under the noise-floor margin without touching the
    others.
    """

    def __init__(self, base_note: int = 60) -> None:
        self.base_note = base_note
        self.bad_pads: dict[int, float] = {}
        self.bad_chord_pads: dict[int, float] = {}
        self.pitch_calls: list[dict[str, Any]] = []
        self.sine_calls = 0
        self.sine_floors: list[float] = []

    def read_wav_ch(self, path: Any, channel: Any) -> tuple[Any, int, int]:
        return np.zeros(int(SR * WAV_SECONDS)), SR, 0

    def analyze_pitch(self, x: Any, sr: int, t0: float, t1: float, *,
                      expected_hz: float | None = None,
                      search_hz: tuple[float, float] | None = None) -> dict:
        self.pitch_calls.append({"t0": t0, "t1": t1, "expected_hz": expected_hz,
                                 "search_hz": search_hz})
        if search_hz is not None:                      # a chord peak
            pad = _pad_from_hz(expected_hz, self.base_note)
            level = -10.0 - self.bad_chord_pads.get(pad, 0.0)
            return {"hz": expected_hz, "expected_hz": expected_hz, "cents_vs": 0.0,
                    "level_dbfs": level, "peak_db": level + 5.0, "confidence": 0.9,
                    "t0": t0, "t1": t1, "sample_rate": sr, "bin_hz": 10.0}
        if expected_hz is not None:                     # a per-pad score
            pad = _pad_from_hz(expected_hz, self.base_note)
            cents = self.bad_pads.get(pad, 0.0)
            return {"hz": expected_hz, "expected_hz": expected_hz, "cents_vs": cents,
                    "level_dbfs": -10.0, "peak_db": -5.0, "confidence": 0.9,
                    "t0": t0, "t1": t1, "sample_rate": sr, "bin_hz": 10.0}
        # the pre-chord noise floor: quiet, so a healthy chord peak clears it
        return {"hz": 0.0, "expected_hz": None, "cents_vs": None, "level_dbfs": -60.0,
                "peak_db": -50.0, "confidence": 0.1, "t0": t0, "t1": t1,
                "sample_rate": sr, "bin_hz": 10.0}

    def analyze_sine(self, path: Any, *, freq: float, edge_s: float,
                     min_rms_dbfs: float) -> dict:
        self.sine_calls += 1
        self.sine_floors.append(min_rms_dbfs)
        return {"glitches": 0, "timing_slips": 0}


@pytest.fixture
def fast(monkeypatch: pytest.MonkeyPatch) -> None:
    """Collapse every human-scale wait the scenario carries.

    `_run` threads these module constants into `build_schedule` and `_wait` by
    bare name, so patching them here reaches both: the whole pad/chord
    schedule collapses onto (near) t=0 and every `ctx.cancelled.wait()` the
    scenario makes returns instantly, without changing what is scored. Used
    by `bench` only — the pure `build_schedule` tests below call it directly
    and assert against the real `LEAD_S`/`CHORD_*` defaults, which live in
    `build_schedule`'s own signature and would not see this patch anyway.
    """
    for name in ("LEAD_S", "STREAM_SETTLE_S", "TAIL_S", "APP_START_SETTLE_S",
                 "CHORD_HOLD_S", "CHORD_SPREAD_S", "CHORD_STEADY_MARGIN_S"):
        monkeypatch.setattr(pitched_kit, name, 0.0)
    # …and the settle a real CDC endpoint needs after enumerating, twice per run.
    monkeypatch.setattr(cdc_mod, "OPEN_SETTLE_S", 0.0)


@pytest.fixture
def bench(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, fast: None):
    """Context + the four seams, all wired to fakes."""

    class Bench:
        def __init__(self) -> None:
            self.cdc = PitchedDevice()
            self.ctx = FakeContext(device=device_in(UsbMode.DEFAULT), workdir=tmp_path,
                                   cancelled=threading.Event(), log=lambda s: None,
                                   cdc_fake=self.cdc)
            self.capture: FakeCapture | None = None
            self.capture_error: HilError | None = None
            self.transport = FakeTransport()
            self.analyze = FakeAnalyze()

        def make_capture(self, device: Device, **kw: Any) -> FakeCapture:
            self.capture = FakeCapture(device, **kw)
            if self.capture_error is not None:
                self.capture.start_error = self.capture_error
            return self.capture

        def make_transport(self, device: Device, kind: str) -> FakeTransport:
            self.transport.kind = kind  # type: ignore[attr-defined]
            return self.transport

        def run(self, **params: Any) -> base.Report:
            return pitched_kit.SCENARIO.run(
                self.ctx, pitched_kit.Params(**params), base.Progress())  # type: ignore[arg-type]

    b = Bench()
    monkeypatch.setattr(pitched_kit, "open_capture", b.make_capture)
    monkeypatch.setattr(pitched_kit, "open_pad_transport", b.make_transport)
    monkeypatch.setattr(pitched_kit, "read_wav_ch", b.analyze.read_wav_ch)
    monkeypatch.setattr(pitched_kit, "analyze_pitch", b.analyze.analyze_pitch)
    monkeypatch.setattr(pitched_kit, "analyze_sine", b.analyze.analyze_sine)
    return b


FAST_PARAMS: dict[str, Any] = {"pads": [0, 1], "hold": 0.05, "gap": 0.01}


# --------------------------------------------------------------------------- end-to-end


def test_registered() -> None:
    assert get("pitched_kit") is pitched_kit.SCENARIO
    assert pitched_kit.SCENARIO.name == "pitched_kit"


def test_happy_path_passes_and_scores_pads_and_chord(bench) -> None:
    rep = bench.run(**FAST_PARAMS)
    assert rep.passed, rep.summary
    assert rep.exit_code == 0
    d = rep.data
    assert d["kit_id"] == 1 and d["kit_name"] == "HIL PITCHED"
    assert [r["pad"] for r in d["pitch"]] == [0, 1]
    assert all(r["pass"] for r in d["pitch"])
    assert d["chord"]["found"] == d["chord"]["expected"] == 2
    assert d["status"]["rt_max_us"] == 800 and d["status"]["steals"] == 0
    # the setup happened over CDC before the profile switch
    assert "KIT_LOAD 1" in bench.cdc.written
    assert bench.cdc.written.index("KIT_LOAD 1") < len(bench.cdc.written) - 1
    assert bench.capture.started and bench.capture.stopped
    assert bench.transport.closed
    assert [a.role for a in rep.artifacts] == ["capture"]
    assert Path(rep.artifacts[0].path).is_file()


def test_sine_check_runs_when_the_hold_clears_the_slice_floor(bench) -> None:
    # pitch_window trims 0.05 s off each end (ATTACK_MARGIN_S/RELEASE_MARGIN_S);
    # the sine check itself needs >= 0.1 s of slice, so the hold has to clear both.
    rep = bench.run(pads=[0], hold=0.25, gap=0.02, chord=False)
    assert rep.passed, rep.summary
    assert bench.analyze.sine_calls == 1


def test_setup_happens_entirely_before_the_profile_switch(bench) -> None:
    bench.run(**FAST_PARAMS)
    switch_index = len(bench.cdc.written)   # nothing more is written until PITCHED_STATUS
    assert "KIT_LIST" in bench.cdc.written[:switch_index]
    assert "APP_LIST" in bench.cdc.written[:switch_index]


def test_an_unknown_kit_name_lists_what_is_there(bench) -> None:
    rep = bench.run(kit_name="nope", **{k: v for k, v in FAST_PARAMS.items() if k != "pads"})
    assert not rep.passed and rep.exit_code == 2
    assert "nope" in rep.summary
    assert "HIL PITCHED" in rep.summary
    assert bench.capture is None   # nothing was recorded, nothing to restore


def test_no_sampler_in_the_launcher_is_exit_2(bench) -> None:
    bench.cdc.apps = ["Mixer"]
    bench.cdc.running = None
    rep = bench.run(**FAST_PARAMS)
    assert not rep.passed and rep.exit_code == 2
    assert "Sampler" in rep.summary
    assert bench.capture is None


def test_a_pad_outside_tolerance_fails_and_names_it(bench) -> None:
    bench.analyze.bad_pads[1] = 50.0
    rep = bench.run(**FAST_PARAMS)
    assert not rep.passed and rep.exit_code == 1
    assert "[1]" in rep.summary
    assert bench.capture.stopped   # the failure is scored, not a bench abort


def test_a_quiet_chord_pad_fails_the_chord_but_not_the_run_setup(bench) -> None:
    # 45 dB down clears neither rule: -40 dB relative to the loudest peak, nor
    # 20 dB above the pre-chord noise floor (both checked in `_score_chord`).
    bench.analyze.bad_chord_pads[1] = 45.0
    rep = bench.run(**FAST_PARAMS)
    assert not rep.passed and rep.exit_code == 1
    assert "chord: only 1/2" in rep.summary


def test_a_silent_capture_is_an_environment_problem_not_16_failed_pads(bench) -> None:
    def silent(device: Device, **kw: Any) -> FakeCapture:
        cap = bench.make_capture(device, **kw)
        cap.result_kw = {"silent": True, "hint": "check the capture route"}
        return cap

    pitched_kit.open_capture = silent  # type: ignore[assignment]
    try:
        rep = bench.run(**FAST_PARAMS)
    finally:
        pitched_kit.open_capture = bench.make_capture  # type: ignore[assignment]
    assert not rep.passed and rep.exit_code == 2
    assert "check the capture route" in rep.summary
    assert bench.analyze.pitch_calls == []   # nothing was scored against it


def test_an_unhealthy_capture_is_also_an_environment_problem(bench) -> None:
    def stalled(device: Device, **kw: Any) -> FakeCapture:
        cap = bench.make_capture(device, **kw)
        cap.result_kw = {"health": {"ok": False, "detail": "captured 0.5 s of 2.0 s"}}
        return cap

    pitched_kit.open_capture = stalled  # type: ignore[assignment]
    try:
        rep = bench.run(**FAST_PARAMS)
    finally:
        pitched_kit.open_capture = bench.make_capture  # type: ignore[assignment]
    assert not rep.passed and rep.exit_code == 2
    assert "capture unhealthy" in rep.summary


def test_pitched_status_failure_fails_the_run_and_says_why(bench) -> None:
    bench.cdc.status_line = "PITCHED: garbage"   # unparseable -> NOT_SUPPORTED
    rep = bench.run(**FAST_PARAMS)
    assert not rep.passed and rep.exit_code == 1
    assert "PITCHED_STATUS unavailable" in rep.summary
    assert any("PITCHED_STATUS" in f for f in rep.data["failures"])


def test_a_refused_pad_message_is_counted_not_raised(bench) -> None:
    bench.transport.fail_on_press = True
    rep = bench.run(**FAST_PARAMS)
    assert bench.capture.stopped   # the take survives, and so does the profile restore
    assert rep.data["send_failures"] > 0


def test_the_audio_profile_is_left_when_the_run_starts_in_it(bench, monkeypatch) -> None:
    """No CDC means the board is already in the audio profile — SysEx is the way back."""
    calls: list[tuple[str, str]] = []

    def fake_set_mode(device: Device, mode: UsbMode, **kw: Any) -> Device:
        calls.append((device.id, mode.value))
        return device_in(mode)

    monkeypatch.setattr(pitched_kit, "set_mode", fake_set_mode)
    bench.ctx.device = device_in(UsbMode.AUDIO)
    rep = bench.run(**FAST_PARAMS)
    assert rep.passed, rep.summary
    assert calls == [("dev_ab12", "default")]


def test_pads_must_be_in_range(bench) -> None:
    with pytest.raises(HilError) as ei:
        bench.run(pads=[0, 16])
    assert ei.value.code == "BAD_ARGS"
    assert bench.capture is None


# --------------------------------------------------------------------------- expected_hz


def test_expected_hz_matches_the_fixture_kit_formula() -> None:
    # pad_notes[pad] = base_note + pad; A4 (69) = 440 Hz. Pad 9 with base_note=60
    # is note 69 exactly, on both defaults.
    assert pitched_kit.expected_hz(9) == pytest.approx(440.0)
    assert pitched_kit.expected_hz(0) == pytest.approx(440.0 * 2 ** ((60 - 69) / 12))
    assert pitched_kit.expected_hz(0, base_note=48) == pytest.approx(440.0 * 2 ** ((48 - 69) / 12))


# --------------------------------------------------------------------------- build_schedule


def test_build_schedule_lays_out_pads_back_to_back() -> None:
    events, chord, total = pitched_kit.build_schedule([0, 1, 2], hold=0.5, gap=0.2, chord=False)
    assert chord is None
    assert [e["pad"] for e in events] == [0, 1, 2]
    assert events[0]["t_press"] == pytest.approx(pitched_kit.LEAD_S)
    assert events[0]["t_release"] == pytest.approx(pitched_kit.LEAD_S + 0.5)
    # each pad starts hold+gap after the previous one's press
    assert events[1]["t_press"] == pytest.approx(events[0]["t_press"] + 0.7)
    assert events[2]["t_press"] == pytest.approx(events[1]["t_press"] + 0.7)
    assert total == pytest.approx(events[2]["t_release"] + 0.2 + pitched_kit.TAIL_S)


def test_build_schedule_chord_spreads_presses_and_trims_the_steady_window() -> None:
    events, chord, total = pitched_kit.build_schedule([0, 1, 2, 3], hold=0.3, gap=0.1, chord=True)
    assert chord is not None
    assert chord["pads"] == [0, 1, 2, 3]
    assert len(chord["t_press"]) == 4
    # every press lands inside the configured spread window
    spread = chord["t_press"][-1] - chord["t_press"][0]
    assert spread == pytest.approx(pitched_kit.CHORD_SPREAD_S, abs=1e-6)
    assert chord["t_release"] > chord["t_press"][-1]
    assert chord["steady_start"] < chord["steady_end"]
    # the steady window sits strictly inside the hold, trimmed both ends
    assert chord["steady_start"] > chord["t_press"][-1]
    assert chord["steady_end"] < chord["t_release"]
    assert total > chord["t_release"]


def test_build_schedule_single_pad_chord_has_no_spread() -> None:
    _events, chord, _total = pitched_kit.build_schedule([5], hold=0.3, gap=0.1, chord=True)
    assert chord["t_press"] == chord["t_press"][:1]  # exactly one press time
    assert len(chord["t_press"]) == 1


def test_pitch_window_trims_attack_and_release() -> None:
    ev = {"t_press": 1.0, "t_release": 1.6}
    t0, t1 = pitched_kit.pitch_window(ev)
    assert t0 == pytest.approx(1.0 + pitched_kit.ATTACK_MARGIN_S)
    assert t1 == pytest.approx(1.6 - pitched_kit.RELEASE_MARGIN_S)


def test_pitch_window_falls_back_when_the_hold_is_too_short() -> None:
    ev = {"t_press": 1.0, "t_release": 1.0 + pitched_kit.ATTACK_MARGIN_S / 2}
    t0, t1 = pitched_kit.pitch_window(ev)
    assert (t0, t1) == (ev["t_press"], ev["t_release"])


# --------------------------------------------------------------------------- t_ref/started_at offset


class _FakeSession:
    """Just enough of `CaptureSession` for the offset math: a `started_at` set once."""

    def __init__(self, started_at: float) -> None:
        self.started_at = started_at


def test_shift_events_and_chord_rebase_onto_session_started_at() -> None:
    """`t_ref = time.monotonic()` is taken after `session.start()` and the settle
    wait, but the windows handed to `analyze_pitch` are relative to the WAV's own
    t=0 — `session.started_at`, set earlier when the stream first came up. The
    scenario computes `offset = t_ref - session.started_at` and must apply it to
    every window, pad and chord alike, or every score reads a slice that starts
    too early by exactly that gap.
    """
    session = _FakeSession(started_at=100.0)
    t_ref = 100.75          # switching the profile + settling took 0.75 s
    offset = t_ref - session.started_at
    assert offset == pytest.approx(0.75)

    events, chord_info, _total = pitched_kit.build_schedule(
        [0, 1], hold=0.5, gap=0.2, chord=True)

    shifted = pitched_kit.shift_events(events, offset)
    for ev, sev in zip(events, shifted, strict=True):
        assert sev["t_press"] == pytest.approx(ev["t_press"] + offset)
        assert sev["t_release"] == pytest.approx(ev["t_release"] + offset)
        assert sev["pad"] == ev["pad"]

    shifted_chord = pitched_kit.shift_chord(chord_info, offset)
    for tp, stp in zip(chord_info["t_press"], shifted_chord["t_press"], strict=True):
        assert stp == pytest.approx(tp + offset)
    assert shifted_chord["t_release"] == pytest.approx(chord_info["t_release"] + offset)
    assert shifted_chord["steady_start"] == pytest.approx(chord_info["steady_start"] + offset)
    assert shifted_chord["steady_end"] == pytest.approx(chord_info["steady_end"] + offset)
    assert shifted_chord["pads"] == chord_info["pads"]


def test_shift_events_is_a_no_op_when_started_at_equals_t_ref() -> None:
    session = _FakeSession(started_at=50.0)
    t_ref = 50.0
    offset = t_ref - session.started_at
    events, _chord, _total = pitched_kit.build_schedule([0], hold=0.3, gap=0.1, chord=False)
    assert pitched_kit.shift_events(events, offset) == events


# --------------------------------------------------------------------------- analyze_pitch kernel


def test_analyze_pitch_recovers_a_single_tone_within_a_cent() -> None:
    from crosspad_hil.analyze import analyze_pitch

    freq = pitched_kit.expected_hz(9)  # 440 Hz exactly
    x = tone(freq, 1.0, amp=0.5)
    r = analyze_pitch(x, SR, 0.1, 0.9, expected_hz=freq)
    assert abs(r["cents_vs"]) < 1.0
    assert r["hz"] == pytest.approx(freq, abs=freq * 2 ** (1 / 1200) - freq)


def test_analyze_pitch_recovers_mixed_pitches_from_a_synthetic_wav(tmp_path: Path) -> None:
    """A mono WAV holding three of the fixture kit's expected pitches at once —
    the same shape a chord slice out of a real capture would take. Each
    frequency is picked out with a narrow `search_hz` band and must come back
    within one cent, matching the tolerance the scenario's own kernel test
    documents (design: pass within +-10 cents; the kernel itself must be far
    tighter than the pass/fail budget it backs).
    """
    from crosspad_hil.analyze import analyze_pitch

    pads = [0, 4, 7]
    freqs = [pitched_kit.expected_hz(p) for p in pads]
    mix = sum(tone(f, 1.5, amp=0.2) for f in freqs)
    wav = write_wav(tmp_path / "chord.wav", mix, sr=SR)

    for f in freqs:
        lo, hi = f * 2 ** (-0.5 / 12), f * 2 ** (0.5 / 12)
        r = analyze_pitch(wav, t0=0.2, t1=1.3, expected_hz=f, search_hz=(lo, hi))
        assert abs(r["cents_vs"]) < 1.0, (f, r)


def test_analyze_pitch_needs_rate_for_an_array() -> None:
    from crosspad_hil.analyze import analyze_pitch

    with pytest.raises(HilError):
        analyze_pitch(np.zeros(4096), None, 0.0, 1.0)


# --------------------------------------------------------------------------- PITCHED: parser


def test_pitched_status_shape_this_scenario_relies_on() -> None:
    """Full grammar coverage lives in test_parsers_cdc.py; this just pins the
    field names `pitch_verdict` and the run actually read (a rename here would
    otherwise pass silently until a board answered it)."""
    p = parse_cdc_reply(GOOD_PITCHED_LINE)
    assert p is not None
    assert {"active", "roots", "steals", "rt_max_us", "load_err"} <= p.keys()


# --------------------------------------------------------------------------- chord scoring floor


def test_score_chord_floor_rejects_a_peak_that_only_clears_the_relative_rule(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A peak within -40 dB of the loudest one still must not count unless it also
    clears the pre-chord noise floor by 20 dB — otherwise a silent pad can be
    "found" via a harmonic of another pad landing in its narrow search band.
    """
    canned = iter([
        {"hz": 1.0, "expected_hz": 1.0, "cents_vs": 0.0, "peak_db": -10.0, "level_dbfs": -10.0},
        {"hz": 1.0, "expected_hz": 1.0, "cents_vs": 0.5, "peak_db": -35.0, "level_dbfs": -55.0},
    ])

    def fake_analyze_pitch(*_args: object, **_kw: object) -> dict:
        return dict(next(canned))

    monkeypatch.setattr(pitched_kit, "analyze_pitch", fake_analyze_pitch)
    scenario = pitched_kit.PitchedKitScenario()
    p = pitched_kit.Params(cents=10.0)
    chord_info = {"pads": [0, 1], "steady_start": 0.0, "steady_end": 0.5}

    result = scenario._score_chord(chord_info, x=None, sr=44100, p=p, noise_floor_dbfs=-50.0)

    peaks = result["peaks"]
    # rel_db for peak 1 is -35 - (-10) = -25, which clears -40 on its own —
    # only the absolute floor (-55 vs -50+20=-30) should fail it.
    assert peaks[0]["pass"] is True
    assert peaks[1]["rel_db"] == pytest.approx(-25.0)
    assert peaks[1]["pass"] is False
    assert result["found"] == 1
    assert result["noise_floor_dbfs"] == pytest.approx(-50.0)


def test_score_chord_skips_the_floor_check_when_no_floor_was_measured(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """`noise_floor_dbfs=None` (the pre-chord window was too short to measure)
    falls back to the old rel_db-only rule rather than failing every peak."""
    canned = iter([
        {"hz": 1.0, "expected_hz": 1.0, "cents_vs": 0.0, "peak_db": -10.0, "level_dbfs": -10.0},
        {"hz": 1.0, "expected_hz": 1.0, "cents_vs": 0.5, "peak_db": -35.0, "level_dbfs": -55.0},
    ])

    def fake_analyze_pitch(*_args: object, **_kw: object) -> dict:
        return dict(next(canned))

    monkeypatch.setattr(pitched_kit, "analyze_pitch", fake_analyze_pitch)
    scenario = pitched_kit.PitchedKitScenario()
    p = pitched_kit.Params(cents=10.0)
    chord_info = {"pads": [0, 1], "steady_start": 0.0, "steady_end": 0.5}

    result = scenario._score_chord(chord_info, x=None, sr=44100, p=p, noise_floor_dbfs=None)

    assert result["found"] == 2
    assert all(r["pass"] for r in result["peaks"])


def test_chord_noise_floor_none_when_the_lead_in_is_too_short() -> None:
    """The floor is the lead-in silence before the first press (the pre-chord
    gap drifts into the chord on a PipeWire take); with the first press 0.1 s
    into the take there is nothing to measure, and the check is skipped."""
    scenario = pitched_kit.PitchedKitScenario()
    scored_events = [{"pad": 0, "t_press": 0.1, "t_release": 0.6}]
    scored_chord = {"pads": [0], "t_press": [1.0], "t_release": 1.5,
                    "steady_start": 1.2, "steady_end": 1.4}

    class _NoLogCtx:
        def log(self, _msg: str) -> None:
            pass

    floor = scenario._chord_noise_floor(scored_events, scored_chord, x=None, sr=44100,
                                        ctx=_NoLogCtx())
    assert floor is None


# --------------------------------------------------------------------------- verdict


def _pad_result(pad: int, cents_vs: float, tol: float = 10.0) -> dict:
    return {"pad": pad, "cents_vs": cents_vs, "pass": abs(cents_vs) <= tol}


GOOD_STATUS = {"active": True, "steals": 0, "rt_max_us": 800, "roots": [57.0, 69.0]}


def test_verdict_passes_a_clean_run() -> None:
    pads = [_pad_result(0, 0.5), _pad_result(1, -2.0)]
    problems = pitched_kit.pitch_verdict(pads, 10.0, GOOD_STATUS, None)
    assert problems == []


def test_verdict_counts_only_this_runs_steals() -> None:
    pads = [_pad_result(0, 0.5)]
    carried = {**GOOD_STATUS, "steals": 12}
    assert pitched_kit.pitch_verdict(pads, 10.0, carried, None, steals_before=12) == []
    problems = pitched_kit.pitch_verdict(pads, 10.0, carried, None, steals_before=10)
    assert any("2 voice steal" in p for p in problems)


def test_verdict_fails_a_pad_outside_tolerance() -> None:
    pads = [_pad_result(0, 0.5), _pad_result(1, 25.0)]
    problems = pitched_kit.pitch_verdict(pads, 10.0, GOOD_STATUS, None)
    assert any("outside 10 cents" in p and "[1]" in p for p in problems)


def test_verdict_fails_on_missing_chord_peak() -> None:
    chord = {"found": 3, "expected": 4}
    problems = pitched_kit.pitch_verdict([], 10.0, GOOD_STATUS, chord)
    assert any("3/4" in p for p in problems)


def test_verdict_fails_on_steals() -> None:
    status = {**GOOD_STATUS, "steals": 2}
    problems = pitched_kit.pitch_verdict([], 10.0, status, None)
    assert any("2 voice steal" in p for p in problems)


def test_verdict_fails_over_rt_budget() -> None:
    status = {**GOOD_STATUS, "rt_max_us": pitched_kit.RT_BUDGET_US}
    problems = pitched_kit.pitch_verdict([], 10.0, status, None)
    assert any("rt_max_us" in p for p in problems)


def test_verdict_fails_when_inactive() -> None:
    status = {**GOOD_STATUS, "active": False}
    problems = pitched_kit.pitch_verdict([], 10.0, status, None)
    assert any("active=0" in p for p in problems)


def test_verdict_names_the_load_err_when_inactive() -> None:
    status = {**GOOD_STATUS, "active": False, "load_err": "zone_0:_no_PSRAM"}
    problems = pitched_kit.pitch_verdict([], 10.0, status, None)
    assert any("active=0 (zone_0:_no_PSRAM)" in p for p in problems)


def test_verdict_fails_on_a_root_off_the_grid() -> None:
    status = {**GOOD_STATUS, "roots": [57.0, 69.2]}
    problems = pitched_kit.pitch_verdict([], 10.0, status, None)
    assert any("roots not within" in p for p in problems)


def test_verdict_fails_when_status_is_missing() -> None:
    problems = pitched_kit.pitch_verdict([], 10.0, None, None)
    assert any("PITCHED_STATUS unavailable" in p for p in problems)


def test_verdict_accumulates_every_problem_at_once() -> None:
    status = {"active": False, "steals": 1, "rt_max_us": 2000, "roots": [57.4]}
    pads = [_pad_result(0, 50.0)]
    chord = {"found": 1, "expected": 2}
    problems = pitched_kit.pitch_verdict(pads, 10.0, status, chord)
    assert len(problems) == 6   # pad, chord, active, steals, rt, roots


# --------------------------------------------------------------------------- the voice limit


def test_a_16_note_chord_needs_only_the_last_10_notes(monkeypatch: pytest.MonkeyPatch) -> None:
    """The engine keeps 10 voices: the first 6 pressed are stolen, not missing."""
    def fake_analyze_pitch(*_a: object, expected_hz: float, **_kw: object) -> dict:
        pad = _pad_from_hz(expected_hz, 60)
        level = -80.0 if pad < 6 else -10.0          # the stolen ones are gone
        return {"hz": expected_hz, "cents_vs": 0.0, "peak_db": level, "level_dbfs": level}

    monkeypatch.setattr(pitched_kit, "analyze_pitch", fake_analyze_pitch)
    chord_info = {"pads": list(range(16)), "steady_start": 0.0, "steady_end": 0.5}
    r = pitched_kit.PitchedKitScenario()._score_chord(
        chord_info, x=None, sr=44100, p=pitched_kit.Params(cents=10.0), voices=10)
    assert (r["found"], r["expected"], r["stolen"]) == (10, 10, [0, 1, 2, 3, 4, 5])
    assert r["stolen_still_sounding"] == []
    status = {**GOOD_STATUS, "steals": 6}
    assert pitched_kit.pitch_verdict([], 10.0, status, r) == []
    assert pitched_kit.pitch_verdict([], 10.0, {**status, "steals": 7}, r) == [
        "7 voice steal(s) during the run, the chord accounts for 6"]


def _chord_take(release_s: float, *, click_at: float | None = None) -> tuple[Any, dict]:
    """A 440 Hz chord stand-in held 0.5..1.5 s, decaying over ``release_s`` after it."""
    t = np.arange(int(SR * 3.0)) / SR
    env = np.clip((t - 0.5) / 0.002, 0.0, 1.0)        # the kit's 2 ms attack
    env = np.where(t > 1.5, np.clip(1.0 - (t - 1.5) / max(release_s, 1e-3), 0.0, 1.0), env)
    x = 0.05 * env * np.sin(2 * np.pi * 440.0 * t)
    if click_at is not None:
        x[int(click_at * SR)] += 0.8
    chord = {"t_press": [0.5], "t_release": 1.5, "steady_start": 0.7, "steady_end": 1.3}
    return x, chord


def test_chord_dynamics_times_the_release_and_finds_a_click() -> None:
    x, chord = _chord_take(0.1)
    d = pitched_kit.chord_dynamics(x, SR, chord, tail_s=1.0)
    assert d["clicks"] == 0 and d["still_sounding"] is False
    assert 50.0 <= d["release_ms"] <= 110.0
    assert pitched_kit.pitch_verdict([], 10.0, GOOD_STATUS,
                                     {"found": 1, "expected": 1, **d}) == []

    x, chord = _chord_take(1.2)
    hung = pitched_kit.chord_dynamics(x, SR, chord, tail_s=1.0)
    assert hung["still_sounding"] is True
    assert pitched_kit.pitch_verdict([], 10.0, GOOD_STATUS, {"found": 1, "expected": 1, **hung}) \
        == ["chord: still sounding at the end of the recording after the release"]

    x, chord = _chord_take(0.6)
    slow = pitched_kit.chord_dynamics(x, SR, chord, tail_s=1.0)
    assert slow["release_ms"] > pitched_kit.RELEASE_MAX_MS and not slow["still_sounding"]

    x, chord = _chord_take(0.1, click_at=1.0)
    assert pitched_kit.chord_dynamics(x, SR, chord, tail_s=1.0)["clicks"] == 1


def test_an_engine_underrun_fails_the_run() -> None:
    assert pitched_kit.pitch_verdict([], 10.0, GOOD_STATUS, None, {"underrun": 1}) == [
        "engine underrun +1"]


# --------------------------------------------------------------------------- IN2 (jack) gain

def test_in2_is_pinned_for_the_run_and_put_back(bench) -> None:
    """The jack comes up at 0 dB since dcfac16; the -50 dBFS sine floor was set
    when the loop ran at the equivalent of +24 dB."""
    rep = bench.run(pads=[0], hold=0.25, gap=0.02, chord=False)
    assert rep.passed, rep.summary
    d = rep.data
    assert (d["ingain_before"]["IN2"], d["ingain_during"]["IN2"], d["ingain_after"]["IN2"]) \
        == (0, 24, 0)
    assert d["ingain_restored"] is True and bench.cdc.ingain["IN2"] == 0
    assert d["min_rms_dbfs"] == -50.0 and bench.analyze.sine_floors == [-50.0]
    # pinned during the CDC setup, before the profile switch
    assert bench.cdc.written.index("INGAIN 1 24") < bench.cdc.written.index("KIT_LIST")


def test_the_sine_floor_follows_the_gain(bench) -> None:
    rep = bench.run(pads=[0], hold=0.25, gap=0.02, chord=False, in2_gain=12)
    assert rep.data["min_rms_dbfs"] == -62.0 and bench.analyze.sine_floors == [-62.0]


def test_in2_is_put_back_when_the_capture_fails(bench) -> None:
    bench.capture_error = HilError(ENV, "card busy")
    rep = bench.run(**FAST_PARAMS)
    assert rep.exit_code == 2
    assert bench.cdc.ingain["IN2"] == 0 and rep.data["ingain_restored"] is True


def test_the_mic_path_and_no_gain_leave_the_jack_alone(bench) -> None:
    bench.run(**FAST_PARAMS, preset="mics")
    bench.run(**FAST_PARAMS, in2_gain=None)
    assert not any(c.startswith("INGAIN") for c in bench.cdc.written)


def test_seconds_the_host_lost_are_not_scored_against_the_pads(bench) -> None:
    def lossy(device: Device, **kw: Any) -> FakeCapture:
        cap = bench.make_capture(device, **kw)
        cap.result_kw = {"health": {"ok": False, "missed_s": 6.3, "max_gap_ms": 40.0,
                                    "overruns": 0, "exclusive": False,
                                    "detail": "captured 3.0 s of 9.3 s"}}
        return cap

    pitched_kit.open_capture = lossy  # type: ignore[assignment]
    try:
        rep = bench.run(**FAST_PARAMS)
    finally:
        pitched_kit.open_capture = bench.make_capture  # type: ignore[assignment]
    assert not rep.passed and rep.exit_code == 2
    assert "capture not measured — host lost 6.3 s" in rep.summary
    assert "through PipeWire" in rep.summary
    assert rep.data["host_lost_s"] == 6.3 and bench.analyze.pitch_calls == []
