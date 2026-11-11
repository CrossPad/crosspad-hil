"""The pitched-instrument engine: load `HIL PITCHED`, play it, measure it.

Companion to `sampler_record` for the second sample engine (design:
docs/superpowers/specs/2026-09-03-pitched-instrument-design.md and plan
2026-09-03-pitched-instrument.md, task T4). Fixture kit:
`tools/make_pitched_kit.py`. It carries 16 pads, `pad_notes[pad] = base_note
+ pad` (60..75 by default), zones at MIDI roots 57/69[/81/93] with alternating
explicit and `"root": "auto"` zones — so one run exercises both the manual
root and the load-time pitch detector.

Three phases, one continuous recording plus a second capture for the chord:

1. **Per-pad.** Press, hold, release, gap — for every selected pad, in order.
   Each pad's steady window is sliced out of the recording and judged two
   ways: :func:`crosspad_hil.analyze.analyze_pitch` (windowed FFT + parabolic
   interpolation) against `440*2**((base_note+pad-69)/12)`, and
   :func:`crosspad_hil.analyze.analyze_sine`'s phase-slip detector on the same
   slice (dropped/repeated samples a level check cannot see).
2. **Chord.** Every selected pad within ~50 ms, held 1 s, released. The
   engine has a voice limit (`PITCHED_STATUS voices=n/<max>`, 10) and steals
   the oldest voice (a 5 ms fade) when a note needs one: of a 16-note chord
   the *last* 10 pressed sound, and those must each show a peak within
   tolerance and within 40 dB of the loudest in the steady middle 0.5 s. The
   stolen ones are reported, not required. From 20 ms into the chord to 300 ms
   after the release there must be no click (a steal is a fade, not a cut),
   the chord must have died away within 300 ms of the release, and the engine
   must not have underrun over the run.
3. **Status.** `PITCHED_STATUS` after the run: `active=1`, no steals beyond
   the chord's own (`len(chord) - voices`),
   `rt_max_us` under one 64-frame block at 44.1 kHz (1450 us), and every root
   (explicit or auto-detected) within 0.05 semitone of a MIDI integer — the
   auto zones prove the detector, not just the explicit ones.

The formula `pad_notes[pad] = base_note + pad` and `hz = 440*2**((note-69)/12)`
is duplicated here rather than imported from `tools/make_pitched_kit.py`:
crosspad-hil does not depend on platform-idf.
"""
from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from crosspad_hil import card_hold, verbs
from crosspad_hil.capture import PRESETS
from crosspad_hil.cdc import CdcLink
from crosspad_hil.devices import UsbMode
from crosspad_hil.errors import (
    BAD_ARGS,
    ENV,
    NO_CDC_IN_AUDIO_MODE,
    NOT_SUPPORTED,
    TIMEOUT,
    HilError,
)
from crosspad_hil.scenarios import _audio, _deps
from crosspad_hil.scenarios.base import Artifact, Context, Progress, Report, register
from crosspad_hil.usbmode import set_mode

SAMPLER_APP = "Sampler"
PAD_COUNT = 16
VELOCITY = 100

# --- timing, seconds --------------------------------------------------------
LEAD_S = 0.5                  # silence before the first hit — the recording's time zero
STREAM_SETTLE_S = 0.4         # let the capture stream settle before the first hit
# A pad slice quieter than this has no sine to check. Set (2026-09-03) when the
# jack loop ran at the equivalent of IN2 +24 dB; it moves with --in2-gain.
SINE_FLOOR_DBFS = -50.0
TAIL_S = 1.0                  # let the last release tail decay into the recording
CHORD_HOLD_S = 1.0            # how long the chord is held once every pad has landed
CHORD_SPREAD_S = 0.05         # every selected pad pressed within this window
CHORD_STEADY_MARGIN_S = 0.2   # trimmed off both ends of the chord hold for the FFT window
ATTACK_MARGIN_S = 0.05        # trimmed off the start of a per-pad window (attack_ms=2 plus slack)
RELEASE_MARGIN_S = 0.05       # trimmed off the end, before the release envelope starts

APP_START_SETTLE_S = 2.0
KIT_LOAD_TIMEOUT_S = 20.0

RT_BUDGET_US = 1450           # one 64-frame block at 44.1 kHz
PITCHED_VOICES = 10           # the engine's voice limit when PITCHED_STATUS does not say
RELEASE_MAX_MS = 300.0        # key-up to silence; the chord used to hang on for >1 s
RELEASE_DROP_DB = 40.0        # "silent" = this far under the chord's steady level...
RELEASE_FLOOR_MARGIN_DB = 6.0  # ...or within this of the lead-in noise floor
CHORD_CLICK_SKIP_S = 0.02     # the chord's own first attack is not a click
ROOT_TOL_SEMITONES = 0.05     # roots (explicit or auto-detected) must land this close to an int
CHORD_FLOOR_MARGIN_DB = 20.0  # a chord peak must clear the pre-chord noise floor by this much

SETUP_CODES: frozenset[str] = frozenset(
    {ENV, NOT_SUPPORTED, TIMEOUT, NO_CDC_IN_AUDIO_MODE}
)


def expected_hz(pad: int, base_note: int = 60) -> float:
    """`pad_notes[pad] = base_note + pad`; `hz = 440 * 2**((note-69)/12)`.

    Duplicated from `tools/make_pitched_kit.py` by design — see module docstring.
    """
    note = base_note + pad
    return 440.0 * (2.0 ** ((note - 69) / 12.0))


@dataclass
class Params:
    kit_name: str = field(default="HIL PITCHED",
                          metadata={"help": "pitched kit to load, by name (case-insensitive)"})
    pads: list[int] | None = field(
        default=None, metadata={"help": "pads to test (comma-separated); default 0-15"})
    hold: float = field(default=0.6, metadata={"help": "seconds a pad is held before release"})
    gap: float = field(default=0.4, metadata={"help": "silence between pads"})
    cents: float = field(default=10.0, metadata={"help": "pitch tolerance, in cents"})
    chord: bool = field(default=True,
                        metadata={"help": "also run the all-selected-pads chord phase"})
    base_note: int = field(
        default=60, metadata={"help": "pad_notes base: pad_notes[pad] = base_note + pad"})
    preset: str = field(
        default="headphone",
        metadata={"help": "capture path: headphone (unity DAC->ADC loop), speaker, mics, line-in"})
    volume: int | None = field(default=None, metadata={"help": "capture codec volume 0..100"})
    exclusive_card: bool = field(
        default=True,
        metadata={"help": "pause the audio-nodes service for the run (bench holder only), so "
                          "the capture opens the card directly instead of through PipeWire"})
    in2_gain: int | None = field(
        default=_audio.IN2_DEFAULT_DB,
        metadata={"help": "line-in (IN2, codec 1) preamp gain in dB for the jack presets; "
                          "pinned over CDC for the run and put back after (the board keeps "
                          "it in NVS); the -50 dBFS sine floor follows it. Omit to leave it"})
    free_loop: bool = field(
        default=True,
        metadata={"help": "on an input pair that hears Line 3/4 (which switches the jack "
                          "loop's driver off), select a pair without it for the run and put "
                          "the old one back (the board keeps it in NVS); for the LINE1 presets"})
    # No per-scenario --logdir: `crosspad-hil run pitched_kit --logdir DIR` (the CLI's
    # own flag, added after every scenario's params) already picks the workdir this
    # writes the capture WAV and per-pad slices into.


# --------------------------------------------------------------------------- seams
# Same shape as sampler_record's: one place a test can replace the audio/analysis
# machinery without a sound card, and one place that turns "the optional extra
# isn't installed" into an ENV (exit 2) rather than a plain FAIL.


def _as_env(exc: HilError) -> HilError:
    if exc.code != NOT_SUPPORTED:
        return exc
    return HilError(ENV, exc.message, hint=exc.hint, **exc.details)


def open_capture(device: Any, **kw: Any) -> Any:
    try:
        return _deps.module("capture").CaptureSession(device, **kw)
    except HilError as e:
        raise _as_env(e) from None


def open_pad_transport(device: Any, kind: str) -> Any:
    return _deps.module("stim").open_transport(device, kind)


def analyze_pitch(*args: Any, **kw: Any) -> dict:
    try:
        return dict(_deps.module("analyze").analyze_pitch(*args, **kw))
    except HilError as e:
        raise _as_env(e) from None


def analyze_sine(*args: Any, **kw: Any) -> dict:
    try:
        return _deps.module("analyze").analyze_sine(*args, **kw).to_dict()
    except HilError as e:
        raise _as_env(e) from None


def read_wav_ch(*args: Any, **kw: Any) -> Any:
    try:
        return _deps.module("analyze").read_wav_ch(*args, **kw)
    except HilError as e:
        raise _as_env(e) from None


class ForgivingTransport:
    """Counts a refused pad write instead of raising it — see sampler_record's twin.

    A dropped SysEx byte mid-run is worth losing one hit over, not the whole
    recording and the USB-profile restore that has to run in `finally`.
    """

    def __init__(self, inner: Any) -> None:
        self._inner = inner
        self.name = getattr(inner, "name", "?")
        self.failures = 0

    def press(self, pad: int, vel: int) -> None:
        try:
            self._inner.press(pad, vel)
        except Exception:  # noqa: BLE001 - a lost message is counted, not raised
            self.failures += 1

    def release(self, pad: int) -> None:
        try:
            self._inner.release(pad)
        except Exception:  # noqa: BLE001
            self.failures += 1

    def close(self) -> None:
        self._inner.close()


# --------------------------------------------------------------------------- schedule


def build_schedule(pads: list[int], hold: float, gap: float, *, chord: bool,
                   lead_s: float = LEAD_S, chord_hold_s: float = CHORD_HOLD_S,
                   chord_spread_s: float = CHORD_SPREAD_S,
                   chord_steady_margin_s: float = CHORD_STEADY_MARGIN_S,
                   tail_s: float = TAIL_S) -> tuple[list[dict], dict | None, float]:
    """(per-pad events, chord event or None, total seconds) — pure, so it is unit-testable.

    A per-pad event is `{pad, t_press, t_release}`. The chord event is
    `{pads, t_press: [...], t_release, steady_start, steady_end}` — one press
    time per pad, spread over `chord_spread_s`, and the FFT window trimmed by
    `chord_steady_margin_s` off both ends of the hold.
    """
    events: list[dict[str, Any]] = []
    t = lead_s
    for pad in pads:
        events.append({"pad": pad, "t_press": round(t, 4), "t_release": round(t + hold, 4)})
        t += hold + gap
    chord_info: dict[str, Any] | None = None
    if chord and pads:
        n = len(pads)
        step = chord_spread_s / (n - 1) if n > 1 else 0.0
        presses = [round(t + i * step, 4) for i in range(n)]
        t_release = round(t + chord_spread_s + chord_hold_s, 4)
        chord_info = {
            "pads": list(pads), "t_press": presses, "t_release": t_release,
            "steady_start": round(t + chord_spread_s + chord_steady_margin_s, 4),
            "steady_end": round(t_release - chord_steady_margin_s, 4),
        }
        t = t_release
    total = round(t + tail_s, 4)
    return events, chord_info, total


def shift_events(events: list[dict[str, Any]], offset: float) -> list[dict[str, Any]]:
    """Re-base schedule-relative `t_press`/`t_release` onto the WAV's own t=0.

    The schedule is built against `t_ref = time.monotonic()` taken once the
    capture stream is already running, but `analyze_pitch` slices the WAV
    relative to `session.started_at` — set earlier, when the stream first
    reported active. `offset = t_ref - session.started_at` is that gap; every
    window handed to the analyzer has to carry it, pad and chord alike, or
    every score is read off a slice that starts too early by exactly that much.
    """
    return [{**ev, "t_press": ev["t_press"] + offset, "t_release": ev["t_release"] + offset}
            for ev in events]


def shift_chord(chord_info: dict[str, Any], offset: float) -> dict[str, Any]:
    """`shift_events`'s twin for the one chord event (list of presses + one release)."""
    return {
        **chord_info,
        "t_press": [t + offset for t in chord_info["t_press"]],
        "t_release": chord_info["t_release"] + offset,
        "steady_start": chord_info["steady_start"] + offset,
        "steady_end": chord_info["steady_end"] + offset,
    }


def pitch_window(ev: dict[str, Any], *, attack_margin_s: float = ATTACK_MARGIN_S,
                 release_margin_s: float = RELEASE_MARGIN_S) -> tuple[float, float]:
    """The steady window of one per-pad event, trimmed for the attack and release."""
    t0 = ev["t_press"] + attack_margin_s
    t1 = ev["t_release"] - release_margin_s
    if t1 <= t0:
        return ev["t_press"], ev["t_release"]
    return t0, t1


def _write_pcm_wav(path: Path, seg: Any, sr: int) -> None:
    """16-bit mono PCM WAV from a float [-1, 1] sequence — analyze_sine needs a path."""
    import wave

    try:
        import numpy as np

        pcm = (np.clip(np.asarray(seg, dtype=np.float64), -1.0, 0.999969) * 32768.0).astype("<i2")
        data = pcm.tobytes()
    except ImportError:
        import struct

        data = b"".join(struct.pack("<h", int(max(-1.0, min(1.0, float(v))) * 32767)) for v in seg)
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sr)
        w.writeframes(data)


def chord_dynamics(x: Any, sr: int, chord: dict[str, Any], *, tail_s: float,
                   noise_floor_dbfs: float | None = None) -> dict[str, Any]:
    """Clicks from just after the first press to RELEASE_MAX_MS after the release
    (the steals and the release itself), and how long after the release the chord
    died away.

    ``release_ms`` is the end of the last 10 ms frame still louder than the
    chord's steady level less RELEASE_DROP_DB (or the noise floor plus a
    margin, whichever is higher); ``still_sounding`` when that frame is the
    last one recorded. ``None`` when the take has no steady level or no tail.
    """
    an = _deps.module("analyze")
    np = an._np()
    x = np.asarray(x, dtype=np.float64)
    out: dict[str, Any] = {"clicks": 0, "click_times_s": [], "release_ms": None,
                           "still_sounding": False}
    # From 20 ms into the chord: the first attack out of silence reads as a
    # spike against a silent local level; the steals (the 11th press on, ~33 ms
    # into a 16-note chord) land on notes already sounding.
    a = max(0, int((chord["t_press"][0] + CHORD_CLICK_SKIP_S) * sr))
    b = min(len(x), int((chord["t_release"] + RELEASE_MAX_MS / 1000.0) * sr))
    if b > a:
        n, times = an.click_metric(x[a:b], sr)
        out["clicks"] = int(n)
        out["click_times_s"] = [round(a / sr + t, 3) for t in times]
    steady = x[int(chord["steady_start"] * sr):int(chord["steady_end"] * sr)]
    level = float(np.sqrt(np.mean(steady ** 2))) if len(steady) else 0.0
    win = max(1, sr // 100)
    r0 = int(chord["t_release"] * sr)
    r1 = min(len(x), int((chord["t_release"] + tail_s) * sr))
    if level <= 0.0 or r1 - r0 < win:
        return out
    thr = 20 * math.log10(level) - RELEASE_DROP_DB
    if noise_floor_dbfs is not None:
        thr = max(thr, noise_floor_dbfs + RELEASE_FLOOR_MARGIN_DB)
    frames = x[r0:r0 + ((r1 - r0) // win) * win].reshape(-1, win)
    dbs = 20 * np.log10(np.sqrt(np.mean(frames ** 2, axis=1)) + 1e-12)
    above = np.flatnonzero(dbs > thr)
    if len(above):
        last = int(above[-1])
        out["release_ms"] = round((last + 1) * win * 1000.0 / sr, 1)
        out["still_sounding"] = last == len(dbs) - 1
    else:
        out["release_ms"] = 0.0
    return out


# --------------------------------------------------------------------------- verdict


def pitch_verdict(pad_results: list[dict], cents_tol: float,
                  status: dict | None, chord_result: dict | None,
                  engine: dict | None = None, steals_before: int = 0) -> list[str]:
    """The FAIL reasons for one run — a pure function so the verdict logic is testable
    with a fabricated result set, no device or capture involved."""
    problems: list[str] = []
    bad = [r["pad"] for r in pad_results if not r.get("pass", False)]
    if bad:
        problems.append(f"pads outside {cents_tol:.0f} cents: {bad}")
    if chord_result is not None and chord_result["found"] < chord_result["expected"]:
        problems.append(f"chord: only {chord_result['found']}/{chord_result['expected']} "
                        "of the notes that keep a voice found within tolerance and -40 dB")
    if chord_result is not None:
        if chord_result.get("clicks"):
            problems.append(f"chord: {chord_result['clicks']} click(s) between the first "
                            f"press and {RELEASE_MAX_MS:.0f} ms after the release "
                            f"(at {chord_result.get('click_times_s')} s)")
        if chord_result.get("still_sounding"):
            problems.append("chord: still sounding at the end of the recording after the "
                            "release")
        elif (chord_result.get("release_ms") or 0.0) > RELEASE_MAX_MS:
            problems.append(f"chord: died away {chord_result['release_ms']:.0f} ms after the "
                            f"release (> {RELEASE_MAX_MS:.0f} ms)")
    for key in ("underrun", "noblock"):
        if ((engine or {}).get(key) or 0) > 0:
            problems.append(f"engine {key} +{engine[key]}")  # type: ignore[index]
    if status is None:
        problems.append("PITCHED_STATUS unavailable")
        return problems
    if not status.get("active", False):
        load_err = status.get("load_err")
        problems.append("engine reports active=0" + (f" ({load_err})" if load_err else ""))
    steals = max(0, int(status.get("steals", 0) or 0) - steals_before)
    allowed = len((chord_result or {}).get("stolen") or [])
    if steals > allowed:
        problems.append(f"{steals} voice steal(s) during the run"
                        + (f", the chord accounts for {allowed}" if allowed else ""))
    rt = int(status.get("rt_max_us", 0) or 0)
    if rt >= RT_BUDGET_US:
        problems.append(f"rt_max_us={rt} >= budget {RT_BUDGET_US}")
    off_roots = [r for r in (status.get("roots") or [])
                if abs(r - round(r)) > ROOT_TOL_SEMITONES]
    if off_roots:
        problems.append(f"roots not within {ROOT_TOL_SEMITONES:g} semitone of a MIDI integer: "
                        f"{off_roots}")
    return problems


def _env(summary: str, data: dict[str, Any], artifacts: list[Artifact]) -> Report:
    return Report(passed=False, summary=summary, data=data, artifacts=artifacts, exit_code=2)


def _wait(ctx: Context, seconds: float) -> None:
    if seconds > 0:
        ctx.cancelled.wait(seconds)
    ctx.check_cancelled()


class PitchedKitScenario:
    name = "pitched_kit"
    Params = Params
    description = "load the pitched-instrument fixture kit, play it and measure the pitches"

    def run(self, ctx: Context, params: Params, progress: Progress) -> Report:
        try:
            p = params
            pads = list(p.pads) if p.pads else list(range(PAD_COUNT))
            if not pads or any(x < 0 or x > 15 for x in pads):
                raise HilError(BAD_ARGS, f"pads must be 0..15, got {p.pads}", pads=p.pads)
            data: dict[str, Any] = {
                "kit_name": p.kit_name, "kit_id": None, "pads": pads,
                "wav": None, "capture": None, "capture_health": None,
                "pitch": [], "chord": None, "status": None, "failures": [],
                "ingain_before": None, "ingain_during": None, "ingain_after": None,
                "ingain_restored": None, "min_rms_dbfs": SINE_FLOOR_DBFS,
                "in_pair_before": None, "in_pair_during": None, "in_pair_after": None,
                "in_pair_restored": None,
            }
            artifacts: list[Artifact] = []
            try:
                try:
                    with card_hold.for_run(ctx, data, enabled=p.exclusive_card,
                                           expected_s=len(pads) * (p.hold + p.gap) + 30):
                        report = self._run(ctx, p, progress, pads, data, artifacts)
                except HilError as e:
                    if e.code not in SETUP_CODES:
                        raise
                    data["failures"].append(f"{e.code}: {e.message}")
                    hint = f" ({e.hint})" if e.hint else ""
                    report = _env(f"pitched_kit: {e.message}{hint}", data, artifacts)
            finally:
                # Pinned before the capture; put back whatever happened since.
                previous = (data["ingain_before"] or {}).get("IN2") \
                    if data["ingain_during"] is not None else None
                # The pair too, unless the refused switch already put it back.
                previous_pair = data["in_pair_before"] \
                    if data["in_pair_during"] is not None \
                    and data["in_pair_restored"] is None else None
                _audio.restore_in2(ctx, previous, data, previous_pair=previous_pair,
                                   set_mode_fn=set_mode)
            for note in (_audio.in2_not_restored(data, previous),
                         _audio.in_pair_not_restored(data, previous_pair)):
                if note is not None:
                    report = Report(passed=False, summary=f"{report.summary}; {note}",
                                    data=data, artifacts=artifacts,
                                    exit_code=report.exit_code or 2)
            return report
        finally:
            ctx.close_all()

    # -- the run --------------------------------------------------------
    def _run(self, ctx: Context, p: Params, progress: Progress, pads: list[int],
             data: dict[str, Any], artifacts: list[Artifact]) -> Report:
        ctx.check_cancelled()
        progress(0, 5, "setting the device up over CDC")

        if ctx.device.usb_mode is not UsbMode.DEFAULT or ctx.device.ports.cdc is None:
            ctx.log("no CDC port — assuming the USB audio profile, switching back to default")
            ctx.device = set_mode(ctx.device, UsbMode.DEFAULT)

        link = ctx.open_cdc()
        try:
            if p.free_loop and PRESETS.get(p.preset, (None, None))[1] == _audio.LOOP_ADC_INPUT:
                # A pair that hears Line 3/4 leaves the loop undriven; the pair
                # first, so its switch cannot undo the IN2 pin below.
                _audio.free_jack_loop(ctx, link, data)
            if p.in2_gain is not None and PRESETS.get(p.preset, (None,))[0] == _audio.JACK_CODEC:
                if _audio.set_in2(ctx, link, p.in2_gain, data) is not None:
                    data["min_rms_dbfs"] = _audio.level_floor(SINE_FLOOR_DBFS, p.in2_gain)
            kit_id, kit_name = self._prepare(ctx, p, link)
            data["kit_id"], data["kit_name"] = kit_id, kit_name
            try:
                status0 = verbs.pitched_status(link)
                data["voices_max"] = status0.get("voices_max") or None
                data["steals0"] = int(status0.get("steals", 0) or 0)
                data["engine0"] = verbs.smpl_counters(link)
            except HilError as e:
                ctx.log(f"before the run: {e.code}: {e.message}")
        finally:
            link.close()   # the CDC endpoint goes away with the profile switch below

        events, chord_info, total_s = build_schedule(
            pads, p.hold, p.gap, chord=p.chord, lead_s=LEAD_S, chord_hold_s=CHORD_HOLD_S,
            chord_spread_s=CHORD_SPREAD_S, chord_steady_margin_s=CHORD_STEADY_MARGIN_S,
            tail_s=TAIL_S)
        ctx.log(f"pitched_kit: {len(pads)} pad(s)" + (" + chord" if chord_info else "")
                + f", ~{total_s:.1f} s")

        logdir = ctx.workdir
        wav = logdir / "pitched_kit.wav"
        session, info, artifact = _audio.start_capture(
            ctx, open_capture, wav, preset=p.preset, volume=p.volume,
            on_open=lambda: progress(1, 5, "switching to the USB audio profile and "
                                            "opening the capture"))
        data["capture_start"] = info
        artifacts.append(artifact)

        transport = None
        try:
            transport = ForgivingTransport(open_pad_transport(ctx.device, "sysex_esp"))
            _wait(ctx, STREAM_SETTLE_S)
            progress(2, 5, f"playing {len(pads)} pad(s)"
                          + (" + chord" if chord_info else ""))
            # Window times passed to analyze_pitch are relative to the WAV's t=0,
            # which is session.started_at (set once the capture stream is actually
            # running), not "now" — t_ref must be re-based onto that, or every
            # window is off by the time the mode switch + route prologue took.
            t_ref = time.monotonic()
            offset = t_ref - session.started_at
            self._play_schedule(ctx, transport, events, chord_info, t_ref)
            _wait(ctx, TAIL_S)
        finally:
            try:
                if transport is not None:
                    transport.close()
            finally:
                _audio.stop_capture(ctx, session, data)
                data["send_failures"] = transport.failures if transport is not None else 0

        why = _audio.host_loss(data, data.get("capture_health"))
        if why:
            return _audio.not_measured("pitched_kit", why, data, artifacts)
        # A silent or unhealthy capture is a bench problem, not 16 pads failing
        # pitch — score nothing against it, report ENV instead (sampler_record's
        # same check, mirrored here before the scoring loop rather than after it).
        reasons = _audio.capture_problems(data.get("capture") or {},
                                          data.get("capture_health") or {})
        if reasons:
            reason = "; ".join(reasons)
            data["failures"].append(f"capture: {reason}")
            return _env(f"pitched_kit: {reason}", data, artifacts)

        progress(3, 5, "scoring the pitches")
        data["t_offset"] = round(offset, 4)
        x, sr, _used_ch = read_wav_ch(wav, "auto")
        pad_expected = {ev["pad"]: expected_hz(ev["pad"], p.base_note) for ev in events}
        scored_events = shift_events(events, offset)
        data["pitch"] = [self._score_pad(ev, x, sr, pad_expected[ev["pad"]], p, logdir,
                                         data["min_rms_dbfs"])
                         for ev in scored_events]

        if chord_info is not None:
            progress(4, 5, "scoring the chord")
            scored_chord = shift_chord(chord_info, offset)
            noise_floor_dbfs = self._chord_noise_floor(scored_events, scored_chord, x, sr, ctx)
            voices = int(data.get("voices_max") or PITCHED_VOICES)
            data["chord"] = self._score_chord(scored_chord, x, sr, p, noise_floor_dbfs,
                                              voices=voices)
            data["chord"].update(chord_dynamics(x, sr, scored_chord, tail_s=TAIL_S,
                                                noise_floor_dbfs=noise_floor_dbfs))

        # -- status: profile is back, CDC is back -------------------------
        link = ctx.open_cdc()
        try:
            data["status"] = verbs.pitched_status(link)
        except HilError as e:
            ctx.log(f"PITCHED_STATUS: {e.code}: {e.message}")
            data["failures"].append(f"PITCHED_STATUS: {e.code}: {e.message}")
        try:
            after = verbs.smpl_counters(link)
            before = data.get("engine0") or {}
            data["engine"] = {k: after[k] - before[k] for k in ("underrun", "noblock")
                              if isinstance(after.get(k), int) and isinstance(before.get(k), int)}
        except HilError as e:
            ctx.log(f"SMPL_PEAK: {e.code}: {e.message}")
        finally:
            link.close()

        progress(5, 5, "done")
        return self._verdict(p, data, artifacts)

    def _prepare(self, ctx: Context, p: Params, link: CdcLink) -> tuple[int, str]:
        kit_id, name, current = _audio.resolve_kit(
            link, kit_name=p.kit_name,
            missing_hint_suffix=" — push it with tools/make_pitched_kit.py")
        _audio.load_kit(ctx, link, kit_id, name, current, wait_s=KIT_LOAD_TIMEOUT_S)
        _audio.ensure_app_running(
            ctx, link, SAMPLER_APP,
            not_installed_reason=f"{SAMPLER_APP} is not in the launcher — nothing routes "
                                 f"NoteOn to the pitched engine",
            settle_s=APP_START_SETTLE_S)
        return int(kit_id), name

    def _play_schedule(self, ctx: Context, transport: ForgivingTransport,
                       events: list[dict], chord_info: dict | None, t_ref: float) -> None:
        def wait_until(t: float) -> None:
            delta = (t_ref + t) - time.monotonic()
            if delta > 0:
                _wait(ctx, delta)
            else:
                ctx.check_cancelled()

        for ev in events:
            wait_until(ev["t_press"])
            transport.press(ev["pad"], VELOCITY)
            wait_until(ev["t_release"])
            transport.release(ev["pad"])
        if chord_info is not None:
            for pad, tp in zip(chord_info["pads"], chord_info["t_press"], strict=True):
                wait_until(tp)
                transport.press(pad, VELOCITY)
            wait_until(chord_info["t_release"])
            for pad in chord_info["pads"]:
                transport.release(pad)

    def _score_pad(self, ev: dict, x: Any, sr: int, expected: float, p: Params,
                   logdir: Path, min_rms_dbfs: float = -50.0) -> dict[str, Any]:
        t0, t1 = pitch_window(ev)
        r = analyze_pitch(x, sr, t0, t1, expected_hz=expected)
        r["pad"] = ev["pad"]
        r["pass"] = r["cents_vs"] is not None and abs(r["cents_vs"]) <= p.cents
        n0, n1 = int(t0 * sr), int(t1 * sr)
        seg = x[n0:n1]
        if len(seg) >= sr * 0.1:
            slice_wav = logdir / f"pad{ev['pad']:02d}_slice.wav"
            try:
                _write_pcm_wav(slice_wav, seg, sr)
                sine = analyze_sine(slice_wav, freq=expected, edge_s=0.02,
                                    min_rms_dbfs=min_rms_dbfs)
                r["sine_glitches"] = sine["glitches"]
                r["sine_timing_slips"] = sine["timing_slips"]
            except HilError as e:
                r["sine_glitches"] = None
                r["sine_timing_slips"] = None
                r["sine_error"] = f"{e.code}: {e.message}"
        else:
            r["sine_glitches"] = None
            r["sine_timing_slips"] = None
        return r

    def _chord_noise_floor(self, scored_events: list[dict], scored_chord: dict,
                          x: Any, sr: int, ctx: Context) -> float | None:
        """RMS level of the lead-in silence before the first pad press, from the
        same WAV — the noise floor a chord peak must clear (see `_score_chord`).

        It used to be the gap between the last release and the chord's first
        press. A take recorded through PipeWire runs on that graph's clock, not
        the card's, and drifts ~0.1 s over a 16-pad run: the 0.4 s gap window
        then ends inside the chord's onset and the "floor" reads -17 dBFS,
        failing all sixteen peaks that were within a cent. The lead-in sits at
        the recording's start, where the drift is nil, and is silence by
        construction (LEAD_S). `None` when the window is too short to measure
        or the analyzer chokes on it; the floor check is then skipped rather
        than the run failing on it.
        """
        if not scored_events:
            return None
        first = scored_events[0]["t_press"]
        t0 = max(0.0, first - LEAD_S + ATTACK_MARGIN_S)
        t1 = first - ATTACK_MARGIN_S
        if t1 - t0 < 0.2:
            return None
        try:
            return analyze_pitch(x, sr, t0, t1)["level_dbfs"]
        except HilError as e:
            ctx.log(f"chord noise-floor measurement skipped: {e.code}: {e.message}")
            return None

    def _score_chord(self, chord_info: dict, x: Any, sr: int, p: Params,
                     noise_floor_dbfs: float | None = None,
                     voices: int = PAD_COUNT) -> dict[str, Any]:
        # Pressed in list order: the engine keeps the last `voices` of them.
        pads = list(chord_info["pads"])
        stolen = pads[:-voices] if len(pads) > voices > 0 else []
        peaks: list[dict[str, Any]] = []
        for pad in pads:
            f = expected_hz(pad, p.base_note)
            lo, hi = f * (2 ** (-0.5 / 12)), f * (2 ** (0.5 / 12))
            r = analyze_pitch(x, sr, chord_info["steady_start"], chord_info["steady_end"],
                              expected_hz=f, search_hz=(lo, hi))
            r["pad"] = pad
            peaks.append(r)
        strongest = max((r["peak_db"] for r in peaks), default=-999.0)
        found = 0
        for r in peaks:
            r["rel_db"] = round(r["peak_db"] - strongest, 2)
            # A peak found only because the whole chord recording is at the
            # noise floor would still pass on the relative -40 dB rule (every
            # "peak" is close to every other one) — the absolute floor catches
            # that: real signal must clear ambient noise by a healthy margin.
            above_floor = (noise_floor_dbfs is None
                          or r["level_dbfs"] - noise_floor_dbfs >= CHORD_FLOOR_MARGIN_DB)
            r["pass"] = bool(r["cents_vs"] is not None and abs(r["cents_vs"]) <= p.cents
                            and r["rel_db"] >= -40.0 and above_floor)
            r["stolen"] = r["pad"] in stolen
            found += int(r["pass"] and not r["stolen"])
        return {"peaks": peaks, "strongest_db": round(strongest, 2),
                "found": found, "expected": len(peaks) - len(stolen),
                "voices": voices, "stolen": stolen,
                "stolen_still_sounding": [r["pad"] for r in peaks if r["stolen"] and r["pass"]],
                "noise_floor_dbfs": noise_floor_dbfs}

    def _verdict(self, p: Params, data: dict[str, Any], artifacts: list[Artifact]) -> Report:
        problems = list(data["failures"])
        problems += pitch_verdict(data["pitch"], p.cents, data.get("status"), data.get("chord"),
                                  data.get("engine"), steals_before=data.get("steals0") or 0)
        if problems:
            return Report(passed=False, summary="pitched_kit: FAIL — " + "; ".join(problems),
                          data=data, artifacts=artifacts, exit_code=1)
        n_ok = sum(1 for r in data["pitch"] if r["pass"])
        chord_note = ""
        if data.get("chord"):
            chord_note = f", chord {data['chord']['found']}/{data['chord']['expected']}"
        status = data.get("status") or {}
        summary = (f"pitched_kit: PASS — {n_ok}/{len(data['pitch'])} pads within "
                  f"{p.cents:.0f} cents{chord_note}, rt_max_us={status.get('rt_max_us')}, "
                  f"steals={status.get('steals')}")
        return Report(passed=True, summary=summary, data=data, artifacts=artifacts, exit_code=0)


SCENARIO = register(PitchedKitScenario())


def main(argv: list[str] | None = None) -> int:
    """Entry point for a platform-idf shim, matching the other scenarios' `main()`."""
    import sys

    from crosspad_hil.cli import main as cli_main

    args = list(sys.argv[1:] if argv is None else argv)
    return cli_main(["run", "pitched_kit", *args])
