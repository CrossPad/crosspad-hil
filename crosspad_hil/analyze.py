"""Offline analysis of a capture (spec §3.2 `crosspad_analyze`): six kinds, one verdict each.

Every measurement here is ported from a tool that has already been believed on real
hardware, and the awkward parts of each are the parts worth keeping:

* **multitone** — ``hil_audio_loopback.analyze``/``judge``: locate the loud region with
  100 ms RMS blocks (a capture always has silence at both ends), then per-tone band
  power, HF junk ratio, glitch bursts and >1 ms dropouts.
* **click**/**silence**/**psd** — ``hil_rt_glitch.click_metric`` and
  ``glitch_analyze``: second-difference outliers against a *local* level, not a global
  threshold. The floor matters: on the parasitic loopback path the return is AM-
  modulated by digital activity, which drags the d2 median down until ordinary tone
  curvature looks like a click. Both sources floor the limit at a fraction of the
  waveform peak, so only real discontinuities survive.
* **sine** — ``glitch_analyze.analyze_sine``: a level-and-spectrum check cannot see a
  stream losing samples, so this one reads the *phase* of a steady tone. Steps in the
  instantaneous phase are dropped or repeated samples; a median filter over the
  residual keeps those and erases the transients a pop makes, which is what separates
  a slipping stream from a noisy one. Repeated takes are compared against each other
  (``glitch_determinism``) because a defect at the same sample offset every time is a
  different bug from one that wanders.
* **onset**/**velocity** — ``hil_velocity``: onsets are anchored to the *noise floor*
  plus a rise, never to the signal peak, or a loud hit raises the bar above every
  quiet one — which is the whole thing being measured. Detected onsets are then
  matched to the known send grid rather than counted, because a drum sample's body
  transient re-crosses the threshold and would otherwise read as an extra hit.

Latency is reported relative to a constant pipeline offset estimated from the first
hit. The absolute number contains the host's MIDI/USB stack and the capture pipeline;
it is comparable across runs on one host, and meaningless as a device latency.

numpy is an optional extra: everything below imports it lazily and raises
``HilError(NOT_SUPPORTED)`` with the install hint when it is missing. ``wav_stats`` is
the exception — the capture path calls it to fill in peak/RMS, so it works without
numpy too (slower, but a capture must never fail for want of an analysis dependency).
"""
from __future__ import annotations

import array
import math
import statistics
import wave
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from .errors import BAD_ARGS, NOT_SUPPORTED, HilError

KINDS = ("multitone", "click", "onset", "velocity", "psd", "silence", "sine")
AUDIO_EXTRA_HINT = "pip install 'crosspad-hil[audio]'"
FULL_SCALE = 32768.0

# Defaults carried over from the tools these are ported from.
MULTITONE_FREQS = (440.0, 997.0, 3181.0)
MULTITONE_MIN_RMS_DBFS = -45.0          # hil_audio_loopback.judge
MULTITONE_MIN_BAND_POWER = 0.02
MULTITONE_MAX_OFFSET_PCT = 2.0
WARN_GLITCH_PER_S = 0.5
CLICK_THRESH = 12.0                     # hil_rt_glitch.click_metric
ONSET_RISE_DB = 12.0                    # hil_velocity.detect_onsets
ONSET_REFRACTORY_S = 0.15
ONSET_RISE_WINDOW_S = 0.012      # how long a hit has to climb rise_db to count as one
ONSET_TOL_S = 0.15
VELOCITY_MIN_RANGE_DB = 12.0            # hil_velocity --min-range-db
VELOCITY_MAX_JITTER_MS = 40.0           # hil_velocity --max-jitter-ms
SILENCE_MAX_RMS_DBFS = -60.0
SILENCE_MAX_GLITCH_PER_S = 0.5
SINE_MIN_RMS_DBFS = -45.0               # glitch_analyze: below this nothing came back
SINE_MIN_BAND_RATIO = 0.05              # share of the energy that must sit on the tone
SINE_MAX_SLIPS = 0                      # a stream either keeps every sample or it does not
SINE_MAX_DRIFT_SAMPLES = 1.0
PHASE_BAND_HZ = 1000.0                  # tone-tracking band; see `phase_residual`
PHASE_PRESENCE_BW_HZ = 200.0            # "is the tone there at all" is a narrower question
GLITCH_TOL_SAMPLES = 96                 # 2 ms at 48 kHz: "the same position" across takes


# --------------------------------------------------------------------------- lazy deps


def _np() -> Any:
    try:
        import numpy  # type: ignore[import-not-found]
    except ImportError as exc:
        raise HilError(NOT_SUPPORTED, "audio analysis needs numpy", hint=AUDIO_EXTRA_HINT) from exc
    return numpy


def _sp_signal() -> Any | None:
    """scipy is preferred for Welch/filters but never required — see `_welch`."""
    try:
        from scipy import signal  # type: ignore[import-not-found]
    except ImportError:
        return None
    return signal


def db(v: float) -> float:
    """dBFS with a -120 floor: digital silence is reported as -120, not as -240."""
    return max(20.0 * math.log10(max(float(v), 1e-12)), -120.0)


# --------------------------------------------------------------------------- wav io


def _open_wav(path: Path) -> tuple[bytes, int, int, int]:
    try:
        with wave.open(str(path), "rb") as w:
            if w.getsampwidth() != 2:
                raise HilError(BAD_ARGS, f"{path}: only 16-bit WAVs are supported, "
                                         f"got {w.getsampwidth() * 8}-bit")
            return w.readframes(w.getnframes()), w.getframerate(), w.getnchannels(), w.getnframes()
    except HilError:
        raise
    except (OSError, wave.Error) as exc:
        raise HilError(BAD_ARGS, f"cannot read {path}: {exc}",
                       hint="capture writes 16-bit PCM WAVs under hil_logs/") from exc


AUTO_CHANNEL = "auto"


def read_wav_ch(path: str | Path,
                channel: int | str | None = AUTO_CHANNEL) -> tuple[Any, int, int | None]:
    """``(samples, sample_rate, channel_used)``; `channel=None` keeps every channel.

    The default is `"auto"`: the channel with the most energy. The capture routes
    this package records are not reliably stereo — on the DAC->ADC loop the whole
    signal lands on one channel and the other sits at the noise floor. Reading
    channel 0 regardless is how a recording containing every hit was once
    reported as "nothing audible at all", which is the worst answer an analyser
    can give: it accuses the firmware of a fault the recording disproves.
    """
    np = _np()
    raw, sr, ch, _frames = _open_wav(Path(path))
    data = np.frombuffer(raw, dtype="<i2").astype(np.float64) / FULL_SCALE
    data = data[: (len(data) // ch) * ch].reshape(-1, ch)
    if channel is None:
        return data, sr, None
    if channel == AUTO_CHANNEL:
        if ch == 1 or len(data) == 0:
            channel = 0
        else:
            channel = int(np.argmax((data ** 2).mean(axis=0)))
    if not isinstance(channel, int) or channel < 0:
        raise HilError(BAD_ARGS, f"channel must be an index or {AUTO_CHANNEL!r}, got {channel!r}")
    if channel >= ch:
        raise HilError(BAD_ARGS, f"{path} has {ch} channel(s); asked for channel {channel}")
    return data[:, channel], sr, channel


def read_wav(path: str | Path, *,
             channel: int | str | None = AUTO_CHANNEL) -> tuple[Any, int]:
    """``(samples, sample_rate)`` as float in [-1, 1); see `read_wav_ch`."""
    x, sr, _used = read_wav_ch(path, channel)
    return x, sr


def wav_stats(path: str | Path) -> dict:
    """Level and duration of a WAV, with a pure-stdlib path so capture never needs numpy."""
    p = Path(path)
    raw, sr, ch, frames = _open_wav(p)
    seconds = frames / float(sr or 1)
    base = {"wav": str(p), "sample_rate": sr, "channels": ch, "frames": frames,
            "seconds": round(seconds, 3)}
    if not raw:
        return {**base, "peak_dbfs": -120.0, "rms_dbfs": -120.0}
    try:
        np = _np()
    except HilError:
        pcm = array.array("h")
        pcm.frombytes(raw[: (len(raw) // 2) * 2])
        peak = max(abs(min(pcm)), abs(max(pcm))) / FULL_SCALE
        rms = math.sqrt(sum(float(s) * s for s in pcm) / len(pcm)) / FULL_SCALE
    else:
        x = np.frombuffer(raw, dtype="<i2").astype(np.float64) / FULL_SCALE
        peak = float(np.abs(x).max())
        rms = float(np.sqrt(np.mean(x ** 2)))
    return {**base, "peak_dbfs": round(db(peak), 2), "rms_dbfs": round(db(rms), 2)}


# --------------------------------------------------------------------------- primitives


def detect_glitches(x: Any, *, win: int = 4800) -> Any:
    """Second-difference outliers with a per-window adaptive threshold.

    One index per burst (glitch_analyze.detect_glitches). The threshold is floored at
    a fraction of the window's own peak so a quiet stretch does not make ordinary
    waveform curvature look like a discontinuity.
    """
    np = _np()
    if len(x) < 3:
        return np.array([], dtype=np.int64)
    d2 = np.abs(np.diff(x, 2))
    idxs: list[int] = []
    win = max(64, int(win))
    for start in range(0, max(len(d2) - win, 0), win):
        seg = d2[start:start + win]
        med = float(np.median(seg))
        limit = max(12 * med + 1e-4, 0.35 * float(np.abs(x[start:start + win + 2]).max()))
        hits = np.flatnonzero(seg > limit)
        if len(hits) == 0:
            continue
        for g in np.split(hits, np.flatnonzero(np.diff(hits) > 32) + 1):
            idxs.append(int(start + g[int(np.argmax(seg[g]))]))
    return np.array(idxs, dtype=np.int64)


def click_metric(x: Any, sr: int, *, thresh: float = CLICK_THRESH) -> tuple[int, list[float]]:
    """Count d2 spikes far above the *local* RMS (hil_rt_glitch.click_metric).

    A hard single-sample pop jumps by many times the local level; a band-limited
    musical transient does not, which is why the comparison is local and not global.
    """
    np = _np()
    win = max(1, sr // 100)                    # ~10 ms
    n = (len(x) // win) * win
    if n < 2 * win:
        return 0, []
    d2 = np.abs(np.diff(x, n=2))
    rms = np.sqrt(np.mean(x[:n].reshape(-1, win) ** 2, axis=1)) + 1e-5
    per_sample = np.repeat(rms, win)
    d2 = d2[:len(per_sample)]
    hits = np.flatnonzero(d2 > thresh * per_sample[:len(d2)])
    clicks: list[int] = []
    last = -(10 ** 9)
    gap = max(1, sr // 500)                    # 2 ms apart = a separate click
    for h in hits:
        if h - last > gap:
            clicks.append(int(h))
        last = int(h)
    return len(clicks), [round(c / sr, 4) for c in clicks[:20]]


def envelope(x: Any, sr: int, *, win_ms: float = 1.0) -> Any:
    np = _np()
    win = max(1, int(sr * win_ms / 1000))
    return np.convolve(np.abs(x), np.ones(win) / win, mode="same")


def detect_onsets(x: Any, sr: int, floor_db: float, *,
                  refractory_s: float = ONSET_REFRACTORY_S,
                  rise_db: float = ONSET_RISE_DB) -> tuple[list[int], Any]:
    """Where the envelope *climbs* by ``rise_db``, at most one hit per refractory window.

    A hit is a rise, not a level. Two earlier rules both failed on real captures,
    in opposite directions, and both are why this one is written the way it is:

    * "count a crossing of floor + rise, re-arm when it falls back under" loses
      every hit that lands on the previous one's tail. A 4 hits/s DRUMS capture
      reported 12 of 16 on a board that had played all 16.
    * "…or re-arm once it climbs back above its trough" then counted the ripple
      inside a decay: 27 onsets for the same 16 hits.

    Measuring the climb over a short window answers both. The level still has to
    clear the noise floor, so a rise inside the noise is not a hit, and the floor
    is measured rather than derived from the loudest hit — a threshold anchored
    to the peak swallows the quiet hits a velocity sweep exists to see.
    """
    np = _np()
    env = envelope(x, sr)
    if len(env) == 0:
        return [], env
    thresh = 10 ** ((floor_db + rise_db) / 20.0)
    loud = 20.0 * np.log10(np.maximum(env, 1e-12))
    lag = max(1, int(ONSET_RISE_WINDOW_S * sr))
    if len(loud) <= lag:
        return ([0] if bool(env[0] > thresh) else []), env
    climb = loud[lag:] - loud[:-lag]
    candidates = np.flatnonzero((climb > rise_db) & (env[lag:] > thresh))
    refractory = max(1, int(refractory_s * sr))
    onsets: list[int] = []
    # A step at time T makes every window starting in [T - lag, T] look like a
    # climb, so the run of candidates *ends* at the hit. Reporting where the run
    # starts would put every onset a full window early, and the reported latency
    # is then systematically wrong by that much rather than by the pipeline's
    # real offset.
    for c in candidates:
        i = int(c)
        if onsets and i - onsets[-1] <= 1:
            onsets[-1] = i
        elif not onsets or i - onsets[-1] >= refractory:
            onsets.append(i)
    return onsets, env


def match_to_schedule(onsets: Sequence[int], schedule: Sequence[float], sr: int, *,
                      tol_s: float = ONSET_TOL_S) -> tuple[list[int | None], list[int], float]:
    """Pair onsets with the known send grid; returns (matched, extra_indices, offset_s).

    Counting crossings one-to-one cannot work: real samples have secondary transients
    that re-cross the threshold. Anchoring to the grid keeps the crossing nearest each
    scheduled hit and reports the rest separately (hil_velocity.match_to_schedule).
    """
    if not onsets or not schedule:
        return [None] * len(schedule), list(range(len(onsets))), 0.0
    offset = onsets[0] / sr - schedule[0]
    used: set[int] = set()
    matched: list[int | None] = []
    for t_send in schedule:
        target = t_send + offset
        best, best_d = None, tol_s
        for i, o in enumerate(onsets):
            if i in used:
                continue
            d = abs(o / sr - target)
            if d < best_d:
                best, best_d = i, d
        if best is None:
            matched.append(None)
        else:
            used.add(best)
            matched.append(onsets[best])
    return matched, [i for i in range(len(onsets)) if i not in used], offset


def align_to_ref(cap: Any, ref: Any, sr: int, *, search_s: float = 3.0) -> tuple[int, float]:
    """Lag of ``ref`` inside ``cap`` by cross-correlation on a 1 s probe.

    Returns (lag_samples, normalized correlation peak). A low peak means the capture
    does not contain the reference at all — check that before trusting the lag.

    A *periodic* reference (a single tone) has no unique lag: every period-aligned
    offset correlates equally, so the peak is trustworthy and the lag is not. Align
    against broadband material — a chirp, a multitone, the real program.
    """
    np = _np()
    probe = ref[:sr]
    seg = cap[: int(search_s * sr) + sr]
    if len(probe) == 0 or len(seg) < len(probe):
        raise HilError(BAD_ARGS, "capture is shorter than the 1 s alignment probe")
    sp = _sp_signal()
    corr = (sp.correlate(seg, probe, mode="valid") if sp is not None
            else np.correlate(seg, probe, mode="valid"))
    lag = int(np.argmax(np.abs(corr)))
    denom = float(np.linalg.norm(probe) * np.linalg.norm(seg[lag:lag + len(probe)])) + 1e-12
    return lag, float(np.abs(corr[lag]) / denom)


PITCH_MIN_HZ = 20.0     # ignore DC/rumble bins below this when no search band is given


def _windowed_mag_spectrum(seg: Any, sr: int) -> tuple[Any, Any]:
    """Hann-windowed magnitude spectrum of one segment: ``(freqs, |rfft|)``.

    Every single-shot (non-Welch) FFT peak-pick in this module starts here —
    :func:`analyze_pitch`'s parabolic-interpolated peak and
    :func:`analyze_multitone`'s per-tone dominant-frequency search both read a
    magnitude spectrum off the same window; only what they do with the bins
    afterwards differs.
    """
    np = _np()
    win = np.hanning(len(seg))
    freqs = np.fft.rfftfreq(len(seg), 1.0 / sr)
    mag = np.abs(np.fft.rfft(seg * win))
    return freqs, mag


def analyze_pitch(wav: str | Path | Any, rate: int | None = None, t0: float = 0.0,
                  t1: float | None = None, *, expected_hz: float | None = None,
                  search_hz: tuple[float, float] | None = None,
                  channel: int | str = AUTO_CHANNEL) -> dict:
    """Dominant frequency of one slice: windowed FFT + parabolic peak interpolation.

    Built for a steady sine — what a pitched-kit pad renders — not a general pitch
    tracker. ``wav`` is either a path (``rate`` then comes from the file) or a 1-D
    array of samples in [-1, 1] (then ``rate`` is required). ``t0``/``t1`` slice the
    signal in seconds (``t1=None`` keeps the tail).

    ``search_hz`` restricts the peak search to one band — pass it to pick a *named*
    partial out of a chord rather than whatever bin happens to be loudest; its return
    ``peak_db`` is then directly comparable across separate calls against the *same*
    slice (same window length, same FFT), which is how a chord's peaks are ranked
    against each other without a multi-pitch detector. ``expected_hz``, if given, adds
    ``cents_vs`` — the measured pitch's deviation from it, in cents.
    """
    np = _np()
    if isinstance(wav, (str, Path)):
        x, sr, _used = read_wav_ch(wav, channel)
    else:
        x = np.asarray(wav, dtype=np.float64)
        if rate is None:
            raise HilError(BAD_ARGS, "analyze_pitch: rate is required for an array input")
        sr = int(rate)
    n0 = max(0, int(t0 * sr))
    n1 = len(x) if t1 is None else min(len(x), int(t1 * sr))
    seg = np.asarray(x[n0:n1], dtype=np.float64)
    if len(seg) < 64:
        raise HilError(BAD_ARGS,
                       f"analyze_pitch: slice [{t0:g},{t1}] is only {len(seg)} samples")
    freqs, mag = _windowed_mag_spectrum(seg, sr)
    if search_hz is not None:
        lo, hi = search_hz
        band = (freqs >= lo) & (freqs <= hi)
        if not bool(band.any()):
            raise HilError(BAD_ARGS, f"analyze_pitch: no FFT bin in [{lo:g},{hi:g}] Hz "
                                     f"(slice is {sr / len(seg):g} Hz/bin)")
        idxs = np.flatnonzero(band)
        k = int(idxs[int(np.argmax(mag[band]))])
    else:
        lo_bin = max(1, int(round(PITCH_MIN_HZ * len(seg) / sr)))
        k = lo_bin + int(np.argmax(mag[lo_bin:]))
    # Parabolic interpolation on the three bins around the peak (classic FFT
    # sub-bin refinement) — a plain bin pick is off by up to half a bin width,
    # which at 44.1 kHz / 4096 pts is +-5 Hz: enough to blow a 10-cent budget.
    if 0 < k < len(mag) - 1:
        a, b, c = float(mag[k - 1]), float(mag[k]), float(mag[k + 1])
        denom = a - 2 * b + c
        delta = 0.5 * (a - c) / denom if denom != 0 else 0.0
        delta = max(-1.0, min(1.0, delta))
    else:
        delta = 0.0
    bin_hz = sr / len(seg)
    hz = float((k + delta) * bin_hz)
    peak_mag = float(mag[k])
    peak_db = round(20.0 * math.log10(peak_mag + 1e-12), 2)
    total_mag = float(np.sum(mag)) + 1e-30
    confidence = round(peak_mag / total_mag, 4)
    rms = float(np.sqrt(np.mean(seg ** 2))) if len(seg) else 0.0
    cents_vs = (1200.0 * math.log2(hz / expected_hz)) if expected_hz and hz > 0 else None
    return {
        "hz": round(hz, 3),
        "expected_hz": round(expected_hz, 3) if expected_hz else None,
        "cents_vs": round(cents_vs, 2) if cents_vs is not None else None,
        "level_dbfs": round(db(rms), 2),
        "peak_db": peak_db,
        "confidence": confidence,
        "t0": round(n0 / sr, 4),
        "t1": round(n1 / sr, 4),
        "sample_rate": sr,
        "bin_hz": round(bin_hz, 4),
    }


# --------------------------------------------------------------------------- tone tracks

TONE_FRAME_MS = 20.0            # one level sample per 20 ms of a sounding tone
TONE_BAND_HZ = 15.0             # +-15 Hz around a candidate is "that tone"
TONE_BAND_PROBES = 5            # offsets across the band; max wins (see below)
TONE_DROPOUT_DB = 10.0          # this far under the window median is a hole
TONE_FLOOR_MARGIN_DB = 20.0     # a tone must clear the pre-hit floor by this to count
TONE_ATTACK_S = 0.05            # trimmed off the front of a hit's window
TONE_RELEASE_S = 0.05           # trimmed off the back, before the fade
TONE_PRE_S = 0.03               # silence sampled just before a hit, for its noise floor
TONE_SWITCH_MIN_FRAMES = 3      # a switch has to hold this long to be one (see below)


def _tone_levels(frames: Any, sr: int, cands: Sequence[float], band_hz: float) -> Any:
    """(n_frames, n_candidates) dB level of each candidate tone in each frame.

    A Hann-windowed DFT evaluated *at the candidate frequency itself*, not at
    whatever FFT bin is nearest: a 20 ms frame is 50 Hz per bin at 44.1 kHz,
    and the two kits' tones for one pad sit 111 Hz apart — barely two bins,
    inside a Hann mainlobe. Asking each named frequency directly puts the other
    candidate past the mainlobe, where the window's first sidelobe is -31 dB,
    so "which of these two is sounding" is answerable per frame.

    The band is sampled at `TONE_BAND_PROBES` offsets across +-`band_hz` and the
    strongest wins: a tone measured a few Hz off (host clock against device
    clock) still lands, rather than being attenuated by its own phase spread
    across the frame and read as a dropout.
    """
    np = _np()
    n = frames.shape[1]
    win = np.hanning(n)
    idx = np.arange(n, dtype=np.float64)
    offsets = np.linspace(-band_hz, band_hz, TONE_BAND_PROBES)
    probes = np.array([f + o for f in cands for o in offsets], dtype=np.float64)
    basis = np.exp(-2j * np.pi * np.outer(idx, probes) / sr) * win[:, None]
    mag = np.abs(frames @ basis) * (2.0 / max(float(win.sum()), 1e-12))
    # (n_frames, n_cands, n_probes) -> the best probe of each candidate's band
    mag = mag.reshape(frames.shape[0], len(cands), TONE_BAND_PROBES).max(axis=2)
    # Same -120 dB floor `db()` uses: digital silence reads as -120, not -240.
    return np.maximum(20.0 * np.log10(np.maximum(mag, 1e-12)), -120.0)


def _framed(x: Any, sr: int, t0: float, t1: float, frame_n: int) -> Any:
    """`[t0, t1)` cut into whole `frame_n` frames; an empty (0, frame_n) array when it
    does not hold one."""
    np = _np()
    n0 = max(0, int(round(t0 * sr)))
    n1 = min(len(x), int(round(t1 * sr)))
    count = max(0, (n1 - n0) // frame_n)
    if count == 0:
        return np.zeros((0, frame_n), dtype=np.float64)
    return np.asarray(x[n0:n0 + count * frame_n], dtype=np.float64).reshape(count, frame_n)


def _runs(mask: Any) -> list[tuple[int, int]]:
    """`(start, length)` of every run of True in a 1-D boolean array."""
    out: list[tuple[int, int]] = []
    start = -1
    for i, flag in enumerate(mask):
        if flag and start < 0:
            start = i
        elif not flag and start >= 0:
            out.append((start, i - start))
            start = -1
    if start >= 0:
        out.append((start, len(mask) - start))
    return out


def _longest_run(mask: Any) -> int:
    """Length of the longest run of True in a 1-D boolean array."""
    return max((n for _start, n in _runs(mask)), default=0)


def _leading_run(mask: Any) -> int:
    """Length of the run of True at index 0, or 0 when the array does not start True."""
    runs = _runs(mask)
    return runs[0][1] if runs and runs[0][0] == 0 else 0


def analyze_tone_tracks(wav: str | Path | Any, hits: Sequence[Mapping[str, Any]], *,
                        tone_s: float, freqs: Mapping[int, Sequence[float]],
                        rate: int | None = None, frame_ms: float = TONE_FRAME_MS,
                        dropout_db: float = TONE_DROPOUT_DB,
                        attack_s: float = TONE_ATTACK_S,
                        release_s: float = TONE_RELEASE_S,
                        band_hz: float = TONE_BAND_HZ,
                        floor_margin_db: float = TONE_FLOOR_MARGIN_DB,
                        pre_s: float = TONE_PRE_S,
                        switch_min_frames: int = TONE_SWITCH_MIN_FRAMES,
                        channel: int | str = AUTO_CHANNEL) -> dict:
    """Score a recording of overlapping fixed-length tones, one window per hit.

    Built for `kit_swap_audio`: sixteen pads round-robin at a few hits/s, each
    playing a 2 s sine whose frequency says *which kit the pad is currently
    loaded with*. `hits` is `[{"t": seconds into the WAV, "pad": int}, ...]` and
    `freqs` maps a pad to its candidate frequencies — here `[kit A Hz, kit B Hz]`,
    but the kernel takes any number.

    Per hit, over `[t + attack_s, t + tone_s - release_s]`, every `frame_ms`
    frame gets a level for each candidate (`_tone_levels`) and the loudest
    candidate is that frame's *dominant*. The hit's `kit` is the candidate
    dominant in the most frames — or `None` when it never rises
    `floor_margin_db` above the same candidate's level in the `pre_s` of
    recording just before the hit, which is the shape of a pad that made no
    sound at all.

    Then, on the dominant tone's own level track:

    * **`late_start_ms`** — the *leading* run of quiet frames, before the tone
      arrives at all. A voice that waited for its first refill starts late; it
      does not have a hole in it, and the two have different causes and
      different fixes. Measured on a real board: a pad hit at t=5.50 whose tone
      began at ~5.72. This run is reported on its own and **excluded** from
      every dropout number below, so a late start is never also counted as a
      220 ms gap in the middle of a tone that in fact played cleanly.
    * **`dropout_frames`** — frames more than `dropout_db` under the window's
      median level, *after* the leading run: the tone is still the right one and
      it did arrive, but it stopped coming out. `dropout_ms` is that in
      milliseconds and `max_gap_ms` is the longest consecutive run, which is the
      number that separates "a block or two" from "an audible hole".
    * **`switch_frames`** — frames where a *different* candidate is dominant
      **and** is itself within `dropout_db` of the window median. The second
      half of that test is what keeps a hole from being reported as a switch: in
      a dropout both candidates collapse and which of them wins is noise, while
      a pad genuinely reloaded mid-tone brings the other tone up at full level.

      That is still not enough on its own. On clean captures single isolated
      frames flip to the other candidate — a harmonic, or noise landing in the
      band for one 20 ms window — and a pad cannot be reloaded for 20 ms and
      back again. So `switch_frames` is reported raw, and beside it
      `switch_run_frames` counts only frames inside a run of at least
      `switch_min_frames` (3, i.e. 60 ms at the default frame). **A verdict must
      read `switch_run_frames`**; the raw count is there to show what was
      discarded. `switch_max_run_ms` is the longest such run.

    Windows never overlap for one pad — 16 pads at 4 hits/s revisit a pad every
    4 s against a 2 s tone — and other pads' tones sit in bands this measurement
    does not look at, so no de-mixing is needed.
    """
    np = _np()
    if isinstance(wav, (str, Path)):
        x, sr, _used = read_wav_ch(wav, channel)
    else:
        x = np.asarray(wav, dtype=np.float64)
        if rate is None:
            raise HilError(BAD_ARGS, "analyze_tone_tracks: rate is required for an array input")
        sr = int(rate)
    frame_n = max(16, int(round(sr * frame_ms / 1000.0)))
    records: list[dict[str, Any]] = []
    for hit in hits:
        pad = int(hit["pad"])
        t = float(hit["t"])
        cands = [float(f) for f in freqs.get(pad, ())]
        rec: dict[str, Any] = {
            "pad": pad, "t": round(t, 4), "kit": None, "kit_hz": None, "missing": True,
            "frames": 0, "late_start_ms": 0.0,
            "dropout_frames": 0, "dropout_ms": 0.0, "max_gap_ms": 0.0,
            "switch_frames": 0, "switch_run_frames": 0, "switch_max_run_ms": 0.0,
            "median_db": None, "peak_db": None, "floor_db": None,
            "dominant_counts": [], "note": None,
        }
        if not cands:
            rec["note"] = "no candidate frequencies for this pad"
            records.append(rec)
            continue
        frames = _framed(x, sr, t + attack_s, t + tone_s - release_s, frame_n)
        if frames.shape[0] < 2:
            rec["note"] = "window is shorter than two frames (recording ended?)"
            records.append(rec)
            continue
        levels = _tone_levels(frames, sr, cands, band_hz)
        rec["frames"] = int(frames.shape[0])

        pre = _framed(x, sr, t - pre_s, t, frame_n)
        floors = (np.median(_tone_levels(pre, sr, cands, band_hz), axis=0)
                  if pre.shape[0] > 0 else np.full(len(cands), -120.0))

        dominant = np.argmax(levels, axis=1)
        counts = np.bincount(dominant, minlength=len(cands))
        rec["dominant_counts"] = [int(c) for c in counts]
        kit = int(np.argmax(counts))
        peak = float(levels[:, kit].max())
        rec["peak_db"] = round(peak, 2)
        rec["floor_db"] = round(float(floors[kit]), 2)
        if peak - float(floors[kit]) < floor_margin_db:
            rec["note"] = "never rose above the pre-hit noise floor"
            records.append(rec)
            continue

        rec["kit"], rec["kit_hz"], rec["missing"] = kit, round(cands[kit], 3), False
        track = levels[:, kit]
        median = float(np.median(track))
        rec["median_db"] = round(median, 2)
        quiet = track < (median - dropout_db)
        # The leading run is the voice arriving late, not a hole in a tone that
        # was already sounding: report it on its own and take it out of the
        # dropout mask, or one late start reads as the run's worst gap.
        lead = _leading_run(quiet)
        rec["late_start_ms"] = round(lead * frame_ms, 2)
        holes = quiet.copy()
        holes[:lead] = False
        rec["dropout_frames"] = int(holes.sum())
        rec["dropout_ms"] = round(float(holes.sum()) * frame_ms, 2)
        rec["max_gap_ms"] = round(_longest_run(holes) * frame_ms, 2)
        loud = levels[np.arange(len(dominant)), dominant] >= (median - dropout_db)
        switched = (dominant != kit) & loud
        rec["switch_frames"] = int(switched.sum())
        runs = [n for _start, n in _runs(switched) if n >= switch_min_frames]
        rec["switch_run_frames"] = int(sum(runs))
        rec["switch_max_run_ms"] = round(max(runs, default=0) * frame_ms, 2)
        records.append(rec)

    totals = {
        "hits": len(records),
        "missing": sum(1 for r in records if r["missing"]),
        "audible": sum(1 for r in records if not r["missing"]),
        "dropout_frames": sum(int(r["dropout_frames"]) for r in records),
        "dropout_ms": round(sum(float(r["dropout_ms"]) for r in records), 2),
        "max_gap_ms": round(max((float(r["max_gap_ms"]) for r in records), default=0.0), 2),
        "switch_frames": sum(int(r["switch_frames"]) for r in records),
        "switch_run_frames": sum(int(r["switch_run_frames"]) for r in records),
        "late_starts": sum(1 for r in records if float(r["late_start_ms"]) > 0),
        "late_start_max_ms": round(
            max((float(r["late_start_ms"]) for r in records), default=0.0), 2),
        "hits_with_dropouts": sum(1 for r in records if r["dropout_frames"]),
        # A hit "switched" only if a run survived the length rule — the verdict
        # reads this, so it must not be counted off the raw frame tally.
        "hits_with_switches": sum(1 for r in records if r["switch_run_frames"]),
    }
    return {"hits": records, "totals": totals, "sample_rate": sr,
            "frame_ms": frame_ms, "tone_s": tone_s, "dropout_db": dropout_db,
            "band_hz": band_hz, "switch_min_frames": switch_min_frames}


def _median_filter(x: Any, kernel: int) -> Any:
    """Odd-length running median, replicating the edge samples.

    ``scipy.signal.medfilt`` pads with zeros, which invents a step at both ends of a
    phase residual — the one thing the slip counter is looking for.
    """
    np = _np()
    k = max(1, int(kernel) | 1)
    if len(x) <= k:
        return x
    padded = np.pad(x, k // 2, mode="edge")
    return np.median(np.lib.stride_tricks.sliding_window_view(padded, k), axis=-1)


def phase_residual(x: Any, sr: int, freq: float, *, bw_hz: float = PHASE_BAND_HZ,
                   presence_bw_hz: float = PHASE_PRESENCE_BW_HZ,
                   edge_s: float = 0.5) -> tuple[Any, float, float]:
    """Instantaneous phase around `freq`, minus its best-fit ramp (glitch_analyze).

    A tone that arrives intact advances its phase by the same amount every sample, so
    the residual is flat. This is where lost and repeated samples show up and nowhere
    else: a dropped sample steps the residual by ``2*pi*freq/sr`` and leaves the
    waveform otherwise unremarkable, which is how a stream that is quietly slipping
    passes a glitch counter.

    The band is selected in the frequency domain and the negative half dropped in the
    same pass, so the analytic signal costs one FFT and needs no scipy. Two details are
    load-bearing. The band is *tapered*, because a brick wall rings: with a rectangular
    mask a one-sample step arrives smeared over tens of milliseconds and the slip
    counter cannot see it at all. And the band is wide (±1 kHz, not the ±200 Hz a
    bandpass for a tone would suggest), because narrowing it in frequency is the same
    thing as blurring the step in time.

    `edge_s` is cut from both ends before the fit and from the result: an FFT treats the
    recording as circular and the wrap lands in those edges.

    Returns ``(residual_rad, est_freq_hz, band_energy_ratio)``. The ratio is measured
    over `presence_bw_hz`, not over the wide tracking band — it answers "was there a
    tone at this frequency", which a band wide enough to see steps would answer yes to
    for a tone most of a kilohertz away.
    """
    np = _np()
    n = len(x)
    if n < 64:
        raise HilError(BAD_ARGS, "recording is too short for a phase measurement")
    spec = np.fft.rfft(x)
    fbins = np.fft.rfftfreq(n, 1.0 / sr)
    off = np.abs(fbins - freq)
    band = off <= bw_hz
    if not bool(band.any()):
        raise HilError(BAD_ARGS, f"no FFT bin within {bw_hz:g} Hz of {freq:g} Hz")
    power = np.abs(spec) ** 2
    near = off <= presence_bw_hz
    band_ratio = float(power[near].sum()) / (float(power.sum()) + 1e-30)
    taper = np.zeros(len(spec))
    taper[band] = 0.5 * (1.0 + np.cos(np.pi * off[band] / bw_hz))
    full = np.zeros(n, dtype=complex)
    full[:len(spec)] = spec * taper
    ph = np.unwrap(np.angle(np.fft.ifft(full)))
    trim = int(max(edge_s, 0.0) * sr)
    if n - 2 * trim < max(64, sr // 10):
        trim = 0
    inner = slice(trim, n - trim) if trim else slice(None)
    idx = np.arange(n)[inner]
    fit = np.vstack([idx, np.ones(len(idx))]).T
    coef, *_ = np.linalg.lstsq(fit, ph[inner], rcond=None)
    return ph[inner] - (coef[0] * idx + coef[1]), float(coef[0] * sr / (2 * math.pi)), band_ratio


def timing_slips(resid: Any, sr: int, freq: float, *, grid_ms: float = 1.0,
                 smooth_ms: float = 21.0, lag_ms: float = 2.0,
                 min_fraction: float = 0.5) -> tuple[int, float, float]:
    """Steps in a phase residual, counted in whole samples (glitch_analyze.analyze_sine).

    Returns ``(slips, net_drift_samples, residual_p2p_rad)``. The residual is decimated
    to a 1 ms grid and median-filtered before differencing, because a median filter
    keeps a step it found but erases a transient shorter than its window — which is
    exactly the difference between a stream that slipped and one that popped.

    The difference is taken across `lag_ms`, not between neighbours, since band-limiting
    spreads even an instantaneous step over a millisecond or two; and a run of
    above-threshold differences is one slip, not one per grid point it took to get
    there. Over a couple of milliseconds an honest clock error contributes nothing, so
    the wider difference costs no sensitivity.

    `net_drift_samples` is what is left after the linear fit, so a constant rate error
    is *not* drift (it is reported as a frequency offset); a rate that changes mid-take
    is.
    """
    np = _np()
    per_sample = 2 * math.pi * freq / sr        # phase one whole sample is worth
    if per_sample <= 0:
        raise HilError(BAD_ARGS, f"freq must be > 0, got {freq!r}")
    step_n = max(1, int(sr * grid_ms / 1000.0))
    rm = _median_filter(resid[::step_n], int(round(smooth_ms / grid_ms)))
    lag = max(1, int(round(lag_ms / grid_ms)))
    if len(rm) <= lag:
        return 0, 0.0, 0.0
    above = np.abs(rm[lag:] - rm[:-lag]) > min_fraction * per_sample
    slips = int(np.sum(np.diff(above.astype(np.int8)) == 1) + (1 if above[0] else 0))
    return slips, float((rm[-1] - rm[0]) / per_sample), float(np.ptp(rm))


def _welch(x: Any, sr: int, nperseg: int) -> tuple[Any, Any]:
    """scipy.signal.welch when scipy is installed, a Hann-window equivalent when not."""
    np = _np()
    sp = _sp_signal()
    nperseg = min(nperseg, len(x))
    if nperseg < 16:
        raise HilError(BAD_ARGS, "recording is too short for a PSD")
    if sp is not None:
        f, p = sp.welch(x, sr, nperseg=nperseg)
        return f, p
    win = np.hanning(nperseg)
    scale = 1.0 / (sr * float((win ** 2).sum()))
    step = nperseg // 2
    acc = None
    count = 0
    for start in range(0, len(x) - nperseg + 1, step):
        spec = np.abs(np.fft.rfft(x[start:start + nperseg] * win)) ** 2
        acc = spec if acc is None else acc + spec
        count += 1
    p = (acc / max(count, 1)) * scale
    p[1:-1] *= 2.0
    return np.fft.rfftfreq(nperseg, 1.0 / sr), p


def glitch_periodicity(idxs: Any, n_samples: int, sr: int, *, top: int = 8) -> dict:
    """Inter-arrival structure of a glitch train (glitch_analyze.periodicity).

    Glitches that arrive on a clock are a different defect from glitches that arrive at
    random: a dominant interval points at a buffer boundary. The histogram alone is
    smeared by jitter, so the impulse train is also given to Welch, which names the rate
    even when no single interval repeats exactly.
    """
    np = _np()
    idxs = np.asarray(idxs, dtype=np.int64)
    if len(idxs) < 3:
        return {}
    iv = np.diff(np.sort(idxs)) / float(sr) * 1000.0
    vals, counts = np.unique(np.round(iv, 1), return_counts=True)
    order = np.argsort(counts)[::-1][:max(1, top)]
    out: dict = {"intervals_top": {f"{vals[i]:.1f}ms": int(counts[i]) for i in order},
                 "iv_median_ms": round(float(np.median(iv)), 2)}
    n_samples = int(n_samples)
    if n_samples >= 1024:
        train = np.zeros(n_samples)
        train[idxs[(idxs >= 0) & (idxs < n_samples)]] = 1.0
        f, p = _welch(train, sr, min(1 << 16, n_samples))
        band = (f > 5) & (f < 2000)
        if bool(band.any()):
            # A perfectly regular train is a comb whose harmonics are all the same
            # height, so the tallest bin is an arbitrary multiple of the rate. The rate
            # is the *lowest* peak that reaches the top of the comb.
            pb = p[band]
            peak = float(pb.max())
            out["train_peak_hz"] = round(float(f[band][int(np.argmax(pb >= peak * 0.5))]), 1)
    return out


def glitch_determinism(runs: Sequence[Any], *,
                       tol_samples: int = GLITCH_TOL_SAMPLES) -> float | None:
    """Fraction of the first run's glitch positions that recur in every other run.

    Near 1.0 the glitches are structural — the same sample offsets on every take, so
    something in the pipeline does the same wrong thing at the same point, and a fix is
    findable. Near 0 they are incidental. Only meaningful for takes that share a time
    origin: captures that were never aligned read as non-deterministic however
    reproducible the defect is.
    """
    np = _np()
    if len(runs) < 2:
        return None
    base = np.asarray(runs[0], dtype=np.int64)
    others = [np.asarray(r, dtype=np.int64) for r in runs[1:]]
    if len(base) == 0:
        return None
    matched = sum(1 for pos in base
                  if all(len(o) and int(np.min(np.abs(o - pos))) <= tol_samples
                         for o in others))
    return round(matched / len(base), 3)


def write_reference(path: str | Path, freqs: Sequence[float] = MULTITONE_FREQS,
                    seconds: float = 5.0, *, level: float = 0.7, sr: int = 48000,
                    channels: int = 2) -> Path:
    """Write the multitone reference `analyze_multitone` expects to see come back.

    10 ms fades at both ends (hil_audio_loopback.write_reference): starting a tone at
    full amplitude puts a click in the reference itself, which the glitch counter would
    then dutifully report on every single run.
    """
    np = _np()
    freqs = [float(f) for f in freqs]
    if not freqs:
        raise HilError(BAD_ARGS, "a reference needs at least one frequency")
    t = np.arange(int(sr * seconds)) / sr
    sig = np.zeros_like(t)
    amp = level / len(freqs)
    for f in freqs:
        sig += amp * np.sin(2 * np.pi * f * t)
    fade = max(1, int(0.010 * sr))
    env = np.ones_like(sig)
    env[:fade] = np.linspace(0, 1, fade)
    env[-fade:] = np.linspace(1, 0, fade)
    pcm = (sig * env * 32767).astype("<i2")
    frames = np.column_stack([pcm] * channels).ravel() if channels > 1 else pcm
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(p), "wb") as w:
        w.setnchannels(channels)
        w.setsampwidth(2)
        w.setframerate(sr)
        w.writeframes(frames.tobytes())
    return p


# --------------------------------------------------------------------------- verdicts


@dataclass
class Verdict:
    """Common head of every analysis result: what was measured and whether it passed."""

    kind: str
    wav: str
    verdict: str                 # "pass" | "warn" | "fail"
    reasons: list[str] = field(default_factory=list)
    seconds: float = 0.0
    sample_rate: int = 0
    channels: int = 0
    # Which channel the numbers describe, once "auto" has chosen one. None for a
    # verdict that reports every channel separately, and for mono captures.
    channel: int | None = None

    @property
    def passed(self) -> bool:
        return self.verdict != "fail"

    def to_dict(self) -> dict:
        return {**asdict(self), "passed": self.passed}


@dataclass
class MultitoneVerdict(Verdict):
    signal_present: bool = False
    rms_dbfs: float = -120.0
    tones: dict = field(default_factory=dict)
    hf_junk_ratio: float = 0.0
    glitch_bursts_per_s: float = 0.0
    dropouts: int = 0
    analyzed_s: float = 0.0


@dataclass
class ClickVerdict(Verdict):
    clicks: int = 0
    clicks_per_s: float = 0.0
    first_click_offsets_s: list[float] = field(default_factory=list)
    rms_dbfs: float = -120.0
    peak_dbfs: float = -120.0


@dataclass
class OnsetVerdict(Verdict):
    expected: int = 0
    matched: int = 0
    missed: list[int] = field(default_factory=list)
    extra: list[float] = field(default_factory=list)
    latency_ms: dict = field(default_factory=dict)
    onsets_s: list[float] = field(default_factory=list)
    noise_floor_dbfs: float = -120.0
    offset_ms: float | None = None


@dataclass
class VelocityVerdict(Verdict):
    expected: int = 0
    matched: int = 0
    missed: list[int] = field(default_factory=list)
    curve: list[dict] = field(default_factory=list)
    per_hit: list[dict] = field(default_factory=list)
    dynamic_range_db: float = 0.0
    monotonic_inversions: int = 0
    latency_ms: dict = field(default_factory=dict)
    noise_floor_dbfs: float = -120.0


@dataclass
class PsdVerdict(Verdict):
    freq_hz: float = 0.0
    carrier_db: float | None = None
    carrier_dbfs: float | None = None
    dom_freq_hz: float | None = None
    sidebands: dict = field(default_factory=dict)
    harmonics: dict = field(default_factory=dict)
    hum: dict = field(default_factory=dict)
    align: dict | None = None


@dataclass
class SilenceVerdict(Verdict):
    rms_dbfs: float = -120.0
    peak_dbfs: float = -120.0
    glitches: int = 0
    glitch_per_s: float = 0.0
    per_channel: dict = field(default_factory=dict)


@dataclass
class SineVerdict(Verdict):
    freq_hz: float = 0.0
    est_freq_hz: float | None = None
    freq_offset_pct: float | None = None
    rms_dbfs: float = -120.0
    peak_dbfs: float = -120.0
    glitches: int = 0
    glitch_per_s: float = 0.0
    timing_slips: int = 0
    net_drift_samples: float = 0.0
    phase_resid_p2p_rad: float = 0.0
    periodicity: dict = field(default_factory=dict)
    align: dict | None = None
    determinism: dict | None = None

# --------------------------------------------------------------------------- kinds


def _head(kind: str, path: Path, sr: int, ch: int, frames: int,
          channel: int | None = None) -> dict:
    head = {"kind": kind, "wav": str(path), "sample_rate": sr, "channels": ch,
            "seconds": round(frames / float(sr or 1), 3)}
    # Which channel the verdict is about: with "auto" that is a decision the
    # reader has to be able to see, not one made silently.
    if ch > 1:
        head["channel"] = channel
    return head


def analyze_multitone(wav: str | Path, *, freqs: Sequence[float] = MULTITONE_FREQS,
                      min_rms_dbfs: float = MULTITONE_MIN_RMS_DBFS,
                      warn_glitch_per_s: float = WARN_GLITCH_PER_S,
                      channel: int | str = AUTO_CHANNEL) -> MultitoneVerdict:
    """Is the reference multitone coming back, at level, in tune and without junk?"""
    np = _np()
    path = Path(wav)
    x, sr, used_ch = read_wav_ch(path, channel)
    _raw, _sr, ch, frames = _open_wav(path)
    head = _head("multitone", path, sr, ch, frames, used_ch)
    freqs = [float(f) for f in freqs]
    if not freqs:
        raise HilError(BAD_ARGS, "multitone analysis needs at least one frequency")

    # Locate the loud region with 100 ms RMS blocks: a capture has silence at both ends.
    blk = max(1, sr // 10)
    n_blocks = len(x) // blk
    if n_blocks < 5:
        return MultitoneVerdict(**head, verdict="fail",
                                reasons=[f"recording is only {len(x) / sr:.2f} s"])
    rms_blocks = np.sqrt(np.mean(x[:n_blocks * blk].reshape(n_blocks, blk) ** 2, axis=1))
    thresh = max(float(rms_blocks.max()) * 0.3, 20.0 / FULL_SCALE)
    active = np.flatnonzero(rms_blocks > thresh)
    start = (int(active[0]) + 2) * blk if len(active) else 0
    end = (int(active[-1]) - 1) * blk if len(active) else 0
    if len(active) < 5 or end - start < sr:
        return MultitoneVerdict(**head, verdict="fail", signal_present=False,
                                rms_dbfs=round(db(float(np.sqrt(np.mean(x ** 2)))), 2),
                                reasons=["no return signal: nothing loud enough for 1 s"])
    seg = x[start:end]
    rms_dbfs = db(float(np.sqrt(np.mean(seg ** 2))))

    fbins, mag = _windowed_mag_spectrum(seg, sr)
    spec = mag ** 2
    total = float(spec.sum()) or 1.0
    tones: dict[str, dict] = {}
    for f in freqs:
        band = (fbins > f * 0.93) & (fbins < f * 1.07)
        wide = (fbins > f * 0.80) & (fbins < f * 1.20)
        dom = float(fbins[wide][int(np.argmax(spec[wide]))]) if wide.any() else 0.0
        tones[f"{f:g}"] = {"band_power": round(float(spec[band].sum()) / total, 4),
                           "dom_freq": round(dom, 1),
                           "offset_pct": round((dom / f - 1) * 100, 2)}
    hf = fbins > max(freqs) * 1.6
    hf_ratio = round(float(spec[hf].sum()) / total, 4)

    d2 = np.abs(np.diff(seg, 2))
    limit = max(12 * float(np.median(d2)) + 1.0 / FULL_SCALE, 0.35 * float(np.abs(seg).max()))
    hits = (d2 > limit).astype(np.int8)
    bursts = int(np.sum(np.diff(hits) == 1))
    bursts_per_s = bursts / (len(seg) / sr)

    zero = (np.abs(seg) < 3.0 / FULL_SCALE).astype(np.int8)
    dz = np.diff(zero)
    starts, ends = np.flatnonzero(dz == 1), np.flatnonzero(dz == -1)
    m = min(len(starts), len(ends))
    dropouts = int(np.sum((ends[:m] - starts[:m]) > sr // 1000))

    reasons = [f"tone {f} Hz missing (band power {t['band_power']})"
               for f, t in tones.items() if t["band_power"] < MULTITONE_MIN_BAND_POWER]
    verdict = "fail" if reasons else "pass"
    if not reasons and rms_dbfs < min_rms_dbfs:
        reasons = [f"level too low: {rms_dbfs:.1f} dBFS < {min_rms_dbfs:.0f}"]
        verdict = "fail"
    if verdict == "pass":
        warns = []
        if bursts_per_s > warn_glitch_per_s:
            warns.append(f"glitches {bursts_per_s:.1f}/s")
        warns += [f"tone {f} Hz off by {t['offset_pct']}%" for f, t in tones.items()
                  if abs(t["offset_pct"]) > MULTITONE_MAX_OFFSET_PCT]
        if dropouts:
            warns.append(f"{dropouts} dropouts")
        if warns:
            verdict, reasons = "warn", warns
    return MultitoneVerdict(**head, verdict=verdict, reasons=reasons, signal_present=True,
                            rms_dbfs=round(rms_dbfs, 2), tones=tones, hf_junk_ratio=hf_ratio,
                            glitch_bursts_per_s=round(bursts_per_s, 2), dropouts=dropouts,
                            analyzed_s=round(len(seg) / sr, 2))


def analyze_click(wav: str | Path, *, thresh: float = CLICK_THRESH,
                  max_clicks_per_s: float = WARN_GLITCH_PER_S,
                  skip_s: float = 0.0, channel: int | str = AUTO_CHANNEL) -> ClickVerdict:
    """Hard discontinuities (pops) against the local signal level."""
    np = _np()
    path = Path(wav)
    x, sr, used_ch = read_wav_ch(path, channel)
    _raw, _sr, ch, frames = _open_wav(path)
    head = _head("click", path, sr, ch, frames, used_ch)
    x = x[int(skip_s * sr):]
    dur = len(x) / float(sr or 1)
    if dur <= 0:
        raise HilError(BAD_ARGS, f"{path} is empty after skipping {skip_s} s")
    clicks, positions = click_metric(x, sr, thresh=thresh)
    per_s = clicks / dur
    reasons = ([f"{clicks} clicks ({per_s:.2f}/s > {max_clicks_per_s}/s)"]
               if per_s > max_clicks_per_s else [])
    return ClickVerdict(**head, verdict="fail" if reasons else "pass", reasons=reasons,
                        clicks=clicks, clicks_per_s=round(per_s, 2),
                        first_click_offsets_s=positions,
                        rms_dbfs=round(db(float(np.sqrt(np.mean(x ** 2)))), 2),
                        peak_dbfs=round(db(float(np.abs(x).max())), 2))


def _schedule_from(expected: Any) -> tuple[list[float], list[int | None], int]:
    """Normalize `expected` into (send times, velocities, count).

    Accepts a count, a list of send times in seconds, or a list of
    ``{"t": seconds, "vel": 0..127}`` / ``[t, vel]`` pairs.
    """
    if expected is None:
        raise HilError(BAD_ARGS, "onset/velocity analysis needs `expected`",
                       hint="a hit count, a list of send times in seconds, or "
                            "[{'t': 0.5, 'vel': 96}, …]")
    if isinstance(expected, bool):
        raise HilError(BAD_ARGS, "`expected` must be a count or a schedule, not a bool")
    if isinstance(expected, int):
        if expected < 0:
            raise HilError(BAD_ARGS, f"expected count must be >= 0, got {expected}")
        return [], [], expected
    if isinstance(expected, dict):
        expected = expected.get("hits") or expected.get("schedule")
        return _schedule_from(expected)
    if not isinstance(expected, (list, tuple)) or not expected:
        raise HilError(BAD_ARGS, f"cannot read a hit schedule from {expected!r}")
    times: list[float] = []
    vels: list[int | None] = []
    for item in expected:
        if isinstance(item, dict):
            if "t" not in item:
                raise HilError(BAD_ARGS, f"schedule entry {item!r} has no 't' (seconds)")
            times.append(float(item["t"]))
            vels.append(int(item["vel"]) if item.get("vel") is not None else None)
        elif isinstance(item, (list, tuple)) and len(item) == 2:
            times.append(float(item[0]))
            vels.append(int(item[1]))
        elif isinstance(item, (int, float)):
            times.append(float(item))
            vels.append(None)
        else:
            raise HilError(BAD_ARGS, f"cannot read a hit from {item!r}")
    order = sorted(range(len(times)), key=lambda i: times[i])
    return [times[i] for i in order], [vels[i] for i in order], len(times)


def _latency_stats(values: Sequence[float]) -> dict:
    if not values:
        return {"p50": None, "p90": None, "max": None, "n": 0}
    s = sorted(values)
    p90 = s[min(len(s) - 1, int(round(0.9 * (len(s) - 1))))]
    return {"p50": round(statistics.median(s), 2), "p90": round(p90, 2),
            "max": round(s[-1], 2), "n": len(s)}


def _floor_dbfs(x: Any, sr: int, first_hit_s: float | None) -> float:
    """The level of the quiet part of the take.

    Ported from hil_velocity as the RMS of everything before the first hit,
    which assumes that stretch is quiet. It is not: entering the USB-audio
    profile pops, and a capture that opens with a -21 dBFS transient over a
    -86 dBFS floor was measured at -32 dBFS. Every hit in that sweep then sat
    under floor + rise and the run reported 0 of 30 on a board that played all
    30.

    So the floor is a low percentile of the windowed level rather than a mean:
    a transient, or a hit inside the window, moves the top of the distribution
    and leaves the bottom where it is. Measured over the pre-roll when there is
    enough of it, over the whole take otherwise — a percentile is safe there
    because a take is mostly silence between hits.
    """
    np = _np()
    end = int((first_hit_s if first_hit_s is not None else 0.5) * sr)
    seg = x[:max(end - int(0.05 * sr), 0)]
    win = max(1, int(0.05 * sr))
    if len(seg) < 4 * win:
        seg = x
    if len(seg) < win:
        return -90.0
    frames = len(seg) // win
    rms = np.sqrt((seg[:frames * win].reshape(frames, win) ** 2).mean(axis=1))
    return db(float(np.percentile(rms, 10)))


def _onsets_and_match(x: Any, sr: int, schedule: Sequence[float], count: int, *,
                      rise_db: float, refractory_s: float, tol_s: float) -> dict:
    floor_db = _floor_dbfs(x, sr, schedule[0] if schedule else None)
    onsets, _env = detect_onsets(x, sr, floor_db, refractory_s=refractory_s, rise_db=rise_db)
    if schedule:
        matched, extra_idx, offset = match_to_schedule(onsets, schedule, sr, tol_s=tol_s)
        latency = [(matched[i] / sr - (schedule[i] + offset)) * 1000.0
                   for i in range(len(schedule)) if matched[i] is not None]
        return {"floor_db": floor_db, "onsets": onsets, "matched": matched,
                "missed": [i for i, m in enumerate(matched) if m is None],
                "extra": [round(onsets[i] / sr, 4) for i in extra_idx],
                "latency": latency, "offset_ms": round(offset * 1000.0, 2)}
    # No grid: the only honest statements are how many onsets there are and which
    # ones are beyond the expected count. Latency needs a send time to subtract.
    matched = [onsets[i] if i < len(onsets) else None for i in range(count)]
    return {"floor_db": floor_db, "onsets": onsets, "matched": matched,
            "missed": [i for i, m in enumerate(matched) if m is None],
            "extra": [round(o / sr, 4) for o in onsets[count:]],
            "latency": [], "offset_ms": None}


def analyze_onset(wav: str | Path, *, expected: Any = None, rise_db: float = ONSET_RISE_DB,
                  refractory_s: float = ONSET_REFRACTORY_S, tol_s: float = ONSET_TOL_S,
                  channel: int | str = AUTO_CHANNEL) -> OnsetVerdict:
    """Did every hit sound, and how consistently late was it?

    ``latency_ms`` is relative to a constant pipeline offset (`offset_ms`), so it
    measures jitter across a run, not the device's absolute trigger latency.
    """
    path = Path(wav)
    x, sr, used_ch = read_wav_ch(path, channel)
    _raw, _sr, ch, frames = _open_wav(path)
    head = _head("onset", path, sr, ch, frames, used_ch)
    schedule, _vels, count = _schedule_from(expected)
    m = _onsets_and_match(x, sr, schedule, count, rise_db=rise_db,
                          refractory_s=refractory_s, tol_s=tol_s)
    n_matched = count - len(m["missed"])
    reasons: list[str] = []
    if m["missed"]:
        reasons.append(f"{len(m['missed'])} of {count} hits never sounded "
                       f"(indices {m['missed'][:8]})")
    verdict = "fail" if reasons else "pass"
    if verdict == "pass" and m["extra"]:
        verdict = "warn"
        reasons = [f"{len(m['extra'])} onset(s) with no scheduled hit"]
    return OnsetVerdict(**head, verdict=verdict, reasons=reasons, expected=count,
                        matched=n_matched, missed=m["missed"], extra=m["extra"],
                        latency_ms=_latency_stats(m["latency"]),
                        onsets_s=[round(o / sr, 4) for o in m["onsets"]],
                        noise_floor_dbfs=round(m["floor_db"], 2), offset_ms=m["offset_ms"])


def analyze_velocity(wav: str | Path, *, expected: Any = None,
                     min_range_db: float = VELOCITY_MIN_RANGE_DB,
                     max_jitter_ms: float = VELOCITY_MAX_JITTER_MS,
                     rise_db: float = ONSET_RISE_DB, tol_s: float = ONSET_TOL_S,
                     channel: int | str = AUTO_CHANNEL) -> VelocityVerdict:
    """Does velocity actually change loudness, monotonically, without dropping hits?"""
    np = _np()
    path = Path(wav)
    x, sr, used_ch = read_wav_ch(path, channel)
    _raw, _sr, ch, frames = _open_wav(path)
    head = _head("velocity", path, sr, ch, frames, used_ch)
    schedule, vels, count = _schedule_from(expected)
    if not schedule or any(v is None for v in vels):
        raise HilError(BAD_ARGS, "velocity analysis needs a schedule with velocities",
                       hint="expected=[{'t': 0.5, 'vel': 96}, …]")
    m = _onsets_and_match(x, sr, schedule, count, rise_db=rise_db,
                          refractory_s=ONSET_REFRACTORY_S, tol_s=tol_s)

    per_hit: list[dict] = []
    for i, onset in enumerate(m["matched"]):
        if onset is None:
            per_hit.append({"i": i, "vel": vels[i], "send_s": round(schedule[i], 4),
                            "onset_s": None, "peak_dbfs": None, "rms_dbfs": None})
            continue
        seg = x[onset:onset + int(0.25 * sr)]     # peak over the 250 ms after the attack
        per_hit.append({
            "i": i, "vel": vels[i], "send_s": round(schedule[i], 4),
            "onset_s": round(onset / sr, 4),
            "peak_dbfs": round(db(float(np.abs(seg).max())) if len(seg) else -120.0, 2),
            "rms_dbfs": round(db(float(np.sqrt(np.mean(seg ** 2)))) if len(seg) else -120.0, 2)})

    by_vel: dict[int, list[float]] = {}
    for h in per_hit:
        if h["peak_dbfs"] is not None:
            by_vel.setdefault(int(h["vel"]), []).append(h["peak_dbfs"])
    ladder = sorted(by_vel)
    top = max(ladder) if ladder else 0
    curve = [{"vel": v,
              "hits": len(by_vel[v]),
              "peak_mean_dbfs": round(statistics.mean(by_vel[v]), 2),
              "peak_std_db": round(statistics.pstdev(by_vel[v]), 2) if len(by_vel[v]) > 1 else 0.0,
              "expected_rel_db": round(20 * math.log10(v / top), 2) if v and top else 0.0}
             for v in ladder]
    if curve:
        ref = curve[-1]["peak_mean_dbfs"]
        for c in curve:
            c["measured_rel_db"] = round(c["peak_mean_dbfs"] - ref, 2)
    means = [c["peak_mean_dbfs"] for c in curve]
    span = round(means[-1] - means[0], 2) if len(means) > 1 else 0.0
    # 1.5 dB of tolerance: below that it is measurement noise, not a real inversion.
    inversions = sum(1 for a, b in zip(means, means[1:], strict=False) if b < a - 1.5)
    latency = _latency_stats(m["latency"])
    jitter = ((latency["p90"] - latency["p50"]) if latency["p50"] is not None else 0.0)

    reasons: list[str] = []
    if m["missed"]:
        reasons.append(f"{len(m['missed'])} of {count} hits never sounded")
    if len(means) > 1 and span < min_range_db:
        reasons.append(f"velocity does not change loudness: span {span:.1f} dB "
                       f"< required {min_range_db:.0f} dB")
    if inversions:
        reasons.append(f"non-monotonic response: {inversions} inversion(s)")
    if jitter > max_jitter_ms:
        reasons.append(f"latency jitter {jitter:.1f} ms > {max_jitter_ms:.0f} ms")
    return VelocityVerdict(**head, verdict="fail" if reasons else "pass", reasons=reasons,
                           expected=count, matched=count - len(m["missed"]), missed=m["missed"],
                           curve=curve, per_hit=per_hit, dynamic_range_db=span,
                           monotonic_inversions=inversions, latency_ms=latency,
                           noise_floor_dbfs=round(m["floor_db"], 2))


def analyze_psd(wav: str | Path, *, freq: float = 997.0, ref: str | Path | None = None,
                nperseg: int = 1 << 15, skip_s: float = 2.0, take_s: float = 10.0,
                max_sideband_dbc: float = -60.0, channel: int | str = AUTO_CHANNEL) -> PsdVerdict:
    """Carrier, sidebands and harmonics in dBc (glitch_analyze.psd_report).

    The ±250 Hz and ±1 kHz pairs are the ones that matter here: USB frame activity
    amplitude-modulates the return path, and that AM lands as sidebands rather than
    as anything you can hear in the waveform.
    """
    np = _np()
    path = Path(wav)
    x, sr, used_ch = read_wav_ch(path, channel)
    _raw, _sr, ch, frames = _open_wav(path)
    head = _head("psd", path, sr, ch, frames, used_ch)
    seg = x[int(skip_s * sr): int((skip_s + take_s) * sr)]
    if len(seg) < sr // 4:
        seg = x
    f, p = _welch(seg, sr, nperseg)
    pdb = 10 * np.log10(p + 1e-20)

    def level_at(fq: float, bw: float = 15.0) -> float | None:
        m = (f > fq - bw) & (f < fq + bw)
        return round(float(pdb[m].max()), 1) if bool(m.any()) else None

    carrier = level_at(freq)
    band = (f > 20) & (f < sr / 2)
    dom = float(f[band][int(np.argmax(pdb[band]))]) if bool(band.any()) else None

    def dbc(fq: float) -> float | None:
        lv = level_at(fq)
        return None if lv is None or carrier is None else round(lv - carrier, 1)

    sidebands = {"-1k": dbc(freq - 1000), "-250": dbc(freq - 250),
                 "+250": dbc(freq + 250), "+1k": dbc(freq + 1000)}
    harmonics = {"h2": dbc(freq * 2), "h3": dbc(freq * 3)}
    hum = {"50": dbc(50.0), "150": dbc(150.0)}
    align_info = None
    if ref is not None:
        rx, _rsr = read_wav(ref, channel=0)
        lag, peak = align_to_ref(x, rx, sr)
        align_info = {"lag_samples": lag, "lag_ms": round(lag / sr * 1000, 2),
                      "corr_peak": round(peak, 3)}

    reasons: list[str] = []
    verdict = "pass"
    if carrier is None:
        verdict, reasons = "fail", [f"no energy near {freq:g} Hz"]
    else:
        if dom is not None and abs(dom - freq) > max(15.0, freq * 0.02):
            verdict = "fail"
            reasons.append(f"strongest bin is {dom:.0f} Hz, not {freq:g} Hz")
        worst = [(k, v) for k, v in sidebands.items() if v is not None and v > max_sideband_dbc]
        if worst and verdict != "fail":
            verdict = "warn"
            reasons += [f"sideband {k} at {v:.0f} dBc" for k, v in worst]
        if align_info is not None and align_info["corr_peak"] < 0.3:
            verdict = "fail" if verdict == "pass" else verdict
            reasons.append(f"capture does not correlate with the reference "
                           f"(peak {align_info['corr_peak']})")
    return PsdVerdict(**head, verdict=verdict, reasons=reasons, freq_hz=freq,
                      carrier_db=carrier, carrier_dbfs=round(db(float(np.abs(seg).max())), 2),
                      dom_freq_hz=round(dom, 1) if dom is not None else None,
                      sidebands=sidebands, harmonics=harmonics, hum=hum, align=align_info)


def analyze_silence(wav: str | Path, *, max_rms_dbfs: float = SILENCE_MAX_RMS_DBFS,
                    max_glitch_per_s: float = SILENCE_MAX_GLITCH_PER_S,
                    skip_s: float = 1.0) -> SilenceVerdict:
    """Noise floor and spontaneous artifacts of the capture path with nothing playing."""
    np = _np()
    path = Path(wav)
    data, sr = read_wav(path, channel=None)
    head = _head("silence", path, sr, data.shape[1], data.shape[0])
    per_channel: dict[str, dict] = {}
    worst_rms, worst_peak, worst_g, worst_gps = -120.0, -120.0, 0, 0.0
    for c in range(data.shape[1]):
        x = data[int(skip_s * sr):, c]           # the first second contains the stream ramp-up
        if len(x) == 0:
            x = data[:, c]
        dur = len(x) / float(sr or 1)
        g = detect_glitches(x)
        rms = db(float(np.sqrt(np.mean(x ** 2))))
        peak = db(float(np.abs(x).max()))
        gps = len(g) / dur if dur else 0.0
        per_channel[f"ch{c}"] = {"rms_dbfs": round(rms, 2), "peak_dbfs": round(peak, 2),
                                 "glitches": int(len(g)), "glitch_per_s": round(gps, 2)}
        worst_rms, worst_peak = max(worst_rms, rms), max(worst_peak, peak)
        worst_g, worst_gps = max(worst_g, int(len(g))), max(worst_gps, gps)
    reasons: list[str] = []
    if worst_rms > max_rms_dbfs:
        reasons.append(f"noise floor {worst_rms:.1f} dBFS > {max_rms_dbfs:.0f} dBFS")
    if worst_gps > max_glitch_per_s:
        reasons.append(f"{worst_g} artifacts in silence ({worst_gps:.2f}/s)")
    return SilenceVerdict(**head, verdict="fail" if reasons else "pass", reasons=reasons,
                          rms_dbfs=round(worst_rms, 2), peak_dbfs=round(worst_peak, 2),
                          glitches=worst_g, glitch_per_s=round(worst_gps, 2),
                          per_channel=per_channel)


def _steady_state(x: Any, sr: int, edge_s: float) -> Any:
    """Drop `edge_s` from both ends, unless that would leave nothing to look at.

    A tone's own start and stop are discontinuities; counting them reports two glitches
    on a flawless capture.
    """
    trim = int(max(edge_s, 0.0) * sr)
    if trim and len(x) > 4 * trim:
        return x[trim: len(x) - trim]
    return x


def _sine_take(x: Any, sr: int, refx: Any | None, edge_s: float) -> tuple[Any, Any, dict | None]:
    """One recording of the tone: (aligned, steady-state interior, alignment info)."""
    align_info = None
    if refx is not None:
        lag, peak = align_to_ref(x, refx, sr)
        align_info = {"lag_samples": lag, "lag_ms": round(lag / float(sr) * 1000, 2),
                      "corr_peak": round(peak, 3)}
        x = x[lag: lag + len(refx)]
    return x, _steady_state(x, sr, edge_s), align_info


def analyze_sine(wav: str | Path, *, freq: float = 997.0, ref: str | Path | None = None,
                 compare: Sequence[str | Path] | None = None,
                 min_rms_dbfs: float = SINE_MIN_RMS_DBFS,
                 max_offset_pct: float = MULTITONE_MAX_OFFSET_PCT,
                 max_slips: int = SINE_MAX_SLIPS,
                 max_drift_samples: float = SINE_MAX_DRIFT_SAMPLES,
                 warn_glitch_per_s: float = WARN_GLITCH_PER_S,
                 min_band_ratio: float = SINE_MIN_BAND_RATIO,
                 edge_s: float = 1.0, channel: int | str = AUTO_CHANNEL) -> SineVerdict:
    """Did a steady tone survive the trip intact — same samples, same clock?

    `psd` asks what the returned spectrum looks like; this asks whether the stream that
    carried it lost samples. They fail on different faults, which is why both exist: a
    path that drops one sample every few seconds is spectrally near-perfect and audibly
    fine, and is still a broken stream.

    With `ref` the capture is aligned to the reference first, but only the correlation
    peak reaches the verdict — a single tone is periodic, so every period-aligned lag
    correlates equally well and the reported lag is not a latency.

    `compare` names other takes of the same signal; their glitch positions are checked
    against this one's (see `glitch_determinism`).
    """
    np = _np()
    path = Path(wav)
    x, sr, used_ch = read_wav_ch(path, channel)
    _raw, _sr, ch, frames = _open_wav(path)
    head = _head("sine", path, sr, ch, frames, used_ch)
    refx = read_wav(ref, channel=0)[0] if ref is not None else None
    seg, act, align_info = _sine_take(x, sr, refx, edge_s)
    dur = len(act) / float(sr or 1)
    if len(seg) < max(64, sr // 4):
        return SineVerdict(**head, verdict="fail", freq_hz=freq, align=align_info,
                           reasons=[f"recording is only {len(seg) / float(sr or 1):.2f} s"])
    rms = db(float(np.sqrt(np.mean(act ** 2))))
    peak_dbfs = round(db(float(np.abs(act).max())), 2)
    if rms < min_rms_dbfs:
        return SineVerdict(**head, verdict="fail", freq_hz=freq, align=align_info,
                           rms_dbfs=round(rms, 2), peak_dbfs=peak_dbfs,
                           reasons=[f"no return signal: {rms:.1f} dBFS "
                                    f"< {min_rms_dbfs:.0f} dBFS"])

    resid, est_freq, band_ratio = phase_residual(seg, sr, freq, edge_s=edge_s)
    if band_ratio < min_band_ratio:
        return SineVerdict(**head, verdict="fail", freq_hz=freq, align=align_info,
                           rms_dbfs=round(rms, 2), peak_dbfs=peak_dbfs,
                           reasons=[f"no tone near {freq:g} Hz "
                                    f"({band_ratio * 100:.1f}% of the energy is in band)"])
    slips, drift, p2p = timing_slips(resid, sr, freq)
    glitches = detect_glitches(act)
    per_s = len(glitches) / dur if dur else 0.0
    offset_pct = (est_freq / freq - 1.0) * 100.0

    determinism = None
    if compare:
        runs = [glitches]
        for other in compare:
            ox, osr = read_wav(other, channel=channel)
            runs.append(detect_glitches(_sine_take(ox, osr, refx, edge_s)[1]))
        determinism = {"runs": len(runs), "tol_samples": GLITCH_TOL_SAMPLES,
                       "replicated": glitch_determinism(runs)}

    reasons: list[str] = []
    if abs(offset_pct) > max_offset_pct:
        reasons.append(f"tone came back at {est_freq:.1f} Hz, {offset_pct:+.2f}% off")
    if slips > max_slips:
        reasons.append(f"timing slips: {slips} (phase steps of half a sample or more)")
    if align_info is not None and align_info["corr_peak"] < 0.3:
        reasons.append(f"capture does not correlate with the reference "
                       f"(peak {align_info['corr_peak']})")
    verdict = "fail" if reasons else "pass"
    if verdict == "pass":
        warns = []
        if per_s > warn_glitch_per_s:
            warns.append(f"glitches {per_s:.2f}/s")
        if abs(drift) > max_drift_samples:
            warns.append(f"phase drifts {drift:+.1f} samples across the take")
        if warns:
            verdict, reasons = "warn", warns
    return SineVerdict(**head, verdict=verdict, reasons=reasons, freq_hz=freq,
                       est_freq_hz=round(est_freq, 3), freq_offset_pct=round(offset_pct, 4),
                       rms_dbfs=round(rms, 2), peak_dbfs=peak_dbfs,
                       glitches=int(len(glitches)), glitch_per_s=round(per_s, 2),
                       timing_slips=slips, net_drift_samples=round(drift, 2),
                       phase_resid_p2p_rad=round(p2p, 3),
                       periodicity=glitch_periodicity(glitches, len(act), sr),
                       align=align_info, determinism=determinism)


_DISPATCH = {"multitone": analyze_multitone, "click": analyze_click, "onset": analyze_onset,
             "velocity": analyze_velocity, "psd": analyze_psd, "silence": analyze_silence,
             "sine": analyze_sine}


def analyze(kind: str, wav: str | Path, *, expected: Any = None, **kw: Any) -> dict:
    """One entry point for every kind: ``{kind, wav, verdict, passed, reasons, …}``."""
    fn = _DISPATCH.get(kind)
    if fn is None:
        raise HilError(BAD_ARGS, f"unknown analysis kind {kind!r}",
                       hint="one of: " + ", ".join(KINDS))
    path = Path(wav)
    if not path.is_file():
        raise HilError(BAD_ARGS, f"no such WAV: {path}",
                       hint="capture writes its file under hil_logs/")
    if kind in ("onset", "velocity"):
        kw["expected"] = expected
    elif expected is not None:
        # The other kinds take their expectation as named knobs (freqs, freq, thresholds).
        if not isinstance(expected, dict):
            raise HilError(BAD_ARGS, f"`expected` for kind {kind!r} must be an object of "
                                     f"analysis options, got {type(expected).__name__}")
        kw.update(expected)
    try:
        return fn(path, **kw).to_dict()
    except TypeError as exc:
        raise HilError(BAD_ARGS, f"bad options for kind {kind!r}: {exc}") from exc
