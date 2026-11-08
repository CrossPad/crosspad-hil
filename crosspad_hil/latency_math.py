"""The arithmetic of the `latency` scenario: clock maps, onsets, statistics.

Nothing here touches hardware; everything is numpy on arrays the scenario
hands in, so every claim the scenario makes about milliseconds can be tested
against a synthetic take with a known answer.

Three pieces, each with one non-obvious rule:

* **Clock maps are lower envelopes.** The duplex stream logs ``(frame, t)``
  pairs: "this frame was at the converter at monotonic time t". Every way such
  a pair can be wrong makes ``t`` *late*, never early -- the timestamp is
  taken after the call returns, the driver's position only moves when a USB
  packet completes, the thread may wait for the GIL. So the map is the line
  under the cloud, not through it: the earliest pair of each short bin, then a
  robust line through those.
* **An onset is the first frame over a threshold set by the hit itself.** The
  threshold is the largest of a multiple of the take's noise, a fraction of
  the hit's own peak and twice whatever was already in the window's run-up, so
  a quiet take, a loud hit and a hit landing on another's tail are all judged
  on their own terms. A hit is ``missing`` (nothing over the noise),
  ``masked`` (the window opens on sound: no start can be read) or ``double``
  (the sound stops and starts again inside its own window).
* **Statistics say how many they are of.** Every summary carries ``n``;
  ``p50_ci95`` is a seeded bootstrap so two runs of one take agree.
"""
from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

# Noise: a low percentile of the 1 ms frame RMS. Most of a latency take is
# silence between hits, so the 25th percentile is the floor even when a third
# of the frames carry sound.
NOISE_FRAME_MS = 1.0
NOISE_PERCENTILE = 25.0
NOISE_FLOOR_MIN = 1.0 / 32768.0      # one LSB: a digitally silent take still has a floor

ONSET_K_NOISE = 8.0      # threshold >= 8x the noise RMS (Gaussian noise: ~1e-15 per sample)
ONSET_REL_PEAK = 0.05    # ... and >= 5 % of the hit's own peak
ONSET_PRE_FACTOR = 2.0   # ... and >= 2x the loudest sample of the run-up
SNR_MIN = 20.0           # a window whose peak is not 26 dB over the noise holds no hit
MASK_REL = 0.25          # a run-up this loud relative to the hit: the window opened on sound
QUIET_MS = 5.0           # the hit's sound has ended once this long under the off level
RETRIGGER_REL = 0.3      # ...and a second sound must reach this share of the first's peak
ENV_FRAME_MS = 0.5

CLOCK_BIN_S = 0.5
CLOCK_MAX_SLOPE_DEV = 0.02   # a fitted rate 2 % off nominal is a broken log, not a clock


# --------------------------------------------------------------------------- lines


def robust_line(x: Sequence[float] | Any, y: Sequence[float] | Any) -> tuple[float, float]:
    """``(slope, intercept)`` of ``y = intercept + slope * x``, Theil–Sen style.

    All pairwise slopes for a few hundred points; for more, the slopes of
    pairs half the data apart (still a median of many independent estimates,
    in linear time). Inputs are centred first so a monotonic clock in the
    hundreds of thousands of seconds keeps its microseconds.
    """
    xa = np.asarray(x, dtype=np.float64)
    ya = np.asarray(y, dtype=np.float64)
    if xa.shape != ya.shape or xa.size < 2:
        raise ValueError("robust_line needs two or more (x, y) pairs")
    x0, y0 = float(xa[0]), float(ya[0])
    xc, yc = xa - x0, ya - y0
    if xc.size <= 300:
        i, j = np.triu_indices(xc.size, 1)
        dx = xc[j] - xc[i]
        dy = yc[j] - yc[i]
    else:
        order = np.argsort(xc, kind="stable")
        xs, ys = xc[order], yc[order]
        h = xs.size // 2
        dx = xs[h:2 * h] - xs[:h]
        dy = ys[h:2 * h] - ys[:h]
    ok = dx != 0
    if not np.any(ok):
        raise ValueError("robust_line: every x is the same")
    slope = float(np.median(dy[ok] / dx[ok]))
    intercept_c = float(np.median(yc - slope * xc))
    return slope, y0 + intercept_c - slope * x0


@dataclass
class ClockFit:
    """``t(frame) = t0 + (frame - f0) * spf``: where a frame was, on the monotonic clock."""

    f0: float
    t0: float
    spf: float                      # seconds per frame
    points: int
    bins: int
    resid_ms: dict[str, float]

    def t(self, frame: Any) -> Any:
        return self.t0 + (np.asarray(frame, dtype=np.float64) - self.f0) * self.spf

    def frame(self, t: Any) -> Any:
        return self.f0 + (np.asarray(t, dtype=np.float64) - self.t0) / self.spf

    @property
    def rate(self) -> float:
        return 1.0 / self.spf

    def to_dict(self) -> dict[str, Any]:
        return {"rate_hz": round(self.rate, 3), "points": self.points, "bins": self.bins,
                "resid_ms": self.resid_ms}


def lower_envelope_fit(frames: Sequence[float] | Any, times: Sequence[float] | Any, *,
                       nominal_rate: float, bin_s: float = CLOCK_BIN_S) -> ClockFit:
    """The line under a cloud of ``(frame, t)`` pairs whose errors are all late.

    The earliest pair of each ``bin_s`` slice of time stands for that slice,
    and a robust line goes through those. Raises ValueError when there is too
    little to fit or the fitted rate is not the stream's (a log that mixes two
    streams, or one whose frame counter jumped).
    """
    f = np.asarray(frames, dtype=np.float64)
    t = np.asarray(times, dtype=np.float64)
    ok = np.isfinite(f) & np.isfinite(t)
    f, t = f[ok], t[ok]
    if f.size < 2:
        raise ValueError("fewer than two clock points")
    spf0 = 1.0 / float(nominal_rate)
    order = np.argsort(t, kind="stable")
    f, t = f[order], t[order]
    resid = (t - t[0]) - (f - f[0]) * spf0
    bins = np.floor((t - t[0]) / bin_s).astype(np.int64)
    edges = np.flatnonzero(np.diff(bins)) + 1
    starts = np.concatenate(([0], edges))
    stops = np.concatenate((edges, [f.size]))
    pick = np.array([a + int(np.argmin(resid[a:b])) for a, b in zip(starts, stops, strict=True)])
    if pick.size >= 2:
        slope, icpt = robust_line(f[pick], t[pick])
    else:
        slope, icpt = robust_line(f, t)
    if not math.isfinite(slope) or abs(slope / spf0 - 1.0) > CLOCK_MAX_SLOPE_DEV:
        raise ValueError(f"clock fit rate {1.0 / slope if slope else float('inf'):.1f} Hz is "
                         f"not the stream's {nominal_rate:.0f} Hz")
    f0 = float(f[0])
    t0 = icpt + slope * f0
    late = (t - (t0 + (f - f0) * slope)) * 1000.0
    summary = {"min": round(float(late.min()), 3),
               "p50": round(float(np.percentile(late, 50)), 3),
               "p95": round(float(np.percentile(late, 95)), 3),
               "max": round(float(late.max()), 3)}
    return ClockFit(f0=f0, t0=float(t0), spf=float(slope), points=int(f.size),
                    bins=int(pick.size), resid_ms=summary)


# --------------------------------------------------------------------------- levels


def frame_levels(x: Any, sr: int, frame_ms: float, *, how: str = "rms") -> Any:
    """Per-frame RMS (or peak) of ``|x|``; the tail shorter than a frame is dropped."""
    n = max(1, int(round(sr * frame_ms / 1000.0)))
    a = np.asarray(x, dtype=np.float64)
    usable = (a.size // n) * n
    if usable == 0:
        return np.zeros(0)
    frames = a[:usable].reshape(-1, n)
    if how == "peak":
        return np.abs(frames).max(axis=1)
    return np.sqrt((frames ** 2).mean(axis=1))


def noise_floor(x: Any, sr: int) -> float:
    """RMS of the quiet part of a take: a low percentile of the 1 ms frame RMS."""
    levels = frame_levels(x, sr, NOISE_FRAME_MS)
    if levels.size == 0:
        return NOISE_FLOOR_MIN
    return max(float(np.percentile(levels, NOISE_PERCENTILE)), NOISE_FLOOR_MIN)


# --------------------------------------------------------------------------- onsets


@dataclass
class Onset:
    status: str                  # ok | missing | masked | double
    index: float | None          # fractional sample index of the threshold crossing
    peak: float
    threshold: float
    pre_peak: float
    events: int

    def to_dict(self) -> dict[str, Any]:
        return {"status": self.status,
                "index": None if self.index is None else round(self.index, 3),
                "peak": round(self.peak, 6), "threshold": round(self.threshold, 6),
                "pre_peak": round(self.pre_peak, 6), "events": self.events}


def find_onset(x: Any, start: float, stop: float, *, sr: int, noise: float,
               pre_ms: float = 20.0) -> Onset:
    """The first crossing in ``x[start:stop]`` and whether it is a clean single hit.

    ``pre_ms`` before ``start`` is the run-up: what was already sounding when
    the window opened. See the module docstring for the four outcomes.
    """
    a = np.asarray(x, dtype=np.float64)
    lo = max(0, int(math.floor(start)))
    hi = min(a.size, int(math.ceil(stop)))
    pre_lo = max(0, lo - int(round(sr * pre_ms / 1000.0)))
    pre = np.abs(a[pre_lo:lo])
    pre_peak = float(pre.max()) if pre.size else 0.0
    if hi - lo < 2:
        return Onset("missing", None, 0.0, 0.0, pre_peak, 0)
    seg = np.abs(a[lo:hi])
    peak = float(seg.max())
    if peak < SNR_MIN * noise:
        return Onset("missing", None, peak, 0.0, pre_peak, 0)
    thr = max(ONSET_K_NOISE * noise, ONSET_REL_PEAK * peak, ONSET_PRE_FACTOR * pre_peak)
    if pre_peak > MASK_REL * peak:
        return Onset("masked", None, peak, thr, pre_peak, 0)
    over = np.flatnonzero(seg > thr)
    if over.size == 0:          # cannot happen with thr <= peak, but a NaN would get here
        return Onset("missing", None, peak, thr, pre_peak, 0)
    first = int(over[0])
    if first == 0:
        return Onset("masked", None, peak, thr, pre_peak, 0)
    below, above = float(seg[first - 1]), float(seg[first])
    frac = (thr - below) / (above - below) if above > below else 1.0
    index = lo + first - 1 + min(max(frac, 0.0), 1.0)
    events = _count_events(seg[first:], sr=sr, noise=noise, peak=peak)
    status = "double" if events > 1 else "ok"
    return Onset(status, float(index), peak, thr, pre_peak, events)


def _count_events(seg: Any, *, sr: int, noise: float, peak: float) -> int:
    """Separate sounds in ``seg`` (which starts at the first one's onset).

    A sound has ended once its envelope stays under the off level for
    ``QUIET_MS``; another starts when the envelope then climbs past
    ``RETRIGGER_REL`` of the first one's peak. Ringing, the codec's tail and a
    slow decay stay one event.
    """
    env = frame_levels(seg, sr, ENV_FRAME_MS, how="peak")
    if env.size == 0:
        return 1
    off = max(ONSET_K_NOISE * noise, ONSET_REL_PEAK * peak)
    on = max(off, RETRIGGER_REL * peak)
    quiet_frames = max(1, int(round(QUIET_MS / ENV_FRAME_MS)))
    events, quiet, sounding = 1, 0, True
    for level in env:
        if sounding:
            quiet = quiet + 1 if level < off else 0
            if quiet >= quiet_frames:
                sounding = False
        elif level > on:
            events += 1
            sounding, quiet = True, 0
    return events


# --------------------------------------------------------------------------- statistics


def describe(values: Sequence[float] | Any, *, ci: bool = True, seed: int = 0,
             resamples: int = 1000) -> dict[str, Any]:
    """``n`` and the spread of ``values`` in their own unit, rounded to 1 µs for ms input."""
    v = np.asarray([float(x) for x in values], dtype=np.float64)
    if v.size == 0:
        return {"n": 0}
    out: dict[str, Any] = {
        "n": int(v.size),
        "min": round(float(v.min()), 3),
        "p5": round(float(np.percentile(v, 5)), 3),
        "p50": round(float(np.percentile(v, 50)), 3),
        "p95": round(float(np.percentile(v, 95)), 3),
        "max": round(float(v.max()), 3),
        "mean": round(float(v.mean()), 3),
        "std": round(float(v.std(ddof=1)) if v.size > 1 else 0.0, 3),
    }
    if ci and v.size >= 5:
        rng = np.random.default_rng(seed)
        meds = np.median(rng.choice(v, size=(resamples, v.size), replace=True), axis=1)
        out["p50_ci95"] = [round(float(np.percentile(meds, 2.5)), 3),
                           round(float(np.percentile(meds, 97.5)), 3)]
    return out


def split_at(times: Sequence[float], values: Sequence[float], t_event: float,
             guard_s: float) -> tuple[list[float], list[float], int]:
    """``(before, after, ambiguous)``: values whose time is clear of ``t_event``.

    ``t_event`` is when the event was *seen*; it happened up to ``guard_s``
    earlier. A value inside that span could be either side and counts in
    neither -- it is only counted.
    """
    before: list[float] = []
    after: list[float] = []
    ambiguous = 0
    for t, v in zip(times, values, strict=True):
        if t < t_event - guard_s:
            before.append(v)
        elif t > t_event:
            after.append(v)
        else:
            ambiguous += 1
    return before, after, ambiguous
