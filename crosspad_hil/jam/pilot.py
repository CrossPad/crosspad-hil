"""The phone's stream: a pilot tone and a timecode, and how to read them back.

The host plays two things into the board for the whole block, routed by the
board's mixer onto the recording bus:

* **the pilot** -- a 12 kHz sine at -26 dBFS. Where it dips the stream had a
  hole; where its phase jumps, samples were lost or repeated between the host
  and the take (``pilot_track``);
* **the timecode** -- band-limited noise (15-19 kHz, -34 dBFS) that repeats every
  2^18 samples. Cross-correlating the take with it says, to the sample, which
  host frame each stretch of the take came from, so a slip is counted exactly
  (``timecode_delay``).

``PilotSource`` renders both with random access (``render(n0, n)`` of any
slice equals that slice of one long render), which is what a duplex stream's
playback side wants.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from crosspad_hil.jam.spec import PilotSpec

INT16 = 32767.0


class PilotSource:
    """Pilot + timecode at ``rate``, the same on every channel."""

    def __init__(self, spec: PilotSpec, rate: int, channels: int = 2) -> None:
        self.spec = spec
        self.rate = int(rate)
        self.channels = int(channels)
        self.table = timecode_table(spec, self.rate) if spec.timecode else None
        f = float(spec.hz)
        # integer phase when the pilot is a whole number of Hz: no drift over hours
        self._fi = int(round(f)) if abs(f - round(f)) < 1e-9 else None
        self._f = f

    def samples(self, n0: int, n: int) -> np.ndarray:
        idx = np.arange(int(n0), int(n0) + int(n), dtype=np.int64)
        if self._fi is not None:
            ph = 2.0 * np.pi * ((idx * self._fi) % self.rate) / self.rate
        else:
            ph = 2.0 * np.pi * self._f * idx / self.rate
        x = self.spec.amp * np.sin(ph)
        if self.table is not None:
            x = x + self.table[idx % self.spec.period]
        return x

    def render(self, n0: int, n: int) -> bytes:
        x = np.clip(np.round(self.samples(n0, n) * INT16), -32768, 32767).astype("<i2")
        if self.channels > 1:
            x = np.repeat(x, self.channels)
        return x.tobytes()

    __call__ = render


def timecode_table(spec: PilotSpec, rate: int) -> np.ndarray:
    """One period of the timecode: unit-modulus random phases on the bins inside the
    band, everything else zero, RMS set to the spec's level."""
    p = int(spec.period)
    rng = np.random.default_rng(int(spec.seed))
    lo, hi = spec.timecode_band
    k0 = int(np.ceil(lo * p / rate))
    k1 = int(np.floor(hi * p / rate))
    spec_c = np.zeros(p // 2 + 1, dtype=np.complex128)
    k1 = min(k1, p // 2 - 1)
    if k1 > k0:
        spec_c[k0:k1 + 1] = np.exp(2j * np.pi * rng.random(k1 - k0 + 1))
    x = np.fft.irfft(spec_c, n=p)
    rms = float(np.sqrt(np.mean(x * x))) or 1.0
    return (x / rms * spec.tc_amp).astype(np.float64)


# --------------------------------------------------------------------------- reading it back

def _bandpass(x: np.ndarray, sr: int, lo: float, hi: float, taps: int = 129) -> np.ndarray:
    from scipy.signal import firwin, oaconvolve

    h = firwin(taps, [lo, hi], pass_zero=False, fs=sr)
    return oaconvolve(x, h, mode="same")


def demod(x: np.ndarray, sr: int, f: float, win_ms: float = 0.5,
          hop: int = 4) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Complex envelope of the tone at ``f``: (frame index, amplitude, phase), every
    ``hop`` frames, with a Hann window of ``win_ms`` (a steady sine of amplitude A
    reads A)."""
    from scipy.signal import oaconvolve

    n = len(x)
    idx = np.arange(n)
    lo = x * np.exp(-2j * np.pi * f * idx / sr)
    w = max(3, int(round(sr * win_ms / 1000.0)) | 1)
    win = np.hanning(w + 2)[1:-1]
    win = win / win.sum()
    y = oaconvolve(lo, win, mode="same")[::hop]
    at = idx[::hop]
    return at, 2.0 * np.abs(y), np.angle(y)


@dataclass
class PilotEvent:
    kind: str           # gap | step
    frame: int
    length: int = 0     # frames (gap)
    depth_db: float = 0.0
    jump_rad: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {"kind": self.kind, "frame": self.frame, "length": self.length,
                "depth_db": round(self.depth_db, 1), "jump_rad": round(self.jump_rad, 3)}


@dataclass
class PilotTrack:
    sr: int
    level: float                      # the pilot's steady amplitude in the take
    events: list[PilotEvent] = field(default_factory=list)
    present: bool = True
    hop: int = 4
    amp: np.ndarray | None = None
    frames: np.ndarray | None = None

    def to_dict(self) -> dict[str, Any]:
        return {"level_dbfs": round(20 * np.log10(max(self.level, 1e-9)), 1),
                "present": self.present, "events": [e.to_dict() for e in self.events]}


GAP_DB = -10.0
GAP_MIN_MS = 0.4
STEP_RAD = 0.3


def pilot_track(x: np.ndarray, sr: int, spec: PilotSpec, *, chunk_s: float = 20.0,
                gap_db: float = GAP_DB, gap_min_ms: float = GAP_MIN_MS,
                step_rad: float = STEP_RAD) -> PilotTrack:
    """Holes and phase jumps of the pilot over a (mono) take, chunk by chunk."""
    n = len(x)
    hop = 4
    amps: list[np.ndarray] = []
    phs: list[np.ndarray] = []
    ats: list[np.ndarray] = []
    chunk = int(chunk_s * sr)
    margin = int(0.01 * sr)
    for c0 in range(0, n, chunk):
        a0 = max(0, c0 - margin)
        a1 = min(n, c0 + chunk + margin)
        seg = _bandpass(np.asarray(x[a0:a1], dtype=np.float64), sr,
                        spec.hz - 900.0, spec.hz + 900.0)
        at, amp, ph = demod(seg, sr, spec.hz, win_ms=0.5, hop=hop)
        keep = (at + a0 >= c0) & (at + a0 < min(n, c0 + chunk))
        ats.append(at[keep] + a0)
        amps.append(amp[keep])
        phs.append(ph[keep])
    if not ats:
        return PilotTrack(sr=sr, level=0.0, present=False)
    at = np.concatenate(ats)
    amp = np.concatenate(amps)
    ph = np.concatenate(phs)
    level = float(np.median(amp)) if len(amp) else 0.0
    track = PilotTrack(sr=sr, level=level, hop=hop, amp=amp, frames=at)
    if level < spec.amp * 0.05:
        track.present = False
        return track
    thr = level * 10 ** (gap_db / 20.0)
    below = amp < thr
    min_pts = max(1, int(round(gap_min_ms / 1000.0 * sr / hop)))
    for s, e in _runs(below):
        if e - s >= min_pts:
            depth = 20 * np.log10(max(float(amp[s:e].min()), 1e-9) / level)
            track.events.append(PilotEvent("gap", int(at[s]), int(at[min(e, len(at) - 1)]
                                                                 - at[s]), depth))
    # phase: the pilot is phase-locked to the device's clock, so the demodulated phase
    # (after the carrier) is flat; a jump is samples lost or repeated
    ok = ~below
    ph_u = np.where(ok, ph, np.nan)
    step = max(1, int(round(0.001 * sr / hop)))       # compare 1 ms apart
    d = _wrap(ph_u[step:] - ph_u[:-step])
    good = np.isfinite(d) & ok[step:] & ok[:-step]
    big = good & (np.abs(d) > step_rad)
    for s, e in _runs(big):
        k = s + int(np.argmax(np.abs(d[s:e])))
        track.events.append(PilotEvent("step", int(at[k + step // 2]),
                                       jump_rad=float(d[k])))
    track.events.sort(key=lambda ev: ev.frame)
    return track


def _wrap(a: np.ndarray) -> np.ndarray:
    return (a + np.pi) % (2 * np.pi) - np.pi


def _runs(mask: np.ndarray) -> list[tuple[int, int]]:
    if not len(mask):
        return []
    m = np.concatenate(([False], np.asarray(mask, dtype=bool), [False]))
    d = np.diff(m.astype(np.int8))
    return list(zip(np.flatnonzero(d == 1).tolist(), np.flatnonzero(d == -1).tolist(),
                    strict=True))


# --------------------------------------------------------------------------- the timecode

@dataclass
class DelayPoint:
    frame: int          # take frame at the window's centre
    delay: int | None   # take frame - host frame (None: no lock)
    rho: float          # normalised correlation at the peak


@dataclass
class DelayTrack:
    points: list[DelayPoint] = field(default_factory=list)
    steps: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        locked = [p for p in self.points if p.delay is not None]
        return {"points": len(self.points), "locked": len(locked),
                "delays": sorted({p.delay for p in locked})[:20], "steps": self.steps}


def _acquire(xb: np.ndarray, c0: int, table: np.ndarray) -> tuple[int, float]:
    p = len(table)
    a = np.zeros(p)
    w = len(xb)
    pos = (c0 + np.arange(w)) % p
    a[pos] = xb
    corr = np.fft.irfft(np.fft.rfft(a) * np.conj(np.fft.rfft(table)), n=p)
    d = int(np.argmax(corr))
    seg = table[(c0 + np.arange(w) - d) % p]
    rho = float(corr[d] / (np.linalg.norm(xb) * np.linalg.norm(seg) + 1e-12))
    return d, rho


def _local(xb: np.ndarray, c0: int, table: np.ndarray, d_prev: int, span: int
           ) -> tuple[int, float]:
    p = len(table)
    w = len(xb)
    base = c0 - d_prev - span
    seg = table[(base + np.arange(w + 2 * span)) % p]
    corr = np.correlate(seg, xb, mode="valid")          # k: d = d_prev + span - k
    k = int(np.argmax(corr))
    d = d_prev + span - k
    ref = seg[k:k + w]
    rho = float(corr[k] / (np.linalg.norm(xb) * np.linalg.norm(ref) + 1e-12))
    return d % p, rho


RHO_LOCK = 0.3


def timecode_delay(x: np.ndarray, sr: int, spec: PilotSpec, *, win: int = 2048,
                   hop: int = 1024, span: int = 48, chunk_s: float = 20.0) -> DelayTrack:
    """Which host frame each stretch of the take came from: ``delay = take - host``.

    A change of delay between two locked windows is a slip of exactly that many
    samples; ``steps`` holds each with where in the take it happened (refined
    to ~3 ms by bisecting the windows)."""
    track = DelayTrack()
    if not spec.timecode:
        return track
    table = timecode_table(spec, sr)
    p = len(table)
    n = len(x)
    lo, hi = spec.timecode_band
    chunk = int(chunk_s * sr)
    d_prev: int | None = None
    for c0 in range(0, n, chunk):
        a0 = max(0, c0 - 256)
        a1 = min(n, c0 + chunk + 256)
        xb_all = _bandpass(np.asarray(x[a0:a1], dtype=np.float64), sr, lo, hi)
        for w0 in range(c0, min(n, c0 + chunk) - win + 1, hop):
            xb = xb_all[w0 - a0:w0 - a0 + win]
            if float(np.sqrt(np.mean(xb * xb))) < spec.tc_amp * 0.1:
                track.points.append(DelayPoint(w0 + win // 2, None, 0.0))
                d_prev = None
                continue
            if d_prev is None:
                d, rho = _acquire(xb, w0, table)
            else:
                d, rho = _local(xb, w0, table, d_prev, span)
                if rho < RHO_LOCK:
                    d, rho = _acquire(xb, w0, table)
            if rho < RHO_LOCK:
                track.points.append(DelayPoint(w0 + win // 2, None, rho))
                d_prev = None
                continue
            track.points.append(DelayPoint(w0 + win // 2, d, rho))
            d_prev = d
    # slips: consecutive locked points whose delay differs
    pts = track.points
    for a, b in zip(pts, pts[1:], strict=False):
        if a.delay is None or b.delay is None or a.delay == b.delay:
            continue
        diff = ((b.delay - a.delay + p // 2) % p) - p // 2
        at = _bisect_slip(x, sr, spec, table, a.frame, b.frame, a.delay, b.delay)
        track.steps.append({"frame": at, "between": [a.frame, b.frame], "samples": int(diff),
                            "from": a.delay, "to": b.delay})
    return track


def _bisect_slip(x: np.ndarray, sr: int, spec: PilotSpec, table: np.ndarray, f0: int, f1: int,
                 d0: int, d1: int, win: int = 256) -> int:
    """Where between two windows the delay changed: small windows scored against both."""
    p = len(table)
    lo, hi = spec.timecode_band
    a0 = max(0, f0 - 1024)
    a1 = min(len(x), f1 + 1024)
    xb = _bandpass(np.asarray(x[a0:a1], dtype=np.float64), sr, lo, hi)
    best = (f0 + f1) // 2
    score_prev = None
    for c in range(f0 - win // 2, f1 - win // 2, win // 4):
        s = c - a0
        if s < 0 or s + win > len(xb):
            continue
        seg = xb[s:s + win]
        r0 = float(np.dot(seg, table[(c + np.arange(win) - d0) % p]))
        r1 = float(np.dot(seg, table[(c + np.arange(win) - d1) % p]))
        sc = r1 - r0
        if score_prev is not None and score_prev <= 0 < sc:
            best = c + win // 2
            break
        score_prev = sc
    return int(best)


# --------------------------------------------------------------------------- exact residual

def _highpass(x: np.ndarray, sr: int, f: float, taps: int = 129) -> np.ndarray:
    from scipy.signal import firwin, oaconvolve

    h = firwin(taps, f, pass_zero=False, fs=sr)
    return oaconvolve(x, h, mode="same")


def segments_from(track: DelayTrack, n: int) -> list[tuple[int, int, int]]:
    """``(f0, f1, delay)`` stretches of the take with one constant delay."""
    out: list[tuple[int, int, int]] = []
    cur_d: int | None = None
    f0 = 0
    steps = {int(s["between"][0]): s for s in track.steps}
    last_frame = 0
    for pt in track.points:
        if pt.delay is None:
            if cur_d is not None:
                out.append((f0, last_frame, cur_d))
                cur_d = None
            continue
        if cur_d is None:
            cur_d, f0 = pt.delay, max(0, pt.frame - 1024)
        elif pt.delay != cur_d:
            cut = next((int(s["frame"]) for k, s in steps.items()
                        if k <= pt.frame and int(s["to"]) == pt.delay), pt.frame - 512)
            out.append((f0, cut, cur_d))
            cur_d, f0 = pt.delay, cut
        last_frame = pt.frame + 1024
    if cur_d is not None:
        out.append((f0, min(n, last_frame), cur_d))
    return out


def residual_gaps(x: np.ndarray, sr: int, spec: PilotSpec, track: DelayTrack, *,
                  hp_hz: float = 10_500.0, win_ms: float = 0.25, thr: float = 0.35,
                  min_ms: float = 0.2) -> dict[str, Any]:
    """Pilot holes to the sample: the take minus the stream it should carry.

    With the timecode's delay known, the host's own signal (pilot + timecode,
    above ``hp_hz`` where no pad lives) is subtracted from the take; what is
    left is noise -- except where the stream had a hole, where it is the stream
    itself, inverted. Works when the path is digital and bit-transparent (the
    SAMPLER capture); ``exact`` says whether it was (else use ``pilot_track``).
    """
    src = PilotSource(spec, sr, 1)
    out: dict[str, Any] = {"exact": False, "gain": None, "gaps": [], "resid_db": None}
    segs = segments_from(track, len(x))
    if not segs:
        return out
    gains, resid_ratio = [], []
    w = max(1, int(round(win_ms / 1000.0 * sr)))
    kern = np.ones(w) / w
    for f0, f1, d in segs:
        if f1 - f0 < sr // 10:
            continue
        xs = np.asarray(x[f0:f1], dtype=np.float64)
        ref = src.samples(f0 - d, f1 - f0)
        xh = _highpass(xs, sr, hp_hz)
        rh = _highpass(ref, sr, hp_hz)
        den = float(np.dot(rh, rh)) or 1e-12
        g = float(np.dot(xh, rh) / den)
        gains.append(g)
        r = xh - g * rh
        env = np.sqrt(np.convolve(r * r, kern, mode="same"))
        ref_env = np.sqrt(np.convolve(rh * rh, kern, mode="same")) * abs(g)
        level = float(np.median(ref_env)) or 1e-12
        resid_ratio.append(float(np.median(env)) / level)
        bad = env > thr * level
        for a, b in _runs(bad):
            if b - a < int(min_ms / 1000.0 * sr):
                continue
            seg_r, seg_ref = r[a:b], rh[a:b] * g
            c = float(np.dot(seg_r, -seg_ref) / (np.dot(seg_ref, seg_ref) + 1e-12))
            out["gaps"].append({"frame": int(f0 + a + w // 2), "length": int(b - a - w + 1),
                                "corr": round(c, 2),
                                "kind": "gap" if c > 0.6 else "noise"})
    if gains:
        out["gain"] = round(float(np.median(gains)), 4)
        rr = float(np.median(resid_ratio)) if resid_ratio else 1.0
        out["resid_db"] = round(float(20 * np.log10(max(rr, 1e-9))), 1)
        out["exact"] = rr < 0.1
    if not out["exact"]:
        out["gaps"] = []
    return out
