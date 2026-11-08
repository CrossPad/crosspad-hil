"""Hearing a bounced beat: every hit where the pattern puts it, at its pad's pitch.

The fixture gives every pad a pitch of its own (``beat_fixture``), so one WAV
answers three questions at once: did each expected hit sound (an onset near
its grid time, after one constant offset for the whole file), was it the
right pad (the spectral peak just after the onset), and did a pad edit take
(an octave up doubles the peak; a reversed slice has no attack and swells).

Needs numpy (crosspad-hil[audio]); the board scenario and the Mucha performer
both call ``analyze_bounce``.
"""

from __future__ import annotations

import wave
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

PITCH_WIN_S = (0.008, 0.098)  # after the onset: past the click, before a 1/16 later
ONSET_TOL_S = 0.015
LAG_RANGE_S = (-0.03, 0.25)  # where the whole file may sit against the grid
PITCH_TOL = 0.03  # 3 %: half a semitone is 2.9 %
REV_EARLY_S = (0.0, 0.08)
REV_LATE_S = (0.18, 0.27)


@dataclass(frozen=True)
class ExpectedHit:
    t: float  # seconds from the bounce's first frame, before the offset
    pad: int
    vel: int
    freq: float  # where its spectral peak must be
    reversed: bool = False
    free_s: float = 1.0  # time until the next hit (a choke cuts a swell short)


def read_mono(path: str | Path) -> tuple[Any, int]:
    import numpy as np

    with wave.open(str(path), "rb") as w:
        sr, ch, width, n = w.getframerate(), w.getnchannels(), w.getsampwidth(), w.getnframes()
        raw = w.readframes(n)
    if width != 2:
        raise ValueError(f"{path}: {8 * width}-bit, want 16")
    x = np.frombuffer(raw, dtype="<i2").astype(np.float64) / 32768.0
    if ch > 1:
        x = x.reshape(-1, ch).mean(axis=1)
    return x, sr


def _envelope(x: Any, sr: int, win_s: float = 0.002) -> Any:
    import numpy as np

    n = max(1, int(win_s * sr))
    return np.sqrt(np.convolve(x * x, np.ones(n) / n, mode="same"))


def _onset_strength(x: Any, sr: int) -> Any:
    """Rise of the log envelope over 5 ms, positive part: a hit's front, not its level."""
    import numpy as np

    env = 20.0 * np.log10(_envelope(x, sr) + 1e-5)
    d = int(0.005 * sr)
    rise = np.zeros_like(env)
    rise[d:] = env[d:] - env[:-d]
    return np.clip(rise, 0.0, None)


def _peak_freq(x: Any, sr: int, t0: float, lo: float = 120.0, hi: float = 2400.0) -> float:
    import numpy as np

    a, b = int((t0 + PITCH_WIN_S[0]) * sr), int((t0 + PITCH_WIN_S[1]) * sr)
    seg = x[max(0, a) : max(0, b)]
    if len(seg) < 64:
        return 0.0
    spec = np.abs(np.fft.rfft(seg * np.hanning(len(seg)), n=1 << 15))
    f = np.fft.rfftfreq(1 << 15, 1.0 / sr)
    band = (f >= lo) & (f <= hi)
    return float(f[band][int(np.argmax(spec[band]))])


def _band_db(x: Any, sr: int, t0: float, t1: float, freq: float) -> float:
    import numpy as np

    a, b = int(t0 * sr), int(t1 * sr)
    seg = x[max(0, a) : max(0, b)]
    if len(seg) < 64:
        return -120.0
    spec = np.abs(np.fft.rfft(seg * np.hanning(len(seg)), n=1 << 15)) / len(seg)
    f = np.fft.rfftfreq(1 << 15, 1.0 / sr)
    band = (f >= freq * 0.94) & (f <= freq * 1.06)
    return float(20.0 * np.log10(spec[band].max() + 1e-9))


def expected_hits(
    hits: Sequence[Any],
    step_s: float,
    freqs: Sequence[float],
    semis: Mapping[int, int] | None = None,
    reversed_pads: Sequence[int] = (),
) -> list[ExpectedHit]:
    """`hits` (objects with step/pad/vel) on a grid of `step_s` seconds."""
    semis = semis or {}
    times = sorted(h.step * step_s for h in hits)
    out = []
    for h in sorted(hits, key=lambda h: (h.step, h.pad)):
        t = h.step * step_s
        later = [u for u in times if u > t + 1e-6]
        out.append(
            ExpectedHit(
                t=t,
                pad=h.pad,
                vel=h.vel,
                freq=freqs[h.pad] * 2.0 ** (semis.get(h.pad, 0) / 12.0),
                reversed=h.pad in reversed_pads,
                free_s=(later[0] - t) if later else 1.0,
            )
        )
    return out


def analyze_bounce(
    path: str | Path,
    expected: Sequence[ExpectedHit],
    *,
    want_s: float,
    onset_tol_s: float = ONSET_TOL_S,
) -> dict[str, Any]:
    """Score one bounced WAV against the hits it should hold. ``failures`` empty = pass."""
    import numpy as np

    x, sr = read_mono(path)
    out: dict[str, Any] = {
        "sr": sr,
        "seconds": round(len(x) / sr, 3),
        "want_s": round(want_s, 3),
        "peak": round(float(np.abs(x).max()) if len(x) else 0.0, 4),
    }
    failures: list[str] = []
    if len(x) / sr < want_s - 0.05:
        failures.append(f"bounce is {len(x) / sr:.3f} s, want >= {want_s:.3f} s")
    if out["peak"] < 0.01:
        failures.append(f"bounce is silent (peak {out['peak']})")
        out["failures"] = failures
        return out

    strength = _onset_strength(x, sr)
    fronts = [e for e in expected if not e.reversed]
    # One offset for the file: the lag where the expected fronts sit on the most rise.
    lags = np.arange(int(LAG_RANGE_S[0] * sr), int(LAG_RANGE_S[1] * sr), int(0.0005 * sr) or 1)
    tol = int(0.004 * sr)

    def score(lag: int) -> float:
        s = 0.0
        for e in fronts:
            c = int(e.t * sr) + lag
            a, b = max(0, c - tol), min(len(strength), c + tol)
            if b > a:
                s += float(strength[a:b].max())
        return s

    best = int(max(lags, key=score)) if fronts else 0
    out["offset_ms"] = round(1000.0 * best / sr, 2)

    rows = []
    normal_db: list[float] = []
    for e in expected:
        t = e.t + best / sr
        row: dict[str, Any] = {
            "t": round(e.t, 3),
            "pad": e.pad + 1,
            "vel": e.vel,
            "want_hz": round(e.freq, 1),
            "rev": e.reversed,
        }
        if e.reversed:
            early = _band_db(x, sr, t + REV_EARLY_S[0], t + REV_EARLY_S[1], e.freq)
            row["early_db"] = round(early, 1)
            if e.free_s >= REV_LATE_S[1]:
                late = _band_db(x, sr, t + REV_LATE_S[0], t + REV_LATE_S[1], e.freq)
                row["late_db"] = round(late, 1)
            rows.append(row)
            continue
        c = int(t * sr)
        w = int(onset_tol_s * sr)
        a, b = max(0, c - w), min(len(strength), c + w)
        seg = strength[a:b]
        if len(seg) == 0 or float(seg.max()) < 6.0:
            row["found"] = False
            failures.append(f"no hit at {e.t:.3f} s (pad {e.pad + 1})")
            rows.append(row)
            continue
        at = a + int(np.argmax(seg))
        row["found"] = True
        row["dt_ms"] = round(1000.0 * (at - c) / sr, 2)
        hz = _peak_freq(x, sr, at / sr)
        row["hz"] = round(hz, 1)
        if abs(hz / e.freq - 1.0) > PITCH_TOL:
            failures.append(f"hit at {e.t:.3f} s: {hz:.0f} Hz, pad {e.pad + 1} is {e.freq:.0f} Hz")
        lvl = _band_db(x, sr, at / sr + PITCH_WIN_S[0], at / sr + PITCH_WIN_S[1], e.freq)
        row["level_db"] = round(lvl, 1)
        normal_db.append(lvl)
        rows.append(row)
    out["hits"] = rows

    if normal_db:
        ref = float(np.median(normal_db))
        for row in rows:
            if not row["rev"]:
                continue
            if row["early_db"] > ref - 12.0:
                failures.append(
                    f"reversed pad {row['pad']} at {row['t']} s has an attack "
                    f"({row['early_db']} dB vs {ref:.1f} dB for a front)"
                )
            if "late_db" in row and row["late_db"] < row["early_db"] + 10.0:
                failures.append(
                    f"reversed pad {row['pad']} at {row['t']} s does not swell "
                    f"({row['early_db']} -> {row['late_db']} dB)"
                )

    # Velocity: the same pad louder at the higher velocity.
    by_pad: dict[int, dict[int, list[float]]] = {}
    for row in rows:
        if row.get("found") and "level_db" in row:
            by_pad.setdefault(row["pad"], {}).setdefault(row["vel"], []).append(row["level_db"])
    vel_checks = []
    for pad, levels in by_pad.items():
        if len(levels) < 2:
            continue
        lo_v, hi_v = min(levels), max(levels)
        lo_db = float(np.mean(levels[lo_v]))
        hi_db = float(np.mean(levels[hi_v]))
        vel_checks.append(
            {"pad": pad, "vel": [lo_v, hi_v], "db": [round(lo_db, 1), round(hi_db, 1)]}
        )
        if hi_db <= lo_db + 0.5:
            failures.append(
                f"pad {pad}: velocity {hi_v} is not louder than {lo_v} "
                f"({hi_db:.1f} vs {lo_db:.1f} dB)"
            )
    out["velocity"] = vel_checks

    # Fronts nobody asked for.
    thr = 12.0
    peaks = []
    i, n, gap = 0, len(strength), int(0.06 * sr)
    while i < n:
        if strength[i] >= thr:
            j = min(n, i + gap)
            k = i + int(np.argmax(strength[i:j]))
            peaks.append(k)
            i = k + gap
        else:
            i += 1
    claimed = [int((e.t * sr) + best) for e in expected]
    win = int(0.04 * sr)
    extra = [
        round(p / sr, 3)
        for p in peaks
        if not any(abs(p - c) <= win for c in claimed)
        and not any(e.reversed and 0 <= p - int(e.t * sr + best) <= int(0.3 * sr) for e in expected)
    ]
    out["extra_s"] = extra
    if extra:
        failures.append(f"{len(extra)} hit(s) nobody played, at {extra[:6]} s")
    out["failures"] = failures
    return out
