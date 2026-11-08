"""A bounce (or the Recorder's take) against the live capture of the same render.

The bounce writes the pads' tap (Sampler + Pitched after their faders, before
the recording bus) to the card in real time; the capture records the bus
("USB rec": the same pads times the route and output volume, plus the host's
stream). Over the bounce's span the two are the same render, so after the
stream is filtered away (everything over 8.5 kHz) and the gain is fitted,
``capture - g * bounce`` is noise -- unless one of the two lost something.
Which one is lower where the residual is not says which path lost it: the
card (``bounce``: the SD writer) or USB IN (``usb_capture``).
"""
from __future__ import annotations

import wave
from pathlib import Path
from typing import Any

import numpy as np

LP_HZ = 8_500.0


def read_wav(path: Path) -> tuple[np.ndarray, int]:
    with wave.open(str(path), "rb") as w:
        sr, ch, width = w.getframerate(), w.getnchannels(), w.getsampwidth()
        raw = w.readframes(w.getnframes())
    if width != 2:
        raise ValueError(f"{path}: {8 * width}-bit, 16 expected")
    x = np.frombuffer(raw, dtype="<i2").reshape(-1, ch).astype(np.float32) / 32768.0
    return x, sr


def _lowpass(x: np.ndarray, sr: int) -> np.ndarray:
    from scipy.signal import firwin, oaconvolve

    h = firwin(255, LP_HZ, fs=sr)
    return oaconvolve(np.asarray(x, dtype=np.float64), h, mode="same")


def _mono(x: np.ndarray) -> np.ndarray:
    return x.mean(axis=1) if x.ndim > 1 else x


def align(capture: np.ndarray, ref: np.ndarray, sr: int, lo: int, hi: int) -> tuple[int, float]:
    """Frame of the capture where ``ref`` starts (searched in [lo, hi]), and the
    normalised correlation there."""
    lo = max(0, lo)
    # the bounce may run past the end of the take: compare what both hold
    n = min(len(ref), 3 * sr, max(0, len(capture) - lo - sr // 20))
    if n < sr // 10:
        return lo, 0.0
    r = _lowpass(_mono(ref[:n]), sr)
    hi = min(len(capture) - n, hi)
    if hi <= lo:
        return lo, 0.0
    seg = _lowpass(_mono(capture[lo:hi + n]), sr)
    from scipy.signal import correlate

    c = correlate(seg, r, mode="valid", method="fft")
    k = int(np.argmax(c))
    e = float(np.linalg.norm(r) * np.linalg.norm(seg[k:k + n])) or 1e-12
    return lo + k, float(c[k] / e)


def diff(capture: np.ndarray, ref: np.ndarray, sr: int, offset: int, *,
         win_s: float = 0.010, thr_db: float = -30.0) -> dict[str, Any]:
    """``capture[offset:] - g * ref`` below 8.5 kHz: the gain, the residual's level and
    every stretch where it is not noise, with which side was the quieter."""
    n = min(len(ref), len(capture) - offset)
    if n <= sr // 10:
        return {"ok": False, "why": "the bounce does not overlap the capture"}
    c = _lowpass(_mono(capture[offset:offset + n]), sr)
    r = _lowpass(_mono(ref[:n]), sr)
    den = float(np.dot(r, r)) or 1e-12
    g = float(np.dot(c, r) / den)
    res = c - g * r
    sig_db = 10 * np.log10(max(float(np.mean(c * c)), 1e-20))
    res_db = 10 * np.log10(max(float(np.mean(res * res)), 1e-20))
    w = max(1, int(win_s * sr))
    m = n // w
    def rms(a: np.ndarray) -> np.ndarray:
        return np.sqrt(np.mean(a[:m * w].reshape(m, w) ** 2, axis=1))
    rr, rc, rb = rms(res), rms(c), rms(g * r)
    # against the clip's own level: a sparse clip's median window is silence
    level = float(np.sqrt(np.mean(c * c))) or 1e-9
    bad = rr > level * 10 ** (thr_db / 20.0)
    spots = []
    i = 0
    while i < m:
        if not bad[i]:
            i += 1
            continue
        j = i
        while j < m and bad[j]:
            j += 1
        side = "capture" if rc[i:j].mean() < rb[i:j].mean() else "bounce"
        spots.append({"frame": int(i * w), "length": int((j - i) * w), "lower": side,
                      "resid_db": round(20 * np.log10(max(float(rr[i:j].max()), 1e-12)
                                                      / level), 1)})
        i = j
    return {"ok": True, "gain": round(g, 4), "resid_db_rel": round(res_db - sig_db, 1),
            "spots": spots[:50], "n_spots": len(spots), "frames": int(n)}
