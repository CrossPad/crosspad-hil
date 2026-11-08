"""How loud each band of the plan is, frame by frame, in each channel of a take.

``BandBank`` runs one short-time Fourier transform per window length over the
take (in chunks, float32) and keeps only the bins that matter: for every band
the largest of the bins within +-15 Hz of it (a tone between two bins is not
read low). A steady sine of amplitude A reads A, so a band's level compares
directly with the model's gains.

Two window lengths, because one cannot do both jobs:

* ``fine`` (12-20 ms, by band group, hop 2 ms) for onsets, holes and ends --
  short enough that a 10 ms hole reads 20 dB down, long enough that the next
  pad of the same kit (>= 136 Hz away) stays out of the main lobe;
* ``ghost`` (40 ms, hop 5 ms) for "nothing should be here": the 120 Hz
  neighbour of a sounding A pad (its B twin) leaks -31 dB through a 20 ms
  Hann, over the -35 dB a ghost is judged at; at 40 ms it is under -45 dB.

``local_envelope`` is the third view: a heterodyne at one frequency around one
moment, 0.25 ms resolution, with phase -- what an onset is timed from.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

PROBE_HZ = 15.0
NFFT = 1024


@dataclass
class BandLevels:
    hop: int
    win: int
    t0: int                                   # frame of the first column's centre
    freqs: list[float]
    left: np.ndarray                          # (bands, columns) float32
    right: np.ndarray
    index: dict[float, int] = field(default_factory=dict)

    def frame(self, col: int | np.ndarray) -> Any:
        return self.t0 + np.asarray(col) * self.hop

    def col(self, frame: float) -> int:
        return int(round((frame - self.t0) / self.hop))

    def get(self, f: float) -> tuple[np.ndarray, np.ndarray]:
        i = self.index[round(f, 3)]
        return self.left[i], self.right[i]

    def both(self, f: float) -> np.ndarray:
        lft, rgt = self.get(f)
        return np.maximum(lft, rgt)

    def has(self, f: float) -> bool:
        return round(f, 3) in self.index


def _bins_for(freqs: list[float], sr: int, nfft: int) -> list[np.ndarray]:
    df = sr / nfft
    out = []
    for f in freqs:
        lo = max(0, int(np.floor((f - PROBE_HZ) / df)))
        hi = min(nfft // 2, int(np.ceil((f + PROBE_HZ) / df)))
        out.append(np.arange(lo, hi + 1))
    return out


def band_levels(x: np.ndarray, sr: int, freqs: list[float], *, win_s: float, hop_s: float,
                chunk_cols: int = 4096) -> BandLevels:
    """Levels of ``freqs`` over a (frames, channels) take."""
    x = np.asarray(x)
    if x.ndim == 1:
        x = x[:, None]
    n = x.shape[0]
    win = max(16, int(round(win_s * sr)))
    hop = max(1, int(round(hop_s * sr)))
    nfft = max(NFFT, 1 << int(np.ceil(np.log2(win))))
    w = np.hanning(win + 2)[1:-1].astype(np.float32)
    scale = 2.0 / float(w.sum())
    uniq = sorted({round(float(f), 3) for f in freqs})
    bins = _bins_for(uniq, sr, nfft)
    allb = np.unique(np.concatenate(bins)) if bins else np.zeros(0, dtype=int)
    pos = {int(b): i for i, b in enumerate(allb)}
    sel = [np.array([pos[int(b)] for b in bb]) for bb in bins]
    ncol = max(0, (n - win) // hop + 1)
    L = np.zeros((len(uniq), ncol), dtype=np.float32)
    R = np.zeros((len(uniq), ncol), dtype=np.float32)
    chans = [0, 1] if x.shape[1] > 1 else [0, 0]
    for c0 in range(0, ncol, chunk_cols):
        c1 = min(ncol, c0 + chunk_cols)
        starts = np.arange(c0, c1) * hop
        idx = starts[:, None] + np.arange(win)[None, :]
        for out, ch in ((L, chans[0]), (R, chans[1])):
            frames = x[idx, ch].astype(np.float32) * w
            spec = np.abs(np.fft.rfft(frames, n=nfft, axis=1)[:, allb]) * scale
            for i, s in enumerate(sel):
                out[i, c0:c1] = spec[:, s].max(axis=1)
            if x.shape[1] == 1:
                R[:, c0:c1] = L[:, c0:c1]
                break
    return BandLevels(hop=hop, win=win, t0=win // 2, freqs=uniq, left=L, right=R,
                      index={f: i for i, f in enumerate(uniq)})


# Fine windows per band group: the next pad of the same kit must stay out of the
# main lobe (+-2/win): A/B 240 Hz apart, FX 200, P >= 136.
def fine_window_s(freq: float, group: str = "") -> float:
    if group == "P":
        return 0.020
    if freq >= 4_000.0:
        return 0.012
    return 0.016


@dataclass
class BandBank:
    sr: int
    fine: dict[str, BandLevels] = field(default_factory=dict)     # by group
    ghost: BandLevels | None = None
    group_of: dict[float, str] = field(default_factory=dict)

    @staticmethod
    def build(x: np.ndarray, sr: int, groups: dict[str, list[float]],
              ghost_freqs: list[float] | None = None, hop_s: float = 0.002,
              ghost_hop_s: float = 0.005) -> BandBank:
        bank = BandBank(sr=sr)
        for g, freqs in groups.items():
            if not freqs:
                continue
            win = 0.020 if g == "P" else (0.012 if min(freqs) >= 4_000.0 else 0.016)
            bank.fine[g] = band_levels(x, sr, freqs, win_s=win, hop_s=hop_s)
            for f in freqs:
                bank.group_of[round(float(f), 3)] = g
        if ghost_freqs:
            bank.ghost = band_levels(x, sr, ghost_freqs, win_s=0.040, hop_s=ghost_hop_s)
        return bank

    def levels(self, f: float) -> BandLevels | None:
        g = self.group_of.get(round(float(f), 3))
        return self.fine.get(g) if g is not None else None


# --------------------------------------------------------------------------- one moment

def local_envelope(x: np.ndarray, sr: int, f: float, center: float, before_s: float,
                   after_s: float, win_s: float = 0.008, hop: int = 11
                   ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(frames, amplitude, phase) of the tone ``f`` around ``center`` (mono ``x``).

    Phase is the demodulated phase: a sample that starts (sin, phase 0) at frame
    t0 reads ``-2 pi f t0 / sr - pi/2`` for as long as it sounds -- a voice's start
    is in its phase, to a fraction of a period."""
    n = len(x)
    pad = int(win_s * sr)
    a = max(0, int(center - before_s * sr) - pad)
    b = min(n, int(center + after_s * sr) + pad)
    if b - a < 8:
        return np.zeros(0), np.zeros(0), np.zeros(0)
    seg = np.asarray(x[a:b], dtype=np.float64)
    idx = np.arange(a, b)
    lo = seg * np.exp(-2j * np.pi * f * idx / sr)
    w = max(5, int(round(win_s * sr)) | 1)
    win = np.hanning(w + 2)[1:-1]
    win /= win.sum()
    y = np.convolve(lo, win, mode="same")
    keep = slice(pad, len(y) - pad if len(y) - pad > pad else len(y), hop)
    return idx[keep].astype(np.float64), 2.0 * np.abs(y[keep]), np.angle(y[keep])


def mono(x: np.ndarray, gain_l: float = 1.0, gain_r: float = 1.0) -> np.ndarray:
    """The channel a voice is louder in (both summed when it is centred)."""
    x = np.asarray(x)
    if x.ndim == 1:
        return x
    if gain_l > 2 * gain_r:
        return x[:, 0]
    if gain_r > 2 * gain_l:
        return x[:, 1]
    return 0.5 * (x[:, 0].astype(np.float64) + x[:, 1])
