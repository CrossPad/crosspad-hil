"""The model's expectation as audio, and the faults a board could add to it.

``render_expected`` turns an ``Expectation`` into the take a perfect board
would have produced: every voice as its tone with the fixture's fade, cut or
released where the model says, the pilot and timecode delayed by the path.
The analysis writes it next to the take (``expected_blockN.wav``) to listen
to side by side; the synthetic tests render it, break it on purpose with a
``Fault`` and check the report finds exactly that, at the right moment.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from crosspad_hil.jam import spec as S
from crosspad_hil.jam.model import INF, Expectation, Voice
from crosspad_hil.jam.pilot import PilotSource

LP_RENDER_GAIN = 10 ** (-19.0 / 20.0)


def _envelope(v: Voice, sr: int, n0: int, n1: int, *, steal_fade_s: float = 0.0) -> np.ndarray:
    """Gain of voice ``v`` for frames [n0, n1)."""
    t = np.arange(n0, n1, dtype=np.float64)
    rel = (t - v.f_on) / sr
    env = np.ones_like(rel)
    fade_on = max(v.fade_on_s, 0.0)
    if fade_on > 0:
        env *= np.clip(rel / fade_on, 0.0, 1.0)
    else:
        env *= (rel >= 0).astype(np.float64)
    if v.shape == "rise":
        dur = (v.f_nat - v.f_on) / sr
        env *= np.exp(-np.clip(dur - rel, 0.0, None) / 0.3)
    if v.f_nat != INF and v.shape != "loop":
        nat = (v.f_nat - t) / sr
        env *= np.clip(nat / S.FADE_S, 0.0, 1.0)
    if v.f_rel != INF and v.release_s:
        rr = (t - v.f_rel) / sr
        env *= np.clip(1.0 - rr / v.release_s, 0.0, 1.0)
    end = v.f_end
    if end != INF and v.end_kind not in ("natural", "released", "retrigger_release"):
        fade = steal_fade_s if v.end_kind == "stolen" else 0.0
        if v.engine == "pitched" and v.end_kind == "stolen":
            fade = 0.005
        if fade > 0:
            env *= np.clip((end + fade * sr - t) / (fade * sr), 0.0, 1.0)
        else:
            env *= (t < end).astype(np.float64)
    elif end != INF:
        env *= (t < end + 1).astype(np.float64)
    env *= (rel >= 0).astype(np.float64)
    return env


@dataclass
class Fault:
    """One thing a board could get wrong. ``kind``:

    drop_hit(vid) / delay_hit(vid, ms) / cut_voice(vid, at_ms) / wrong_kit(vid, freq)
    zero_block(frame, ms, scope="all"|"pads") / click(frame, amp) / ghost(frame, freq, ms)
    pilot_gap(frame, ms) / capture_drop(frame, samples) / clip(frame)
    """

    kind: str
    vid: int | None = None
    frame: int | None = None
    ms: float = 0.0
    amp: float = 0.5
    freq: float = 0.0
    samples: int = 0
    scope: str = "all"
    at_ms: float = 0.0
    meta: dict[str, Any] = field(default_factory=dict)


def render_expected(exp: Expectation, *, pilot: S.PilotSpec | None = None, delay: int = 0,
                    faults: list[Fault] | None = None, steal_fade_s: float = 0.0,
                    noise: float = 0.0, seed: int = 1, n: int | None = None) -> np.ndarray:
    """(frames, 2) float64 in [-1, 1]: the voices, the stream delayed by ``delay``."""
    faults = list(faults or [])
    sr = exp.sr
    n = int(n if n is not None else exp.n)
    pads = np.zeros((n, 2))
    by_vid = {f.vid: f for f in faults if f.vid is not None}
    for v in exp.voices:
        f = by_vid.get(v.vid)
        if f is not None and f.kind == "drop_hit":
            continue
        vv = v
        if f is not None and f.kind in ("delay_hit", "cut_voice", "wrong_kit"):
            vv = Voice(**{k: getattr(v, k) for k in v.__dataclass_fields__})
            if f.kind == "delay_hit":
                shift = f.ms / 1000.0 * sr
                vv.f_on += shift
                vv.f_nat += shift
                if vv.end_kind in ("natural", "released"):
                    vv.f_end += shift
                    vv.f_rel += shift
            elif f.kind == "cut_voice":
                vv.f_end = min(vv.f_end, v.f_on + f.at_ms / 1000.0 * sr)
                vv.end_kind = "fault_cut"
        if vv.f_on >= n:
            continue
        n0 = max(0, int(np.floor(vv.f_on)))
        last = vv.f_end if vv.f_end != INF else n
        if vv.f_rel != INF and vv.release_s:
            last = min(last, vv.f_rel + vv.release_s * sr + 1)
        n1 = min(n, int(np.ceil(last)) + 1)
        if n1 <= n0:
            continue
        env = _envelope(vv, sr, n0, n1, steal_fade_s=steal_fade_s)
        freq = vv.freqs[0] if vv.freqs else 0.0
        if f is not None and f.kind == "wrong_kit":
            freq = f.freq
        t = np.arange(n0, n1, dtype=np.float64)
        wave = np.sin(2 * np.pi * freq * (t - vv.f_on) / sr) * env
        if vv.feature == "lp":
            wave *= LP_RENDER_GAIN          # a 12 dB/oct low-pass 1.6 octaves under the tone
        pads[n0:n1, 0] += vv.gain_l * wave
        pads[n0:n1, 1] += vv.gain_r * wave
    stream = np.zeros((n, 2))
    if pilot is not None:
        src = PilotSource(pilot, sr, 1)
        start = int(delay)
        if start < n:
            s = src.samples(0, n - start)
            for w in exp.pilot_off:
                if w.kind in ("stream_pause",):
                    a, b = int(w.f0 + 0.05 * sr) - start, int(w.f1 - 0.3 * sr) - start
                    s[max(0, a):max(0, b)] = 0.0
            stream[start:, 0] = s
            stream[start:, 1] = s
    for f in faults:
        if f.kind == "pilot_gap" and f.frame is not None:
            a = int(f.frame)
            stream[a:a + int(f.ms / 1000.0 * sr)] = 0.0
        elif f.kind == "zero_block" and f.frame is not None and f.scope == "pads":
            a = int(f.frame)
            pads[a:a + int(f.ms / 1000.0 * sr)] = 0.0
        elif f.kind == "ghost" and f.frame is not None:
            a = int(f.frame)
            b = min(n, a + int(f.ms / 1000.0 * sr))
            t = np.arange(a, b)
            g = np.sin(2 * np.pi * f.freq * (t - a) / sr) * f.amp
            fd = max(1, int(S.FADE_S * sr))
            ramp = np.minimum(1.0, np.minimum((t - a) / fd, (b - 1 - t) / fd))
            pads[a:b, 0] += g * ramp
            pads[a:b, 1] += g * ramp
    x = pads + stream
    for f in faults:
        if f.kind == "zero_block" and f.frame is not None and f.scope == "all":
            a = int(f.frame)
            x[a:a + int(f.ms / 1000.0 * sr)] = 0.0
        elif f.kind == "click" and f.frame is not None:
            x[int(f.frame)] += f.amp
        elif f.kind == "clip" and f.frame is not None:
            a = int(f.frame)
            x[a:a + 8] = 1.0
    if noise:
        x += np.random.default_rng(seed).normal(0.0, noise, x.shape)
    for f in faults:
        if f.kind == "capture_drop" and f.frame is not None:
            a = int(f.frame)
            k = int(f.samples)
            x = np.concatenate([x[:a], x[a + k:], np.zeros((k, 2))])
    return np.clip(x, -1.0, 1.0)


def to_int16(x: np.ndarray) -> np.ndarray:
    return np.clip(np.round(x * 32767.0), -32768, 32767).astype("<i2")


def write_wav(path: Any, x: np.ndarray, sr: int) -> None:
    import wave

    data = to_int16(x)
    with wave.open(str(path), "wb") as w:
        w.setnchannels(data.shape[1] if data.ndim > 1 else 1)
        w.setsampwidth(2)
        w.setframerate(int(sr))
        w.writeframes(data.tobytes())
