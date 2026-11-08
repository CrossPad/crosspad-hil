"""The "make a beat" fixture: one loop of eight known hits, and what the board should make of it.

The workflow never records a microphone: it starts from this loop, written
here, byte for byte the same on every run. Eight short mallet hits, one per
eighth note of a bar at the workflow's tempo, each at a pitch of its own, so
that every later stage can be checked against the file instead of by ear:

* the Slicer must cut it at exactly these eight onsets (``onsets_frames``);
* the kit it saves has the eight hits on pads 1..8, so a pad is known by its
  pitch (``freqs``) wherever it sounds -- in a bounce, in a song;
* a pad pitched in the pad editor sounds at ``freq * 2**(semis/12)``, a
  reversed one swells instead of decaying.

Stdlib only (``wave``, ``math``, ``struct``), like ``crosspad_hil.fixtures``:
the Mucha performer and a bench box without numpy both build it.
"""

from __future__ import annotations

import math
import struct
import wave
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

SAMPLE_RATE = 44100  # the engine's rate: kits play 1:1
PEAK = 0.5  # -6 dBFS, room for the attack's click
ATTACK_S = 0.002  # linear rise of the tone
DECAY_TAU_S = 0.045  # exponential decay: -58 dB by the next eighth at 100 BPM
CLICK_S = 0.003  # a short noise burst on the onset, for a drum-like front
CLICK_GAIN = 0.25
HARMONIC2 = 0.3  # a second partial, so it is a mallet, not a test tone
LCG_SEED = 0x1EDE  # the click's noise: a fixed generator, never random()


@dataclass(frozen=True)
class Hit:
    name: str  # what the pad is called in the video and the report
    freq: float  # fundamental, Hz


# A minor pentatonic upwards from A3: neighbours are at least two semitones
# apart, and an octave up (the pad editor's +12) lands on none of them.
DEFAULT_HITS: tuple[Hit, ...] = (
    Hit("A3", 220.00),
    Hit("C4", 261.63),
    Hit("E4", 329.63),
    Hit("G4", 392.00),
    Hit("A4", 440.00),
    Hit("C5", 523.25),
    Hit("E5", 659.26),
    Hit("G5", 783.99),
)


@dataclass(frozen=True)
class Loop:
    bpm: float
    hits: tuple[Hit, ...]
    sample_rate: int = SAMPLE_RATE

    @property
    def spacing_s(self) -> float:
        """One eighth note: the distance between two hits."""
        return 60.0 / self.bpm / 2.0

    @property
    def frames(self) -> int:
        return int(round(self.spacing_s * self.sample_rate)) * len(self.hits)

    @property
    def onsets_frames(self) -> list[int]:
        step = int(round(self.spacing_s * self.sample_rate))
        return [i * step for i in range(len(self.hits))]

    @property
    def freqs(self) -> list[float]:
        return [h.freq for h in self.hits]


def _lcg(seed: int):
    state = seed & 0xFFFFFFFF
    while True:
        state = (1664525 * state + 1013904223) & 0xFFFFFFFF
        yield (state / 0xFFFFFFFF) * 2.0 - 1.0


def render(loop: Loop) -> list[float]:
    """The loop as mono floats in -1..1."""
    sr = loop.sample_rate
    step = int(round(loop.spacing_s * sr))
    out = [0.0] * loop.frames
    noise = _lcg(LCG_SEED)
    attack = max(1, int(ATTACK_S * sr))
    click = int(CLICK_S * sr)
    for i, hit in enumerate(loop.hits):
        base = i * step
        w = 2.0 * math.pi * hit.freq / sr
        for n in range(step):
            env = (n / attack if n < attack else 1.0) * math.exp(-n / (DECAY_TAU_S * sr))
            v = math.sin(w * n) + HARMONIC2 * math.sin(2.0 * w * n)
            v = v / (1.0 + HARMONIC2) * env
            if n < click:
                v += CLICK_GAIN * next(noise) * (1.0 - n / click)
            out[base + n] = v * PEAK
    return out


def write_wav(path: Path, samples: Sequence[float], sample_rate: int = SAMPLE_RATE) -> Path:
    """16-bit stereo, both channels the same: what the Recorder writes."""
    path.parent.mkdir(parents=True, exist_ok=True)
    frames = bytearray()
    for v in samples:
        s = max(-32767, min(32767, int(round(v * 32767))))
        frames += struct.pack("<hh", s, s)
    with wave.open(str(path), "wb") as w:
        w.setnchannels(2)
        w.setsampwidth(2)
        w.setframerate(sample_rate)
        w.writeframes(bytes(frames))
    return path


def build(path: Path, bpm: float, hits: Sequence[Hit] = DEFAULT_HITS) -> Loop:
    """Write the loop to `path` and return what it holds."""
    loop = Loop(bpm=bpm, hits=tuple(hits))
    write_wav(path, render(loop), loop.sample_rate)
    return loop


def hits_from_spec(spec: Sequence[dict]) -> tuple[Hit, ...]:
    """The workflow file's `fixture.hits` rows ({name, freq}) as Hits."""
    return tuple(Hit(str(h["name"]), float(h["freq"])) for h in spec)
