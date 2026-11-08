"""Fixture kits this package can build and push itself, without platform-idf.

A scenario that needs a purpose-built kit used to be a scenario you could only
run after remembering to run a tool in another repo first — and the failure mode
was `ENV: no kit named 'HIL SINE A'` several seconds into a run. The generators
live here so `kit_swap_audio` can provision a card that does not have them
(see `scenarios/_audio.ensure_kits`).

`platform-idf/tools/make_sine_kits.py` remains the hand-run tool and produces
**byte-identical** output: same fade, same quantisation, same key order in
kit.json. `tests/test_fixtures.py` generates both ways and compares the bytes
whenever that repo is next to this one, because "the same kit" is a claim about
bytes, not about intent — a fixture that differs between the two paths would
make a recording score differently depending on who pushed it.

Everything here is stdlib (`wave`, `math`, `struct`, `json`): provisioning must
work on a bench box that has no numpy.
"""
from __future__ import annotations

import json
import math
import struct
import wave
from collections.abc import Callable, Sequence
from pathlib import Path

# --- shared with platform-idf/tools/_kit_fixture.py -------------------------
FADE_S = 0.005               # 5 ms fade in and out, on every fixture sample
INT16_FULL_SCALE = 32767

# --- the sine kits ----------------------------------------------------------
SINE_PAD_COUNT = 16
SINE_SAMPLE_RATE = 44100     # kits ship at 44.1 kHz; the engine plays files 1:1
SINE_SECONDS = 2.0           # the tone length kit_swap_audio's scoring window assumes
SINE_PEAK = 0.12             # ~-18 dBFS; 8 overlapping tones stay under full scale
SINE_PAD_COLOR = 2162448     # the fixture colour make_tone_kit uses
SINE_PAD_VOLUME = 100

STEP_HZ = 222.0
BASE_A_HZ = 300.0
BASE_B_HZ = 411.0

SINE_SAMPLE_NAMES = [f"pad{p:02d}.wav" for p in range(SINE_PAD_COUNT)]


def freq_a(pad: int) -> float:
    """Kit A, pad p: 300 + 222*p Hz."""
    return BASE_A_HZ + STEP_HZ * pad


def freq_b(pad: int) -> float:
    """Kit B, pad p: 411 + 222*p Hz — every A tone's neighbour, 111 Hz above."""
    return BASE_B_HZ + STEP_HZ * pad


# (directory on the card, kit.json name, frequency formula)
SINE_KITS: tuple[tuple[str, str, Callable[[int], float]], ...] = (
    ("HIL_SINE_A", "HIL SINE A", freq_a),
    ("HIL_SINE_B", "HIL SINE B", freq_b),
)


def fade_gain(i: int, n: int, fade: int) -> float:
    """Linear fade-in/out envelope for sample `i` of `n`, `fade` samples deep."""
    if i < fade:
        return i / fade
    if i >= n - fade:
        return (n - 1 - i) / fade
    return 1.0


def quantize(v: float) -> int:
    """Clip to full scale and quantize one float sample to int16."""
    return int(max(-1.0, min(1.0, v)) * INT16_FULL_SCALE)


def sine_frames(freqs: Sequence[float], *, rate: int, seconds: float, peak: float,
                channels: int) -> list[tuple[int, ...]]:
    """Faded sum of `freqs`, the same signal on every channel."""
    n = int(rate * seconds)
    fade = max(1, int(rate * FADE_S))
    out = []
    for i in range(n):
        t = i / rate
        v = sum(math.sin(2 * math.pi * f * t) for f in freqs) / len(freqs) * peak
        s = quantize(v * fade_gain(i, n, fade))
        out.append((s,) * channels)
    return out


def write_wav(path: Path, samples, *, channels: int, rate: int) -> None:
    """Write 16-bit PCM frames (one tuple of `channels` ints per frame)."""
    frames = bytearray()
    fmt = "<" + "h" * channels
    for frame in samples:
        frames += struct.pack(fmt, *frame)
    with wave.open(str(path), "wb") as w:
        w.setnchannels(channels)
        w.setsampwidth(2)
        w.setframerate(rate)
        w.writeframes(bytes(frames))


def write_sine_kit(root: Path, kit_name: str, freq: Callable[[int], float], *,
                   seconds: float = SINE_SECONDS, rate: int = SINE_SAMPLE_RATE,
                   peak: float = SINE_PEAK) -> Path:
    """One kit directory: kit.json plus one stereo sine per pad. Returns `root`."""
    (root / "SAMPLES").mkdir(parents=True, exist_ok=True)
    pads = []
    for pad in range(SINE_PAD_COUNT):
        frames = sine_frames([freq(pad)], rate=rate, seconds=seconds, peak=peak, channels=2)
        write_wav(root / "SAMPLES" / SINE_SAMPLE_NAMES[pad], frames, channels=2, rate=rate)
        # playbackMode absent = one-shot: the tone runs to its end whatever the
        # gate does, which is what makes a fixed-length scoring window possible.
        pads.append({"color": SINE_PAD_COLOR, "color_pressed": SINE_PAD_COLOR,
                     "choke_group": 0, "cycle": False, "pitch": 0,
                     "volume": SINE_PAD_VOLUME, "pan": 0,
                     "samples": [SINE_SAMPLE_NAMES[pad]]})
    kit = {"name": kit_name, "author": "hil",
           "description": [f"per-pad sine fixture for kit_swap_audio ({kit_name})"],
           "version": "1.0", "created_at": "2026-09-06", "pads": pads}
    (root / "kit.json").write_text(json.dumps(kit))
    return root


def write_sine_kits(root: Path, *, seconds: float = SINE_SECONDS,
                    peak: float = SINE_PEAK,
                    rate: int = SINE_SAMPLE_RATE) -> dict[str, Path]:
    """Both interleaved sine kits under `root`; `{kit name: directory}`.

    The directory basename is what the card gets (`HIL_SINE_A`), the key is the
    name the firmware reports in `KIT_LIST` (`HIL SINE A`) — the two differ, and
    a provisioner needs both.
    """
    root = Path(root)
    out: dict[str, Path] = {}
    for dirname, name, freq in SINE_KITS:
        out[name] = write_sine_kit(root / dirname, name, freq, seconds=seconds,
                                   rate=rate, peak=peak)
    return out


def kit_files(kit_dir: Path) -> list[Path]:
    """kit.json plus every SAMPLES/*.wav, in the order a push should send them.

    kit.json goes last on purpose: it is what a kit scan keys off, so a card
    interrupted mid-push has an incomplete directory rather than a kit that
    claims samples it does not have.
    """
    return [*sorted((kit_dir / "SAMPLES").glob("*.wav")), kit_dir / "kit.json"]


# --- the pitched kit (platform-idf/tools/make_pitched_kit.py) ----------------
# The second sample engine: 1, 2 or 4 recorded zones, every pad playing them
# transposed to its own MIDI note. Same bytes as the hand-run tool, so a card
# provisioned from either scores the same.

PITCHED_DIR = "HIL_PITCHED"
PITCHED_NAME = "HIL PITCHED"
PITCHED_ZONE_ROOTS: dict[int, list[int]] = {1: [69], 2: [57, 69], 4: [57, 69, 81, 93]}
PITCHED_SECONDS_BY_ZONES = {1: 1.0, 2: 1.0, 4: 0.5}
PITCHED_PEAK = 0.5            # -6 dBFS
PITCHED_BASE_NOTE = 60        # pad_notes run base_note..base_note+15
PITCHED_RELEASE_MS = 100
PITCHED_ATTACK_MS = 2
PITCHED_MAX_VOICES = 24
PITCHED_PAD_VOLUME = 20       # a 16-pad chord of -6 dBFS sines stays under full scale


def note_to_hz(note: float) -> float:
    """MIDI note -> Hz, A4 (69) = 440 Hz."""
    return 440.0 * (2.0 ** ((note - 69.0) / 12.0))


def write_pitched_kit(root: Path, *, zones: int = 2, rate: int = SINE_SAMPLE_RATE,
                      seconds: float | None = None,
                      pad_volume: int = PITCHED_PAD_VOLUME) -> Path:
    """kit.json plus one mono sine per zone; even zones an explicit root, odd
    zones ``"root": "auto"`` (both paths in one kit). Returns `root`."""
    if zones not in PITCHED_ZONE_ROOTS:
        raise ValueError(f"zones must be one of {sorted(PITCHED_ZONE_ROOTS)}, got {zones}")
    seconds = PITCHED_SECONDS_BY_ZONES[zones] if seconds is None else seconds
    samples_dir = root / "SAMPLES"
    samples_dir.mkdir(parents=True, exist_ok=True)
    zone_specs = []
    for i, note in enumerate(PITCHED_ZONE_ROOTS[zones]):
        fname = f"z{note}.wav"
        frames = sine_frames([note_to_hz(note)], rate=rate, seconds=seconds,
                             peak=PITCHED_PEAK, channels=1)
        write_wav(samples_dir / fname, frames, channels=1, rate=rate)
        zone_specs.append({"sample": fname, "root": note if i % 2 == 0 else "auto"})
    pads = [{"note": i, "color": SINE_PAD_COLOR, "color_pressed": SINE_PAD_COLOR,
             "playbackMode": 1, "volume": pad_volume, "pan": 0, "samples": []}
            for i in range(SINE_PAD_COUNT)]
    kit = {
        "name": PITCHED_NAME, "author": "hil", "engine": "pitched", "type": 0,
        "instrument": {
            "zones": zone_specs,
            "pad_notes": [PITCHED_BASE_NOTE + i for i in range(SINE_PAD_COUNT)],
            "base_note": PITCHED_BASE_NOTE, "release_ms": PITCHED_RELEASE_MS,
            "attack_ms": PITCHED_ATTACK_MS, "max_voices": PITCHED_MAX_VOICES,
        },
        "pads": pads,
    }
    (root / "kit.json").write_text(json.dumps(kit, indent=2))
    return root


def write_pitched_kits(root: Path) -> dict[str, Path]:
    """`{"HIL PITCHED": <root>/HIL_PITCHED}`, the shape `_audio.ensure_kits` wants."""
    root = Path(root)
    return {PITCHED_NAME: write_pitched_kit(root / PITCHED_DIR)}


def write_live_set_kits(root: Path) -> dict[str, Path]:
    """Both sine kits and the pitched kit: everything `live_set` loads."""
    return {**write_sine_kits(root), **write_pitched_kits(root)}


# --- the click kit (latency) -------------------------------------------------
# A latency onset needs a sample that starts at full level on its first frame:
# the sine kits fade in over 5 ms, and a threshold on a ramp reads the ramp's
# slope, not the start. Every pad here is a short burst of deterministic noise
# (its own seed, so two pads are told apart), opened by a full-scale doublet
# so the first frame is unambiguous even after the DAC/ADC filters, faded out
# over 1 ms and padded with silence to a fixed length.

CLICK_DIR = "HIL_CLICK"
CLICK_NAME = "HIL CLICK"
CLICK_NOISE_S = 0.020        # the burst
CLICK_FADE_S = 0.001         # its fade-out; there is no fade-in on purpose
CLICK_SECONDS = 0.050        # burst + silence: a one-shot that is over before the next hit
CLICK_PEAK = 0.25            # ~-12 dBFS: under the jack loop's compression knee (~0.2 FS
                             # after the analog stage), with room for two overlapping voices
CLICK_SAMPLE_NAMES = [f"click{p:02d}.wav" for p in range(SINE_PAD_COUNT)]
_LCG_A, _LCG_C, _LCG_M = 1103515245, 12345, 2 ** 31


def click_seed(pad: int) -> int:
    """The noise seed of pad `pad` (the host-side burst uses its own, past the pads)."""
    return 0x5EED + 7919 * pad


def click_samples(seed: int, *, rate: int = SINE_SAMPLE_RATE, noise_s: float = CLICK_NOISE_S,
                  fade_s: float = CLICK_FADE_S, seconds: float = CLICK_SECONDS,
                  peak: float = CLICK_PEAK) -> list[float]:
    """One click as floats in [-peak, peak]: doublet, LCG noise, 1 ms fade-out, silence.

    Pure stdlib so the card can be provisioned from a box without numpy, and
    the same function renders the host-side burst the latency scenario plays
    into the board.
    """
    n_total = max(1, int(round(rate * seconds)))
    n_noise = min(n_total, max(2, int(round(rate * noise_s))))
    n_fade = max(1, int(round(rate * fade_s)))
    x = seed & (_LCG_M - 1)
    out = [0.0] * n_total
    for i in range(n_noise):
        if i == 0:
            v = 1.0
        elif i == 1:
            v = -1.0
        else:
            x = (_LCG_A * x + _LCG_C) % _LCG_M
            v = x / (_LCG_M / 2.0) - 1.0
        gain = 1.0 if i < n_noise - n_fade else (n_noise - 1 - i) / n_fade
        out[i] = v * peak * gain
    return out


def write_click_kit(root: Path, *, rate: int = SINE_SAMPLE_RATE) -> Path:
    """kit.json plus one stereo click per pad. Returns `root`."""
    (root / "SAMPLES").mkdir(parents=True, exist_ok=True)
    pads = []
    for pad in range(SINE_PAD_COUNT):
        frames = [(quantize(v),) * 2 for v in click_samples(click_seed(pad), rate=rate)]
        write_wav(root / "SAMPLES" / CLICK_SAMPLE_NAMES[pad], frames, channels=2, rate=rate)
        # One-shot (no playbackMode), like the sine kits: the click plays out
        # whatever the gate does.
        pads.append({"color": SINE_PAD_COLOR, "color_pressed": SINE_PAD_COLOR,
                     "choke_group": 0, "cycle": False, "pitch": 0,
                     "volume": SINE_PAD_VOLUME, "pan": 0,
                     "samples": [CLICK_SAMPLE_NAMES[pad]]})
    kit = {"name": CLICK_NAME, "author": "hil",
           "description": ["per-pad noise click for the latency scenario: full level "
                           "from the first frame"],
           "version": "1.0", "created_at": "2026-10-06", "pads": pads}
    (root / "kit.json").write_text(json.dumps(kit))
    return root


def write_click_kits(root: Path) -> dict[str, Path]:
    """`{"HIL CLICK": <root>/HIL_CLICK}`, the shape `_audio.ensure_kits` wants."""
    root = Path(root)
    return {CLICK_NAME: write_click_kit(root / CLICK_DIR)}
