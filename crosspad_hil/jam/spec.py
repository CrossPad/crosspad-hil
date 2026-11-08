"""The jam's fixture kits and spectrum plan: every sound has a frequency of its own.

A take of the jam is scored band by band: each pad of each kit, the pilot, the
timecode and two quiet bands each own a slice of the spectrum, so "which pad
sounded, from which kit, when" can be read off the recording without knowing
anything else. This module is the single source of those frequencies; the
kits pushed to the card are generated from it (stdlib only -- provisioning
runs on a bench box without numpy) and the analysis reads the same tables.

The plan, at 44.1 kHz (the engine plays a file 1:1, so at another engine
rate every pad frequency is scaled by rate/44100 and every length by the
inverse -- ``rendered``):

* **JAM A / JAM B** (sampler) -- ``230 + 240 p`` / ``350 + 240 p``: interleaved
  120 Hz apart, so a swap A<->B moves every pad to a band of its own. Pads 0-3
  (the sequencer's kick/snare/hat/open) and 4-7 on A (the overdub's hi-perc)
  are short, so a pattern's every hit starts from silence; pads 8-11 and 4-7
  on B are 6 s and A's pad 15 is 10 s, long enough that the SD streamer has to
  keep refilling them (a 0.6 s tone fits in its ~96 kB preload and never
  touches the card again).
* **JAM FX** (sampler, 4.3-7.9 kHz) -- one pad per kit feature: pitch, reverse,
  low/high pass, pan, volume, choke, hold, loop, two layers, a slice.
* **JAM P** (pitched) -- the whole-tone notes 85, 87 .. 115 (1109-6272 Hz) from
  three zones. It shares the range of the others, but never sounds with them:
  loading it stops the sampler engine.
* **pilot** 12 kHz (+-1 kHz kept free), **timecode** 15-19 kHz, **quiet** 9-10.5
  and 13.5-14.5 kHz (nothing should ever be there: a click lands in them).
* **the click** (1000 / 1500 Hz, the metronome) is kept 50 Hz clear of every
  pad band. 60 Hz is not possible with any interleave of two kits: 1000 and
  1500 Hz sit 20 Hz apart modulo 120 Hz, so the best placement leaves 50 Hz.

Kits are written byte-for-byte deterministically (``write_jam_kits``), the
same way ``crosspad_hil.fixtures`` does.
"""
from __future__ import annotations

import json
import math
from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path

from crosspad_hil import fixtures

SR_FILE = 44_100              # the kits' sample rate; the engine plays them 1:1
PADS = 16

JAM_A = "HIL JAM A"
JAM_B = "HIL JAM B"
JAM_FX = "HIL JAM FX"
JAM_P = "HIL JAM P"
KIT_NAMES: tuple[str, ...] = (JAM_A, JAM_B, JAM_FX, JAM_P)
KIT_DIRS = {JAM_A: "HIL_JAM_A", JAM_B: "HIL_JAM_B", JAM_FX: "HIL_JAM_FX", JAM_P: "HIL_JAM_P"}

STEP_HZ = 240.0
BASE_A_HZ = 230.0
BASE_B_HZ = 350.0
PEAK = 0.07                   # one voice; 8 voices + pilot + timecode stay under 0.9 FS
FADE_S = fixtures.FADE_S

SHORT_S = {0: 0.18, 1: 0.18, 2: 0.12, 3: 0.20}     # kick, snare, hat, open
HIPERC_S = 0.10                                    # A pads 4-7: the overdub's 16ths
DEFAULT_S = 2.0
LONG_S = 6.0
LONGEST_S = 10.0

PILOT_HZ = 12_000.0
PILOT_DBFS = -26.0
PILOT_GUARD_HZ = 1_000.0
TIMECODE_BAND = (15_000.0, 19_000.0)
TIMECODE_DBFS = -34.0
TIMECODE_PERIOD = 1 << 18
QUIET_BANDS: tuple[tuple[float, float], ...] = ((9_000.0, 10_500.0), (13_500.0, 14_500.0))
CLICK_HZ: tuple[float, ...] = (1_000.0, 1_500.0)
CLICK_GUARD_HZ = 50.0
RESOLUTION_HZ = 100.0         # two bands that may sound together are at least this far apart

P_BASE_NOTE = 85
P_NOTES = [P_BASE_NOTE + 2 * i for i in range(PADS)]          # 85, 87 .. 115
P_ZONE_ROOTS = [85, 97, 109]
P_ZONE_S = 2.0
P_PEAK = 0.5
P_PAD_VOLUME = 12
P_ATTACK_MS = 2
P_RELEASE_MS = 100
P_MAX_VOICES = 10


def note_hz(note: float) -> float:
    return 440.0 * 2.0 ** ((note - 69.0) / 12.0)


def freq_a(pad: int) -> float:
    return BASE_A_HZ + STEP_HZ * pad


def freq_b(pad: int) -> float:
    return BASE_B_HZ + STEP_HZ * pad


# --------------------------------------------------------------------------- specs

@dataclass(frozen=True)
class LayerSpec:
    """One sample file of a pad. ``shape``: flat (faded sine), decay (fade-in, then
    e^-t/0.3 s), two_tone (``freq`` then ``freq2``, ``seconds`` each half), loop
    (whole periods, no fade: seamless). ``start``/``end`` (s) make it a slice."""

    freq: float
    seconds: float
    shape: str = "flat"
    freq2: float = 0.0
    seconds2: float = 0.0
    start_s: float = 0.0
    end_s: float = 0.0
    peak: float = PEAK
    channels: int = 2


@dataclass(frozen=True)
class SamplerPadSpec:
    layers: tuple[LayerSpec, ...]
    mode: int = 0                # playbackMode: 0 one-shot, 1 hold, 2 loop (toggle)
    choke: int = 0
    volume: int = 100
    pan: int = 0                 # -100 .. 100
    pitch: int = 0               # semitones, rendered at load into .fx/
    hp: int = 0                  # Hz
    lp: int = 0                  # Hz
    reverse: bool = False
    cycle: bool = False          # several layers: round robin
    feature: str = "plain"

    @property
    def fx(self) -> bool:
        return bool(self.pitch or self.hp or self.lp or self.reverse)


@dataclass(frozen=True)
class PitchedKitSpec:
    zone_roots: tuple[int, ...]
    zone_seconds: float
    pad_notes: tuple[int, ...]
    attack_ms: int
    release_ms: int
    pad_volume: int
    max_voices: int
    peak: float = P_PEAK


@dataclass(frozen=True)
class KitSpec:
    name: str
    dirname: str
    engine: str                  # "sampler" | "pitched"
    pads: tuple[SamplerPadSpec, ...] = ()
    pitched: PitchedKitSpec | None = None


@dataclass(frozen=True)
class PilotSpec:
    hz: float = PILOT_HZ
    dbfs: float = PILOT_DBFS
    timecode: bool = True
    timecode_band: tuple[float, float] = TIMECODE_BAND
    timecode_dbfs: float = TIMECODE_DBFS
    period: int = TIMECODE_PERIOD
    seed: int = 0x5EED

    @property
    def amp(self) -> float:
        return 10.0 ** (self.dbfs / 20.0)

    @property
    def tc_amp(self) -> float:
        return 10.0 ** (self.timecode_dbfs / 20.0) if self.timecode else 0.0

    def to_dict(self) -> dict:
        return {"hz": self.hz, "dbfs": self.dbfs, "timecode": self.timecode,
                "timecode_band": list(self.timecode_band), "timecode_dbfs": self.timecode_dbfs,
                "period": self.period, "seed": self.seed}

    @staticmethod
    def from_dict(d: dict) -> PilotSpec:
        return PilotSpec(hz=float(d.get("hz", PILOT_HZ)), dbfs=float(d.get("dbfs", PILOT_DBFS)),
                         timecode=bool(d.get("timecode", True)),
                         timecode_band=tuple(d.get("timecode_band", TIMECODE_BAND)),  # type: ignore[arg-type]
                         timecode_dbfs=float(d.get("timecode_dbfs", TIMECODE_DBFS)),
                         period=int(d.get("period", TIMECODE_PERIOD)),
                         seed=int(d.get("seed", 0x5EED)))


# --------------------------------------------------------------------------- the kits

def _flat(freq: float, seconds: float) -> LayerSpec:
    return LayerSpec(freq=freq, seconds=seconds)


def _sine_kit(name: str, freq, seconds: dict[int, float]) -> KitSpec:
    pads = tuple(SamplerPadSpec(layers=(_flat(freq(p), seconds.get(p, DEFAULT_S)),))
                 for p in range(PADS))
    return KitSpec(name=name, dirname=KIT_DIRS[name], engine="sampler", pads=pads)


def _loop_seconds(freq: float, approx_s: float, rate: int = SR_FILE) -> float:
    """A length holding a whole number of periods (so the loop has no seam)."""
    g = math.gcd(int(round(freq)), rate)
    block = rate // g                       # samples per whole-cycle block
    n = max(1, int(round(approx_s * rate / block))) * block
    return n / rate


FX_PLAIN = (4_300.0, 4_500.0, 4_700.0, 4_900.0)


def _fx_kit() -> KitSpec:
    p: list[SamplerPadSpec] = []
    for i, f in enumerate(FX_PLAIN):          # pads 0-3: the sequencer's, short and plain
        p.append(SamplerPadSpec(layers=(_flat(f, SHORT_S[i]),), feature="plain"))
    # pad 1: a slice of a two-tone file -- only its second tone may be heard
    p[1] = SamplerPadSpec(layers=(LayerSpec(freq=7_900.0, seconds=0.5, shape="two_tone",
                                            freq2=FX_PLAIN[1], seconds2=0.3,
                                            start_s=0.5, end_s=0.8),), feature="slice")
    # pad 2: two layers, round robin -- either tone is right
    p[2] = SamplerPadSpec(layers=(_flat(FX_PLAIN[2], SHORT_S[2]), _flat(7_700.0, SHORT_S[2])),
                          cycle=True, feature="layers")
    p.append(SamplerPadSpec(layers=(_flat(5_100.0 / 2 ** (7 / 12), 2.0),), pitch=7,
                            feature="pitch_up"))                         # 4 -> 5100 Hz
    p.append(SamplerPadSpec(layers=(_flat(5_300.0 * 2 ** (5 / 12), 2.0),), pitch=-5,
                            feature="pitch_down"))                       # 5 -> 5300 Hz
    p.append(SamplerPadSpec(layers=(LayerSpec(freq=5_500.0, seconds=1.0, shape="decay"),),
                            reverse=True, feature="reverse"))            # 6
    p.append(SamplerPadSpec(layers=(_flat(5_700.0, 1.0),), lp=1_900, feature="lp"))   # 7
    p.append(SamplerPadSpec(layers=(_flat(5_900.0, 1.0),), hp=1_967, feature="hp"))   # 8
    p.append(SamplerPadSpec(layers=(_flat(6_100.0, 1.0),), pan=-100, feature="pan_left"))
    p.append(SamplerPadSpec(layers=(_flat(6_300.0, 1.0),), pan=100, feature="pan_right"))
    p.append(SamplerPadSpec(layers=(_flat(6_500.0, 1.0),), volume=50, feature="volume"))
    p.append(SamplerPadSpec(layers=(_flat(6_700.0, 2.0),), choke=1, feature="choke"))  # 12
    p.append(SamplerPadSpec(layers=(_flat(6_900.0, 2.0),), choke=1, feature="choke"))  # 13
    p.append(SamplerPadSpec(layers=(_flat(7_100.0, 3.0),), mode=1, feature="hold"))    # 14
    p.append(SamplerPadSpec(layers=(LayerSpec(freq=7_300.0, seconds=_loop_seconds(7_300.0, 4.0),
                                              shape="loop"),), mode=2, feature="loop"))  # 15
    return KitSpec(name=JAM_FX, dirname=KIT_DIRS[JAM_FX], engine="sampler", pads=tuple(p))


def _p_kit() -> KitSpec:
    return KitSpec(name=JAM_P, dirname=KIT_DIRS[JAM_P], engine="pitched",
                   pitched=PitchedKitSpec(zone_roots=tuple(P_ZONE_ROOTS), zone_seconds=P_ZONE_S,
                                          pad_notes=tuple(P_NOTES), attack_ms=P_ATTACK_MS,
                                          release_ms=P_RELEASE_MS, pad_volume=P_PAD_VOLUME,
                                          max_voices=P_MAX_VOICES))


def jam_kits() -> dict[str, KitSpec]:
    a_len = {**SHORT_S, **{p: HIPERC_S for p in range(4, 8)},
             **{p: LONG_S for p in range(8, 12)}, 15: LONGEST_S}
    b_len = {**SHORT_S, **{p: LONG_S for p in range(4, 8)}}
    return {JAM_A: _sine_kit(JAM_A, freq_a, a_len), JAM_B: _sine_kit(JAM_B, freq_b, b_len),
            JAM_FX: _fx_kit(), JAM_P: _p_kit()}


# --------------------------------------------------------------------------- what is heard

@dataclass(frozen=True)
class Heard:
    """A pad as the take should carry it, at the engine's rate.

    ``freqs``: the layers' tones (a round-robin pad may play any one).
    ``seconds``: how long a one-shot sounds. ``gain_l``/``gain_r``: linear
    amplitude per channel at velocity 127 before the desk. ``shape``: flat,
    rise (reversed decay), loop, hold. ``level_known``: False when a filter
    sets the level (the cutoff's slope is the firmware's, not modelled).
    """

    kit: str
    pad: int
    freqs: tuple[float, ...]
    seconds: float
    gain_l: float
    gain_r: float
    shape: str
    mode: int
    choke: int
    level_known: bool = True
    edge_end: bool = False
    fade_s: float = FADE_S
    feature: str = "plain"
    absent_freqs: tuple[float, ...] = field(default_factory=tuple)


def _pan_gains(pan: int) -> tuple[float, float]:
    """Linear balance: pan -100 = left only, +100 right only, 0 both at unity."""
    x = max(-100, min(100, int(pan))) / 100.0
    return (1.0 - max(0.0, x), 1.0 + min(0.0, x))


def heard(kit: KitSpec, pad: int, rate: int = SR_FILE) -> Heard:
    scale = rate / float(SR_FILE)
    if kit.engine == "pitched":
        ps = kit.pitched
        assert ps is not None
        amp = ps.peak * ps.pad_volume / 100.0
        return Heard(kit=kit.name, pad=pad, freqs=(note_hz(ps.pad_notes[pad]),),
                     seconds=ps.zone_seconds, gain_l=amp, gain_r=amp, shape="hold", mode=1,
                     choke=0, fade_s=ps.attack_ms / 1000.0, feature="pitched")
    sp = kit.pads[pad]
    ratio = 2.0 ** (sp.pitch / 12.0)
    freqs: list[float] = []
    absent: list[float] = []
    seconds = 0.0
    shape = "flat"
    for layer in sp.layers:
        if layer.shape == "two_tone":
            freqs.append(layer.freq2 * ratio * scale)
            absent.append(layer.freq * ratio * scale)
            seconds = max(seconds, (layer.end_s - layer.start_s) / ratio / scale)
        else:
            freqs.append(layer.freq * ratio * scale)
            seconds = max(seconds, layer.seconds / ratio / scale)
        if layer.shape == "loop":
            shape = "loop"
        elif layer.shape == "decay" and sp.reverse:
            shape = "rise"
    if sp.mode == 1:
        shape = "hold"
    gl, gr = _pan_gains(sp.pan)
    amp = sp.layers[0].peak * sp.volume / 100.0
    return Heard(kit=kit.name, pad=pad, freqs=tuple(freqs), seconds=seconds, gain_l=amp * gl,
                 gain_r=amp * gr, shape=shape, mode=sp.mode, choke=sp.choke,
                 level_known=not (sp.lp or sp.hp), edge_end=sp.reverse,
                 fade_s=0.0 if shape == "loop" else FADE_S, feature=sp.feature,
                 absent_freqs=tuple(absent))


def all_bands(kits: dict[str, KitSpec] | None = None, rate: int = SR_FILE
              ) -> list[tuple[str, int, float]]:
    """``(kit, pad, Hz)`` of every tone the kits can play, at ``rate``."""
    kits = kits or jam_kits()
    out = []
    for name, kit in kits.items():
        for pad in range(PADS):
            h = heard(kit, pad, rate)
            out.extend((name, pad, f) for f in (*h.freqs, *h.absent_freqs))
    return out


# --------------------------------------------------------------------------- the plan check

# Kits that can sound at the same moment: a swap leaves the old kit's voices
# ringing next to the new one's, and the sequencer plays whichever is loaded.
# The pitched kit never sounds with a sampler kit (loading it stops that engine).
CO_SOUNDING: tuple[tuple[str, ...], ...] = ((JAM_A, JAM_B, JAM_FX), (JAM_P,))


def check_plan(rate: int = SR_FILE, kits: dict[str, KitSpec] | None = None,
               pilot: PilotSpec | None = None) -> list[str]:
    """Everything wrong with the plan at ``rate``; empty = every band is its own."""
    kits = kits or jam_kits()
    pilot = pilot or PilotSpec()
    scale = rate / float(SR_FILE)
    problems: list[str] = []
    bands = all_bands(kits, rate)
    nyq = rate / 2.0
    for group in CO_SOUNDING:
        mine = sorted((f, k, p) for k, p, f in bands if k in group)
        for (f1, k1, p1), (f2, k2, p2) in zip(mine, mine[1:], strict=False):
            if f2 - f1 < RESOLUTION_HZ:
                problems.append(f"{k1} pad {p1} ({f1:.0f} Hz) and {k2} pad {p2} ({f2:.0f} Hz) "
                                f"are {f2 - f1:.0f} Hz apart (< {RESOLUTION_HZ:.0f})")
    for k, p, f in bands:
        if f >= 0.45 * rate:
            problems.append(f"{k} pad {p}: {f:.0f} Hz too close to Nyquist ({nyq:.0f})")
        if abs(f - pilot.hz) < PILOT_GUARD_HZ:
            problems.append(f"{k} pad {p} ({f:.0f} Hz) inside the pilot's +-1 kHz")
        lo, hi = pilot.timecode_band
        if lo - 500 <= f <= hi + 500:
            problems.append(f"{k} pad {p} ({f:.0f} Hz) inside the timecode band")
        for qa, qb in QUIET_BANDS:
            if qa - 300 <= f <= qb + 300:
                problems.append(f"{k} pad {p} ({f:.0f} Hz) inside quiet band {qa:.0f}-{qb:.0f}")
        for c in CLICK_HZ:
            if abs(f - c) < CLICK_GUARD_HZ - 1e-6:
                problems.append(f"{k} pad {p} ({f:.0f} Hz) {abs(f - c):.0f} Hz from the "
                                f"click's {c:.0f} Hz (< {CLICK_GUARD_HZ:.0f})")
    if pilot.timecode_band[1] >= 0.45 * rate:
        problems.append(f"timecode band reaches {pilot.timecode_band[1]:.0f} Hz at {rate} Hz")
    for qa, qb in QUIET_BANDS:
        if abs((qa + qb) / 2 - pilot.hz) < PILOT_GUARD_HZ + (qb - qa) / 2:
            problems.append(f"quiet band {qa:.0f}-{qb:.0f} overlaps the pilot's guard")
    del scale
    return problems


def worst_case_peak(rate: int = SR_FILE, voices: int = 8,
                    pilot: PilotSpec | None = None) -> float:
    """Full-scale fraction when the loudest ``voices`` sampler voices, the pilot and the
    timecode peak together (the bus has no limiter: over 1.0 it clips)."""
    pilot = pilot or PilotSpec()
    kits = jam_kits()
    gains = sorted((max(heard(k, p, rate).gain_l, heard(k, p, rate).gain_r)
                    for k in kits.values() if k.engine == "sampler" for p in range(PADS)),
                   reverse=True)
    sampler = sum(gains[:voices])
    ps = kits[JAM_P].pitched
    assert ps is not None
    pitched = ps.max_voices * ps.peak * ps.pad_volume / 100.0
    # timecode: crest factor of band-limited noise, ~4 x RMS
    return max(sampler, pitched) + pilot.amp + 4.0 * pilot.tc_amp


# --------------------------------------------------------------------------- the files

def _frames(layer: LayerSpec, rate: int = SR_FILE) -> list[float]:
    def tone(freq: float, seconds: float, fade: bool = True) -> list[float]:
        n = int(round(rate * seconds))
        fd = max(1, int(rate * FADE_S)) if fade else 0
        out = []
        for i in range(n):
            g = fixtures.fade_gain(i, n, fd) if fd else 1.0
            out.append(math.sin(2 * math.pi * freq * i / rate) * g)
        return out

    if layer.shape == "flat":
        return [v * layer.peak for v in tone(layer.freq, layer.seconds)]
    if layer.shape == "loop":
        return [v * layer.peak for v in tone(layer.freq, layer.seconds, fade=False)]
    if layer.shape == "decay":
        n = int(round(rate * layer.seconds))
        fd = max(1, int(rate * FADE_S))
        out = []
        for i in range(n):
            g = (i / fd if i < fd else 1.0) * math.exp(-(i / rate) / 0.3)
            if i >= n - fd:
                g *= (n - 1 - i) / fd
            out.append(math.sin(2 * math.pi * layer.freq * i / rate) * g * layer.peak)
        return out
    if layer.shape == "two_tone":
        return [v * layer.peak for v in (tone(layer.freq, layer.seconds)
                                         + tone(layer.freq2, layer.seconds2))]
    raise ValueError(f"unknown layer shape {layer.shape!r}")


def _write_mono_or_stereo(path: Path, samples: Iterable[float], channels: int,
                          rate: int = SR_FILE) -> None:
    frames = [(fixtures.quantize(v),) * channels for v in samples]
    fixtures.write_wav(path, frames, channels=channels, rate=rate)


def write_kit(root: Path, kit: KitSpec) -> Path:
    """One kit directory (kit.json + SAMPLES/) under ``root``; returns its path."""
    out = Path(root) / kit.dirname
    (out / "SAMPLES").mkdir(parents=True, exist_ok=True)
    color = fixtures.SINE_PAD_COLOR
    if kit.engine == "pitched":
        ps = kit.pitched
        assert ps is not None
        zones = []
        for note in ps.zone_roots:
            name = f"z{note}.wav"
            n = int(round(SR_FILE * ps.zone_seconds))
            fd = max(1, int(SR_FILE * FADE_S))
            hz = note_hz(note)
            _write_mono_or_stereo(out / "SAMPLES" / name,
                                  (math.sin(2 * math.pi * hz * i / SR_FILE)
                                   * fixtures.fade_gain(i, n, fd) * ps.peak for i in range(n)),
                                  channels=1)
            zones.append({"sample": name, "root": note})
        doc = {"name": kit.name, "author": "hil", "engine": "pitched", "type": 0,
               "description": ["live_jam pitched fixture: whole tones 85..115 from 3 zones"],
               "instrument": {"zones": zones, "pad_notes": list(ps.pad_notes),
                              "base_note": ps.pad_notes[0], "release_ms": ps.release_ms,
                              "attack_ms": ps.attack_ms, "max_voices": ps.max_voices},
               "pads": [{"note": i, "color": color, "color_pressed": color, "playbackMode": 1,
                         "volume": ps.pad_volume, "pan": 0, "samples": []}
                        for i in range(PADS)]}
    else:
        pads = []
        for p, sp in enumerate(kit.pads):
            samples: list = []
            for j, layer in enumerate(sp.layers):
                name = f"p{p:02d}{'abcdefgh'[j]}.wav"
                _write_mono_or_stereo(out / "SAMPLES" / name, _frames(layer), layer.channels)
                if layer.end_s > layer.start_s:
                    samples.append({"sample": name, "start": int(round(layer.start_s * SR_FILE)),
                                    "end": int(round(layer.end_s * SR_FILE))})
                else:
                    samples.append(name)
            entry: dict = {"color": color, "color_pressed": color, "choke_group": sp.choke,
                           "cycle": sp.cycle, "pitch": sp.pitch, "volume": sp.volume,
                           "pan": sp.pan, "playbackMode": sp.mode, "samples": samples}
            if sp.hp:
                entry["hp"] = sp.hp
            if sp.lp:
                entry["lp"] = sp.lp
            if sp.reverse:
                entry["reverse"] = True
            pads.append(entry)
        doc = {"name": kit.name, "author": "hil",
               "description": [f"live_jam fixture ({kit.name}): every pad its own tone"],
               "version": "1.0", "created_at": "2026-10-07", "pads": pads}
    (out / "kit.json").write_text(json.dumps(doc, indent=2))
    return out


def write_jam_kits(root: Path) -> dict[str, Path]:
    """All four jam kits under ``root``: ``{kit name: directory}`` (``_audio.ensure_kits``)."""
    return {name: write_kit(Path(root), kit) for name, kit in jam_kits().items()}
