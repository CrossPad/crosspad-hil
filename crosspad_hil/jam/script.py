"""The jam, step by step: what is played, when, and what each step proves.

``build_script(seed, ...)`` lays out the whole set -- the CDC windows between
the USB-audio blocks and every timed step inside a block -- deterministically
for a seed. Every step names the features it exercises, what it should do
(``expect``) and how the analysis checks it (``verify``); a step that needs a
capability the board does not have is kept with ``skip`` and the reason, so
the coverage table says what was not measured rather than leaving it out.

Times are seconds from the block's start; ``scale`` shrinks every block (the
synthetic tests run the same script in seconds).
"""
from __future__ import annotations

import random
from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any

from crosspad_hil.jam import spec as S


class Feature(str, Enum):
    STREAM_IN = "stream_in"                   # the phone's stream reaches the bus
    CAPTURE_BUS = "capture_sampler_bus"       # USB IN = the digital recording bus
    PROFILE_SWITCH = "profile_switch"
    PADS_SYSEX = "pads_sysex"
    PADS_COMPANION = "pads_companion"
    PADS_CDC = "pads_cdc"
    RETRIGGER = "retrigger"
    FOUR = "four_alternating"
    MANY = "many_round_robin"
    POLY_LIMIT = "sampler_polyphony_steal"
    VELOCITY = "velocity"
    ONE_SHOT = "one_shot"
    HOLD = "hold_mode"
    LOOP = "loop_mode"
    CHOKE = "choke_group"
    LAYERS = "layers_cycle"
    SLICE = "slice"
    PAN = "pan"
    VOLUME = "pad_volume"
    PITCH = "fx_pitch"
    REVERSE = "fx_reverse"
    LP = "fx_lowpass"
    HP = "fx_highpass"
    FX_RENDER = "fx_render_on_load"
    STREAMING = "long_samples_streaming"
    SWAP = "kit_swap_sampler"
    SWAP_COMPANION = "kit_swap_companion"
    ENGINE_SWITCH = "engine_switch"
    PITCHED = "pitched_notes"
    PITCHED_STEAL = "pitched_cap_steal"
    PITCHED_RETRIG = "pitched_retrigger"
    SEQ_REC = "seq_rec_countin"
    SEQ_OVERDUB = "seq_overdub"
    QUANT_16 = "quant_1_16"
    QUANT_8T = "quant_1_8T"
    SCENES = "scenes"
    DOUBLE = "scene_double"
    ERASE = "pad_erase"
    SONG = "song_mode"
    CLOCK_TRANSPORT = "midi_clock_stop_continue_start"
    CLOCK_TIMEOUT = "midi_clock_handback"
    BOUNCE_SYSEX = "bounce_sysex"
    BOUNCE_CDC = "bounce_cdc"
    MIX_PRESET = "mixer_preset"
    MIX_ROUTE = "mixer_route"
    CODEC_NEG = "codec_negative_control"
    PREVIEW = "preview"
    APP_SWITCH = "app_switch"
    COMPANION = "companion_status"
    ECHO = "sysex_echo_rtt"
    STREAM_PAUSE = "stream_pause"
    RECORDER = "recorder_pads"
    NOTE_NEG = "host_note_negative"
    SOAK = "soak"


@dataclass
class Hit:
    t: float              # seconds from the step's start
    pad: int
    vel: int = 100
    hold_ms: float = 40.0
    via: str = "sysex"    # sysex | companion


@dataclass
class Step:
    id: str
    block: str
    t_rel: float
    dur: float
    action: str
    args: dict[str, Any] = field(default_factory=dict)
    needs: list[str] = field(default_factory=list)
    expect: str = ""
    verify: str = ""
    features: list[str] = field(default_factory=list)
    flags: list[str] = field(default_factory=list)
    hits: list[Hit] = field(default_factory=list)
    skip: str = ""

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["hits"] = [[round(h.t, 4), h.pad, h.vel, round(h.hold_ms, 1), h.via] for h in self.hits]
        return d


@dataclass
class Block:
    name: str
    title: str
    duration: float
    kit0: str
    steps: list[Step] = field(default_factory=list)
    before: list[str] = field(default_factory=list)      # CDC window that arms it
    after: list[str] = field(default_factory=list)       # CDC window that closes it

    def to_dict(self) -> dict[str, Any]:
        return {"name": self.name, "title": self.title, "duration": round(self.duration, 3),
                "kit0": self.kit0, "steps": [s.to_dict() for s in self.steps],
                "before": self.before, "after": self.after}


@dataclass
class WindowOp:
    op: str
    features: list[str] = field(default_factory=list)
    verify: str = ""
    skip: str = ""


@dataclass
class JamScript:
    seed: int
    scale: float
    bpm: int
    blocks: list[Block]
    windows: dict[str, list[WindowOp]]
    order: list[str]
    skipped: list[dict[str, str]] = field(default_factory=list)

    def block(self, name: str) -> Block:
        return next(b for b in self.blocks if b.name == name)

    def to_dict(self) -> dict[str, Any]:
        return {"seed": self.seed, "scale": self.scale, "bpm": self.bpm,
                "blocks": [b.to_dict() for b in self.blocks],
                "windows": {k: [asdict(o) for o in v] for k, v in self.windows.items()},
                "order": self.order, "skipped": self.skipped}

    @property
    def uac_seconds(self) -> float:
        return sum(b.duration for b in self.blocks)


# --------------------------------------------------------------------------- patterns

def one(rate: float, dur: float, pad: int, rng: random.Random) -> list[Hit]:
    n = int(dur * rate)
    return [Hit(t=i / rate, pad=pad, vel=100) for i in range(n)]


def four(rate: float, dur: float, pads: list[int], rng: random.Random) -> list[Hit]:
    n = int(dur * rate)
    return [Hit(t=i / rate, pad=pads[i % len(pads)], vel=100) for i in range(n)]


def many(rate: float, dur: float, chord_every: float, rng: random.Random,
         chord: int = 8) -> list[Hit]:
    hits: list[Hit] = []
    n = int(dur * rate)
    for i in range(n):
        hits.append(Hit(t=i / rate, pad=i % 16))
    k = 1
    while k * chord_every < dur:
        t = k * chord_every
        pads = rng.sample(range(16), chord)
        for j, p in enumerate(pads):
            hits.append(Hit(t=t + j * 0.0005, pad=p))
        k += 1
    hits.sort(key=lambda h: h.t)
    return hits


def velocity_ramp(pad: int, gap: float) -> list[Hit]:
    return [Hit(t=i * gap, pad=pad, vel=v) for i, v in enumerate(range(8, 128, 7))]


def legato(dur: float, rate: float, rng: random.Random) -> list[Hit]:
    hits, t, p = [], 0.0, 7
    while t < dur:
        hold = rng.uniform(150.0, 300.0)
        hits.append(Hit(t=t, pad=p, vel=rng.randint(80, 120), hold_ms=hold))
        p = max(0, min(15, p + rng.choice((-2, -1, 1, 2))))
        t += 1.0 / rate
    return hits


def chords(n: int, every: float, size: int, hold_ms: float, rng: random.Random) -> list[Hit]:
    hits = []
    for k in range(n):
        for j, p in enumerate(rng.sample(range(16), size)):
            hits.append(Hit(t=k * every + j * 0.0005, pad=p, vel=100, hold_ms=hold_ms))
    return hits


def fx_tour() -> list[Hit]:
    """One hit (or a gesture) per FX pad, spaced so each is heard on its own."""
    plan: list[Hit] = []
    t = 0.0

    def add(pad: int, gap: float, hold: float = 40.0, n: int = 1, every: float = 0.0) -> None:
        nonlocal t
        for i in range(n):
            plan.append(Hit(t=t + i * every, pad=pad, hold_ms=hold))
        t += gap + (n - 1) * every

    add(4, 1.6)                 # pitch +7: 5100 Hz, 1.33 s
    add(5, 2.9)                 # pitch -5: 5300 Hz, 2.7 s
    add(6, 1.3)                 # reverse: a rise ending on an edge
    add(7, 1.3)                 # low pass: quieter
    add(8, 1.3)                 # high pass: passes
    add(9, 1.3)                 # pan left
    add(10, 1.3)                # pan right
    add(11, 1.3)                # volume 50
    add(12, 0.6)                # choke 1 ...
    add(13, 2.3)                # ... 13 cuts 12
    add(14, 1.0, hold=500.0)    # hold: ends at the release
    add(15, 2.5)                # loop on
    add(15, 0.6)                # loop off (toggle)
    add(2, 0.5, n=4, every=0.4)  # two layers, round robin
    add(1, 0.6, n=2, every=0.5)  # the slice: only its second tone
    return plan


# --------------------------------------------------------------------------- the set

BLOCK_S = {"B1": 240.0, "B1b": 30.0, "B2": 255.0, "B3": 200.0}


def build_script(seed: int = 1, *, minutes: float = 12.5, bpm: int = 120, soak_min: float = 0.0,
                 sections: list[str] | None = None, caps: dict[str, Any] | None = None,
                 one_rate: float = 12.0, four_rate: float = 8.0, many_rate: float = 16.0,
                 chord_every: float = 4.0, stream_pause_mode: str = "drop") -> JamScript:
    caps = caps or {}
    rng = random.Random(seed)
    base = sum(BLOCK_S.values())
    scale = max(0.01, minutes * 60.0 / base)
    want = set(sections or ["all"])

    def on(section: str) -> bool:
        return "all" in want or section in want

    def T(x: float) -> float:  # noqa: N802 - reads like the script's own time unit
        return x * scale

    skipped: list[dict[str, str]] = []
    beat = 60.0 / bpm

    def step(block: str, sid: str, t: float, dur: float, action: str, *, features: list[Feature],
             expect: str, verify: str, flags: list[str] | None = None,
             hits: list[Hit] | None = None, needs: list[str] | None = None,
             section: str = "", **args: Any) -> Step:
        s = Step(id=f"{block}.{sid}", block=block, t_rel=T(t), dur=T(dur), action=action,
                 args=args, needs=needs or [], expect=expect, verify=verify,
                 features=[f.value for f in features], flags=flags or [], hits=hits or [])
        if section and not on(section):
            s.skip = f"section {section!r} not selected"
        for need in s.needs:
            if caps and not caps.get(need, True):
                s.skip = s.skip or f"the board has no {need}"
        if s.skip:
            skipped.append({"step": s.id, "why": s.skip,
                            "features": ",".join(s.features)})
        return s

    def fit(hits: list[Hit], dur: float) -> list[Hit]:
        """Hits keep their own timing (a rate, a groove); a shorter set plays fewer."""
        return [h for h in hits if 0.0 <= h.t < T(dur)]

    blocks: list[Block] = []

    # ---- B1: build the beat ------------------------------------------------
    b1 = Block("B1", "budowa bitu: one, four + swapy, preview, velocity, bounce #1",
               T(BLOCK_S["B1"]), S.JAM_A, before=["W0"], after=["W1"])
    b1.steps += [
        step("B1", "stream", 0, BLOCK_S["B1"], "listen",
             features=[Feature.STREAM_IN, Feature.CAPTURE_BUS, Feature.ECHO, Feature.SCENES,
                       Feature.QUANT_16, Feature.SEQ_REC, Feature.PROFILE_SWITCH],
             expect="pilot continuous; S1 (recorded in W0) plays its passes",
             verify="pilot/timecode: no gap, no slip; every S1 event in the take (SEQ_MISS)"),
        step("B1", "one", 15, 45, "pattern", flags=["one"],
             features=[Feature.PADS_SYSEX, Feature.RETRIGGER, Feature.ONE_SHOT],
             expect="pad 12 restarts at every hit, one voice",
             verify="each hit's restart found and on time (LATE), no hole, no click",
             hits=fit(one(one_rate, T(45), 12, rng), 45)),
        step("B1", "four", 60, 75, "pattern", flags=["four", "streaming"],
             features=[Feature.FOUR, Feature.STREAMING],
             expect="pads 8-11 (6 s samples) alternate; the SD streamer keeps refilling",
             verify="every hit on time; no hole in any long tone",
             hits=fit(four(four_rate, T(75), [8, 9, 10, 11], rng), 75)),
        step("B1", "swap_ab", 75, 0.1, "swap", flags=["swap"], kit=S.JAM_B,
             features=[Feature.SWAP, Feature.COMPANION],
             expect="free pads switch to B at once, sounding ones when their voice ends",
             verify="new hits in B's bands; no reversion; <=1 held hit, <=50 ms (LATE_SWAP)"),
        step("B1", "swap_ba", 120, 0.1, "swap", flags=["swap"], kit=S.JAM_A,
             features=[Feature.SWAP], expect="back to A", verify="as above"),
        step("B1", "preview", 137, 4, "preview", pads=[0, 5, 9], gap=1.2,
             features=[Feature.PREVIEW],
             expect="the pad's first sample through the sampler engine (a voice)",
             verify="its tone appears, no hole (a voice in the steal model)"),
        step("B1", "codec_neg", 142, 4, "codec_neg",
             features=[Feature.CODEC_NEG],
             expect="codec volume/mute moves; the digital bus does not",
             verify="no level change, no gap on the take around it"),
        step("B1", "velocity", 147, 9, "pattern", flags=["velocity"],
             features=[Feature.VELOCITY],
             expect="pad 5 louder with every hit", verify="level rises monotonically",
             hits=fit(velocity_ramp(5, 0.45), 9)),
        step("B1", "companion_pads", 157, 6, "pattern", flags=["companion"],
             features=[Feature.PADS_COMPANION],
             expect="pads struck through the companion verb sound",
             verify="each hit found (latency per path)", needs=["companion"],
             hits=fit([Hit(t=0.5 * i, pad=6 + i % 2, via="companion")
                           for i in range(10)], 6)),
        step("B1", "note_neg", 165, 3, "note_neg", note=60,
             features=[Feature.NOTE_NEG],
             expect="a host MIDI note plays nothing (SamplerPadLogic::onNoteOn is empty)",
             verify="no tone anywhere (product gap if one appears); always closed by NoteOff"),
        step("B1", "stream_pause", 172, 2, "stream_pause", seconds=T(2.0) if scale < 1 else 2.0,
             mode=stream_pause_mode, flags=["stream_pause"], section="stream",
             features=[Feature.STREAM_PAUSE],
             expect="the pilot stops and comes back; the pads do not notice",
             verify="pilot gap only inside the pause; no pad finding around it"),
        step("B1", "bounce1_one", 190, 30, "pattern", flags=["one", "bounce"],
             features=[Feature.RETRIGGER],
             expect="pad 12 keeps playing over the bounce", verify="as one",
             hits=fit(one(one_rate, T(30), 12, rng), 30)),
        step("B1", "bounce1", 192, 0.1, "bounce", scene="S1", track=0, flags=["bounce"],
             needs=["sysex_bounce"], section="bounce",
             features=[Feature.BOUNCE_SYSEX],
             expect="S1 once from bar 1 into a clip on track 1, then the sequencer stops",
             verify="ASONG bounce=2, clips+1; the file vs the capture (residual); no S1 "
                    "event after it"),
    ]
    blocks.append(b1)

    # ---- B1b: the overdub layer --------------------------------------------
    b1b = Block("B1b", "overdub: warstwa hi-perc na S1 (uzbrojona i zamknięta w oknach CDC)",
                T(BLOCK_S["B1b"]), S.JAM_A, before=["W1a"], after=["W1"])
    hp_hits = [Hit(t=max(0.0, i * beat / 4 + rng.uniform(-0.02, 0.02)), pad=4 + i % 4)
               for i in range(int(8.0 / (beat / 4)))]
    b1b.steps += [
        step("B1b", "stream", 0, BLOCK_S["B1b"], "listen", features=[Feature.STREAM_IN],
             expect="pilot continuous", verify="Tier A"),
        step("B1b", "layer", 4, 8, "pattern", flags=["overdub"], section="build",
             features=[Feature.SEQ_OVERDUB, Feature.QUANT_16],
             expect="hits land on the 1/16 grid of S1 and sound from the next pass on",
             verify="SEQ_DUMP after vs quantise() of each hit (REC_MISS/REC_WRONG_POS); "
                    "the new events heard in later passes",
             hits=fit(hp_hits, 8)),
    ]
    blocks.append(b1b)

    # ---- B2: the performance -----------------------------------------------
    b2 = Block("B2", "występ: song, many, FX, MIDI clock, UI, swap companion, bounce #2",
               T(BLOCK_S["B2"]), S.JAM_A, before=["W1"], after=["W2"])
    b2.steps += [
        step("B2", "stream", 0, BLOCK_S["B2"], "listen",
             features=[Feature.STREAM_IN, Feature.SONG, Feature.DOUBLE, Feature.ERASE,
                       Feature.MIX_PRESET, Feature.RECORDER],
             expect="song S1 x2, S2 (4 bars) x1, looping; open hat erased; JAM-DUCK desk",
             verify="every song pass's events; no p3 (SEQ_EXTRA); level per desk"),
        step("B2", "many", 10, 60, "pattern", flags=["many"],
             features=[Feature.MANY, Feature.POLY_LIMIT],
             expect="16 pads round robin + an 8-pad chord: 8 voices, the oldest stolen",
             verify="STOLEN where the model says, never EARLY_CUT",
             hits=fit(many(many_rate, T(60), chord_every, rng), 60)),
        step("B2", "swap_fx", 72, 0.1, "swap", kit=S.JAM_FX, flags=["swap", "fx_render"],
             features=[Feature.SWAP, Feature.FX_RENDER],
             expect="the FX kit renders its .fx files on the card, then loads",
             verify="no glitch while the card is written"),
        step("B2", "fx_tour", 78, 40, "pattern", flags=["fx"],
             features=[Feature.PITCH, Feature.REVERSE, Feature.LP, Feature.HP, Feature.PAN,
                       Feature.VOLUME, Feature.CHOKE, Feature.HOLD, Feature.LOOP,
                       Feature.LAYERS, Feature.SLICE],
             expect="each FX pad sounds its feature",
             verify="tone after pitch; pan L/R; choke cuts; hold ends at release; loop "
                    "toggles; slice plays only its second tone",
             hits=fit(fx_tour(), 40)),
        step("B2", "swap_fa", 120, 0.1, "swap", kit=S.JAM_A, flags=["swap"],
             features=[Feature.SWAP], expect="back to A", verify="as swaps"),
        step("B2", "clock", 125, 46, "clock", bpm=bpm, flags=["clock"], section="clock",
             plan=[["F8", 0.0], ["FC", 28.0], ["FB", 30.0], ["FC", 34.0], ["FA", 36.0],
                   ["F8_stop", 42.0]],
             features=[Feature.CLOCK_TRANSPORT, Feature.CLOCK_TIMEOUT],
             expect="Stop stops within 20 ms, Continue resumes, Start restarts at bar 1 "
                    "(no burst of the rest of the loop), silence hands the clock back",
             verify="the sequencer's onsets per transport segment (fitted); none while "
                    "stopped (SEQ_EXTRA)"),
        step("B2", "ui", 175, 7, "ui", apps=["Mixer", "Sampler"], flags=["ui"],
             needs=["companion"], section="ui",
             features=[Feature.APP_SWITCH, Feature.COMPANION],
             expect="the Mixer comes up and goes; closing it writes its desk to NVS",
             verify="Tier A clean; pads unverifiable while another app is in front"),
        step("B2", "four_companion", 188, 22, "pattern", flags=["four"],
             features=[Feature.FOUR, Feature.SWAP_COMPANION],
             expect="as four, with swaps through the companion verb",
             verify="as swaps",
             hits=fit(four(four_rate, T(22), [8, 9, 10, 11], rng), 22)),
        step("B2", "swap_c1", 193, 0.1, "swap", kit=S.JAM_B, via="companion",
             flags=["swap"], needs=["companion"], features=[Feature.SWAP_COMPANION],
             expect="A->B through companion KIT_LOAD", verify="as swaps"),
        step("B2", "swap_c2", 203, 0.1, "swap", kit=S.JAM_A, via="companion",
             flags=["swap"], needs=["companion"], features=[Feature.SWAP_COMPANION],
             expect="B->A", verify="as swaps"),
        step("B2", "bounce2_four", 215, 30, "pattern", flags=["four", "bounce"],
             features=[Feature.FOUR], expect="pads 12-15 alternate over the bounce",
             verify="as four",
             hits=fit(four(4.0, T(30), [12, 13, 14, 15], rng), 30)),
        step("B2", "bounce2", 217, 0.1, "bounce", scene="S2", track=1, flags=["bounce"],
             needs=["sysex_bounce"], section="bounce", features=[Feature.BOUNCE_SYSEX],
             expect="S2 (4 bars after DOUBLE) once into track 2", verify="as bounce #1"),
    ]
    blocks.append(b2)

    # ---- B3: pitched -------------------------------------------------------
    b3 = Block("B3", "pitched: S3 w triolach, przełączenie silnika, melodia, akordy, bounce #4",
               T(BLOCK_S["B3"]), S.JAM_A, before=["W2"], after=["FINAL"])
    b3.steps += [
        step("B3", "stream", 0, BLOCK_S["B3"], "listen",
             features=[Feature.STREAM_IN, Feature.QUANT_8T, Feature.BOUNCE_CDC],
             expect="S3 (1/8T) plays; JAM-FULL desk", verify="S3 events on the triplet grid"),
        step("B3", "swap_ap", 12, 0.1, "swap", kit=S.JAM_P, flags=["swap", "engine_switch"],
             features=[Feature.ENGINE_SWITCH],
             expect="the sampler's voices stop; the scene's notes become pitched blips",
             verify="no old-kit tone after the switch window (GHOST)"),
        step("B3", "legato", 20, 55, "pattern", flags=["pitched"],
             features=[Feature.PITCHED],
             expect="notes from pad_notes, each ending 100 ms after its release",
             verify="onset on time; the end where the release puts it",
             hits=fit(legato(T(55), 3.0, rng), 55)),
        step("B3", "chords", 80, 26, "pattern", flags=["pitched", "chords"],
             features=[Feature.PITCHED_STEAL],
             expect="12-note chords: cap 10, two voices stolen each (5 ms fade)",
             verify="the stolen ones end at the steal; shed stays 0",
             hits=fit(chords(12, 2.0, 12, 700.0, rng), 26)),
        step("B3", "retrig", 108, 4, "pattern", flags=["pitched"],
             features=[Feature.PITCHED_RETRIG],
             expect="the same note again releases the previous voice",
             verify="each restart found",
             hits=fit([Hit(t=0.4 * i, pad=7, hold_ms=300.0) for i in range(8)], 4)),
        step("B3", "swap_pa", 115, 0.1, "swap", kit=S.JAM_A, flags=["swap", "engine_switch"],
             features=[Feature.ENGINE_SWITCH], expect="back to the sampler", verify="as swaps"),
        step("B3", "four_a", 118, 18, "pattern", flags=["four"], features=[Feature.FOUR],
             expect="as four", verify="as four",
             hits=fit(four(four_rate, T(18), [8, 9, 10, 11], rng), 18)),
        step("B3", "swap_ap2", 138, 0.1, "swap", kit=S.JAM_P, flags=["swap", "engine_switch"],
             features=[Feature.ENGINE_SWITCH], expect="pitched again", verify="as swaps"),
        step("B3", "bounce4_chords", 145, 40, "pattern", flags=["pitched", "bounce"],
             features=[Feature.PITCHED_STEAL], expect="chords over the bounce",
             verify="as chords",
             hits=fit(chords(12, 3.0, 12, 900.0, rng), 40)),
        step("B3", "bounce4", 150, 0.1, "bounce", scene="S3", track=3, flags=["bounce"],
             needs=["sysex_bounce"], section="bounce", features=[Feature.BOUNCE_SYSEX],
             expect="S3 (1 bar, triplets) once into track 4", verify="as bounce #1"),
    ]
    blocks.append(b3)

    # ---- SOAK: one long block ----------------------------------------------
    if soak_min > 0:
        dur = soak_min * 60.0
        soak = Block("SOAK", f"soak: {soak_min:g} min ciągłego UAC (znany underrun po 5-6 min)",
                     dur, S.JAM_A, before=["W3"], after=["FINAL"])
        soak.steps.append(Step(id="SOAK.stream", block="SOAK", t_rel=0.0, dur=dur,
                               action="listen", features=[Feature.SOAK.value,
                                                          Feature.STREAM_IN.value],
                               expect="nothing breaks in a long session",
                               verify="Tier A over the whole block"))
        t = 10.0
        k = 0
        while t + 40 < dur:
            hits = four(6.0, 30, [8, 9, 10, 11], rng) if k % 3 else many(12.0, 30, 6.0, rng)
            soak.steps.append(Step(id=f"SOAK.p{k}", block="SOAK", t_rel=t, dur=30.0,
                                   action="pattern", flags=["soak", "many" if not k % 3
                                                            else "four"],
                                   features=[Feature.SOAK.value], hits=hits,
                                   expect="as four/many", verify="as four/many"))
            if k % 3 == 1:
                soak.steps.append(Step(id=f"SOAK.swap{k}", block="SOAK", t_rel=t + 15,
                                       dur=0.1, action="swap",
                                       args={"kit": S.JAM_B if (k // 3) % 2 == 0
                                             else S.JAM_A}, flags=["swap"],
                                       features=[Feature.SWAP.value]))
            t += 45.0
            k += 1
        blocks.append(soak)

    else:
        skipped.append({"step": "SOAK", "why": "--soak-min 0: no long block (the known "
                        "underrun shows after 5-6 min of one UAC session)",
                        "features": Feature.SOAK.value})

    order = ["PREP", "W0", "B1", "W1a", "B1b", "W1", "B2", "W2", "B3"]
    if soak_min > 0:
        order += ["W3", "SOAK"]
    order += ["FINAL"]

    windows: dict[str, list[WindowOp]] = {
        "PREP": [WindowOp("normalise: default profile, pins journal, card hold, hub check"),
                 WindowOp("console (reset) + boot + SD mode"),
                 WindowOp("home snapshot (SEQ, SONG, CLICK, MIX, route, KIT, APP, ASONG, "
                          "caps; sequences.json + desk slots fetched) into the pins journal "
                          "before any change"),
                 WindowOp("ensure_kits (JAM A/B/FX/P); delete JAM_FX/.fx",
                          [Feature.FX_RENDER.value]),
                 WindowOp("probe capabilities (companion SEQ?, bounce, note-out, companion)"),
                 WindowOp("desks: HILHOME saved to slot 6; JAM-DUCK/-FULL built from it "
                          "(USB->rec on, Click/Song/IN off, no solo, Out 1/2 muted unless "
                          "--audible) into slots 4/5: USB->rec on in both",
                          [Feature.MIX_PRESET.value, Feature.MIX_ROUTE.value]),
                 WindowOp("ASONG OPEN hil_live_jam (bounces never land in your project)")],
        "W0": [WindowOp("S1: ADD, SCENE, BARS S1 2, BPM, QUANT 2, COUNTIN 1, REC (guarded: the "
                        "active scene is the jam's), base pattern by PAD_PRESS with a hit a "
                        "32nd early for beat one, PLAY",
                        [Feature.SEQ_REC.value, Feature.PADS_CDC.value]),
               WindowOp("CLICK anchor, kit JAM A, Sampler app")],
        "W1a": [WindowOp("readbacks; FS_GET bounce #1; SEQ PLAY; SEQ OVERDUB on S1 (guarded)",
                         [Feature.SEQ_OVERDUB.value])],
        "W1": [WindowOp("SEQ PLAY (closes the overdub); SEQ_DUMP (the layer)"),
               WindowOp("S2: ADD, SCENE, BARS 2, REC groove on 8-11, PLAY, DOUBLE -> 4 bars",
                        [Feature.DOUBLE.value]),
               WindowOp("SEQ SCENE S1, ERASE 1, PAD_PRESS 3, ERASE 0 (the open hat out)",
                        [Feature.ERASE.value]),
               WindowOp("SONG CLEAR, ADD S1 2, ADD S2 1, LOOP 1, PLAY", [Feature.SONG.value]),
               WindowOp("MIX_PRESET LOAD 4 (JAM-DUCK)", [Feature.MIX_PRESET.value]),
               WindowOp("Recorder: STOOL IN PADS, REC (runs through B2)",
                        [Feature.RECORDER.value])],
        "W2": [WindowOp("readbacks; FS_GET bounce #2; Recorder STOP + windows"),
               WindowOp("bounce #3 over CDC: ASONG BOUNCE S1 2 while CDC pads play",
                        [Feature.BOUNCE_CDC.value, Feature.PADS_CDC.value]),
               WindowOp("S3: ADD, BARS S3 1, QUANT 4 (1/8T), REC pads 0-3 on triplets, PLAY",
                        [Feature.QUANT_8T.value]),
               WindowOp("SONG CLEAR; SEQ SCENE S3; MIX_PRESET LOAD 5 (JAM-FULL)")],
        "FINAL": [WindowOp("readbacks, FS_GET bounce #4, ASONG REPORT, RESET_LOG, MEM, WORKER, "
                           "PITCHED_STATUS, ARENA"),
                  WindowOp("put back: SEQ/ASONG stop, desk (LOAD 6) + slots 4-6, route, kit by "
                           "name, app, project, sequences.json (FS_PUT, after dirty=0), "
                           "reboot, board_state diff")],
    }
    if soak_min > 0:
        windows["W3"] = [WindowOp("readbacks; SONG CLEAR; SEQ SCENE S1; PLAY; anchor")]
    return JamScript(seed=seed, scale=scale, bpm=bpm, blocks=blocks, windows=windows,
                     order=order, skipped=skipped)


def all_features() -> list[str]:
    return [f.value for f in Feature]


def coverage(script: JamScript) -> dict[str, dict[str, Any]]:
    """{feature: {steps: [...], skipped: [...]}} -- what the script means to exercise."""
    out: dict[str, dict[str, Any]] = {f: {"steps": [], "skipped": []} for f in all_features()}
    for b in script.blocks:
        for s in b.steps:
            for f in s.features:
                out.setdefault(f, {"steps": [], "skipped": []})
                (out[f]["skipped"] if s.skip else out[f]["steps"]).append(s.id)
    for name, ops in script.windows.items():
        for op in ops:
            for f in op.features:
                out.setdefault(f, {"steps": [], "skipped": []})
                (out[f]["skipped"] if op.skip else out[f]["steps"]).append(f"{name}:{op.op[:30]}")
    for d in script.skipped:
        for f in str(d.get("features", "")).split(","):
            if f and d["step"] not in out.setdefault(f, {"steps": [], "skipped": []})["skipped"]:
                out[f]["skipped"].append(d["step"])
    return out


def script_table(script: JamScript) -> str:
    """The set as a table (``--dry-run``)."""
    lines = [f"live_jam script seed={script.seed} bpm={script.bpm} scale={script.scale:.3f} "
             f"UAC={script.uac_seconds / 60:.1f} min", ""]
    for name in script.order:
        if name in script.windows:
            lines.append(f"[{name}] CDC")
            for op in script.windows[name]:
                lines.append(f"    - {op.op}" + (f"  (SKIP: {op.skip})" if op.skip else ""))
            continue
        b = script.block(name)
        lines.append(f"[{name}] UAC {b.duration:.1f} s -- {b.title}")
        for s in b.steps:
            hits = f" hits={len(s.hits)}" if s.hits else ""
            skip = f"  SKIP: {s.skip}" if s.skip else ""
            lines.append(f"    {s.t_rel:7.2f}s +{s.dur:6.2f}s {s.id:<22} {s.action:<12}"
                         f"{hits}{skip}")
            if s.expect:
                lines.append(f"{'':12}expect: {s.expect}")
            if s.verify:
                lines.append(f"{'':12}verify: {s.verify}")
    if script.skipped:
        lines += ["", "skipped:"] + [f"    {d['step']}: {d['why']}" for d in script.skipped]
    return "\n".join(lines)
