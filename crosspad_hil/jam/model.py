"""What one block of the jam should sound like, voice by voice, in frames of its take.

Everything the board was asked to do in a block -- the pads the host struck,
the sequencer's passes, kit swaps, bounces, transport, the phone's stream
pausing -- becomes a list of ``Voice``s: which band (kit, pad), when it starts,
when and why it ends, how loud, and whether the model can vouch for it at all
(``uncertain``). The detectors compare a take against exactly this.

Three ways an event lands in the take's frames:

* a host hit: ``frame(t_send) + L_hit`` (the capture clock maps host time to
  frames; ``L_hit`` per path, seeded from the calibration and re-measured);
* a sequencer event: ``pass_start + pos`` -- the pass start comes from the
  CLICK anchor read just before the block and is re-fitted to the take
  (``SeqProbe``), so the device's own sample clock times the grid;
* an event whose moment the host cannot know (a bounce's first bar, a MIDI
  Start, an app switch): a window ``[t_req, t_req + W]`` the fit searches,
  and everything inside it is ``uncertain`` until fitted.

The voice engines are copies of the firmware's rules, not of its code: the
sampler's 16-channel budget (8 stereo voices, the one furthest into its
sample is stolen), one voice per pad (a hit restarts it), choke groups, hold
and loop pads; the pitched engine's cap of 10 (a releasing voice is stolen
first, then the oldest held; 5 ms fade), and a release that ends a held note.
Where the firmware's choice is not visible from outside (a steal tie inside one
render block) the model says so rather than guessing.
"""
from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from typing import Any

from crosspad_hil.jam import spec as S
from crosspad_hil.jam.actions import Action
from crosspad_hil.jam.calibration import Calibration

BLOCK = 64                     # frames the engine renders at a time
HF_SPLATTER_HZ = 4_000.0       # a voice this high: its fades reach the quiet bands
SAMPLER_UNITS = 16             # WAV channels the sampler mixes (stereo voice = 2)
PITCHED_CAP = S.P_MAX_VOICES
MAX_EVENTS = 192               # PatternSequencer::kMaxEvents
BEATS_PER_BAR = 4
INF = float("inf")

# Quantization enum (SequencerTypes.hpp) -> grid lines per beat; 127 = off.
GRID_PER_BEAT = {0: 1, 1: 2, 2: 4, 3: 8, 4: 3, 5: 6, 127: 0}
QUANT_NAME = {0: "1/4", 1: "1/8", 2: "1/16", 3: "1/32", 4: "1/8T", 5: "1/16T", 127: "off"}
UNQUANTISED_GRACE_DIV = 8


# --------------------------------------------------------------------------- the grid

def grid_per_beat(quant: int) -> int:
    return GRID_PER_BEAT.get(int(quant), 0)


def samples_per_beat(bpm: float, sr: int) -> int:
    return int(round(sr * 60.0 / float(bpm)))


def quantise(pos: int, spb: int, per: int, loop: int) -> int:
    """PatternSequencer::quantise, integer for integer: the nearest line k, placed at
    k*spb/per (not k steps of spb/per), and a hit rounding up to the loop end is 0."""
    if not per or spb < 2:
        return int(pos)
    k = (int(pos) * per * 2 + spb) // (spb * 2)
    snapped = (k * spb) // per
    return 0 if (loop and snapped >= loop) else int(snapped)


def downbeat_grace(spb: int, per: int) -> int:
    """How early a count-in hit may be and still count as beat one."""
    if per:
        return (spb // per) // 2
    return spb // UNQUANTISED_GRACE_DIV


# --------------------------------------------------------------------------- outputs

@dataclass
class Voice:
    vid: int
    src: str                       # host | seq | preview
    cause: str                     # "a<aid>" | "seq:<scene>:<i>:<pass>"
    aid: int | None
    kit: str
    pad: int
    engine: str                    # sampler | pitched
    freqs: tuple[float, ...]
    f_on: float
    f_nat: float
    f_end: float
    end_kind: str = "natural"
    gain_l: float = 0.0
    gain_r: float = 0.0
    level_known: bool = True
    shape: str = "flat"
    fade_on_s: float = S.FADE_S
    restart: bool = False          # starts while the same band still sounds (a dip, not a rise)
    uncertain: str = ""            # non-empty: the take cannot be held against this voice
    end_uncertain: str = ""
    tol_on: float = 0.0            # frames of allowed onset uncertainty beyond the usual
    late_ok_ms: float = 0.0        # a swap may hold this hit back up to this long
    steal_cands: tuple[int, ...] = ()
    f_rel: float = INF             # when a held voice was released
    release_s: float = 0.0
    absent: tuple[float, ...] = ()
    seq: dict[str, Any] | None = None
    choke: int = 0
    alt_freqs: tuple[float, ...] = ()   # another kit's tones that are also right (untimed swap)
    held_by: int | None = None          # the swap (action id) whose reload held this hit
    feature: str = "plain"              # the kit feature this pad shows (spec.SamplerPadSpec)

    @property
    def amp(self) -> float:
        return max(self.gain_l, self.gain_r)

    def half_end(self, sr: int) -> float:
        """Where the voice's level falls through half of its steady level."""
        if self.end_kind in ("released", "retrigger_release") and self.release_s:
            return min(self.f_end, self.f_rel + 0.5 * self.release_s * sr)
        if self.end_kind == "natural" and self.shape != "rise":
            return self.f_end - 0.5 * S.FADE_S * sr
        return self.f_end

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        for k in ("f_nat", "f_end", "f_rel"):
            if d[k] == INF:
                d[k] = None
        return d


@dataclass
class Edge:
    frame: float
    kind: str
    vid: int | None = None
    by: int | None = None           # the voice whose start caused it
    tol_ms: float = 2.5             # how precisely the model knows where it is


@dataclass
class Window:
    f0: float
    f1: float
    kind: str
    why: str = ""
    ref: int | None = None          # an action id

    def covers(self, f: float, pad: float = 0.0) -> bool:
        return self.f0 - pad <= f <= self.f1 + pad


@dataclass
class Expectation:
    sr: int
    n: int
    voices: list[Voice] = field(default_factory=list)
    edges: list[Edge] = field(default_factory=list)
    uncertain: list[Window] = field(default_factory=list)
    pilot_off: list[Window] = field(default_factory=list)
    flags: list[Window] = field(default_factory=list)
    seq_events: list[dict[str, Any]] = field(default_factory=list)
    rec_events: list[dict[str, Any]] = field(default_factory=list)
    runs: list[dict[str, Any]] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)
    poly: list[tuple[float, int]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {"sr": self.sr, "n": self.n, "voices": [v.to_dict() for v in self.voices],
                "edges": [asdict(e) for e in self.edges],
                "uncertain": [asdict(w) for w in self.uncertain],
                "pilot_off": [asdict(w) for w in self.pilot_off],
                "flags": [asdict(w) for w in self.flags], "seq_events": self.seq_events,
                "rec_events": self.rec_events, "runs": self.runs, "notes": self.notes}

    def flags_at(self, f: float) -> list[str]:
        return sorted({w.kind for w in self.flags if w.covers(f)})

    def uncertain_at(self, f: float, pad: float = 0.0) -> Window | None:
        for w in self.uncertain:
            if w.covers(f, pad):
                return w
        return None


# --------------------------------------------------------------------------- the block's inputs

@dataclass
class BlockInput:
    """What the analysis knows about one block, all already on the capture's frames."""

    name: str
    sr: int
    n: int
    frame_of: Callable[[float], float]          # host monotonic -> capture frame
    actions: list[Action]
    kits: dict[str, S.KitSpec]
    kit0: str                                   # kit loaded when the block began
    seq_before: dict[str, Any] = field(default_factory=dict)
    seq_after: dict[str, Any] = field(default_factory=dict)
    dump_before: list[dict[str, int]] = field(default_factory=list)
    dump_after: list[dict[str, int]] = field(default_factory=list)
    song: dict[str, Any] = field(default_factory=dict)
    click: dict[str, Any] = field(default_factory=dict)     # with "t_host"
    swap_rows: list[dict[str, Any]] = field(default_factory=list)
    desk: dict[str, Any] = field(default_factory=dict)     # MixModel.from_desk input
    stream_events: list[dict[str, Any]] = field(default_factory=list)
    t_stream_start: float | None = None
    t_stream_stop: float | None = None
    fx_first_load: bool = False


class SeqProbe:
    """How the model asks the take where the sequencer actually was.

    ``fit(freqs, rel_frames, lo, hi) -> offset | None``: the frame at which a
    pattern of onsets (``rel_frames`` from its start, one band each) best
    lines up with the take, searched in ``[lo, hi]``. The default knows
    nothing and keeps the prediction (a model-only build, as in tests).
    """

    def fit(self, freqs: list[tuple[float, ...]], rel: list[float], lo: float,
            hi: float, prior: float | None = None) -> float | None:
        return prior


# --------------------------------------------------------------------------- the mixer

@dataclass
class ChannelGain:
    vol: float = 1.0
    pan: float = 0.0
    muted: bool = False
    rec: bool = True
    rec_vol: float = 1.0

    def lr(self, out_vol: float) -> tuple[float, float]:
        if self.muted or not self.rec:
            return 0.0, 0.0
        g = self.vol * self.rec_vol * out_vol
        # The mixer's balance is linear: 1-pan on the far side.
        return g * (1.0 - max(0.0, self.pan)), g * (1.0 + min(0.0, self.pan))


@dataclass
class MixModel:
    """Gains on the recording bus ("USB rec", CAPTURE_BUS 2) per channel."""

    channels: dict[str, ChannelGain] = field(default_factory=lambda: {
        "Sampler": ChannelGain(), "Pitched": ChannelGain(), "USB": ChannelGain(),
        "Preview": ChannelGain(rec=False), "Song": ChannelGain(rec=False),
        "Click": ChannelGain(rec=False)})
    out_vol: float = 1.0
    rec_out: int = 2

    @staticmethod
    def from_desk(desk: dict[str, Any] | None, rec_out: int = 2) -> MixModel:
        """From a desk JSON (``{"channels": [...], "outputs": [...]}``) or a MIX_LVL
        reply (``{"channels": [{name, vol, mute, route}], "outputs": [...]}``)."""
        m = MixModel(rec_out=rec_out)
        if not desk:
            return m
        for c in desk.get("channels", []):
            name = str(c.get("name"))
            g = ChannelGain()
            g.vol = float(c.get("volume", c.get("vol", 1.0)))
            g.pan = float(c.get("pan", 0.0))
            g.muted = bool(c.get("muted", c.get("mute", False)))
            routes = c.get("routes")
            if isinstance(routes, list) and routes and isinstance(routes[0], dict):
                r = next((r for r in routes if int(r.get("out", -1)) == rec_out), None)
                g.rec = bool(r and r.get("enabled"))
                g.rec_vol = float(r.get("volume", 1.0)) if r else 1.0
            elif isinstance(c.get("route"), list):
                bits = c["route"]
                g.rec = bool(bits[rec_out]) if len(bits) > rec_out else False
            m.channels[name] = g
        outs = desk.get("outputs") or []
        for i, o in enumerate(outs):
            idx = int(o.get("out", i))
            if idx == rec_out:
                m.out_vol = float(o.get("volume", o.get("vol", 1.0)))
        return m

    def gains(self, channel: str) -> tuple[float, float]:
        g = self.channels.get(channel)
        return (0.0, 0.0) if g is None else g.lr(self.out_vol)


# --------------------------------------------------------------------------- the sequencer

@dataclass
class Run:
    """The sequencer playing from ``f_anchor`` (a pass start) until ``f_stop``."""

    f_from: float
    f_stop: float
    f_anchor: float
    schedule: list[tuple[int, int]]      # [(scene, repeats)] in order; repeats 0 = forever
    loop: bool = True
    first_block: int = 0
    first_pass: int = 0
    kind: str = "loop"                   # loop | bounce | resume | start
    fit_lo: float | None = None
    fit_hi: float | None = None
    fitted: bool = False
    pos0: int = 0                        # the first pass starts this far into the scene


class TransportModel:
    """The sequencer's passes in a block: scenes, loop lengths, song order."""

    def __init__(self, sr: int, bpm: float, bars: list[int], quant: int) -> None:
        self.sr = sr
        self.bpm = float(bpm)
        self.bars = list(bars)
        self.quant = int(quant)
        self.spb = samples_per_beat(self.bpm, sr)

    def loop_len(self, scene: int) -> int:
        b = self.bars[scene] if 0 <= scene < len(self.bars) and self.bars[scene] else 1
        return int(b) * BEATS_PER_BAR * self.spb

    def passes(self, run: Run, f_until: float) -> list[dict[str, Any]]:
        """``[{scene, f0, length, k, block, rep}]`` from the run's anchor to ``f_until``."""
        out: list[dict[str, Any]] = []
        sched = run.schedule or [(0, 0)]
        bi, rep = run.first_block % len(sched), run.first_pass
        f = run.f_anchor - run.pos0
        k = 0
        guard = 0
        while f < min(f_until, run.f_stop) and guard < 100_000:
            guard += 1
            scene, reps = sched[bi]
            length = self.loop_len(scene)
            out.append({"scene": scene, "f0": f, "length": length, "k": k, "block": bi,
                        "rep": rep})
            f += length
            k += 1
            if run.kind == "bounce":
                break
            if reps and rep + 1 >= reps:
                bi += 1
                rep = 0
                if bi >= len(sched):
                    if not run.loop:
                        break
                    bi = 0
            else:
                rep += 1
        return out


def _dump_by_scene(dump: list[dict[str, int]]) -> dict[int, list[dict[str, int]]]:
    out: dict[int, list[dict[str, int]]] = {}
    for e in dump:
        out.setdefault(int(e["seq"]), []).append(dict(e))
    for v in out.values():
        v.sort(key=lambda e: (e["pos"], e["i"]))
    return out


def parse_scene_field(value: Any) -> tuple[int, int]:
    """``scene=A>P`` -> (sounding, queued)."""
    s = str(value)
    a, _, b = s.partition(">")
    try:
        return int(a), int(b or a)
    except ValueError:
        return 0, 0


# --------------------------------------------------------------------------- the engines

class SamplerVoiceSim:
    """The sampler's mixing rules: a 16-channel budget, one voice per pad."""

    def __init__(self, sr: int, cal: Calibration, units: int = SAMPLER_UNITS) -> None:
        self.sr = sr
        self.cal = cal
        self.units = units
        self.active: list[Voice] = []

    def _retire(self, f: float) -> None:
        self.active = [v for v in self.active if v.f_end > f]

    def _cut(self, v: Voice, f: float, kind: str) -> None:
        if v.f_end > f:
            v.f_end = f + (0.005 * self.sr if kind == "stolen" and self.cal.steal == "fade"
                           else 0.0)
            v.end_kind = kind

    def stop_all(self, f: float, kind: str, uncertain_s: float = 0.0) -> list[Voice]:
        self._retire(f)
        cut = list(self.active)
        for v in cut:
            if uncertain_s:
                v.end_uncertain = f"{kind}: ends somewhere in {uncertain_s:.1f} s"
                v.f_end = min(v.f_end, f + uncertain_s * self.sr)
                v.end_kind = kind
            else:
                self._cut(v, f, kind)
        self.active = [] if not uncertain_s else self.active
        return cut

    def release(self, f: float, pad: int, kit: str) -> None:
        for v in self.active:
            if v.pad == pad and v.shape == "hold" and v.f_rel == INF:
                v.f_rel = f
                rel = self.cal.hold_release_ms / 1000.0
                end = f + rel * self.sr
                if end < v.f_end:
                    v.f_end = end
                    v.end_kind = "released"
                    v.release_s = rel

    def press(self, v: Voice) -> list[tuple[str, Voice]]:
        """Start ``v`` (its f_on); returns ``[(why, other voice)]`` it ended."""
        f = v.f_on
        self._retire(f)
        ended: list[tuple[str, Voice]] = []
        same = [o for o in self.active if o.pad == v.pad]
        heard_mode = v.shape
        if heard_mode == "loop" and same:
            # A loop pad toggles: the press that finds it playing stops it.
            for o in same:
                self._cut(o, f, "loop_toggle")
                ended.append(("loop_toggle", o))
            self._retire(f)
            v.uncertain = v.uncertain or ""
            v.f_end = f            # no new voice: marks the press as a stop
            v.end_kind = "toggle_off"
            return ended
        for o in same:
            self._cut(o, f, "retrigger")
            ended.append(("retrigger", o))
            v.restart = True
        self._retire(f)
        if v.choke:
            for o in list(self.active):
                if o.pad != v.pad and o.choke == v.choke:
                    self._cut(o, f, "choked")
                    ended.append(("choked", o))
            self._retire(f)
        if 2 * len(self.active) + 2 > self.units and self.active:
            # Over the budget: the voice furthest into its sample goes. Voices started
            # inside one render block are a tie the engine settles; the model marks
            # them all and leaves their ends open.
            oldest = min(o.f_on for o in self.active)
            ties = [o for o in self.active if o.f_on - oldest <= BLOCK]
            victim = min(ties, key=lambda o: (o.f_on, o.vid))
            if len(ties) > 1:
                for o in ties:
                    o.end_uncertain = o.end_uncertain or "steal tie inside one render block"
                    o.steal_cands = tuple(t.vid for t in ties)
            self._cut(victim, f, "stolen")
            ended.append(("stolen", victim))
            self._retire(f)
        self.active.append(v)
        return ended


class PitchedVoiceSim:
    """The pitched engine: a cap of 10 sounding voices (held or releasing)."""

    def __init__(self, sr: int, cal: Calibration, cap: int = PITCHED_CAP) -> None:
        self.sr = sr
        self.cal = cal
        self.cap = cap
        self.active: list[Voice] = []

    def _retire(self, f: float) -> None:
        self.active = [v for v in self.active if v.f_end > f]

    def stop_all(self, f: float, kind: str, uncertain_s: float = 0.0) -> list[Voice]:
        self._retire(f)
        cut = list(self.active)
        for v in cut:
            if uncertain_s:
                v.end_uncertain = f"{kind}: ends somewhere in {uncertain_s:.1f} s"
                v.f_end = min(v.f_end, f + uncertain_s * self.sr)
            else:
                v.f_end = min(v.f_end, f)
            v.end_kind = kind
        if not uncertain_s:
            self.active = []
        return cut

    def _release(self, v: Voice, f: float, kind: str = "released") -> None:
        rel = self.cal.pitched_release_ms / 1000.0
        if f < v.f_rel:
            v.f_rel = f
            end = f + rel * self.sr
            if end < v.f_end:
                v.f_end = end
                v.end_kind = kind
                v.release_s = rel

    def release(self, f: float, pad: int, kit: str) -> None:
        for v in self.active:
            if v.pad == pad and v.f_rel == INF:
                self._release(v, f)

    def press(self, v: Voice, hold_s: float) -> list[tuple[str, Voice]]:
        f = v.f_on
        self._retire(f)
        ended: list[tuple[str, Voice]] = []
        for o in self.active:
            if o.pad == v.pad and o.f_rel > f:
                self._release(o, f, "retrigger_release")
                ended.append(("retrigger_release", o))
        if any(o.pad == v.pad for o in self.active):
            v.restart = True
            v.uncertain = v.uncertain or "a pitched retrigger overlaps its own release"
        while len(self.active) + 1 > self.cap:
            releasing = [o for o in self.active if o.f_rel <= f]
            victim = (min(releasing, key=lambda o: (o.f_rel, o.vid)) if releasing
                      else min(self.active, key=lambda o: (o.f_on, o.vid)))
            victim.f_end = min(victim.f_end, f + self.cal.pitched_steal_fade_ms / 1000.0 * self.sr)
            victim.end_kind = "stolen"
            ended.append(("stolen", victim))
            self.active.remove(victim)
        if hold_s > 0:
            self._release(v, f + hold_s * self.sr)
        self.active.append(v)
        return ended


# --------------------------------------------------------------------------- swaps

@dataclass
class Swap:
    aid: int | None
    f_req: float
    old: str
    new: str
    row: dict[str, Any] | None
    max_load_s: float

    def pad_window(self, pad: int, sr: int) -> tuple[float, float, bool]:
        """(reload start, reload end, timed) of ``pad``, frames. Untimed: the whole
        window of a load the log did not describe."""
        if not self.row or not isinstance(self.row.get("pads"), list):
            return self.f_req, self.f_req + self.max_load_s * sr, False
        done = [int(d) for d in self.row["pads"]]
        if pad >= len(done):
            return self.f_req, self.f_req + self.max_load_s * sr, False
        parse = float(self.row.get("parse_ms") or 0.0)
        end = done[pad]
        if end < 0:                               # untouched: switched once parsed
            t = self.f_req + parse / 1000.0 * sr
            return t, t, True
        ends = sorted(d for d in done if d >= 0)
        start = max([parse] + [e for e in ends if e < end])
        deferred = int(self.row.get("deferred") or 0)
        if deferred >> pad & 1:
            spans = [done[p] - max([parse] + [e for e in ends if e < done[p]])
                     for p in range(len(done)) if done[p] >= 0 and not deferred >> p & 1]
            start = max(start, end - (max(spans) if spans else 0.0))
        return self.f_req + start / 1000.0 * sr, self.f_req + end / 1000.0 * sr, True


class SwapModel:
    def __init__(self, kit0: str, sr: int) -> None:
        self.kit0 = kit0
        self.sr = sr
        self.swaps: list[Swap] = []

    def add(self, s: Swap) -> None:
        self.swaps.append(s)

    def kit_for(self, pad: int, f: float) -> tuple[str, float, str, str | None, int | None]:
        """(kit, onset frame -- later when a reload held the hit --, why uncertain,
        the other kit that is equally right inside an untimed swap, the swap that
        held the hit)."""
        current = self.kit0
        for s in self.swaps:
            if s.f_req > f:
                break
            a, b, timed = s.pad_window(pad, self.sr)
            if not timed:
                if f <= b:
                    return s.new, f, "", current, None
                current = s.new
                continue
            if f < a:
                return current, f, "", None, None
            if f < b:
                return s.new, b, "", None, s.aid
            current = s.new
        return current, f, "", None, None

    def last_kit(self) -> str:
        return self.swaps[-1].new if self.swaps else self.kit0


# --------------------------------------------------------------------------- building it

HOST_SOURCES = {"pad": "sysex", "pad_companion": "companion", "pad_cdc": "cdc",
                "vpad": "vpad", "preview": "preview"}


def _ms(cal: Calibration, src: str) -> float:
    return float(cal.l_hit_ms.get(src, cal.l_hit_ms.get("sysex", 8.0)))


def build_expectation(b: BlockInput, cal: Calibration, probe: SeqProbe | None = None,
                      observed: dict[str, float] | None = None,
                      restarts: set[str] | None = None,
                      shift_ms: dict[str, float] | None = None,
                      dropped: set[str] | None = None) -> Expectation:
    """The block's expected voices. ``observed`` ({cause: onset frame}) replaces the
    predicted start of every voice the take has already timed: rebuilt that way
    the engines steal, cut and end exactly where they did, so what follows each
    start (holes, ends, clicks) is judged against what actually started when."""
    probe = probe or SeqProbe()
    observed = observed or {}
    restarts = restarts or set()
    shift_ms = shift_ms or {}
    dropped = dropped or set()
    sr = b.sr
    exp = Expectation(sr=sr, n=b.n)
    fr = b.frame_of
    mix = MixModel.from_desk(b.desk)
    acts = sorted((a for a in b.actions if a.ok), key=lambda a: a.t_mono)

    # ---- swaps, from the kit actions and the board's own log of them
    swaps = SwapModel(b.kit0, sr)
    kit_now = b.kit0
    rows = list(b.swap_rows)
    loaded_fx: set[str] = set()
    for a in acts:
        if a.kind != "kit":
            continue
        new = str(a.args.get("name"))
        f_req = fr(a.t_mono)
        row = rows.pop(0) if rows else None
        old_engine = b.kits[kit_now].engine if kit_now in b.kits else "sampler"
        new_engine = b.kits[new].engine if new in b.kits else "sampler"
        max_load = cal.swap_max_load_ms / 1000.0
        if new_engine != old_engine:
            max_load = cal.engine_switch_ms / 1000.0
        if new == S.JAM_FX and b.fx_first_load and new not in loaded_fx:
            max_load = cal.fx_render_ms / 1000.0
            exp.flags.append(Window(f_req, f_req + max_load * sr, "fx_render", "first load "
                                    "of the FX kit: rendered to .fx/ on the card", a.aid))
            loaded_fx.add(new)
        swaps.add(Swap(aid=a.aid, f_req=f_req, old=kit_now, new=new, row=row,
                       max_load_s=max_load))
        end = f_req + (max(0, *(row.get("pads") or [0])) / 1000.0 * sr if row
                       and row.get("pads") else max_load * sr)
        exp.flags.append(Window(f_req, end, "swap", f"{kit_now}->{new}", a.aid))
        if new_engine != old_engine:
            w_end = f_req + cal.engine_switch_ms / 1000.0 * sr
            exp.flags.append(Window(f_req, w_end, "engine_switch",
                                    f"{old_engine}->{new_engine}", a.aid))
            # the old engine's voices stop somewhere in here: each may end with a cut
            exp.uncertain.append(Window(f_req, w_end, "engine_switch",
                                        f"{old_engine} voices stop when the {new_engine} "
                                        "engine takes over", a.aid))
        kit_now = new

    # ---- the sequencer: runs of passes, then the events in them
    seq0 = b.seq_before or {}
    bars = list(seq0.get("bars") or [1] * 8)
    bpm = float(b.click.get("bpm") or seq0.get("bpm") or 120.0)
    tm = TransportModel(sr, bpm, bars, int(seq0.get("quant", 2)))
    runs = _runs(b, tm, acts, cal, exp)
    events_by_scene = _dump_by_scene(b.dump_before)
    fresh = _overdub(b, tm, acts, cal, exp, runs, events_by_scene)
    seq_triggers = _seq_triggers(b, tm, runs, events_by_scene, fresh, cal, probe, exp, swaps)

    # ---- everything that starts or stops a voice, in frame order
    trig: list[tuple[float, int, str, dict[str, Any]]] = []
    order = 0
    keys: dict[int, float] = {}

    def key(f_pred: float, cause: str, src: str) -> float:
        """Sort key: the send order (the prediction, shifted by the path's measured
        latency) unless the take timed the voice somewhere else."""
        corrected = f_pred + shift_ms.get(src, 0.0) / 1000.0 * sr
        f_obs = observed.get(cause)
        if f_obs is None or abs(f_obs - corrected) < 0.003 * sr:
            return corrected
        return f_obs
    for a in acts:
        src = HOST_SOURCES.get(a.kind)
        if src is None or f"a{a.aid}" in dropped:
            continue
        lat = _ms(cal, a.args.get("via", src) if a.kind == "pad" else src) / 1000.0 * sr
        f_pred = fr(a.t_mono) + lat
        f = observed.get(f"a{a.aid}", f_pred)
        order += 1
        keys[order] = key(f_pred, f"a{a.aid}", "preview" if a.kind == "preview" else "host")
        trig.append((f, order, a.kind, {"action": a}))
        hold = float(a.args.get("hold_ms") or 0.0)
        if a.kind != "preview" and hold > 0:
            order += 1
            keys[order] = keys[order - 1] + hold / 1000.0 * sr
            trig.append((f + hold / 1000.0 * sr, order, "release", {"action": a}))
    for f, payload in seq_triggers:
        if payload["cause"] in dropped:
            continue
        order += 1
        keys[order] = key(f, payload["cause"], "seq")
        trig.append((observed.get(payload["cause"], f), order, "seq", payload))
    for s in swaps.swaps:
        order += 1
        keys[order] = s.f_req
        trig.append((s.f_req, order, "swap", {"swap": s}))
    trig.sort(key=lambda x: (keys[x[1]], x[1]))

    sampler = SamplerVoiceSim(sr, cal)
    pitched = PitchedVoiceSim(sr, cal)
    vid = 0
    kits = b.kits
    for f, _o, kind, p in trig:
        if kind == "swap":
            s: Swap = p["swap"]
            oe = kits[s.old].engine if s.old in kits else "sampler"
            ne = kits[s.new].engine if s.new in kits else "sampler"
            if oe != ne:
                sim = sampler if oe == "sampler" else pitched
                for v in sim.stop_all(f, "engine_switch",
                                      uncertain_s=cal.engine_switch_ms / 1000.0):
                    exp.edges.append(Edge(f, "engine_switch", v.vid))
            continue
        if kind == "release":
            a = p["action"]
            pad = int(a.args.get("pad", 0))
            kit = swaps.kit_for(pad, f)[0]
            eng = kits[kit].engine if kit in kits else "sampler"
            (pitched if eng == "pitched" else sampler).release(f, pad, kit)
            continue
        if kind == "seq":
            pad, vel = int(p["pad"]), int(p["vel"])
            src, cause, aid = "seq", p["cause"], None
            hold_s = cal.seq_note_hold_ms / 1000.0
            unc = p.get("uncertain", "")
        else:
            a = p["action"]
            pad = int(a.args.get("pad", 0))
            vel = int(a.args.get("vel", 100)) if kind != "preview" else cal.preview_vel
            src = "preview" if kind == "preview" else "host"
            cause, aid = f"a{a.aid}", a.aid
            hold_s = float(a.args.get("hold_ms") or 0.0) / 1000.0
            unc = ""
        kit, f_on, why, alt, held_by = swaps.kit_for(pad, f)
        if kit not in kits:
            continue
        ks = kits[kit]
        h = S.heard(ks, pad, sr)
        late_ms = max(0.0, (f_on - f) / sr * 1000.0)
        vid += 1
        engine = ks.engine
        g = cal.vel_gain(vel)
        ml, mr = mix.gains("Pitched" if engine == "pitched" else "Sampler")
        level_db = 10 ** (cal.level_offset_db / 20.0)
        if kind == "preview":
            # The kit browser's audition: the pad's first sample at volume 100, pan 0.
            first = ks.pads[pad].layers[0] if engine == "sampler" else None
            gl = gr = (first.peak if first else S.PEAK) * g
            freqs = (h.freqs[0],)
            shape, seconds = "flat", h.seconds
        else:
            gl, gr = h.gain_l * g, h.gain_r * g
            freqs, shape, seconds = h.freqs, h.shape, h.seconds
        v = Voice(vid=vid, src=src, cause=cause, aid=aid, kit=kit, pad=pad, engine=engine,
                  freqs=freqs, f_on=f_on, f_nat=f_on + (INF if shape == "loop" else
                                                         seconds * sr),
                  f_end=f_on + (INF if shape == "loop" else seconds * sr),
                  gain_l=gl * ml * level_db, gain_r=gr * mr * level_db,
                  level_known=h.level_known, shape=shape, fade_on_s=h.fade_s,
                  absent=h.absent_freqs, seq=p if kind == "seq" else None)
        if why or unc:
            v.uncertain = why or unc
        if late_ms:
            v.late_ok_ms = late_ms
            v.held_by = held_by
        v.choke = h.choke
        v.feature = h.feature
        if shape == "rise":
            # it starts near silence and swells: there is no start to time, only its end
            v.uncertain = v.uncertain or "a reversed decay rises from silence: no onset to time"
        if alt and alt in kits and alt != kit:
            if kits[alt].engine != engine:
                v.uncertain = v.uncertain or f"inside the engine switch {alt}->{kit}"
            else:
                v.alt_freqs = S.heard(kits[alt], pad, sr).freqs
                v.tol_on = max(v.tol_on, 0.05 * sr)
        if engine == "pitched":
            ended = pitched.press(v, hold_s if hold_s > 0 else seconds)
        else:
            ended = sampler.press(v)
            if v.end_kind == "toggle_off":
                vid -= 1
                for why_e, o in ended:
                    exp.edges.append(Edge(f_on, why_e, o.vid, by=None))
                continue
        # how well the start that causes the edges is known
        if v.cause in observed:
            etol = 5.0 if v.cause in restarts or v.restart else 2.5
        else:
            etol = cal.seq_jitter_ms + 3.0 if kind == "seq" else 8.0
        for why_e, o in ended:
            if why_e in ("stolen", "choked", "retrigger", "loop_toggle"):
                exp.edges.append(Edge(f_on, why_e, o.vid, by=v.vid, tol_ms=etol))
                if o.engine == "pitched":
                    exp.edges.append(Edge(o.f_end, why_e + "_fade_end", o.vid, by=v.vid,
                                          tol_ms=etol))
        if shape == "loop" or v.fade_on_s == 0.0:
            exp.edges.append(Edge(f_on, "hard_start", v.vid, by=v.vid, tol_ms=etol))
        if max(freqs or (0.0,)) >= HF_SPLATTER_HZ:
            # a short fade of a tone this high splatters into the quiet band above it
            exp.edges.append(Edge(f_on, "hf_fade", v.vid, by=v.vid, tol_ms=etol + 6.0))
        exp.voices.append(v)

    # voices still sounding at the block end are cut there
    for v in exp.voices:
        if max(v.freqs or (0.0,)) >= HF_SPLATTER_HZ and v.f_end < b.n:
            exp.edges.append(Edge(v.half_end(sr), "hf_fade", v.vid, tol_ms=12.0))
        if v.f_end > b.n:
            v.f_end = float(b.n)
            v.end_kind = "block_end"
        if v.shape == "rise" and v.end_kind == "natural":
            exp.edges.append(Edge(v.f_end, "reverse_end", v.vid))

    _mark_churn(exp, sr)
    _pilot_windows(b, cal, exp)
    _flag_actions(b, acts, cal, exp)
    exp.poly = _poly_curve(exp.voices)
    return exp


CHURN_WINDOW_S = 0.006
CHURN_STARTS = 4


def _mark_churn(exp: Expectation, sr: int) -> None:
    """A chord landing on a full engine: several voices start inside a few ms and each
    takes one. Which ones survive is decided inside one or two render blocks, finer
    than any onset can be read -- the model does not pretend to know, and says so."""
    limit = {"sampler": 7, "pitched": PITCHED_CAP - 1}
    w = CHURN_WINDOW_S * sr
    by_engine: dict[str, list[Voice]] = {}
    for v in exp.voices:
        by_engine.setdefault(v.engine, []).append(v)
    for eng, vs in by_engine.items():
        vs.sort(key=lambda v: v.f_on)
        ons = [v.f_on for v in vs]
        import bisect

        for v in vs:
            i = bisect.bisect_left(ons, v.f_on - w)
            j = bisect.bisect_right(ons, v.f_on + w)
            if j - i < CHURN_STARTS:
                continue
            busy = sum(1 for o in vs if o.f_on < v.f_on - w and o.f_end > v.f_on)
            if busy < limit.get(eng, 7) - (j - i):
                continue
            why = f"{j - i} voices start within {CHURN_WINDOW_S * 1000:.0f} ms on a full engine"
            v.uncertain = v.uncertain or why
            for o in vs:
                if o.end_kind == "stolen" and abs(o.f_end - v.f_on) <= w:
                    o.end_uncertain = o.end_uncertain or why


def _runs(b: BlockInput, tm: TransportModel, acts: list[Action], cal: Calibration,
          exp: Expectation) -> list[Run]:
    """The sequencer's runs in the block: from the CLICK anchor, then whatever a bounce
    or a transport message changed (each with a window its start is fitted in)."""
    sr = b.sr
    fr = b.frame_of
    runs: list[Run] = []
    seq0 = b.seq_before or {}
    playing = int(seq0.get("state", 0)) in (3, 4) and bool(b.click.get("running", 1))
    active, pending = parse_scene_field(seq0.get("scene", "0>0"))
    song = b.song or {}
    song_mode = bool(song.get("mode")) and song.get("blocks")
    if song_mode:
        schedule = [(int(s), int(r)) for s, r in song["blocks"]]
        first_block = int(song.get("pos", 0))
        first_pass = int(song.get("pass", 0))
        loop = bool(song.get("loop", 1))
    else:
        schedule = [(active, 1), (pending, 0)] if pending != active else [(active, 0)]
        first_block, first_pass, loop = 0, 0, False
    if playing and "t_host" in b.click:
        pos = int(b.click.get("pos", 0))
        f_anchor = fr(float(b.click["t_host"])) - pos + cal.seq_latency_ms / 1000.0 * sr
        # pass start before the block: walk it forward to the block's start
        runs.append(Run(f_from=0.0, f_stop=float(b.n), f_anchor=f_anchor, schedule=schedule,
                        loop=loop or not song_mode, first_block=first_block,
                        first_pass=first_pass, kind="loop",
                        fit_lo=f_anchor - 0.05 * sr, fit_hi=f_anchor + 0.05 * sr))
    stopped_pos: float | None = None
    for a in acts:
        f = fr(a.t_mono)
        if a.kind == "bounce":
            scene = int(a.args.get("scene", 0))
            w = cal.bounce_start_s * sr
            for r in runs:
                if r.f_stop > f:
                    r.f_stop = f + w
            exp.uncertain.append(Window(f, f + w, "bounce_start", "the bounce's first bar is "
                                        "somewhere in here", a.aid))
            length = tm.loop_len(scene)
            runs.append(Run(f_from=f, f_stop=f + w + length, f_anchor=f, schedule=[(scene, 1)],
                            loop=False, kind="bounce", fit_lo=f, fit_hi=f + w))
            exp.flags.append(Window(f, f + w + length, "bounce", f"scene {scene}", a.aid))
        elif a.kind == "transport":
            what = str(a.args.get("what"))
            w = cal.transport_s * sr
            if what == "FC":
                for r in runs:
                    if r.f_stop > f:
                        if r.f_from <= f:
                            stopped_pos = f - r.f_anchor
                        r.f_stop = f + 0.02 * sr
                exp.uncertain.append(Window(f, f + 0.05 * sr, "transport_stop", "MIDI Stop",
                                            a.aid))
            elif what in ("FA", "FB"):
                last = runs[-1] if runs else None
                sched = last.schedule if last else [(active, 0)]
                if any(r.f_from <= f < r.f_stop for r in runs):
                    for r in runs:
                        if r.f_stop > f:
                            r.f_stop = f
                pos0 = 0
                if what == "FB" and stopped_pos is not None and last is not None:
                    pos0 = int(stopped_pos % tm.loop_len(sched[0][0]))
                runs.append(Run(f_from=f, f_stop=float(b.n), f_anchor=f, schedule=sched,
                                loop=True, kind="start" if what == "FA" else "resume",
                                fit_lo=f, fit_hi=f + w, pos0=pos0))
                exp.uncertain.append(Window(f, f + w, "transport_start",
                                            f"MIDI {'Start' if what == 'FA' else 'Continue'}",
                                            a.aid))
        elif a.kind == "clock_tempo":
            exp.uncertain.append(Window(f, float(b.n), "clock_tempo",
                                        f"host tempo {a.args.get('bpm')} BPM is not the "
                                        "scene's: the grid is not modelled", a.aid))
    exp.runs = [{"kind": r.kind, "f_from": r.f_from, "f_stop": r.f_stop, "f_anchor": r.f_anchor,
                 "schedule": r.schedule} for r in runs]
    return runs


def _overdub(b: BlockInput, tm: TransportModel, acts: list[Action], cal: Calibration,
             exp: Expectation, runs: list[Run], events: dict[int, list[dict[str, int]]]
             ) -> dict[tuple[int, int, int], int]:
    """While the scene records (OVERDUB/REC armed before the block), where each host hit
    should land (``rec_events``), and from which pass the new events sound."""
    seq0 = b.seq_before or {}
    mode = int(seq0.get("mode", 0))
    if mode not in (2, 3) or not runs:
        return {}
    sr = b.sr
    per = grid_per_beat(tm.quant)
    active, _ = parse_scene_field(seq0.get("scene", "0>0"))
    loop = tm.loop_len(active)
    run = runs[0]
    n0 = sum(1 for e in b.dump_before if int(e["seq"]) == active)
    fresh: dict[tuple[int, int, int], int] = {}
    count = n0
    for a in acts:
        if a.kind != "pad":
            continue
        f_on = b.frame_of(a.t_mono) + _ms(cal, str(a.args.get("via", "sysex"))) / 1000.0 * sr
        if f_on >= run.f_stop or f_on < run.f_from:
            continue
        rel = f_on - cal.rec_offset_frames - (run.f_anchor - run.pos0)
        k = int(math.floor(rel / loop))
        pos = int(rel - k * loop)
        q = quantise(pos, tm.spb, per, loop)
        grid = tm.spb // per if per else 0
        ambiguous = bool(grid) and abs((pos % grid) - grid / 2.0) < cal.ambiguous_ms / 1000 * sr
        alt = quantise(pos + (grid // 2 if (pos % grid) < grid / 2 else -(grid // 2)),
                       tm.spb, per, loop) if ambiguous else q
        recorded = count < MAX_EVENTS
        if recorded:
            count += 1
        key = (active, int(a.args.get("pad", 0)), q)
        fresh.setdefault(key, k + 1)
        if ambiguous:
            fresh.setdefault((active, int(a.args.get("pad", 0)), alt), k + 1)
        exp.rec_events.append({"aid": a.aid, "scene": active, "pad": int(a.args.get("pad", 0)),
                               "pos": q, "alt": alt if ambiguous else None,
                               "ambiguous": ambiguous, "pass": k, "recorded": recorded,
                               "raw_pos": pos})
    # what the take shows from the next pass on: the dump after the block
    after = _dump_by_scene(b.dump_after).get(active, [])
    known = {(e["pad"], e["pos"]) for e in events.get(active, [])}
    added = [e for e in after if (e["pad"], e["pos"]) not in known]
    if added:
        events.setdefault(active, [])
        for e in added:
            e = dict(e)
            e["_from_pass"] = fresh.get((active, e["pad"], e["pos"]), 10 ** 9)
            events[active].append(e)
        events[active].sort(key=lambda e: (e["pos"], e["i"]))
    exp.flags.append(Window(run.f_from, run.f_stop, "overdub", f"scene {active}"))
    return fresh


def _seq_triggers(b: BlockInput, tm: TransportModel, runs: list[Run],
                  events: dict[int, list[dict[str, int]]], fresh: dict, cal: Calibration,
                  probe: SeqProbe, exp: Expectation, swaps: SwapModel
                  ) -> list[tuple[float, dict[str, Any]]]:
    sr = b.sr
    out: list[tuple[float, dict[str, Any]]] = []
    for ri, run in enumerate(runs):
        def plan(r: Run, ri: int = ri) -> list[tuple[float, dict[str, Any]]]:
            res = []
            for ps in tm.passes(r, float(b.n)):
                for e in events.get(ps["scene"], []):
                    # an event the overdub added sounds from the pass after its hit
                    if ri == 0 and "_from_pass" in e and ps["k"] < e["_from_pass"]:
                        continue
                    if ps["k"] == 0 and e["pos"] < r.pos0:
                        continue      # a resumed pass starts part-way in
                    f = ps["f0"] + e["pos"]
                    if f < r.f_from or f >= r.f_stop or f >= b.n:
                        continue
                    res.append((f, {"pad": e["pad"], "vel": e["vel"], "scene": ps["scene"],
                                    "i": e["i"], "pass": ps["k"], "run": ri,
                                    "cause": f"seq:{ri}:{ps['scene']}:{e['i']}:{ps['k']}"}))
            return res

        pl = plan(run)
        if pl and run.fit_lo is not None:
            head = pl[:24]
            freqs = [S.heard(b.kits[k], int(p["pad"]), sr).freqs
                     if (k := swaps.kit_for(int(p["pad"]), f)[0]) in b.kits else ()
                     for f, p in head]
            rel = [f - head[0][0] for f, _ in head]
            prior = head[0][0]
            lo = run.fit_lo + (prior - run.f_anchor)
            hi = (run.fit_hi if run.fit_hi is not None else run.f_anchor) + (prior - run.f_anchor)
            got = probe.fit(freqs, rel, min(lo, prior), max(hi, prior), prior=prior)
            if got is not None:
                shift = got - prior
                run.f_anchor += shift
                run.fitted = True
                pl = plan(run)
                exp.runs[ri]["f_anchor"] = run.f_anchor
                exp.runs[ri]["fitted_shift_ms"] = round(shift / sr * 1000.0, 3)
            elif run.kind != "loop":
                for _f, p in pl:
                    p["uncertain"] = f"{run.kind}: its start could not be found in the take"
        for f, p in pl:
            unc = exp.uncertain_at(f)
            if unc is not None and unc.kind in ("bounce_start", "transport_start",
                                                "transport_stop", "clock_tempo"):
                fitted_start = run.fitted and run.kind != "loop" and unc.kind in (
                    "bounce_start", "transport_start")
                if not fitted_start:
                    p["uncertain"] = p.get("uncertain") or unc.why or unc.kind
            exp.seq_events.append({"frame": f, **p})
            out.append((f, p))
    return out


def _pilot_windows(b: BlockInput, cal: Calibration, exp: Expectation) -> None:
    sr = b.sr
    fr = b.frame_of
    if b.t_stream_start is not None:
        exp.pilot_off.append(Window(0.0, fr(b.t_stream_start) + 0.5 * sr, "stream_start",
                                    "the host stream has not reached the take yet"))
    else:
        exp.pilot_off.append(Window(0.0, 0.5 * sr, "stream_start"))
    if b.t_stream_stop is not None:
        exp.pilot_off.append(Window(fr(b.t_stream_stop) - 0.05 * sr, float(b.n), "stream_stop",
                                    "the host stopped streaming"))
    else:
        exp.pilot_off.append(Window(b.n - 0.3 * sr, float(b.n), "block_end"))
    stop_t = None
    for e in b.stream_events:
        k = e.get("kind")
        if k in ("out_stop", "out_mute"):
            stop_t = float(e["t"])
            if k == "out_mute":
                exp.pilot_off.append(Window(fr(stop_t) - 0.05 * sr,
                                            fr(stop_t + float(e.get("seconds", 0)))
                                            + cal.stream_settle_s * sr,
                                            "stream_pause", "the phone muted its stream"))
        elif k == "out_start" and stop_t is not None:
            exp.pilot_off.append(Window(fr(stop_t) - 0.05 * sr,
                                        fr(float(e["t"])) + cal.stream_settle_s * sr,
                                        "stream_pause", "the phone stopped its stream"))
            stop_t = None
        elif k == "xrun" and e.get("side") == "out":
            t = float(e["t"])
            exp.pilot_off.append(Window(fr(t) - 0.1 * sr, fr(t) + 0.3 * sr, "host_xrun",
                                        "host playback xrun"))
    for w in exp.pilot_off:
        if w.kind != "stream_start" or w.f0 > 0:
            exp.edges.append(Edge(w.f0, "pilot_edge"))
            exp.edges.append(Edge(w.f1, "pilot_edge"))


FLAG_KINDS = {"app": ("ui", 3.0), "codec": ("codec", 1.0), "note_neg": ("note_neg", 1.0),
              "stream_pause": ("stream_pause", 3.0), "pad_companion": ("companion", 0.2),
              "companion": ("companion", 0.2), "transport": ("clock", 1.0),
              "clock_start": ("clock", 1.0), "vpad_burst": ("vpad", 2.0)}


def _flag_actions(b: BlockInput, acts: list[Action], cal: Calibration,
                  exp: Expectation) -> None:
    sr = b.sr
    fr = b.frame_of
    ui_from: float | None = None
    for a in acts:
        f = fr(a.t_mono)
        if a.kind in FLAG_KINDS:
            kind, dur = FLAG_KINDS[a.kind]
            dur = float(a.args.get("seconds", dur)) if a.kind == "stream_pause" else dur
            exp.flags.append(Window(f, f + dur * sr, kind, str(a.args), a.aid))
        if a.kind == "app":
            name = str(a.args.get("name"))
            if name != "Sampler" and ui_from is None:
                ui_from = f
            elif name == "Sampler" and ui_from is not None:
                exp.uncertain.append(Window(ui_from, f + 1.0 * sr, "ui", "another app in front: "
                                            "where the pads go is that app's", a.aid))
                if any(str(x.args.get("name")) == "Mixer" for x in acts if x.kind == "app"):
                    exp.flags.append(Window(f, f + 1.0 * sr, "nvs_commit", "closing the Mixer "
                                            "writes its desk to NVS", a.aid))
                ui_from = None
        if a.kind == "step":
            dur = float(a.args.get("dur") or 0.0)
            for fl in a.args.get("flags") or []:
                exp.flags.append(Window(f, f + dur * sr, str(fl), str(a.args.get("id")), a.aid))
    if ui_from is not None:
        exp.uncertain.append(Window(ui_from, float(b.n), "ui", "another app in front"))
    # voices whose onset falls inside an uncertain window cannot be held against the take
    for v in exp.voices:
        if v.uncertain:
            continue
        w = exp.uncertain_at(v.f_on)
        if w is not None and w.kind in ("ui",):
            v.uncertain = w.why or w.kind


def _poly_curve(voices: list[Voice]) -> list[tuple[float, int]]:
    ev: list[tuple[float, int]] = []
    for v in voices:
        if v.uncertain:
            continue
        ev.append((v.f_on, 1))
        ev.append((v.f_end, -1))
    ev.sort()
    out, n = [], 0
    for f, d in ev:
        n += d
        out.append((f, n))
    return out
