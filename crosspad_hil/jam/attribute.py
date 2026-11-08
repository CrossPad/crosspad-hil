"""From findings to glitches: one record per moment, with everything around it.

A 20 ms hole in the whole mix shows up as a pilot gap, a hole in every voice
that was sounding and a click at each edge -- one glitch, not seven findings.
``group`` merges findings within a few ms of each other; ``ContextBuilder``
gives the glitch every clock (``Timebase.axes``), the script step and the
actions around it, what the model had sounding, what the board printed, the
host stream's own events, and ``classify_cause`` weighs that evidence into a
cause with a confidence -- the same rules a person would apply looking at the
same lines, in the order the macierz przyczyn (cause matrix) of the design
gives them.
"""
from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, field
from typing import Any

from crosspad_hil.jam.actions import Action
from crosspad_hil.jam.clocks import Timebase
from crosspad_hil.jam.detect import GLITCH, GLITCH_CLASSES, UNVERIFIABLE, Finding
from crosspad_hil.jam.model import Expectation
from crosspad_hil.jam.telemetry import Telemetry

GROUP_MS = 15.0
ACTIVE_S = 0.25

PAD_CLASSES = {"MISSING", "LATE", "HOLE", "EARLY_CUT", "WRONG_KIT", "GHOST", "SEQ_MISS",
               "SEQ_EXTRA", "CHOKE_MISS", "HOLD_STUCK", "LEVEL_PAN", "LATE_SWAP"}
STREAM_CLASSES = {"PILOT_GAP", "SLIP"}


@dataclass
class GlitchRecord:
    gid: str
    block: str
    kind: str
    classes: dict[str, int]
    frame: float
    length_ms: float
    axes: dict[str, Any] = field(default_factory=dict)
    step: str | None = None
    action: dict[str, Any] | None = None           # the action the finding belongs to
    actions_active: list[dict[str, Any]] = field(default_factory=list)
    last_actions: list[dict[str, Any]] = field(default_factory=list)
    model_state: dict[str, Any] = field(default_factory=dict)
    spk: dict[str, Any] = field(default_factory=dict)
    engine: list[str] = field(default_factory=list)
    console: list[str] = field(default_factory=list)
    host: dict[str, Any] = field(default_factory=dict)
    env: dict[str, Any] = field(default_factory=dict)
    cause: str = "unknown"
    confidence: float = 0.0
    evidence: list[str] = field(default_factory=list)
    findings: list[dict[str, Any]] = field(default_factory=list)
    status: str = GLITCH
    snippet: dict[str, str] = field(default_factory=dict)
    cause_ref: str = ""            # the model's name for what it belongs to (a hit, a seq event)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def group(findings: list[Finding], sr: int, ms: float = GROUP_MS) -> list[list[Finding]]:
    """Findings within ``ms`` of each other (chained) are one glitch. Per-voice findings
    that carry a hit (an action) stay their own glitch unless something else is there."""
    fs = sorted(findings, key=lambda f: f.frame)
    out: list[list[Finding]] = []
    gap = ms / 1000.0 * sr
    for f in fs:
        if out and f.frame - max(g.frame + g.length for g in out[-1]) <= gap:
            out[-1].append(f)
        else:
            out.append([f])
    return out


def _act(a: Action, t: float) -> dict[str, Any]:
    return {"aid": a.aid, "kind": a.kind, "step": a.step, "args": a.args,
            "age_ms": round((t - a.t_mono) * 1000.0, 1), "ok": a.ok}


class ContextBuilder:
    def __init__(self, block: str, exp: Expectation, tb: Timebase, actions: list[Action],
                 tel: Telemetry | None, duplex: dict[str, Any] | None,
                 env: dict[str, Any] | None = None,
                 counters: dict[str, Any] | None = None) -> None:
        self.block = block
        self.exp = exp
        self.tb = tb
        self.actions = sorted(actions, key=lambda a: a.t_mono)
        self.by_aid = {a.aid: a for a in actions}
        self.tel = tel
        self.duplex = duplex or {}
        self.env = env or {}
        self.counters = counters or {}

    def build(self, gid: str, fs: list[Finding]) -> GlitchRecord:
        sr = self.exp.sr
        glitchy = [f for f in fs if f.status == GLITCH] or fs
        primary = min(glitchy, key=lambda f: (_rank(f.cls), f.frame))
        frame = min(f.frame for f in glitchy)
        if any(f.cls in ("PILOT_GAP", "SLIP", "CAPTURE_SLIP") for f in glitchy):
            frame = min(f.frame for f in glitchy if f.cls in ("PILOT_GAP", "SLIP",
                                                               "CAPTURE_SLIP"))
        end = max(f.frame + f.length for f in glitchy)
        classes = dict(Counter(f.cls for f in glitchy))
        rec = GlitchRecord(gid=gid, block=self.block, kind=primary.cls, classes=classes,
                           frame=frame, length_ms=round((end - frame) / sr * 1000.0, 2),
                           findings=[f.to_dict() for f in fs],
                           status=GLITCH if any(f.status == GLITCH for f in fs)
                           else UNVERIFIABLE)
        rec.axes = self.tb.axes(frame)
        t = rec.axes["t_host"]
        rec.step = rec.axes.get("step")
        aid = next((f.aid for f in glitchy if f.aid), None)
        rec.cause_ref = next((f.cause for f in glitchy if f.cause), "")
        if aid is not None and aid in self.by_aid:
            rec.action = _act(self.by_aid[aid], t)
            rec.step = rec.step or self.by_aid[aid].step
        rec.actions_active = [_act(a, t) for a in self.actions
                              if abs(a.t_mono - t) <= ACTIVE_S and a.kind != "step"]
        rec.last_actions = [_act(a, t) for a in self.actions if a.t_mono <= t
                            and a.kind != "step"][-10:]
        if rec.action is None:
            near = [a for a in self.actions if 0 <= t - a.t_mono <= ACTIVE_S
                    and a.kind != "step"]
            if near:
                rec.action = _act(near[-1], t)
        rec.model_state = self._model_state(frame)
        if self.tel is not None and self.tel.rows:
            self._telemetry(rec, t)
        rec.host = self._host(t)
        rec.env = dict(self.env)
        rec.cause, rec.confidence, rec.evidence = classify_cause(rec, glitchy, self.counters)
        return rec

    def _model_state(self, frame: float) -> dict[str, Any]:
        exp = self.exp
        sounding = [v for v in exp.voices if v.f_on <= frame < v.f_end]
        kits = Counter(v.kit for v in sounding)
        return {"voices": len(sounding), "sampler": sum(1 for v in sounding
                                                        if v.engine == "sampler"),
                "pitched": sum(1 for v in sounding if v.engine == "pitched"),
                "kits": dict(kits), "flags": exp.flags_at(frame),
                "uncertain": (exp.uncertain_at(frame).kind if exp.uncertain_at(frame)
                              else None),
                "bands": sorted({round(v.freqs[0]) for v in sounding if v.freqs})[:24]}

    def _telemetry(self, rec: GlitchRecord, t: float) -> None:
        tel = self.tel
        assert tel is not None
        rows = tel.around(t, 1.5, 0.5)
        rec.console = [f"{r.t_host - t:+.3f}s {r.line}" for r in rows if r.kind != "spk"][:40]
        a, b = tel.spk_covering(t)
        spk: dict[str, Any] = {}
        if a is not None:
            spk["before"] = a.fields
        if b is not None:
            spk["after"] = b.fields
            spk["after_dt_s"] = round((b.t_host or b.t_arr) - t, 3)
        if a is not None and b is not None:
            spk["delta"] = {k: b.fields.get(k, 0) - a.fields.get(k, 0)
                            for k in ("under", "refused", "cap_drop", "cap_under")
                            if isinstance(b.fields.get(k), int)
                            and isinstance(a.fields.get(k), int)}
        rec.spk = spk
        rec.engine = [r.line for r in tel.around(t, 2.0, 1.5, kinds=("underrun", "pool"))]
        rec.env["sd_writes"] = [r.line for r in tel.around(t, 0.3, 0.3, kinds=("sd_write",
                                                                             "nvs"))]
        rec.env["log_lines_20ms"] = len([r for r in tel.around(t, 0.02, 0.02)])

    def _host(self, t: float) -> dict[str, Any]:
        ev = self.duplex.get("events") or []
        near = [e for e in ev if abs(float(e.get("t", -1e9)) - t) <= 0.05]
        return {"stream_events": near, "xruns": [e for e in near if e.get("kind") == "xrun"],
                "exclusive": self.duplex.get("exclusive"),
                "clock_resid_ms": self.duplex.get("clock_resid_ms")}


_RANK = {c: i for i, c in enumerate(("CAPTURE_SLIP", "SLIP", "PILOT_GAP", "HOLE", "MISSING",
                                     "SEQ_MISS", "WRONG_KIT", "EARLY_CUT", "LATE",
                                     "LATE_SWAP", "SEQ_EXTRA", "GHOST", "CLICK"))}


def _rank(cls: str) -> int:
    return _RANK.get(cls, 50)


def classify_cause(rec: GlitchRecord, fs: list[Finding],
                   counters: dict[str, Any] | None = None) -> tuple[str, float, list[str]]:
    """The likeliest cause and why, from the findings' shape and the evidence around."""
    counters = counters or {}
    cls = set(rec.classes)
    pads = bool(cls & PAD_CLASSES)
    stream = bool(cls & STREAM_CLASSES)
    hole_all = "PILOT_GAP" in cls and "HOLE" in cls
    flags = set(rec.model_state.get("flags") or [])
    delta = (rec.spk or {}).get("delta") or {}
    xr = rec.host.get("xruns") or []
    ev: list[str] = []
    scores: dict[str, float] = defaultdict(float)

    if any(e.get("side") == "out" for e in xr):
        scores["host_playback"] += 0.9
        ev.append("host playback xrun within 50 ms")
    if any(e.get("side") == "in" for e in xr):
        scores["host_capture"] += 0.9
        ev.append("host capture xrun within 50 ms")
    if "CAPTURE_SLIP" in cls:
        scores["usb_capture"] += 0.8
        ev.append("the pads and the stream jumped by the same samples")
        if delta.get("cap_drop") or delta.get("cap_under"):
            scores["usb_capture"] += 0.15
            ev.append(f"SPK cap_drop/cap_under +{delta.get('cap_drop', 0)}"
                      f"/+{delta.get('cap_under', 0)} around it")
    if stream and not pads and not hole_all:
        scores["usb_stream"] += 0.5
        ev.append("only the host's stream is broken; the pads are intact")
        if delta.get("under") or delta.get("refused"):
            scores["usb_stream"] += 0.35
            ev.append(f"SPK under/refused +{delta.get('under', 0)}/+{delta.get('refused', 0)} "
                      "in the line covering it")
        else:
            ev.append("no SPK counter moved: the device or the host/link (SPK lines are "
                      ">= 1 s apart)")
    if hole_all:
        if delta.get("cap_drop") or delta.get("cap_under"):
            scores["usb_capture"] += 0.7
            ev.append("the whole mix went silent and the capture ring dropped")
        else:
            scores["render_overload"] += 0.45
            ev.append("the whole mix (pads and stream) went silent together")
    if pads and not stream:
        if rec.engine:
            scores["engine_underrun"] += 0.7
            ev.append(f"engine report near it: {rec.engine[0][:90]}")
        if "HOLE" in cls or "CLICK" in cls:
            scores["engine_underrun"] += 0.2
    if "swap" in flags or "engine_switch" in flags:
        scores["swap"] += 0.4 + (0.2 if "WRONG_KIT" in cls or "LATE_SWAP" in cls else 0.0)
        ev.append("inside a kit swap")
    if "fx_render" in flags:
        scores["fx_render_io"] += 0.45
        ev.append("the FX kit was rendering to the card")
    if "bounce" in flags:
        scores["bounce"] += 0.35
        ev.append("a bounce was writing to the card")
    if "ui" in flags or any(a.get("kind") == "app" for a in rec.actions_active):
        scores["ui"] += 0.35
        ev.append("an app switch within 250 ms")
    if "nvs_commit" in flags:
        scores["nvs_commit"] += 0.4
        ev.append("the Mixer closed: its desk was being written to NVS")
    if (rec.env.get("sd_writes") or []):
        scores["sd_writer"] += 0.35
        ev.append(f"card write logged: {rec.env['sd_writes'][0][:80]}")
    if rec.env.get("log_lines_20ms"):
        scores["log_busywait"] += 0.15
        ev.append("a console line within 20 ms (UART0 printing blocks its task)")
    if "EARLY_CUT" in cls and rec.model_state.get("sampler", 0) >= 8:
        scores["polyphony"] += 0.5
        ev.append("8 voices sounding: the engine's budget")
    if ("MISSING" in cls or "SEQ_MISS" in cls) and counters.get("inj_drop"):
        scores["inject_queue"] += 0.6
        ev.append(f"PAD_STATS inj_drop +{counters['inj_drop']} this block")
    if cls & {"SEQ_MISS", "LATE"} and rec.classes.get("LATE", 0) + rec.classes.get(
            "SEQ_MISS", 0) >= 2:
        scores["seq_poll_stall"] += 0.4
        ev.append("several sequencer events late or missing together")
    if not scores:
        return "unknown", 0.0, ev or ["nothing around it points anywhere"]
    cause, score = max(scores.items(), key=lambda kv: kv[1])
    return cause, round(min(1.0, score), 2), ev


def flag_table(records: list[GlitchRecord], minutes_by_flag: dict[str, float]
               ) -> dict[str, dict[str, Any]]:
    """Glitches per minute by segment flag: where (in what conditions) they cluster."""
    counts: Counter[str] = Counter()
    for r in records:
        if r.status != GLITCH:
            continue
        fl = r.model_state.get("flags") or ["(none)"]
        for f in fl:
            counts[f] += 1
    out = {}
    for f in sorted(set(counts) | set(minutes_by_flag)):
        mins = minutes_by_flag.get(f, 0.0)
        out[f] = {"glitches": counts.get(f, 0), "minutes": round(mins, 2),
                  "per_min": round(counts.get(f, 0) / mins, 2) if mins > 0 else None}
    return out


def is_glitch_class(cls: str) -> bool:
    return cls in GLITCH_CLASSES
