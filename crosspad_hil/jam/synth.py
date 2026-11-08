"""A jam's run directory without a board: what the analysis is tested against.

``synth_run`` plays a short set the way the scenario would -- the same action
log, the same readbacks, the same console with arrival times, a take rendered
from the model with the pilot and timecode on it -- optionally with faults put
in, and writes it all where ``analysis.analyze_run`` looks. The fake board of
the scenario's tests renders its takes with ``render_take`` the same way.
"""
from __future__ import annotations

import json
import random
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from crosspad_hil.jam import spec as S
from crosspad_hil.jam.actions import Action
from crosspad_hil.jam.calibration import Calibration
from crosspad_hil.jam.model import BlockInput, Expectation, build_expectation
from crosspad_hil.jam.render import Fault, render_expected, write_wav

T0 = 10_000.0           # host monotonic at the first action
BOOT_T = T0 - 600.0     # when the board's log clock started
PATH_DELAY = 1789       # frames from the host's stream to the take


@dataclass
class SynthStep:
    id: str
    t: float
    dur: float
    kind: str = "pattern"                  # pattern | swap | preview | stream_pause | bounce
    hits: list[tuple[float, int, int, float]] = field(default_factory=list)  # t, pad, vel, hold
    args: dict[str, Any] = field(default_factory=dict)
    flags: list[str] = field(default_factory=list)


def s1_dump(sr: int = 44_100, bpm: int = 120) -> list[dict[str, int]]:
    """S1 (2 bars): kick p0 on 1 and 3, snare p1 on 2 and 4, hat p2 on the 8ths,
    open hat p3 on the 'and' of 4 -- as SEQ_DUMP lists it."""
    spb = int(round(sr * 60.0 / bpm))
    ev = []
    for bar in range(2):
        b0 = bar * 4 * spb
        ev += [(b0, 0), (b0 + 2 * spb, 0), (b0 + spb, 1), (b0 + 3 * spb, 1)]
        ev += [(b0 + k * spb // 2, 2) for k in range(8)]
        ev += [(b0 + 3 * spb + spb // 2, 3)]
    ev.sort()
    return [{"seq": 0, "i": i, "pos": p, "pad": pad, "vel": 100} for i, (p, pad) in enumerate(ev)]


def mini_steps(rng: random.Random) -> list[SynthStep]:
    """~20 s that touch every mechanism the detectors model."""
    def jit() -> float:
        return rng.uniform(-6e-4, 6e-4)

    steps = [SynthStep("B1.stream", 0.0, 21.0, kind="listen")]
    steps.append(SynthStep("B1.one", 1.0, 3.0, hits=[(i / 12 + jit(), 12, 100, 40.0)
                                                     for i in range(36)], flags=["one"]))
    steps.append(SynthStep("B1.four", 4.5, 3.0, hits=[(i / 8 + jit(), 8 + i % 4, 100, 40.0)
                                                      for i in range(24)], flags=["four"]))
    steps.append(SynthStep("B1.swap_ab", 6.0, 0.1, kind="swap", args={"kit": S.JAM_B},
                           flags=["swap"]))
    steps.append(SynthStep("B1.preview", 8.0, 1.0, kind="preview", args={"pads": [5]}))
    many = [(i / 16 + jit(), i % 16, 100, 40.0) for i in range(48)]
    many += [(1.5 + j * 0.0005, p, 100, 40.0) for j, p in enumerate(rng.sample(range(16), 8))]
    steps.append(SynthStep("B1.many", 9.5, 3.0, hits=sorted(many), flags=["many"]))
    steps.append(SynthStep("B1.stream_pause", 13.0, 1.0, kind="stream_pause",
                           args={"seconds": 1.0, "mode": "silence"}, flags=["stream_pause"]))
    steps.append(SynthStep("B1.swap_ap", 14.5, 0.1, kind="swap", args={"kit": S.JAM_P},
                           flags=["swap", "engine_switch"]))
    leg = []
    t, p = 0.0, 7
    while t < 2.5:
        leg.append((t + jit(), p, 100, rng.uniform(150, 300)))
        p = max(0, min(15, p + rng.choice((-2, -1, 1, 2))))
        t += 0.34
    for j, pad in enumerate(rng.sample(range(16), 12)):
        leg.append((2.8 + j * 0.0005, pad, 100, 700.0))
    steps.append(SynthStep("B1.pitched", 17.0, 4.0, hits=sorted(leg), flags=["pitched"]))
    return steps


def _swap_row(seq: int, kit_id: int, req_ms: int, ok: bool = True,
              pad_ms: int = 60) -> dict[str, Any]:
    pads = [30 + pad_ms * (i + 1) // 4 for i in range(16)]
    return {"seq": seq, "kit": kit_id, "ok": ok, "req_ms": req_ms, "parse_ms": 30,
            "done_ms": max(pads), "dropped_hits": 0, "late_hits": 0, "underruns": 0,
            "noblock": 0, "deferred": 0, "forced": 0, "pads": pads, "rt_gap_max_us": 900}


def synth_run(run_dir: Path, *, seed: int = 1, sr: int = 44_100,
              faults: Callable[[Expectation, int], list[Fault]] | None = None,
              steps: list[SynthStep] | None = None, spk_under_at: float | None = None,
              hung: bool = False, noise: float = 2e-5, duplex_events: list[dict] | None = None
              ) -> dict[str, Any]:
    """Write a whole run directory; returns what was put in (expectation, faults, frames)."""
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    rng = random.Random(seed)
    steps = steps if steps is not None else mini_steps(rng)
    kits = S.jam_kits()
    name = "B1"
    dur = max(s.t + s.dur for s in steps) + 1.0
    n = int(dur * sr)
    t_cap0 = T0 - 0.2
    frame_of = (lambda t: (t - t_cap0) * sr)  # noqa: E731
    acts: list[Action] = []
    aid = 0

    def act(kind: str, t: float, step: str, **args: Any) -> Action:
        nonlocal aid
        aid += 1
        a = Action(t_mono=t, t_mono_end=t + 2e-4, t_wall=t + 1.7e9, block=name, step=step,
                   kind=kind, args=args, aid=aid)
        acts.append(a)
        return a

    swap_rows = []
    kit_ids = {S.JAM_A: 20, S.JAM_B: 21, S.JAM_FX: 22, S.JAM_P: 23}
    for st in steps:
        t0 = T0 + st.t
        act("step", t0, st.id, id=st.id, dur=st.dur, flags=st.flags)
        if st.kind == "pattern":
            for (t, pad, vel, hold) in st.hits:
                act("pad", t0 + t, st.id, pad=pad, vel=vel, hold_ms=hold, via="sysex")
        elif st.kind == "swap":
            a = act("kit", t0, st.id, name=st.args["kit"], kit=kit_ids[st.args["kit"]],
                    via="sysex")
            swap_rows.append(_swap_row(len(swap_rows) + 1, kit_ids[st.args["kit"]],
                                       int((a.t_mono - BOOT_T) * 1000) - 760))
        elif st.kind == "preview":
            for i, pad in enumerate(st.args.get("pads", [])):
                act("preview", t0 + 0.4 * i, st.id, pad=pad)
        elif st.kind == "stream_pause":
            act("stream_pause", t0, st.id, **st.args)
    # the stream's own events (the duplex thread logs them)
    events = list(duplex_events or [])
    for a in acts:
        if a.kind == "stream_pause":
            events.append({"kind": "out_mute", "t": a.t_mono, "seconds": a.args["seconds"],
                           "frames_in": 0, "frames_out": 0})
            events.append({"kind": "out_unmute", "t": a.t_mono + a.args["seconds"],
                           "frames_in": 0, "frames_out": 0})
    spb = int(round(sr * 0.5))
    loop = 2 * 4 * spb
    t_click = T0 - 5.0
    pos = int(((t_click - (T0 - 7.3)) * sr) % loop)
    seq = {"mode": 1, "state": 4, "scenes": 1, "scene": "0>0", "bars": [2, 1, 1, 1, 1, 1, 1, 1],
           "ev": [len(s1_dump(sr))], "bpm": 120.0, "quant": 2, "dirty": 0, "countin": 1}
    click = {"running": 1, "bpm": 120.0, "pos": pos, "loop": loop, "t_host": t_click}
    dump = s1_dump(sr)
    b = BlockInput(name=name, sr=sr, n=n, frame_of=frame_of, actions=acts, kits=kits,
                   kit0=S.JAM_A, seq_before=seq, seq_after=seq, dump_before=dump,
                   dump_after=dump, click=click, swap_rows=swap_rows, stream_events=events,
                   t_stream_start=T0 - 0.15, t_stream_stop=T0 + dur - 0.6)
    cal = Calibration()
    exp = build_expectation(b, cal)
    fl = faults(exp, sr) if faults else []
    pilot = S.PilotSpec()
    # A hit the engine got late or never got changes what it did to the others (a
    # steal moves or does not happen): those faults are played through the model.
    by_vid = {v.vid: v for v in exp.voices}
    seen = {by_vid[f.vid].cause: by_vid[f.vid].f_on + f.ms / 1000.0 * sr
            for f in fl if f.kind == "delay_hit" and f.vid in by_vid}
    gone = {by_vid[f.vid].cause for f in fl if f.kind == "drop_hit" and f.vid in by_vid}
    played = build_expectation(b, cal, observed=seen, dropped=gone) if seen or gone else exp
    rest = [f for f in fl if f.kind not in ("delay_hit", "drop_hit")]
    x = render_expected(played, pilot=pilot, delay=PATH_DELAY + int(0.05 * sr), faults=rest,
                        noise=noise, seed=seed)
    (run_dir / "blocks" / name).mkdir(parents=True, exist_ok=True)
    write_wav(run_dir / "blocks" / name / "capture.wav", x, sr)
    # the duplex clock log: late, never early
    pts = []
    for k in range(0, n, sr // 50):
        t = t_cap0 + k / sr + abs(rng.gauss(0.0, 4e-4)) + 1e-4
        pts.append([k, t])
    counters0 = {"pads": {"press": 0, "inj_drop": 0, "uart_ovf": 0, "sbuf_drop": 0,
                          "usb_drop": 0}, "smpl": {"underrun": 0, "noblock": 0}, "resets": 0,
                 "pitched": {"shed": 0}}
    blk = {"name": name, "sr": sr, "wav": f"blocks/{name}/capture.wav", "t_capture_start": t_cap0,
           "t_stream_start": T0 - 0.15, "t_stream_stop": T0 + dur - 0.6,
           "capture_points": pts, "duplex": {"health": {"ok": True, "missed_s": 0.0,
                                                         "overruns": 0, "out_xruns": 0},
                                             "events": events, "exclusive": True},
           "pilot": pilot.to_dict(), "kit0": S.JAM_A, "fx_first_load": False,
           "readback_before": {"seq": seq, "seq_dump": dump, "song": {}, "click": click,
                               "counters": counters0},
           "readback_after": {"seq": seq, "seq_dump": dump, "counters": counters0},
           "swap_rows": swap_rows, "desk": {},
           "hung": ({"t_s": 12.0, "reason": "capture stopped: no frames for 2 s",
                     "console_tail": ["E (1) task_wdt: Task watchdog got triggered"]}
                    if hung else None)}
    (run_dir / "blocks" / name / "block.json").write_text(json.dumps(blk), encoding="utf-8")
    with open(run_dir / "actions.jsonl", "w", encoding="utf-8") as fh:
        for a in acts:
            fh.write(json.dumps(a.to_dict()) + "\n")
    # the console: SPK lines once a second, with the board's log ms and arrival times
    rows = []
    seqn = 0
    under = 0
    for k in range(int(dur) + 2):
        t = T0 - 0.5 + k
        if spk_under_at is not None and t >= T0 + spk_under_at:
            under = 3
        ms = int((t - BOOT_T) * 1000)
        line = (f"I ({ms}) usb_audio_bridge: SPK #{k + 1}: len=176 taken=44 ring=600 "
                f"under={under} refused=0 cap_ring=330 cap_drop=0 ring_min=420")
        seqn += 1
        rows.append(f"{seqn}\t{t + (len(line) + 2) * 10 / 115200 + 0.0008:.6f}\t"
                    f"{t + 1.7e9:.6f}\t{line}")
    for a in acts:
        if a.kind == "kit":
            ms = int((a.t_mono - BOOT_T) * 1000) + 3
            seqn += 1
            rows.append(f"{seqn}\t{a.t_mono + 0.01:.6f}\t{a.t_wall:.6f}\t"
                        f"I ({ms}) hil_control: KIT_LOAD {a.args['kit']} started")
    rows.sort(key=lambda r: float(r.split("\t")[1]))
    (run_dir / "console.tsv").write_text("\n".join(rows) + "\n", encoding="utf-8")
    from crosspad_hil.jam.script import Block, JamScript, Step

    blk_s = Block(name, "synthetic", dur, S.JAM_A, steps=[Step(id=s.id, block=name, t_rel=s.t,
                                                              dur=s.dur, action=s.kind,
                                                              features=[])
                                                         for s in steps])
    script = JamScript(seed=seed, scale=1.0, bpm=120, blocks=[blk_s], windows={},
                       order=[name])
    (run_dir / "script.json").write_text(json.dumps(script.to_dict()), encoding="utf-8")
    (run_dir / "env.json").write_text(json.dumps({"fw": "synthetic", "rate": sr}),
                                      encoding="utf-8")
    return {"expectation": exp, "faults": fl, "sr": sr, "actions": acts, "frame_of": frame_of,
            "n": n, "x": x}


def render_take(b: BlockInput, cal: Calibration | None = None, *, delay: int = PATH_DELAY,
                faults: list[Fault] | None = None, noise: float = 2e-5) -> np.ndarray:
    """The take a perfect board would have recorded for ``b`` (fake boards use it)."""
    exp = build_expectation(b, cal or Calibration())
    return render_expected(exp, pilot=S.PilotSpec(), delay=delay, faults=faults, noise=noise)
