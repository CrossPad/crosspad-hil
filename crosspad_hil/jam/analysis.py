"""One pure function from a jam's run directory to its verdict: ``analyze_run``.

Nothing here touches a board: everything the scenario learnt while it ran is
in the run directory (the takes, the clock logs, ``actions.jsonl``, the
console with arrival times, the readbacks, the bounces), so a run can be
analysed again later -- with other tolerances, a new calibration, a fixed
detector -- and give the same answer for the same inputs
(``crosspad-hil jam analyze <run_dir>``).

Per block: the capture clock; the board's state before and after; the model's
expectation (``model.build_expectation``, the sequencer re-fitted to the take);
every detector; findings grouped into glitches, each with its clocks, step,
actions and the evidence around it (``attribute``). Then the counters, the
bounces, the coverage, and one verdict.
"""
from __future__ import annotations

import json
import math
import wave
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from crosspad_hil.jam import attribute as A
from crosspad_hil.jam import bounce as BO
from crosspad_hil.jam import detect as D
from crosspad_hil.jam import report as R
from crosspad_hil.jam import spec as S
from crosspad_hil.jam.actions import Action, load_actions
from crosspad_hil.jam.bands import BandBank
from crosspad_hil.jam.calibration import Calibration
from crosspad_hil.jam.clocks import CaptureClock, Timebase
from crosspad_hil.jam.model import (
    BlockInput,
    Expectation,
    build_expectation,
    grid_per_beat,
    samples_per_beat,
)
from crosspad_hil.jam.render import render_expected, write_wav
from crosspad_hil.jam.telemetry import Telemetry

REPORT = "jam_report.json"
COUNTER_KEYS_PADS = ("inj_drop", "uart_ovf", "sbuf_drop", "usb_drop")
COUNTER_KEYS_ENGINE = ("underrun", "noblock")
COUNTER_KEYS_POOL = ("esc", "fixed", "leak", "torn")
# SMPL_ENGINE: the native engine's -- a voice refused its ring plays its resident head
# and stops; drop = a note with no voice to take.
COUNTER_KEYS_NATIVE = ("underrun", "norings", "drop")


@dataclass
class BlockAnalysis:
    name: str
    seconds: float
    findings: list[D.Finding] = field(default_factory=list)
    glitches: list[A.GlitchRecord] = field(default_factory=list)
    measured: dict[str, Any] = field(default_factory=dict)
    pilot: dict[str, Any] = field(default_factory=dict)
    not_measured: str | None = None
    counters: list[dict[str, Any]] = field(default_factory=list)
    expectation: Expectation | None = None
    lanes: dict[str, Any] = field(default_factory=dict)
    clock: CaptureClock | None = None


@dataclass
class JamAnalysis:
    run_dir: str
    blocks: list[BlockAnalysis] = field(default_factory=list)
    doc: dict[str, Any] = field(default_factory=dict)
    measured: dict[str, Any] = field(default_factory=dict)

    @property
    def status(self) -> str:
        return str(self.doc.get("verdict", {}).get("status"))

    @property
    def exit_code(self) -> int:
        return int(self.doc.get("verdict", {}).get("exit", 2))

    @property
    def glitches(self) -> list[dict[str, Any]]:
        return list(self.doc.get("glitches", []))


def _json(path: Path, default: Any) -> Any:
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return default


def read_take(path: Path) -> tuple[np.ndarray, int]:
    with wave.open(str(path), "rb") as w:
        sr, ch, width = w.getframerate(), w.getnchannels(), w.getsampwidth()
        raw = w.readframes(w.getnframes())
    if width != 2:
        raise ValueError(f"{path}: {8 * width}-bit; the jam reads 16-bit")
    x = np.frombuffer(raw, dtype="<i2").reshape(-1, ch).astype(np.float32) / np.float32(32768.0)
    if ch == 1:
        x = np.repeat(x, 2, axis=1)
    return x, sr


def _groups(kits: dict[str, S.KitSpec], sr: int) -> tuple[dict[str, list[float]], list[float]]:
    groups: dict[str, set[float]] = {"AB": set(), "FX": set(), "P": set()}
    for name, kit in kits.items():
        for pad in range(S.PADS):
            h = S.heard(kit, pad, sr)
            g = "P" if kit.engine == "pitched" else ("FX" if name == S.JAM_FX else "AB")
            for f in (*h.freqs, *h.absent_freqs):
                groups[g].add(round(f, 3))
    ghost = sorted({f for v in groups.values() for f in v} | {round(c, 3) for c in S.CLICK_HZ})
    return {k: sorted(v) for k, v in groups.items()}, ghost


def _delta(after: dict[str, Any] | None, before: dict[str, Any] | None,
           keys: tuple[str, ...]) -> dict[str, int]:
    out: dict[str, int] = {}
    for k in keys:
        a, b = (after or {}).get(k), (before or {}).get(k)
        if isinstance(a, (int, float)) and isinstance(b, (int, float)) and a - b > 0:
            out[k] = int(a - b)
    return out


def counter_findings(name: str, blk: dict[str, Any], tel: Telemetry | None,
                     t_span: tuple[float, float] | None) -> list[dict[str, Any]]:
    """Every board counter that moved in the block: a glitch the take may not show."""
    out: list[dict[str, Any]] = []
    rb0 = (blk.get("readback_before") or {}).get("counters") or {}
    rb1 = (blk.get("readback_after") or {}).get("counters") or {}

    def add(what: str, d: dict[str, int], hint: str = "") -> None:
        for k, v in d.items():
            out.append({"block": name, "counter": f"{what}.{k}", "delta": v,
                        "note": f"{what} {k} +{v}" + (f" ({hint})" if hint else "")})

    add("pads", _delta(rb1.get("pads"), rb0.get("pads"), COUNTER_KEYS_PADS),
        "a pad hit lost on its way in")
    add("engine", _delta(rb1.get("smpl"), rb0.get("smpl"), COUNTER_KEYS_ENGINE),
        "the sampler engine ran dry")
    add("pool", _delta(rb1.get("pool"), rb0.get("pool"), COUNTER_KEYS_POOL),
        "the stream-block pool")
    add("native", _delta(rb1.get("native"), rb0.get("native"), COUNTER_KEYS_NATIVE),
        "the native sampler engine")
    add("pitched", _delta(rb1.get("pitched"), rb0.get("pitched"), ("shed",)),
        "voices shed by an overloaded render")
    for row in blk.get("swap_rows") or []:
        for k in ("dropped_hits", "underruns", "noblock"):
            if int(row.get(k) or 0) > 0:
                out.append({"block": name, "counter": f"swap.{k}", "delta": int(row[k]),
                            "note": f"swap seq {row.get('seq')} {k}={row[k]}"})
    asong0 = (rb0.get("asong") or {}).get("under")
    asong1 = (rb1.get("asong") or {}).get("under")
    if isinstance(asong0, int) and isinstance(asong1, int) and asong1 > asong0:
        out.append({"block": name, "counter": "asong.under", "delta": asong1 - asong0,
                    "note": f"song playback underruns +{asong1 - asong0}"})
    r0, r1 = rb0.get("resets"), rb1.get("resets")
    if isinstance(r0, int) and isinstance(r1, int) and r1 > r0:
        out.append({"block": name, "counter": "resets", "delta": r1 - r0,
                    "note": f"{r1 - r0} reset(s)"})
    if tel is not None and t_span is not None:
        t0, t1 = t_span
        rows = [r for r in tel.of_kind("spk", t0=t0, t1=t1 + 2.0)]
        if rows:
            first, last = rows[0].fields, rows[-1].fields
            prev = [r for r in tel.of_kind("spk", t1=t0)]
            base = prev[-1].fields if prev else first
            under = int(last.get("under") or 0)
            # the stream stopping before the switch out leaves one under on the last line
            if len(rows) >= 2 and int(last.get("ring", -1) or 0) == 0 and \
                    under > int(rows[-2].fields.get("under") or 0):
                under = int(rows[-2].fields.get("under") or 0)
            for k, v in (("under", under), ("refused", int(last.get("refused") or 0))):
                if v > 0:
                    out.append({"block": name, "counter": f"spk.{k}", "delta": v,
                                "note": f"usb_audio_bridge {k}={v}: the host stream ran dry or "
                                        "overflowed the board's ring"})
            for k in ("cap_drop", "cap_under"):
                a, b = base.get(k), last.get(k)
                if isinstance(a, int) and isinstance(b, int) and b > a:
                    out.append({"block": name, "counter": f"spk.{k}", "delta": b - a,
                                "note": f"usb_audio_bridge {k} +{b - a}: the capture ring"})
        for r in tel.of_kind("fatal", t0=t0, t1=t1):
            out.append({"block": name, "counter": "fatal", "delta": 1, "note": r.line[:160]})
        for r in tel.of_kind("underrun", t0=t0, t1=t1 + 2.0):
            out.append({"block": name, "counter": "engine.underrun_line", "delta": 1,
                        "note": r.line[:160], "t_host": r.t_host})
    if blk.get("hung"):
        h = blk["hung"]
        out.append({"block": name, "counter": "hung", "delta": 1,
                    "note": f"board hung at t={h.get('t_s')} s: {h.get('reason')}"})
    return out


def analyze_block(run_dir: Path, blk: dict[str, Any], actions: list[Action],
                  tel: Telemetry | None, cal: Calibration, tol: D.Tolerances,
                  env: dict[str, Any], steps: list[dict[str, Any]], snippets: int,
                  gid0: int, write_expected: bool = True) -> BlockAnalysis:
    name = blk["name"]
    wav = run_dir / blk.get("wav", f"blocks/{name}/capture.wav")
    x, sr = read_take(wav)
    n = len(x)
    ba = BlockAnalysis(name=name, seconds=n / sr)
    kits = S.jam_kits()
    duplex = blk.get("duplex") or {}
    events = duplex.get("events") or []
    breaks = [float(e["t"]) for e in events if e.get("kind") == "xrun" and e.get("side") == "in"]
    clock = CaptureClock.from_points([tuple(p) for p in blk.get("capture_points") or []], sr,
                                     breaks=breaks, t_start=blk.get("t_capture_start"))
    ba.clock = clock
    mine = [a for a in actions if a.block == name]
    rb0 = blk.get("readback_before") or {}
    rb1 = blk.get("readback_after") or {}
    pilot = S.PilotSpec.from_dict(blk.get("pilot") or {})
    b = BlockInput(name=name, sr=sr, n=n, frame_of=clock.frame_of, actions=mine, kits=kits,
                   kit0=str(blk.get("kit0") or S.JAM_A), seq_before=rb0.get("seq") or {},
                   seq_after=rb1.get("seq") or {}, dump_before=rb0.get("seq_dump") or [],
                   dump_after=rb1.get("seq_dump") or [], song=rb0.get("song") or {},
                   click=rb0.get("click") or {}, swap_rows=blk.get("swap_rows") or [],
                   desk=blk.get("desk") or {}, stream_events=events,
                   t_stream_start=blk.get("t_stream_start"), t_stream_stop=blk.get("t_stream_stop"),
                   fx_first_load=bool(blk.get("fx_first_load")))
    groups, ghost = _groups(kits, sr)
    bank = BandBank.build(x, sr, groups, ghost_freqs=ghost)
    probe = D.BandProbe(bank, sr)
    exp1 = build_expectation(b, cal, probe=probe)
    rebuilt: list[Expectation] = []

    def rebuild(seen: dict[str, float], restarts: set[str], shift: dict[str, float]
                ) -> Expectation:
        rebuilt.append(build_expectation(b, cal, probe=probe, observed=seen,
                                         restarts=restarts, shift_ms=shift))
        return rebuilt[-1]

    fs, measured, _res = D.score_voices(x, exp1, bank, tol, kits, rebuild=rebuild)
    exp = rebuilt[-1]
    ba.expectation = exp
    ba.measured = {k: v for k, v in measured.items() if k not in ("ends", "steal_moments")}
    pf, pinfo = D.pilot_findings(x, sr, pilot, exp, tol)
    D.classify_slips(x, sr, exp, [f for f in pf if f.cls == "SLIP"])
    ba.pilot = pinfo
    fs += pf
    fs += D.hf_clicks(x, sr, exp, tol, explain=measured.get("steal_moments"))
    seq_pads = {int(e["pad"]) for e in exp.seq_events}
    fs += D.ghost_scan(bank, exp, tol, kits=kits, seq_pads=seq_pads,
                       ends=measured.get("ends"))
    fs += D.clip_scan(x, sr)
    if exp.rec_events:
        seq0 = b.seq_before
        spb = samples_per_beat(float(seq0.get("bpm") or 120.0), sr)
        frames = {a.aid: clock.frame_of(a.t_mono) for a in mine}
        fs += D.rec_findings(exp, b.dump_before, b.dump_after, spb,
                             grid_per_beat(int(seq0.get("quant", 2))), frames)
    # host losses: what the host could not hear is not the board's
    unmeasured_spans = []
    for e in events:
        if e.get("kind") == "xrun":
            f = clock.frame_of(float(e["t"]))
            unmeasured_spans.append((f - 0.05 * sr, f + 0.3 * sr))
    for f in fs:
        if any(a <= f.frame <= b2 for a, b2 in unmeasured_spans):
            f.status = D.UNVERIFIABLE
            f.note = (f.note + "; " if f.note else "") + "host xrun: not the board's"
    health = duplex.get("health") or {}
    if health.get("missed_s", 0) and float(health.get("missed_s") or 0) > 1.0:
        ba.not_measured = f"the host lost {health['missed_s']} s of the capture"
    if clock.note and not clock.fits:
        ba.not_measured = ba.not_measured or f"no capture clock: {clock.note}"
    if blk.get("not_measured"):
        ba.not_measured = blk["not_measured"]
    ba.findings = fs
    # glitches
    tb = Timebase(sr=sr, capture=clock, log=tel.clock if tel else None,
                  wall_offset=float(np.median([a.t_wall - a.t_mono for a in mine]))
                  if mine else 0.0, steps=[s for s in steps if s.get("block") == name],
                  runs=exp.runs, bpm=float((b.click or {}).get("bpm") or 120.0),
                  bars=list((b.seq_before or {}).get("bars") or []))
    pads0, pads1 = (rb0.get("counters") or {}).get("pads"), (rb1.get("counters") or {}).get("pads")
    ctx = A.ContextBuilder(name, exp, tb, mine, tel, {**duplex, **health, "events": events},
                           env=env, counters=_delta(pads1, pads0, COUNTER_KEYS_PADS))
    judged = [f for f in fs if f.status in (D.GLITCH, D.UNVERIFIABLE)
              and (f.cls in D.GLITCH_CLASSES)]
    gid = gid0
    for grp in A.group(judged, sr):
        if not any(f.status == D.GLITCH for f in grp) and all(
                f.cls in ("MISSING", "SEQ_MISS") for f in grp):
            continue          # an unreadable restart alone is a blind spot, not a record
        gid += 1
        ba.glitches.append(ctx.build(f"g{gid:04d}", grp))
    t_span = (clock.t_of(0), clock.t_of(n)) if clock.fits else None
    ba.counters = counter_findings(name, blk, tel, t_span)
    lost = [a for a in mine if a.kind == "midi_in_lost"]
    if lost:
        ba.counters.append({"block": name, "counter": "midi_in_lost", "delta": len(lost),
                            "note": f"{len(lost)}x the board's MIDI replies to the host lost "
                                    "while its console went on (usb_drop), not a hang"})
    # lanes for the page
    ba.lanes = {
        "actions_lane": [[round(clock.frame_of(a.t_mono) / sr, 4), a.kind,
                          f"a{a.aid} {a.kind} {a.args}"[:120]] for a in mine
                         if a.kind not in ("step",)][:4000],
        "flags_lane": [[round(w.f0 / sr, 3), round(min(w.f1, n) / sr, 3), w.kind]
                       for w in exp.flags][:500],
        "poly_lane": [[round(f / sr, 3), k] for f, k in exp.poly][::max(1, len(exp.poly) // 3000)],
    }
    exp_x = None
    if write_expected or snippets:
        exp_x = render_expected(exp, pilot=pilot,
                                delay=int((pinfo.get("timecode", {}).get("delays") or [0])[0]),
                                n=n).astype(np.float32)
        if write_expected:
            write_wav(run_dir / f"expected_{name}.wav", exp_x, sr)
    if snippets:
        recs = [g.to_dict() for g in ba.glitches]
        R.cut_snippets(run_dir / "glitches", x, exp_x, sr, recs, snippets)
        for g, rec in zip(ba.glitches, recs, strict=True):
            g.snippet = rec.get("snippet", {})
    return ba


def bounce_findings(run_dir: Path, bounces: list[dict[str, Any]],
                    blocks: dict[str, BlockAnalysis]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for bo in bounces:
        entry: dict[str, Any] = {"name": bo.get("name"), "block": bo.get("block"),
                                 "via": bo.get("via"), "ok": True, "problems": []}
        if int(bo.get("state", 0)) != 2:
            entry["problems"].append(f"bounce state {bo.get('state')} "
                                     f"({bo.get('err') or 'did not complete'})")
        c0, c1 = bo.get("clips_before"), bo.get("clips_after")
        if isinstance(c0, int) and isinstance(c1, int) and c1 != c0 + 1:
            entry["problems"].append(f"clips on its track went {c0} -> {c1}")
        local = bo.get("local")
        if local and (run_dir / local).exists():
            try:
                ref, sr = BO.read_wav(run_dir / local)
            except (OSError, ValueError, EOFError, wave.Error) as e:
                entry["problems"].append(f"the file does not read: {e}")
                ref, sr = None, 0
            if ref is not None:
                peak = float(np.abs(ref).max()) if len(ref) else 0.0
                entry["seconds"] = round(len(ref) / sr, 3)
                if peak < 10 ** (-60 / 20):
                    entry["problems"].append("the bounced file is silent")
                from crosspad_hil.scenarios.live_set import score_take

                try:
                    take = score_take(run_dir / local)
                    entry["take"] = {k: take.get(k) for k in ("clicks", "dropouts", "holes",
                                                              "hole_max_ms", "peak_dbfs")}
                    if take.get("clicks") or take.get("dropouts") or take.get("holes"):
                        entry["problems"].append(
                            f"the file has {take.get('clicks')} click(s), {take.get('holes')} "
                            f"hole(s) (longest {take.get('hole_max_ms')} ms)")
                except Exception as e:  # noqa: BLE001 - a scoring crash is reported
                    entry["problems"].append(f"scoring the file: {e}")
                ba = blocks.get(str(bo.get("block")))
                if ba is not None and bo.get("via") == "sysex" and ba.clock is not None:
                    wavp = run_dir / f"blocks/{ba.name}/capture.wav"
                    try:
                        cap, csr = read_take(wavp)
                        f_req = float(ba.clock.frame_of(float(bo.get("t_req") or 0.0)))
                        off, rho = BO.align(cap, ref, csr, int(f_req - 0.5 * csr),
                                            int(f_req + 2.5 * csr))
                        d = BO.diff(cap, ref, csr, off)
                        d["offset"], d["rho"] = off, round(rho, 3)
                        entry["vs_capture"] = d
                        for sp in d.get("spots", [])[:10]:
                            entry["problems"].append(
                                f"bounce vs capture differ at +{sp['frame'] / csr:.3f} s "
                                f"({sp['length'] / csr * 1000:.0f} ms, {sp['lower']} lower: "
                                f"{'the card write' if sp['lower'] == 'bounce' else 'USB IN'})")
                    except (OSError, ValueError) as e:
                        entry["vs_capture"] = {"ok": False, "why": str(e)}
        elif bo.get("state") == 2:
            entry["problems"].append("the clip was not fetched")
        entry["ok"] = not entry["problems"]
        out.append(entry)
    return out


def _coverage(script: dict[str, Any], actions: list[Action], glitches: list[dict[str, Any]],
              blocks: list[BlockAnalysis]) -> dict[str, dict[str, Any]]:
    from crosspad_hil.jam.script import all_features

    ran = {a.args.get("id") for a in actions if a.kind == "step" and a.ok}
    win_ok = {str(a.args.get("op")) for a in actions if a.kind == "window_op" and a.ok}
    bad_steps = {g.get("step") for g in glitches if g.get("status") == "glitch"}
    unmeasured = {b.name for b in blocks if b.not_measured}
    out: dict[str, dict[str, Any]] = {f: {"steps": [], "status": "not_run"}
                                      for f in all_features()}
    for blk in script.get("blocks", []):
        for s in blk.get("steps", []):
            for f in s.get("features", []):
                r = out.setdefault(f, {"steps": [], "status": "not_run"})
                r["steps"].append(s["id"])
                if s.get("skip"):
                    r.setdefault("skipped", []).append(f"{s['id']}: {s['skip']}")
                    continue
                if s["id"] in ran:
                    st = "failed" if s["id"] in bad_steps else (
                        "exercised" if blk["name"] in unmeasured else "verified")
                    if r["status"] in ("not_run", "verified") or st == "failed":
                        r["status"] = st if r["status"] != "failed" else "failed"
    for name, ops in (script.get("windows") or {}).items():
        for op in ops:
            for f in op.get("features", []):
                r = out.setdefault(f, {"steps": [], "status": "not_run"})
                r["steps"].append(f"{name}")
                if op.get("op", "")[:40] in {w[:40] for w in win_ok} and r["status"] == "not_run":
                    r["status"] = "exercised"
    return out


def analyze_run(run_dir: Path | str, *, tolerances: D.Tolerances | None = None,
                calibration: Calibration | None = None, snippets: int = 40,
                allow_partial: bool = False, write: bool = True,
                write_expected: bool = True) -> JamAnalysis:
    run_dir = Path(run_dir)
    tol = tolerances or D.Tolerances()
    script = _json(run_dir / "script.json", {})
    env = _json(run_dir / "env.json", {})
    cal = calibration or Calibration.from_dict(_json(run_dir / "calibration.json", {}))
    actions = load_actions(run_dir / "actions.jsonl")
    tel = Telemetry.load_tsv(run_dir / "console.tsv") if (run_dir / "console.tsv").exists() \
        else None
    block_docs = []
    for p in sorted((run_dir / "blocks").glob("*/block.json")):
        d = _json(p, None)
        if d:
            block_docs.append(d)
    order = {n: i for i, n in enumerate(script.get("order") or [])}
    block_docs.sort(key=lambda d: order.get(d["name"], 99))
    steps = []
    for a in actions:
        if a.kind == "step":
            steps.append({"id": a.args.get("id"), "block": a.block, "t0": a.t_mono,
                          "t1": a.t_mono + float(a.args.get("dur") or 0.0)})
    ja = JamAnalysis(run_dir=str(run_dir))
    gid = 0
    for blk in block_docs:
        try:
            ba = analyze_block(run_dir, blk, actions, tel, cal, tol, env, steps, snippets, gid,
                               write_expected=write_expected and write)
        except (OSError, ValueError) as e:
            ba = BlockAnalysis(name=blk["name"], seconds=0.0,
                               not_measured=f"the take could not be analysed: {e}")
            ba.counters = counter_findings(blk["name"], blk, tel, None)
        gid += len(ba.glitches)
        ja.blocks.append(ba)
    bounces = bounce_findings(run_dir, _json(run_dir / "bounces.json", []),
                              {b.name: b for b in ja.blocks})
    glitches = [g.to_dict() for b in ja.blocks for g in b.glitches]
    counters = [c for b in ja.blocks for c in b.counters]
    for rc in _json(run_dir / "recs.json", []):
        if rc.get("missing") or rc.get("extra"):
            counters.append({"block": rc.get("window"), "counter": "rec",
                             "delta": len(rc.get("missing") or []) + len(rc.get("extra") or []),
                             "note": f"{rc.get('window')}: the take of scene {rc.get('scene')} "
                                     f"holds {rc.get('got')} events, {rc.get('want')} played; "
                                     f"missing {rc.get('missing')[:6]} extra "
                                     f"{rc.get('extra')[:6]} (pad, pos)"})
    recorder = _recorder_probe(run_dir, env, {b.name: b for b in ja.blocks})
    if recorder.get("problems"):
        counters.append({"block": "B2", "counter": "recorder", "delta": 1,
                         "note": "Recorder vs capture: " + "; ".join(recorder["problems"])})
    for bo in bounces:
        if not bo["ok"]:
            counters.append({"block": bo.get("block"), "counter": f"bounce.{bo.get('name')}",
                             "delta": 1, "note": f"{bo.get('name')}: "
                                                 + "; ".join(bo["problems"])})
    # steps on each take's own time axis (for labels and the page)
    step_rows = []
    for b in ja.blocks:
        exp = b.expectation
        if exp is None:
            continue
        blk = next(d for d in block_docs if d["name"] == b.name)
        clock = CaptureClock.from_points([tuple(p) for p in blk.get("capture_points") or []],
                                         exp.sr, t_start=blk.get("t_capture_start"))
        for s in steps:
            if s["block"] == b.name:
                step_rows.append({"id": s["id"], "block": b.name,
                                  "wav_s": round(clock.frame_of(s["t0"]) / exp.sr, 4),
                                  "wav_s1": round(clock.frame_of(s["t1"]) / exp.sr, 4)})
    minutes: Counter[str] = Counter()
    for b in ja.blocks:
        if b.expectation is None:
            continue
        sr = b.expectation.sr
        minutes["(none)"] += b.seconds / 60.0
        for w in b.expectation.flags:
            minutes[w.kind] += max(0.0, min(w.f1, b.expectation.n) - w.f0) / sr / 60.0
    by_flag = A.flag_table([g for b in ja.blocks for g in b.glitches], dict(minutes))
    coverage = _coverage(script, actions, glitches, ja.blocks)
    missing = [f for f, r in coverage.items() if r["status"] == "not_run"]
    not_measured = [f"{b.name}: {b.not_measured}" for b in ja.blocks if b.not_measured]
    left = _json(run_dir / "restore.json", {})
    left_changed = [f"{k}: {v}" for k, v in (left.get("left") or {}).items()
                    if v not in ("restored", "unchanged", None)
                    and not str(v).startswith("removed")]
    status, code, lines = R.verdict(glitches, counters, not_measured, missing, allow_partial,
                                    left_changed)
    lat: dict[str, list[float]] = {}
    for b in ja.blocks:
        for k, v in (b.measured.get("latency_ms") or {}).items():
            lat.setdefault(k, []).append(v["p50"])
    ja.measured = {"l_hit_ms": {k: round(float(np.median(v)), 2) for k, v in lat.items()
                                if k == "host"},
                   "seq_latency_ms": (round(float(np.median(lat["seq"])), 2)
                                      if lat.get("seq") else None),
                   "level_offset_db": (round(float(np.median([b.measured.get(
                       "level_offset_db", 0.0) for b in ja.blocks if b.measured])), 2)
                       if ja.blocks else None)}
    if ja.measured["l_hit_ms"].get("host") is not None:
        ja.measured["l_hit_ms"] = {"sysex": ja.measured["l_hit_ms"]["host"]}
    gaps = _product_gaps(actions, glitches, ja.blocks)
    ja.doc = {
        "scenario": "live_jam", "run_dir": str(run_dir), "seed": script.get("seed"),
        "uac_seconds": sum(b.seconds for b in ja.blocks),
        "verdict": {"status": status, "exit": code, "lines": lines},
        "summary": {"glitches": len([g for g in glitches if g["status"] == "glitch"]),
                    "unverifiable": len([g for g in glitches if g["status"] != "glitch"]),
                    "by_class": dict(Counter(g["kind"] for g in glitches
                                             if g["status"] == "glitch")),
                    "by_cause": dict(Counter(g["cause"] for g in glitches
                                             if g["status"] == "glitch")),
                    "by_step": dict(Counter(str(g.get("step")) for g in glitches
                                            if g["status"] == "glitch")),
                    "counters": len(counters),
                    "blind": {b.name: {"unverifiable_voices": b.measured.get("unverifiable"),
                                       "judged_voices": b.measured.get("judged")}
                              for b in ja.blocks}},
        "glitches": glitches, "counters": counters, "bounces": bounces, "recorder": recorder,
        "blocks": [{"name": b.name, "seconds": round(b.seconds, 3), "measured": b.measured,
                    "pilot": b.pilot, "not_measured": b.not_measured,
                    "findings_by_class": dict(Counter(f.cls for f in b.findings)),
                    "findings_ok": dict(Counter(f.cls for f in b.findings
                                                if f.status in (D.OK, D.INFO))),
                    "runs": b.expectation.runs if b.expectation else [],
                    **b.lanes} for b in ja.blocks],
        "steps": step_rows, "by_flag": by_flag, "coverage": coverage,
        "measured": ja.measured, "product_gaps": gaps, "left": left,
        "tolerances": tol.to_dict(), "calibration": cal.to_dict(), "env": env,
    }
    if write:
        R.write_json(run_dir / REPORT, ja.doc)
        R.write_timeline_md(run_dir / "timeline.md", ja.doc)
        R.write_timeline_html(run_dir / "timeline.html", ja.doc)
        for b in ja.blocks:
            R.write_audacity_labels(run_dir / f"labels_{b.name}.txt", ja.doc, b.name)
    return ja


def _recorder_probe(run_dir: Path, env: dict[str, Any], blocks: dict[str, BlockAnalysis]
                    ) -> dict[str, Any]:
    """The Recorder's window (10 s of its take of B2) against B2's capture."""
    win = ((env.get("recorder") or {}).get("window") or {})
    local = win.get("local")
    if not local or not win.get("ok") or not Path(local).exists():
        return {"checked": False, "why": win.get("err") or "no window fetched"}
    ba = blocks.get("B2")
    if ba is None:
        return {"checked": False, "why": "no B2 analysis"}
    try:
        ref, sr = BO.read_wav(Path(local))
        cap, csr = read_take(run_dir / "blocks/B2/capture.wav")
    except (OSError, ValueError) as e:
        return {"checked": False, "why": str(e)}
    if sr != csr:
        return {"checked": False, "why": f"recorder {sr} Hz, capture {csr} Hz"}
    off, rho = BO.align(cap, ref, sr, 0, len(cap) - len(ref))
    d = BO.diff(cap, ref, sr, off)
    out = {"checked": True, "offset_s": round(off / sr, 3), "rho": round(rho, 3), **d,
           "problems": []}
    if rho < 0.5:
        out["problems"].append(f"the window does not line up with the capture (rho {rho:.2f})")
    for sp in d.get("spots", [])[:5]:
        out["problems"].append(f"+{(off + sp['frame']) / sr:.3f} s of B2: {sp['lower']} lower")
    return out


def _product_gaps(actions: list[Action], glitches: list[dict[str, Any]],
                  blocks: list[BlockAnalysis]) -> list[str]:
    out = []
    negs = [a for a in actions if a.kind == "note_neg"]
    if negs:
        sounded = [g for g in glitches if g["kind"] in ("GHOST", "SEQ_EXTRA") and any(
            abs(float((g.get("axes") or {}).get("t_host", -1e9)) - a.t_mono) < 0.5 for a in negs)]
        out.append("host MIDI NoteOn on the ESP port: " + (
            "SOUNDED -- SamplerPadLogic::onNoteOn is no longer empty" if sounded else
            "silent, as the firmware has it (SamplerPadLogic::onNoteOn is empty); a pad "
            "state stays set until the NoteOff, which can defer a kit swap up to 4 s"))
    out.append("a sequencer note is released on the next poll (~2 ms): a pitched pad in a "
               "scene is a ~100 ms blip, whatever its length")
    out.append("SWING is UI-only and not readable: not covered")
    for b in blocks:
        for r in (b.expectation.runs if b.expectation else []):
            if r.get("kind") == "start" and "fitted_shift_ms" not in r:
                out.append(f"{b.name}: after MIDI Start the sequencer's restart was not found "
                           "in the take (a burst or no restart: see SEQ findings there)")
    return out


def finite(x: float | None) -> float | None:
    return None if x is None or not math.isfinite(x) else x
