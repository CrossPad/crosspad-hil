"""Where a take departs from the model: one ``Finding`` per departure, on frames.

Tier A needs no model -- the pilot and timecode (``pilot_findings``), clicks in
the quiet bands (``hf_clicks``; the model only *explains* the ones it caused),
full-scale samples. Tier B holds the take against the voices the model
expects (``score_voices``: missing, late, holes, early cuts, wrong kit, level;
``ghost_scan``: sound where nothing should be; ``rec_findings``: what an
overdub recorded where).

How an onset is timed (``measure_onset``): the band's complex envelope around
the expected moment, its state before (``c0``: silence, or the same pad's
previous voice) and after (``c1``); the onset is where the envelope is half
way from one to the other. That reads a restart inside a sounding tone as well
as a start from silence -- a restarted sample comes back at phase zero, so
``c1`` differs from ``c0`` even at the same level -- and it is centred: a
symmetric window puts the half-way point at the middle of the sample's fade-in.
"""
from __future__ import annotations

import math
from collections import defaultdict
from dataclasses import asdict, dataclass, field
from typing import Any

import numpy as np

from crosspad_hil.jam import spec as S
from crosspad_hil.jam.bands import BandBank, BandLevels, local_envelope, mono
from crosspad_hil.jam.model import INF, Expectation, SeqProbe, Voice

GLITCH = "glitch"
OK = "ok"
INFO = "info"
UNVERIFIABLE = "unverifiable"

# Classes that fail the verdict when their status is "glitch".
GLITCH_CLASSES = ("MISSING", "LATE", "LATE_SWAP", "HOLE", "EARLY_CUT", "WRONG_KIT", "GHOST",
                  "CLICK", "PILOT_GAP", "SLIP", "CAPTURE_SLIP", "SEQ_MISS", "SEQ_EXTRA",
                  "REC_MISS", "REC_WRONG_POS", "BOUNCE", "COUNTER", "CLIP", "HUNG",
                  "CLICK_LEAK", "CHOKE_MISS", "HOLD_STUCK", "LEVEL_PAN", "FX_MISMATCH")


@dataclass
class Tolerances:
    late_ms: float = 10.0
    seq_late_ms: float = 8.0
    early_ms: float = 15.0
    late_swap_ms: float = 50.0
    late_swap_per_swap: int = 1
    strict: bool = False
    missing_db: float = -12.0
    hole_db: float = -10.0
    ghost_db: float = -35.0
    level_db: float = 3.0
    pan_db: float = -15.0
    click_db: float = 20.0
    click_abs_dbfs: float = -72.0
    click_explain_ms: float = 2.0
    early_cut_ms: float = 15.0
    long_ms: float = 30.0
    edge_ms: float = 12.0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class Finding:
    cls: str
    frame: float
    length: float = 0.0
    tier: str = "B"
    status: str = GLITCH
    vid: int | None = None
    aid: int | None = None
    cause: str = ""
    src: str = ""
    kit: str = ""
    pad: int | None = None
    band: float | None = None
    measured: Any = None
    expected: Any = None
    delta: Any = None
    note: str = ""
    evidence: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        for k in ("frame", "length", "band"):
            if isinstance(d.get(k), float):
                d[k] = round(d[k], 2)
        return d


def _db(a: float, b: float) -> float:
    return 20.0 * math.log10(max(a, 1e-12) / max(b, 1e-12))


# --------------------------------------------------------------------------- onsets

@dataclass
class Onset:
    frame: float | None
    level: float
    c0: complex
    c1: complex
    status: str            # found | missing | unverifiable
    why: str = ""


def measure_onset(xm: np.ndarray, sr: int, f: float, pred: float, amp_exp: float, *,
                  before_s: float = 0.04, after_s: float = 0.15, early_s: float = 0.012,
                  win_s: float = 0.016, fade_s: float = S.FADE_S,
                  prev_on: float | None = None, next_on: float | None = None,
                  restart: bool = False, prev_end: float | None = None) -> Onset:
    """Time a voice's start at ``f`` near ``pred`` (frames); see the module docstring.

    ``prev_on``/``prev_end``: the previous voice on this band. A restart reads the
    band's state after that voice settled; a fresh start begins from silence, and
    is looked for only once that voice has gone."""
    half_w = 0.5 * win_s * sr
    if restart:
        settled = -INF if prev_on is None else prev_on + half_w + fade_s * sr + 0.002 * sr
    else:
        settled = -INF if prev_end is None else prev_end + half_w
    lo = max(pred - early_s * sr, settled)
    hi = pred + after_s * sr
    if next_on is not None:
        hi = min(hi, next_on - half_w)
    if hi - lo < 0.004 * sr or hi < pred + 0.004 * sr:
        return Onset(None, 0.0, 0j, 0j, UNVERIFIABLE, "the next hit on this band comes "
                     "before this one can be read")
    # the band's state before: after the previous voice on it settled, before the
    # earliest this one may start
    c0_to = lo
    c0_from = max(settled, lo - 0.005 * sr)
    if c0_to - c0_from < 0.001 * sr:
        c0_from = c0_to - 0.001 * sr
    fr, amp, ph = local_envelope(xm, sr, f, pred, (pred - c0_from) / sr + 0.002,
                                 (hi - pred) / sr + 0.01, win_s=win_s)
    if len(fr) == 0:
        return Onset(None, 0.0, 0j, 0j, UNVERIFIABLE, "no audio around it")
    y = amp * np.exp(1j * ph)
    sel0 = (fr >= c0_from) & (fr <= c0_to)
    c0 = complex(np.mean(y[sel0])) if np.any(sel0) else 0j
    # whether the band already sounds is read, not assumed: the model's steal picks can
    # be wrong once a fault has moved a hit
    restart = abs(c0) >= 0.25 * amp_exp
    if not restart:
        c0 = 0j
    selw = (fr >= lo) & (fr <= hi)
    if not np.any(selw):
        return Onset(None, 0.0, c0, c0, UNVERIFIABLE, "window too short")
    yw, fw = y[selw], fr[selw]
    d = np.abs(yw - c0)
    hop = max(1.0, float(fr[1] - fr[0])) if len(fr) > 1 else 1.0
    span = max(2, int(0.004 * sr / hop))
    # the first change, and the state the band settles in after it -- not the biggest
    # change in the window: a hole later on must not be taken for this voice's start
    moved = np.flatnonzero(d >= 0.3 * amp_exp)
    k_big = None
    if len(moved):
        for k in range(int(moved[0]), len(yw) - span):
            seg = yw[k:k + span]
            lvl = abs(seg[0])
            # settled, far from where it was, and sounding (a hole is not a start)
            if float(np.abs(seg - seg[0]).max()) < 0.15 * max(lvl, amp_exp) \
                    and abs(seg[0] - c0) >= 0.3 * amp_exp and lvl >= 0.4 * amp_exp:
                k_big = k
                break
    if k_big is None:
        if restart and abs(c0) >= 0.5 * amp_exp:
            return Onset(None, abs(c0), c0, c0, UNVERIFIABLE,
                         "the restart left the band where it was (same phase) or never came")
        return Onset(None, float(np.abs(yw).max()), c0, c0, "missing",
                     "nothing started in the band")
    tail = yw[k_big:k_big + span]
    c1 = complex(np.median(tail.real) + 1j * np.median(tail.imag))
    jump = abs(c1 - c0)
    if jump < 0.3 * amp_exp:
        if restart and abs(c0) >= 0.5 * amp_exp:
            return Onset(None, abs(c0), c0, c1, UNVERIFIABLE,
                         "the restart left the band where it was (same phase) or never came")
        return Onset(None, abs(c1), c0, c1, "missing", "nothing started in the band")
    half = 0.5 * jump
    idx = np.flatnonzero(d[:k_big + 1] >= half)
    k = int(idx[0]) if len(idx) else k_big
    if k > 0:
        d0, d1 = d[k - 1], d[k]
        frac = (half - d0) / (d1 - d0) if d1 > d0 else 0.0
        t = fw[k - 1] + frac * (fw[k] - fw[k - 1])
    else:
        t = fw[0]
    bias = (0.5 if abs(c0) < 0.25 * amp_exp else 0.25) * fade_s * sr
    return Onset(float(t - bias), abs(c1), c0, c1, "found")


# --------------------------------------------------------------------------- per voice

@dataclass
class VoiceResult:
    vid: int
    status: str
    onset: float | None = None
    resid_ms: float | None = None
    level: float | None = None
    end: float | None = None
    freq: float | None = None


def _band_voices(voices: list[Voice], sr: int) -> dict[float, list[tuple[float, float]]]:
    """``{band: [(onset, gone)]}``, sorted: who else sounds in each band, and until when."""
    out: dict[float, list[tuple[float, float]]] = defaultdict(list)
    for v in voices:
        gone = v.f_end if v.f_end != INF else float("inf")
        if v.f_rel != INF and v.release_s:
            gone = min(gone, v.f_rel + v.release_s * sr)
        for f in (*v.freqs, *v.alt_freqs):
            out[round(f, 3)].append((v.f_on, gone))
    for k in out:
        out[k].sort()
    return out


def _neighbours(lst: list[tuple[float, float]], f: float
                ) -> tuple[float | None, float | None, float | None]:
    """(previous voice's onset, when it was gone, next voice's onset) around frame f."""
    import bisect

    ons = [a for a, _ in lst]
    i = bisect.bisect_left(ons, f - 1e-6)
    prev_on = lst[i - 1][0] if i > 0 else None
    prev_end = max((e for _a, e in lst[:i]), default=None) if i > 0 else None
    j = bisect.bisect_right(ons, f + 1e-6)
    nxt = ons[j] if j < len(ons) else None
    return prev_on, prev_end, nxt


def measure_voices(x: np.ndarray, exp: Expectation, tol: Tolerances
                   ) -> tuple[list[VoiceResult], dict[str, Any], list[Finding]]:
    """Pass 1: every voice the model vouches for, timed in the take (no judgement
    yet, except a voice nothing could be read for)."""
    sr = exp.sr
    results: list[VoiceResult] = []
    unread: list[Finding] = []
    band_on = _band_voices(exp.voices, sr)
    measured: dict[str, Any] = {"voices": len(exp.voices), "judged": 0, "unverifiable": 0}
    mono_cache: dict[tuple[bool, bool], np.ndarray] = {}

    def chan(v: Voice) -> np.ndarray:
        key = (v.gain_l > 2 * v.gain_r, v.gain_r > 2 * v.gain_l)
        if key not in mono_cache:
            mono_cache[key] = mono(x, v.gain_l if key[0] else 1.0, v.gain_r if key[1] else 1.0)
        return mono_cache[key]

    for v in exp.voices:
        if v.uncertain or not v.freqs or v.amp <= 0 or v.f_on < 0.05 * sr \
                or v.f_on > exp.n - 0.05 * sr:
            measured["unverifiable"] += 1
            results.append(VoiceResult(v.vid, UNVERIFIABLE))
            continue
        xm = chan(v)
        best: Onset | None = None
        best_f = v.freqs[0]
        win = 0.020 if v.engine == "pitched" else (0.012 if v.freqs[0] >= 4_000 else 0.016)
        late_s = v.late_ok_ms / 1000.0
        # a filter decides this voice's level (not modelled): time it against a low guess
        amp_exp = v.amp if v.level_known else v.amp * 10 ** (FILTER_GUESS_DB / 20.0)
        # never look past the voice's own sound: its end is a change in the band too
        own = (min(v.f_nat, v.f_end) - v.f_on) / sr if v.f_nat != INF else 1.0
        after = min(0.15 + late_s + v.tol_on / sr, max(0.01, 0.8 * own))
        for f in (*v.freqs, *v.alt_freqs):
            prev, prev_end, nxt = _neighbours(band_on.get(round(f, 3), []), v.f_on)
            on = measure_onset(xm, sr, f, v.f_on, amp_exp, win_s=win, fade_s=v.fade_on_s,
                               prev_on=prev, next_on=nxt, restart=v.restart, prev_end=prev_end,
                               after_s=after)
            if best is None or (on.status == "found" and (best.status != "found"
                                                          or on.level > best.level)):
                best, best_f = on, f
        assert best is not None
        if best.status == UNVERIFIABLE:
            measured["unverifiable"] += 1
            results.append(VoiceResult(v.vid, UNVERIFIABLE))
            unread.append(_f("MISSING" if v.src != "seq" else "SEQ_MISS", v, v.f_on,
                             status=UNVERIFIABLE, note=best.why))
            continue
        measured["judged"] += 1
        if best.status == "missing":
            results.append(VoiceResult(v.vid, "missing", level=best.level, freq=best_f))
            continue
        assert best.frame is not None
        results.append(VoiceResult(v.vid, "found", onset=best.frame,
                                   resid_ms=(best.frame - v.f_on) / sr * 1000.0,
                                   level=best.level, freq=best_f))
    return results, measured, unread


def judge_onsets(x: np.ndarray, exp: Expectation, results: list[VoiceResult],
                 measured: dict[str, Any], bank: BandBank, tol: Tolerances,
                 kits: dict[str, S.KitSpec]) -> list[Finding]:
    """Pass 1's verdicts: missing, wrong kit, late (against the path's own median),
    held by a swap, level, pan."""
    sr = exp.sr
    findings: list[Finding] = []
    by_vid = {v.vid: v for v in exp.voices}
    resid: dict[str, list[float]] = defaultdict(list)
    level_ratio: list[float] = []
    for res in results:
        v = by_vid[res.vid]
        if res.status == "found":
            resid[_src_key(v)].append(res.resid_ms or 0.0)
            if v.level_known and res.level:
                level_ratio.append(res.level / v.amp)
    med = {k: float(np.median(vs)) for k, vs in resid.items() if vs}
    measured["latency_ms"] = {k: {"p50": round(m, 2), "n": len(resid[k]),
                                  "p95": round(float(np.percentile(resid[k], 95)), 2),
                                  "max": round(float(np.max(resid[k])), 2)}
                              for k, m in med.items()}
    offset_db = _db(float(np.median(level_ratio)), 1.0) if level_ratio else 0.0
    measured["level_offset_db"] = round(offset_db, 2)
    late_swaps: dict[int, list[float]] = defaultdict(list)
    for res in results:
        v = by_vid[res.vid]
        if res.status == "missing":
            wk = _wrong_kit_at(bank, v, kits, v.f_on, sr)
            if wk is not None:
                findings.append(_f("WRONG_KIT", v, v.f_on, measured=wk[0],
                                   expected=list(v.freqs), note=f"the pad played {wk[1]}'s "
                                   "sample instead"))
            else:
                findings.append(_f("SEQ_MISS" if v.src == "seq" else "MISSING", v, v.f_on,
                                   measured=round(_db(res.level or 0.0, v.amp), 1),
                                   expected=0.0, note="nothing started in the band"))
            continue
        if res.status != "found":
            continue
        key = _src_key(v)
        d = (res.resid_ms or 0.0) - med.get(key, 0.0)
        limit = tol.seq_late_ms if v.src == "seq" else tol.late_ms
        if v.late_ok_ms > 0 and d <= limit:
            late_swaps[_swap_of(v)].append(v.late_ok_ms)
        if d > limit:
            findings.append(_f("LATE", v, res.onset or v.f_on, measured=round(d, 2),
                               expected=f"<= {limit:g} ms after the path's median "
                                        f"({med.get(key, 0.0):.1f} ms)",
                               delta=round(d, 2), note=f"{key}"))
        elif d < -tol.early_ms:
            findings.append(_f("LATE", v, res.onset or v.f_on, measured=round(d, 2),
                               expected=f">= -{tol.early_ms:g} ms", delta=round(d, 2),
                               note="early: the voice started before it was asked for"))
        if v.level_known and res.level is not None:
            ld = _db(res.level, v.amp) - offset_db
            if abs(ld) > tol.level_db:
                findings.append(_f("LEVEL", v, res.onset or v.f_on, status=INFO,
                                   measured=round(ld, 1), expected=0.0,
                                   note=f"{ld:+.1f} dB against the block's level"))
        if (v.gain_l > 0 and v.gain_r == 0) or (v.gain_r > 0 and v.gain_l == 0):
            findings.extend(_pan_check(x, sr, v, res, tol))
        if v.feature == "lp" and res.level is not None and res.level > v.amp * 10 ** (-6 / 20):
            findings.append(_f("FX_MISMATCH", v, res.onset or v.f_on,
                               measured=round(_db(res.level, v.amp), 1), expected="<= -6 dB",
                               note="the low-pass pad came through unfiltered"))
    for swap, lates in late_swaps.items():
        allowed = 0 if tol.strict else tol.late_swap_per_swap
        worst = max(lates)
        if len(lates) > allowed or worst > tol.late_swap_ms:
            vs = [v for v in exp.voices if _swap_of(v) == swap and v.late_ok_ms > 0]
            findings.append(_f("LATE_SWAP", vs[0], vs[0].f_on, measured=len(lates),
                               expected=f"<= {allowed} held hit(s) per swap, "
                                        f"<= {tol.late_swap_ms:g} ms",
                               delta=round(worst, 1),
                               note=f"{len(lates)} hit(s) held by the reload, worst "
                                    f"{worst:.0f} ms"))
    measured["late_swaps"] = {str(k): v for k, v in late_swaps.items()}
    return findings


def observed_onsets(exp: Expectation, results: list[VoiceResult]) -> dict[str, float]:
    """{cause: measured onset} for every voice the take timed."""
    by_vid = {v.vid: v for v in exp.voices}
    return {by_vid[r.vid].cause: float(r.onset) for r in results
            if r.status == "found" and r.onset is not None}


def restart_causes(exp: Expectation, results: list[VoiceResult]) -> set[str]:
    by_vid = {v.vid: v for v in exp.voices}
    return {by_vid[r.vid].cause for r in results if r.status == "found" and by_vid[r.vid].restart}


def latency_shift(measured: dict[str, Any]) -> dict[str, float]:
    return {k: float(v["p50"]) for k, v in (measured.get("latency_ms") or {}).items()}


def judge_bodies(exp: Expectation, bank: BandBank, tol: Tolerances,
                 found: dict[str, float]) -> tuple[list[Finding], dict[int, float]]:
    """Pass 2, on the expectation rebuilt from the observed onsets: holes inside every
    voice that started, and whether it ended where an engine can end it. Returns the
    findings and each voice's end as read from the take."""
    out: list[Finding] = []
    ends: dict[int, float] = {}
    sr = exp.sr
    moments = _end_moments(exp, found)
    band_on = _band_voices(exp.voices, sr)
    for v in exp.voices:
        if v.uncertain or v.cause not in found or not v.freqs:
            continue
        for f in v.freqs:
            lv = bank.levels(f)
            if lv is None:
                continue
            _p, _e, nxt = _neighbours(band_on.get(round(f, 3), []), v.f_on)
            fs, end = _body(lv, sr, v, found[v.cause], tol, exp,
                            moments.get(v.engine, []), f, next_on=nxt)
            out.extend(fs)
            if end is not None:
                ends[v.vid] = end
            break
    return out, ends


def score_voices(x: np.ndarray, exp: Expectation, bank: BandBank, tol: Tolerances,
                 kits: dict[str, S.KitSpec] | None = None, rebuild: Any = None
                 ) -> tuple[list[Finding], dict[str, Any], list[VoiceResult]]:
    """Both passes. ``rebuild(observed) -> Expectation`` re-runs the model on the
    measured onsets (without it the first expectation is judged as it is)."""
    kits = kits or S.jam_kits()
    results, measured, unread = measure_voices(x, exp, tol)
    findings = unread + judge_onsets(x, exp, results, measured, bank, tol, kits)
    found = observed_onsets(exp, results)
    exp2 = rebuild(found, restart_causes(exp, results), latency_shift(measured)) \
        if rebuild is not None else exp
    body, ends = judge_bodies(exp2, bank, tol, found)
    findings += body
    measured["rebuilt"] = rebuild is not None
    measured["ends"] = ends
    measured["steal_moments"] = [f for fs_ in _end_moments(exp2, found).values() for f in fs_]
    return findings, measured, results


def _swap_of(v: Voice) -> int:
    return int(v.held_by or 0)


def _src_key(v: Voice) -> str:
    if v.src == "seq":
        return "seq"
    if v.src == "preview":
        return "preview"
    return "host"


def _f(cls: str, v: Voice, frame: float, **kw: Any) -> Finding:
    return Finding(cls=cls, frame=float(frame), vid=v.vid, aid=v.aid, cause=v.cause,
                   src=v.src, kit=v.kit, pad=v.pad, band=v.freqs[0] if v.freqs else None,
                   tier="B", **kw)


def _pan_check(x: np.ndarray, sr: int, v: Voice, res: VoiceResult,
               tol: Tolerances) -> list[Finding]:
    if res.onset is None or res.freq is None or x.ndim < 2:
        return []
    weak = 1 if v.gain_l >= v.gain_r else 0
    strong = 1 - weak
    mid = res.onset + 0.03 * sr
    _f0, a_w, _p = local_envelope(x[:, weak], sr, res.freq, mid, 0.005, 0.005, win_s=0.012)
    _f1, a_s, _q = local_envelope(x[:, strong], sr, res.freq, mid, 0.005, 0.005, win_s=0.012)
    if not len(a_w) or not len(a_s):
        return []
    d = _db(float(np.median(a_w)), float(np.median(a_s)))
    if d > tol.pan_db:
        return [_f("LEVEL_PAN", v, res.onset, measured=round(d, 1),
                   expected=f"<= {tol.pan_db:g} dB in the far channel",
                   note="the pan did not take the voice off the other side")]
    return []


def _end_moments(exp: Expectation, found: dict[str, float]) -> dict[str, list[float]]:
    """Per engine, every moment a voice may have been taken by another: each start
    while the engine was at (or one under) its limit."""
    sr = exp.sr
    out: dict[str, list[float]] = {"sampler": [], "pitched": []}
    limit = {"sampler": 7, "pitched": 9}
    for v in exp.voices:
        on = found.get(v.cause, v.f_on)
        n = sum(1 for o in exp.voices if o.engine == v.engine and o.vid != v.vid
                and o.f_on <= on < o.f_end + 0.002 * sr)
        if n >= limit.get(v.engine, 7):
            out.setdefault(v.engine, []).append(on)
    for k in out:
        out[k].sort()
    return out


def _near(sorted_list: list[float], f: float, lim: float) -> bool:
    import bisect

    i = bisect.bisect_left(sorted_list, f - lim)
    return i < len(sorted_list) and sorted_list[i] <= f + lim


def _body(lv: BandLevels, sr: int, v: Voice, on: float, tol: Tolerances, exp: Expectation,
          moments: list[float], freq: float, next_on: float | None = None
          ) -> tuple[list[Finding], float | None]:
    """Holes inside one voice and whether its end (read from the take) is one the
    engines can make. Returns (findings, the measured end)."""
    out: list[Finding] = []
    env = lv.both(freq)
    edge = tol.edge_ms / 1000.0 * sr + lv.win / 2
    model_end = v.half_end(sr)
    restart_end = v.end_kind in ("retrigger", "toggle_off", "retrigger_release")
    longest = model_end if restart_end else max(model_end, min(v.f_nat, exp.n))
    a = lv.col(on + edge)
    b = min(len(env) - 1, lv.col(min(exp.n, longest + (0.0 if restart_end else 0.25 * sr))))
    if next_on is not None:
        # the next voice in this band: what sounds after it is not this one
        b = min(b, lv.col(next_on - lv.win / 2))
    if b - a < 3:
        return out, None
    # the voice's own level: just after it settled (later it may have been cut), and
    # only while it should still be at it (a 2 ms sequencer blip is all release)
    steady_to = min(b, a + max(3, int(0.04 * sr / lv.hop)), lv.col(model_end - edge))
    if steady_to - a < 3:
        return out, None
    ref = float(np.median(env[a:steady_to]))
    if ref <= 0:
        return out, None
    meas_end: float | None = None
    # where the voice stopped, if it did before the model's end: an end stays ended,
    # a dip the voice comes back from is a hole (a restart bounds the search)
    b_end = lv.col(model_end - edge) if restart_end else b
    if b_end - a >= 3:
        below = env[a:b_end] < 0.5 * ref
        gone = max(2, int(END_HOLD_S * sr / lv.hop))
        for s0, e0 in _runs(below):
            if e0 - s0 >= gone or a + e0 >= b_end:
                meas_end = float(lv.frame(a + s0))
                break
    hole_to = lv.col(min(model_end, meas_end if meas_end is not None else model_end) - edge)
    if hole_to - a >= 3:
        seg = env[a:hole_to]
        thr = ref * 10 ** (tol.hole_db / 20.0)
        for s0, e0 in _runs(seg < thr):
            f0 = float(lv.frame(a + s0))
            if exp.uncertain_at(f0) is not None:
                continue
            depth = _db(float(seg[s0:e0].min()), ref)
            out.append(_f("HOLE", v, f0, length=float((e0 - s0) * lv.hop),
                          measured=round(depth, 1), expected=f">= {tol.hole_db:g} dB",
                          note=f"{(e0 - s0) * lv.hop / sr * 1000:.0f} ms under the voice's "
                               "own level"))
    if v.end_uncertain or meas_end is None or (v.f_end == INF and not restart_end) \
            or v.end_kind == "block_end":
        return out, meas_end
    if restart_end:
        # gone before its own restart: only another voice taking it may do that
        if _near(moments, meas_end, 0.005 * sr + lv.hop):
            out.append(_f("STEAL_DIFFERS", v, meas_end, status=INFO,
                          note="taken by another voice before its restart"))
        else:
            out.append(_f("EARLY_CUT", v, meas_end,
                          measured=round((meas_end - model_end) / sr * 1000.0, 1),
                          note="the voice stopped before its restart, with no voice "
                               "starting to take it"))
        return out, meas_end
    d_model = (meas_end - model_end) / sr * 1000.0
    nat = (min(v.f_nat, exp.n) - 0.5 * S.FADE_S * sr) if v.f_nat != INF else None
    steal_ok = _near(moments, meas_end, 0.005 * sr + lv.hop)
    if abs(d_model) <= tol.early_cut_ms:
        if v.end_kind in ("stolen", "choked"):
            out.append(_f("STOLEN" if v.end_kind == "stolen" else "CHOKED", v, meas_end,
                          status=OK, measured=round(d_model, 1),
                          note="ended where the engine takes it"))
        return out, meas_end
    if steal_ok:
        out.append(_f("STEAL_DIFFERS", v, meas_end, status=INFO, measured=round(d_model, 1),
                      note="ended at another voice's start with the engine at its limit: "
                           "a steal the model gave to another voice"))
        return out, meas_end
    if nat is not None and abs(meas_end - nat) <= tol.early_cut_ms / 1000.0 * sr:
        if v.end_kind == "choked":
            out.append(_f("CHOKE_MISS", v, model_end, measured=round(d_model, 1),
                          note="the choke group did not cut this voice"))
        elif v.end_kind == "released" and v.shape == "hold":
            out.append(_f("HOLD_STUCK", v, model_end, measured=round(d_model, 1),
                          note="a hold pad kept sounding after its release"))
        elif v.end_kind == "stolen":
            out.append(_f("STEAL_DIFFERS", v, model_end, status=INFO,
                          measured=round(d_model, 1),
                          note="not stolen: the engine took another voice"))
        return out, meas_end
    if d_model < 0:
        out.append(_f("EARLY_CUT", v, meas_end, measured=round(d_model, 1),
                      expected=f"ends at {model_end / sr:.3f} s ({v.end_kind})",
                      note=f"the voice stopped {-d_model:.0f} ms before it should have, "
                           "with no voice starting to take it"))
    elif v.end_kind == "choked":
        out.append(_f("CHOKE_MISS", v, model_end, measured=round(d_model, 1),
                      note="the choke group did not cut this voice"))
    elif v.shape == "hold" and v.end_kind == "released":
        out.append(_f("HOLD_STUCK", v, model_end, measured=round(d_model, 1),
                      note="a hold pad kept sounding after its release"))
    return out, meas_end


END_HOLD_S = 0.040
FILTER_GUESS_DB = -40.0


def _wrong_kit_at(bank: BandBank, v: Voice, kits: dict[str, S.KitSpec], f: float,
                  sr: int) -> tuple[float, str] | None:
    for name, kit in kits.items():
        if name == v.kit or kit.engine != "sampler" or v.engine != "sampler":
            continue
        for g in S.heard(kit, v.pad, sr).freqs:
            lv = bank.levels(g)
            if lv is None:
                continue
            env = lv.both(g)
            c_on = lv.col(f + 0.03 * sr)
            c_pre = lv.col(f - 0.03 * sr)
            if 0 <= c_pre < len(env) and 0 <= c_on < len(env) and \
                    env[c_on] > 0.3 * v.amp and env[c_on] > 4 * env[c_pre]:
                return g, name
    return None


def _runs(mask: np.ndarray) -> list[tuple[int, int]]:
    m = np.concatenate(([False], np.asarray(mask, dtype=bool), [False]))
    d = np.diff(m.astype(np.int8))
    return list(zip(np.flatnonzero(d == 1).tolist(), np.flatnonzero(d == -1).tolist(),
                    strict=True))


# --------------------------------------------------------------------------- clicks

QUIET_TAPS = 255


def hf_clicks(x: np.ndarray, sr: int, exp: Expectation | None, tol: Tolerances,
              explain: list[float] | None = None) -> list[Finding]:
    """Energy in the quiet bands far over their floor: a discontinuity somewhere.
    Each is explained by an edge the model expects within +-2 ms, or reported."""
    from scipy.signal import firwin, oaconvolve

    xs = np.asarray(x, dtype=np.float64)
    if xs.ndim == 1:
        xs = xs[:, None]
    env = np.zeros(xs.shape[0])
    for lo, hi in S.QUIET_BANDS:
        if hi >= 0.48 * sr:
            continue
        h = firwin(QUIET_TAPS, [lo, hi], pass_zero=False, fs=sr)
        for c in range(xs.shape[1]):
            env = np.maximum(env, np.abs(oaconvolve(xs[:, c], h, mode="same")))
    from scipy.ndimage import maximum_filter1d

    w = max(1, int(0.0005 * sr))
    if w > 1:
        env = maximum_filter1d(env, size=w)
    sec = sr
    nblk = max(1, len(env) // sec)
    floors = np.array([np.percentile(env[i * sec:(i + 1) * sec], 10) for i in range(nblk)])
    floor = np.repeat(floors, sec)
    if len(floor) < len(env):
        floor = np.concatenate([floor, np.full(len(env) - len(floor), floors[-1])])
    thr = np.maximum(floor * 10 ** (tol.click_db / 20.0), 10 ** (tol.click_abs_dbfs / 20.0))
    hot = env > thr
    lim0 = tol.click_explain_ms / 1000.0 * sr
    spans = sorted([(f - lim0, f + lim0) for f in (explain or [])]
                   + ([(e.frame - max(lim0, e.tol_ms / 1000.0 * sr),
                        e.frame + max(lim0, e.tol_ms / 1000.0 * sr)) for e in exp.edges]
                      if exp is not None else []))
    starts = [a for a, _ in spans]
    reach = []
    m = -math.inf
    for _a, b2 in spans:
        m = max(m, b2)
        reach.append(m)
    off = exp.pilot_off if exp is not None else []
    import bisect

    out: list[Finding] = []
    gap = int(0.002 * sr)
    runs = _runs(hot)
    merged: list[list[int]] = []
    for s, e in runs:
        if merged and s - merged[-1][1] <= gap:
            merged[-1][1] = e
        else:
            merged.append([s, e])
    lim = lim0
    for s, e in merged:
        k = s + int(np.argmax(env[s:e]))
        i = bisect.bisect_right(starts, k) - 1
        explained = (i >= 0 and reach[i] >= k) or any(w.covers(k, lim) for w in off)
        unc = exp.uncertain_at(k) if exp is not None else None
        lvl = 20 * math.log10(max(float(env[k]), 1e-12))
        if explained:
            continue
        out.append(Finding(cls="CLICK", frame=float(k), length=float(e - s), tier="A",
                           status=UNVERIFIABLE if unc is not None else GLITCH,
                           measured=round(lvl, 1),
                           expected=f"<= {tol.click_db:g} dB over the quiet bands' floor",
                           note="energy in the quiet bands: a discontinuity"
                                + (f" (inside {unc.kind})" if unc is not None else "")))
    return out


def clip_scan(x: np.ndarray, sr: int) -> list[Finding]:
    xs = np.abs(np.asarray(x))
    if xs.ndim > 1:
        xs = xs.max(axis=1)
    hot = xs >= 32766.0 / 32768.0
    out = []
    for s, e in _runs(hot)[:50]:
        out.append(Finding(cls="CLIP", frame=float(s), length=float(e - s), tier="A",
                           measured="full scale", note="the bus clipped (no limiter)"))
    return out


# --------------------------------------------------------------------------- the stream

def pilot_findings(x: np.ndarray, sr: int, spec: S.PilotSpec, exp: Expectation | None,
                   tol: Tolerances) -> tuple[list[Finding], dict[str, Any]]:
    """Holes and slips of the host's stream (Tier A)."""
    from crosspad_hil.jam import pilot as P

    xm = np.asarray(x[:, 0] if np.ndim(x) > 1 else x, dtype=np.float64)
    info: dict[str, Any] = {}
    out: list[Finding] = []
    off = exp.pilot_off if exp is not None else []

    def expected_off(f: float, pad: float = 0.0) -> bool:
        return any(w.covers(f, pad) for w in off)

    tc = P.timecode_delay(xm, sr, spec)
    info["timecode"] = tc.to_dict()
    resid = P.residual_gaps(xm, sr, spec, tc) if spec.timecode else {"exact": False}
    info["residual"] = {k: v for k, v in resid.items() if k != "gaps"}
    noise_frames = [g["frame"] for g in resid.get("gaps", []) if g["kind"] == "noise"]
    if resid.get("exact"):
        for g in resid["gaps"]:
            if g["kind"] != "gap" or expected_off(g["frame"], 0.002 * sr):
                continue
            out.append(Finding(cls="PILOT_GAP", frame=float(g["frame"]),
                               length=float(g["length"]), tier="A",
                               measured=f"{g['length'] / sr * 1000:.2f} ms",
                               note="the host's stream is missing from the take here "
                                    "(subtracted to the sample)",
                               evidence={"corr": g["corr"]}))
    else:
        tr = P.pilot_track(xm, sr, spec)
        info["pilot"] = tr.to_dict()
        for e in tr.events:
            if e.kind != "gap" or expected_off(e.frame, 0.002 * sr):
                continue
            out.append(Finding(cls="PILOT_GAP", frame=float(e.frame), length=float(e.length),
                               tier="A", measured=f"{e.depth_db:.0f} dB",
                               note="the pilot dropped out (demodulated)"))
    for st in tc.steps:
        f = float(st["frame"])
        if expected_off(f, 0.05 * sr) or _between_off(off, st["between"]):
            continue
        near = [k for k in noise_frames if abs(k - f) < 0.012 * sr]
        if near:
            f = float(min(near, key=lambda k: abs(k - f)))
        out.append(Finding(cls="SLIP", frame=f, tier="A", measured=int(st["samples"]),
                           expected=0, delta=int(st["samples"]),
                           note=f"{abs(st['samples'])} sample(s) "
                                f"{'lost' if st['samples'] < 0 else 'repeated'} between the "
                                "host and the take",
                           evidence={"from": st["from"], "to": st["to"]}))
    return out, info


def _between_off(off: list[Any], between: list[int]) -> bool:
    a, b = between
    return any(w.f0 <= b and w.f1 >= a for w in off)


def classify_slips(x: np.ndarray, sr: int, exp: Expectation, slips: list[Finding]) -> None:
    """A slip the pads share is the capture's (CAPTURE_SLIP); one only the stream
    shows is the stream's. Read from the phase of a pad tone sounding across it."""
    for s in slips:
        k = int(s.measured or 0)
        f = s.frame
        across = [v for v in exp.voices if not v.uncertain and v.f_on + 0.03 * sr < f
                  < v.half_end(sr) - 0.03 * sr and v.shape in ("flat", "hold", "loop")]
        if not across or not k:
            s.evidence["pads"] = "no pad tone across it: the stream's or the capture's"
            continue
        v = max(across, key=lambda v: v.amp)
        xm = mono(x, v.gain_l, v.gain_r)
        f0 = v.freqs[0]
        _a, _b, ph_b = local_envelope(xm, sr, f0, f - 0.012 * sr, 0.004, 0.004)
        _c, _d, ph_a = local_envelope(xm, sr, f0, f + 0.012 * sr, 0.004, 0.004)
        if not len(ph_b) or not len(ph_a):
            continue
        d = float(np.angle(np.exp(1j * (np.median(ph_a) - np.median(ph_b)))))
        want = float(np.angle(np.exp(-1j * 2 * np.pi * f0 * k / sr)))
        shared = abs(float(np.angle(np.exp(1j * (d - want))))) < 0.4 and abs(want) > 0.3
        s.evidence["pad_phase_jump"] = round(d, 3)
        s.evidence["pad_phase_if_shared"] = round(want, 3)
        if shared:
            s.cls = "CAPTURE_SLIP"
            s.note = (f"{abs(k)} sample(s) lost on the way to the host: the pads "
                      "and the stream jumped together")


# --------------------------------------------------------------------------- nothing there

SPLATTER_HZ = 600.0
NEIGHBOUR_HZ = 100.0         # a steady tone's 40 ms sidelobes: under -42 dB from 4 bins on


def ghost_scan(bank: BandBank, exp: Expectation, tol: Tolerances, ref_amp: float = S.PEAK,
               kits: dict[str, S.KitSpec] | None = None,
               seq_pads: set[int] | None = None,
               ends: dict[int, float] | None = None) -> list[Finding]:
    """Sound in a band of the plan where no voice is expected (40 ms windows)."""
    lv = bank.ghost
    if lv is None:
        return []
    sr = exp.sr
    kits = kits or S.jam_kits()
    owner: dict[float, tuple[str, int]] = {}
    for name, kit in kits.items():
        for pad in range(S.PADS):
            h = S.heard(kit, pad, sr)
            for f in (*h.freqs, *h.absent_freqs):
                owner.setdefault(round(f, 3), (name, pad))
    thr = ref_amp * 10 ** (tol.ghost_db / 20.0)
    ncol = lv.left.shape[1]
    out: list[Finding] = []
    edge_cols = [lv.col(e.frame) for e in exp.edges]
    for f in lv.freqs:
        busy = np.zeros(ncol, dtype=bool)
        for v in exp.voices:
            fs = (*v.freqs, *v.alt_freqs)
            near = min((abs(g - f) for g in fs), default=INF)
            end = v.f_end if v.f_end != INF else exp.n
            if ends and v.vid in ends:
                end = max(end, ends[v.vid])
            if near < NEIGHBOUR_HZ or (v.uncertain and near < 400.0):
                a = max(0, lv.col(v.f_on - 0.04 * sr))
                b = min(ncol, lv.col(end + 0.08 * sr) + 1)
                busy[a:b] = True
            elif near < SPLATTER_HZ:
                # a fade's splatter reaches the neighbours for a window's length
                for t0, t1 in ((v.f_on - 0.03 * sr, v.f_on + 0.035 * sr),
                               (v.half_end(sr) - 0.03 * sr, end + 0.04 * sr)):
                    busy[max(0, lv.col(t0)):min(ncol, lv.col(t1) + 1)] = True
        for w in exp.uncertain:
            busy[max(0, lv.col(w.f0) - 8):min(ncol, lv.col(w.f1) + 8)] = True
        for c in edge_cols:
            busy[max(0, c - 8):min(ncol, c + 9)] = True
        env = lv.both(f)
        hot = (env > thr) & ~busy
        for s, e in _runs(hot):
            if e - s < 2:
                continue
            kit, pad = owner.get(round(f, 3), ("", -1))
            cls = "GHOST"
            note = f"{_db(float(env[s:e].max()), ref_amp):.0f} dB in {kit or '?'} pad {pad}'s band"
            if f in [round(c, 3) for c in S.CLICK_HZ]:
                cls, note = "CLICK_LEAK", "the metronome is in the take"
            elif seq_pads and pad in seq_pads:
                cls, note = "SEQ_EXTRA", "a sequencer pad sounded where no event was due"
            out.append(Finding(cls=cls, frame=float(lv.frame(s)), length=float((e - s) * lv.hop),
                               tier="B", kit=kit, pad=pad if pad >= 0 else None, band=f,
                               measured=round(_db(float(env[s:e].max()), ref_amp), 1),
                               expected=f"<= {tol.ghost_db:g} dB", note=note))
    return out


# --------------------------------------------------------------------------- overdub

def rec_findings(exp: Expectation, dump_before: list[dict[str, int]],
                 dump_after: list[dict[str, int]], spb: int, per: int,
                 frames: dict[int, float]) -> list[Finding]:
    """Each overdubbed hit against where the board says it put it."""
    out: list[Finding] = []
    before = {(e["seq"], e["pad"], e["pos"]) for e in dump_before}
    added: dict[tuple[int, int], list[int]] = defaultdict(list)
    for e in dump_after:
        if (e["seq"], e["pad"], e["pos"]) not in before:
            added[(e["seq"], e["pad"])].append(e["pos"])
    grid = spb // per if per else spb
    for r in exp.rec_events:
        if not r.get("recorded", True):
            continue
        key = (r["scene"], r["pad"])
        cands = added.get(key, [])
        want = [r["pos"]] + ([r["alt"]] if r.get("alt") is not None else [])
        hit = next((p for p in want if p in cands), None)
        frame = frames.get(r["aid"], 0.0)
        if hit is not None:
            cands.remove(hit)
            continue
        near = [p for p in cands if abs(p - r["pos"]) <= grid]
        if near:
            out.append(Finding(cls="REC_WRONG_POS", frame=frame, aid=r["aid"], pad=r["pad"],
                               tier="B", measured=near[0], expected=want, delta=near[0] - r["pos"],
                               note="the overdub recorded the hit on another grid line"))
            cands.remove(near[0])
        else:
            out.append(Finding(cls="REC_MISS", frame=frame, aid=r["aid"], pad=r["pad"], tier="B",
                               expected=want, note="the overdub did not record the hit"))
    return out


# --------------------------------------------------------------------------- the seq fit

class BandProbe(SeqProbe):
    """``SeqProbe`` over a take: rising edges in the bands' fine levels."""

    def __init__(self, bank: BandBank, sr: int, amp: float = S.PEAK * 0.3) -> None:
        self.bank = bank
        self.sr = sr
        self.amp = amp
        self._cache: dict[float, np.ndarray] = {}

    def onsets(self, f: float) -> np.ndarray:
        key = round(f, 3)
        if key in self._cache:
            return self._cache[key]
        lv = self.bank.levels(f)
        if lv is None:
            self._cache[key] = np.zeros(0)
            return self._cache[key]
        env = lv.both(f)
        hi = env >= self.amp
        lo = env < 0.5 * self.amp
        out = []
        armed = True
        quiet = 0
        need = max(1, int(0.008 * self.sr / lv.hop))
        for i in range(len(env)):
            if lo[i]:
                quiet += 1
                if quiet >= need:
                    armed = True
            else:
                quiet = 0
            if armed and hi[i]:
                # half of the local peak: a symmetric window puts it mid-fade
                seg = env[max(0, i - need):i + 3 * need]
                peak = float(seg.max()) if len(seg) else float(env[i])
                j = max(0, i - need)
                while j < i and env[j] < 0.5 * peak:
                    j += 1
                if j > 0 and env[j] > env[j - 1]:
                    frac = (0.5 * peak - env[j - 1]) / (env[j] - env[j - 1])
                    t = float(lv.frame(j - 1)) + frac * lv.hop
                else:
                    t = float(lv.frame(j))
                out.append(t - 0.5 * S.FADE_S * self.sr)
                armed = False
        self._cache[key] = np.asarray(out)
        return self._cache[key]

    def fit(self, freqs: list[tuple[float, ...]], rel: list[float], lo: float, hi: float,
            prior: float | None = None) -> float | None:
        if not rel:
            return prior
        tol = 0.012 * self.sr
        cand: list[float] = []
        for fs, r in zip(freqs[:6], rel[:6], strict=False):
            for f in fs:
                on = self.onsets(f)
                sel = on[(on >= lo + r - tol) & (on <= hi + r + tol)]
                cand.extend((sel - r).tolist())
        if not cand:
            return None
        best, best_score = None, -1.0
        for c in cand:
            score = 0
            res = []
            for fs, r in zip(freqs, rel, strict=False):
                hits = [float(np.min(np.abs(self.onsets(f) - (c + r)))) for f in fs
                        if len(self.onsets(f))]
                if hits and min(hits) <= tol:
                    score += 1
                    res.append(min(hits))
            closeness = -abs(c - (prior if prior is not None else c)) / self.sr
            s = score + 0.01 * closeness
            if s > best_score:
                best, best_score = c, s
        if best is None or best_score < max(2.0, 0.5 * len(rel)) - 0.5:
            return None
        resid = []
        for fs, r in zip(freqs, rel, strict=False):
            for f in fs:
                on = self.onsets(f)
                if len(on):
                    d = on - (best + r)
                    k = int(np.argmin(np.abs(d)))
                    if abs(d[k]) <= tol:
                        resid.append(float(d[k]))
        return float(best + (np.median(resid) if resid else 0.0))
