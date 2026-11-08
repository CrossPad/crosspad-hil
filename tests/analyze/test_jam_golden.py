"""The jam's analysis against takes with a known answer.

A short set is played the way the scenario plays it (``jam.synth``), rendered
from the model, and broken on purpose: each fault must come back as exactly one
glitch, of the right class, within 5 ms of where it was put, named with the
step it happened in and the hit it belongs to. A clean take must come back
clean -- for several seeds and at 48 kHz, where every pad tone moves.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from crosspad_hil.jam import analysis, synth
from crosspad_hil.jam.model import Expectation, Voice
from crosspad_hil.jam.render import Fault

MS = 5.0


def pick(exp: Expectation, sr: int, src: str, t0: float, t1: float, *, pad: int | None = None,
         restart: bool | None = None) -> Voice:
    for v in exp.voices:
        if v.src == src and t0 * sr < v.f_on < t1 * sr and not v.uncertain \
                and (pad is None or v.pad == pad) and (restart is None or v.restart == restart):
            return v
    raise AssertionError(f"no {src} voice in {t0}..{t1} s")


def run(tmp_path: Path, **kw) -> tuple[dict, analysis.JamAnalysis]:
    info = synth.synth_run(tmp_path / "run", **kw)
    ja = analysis.analyze_run(tmp_path / "run", snippets=4, allow_partial=True)
    return info, ja


def glitches(ja: analysis.JamAnalysis) -> list[dict]:
    return [g for g in ja.doc["glitches"] if g["status"] == "glitch"]


@pytest.mark.parametrize("seed,sr", [(1, 44_100), (2, 44_100), (3, 44_100), (1, 48_000)])
def test_a_clean_jam_is_a_zero_glitch_pass(tmp_path: Path, seed: int, sr: int) -> None:
    _info, ja = run(tmp_path, seed=seed, sr=sr)
    assert glitches(ja) == [] and ja.doc["counters"] == []
    assert ja.status == "ZERO-GLITCH PASS" and ja.exit_code == 0
    b = ja.doc["blocks"][0]
    assert b["measured"]["judged"] > 200                        # most voices are judged
    assert b["pilot"]["residual"]["exact"] is True              # the stream subtracts exactly
    assert b["findings_ok"].get("STOLEN", 0) > 0                # the engine stole: allowed


def _case(name: str):
    def hole20(e, sr):
        return [Fault("zero_block", frame=int(5.2 * sr), ms=20)], 5.2, "PILOT_GAP", \
            "B1.four", None

    def click(e, sr):
        return [Fault("click", frame=int(3.33 * sr), amp=0.2)], 3.33, "CLICK", "B1.one", None

    def missing(e, sr):
        v = pick(e, sr, "host", 10.0, 11.0, restart=False)
        return [Fault("drop_hit", vid=v.vid)], v.f_on / sr, "MISSING", "B1.many", v.aid

    def late(e, sr):
        v = pick(e, sr, "host", 11.5, 12.5, restart=False)
        return [Fault("delay_hit", vid=v.vid, ms=80)], v.f_on / sr + 0.080, "LATE", \
            "B1.many", v.aid

    def cut(e, sr):
        v = pick(e, sr, "host", 4.5, 6.0, pad=9)
        return [Fault("cut_voice", vid=v.vid, at_ms=150)], v.f_on / sr + 0.150, "EARLY_CUT", \
            "B1.four", v.aid

    def pilot_gap(e, sr):
        return [Fault("pilot_gap", frame=int(7.7 * sr), ms=4)], 7.7, "PILOT_GAP", "B1.stream", \
            None

    def slip(e, sr):
        return [Fault("capture_drop", frame=int(2.5 * sr), samples=37)], 2.5, "CAPTURE_SLIP", \
            "B1.one", None

    def seq_miss(e, sr):
        v = pick(e, sr, "seq", 1.0, 3.0, restart=False)
        return [Fault("drop_hit", vid=v.vid)], v.f_on / sr, "SEQ_MISS", "B1.stream", None

    return {"hole20": hole20, "click": click, "missing": missing, "late": late, "cut": cut,
            "pilot_gap": pilot_gap, "slip": slip, "seq_miss": seq_miss}[name]


@pytest.mark.parametrize("name", ["hole20", "click", "missing", "late", "cut", "pilot_gap",
                                  "slip", "seq_miss"])
def test_each_fault_is_one_glitch_at_its_moment_in_its_step(tmp_path: Path, name: str) -> None:
    box: dict = {}

    def faults(exp: Expectation, sr: int) -> list[Fault]:
        fl, t, cls, step, aid = _case(name)(exp, sr)
        box.update(t=t, cls=cls, step=step, aid=aid)
        return fl

    info, ja = run(tmp_path, seed=1, faults=faults)
    gl = glitches(ja)
    assert len(gl) == 1, [(g["kind"], g["axes"]["wav_s"]) for g in gl]
    g = gl[0]
    assert box["cls"] in g["classes"], g["classes"]
    assert abs(g["axes"]["wav_s"] - box["t"]) * 1000.0 <= MS, (g["axes"]["wav_s"], box["t"])
    assert g["step"] == box["step"]
    if box["aid"] is not None:
        assert g["action"]["aid"] == box["aid"]
    # every clock is there
    for k in ("frame", "wav_s", "t_host", "t_wall", "wall", "dev_log_ms", "step"):
        assert k in g["axes"], k
    assert ja.status == "FAIL" and ja.exit_code == 1


def test_the_causes_are_told_apart(tmp_path: Path) -> None:
    def hole(exp, sr):
        return [Fault("zero_block", frame=int(5.2 * sr), ms=20)]

    _i, ja = run(tmp_path / "a", faults=hole)
    assert glitches(ja)[0]["cause"] == "render_overload"       # pads and stream together
    _i, jb = run(tmp_path / "b", faults=lambda e, sr: [Fault("pilot_gap", frame=int(7.7 * sr),
                                                             ms=4)])
    assert glitches(jb)[0]["cause"] == "usb_stream"            # the stream alone
    _i, jc = run(tmp_path / "c", faults=lambda e, sr: [Fault("capture_drop", frame=int(2.5 * sr),
                                                             samples=37)])
    g = glitches(jc)[0]
    slip = next(f for f in g["findings"] if f["cls"] == "CAPTURE_SLIP")
    assert g["cause"] == "usb_capture" and slip["measured"] == -37


def test_a_stream_fault_with_the_bridge_counting_it_is_the_stream_and_a_counter(
        tmp_path: Path) -> None:
    _i, ja = run(tmp_path, spk_under_at=7.6,
                 faults=lambda e, sr: [Fault("pilot_gap", frame=int(7.7 * sr), ms=4)])
    g = glitches(ja)[0]
    assert g["cause"] == "usb_stream" and g["confidence"] >= 0.8
    assert g["spk"]["delta"]["under"] == 3
    assert any(c["counter"] == "spk.under" for c in ja.doc["counters"])


def test_a_hung_block_fails_with_its_reason(tmp_path: Path) -> None:
    _i, ja = run(tmp_path, hung=True)
    assert ja.status == "FAIL"
    assert any(c["counter"] == "hung" and "no frames for 2 s" in c["note"]
               for c in ja.doc["counters"])


def test_a_host_xrun_is_the_host_not_the_board(tmp_path: Path) -> None:
    """A hole the host's own capture xrun explains is not charged to the board."""
    t_x = synth.T0 - 0.2 + 5.2            # the take's 5.2 s on the host clock

    _i, ja = run(tmp_path, faults=lambda e, sr: [Fault("zero_block", frame=int(5.2 * sr), ms=20)],
                 duplex_events=[{"kind": "xrun", "side": "in", "t": t_x}])
    assert glitches(ja) == []
    assert ja.doc["glitches"] and all(g["status"] != "glitch" for g in ja.doc["glitches"])


def test_the_report_files(tmp_path: Path) -> None:
    _i, ja = run(tmp_path, faults=lambda e, sr: [Fault("click", frame=int(3.33 * sr), amp=0.2)])
    rd = tmp_path / "run"
    doc = json.loads((rd / analysis.REPORT).read_text())
    assert doc["verdict"]["status"] == "FAIL" and doc["glitches"][0]["gid"] == "g0001"
    page = (rd / "timeline.html").read_text()
    assert "g0001" in page and "http://" not in page and "https://" not in page
    assert "| g0001 |" in (rd / "timeline.md").read_text()
    labels = (rd / "labels_B1.txt").read_text()
    assert "g0001 CLICK" in labels and "B1.one" in labels
    snip = doc["glitches"][0]["snippet"]
    assert Path(snip["cap"]).exists() and Path(snip["diff"]).exists()
    assert (rd / "expected_B1.wav").exists()


def test_the_analysis_is_repeatable(tmp_path: Path) -> None:
    _i, ja = run(tmp_path, faults=lambda e, sr: [Fault("click", frame=int(3.33 * sr), amp=0.2)])
    again = analysis.analyze_run(tmp_path / "run", snippets=0, allow_partial=True, write=False)
    assert [(g["kind"], g["frame"]) for g in again.doc["glitches"]] == \
        [(g["kind"], g["frame"]) for g in ja.doc["glitches"]]
