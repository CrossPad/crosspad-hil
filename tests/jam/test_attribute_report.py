"""Causes from evidence, and the verdict's three answers."""
from __future__ import annotations

from crosspad_hil.jam import report as R
from crosspad_hil.jam.attribute import GlitchRecord, classify_cause, group
from crosspad_hil.jam.detect import Finding


def rec(classes: dict[str, int], **kw) -> GlitchRecord:
    r = GlitchRecord(gid="g1", block="B1", kind=next(iter(classes)), classes=classes,
                     frame=0.0, length_ms=1.0)
    r.model_state = {"flags": kw.pop("flags", []), "sampler": kw.pop("sampler", 2)}
    r.spk = kw.pop("spk", {})
    r.host = {"xruns": kw.pop("xruns", [])}
    r.engine = kw.pop("engine", [])
    r.env = kw.pop("env", {})
    return r


def cause(classes: dict[str, int], **kw) -> str:
    return classify_cause(rec(classes, **kw), [], kw.pop("counters", {}))[0]


def test_the_cause_matrix() -> None:
    assert cause({"PILOT_GAP": 1}, spk={"delta": {"under": 2}}) == "usb_stream"
    assert cause({"PILOT_GAP": 1}, xruns=[{"side": "out"}]) == "host_playback"
    assert cause({"CAPTURE_SLIP": 1}) == "usb_capture"
    assert cause({"PILOT_GAP": 1, "HOLE": 3}) == "render_overload"
    assert cause({"PILOT_GAP": 1, "HOLE": 3}, spk={"delta": {"cap_drop": 4}}) == "usb_capture"
    assert cause({"HOLE": 1}, engine=["engine underrun_at=5ms"]) == "engine_underrun"
    assert cause({"WRONG_KIT": 1}, flags=["swap"]) == "swap"
    assert cause({"EARLY_CUT": 1}, sampler=8) == "polyphony"
    assert cause({"MISSING": 1}, counters={"inj_drop": 3}) in ("inject_queue",)
    assert cause({"CLICK": 1}) in ("unknown", "log_busywait")


def test_findings_close_together_are_one_glitch() -> None:
    fs = [Finding(cls="PILOT_GAP", frame=1000.0, length=800.0),
          Finding(cls="HOLE", frame=1100.0), Finding(cls="CLICK", frame=2300.0),
          Finding(cls="CLICK", frame=90_000.0)]
    groups = group(fs, 44_100)
    assert [len(g) for g in groups] == [3, 1]


def test_the_verdict_has_three_answers() -> None:
    g = {"gid": "g1", "block": "B1", "status": "glitch", "kind": "HOLE", "classes": {"HOLE": 1},
         "cause": "swap", "confidence": 0.6, "axes": {"wav_s": 1.0, "wall": "12:00:00.000"},
         "step": "B1.four"}
    assert R.verdict([g], [], [], [], False, [])[:2] == ("FAIL", 1)
    assert R.verdict([], [{"block": "B1", "note": "engine underrun +1"}], [], [], False, [])[0] \
        == "FAIL"
    assert R.verdict([], [], ["B2: host lost 3 s"], [], False, [])[:2] == ("not measured", 2)
    assert R.verdict([], [], [], ["bounce_sysex"], False, [])[1] == 2
    assert R.verdict([], [], [], ["bounce_sysex"], True, [])[:2] == ("ZERO-GLITCH PASS", 0)
    status, code, lines = R.verdict([], [], [], [], False, ["desk: ERR"])
    assert (status, code) == ("ZERO-GLITCH PASS", 0) and lines == ["LEFT CHANGED: desk: ERR"]
