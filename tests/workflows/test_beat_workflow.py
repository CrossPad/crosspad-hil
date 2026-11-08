"""The beat workflow file, its fixture, and the bounce analysis on a synthetic bounce."""

from __future__ import annotations

from pathlib import Path

import pytest

from crosspad_hil import workflows
from crosspad_hil.workflows import beat_fixture as bf

np = pytest.importorskip("numpy")
from crosspad_hil.workflows import beat_audio as ba  # noqa: E402


def test_depths_nest():
    wf = workflows.load("beat")
    q, m, f = (set(s.id for s in wf.for_depth(d)) for d in workflows.DEPTHS)
    assert q < m < f
    assert {"preflight", "slice", "record", "bounce", "settings", "cleanup"} <= q
    assert wf.for_depth("quick")[-1].op == "cleanup"


def test_expected_scene():
    wf = workflows.load("beat")
    bars, hits = workflows.expected_scene(wf, "quick")
    assert bars == 2 and len(hits) == 12
    bars, hits = workflows.expected_scene(wf, "medium")
    assert bars == 4 and len(hits) == 36
    bars, hits = workflows.expected_scene(wf, "full")
    assert bars == 4 and all(h.pad != wf.raw["erase_pad"] for h in hits)


def test_fixture_is_deterministic(tmp_path: Path):
    a = bf.build(tmp_path / "a.wav", 100)
    bf.build(tmp_path / "b.wav", 100)
    assert (tmp_path / "a.wav").read_bytes() == (tmp_path / "b.wav").read_bytes()
    assert a.frames == 105840 and a.onsets_frames[1] == 13230


def _bounce(tmp_path: Path, hits, step_s, semis=None, rev=(), lag_s=0.012, sr=44100):
    loop = bf.Loop(bpm=100, hits=bf.DEFAULT_HITS, sample_rate=sr)
    x = np.array(bf.render(loop))
    step = loop.onsets_frames[1]
    slices = [x[i * step : (i + 1) * step] for i in range(8)]
    semis = semis or {}
    total = int((max(h.step for h in hits) + 16) * step_s * sr)
    out = np.zeros(total)
    starts = sorted((int(h.step * step_s * sr + lag_s * sr), h) for h in hits)
    for k, (s, h) in enumerate(starts):
        sl = slices[h.pad]
        if h.pad in rev:
            sl = sl[::-1]
        r = 2.0 ** (semis.get(h.pad, 0) / 12.0)
        if r != 1.0:
            idx = np.arange(0, len(sl) - 1, r)
            sl = np.interp(idx, np.arange(len(sl)), sl)
        end = starts[k + 1][0] if k + 1 < len(starts) else total
        seg = sl[: max(0, end - s)] * (h.vel / 127.0)
        out[s : s + len(seg)] += seg
    p = tmp_path / "bounce.wav"
    bf.write_wav(p, out.tolist(), sr)
    return p


@pytest.mark.parametrize("depth", ["quick", "medium", "full"])
def test_analysis_passes_a_right_bounce(tmp_path: Path, depth: str):
    wf = workflows.load("beat")
    bars, hits = workflows.expected_scene(wf, depth)
    edits = wf.edits(depth)
    semis = {e["pad"]: e["value"] for e in edits if e["control"] == "PITCH"}
    rev = [e["pad"] for e in edits if e["control"] == "REV"]
    p = _bounce(tmp_path, hits, wf.step_s, semis, rev)
    exp = ba.expected_hits(
        hits, wf.step_s, [h["freq"] for h in wf.raw["fixture"]["hits"]], semis, rev
    )
    res = ba.analyze_bounce(p, exp, want_s=bars * wf.bar_s - 0.3)
    assert res["failures"] == [], res["failures"]
    assert abs(res["offset_ms"] - 12.0) < 1.5


def test_analysis_catches_a_wrong_pad_and_a_missing_hit(tmp_path: Path):
    wf = workflows.load("beat")
    bars, hits = workflows.expected_scene(wf, "quick")
    played = [workflows.Hit(h.step, 7 if i == 3 else h.pad, h.vel) for i, h in enumerate(hits)][:-1]
    p = _bounce(tmp_path, played, wf.step_s)
    exp = ba.expected_hits(hits, wf.step_s, [h["freq"] for h in wf.raw["fixture"]["hits"]])
    res = ba.analyze_bounce(p, exp, want_s=bars * wf.bar_s - 0.3)
    text = " ".join(res["failures"])
    assert "Hz, pad" in text and "no hit at" in text
