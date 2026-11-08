"""The phone's stream: rendered with random access, read back to the sample."""
from __future__ import annotations

import numpy as np

from crosspad_hil.jam.pilot import PilotSource, pilot_track, residual_gaps, timecode_delay
from crosspad_hil.jam.spec import PilotSpec

SR = 44_100


def take(seconds: float, delay: int = 1234, noise: float = 1e-5) -> np.ndarray:
    src = PilotSource(PilotSpec(), SR, 1)
    host = src.samples(0, int(seconds * SR))
    x = np.concatenate([np.zeros(delay), host])[: len(host)]
    x = x + 0.07 * np.sin(2 * np.pi * 3110.0 * np.arange(len(x)) / SR)     # a pad
    return x + np.random.default_rng(1).normal(0.0, noise, len(x))


def test_slices_render_exactly_what_one_long_render_does() -> None:
    src = PilotSource(PilotSpec(), SR, 2)
    whole = src.render(0, 30_000)
    parts = src.render(0, 7_000) + src.render(7_000, 13_000) + src.render(20_000, 10_000)
    assert parts == whole


def test_the_timecode_counts_lost_and_repeated_samples_exactly() -> None:
    x = take(8.0)
    s = int(3.0 * SR)
    x = np.concatenate([x[:s], x[s + 11:]])                         # 11 lost
    r = int(5.5 * SR)
    x = np.concatenate([x[:r], x[r - 5:r], x[r:]])                  # 5 repeated
    tr = timecode_delay(x, SR, PilotSpec())
    steps = [(st["samples"], round(st["frame"] / SR, 2)) for st in tr.steps]
    assert [k for k, _t in steps] == [-11, 5]
    assert abs(steps[0][1] - 3.0) <= 0.01 and abs(steps[1][1] - 5.5) <= 0.01


def test_short_holes_in_the_stream_are_found_to_the_sample() -> None:
    x = take(6.0)
    holes = [(int(2.0 * SR), 22), (int(4.0 * SR), 64)]             # 0.5 ms and 1.45 ms
    for a, n in holes:
        x[a:a + n] = 0.0
    tr = timecode_delay(x, SR, PilotSpec())
    got = residual_gaps(x, SR, PilotSpec(), tr)
    assert got["exact"] is True
    gaps = [g for g in got["gaps"] if g["kind"] == "gap" and g["frame"] > SR // 2]
    assert len(gaps) == 2
    for g, (a, n) in zip(gaps, holes, strict=True):
        assert abs(g["frame"] - a) <= 0.0005 * SR
        assert abs(g["length"] - n) <= 0.0005 * SR


def test_a_clean_minute_raises_nothing() -> None:
    x = take(60.0, noise=10 ** (-90 / 20))
    tr = timecode_delay(x, SR, PilotSpec())
    assert tr.steps == []
    got = residual_gaps(x, SR, PilotSpec(), tr)
    assert [g for g in got["gaps"] if g["frame"] > SR // 2] == []
    pt = pilot_track(x[SR:], SR, PilotSpec())
    assert pt.present and [e for e in pt.events if e.kind == "gap"] == []
