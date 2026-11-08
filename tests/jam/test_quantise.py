"""quantise() is PatternSequencer::quantise, integer for integer."""
from __future__ import annotations

import random

from crosspad_hil.jam.model import downbeat_grace, grid_per_beat, quantise, samples_per_beat


def ref(pos: int, spb: int, per: int, loop: int) -> int:
    """The C++ line by line (uint64 arithmetic)."""
    if not per or spb < 2:
        return pos
    k = (pos * per * 2 + spb) // (spb * 2)
    snapped = (k * spb) // per
    return 0 if (loop and snapped >= loop) else snapped


def test_vectors() -> None:
    spb = samples_per_beat(120, 44_100)
    assert spb == 22_050
    loop = 8 * spb
    per16 = grid_per_beat(2)
    assert quantise(0, spb, per16, loop) == 0
    assert quantise(2_700, spb, per16, loop) == 0                  # under half a 16th
    assert quantise(2_800, spb, per16, loop) == 5_512              # 22050/4 rounded down
    assert quantise(loop - 100, spb, per16, loop) == 0             # rounds to the loop end = 0
    per8t = grid_per_beat(4)
    assert per8t == 3
    assert quantise(7_350, spb, per8t, loop) == 7_350              # k*spb/per, not k steps
    assert quantise(14_700, spb, per8t, loop) == 14_700
    assert quantise(1234, spb, grid_per_beat(127), loop) == 1234   # off
    assert downbeat_grace(spb, per16) == 2_756
    assert downbeat_grace(spb, 0) == spb // 8


def test_matches_the_reference_everywhere() -> None:
    rng = random.Random(7)
    for _ in range(5000):
        bpm = rng.choice([90, 100, 120, 174])
        spb = samples_per_beat(bpm, rng.choice([44_100, 48_000]))
        per = grid_per_beat(rng.choice([0, 1, 2, 3, 4, 5, 127]))
        loop = rng.choice([1, 2, 4, 8]) * 4 * spb
        pos = rng.randrange(loop)
        assert quantise(pos, spb, per, loop) == ref(pos, spb, per, loop)
