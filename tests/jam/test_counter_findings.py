from __future__ import annotations

from crosspad_hil.jam.analysis import counter_findings


def _blk(before: dict, after: dict) -> dict:
    return {"readback_before": {"counters": before}, "readback_after": {"counters": after}}


def test_native_engine_counters_that_moved_are_findings() -> None:
    blk = _blk({"native": {"underrun": 1, "norings": 0, "drop": 0, "notes": 10}},
               {"native": {"underrun": 3, "norings": 1, "drop": 0, "notes": 90}})
    found = {f["counter"]: f["delta"] for f in counter_findings("A", blk, None, None)}
    assert found == {"native.underrun": 2, "native.norings": 1}


def test_a_native_only_image_with_no_pool_reports_nothing_from_it() -> None:
    # SMPL_POOL answers ERR only_native there: no "pool" counters on either side.
    blk = _blk({"native": {"underrun": 0}}, {"native": {"underrun": 0}})
    assert counter_findings("A", blk, None, None) == []
