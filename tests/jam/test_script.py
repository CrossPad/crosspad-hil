"""The set itself: deterministic, scaled, every feature with a step that proves it."""
from __future__ import annotations

from crosspad_hil.jam import script as J


def test_the_same_seed_is_the_same_set() -> None:
    a, b = J.build_script(7), J.build_script(7)
    assert a.to_dict() == b.to_dict()
    assert J.build_script(8).to_dict() != a.to_dict()


def test_the_set_lasts_what_was_asked() -> None:
    s = J.build_script(1, minutes=12.5)
    assert abs(s.uac_seconds / 60.0 - 12.5) / 12.5 < 0.05
    short = J.build_script(1, minutes=0.5)
    assert abs(short.uac_seconds / 60.0 - 0.5) / 0.5 < 0.05
    soak = J.build_script(1, minutes=12.5, soak_min=10)
    assert soak.block("SOAK").duration == 600.0 and "SOAK" in soak.order


def test_at_most_eight_swaps_a_block() -> None:
    """KIT_SWAP_LOG keeps the last 8: more would leave swaps without their row."""
    s = J.build_script(1, soak_min=10)
    for b in s.blocks:
        assert sum(1 for st in b.steps if st.action == "swap") <= 8, b.name


def test_every_feature_has_a_step_with_expect_and_verify() -> None:
    s = J.build_script(1)
    cov = J.coverage(s)
    missing = [f for f, r in cov.items() if not r["steps"] and not r["skipped"]]
    assert missing == []
    assert cov["soak"]["skipped"] == ["SOAK"]          # off by default, and it says so
    for b in s.blocks:
        for st in b.steps:
            assert st.expect and st.verify, st.id


def test_a_missing_capability_or_section_is_a_skip_with_a_reason() -> None:
    s = J.build_script(1, caps={"sysex_bounce": False, "companion": True},
                       sections=["build", "stream", "bounce"])
    skipped = {d["step"]: d["why"] for d in s.skipped}
    assert skipped["B1.bounce1"] == "the board has no sysex_bounce"
    assert "section 'clock' not selected" == skipped["B2.clock"]
    table = J.script_table(s)
    assert "SKIP: the board has no sysex_bounce" in table and "[W0] CDC" in table


def test_hits_are_inside_their_step() -> None:
    s = J.build_script(3, minutes=2.0)
    for b in s.blocks:
        for st in b.steps:
            for h in st.hits:
                assert 0.0 <= h.t <= st.dur + 1e-9, (st.id, h.t, st.dur)
