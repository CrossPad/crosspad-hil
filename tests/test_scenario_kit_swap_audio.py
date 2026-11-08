"""Scenario `kit_swap_audio`, the parts that do not need a sound card.

The recording half is covered by `test_analyze_tone_tracks.py`; what is left
here is everything the scenario itself decides: the swap schedule, the per-pad
kit-sequence rule (deferral is legal, going backwards is not), the `KIT_SWAP_LOG`
parsing, and the verdict — all pure functions, plus one `FakeSerial` device for
the CDC verb, in `test_scenario_kit_churn.py`'s style.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from crosspad_hil import verbs
from crosspad_hil.cdc import CdcLink
from crosspad_hil.errors import BAD_ARGS, HilError
from crosspad_hil.midi import sysex_kit_load
from crosspad_hil.parsers import parse_cdc_reply
from crosspad_hil.scenarios import get, kit_swap_audio
from crosspad_hil.scenarios.kit_swap_audio import (
    Params,
    build_swaps,
    candidate_freqs,
    freq_a,
    freq_b,
    kit_sequence_problems,
    stage_kits,
    swap_verdict,
)
from tests.fakes import FakeSerial

SWAP_ROW = (
    "KITSWAP: seq=3 kit=7 ok=1 req_ms=142300 parse_ms=61 done_ms=180 gated_blocks=2 "
    "gate_max_us=940 gate_total_us=1480 gate_count=2 deferred=0x0024 forced=0x0000 "
    "dropped_hits=0 pads=12,-1,-1,-1,-1,180,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1"
)


# --------------------------------------------------------------------------- registry


def test_registered_and_defaults() -> None:
    assert get("kit_swap_audio") is kit_swap_audio.SCENARIO
    assert kit_swap_audio.SCENARIO.name == "kit_swap_audio"
    p = Params()
    assert (p.kit_a, p.kit_b, p.swaps, p.hit_rate, p.dwell, p.lead, p.pads, p.velocity,
            p.gate_ms, p.preset, p.volume, p.dropout_db, p.max_gap_ms, p.max_late_ms,
            p.no_provision, p.keep_audio_mode, p.out) == (
        "HIL SINE A", "HIL SINE B", 3, 4.0, 6.0, 3.0, None, 100, 60, "headphone",
        None, 10.0, 5.0, 20.0, False, False, None)


@pytest.mark.parametrize("no_provision, want", [(False, True), (True, False)])
def test_provisioning_is_on_by_default_and_off_behind_the_flag(no_provision: bool,
                                                               want: bool) -> None:
    """A card without the fixtures is something the host can fix; `--no-provision`
    is for a bench where writing to the SD card is not wanted."""
    seen: dict[str, object] = {}

    def fake_ensure(ctx, link, names, builder, **kw):     # noqa: ANN001, ANN202
        seen.update(names=list(names), builder=builder, **kw)
        return link, []

    from crosspad_hil import fixtures
    from crosspad_hil.scenarios import _audio
    real, _audio.ensure_kits = _audio.ensure_kits, fake_ensure
    try:
        link, provisioned = kit_swap_audio.SCENARIO._provision(
            None, Params(no_provision=no_provision), "LINK", "CONSOLE")
    finally:
        _audio.ensure_kits = real
    assert (link, provisioned) == ("LINK", [])
    assert seen["names"] == ["HIL SINE A", "HIL SINE B"]
    assert seen["builder"] is fixtures.write_sine_kits
    assert seen["console"] == "CONSOLE"
    assert seen["provision"] is want
    assert "make_sine_kits.py" in str(seen["missing_hint_suffix"])


def test_the_two_fixture_ladders_are_interleaved_and_never_collide() -> None:
    """111 Hz apart per pad, and no harmonic of either ladder inside +-15 Hz of
    another tone — the assumption `analyze_tone_tracks` is measuring under."""
    tones = [f(p) for p in range(16) for f in (freq_a, freq_b)]
    for p in range(16):
        assert freq_b(p) - freq_a(p) == pytest.approx(111.0)
    for t in tones:
        for harmonic in (2 * t, 3 * t):
            for other in tones:
                if other != t:
                    assert abs(harmonic - other) > 15.0
    assert candidate_freqs([0, 5])[5] == [freq_a(5), freq_b(5)]


# --------------------------------------------------------------------------- schedule


def test_build_swaps_alternates_b_a_b_from_lead_every_dwell() -> None:
    swaps = build_swaps(3, lead=3.0, dwell=6.0)
    assert [s["t"] for s in swaps] == [3.0, 9.0, 15.0]
    assert [s["kit_index"] for s in swaps] == [1, 0, 1]
    assert [s["stage"] for s in swaps] == [1, 2, 3]
    # stage 0 is the kit the run started on, so the ladder is A B A B
    assert stage_kits(swaps) == [0, 1, 0, 1]


# --------------------------------------------------------------------------- sequence rule


def hit(t: float, pad: int, kit: int | None) -> dict:
    return {"t": t, "pad": pad, "kit": kit}


SWAPS = build_swaps(3, lead=3.0, dwell=6.0)      # B at 3 s, A at 9 s, B at 15 s


def test_a_clean_alternation_has_no_sequence_problems() -> None:
    hits = [hit(1.0, 0, 0), hit(5.0, 0, 1), hit(11.0, 0, 0), hit(17.0, 0, 1)]
    assert kit_sequence_problems(hits, SWAPS) == []


def test_a_hit_still_on_the_old_kit_after_the_swap_is_a_deferred_reload_not_a_fault() -> None:
    """The pad was sounding when the swap was asked for; the loader waited. That
    is the behaviour under test, so it must not be reported as a violation."""
    hits = [hit(1.0, 0, 0), hit(3.5, 0, 0), hit(5.0, 0, 1), hit(11.0, 0, 0)]
    assert kit_sequence_problems(hits, SWAPS) == []


def test_going_back_to_an_older_kit_is_a_violation() -> None:
    hits = [hit(1.0, 4, 0), hit(5.0, 4, 1), hit(7.0, 4, 0)]
    problems = kit_sequence_problems(hits, SWAPS)
    assert len(problems) == 1
    assert "pad 4" in problems[0] and "went back" in problems[0]


def test_a_kit_heard_before_it_was_ever_requested_is_a_violation() -> None:
    hits = [hit(1.0, 9, 1)]          # kit B at 1 s; the first B swap is at 3 s
    problems = kit_sequence_problems(hits, SWAPS)
    assert len(problems) == 1
    assert "before it was ever requested" in problems[0]


def test_missing_hits_are_skipped_by_the_sequence_rule() -> None:
    """A hit that made no sound is already a FAIL under the audible rule; giving
    it a stage too would report the same defect twice, in two vocabularies."""
    hits = [hit(1.0, 2, 0), hit(5.0, 2, None), hit(11.0, 2, 0)]
    assert kit_sequence_problems(hits, SWAPS) == []


def test_pads_are_judged_independently() -> None:
    hits = [hit(1.0, 0, 0), hit(5.0, 0, 1),        # pad 0 reloaded promptly
            hit(4.0, 1, 0), hit(8.0, 1, 0),        # pad 1 deferred all the way
            hit(5.0, 2, 1), hit(7.0, 2, 0)]        # pad 2 went back
    problems = kit_sequence_problems(hits, SWAPS)
    assert len(problems) == 1 and "pad 2" in problems[0]


# --------------------------------------------------------------------------- KITSWAP


def test_parse_one_kitswap_row() -> None:
    p = parse_cdc_reply(SWAP_ROW)
    assert p is not None and p["kind"] == "kitswap" and p["end"] is False
    assert (p["seq"], p["kit"], p["ok"]) == (3, 7, True)
    assert (p["req_ms"], p["parse_ms"], p["done_ms"]) == (142300, 61, 180)
    assert (p["gated_blocks"], p["gate_max_us"], p["gate_total_us"], p["gate_count"]) \
        == (2, 940, 1480, 2)
    # the masks arrive as hex and must be usable as ints: pads 2 and 5 deferred
    assert p["deferred"] == 0x24 and p["deferred_hex"] == "0x0024"
    assert p["forced"] == 0 and p["dropped_hits"] == 0
    assert [i for i in range(16) if p["deferred"] & (1 << i)] == [2, 5]
    assert len(p["pads"]) == 16 and p["pads"][0] == 12 and p["pads"][5] == 180
    assert p["pads"][1] == -1


def test_extra_engine_counters_pass_through_untouched() -> None:
    """The row is parsed as key=value, not against a fixed field list: the
    firmware already emits `underruns=`/`noblock=` beside the gate counters, and
    the next one it adds must not need a host release."""
    p = parse_cdc_reply(SWAP_ROW.replace("deferred=", "underruns=1 noblock=2 deferred="))
    assert p is not None
    assert (p["underruns"], p["noblock"]) == (1, 2)
    assert p["seq"] == 3 and p["deferred"] == 0x24 and len(p["pads"]) == 16


def test_parse_kitswap_end_line_and_junk() -> None:
    assert parse_cdc_reply("KITSWAP: end count=2") == {
        "kind": "kitswap", "end": True, "count": 2}
    assert parse_cdc_reply("KITSWAP: nonsense here") is None
    assert parse_cdc_reply("KITSWAP: seq=1 kit=2 deferred=0xZZ") is None


def make_link(lines: dict[str, list[str]]) -> tuple[CdcLink, FakeSerial]:
    fake = FakeSerial([(cmd, reply) for cmd, reply in lines.items()])
    link = CdcLink("/dev/ttyACM0", serial_factory=lambda path, **kw: fake)
    link.open()
    return link, fake


def test_kit_swap_log_reads_the_ring_oldest_first() -> None:
    rows = [SWAP_ROW, SWAP_ROW.replace("seq=3", "seq=4").replace("kit=7", "kit=2")]
    link, _fake = make_link({"KIT_SWAP_LOG": [*rows, "KITSWAP: end count=2"]})
    try:
        out = verbs.kit_swap_log(link)
    finally:
        link.close()
    assert [r["seq"] for r in out] == [3, 4]
    assert [r["kit"] for r in out] == [7, 2]
    assert "kind" not in out[0] and "end" not in out[0]


def test_kit_swap_log_on_firmware_that_does_not_know_the_verb_is_empty_plus_a_log_line() -> None:
    """The scenario treats the log as evidence, not as a gate: an unknown verb
    (the device echoes it back) must not cost the run its verdict."""
    link, _fake = make_link({})
    said: list[str] = []
    try:
        out = verbs.kit_swap_log(link, log=said.append)
    finally:
        link.close()
    assert out == []
    assert said and "KIT_SWAP_LOG" in said[0]


def test_kit_swap_log_says_so_when_the_trailer_and_the_rows_disagree() -> None:
    link, _fake = make_link({"KIT_SWAP_LOG": [SWAP_ROW, "KITSWAP: end count=3"]})
    said: list[str] = []
    try:
        out = verbs.kit_swap_log(link, log=said.append)
    finally:
        link.close()
    assert len(out) == 1
    assert any("trailer says 3" in s for s in said)


def test_sysex_kit_load_frame() -> None:
    assert sysex_kit_load(7) == bytes([0xF0, 0x7D, 0x1D, 0x0B, 0x07, 0xF7])
    with pytest.raises(HilError) as ei:
        sysex_kit_load(200)          # a kit id has to fit in a SysEx data byte
    assert ei.value.code == BAD_ARGS


# --------------------------------------------------------------------------- verdict


def base_data(**over) -> dict:
    data = {
        "failures": [], "fatal": [], "reboots": 0, "last_kit_id": 5,
        "kit_status": {"current": 5, "loading": False, "pending": -1, "name": "HIL SINE B"},
        "scored": [{"pad": 0, "t": 1.0, "kit": 0, "missing": False, "switch_frames": 0,
                    "switch_run_frames": 0, "late_start_ms": 0.0,
                    "max_gap_ms": 0.0, "dropout_frames": 0, "dropout_ms": 0.0}],
        "sequence_problems": [],
    }
    data.update(over)
    return data


def test_verdict_passes_a_clean_run() -> None:
    assert swap_verdict(Params(), base_data()) == []


def test_verdict_fails_on_a_crash() -> None:
    problems = swap_verdict(Params(), base_data(fatal=["Guru Meditation Error"], reboots=1))
    assert problems and "crash" in problems[0]


def test_verdict_fails_when_the_device_did_not_settle_on_the_last_kit() -> None:
    status = {"current": 4, "loading": False, "pending": -1, "name": "HIL SINE A"}
    problems = swap_verdict(Params(), base_data(kit_status=status))
    assert any("not the last one requested (5)" in s for s in problems)
    problems = swap_verdict(Params(), base_data(kit_status=None))
    assert any("KIT_STATUS unavailable" in s for s in problems)
    still = {"current": 5, "loading": True, "pending": -1, "name": "HIL SINE B"}
    assert swap_verdict(Params(), base_data(kit_status=still))


def test_verdict_fails_on_a_missing_hit_a_switch_and_a_hole() -> None:
    scored = [
        {"pad": 1, "t": 2.0, "kit": None, "missing": True, "switch_frames": 0,
         "switch_run_frames": 0, "max_gap_ms": 0.0},
        {"pad": 5, "t": 4.0, "kit": 0, "missing": False, "switch_frames": 7,
         "switch_run_frames": 7, "max_gap_ms": 0.0},
        {"pad": 7, "t": 6.0, "kit": 1, "missing": False, "switch_frames": 0,
         "switch_run_frames": 0, "max_gap_ms": 38.0},
    ]
    problems = swap_verdict(Params(), base_data(scored=scored))
    assert any("made no sound" in s for s in problems)
    assert any("changed kit mid-tone" in s and "pad 5" in s for s in problems)
    assert any("holes up to 38 ms" in s for s in problems)


def test_isolated_flipped_frames_alone_do_not_fail_the_run() -> None:
    """The raw `switch_frames` counts harmonics and noise; only a run long enough
    to be a reload (`switch_run_frames`) is a verdict."""
    scored = [{"pad": 5, "t": 4.0, "kit": 0, "missing": False, "switch_frames": 4,
               "switch_run_frames": 0, "max_gap_ms": 0.0}]
    assert swap_verdict(Params(), base_data(scored=scored)) == []


def test_a_late_start_over_the_limit_fails_and_under_it_does_not() -> None:
    """A voice that waited for its first refill started late; it has no hole in
    it, so it is gated on its own number, not folded into the gap rule."""
    scored = [{"pad": 2, "t": 1.0, "kit": 0, "missing": False, "switch_frames": 0,
               "switch_run_frames": 0, "max_gap_ms": 0.0, "late_start_ms": 140.0},
              {"pad": 3, "t": 5.0, "kit": 0, "missing": False, "switch_frames": 0,
               "switch_run_frames": 0, "max_gap_ms": 0.0, "late_start_ms": 20.0}]
    problems = swap_verdict(Params(), base_data(scored=scored))
    assert problems == ["1 hit(s) started late (up to 140 ms)"]
    # under the limit it is a WARN, i.e. not a problem at all here
    assert swap_verdict(Params(max_late_ms=200.0), base_data(scored=scored)) == []


def test_a_gap_at_or_under_the_limit_is_not_a_failure() -> None:
    scored = [{"pad": 3, "t": 1.0, "kit": 0, "missing": False, "switch_frames": 0,
               "switch_run_frames": 0, "max_gap_ms": 5.0}]
    assert swap_verdict(Params(), base_data(scored=scored)) == []
    assert swap_verdict(Params(max_gap_ms=4.0), base_data(scored=scored))


def test_verdict_carries_the_sequence_problems_through() -> None:
    problems = swap_verdict(Params(), base_data(sequence_problems=["pad 2 went back"]))
    assert problems == ["pad 2 went back"]


# --------------------------------------------------------------------------- arg guards


def test_a_dwell_that_does_not_outlast_the_tone_is_refused(tmp_path: Path) -> None:
    """A swap fired inside a tone that is still being scored cannot be told from
    a pad reloaded mid-sample, which is the one defect this scenario exists for."""
    import threading

    from crosspad_hil.errors import BAD_ARGS, HilError
    from crosspad_hil.scenarios import base
    from tests.test_scenario_kit_churn import make_device

    ctx = base.Context(device=make_device(), workdir=tmp_path,
                       cancelled=threading.Event(), log=lambda s: None)
    with pytest.raises(HilError) as ei:
        kit_swap_audio.SCENARIO.run(ctx, Params(dwell=1.5), base.Progress())
    assert ei.value.code == BAD_ARGS
    with pytest.raises(HilError) as ei:
        kit_swap_audio.SCENARIO.run(ctx, Params(pads=[0, 99]), base.Progress())
    assert ei.value.code == BAD_ARGS
    with pytest.raises(HilError) as ei:
        kit_swap_audio.SCENARIO.run(ctx, Params(swaps=0), base.Progress())
    assert ei.value.code == BAD_ARGS
