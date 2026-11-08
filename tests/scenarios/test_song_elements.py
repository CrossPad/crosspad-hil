"""Scenario `song_elements` without a board: the host-side readers it judges the card with."""
from __future__ import annotations

from crosspad_hil.scenarios import get, song_elements
from crosspad_hil.scenarios.song_elements import (
    element_blocks,
    is_new_song,
    new_names,
    says_busy,
    seq_shape,
)

SONG = {"tracks": [
    {"clips": [{"file": "rec_001.wav", "bar": 0, "bars": 4}]},
    {"kind": 1, "clips": [{"bar": 0, "bars": 2, "from": 0, "notes": {"bars": 2, "ev": ""}}]},
    {"clips": [{"file": "BEA5.wav", "bar": 1, "bars": 2, "element": ".elements/E1"}]},
    {"clips": []},
]}


def test_the_scenario_is_registered():
    assert get("song_elements") is song_elements.SCENARIO


def test_element_blocks_are_the_clips_that_name_an_element():
    assert element_blocks(SONG) == [SONG["tracks"][2]["clips"][0]]
    assert element_blocks({"tracks": [{"clips": [{"file": "a.wav"}]}]}) == []
    assert element_blocks({}) == []


def test_new_names_are_what_a_run_added_sorted():
    assert new_names(["a", "b"], ["c", "b", "a", "d"]) == ["c", "d"]
    assert new_names(["a"], ["a"]) == []
    assert new_names([], []) == []


def test_two_takes_differ_in_their_seq_shape():
    first = [{"bar": 0, "bars": 2, "from": 0, "notes": 4, "own": 1}]
    second = [{"bar": 0, "bars": 1, "from": 0, "notes": 2, "own": 1},
              {"bar": 1, "bars": 1, "from": 2, "notes": 4, "own": 1}]
    assert seq_shape(first) == [(0, 2, 0, 4)]
    assert seq_shape(first) != seq_shape(second)


def test_the_progress_box_is_told_by_its_stage_words():
    assert says_busy(["Song 13 12 bars", "BAKING ELEMENT 45%"])
    assert says_busy(["KEEPING ELEMENT 0%"])
    assert not says_busy(["Song 13 12 bars", "ELEMENT 1", "EDIT"])


def test_a_timed_out_new_never_marks_the_users_song_for_deletion():
    before = ["BEAT LOFI 80", "hil_live_set"]
    assert not is_new_song("hil_live_set", before)      # NEW did not open anything
    assert not is_new_song("", before)
    assert is_new_song("Song 13", before)
