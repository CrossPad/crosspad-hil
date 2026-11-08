from crosspad_hil.scenarios import beat_session, beat_to_song, get
from crosspad_hil.scenarios.beat_session import _BeatRun


def row(t, pos, notes, song=0.0, play=1):
    return {"t": t, "pos": pos, "notes": notes, "song": song, "smp": 0.0, "play": play}


def test_both_scenarios_are_registered():
    assert get("beat_session") is beat_session.SCENARIO
    assert get("beat_to_song") is beat_to_song.SCENARIO


def test_a_pass_is_whole_from_its_first_bar_to_its_last_and_counts_the_voices_started():
    # A four-bar song looked at about once a second: two passes, the song going round.
    rows = [row(0, 0.5, 10), row(1, 1.5, 12), row(2, 2.5, 14), row(3, 3.6, 16),
            row(4, 0.4, 16), row(5, 1.6, 16), row(6, 2.4, 16), row(7, 3.5, 16)]
    passes = _BeatRun.passes(rows, 0, 4)
    assert [p["notes"] for p in passes] == [6, 0]   # the second pass started no voice: a gap


def test_a_pass_seen_from_its_middle_is_not_counted():
    rows = [row(0, 1.5, 10), row(1, 2.5, 12), row(2, 3.6, 14), row(3, 0.2, 14)]
    assert _BeatRun.passes(rows, 0, 4) == []
