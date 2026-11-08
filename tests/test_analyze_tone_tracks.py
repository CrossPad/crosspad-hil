"""`analyze_tone_tracks` against synthesized recordings — no board, no sound card.

Each signal here is built so exactly one of the four answers the kernel gives is
forced: a clean run has to score no dropouts and the right kit per hit, a tone
with a hole punched in it has to report that hole and *not* a kit switch, a tone
that changes frequency mid-window has to report a switch and not merely a hole,
and a hit nobody played has to come back missing rather than as a quiet pass.
Those are the four verdict inputs `kit_swap_audio` has, so this is the whole
contract.

The frequency ladders are the fixture's (`tools/make_sine_kits.py`), duplicated
here for the same reason the scenario duplicates them.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from crosspad_hil.analyze import analyze_tone_tracks
from crosspad_hil.errors import BAD_ARGS, HilError
from tests.test_analyze import write_wav

SR = 44100
TONE_S = 2.0
PEAK = 0.12
ATTACK_S = 0.05        # analyze_tone_tracks' default window trim, needed to place a hole


def freq_a(pad: int) -> float:
    return 300.0 + 222.0 * pad


def freq_b(pad: int) -> float:
    return 411.0 + 222.0 * pad


def freqs(pads=range(16)) -> dict[int, list[float]]:
    return {p: [freq_a(p), freq_b(p)] for p in pads}


def tone(f: float, seconds: float, amp: float = PEAK) -> np.ndarray:
    t = np.arange(int(SR * seconds)) / SR
    return amp * np.sin(2 * np.pi * f * t)


def render(hits, total_s: float = 20.0) -> tuple[np.ndarray, list[dict]]:
    """`hits`: (t, pad, kit_index or None). Returns the mix and the hit list."""
    x = np.zeros(int(SR * total_s))
    out = []
    for t, pad, kit in hits:
        if kit is not None:
            s = tone((freq_a, freq_b)[kit](pad), TONE_S)
            n = int(t * SR)
            s = s[: len(x) - n]          # a tone may run past the end of the recording
            x[n:n + len(s)] += s
        out.append({"t": t, "pad": pad})
    return x, out


# --------------------------------------------------------------------------- clean


def test_clean_run_scores_every_hit_on_the_right_kit() -> None:
    """Sixteen pads round-robin at 4 hits/s: windows of one pad never overlap
    (a pad is revisited every 4 s against a 2 s tone) and neighbouring pads'
    tones sit in bands this measurement does not look at."""
    plan = [(1.0 + i * 0.25, i % 16, 0 if i < 16 else 1) for i in range(32)]
    x, hits = render(plan, total_s=14.0)
    r = analyze_tone_tracks(x, hits, rate=SR, tone_s=TONE_S, freqs=freqs())
    assert r["totals"] == {
        "hits": 32, "missing": 0, "audible": 32, "dropout_frames": 0, "dropout_ms": 0.0,
        "max_gap_ms": 0.0, "switch_frames": 0, "switch_run_frames": 0,
        "late_starts": 0, "late_start_max_ms": 0.0,
        "hits_with_dropouts": 0, "hits_with_switches": 0,
    }
    assert [h["kit"] for h in r["hits"]] == [0] * 16 + [1] * 16
    for h, (_t, pad, kit) in zip(r["hits"], plan, strict=True):
        assert h["kit_hz"] == pytest.approx((freq_a, freq_b)[kit](pad), abs=0.01)
        assert h["missing"] is False


def test_the_kit_is_read_per_pad_not_globally() -> None:
    """Half the pads deferred, half reloaded: the kernel must answer per hit."""
    plan = [(1.0 + i * 0.25, i, 0 if i % 2 else 1) for i in range(16)]
    x, hits = render(plan, total_s=10.0)
    r = analyze_tone_tracks(x, hits, rate=SR, tone_s=TONE_S, freqs=freqs())
    assert [h["kit"] for h in r["hits"]] == [0 if i % 2 else 1 for i in range(16)]
    assert r["totals"]["switch_frames"] == 0


# --------------------------------------------------------------------------- holes


def test_a_30_ms_hole_is_reported_as_a_30_ms_gap_not_a_switch() -> None:
    x, hits = render([(1.0, 5, 0)], total_s=6.0)
    # Punch the hole on the scoring grid: the window starts at t + attack_s, and
    # 0.5 s is a whole number of 10 ms frames from there.
    start = int((1.0 + ATTACK_S + 0.5) * SR)
    x[start:start + int(0.030 * SR)] = 0.0
    r = analyze_tone_tracks(x, hits, rate=SR, tone_s=TONE_S, freqs=freqs(), frame_ms=10.0)
    h = r["hits"][0]
    assert h["kit"] == 0 and h["missing"] is False
    assert h["dropout_frames"] == 3
    assert h["dropout_ms"] == pytest.approx(30.0)
    assert h["max_gap_ms"] == pytest.approx(30.0)
    # In a hole both candidates collapse; which of them wins the frame is noise,
    # and calling that a kit switch would turn every dropout into a FAIL.
    assert h["switch_frames"] == 0
    assert r["totals"]["hits_with_dropouts"] == 1


def test_two_separate_holes_add_up_but_max_gap_is_the_longest_one() -> None:
    x, hits = render([(1.0, 2, 1)], total_s=6.0)
    base = int((1.0 + ATTACK_S) * SR)
    x[base + int(0.30 * SR): base + int(0.30 * SR) + int(0.020 * SR)] = 0.0
    x[base + int(0.90 * SR): base + int(0.90 * SR) + int(0.050 * SR)] = 0.0
    r = analyze_tone_tracks(x, hits, rate=SR, tone_s=TONE_S, freqs=freqs(), frame_ms=10.0)
    h = r["hits"][0]
    assert h["kit"] == 1
    assert h["dropout_ms"] == pytest.approx(70.0)
    assert h["max_gap_ms"] == pytest.approx(50.0)


def test_a_dropout_shallower_than_the_threshold_is_not_a_dropout() -> None:
    x, hits = render([(1.0, 7, 0)], total_s=6.0)
    base = int((1.0 + ATTACK_S) * SR)
    x[base + int(0.50 * SR): base + int(0.55 * SR)] *= 0.5        # -6 dB, under the 10 dB gate
    r = analyze_tone_tracks(x, hits, rate=SR, tone_s=TONE_S, freqs=freqs(), frame_ms=10.0)
    assert r["hits"][0]["dropout_frames"] == 0
    assert r["totals"]["dropout_ms"] == 0.0


# --------------------------------------------------------------------------- switches


def test_a_tone_that_changes_frequency_mid_window_reports_switch_frames() -> None:
    """A pad reloaded while it was still sounding: the sample it was playing is
    replaced by the other kit's, at full level, halfway through."""
    x = np.zeros(int(SR * 6.0))
    n = int(1.0 * SR)
    s = tone(freq_a(9), TONE_S)
    s[int(1.0 * SR):] = tone(freq_b(9), 1.0)
    x[n:n + len(s)] += s
    r = analyze_tone_tracks(x, [{"t": 1.0, "pad": 9}], rate=SR, tone_s=TONE_S, freqs=freqs())
    h = r["hits"][0]
    assert h["switch_frames"] > 0
    # the run rule must not swallow a real reload: half the window is 47 frames
    assert h["switch_run_frames"] == h["switch_frames"]
    assert h["switch_max_run_ms"] > 3 * 20.0
    assert r["totals"]["switch_run_frames"] == h["switch_run_frames"]
    assert r["totals"]["hits_with_switches"] == 1
    # The half that dominates decides the reported kit; either half is a FAIL,
    # so the assertion is on switch_frames, not on which kit won.
    assert h["kit"] in (0, 1)
    assert h["dominant_counts"][0] > 0 and h["dominant_counts"][1] > 0


def test_an_isolated_flipped_frame_is_not_a_switch() -> None:
    """On clean captures a single frame flips to the other candidate — a
    harmonic, or noise landing in the band for one 20 ms window. A pad cannot be
    reloaded for 20 ms and back again, so the raw count is reported and the run
    rule is what a verdict reads."""
    x = np.zeros(int(SR * 6.0))
    n = int(1.0 * SR)
    s = tone(freq_a(9), TONE_S)
    # one 20 ms burst of the other kit's tone, on top, loud enough to dominate
    base = int((1.0 + ATTACK_S + 0.60) * SR) - n
    s[base:base + int(0.020 * SR)] += tone(freq_b(9), 0.020, amp=PEAK * 3)
    x[n:n + len(s)] += s
    r = analyze_tone_tracks(x, [{"t": 1.0, "pad": 9}], rate=SR, tone_s=TONE_S, freqs=freqs())
    h = r["hits"][0]
    assert h["switch_frames"] >= 1              # the raw tally still shows it
    assert h["switch_run_frames"] == 0          # …and the run rule discards it
    assert h["switch_max_run_ms"] == 0.0
    assert r["totals"]["hits_with_switches"] == 0


def test_the_run_length_for_a_switch_is_tunable() -> None:
    x = np.zeros(int(SR * 6.0))
    n = int(1.0 * SR)
    s = tone(freq_a(9), TONE_S)
    s[int(1.0 * SR):] = tone(freq_b(9), 1.0)
    x[n:n + len(s)] += s
    hit = [{"t": 1.0, "pad": 9}]
    strict = analyze_tone_tracks(x, hit, rate=SR, tone_s=TONE_S, freqs=freqs(),
                                 switch_min_frames=999)
    assert strict["hits"][0]["switch_frames"] > 0
    assert strict["hits"][0]["switch_run_frames"] == 0
    assert strict["switch_min_frames"] == 999


# --------------------------------------------------------------------------- late starts


def test_a_tone_that_starts_200_ms_late_is_a_late_start_not_a_dropout() -> None:
    """The measured case: a pad hit at t=5.50 whose tone began at ~5.72. Folding
    that into `max_gap_ms` would blame the swap for a 220 ms hole in a tone that
    then played perfectly."""
    x = np.zeros(int(SR * 10.0))
    n = int(5.70 * SR)                       # 200 ms after the hit at 5.50
    s = tone(freq_a(5), TONE_S)
    x[n:n + len(s)] += s
    r = analyze_tone_tracks(x, [{"t": 5.50, "pad": 5}], rate=SR, tone_s=TONE_S,
                            freqs=freqs())
    h = r["hits"][0]
    assert h["missing"] is False and h["kit"] == 0
    # the window opens at t + attack_s, so the leading quiet run is the 200 ms
    # delay minus that 50 ms trim, rounded down to whole 20 ms frames
    assert h["late_start_ms"] == pytest.approx(200.0 - ATTACK_S * 1000.0, abs=20.0)
    assert h["late_start_ms"] == 140.0
    assert h["dropout_frames"] == 0          # excluded from the dropout numbers
    assert h["max_gap_ms"] == 0.0
    assert r["totals"]["late_starts"] == 1
    assert r["totals"]["late_start_max_ms"] == 140.0
    assert r["totals"]["max_gap_ms"] == 0.0


def test_a_late_start_and_a_real_hole_are_reported_separately() -> None:
    """Both at once: only the leading run is the late start, and the hole in the
    middle still has to be a hole."""
    x = np.zeros(int(SR * 8.0))
    n = int(1.20 * SR)                       # 200 ms after the hit at 1.00
    s = tone(freq_a(3), TONE_S)
    hole = int((1.0 + ATTACK_S + 0.80) * SR) - n
    s[hole:hole + int(0.040 * SR)] = 0.0
    x[n:n + len(s)] += s
    r = analyze_tone_tracks(x, [{"t": 1.0, "pad": 3}], rate=SR, tone_s=TONE_S,
                            freqs=freqs(), frame_ms=10.0)
    h = r["hits"][0]
    assert h["late_start_ms"] == pytest.approx(150.0, abs=10.0)
    assert h["dropout_frames"] == 4
    assert h["max_gap_ms"] == pytest.approx(40.0)


def test_a_tone_that_starts_on_time_has_no_late_start() -> None:
    x, hits = render([(1.0, 6, 0)], total_s=6.0)
    h = analyze_tone_tracks(x, hits, rate=SR, tone_s=TONE_S, freqs=freqs())["hits"][0]
    assert h["late_start_ms"] == 0.0


# --------------------------------------------------------------------------- missing


def test_a_hit_that_made_no_sound_is_missing() -> None:
    plan = [(1.0, 0, 0), (4.0, 1, None), (7.0, 2, 0)]
    x, hits = render(plan, total_s=12.0)
    r = analyze_tone_tracks(x, hits, rate=SR, tone_s=TONE_S, freqs=freqs())
    assert [h["missing"] for h in r["hits"]] == [False, True, False]
    assert r["hits"][1]["kit"] is None
    assert "noise floor" in r["hits"][1]["note"]
    assert r["totals"]["missing"] == 1
    assert r["totals"]["audible"] == 2


def test_a_tone_buried_in_its_own_pre_hit_noise_floor_is_missing() -> None:
    """The floor is measured per candidate just before the hit, so a band that
    was already this loud does not count as a pad that sounded."""
    x = np.zeros(int(SR * 6.0))
    x += tone(freq_a(4), 6.0)          # the band is occupied for the whole recording
    r = analyze_tone_tracks(x, [{"t": 2.0, "pad": 4}], rate=SR, tone_s=TONE_S, freqs=freqs())
    assert r["hits"][0]["missing"] is True


# --------------------------------------------------------------------------- plumbing


def test_reads_a_wav_from_disk_and_reports_its_rate(tmp_path: Path) -> None:
    x, hits = render([(1.0, 3, 1)], total_s=5.0)
    path = write_wav(tmp_path / "tones.wav", x, sr=SR)
    r = analyze_tone_tracks(path, hits, tone_s=TONE_S, freqs=freqs())
    assert r["sample_rate"] == SR
    assert r["frame_ms"] == 20.0 and r["tone_s"] == TONE_S
    assert r["hits"][0]["kit"] == 1


def test_an_array_without_a_rate_is_bad_args() -> None:
    with pytest.raises(HilError) as ei:
        analyze_tone_tracks(np.zeros(1000), [{"t": 0.0, "pad": 0}], tone_s=TONE_S,
                            freqs=freqs())
    assert ei.value.code == BAD_ARGS


def test_a_window_past_the_end_of_the_recording_is_reported_not_raised() -> None:
    """The tail can be short — the run was cancelled, or the capture stopped
    early. That is a hit with no verdict, not an exception mid-scoring."""
    x, hits = render([(1.0, 0, 0), (4.97, 1, 0)], total_s=5.0)
    r = analyze_tone_tracks(x, hits, rate=SR, tone_s=TONE_S, freqs=freqs())
    assert r["hits"][1]["frames"] == 0
    assert r["hits"][1]["missing"] is True
    assert "shorter than two frames" in r["hits"][1]["note"]


def test_a_pad_with_no_candidates_is_reported_not_raised() -> None:
    x, hits = render([(1.0, 0, 0)], total_s=5.0)
    r = analyze_tone_tracks(x, hits, rate=SR, tone_s=TONE_S, freqs={})
    assert r["hits"][0]["missing"] is True
    assert "no candidate frequencies" in r["hits"][0]["note"]
