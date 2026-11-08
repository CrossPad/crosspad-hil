"""The phase kernels: sample slips, glitch periodicity, cross-run determinism.

These are the measurements `glitch_analyze.py` had and the other kinds do not. Each
signal below is built so that exactly one of them has to fire: a tone with a sample
deleted is spectrally indistinguishable from a clean one, a tone with a pop added is
not, and only the first is a slipping stream.
"""
from __future__ import annotations

import numpy as np
import pytest

from crosspad_hil.analyze import (
    KINDS,
    analyze,
    analyze_sine,
    glitch_determinism,
    glitch_periodicity,
    phase_residual,
    timing_slips,
)
from tests.test_analyze import SR, chirp, noise, tone, write_wav

FREQ = 997.0


def pops(x: np.ndarray, times, amp: float = 0.8) -> np.ndarray:
    """One-sample discontinuities: loud, local, and not a timing error."""
    y = x.copy()
    for t in times:
        y[int(t * SR)] += amp
    return y


def drop_samples(x: np.ndarray, t: float, count: int = 1) -> np.ndarray:
    """Delete `count` samples — what a stream that skipped a frame actually did."""
    return np.delete(x, np.arange(int(t * SR), int(t * SR) + count))


# --------------------------------------------------------------------------- primitives


def test_phase_residual_is_flat_and_names_the_frequency() -> None:
    resid, est, band_ratio = phase_residual(tone(FREQ, 4.0) + noise(4.0), SR, FREQ)
    assert est == pytest.approx(FREQ, abs=0.5)
    assert band_ratio > 0.9
    assert float(np.ptp(resid)) < 0.05          # rad, i.e. well under a third of a sample


def test_phase_residual_reports_the_band_share_not_the_loudest_tone() -> None:
    """A capture with no energy near the asked-for frequency must be visible as such."""
    _resid, _est, band_ratio = phase_residual(tone(5000.0, 3.0), SR, FREQ)
    assert band_ratio < 0.01


def test_timing_slips_counts_a_dropped_sample() -> None:
    x = drop_samples(tone(FREQ, 4.0) + noise(4.0), 2.0)
    resid, _est, _r = phase_residual(x, SR, FREQ)
    slips, drift, p2p = timing_slips(resid, SR, FREQ)
    assert slips == 1
    assert p2p == pytest.approx(2 * np.pi * FREQ / SR, rel=0.15)   # one whole sample of phase
    # The straight-line fit absorbs part of a step taken mid-recording, so the leftover
    # end-to-end drift understates it — the slip count, not the drift, is the detector.
    assert 0.3 < abs(drift) < 0.8


def test_timing_slips_ignores_a_pop() -> None:
    """A pop is a discontinuity in amplitude; the stream kept every sample."""
    resid, _est, _r = phase_residual(pops(tone(FREQ, 4.0), [1.13, 2.13, 3.13]), SR, FREQ)
    assert timing_slips(resid, SR, FREQ)[0] == 0


def test_glitch_periodicity_finds_a_regular_train() -> None:
    idxs = np.arange(10) * (SR // 10) + 1000            # one every 100 ms
    r = glitch_periodicity(idxs, SR * 2, SR)
    assert r["iv_median_ms"] == pytest.approx(100.0, abs=0.5)
    assert r["train_peak_hz"] == pytest.approx(10.0, abs=1.0)
    assert r["intervals_top"]["100.0ms"] == 9


def test_glitch_periodicity_needs_three_points() -> None:
    assert glitch_periodicity(np.array([10, 20]), SR, SR) == {}


def test_glitch_determinism_scores_repeated_positions() -> None:
    base = np.array([1000, 5000, 9000])
    assert glitch_determinism([base, base + 40, base - 30]) == 1.0
    # Two of three land within tolerance on every other run; the third moves too far.
    moved = np.array([1000, 5000, 40000])
    assert glitch_determinism([base, moved]) == pytest.approx(2 / 3, abs=0.01)
    assert glitch_determinism([base, np.array([], dtype=np.int64)]) == 0.0


def test_glitch_determinism_needs_two_runs_with_something_in_them() -> None:
    assert glitch_determinism([np.array([1000])]) is None
    assert glitch_determinism([np.array([], dtype=np.int64), np.array([10])]) is None


# --------------------------------------------------------------------------- kind


def test_sine_passes_on_a_clean_tone(tmp_path) -> None:
    p = write_wav(tmp_path / "s.wav", tone(FREQ, 5.0, amp=0.4) + noise(5.0))
    r = analyze_sine(p).to_dict()
    assert r["verdict"] == "pass" and r["passed"]
    assert r["est_freq_hz"] == pytest.approx(FREQ, abs=0.5)
    assert r["timing_slips"] == 0 and r["glitches"] == 0
    assert abs(r["net_drift_samples"]) < 1.0


def test_sine_fails_on_a_dropped_sample(tmp_path) -> None:
    x = drop_samples(tone(FREQ, 5.0, amp=0.4) + noise(5.0), 2.5)
    r = analyze_sine(write_wav(tmp_path / "s.wav", x)).to_dict()
    assert r["verdict"] == "fail" and r["timing_slips"] == 1
    assert any("timing slip" in reason for reason in r["reasons"])


def test_sine_warns_on_pops_without_calling_them_slips(tmp_path) -> None:
    x = pops(tone(FREQ, 5.0, amp=0.1), [1.37, 1.87, 2.37, 2.87])
    r = analyze_sine(write_wav(tmp_path / "s.wav", x)).to_dict()
    assert r["verdict"] == "warn" and r["passed"]
    assert r["timing_slips"] == 0 and r["glitches"] >= 4
    assert r["periodicity"]["iv_median_ms"] == pytest.approx(500.0, abs=1.0)


def test_sine_fails_when_the_tone_came_back_detuned(tmp_path) -> None:
    p = write_wav(tmp_path / "s.wav", tone(FREQ * 1.05, 5.0, amp=0.4))
    r = analyze_sine(p).to_dict()
    assert r["verdict"] == "fail"
    assert any("% off" in reason for reason in r["reasons"])


def test_sine_fails_when_there_is_no_tone_at_that_frequency(tmp_path) -> None:
    p = write_wav(tmp_path / "s.wav", tone(5000.0, 5.0, amp=0.4))
    r = analyze_sine(p).to_dict()
    assert r["verdict"] == "fail"
    assert any("no tone near" in reason for reason in r["reasons"])


def test_sine_fails_on_a_silent_capture(tmp_path) -> None:
    r = analyze_sine(write_wav(tmp_path / "q.wav", noise(5.0))).to_dict()
    assert r["verdict"] == "fail" and "no return signal" in r["reasons"][0]


def test_sine_fails_on_a_capture_too_short_to_measure(tmp_path) -> None:
    r = analyze_sine(write_wav(tmp_path / "t.wav", tone(FREQ, 0.1))).to_dict()
    assert r["verdict"] == "fail" and "only" in r["reasons"][0]


def test_sine_aligns_to_a_reference_and_judges_only_the_peak(tmp_path) -> None:
    """A tone is periodic: the correlation peak is trustworthy, the lag is not."""
    ref = write_wav(tmp_path / "ref.wav", tone(FREQ, 3.0, amp=0.4))
    cap = np.concatenate([noise(0.4), tone(FREQ, 3.0, amp=0.4) + noise(3.0)])
    r = analyze_sine(write_wav(tmp_path / "c.wav", cap), ref=ref, edge_s=0.3).to_dict()
    assert r["passed"] and r["align"]["corr_peak"] > 0.8


def test_sine_fails_when_the_capture_is_not_the_reference(tmp_path) -> None:
    ref = write_wav(tmp_path / "ref.wav", chirp(100.0, 10000.0, 3.0))
    cap = write_wav(tmp_path / "c.wav", tone(FREQ, 4.0, amp=0.4) + noise(4.0))
    r = analyze_sine(cap, ref=ref).to_dict()
    assert r["verdict"] == "fail"
    assert any("does not correlate" in reason for reason in r["reasons"])


def test_sine_compares_takes_for_deterministic_glitch_positions(tmp_path) -> None:
    base = tone(FREQ, 5.0, amp=0.1)
    same = [write_wav(tmp_path / f"r{i}.wav", pops(base, [1.37, 2.37, 3.37]))
            for i in range(3)]
    r = analyze_sine(same[0], compare=same[1:]).to_dict()
    assert r["determinism"]["runs"] == 3 and r["determinism"]["replicated"] == 1.0

    elsewhere = write_wav(tmp_path / "other.wav", pops(base, [1.13, 2.91, 3.09]))
    r2 = analyze_sine(same[0], compare=[elsewhere]).to_dict()
    assert r2["determinism"]["replicated"] == 0.0


def test_sine_is_reachable_through_the_dispatch(tmp_path) -> None:
    p = write_wav(tmp_path / "s.wav", tone(440.0, 5.0, amp=0.4) + noise(5.0))
    assert "sine" in KINDS
    r = analyze("sine", p, expected={"freq": 440.0})
    assert r["kind"] == "sine" and r["passed"]
