"""Analysis kinds against synthesized WAVs — no audio hardware, no device.

Every signal here is built to make one verdict inevitable: a clean multitone must
pass, a multitone with one tone missing must fail on that tone, hits whose amplitude
tracks velocity must produce a monotonic curve, and so on. The point is that the
ported measurements still say what they said in platform-idf/tools.
"""
from __future__ import annotations

import math
import sys
import wave
from pathlib import Path

import numpy as np
import pytest

from crosspad_hil.analyze import (
    KINDS,
    align_to_ref,
    analyze,
    analyze_click,
    analyze_multitone,
    analyze_onset,
    analyze_psd,
    analyze_silence,
    analyze_velocity,
    click_metric,
    detect_glitches,
    wav_stats,
)
from crosspad_hil.errors import BAD_ARGS, NOT_SUPPORTED, HilError

SR = 48000
RNG = np.random.default_rng(7)


def write_wav(path: Path, data: np.ndarray, sr: int = SR) -> Path:
    """`data`: float in [-1, 1], shape (n,) mono or (n, ch)."""
    if data.ndim == 1:
        data = data.reshape(-1, 1)
    pcm = (np.clip(data, -1.0, 0.999969) * 32768.0).astype("<i2")
    with wave.open(str(path), "wb") as w:
        w.setnchannels(data.shape[1])
        w.setsampwidth(2)
        w.setframerate(sr)
        w.writeframes(pcm.tobytes())
    return path


def tone(freq: float, seconds: float, amp: float = 0.3, sr: int = SR) -> np.ndarray:
    t = np.arange(int(seconds * sr)) / sr
    return amp * np.sin(2 * np.pi * freq * t)


def noise(seconds: float, amp: float = 1e-4, sr: int = SR) -> np.ndarray:
    return amp * RNG.standard_normal(int(seconds * sr))


def chirp(f0: float, f1: float, seconds: float, amp: float = 0.5, sr: int = SR) -> np.ndarray:
    """Log sweep (glitch_capture.gen_chirp) — broadband, so cross-correlation is unique."""
    t = np.arange(int(seconds * sr)) / sr
    k = (f1 / f0) ** (1 / seconds)
    return amp * np.sin(2 * np.pi * f0 * (k ** t - 1) / math.log(k))


def multitone(freqs, seconds: float, amp: float = 0.25, sr: int = SR) -> np.ndarray:
    sig = np.zeros(int(seconds * sr))
    for i, f in enumerate(freqs):
        sig += amp * np.sin(2 * np.pi * f * np.arange(len(sig)) / sr + i * 1.234)
    return sig / max(1.0, np.abs(sig).max() / 0.7)


def padded(sig: np.ndarray, pad_s: float = 0.6, sr: int = SR) -> np.ndarray:
    """A real capture has silence at both ends — the block-RMS gate has to find it."""
    pad = noise(pad_s, sr=sr)
    return np.concatenate([pad, sig, pad])


def hits(schedule, *, decay_s: float = 0.18, freq: float = 220.0,
         seconds: float | None = None, sr: int = SR) -> np.ndarray:
    """Percussive hits: `schedule` = [(t_seconds, amplitude), …]."""
    total = seconds if seconds is not None else max(t for t, _ in schedule) + 1.0
    x = noise(total, sr=sr)
    env_n = int(decay_s * sr)
    env = np.exp(-np.arange(env_n) / (0.03 * sr))
    for t, amp in schedule:
        i = int(t * sr)
        burst = amp * env * np.sin(2 * np.pi * freq * np.arange(env_n) / sr)
        x[i:i + env_n] += burst[: len(x) - i]
    return x


# --------------------------------------------------------------------------- wav_stats


def test_wav_stats_reports_level_and_duration(tmp_path: Path) -> None:
    p = write_wav(tmp_path / "t.wav", tone(1000.0, 2.0, amp=0.5))
    st = wav_stats(p)
    assert st["seconds"] == pytest.approx(2.0, abs=0.01)
    assert st["sample_rate"] == SR and st["channels"] == 1 and st["frames"] == 2 * SR
    assert st["peak_dbfs"] == pytest.approx(-6.0, abs=0.5)     # 0.5 FS
    assert st["rms_dbfs"] == pytest.approx(-9.0, abs=0.5)      # 0.5/sqrt(2)


def test_wav_stats_without_numpy_matches_the_numpy_path(tmp_path: Path,
                                                        monkeypatch: pytest.MonkeyPatch) -> None:
    """Capture calls wav_stats to fill in peak/RMS — it must not need the audio extra."""
    p = write_wav(tmp_path / "t.wav", tone(1000.0, 0.5, amp=0.25))
    with_np = wav_stats(p)
    monkeypatch.setitem(sys.modules, "numpy", None)
    without_np = wav_stats(p)
    assert without_np["peak_dbfs"] == pytest.approx(with_np["peak_dbfs"], abs=0.05)
    assert without_np["rms_dbfs"] == pytest.approx(with_np["rms_dbfs"], abs=0.05)


def test_wav_stats_on_silence(tmp_path: Path) -> None:
    p = write_wav(tmp_path / "q.wav", np.zeros(SR))
    assert wav_stats(p)["peak_dbfs"] == -120.0


def test_missing_numpy_raises_not_supported(tmp_path: Path,
                                            monkeypatch: pytest.MonkeyPatch) -> None:
    p = write_wav(tmp_path / "t.wav", tone(997.0, 0.5))
    monkeypatch.setitem(sys.modules, "numpy", None)
    with pytest.raises(HilError) as e:
        analyze("click", p)
    assert e.value.code == NOT_SUPPORTED
    assert "crosspad-hil[audio]" in (e.value.hint or "")


def test_non_16bit_wav_is_rejected(tmp_path: Path) -> None:
    p = tmp_path / "f32.wav"
    with wave.open(str(p), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(4)
        w.setframerate(SR)
        w.writeframes(b"\0" * 4 * 100)
    with pytest.raises(HilError) as e:
        wav_stats(p)
    assert e.value.code == BAD_ARGS


# --------------------------------------------------------------------------- multitone


def test_multitone_passes_on_the_reference_signal(tmp_path: Path) -> None:
    p = write_wav(tmp_path / "m.wav", padded(multitone([440, 997, 3181], 3.0)))
    r = analyze_multitone(p).to_dict()
    assert r["verdict"] == "pass" and r["passed"] and r["signal_present"]
    for f in ("440", "997", "3181"):
        assert r["tones"][f]["band_power"] > 0.02
        assert abs(r["tones"][f]["offset_pct"]) < 1.0
    assert r["dropouts"] == 0 and r["analyzed_s"] > 1.0


def test_multitone_fails_when_a_tone_is_missing(tmp_path: Path) -> None:
    p = write_wav(tmp_path / "m.wav", padded(multitone([440, 3181], 3.0)))
    r = analyze_multitone(p).to_dict()
    assert r["verdict"] == "fail" and not r["passed"]
    assert any("997" in reason for reason in r["reasons"])


def test_multitone_fails_on_a_silent_capture(tmp_path: Path) -> None:
    p = write_wav(tmp_path / "q.wav", noise(3.0))
    r = analyze_multitone(p).to_dict()
    assert r["verdict"] == "fail" and r["signal_present"] is False
    assert "no return signal" in r["reasons"][0]


def test_multitone_warns_on_dropouts(tmp_path: Path) -> None:
    sig = multitone([440, 997, 3181], 3.0)
    sig[int(1.0 * SR): int(1.0 * SR) + SR // 200] = 0.0       # 5 ms of digital silence
    r = analyze_multitone(write_wav(tmp_path / "m.wav", padded(sig))).to_dict()
    assert r["verdict"] == "warn" and r["passed"]
    assert r["dropouts"] >= 1


# --------------------------------------------------------------------------- click


def test_click_is_clean_on_a_plain_tone(tmp_path: Path) -> None:
    p = write_wav(tmp_path / "c.wav", tone(440.0, 3.0, amp=0.4) + noise(3.0))
    r = analyze_click(p).to_dict()
    assert r["verdict"] == "pass" and r["clicks"] == 0


def test_click_finds_inserted_pops(tmp_path: Path) -> None:
    # A click is a discontinuity many times the *local* level — a pop no louder than
    # the programme material is inaudible and is deliberately not counted.
    x = tone(440.0, 3.0, amp=0.05)
    for t in (0.5, 1.25, 2.0):
        x[int(t * SR)] += 0.8                                  # one-sample pop
    r = analyze_click(write_wav(tmp_path / "c.wav", x)).to_dict()
    assert r["clicks"] == 3 and r["verdict"] == "fail"
    assert r["first_click_offsets_s"][0] == pytest.approx(0.5, abs=0.01)


def test_click_metric_collapses_a_burst_into_one_click() -> None:
    x = tone(440.0, 1.0, amp=0.05)
    x[24000:24005] += 0.8
    count, positions = click_metric(x, SR)
    assert count == 1 and positions[0] == pytest.approx(0.5, abs=0.005)


# --------------------------------------------------------------------------- onset


def test_onset_matches_every_scheduled_hit(tmp_path: Path) -> None:
    schedule = [round(0.5 + i * 0.4, 3) for i in range(8)]
    x = hits([(t + 0.02, 0.5) for t in schedule], seconds=5.0)   # 20 ms pipeline offset
    r = analyze_onset(write_wav(tmp_path / "o.wav", x), expected=schedule).to_dict()
    assert r["verdict"] == "pass" and r["expected"] == 8 and r["matched"] == 8
    assert r["missed"] == [] and r["extra"] == []
    assert r["latency_ms"]["p50"] is not None and abs(r["latency_ms"]["p50"]) < 5.0
    assert r["latency_ms"]["max"] < 10.0
    assert r["offset_ms"] == pytest.approx(20.0, abs=6.0)


def test_onset_reports_a_hit_that_never_sounded(tmp_path: Path) -> None:
    schedule = [round(0.5 + i * 0.4, 3) for i in range(6)]
    played = [(t, 0.5) for i, t in enumerate(schedule) if i != 3]
    r = analyze_onset(write_wav(tmp_path / "o.wav", hits(played, seconds=4.0)),
                      expected=schedule).to_dict()
    assert r["verdict"] == "fail" and r["missed"] == [3] and r["matched"] == 5


def test_onset_warns_on_an_unscheduled_onset(tmp_path: Path) -> None:
    schedule = [0.5, 1.5, 2.5]
    played = [(t, 0.5) for t in schedule] + [(3.4, 0.5)]
    r = analyze_onset(write_wav(tmp_path / "o.wav", hits(played, seconds=4.2)),
                      expected=schedule).to_dict()
    assert r["verdict"] == "warn" and r["passed"]
    assert len(r["extra"]) == 1 and r["extra"][0] == pytest.approx(3.4, abs=0.05)


def test_onset_accepts_a_bare_count(tmp_path: Path) -> None:
    played = [(0.5, 0.5), (1.2, 0.5), (1.9, 0.5)]
    r = analyze_onset(write_wav(tmp_path / "o.wav", hits(played, seconds=3.0)),
                      expected=3).to_dict()
    assert r["expected"] == 3 and r["matched"] == 3 and r["latency_ms"]["p50"] is None


def test_onset_needs_an_expectation(tmp_path: Path) -> None:
    p = write_wav(tmp_path / "o.wav", hits([(0.5, 0.5)], seconds=2.0))
    with pytest.raises(HilError) as e:
        analyze_onset(p)
    assert e.value.code == BAD_ARGS and "expected" in e.value.message


# --------------------------------------------------------------------------- velocity


def vel_schedule(vels, *, start: float = 0.5, period: float = 0.5) -> list[dict]:
    return [{"t": round(start + i * period, 3), "vel": v} for i, v in enumerate(vels)]


def test_velocity_curve_tracks_amplitude(tmp_path: Path) -> None:
    vels = [16, 32, 64, 96, 127]
    sched = vel_schedule(vels)
    x = hits([(h["t"] + 0.015, h["vel"] / 127 * 0.7) for h in sched], seconds=4.0)
    r = analyze_velocity(write_wav(tmp_path / "v.wav", x), expected=sched).to_dict()
    assert r["verdict"] == "pass", r["reasons"]
    assert r["matched"] == 5 and r["missed"] == []
    assert [c["vel"] for c in r["curve"]] == vels
    assert r["monotonic_inversions"] == 0
    assert r["dynamic_range_db"] == pytest.approx(20 * math.log10(127 / 16), abs=3.0)
    assert r["curve"][-1]["measured_rel_db"] == 0.0
    assert all(h["peak_dbfs"] is not None for h in r["per_hit"])


def test_velocity_fails_when_loudness_does_not_move(tmp_path: Path) -> None:
    sched = vel_schedule([16, 32, 64, 96, 127])
    x = hits([(h["t"], 0.5) for h in sched], seconds=4.0)        # every hit equally loud
    r = analyze_velocity(write_wav(tmp_path / "v.wav", x), expected=sched).to_dict()
    assert r["verdict"] == "fail"
    assert any("does not change loudness" in reason for reason in r["reasons"])


def test_velocity_flags_an_inverted_response(tmp_path: Path) -> None:
    vels = [16, 32, 64, 96, 127]
    sched = vel_schedule(vels)
    amps = [0.05, 0.6, 0.3, 0.5, 0.7]                            # 64 quieter than 32
    x = hits([(h["t"], a) for h, a in zip(sched, amps, strict=True)], seconds=4.0)
    r = analyze_velocity(write_wav(tmp_path / "v.wav", x), expected=sched).to_dict()
    assert r["monotonic_inversions"] >= 1 and r["verdict"] == "fail"


def test_velocity_requires_velocities_in_the_schedule(tmp_path: Path) -> None:
    p = write_wav(tmp_path / "v.wav", hits([(0.5, 0.5)], seconds=2.0))
    with pytest.raises(HilError) as e:
        analyze_velocity(p, expected=[0.5, 1.0])
    assert e.value.code == BAD_ARGS


# --------------------------------------------------------------------------- psd


def test_psd_finds_the_carrier(tmp_path: Path) -> None:
    p = write_wav(tmp_path / "s.wav", tone(997.0, 6.0, amp=0.5) + noise(6.0))
    r = analyze_psd(p, freq=997.0, skip_s=0.5, take_s=4.0).to_dict()
    assert r["verdict"] == "pass", r["reasons"]
    assert r["dom_freq_hz"] == pytest.approx(997.0, abs=5.0)
    assert r["harmonics"]["h2"] < -40


def test_psd_sees_250_hz_amplitude_modulation_as_sidebands(tmp_path: Path) -> None:
    t = np.arange(int(6.0 * SR)) / SR
    am = (1.0 + 0.3 * np.sin(2 * np.pi * 250.0 * t)) * 0.5 * np.sin(2 * np.pi * 997.0 * t)
    r = analyze_psd(write_wav(tmp_path / "am.wav", am), freq=997.0,
                    skip_s=0.5, take_s=4.0).to_dict()
    assert r["verdict"] == "warn"
    assert r["sidebands"]["+250"] > -30 and r["sidebands"]["-250"] > -30
    assert any("sideband" in reason for reason in r["reasons"])


def test_psd_fails_when_the_carrier_is_elsewhere(tmp_path: Path) -> None:
    p = write_wav(tmp_path / "s.wav", tone(440.0, 6.0, amp=0.5))
    r = analyze_psd(p, freq=997.0, skip_s=0.5, take_s=4.0).to_dict()
    assert r["verdict"] == "fail"
    assert r["dom_freq_hz"] == pytest.approx(440.0, abs=5.0)
    assert "not 997 Hz" in r["reasons"][0]


def test_psd_aligns_against_a_reference(tmp_path: Path) -> None:
    """A pure tone correlates strongly at every period-aligned lag, so only the
    correlation peak is meaningful here — the lag needs broadband material."""
    ref = tone(997.0, 4.0, amp=0.5)
    cap = np.concatenate([noise(0.25), ref])
    r = analyze_psd(write_wav(tmp_path / "cap.wav", cap), freq=997.0, skip_s=0.5, take_s=3.0,
                    ref=write_wav(tmp_path / "ref.wav", ref)).to_dict()
    assert r["align"]["corr_peak"] > 0.9
    assert r["verdict"] == "pass", r["reasons"]


def test_psd_fails_when_the_capture_is_not_the_reference(tmp_path: Path) -> None:
    ref = write_wav(tmp_path / "ref.wav", chirp(100.0, 10000.0, 4.0))
    cap = write_wav(tmp_path / "cap.wav", tone(997.0, 5.0, amp=0.5))
    r = analyze_psd(cap, freq=997.0, skip_s=0.5, take_s=3.0, ref=ref).to_dict()
    assert r["verdict"] == "fail"
    assert any("correlate" in reason for reason in r["reasons"])


def test_align_to_ref_finds_the_offset() -> None:
    ref = chirp(100.0, 10000.0, 2.0)                            # broadband: one true peak
    cap = np.concatenate([np.zeros(1234), ref])
    lag, peak = align_to_ref(cap, ref, SR)
    assert lag == pytest.approx(1234, abs=8) and peak > 0.9


# --------------------------------------------------------------------------- silence


def test_silence_passes_on_a_quiet_capture(tmp_path: Path) -> None:
    p = write_wav(tmp_path / "q.wav", np.column_stack([noise(4.0), noise(4.0)]))
    r = analyze_silence(p).to_dict()
    assert r["verdict"] == "pass" and r["channels"] == 2
    assert r["rms_dbfs"] < -60 and set(r["per_channel"]) == {"ch0", "ch1"}


def test_silence_fails_on_a_noisy_capture(tmp_path: Path) -> None:
    p = write_wav(tmp_path / "n.wav", noise(4.0, amp=0.05))
    r = analyze_silence(p).to_dict()
    assert r["verdict"] == "fail"
    assert any("noise floor" in reason for reason in r["reasons"])


def test_silence_counts_spontaneous_artifacts(tmp_path: Path) -> None:
    x = noise(4.0)
    for t in (1.5, 2.0, 2.5, 3.0):
        x[int(t * SR)] += 0.4
    r = analyze_silence(write_wav(tmp_path / "g.wav", x)).to_dict()
    assert r["glitches"] >= 4 and r["verdict"] == "fail"
    assert any("artifacts" in reason for reason in r["reasons"])


def test_detect_glitches_ignores_ordinary_curvature() -> None:
    assert len(detect_glitches(tone(997.0, 2.0, amp=0.5))) == 0


# --------------------------------------------------------------------------- dispatch


def test_analyze_dispatches_every_kind(tmp_path: Path) -> None:
    wavs = {
        "multitone": write_wav(tmp_path / "mt.wav", padded(multitone([440, 997, 3181], 3.0))),
        "click": write_wav(tmp_path / "cl.wav", tone(440.0, 2.0, amp=0.4)),
        "psd": write_wav(tmp_path / "ps.wav", tone(997.0, 6.0, amp=0.5)),
        "silence": write_wav(tmp_path / "si.wav", noise(3.0)),
        "sine": write_wav(tmp_path / "sn.wav", tone(997.0, 6.0, amp=0.5)),
    }
    for kind, path in wavs.items():
        r = analyze(kind, path)
        assert r["kind"] == kind and r["wav"] == str(path)
        assert r["verdict"] in ("pass", "warn", "fail") and isinstance(r["passed"], bool)
    sched = vel_schedule([32, 127], period=1.0)
    x = hits([(h["t"], h["vel"] / 127 * 0.7) for h in sched], seconds=3.0)
    wav = write_wav(tmp_path / "hi.wav", x)
    assert analyze("onset", wav, expected=[h["t"] for h in sched])["matched"] == 2
    assert analyze("velocity", wav, expected=sched)["matched"] == 2
    assert set(KINDS) == set(wavs) | {"onset", "velocity"}


def test_analyze_rejects_an_unknown_kind(tmp_path: Path) -> None:
    p = write_wav(tmp_path / "t.wav", tone(440.0, 1.0))
    with pytest.raises(HilError) as e:
        analyze("spectrogram", p)
    assert e.value.code == BAD_ARGS and "multitone" in (e.value.hint or "")


def test_analyze_rejects_a_missing_file(tmp_path: Path) -> None:
    with pytest.raises(HilError) as e:
        analyze("click", tmp_path / "nope.wav")
    assert e.value.code == BAD_ARGS


def test_analyze_passes_options_through_expected(tmp_path: Path) -> None:
    p = write_wav(tmp_path / "m.wav", padded(multitone([300, 900], 3.0)))
    r = analyze("multitone", p, expected={"freqs": [300, 900]})
    assert r["verdict"] == "pass" and set(r["tones"]) == {"300", "900"}
