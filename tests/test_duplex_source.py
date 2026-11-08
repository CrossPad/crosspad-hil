"""The duplex stream plays a continuous source (the jam's pilot) and logs its events."""
from __future__ import annotations

import time
from pathlib import Path

import numpy as np

from crosspad_hil import duplex
from crosspad_hil.jam.pilot import PilotSource
from crosspad_hil.jam.spec import PilotSpec
from tests.test_latency_units import FakeAlsa

SR = 44_100


def test_the_source_is_what_reaches_the_card_sample_for_sample(tmp_path: Path) -> None:
    alsa = FakeAlsa()
    src = PilotSource(PilotSpec(), SR, 2)
    rec = duplex.DuplexRecording(alsa, "plughw:Fake", tmp_path / "t.wav", rate=SR,
                                 source=src.render, decimate_s=0.01)
    rec.start()
    time.sleep(0.3)
    rec.stop()
    out = np.frombuffer(bytes(alsa.out), dtype="<i2").reshape(-1, 2)
    want = np.frombuffer(src.render(0, len(out)), dtype="<i2").reshape(-1, 2)
    assert len(out) > SR // 10 and np.array_equal(out, want)
    pts = rec.capture_points()
    assert 5 <= len(pts) <= 0.3 / 0.01 + 10            # decimated: one per 10 ms bin


def test_point_log_keeps_the_earliest_looking_point_of_each_bin() -> None:
    log = duplex.PointLog(1000.0, bin_s=0.010)
    for f, t in [(0, 0.0005), (1, 0.0025), (2, 0.0021), (12, 0.0130), (15, 0.0151)]:
        log.add(f, t)
    # bin 0: residuals 0.0005, 0.0015, 0.0001 -> frame 2; bin 1: 0.001, 0.0001 -> frame 15
    assert log.snapshot() == [(2, 0.0021), (15, 0.0151)]


def test_a_mute_pause_is_logged_and_silences_the_source(tmp_path: Path) -> None:
    alsa = FakeAlsa()
    src = PilotSource(PilotSpec(), SR, 2)
    rec = duplex.DuplexRecording(alsa, "plughw:Fake", tmp_path / "t.wav", rate=SR,
                                 source=src.render)
    rec.start()
    time.sleep(0.1)
    rec.pause_out(0.15, mode="drop")        # FakeAlsa cannot unlink: falls back to silence
    time.sleep(0.4)
    rec.stop()
    kinds = [e["kind"] for e in rec.events]
    assert "out_pause_fallback" in kinds and "out_mute" in kinds and "out_unmute" in kinds
    out = np.frombuffer(bytes(alsa.out), dtype="<i2").reshape(-1, 2)[:, 0]
    assert (out == 0).sum() > SR // 20                 # a stretch of silence was played
    assert rec.health()["events"]
