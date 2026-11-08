"""The clocks a glitch is named on."""
from __future__ import annotations

import random

import pytest

from crosspad_hil.jam.clocks import CaptureClock, DeviceLogClock, TimerOffset, dev_ts

SR = 44_100


def test_the_device_log_clock_from_bursty_arrivals_and_a_reboot() -> None:
    rng = random.Random(3)
    rows = []
    seq = 0
    boot = 1000.0
    for k in range(400):                                   # 400 s of log, 1 line a second
        ms = 500 + k * 1000
        t = boot + ms / 1000.0 * (1 + 20e-6)               # the board's clock 20 ppm fast
        late = rng.choice([0.0, 0.0, 0.0, rng.uniform(0.0, 0.2)])  # CDC/USB bursts
        line = f"I ({ms}) tag: line {k}"
        seq += 1
        rows.append((seq, t + late + (len(line) + 2) * 10 / 115200, 0.0, line))
    for k in range(30):                                    # rebooted: the log clock restarts
        ms = 300 + k * 1000
        seq += 1
        rows.append((seq, 1500.0 + ms / 1000.0 + 0.001, 0.0, f"I ({ms}) tag: again"))
    clk = DeviceLogClock.fit(rows)
    assert len(clk.epochs) == 2
    t = clk.to_host(200_500, clk.epoch_of(seq=200))
    assert t == pytest.approx(boot + 200.5 * (1 + 20e-6), abs=0.002)
    assert clk.to_host(10_300, clk.epoch_of(seq=seq)) == pytest.approx(1510.3, abs=0.003)


def test_dev_ts_ignores_ansi_and_printf_lines() -> None:
    assert dev_ts("\x1b[0;32mI (12345) tag: x\x1b[0m") == 12345
    assert dev_ts("engine underrun_at=5ms") is None


def test_the_capture_clock_is_split_at_an_xrun() -> None:
    rng = random.Random(1)
    pts = [(k, 50.0 + k / SR + abs(rng.gauss(0, 3e-4))) for k in range(0, 10 * SR, SR // 100)]
    # an xrun at 55 s: the take after it is 0.1 s later on the host clock
    pts = [(f, t + (0.1 if t > 55.0 else 0.0)) for f, t in pts]
    clk = CaptureClock.from_points(pts, SR, breaks=[55.0])
    assert len(clk.fits) == 2
    assert clk.t_of(2 * SR) == pytest.approx(52.0, abs=0.002)
    assert clk.frame_of(57.1) == pytest.approx(7 * SR, abs=0.002 * SR)


def test_timer_offset_is_the_median_of_the_pairs() -> None:
    assert TimerOffset.fit([(1760.0, 1000.0), (2761.0, 2000.0), (3759.0, 3000.0)]).offset_ms \
        == 760.0
    assert TimerOffset.fit([]).offset_ms == 760.0
