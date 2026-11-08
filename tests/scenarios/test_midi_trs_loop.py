"""Scenario `midi_trs_loop`: the stream encoding and the judgement of what came back."""
from __future__ import annotations

import pytest

from crosspad_hil.scenarios.midi_trs_loop import PERIOD, analyze, ber_from_mer, decode, encode


def test_encode_decode_round_trip_over_a_period() -> None:
    for seq in range(PERIOD):
        msg = encode(seq)
        assert msg[0] == 0x90 and msg[1] < 32 and 1 <= msg[2] <= 127
        assert decode(msg) == seq


def test_decode_refuses_what_is_not_the_stream() -> None:
    assert decode([0x9F, 1, 1]) is None      # another channel
    assert decode([0x90, 40, 1]) is None     # a pad's note
    assert decode([0x90, 1, 0]) is None      # velocity 0 = note-off
    assert decode([0x90, 1]) is None


def test_analyze_clean_stream() -> None:
    sent = [i * 0.001 for i in range(100)]
    rx = [(t + 0.002, encode(i)) for i, t in enumerate(sent)]
    r = analyze(sent, rx)
    assert r["delivered"] == 100 and r["lost"] == 0 and r["in_order"] == 99
    assert r["gaps"] == 0 and r["spurious"] == 0 and r["foreign"] == 0
    assert r["lat_p50_ms"] == pytest.approx(2.0)
    assert r["goodput"] == pytest.approx(1000.0, rel=1e-6)


def test_analyze_loss_corruption_and_foreign() -> None:
    sent = [i * 0.001 for i in range(10)]
    rx = [(sent[0] + 0.001, encode(0)), (sent[1] + 0.001, encode(1)),
          (sent[3] + 0.001, encode(3)),                     # 2 lost: a gap
          (sent[4] + 0.001, [0xB0, 7, 100]),                # foreign
          (sent[5] + 0.001, encode(9)),                     # 9 before it was sent: corrupted
          (sent[6] + 0.001, encode(1))]                     # a repeat
    r = analyze(sent, rx)
    assert r["delivered"] == 3 and r["lost"] == 7
    assert r["in_order"] == 1 and r["gaps"] == 1
    assert r["spurious"] == 2 and r["foreign"] == 1


def test_analyze_untimed_source_skips_latency() -> None:
    sent = [10.0 + i for i in range(5)]       # the ESP's clock, not ours
    rx = [(1.0 + i, encode(i)) for i in range(5)]
    r = analyze(sent, rx, timed=False)
    assert r["delivered"] == 5 and r["lat_p50_ms"] is None


def test_ber_from_message_error_rate() -> None:
    assert ber_from_mer(0, 1000) == 0.0
    assert ber_from_mer(1000, 1000) == 1.0
    assert ber_from_mer(30, 1000) == pytest.approx(1.0e-3, rel=0.02)
    assert ber_from_mer(0, 0) is None
