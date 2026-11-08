"""The engines' rules as the model plays them."""
from __future__ import annotations

from crosspad_hil.jam.calibration import Calibration
from crosspad_hil.jam.model import INF, PitchedVoiceSim, SamplerVoiceSim, Voice

SR = 44_100


def v(vid: int, pad: int, on: float, dur: float = 2.0, shape: str = "flat", choke: int = 0,
      engine: str = "sampler") -> Voice:
    end = INF if shape == "loop" else on + dur * SR
    return Voice(vid=vid, src="host", cause=f"a{vid}", aid=vid, kit="k", pad=pad, engine=engine,
                 freqs=(1000.0 + pad,), f_on=on, f_nat=end, f_end=end, gain_l=0.07,
                 gain_r=0.07, shape=shape, choke=choke)


def test_sampler_steals_the_voice_furthest_into_its_sample() -> None:
    sim = SamplerVoiceSim(SR, Calibration())
    voices = [v(i, i, i * 1000.0) for i in range(8)]
    for x in voices:
        sim.press(x)
    ninth = v(9, 9, 20_000.0)
    ended = sim.press(ninth)
    assert ended == [("stolen", voices[0])] and voices[0].f_end == 20_000.0
    assert len(sim.active) == 8


def test_a_steal_tie_inside_one_block_is_left_open() -> None:
    sim = SamplerVoiceSim(SR, Calibration())
    voices = [v(i, i, 100.0 + (10 if i < 2 else 1000 * i)) for i in range(8)]
    for x in voices:
        sim.press(x)
    sim.press(v(9, 9, 50_000.0))
    assert voices[0].end_uncertain and voices[1].end_uncertain
    assert set(voices[0].steal_cands) == {0, 1}


def test_one_voice_per_pad_a_hit_restarts_it() -> None:
    sim = SamplerVoiceSim(SR, Calibration())
    a, b = v(1, 3, 0.0), v(2, 3, 5000.0)
    sim.press(a)
    assert sim.press(b) == [("retrigger", a)] and b.restart and a.f_end == 5000.0


def test_choke_hold_and_loop() -> None:
    sim = SamplerVoiceSim(SR, Calibration())
    a, b = v(1, 12, 0.0, choke=1), v(2, 13, 1000.0, choke=1)
    sim.press(a)
    assert sim.press(b) == [("choked", a)]
    h = v(3, 14, 2000.0, dur=3.0, shape="hold")
    sim.press(h)
    sim.release(2000.0 + 0.5 * SR, 14, "k")
    assert h.end_kind == "released" and abs(h.f_end - (2000.0 + 0.505 * SR)) < 1
    lp = v(4, 15, 3000.0, shape="loop")
    sim.press(lp)
    off = v(5, 15, 9000.0, shape="loop")
    assert sim.press(off) == [("loop_toggle", lp)] and off.end_kind == "toggle_off"


def test_pitched_cap_takes_a_releasing_voice_first_then_the_oldest() -> None:
    sim = PitchedVoiceSim(SR, Calibration(), cap=3)
    a, b, c = (v(i, i, i * 100.0, engine="pitched") for i in range(3))
    sim.press(a, 1.0)
    sim.press(b, 0.001)           # released almost at once: in release
    sim.press(c, 1.0)
    d = v(3, 3, 1000.0, engine="pitched")
    ended = sim.press(d, 1.0)
    assert ended == [("stolen", b)]                 # the releasing one, not the oldest
    e = v(4, 4, 2000.0, engine="pitched")
    assert sim.press(e, 1.0) == [("stolen", a)]     # then the oldest held
    assert a.f_end == 2000.0 + 0.005 * SR


def test_pitched_retrigger_releases_the_previous_voice() -> None:
    sim = PitchedVoiceSim(SR, Calibration())
    a, b = v(1, 7, 0.0, engine="pitched"), v(2, 7, 4000.0, engine="pitched")
    sim.press(a, 1.0)
    assert sim.press(b, 1.0) == [("retrigger_release", a)]
    assert a.end_kind == "retrigger_release" and abs(a.f_end - (4000.0 + 0.1 * SR)) < 1
