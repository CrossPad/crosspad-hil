"""card_hold: the audio-nodes service paused for a run, and always started again."""
from __future__ import annotations

import subprocess

import pytest

from crosspad_hil import card_hold, pins
from crosspad_hil.bench import Bench

DEV = "dev_1ede"
ME = "crosspad-stm32-r20-e9"


@pytest.fixture(autouse=True)
def _i_hold_the_bench() -> None:
    Bench(log=lambda m: None).claim(DEV, ME, "live_set validation")


class Systemd:
    def __init__(self, active: bool = True, start_works: bool = True) -> None:
        self.active, self.start_works = active, start_works
        self.calls: list[str] = []

    def __call__(self, argv: list[str]) -> subprocess.CompletedProcess:
        verb = argv[2]
        self.calls.append(verb)
        if verb == "stop":
            self.active = False
        elif verb == "start" and self.start_works:
            self.active = True
        out = "active\n" if self.active else "inactive\n"
        return subprocess.CompletedProcess(argv, 0 if self.active else 3, out, "")


def run_hold(sd: Systemd, log=lambda m: None, **kw) -> dict:
    kw = {"device": DEV, "holder": ME, **kw}
    with card_hold.exclusive_card(log, run=sd, sleep=lambda s: None, **kw) as info:
        seen = dict(info)
        seen["active_inside"] = sd.active
        seen["journal_inside"] = pins.pending(card_hold.PIN_DEVICE)
    return seen


def test_paused_for_the_run_and_started_again() -> None:
    sd = Systemd(active=True)
    seen = run_hold(sd)
    assert seen["paused"] is True and seen["active_inside"] is False
    assert seen["journal_inside"] == {"audio_nodes": "active"}   # a kill now is recoverable
    assert sd.active is True and pins.pending(card_hold.PIN_DEVICE) == {}


def test_started_again_when_the_run_fails() -> None:
    sd = Systemd(active=True)
    with pytest.raises(RuntimeError):
        with card_hold.exclusive_card(lambda m: None, device=DEV, holder=ME,
                                      run=sd, sleep=lambda s: None):
            raise RuntimeError("the block blew up")
    assert sd.active is True and pins.pending(card_hold.PIN_DEVICE) == {}


def test_a_service_that_was_not_running_is_left_alone() -> None:
    sd = Systemd(active=False)
    assert run_hold(sd)["paused"] is False
    assert sd.calls == ["is-active"]


def test_disabled_touches_nothing() -> None:
    sd = Systemd(active=True)
    assert run_hold(sd, enabled=False)["paused"] is False and sd.calls == []


def test_a_pause_a_killed_run_left_is_undone() -> None:
    """kill -9 between the stop and the start: the journal still says "active"."""
    pins.record(card_hold.PIN_DEVICE, card_hold.PIN_KEY, "active")
    sd = Systemd(active=False)
    run_hold(sd)
    assert sd.active is True and pins.pending(card_hold.PIN_DEVICE) == {}


def test_a_service_that_will_not_start_stays_in_the_journal() -> None:
    sd = Systemd(active=True, start_works=False)
    run_hold(sd)
    assert pins.pending(card_hold.PIN_DEVICE) == {"audio_nodes": "active"}


def test_the_pause_says_who_and_for_how_long() -> None:
    said: list[str] = []
    seen = run_hold(Systemd(active=True), log=said.append, expected_s=300)
    assert seen["holder"] == ME and seen["expected_s"] == 300
    line = next(m for m in said if "paused by" in m)
    assert ME in line and DEV in line and "~5 min (until ~" in line


def test_the_holder_may_come_from_the_environment(monkeypatch) -> None:
    monkeypatch.setenv("CROSSPAD_BENCH_HOLDER", ME)
    assert run_hold(Systemd(active=True), holder=None)["paused"] is True


@pytest.mark.parametrize("holder, why", [
    (None, "names no bench holder"),
    ("someone-else", f"is {ME}'s, not someone-else's"),
])
def test_only_the_bench_holder_may_pause_it(holder, why) -> None:
    sd = Systemd(active=True)
    said: list[str] = []
    seen = run_hold(sd, log=said.append, holder=holder)
    assert seen["paused"] is False and why in seen["refused"]
    assert "stop" not in sd.calls and seen["active_inside"] is True
    assert any("goes through PipeWire" in m for m in said)


def test_nobody_holding_the_lease_is_a_refusal_too() -> None:
    Bench(log=lambda m: None).release(DEV, ME)
    seen = run_hold(Systemd(active=True))
    assert seen["paused"] is False and "nobody holds" in seen["refused"]


def test_a_left_pause_is_undone_at_once_for_a_run_that_may_not_pause() -> None:
    pins.record(card_hold.PIN_DEVICE, card_hold.PIN_KEY, "active")
    sd = Systemd(active=False)
    seen = run_hold(sd, holder="someone-else")
    assert seen["paused"] is False and seen["active_inside"] is True
    assert pins.pending(card_hold.PIN_DEVICE) == {}


def test_for_run_takes_the_device_and_holder_of_the_scenario() -> None:
    from types import SimpleNamespace

    said: list[str] = []
    ctx = SimpleNamespace(log=said.append, holder=ME, device=SimpleNamespace(id=DEV))
    data: dict = {}
    sd = Systemd(active=True)
    with card_hold.for_run(ctx, data, expected_s=40, run=sd, sleep=lambda s: None) as hold:
        assert hold["paused"] is True and sd.active is False
    assert sd.active is True
    assert any("paused by " + ME in m for m in data["card_hold_log"]) and said
