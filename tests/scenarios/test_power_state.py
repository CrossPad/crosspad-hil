"""Scenario `power_state`: decoding the STM power debug words and judging a run.

Byte layouts mirror CrossPad_STM32_r2 reg_map.c REG_DEBUG +0x7C..+0x8F and
charger_debug(); the run loop is exercised against a fake STM_REG endpoint.
"""
from __future__ import annotations

import threading
from pathlib import Path
from typing import Any

import pytest

from crosspad_hil.scenarios import base, power_state

# Healthy profile as measured on dev_1ede with STM 1.9 + power fixes.
GOOD_REGS = (0x12, 0x70, 0x1A, 0xED)
# The same chip after its watchdog expired unfed (measured on STM 1.8, 89d3785).
RESET_REGS = (0x52, 0x30, 0x3A, 0x9D)


def raw(regs=GOOD_REGS, *, reapplied=0, ico=3, gate=1, nobatt=0, target=1000, pg=1,
        chrg=2, sleep=0, led=0, pwr=0, rst=0x04, uptime=120_000) -> bytes:
    words = [
        regs[0] | regs[1] << 8 | regs[2] << 16 | regs[3] << 24,
        reapplied | ico << 8 | gate << 16 | nobatt << 24,
        target | pg << 16 | chrg << 24,
        sleep | led << 8 | pwr << 16 | rst << 24,
        uptime,
    ]
    return b"".join(w.to_bytes(4, "little") for w in words)


def test_decode_names_every_field() -> None:
    s = power_state.decode(raw(reapplied=2, ico=1, gate=2, nobatt=1, target=750, pg=0,
                               chrg=3, sleep=2, led=1, pwr=5, uptime=91_234))
    assert (s["reg00"], s["reg02"], s["reg03"], s["reg07"]) == GOOD_REGS
    assert s["cfg_reapplied"] == 2 and s["ico_state"] == "BASELINE" and s["gate"] == "STM"
    assert s["no_battery"] and s["ico_target_ma"] == 750 and not s["power_good"]
    assert s["chrg_stat"] == 3 and s["sleep_state"] == "USB_SUSPENDED"
    assert s["led1_mode"] == "MANUAL" and s["pwr_state"] == "CHARGE"
    assert s["uptime_ms"] == 91_234


def test_good_profile_has_no_problems() -> None:
    assert power_state.profile_problems(power_state.decode(raw())) == []


def test_chip_default_mode_is_named_bit_by_bit() -> None:
    problems = power_state.profile_problems(power_state.decode(raw(RESET_REGS)))
    text = " ".join(problems)
    for needle in ("EN_ILIM", "CONV_RATE", "OTG_CONFIG", "watchdog code 1"):
        assert needle in text


def test_evaluate_clean_run() -> None:
    samples = [power_state.decode(raw(uptime=t)) for t in (10_000, 60_000, 120_000)]
    assert power_state.evaluate(samples) == []


def test_evaluate_flags_reapply_reset_and_stuck_states() -> None:
    samples = [
        power_state.decode(raw(uptime=50_000)),
        power_state.decode(raw(uptime=95_000, reapplied=1)),
        power_state.decode(raw(uptime=3_000, sleep=2, pwr=2, led=1)),
    ]
    findings = " | ".join(power_state.evaluate(samples))
    assert "re-applied during the run" in findings
    assert "STM reset during the run" in findings
    assert "USB_SUSPENDED" in findings and "SHIP_PENDING" in findings
    assert "LED1 in MANUAL" in findings


def test_evaluate_flags_a_watchdog_reset() -> None:
    findings = power_state.evaluate([power_state.decode(raw(rst=0x20))])
    assert any("IWDG reset" in f for f in findings)
    assert power_state.evaluate([power_state.decode(raw(rst=0x04))]) == []


def test_evaluate_reports_each_problem_once() -> None:
    samples = [power_state.decode(raw(RESET_REGS, uptime=t)) for t in (90_000, 95_000)]
    findings = power_state.evaluate(samples)
    assert len([f for f in findings if f.startswith("EN_ILIM")]) == 1


# --------------------------------------------------------------------------- run loop


class FakeStm:
    """STM_REG answers: identity, then the power debug block per call."""

    def __init__(self, proto: int, blocks: list[bytes]) -> None:
        self.proto = proto
        self.blocks = blocks
        self.reads = 0

    def stm_reg(self, link: Any, addr: int, length: int) -> dict[str, Any]:
        if addr == power_state.PROTO_ADDR:
            data = bytes([self.proto >> 8, self.proto & 0xFF, 1, 9])
        else:
            data = self.blocks[min(self.reads, len(self.blocks) - 1)]
            self.reads += 1
        return {"bytes": list(data[:length])}


class FakeLink:
    def close(self) -> None:
        pass


def run(monkeypatch: pytest.MonkeyPatch, tmp_path: Path, stm: FakeStm,
        params: power_state.Params) -> base.Report:
    monkeypatch.setattr(power_state.verbs, "stm_reg", stm.stm_reg)
    ctx = base.Context(device=None, workdir=tmp_path, cancelled=threading.Event(),  # type: ignore[arg-type]
                       log=lambda _line: None)
    monkeypatch.setattr(ctx, "open_cdc", lambda *a, **k: FakeLink())
    return power_state.SCENARIO.run(ctx, params, base.Progress())


def test_run_waits_for_uptime_then_passes(monkeypatch, tmp_path) -> None:
    stm = FakeStm(0x0010, [raw(uptime=t) for t in (20_000, 40_000, 60_000, 80_000, 100_000)])
    rep = run(monkeypatch, tmp_path, stm, power_state.Params(settle_s=0.01, interval_s=0.0))
    assert rep.passed and rep.exit_code == 0
    assert rep.data["watchdog_judged"] and len(rep.data["samples"]) >= 5


def test_run_fails_when_the_chip_resets(monkeypatch, tmp_path) -> None:
    stm = FakeStm(0x0010, [raw(uptime=60_000), raw(RESET_REGS, uptime=100_000)])
    rep = run(monkeypatch, tmp_path, stm, power_state.Params(settle_s=0.01, interval_s=0.0))
    assert not rep.passed and rep.exit_code == 1
    assert any("EN_ILIM" in f for f in rep.data["findings"])


def test_old_stm_is_an_environment_problem(monkeypatch, tmp_path) -> None:
    rep = run(monkeypatch, tmp_path, FakeStm(0x000F, [raw()]), power_state.Params())
    assert rep.exit_code == 2 and "0x000F" in rep.summary


def test_stm_without_debug_words_is_an_environment_problem(monkeypatch, tmp_path) -> None:
    rep = run(monkeypatch, tmp_path, FakeStm(0x0010, [bytes(20)]), power_state.Params())
    assert rep.exit_code == 2 and "lacks" in rep.summary


def test_quick_mode_takes_one_sample(monkeypatch, tmp_path) -> None:
    stm = FakeStm(0x0010, [raw(uptime=5_000)])
    rep = run(monkeypatch, tmp_path, stm, power_state.Params(settle_s=0.0))
    assert rep.passed and len(rep.data["samples"]) == 1 and not rep.data["watchdog_judged"]
