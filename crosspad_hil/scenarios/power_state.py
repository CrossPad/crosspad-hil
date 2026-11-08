"""STM power path, unattended: does the charger keep its profile, and did the
power state machine end up where it should?

Reads the STM power debug words (REG_DEBUG +0x7C..+0x8F, STM firmware 1.9 with
the power fixes) over ``STM_REG`` on the ESP CDC, once at the start and then
every ``interval_s`` until the STM has been up long enough for the BQ25895's
80 s I2C watchdog to have expired at least once. Nothing on the board is
written; no SWD, no person at the bench.

What fails the run:

  * the charger profile is gone — EN_ILIM set (the ILIM pin limits input
    current), CONV_RATE clear, auto-DPDM on the floating D+/D-, OTG_CONFIG set
    (boost back-feeds the ports once the cable is out), watchdog not 80 s.
    That is the chip in default mode: its watchdog expired unfed.
  * the profile had to be re-applied during the run (the STM noticed the chip
    reset itself — the watchdog is not being fed)
  * the STM reset during the run (its uptime went backwards)
  * the sleep state machine is stuck in USB_SUSPENDED while the board answers
    (it is awake, so that state is stale), or the ESP-facing power state is
    stuck at SHIP_PENDING
  * LED1 left in MANUAL, so no charging / low-battery / fault indication
  * the STM's last boot was an independent-watchdog reset (it hung)

The checks that need a person — cable out and in, host suspend, a 7 s button
hold with the cable in, charger into a powered-off unit, auto-off after dim —
are left to a manual checklist.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

from crosspad_hil import verbs
from crosspad_hil.scenarios import register
from crosspad_hil.scenarios.base import Context, Progress, Report

PROTO_ADDR = 0x0004          # REG_IDENT_PROTO_HI/LO, FW_MAJOR, FW_MINOR
POWER_DEBUG_ADDR = 0x107C    # REG_DEBUG +0x7C: five little-endian u32 words
POWER_DEBUG_LEN = 20
MIN_PROTO = 0x0010

# BQ25895 bits the profile pins and a chip reset flips.
R00_EN_ILIM = 0x40
R02_CONV_RATE = 0x40
R02_AUTO_DPDM = 0x01
R03_OTG_CONFIG = 0x20
R07_WDT_SHIFT = 4
WDT_80S = 2

# The chip watchdog is 80 s from the profile going in at STM boot; past this
# much STM uptime it has had its chance to expire.
WATCHDOG_JUDGE_UPTIME_MS = 90_000

SLEEP_STATES = {0: "AWAKE", 1: "LIGHT", 2: "USB_SUSPENDED", 3: "DEEP", 4: "CHARGE_ANIMATION"}
PWR_STATES = {0: "AWAKE", 1: "SLEEP_PENDING", 2: "SHIP_PENDING", 3: "WOKE", 4: "LIGHT", 5: "CHARGE"}
ICO_STATES = {0: "IDLE", 1: "BASELINE", 2: "RAMP", 3: "LOCKED"}
GATES = {0: "none", 1: "ESP", 2: "STM", 3: "both"}
# RCC_CSR[31:24] on the STM32G0: bit 5 of that byte is IWDGRSTF.
RESET_IWDG = 0x20


@dataclass
class Params:
    settle_s: float = field(
        default=100.0,
        metadata={"help": "least seconds to watch; the run also waits until the STM has "
                          "been up 90 s, so the 80 s chip watchdog had its chance. "
                          "0 = one sample, profile only"},
    )
    interval_s: float = field(default=5.0, metadata={"help": "seconds between samples"})
    max_wait_s: float = field(
        default=240.0, metadata={"help": "give up waiting for STM uptime past this"}
    )


def _u32(raw: bytes, word: int) -> int:
    return int.from_bytes(raw[word * 4:word * 4 + 4], "little")


def decode(raw: bytes) -> dict[str, Any]:
    """The five power debug words as named fields."""
    if len(raw) != POWER_DEBUG_LEN:
        raise ValueError(f"power debug block is {POWER_DEBUG_LEN} bytes, got {len(raw)}")
    w0, w1, w2, w3, w4 = (_u32(raw, i) for i in range(5))
    return {
        "reg00": w0 & 0xFF, "reg02": (w0 >> 8) & 0xFF,
        "reg03": (w0 >> 16) & 0xFF, "reg07": (w0 >> 24) & 0xFF,
        "cfg_reapplied": w1 & 0xFF,
        "ico_state": ICO_STATES.get((w1 >> 8) & 0xFF, str((w1 >> 8) & 0xFF)),
        "gate": GATES.get((w1 >> 16) & 0xFF, str((w1 >> 16) & 0xFF)),
        "no_battery": bool((w1 >> 24) & 0xFF),
        "ico_target_ma": w2 & 0xFFFF,
        "power_good": bool((w2 >> 16) & 0xFF),
        "chrg_stat": (w2 >> 24) & 0xFF,
        "sleep_state": SLEEP_STATES.get(w3 & 0xFF, str(w3 & 0xFF)),
        "led1_mode": "MANUAL" if (w3 >> 8) & 0xFF else "AUTO",
        "pwr_state": PWR_STATES.get((w3 >> 16) & 0xFF, str((w3 >> 16) & 0xFF)),
        "reset_flags": (w3 >> 24) & 0xFF,   # RCC_CSR[31:24] sampled at STM boot
        "uptime_ms": w4,
    }


def profile_problems(s: dict[str, Any]) -> list[str]:
    """What is wrong with the charger profile in one sample."""
    out: list[str] = []
    if s["reg00"] & R00_EN_ILIM:
        out.append("EN_ILIM=1: the ILIM pin limits input current (chip in default mode)")
    if not s["reg02"] & R02_CONV_RATE:
        out.append("CONV_RATE=0: continuous ADC off (chip in default mode)")
    if s["reg02"] & R02_AUTO_DPDM:
        out.append("AUTO_DPDM=1: BC1.2 detection on the floating D+/D- sets IINLIM on plug-in")
    if s["reg03"] & R03_OTG_CONFIG:
        out.append("OTG_CONFIG=1: boost back-feeds the USB ports once the cable is out")
    if (s["reg07"] >> R07_WDT_SHIFT) & 0x3 != WDT_80S:
        out.append(f"watchdog code {(s['reg07'] >> R07_WDT_SHIFT) & 0x3}, profile sets 2 (80 s)")
    return out


def evaluate(samples: list[dict[str, Any]]) -> list[str]:
    """Every finding over the run, each named once with when it first showed."""
    findings: list[str] = []
    seen: set[str] = set()

    def note(problem: str, where: str = "") -> None:
        if problem not in seen:
            seen.add(problem)
            findings.append(f"{problem} — first {where}" if where else problem)

    for s in samples:
        at = f"at STM uptime {s['uptime_ms'] / 1000:.0f} s"
        for p in profile_problems(s):
            note(p, at)
        if s["sleep_state"] == "USB_SUSPENDED":
            note("sleep state USB_SUSPENDED while the board answers (stale)", at)
        if s["pwr_state"] == "SHIP_PENDING":
            note("power state SHIP_PENDING while the board answers (stuck)", at)
        if s["reset_flags"] & RESET_IWDG:
            note("the STM's last boot was an IWDG reset (a hang before this run)", at)
        if s["led1_mode"] != "AUTO":
            note("LED1 in MANUAL: no charging / low-battery / fault indication", at)

    for prev, cur in zip(samples, samples[1:], strict=False):
        if cur["uptime_ms"] < prev["uptime_ms"]:
            note(f"STM reset during the run (uptime {prev['uptime_ms'] / 1000:.0f} s → "
                 f"{cur['uptime_ms'] / 1000:.0f} s)")
        elif cur["cfg_reapplied"] != prev["cfg_reapplied"]:
            note(f"charger profile re-applied during the run (count {prev['cfg_reapplied']} → "
                 f"{cur['cfg_reapplied']}): the chip reset itself, its watchdog is not fed")
    return findings


def _line(s: dict[str, Any]) -> str:
    return (f"  {s['uptime_ms'] / 1000:6.0f} s  REG00 {s['reg00']:02X} 02 {s['reg02']:02X} "
            f"03 {s['reg03']:02X} 07 {s['reg07']:02X}  re-applied {s['cfg_reapplied']}  "
            f"ICO {s['ico_state']} {s['ico_target_ma']} mA  gate {s['gate']}  "
            f"PG {int(s['power_good'])} CHRG {s['chrg_stat']}  "
            f"{s['sleep_state']}/{s['pwr_state']}  LED1 {s['led1_mode']}")


def _env(summary: str, data: dict[str, Any]) -> Report:
    return Report(passed=False, summary=summary, data=data, artifacts=[], exit_code=2)


class PowerStateScenario:
    name = "power_state"
    Params = Params
    description = ("STM power path, unattended: BQ25895 profile past its 80 s watchdog, "
                   "profile re-applies, STM resets, stuck sleep/ship state, LED1 mode")

    def run(self, ctx: Context, params: Params, progress: Progress) -> Report:
        data: dict[str, Any] = {"samples": []}
        ctx.partial = data
        try:
            link = ctx.open_cdc()
            try:
                ident = bytes(verbs.stm_reg(link, PROTO_ADDR, 4)["bytes"])
                proto = (ident[0] << 8) | ident[1]
                data["stm"] = {"proto": f"0x{proto:04X}", "fw": f"{ident[2]}.{ident[3]}"}
                if proto < MIN_PROTO:
                    return _env(f"STM proto 0x{proto:04X} predates the power debug words "
                                f"(needs 0x{MIN_PROTO:04X}, firmware 1.9 with the power fixes)",
                                data)

                def sample() -> dict[str, Any]:
                    raw = bytes(verbs.stm_reg(link, POWER_DEBUG_ADDR, POWER_DEBUG_LEN)["bytes"])
                    s = decode(raw)
                    data["samples"].append(s)
                    ctx.log(_line(s))
                    return s

                first = sample()
                if first["uptime_ms"] == 0:
                    return _env("the STM answers zeros for the power debug words — its "
                                "firmware lacks them (fix/power-fsm 2c384ff or later)", data)

                start = time.monotonic()
                last = first
                while params.settle_s > 0:
                    waited = time.monotonic() - start
                    done_settle = waited >= params.settle_s
                    done_uptime = last["uptime_ms"] >= WATCHDOG_JUDGE_UPTIME_MS
                    if done_settle and done_uptime:
                        break
                    if waited >= params.max_wait_s:
                        data["window_short"] = True
                        break
                    if ctx.cancelled.wait(params.interval_s):
                        break
                    last = sample()
                    progress(int(waited), int(max(params.settle_s, 1)),
                             f"STM up {last['uptime_ms'] / 1000:.0f} s")
            finally:
                link.close()

            samples = data["samples"]
            findings = evaluate(samples)
            data["findings"] = findings
            judged = samples[-1]["uptime_ms"] >= WATCHDOG_JUDGE_UPTIME_MS
            data["watchdog_judged"] = judged

            lines = [_line(samples[0])]
            if len(samples) > 1:
                lines.append(_line(samples[-1]))
            if findings:
                lines += ["", "FAIL:"] + [f"  - {f}" for f in findings]
                return Report(passed=False, summary="\n".join(lines), data=data,
                              artifacts=[], exit_code=1)
            lines.append("")
            lines.append("profile held past the chip watchdog" if judged else
                         "profile OK so far — STM uptime too short to judge the 80 s watchdog")
            return Report(passed=True, summary="\n".join(lines), data=data,
                          artifacts=[], exit_code=0)
        finally:
            ctx.close_all()


SCENARIO = register(PowerStateScenario())
