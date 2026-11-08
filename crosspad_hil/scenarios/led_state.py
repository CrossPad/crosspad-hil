"""Dump what the firmware thinks the pad LEDs should be showing.

Port of platform-idf/tools/hil_led_state.py. Prints the LED controller's model
— per-pad colour, brightness, animation flag — plus the last power-state
notification the STM sent. Run it while the pads look wrong: the answer to
"why are they dark?" is one of

  * colours are 000000        -> whatever painted them last painted black
  * bri=0 (or cfgbri=0)       -> brightness, not colour
  * anim=1                    -> an animation owns the strip and blocks repaints
  * pwr/pwrN did not change   -> the wake notification never reached the ESP
  * everything looks right    -> the frame is correct and the fault is below the
                                 driver: the RMT transmission or the strip rail

This is a diagnostic dump, not a check: it always passes.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from crosspad_hil import verbs
from crosspad_hil.scenarios import register
from crosspad_hil.scenarios.base import Context, Progress, Report

# from hil_led_state.main: time.sleep(2) between polls
WATCH_PERIOD_S: float = 2.0


@dataclass
class Params:
    watch: bool = field(
        default=False, metadata={"help": "poll every 2 s until cancelled (Ctrl-C)"}
    )


def render_grid(state: dict[str, Any]) -> str:
    """Text block of hil_led_state.render(): header, 4x4 grid, black-model note."""
    colors: list[str] = list(state.get("colors", []))
    out: list[str] = []
    out.append(
        f"  brightness {state.get('brightness')}   settings {state.get('cfgbri')}"
        f"   animating {int(bool(state.get('anim', 0)))}"
        f"   coalesced {int(bool(state.get('coalesce', 0)))}"
    )
    out.append(
        f"  last power state 0x{int(state.get('pwr', 0xFF)):02X}"
        f"   notifications {state.get('pwr_count')}"
        f"   (0x00 AWAKE · 0x03 WOKE · 0x04 LIGHT)"
    )
    txfail = int(state.get("txfail", 0))
    if txfail != 0:
        out.append(
            f"  {txfail} frame(s) never reached the strip — the model is "
            f"right and the wire is not"
        )
    out.append("")
    # 4x4 as it sits under your hands: pad 12..15 on the top row.
    for row in range(3, -1, -1):
        cells = []
        for col in range(4):
            idx = row * 4 + col
            c = colors[idx] if idx < len(colors) else "??????"
            cells.append(f"{idx:2d}:{c}")
        out.append("   " + "  ".join(cells))
    out.append("")
    if colors and all(c == "000000" for c in colors):
        out.append("  every pad is black in the model — nothing repainted them")
    return "\n".join(out)


class LedStateScenario:
    name = "led_state"
    Params = Params
    description = "dump the LED controller model (colours, brightness, anim, power state)"

    def run(self, ctx: Context, params: Params, progress: Progress) -> Report:
        try:
            link = ctx.open_cdc()
            samples = 0
            state: dict[str, Any] = {}
            grid = ""
            try:
                while True:
                    state = verbs.led_state(link)
                    samples += 1
                    grid = render_grid(state)
                    progress(samples, None, grid)
                    ctx.log(grid)
                    if not params.watch:
                        break
                    # A cancelled watch is the normal way to stop it, not an error.
                    if ctx.cancelled.wait(WATCH_PERIOD_S):
                        break
                    ctx.log("-" * 60)
            finally:
                link.close()

            colors = list(state.get("colors", []))
            data = {
                "leds": state,
                "all_black": bool(colors) and all(c == "000000" for c in colors),
                "samples": samples,
            }
            return Report(passed=True, summary=grid, data=data, artifacts=[], exit_code=0)


        finally:
            ctx.close_all()
SCENARIO = register(LedStateScenario())
