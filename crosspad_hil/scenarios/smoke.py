"""Boot smoke test — port of platform-idf/tools/hil_smoke.py.

Resets the device through the STM32 VCP bridge (esptool-style DTR/RTS pulse —
the bridge turns it into EN/IO0), captures the boot log and asserts:

  * every required marker appears (knowledge/markers.yaml `required`),
  * no E-level line outside the allow-list (`error_allow`),
  * no Guru Meditation / abort / assert / boot loop (`fatal_patterns`).

Exit code 0 = PASS, 1 = FAIL, 2 = no boot banner at all / flash failed.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from crosspad_hil.console import BootResult
from crosspad_hil.errors import HilError
from crosspad_hil.knowledge import load as load_knowledge
from crosspad_hil.ota import flash
from crosspad_hil.scenarios import register
from crosspad_hil.scenarios.base import Artifact, Context, Progress, Report

# from hil_smoke.capture_boot: after the last required marker keep reading for
# 3 s so a late error is still caught.
SETTLE_S: float = 3.0
# from hil_smoke.main: time.sleep(3)  # let the device reboot and re-enumerate
POST_FLASH_S: float = 3.0


@dataclass
class Params:
    flash: str | None = field(
        default=None,
        metadata={"help": "OTA-flash this firmware image (e.g. build/CrossPad.bin) first"},
    )
    timeout: int = field(
        default=25,
        metadata={"help": "boot capture window in seconds"},
    )


def evaluate(boot: BootResult, lines: list[str], optional: list[str]) -> dict[str, Any]:
    """hil_smoke.evaluate on top of Console.wait_boot: same keys, same verdict rule."""
    text = "\n".join(lines)
    data: dict[str, Any] = {
        "pass": False,
        "missing": list(boot.missing),
        "optional_missing": [m for m in optional if m not in text],
        "errors": [e["line"] for e in boot.errors],
        "fatal": [f["line"] for f in boot.fatal],
        "bootloops": int(boot.bootloops),
    }
    data["pass"] = (
        not data["missing"]
        and not data["errors"]
        and not data["fatal"]
        and data["bootloops"] == 0
    )
    return data


def _summary(data: dict[str, Any]) -> str:
    """The lines hil_smoke.main printed in non-JSON mode."""
    lines: list[str] = []
    for m in data["missing"]:
        lines.append(f"MISSING MARKER: {m}")
    for m in data["optional_missing"]:
        lines.append(f"note: optional marker absent: {m}")
    for e in data["errors"]:
        lines.append(f"ERROR LINE: {e}")
    for f in data["fatal"]:
        lines.append(f"FATAL: {f}")
    if data["bootloops"]:
        lines.append(f"BOOT LOOP: device reset {data['bootloops']} extra time(s)")
    lines.append("HIL SMOKE: PASS" if data["pass"] else "HIL SMOKE: FAIL")
    return "\n".join(lines)


class SmokeScenario:
    name = "smoke"
    Params = Params
    description = "reset, capture the boot log, check markers / E-lines / fatals / boot loops"

    def run(self, ctx: Context, params: Params, progress: Progress) -> Report:
        try:
            console_log = ctx.workdir / "console.log"
            artifacts = [Artifact(path=str(console_log), mime="text/plain", role="console")]
            flash_result: dict[str, Any] | None = None

            if params.flash is not None:
                progress(0, 3, f"flashing {params.flash}")
                try:
                    flash_result = flash(
                        ctx.device, Path(params.flash), transport="ota", wait_boot=False
                    )
                except HilError as e:
                    return Report(
                        passed=False,
                        summary=f"FLASH FAILED: {e.message}",
                        data={
                            "pass": False, "missing": [], "optional_missing": [],
                            "errors": [], "fatal": [], "bootloops": 0,
                            "flash_error": e.to_dict(),
                        },
                        artifacts=[],
                        exit_code=2,
                    )
                ctx.cancelled.wait(POST_FLASH_S)
                ctx.check_cancelled()

            markers = load_knowledge("markers")
            boot_markers: list[str] = list(markers["boot_markers"])
            optional: list[str] = list(markers.get("optional", []))

            progress(1, 3, "resetting device and capturing boot")
            console = ctx.open_console(reset=True)
            try:
                boot = console.wait_boot(timeout_s=float(params.timeout), settle_s=SETTLE_S)
                ctx.check_cancelled()
                captured = console.read(since_seq=0, limit=50_000)
                port = console.snapshot()["port"]
            finally:
                console.close()

            progress(2, 3, "evaluating")
            lines = [line for _seq, line in captured.lines]
            # hil_smoke.capture_boot: no BOOT_MARKER at all means the wrong port or
            # a dead board, which is an environment failure, not a failed boot.
            saw_banner = any(any(m in line for m in boot_markers) for line in lines)
            if not saw_banner:
                data = {
                    "pass": False, "missing": list(boot.missing), "optional_missing": optional,
                    "errors": [], "fatal": [], "bootloops": 0,
                    "seconds": boot.seconds, "port": port,
                }
                if flash_result is not None:
                    data["flash"] = flash_result.get("flash")
                return Report(
                    passed=False,
                    summary="No boot banner on the console port — is the device connected, "
                            "and is the STM32 bridge VCP (0x0483:0x5740) the console?",
                    data=data,
                    artifacts=artifacts,
                    exit_code=2,
                )

            data = evaluate(boot, lines, optional)
            data["seconds"] = boot.seconds
            data["port"] = port
            if flash_result is not None:
                data["flash"] = flash_result.get("flash")
            progress(3, 3, "done")
            return Report(
                passed=data["pass"],
                summary=_summary(data),
                data=data,
                artifacts=artifacts,
                exit_code=0 if data["pass"] else 1,
            )


        finally:
            ctx.close_all()
SCENARIO = register(SmokeScenario())
