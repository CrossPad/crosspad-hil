"""Install a matching package, switch OTA slots and back, verify the same image.

Exercises platform-idf's firmware library / A/B switch (`main/fw_slots.cpp`,
`main/fw_library.cpp`; docs/subsystems/firmware-library.md) end to end: a
package for the running build must already be on the SD card (this scenario
never builds or pushes one — see the "no package" failure message below for
the two commands that do). It never opens the STM bridge console: FW_SWITCH
causes its own reboot, and the CDC port re-enumerates on the same VID:PID —
opening the console would only add a second, unwanted reset.

Flow: FW_SLOTS (must be confirmed, rollback-capable) -> APP_SHA -> FW_LIST ->
FW_INSTALL a matching package -> poll to phase=done -> FW_SWITCH -> wait for
reboot -> FW_SLOTS (other slot, confirmed within ~10-16s) -> APP_SHA (same as
before) -> FW_SWITCH back -> wait for reboot -> FW_SLOTS (original slot,
confirmed again).

Exit codes: 0 pass, 1 firmware fault (install/switch refused or failed, image
did not confirm, APP_SHA differs), 2 bench problem (no confirmed/rollback-
capable starting point, no matching package on the card).
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

from crosspad_hil import verbs
from crosspad_hil.cdc import CdcLink
from crosspad_hil.devices import discover, select
from crosspad_hil.errors import NO_DEVICE, NOT_SUPPORTED, TIMEOUT, HilError
from crosspad_hil.scenarios.base import Context, Progress, Report, register

POLL_S = 0.5                   # FW_SLOTS confirm-state poll cadence
INSTALL_POLL_S = 0.5           # FW_INSTALL bare-poll cadence
# "give the board ~3s after re-enumeration before the first verb" (brief) --
# FW_SLOTS does not time out during the boot kit scan the way KIT_LIST does,
# but wait_ready() only proves the app registry answered, not that every
# other subsystem has finished settling.
REBOOT_SETTLE_S = 3.0
# CdcLink.open()'s own default wait (ENUM_WAIT_S, 15s) is sized for the CDC
# node coming back, not for a full A/B switch reboot to finish behind it: SD
# mount, the boot kit scan and (on a mismatch) a 4.2-4.4s assets resync all
# run before app_main() is far enough along for APP_LIST to answer, and on
# this bench's 1-bit SD card that measurably exceeds 15s (seen on hardware:
# the port was gone 20+ s after a switch, then came back healthy and already
# confirmed). Retry ctx.open_cdc() itself across a longer budget rather than
# reporting a bench problem for a reboot that was still simply in progress.
REOPEN_TIMEOUT_S = 90.0
REOPEN_RETRY_S = 2.0


@dataclass
class Params:
    package: str | None = field(
        default=None,
        metadata={"help": "install this package by name instead of auto-selecting the first "
                          "FW_LIST row whose ver= matches the running build and verdict=ok"})
    install_timeout: float = field(
        default=120.0, metadata={"help": "seconds to wait for FW_INSTALL to reach phase=done"})
    confirm_timeout: float = field(
        default=30.0,
        metadata={"help": "seconds to wait for a switched-to slot to reach state=confirmed"})
    list_timeout: float = field(
        default=15.0, metadata={"help": "seconds to wait for FW_LIST's SD-card scan"})


def _env(summary: str, data: dict[str, Any]) -> Report:
    return Report(passed=False, summary=summary, data=data, artifacts=[], exit_code=2)


def _fail(summary: str, data: dict[str, Any]) -> Report:
    return Report(passed=False, summary=summary, data=data, artifacts=[], exit_code=1)


def _running_slot(slots: list[dict[str, Any]]) -> dict[str, Any]:
    for s in slots:
        if s["running"]:
            return s
    raise HilError(NOT_SUPPORTED, "FW_SLOTS: no slot reports running=1", slots=slots)


class FwSwitchScenario:
    name = "fw_switch"
    Params = Params
    description = "install a matching package, switch OTA slots and back, verify same image"

    def run(self, ctx: Context, params: Params, progress: Progress) -> Report:
        p = params
        data: dict[str, Any] = {
            "package": None, "label_before": None, "label_after": None, "phases_s": {},
        }
        ctx.partial = data
        t0 = time.monotonic()

        def mark(phase: str) -> None:
            data["phases_s"][phase] = round(time.monotonic() - t0, 3)

        try:
            ctx.check_cancelled()
            link = ctx.open_cdc()

            progress(0, 9, "FW_SLOTS")
            slots0 = verbs.fw_slots(link)
            running0 = _running_slot(slots0["slots"])
            data["label_before"] = running0["slot"]
            data["rollback_supported"] = slots0["rollback"]
            if running0["state"] != "confirmed":
                return _env(
                    f"fw_switch: running slot {running0['slot']} is {running0['state']}, not "
                    "confirmed -- a trial is in progress; wait it out or FW_CONFIRM it first",
                    data)
            if not slots0["rollback"]:
                return _env(
                    "fw_switch: the flashed bootloader reports no rollback support "
                    "(tools/bench.py recover reflashes one that does)", data)

            progress(1, 9, "APP_SHA")
            sha_before = verbs.app_sha(link)["sha256"]
            data["sha_before"] = sha_before
            running_ver = running0["ver"]
            mark("preflight")

            progress(2, 9, "FW_LIST")
            listing = verbs.fw_list(link, timeout_s=p.list_timeout)
            if p.package:
                match = next((pk for pk in listing["packages"] if pk["name"] == p.package), None)
                if match is None:
                    have = sorted(pk["name"] for pk in listing["packages"])
                    return _env(f"fw_switch: package {p.package!r} not on the card "
                                f"(have: {', '.join(have) or 'none'})", data)
            else:
                match = next((pk for pk in listing["packages"]
                             if pk["ver"] == running_ver and pk["verdict"] == "ok"), None)
                if match is None:
                    return _env(
                        "fw_switch: no package of the running build on the card -- run "
                        "`python3 tools/make_fw_package.py build_v2 <name>` and push it with "
                        "`tools/fs_transfer.py push`", data)
            data["package"] = match["name"]
            data["package_ver"] = match["ver"]
            ctx.log(f"[fw_switch] installing {match['name']!r} (ver={match['ver']}, "
                    f"running ver={running_ver})")
            mark("fw_list")

            progress(3, 9, f"FW_INSTALL {match['name']}")
            install = verbs.fw_install(link, match["name"])
            data["install_warn_stm_proto_older"] = install["warn_stm_proto_older"]
            deadline = time.monotonic() + p.install_timeout
            status = verbs.fw_install_status(link)
            while status["phase"] not in ("done", "failed"):
                if time.monotonic() >= deadline:
                    data["install_status"] = status
                    return _fail(f"fw_switch: FW_INSTALL of {match['name']!r} still "
                                 f"{status['phase']} after {p.install_timeout:.0f}s", data)
                ctx.cancelled.wait(INSTALL_POLL_S)
                ctx.check_cancelled()
                status = verbs.fw_install_status(link)
            data["install_status"] = status
            if status["phase"] == "failed":
                return _fail(f"fw_switch: FW_INSTALL of {match['name']!r} failed: "
                             f"{status.get('error') or '?'}", data)
            mark("install")

            progress(4, 9, "FW_SWITCH")
            verbs.fw_switch(link)
            link = self._reopen(ctx, link)
            mark("switch1_reboot")

            progress(5, 9, "FW_SLOTS after switch")
            slots1 = verbs.fw_slots(link)
            running1 = _running_slot(slots1["slots"])
            data["label_after"] = running1["slot"]
            if running1["slot"] == running0["slot"]:
                return _fail(f"fw_switch: running slot is still {running1['slot']} after "
                             "FW_SWITCH -- the boot partition did not move", data)
            confirmed1 = self._wait_confirmed(ctx, link, running1["slot"], p.confirm_timeout, data)
            if confirmed1 is None:
                return _fail(f"fw_switch: slot {data['label_after']} never reached "
                             f"state=confirmed within {p.confirm_timeout:.0f}s", data)
            mark("confirm1")

            progress(6, 9, "APP_SHA after switch")
            sha_after = verbs.app_sha(link)["sha256"]
            data["sha_after"] = sha_after
            if sha_after != sha_before:
                return _fail("fw_switch: APP_SHA changed across the switch "
                             f"({sha_before} -> {sha_after}) -- not the same image", data)
            mark("verify")

            progress(7, 9, "FW_SWITCH back")
            verbs.fw_switch(link)
            link = self._reopen(ctx, link)
            mark("switch2_reboot")

            progress(8, 9, "FW_SLOTS back")
            confirmed2 = self._wait_confirmed(ctx, link, running0["slot"], p.confirm_timeout, data)
            if confirmed2 is None:
                return _fail(f"fw_switch: slot {running0['slot']} never returned to "
                             f"state=confirmed within {p.confirm_timeout:.0f}s of switching back",
                             data)
            mark("confirm2")

            progress(9, 9, "done")
            summary = (f"fw_switch: PASS -- {running0['slot']}->{data['label_after']}->"
                       f"{running0['slot']}, package {match['name']!r}, same image both sides")
            ctx.log(f"[fw_switch] {summary}")
            return Report(passed=True, summary=summary, data=data, artifacts=[], exit_code=0)
        finally:
            ctx.close_all()

    @staticmethod
    def _reopen(ctx: Context, link: CdcLink) -> CdcLink:
        """Close `link` (about to vanish under FW_SWITCH's own reboot) and hand
        back a fresh one once the board answers again.

        Never opens the STM bridge console (`ctx.open_console()`): on this
        hardware that resets the ESP by itself, and FW_SWITCH already causes
        the only reboot this scenario wants. `ctx.open_cdc()` + `wait_ready()`
        is the same reboot-then-reopen shape `fs_transfer`/`kits.push_kit`
        already use -- the port resolver inside CdcLink follows the board if
        it re-enumerates onto a different node.

        `ctx.open_cdc()` alone only waits `CdcLink`'s own default 15s for the
        node to reappear (ENUM_WAIT_S), sized for a light reset -- a full A/B
        switch reboot runs the boot kit scan and, on an assets mismatch, a
        4.2-4.4s partition resync before app_main() gets anywhere near ready,
        and measured on this bench's 1-bit SD card that can still be going
        past 15s. Retry the open itself across REOPEN_TIMEOUT_S rather than
        reporting a bench problem for a reboot that was still in progress.
        """
        link.close()
        deadline = time.monotonic() + REOPEN_TIMEOUT_S
        while True:
            try:
                new_link = ctx.open_cdc()
                break
            except HilError as exc:
                if exc.code not in (NO_DEVICE, TIMEOUT):
                    raise
                if time.monotonic() >= deadline:
                    raise HilError(
                        exc.code,
                        f"fw_switch: board did not come back within "
                        f"{REOPEN_TIMEOUT_S:.0f}s of FW_SWITCH ({exc.message})",
                        hint=exc.hint) from exc
                ctx.cancelled.wait(REOPEN_RETRY_S)
                ctx.check_cancelled()
                # Context.open_cdc() trusts ctx.device.ports.cdc without
                # re-discovering it, and CdcLink.open()'s own port_resolver
                # can leave that stale at None for good (a transitional scan
                # mid-reboot that saw the console/MIDI ports but not yet CDC)
                # -- every later ctx.open_cdc() in this loop would then fail
                # on that same snapshot instead of the device it describes.
                # Refresh it before every retry rather than trust it once.
                try:
                    ctx.device = select(discover(), ctx.device.id)
                except HilError:
                    pass
        new_link.wait_ready()
        ctx.cancelled.wait(REBOOT_SETTLE_S)
        ctx.check_cancelled()
        return new_link

    @staticmethod
    def _wait_confirmed(ctx: Context, link: CdcLink, expected_label: str, timeout_s: float,
                        data: dict[str, Any]) -> dict[str, Any] | None:
        """Poll FW_SLOTS until `expected_label` is the running slot and
        state=confirmed, or `timeout_s` runs out (the confirm timer fires
        ~10-16s after boot per docs/subsystems/firmware-library.md)."""
        deadline = time.monotonic() + timeout_s
        while True:
            slots = verbs.fw_slots(link)
            running = _running_slot(slots["slots"])
            data["rollback_supported"] = slots["rollback"]
            if running["slot"] == expected_label and running["state"] == "confirmed":
                return running
            if time.monotonic() >= deadline:
                data["last_slots"] = slots["slots"]
                return None
            ctx.cancelled.wait(POLL_S)
            ctx.check_cancelled()


SCENARIO = register(FwSwitchScenario())
