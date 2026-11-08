"""What a board is set to, read in one go, and what changed between two readings.

Every scenario is supposed to hand the board on the way it found it. One that
does not leaves its state to the next: ``ble_midi`` once ended every run with
BLE up, and four hours into a release gate the board had 63 kB of internal RAM
left and Sampler pads that no longer sounded. A reading before and after each
step, compared, says which step left what behind.

Two kinds of difference:

* **persistent** -- state the board keeps (NVS) or that outlives the step and
  changes what the next one measures: BLE running/role, the USB serial guard,
  WiFi enabled, the capture route, the input gains. A step that changes one of
  these and does not put it back is a finding;
* **informational** -- the running app and the current kit, which a step may
  legitimately leave elsewhere (``smoke`` resets the board into the launcher).

Memory is read too, for the record, but never diffed: it always moves.
Every field is best-effort -- a firmware without a verb leaves that field None,
and a None on either side is not a difference.
"""
from __future__ import annotations

import re
from typing import Any

from crosspad_hil import verbs
from crosspad_hil.cdc import CdcLink
from crosspad_hil.errors import HilError

PERSISTENT: tuple[str, ...] = (
    "ble_running", "ble_mode", "usb_guard", "wifi_enabled", "audio_route", "ingain",
)
INFORMATIONAL: tuple[str, ...] = ("app", "kit")
MEMORY: tuple[str, ...] = ("int_free", "psram_free")

_AUDIO_RE = re.compile(r"mic_src=(-?\d+)\s+adc=\[(-?\d+),(-?\d+)\]")
_GUARD_RE = re.compile(r"USB_GUARD:\s*(\d)")


def _int(value: Any) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def read(link: CdcLink) -> dict[str, Any]:
    """One reading; a field the board will not give is None."""
    state: dict[str, Any] = {k: None for k in (*PERSISTENT, *INFORMATIONAL, *MEMORY)}

    def field(fn: Any) -> None:
        try:
            fn()
        except HilError:
            pass

    def ble() -> None:
        st = verbs.ble_status(link)
        if _int(st.get("supported")) == 0:
            return
        running = _int(st.get("running"))
        state["ble_running"] = None if running is None else running == 1
        state["ble_mode"] = st.get("mode")

    def guard() -> None:
        m = _GUARD_RE.search(link.transact("USB_GUARD", "USB_GUARD:").line or "")
        if m:
            state["usb_guard"] = m.group(1) == "1"

    def wifi() -> None:
        enabled = _int(verbs.wifi_status(link).get("enabled"))
        state["wifi_enabled"] = None if enabled is None else enabled == 1

    def route() -> None:
        m = _AUDIO_RE.search(link.transact("AUDIO_STATUS", "AUDIO:").line or "")
        if m:
            state["audio_route"] = [int(m.group(1)), int(m.group(2)), int(m.group(3))]

    def gain() -> None:
        state["ingain"] = verbs.ingain(link)

    def app() -> None:
        # "launcher" rather than None: None is "could not read" to diff()
        state["app"] = verbs.app_list(link).get("running") or "launcher"

    def kit() -> None:
        state["kit"] = verbs.kit_status(link).get("current")

    def mem() -> None:
        m = verbs.mem(link)
        state["int_free"], state["psram_free"] = m.get("int_free"), m.get("psram_free")

    for fn in (ble, guard, wifi, route, gain, app, kit, mem):
        field(fn)
    return state


def diff(before: dict[str, Any] | None, after: dict[str, Any] | None) -> list[dict[str, Any]]:
    """``[{key, before, after, persistent}]`` for every field that moved; memory
    and fields either reading could not give are left out."""
    if not before or not after:
        return []
    out = []
    for key in (*PERSISTENT, *INFORMATIONAL):
        a, b = before.get(key), after.get(key)
        if a is None or b is None or a == b:
            continue
        out.append({"key": key, "before": a, "after": b, "persistent": key in PERSISTENT})
    return out


def describe(changes: list[dict[str, Any]]) -> str:
    """``"ble_running False -> True, usb_guard True -> False"``."""
    return ", ".join(f"{c['key']} {c['before']} -> {c['after']}" for c in changes)
