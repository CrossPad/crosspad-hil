"""One-call device state with UI refs.

A snapshot is what `ui focus REF` and the MCP `crosspad_ui` tool navigate from:
`ENC_GROUP` gives every widget the encoder can reach, `ENC_FOCUS` where it is
now, and `ref_to_delta()` turns a ref from the last snapshot into the
`ENC_ROTATE` count that reaches it. Everything else is the state verbs, one
section each, so a failed verb costs one section and not the whole picture.
"""
from __future__ import annotations

import itertools
import time
import uuid
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from typing import Any

from . import verbs
from .cdc import CdcLink
from .console import Console
from .devices import Device
from .errors import BAD_ARGS, TIMEOUT, HilError

DEFAULT_INCLUDE = ("apps", "ui", "kit", "leds", "pads", "mem", "ble", "console")
_MEM_COMPARE_KEYS = ("int_free", "psram_free", "int_largest")


@dataclass
class Snapshot:
    snapshot_id: str
    device: str
    usb_mode: str
    apps: dict | None
    ui: dict | None
    kit: dict | None
    leds: dict | None
    pads: dict | None
    mem: dict | None
    ble: dict | None
    console: dict | None
    ts: float
    changed: list[str]

    def to_dict(self) -> dict:
        return asdict(self)


def _none_if_dash(value: Any) -> str | None:
    if value is None or value == "-" or value == "":
        return None
    return str(value)


def _section_ui(link: CdcLink) -> dict[str, Any]:
    focus = verbs.enc_focus(link)
    group = verbs.enc_group(link)["group"]
    ui = verbs.ui_state(link)
    # ENCFOCUS idx= is lv_obj_get_index (child index in the parent), not the
    # group position. Match the object pointer against ENC_GROUP to get the
    # index the refs are minted from.
    ref: str | None = None
    index = -1
    for item in group:
        if item["ptr"] == focus["ptr"]:
            ref = item["ref"]
            index = int(item["index"])
            break
    return {
        "focus": {"ref": ref, "index": index, "label": focus["label"]},
        "group": group,
        "drawer": bool(int(ui.get("drawer", 0))),
        "theme": int(ui.get("theme", 0)),
        "app": _none_if_dash(ui.get("app")),
    }


def _section_pads(link: CdcLink) -> dict[str, Any]:
    out = verbs.pad_stats(link)
    out["notes"] = verbs.pad_notes(link)["notes"]
    return out


def _section_console(console: Console | None) -> dict[str, Any] | None:
    if console is None:
        return None
    d = console.snapshot()
    return {
        "handle": None,
        "fatals": list(d.get("fatals", [])),
        "reboots": int(d.get("reboots", 0)),
        "cdc_drops": int(d.get("cdc_drops", 0)),
        "since_seq": int(d.get("seq", 0)),
    }


def _mem_key(section: dict | None) -> Any:
    if not isinstance(section, dict) or "error" in section:
        return section
    return tuple(section.get(k) for k in _MEM_COMPARE_KEYS)


def _changed(current: dict[str, Any], previous: Snapshot | None,
             keys: Sequence[str]) -> list[str]:
    if previous is None:
        return []
    prev = previous.to_dict()
    changed: list[str] = []
    for key in keys:
        a, b = current.get(key), prev.get(key)
        if key == "mem":
            a, b = _mem_key(a), _mem_key(b)
        if a != b:
            changed.append(key)
    return changed


def take_snapshot(device: Device, link: CdcLink, *, console: Console | None = None,
                  include: Sequence[str] = DEFAULT_INCLUDE, previous: Snapshot | None = None,
                  counter: itertools.count | None = None) -> Snapshot:
    include = tuple(include)
    unknown = [k for k in include if k not in DEFAULT_INCLUDE]
    if unknown:
        raise HilError(BAD_ARGS, f"unknown snapshot section(s): {', '.join(unknown)}",
                       hint=f"include must be a subset of {list(DEFAULT_INCLUDE)}",
                       unknown=unknown)

    readers = {
        "apps": lambda: verbs.app_list(link),
        "ui": lambda: _section_ui(link),
        "kit": lambda: verbs.kit_status(link),
        "leds": lambda: verbs.led_state(link),
        "pads": lambda: _section_pads(link),
        "mem": lambda: verbs.mem(link),
        "ble": lambda: verbs.ble_status(link),
    }
    sections: dict[str, dict | None] = {k: None for k in DEFAULT_INCLUDE}
    first_device_section = True
    for key in include:
        if key == "console":
            sections["console"] = _section_console(console)
            continue
        try:
            sections[key] = readers[key]()
        except HilError as err:
            # A dead link shows up as a timeout on the first thing asked; every
            # later timeout is that one verb (or a UI lock) and stays local.
            if err.code == TIMEOUT and first_device_section:
                raise
            sections[key] = {"error": err.to_dict()}
        first_device_section = False

    usb_mode = device.usb_mode.value if hasattr(device.usb_mode, "value") else str(device.usb_mode)
    current = {**sections, "usb_mode": usb_mode}
    changed = _changed(current, previous, [*include, "usb_mode"])
    snapshot_id = f"snap_{next(counter)}" if counter is not None else f"snap_{uuid.uuid4().hex[:6]}"
    return Snapshot(
        snapshot_id=snapshot_id,
        device=device.id,
        usb_mode=usb_mode,
        apps=sections["apps"],
        ui=sections["ui"],
        kit=sections["kit"],
        leds=sections["leds"],
        pads=sections["pads"],
        mem=sections["mem"],
        ble=sections["ble"],
        console=sections["console"],
        ts=time.time(),
        changed=changed,
    )


def ref_to_delta(group: list[dict], focus_index: int, ref: str) -> int:
    for item in group:
        if item.get("ref") == ref:
            return int(item["index"]) - int(focus_index)
    known = [str(item.get("ref")) for item in group]
    raise HilError(BAD_ARGS, f"unknown ui ref {ref!r}",
                   hint="refs come from the latest snapshot's ui.group; take a new snapshot",
                   ref=ref, known=known)
