"""Typed CDC verbs: one function per firmware command in hil_control.cpp.

Every function takes a `CdcLink` first, validates its arguments locally
(HilError BAD_ARGS before a byte is written), sends the command and returns a
plain dict. Two hygiene rules from the old scripts are enforced here rather
than left to callers:

* `OK` is never the ack of *your* command (PAD_PRESS answers `OK` exactly like
  KIT_LOAD does — hil_kit_churn.py). `kit_load` therefore waits on KIT_STATUS
  and `app_start` polls APP_LIST; neither trusts the `OK`.
* A reply that cannot be parsed is reported as NOT_SUPPORTED with a hint —
  the usual cause is firmware that predates the verb.

Argument limits are copied from the sscanf guards in hil_control.cpp.
"""
from __future__ import annotations

import re
import time
from collections.abc import Callable
from typing import Any

from .cdc import CdcLink, Reply
from .errors import BAD_ARGS, NOT_SUPPORTED, TIMEOUT, HilError
from .parsers import parse_cdc_reply, parse_enc_group

_POLL_S = 0.25
_APP_NAME_MAX = 31          # hil_control.cpp: char name[32]; sscanf "%31s"
_BLE_ADDR_RE = re.compile(r"^[0-9A-Fa-f]{2}(:[0-9A-Fa-f]{2}){5}$")   # BLE_CONNECT %23s


# ── helpers ────────────────────────────────────────────────────────────

def _check_range(name: str, value: int, lo: int, hi: int) -> None:
    if not isinstance(value, int) or isinstance(value, bool) or value < lo or value > hi:
        raise HilError(BAD_ARGS, f"{name} must be an int in {lo}..{hi}, got {value!r}",
                       hint=f"{name}={lo}..{hi}", arg=name, value=value)


def _raise_err(verb: str, reply: Reply, code: str) -> None:
    parsed = reply.parsed or {}
    message = parsed.get("message") or reply.line
    raise HilError(code, f"{verb}: device replied ERR {message}".rstrip(),
                   hint="the device rejected the command; check the arguments "
                        "and what the state verbs report", reply=reply.line)


def _scanning(exc: HilError) -> bool:
    """True for the device's "ERR scanning": listKits() is still walking the card.

    Not a refusal a scenario can fix by changing its arguments — the kit vector
    is being grown in place, so no reader may touch it. The boot scan waits for
    the launcher before it starts, which is why this can arrive well after boot.
    """
    return "ERR scanning" in exc.message


def _kind(verb: str, reply: Reply, kind: str) -> dict[str, Any]:
    """The parsed reply if it is of `kind`, else NOT_SUPPORTED."""
    parsed = reply.parsed
    if parsed is not None and parsed.get("kind") == "err":
        _raise_err(verb, reply, NOT_SUPPORTED)
    if parsed is None or parsed.get("kind") != kind:
        raise HilError(NOT_SUPPORTED, f"{verb}: unparseable reply",
                       hint="firmware predates this verb?", line=reply.line)
    return parsed


def _ok(link: CdcLink, cmd: str, *, err_code: str = NOT_SUPPORTED,
        timeout_s: float = 2.0) -> dict[str, Any]:
    """Send an OK/ERR verb. expect=None so cdc.yaml makes the link accept ERR too."""
    verb = cmd.split(" ", 1)[0]
    reply = link.transact(cmd, None, timeout_s=timeout_s)
    parsed = reply.parsed or {}
    if parsed.get("kind") == "err":
        _raise_err(verb, reply, err_code)
    if parsed.get("kind") != "ok":
        raise HilError(NOT_SUPPORTED, f"{verb}: unparseable reply",
                       hint="firmware predates this verb?", line=reply.line)
    return {"ok": True}


def _query(link: CdcLink, cmd: str, prefix: str, kind: str,
           timeout_s: float | None = None) -> dict[str, Any]:
    verb = cmd.split(" ", 1)[0]
    if timeout_s is None:
        return _kind(verb, link.transact(cmd, prefix), kind)
    return _kind(verb, link.transact(cmd, prefix, timeout_s=timeout_s), kind)


def _without_kind(parsed: dict[str, Any]) -> dict[str, Any]:
    return {k: v for k, v in parsed.items() if k != "kind"}


# ── app_* ──────────────────────────────────────────────────────────────

def app_list(link: CdcLink) -> dict[str, Any]:
    p = _query(link, "APP_LIST", "APPS:", "apps")
    return {"apps": list(p.get("apps", [])), "running": p.get("running")}


def app_start(link: CdcLink, name: str, wait_s: float = 3.0) -> dict[str, Any]:
    if not isinstance(name, str) or not name or len(name) > _APP_NAME_MAX \
            or any(ch.isspace() for ch in name):
        raise HilError(BAD_ARGS, f"app name must be 1..{_APP_NAME_MAX} chars without "
                       f"whitespace, got {name!r}", hint="APP_LIST names the apps", name=name)
    _ok(link, f"APP_START {name}", err_code=BAD_ARGS)
    # The OK only means the launch was queued on the LVGL thread (hil_control.cpp
    # lv_async_call_locked). APP_LIST running= is the state that proves it.
    deadline = time.monotonic() + max(0.0, wait_s)
    last: str | None = None
    while True:
        last = app_list(link)["running"]
        if last is not None and last.lower() == name.lower():
            return {"running": last}
        if time.monotonic() >= deadline:
            break
        time.sleep(_POLL_S)
    raise HilError(TIMEOUT, f"APP_START {name}: not running after {wait_s:.1f}s "
                   f"(APP_LIST running={last or '-'})",
                   hint="another app may still be tearing down; APP_DESTROY first",
                   running=last, name=name)


def app_stop(link: CdcLink) -> dict[str, Any]:
    return _ok(link, "APP_STOP")


def app_destroy(link: CdcLink) -> dict[str, Any]:
    return _ok(link, "APP_DESTROY")


def app_self_close(link: CdcLink) -> dict[str, Any]:
    return _ok(link, "APP_SELF_CLOSE")


def app_versions(link: CdcLink) -> dict[str, Any]:
    # hil_control.cpp handle_app_versions: one "APPVER: <component> id=… commit=…
    # ref=… dirty=…" per component, then the "LIBVER: core=… gui=… stm_proto=…"
    # summary, then optionally "BOARD: fw=<rev> pcb=<n|->", then "APPVER: end
    # count=N". LIBVER and BOARD are collected, not filtered out by prefix.
    lines = link.transact_multi("APP_VERSIONS", ("APPVER:", "LIBVER:", "BOARD:"),
                                end="APPVER: end")
    components: list[dict[str, Any]] = []
    count: int | None = None
    libver: dict[str, str] = {}
    board: dict[str, str] = {}
    for line in lines:
        body = line.strip()
        if body.startswith("LIBVER:"):
            for field in body[len("LIBVER:"):].split():
                key, _, value = field.partition("=")
                if value:
                    libver[key] = value
            continue
        if body.startswith("BOARD:"):
            for field in body[len("BOARD:"):].split():
                key, _, value = field.partition("=")
                if value:
                    board[key] = value
            continue
        parsed = parse_cdc_reply(line)
        if not parsed or parsed.get("kind") != "appver":
            continue
        if parsed.get("end"):
            count = int(parsed.get("count", len(components)))
            continue
        components.append({
            "component": parsed.get("component"),
            "id": parsed.get("id"),
            "commit": parsed.get("commit"),
            "ref": parsed.get("ref"),
            "dirty": bool(parsed.get("dirty", False)),
        })
    if count is None:
        raise HilError(NOT_SUPPORTED, "APP_VERSIONS: no 'APPVER: end' line",
                       hint="firmware predates APP_VERSIONS", lines=lines)
    return {"components": components, "count": count, "libver": libver, "board": board}


def settings_reset(link: CdcLink) -> dict[str, Any]:
    """Erase the settings namespaces and reboot, so the next boot takes defaults.

    The only way to observe a default on a board that has already saved the key —
    which is every board anyone has ever configured. Destructive on purpose, and
    the device insists on the literal FORCE.
    """
    reply = link.transact("SETTINGS_RESET FORCE", expect="SETTINGS_RESET:")
    out: dict[str, Any] = {}
    for field in reply.line.split(":", 1)[1].split():
        key, _, value = field.partition("=")
        if value.isdigit():
            out[key] = int(value)
    return out


_SETTINGS_STATUS_RE = re.compile(
    r"^SETTINGS:\s+busy=(\d)\s+op=(\S)\s+ok=(-?\d+)\s+err=(.*?)\s+path=(\S+)\s*$")


def _settings_job(link: CdcLink, cmd: str) -> dict[str, Any]:
    """SETTINGS_EXPORT / SETTINGS_IMPORT: "OK <path>" = the job started, "ERR busy|path"."""
    verb = cmd.split(" ", 1)[0]
    reply = link.transact(cmd, "OK")
    if (reply.parsed or {}).get("kind") == "err":
        _raise_err(verb, reply, BAD_ARGS)
    path = reply.line[len("OK"):].strip()
    return {"started": True, "path": path or None}


def settings_export(link: CdcLink, path: str | None = None) -> dict[str, Any]:
    """Start writing the Settings -> Backup JSON (default /sdcard/crosspad-settings.json).
    The LVGL thread does the writing; settings_status() says when it is done."""
    return _settings_job(link, f"SETTINGS_EXPORT {path}" if path else "SETTINGS_EXPORT")


def settings_import(link: CdcLink, path: str) -> dict[str, Any]:
    """Start validating, applying and persisting a backup JSON; result in settings_status()."""
    return _settings_job(link, f"SETTINGS_IMPORT {path}")


def settings_status(link: CdcLink) -> dict[str, Any]:
    """{busy, op: "export"|"import"|None, ok: True|False|None, err, path} of the last job."""
    reply = link.transact("SETTINGS_STATUS", "SETTINGS:")
    if (reply.parsed or {}).get("kind") == "err":
        _raise_err("SETTINGS_STATUS", reply, NOT_SUPPORTED)
    m = _SETTINGS_STATUS_RE.match(reply.line)
    if m is None:
        raise HilError(NOT_SUPPORTED, "SETTINGS_STATUS: unparseable reply",
                       hint="firmware predates this verb?", line=reply.line)
    busy, op, ok, err, path = m.groups()
    return {"busy": busy == "1", "op": {"e": "export", "i": "import"}.get(op),
            "ok": {"1": True, "0": False}.get(ok), "err": None if err == "-" else err,
            "path": None if path == "-" else path}


def imu(link: CdcLink) -> dict[str, Any]:
    """Fused attitude, rate and acceleration, plus the IMU->MIDI slot table.

    Until this existed, "is the IMU alive" could only be answered by the absence
    of an error line in the console, which is how a working sensor spent a day
    being described as dead.
    """
    lines = link.transact_multi("IMU", "IMU:", end="IMU: end")
    out: dict[str, Any] = {"slots": []}
    for line in lines:
        body = line.split(":", 1)[1].strip()
        if body.startswith("end"):
            continue
        if body.startswith("slot"):
            slot: dict[str, Any] = {}
            for field in body.split():
                key, _, value = field.partition("=")
                if value:
                    slot[key] = int(value)
                else:
                    slot["index"] = int(key.removeprefix("slot"))
            out["slots"].append(slot)
            continue
        # The measurement row: scalars, and comma-separated triples for the
        # gyro and the accelerometer.
        for field in body.split():
            key, _, value = field.partition("=")
            if not value:
                continue
            try:
                nums = [float(p) for p in value.split(",")]
            except ValueError:
                out[key] = value
                continue
            out[key] = nums[0] if len(nums) == 1 else nums
    return out


def imu_midi(link: CdcLink, on: bool) -> dict[str, Any]:
    """Turn board-motion-to-MIDI on or off. Persisted; the task only exists while on."""
    return _ok(link, f"IMU_MIDI {1 if on else 0}")


def heap_trace(link: CdcLink, what: str) -> dict[str, Any]:
    """Standalone heap tracing on the device; internal RAM unless DUMP carries ALL.

    START [records] arms it (LEAKS mode: a record survives only while its block
    is live), STOP freezes it, DUMP [min_bytes] [ALL] returns the live allocations
    grouped by their first two callers, largest group first. The PCs are raw;
    resolve them on the host with addr2line against the ELF the board runs.
    Same reply shape for all three: {"groups": [...], "records", "on"}.
    """
    parts = what.split()
    if not parts or parts[0].upper() not in ("START", "STOP", "DUMP"):
        raise HilError(BAD_ARGS, f"HEAP_TRACE: {what!r} is not START|STOP|DUMP")
    lines = link.transact_multi(f"HEAP_TRACE {what}", "HEAP_TRACE:",
                                end="HEAP_TRACE: end", timeout_s=20.0)
    groups: list[dict[str, Any]] = []
    out: dict[str, Any] = {"groups": groups}
    for line in lines:
        body = line.split(":", 1)[1].strip()
        fields: dict[str, Any] = {}
        for field in body.split()[1:] if body.startswith("end") else body.split():
            key, _, value = field.partition("=")
            if not value:
                continue
            if key == "pc":
                fields[key] = value.split(",")
                continue
            try:
                fields[key] = int(value)
            except ValueError:
                fields[key] = value
        if body.startswith("end"):
            out.update(fields)
        elif fields:
            groups.append(fields)
    return out


def imu_gesture(link: CdcLink, gesture: str) -> None:
    """Post one motion gesture on the event bus (`IMU_GESTURE`), as the IMU would.

    `gesture`: TAP, DOUBLE_TAP, FREE_FALL, WAKE_UP, ORIENTATION or STILL.
    """
    gesture = gesture.strip().upper()
    if gesture not in ("TAP", "DOUBLE_TAP", "FREE_FALL", "WAKE_UP", "ORIENTATION", "STILL"):
        raise HilError(BAD_ARGS, f"unknown gesture {gesture!r}",
                       hint="TAP|DOUBLE_TAP|FREE_FALL|WAKE_UP|ORIENTATION|STILL")
    _ok(link, f"IMU_GESTURE {gesture}")


def imu_ths(link: CdcLink, tap: int, wake: int) -> None:
    """Set the IMU's embedded tap (0-31) and wake-up (0-63) thresholds; not persisted."""
    if not (0 <= tap <= 31 and 0 <= wake <= 63):
        raise HilError(BAD_ARGS, f"IMU_THS: tap {tap} / wake {wake} out of range (0-31 / 0-63)")
    link.transact(f"IMU_THS {tap} {wake}", "OK")


def ulp(link: CdcLink) -> dict[str, int]:
    """What the ULP RISC-V coprocessor program has counted.

    ``heartbeat`` is one per coprocessor run, every ``period_ms``, whatever the
    main CPU is doing.
    """
    reply = link.transact("ULP", "ULP:")
    out: dict[str, int] = {}
    for field in reply.line.split(":", 1)[1].split():
        key, _, value = field.partition("=")
        if value:
            out[key] = int(value)
    return out


def worker(link: CdcLink) -> list[dict[str, Any]]:
    """The bsp_worker job table: one dict per job.

    The housekeeping loops that used to be tasks of their own (softoff,
    pad_anim, imu_midi, sd_hotplug, perfmon) run from one task; this is
    how a bench sees whether any of them is too slow for a shared loop --
    ``overruns`` counts due dates missed by more than a period, ``max_us`` is
    the longest body so far, and a max_us near period*1000 is the warning.
    """
    lines = link.transact_multi("WORKER", "WORKER:", end="WORKER: end", timeout_s=5.0)
    jobs: list[dict[str, Any]] = []
    for line in lines:
        body = line.split(":", 1)[1].strip()
        if body.startswith("end"):
            continue
        job: dict[str, Any] = {}
        for field in body.split():
            key, _, value = field.partition("=")
            if not value:
                continue
            try:
                job[key] = int(value)
            except ValueError:
                job[key] = value
        if "name" in job:
            jobs.append(job)
    return jobs


_UITREE_RE = re.compile(
    r"i=(\d+) p=(-?\d+) cls=(\S+) x=(-?\d+) y=(-?\d+) w=(-?\d+) h=(-?\d+) "
    r"int=([01]) hit=([01]) chk=([01]) dis=([01]) foc=([01]) text=(.*)$")


def ui_tree(link: CdcLink, timeout_s: float = 5.0) -> dict[str, Any]:
    """``UI_TREE``: every visible widget of the screen and the top layer, pre-order.

    ``{"running": <app or "-">, "nodes": [{i, parent, cls, x, y, w, h, interactive, hit,
    checked, disabled, focused, text}]}``, in LCD pixels (what ``TOUCH_TAP`` takes).
    ``hit``: a press on the widget's middle reaches it, not something drawn over it.
    ``text`` is the widget's own label, else the first label under it; "" for none."""
    lines = link.transact_multi("UI_TREE", "UITREE:", end="UITREE: end", timeout_s=timeout_s)
    nodes: list[dict[str, Any]] = []
    running = "-"
    for line in lines:
        body = line.split(":", 1)[1].strip()
        if body.startswith("end"):
            running = _kv_line(body).get("running", "-")
            continue
        m = _UITREE_RE.match(body)
        if not m:
            continue
        nodes.append({"i": int(m.group(1)), "parent": int(m.group(2)), "cls": m.group(3),
                      "x": int(m.group(4)), "y": int(m.group(5)), "w": int(m.group(6)),
                      "h": int(m.group(7)), "interactive": m.group(8) == "1",
                      "hit": m.group(9) == "1", "checked": m.group(10) == "1",
                      "disabled": m.group(11) == "1", "focused": m.group(12) == "1",
                      "text": m.group(13).strip()})
    return {"running": str(running), "nodes": nodes}


def touch_tap(link: CdcLink, x: int, y: int, hold_ms: int = 80) -> dict[str, Any]:
    """``TOUCH_TAP x y [hold_ms]``: a finger on the panel at LCD (x, y), through the
    panel's read path. Replies at once; the contact lasts ``hold_ms`` after it."""
    _check_range("hold_ms", hold_ms, 1, 5000)
    return _ok(link, f"TOUCH_TAP {x} {y} {hold_ms}", err_code=BAD_ARGS)


def touch_drag(link: CdcLink, x1: int, y1: int, x2: int, y2: int, ms: int = 300,
               rest_ms: int = 0) -> dict[str, Any]:
    """``TOUCH_DRAG``: a finger pressed at (x1, y1), moved to (x2, y2) over @p ms,
    rested @p rest_ms, lifted. Replies at once; the contact takes ms + rest_ms."""
    _check_range("ms", ms, 1, 5000)
    _check_range("rest_ms", rest_ms, 0, 5000)
    return _ok(link, f"TOUCH_DRAG {x1} {y1} {x2} {y2} {ms} {rest_ms}", err_code=BAD_ARGS)


def topbar(link: CdcLink) -> dict[str, Any]:
    """``TOPBAR``: the top row, the drawer, the encoder's place, the last swipe,
    ``map`` (the MODE map is up) and ``pads`` (the active pad logic)."""
    return _kv_line(_line(link, "TOPBAR", "TOPBAR:"), "TOPBAR:")


def touch_trace(link: CdcLink, what: str) -> dict[str, Any]:
    """Raw touch samples, taken by wrapping the indev read_cb on the device.

    LVGL's own gesture detector is unusable for a screen-wide swipe: it gives up
    the moment the finger lands on a scrollable object, which on this UI is the
    launcher, every list and every settings page. So the swipe has to be built
    from the coordinate stream, and this is how that stream gets off the board.

    Every subcommand answers in the same shape:
    {"samples": [...], "n", "on", "swipes", "last"}. Only DUMP carries rows;
    ON/OFF/CLEAR return the trailer alone, so the caller never has to know which
    reply shape to read. Samples are oldest first.

    ``swipes`` and ``last`` are the recogniser's own counters, and they are what
    separates "the swipe was not recognised" from "it was recognised and the
    action did nothing" -- two failures that look identical on a screen.
    """
    what = what.upper()
    if what not in ("ON", "OFF", "CLEAR", "DUMP"):
        raise HilError(BAD_ARGS, f"TOUCH_TRACE: {what} is not ON|OFF|CLEAR|DUMP")

    lines = link.transact_multi(f"TOUCH_TRACE {what}", "TOUCH_TRACE:",
                                end="TOUCH_TRACE: end", timeout_s=20.0)
    samples: list[dict[str, Any]] = []
    out: dict[str, Any] = {"samples": samples}
    for line in lines:
        body = line.split(":", 1)[1].strip()
        if body.startswith("end"):
            for field in body.split()[1:]:
                key, _, value = field.partition("=")
                if not value:
                    continue
                try:
                    out[key] = int(value)
                except ValueError:
                    out[key] = value   # last=<direction>
            continue
        parts = body.split()
        if len(parts) != 4:
            continue
        t_ms, pressed, x, y = (int(p) for p in parts)
        samples.append({"t_ms": t_ms, "pressed": bool(pressed), "x": x, "y": y})
    return out


# ── kit_* ──────────────────────────────────────────────────────────────

def kit_list(link: CdcLink) -> dict[str, Any]:
    """Multi-line since the single-line form was outgrown twice; the trailer
    carries the count, so a short listing is a failure rather than a guess."""
    from .parsers import parse_cdc_reply

    kits: list[dict[str, Any]] = []
    current, count = -1, None
    try:
        lines = link.transact_multi("KIT_LIST", "KITS:", end="KITS: end")
    except HilError as exc:
        if not _scanning(exc):
            raise
        raise HilError(TIMEOUT, "KIT_LIST: the kit scan is still walking the card",
                       hint="retry in a second; the scan holds the kit vector and "
                            "the boot one starts only once the launcher is up") from exc
    for line in lines:
        parsed = parse_cdc_reply(line)
        if not parsed or parsed.get("kind") != "kits":
            continue
        if parsed.get("end"):
            current = int(parsed.get("current", -1))
            count = int(parsed.get("count", 0))
            continue
        kits.extend({"id": int(k["id"]), "name": str(k["name"])} for k in parsed["kits"])
    if count is not None and count != len(kits):
        raise HilError(NOT_SUPPORTED,
                       f"KIT_LIST: trailer says {count} kits, {len(kits)} arrived",
                       hint="a dropped line, not a short card", got=len(kits), want=count)
    return {"kits": kits, "current": current, "count": count if count is not None else len(kits)}


def kit_status(link: CdcLink) -> dict[str, Any]:
    p = _query(link, "KIT_STATUS", "KITSTATUS:", "kitstatus")
    return {"current": int(p["current"]), "loading": bool(p["loading"]),
            "pending": int(p["pending"]), "name": p.get("name")}


def wave_status(link: CdcLink) -> dict[str, Any]:
    """Waveform-cache coverage of the current kit: total/cached layers, the
    (pad, layer) pairs the Sampler would draw empty, and the loader's counters."""
    return _without_kind(_query(link, "WAVE_STATUS", "WAVE:", "wave"))


_KITSWAP_RING = 8          # hil_control.cpp keeps the last 8 swaps


def kit_swap_log(link: CdcLink, *,
                 log: Callable[[str], None] | None = None) -> list[dict[str, Any]]:
    """The device's own account of its last swaps, oldest first — evidence, not a gate.

    One row per swap: when it was requested, how long parsing and the pad
    reloads took, how much render time the swap gate cost, which pads waited for
    their sample to finish (`deferred`) and which were reloaded while still
    sounding (`forced`).

    A firmware that does not know the verb answers nothing parseable, and that
    is deliberately *not* an error here: `kit_swap_audio` scores the recording,
    and this log only explains what the recording shows. An unknown verb, a
    timeout or a garbled reply all yield an empty list plus one line through
    `log`, so the scenario keeps its verdict either way.
    """
    try:
        lines = link.transact_multi("KIT_SWAP_LOG", "KITSWAP:", end="KITSWAP: end")
    except HilError as exc:
        if log is not None:
            log(f"KIT_SWAP_LOG: {exc.code}: {exc.message} — no swap log this run")
        return []
    rows: list[dict[str, Any]] = []
    count: int | None = None
    for line in lines:
        parsed = parse_cdc_reply(line)
        if not parsed or parsed.get("kind") != "kitswap":
            continue
        if parsed.get("end"):
            count = int(parsed.get("count", 0))
            continue
        rows.append({k: v for k, v in parsed.items() if k not in ("kind", "end")})
    if count is not None and count != len(rows) and log is not None:
        log(f"KIT_SWAP_LOG: trailer says {count} rows, {len(rows)} arrived "
            "— lines lost on the CDC under the pad storm")
    if not rows and log is not None:
        log("KIT_SWAP_LOG: no rows (firmware predates the verb, or the ring is empty)")
    return rows[-_KITSWAP_RING:]


def kit_load(link: CdcLink, kit_id: int, wait_s: float = 15.0) -> dict[str, Any]:
    if not isinstance(kit_id, int) or isinstance(kit_id, bool) or kit_id < 0:
        raise HilError(BAD_ARGS, f"kit_id must be a non-negative int, got {kit_id!r}",
                       hint="KIT_LIST names the kits", kit_id=kit_id)
    try:
        _ok(link, f"KIT_LOAD {kit_id}", err_code=BAD_ARGS)
    except HilError as exc:
        if not _scanning(exc):
            raise
        raise HilError(TIMEOUT, f"KIT_LOAD {kit_id}: the kit scan is still walking the card",
                       hint="retry in a second; the id you quoted came from a "
                            "listing the scan has not finished",
                       kit_id=kit_id) from exc
    # From hil_kit_churn.parse_status: landed == current==kit and not loading and
    # nothing pending. KIT_LIST's current= alone cannot tell "loading" from
    # "the request went nowhere".
    deadline = time.monotonic() + max(0.0, wait_s)
    status = kit_status(link)
    while True:
        if status["current"] == kit_id and not status["loading"] and status["pending"] == -1:
            return status
        if time.monotonic() >= deadline:
            break
        time.sleep(_POLL_S)
        status = kit_status(link)
    raise HilError(TIMEOUT, f"KIT_LOAD {kit_id}: not current after {wait_s:.1f}s "
                   f"(current={status['current']} loading={int(status['loading'])} "
                   f"pending={status['pending']})",
                   hint="a load that never finishes blocks every later swap; "
                        "check the console for SD errors", status=status, kit_id=kit_id)


# ── pad_* ──────────────────────────────────────────────────────────────

def pad_press(link: CdcLink, idx: int, vel: int = 127) -> dict[str, Any]:
    _check_range("idx", idx, 0, 15)
    _check_range("vel", vel, 0, 127)
    return _ok(link, f"PAD_PRESS {idx} {vel}")


def pad_release(link: CdcLink, idx: int) -> dict[str, Any]:
    _check_range("idx", idx, 0, 15)
    return _ok(link, f"PAD_RELEASE {idx}")


def pad_pressure(link: CdcLink, idx: int, val: int) -> dict[str, Any]:
    _check_range("idx", idx, 0, 15)
    _check_range("val", val, 0, 255)
    return _ok(link, f"PAD_PRESSURE {idx} {val}")


def pad_pan(link: CdcLink, idx: int, val: int) -> dict[str, Any]:
    """Where a held pad is pressed, 0 left .. 64 middle .. 127 right (`PAD_PAN`)."""
    _check_range("idx", idx, 0, 15)
    _check_range("val", val, 0, 127)
    return _ok(link, f"PAD_PAN {idx} {val}")


# Every counter PADSTATS carries past the first four (firmware since 2026-09): where a
# hit can be lost on its way in -- the inject queue, the STM UART, its sbuf, USB MIDI.
_PAD_STATS_COUNTERS = ("inj_drop", "stm", "at", "rxmsg", "uart_ovf", "sbuf_drop", "usb_tx",
                       "usb_drop")


def pad_stats(link: CdcLink, reset: bool = False) -> dict[str, Any]:
    """PADSTATS: press/release/played/freeslots and, when the firmware sends them, the
    loss counters (inj_drop, uart_ovf, sbuf_drop, usb_drop, ...). A scenario that diffs
    them must see them: dropping them here kept every pad-path gate at zero."""
    if reset:
        _ok(link, "PAD_STATS_RESET")
    p = _query(link, "PAD_STATS", "PADSTATS:", "padstats")
    out = {"press": int(p["press"]), "release": int(p["release"]),
           "played": int(p["played"]), "freeslots": int(p["freeslots"])}
    for key in _PAD_STATS_COUNTERS:
        if isinstance(p.get(key), int):
            out[key] = int(p[key])
    return out


def pad_notes(link: CdcLink) -> dict[str, Any]:
    p = _query(link, "PAD_NOTES", "PADNOTES:", "padnotes")
    return {"notes": {int(k): int(v) for k, v in p.get("notes", {}).items()}}


def pad_info(link: CdcLink, idx: int) -> dict[str, Any]:
    _check_range("idx", idx, 0, 15)
    reply = link.transact(f"PAD_INFO {idx}", "PADINFO:")
    if reply.parsed is not None and reply.parsed.get("kind") == "err":
        _raise_err("PAD_INFO", reply, BAD_ARGS)
    if not reply.line.startswith("PADINFO:"):
        raise HilError(NOT_SUPPORTED, "PAD_INFO: unparseable reply",
                       hint="firmware predates this verb?", line=reply.line)
    return {"raw": reply.line}


def pad_mix(link: CdcLink) -> dict[str, Any]:
    """The current kit's sixteen pad volumes (`PAD_MIX`), in the kit's units.

    {"kit", "name", "max" (127, a pitched kit 100), "loading", "pads": [{"pad",
    "vol", "pan", "sample"}] * 16}. Prefixes are passed explicitly so a hub
    started before the verb existed still carries it.
    """
    try:
        lines = link.transact_multi("PAD_MIX", "PADMIX:", end="PADMIX: end")
    except HilError as exc:
        if not _scanning(exc):
            raise
        raise HilError(TIMEOUT, "PAD_MIX: the kit scan is still walking the card",
                       hint="retry in a second") from exc
    out: dict[str, Any] = {"kit": -1, "name": None, "max": 127, "loading": False, "pads": []}
    count = None
    for line in lines:
        parsed = parse_cdc_reply(line)
        if not parsed:
            continue
        if parsed.get("kind") == "err":
            raise HilError(NOT_SUPPORTED, f"PAD_MIX: device replied ERR {parsed['message']}",
                           hint="no kit loaded, or firmware predates the verb", reply=line)
        if parsed.get("kind") != "padmix":
            continue
        if parsed.get("end"):
            count = int(parsed["count"])
        elif "pad" in parsed:
            out["pads"].append({k: parsed[k] for k in ("pad", "vol", "pan", "sample")})
        else:
            out.update({k: parsed[k] for k in ("kit", "name", "max", "loading")})
    if count is None or len(out["pads"]) != count:
        raise HilError(NOT_SUPPORTED, f"PAD_MIX: {len(out['pads'])} pad rows arrived, "
                       f"trailer {count}", hint="a dropped line, or firmware predates the verb")
    return out


def pad_mix_set(link: CdcLink, idx: int, vol: int) -> dict[str, Any]:
    """One pad's volume, live (`PAD_MIX idx vol`): into the kit, not the card."""
    _check_range("idx", idx, 0, 15)
    _check_range("vol", vol, 0, 127)
    reply = link.transact(f"PAD_MIX {idx} {vol}", "OK")
    if (reply.parsed or {}).get("kind") == "err":
        _raise_err("PAD_MIX", reply, BAD_ARGS)
    return {"ok": True}


def kit_save(link: CdcLink, wait_s: float = 5.0) -> dict[str, Any]:
    """The current kit into its kit.json (`KIT_SAVE`), waiting for KIT_SAVE_STATUS."""
    reply = link.transact("KIT_SAVE", "OK")
    if (reply.parsed or {}).get("kind") == "err":
        _raise_err("KIT_SAVE", reply, NOT_SUPPORTED)
    deadline = time.monotonic() + max(0.0, wait_s)
    while True:
        st = _query(link, "KIT_SAVE_STATUS", "KITSAVE:", "kitsave")
        if not st.get("busy") and st.get("ok") in (0, 1):
            if st["ok"] != 1:
                raise HilError(NOT_SUPPORTED, f"KIT_SAVE: kit {st.get('kit')} was not written",
                               hint="the card refused it, or a kit load started meanwhile",
                               status=_without_kind(st))
            return _without_kind(st)
        if time.monotonic() >= deadline:
            raise HilError(TIMEOUT, f"KIT_SAVE: not done after {wait_s:.1f}s",
                           status=_without_kind(st))
        time.sleep(0.05)


# ── enc_* / ui ─────────────────────────────────────────────────────────

def enc_rotate(link: CdcLink, delta: int) -> dict[str, Any]:
    _check_range("delta", delta, -16, 16)      # firmware clamps at ±16
    if delta == 0:
        raise HilError(BAD_ARGS, "delta must be non-zero (ENC_ROTATE 0 is ignored)",
                       hint="delta=-16..16, not 0", arg="delta", value=0)
    return _ok(link, f"ENC_ROTATE {delta}")


def enc_press(link: CdcLink, ms: int = 80) -> dict[str, Any]:
    _check_range("ms", ms, 1, 5000)            # firmware clamps at 5000
    return _ok(link, f"ENC_PRESS {ms}")


def enc_group(link: CdcLink) -> dict[str, Any]:
    # "ENCGROUP: count=N" then N lines "  [i] <ptr> <label>"; no end marker, so
    # the link reads until the port goes idle.
    lines = link.transact_multi("ENC_GROUP", "ENCGROUP:", end=None)
    if not lines:
        raise HilError(NOT_SUPPORTED, "ENC_GROUP: no reply",
                       hint="firmware predates this verb?")
    if lines[0].startswith("ENCGROUP: ERR"):
        raise HilError(TIMEOUT, f"ENC_GROUP: {lines[0]}",
                       hint="LVGL lock held >1 s; the UI thread is busy or stuck",
                       line=lines[0])
    return {"group": parse_enc_group(lines)}


def enc_focus(link: CdcLink) -> dict[str, Any]:
    reply = link.transact("ENC_FOCUS", "ENCFOCUS:")
    if reply.line.startswith("ENCFOCUS: ERR"):
        raise HilError(TIMEOUT, f"ENC_FOCUS: {reply.line}",
                       hint="LVGL lock held >1 s; the UI thread is busy or stuck",
                       line=reply.line)
    p = _kind("ENC_FOCUS", reply, "encfocus")
    out = {"index": int(p["index"]), "label": str(p.get("label", "")), "ptr": str(p["ptr"])}
    if "editing" in p:
        out["editing"] = bool(p["editing"])
    return out


def enc_state(link: CdcLink) -> dict[str, Any]:
    return _without_kind(_query(link, "ENC_STATE", "ENC:", "enc"))


def ui_state(link: CdcLink) -> dict[str, Any]:
    return _without_kind(_query(link, "UI_STATE", "UI:", "ui"))


def daw_state(link: CdcLink) -> dict[str, Any]:
    """What DAW Control shows: rows, transport bits, beat count, host id.

    `beat` is consumed by the app's 50 ms flash; compare `beats` (a count of
    non-zero beat frames) between two reads instead.
    """
    return _without_kind(_query(link, "DAW_STATE", "DAW:", "daw"))


# ── led / mem / stats ──────────────────────────────────────────────────

def led_state(link: CdcLink) -> dict[str, Any]:
    p = _without_kind(_query(link, "LED_STATE", "LEDS:", "leds"))
    colors = list(p.get("colors", []))
    if len(colors) != 16:
        raise HilError(NOT_SUPPORTED, f"LED_STATE: expected 16 colors, got {len(colors)}",
                       hint="truncated reply?", colors=colors)
    p["colors"] = colors
    return p


def seq(link: CdcLink, action: str | None = None,
        timeout_s: float | None = None) -> dict[str, Any]:
    """The loop recorder: `action` (REC, PLAY, "BARS 1", "COUNTIN 0", ...) first, then the
    report. A kit load saves a dirty sequence to the card first, and the CDC loop
    waits for that write: give `timeout_s` when one may be under way."""
    return _without_kind(_query(link, f"SEQ {action}" if action else "SEQ", "SEQ:", "seq",
                                timeout_s))


def song(link: CdcLink, action: str | None = None) -> dict[str, Any]:
    """The scene arrangement: `action` ("ADD s r", "PLAY 0", ...) first, then the report."""
    return _without_kind(_query(link, f"SONG {action}" if action else "SONG", "SONG:", "song"))


def asong(link: CdcLink, action: str | None = None,
          timeout_s: float | None = None) -> dict[str, Any]:
    """The audio song. Every action only asks (ok=1: taken); poll the bare verb for
    open=/play=/bounce=. open: 0 idle 1 busy 2 ready 3 failed; bounce: 0 idle 1 running
    2 done 3 failed."""
    return _without_kind(_query(link, f"ASONG {action}" if action else "ASONG", "ASONG:", "asong",
                                timeout_s))


def _asong_request(link: CdcLink, action: str, what: str) -> dict[str, Any]:
    """An ASONG action that can be refused: ok=0 in the answering status is the refusal."""
    st = asong(link, action)
    if st.get("ok") != 1:
        raise HilError(BAD_ARGS, f"ASONG {what} refused (ok=0)",
                       hint="no free track, or the song is not open", status=st)
    return st


def asong_arm(link: CdcLink, track: int) -> dict[str, Any]:
    """``ASONG ARM track``: the seq track REC/DUB take onto (Arrange's ARM)."""
    _check_range("track", track, 0, 3)
    return asong(link, f"ARM {track}")


def asong_seqtrack(link: CdcLink) -> dict[str, Any]:
    """``ASONG SEQTRACK``: a kit track on the lowest free track, armed, with a scene of its
    own, playing the pads' kit (route 0). BAD_ARGS when the song has no free track."""
    return _asong_request(link, "SEQTRACK", "SEQTRACK")


def asong_kit2(link: CdcLink, kit_json: str) -> dict[str, Any]:
    """``ASONG KIT2 kitjson``: like :func:`asong_seqtrack`, on the song's second kit
    (route 1); ``kit_json`` is a kit.json path as the kit list names it."""
    if not kit_json or any(c in kit_json for c in "\r\n"):
        raise HilError(BAD_ARGS, f"kit_json must be a one-line path, got {kit_json!r}")
    return _asong_request(link, f"KIT2 {kit_json}", "KIT2")


def asong_seq(link: CdcLink, track: int | None = None, timeout_s: float = 3.0) -> dict[str, Any]:
    """``ASONG_SEQ [track]``: a seq track's blocks and the take state.

    Rows ``ASEQ: bar= bars= from= notes= own=`` (own=1 for notes of the block's own, 0 for
    the scene's), then ``ASEQ: track= take= punchable= edit= route= armed=``. Returns
    ``{"blocks": [...], **trailer}``; ``track`` is -1 when no seq track matches. Without an
    engine the board answers ``ASEQ: no engine`` -> NOT_SUPPORTED."""
    if track is not None:
        _check_range("track", track, 0, 3)
    lines = link.transact_multi(f"ASONG_SEQ {track}" if track is not None else "ASONG_SEQ",
                                "ASEQ:", timeout_s=timeout_s)
    blocks: list[dict[str, Any]] = []
    trailer: dict[str, Any] | None = None
    for line in lines:
        body = line.split(":", 1)[1].strip()
        if body == "no engine":
            raise HilError(NOT_SUPPORTED, "ASONG_SEQ: no song engine", line=line)
        kv = _kv_line(body)
        if "bar" in kv:
            blocks.append(kv)
        elif "track" in kv:
            trailer = kv
    if trailer is None:
        raise HilError(NOT_SUPPORTED, "ASONG_SEQ: no trailer line",
                       hint="firmware predates this verb?", lines=lines)
    return {"blocks": blocks, **trailer}


def kit2(link: CdcLink) -> dict[str, Any]:
    """``KIT2``: the song's second kit (ISecondKit): state, want/have kit, slots, pool,
    voices, counters, ``kit1_lim`` and ``engine``. ``ERR no_second_kit`` -> NOT_SUPPORTED."""
    return _kv_line(_line(link, "KIT2", "KIT2:"), "KIT2:")


def inst_status(link: CdcLink) -> dict[str, Any]:
    """``INST_STATUS``: Pitch Play's instrument host -- ``state`` (idle, busy, ready,
    failed), ``owned``, ``beside`` (built beside the kits' engines), ``last_ms``, ``msg``."""
    return _kv_line(_line(link, "INST_STATUS", "INST:"), "INST:")


def kit2_note(link: CdcLink, pad: int, velocity: int = 100) -> dict[str, Any]:
    """``KIT2 NOTE pad vel``: play a pad of the second kit without a song (vel clamps to 1..127)."""
    _check_range("pad", pad, 0, 15)
    return _ok_any(link, f"KIT2 NOTE {pad} {velocity}", NOT_SUPPORTED)


def kit2_off(link: CdcLink, pad: int) -> dict[str, Any]:
    """``KIT2 OFF pad``: release a pad of the second kit."""
    _check_range("pad", pad, 0, 15)
    return _ok_any(link, f"KIT2 OFF {pad}", NOT_SUPPORTED)


def reset_count(link: CdcLink) -> int:
    """How many resets the board's RESET_LOG ring has recorded since it was cleared."""
    return int(_query(link, "RESET_LOG", "RSTLOG:", "rstlog")["n"])


def clock_status(link: CdcLink, reset: bool = False) -> dict[str, Any]:
    """MIDI clock state (`CLOCK_STATUS`); `reset` clears the jitter/late/gap stats after reading."""
    return _without_kind(_query(link, "CLOCK_STATUS RESET" if reset else "CLOCK_STATUS",
                                "CLOCK:", "clock"))


def mem(link: CdcLink) -> dict[str, Any]:
    return _without_kind(_query(link, "MEM", "MEM:", "mem"))


def mem_blocks(link: CdcLink) -> dict[str, Any]:
    # Two lines by design (hil_control.cpp: the CDC endpoint takes ~512 B per
    # write, the block list would be cut off if it shared the first).
    lines = link.transact_multi("MEM_BLOCKS", "MEMBLK:", end="MEMBIG:")
    summary: dict[str, Any] | None = None
    big: list[dict[str, Any]] = []
    for line in lines:
        parsed = parse_cdc_reply(line)
        if not parsed:
            continue
        if parsed.get("kind") == "memblk":
            summary = _without_kind(parsed)
        elif parsed.get("kind") == "membig":
            big = list(parsed.get("blocks", []))
    if summary is None:
        raise HilError(NOT_SUPPORTED, "MEM_BLOCKS: no MEMBLK line",
                       hint="firmware predates this verb?", lines=lines)
    return {"summary": summary, "big": big}


def cdc_stats(link: CdcLink) -> dict[str, Any]:
    p = _query(link, "CDC_STATS", "CDCSTATS:", "cdcstats")
    out: dict[str, Any] = {"rx": int(p["rx"]), "cmds": int(p["cmds"]), "drop": int(p["drop"]),
                           "maxcmd_us": int(p.get("maxcmd_us", 0))}
    # Since firmware 2.0: the least main-task stack any command left free since
    # boot, the verb that did it, and what the previous command left.
    for key in ("stack_min", "stack_last"):
        if str(p.get(key, "")).isdigit():
            out[key] = int(p[key])
    if "stack_verb" in p:
        out["stack_verb"] = p["stack_verb"]
    return out


def ota_prefs(link: CdcLink) -> bool:
    """Whether an OTA over CDC is accepted without the on-screen Allow
    (NVS ota_prefs/auto_accept). Read-only."""
    reply = link.transact("OTA_PREFS", "OTA_PREFS:", timeout_s=2.0)
    m = re.match(r"OTA_PREFS:\s*auto_accept=([01])\s*$", reply.line.strip())
    if m is None:
        raise HilError(NOT_SUPPORTED, f"OTA_PREFS: unparseable reply {reply.line!r}",
                       hint="firmware predates this verb?", line=reply.line)
    return m.group(1) == "1"


def lvgl_stats(link: CdcLink) -> dict[str, Any]:
    """LVGL refresh accounting since the last read (the device clears it):
    refresh count and max/avg µs, render µs (total/max), flush-wait µs
    (total/max) and flush count. Bracket a UI action with two calls."""
    p = _without_kind(_query(link, "LVGL_STATS", "LVGLSTATS:", "lvglstats"))
    return {k: int(v) for k, v in p.items()}


def audio_level(link: CdcLink) -> dict[str, Any]:
    p = _query(link, "AUDIO_LEVEL", "AUDIOLVL:", "audiolvl")
    return {"left": float(p["left"]), "right": float(p["right"]),
            "amp": bool(p["amp"]), "allowed": bool(p["allowed"])}


_INGAIN_PAIR = re.compile(r"(\w+)=(-?\d+)")


def ingain(link: CdcLink, input_index: int | None = None,
           db: int | None = None) -> dict[str, int]:
    """The inputs' preamp gain in dB, ``{"IN1": 24, "IN2": 0}``; IN1 is codec 0
    (the mics), IN2 codec 1 (the line-in jack, the bench's DAC->ADC loop).

    With ``input_index`` and ``db`` it sets that input first and answers with
    the new state. **The board keeps the setting** (NVS), so a caller that
    changes it for a measurement puts the old value back afterwards.
    """
    if (input_index is None) != (db is None):
        raise HilError(BAD_ARGS, "ingain: pass both input_index and db, or neither")
    cmd = "INGAIN" if input_index is None else f"INGAIN {int(input_index)} {int(db)}"
    reply = link.transact(cmd, "INGAIN:")
    line = (reply.line or "").strip()
    if line.startswith("ERR"):
        raise HilError(NOT_SUPPORTED, f"INGAIN: device replied {line}",
                       hint="this board has no input-gain control", reply=line)
    if not line.startswith("INGAIN:"):
        raise HilError(NOT_SUPPORTED, "INGAIN: unparseable reply",
                       hint="firmware predates this verb?", line=line)
    return {k: int(v) for k, v in _INGAIN_PAIR.findall(line[len("INGAIN:"):])}


def smpl_peak(link: CdcLink) -> dict[str, Any]:
    p = _query(link, "SMPL_PEAK", "SMPLPEAK:", "smplpeak")
    return {"peak": int(p["peak"]), "free": int(p["free"])}


def smpl_counters(link: CdcLink) -> dict[str, Any]:
    """SMPL_PEAK with the engine counters newer firmware appends: ``poly`` (the voice
    cap), ``rt_max_us`` and ``rt_busy_max_us`` (read-clear), ``rxskip``, ``unpaced``,
    ``noblock`` and ``underrun`` (cumulative; ``underrun`` is the native engine's when it
    holds the kit), ``overruns``. Absent keys: older firmware. ``load_main``/``maxc_*``
    are gone (always nan; asking a stopped prebuilt engine for them crashed)."""
    return _without_kind(_query(link, "SMPL_PEAK", "SMPLPEAK:", "smplpeak"))


def smpl_engine(link: CdcLink) -> dict[str, Any]:
    """``SMPL_ENGINE``: which sample engine holds the kit and the native engine's state
    -- ``engine`` (native|marcel|none), ``state`` (0 idle 1 taking 2 ready 3 failed),
    ``why``, ``kit`` (the kit manager id the engine holds, -1 with none), the pool, voice
    and card counters, ``bg_stack_free`` (audio_bg's stack high-water), ``preview`` (the
    audition: 0 idle 1 priming 2 playing 3 finished) and ``prev_pk`` (its peak since the
    last read), and ``engines`` (the engines this image was built with; ``["native"]``
    unless the prebuilt one was compiled in). Older firmware lacks the newer keys."""
    return _without_kind(_query(link, "SMPL_ENGINE", "SMPLENG:", "smpleng"))


def smpl_engine_select(link: CdcLink, engine: str) -> dict[str, Any]:
    """``SMPL_ENGINE native|marcel``: hand the kit to that engine. ``native`` answers at
    once and takes the kit on its own task -- poll :func:`smpl_engine` for ``state``.
    ``marcel`` raises NOT_SUPPORTED on an image without the prebuilt engine."""
    if engine not in ("native", "marcel"):
        raise HilError(BAD_ARGS, f"engine must be native or marcel, got {engine!r}", engine=engine)
    reply = link.transact(f"SMPL_ENGINE {engine}", "OK")
    line = (reply.line or "").strip()
    if not line.startswith("OK"):
        raise HilError(NOT_SUPPORTED, f"SMPL_ENGINE {engine}: device replied {line}",
                       hint="ERR only_native: this image has the native engine only",
                       reply=line)
    return {"ok": True, "reply": line}


def smpl_pitch(link: CdcLink, pad: int, semitones: int) -> dict[str, Any]:
    """``SMPL_PITCH pad semitones``: a pad's step through the pad editor's door (the
    native engine takes it as a step, no file is rendered). ``{pad, semi, engine}``."""
    _check_range("pad", pad, 0, 15)
    _check_range("semitones", semitones, -24, 24)
    return _without_kind(_query(link, f"SMPL_PITCH {pad} {semitones}", "SMPLPITCH:", "smplpitch"))


# ── pitched instrument engine ───────────────────────────────────────────

def pitched_status(link: CdcLink) -> dict[str, Any]:
    """The second sample engine's live state: bank, voices, RT cost, steals."""
    return _without_kind(_query(link, "PITCHED_STATUS", "PITCHED:", "pitched"))


_BENCH_TIMEOUT_BASE_S = 10.0
_BENCH_TIMEOUT_PER_ITER_S = 0.05
_BENCH_TIMEOUT_MAX_S = 300.0


def pitched_bench(link: CdcLink, iters: int | None = None, *,
                  timeout_s: float | None = None) -> dict[str, Any]:
    """IFft 1024/2048/4096 fwd+inv and one Hermite voice render, N times.

    One "BENCH: name=value ..." line per measurement, terminated by "OK" —
    the evidence behind the varispeed-not-spectral per-voice decision. The
    bench task yields a tick per iteration for the idle-task watchdog, so a
    large `iters` needs a matching host timeout: left unset, it scales as
    `10 s + iters * 0.05 s`, capped at 300 s, instead of the flat 10 s that
    used to time a large run out before the device could finish it.
    """
    if iters is not None:
        _check_range("iters", iters, 1, 1_000_000)
    if timeout_s is None:
        timeout_s = min(_BENCH_TIMEOUT_MAX_S,
                        _BENCH_TIMEOUT_BASE_S + (iters or 1) * _BENCH_TIMEOUT_PER_ITER_S)
    cmd = "PITCHED_BENCH" + (f" {iters}" if iters is not None else "")
    lines = link.transact_multi(cmd, "BENCH:", end="OK", timeout_s=timeout_s)
    measurements: dict[str, int] = {}
    for line in lines:
        if not line.startswith("BENCH:"):
            continue
        for tok in line[len("BENCH:"):].strip().split():
            key, sep, value = tok.partition("=")
            if sep:
                try:
                    measurements[key] = int(value)
                except ValueError:
                    measurements[key] = value  # type: ignore[assignment]
    return {"measurements": measurements, "lines": lines}


# ── ble_* ──────────────────────────────────────────────────────────────

def ble_status(link: CdcLink) -> dict[str, Any]:
    return _without_kind(_query(link, "BLE_STATUS", "BLE:", "ble"))


# BLE_START brings NimBLE up synchronously and, with BLE disabled in NVS or the
# internal heap already spoken for, fails only after the controller gives up --
# several seconds after the 2 s a plain OK/ERR verb gets. Silence in that window
# read as TIMEOUT where the reply was an ERR the scenario knows how to map.
BLE_START_TIMEOUT_S = 15.0
# While WiFi holds the radio a BLE start is put off, not made: the board answers
# this and starts BLE itself once the radio stops (after the linger).
BLE_DEFERRED_REPLY = "OK deferred=wifi"


# Firmware that runs BLE_START / BLE_STOP / BLE_ENABLE off the CDC task (on the
# LVGL thread, as the Settings toggle does) answers this at once; the outcome is
# BLE_STATUS req=pending|ok|err|deferred. Older firmware answered the outcome.
BLE_QUEUED_REPLY = "OK queued"
BLE_JOB_POLL_S = 0.2


def _ok_or_deferred(link: CdcLink, cmd: str, timeout_s: float = 2.0) -> dict[str, Any]:
    """A BLE stack verb: ``{"ok": True}``, ``{"ok": True, "deferred": True}`` (WiFi
    holds the radio), or NOT_SUPPORTED when the board refused it (no RAM for the
    stack). Waits for the outcome when the board queued the job."""
    verb = cmd.split(" ", 1)[0]
    deadline = time.monotonic() + timeout_s
    reply = link.transact(cmd, None, timeout_s=timeout_s)
    line = reply.line.strip()
    if line == BLE_DEFERRED_REPLY:
        return {"ok": True, "deferred": True}
    if line == BLE_QUEUED_REPLY:
        return _ble_job_outcome(link, verb, deadline)
    parsed = reply.parsed or {}
    if parsed.get("kind") == "err":
        _raise_err(verb, reply, NOT_SUPPORTED)
    if parsed.get("kind") != "ok":
        raise HilError(NOT_SUPPORTED, f"{verb}: unparseable reply",
                       hint="firmware predates this verb?", line=reply.line)
    return {"ok": True}


def _ble_job_outcome(link: CdcLink, verb: str, deadline: float) -> dict[str, Any]:
    """Poll BLE_STATUS req= until the queued stack job has an outcome."""
    while True:
        st = ble_status(link)
        req = st.get("req")
        if req == "ok":
            return {"ok": True}
        if req == "deferred":
            return {"ok": True, "deferred": True}
        if req == "err":
            raise HilError(NOT_SUPPORTED, f"{verb}: device refused it (BLE_STATUS req=err)",
                           hint="the stack could not come up now -- no internal block "
                                "large enough; see MEM", line=str(st))
        if req != "pending":
            # A firmware without req= never answers OK queued; anything else is a
            # status this helper does not know -- do not spin on it.
            raise HilError(NOT_SUPPORTED, f"{verb}: BLE_STATUS req={req!r}",
                           hint="firmware predates BLE_STATUS req=?", line=str(st))
        if time.monotonic() >= deadline:
            raise HilError(TIMEOUT, f"{verb}: still pending after the timeout",
                           hint="the LVGL thread runs the job; is the UI stuck? (LVGL_STATS)")
        time.sleep(BLE_JOB_POLL_S)


def ble_start(link: CdcLink, mode: int | None = None) -> dict[str, Any]:
    # BLE_START [0|1] — 0 = Host (central), 1 = Server (peripheral); no arg =
    # the mode in settings. {"deferred": True} = WiFi has the radio, BLE follows it.
    if mode is None:
        return _ok_or_deferred(link, "BLE_START", timeout_s=BLE_START_TIMEOUT_S)
    _check_range("mode", mode, 0, 1)
    return _ok_or_deferred(link, f"BLE_START {mode}", timeout_s=BLE_START_TIMEOUT_S)


def ble_stop(link: CdcLink) -> dict[str, Any]:
    _ok_or_deferred(link, "BLE_STOP", timeout_s=BLE_START_TIMEOUT_S)
    return {"ok": True}


def audio_rate(link: CdcLink, hz: int | None = None) -> dict[str, Any]:
    """Engine sample rate: ``running`` is what the I2S clock does now, ``configured``
    what NVS asks for at the next boot. Passing ``hz`` (44100 or 48000) persists it;
    the clock only follows after a reboot. Kits are 44.1 kHz and the sample engine
    plays them 1:1, so a board left at 48 kHz plays every pad 8.8 % sharp.
    """
    if hz is not None and hz not in (44100, 48000):
        raise HilError(BAD_ARGS, f"audio_rate: {hz} is not 44100 or 48000")
    cmd = "AUDIO_RATE" + (f" {hz}" if hz is not None else "")
    p = _query(link, cmd, "AUDIO_RATE:", "audiorate")
    return {"running": int(p["running"]), "configured": int(p["configured"])}


def ble_enable(link: CdcLink, on: bool) -> dict[str, Any]:
    """Persist wireless.enableBLE and apply it now (BLE_START/STOP do not persist).

    With BLE up this board sits at ~8 kB internal free and a kit load can abort
    it; ``ble_enable(link, False)`` is how a bench puts it in the rc1 shape and
    keeps it there across the reboots a gate causes.
    """
    return _ok_or_deferred(link, f"BLE_ENABLE {1 if on else 0}", timeout_s=BLE_START_TIMEOUT_S)


def ble_scan(link: CdcLink, ms: int = 5000) -> dict[str, Any]:
    _check_range("ms", ms, 1, 65535)           # startScan((uint16_t)a)
    return _ok(link, f"BLE_SCAN {ms}")


def ble_devices(link: CdcLink) -> dict[str, Any]:
    p = _query(link, "BLE_DEVICES", "BLEDEV:", "bledev")
    devices = [{"addr": d["addr"], "name": d.get("name"), "rssi": int(d["rssi"])}
               for d in p.get("devices", [])]
    return {"count": int(p.get("count", len(devices))), "devices": devices}


def ble_connect(link: CdcLink, addr: str) -> dict[str, Any]:
    if not isinstance(addr, str) or not _BLE_ADDR_RE.match(addr):
        raise HilError(BAD_ARGS, f"addr must look like aa:bb:cc:dd:ee:ff, got {addr!r}",
                       hint="BLE_DEVICES lists addresses after BLE_SCAN", addr=addr)
    return _ok(link, f"BLE_CONNECT {addr}", err_code=BAD_ARGS)


def ble_disconnect(link: CdcLink) -> dict[str, Any]:
    return _ok(link, "BLE_DISCONNECT")


def ble_send(link: CdcLink, note: int, vel: int = 100) -> dict[str, Any]:
    _check_range("note", note, 0, 127)
    _check_range("vel", vel, 0, 127)
    return _ok(link, f"BLE_SEND {note} {vel}")


def ble_txoff(link: CdcLink, semis: int) -> dict[str, Any]:
    _check_range("semis", semis, -127, 127)
    return _ok(link, f"BLE_TXOFF {semis}")


# ── network mode + wifi_* ──────────────────────────────────────────────

NET_TABS = ("wifi", "store", "updates")
WIFI_SCAN_TIMEOUT_S = 20.0   # device bound: radio start + scan + one retry, 15 s
WIFI_PING_TIMEOUT_S = 35.0   # device bound: 10 s for the link + 20 s connect and handshake


def net_mode_status(link: CdcLink) -> dict[str, Any]:
    p = _query(link, "NET_MODE", "NET_MODE:", "net_mode")
    return {"active": p.get("active") == 1, "tab": p.get("tab")}


def net_mode_set(link: CdcLink, on: bool, tab: str | None = None) -> dict[str, Any]:
    """Reboot into (or out of) network mode. "OK reboot=1" is followed by a restart
    ~150 ms later and no second reply; "OK unchanged" means it was there already.
    Network mode has no APP_LIST: reopen and poll net_mode_status(), not wait_ready()."""
    if tab is not None and tab not in NET_TABS:
        raise HilError(BAD_ARGS, f"net_mode_set: tab {tab!r} is not one of {', '.join(NET_TABS)}")
    cmd = f"NET_MODE {1 if on else 0}" + (f" TAB={tab}" if tab else "")
    reply = link.transact(cmd, "OK", timeout_s=5.0)
    line = reply.line.strip()
    if line.startswith("ERR"):
        _raise_err("NET_MODE", reply, NOT_SUPPORTED)
    if line not in ("OK reboot=1", "OK unchanged"):
        raise HilError(NOT_SUPPORTED, f"NET_MODE: unparseable reply {reply.line!r}",
                       hint="firmware predates this verb?", line=reply.line)
    return {"ok": True, "reboot": line == "OK reboot=1"}


def _wifi_quote(s: str) -> str:
    # The device's argument parser has no escapes.
    if '"' in s or "\n" in s or "\r" in s:
        raise HilError(BAD_ARGS, "wifi: a double quote or line break cannot be sent in an SSID "
                                 "or password")
    return f'"{s}"'


def wifi_status(link: CdcLink) -> dict[str, Any]:
    return _without_kind(_query(link, "WIFI_STATUS", "WIFI:", "wifi"))


def wifi_enable(link: CdcLink, on: bool) -> dict[str, Any]:
    return _ok(link, f"WIFI_ENABLE {1 if on else 0}")


def wifi_set(link: CdcLink, ssid: str, password: str = "") -> dict[str, Any]:
    return _ok(link, f"WIFI_SET {_wifi_quote(ssid)} {_wifi_quote(password)}")


def wifi_list(link: CdcLink) -> list[str]:
    return list(_query(link, "WIFI_LIST", "WIFI_NET:", "wifi_net")["nets"])


def wifi_forget(link: CdcLink, ssid: str) -> dict[str, Any]:
    return _ok(link, f"WIFI_FORGET {_wifi_quote(ssid)}")


def wifi_up(link: CdcLink) -> dict[str, Any]:
    return _ok(link, "WIFI_UP")


def wifi_down(link: CdcLink) -> dict[str, Any]:
    return _ok(link, "WIFI_DOWN")


def wifi_scan(link: CdcLink) -> list[dict[str, Any]]:
    reply = link.transact("WIFI_SCAN", "WIFI_AP:", timeout_s=WIFI_SCAN_TIMEOUT_S)
    return list(_kind("WIFI_SCAN", reply, "wifi_ap")["aps"])


def wifi_ping(link: CdcLink) -> dict[str, Any]:
    reply = link.transact("WIFI_PING", "WIFI_PING:", timeout_s=WIFI_PING_TIMEOUT_S)
    return _without_kind(_kind("WIFI_PING", reply, "wifi_ping"))


# ── misc ───────────────────────────────────────────────────────────────

def audio_tasks(link: CdcLink, on: bool) -> dict[str, Any]:
    return _ok(link, f"AUDIO_TASKS {1 if on else 0}")


def bootloader_request(link: CdcLink) -> dict[str, Any]:
    # requestBootloader.py CDC_MESSAGE. The device drops into ROM download mode
    # and never answers; the CDC port disappears and re-enumerates as
    # 0x303A:{0x1001,0x0009}.
    link.send("BOOTLOADER_REQUEST")
    return {"sent": True}


def pwr_gesture(link: CdcLink, hold: bool = False) -> dict[str, Any]:
    """Inject the power button's short click (back) or half-second hold (drawer).

    The button itself cannot be pressed remotely; this enters where the STM
    register poll would have, so the gesture resolves through exactly the code
    a finger reaches (crosspad_gui::powerBack / powerToggleQuickSettings).
    """
    return _ok(link, "PWR_GESTURE " + ("HOLD" if hold else "CLICK"))


def coredump_info(link: CdcLink) -> dict[str, Any]:
    """What the coredump partition holds: {'present': bool, 'seq': int|None, ...}.

    Polled rather than parsed from the console, because a panic that reboots
    the board faster than the console is read still leaves this behind.
    """
    lines = link.transact_multi("COREDUMP_INFO", "COREDUMP:", end=None)
    if not lines:
        raise HilError(NOT_SUPPORTED, "COREDUMP_INFO: no reply",
                       hint="firmware predates the reset black box?")
    head = lines[0]
    out: dict[str, Any] = {"present": "none" not in head, "line": head,
                           "backtrace": next((ln for ln in lines[1:] if ln.startswith("bt:")), "")}
    for key in ("seq", "task", "pc", "cause"):
        m = re.search(rf"\b{key}=(\S+)", head)
        if m:
            out[key] = int(m.group(1)) if key in ("seq", "cause") else m.group(1)
    return out


def stm_dfu(link: CdcLink) -> dict[str, Any]:
    _ok(link, "STM_DFU")
    return {"sent": True}


def stm_reg(link: CdcLink, addr: int, length: int = 1) -> dict[str, Any]:
    """Raw STM32 register-map read. `STM_REG 0x0006 2` is the live firmware
    version — STM_FW_INFO's running= is a cache the ESP fills at its own boot."""
    if not 0 <= addr <= 0xFFFF or not 1 <= length <= 64:
        raise HilError(BAD_ARGS, f"STM_REG: addr 0..0xFFFF and len 1..64, got {addr:#x}/{length}",
                       addr=addr, length=length)
    reply = link.transact(f"STM_REG {addr:04X} {length}", "REG")
    body = reply.line.split(":", 1)[-1]
    try:
        data = bytes(int(tok, 16) for tok in body.split())
    except ValueError as exc:
        raise HilError(NOT_SUPPORTED, f"STM_REG: unparseable reply {reply.line!r}",
                       line=reply.line) from exc
    if len(data) != length:
        raise HilError(NOT_SUPPORTED, f"STM_REG: asked {length} bytes, got {len(data)}",
                       line=reply.line)
    return {"addr": addr, "len": length, "bytes": list(data), "hex": data.hex()}


def stm_fw_info(link: CdcLink) -> dict[str, Any]:
    return _without_kind(_query(link, "STM_FW_INFO", "STM_FW", "stm_fw"))


# ── firmware library / A/B switch (main/fw_slots.cpp, main/fw_library.cpp) ──
# docs/subsystems/firmware-library.md is the authority on the wire format and
# every trap below.

_FW_SLOT_RE = re.compile(
    r"slot=(\S)\s+running=(\d)\s+state=(\S+)\s+pkg=(\S+)\s+ver=(\S+)\s+kit=(\S+)$")
_FW_SLOTS_TRAILER_RE = re.compile(r"rollback=(\d)$")
_FWPKG_RE = re.compile(
    r"name=(\S+)\s+kit=(\S+)\s+ver=(\S+)\s+rev=(\d+)\s+size=(\d+)\s+assets=(\d)\s+"
    r"settings=(\d)\s+verdict=(.*)$")
_FWPKG_COUNT_RE = re.compile(r"count=(\d+)$")
_FWINST_RE = re.compile(r"phase=(\S+)\s+percent=(\d+)\s+error=(.*)$")


def fw_slots(link: CdcLink) -> dict[str, Any]:
    """Both OTA slots and whether the flashed bootloader can roll one back.

    Two rows ("FWSLOT: slot=A|B running=0|1 state=empty|installed|trial|
    confirmed|rolledback pkg=<name>|- ver=<v>|- kit=<k>|-"), then the trailer
    ("FWSLOT: rollback=0|1") that ends the reply -- the end marker this call
    needs to read all three lines instead of only the first.
    """
    lines = link.transact_multi("FW_SLOTS", "FWSLOT:", end="FWSLOT: rollback")
    slots: list[dict[str, Any]] = []
    rollback: bool | None = None
    for line in lines:
        body = line.split(":", 1)[1].strip()
        m = _FW_SLOT_RE.match(body)
        if m:
            slot, running, state, pkg, ver, kit = m.groups()
            slots.append({
                "slot": slot, "running": running == "1", "state": state,
                "pkg": None if pkg == "-" else pkg,
                "ver": None if ver == "-" else ver,
                "kit": None if kit == "-" else kit,
            })
            continue
        m = _FW_SLOTS_TRAILER_RE.match(body)
        if m:
            rollback = m.group(1) == "1"
    if rollback is None or len(slots) != 2:
        raise HilError(NOT_SUPPORTED, "FW_SLOTS: unparseable reply",
                       hint="firmware predates this verb?", lines=lines)
    return {"slots": slots, "rollback": rollback}


def fw_list(link: CdcLink, *, timeout_s: float = 15.0) -> dict[str, Any]:
    """Packages in the SD card's firmware library (main/fw_library.cpp scan).

    One row per package; ``verdict`` is ``"ok"`` when FW_INSTALL would accept
    it, else the reason it would refuse ("built for another board revision",
    …). The scan walks up to 16 images off the same card the sample engine
    streams from, so it needs longer than a plain query -- 15 s by default.

    A busy scan or a task that could not be created answers "FWPKG: ERR busy"
    / "FWPKG: ERR no task" *with* the FWPKG: prefix, unlike KIT_LIST's bare
    "ERR scanning" -- that means this call is not told apart from a still-
    running scan until the full timeout elapses, because nothing else ever
    arrives to end the wait early.
    """
    lines = link.transact_multi("FW_LIST", "FWPKG:", end="FWPKG: count", timeout_s=timeout_s)
    if len(lines) == 1:
        body = lines[0].split(":", 1)[1].strip()
        if body.startswith("ERR"):
            raise HilError(NOT_SUPPORTED, f"FW_LIST: device replied {lines[0]}",
                           hint="the scan is busy or could not start a task; retry",
                           reply=lines[0])
    packages: list[dict[str, Any]] = []
    count: int | None = None
    for line in lines:
        body = line.split(":", 1)[1].strip()
        m = _FWPKG_COUNT_RE.match(body)
        if m:
            count = int(m.group(1))
            continue
        m = _FWPKG_RE.match(body)
        if not m:
            continue
        name, kit, ver, rev, size, assets, settings, verdict = m.groups()
        packages.append({
            "name": name, "kit": kit, "ver": ver, "rev": int(rev), "size": int(size),
            "assets": assets == "1", "settings": settings == "1", "verdict": verdict,
        })
    if count is not None and count != len(packages):
        raise HilError(NOT_SUPPORTED,
                       f"FW_LIST: trailer says {count} packages, {len(packages)} arrived",
                       hint="a dropped line, not a short card", got=len(packages), want=count)
    return {"packages": packages, "count": count if count is not None else len(packages)}


def fw_install(link: CdcLink, name: str, *, force: bool = False) -> dict[str, Any]:
    """Start installing ``name`` into the non-running slot (asynchronous).

    "OK" (installed clean) or "OK warn=stm_proto_older" (installed, but the
    package expects an older STM32 protocol than what is currently running);
    an ERR means the device refused (another install running, no such
    package, its own verdict, battery low, or mid kit-load) and nothing
    changed. Poll :func:`fw_install_status` for the write itself.
    """
    if not isinstance(name, str) or not name or any(ch.isspace() for ch in name):
        raise HilError(BAD_ARGS, f"name must be a non-empty string with no whitespace, "
                       f"got {name!r}", hint="FW_LIST names the packages", name=name)
    cmd = f"FW_INSTALL {name}" + (" FORCE" if force else "")
    reply = link.transact(cmd, "OK", timeout_s=5.0)
    line = reply.line.strip()
    if line.startswith("ERR"):
        _raise_err("FW_INSTALL", reply, NOT_SUPPORTED)
    if line not in ("OK", "OK warn=stm_proto_older"):
        raise HilError(NOT_SUPPORTED, f"FW_INSTALL: unparseable reply {reply.line!r}",
                       hint="firmware predates this verb?", line=reply.line)
    return {"ok": True, "warn_stm_proto_older": line == "OK warn=stm_proto_older"}


def fw_install_status(link: CdcLink) -> dict[str, Any]:
    """Poll the running (or last) install: phase/percent/error.

    Bare ``FW_INSTALL`` -- the same verb as :func:`fw_install`, sent with no
    name, always answers the status line instead of starting anything.
    """
    reply = link.transact("FW_INSTALL", "FWINST:", timeout_s=2.0)
    m = _FWINST_RE.match(reply.line.split(":", 1)[1].strip())
    if not m:
        raise HilError(NOT_SUPPORTED, f"FW_INSTALL: unparseable status {reply.line!r}",
                       hint="firmware predates this verb?", line=reply.line)
    phase, percent, error = m.groups()
    return {"phase": phase, "percent": int(percent), "error": None if error == "-" else error}


def fw_switch(link: CdcLink) -> dict[str, Any]:
    """Ask the running slot to hand the boot partition to the other one.

    Success answers "OK reboot=1" and the board restarts ~200 ms later -- no
    second reply arrives, so the caller must reopen the link once the device
    is back (``ctx.open_cdc()`` then ``link.wait_ready()``, the same shape
    every other reboot-then-reopen spot in this package uses). A refusal
    ("ERR running slot not confirmed yet", "ERR other slot did not start
    last time", "ERR busy", "ERR no task", …) means nothing happened.
    """
    reply = link.transact("FW_SWITCH", "OK", timeout_s=5.0)
    line = reply.line.strip()
    if line.startswith("ERR"):
        _raise_err("FW_SWITCH", reply, NOT_SUPPORTED)
    if line != "OK reboot=1":
        raise HilError(NOT_SUPPORTED, f"FW_SWITCH: unparseable reply {reply.line!r}",
                       hint="firmware predates this verb?", line=reply.line)
    return {"ok": True, "reboot": True}


def fw_confirm(link: CdcLink) -> dict[str, Any]:
    """Mark the running (still-trial) image valid now instead of waiting for
    the 10 s timer. Fire-and-forget: "OK" only proves the command arrived,
    not that the confirm task's write has landed yet -- poll fw_slots()."""
    return _ok(link, "FW_CONFIRM")


def app_sha(link: CdcLink) -> dict[str, Any]:
    """The running image's ``app_elf_sha256``, hex -- compare with
    ``sha256sum build_<rev>/CrossPad.elf`` to prove a claimed "flashed"
    actually replaced the binary (LIBVER/commit alone can read identical
    across two builds of the same commit, which is exactly what a silent
    rollback inside a trial window looks like)."""
    reply = link.transact("APP_SHA", "APPSHA:", timeout_s=2.0)
    sha = reply.line.split(":", 1)[1].strip()
    if not re.fullmatch(r"[0-9a-f]{64}", sha):
        raise HilError(NOT_SUPPORTED, f"APP_SHA: unparseable reply {reply.line!r}",
                       hint="firmware predates this verb?", line=reply.line)
    return {"sha256": sha}


# ── firmware updates (platform-idf main/fw_updater*.cpp) ──────────────────
# FW_CHECK / FW_DOWNLOAD only start a job on the board (normal boot or network
# mode); progress and the outcome are read with FW_UPD_STATUS. FW_CATALOG lists
# this boot's last fetch. docs/subsystems/firmware-updates.md

# The device's token buffers (fw_updater_verbs.cpp take_token, refused when a
# token does not fit): a kit id is at most 15 characters (FWU_KIT_LEN 16), a
# version at most 23 (kFwVersionMax 24) plus an optional leading 'v'.
_FW_TOKEN_MAX = {"kit": 15, "version": 24}


def _fw_token(what: str, value: str) -> str:
    cap = _FW_TOKEN_MAX[what]
    if not value or len(value) > cap or any(c.isspace() for c in value):
        raise HilError(BAD_ARGS, f"{what} {value!r}: one word, at most {cap} characters")
    return value


def fw_upd_status(link: CdcLink) -> dict[str, Any]:
    return _without_kind(_query(link, "FW_UPD_STATUS", "FWUPD:", "fwupd"))


def fw_check(link: CdcLink) -> dict[str, Any]:
    """Start a catalog check: ERR busy while a job runs, ERR no_mem when the job task cannot
    start. The job takes the WiFi lease itself; a refused one is FW_UPD_STATUS state=failed
    err=disabled / transfer_mode / no_mem (platform-idf PR #47; older images answered ERR)."""
    return _ok(link, "FW_CHECK")


def fw_catalog(link: CdcLink, kit: str | None = None) -> dict[str, Any]:
    """The last fetched catalog, one line capped at 448 B: count=rows shown, total=rows matching."""
    cmd = f"FW_CATALOG {_fw_token('kit', kit)}" if kit else "FW_CATALOG"
    return _without_kind(_query(link, cmd, "FWCAT:", "fwcat"))


def fw_download(link: CdcLink, kit: str, version: str, *, install: bool = False) -> dict[str, Any]:
    cmd = f"FW_DOWNLOAD {_fw_token('kit', kit)} {_fw_token('version', version)}"
    return _ok(link, cmd + (" INSTALL" if install else ""))


def fw_upd_cfg(link: CdcLink, *, boot: bool | None = None,
               gc_mb: int | None = None) -> dict[str, Any]:
    """boot = Check at startup (persisted); gc_mb = the library limit in MB (0 = the
    default 512), RAM only: gone at the next boot."""
    parts = []
    if boot is not None:
        parts.append(f"boot={1 if boot else 0}")
    if gc_mb is not None:
        if int(gc_mb) < 0:
            raise HilError(BAD_ARGS, f"fw_upd_cfg: gc_mb {gc_mb} is negative")
        parts.append(f"gc_mb={int(gc_mb)}")
    if not parts:
        raise HilError(BAD_ARGS, "fw_upd_cfg: nothing to set")
    return _ok(link, "FW_UPD_CFG " + " ".join(parts))


def fw_upd_channel(link: CdcLink, kit: str, channel: str) -> dict[str, Any]:
    if channel not in ("stable", "beta"):
        raise HilError(BAD_ARGS, f"channel {channel!r}: stable or beta")
    return _ok(link, f"FW_UPD_CHANNEL {_fw_token('kit', kit)} {channel}")


def fw_upd_url(link: CdcLink, url: str | None) -> dict[str, Any]:
    """Point the board's updater at `url` (persisted); None goes back to crosspad.app."""
    if url is not None and (not url or any(c.isspace() for c in url)):
        raise HilError(BAD_ARGS, f"fw_upd_url {url!r}: a URL has no whitespace")
    return _ok(link, f"FW_UPD_URL {url or 'DEFAULT'}")


def fw_ui(link: CdcLink) -> dict[str, Any]:
    """What the Firmware views show: tab, selected tile, open kit card, item
    count, which host draws them (app / net), the card cache's state and
    whether the view worker still has work (icons, scans) queued."""
    return _without_kind(_query(link, "FW_UI", "FWUI:", "fwui"))


def usb_msc_status(link: CdcLink) -> dict[str, Any]:
    """Bare ``USB_MSC`` query: whether the SD-as-USB-drive boot mode is
    active, card presence/state and read/write counters. Setting the mode
    (``USB_MSC 0|1``) reboots the board into a mode with no CDC endpoint and
    is not wrapped here -- no HIL scenario should ever send it."""
    reply = link.transact("USB_MSC", "USB_MSC:", timeout_s=2.0)
    body = reply.line.split(":", 1)[1].strip()
    fields: dict[str, str] = {}
    for tok in body.split():
        key, _, value = tok.partition("=")
        if value:
            fields[key] = value
    if "active" not in fields:
        raise HilError(NOT_SUPPORTED, f"USB_MSC: unparseable reply {reply.line!r}",
                       hint="firmware predates this verb?", line=reply.line)
    return {
        "active": fields["active"] == "1",
        "card": fields.get("card") == "1",
        "state": fields.get("state"),
        "size_mb": int(fields["size_mb"]) if fields.get("size_mb", "").isdigit() else None,
        "read_kb": int(fields["read_kb"]) if fields.get("read_kb", "").isdigit() else None,
        "write_kb": int(fields["write_kb"]) if fields.get("write_kb", "").isdigit() else None,
    }


def usb_guard(link: CdcLink, on: bool | None = None) -> bool:
    """The USB serial guard (Allow/Deny before the CDC profile), after an optional
    set. Setting it PERSISTS (usbSerialGuard); bare ``USB_GUARD`` only reads."""
    cmd = "USB_GUARD" if on is None else f"USB_GUARD {1 if on else 0}"
    reply = link.transact(cmd, "USB_GUARD:", timeout_s=2.0)
    m = re.match(r"USB_GUARD:\s*([01])\s*$", reply.line.strip())
    if m is None:
        raise HilError(NOT_SUPPORTED, f"USB_GUARD: unparseable reply {reply.line!r}",
                       hint="firmware predates this verb?", line=reply.line)
    return m.group(1) == "1"


# ── filesystem (main/file_transfer.cpp) ────────────────────────────────
# The streaming halves (ASSETS_BEGIN / FS_PUT / FS_GET) live in crosspad_hil.fs:
# they take the pipe out of newline framing, which transact() cannot express.

def fs_roots(link: CdcLink) -> dict[str, Any]:
    """{mount: {"total": int|None, "used": int|None}} — None where not mounted."""
    roots: dict[str, Any] = {}
    for line in link.transact_multi("FS_ROOTS", "FS_ROOTS:", end="FS_END"):
        parts = line.split()
        if len(parts) != 4:
            continue
        _, mount, total, used = parts
        roots[mount] = {
            "total": None if total == "-" else int(total),
            "used": None if used == "-" else int(used),
        }
    return {"roots": roots}


def fs_list(link: CdcLink, path: str, timeout_s: float = 10.0) -> dict[str, Any]:
    entries: list[dict[str, Any]] = []
    # An empty directory is its end line alone: no FS_LIST: row to start from. A card
    # busy with a bake answers a listing of fifty songs in seconds.
    for line in link.transact_multi(f"FS_LIST {path}", ("FS_LIST:", "FS_END"), end="FS_END",
                                    timeout_s=timeout_s, require_end=True):
        parts = line.split(" ", 3)
        if len(parts) != 4:
            continue
        _, kind, size, name = parts
        entries.append({"name": name, "dir": kind == "D", "size": int(size)})
    return {"path": path, "entries": entries}


def fs_stat(link: CdcLink, path: str) -> dict[str, Any]:
    reply = link.transact(f"FS_STAT {path}", "FS_STAT")
    parts = reply.line.split()
    if len(parts) != 3:
        raise HilError(NOT_SUPPORTED, f"FS_STAT: unparseable reply {reply.line!r}",
                       line=reply.line)
    return {"path": path, "size": int(parts[1]), "dir": parts[2] == "D"}


def fs_mkdir(link: CdcLink, path: str) -> dict[str, Any]:
    _ok(link, f"FS_MKDIR {path}", err_code=BAD_ARGS)
    return {"path": path}


def fs_delete(link: CdcLink, path: str) -> dict[str, Any]:
    _ok(link, f"FS_DELETE {path}", err_code=BAD_ARGS)
    return {"path": path}


VERBS: dict[str, Callable[..., dict[str, Any]]] = {
    "app_list": app_list, "app_start": app_start, "app_stop": app_stop,
    "app_destroy": app_destroy, "app_self_close": app_self_close,
    "app_versions": app_versions,
    "kit_list": kit_list, "kit_status": kit_status, "kit_load": kit_load,
    "kit_swap_log": kit_swap_log,
    "wave_status": wave_status,
    "pad_press": pad_press, "pad_release": pad_release, "pad_pressure": pad_pressure,
    "pad_pan": pad_pan,
    "pad_stats": pad_stats, "pad_notes": pad_notes, "pad_info": pad_info,
    "pad_mix": pad_mix, "pad_mix_set": pad_mix_set, "kit_save": kit_save,
    "enc_rotate": enc_rotate, "enc_press": enc_press, "enc_group": enc_group,
    "enc_focus": enc_focus, "enc_state": enc_state, "ui_state": ui_state,
    "led_state": led_state, "mem": mem, "mem_blocks": mem_blocks, "clock_status": clock_status,
    "seq": seq, "song": song, "asong": asong, "reset_count": reset_count,
    "settings_reset": settings_reset, "settings_export": settings_export,
    "settings_import": settings_import, "settings_status": settings_status,
    "imu": imu, "imu_midi": imu_midi, "imu_gesture": imu_gesture, "touch_trace": touch_trace,
    "worker": worker, "ulp": ulp, "imu_ths": imu_ths, "heap_trace": heap_trace,
    "cdc_stats": cdc_stats, "lvgl_stats": lvgl_stats, "audio_level": audio_level,
    "smpl_peak": smpl_peak, "smpl_counters": smpl_counters,
    "smpl_engine": smpl_engine, "smpl_engine_select": smpl_engine_select, "smpl_pitch": smpl_pitch,
    "ingain": ingain,
    "daw_state": daw_state,
    "pitched_status": pitched_status, "pitched_bench": pitched_bench,
    "ble_status": ble_status, "ble_start": ble_start, "ble_stop": ble_stop,
    "ble_enable": ble_enable,
    "ble_scan": ble_scan, "ble_devices": ble_devices, "ble_connect": ble_connect,
    "ble_disconnect": ble_disconnect, "ble_send": ble_send, "ble_txoff": ble_txoff,
    "net_mode_status": net_mode_status, "net_mode_set": net_mode_set,
    "wifi_status": wifi_status, "wifi_enable": wifi_enable, "wifi_set": wifi_set,
    "wifi_list": wifi_list, "wifi_forget": wifi_forget, "wifi_up": wifi_up,
    "wifi_down": wifi_down, "wifi_scan": wifi_scan, "wifi_ping": wifi_ping,
    "audio_tasks": audio_tasks, "audio_rate": audio_rate, "bootloader_request": bootloader_request,
    "stm_dfu": stm_dfu, "pwr_gesture": pwr_gesture, "coredump_info": coredump_info,
    "stm_reg": stm_reg, "stm_fw_info": stm_fw_info,
    "fs_roots": fs_roots, "fs_list": fs_list, "fs_stat": fs_stat,
    "fs_mkdir": fs_mkdir, "fs_delete": fs_delete,
    "fw_slots": fw_slots, "fw_list": fw_list, "fw_install": fw_install,
    "fw_install_status": fw_install_status, "fw_switch": fw_switch, "fw_confirm": fw_confirm,
    "app_sha": app_sha, "usb_msc_status": usb_msc_status, "usb_guard": usb_guard,
    "ota_prefs": ota_prefs,
    "fw_upd_status": fw_upd_status, "fw_check": fw_check, "fw_catalog": fw_catalog,
    "fw_download": fw_download, "fw_upd_cfg": fw_upd_cfg, "fw_upd_channel": fw_upd_channel,
    "fw_upd_url": fw_upd_url, "fw_ui": fw_ui,
}


# ── live_jam: the sequencer's grid, the mixer desk, the engines' pools ──────
# Replies parsed here as plain key=value: none of these is in knowledge/cdc.yaml's
# typed parsers, and every one is evidence a scenario keeps verbatim beside the
# parsed dict ("line").

def _num(value: str) -> Any:
    try:
        return int(value)
    except ValueError:
        pass
    try:
        return float(value)
    except ValueError:
        return value


def _kv_line(line: str, prefix: str | None = None) -> dict[str, Any]:
    """``key=value`` tokens of one reply line (ints and floats promoted)."""
    body = line[len(prefix):] if prefix and line.startswith(prefix) else line
    out: dict[str, Any] = {}
    for tok in body.split():
        key, sep, value = tok.partition("=")
        if sep and key:
            out[key] = _num(value)
    return out


def _ok_any(link: CdcLink, cmd: str, err_code: str = BAD_ARGS) -> dict[str, Any]:
    """An OK/ERR verb that knowledge/cdc.yaml does not list (MIX_ROUTE, PITCHED_CAP)."""
    verb = cmd.split(" ", 1)[0]
    reply = link.transact(cmd, "OK")
    parsed = reply.parsed or {}
    if parsed.get("kind") == "err" or (reply.line or "").startswith("ERR"):
        _raise_err(verb, reply, err_code)
    return {"ok": True}


def _line(link: CdcLink, cmd: str, prefix: str, timeout_s: float | None = None) -> str:
    reply = link.transact(cmd, prefix, timeout_s=timeout_s)
    line = (reply.line or "").strip()
    if line.startswith("ERR") or (reply.parsed or {}).get("kind") == "err":
        raise HilError(NOT_SUPPORTED, f"{cmd.split(' ', 1)[0]}: device replied {line}",
                       hint="firmware predates this verb, or the arguments are wrong",
                       reply=line)
    if not line.startswith(prefix):
        raise HilError(NOT_SUPPORTED, f"{cmd.split(' ', 1)[0]}: unparseable reply",
                       hint="firmware predates this verb?", line=line)
    return line


def asong_dbg(link: CdcLink) -> dict[str, Any]:
    """``ASONG_DBG``: the song engine's debug line. ``seg`` is the segments per track and
    ``ctx`` the context stem's (segments, descriptor open, solo), each as a list of ints;
    the rest are the line's own numbers (``ringfill``, ``play``, ``job`` ...)."""
    line = _line(link, "ASONG_DBG", "ASONGDBG:")
    kv = _kv_line(line, "ASONGDBG:")
    for key in ("seg", "ctx"):
        if key in kv:
            kv[key] = [int(n) for n in str(kv[key]).split("/")]
    kv["line"] = line
    return kv


def outvol(link: CdcLink, out: int | None = None, pct: int | None = None) -> dict[str, Any]:
    """``OUTVOL``: bare reads ``{out1, out2, spk_amp}`` (the saved settings), ``OUTVOL out pct``
    sets that codec's level at once (out 1 = line out, out 2 = phones and speakers)."""
    if out is None:
        line = _line(link, "OUTVOL", "OUTVOL:")
        head = line.partition("(")[0]
        kv = _kv_line(head, "OUTVOL:")
        kv["line"] = line
        return kv
    _ok_any(link, f"OUTVOL {out} {pct}")
    return {"out": out, "pct": pct}


def seq_dump(link: CdcLink, timeout_s: float = 5.0) -> list[dict[str, int]]:
    """Every recorded event of every scene: ``[{seq, i, pos, pad, vel}]``, positions in
    samples from the loop start, already quantised (``SEQ_DUMP``)."""
    lines = link.transact_multi("SEQ_DUMP", "SEQDUMP:", end="SEQDUMP: end",
                                timeout_s=timeout_s)
    out: list[dict[str, int]] = []
    for line in lines:
        kv = _kv_line(line.strip(), "SEQDUMP:")
        if {"seq", "i", "pos", "pad", "vel"} <= set(kv):
            out.append({k: int(kv[k]) for k in ("seq", "i", "pos", "pad", "vel")})
    return out


def click(link: CdcLink, on: int | None = None, bpm: int | None = None,
          bars: int | None = None) -> dict[str, Any]:
    """The metronome -- the sequencer's clock: running, bpm, bar/beat and
    ``pos``/``loop`` in samples. Arguments set it first (``CLICK on [bpm [bars]]``)."""
    args = [str(int(v)) for v in (on, bpm, bars) if v is not None]
    line = _line(link, " ".join(["CLICK", *args]), "CLICK:")
    kv = _kv_line(line, "CLICK:")
    pos = str(kv.get("pos", ""))
    if "/" in pos:
        a, _, b = pos.partition("/")
        kv["pos"], kv["loop"] = int(a), int(b)
    kv["line"] = line
    return kv


def mix_lvl(link: CdcLink) -> dict[str, Any]:
    """The mixer desk: ``{"channels": [{ch, name, pk_l, pk_r, vol, mute, route:[0|1..]}],
    "outputs": [{out, pk_l, pk_r, vol, mute}]}`` (``MIX_LVL``)."""
    lines = link.transact_multi("MIX_LVL", "MIXLVL:", end="MIXLVL: end")
    chans: list[dict[str, Any]] = []
    outs: list[dict[str, Any]] = []
    for line in lines:
        kv = _kv_line(line.strip(), "MIXLVL:")
        pk = str(kv.pop("pk", "0/0"))
        a, _, b = pk.partition("/")
        kv["pk_l"], kv["pk_r"] = _num(a), _num(b or "0")
        if "ch" in kv:
            # Read off the line, not out of kv: _num turns "011" into 11 and "000" into 0,
            # and a send on the first outputs is exactly what a leading zero says is off.
            m = re.search(r"\broute=([01]*)", line)
            kv["route"] = [int(c) for c in (m.group(1) if m else "")]
            chans.append(kv)
        elif "out" in kv:
            outs.append(kv)
    return {"channels": chans, "outputs": outs}


def mix_route(link: CdcLink, ch: int, out: int, on: bool) -> dict[str, Any]:
    """One send on or off (``MIX_ROUTE ch out 0|1``), the way the Mixer's ROUTE sets it."""
    _check_range("ch", ch, 0, 31)
    _check_range("out", out, 0, 7)
    return _ok_any(link, f"MIX_ROUTE {ch} {out} {1 if on else 0}")


def mix_mute(link: CdcLink, ch: int, on: bool) -> dict[str, Any]:
    """A mixer channel's mute (``MIX_MUTE ch 0|1``, MIX_LVL numbering), as the Mixer's M
    button sets it; the desk is saved, so it survives a reboot."""
    _check_range("ch", ch, 0, 31)
    return _ok_any(link, f"MIX_MUTE {ch} {1 if on else 0}")


def mix_vol(link: CdcLink, ch: int, pct: int) -> dict[str, Any]:
    """A mixer channel's level (``MIX_VOL ch pct``): linear percent, 100 = unity
    (``MIX_LVL vol=1.00``), as the Mixer's fader sets it; saved."""
    _check_range("ch", ch, 0, 31)
    _check_range("pct", pct, 0, 100)
    return _ok_any(link, f"MIX_VOL {ch} {pct}")


def mix_out_mute(link: CdcLink, out: int, on: bool) -> dict[str, Any]:
    """A mixer output's mute (``MIX_OUT_MUTE out 0|1``); saved."""
    _check_range("out", out, 0, 7)
    return _ok_any(link, f"MIX_OUT_MUTE {out} {1 if on else 0}")


def mix_out_vol(link: CdcLink, out: int, pct: int) -> dict[str, Any]:
    """A mixer output's level (``MIX_OUT_VOL out pct``), linear percent 0..100; saved."""
    _check_range("out", out, 0, 7)
    _check_range("pct", pct, 0, 100)
    return _ok_any(link, f"MIX_OUT_VOL {out} {pct}")


_MIXPRE_SLOT = re.compile(r"\b([1-6])=([sfe]):(.*?)(?= [1-6]=[sfe]:|$)")


def mix_preset(link: CdcLink, action: str | None = None) -> dict[str, Any]:
    """The Mixer's presets: ``action`` ("LOAD n", "SAVE n name", "UNDO") first, then
    ``{ok, undo, dir, slots: {n: {kind: saved|factory|empty, name}}}``.

    LOAD and UNDO also write the desk to NVS (the mixer's state callback): the
    board keeps what was loaded across a reboot.
    """
    line = _line(link, f"MIX_PRESET {action}" if action else "MIX_PRESET", "MIXPRE:")
    head = _kv_line(line.split(" 1=", 1)[0], "MIXPRE:")
    kinds = {"s": "saved", "f": "factory", "e": "empty"}
    slots = {int(m.group(1)): {"kind": kinds[m.group(2)], "name": m.group(3).strip()}
             for m in _MIXPRE_SLOT.finditer(line)}
    return {"ok": int(head.get("ok", 0)) == 1, "undo": int(head.get("undo", 0)) == 1,
            "dir": head.get("dir"), "slots": slots, "line": line}


def smpl_last(link: CdcLink) -> dict[str, Any]:
    """``SMPL_LAST``: the engine's last note events and the streamer's counters.
    **Read-clear**: late_max_ms, late_lost, stream_read_max_us, win_max_us restart
    at every read -- read it at block boundaries only."""
    line = _line(link, "SMPL_LAST", "SMPLLAST:")
    kv = _kv_line(line, "SMPLLAST:")
    events = str(kv.get("events", ""))
    kv["events"] = [e for e in events.split(",") if e]
    kv["line"] = line
    return kv


def smpl_pool(link: CdcLink) -> dict[str, Any]:
    """``SMPL_POOL``: the prebuilt engine's stream-block pool -- how close the voices
    came to running out. NOT_SUPPORTED (``ERR only_native``) on an image without that
    engine; the native engine's pool is in :func:`smpl_engine`."""
    line = _line(link, "SMPL_POOL", "SMPLPOOL:")
    kv = _kv_line(line, "SMPLPOOL:")
    for key in ("busy", "used"):
        a, sep, b = str(kv.get(key, "")).partition("/")
        if sep:
            kv[key], kv[f"{key}_max"] = _num(a), _num(b)
    kv["line"] = line
    return kv


def smpl_poly(link: CdcLink, total: int | None = None,
              streaming: int | None = None) -> dict[str, Any]:
    """The native engine's voice caps (``SMPL_POLY [total [streaming]]``): bare reports,
    ``total`` sets the cap on sounding voices, ``streaming`` the cap on voices fed off
    the card (1..28 each; both last until the next kit take). ``{max_voices,
    max_streaming}``; 0/0 = no kit taken.

    The prebuilt engine's ``SMPL_POLY n`` (0..17 WAV channels, answers OK) exists only
    in an image built with it, while it holds the kit; this helper does not drive it."""
    cmd = "SMPL_POLY"
    if total is not None:
        _check_range("total", total, 1, 28)
        cmd += f" {total}"
        if streaming is not None:
            _check_range("streaming", streaming, 1, 28)
            cmd += f" {streaming}"
    elif streaming is not None:
        raise HilError(BAD_ARGS, "streaming needs total", hint="SMPL_POLY <total> <streaming>")
    # The streaming cap takes the engine's locks for 20 ms at most and answers
    # "ERR busy" while a kit take holds them -- the CDC task waits for nobody --
    # so a set right after a kit load is retried here until the take is done.
    deadline = time.monotonic() + SMPL_POLY_BUSY_RETRY_S
    while True:
        try:
            return _without_kind(_query(link, cmd, "SMPLPOLY:", "smplpoly"))
        except HilError as exc:
            if "busy" not in exc.message or time.monotonic() >= deadline:
                raise
            time.sleep(0.25)


SMPL_POLY_BUSY_RETRY_S = 15.0


def pitched_cap(link: CdcLink, n: int) -> dict[str, Any]:
    """The pitched engine's voice cap (``PITCHED_CAP n``), from the next load. Until reboot."""
    _check_range("n", n, 1, 28)
    return _ok_any(link, f"PITCHED_CAP {n}")


def arena(link: CdcLink) -> dict[str, Any]:
    """``ARENA``: the PSRAM block the sample engines take turns owning:
    ``{cap_kb, used_kb, owner}``, owner native|marcel|pitched|none."""
    return _kv_line(_line(link, "ARENA", "ARENA:"), "ARENA:")


def stm_vpad(link: CdcLink, rate_hz: int | None = None, count: int | None = None,
             hold_ms: int = 100, mode: int = 0) -> dict[str, Any]:
    """STM-side virtual pads on the physical pad path; bare = status, ``rate_hz=0`` = STOP."""
    if rate_hz == 0:
        return _ok(link, "STM_VPAD STOP")
    cmd = "STM_VPAD" if rate_hz is None else f"STM_VPAD {rate_hz} {count or 1} {hold_ms} {mode}"
    return _kv_line(_line(link, cmd, "VPAD:"), "VPAD:")


def midi_trs(link: CdcLink, action: str | None = None) -> dict[str, Any]:
    """The STM's TRS MIDI pair (STM 1.14): ``action`` ("TYPE A", "SKEW -35",
    "FLIP 2 1000", "RESET", "ROUTE 4E 04", "ROUTE SAVED") first, then the state
    and the per-segment counters. ``type`` is ``?`` on an STM without it."""
    line = _line(link, f"MIDI_TRS {action}" if action else "MIDI_TRS", "MIDI_TRS:")
    return _kv_line(line, "MIDI_TRS:")


def midi_loop(link: CdcLink, action: str | None = None) -> dict[str, Any]:
    """The ESP's end of the MIDI loop test: ``action`` ("CAPTURE 1", "SEND 2000 500",
    "STOP") first, then its counters."""
    line = _line(link, f"MIDI_LOOP {action}" if action else "MIDI_LOOP", "MIDI_LOOP:")
    return _kv_line(line, "MIDI_LOOP:")


def stool(link: CdcLink, action: str | None = None, timeout_s: float = 5.0) -> dict[str, Any]:
    """The sample tools (Recorder): ``action`` ("IN PADS", "REC", "STOP", ...) first, then
    the report. Names come last on the line (``file=``, ``err=``)."""
    line = _line(link, f"STOOL {action}" if action else "STOOL", "STOOL:", timeout_s)
    head, _, tail = line.partition(" n=")
    kv = _kv_line(head, "STOOL:")
    for key in ("file", "err", "result", "first", "dir"):
        m = re.search(rf"\b{key}=(\S*)", tail)
        if m:
            kv[key] = m.group(1)
    kv["line"] = line
    return kv


def song_text(link: CdcLink, action: str | None = None) -> dict[str, Any]:
    """``SONG`` as key=value plus ``blocks`` [(scene, repeats)], without the typed parser."""
    line = _line(link, f"SONG {action}" if action else "SONG", "SONG:")
    head, _, blocks = line.partition("blocks=")
    kv = _kv_line(head, "SONG:")
    kv["blocks"] = [[int(x) for x in b.split(":")] for b in blocks.strip().split(",")
                    if ":" in b]
    kv["line"] = line
    return kv


VERBS.update({
    "seq_dump": seq_dump, "click": click, "mix_lvl": mix_lvl, "mix_route": mix_route,
    "mix_mute": mix_mute, "mix_vol": mix_vol, "mix_out_mute": mix_out_mute,
    "mix_out_vol": mix_out_vol,
    "mix_preset": mix_preset, "smpl_last": smpl_last, "smpl_pool": smpl_pool,
    "smpl_poly": smpl_poly, "pitched_cap": pitched_cap, "arena": arena, "stm_vpad": stm_vpad,
    "stool": stool, "song_text": song_text, "midi_trs": midi_trs, "midi_loop": midi_loop,
    "ui_tree": ui_tree, "touch_tap": touch_tap, "touch_drag": touch_drag, "topbar": topbar,
    "asong_arm": asong_arm, "asong_seqtrack": asong_seqtrack, "asong_kit2": asong_kit2,
    "asong_seq": asong_seq, "asong_dbg": asong_dbg, "outvol": outvol, "kit2": kit2,
    "inst_status": inst_status,
    "kit2_note": kit2_note, "kit2_off": kit2_off,
})
