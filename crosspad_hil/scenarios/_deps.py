"""Lazy access to the P1 sibling modules `capture`, `analyze`, `stim` and `ble`.

Those four modules are written by other hands and land independently of the
scenarios that use them. Importing them at module scope would make every
scenario in this package unimportable until all four exist, and the CLI
registry imports every builtin — so a missing `analyze` would take `smoke` down
with it. Everything here therefore imports *inside* the call, turns "module not
written yet" into an ``ENV`` HilError (exit 2, an environment problem, not a
firmware failure), and keeps every assumption about their shape in one file.

The contracts assumed, one place so they can be checked against the real
modules in one diff:

``capture.capture_start(device, seconds=..., preset=..., out=..., resume_audio_tasks=...)``
    -> an opaque handle.
``capture.capture_stop(handle)``
    -> ``{wav, seconds, peak_dbfs, rms_dbfs, overruns, silent}``.
``analyze.analyze(kind, wav, expected=None)``
    -> a dict; ``kind`` in multitone|click|onset|velocity|psd|silence. The
    ``expected`` payload per kind is documented at each wrapper below.
``stim.start(device, pads=…, rate_hz=…|pattern=…, seconds=…, velocity=…,``
``          gate_ms=…, transport="cdc"|"sysex_stm"|"sysex_esp", link=…)``
    -> a handle; ``stim.status(handle)`` -> ``{sent, ...}``; ``stim.stop(handle)``
    -> the same shape, final.
``ble.scan(timeout_s=…, service=…)``
    -> ``[{address, name, service_uuids}]``.
``ble.connect(address, timeout_s=…)``
    -> a client with ``properties(uuid)``, ``start_notify(uuid, cb)``,
    ``write(uuid, data, response=False)``, ``disconnect()``.

A wrapper never invents behaviour: if a module answers with a differently
spelled key the adapter here normalises it, and nothing downstream has to know.
"""
from __future__ import annotations

import importlib
import time
from typing import Any

from crosspad_hil.errors import ENV, HilError

# The four modules this package leans on but does not own.
SIBLINGS: tuple[str, ...] = ("capture", "analyze", "stim", "ble")

_HINTS: dict[str, str] = {
    "capture": "UAC2 recording lives in crosspad_hil.capture (P1); "
               "install crosspad-hil[audio] and make sure the module is present",
    "analyze": "signal kernels live in crosspad_hil.analyze (P1); "
               "install crosspad-hil[audio]",
    "stim": "the pad stimulus scheduler lives in crosspad_hil.stim (P1)",
    "ble": "the BLE client lives in crosspad_hil.ble (P1); install crosspad-hil[ble]",
}


def module(name: str) -> Any:
    """Import ``crosspad_hil.<name>``, or raise ENV naming what is missing.

    Only the module itself being absent is translated; an ImportError raised
    from *inside* a module that does exist is a real bug and propagates.
    """
    qualified = f"crosspad_hil.{name}"
    try:
        return importlib.import_module(qualified)
    except ModuleNotFoundError as e:
        if e.name not in (qualified, name):
            raise
        raise HilError(
            ENV,
            f"{qualified} is not available",
            hint=_HINTS.get(name, f"{qualified} has not been installed"),
            module=qualified,
        ) from None


def available(name: str) -> bool:
    try:
        module(name)
    except HilError:
        return False
    return True


# ── capture ──────────────────────────────────────────────────────────────────

def capture_start(device: Any, **kw: Any) -> Any:
    return module("capture").capture_start(device, **kw)


def capture_stop(handle: Any) -> dict[str, Any]:
    result = module("capture").capture_stop(handle)
    return dict(result)


# ── analyze ──────────────────────────────────────────────────────────────────

def analyze(kind: str, wav: Any, expected: Any = None) -> dict[str, Any]:
    return dict(module("analyze").analyze(kind, str(wav), expected=expected))


def analyze_segments(kind: str, wav: Any, segments: list[dict[str, Any]],
                     **extra: Any) -> dict[str, dict[str, Any]]:
    """Run one kernel over named time ranges of a capture.

    ``segments`` is ``[{"name", "start", "end"}]`` in seconds. The result is
    keyed by name. Kernels that report the ranges under ``phases`` rather than
    ``segments`` are accepted too — the two spellings mean the same thing and
    nothing above this line should care which one landed.
    """
    expected = {"segments": list(segments), **extra}
    out = analyze(kind, wav, expected=expected)
    body = out.get("segments")
    if body is None:
        body = out.get("phases")
    if body is None:
        raise HilError(
            ENV,
            f"analyze({kind!r}) returned neither 'segments' nor 'phases'",
            hint="the segmented form of the kernel is what a phased capture needs",
            keys=sorted(out),
        )
    return {str(k): dict(v) for k, v in body.items()}


# ── stim ─────────────────────────────────────────────────────────────────────

def stim_start(device: Any, **kw: Any) -> Any:
    return module("stim").start(device, **kw)


def stim_status(handle: Any) -> dict[str, Any]:
    return dict(module("stim").status(handle))


def stim_stop(handle: Any) -> dict[str, Any]:
    return dict(module("stim").stop(handle))


def stim_run(cancelled: Any, device: Any, *, poll_s: float = 0.2,
             timeout_s: float | None = None, **kw: Any) -> dict[str, Any]:
    """Start a stimulus, wait for it to finish, return its final status.

    Cancellation stops the scheduler rather than abandoning it: a stimulus left
    running writes pad presses into whatever the next scenario is doing.
    """
    handle = stim_start(device, **kw)
    seconds = kw.get("seconds")
    deadline = None
    if timeout_s is not None:
        deadline = time.monotonic() + timeout_s
    elif seconds is not None:
        deadline = time.monotonic() + float(seconds) + 5.0
    try:
        while True:
            status = stim_status(handle)
            if not status.get("running", False):
                break
            if deadline is not None and time.monotonic() > deadline:
                break
            if cancelled is not None and cancelled.wait(poll_s):
                break
    finally:
        status = stim_stop(handle)
    return status


# ── ble ──────────────────────────────────────────────────────────────────────

def ble_scan(**kw: Any) -> list[dict[str, Any]]:
    return [dict(d) for d in module("ble").scan(**kw)]


def ble_connect(address: str, **kw: Any) -> Any:
    return module("ble").connect(address, **kw)


__all__ = [
    "SIBLINGS", "analyze", "analyze_segments", "available", "ble_connect", "ble_scan",
    "capture_start", "capture_stop", "module", "stim_run", "stim_start", "stim_status",
    "stim_stop",
]
