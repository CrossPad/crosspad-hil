"""The CrossPad's sound card for this run alone: the audio-nodes service paused.

The bench's `crosspad-audio-nodes` user service keeps the card open for
PipeWire (``crosspad_out`` / ``crosspad_mic``). A capture that finds the card
busy falls back to recording *through* PipeWire, and that path loses seconds
under load -- 22.3 s of a five-minute live_set block on the bench -- which the
scoring then reads as clicks, holes and inaudible pads on the device.

So a measurement takes the card: the service is stopped for the run (the card
is then free for a direct ``plughw:`` capture and an ``aplay`` stream) and
started again afterwards. The pause is written to the pin journal first, so
a run killed in between still gets the service back on the next run.

Pausing it takes ``crosspad_out`` / ``crosspad_mic`` away from every other
session on this PC, so only the session holding the bench lease on the board
may do it, and it says who and for how long. Anyone else's run leaves the
service alone and records through PipeWire -- whose losses the scenarios then
report as the host's, not the device's.
"""
from __future__ import annotations

import os
import subprocess
import time
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from typing import Any

from crosspad_hil import pins
from crosspad_hil.audio_nodes import UNIT_NAME
from crosspad_hil.bench import HOLDER_ENV, Bench
from crosspad_hil.errors import HilError

PIN_DEVICE = "host"          # the journal's key for state that belongs to this PC
PIN_KEY = "audio_nodes"
RELEASE_S = 1.5              # PipeWire letting go of the device after the stop

Run = Callable[[list[str]], subprocess.CompletedProcess]


def _run(argv: list[str]) -> subprocess.CompletedProcess:
    return subprocess.run(argv, capture_output=True, text=True, timeout=20.0, check=False)


def service_active(run: Run | None = None) -> bool:
    run = run or _run
    try:
        r = run(["systemctl", "--user", "is-active", UNIT_NAME])
    except (OSError, subprocess.SubprocessError):
        return False
    return (r.stdout or "").strip() == "active"


def _start(run: Run | None = None) -> bool:
    run = run or _run
    try:
        run(["systemctl", "--user", "start", UNIT_NAME])
    except (OSError, subprocess.SubprocessError):
        return False
    return service_active(run)


def _restore(_link: Any, value: Any) -> bool:
    """pins restorer: a pause an unfinished run left is undone by starting it."""
    return value != "active" or _start()


pins.RESTORERS[PIN_KEY] = _restore


def pause_permit(bench: Bench, device: str | None,
                 holder: str | None) -> tuple[dict[str, Any] | None, str | None]:
    """``(holder's lease on device, None)`` when this run may pause the service,
    else ``(None, why not)``."""
    if not holder:
        return None, f"this run names no bench holder (--holder / {HOLDER_ENV})"
    if not device:
        return None, "this run names no device"
    try:
        lease = bench.status(device, history=0)["devices"][0].get("lease")
    except (OSError, HilError) as e:
        return None, f"the bench lease could not be read: {e}"
    if lease is None:
        return None, f"nobody holds the bench lease on {device}"
    if lease["holder"] != holder:
        return None, f"the bench lease on {device} is {lease['holder']}'s, not {holder}'s"
    return lease, None


def _for_how_long(expected_s: float | None, clock: Callable[[], float]) -> str:
    if not expected_s:
        return "for this measurement"
    until = time.strftime("%H:%M", time.localtime(clock() + expected_s))
    return f"for ~{expected_s / 60.0:.0f} min (until ~{until})" if expected_s >= 90 \
        else f"for ~{expected_s:.0f} s (until ~{until})"


@contextmanager
def exclusive_card(log: Callable[[str], None], *, device: str | None = None,
                   holder: str | None = None, expected_s: float | None = None,
                   enabled: bool = True, run: Run | None = None,
                   sleep: Callable[[float], None] | None = None,
                   bench: Bench | None = None,
                   clock: Callable[[], float] = time.time) -> Iterator[dict[str, Any]]:
    """Pause the audio-nodes service for the ``with`` block; ``{"paused": bool, ...}``.

    Only for ``holder`` holding the bench lease on ``device`` (``holder``
    defaults to ``$CROSSPAD_BENCH_HOLDER``); otherwise ``"refused"`` says why
    and the service is left running. Nothing to do when disabled or when the
    service is not running. A pause an unfinished run left in the journal is
    undone either way: kept until the end of this block when this run may
    pause, undone at once when it may not.
    """
    run = run or _run
    sleep = sleep or time.sleep
    info: dict[str, Any] = {"paused": False}
    if not enabled:
        yield info
        return
    left_paused = PIN_KEY in pins.pending(PIN_DEVICE)
    if not left_paused and not service_active(run):
        yield info
        return
    holder = holder or os.environ.get(HOLDER_ENV) or None
    lease, why_not = pause_permit(bench or Bench(log=log), device, holder)
    if lease is None:
        info["refused"] = why_not
        if left_paused and _start(run):
            pins.clear(PIN_DEVICE, PIN_KEY)
            log(f"{UNIT_NAME} was left paused by an unfinished run: running again")
        log(f"{UNIT_NAME} left running ({why_not}): only the bench holder may pause it, "
            "so this capture goes through PipeWire")
        yield info
        return
    if not left_paused:
        pins.record(PIN_DEVICE, PIN_KEY, "active")
    try:
        run(["systemctl", "--user", "stop", UNIT_NAME])
    except (OSError, subprocess.SubprocessError) as e:
        log(f"{UNIT_NAME} could not be paused: {e}")
    span = _for_how_long(expected_s, clock)
    info.update(paused=True, holder=holder, expected_s=expected_s,
                lease_expires_at=lease.get("expires_at"))
    log(f"{UNIT_NAME} paused by {holder} (bench lease on {device} until "
        f"{lease.get('expires_at')}) {span}: crosspad_out / crosspad_mic are gone "
        "for every session until it runs again")
    sleep(RELEASE_S)
    try:
        yield info
    finally:
        if _start(run):
            pins.clear(PIN_DEVICE, PIN_KEY)
            log(f"{UNIT_NAME} running again")
        else:
            log(f"{UNIT_NAME} did NOT start again (left in the pin journal)")


def for_run(ctx: Any, data: dict[str, Any] | None = None, *, enabled: bool = True,
            expected_s: float | None = None, **kw: Any) -> Any:
    """``exclusive_card`` for a scenario's context: its device, its holder, its log;
    what happened is kept in ``data["card_hold"]``."""
    log = ctx.log
    if data is not None:
        def log(m: str, _log: Callable[[str], None] = ctx.log) -> None:
            _log(m)
            data.setdefault("card_hold_log", []).append(m)
    return exclusive_card(log, device=ctx.device.id, holder=getattr(ctx, "holder", None),
                          expected_s=expected_s, enabled=enabled, **kw)
