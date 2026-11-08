"""USB profile switching (SysEx 0x1B) with wait-for-enumeration and a restoring context.

Ported from platform-idf tools: ota_flash.py send_midi_switch_to_cdc()/wait_for_cdc(),
hil_audio_loopback.py ensure_audio_mode()/restore_default_mode(),
hil_usb_mode_cycle.py wait_for(). The frame always goes to the ESP native MIDI port:
the STM bridge drops it silently (measured, hil_audio_loopback.py esp_midi_port()).
"""
from __future__ import annotations

import subprocess
import sys
import threading
import time
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import replace
from typing import Any

from crosspad_hil.audio_nodes import hold_defaults as _hold_defaults
from crosspad_hil.devices import Device, UsbMode, discover
from crosspad_hil.errors import BAD_ARGS, NO_DEVICE, TIMEOUT, HilError
from crosspad_hil.midi import MidiIO, MidiRole, sysex_usb_mode

POLL_S = 0.5  # ota_flash.py wait_for_cdc() poll interval
# Budget for opening the ESP MIDI port and sending one SysEx frame. rtmidi/ALSA are C
# extensions with no timeout of their own: a USB MIDI endpoint that the firmware left
# half-torn-down (audio init failed mid mode-switch) can make open_port()/send_message()/
# close_port() block in the kernel indefinitely, well outside the poll loop below that
# only starts *after* this call returns. That is a real failure seen in the field (a
# release_gate run stuck 4.5h in usb_mode_cycle with no console output and no report),
# and it cannot be fixed by adding a deadline check anywhere in Python — the call itself
# never returns control. So it runs on a daemon thread that this function abandons if it
# does not finish in time; the thread cannot be killed, only ignored.
SEND_TIMEOUT_S = 5.0


def _send_request(device: Device, mode: UsbMode, midi_factory: Callable[..., Any]) -> None:
    """Open the ESP MIDI port and send the mode-switch SysEx, bounded by SEND_TIMEOUT_S
    regardless of how the underlying MIDI backend behaves.

    With the real backend this runs in a **subprocess**, not a thread. Release
    gate #4 (2026-09-02) froze for good inside this call: rtmidi's ALSA input
    teardown blocked holding the GIL once the device re-enumerated, so the
    abandoning thread below never got the interpreter back, nor did any other
    thread -- the per-step watchdog, the stall dumper, the main loop. A thread
    can only be abandoned; a process can be killed. An injected ``midi_factory``
    (tests, fakes) keeps the in-process thread path, since a fake cannot cross
    a process boundary.
    """
    if midi_factory is MidiIO:
        _send_request_isolated(device, mode)
        return

    box: dict[str, BaseException] = {}

    def worker() -> None:
        try:
            io = midi_factory(device, MidiRole.ESP, listen=False)
            with io:
                io.send_sysex(sysex_usb_mode(mode))
        except BaseException as exc:  # noqa: BLE001 - re-raised on the caller's thread below
            box["error"] = exc

    t = threading.Thread(target=worker, name="usbmode-send", daemon=True)
    t.start()
    t.join(SEND_TIMEOUT_S)
    if t.is_alive():
        raise _wedged(device, mode)
    if "error" in box:
        raise box["error"]


def _wedged(device: Device, mode: UsbMode) -> HilError:
    return HilError(
        TIMEOUT,
        f"{device.id}: sending the mode-switch request ({mode.value}) did not return "
        f"within {SEND_TIMEOUT_S:.0f}s — the ESP MIDI port looks wedged",
        hint="a USB MIDI endpoint left mid-teardown can block open/send/close forever; "
             "power-cycle the device",
        device=device.id, mode=mode.value,
    )


def _send_request_isolated(device: Device, mode: UsbMode) -> None:
    """``python -m crosspad_hil.usbmode <device-id> <mode>`` with a hard timeout."""
    cmd = [sys.executable, "-m", "crosspad_hil.usbmode", device.id, mode.value]
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True,
                              timeout=SEND_TIMEOUT_S + 5.0, check=False)
    except subprocess.TimeoutExpired as exc:
        raise _wedged(device, mode) from exc
    if proc.returncode != 0:
        detail = (proc.stderr or proc.stdout or "").strip().splitlines()
        raise HilError(
            NO_DEVICE if "NO_DEVICE" in (proc.stderr or "") else TIMEOUT,
            f"{device.id}: mode-switch sender exited {proc.returncode}"
            + (f": {detail[-1]}" if detail else ""),
            hint="the ESP MIDI port could not be opened or written; run `crosspad-hil devices`",
            device=device.id, mode=mode.value, returncode=proc.returncode,
        )


def _send_main(argv: list[str]) -> int:
    """Child side of _send_request_isolated: find the device, send, exit."""
    if len(argv) != 2:
        print("usage: python -m crosspad_hil.usbmode <device-id> <default|audio>",
              file=sys.stderr)
        return 2
    device_id, mode = argv
    try:
        found = [d for d in discover() if d.id == device_id]
        if not found:
            raise HilError(NO_DEVICE, f"{device_id} not found")
        io = MidiIO(found[0], MidiRole.ESP, listen=False)
        with io:
            io.send_sysex(sysex_usb_mode(UsbMode(mode)))
    except HilError as exc:
        print(f"{exc.code}: {exc.message}", file=sys.stderr)
        return 1
    return 0


def _reached(d: Device, mode: UsbMode) -> bool:
    if d.usb_mode == mode:
        return True
    if mode is UsbMode.DEFAULT and d.ports.cdc is not None:
        return True
    if mode is UsbMode.AUDIO and d.ports.uac2 is not None:
        return True
    return False


# The CDC profile is guarded on the device (USB_GUARD, on by default): the
# board shows Allow/Deny for 30 s before it re-enumerates, so the wait for
# `default` covers the prompt plus enumeration.
DEFAULT_TIMEOUT_S = 20.0
GUARDED_TIMEOUT_S = 45.0


def set_mode(device: Device, mode: UsbMode, *, wait: bool = True, timeout_s: float | None = None,
             discover_fn: Callable[[], list[Device]] = discover,
             midi_factory: Callable[..., Any] = MidiIO,
             sleep: Callable[[float], None] = time.sleep,
             clock: Callable[[], float] = time.monotonic) -> Device:
    """Ask the device for `mode` over the ESP MIDI port and (optionally) wait for it to show up."""
    mode = UsbMode(mode)
    if mode not in (UsbMode.DEFAULT, UsbMode.AUDIO):
        raise HilError(BAD_ARGS, f"usb mode {mode.value!r} cannot be set",
                       hint="use 'default' or 'audio'; bootloader is ota.request_bootloader()")
    if device.ports.esp_midi is None:
        raise HilError(NO_DEVICE, f"{device.id} has no ESP MIDI port; cannot switch USB mode",
                       hint="the 0x1B frame must reach the ESP native MIDI port, "
                            "not the STM bridge")
    if timeout_s is None:
        timeout_s = GUARDED_TIMEOUT_S if mode is UsbMode.DEFAULT else DEFAULT_TIMEOUT_S
    if mode is UsbMode.AUDIO:
        # The card this brings up is a test's, not one the user plugged in:
        # the desktop's sound stays where it is until this process exits.
        _hold_defaults()
    _send_request(device, mode, midi_factory)
    if not wait:
        return replace(device, usb_mode=mode)

    deadline = clock() + timeout_s
    last_seen: str | None = None
    while True:
        for d in discover_fn():
            if d.id != device.id:
                continue
            last_seen = d.usb_mode.value
            if _reached(d, mode):
                return d
        if clock() >= deadline:
            raise HilError(TIMEOUT,
                           f"{device.id} did not re-enumerate in mode {mode.value!r} "
                           f"within {timeout_s:.0f} s",
                           hint=("the board may be asking Allow/Deny on its screen (USB serial "
                                 "guard): press Allow, or `USB_GUARD 0` over CDC once on a bench "
                                 "board; otherwise it may have re-enumerated with a different "
                                 "serial — run `crosspad-hil devices`")
                                if mode is UsbMode.DEFAULT else
                                "device may have re-enumerated with a different serial; "
                                "run `crosspad-hil devices`",
                           device=device.id, mode=mode.value, last_seen=last_seen)
        sleep(POLL_S)


@contextmanager
def audio_mode(device: Device, *, keep: bool = False, **kw: Any) -> Iterator[Device]:
    """Enter the MIDI+UAC2 profile; leave it on exit unless keep=True (spec §2.4)."""
    entered = set_mode(device, UsbMode.AUDIO, **kw)
    try:
        yield entered
    finally:
        if not keep:
            set_mode(entered, UsbMode.DEFAULT, **kw)


if __name__ == "__main__":
    sys.exit(_send_main(sys.argv[1:]))
