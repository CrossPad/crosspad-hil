"""Scenario `daw_control`: the DAW Control app's wire protocol, end to end.

Design: platform-idf/docs/superpowers/specs/2026-09-06-daw-control-design.md
("Wire protocol", "Test"). Everything here goes over the ESP native MIDI port
(SysEx `F0 7D 1E …`, `crosspad_hil.midi.sysex_daw_*`) plus the usual CDC verbs
to start the app and read `LED_STATE`:

1. `hello` (0x01) must come back with 16 pads — the one thing every future
   host script depends on.
2. `led_all` (0x03) paints a known per-pad pattern; `LED_STATE` must report it
   back, 7-bit-rounded and doubled (`(v>>1)<<1`) the way the device stores it.
3. A pad press/release is simulated the way HIL always simulates one — the
   existing audio-route SysEx pad stimulus (`sysex_pad`, `0x1D 0x07/0x08`)
   rather than real hardware — and must turn into a NoteOn/NoteOff at
   `base_note + pad` on the configured channel, exactly like a host script
   would see it.
4. `text` (0x04) must not upset the app: it keeps running, and a further host
   `hello` is acknowledged with the board's `mode` frame (a hello reply comes
   too, unless the device's own hello went out less than a second earlier —
   the host answers every device hello, so that rule is what stops the
   ping-pong). Runs beside a live DAW: the scenario is a second host.
5. Control mode (dawcontrol 0.3.0): `IMU_GESTURE DOUBLE_TAP` over CDC must
   make the app send `mode` 3 and report `mode=3` in `DAW_STATE`; the same pad
   press must then come back as a `pad` frame (0x09) and no note, `PAD_PAN` on
   the held pad as a `pan` frame (0x0B); a second double tap returns to
   `mode` 0. Older firmware (hello caps without
   `control`) skips this step rather than failing it.

Exit codes: a MIDI port or the app missing, or a DAW already connected to the
app (its script repaints the pads under the test), is a bench problem (ENV,
exit 2); any protocol mismatch is the firmware failing the test (exit 1).
"""
from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any

from crosspad_hil import verbs
from crosspad_hil.errors import ENV, HilError
from crosspad_hil.midi import (
    MidiIO,
    MidiRole,
    parse_daw_hello,
    parse_daw_mode,
    parse_daw_pad,
    parse_daw_pan,
    sysex_daw_hello,
    sysex_daw_led_all,
    sysex_daw_text,
    sysex_pad,
)
from crosspad_hil.scenarios._audio import ensure_app_running
from crosspad_hil.scenarios.base import Context, Progress, Report, register

#: The app's CDC/registry name. `verbs.app_start` rejects whitespace (the
#: firmware reads it with `sscanf("%31s", ...)`, which stops at the first
#: space), so the REGISTER_APP identifier — not the launcher's display title
#: "DAW Control" from the design doc — is what goes over the wire here.
APP_NAME = "DawControl"
BASE_NOTE = 36            # the host default from the design doc ("Wire protocol")
CHANNEL = 9               # 0-based; "channel 10" as a host script would say it
TEST_PAD = 3
TEST_VEL = 100
TEST_PAN = 100            # right of the middle (64), inside the 1..127 the wire carries
RX_SETTLE_S = 0.3
HOST_SETTLE_S = 1.0       # time for a live DAW script to answer the app's start-up hello
HELLO_TIMEOUT_S = 2.0
NOTE_TIMEOUT_S = 2.0
MODE_TIMEOUT_S = 2.0
WANT_PADS = 16
MODE_PADS = 0
MODE_CONTROL = 3
STEPS = 5


@dataclass
class Params:
    """Nothing to configure in v1 — one fixed pattern, one fixed pad."""


def _round7(v: int) -> int:
    """What the device reports back for a 0..255 value it received as 7-bit: (v>>1)<<1."""
    return (v >> 1) << 1


def _led_pattern() -> list[tuple[int, int, int]]:
    """pad i -> (i*16, 255-i*16, 64) — a pattern where every pad differs."""
    return [(i * 16, 255 - i * 16, 64) for i in range(16)]


def _expected_led_colors(pattern: list[tuple[int, int, int]]) -> list[str]:
    return ["".join(f"{_round7(c):02X}" for c in rgb) for rgb in pattern]


def _is_note_on(msg: list[int] | None, note: int) -> bool:
    return (msg is not None and len(msg) >= 3 and (msg[0] & 0xF0) == 0x90
            and msg[0] & 0x0F == CHANNEL and msg[1] == note and msg[2] > 0)


def _is_note_off(msg: list[int] | None, note: int) -> bool:
    if msg is None or len(msg) < 3 or msg[0] & 0x0F != CHANNEL or msg[1] != note:
        return False
    status = msg[0] & 0xF0
    return status == 0x80 or (status == 0x90 and msg[2] == 0)


class DawControlScenario:
    name = "daw_control"
    Params = Params
    description = "DAW Control app: hello, LED paint, pad->note, text — over the ESP MIDI port"

    def run(self, ctx: Context, params: Params, progress: Progress) -> Report:
        data: dict[str, Any] = {}
        link = None
        try:
            if ctx.device.ports.esp_midi is None:
                raise HilError(ENV, f"{ctx.device.id} has no ESP native MIDI port",
                               hint="the DAW Control protocol only answers there")

            link = ctx.open_cdc()
            ensure_app_running(ctx, link, APP_NAME,
                               not_installed_reason=f"{APP_NAME!r} is not installed")
            # A DAW script on the same port answers the app's hello and keeps
            # painting the pads, so the LED and note checks below race it.
            ctx.cancelled.wait(HOST_SETTLE_S)
            ctx.check_cancelled()
            try:
                live = verbs.daw_state(link)
            except HilError:
                live = None
            if live is not None and int(live.get("connected", 0)) == 1:
                raise HilError(ENV, "a DAW host is already talking to DAW Control",
                               hint="close the DAW (or its CrossPad script) for this "
                                    "scenario; it needs the MIDI port to itself")

            io = MidiIO(ctx.device, MidiRole.ESP)
            io.open()
            ctx.opened.append(io)
            io.drain()
            ctx.cancelled.wait(RX_SETTLE_S)
            ctx.check_cancelled()

            # --- hello ---------------------------------------------------
            progress(0, STEPS, "hello")
            io.send_sysex(sysex_daw_hello())
            hello = self._wait_hello(io, ctx)
            data["hello"] = hello
            if hello is None:
                return Report(passed=False,
                              summary="daw_control: no hello reply from the app",
                              data=data, artifacts=[], exit_code=1)
            if hello["pads"] != WANT_PADS:
                return Report(passed=False,
                              summary=f"daw_control: hello reports pads={hello['pads']}, "
                                      f"want {WANT_PADS}",
                              data=data, artifacts=[], exit_code=1)

            # --- led_all ---------------------------------------------------
            progress(1, STEPS, "led_all")
            pattern = _led_pattern()
            io.send_sysex(sysex_daw_led_all(pattern))
            ctx.cancelled.wait(RX_SETTLE_S)
            ctx.check_cancelled()
            leds = verbs.led_state(link)
            expected = _expected_led_colors(pattern)
            data["led_expected"] = expected
            data["led_actual"] = leds["colors"]
            if leds["colors"] != expected:
                return Report(passed=False,
                              summary="daw_control: LED_STATE does not match the led_all pattern",
                              data=data, artifacts=[], exit_code=1)

            # --- pad -> note -------------------------------------------------
            progress(2, STEPS, f"pad {TEST_PAD} press/release -> note")
            note = BASE_NOTE + TEST_PAD
            io.drain()
            io.send_sysex(sysex_pad(True, TEST_PAD, TEST_VEL))
            note_on = io.receive_note(NOTE_TIMEOUT_S)
            io.send_sysex(sysex_pad(False, TEST_PAD))
            note_off = io.receive_note(NOTE_TIMEOUT_S)
            data["note_on"] = note_on
            data["note_off"] = note_off
            if not _is_note_on(note_on, note):
                return Report(passed=False,
                              summary=f"daw_control: pad {TEST_PAD} press did not produce "
                                      f"NoteOn {note} ch{CHANNEL + 1} (got {note_on})",
                              data=data, artifacts=[], exit_code=1)
            if not _is_note_off(note_off, note):
                return Report(passed=False,
                              summary=f"daw_control: pad {TEST_PAD} release did not produce "
                                      f"NoteOff {note} ch{CHANNEL + 1} (got {note_off})",
                              data=data, artifacts=[], exit_code=1)

            # --- text, then hello again --------------------------------------
            progress(3, STEPS, "text + hello re-sync")
            io.send_sysex(sysex_daw_text(0, "HIL"))
            ctx.cancelled.wait(RX_SETTLE_S)
            ctx.check_cancelled()
            apps = verbs.app_list(link)
            data["running_after_text"] = apps.get("running")
            if apps.get("running") != APP_NAME:
                return Report(passed=False,
                              summary="daw_control: the app is no longer running after `text`",
                              data=data, artifacts=[], exit_code=1)
            # DAW_STATE (firmware with the crosspad-dawcontrol snapshot) shows
            # the row the host just sent; older firmware answers nothing and
            # the check is skipped rather than failed.
            try:
                state = verbs.daw_state(link)
            except HilError:
                state = None
            data["daw_state_after_text"] = state
            if state is not None and state["rows"][0] != "HIL":
                return Report(passed=False,
                              summary=f"daw_control: row 0 shows {state['rows'][0]!r}, "
                                      "not the 'HIL' the host sent",
                              data=data, artifacts=[], exit_code=1)
            has_control = bool(hello["caps"].get("control"))
            io.drain()
            io.send_sysex(sysex_daw_hello())
            if has_control:
                mode_after_hello = self._wait_frame(io, ctx, parse_daw_mode, MODE_TIMEOUT_S)
                data["mode_after_hello"] = mode_after_hello
                if mode_after_hello != MODE_PADS:
                    return Report(passed=False,
                                  summary="daw_control: a host hello after `text` was not "
                                          f"acknowledged with mode {MODE_PADS} "
                                          f"(got {mode_after_hello})",
                                  data=data, artifacts=[], exit_code=1)

            # --- control mode: double tap -> mode 3 -> pad frame, no note ----
            progress(4, STEPS, "double tap -> control mode")
            if has_control:
                fail = self._check_control_mode(io, link, ctx, data)
                if fail is not None:
                    return Report(passed=False, summary=fail, data=data,
                                  artifacts=[], exit_code=1)
            else:
                data["control_mode"] = "skipped: firmware without the control cap"

            progress(STEPS, STEPS, "done")
            return Report(passed=True, summary="daw_control: PASS", data=data,
                          artifacts=[], exit_code=0)
        finally:
            if link is not None:
                try:
                    verbs.app_stop(link)
                except HilError:
                    pass
            ctx.close_all()

    def _check_control_mode(self, io: MidiIO, link, ctx: Context,
                            data: dict[str, Any]) -> str | None:
        """Returns a failure summary, or None when the control mode round trip held."""
        io.drain()
        verbs.imu_gesture(link, "DOUBLE_TAP")
        mode = self._wait_frame(io, ctx, parse_daw_mode, MODE_TIMEOUT_S)
        data["mode_after_double_tap"] = mode
        if mode != MODE_CONTROL:
            return f"daw_control: double tap did not send mode {MODE_CONTROL} (got {mode})"
        state = verbs.daw_state(link)
        data["daw_state_control"] = state
        if int(state.get("mode", -1)) != MODE_CONTROL:
            return f"daw_control: DAW_STATE mode={state.get('mode')} after the double tap"

        io.drain()
        io.send_sysex(sysex_pad(True, TEST_PAD, TEST_VEL))
        pad = self._wait_frame(io, ctx, parse_daw_pad, NOTE_TIMEOUT_S)
        # The held pad's left/right balance, as the STM would stream it,
        # must come back as a `pan` frame with the same value.
        verbs.pad_pan(link, TEST_PAD, TEST_PAN)
        pan = self._wait_frame(io, ctx, parse_daw_pan, NOTE_TIMEOUT_S)
        io.send_sysex(sysex_pad(False, TEST_PAD))
        release = self._wait_frame(io, ctx, parse_daw_pad, NOTE_TIMEOUT_S)
        stray_note = io.receive_note(RX_SETTLE_S)
        data["control_pad_press"] = pad
        data["control_pad_pan"] = pan
        data["control_pad_release"] = release
        data["control_stray_note"] = stray_note
        if pan != (TEST_PAD, TEST_PAN):
            return (f"daw_control: PAD_PAN {TEST_PAN} on held pad {TEST_PAD} gave {pan}, "
                    "not a pan frame")
        if stray_note is not None:
            return f"daw_control: control mode still sent a note ({stray_note})"
        if pad != (TEST_PAD, True):
            return f"daw_control: control-mode press of pad {TEST_PAD} gave {pad}, not a pad frame"
        if release != (TEST_PAD, False):
            return f"daw_control: control-mode release of pad {TEST_PAD} gave {release}"

        io.drain()
        verbs.imu_gesture(link, "DOUBLE_TAP")
        mode = self._wait_frame(io, ctx, parse_daw_mode, MODE_TIMEOUT_S)
        data["mode_after_second_double_tap"] = mode
        if mode != MODE_PADS:
            return f"daw_control: second double tap did not send mode {MODE_PADS} (got {mode})"
        return None

    @staticmethod
    def _wait_frame(io: MidiIO, ctx: Context, parse, timeout_s: float):
        """The first SysEx `parse` accepts within `timeout_s`, else None. Notes are skipped."""
        deadline = time.monotonic() + timeout_s
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                return None
            frame = io.receive(min(remaining, 0.25))
            if frame is None:
                ctx.check_cancelled()
                continue
            try:
                return parse(frame)
            except HilError:
                continue

    @staticmethod
    def _wait_hello(io: MidiIO, ctx: Context, timeout_s: float = HELLO_TIMEOUT_S) -> dict | None:
        deadline = time.monotonic() + timeout_s
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                return None
            frame = io.receive(min(remaining, 0.25))
            if frame is None:
                ctx.check_cancelled()
                continue
            try:
                return parse_daw_hello(frame)
            except HilError:
                continue    # not a hello reply; keep waiting for one


SCENARIO = register(DawControlScenario())
