"""USB MIDI access (rtmidi) and the CrossPad SysEx catalog.

Frames, byte layouts and the replies are ported from platform-idf:
  tools/hil_midi_bench.py (echo_frame), tools/hil_sampler_record.py (Midi.query_route),
  tools/ota_flash.py (SYSEX_USB_MODE_DEFAULT), tools/requestBootloader.py (SYSEX_DATA),
  main/audio_route_control.cpp (send_query_reply, AUDIO_ROUTE_SUB_ECHO).
"""
from __future__ import annotations

import queue
import time
from collections.abc import Callable
from enum import Enum
from typing import Any

from crosspad_hil import knowledge
from crosspad_hil.devices import Device, MidiPortInfo, UsbMode
from crosspad_hil.errors import (
    BAD_ARGS,
    BAD_SYSEX,
    DENIED_SYSEX,
    ENV,
    NO_DEVICE,
    NOT_SUPPORTED,
    TIMEOUT,
    HilError,
)

SYSEX_START = 0xF0
SYSEX_END = 0xF7


class MidiRole(str, Enum):
    ESP = "esp"
    STM = "stm"


def _sysex_knowledge() -> dict:
    return knowledge.load("sysex")


def _check7(value: int, what: str) -> int:
    if not isinstance(value, int) or value < 0 or value > 0x7F:
        raise HilError(BAD_ARGS, f"{what} must be 0..127, got {value!r}")
    return value


# ---------------------------------------------------------------------------
# builders
# ---------------------------------------------------------------------------

def sysex_usb_mode(mode: UsbMode) -> bytes:
    """F0 7D 1B 01|02 F7 — ota_flash SYSEX_USB_MODE_DEFAULT / audio_loopback SYSEX_AUDIO."""
    k = _sysex_knowledge()
    if mode == UsbMode.DEFAULT:
        payload = int(k["usb_mode"]["default"])
    elif mode == UsbMode.AUDIO:
        payload = int(k["usb_mode"]["audio"])
    else:
        raise HilError(BAD_ARGS, f"usb mode {mode.value} cannot be requested over SysEx",
                       hint="only 'default' and 'audio' are switchable")
    return bytes([SYSEX_START, int(k["manufacturer"]), int(k["usb_mode"]["id"]),
                  payload, SYSEX_END])


def sysex_audio_route(sub: int, *args: int) -> bytes:
    """F0 7D 1D <sub> <args…> F7 — hil_sampler_record.py Midi.route()."""
    k = _sysex_knowledge()
    _check7(sub, "sub")
    body = [_check7(a, "arg") for a in args]
    return bytes([SYSEX_START, int(k["manufacturer"]), int(k["audio_route"]["id"]),
                  sub, *body, SYSEX_END])


def sysex_pad(on: bool, pad: int, vel: int = 127) -> bytes:
    """Pad press 0x07 <pad> <vel> / release 0x08 <pad> — hil_sampler_record.py SysexPads."""
    k = _sysex_knowledge()["audio_route"]["subs"]
    if not isinstance(pad, int) or pad < 0 or pad > 15:
        raise HilError(BAD_ARGS, f"pad must be 0..15, got {pad!r}")
    if on:
        return sysex_audio_route(int(k["pad_press"]), pad, _check7(vel, "vel"))
    return sysex_audio_route(int(k["pad_release"]), pad)


def sysex_kit_load(kit_id: int) -> bytes:
    """F0 7D 1D 0B <kit> F7 — the CDC ``KIT_LOAD`` path, reachable without CDC.

    The USB-audio profile has no CDC endpoint at all, so a scenario that swaps
    kits *while recording* (``kit_swap_audio``) cannot ask for the swap the
    usual way. This is the same request arriving over the ESP MIDI port; the
    device queues it exactly as it queues the text verb, coalescing included.
    """
    k = _sysex_knowledge()["audio_route"]["subs"]
    return sysex_audio_route(int(k["kit_load"]), _check7(kit_id, "kit_id"))


def sysex_bounce(scene: int, track: int) -> bytes:
    """F0 7D 1D 0C <scene> <track> F7 — CDC ``ASONG BOUNCE`` from the USB-audio profile.

    Both 0-based. The console answers ``BOUNCE <s> <t> started|refused``; the
    result (``ASONG bounce=``, the clip) is read over CDC afterwards. A bounce
    plays the scene once from its first bar and stops the sequencer.
    """
    k = _sysex_knowledge()["audio_route"]["subs"]
    return sysex_audio_route(int(k["bounce"]), _check7(scene, "scene"), _check7(track, "track"))


def sysex_preview(pad: int) -> bytes:
    """F0 7D 1D 0A <pad> F7 — audition the pad's first sample (the kit browser's preview).

    It plays through the sampler engine (note 16, volume 100, pan 0), so it
    takes a voice like a pad does.
    """
    if not isinstance(pad, int) or pad < 0 or pad > 15:
        raise HilError(BAD_ARGS, f"pad must be 0..15, got {pad!r}")
    k = _sysex_knowledge()["audio_route"]["subs"]
    return sysex_audio_route(int(k["preview"]), pad)


def sysex_mic_src(src: int) -> bytes:
    """F0 7D 1D 02 <src> F7 — what USB IN records: 0/1 a codec's ADC, 2 both summed,
    3 the pads-only mixer bus ("USB rec", digital: what a take is made of)."""
    if not isinstance(src, int) or src < 0 or src > 3:
        raise HilError(BAD_ARGS, f"mic_src must be 0..3, got {src!r}")
    k = _sysex_knowledge()["audio_route"]["subs"]
    return sysex_audio_route(int(k["mic_src"]), src)


def sysex_echo(seq: int) -> bytes:
    """28-bit sequence, 7 bits per byte, MSB first — hil_midi_bench.py echo_frame()."""
    if not isinstance(seq, int) or seq < 0 or seq > 0x0FFFFFFF:
        raise HilError(BAD_ARGS, f"echo seq must be 0..0x0FFFFFFF, got {seq!r}")
    k = _sysex_knowledge()["audio_route"]["subs"]
    return sysex_audio_route(int(k["echo"]),
                             (seq >> 21) & 0x7F, (seq >> 14) & 0x7F,
                             (seq >> 7) & 0x7F, seq & 0x7F)


def decode_echo_seq(frame: bytes) -> int:
    """Inverse of sysex_echo — hil_midi_bench.py parse_replies()."""
    return ((frame[4] & 0x7F) << 21) | ((frame[5] & 0x7F) << 14) \
        | ((frame[6] & 0x7F) << 7) | (frame[7] & 0x7F)


def sysex_query() -> bytes:
    """F0 7D 1D 10 F7 — hil_midi_stress.py QUERY."""
    k = _sysex_knowledge()["audio_route"]["subs"]
    return sysex_audio_route(int(k["query"]))


def _sysex_daw_knowledge() -> dict:
    return _sysex_knowledge()["daw"]


def sysex_daw(sub: int, *args: int) -> bytes:
    """F0 7D 1E <sub> <args…> F7 — the DAW Control app's generic SysEx frame.

    Byte layouts: docs/superpowers/specs/2026-09-06-daw-control-design.md
    "Wire protocol". Every argument is 7-bit like every other CrossPad SysEx.
    """
    k = _sysex_knowledge()
    _check7(sub, "sub")
    body = [_check7(a, "arg") for a in args]
    return bytes([SYSEX_START, int(k["manufacturer"]), int(k["daw"]["id"]),
                  sub, *body, SYSEX_END])


def sysex_daw_hello(daw_id: int = 0, proto_ver: int = 1) -> bytes:
    """0x01 hello, host->dev: daw_id (0 other, 1 FL Studio, 2 Ableton), proto_ver."""
    k = _sysex_daw_knowledge()
    return sysex_daw(int(k["subs"]["hello"]), _check7(daw_id, "daw_id"),
                     _check7(proto_ver, "proto_ver"))


def daw_ids() -> dict[str, int]:
    """DAW protocol daw_id per name (knowledge `daw.daw_ids`)."""
    return {name: int(v) for name, v in _sysex_daw_knowledge()["daw_ids"].items()}


def parse_daw_launch(frame: bytes | list[int]) -> int | None:
    """daw_id of a device->hub `launch` frame (F0 7D 1E 0C id F7); None for any other frame."""
    k = _sysex_knowledge()
    if len(frame) != 6 or frame[0] != SYSEX_START or frame[1] != int(k["manufacturer"]) \
            or frame[2] != int(k["daw"]["id"]) or frame[3] != int(k["daw"]["subs"]["launch"]) \
            or frame[5] != SYSEX_END:
        return None
    return frame[4]


def _check_byte(value: int, what: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value < 0 or value > 255:
        raise HilError(BAD_ARGS, f"{what} must be 0..255, got {value!r}")
    return value


def sysex_daw_led(pad: int, r: int, g: int, b: int) -> bytes:
    """0x02 led: one pad's resting colour. r/g/b are 0..255 host-side, sent as
    7-bit (>>1) — the device doubles them back, so a round trip loses the low bit."""
    k = _sysex_daw_knowledge()
    if not isinstance(pad, int) or isinstance(pad, bool) or pad < 0 or pad > 15:
        raise HilError(BAD_ARGS, f"pad must be 0..15, got {pad!r}")
    return sysex_daw(int(k["subs"]["led"]), pad, _check_byte(r, "r") >> 1,
                     _check_byte(g, "g") >> 1, _check_byte(b, "b") >> 1)


def sysex_daw_led_all(colors: Any) -> bytes:
    """0x03 led_all: exactly 16 x (r, g, b), pad 0..15 in order."""
    k = _sysex_daw_knowledge()
    colors = list(colors)
    if len(colors) != 16:
        raise HilError(BAD_ARGS, f"led_all needs exactly 16 colours, got {len(colors)}")
    args: list[int] = []
    for pad, (r, g, b) in enumerate(colors):
        args += [_check_byte(r, f"pad {pad} r") >> 1, _check_byte(g, f"pad {pad} g") >> 1,
                 _check_byte(b, f"pad {pad} b") >> 1]
    return sysex_daw(int(k["subs"]["led_all"]), *args)


def sysex_daw_text(row: int, text: str) -> bytes:
    """0x04 text: one LCD row (0..2). ASCII 0x20..0x7E, truncated at 24 chars."""
    k = _sysex_daw_knowledge()
    if not isinstance(row, int) or isinstance(row, bool) or row < 0 or row > 2:
        raise HilError(BAD_ARGS, f"row must be 0..2, got {row!r}")
    if not isinstance(text, str):
        raise HilError(BAD_ARGS, f"text must be a str, got {text!r}")
    text = text[:24]
    bad = [ch for ch in text if not (0x20 <= ord(ch) <= 0x7E)]
    if bad:
        raise HilError(BAD_ARGS, f"text must be printable ASCII (0x20..0x7E), got {text!r}",
                       hint="no newlines or control characters")
    return sysex_daw(int(k["subs"]["text"]), row, *(ord(ch) for ch in text))


def sysex_daw_beat(value: int) -> bytes:
    """0x05 beat: 0 off / 1 bar / 2 beat."""
    k = _sysex_daw_knowledge()
    return sysex_daw(int(k["subs"]["beat"]), _check7(value, "value"))


def sysex_daw_transport(playing: bool, recording: bool, loop: bool) -> bytes:
    """0x06 transport: bits 0 playing, 1 recording, 2 loop."""
    k = _sysex_daw_knowledge()
    bits = (1 if playing else 0) | (2 if recording else 0) | (4 if loop else 0)
    return sysex_daw(int(k["subs"]["transport"]), bits)


def sysex_daw_mode(mode: int) -> bytes:
    """0x07 mode: 0 pads / 1 steps / 2 clips (others are accepted and echoed)."""
    k = _sysex_daw_knowledge()
    return sysex_daw(int(k["subs"]["mode"]), _check7(mode, "mode"))


def sysex_daw_config(base_note: int, channel: int) -> bytes:
    """0x08 config: base_note, channel — channel is 1..16 on the API, 0..15 on the wire."""
    k = _sysex_daw_knowledge()
    if not isinstance(channel, int) or isinstance(channel, bool) or channel < 1 or channel > 16:
        raise HilError(BAD_ARGS, f"channel must be 1..16, got {channel!r}")
    return sysex_daw(int(k["subs"]["config"]), _check7(base_note, "base_note"), channel - 1)


def sysex_daw_bye() -> bytes:
    """0x7F bye: host script unloading, no args."""
    k = _sysex_daw_knowledge()
    return sysex_daw(int(k["subs"]["bye"]))


def parse_daw_hello(frame: bytes) -> dict:
    """Device->host hello reply: F0 7D 1E 01 proto_ver pads rows caps F7.

    caps bits: 0 rgb, 1 text, 2 encoder, 3 aftertouch.
    """
    k = _sysex_daw_knowledge()
    daw_id_ = int(_sysex_knowledge()["daw"]["id"])
    hello_sub = int(k["subs"]["hello"])
    if len(frame) != 9 or frame[0] != SYSEX_START or frame[2] != daw_id_ \
            or frame[3] != hello_sub:
        raise HilError(BAD_SYSEX, "not a DAW hello reply", frame=frame.hex(" ").upper())
    caps_byte = frame[7]
    cap_bits = k["caps"]
    return {
        "proto_ver": frame[4], "pads": frame[5], "rows": frame[6],
        "caps": {name: bool(caps_byte & (1 << int(bit))) for name, bit in cap_bits.items()},
    }


def parse_daw_mode(frame: bytes) -> int:
    """Device->host mode: F0 7D 1E 07 mode F7 — 0 pads / 3 control (knowledge `daw.modes`)."""
    k = _sysex_daw_knowledge()
    daw_id_ = int(_sysex_knowledge()["daw"]["id"])
    if len(frame) != 6 or frame[0] != SYSEX_START or frame[2] != daw_id_ \
            or frame[3] != int(k["subs"]["mode"]):
        raise HilError(BAD_SYSEX, "not a DAW mode frame", frame=frame.hex(" ").upper())
    return frame[4]


def parse_daw_pad(frame: bytes) -> tuple[int, bool]:
    """Device->host control-mode pad: F0 7D 1E 09 pad state F7 -> (pad, pressed)."""
    k = _sysex_daw_knowledge()
    daw_id_ = int(_sysex_knowledge()["daw"]["id"])
    if len(frame) != 7 or frame[0] != SYSEX_START or frame[2] != daw_id_ \
            or frame[3] != int(k["subs"]["pad"]):
        raise HilError(BAD_SYSEX, "not a DAW pad frame", frame=frame.hex(" ").upper())
    return frame[4], frame[5] == 1


def parse_daw_pan(frame: bytes) -> tuple[int, int]:
    """Device->host control-mode pan: F0 7D 1E 0B pad value F7 -> (pad, value 1..127, 64 middle)."""
    k = _sysex_daw_knowledge()
    daw_id_ = int(_sysex_knowledge()["daw"]["id"])
    if len(frame) != 7 or frame[0] != SYSEX_START or frame[2] != daw_id_ \
            or frame[3] != int(k["subs"]["pan"]):
        raise HilError(BAD_SYSEX, "not a DAW pan frame", frame=frame.hex(" ").upper())
    return frame[4], frame[5]


def sysex_companion(verb: str) -> bytes:
    """F0 7D 1F 01 <ASCII verb> F7 — one companion verb (knowledge `companion`)."""
    return _companion_frame(verb, "request")


def sysex_companion_notify(verb: str) -> bytes:
    """F0 7D 1F 04 <ASCII verb> F7 — a companion verb the device runs without answering."""
    return _companion_frame(verb, "notify")


def _companion_frame(verb: str, sub: str) -> bytes:
    k = _sysex_knowledge()["companion"]
    if not isinstance(verb, str) or not verb.strip():
        raise HilError(BAD_ARGS, f"verb must be a non-empty str, got {verb!r}")
    bad = [ch for ch in verb if not (0x20 <= ord(ch) <= 0x7E)]
    if bad:
        raise HilError(BAD_ARGS, f"verb must be printable ASCII (0x20..0x7E), got {verb!r}",
                       hint="one line, no newline — the device adds it")
    verb = verb.strip()
    if len(verb) > int(k["max_request"]):
        raise HilError(BAD_ARGS, f"verb longer than {k['max_request']} bytes: {len(verb)}")
    return bytes([SYSEX_START, int(_sysex_knowledge()["manufacturer"]), int(k["id"]),
                  int(k["subs"][sub]), *(ord(ch) for ch in verb), SYSEX_END])


def parse_companion_chunk(frame: bytes) -> tuple[str, bool] | None:
    """(text, last) for a 0x1F reply chunk; None for any other frame."""
    k = _sysex_knowledge()["companion"]
    if len(frame) < 5 or frame[0] != SYSEX_START or frame[2] != int(k["id"]):
        return None
    sub = frame[3]
    if sub == int(k["subs"]["reply_last"]):
        last = True
    elif sub == int(k["subs"]["reply_more"]):
        last = False
    else:
        return None
    return bytes(frame[4:-1]).decode("ascii", errors="replace"), last


def companion_command(io: MidiIO, verb: str, timeout_s: float = 5.0,
                      accept: tuple[str, ...] | None = None) -> str:
    """Send one companion verb over SysEx and return the CDC reply text.

    Chunks are concatenated until the 0x03 one; frames of other ids in between
    (a DAW hello, an audio-route reply) are skipped. An "ERR ..." reply is
    returned, not raised — it is the device's answer, the caller reads it.
    With `accept`, only a complete reply starting with one of those prefixes is
    the answer: any other (another listener's reply on the same port, e.g. the
    hub's) is dropped and reading goes on until the timeout.
    """
    io.drain()
    io.send_sysex(sysex_companion(verb))
    deadline = time.monotonic() + timeout_s
    parts: list[str] = []
    while True:
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise HilError(TIMEOUT, f"no companion reply to {verb!r} within {timeout_s} s",
                           hint="replies come back on the ESP native MIDI IN port only; "
                                "is the firmware new enough for SysEx 0x1F (core 1.12)?")
        frame = io.receive(remaining)
        if frame is None:
            continue
        chunk = parse_companion_chunk(frame)
        if chunk is None:
            continue
        parts.append(chunk[0])
        if chunk[1]:
            text = "".join(parts)
            if accept is None or text.startswith(accept):
                return text
            parts = []


def sysex_bootloader_esp() -> bytes:
    """F0 7D 19 00 F7 — requestBootloader.py SYSEX_DATA."""
    k = _sysex_knowledge()
    return bytes([SYSEX_START, int(k["manufacturer"]), int(k["bootloader"]["id"]),
                  int(k["bootloader"]["esp"]), SYSEX_END])


def parse_query_reply(frame: bytes) -> dict:
    """14-byte reply: F0 7D 1D 10 micSrc adc0 adc1 out0 out1 v0 v1 m0 m1 F7.

    Layout from audio_route_control.cpp send_query_reply()/get_state() and
    hil_sampler_record.py Midi.query_route().
    """
    k = _sysex_knowledge()
    route_id = int(k["audio_route"]["id"])
    query_sub = int(k["audio_route"]["subs"]["query"])
    if len(frame) < 14 or frame[0] != SYSEX_START or frame[2] != route_id \
            or frame[3] != query_sub:
        raise HilError(BAD_SYSEX, "not an audio-route query reply",
                       frame=frame.hex(" ").upper())
    b = frame
    return {"mic_src": b[4], "adc": [b[5], b[6]], "out": [b[7], b[8]],
            "vol": [b[9], b[10]], "mute": [b[11], b[12]]}


# ---------------------------------------------------------------------------
# MidiIO
# ---------------------------------------------------------------------------

def _unregister(handle: Any) -> None:
    """Drop the ALSA sequencer client. ``close_port()`` releases only the port."""
    try:
        handle.delete()
    except Exception:  # noqa: BLE001 - teardown must not mask the caller's result
        pass


def _rtmidi_factories() -> tuple[Callable[[], Any], Callable[[], Any]]:
    try:
        import rtmidi  # type: ignore[import-not-found]
    except ImportError as exc:
        raise HilError(ENV, "python-rtmidi is not installed",
                       hint="pip install python-rtmidi") from exc
    return rtmidi.MidiOut, rtmidi.MidiIn


class MidiIO:
    """One MIDI port pair of a Device: the ESP native port (IN+OUT) or the STM bridge (OUT).

    Replies (echo 0x1D 09, query 0x1D 10) only ever come back on the ESP port, and a
    mode-switch frame sent to the STM bridge is silently dropped — see the hygiene table
    in the spec (§2.4). The role therefore decides which index of the Device is opened.
    """

    def __init__(self, device: Device, role: MidiRole = MidiRole.ESP, *,
                 backend: str = "rtmidi",
                 out_factory: Callable[[], Any] | None = None,
                 in_factory: Callable[[], Any] | None = None,
                 listen: bool = True) -> None:
        if backend != "rtmidi":
            raise HilError(NOT_SUPPORTED, f"midi backend {backend!r} not supported in P0",
                           hint="use backend='rtmidi'")
        self._device = device
        self._role = MidiRole(role)
        # listen=False opens no input port. rtmidi's ALSA input runs its own
        # thread, and close_port() joins it holding the GIL; when the port's
        # device vanishes underneath it (the very next thing a USB mode switch
        # does) that join never returns and every Python thread in the process
        # stops with it -- watchdogs included. A one-way sender has no business
        # owning an input port.
        self._listen = listen
        self._out_factory = out_factory
        self._in_factory = in_factory
        self._out: Any = None
        self._in: Any = None
        self._rx: queue.Queue[bytes] = queue.Queue()
        self._rx_note: queue.Queue[list[int]] = queue.Queue()
        self._denylist: set[tuple[int, int, int]] = self._load_denylist()
        self.port_name: str | None = None

    @staticmethod
    def _load_denylist() -> set[tuple[int, int, int]]:
        k = _sysex_knowledge()
        manu = int(k["manufacturer"])
        denied: set[tuple[int, int, int]] = set()
        for entry in k.get("host_denylist", []):
            cmd, arg = int(entry[0]), int(entry[1])
            denied.add((manu, cmd, arg))
        return denied

    @property
    def role(self) -> MidiRole:
        return self._role

    @property
    def is_open(self) -> bool:
        return self._out is not None

    def _port_info(self) -> MidiPortInfo:
        info = self._device.ports.esp_midi if self._role is MidiRole.ESP \
            else self._device.ports.stm_midi
        if info is None:
            raise HilError(NO_DEVICE, f"{self._device.id} has no {self._role.value} MIDI port",
                           hint="is the device enumerated? run `crosspad-hil devices`")
        return info

    def open(self) -> None:
        if self.is_open:
            return
        info = self._port_info()
        if info.rtmidi_out is None:
            raise HilError(NO_DEVICE, f"{self._role.value} MIDI port has no rtmidi out index",
                           port=info.name)
        out_factory, in_factory = self._out_factory, self._in_factory
        if out_factory is None or in_factory is None:
            rt_out, rt_in = _rtmidi_factories()
            out_factory = out_factory or rt_out
            in_factory = in_factory or rt_in
        out = out_factory()
        out.open_port(info.rtmidi_out)
        self.port_name = info.name
        self._out = out
        if self._listen and self._role is MidiRole.ESP and info.rtmidi_in is not None:
            inp = in_factory()
            inp.open_port(info.rtmidi_in)
            inp.ignore_types(sysex=False, timing=True, active_sense=True)
            inp.set_callback(self._on_message, None)
            self._in = inp

    def close(self) -> None:
        if self._in is not None:
            try:
                self._in.cancel_callback()
                self._in.close_port()
            finally:
                _unregister(self._in)
                self._in = None
        if self._out is not None:
            try:
                self._out.close_port()
            finally:
                _unregister(self._out)
                self._out = None

    def __enter__(self) -> MidiIO:
        self.open()
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    def _on_message(self, event: tuple[list[int], float], data: Any) -> None:
        msg, _delta = event
        if not msg:
            return
        if msg[0] == SYSEX_START:
            self._rx.put(bytes(msg))
        elif msg[0] < 0xF0:
            # Channel voice messages (note on/off, poly aftertouch, CC, ...):
            # a separate queue so a DAW-control test can watch pad presses
            # turn into notes without them getting lost behind sysex traffic
            # (and without a scenario that only wants sysex ever seeing one).
            self._rx_note.put(list(msg))

    def _require_open(self) -> Any:
        if self._out is None:
            raise HilError(BAD_ARGS, "MidiIO not open", hint="call open() first")
        return self._out

    def send_sysex(self, frame: bytes) -> None:
        out = self._require_open()
        frame = bytes(frame)
        if len(frame) < 2 or frame[0] != SYSEX_START or frame[-1] != SYSEX_END:
            raise HilError(BAD_SYSEX, "frame must start with F0 and end with F7",
                           frame=frame.hex(" ").upper())
        if any(b >= 0x80 for b in frame[1:-1]):
            raise HilError(BAD_SYSEX, "SysEx payload bytes must be 0..127",
                           frame=frame.hex(" ").upper())
        if len(frame) >= 5 and (frame[1], frame[2], frame[3]) in self._denylist:
            raise HilError(DENIED_SYSEX,
                           f"{frame.hex(' ').upper()} is never sent from a host",
                           hint="F0 7D 19 01 F7 hangs the STM uartLoop; use STM_DFU over CDC",
                           frame=frame.hex(" ").upper())
        out.send_message(list(frame))

    def send_note(self, on: bool, note: int, vel: int = 100, channel: int = 0) -> None:
        out = self._require_open()
        _check7(note, "note")
        _check7(vel, "vel")
        if not isinstance(channel, int) or channel < 0 or channel > 15:
            raise HilError(BAD_ARGS, f"channel must be 0..15, got {channel!r}")
        if on:
            out.send_message([0x90 | channel, note, vel])
        else:
            out.send_message([0x80 | channel, note, 0])

    def send_realtime(self, status: int) -> None:
        """One MIDI realtime byte (0xF8 clock, 0xFA start, 0xFB continue, 0xFC stop)."""
        if not 0xF8 <= status <= 0xFF:
            raise HilError(BAD_ARGS, f"not a realtime status: {status:#x}")
        self._require_open().send_message([status])

    def drain(self) -> int:
        """Discard queued SysEx frames; returns how many were dropped."""
        n = 0
        while True:
            try:
                self._rx.get_nowait()
                n += 1
            except queue.Empty:
                return n

    def receive(self, timeout_s: float = 1.0) -> bytes | None:
        if self._role is not MidiRole.ESP:
            raise HilError(NOT_SUPPORTED, "the STM bridge MIDI port has no usable IN side",
                           hint="replies only arrive on the ESP native MIDI port")
        self._require_open()
        try:
            return self._rx.get(timeout=max(timeout_s, 0.0))
        except queue.Empty:
            return None

    def receive_note(self, timeout_s: float = 1.0) -> list[int] | None:
        """Next queued channel-voice message (note on/off, aftertouch, CC, ...).

        Kept separate from `receive()` on purpose: existing callers only ever
        want SysEx and must keep seeing exactly that, so note/CC traffic (the
        DAW Control app's pad->note bridge) gets its own queue rather than
        being interleaved into the SysEx one.
        """
        if self._role is not MidiRole.ESP:
            raise HilError(NOT_SUPPORTED, "the STM bridge MIDI port has no usable IN side",
                           hint="replies only arrive on the ESP native MIDI port")
        self._require_open()
        try:
            return self._rx_note.get(timeout=max(timeout_s, 0.0))
        except queue.Empty:
            return None


# ---------------------------------------------------------------------------
# helpers on top of MidiIO
# ---------------------------------------------------------------------------

def query_route(io: MidiIO, timeout_s: float = 1.0) -> dict:
    """Send 0x1D 10 and return the parsed state — hil_sampler_record.py Midi.query_route()."""
    k = _sysex_knowledge()
    route_id = int(k["audio_route"]["id"])
    query_sub = int(k["audio_route"]["subs"]["query"])
    io.drain()
    io.send_sysex(sysex_query())
    deadline = time.monotonic() + timeout_s
    while True:
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise HilError(TIMEOUT, f"no audio-route query reply within {timeout_s} s",
                           hint="replies come back on the ESP native MIDI IN port only")
        frame = io.receive(remaining)
        if frame is None:
            continue
        if len(frame) >= 14 and frame[2] == route_id and frame[3] == query_sub:
            return parse_query_reply(frame)


def _percentile(sorted_ms: list[float], p: float) -> float:
    idx = int(round((len(sorted_ms) - 1) * p))
    return sorted_ms[min(max(idx, 0), len(sorted_ms) - 1)]


def echo_rtt(io: MidiIO, n: int = 20, timeout_s: float = 1.0, *,
             clock: Callable[[], float] = time.monotonic) -> dict:
    """Round-trip time of n echo frames (0x1D 09), one in flight at a time.

    {sent, received, lost, rtt_ms: {p50, p90, max}}; the percentiles are None when nothing
    came back. A reply carrying a different seq than the one in flight is a late echo of an
    already-lost frame and is ignored.
    """
    if n <= 0:
        raise HilError(BAD_ARGS, f"n must be >= 1, got {n}")
    k = _sysex_knowledge()
    route_id = int(k["audio_route"]["id"])
    echo_sub = int(k["audio_route"]["subs"]["echo"])
    io.drain()
    rtts: list[float] = []
    sent = 0
    for i in range(1, n + 1):
        seq = i & 0x0FFFFFFF
        t0 = clock()
        io.send_sysex(sysex_echo(seq))
        sent += 1
        deadline = t0 + timeout_s
        while True:
            remaining = deadline - clock()
            if remaining <= 0:
                break
            frame = io.receive(remaining)
            if frame is None:
                continue
            if len(frame) >= 9 and frame[2] == route_id and frame[3] == echo_sub \
                    and decode_echo_seq(frame) == seq:
                rtts.append((clock() - t0) * 1000.0)
                break
    received = len(rtts)
    if rtts:
        s = sorted(rtts)
        stats: dict[str, float | None] = {"p50": _percentile(s, 0.50),
                                          "p90": _percentile(s, 0.90), "max": s[-1]}
    else:
        stats = {"p50": None, "p90": None, "max": None}
    return {"sent": sent, "received": received, "lost": sent - received, "rtt_ms": stats}
