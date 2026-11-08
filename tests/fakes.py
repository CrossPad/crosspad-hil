"""Hardware stand-ins for the test suite.

FakeSerial mimics the two pyserial ports a CrossPad exposes (ESP CDC 0x303A:0x3456 and the
STM32 bridge VCP 0x0483:0x5740) closely enough for every module above serial_open.py:
scripted command->reply lines, unsolicited console lines via feed(), and a full record of
DTR/RTS assignments so hygiene tests can assert "never asserted DTR".
"""
from __future__ import annotations

import time
from collections import deque
from collections.abc import Callable, Iterable, Sequence
from typing import Any


class FakeSerial:
    """pyserial-shaped fake. Constructible with no arguments (serial_open sets attributes
    then calls open(), exactly like ``serial.Serial()``)."""

    def __init__(
        self,
        script: Sequence[tuple[str, str | list[str]]] = (),
        *,
        timeout: float = 0.2,
    ) -> None:
        self._script: dict[str, list[list[str]]] = {}
        for cmd, reply in script:
            lines = [reply] if isinstance(reply, str) else list(reply)
            self._script.setdefault(cmd, []).append(lines)
        self.port: str | None = None
        self.baudrate: int = 115200
        self.timeout: float | None = timeout
        self.is_open: bool = False
        self.written: list[str] = []
        self.control_history: list[tuple[str, bool]] = []
        self._dtr: bool = False
        self._rts: bool = False
        self._rx: deque[bytes] = deque()
        self._partial: bytes = b""

    # -- control lines -------------------------------------------------
    @property
    def dtr(self) -> bool:
        return self._dtr

    @dtr.setter
    def dtr(self, value: bool) -> None:
        self._dtr = bool(value)
        self.control_history.append(("dtr", bool(value)))

    @property
    def rts(self) -> bool:
        return self._rts

    @rts.setter
    def rts(self, value: bool) -> None:
        self._rts = bool(value)
        self.control_history.append(("rts", bool(value)))

    # -- lifecycle -----------------------------------------------------
    def open(self) -> None:
        self.is_open = True

    def close(self) -> None:
        self.is_open = False

    def flush(self) -> None:
        return None

    def reset_input_buffer(self) -> None:
        self._rx.clear()
        self._partial = b""

    # -- host -> device ------------------------------------------------
    def write(self, data: bytes) -> int:
        text = data.decode("utf-8", errors="replace")
        for raw in text.split("\n"):
            cmd = raw.rstrip("\r")
            if not cmd:
                continue
            self.written.append(cmd)
            queued = self._script.get(cmd)
            if queued is None:
                # Device behaviour: hil_control_handle_cdc() returned false -> echo.
                self._rx.append((cmd + "\r\n").encode())
            else:
                # Repeated script entries for one command are consumed in order;
                # the last one repeats (KIT_STATUS polling needs this).
                lines = queued.pop(0) if len(queued) > 1 else queued[0]
                for line in lines:
                    self._rx.append((line + "\r\n").encode())
        return len(data)

    def feed(self, lines: Iterable[str]) -> None:
        """Push unsolicited lines (console traffic, async replies)."""
        for line in lines:
            self._rx.append((line + "\r\n").encode())

    # -- device -> host ------------------------------------------------
    @property
    def in_waiting(self) -> int:
        return len(self._partial) + sum(len(b) for b in self._rx)

    def _wait_for_data(self) -> bool:
        if self._partial or self._rx:
            return True
        if self.timeout:
            time.sleep(self.timeout)
        return bool(self._partial or self._rx)

    def readline(self) -> bytes:
        if not self._wait_for_data():
            return b""
        if self._partial:
            buf = self._partial
            self._partial = b""
            nl = buf.find(b"\n")
            if nl >= 0:
                self._partial = buf[nl + 1:]
                return buf[: nl + 1]
            return buf + (self._rx.popleft() if self._rx else b"")
        return self._rx.popleft()

    def read(self, size: int = 1) -> bytes:
        if not self._wait_for_data():
            return b""
        buf = self._partial
        self._partial = b""
        while len(buf) < size and self._rx:
            buf += self._rx.popleft()
        out, self._partial = buf[:size], buf[size:]
        return out


# ---------------------------------------------------------------------------
# MIDI fakes (rtmidi-shaped, callback-driven). Used by test_midi/usbmode/ota.
# ---------------------------------------------------------------------------
class FakeMidiIn:
    """rtmidi.MidiIn look-alike. Messages arrive via inject() and reach the callback."""

    def __init__(self, ports: list[str]) -> None:
        self._ports = list(ports)
        self.opened: int | None = None
        self.closed = False
        self.deleted = False
        self.sysex_ignored = True
        self._callback: Callable[[tuple[list[int], float], Any], None] | None = None
        self._data: Any = None

    def get_ports(self) -> list[str]:
        return list(self._ports)

    def open_port(self, index: int) -> None:
        if index >= len(self._ports):
            raise RuntimeError(f"no MIDI in port {index}")
        self.opened = index
        self.closed = False

    def close_port(self) -> None:
        self.closed = True

    def ignore_types(self, sysex: bool = True, timing: bool = True,
                     active_sense: bool = True) -> None:
        self.sysex_ignored = sysex

    def set_callback(self, fn: Callable[[tuple[list[int], float], Any], None],
                     data: Any = None) -> None:
        self._callback = fn
        self._data = data

    def cancel_callback(self) -> None:
        self._callback = None

    def delete(self) -> None:
        self.deleted = True

    def inject(self, msg: list[int], delta: float = 0.0) -> None:
        if self._callback is not None:
            self._callback((list(msg), delta), self._data)


class FakeMidiOut:
    """rtmidi.MidiOut look-alike; every send is recorded and offered to a responder."""

    def __init__(self, ports: list[str], bus: FakeMidiBus) -> None:
        self._ports = list(ports)
        self._bus = bus
        self.opened: int | None = None
        self.closed = False
        self.deleted = False

    def get_ports(self) -> list[str]:
        return list(self._ports)

    def open_port(self, index: int) -> None:
        if index >= len(self._ports):
            raise RuntimeError(f"no MIDI out port {index}")
        self.opened = index
        self.closed = False

    def close_port(self) -> None:
        self.closed = True

    def delete(self) -> None:
        self.deleted = True

    def send_message(self, msg: list[int]) -> None:
        self._bus.sent.append(list(msg))
        if self._bus.responder is not None:
            for reply in self._bus.responder(list(msg)):
                self._bus.inject(reply)


class FakeMidiBus:
    """One out + one in port pair wired together.

    responder(msg) -> list of reply messages delivered synchronously to the IN side
    (before send_message returns) — exactly what a device echo looks like to rtmidi.
    """

    def __init__(self, out_ports: list[str] | None = None, in_ports: list[str] | None = None,
                 responder: Callable[[list[int]], list[list[int]]] | None = None) -> None:
        self.out_ports = out_ports if out_ports is not None else ["Crosspad MIDI 1"]
        self.in_ports = in_ports if in_ports is not None else ["Crosspad MIDI 1"]
        self.responder = responder
        self.sent: list[list[int]] = []
        self.outs: list[FakeMidiOut] = []
        self.ins: list[FakeMidiIn] = []

    def out_factory(self) -> FakeMidiOut:
        o = FakeMidiOut(self.out_ports, self)
        self.outs.append(o)
        return o

    def in_factory(self) -> FakeMidiIn:
        i = FakeMidiIn(self.in_ports)
        self.ins.append(i)
        return i

    def inject(self, msg: list[int]) -> None:
        for i in self.ins:
            if i.opened is not None and not i.closed:
                i.inject(msg)


class AudioDesk:
    """The mixer and codec state of a CrossPad, and the CDC verbs that read and write it.

    Mirrors hil_control.cpp (MIX_LVL, MIX_ROUTE, MIX_MUTE/VOL, MIX_OUT_MUTE/VOL, IN_PAIR)
    and audio_route_control.cpp (AUDIO_STATUS, AUDIO_ADC/OUT/VOL/MUTE/MIC_SRC). Tests set
    ``desk.channels[i]["mute"]`` directly, the way the Mixer app would. ``reply(cmd)``
    returns the reply lines, or None for a verb the desk does not own (the caller echoes
    it, as the device does).

    ``refuse``: verb prefixes answered ``ERR`` (an ADC held by a claim, say).
    ``no_mixer``: a build without the Mixer, which echoes MIX_LVL / MIX_ROUTE.
    ``no_mix_verbs``: a firmware before MIX_MUTE / MIX_VOL / MIX_OUT_*, which echoes them.
    """

    ADC = {"DIFF": 0, "LINE1": 1, "LINE2": 2}
    DAC = {"LINE1": 1, "LINE2": 2, "ALL": 3}

    MIX_VERBS = ("MIX_MUTE", "MIX_VOL", "MIX_OUT_MUTE", "MIX_OUT_VOL")

    def __init__(self, *, no_mixer: bool = False, refuse: tuple[str, ...] = (),
                 no_mix_verbs: bool = False) -> None:
        self.no_mixer = no_mixer
        self.no_mix_verbs = no_mix_verbs
        self.refuse = refuse
        self.channels: list[dict[str, Any]] = [
            {"name": "Sampler", "vol": 0.80, "mute": 0, "route": [1, 1, 1]},
            {"name": "USB", "vol": 1.00, "mute": 0, "route": [1, 1, 0]},
        ]
        self.outputs: list[dict[str, Any]] = [{"vol": 1.0, "mute": 0} for _ in range(3)]
        self.mic_src = 1
        self.adc = [2, 1]
        self.out = [3, 3]
        self.vol = [64, 64]
        self.mute = [0, 0]
        self.in_pair = 0
        self.commands: list[str] = []         # writes only: what a restore had to send

    def state(self) -> dict[str, Any]:
        return {"mic_src": self.mic_src, "adc": list(self.adc), "out": list(self.out),
                "vol": list(self.vol), "mute": list(self.mute), "in_pair": self.in_pair,
                "routes": [list(c["route"]) for c in self.channels],
                "ch_mute": [c["mute"] for c in self.channels]}

    def reply(self, cmd: str) -> list[str] | None:
        verb, *args = cmd.split()
        if verb == "MIX_LVL" and not self.no_mixer:
            lines = [f"MIXLVL: ch={i} name={c['name']} pk=0.0000/0.0000 vol={c['vol']:.2f} "
                     f"mute={c['mute']} route={''.join(str(b) for b in c['route'])}"
                     for i, c in enumerate(self.channels)]
            lines += [f"MIXLVL: out={i} pk=0.0000/0.0000 vol={o['vol']:.2f} mute={o['mute']}"
                      for i, o in enumerate(self.outputs)]
            return [*lines, "MIXLVL: end"]
        if verb == "AUDIO_STATUS":
            def lst(v: list[int]) -> str:
                return "[" + ",".join(str(x) for x in v) + "]"
            return [f"AUDIO: mic_src={self.mic_src} adc={lst(self.adc)} out={lst(self.out)} "
                    f"vol={lst(self.vol)} mute={lst(self.mute)}"]
        if verb == "IN_PAIR":
            if args:
                self.commands.append(cmd)
                self.in_pair = int(args[0])
            return [f"INPAIR: ok=1 pair={self.in_pair} 0=Mic 1=Line"]
        mine = verb in ("MIX_ROUTE", "AUDIO_ADC", "AUDIO_OUT", "AUDIO_VOL", "AUDIO_MUTE",
                        "AUDIO_MIC_SRC", *self.MIX_VERBS)
        if not mine or (verb == "MIX_ROUTE" and self.no_mixer):
            return None
        if verb in self.MIX_VERBS and (self.no_mixer or self.no_mix_verbs):
            return None
        if any(cmd.startswith(r) for r in self.refuse):
            return ["ERR"]
        self.commands.append(cmd)
        if verb == "MIX_ROUTE":
            ch, out, on = (int(a) for a in args)
            self.channels[ch]["route"][out] = on
        elif verb in self.MIX_VERBS:
            i, v = (int(a) for a in args)
            table = self.outputs if verb.startswith("MIX_OUT") else self.channels
            if i >= len(table) or (verb.endswith("MUTE") and v not in (0, 1)) or not 0 <= v <= 100:
                self.commands.pop()
                return [f"ERR {verb} bad args"]
            if verb.endswith("MUTE"):
                table[i]["mute"] = v
            else:
                table[i]["vol"] = v / 100.0
        elif verb == "AUDIO_ADC":
            self.adc[int(args[0])] = self.ADC[args[1]]
        elif verb == "AUDIO_OUT":
            self.out[int(args[0])] = self.DAC[args[1]]
        elif verb == "AUDIO_VOL":
            self.vol[int(args[0])] = int(args[1])
        elif verb == "AUDIO_MUTE":
            self.mute[int(args[0])] = int(args[1])
        elif verb == "AUDIO_MIC_SRC":
            self.mic_src = int(args[0])
        return ["OK"]
