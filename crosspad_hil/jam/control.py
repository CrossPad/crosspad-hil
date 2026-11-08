"""How the jam talks to the board in each profile, and what it can say there.

In the USB-audio profile there is no CDC: everything goes over the ESP's MIDI
port -- SysEx 0x1D (pads, kit, bounce, preview, codec, echo), the companion
verbs 0x1F (on the firmware's allowlist only: KIT_STATUS, UI_STATE, APP_*,
PAD_*, KIT_LOAD, ...) and realtime bytes (F8 FA FB FC). ``UacLink`` owns the
port: one lock for sending, one thread that reads every reply and hands it to
whoever waits for it (an echo, a companion answer), so a probe never steals
the frames a timed send is waiting for, and every send is stamped and logged.

``Capabilities`` is what ``probe`` found the board can do (companion SEQ? the
bounce SysEx? notes out?), so the script can skip what the firmware does not
have instead of failing on it.
"""
from __future__ import annotations

import queue
import threading
import time
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from typing import Any

from crosspad_hil.jam.actions import ActionLog
from crosspad_hil.midi import (
    decode_echo_seq,
    parse_companion_chunk,
    sysex_audio_route,
    sysex_bounce,
    sysex_companion,
    sysex_companion_notify,
    sysex_echo,
    sysex_kit_load,
    sysex_pad,
    sysex_preview,
    sysex_query,
)

ECHO_SUB = 0x09
ROUTE_ID = 0x1D
COMPANION_ID = 0x1F


@dataclass
class Capabilities:
    companion: bool = False            # companion verbs answer at all
    companion_seq: bool = False        # SEQ through the companion (not on 724d3ad's list)
    sysex_bounce: bool = False         # 1D 0C (a console "BOUNCE ... started|refused")
    note_out: bool = False             # pads send notes to the host
    stm_midi: bool = False             # the STM bridge port exists (passive clock listening)
    fw_sha: str = ""
    sd: dict[str, Any] = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def as_needs(self) -> dict[str, bool]:
        return {"companion": self.companion, "companion_seq": self.companion_seq,
                "sysex_bounce": self.sysex_bounce, "note_out": self.note_out}


class UacLink:
    """The ESP MIDI port in the audio profile: timed sends, echoes, companion verbs."""

    def __init__(self, io: Any, log: ActionLog, *, clock: Callable[[], float] = time.monotonic
                 ) -> None:
        self.io = io
        self.log = log
        self.clock = clock
        self._send_lock = threading.Lock()
        self._halt = threading.Event()
        self._thread: threading.Thread | None = None
        self._echo_seq = 1
        self.echo_sent: dict[int, float] = {}
        self.echo_rtt_ms: list[tuple[float, float]] = []      # (t_send, rtt)
        self.echo_misses = 0
        self.companion_misses = 0
        self._comp_q: queue.Queue[tuple[str, bool]] = queue.Queue()
        self._comp_lock = threading.Lock()
        self.notes_in: list[tuple[float, list[int]]] = []
        self.other: list[bytes] = []

    # -- the reader -----------------------------------------------------------
    def start(self) -> None:
        if self._thread is None:
            self._thread = threading.Thread(target=self._read, name="jam_uac_rx", daemon=True)
            self._thread.start()

    def stop(self) -> None:
        self._halt.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
            self._thread = None

    def _read(self) -> None:
        while not self._halt.is_set():
            try:
                frame = self.io.receive(0.05)
            except Exception:  # noqa: BLE001 - a port that went away ends the reader
                return
            if frame is None:
                note = getattr(self.io, "receive_note", None)
                if note is not None:
                    try:
                        msg = note(0.0)
                    except Exception:  # noqa: BLE001
                        msg = None
                    if msg:
                        self.notes_in.append((self.clock(), msg))
                continue
            t = self.clock()
            if len(frame) >= 9 and frame[2] == ROUTE_ID and frame[3] == ECHO_SUB:
                seq = decode_echo_seq(frame)
                t0 = self.echo_sent.pop(seq, None)
                if t0 is not None:
                    self.echo_rtt_ms.append((t0, (t - t0) * 1000.0))
                continue
            chunk = parse_companion_chunk(frame)
            if chunk is not None:
                self._comp_q.put(chunk)
                continue
            self.other.append(bytes(frame))

    # -- sends ---------------------------------------------------------------
    def _send(self, kind: str, frame: bytes | None, fn: Callable[[], None] | None = None,
              **args: Any) -> Any:
        with self._send_lock:
            t0 = self.clock()
            err = None
            try:
                if fn is not None:
                    fn()
                else:
                    self.io.send_sysex(frame)
            except Exception as e:  # noqa: BLE001 - a lost send is logged, not raised
                err = f"{type(e).__name__}: {e}"
            t1 = self.clock()
        return self.log.emit(kind, t0=t0, t1=t1, ok=err is None, err=err, **args)

    def pad(self, pad: int, vel: int = 100, *, hold_ms: float = 40.0, via: str = "sysex") -> Any:
        if via == "companion":
            return self._send("pad", sysex_companion_notify(f"PAD_PRESS {pad} {vel}"),
                              pad=pad, vel=vel, hold_ms=hold_ms, via="companion")
        return self._send("pad", sysex_pad(True, pad, vel), pad=pad, vel=vel, hold_ms=hold_ms,
                          via="sysex")

    def release(self, pad: int, *, via: str = "sysex") -> Any:
        frame = sysex_companion_notify(f"PAD_RELEASE {pad}") if via == "companion" \
            else sysex_pad(False, pad)
        return self._send("release", frame, pad=pad, via=via)

    def kit(self, kit_id: int, name: str, *, via: str = "sysex") -> Any:
        frame = sysex_companion_notify(f"KIT_LOAD {kit_id}") if via == "companion" \
            else sysex_kit_load(kit_id)
        return self._send("kit", frame, kit=kit_id, name=name, via=via)

    def bounce(self, scene: int, track: int, label: str) -> Any:
        return self._send("bounce", sysex_bounce(scene, track), scene=scene, track=track,
                          label=label, via="sysex")

    def preview(self, pad: int) -> Any:
        return self._send("preview", sysex_preview(pad), pad=pad)

    def route(self, sub: int, *args: int, what: str = "codec") -> Any:
        return self._send("codec", sysex_audio_route(sub, *args), sub=sub, values=list(args),
                          what=what)

    def note(self, on: bool, note: int, vel: int = 100) -> Any:
        return self._send("note_neg" if on else "note_neg_off", None,
                          fn=lambda: self.io.send_note(on, note, vel), note=note, on=on)

    def realtime(self, status: int, what: str) -> Any:
        return self._send("transport" if status != 0xF8 else "clock_tick", None,
                          fn=lambda: self.io.send_realtime(status), what=what)

    def notify(self, verb: str, kind: str = "companion", **args: Any) -> Any:
        return self._send(kind, sysex_companion_notify(verb), verb=verb, **args)

    def echo(self) -> int:
        seq = self._echo_seq
        self._echo_seq += 1
        with self._send_lock:
            self.echo_sent[seq] = self.clock()
            try:
                self.io.send_sysex(sysex_echo(seq))
            except Exception:  # noqa: BLE001
                self.echo_misses += 1
        return seq

    def query_route(self) -> None:
        with self._send_lock:
            self.io.send_sysex(sysex_query())

    def companion(self, verb: str, timeout_s: float = 1.0) -> str | None:
        """One companion request; ``None`` on silence (counted: the watchdog reads it)."""
        with self._comp_lock:
            while not self._comp_q.empty():
                self._comp_q.get_nowait()
            with self._send_lock:
                try:
                    self.io.send_sysex(sysex_companion(verb))
                except Exception:  # noqa: BLE001
                    self.companion_misses += 1
                    return None
            parts: list[str] = []
            deadline = self.clock() + timeout_s
            while True:
                left = deadline - self.clock()
                if left <= 0:
                    self.companion_misses += 1
                    return None
                try:
                    text, last = self._comp_q.get(timeout=left)
                except queue.Empty:
                    self.companion_misses += 1
                    return None
                parts.append(text)
                if last:
                    self.companion_misses = 0
                    return "".join(parts)

    def echo_pending_over(self, seconds: float) -> int:
        now = self.clock()
        return sum(1 for t in self.echo_sent.values() if now - t > seconds)
