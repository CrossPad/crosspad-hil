"""Console session on the STM32 bridge VCP (0x0483:0x5740).

The bridge carries the ESP32's UART0 console out over USB and survives ESP resets
and USB-profile switches, so it is the only port that ever sees the ROM banner.
Two traps, both from platform-idf/tools/hil_kit_churn.py, are encoded here and
nowhere else:

* the bridge emulates esptool's DTR/RTS auto-reset, so the port is opened with
  both deasserted (``serial_open.open_serial``) and the only reset is
  :meth:`Console.reset` (``serial_open.reset_pulse``, which always releases RTS);
* the log carries UTF-8 (perfmon box drawing, the panic banner), so decoding is
  ``errors="replace"`` and the log file is opened the same way — a reader that
  dies on the panic line dies exactly when the log matters.

A background thread owns the port. Lines are numbered (``seq``), kept in a
bounded ring, appended to an optional log file and fed to a
:class:`~crosspad_hil.parsers.ConsoleParser`; the resulting events are kept in a
second ring and handed to ``on_event`` (the daemon turns them into
``console.*`` events).
"""
from __future__ import annotations

import itertools
import re
import sys
import threading
import time
from collections import deque
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import IO, Any

from crosspad_hil.knowledge import load
from crosspad_hil.locks import PortLock
from crosspad_hil.parsers import ConsoleEvent, ConsoleParser
from crosspad_hil.serial_open import open_serial, reset_pulse

READ_CHUNK = 512  # from hil_kit_churn.py Console.run: ser.read(512)


def _in_waiting(ser: Any) -> int | None:
    """Bytes the port already holds, or None when the object cannot say (a test double)."""
    try:
        n = ser.in_waiting
    except Exception:  # noqa: BLE001 - a closed port, or a fake without the property
        return None
    return int(n) if isinstance(n, int) else None
EVENT_RING = 10_000
CONTEXT_LINES = 20


@dataclass
class ReadResult:
    lines: list[tuple[int, str]]
    next_seq: int
    lines_lost: int


@dataclass
class ExpectResult:
    hit: str | None
    rejected: str | None
    seq: int | None
    context: list[str]
    elapsed_s: float


@dataclass
class BootResult:
    complete: bool
    missing: list[str]
    fatal: list[dict]
    errors: list[dict]
    bootloops: int
    seconds: float


class Console:
    """Tail of the ESP console with a ring buffer, parser events and expect()."""

    def __init__(
        self,
        port: str,
        *,
        log_path: Path | None = None,
        ring_size: int = 50_000,
        knowledge: dict | None = None,
        required: list[str] | None = None,
        serial_factory: Callable = open_serial,
        on_event: Callable[[ConsoleEvent], None] | None = None,
        tsv_path: Path | None = None,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self.port = port
        self.log_path: Path | None = Path(log_path) if log_path is not None else None
        # Every line with the host time it arrived (monotonic and wall), so a
        # device log timestamp can be mapped onto the host clock afterwards
        # (jam.clocks.DeviceLogClock). One row per line: seq, t_arr, t_wall, line.
        self.tsv_path: Path | None = Path(tsv_path) if tsv_path is not None else None
        self._clock = clock
        self._knowledge: dict = knowledge if knowledge is not None else load("markers")
        self._parser = ConsoleParser(knowledge=self._knowledge, required=required)
        self._serial_factory = serial_factory
        self._on_event = on_event
        self._ring: deque[tuple[int, str]] = deque(maxlen=ring_size)
        # Parallel to _ring, same length: (t_arr monotonic, t_wall) of each line.
        self._times: deque[tuple[float, float]] = deque(maxlen=ring_size)
        self._tsv: IO[str] | None = None
        self._events: deque[ConsoleEvent] = deque(maxlen=EVENT_RING)
        self._cond = threading.Condition()
        self._seq = 0
        self._lines_lost = 0
        # Baselines for wait_boot(): entries recorded before the last reset()/open()
        # belong to an earlier boot.
        self._fatal_base = 0
        self._error_base = 0
        self._ser: Any = None
        self._log: IO[str] | None = None
        self._lock: PortLock | None = None
        self._thread: threading.Thread | None = None
        self._alive = False

    # ── properties ─────────────────────────────────────────────────────────
    @property
    def seq(self) -> int:
        """Sequence number of the last line ingested (0 before any line)."""
        return self._seq

    # ── lifecycle ──────────────────────────────────────────────────────────
    def open(self, reset: bool = False) -> None:
        """Lock the port, open it with DTR/RTS deasserted, start the reader.

        ``reset=True`` pulses the board through :meth:`reset` *after* the reader
        is running, so the banner is captured from its first byte.
        """
        if self._ser is not None:
            return
        lock = PortLock(self.port, "console")
        lock.acquire()
        try:
            ser = self._serial_factory(self.port, timeout=0.2)
        except Exception:
            lock.release()
            raise
        self._lock = lock
        self._ser = ser
        if self.log_path is not None:
            self.log_path.parent.mkdir(parents=True, exist_ok=True)
            # from hil_kit_churn.py Console.__init__: utf-8 + replace, line-buffered
            self._log = open(  # noqa: SIM115 - closed in close()
                self.log_path, "a", buffering=1, encoding="utf-8", errors="replace"
            )
        if self.tsv_path is not None:
            self.tsv_path.parent.mkdir(parents=True, exist_ok=True)
            self._tsv = open(  # noqa: SIM115 - closed in close()
                self.tsv_path, "a", buffering=1, encoding="utf-8", errors="replace"
            )
        self._alive = True
        self._thread = threading.Thread(
            target=self._read_loop, name=f"console:{self.port}", daemon=True
        )
        self._thread.start()
        if reset:
            self.reset()

    def close(self) -> None:
        """Stop the reader, close the port and the log, release the lock. Idempotent."""
        self._alive = False
        thread, self._thread = self._thread, None
        if thread is not None and thread is not threading.current_thread():
            thread.join(timeout=1.0)
        ser, self._ser = self._ser, None
        if ser is not None:
            try:
                ser.close()
            except Exception:  # noqa: BLE001 - a port that vanished is still closed
                pass
        log, self._log = self._log, None
        if log is not None:
            log.close()
        tsv, self._tsv = self._tsv, None
        if tsv is not None:
            tsv.close()
        lock, self._lock = self._lock, None
        if lock is not None:
            lock.release()

    def reset(self) -> None:
        """esptool-style reset through the bridge; boot tracking starts over."""
        if self._ser is None:
            return
        try:
            reset_pulse(self._ser)  # from hil_smoke.py reset_pulse (always releases RTS)
        finally:
            with self._cond:
                self._parser.reset_boot_tracking()
                snap = self._parser.snapshot()
                self._fatal_base = len(snap["fatals"])
                self._error_base = len(snap["errors"])

    # ── reader thread ──────────────────────────────────────────────────────
    def _read_loop(self) -> None:
        # from hil_kit_churn.py Console.run: read(512), split on b"\n", strip "\r"
        buf = b""
        while self._alive:
            ser = self._ser
            if ser is None:
                break
            try:
                # Read what is there, not a fixed 512: a full-size read waits for
                # 512 bytes or the port timeout (0.2 s), and every line in it would
                # be stamped that late. One byte first blocks until something comes.
                waiting = _in_waiting(ser)
                chunk = ser.read(min(waiting, 4 * READ_CHUNK) if waiting else
                                 (1 if waiting == 0 else READ_CHUNK))
            except Exception:  # noqa: BLE001 - SerialException/OSError: port went away
                break
            if not chunk:
                continue
            t_arr, t_wall = self._clock(), time.time()
            buf += chunk
            while b"\n" in buf:
                raw, buf = buf.split(b"\n", 1)
                self._ingest(raw.decode("utf-8", "replace").rstrip("\r"), t_arr, t_wall)

    def _ingest(self, line: str, t_arr: float | None = None, t_wall: float | None = None) -> None:
        if t_arr is None:
            t_arr, t_wall = self._clock(), time.time()
        with self._cond:
            self._seq += 1
            seq = self._seq
            if self._ring.maxlen is not None and len(self._ring) == self._ring.maxlen:
                self._lines_lost += 1
            self._ring.append((seq, line))
            self._times.append((float(t_arr), float(t_wall or 0.0)))
            if self._log is not None:
                self._log.write(line + "\n")
            if self._tsv is not None:
                self._tsv.write(f"{seq}\t{t_arr:.6f}\t{t_wall or 0.0:.6f}\t"
                                + line.replace("\t", " ") + "\n")
            events = self._parser.feed(seq, line)
            self._events.extend(events)
            self._cond.notify_all()
        if self._on_event is not None:
            for ev in events:
                try:
                    self._on_event(ev)
                except Exception as exc:  # noqa: BLE001 - a callback must not kill the reader
                    print(f"console: on_event raised {exc!r} for {ev.kind}", file=sys.stderr)

    # ── ring access (all callers hold self._cond) ──────────────────────────
    def _oldest_seq(self) -> int:
        return self._ring[0][0] if self._ring else self._seq + 1

    def _from(self, start_seq: int) -> itertools.islice:
        """Iterate ring entries with seq >= start_seq without a Python-level scan."""
        offset = max(0, start_seq - self._oldest_seq())
        return itertools.islice(self._ring, offset, None)

    def _context(self, seq: int) -> list[str]:
        start = max(self._oldest_seq(), seq - CONTEXT_LINES + 1)
        return [line for s, line in self._from(start) if s <= seq]

    def _tail(self) -> list[str]:
        first = max(self._oldest_seq(), self._seq - CONTEXT_LINES + 1)
        return [line for _s, line in self._from(first)]

    def _select(
        self, since_seq: int | None, rx: re.Pattern[str] | None, limit: int
    ) -> tuple[list[tuple[int, str]], int]:
        oldest = self._oldest_seq()
        if since_seq is None:
            start, lost = oldest, 0
        else:
            start, lost = since_seq, max(0, oldest - since_seq)
        out: list[tuple[int, str]] = []
        for s, line in self._from(start):
            if s < start:
                continue
            if rx is not None and not rx.search(line):
                continue
            out.append((s, line))
            if len(out) >= limit:
                break
        return out, lost

    # ── public reads ───────────────────────────────────────────────────────
    def read(
        self,
        since_seq: int | None = None,
        wait_ms: int = 0,
        match: str | None = None,
        limit: int = 2000,
    ) -> ReadResult:
        """Lines with seq >= since_seq (oldest kept line when None), optionally filtered.

        ``wait_ms`` blocks until at least one matching line exists or the wait ends.
        """
        rx = re.compile(match) if match else None
        deadline = time.monotonic() + wait_ms / 1000.0
        with self._cond:
            while True:
                lines, lost = self._select(since_seq, rx, limit)
                if lines or wait_ms <= 0:
                    break
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    break
                self._cond.wait(remaining)
            if lines:
                next_seq = lines[-1][0] + 1
            elif since_seq is not None:
                next_seq = since_seq
            else:
                next_seq = self._seq + 1
        return ReadResult(lines=lines, next_seq=next_seq, lines_lost=lost)

    def start_tsv(self, path: Path) -> None:
        """Write every line from now on -- and the ones already in the ring -- to ``path``
        as ``seq<TAB>t_arr<TAB>t_wall<TAB>line``, for a console someone else opened."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with self._cond:
            if self._tsv is not None:
                return
            self.tsv_path = path
            fh = open(path, "a", buffering=1, encoding="utf-8",  # noqa: SIM115
                      errors="replace")
            for (s, line), (ta, tw) in zip(self._ring, self._times, strict=False):
                fh.write(f"{s}\t{ta:.6f}\t{tw:.6f}\t" + line.replace("\t", " ") + "\n")
            self._tsv = fh

    def read_timed(
        self, since_seq: int | None = None, match: str | None = None, limit: int = 100_000
    ) -> list[tuple[int, float, float, str]]:
        """``(seq, t_arr, t_wall, line)`` for lines with seq >= since_seq: when each line
        reached the host (monotonic and wall clock), for mapping device time to host time."""
        rx = re.compile(match) if match else None
        out: list[tuple[int, float, float, str]] = []
        with self._cond:
            start = self._oldest_seq() if since_seq is None else since_seq
            offset = max(0, start - self._oldest_seq())
            for (s, line), (ta, tw) in zip(itertools.islice(self._ring, offset, None),
                                           itertools.islice(self._times, offset, None),
                                           strict=False):
                if rx is not None and not rx.search(line):
                    continue
                out.append((s, ta, tw, line))
                if len(out) >= limit:
                    break
        return out

    def expect(
        self,
        patterns: list[str],
        reject: Sequence[str] = (),
        timeout_s: float = 30.0,
        *,
        since_seq: int | None = None,
    ) -> ExpectResult:
        """First line (after the call, or from ``since_seq``) matching a pattern.

        Reject patterns are tested first on every line. ``hit``/``rejected`` carry
        the pattern source; ``context`` is the 20 lines ending at the match.
        """
        pats = [(p, re.compile(p)) for p in patterns]
        rejs = [(p, re.compile(p)) for p in reject]
        t0 = time.monotonic()
        deadline = t0 + timeout_s
        with self._cond:
            cursor = since_seq if since_seq is not None else self._seq + 1
            while True:
                for s, line in self._from(cursor):
                    if s < cursor:
                        continue
                    cursor = s + 1
                    for p, rx in rejs:
                        if rx.search(line):
                            return ExpectResult(
                                hit=None, rejected=p, seq=s, context=self._context(s),
                                elapsed_s=time.monotonic() - t0,
                            )
                    for p, rx in pats:
                        if rx.search(line):
                            return ExpectResult(
                                hit=p, rejected=None, seq=s, context=self._context(s),
                                elapsed_s=time.monotonic() - t0,
                            )
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    return ExpectResult(
                        hit=None, rejected=None, seq=None, context=self._tail(),
                        elapsed_s=time.monotonic() - t0,
                    )
                self._cond.wait(remaining)

    def wait_boot(self, timeout_s: float | None = None, settle_s: float = 3.0) -> BootResult:
        """Wait for every required marker (parser ``boot_complete``), then ``settle_s``.

        from hil_smoke.py capture_boot: after the last required marker keep reading
        3 s for late errors; from hil_stability.py BOOT_WINDOW_S = 45 (knowledge
        ``boot_timeout_s``) as the default deadline.
        """
        timeout = (
            float(timeout_s)
            if timeout_s is not None
            else float(self._knowledge.get("boot_timeout_s", 45))
        )
        t0 = time.monotonic()
        deadline = t0 + timeout
        settle_until: float | None = None
        with self._cond:
            while True:
                snap = self._parser.snapshot()
                now = time.monotonic()
                if snap["boot_complete"] and settle_until is None:
                    settle_until = now + settle_s
                if settle_until is not None and now >= settle_until:
                    break
                if now >= deadline:
                    break
                wake = settle_until if settle_until is not None else deadline
                self._cond.wait(max(0.0, min(wake - now, 0.25)))
            snap = self._parser.snapshot()
            fatal = list(snap["fatals"][self._fatal_base:])
            errors = list(snap["errors"][self._error_base:])
        return BootResult(
            complete=bool(snap["boot_complete"]),
            missing=list(snap["missing_markers"]),
            fatal=fatal,
            errors=errors,
            bootloops=int(snap["bootloops"]),
            seconds=time.monotonic() - t0,
        )

    def knowledge_reboot_marker(self) -> str:
        """The console substring that marks a fresh boot (regex-escaped)."""
        return re.escape(str(self._knowledge.get("reboot", "ESP-ROM:")))

    def forget_boot_history(self) -> None:
        """Drop the reboot/marker history accumulated so far.

        Used when the caller itself caused the reset (opening the bridge VCP)
        and must not report it as a device fault.
        """
        self._parser.reset_boot_tracking()
        self._parser.forget_reboots()

    def snapshot(self) -> dict:
        """Parser snapshot plus ``seq``, ``lines_lost``, ``log_path`` and ``port``."""
        with self._cond:
            snap = self._parser.snapshot()
            snap.update(
                seq=self._seq,
                lines_lost=self._lines_lost,
                log_path=str(self.log_path) if self.log_path is not None else None,
                port=self.port,
            )
        return snap

    def events(self, since_seq: int = 0) -> list[ConsoleEvent]:
        """Parser events whose line seq is greater than ``since_seq``."""
        with self._cond:
            return [ev for ev in self._events if ev.seq > since_seq]


__all__ = [
    "CONTEXT_LINES",
    "EVENT_RING",
    "READ_CHUNK",
    "BootResult",
    "Console",
    "ExpectResult",
    "ReadResult",
]
