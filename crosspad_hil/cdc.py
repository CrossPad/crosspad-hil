"""CDC control link on the ESP native USB CDC (0x303A:0x3456), default profile.

Ported from platform-idf/tools/hil_kit_churn.py ``class Cdc``. The design that
looks obvious — lock, write, sleep, read — starves everyone else on the port
(measured there: 8 pad hits/s requested, 1.8 delivered while a status poll ran).
So one reader thread owns the port and nothing else reads it. A request registers
the reply prefix it wants *before* writing, writers hold a lock only for the
``write()`` itself, and unrelated traffic is simply not theirs.

Three device behaviours the client must know (main/main.cpp ``cdc_dispatch_one``,
main/hil_control.cpp):

* a command no handler matched is **echoed back verbatim** — that is the only
  "unknown verb" signal, so a waiter that sees its own line raises
  ``UNKNOWN_VERB``;
* ``OK`` is the reply of ``PAD_PRESS`` as much as of ``KIT_LOAD`` — a prefix
  match is never proof that *your* command was obeyed; ask a state verb;
* ``app_queue`` is 64 deep and drops on overflow; ``CDC_STATS`` counts the
  drops, and :meth:`CdcLink.burst` reads it before and after.
"""
from __future__ import annotations

import re
import threading
import time
from collections import deque
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Any

from serial import SerialException

from crosspad_hil.errors import BAD_ARGS, NO_DEVICE, NOT_SUPPORTED, TIMEOUT, UNKNOWN_VERB, HilError
from crosspad_hil.knowledge import load
from crosspad_hil.locks import PortLock
from crosspad_hil.parsers import parse_cdc_reply
from crosspad_hil.serial_open import open_serial

OPEN_SETTLE_S = 0.4  # from hil_kit_churn.py Cdc.__init__: sleep(0.4) then reset_input_buffer()
READ_CHUNK = 512  # from hil_kit_churn.py Cdc._read_loop: ser.read(512)
UNMATCHED_RING = 200
# A reply that continues on lines starting with a space (RESET_LOG's " #1 ..."
# rows, COREDUMP_INFO's " bt:" / " exc:") sends them right behind its first
# line; this long without one ends it.
FOLLOW_IDLE_S = 0.15
BURST_SETTLE_S = 0.2  # let the prio-1 main loop drain app_queue before CDC_STATS
# from hil_control.cpp: "CDCSTATS: rx=%u cmds=%u drop=%u"
CDC_DROP_RE = re.compile(r"CDCSTATS:.*\bdrop=(\d+)")
_ERR_LINE_RE = re.compile(r"^(?:ERR|[A-Z][A-Z0-9]*_ERR(?:OR)?)(?:\s|$)")

DEFAULT_TIMEOUT_S = 2.0   # a CDC verb answers within this unless cdc.yaml says otherwise


@dataclass
class Reply:
    line: str
    parsed: dict | None
    rtt_ms: float
    extra_lines: list[str]
    # The reply's own continuation lines (sent starting with a space), stripped.
    more_lines: list[str] = field(default_factory=list)


@dataclass
class BurstResult:
    sent: int
    seconds: float
    drops_before: int
    drops_after: int
    dropped: int


def is_err_line(line: str) -> bool:
    """``ERR …`` or ``<FAMILY>_ERR …`` / ``<FAMILY>_ERROR …`` — a device refusing a verb."""
    return bool(_ERR_LINE_RE.match(line))


def resolve_expect(verbs: dict[str, dict], cmd: str, expect: str | None) -> tuple[str, ...] | None:
    """Reply prefixes accepted for ``cmd``; None for a verb that never replies.

    ``expect=None`` derives the prefix from ``knowledge/cdc.yaml`` (first word).
    "ERR ..." is a valid answer to every verb, not only the OK ones.
    """
    if expect is None:
        verb = cmd.strip().split(" ", 1)[0]
        entry = verbs.get(verb)
        if entry is None:
            raise HilError(
                UNKNOWN_VERB,
                f"{verb} is not in knowledge/cdc.yaml",
                hint="pass expect=<reply prefix> (or expect='OK') to send it anyway",
                verb=verb,
                cmd=cmd.strip(),
            )
        reply_kind = entry.get("reply")
        if reply_kind is None:
            return None
        expect = str(entry.get("prefix", "")) if reply_kind == "multi" else str(reply_kind)
    return ("OK", "ERR") if expect == "OK" else (expect, "ERR")


def verb_timeout_s(verbs: dict[str, dict], verb: str, timeout_s: float | None) -> float:
    """The caller's timeout, else the verb's own: one that blocks the CDC loop
    (STM_FW_UPDATE reflashes the STM, ~10 s) says how long in cdc.yaml."""
    if timeout_s is not None:
        return float(timeout_s)
    return float((verbs.get(verb) or {}).get("timeout_s", DEFAULT_TIMEOUT_S))


@dataclass
class _Waiter:
    """One outstanding request: what it waits for and what it has collected."""

    prefixes: tuple[str, ...]
    echo: str
    multi: bool = False
    end: str | None = None
    follow: bool = False          # take the continuation lines behind the reply
    lines: list[str] = field(default_factory=list)
    extra: list[str] = field(default_factory=list)
    more: list[str] = field(default_factory=list)
    last_t: float = 0.0
    done: bool = False
    echoed: bool = False
    event: threading.Event = field(default_factory=threading.Event)

    def offer(self, line: str, now: float) -> bool:
        """Claim ``line`` if it belongs to this request. Returns True when claimed."""
        if self.done:
            return False
        if line == self.echo:
            self.echoed = True
            self.done = True
            self.event.set()
            return True
        if not self.multi:
            if line.startswith(self.prefixes) or is_err_line(line):
                self.lines.append(line)
                self.last_t = now
                self.done = True
                self.event.set()
                return True
            return False
        if not self.lines:
            # A refusal comes instead of the first row, never among them — after
            # that an interleaved ERR belongs to some other command.
            if is_err_line(line):
                self.lines.append(line)
                self.done = True
                self.event.set()
                return True
            if not line.startswith(self.prefixes):
                return False
        elif self.end is not None and not (
            line.startswith(self.prefixes) or line.startswith(self.end)
        ):
            return False
        self.lines.append(line)
        self.last_t = now
        if self.end is not None and line.startswith(self.end):
            self.done = True
            self.event.set()
        return True


# The ESP is gone from the bus for ~2-4 s after a reset; 15 s covers a slow hub.
ENUM_WAIT_S = 15.0
# A full boot takes ~8 s; 25 s covers a slow SD mount.
READY_WAIT_S = 25.0


class CdcLink:
    """One reader thread, prefix waiters, a write lock held only for write()."""

    def __init__(
        self,
        port: str,
        *,
        serial_factory: Callable = open_serial,
        knowledge: dict | None = None,
        purpose: str = "cdc",
    ) -> None:
        self.port = port
        self.purpose = purpose
        self._serial_factory = serial_factory
        self._knowledge: dict = knowledge if knowledge is not None else load("cdc")
        self._verbs: dict[str, dict] = dict(self._knowledge.get("verbs", {}))
        self._ser: Any = None
        self._lock: PortLock | None = None
        self._wlock = threading.Lock()
        self._rlock = threading.Lock()
        self._waiters: list[_Waiter] = []
        # Who claimed the line just before: a continuation line is theirs.
        self._last_owner: _Waiter | None = None
        self._unmatched: deque[str] = deque(maxlen=UNMATCHED_RING)
        self._rx_bytes = 0
        self._alive = False
        self._thread: threading.Thread | None = None
        self._port_resolver: Callable[[], str | None] | None = None

    # ── lifecycle ──────────────────────────────────────────────────────────
    def open(self, wait_s: float = ENUM_WAIT_S,
             port_resolver: Callable[[], str | None] | None = None) -> None:
        """Lock the port, open it with DTR/RTS deasserted, start the reader.

        The ESP re-enumerates after every reset, and opening the STM bridge VCP
        is itself a reset on this hardware, so the CDC node can be missing for a
        few seconds. Rather than failing on a race the caller cannot see, wait
        for it to come back.
        """
        if self._ser is not None:
            return
        self._port_resolver = port_resolver
        lock = PortLock(self.port, self.purpose)
        lock.acquire()
        deadline = time.monotonic() + max(0.0, wait_s)
        last: Exception | None = None
        ser = None
        while True:
            try:
                ser = self._serial_factory(self.port, timeout=0.2)
                break
            except (OSError, SerialException) as exc:
                last = exc
                if port_resolver is not None:
                    # A re-enumerated board can come back on a different node
                    # (/dev/ttyACM1 -> /dev/ttyACM2), so ask who it is now.
                    fresh = port_resolver()
                    if fresh and fresh != self.port:
                        self._lock_port(fresh, lock)
                        lock = self._lock
                        continue
                if time.monotonic() >= deadline:
                    lock.release()
                    raise HilError(
                        NO_DEVICE,
                        f"{self.port} did not appear within {wait_s:.0f}s",
                        hint="the ESP re-enumerates after a reset; "
                             "run `crosspad-hil devices` to see what is attached now",
                        port=self.port,
                        cause=repr(last),
                    ) from None
                time.sleep(0.25)
            except Exception:
                lock.release()
                raise
        self._lock = lock
        self._ser = ser
        # from hil_kit_churn.py Cdc.__init__: let the enumeration settle, drop stale bytes
        if OPEN_SETTLE_S > 0:
            time.sleep(OPEN_SETTLE_S)
        try:
            ser.reset_input_buffer()
        except Exception:  # noqa: BLE001 - not every serial-like object has one
            pass
        self._alive = True
        self._thread = threading.Thread(
            target=self._read_loop, name=f"cdc:{self.port}", daemon=True
        )
        self._thread.start()

    def _lock_port(self, new_port: str, old_lock: PortLock) -> None:
        """Move the port lock to a node the board re-enumerated onto."""
        old_lock.release()
        self.port = new_port
        lock = PortLock(new_port, self.purpose)
        lock.acquire()
        self._lock = lock

    def _reopen_on(self, new_port: str) -> None:
        """Close this link and bring it up again on another node."""
        resolver = self._port_resolver
        self.close()
        self.port = new_port
        self.open(port_resolver=resolver)

    def wait_ready(self, timeout_s: float = READY_WAIT_S) -> None:
        """Block until the firmware answers on this link.

        The port node appears as soon as USB enumerates, but the application
        that serves CDC verbs is only up once the boot finishes (~8 s on this
        board). Without this, the first verb after a reset times out.
        """
        deadline = time.monotonic() + max(0.0, timeout_s)
        last: HilError | None = None
        while True:
            try:
                # APP_LIST rather than CDC_STATS: the CDC handler answers as soon
                # as USB is up, but the app registry fills in later, and a scenario
                # that reads an empty launcher would report "no apps" instead of
                # waiting. An empty list means "not ready yet".
                reply = self.transact("APP_LIST", timeout_s=1.5)
                if (reply.parsed or {}).get("apps"):
                    return
                # Answering, with no apps: still booting -- or a boot that never
                # registers any (USB transfer mode); the deadline holds either way.
                last = HilError(TIMEOUT, "APP_LIST answered with no apps", line=reply.line)
            except HilError as exc:
                last = exc
                # The node we hold may be the pre-reset one, still open but dead
                # (the board came back on a different node). Follow it.
                if self._port_resolver is not None:
                    fresh = self._port_resolver()
                    if fresh and fresh != self.port:
                        self._reopen_on(fresh)
                        continue
            if time.monotonic() >= deadline:
                raise HilError(
                    TIMEOUT,
                    f"{self.port} opened but the app registry was not ready "
                    f"within {timeout_s:.0f}s",
                    hint="the board is still booting, in USB transfer mode (no apps), or "
                         "in the USB-audio profile where CDC does not exist",
                    port=self.port,
                    cause=last.message,
                ) from None
            time.sleep(0.5)

    def reader_alive(self) -> bool:
        """Open and still reading. False once the port went away under the reader
        (the board reset or re-enumerated): the link must be reopened."""
        thread = self._thread
        return self._ser is not None and thread is not None and thread.is_alive()

    def close(self) -> None:
        """Stop the reader, close the port, release the lock. Idempotent."""
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
        lock, self._lock = self._lock, None
        if lock is not None:
            lock.release()
        with self._rlock:
            for w in self._waiters:
                w.done = True
                w.event.set()
            self._waiters.clear()

    # ── reader thread ──────────────────────────────────────────────────────
    def _read_loop(self) -> None:
        # from hil_kit_churn.py Cdc._read_loop
        buf = b""
        while self._alive:
            ser = self._ser
            if ser is None:
                break
            try:
                chunk = ser.read(READ_CHUNK)
            except Exception:  # noqa: BLE001 - SerialException/OSError: port went away
                break
            if not chunk:
                continue
            self._rx_bytes += len(chunk)
            buf += chunk
            while b"\n" in buf:
                raw, buf = buf.split(b"\n", 1)
                text = raw.decode("utf-8", "replace")
                line = text.strip()
                if line:
                    self._dispatch(line, cont=text[:1] in (" ", "\t"))

    def _dispatch(self, line: str, cont: bool = False) -> None:
        """Hand ``line`` to the request it answers. ``cont``: it arrived starting
        with a space -- the firmware's continuation of the line before it."""
        now = time.monotonic()
        with self._rlock:
            owner = self._last_owner
            if cont and owner is not None and owner.follow and owner in self._waiters \
                    and not (owner.multi and owner.done):
                (owner.lines if owner.multi else owner.more).append(line)
                owner.last_t = now
                return
            for w in self._waiters:
                if w.offer(line, now):
                    self._last_owner = w
                    return
            self._last_owner = None
            self._unmatched.append(line)
            for w in self._waiters:
                if not w.done:
                    w.extra.append(line)

    def _register(self, w: _Waiter) -> None:
        with self._rlock:
            self._waiters.append(w)

    def _unregister(self, w: _Waiter) -> None:
        with self._rlock:
            try:
                self._waiters.remove(w)
            except ValueError:
                pass

    # ── writes ─────────────────────────────────────────────────────────────
    def send(self, cmd: str) -> None:
        """Fire and forget — the pad stimulus, which must never stall."""
        ser = self._ser
        if ser is None:
            raise HilError(BAD_ARGS, "CdcLink is not open", hint="call open() first",
                           port=self.port)
        with self._wlock:
            try:
                ser.write((cmd + "\n").encode("utf-8"))
            except Exception:  # noqa: BLE001 - from hil_kit_churn.py Cdc.send: never stall
                pass

    def _timeout_error(
        self, cmd: str, prefixes: tuple[str, ...], timeout_s: float, rx_bytes: int
    ) -> HilError:
        if rx_bytes == 0:
            hint = (
                "no bytes at all arrived on the CDC port: the device may be in the "
                "MIDI+UAC2 profile (no CDC endpoint) — usbmode.set mode=default — or it "
                "is rebooting; check `crosspad-hil devices`"
            )
        else:
            hint = (
                f"the device is talking but never answered {'/'.join(prefixes)}; another "
                "session may have consumed the reply, or the verb does not reply in this "
                "state — unmatched() shows what did arrive"
            )
        return HilError(
            TIMEOUT,
            f"no {'/'.join(prefixes)} reply to {cmd!r} within {timeout_s:g}s",
            hint=hint,
            cmd=cmd,
            expect=list(prefixes),
            timeout_s=timeout_s,
            rx_bytes=rx_bytes,
        )

    def _echo_error(self, cmd: str) -> HilError:
        return HilError(
            UNKNOWN_VERB,
            f"device echoed {cmd!r}: no CDC handler matched it",
            hint=(
                "the verb is misspelled, absent from this firmware, or not available in "
                "this USB profile; knowledge/cdc.yaml lists the catalog"
            ),
            cmd=cmd,
        )

    # ── request / reply ────────────────────────────────────────────────────
    def _timeout_for(self, verb: str, timeout_s: float | None) -> float:
        return verb_timeout_s(self._verbs, verb, timeout_s)

    def transact(self, cmd: str, expect: str | None = None,
                 timeout_s: float | None = None, follow: bool | None = None) -> Reply:
        """Send ``cmd`` and return the first reply line starting with ``expect``.

        ``expect=None`` derives the prefix from ``knowledge/cdc.yaml`` for the verb
        (first word). ``"OK"`` verbs accept ``OK`` or ``ERR …``. An echo of the sent
        line raises ``UNKNOWN_VERB``; silence raises ``TIMEOUT`` with a hint that
        names UAC2 mode when not a single byte arrived.

        ``follow``: also collect the lines the reply continues on (sent starting
        with a space) into ``Reply.more_lines``, until FOLLOW_IDLE_S passes
        without one. ``None`` takes the verb's ``continuation`` from cdc.yaml.
        """
        cmd = cmd.strip()
        verb = cmd.split(" ", 1)[0]
        timeout_s = self._timeout_for(verb, timeout_s)
        if follow is None:
            follow = bool((self._verbs.get(verb) or {}).get("continuation"))
        prefixes = resolve_expect(self._verbs, cmd, expect)
        if prefixes is None:
            self.send(cmd)
            return Reply(line="", parsed=None, rtt_ms=0.0, extra_lines=[])
        waiter = _Waiter(prefixes=prefixes, echo=cmd, follow=follow)
        rx0 = self._rx_bytes
        # Registered before the write so an instant reply cannot be missed.
        self._register(waiter)
        t0 = time.monotonic()
        rtt_ms = 0.0
        try:
            self.send(cmd)
            waiter.event.wait(timeout_s)
            rtt_ms = (time.monotonic() - t0) * 1000.0
            if follow and waiter.lines:
                deadline = t0 + timeout_s
                while time.monotonic() < deadline:
                    with self._rlock:
                        quiet = time.monotonic() - waiter.last_t
                    if quiet >= FOLLOW_IDLE_S:
                        break
                    time.sleep(0.02)
        finally:
            self._unregister(waiter)
        if waiter.echoed:
            raise self._echo_error(cmd)
        if not waiter.lines:
            raise self._timeout_error(cmd, prefixes, timeout_s, self._rx_bytes - rx0)
        line = waiter.lines[0]
        with self._rlock:
            more, extra = list(waiter.more), list(waiter.extra)
        return Reply(line=line, parsed=parse_cdc_reply(line), rtt_ms=rtt_ms, extra_lines=extra,
                     more_lines=more)

    def transact_multi(
        self,
        cmd: str,
        expect: str | tuple[str, ...],
        end: str | None = None,
        idle_ms: int = 200,
        timeout_s: float = 3.0,
        require_end: bool = False,
    ) -> list[str]:
        """Multi-line reply: from the first ``expect`` line to ``end`` (inclusive).

        ``require_end``: a reply cut off before its ``end`` line is a TIMEOUT, not the
        lines that came -- for a reply whose rest would otherwise land, late, in whatever
        is asked next (a listing).

        With ``end=None`` the reply ends after ``idle_ms`` without a new line
        (``ENC_GROUP``). With ``end`` given only lines starting with ``expect`` or
        ``end`` are kept, so interleaved pad ``OK``s do not land in the result.
        """
        cmd = cmd.strip()
        # A reply may mix prefixes: APP_VERSIONS ends its APPVER: rows with one
        # LIBVER: line. Keeping a single expected prefix silently dropped it.
        prefixes = (expect,) if isinstance(expect, str) else tuple(expect)
        waiter = _Waiter(prefixes=prefixes, echo=cmd, multi=True, end=end, follow=True)
        rx0 = self._rx_bytes
        self._register(waiter)
        deadline = time.monotonic() + timeout_s
        idle_s = idle_ms / 1000.0
        try:
            self.send(cmd)
            while True:
                if end is not None:
                    waiter.event.wait(0.02)
                else:
                    time.sleep(0.02)
                now = time.monotonic()
                with self._rlock:
                    echoed = waiter.echoed
                    done = waiter.done
                    n_lines = len(waiter.lines)
                    last_t = waiter.last_t
                if echoed:
                    raise self._echo_error(cmd)
                if done:
                    break
                if end is None and n_lines and now - last_t >= idle_s:
                    break
                if now >= deadline:
                    break
        finally:
            self._unregister(waiter)
        with self._rlock:
            lines = list(waiter.lines)
            done = waiter.done
        if not lines:
            raise self._timeout_error(cmd, prefixes, timeout_s, self._rx_bytes - rx0)
        if require_end and end is not None and not done and not is_err_line(lines[0]):
            # Cut off before its end line: the rest arrives late, into whatever is asked
            # next (a listing of SONGS came back as a listing of .elements).
            raise HilError(TIMEOUT, f"{cmd.split(' ', 1)[0]}: {len(lines)} line(s), no {end!r} "
                           f"within {timeout_s:.1f}s", hint="the reply was cut off; ask again "
                           "once the device is less busy", cmd=cmd, lines=len(lines))
        if is_err_line(lines[0]):
            raise HilError(
                NOT_SUPPORTED,
                f"{cmd.split(' ', 1)[0]}: device replied {lines[0]}",
                hint="the device refused the command; the reply names why",
                cmd=cmd,
                reply=lines[0],
            )
        return lines

    # ── burst ──────────────────────────────────────────────────────────────
    def _drops(self) -> int:
        reply = self.transact("CDC_STATS", expect="CDCSTATS:")
        m = CDC_DROP_RE.search(reply.line)
        return int(m.group(1)) if m else 0

    def burst(self, cmds: Sequence[str], rate_hz: float) -> BurstResult:
        """Send ``cmds`` paced at ``rate_hz`` and account app_queue drops via CDC_STATS."""
        if rate_hz <= 0:
            raise HilError(BAD_ARGS, f"rate_hz must be > 0, got {rate_hz!r}", rate_hz=rate_hz)
        before = self._drops()
        period = 1.0 / rate_hz
        t0 = time.monotonic()
        for i, cmd in enumerate(cmds):
            target = t0 + i * period
            now = time.monotonic()
            if target > now:
                time.sleep(target - now)
            self.send(cmd)
        seconds = time.monotonic() - t0
        if BURST_SETTLE_S > 0:
            time.sleep(BURST_SETTLE_S)
        after = self._drops()
        return BurstResult(
            sent=len(cmds),
            seconds=seconds,
            drops_before=before,
            drops_after=after,
            dropped=max(0, after - before),
        )

    def unmatched(self) -> list[str]:
        """Lines no request claimed (bounded to the last 200); not cleared."""
        with self._rlock:
            return list(self._unmatched)


__all__ = [
    "BURST_SETTLE_S",
    "CDC_DROP_RE",
    "OPEN_SETTLE_S",
    "READ_CHUNK",
    "UNMATCHED_RING",
    "BurstResult",
    "CdcLink",
    "Reply",
    "resolve_expect",
    "verb_timeout_s",
]
