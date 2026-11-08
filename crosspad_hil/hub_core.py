"""What the hub decides: who is connected, who holds the port, which path a verb takes.

No sockets and no rtmidi here — the server hands in sessions, `find_board`
says what is attached now, `link_factory` opens a CdcLink and `companion`
runs one 0x1F verb over the hub's own board MIDI. Shared requests are
serialised: two clients' "OK" replies are indistinguishable on the wire.

An exclusive op (wire ops `lease` / `release`) is the short hold a client
takes for anything that leaves line mode or resets the board: the hub closes
its own link so the holder can open the port itself, and refuses everyone
else with PORT_BUSY until it ends. It is not the bench lease (bench.py): that
one says which session is testing on the board and is checked by the clients
before they ever reach the hub.
"""
from __future__ import annotations

import dataclasses
import threading
import time
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from typing import Any

from crosspad_hil.cdc import is_err_line, resolve_expect
from crosspad_hil.devices import Device, UsbMode
from crosspad_hil.errors import (
    BAD_ARGS,
    NO_CDC_IN_AUDIO_MODE,
    NO_DEVICE,
    NOT_SUPPORTED,
    PORT_BUSY,
    TIMEOUT,
    UNKNOWN_VERB,
    HilError,
)
from crosspad_hil.knowledge import load
from crosspad_hil.parsers import parse_cdc_reply

CDC_IDLE_S = 5.0
# discover() lists serial, MIDI and audio devices: too slow to run on every verb of a
# 5 Hz poll. A dead link always rediscovers at once.
BOARD_CACHE_S = 1.0
COMPANION_TIMEOUT_S = 5.0
MULTI_IDLE_MS = 200
MULTI_TIMEOUT_S = 3.0


@dataclasses.dataclass
class _Exclusive:
    session_id: int
    client: str
    purpose: str
    pid: int
    since: float


class HubCore:
    def __init__(self, *, find_board: Callable[[], Device | None],
                 link_factory: Callable[[str], Any],
                 companion: Callable[[str, float], str] | None = None,
                 clock: Callable[[], float] = time.monotonic) -> None:
        self._find_board = find_board
        self._link_factory = link_factory
        self._companion = companion
        self._clock = clock
        self._knowledge_verbs: dict[str, dict] = dict(load("cdc").get("verbs", {}))
        self._sessions: dict[int, Any] = {}
        self._exclusive: _Exclusive | None = None
        self._link: Any = None
        self._q = threading.Lock()
        self._state = threading.Lock()
        self._idle_since: float | None = clock()
        self._board_seen: Device | None = None
        self._board_at: float | None = None

    # -- sessions ----------------------------------------------------------

    def connect(self, session: Any) -> None:
        with self._state:
            self._sessions[session.id] = session
            self._idle_since = None

    def disconnect(self, session: Any) -> None:
        with self._state:
            self._sessions.pop(session.id, None)
            if self._exclusive is not None and self._exclusive.session_id == session.id:
                self._exclusive = None
            if not self._sessions:
                self._idle_since = self._clock()

    def tick(self) -> None:
        with self._state:
            idle = self._idle_since is not None and \
                self._clock() - self._idle_since >= CDC_IDLE_S
        if idle:
            with self._q:
                self._close_link()

    def close(self) -> None:
        with self._q:
            self._close_link()

    # -- dispatch ----------------------------------------------------------

    def handle(self, session: Any, request: dict) -> Any:
        op = request.get("op")
        args = request.get("args") or {}
        if op == "status":
            return self._status()
        if op == "lease":
            return self._grant(session, str(args.get("purpose", "")))
        if op == "release":
            return self._release(session)
        self._check_exclusive(session)
        if op == "transact":
            timeout_s = args.get("timeout_s")
            return self._transact(session, str(args["cmd"]), args.get("expect"),
                                  None if timeout_s is None else float(timeout_s),
                                  args.get("follow"))
        if op == "transact_multi":
            return self._transact_multi(session, args)
        if op == "send":
            with self._port(session):
                self._cdc_link("send").send(str(args["cmd"]))
            return {"ok": True}
        if op == "burst":
            with self._port(session):
                r = self._cdc_link("burst").burst([str(c) for c in args["cmds"]],
                                                  float(args["rate_hz"]))
            return dataclasses.asdict(r)
        raise HilError(BAD_ARGS, f"unknown hub op {op!r}")

    # -- exclusive op --------------------------------------------------------

    def _holder(self) -> dict | None:
        ex = self._exclusive
        if ex is None:
            return None
        return {"client": ex.client, "purpose": ex.purpose, "pid": ex.pid, "since": ex.since}

    def _busy(self) -> HilError:
        h = self._holder() or {}
        return HilError(PORT_BUSY,
                        f"the board's port is held by {h.get('client')} for {h.get('purpose')} "
                        f"(pid {h.get('pid')})",
                        hint="wait for it to finish; `crosspad-hil midi hub status` shows it",
                        holder=h)

    @contextmanager
    def _port(self, session: Any) -> Iterator[None]:
        """The port, for a session nobody else's exclusive op shuts out. Checked again
        once the port is ours: an op granted while this one waited for it must win."""
        with self._q:
            self._check_exclusive(session)
            yield

    def _check_exclusive(self, session: Any) -> None:
        with self._state:
            if self._exclusive is not None and self._exclusive.session_id != session.id:
                raise self._busy()

    def _grant(self, session: Any, purpose: str) -> dict:
        with self._state:
            if self._exclusive is not None and self._exclusive.session_id != session.id:
                raise self._busy()
            self._exclusive = _Exclusive(session.id, session.client, purpose or session.purpose,
                                         session.pid, time.time())
        with self._q:
            self._close_link()
        return {"ok": True}

    def _release(self, session: Any) -> dict:
        with self._state:
            if self._exclusive is not None and self._exclusive.session_id == session.id:
                self._exclusive = None
        return {"ok": True}

    # -- board -------------------------------------------------------------

    def _status(self) -> dict:
        dev = self._discover(fresh=True)
        with self._state:
            clients = [{"client": s.client, "purpose": s.purpose, "pid": s.pid}
                       for s in self._sessions.values()]
            holder = self._holder()
        return {"running": True, "board": None if dev is None else dev.id,
                "usb_mode": UsbMode.UNKNOWN.value if dev is None else dev.usb_mode.value,
                "cdc_open": self._link is not None, "clients": clients, "lease": holder}

    def _close_link(self) -> None:
        link, self._link = self._link, None
        if link is not None:
            link.close()

    def _discover(self, fresh: bool = False) -> Device | None:
        now = self._clock()
        if fresh or self._board_at is None or now - self._board_at >= BOARD_CACHE_S:
            self._board_seen = self._find_board()
            self._board_at = now
        return self._board_seen

    def _board(self, fresh: bool = False) -> Device:
        dev = self._discover(fresh)
        if dev is None:
            raise HilError(NO_DEVICE, "no CrossPad attached",
                           hint="`crosspad-hil devices` lists what the host sees")
        return dev

    def _cdc_link(self, what: str) -> Any:
        """Caller holds self._q."""
        dev = self._board()
        if self._link is not None and not self._link.reader_alive():
            self._close_link()
            dev = self._board(fresh=True)
        if dev.ports.cdc is None:
            self._close_link()
            if dev.usb_mode == UsbMode.AUDIO:
                raise HilError(NO_CDC_IN_AUDIO_MODE,
                               f"{what} needs the CDC profile; {dev.id} is in MIDI+UAC2",
                               hint="crosspad-hil usb-mode set default", device=dev.id)
            raise HilError(NO_DEVICE, f"{dev.id} has no CDC port "
                                      f"(usb_mode={dev.usb_mode.value})", device=dev.id)
        path = dev.ports.cdc.path
        if self._link is not None and self._link.port != path:
            self._close_link()
        if self._link is None:
            self._link = self._link_factory(path)
        return self._link

    def _use_companion(self) -> bool:
        dev = self._board()
        return dev.ports.cdc is None and dev.usb_mode == UsbMode.AUDIO \
            and self._companion is not None

    def _transact(self, session: Any, cmd: str, expect: str | None, timeout_s: float | None,
                  follow: bool | None) -> dict:
        with self._port(session):
            if self._use_companion():
                return self._companion_transact(cmd, expect)
            r = self._cdc_link("transact").transact(cmd, expect=expect, timeout_s=timeout_s,
                                                    follow=follow)
        return {"line": r.line, "parsed": r.parsed, "rtt_ms": r.rtt_ms,
                "more_lines": r.more_lines, "extra_lines": r.extra_lines}

    def _transact_multi(self, session: Any, args: dict) -> dict:
        cmd = str(args["cmd"])
        expect = args["expect"]
        prefixes = (expect,) if isinstance(expect, str) else tuple(expect)
        end = args.get("end")
        with self._port(session):
            if self._use_companion():
                lines = self._companion_lines(cmd)
                kept = [ln for ln in lines if ln.startswith(prefixes)
                        or (end is not None and ln.startswith(end))]
                if not kept:
                    raise HilError(TIMEOUT, f"no {'/'.join(prefixes)} in the companion reply "
                                            f"to {cmd!r}", cmd=cmd, reply=lines[:3])
                if is_err_line(kept[0]):
                    raise HilError(NOT_SUPPORTED, f"device replied {kept[0]}", cmd=cmd,
                                   reply=kept[0])
                return {"lines": kept}
            lines = self._cdc_link("transact_multi").transact_multi(
                cmd, prefixes, end=end, idle_ms=int(args.get("idle_ms", MULTI_IDLE_MS)),
                timeout_s=float(args.get("timeout_s", MULTI_TIMEOUT_S)))
        return {"lines": lines}

    def _companion_lines(self, cmd: str) -> list[str]:
        assert self._companion is not None
        text = self._companion(cmd.strip(), COMPANION_TIMEOUT_S)
        lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
        if lines == [cmd.strip()]:
            raise HilError(UNKNOWN_VERB, f"device echoed {cmd.strip()!r}: no handler matched it",
                           hint="not every CDC verb is reachable over SysEx 0x1F", cmd=cmd)
        return lines

    def _companion_transact(self, cmd: str, expect: str | None) -> dict:
        prefixes = resolve_expect(self._knowledge_verbs, cmd, expect)
        lines = self._companion_lines(cmd)
        if prefixes is None:
            return {"line": "", "parsed": None, "rtt_ms": 0.0, "more_lines": [],
                    "extra_lines": []}
        for i, ln in enumerate(lines):
            if ln.startswith(prefixes):
                return {"line": ln, "parsed": parse_cdc_reply(ln), "rtt_ms": 0.0,
                        "more_lines": [], "extra_lines": lines[:i] + lines[i + 1:]}
        raise HilError(TIMEOUT, f"no {'/'.join(prefixes)} in the companion reply to {cmd!r}",
                       cmd=cmd, reply=lines[:3])
