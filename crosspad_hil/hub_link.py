"""Reaching the board through the hub: `open_link()` for verbs, `exclusive()` for the rest.

A HubLink has CdcLink's surface, so verbs, scenarios and the daemon do not
care which one they hold. `exclusive()` asks the hub for an exclusive op: the
hub closes its own CDC and keeps every other client out until the block ends.
Inside it this process opens the port directly, exactly as before the hub
existed, and `open_link()` hands out direct links too.

This is the port layer only. Who is testing on the board is the bench lease
(bench.py), which the CLI and the daemon check before they get here; the hub
never looks at it.
"""
from __future__ import annotations

import os
import subprocess
import sys
import threading
import time
from collections.abc import Callable, Iterator, Sequence
from contextlib import contextmanager
from pathlib import Path
from typing import Any

from crosspad_hil import hub_ipc
from crosspad_hil.cdc import BurstResult, CdcLink, Reply, verb_timeout_s
from crosspad_hil.devices import Device, UsbMode
from crosspad_hil.errors import ENV, NO_CDC_IN_AUDIO_MODE, NO_DEVICE, HilError
from crosspad_hil.knowledge import load
from crosspad_hil.serial_open import open_serial

AUTOSTART_WAIT_S = 3.0
AUTOSTART_POLL_S = 0.1
# On top of the verb's own timeout: the hub may first wait for another client's verb.
RPC_MARGIN_S = 5.0
READY_POLL_S = 0.5
READY_PROBE_TIMEOUT_S = 1.5
READY_WAIT_S = 25.0
NO_HUB_ENV = "CROSSPAD_HIL_NO_HUB"
CLIENT_ENV = "CROSSPAD_HIL_CLIENT"
_TRUTHY = ("1", "true", "yes")
CLI_AUDIO_HINT = "crosspad-hil usb-mode set default"

# One exclusive op per process however many holds share it: the hub
# connection that owns it lives here, taken by the first hold and given back
# by the last, in whatever order they end.
_held_guard = threading.Lock()
_held = 0
_held_conn: hub_ipc.Connection | None = None


def hub_enabled() -> bool:
    return os.environ.get(NO_HUB_ENV, "").strip().lower() not in _TRUTHY


def set_no_hub(flag: bool) -> None:
    """`--no-hub`: this process and its children talk to the board directly."""
    if flag:
        os.environ[NO_HUB_ENV] = "1"


def client_name() -> str:
    return os.environ.get(CLIENT_ENV, "").strip() or "crosspad-hil"


def holds_exclusive() -> bool:
    """True while this process is inside an exclusive op."""
    with _held_guard:
        return _held > 0


def direct() -> bool:
    """Open the port ourselves: no hub, or the hub already stepped aside for us."""
    return not hub_enabled() or holds_exclusive()


def _spawn_hub() -> None:
    kwargs: dict[str, Any] = {"stdin": subprocess.DEVNULL, "stdout": subprocess.DEVNULL,
                              "stderr": subprocess.DEVNULL, "close_fds": True}
    if sys.platform == "win32":
        kwargs["creationflags"] = (subprocess.DETACHED_PROCESS
                                   | subprocess.CREATE_NEW_PROCESS_GROUP)
    else:
        kwargs["start_new_session"] = True
    # `-m` puts the working directory first on sys.path: run from our own private
    # dir, not wherever the caller stands, or a crosspad_hil/ there would run instead.
    kwargs["cwd"] = str(hub_ipc.private_dir(hub_ipc.endpoint_file().parent))
    subprocess.Popen([sys.executable, "-m", "crosspad_hil.cli", "midi", "hub", "run"], **kwargs)


def connect_or_start(purpose: str, *, lock_dir: Path | None = None,
                     spawn: Callable[[], None] | None = None,
                     wait_s: float = AUTOSTART_WAIT_S) -> hub_ipc.Connection:
    try:
        return hub_ipc.Connection.open(lock_dir, client=client_name(), purpose=purpose)
    except HilError as e:
        if e.details.get("reason") != "no_hub":
            raise
    (spawn or _spawn_hub)()
    deadline = time.monotonic() + wait_s
    last: HilError | None = None
    while time.monotonic() < deadline:
        try:
            return hub_ipc.Connection.open(lock_dir, client=client_name(), purpose=purpose)
        except HilError as e:
            last = e
            time.sleep(AUTOSTART_POLL_S)
    raise HilError(ENV, f"the CrossPad hub did not start within {wait_s:g}s",
                   hint="run `crosspad-hil midi hub run` in a terminal to see why, or pass "
                        "--no-hub", reason="no_hub",
                   cause=None if last is None else last.message)


class HubLink:
    """CdcLink's surface over the hub's control socket."""

    def __init__(self, conn: hub_ipc.Connection, purpose: str = "") -> None:
        self.port = "hub"
        self._conn: hub_ipc.Connection | None = conn
        self._purpose = purpose
        self._verbs: dict[str, dict] = dict(load("cdc").get("verbs", {}))

    def _call(self, op: str, args: dict, wait_s: float) -> Any:
        if self._conn is None:
            raise HilError(ENV, "HubLink is closed", reason="closed")
        if self._conn.closed:
            # A timeout or a restarted hub ended the last connection; one new one.
            self._conn = connect_or_start(self._purpose)
        return self._conn.call(op, args, timeout_s=wait_s + RPC_MARGIN_S)

    def open(self, *args: Any, **kwargs: Any) -> None:
        return None

    def reader_alive(self) -> bool:
        return self._conn is not None

    def wait_ready(self, timeout_s: float = READY_WAIT_S) -> None:
        deadline = time.monotonic() + timeout_s
        last: HilError | None = None
        while time.monotonic() < deadline:
            try:
                if (self.transact("APP_LIST", timeout_s=READY_PROBE_TIMEOUT_S).parsed
                        or {}).get("apps"):
                    return
            except HilError as e:
                last = e
            time.sleep(READY_POLL_S)
        raise HilError(ENV, f"the board did not become ready through the hub within "
                            f"{timeout_s:g}s", reason="not_ready",
                       cause=None if last is None else last.message)

    def transact(self, cmd: str, expect: str | None = None, timeout_s: float | None = None,
                 follow: bool | None = None) -> Reply:
        wait_s = verb_timeout_s(self._verbs, cmd.strip().split(" ", 1)[0], timeout_s)
        r = self._call("transact", {"cmd": cmd, "expect": expect, "timeout_s": timeout_s,
                                    "follow": follow}, wait_s)
        return Reply(line=r["line"], parsed=r["parsed"], rtt_ms=r["rtt_ms"],
                     extra_lines=list(r["extra_lines"]), more_lines=list(r["more_lines"]))

    def transact_multi(self, cmd: str, expect: str | tuple[str, ...], end: str | None = None,
                       idle_ms: int = 200, timeout_s: float = 3.0) -> list[str]:
        exp = expect if isinstance(expect, str) else list(expect)
        return list(self._call("transact_multi", {"cmd": cmd, "expect": exp, "end": end,
                                                  "idle_ms": idle_ms, "timeout_s": timeout_s},
                               timeout_s)["lines"])

    def send(self, cmd: str) -> None:
        self._call("send", {"cmd": cmd}, 0.0)

    def burst(self, cmds: Sequence[str], rate_hz: float) -> BurstResult:
        seconds = len(cmds) / rate_hz if rate_hz > 0 else 0.0
        return BurstResult(**self._call("burst", {"cmds": list(cmds), "rate_hz": rate_hz},
                                        seconds))

    def unmatched(self) -> list[str]:
        return []

    def close(self) -> None:
        conn, self._conn = self._conn, None
        if conn is not None:
            conn.close()


def require_cdc(device: Device, audio_hint: str = CLI_AUDIO_HINT) -> str:
    """The CDC path, or the error a direct open would hit anyway."""
    if device.ports.cdc is None:
        if device.usb_mode == UsbMode.AUDIO:
            raise HilError(NO_CDC_IN_AUDIO_MODE,
                           f"{device.id} is in the MIDI+UAC2 profile; CDC endpoint absent",
                           hint=audio_hint, device=device.id)
        raise HilError(NO_DEVICE, f"{device.id} has no CDC port "
                                  f"(usb_mode={device.usb_mode.value})",
                       hint="is the board in bootloader/DFU?", device=device.id)
    return device.ports.cdc.path


def open_link(device: Device, *, purpose: str, serial_factory: Callable[..., Any] = open_serial,
              port_resolver: Callable[[], str | None] | None = None,
              audio_hint: str = CLI_AUDIO_HINT) -> CdcLink | HubLink:
    """A link for verbs: through the hub, or direct when `direct()` says so.

    Through the hub the audio profile is the hub's business (it answers over
    SysEx 0x1F), so nothing is checked here.
    """
    if not direct():
        return HubLink(connect_or_start(purpose), purpose)
    link = CdcLink(require_cdc(device, audio_hint), serial_factory=serial_factory,
                   purpose=purpose)
    link.open(port_resolver=port_resolver)
    return link


class ExclusiveHold:
    """One hold on this process's exclusive op; `release()` is idempotent. For holds that
    outlive a `with` block (a stimulus or a capture behind a daemon handle)."""

    def __init__(self) -> None:
        self._released = False

    def release(self) -> None:
        global _held, _held_conn
        with _held_guard:
            if self._released:
                return
            self._released = True
            _held -= 1
            if _held:
                return
            conn, _held_conn = _held_conn, None
            if conn is None:
                return
            try:
                conn.call("release")
            except HilError:
                pass
            finally:
                conn.close()


def hold_exclusive(purpose: str, *, lock_dir: Path | None = None) -> ExclusiveHold:
    """Take an exclusive op; PORT_BUSY (details.holder) when another client holds one.

    Re-entrant per process: a process already holding one (a scenario that
    pushes a kit, a capture with a transfer inside) only counts up, and the
    hub lets go when the last hold ends. Without the hub it only marks this
    process.
    """
    global _held, _held_conn
    with _held_guard:
        if _held == 0 and hub_enabled():
            conn = connect_or_start(purpose, lock_dir=lock_dir)
            try:
                conn.call("lease", {"purpose": purpose})
            except HilError:
                conn.close()
                raise
            _held_conn = conn
        _held += 1
    return ExclusiveHold()


@contextmanager
def exclusive(purpose: str, *, lock_dir: Path | None = None) -> Iterator[None]:
    hold = hold_exclusive(purpose, lock_dir=lock_dir)
    try:
        yield
    finally:
        hold.release()
