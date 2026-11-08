"""NDJSON daemon on stdio (`crosspad-hil serve`), the transport crosspad-mcp talks to.

Request  {"id": int, "op": str, "args": dict}
Response {"id", "ok": true, "result"} | {"id", "ok": false, "error": HilError.to_dict()}
Event    {"ev": str, ...}          (console.*, task.*)
Logs go to stderr only. Pattern mirrors crosspad-mcp/tracer/swd_tracer.py: a
reader thread consumes stdin, every op runs on a worker, stdout writes are
serialized by one lock so lines never interleave.
"""
from __future__ import annotations

import dataclasses
import itertools
import json
import os
import sys
import threading
import time
import traceback
from collections.abc import Callable, Iterator
from concurrent.futures import Future, ThreadPoolExecutor
from contextlib import contextmanager
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, TextIO

from . import __version__, hub_link, midi_hub, verbs
from . import ble as ble_mod
from . import capture as cap_mod
from . import diagnose as diag
from . import fs as fs_mod
from . import stim as stim_mod
from .analyze import analyze as analyze_wav
from .bench import Bench
from .capture import CaptureSession
from .cdc import CdcLink
from .console import Console
from .devices import Backends, Device, UsbMode, discover, select
from .errors import (
    BAD_ARGS,
    CANCELLED,
    ENV,
    HANDLE_EXPIRED,
    NO_DEVICE,
    PORT_BUSY,
    HilError,
)
from .knowledge import KNOWLEDGE_NAMES
from .knowledge import load as load_knowledge
from .midi import MidiIO, MidiRole, echo_rtt, query_route
from .ota import ota_flash
from .parsers import ConsoleEvent
from .resources import fd_counts
from .scenarios import base as scen
from .scenarios import get as get_scenario

# Import every scenario module so the catalog is complete before the parser
# (or the daemon's scenario.list) is built. Naming them one by one here is
# how midi_bench and midi_stress shipped written, registered by nobody and
# reachable from nothing.
from .scenarios import load_builtin as _load_builtin_scenarios
from .scenarios import names as scenario_names
from .serial_open import open_serial
from .snapshot import Snapshot, take_snapshot
from .usbmode import set_mode

_load_builtin_scenarios()

_MISSING = object()
SERVE_AUDIO_HINT = "usbmode.set mode=default (SysEx 0x1B on the ESP MIDI port) then retry"


def _json_default(o: Any) -> Any:
    if dataclasses.is_dataclass(o) and not isinstance(o, type):
        return dataclasses.asdict(o)
    if isinstance(o, Enum):
        return o.value
    if isinstance(o, Path):
        return str(o)
    if isinstance(o, (set, frozenset)):
        return sorted(o)
    if isinstance(o, bytes):
        return o.hex()
    return str(o)


def _arg(args: dict, name: str, typ: type | tuple[type, ...], default: Any = _MISSING) -> Any:
    if name not in args or args[name] is None:
        if default is _MISSING:
            raise HilError(BAD_ARGS, f"missing argument {name!r}")
        return default
    v = args[name]
    if typ is float and isinstance(v, int) and not isinstance(v, bool):
        v = float(v)
    if typ is int and isinstance(v, bool):
        raise HilError(BAD_ARGS, f"argument {name!r} must be int, got bool")
    if not isinstance(v, typ):
        want = typ.__name__ if isinstance(typ, type) else "/".join(t.__name__ for t in typ)
        raise HilError(BAD_ARGS, f"argument {name!r} must be {want}, got {type(v).__name__}")
    return v


# --------------------------------------------------------------------------- handles

@dataclass
class _Entry:
    obj: Any
    prefix: str
    created: float
    last_used: float
    ttl_s: float | None


class HandleRegistry:
    """Thread-safe handle table with idle TTL. Handles look like "con_1", "task_3"."""

    def __init__(self, clock: Callable[[], float] = time.monotonic,
                 on_expire: Callable[[str, Any], None] | None = None) -> None:
        self._clock = clock
        self._on_expire = on_expire
        self._lock = threading.Lock()
        self._entries: dict[str, _Entry] = {}
        self._counters: dict[str, itertools.count] = {}

    def mint(self, prefix: str, obj: Any, ttl_s: float | None) -> str:
        with self._lock:
            counter = self._counters.setdefault(prefix, itertools.count(1))
            handle = f"{prefix}_{next(counter)}"
            now = self._clock()
            self._entries[handle] = _Entry(obj, prefix, now, now, ttl_s)
            return handle

    @staticmethod
    def _expired(e: _Entry, now: float) -> bool:
        return e.ttl_s is not None and (now - e.last_used) > e.ttl_s

    def get(self, handle: str) -> Any:
        expired: _Entry | None = None
        with self._lock:
            e = self._entries.get(handle)
            now = self._clock()
            if e is None:
                raise HilError(HANDLE_EXPIRED, f"unknown handle {handle!r}", hint="open again",
                               handle=handle)
            if self._expired(e, now):
                expired = self._entries.pop(handle)
            else:
                e.last_used = now
                return e.obj
        if self._on_expire is not None and expired is not None:
            self._on_expire(handle, expired.obj)
        assert expired is not None
        raise HilError(HANDLE_EXPIRED,
                       f"handle {handle} expired after {expired.ttl_s:.0f}s idle",
                       hint="open again", handle=handle)

    def touch(self, handle: str) -> None:
        with self._lock:
            e = self._entries.get(handle)
            if e is not None:
                e.last_used = self._clock()

    def drop(self, handle: str) -> Any | None:
        with self._lock:
            e = self._entries.pop(handle, None)
        return None if e is None else e.obj

    def sweep(self) -> list[str]:
        dead: list[tuple[str, _Entry]] = []
        with self._lock:
            now = self._clock()
            for h, e in list(self._entries.items()):
                if self._expired(e, now):
                    dead.append((h, self._entries.pop(h)))
        for h, e in dead:
            if self._on_expire is not None:
                self._on_expire(h, e.obj)
        return [h for h, _ in dead]

    def list(self) -> list[dict]:
        with self._lock:
            now = self._clock()
            return [{"handle": h, "kind": e.prefix, "age_s": round(now - e.created, 1),
                     "idle_s": round(now - e.last_used, 1), "ttl_s": e.ttl_s}
                    for h, e in self._entries.items()]

    def all_handles(self) -> list[str]:
        with self._lock:
            return list(self._entries)


# --------------------------------------------------------------------------- sessions

@dataclass
class _ConsoleSession:
    console: Console
    device_id: str
    port: str
    log_path: str | None
    lock: threading.Lock = field(default_factory=threading.Lock)

    def close(self) -> None:
        self.console.close()


@dataclass
class _CdcSession:
    link: CdcLink
    device_id: str
    lock: threading.Lock = field(default_factory=threading.Lock)

    def close(self) -> None:
        self.link.close()


@dataclass
class _CaptureSession:
    """A capture in flight. ``close()`` stops it and puts the USB profile back —
    handle expiry and daemon shutdown both go through here, so a forgotten `cap_*`
    never leaves the board stuck in the UAC2 profile with no CDC endpoint."""

    capture: CaptureSession
    device_id: str
    last_result: dict | None = None
    # The exclusive op the capture holds: it switches the USB profile under every client.
    hold: hub_link.ExclusiveHold | None = None

    def close(self) -> None:
        try:
            if self.capture.running:
                try:
                    self.last_result = self.capture.stop().to_dict()
                    return
                except HilError:
                    pass
            self.capture.close()
        finally:
            self.release_hold()

    def release_hold(self) -> None:
        if self.hold is not None:
            self.hold.release()


@dataclass
class _StimSession:
    """A running pad stimulus. ``close()`` is what handle expiry and shutdown call."""

    stim: stim_mod.Stimulus
    device_id: str
    last_status: dict | None = None
    # The exclusive op a stimulus with its own CDC link holds until it stops.
    hold: hub_link.ExclusiveHold | None = None

    def close(self) -> None:
        try:
            if self.stim.running:
                self.last_status = self.stim.stop()
        finally:
            self.release_hold()

    def release_hold(self) -> None:
        if self.hold is not None:
            self.hold.release()


@dataclass
class _BleSession:
    """A BLE MIDI peripheral this PC's radio is connected to."""

    link: ble_mod.BleMidiLink
    device_id: str | None = None

    def close(self) -> None:
        self.link.disconnect()


@dataclass
class _Task:
    handle: str
    kind: str
    status: str = "working"
    progress: int = 0
    total: int | None = None
    message: str = ""
    result: Any = None
    error: dict | None = None
    cancelled: threading.Event = field(default_factory=threading.Event)
    done: threading.Event = field(default_factory=threading.Event)
    future: Future | None = None

    def to_dict(self) -> dict:
        d: dict[str, Any] = {"task": self.handle, "kind": self.kind, "status": self.status,
                             "progress": self.progress, "total": self.total,
                             "message": self.message}
        if self.result is not None:
            d["result"] = self.result
        if self.error is not None:
            d["error"] = self.error
        return d

    def close(self) -> None:
        self.cancelled.set()


class DaemonProgress(scen.Progress):
    def __init__(self, daemon: Daemon, task: _Task) -> None:
        self._daemon = daemon
        self._task = task

    def __call__(self, progress: int, total: int | None, message: str) -> None:
        self._task.progress = int(progress)
        self._task.total = total
        self._task.message = message
        self._daemon.emit({"ev": "task.progress", "task": self._task.handle,
                           "progress": int(progress), "total": total, "message": message})


# --------------------------------------------------------------------------- daemon

class Daemon:
    SWEEP_S = 5.0
    DEV_LOCK_WAIT_S = 60.0
    # A stimulus handle lives until stop(), 15 minutes at the outside (design §5 handles):
    # a forgotten one holds the CDC port and keeps hitting pads nobody is watching.
    STIM_TTL_S = 900.0
    CAPTURE_TTL_S = 900.0
    # How often the leases of running tasks are renewed.
    BENCH_RENEW_S = 60.0

    def __init__(self, stdin: TextIO, stdout: TextIO, *,
                 backends: Backends | None = None,
                 serial_factory: Callable[..., Any] = open_serial,
                 clock: Callable[[], float] = time.monotonic,
                 midi_factory: Callable[..., Any] = MidiIO,
                 run_scenario_fn: Callable[..., scen.Report] = scen.run_scenario,
                 ota_flash_fn: Callable[..., dict] = ota_flash,
                 set_mode_fn: Callable[..., Device] = set_mode,
                 stim_start_fn: Callable[..., stim_mod.Stimulus] = stim_mod.start_stimulus,
                 ble_scan_fn: Callable[..., list] = ble_mod.scan,
                 ble_connect_fn: Callable[..., ble_mod.BleMidiLink] = ble_mod.connect,
                 diagnose_fn: Callable[..., dict] = diag.diagnose,
                 capture_factory: Callable[..., CaptureSession] = CaptureSession,
                 analyze_fn: Callable[..., dict] = analyze_wav,
                 handle_ttl_s: float = 600.0,
                 bench: Bench | None = None,
                 log: Callable[[str], None] | None = None) -> None:
        self._stdin = stdin
        self._stdout = stdout
        self._backends = backends
        self._serial_factory = serial_factory
        self._clock = clock
        self._midi_factory = midi_factory
        self._run_scenario_fn = run_scenario_fn
        self._ota_flash_fn = ota_flash_fn
        self._set_mode_fn = set_mode_fn
        self._stim_start_fn = stim_start_fn
        self._ble_scan_fn = ble_scan_fn
        self._ble_connect_fn = ble_connect_fn
        self._diagnose_fn = diagnose_fn
        self._capture_factory = capture_factory
        self._analyze_fn = analyze_fn
        self._ttl = handle_ttl_s
        self._log = log or (lambda m: (sys.stderr.write(m.rstrip("\n") + "\n"), sys.stderr.flush()))
        self._bench = bench if bench is not None else Bench(log=self._log)
        # device id -> holder, for every task that holds a board: the sweep renews
        # those leases, so an hours-long scenario does not lose the board halfway.
        self._bench_keep: dict[str, tuple[str, int]] = {}
        self._bench_keep_guard = threading.Lock()
        self._bench_renewed_at = 0.0
        self._tls = threading.local()
        self._out_lock = threading.Lock()
        self._pool = ThreadPoolExecutor(max_workers=8, thread_name_prefix="hil-op")
        self._handles = HandleRegistry(clock=clock, on_expire=self._on_expire)
        self._devlocks: dict[str, threading.Lock] = {}
        self._devlocks_guard = threading.Lock()
        self._snap_counter = itertools.count(1)
        self._stop = threading.Event()
        self._t0 = time.monotonic()
        self._ops_total = 0
        self._ops_guard = threading.Lock()
        self._ops: dict[str, Callable[[dict], Any]] = {
            "serve.ping": self._op_ping,
            "serve.stats": self._op_stats,
            "serve.shutdown": self._op_shutdown,
            "devices.list": self._op_devices_list,
            "devices.doctor": self._op_devices_doctor,
            "knowledge.get": self._op_knowledge_get,
            "bench.claim": self._op_bench_claim,
            "bench.renew": self._op_bench_renew,
            "bench.release": self._op_bench_release,
            "bench.status": self._op_bench_status,
            "bench.check": self._op_bench_check,
            "console.open": self._op_console_open,
            "console.read": self._op_console_read,
            "console.expect": self._op_console_expect,
            "console.reset": self._op_console_reset,
            "console.wait_boot": self._op_console_wait_boot,
            "console.snapshot": self._op_console_snapshot,
            "console.close": self._op_console_close,
            "cdc.open": self._op_cdc_open,
            "cdc.transact": self._op_cdc_transact,
            "cdc.verb": self._op_cdc_verb,
            "cdc.burst": self._op_cdc_burst,
            "cdc.close": self._op_cdc_close,
            "midi.sysex": self._op_midi_sysex,
            "midi.note": self._op_midi_note,
            "midi.echo_rtt": self._op_midi_echo_rtt,
            "midi.query_route": self._op_midi_query_route,
            "usbmode.set": self._op_usbmode_set,
            "ota.flash": self._op_ota_flash,
            "snapshot.take": self._op_snapshot_take,
            "stim.start": self._op_stim_start,
            "stim.status": self._op_stim_status,
            "stim.stop": self._op_stim_stop,
            "ble.scan": self._op_ble_scan,
            "ble.connect": self._op_ble_connect,
            "ble.status": self._op_ble_status,
            "ble.send": self._op_ble_send,
            "ble.listen": self._op_ble_listen,
            "ble.latency": self._op_ble_latency,
            "ble.disconnect": self._op_ble_disconnect,
            "capture.start": self._op_capture_start,
            "capture.health": self._op_capture_health,
            "capture.stop": self._op_capture_stop,
            "analyze.wav": self._op_analyze_wav,
            "diagnose.crash": self._op_diagnose_crash,
            "scenario.list": self._op_scenario_list,
            "scenario.run": self._op_scenario_run,
            "task.status": self._op_task_status,
            "task.wait": self._op_task_wait,
            "task.cancel": self._op_task_cancel,
            "task.list": self._op_task_list,
            "fs.pull": self._op_fs_pull,
            "fs.push": self._op_fs_push,
            "hub.status": self._op_hub_status,
        }

    # ----------------------------------------------------------------- transport

    def run(self) -> None:
        reader = threading.Thread(target=self._reader, name="ndjson-reader", daemon=True)
        reader.start()
        try:
            while not self._stop.wait(self.SWEEP_S):
                for h in self._handles.sweep():
                    self._log(f"handle {h} expired")
                self._bench_keepalive()
        finally:
            self._shutdown_all()

    def _reader(self) -> None:
        try:
            for raw in self._stdin:
                line = raw.strip()
                if not line:
                    continue
                try:
                    req = json.loads(line)
                except ValueError as e:
                    self.emit({"id": None, "ok": False,
                               "error": HilError(BAD_ARGS, f"malformed JSON: {e}").to_dict()})
                    continue
                if not isinstance(req, dict):
                    self.emit({"id": None, "ok": False,
                               "error": HilError(BAD_ARGS, "request must be an object").to_dict()})
                    continue
                if self._stop.is_set():
                    break
                self._pool.submit(self._serve_one, req)
        finally:
            self._stop.set()  # EOF: the parent went away, or serve.shutdown was queued

    def _serve_one(self, req: dict) -> None:
        resp = self.handle(req)
        self.emit(resp)
        if req.get("op") == "serve.shutdown" and resp.get("ok"):
            self._stop.set()

    def handle(self, req: dict) -> dict:
        rid = req.get("id")
        op = req.get("op")
        args = req.get("args", {})
        try:
            if not isinstance(op, str):
                raise HilError(BAD_ARGS, "missing 'op'")
            if args is None:
                args = {}
            if not isinstance(args, dict):
                raise HilError(BAD_ARGS, "'args' must be an object")
            with self._ops_guard:
                self._ops_total += 1
            fn = self._ops.get(op)
            if fn is None:
                raise HilError(BAD_ARGS, f"unknown op {op!r}",
                               hint="ops: " + ", ".join(sorted(self._ops)))
            self._tls.op = op
            return {"id": rid, "ok": True, "result": fn(args)}
        except HilError as e:
            return {"id": rid, "ok": False, "error": e.to_dict()}
        except Exception as e:  # noqa: BLE001 — every failure must become a response
            self._log(traceback.format_exc())
            return {"id": rid, "ok": False,
                    "error": HilError(ENV, f"{type(e).__name__}: {e}", op=op).to_dict()}

    def emit(self, ev: dict) -> None:
        line = json.dumps(ev, default=_json_default)
        with self._out_lock:
            self._stdout.write(line + "\n")
            self._stdout.flush()

    def _shutdown_all(self) -> None:
        # 1. tell every task to stop, 2. let in-flight ops write their responses,
        # 3. close whatever is still open (ports, tasks) so the locks are released.
        for h in self._handles.all_handles():
            try:
                obj = self._handles.get(h)
            except HilError:
                continue
            if isinstance(obj, _Task):
                obj.cancelled.set()
        self._pool.shutdown(wait=True, cancel_futures=True)
        for h in self._handles.all_handles():
            obj = self._handles.drop(h)
            if obj is not None:
                self._safe_close(h, obj)

    def _on_expire(self, handle: str, obj: Any) -> None:
        self._safe_close(handle, obj)

    def _safe_close(self, handle: str, obj: Any) -> None:
        try:
            obj.close()
        except Exception as e:  # noqa: BLE001
            self._log(f"closing {handle}: {e}")

    # ----------------------------------------------------------------- devices / locks

    def _resolve(self, args: dict) -> Device:
        """The device an op is about to touch. Every op that touches a board comes
        through here, so this is also where the bench lease is checked."""
        device = args.get("device")
        if device is not None and not isinstance(device, str):
            raise HilError(BAD_ARGS, "'device' must be a string (id or port path)")
        dev = select(discover(self._backends), device)
        self._bench.check(dev.id, _arg(args, "holder", str, None),
                          getattr(self._tls, "op", "?"),
                          force=_arg(args, "bench_force", bool, False))
        return dev

    # ----------------------------------------------------------------- bench lease

    def _bench_device(self, args: dict, *, required: bool = True) -> str | None:
        """A lease is keyed by device id. An id is taken as given (a board in DFU,
        or unplugged, can still be claimed); a port path, or nothing, goes through
        discovery; with no board found, the one board the bench knows will do."""
        device = _arg(args, "device", str, None)
        if device and device.startswith("dev_"):
            return device
        try:
            return select(discover(self._backends), device).id
        except HilError:
            if device:
                raise
            known = self._bench.known_devices()
            if len(known) == 1:
                return known[0]
            if required:
                raise
            return None

    @contextmanager
    def _bench_hold(self, device_id: str, args: dict) -> Iterator[None]:
        """Keep the caller's lease alive while a task works on the board."""
        holder = _arg(args, "holder", str, None)
        if not holder:
            yield
            return
        with self._bench_keep_guard:
            _h, n = self._bench_keep.get(device_id, (holder, 0))
            self._bench_keep[device_id] = (holder, n + 1)
        try:
            yield
        finally:
            with self._bench_keep_guard:
                h, n = self._bench_keep.get(device_id, (holder, 1))
                if n <= 1:
                    self._bench_keep.pop(device_id, None)
                else:
                    self._bench_keep[device_id] = (h, n - 1)

    def _bench_keepalive(self) -> None:
        now = time.monotonic()
        if now - self._bench_renewed_at < self.BENCH_RENEW_S:
            return
        self._bench_renewed_at = now
        with self._bench_keep_guard:
            held = [(d, h) for d, (h, _n) in self._bench_keep.items()]
        for device_id, holder in held:
            self._bench.touch(device_id, holder)

    def _op_bench_claim(self, args: dict) -> dict:
        return self._bench.claim(self._bench_device(args), _arg(args, "holder", str),
                                 _arg(args, "purpose", str, ""),
                                 _arg(args, "ttl_s", (int, float), None),
                                 force=_arg(args, "bench_force", bool, False))

    def _op_bench_renew(self, args: dict) -> dict:
        return self._bench.renew(self._bench_device(args), _arg(args, "holder", str),
                                 _arg(args, "ttl_s", (int, float), None))

    def _op_bench_release(self, args: dict) -> dict:
        return self._bench.release(self._bench_device(args), _arg(args, "holder", str),
                                   _arg(args, "firmware_left", str, None),
                                   force=_arg(args, "bench_force", bool, False))

    def _op_bench_status(self, args: dict) -> dict:
        device = _arg(args, "device", str, None)
        if device is not None and not device.startswith("dev_"):
            device = self._bench_device(args)
        return self._bench.status(device, history=_arg(args, "history", int, 10))

    def _op_bench_check(self, args: dict) -> dict:
        """For callers that touch the board outside the daemon (idf.py, an ST-Link):
        may ``holder`` do ``op`` now? Without a resolvable device, every lease is checked."""
        holder = _arg(args, "holder", str, None)
        op = _arg(args, "op", str, "external")
        force = _arg(args, "bench_force", bool, False)
        device = self._bench_device(args, required=False)
        if device is None:
            return self._bench.check_all(holder, op, force=force)
        return self._bench.check(device, holder, op, force=force)

    @contextmanager
    def _dev_lock(self, device_id: str) -> Iterator[None]:
        with self._devlocks_guard:
            lock = self._devlocks.setdefault(device_id, threading.Lock())
        if not lock.acquire(timeout=self.DEV_LOCK_WAIT_S):
            raise HilError(PORT_BUSY,
                           f"{device_id} is busy (held for "
                           f"{self.DEV_LOCK_WAIT_S:.0f}s by another op)",
                           hint="another op or task on this device is running; task.list / "
                                "task.cancel", device=device_id)
        try:
            yield
        finally:
            lock.release()

    def _open_cdc(self, dev: Device) -> CdcLink | hub_link.HubLink:
        return hub_link.open_link(dev, purpose="serve", serial_factory=self._serial_factory,
                                  audio_hint=SERVE_AUDIO_HINT)

    def _console_port(self, dev: Device) -> str:
        if dev.ports.console is None:
            raise HilError(NO_DEVICE, f"{dev.id} has no STM32 bridge VCP (0x0483:0x5740)",
                           hint="the console comes off the STM bridge, not the ESP CDC",
                           device=dev.id)
        return dev.ports.console.path

    def _session(self, args: dict, cls: type, key: str = "handle") -> Any:
        handle = _arg(args, key, str)
        sess = self._handles.get(handle)
        if not isinstance(sess, cls):
            raise HilError(BAD_ARGS, f"{handle} is not a {cls.__name__.strip('_').lower()} handle")
        device_id = getattr(sess, "device_id", None)
        if device_id:
            # using an open handle is board activity too: it renews the holder's lease
            self._bench.touch(device_id, _arg(args, "holder", str, None))
        return sess

    @contextmanager
    def _cdc_from(self, args: dict) -> Iterator[CdcLink]:
        if args.get("handle") is not None:
            sess = self._session(args, _CdcSession)
            with sess.lock:
                yield sess.link
            return
        dev = self._resolve(args)
        with self._dev_lock(dev.id):
            link = self._open_cdc(dev)
            try:
                yield link
            finally:
                link.close()

    # ----------------------------------------------------------------- tasks

    def _start_task(self, kind: str, fn: Callable[[_Task], Any]) -> str:
        task = _Task(handle="", kind=kind)
        handle = self._handles.mint("task", task, None)
        task.handle = handle

        def body() -> None:
            try:
                if task.cancelled.is_set():
                    raise HilError(CANCELLED, "cancelled before start")
                task.result = fn(task)
                task.status = "completed"
                self.emit({"ev": "task.done", "task": handle, "status": "completed",
                           "result": task.result})
            except HilError as e:
                task.error = e.to_dict()
                task.status = "cancelled" if e.code == CANCELLED else "failed"
                self.emit({"ev": "task.done", "task": handle, "status": task.status,
                           "error": task.error})
            except Exception as e:  # noqa: BLE001
                self._log(traceback.format_exc())
                task.error = HilError(ENV, f"{type(e).__name__}: {e}").to_dict()
                task.status = "failed"
                self.emit({"ev": "task.done", "task": handle, "status": "failed",
                           "error": task.error})
            finally:
                task.done.set()

        task.future = self._pool.submit(body)
        return handle

    def _task(self, args: dict) -> _Task:
        return self._session(args, _Task, key="task")

    # ----------------------------------------------------------------- ops: serve

    def _op_ping(self, args: dict) -> dict:
        return {"version": __version__, "uptime_s": round(time.monotonic() - self._t0, 3)}

    def _op_stats(self, args: dict) -> dict:
        by_kind: dict[str, int] = {}
        handles = self._handles.list()
        for h in handles:
            by_kind[h["kind"]] = by_kind.get(h["kind"], 0) + 1
        with self._ops_guard:
            ops_total = self._ops_total
        return {
            "version": __version__,
            "pid": os.getpid(),
            "uptime_s": round(time.monotonic() - self._t0, 3),
            "ops_total": ops_total,
            "handles": by_kind,
            "handles_total": len(handles),
            "threads": threading.active_count(),
            **fd_counts(),
        }

    def _op_shutdown(self, args: dict) -> dict:
        return {"ok": True}

    # ----------------------------------------------------------------- ops: devices

    def _op_devices_list(self, args: dict) -> dict:
        return {"devices": [d.to_dict() for d in discover(self._backends)]}

    def _op_devices_doctor(self, args: dict) -> dict:
        from .cli import doctor_checks

        return {"checks": doctor_checks(self._backends)}

    # ----------------------------------------------------------------- ops: knowledge

    def _op_knowledge_get(self, args: dict) -> dict:
        name = _arg(args, "name", str)
        if name not in KNOWLEDGE_NAMES:
            raise HilError(BAD_ARGS, "unknown knowledge file",
                           hint="one of: cdc, sysex, markers")
        return load_knowledge(name)

    # ----------------------------------------------------------------- ops: console

    def _op_console_open(self, args: dict) -> dict:
        dev = self._resolve(args)
        port = self._console_port(dev)
        reset = _arg(args, "reset", bool, False)
        log_to = _arg(args, "log_to", str, None)
        holder: dict[str, str] = {}

        def on_event(ev: ConsoleEvent) -> None:
            h = holder.get("handle")
            if h is not None:
                self._console_event(h, ev)

        con = Console(port, log_path=Path(log_to) if log_to else None,
                      serial_factory=self._serial_factory, on_event=on_event)
        with self._dev_lock(dev.id):
            if reset:
                with hub_link.exclusive("console reset"):
                    con.open(reset=True)
            else:
                con.open(reset=False)
        sess = _ConsoleSession(con, dev.id, port, log_to)
        holder["handle"] = self._handles.mint("con", sess, self._ttl)
        return {"handle": holder["handle"], "port": port, "log_path": log_to}

    def _console_event(self, handle: str, ev: ConsoleEvent) -> None:
        base = {"handle": handle, "seq": ev.seq}
        if ev.kind == "fatal":
            self.emit({"ev": "console.fatal", **base, "pattern": ev.data.get("pattern"),
                       "line": ev.line})
        elif ev.kind == "reboot":
            self.emit({"ev": "console.reboot", **base, "count": ev.data.get("count"),
                       "line": ev.line})
        elif ev.kind == "cdc_drops":
            self.emit({"ev": "console.cdc_drops", **base, "dropped": ev.data.get("dropped")})
        elif ev.kind == "kit":
            self.emit({"ev": "console.kit", **base, "kit": ev.data.get("kit"),
                       "state": ev.data.get("state")})
        elif ev.kind == "boot_complete":
            self.emit({"ev": "console.boot_complete", **base,
                       "missing": list(ev.data.get("missing") or [])})

    def _op_console_read(self, args: dict) -> dict:
        sess = self._session(args, _ConsoleSession)
        with sess.lock:
            r = sess.console.read(since_seq=_arg(args, "since_seq", int, None),
                                  wait_ms=_arg(args, "wait_ms", int, 0),
                                  match=_arg(args, "match", str, None),
                                  limit=min(_arg(args, "limit", int, 2000), 2000))
        return {"lines": [[seq, line] for seq, line in r.lines], "next_seq": r.next_seq,
                "lines_lost": r.lines_lost}

    def _op_console_expect(self, args: dict) -> dict:
        sess = self._session(args, _ConsoleSession)
        patterns = _arg(args, "patterns", list)
        reject = _arg(args, "reject", list, [])
        timeout_s = _arg(args, "timeout_s", (int, float), 30.0)
        if not patterns or not all(isinstance(p, str) for p in patterns):
            raise HilError(BAD_ARGS, "'patterns' must be a non-empty list of regex strings")
        with sess.lock:
            r = sess.console.expect([str(p) for p in patterns], [str(p) for p in reject],
                                    timeout_s=float(timeout_s))
        return dataclasses.asdict(r)

    def _op_console_reset(self, args: dict) -> dict:
        sess = self._session(args, _ConsoleSession)
        with sess.lock, self._dev_lock(sess.device_id), hub_link.exclusive("console reset"):
            sess.console.reset()
        return {"ok": True}

    def _op_console_wait_boot(self, args: dict) -> dict:
        sess = self._session(args, _ConsoleSession)
        timeout_s = _arg(args, "timeout_s", (int, float), None)
        with sess.lock:
            r = sess.console.wait_boot(timeout_s=None if timeout_s is None else float(timeout_s))
        return dataclasses.asdict(r)

    def _op_console_snapshot(self, args: dict) -> dict:
        sess = self._session(args, _ConsoleSession)
        with sess.lock:
            return sess.console.snapshot()

    def _op_console_close(self, args: dict) -> dict:
        handle = _arg(args, "handle", str)
        sess = self._handles.drop(handle)
        if sess is None:
            raise HilError(HANDLE_EXPIRED, f"unknown handle {handle!r}", hint="open again")
        self._safe_close(handle, sess)
        return {"ok": True}

    # ----------------------------------------------------------------- ops: cdc

    def _op_cdc_open(self, args: dict) -> dict:
        dev = self._resolve(args)
        with self._dev_lock(dev.id):
            link = self._open_cdc(dev)
        return {"handle": self._handles.mint("cdc", _CdcSession(link, dev.id), self._ttl),
                "port": dev.ports.cdc.path if dev.ports.cdc else None}

    def _op_cdc_transact(self, args: dict) -> dict:
        cmd = _arg(args, "cmd", str)
        expect = _arg(args, "expect", str, None)
        timeout_s = _arg(args, "timeout_s", (int, float), 2.0)
        # The reply's continuation lines (" bt:", " #n"): by cdc.yaml, or asked for.
        follow = _arg(args, "follow", bool, None)
        with self._cdc_from(args) as link:
            r = link.transact(cmd, expect=expect, timeout_s=float(timeout_s), follow=follow)
        return {"line": r.line, "parsed": r.parsed, "rtt_ms": r.rtt_ms,
                "more_lines": r.more_lines, "extra_lines": r.extra_lines}

    def _op_cdc_verb(self, args: dict) -> Any:
        verb = _arg(args, "verb", str)
        vargs = _arg(args, "args", dict, {})
        fn = getattr(verbs, verb, None)
        if fn is None or verb.startswith("_") or not callable(fn):
            raise HilError(BAD_ARGS, f"unknown verb {verb!r}",
                           hint="verb = a function name in crosspad_hil.verbs, e.g. kit_status")
        with self._cdc_from(args) as link:
            try:
                r = fn(link, **vargs)
            except TypeError as e:
                raise HilError(BAD_ARGS, f"{verb}: {e}") from e
        return r if r is not None else {"ok": True}

    def _op_cdc_burst(self, args: dict) -> dict:
        sess = self._session(args, _CdcSession)
        cmds = _arg(args, "cmds", list)
        rate_hz = float(_arg(args, "rate_hz", (int, float)))
        if not cmds or not all(isinstance(c, str) for c in cmds):
            raise HilError(BAD_ARGS, "'cmds' must be a non-empty list of strings")
        with sess.lock:
            r = sess.link.burst([str(c) for c in cmds], rate_hz)
        return dataclasses.asdict(r)

    def _op_cdc_close(self, args: dict) -> dict:
        handle = _arg(args, "handle", str)
        sess = self._handles.drop(handle)
        if sess is None:
            raise HilError(HANDLE_EXPIRED, f"unknown handle {handle!r}", hint="open again")
        self._safe_close(handle, sess)
        return {"ok": True}

    # ----------------------------------------------------------------- ops: midi

    @contextmanager
    def _midi(self, args: dict) -> Iterator[Any]:
        dev = self._resolve(args)
        role_s = _arg(args, "role", str, "esp")
        try:
            role = MidiRole(role_s)
        except ValueError as e:
            raise HilError(BAD_ARGS, f"role must be esp|stm, got {role_s!r}") from e
        io = self._midi_factory(dev, role)
        with self._dev_lock(dev.id):
            io.open()
            try:
                yield io
            finally:
                io.close()

    def _op_midi_sysex(self, args: dict) -> dict:
        hex_s = _arg(args, "frame", str)
        try:
            frame = bytes.fromhex(hex_s.replace(" ", "").replace(",", ""))
        except ValueError as e:
            raise HilError(BAD_ARGS, f"frame is not a hex string: {hex_s!r}") from e
        with self._midi(args) as io:
            io.send_sysex(frame)
        return {"sent": len(frame)}

    def _op_midi_note(self, args: dict) -> dict:
        on = _arg(args, "on", bool)
        note = _arg(args, "note", int)
        vel = _arg(args, "vel", int, 100)
        channel = _arg(args, "channel", int, 0)
        if not 0 <= note <= 127 or not 0 <= vel <= 127 or not 0 <= channel <= 15:
            raise HilError(BAD_ARGS, "note/vel 0..127, channel 0..15")
        with self._midi(args) as io:
            io.send_note(on, note, vel, channel)
        return {"sent": 1, "on": on, "note": note, "vel": vel, "channel": channel}

    def _op_midi_echo_rtt(self, args: dict) -> dict:
        n = _arg(args, "n", int, 20)
        with self._midi(args) as io:
            return echo_rtt(io, n=n)

    def _op_midi_query_route(self, args: dict) -> dict:
        with self._midi(args) as io:
            return query_route(io)

    # ----------------------------------------------------------------- ops: usbmode / ota

    def _op_usbmode_set(self, args: dict) -> dict:
        dev = self._resolve(args)
        mode_s = _arg(args, "mode", str)
        try:
            mode = UsbMode(mode_s)
        except ValueError as e:
            raise HilError(BAD_ARGS, f"mode must be default|audio, got {mode_s!r}") from e
        if mode not in (UsbMode.DEFAULT, UsbMode.AUDIO):
            raise HilError(BAD_ARGS, f"mode must be default|audio, got {mode_s!r}")
        wait = _arg(args, "wait", bool, True)
        with self._dev_lock(dev.id), hub_link.exclusive("usb-mode set"):
            new = self._set_mode_fn(dev, mode, wait=wait,
                                    discover_fn=lambda: discover(self._backends),
                                    midi_factory=self._midi_factory)
        return new.to_dict()

    def _op_ota_flash(self, args: dict) -> dict:
        dev = self._resolve(args)
        firmware = Path(_arg(args, "firmware", str))
        if not firmware.is_file():
            raise HilError(BAD_ARGS, f"firmware not found: {firmware}")
        delta_base = _arg(args, "delta_base", str, None)
        wait_boot = _arg(args, "wait_boot", bool, True)
        console_port = self._console_port(dev) if wait_boot else None

        def body(task: _Task) -> dict:
            def progress(done: int, total: int) -> None:
                task.progress, task.total, task.message = done, total, f"{done}/{total} bytes"
                self.emit({"ev": "task.progress", "task": task.handle, "progress": done,
                           "total": total, "message": task.message})

            with self._dev_lock(dev.id), self._bench_hold(dev.id, args), \
                    hub_link.exclusive("ota flash"):
                flashed = self._ota_flash_fn(
                    dev, firmware, delta_base=Path(delta_base) if delta_base else None,
                    progress=progress, serial_factory=self._serial_factory,
                    set_mode_fn=self._set_mode_fn)
                boot: dict | None = None
                if console_port is not None:
                    con = Console(console_port, serial_factory=self._serial_factory)
                    con.open(reset=False)
                    try:
                        boot = dataclasses.asdict(con.wait_boot())
                    finally:
                        con.close()
            return {"flash": flashed, "boot": boot}

        return {"task": self._start_task("ota.flash", body)}

    # ----------------------------------------------------------------- ops: snapshot

    def _op_snapshot_take(self, args: dict) -> dict:
        dev = self._resolve(args)
        include = _arg(args, "include", list, None)
        previous = _arg(args, "previous", dict, None)
        prev_obj: Snapshot | None = None
        if previous is not None:
            names = {f.name for f in dataclasses.fields(Snapshot)}
            prev_obj = Snapshot(**{k: v for k, v in previous.items() if k in names})
        console = None
        console_sess: _ConsoleSession | None = None
        if args.get("console_handle") is not None:
            console_sess = self._session(args, _ConsoleSession, key="console_handle")
            console = console_sess.console
        kw: dict[str, Any] = {"console": console, "previous": prev_obj,
                              "counter": self._snap_counter}
        if include is not None:
            kw["include"] = tuple(str(i) for i in include)
        if args.get("cdc_handle") is not None:
            sess = self._session(args, _CdcSession, key="cdc_handle")
            with sess.lock:
                return take_snapshot(dev, sess.link, **kw).to_dict()
        with self._dev_lock(dev.id):
            link = self._open_cdc(dev)
            try:
                return take_snapshot(dev, link, **kw).to_dict()
            finally:
                link.close()

    # ----------------------------------------------------------------- ops: stimulus

    def _op_stim_start(self, args: dict) -> dict:
        """Start playing the pads. Returns a ``stim_*`` handle; the schedule runs on its own.

        ``cdc_handle`` makes the stimulus share an already-open CDC session instead of
        opening a second one — the CDC port takes an exclusive lock, so without it a caller
        cannot read ``KIT_STATUS`` while its own stimulus is playing.
        """
        dev = self._resolve(args)
        kw: dict[str, Any] = {
            "seconds": float(_arg(args, "seconds", (int, float), stim_mod.DEFAULT_SECONDS)),
            "velocity": args.get("velocity", stim_mod.DEFAULT_VELOCITY),
            "gate_ms": float(_arg(args, "gate_ms", (int, float), stim_mod.DEFAULT_GATE_MS)),
            "humanize_ms": float(_arg(args, "humanize_ms", (int, float), 0.0)),
            "chord": bool(_arg(args, "chord", bool, False)),
            "transport": _arg(args, "transport", str, "cdc"),
            "seed": args.get("seed"),
            "backoff": bool(_arg(args, "backoff", bool, True)),
            "serial_factory": self._serial_factory,
            "midi_factory": self._midi_factory,
        }
        pattern = _arg(args, "pattern", list, None)
        if pattern is not None:
            kw["pattern"] = pattern
        else:
            kw["pads"] = args.get("pads")
            kw["rate_hz"] = args.get("rate_hz")
        hold: hub_link.ExclusiveHold | None = None
        if args.get("cdc_handle") is not None:
            kw["link"] = self._session(args, _CdcSession, key="cdc_handle").link
            stim = self._stim_start_fn(dev, **kw)
        else:
            # Its own links open the port directly: the hub steps aside until stim.stop.
            hold = hub_link.hold_exclusive("stim")
            try:
                with self._dev_lock(dev.id):
                    stim = self._stim_start_fn(dev, **kw)
            except BaseException:
                hold.release()
                raise
        handle = self._handles.mint("stim", _StimSession(stim, dev.id, hold=hold),
                                    self.STIM_TTL_S)
        return {"handle": handle, "device": dev.id, "plan": stim.plan.to_dict()}

    def _op_stim_status(self, args: dict) -> dict:
        sess = self._session(args, _StimSession)
        if not sess.stim.running and sess.last_status is not None:
            return sess.last_status
        return sess.stim.status()

    def _op_stim_stop(self, args: dict) -> dict:
        handle = _arg(args, "handle", str)
        sess = self._session(args, _StimSession)
        try:
            status = sess.stim.stop() if sess.stim.running else (
                sess.last_status or sess.stim.status(refresh=False))
        finally:
            sess.release_hold()
        sess.last_status = status
        self._handles.drop(handle)
        return status

    # ----------------------------------------------------------------- ops: capture

    def _op_capture_start(self, args: dict) -> dict:
        """Enter the UAC2 profile, route the DAC->ADC loop, start recording.

        Returns a ``cap_*`` handle immediately: the recording runs on the audio
        backend's own thread, so the caller is free to play the pads meanwhile —
        which is the only way a capture of anything is going to happen.
        """
        dev = self._resolve(args)
        kw: dict[str, Any] = {
            "preset": _arg(args, "preset", str, cap_mod.DEFAULT_PRESET),
            "resume_audio_tasks": bool(_arg(args, "resume_audio_tasks", bool, True)),
            "keep_mode": bool(_arg(args, "keep_mode", bool, False)),
            "channels": int(_arg(args, "channels", int, cap_mod.CHANNELS_DEFAULT)),
            "midi_factory": self._midi_factory,
            "set_mode_fn": self._set_mode_fn,
            "discover_fn": lambda: discover(self._backends),
            "log": self._log,
        }
        for name, typ in (("seconds", (int, float)), ("volume", int), ("sample_rate", int),
                          ("out", str), ("logdir", str)):
            if args.get(name) is not None:
                kw[name] = _arg(args, name, typ)
        sess = self._capture_factory(dev, **kw)
        hold = hub_link.hold_exclusive("capture")
        try:
            with self._dev_lock(dev.id):
                info = sess.start()
        except BaseException:
            hold.release()
            raise
        handle = self._handles.mint("cap", _CaptureSession(sess, dev.id, hold=hold),
                                    self.CAPTURE_TTL_S)
        return {"handle": handle, **info}

    def _op_capture_health(self, args: dict) -> dict:
        sess = self._session(args, _CaptureSession)
        return sess.capture.health()

    def _op_capture_stop(self, args: dict) -> dict:
        handle = _arg(args, "handle", str)
        sess = self._session(args, _CaptureSession)
        try:
            if sess.capture.running:
                with self._dev_lock(sess.device_id):
                    sess.last_result = sess.capture.stop().to_dict()
        finally:
            sess.release_hold()
        result = sess.last_result or {}
        self._handles.drop(handle)
        return result

    # ----------------------------------------------------------------- ops: analyze

    def _op_analyze_wav(self, args: dict) -> dict:
        """Offline: a WAV in, one typed verdict out. Touches no device at all."""
        kind = _arg(args, "kind", str)
        wav = _arg(args, "wav", str)
        options = _arg(args, "options", dict, None) or {}
        return self._analyze_fn(kind, wav, expected=args.get("expected"), **options)

    # ----------------------------------------------------------------- ops: ble (host radio)

    def _op_ble_scan(self, args: dict) -> dict:
        timeout_s = float(_arg(args, "timeout_s", (int, float),
                               ble_mod.DEFAULT_SCAN_TIMEOUT_S))
        found = self._ble_scan_fn(timeout_s)
        return {"count": len(found), "peripherals": [p.to_dict() for p in found]}

    def _op_ble_connect(self, args: dict) -> dict:
        """Connect this PC's radio to the board (address from ``BLE_STATUS self=`` by default)."""
        address = _arg(args, "address", str, None)
        timeout_s = float(_arg(args, "scan_timeout_s", (int, float),
                               ble_mod.DEFAULT_SCAN_TIMEOUT_S))
        device_id: str | None = None
        if address is not None:
            ble = self._ble_connect_fn(address, link=None, scan_timeout_s=timeout_s)
        elif args.get("cdc_handle") is not None:
            sess = self._session(args, _CdcSession, key="cdc_handle")
            with sess.lock:
                ble = self._ble_connect_fn(None, link=sess.link, scan_timeout_s=timeout_s)
            device_id = sess.device_id
        else:
            dev = self._resolve(args)
            device_id = dev.id
            with self._dev_lock(dev.id):
                link = self._open_cdc(dev)
                try:
                    ble = self._ble_connect_fn(None, link=link, scan_timeout_s=timeout_s)
                finally:
                    link.close()
        handle = self._handles.mint("ble", _BleSession(ble, device_id), self._ttl)
        return {"handle": handle, **ble.status()}

    def _op_ble_status(self, args: dict) -> dict:
        return self._session(args, _BleSession).link.status()

    def _op_ble_send(self, args: dict) -> dict:
        sess = self._session(args, _BleSession)
        if args.get("messages") is not None:
            msgs = ble_mod.messages_from(_arg(args, "messages", list))
        else:
            note = _arg(args, "note", int)
            vel = _arg(args, "vel", int, 100)
            on = _arg(args, "on", bool, True)
            msgs = [ble_mod.note_message(on, note, vel, _arg(args, "channel", int, 0))]
        return sess.link.send(msgs)

    def _op_ble_listen(self, args: dict) -> dict:
        sess = self._session(args, _BleSession)
        seconds = float(_arg(args, "seconds", (int, float), 2.0))
        expect = _arg(args, "expect", int, None)
        return sess.link.listen(seconds, expect=expect)

    def _op_ble_latency(self, args: dict) -> dict:
        sess = self._session(args, _BleSession)
        n = _arg(args, "n", int, 8)
        if args.get("cdc_handle") is not None:
            cdc_sess = self._session(args, _CdcSession, key="cdc_handle")
            with cdc_sess.lock:
                return ble_mod.latency(sess.link, cdc_sess.link, n=n)
        dev = self._resolve(args)
        with self._dev_lock(dev.id):
            link = self._open_cdc(dev)
            try:
                return ble_mod.latency(sess.link, link, n=n)
            finally:
                link.close()

    def _op_ble_disconnect(self, args: dict) -> dict:
        handle = _arg(args, "handle", str)
        sess = self._handles.drop(handle)
        if sess is None:
            raise HilError(HANDLE_EXPIRED, f"unknown handle {handle!r}", hint="connect again")
        self._safe_close(handle, sess)
        return {"ok": True}

    # ----------------------------------------------------------------- ops: diagnose

    def _op_diagnose_crash(self, args: dict) -> dict:
        """Crash bundle from a log file, an open console handle, or the board's console now."""
        kw: dict[str, Any] = {
            "elf": _arg(args, "elf", str, None),
            "addr2line": _arg(args, "addr2line", str, None),
            "root": _arg(args, "root", str, None),
            "context_dir": _arg(args, "context_dir", str, "hil_logs"),
            "context_lines": _arg(args, "context_lines", int, diag.DEFAULT_CONTEXT_LINES),
            "decode": _arg(args, "decode", bool, True),
        }
        if args.get("log_file") is not None:
            return self._diagnose_fn(log_file=_arg(args, "log_file", str), **kw)
        if args.get("console_handle") is not None:
            sess = self._session(args, _ConsoleSession, key="console_handle")
            lines = [line for _seq, line in sess.console.read(since_seq=0, limit=50_000).lines]
            return self._diagnose_fn(lines=lines, source=f"console:{sess.device_id}", **kw)
        dev = self._resolve(args)
        seconds = float(_arg(args, "seconds", (int, float), 3.0))
        with self._dev_lock(dev.id):
            con = Console(self._console_port(dev), serial_factory=self._serial_factory)
            con.open()
            try:
                time.sleep(seconds)
                lines = [line for _seq, line in con.read(since_seq=0, limit=50_000).lines]
            finally:
                con.close()
        return self._diagnose_fn(lines=lines, source=f"console:{dev.id}", **kw)

    # ----------------------------------------------------------------- ops: scenarios / tasks

    def _op_scenario_list(self, args: dict) -> dict:
        out = []
        for name in scenario_names():
            s = get_scenario(name)
            params = [{"name": p["name"], "type": p["type"], "default": p["default"],
                       "help": p["help"]} for p in scen.params_schema(s.Params)]
            out.append({"name": name, "description": s.description, "params": params})
        return {"scenarios": out}

    def _op_scenario_run(self, args: dict) -> dict:
        name = _arg(args, "name", str)
        raw = _arg(args, "params", dict, {})
        try:
            s = get_scenario(name)
        except (KeyError, HilError) as e:
            raise HilError(BAD_ARGS, f"unknown scenario {name!r}",
                           hint="scenario.list names them") from e
        params = scen.params_from_dict(s.Params, raw)  # unknown key / bad type -> BAD_ARGS
        dev = self._resolve(args)
        holder = _arg(args, "holder", str, None)

        def body(task: _Task) -> dict:
            with self._dev_lock(dev.id), self._bench_hold(dev.id, args):
                report = self._run_scenario_fn(name, params, device=dev,
                                               progress=DaemonProgress(self, task),
                                               cancelled=task.cancelled, log=self._log,
                                               **({"holder": holder} if holder else {}))
            return dataclasses.asdict(report)

        return {"task": self._start_task(f"scenario:{name}", body)}

    def _op_task_status(self, args: dict) -> dict:
        return self._task(args).to_dict()

    def _op_task_wait(self, args: dict) -> dict:
        task = self._task(args)
        timeout_s = float(_arg(args, "timeout_s", (int, float), 30.0))
        task.done.wait(timeout_s)
        return task.to_dict()

    def _op_task_cancel(self, args: dict) -> dict:
        task = self._task(args)
        task.cancelled.set()
        if task.future is not None and task.future.cancel():
            task.status = "cancelled"
            task.error = HilError(CANCELLED, "cancelled before start").to_dict()
            task.done.set()
            self.emit({"ev": "task.done", "task": task.handle, "status": "cancelled",
                       "error": task.error})
        return {"ok": True}

    def _op_task_list(self, args: dict) -> dict:
        tasks = []
        for h in self._handles.all_handles():
            if not h.startswith("task_"):
                continue
            try:
                t = self._handles.get(h)
            except HilError:
                continue
            if isinstance(t, _Task):
                tasks.append(t.to_dict())
        return {"tasks": tasks}

    # ----------------------------------------------------------------- ops: files

    def _op_fs_pull(self, args: dict) -> dict:
        """Device file -> host file. Binary framing leaves line mode: an exclusive op."""
        dev = self._resolve(args)
        remote = _arg(args, "remote", str)
        local = Path(_arg(args, "local", str))
        with self._dev_lock(dev.id), hub_link.exclusive("fs transfer"):
            return fs_mod.pull(dev, remote, local, serial_factory=self._serial_factory,
                               set_mode_fn=self._set_mode_fn)

    def _op_fs_push(self, args: dict) -> dict:
        """Host file -> device path, under an exclusive op like fs.pull."""
        dev = self._resolve(args)
        local = Path(_arg(args, "local", str))
        remote = _arg(args, "remote", str)
        with self._dev_lock(dev.id), hub_link.exclusive("fs transfer"):
            return fs_mod.push(dev, local, remote, serial_factory=self._serial_factory,
                               set_mode_fn=self._set_mode_fn)

    # ----------------------------------------------------------------- ops: hub

    def _op_hub_status(self, args: dict) -> dict:
        """The hub's clients, exclusive op and CDC state; {"running": false} when no hub
        answers. Never starts one."""
        return midi_hub.control_status() or {"running": False}

def main(argv: list[str] | None = None) -> int:  # pragma: no cover - `crosspad-hil serve` path
    Daemon(sys.stdin, sys.stdout).run()
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
