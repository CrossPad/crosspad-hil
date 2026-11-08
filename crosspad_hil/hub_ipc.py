"""The hub's control socket: one NDJSON line per request, response or event.

Same envelope as `serve` (request {id, op, args}; response {id, ok, result} or
{id, ok: false, error}). The first request on a connection is `hello`
{client, purpose, pid[, token]} — that is what PORT_BUSY reports back to the
next client. POSIX listens on a Unix socket in the per-user lock dir; Windows
on 127.0.0.1 with a random token, since a loopback port is open to every local
user. `hub.json` (0600) says which.

Both ends trust the endpoint only in a directory this user owns and nobody
else can write (`<tmp>/crosspad-hil-<uid>` is a guessable name another user
could create first), and a client that has not said hello gets bounded lines, HELLO_TIMEOUT_S in
all, and only MAX_PENDING_HELLO of them wait at once.
"""
from __future__ import annotations

import itertools
import json
import os
import secrets
import socket
import socketserver
import stat
import sys
import threading
from collections.abc import Callable
from pathlib import Path
from typing import Any

from crosspad_hil.errors import ALL_CODES, ENV, HilError
from crosspad_hil.locks import default_lock_dir

ENDPOINT_NAME = "hub.json"
SOCKET_NAME = "hub.sock"
TOKEN_BYTES = 16
PRIVATE_MODE = 0o600
PRIVATE_DIR_MODE = 0o700
GROUP_OTHER_BITS = 0o077
MAX_LINE_BYTES = 1 << 20
HELLO_TIMEOUT_S = 5.0
MAX_PENDING_HELLO = 16


def endpoint_file(lock_dir: Path | None = None) -> Path:
    return (Path(lock_dir) if lock_dir is not None else default_lock_dir()) / ENDPOINT_NAME


def _unix_available() -> bool:
    return hasattr(socket, "AF_UNIX") and sys.platform != "win32"


def _unsafe(path: Path, why: str) -> HilError:
    return HilError(ENV, f"{path} is not safe to trust: {why}",
                    hint=f"remove {path} (or set XDG_RUNTIME_DIR) and start the hub again",
                    reason="unsafe_dir", path=str(path))


def private_dir(d: Path) -> Path:
    """Create ``d`` owner-only, or accept it only when it is a real directory of ours."""
    d.mkdir(mode=PRIVATE_DIR_MODE, parents=True, exist_ok=True)
    if os.name == "nt":
        return d
    st = os.lstat(d)
    if not stat.S_ISDIR(st.st_mode):
        raise _unsafe(d, "not a directory")
    if st.st_uid != os.getuid():
        raise _unsafe(d, f"owned by uid {st.st_uid}")
    if st.st_mode & GROUP_OTHER_BITS:
        os.chmod(d, PRIVATE_DIR_MODE)
    return d


def _owned_file(path: Path) -> None:
    if os.name == "nt":
        return
    st = os.lstat(path)
    if stat.S_ISLNK(st.st_mode) or st.st_uid != os.getuid():
        raise _unsafe(path, "a link, or owned by someone else")


def _write_private(path: Path, text: str) -> None:
    private_dir(path.parent)
    tmp = path.with_suffix(".tmp")
    try:
        tmp.unlink()
    except FileNotFoundError:
        pass
    flags = os.O_CREAT | os.O_EXCL | os.O_WRONLY | getattr(os, "O_NOFOLLOW", 0)
    fd = os.open(str(tmp), flags, PRIVATE_MODE)
    with os.fdopen(fd, "w", encoding="utf-8") as fh:
        fh.write(text)
    os.replace(tmp, path)


def _error_from_dict(err: dict) -> HilError:
    code = err.get("code") if err.get("code") in ALL_CODES else ENV
    return HilError(code, str(err.get("message", "")), hint=err.get("hint"),
                    **dict(err.get("details") or {}))


class Session:
    _ids = itertools.count(1)

    def __init__(self, wfile: Any, wlock: threading.Lock) -> None:
        self.id = next(Session._ids)
        self.client = "?"
        self.purpose = ""
        self.pid = 0
        self._wfile = wfile
        self._wlock = wlock

    def _write(self, obj: dict) -> None:
        data = (json.dumps(obj, default=str) + "\n").encode("utf-8")
        with self._wlock:
            self._wfile.write(data)
            self._wfile.flush()

    def send_event(self, ev: dict) -> None:
        self._write(ev)


class Server:
    def __init__(self, handler: Callable[[Session, dict], Any], *,
                 lock_dir: Path | None = None, use_unix: bool | None = None) -> None:
        self._handler = handler
        self._dir = Path(lock_dir) if lock_dir is not None else default_lock_dir()
        self._unix = _unix_available() if use_unix is None else use_unix
        self._token = "" if self._unix else secrets.token_hex(TOKEN_BYTES)
        self._srv: socketserver.BaseServer | None = None
        self._thread: threading.Thread | None = None
        self.on_connect: Callable[[Session], None] | None = None
        self.on_disconnect: Callable[[Session], None] | None = None
        self._pending = 0
        self._pending_lock = threading.Lock()

    def start(self) -> None:
        outer = self

        class _H(socketserver.StreamRequestHandler):
            timeout = HELLO_TIMEOUT_S

            def handle(self) -> None:
                outer._serve_connection(self.rfile, self.wfile, self.connection)

        private_dir(self._dir)
        if self._unix:
            path = self._dir / SOCKET_NAME
            try:
                path.unlink()
            except FileNotFoundError:
                pass
            srv: socketserver.BaseServer = socketserver.ThreadingUnixStreamServer(str(path), _H)
            os.chmod(path, PRIVATE_MODE)
            info = {"kind": "unix", "path": str(path), "pid": os.getpid()}
        else:
            srv = socketserver.ThreadingTCPServer(("127.0.0.1", 0), _H)
            info = {"kind": "tcp", "port": srv.server_address[1], "token": self._token,
                    "pid": os.getpid()}
        srv.daemon_threads = True
        self._srv = srv
        _write_private(endpoint_file(self._dir), json.dumps(info))
        self._thread = threading.Thread(target=srv.serve_forever, name="hub-ipc", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        srv, self._srv = self._srv, None
        if srv is not None:
            srv.shutdown()
            srv.server_close()
        for p in (endpoint_file(self._dir), self._dir / SOCKET_NAME):
            if p.name == SOCKET_NAME and not self._unix:
                continue
            try:
                p.unlink()
            except FileNotFoundError:
                pass

    def _admit(self) -> bool:
        with self._pending_lock:
            if self._pending >= MAX_PENDING_HELLO:
                return False
            self._pending += 1
            return True

    def _admitted(self) -> None:
        with self._pending_lock:
            self._pending -= 1

    def _serve_connection(self, rfile: Any, wfile: Any, sock: socket.socket) -> None:
        if not self._admit():
            return
        session = Session(wfile, threading.Lock())
        greeted = False
        waiting = True

        def _drop() -> None:
            try:
                sock.shutdown(socket.SHUT_RDWR)
            except OSError:
                pass

        # The per-recv timeout alone lets a client trickle one byte at a time.
        deadline = threading.Timer(HELLO_TIMEOUT_S, _drop)
        deadline.daemon = True
        deadline.start()
        try:
            while True:
                raw = rfile.readline(MAX_LINE_BYTES + 1)
                if not raw or len(raw) > MAX_LINE_BYTES:
                    return
                try:
                    req = json.loads(raw.decode("utf-8"))
                    rid = req.get("id")
                except (ValueError, AttributeError):
                    continue
                try:
                    if req.get("op") == "hello":
                        args = req.get("args") or {}
                        if self._token and args.get("token") != self._token:
                            raise HilError(ENV, "bad hub token", reason="bad_token")
                        session.client = str(args.get("client", "?"))
                        session.purpose = str(args.get("purpose", ""))
                        session.pid = int(args.get("pid", 0))
                        greeted = True
                        deadline.cancel()
                        sock.settimeout(None)
                        waiting = False
                        self._admitted()
                        if self.on_connect is not None:
                            self.on_connect(session)
                        result: Any = {"session": session.id}
                    elif not greeted:
                        raise HilError(ENV, "hello first", reason="no_hello")
                    else:
                        result = self._handler(session, req)
                    session._write({"id": rid, "ok": True, "result": result})
                except HilError as e:
                    session._write({"id": rid, "ok": False, "error": e.to_dict()})
                    if not greeted:
                        return
        except (OSError, ValueError):
            pass
        finally:
            deadline.cancel()
            if waiting:
                self._admitted()
            if greeted and self.on_disconnect is not None:
                self.on_disconnect(session)


class Connection:
    def __init__(self, sock: socket.socket) -> None:
        self._sock = sock
        self._r = sock.makefile("rb")
        self._ids = itertools.count(1)
        self._lock = threading.Lock()
        self.closed = False

    @classmethod
    def open(cls, lock_dir: Path | None = None, *, client: str, purpose: str,
             timeout_s: float = 2.0) -> Connection:
        f = endpoint_file(lock_dir)
        if f.parent.exists():
            private_dir(f.parent)
        try:
            _owned_file(f)
            info = json.loads(f.read_text(encoding="utf-8"))
            if info.get("kind") == "unix":
                sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
                sock.settimeout(timeout_s)
                sock.connect(info["path"])
            else:
                sock = socket.create_connection(("127.0.0.1", int(info["port"])), timeout_s)
        except (OSError, ValueError, KeyError, TypeError) as exc:
            raise HilError(ENV, "the CrossPad hub is not running",
                           hint="crosspad-hil midi hub run (or let a client start it)",
                           reason="no_hub", cause=repr(exc)) from None
        conn = cls(sock)
        try:
            conn.call("hello", {"client": client, "purpose": purpose, "pid": os.getpid(),
                                "token": info.get("token", "")}, timeout_s=timeout_s)
        except HilError:
            conn.close()
            raise
        return conn

    def call(self, op: str, args: dict | None = None, timeout_s: float = 10.0) -> Any:
        with self._lock:
            if self.closed:
                raise HilError(ENV, "the connection to the hub is closed", reason="hub_gone")
            rid = next(self._ids)
            try:
                self._sock.settimeout(timeout_s)
                self._sock.sendall((json.dumps({"id": rid, "op": op, "args": args or {}})
                                    + "\n").encode("utf-8"))
            except OSError:
                self.close()
                raise HilError(ENV, "hub closed the connection", reason="hub_gone") from None
            while True:
                try:
                    raw = self._r.readline()
                except TimeoutError:
                    # A timed-out socket file refuses every later read: this
                    # connection is finished, the caller opens a new one.
                    self.close()
                    raise HilError(ENV, f"hub did not answer {op} within {timeout_s:g}s",
                                   reason="hub_timeout") from None
                except OSError:
                    self.close()
                    raise HilError(ENV, "hub closed the connection", reason="hub_gone") from None
                if not raw:
                    self.close()
                    raise HilError(ENV, "hub closed the connection", reason="hub_gone")
                msg = json.loads(raw.decode("utf-8"))
                if msg.get("id") != rid:
                    continue
                if msg.get("ok"):
                    return msg.get("result")
                raise _error_from_dict(msg.get("error") or {})

    def close(self) -> None:
        self.closed = True
        for c in (self._r, self._sock):
            try:
                c.close()
            except OSError:
                pass
