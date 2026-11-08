"""Cross-process port locks (spec §2.4, §4.3).

One lock file per port under ``$XDG_RUNTIME_DIR/crosspad-hil`` (or a per-user
temp dir), created with ``O_CREAT|O_EXCL`` so two processes cannot both win.
The file holds ``{"pid", "purpose", "port", "ts"}`` so ``doctor`` and the
``PORT_BUSY`` error can say *who* has the port and *why*. A lock whose pid is
no longer alive is stale and reclaimed with a log line; no psutil needed.
"""
from __future__ import annotations

import getpass
import hashlib
import json
import logging
import os
import stat
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

from crosspad_hil.errors import ENV, PORT_BUSY, HilError

_log = logging.getLogger("crosspad_hil.locks")

_RECLAIM_ATTEMPTS = 3


def default_lock_dir() -> Path:
    """``$XDG_RUNTIME_DIR/crosspad-hil`` when set, else ``<tmp>/crosspad-hil-<uid>``."""
    xdg = os.environ.get("XDG_RUNTIME_DIR")
    if xdg:
        return Path(xdg) / "crosspad-hil"
    uid: str
    if hasattr(os, "getuid"):
        uid = str(os.getuid())
    else:
        try:
            uid = getpass.getuser()
        except Exception:  # noqa: BLE001 - no user database, nothing better to key on
            uid = "user"
    return Path(tempfile.gettempdir()) / f"crosspad-hil-{uid}"


def lock_file_name(port: str) -> str:
    """``sha1(port)[:12] + ".lock"`` — a port path is not a safe file name."""
    return hashlib.sha1(port.encode("utf-8")).hexdigest()[:12] + ".lock"


def pid_alive(pid: int) -> bool:
    """True when ``pid`` names a running process. POSIX: ``kill(pid, 0)``;
    Windows: ``OpenProcess`` with query-limited rights."""
    if pid <= 0:
        return False
    if pid == os.getpid():
        return True
    if sys.platform == "win32":
        import ctypes

        process_query_limited_information = 0x1000
        kernel32 = ctypes.windll.kernel32  # type: ignore[attr-defined]
        handle = kernel32.OpenProcess(process_query_limited_information, False, pid)
        if not handle:
            return False
        kernel32.CloseHandle(handle)
        return True
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True  # exists, owned by someone else
    except OverflowError:
        return False
    return True


# Never through a link: without $XDG_RUNTIME_DIR the lock dir is under the
# shared temp dir. 0 where the platform has no such flag (Windows).
_NOFOLLOW = getattr(os, "O_NOFOLLOW", 0)


def private_lock_dir(directory: Path) -> Path:
    """`directory`, made if missing, as a real directory of this user that
    nobody else may write into -- or HilError(ENV).

    One this user owns but left group- or world-writable (made under umask
    0002 before this check) is narrowed to 0700. A link, or a directory
    somebody else owns, is refused: whoever owns it decides what a lock there
    says. Once it is ours and private nobody can swap it -- the temp dir is
    sticky and $XDG_RUNTIME_DIR is the user's own.
    """
    directory = Path(directory)
    directory.parent.mkdir(parents=True, exist_ok=True)
    try:
        directory.mkdir(mode=0o700)
    except FileExistsError:
        pass
    if not hasattr(os, "getuid"):
        return directory
    st = os.lstat(directory)
    if not stat.S_ISDIR(st.st_mode) or st.st_uid != os.getuid():
        raise HilError(ENV, f"lock dir {directory} is not this user's own directory",
                       hint=f"remove {directory} (a link, or made by another user) and retry")
    if st.st_mode & (stat.S_IWGRP | stat.S_IWOTH):
        os.chmod(directory, 0o700)
    return directory


def _read_holder(path: Path) -> dict[str, Any] | None:
    """Parse a lock file; None when missing, corrupt or a link."""
    try:
        fd = os.open(str(path), os.O_RDONLY | _NOFOLLOW)
        with os.fdopen(fd, encoding="utf-8") as fh:
            data = json.loads(fh.read())
    except (OSError, ValueError):
        return None
    if not isinstance(data, dict) or not isinstance(data.get("pid"), int):
        return None
    return data


class PortLock:
    """Exclusive, cross-process claim on one serial port path."""

    def __init__(self, port: str, purpose: str, lock_dir: Path | None = None) -> None:
        self.port = port
        self.purpose = purpose
        self.lock_dir = Path(lock_dir) if lock_dir is not None else default_lock_dir()
        self.path = self.lock_dir / lock_file_name(port)
        self._held = False

    # -- acquire / release -------------------------------------------------

    def acquire(self) -> None:
        """Create the lock file exclusively; reclaim a stale one; else PORT_BUSY."""
        if self._held:
            return
        private_lock_dir(self.lock_dir)
        payload = json.dumps(
            {"pid": os.getpid(), "purpose": self.purpose, "port": self.port, "ts": time.time()}
        )
        for _attempt in range(_RECLAIM_ATTEMPTS):
            try:
                fd = os.open(str(self.path),
                             os.O_CREAT | os.O_EXCL | os.O_WRONLY | _NOFOLLOW, 0o600)
            except FileExistsError:
                holder = _read_holder(self.path)
                if holder is not None and pid_alive(int(holder["pid"])):
                    raise HilError(
                        PORT_BUSY,
                        f"{self.port} is held by pid {holder['pid']} "
                        f"({holder.get('purpose', '?')})",
                        hint=(
                            f"close that session, or if pid {holder['pid']} is not a "
                            f"crosspad-hil process delete {self.path}"
                        ),
                        pid=int(holder["pid"]),
                        purpose=str(holder.get("purpose", "")),
                        port=self.port,
                        lock_file=str(self.path),
                    ) from None
                # stale (dead pid) or corrupt: reclaim
                _log.warning(
                    "reclaiming stale lock %s (pid=%s purpose=%s)",
                    self.path,
                    None if holder is None else holder.get("pid"),
                    None if holder is None else holder.get("purpose"),
                )
                try:
                    self.path.unlink()
                except FileNotFoundError:
                    pass
                continue
            with os.fdopen(fd, "w", encoding="utf-8") as fh:
                fh.write(payload)
            self._held = True
            return
        raise HilError(
            PORT_BUSY,
            f"{self.port}: lock file {self.path} keeps reappearing",
            hint="another process is racing for this port; retry",
            port=self.port,
            lock_file=str(self.path),
            pid=0,
            purpose="",
        )

    def release(self) -> None:
        """Remove the lock file if this process wrote it. Idempotent."""
        holder = _read_holder(self.path)
        if holder is None:
            self._held = False
            return
        if holder["pid"] != os.getpid():
            self._held = False
            return
        try:
            self.path.unlink()
        except FileNotFoundError:
            pass
        self._held = False

    def __enter__(self) -> PortLock:
        self.acquire()
        return self

    def __exit__(self, exc_type: object, exc: object, tb: object) -> None:
        self.release()

    # -- inspection --------------------------------------------------------

    @staticmethod
    def holders(lock_dir: Path | None = None) -> list[dict]:
        """Every lock file in ``lock_dir`` as ``{port, pid, purpose, alive}``."""
        directory = Path(lock_dir) if lock_dir is not None else default_lock_dir()
        if not directory.is_dir():
            return []
        rows: list[dict] = []
        for path in sorted(directory.glob("*.lock")):
            holder = _read_holder(path)
            if holder is None:
                continue
            pid = int(holder["pid"])
            rows.append(
                {
                    "port": str(holder.get("port", "")),
                    "pid": pid,
                    "purpose": str(holder.get("purpose", "")),
                    "alive": pid_alive(pid),
                }
            )
        return rows
