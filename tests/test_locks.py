"""PortLock: O_EXCL lock files with holder pid/purpose; stale locks reclaimed."""
from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from crosspad_hil import locks
from crosspad_hil.errors import PORT_BUSY, HilError
from crosspad_hil.locks import PortLock


def _lock_path(lock_dir: Path, port: str) -> Path:
    return lock_dir / locks.lock_file_name(port)


def test_lock_file_name_is_sha1_prefix() -> None:
    name = locks.lock_file_name("/dev/ttyACM0")
    assert name.endswith(".lock")
    assert len(name) == 12 + len(".lock")
    assert name == locks.lock_file_name("/dev/ttyACM0")
    assert name != locks.lock_file_name("/dev/ttyACM1")


def test_acquire_writes_json_and_release_removes(tmp_path: Path) -> None:
    lock = PortLock("/dev/ttyACM0", "console", lock_dir=tmp_path)
    lock.acquire()
    p = _lock_path(tmp_path, "/dev/ttyACM0")
    data = json.loads(p.read_text())
    assert data["pid"] == os.getpid()
    assert data["purpose"] == "console"
    assert data["port"] == "/dev/ttyACM0"
    assert isinstance(data["ts"], float)
    lock.release()
    assert not p.exists()


def test_context_manager_releases_on_exception(tmp_path: Path) -> None:
    p = _lock_path(tmp_path, "COM3")
    with pytest.raises(ValueError):
        with PortLock("COM3", "cdc", lock_dir=tmp_path):
            assert p.exists()
            raise ValueError("inside")
    assert not p.exists()


def test_busy_lock_raises_port_busy_with_pid_and_purpose(tmp_path: Path) -> None:
    first = PortLock("/dev/ttyACM1", "console", lock_dir=tmp_path)
    first.acquire()
    try:
        second = PortLock("/dev/ttyACM1", "cdc", lock_dir=tmp_path)
        with pytest.raises(HilError) as ei:
            second.acquire()
        err = ei.value
        assert err.code == PORT_BUSY
        assert err.details["pid"] == os.getpid()
        assert err.details["purpose"] == "console"
        assert err.details["port"] == "/dev/ttyACM1"
        assert err.hint is not None and str(os.getpid()) in err.hint
        # the loser did not touch the winner's file
        assert json.loads(_lock_path(tmp_path, "/dev/ttyACM1").read_text())["purpose"] == "console"
    finally:
        first.release()


def test_stale_lock_is_reclaimed(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    p = _lock_path(tmp_path, "/dev/ttyACM2")
    p.write_text(json.dumps({"pid": 999_999_999, "purpose": "console",
                             "port": "/dev/ttyACM2", "ts": 1.0}))
    monkeypatch.setattr(locks, "pid_alive", lambda pid: pid == os.getpid())
    lock = PortLock("/dev/ttyACM2", "cdc", lock_dir=tmp_path)
    lock.acquire()
    try:
        data = json.loads(p.read_text())
        assert data["pid"] == os.getpid()
        assert data["purpose"] == "cdc"
    finally:
        lock.release()


def test_corrupt_lock_is_reclaimed(tmp_path: Path) -> None:
    p = _lock_path(tmp_path, "/dev/ttyACM3")
    p.write_text("not json at all")
    with PortLock("/dev/ttyACM3", "console", lock_dir=tmp_path):
        assert json.loads(p.read_text())["pid"] == os.getpid()


def test_release_is_idempotent_and_does_not_steal(tmp_path: Path) -> None:
    lock = PortLock("/dev/ttyACM4", "console", lock_dir=tmp_path)
    lock.release()  # never acquired: no error
    p = _lock_path(tmp_path, "/dev/ttyACM4")
    p.write_text(json.dumps({"pid": os.getpid() + 1, "purpose": "x",
                             "port": "/dev/ttyACM4", "ts": 1.0}))
    lock.release()  # not ours: left alone
    assert p.exists()


def test_holders_lists_alive_flag(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(locks, "pid_alive", lambda pid: pid == os.getpid())
    _lock_path(tmp_path, "/dev/dead").write_text(json.dumps(
        {"pid": 424242, "purpose": "console", "port": "/dev/dead", "ts": 2.0}))
    with PortLock("/dev/live", "cdc", lock_dir=tmp_path):
        rows = sorted(PortLock.holders(tmp_path), key=lambda r: r["port"])
    assert rows == [
        {"port": "/dev/dead", "pid": 424242, "purpose": "console", "alive": False},
        {"port": "/dev/live", "pid": os.getpid(), "purpose": "cdc", "alive": True},
    ]


def test_holders_empty_when_dir_missing(tmp_path: Path) -> None:
    assert PortLock.holders(tmp_path / "nope") == []


def test_pid_alive_self_and_impossible() -> None:
    assert locks.pid_alive(os.getpid()) is True
    assert locks.pid_alive(2_000_000_000) is False


def test_default_lock_dir_prefers_xdg(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv("XDG_RUNTIME_DIR", str(tmp_path))
    assert locks.default_lock_dir() == tmp_path / "crosspad-hil"
    monkeypatch.delenv("XDG_RUNTIME_DIR")
    d = locks.default_lock_dir()
    assert d.name.startswith("crosspad-hil-")


@pytest.mark.skipif(not hasattr(os, "getuid"), reason="POSIX ownership")
def test_a_lock_is_never_written_through_a_link(tmp_path: Path) -> None:
    """Without $XDG_RUNTIME_DIR the lock dir is under the shared temp dir."""
    victim = tmp_path / "victim"
    victim.write_text("keep")
    d = tmp_path / "locks"
    d.mkdir(mode=0o700)
    _lock_path(d, "/dev/ttyACM9").symlink_to(victim)
    lock = PortLock("/dev/ttyACM9", "cdc", lock_dir=d)
    lock.acquire()     # the link reads as no holder: reclaimed, a real file made
    assert victim.read_text() == "keep"
    assert not _lock_path(d, "/dev/ttyACM9").is_symlink()
    lock.release()


@pytest.mark.skipif(not hasattr(os, "getuid"), reason="POSIX ownership")
def test_the_lock_dir_is_narrowed_to_this_user(tmp_path: Path) -> None:
    import stat
    d = tmp_path / "locks"
    d.mkdir()
    d.chmod(0o775)          # what umask 0002 made before
    with PortLock("/dev/ttyACM8", "cdc", lock_dir=d):
        assert stat.S_IMODE(d.stat().st_mode) == 0o700
        assert stat.S_IMODE(_lock_path(d, "/dev/ttyACM8").stat().st_mode) == 0o600


@pytest.mark.skipif(not hasattr(os, "getuid"), reason="POSIX ownership")
def test_a_lock_dir_that_is_a_link_is_refused(tmp_path: Path) -> None:
    real = tmp_path / "elsewhere"
    real.mkdir(mode=0o700)
    link = tmp_path / "locks"
    link.symlink_to(real)
    with pytest.raises(HilError):
        PortLock("/dev/ttyACM7", "cdc", lock_dir=link).acquire()
