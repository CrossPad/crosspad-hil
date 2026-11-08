"""hub_ipc: endpoint file, hello, request/response, errors, disconnect callback."""
from __future__ import annotations

import os
import stat
import time

import pytest

from crosspad_hil import hub_ipc
from crosspad_hil.errors import ENV, PORT_BUSY, HilError


@pytest.fixture(params=[True, False], ids=["unix", "tcp"])
def server(request, tmp_path):
    if request.param and not hasattr(__import__("socket"), "AF_UNIX"):
        pytest.skip("no AF_UNIX here")
    seen: list = []

    def handler(session, req):
        seen.append((session.client, req["op"]))
        if req["op"] == "echo":
            return {"args": req.get("args", {}), "client": session.client}
        if req["op"] == "busy":
            raise HilError(PORT_BUSY, "held", holder={"client": "x", "purpose": "y",
                                                      "pid": 1, "since": 0.0})
        raise HilError(ENV, "unknown op")

    srv = hub_ipc.Server(handler, lock_dir=tmp_path, use_unix=request.param)
    gone: list = []
    came: list = []
    srv.on_connect = lambda s: came.append(s.client)
    srv.on_disconnect = lambda s: gone.append(s.client)
    srv.came = came
    srv.start()
    yield srv, tmp_path, seen, gone
    srv.stop()


def test_endpoint_file_is_private(server):
    _, d, _, _ = server
    f = hub_ipc.endpoint_file(d)
    assert f.is_file()
    if os.name != "nt":
        assert stat.S_IMODE(f.stat().st_mode) == 0o600


def test_call_round_trip_carries_client(server):
    _, d, seen, _ = server
    c = hub_ipc.Connection.open(d, client="cp-tools", purpose="pads")
    try:
        assert c.call("echo", {"a": 1}) == {"args": {"a": 1}, "client": "cp-tools"}
    finally:
        c.close()
    assert seen == [("cp-tools", "echo")]
    assert server[0].came == ["cp-tools"]


def test_remote_hil_error_is_rebuilt(server):
    _, d, _, _ = server
    c = hub_ipc.Connection.open(d, client="t", purpose="p")
    try:
        with pytest.raises(HilError) as e:
            c.call("busy")
        assert e.value.code == PORT_BUSY
        assert e.value.details["holder"]["client"] == "x"
    finally:
        c.close()


def test_disconnect_callback(server):
    _, d, _, gone = server
    c = hub_ipc.Connection.open(d, client="gone-client", purpose="p")
    c.call("echo")
    c.close()
    deadline = time.monotonic() + 2
    while not gone and time.monotonic() < deadline:
        time.sleep(0.01)
    assert gone == ["gone-client"]


def test_no_endpoint_is_env_no_hub(tmp_path):
    with pytest.raises(HilError) as e:
        hub_ipc.Connection.open(tmp_path, client="t", purpose="p")
    assert e.value.code == ENV and e.value.details["reason"] == "no_hub"


def test_stop_removes_endpoint(tmp_path):
    srv = hub_ipc.Server(lambda s, r: None, lock_dir=tmp_path, use_unix=False)
    srv.start()
    srv.stop()
    assert not hub_ipc.endpoint_file(tmp_path).exists()


def test_bad_token_refused(tmp_path):
    srv = hub_ipc.Server(lambda s, r: {"ok": True}, lock_dir=tmp_path, use_unix=False)
    srv.start()
    try:
        f = hub_ipc.endpoint_file(tmp_path)
        info = __import__("json").loads(f.read_text())
        info["token"] = "wrong"
        f.write_text(__import__("json").dumps(info))
        with pytest.raises(HilError):
            hub_ipc.Connection.open(tmp_path, client="t", purpose="p")
    finally:
        srv.stop()


# -- trust: who made the endpoint, and what a client may send before hello ------

posix_only = pytest.mark.skipif(os.name == "nt", reason="POSIX ownership and modes")


@posix_only
def test_open_refuses_a_dir_owned_by_someone_else(tmp_path, monkeypatch):
    srv = hub_ipc.Server(lambda s, r: {"ok": True}, lock_dir=tmp_path)
    srv.start()
    try:
        other = os.getuid() + 1
        monkeypatch.setattr(hub_ipc.os, "getuid", lambda: other)
        with pytest.raises(HilError) as e:
            hub_ipc.Connection.open(tmp_path, client="t", purpose="p")
        assert e.value.details["reason"] == "unsafe_dir"
    finally:
        monkeypatch.undo()
        srv.stop()


@posix_only
def test_start_refuses_a_dir_owned_by_someone_else(tmp_path, monkeypatch):
    other = os.getuid() + 1
    monkeypatch.setattr(hub_ipc.os, "getuid", lambda: other)
    with pytest.raises(HilError) as e:
        hub_ipc.Server(lambda s, r: None, lock_dir=tmp_path).start()
    assert e.value.details["reason"] == "unsafe_dir"


@posix_only
def test_open_dir_is_tightened_to_owner_only(tmp_path):
    d = tmp_path / "shared"
    d.mkdir()
    d.chmod(0o777)
    srv = hub_ipc.Server(lambda s, r: None, lock_dir=d, use_unix=False)
    srv.start()
    try:
        assert stat.S_IMODE(d.stat().st_mode) == 0o700
    finally:
        srv.stop()


@posix_only
def test_endpoint_write_does_not_follow_a_symlink(tmp_path):
    victim = tmp_path / "victim.txt"
    victim.write_text("keep me")
    d = tmp_path / "locks"
    d.mkdir(mode=0o700)
    (d / "hub.tmp").symlink_to(victim)
    srv = hub_ipc.Server(lambda s, r: None, lock_dir=d, use_unix=False)
    srv.start()
    try:
        assert victim.read_text() == "keep me"
        assert hub_ipc.endpoint_file(d).is_file()
    finally:
        srv.stop()


def _raw_socket(d):
    info = __import__("json").loads(hub_ipc.endpoint_file(d).read_text())
    if info["kind"] == "unix":
        s = __import__("socket").socket(__import__("socket").AF_UNIX)
        s.connect(info["path"])
    else:
        s = __import__("socket").create_connection(("127.0.0.1", info["port"]))
    s.settimeout(5.0)
    return s


def test_overlong_line_closes_the_connection(server, monkeypatch):
    _, d, _, _ = server
    monkeypatch.setattr(hub_ipc, "MAX_LINE_BYTES", 1024)
    s = _raw_socket(d)
    try:
        s.sendall(b"x" * 4096)
        assert s.recv(1) == b""
    finally:
        s.close()
    c = hub_ipc.Connection.open(d, client="after", purpose="p")
    try:
        assert c.call("echo")["client"] == "after"
    finally:
        c.close()


def test_silent_client_is_dropped_before_hello(tmp_path, monkeypatch):
    monkeypatch.setattr(hub_ipc, "HELLO_TIMEOUT_S", 0.2)
    srv = hub_ipc.Server(lambda s, r: None, lock_dir=tmp_path, use_unix=False)
    srv.start()
    try:
        s = _raw_socket(tmp_path)
        try:
            assert s.recv(1) == b""
        finally:
            s.close()
    finally:
        srv.stop()


def test_a_slow_client_is_dropped_by_the_hello_deadline(tmp_path, monkeypatch):
    monkeypatch.setattr(hub_ipc, "HELLO_TIMEOUT_S", 0.3)
    srv = hub_ipc.Server(lambda s, r: None, lock_dir=tmp_path, use_unix=False)
    srv.start()
    try:
        s = _raw_socket(tmp_path)
        try:
            deadline = time.monotonic() + 3.0
            closed = False
            while time.monotonic() < deadline:
                try:
                    s.sendall(b"x")
                except OSError:
                    closed = True
                    break
                s.settimeout(0.1)
                try:
                    if s.recv(1) == b"":
                        closed = True
                        break
                except TimeoutError:
                    pass
                except ConnectionError:
                    # Windows: the server's close arrives as an abort or a
                    # reset (WinError 10053/10054), not as an empty read.
                    closed = True
                    break
            assert closed
        finally:
            s.close()
    finally:
        srv.stop()


def test_connections_waiting_for_hello_are_capped(tmp_path, monkeypatch):
    monkeypatch.setattr(hub_ipc, "MAX_PENDING_HELLO", 2)
    srv = hub_ipc.Server(lambda s, r: {"ok": True}, lock_dir=tmp_path, use_unix=False)
    srv.start()
    held = [_raw_socket(tmp_path), _raw_socket(tmp_path)]
    try:
        time.sleep(0.2)
        extra = _raw_socket(tmp_path)
        try:
            assert extra.recv(1) == b""
        finally:
            extra.close()
    finally:
        for s in held:
            s.close()
        srv.stop()


def test_a_timed_out_call_leaves_only_hil_errors_behind(tmp_path):
    def handler(session, req):
        if req["op"] == "slow":
            time.sleep(1.0)
        return {"ok": True}

    srv = hub_ipc.Server(handler, lock_dir=tmp_path, use_unix=False)
    srv.start()
    try:
        c = hub_ipc.Connection.open(tmp_path, client="t", purpose="p")
        with pytest.raises(HilError) as first:
            c.call("slow", timeout_s=0.2)
        assert first.value.details["reason"] == "hub_timeout"
        assert c.closed
        with pytest.raises(HilError) as second:
            c.call("fast")
        assert second.value.details["reason"] == "hub_gone"
    finally:
        srv.stop()
