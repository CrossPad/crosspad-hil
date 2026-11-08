"""HubLink against an in-process hub; exclusive ops; autostart; stale endpoint; no-hub."""
from __future__ import annotations

import json
import threading

import pytest

from crosspad_hil import cdc as cdc_mod
from crosspad_hil import hub_ipc, hub_link
from crosspad_hil.cdc import CdcLink, Reply
from crosspad_hil.errors import PORT_BUSY, HilError
from crosspad_hil.hub_core import HubCore
from tests.fakes import FakeSerial
from tests.test_cdc import CDC_KNOWLEDGE, CDCSTATS_0, KITSTATUS
from tests.test_hub_core import _dev

pytestmark = pytest.mark.uses_hub


@pytest.fixture
def hub(monkeypatch):
    monkeypatch.delenv(hub_link.NO_HUB_ENV, raising=False)
    monkeypatch.setattr(cdc_mod, "OPEN_SETTLE_S", 0)

    def link_factory(port):
        link = CdcLink(port, serial_factory=lambda *a, **k: FakeSerial([
            ("KIT_STATUS", KITSTATUS),
            ("CDC_STATS", CDCSTATS_0),
            ("APP_VERSIONS", ["APPVER: a id=- commit=- ref=- dirty=0", "APPVER: end count=1"]),
        ], timeout=0.01), knowledge=CDC_KNOWLEDGE, purpose="hub")
        link.open()
        return link

    core = HubCore(find_board=lambda: _dev(), link_factory=link_factory)
    core._knowledge_verbs = CDC_KNOWLEDGE["verbs"]
    srv = hub_ipc.Server(core.handle)
    srv.on_connect, srv.on_disconnect = core.connect, core.disconnect
    srv.start()
    yield core
    srv.stop()
    core.close()


def test_transact_through_hub(hub):
    link = hub_link.open_link(_dev(), purpose="test")
    try:
        assert isinstance(link, hub_link.HubLink)
        r = link.transact("KIT_STATUS")
        assert isinstance(r, Reply) and r.parsed["name"] == "DRUMS" and r.more_lines == []
        assert link.transact_multi("APP_VERSIONS", "APPVER:", end="APPVER: end")[-1] \
            .startswith("APPVER: end")
    finally:
        link.close()


def test_two_clients_interleave_without_crossing_replies(hub):
    rounds = 25
    got: dict[str, list[str]] = {"a": [], "b": []}

    def client(key, cmd):
        link = hub_link.HubLink(hub_link.connect_or_start(key))
        try:
            for _ in range(rounds):
                got[key].append(link.transact(cmd).line)
        finally:
            link.close()

    threads = [threading.Thread(target=client, args=("a", "KIT_STATUS")),
               threading.Thread(target=client, args=("b", "CDC_STATS"))]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert got == {"a": [KITSTATUS] * rounds, "b": [CDCSTATS_0] * rounds}


def test_exclusive_blocks_other_clients_and_goes_direct_inside(hub):
    other = hub_link.open_link(_dev(), purpose="pads")
    try:
        with hub_link.exclusive("ota flash"):
            assert hub_link.direct() is True
            with pytest.raises(HilError) as e:
                other.transact("KIT_STATUS")
            assert e.value.code == PORT_BUSY
            assert e.value.details["holder"]["purpose"] == "ota flash"
            inner = hub_link.open_link(_dev(), purpose="ota flash",
                                       serial_factory=lambda *a, **k: FakeSerial())
            try:
                assert isinstance(inner, CdcLink)
            finally:
                inner.close()
        assert hub_link.direct() is False
        assert other.transact("KIT_STATUS").line == KITSTATUS
    finally:
        other.close()


def test_exclusive_is_reentrant_in_one_process(hub):
    with hub_link.exclusive("scenario smoke"):
        with hub_link.exclusive("kit push"):
            assert hub_link.holds_exclusive()
        assert hub_link.holds_exclusive()
    assert not hub_link.holds_exclusive()


def test_hold_survives_the_block_until_released(hub):
    other = hub_link.HubLink(hub_link.connect_or_start("pads"))
    hold = hub_link.hold_exclusive("stim")
    try:
        with pytest.raises(HilError):
            other.transact("KIT_STATUS")
        hold.release()
        hold.release()
        assert other.transact("KIT_STATUS").line == KITSTATUS
    finally:
        other.close()


def test_no_hub_env_gives_cdclink(monkeypatch):
    monkeypatch.setenv(hub_link.NO_HUB_ENV, "1")
    monkeypatch.setattr(cdc_mod, "OPEN_SETTLE_S", 0)
    link = hub_link.open_link(_dev(), purpose="test",
                              serial_factory=lambda *a, **k: FakeSerial())
    try:
        assert isinstance(link, CdcLink)
    finally:
        link.close()
    with hub_link.exclusive("ota flash"):
        assert hub_link.holds_exclusive()
    assert not hub_link.holds_exclusive()


def test_stale_endpoint_autostarts(monkeypatch):
    monkeypatch.delenv(hub_link.NO_HUB_ENV, raising=False)
    f = hub_ipc.endpoint_file()
    f.parent.mkdir(parents=True, exist_ok=True)
    f.write_text(json.dumps({"kind": "tcp", "port": 1, "token": "x"}))
    started = threading.Event()
    box = {}

    def spawn():
        srv = hub_ipc.Server(lambda s, r: {"ok": True}, use_unix=False)
        srv.start()
        box["srv"] = srv
        started.set()

    conn = hub_link.connect_or_start("test", spawn=spawn)
    try:
        assert started.is_set()
        assert conn.call("anything") == {"ok": True}
    finally:
        conn.close()
        box["srv"].stop()


def test_autostart_gives_up(monkeypatch):
    monkeypatch.delenv(hub_link.NO_HUB_ENV, raising=False)
    with pytest.raises(HilError) as e:
        hub_link.connect_or_start("test", spawn=lambda: None, wait_s=0.2)
    assert e.value.details["reason"] == "no_hub"


def test_autostart_runs_the_hub_from_the_private_dir(tmp_path, monkeypatch):
    monkeypatch.setenv("XDG_RUNTIME_DIR", str(tmp_path))
    seen: dict = {}
    monkeypatch.setattr(hub_link.subprocess, "Popen", lambda argv, **kw: seen.update(kw))
    hub_link._spawn_hub()
    assert seen["cwd"] == str(hub_ipc.endpoint_file().parent)


@pytest.mark.uses_hub
def test_a_hub_link_reconnects_once_its_connection_is_gone(hub):
    link = hub_link.open_link(_dev(), purpose="pads")
    try:
        link._conn.close()
        assert link.transact("KIT_STATUS").line == KITSTATUS
    finally:
        link.close()


@pytest.mark.uses_hub
def test_the_exclusive_op_lasts_until_the_last_hold_ends_in_any_order(hub):
    other = hub_link.HubLink(hub_link.connect_or_start("pads"), "pads")
    try:
        capture = hub_link.hold_exclusive("capture")
        transfer = hub_link.hold_exclusive("fs transfer")
        capture.release()                     # first in, first out
        with pytest.raises(HilError) as e:
            other.transact("KIT_STATUS")
        assert e.value.code == PORT_BUSY
        transfer.release()
        assert other.transact("KIT_STATUS").line == KITSTATUS
    finally:
        other.close()


@pytest.mark.uses_hub
def test_two_threads_taking_holds_at_once_never_refuse_each_other(hub):
    import threading

    errors: list = []
    holds: list = []

    def take():
        try:
            holds.append(hub_link.hold_exclusive("parallel"))
        except HilError as e:
            errors.append(e)

    threads = [threading.Thread(target=take) for _ in range(4)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    for h in holds:
        h.release()
    assert errors == []
