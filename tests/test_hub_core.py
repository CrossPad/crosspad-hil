"""HubCore: shared CDC, exclusive ops, idle close, re-enumeration, audio-profile routing."""
from __future__ import annotations

import dataclasses
import time

import pytest

from crosspad_hil import cdc as cdc_mod
from crosspad_hil.cdc import CdcLink
from crosspad_hil.devices import Device, Ports, SerialPortInfo, UsbMode
from crosspad_hil.errors import NO_CDC_IN_AUDIO_MODE, NO_DEVICE, PORT_BUSY, UNKNOWN_VERB, HilError
from crosspad_hil.hub_core import BOARD_CACHE_S, CDC_IDLE_S, HubCore
from tests.fakes import FakeSerial
from tests.test_cdc import CDC_KNOWLEDGE, KITSTATUS


class S:
    def __init__(self, sid, client="c", purpose="p", pid=10):
        self.id, self.client, self.purpose, self.pid = sid, client, purpose, pid


class Clock:
    def __init__(self):
        self.t = 100.0

    def __call__(self):
        return self.t


class KillableSerial(FakeSerial):
    """A port that can vanish under the reader, as a reset board's CDC node does."""

    def __init__(self, script):
        super().__init__(script)
        self.dead = False

    def read(self, size: int = 1) -> bytes:
        if self.dead:
            raise OSError("device disconnected")
        return super().read(size)


def _dev(path="/dev/fake-acm0", mode=UsbMode.DEFAULT):
    cdc = None if path is None else SerialPortInfo(path=path, vid=0x303A, pid=0x3456,
                                                   serial="AB", product="CrossPad",
                                                   location="1-1")
    return Device(id="AB", serial="AB", usb_mode=mode,
                  ports=dataclasses.replace(Ports(), cdc=cdc))


@pytest.fixture
def env(monkeypatch):
    monkeypatch.setattr(cdc_mod, "OPEN_SETTLE_S", 0)
    state = {"dev": _dev(), "opened": [], "serials": [], "companion": [], "found": 0}

    def find_board():
        state["found"] += 1
        return state["dev"]

    def link_factory(port):
        ser = KillableSerial([("KIT_STATUS", KITSTATUS)])
        state["serials"].append(ser)
        link = CdcLink(port, serial_factory=lambda *a, **k: ser, knowledge=CDC_KNOWLEDGE,
                       purpose="hub")
        link.open()
        state["opened"].append(link)
        return link

    def companion(verb, timeout_s):
        state["companion"].append(verb)
        return KITSTATUS if verb == "KIT_STATUS" else verb

    clock = Clock()
    core = HubCore(find_board=find_board, link_factory=link_factory,
                   companion=companion, clock=clock)
    core._knowledge_verbs = CDC_KNOWLEDGE["verbs"]
    yield core, state, clock
    core.close()


def _t(core, s, cmd):
    return core.handle(s, {"op": "transact", "args": {"cmd": cmd}})


def test_lazy_open_and_shared_use(env):
    core, state, _ = env
    a, b = S(1, "cp-tools"), S(2, "mcp")
    core.connect(a)
    core.connect(b)
    assert state["opened"] == []
    r = _t(core, a, "KIT_STATUS")
    assert r["line"] == KITSTATUS and r["more_lines"] == []
    assert _t(core, b, "KIT_STATUS")["parsed"]["name"] == "DRUMS"
    assert len(state["opened"]) == 1


def test_status_lists_clients(env):
    core, _, _ = env
    core.connect(S(1, "cp-tools", "pads", 42))
    st = core.handle(S(1), {"op": "status", "args": {}})
    assert st["clients"] == [{"client": "cp-tools", "purpose": "pads", "pid": 42}]
    assert st["lease"] is None and st["usb_mode"] == "default"


def test_exclusive_blocks_others_and_closes_link(env):
    core, state, _ = env
    a, b = S(1, "bench", "flash", 7), S(2, "cp-tools")
    core.connect(a)
    core.connect(b)
    _t(core, b, "KIT_STATUS")
    core.handle(a, {"op": "lease", "args": {"purpose": "ota flash"}})
    assert state["opened"][0]._ser is None
    with pytest.raises(HilError) as e:
        _t(core, b, "KIT_STATUS")
    assert e.value.code == PORT_BUSY
    assert e.value.details["holder"]["client"] == "bench"
    assert e.value.details["holder"]["purpose"] == "ota flash"
    with pytest.raises(HilError):
        core.handle(b, {"op": "lease", "args": {"purpose": "x"}})


def test_exclusive_released_on_disconnect(env):
    core, _, _ = env
    a, b = S(1, "bench"), S(2, "cp-tools")
    core.connect(a)
    core.connect(b)
    core.handle(a, {"op": "lease", "args": {"purpose": "ota flash"}})
    core.disconnect(a)
    assert _t(core, b, "KIT_STATUS")["line"] == KITSTATUS


def test_idle_close_after_last_client(env):
    core, state, clock = env
    a = S(1)
    core.connect(a)
    _t(core, a, "KIT_STATUS")
    core.disconnect(a)
    clock.t += CDC_IDLE_S - 0.1
    core.tick()
    assert state["opened"][0]._ser is not None
    clock.t += 0.2
    core.tick()
    assert state["opened"][0]._ser is None
    assert core.handle(S(9), {"op": "status", "args": {}})["cdc_open"] is False


def test_board_discovery_is_cached_between_verbs(env):
    core, state, clock = env
    a = S(1)
    core.connect(a)
    _t(core, a, "KIT_STATUS")
    _t(core, a, "KIT_STATUS")
    assert state["found"] == 1
    clock.t += BOARD_CACHE_S
    _t(core, a, "KIT_STATUS")
    assert state["found"] == 2


def test_reopens_when_cdc_path_changes(env):
    core, state, clock = env
    a = S(1)
    core.connect(a)
    _t(core, a, "KIT_STATUS")
    state["dev"] = _dev("/dev/fake-acm1")
    clock.t += BOARD_CACHE_S
    _t(core, a, "KIT_STATUS")
    assert [link.port for link in state["opened"]] == ["/dev/fake-acm0", "/dev/fake-acm1"]
    assert state["opened"][0]._ser is None


def test_reopens_when_the_port_died_on_the_same_path(env):
    core, state, _ = env
    a = S(1)
    core.connect(a)
    _t(core, a, "KIT_STATUS")
    state["serials"][0].dead = True
    deadline = time.monotonic() + 2.0
    while state["opened"][0].reader_alive() and time.monotonic() < deadline:
        time.sleep(0.01)
    assert _t(core, a, "KIT_STATUS")["line"] == KITSTATUS
    assert [link.port for link in state["opened"]] == ["/dev/fake-acm0", "/dev/fake-acm0"]


def test_audio_profile_goes_through_companion(env):
    core, state, _ = env
    state["dev"] = _dev(None, UsbMode.AUDIO)
    a = S(1)
    core.connect(a)
    r = _t(core, a, "KIT_STATUS")
    assert r["line"] == KITSTATUS and r["parsed"]["name"] == "DRUMS"
    assert state["companion"] == ["KIT_STATUS"]


def test_audio_profile_echo_is_unknown_verb(env):
    core, state, _ = env
    state["dev"] = _dev(None, UsbMode.AUDIO)
    core.connect(S(1))
    with pytest.raises(HilError) as e:
        core.handle(S(1), {"op": "transact", "args": {"cmd": "KIT_LOAD 3"}})
    assert e.value.code == UNKNOWN_VERB


def test_audio_profile_burst_needs_cdc(env):
    core, state, _ = env
    state["dev"] = _dev(None, UsbMode.AUDIO)
    core.connect(S(1))
    with pytest.raises(HilError) as e:
        core.handle(S(1), {"op": "burst", "args": {"cmds": ["PAD_PRESS 0"], "rate_hz": 10}})
    assert e.value.code == NO_CDC_IN_AUDIO_MODE


def test_no_board(env):
    core, state, _ = env
    state["dev"] = None
    core.connect(S(1))
    with pytest.raises(HilError) as e:
        _t(core, S(1), "KIT_STATUS")
    assert e.value.code == NO_DEVICE


def test_a_verb_queued_behind_the_port_cannot_reopen_it_after_an_exclusive_op(env):
    """The check that lets a verb in must hold when it reaches the port, not before."""
    import threading

    core, state, _ = env
    poller, flasher = S(1, "cp-tools"), S(2, "bench", "flash", 7)
    core.connect(poller)
    core.connect(flasher)
    _t(core, poller, "KIT_STATUS")
    seen: dict = {}

    def poll():
        try:
            _t(core, poller, "KIT_STATUS")
            seen["poll"] = "ran"
        except HilError as e:
            seen["poll"] = e.code

    core._q.acquire()                         # a verb on the port right now
    p = threading.Thread(target=poll)
    p.start()
    time.sleep(0.1)                           # the poll is past its check, waiting for the port
    g = threading.Thread(target=lambda: core.handle(flasher, {"op": "lease",
                                                              "args": {"purpose": "ota"}}))
    g.start()
    deadline = time.monotonic() + 2
    while core._exclusive is None and time.monotonic() < deadline:
        time.sleep(0.01)
    core._q.release()
    p.join(2)
    g.join(2)
    assert seen["poll"] == PORT_BUSY
    assert core._link is None
