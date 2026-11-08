"""BLE MIDI from the host side: packet codec, peripheral selection, link, latency.

No radio and no bleak: the scanner and the GATT client are injected. What is tested is the
packet encoding a real host produces, the rule for deciding *which* peripheral is the board,
and the accounting the latency measurement does.
"""
from __future__ import annotations

import io
import threading

import pytest

from crosspad_hil import ble
from crosspad_hil.devices import Backends, SerialPortInfo
from crosspad_hil.errors import (
    AMBIGUOUS_DEVICE,
    BAD_ARGS,
    ENV,
    NO_DEVICE,
    NOT_SUPPORTED,
    TIMEOUT,
    HilError,
)
from crosspad_hil.serve import Daemon

CDC = "/dev/ttyACM0"
CON = "/dev/ttyACM1"
ADDR = "AA:BB:CC:DD:EE:FF"


# --------------------------------------------------------------------------- codec

def test_encode_is_a_header_then_timestamped_messages():
    packet = ble.encode([[0x90, 60, 100]], ts_ms=0x1234)
    assert packet[0] == 0x80 | ((0x1234 >> 7) & 0x3F)
    assert packet[1] == 0x80 | (0x1234 & 0x7F)
    assert packet[2:] == bytes([0x90, 60, 100])


def test_round_trip_of_several_messages_in_one_packet():
    """What a host sends when the events fall inside one connection interval."""
    msgs = [[0x90, 60, 90], [0x80, 60, 0], [0x90, 62, 90], [0x80, 62, 0]]
    assert ble.decode(ble.encode(msgs)) == [bytes(m) for m in msgs]


def test_decode_handles_running_status():
    # header, ts+status note-on, then a second note with no status byte of its own.
    packet = bytes([0x80, 0x81, 0x90, 60, 100, 62, 100])
    assert ble.decode(packet) == [bytes([0x90, 60, 100]), bytes([0x90, 62, 100])]


def test_decode_of_an_empty_or_header_only_packet_is_empty():
    assert ble.decode(b"") == []
    assert ble.decode(bytes([0x80])) == []


def test_encode_refuses_a_message_without_a_status_byte():
    with pytest.raises(HilError) as e:
        ble.encode([[60, 100]])
    assert e.value.code == BAD_ARGS
    with pytest.raises(HilError):
        ble.encode([])


@pytest.mark.parametrize("note,vel,channel", [(128, 100, 0), (60, 200, 0), (60, 100, 16),
                                              (-1, 100, 0)])
def test_note_message_range_checks(note, vel, channel):
    with pytest.raises(HilError) as e:
        ble.note_message(True, note, vel, channel)
    assert e.value.code == BAD_ARGS


def test_note_off_is_a_note_on_with_velocity_zero_too():
    assert ble.describe(bytes([0x90, 60, 0]))["kind"] == "note_off"
    assert ble.describe(bytes([0x80, 60, 0]))["kind"] == "note_off"
    on = ble.describe(bytes([0x90, 60, 100]))
    assert (on["kind"], on["note"], on["velocity"], on["channel"]) == ("note_on", 60, 100, 0)


def test_messages_from_accepts_hex_and_byte_lists():
    assert ble.messages_from(["90 3C 64", [0x80, 60, 0]]) == [[0x90, 60, 100], [0x80, 60, 0]]
    with pytest.raises(HilError):
        ble.messages_from(["3C 64"])            # no status byte
    with pytest.raises(HilError):
        ble.messages_from(["zz"])
    with pytest.raises(HilError):
        ble.messages_from([])


# --------------------------------------------------------------------------- scanning

def peripherals(*items) -> list[ble.BlePeripheral]:
    return list(items)


def midi(address: str, name: str | None, rssi: int = -50) -> ble.BlePeripheral:
    return ble.BlePeripheral(address=address, name=name, rssi=rssi,
                             service_uuids=[ble.MIDI_SERVICE_UUID])


def test_scan_keeps_only_midi_peripherals_and_sorts_by_signal():
    async def discover(_timeout):
        return [ble.BlePeripheral("11:11:11:11:11:11", "Speaker", -30, ["0000180f-0000-1000-"
                                                                       "8000-00805f9b34fb"]),
                midi("22:22:22:22:22:22", "CrossPad", -70),
                midi("33:33:33:33:33:33", "Other MIDI", -40)]

    found = ble.scan(0.01, discover_fn=discover)
    assert [p.address for p in found] == ["33:33:33:33:33:33", "22:22:22:22:22:22"]


def test_scan_can_return_everything_it_saw():
    async def discover(_timeout):
        return [ble.BlePeripheral("11:11:11:11:11:11", "Speaker", -30, [])]

    assert len(ble.scan(0.01, midi_only=False, discover_fn=discover)) == 1


def test_scan_timeout_must_be_positive():
    with pytest.raises(HilError) as e:
        ble.scan(0)
    assert e.value.code == BAD_ARGS


def test_a_scan_that_never_returns_is_a_timeout():
    async def hang(_timeout):
        import asyncio

        await asyncio.sleep(60)

    with pytest.raises(HilError) as e:
        ble.scan(0.001, discover_fn=hang)
    assert e.value.code == TIMEOUT


def test_the_board_is_chosen_by_the_address_it_reports():
    found = peripherals(midi("11:11:11:11:11:11", "CrossPad"), midi(ADDR, "CrossPad"))
    assert ble.select_peripheral(found, address=ADDR.lower()).address == ADDR


def test_an_address_that_is_not_advertising_is_not_silently_swapped():
    """The failure that motivated this: a run once attached to a passing speaker."""
    found = peripherals(midi("11:11:11:11:11:11", "OB-4"))
    with pytest.raises(HilError) as e:
        ble.select_peripheral(found, address=ADDR)
    assert e.value.code == NO_DEVICE
    assert "OB-4" in e.value.message
    assert e.value.details["seen"]


def test_the_name_is_a_fallback_only_and_refuses_a_tie():
    one = peripherals(midi(ADDR, "CrossPad"), midi("11:11:11:11:11:11", "OB-4"))
    assert ble.select_peripheral(one).address == ADDR
    two = peripherals(midi(ADDR, "CrossPad"), midi("11:11:11:11:11:11", "CrossPad"))
    with pytest.raises(HilError) as e:
        ble.select_peripheral(two)
    assert e.value.code == AMBIGUOUS_DEVICE
    assert len(e.value.details["candidates"]) == 2


def test_nothing_named_like_the_board_is_no_device():
    with pytest.raises(HilError) as e:
        ble.select_peripheral(peripherals(midi("11:11:11:11:11:11", "OB-4")))
    assert e.value.code == NO_DEVICE
    assert "BLE_START 1" in (e.value.hint or "")


# --------------------------------------------------------------------------- the link

class FakeChar:
    properties = ["read", "write-without-response", "notify"]


class FakeServices:
    def get_characteristic(self, uuid: str):
        return FakeChar() if uuid == ble.MIDI_CHAR_UUID else None


class FakeClient:
    """The awaitable surface bleak exposes, plus push() to fake a notification."""

    def __init__(self, address: str, timeout: float = 20.0) -> None:
        self.address = address
        self.timeout = timeout
        self.connected = False
        self.notifying = False
        self.writes: list[bytes] = []
        self.services = FakeServices()
        self._cb = None

    async def connect(self) -> None:
        self.connected = True

    async def disconnect(self) -> None:
        self.connected = False

    async def start_notify(self, _uuid: str, cb) -> None:
        self.notifying = True
        self._cb = cb

    async def stop_notify(self, _uuid: str) -> None:
        self.notifying = False

    async def write_gatt_char(self, _uuid: str, data: bytes, response: bool = True) -> None:
        assert response is False, "BLE MIDI writes are write-without-response"
        self.writes.append(bytes(data))

    def push(self, messages) -> None:
        assert self._cb is not None
        self._cb(None, bytearray(ble.encode(messages)))


@pytest.fixture
def link():
    client = FakeClient(ADDR)
    lk = ble.BleMidiLink(ADDR, name="CrossPad", client_factory=lambda a, t: client)
    lk.connect()
    lk.client = client  # type: ignore[attr-defined]
    yield lk
    lk.disconnect()


def test_connect_subscribes_and_reports_the_properties(link):
    st = link.status()
    assert st["connected"] is True
    assert st["notify"] and st["write_without_response"]
    assert link.client.notifying is True


def test_connecting_twice_is_a_no_op(link):
    assert link.connect()["connected"] is True


def test_send_writes_one_packet_per_call(link):
    link.send_note(True, 60, 100)
    link.send_note(False, 60)
    assert len(link.client.writes) == 2
    assert ble.decode(link.client.writes[0]) == [bytes([0x90, 60, 100])]
    assert ble.decode(link.client.writes[1]) == [bytes([0x80, 60, 0])]
    assert link.status()["tx_messages"] == 2


def test_send_packs_several_messages_into_one_write(link):
    link.send([[0x90, 60, 90], [0x80, 60, 0]])
    assert len(link.client.writes) == 1
    assert len(ble.decode(link.client.writes[0])) == 2


def test_a_subscribe_that_fails_hangs_up_before_raising():
    """Connected but start_notify refused: the host must not keep the board's link."""
    class NoNotify(FakeClient):
        async def start_notify(self, _uuid: str, cb) -> None:
            raise RuntimeError("characteristic not found")

    client = NoNotify(ADDR)
    lk = ble.BleMidiLink(ADDR, client_factory=lambda a, t: client)
    with pytest.raises(RuntimeError):
        lk.connect()
    assert client.connected is False
    assert lk.connected is False


def test_send_before_connect_is_refused():
    lk = ble.BleMidiLink(ADDR, client_factory=lambda a, t: FakeClient(a))
    with pytest.raises(HilError) as e:
        lk.send_note(True, 60)
    assert e.value.code == NO_DEVICE


def test_notifications_are_decoded_with_the_moment_they_arrived(link):
    link.client.push([[0x90, 44, 100], [0x80, 44, 0]])
    got = link.received()
    assert [m["kind"] for m in got] == ["note_on", "note_off"]
    assert got[0]["note"] == 44
    assert got[0]["t"] > 0


def test_wait_for_returns_false_when_nothing_arrives(link):
    assert link.wait_for(1, timeout_s=0.05) is False
    link.client.push([[0x90, 60, 100]])
    assert link.wait_for(1, timeout_s=0.5) is True


def test_listen_stops_early_once_it_has_what_it_expected(link):
    def later() -> None:
        link.client.push([[0x90, 60, 100]])

    threading.Timer(0.02, later).start()
    r = link.listen(5.0, expect=1)
    assert r["count"] == 1 and r["notes_on"] == 1


def test_clear_rx_forgets_what_came_before(link):
    link.client.push([[0x90, 60, 100]])
    link.clear_rx()
    assert link.received() == []


def test_disconnect_is_idempotent_and_stops_the_loop(link):
    link.disconnect()
    assert link.disconnect() == {"address": ADDR, "connected": False}
    assert link.client.connected is False


def test_connect_scans_when_no_address_is_given():
    client = FakeClient(ADDR)

    async def discover(_timeout):
        return [midi(ADDR, "CrossPad")]

    lk = ble.connect(None, link=None, scan_timeout_s=0.01, discover_fn=discover,
                     client_factory=lambda a, t: client)
    try:
        assert lk.address == ADDR and lk.connected
    finally:
        lk.disconnect()


# --------------------------------------------------------------------------- latency

class FakeCdc:
    """A CDC link whose BLE_SEND makes the fake radio deliver a note."""

    def __init__(self, client: FakeClient, *, drop: set[int] = frozenset()) -> None:
        self.client = client
        self.sent: list[str] = []
        self.drop = set(drop)

    def send(self, cmd: str) -> None:
        self.sent.append(cmd)
        note = int(cmd.split()[1])
        if note not in self.drop:
            self.client.push([[0x90, note, 100]])


def test_latency_times_every_note_that_came_back(link):
    cdc = FakeCdc(link.client)
    r = ble.latency(link, cdc, n=3, timeout_s=0.5, gap_s=0.0)
    assert r["sent"] == 3 and r["received"] == 3 and r["lost"] == 0
    assert [c.split()[0] for c in cdc.sent] == ["BLE_SEND"] * 3
    assert r["rtt_ms"]["p50"] >= 0 and r["rtt_ms"]["max"] < 500


def test_latency_counts_a_note_that_never_came_back(link):
    cdc = FakeCdc(link.client, drop={61})
    r = ble.latency(link, cdc, n=3, base_note=60, timeout_s=0.05, gap_s=0.0)
    assert r["received"] == 2 and r["lost"] == 1 and r["lost_notes"] == [61]


def test_latency_needs_a_positive_count(link):
    with pytest.raises(HilError) as e:
        ble.latency(link, FakeCdc(link.client), n=0)
    assert e.value.code == BAD_ARGS


# --------------------------------------------------------------------------- environment

def test_require_bleak_names_the_extra(monkeypatch):
    import builtins

    real = builtins.__import__

    def fake(name, *a, **kw):
        if name == "bleak":
            raise ImportError("no bleak here")
        return real(name, *a, **kw)

    monkeypatch.setattr(builtins, "__import__", fake)
    with pytest.raises(HilError) as e:
        ble.require_bleak()
    assert e.value.code == ENV
    assert "crosspad-hil[ble]" in (e.value.hint or "")


@pytest.mark.skipif(__import__("sys").platform == "win32", reason="Windows can advertise")
def test_host_peripheral_is_refused_where_it_cannot_work():
    assert ble.HostPeripheral.available() is False
    with pytest.raises(HilError) as e:
        ble.HostPeripheral().start()
    assert e.value.code == NOT_SUPPORTED
    assert "Windows" in e.value.message


# --------------------------------------------------------------------------- daemon ops

def _backends() -> Backends:
    ports = [
        SerialPortInfo(path=CDC, vid=0x303A, pid=0x3456, serial="AB12", product="CrossPad",
                       location="1-1.2:1.0"),
        SerialPortInfo(path=CON, vid=0x0483, pid=0x5740, serial="STM1",
                       product="CrossPad MIDI+Serial", location="1-1.3:1.0"),
    ]
    return Backends(list_serial=lambda: list(ports), list_midi=lambda: [], list_audio=lambda: [])


def _daemon(connect_fn=None, scan_fn=None) -> Daemon:
    return Daemon(io.StringIO(), io.StringIO(), backends=_backends(),
                  ble_scan_fn=scan_fn or (lambda t: [midi(ADDR, "CrossPad")]),
                  ble_connect_fn=connect_fn or (lambda *a, **kw: None))


def test_daemon_ble_scan_lists_what_the_radio_saw():
    r = _daemon().handle({"id": 1, "op": "ble.scan", "args": {"timeout_s": 0.01}})
    assert r["ok"]
    assert r["result"]["count"] == 1
    assert r["result"]["peripherals"][0]["address"] == ADDR


def test_daemon_ble_connect_send_listen_disconnect(tmp_path, monkeypatch):
    monkeypatch.setenv("XDG_RUNTIME_DIR", str(tmp_path))
    client = FakeClient(ADDR)
    lk = ble.BleMidiLink(ADDR, name="CrossPad", client_factory=lambda a, t: client)
    lk.connect()
    daemon = _daemon(connect_fn=lambda *a, **kw: lk)
    r = daemon.handle({"id": 1, "op": "ble.connect", "args": {"address": ADDR}})
    assert r["ok"], r
    handle = r["result"]["handle"]
    assert handle.startswith("ble_") and r["result"]["connected"] is True

    sent = daemon.handle({"id": 2, "op": "ble.send",
                          "args": {"handle": handle, "note": 60, "vel": 90}})
    assert sent["ok"] and ble.decode(client.writes[0]) == [bytes([0x90, 60, 90])]

    hex_send = daemon.handle({"id": 3, "op": "ble.send",
                              "args": {"handle": handle, "messages": ["80 3C 00"]}})
    assert hex_send["ok"] and ble.decode(client.writes[1]) == [bytes([0x80, 60, 0])]

    threading.Timer(0.02, lambda: client.push([[0x90, 44, 100]])).start()
    heard = daemon.handle({"id": 4, "op": "ble.listen",
                           "args": {"handle": handle, "seconds": 2.0, "expect": 1}})
    assert heard["ok"] and heard["result"]["count"] == 1

    st = daemon.handle({"id": 5, "op": "ble.status", "args": {"handle": handle}})
    assert st["ok"] and st["result"]["tx_messages"] == 2

    bye = daemon.handle({"id": 6, "op": "ble.disconnect", "args": {"handle": handle}})
    assert bye["ok"] and client.connected is False
    assert not daemon.handle({"id": 7, "op": "ble.status", "args": {"handle": handle}})["ok"]


def test_daemon_ble_status_rejects_a_handle_of_the_wrong_kind(tmp_path, monkeypatch):
    monkeypatch.setenv("XDG_RUNTIME_DIR", str(tmp_path))
    daemon = _daemon()
    r = daemon.handle({"id": 1, "op": "ble.status", "args": {"handle": "con_9"}})
    assert not r["ok"]
