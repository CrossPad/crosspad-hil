"""AlsaBleMidiLink against a fake rtmidi and a fake bluetoothctl."""
from __future__ import annotations

import pytest

from crosspad_hil import ble
from crosspad_hil.errors import HilError


class FakePort:
    def __init__(self, ports: list[str]) -> None:
        self.ports = ports
        self.opened: int | None = None
        self.sent: list[list[int]] = []
        self.cb = None
        self.closed = False

    def get_ports(self) -> list[str]:
        return list(self.ports)

    def ignore_types(self, **_kw) -> None: ...

    def set_callback(self, cb) -> None:
        self.cb = cb

    def open_port(self, idx: int) -> None:
        self.opened = idx

    def close_port(self) -> None:
        self.closed = True

    def send_message(self, m: list[int]) -> None:
        self.sent.append(list(m))


class FakeRtmidi:
    def __init__(self, ports: list[str]) -> None:
        self.ports = ports
        self.ins: list[FakePort] = []
        self.outs: list[FakePort] = []

    def MidiIn(self) -> FakePort:  # noqa: N802 - rtmidi's spelling
        p = FakePort(self.ports)
        self.ins.append(p)
        return p

    def MidiOut(self) -> FakePort:  # noqa: N802
        p = FakePort(self.ports)
        self.outs.append(p)
        return p


class FakeCtl:
    def __init__(self, answer: str = "Connection successful", info: str = "") -> None:
        self.calls: list[list[str]] = []
        self.answer = answer
        self.info = info

    def __call__(self, args, timeout_s: float) -> str:
        self.calls.append(list(args))
        if args[0] == "info":
            return self.info
        return self.answer if args[0] == "connect" else "Successful disconnected"


def make(ports: list[str], answer: str = "Connection successful", *, forget: bool = False,
         info: str = ""):
    rt = FakeRtmidi(ports)
    ctl = FakeCtl(answer, info)
    link = ble.AlsaBleMidiLink("44:1B:F6:8B:22:3A", name="CrossPad", timeout_s=0.3,
                               run_cmd=ctl, rtmidi_module=rt, clock=lambda: 1.0,
                               forget=forget)
    return link, rt, ctl


ADDR = "44:1B:F6:8B:22:3A"
UNPAIRED = ("Device 44:1B:F6:8B:22:3A (public)\n\tName: CrossPad\n\tPaired: no\n"
            "\tBonded: no\n\tTrusted: no\n\tConnected: no\n")


def test_connect_finds_the_bluez_port_and_opens_both_directions() -> None:
    link, rt, ctl = make(["Midi Through:Midi Through Port-0 14:0",
                          "CrossPad MIDI+Serial:CrossPad MIDI+Serial MIDI 1 20:0",
                          "CrossPad:CrossPad Bluetooth 145:0"])
    st = link.connect()
    assert ctl.calls[0] == ["connect", "44:1B:F6:8B:22:3A"]
    assert st["transport"] == "alsa" and st["port"].endswith("145:0")
    assert rt.ins[0].opened == 2 and rt.outs[0].opened == 2
    assert st["notify"] and st["write_without_response"]


def test_notes_arrive_and_are_described_like_gatt_ones() -> None:
    link, rt, _ = make(["CrossPad:CrossPad Bluetooth 145:0"])
    link.connect()
    rt.ins[0].cb(([0x90, 41, 100], 0.0), None)
    rt.ins[0].cb(([0x80, 41, 0], 0.0), None)
    assert link.wait_for(2, 0.1)
    got = link.received()
    assert [m["kind"] for m in got] == ["note_on", "note_off"]
    assert got[0]["note"] == 41 and got[0]["t"] == 1.0


def test_send_writes_each_message_to_the_port() -> None:
    link, rt, _ = make(["CrossPad:CrossPad Bluetooth 145:0"])
    link.connect()
    link.send([ble.note_message(True, 60, 90), ble.note_message(False, 60)])
    assert rt.outs[0].sent == [[0x90, 60, 90], [0x80, 60, 0]]
    assert link.status()["tx_messages"] == 2


def test_no_port_means_no_device_and_a_disconnect() -> None:
    link, _, ctl = make(["Midi Through:Midi Through Port-0 14:0"])
    with pytest.raises(HilError) as e:
        link.connect()
    assert e.value.code == "NO_DEVICE"
    assert ctl.calls[-1][0] == "disconnect"


def test_bluetoothctl_refusal_is_no_device() -> None:
    link, _, _ = make(["CrossPad:CrossPad Bluetooth 145:0"], answer="Device not available")
    with pytest.raises(HilError) as e:
        link.connect()
    assert e.value.code == "NO_DEVICE"


def test_disconnect_closes_ports_and_tells_bluetoothd() -> None:
    link, rt, ctl = make(["CrossPad:CrossPad Bluetooth 145:0"])
    link.connect()
    link.disconnect()
    assert rt.ins[0].closed and rt.outs[0].closed
    assert ctl.calls[-1] == ["disconnect", "44:1B:F6:8B:22:3A"]
    assert not link.connected


def test_a_port_that_will_not_open_still_disconnects() -> None:
    """bluetoothctl connected, rtmidi refused: the host must hang up before raising."""
    link, rt, ctl = make(["CrossPad:CrossPad Bluetooth 145:0"])

    def refuse(_idx: int) -> None:
        raise RuntimeError("ALSA error making port connection")

    real_midi_in = rt.MidiIn

    def midi_in() -> FakePort:
        p = real_midi_in()
        p.open_port = refuse    # type: ignore[method-assign]
        return p

    rt.MidiIn = midi_in         # type: ignore[method-assign]
    with pytest.raises(RuntimeError):
        link.connect()
    assert ctl.calls[-1] == ["disconnect", ADDR]
    assert rt.ins[0].closed and not link.connected and link.port is None


def test_forget_removes_an_entry_the_run_created() -> None:
    link, _, ctl = make(["CrossPad:CrossPad Bluetooth 145:0"], forget=True, info=UNPAIRED)
    link.connect()
    link.disconnect()
    assert ctl.calls[0] == ["info", ADDR]
    assert ctl.calls[-2:] == [["disconnect", ADDR], ["remove", ADDR]]


def test_forget_removes_after_a_failed_open_too(monkeypatch) -> None:
    monkeypatch.setattr(ble, "ALSA_PORT_WAIT_S", 0.0)
    link, _, ctl = make(["Midi Through:Midi Through Port-0 14:0"], forget=True,
                        info=f"Device {ADDR} not available")
    with pytest.raises(HilError):
        link.connect()
    assert ctl.calls[-2:] == [["disconnect", ADDR], ["remove", ADDR]]


@pytest.mark.parametrize("info", [
    UNPAIRED.replace("Paired: no", "Paired: yes"),
    UNPAIRED.replace("Bonded: no", "Bonded: yes"),
    UNPAIRED.replace("Trusted: no", "Trusted: yes"),
    "",                                             # unreadable: assume it was known
])
def test_forget_keeps_a_board_this_host_already_knew(info: str) -> None:
    link, _, ctl = make(["CrossPad:CrossPad Bluetooth 145:0"], forget=True, info=info)
    link.connect()
    link.disconnect()
    assert ["remove", ADDR] not in ctl.calls


def test_without_forget_nothing_is_asked_or_removed() -> None:
    link, _, ctl = make(["CrossPad:CrossPad Bluetooth 145:0"], info=UNPAIRED)
    link.connect()
    link.disconnect()
    assert [c[0] for c in ctl.calls] == ["connect", "disconnect"]


def test_open_midi_link_passes_forget_to_the_bluez_link(monkeypatch) -> None:
    made: dict = {}

    class Fake:
        def __init__(self, address, **kw) -> None:
            made.update(kw, address=address)

        def connect(self) -> None: ...

    monkeypatch.setattr(ble, "AlsaBleMidiLink", Fake)
    monkeypatch.setattr(ble.sys, "platform", "linux")
    ble.open_midi_link(ADDR, name="CrossPad", which=lambda _: "/usr/bin/bluetoothctl",
                       forget=True)
    assert made["forget"] is True and made["address"] == ADDR
