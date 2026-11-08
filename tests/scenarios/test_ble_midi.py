"""Scenario `ble_midi` against fakes — no radio, no board.

The CDC fake is the device half: it counts the BLE messages it is told about,
reports a role, an address and a transpose, and turns a PAD_PRESS into a
notification on the fake radio. The radio half (scan, the GATT link, the latency
probe, the Windows host peripheral) is replaced at the scenario's seams, so what
is tested is the scenario: the order of the checks, what each one asserts, and
that the board is left in the role it was found in whatever happened.
"""
from __future__ import annotations

import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest

from crosspad_hil import cdc as cdc_mod
from crosspad_hil.ble import BlePeripheral, describe
from crosspad_hil.cdc import CdcLink
from crosspad_hil.devices import Device, MidiPortInfo, Ports, SerialPortInfo, UsbMode
from crosspad_hil.errors import ENV, NO_DEVICE, NOT_SUPPORTED, HilError
from crosspad_hil.scenarios import base, ble_midi, get
from tests.fakes import FakeSerial

ADDR = "AA:BB:CC:DD:EE:FF"
CDC_PORT = SerialPortInfo(path="/dev/ttyACM0", vid=0x303A, pid=0x3456,
                          serial="CP-1", product="CrossPad", location="1-2.1")
CONSOLE_PORT = SerialPortInfo(path="/dev/ttyACM1", vid=0x0483, pid=0x5740,
                              serial="STM-1", product="CrossPad", location="1-2.2")
ESP_MIDI = MidiPortInfo(name="Crosspad MIDI 1", rtmidi_out=2, rtmidi_in=2,
                        alsa_hw="hw:4,0,0", rawmidi="/dev/snd/midiC4D0")


class BleDevice(FakeSerial):
    """The device half of the link: BLE_* verbs, the pad counters, the note map."""

    def __init__(self) -> None:
        super().__init__(timeout=0.01)
        self.supported = 1
        self.start_refused = False
        self.state = "connected"
        self.mode = "server"
        self.self_addr = ADDR
        self.notes = {p: 36 + p for p in range(16)}
        self.txoff = 0
        self.rx_msg = 0
        self.tx_err = 0
        self.played = 0
        self.press = 0
        self.engine_plays = 1                   # 0 = notes arrive, nothing sounds
        self.role_calls: list[int | None] = []
        self.int_free = [90_000, 50_000]        # BLE off, BLE running
        self.mem_reads = 0
        self.peer_devices: list[str] = []
        self.connected_to: str | None = None
        self.radio: FakeLink | None = None      # what a pad hit notifies
        self.host_peer: Any = None              # …when the board is the central
        self.running = 1                        # BLE_STATUS running=; BLE_STOP/START move it

    # What the host radio writes into the device. A note the pad grid owns does
    # not stop at the parser: the firmware routes it through handlePadPress(),
    # the same entry point a physical pad uses, so it sounds. That is why
    # rx_msg and played are two counters and not one — a note can arrive and
    # still not play.
    def receive(self, messages: list[list[int]]) -> None:
        self.rx_msg += len(messages)
        grid = set(self.notes.values())
        for m in messages:
            if len(m) >= 3 and (m[0] & 0xF0) == 0x90 and m[2] > 0 and m[1] in grid:
                self.press += 1
                self.played += self.engine_plays

    def write(self, data: bytes) -> int:
        for raw in data.decode("utf-8", "replace").split("\n"):
            cmd = raw.strip()
            if not cmd:
                continue
            self.written.append(cmd)
            if cmd == "BLE_STATUS":
                self.feed([f"BLE: supported={self.supported} running={self.running} "
                           f"state={self.state} "
                           f"mode={self.mode} self={self.self_addr} peer={self.connected_to or '-'} "
                           f"itvl=12 rx_msg={self.rx_msg} tx_pkt=3 tx_err={self.tx_err} "
                           f"tx_drop=0 txoff={self.txoff}"])
            elif cmd.startswith("BLE_START"):
                parts = cmd.split()
                mode = int(parts[1]) if len(parts) > 1 else None
                self.role_calls.append(mode)
                if self.start_refused and mode == 1:
                    self.feed(["ERR"])
                    continue
                if mode is not None:
                    self.mode = "server" if mode else "host"
                self.running = 1
                self.feed(["OK"])
            elif cmd in ("BLE_STOP", "BLE_DISCONNECT"):
                if cmd == "BLE_DISCONNECT":
                    self.connected_to = None
                else:
                    self.running = 0
                self.feed(["OK"])
            elif cmd.startswith("BLE_SCAN"):
                self.feed(["OK"])
            elif cmd.startswith("BLE_CONNECT "):
                self.connected_to = cmd.split(" ", 1)[1]
                self.feed(["OK"])
            elif cmd.startswith("BLE_SEND "):
                note = int(cmd.split()[1])
                if self.radio is not None:
                    self.radio.notify([0x90, note + self.txoff, 100])
                self.feed(["OK"])
            elif cmd == "BLE_DEVICES":
                body = "".join(f" | {a} Peer -50" for a in self.peer_devices)
                self.feed([f"BLEDEV: count={len(self.peer_devices)}{body}"])
            elif cmd == "MEM":
                free = self.int_free[min(self.mem_reads, len(self.int_free) - 1)]
                self.mem_reads += 1
                self.feed([f"MEM: int_free={free} int_largest={free} int_min={free} "
                           f"int_blocks=12 psram_free=8000000 psram_largest=4000000 "
                           f"psram_blocks=3"])
            elif cmd == "PAD_NOTES":
                body = " ".join(f"{p}:{n}" for p, n in sorted(self.notes.items()))
                self.feed([f"PADNOTES: {body}"])
            elif cmd == "PAD_STATS_RESET":
                self.press = self.played = 0
                self.feed(["OK"])
            elif cmd == "PAD_STATS":
                self.feed([f"PADSTATS: press={self.press} release={self.press} "
                           f"played={self.played} freeslots=7"])
            elif cmd.startswith("PAD_PRESS "):
                pad = int(cmd.split()[1])
                self.press += 1
                self.played += 1
                note = self.notes.get(pad, 36 + pad) + self.txoff
                if self.radio is not None:
                    self.radio.notify([0x90, note, 100])
                if self.host_peer is not None:
                    self.host_peer.written_by_board.append([0x90, note, 100])
                self.feed(["OK"])
            elif cmd.startswith("PAD_RELEASE "):
                pad = int(cmd.split()[1])
                note = self.notes.get(pad, 36 + pad) + self.txoff
                if self.radio is not None:
                    self.radio.notify([0x80, note, 0])
                if self.host_peer is not None:
                    self.host_peer.written_by_board.append([0x80, note, 0])
                self.feed(["OK"])
            elif cmd.startswith("KIT_LOAD "):
                self.feed(["OK"])
            elif cmd == "KIT_STATUS":
                self.feed(["KITSTATUS: current=8 loading=0 pending=-1 name=DRUMS"])
            elif cmd == "APP_LIST":
                self.feed(["APPS: Sampler,Mixer running=Sampler"])
            elif cmd.startswith("APP_START "):
                self.feed(["OK"])
            else:
                self._rx.append((cmd + "\r\n").encode())
        return len(data)


class FakeLink:
    """A BleMidiLink stand-in: what the host wrote, and what it was notified of."""

    def __init__(self, device: BleDevice, *, properties: list[str] | None = None) -> None:
        self._device = device
        self.properties = properties if properties is not None else [
            "write-without-response", "notify"]
        self.sent: list[list[list[int]]] = []
        self.disconnected = False
        self._rx: list[dict] = []

    # -- what the scenario calls
    def status(self) -> dict:
        return {"properties": list(self.properties),
                "write_without_response": "write-without-response" in self.properties,
                "notify": "notify" in self.properties}

    def send(self, messages: list[list[int]]) -> None:
        self.sent.append([list(m) for m in messages])
        self._device.receive(messages)

    def send_note(self, on: bool, note: int, vel: int = 100) -> None:
        self.send([[0x90 if on else 0x80, note, vel if on else 0]])

    def send_hit(self, note: int, vel: int = 100, gate_s: float = 0.05) -> None:
        self.send_note(True, note, vel)
        self.send_note(False, note)

    def clear_rx(self) -> None:
        self._rx.clear()

    def received(self) -> list[dict]:
        return list(self._rx)

    def disconnect(self) -> None:
        self.disconnected = True

    # -- what the device fake calls
    def notify(self, msg: list[int]) -> None:
        self._rx.append({"t": 0.0, **describe(bytes(msg))})


class FakeHostPeripheral:
    """The Windows GATT server stand-in for the board's Host role."""

    def __init__(self, device: BleDevice) -> None:
        self._device = device
        self.started = False
        self.stopped = False
        self.notified: list[list[list[int]]] = []
        self.written_by_board: list[list[int]] = []
        self.clients = 1

    def start(self) -> dict:
        self.started = True
        self._device.host_peer = self
        return {"advertising": True}

    def notify(self, messages: list[list[int]]) -> None:
        self.notified.append([list(m) for m in messages])
        self._device.receive(messages)

    def subscribers(self) -> int:
        return self.clients

    def writes(self, *, clear: bool = False) -> list[dict]:
        out = [describe(bytes(m)) for m in self.written_by_board]
        if clear:
            self.written_by_board.clear()
        return out

    def stop(self) -> dict:
        self.stopped = True
        self._device.host_peer = None
        return {"advertising": False}


@dataclass
class FakeContext(base.Context):
    cdc_fake: BleDevice = None  # type: ignore[assignment]

    def open_cdc(self) -> CdcLink:
        fake = self.cdc_fake
        link = CdcLink(self.device.ports.cdc.path, serial_factory=lambda path, **kw: fake)
        link.open()
        self.opened.append(link)
        return link


def make_device() -> Device:
    return Device(id="dev_ab12", serial="CP-1", usb_mode=UsbMode.DEFAULT,
                  ports=Ports(cdc=CDC_PORT, console=CONSOLE_PORT, esp_midi=ESP_MIDI))


@pytest.fixture(autouse=True)
def fast(monkeypatch: pytest.MonkeyPatch) -> None:
    """Collapse the old script's settle times; every one of them is a sleep."""
    for name in ("BLE_STOP_SETTLE_S", "BLE_START_SETTLE_S", "APP_START_SETTLE_S",
                 "NOTIFY_SETTLE_S", "NOTE_GATE_S", "NOTE_GAP_S", "RX_SETTLE_S",
                 "PAD_GAP_S", "PAD_SETTLE_S", "ADV_SETTLE_S", "ROLE_SETTLE_S",
                 "HOST_SCAN_WAIT_S", "HOST_CONNECT_SETTLE_S", "HOST_NOTIFY_GAP_S",
                 "HOST_NOTIFY_SETTLE_S", "HOST_PAD_SETTLE_S"):
        monkeypatch.setattr(ble_midi, name, 0.0)
    monkeypatch.setattr(cdc_mod, "OPEN_SETTLE_S", 0.0)


@pytest.fixture
def bench(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    class Bench:
        def __init__(self) -> None:
            self.cdc = BleDevice()
            self.ctx = FakeContext(device=make_device(), workdir=tmp_path,
                                   cancelled=threading.Event(), log=lambda s: None,
                                   cdc_fake=self.cdc)
            self.link = FakeLink(self.cdc)
            self.cdc.radio = self.link
            self.peripherals = [BlePeripheral(address=ADDR, name="CrossPad", rssi=-40,
                                              service_uuids=[
                                                  "03b80e5a-ede8-4b33-a751-6ce34ec4c700"])]
            self.scanned: list[float] = []
            self.host = FakeHostPeripheral(self.cdc)
            self.host_available = True
            self.latency_result = {"sent": 8, "received": 8, "lost": 0,
                                   "rtt_ms": {"min": 6.0, "p50": 18.0, "p90": 25.0,
                                              "max": 30.0}}

        def scan(self, timeout_s: float) -> list[BlePeripheral]:
            self.scanned.append(timeout_s)
            return list(self.peripherals)

        def open_link(self, address: str, *, name: str | None, timeout_s: float) -> FakeLink:
            self.link.address = address  # type: ignore[attr-defined]
            return self.link

        def latency(self, ble: Any, link: CdcLink, **kw: Any) -> dict:
            # The real one drives BLE_SEND; keep that visible in the transcript.
            for i in range(int(kw.get("n", 8))):
                link.send(f"BLE_SEND {int(kw.get('base_note', 60)) + i} 100")
            return dict(self.latency_result)

        def run(self, **params: Any) -> base.Report:
            return ble_midi.SCENARIO.run(
                self.ctx, ble_midi.Params(**params), base.Progress())  # type: ignore[arg-type]

    b = Bench()
    monkeypatch.setattr(ble_midi, "scan_peripherals", b.scan)
    monkeypatch.setattr(ble_midi, "open_link", b.open_link)
    monkeypatch.setattr(ble_midi, "measure_latency", b.latency)
    monkeypatch.setattr(ble_midi, "host_peripheral", lambda: b.host)
    monkeypatch.setattr(ble_midi, "host_peripheral_available", lambda: b.host_available)
    return b


def names_of(rep: base.Report) -> list[str]:
    return [c["name"] for c in rep.data["checks"]]


# --------------------------------------------------------------------------- unit


def test_registered_and_defaults() -> None:
    assert get("ble_midi") is ble_midi.SCENARIO
    p = ble_midi.Params()
    assert (p.kit, p.name, p.scan_timeout, p.connect_timeout, p.host_mode) == (
        None, "CrossPad", 15.0, 20.0, False)


def test_int_reads_a_dash_as_the_default() -> None:
    assert ble_midi._int("12") == 12
    assert ble_midi._int(None) == 0            # the parser turns "-" into None
    assert ble_midi._int("nope", -1) == -1


# --------------------------------------------------------------------------- server role


def test_every_check_passes_against_a_healthy_board(bench) -> None:
    rep = bench.run()
    assert rep.passed, rep.summary
    assert rep.exit_code == 0
    assert names_of(rep) == [
        "advertises the BLE MIDI service",
        "characteristic has write-without-response + notify",
        "device reports the link up",
        "every note reached the device",
        "notes reached the sample engine",
        "multi-message packet decoded",
        "pad hits arrive as BLE MIDI notes",
        "note numbers match the pad map",
        "notes come back over the radio",
        "no BLE transmit errors",
    ]
    assert rep.summary == "ble_midi: PASS — 10/10 checks passed"
    # the RAM cost is measured, not assumed
    assert rep.data["ram"] == {"int_free_off": 90_000, "int_free_on": 50_000, "cost": 40_000}
    assert rep.data["conn_interval_ms"] == pytest.approx(15.0)
    assert rep.data["latency_ms"]["p50"] == 18.0
    assert rep.data["peripheral"]["address"] == ADDR
    assert bench.link.disconnected


def test_the_board_is_matched_on_the_address_it_reports(bench) -> None:
    """Not "the first BLE MIDI advertiser": that once tested a passing speaker."""
    other = BlePeripheral(address="11:22:33:44:55:66", name="CrossPad", rssi=-30,
                          service_uuids=["03b80e5a-ede8-4b33-a751-6ce34ec4c700"])
    bench.peripherals = [other, *bench.peripherals]
    rep = bench.run(scan_timeout=3.0)
    assert rep.passed, rep.summary
    assert bench.scanned == [3.0]
    assert rep.data["peripheral"]["address"] == ADDR


def test_a_board_that_does_not_advertise_fails_the_first_check_and_stops(bench) -> None:
    bench.peripherals = []
    rep = bench.run()
    assert not rep.passed and rep.exit_code == 1
    assert names_of(rep) == ["advertises the BLE MIDI service"]
    assert rep.data["failures"] == ["advertises the BLE MIDI service"]
    assert bench.link.disconnected is False        # never connected, nothing to release
    assert bench.cdc.role_calls[-1] == 1           # …and the role is still restored


def test_two_boards_answering_to_one_name_is_a_failed_check_not_a_guess(bench) -> None:
    bench.cdc.self_addr = "-"                      # firmware predating self=
    bench.peripherals.append(BlePeripheral(address="11:22:33:44:55:66", name="CrossPad",
                                           rssi=-30, service_uuids=[
                                               "03b80e5a-ede8-4b33-a751-6ce34ec4c700"]))
    rep = bench.run()
    assert not rep.passed
    assert "advertising as 'CrossPad'" in rep.data["checks"][0]["detail"]


def test_a_characteristic_without_notify_is_named(bench) -> None:
    bench.link.properties = ["write-without-response"]
    rep = bench.run()
    assert not rep.passed
    assert rep.data["failures"] == ["characteristic has write-without-response + notify"]
    assert rep.data["checks"][1]["detail"] == "write-without-response"


def test_notes_that_never_reach_the_sample_engine_are_told_from_notes_that_never_arrived(bench) -> None:
    """The two counters are read independently, which is the whole point."""
    class DeafDevice(BleDevice):
        def receive(self, messages: list[list[int]]) -> None:
            pass                                   # the radio wrote, the device parsed nothing

    bench.cdc.__class__ = DeafDevice
    rep = bench.run()
    assert not rep.passed
    assert "every note reached the device" in rep.data["failures"]
    assert "multi-message packet decoded" in rep.data["failures"]
    # Nothing was parsed, so nothing sounded either: a deaf radio fails both.
    assert "notes reached the sample engine" in rep.data["failures"]

    # The counters earn their separation on the other case — the notes arrive
    # and are parsed, and the engine behind them is the thing that is dead.
    bench.cdc.__class__ = BleDevice
    bench.cdc.engine_plays = 0
    rep = bench.run()
    assert not rep.passed
    assert "every note reached the device" not in rep.data["failures"]
    assert "notes reached the sample engine" in rep.data["failures"]


def test_the_expected_notes_are_the_pad_map_plus_the_send_transpose(bench) -> None:
    bench.cdc.txoff = 36
    rep = bench.run()
    assert rep.passed, rep.summary
    check = next(c for c in rep.data["checks"] if c["name"] == "note numbers match the pad map")
    assert check["ok"] and "transpose +36" in check["detail"]
    assert rep.data["txoff"] == 36


def test_a_note_map_the_device_does_not_use_fails_that_check(bench) -> None:
    rep_notes = {p: 36 + p for p in range(16)}
    bench.cdc.notes = rep_notes
    # the device answers PAD_NOTES with one map and sends another
    bench.cdc.txoff = 0
    original = bench.cdc.write

    def shifted(data: bytes) -> int:
        text = data.decode()
        if text.startswith("PAD_PRESS"):
            bench.cdc.notes = {p: 100 + p for p in range(16)}
        out = original(data)
        bench.cdc.notes = rep_notes
        return out

    bench.cdc.write = shifted  # type: ignore[method-assign]
    rep = bench.run()
    assert not rep.passed
    assert rep.data["failures"] == ["note numbers match the pad map"]


def test_notes_that_do_not_come_back_fail_the_latency_check(bench) -> None:
    bench.latency_result = {"sent": 8, "received": 6, "lost": 2, "rtt_ms": None}
    rep = bench.run()
    assert not rep.passed
    assert rep.data["failures"] == ["notes come back over the radio"]
    assert rep.data["checks"][-2]["detail"] == "6 of 8 arrived"


def test_transmit_errors_fail_the_last_check(bench) -> None:
    bench.cdc.tx_err = 3
    rep = bench.run()
    assert not rep.passed
    assert rep.data["failures"] == ["no BLE transmit errors"]


def test_a_kit_is_loaded_and_the_sampler_started_when_asked(bench) -> None:
    rep = bench.run(kit=8)
    assert rep.passed, rep.summary
    assert "KIT_LOAD 8" in bench.cdc.written
    assert "APP_START Sampler" in bench.cdc.written


def test_firmware_without_ble_is_exit_2(bench) -> None:
    bench.cdc.supported = 0
    rep = bench.run()
    assert not rep.passed and rep.exit_code == 2
    assert "CONFIG_BT_NIMBLE_ENABLED" in rep.summary
    assert bench.cdc.role_calls == []       # nothing was started, nothing to restore


def test_a_refused_ble_start_is_exit_2_not_a_missing_advert(bench) -> None:
    bench.cdc.start_refused = True
    rep = bench.run()
    assert not rep.passed and rep.exit_code == 2
    assert "BLE_START refused" in rep.summary


def test_no_note_map_is_exit_2(bench) -> None:
    bench.cdc.notes = {}
    rep = bench.run()
    assert not rep.passed and rep.exit_code == 2
    assert "PAD_NOTES" in rep.summary


def test_the_role_the_board_was_found_in_is_restored(bench) -> None:
    bench.cdc.mode = "host"
    rep = bench.run()
    assert rep.data["mode_before"] == "host"
    # 1 to run the server-role checks, 0 to put it back the way it was
    assert bench.cdc.role_calls == [1, 0]
    assert rep.data["restored"] is True


def test_the_role_is_restored_even_when_a_check_blows_up(bench, monkeypatch) -> None:
    def explode(*_a: Any, **_kw: Any) -> Any:
        raise HilError(ENV, "the radio fell over")

    monkeypatch.setattr(ble_midi, "scan_peripherals", explode)
    with pytest.raises(HilError):
        bench.run()
    assert bench.cdc.role_calls[-1] == 1


def test_a_board_found_with_ble_off_is_left_with_ble_off(bench) -> None:
    """The gate's ble_midi used to end every run with BLE_START 1: a board that
    started with the stack down kept ~30 kB less internal RAM for every later
    step (int_free 63 kB, Sampler pads quiet in the release gate)."""
    bench.cdc.running = 0
    rep = bench.run()
    assert rep.passed, rep.summary
    assert rep.data["running_before"] is False
    assert bench.cdc.written[-1] == "BLE_STOP" and bench.cdc.running == 0
    assert bench.cdc.role_calls == [1]          # only the run's own start


def test_ble_off_is_restored_even_when_a_check_blows_up(bench, monkeypatch) -> None:
    def explode(*_a: Any, **_kw: Any) -> Any:
        raise HilError(ENV, "the radio fell over")

    bench.cdc.running = 0
    monkeypatch.setattr(ble_midi, "scan_peripherals", explode)
    with pytest.raises(HilError):
        bench.run()
    assert bench.cdc.running == 0


def test_a_restore_that_fails_is_exit_2_not_a_pass(bench, monkeypatch) -> None:
    def refuse(link: Any, mode: Any = None) -> Any:
        if bench.cdc.role_calls:                   # the run's own start went through
            raise HilError(NOT_SUPPORTED, "BLE_START: device replied ERR")
        bench.cdc.role_calls.append(mode)
        bench.cdc.running = 1
        return {}

    from crosspad_hil import verbs
    monkeypatch.setattr(verbs, "ble_start", refuse)
    rep = bench.run()
    assert rep.exit_code == 2 and "could not be left running" in rep.summary


# --------------------------------------------------------------------------- host role


def test_host_mode_is_skipped_loudly_where_it_cannot_be_run(bench) -> None:
    bench.host_available = False
    rep = bench.run(host_mode=True)
    assert rep.passed
    assert rep.data["host_mode"]["skipped"] is True
    assert "host role skipped" in rep.summary
    assert bench.host.started is False
    # no check was invented for it: a skipped check reporting PASS is worse than none
    assert "scan finds a BLE MIDI peripheral" not in names_of(rep)


def test_host_mode_drives_the_board_as_a_central(bench) -> None:
    bench.cdc.peer_devices = ["99:88:77:66:55:44"]
    rep = bench.run(host_mode=True)
    assert rep.passed, rep.summary
    assert names_of(rep)[-4:] == [
        "scan finds a BLE MIDI peripheral",
        "board connects as central",
        "board subscribed to the peer",
        "peer notifications reach the board",
    ] + [] or names_of(rep)[-1] == "pad hits reach the peer"
    assert "BLE_SCAN 8000" in bench.cdc.written
    assert "BLE_CONNECT 99:88:77:66:55:44" in bench.cdc.written
    assert bench.host.started and bench.host.stopped
    assert "BLE_DISCONNECT" in bench.cdc.written
    assert rep.data["host_mode"]["writes"] == 6
    assert bench.cdc.role_calls == [1, 0, 1]      # server checks, host checks, restore


def test_host_mode_stops_when_the_board_finds_nothing(bench) -> None:
    bench.cdc.peer_devices = []
    rep = bench.run(host_mode=True)
    assert not rep.passed and rep.exit_code == 1
    assert rep.data["failures"] == ["scan finds a BLE MIDI peripheral"]
    assert bench.host.stopped                     # the advert is taken down regardless
    assert bench.cdc.role_calls[-1] == 1


def test_host_mode_notices_a_peer_the_board_never_subscribed_to(bench) -> None:
    bench.cdc.peer_devices = ["99:88:77:66:55:44"]
    bench.host.clients = 0
    rep = bench.run(host_mode=True)
    assert not rep.passed
    assert "board subscribed to the peer" in rep.data["failures"]


def test_a_missing_cdc_port_is_the_frameworks_environment_error(bench) -> None:
    bench.ctx.device = Device(id="dev_ab12", serial="CP-1", usb_mode=UsbMode.AUDIO,
                              ports=Ports(cdc=None, console=CONSOLE_PORT, esp_midi=ESP_MIDI))

    def no_cdc() -> CdcLink:
        raise HilError(NO_DEVICE, "no CDC endpoint in the MIDI+UAC2 profile")

    bench.ctx.open_cdc = no_cdc  # type: ignore[method-assign]
    with pytest.raises(HilError) as e:
        bench.run()
    assert e.value.code == NO_DEVICE
