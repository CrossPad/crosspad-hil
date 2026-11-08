from __future__ import annotations

import sys
import types
from dataclasses import replace
from pathlib import Path

import pytest

from crosspad_hil.devices import (
    AudioCardInfo,
    Device,
    MidiPortInfo,
    Ports,
    SerialPortInfo,
    UsbMode,
)
from crosspad_hil.errors import FLASH_FAILED, NO_DEVICE, NOT_SUPPORTED, HilError
from crosspad_hil.midi import MidiIO
from crosspad_hil.ota import (
    CHUNK_SIZE,
    OTA_ACK_TIMEOUT_S,
    OTA_WAIT_TIMEOUT_S,
    firmware_version,
    flash,
    ota_flash,
    request_bootloader,
)
from tests.fakes import FakeMidiBus

CDC = SerialPortInfo(path="/dev/ttyACM0", vid=0x303A, pid=0x3456, serial="ABCD1234",
                     product="Crosspad", location="1-2:1.0")
CON = SerialPortInfo(path="/dev/ttyACM1", vid=0x0483, pid=0x5740, serial="STM1",
                     product="CrossPad MIDI+Serial", location="1-2:1.2")
BOOT = SerialPortInfo(path="/dev/ttyACM2", vid=0x303A, pid=0x1001, serial=None,
                      product="USB JTAG/serial debug unit", location="1-2:1.0")
MIDI = MidiPortInfo(name="Crosspad MIDI 1", rtmidi_out=0, rtmidi_in=0, alsa_hw=None,
                    rawmidi=None)
UAC = AudioCardInfo(name="Crosspad Audio", sounddevice_index=3, alsa_id="hw:3")


def dev_default() -> Device:
    return Device(id="dev_1234", serial="ABCD1234", usb_mode=UsbMode.DEFAULT,
                  ports=Ports(cdc=CDC, console=CON, esp_midi=MIDI))


def dev_audio() -> Device:
    return Device(id="dev_1234", serial="ABCD1234", usb_mode=UsbMode.AUDIO,
                  ports=Ports(uac2=UAC, console=CON, esp_midi=MIDI))


def make_firmware(tmp_path: Path, size: int, version: str = "v2.0.1-7-gabc") -> Path:
    body = bytearray((i * 7 + 3) & 0xFF for i in range(size))
    ver = version.encode() + b"\0"
    body[48:48 + len(ver)] = ver
    p = tmp_path / "CrossPad.bin"
    p.write_bytes(bytes(body))
    return p


class FakeOtaSerial:
    """Plays the device side of ota_cdc.cpp: OTA_BEGIN handshake, per-write 'OK recv/total',
    final OTA_OK. Text commands are matched; binary writes are counted as firmware bytes."""

    def __init__(self, *, wait_first: bool = False, ready: bool = True,
                 fail_after: int | None = None, drop_ack_after: int | None = None) -> None:
        self.wait_first = wait_first
        self.ready = ready
        self.fail_after = fail_after
        self.drop_ack_after = drop_ack_after
        self.queue: list[bytes] = []
        self.written: list[bytes] = []
        self.timeout: float = 0.0
        self.timeouts: list[float] = []
        self.total = 0
        self.received = 0
        self.began: str | None = None
        self.is_open = True
        self.dtr = False
        self.rts = False
        self.control_history: list[tuple[str, bool]] = []
        self.input_resets = 0

    def __setattr__(self, name: str, value) -> None:
        if name == "timeout" and "timeouts" in self.__dict__:
            self.__dict__["timeouts"].append(value)
        if name in ("dtr", "rts") and "control_history" in self.__dict__:
            self.__dict__["control_history"].append((name, value))
        object.__setattr__(self, name, value)

    def reset_input_buffer(self) -> None:
        self.input_resets += 1

    def write(self, data: bytes) -> int:
        self.written.append(bytes(data))
        if self.began is None:
            text = data.decode("ascii", "replace").strip()
            if not text.startswith("OTA_"):
                return len(data)              # BOOTLOADER_REQUEST / STM_DFU: no reply
            self.began = text
            parts = text.split(" ")
            self.total = int(parts[1])
            if self.wait_first:
                self.queue.append(b"OTA_WAIT\r\n")
            self.queue.append(b"OTA_READY\r\n" if self.ready else b"OTA_ERROR no_partition\r\n")
            return len(data)
        self.received += len(data)
        if self.fail_after is not None and self.received >= self.fail_after:
            self.queue.append(b"OTA_ERROR write_ESP_ERR_FLASH_OP_FAIL\r\n")
            return len(data)
        if self.drop_ack_after is not None and self.received >= self.drop_ack_after:
            return len(data)
        self.queue.append(f"OK {self.received}/{self.total}\r\n".encode())
        if self.received >= self.total:
            self.queue.append(b"I (1234) ota_cdc: OTA finished\r\n")  # noise before OTA_OK
            self.queue.append(b"OTA_OK\r\n")
        return len(data)

    def readline(self) -> bytes:
        if self.queue:
            return self.queue.pop(0)
        return b""

    def close(self) -> None:
        self.is_open = False


def factory_for(ser: FakeOtaSerial):
    calls: list[dict] = []

    def factory(path: str, **kw):
        calls.append({"path": path, **kw})
        return ser
    factory.calls = calls  # type: ignore[attr-defined]
    return factory


def test_firmware_version_offset_48(tmp_path: Path) -> None:
    fw = make_firmware(tmp_path, 100, "v1.2.3")
    assert firmware_version(fw) == "v1.2.3"
    assert firmware_version(tmp_path / "missing.bin") == ""


def test_ota_flash_full_dialogue(tmp_path: Path) -> None:
    fw = make_firmware(tmp_path, CHUNK_SIZE * 2 + 100)
    ser = FakeOtaSerial()
    fac = factory_for(ser)
    seen: list[tuple[int, int]] = []
    ticks = iter(range(0, 1000))
    r = ota_flash(dev_default(), fw, progress=lambda a, b: seen.append((a, b)),
                  serial_factory=fac, sleep=lambda s: None, clock=lambda: next(ticks) * 0.5)
    assert fac.calls[0]["path"] == "/dev/ttyACM0"
    assert fac.calls[0]["baud"] == 115200 and fac.calls[0]["timeout"] == OTA_ACK_TIMEOUT_S
    assert ser.input_resets == 1
    assert ser.began == f"OTA_BEGIN {fw.stat().st_size} v2.0.1-7-gabc"
    assert ser.written[0] == f"OTA_BEGIN {fw.stat().st_size} v2.0.1-7-gabc\n".encode()
    chunks = ser.written[1:]
    assert [len(c) for c in chunks] == [CHUNK_SIZE, CHUNK_SIZE, 100]
    assert b"".join(chunks) == fw.read_bytes()
    total = fw.stat().st_size
    assert seen == [(CHUNK_SIZE, total), (2 * CHUNK_SIZE, total), (total, total)]
    assert r["bytes"] == total and r["mode"] == "full" and r["version"] == "v2.0.1-7-gabc"
    assert r["seconds"] > 0 and r["kbps"] > 0
    assert not ser.is_open


def test_ota_flash_ota_wait_extends_timeout(tmp_path: Path) -> None:
    fw = make_firmware(tmp_path, 10)
    ser = FakeOtaSerial(wait_first=True)
    ota_flash(dev_default(), fw, serial_factory=factory_for(ser), sleep=lambda s: None)
    # ota_flash.py ota_begin(): 90 s while waiting for the on-device confirmation, then 30 s
    assert OTA_WAIT_TIMEOUT_S in ser.timeouts
    assert ser.timeouts[-1] == OTA_ACK_TIMEOUT_S or ser.timeouts[-1] == 60


def test_ota_flash_error_paths(tmp_path: Path) -> None:
    fw = make_firmware(tmp_path, CHUNK_SIZE * 3)
    ser = FakeOtaSerial(fail_after=CHUNK_SIZE * 2)
    with pytest.raises(HilError) as e:
        ota_flash(dev_default(), fw, serial_factory=factory_for(ser), sleep=lambda s: None)
    assert e.value.code == FLASH_FAILED
    assert e.value.details["reply"].startswith("OTA_ERROR write_")
    assert not ser.is_open

    ser2 = FakeOtaSerial(ready=False)
    with pytest.raises(HilError) as e:
        ota_flash(dev_default(), fw, serial_factory=factory_for(ser2), sleep=lambda s: None)
    assert e.value.code == FLASH_FAILED
    assert "OTA_READY" in e.value.message

    ser3 = FakeOtaSerial(drop_ack_after=CHUNK_SIZE)
    with pytest.raises(HilError) as e:
        ota_flash(dev_default(), fw, serial_factory=factory_for(ser3), sleep=lambda s: None)
    assert e.value.code == FLASH_FAILED
    assert "no ack" in e.value.message

    with pytest.raises(HilError) as e:
        ota_flash(dev_default(), tmp_path / "nope.bin", serial_factory=factory_for(ser))
    assert e.value.code == FLASH_FAILED


def test_ota_flash_leaves_audio_mode_first(tmp_path: Path) -> None:
    fw = make_firmware(tmp_path, 10)
    ser = FakeOtaSerial()
    switched: list[UsbMode] = []

    def fake_set_mode(device: Device, mode: UsbMode, **kw) -> Device:
        switched.append(mode)
        return dev_default()

    ota_flash(dev_audio(), fw, serial_factory=factory_for(ser), set_mode_fn=fake_set_mode,
              sleep=lambda s: None)
    assert switched == [UsbMode.DEFAULT]
    assert ser.began is not None


def test_ota_flash_delta_requires_detools(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    fw = make_firmware(tmp_path, 300)
    base = tmp_path / "base.bin"
    base.write_bytes(bytes(300))
    monkeypatch.setitem(sys.modules, "detools", None)   # import fails
    with pytest.raises(HilError) as e:
        ota_flash(dev_default(), fw, delta_base=base, serial_factory=factory_for(FakeOtaSerial()))
    assert e.value.code == NOT_SUPPORTED

    fake = types.ModuleType("detools")

    def create_patch(ffrom, fto, fpatch, compression="heatshrink") -> None:
        assert compression == "heatshrink"
        fpatch.write(b"PATCH" + fto.read()[:20])
    fake.create_patch = create_patch  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "detools", fake)
    ser = FakeOtaSerial()
    r = ota_flash(dev_default(), fw, delta_base=base, serial_factory=factory_for(ser),
                  sleep=lambda s: None)
    assert ser.began == "OTA_DELTA 25 v2.0.1-7-gabc"
    assert r["mode"] == "delta" and r["bytes"] == 25


def test_request_bootloader_cdc_then_poll(tmp_path: Path) -> None:
    ser = FakeOtaSerial()
    boot_dev = Device(id="dev_boot", serial=None, usb_mode=UsbMode.BOOTLOADER,
                      ports=Ports(bootloader=BOOT))
    inventories = iter([[dev_default()], [], [boot_dev]])
    bus = FakeMidiBus()
    slept: list[float] = []
    r = request_bootloader(dev_default(), "esp", method="cdc,midi",
                           serial_factory=factory_for(ser),
                           midi_factory=lambda d, role: MidiIO(d, role, out_factory=bus.out_factory,
                                                               in_factory=bus.in_factory),
                           discover_fn=lambda: next(inventories), sleep=slept.append)
    assert ser.written == [b"BOOTLOADER_REQUEST\n"]
    assert bus.sent == []                      # CDC succeeded; MIDI not attempted
    assert r == {"bootloader_port": "/dev/ttyACM2"}
    assert 3.0 in slept                        # BOOTLOADER_SETTLE_S before the first scan


def test_request_bootloader_midi_fallback_and_timeout() -> None:
    bus = FakeMidiBus()
    no_cdc = replace(dev_default(), ports=Ports(esp_midi=MIDI, console=CON))
    clock = iter([0.0, 0.0, 4.0, 8.0, 12.0, 16.0])
    r = request_bootloader(no_cdc, "esp", method="cdc,midi", timeout_s=10.0,
                           midi_factory=lambda d, role: MidiIO(d, role, out_factory=bus.out_factory,
                                                               in_factory=bus.in_factory),
                           discover_fn=lambda: [], sleep=lambda s: None,
                           clock=lambda: next(clock))
    assert bus.sent == [[0xF0, 0x7D, 0x19, 0x00, 0xF7]]
    assert r == {"bootloader_port": None}
    nothing = replace(dev_default(), ports=Ports(console=CON))
    with pytest.raises(HilError) as e:
        request_bootloader(nothing, "esp", discover_fn=lambda: [])
    assert e.value.code == NO_DEVICE


def test_request_bootloader_stm_sends_dfu_only() -> None:
    ser = FakeOtaSerial()
    r = request_bootloader(dev_default(), "stm", serial_factory=factory_for(ser),
                           discover_fn=lambda: [], sleep=lambda s: None)
    assert ser.written == [b"STM_DFU\n"]
    assert r == {"bootloader_port": None}
    with pytest.raises(HilError) as e:
        request_bootloader(dev_default(), "fpga")
    assert e.value.code == NOT_SUPPORTED


class FakeConsole:
    def __init__(self, port: str, **kw) -> None:
        self.port = port
        self.kw = kw
        self.opened: list[bool] = []
        self.closed = False

    def open(self, reset: bool = False) -> None:
        self.opened.append(reset)

    def close(self) -> None:
        self.closed = True

    def wait_boot(self, timeout_s=None, settle_s: float = 3.0):
        from crosspad_hil.console import BootResult
        return BootResult(complete=True, missing=[], fatal=[], errors=[], bootloops=0,
                          seconds=12.5)


def test_flash_composes_ota_and_wait_boot(tmp_path: Path) -> None:
    fw = make_firmware(tmp_path, 10)
    made: list[FakeConsole] = []

    def console_factory(port: str, **kw) -> FakeConsole:
        c = FakeConsole(port, **kw)
        made.append(c)
        return c

    def fake_ota(device, firmware, *, progress=None, **kw) -> dict:
        return {"bytes": 10, "seconds": 1.0, "kbps": 0.01, "version": "v", "mode": "full"}

    r = flash(dev_default(), fw, ota_flash_fn=fake_ota, console_factory=console_factory)
    assert r["flash"]["bytes"] == 10
    assert r["boot"]["complete"] is True and r["boot"]["seconds"] == 12.5
    assert made[0].port == "/dev/ttyACM1" and made[0].opened == [False] and made[0].closed

    r2 = flash(dev_default(), fw, wait_boot=False, ota_flash_fn=fake_ota,
               console_factory=console_factory)
    assert r2["boot"] is None and len(made) == 1

    given = FakeConsole("/dev/ttyACM1")
    r3 = flash(dev_default(), fw, console=given, ota_flash_fn=fake_ota,
               console_factory=console_factory)
    assert r3["boot"]["complete"] is True and given.opened == [] and not given.closed

    with pytest.raises(HilError) as e:
        flash(dev_default(), fw, transport="uart", ota_flash_fn=fake_ota)
    assert e.value.code == NOT_SUPPORTED
