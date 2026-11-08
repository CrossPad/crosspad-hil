"""OTA over USB CDC, ROM-bootloader request, and flash-then-wait-for-boot.

Byte-for-byte port of platform-idf/tools/ota_flash.py (ota_begin, send_data_to_device,
get_firmware_version, generate_delta_patch) and tools/requestBootloader.py
(try_bootloader_request, find_bootloader_device polling). The device side is
main/ota_cdc.cpp: OTA_BEGIN|OTA_DELTA <size> <version> → OTA_WAIT? → OTA_READY;
per 4096-byte write "OK <recv>/<total>"; OTA_OK or "OTA_ERROR <reason>".
"""
from __future__ import annotations

import io
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

from crosspad_hil.console import Console
from crosspad_hil.devices import Device, UsbMode, discover
from crosspad_hil.errors import (
    FLASH_FAILED,
    NO_CDC_IN_AUDIO_MODE,
    NO_DEVICE,
    NOT_SUPPORTED,
    TIMEOUT,
    HilError,
)
from crosspad_hil.midi import MidiIO, MidiRole, sysex_bootloader_esp
from crosspad_hil.serial_open import open_serial
from crosspad_hil.usbmode import set_mode

CHUNK_SIZE = 4096            # ota_flash.py CHUNK_SIZE (== OTA_WRITE_BUF_SIZE on device)
OTA_ACK_TIMEOUT_S = 30       # ota_flash.py serial.Serial(port, 115200, timeout=30)
OTA_WAIT_TIMEOUT_S = 90      # ota_flash.py ota_begin(): user confirmation on the device
OTA_FINAL_TIMEOUT_S = 60     # ota_flash.py send_data_to_device(): delta finalization
OTA_FINAL_READS = 30         # ota_flash.py: for _ in range(30)
OTA_CONNECT_SETTLE_S = 0.5   # ota_flash.py: time.sleep(0.5) after opening the port
BOOTLOADER_SETTLE_S = 3.0    # requestBootloader.py: time.sleep(3) before scanning
BOOTLOADER_POLL_S = 0.5      # requestBootloader.py: poll_interval
CDC_BOOTLOADER_CMD = "BOOTLOADER_REQUEST"   # main.cpp check_usb_bootloader_request
CDC_STM_DFU_CMD = "STM_DFU"


def firmware_version(path: Path) -> str:
    """ESP-IDF app version: esp_app_desc_t.version at bin offset 48 (0x30), 32 bytes."""
    try:
        with open(path, "rb") as f:
            f.seek(48)
            return f.read(32).split(b"\0")[0].decode("utf-8", "ignore")
    except OSError:
        return ""


def _readline(ser: Any) -> str:
    return ser.readline().decode(errors="ignore").strip()


def _fail(ser: Any, message: str, **details: Any) -> HilError:
    try:
        ser.close()
    except Exception:
        pass
    return HilError(FLASH_FAILED, message, hint="check the device console (STM VCP) for ota_cdc "
                    "logs; a rejected OTA_WAIT shows as OTA_ERROR rejected", **details)


def _ota_begin(ser: Any, command: str) -> None:
    """ota_flash.py ota_begin(): OTA_BEGIN/OTA_DELTA, tolerate OTA_WAIT, require OTA_READY."""
    ser.write(command.encode())
    response = _readline(ser)
    if response == "OTA_WAIT":
        ser.timeout = OTA_WAIT_TIMEOUT_S
        response = _readline(ser)
        ser.timeout = OTA_ACK_TIMEOUT_S
    if response != "OTA_READY":
        raise _fail(ser, f"expected OTA_READY, got {response!r}", reply=response)


def _send_data(ser: Any, data: bytes, *, progress: Callable[[int, int], None] | None,
               deadline: float, clock: Callable[[], float]) -> None:
    """ota_flash.py send_data_to_device(): chunked write, ack per chunk, final OTA_OK."""
    total = len(data)
    sent = 0
    last_reported = 0
    while sent < total:
        if clock() > deadline:
            ser.close()
            raise HilError(TIMEOUT, f"OTA exceeded its time budget at {sent}/{total} bytes",
                           sent=sent, total=total)
        chunk_end = min(sent + CHUNK_SIZE, total)
        ser.write(data[sent:chunk_end])
        sent = chunk_end
        while True:
            response = _readline(ser)
            if not response:
                raise _fail(ser, f"no ack from device (sent {sent}/{total})",
                            sent=sent, total=total)
            if response.startswith("OK "):
                received = int(response[3:].split("/")[0])
                if progress is not None and received > last_reported:
                    last_reported = received
                    progress(received, total)
                break
            if response == "OTA_OK":
                return
            if response.startswith("OTA_ERROR"):
                raise _fail(ser, f"device reported {response}", reply=response,
                            sent=sent, total=total)
            # stale echo / log noise: keep reading
    ser.timeout = OTA_FINAL_TIMEOUT_S
    response = ""
    for _ in range(OTA_FINAL_READS):
        response = _readline(ser)
        if response == "OTA_OK":
            return
        if response.startswith("OK "):
            received = int(response[3:].split("/")[0])
            if progress is not None and received > last_reported:
                last_reported = received
                progress(received, total)
            continue
        if response.startswith("OTA_ERROR"):
            raise _fail(ser, f"device reported {response}", reply=response,
                        sent=sent, total=total)
        if not response:
            break
    raise _fail(ser, f"no OTA_OK after transfer; final response {response!r}",
                reply=response, sent=sent, total=total)


def _delta_patch(base: Path, new: Path) -> bytes:
    """ota_flash.py generate_delta_patch(): detools heatshrink patch, in memory."""
    try:
        import detools  # type: ignore[import-not-found]
    except ImportError as exc:
        raise HilError(NOT_SUPPORTED, "delta OTA needs the 'detools' package",
                       hint="pip install detools") from exc
    if detools is None:
        raise HilError(NOT_SUPPORTED, "delta OTA needs the 'detools' package",
                       hint="pip install detools")
    patch = io.BytesIO()
    with open(base, "rb") as base_f, open(new, "rb") as new_f:
        detools.create_patch(base_f, new_f, patch, compression="heatshrink")
    return patch.getvalue()


def ota_flash(device: Device, firmware: Path, *, delta_base: Path | None = None,
              progress: Callable[[int, int], None] | None = None, timeout_s: float = 300.0,
              serial_factory: Callable[..., Any] = open_serial,
              set_mode_fn: Callable[..., Device] = set_mode,
              sleep: Callable[[float], None] = time.sleep,
              clock: Callable[[], float] = time.monotonic) -> dict:
    """Full or delta OTA over the ESP CDC port. Returns {bytes, seconds, kbps, version, mode}."""
    firmware = Path(firmware)
    if not firmware.is_file():
        raise HilError(FLASH_FAILED, f"firmware not found: {firmware}",
                       hint="build first: idf.py build (build/CrossPad.bin)")
    if delta_base is not None and not Path(delta_base).is_file():
        raise HilError(FLASH_FAILED, f"base firmware not found: {delta_base}")

    if device.usb_mode is UsbMode.AUDIO:
        # ota_flash.py ensure_port(): MIDI+UAC2 has no CDC → SysEx 0x1B 01 and wait
        device = set_mode_fn(device, UsbMode.DEFAULT)
    if device.ports.cdc is None:
        raise HilError(NO_CDC_IN_AUDIO_MODE if device.usb_mode is UsbMode.AUDIO else NO_DEVICE,
                       f"{device.id} has no CDC port for OTA",
                       hint="usbmode.set default, or check `crosspad-hil devices`")

    version = firmware_version(firmware)
    if delta_base is not None:
        payload = _delta_patch(Path(delta_base), firmware)
        command = f"OTA_DELTA {len(payload)} {version}\n"
        mode = "delta"
    else:
        payload = firmware.read_bytes()
        command = f"OTA_BEGIN {len(payload)} {version}\n"
        mode = "full"

    ser = serial_factory(device.ports.cdc.path, baud=115200, timeout=OTA_ACK_TIMEOUT_S)
    sleep(OTA_CONNECT_SETTLE_S)
    if hasattr(ser, "reset_input_buffer"):
        ser.reset_input_buffer()
    t0 = clock()
    _ota_begin(ser, command)
    _send_data(ser, payload, progress=progress, deadline=t0 + timeout_s, clock=clock)
    elapsed = max(clock() - t0, 1e-6)
    try:
        ser.close()
    except Exception:
        pass
    return {"bytes": len(payload), "seconds": round(elapsed, 3),
            "kbps": round(len(payload) / elapsed / 1024, 1), "version": version, "mode": mode}


def _send_cdc_line(device: Device, text: str, serial_factory: Callable[..., Any],
                   sleep: Callable[[float], None]) -> bool:
    """requestBootloader.py send_usb_cdc_message(): write line, no reply expected."""
    if device.ports.cdc is None:
        return False
    ser = serial_factory(device.ports.cdc.path, baud=115200, timeout=2)
    try:
        sleep(0.1)
        ser.write(text.encode("utf-8") + b"\n")
        if hasattr(ser, "flush"):
            ser.flush()
        sleep(0.1)
    finally:
        ser.close()
    return True


def request_bootloader(device: Device, target: str = "esp", *, method: str = "cdc,midi",
                       timeout_s: float = 10.0,
                       serial_factory: Callable[..., Any] = open_serial,
                       midi_factory: Callable[..., Any] = MidiIO,
                       discover_fn: Callable[[], list[Device]] = discover,
                       sleep: Callable[[float], None] = time.sleep,
                       clock: Callable[[], float] = time.monotonic) -> dict:
    """Ask the ESP for its ROM bootloader (or the STM for DFU). {bootloader_port: path|None}."""
    if target == "stm":
        if not _send_cdc_line(device, CDC_STM_DFU_CMD, serial_factory, sleep):
            raise HilError(NO_DEVICE, f"{device.id} has no CDC port to send {CDC_STM_DFU_CMD}")
        return {"bootloader_port": None}
    if target != "esp":
        raise HilError(NOT_SUPPORTED, f"bootloader target {target!r}", hint="use 'esp' or 'stm'")

    methods = [m.strip().lower() for m in method.split(",") if m.strip()]
    for m in methods:
        if m not in ("cdc", "midi"):
            raise HilError(NOT_SUPPORTED, f"bootloader method {m!r}", hint="cdc, midi")
    sent = False
    for m in methods:
        if m == "cdc" and _send_cdc_line(device, CDC_BOOTLOADER_CMD, serial_factory, sleep):
            sent = True
            break
        if m == "midi" and device.ports.esp_midi is not None:
            # requestBootloader.py send_sysex_message(): F0 7D 19 00 F7 on the ESP port
            with midi_factory(device, MidiRole.ESP) as mio:
                mio.send_sysex(sysex_bootloader_esp())
            sleep(0.1)
            sent = True
            break
    if not sent:
        raise HilError(NO_DEVICE, f"{device.id} has neither a CDC nor an ESP MIDI port",
                       hint="is it already in bootloader mode? run `crosspad-hil devices`")

    sleep(BOOTLOADER_SETTLE_S)
    deadline = clock() + timeout_s
    while True:
        for d in discover_fn():
            if d.ports.bootloader is not None:
                return {"bootloader_port": d.ports.bootloader.path}
        if clock() >= deadline:
            return {"bootloader_port": None}
        sleep(BOOTLOADER_POLL_S)


def flash(device: Device, firmware: Path, *, transport: str = "ota", wait_boot: bool = True,
          console: Console | None = None, progress: Callable[[int, int], None] | None = None,
          ota_flash_fn: Callable[..., dict] = ota_flash,
          console_factory: Callable[..., Any] = Console) -> dict:
    """ota_flash() and then Console.wait_boot() on the STM VCP. {flash, boot}."""
    if transport != "ota":
        raise HilError(NOT_SUPPORTED, f"flash transport {transport!r} is not available in P0",
                       hint="use idf.py -p <PORT> flash for UART, or transport='ota'")
    result = ota_flash_fn(device, firmware, progress=progress)
    boot: dict | None = None
    if wait_boot:
        if console is not None:
            boot = dict(vars(console.wait_boot()))
        elif device.ports.console is not None:
            own = console_factory(device.ports.console.path)
            own.open(reset=False)
            try:
                boot = dict(vars(own.wait_boot()))
            finally:
                own.close()
    return {"flash": result, "boot": boot}
