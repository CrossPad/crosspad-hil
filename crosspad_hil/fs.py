"""Bulk transfers over the CrossPad CDC port: the assets image, and files.

Device side is platform-idf `main/file_transfer.cpp`. These three verbs take the
pipe out of newline framing for a byte count announced in their header line, so
they cannot go through `CdcLink.transact()` the way every text verb does — the
same reason `ota.py` exists beside `verbs.py`.

    ASSETS_BEGIN <size> <crc32> [FORCE]  -> ASSETS_READY -> payload -> ASSETS_OK
    FS_PUT <size> <crc32> <path>         -> FS_READY     -> payload -> FS_OK <crc>
    FS_GET <path> [off] [len]            -> FS_DATA <n>  -> payload -> FS_OK <crc>

CRC32 is zlib.crc32, which is what `esp_crc32_le(0, ...)` computes.

`assets_flash()` reboots the device on success — the launcher holds icon state
that a remount would not refresh — so callers must wait for re-enumeration.
"""
from __future__ import annotations

import time
import zlib
from collections.abc import Callable
from pathlib import Path
from typing import Any

from .devices import Device, UsbMode
from .errors import FLASH_FAILED, NO_CDC_IN_AUDIO_MODE, NO_DEVICE, TIMEOUT, HilError
from .serial_open import open_serial
from .usbmode import set_mode

CHUNK_SIZE = 4096
# Chunks the host may have unacked at once. One is strict lockstep and costs a
# full USB turnaround per 4 KB (measured 91 KB/s); two hides that latency behind
# the next write (139 KB/s). Larger windows do not help and eventually lose
# bytes -- the device's CDC RX buffer is 128 B, so the acks are the flow control.
ACK_WINDOW = 2
ACK_TIMEOUT_S = 10.0
# The device erases and rewrites a 1 MB partition, then reads it back to verify.
ASSETS_FINAL_TIMEOUT_S = 60.0
CONNECT_SETTLE_S = 0.5


def _readline(ser: Any) -> str:
    return ser.readline().decode(errors="ignore").strip()


def _fail(ser: Any, message: str, **details: Any) -> HilError:
    try:
        ser.close()
    except Exception:
        pass
    return HilError(FLASH_FAILED, message,
                    hint="the STM VCP console carries the FX log lines", **details)


def _open_cdc(device: Device, set_mode_fn: Callable[..., Device],
              serial_factory: Callable[..., Any], sleep: Callable[[float], None]) -> Any:
    if device.usb_mode is UsbMode.AUDIO:
        device = set_mode_fn(device, UsbMode.DEFAULT)
    if device.ports.cdc is None:
        raise HilError(NO_CDC_IN_AUDIO_MODE if device.usb_mode is UsbMode.AUDIO else NO_DEVICE,
                       f"{device.id} has no CDC port",
                       hint="usbmode.set default, or check `crosspad-hil devices`")
    ser = serial_factory(device.ports.cdc.path, baud=115200, timeout=ACK_TIMEOUT_S)
    sleep(CONNECT_SETTLE_S)
    if hasattr(ser, "reset_input_buffer"):
        ser.reset_input_buffer()
    return ser


def _expect(ser: Any, want: str, err_prefixes: tuple[str, ...]) -> str:
    line = _readline(ser)
    if line.startswith(err_prefixes):
        raise _fail(ser, f"device reported {line}", reply=line)
    if not line.startswith(want):
        raise _fail(ser, f"expected {want}, got {line!r}", reply=line)
    return line


def _send_payload(ser: Any, data: bytes, *, ok_prefix: str, err_prefixes: tuple[str, ...],
                  progress: Callable[[int, int], None] | None,
                  deadline: float, clock: Callable[[], float],
                  final_timeout_s: float) -> str:
    """Chunked write with an ``OK n/total`` per chunk, terminated by ``ok_prefix``."""
    total = len(data)
    sent = acked = reported = 0
    while sent < total:
        if clock() > deadline:
            raise _fail(ser, f"transfer exceeded its budget at {sent}/{total} bytes",
                        sent=sent, total=total)
        while sent < total and (sent - acked) < ACK_WINDOW * CHUNK_SIZE:
            end = min(sent + CHUNK_SIZE, total)
            ser.write(data[sent:end])
            sent = end
        line = _readline(ser)
        if not line:
            raise _fail(ser, f"no ack from device (sent {sent}/{total})",
                        sent=sent, total=total)
        if line.startswith("OK "):
            acked = int(line[3:].split("/")[0])
            if progress is not None and acked > reported:
                reported = acked
                progress(acked, total)
        elif line.startswith(ok_prefix):
            return line
        elif line.startswith(err_prefixes):
            raise _fail(ser, f"device reported {line}", reply=line,
                        sent=sent, total=total)
        # anything else is log noise on the same pipe: keep reading

    ser.timeout = final_timeout_s
    deadline_final = clock() + final_timeout_s
    while clock() < deadline_final:
        line = _readline(ser)
        if line.startswith(ok_prefix):
            return line
        if line.startswith(err_prefixes):
            raise _fail(ser, f"device reported {line}", reply=line)
        if line.startswith("OK "):
            continue
        if not line:
            break
    raise HilError(TIMEOUT, f"no {ok_prefix} after the whole payload was sent",
                   hint="the device may still be verifying; check the console")


def assets_flash(device: Device, image: Path, *, force: bool = False,
                 progress: Callable[[int, int], None] | None = None,
                 timeout_s: float = 300.0,
                 serial_factory: Callable[..., Any] = open_serial,
                 set_mode_fn: Callable[..., Device] = set_mode,
                 sleep: Callable[[float], None] = time.sleep,
                 clock: Callable[[], float] = time.monotonic) -> dict[str, Any]:
    """Write a whole LittleFS assets image. The device reboots on success."""
    image = Path(image)
    if not image.is_file():
        raise HilError(FLASH_FAILED, f"assets image not found: {image}",
                       hint="idf.py build leaves it at build/assets.bin")
    payload = image.read_bytes()
    crc = zlib.crc32(payload) & 0xFFFFFFFF
    ser = _open_cdc(device, set_mode_fn, serial_factory, sleep)
    t0 = clock()
    cmd = f"ASSETS_BEGIN {len(payload)} {crc:08x}" + (" FORCE" if force else "") + "\n"
    ser.write(cmd.encode())
    _expect(ser, "ASSETS_READY", ("ASSETS_ERROR", "ERR"))
    _send_payload(ser, payload, ok_prefix="ASSETS_OK", err_prefixes=("ASSETS_ERROR", "ERR"),
                  progress=progress, deadline=t0 + timeout_s, clock=clock,
                  final_timeout_s=ASSETS_FINAL_TIMEOUT_S)
    elapsed = max(clock() - t0, 1e-6)
    try:
        ser.close()
    except Exception:
        pass
    return {"bytes": len(payload), "crc32": f"{crc:08x}", "seconds": round(elapsed, 3),
            "kbps": round(len(payload) / elapsed / 1024, 1), "rebooted": True}


def push(device: Device, local: Path, remote: str, *,
         progress: Callable[[int, int], None] | None = None, timeout_s: float = 300.0,
         serial_factory: Callable[..., Any] = open_serial,
         set_mode_fn: Callable[..., Device] = set_mode,
         sleep: Callable[[float], None] = time.sleep,
         clock: Callable[[], float] = time.monotonic) -> dict[str, Any]:
    """Host file -> device path. The device only renames it into place on a CRC match."""
    local = Path(local)
    if not local.is_file():
        raise HilError(FLASH_FAILED, f"file not found: {local}")
    payload = local.read_bytes()
    crc = zlib.crc32(payload) & 0xFFFFFFFF
    ser = _open_cdc(device, set_mode_fn, serial_factory, sleep)
    t0 = clock()
    ser.write(f"FS_PUT {len(payload)} {crc:08x} {remote}\n".encode())
    _expect(ser, "FS_READY", ("FS_ERR", "ERR"))
    reply = _send_payload(ser, payload, ok_prefix="FS_OK", err_prefixes=("FS_ERR", "ERR"),
                          progress=progress, deadline=t0 + timeout_s, clock=clock,
                          final_timeout_s=ACK_TIMEOUT_S)
    elapsed = max(clock() - t0, 1e-6)
    try:
        ser.close()
    except Exception:
        pass
    return {"bytes": len(payload), "crc32": f"{crc:08x}", "device_crc32": reply.split()[-1],
            "remote": remote, "seconds": round(elapsed, 3)}


def assets_dump(device: Device, local: Path, *, offset: int = 0,
                length: int | None = None, timeout_s: float = 300.0,
                serial_factory: Callable[..., Any] = open_serial,
                set_mode_fn: Callable[..., Device] = set_mode,
                sleep: Callable[[float], None] = time.sleep,
                clock: Callable[[], float] = time.monotonic) -> dict[str, Any]:
    """Read the assets partition back as a raw image (what ASSETS_BEGIN takes).

    This reads the flash, not the mounted filesystem, so the result round-trips:
    dump it, flash it back, and the partition is byte-identical.
    """
    cmd = f"ASSETS_DUMP {offset} {length or 0}"
    return _receive(device, cmd, local, timeout_s=timeout_s, serial_factory=serial_factory,
                    set_mode_fn=set_mode_fn, sleep=sleep, clock=clock, remote="<assets>")


def pull(device: Device, remote: str, local: Path, *, offset: int = 0,
         length: int | None = None, timeout_s: float = 300.0,
         serial_factory: Callable[..., Any] = open_serial,
         set_mode_fn: Callable[..., Device] = set_mode,
         sleep: Callable[[float], None] = time.sleep,
         clock: Callable[[], float] = time.monotonic) -> dict[str, Any]:
    """Device path -> host file. The CRC arrives in the trailer, after the bytes."""
    # off and len lead so the path can be the rest of the line: kit names have
    # spaces in them and a token-wise parse addressed the wrong file.
    cmd = f"FS_GET {offset} {length or 0} {remote}"
    return _receive(device, cmd, local, timeout_s=timeout_s, serial_factory=serial_factory,
                    set_mode_fn=set_mode_fn, sleep=sleep, clock=clock, remote=remote)


def _receive(device: Device, cmd: str, local: Path, *, timeout_s: float,
             serial_factory: Callable[..., Any], set_mode_fn: Callable[..., Device],
             sleep: Callable[[float], None], clock: Callable[[], float],
             remote: str) -> dict[str, Any]:
    """FS_DATA <len> -> exactly that many bytes -> FS_OK <crc> / FS_TRUNC."""
    ser = _open_cdc(device, set_mode_fn, serial_factory, sleep)
    t0 = clock()
    ser.write((cmd + "\n").encode())
    header = _expect(ser, "FS_DATA", ("FS_ERR", "ERR"))
    try:
        want = int(header.split()[1])
    except (IndexError, ValueError) as exc:
        raise _fail(ser, f"FS_DATA without a length: {header!r}", reply=header) from exc

    data = bytearray()
    while len(data) < want:
        if clock() > t0 + timeout_s:
            raise _fail(ser, f"pull stalled at {len(data)}/{want} bytes",
                        got=len(data), want=want)
        chunk = ser.read(min(CHUNK_SIZE, want - len(data)))
        if not chunk:
            raise _fail(ser, f"pull stalled at {len(data)}/{want} bytes",
                        got=len(data), want=want)
        data.extend(chunk)

    # The device writes the trailer as "\r\nFS_OK <crc>\r\n" so it starts on a
    # fresh line whatever the payload ended with; that leading break reads back
    # as one empty line before it.
    trailer = _readline(ser)
    while trailer == "" and clock() < t0 + timeout_s:
        trailer = _readline(ser)
    if trailer.startswith("FS_TRUNC"):
        raise _fail(ser, f"device truncated the read: {trailer}", reply=trailer)
    if not trailer.startswith("FS_OK"):
        raise _fail(ser, f"expected FS_OK trailer, got {trailer!r}", reply=trailer)
    device_crc = trailer.split()[-1]
    local_crc = f"{zlib.crc32(bytes(data)) & 0xFFFFFFFF:08x}"
    if device_crc.lower() != local_crc:
        raise _fail(ser, f"CRC mismatch: device says {device_crc}, bytes hash to {local_crc}",
                    device_crc32=device_crc, host_crc32=local_crc, bytes=len(data))
    Path(local).write_bytes(bytes(data))
    elapsed = max(clock() - t0, 1e-6)
    try:
        ser.close()
    except Exception:
        pass
    return {"bytes": len(data), "crc32": local_crc, "remote": remote,
            "local": str(local), "seconds": round(elapsed, 3)}
