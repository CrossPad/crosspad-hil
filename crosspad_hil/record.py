"""Serial transcripts: record a real session to ndjson, replay it without hardware.

Format (one JSON object per line):
    {"t": 0.0,  "dir": "meta", "version": 1, "port": "/dev/ttyACM0", "baudrate": 115200}
    {"t": 0.01, "dir": "ctl", "attr": "dtr", "value": false}
    {"t": 0.02, "dir": "tx",  "data": "4b49545f5354415455530a"}          # KIT_STATUS\n
    {"t": 0.03, "dir": "rx",  "data": "4b4954535441545553..."}           # KITSTATUS: …\r\n
`t` is seconds since the recorder was created. Empty reads (timeouts) are not recorded.

`crosspad-hil record --out FILE` wraps the device port in RecordingSerial; tests then
build a CdcLink/Console with `serial_factory=replay_factory(FILE)`.
"""
from __future__ import annotations

import json
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

from crosspad_hil.serial_open import open_serial

TRANSCRIPT_VERSION = 1


class ReplayMismatch(AssertionError):
    """The client wrote something other than what the transcript recorded next."""


class RecordingSerial:
    """Wrap a serial-like object and log every write/read/control change to ndjson."""

    def __init__(self, inner: Any, path: Path) -> None:
        self._inner = inner
        self._path = Path(path)
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._file = self._path.open("a", encoding="utf-8", buffering=1)
        self._t0 = time.monotonic()
        self._emit({
            "dir": "meta",
            "version": TRANSCRIPT_VERSION,
            "port": getattr(inner, "port", None),
            "baudrate": getattr(inner, "baudrate", None),
        })

    def _emit(self, entry: dict[str, Any]) -> None:
        row = {"t": round(time.monotonic() - self._t0, 6)}
        row.update(entry)
        self._file.write(json.dumps(row) + "\n")

    # ── data ──
    def write(self, data: bytes) -> int:
        self._emit({"dir": "tx", "data": bytes(data).hex()})
        return self._inner.write(data)

    def read(self, size: int = 1) -> bytes:
        data = self._inner.read(size)
        if data:
            self._emit({"dir": "rx", "data": bytes(data).hex()})
        return data

    def readline(self) -> bytes:
        data = self._inner.readline()
        if data:
            self._emit({"dir": "rx", "data": bytes(data).hex()})
        return data

    # ── control lines ──
    @property
    def dtr(self) -> bool:
        return bool(getattr(self._inner, "dtr", False))

    @dtr.setter
    def dtr(self, value: bool) -> None:
        self._emit({"dir": "ctl", "attr": "dtr", "value": bool(value)})
        self._inner.dtr = value

    @property
    def rts(self) -> bool:
        return bool(getattr(self._inner, "rts", False))

    @rts.setter
    def rts(self, value: bool) -> None:
        self._emit({"dir": "ctl", "attr": "rts", "value": bool(value)})
        self._inner.rts = value

    # ── lifecycle ──
    def close(self) -> None:
        try:
            self._inner.close()
        finally:
            if not self._file.closed:
                self._file.close()

    def __getattr__(self, name: str) -> Any:
        # Everything else (timeout, port, baudrate, is_open, in_waiting, reset_input_buffer,
        # flush, open, …) is the wrapped object's.
        return getattr(self._inner, name)

    def __setattr__(self, name: str, value: Any) -> None:
        if name.startswith("_") or name in ("dtr", "rts"):
            object.__setattr__(self, name, value)
        else:
            setattr(self._inner, name, value)


class ReplaySerial:
    """A serial-like object driven by a transcript written by RecordingSerial."""

    def __init__(self, path: Path, *, speed: float = 0.0, lenient: bool = False) -> None:
        self._path = Path(path)
        self.speed = speed
        self.lenient = lenient
        self._entries: list[dict[str, Any]] = []
        self.port: str | None = None
        self.baudrate: int | None = None
        for line in self._path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            entry = json.loads(line)
            if entry.get("dir") == "meta":
                self.port = entry.get("port")
                self.baudrate = entry.get("baudrate")
                continue
            if entry.get("dir") in ("tx", "rx", "ctl"):
                self._entries.append(entry)
        self._cursor = 0
        self._pending = bytearray()
        self._last_t: float | None = None
        self.timeout: float = 0.2
        self.is_open = True
        self.written: list[bytes] = []
        self.control_history: list[tuple[str, bool]] = []
        self._dtr = False
        self._rts = False

    # ── transcript walking ──
    def _next_data_index(self) -> int | None:
        """Index of the next tx/rx entry at or after the cursor (ctl entries are skipped)."""
        i = self._cursor
        while i < len(self._entries):
            if self._entries[i]["dir"] in ("tx", "rx"):
                return i
            i += 1
        return None

    def _sleep_for(self, entry: dict[str, Any]) -> None:
        t = float(entry.get("t", 0.0))
        if self._last_t is not None and self.speed > 0.0:
            delay = (t - self._last_t) * self.speed
            if delay > 0:
                time.sleep(delay)
        self._last_t = t

    def _pull_rx(self) -> bool:
        """Move one rx entry into the pending buffer. False if the next data entry is a tx/end."""
        i = self._next_data_index()
        if i is None or self._entries[i]["dir"] != "rx":
            return False
        entry = self._entries[i]
        self._sleep_for(entry)
        self._pending += bytes.fromhex(entry["data"])
        self._cursor = i + 1
        return True

    # ── data ──
    def write(self, data: bytes) -> int:
        data = bytes(data)
        self.written.append(data)
        # A device may have replied before the host got round to reading: keep those rx
        # entries rather than skipping them.
        while True:
            i = self._next_data_index()
            if i is None:
                expected: bytes | None = None
                break
            if self._entries[i]["dir"] == "rx":
                self._pending += bytes.fromhex(self._entries[i]["data"])
                self._cursor = i + 1
                continue
            expected = bytes.fromhex(self._entries[i]["data"])
            self._last_t = float(self._entries[i].get("t", 0.0))
            self._cursor = i + 1
            break
        if expected != data and not self.lenient:
            raise ReplayMismatch(
                f"transcript {self._path.name} entry {self._cursor - 1}: "
                f"expected tx {expected!r}, got {data!r}"
            )
        return len(data)

    def read(self, size: int = 1) -> bytes:
        if not self._pending and not self._pull_rx():
            time.sleep(min(self.timeout, 0.01))
            return b""
        out = bytes(self._pending[:size])
        del self._pending[:size]
        return out

    def readline(self) -> bytes:
        while b"\n" not in self._pending:
            if not self._pull_rx():
                break
        if not self._pending:
            time.sleep(min(self.timeout, 0.01))
            return b""
        nl = self._pending.find(b"\n")
        end = len(self._pending) if nl < 0 else nl + 1
        out = bytes(self._pending[:end])
        del self._pending[:end]
        return out

    @property
    def in_waiting(self) -> int:
        return len(self._pending)

    def reset_input_buffer(self) -> None:
        self._pending.clear()

    def flush(self) -> None:
        return None

    # ── control lines ──
    @property
    def dtr(self) -> bool:
        return self._dtr

    @dtr.setter
    def dtr(self, value: bool) -> None:
        self._dtr = bool(value)
        self.control_history.append(("dtr", bool(value)))

    @property
    def rts(self) -> bool:
        return self._rts

    @rts.setter
    def rts(self, value: bool) -> None:
        self._rts = bool(value)
        self.control_history.append(("rts", bool(value)))

    # ── lifecycle / assertions ──
    def open(self) -> None:
        self.is_open = True

    def close(self) -> None:
        self.is_open = False

    def remaining(self) -> int:
        """Unconsumed tx/rx entries (ctl entries never count)."""
        return sum(1 for e in self._entries[self._cursor:] if e["dir"] in ("tx", "rx"))

    def assert_exhausted(self) -> None:
        left = self.remaining()
        if left or self._pending:
            nxt = self._next_data_index()
            preview = self._entries[nxt] if nxt is not None else None
            raise AssertionError(
                f"transcript {self._path.name}: {left} entries and {len(self._pending)} pending "
                f"bytes not consumed; next={preview!r}"
            )


def recording_factory(
    path: Path, base: Callable[..., Any] = open_serial
) -> Callable[..., RecordingSerial]:
    """A `serial_factory` that records: `recording_factory(out)(port, **kw)`."""

    def factory(port: str, **kw: Any) -> RecordingSerial:
        return RecordingSerial(base(port, **kw), path)

    return factory


def replay_factory(path: Path, **replay_kw: Any) -> Callable[..., ReplaySerial]:
    """A `serial_factory` that replays: `replay_factory(fixture, speed=0.0)(port, **kw)`."""

    def factory(port: str, **kw: Any) -> ReplaySerial:
        ser = ReplaySerial(path, **replay_kw)
        timeout = kw.get("timeout")
        if timeout is not None:
            ser.timeout = float(timeout)
        return ser

    return factory
