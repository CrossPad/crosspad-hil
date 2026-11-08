"""serial_open: DTR/RTS deasserted before open; reset_pulse always releases RTS."""
from __future__ import annotations

import sys

import pytest

from crosspad_hil import serial_open
from tests.fakes import FakeSerial

PULSE = [("dtr", False), ("rts", True), ("dtr", False), ("rts", False), ("dtr", False)]


class OpenableFake(FakeSerial):
    """FakeSerial plus the attribute-then-open() protocol pyserial uses."""

    def __init__(self) -> None:
        super().__init__([])
        self.port: str | None = None
        self.baudrate: int | None = None
        self.opened_with: dict | None = None

    def open(self) -> None:
        self.opened_with = {"dtr": self.dtr, "rts": self.rts, "port": self.port}
        self.control_history.append(("open", True))
        self.is_open = True


def test_open_serial_deasserts_dtr_rts_before_open() -> None:
    ser = serial_open.open_serial("/dev/ttyFAKE0", serial_cls=OpenableFake)
    assert ser.opened_with == {"dtr": False, "rts": False, "port": "/dev/ttyFAKE0"}
    # both control lines were written BEFORE open(), nothing after
    hist = ser.control_history
    open_idx = hist.index(("open", True))
    assert ("dtr", False) in hist[:open_idx]
    assert ("rts", False) in hist[:open_idx]
    assert hist[open_idx + 1:] == []


def test_open_serial_applies_baud_and_timeout() -> None:
    ser = serial_open.open_serial("COM7", baud=921600, timeout=1.5, serial_cls=OpenableFake)
    assert ser.baudrate == 921600
    assert ser.timeout == 1.5
    assert ser.is_open


def test_open_serial_reset_pulses_after_open(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(serial_open.time, "sleep", lambda s: None)
    ser = serial_open.open_serial("/dev/ttyFAKE0", reset=True, serial_cls=OpenableFake)
    hist = ser.control_history
    open_idx = hist.index(("open", True))
    # DTR low, RTS high, 100 ms, RTS low -- each RTS write followed by a DTR write
    assert hist[open_idx + 1:] == PULSE


def test_reset_pulse_sequence_and_timing(monkeypatch: pytest.MonkeyPatch) -> None:
    slept: list[float] = []
    monkeypatch.setattr(serial_open.time, "sleep", slept.append)
    ser = FakeSerial([])
    ser.control_history.clear()
    serial_open.reset_pulse(ser)
    assert ser.control_history == PULSE
    assert slept == [0.1]
    assert ser.rts is False


def test_reset_pulse_releases_rts_on_exception(monkeypatch: pytest.MonkeyPatch) -> None:
    def boom(_s: float) -> None:
        raise RuntimeError("interrupted mid-pulse")

    monkeypatch.setattr(serial_open.time, "sleep", boom)
    ser = FakeSerial([])
    ser.control_history.clear()
    with pytest.raises(RuntimeError):
        serial_open.reset_pulse(ser)
    assert ser.control_history[-2:] == [("rts", False), ("dtr", False)]
    assert ser.rts is False


def test_reset_pulse_rewrites_dtr_after_every_rts_write(monkeypatch: pytest.MonkeyPatch) -> None:
    """Windows' usbser.sys sends SET_CONTROL_LINE_STATE only when DTR is written.

    An RTS write alone never reaches the STM bridge, so the ESP was never reset
    on Windows while esptool, which rewrites DTR after RTS, reset it fine.
    """
    monkeypatch.setattr(serial_open.time, "sleep", lambda s: None)
    ser = FakeSerial([])
    ser.control_history.clear()
    serial_open.reset_pulse(ser)
    hist = ser.control_history
    for i, (name, _value) in enumerate(hist):
        if name == "rts":
            assert hist[i + 1] == ("dtr", False)


def test_raw_tty_is_the_default_on_linux_with_a_way_back(monkeypatch) -> None:
    """Not opening the board's console with a reset is the behaviour people want.

    The escape hatch stays because this is a hand-rolled tty layer on one
    platform: a bench that hits a corner of it needs a way back that is not a
    code change.
    """
    import sys as _sys

    from crosspad_hil import serial_open as so

    monkeypatch.delenv("CROSSPAD_HIL_RAW_TTY", raising=False)
    assert so._raw_tty_enabled() is _sys.platform.startswith("linux")
    for on in ("1", "true", "YES", "on"):
        monkeypatch.setenv("CROSSPAD_HIL_RAW_TTY", on)
        assert so._raw_tty_enabled() is True
    for off in ("0", "false", "NO", "off"):
        monkeypatch.setenv("CROSSPAD_HIL_RAW_TTY", off)
        assert so._raw_tty_enabled() is False


def test_an_injected_serial_class_is_never_replaced_by_the_raw_path(monkeypatch) -> None:
    """Tests inject FakeSerial; the raw path must not steal that."""
    from crosspad_hil import serial_open as so
    from tests.fakes import FakeSerial

    monkeypatch.setenv("CROSSPAD_HIL_RAW_TTY", "1")
    ser = so.open_serial("/dev/null", serial_cls=FakeSerial)
    assert isinstance(ser, FakeSerial)


@pytest.mark.skipif(not sys.platform.startswith("linux"),
                     reason="RawTty and pty are Linux-only")
def test_raw_tty_reads_and_writes_through_a_pty(monkeypatch) -> None:
    """Exercise the raw handle against a real tty, without needing a board."""
    import os
    import pty

    from crosspad_hil.serial_open import RawTty

    primary, secondary = pty.openpty()
    try:
        tty = RawTty(os.ttyname(secondary), 115200, 0.5)
        try:
            os.write(primary, b"hello\n")
            assert tty.read(6) == b"hello\n"
            assert tty.write(b"pong") == 4
            assert os.read(primary, 4) == b"pong"
            os.write(primary, b"line one\nrest")
            assert tty.readline() == b"line one\n"
            assert tty.in_waiting == 4
            tty.flush()                        # nothing pending; must not raise
            tty.timeout = 0.05
            assert tty.read(4) == b"rest"
            assert tty.read(4) == b""          # a timeout is empty, never a hang
            assert tty.readline() == b""
        finally:
            tty.close()
    finally:
        os.close(primary)
        os.close(secondary)
