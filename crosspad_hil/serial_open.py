"""Opening serial ports without resetting or wedging the board.

Two traps, both measured on hardware:

* pyserial asserts DTR and RTS on ``open()``. The STM32 bridge emulates
  esptool's auto-reset, so an STM VCP opened the default way REBOOTS THE ESP
  (hil_stability.py, hil_kit_churn.py). The ESP's own native-USB CDC on
  Windows wedges when the host asserts DTR/RTS on open (hil_kit_churn.py Cdc).
  Both lines are therefore driven low BEFORE the port is opened, which is only
  possible with the attribute-then-open() form of the pyserial API.

  That is necessary and not sufficient. On Linux the kernel raises both lines
  when a tty is opened and nobody else holds it, before pyserial can lower
  them, and the pulse is long enough for the bridge: every console.log from a
  ``reset=False`` open still began with ``rst:0x1 (POWERON)``, and the kit
  loaded before it was gone. So a guard fd is opened first, the lines are
  cleared through it with an ioctl, HUPCL is cleared so closing does not drop
  DTR either, and only then is the pyserial handle opened — on a tty that is
  already open, so the kernel does not raise anything. Measured: the board
  keeps its loaded kit across the open.
* A reset is an explicit, deliberate pulse (hil_smoke.py reset_pulse), and the
  pulse always releases RTS again even if interrupted — a board held in reset
  by an exception is a board nobody can talk to.
"""
from __future__ import annotations

import array
import errno
import os
import select
import struct
import sys
import time
from typing import Any

import serial

if sys.platform.startswith("linux"):
    import fcntl
    import termios

_TIOCMBIC = 0x5417
_TIOCMBIS = 0x5416
_TIOCM_DTR = 0x002
_TIOCM_RTS = 0x004


class RawTty:
    """A serial handle that never lets the modem lines rise.

    pyserial is what reboots the board, not the file descriptor: holding the
    port open with DTR/RTS low and *then* letting pyserial open it resets the
    ESP anyway, while a raw descriptor doing the same reads the console with
    the loaded kit still loaded afterwards. Both measured on a rev2 board.

    So on Linux the console and CDC handles are raw descriptors. The surface is
    only what `Console`, `CdcLink` and `reset_pulse` use — read, write, close,
    reset_input_buffer, timeout, and the two lines — because a reset has to
    stay possible when it is what the caller asked for.
    """

    def __init__(self, path: str, baud: int, timeout: float) -> None:
        self.port = path
        self.baudrate = baud
        self.timeout = timeout
        self.is_open = False
        self.fd = os.open(path, os.O_RDWR | os.O_NOCTTY | os.O_NONBLOCK)
        try:
            self._lines(False, False)
            attrs = termios.tcgetattr(self.fd)
            iflag, oflag, cflag, lflag, _ispeed, _ospeed, cc = attrs
            # Raw: no translation, no echo, no flow control, no controlling tty,
            # and no HUPCL — dropping DTR on close is another reset.
            # PARMRK is on the list because the tty keeps it across reopen and
            # Chromium's Web Serial sets it: with PARMRK and no ISTRIP the line
            # discipline doubles every legitimate 0xFF byte, and a binary pull
            # comes back 60-odd bytes long with a garbage trailer.
            iflag &= ~(termios.IXON | termios.IXOFF | termios.IXANY | termios.ICRNL
                       | termios.INLCR | termios.IGNCR | termios.BRKINT | termios.ISTRIP
                       | termios.PARMRK | termios.INPCK)
            oflag &= ~termios.OPOST
            lflag &= ~(termios.ECHO | termios.ECHONL | termios.ICANON
                       | termios.ISIG | termios.IEXTEN)
            cflag &= ~(termios.CSIZE | termios.PARENB | termios.CSTOPB
                       | termios.CRTSCTS | termios.HUPCL)
            cflag |= termios.CS8 | termios.CLOCAL | termios.CREAD
            cc = list(cc)
            cc[termios.VMIN] = 0
            cc[termios.VTIME] = 0
            speed = getattr(termios, f"B{baud}")
            termios.tcsetattr(self.fd, termios.TCSANOW,
                              [iflag, oflag, cflag, lflag, speed, speed, cc])
            self._lines(False, False)
            self.is_open = True
        except BaseException:
            os.close(self.fd)
            raise

    def _lines(self, dtr: bool, rts: bool) -> None:
        bits = (_TIOCM_DTR if dtr else 0) | (_TIOCM_RTS if rts else 0)
        clear = (_TIOCM_DTR | _TIOCM_RTS) & ~bits
        try:
            if clear:
                fcntl.ioctl(self.fd, _TIOCMBIC, struct.pack("I", clear))
            if bits:
                fcntl.ioctl(self.fd, _TIOCMBIS, struct.pack("I", bits))
        except OSError as exc:
            # A pty has no modem lines, and neither does every USB bridge. There
            # is then nothing that could pulse the board's reset, so this is a
            # port without the trap rather than a port that failed to open.
            if exc.errno not in (errno.ENOTTY, errno.EINVAL):
                raise
        self._dtr, self._rts = dtr, rts

    @property
    def dtr(self) -> bool:
        return self._dtr

    @dtr.setter
    def dtr(self, value: bool) -> None:
        self._lines(bool(value), self._rts)

    @property
    def rts(self) -> bool:
        return self._rts

    @rts.setter
    def rts(self, value: bool) -> None:
        self._lines(self._dtr, bool(value))

    def read(self, size: int = 1) -> bytes:
        """Up to `size` bytes, giving up after `timeout` seconds like pyserial."""
        deadline = time.monotonic() + (self.timeout or 0.0)
        out = b""
        while len(out) < size:
            remaining = deadline - time.monotonic()
            if remaining <= 0 and out:
                break
            ready, _, _ = select.select([self.fd], [], [], max(remaining, 0.0))
            if not ready:
                break
            try:
                chunk = os.read(self.fd, size - len(out))
            except BlockingIOError:
                continue
            except OSError as exc:
                raise serial.SerialException(f"{self.port}: {exc}") from exc
            if not chunk:
                break
            out += chunk
        return out

    def readline(self) -> bytes:
        """Up to and including the next newline, or what arrived before timeout.

        ota.py reads the bootloader's replies a line at a time, so this is not
        optional decoration: without it the OTA path cannot use a raw handle.
        """
        deadline = time.monotonic() + (self.timeout or 0.0)
        out = bytearray()
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                break
            ready, _, _ = select.select([self.fd], [], [], max(remaining, 0.0))
            if not ready:
                break
            try:
                chunk = os.read(self.fd, 1)
            except BlockingIOError:
                continue
            except OSError as exc:
                raise serial.SerialException(f"{self.port}: {exc}") from exc
            if not chunk:
                break
            out += chunk
            if chunk == b"\n":
                break
        return bytes(out)

    @property
    def in_waiting(self) -> int:
        buf = array.array("i", [0])
        try:
            fcntl.ioctl(self.fd, termios.FIONREAD, buf)
        except OSError:
            return 0
        return int(buf[0])

    def flush(self) -> None:
        """Block until what was written has actually gone out."""
        try:
            termios.tcdrain(self.fd)
        except termios.error:
            pass

    def write(self, data: bytes) -> int:
        written = 0
        while written < len(data):
            _, ready, _ = select.select([], [self.fd], [], self.timeout or 1.0)
            if not ready:
                break
            written += os.write(self.fd, data[written:])
        return written

    def reset_input_buffer(self) -> None:
        termios.tcflush(self.fd, termios.TCIFLUSH)

    def close(self) -> None:
        if self.is_open:
            self.is_open = False
            # A tty keeps its termios across close and reopen. VMIN=0 left
            # behind makes poll() always report the port readable and read()
            # return 0 bytes, which Chromium's Web Serial takes for a lost
            # device — the browser companion then cannot talk to a board hil
            # has touched. Hand the port back the way the kernel created it.
            try:
                attrs = termios.tcgetattr(self.fd)
                cc = list(attrs[6])
                cc[termios.VMIN] = 1
                cc[termios.VTIME] = 0
                attrs[6] = cc
                termios.tcsetattr(self.fd, termios.TCSANOW, attrs)
            except (OSError, termios.error):
                pass
            os.close(self.fd)


def _raw_tty_enabled() -> bool:
    """Default on Linux. ``CROSSPAD_HIL_RAW_TTY=0`` goes back to pyserial.

    Verified on a rev2 board: ``console --expect`` through the raw handle left
    kit 8 current and produced no USB re-enumeration at all, where the same
    open through pyserial rebooted the ESP and lost the kit. The escape hatch
    stays because this is a hand-rolled tty layer on one platform, and a bench
    that hits a corner of it needs a way back that is not a code change.
    """
    flag = os.environ.get("CROSSPAD_HIL_RAW_TTY", "").strip().lower()
    if flag in {"0", "false", "no", "off"}:
        return False
    if flag in {"1", "true", "yes", "on"}:
        return True
    return sys.platform.startswith("linux")


def _quiet_guard_fd(path: str) -> int | None:
    """Hold `path` open with DTR/RTS low, so the next open raises nothing.

    Returns the fd to close once the real handle is open, or None where this
    cannot be done (any platform but Linux, a port that is not a tty, a device
    that vanished between discovery and here). Failure is never fatal: the
    caller then gets exactly the behaviour it had before, which is a reset it
    can see rather than an error it cannot act on.
    """
    if not sys.platform.startswith("linux"):
        return None
    try:
        fd = os.open(path, os.O_RDWR | os.O_NOCTTY | os.O_NONBLOCK)
    except OSError:
        return None
    try:
        fcntl.ioctl(fd, _TIOCMBIC, struct.pack("I", _TIOCM_DTR | _TIOCM_RTS))
        attrs = termios.tcgetattr(fd)
        attrs[2] &= ~termios.HUPCL
        termios.tcsetattr(fd, termios.TCSANOW, attrs)
    except (OSError, termios.error):
        os.close(fd)
        return None
    return fd


def _set_rts(ser: Any, value: bool) -> None:
    """Write RTS, then DTR again with its own value, as esptool's ``_setRTS`` does.

    Windows' usbser.sys sends SET_CONTROL_LINE_STATE only when DTR is written:
    an RTS write alone never leaves the host, so the STM bridge never saw the
    pulse and the ESP was not reset on Windows, while esptool reset it fine.
    """
    ser.rts = value
    ser.dtr = ser.dtr


def reset_pulse(ser: Any) -> None:
    """esptool-style auto-reset: EN low via RTS, IO0 high via DTR.

    Ported from hil_smoke.py ``reset_pulse``; the ``finally`` is the hygiene
    invariant (spec §2.4): RTS is released no matter what.
    """
    ser.dtr = False
    _set_rts(ser, True)
    try:
        time.sleep(0.1)
    finally:
        _set_rts(ser, False)


def open_serial(
    path: str,
    *,
    baud: int = 115200,
    timeout: float = 0.2,
    reset: bool = False,
    serial_cls: type = serial.Serial,
) -> serial.Serial:
    """Open ``path`` with DTR/RTS deasserted before the port is opened.

    ``serial_cls`` is injectable so tests run on ``tests.fakes.FakeSerial``.
    With ``reset=True`` the board is reset with ``reset_pulse`` once the port
    is open (the caller wants the boot banner from the very first line).
    """
    # from hil_kit_churn.py Cdc.__init__ / hil_stability.py open_port
    ser = serial_cls()
    ser.port = path
    ser.baudrate = baud
    ser.timeout = timeout
    ser.dtr = False
    ser.rts = False
    if _raw_tty_enabled() and serial_cls is serial.Serial and sys.platform.startswith("linux"):
        raw = RawTty(path, baud, timeout)
        if reset:
            reset_pulse(raw)
        return raw  # type: ignore[return-value]
    # Skipped when the caller wants a reset anyway — there is nothing to protect.
    guard = None if reset else _quiet_guard_fd(path)
    try:
        ser.open()
    finally:
        if guard is not None:
            os.close(guard)
    if reset:
        reset_pulse(ser)
    return ser
