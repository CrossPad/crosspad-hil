"""BLE MIDI from this PC's own radio — scan, connect, send, listen, latency, host peripheral.

Ported from ``platform-idf/tools/hil_ble_midi.py``. Everything the device does over BLE is
already reachable through the ``BLE_*`` CDC verbs (:mod:`crosspad_hil.verbs`); this module is
the *other* half of the link — the host radio that makes each leg falsifiable. A note that
never sounded can then be told apart from one that never left the radio, because the device's
counters and the host's notifications are read independently.

Four things here are not obvious and are the reason this is more than a bleak snippet.

* **The service UUID is not an identity.** ``03B80E5A-…`` is shared by every BLE MIDI
  peripheral in range, and the name ``CrossPad`` by every CrossPad on the bench. A run that
  took "the first BLE MIDI advertiser" once attached to a passing speaker and reported six
  confident failures about a device that was never under test. :func:`select_peripheral`
  matches on the address the board itself reports as ``self=`` in ``BLE_STATUS``, and refuses
  rather than guesses when it cannot.
* **BLE MIDI packets carry their own timestamps.** One packet is a header byte plus
  ``(timestamp, message)`` pairs, and a host that has several events inside one connection
  interval packs them together — so :func:`decode` must handle running status and multiple
  messages, and :func:`encode` must be able to produce them.
* **Latency is measured on the path with nothing else in it.** ``BLE_SEND`` hands a note
  straight to the transport, so what is timed is the CDC write plus the radio — not the pad
  scan or the sample engine. Nothing may block while waiting: a synchronous CDC read between
  the write and the notification would be counted as radio latency.
* **bleak is asyncio, the rest of this package is not.** :class:`BleMidiLink` owns one event
  loop on a background thread and marshals every call onto it, so callers (the CLI, the
  daemon's thread pool, a scenario) stay synchronous. The loop is stopped by ``close()``.

The host-peripheral role (:class:`HostPeripheral`) is Windows-only: WinRT's GATT server is the
one API that will advertise a service and accept a connection without a second radio, which is
what the board's Host (central) role needs to have something to find. Elsewhere it raises
``HilError(NOT_SUPPORTED)`` rather than pretending.
"""
from __future__ import annotations

import asyncio
import concurrent.futures
import statistics
import sys
import threading
import time
from collections import deque
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass, field
from typing import Any

from .cdc import CdcLink
from .errors import AMBIGUOUS_DEVICE, BAD_ARGS, ENV, NO_DEVICE, NOT_SUPPORTED, TIMEOUT, HilError

# BLE MIDI 1.0 (MMA/AMEI RP-052). The same two UUIDs the firmware registers in
# components/crosspad-platform-idf/EspBleMidi.cpp.
MIDI_SERVICE_UUID = "03b80e5a-ede8-4b33-a751-6ce34ec4c700"
MIDI_CHAR_UUID = "7772e5db-3868-4112-a1a9-f2669d106bf3"

DEFAULT_SCAN_TIMEOUT_S = 10.0
DEFAULT_CONNECT_TIMEOUT_S = 20.0
DEFAULT_NAME = "CrossPad"
RX_RING = 4096

# Data-byte count per MIDI status nibble; 0xF (system) is not carried over BLE MIDI in
# anything the CrossPad sends, and an unknown status is skipped rather than guessed.
DATA_BYTES: dict[int, int] = {0x8: 2, 0x9: 2, 0xA: 2, 0xB: 2, 0xC: 1, 0xD: 1, 0xE: 2}


# --------------------------------------------------------------------------- packet codec

def encode(messages: Sequence[Sequence[int]], ts_ms: int | None = None) -> bytes:
    """MIDI messages -> one BLE MIDI packet (header byte + timestamped messages).

    The 13-bit timestamp is split 6 high bits into the header and 7 low bits into each
    message's timestamp byte, which is what the firmware's tx task produces when it coalesces
    a connection interval's worth of events into one packet.
    """
    if not messages:
        raise HilError(BAD_ARGS, "a BLE MIDI packet needs at least one message")
    if ts_ms is None:
        ts_ms = int(time.monotonic() * 1000)
    ts = int(ts_ms) & 0x1FFF
    out = bytearray([0x80 | ((ts >> 7) & 0x3F)])
    for m in messages:
        data = bytes(m)
        if not data or not data[0] & 0x80:
            raise HilError(BAD_ARGS, f"message {list(m)} does not start with a status byte")
        out.append(0x80 | (ts & 0x7F))
        out.extend(data)
    return bytes(out)


def decode(data: bytes) -> list[bytes]:
    """One BLE MIDI packet -> the MIDI messages in it (running status handled)."""
    msgs: list[bytes] = []
    i, status, buf = 1, 0, []
    while i < len(data):
        b = data[i]
        if b & 0x80:                      # timestamp byte; a status byte may follow
            i += 1
            if i >= len(data):
                break
            s = data[i]
            if s & 0x80:
                status, buf, i = s, [], i + 1
                continue
            b = s
        i += 1
        if not status:
            continue
        buf.append(b)
        if len(buf) >= DATA_BYTES.get(status >> 4, 0):
            msgs.append(bytes([status, *buf]))
            buf = []
    return msgs


def note_message(on: bool, note: int, vel: int = 100, channel: int = 0) -> list[int]:
    for name, value, hi in (("note", note, 127), ("vel", vel, 127), ("channel", channel, 15)):
        if not isinstance(value, int) or isinstance(value, bool) or not 0 <= value <= hi:
            raise HilError(BAD_ARGS, f"{name} must be 0..{hi}, got {value!r}")
    return [(0x90 if on else 0x80) | channel, note, vel if on else 0]


def describe(msg: bytes) -> dict[str, Any]:
    """A decoded message as fields, so a report does not make the reader parse hex."""
    kind = {0x8: "note_off", 0x9: "note_on", 0xA: "aftertouch", 0xB: "cc",
            0xC: "program_change", 0xD: "channel_pressure", 0xE: "pitch_bend"}
    out: dict[str, Any] = {"bytes": msg.hex(), "kind": kind.get(msg[0] >> 4, "unknown"),
                           "channel": msg[0] & 0x0F}
    if len(msg) >= 2:
        out["data1"] = msg[1]
    if len(msg) >= 3:
        out["data2"] = msg[2]
    if out["kind"] in ("note_on", "note_off") and len(msg) >= 3:
        out["note"], out["velocity"] = msg[1], msg[2]
        if out["kind"] == "note_on" and msg[2] == 0:
            out["kind"] = "note_off"      # running-status note-off, as a real host sends it
    return out


# --------------------------------------------------------------------------- bleak plumbing

def require_bleak() -> Any:
    """Import bleak or say what to install — the BLE extra is optional on purpose."""
    try:
        import bleak
    except ImportError as e:
        raise HilError(ENV, "bleak is not installed, so this PC's radio cannot be used",
                       hint="pip install 'crosspad-hil[ble]' (the device-side BLE_* verbs "
                            "work without it)") from e
    return bleak


class _Runner:
    """One asyncio loop on a background thread, so callers stay synchronous."""

    def __init__(self) -> None:
        self._loop: asyncio.AbstractEventLoop | None = None
        self._thread: threading.Thread | None = None

    def start(self) -> asyncio.AbstractEventLoop:
        if self._loop is not None:
            return self._loop
        loop = asyncio.new_event_loop()
        ready = threading.Event()

        def body() -> None:
            asyncio.set_event_loop(loop)
            loop.call_soon(ready.set)
            loop.run_forever()

        self._thread = threading.Thread(target=body, name="ble-loop", daemon=True)
        self._thread.start()
        ready.wait(5.0)
        self._loop = loop
        return loop

    @property
    def loop(self) -> asyncio.AbstractEventLoop | None:
        return self._loop

    def call(self, coro: Any, timeout_s: float, what: str) -> Any:
        loop = self.start()
        fut = asyncio.run_coroutine_threadsafe(coro, loop)
        try:
            return fut.result(timeout_s)
        except (TimeoutError, concurrent.futures.TimeoutError) as e:
            fut.cancel()
            raise HilError(TIMEOUT, f"{what} did not finish in {timeout_s:.1f}s",
                           hint="is the board advertising? BLE_STATUS says state= and self=",
                           operation=what) from e

    def stop(self) -> None:
        loop, thread = self._loop, self._thread
        self._loop, self._thread = None, None
        if loop is None:
            return
        loop.call_soon_threadsafe(loop.stop)
        if thread is not None:
            thread.join(5.0)
        loop.close()


# --------------------------------------------------------------------------- scanning

@dataclass
class BlePeripheral:
    """One BLE MIDI advertiser this radio saw."""

    address: str
    name: str | None = None
    rssi: int | None = None
    service_uuids: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {"address": self.address, "name": self.name, "rssi": self.rssi,
                "service_uuids": list(self.service_uuids)}


async def _bleak_discover(timeout_s: float) -> list[BlePeripheral]:
    bleak = require_bleak()
    seen = await bleak.BleakScanner.discover(timeout=timeout_s, return_adv=True)
    out: list[BlePeripheral] = []
    for dev, adv in seen.values():
        uuids = [str(u).lower() for u in (adv.service_uuids or [])]
        out.append(BlePeripheral(address=dev.address,
                                 name=dev.name or getattr(adv, "local_name", None),
                                 rssi=getattr(adv, "rssi", None), service_uuids=uuids))
    return out


def scan(timeout_s: float = DEFAULT_SCAN_TIMEOUT_S, *, midi_only: bool = True,
         discover_fn: Callable[[float], Any] | None = None) -> list[BlePeripheral]:
    """Scan this PC's radio. ``discover_fn`` is an injectable coroutine function for tests."""
    if timeout_s <= 0:
        raise HilError(BAD_ARGS, f"scan timeout must be positive, got {timeout_s}")
    runner = _Runner()
    try:
        found = runner.call((discover_fn or _bleak_discover)(timeout_s),
                            timeout_s + 15.0, "BLE scan")
    finally:
        runner.stop()
    peripherals = [p if isinstance(p, BlePeripheral) else BlePeripheral(**p) for p in found]
    if midi_only:
        peripherals = [p for p in peripherals
                       if MIDI_SERVICE_UUID in [u.lower() for u in p.service_uuids]]
    peripherals.sort(key=lambda p: (-(p.rssi if p.rssi is not None else -999), p.address))
    return peripherals


def self_address(link: CdcLink) -> str | None:
    """The board's own BLE address, from ``BLE_STATUS self=``; None when it does not report one."""
    from . import verbs

    value = str(verbs.ble_status(link).get("self") or "").strip().upper()
    return value if value and value != "-" else None


def select_peripheral(peripherals: Sequence[BlePeripheral], *, address: str | None = None,
                      name: str = DEFAULT_NAME) -> BlePeripheral:
    """Pick the board out of what the radio saw — by address, never by "the first MIDI thing".

    Falls back to the advertised name only for firmware predating ``self=`` in ``BLE_STATUS``,
    and refuses when more than one device answers to it: two CrossPads on a bench is normal,
    and testing the wrong one produces confident nonsense.
    """
    listing = ", ".join(f"{p.address} {p.name!r}" for p in peripherals) or "none"
    if address:
        want = address.upper()
        hit = [p for p in peripherals if p.address.upper() == want]
        if hit:
            return hit[0]
        raise HilError(NO_DEVICE, f"the board says its BLE address is {want}, "
                                  f"but it is not among the BLE MIDI advertisers [{listing}]",
                       hint="is BLE running (BLE_START 1) and in Server mode? "
                            "scan again with a longer timeout",
                       address=want, seen=[p.to_dict() for p in peripherals])
    named = [p for p in peripherals if (p.name or "") == name]
    if len(named) == 1:
        return named[0]
    if len(named) > 1:
        raise HilError(AMBIGUOUS_DEVICE,
                       f"{len(named)} devices are advertising as {name!r} [{listing}]",
                       hint="pass address=… , or flash firmware that reports self= in "
                            "BLE_STATUS so the board names itself",
                       candidates=[p.to_dict() for p in named])
    raise HilError(NO_DEVICE, f"no BLE MIDI peripheral named {name!r}; saw [{listing}]",
                   hint="BLE_START 1 puts the board in Server (peripheral) mode",
                   seen=[p.to_dict() for p in peripherals])


# --------------------------------------------------------------------------- the link

@dataclass
class RxMessage:
    """A message the host radio received, with the moment it arrived."""

    t: float
    msg: bytes

    def to_dict(self) -> dict[str, Any]:
        return {"t": round(self.t, 6), **describe(self.msg)}


def _bleak_client_factory(address: str, timeout_s: float) -> Any:
    bleak = require_bleak()
    return bleak.BleakClient(address, timeout=timeout_s)


class BleMidiLink:
    """A connected BLE MIDI peripheral, driven synchronously.

    The client object is injectable so tests need no radio: anything with the awaitable
    ``connect``/``disconnect``/``start_notify``/``write_gatt_char`` surface bleak exposes will
    do. Received messages go into a bounded ring with the ``perf_counter`` moment they
    arrived, which is what makes the latency number meaningful.
    """

    def __init__(self, address: str, *, name: str | None = None,
                 timeout_s: float = DEFAULT_CONNECT_TIMEOUT_S,
                 client_factory: Callable[[str, float], Any] = _bleak_client_factory,
                 clock: Callable[[], float] = time.perf_counter) -> None:
        self.address = address
        self.name = name
        self.properties: list[str] = []
        self._timeout_s = float(timeout_s)
        self._client_factory = client_factory
        self._clock = clock
        self._runner = _Runner()
        self._client: Any = None
        self._lock = threading.Lock()
        self._rx: deque[RxMessage] = deque(maxlen=RX_RING)
        self._arrived = threading.Event()
        self._sent_packets = 0
        self._sent_messages = 0
        self._connected = False

    # -- lifecycle -----------------------------------------------------
    def connect(self) -> dict[str, Any]:
        if self._connected:
            return self.status()
        client = self._client_factory(self.address, self._timeout_s)

        async def go() -> list[str]:
            await client.connect()
            props: list[str] = []
            services = getattr(client, "services", None)
            getter = getattr(services, "get_characteristic", None) if services else None
            if getter is not None:
                ch = getter(MIDI_CHAR_UUID)
                props = [str(p) for p in getattr(ch, "properties", [])] if ch else []
            await client.start_notify(MIDI_CHAR_UUID, self._on_notify)
            return props

        try:
            self.properties = self._runner.call(go(), self._timeout_s + 10.0,
                                                f"connect to {self.address}")
        except BaseException:
            # Connected but not subscribed (or not answered in time): hang up, or
            # the host keeps holding the board's only link after the error.
            try:
                self._runner.call(client.disconnect(), 15.0, f"disconnect from {self.address}")
            except Exception:  # noqa: BLE001 - the connect error is the one to report
                pass
            self._runner.stop()
            raise
        self._client = client
        self._connected = True
        return self.status()

    def disconnect(self) -> dict[str, Any]:
        client, self._client = self._client, None
        self._connected = False
        if client is not None:
            async def go() -> None:
                try:
                    await client.stop_notify(MIDI_CHAR_UUID)
                except Exception:  # noqa: BLE001 - a peer that already dropped is not an error
                    pass
                await client.disconnect()

            try:
                self._runner.call(go(), 15.0, f"disconnect from {self.address}")
            finally:
                self._runner.stop()
        else:
            self._runner.stop()
        return {"address": self.address, "connected": False}

    close = disconnect

    @property
    def connected(self) -> bool:
        return self._connected

    # -- rx ------------------------------------------------------------
    def _on_notify(self, _sender: Any, data: bytearray) -> None:
        now = self._clock()
        with self._lock:
            for msg in decode(bytes(data)):
                self._rx.append(RxMessage(t=now, msg=msg))
        self._arrived.set()

    def received(self, *, clear: bool = False) -> list[dict[str, Any]]:
        with self._lock:
            out = [m.to_dict() for m in self._rx]
            if clear:
                self._rx.clear()
        return out

    def clear_rx(self) -> None:
        with self._lock:
            self._rx.clear()
        self._arrived.clear()

    def count(self) -> int:
        with self._lock:
            return len(self._rx)

    def wait_for(self, count: int = 1, timeout_s: float = 2.0) -> bool:
        """Block until ``count`` messages have arrived since the last :meth:`clear_rx`."""
        deadline = time.monotonic() + timeout_s
        while self.count() < count:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                return False
            self._arrived.wait(min(remaining, 0.05))
            self._arrived.clear()
        return True

    def listen(self, seconds: float, *, expect: int | None = None) -> dict[str, Any]:
        """Collect notifications for ``seconds`` (stopping early once ``expect`` arrived)."""
        self.clear_rx()
        if expect:
            self.wait_for(expect, seconds)
        else:
            time.sleep(max(0.0, seconds))
        msgs = self.received()
        return {"seconds": seconds, "count": len(msgs), "messages": msgs,
                "notes_on": sum(1 for m in msgs if m["kind"] == "note_on")}

    # -- tx ------------------------------------------------------------
    def send(self, messages: Sequence[Sequence[int]], *, ts_ms: int | None = None,
             timeout_s: float = 5.0) -> dict[str, Any]:
        """Write one packet, several messages at a time if asked — what a real host does."""
        if not self._connected or self._client is None:
            raise HilError(NO_DEVICE, f"not connected to {self.address}",
                           hint="connect() first")
        packet = encode(messages, ts_ms)
        self._runner.call(
            self._client.write_gatt_char(MIDI_CHAR_UUID, packet, response=False),
            timeout_s, "BLE write")
        with self._lock:
            self._sent_packets += 1
            self._sent_messages += len(messages)
        return {"packet": packet.hex(), "messages": len(messages), "bytes": len(packet)}

    def send_note(self, on: bool, note: int, vel: int = 100, channel: int = 0) -> dict[str, Any]:
        return self.send([note_message(on, note, vel, channel)])

    def send_hit(self, note: int, vel: int = 100, gate_s: float = 0.05) -> dict[str, Any]:
        """Note-on, wait the gate, note-off — one hit, as a player would send it."""
        self.send_note(True, note, vel)
        time.sleep(max(0.0, gate_s))
        return self.send_note(False, note)

    # -- reporting -----------------------------------------------------
    def status(self) -> dict[str, Any]:
        with self._lock:
            rx = len(self._rx)
            tx_packets, tx_messages = self._sent_packets, self._sent_messages
        return {"address": self.address, "name": self.name, "connected": self._connected,
                "properties": list(self.properties), "rx_messages": rx,
                "tx_packets": tx_packets, "tx_messages": tx_messages,
                "notify": "notify" in self.properties,
                "write_without_response": "write-without-response" in self.properties}


# --------------------------------------------------------------------------- BlueZ + ALSA
ALSA_PORT_WAIT_S = 8.0          # bluetoothd creates the sequencer client after ServicesResolved
ALSA_PORT_POLL_S = 0.25
BLUETOOTHCTL_TIMEOUT_S = 20.0


def _run_bluetoothctl(args: Sequence[str], timeout_s: float) -> str:
    import subprocess
    try:
        done = subprocess.run(["bluetoothctl", *args], capture_output=True, text=True,
                              timeout=timeout_s, env={**__import__("os").environ, "LC_ALL": "C"})
    except (OSError, subprocess.SubprocessError) as exc:
        raise HilError(ENV, f"bluetoothctl {' '.join(args)} failed to run: {exc}",
                       hint="install bluez, or use a host without its MIDI driver") from exc
    return (done.stdout or "") + (done.stderr or "")


def _import_rtmidi() -> Any:
    try:
        import rtmidi
    except ImportError as exc:
        raise HilError(NOT_SUPPORTED, "python-rtmidi is unavailable",
                       hint="pip install python-rtmidi") from exc
    return rtmidi


class AlsaBleMidiLink:
    """The board over BlueZ's own MIDI driver: an ALSA sequencer port, not GATT.

    bluetoothd (5.6x+ built with MIDI) claims the BLE MIDI service the moment a
    device carrying it connects, creates a sequencer client named after the
    device ("CrossPad" -> "CrossPad Bluetooth") and hides the service from D-Bus
    — so bleak sees only GATT 0x1801 while a DAW sees a MIDI port. This link is
    what that DAW sees: rtmidi on the port, bluetoothctl for the connection.
    Same surface as :class:`BleMidiLink`, so the scenario does not care which
    one it holds. BlueZ packs and timestamps the BLE packets itself, so ``ts_ms``
    is accepted and ignored.
    """

    def __init__(self, address: str, *, name: str | None = None,
                 timeout_s: float = DEFAULT_CONNECT_TIMEOUT_S,
                 run_cmd: Callable[[Sequence[str], float], str] = _run_bluetoothctl,
                 rtmidi_module: Any = None,
                 clock: Callable[[], float] = time.perf_counter,
                 forget: bool = False) -> None:
        self.address = address
        self.name = name or DEFAULT_NAME
        # forget=True: `bluetoothctl remove` the board on disconnect, but only
        # when it was not paired, bonded or trusted before this link connected.
        self._forget = forget
        self._remove_on_disconnect = False
        self.port: str | None = None
        # What the BlueZ driver needs of the characteristic, and what the
        # firmware declares (gatttool: props 0x16).
        self.properties: list[str] = ["read", "write-without-response", "notify"]
        self._timeout_s = float(timeout_s)
        self._run = run_cmd
        self._rtmidi = rtmidi_module
        self._clock = clock
        self._in: Any = None
        self._out: Any = None
        self._lock = threading.Lock()
        self._rx: deque[RxMessage] = deque(maxlen=RX_RING)
        self._arrived = threading.Event()
        self._sent_packets = 0
        self._sent_messages = 0
        self._connected = False

    # -- lifecycle -----------------------------------------------------
    def _port_index(self, ports: Sequence[str]) -> int | None:
        for i, p in enumerate(ports):
            if "Bluetooth" in p and self.name.lower() in p.lower():
                return i
        return None

    def _known_before(self) -> bool:
        """True unless ``bluetoothctl info`` shows no pairing, bond or trust.

        Anything it cannot read counts as known: the cleanup only ever removes
        an entry this host demonstrably had no relationship with.
        """
        try:
            info = self._run(["info", self.address], BLUETOOTHCTL_TIMEOUT_S)
        except HilError:
            return True
        if "not available" in info:
            return False
        return not ("Paired: no" in info and "Trusted: no" in info
                    and "Bonded: yes" not in info)

    def _drop_cache_entry(self) -> None:
        if not self._remove_on_disconnect:
            return
        self._remove_on_disconnect = False
        try:
            self._run(["remove", self.address], BLUETOOTHCTL_TIMEOUT_S)
        except HilError:
            pass

    def connect(self) -> dict[str, Any]:
        if self._connected:
            return self.status()
        rtmidi = self._rtmidi or _import_rtmidi()
        self._remove_on_disconnect = self._forget and not self._known_before()
        out = self._run(["connect", self.address], min(self._timeout_s, BLUETOOTHCTL_TIMEOUT_S))
        if "successful" not in out.lower():
            self._drop_cache_entry()
            raise HilError(NO_DEVICE, f"bluetoothctl could not connect {self.address}: "
                           f"{out.strip().splitlines()[-1] if out.strip() else 'no output'}",
                           hint="is the board advertising (BLE_STATUS state=advertising)? "
                                "a removed device needs a scan first")
        midi_in = midi_out = None
        try:
            midi_in = rtmidi.MidiIn()
            deadline = time.monotonic() + max(self._timeout_s, ALSA_PORT_WAIT_S)
            idx = self._port_index(midi_in.get_ports())
            while idx is None and time.monotonic() < deadline:
                time.sleep(ALSA_PORT_POLL_S)
                idx = self._port_index(midi_in.get_ports())
            if idx is None:
                raise HilError(NO_DEVICE, f"connected to {self.address} but no ALSA port "
                               f"'{self.name} Bluetooth' appeared",
                               hint="bluetoothd without the MIDI driver: use the bleak link")
            ports = midi_in.get_ports()
            self.port = ports[idx]
            midi_in.ignore_types(sysex=False, timing=False, active_sense=False)
            midi_in.set_callback(self._on_message)
            midi_in.open_port(idx)
            midi_out = rtmidi.MidiOut()
            midi_out.open_port(self._port_index(midi_out.get_ports()) or idx)
        except BaseException:
            # bluetoothctl connected; a port that will not open must not leave
            # the host holding the board's only link.
            for port in (midi_in, midi_out):
                if port is not None:
                    try:
                        port.close_port()
                    except Exception:  # noqa: BLE001 - teardown
                        pass
            midi_in = midi_out = None   # rtmidi frees the sequencer client on delete
            self.port = None
            try:
                self._run(["disconnect", self.address], BLUETOOTHCTL_TIMEOUT_S)
            except HilError:
                pass
            self._drop_cache_entry()
            raise
        self._in, self._out = midi_in, midi_out
        self._connected = True
        return self.status()

    def disconnect(self) -> dict[str, Any]:
        midi_in, midi_out = self._in, self._out
        self._in = self._out = None
        self._connected = False
        for port in (midi_in, midi_out):
            if port is not None:
                try:
                    port.close_port()
                except Exception:  # noqa: BLE001 - teardown
                    pass
        # rtmidi frees the sequencer client on delete, not on close_port().
        del midi_in, midi_out
        self._run(["disconnect", self.address], BLUETOOTHCTL_TIMEOUT_S)
        self._drop_cache_entry()
        return self.status()

    @property
    def connected(self) -> bool:
        return self._connected

    # -- rx ------------------------------------------------------------
    def _on_message(self, event: tuple[list[int], float], _data: Any = None) -> None:
        message, _delta = event
        now = self._clock()
        with self._lock:
            self._rx.append(RxMessage(t=now, msg=bytes(message)))
        self._arrived.set()

    def received(self, *, clear: bool = False) -> list[dict[str, Any]]:
        with self._lock:
            out = [m.to_dict() for m in self._rx]
            if clear:
                self._rx.clear()
        return out

    def clear_rx(self) -> None:
        with self._lock:
            self._rx.clear()
        self._arrived.clear()

    def count(self) -> int:
        with self._lock:
            return len(self._rx)

    def wait_for(self, count: int = 1, timeout_s: float = 2.0) -> bool:
        deadline = time.monotonic() + timeout_s
        while self.count() < count:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                return False
            self._arrived.wait(min(remaining, 0.05))
            self._arrived.clear()
        return True

    def listen(self, seconds: float, *, expect: int | None = None) -> dict[str, Any]:
        self.clear_rx()
        if expect:
            self.wait_for(expect, seconds)
        else:
            time.sleep(max(0.0, seconds))
        msgs = self.received()
        return {"seconds": seconds, "count": len(msgs), "messages": msgs,
                "notes_on": sum(1 for m in msgs if m["kind"] == "note_on")}

    # -- tx ------------------------------------------------------------
    def send(self, messages: Sequence[Sequence[int]], *, ts_ms: int | None = None,
             timeout_s: float = 5.0) -> dict[str, Any]:
        del ts_ms, timeout_s   # BlueZ timestamps and packs; rtmidi writes are immediate
        if not self._connected or self._out is None:
            raise HilError(NO_DEVICE, f"not connected to {self.address}", hint="connect() first")
        for m in messages:
            self._out.send_message(list(m))
        with self._lock:
            self._sent_packets += 1
            self._sent_messages += len(messages)
        return {"packet": "", "messages": len(messages), "bytes": sum(len(m) for m in messages)}

    def send_note(self, on: bool, note: int, vel: int = 100, channel: int = 0) -> dict[str, Any]:
        return self.send([note_message(on, note, vel, channel)])

    def send_hit(self, note: int, vel: int = 100, gate_s: float = 0.05) -> dict[str, Any]:
        self.send_note(True, note, vel)
        time.sleep(max(0.0, gate_s))
        return self.send_note(False, note)

    # -- reporting -----------------------------------------------------
    def status(self) -> dict[str, Any]:
        with self._lock:
            rx = len(self._rx)
            tx_packets, tx_messages = self._sent_packets, self._sent_messages
        return {"address": self.address, "name": self.name, "connected": self._connected,
                "transport": "alsa", "port": self.port,
                "properties": list(self.properties), "rx_messages": rx,
                "tx_packets": tx_packets, "tx_messages": tx_messages,
                "notify": True, "write_without_response": True}


def open_midi_link(address: str, *, name: str | None = None,
                   timeout_s: float = DEFAULT_CONNECT_TIMEOUT_S,
                   which: Callable[[str], str | None] | None = None,
                   forget: bool = False) -> Any:
    """The link a host actually offers: BlueZ's MIDI port where bluetoothd runs,
    a GATT client (bleak) elsewhere. On a Linux host with the driver the GATT
    route cannot work at all — the service is claimed before any client sees it.
    ``forget`` removes the BlueZ entry on disconnect if the host had not paired
    the board before (BlueZ route only)."""
    import shutil
    which = which or shutil.which
    if sys.platform.startswith("linux") and which("bluetoothctl"):
        ble: Any = AlsaBleMidiLink(address, name=name, timeout_s=timeout_s, forget=forget)
    else:
        ble = BleMidiLink(address, name=name, timeout_s=timeout_s)
    ble.connect()
    return ble


def connect(address: str | None = None, *, link: CdcLink | None = None,
            name: str = DEFAULT_NAME, scan_timeout_s: float = DEFAULT_SCAN_TIMEOUT_S,
            timeout_s: float = DEFAULT_CONNECT_TIMEOUT_S,
            discover_fn: Callable[[float], Any] | None = None,
            client_factory: Callable[[str, float], Any] = _bleak_client_factory) -> BleMidiLink:
    """Connect to the board: address if given, else scan and match ``BLE_STATUS self=``."""
    if address is None:
        want = self_address(link) if link is not None else None
        found = scan(scan_timeout_s, discover_fn=discover_fn)
        peer = select_peripheral(found, address=want, name=name)
        address, name = peer.address, peer.name or name
    ble = BleMidiLink(address, name=name, timeout_s=timeout_s, client_factory=client_factory)
    ble.connect()
    return ble


# --------------------------------------------------------------------------- latency

def latency(ble: BleMidiLink, link: CdcLink, *, n: int = 8, base_note: int = 60,
            vel: int = 100, timeout_s: float = 2.0, gap_s: float = 0.25) -> dict[str, Any]:
    """Device -> host round trip, measured on the path with nothing else in it.

    ``BLE_SEND`` hands the note straight to the transport, so what is timed is the CDC write
    plus the radio — not the pad scan, not the sample engine. The write is fire-and-forget
    (``CdcLink.send``): waiting for the ``OK`` would put a CDC read inside the measurement,
    and that ``OK`` is not necessarily this command's anyway.
    """
    if n <= 0:
        raise HilError(BAD_ARGS, f"n must be positive, got {n}")
    samples: list[float] = []
    lost: list[int] = []
    for i in range(n):
        note = base_note + (i % max(1, 128 - base_note))
        ble.clear_rx()
        t0 = time.perf_counter()
        link.send(f"BLE_SEND {note} {vel}")
        if ble.wait_for(1, timeout_s):
            first = ble.received()[0]
            samples.append((first["t"] - t0) * 1000.0)
        else:
            lost.append(note)
        time.sleep(gap_s)
    return {"sent": n, "received": len(samples), "lost": len(lost), "lost_notes": lost,
            "rtt_ms": _percentiles(samples)}


def _percentiles(values: Sequence[float]) -> dict[str, float] | None:
    if not values:
        return None
    ordered = sorted(values)
    return {"min": round(ordered[0], 2),
            "p50": round(statistics.median(ordered), 2),
            "p90": round(ordered[min(len(ordered) - 1, int(len(ordered) * 0.9))], 2),
            "max": round(ordered[-1], 2)}


# --------------------------------------------------------------------------- host peripheral

class HostPeripheral:
    """Advertise a BLE MIDI peripheral from this PC, so the board's Host role has a target.

    Windows only: WinRT's ``GattServiceProvider`` is the one cross-vendor API that advertises
    a service and accepts a connection without a second radio. On Linux this would mean a
    BlueZ GATT server over D-Bus, which bleak does not provide; the check is refused rather
    than faked, because a skipped test that reports PASS is worse than no test.
    """

    def __init__(self, *, name: str = "crosspad-hil") -> None:
        self.name = name
        self._runner = _Runner()
        self._provider: Any = None
        self._char: Any = None
        self._writes: list[bytes] = []
        self._lock = threading.Lock()

    @staticmethod
    def available() -> bool:
        if sys.platform != "win32":
            return False
        try:
            import winrt.windows.devices.bluetooth.genericattributeprofile  # noqa: F401
        except ImportError:
            return False
        return True

    def _require(self) -> Any:
        if sys.platform != "win32":
            raise HilError(NOT_SUPPORTED,
                           f"advertising a BLE MIDI peripheral needs the Windows GATT server "
                           f"API; this is {sys.platform}",
                           hint="test the board's Host role from a Windows PC, or connect it "
                                "to a real BLE MIDI peripheral and use BLE_DEVICES/BLE_CONNECT")
        try:
            from winrt.windows.devices.bluetooth import genericattributeprofile as gatt
        except ImportError as e:
            raise HilError(ENV, "the WinRT GATT server API is not installed",
                           hint="pip install winrt-Windows.Devices.Bluetooth "
                                "winrt-Windows.Storage.Streams") from e
        return gatt

    def start(self, timeout_s: float = 20.0) -> dict[str, Any]:
        gatt = self._require()
        import uuid as _uuid

        loop = self._runner.start()

        async def go() -> tuple[Any, Any]:
            r = await gatt.GattServiceProvider.create_async(_uuid.UUID(MIDI_SERVICE_UUID))
            provider = r.service_provider
            par = gatt.GattLocalCharacteristicParameters()
            par.characteristic_properties = (
                gatt.GattCharacteristicProperties.WRITE_WITHOUT_RESPONSE
                | gatt.GattCharacteristicProperties.NOTIFY
                | gatt.GattCharacteristicProperties.READ)
            par.read_protection_level = gatt.GattProtectionLevel.PLAIN
            par.write_protection_level = gatt.GattProtectionLevel.PLAIN
            cr = await provider.service.create_characteristic_async(
                _uuid.UUID(MIDI_CHAR_UUID), par)
            return provider, cr.characteristic

        provider, char = self._runner.call(go(), timeout_s, "create the GATT service")
        char.add_write_requested(lambda sender, args: self._on_write(loop, args))
        adv = gatt.GattServiceProviderAdvertisingParameters()
        adv.is_connectable = True
        adv.is_discoverable = True
        provider.start_advertising_with_parameters(adv)
        self._provider, self._char = provider, char
        return {"advertising": True, "service": MIDI_SERVICE_UUID}

    def _on_write(self, loop: asyncio.AbstractEventLoop, args: Any) -> None:
        # WinRT releases the event args as soon as the handler returns, so the request has to
        # be claimed with a deferral before awaiting anything.
        from winrt.windows.storage.streams import DataReader

        deferral = args.get_deferral()

        async def read() -> None:
            try:
                req = await args.get_request_async()
                reader = DataReader.from_buffer(req.value)
                buf = bytearray(req.value.length)
                reader.read_bytes(buf)
                with self._lock:
                    self._writes.extend(decode(bytes(buf)))
            finally:
                deferral.complete()

        asyncio.run_coroutine_threadsafe(read(), loop)

    def notify(self, messages: Sequence[Sequence[int]], timeout_s: float = 5.0) -> dict[str, Any]:
        if self._char is None:
            raise HilError(NO_DEVICE, "the peripheral is not advertising", hint="start() first")
        from winrt.windows.storage.streams import DataWriter

        writer = DataWriter()
        packet = encode(messages)
        writer.write_bytes(packet)
        self._runner.call(self._char.notify_value_async(writer.detach_buffer()),
                          timeout_s, "GATT notify")
        return {"packet": packet.hex(), "messages": len(messages)}

    def subscribers(self) -> int:
        return len(getattr(self._char, "subscribed_clients", []) or []) if self._char else 0

    def writes(self, *, clear: bool = False) -> list[dict[str, Any]]:
        with self._lock:
            out = [describe(m) for m in self._writes]
            if clear:
                self._writes.clear()
        return out

    def stop(self) -> dict[str, Any]:
        provider, self._provider, self._char = self._provider, None, None
        if provider is not None:
            provider.stop_advertising()
        self._runner.stop()
        return {"advertising": False}


def messages_from(spec: Iterable[Any]) -> list[list[int]]:
    """``[[0x90,60,100], "90 3C 64"]`` -> message lists, so CLI and daemon share one parser."""
    out: list[list[int]] = []
    for item in spec:
        if isinstance(item, str):
            try:
                data = list(bytes.fromhex(item.replace(",", " ").replace("0x", "").strip()))
            except ValueError as e:
                raise HilError(BAD_ARGS, f"{item!r} is not hex bytes, e.g. '90 3C 64'") from e
        elif isinstance(item, (list, tuple)):
            data = [int(b) for b in item]
        else:
            raise HilError(BAD_ARGS, f"a message is hex or a list of bytes, got {item!r}")
        if not data or not 0 <= data[0] <= 0xFF or not data[0] & 0x80:
            raise HilError(BAD_ARGS, f"message {data} does not start with a status byte")
        out.append(data)
    if not out:
        raise HilError(BAD_ARGS, "no messages given")
    return out


__all__ = [
    "DATA_BYTES",
    "MIDI_CHAR_UUID",
    "MIDI_SERVICE_UUID",
    "BleMidiLink",
    "BlePeripheral",
    "HostPeripheral",
    "connect",
    "decode",
    "describe",
    "encode",
    "latency",
    "messages_from",
    "note_message",
    "require_bleak",
    "scan",
    "select_peripheral",
    "self_address",
]
