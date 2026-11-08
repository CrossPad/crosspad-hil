"""The BLE MIDI link, driven from this PC's own radio — both directions, both roles.

Port of ``platform-idf/tools/hil_ble_midi.py``. The host radio is the second half
of this test: it connects to the board's BLE MIDI service, subscribes, and then
drives MIDI both ways while the device's CDC port answers what it saw. That is
what makes every leg falsifiable — a note that never sounded can be told apart
from one that never left the radio, because the device's counters and the host's
notifications are read independently.

What it checks, in the old script's order:

1. the board advertises the BLE MIDI service and is connectable;
2. the characteristic exposes write-without-response + notify, as the spec
   requires, and the negotiated connection interval is reported;
3. PC → device: notes reach PadManager and the sample engine (``PAD_STATS
   played`` climbs), including several messages packed into one packet;
4. device → PC: pad hits arrive as correctly encoded BLE MIDI notifications, at
   the note numbers the pad map plus the send transpose predicts;
5. round-trip latency on the one path with nothing else in it (``BLE_SEND``);
6. what the stack costs: free internal RAM with BLE stopped vs running.

Two traps this inherits and keeps. **The board must be the peripheral** for any
of that, so the run puts it in Server mode and restores the role it found — and
``--host-mode`` needs the opposite: something for the board's Host role to
connect *to*, which only a Windows host can advertise without a second radio.
Elsewhere that half is skipped, loudly, rather than faked. **The service UUID is
not an identity**: it is shared by every BLE MIDI peripheral in range, and one
run once attached to a passing speaker and reported six confident failures about
a device that was never under test. The board is matched on the address it
reports as ``self=`` in ``BLE_STATUS``.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from crosspad_hil import verbs
from crosspad_hil.cdc import CdcLink
from crosspad_hil.errors import AMBIGUOUS_DEVICE, NO_DEVICE, NOT_SUPPORTED, HilError
from crosspad_hil.scenarios import _deps
from crosspad_hil.scenarios.base import Context, Progress, Report, register

SAMPLER_APP = "Sampler"

# --- timing and thresholds, every one the old script's literal ----------------
BLE_STOP_SETTLE_S = 1.0        # time.sleep(1.0) after BLE_STOP, before MEM
BLE_START_SETTLE_S = 1.5       # time.sleep(1.5) after BLE_START 1, before MEM
KIT_LOAD_TIMEOUT_S = 20.0      # the old 40 × 0.5 s poll for `current == kit`
APP_START_SETTLE_S = 2.0       # time.sleep(2.0) after APP_START Sampler
NOTIFY_SETTLE_S = 0.4          # after start_notify, before believing BLE_STATUS
NOTE_GATE_S = 0.05             # note-on … note-off
NOTE_GAP_S = 0.05              # note-off … next note-on
RX_SETTLE_S = 0.6              # let the device parse the burst before reading rx_msg
PAD_GAP_S = 0.15               # between PAD_PRESS / PAD_RELEASE / the next pad
PAD_SETTLE_S = 0.4             # after the last pad, before counting notifications
MAX_PADS = 16                  # `pads = sorted(notes)[:16]`
DEVICE_TO_HOST_PADS = 8        # `for pad in pads[:8]`
PACKED_MESSAGES = 4            # note-on/off for two pads in one packet
LATENCY_NOTES = 8              # `for note in range(60, 68)`
LATENCY_MIN = 7                # `len(latencies) >= 7`
LATENCY_BASE_NOTE = 60

# --- host (central) role ------------------------------------------------------
ADV_SETTLE_S = 2.0             # advertising … BLE_START 0
ROLE_SETTLE_S = 1.5            # BLE_START 0 … BLE_SCAN
HOST_SCAN_MS = 8000            # BLE_SCAN 8000
HOST_SCAN_WAIT_S = 9.0         # …and wait it out before BLE_DEVICES
HOST_CONNECT_SETTLE_S = 6.0    # BLE_CONNECT … BLE_STATUS
HOST_NOTIFY_GAP_S = 0.2
HOST_NOTIFY_SETTLE_S = 1.0
HOST_PAD_SETTLE_S = 1.2
HOST_NOTES: tuple[int, ...] = (44, 45, 46)
HOST_PADS = 3
HOST_SKIP_HINT = ("the board's Host role needs a BLE MIDI peripheral to find; only a "
                  "Windows host can advertise one without a second radio")


@dataclass
class Params:
    kit: int | None = field(
        default=None,
        metadata={"help": "load this kit and start the Sampler first, so the note-to-sample "
                          "path is exercised too"})
    name: str = field(
        default="CrossPad",
        metadata={"help": "advertised name, used only when the firmware does not report "
                          "self= in BLE_STATUS"})
    scan_timeout: float = field(default=15.0, metadata={"help": "seconds to scan for the board"})
    connect_timeout: float = field(
        default=20.0, metadata={"help": "seconds to wait for the GATT connection"})
    host_mode: bool = field(
        default=False,
        metadata={"help": "also test the board's Host (central) role: this PC advertises a "
                          "BLE MIDI peripheral for it to find (Windows only)"})


# --------------------------------------------------------------------------- seams
# One line each, so a test can drive the whole run without a radio.


def scan_peripherals(timeout_s: float) -> list[Any]:
    """What this PC's radio can see advertising BLE MIDI."""
    return _deps.module("ble").scan(timeout_s)


def open_link(address: str, *, name: str | None, timeout_s: float) -> Any:
    """Connect to the board — BlueZ's MIDI port on Linux, a GATT client elsewhere
    (:func:`crosspad_hil.ble.open_midi_link`). The BlueZ entry the run creates
    is removed again on disconnect; a board paired before the run is kept."""
    return _deps.module("ble").open_midi_link(address, name=name, timeout_s=timeout_s,
                                              forget=True)


def measure_latency(ble: Any, link: CdcLink, **kw: Any) -> dict[str, Any]:
    """Device → host round trip on the path with nothing else in it (``BLE_SEND``)."""
    return dict(_deps.module("ble").latency(ble, link, **kw))


def host_peripheral() -> Any:
    """A BLE MIDI peripheral advertised from this PC, for the board's Host role."""
    return _deps.module("ble").HostPeripheral()


def host_peripheral_available() -> bool:
    return bool(_deps.module("ble").HostPeripheral.available())


# --------------------------------------------------------------------------- helpers


class Checks:
    """The old script's ``check(name, ok, detail)``: a named, falsifiable verdict.

    Everything is recorded, passing or not, because "9 of 10 checks passed" is
    only useful when the tenth is named — and a check that was never reached is
    visibly absent rather than silently green.
    """

    def __init__(self, ctx: Context, data: dict[str, Any]) -> None:
        self._ctx = ctx
        self._data = data

    def add(self, name: str, ok: bool, detail: str = "") -> bool:
        ok = bool(ok)
        self._data["checks"].append({"name": name, "ok": ok, "detail": detail})
        self._ctx.log(f"  {'PASS' if ok else 'FAIL'}  {name}" + (f" - {detail}" if detail else ""))
        if not ok:
            self._data["failures"].append(name)
        return ok


def _env(summary: str, data: dict[str, Any]) -> Report:
    return Report(passed=False, summary=summary, data=data, artifacts=[], exit_code=2)


def _wait(ctx: Context, seconds: float) -> None:
    """Sleep that returns early on cancel and then raises CANCELLED."""
    if seconds > 0:
        ctx.cancelled.wait(seconds)
    ctx.check_cancelled()


def _int(value: Any, default: int = 0) -> int:
    """A BLE_STATUS field as a number: the parser leaves '-' as None."""
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


class BleMidiScenario:
    name = "ble_midi"
    Params = Params
    description = "drive BLE MIDI both ways from this PC's radio, in both of the board's roles"

    def run(self, ctx: Context, params: Params, progress: Progress) -> Report:
        try:
            p = params
            data: dict[str, Any] = {
                "checks": [], "failures": [], "ram": None, "conn_interval_ms": None,
                "latency_ms": None, "peripheral": None, "notes": None, "txoff": 0,
                "kit": p.kit, "mode_before": None, "running_before": None, "restored": True,
                "host_mode": None,
            }
            ctx.check_cancelled()
            link = ctx.open_cdc()
            status = verbs.ble_status(link)
            if _int(status.get("supported")) != 1:
                return _env("ble_midi: this firmware has no BLE MIDI support "
                            "(CONFIG_BT_NIMBLE_ENABLED off)", data)
            prev_mode = str(status.get("mode") or "server")
            data["mode_before"] = prev_mode
            # The setup stops and starts the stack; a board found with BLE off has
            # to end with it off: left up, NimBLE holds ~30 kB of internal RAM for
            # the rest of the gate (int_free fell to 63 kB, Sampler pads went quiet).
            was_running = _int(status.get("running")) == 1
            data["running_before"] = was_running

            try:
                progress(0, 4, "measuring what the BLE stack costs")
                refused = self._measure_ram(ctx, link, data)
                if refused is not None:
                    return refused
                if p.kit is not None:
                    progress(1, 4, f"loading kit {p.kit} and starting the {SAMPLER_APP}")
                    self._load_kit(ctx, p, link)
                notes = dict(verbs.pad_notes(link).get("notes") or {})
                if not notes:
                    return _env("ble_midi: PAD_NOTES gave nothing — cannot map notes to pads",
                                data)
                data["notes"] = notes
                progress(2, 4, "server role: this PC connects to the board")
                self._server_role(ctx, p, link, notes, data)
                if p.host_mode:
                    progress(3, 4, "host role: the board connects to this PC")
                    self._host_role(ctx, link, data)
            finally:
                # The host-mode check switches the role, so this has to run
                # whichever way round the run started.
                self._restore(ctx, link, prev_mode, was_running, data)
            progress(4, 4, "done")
            return self._verdict(data)
        finally:
            ctx.close_all()

    # -- device-side preparation -------------------------------------------
    def _measure_ram(self, ctx: Context, link: CdcLink, data: dict[str, Any]) -> Report | None:
        """What the stack costs, measured rather than assumed.

        This doubles as the run's setup: BLE ends up started in Server mode
        whatever the persisted role was, and its counters start from there —
        which is what makes the bare ``rx_msg`` comparisons below meaningful.
        """
        verbs.ble_stop(link)
        _wait(ctx, BLE_STOP_SETTLE_S)
        off = verbs.mem(link)
        try:
            verbs.ble_start(link, 1)
        except HilError as exc:
            if exc.code != NOT_SUPPORTED:
                raise
            # BLE_START answers ERR when the stack cannot come up now: with
            # wireless.enableBLE off the board skipped the lazy start at boot,
            # and by the time a gate reaches this step no internal block is
            # large enough for the controller (~30 kB contiguous). That is the
            # rc1 shape of the board, not a MIDI defect.
            return _env("ble_midi: BLE_START refused -- BLE is off in settings and cannot "
                        "start this late in the boot; `crosspad-hil ble enable on`, reboot, "
                        "and run again (or accept that rc1 ships without BLE)", data)
        _wait(ctx, BLE_START_SETTLE_S)
        on = verbs.mem(link)
        cost = _int(off.get("int_free")) - _int(on.get("int_free"))
        data["ram"] = {"int_free_off": _int(off.get("int_free")),
                       "int_free_on": _int(on.get("int_free")), "cost": cost}
        ctx.log(f"internal RAM: {data['ram']['int_free_off']} B free with BLE off, "
                f"{data['ram']['int_free_on']} B with it running (cost {cost} B)")

    def _load_kit(self, ctx: Context, p: Params, link: CdcLink) -> None:
        verbs.kit_load(link, int(p.kit), wait_s=KIT_LOAD_TIMEOUT_S)
        verbs.app_start(link, SAMPLER_APP)
        _wait(ctx, APP_START_SETTLE_S)

    # -- server (peripheral) role ------------------------------------------
    def _server_role(self, ctx: Context, p: Params, link: CdcLink, notes: dict[int, int],
                     data: dict[str, Any]) -> None:
        ble = _deps.module("ble")
        check = Checks(ctx, data)

        ctx.log("scanning for the CrossPad")
        want = ble.self_address(link)
        found = scan_peripherals(p.scan_timeout)
        try:
            peer = ble.select_peripheral(found, address=want, name=p.name)
            detail = f"{peer.address} matches self={want}" if want else \
                     f"{peer.address} {peer.name!r} (matched by name, no self= reported)"
        except HilError as e:
            if e.code not in (NO_DEVICE, AMBIGUOUS_DEVICE):
                raise
            # Not finding the board is a failed check, not a broken bench: the
            # radio worked, the board did not answer it.
            check.add("advertises the BLE MIDI service", False, e.message)
            return
        check.add("advertises the BLE MIDI service", True, detail)
        data["peripheral"] = peer.to_dict()

        bl = open_link(peer.address, name=peer.name, timeout_s=p.connect_timeout)
        try:
            st = bl.status()
            check.add("characteristic has write-without-response + notify",
                      st["write_without_response"] and st["notify"],
                      ",".join(st["properties"]))
            _wait(ctx, NOTIFY_SETTLE_S)

            dev = verbs.ble_status(link)
            itvl = _int(dev.get("itvl"))
            data["conn_interval_ms"] = itvl * 1.25
            check.add("device reports the link up", dev.get("state") == "connected",
                      f"state={dev.get('state')} interval={itvl * 1.25:.2f} ms")

            pads = sorted(notes)[:MAX_PADS]
            self._pc_to_device(ctx, link, bl, notes, pads, check)
            self._packed_packet(ctx, link, bl, notes, pads, check)
            self._device_to_pc(ctx, link, bl, notes, pads, data, check)

            lat = measure_latency(bl, link, n=LATENCY_NOTES, base_note=LATENCY_BASE_NOTE)
            data["latency_ms"] = lat.get("rtt_ms")
            check.add("notes come back over the radio",
                      int(lat.get("received", 0)) >= LATENCY_MIN,
                      f"{lat.get('received')} of {LATENCY_NOTES} arrived")
            if lat.get("rtt_ms"):
                r = lat["rtt_ms"]
                ctx.log(f"  device -> host latency: min {r['min']} ms, median {r['p50']} ms, "
                        f"max {r['max']} ms")

            final = verbs.ble_status(link)
            check.add("no BLE transmit errors", _int(final.get("tx_err")) == 0,
                      f"tx_pkt={final.get('tx_pkt')} tx_err={final.get('tx_err')} "
                      f"tx_drop={final.get('tx_drop')}")
        finally:
            # The radio holds the connection open until it is told otherwise,
            # and the host-mode half needs the board free to go looking.
            try:
                bl.disconnect()
            except Exception as e:  # noqa: BLE001 - teardown must not mask a check
                ctx.log(f"BLE disconnect: {e}")

    def _pc_to_device(self, ctx: Context, link: CdcLink, bl: Any, notes: dict[int, int],
                      pads: list[int], check: Checks) -> None:
        verbs.pad_stats(link, reset=True)
        for pad in pads:
            bl.send_hit(notes[pad], 100, gate_s=NOTE_GATE_S)
            _wait(ctx, NOTE_GAP_S)
        _wait(ctx, RX_SETTLE_S)

        rx = _int(verbs.ble_status(link).get("rx_msg"))
        check.add("every note reached the device", rx >= 2 * len(pads),
                  f"{rx} of {2 * len(pads)} messages parsed")
        played = verbs.pad_stats(link)["played"]
        check.add("notes reached the sample engine", played >= len(pads),
                  f"played={played} of {len(pads)}")

    def _packed_packet(self, ctx: Context, link: CdcLink, bl: Any, notes: dict[int, int],
                       pads: list[int], check: Checks) -> None:
        """Several messages in one packet — the encoding a real host uses when the
        events fall inside one connection interval."""
        ble = _deps.module("ble")
        before = _int(verbs.ble_status(link).get("rx_msg"))
        bl.send([ble.note_message(True, notes[pads[0]], 90),
                 ble.note_message(False, notes[pads[0]]),
                 ble.note_message(True, notes[pads[1]], 90),
                 ble.note_message(False, notes[pads[1]])])
        _wait(ctx, RX_SETTLE_S)
        after = _int(verbs.ble_status(link).get("rx_msg"))
        check.add("multi-message packet decoded", after - before == PACKED_MESSAGES,
                  f"{after - before} of {PACKED_MESSAGES} messages")

    def _device_to_pc(self, ctx: Context, link: CdcLink, bl: Any, notes: dict[int, int],
                      pads: list[int], data: dict[str, Any], check: Checks) -> None:
        bl.clear_rx()
        for pad in pads[:DEVICE_TO_HOST_PADS]:
            verbs.pad_press(link, pad, 100)
            _wait(ctx, PAD_GAP_S)
            verbs.pad_release(link, pad)
            _wait(ctx, PAD_GAP_S)
        _wait(ctx, PAD_SETTLE_S)

        received = bl.received()
        # describe() reads a note-on with velocity 0 as the note-off it is, which
        # is what a host does; the firmware sends a real 0x80 either way.
        got_on = [m for m in received if m["kind"] == "note_on"]
        got_off = [m for m in received if m["kind"] == "note_off"]
        check.add("pad hits arrive as BLE MIDI notes",
                  len(got_on) >= DEVICE_TO_HOST_PADS,
                  f"{len(got_on)} note-on, {len(got_off)} note-off")

        # The send transpose is part of what should be on the wire, so the
        # expected notes are the pad map shifted by it — not the pad map.
        txoff = _int(verbs.ble_status(link).get("txoff"))
        data["txoff"] = txoff
        expected = {n + txoff for n in notes.values()}
        check.add("note numbers match the pad map",
                  all(m["note"] in expected for m in got_on),
                  f"transpose {txoff:+d}, got "
                  + ",".join(str(m["note"]) for m in got_on[:DEVICE_TO_HOST_PADS]))

    # -- host (central) role -----------------------------------------------
    def _host_role(self, ctx: Context, link: CdcLink, data: dict[str, Any]) -> None:
        """Advertise a BLE MIDI peripheral here and let the board come and find it."""
        ble = _deps.module("ble")
        check = Checks(ctx, data)
        if not host_peripheral_available():
            # Skipped, not faked, and not counted as a check either way — the old
            # script's behaviour, and a skipped check that reports PASS would be
            # worse than no check.
            ctx.log("host-mode check skipped: " + HOST_SKIP_HINT)
            data["host_mode"] = {"skipped": True, "reason": HOST_SKIP_HINT}
            return
        data["host_mode"] = {"skipped": False, "devices": None, "writes": 0}

        peripheral = host_peripheral()
        peripheral.start()
        _wait(ctx, ADV_SETTLE_S)
        try:
            ctx.log("host mode: the board now looks for this PC")
            verbs.ble_start(link, 0)
            _wait(ctx, ROLE_SETTLE_S)
            verbs.ble_scan(link, HOST_SCAN_MS)
            _wait(ctx, HOST_SCAN_WAIT_S)
            listing = verbs.ble_devices(link)
            data["host_mode"]["devices"] = listing
            if not check.add("scan finds a BLE MIDI peripheral",
                             bool(listing.get("devices")), str(listing)):
                return
            addr = listing["devices"][0]["addr"]

            verbs.ble_connect(link, addr)
            _wait(ctx, HOST_CONNECT_SETTLE_S)
            st = verbs.ble_status(link)
            check.add("board connects as central", st.get("state") == "connected",
                      f"peer={st.get('peer')} interval={_int(st.get('itvl')) * 1.25:.2f} ms")
            check.add("board subscribed to the peer", peripheral.subscribers() > 0,
                      f"{peripheral.subscribers()} subscribed client(s)")

            before = _int(verbs.ble_status(link).get("rx_msg"))
            for note in HOST_NOTES:
                peripheral.notify([ble.note_message(True, note, 100)])
                peripheral.notify([ble.note_message(False, note)])
                _wait(ctx, HOST_NOTIFY_GAP_S)
            _wait(ctx, HOST_NOTIFY_SETTLE_S)
            after = _int(verbs.ble_status(link).get("rx_msg"))
            expected = 2 * len(HOST_NOTES)
            check.add("peer notifications reach the board", after - before == expected,
                      f"{after - before} of {expected} messages")

            peripheral.writes(clear=True)
            for pad in range(HOST_PADS):
                verbs.pad_press(link, pad, 100)
                _wait(ctx, HOST_NOTIFY_GAP_S)
                verbs.pad_release(link, pad)
                _wait(ctx, HOST_NOTIFY_GAP_S)
            _wait(ctx, HOST_PAD_SETTLE_S)
            writes = peripheral.writes()
            data["host_mode"]["writes"] = len(writes)
            check.add("pad hits reach the peer", len(writes) >= 2 * HOST_PADS,
                      f"{len(writes)} messages written to the peripheral")
        finally:
            try:
                verbs.ble_disconnect(link)
            except HilError as e:
                ctx.log(f"BLE_DISCONNECT: {e.code}: {e.message}")
            peripheral.stop()

    # -- teardown and verdict ----------------------------------------------
    def _restore(self, ctx: Context, link: CdcLink, prev_mode: str, was_running: bool,
                 data: dict[str, Any]) -> None:
        """Leave the board as it was found: stack up in its role, or stack down."""
        try:
            if was_running:
                verbs.ble_start(link, 0 if prev_mode == "host" else 1)
            else:
                verbs.ble_stop(link)
        except HilError as e:
            data["restored"] = False
            want = f"running in {prev_mode!r} mode" if was_running else "stopped"
            ctx.log(f"could not leave BLE {want}: {e.code}: {e.message}")

    def _verdict(self, data: dict[str, Any]) -> Report:
        checks: list[dict] = data["checks"]
        passed = sum(1 for c in checks if c["ok"])
        total = len(checks)
        body = f"{passed}/{total} checks passed"
        if data["failures"]:
            return Report(passed=False,
                          summary=f"ble_midi: FAIL — {body}; failed: "
                                  + ", ".join(data["failures"]),
                          data=data, artifacts=[], exit_code=1)
        if not checks:
            return _env("ble_midi: no check ran — the radio never got as far as the board",
                        data)
        note = ""
        if data.get("host_mode") and data["host_mode"].get("skipped"):
            note = " (host role skipped: needs a Windows host)"
        if not data.get("restored", True):
            want = "running" if data.get("running_before") else "stopped"
            return _env(f"ble_midi: {body}, but BLE could not be left {want} as found — "
                        "the next step would run with the wrong stack state", data)
        return Report(passed=True, summary=f"ble_midi: PASS — {body}{note}",
                      data=data, artifacts=[], exit_code=0)


SCENARIO = register(BleMidiScenario())


def main(argv: list[str] | None = None) -> int:
    """Entry point for the platform-idf shim `tools/hil_ble_midi.py`."""
    import sys

    from crosspad_hil.cli import main as cli_main

    args = list(sys.argv[1:] if argv is None else argv)
    return cli_main(["run", "ble_midi", *args])
