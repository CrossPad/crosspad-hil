"""A fixed pair of MIDI ports the DAW opens instead of the CrossPad itself.

Wine enumerates ALSA MIDI ports once, when the DAW starts, matches incoming
events to the device it opened by their ALSA source address, and never lets
go of an input that vanished ("already allocated"). So a DAW under Wine can
only follow the CrossPad when the board was there first, stays put, and
never re-enumerates — which a USB profile switch or an OTA flash does every
time.

The hub is a user-space ALSA sequencer client that lives from login
(`systemd --user`): one readable port and one writable port, both called
"CrossPad Hub". The DAW opens those. Whenever a CrossPad ESP MIDI port is
present the hub bridges it: board → hub-readable (the DAW and any host tool
hear it), hub-writable → board (what the DAW and host tools send). Traffic
written into the hub is also mirrored to the readable side, so a host tool
sharing the DAW's script (the fl-studio-mcp bridge) talks to the DAW through
the same two ports — no relay through the board, no `SYSEX_ECHO`. Nothing
ever flows readable → writable, so there is no loop; each consumer ignores
the frames that are not for it (the CrossPad script drops non-hello device
frames, the bridge drops its own responses).

The hub is also the host endpoint the board hears from: while a board is
attached it sends `HOST_INFO` (this computer's CPU, RAM and clock) every 2 s
and `FW_LATEST` (host_report.latest_for) every minute, as companion
*notify* frames (0x1F 04) — the board runs them and never answers, so no reply
lands in a browser's or crosspad-hil's own companion exchange. That is what
lights the USB glyph on the board's status bar. One frame per poll: the board
runs one companion verb at a time and drops a notify that finds it busy.
The exception is the version check (verify_tick): every VERIFY_EVERY_S (600 s; after
an unanswered request, or while no crosspad.app catalog is held yet, VERIFY_RETRY_S
= 30 s) the hub *requests* FW_UPD_STATUS and then FW_LATEST (0x1F 01,
one at a time) and swallows the replies (_take_reply) instead of forwarding
them; the verdict goes to hub_versions.json.

A DAW that speaks only raw MIDI devices (REAPER on Linux: ALSA raw or JACK,
never the sequencer) cannot open the hub's ports, and the board's own raw
device is the hub's. For it the hub bridges a virtual raw card too
(`snd-virmidi`, "VirMIDI"): board → VirMIDI beside the DAW ports, VirMIDI →
board. Nothing written into VirMIDI is mirrored back into it, so the DAW never
hears its own frames.

The same report says which DAWs this computer can start and which already run
(`daws=`, daw_launcher). DAW Control answers a button press with a DAW
`launch` frame; the hub starts that DAW and keeps the frame to itself — it is
addressed to the hub, not to whatever DAW is listening.
"""
from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import threading
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any, Protocol

from crosspad_hil import daw_launcher, fw_catalog, host_report, hub_ipc
from crosspad_hil.cdc import CdcLink
from crosspad_hil.devices import HUB_MIDI_NAME, Device, discover, is_esp_midi_name, select
from crosspad_hil.errors import BAD_ARGS, NO_DEVICE, TIMEOUT, HilError
from crosspad_hil.hub_core import HubCore
from crosspad_hil.locks import PortLock
from crosspad_hil.midi import (
    parse_companion_chunk,
    parse_daw_launch,
    sysex_companion,
    sysex_companion_notify,
)

HUB_NAME = HUB_MIDI_NAME
UNIT_NAME = "crosspad-midi-hub.service"
POLL_S = 1.0
REPORT_S = 2.0          # the board calls a host gone after three missed reports
LATEST_EVERY_S = 60.0
HUB_LOCK_KEY = "crosspad-hub"     # the singleton: one hub per user, whoever started it
BOARD_BUSY_REPLY = "ERR busy"
# The board runs a notify within one main-loop pass: ERR busy past this is its own.
NOTIFY_SETTLE_S = 0.5
BUSY_RETRY_S = 0.02
STATUS_TIMEOUT_S = 2.0
VERIFY_EVERY_S = 600.0
VERIFY_RETRY_S = 30.0     # after a request that went unanswered
VERIFY_TIMEOUT_S = 5.0
VERIFY_VERBS = ("FW_UPD_STATUS", "FW_LATEST")
_REPLY_PREFIX = {"FW_UPD_STATUS": "FWUPD", "FW_LATEST": "FW_LATEST"}
MAX_REPLY_CHARS = 512
VERSIONS_PATH = Path.home() / ".cache" / "crosspad-hil" / "hub_versions.json"

# The board's reply is untrusted: only strings of these shapes reach VERSIONS_PATH.
_KIT_RE = re.compile(r"[a-z0-9_-]{1,32}")
_VERSION_RE = re.compile(r"v?\d+\.\d+\.\d+(-[0-9A-Za-z.-]{1,32})?")
_NUMBER_RE = re.compile(r"\d{1,12}")

# The ESP native port ("Crosspad" / "Crosspad Audio" card), never the STM
# bridge ("CrossPad MIDI+Serial") and never our own hub.
is_esp_port = is_esp_midi_name

VIRMIDI_MATCH = "VirMIDI"   # snd-virmidi's sequencer port, "VirMIDI <card>-<dev>"


def is_virmidi_port(name: str) -> bool:
    return VIRMIDI_MATCH in name


_DEV_BUILD_RE = re.compile(r".+-\d+-g[0-9a-f]+(-dirty)?")    # git describe suffix
RECHECK_VERDICTS = ("catalog_unknown", "unknown")


def _is_dev_build(running: str) -> bool:
    """A git-describe build, or a tagged commit with local changes (`v1.1.0-rc4-dirty`)."""
    return _DEV_BUILD_RE.fullmatch(running) is not None or running.endswith("-dirty")


def verdict(status: dict, latest: dict, catalog: str | None,
            catalog_held: bool | None = None) -> str:
    """What the board's own reading says against crosspad.app's newest version.

    `catalog` None has two causes: no catalog was ever fetched (`catalog_unknown`, looked at again
    soon) or the catalog lists nothing on the board's channel (`no_release_on_channel`, normal
    pace). `catalog_held` says which; left out, a None `catalog` means none held."""
    running = (status.get("running") or "").removeprefix("v")
    if not running:
        return "unknown"
    if not catalog:
        return "no_release_on_channel" if catalog_held else "catalog_unknown"
    if _is_dev_build(running):
        return "dev_build"
    if running == catalog:
        return "up_to_date"
    if latest.get("latest") == catalog and latest.get("update") == "1":
        return "update_offered"
    mine, theirs = fw_catalog.order_key(running), fw_catalog.order_key(catalog)
    if mine is None or theirs is None:
        return "unordered"                  # a version string the board's ordering cannot rank
    if mine > theirs:
        return "board_ahead"
    return "board_behind_catalog"


def _clean(value: str | None, pattern: re.Pattern[str]) -> str | None:
    return value if value is not None and pattern.fullmatch(value) else None


class InPort(Protocol):
    def close_port(self) -> None: ...
    def set_callback(self, cb: Callable[[tuple[list[int], float], Any], None]) -> None: ...


class OutPort(Protocol):
    def send_message(self, message: list[int]) -> None: ...
    def close_port(self) -> None: ...


class Ports(Protocol):
    """The rtmidi surface the hub needs; tests hand in a fake."""

    def list_inputs(self) -> list[str]: ...
    def list_outputs(self) -> list[str]: ...
    def open_virtual_in(self, name: str) -> InPort: ...
    def open_virtual_out(self, name: str) -> OutPort: ...
    def open_in(self, name: str) -> InPort: ...
    def open_out(self, name: str) -> OutPort: ...


class RtmidiPorts:
    def __init__(self) -> None:
        import rtmidi
        self._rtmidi = rtmidi
        # Every MidiIn/MidiOut is an ALSA sequencer client and rtmidi frees
        # it lazily; a fresh pair per poll exhausts the client table in a
        # couple of minutes ("error creating ALSA sequencer client object").
        # One long-lived pair does all the listing.
        self._probe_in = rtmidi.MidiIn()
        self._probe_out = rtmidi.MidiOut()

    def list_inputs(self) -> list[str]:
        return list(self._probe_in.get_ports())

    def list_outputs(self) -> list[str]:
        return list(self._probe_out.get_ports())

    def open_virtual_in(self, name: str) -> InPort:
        p = self._rtmidi.MidiIn()
        p.ignore_types(sysex=False, timing=True, active_sense=True)
        p.open_virtual_port(name)
        return p

    def open_virtual_out(self, name: str) -> OutPort:
        p = self._rtmidi.MidiOut()
        p.open_virtual_port(name)
        return p

    def open_in(self, name: str) -> InPort:
        p = self._rtmidi.MidiIn()
        p.ignore_types(sysex=False, timing=True, active_sense=True)
        p.open_port(p.get_ports().index(name))
        return p

    def open_out(self, name: str) -> OutPort:
        p = self._rtmidi.MidiOut()
        p.open_port(p.get_ports().index(name))
        return p


class Hub:
    def __init__(self, ports: Ports, log: Callable[[str], None] = print, *,
                 host_info: Callable[[], str] | None = None,
                 latest: Callable[[], str | None] | None = None,
                 launch: Callable[[int], str] | None = None,
                 catalog_latest: Callable[[str, str], str | None] | None = None,
                 catalog_held: Callable[[], bool] | None = None,
                 versions_path: Path = VERSIONS_PATH) -> None:
        """`host_info` builds the HOST_INFO line, `latest` returns the newest
        firmware version and `launch` starts a DAW by daw_id (DawLauncher.launch);
        without `host_info` the hub only bridges. `catalog_latest(kit, channel)`
        is crosspad.app's newest version: with it the hub asks the board what it
        runs and records a verdict in `versions_path`. `catalog_held()` says whether any catalog
        has been fetched (CatalogCache.has_catalog): only then is a missing newest version "no
        release on this channel" rather than "catalog unknown"."""
        self._ports = ports
        self._log = log
        self._lock = threading.Lock()
        # the rtmidi callback and the poll loop both write the board
        self._send_lock = threading.Lock()
        self._host_info = host_info
        self._latest = latest
        self._launch = launch
        self._next_info = 0.0
        self._next_latest = 0.0
        self._catalog_latest = catalog_latest
        self._catalog_held = catalog_held
        self._versions_path = versions_path
        # the rtmidi callback (replies) and the poll loop (requests) share these
        self._verify_lock = threading.Lock()
        self._queue: list[str] = []
        self._pending: tuple[str, float] | None = None
        self._parts: list[str] = []
        self._round: dict[str, dict[str, str]] = {}
        self._next_verify = 0.0
        self._tick = 0.0
        self.reports_sent = 0
        self.hub_in: InPort | None = None     # writable from outside (DAW → us)
        self.hub_out: OutPort | None = None   # readable from outside (us → DAW)
        self.board_in: InPort | None = None
        self.board_out: OutPort | None = None
        self.board_name: str | None = None
        self.board_kit: str | None = None
        self.board_channel: str | None = None
        self.raw_in: InPort | None = None     # VirMIDI: what a raw-MIDI DAW writes
        self.raw_out: OutPort | None = None   # VirMIDI: what a raw-MIDI DAW reads
        self.raw_name: str | None = None
        self.forwarded_to_board = 0
        self.forwarded_to_daw = 0
        # One 0x1F request on the board at a time, the hub's own notify frames included:
        # the board runs one companion verb at a time and answers a second with ERR busy.
        self._companion_lock = threading.Lock()
        self._companion_parts: list[str] | None = None
        self._companion_done = threading.Event()
        self._last_notify = float("-inf")         # monotonic time of the hub's own last notify
        # Without virtual ports (WinMM) the board's ports are single-client: taking
        # them would lock DAWs and the USB profile switch out, and bridge to nobody.
        self._shares_board_midi = True

    # -- lifecycle ---------------------------------------------------------

    def open(self) -> None:
        try:
            self.hub_out = self._ports.open_virtual_out(HUB_NAME)
            self.hub_in = self._ports.open_virtual_in(HUB_NAME)
        except Exception as exc:  # noqa: BLE001 - WinMM has no virtual ports
            _close(self.hub_out, self.hub_in)
            self.hub_out = self.hub_in = None
            self._shares_board_midi = False
            self._log(f"no virtual MIDI ports here ({exc}); the board's MIDI stays free for "
                      f"others, the hub arbitrates its CDC only")
            return
        self.hub_in.set_callback(self._from_hub)
        self._log(f"hub ports '{HUB_NAME}' up")

    def close(self) -> None:
        self.detach()
        self._detach_raw()
        for p in (self.hub_in, self.hub_out):
            if p is not None:
                p.close_port()
        self.hub_in = self.hub_out = None

    # -- board -------------------------------------------------------------

    def poll(self) -> None:
        """Attach to the ESP port when it is there, detach when it is gone; same for VirMIDI."""
        if not self._shares_board_midi:
            return
        all_inputs = self._ports.list_inputs()
        all_outputs = self._ports.list_outputs()
        self._poll_raw(all_inputs, all_outputs)
        inputs = [n for n in all_inputs if is_esp_port(n)]
        outputs = [n for n in all_outputs if is_esp_port(n)]
        if self.board_name is not None:
            if self.board_name in inputs and self.board_name in outputs:
                return
            self.detach()
        common = [n for n in inputs if n in outputs]
        if common:
            self.attach(common[0])

    def attach(self, name: str) -> None:
        try:
            board_out = self._ports.open_out(name)
            board_in = self._ports.open_in(name)
        except Exception as exc:  # noqa: BLE001 - a half-enumerated port is normal for a moment
            self._log(f"board port {name!r}: cannot open yet ({exc})")
            return
        with self._lock:
            self.board_out = board_out
            self.board_in = board_in
            self.board_name = name
        self._next_info = self._next_latest = 0.0     # a fresh board hears from us at once
        with self._verify_lock:
            self._next_verify = 0.0
            self._queue, self._pending, self._parts, self._round = [], None, [], {}
        board_in.set_callback(self._from_board)
        self._log(f"board {name!r} attached")

    def detach(self) -> None:
        with self._lock:
            board_in, board_out, name = self.board_in, self.board_out, self.board_name
            self.board_in = self.board_out = None
            self.board_name = None
        _close(board_in, board_out)
        if name is not None:
            self._log(f"board {name!r} detached")

    # -- VirMIDI (raw-MIDI DAWs) -------------------------------------------

    def _poll_raw(self, inputs: list[str], outputs: list[str]) -> None:
        if self.raw_name is not None:
            if self.raw_name in inputs and self.raw_name in outputs:
                return
            self._detach_raw()
        common = sorted(n for n in inputs if is_virmidi_port(n) and n in outputs)
        if not common:
            return
        try:
            raw_out = self._ports.open_out(common[0])
            raw_in = self._ports.open_in(common[0])
        except Exception as exc:  # noqa: BLE001 - same as a half-enumerated board port
            self._log(f"VirMIDI {common[0]!r}: cannot open yet ({exc})")
            return
        with self._lock:
            self.raw_out, self.raw_in, self.raw_name = raw_out, raw_in, common[0]
        raw_in.set_callback(self._from_raw)
        self._log(f"VirMIDI {common[0]!r} attached")

    def _detach_raw(self) -> None:
        with self._lock:
            raw_in, raw_out, name = self.raw_in, self.raw_out, self.raw_name
            self.raw_in = self.raw_out = None
            self.raw_name = None
        _close(raw_in, raw_out)
        if name is not None:
            self._log(f"VirMIDI {name!r} detached")

    # -- forwarding --------------------------------------------------------

    def _from_board(self, event: tuple[list[int], float], _data: Any = None) -> None:
        message = event[0]
        daw_id = parse_daw_launch(message)
        if daw_id is not None:
            if self._launch is not None:
                self._log(f"board asks to launch daw_id {daw_id}: {self._launch(daw_id)}")
            return
        if self._take_reply(message):
            return
        # A client's request through the hub (companion()): its reply is the client's,
        # never the DAW's. Any other 0x1F reply was asked for by someone on the ports.
        parts = self._companion_parts
        if parts is not None:
            chunk = parse_companion_chunk(bytes(message))
            if chunk is not None:
                parts.append(chunk[0])
                if chunk[1]:
                    self._companion_done.set()
                return
        if self.hub_out is not None:
            self.hub_out.send_message(message)
            self.forwarded_to_daw += 1
        with self._lock:
            raw_out = self.raw_out
        if raw_out is not None:
            try:
                raw_out.send_message(message)
            except Exception as exc:  # noqa: BLE001 - the module was unloaded
                self._log(f"VirMIDI write failed ({exc}); detaching")
                self._detach_raw()

    # -- version verification ----------------------------------------------

    def verify_tick(self, now: float) -> None:
        """Companion requests FW_UPD_STATUS then FW_LATEST, one at a time, every
        VERIFY_EVERY_S; replies land in _from_board."""
        if self.board_name is None or self._catalog_latest is None:
            return                     # no catalog source (the plain hub): no verification
        with self._verify_lock:
            self._tick = now
            if self._pending is not None:
                if now - self._pending[1] < VERIFY_TIMEOUT_S:
                    return
                self._log(f"companion {self._pending[0]}: no reply in {VERIFY_TIMEOUT_S:.0f}s")
                self._queue, self._pending, self._parts, self._round = [], None, [], {}
                self._next_verify = now + VERIFY_RETRY_S
                return
            if not self._queue and now >= self._next_verify:
                self._next_verify = now + VERIFY_EVERY_S
                self._queue = list(VERIFY_VERBS)
                self._round = {}
            if not self._queue:
                return
            verb = self._queue.pop(0)
            # set before the write: the reply may be fast
            self._pending, self._parts = (verb, now), []
        if not self._write_board(list(sysex_companion(verb))):
            with self._verify_lock:
                self._queue, self._pending, self._parts, self._round = [], None, [], {}

    def _take_reply(self, message: list[int]) -> bool:
        """True when `message` is a chunk of the reply to our request (not for the DAW)."""
        done: tuple[str, str] | None = None
        with self._verify_lock:
            if self._pending is None:
                return False
            try:
                chunk = parse_companion_chunk(bytes(message))
            except (ValueError, TypeError):      # not bytes-like: someone else's frame
                return False
            if chunk is None:
                return False
            verb = self._pending[0]
            prefix = _REPLY_PREFIX[verb]
            if not self._parts and not chunk[0].startswith((prefix, "ERR")):
                return False                     # a reply to somebody else's verb
            self._parts.append(chunk[0])
            if sum(len(p) for p in self._parts) > MAX_REPLY_CHARS:
                self._log(f"companion {verb}: reply too long, dropped")
                self._queue, self._pending, self._parts, self._round = [], None, [], {}
                return True
            if chunk[1]:
                done = (verb, "".join(self._parts))
                self._pending, self._parts = None, []
        if done is not None:
            try:
                self._on_reply(*done)
            except Exception as exc:  # noqa: BLE001 - an odd reply is dropped, never raised into rtmidi
                self._log(f"companion {done[0]}: reply dropped ({exc})")
                with self._verify_lock:
                    self._queue, self._round = [], {}
                    # e.g. "ERR busy": not an answer; ask again soon, not in VERIFY_EVERY_S
                    self._next_verify = min(self._next_verify, self._tick + VERIFY_RETRY_S)
        return True

    def _on_reply(self, verb: str, text: str) -> None:
        if not text.startswith(_REPLY_PREFIX[verb]):
            raise ValueError(f"board answered {text[:40]!r}")
        fields = fw_catalog.reply_fields(text)
        if verb == "FW_UPD_STATUS":
            kit = _clean(fields.get("kit"), _KIT_RE)
            channel = _clean(fields.get("channel"), _KIT_RE)
            if kit is None or channel is None:
                raise ValueError("no usable kit/channel")
            self.board_kit, self.board_channel = kit, channel
            fields = {"kit": kit, "channel": channel,
                      "running": _clean(fields.get("running"), _VERSION_RE) or "",
                      "last_check": _clean(fields.get("last_check"), _NUMBER_RE) or ""}
        else:
            fields = {"latest": _clean(fields.get("latest"), _VERSION_RE) or "",
                      "update": fields.get("update") if fields.get("update") in ("0", "1") else ""}
        with self._verify_lock:
            self._round[verb] = fields
            if not all(v in self._round for v in VERIFY_VERBS):
                return
            st, lt = self._round["FW_UPD_STATUS"], self._round["FW_LATEST"]
            self._round = {}
        # held first: a fetch that lands between the two reads shows up as a version, never as
        # "nothing on this channel"
        held = self._catalog_held() if self._catalog_held is not None else False
        catalog = self._catalog_latest(st["kit"], st["channel"]) if self._catalog_latest else None
        catalog = _clean(catalog, _VERSION_RE)
        record = {"board": self.board_name, "time": int(time.time()), "kit": st["kit"],
                  "channel": st["channel"], "running": st["running"] or None,
                  "board_latest": lt["latest"] or None, "catalog_latest": catalog,
                  "update": lt["update"] or None, "last_check": st["last_check"] or None,
                  "verdict": verdict(st, lt, catalog, held)}
        _write_json_atomic(self._versions_path, record)
        if record["verdict"] in RECHECK_VERDICTS:
            # the catalog cache fills in the background: look again soon, not in VERIFY_EVERY_S
            with self._verify_lock:
                self._next_verify = min(self._next_verify, self._tick + VERIFY_RETRY_S)
        self._log(f"versions: {record['verdict']} running={record['running']} catalog={catalog}")

    def _from_raw(self, event: tuple[list[int], float], _data: Any = None) -> None:
        """A raw-MIDI DAW → the board only; it must not hear itself."""
        message = event[0]
        if self._write_board(message):
            self.forwarded_to_board += 1

    def _from_hub(self, event: tuple[list[int], float], _data: Any = None) -> None:
        message = event[0]
        # Mirror first: a host tool waiting for the DAW's reply must see it
        # even while no board is attached.
        if self.hub_out is not None:
            self.hub_out.send_message(message)
        if self._write_board(message):
            self.forwarded_to_board += 1

    def _write_board(self, message: list[int]) -> bool:
        with self._lock:
            out = self.board_out
        if out is None:
            return False
        try:
            with self._send_lock:
                out.send_message(message)
            return True
        except Exception as exc:  # noqa: BLE001 - the board went away mid-write
            self._log(f"board write failed ({exc}); detaching")
            self.detach()
            return False

    # -- host report -------------------------------------------------------

    def report(self, now: float) -> None:
        """At most one notify per call: HOST_INFO when due, else FW_LATEST when due. A due version
        check (verify_tick) is a request of its own, sent first, and takes the slot's reply.
        Skipped while a client's companion request is in flight."""
        if self._host_info is None or self.board_name is None:
            return
        if not self._companion_lock.acquire(blocking=False):
            return
        try:
            self.verify_tick(now)
            if now >= self._next_info:
                self._next_info = now + REPORT_S
                if self._write_board(list(sysex_companion_notify(self._host_info()))):
                    self.reports_sent += 1
                    self._last_notify = time.monotonic()
                return
            if self._latest is not None and now >= self._next_latest:
                self._next_latest = now + LATEST_EVERY_S
                try:
                    version = self._latest()
                    frame = (list(sysex_companion_notify("FW_LATEST " + version))
                             if version else None)
                except Exception as exc:  # noqa: BLE001 - FW_LATEST is a hint; skip this tick
                    self._log(f"FW_LATEST skipped ({exc})")
                    return
                if frame and self._write_board(frame):
                    self._last_notify = time.monotonic()
        finally:
            self._companion_lock.release()

    # -- companion ---------------------------------------------------------

    def companion(self, verb: str, timeout_s: float) -> str:
        """Run one companion verb on the attached board; the reply goes to the caller only.

        The board answers ERR busy while it still runs the hub's own notify: that
        answer, within NOTIFY_SETTLE_S of one, is asked again rather than handed on.
        A version request of the hub's own (verify_tick) still out is answered first.
        """
        with self._companion_lock:
            deadline = time.monotonic() + timeout_s
            self._wait_for_version_check(verb, deadline)
            while True:
                reply = self._companion_once(verb, max(0.0, deadline - time.monotonic()))
                if reply.strip() != BOARD_BUSY_REPLY or \
                        time.monotonic() - self._last_notify > NOTIFY_SETTLE_S:
                    return reply
                time.sleep(BUSY_RETRY_S)

    def _companion_once(self, verb: str, timeout_s: float) -> str:
        if self.board_name is None:
            raise HilError(NO_DEVICE, "no CrossPad MIDI port attached to the hub",
                           hint="is the board plugged in and in the MIDI+UAC2 profile?")
        self._companion_done.clear()
        self._companion_parts = []
        try:
            if not self._write_board(list(sysex_companion(verb))):
                raise HilError(NO_DEVICE, "the board went away while sending", verb=verb)
            if not self._companion_done.wait(timeout_s):
                raise HilError(TIMEOUT, f"no companion reply to {verb!r} within "
                                        f"{timeout_s:g} s",
                               hint="firmware older than core 1.12 has no SysEx 0x1F",
                               verb=verb)
            return "".join(self._companion_parts)
        finally:
            self._companion_parts = None

    def _wait_for_version_check(self, verb: str, deadline: float) -> None:
        """Until the hub's own FW_UPD_STATUS / FW_LATEST request has its reply: a chunk
        that came in while both were out could belong to either. One the board never
        answers is given up after VERIFY_TIMEOUT_S, as verify_tick would."""
        give_up = time.monotonic() + VERIFY_TIMEOUT_S
        while True:
            now = time.monotonic()
            with self._verify_lock:
                if self._pending is None:
                    return
                if now >= give_up:
                    self._log(f"companion {self._pending[0]}: no reply in "
                              f"{VERIFY_TIMEOUT_S:.0f}s")
                    self._queue, self._pending, self._parts, self._round = [], None, [], {}
                    self._next_verify = self._tick + VERIFY_RETRY_S
                    return
            if now >= deadline:
                raise HilError(TIMEOUT, f"{verb!r} not sent: the hub's own version check "
                                        f"still waits for the board", verb=verb)
            time.sleep(BUSY_RETRY_S)


def _write_json_atomic(path: Path, data: dict) -> None:
    """Temp file in the same directory, then os.replace: a reader never sees half a file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=path.name + ".", suffix=".tmp")
    try:
        with os.fdopen(fd, "w") as f:
            f.write(json.dumps(data, indent=2) + "\n")
        os.replace(tmp, path)
    except BaseException:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise


def _close(*ports: Any) -> None:
    for p in ports:
        if p is None:
            continue
        try:
            p.close_port()
            # close_port() keeps the ALSA client; only delete() frees it,
            # and a board that re-enumerates every few minutes would
            # otherwise walk the client table to exhaustion.
            delete = getattr(p, "delete", None)
            if delete is not None:
                delete()
        except Exception:  # noqa: BLE001 - the device is already gone
            pass


def _default_find_board() -> Device | None:
    try:
        return select(discover(), None)
    except HilError:
        return None


def _open_hub_link(port: str) -> CdcLink:
    link = CdcLink(port, purpose="hub")
    link.open()
    return link


def run(log: Callable[[str], None] = print, *, ports: Ports | None = None,
        sleep: Callable[[float], None] = time.sleep, rounds: int | None = None,
        clock: Callable[[], float] = time.monotonic, lock_dir: Path | None = None,
        find_board: Callable[[], Device | None] | None = None,
        link_factory: Callable[[str], Any] | None = None) -> None:
    """Hold the hub open, arbitrate the board's port, follow the board and report to it.

    `rounds` bounds the loop for tests. A second hub finds the singleton lock
    taken and returns at once: the first one already serves everybody.
    """
    singleton = PortLock(HUB_LOCK_KEY, "hub", lock_dir=lock_dir)
    try:
        singleton.acquire()
    except HilError as e:
        log(f"another hub is running (pid {e.details.get('pid')}); nothing to do")
        return
    hub: Hub | None = None
    core: HubCore | None = None
    server: hub_ipc.Server | None = None
    try:
        launcher = daw_launcher.DawLauncher(log=log)
        host_info: Callable[[], str] | None = None
        try:
            cpu = host_report.CpuMeter()
            host_info = lambda: host_report.host_info_line(cpu, daws=launcher.masks())  # noqa: E731
        except OSError:
            log("no /proc here: the board gets no HOST_INFO, the port is arbitrated all the same")
        cache = fw_catalog.CatalogCache()
        holder: list[Hub] = []

        def latest() -> str | None:
            board = holder[0] if holder else None
            return host_report.latest_for(cache, (board and board.board_kit) or "full",
                                          (board and board.board_channel) or "stable")
        hub = Hub(ports or RtmidiPorts(), log, host_info=host_info,
                  latest=latest, launch=launcher.launch,
                  catalog_latest=lambda kit, ch: cache.latest(kit, ch),
                  catalog_held=lambda: cache.has_catalog)
        holder.append(hub)
        core = HubCore(find_board=find_board or _default_find_board,
                       link_factory=link_factory or _open_hub_link,
                       companion=hub.companion, clock=clock)
        server = hub_ipc.Server(core.handle, lock_dir=lock_dir)
        server.on_connect = core.connect
        server.on_disconnect = core.disconnect
        hub.open()
        server.start()
        n = 0
        while rounds is None or n < rounds:
            hub.poll()
            hub.report(clock())
            core.tick()
            sleep(POLL_S)
            n += 1
    finally:
        if server is not None:
            server.stop()
        if core is not None:
            core.close()
        if hub is not None:
            hub.close()
        singleton.release()


# -- install ---------------------------------------------------------------

def unit_text(exe: str) -> str:
    return f"""[Unit]
Description=CrossPad MIDI hub (fixed ALSA ports for DAWs under Wine)
After=sound.target

[Service]
ExecStart={exe} midi hub run
Restart=on-failure
RestartSec=2

[Install]
WantedBy=default.target
"""


def unit_path(home: Path | None = None) -> Path:
    return (home or Path.home()) / ".config/systemd/user" / UNIT_NAME


def install(exe: str | None = None, *, home: Path | None = None,
            systemctl: Callable[[list[str]], None] | None = None) -> dict:
    exe = exe or shutil.which("crosspad-hil") or sys.argv[0]
    if not exe or not Path(exe).is_absolute():
        raise HilError(BAD_ARGS, f"cannot resolve an absolute crosspad-hil path from {exe!r}",
                       hint="pass --exe /full/path/to/crosspad-hil")
    unit = unit_path(home)
    unit.parent.mkdir(parents=True, exist_ok=True)
    unit.write_text(unit_text(exe))
    run_systemctl = systemctl or _systemctl_user
    run_systemctl(["daemon-reload"])
    run_systemctl(["enable", "--now", UNIT_NAME])
    return {"unit": str(unit), "exe": exe, "ports": HUB_NAME}


def uninstall(*, home: Path | None = None,
              systemctl: Callable[[list[str]], None] | None = None) -> dict:
    unit = unit_path(home)
    run_systemctl = systemctl or _systemctl_user
    if unit.exists():
        run_systemctl(["disable", "--now", UNIT_NAME])
        unit.unlink()
    run_systemctl(["daemon-reload"])
    return {"unit": str(unit)}


def _systemctl_user(args: list[str]) -> None:
    subprocess.run(["systemctl", "--user", *args], check=True)


def status() -> dict:
    active = subprocess.run(["systemctl", "--user", "is-active", UNIT_NAME],
                            capture_output=True, text=True, check=False).stdout.strip()
    ports = RtmidiPorts()
    hub_in = any(HUB_NAME in n for n in ports.list_outputs())   # our readable side
    hub_out = any(HUB_NAME in n for n in ports.list_inputs())   # our writable side
    esp = [n for n in ports.list_inputs() if is_esp_port(n)]
    return {"service": active or "unknown", "hub_readable": hub_in, "hub_writable": hub_out,
            "board_ports": esp, "control": control_status()}


def control_status(lock_dir: Path | None = None) -> dict | None:
    """The running hub's own view (clients, exclusive op, CDC), or None when none answers.
    Never starts a hub."""
    try:
        conn = hub_ipc.Connection.open(lock_dir, client="crosspad-hil", purpose="status",
                                       timeout_s=STATUS_TIMEOUT_S)
    except HilError:
        return None
    try:
        return conn.call("status", timeout_s=STATUS_TIMEOUT_S)
    except HilError:
        return None
    finally:
        conn.close()
