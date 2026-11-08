"""Scenario `back_churn` against a stateful fake device (no hardware).

The fake models what the scenario actually depends on: a screen stack. An app
start pushes a level, a press inside pushes another (with its own encoder
group), and a power click pops exactly one — which is the contract the firmware
fix established and the thing this scenario is here to hold. A device that pops
the whole stack on one click, or that stops answering, is modelled too, because
those are the two ways the real board failed.
"""
from __future__ import annotations

import threading
from pathlib import Path
from typing import Any

import pytest

from crosspad_hil.devices import Device, Ports, SerialPortInfo, UsbMode
from crosspad_hil.scenarios import back_churn, base, get
from tests.fakes import AudioDesk, FakeSerial

CDC_PATH = "/dev/ttyACM0"
CONSOLE_PATH = "/dev/ttyACM1"
LAUNCHER = ["Sampler", "Mixer", "Fishtank", "Settings", "Power OFF"]
GURU = "Guru Meditation Error: Core  1 panic'ed (LoadProhibited). Exception was unhandled."

FAST: dict[str, float] = {"dwell": 0.0, "settle": 0.0}


class StackDevice(FakeSerial):
    """A CDC endpoint with a screen stack: APP_START pushes, ENC_PRESS pushes,
    PWR_GESTURE CLICK pops one level.

    * ``pop_all``    — a click destroys the app whatever level it is on (the
      pre-fix firmware: the gesture never saw the levels).
    * ``die_at``     — stop answering once the stack is this deep and a click
      arrives (the panic, from the harness's side of the wire).
    * ``leak``       — internal bytes lost per completed visit, by app name.
    * ``kit_swap``   — the app whose first press loads another kit, which then
      holds ``kit_cost`` bytes until ``KIT_LOAD 0`` puts the old one back (the
      selector's footprint noise, which is not a leak).
    * ``tail_bytes`` — an async teardown still holding this much for the next
      ``tail_reads`` MEM replies after a visit ends (not a leak either).
    * ``deferred``   — ``(holder, releaser, bytes)``: a visit to ``holder``
      keeps ``bytes`` until ``releaser`` starts (the kit restore's deferred
      release; a leak in one column, a gain in the next, flat per round).
    * ``settings``   — answer BLE_STATUS / USB_GUARD / WIFI_STATUS from
      ``{"ble": 0|1, "guard": 0|1, "wifi": 0|1}`` (and take their setters);
      None echoes them like any verb this firmware does not know.
    """

    def __init__(self, *, pop_all: bool = False, die_at: int | None = None,
                 leak: dict[str, int] | None = None, depth_limit: int = 3,
                 kit_swap: str | None = None, kit_cost: int = 0,
                 tail_bytes: int = 0, tail_reads: int = 0,
                 deferred: tuple[str, str, int] | None = None,
                 settings: dict[str, int] | None = None) -> None:
        super().__init__(timeout=0.01)
        self.settings = settings
        self.deferred = deferred
        self.held = False
        self.tail_bytes = tail_bytes
        self.tail_reads = tail_reads
        self.tail_left = 0
        self.kit = 0
        self.kit_swap = kit_swap
        self.kit_cost = kit_cost
        self.stack: list[str] = []          # [app, level1, level2, ...]
        self.running: str | None = None
        self.int_free = 100_000
        self.pop_all = pop_all
        self.die_at = die_at
        self.leak = leak or {}
        self.depth_limit = depth_limit
        self.dead = False
        self.focus = 0
        self.visits: dict[str, int] = {}
        self.pad_hits = 0
        self.pad_played = 0
        self.pad_targets: set[str] = set()
        self.silent_pads = False      # hits reach the board, the engine makes none
        self.max_depth = 0
        self.coredump_seq: int | None = None

    # ── screen model ────────────────────────────────────────────────────
    def _group(self) -> list[str]:
        if not self.stack:
            return LAUNCHER
        return [f"{self.stack[-1]}.item{i}" for i in range(3)] + ["Close"]

    def _push(self, label: str) -> None:
        self.stack.append(label)
        self.focus = 0
        self.max_depth = max(self.max_depth, len(self.stack) - 1)

    def _finish_visit(self) -> None:
        if self.running:
            self.visits[self.running] = self.visits.get(self.running, 0) + 1
            self.int_free -= self.leak.get(self.running, 0)
            self.tail_left = self.tail_reads
            if self.deferred and self.running == self.deferred[0] and not self.held:
                self.int_free -= self.deferred[2]
                self.held = True
        self.running = None
        self.stack = []
        self.focus = 0

    # ── wire ────────────────────────────────────────────────────────────
    def write(self, data: bytes) -> int:  # noqa: C901 - a flat verb table
        cmd = data.decode("utf-8", "replace").strip()
        self.written.append(cmd)
        if self.dead:
            return len(data)
        group = self._group()
        if cmd == "APP_LIST":
            self.feed([f"APPS: {','.join(LAUNCHER)} running={self.running or '-'}"])
        elif cmd.startswith("APP_START "):
            name = cmd.split(" ", 1)[1]
            if name not in LAUNCHER:
                self.feed(["ERR unknown app"])
            else:
                if self.deferred and name == self.deferred[1] and self.held:
                    self.int_free += self.deferred[2]
                    self.held = False
                self.running = name
                self._push(name)
                self.feed(["OK"])
        elif cmd == "APP_STOP":
            self._finish_visit()
            self.feed(["OK"])
        elif cmd == "ENC_GROUP":
            lines = [f"ENCGROUP: count={len(group)}"]
            lines += [f"[{i}] 0x{(hash((tuple(self.stack), i)) & 0xffffff):06x} {lb}"
                      for i, lb in enumerate(group)]
            self.feed(lines)
        elif cmd == "ENC_FOCUS":
            self.feed([f"ENCFOCUS: obj=0x1000 idx={self.focus} "
                       f"text={group[self.focus % len(group)]}"])
        elif cmd.startswith("ENC_ROTATE"):
            self.focus = (self.focus + int(cmd.split()[1])) % len(group)
            self.feed(["OK"])
        elif cmd.startswith("ENC_PRESS"):
            label = group[self.focus % len(group)]
            if label == "Close":
                self.stack.pop()
                if not self.stack:
                    self._finish_visit()
            elif len(self.stack) - 1 < self.depth_limit:
                if self.running == self.kit_swap and self.kit == 0:
                    self.kit = 1
                    self.int_free -= self.kit_cost
                self._push(label)
            self.feed(["OK"])
        elif cmd.startswith("PWR_GESTURE"):
            if "HOLD" in cmd:
                self._push("drawer")
            elif self.stack:
                if self.die_at is not None and len(self.stack) >= self.die_at:
                    self.dead = True
                    return len(data)
                if self.pop_all:
                    self._finish_visit()
                else:
                    self.stack.pop()
                    if not self.stack:
                        self._finish_visit()
            self.feed(["OK"])
        elif cmd == "MEM":
            shown = self.int_free - (self.tail_bytes if self.tail_left > 0 else 0)
            self.tail_left = max(0, self.tail_left - 1)
            self.feed([f"MEM: int_free={shown} int_largest=65536 int_min=90000 "
                       f"int_blocks=400 psram_free=7000000 psram_largest=6000000 "
                       f"psram_blocks=120"])
        elif cmd == "COREDUMP_INFO":
            if self.coredump_seq is None:
                self.feed(["COREDUMP: none"])
            else:
                self.feed([f"COREDUMP: seq={self.coredump_seq} task=taskLVGL pc=0x4210b1e2 "
                           "cause=28 vaddr=0xffffffff depth=7", "bt: 0x4210b1e2"])
        elif cmd == "KIT_STATUS":
            self.feed([f"KITSTATUS: current={self.kit} loading=0 pending=-1 name=DRUMS"])
        elif cmd.startswith("KIT_LOAD "):
            wanted = int(cmd.split()[1])
            if wanted == 0 and self.kit == 1:
                self.int_free += self.kit_cost
            self.kit = wanted
            self.feed(["OK"])
        elif cmd.startswith("PAD_PRESS "):
            self.pad_hits += 1
            if not self.silent_pads:
                self.pad_played += 1
            if self.running:
                self.pad_targets.add(self.running)
            self.feed(["OK"])
        elif cmd.startswith("PAD_RELEASE "):
            self.feed(["OK"])
        elif cmd == "PAD_STATS":
            self.feed([f"PADSTATS: press={self.pad_hits} release={self.pad_hits} "
                       f"played={self.pad_played} freeslots=8"])
        elif cmd == "SMPL_PEAK":
            self.feed(["SMPLPEAK: 0 free=8"])
        elif self.settings is not None and cmd == "BLE_STATUS":
            self.feed([f"BLE: supported=1 running={self.settings['ble']} state=idle "
                       "mode=server self=- peer=- itvl=0"])
        elif self.settings is not None and cmd == "BLE_STOP":
            self.settings["ble"] = 0
            self.feed(["OK"])
        elif self.settings is not None and cmd.startswith("BLE_START"):
            self.settings["ble"] = 1
            self.feed(["OK"])
        elif self.settings is not None and cmd.startswith("USB_GUARD"):
            if cmd != "USB_GUARD":
                self.settings["guard"] = int(cmd.split()[1])
            self.feed([f"USB_GUARD: {self.settings['guard']}"])
        elif self.settings is not None and cmd == "WIFI_STATUS":
            self.feed([f'WIFI: state=off ssid="" rssi=0 ip=- leases=0 fail=0 reason=0 '
                       f"enabled={self.settings['wifi']}"])
        elif self.settings is not None and cmd.startswith("WIFI_ENABLE "):
            self.settings["wifi"] = int(cmd.split()[1])
            self.feed(["OK"])
        else:
            self.feed([cmd])
        return len(data)


def make_device() -> Device:
    return Device(
        id="dev_ab12", serial="CP-1", usb_mode=UsbMode.DEFAULT,
        ports=Ports(
            cdc=SerialPortInfo(path=CDC_PATH, vid=0x303A, pid=0x3456,
                               serial="CP-1", product="Crosspad", location="1-2.1"),
            console=SerialPortInfo(path=CONSOLE_PATH, vid=0x0483, pid=0x5740,
                                   serial="STM-1", product="CrossPad MIDI+Serial",
                                   location="1-2.2"),
        ),
    )


def make_ctx(tmp_path: Path, dev: StackDevice) -> tuple[base.Context, FakeSerial]:
    con = FakeSerial(timeout=0.01)
    ports = {CDC_PATH: dev, CONSOLE_PATH: con}

    def factory(path: str, **kw: Any) -> FakeSerial:
        return ports[path]

    ctx = base.Context(device=make_device(), workdir=tmp_path,
                       cancelled=threading.Event(), log=lambda s: None,
                       serial_factory=factory)
    return ctx, con


@pytest.fixture(autouse=True)
def lock_dir(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv("XDG_RUNTIME_DIR", str(tmp_path / "run"))
    # The fake answers MEM at once; the settle cadence only slows the tests.
    monkeypatch.setattr(back_churn, "HEAP_SETTLE_POLL_S", 0.0)


def test_registered_and_defaults() -> None:
    assert get("back_churn") is back_churn.SCENARIO
    assert back_churn.SCENARIO.name == "back_churn"
    p = back_churn.Params()
    assert (p.rounds, p.depth, p.clicks, p.apps, p.skip, p.cases, p.leak_bytes) == \
        (8, 3, 8, None, None, None, 32)
    # Settings is a target here, unlike in app_churn: it is the multi-level app.
    assert back_churn.DEFAULT_SKIP == ("Power OFF", "Firmware")


def test_deny_and_up_lists() -> None:
    # The blind descent must not press anything destructive...
    for label in ("Power OFF", "Delete sample", "Factory reset", "Format SD", "Update"):
        assert back_churn.DENY.search(label), label
    # ...and must not mistake a way out for a way in.
    for label in ("Back", "Close", "Go back...", "Cancel"):
        assert label.strip().lower() in back_churn.UP, label
    for label in ("Sampler", "Brightness", "MIDI", "Advanced", "Edit", "Browse"):
        assert not back_churn.DENY.search(label), label
    # Saved radio/USB settings: a blind toggle there outlives the step.
    for label in ("Bluetooth", "BLE MIDI", "Wireless", "WiFi", "Wi-Fi", "USB serial guard"):
        assert back_churn.DENY.search(label), label
    for label in ("Mixer", "Tables", "Enable pads"):
        assert not back_churn.DENY.search(label), label
    # A take, the Recorder's or the top bar's, writes to the card.
    for label in ("REC", "Rec", "REC MODE"):
        assert back_churn.DENY.search(label), label
    for label in ("Recent", "Reverb", "Preview"):
        assert not back_churn.DENY.search(label), label


def test_is_launcher() -> None:
    assert back_churn.is_launcher(LAUNCHER, LAUNCHER) is True
    assert back_churn.is_launcher(["Sampler", "Mixer", "Fishtank"], LAUNCHER) is True
    # Same number of focusables, none of them the launcher's: not home.
    assert back_churn.is_launcher(["a", "b", "c", "d", "e"], LAUNCHER) is False


def test_clean_run_passes(tmp_path: Path) -> None:
    dev = StackDevice()
    ctx, con = make_ctx(tmp_path, dev)
    report = back_churn.SCENARIO.run(
        ctx, back_churn.Params(rounds=4, cases=["none"], **FAST), base.Progress())
    assert report.passed is True
    assert report.exit_code == 0
    assert report.data["targets"] == ["Sampler", "Mixer", "Fishtank", "Settings"]
    assert report.data["skipped"] == ["Power OFF"]
    assert report.data["failures"] == [] and report.data["leaks"] == []
    # Three levels down, and one click per level plus one for the app itself.
    assert all(d == [3, 3, 3, 3] for d in report.data["depths"].values())
    assert all(c == [4, 4, 4, 4] for c in report.data["clicks"].values())
    assert dev.stack == [] and dev.running is None
    assert "PASS — every target unwound" in report.summary
    # Opened deasserted: no ESP reset from the console side.
    assert ("rts", True) not in con.control_history
    assert ("dtr", True) not in con.control_history


def test_kit_loaded_by_a_visit_is_put_back_before_sampling(tmp_path: Path) -> None:
    """A press in the selector loads a kit; its footprint is not the app's leak."""
    dev = StackDevice(kit_swap="Sampler", kit_cost=12_000)
    ctx, _con = make_ctx(tmp_path, dev)
    report = back_churn.SCENARIO.run(
        ctx, back_churn.Params(rounds=4, apps=["Sampler"], **FAST), base.Progress())
    assert report.passed is True
    assert report.data["kit_home"] == 0
    assert report.data["kit_restores_failed"] == 0
    assert report.data["leaks"] == []
    assert dev.written.count("KIT_LOAD 0") == 4 and dev.kit == 0
    # Sampled after the restore, every visit hands back exactly what it took.
    assert len(set(report.data["trend"]["Sampler"])) == 1


def test_async_tail_is_not_charged_to_the_visit(tmp_path: Path) -> None:
    """Bytes an async teardown still holds at sample time are not a leak."""
    dev = StackDevice(tail_bytes=660, tail_reads=2)
    ctx, _con = make_ctx(tmp_path, dev)
    report = back_churn.SCENARIO.run(
        ctx, back_churn.Params(rounds=4, depth=1, apps=["Mixer"], **FAST), base.Progress())
    assert report.passed is True and report.data["leaks"] == []
    assert len(set(report.data["trend"]["Mixer"])) == 1


def test_memory_handed_back_by_the_next_app_is_not_a_leak(tmp_path: Path) -> None:
    """A leak in one column mirrored by a gain in the next, flat per round."""
    dev = StackDevice(deferred=("Sampler", "Mixer", 650))
    ctx, _con = make_ctx(tmp_path, dev)
    report = back_churn.SCENARIO.run(
        ctx, back_churn.Params(rounds=4, depth=1, apps=["Sampler", "Mixer"], **FAST),
        base.Progress())
    assert report.passed is True and report.data["leaks"] == []
    assert report.data["slopes"]["Sampler"]["per_visit"] == 650.0
    assert report.data["slopes"]["Sampler"]["refunded"] is True
    assert report.data["slopes"]["Mixer"]["per_visit"] == -650.0
    assert report.data["round_slope"]["per_round"] == 0.0
    assert "handed back before the round ended" in report.summary


def test_one_click_per_level(tmp_path: Path) -> None:
    """Depth 1 costs two clicks, depth 3 costs four: the levels are real."""
    dev = StackDevice()
    ctx, _con = make_ctx(tmp_path, dev)
    report = back_churn.SCENARIO.run(
        ctx, back_churn.Params(rounds=1, depth=1, apps=["Mixer"], **FAST), base.Progress())
    assert report.data["depths"]["Mixer"] == [1]
    assert report.data["clicks"]["Mixer"] == [2]
    assert report.passed is True


def test_click_that_destroys_the_app_from_depth_is_not_a_pass(tmp_path: Path) -> None:
    """The pre-fix firmware: one click leaves the app from three levels down.

    It still reaches the launcher, so the run passes — but it does it in one
    click, and that is what the depth/clicks table has to make visible.
    """
    dev = StackDevice(pop_all=True)
    ctx, _con = make_ctx(tmp_path, dev)
    report = back_churn.SCENARIO.run(
        ctx, back_churn.Params(rounds=2, apps=["Sampler"], **FAST), base.Progress())
    assert report.data["depths"]["Sampler"] == [3, 3]
    assert report.data["clicks"]["Sampler"] == [1, 1]


def test_panic_is_a_failure(tmp_path: Path) -> None:
    dev = StackDevice(die_at=4)
    ctx, _con = make_ctx(tmp_path, dev)
    report = back_churn.SCENARIO.run(
        ctx, back_churn.Params(rounds=2, apps=["Sampler"], **FAST), base.Progress())
    assert report.passed is False
    assert report.exit_code == 1
    assert "stopped answering" in report.data["failures"][0]


def test_stuck_is_a_failure(tmp_path: Path) -> None:
    """Clicks that never reach the launcher fail, and do not stall the run."""
    dev = StackDevice()
    dev.pop_all = False
    ctx, _con = make_ctx(tmp_path, dev)
    report = back_churn.SCENARIO.run(
        ctx, back_churn.Params(rounds=1, depth=3, clicks=2, apps=["Sampler"], **FAST),
        base.Progress())
    assert report.passed is False
    assert "2 clicks did not reach the launcher" in report.data["failures"][0]
    # Recovered with APP_STOP so the next target starts from the launcher.
    assert dev.running is None


def test_coredump_between_visits_fails(tmp_path: Path) -> None:
    class DumpingDevice(StackDevice):
        def write(self, data: bytes) -> int:
            n = super().write(data)
            if data.decode(errors="replace").strip().startswith("PWR_GESTURE"):
                self.coredump_seq = 7
            return n

    dev = DumpingDevice()
    ctx, _con = make_ctx(tmp_path, dev)
    report = back_churn.SCENARIO.run(
        ctx, back_churn.Params(rounds=2, apps=["Mixer"], **FAST), base.Progress())
    assert report.passed is False
    assert "core dump appeared" in report.data["failures"][0]


def test_a_visit_that_ends_early_still_puts_the_kit_back(tmp_path: Path) -> None:
    """A core dump ends the run straight after the visit that changed the kit;
    the finally still loads kit 0 back for the gate's next step."""
    class DumpingDevice(StackDevice):
        def write(self, data: bytes) -> int:
            n = super().write(data)
            if data.decode(errors="replace").strip().startswith("PWR_GESTURE"):
                self.coredump_seq = 7
            return n

    dev = DumpingDevice(kit_swap="Sampler", kit_cost=12_000)
    ctx, _con = make_ctx(tmp_path, dev)
    report = back_churn.SCENARIO.run(
        ctx, back_churn.Params(rounds=2, apps=["Sampler"], **FAST), base.Progress())
    assert "core dump appeared" in report.data["failures"][0]
    assert dev.kit == 0


def test_leak_is_named(tmp_path: Path) -> None:
    dev = StackDevice(leak={"Fishtank": 5000})
    ctx, _con = make_ctx(tmp_path, dev)
    report = back_churn.SCENARIO.run(
        ctx, back_churn.Params(rounds=5, cases=["none"], leak_bytes=2048, **FAST),
        base.Progress())
    assert report.passed is False
    assert report.data["slopes"]["Fishtank"]["leak"] is True
    assert report.data["slopes"]["Fishtank"]["per_visit"] == 5000.0
    assert report.data["slopes"]["Mixer"]["leak"] is False
    assert report.data["leaks"][0].startswith("Fishtank: ~5000 B lost per visit")


def test_shallow_run_says_it_proved_nothing(tmp_path: Path) -> None:
    """A device with no levels passes, but not silently."""
    dev = StackDevice(depth_limit=0)
    ctx, _con = make_ctx(tmp_path, dev)
    report = back_churn.SCENARIO.run(
        ctx, back_churn.Params(rounds=4, apps=["Mixer"], **FAST), base.Progress())
    assert report.passed is True
    assert "NOT exercised" in report.summary


def test_drawer_case_is_clicked_shut(tmp_path: Path) -> None:
    dev = StackDevice()
    ctx, _con = make_ctx(tmp_path, dev)
    report = back_churn.SCENARIO.run(
        ctx, back_churn.Params(rounds=1, cases=["drawer"], apps=None,
                               skip=["Sampler", "Mixer", "Fishtank", "Settings", "Power OFF"],
                               **FAST),
        base.Progress())
    assert "drawer" in report.data["targets"]
    assert report.data["clicks"]["drawer"] == [1]
    assert report.passed is True


def test_pads_play_inside_the_sampler(tmp_path: Path) -> None:
    dev = StackDevice()
    ctx, _con = make_ctx(tmp_path, dev)
    report = back_churn.SCENARIO.run(
        ctx, back_churn.Params(rounds=3, cases=["none"], pads=2, **FAST),
        base.Progress())
    assert report.passed is True
    assert dev.pad_hits == 6                    # two hits on each of three visits
    assert dev.pad_targets == {"Sampler"}       # and on no other app
    assert "6 hit(s),    6 played" in report.summary


class HandOffDevice(StackDevice):
    """A Sampler level that hands the pad to another app, as a pad's menu opens
    the Slicer or Pitch Play: the press ends the Sampler and starts the Mixer,
    whose pads the sampler engine never plays."""

    def write(self, data: bytes) -> int:
        cmd = data.decode("utf-8", "replace").strip()
        if (cmd.startswith("ENC_PRESS") and self.running == "Sampler"
                and self._group()[self.focus % len(self._group())] == "Sampler.item1"):
            self.written.append(cmd)
            self.running = "Mixer"
            self.stack = ["Mixer"]
            self.feed(["OK"])
            return len(data)
        if cmd.startswith("PAD_PRESS ") and self.running != "Sampler":
            self.written.append(cmd)
            self.pad_hits += 1
            self.pad_targets.add(str(self.running))
            self.feed(["OK"])
            return len(data)
        return super().write(data)


def test_pads_are_not_played_in_an_app_the_sampler_handed_off_to(tmp_path: Path) -> None:
    dev = HandOffDevice()
    ctx, _con = make_ctx(tmp_path, dev)
    report = back_churn.SCENARIO.run(
        ctx, back_churn.Params(rounds=3, cases=["none"], pads=2, **FAST),
        base.Progress())
    assert report.passed is True
    assert dev.pad_targets == {"Sampler"}       # round 2 ended in the Mixer
    assert dev.pad_hits == 4
    assert "4 hit(s),    4 played   (1 visit(s) left for Mixer)" in report.summary


def test_pads_can_be_turned_off(tmp_path: Path) -> None:
    dev = StackDevice()
    ctx, _con = make_ctx(tmp_path, dev)
    report = back_churn.SCENARIO.run(
        ctx, back_churn.Params(rounds=2, cases=["none"], pads=0, **FAST),
        base.Progress())
    assert report.passed is True
    assert dev.pad_hits == 0
    assert "pads played" not in report.summary


def test_pads_that_never_sound_are_a_failure(tmp_path: Path) -> None:
    dev = StackDevice()
    dev.silent_pads = True
    ctx, _con = make_ctx(tmp_path, dev)
    report = back_churn.SCENARIO.run(
        ctx, back_churn.Params(rounds=3, cases=["none"], pads=2, **FAST),
        base.Progress())
    assert report.passed is False
    assert "played none" in report.summary


# ── sampler_explorer on the device-look Sampler ──────────────────────────────
# Ported from #4 (fix/back-churn-explorer), which #26 superseded; the fake now
# follows what #26 found on dev_1ede: the pad editor opens on no pad, a pad hit
# while it is open shows that pad, and only then does the file button open
# the card's browser.

EDIT = "EDIT "
SAMPLE = " kick.wav"
NO_SAMPLE = " pick a s..."
BACK_ARROW = ""
FOLDER = "C3B0 - Black Tesla"


class SamplerDevice(StackDevice):
    """The Sampler as the device-look layout draws it: EDIT on the panel opens the
    pad editor, which shows no pad until one is hit; its file button is named
    after the sample and does nothing while there is no pad."""

    def __init__(self) -> None:
        super().__init__()
        self.pad_shown = False

    def _group(self) -> list[str]:
        top = self.stack[-1] if self.stack else None
        if top == "Sampler":
            return [BACK_ARROW, "DRUMS", "FULL LEVEL", "16 LEVELS", EDIT]
        if top == EDIT:
            return [BACK_ARROW, "PAD 1" if self.pad_shown else "PAD -",
                    SAMPLE if self.pad_shown else NO_SAMPLE, "REC MODE", "START"]
        if top == SAMPLE:
            # The explorer's back arrow first: an icon-only label is not a folder.
            return [BACK_ARROW, FOLDER, "kick.wav", "Close", "Load"]
        return super()._group()

    def write(self, data: bytes) -> int:
        cmd = data.decode("utf-8", "replace").strip()
        if cmd.startswith("APP_START "):
            self.pad_shown = False
        elif cmd.startswith("PAD_PRESS ") and self.stack[-1:] in (["Sampler"], [EDIT]):
            self.pad_shown = True
        elif cmd.startswith("ENC_PRESS") and not self.dead:
            group = self._group()
            if group[self.focus % len(group)] == NO_SAMPLE:
                self.written.append(cmd)
                self.feed(["OK"])   # no pad on screen: the file button does nothing
                return len(data)
        return super().write(data)


def test_sampler_explorer_walks_edit_and_the_file_button(tmp_path: Path) -> None:
    """EDIT, a pad hit so the file button has a pad, the file button, then a folder
    (not the explorer's back arrow); unwound back to the launcher."""
    dev = SamplerDevice()
    ctx, _con = make_ctx(tmp_path, dev)
    report = back_churn.SCENARIO.run(
        ctx, back_churn.Params(rounds=1, apps=["Sampler"], cases=["sampler_explorer"],
                               **FAST),
        base.Progress())
    assert report.data["failures"] == []
    assert report.data["paths"]["sampler_explorer"] == ["EDIT", "the sample button", FOLDER]
    assert report.data["depths"]["sampler_explorer"] == [3]
    assert report.data["clicks"]["sampler_explorer"] == [4]     # one per level, the app's too
    assert dev.pad_shown is True
    assert dev.stack == [] and dev.running is None


def test_settings_a_visit_flips_are_put_back(tmp_path: Path) -> None:
    """A blind press that brings BLE up and drops the USB guard is undone."""
    class FlippingDevice(StackDevice):
        def write(self, data: bytes) -> int:
            if data.decode(errors="replace").strip() == "APP_START Settings":
                self.settings = {**(self.settings or {}), "ble": 1, "guard": 0}
            return super().write(data)

    dev = FlippingDevice(settings={"ble": 0, "guard": 1, "wifi": 0})
    ctx, _con = make_ctx(tmp_path, dev)
    report = back_churn.SCENARIO.run(
        ctx, back_churn.Params(rounds=1, depth=1, apps=["Settings"], **FAST), base.Progress())
    assert report.passed is True
    assert report.data["settings_home"] == {"ble_running": False, "ble_mode": "server",
                                            "usb_guard": True, "wifi_enabled": False}
    assert report.data["settings_restore_failed"] == []
    assert dev.settings == {"ble": 0, "guard": 1, "wifi": 0}
    assert "BLE_STOP" in dev.written and "USB_GUARD 1" in dev.written
    assert not any(c.startswith("WIFI_ENABLE") for c in dev.written)


def test_settings_nothing_written_when_nothing_changed(tmp_path: Path) -> None:
    dev = StackDevice(settings={"ble": 1, "guard": 1, "wifi": 1})
    ctx, _con = make_ctx(tmp_path, dev)
    report = back_churn.SCENARIO.run(
        ctx, back_churn.Params(rounds=1, depth=1, apps=["Mixer"], **FAST), base.Progress())
    assert report.passed is True
    assert report.data["settings_home"]["ble_running"] is True
    assert report.data["settings_restore_failed"] == []
    setters = [c for c in dev.written
               if c.startswith(("BLE_START", "BLE_STOP", "WIFI_ENABLE", "USB_GUARD "))]
    assert setters == []


def test_settings_restore_failure_is_recorded_not_raised(tmp_path: Path) -> None:
    """A refused BLE_START is logged and recorded; the run's verdict stands."""
    class RefusingDevice(StackDevice):
        def write(self, data: bytes) -> int:
            cmd = data.decode(errors="replace").strip()
            if cmd == "APP_START Mixer":
                self.settings = {**(self.settings or {}), "ble": 0}
            if cmd.startswith("BLE_START"):
                self.written.append(cmd)
                self.feed(["ERR no_mem"])
                return len(data)
            return super().write(data)

    dev = RefusingDevice(settings={"ble": 1, "guard": 0, "wifi": 0})
    ctx, _con = make_ctx(tmp_path, dev)
    report = back_churn.SCENARIO.run(
        ctx, back_churn.Params(rounds=1, depth=1, apps=["Mixer"], **FAST), base.Progress())
    assert report.passed is True
    assert report.data["settings_restore_failed"] == ["ble"]
    assert "BLE_START 1" in dev.written


class DeskStackDevice(StackDevice):
    """StackDevice plus the mixer/codec verbs; every app start moves the desk."""

    def __init__(self, desk: AudioDesk) -> None:
        super().__init__()
        self.desk = desk

    def write(self, data: bytes) -> int:
        cmd = data.decode("utf-8", "replace").strip()
        lines = self.desk.reply(cmd)
        if lines is not None:
            self.written.append(cmd)
            self.feed(lines)
            return len(data)
        if cmd.startswith("APP_START "):
            self.desk.vol = [28, 50]
            self.desk.in_pair = 1
            self.desk.channels[0]["route"] = [0, 1, 1]
        return super().write(data)


def test_audio_state_a_blind_walk_moved_is_put_back(tmp_path: Path) -> None:
    desk = AudioDesk()
    home = desk.state()
    dev = DeskStackDevice(desk)
    ctx, _con = make_ctx(tmp_path, dev)
    logged: list[str] = []
    ctx.log = logged.append
    report = back_churn.SCENARIO.run(
        ctx, back_churn.Params(rounds=2, cases=["none"], **FAST), base.Progress())
    assert report.passed is True
    assert desk.state() == home
    assert sorted(report.data["audio_state_restored"]) == [
        "IN pair: 1 -> 0", "codec volume codec 0: 28 -> 64", "codec volume codec 1: 50 -> 64",
        "mixer ch 0 Sampler send to out 0: 0 -> 1"]
    assert report.data["audio_state_unrestorable"] == [] and report.data["audio_state_restore_failed"] == []
    assert any("audio_state: restored IN pair: 1 -> 0" in line for line in logged)
