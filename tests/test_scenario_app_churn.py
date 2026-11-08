"""Scenario `app_churn` against a stateful fake device (no hardware)."""
from __future__ import annotations

import threading
from pathlib import Path
from typing import Any

import pytest

from crosspad_hil.devices import Device, Ports, SerialPortInfo, UsbMode
from crosspad_hil.errors import CANCELLED, HilError
from crosspad_hil.scenarios import app_churn, base, get
from tests.fakes import AudioDesk, FakeSerial

CDC_PATH = "/dev/ttyACM0"
CONSOLE_PATH = "/dev/ttyACM1"
LAUNCHER = ["Sampler", "Mixer", "Leaky", "Settings", "Power OFF"]
GURU = "Guru Meditation Error: Core  1 panic'ed (StoreProhibited). Exception was unhandled."


class ChurnDevice(FakeSerial):
    """A CDC endpoint that behaves like hil_control.cpp for the churn verbs.

    APP_LIST reports the running app ("APPS: A,B running=X" / "running=-"),
    APP_START <name> answers OK (or "ERR unknown app") and makes it run,
    APP_STOP answers OK and returns to the launcher, MEM reports an internal
    heap that drops `leak_per_visit[app]` bytes every time that app is closed —
    a leak with a name attached. Anything else is echoed, as the device does.
    """

    def __init__(self, leak_per_visit: dict[str, int] | None = None) -> None:
        super().__init__(timeout=0.01)
        self.running: str | None = None
        self.int_free = 100_000
        self.leak_per_visit = leak_per_visit or {}
        self.visits: dict[str, int] = {}

    def write(self, data: bytes) -> int:
        cmd = data.decode("utf-8", "replace").strip()
        self.written.append(cmd)
        if cmd == "APP_LIST":
            self.feed([f"APPS: {','.join(LAUNCHER)} running={self.running or '-'}"])
        elif cmd.startswith("APP_START "):
            name = cmd.split(" ", 1)[1]
            if name not in LAUNCHER:
                self.feed(["ERR unknown app"])
            else:
                self.running = name
                self.feed(["OK"])
        elif cmd == "APP_STOP":
            if self.running is not None:
                self.visits[self.running] = self.visits.get(self.running, 0) + 1
                self.int_free -= self.leak_per_visit.get(self.running, 0)
                self.running = None
            self.feed(["OK"])
        elif cmd == "MEM":
            self.feed([f"MEM: int_free={self.int_free} int_largest=65536 int_min=90000 "
                       f"int_blocks=400 psram_free=7000000 psram_largest=6000000 "
                       f"psram_blocks=120"])
        elif cmd == "KIT_STATUS":
            self.feed(["KITSTATUS: current=0 loading=0 pending=-1 name=Basic Kit"])
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


def make_ctx(tmp_path: Path, dev: ChurnDevice) -> tuple[base.Context, FakeSerial]:
    con = FakeSerial(timeout=0.01)
    ports = {CDC_PATH: dev, CONSOLE_PATH: con}

    def factory(path: str, **kw: Any) -> FakeSerial:
        return ports[path]

    ctx = base.Context(device=make_device(), workdir=tmp_path,
                       cancelled=threading.Event(), log=lambda s: None,
                       serial_factory=factory)
    return ctx, con


FAST: dict[str, float] = {"dwell": 0.0, "settle": 0.0}


@pytest.fixture(autouse=True)
def lock_dir(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv("XDG_RUNTIME_DIR", str(tmp_path / "run"))


def test_registered_and_defaults() -> None:
    assert get("app_churn") is app_churn.SCENARIO
    assert app_churn.SCENARIO.name == "app_churn"
    p = app_churn.Params()
    # Eight rounds and 32 B/visit: three rounds cannot measure a leak at all,
    # and the bar has to sit below the ~56 B/visit one that was really there.
    assert (p.rounds, p.apps, p.skip, p.dwell, p.settle, p.leak_bytes) == \
        (8, None, None, 1.0, 1.2, 32)
    assert app_churn.DEFAULT_SKIP == ("Power OFF", "Firmware", "Settings")


def test_slope_rule() -> None:
    assert app_churn.slope([]) is None
    assert app_churn.slope([1, 2]) is None
    # head = mean(series[1:3]), tail = mean(series[-2:]), visits = len - 2
    assert app_churn.slope([100, 90, 80, 70, 60]) == (85.0, 65.0, 3, 20.0)
    assert app_churn.slope([100, 100, 100]) == (100.0, 100.0, 1, 0.0)


def test_clean_run_passes(tmp_path: Path) -> None:
    dev = ChurnDevice()
    ctx, con = make_ctx(tmp_path, dev)
    report = app_churn.SCENARIO.run(ctx, app_churn.Params(rounds=4, **FAST), base.Progress())
    assert report.passed is True
    assert report.exit_code == 0
    assert report.data["apps"] == LAUNCHER
    assert report.data["targets"] == ["Sampler", "Mixer", "Leaky"]
    assert report.data["skipped"] == ["Settings", "Power OFF"]
    assert report.data["leaks"] == []
    assert report.data["failures"] == []
    assert report.data["fatal"] == [] and report.data["reboots"] == 0
    assert all(len(s) == 4 for s in report.data["trend"].values())
    assert all(v["leak"] is False for v in report.data["slopes"].values())
    assert dev.visits == {"Sampler": 4, "Mixer": 4, "Leaky": 4}
    assert dev.running is None
    # The console was opened deasserted and never pulsed: no ESP reset on open.
    assert ("rts", True) not in con.control_history
    assert ("dtr", True) not in con.control_history
    assert "PASS — no crashes" in report.summary
    assert [a.role for a in report.artifacts] == ["console"]
    assert report.artifacts[0].path == str(tmp_path / "console.log")


def test_leak_detected(tmp_path: Path) -> None:
    dev = ChurnDevice(leak_per_visit={"Leaky": 5000})
    ctx, _con = make_ctx(tmp_path, dev)
    report = app_churn.SCENARIO.run(
        ctx, app_churn.Params(rounds=5, leak_bytes=2048, **FAST), base.Progress())
    assert report.passed is False
    assert report.exit_code == 1
    assert report.data["slopes"]["Leaky"]["leak"] is True
    assert report.data["slopes"]["Leaky"]["per_visit"] == 5000.0
    assert report.data["slopes"]["Sampler"]["leak"] is False
    assert report.data["slopes"]["Sampler"]["per_visit"] == 0.0
    assert len(report.data["leaks"]) == 1
    assert report.data["leaks"][0].startswith("Leaky: ~5000 B lost per visit")
    assert "<-- LEAK ~5000 B/visit" in report.summary
    assert "FAIL — 1 problem(s):" in report.summary


def test_leak_under_threshold_passes(tmp_path: Path) -> None:
    dev = ChurnDevice(leak_per_visit={"Mixer": 1000})
    ctx, _con = make_ctx(tmp_path, dev)
    report = app_churn.SCENARIO.run(
        ctx, app_churn.Params(rounds=5, leak_bytes=2048, **FAST), base.Progress())
    assert report.passed is True
    assert report.data["slopes"]["Mixer"]["per_visit"] == 1000.0
    assert report.data["slopes"]["Mixer"]["leak"] is False


def test_apps_subset_and_skip(tmp_path: Path) -> None:
    dev = ChurnDevice()
    ctx, _con = make_ctx(tmp_path, dev)
    report = app_churn.SCENARIO.run(
        ctx, app_churn.Params(rounds=3, apps=["Mixer"], **FAST), base.Progress())
    assert report.passed is True
    assert report.data["targets"] == ["Mixer"]
    assert report.data["skipped"] == []
    assert dev.visits == {"Mixer": 3}

    dev2 = ChurnDevice()
    ctx2, _con2 = make_ctx(tmp_path, dev2)
    report2 = app_churn.SCENARIO.run(
        ctx2, app_churn.Params(rounds=3, skip=["Sampler", "Mixer"], **FAST), base.Progress())
    assert report2.data["targets"] == ["Leaky", "Settings", "Power OFF"]
    assert report2.data["skipped"] == ["Sampler", "Mixer"]


def test_unknown_app_is_env_error(tmp_path: Path) -> None:
    dev = ChurnDevice()
    ctx, _con = make_ctx(tmp_path, dev)
    report = app_churn.SCENARIO.run(
        ctx, app_churn.Params(rounds=3, apps=["Nope"], **FAST), base.Progress())
    assert report.passed is False
    assert report.exit_code == 2
    assert "not in the launcher: Nope" in report.summary
    assert dev.visits == {}


def test_running_app_is_stopped_first(tmp_path: Path) -> None:
    dev = ChurnDevice()
    dev.running = "Mixer"
    ctx, _con = make_ctx(tmp_path, dev)
    app_churn.SCENARIO.run(ctx, app_churn.Params(rounds=1, **FAST), base.Progress())
    # Context.open_cdc probes the link with an APP_LIST of its own after a console
    # open, so the scenario's listing is not always the first line on the wire.
    # The order is what matters: it stops what was running before it starts
    # anything, and it learned there was something to stop from an APP_LIST.
    stop = dev.written.index("APP_STOP")
    assert dev.written[stop - 1] == "APP_LIST"
    assert not any(w.startswith("APP_START") for w in dev.written[:stop])


def test_fatal_on_console_fails_fast(tmp_path: Path) -> None:
    dev = ChurnDevice()
    ctx, con = make_ctx(tmp_path, dev)

    class PanicOnFirstVisit(base.Progress):
        """The console shows a panic while the first app is open."""

        def __call__(self, progress: int, total: int | None, message: str) -> None:
            if progress == 1:
                con.feed([GURU])

    report = app_churn.SCENARIO.run(
        ctx, app_churn.Params(rounds=5, **FAST), PanicOnFirstVisit())
    assert report.passed is False
    assert report.exit_code == 1
    assert report.data["fatal"] == [GURU]
    assert any("Sampler: device fault on round 1" in f for f in report.data["failures"])
    # It stopped at the first visit rather than churning on over a crash.
    assert sum(dev.visits.values()) == 1
    assert report.data["trend"]["Sampler"] == []


def test_reboot_without_fatal_line_fails(tmp_path: Path) -> None:
    dev = ChurnDevice()
    ctx, con = make_ctx(tmp_path, dev)

    class RebootOnSecondVisit(base.Progress):
        def __call__(self, progress: int, total: int | None, message: str) -> None:
            if progress == 2:
                con.feed(["ESP-ROM:esp32s3-20210327", "rst:0x7 (TG0WDT_SYS_RST),boot:0x8"])

    report = app_churn.SCENARIO.run(
        ctx, app_churn.Params(rounds=5, **FAST), RebootOnSecondVisit())
    assert report.passed is False
    assert report.data["reboots"] == 1
    assert any("Mixer: device rebooted on round 1" in f for f in report.data["failures"])
    assert sum(dev.visits.values()) == 2


def test_cancel_stops_between_visits(tmp_path: Path) -> None:
    dev = ChurnDevice()
    ctx, _con = make_ctx(tmp_path, dev)

    class CancelAfterTwo(base.Progress):
        def __call__(self, progress: int, total: int | None, message: str) -> None:
            if progress >= 2:
                ctx.cancelled.set()

    with pytest.raises(HilError) as ei:
        app_churn.SCENARIO.run(ctx, app_churn.Params(rounds=10, **FAST), CancelAfterTwo())
    assert ei.value.code == CANCELLED
    assert sum(dev.visits.values()) == 2


def test_an_error_mid_visit_still_closes_the_app(tmp_path: Path) -> None:
    """APP_STOP refused once mid-visit: the run fails with that error, and the
    app it opened is closed on the way out instead of being left on screen."""
    class BusyOnce(ChurnDevice):
        def __init__(self) -> None:
            super().__init__()
            self.refused = False

        def write(self, data: bytes) -> int:
            cmd = data.decode("utf-8", "replace").strip()
            if cmd == "APP_STOP" and self.running == "Mixer" and not self.refused:
                self.refused = True
                self.written.append(cmd)
                self.feed(["ERR busy"])
                return len(data)
            return super().write(data)

    dev = BusyOnce()
    ctx, _con = make_ctx(tmp_path, dev)
    with pytest.raises(HilError) as ei:
        app_churn.SCENARIO.run(ctx, app_churn.Params(rounds=1, apps=["Mixer"], **FAST),
                               base.Progress())
    assert "busy" in ei.value.message
    assert dev.written.count("APP_STOP") == 2
    assert dev.running is None


def test_a_failed_close_on_the_way_out_does_not_mask_the_error(tmp_path: Path) -> None:
    class NeverStops(ChurnDevice):
        def write(self, data: bytes) -> int:
            cmd = data.decode("utf-8", "replace").strip()
            if cmd == "APP_STOP" and self.running is not None:
                first = "APP_STOP" not in self.written
                self.written.append(cmd)
                self.feed(["ERR busy" if first else "ERR wedged"])
                return len(data)
            return super().write(data)

    dev = NeverStops()
    ctx, _con = make_ctx(tmp_path, dev)
    with pytest.raises(HilError) as ei:
        app_churn.SCENARIO.run(ctx, app_churn.Params(rounds=1, apps=["Mixer"], **FAST),
                               base.Progress())
    assert "busy" in ei.value.message          # the visit's error, not the cleanup's
    assert dev.written.count("APP_STOP") == 2


class GatedDevice(ChurnDevice):
    """A board with no current kit, which is what a cold boot looks like.

    The firmware puts the kit-selector gate in front of the Sampler until a kit
    is loaded, and APP_LIST reports running=- while the overlay is up — so the
    launch is refused and the churn blames the app for the bench's state.
    """

    def __init__(self) -> None:
        super().__init__()
        self.kit = -1

    def write(self, data: bytes) -> int:
        cmd = data.decode("utf-8", "replace").strip()
        if cmd == "KIT_STATUS":
            self.written.append(cmd)
            name = "-" if self.kit < 0 else "DRUMS"
            self.feed([f"KITSTATUS: current={self.kit} loading=0 pending=-1 name={name}"])
            return len(data)
        if cmd == "KIT_LIST":
            self.written.append(cmd)
            self.feed(["KITS: 0:Basic Kit,8:DRUMS current=-1"])
            return len(data)
        if cmd.startswith("KIT_LOAD "):
            self.written.append(cmd)
            self.kit = int(cmd.split()[1])
            self.feed(["OK"])
            return len(data)
        if cmd.startswith("APP_START ") and cmd.split(" ", 1)[1] == "Sampler" and self.kit < 0:
            self.written.append(cmd)
            self.feed(["APPS: X running=-"])       # the gate, not the app
            return len(data)
        return super().write(data)


def test_a_kit_is_seeded_so_the_sampler_is_launchable(tmp_path: Path) -> None:
    dev = GatedDevice()
    ctx, _con = make_ctx(tmp_path, dev)
    rep = app_churn.SCENARIO.run(ctx, app_churn.Params(rounds=2, **FAST), base.Progress())
    assert dev.kit >= 0, "no kit was seeded, so the Sampler could not launch"
    first_load = next(i for i, c in enumerate(dev.written) if c.startswith("KIT_LOAD"))
    first_start = next(i for i, c in enumerate(dev.written) if c.startswith("APP_START"))
    assert first_load < first_start
    assert not any("Sampler" in f and "refused" in f for f in rep.data["failures"])


def test_a_small_real_leak_is_found_at_the_default_rounds(tmp_path: Path) -> None:
    """~56 B/visit is the launcher's un-freed lv_group, and it was invisible.

    Three rounds put the head and tail windows zero visits apart, so per_visit
    divided by nothing and every app reported 0 B/visit however badly it leaked
    — the 2048 B threshold was not even reached for.
    """
    dev = ChurnDevice(leak_per_visit={"Leaky": 56})
    ctx, _con = make_ctx(tmp_path, dev)
    report = app_churn.SCENARIO.run(
        ctx, app_churn.Params(**FAST), base.Progress())
    assert report.passed is False
    assert any("Leaky" in leak for leak in report.data["leaks"]), report.summary
    assert report.data["slopes"]["Leaky"]["measurable"] is True


def test_a_run_too_short_to_measure_says_so_instead_of_passing(tmp_path: Path) -> None:
    """Silence is not a pass: "no leak found" must not read as "no leak"."""
    dev = ChurnDevice(leak_per_visit={"Leaky": 4096})
    ctx, _con = make_ctx(tmp_path, dev)
    report = app_churn.SCENARIO.run(
        ctx, app_churn.Params(rounds=3, **FAST), base.Progress())
    assert report.passed is True                     # nothing crashed
    assert report.data["leaks"] == []                # and nothing was measurable
    assert "Leaky" in report.data["unmeasured"]
    assert "INCONCLUSIVE" in report.summary


class DeskChurnDevice(ChurnDevice):
    """ChurnDevice plus the mixer/codec verbs; every APP_START of `meddler` moves the desk,
    the way a blind press on the volume overlay or the Mixer's mute does."""

    def __init__(self, desk: AudioDesk, meddler: str = "Mixer") -> None:
        super().__init__()
        self.desk = desk
        self.meddler = meddler

    def write(self, data: bytes) -> int:
        cmd = data.decode("utf-8", "replace").strip()
        lines = self.desk.reply(cmd)
        if lines is not None:
            self.written.append(cmd)
            self.feed(lines)
            return len(data)
        if cmd == f"APP_START {self.meddler}":
            self.desk.vol = [28, 50]
            self.desk.adc = [1, 2]
            self.desk.channels[0]["mute"] = 1
        return super().write(data)


def test_audio_state_a_visit_moved_is_put_back(tmp_path: Path) -> None:
    desk = AudioDesk()
    home = desk.state()
    ctx, _con = make_ctx(tmp_path, DeskChurnDevice(desk))
    logged: list[str] = []
    ctx.log = logged.append
    report = app_churn.SCENARIO.run(ctx, app_churn.Params(rounds=2, **FAST), base.Progress())
    assert report.passed is True
    st = desk.state()
    assert (st["vol"], st["adc"]) == (home["vol"], home["adc"])
    assert report.data["audio_state_restored"] == [
        "ADC input codec 0: 1 -> 2", "ADC input codec 1: 2 -> 1",
        "codec volume codec 0: 28 -> 64", "codec volume codec 1: 50 -> 64",
        "mixer ch 0 Sampler mute: 1 -> 0"]
    assert report.data["audio_state_unrestorable"] == []
    assert st["ch_mute"] == [0, 0]                     # MIX_MUTE put it back
    assert any("audio_state: restored codec volume codec 0: 28 -> 64" in line for line in logged)
    assert any("audio_state: restored mixer ch 0 Sampler mute: 1 -> 0" in line for line in logged)


def test_audio_state_untouched_when_nothing_moved(tmp_path: Path) -> None:
    desk = AudioDesk()
    ctx, _con = make_ctx(tmp_path, DeskChurnDevice(desk, meddler="Nobody"))
    report = app_churn.SCENARIO.run(ctx, app_churn.Params(rounds=2, **FAST), base.Progress())
    assert report.passed is True and desk.commands == []
    assert report.data["audio_state_restored"] == []


class ReloadingDevice(ChurnDevice):
    """Leaving an app that held the engine (on the board, Pitch Play giving its
    instrument back) reloads the kit: for a few KIT_STATUS polls the load runs
    and holds 400 B of the heap, then hands them back. A sample taken
    mid-reload would call that a leak of the app."""

    def __init__(self) -> None:
        super().__init__()
        self.reloading = 0

    def write(self, data: bytes) -> int:
        cmd = data.decode("utf-8", "replace").strip()
        if cmd == "APP_STOP" and self.running == "Sampler":
            self.reloading = 3
            self.int_free -= 400
        if cmd == "KIT_STATUS" and self.reloading:
            self.written.append(cmd)
            self.reloading -= 1
            if not self.reloading:
                self.int_free += 400
            self.feed([f"KITSTATUS: current=0 loading={1 if self.reloading else 0} "
                       f"pending=-1 name=Basic Kit"])
            return len(data)
        return super().write(data)


def test_a_kit_reloading_after_the_close_is_waited_for_not_called_a_leak(tmp_path: Path) -> None:
    dev = ReloadingDevice()
    ctx, _con = make_ctx(tmp_path, dev)
    rep = app_churn.SCENARIO.run(ctx, app_churn.Params(rounds=4, apps=["Sampler", "Mixer"], **FAST),
                                 base.Progress())
    assert rep.passed, rep.summary
    assert dev.written.count("KIT_STATUS") >= 4 * 3
