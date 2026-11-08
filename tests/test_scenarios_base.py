"""Scenario framework: registry, argparse bridge, Context, run_scenario."""
from __future__ import annotations

import argparse
import json
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pytest

from crosspad_hil.devices import Device, Ports, SerialPortInfo, UsbMode
from crosspad_hil.errors import CANCELLED, ENV, NO_CDC_IN_AUDIO_MODE, NO_DEVICE, HilError
from crosspad_hil.scenarios import base
from crosspad_hil.scenarios.base import (
    Artifact,
    Context,
    Progress,
    Report,
    argparse_to_params,
    get,
    names,
    params_schema,
    params_to_argparse,
    register,
    run_scenario,
)
from tests.fakes import FakeSerial

# ── fixtures ────────────────────────────────────────────────────────────────


def make_device(usb_mode: UsbMode = UsbMode.DEFAULT, with_cdc: bool = True) -> Device:
    cdc = SerialPortInfo(
        path="/dev/fake-cdc", vid=0x303A, pid=0x3456, serial="ABC123",
        product="Crosspad", location="1-2.3",
    ) if with_cdc else None
    console = SerialPortInfo(
        path="/dev/fake-console", vid=0x0483, pid=0x5740, serial="STM1",
        product="CrossPad MIDI+Serial", location="1-2.4",
    )
    return Device(id="dev_test", serial="ABC123", usb_mode=usb_mode,
                  ports=Ports(cdc=cdc, console=console))


@pytest.fixture(autouse=True)
def _lock_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("XDG_RUNTIME_DIR", str(tmp_path / "run"))
    (tmp_path / "run").mkdir()


@dataclass
class DummyParams:
    rounds: int = 3
    dwell: float = 1.5
    label: str | None = None
    kits: list[int] | None = None
    apps: list[str] | None = field(default=None, metadata={"help": "apps to cycle"})
    no_play: bool = False
    verbose: bool = field(default=True, metadata={"help": "chatty"})


class DummyScenario:
    name = "dummy"
    Params = DummyParams
    description = "a scenario that does nothing on hardware"

    def __init__(self) -> None:
        self.calls: list[tuple[int, int | None, str]] = []

    def run(self, ctx: Context, params: DummyParams, progress: Progress) -> Report:
        for i in range(params.rounds):
            ctx.check_cancelled()
            progress(i + 1, params.rounds, f"round {i + 1}")
        return Report(passed=True, summary=f"{params.rounds} rounds", data={"rounds": params.rounds},
                      artifacts=[Artifact(path="console.log", mime="text/plain", role="log")],
                      exit_code=0)


class RaisingScenario:
    name = "raising"
    Params = DummyParams
    description = "raises what its params say"

    def run(self, ctx: Context, params: DummyParams, progress: Progress) -> Report:
        ctx.partial = {"rounds_done": 0, "error": "from the scenario"}
        ctx.partial["rounds_done"] = 6      # by reference: seen after the raise
        raise HilError(params.label or ENV, "boom", hint="fix the bench")


class FailingScenario:
    name = "failing"
    Params = DummyParams
    description = "returns a failed report"

    def run(self, ctx: Context, params: DummyParams, progress: Progress) -> Report:
        return Report(passed=False, summary="3 of 5 hits missing", data={}, artifacts=[], exit_code=1)


class CrashingScenario:
    name = "crashing"
    Params = DummyParams
    description = "throws a plain exception"

    def run(self, ctx: Context, params: DummyParams, progress: Progress) -> Report:
        ctx.partial = {"rounds_done": 6}
        raise ValueError("unexpected")


@pytest.fixture
def registered() -> DummyScenario:
    s = DummyScenario()
    register(s)
    register(RaisingScenario())
    register(FailingScenario())
    register(CrashingScenario())
    return s


# ── registry ────────────────────────────────────────────────────────────────


def test_register_get_names(registered: DummyScenario) -> None:
    assert get("dummy") is registered
    assert "dummy" in names()
    assert names() == sorted(names())


def test_get_unknown_raises_bad_args() -> None:
    with pytest.raises(HilError) as ei:
        get("no-such-scenario")
    assert ei.value.code == "BAD_ARGS"
    assert "no-such-scenario" in ei.value.message


def test_register_replaces_same_name() -> None:
    a = DummyScenario()
    b = DummyScenario()
    register(a)
    register(b)
    assert get("dummy") is b


# ── argparse bridge ──────────────────────────────────────────────────────────


def test_params_to_argparse_defaults() -> None:
    p = argparse.ArgumentParser()
    params_to_argparse(DummyParams, p)
    ns = p.parse_args([])
    params = argparse_to_params(DummyParams, ns)
    assert params == DummyParams()


def test_params_to_argparse_all_flag_kinds() -> None:
    p = argparse.ArgumentParser()
    params_to_argparse(DummyParams, p)
    ns = p.parse_args([
        "--rounds", "7", "--dwell", "0.25", "--label", "x",
        "--kits", "8,12,16", "--apps", "Sampler,Sequencer", "--no-play", "--no-verbose",
    ])
    params = argparse_to_params(DummyParams, ns)
    assert params == DummyParams(rounds=7, dwell=0.25, label="x", kits=[8, 12, 16],
                                 apps=["Sampler", "Sequencer"], no_play=True, verbose=False)


def test_bool_flags_have_both_forms() -> None:
    p = argparse.ArgumentParser()
    params_to_argparse(DummyParams, p)
    assert argparse_to_params(DummyParams, p.parse_args(["--verbose"])).verbose is True
    assert argparse_to_params(DummyParams, p.parse_args(["--no-play"])).no_play is True
    assert argparse_to_params(DummyParams, p.parse_args(["--no-no-play"])).no_play is False


def test_help_metadata_and_default_in_help() -> None:
    p = argparse.ArgumentParser()
    params_to_argparse(DummyParams, p)
    text = p.format_help()
    assert "apps to cycle" in text
    assert "--kebab" not in text
    assert "--no-play" in text
    assert "(default: 3)" in text


def test_empty_list_flag_gives_empty_list() -> None:
    p = argparse.ArgumentParser()
    params_to_argparse(DummyParams, p)
    assert argparse_to_params(DummyParams, p.parse_args(["--kits", ""])).kits == []


def test_params_schema() -> None:
    schema = params_schema(DummyParams)
    by_name = {row["name"]: row for row in schema}
    assert by_name["rounds"] == {"name": "rounds", "type": "int", "default": 3, "help": ""}
    assert by_name["kits"]["type"] == "list[int] | None"
    assert by_name["apps"]["help"] == "apps to cycle"
    assert by_name["verbose"]["default"] is True


def test_unsupported_annotation_is_type_error() -> None:
    @dataclass
    class Bad:
        weird: dict[str, int] = field(default_factory=dict)

    with pytest.raises(TypeError):
        params_to_argparse(Bad, argparse.ArgumentParser())


# ── Context ──────────────────────────────────────────────────────────────────


def test_context_open_cdc_uses_injected_factory(tmp_path: Path) -> None:
    created: list[FakeSerial] = []

    def factory(path: str, **kw: Any) -> FakeSerial:
        fake = FakeSerial(script=[
            ("KIT_STATUS", "KITSTATUS: current=3 loading=0 pending=-1 name=DRUMS"),
        ])
        fake.port = path
        created.append(fake)
        return fake

    ctx = Context(device=make_device(), workdir=tmp_path, cancelled=threading.Event(),
                  log=lambda s: None, serial_factory=factory)
    link = ctx.open_cdc()
    try:
        reply = link.transact("KIT_STATUS")
        assert reply.parsed is not None
        assert reply.parsed["current"] == 3
        assert reply.parsed["name"] == "DRUMS"
    finally:
        link.close()
    assert created and created[0].port == "/dev/fake-cdc"


def test_context_open_console_logs_into_workdir(tmp_path: Path) -> None:
    created: list[FakeSerial] = []

    def factory(path: str, **kw: Any) -> FakeSerial:
        fake = FakeSerial()
        created.append(fake)
        return fake

    ctx = Context(device=make_device(), workdir=tmp_path, cancelled=threading.Event(),
                  log=lambda s: None, serial_factory=factory)
    con = ctx.open_console()
    try:
        created[0].feed(["I (100) main: Platform fully initialized"])
        res = con.read(wait_ms=500, match="Platform fully")
        assert any("Platform fully initialized" in line for _, line in res.lines)
    finally:
        con.close()
    assert (tmp_path / "console.log").exists()
    # hygiene: opening the console never asserted DTR or RTS on the fake
    assert ("dtr", True) not in created[0].control_history
    assert ("rts", True) not in created[0].control_history


def test_context_open_cdc_in_audio_mode() -> None:
    ctx = Context(device=make_device(UsbMode.AUDIO, with_cdc=False), workdir=Path("."),
                  cancelled=threading.Event(), log=lambda s: None)
    with pytest.raises(HilError) as ei:
        ctx.open_cdc()
    assert ei.value.code == NO_CDC_IN_AUDIO_MODE


def test_context_open_cdc_without_port() -> None:
    ctx = Context(device=make_device(with_cdc=False), workdir=Path("."),
                  cancelled=threading.Event(), log=lambda s: None)
    with pytest.raises(HilError) as ei:
        ctx.open_cdc()
    assert ei.value.code == NO_DEVICE


def test_check_cancelled() -> None:
    ev = threading.Event()
    ctx = Context(device=make_device(), workdir=Path("."), cancelled=ev, log=lambda s: None)
    ctx.check_cancelled()
    ev.set()
    with pytest.raises(HilError) as ei:
        ctx.check_cancelled()
    assert ei.value.code == CANCELLED


# ── run_scenario ─────────────────────────────────────────────────────────────


def test_run_scenario_writes_report(tmp_path: Path, registered: DummyScenario) -> None:
    progress_calls: list[tuple[int, int | None, str]] = []

    class P(Progress):
        def __call__(self, progress: int, total: int | None, message: str) -> None:
            progress_calls.append((progress, total, message))

    logs: list[str] = []
    workdir = tmp_path / "run1"
    report = run_scenario("dummy", DummyParams(rounds=2), device=make_device(), workdir=workdir,
                          progress=P(), log=logs.append)
    assert report.passed is True
    assert report.exit_code == 0
    assert progress_calls == [(1, 2, "round 1"), (2, 2, "round 2")]
    doc = json.loads((workdir / "report.json").read_text(encoding="utf-8"))
    assert doc["scenario"] == "dummy"
    assert doc["passed"] is True
    assert doc["exit_code"] == 0
    assert doc["params"] == {"rounds": 2, "dwell": 1.5, "label": None, "kits": None,
                             "apps": None, "no_play": False, "verbose": True}
    assert doc["data"] == {"rounds": 2}
    roles = {a["role"] for a in doc["artifacts"]}
    assert roles == {"log", "report"}
    assert report.artifacts[-1].path == str(workdir / "report.json")
    assert doc["seconds"] >= 0


def test_run_scenario_default_workdir(tmp_path: Path, registered: DummyScenario,
                                      monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(tmp_path)
    report = run_scenario("dummy", DummyParams(rounds=1), device=make_device(), log=lambda s: None)
    report_path = Path(report.artifacts[-1].path)
    assert report_path.parent.parent == tmp_path / "hil_logs"
    assert report_path.parent.name.startswith("dummy_")
    assert len(report_path.parent.name) == len("dummy_") + len("20260826_120000")


def test_run_scenario_failed_report_exit_1(tmp_path: Path, registered: DummyScenario) -> None:
    report = run_scenario("failing", DummyParams(), device=make_device(), workdir=tmp_path,
                          log=lambda s: None)
    assert report.passed is False
    assert report.exit_code == 1


def test_run_scenario_env_error_exit_2(tmp_path: Path, registered: DummyScenario) -> None:
    report = run_scenario("raising", DummyParams(label=ENV), device=make_device(),
                          workdir=tmp_path, log=lambda s: None)
    assert report.passed is False
    assert report.exit_code == 2
    assert report.data["error"]["code"] == ENV
    assert report.data["error"]["hint"] == "fix the bench"
    doc = json.loads((tmp_path / "report.json").read_text(encoding="utf-8"))
    assert doc["exit_code"] == 2


def test_run_scenario_no_device_is_env(tmp_path: Path, registered: DummyScenario) -> None:
    report = run_scenario("raising", DummyParams(label=NO_DEVICE), device=make_device(),
                          workdir=tmp_path, log=lambda s: None)
    assert report.exit_code == 2


def test_run_scenario_other_hil_error_exit_1(tmp_path: Path, registered: DummyScenario) -> None:
    report = run_scenario("raising", DummyParams(label="TIMEOUT"), device=make_device(),
                          workdir=tmp_path, log=lambda s: None)
    assert report.exit_code == 1
    assert report.data["error"]["code"] == "TIMEOUT"


def test_run_scenario_cancelled(tmp_path: Path, registered: DummyScenario) -> None:
    ev = threading.Event()
    ev.set()
    report = run_scenario("dummy", DummyParams(rounds=5), device=make_device(), workdir=tmp_path,
                          cancelled=ev, log=lambda s: None)
    assert report.passed is False
    assert report.exit_code == 1
    assert report.data["cancelled"] is True
    assert report.data["error"]["code"] == CANCELLED
    assert report.summary.startswith("cancelled")


def test_run_scenario_crash_still_writes_report(tmp_path: Path,
                                                registered: DummyScenario) -> None:
    report = run_scenario("crashing", DummyParams(), device=make_device(), workdir=tmp_path,
                          log=lambda s: None)
    assert report.exit_code == 1
    assert report.data["error"]["code"] == "INTERNAL"
    assert "ValueError" in report.data["error"]["message"]
    assert (tmp_path / "report.json").exists()


def test_run_scenario_unknown_name() -> None:
    with pytest.raises(HilError) as ei:
        run_scenario("nope", DummyParams(), device=make_device())
    assert ei.value.code == "BAD_ARGS"


def test_run_scenario_discovers_when_no_device(tmp_path: Path, registered: DummyScenario,
                                               monkeypatch: pytest.MonkeyPatch) -> None:
    dev = make_device()
    monkeypatch.setattr(base, "discover", lambda backends=None: [dev])
    report = run_scenario("dummy", DummyParams(rounds=1), workdir=tmp_path, log=lambda s: None)
    assert report.passed is True
    doc = json.loads((tmp_path / "report.json").read_text(encoding="utf-8"))
    assert doc["device"] == "dev_test"


def test_report_to_dict() -> None:
    r = Report(passed=True, summary="ok", data={"a": 1},
               artifacts=[Artifact(path="x.wav", mime="audio/wav", role="capture")], exit_code=0)
    assert r.to_dict() == {"passed": True, "summary": "ok", "data": {"a": 1},
                           "artifacts": [{"path": "x.wav", "mime": "audio/wav", "role": "capture"}],
                           "exit_code": 0}


def test_package_exports_and_load_builtin() -> None:
    import crosspad_hil.scenarios as pkg

    assert pkg.register is register and pkg.get is get and pkg.names is names
    # Pin the P0 five by containment, not by equality: the catalog grows a name
    # whenever a scenario is planned, and load_builtin() skips one whose module
    # is not written yet — so equality here would fail on intent, not on a bug.
    assert set(pkg.BUILTIN_MODULES) >= {"smoke", "app_churn", "kit_churn", "led_state", "usb_mode_cycle"}
    assert len(set(pkg.BUILTIN_MODULES)) == len(pkg.BUILTIN_MODULES)
    loaded = pkg.load_builtin()
    assert set(loaded) <= set(pkg.BUILTIN_MODULES)


def test_a_second_run_on_the_same_board_is_refused(tmp_path: Path,
                                                    registered: DummyScenario) -> None:
    """Two scenarios on one board collide even when they share no port.

    A scenario that switches the USB profile changes which ports the board has,
    so port locks alone let a second run tear the first one's device state out
    from under it.
    """
    from crosspad_hil.locks import PortLock

    dev = make_device()
    held = PortLock(f"device:{dev.id}", "scenario other")
    held.acquire()
    try:
        rep = run_scenario("dummy", DummyParams(rounds=1), device=dev,
                           workdir=tmp_path / "run", log=lambda _m: None)
    finally:
        held.release()

    assert rep.exit_code == 2 and not rep.passed
    assert rep.data["error"]["code"] == "PORT_BUSY"
    assert (tmp_path / "run" / "report.json").exists()


# ── partial data survives a run that dies mid-way ────────────────────────────


def test_partial_data_survives_a_hilerror(tmp_path: Path, registered: DummyScenario) -> None:
    report = run_scenario("raising", DummyParams(label="TIMEOUT"), device=make_device(),
                          workdir=tmp_path, log=lambda s: None)
    assert report.exit_code == 1
    assert report.data["rounds_done"] == 6
    # The failure is the report's own, never whatever the scenario happened to
    # leave under that key.
    assert report.data["error"]["code"] == "TIMEOUT"


def test_partial_data_survives_an_env_error(tmp_path: Path, registered: DummyScenario) -> None:
    report = run_scenario("raising", DummyParams(label=ENV), device=make_device(),
                          workdir=tmp_path, log=lambda s: None)
    assert report.exit_code == 2
    assert report.data["rounds_done"] == 6
    assert report.data["error"]["code"] == ENV


def test_partial_data_survives_cancellation(tmp_path: Path, registered: DummyScenario) -> None:
    report = run_scenario("raising", DummyParams(label=CANCELLED), device=make_device(),
                          workdir=tmp_path, log=lambda s: None)
    assert report.data["cancelled"] is True
    assert report.data["rounds_done"] == 6


def test_partial_data_survives_a_crash(tmp_path: Path, registered: DummyScenario) -> None:
    report = run_scenario("crashing", DummyParams(), device=make_device(),
                          workdir=tmp_path, log=lambda s: None)
    assert report.exit_code == 1
    assert report.data["rounds_done"] == 6
    assert report.data["error"]["code"] == "INTERNAL"


def test_a_scenario_that_publishes_nothing_reports_only_the_error(
        tmp_path: Path, registered: DummyScenario) -> None:
    report = run_scenario("failing", DummyParams(), device=make_device(),
                          workdir=tmp_path, log=lambda s: None)
    assert report.data == {}


# ── the hub's exclusive op around a run ─────────────────────────────────────


def test_run_scenario_holds_an_exclusive_op_for_the_whole_run(
        tmp_path: Path, registered: DummyScenario, exclusive_log) -> None:
    report = run_scenario("dummy", DummyParams(rounds=1), device=make_device(),
                          workdir=tmp_path / "r", log=lambda s: None)
    assert report.passed
    assert exclusive_log == [("take", "scenario dummy"), ("release", "scenario dummy")]


def test_run_scenario_releases_the_exclusive_op_when_the_scenario_crashes(
        tmp_path: Path, registered: DummyScenario, exclusive_log) -> None:
    run_scenario("crashing", DummyParams(), device=make_device(), workdir=tmp_path / "r",
                 log=lambda s: None)
    assert exclusive_log[-1] == ("release", "scenario crashing")


class SharedScenario(DummyScenario):
    name = "shared"
    exclusive = False


def test_a_scenario_that_only_sends_verbs_takes_no_exclusive_op(
        tmp_path: Path, exclusive_log) -> None:
    register(SharedScenario())
    run_scenario("shared", DummyParams(rounds=1), device=make_device(),
                 workdir=tmp_path / "r", log=lambda s: None)
    assert exclusive_log == []


@pytest.mark.uses_hub
def test_context_open_cdc_goes_through_the_hub_outside_an_exclusive_op(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from crosspad_hil import hub_link

    class Conn:
        closed = False

        def call(self, op, args=None, timeout_s=10.0):
            return {"line": "KITSTATUS: current=3 loading=0 pending=-1 name=DRUMS",
                    "parsed": {"name": "DRUMS"}, "rtt_ms": 1.0, "more_lines": [],
                    "extra_lines": []}

        def close(self):
            return None

    monkeypatch.setattr(hub_link, "connect_or_start", lambda purpose, **kw: Conn())
    ctx = Context(device=make_device(UsbMode.AUDIO, with_cdc=False), workdir=tmp_path,
                  cancelled=threading.Event(), log=lambda s: None,
                  serial_factory=lambda *a, **k: FakeSerial())
    link = ctx.open_cdc()
    assert isinstance(link, hub_link.HubLink)
    assert link.transact("KIT_STATUS").parsed["name"] == "DRUMS"
    ctx.close_all()
