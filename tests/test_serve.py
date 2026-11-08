"""Daemon tests: handle() dispatch, events, tasks, handle expiry, stdio round trip."""
from __future__ import annotations

import io
import json
import threading
import time
from dataclasses import dataclass, field

import pytest

from crosspad_hil import serve
from crosspad_hil.devices import Backends, SerialPortInfo, discover
from crosspad_hil.errors import BAD_ARGS, HANDLE_EXPIRED, HilError
from crosspad_hil.knowledge import load
from crosspad_hil.scenarios import base as scen
from tests.fakes import FakeSerial

CDC = "/dev/ttyACM0"
CON = "/dev/ttyACM1"


def _backends() -> Backends:
    ports = [
        SerialPortInfo(path=CDC, vid=0x303A, pid=0x3456, serial="AB12CD34",
                       product="CrossPad", location="1-1.2:1.0"),
        SerialPortInfo(path=CON, vid=0x0483, pid=0x5740, serial="STM0001",
                       product="CrossPad MIDI+Serial", location="1-1.3:1.0"),
    ]
    return Backends(list_serial=lambda: list(ports), list_midi=lambda: [],
                    list_audio=lambda: [])


@pytest.fixture
def lock_dir(monkeypatch, tmp_path):
    monkeypatch.setenv("XDG_RUNTIME_DIR", str(tmp_path))
    return tmp_path


class Clock:
    def __init__(self) -> None:
        self.t = 1000.0

    def __call__(self) -> float:
        return self.t


@dataclass
class Rig:
    daemon: serve.Daemon
    out: io.StringIO
    fakes: dict[str, FakeSerial]
    clock: Clock
    dev_id: str = ""
    lock: threading.Lock = field(default_factory=threading.Lock)

    def events(self, name: str) -> list[dict]:
        evs = []
        for line in self.out.getvalue().splitlines():
            obj = json.loads(line)
            if obj.get("ev") == name:
                evs.append(obj)
        return evs

    def wait(self, pred, timeout: float = 3.0) -> bool:
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if pred():
                return True
            time.sleep(0.02)
        return False


def _rig(lock_dir, cdc_script=(), **kw) -> Rig:
    fakes = {CDC: FakeSerial(list(cdc_script)), CON: FakeSerial([])}
    out = io.StringIO()
    clock = Clock()

    def factory(path: str, **_: object) -> FakeSerial:
        return fakes[path]

    d = serve.Daemon(io.StringIO(), out, backends=_backends(), serial_factory=factory,
                     clock=clock, handle_ttl_s=60.0, **kw)
    dev_id = discover(_backends())[0].id
    return Rig(d, out, fakes, clock, dev_id)


def _ok(resp: dict) -> dict:
    assert resp["ok"] is True, resp
    return resp["result"]


def _err(resp: dict) -> dict:
    assert resp["ok"] is False, resp
    return resp["error"]


# ---------------------------------------------------------------- HandleRegistry

def test_registry_mint_get_expire():
    clock = Clock()
    closed: list[tuple[str, object]] = []
    reg = serve.HandleRegistry(clock=clock, on_expire=lambda h, o: closed.append((h, o)))
    h1 = reg.mint("con", "A", ttl_s=10)
    h2 = reg.mint("con", "B", ttl_s=None)
    t1 = reg.mint("task", "T", ttl_s=None)
    assert (h1, h2, t1) == ("con_1", "con_2", "task_1")
    assert reg.get(h1) == "A"
    clock.t += 11
    with pytest.raises(HilError) as ei:
        reg.get(h1)
    assert ei.value.code == HANDLE_EXPIRED and ei.value.hint == "open again"
    assert closed == [(h1, "A")]
    assert reg.get(h2) == "B"
    with pytest.raises(HilError) as ei:
        reg.get("con_99")
    assert ei.value.code == HANDLE_EXPIRED
    assert reg.drop(h2) == "B" and reg.drop(h2) is None
    assert [e["handle"] for e in reg.list()] == [t1]


def test_registry_sweep_and_touch():
    clock = Clock()
    reg = serve.HandleRegistry(clock=clock)
    h = reg.mint("cdc", object(), ttl_s=5)
    clock.t += 4
    reg.touch(h)
    clock.t += 4
    assert reg.sweep() == []
    clock.t += 2
    assert reg.sweep() == [h]
    assert reg.list() == []


# ---------------------------------------------------------------- dispatch basics

def test_ping_and_unknown_op(lock_dir):
    rig = _rig(lock_dir)
    r = _ok(rig.daemon.handle({"id": 1, "op": "serve.ping", "args": {}}))
    assert r["version"] and r["uptime_s"] >= 0
    e = _err(rig.daemon.handle({"id": 2, "op": "nope.op"}))
    assert e["code"] == BAD_ARGS and "serve.ping" in e["hint"]
    e = _err(rig.daemon.handle({"id": 3, "args": {}}))
    assert e["code"] == BAD_ARGS


def test_devices_list_and_doctor(lock_dir):
    rig = _rig(lock_dir)
    r = _ok(rig.daemon.handle({"id": 1, "op": "devices.list"}))
    assert r["devices"][0]["id"] == rig.dev_id
    r = _ok(rig.daemon.handle({"id": 2, "op": "devices.doctor"}))
    assert {c["name"] for c in r["checks"]} >= {"python", "devices", "locks"}


def test_knowledge_get(lock_dir):
    rig = _rig(lock_dir)
    r = _ok(rig.daemon.handle({"id": 1, "op": "knowledge.get", "args": {"name": "cdc"}}))
    assert r == load("cdc")
    assert "APP_LIST" in r["verbs"]
    e = _err(rig.daemon.handle({"id": 2, "op": "knowledge.get", "args": {"name": "nope"}}))
    assert e["code"] == BAD_ARGS and e["hint"] == "one of: cdc, sysex, markers"
    e = _err(rig.daemon.handle({"id": 3, "op": "knowledge.get", "args": {}}))
    assert e["code"] == BAD_ARGS


def test_cdc_verb_ephemeral_and_handle(lock_dir):
    rig = _rig(lock_dir, [("APP_LIST", "APPS: Sampler,Settings running=-"),
                          ("KIT_STATUS", "KITSTATUS: current=2 loading=0 pending=-1 name=X")])
    r = _ok(rig.daemon.handle({"id": 1, "op": "cdc.verb",
                               "args": {"device": rig.dev_id, "verb": "app_list", "args": {}}}))
    assert r == {"apps": ["Sampler", "Settings"], "running": None}
    h = _ok(rig.daemon.handle({"id": 2, "op": "cdc.open", "args": {"device": rig.dev_id}}))["handle"]
    assert h == "cdc_1"
    r = _ok(rig.daemon.handle({"id": 3, "op": "cdc.transact",
                               "args": {"handle": h, "cmd": "KIT_STATUS"}}))
    assert r["parsed"]["current"] == 2 and r["line"].startswith("KITSTATUS:")
    e = _err(rig.daemon.handle({"id": 4, "op": "cdc.verb",
                                "args": {"handle": h, "verb": "no_such_verb", "args": {}}}))
    assert e["code"] == BAD_ARGS
    e = _err(rig.daemon.handle({"id": 5, "op": "cdc.verb",
                                "args": {"handle": h, "verb": "pad_press", "args": {"idx": 99}}}))
    assert e["code"] == BAD_ARGS
    assert _ok(rig.daemon.handle({"id": 6, "op": "cdc.close", "args": {"handle": h}})) == {"ok": True}
    e = _err(rig.daemon.handle({"id": 7, "op": "cdc.transact", "args": {"handle": h, "cmd": "MEM"}}))
    assert e["code"] == HANDLE_EXPIRED
    assert rig.fakes[CDC].dtr is False and rig.fakes[CDC].rts is False


# ---------------------------------------------------------------- console + events

def test_console_open_fatal_event(lock_dir):
    rig = _rig(lock_dir)
    r = _ok(rig.daemon.handle({"id": 1, "op": "console.open", "args": {"device": rig.dev_id}}))
    assert r["handle"] == "con_1" and r["port"] == CON
    con = rig.fakes[CON]
    assert con.dtr is False and con.rts is False  # hygiene: STM VCP never asserted
    con.feed(["I (1234) main: hello",
              "Guru Meditation Error: Core  0 panic'ed (LoadProhibited). Exception was unhandled."])
    assert rig.wait(lambda: rig.events("console.fatal"))
    ev = rig.events("console.fatal")[0]
    assert ev["handle"] == "con_1" and ev["pattern"] == "Guru Meditation"
    assert "panic'ed" in ev["line"] and isinstance(ev["seq"], int)
    r = _ok(rig.daemon.handle({"id": 2, "op": "console.read",
                               "args": {"handle": "con_1", "match": "hello"}}))
    assert len(r["lines"]) == 1 and r["lines"][0][1].endswith("hello")
    r = _ok(rig.daemon.handle({"id": 3, "op": "console.snapshot", "args": {"handle": "con_1"}}))
    assert len(r["fatals"]) == 1
    assert _ok(rig.daemon.handle({"id": 4, "op": "console.close", "args": {"handle": "con_1"}}))


def test_console_reboot_and_cdc_drop_events(lock_dir):
    rig = _rig(lock_dir)
    _ok(rig.daemon.handle({"id": 1, "op": "console.open", "args": {"device": rig.dev_id}}))
    con = rig.fakes[CON]
    con.feed(["ESP-ROM:esp32s3-20210327",
              "W (5000) CDC: 12 commands dropped",
              "ESP-ROM:esp32s3-20210327"])
    assert rig.wait(lambda: rig.events("console.cdc_drops"))
    assert rig.events("console.cdc_drops")[0]["dropped"] == 12
    assert rig.wait(lambda: rig.events("console.reboot"))
    reboots = rig.events("console.reboot")
    assert reboots[-1]["handle"] == "con_1"
    assert isinstance(reboots[-1]["count"], int) and reboots[-1]["count"] >= len(reboots) - 1
    assert reboots[-1]["line"].startswith("ESP-ROM:esp32s3")


def test_console_reset_pulses_and_releases(lock_dir):
    rig = _rig(lock_dir)
    _ok(rig.daemon.handle({"id": 1, "op": "console.open", "args": {"device": rig.dev_id}}))
    _ok(rig.daemon.handle({"id": 2, "op": "console.reset", "args": {"handle": "con_1"}}))
    con = rig.fakes[CON]
    assert ("rts", True) in con.control_history
    assert con.rts is False and con.dtr is False


# ---------------------------------------------------------------- handle expiry

def test_handle_expiry(lock_dir):
    rig = _rig(lock_dir)
    _ok(rig.daemon.handle({"id": 1, "op": "console.open", "args": {"device": rig.dev_id}}))
    rig.clock.t += 61
    e = _err(rig.daemon.handle({"id": 2, "op": "console.read", "args": {"handle": "con_1"}}))
    assert e["code"] == HANDLE_EXPIRED and e["hint"] == "open again"
    assert rig.fakes[CON].is_open is False  # the sweep closed it


# ---------------------------------------------------------------- scenario tasks

@dataclass
class FakeParams:
    rounds: int = 3
    label: str = "x"


class FakeScenario:
    name = "fake_scn"
    Params = FakeParams
    description = "test scenario"

    def run(self, ctx, params, progress):  # pragma: no cover - replaced by run_scenario_fn
        raise AssertionError("not used")


def _fake_run(started: threading.Event, release: threading.Event):
    def run(name, params, *, device=None, workdir=None, progress=None, cancelled=None, log=print):
        started.set()
        for i in range(1, params.rounds + 1):
            if cancelled.is_set():
                raise HilError("CANCELLED", "cancelled by request")
            progress(i, params.rounds, f"round {i}/{params.rounds}")
            release.wait(2.0)
        return scen.Report(passed=True, summary=f"{name} ok", data={"rounds": params.rounds},
                           artifacts=[scen.Artifact(path="report.json", mime="application/json",
                                                    role="report")], exit_code=0)
    return run


def test_scenario_list_and_task_lifecycle(lock_dir):
    scen.register(FakeScenario())
    started, release = threading.Event(), threading.Event()
    release.set()
    rig = _rig(lock_dir, run_scenario_fn=_fake_run(started, release))
    r = _ok(rig.daemon.handle({"id": 1, "op": "scenario.list"}))
    entry = next(s for s in r["scenarios"] if s["name"] == "fake_scn")
    assert entry["description"] == "test scenario"
    assert {p["name"] for p in entry["params"]} == {"rounds", "label"}
    assert next(p for p in entry["params"] if p["name"] == "rounds")["default"] == 3

    r = _ok(rig.daemon.handle({"id": 2, "op": "scenario.run",
                               "args": {"name": "fake_scn", "params": {"rounds": 2},
                                        "device": rig.dev_id}}))
    task = r["task"]
    assert task == "task_1"
    r = _ok(rig.daemon.handle({"id": 3, "op": "task.wait", "args": {"task": task, "timeout_s": 3}}))
    assert r["status"] == "completed"
    assert r["result"]["passed"] is True and r["result"]["data"]["rounds"] == 2
    assert r["progress"] == 2 and r["total"] == 2
    progress = rig.events("task.progress")
    assert [p["progress"] for p in progress] == [1, 2]
    done = rig.events("task.done")
    assert done and done[0]["task"] == task and done[0]["status"] == "completed"
    assert done[0]["result"]["artifacts"][0]["path"] == "report.json"
    assert _ok(rig.daemon.handle({"id": 4, "op": "task.list"}))["tasks"][0]["task"] == task


def test_scenario_bad_params_and_cancel(lock_dir):
    scen.register(FakeScenario())
    started, release = threading.Event(), threading.Event()
    rig = _rig(lock_dir, run_scenario_fn=_fake_run(started, release))
    e = _err(rig.daemon.handle({"id": 1, "op": "scenario.run",
                                "args": {"name": "fake_scn", "params": {"bogus": 1}}}))
    assert e["code"] == BAD_ARGS
    e = _err(rig.daemon.handle({"id": 2, "op": "scenario.run", "args": {"name": "no_such"}}))
    assert e["code"] == BAD_ARGS

    task = _ok(rig.daemon.handle({"id": 3, "op": "scenario.run",
                                  "args": {"name": "fake_scn", "params": {"rounds": 5}}}))["task"]
    assert started.wait(2.0)
    r = _ok(rig.daemon.handle({"id": 4, "op": "task.status", "args": {"task": task}}))
    assert r["status"] == "working"
    assert _ok(rig.daemon.handle({"id": 5, "op": "task.cancel", "args": {"task": task}})) == {"ok": True}
    release.set()
    r = _ok(rig.daemon.handle({"id": 6, "op": "task.wait", "args": {"task": task, "timeout_s": 3}}))
    assert r["status"] == "cancelled" and r["error"]["code"] == "CANCELLED"
    e = _err(rig.daemon.handle({"id": 7, "op": "task.status", "args": {"task": "task_42"}}))
    assert e["code"] == HANDLE_EXPIRED


# ---------------------------------------------------------------- stdio round trip

def test_run_stdio_pipe(lock_dir):
    reqs = [
        {"id": 1, "op": "serve.ping", "args": {}},
        {"id": 2, "op": "devices.list"},
        "this is not json",
        {"id": 3, "op": "serve.shutdown"},
    ]
    stdin = io.StringIO("".join((json.dumps(r) if isinstance(r, dict) else r) + "\n" for r in reqs))
    out = io.StringIO()
    fakes = {CDC: FakeSerial([]), CON: FakeSerial([])}
    d = serve.Daemon(stdin, out, backends=_backends(), serial_factory=lambda p, **_: fakes[p])
    d.SWEEP_S = 0.05
    t = threading.Thread(target=d.run, daemon=True)
    t.start()
    t.join(5.0)
    assert not t.is_alive(), "daemon did not stop on serve.shutdown"
    lines = [json.loads(line) for line in out.getvalue().splitlines()]
    by_id = {r.get("id"): r for r in lines if "id" in r}
    assert by_id[1]["ok"] and by_id[1]["result"]["version"]
    assert by_id[2]["ok"] and by_id[2]["result"]["devices"]
    assert by_id[3] == {"id": 3, "ok": True, "result": {"ok": True}}
    assert by_id[None]["ok"] is False and by_id[None]["error"]["code"] == BAD_ARGS


# ---------------------------------------------------------------- serve.stats

def test_stats_reports_version_ops_and_fds(lock_dir):
    rig = _rig(lock_dir)
    _ok(rig.daemon.handle({"id": 1, "op": "serve.ping", "args": {}}))
    r = _ok(rig.daemon.handle({"id": 2, "op": "serve.stats", "args": {}}))
    assert r["version"] and r["pid"] > 0
    assert r["uptime_s"] >= 0 and r["threads"] >= 1
    assert r["ops_total"] == 2
    assert r["handles"] == {} and r["handles_total"] == 0
    assert r["open_fds"] >= 0 and r["alsa_seq_clients"] >= 0


def test_stats_counts_live_handles_by_kind(lock_dir):
    rig = _rig(lock_dir)
    h = _ok(rig.daemon.handle({"id": 1, "op": "console.open",
                               "args": {"device": rig.dev_id}}))["handle"]
    r = _ok(rig.daemon.handle({"id": 2, "op": "serve.stats", "args": {}}))
    assert r["handles"] == {"con": 1} and r["handles_total"] == 1
    _ok(rig.daemon.handle({"id": 3, "op": "console.close", "args": {"handle": h}}))
    r = _ok(rig.daemon.handle({"id": 4, "op": "serve.stats", "args": {}}))
    assert r["handles"] == {} and r["handles_total"] == 0


def test_stats_counts_failed_ops_too(lock_dir):
    rig = _rig(lock_dir)
    _err(rig.daemon.handle({"id": 1, "op": "nope.op"}))
    r = _ok(rig.daemon.handle({"id": 2, "op": "serve.stats", "args": {}}))
    assert r["ops_total"] == 2


# ---------------------------------------------------------------- the hub

def test_hub_status_without_a_hub(lock_dir):
    r = _rig(lock_dir)
    assert _ok(r.daemon.handle({"id": 1, "op": "hub.status", "args": {}})) == {"running": False}


def test_usbmode_set_runs_under_an_exclusive_op(lock_dir, exclusive_log):
    seen: list = []

    def set_mode(dev, mode, **kw):
        seen.append(list(exclusive_log))
        return dev

    r = _rig(lock_dir, set_mode_fn=set_mode)
    _ok(r.daemon.handle({"id": 1, "op": "usbmode.set", "args": {"mode": "audio"}}))
    assert seen == [[("take", "usb-mode set")]]
    assert exclusive_log[-1] == ("release", "usb-mode set")


class _FakeStim:
    running = True

    class plan:
        @staticmethod
        def to_dict():
            return {"hits": 1}

    def stop(self):
        self.running = False
        return {"sent": 1}

    def status(self, refresh=True):
        return {"sent": 1}


def test_stim_with_its_own_link_holds_the_exclusive_op_until_stop(lock_dir, exclusive_log):
    r = _rig(lock_dir, stim_start_fn=lambda dev, **kw: _FakeStim())
    h = _ok(r.daemon.handle({"id": 1, "op": "stim.start", "args": {"pads": "0"}}))["handle"]
    assert exclusive_log == [("take", "stim")]
    _ok(r.daemon.handle({"id": 2, "op": "stim.stop", "args": {"handle": h}}))
    assert exclusive_log == [("take", "stim"), ("release", "stim")]


def test_bench_refusal_comes_before_the_hub(lock_dir, monkeypatch):
    from crosspad_hil import hub_link

    monkeypatch.delenv(hub_link.NO_HUB_ENV, raising=False)

    def no_hub(*a, **kw):
        raise AssertionError("the hub was asked before the bench lease was checked")

    monkeypatch.setattr(hub_link, "connect_or_start", no_hub)
    r = _rig(lock_dir)
    _ok(r.daemon.handle({"id": 1, "op": "bench.claim",
                         "args": {"device": r.dev_id, "holder": "pidf"}}))
    for op, args in (("cdc.transact", {"cmd": "APP_LIST"}), ("usbmode.set", {"mode": "audio"})):
        err = _err(r.daemon.handle({"id": 2, "op": op, "args": {**args, "holder": "twin"}}))
        assert err["code"] == "BENCH_BUSY" and err["details"]["holder"] == "pidf"


def test_fs_push_runs_under_an_exclusive_op(lock_dir, exclusive_log, monkeypatch, tmp_path):
    seen: list = []

    def push(dev, local, remote, **kw):
        seen.append(list(exclusive_log))
        return {"bytes": 3, "remote": remote}

    monkeypatch.setattr(serve.fs_mod, "push", push)
    f = tmp_path / "a.txt"
    f.write_text("abc")
    r = _rig(lock_dir)
    res = _ok(r.daemon.handle({"id": 1, "op": "fs.push",
                               "args": {"local": str(f), "remote": "/sdcard/a.txt"}}))
    assert res == {"bytes": 3, "remote": "/sdcard/a.txt"}
    assert seen == [[("take", "fs transfer")]]
