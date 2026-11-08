"""CLI tests: argparse tree, injected backends/serial, exit codes. No hardware."""
from __future__ import annotations

import io
import json
import wave
from contextlib import contextmanager
from pathlib import Path

import pytest

from crosspad_hil import cli
from crosspad_hil.devices import Backends, SerialPortInfo
from crosspad_hil.scenarios.base import Report
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


def _deps(script, **kw) -> tuple[cli.CliDeps, io.StringIO, io.StringIO, FakeSerial]:
    fake = FakeSerial(script)
    out, err = io.StringIO(), io.StringIO()

    def factory(path: str, **_: object) -> FakeSerial:
        return fake

    deps = cli.CliDeps(backends=_backends(), serial_factory=factory,
                       stdout=out, stderr=err, sleep=lambda s: None, **kw)
    return deps, out, err, fake


def test_devices_json(lock_dir):
    deps, out, _, _ = _deps([])
    rc = cli.main(["devices", "--json"], deps=deps)
    assert rc == 0
    data = json.loads(out.getvalue())
    assert len(data["devices"]) == 1
    dev = data["devices"][0]
    assert dev["usb_mode"] == "default"
    assert dev["ports"]["cdc"]["path"] == CDC
    assert dev["ports"]["console"]["path"] == CON
    assert dev["id"].startswith("dev_")


def test_devices_human(lock_dir):
    deps, out, _, _ = _deps([])
    assert cli.main(["devices"], deps=deps) == 0
    text = out.getvalue()
    assert "dev_" in text and CDC in text and "default" in text


def test_cdc_app_list(lock_dir):
    deps, out, _, fake = _deps([("APP_LIST", "APPS: Sampler,Sequencer,Settings running=-")])
    rc = cli.main(["cdc", "APP_LIST", "--json"], deps=deps)
    assert rc == 0
    data = json.loads(out.getvalue())
    assert data["line"].startswith("APPS:")
    assert data["parsed"]["apps"] == ["Sampler", "Sequencer", "Settings"]
    assert data["parsed"]["running"] is None
    assert "APP_LIST" in fake.written
    # hygiene: the ESP CDC is opened with DTR/RTS deasserted
    assert fake.dtr is False and fake.rts is False


def test_app_list_typed(lock_dir):
    deps, out, _, _ = _deps([("APP_LIST", "APPS: Sampler,Sequencer running=Sampler")])
    assert cli.main(["--json", "app", "list"], deps=deps) == 0
    data = json.loads(out.getvalue())
    assert data == {"apps": ["Sampler", "Sequencer"], "running": "Sampler"}


def test_device_err_reply_exits_1(lock_dir):
    deps, out, _, _ = _deps([("APP_START Nope", "ERR unknown app")])
    rc = cli.main(["cdc", "APP_START", "Nope", "--json"], deps=deps)
    assert rc == 1
    assert json.loads(out.getvalue())["parsed"]["kind"] == "err"


def test_kit_load_wait(lock_dir):
    deps, out, _, fake = _deps([
        ("KIT_LOAD 3", "OK"),
        ("KIT_STATUS", "KITSTATUS: current=3 loading=0 pending=-1 name=DRUMS"),
    ])
    rc = cli.main(["kit", "load", "3", "--wait", "--json"], deps=deps)
    assert rc == 0
    data = json.loads(out.getvalue())
    assert data == {"current": 3, "loading": False, "pending": -1, "name": "DRUMS"}
    assert "KIT_LOAD 3" in fake.written and "KIT_STATUS" in fake.written


def test_ble_start_role_mapping(lock_dir):
    # Firmware contract (main/hil_control.cpp, BLE_START verb): 0 = Host (central),
    # 1 = Server (peripheral/advertising) -- the inverse of what "server"/"host" as
    # English words might suggest. `ble start server` must send BLE_START 1 and
    # `ble start host` must send BLE_START 0; getting this backwards puts the board
    # in the opposite radio role from the one asked for.
    deps, _, _, fake = _deps([("BLE_START 1", "OK"), ("BLE_START 0", "OK")])
    assert cli.main(["ble", "start", "server"], deps=deps) == 0
    assert cli.main(["ble", "start", "host"], deps=deps) == 0
    assert fake.written == ["BLE_START 1", "BLE_START 0"]


def test_ble_start_no_role_sends_bare_command(lock_dir):
    # No role given -> firmware falls back to the mode already in settings; the CLI
    # must not invent a mode of its own.
    deps, _, _, fake = _deps([("BLE_START", "OK")])
    assert cli.main(["ble", "start"], deps=deps) == 0
    assert fake.written == ["BLE_START"]


def test_led_state_grid(lock_dir):
    colors = ",".join(["000000"] * 15 + ["FF0000"])
    deps, out, _, _ = _deps([
        ("LED_STATE", f"LEDS: bri=80 anim=0 coalesce=1 cfgbri=80 pwr=0x00 pwrN=3 txfail=0 colors={colors}"),
    ])
    assert cli.main(["led", "state"], deps=deps) == 0
    text = out.getvalue()
    assert "15:FF0000" in text and " 0:000000" in text
    assert "brightness 80" in text


def test_run_smoke_json(lock_dir):
    calls: list[dict] = []

    def fake_run(name, params, *, device=None, workdir=None, progress=None,
                 cancelled=None, log=print) -> Report:
        calls.append({"name": name, "params": params, "device": device, "workdir": workdir})
        progress(1, 1, "boot complete")
        return Report(passed=True, summary="boot ok in 12.3 s", data={"missing": []},
                      artifacts=[], exit_code=0)

    deps, out, _, _ = _deps([], run_scenario_fn=fake_run)
    rc = cli.main(["run", "smoke", "--timeout", "30", "--json"], deps=deps)
    assert rc == 0
    data = json.loads(out.getvalue())
    assert data["passed"] is True and data["summary"] == "boot ok in 12.3 s"
    assert calls[0]["name"] == "smoke"
    assert calls[0]["params"].timeout == 30
    assert calls[0]["device"].id.startswith("dev_")
    assert Path(calls[0]["workdir"]).name.startswith("smoke_")


def test_run_failed_scenario_exits_1(lock_dir):
    def fake_run(name, params, **_) -> Report:
        return Report(passed=False, summary="missing marker", data={}, artifacts=[], exit_code=1)

    deps, out, _, _ = _deps([], run_scenario_fn=fake_run)
    assert cli.main(["run", "smoke"], deps=deps) == 1
    assert "FAIL" in out.getvalue()


def test_no_device_exit_2(lock_dir):
    deps, out, err, _ = _deps([])
    rc = cli.main(["-d", "dev_zzzz", "app", "list"], deps=deps)
    assert rc == 2
    assert err.getvalue().startswith("error: NO_DEVICE:")
    assert out.getvalue() == ""


def test_no_device_exit_2_json(lock_dir):
    deps, out, err, _ = _deps([])
    rc = cli.main(["--json", "-d", "dev_zzzz", "app", "list"], deps=deps)
    assert rc == 2
    data = json.loads(out.getvalue())
    assert data["ok"] is False and data["error"]["code"] == "NO_DEVICE"


def test_usage_error_exit_2(lock_dir):
    deps, _, err, _ = _deps([])
    assert cli.main(["kit", "load"], deps=deps) == 2
    assert "usage" in err.getvalue()


def test_help_exit_0(lock_dir):
    deps, out, _, _ = _deps([])
    assert cli.main(["--help"], deps=deps) == 0
    text = out.getvalue()
    for cmd in ("devices", "doctor", "console", "cdc", "app", "kit", "pad", "ui", "led",
                "mem", "ble", "midi", "usb-mode", "snapshot", "flash", "bootloader",
                "run", "record", "serve"):
        assert cmd in text


def test_run_lists_p0_scenarios(lock_dir):
    deps, out, _, _ = _deps([])
    assert cli.main(["run", "--help"], deps=deps) == 0
    text = out.getvalue()
    for name in ("smoke", "app_churn", "kit_churn", "led_state", "usb_mode_cycle"):
        assert name in text


def test_bad_arg_exit_2(lock_dir):
    deps, _, err, _ = _deps([])
    assert cli.main(["pad", "press", "16"], deps=deps) == 2
    assert "BAD_ARGS" in err.getvalue()


def test_doctor_checks_shape(lock_dir):
    checks = cli.doctor_checks(_backends())
    names = {c["name"] for c in checks}
    assert {"python", "import:serial", "import:rtmidi", "import:yaml", "devices", "locks"} <= names
    for c in checks:
        assert set(c) == {"name", "ok", "detail", "fix"}


def test_device_selection_by_port_path(lock_dir):
    deps, out, _, _ = _deps([("KIT_STATUS", "KITSTATUS: current=1 loading=0 pending=-1 name=A")])
    assert cli.main(["-d", CDC, "kit", "status", "--json"], deps=deps) == 0
    assert json.loads(out.getvalue())["current"] == 1


def test_snapshot_render_marks_focus():
    snap = {
        "snapshot_id": "snap_1", "device": "dev_ab12", "usb_mode": "default",
        "apps": {"apps": ["Sampler"], "running": None},
        "ui": {"focus": {"ref": "e1", "index": 1, "label": "Sampler"},
               "group": [{"ref": "e0", "index": 0, "ptr": "0x3c0", "label": "Sequencer"},
                         {"ref": "e1", "index": 1, "ptr": "0x3c1", "label": "Sampler"}],
               "drawer": True, "theme": 0, "app": None},
        "kit": {"current": 3, "loading": False, "pending": -1, "name": "DRUMS"},
        "leds": {"brightness": 80, "anim": 0, "coalesce": 1, "cfgbri": 80, "pwr": 0,
                 "pwr_count": 1, "txfail": 0, "colors": ["000000"] * 16},
        "pads": None, "mem": {"free": 60000, "largest": 30000, "min": 50000},
        "ble": None, "console": {"fatals": [], "reboots": 0, "cdc_drops": 0, "since_seq": 10},
        "ts": 1.0, "changed": [],
    }
    text = cli.render_snapshot(snap)
    assert "> e1  Sampler" in text and "  e0  Sequencer" in text
    assert "15:000000" in text and "DRUMS" in text


# --------------------------------------------------------------------------- kit push

def _write_test_wav(path: Path, seconds: float = 0.05, rate: int = 44100) -> None:
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(rate)
        w.writeframes(b"\x00\x00" * int(rate * seconds))


class KitPushDevice(FakeSerial):
    """CDC endpoint for `kit push`: KIT_LIST, FS_MKDIR, KIT_STATUS, KIT_LOAD, APP_LIST.

    File transfer itself (FS_PUT) is not modelled here — the CLI's `push_fn`
    is monkeypatched instead, exactly like `fs.push` is stood in for in
    `test_ensure_kits.py`'s `Pusher`, so no port is opened for the payload.
    """

    def __init__(self, kits: list[str]) -> None:
        super().__init__()
        self.kits = list(kits)
        self.current = 0
        self.mkdirs: list[str] = []

    def write(self, data: bytes) -> int:
        for raw in data.decode("utf-8", "replace").split("\n"):
            cmd = raw.strip()
            if not cmd:
                continue
            self.written.append(cmd)
            if cmd == "KIT_LIST":
                body = ",".join(f"{i}:{n}" for i, n in enumerate(self.kits))
                self.feed([f"KITS: {body}",
                           f"KITS: end count={len(self.kits)} current={self.current}"])
            elif cmd.startswith("FS_MKDIR "):
                self.mkdirs.append(cmd.split(" ", 1)[1])
                self.feed(["OK"])
            elif cmd.startswith("KIT_LOAD "):
                self.current = int(cmd.split(" ", 1)[1])
                self.feed(["OK"])
            elif cmd == "KIT_STATUS":
                name = self.kits[self.current] if 0 <= self.current < len(self.kits) else "-"
                self.feed([f"KITSTATUS: current={self.current} loading=0 pending=-1 name={name}"])
            elif cmd == "APP_LIST":
                self.feed(["APPS: Sampler running=Sampler"])
            else:
                self._rx.append((cmd + "\r\n").encode())
        return len(data)


class FakeConsoleForPush:
    """`Console.reset()` + `wait_boot()` + `forget_boot_history()`, and nothing else.

    `reset()` plays the part of the reboot that lets the kit scan see what
    was just pushed — it appends the kit name to the fake card's listing.
    """

    def __init__(self, device: KitPushDevice, kit_name: str) -> None:
        self._device = device
        self._kit_name = kit_name
        self.resets = 0
        self.forgot = 0

    def reset(self) -> None:
        self.resets += 1
        self._device.kits.append(self._kit_name)

    def wait_boot(self, timeout_s: float = 0.0, settle_s: float = 0.0):
        return type("Boot", (), {"missing": []})()

    def forget_boot_history(self) -> None:
        self.forgot += 1


def test_kit_push_builds_and_pushes(lock_dir, tmp_path, monkeypatch):
    src = tmp_path / "wavs"
    src.mkdir()
    _write_test_wav(src / "kick.wav")
    _write_test_wav(src / "snare.wav")

    device = KitPushDevice(["Basic"])
    console = FakeConsoleForPush(device, "My Kit")

    @contextmanager
    def fake_console(deps, dev, *, reset=False, log_path=None):
        yield console

    monkeypatch.setattr(cli, "_console", fake_console)

    pushed: list[tuple[str, str]] = []

    def fake_push(dev, local, remote, **kw):
        pushed.append((Path(local).name, remote))
        return {"bytes": Path(local).stat().st_size, "remote": remote}

    def factory(path: str, **_: object) -> KitPushDevice:
        return device

    out, err = io.StringIO(), io.StringIO()
    deps = cli.CliDeps(backends=_backends(), serial_factory=factory, stdout=out, stderr=err,
                       sleep=lambda s: None, push_fn=fake_push)

    rc = cli.main(["kit", "push", str(src), "--name", "My Kit",
                  "--logdir", str(tmp_path / "logs"), "--json"], deps=deps)
    assert rc == 0
    data = json.loads(out.getvalue())
    assert data == {"name": "My Kit", "dir": "MY_KIT", "bytes": data["bytes"],
                    "kit_id": 1, "loaded": False}
    assert data["bytes"] > 0

    assert console.resets == 1 and console.forgot == 1
    assert device.mkdirs == ["/sdcard/crosspad/kits/MY_KIT",
                             "/sdcard/crosspad/kits/MY_KIT/SAMPLES"]
    # 2 samples + kit.json, kit.json last
    assert len(pushed) == 3
    assert pushed[-1][0] == "kit.json"
    assert all(remote.startswith("/sdcard/crosspad/kits/MY_KIT") for _name, remote in pushed)


def test_kit_push_default_name_is_the_directory_basename(lock_dir, tmp_path, monkeypatch):
    src = tmp_path / "my-drum-kit"
    src.mkdir()
    _write_test_wav(src / "kick.wav")

    device = KitPushDevice([])
    console = FakeConsoleForPush(device, "my-drum-kit")

    @contextmanager
    def fake_console(deps, dev, *, reset=False, log_path=None):
        yield console

    monkeypatch.setattr(cli, "_console", fake_console)

    def fake_push(dev, local, remote, **kw):
        return {"bytes": Path(local).stat().st_size, "remote": remote}

    def factory(path: str, **_: object) -> KitPushDevice:
        return device

    deps = cli.CliDeps(backends=_backends(), serial_factory=factory,
                       stdout=io.StringIO(), stderr=io.StringIO(),
                       sleep=lambda s: None, push_fn=fake_push)
    rc = cli.main(["kit", "push", str(src), "--json", "--load"],
                  deps=deps)
    assert rc == 0
    assert json.loads(deps.stdout.getvalue()) == {
        "name": "my-drum-kit", "dir": "MY_DRUM_KIT",
        "bytes": json.loads(deps.stdout.getvalue())["bytes"], "kit_id": 0, "loaded": True,
    }
    assert "KIT_LOAD 0" in device.written


def test_sigterm_unwinds_like_ctrl_c() -> None:
    """kill <pid> must run the scenario's finally blocks (IN2, the USB profile)."""
    import signal as _signal

    with pytest.raises(KeyboardInterrupt):
        cli._sigterm_as_interrupt(_signal.SIGTERM, None)
    saved = _signal.getsignal(_signal.SIGTERM)
    try:
        cli._install_sigterm_handler()
        assert _signal.getsignal(_signal.SIGTERM) is cli._sigterm_as_interrupt
    finally:
        _signal.signal(_signal.SIGTERM, saved)


def test_cdc_prints_the_lines_a_reply_continues_on(lock_dir):
    """`crosspad-hil cdc` dropped every reply line that starts with a space."""
    reply = ["COREDUMP: seq=3 task=main pc=0x40381234 cause=28",
             " bt: 0x40381234 0x40382000", " exc: tcb=0x3fc9a000 a0=0x80381234"]
    deps, out, _, _ = _deps([("COREDUMP_INFO", reply)])
    assert cli.main(["cdc", "COREDUMP_INFO", "--json"], deps=deps) == 0
    data = json.loads(out.getvalue())
    assert data["more_lines"] == ["bt: 0x40381234 0x40382000", "exc: tcb=0x3fc9a000 a0=0x80381234"]
    deps, out, _, _ = _deps([("COREDUMP_INFO", reply)])
    assert cli.main(["cdc", "COREDUMP_INFO"], deps=deps) == 0
    assert out.getvalue().splitlines()[:3] == reply


# ---------------------------------------------------------------- the hub

def test_no_hub_flag_sets_env(monkeypatch):
    monkeypatch.delenv("CROSSPAD_HIL_NO_HUB", raising=False)
    cli._apply_global_flags(cli.build_parser().parse_args(["--no-hub", "devices"]))
    import os

    assert os.environ["CROSSPAD_HIL_NO_HUB"] == "1"


def test_no_hub_flag_after_the_command(monkeypatch):
    monkeypatch.delenv("CROSSPAD_HIL_NO_HUB", raising=False)
    cli._apply_global_flags(cli.build_parser().parse_args(["devices", "--no-hub"]))
    import os

    assert os.environ["CROSSPAD_HIL_NO_HUB"] == "1"


def test_serve_client_flag_sets_env(monkeypatch):
    monkeypatch.delenv("CROSSPAD_HIL_CLIENT", raising=False)
    cli._apply_global_flags(cli.build_parser().parse_args(["serve", "--client", "cp-tools"]))
    import os

    assert os.environ["CROSSPAD_HIL_CLIENT"] == "cp-tools"


def test_flash_runs_under_an_exclusive_op(lock_dir, exclusive_log, tmp_path):
    fw = tmp_path / "fw.bin"
    fw.write_bytes(b"\0" * 16)
    seen: list = []

    def flash(dev, path, **kw):
        seen.append(list(exclusive_log))
        return {"flash": {"bytes": 16}, "boot": None}

    deps, _, _, _ = _deps([], flash_fn=flash)
    assert cli.main(["flash", str(fw)], deps=deps) == 0
    assert seen == [[("take", "ota flash")]]
    assert exclusive_log[-1] == ("release", "ota flash")


@pytest.mark.uses_hub
def test_bench_busy_comes_before_the_hub(lock_dir, monkeypatch):
    from crosspad_hil import hub_link

    def no_hub(*a, **kw):
        raise AssertionError("the hub was asked before the bench lease was checked")

    monkeypatch.setattr(hub_link, "connect_or_start", no_hub)
    deps, _, err, _ = _deps([])
    assert cli.main(["bench", "claim", "smoke", "--holder", "pidf"], deps=deps) == 0
    assert cli.main(["cdc", "LED_STATE", "--holder", "twin"], deps=deps) == 2
    assert "BENCH_BUSY" in err.getvalue() and "claimed by pidf" in err.getvalue()


def test_midi_hub_versions_prints_the_last_reading(lock_dir, monkeypatch):
    from crosspad_hil import midi_hub
    path = lock_dir / "hub_versions.json"
    monkeypatch.setattr(midi_hub, "VERSIONS_PATH", path)
    deps, out, _, _ = _deps([])
    assert cli.main(["midi", "hub", "versions"], deps=deps) == 2
    assert "no reading yet" in out.getvalue()
    path.write_text(json.dumps({
        "board": "Crosspad", "verdict": "update_offered", "running": "v1.1.0-rc2",
        "board_latest": "1.1.0-rc3", "catalog_latest": "1.1.0-rc3", "kit": "full",
        "channel": "beta"}))
    deps, out, _, _ = _deps([])
    assert cli.main(["midi", "hub", "versions"], deps=deps) == 0
    assert "update_offered running=v1.1.0-rc2" in out.getvalue()
    path.write_text("{not json")
    deps, out, _, _ = _deps([])
    assert cli.main(["midi", "hub", "versions"], deps=deps) == 2
