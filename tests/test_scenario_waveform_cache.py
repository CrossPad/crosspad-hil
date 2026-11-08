"""Scenario `waveform_cache` against a scripted CDC link (no hardware)."""
from __future__ import annotations

import threading
from pathlib import Path
from typing import Any

import pytest

from crosspad_hil.devices import Device, Ports, SerialPortInfo, UsbMode
from crosspad_hil.parsers import parse_cdc_reply
from crosspad_hil.scenarios import base, get, waveform_cache
from tests.fakes import FakeSerial

CDC_PATH = "/dev/ttyACM0"
CONSOLE_PATH = "/dev/ttyACM1"

KITS = "KITS: 0:Basic Kit,1:DRUMS current=-1"


def wave(kit: int, total: int, cached: int, missing: str, **counters: int) -> str:
    c = {"queued": 0, "req": total, "ok": cached, "retry": 0, "fail_info": 0,
         "fail_read": 0, "alloc_fail": 0, "drop": 0}
    c.update(counters)
    return (f"WAVE: kit={kit} total={total} cached={cached} loader=1 queued={c['queued']} "
            f"req={c['req']} ok={c['ok']} retry={c['retry']} fail_info={c['fail_info']} "
            f"fail_read={c['fail_read']} alloc_fail={c['alloc_fail']} drop={c['drop']} "
            f"missing={missing}")


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


def make_ctx(tmp_path: Path, script: list[tuple[str, str]]) -> tuple[base.Context, FakeSerial]:
    fake = FakeSerial(script=script, timeout=0.01)

    def factory(path: str, **kw: Any) -> FakeSerial:
        assert path == CDC_PATH, path
        return fake

    ctx = base.Context(device=make_device(), workdir=tmp_path,
                       cancelled=threading.Event(), log=lambda s: None,
                       serial_factory=factory)
    return ctx, fake


@pytest.fixture(autouse=True)
def lock_dir(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv("XDG_RUNTIME_DIR", str(tmp_path / "run"))
    monkeypatch.setattr(waveform_cache, "POLL_S", 0.0)


def test_registered() -> None:
    assert get("waveform_cache") is waveform_cache.SCENARIO
    assert waveform_cache.Params().play is False


def test_parse_wave_reply() -> None:
    p = parse_cdc_reply(wave(3, 20, 18, "4:0,9:2", retry=2, fail_read=1))
    assert p is not None
    assert p["kind"] == "wave"
    assert p["kit"] == 3 and p["total"] == 20 and p["cached"] == 18
    assert p["missing"] == [(4, 0), (9, 2)]
    assert p["retry"] == 2 and p["fail_read"] == 1
    assert p["loader"] is True
    assert parse_cdc_reply(wave(0, 5, 5, "-"))["missing"] == []


def kit_load_script(kit: int) -> list[tuple[str, str]]:
    # KIT_LOAD is answered OK, then KIT_STATUS reports it current.
    return [(f"KIT_LOAD {kit}", "OK"),
            ("KIT_STATUS", f"KITSTATUS: current={kit} loading=0 pending=-1 name=K{kit}")]


def test_every_layer_cached_passes(tmp_path: Path) -> None:
    script = [("KIT_LIST", KITS), ("WAVE_STATUS", wave(-1, 0, 0, "-"))]
    script += kit_load_script(0) + [("WAVE_STATUS", wave(0, 16, 16, "-"))]
    script += kit_load_script(1) + [("WAVE_STATUS", wave(1, 20, 17, "3:0,7:1,7:2")),
                                    ("WAVE_STATUS", wave(1, 20, 20, "-"))]
    ctx, _ = make_ctx(tmp_path, script)
    report = waveform_cache.SCENARIO.run(
        ctx, waveform_cache.Params(timeout=5.0, settle=0.0), base.Progress())
    assert report.passed is True, report.summary
    assert report.exit_code == 0
    assert [k["cached"] for k in report.data["kits"]] == [16, 20]
    assert report.data["kits"][1]["missing"] == []


def test_missing_layer_fails_and_names_it(tmp_path: Path) -> None:
    script = [("KIT_LIST", KITS), ("WAVE_STATUS", wave(-1, 0, 0, "-"))]
    script += kit_load_script(1) + [("WAVE_STATUS", wave(1, 20, 19, "11:1", fail_read=3))] * 3
    ctx, _ = make_ctx(tmp_path, script)
    report = waveform_cache.SCENARIO.run(
        ctx, waveform_cache.Params(kits=[1], timeout=0.05, settle=0.0), base.Progress())
    assert report.passed is False
    assert report.exit_code == 1
    assert "pad:layer 11:1" in report.summary
    assert "fail_read=3" in report.summary
    assert report.data["kits"][0]["missing"] == [(11, 1)]


def test_firmware_without_verb_is_environment(tmp_path: Path) -> None:
    script = [("KIT_LIST", KITS), ("WAVE_STATUS", "WAVE_STATUS")]  # echoed = unknown verb
    ctx, _ = make_ctx(tmp_path, script)
    report = waveform_cache.SCENARIO.run(
        ctx, waveform_cache.Params(timeout=0.05, settle=0.0), base.Progress())
    assert report.passed is False
    assert report.exit_code == 2
