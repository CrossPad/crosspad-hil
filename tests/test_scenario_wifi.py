"""Scenario `wifi` without a board: the credential gate (exit 2) and its registration."""
from __future__ import annotations

import threading
from types import SimpleNamespace

import pytest

from crosspad_hil.devices import Device, Ports, SerialPortInfo, UsbMode
from crosspad_hil.errors import TIMEOUT, HilError
from crosspad_hil.scenarios import base, get, wifi


def make_device() -> Device:
    return Device(
        id="dev_ab12", serial="CP-1", usb_mode=UsbMode.DEFAULT,
        ports=Ports(cdc=SerialPortInfo(path="/dev/ttyACM0", vid=0x303A, pid=0x3456,
                                       serial="CP-1", product="CrossPad", location="1-2.1")),
    )


def test_registered():
    assert get("wifi").name == "wifi"


def test_without_credentials_exits_2_before_touching_the_board(tmp_path, monkeypatch):
    monkeypatch.delenv("CROSSPAD_WIFI_SSID", raising=False)
    monkeypatch.delenv("CROSSPAD_WIFI_PASS", raising=False)
    report = base.run_scenario("wifi", wifi.Params(), device=make_device(), workdir=tmp_path,
                               log=lambda s: None)
    assert report.exit_code == 2
    assert "CROSSPAD_WIFI_SSID" in report.summary


class _Ctx:
    def __init__(self, link) -> None:
        self.link = link
        self.cancelled = threading.Event()
        self.partial: dict = {}
        self.logs: list[str] = []

    def open_cdc(self):
        return self.link

    def check_cancelled(self) -> None:
        pass

    def log(self, line: str) -> None:
        self.logs.append(line)

    def close_all(self) -> None:
        pass


def test_failure_after_the_wrong_password_restores_the_board(monkeypatch) -> None:
    """The wrong-password check overwrites the saved password; a failure right after it
    must still re-save the right one and put the recorded enable states back."""
    monkeypatch.setenv("CROSSPAD_WIFI_SSID", "net")
    monkeypatch.setenv("CROSSPAD_WIFI_PASS", "rightpass")
    calls: list[tuple] = []
    link = SimpleNamespace(close=lambda: None)

    def wifi_set(_l, ssid, password=""):
        calls.append(("wifi_set", ssid, password))

    def wifi_status(_l):
        if calls and calls[-1] == ("wifi_set", "net", "rightpassx"):
            raise HilError(TIMEOUT, "injected")
        return {"state": "up", "enabled": 0}

    fakes = {
        "wifi_status": wifi_status, "wifi_set": wifi_set,
        "wifi_enable": lambda _l, on: calls.append(("wifi_enable", on)),
        # running until the scenario stops it; the fake's enable does not start it again
        "ble_status": lambda _l: {"running": 0 if ("ble_stop",) in calls else 1},
        "ble_stop": lambda _l: calls.append(("ble_stop",)),
        "ble_enable": lambda _l, on: calls.append(("ble_enable", on)),
        "ble_start": lambda _l, mode=None: calls.append(("ble_start",)),
        "wifi_ping": lambda _l: {"ok": True, "ms": 1, "status": 404},
        "reset_count": lambda _l: 0,
        "net_mode_status": lambda _l: {"active": False, "tab": None},
        "app_list": lambda _l: {"apps": ["Sampler"], "running": None},
    }
    for name, fn in fakes.items():
        monkeypatch.setattr(wifi.verbs, name, fn)
    ctx = _Ctx(link)
    with pytest.raises(HilError):
        wifi.SCENARIO.run(ctx, wifi.Params(), lambda *a: None)
    tail = calls[calls.index(("wifi_set", "net", "rightpassx")) + 1:]
    assert tail == [("wifi_set", "net", "rightpass"), ("wifi_enable", False),
                    ("ble_enable", True), ("ble_start",)]


def test_a_failed_restore_step_does_not_skip_the_next(monkeypatch) -> None:
    calls: list[str] = []
    ctx = _Ctx(SimpleNamespace(close=lambda: None))

    def boom(*_a):
        raise HilError(TIMEOUT, "boom")

    monkeypatch.setattr(wifi.verbs, "net_mode_status", lambda _l: {"active": False})
    monkeypatch.setattr(wifi.verbs, "wifi_set", boom)
    monkeypatch.setattr(wifi.verbs, "wifi_enable", lambda _l, on: calls.append("wifi_enable"))
    monkeypatch.setattr(wifi.verbs, "ble_enable", lambda _l, on: calls.append("ble_enable"))
    wifi._restore(ctx, ctx.link, wifi._Original(True, False), "net", "pw", False)
    assert calls == ["wifi_enable", "ble_enable"]
    assert any("re-saving" in line for line in ctx.logs)


def test_restore_tolerates_the_board_still_being_in_normal_mode(monkeypatch) -> None:
    """in_net is set before NET_MODE 1 is sent; if that never took, NET_MODE 0 answers
    "OK unchanged" and the restore still re-saves and restores."""
    calls: list[tuple] = []
    ctx = _Ctx(SimpleNamespace(close=lambda: None))
    monkeypatch.setattr(wifi, "REBOOT_SETTLE_S", 0.0)
    monkeypatch.setattr(wifi.verbs, "net_mode_status", lambda _l: {"active": False})
    monkeypatch.setattr(wifi.verbs, "net_mode_set", lambda _l, on, tab=None: calls.append(("net", on)) or
                        {"ok": True, "reboot": False})
    monkeypatch.setattr(wifi.verbs, "wifi_set", lambda _l, s, p="": calls.append(("set", p)))
    monkeypatch.setattr(wifi.verbs, "wifi_enable", lambda _l, on: calls.append(("en", on)))
    monkeypatch.setattr(wifi.verbs, "ble_enable", lambda _l, on: calls.append(("ble", on)))
    wifi._restore(ctx, ctx.link, wifi._Original(True, False), "net", "pw", True)
    assert calls == [("net", False), ("set", "pw"), ("en", True), ("ble", False)]


def test_to_launcher_stops_the_running_app_and_waits_for_none(monkeypatch) -> None:
    """kit_churn leaves the Sampler with a kit open, where the PSRAM gate refuses the radio by
    design: the scenario stops it and waits until APP_LIST shows no app before any WiFi check."""
    calls: list[str] = []
    listings = iter(["Sampler", "Sampler", None])
    monkeypatch.setattr(wifi, "POLL_S", 0.0)
    monkeypatch.setattr(wifi.verbs, "app_list",
                        lambda _l: calls.append("app_list") or {"apps": [], "running": next(listings)})
    monkeypatch.setattr(wifi.verbs, "app_stop", lambda _l: calls.append("app_stop") or {"ok": True})
    ctx = _Ctx(SimpleNamespace(close=lambda: None))
    assert wifi._to_launcher(ctx, ctx.link) == "Sampler"
    assert calls == ["app_list", "app_stop", "app_list", "app_list"]


def test_to_launcher_leaves_the_launcher_alone(monkeypatch) -> None:
    calls: list[str] = []
    monkeypatch.setattr(wifi.verbs, "app_list", lambda _l: {"apps": [], "running": None})
    monkeypatch.setattr(wifi.verbs, "app_stop", lambda _l: calls.append("app_stop"))
    ctx = _Ctx(SimpleNamespace(close=lambda: None))
    assert wifi._to_launcher(ctx, ctx.link) is None
    assert calls == []


def test_to_launcher_times_out_when_the_app_never_stops(monkeypatch) -> None:
    monkeypatch.setattr(wifi, "POLL_S", 0.0)
    monkeypatch.setattr(wifi, "LAUNCHER_TIMEOUT_S", 0.0)
    monkeypatch.setattr(wifi.verbs, "app_list", lambda _l: {"apps": [], "running": "Sampler"})
    monkeypatch.setattr(wifi.verbs, "app_stop", lambda _l: {"ok": True})
    ctx = _Ctx(SimpleNamespace(close=lambda: None))
    with pytest.raises(HilError) as exc:
        wifi._to_launcher(ctx, ctx.link)
    assert exc.value.code == TIMEOUT


def test_run_records_the_starting_app(monkeypatch) -> None:
    monkeypatch.setenv("CROSSPAD_WIFI_SSID", "net")
    monkeypatch.setenv("CROSSPAD_WIFI_PASS", "rightpass")
    monkeypatch.setattr(wifi, "_to_launcher", lambda _c, _l: "Sampler")

    def stop_here(_l):
        raise HilError(TIMEOUT, "stop after the launcher step")

    monkeypatch.setattr(wifi.verbs, "wifi_status", stop_here)
    ctx = _Ctx(SimpleNamespace(close=lambda: None))
    with pytest.raises(HilError):
        wifi.SCENARIO.run(ctx, wifi.Params(), lambda *a: None)
    assert ctx.partial["start_app"] == "Sampler"
