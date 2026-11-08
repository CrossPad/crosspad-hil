"""board_state: one reading of what the board is set to, and the diff between two."""
from __future__ import annotations

import uuid

from crosspad_hil import board_state
from crosspad_hil.cdc import CdcLink
from tests.fakes import FakeSerial

SCRIPT = [
    ("BLE_STATUS", "BLE: supported=1 running=0 state=idle mode=server self=- peer=- itvl=0"),
    ("USB_GUARD", "USB_GUARD: 1"),
    ("WIFI_STATUS", 'WIFI: state=off ssid="" rssi=0 ip=- leases=0 fail=0 reason=0 enabled=0'),
    ("AUDIO_STATUS", "AUDIO: mic_src=1 adc=[1,1] out=0 vol=80 mute=0"),
    ("INGAIN", "INGAIN: IN1=24 IN2=0"),
    ("APP_LIST", "APPS: Sampler,Mixer running=-"),
    ("KIT_STATUS", "KITSTATUS: current=2 loading=0 pending=-1 name=HIL SINE A"),
    ("MEM", "MEM: int_free=120000 int_largest=60000 psram_free=4000000 psram_largest=3900000"),
]


def link_for(script: list[tuple[str, str]]) -> CdcLink:
    fake = FakeSerial(script)
    link = CdcLink(f"fake-state-{uuid.uuid4().hex[:8]}", serial_factory=lambda path, **kw: fake)
    link.open()
    return link


def test_read_gives_every_field() -> None:
    state = board_state.read(link_for(SCRIPT))
    assert state == {
        "ble_running": False, "ble_mode": "server", "usb_guard": True, "wifi_enabled": False,
        "audio_route": [1, 1, 1], "ingain": {"IN1": 24, "IN2": 0},
        "app": "launcher", "kit": 2, "int_free": 120000, "psram_free": 4000000,
    }


def test_a_verb_the_firmware_lacks_leaves_its_field_none() -> None:
    state = board_state.read(link_for([s for s in SCRIPT if s[0] not in ("INGAIN", "USB_GUARD")]))
    assert state["ingain"] is None and state["usb_guard"] is None
    assert state["kit"] == 2


def test_diff_splits_persistent_from_informational_and_ignores_memory() -> None:
    before = {"ble_running": False, "usb_guard": True, "app": "launcher", "kit": 2,
              "int_free": 120000, "ingain": None}
    after = {"ble_running": True, "usb_guard": True, "app": "Sampler", "kit": 2,
             "int_free": 63000, "ingain": {"IN1": 24, "IN2": 24}}
    changes = board_state.diff(before, after)
    assert changes == [
        {"key": "ble_running", "before": False, "after": True, "persistent": True},
        {"key": "app", "before": "launcher", "after": "Sampler", "persistent": False},
    ]
    assert board_state.describe([changes[0]]) == "ble_running False -> True"
    assert board_state.diff(None, after) == [] and board_state.diff(before, None) == []
