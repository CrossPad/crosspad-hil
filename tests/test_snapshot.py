"""take_snapshot over a scripted FakeSerial, ref arithmetic. No hardware."""
from __future__ import annotations

import itertools
import uuid
from collections.abc import Iterator

import pytest

from crosspad_hil import snapshot as snap
from crosspad_hil.cdc import CdcLink
from crosspad_hil.devices import Device, Ports, UsbMode
from crosspad_hil.errors import HilError
from tests.fakes import FakeSerial

LEDS = "LEDS: bri=40 anim=0 coalesce=1 cfgbri=60 pwr=0x00 pwrN=3 txfail=0 colors=" + \
    ",".join("000000" for _ in range(16))
PADNOTES = "PADNOTES:" + "".join(f" {i}:{36 + i}" for i in range(16))
BLE = ("BLE: supported=1 running=0 state=idle mode=server self=- peer=- itvl=0 "
       "txoff=0 rxoff=0 tx_msg=0 tx_pkt=0 tx_drop=0 tx_err=0 rx_msg=0 rx_pkt=0")
MEM = ("MEM: int_free=76000 int_largest=32000 int_min=18000 int_blocks=900 "
       "psram_free=6000000 psram_largest=4000000 psram_blocks=300")


def full_script(*, running: str = "-", focus_ptr: str = "0x3fcb5678", kit: int = 3,
                int_free: int = 76000) -> list[tuple[str, str | list[str]]]:
    return [
        ("APP_LIST", f"APPS: Sampler,Sequencer,Settings running={running}"),
        ("ENC_FOCUS", f"ENCFOCUS: obj={focus_ptr} idx=7 text=Sequencer"),
        ("ENC_GROUP", ["ENCGROUP: count=3",
                       "  [0] 0x3fcb1234 Sampler",
                       "  [1] 0x3fcb5678 Sequencer",
                       "  [2] 0x3fcb9abc Settings"]),
        ("UI_STATE", "UI: display=yes touch=yes drawer=1 lcd=80 rgb=60 theme=2 bt_icon=hidden app=-"),
        ("KIT_STATUS", f"KITSTATUS: current={kit} loading=0 pending=-1 name=DRUMS"),
        ("LED_STATE", LEDS),
        ("PAD_STATS", "PADSTATS: press=12 release=11 played=12 freeslots=6"),
        ("PAD_NOTES", PADNOTES),
        ("MEM", MEM.replace("int_free=76000", f"int_free={int_free}")),
        ("BLE_STATUS", BLE),
    ]


class FakeConsole:
    """Only what take_snapshot reads: Console.snapshot()."""

    def __init__(self, fatals=(), reboots: int = 0, cdc_drops: int = 0, seq: int = 100) -> None:
        self._d = {"fatals": list(fatals), "reboots": reboots, "cdc_drops": cdc_drops,
                   "seq": seq, "lines_lost": 0, "log_path": None, "port": "/dev/fake"}

    def snapshot(self) -> dict:
        return dict(self._d)


def make_device() -> Device:
    return Device(id="dev_3f2a", serial="ABC123", usb_mode=UsbMode.DEFAULT, ports=Ports())


@pytest.fixture
def make_link() -> Iterator:
    opened: list[CdcLink] = []

    def _make(script) -> CdcLink:
        fake = FakeSerial(list(script))
        link = CdcLink(f"fake-snap-{uuid.uuid4().hex[:8]}",
                       serial_factory=lambda path, **kw: fake)
        link.open()
        link.fake = fake  # type: ignore[attr-defined]
        opened.append(link)
        return link

    yield _make
    for link in opened:
        link.close()


def test_full_snapshot_shape(make_link) -> None:
    link = make_link(full_script())
    s = snap.take_snapshot(make_device(), link, console=FakeConsole(seq=4321),
                           counter=itertools.count(1))
    assert s.snapshot_id == "snap_1"
    assert s.device == "dev_3f2a" and s.usb_mode == "default"
    assert s.apps == {"apps": ["Sampler", "Sequencer", "Settings"], "running": None}
    assert s.ui["focus"] == {"ref": "e1", "index": 1, "label": "Sequencer"}
    assert [g["ref"] for g in s.ui["group"]] == ["e0", "e1", "e2"]
    assert s.ui["drawer"] is True and s.ui["theme"] == 2 and s.ui["app"] is None
    assert s.kit == {"current": 3, "loading": False, "pending": -1, "name": "DRUMS"}
    assert len(s.leds["colors"]) == 16
    assert s.pads["press"] == 12 and s.pads["notes"][0] == 36
    assert s.mem["int_free"] == 76000
    assert s.ble["supported"] == 1
    assert s.console == {"handle": None, "fatals": [], "reboots": 0, "cdc_drops": 0,
                         "since_seq": 4321}
    assert s.changed == []
    assert s.ts > 0
    d = s.to_dict()
    assert set(d) == {"snapshot_id", "device", "usb_mode", "apps", "ui", "kit", "leds",
                      "pads", "mem", "ble", "console", "ts", "changed"}


def test_snapshot_ids_advance_with_counter(make_link) -> None:
    link = make_link(full_script())
    c = itertools.count(7)
    assert snap.take_snapshot(make_device(), link, include=("kit",), counter=c).snapshot_id == "snap_7"
    assert snap.take_snapshot(make_device(), link, include=("kit",), counter=c).snapshot_id == "snap_8"


def test_include_subset_only_sends_those_verbs(make_link) -> None:
    link = make_link(full_script())
    s = snap.take_snapshot(make_device(), link, include=("kit", "leds"))
    assert s.kit is not None and s.leds is not None
    assert s.apps is None and s.ui is None and s.pads is None
    assert s.mem is None and s.ble is None and s.console is None
    assert set(link.fake.written) == {"KIT_STATUS", "LED_STATE"}


def test_include_unknown_section_is_bad_args(make_link) -> None:
    link = make_link(full_script())
    with pytest.raises(HilError) as ei:
        snap.take_snapshot(make_device(), link, include=("kit", "bogus"))
    assert ei.value.code == "BAD_ARGS"
    assert link.fake.written == []


def test_console_section_without_console_is_none(make_link) -> None:
    link = make_link(full_script())
    s = snap.take_snapshot(make_device(), link, include=("console", "kit"))
    assert s.console is None and s.kit is not None


def test_focus_pointer_not_in_group(make_link) -> None:
    link = make_link(full_script(focus_ptr="0x0"))
    s = snap.take_snapshot(make_device(), link, include=("ui",))
    assert s.ui["focus"] == {"ref": None, "index": -1, "label": "Sequencer"}


def test_changed_detection(make_link) -> None:
    first = snap.take_snapshot(make_device(), make_link(full_script()), include=("apps", "kit", "mem"))
    same = snap.take_snapshot(make_device(), make_link(full_script()),
                              include=("apps", "kit", "mem"), previous=first)
    assert same.changed == []
    later = snap.take_snapshot(make_device(), make_link(full_script(running="Sampler", kit=5)),
                               include=("apps", "kit", "mem"), previous=first)
    assert sorted(later.changed) == ["apps", "kit"]


def test_changed_ignores_small_mem_wobble_but_sees_free_change(make_link) -> None:
    first = snap.take_snapshot(make_device(), make_link(full_script()), include=("mem",))
    wobble = snap.take_snapshot(make_device(), make_link(full_script(int_free=76000)),
                                include=("mem",), previous=first)
    assert wobble.changed == []
    drop = snap.take_snapshot(make_device(), make_link(full_script(int_free=60000)),
                              include=("mem",), previous=first)
    assert drop.changed == ["mem"]


def test_changed_includes_usb_mode(make_link) -> None:
    first = snap.take_snapshot(make_device(), make_link(full_script()), include=("kit",))
    dev = make_device()
    dev.usb_mode = UsbMode.AUDIO
    second = snap.take_snapshot(dev, make_link(full_script()), include=("kit",), previous=first)
    assert second.changed == ["usb_mode"]


def test_failed_section_recorded_not_raised(make_link) -> None:
    script = [s for s in full_script() if s[0] != "BLE_STATUS"]
    script.append(("BLE_STATUS", "ERR"))
    link = make_link(script)
    s = snap.take_snapshot(make_device(), link, include=("kit", "ble"))
    assert s.kit["current"] == 3
    assert "error" in s.ble and s.ble["error"]["code"] == "NOT_SUPPORTED"


GROUP = [{"ref": "e0", "index": 0, "ptr": "0x1", "label": "A"},
         {"ref": "e1", "index": 1, "ptr": "0x2", "label": "B"},
         {"ref": "e2", "index": 2, "ptr": "0x3", "label": "C"}]


def test_ref_to_delta_positive() -> None:
    assert snap.ref_to_delta(GROUP, 0, "e2") == 2


def test_ref_to_delta_negative() -> None:
    assert snap.ref_to_delta(GROUP, 2, "e0") == -2


def test_ref_to_delta_zero() -> None:
    assert snap.ref_to_delta(GROUP, 1, "e1") == 0


def test_ref_to_delta_unknown() -> None:
    with pytest.raises(HilError) as ei:
        snap.ref_to_delta(GROUP, 0, "e9")
    assert ei.value.code == "BAD_ARGS"
    assert ei.value.details["known"] == ["e0", "e1", "e2"]
