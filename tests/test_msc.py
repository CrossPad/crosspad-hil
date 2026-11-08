"""crosspad_hil.msc without a board: finding the card's disk, mounting it, ejecting it."""
from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import pytest

from crosspad_hil import msc
from crosspad_hil.errors import ENV, HilError

LSBLK = [
    {"name": "nvme0n1", "type": "disk", "vendor": None, "model": "Samsung SSD", "tran": "nvme"},
    {"name": "sdd", "type": "disk", "vendor": "CrossPad", "model": "SD Card         ",
     "tran": "usb", "children": [{"name": "sdd1", "type": "part"}]},
]


def test_the_board_is_found_by_its_inquiry_strings():
    assert msc.find_disk(LSBLK) == ("/dev/sdd", "/dev/sdd1")


def test_a_card_without_partitions_is_mounted_whole():
    card = [dict(LSBLK[1], children=[])]
    assert msc.find_disk(card) == ("/dev/sdd", "/dev/sdd")


def test_other_disks_are_not_the_board():
    assert msc.find_disk(LSBLK[:1]) is None


class Runner:
    def __init__(self, answers: dict[str, list]) -> None:
        self.answers = answers
        self.calls: list[list[str]] = []

    def __call__(self, cmd: Sequence[str]) -> str:
        self.calls.append(list(cmd))
        got = self.answers[f"{cmd[0]} {cmd[1]}"].pop(0)
        if isinstance(got, Exception):
            raise got
        return got


def test_mount_is_asked_again_until_udisks_knows_the_disk():
    run = Runner({"udisksctl mount": [HilError(ENV, "Error looking up object for device"),
                                      "Mounted /dev/sdd1 at /media/u/0403-0201"]})
    assert msc.mount("/dev/sdd1", run=run, wait=lambda s: None) == Path("/media/u/0403-0201")
    assert len(run.calls) == 2


def test_a_drive_the_desktop_mounted_is_used_where_it_is():
    run = Runner({"udisksctl mount": [HilError(ENV, "GDBus.Error:...AlreadyMounted: ...")],
                  "lsblk -no": ["/media/u/0403-0201\n"]})
    assert msc.mount("/dev/sdd1", run=run, wait=lambda s: None) == Path("/media/u/0403-0201")


def test_another_mount_error_is_not_retried():
    run = Runner({"udisksctl mount": [HilError(ENV, "Not authorized")]})
    with pytest.raises(HilError):
        msc.mount("/dev/sdd1", run=run, wait=lambda s: None)
    assert len(run.calls) == 1


def test_eject_goes_through_the_drive_not_power_off():
    run = Runner({"udisksctl info": ["  Drive: '/org/freedesktop/UDisks2/drives/CrossPad_SD'\n"],
                  "gdbus call": [""]})
    msc.eject("/dev/sdd", run=run)
    assert run.calls[-1][-2:] == ["org.freedesktop.UDisks2.Drive.Eject", "{}"]
    assert all("power-off" not in c for c in run.calls)
