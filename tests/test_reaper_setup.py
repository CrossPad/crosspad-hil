"""reaper_setup: the control script copied into REAPER and started with it."""
from __future__ import annotations

from pathlib import Path

import pytest

from crosspad_hil import reaper_setup as rs
from crosspad_hil.errors import HilError


def _script(tmp_path: Path, body: str = "-- v1\n") -> Path:
    p = tmp_path / "CrossPad.lua"
    p.write_text(body)
    return p


def test_install_copies_and_adds_the_startup_line_once(tmp_path: Path) -> None:
    res = tmp_path / "REAPER"
    (res / "Scripts").mkdir(parents=True)
    (res / "Scripts/__startup.lua").write_text("-- mine\nreaper.ShowConsoleMsg('hi')")
    r = rs.install(res, _script(tmp_path), midi=False)
    rs.install(res, _script(tmp_path, "-- v2\n"), midi=False)
    startup = (res / "Scripts/__startup.lua").read_text()
    assert startup.startswith("-- mine\nreaper.ShowConsoleMsg('hi')\n")
    assert startup.count(rs.STARTUP_LINE) == 1
    assert (res / "Scripts/CrossPad/CrossPad.lua").read_text() == "-- v2\n"
    assert r["script_installed"] and r["starts_with_reaper"]


def test_install_creates_the_startup_file(tmp_path: Path) -> None:
    res = tmp_path / "REAPER"
    res.mkdir()
    rs.install(res, _script(tmp_path), midi=False)
    assert rs.STARTUP_LINE in (res / "Scripts/__startup.lua").read_text()


def test_install_refuses_without_reaper_or_script(tmp_path: Path) -> None:
    with pytest.raises(HilError):
        rs.install(tmp_path / "none", _script(tmp_path))
    with pytest.raises(HilError):
        rs.install(tmp_path, tmp_path / "missing.lua")


def test_status_of_a_fresh_reaper(tmp_path: Path) -> None:
    assert rs.status(tmp_path) == {"resource": str(tmp_path),
                                   "script": str(tmp_path / "Scripts/CrossPad/CrossPad.lua"),
                                   "script_installed": False, "starts_with_reaper": False}


def test_find_script_in_a_platform_checkout(tmp_path: Path) -> None:
    assert rs.find_script(tmp_path) is None
    p = tmp_path / rs.REPO_SCRIPT
    p.parent.mkdir(parents=True)
    p.write_text("")
    assert rs.find_script(tmp_path) == p


CACHE = """[mididevcache]
ot0=1
on0=virtual
ot4=1
on4=hw:VirMIDI
ot5=1
on5=hw:VirMIDI,0,1
it0=1
in0=virtual
it4=1
in4=hw:VirMIDI
it40=1
in40=hw:Other
"""
INI = """[.swell]
ui_scale=1.0 // scales the sizes in libSwell.colortheme

[reaper]
alsa_indev=hw:Pro
wnd_y=80

[verchk]
lastt=1790459263
"""


def _reaper(tmp_path: Path, ini: str = INI) -> Path:
    res = tmp_path / "REAPER"
    res.mkdir()
    (res / rs.DEVICE_CACHE).write_text(CACHE)
    (res / "reaper.ini").write_text(ini)
    return res


def test_enable_virmidi_sets_the_bit_reaper_itself_writes(tmp_path: Path) -> None:
    res = _reaper(tmp_path)
    r = rs.enable_virmidi(res, running=lambda: False)
    text = (res / "reaper.ini").read_text()
    assert r["enabled"]
    assert "midiins=16\nmidiouts=16\n\n[verchk]" in text           # REAPER's own values for index 4
    assert text.startswith("[.swell]\nui_scale=1.0 // scales")      # nothing else rewritten


def test_enable_virmidi_keeps_other_enabled_devices(tmp_path: Path) -> None:
    res = _reaper(tmp_path, INI.replace("wnd_y=80", "wnd_y=80\nmidiins=3\nmidiouts_h=0"))
    rs.enable_virmidi(res, running=lambda: False)
    text = (res / "reaper.ini").read_text()
    assert "midiins=19" in text and "midiouts=16" in text and "midiouts_h=0" in text


def test_a_device_above_31_goes_to_the_next_word() -> None:
    assert "midiins_h=256" in rs.set_enabled_bit("[reaper]\n", "midiins", 40)


def test_enable_virmidi_refuses_while_reaper_runs_or_before_it_saw_the_card(tmp_path: Path) -> None:
    res = _reaper(tmp_path)
    assert not rs.enable_virmidi(res, running=lambda: True)["enabled"]
    assert (res / "reaper.ini").read_text() == INI
    (res / rs.DEVICE_CACHE).write_text("[mididevcache]\nin0=virtual\non0=virtual\n")
    assert "not in REAPER's device list" in rs.enable_virmidi(res, running=lambda: False)["why"]
