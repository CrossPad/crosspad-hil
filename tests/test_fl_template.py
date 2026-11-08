"""The CrossPad project template: inserts named and pointed at the pad stereo pair."""
import struct
from pathlib import Path

import pytest

from crosspad_hil import fl_template as ft
from crosspad_hil.errors import HilError
from crosspad_hil.fpc import _encode_event, _events, _from_utf16


def _insert(name: str | None = None, color: int | None = None, input_=-1, output=-1) -> bytes:
    """One insert run the way FL writes it: [name] slots... input output [color]."""
    out = bytearray()
    if name is not None:
        out += _encode_event(ft.EV_NAME, name.encode("utf-16-le") + b"\0\0")
    out += _encode_event(ft.EV_SLOT_HEAD, bytes(12))
    for slot in range(10):
        out += _encode_event(98, struct.pack("<H", slot))
    out += _encode_event(165, struct.pack("<I", 3)) + _encode_event(49, b"\x00")
    out += _encode_event(ft.EV_INPUT, struct.pack("<i", input_))
    out += _encode_event(ft.EV_OUTPUT, struct.pack("<i", output))
    if color is not None:
        out += _encode_event(ft.EV_COLOR, struct.pack("<I", color))
    return bytes(out)


def _flp(inserts: int = 4) -> bytes:
    body = _encode_event(199, b"\x01" * 13) + _encode_event(156, struct.pack("<I", 140000))
    body += _insert(output=0)                                  # master
    for _ in range(inserts):
        body += _insert()
    body += _encode_event(38, b"\x01")                          # something after the mixer
    head = b"FLhd" + struct.pack("<I", 6) + bytes(6)
    return head + b"FLdt" + struct.pack("<I", len(body)) + body


def _runs(data: bytes):
    """(name, input, color) per insert, from the written stream."""
    _, body, _ = ft._split(data)
    runs, cur = [], {}
    for ev, payload in _events(body):
        if ev == ft.EV_NAME:
            cur["name"] = _from_utf16(payload)
        elif ev == ft.EV_INPUT:
            cur["input"] = struct.unpack("<i", payload)[0]
        elif ev == ft.EV_COLOR:                    # FL writes it after Output
            runs[-1]["color"] = struct.unpack("<I", payload)[0]
        elif ev == ft.EV_OUTPUT:
            runs.append(cur)
            cur = {}
    return runs


def test_insert_inputs_reads_master_first():
    assert ft.insert_inputs(_flp(3)) == [-1, -1, -1, -1]


def test_build_names_and_routes_only_the_requested_inserts():
    out = ft.build_template(_flp(4), ft.crosspad_inserts(2, 0))
    runs = _runs(out)
    assert [r.get("name") for r in runs] == [None, "CrossPad 1", "CrossPad 2", None, None]
    assert [r["input"] for r in runs] == [-1, 0, 0, -1, -1]
    assert runs[1]["color"] == ft.CROSSPAD_COLOR and "color" not in runs[3]


def test_single_track_keeps_the_plain_name():
    runs = _runs(ft.build_template(_flp(2), ft.crosspad_inserts(1, 3)))
    assert runs[1] == {"name": "CrossPad", "input": 3, "color": ft.CROSSPAD_COLOR}


def test_everything_else_survives_byte_for_byte():
    src = _flp(2)
    out = ft.build_template(src, [])
    assert out == src
    out = ft.build_template(src, ft.crosspad_inserts(1))
    head, body, tail = ft._split(out)
    assert head == src[:14] and tail == b""
    assert body.endswith(_encode_event(38, b"\x01"))


def test_refuses_master_and_missing_inserts():
    with pytest.raises(HilError):
        ft.build_template(_flp(2), [ft.InsertSpec(0, "x", 0)])
    with pytest.raises(HilError):
        ft.build_template(_flp(2), [ft.InsertSpec(5, "x", 0)])
    with pytest.raises(HilError):
        ft.build_template(b"RIFF" + bytes(20), [])


def test_install_writes_every_user_profile(tmp_path):
    drive_c = tmp_path / "drive_c"
    empty = drive_c / "Program Files/Image-Line/FL Studio 2026" / ft.EMPTY_REL
    empty.parent.mkdir(parents=True)
    empty.write_bytes(_flp(3))
    for user in ("matixan", "steamuser"):
        (drive_c / "users" / user / ft.TEMPLATE_DIR).mkdir(parents=True)
    r = ft.install(drive_c, tracks=1)
    assert len(r["written"]) == 2 and r["input"] == ft.CROSSPAD_INPUT
    for path in r["written"]:
        assert Path(path).as_posix().endswith("Templates/CrossPad/CrossPad.flp")
        assert ft.insert_inputs(open(path, "rb").read()) == [-1, 0, -1, -1]


def test_install_needs_fl_and_a_user_folder(tmp_path):
    with pytest.raises(HilError):
        ft.install(tmp_path)
    empty = tmp_path / "Program Files/Image-Line/FL Studio 21" / ft.EMPTY_REL
    empty.parent.mkdir(parents=True)
    empty.write_bytes(_flp(1))
    with pytest.raises(HilError):
        ft.install(tmp_path)
