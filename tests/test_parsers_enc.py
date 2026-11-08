"""parse_enc_group: the multi-line ENC_GROUP dump → [{ref, index, ptr, label}]."""
from __future__ import annotations

import pytest

from crosspad_hil.parsers import parse_enc_group

# hil_control.cpp handle_enc_group:
#   "ENCGROUP: count=%u"            then per object
#   "  [%u] %p %s"                  (index, pointer, widget_label — may be "" or contain spaces)
SETTINGS_DUMP = [
    "ENCGROUP: count=5",
    "  [0] 0x3fcb0010 Display",
    "  [1] 0x3fcb0120 Audio",
    "  [2] 0x3fcb0230 BLE MIDI",
    "  [3] 0x3fcb0340 Send transpose",
    "  [4] 0x3fcb0450 ",
]


def test_settings_dump() -> None:
    assert parse_enc_group(SETTINGS_DUMP) == [
        {"ref": "e0", "index": 0, "ptr": "0x3fcb0010", "label": "Display"},
        {"ref": "e1", "index": 1, "ptr": "0x3fcb0120", "label": "Audio"},
        {"ref": "e2", "index": 2, "ptr": "0x3fcb0230", "label": "BLE MIDI"},
        {"ref": "e3", "index": 3, "ptr": "0x3fcb0340", "label": "Send transpose"},
        {"ref": "e4", "index": 4, "ptr": "0x3fcb0450", "label": ""},
    ]


def test_crlf_and_no_trailing_space_tolerated() -> None:
    lines = ["ENCGROUP: count=2\r\n", "  [0] 0x3fcb0010 Display\r\n", "  [1] 0x3fcb0120\r\n"]
    assert parse_enc_group(lines) == [
        {"ref": "e0", "index": 0, "ptr": "0x3fcb0010", "label": "Display"},
        {"ref": "e1", "index": 1, "ptr": "0x3fcb0120", "label": ""},
    ]


def test_empty_group() -> None:
    assert parse_enc_group(["ENCGROUP: count=0"]) == []


@pytest.mark.parametrize("lines", [[], ["ENCGROUP: ERR lock"], ["OK"], ["garbage"]])
def test_no_entries(lines: list[str]) -> None:
    assert parse_enc_group(lines) == []


def test_head_line_is_optional_and_noise_is_skipped() -> None:
    lines = [
        "I (1) hil_control: unrelated",
        "  [0] 0x3fcb0010 Sampler",
        "OK",
        "  [1] 0x3fcb0120 Sequencer",
    ]
    assert parse_enc_group(lines) == [
        {"ref": "e0", "index": 0, "ptr": "0x3fcb0010", "label": "Sampler"},
        {"ref": "e1", "index": 1, "ptr": "0x3fcb0120", "label": "Sequencer"},
    ]


def test_index_comes_from_the_firmware_not_position() -> None:
    # If a line was lost in transit the remaining indices must still be the device's own.
    lines = ["ENCGROUP: count=3", "  [0] 0x1 A", "  [2] 0x3 C"]
    out = parse_enc_group(lines)
    assert [e["index"] for e in out] == [0, 2]
    assert [e["ref"] for e in out] == ["e0", "e2"]


def test_label_with_brackets_and_multiple_spaces() -> None:
    lines = ["  [7] 0x3fcb0999 Vol  [dB]  -6"]
    assert parse_enc_group(lines) == [
        {"ref": "e7", "index": 7, "ptr": "0x3fcb0999", "label": "Vol  [dB]  -6"}
    ]
