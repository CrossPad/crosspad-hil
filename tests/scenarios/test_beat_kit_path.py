"""beat's pad_edit reads kit.json where the device says the kit is."""

from crosspad_hil.scenarios.beat import kit_json_from_padinfo


def test_slicer_kit_from_padinfo() -> None:
    raw = (
        "PADINFO: idx=0 num=0 kit=/crosspad/slicer/BEAT_11090108/kit.json vol=100 pan=0 "
        "mode=0 dirty=0 count=1 | [0] 'loop.wav' 0..12800"
    )
    assert kit_json_from_padinfo(raw) == "/sdcard/crosspad/slicer/BEAT_11090108/kit.json"


def test_path_already_on_sdcard_is_kept() -> None:
    raw = "PADINFO: idx=0 kit=/sdcard/crosspad/kits/X/kit.json count=1"
    assert kit_json_from_padinfo(raw) == "/sdcard/crosspad/kits/X/kit.json"


def test_no_kit_on_the_line() -> None:
    assert kit_json_from_padinfo("PADINFO: idx=0 num=0 count=0") is None
