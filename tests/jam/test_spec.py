"""The jam's spectrum plan and the fixture kits built from it."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

from crosspad_hil.jam import spec as S


def test_every_band_is_its_own_at_44k1() -> None:
    assert S.check_plan(44_100) == []


def test_at_48k_only_the_click_guard_is_closer() -> None:
    """Files play 1:1, so at 48 kHz every pad moves up 8.8 %; the pads still keep
    clear of each other, the pilot, the timecode and the quiet bands."""
    problems = S.check_plan(48_000)
    assert problems and all("click" in p for p in problems)


def test_the_worst_case_stays_under_full_scale() -> None:
    assert S.worst_case_peak() < 0.9


def test_the_slice_and_layer_pads_say_what_may_and_may_not_be_heard() -> None:
    fx = S.jam_kits()[S.JAM_FX]
    sl = S.heard(fx, 1)
    assert sl.freqs == (4_500.0,) and sl.absent_freqs == (7_900.0,)
    assert abs(sl.seconds - 0.3) < 1e-9
    assert S.heard(fx, 2).freqs == (4_700.0, 7_700.0)
    assert abs(S.heard(fx, 4).freqs[0] - 5_100.0) < 1e-6          # pitch +7 lands here
    assert S.heard(fx, 9).gain_r == 0.0 and S.heard(fx, 10).gain_l == 0.0
    assert S.heard(fx, 15).shape == "loop" and S.heard(fx, 14).shape == "hold"
    assert S.heard(fx, 7).level_known is False                     # the low-pass decides


def test_kits_are_written_deterministically_with_the_parser_fields(tmp_path: Path) -> None:
    a = S.write_jam_kits(tmp_path / "a")
    b = S.write_jam_kits(tmp_path / "b")

    def digest(root: Path) -> str:
        h = hashlib.sha256()
        for p in sorted(root.rglob("*")):
            if p.is_file():
                h.update(p.relative_to(root).as_posix().encode())
                h.update(p.read_bytes())
        return h.hexdigest()

    for name in S.KIT_NAMES:
        assert digest(a[name]) == digest(b[name])
    fx = json.loads((a[S.JAM_FX] / "kit.json").read_text())
    pads = fx["pads"]
    assert pads[4]["pitch"] == 7 and pads[6]["reverse"] is True and pads[7]["lp"] == 1900
    assert pads[8]["hp"] == 1967 and pads[12]["choke_group"] == 1 and pads[15]["playbackMode"] == 2
    assert isinstance(pads[1]["samples"][0], dict) and pads[1]["samples"][0]["start"] == 22050
    p = json.loads((a[S.JAM_P] / "kit.json").read_text())
    assert p["engine"] == "pitched" and p["instrument"]["max_voices"] == 10
    assert p["instrument"]["pad_notes"] == list(range(85, 117, 2))
    assert "gains" not in json.dumps(fx)
