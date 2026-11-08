"""The fixture kits this package builds and pushes itself.

Two claims are tested. First, generation is deterministic: the same arguments
produce the same bytes, run to run, so a card provisioned today and one
provisioned last week score the same recording the same way. Second — whenever
platform-idf is checked out beside this repo — `tools/make_sine_kits.py`'s own
generator produces *byte-identical* kits, because "the hand-run tool and the
scenario push the same fixture" is a claim about bytes, not about intent.
"""
from __future__ import annotations

import json
import wave
from pathlib import Path

import pytest

from crosspad_hil import fixtures

PLATFORM_TOOLS = Path(__file__).resolve().parents[2] / "platform-idf" / "tools"


def digest(root: Path) -> dict[str, bytes]:
    """{path relative to root: contents} for every file under it."""
    return {str(p.relative_to(root)): p.read_bytes()
            for p in sorted(root.rglob("*")) if p.is_file()}


def test_write_sine_kits_lays_out_both_kits(tmp_path: Path) -> None:
    built = fixtures.write_sine_kits(tmp_path)
    assert set(built) == {"HIL SINE A", "HIL SINE B"}
    # the key is the name KIT_LIST reports, the directory is what the card gets
    assert built["HIL SINE A"].name == "HIL_SINE_A"
    assert built["HIL SINE B"].name == "HIL_SINE_B"
    for name, root in built.items():
        kit = json.loads((root / "kit.json").read_text())
        assert kit["name"] == name
        assert len(kit["pads"]) == 16
        assert kit["pads"][0] == {"color": 2162448, "color_pressed": 2162448,
                                  "choke_group": 0, "cycle": False, "pitch": 0,
                                  "volume": 100, "pan": 0, "samples": ["pad00.wav"]}
        assert "playbackMode" not in kit["pads"][0]      # absent = one-shot
        assert len(list((root / "SAMPLES").glob("*.wav"))) == 16


def test_the_samples_are_stereo_44k1_two_second_16_bit(tmp_path: Path) -> None:
    built = fixtures.write_sine_kits(tmp_path)
    with wave.open(str(built["HIL SINE A"] / "SAMPLES" / "pad00.wav")) as w:
        assert (w.getnchannels(), w.getframerate(), w.getsampwidth()) == (2, 44100, 2)
        assert w.getnframes() == 88200                   # 2.0 s at 44.1 kHz


def test_generation_is_deterministic(tmp_path: Path) -> None:
    fixtures.write_sine_kits(tmp_path / "one")
    fixtures.write_sine_kits(tmp_path / "two")
    assert digest(tmp_path / "one") == digest(tmp_path / "two")


def test_seconds_and_peak_change_the_bytes(tmp_path: Path) -> None:
    """The knobs have to reach the samples, or a shorter fixture would silently
    still be 2 s and every scoring window would be measured against the wrong one."""
    a = fixtures.write_sine_kits(tmp_path / "a")["HIL SINE A"]
    b = fixtures.write_sine_kits(tmp_path / "b", seconds=1.0)["HIL SINE A"]
    c = fixtures.write_sine_kits(tmp_path / "c", peak=0.25)["HIL SINE A"]
    wav = Path("SAMPLES") / "pad00.wav"
    assert len(digest(b)[str(wav)]) < len(digest(a)[str(wav)])
    assert digest(c)[str(wav)] != digest(a)[str(wav)]


def test_the_two_ladders_are_interleaved() -> None:
    assert fixtures.freq_a(0) == 300.0 and fixtures.freq_b(0) == 411.0
    assert fixtures.freq_a(15) == 300.0 + 222.0 * 15
    for p in range(16):
        assert fixtures.freq_b(p) - fixtures.freq_a(p) == pytest.approx(111.0)


def test_kit_files_sends_kit_json_last(tmp_path: Path) -> None:
    """kit.json is what the kit scan keys off: a push interrupted before it
    leaves an incomplete directory, not a kit claiming samples it does not have."""
    root = fixtures.write_sine_kits(tmp_path)["HIL SINE B"]
    files = fixtures.kit_files(root)
    assert len(files) == 17
    assert files[-1].name == "kit.json"
    assert all(f.parent.name == "SAMPLES" for f in files[:-1])


@pytest.mark.skipif(not (PLATFORM_TOOLS / "make_sine_kits.py").is_file(),
                    reason="platform-idf is not checked out beside this repo")
def test_byte_identical_to_the_platform_idf_tool(tmp_path: Path,
                                                 monkeypatch: pytest.MonkeyPatch) -> None:
    """`tools/make_sine_kits.py` prefers this package when it is importable; its
    own generator is the fallback for a bare checkout, and it is the one compared
    here — the two implementations have to agree sample for sample."""
    import importlib.util

    monkeypatch.syspath_prepend(str(PLATFORM_TOOLS))
    spec = importlib.util.spec_from_file_location(
        "make_sine_kits_under_test", PLATFORM_TOOLS / "make_sine_kits.py")
    tool = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(tool)

    for dirname, name, freq in tool.KITS:                # the tool's own generator
        tool.write_kit(tmp_path / "tool" / dirname, name, freq)
    fixtures.write_sine_kits(tmp_path / "lib")
    assert digest(tmp_path / "tool") == digest(tmp_path / "lib")


def test_pitched_kit_shape(tmp_path: Path) -> None:
    out = fixtures.write_pitched_kits(tmp_path)
    assert list(out) == ["HIL PITCHED"] and out["HIL PITCHED"].name == "HIL_PITCHED"
    kit = json.loads((out["HIL PITCHED"] / "kit.json").read_text())
    assert kit["engine"] == "pitched"
    assert [z["root"] for z in kit["instrument"]["zones"]] == [57, "auto"]
    assert kit["instrument"]["pad_notes"] == list(range(60, 76))
    assert sorted(p.name for p in (out["HIL PITCHED"] / "SAMPLES").iterdir()) == \
        ["z57.wav", "z69.wav"]
    assert set(fixtures.write_live_set_kits(tmp_path / "all")) == \
        {"HIL SINE A", "HIL SINE B", "HIL PITCHED"}


@pytest.mark.skipif(not (PLATFORM_TOOLS / "make_pitched_kit.py").is_file(),
                    reason="platform-idf is not checked out beside this repo")
def test_pitched_kit_byte_identical_to_the_platform_idf_tool(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import importlib.util

    monkeypatch.syspath_prepend(str(PLATFORM_TOOLS))
    spec = importlib.util.spec_from_file_location(
        "make_pitched_kit_under_test", PLATFORM_TOOLS / "make_pitched_kit.py")
    tool = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(tool)
    for zones in (1, 2, 4):
        tool.write_kit(tmp_path / f"tool{zones}", zones=zones, rate=44100,
                       seconds=tool.SECONDS_BY_ZONES[zones])
        fixtures.write_pitched_kit(tmp_path / f"lib{zones}", zones=zones)
        assert digest(tmp_path / f"tool{zones}") == digest(tmp_path / f"lib{zones}")
