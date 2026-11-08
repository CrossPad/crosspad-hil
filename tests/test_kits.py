"""`crosspad_hil.kits`: build_kit's WAV conversion (rate/bit-depth/channels),
kit.json shape, pad-count edge cases, dirname derivation, and the
push_kit/reboot/scan helpers moved here from scenarios/_audio.py.

Everything here is stdlib (`wave`, `struct`, `math`) — no numpy, no device.
"""
from __future__ import annotations

import json
import math
import struct
import wave
from pathlib import Path

import pytest

from crosspad_hil import kits
from crosspad_hil.errors import BAD_ARGS, HilError

# --------------------------------------------------------------------------- synthetic WAVs

def _tone_frames(freq: float, seconds: float, rate: int, channels: int) -> list[tuple[int, ...]]:
    n = int(rate * seconds)
    out = []
    for i in range(n):
        v = math.sin(2 * math.pi * freq * i / rate)
        s = int(v * 16000)
        out.append((s,) * channels)
    return out


def write_pcm_wav(path: Path, frames: list[tuple[int, ...]], *, channels: int, rate: int,
                  sampwidth: int) -> None:
    """Write PCM WAV at an arbitrary bit depth (8/16/24/32), stdlib `wave` + raw bytes."""
    raw = bytearray()
    for frame in frames:
        for v in frame:
            if sampwidth == 1:
                raw.append(max(0, min(255, int(v / 256) + 128)))
            elif sampwidth == 2:
                raw += struct.pack("<h", max(-32768, min(32767, v)))
            elif sampwidth == 3:
                iv = max(-8388608, min(8388607, v * 256))
                iv &= 0xFFFFFF
                raw += bytes([iv & 0xFF, (iv >> 8) & 0xFF, (iv >> 16) & 0xFF])
            elif sampwidth == 4:
                raw += struct.pack("<i", max(-2**31, min(2**31 - 1, v * 65536)))
            else:
                raise ValueError(sampwidth)
    with wave.open(str(path), "wb") as w:
        w.setnchannels(channels)
        w.setsampwidth(sampwidth)
        w.setframerate(rate)
        w.writeframes(bytes(raw))


def read_pcm_wav(path: Path) -> tuple[int, int, int, int]:
    """(channels, sampwidth, rate, nframes) of a WAV file."""
    with wave.open(str(path), "rb") as w:
        return w.getnchannels(), w.getsampwidth(), w.getframerate(), w.getnframes()


# --------------------------------------------------------------------------- build_kit

def test_build_kit_converts_rate_bitdepth_and_channels(tmp_path: Path) -> None:
    src = tmp_path / "src"
    src.mkdir()
    # 8-bit mono 22050 Hz
    write_pcm_wav(src / "a.wav", _tone_frames(440, 0.5, 22050, 1),
                  channels=1, rate=22050, sampwidth=1)
    # 16-bit stereo 48000 Hz
    write_pcm_wav(src / "b.wav", _tone_frames(220, 0.5, 48000, 2),
                  channels=2, rate=48000, sampwidth=2)
    # 24-bit mono 44100 Hz (already the kit rate: no resampling)
    write_pcm_wav(src / "c.wav", _tone_frames(330, 0.5, 44100, 1),
                  channels=1, rate=44100, sampwidth=3)

    kit_dir = kits.build_kit(src, tmp_path / "out", "My Test Kit", rate=44100)
    assert kit_dir == tmp_path / "out" / "MY_TEST_KIT"
    assert (kit_dir / "kit.json").is_file()

    for pad, (_name, src_rate) in enumerate([("a.wav", 22050), ("b.wav", 48000), ("c.wav", 44100)]):
        wav_path = kit_dir / "SAMPLES" / f"pad{pad:02d}.wav"
        assert wav_path.is_file()
        channels, sampwidth, rate, nframes = read_pcm_wav(wav_path)
        assert channels == 2                      # mono duplicated, stereo kept
        assert sampwidth == 2                      # always 16-bit out
        assert rate == 44100
        expected = round(int(src_rate * 0.5) * 44100 / src_rate)
        assert abs(nframes - expected) <= 1        # within +/-1 frame


def test_build_kit_kit_json_shape(tmp_path: Path) -> None:
    src = tmp_path / "src"
    src.mkdir()
    write_pcm_wav(src / "kick.wav", _tone_frames(60, 0.2, 44100, 2),
                  channels=2, rate=44100, sampwidth=2)
    kit_dir = kits.build_kit(src, tmp_path / "out", "Drums", rate=44100)
    kit = json.loads((kit_dir / "kit.json").read_text())

    assert kit["name"] == "Drums"
    assert kit["author"] == "crosspad-hil"
    assert isinstance(kit["description"], list) and kit["description"]
    assert kit["version"] == "1.0"
    assert kit["created_at"]                      # ISO date, don't pin "today"
    assert len(kit["created_at"]) == 10 and kit["created_at"][4] == "-"
    assert len(kit["pads"]) == 16

    pad0 = kit["pads"][0]
    assert set(pad0) == {"color", "color_pressed", "choke_group", "cycle", "pitch",
                         "volume", "pan", "samples"}
    assert pad0["color"] == pad0["color_pressed"]
    assert pad0["choke_group"] == 0
    assert pad0["cycle"] is False
    assert pad0["pitch"] == 0
    assert pad0["volume"] == 100
    assert pad0["pan"] == 0
    assert pad0["samples"] == ["pad00.wav"]

    # colours differ pad to pad (a real palette, not one colour repeated)
    colors = {p["color"] for p in kit["pads"]}
    assert len(colors) > 1


def test_build_kit_fewer_than_sixteen_leaves_empty_pads(tmp_path: Path) -> None:
    src = tmp_path / "src"
    src.mkdir()
    write_pcm_wav(src / "only_one.wav", _tone_frames(440, 0.1, 44100, 2),
                  channels=2, rate=44100, sampwidth=2)
    kit_dir = kits.build_kit(src, tmp_path / "out", "One Pad", rate=44100)
    kit = json.loads((kit_dir / "kit.json").read_text())
    assert kit["pads"][0]["samples"] == ["only_one.wav".replace("only_one.wav", "pad00.wav")]
    for pad in kit["pads"][1:]:
        assert pad["samples"] == []
    assert not (kit_dir / "SAMPLES" / "pad01.wav").exists()


def test_build_kit_more_than_sixteen_is_bad_args(tmp_path: Path) -> None:
    src = tmp_path / "src"
    src.mkdir()
    for i in range(17):
        write_pcm_wav(src / f"s{i:02d}.wav", _tone_frames(440, 0.05, 44100, 1),
                      channels=1, rate=44100, sampwidth=2)
    with pytest.raises(HilError) as e:
        kits.build_kit(src, tmp_path / "out", "Too Many", rate=44100)
    assert e.value.code == BAD_ARGS
    assert e.value.details["count"] == 17


def test_build_kit_pads_sorted_case_insensitively_by_name(tmp_path: Path) -> None:
    src = tmp_path / "src"
    src.mkdir()
    for name in ["Bravo.wav", "alpha.wav", "Charlie.WAV"]:
        write_pcm_wav(src / name, _tone_frames(300, 0.05, 44100, 1),
                      channels=1, rate=44100, sampwidth=2)
    kit_dir = kits.build_kit(src, tmp_path / "out", "Sorted", rate=44100)
    kit = json.loads((kit_dir / "kit.json").read_text())
    # alpha < Bravo < Charlie, case-insensitively
    assert [p["samples"] for p in kit["pads"][:3]] == \
        [["pad00.wav"], ["pad01.wav"], ["pad02.wav"]]


@pytest.mark.parametrize("name,expected", [
    ("My Test Kit", "MY_TEST_KIT"),
    ("drum & bass!!", "DRUM_BASS"),
    ("---", "KIT"),
    ("", "KIT"),
    ("Already_Fine", "ALREADY_FINE"),
])
def test_dirname_derivation(name: str, expected: str, tmp_path: Path) -> None:
    src = tmp_path / "src"
    src.mkdir()
    kit_dir = kits.build_kit(src, tmp_path / "out", name, rate=44100)
    assert kit_dir.name == expected


def test_build_kit_rejects_a_non_pcm_wav(tmp_path: Path) -> None:
    src = tmp_path / "src"
    src.mkdir()
    bad = src / "float.wav"
    # A minimal RIFF/WAVE header with format tag 3 (IEEE float) — wave.open()
    # refuses to read it; build_kit must rename the failure to the file it's in.
    fmt_chunk = struct.pack("<HHLLHH", 3, 1, 44100, 44100 * 4, 4, 32)
    data_chunk = struct.pack("<4f", 0.0, 0.1, -0.1, 0.0)
    riff = b"WAVEfmt " + struct.pack("<L", len(fmt_chunk)) + fmt_chunk \
        + b"data" + struct.pack("<L", len(data_chunk)) + data_chunk
    bad.write_bytes(b"RIFF" + struct.pack("<L", len(riff)) + riff)
    with pytest.raises(HilError) as e:
        kits.build_kit(src, tmp_path / "out", "Bad", rate=44100)
    assert e.value.code == BAD_ARGS
    assert "float.wav" in e.value.message
