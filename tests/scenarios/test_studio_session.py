"""Scenario `studio_session` without a board: the fixture gate and the host-side checks."""
from __future__ import annotations

import gzip
import hashlib
import json
import struct
from pathlib import Path

import pytest

from crosspad_hil.devices import Device, Ports, SerialPortInfo, UsbMode
from crosspad_hil.scenarios import base, get, studio_session
from crosspad_hil.scenarios.studio_session import (
    bar_frames,
    compare_set,
    parse_live_set,
    verify_fixture,
    wav_frames,
)


def make_device() -> Device:
    return Device(
        id="dev_ab12", serial="CP-1", usb_mode=UsbMode.DEFAULT,
        ports=Ports(cdc=SerialPortInfo(path="/dev/ttyACM0", vid=0x303A, pid=0x3456,
                                       serial="CP-1", product="CrossPad", location="1-2.1")),
    )


def make_fixture(root: Path) -> Path:
    """sd/ with one kit and an sd.sha256 that matches it, as the studio fixture lays it out."""
    kit = root / "sd" / "crosspad" / "kits" / "Malpa - Swiete Slowa"
    (kit / "SAMPLES").mkdir(parents=True)
    (kit / "kit.json").write_text(json.dumps({"name": "ŚWIĘTE SŁOWA", "pads": []}),
                                  encoding="utf-8")
    (kit / "SAMPLES" / "1.wav").write_bytes(b"RIFF....WAVE")
    lines = []
    for f in sorted(p for p in (root / "sd").rglob("*") if p.is_file()):
        rel = f.relative_to(root / "sd").as_posix()
        lines.append(f"{hashlib.sha256(f.read_bytes()).hexdigest()}  ./{rel}")
    (root / "sd.sha256").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return kit


def test_registered():
    assert get("studio_session").name == "studio_session"


def test_fixture_kits_by_name(tmp_path):
    kit = make_fixture(tmp_path)
    kits, problems = verify_fixture(tmp_path)
    assert problems == []
    assert kits == {"ŚWIĘTE SŁOWA": kit}


def test_a_changed_sample_is_not_the_session_kit(tmp_path):
    kit = make_fixture(tmp_path)
    (kit / "SAMPLES" / "1.wav").write_bytes(b"other")
    _, problems = verify_fixture(tmp_path)
    assert problems == ["sha256 differs: ./crosspad/kits/Malpa - Swiete Slowa/SAMPLES/1.wav"]


def test_a_folder_without_sums_is_not_the_fixture(tmp_path):
    _, problems = verify_fixture(tmp_path)
    assert "not the studio fixture" in problems[0]


def test_no_fixture_is_a_bench_problem(tmp_path, monkeypatch):
    monkeypatch.delenv(studio_session.FIXTURE_ENV, raising=False)
    report = base.run_scenario("studio_session", studio_session.Params(), device=make_device(),
                               workdir=tmp_path, log=lambda s: None)
    assert report.exit_code == 2
    assert studio_session.FIXTURE_ENV in report.summary


def test_a_kit_the_fixture_lacks_is_a_bench_problem(tmp_path):
    make_fixture(tmp_path / "fx")
    params = studio_session.Params(fixture=str(tmp_path / "fx"), kit="BASIC")
    report = base.run_scenario("studio_session", params, device=make_device(),
                               workdir=tmp_path / "w", log=lambda s: None)
    assert report.exit_code == 2
    assert "'BASIC'" in report.summary


def wav_header(frames: int, *, channels: int = 2, rate: int = 44100, extra: bytes = b"") -> bytes:
    fmt = struct.pack("<HHIIHH", 1, channels, rate, rate * channels * 2, channels * 2, 16)
    body = b"fmt " + struct.pack("<I", len(fmt)) + fmt + extra
    data = struct.pack("<I", frames * channels * 2)
    return b"RIFF" + struct.pack("<I", 0) + b"WAVE" + body + b"data" + data


def test_wav_frames_reads_the_data_chunk():
    assert wav_frames(wav_header(227613)) == (227613, 2, 44100)


def test_wav_frames_skips_chunks_before_data():
    odd = b"LIST" + struct.pack("<I", 3) + b"abc\x00"     # odd size: one pad byte
    assert wav_frames(wav_header(10, channels=1, extra=odd)) == (10, 1, 44100)


def test_wav_frames_refuses_what_is_not_a_wav():
    with pytest.raises(ValueError):
        wav_frames(b"ID3\x04" + b"\x00" * 60)


def test_bar_frames_at_the_session_tempo():
    assert bar_frames(2, 93) == pytest.approx(227612.903, abs=1e-3)


LIVE_SET = """<?xml version="1.0" encoding="UTF-8"?>
<Ableton><LiveSet>
 <Tracks>
  <AudioTrack Id="1"><Name><EffectiveName Value="1 Audio"/></Name>
   <DeviceChain><Mixer><Volume><Manual Value="0.9"/></Volume></Mixer>
    <MainSequencer><ClipSlotList/><Sample><ArrangerAutomation><Events>
     <AudioClip Id="0" Time="0"><SampleRef><FileRef>
      <RelativePath Value="clips/seq_1.wav"/></FileRef></SampleRef></AudioClip>
    </Events></ArrangerAutomation></Sample></MainSequencer></DeviceChain></AudioTrack>
  <AudioTrack Id="2"><Name><EffectiveName Value="2 Audio"/></Name>
   <DeviceChain><Mixer><Volume><Manual Value="0.75"/></Volume></Mixer>
    <MainSequencer><Sample><ArrangerAutomation><Events>
     <AudioClip Id="1" Time="4"><SampleRef><FileRef>
      <RelativePath Value="clips/rec_001.wav"/></FileRef></SampleRef></AudioClip>
    </Events></ArrangerAutomation></Sample></MainSequencer></DeviceChain></AudioTrack>
 </Tracks>
 <MainTrack><DeviceChain><Mixer><Tempo><Manual Value="93"/></Tempo></Mixer></DeviceChain>
 </MainTrack>
</LiveSet></Ableton>"""


def test_parse_live_set_from_the_gzip_the_board_writes():
    live = parse_live_set(gzip.compress(LIVE_SET.encode()))
    assert live["tempo"] == 93.0
    assert [t["name"] for t in live["tracks"]] == ["1 Audio", "2 Audio"]
    assert [t["volume"] for t in live["tracks"]] == [0.9, 0.75]
    assert live["tracks"][1]["clips"] == [{"time": 4.0, "path": "clips/rec_001.wav"}]


def song(*bars_per_track: list[int]) -> dict:
    return {"bpm": 93, "tracks": [{"clips": [{"file": f"clips/c{b}.wav", "bar": b} for b in bars]}
                                  for bars in bars_per_track]}


def test_a_set_that_matches_the_song():
    live = parse_live_set(LIVE_SET.encode())
    assert compare_set(song([0], [1]), live, 93) == []


def test_seq_blocks_are_not_audio_clips():
    live = parse_live_set(LIVE_SET.encode())
    s = song([0], [1])
    s["tracks"][0]["clips"].append({"bar": 4, "notes": "0:36:100"})
    assert compare_set(s, live, 93) == []


def test_a_clip_on_another_bar_and_the_wrong_tempo():
    live = parse_live_set(LIVE_SET.encode())
    diffs = compare_set(song([0], [2]), live, 120)
    assert diffs == ["the Set's tempo is 93.0 (want 120)",
                     "track 2: clips at beats [4.0] in the Set, [8.0] in song.json"]


def test_a_track_missing_from_the_set():
    live = parse_live_set(LIVE_SET.encode())
    assert compare_set(song([0], [1], [], []), live, 93) == [
        "the Set has 2 audio tracks, song.json 4"]
