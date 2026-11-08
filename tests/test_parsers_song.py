"""Replies of the audio song verbs: ASONG, SONG, SEQ, RESET_LOG."""
from __future__ import annotations

from crosspad_hil.parsers import parse_cdc_reply


def test_asong_names_with_spaces_and_an_error() -> None:
    line = ("ASONG: ok=1 open=2 bpm=120 len=8 sync=0 play=1 pos=3.25 under=0 openfail=2 "
            "bounce=3 done=100 total=400 clips=1/0/2/0 file=Kit_S1_01.wav project=My Song 2 "
            "err=card too slow (stalled 900 ms, buffer 740 ms)")
    p = parse_cdc_reply(line)
    assert p is not None and p["kind"] == "asong"
    assert p["ok"] == 1 and p["open"] == 2 and p["play"] == 1
    assert p["pos"] == 3.25
    assert p["openfail"] == 2
    assert p["bounce"] == 3 and p["done"] == 100 and p["total"] == 400
    assert p["clips"] == [1, 0, 2, 0]
    assert p["file"] == "Kit_S1_01.wav"
    assert p["project"] == "My Song 2"
    assert p["err"] == "card too slow (stalled 900 ms, buffer 740 ms)"


def test_asong_without_error_or_file() -> None:
    p = parse_cdc_reply("ASONG: ok=1 open=0 bpm=0 len=0 sync=0 play=0 pos=0.00 under=0 "
                        "openfail=0 bounce=0 done=0 total=0 clips=0/0/0/0 file=- project=")
    assert p is not None and p["kind"] == "asong"
    assert p["file"] is None
    assert p["project"] == ""
    assert p["err"] is None


def test_song_blocks() -> None:
    p = parse_cdc_reply("SONG: mode=1 loop=0 len=3 pos=1 pass=0 scene=1 blocks=0:2,1:1,0:4")
    assert p is not None and p["kind"] == "song"
    assert p["mode"] == 1 and p["loop"] == 0 and p["pos"] == 1
    assert p["blocks"] == [(0, 2), (1, 1), (0, 4)]


def test_song_empty_blocks() -> None:
    p = parse_cdc_reply("SONG: mode=0 loop=1 len=0 pos=0 pass=0 scene=0 blocks=")
    assert p is not None and p["blocks"] == []


def test_seq_file_path_with_spaces() -> None:
    p = parse_cdc_reply("SEQ: mode=2 state=4 scenes=1 scene=0>0 bars=1/4/4/4 ev=3/0/0/0 bpm=120.0 "
                        "quant=2 pos=0.500 dirty=1 erase=0 countin=0 "
                        "file=/sdcard/KITS/My Kit/sequences.json")
    assert p is not None and p["kind"] == "seq"
    assert p["state"] == 4 and p["countin"] == 0
    assert p["ev"] == [3, 0, 0, 0]
    assert p["file"] == "/sdcard/KITS/My Kit/sequences.json"


def test_rstlog_count() -> None:
    p = parse_cdc_reply("RSTLOG: n=7 coredump_seq=2")
    assert p == {"kind": "rstlog", "n": 7, "coredump_seq": 2}
