"""RecordingSerial writes an ndjson transcript; ReplaySerial plays it back identically."""
from __future__ import annotations

import json
import time
from pathlib import Path

import pytest

from crosspad_hil.record import (
    TRANSCRIPT_VERSION,
    RecordingSerial,
    ReplayMismatch,
    ReplaySerial,
    recording_factory,
    replay_factory,
)
from tests.fakes import FakeSerial

# Reply lines exactly as hil_control.cpp formats them (handle_kit_status / handle_app_list).
SCRIPT = [
    ("KIT_STATUS", "KITSTATUS: current=3 loading=0 pending=-1 name=DRUMS"),
    ("APP_LIST", "APPS: Sampler,Sequencer,Settings running=-"),
    ("PAD_PRESS 0 100", "OK"),
]


def drive(ser: object) -> list[bytes]:
    """The session every test records and replays: hygiene, three commands, one timeout."""
    ser.dtr = False  # type: ignore[attr-defined]
    ser.rts = False  # type: ignore[attr-defined]
    got: list[bytes] = []
    ser.write(b"KIT_STATUS\n")  # type: ignore[attr-defined]
    got.append(ser.readline())  # type: ignore[attr-defined]
    ser.write(b"APP_LIST\n")  # type: ignore[attr-defined]
    got.append(ser.readline())  # type: ignore[attr-defined]
    got.append(ser.readline())  # type: ignore[attr-defined]  # nothing pending → b""
    ser.write(b"PAD_PRESS 0 100\n")  # type: ignore[attr-defined]
    got.append(ser.read(64))  # type: ignore[attr-defined]
    ser.close()  # type: ignore[attr-defined]
    return got


@pytest.fixture
def transcript(tmp_path: Path) -> tuple[Path, list[bytes]]:
    fake = FakeSerial(script=SCRIPT)
    fake.timeout = 0.05
    path = tmp_path / "session.ndjson"
    rec = RecordingSerial(fake, path)
    got = drive(rec)
    assert fake.control_history[:2] == [("dtr", False), ("rts", False)]
    assert fake.written[:3] == ["KIT_STATUS", "APP_LIST", "PAD_PRESS 0 100"]
    return path, got


def test_recording_format(transcript: tuple[Path, list[bytes]]) -> None:
    path, got = transcript
    lines = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    assert lines[0]["dir"] == "meta" and lines[0]["version"] == TRANSCRIPT_VERSION
    dirs = [entry["dir"] for entry in lines[1:]]
    # ctl, ctl, tx, rx, tx, rx, tx, rx — the empty readline() is not recorded
    assert dirs == ["ctl", "ctl", "tx", "rx", "tx", "rx", "tx", "rx"]
    assert lines[1] == {"t": lines[1]["t"], "dir": "ctl", "attr": "dtr", "value": False}
    assert bytes.fromhex(lines[3]["data"]) == b"KIT_STATUS\n"
    assert bytes.fromhex(lines[4]["data"]) == b"KITSTATUS: current=3 loading=0 pending=-1 name=DRUMS\r\n"
    ts = [entry["t"] for entry in lines]
    assert ts == sorted(ts) and ts[0] < 0.01
    assert got[0] == b"KITSTATUS: current=3 loading=0 pending=-1 name=DRUMS\r\n"
    assert got[2] == b""
    assert got[3] == b"OK\r\n"


def test_replay_identical(transcript: tuple[Path, list[bytes]]) -> None:
    path, recorded = transcript
    rep = ReplaySerial(path)
    replayed = drive(rep)
    assert replayed == recorded
    assert rep.control_history == [("dtr", False), ("rts", False)]
    assert rep.written == [b"KIT_STATUS\n", b"APP_LIST\n", b"PAD_PRESS 0 100\n"]
    assert rep.remaining() == 0
    rep.assert_exhausted()


def test_replay_is_instant_at_speed_zero(transcript: tuple[Path, list[bytes]]) -> None:
    path, _ = transcript
    rep = ReplaySerial(path, speed=0.0)
    rep.timeout = 0.01
    t0 = time.monotonic()
    drive(rep)
    assert time.monotonic() - t0 < 0.5


def test_replay_scales_delays(tmp_path: Path) -> None:
    path = tmp_path / "slow.ndjson"
    entries = [
        {"t": 0.0, "dir": "meta", "version": TRANSCRIPT_VERSION, "port": None, "baudrate": None},
        {"t": 0.0, "dir": "tx", "data": b"KIT_STATUS\n".hex()},
        {"t": 0.20, "dir": "rx", "data": b"KITSTATUS: current=1 loading=0 pending=-1 name=A\r\n".hex()},
    ]
    path.write_text("\n".join(json.dumps(e) for e in entries) + "\n", encoding="utf-8")
    rep = ReplaySerial(path, speed=0.5)
    rep.write(b"KIT_STATUS\n")
    t0 = time.monotonic()
    line = rep.readline()
    elapsed = time.monotonic() - t0
    assert line.startswith(b"KITSTATUS:")
    assert 0.08 <= elapsed < 0.5


def test_replay_mismatch_raises(transcript: tuple[Path, list[bytes]]) -> None:
    path, _ = transcript
    rep = ReplaySerial(path)
    with pytest.raises(ReplayMismatch) as ei:
        rep.write(b"APP_LIST\n")
    assert "KIT_STATUS" in str(ei.value)
    assert "APP_LIST" in str(ei.value)


def test_replay_lenient_accepts_mismatch(transcript: tuple[Path, list[bytes]]) -> None:
    path, _ = transcript
    rep = ReplaySerial(path, lenient=True)
    rep.write(b"APP_LIST\n")  # accepted, cursor moves past the recorded KIT_STATUS tx
    assert rep.readline() == b"KITSTATUS: current=3 loading=0 pending=-1 name=DRUMS\r\n"


def test_replay_readline_splits_multi_line_chunk(tmp_path: Path) -> None:
    path = tmp_path / "chunk.ndjson"
    chunk = b"ENCGROUP: count=2\r\n  [0] 0x3fc9a000 Sampler\r\n  [1] 0x3fc9a100 Settings\r\n"
    entries = [
        {"t": 0.0, "dir": "meta", "version": TRANSCRIPT_VERSION, "port": None, "baudrate": None},
        {"t": 0.0, "dir": "tx", "data": b"ENC_GROUP\n".hex()},
        {"t": 0.01, "dir": "rx", "data": chunk.hex()},
    ]
    path.write_text("\n".join(json.dumps(e) for e in entries) + "\n", encoding="utf-8")
    rep = ReplaySerial(path)
    rep.write(b"ENC_GROUP\n")
    assert rep.readline() == b"ENCGROUP: count=2\r\n"
    assert rep.readline() == b"  [0] 0x3fc9a000 Sampler\r\n"
    assert rep.read(4) == b"  [1"
    assert rep.readline() == b"] 0x3fc9a100 Settings\r\n"
    assert rep.readline() == b""


def test_replay_does_not_serve_reply_before_its_command(transcript: tuple[Path, list[bytes]]) -> None:
    path, _ = transcript
    rep = ReplaySerial(path)
    rep.timeout = 0.01
    assert rep.readline() == b""  # cursor sits on the first tx: nothing to read yet
    rep.write(b"KIT_STATUS\n")
    assert rep.readline().startswith(b"KITSTATUS:")
    assert rep.readline() == b""  # the APPS reply waits for APP_LIST


def test_replay_exhausted_reports_leftover(transcript: tuple[Path, list[bytes]]) -> None:
    path, _ = transcript
    rep = ReplaySerial(path)
    rep.write(b"KIT_STATUS\n")
    assert rep.remaining() > 0
    with pytest.raises(AssertionError):
        rep.assert_exhausted()


def test_factories(tmp_path: Path) -> None:
    out = tmp_path / "f.ndjson"
    fake = FakeSerial(script=SCRIPT)
    made: list[str] = []

    def base(path: str, **kw: object) -> FakeSerial:
        made.append(path)
        return fake

    rec = recording_factory(out, base=base)("/dev/ttyACM9", baud=115200)
    assert isinstance(rec, RecordingSerial)
    assert made == ["/dev/ttyACM9"]
    rec.write(b"KIT_STATUS\n")
    assert rec.readline().startswith(b"KITSTATUS:")
    rec.close()
    rep = replay_factory(out)("/dev/ttyACM9", baud=115200)
    assert isinstance(rep, ReplaySerial)
    rep.write(b"KIT_STATUS\n")
    assert rep.readline().startswith(b"KITSTATUS:")


def test_checked_in_fixture_replays() -> None:
    fixture = Path(__file__).parent / "fixtures" / "transcripts" / "kit_status_app_list.ndjson"
    rep = ReplaySerial(fixture)
    rep.dtr = False
    rep.rts = False
    rep.write(b"KIT_STATUS\n")
    assert rep.readline() == b"KITSTATUS: current=3 loading=0 pending=-1 name=DRUMS\r\n"
    rep.write(b"APP_LIST\n")
    assert rep.readline() == b"APPS: Sampler,Sequencer,Settings running=-\r\n"
    rep.assert_exhausted()
