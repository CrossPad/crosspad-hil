"""Scenario `song_bounce` against a fake board that models the song engine's replies."""
from __future__ import annotations

import threading
from dataclasses import dataclass
from pathlib import Path

from crosspad_hil.cdc import CdcLink
from crosspad_hil.devices import Device, Ports, SerialPortInfo, UsbMode
from crosspad_hil.scenarios import base, song_bounce
from tests.fakes import FakeSerial

CDC = SerialPortInfo(path="/dev/ttyACM0", vid=0x303A, pid=0x3456, serial="CP-1",
                     product="CrossPad", location="1-2.1")


class SongBoard(FakeSerial):
    """Enough of SEQ / ASONG / RESET_LOG to walk the scenario; faults on demand."""

    def __init__(self, *, bounce_fails: bool = False, underruns: int = 0,
                 resets_on_restart: bool = False, open_fails: bool = False) -> None:
        super().__init__()
        self.bounce_fails = bounce_fails
        self.underruns = underruns
        self.resets_on_restart = resets_on_restart
        self.open_fails = open_fails
        self.ev = [7, 0, 0, 0, 0, 0, 0, 0]     # the player's own take in scene 1
        self.scenes = 1
        # The scenes are the open song's (core SongScenes): another song's come
        # in once nothing plays them; the songs not open keep theirs here.
        self.song = "Song 1"
        self.wanted = "Song 1"
        self.shelf: dict[str, tuple[int, list[int]]] = {}
        self.opening = ""
        self.seq_running = False
        self.bounce_err = ""
        self.active = 0
        self.countin = 1
        self.recording = False
        self.open_state = 0
        self.open_polls = 0
        self.bounce = 0
        self.bounce_polls = 0
        self.clips = [0, 0, 0, 0]
        self.playing = False
        self.plays = 0
        self.resets = 3

    def _seq(self) -> str:
        return (f"SEQ: mode={2 if self.recording else 1} state=4 scenes={self.scenes} "
                f"scene={self.active}>{self.active} bars=4/1/4/4/4/4/4/4 "
                f"ev={'/'.join(str(e) for e in self.ev)} bpm=120.0 quant=2 pos=0.250 dirty=0 "
                f"erase=0 countin={self.countin} file=/sdcard/SONGS/{self.song}/sequences.json")

    def _asong(self, ok: int = 1) -> str:
        if self.open_state == 1:
            self.open_polls += 1
            if self.open_polls >= 2:
                self.open_state = 3 if self.open_fails else 2
                if self.open_state == 2:
                    self.wanted = self.opening
        if self.bounce == 1:
            self.bounce_polls += 1
            if self.bounce_polls >= 2:
                self.bounce = 3 if self.bounce_fails else 2
                if not self.bounce_fails:
                    self.clips[0] += 1
        err = f" err={self.bounce_err or 'card too slow'}" if self.bounce == 3 else ""
        return (f"ASONG: ok={ok} open={self.open_state} bpm=120 len=1 sync=0 play={int(self.playing)} "
                f"pos={'1.50' if self.playing else '0.00'} under={self.underruns if self.playing else 0} "
                f"openfail=0 bounce={self.bounce} done=0 total=96000 "
                f"clips={'/'.join(str(c) for c in self.clips)} file=Test_Kit_S1_01.wav "
                f"project=hil song{err}")

    def _sweep(self) -> None:
        if self.wanted == self.song or self.seq_running or self.playing:
            return
        self.shelf[self.song] = (self.scenes, self.ev)
        self.scenes, self.ev = self.shelf.pop(self.wanted, (1, [0] * 8))
        self.song, self.active = self.wanted, 0

    def write(self, data: bytes) -> int:
        for raw in data.decode("utf-8", "replace").split("\n"):
            cmd = raw.strip()
            if not cmd:
                continue
            self.written.append(cmd)
            self._sweep()
            if cmd.startswith("SEQ"):
                arg = cmd[3:].strip()
                if arg == "REC":
                    self.recording = True
                    self.seq_running = True
                elif arg in ("PLAY", "STOP"):
                    self.recording = False
                    self.seq_running = arg == "PLAY"
                    self._sweep()
                elif arg == "ADD":
                    self.scenes += 1
                elif arg == "CLEAR":
                    self.ev[self.active] = 0
                elif arg.startswith("SCENE "):
                    self.active = int(arg.split()[1])
                elif arg.startswith("COUNTIN "):
                    self.countin = int(arg.split()[1])
                elif arg.startswith("DEL "):
                    del self.ev[int(arg.split()[1])]
                    self.ev.append(0)
                    self.scenes -= 1
                    self.active = 0
                self.feed([self._seq()])
            elif cmd.startswith("PAD_PRESS"):
                if self.recording:
                    self.ev[self.active] += 1
                self.feed(["OK"])
            elif cmd.startswith("PAD_RELEASE"):
                self.feed(["OK"])
            elif cmd.startswith("ASONG"):
                arg = cmd[5:].strip()
                if arg.startswith("OPEN"):
                    self.open_state = 1
                    self.open_polls = 0
                    self.opening = arg.split(" ", 1)[1]
                elif arg.startswith("BOUNCE"):
                    self.bounce = 1
                    self.bounce_polls = 0
                    self.bounce_err = ""
                    if int(arg.split()[1]) >= self.scenes:
                        self.bounce, self.bounce_err = 3, "no such scene"
                elif arg.startswith("PLAY"):
                    self.plays += 1
                    self.playing = True
                elif arg == "STOP":
                    if self.resets_on_restart and self.plays > 1:
                        self.resets += 1
                    self.playing = False
                    self.seq_running = False     # the song takes the pattern with it
                    self._sweep()
                self.feed([self._asong()])
            elif cmd.startswith("RESET_LOG"):
                self.feed([f"RSTLOG: n={self.resets} coredump_seq=0"])
            else:
                self._rx.append((cmd + "\r\n").encode())
        return len(data)


@dataclass
class FakeContext(base.Context):
    board: SongBoard = None  # type: ignore[assignment]

    def open_cdc(self) -> CdcLink:
        link = CdcLink(self.device.ports.cdc.path, serial_factory=lambda path, **kw: self.board)
        link.open()
        self.opened.append(link)
        return link


FAST = song_bounce.Params(pads=3, rounds=5, play_s=0.0, poll_s=0.0, bar_s=0.0,
                          open_timeout_s=2.0, bounce_timeout_s=2.0)


def run(tmp_path: Path, board: SongBoard, params: song_bounce.Params = FAST) -> base.Report:
    device = Device(id="dev_ab12", serial="CP-1", usb_mode=UsbMode.DEFAULT, ports=Ports(cdc=CDC))
    ctx = FakeContext(device=device, workdir=tmp_path, cancelled=threading.Event(),
                      log=lambda s: None, board=board)
    return song_bounce.SCENARIO.run(ctx, params, base.Progress())


def test_record_bounce_play_restart_passes(tmp_path) -> None:
    board = SongBoard()
    report = run(tmp_path, board)
    assert report.passed, report.summary
    assert report.exit_code == 0
    assert "SEQ COUNTIN 0" in board.written          # a take that records at once
    assert "ASONG BOUNCE 1 0" in board.written       # a scene of its own, after the player's
    assert board.clips[0] == 2                        # the bounce, and the one after STOP
    assert board.plays == FAST.rounds + 2


def test_the_players_scenes_are_left_as_found(tmp_path) -> None:
    board = SongBoard()
    run(tmp_path, board)
    assert "SEQ CLEAR" not in board.written
    assert board.written[-2:] == ["SEQ DEL 1", "SEQ COUNTIN 1"]
    assert board.shelf["Song 1"] == (1, [7, 0, 0, 0, 0, 0, 0, 0])
    assert board.scenes == 1 and board.countin == 1


def test_the_scene_is_recorded_in_the_song_it_is_bounced_into(tmp_path) -> None:
    """The scenes are the open song's: a scene recorded before OPEN stays with the
    song that was open, and once the pattern stops the project's come in -- the
    bounce after STOP then found no such scene (2.x bench run, 2026-10-20)."""
    board = SongBoard()
    report = run(tmp_path, board)
    assert report.passed, report.summary
    w = board.written
    assert w.index(f"ASONG OPEN {FAST.project}") < w.index("SEQ ADD")


def test_a_failed_bounce_fails_with_its_reason(tmp_path) -> None:
    report = run(tmp_path, SongBoard(bounce_fails=True))
    assert not report.passed
    assert report.exit_code == 1
    assert "card too slow" in report.summary


def test_underruns_while_playing_fail(tmp_path) -> None:
    report = run(tmp_path, SongBoard(underruns=2))
    assert not report.passed
    assert "under=2" in report.summary


def test_a_reset_during_stop_play_fails(tmp_path) -> None:
    report = run(tmp_path, SongBoard(resets_on_restart=True))
    assert not report.passed
    assert "reset" in report.summary


def test_an_unreadable_song_fails(tmp_path) -> None:
    report = run(tmp_path, SongBoard(open_fails=True))
    assert not report.passed
    assert "open" in report.summary
