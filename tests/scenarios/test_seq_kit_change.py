"""Scenario `seq_kit_change` against a fake board that models scenes and kit loads."""
from __future__ import annotations

import threading
from dataclasses import dataclass
from pathlib import Path

from crosspad_hil.cdc import CdcLink
from crosspad_hil.devices import Device, Ports, SerialPortInfo, UsbMode
from crosspad_hil.scenarios import base, seq_kit_change
from tests.fakes import FakeSerial

CDC = SerialPortInfo(path="/dev/ttyACM0", vid=0x303A, pid=0x3456, serial="CP-1",
                     product="CrossPad", location="1-2.1")


class SeqBoard(FakeSerial):
    """SEQ ADD/SCENE/REC/PLAY/DEL, KIT_LIST/KIT_STATUS/KIT_LOAD; old firmware on demand."""

    def __init__(self, *, per_kit: bool = False, scenes: int = 2) -> None:
        super().__init__()
        self.per_kit = per_kit          # the firmware before 26.09: a kit load swaps the scenes
        self.ev = [5] + [0] * 7          # the player's own take in scene 1
        self.scenes = scenes
        self.active = 0
        self.recording = False
        self.kit = 0
        self.countin = 1

    def _seq(self) -> str:
        file = "/sdcard/crosspad/kits/K0/sequences.json" if self.per_kit else "/sdcard/crosspad/sequences.json"
        return (f"SEQ: mode={2 if self.recording else 1} state=4 scenes={self.scenes} "
                f"scene={self.active}>{self.active} bars=4/4/4/4/4/4/4/4 "
                f"ev={'/'.join(str(e) for e in self.ev)} bpm=120.0 quant=2 pos=0.250 dirty=0 "
                f"erase=0 countin={self.countin} file={file}")

    def _seq_cmd(self, arg: str) -> None:
        if arg == "ADD":
            self.scenes += 1
        elif arg.startswith("SCENE "):
            self.active = int(arg.split()[1])
        elif arg.startswith("DEL "):
            gone = int(arg.split()[1])
            del self.ev[gone]
            self.ev.append(0)
            self.scenes -= 1
            self.active = 0
        elif arg == "REC":
            self.recording = True
        elif arg in ("PLAY", "STOP"):
            self.recording = False
        elif arg.startswith("COUNTIN "):
            self.countin = int(arg.split()[1])

    def write(self, data: bytes) -> int:
        for raw in data.decode("utf-8", "replace").split("\n"):
            cmd = raw.strip()
            if not cmd:
                continue
            self.written.append(cmd)
            if cmd.startswith("SEQ"):
                self._seq_cmd(cmd[3:].strip())
                self.feed([self._seq()])
            elif cmd.startswith("PAD_PRESS"):
                if self.recording:
                    self.ev[self.active] += 1
                self.feed(["OK"])
            elif cmd.startswith("PAD_RELEASE"):
                self.feed(["OK"])
            elif cmd.startswith("KIT_LIST"):
                self.feed(["KITS: 0:K0", "KITS: 1:K1", f"KITS: end current={self.kit} count=2"])
            elif cmd.startswith("KIT_STATUS"):
                self.feed([f"KITSTATUS: current={self.kit} loading=0 pending=-1 name=K{self.kit}"])
            elif cmd.startswith("KIT_LOAD"):
                self.kit = int(cmd.split()[1])
                if self.per_kit:
                    self.ev = [0] * 8
                    self.scenes = 1
                self.feed(["OK"])
            else:
                self._rx.append((cmd + "\r\n").encode())
        return len(data)


@dataclass
class FakeContext(base.Context):
    board: SeqBoard = None  # type: ignore[assignment]

    def open_cdc(self) -> CdcLink:
        link = CdcLink(self.device.ports.cdc.path, serial_factory=lambda path, **kw: self.board)
        link.open()
        self.opened.append(link)
        return link


FAST = seq_kit_change.Params(pads=3, bar_s=0.0, load_timeout_s=1.0, settle_s=0.0)


def run(tmp_path: Path, board: SeqBoard) -> base.Report:
    device = Device(id="dev_ab12", serial="CP-1", usb_mode=UsbMode.DEFAULT, ports=Ports(cdc=CDC))
    ctx = FakeContext(device=device, workdir=tmp_path, cancelled=threading.Event(),
                      log=lambda s: None, board=board)
    return seq_kit_change.SCENARIO.run(ctx, FAST, base.Progress())


def test_a_take_that_survives_both_kit_loads_passes(tmp_path) -> None:
    board = SeqBoard()
    report = run(tmp_path, board)
    assert report.passed, report.summary
    assert "KIT_LOAD 1" in board.written and "KIT_LOAD 0" in board.written
    assert "SEQ SCENE 2" in board.written            # a scene of its own, after the player's two


def test_the_players_scenes_are_left_as_found(tmp_path) -> None:
    board = SeqBoard()
    run(tmp_path, board)
    assert board.written[-2:] == ["SEQ DEL 2", "SEQ COUNTIN 1"]
    assert board.countin == 1 and board.scenes == 2 and board.ev[0] == 5 and board.ev[2] == 0


def test_scenes_that_follow_the_kit_fail(tmp_path) -> None:
    report = run(tmp_path, SeqBoard(per_kit=True))
    assert not report.passed and report.exit_code == 1
    assert "kit's folder" in report.summary and "after kit 1" in report.summary


def test_no_free_scene_is_a_bench_problem(tmp_path) -> None:
    board = SeqBoard(scenes=8)
    report = run(tmp_path, board)
    assert report.exit_code == 2
    assert not any(c.startswith("SEQ DEL") for c in board.written)
