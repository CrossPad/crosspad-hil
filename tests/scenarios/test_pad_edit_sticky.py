"""Scenario `pad_edit_sticky` against a fake Sampler pad editor."""
from __future__ import annotations

import threading
from dataclasses import dataclass
from pathlib import Path

from crosspad_hil.cdc import CdcLink
from crosspad_hil.devices import Device, Ports, SerialPortInfo, UsbMode
from crosspad_hil.scenarios import base, pad_edit_sticky
from tests.fakes import AudioDesk, FakeSerial

CDC = SerialPortInfo(path="/dev/ttyACM0", vid=0x303A, pid=0x3456, serial="CP-1",
                     product="CrossPad", location="1-2.1")


class EditorBoard(FakeSerial):
    """Sampler → Advanced → Edit, the pad editor's encoder group and its edit mode.

    * ``drop_on_hit`` — a pad hit leaves edit mode (the firmware before 06.10.2026).
    * ``legacy``      — ENCFOCUS without edit= (older firmware still).
    """

    def __init__(self, *, drop_on_hit: bool = False, legacy: bool = False) -> None:
        super().__init__(timeout=0.01)
        self.drop_on_hit = drop_on_hit
        self.legacy = legacy
        self.screen = "launcher"
        self.focus = 0
        self.editing = False
        self.pad = 0
        self.choke = [0] * 16
        self.choke[10] = 7               # at the top: the turn has to go down
        self.kit = 2

    def _group(self) -> list[str]:
        if self.screen == "sampler":
            return ["Play", "Advanced", "Kits"]
        if self.screen == "advanced":
            return ["Edit", "Mixer", "Close"]
        if self.screen == "editor":
            c = self.choke[self.pad]
            return ["<", f"PAD {self.pad + 1}", "REC MODE", "VOL 100", f"CHOKE {c or '-'}", "START", "END"]
        return ["Sampler", "Settings"]

    def write(self, data: bytes) -> int:
        cmd = data.decode("utf-8", "replace").strip()
        self.written.append(cmd)
        group = self._group()
        label = group[self.focus % len(group)]
        if cmd == "KIT_STATUS":
            self.feed([f"KITSTATUS: current={self.kit} loading=0 pending=-1 name=K"])
        elif cmd.startswith("KIT_LOAD"):
            self.feed(["OK"])
        elif cmd.startswith("APP_START"):
            self.screen, self.focus = "sampler", 0
            self.feed(["OK"])
        elif cmd == "APP_LIST":
            running = "-" if self.screen == "launcher" else "Sampler"
            self.feed([f"APPS: Sampler,Settings running={running}"])
        elif cmd == "APP_STOP":
            self.screen, self.focus, self.editing = "launcher", 0, False
            self.feed(["OK"])
        elif cmd == "ENC_GROUP":
            self.feed([f"ENCGROUP: count={len(group)}"]
                      + [f"  [{i}] 0x{1000 + i:x} {lb}" for i, lb in enumerate(group)])
        elif cmd == "ENC_FOCUS":
            edit = "" if self.legacy else f"edit={int(self.editing)} "
            self.feed([f"ENCFOCUS: obj=0x{1000 + self.focus:x} idx={self.focus} {edit}text={label}"])
        elif cmd.startswith("ENC_ROTATE"):
            d = int(cmd.split()[1])
            if self.editing and label.startswith("CHOKE"):
                self.choke[self.pad] = max(0, min(7, self.choke[self.pad] + d))
            elif not self.editing:
                self.focus = (self.focus + d) % len(group)
            self.feed(["OK"])
        elif cmd.startswith("ENC_PRESS"):
            if self.screen == "sampler" and label == "Advanced":
                self.screen, self.focus = "advanced", 0
            elif self.screen == "advanced" and label == "Edit":
                self.screen, self.focus = "editor", 5
            elif self.screen == "editor" and label.startswith("CHOKE"):
                self.editing = not self.editing
            self.feed(["OK"])
        elif cmd.startswith("PAD_PRESS"):
            if self.screen == "editor":
                self.pad = int(cmd.split()[1])
                if self.drop_on_hit:
                    self.editing = False
            self.feed(["OK"])
        elif cmd.startswith("PAD_RELEASE"):
            self.feed(["OK"])
        else:
            self.feed([cmd])
        return len(data)


@dataclass
class FakeContext(base.Context):
    board: EditorBoard = None  # type: ignore[assignment]

    def open_cdc(self) -> CdcLink:
        link = CdcLink(self.device.ports.cdc.path, serial_factory=lambda path, **kw: self.board)
        link.open()
        self.opened.append(link)
        return link


FAST = pad_edit_sticky.Params(settle_s=0.0, load_timeout_s=1.0)


def run(tmp_path: Path, board: EditorBoard, params: pad_edit_sticky.Params = FAST) -> base.Report:
    device = Device(id="dev_ab12", serial="CP-1", usb_mode=UsbMode.DEFAULT, ports=Ports(cdc=CDC))
    ctx = FakeContext(device=device, workdir=tmp_path, cancelled=threading.Event(),
                      log=lambda s: None, board=board)
    return pad_edit_sticky.SCENARIO.run(ctx, params, base.Progress())


def test_edit_mode_kept_across_hits_passes(tmp_path) -> None:
    board = EditorBoard()
    report = run(tmp_path, board)
    assert report.passed, report.summary
    assert [h["pad"] for h in report.data["hops"]] == [6, 11, 16]
    assert report.data["hops"][1]["turned"] == 6          # CHOKE 7 turns down, not into the clamp


def test_edit_mode_dropped_by_a_hit_fails(tmp_path) -> None:
    report = run(tmp_path, EditorBoard(drop_on_hit=True))
    assert report.exit_code == 1
    assert "edit mode dropped by the hit" in report.summary
    assert len(report.data["failures"]) == 3              # every hop, not just the first


def test_turns_are_put_back_and_the_kit_reloaded(tmp_path) -> None:
    board = EditorBoard()
    run(tmp_path, board)
    assert board.choke[10] == 7 and board.choke[5] == 0
    assert board.written.index("KIT_LOAD 2") > board.written.index("APP_STOP")
    assert not board.editing


def test_firmware_without_edit_field_is_a_bench_problem(tmp_path) -> None:
    report = run(tmp_path, EditorBoard(legacy=True))
    assert report.exit_code == 2
    assert "edit=" in report.summary


def test_one_pad_is_not_a_test(tmp_path) -> None:
    report = run(tmp_path, EditorBoard(), pad_edit_sticky.Params(pads="3", settle_s=0.0))
    assert report.exit_code == 2


class DeskEditorBoard(EditorBoard):
    """The editor fake plus the mixer/codec verbs; an encoder press moves the desk."""

    def __init__(self, desk: AudioDesk) -> None:
        super().__init__()
        self.desk = desk

    def write(self, data: bytes) -> int:
        cmd = data.decode("utf-8", "replace").strip()
        lines = self.desk.reply(cmd)
        if lines is not None:
            self.written.append(cmd)
            self.feed(lines)
            return len(data)
        if cmd.startswith("ENC_PRESS"):
            self.desk.vol = [28, 50]
            self.desk.mute = [1, 0]
        return super().write(data)


def test_audio_state_an_encoder_press_moved_is_put_back(tmp_path) -> None:
    desk = AudioDesk()
    home = desk.state()
    board = DeskEditorBoard(desk)
    logged: list[str] = []
    device = Device(id="dev_ab12", serial="CP-1", usb_mode=UsbMode.DEFAULT, ports=Ports(cdc=CDC))
    ctx = FakeContext(device=device, workdir=tmp_path, cancelled=threading.Event(),
                      log=logged.append, board=board)
    report = pad_edit_sticky.SCENARIO.run(ctx, FAST, base.Progress())
    assert report.passed, report.summary
    assert desk.state() == home
    assert report.data["audio_state_restored"] == [
        "codec volume codec 0: 28 -> 64", "codec volume codec 1: 50 -> 64",
        "codec mute codec 0: 1 -> 0"]
    assert any("audio_state: restored codec mute codec 0: 1 -> 0" in line for line in logged)
