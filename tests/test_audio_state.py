"""audio_state: read the mixer/codec state, put back what a UI walk moved."""
from __future__ import annotations

import uuid
from typing import Any

from crosspad_hil import audio_state
from crosspad_hil.cdc import CdcLink
from tests.fakes import AudioDesk, FakeSerial


class DeskSerial(FakeSerial):
    """A CDC endpoint that is only an AudioDesk; everything else is echoed."""

    def __init__(self, desk: AudioDesk) -> None:
        super().__init__(timeout=0.01)
        self.desk = desk

    def write(self, data: bytes) -> int:
        cmd = data.decode("utf-8", "replace").strip()
        self.written.append(cmd)
        lines = self.desk.reply(cmd)
        self.feed(lines if lines is not None else [cmd])
        return len(data)


def link_for(desk: AudioDesk) -> CdcLink:
    ser = DeskSerial(desk)
    link = CdcLink(f"fake-audio-{uuid.uuid4().hex[:8]}", serial_factory=lambda path, **kw: ser)
    link.open()
    return link


def logger() -> tuple[list[str], Any]:
    lines: list[str] = []
    return lines, lines.append


def test_parse_audio_status() -> None:
    st = audio_state.parse_audio_status(
        "AUDIO: mic_src=1 adc=[2,1] out=[3,3] vol=[28,50] mute=[0,1]")
    assert st == {"mic_src": 1, "adc": [2, 1], "out": [3, 3], "vol": [28, 50], "mute": [0, 1]}
    # an older firmware printed scalars
    old = audio_state.parse_audio_status("AUDIO: mic_src=1 adc=[1,1] out=0 vol=80 mute=0")
    assert old["adc"] == [1, 1] and old["vol"] == [80] and old["mute"] == [0]


def test_snapshot_reads_the_desk_the_codecs_and_the_pair() -> None:
    snap = audio_state.snapshot(link_for(AudioDesk()))
    assert snap["in_pair"] == 0
    assert snap["audio"] == {"mic_src": 1, "adc": [2, 1], "out": [3, 3], "vol": [64, 64],
                             "mute": [0, 0]}
    sampler = snap["mixer"]["channels"][0]
    assert sampler == {"ch": 0, "name": "Sampler", "vol": 0.8, "mute": 0, "route": [1, 1, 1]}
    assert len(snap["mixer"]["outputs"]) == 3


def test_a_build_without_the_mixer_leaves_that_part_none() -> None:
    snap = audio_state.snapshot(link_for(AudioDesk(no_mixer=True)))
    assert snap["mixer"] is None and snap["audio"]["adc"] == [2, 1]


def test_a_board_that_answers_nothing_gives_an_empty_snapshot() -> None:
    ser = FakeSerial(timeout=0.01)          # echoes every verb: "no handler matched"
    link = CdcLink("fake-audio-none", serial_factory=lambda path, **kw: ser)
    link.open()
    assert audio_state.snapshot(link) == {"mixer": None, "audio": None, "in_pair": None}


def test_nothing_moved_sends_nothing() -> None:
    desk = AudioDesk()
    link = link_for(desk)
    snap = audio_state.snapshot(link)
    lines, log = logger()
    res = audio_state.restore(link, snap, log)
    assert res == {"restored": [], "unrestorable": [], "failed": [], "unread": False}
    assert desk.commands == [] and lines == []


def test_restore_puts_back_volume_adc_route_mute_and_pair_and_logs_each() -> None:
    desk = AudioDesk()
    link = link_for(desk)
    snap = audio_state.snapshot(link)
    home = desk.state()
    # what a blind walk did on 2026-10-09: the overlay's volumes, swapped ADCs, a cut send
    desk.vol = [28, 50]
    desk.adc = [1, 2]
    desk.mute = [0, 1]
    desk.in_pair = 1
    desk.channels[0]["route"] = [0, 1, 1]
    lines, log = logger()
    res = audio_state.restore(link, snap, log)
    assert desk.state() == home
    assert res["failed"] == [] and res["unrestorable"] == []
    assert len(res["restored"]) == 7
    assert "codec volume codec 0: 28 -> 64" in res["restored"]
    assert "mixer ch 0 Sampler send to out 0: 0 -> 1" in res["restored"]
    for r in res["restored"]:
        assert f"audio_state: restored {r}" in lines
    # the pair first (it moves the ADCs under it), then the inputs
    assert desk.commands[0] == "IN_PAIR 0"
    assert desk.commands.index("AUDIO_ADC 0 LINE2") < desk.commands.index("AUDIO_VOL 0 64")


def test_a_muted_mixer_channel_and_its_level_are_restored() -> None:
    desk = AudioDesk()
    link = link_for(desk)
    snap = audio_state.snapshot(link)
    desk.channels[0]["mute"] = 1
    desk.channels[0]["vol"] = 0.1
    desk.outputs[2]["mute"] = 1
    desk.outputs[1]["vol"] = 0.5
    lines, log = logger()
    res = audio_state.restore(link, snap, log)
    assert res["failed"] == [] and res["unrestorable"] == []
    assert res["restored"] == ["mixer ch 0 Sampler mute: 1 -> 0",
                               "mixer ch 0 Sampler volume: 0.1 -> 0.8",
                               "mixer out 2 mute: 1 -> 0",
                               "mixer out 1 volume: 0.5 -> 1.0"]
    assert desk.commands == ["MIX_MUTE 0 0", "MIX_VOL 0 80", "MIX_OUT_MUTE 2 0", "MIX_OUT_VOL 1 100"]
    assert desk.channels[0]["mute"] == 0 and desk.channels[0]["vol"] == 0.8
    assert desk.outputs[2]["mute"] == 0 and desk.outputs[1]["vol"] == 1.0
    assert "audio_state: restored mixer ch 0 Sampler mute: 1 -> 0" in lines


def test_a_firmware_without_the_mixer_verbs_is_reported_not_restored() -> None:
    desk = AudioDesk(no_mix_verbs=True)
    link = link_for(desk)
    snap = audio_state.snapshot(link)
    desk.channels[0]["mute"] = 1
    desk.channels[0]["vol"] = 0.1
    desk.vol = [28, 64]
    lines, log = logger()
    res = audio_state.restore(link, snap, log)
    assert res["restored"] == ["codec volume codec 0: 28 -> 64"]
    assert res["unrestorable"] == ["mixer ch 0 Sampler mute: 1 -> 0",
                                   "mixer ch 0 Sampler volume: 0.1 -> 0.8"]
    assert res["failed"] == []
    assert desk.channels[0]["mute"] == 1
    # asked once (the echo), then not again
    assert sum("predates" in line for line in lines) == 1


def test_a_level_above_unity_is_reported_not_guessed() -> None:
    desk = AudioDesk()
    desk.channels[0]["vol"] = 1.5
    link = link_for(desk)
    snap = audio_state.snapshot(link)
    desk.channels[0]["vol"] = 0.5
    res = audio_state.restore(link, snap, None)
    assert res["unrestorable"] == ["mixer ch 0 Sampler volume: 0.5 -> 1.5"]
    assert desk.commands == []


def test_a_verb_the_firmware_refuses_is_a_failure_not_an_exception() -> None:
    desk = AudioDesk(refuse=("AUDIO_ADC",))
    link = link_for(desk)
    snap = audio_state.snapshot(link)
    desk.adc = [1, 2]
    desk.vol = [10, 64]
    lines, log = logger()
    res = audio_state.restore(link, snap, log)
    assert res["restored"] == ["codec volume codec 0: 10 -> 64"]
    assert len(res["failed"]) == 2 and all("ADC input" in f for f in res["failed"])
    assert any("could not be restored" in line for line in lines)
    assert desk.vol == [64, 64]


def test_an_ok_that_did_not_stick_is_a_failure() -> None:
    class Sticky(AudioDesk):
        def reply(self, cmd: str) -> list[str] | None:
            out = super().reply(cmd)
            if cmd.startswith("AUDIO_VOL 0"):
                self.vol[0] = 28                          # the firmware overrode it
            return out

    desk = Sticky()
    link = link_for(desk)
    snap = audio_state.snapshot(link)
    desk.vol = [28, 64]
    lines, log = logger()
    res = audio_state.restore(link, snap, log)
    assert res["restored"] == [] and res["failed"] == ["codec volume codec 0: 28 -> 64"]
    assert any("did not stick" in line for line in lines)


def test_restore_with_no_snapshot_or_a_dead_board_is_a_noop() -> None:
    desk = AudioDesk()
    link = link_for(desk)
    assert audio_state.restore(link, None)["unread"] is True
    assert audio_state.put_back(None, {"mixer": None}) is None
    assert audio_state.put_back(link, None) is None
    assert desk.commands == []


def test_put_back_records_into_data() -> None:
    desk = AudioDesk()
    link = link_for(desk)
    snap = audio_state.snapshot(link)
    desk.vol = [28, 64]
    desk.channels[0]["mute"] = 1
    data: dict[str, Any] = {}
    audio_state.put_back(link, snap, None, data)
    assert data["audio_state_restored"] == ["codec volume codec 0: 28 -> 64",
                                            "mixer ch 0 Sampler mute: 1 -> 0"]
    assert data["audio_state_unrestorable"] == []
    assert data["audio_state_restore_failed"] == []


def test_a_channel_renamed_between_snapshots_is_not_compared() -> None:
    desk = AudioDesk()
    link = link_for(desk)
    snap = audio_state.snapshot(link)
    desk.channels[0]["name"] = "Other"
    desk.channels[0]["route"] = [0, 0, 0]
    assert audio_state.diff(snap, audio_state.snapshot(link)) == []


# --- ensure_audible -------------------------------------------------------------

def test_a_healthy_board_needs_nothing() -> None:
    desk = AudioDesk()
    res = audio_state.ensure_audible(link_for(desk))
    assert res == {"fixed": [], "unfixable": [], "unread": []} and desk.commands == []


def test_no_send_and_a_low_codec_volume_are_fixed_with_the_verbs() -> None:
    desk = AudioDesk()
    desk.channels[0]["route"] = [0, 0, 0]
    desk.vol = [28, 50]
    lines, log = logger()
    res = audio_state.ensure_audible(link_for(desk), log)
    assert res["unfixable"] == [] and len(res["fixed"]) == 4
    assert desk.channels[0]["route"] == [1, 1, 1] and desk.vol == [64, 50]
    assert "MIX_ROUTE 0 2 1" in desk.commands and "AUDIO_VOL 0 64" in desk.commands
    assert "AUDIO_VOL 1 64" not in desk.commands          # 50 is fine
    assert any("codec 0 output volume 28" in line for line in lines)


def test_volume_at_the_threshold_is_left_alone() -> None:
    desk = AudioDesk()
    desk.vol = [40, 40]
    assert audio_state.ensure_audible(link_for(desk))["fixed"] == []


def test_a_muted_sampler_is_unmuted() -> None:
    desk = AudioDesk()
    desk.channels[0]["mute"] = 1
    res = audio_state.ensure_audible(link_for(desk))
    assert res["unfixable"] == [] and len(res["fixed"]) == 1
    assert "'Sampler' was muted" in res["fixed"][0]
    assert desk.commands == ["MIX_MUTE 0 0"] and desk.channels[0]["mute"] == 0


def test_a_muted_sampler_on_a_firmware_without_mix_mute_is_unfixable() -> None:
    desk = AudioDesk(no_mix_verbs=True)
    desk.channels[0]["mute"] = 1
    res = audio_state.ensure_audible(link_for(desk))
    assert res["fixed"] == [] and len(res["unfixable"]) == 1
    assert "muted" in res["unfixable"][0] and "MIX_MUTE 0 0" in res["unfixable"][0]
    assert desk.channels[0]["mute"] == 1


def test_a_fix_the_board_refuses_is_unfixable() -> None:
    desk = AudioDesk(refuse=("AUDIO_VOL",))
    desk.vol = [28, 64]
    res = audio_state.ensure_audible(link_for(desk))
    assert res["fixed"] == [] and "AUDIO_VOL 0 64" in res["unfixable"][0]


def test_a_fix_that_does_not_stick_is_unfixable() -> None:
    class Sticky(AudioDesk):
        def reply(self, cmd: str) -> list[str] | None:
            out = super().reply(cmd)
            if cmd.startswith("AUDIO_VOL 0"):
                self.vol[0] = 28
            return out

    desk = Sticky()
    desk.vol = [28, 64]
    res = audio_state.ensure_audible(link_for(desk))
    assert res["fixed"] == [] and "still wrong" in res["unfixable"][0]


def test_unreadable_state_is_not_a_misconfiguration() -> None:
    res = audio_state.ensure_audible(link_for(AudioDesk(no_mixer=True)))
    assert res["unfixable"] == [] and res["unread"] == ["mixer (MIX_LVL)"]
    ser = FakeSerial(timeout=0.01)
    link = CdcLink("fake-audio-none2", serial_factory=lambda path, **kw: ser)
    link.open()
    res = audio_state.ensure_audible(link)
    assert res["unfixable"] == [] and len(res["unread"]) == 2


def test_a_desk_with_no_sampler_channel_says_nothing() -> None:
    desk = AudioDesk()
    desk.channels[0]["name"] = "Pads"
    desk.channels[0]["mute"] = 1
    assert audio_state.ensure_audible(link_for(desk))["unfixable"] == []


def test_a_board_that_answers_nothing_is_asked_once() -> None:
    class Dead(FakeSerial):
        def write(self, data: bytes) -> int:
            self.written.append(data.decode().strip())
            return len(data)                              # no echo, no reply: no board

    ser = Dead(timeout=0.01)
    link = CdcLink("fake-audio-dead", serial_factory=lambda path, **kw: ser)
    link.open()
    assert audio_state.snapshot(link) == {"mixer": None, "audio": None, "in_pair": None}
    assert ser.written == ["MIX_LVL"]
