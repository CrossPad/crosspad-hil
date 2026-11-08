from __future__ import annotations

import pytest

from crosspad_hil.devices import Device, MidiPortInfo, Ports, UsbMode
from crosspad_hil.errors import (
    BAD_ARGS,
    BAD_SYSEX,
    DENIED_SYSEX,
    NO_DEVICE,
    NOT_SUPPORTED,
    TIMEOUT,
    HilError,
)
from crosspad_hil.midi import (
    MidiIO,
    MidiRole,
    companion_command,
    echo_rtt,
    parse_companion_chunk,
    parse_daw_hello,
    parse_query_reply,
    query_route,
    sysex_audio_route,
    sysex_bootloader_esp,
    sysex_companion,
    sysex_daw,
    sysex_daw_beat,
    sysex_daw_bye,
    sysex_daw_config,
    sysex_daw_hello,
    sysex_daw_led,
    sysex_daw_led_all,
    sysex_daw_mode,
    sysex_daw_text,
    sysex_daw_transport,
    sysex_echo,
    sysex_pad,
    sysex_query,
    sysex_usb_mode,
)
from tests.fakes import FakeMidiBus


def make_device(esp_out: int | None = 0, esp_in: int | None = 0,
                stm_out: int | None = 1) -> Device:
    return Device(
        id="dev_1234",
        serial="ABCD1234",
        usb_mode=UsbMode.DEFAULT,
        ports=Ports(
            esp_midi=MidiPortInfo(name="Crosspad MIDI 1", rtmidi_out=esp_out,
                                  rtmidi_in=esp_in, alsa_hw="hw:2,0,0", rawmidi=None),
            stm_midi=MidiPortInfo(name="CrossPad MIDI+Serial 2", rtmidi_out=stm_out,
                                  rtmidi_in=None, alsa_hw="hw:3,0,0", rawmidi=None),
        ),
    )


def make_io(bus: FakeMidiBus, role: MidiRole = MidiRole.ESP, device: Device | None = None) -> MidiIO:
    return MidiIO(device or make_device(), role,
                  out_factory=bus.out_factory, in_factory=bus.in_factory)


# ---- builders: byte-exact --------------------------------------------------

def test_sysex_usb_mode_bytes() -> None:
    assert sysex_usb_mode(UsbMode.DEFAULT) == bytes([0xF0, 0x7D, 0x1B, 0x01, 0xF7])
    assert sysex_usb_mode(UsbMode.AUDIO) == bytes([0xF0, 0x7D, 0x1B, 0x02, 0xF7])
    with pytest.raises(HilError) as e:
        sysex_usb_mode(UsbMode.BOOTLOADER)
    assert e.value.code == BAD_ARGS


def test_sysex_audio_route_and_pad_bytes() -> None:
    # hil_audio_loopback.py SYSEX_ROUTE_LOOP[0]: "F0 7D 1D 01 01 01 F7"
    assert sysex_audio_route(0x01, 1, 1) == bytes([0xF0, 0x7D, 0x1D, 0x01, 0x01, 0x01, 0xF7])
    assert sysex_pad(True, 3, 100) == bytes([0xF0, 0x7D, 0x1D, 0x07, 0x03, 0x64, 0xF7])
    assert sysex_pad(False, 3) == bytes([0xF0, 0x7D, 0x1D, 0x08, 0x03, 0xF7])
    assert sysex_query() == bytes([0xF0, 0x7D, 0x1D, 0x10, 0xF7])
    assert sysex_bootloader_esp() == bytes([0xF0, 0x7D, 0x19, 0x00, 0xF7])
    with pytest.raises(HilError) as e:
        sysex_pad(True, 16)
    assert e.value.code == BAD_ARGS
    with pytest.raises(HilError) as e:
        sysex_audio_route(0x01, 1, 200)
    assert e.value.code == BAD_ARGS


def test_sysex_echo_roundtrip_msb_first() -> None:
    # hil_midi_bench.py echo_frame(): (seq>>21)&0x7F, (seq>>14)&0x7F, (seq>>7)&0x7F, seq&0x7F
    seq = (1 << 27) | 0x123456
    frame = sysex_echo(seq)
    assert frame[:4] == bytes([0xF0, 0x7D, 0x1D, 0x09]) and frame[-1] == 0xF7
    assert len(frame) == 9
    assert all(b < 0x80 for b in frame[4:8])
    decoded = ((frame[4] & 0x7F) << 21) | ((frame[5] & 0x7F) << 14) \
        | ((frame[6] & 0x7F) << 7) | (frame[7] & 0x7F)
    assert decoded == seq
    assert sysex_echo(0) == bytes([0xF0, 0x7D, 0x1D, 0x09, 0, 0, 0, 0, 0xF7])
    with pytest.raises(HilError) as e:
        sysex_echo(1 << 28)
    assert e.value.code == BAD_ARGS


def test_parse_query_reply_layout() -> None:
    # audio_route_control.cpp send_query_reply: F0 7D 1D 10 micSrc adc0 adc1 out0 out1 v0 v1 m0 m1 F7
    frame = bytes([0xF0, 0x7D, 0x1D, 0x10, 1, 2, 1, 3, 2, 80, 90, 0, 1, 0xF7])
    assert parse_query_reply(frame) == {
        "mic_src": 1, "adc": [2, 1], "out": [3, 2], "vol": [80, 90], "mute": [0, 1],
    }
    with pytest.raises(HilError) as e:
        parse_query_reply(bytes([0xF0, 0x7D, 0x1D, 0x09, 0, 0, 0, 0, 0xF7]))
    assert e.value.code == BAD_SYSEX


# ---- MidiIO -----------------------------------------------------------------

def test_send_sysex_denylist_refuses_stm_bootloader_frame() -> None:
    bus = FakeMidiBus()
    io = make_io(bus)
    io.open()
    with pytest.raises(HilError) as e:
        io.send_sysex(bytes([0xF0, 0x7D, 0x19, 0x01, 0xF7]))
    assert e.value.code == DENIED_SYSEX
    assert bus.sent == []
    io.close()


def test_send_sysex_validates_framing() -> None:
    bus = FakeMidiBus()
    with make_io(bus) as io:
        for bad in (b"", bytes([0x7D, 0x1D, 0x10, 0xF7]), bytes([0xF0, 0x7D, 0x1D, 0x10]),
                    bytes([0xF0, 0x7D, 0x1D, 0x80, 0xF7])):
            with pytest.raises(HilError) as e:
                io.send_sysex(bad)
            assert e.value.code == BAD_SYSEX
        io.send_sysex(sysex_query())
    assert bus.sent == [[0xF0, 0x7D, 0x1D, 0x10, 0xF7]]
    assert bus.outs[0].closed and bus.ins[0].closed


def test_send_note_and_role_port_selection() -> None:
    bus = FakeMidiBus(out_ports=["Crosspad MIDI 1", "CrossPad MIDI+Serial 2"])
    with make_io(bus, MidiRole.STM) as io:
        assert bus.outs[0].opened == 1
        assert bus.ins == []  # STM role opens no IN port
        io.send_note(True, 60, 100, channel=2)
        io.send_note(False, 60)
        with pytest.raises(HilError) as e:
            io.receive(0.01)
        assert e.value.code == NOT_SUPPORTED
    assert bus.sent == [[0x92, 60, 100], [0x80, 60, 0]]
    with pytest.raises(HilError) as e:
        make_io(bus, device=make_device(esp_out=None)).open()
    assert e.value.code == NO_DEVICE
    with pytest.raises(HilError) as e:
        make_io(bus).send_note(True, 60)
    assert e.value.code == BAD_ARGS


def test_receive_returns_only_sysex_from_esp_in() -> None:
    bus = FakeMidiBus()
    with make_io(bus) as io:
        assert bus.ins[0].sysex_ignored is False
        bus.inject([0x90, 60, 100])            # note-on: dropped
        bus.inject([0xF0, 0x7D, 0x1D, 0x10, 0xF7])
        assert io.receive(0.05) == bytes([0xF0, 0x7D, 0x1D, 0x10, 0xF7])
        assert io.receive(0.01) is None


def test_query_route_reads_reply() -> None:
    def responder(msg: list[int]) -> list[list[int]]:
        if msg[:4] == [0xF0, 0x7D, 0x1D, 0x10]:
            return [[0xF0, 0x7D, 0x1D, 0x10, 0, 2, 1, 1, 2, 70, 70, 0, 0, 0xF7]]
        return []

    bus = FakeMidiBus(responder=responder)
    with make_io(bus) as io:
        bus.inject([0xF0, 0x7D, 0x1D, 0x09, 1, 2, 3, 4, 0xF7])   # stale frame, must be drained
        assert query_route(io, timeout_s=0.2) == {
            "mic_src": 0, "adc": [2, 1], "out": [1, 2], "vol": [70, 70], "mute": [0, 0],
        }
    with make_io(FakeMidiBus()) as io:
        with pytest.raises(HilError) as e:
            query_route(io, timeout_s=0.05)
        assert e.value.code == TIMEOUT


def test_echo_rtt_loss_accounting() -> None:
    def responder(msg: list[int]) -> list[list[int]]:
        # device echo (audio_route_control.cpp AUDIO_ROUTE_SUB_ECHO): mirrors the 4 bytes
        if msg[:4] == [0xF0, 0x7D, 0x1D, 0x09]:
            seq = (msg[4] << 21) | (msg[5] << 14) | (msg[6] << 7) | msg[7]
            if seq % 3 == 0:
                return []                     # drop every third
            return [[0xF0, 0x7D, 0x1D, 0x09, msg[4], msg[5], msg[6], msg[7], 0xF7]]
        return []

    ticks = iter(range(0, 10_000))

    def clock() -> float:
        return next(ticks) * 0.001            # 1 ms per call

    bus = FakeMidiBus(responder=responder)
    with make_io(bus) as io:
        r = echo_rtt(io, n=9, timeout_s=0.02, clock=clock)
    assert r["sent"] == 9
    assert r["received"] == 6
    assert r["lost"] == 3
    assert set(r["rtt_ms"]) == {"p50", "p90", "max"}
    assert r["rtt_ms"]["max"] >= r["rtt_ms"]["p90"] >= r["rtt_ms"]["p50"] > 0
    assert len(bus.sent) == 9
    with make_io(FakeMidiBus()) as io:
        r = echo_rtt(io, n=2, timeout_s=0.01)
    assert r == {"sent": 2, "received": 0, "lost": 2,
                 "rtt_ms": {"p50": None, "p90": None, "max": None}}


# ---- teardown: the sequencer client, not just the port ---------------------

def test_close_unregisters_both_handles() -> None:
    bus = FakeMidiBus()
    io = make_io(bus)
    io.open()
    io.close()
    assert [o.deleted for o in bus.outs] == [True]
    assert [i.deleted for i in bus.ins] == [True]


def test_repeated_open_close_leaves_nothing_registered() -> None:
    bus = FakeMidiBus()
    for _ in range(5):
        io = make_io(bus)
        io.open()
        io.close()
    assert len(bus.outs) == 5 and len(bus.ins) == 5
    assert all(o.deleted for o in bus.outs)
    assert all(i.deleted for i in bus.ins)


def test_close_is_idempotent() -> None:
    bus = FakeMidiBus()
    io = make_io(bus)
    io.open()
    io.close()
    io.close()
    assert len(bus.outs) == 1 and bus.outs[0].deleted


def test_stm_role_opens_and_unregisters_out_only() -> None:
    bus = FakeMidiBus()
    io = make_io(bus, MidiRole.STM, make_device(stm_out=0))
    io.open()
    io.close()
    assert [o.deleted for o in bus.outs] == [True]
    assert bus.ins == []


# ---- DAW control: byte-exact builders --------------------------------------

def test_sysex_daw_generic_builder() -> None:
    assert sysex_daw(0x01, 1, 1) == bytes([0xF0, 0x7D, 0x1E, 0x01, 0x01, 0x01, 0xF7])
    with pytest.raises(HilError) as e:
        sysex_daw(0x01, 200)
    assert e.value.code == BAD_ARGS


def test_sysex_daw_hello_bytes() -> None:
    assert sysex_daw_hello() == bytes([0xF0, 0x7D, 0x1E, 0x01, 0x00, 0x01, 0xF7])
    assert sysex_daw_hello(daw_id=1, proto_ver=1) == \
        bytes([0xF0, 0x7D, 0x1E, 0x01, 0x01, 0x01, 0xF7])


def test_sysex_daw_led_doubles_on_device_side_halves_on_the_wire() -> None:
    # the device doubles a 7-bit value back: 255 host-side -> 0x7F on the wire
    assert sysex_daw_led(3, 255, 0, 128) == \
        bytes([0xF0, 0x7D, 0x1E, 0x02, 0x03, 0x7F, 0x00, 0x40, 0xF7])
    with pytest.raises(HilError) as e:
        sysex_daw_led(16, 0, 0, 0)
    assert e.value.code == BAD_ARGS
    with pytest.raises(HilError) as e:
        sysex_daw_led(0, 256, 0, 0)
    assert e.value.code == BAD_ARGS


def test_sysex_daw_led_all_requires_sixteen() -> None:
    colors = [(i * 16, 255 - i * 16, 64) for i in range(16)]
    frame = sysex_daw_led_all(colors)
    assert frame[:4] == bytes([0xF0, 0x7D, 0x1E, 0x03])
    assert frame[-1] == 0xF7
    assert len(frame) == 4 + 16 * 3 + 1
    # pad 0: (0, 255, 64) -> (0, 0x7F, 0x20)
    assert frame[4:7] == bytes([0, 0x7F, 0x20])
    with pytest.raises(HilError) as e:
        sysex_daw_led_all(colors[:15])
    assert e.value.code == BAD_ARGS
    with pytest.raises(HilError) as e:
        sysex_daw_led_all(colors + [(0, 0, 0)])
    assert e.value.code == BAD_ARGS


def test_sysex_daw_text_truncates_and_rejects_non_printable() -> None:
    frame = sysex_daw_text(0, "HIL")
    assert frame == bytes([0xF0, 0x7D, 0x1E, 0x04, 0x00, ord("H"), ord("I"), ord("L"), 0xF7])
    long_frame = sysex_daw_text(1, "x" * 40)
    assert len(long_frame) == 4 + 1 + 24 + 1
    assert sysex_daw_text(2, "") == bytes([0xF0, 0x7D, 0x1E, 0x04, 0x02, 0xF7])
    with pytest.raises(HilError) as e:
        sysex_daw_text(3, "nope")
    assert e.value.code == BAD_ARGS
    with pytest.raises(HilError) as e:
        sysex_daw_text(0, "line\nbreak")
    assert e.value.code == BAD_ARGS


def test_sysex_daw_beat_transport_mode() -> None:
    assert sysex_daw_beat(2) == bytes([0xF0, 0x7D, 0x1E, 0x05, 0x02, 0xF7])
    assert sysex_daw_transport(True, False, True) == \
        bytes([0xF0, 0x7D, 0x1E, 0x06, 0x05, 0xF7])
    assert sysex_daw_transport(False, False, False) == \
        bytes([0xF0, 0x7D, 0x1E, 0x06, 0x00, 0xF7])
    assert sysex_daw_mode(1) == bytes([0xF0, 0x7D, 0x1E, 0x07, 0x01, 0xF7])


def test_sysex_daw_config_channel_is_1_based_on_the_api() -> None:
    assert sysex_daw_config(36, 10) == bytes([0xF0, 0x7D, 0x1E, 0x08, 36, 9, 0xF7])
    with pytest.raises(HilError) as e:
        sysex_daw_config(36, 0)
    assert e.value.code == BAD_ARGS
    with pytest.raises(HilError) as e:
        sysex_daw_config(36, 17)
    assert e.value.code == BAD_ARGS


def test_sysex_daw_bye() -> None:
    assert sysex_daw_bye() == bytes([0xF0, 0x7D, 0x1E, 0x7F, 0xF7])


def test_parse_daw_hello() -> None:
    frame = bytes([0xF0, 0x7D, 0x1E, 0x01, 1, 16, 3, 0b1111, 0xF7])
    assert parse_daw_hello(frame) == {
        "proto_ver": 1, "pads": 16, "rows": 3,
        "caps": {"rgb": True, "text": True, "encoder": True, "aftertouch": True,
                 "control": False},
    }
    frame2 = bytes([0xF0, 0x7D, 0x1E, 0x01, 1, 16, 3, 0b0001, 0xF7])
    assert parse_daw_hello(frame2)["caps"] == \
        {"rgb": True, "text": False, "encoder": False, "aftertouch": False, "control": False}
    with pytest.raises(HilError) as e:
        parse_daw_hello(bytes([0xF0, 0x7D, 0x1E, 0x02, 1, 16, 3, 0, 0xF7]))
    assert e.value.code == BAD_SYSEX
    with pytest.raises(HilError) as e:
        parse_daw_hello(bytes([0xF0, 0x7D, 0x1E, 0x01, 1, 16, 0xF7]))
    assert e.value.code == BAD_SYSEX


# ---- note/CC queue separate from the SysEx queue ---------------------------

def test_receive_note_queues_channel_voice_messages_separately() -> None:
    bus = FakeMidiBus()
    with make_io(bus) as io:
        bus.inject([0x99, 39, 100])              # note-on ch10
        bus.inject([0xF0, 0x7D, 0x1E, 0x01, 1, 16, 3, 0, 0xF7])   # sysex: own queue
        bus.inject([0x89, 39, 0])                 # note-off ch10
        assert io.receive_note(0.05) == [0x99, 39, 100]
        assert io.receive(0.05) == bytes([0xF0, 0x7D, 0x1E, 0x01, 1, 16, 3, 0, 0xF7])
        assert io.receive_note(0.05) == [0x89, 39, 0]
        assert io.receive_note(0.01) is None
        assert io.receive(0.01) is None


# -- companion control over SysEx (0x1F) -------------------------------------

def test_sysex_companion_frame_is_ascii_after_the_request_sub() -> None:
    frame = sysex_companion("PAD_PRESS 3 100")
    assert frame[:4] == bytes([0xF0, 0x7D, 0x1F, 0x01])
    assert frame[4:-1] == b"PAD_PRESS 3 100"
    assert frame[-1] == 0xF7
    with pytest.raises(HilError) as ei:
        sysex_companion("UI_STATE\n")
    assert ei.value.code == BAD_ARGS
    with pytest.raises(HilError):
        sysex_companion("X" * 191)


def test_parse_companion_chunk_more_last_and_foreign() -> None:
    assert parse_companion_chunk(bytes([0xF0, 0x7D, 0x1F, 0x02, *b"AB", 0xF7])) == ("AB", False)
    assert parse_companion_chunk(bytes([0xF0, 0x7D, 0x1F, 0x03, *b"OK\r\n", 0xF7])) == ("OK\r\n", True)
    assert parse_companion_chunk(bytes([0xF0, 0x7D, 0x1D, 0x10, 0, 0xF7])) is None
    assert parse_companion_chunk(bytes([0xF0, 0x7D, 0x1F, 0x01, *b"X", 0xF7])) is None


class _FakeCompanionIO:
    def __init__(self, replies: list[bytes]) -> None:
        self.sent: list[bytes] = []
        self._replies = list(replies)
        self.drained = 0

    def drain(self) -> int:
        self.drained += 1
        return 0

    def send_sysex(self, frame: bytes) -> None:
        self.sent.append(bytes(frame))

    def receive(self, timeout_s: float) -> bytes | None:
        return self._replies.pop(0) if self._replies else None


def test_companion_command_concatenates_chunks_and_skips_other_frames() -> None:
    io = _FakeCompanionIO([
        bytes([0xF0, 0x7D, 0x1D, 0x10, *([0] * 9), 0xF7]),          # an audio-route reply in between
        bytes([0xF0, 0x7D, 0x1F, 0x02, *b"APPS: Sampler ", 0xF7]),
        bytes([0xF0, 0x7D, 0x1F, 0x03, *b"running=Sampler\r\n", 0xF7]),
    ])
    assert companion_command(io, "APP_LIST") == "APPS: Sampler running=Sampler\r\n"
    assert io.sent == [sysex_companion("APP_LIST")]
    assert io.drained == 1


def test_companion_command_returns_err_text_and_times_out_without_last_chunk() -> None:
    io = _FakeCompanionIO([bytes([0xF0, 0x7D, 0x1F, 0x03, *b"ERR busy\r\n", 0xF7])])
    assert companion_command(io, "UI_STATE") == "ERR busy\r\n"
    io = _FakeCompanionIO([bytes([0xF0, 0x7D, 0x1F, 0x02, *b"half", 0xF7])])
    with pytest.raises(HilError) as ei:
        companion_command(io, "UI_STATE", timeout_s=0.05)
    assert ei.value.code == TIMEOUT


def _reply_frame(text: str, last: bool = True) -> bytes:
    return bytes([0xF0, 0x7D, 0x1F, 0x03 if last else 0x02, *text.encode("ascii"), 0xF7])


def test_companion_command_accept_drops_a_reply_to_somebody_elses_verb() -> None:
    io = _FakeCompanionIO([
        _reply_frame("FWUPD: state=idle running=v1.1.0-rc2"),           # the hub's request
        _reply_frame("FW_LATEST: latest=1.1.0-rc3 update=1"),
    ])
    out = companion_command(io, "FW_LATEST", accept=("FW_LATEST:",))
    assert out == "FW_LATEST: latest=1.1.0-rc3 update=1"


def test_companion_command_accept_keeps_reading_until_the_timeout() -> None:
    io = _FakeCompanionIO([_reply_frame("FWUPD: state=idle")])
    with pytest.raises(HilError) as ei:
        companion_command(io, "FW_LATEST", timeout_s=0.05, accept=("FW_LATEST:",))
    assert ei.value.code == TIMEOUT
    io = _FakeCompanionIO([_reply_frame("ERR busy")])
    assert companion_command(io, "FW_CHECK", accept=("OK", "ERR")) == "ERR busy"
