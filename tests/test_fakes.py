"""The fakes are the test suite's hardware; their behaviour must match the real device."""
from __future__ import annotations

from tests.fakes import FakeMidiBus, FakeSerial


def test_scripted_reply_is_returned_with_crlf() -> None:
    ser = FakeSerial([("KIT_STATUS", "KITSTATUS: current=3 loading=0 pending=-1 name=DRUMS")],
                     timeout=0.01)
    ser.open()
    ser.write(b"KIT_STATUS\n")
    assert ser.readline() == b"KITSTATUS: current=3 loading=0 pending=-1 name=DRUMS\r\n"
    assert ser.readline() == b""


def test_multi_line_reply_comes_out_in_order() -> None:
    ser = FakeSerial([("ENC_GROUP", ["ENCGROUP: count=2", "  [0] 0x3fc9a000 Sampler",
                                     "  [1] 0x3fc9a100 Settings"])], timeout=0.01)
    ser.open()
    ser.write(b"ENC_GROUP\n")
    assert ser.readline() == b"ENCGROUP: count=2\r\n"
    assert ser.readline() == b"  [0] 0x3fc9a000 Sampler\r\n"
    assert ser.readline() == b"  [1] 0x3fc9a100 Settings\r\n"


def test_unknown_command_is_echoed_back_like_the_device() -> None:
    # hil_control_handle_cdc() returns false for unknown verbs and main.cpp echoes the line
    ser = FakeSerial(timeout=0.01)
    ser.open()
    ser.write(b"FROBNICATE 1\n")
    assert ser.readline() == b"FROBNICATE 1\r\n"


def test_written_records_decoded_commands_without_newline() -> None:
    ser = FakeSerial([("PAD_PRESS 3 100", "OK")], timeout=0.01)
    ser.open()
    ser.write(b"PAD_PRESS 3 100\n")
    ser.write(b"PAD_RELEASE 3\r\n")
    assert ser.written == ["PAD_PRESS 3 100", "PAD_RELEASE 3"]


def test_feed_pushes_unsolicited_lines() -> None:
    ser = FakeSerial(timeout=0.01)
    ser.open()
    ser.feed(["ESP-ROM:esp32s3-20210327", "I (123) main: All systems operational"])
    assert ser.readline() == b"ESP-ROM:esp32s3-20210327\r\n"
    assert ser.readline() == b"I (123) main: All systems operational\r\n"


def test_control_history_records_every_dtr_rts_assignment() -> None:
    ser = FakeSerial()
    ser.dtr = False
    ser.rts = True
    ser.rts = False
    assert ser.control_history == [("dtr", False), ("rts", True), ("rts", False)]
    assert ser.dtr is False and ser.rts is False


def test_pyserial_shaped_open_close() -> None:
    ser = FakeSerial()
    assert ser.is_open is False
    ser.port = "/dev/ttyACM9"
    ser.baudrate = 115200
    ser.timeout = 0.2
    ser.open()
    assert ser.is_open is True
    ser.close()
    assert ser.is_open is False


def test_read_n_bytes_and_in_waiting() -> None:
    ser = FakeSerial([("MEM", "MEM: int_free=1")], timeout=0.01)
    ser.open()
    ser.write(b"MEM\n")
    assert ser.in_waiting == len(b"MEM: int_free=1\r\n")
    assert ser.read(4) == b"MEM:"
    assert ser.readline() == b" int_free=1\r\n"


def test_fake_midi_bus_delivers_out_messages_to_the_in_side() -> None:
    bus = FakeMidiBus(responder=lambda msg: [[0xF0, 0x7D, 0x1D, 0x10, 0xF7]] if msg[2] == 0x1D else [])
    out, inp = bus.out_factory(), bus.in_factory()
    out.open_port(0)
    inp.open_port(0)
    got: list[list[int]] = []
    inp.set_callback(lambda ev, _d: got.append(ev[0]))
    out.send_message([0xF0, 0x7D, 0x1D, 0x10, 0xF7])
    assert bus.sent == [[0xF0, 0x7D, 0x1D, 0x10, 0xF7]]
    assert got == [[0xF0, 0x7D, 0x1D, 0x10, 0xF7]]
