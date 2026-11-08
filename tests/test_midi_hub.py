"""midi_hub: the fixed ALSA port pair that bridges a DAW to whichever CrossPad is there."""
from __future__ import annotations

import json
import sys
import threading
import time
from pathlib import Path

import pytest

from crosspad_hil import hub_ipc
from crosspad_hil import midi_hub as mh
from crosspad_hil.errors import NO_DEVICE, TIMEOUT, HilError
from crosspad_hil.locks import PortLock
from crosspad_hil.midi import sysex_companion

ESP = "Crosspad:Crosspad MIDI 1 32:0"
ESP_AUDIO = "Crosspad Audio:Crosspad Audio MIDI 1 32:0"
STM = "CrossPad MIDI+Serial:CrossPad MIDI+Serial MIDI 1 20:0"
HUB = "RtMidi:CrossPad Hub 129:0"
VIRMIDI = "Virtual Raw MIDI 5-0:VirMIDI 5-0 36:0"


class FakePort:
    def __init__(self, name: str, log: list) -> None:
        self.name = name
        self.log = log
        self.cb = None
        self.closed = False

    def set_callback(self, cb) -> None:
        self.cb = cb

    def send_message(self, message) -> None:
        if self.closed:
            raise RuntimeError("port closed")
        self.log.append((self.name, list(message)))

    def close_port(self) -> None:
        self.closed = True

    def feed(self, message) -> None:
        assert self.cb is not None
        self.cb((list(message), 0.0), None)


class FakePorts:
    def __init__(self) -> None:
        self.inputs: list[str] = [STM]
        self.outputs: list[str] = [STM]
        self.sent: list = []
        self.opened: dict[str, FakePort] = {}

    def list_inputs(self):
        return list(self.inputs)

    def list_outputs(self):
        return list(self.outputs)

    def _open(self, key: str) -> FakePort:
        p = FakePort(key, self.sent)
        self.opened[key] = p
        return p

    def open_virtual_in(self, name):
        return self._open("hub-in")

    def open_virtual_out(self, name):
        return self._open("hub-out")

    def open_in(self, name):
        return self._open("raw-in" if "VirMIDI" in name else "board-in")

    def open_out(self, name):
        return self._open("raw-out" if "VirMIDI" in name else "board-out")


def test_esp_port_matching_skips_stm_bridge_and_hub() -> None:
    assert mh.is_esp_port(ESP) and mh.is_esp_port(ESP_AUDIO)
    assert not mh.is_esp_port(STM)
    assert not mh.is_esp_port(HUB)


def test_hub_attaches_when_board_appears_and_detaches_when_it_goes() -> None:
    ports = FakePorts()
    lines: list[str] = []
    hub = mh.Hub(ports, lines.append)
    hub.open()
    hub.poll()
    assert hub.board_name is None
    ports.inputs.append(ESP)
    ports.outputs.append(ESP)
    hub.poll()
    assert hub.board_name == ESP
    ports.inputs.remove(ESP)
    ports.outputs.remove(ESP)
    hub.poll()
    assert hub.board_name is None
    assert ports.opened["board-in"].closed and ports.opened["board-out"].closed
    hub.close()
    assert any("attached" in line for line in lines) and any("detached" in line for line in lines)


def test_board_traffic_goes_to_the_daw_only() -> None:
    ports = FakePorts()
    ports.inputs.append(ESP)
    ports.outputs.append(ESP)
    hub = mh.Hub(ports, lambda _: None)
    hub.open()
    hub.poll()
    ports.opened["board-in"].feed([0x99, 36, 100])
    assert ports.sent == [("hub-out", [0x99, 36, 100])]
    assert hub.forwarded_to_daw == 1


def test_daw_traffic_goes_to_the_board_and_is_mirrored() -> None:
    ports = FakePorts()
    ports.inputs.append(ESP)
    ports.outputs.append(ESP)
    hub = mh.Hub(ports, lambda _: None)
    hub.open()
    hub.poll()
    frame = [0xF0, 0x7D, 0x1E, 0x04, 0x00, 0x48, 0x49, 0xF7]
    ports.opened["hub-in"].feed(frame)
    assert ports.sent == [("hub-out", frame), ("board-out", frame)]
    assert hub.forwarded_to_board == 1


def test_daw_traffic_is_still_mirrored_without_a_board() -> None:
    ports = FakePorts()
    hub = mh.Hub(ports, lambda _: None)
    hub.open()
    hub.poll()
    ports.opened["hub-in"].feed([0xB0, 1, 5])
    assert ports.sent == [("hub-out", [0xB0, 1, 5])]


def test_a_failed_board_write_detaches_until_the_next_poll() -> None:
    ports = FakePorts()
    ports.inputs.append(ESP)
    ports.outputs.append(ESP)
    hub = mh.Hub(ports, lambda _: None)
    hub.open()
    hub.poll()
    ports.opened["board-out"].closed = True      # the USB device went away
    ports.opened["hub-in"].feed([0x90, 60, 1])
    assert hub.board_name is None
    hub.poll()                                    # still listed → re-attached
    assert hub.board_name == ESP


# The hub reports host load from /proc/stat (host_report.py: "Linux only, like the hub").
@pytest.mark.skipif(sys.platform == "win32", reason="the MIDI hub is Linux-only (/proc/stat)")
def test_run_loop_is_bounded_for_tests() -> None:
    ports = FakePorts()
    mh.run(lambda _: None, ports=ports, sleep=lambda s: None, rounds=2)
    assert ports.opened["hub-in"].closed and ports.opened["hub-out"].closed


@pytest.mark.skipif(sys.platform == "win32", reason="systemd user units are Linux-only")
def test_install_writes_unit_and_enables(tmp_path: Path) -> None:
    calls: list[list[str]] = []
    r = mh.install("/opt/hil/bin/crosspad-hil", home=tmp_path, systemctl=calls.append)
    unit = Path(r["unit"])
    assert "ExecStart=/opt/hil/bin/crosspad-hil midi hub run" in unit.read_text()
    assert calls == [["daemon-reload"], ["enable", "--now", mh.UNIT_NAME]]
    calls.clear()
    mh.uninstall(home=tmp_path, systemctl=calls.append)
    assert not unit.exists() and calls[0] == ["disable", "--now", mh.UNIT_NAME]


def _reporting_hub(ports: FakePorts, latest: str | None = "1.0.2") -> mh.Hub:
    ports.inputs.append(ESP)
    ports.outputs.append(ESP)
    hub = mh.Hub(ports, lambda _: None, host_info=lambda: "HOST_INFO cpu=5 ram=1/2 time=3 tz=0",
                 latest=lambda: latest)
    hub.open()
    hub.poll()
    return hub


def _notify(verb: str) -> list[int]:
    return [0xF0, 0x7D, 0x1F, 0x04, *verb.encode("ascii"), 0xF7]


def test_report_goes_to_the_board_only_as_notify_frames() -> None:
    ports = FakePorts()
    hub = _reporting_hub(ports)
    hub.report(100.0)
    hub.report(101.0)
    assert ports.sent == [("board-out", _notify("HOST_INFO cpu=5 ram=1/2 time=3 tz=0")),
                          ("board-out", _notify("FW_LATEST 1.0.2"))]
    assert hub.reports_sent == 1


def test_report_sends_one_frame_per_call_and_keeps_its_intervals() -> None:
    ports = FakePorts()
    hub = _reporting_hub(ports)
    for t in (100.0, 101.0, 101.5, 102.0, 103.0, 104.0):
        hub.report(t)
    verbs = [bytes(m[4:-1]).decode().split(" ")[0] for _, m in ports.sent]
    assert verbs == ["HOST_INFO", "FW_LATEST", "HOST_INFO", "HOST_INFO"]


def test_a_raising_latest_callback_skips_the_tick_but_not_the_loop() -> None:
    ports = FakePorts()
    ports.inputs.append(ESP)
    ports.outputs.append(ESP)
    logged: list[str] = []

    def boom() -> str:
        raise RuntimeError("catalog exploded")
    hub = mh.Hub(ports, logged.append, host_info=lambda: "HOST_INFO cpu=5 ram=1/2 time=3 tz=0",
                 latest=boom)
    hub.open()
    hub.poll()
    hub.report(100.0)
    hub.report(101.0)                      # FW_LATEST tick: raises inside, must not escape
    hub.report(102.0)
    assert any("FW_LATEST skipped" in m for m in logged)
    assert {bytes(m[4:-1]).decode().split(" ")[0] for _, m in ports.sent} == {"HOST_INFO"}
    assert len(ports.sent) == 2            # the loop went on: the next HOST_INFO still went out


def test_a_version_the_frame_builder_refuses_skips_the_tick() -> None:
    ports = FakePorts()
    hub = _reporting_hub(ports, latest="1.0.0\nX")
    hub.report(100.0)
    hub.report(101.0)
    assert len(ports.sent) == 1


def test_report_without_a_version_sends_only_host_info() -> None:
    ports = FakePorts()
    hub = _reporting_hub(ports, latest=None)
    hub.report(100.0)
    hub.report(101.0)
    assert [m for _, m in ports.sent] == [_notify("HOST_INFO cpu=5 ram=1/2 time=3 tz=0")]


def test_report_is_silent_without_a_board_and_restarts_on_attach() -> None:
    ports = FakePorts()
    hub = _reporting_hub(ports)
    hub.report(100.0)
    ports.inputs.remove(ESP)
    ports.outputs.remove(ESP)
    hub.poll()
    ports.sent.clear()
    hub.report(102.0)
    assert ports.sent == []
    ports.inputs.append(ESP)
    ports.outputs.append(ESP)
    hub.poll()
    hub.report(102.5)                              # due at once, not 2 s after the last one
    assert [m for _, m in ports.sent] == [_notify("HOST_INFO cpu=5 ram=1/2 time=3 tz=0")]


def test_hub_without_host_info_only_bridges() -> None:
    ports = FakePorts()
    ports.inputs.append(ESP)
    ports.outputs.append(ESP)
    hub = mh.Hub(ports, lambda _: None)
    hub.open()
    hub.poll()
    hub.report(100.0)
    assert ports.sent == []


def test_a_launch_frame_goes_to_the_launcher_never_to_the_daw() -> None:
    ports = FakePorts()
    ports.inputs.append(ESP)
    ports.outputs.append(ESP)
    asked: list[int] = []
    hub = mh.Hub(ports, lambda _: None, launch=lambda d: asked.append(d) or "started")
    hub.open()
    hub.poll()
    ports.opened["board-in"].feed([0xF0, 0x7D, 0x1E, 0x0C, 0x01, 0xF7])
    ports.opened["board-in"].feed([0xF0, 0x7D, 0x1E, 0x01, 0x01, 0x10, 0x03, 0x1F, 0xF7])
    assert asked == [1]
    assert [m for _, m in ports.sent] == [[0xF0, 0x7D, 0x1E, 0x01, 0x01, 0x10, 0x03, 0x1F, 0xF7]]


def _hub_with_virmidi(ports: FakePorts) -> mh.Hub:
    ports.inputs += [ESP, VIRMIDI]
    ports.outputs += [ESP, VIRMIDI]
    hub = mh.Hub(ports, lambda _: None)
    hub.open()
    hub.poll()
    return hub


def test_board_traffic_also_reaches_virmidi() -> None:
    ports = FakePorts()
    hub = _hub_with_virmidi(ports)
    assert hub.raw_name == VIRMIDI and hub.board_name == ESP
    ports.opened["board-in"].feed([0x99, 36, 100])
    assert ports.sent == [("hub-out", [0x99, 36, 100]), ("raw-out", [0x99, 36, 100])]


def test_virmidi_traffic_goes_to_the_board_and_is_not_echoed() -> None:
    ports = FakePorts()
    _hub_with_virmidi(ports)
    frame = [0xF0, 0x7D, 0x1E, 0x0A, 0xF7]
    ports.opened["raw-in"].feed(frame)
    assert ports.sent == [("board-out", frame)]


def test_hub_ports_traffic_does_not_reach_virmidi() -> None:
    ports = FakePorts()
    _hub_with_virmidi(ports)
    ports.opened["hub-in"].feed([0xB0, 1, 5])
    assert [name for name, _ in ports.sent] == ["hub-out", "board-out"]


def test_virmidi_detaches_when_the_module_goes() -> None:
    ports = FakePorts()
    hub = _hub_with_virmidi(ports)
    ports.inputs.remove(VIRMIDI)
    ports.outputs.remove(VIRMIDI)
    hub.poll()
    assert hub.raw_name is None and ports.opened["raw-in"].closed
    assert hub.board_name == ESP


# -- companion verbs through the hub's own board port -------------------------

def _reply(text: str, last: bool = True) -> list[int]:
    return [0xF0, 0x7D, 0x1F, 0x03 if last else 0x02, *text.encode("ascii"), 0xF7]


def _attached_hub() -> tuple[mh.Hub, FakePorts]:
    ports = FakePorts()
    ports.inputs.append(ESP)
    ports.outputs.append(ESP)
    hub = mh.Hub(ports, lambda _: None, host_info=lambda: "HOST_INFO x")
    hub.open()
    hub.poll()
    return hub, ports


def _answer_when_sent(ports: FakePorts, chunks: list[list[int]]) -> threading.Thread:
    request = list(sysex_companion("KIT_STATUS"))

    def run() -> None:
        for _ in range(200):
            if any(m == request for _, m in ports.sent):
                for c in chunks:
                    ports.opened["board-in"].feed(c)
                return
            time.sleep(0.01)

    t = threading.Thread(target=run)
    t.start()
    return t


def test_companion_round_trip_two_chunks() -> None:
    hub, ports = _attached_hub()
    t = _answer_when_sent(ports, [_reply("KITSTATUS: current=3 ", last=False),
                                  _reply("name=DRUMS")])
    assert hub.companion("KIT_STATUS", 2.0) == "KITSTATUS: current=3 name=DRUMS"
    t.join()


def test_companion_reply_not_forwarded_to_daw() -> None:
    hub, ports = _attached_hub()
    note_on = [0x90, 60, 100]
    t = _answer_when_sent(ports, [note_on, _reply("KITSTATUS: name=DRUMS")])
    hub.companion("KIT_STATUS", 2.0)
    t.join()
    to_daw = [m for name, m in ports.sent if name == "hub-out"]
    assert note_on in to_daw
    assert all(m[:3] != [0xF0, 0x7D, 0x1F] for m in to_daw)


def test_report_waits_while_companion_in_flight() -> None:
    hub, _ = _attached_hub()
    with hub._companion_lock:
        before = hub.reports_sent
        hub.report(1e9)
        assert hub.reports_sent == before
    hub.report(1e9)
    assert hub.reports_sent == before + 1


def test_companion_without_board_and_timeout() -> None:
    hub = mh.Hub(FakePorts(), lambda _: None)
    hub.open()
    with pytest.raises(HilError) as e:
        hub.companion("KIT_STATUS", 0.1)
    assert e.value.code == NO_DEVICE
    hub2, _ = _attached_hub()
    with pytest.raises(HilError) as e:
        hub2.companion("KIT_STATUS", 0.1)
    assert e.value.code == TIMEOUT


# -- the control socket: run() serves it, one hub per user -------------------

def _wait_endpoint(lock_dir: Path) -> None:
    f = hub_ipc.endpoint_file(lock_dir)
    for _ in range(300):
        if f.exists():
            return
        time.sleep(0.01)
    raise AssertionError("the hub never wrote its endpoint")


def _run_in_thread(lock_dir: Path, rounds: int) -> threading.Thread:
    t = threading.Thread(target=mh.run, kwargs=dict(
        log=lambda _: None, ports=FakePorts(), sleep=lambda s: time.sleep(0.01),
        rounds=rounds, lock_dir=lock_dir, find_board=lambda: None,
        link_factory=lambda port: None))
    t.start()
    return t


@pytest.mark.skipif(sys.platform == "win32", reason="the MIDI hub is Linux-only (/proc/stat)")
def test_run_serves_status_and_cleans_up(tmp_path: Path) -> None:
    t = _run_in_thread(tmp_path, rounds=100)
    _wait_endpoint(tmp_path)
    c = hub_ipc.Connection.open(tmp_path, client="test", purpose="status")
    try:
        st = c.call("status")
        assert st["running"] is True and st["clients"][0]["client"] == "test"
    finally:
        c.close()
    t.join()
    assert not hub_ipc.endpoint_file(tmp_path).exists()


def test_second_hub_exits_quietly(tmp_path: Path) -> None:
    held = PortLock(mh.HUB_LOCK_KEY, "hub", lock_dir=tmp_path)
    held.acquire()
    lines: list[str] = []
    try:
        mh.run(lines.append, ports=FakePorts(), sleep=lambda s: None, rounds=5,
               lock_dir=tmp_path, find_board=lambda: None, link_factory=lambda port: None)
        assert not hub_ipc.endpoint_file(tmp_path).exists()
        assert any("another hub is running" in ln for ln in lines)
    finally:
        held.release()


@pytest.mark.skipif(sys.platform == "win32", reason="the MIDI hub is Linux-only (/proc/stat)")
def test_dropped_connection_releases_the_exclusive_op(tmp_path: Path) -> None:
    t = _run_in_thread(tmp_path, rounds=150)
    _wait_endpoint(tmp_path)
    a = hub_ipc.Connection.open(tmp_path, client="flasher", purpose="ota")
    a.call("lease", {"purpose": "ota flash"})
    a.close()
    b = hub_ipc.Connection.open(tmp_path, client="cp-tools", purpose="pads")
    try:
        for _ in range(100):
            if b.call("status")["lease"] is None:
                break
            time.sleep(0.01)
        assert b.call("status")["lease"] is None
    finally:
        b.close()
    t.join()


def test_open_without_virtual_ports() -> None:
    class NoVirtual(FakePorts):
        def open_virtual_in(self, name):
            raise NotImplementedError("WinMM has no virtual ports")

        def open_virtual_out(self, name):
            raise NotImplementedError("WinMM has no virtual ports")

    hub = mh.Hub(NoVirtual(), lambda _: None)
    hub.open()
    assert hub.hub_in is None and hub.hub_out is None


def test_control_status_is_none_without_a_hub(tmp_path: Path) -> None:
    assert mh.control_status(tmp_path) is None


def test_render_hub_control() -> None:
    from crosspad_hil.cli import render_hub_control

    assert render_hub_control(None) == "control=down"
    text = render_hub_control({"clients": [{}, {}], "cdc_open": True,
                               "lease": {"client": "bench", "purpose": "ota flash", "pid": 7}})
    assert text == "clients=2 exclusive=bench (ota flash, pid 7) cdc=open"


def test_run_serves_without_proc_and_reports_nothing(tmp_path: Path, monkeypatch) -> None:
    """Windows and macOS have no /proc: the hub still arbitrates, it just sends no HOST_INFO."""
    def no_proc(*_a, **_k):
        raise FileNotFoundError("/proc/stat")

    monkeypatch.setattr(mh.host_report, "CpuMeter", no_proc)
    t = _run_in_thread(tmp_path, rounds=100)
    _wait_endpoint(tmp_path)
    c = hub_ipc.Connection.open(tmp_path, client="test", purpose="status")
    try:
        assert c.call("status")["running"] is True
    finally:
        c.close()
    t.join()


def test_a_failed_start_gives_the_singleton_back(tmp_path: Path, monkeypatch) -> None:
    def refuse(self):
        raise OSError("address in use")

    monkeypatch.setattr(mh.hub_ipc.Server, "start", refuse)
    with pytest.raises(OSError):
        mh.run(lambda _: None, ports=FakePorts(), sleep=lambda s: None, rounds=1,
               lock_dir=tmp_path, find_board=lambda: None, link_factory=lambda port: None)
    again = PortLock(mh.HUB_LOCK_KEY, "hub", lock_dir=tmp_path)
    again.acquire()
    again.release()


def test_a_board_still_busy_with_the_hubs_notify_is_asked_again() -> None:
    """The board runs one companion verb at a time; right after the hub's own
    HOST_INFO it answers ERR busy, which is the hub's doing, not the client's."""
    hub, ports = _attached_hub()
    hub.report(1e9)                               # HOST_INFO goes out
    request = list(sysex_companion("KIT_STATUS"))
    board_in = ports.opened["board-in"]
    answers = [_reply("ERR busy"), _reply("KITSTATUS: name=DRUMS")]

    def board() -> None:
        answered = 0
        deadline = time.monotonic() + 3
        while answered < len(answers) and time.monotonic() < deadline:
            asked = sum(1 for _, m in ports.sent if m == request)
            if asked > answered:
                board_in.feed(answers[answered])
                answered += 1
            time.sleep(0.005)

    t = threading.Thread(target=board)
    t.start()
    assert hub.companion("KIT_STATUS", 2.0) == "KITSTATUS: name=DRUMS"
    t.join()


def test_err_busy_long_after_a_notify_is_the_boards_answer() -> None:
    hub, ports = _attached_hub()
    t = _answer_when_sent(ports, [_reply("ERR busy")])
    assert hub.companion("KIT_STATUS", 2.0) == "ERR busy"
    t.join()


def test_without_virtual_ports_the_board_midi_is_left_to_others() -> None:
    """WinMM ports are single-client: a hub there must not take the board's MIDI
    from a DAW or from the USB profile switch."""
    class NoVirtual(FakePorts):
        def open_virtual_in(self, name):
            raise NotImplementedError("WinMM has no virtual ports")

        def open_virtual_out(self, name):
            raise NotImplementedError("WinMM has no virtual ports")

    ports = NoVirtual()
    ports.inputs.append(ESP)
    ports.outputs.append(ESP)
    hub = mh.Hub(ports, lambda _: None)
    hub.open()
    hub.poll()
    assert hub.board_name is None
    assert "board-in" not in ports.opened


# -- version verification over companion SysEx --------------------------------

STATUS = ("FWUPD: state=idle running=v1.1.0-rc2 kit=full latest=1.1.0-rc3 channel=beta "
          "progress=0 last_check=123 err=-")
LATEST = "FW_LATEST: latest=1.1.0-rc3 running=v1.1.0-rc2 update=1"


def reply(text: str) -> list[int]:
    return [0xF0, 0x7D, 0x1F, 0x03, *text.encode("ascii"), 0xF7]


def attached_hub(tmp_path, catalog="1.1.0-rc3", log=None):
    ports = FakePorts()
    hub = mh.Hub(ports, log or (lambda s: None), host_info=lambda: "HOST_INFO cpu=1",
                 catalog_latest=lambda kit, ch: catalog, versions_path=tmp_path / "v.json")
    hub.open()
    ports.inputs.append(ESP)
    ports.outputs.append(ESP)
    hub.poll()
    return ports, hub


def test_hub_requests_status_then_latest_and_records_a_verdict(tmp_path):
    ports, hub = attached_hub(tmp_path)
    hub.verify_tick(0.0)
    assert ports.sent[-1][1] == list(sysex_companion("FW_UPD_STATUS"))
    hub._from_board((reply(STATUS), 0.0))
    hub.verify_tick(1.0)
    assert ports.sent[-1][1] == list(sysex_companion("FW_LATEST"))
    hub._from_board((reply(LATEST), 0.0))
    data = json.loads((tmp_path / "v.json").read_text())
    assert data["verdict"] == "update_offered" and data["kit"] == "full" and data["channel"] == "beta"
    assert data["running"] == "v1.1.0-rc2" and data["catalog_latest"] == "1.1.0-rc3"
    assert hub.board_kit == "full" and hub.board_channel == "beta"
    assert list(tmp_path.iterdir()) == [tmp_path / "v.json"]       # no temp file left behind


def test_a_multi_chunk_reply_is_joined(tmp_path):
    ports, hub = attached_hub(tmp_path)
    hub.verify_tick(0.0)
    half = len(STATUS) // 2
    hub._from_board(([0xF0, 0x7D, 0x1F, 0x02, *STATUS[:half].encode(), 0xF7], 0.0))
    hub._from_board((reply(STATUS[half:]), 0.0))
    hub.verify_tick(1.0)
    assert hub.board_channel == "beta"


def test_requested_reply_is_not_forwarded_to_the_daw(tmp_path):
    ports, hub = attached_hub(tmp_path)
    hub.verify_tick(0.0)
    before = hub.forwarded_to_daw
    hub._from_board((reply("FWUPD: state=idle running=v1 kit=full latest=- channel=stable err=-"), 0.0))
    assert hub.forwarded_to_daw == before


def test_foreign_reply_is_forwarded(tmp_path):
    ports, hub = attached_hub(tmp_path)
    before = hub.forwarded_to_daw
    hub._from_board((reply("KITSTATUS: current=x"), 0.0))
    assert hub.forwarded_to_daw == before + 1


def test_a_reply_to_another_verb_is_forwarded_while_we_wait(tmp_path):
    ports, hub = attached_hub(tmp_path)
    hub.verify_tick(0.0)
    before = hub.forwarded_to_daw
    hub._from_board((reply("KITSTATUS: current=x"), 0.0))
    assert hub.forwarded_to_daw == before + 1 and hub._pending is not None


def test_unanswered_request_expires(tmp_path):
    ports, hub = attached_hub(tmp_path)
    hub.verify_tick(0.0)
    hub.verify_tick(mh.VERIFY_TIMEOUT_S + 0.1)
    assert hub._pending is None


def test_no_catalog_source_means_no_verification(tmp_path):
    ports = FakePorts()
    hub = mh.Hub(ports, lambda s: None, versions_path=tmp_path / "v.json")
    hub.open()
    ports.inputs.append(ESP)
    ports.outputs.append(ESP)
    hub.poll()
    hub.verify_tick(0.0)
    assert ports.sent == [] and hub._pending is None


def test_report_drives_verification_next_to_the_host_notifies(tmp_path):
    ports, hub = attached_hub(tmp_path)
    hub.report(100.0)
    assert list(sysex_companion("FW_UPD_STATUS")) in [m for _, m in ports.sent]


@pytest.mark.parametrize("text", [
    "ERR unknown verb",
    "FWUPD: running=v1.1.0-rc2 kit=Full! channel=beta",
    "FWUPD: running=v1.1.0-rc2 kit=../../etc channel=beta",
    "FWUPD: running=v1.1.0-rc2 kit=full channel=" + "a" * 200,
    "FWUPD",
    "FWUPD: " + "k=v " * 400,
])
def test_a_malformed_status_reply_is_dropped_not_raised(tmp_path, text):
    logged: list[str] = []
    ports, hub = attached_hub(tmp_path, log=logged.append)
    hub.verify_tick(0.0)
    hub._from_board((reply(text), 0.0))
    assert not (tmp_path / "v.json").exists()
    assert hub.board_kit is None and hub.board_channel is None
    assert hub._pending is None


def test_odd_values_are_bounded_before_they_reach_the_file(tmp_path):
    ports, hub = attached_hub(tmp_path)
    hub.verify_tick(0.0)
    hub._from_board((reply("FWUPD: running=<script> kit=full channel=stable last_check=x"), 0.0))
    hub.verify_tick(1.0)
    hub._from_board((reply("FW_LATEST: latest=$(rm) update=7"), 0.0))
    data = json.loads((tmp_path / "v.json").read_text())
    assert data["running"] is None and data["board_latest"] is None
    assert data["update"] is None and data["last_check"] is None
    assert data["verdict"] == "unknown"


def test_a_failing_versions_write_does_not_escape(tmp_path):
    logged: list[str] = []
    ports = FakePorts()
    blocker = tmp_path / "file"
    blocker.write_text("x")
    hub = mh.Hub(ports, logged.append, catalog_latest=lambda k, c: "1.1.0",
                 versions_path=blocker / "v.json")
    hub.open()
    ports.inputs.append(ESP)
    ports.outputs.append(ESP)
    hub.poll()
    hub.verify_tick(0.0)
    hub._from_board((reply(STATUS), 0.0))
    hub.verify_tick(1.0)
    hub._from_board((reply(LATEST), 0.0))
    assert any("reply dropped" in m for m in logged)


def test_verdicts():
    assert mh.verdict({"running": "v1.1.0-rc3"}, {"latest": "1.1.0-rc3", "update": "0"}, "1.1.0-rc3") == "up_to_date"
    assert mh.verdict({"running": "v1.1.0-rc2"}, {"latest": "1.1.0-rc3", "update": "1"}, "1.1.0-rc3") == "update_offered"
    assert mh.verdict({"running": "v1.1.0-rc2"}, {"latest": "1.1.0-rc2", "update": "0"}, "1.1.0-rc3") == "board_behind_catalog"
    assert mh.verdict({"running": "v1.1.0-rc2"}, {"latest": "-", "update": "0"}, None) == "catalog_unknown"


def test_verdicts_for_dev_builds_boards_ahead_and_missing_readings():
    real = mh.fw_catalog.reply_fields(
        "FWUPD: state=up_to_date running=v1.1.0-rc4-33-gf6be957-dirty kit=full latest=- "
        "channel=stable progress=0 last_check=1791754100 err=- stack_free=3144 inst_stack_free=0")
    assert mh.verdict(real, {"latest": "-", "update": "0"}, "1.1.0-rc4") == "dev_build"
    assert mh.verdict({"running": "v1.1.0-rc4-2-gabc1234"}, {}, "1.1.0-rc4") == "dev_build"
    assert mh.verdict({"running": "v1.1.0-rc4"}, {"latest": "1.1.0-rc3", "update": "0"},
                      "1.1.0-rc3") == "board_ahead"
    assert mh.verdict({"running": "v1.1.0-rc10"}, {}, "1.1.0-rc9") == "board_ahead"
    assert mh.verdict({}, {"latest": "1.1.0-rc3", "update": "1"}, "1.1.0-rc3") == "unknown"
    assert mh.verdict({"running": ""}, {}, "1.1.0-rc3") == "unknown"


def test_the_real_bench_reply_is_recorded_as_dev_build(tmp_path):
    ports, hub = attached_hub(tmp_path, catalog="1.1.0-rc4")
    hub.verify_tick(0.0)
    hub._from_board((reply("FWUPD: state=up_to_date running=v1.1.0-rc4-33-gf6be957-dirty kit=full "
                           "latest=- channel=stable progress=0 last_check=1791754100 err=- "
                           "stack_free=3144 inst_stack_free=0"), 0.0))
    hub.verify_tick(1.0)
    hub._from_board((reply("FW_LATEST: latest=- running=v1.1.0-rc4-33-gf6be957-dirty update=0"), 0.0))
    data = json.loads((tmp_path / "v.json").read_text())
    assert data["verdict"] == "dev_build" and data["running"] == "v1.1.0-rc4-33-gf6be957-dirty"


def _round(hub, t):
    """One full verification round starting at tick `t`."""
    hub.verify_tick(t)
    hub._from_board((reply(STATUS), 0.0))
    hub.verify_tick(t + 1)
    hub._from_board((reply(LATEST), 0.0))


def test_an_unknown_catalog_is_rechecked_after_the_short_retry(tmp_path):
    catalog = {"v": None}
    ports = FakePorts()
    hub = mh.Hub(ports, lambda s: None, host_info=lambda: "HOST_INFO cpu=1",
                 catalog_latest=lambda kit, ch: catalog["v"], versions_path=tmp_path / "v.json")
    hub.open()
    ports.inputs.append(ESP)
    ports.outputs.append(ESP)
    hub.poll()
    _round(hub, 0.0)
    assert json.loads((tmp_path / "v.json").read_text())["verdict"] == "catalog_unknown"
    n = len(ports.sent)
    hub.verify_tick(mh.VERIFY_RETRY_S - 1)          # 29 s after the round: not yet
    assert len(ports.sent) == n
    catalog["v"] = "1.1.0-rc3"
    _round(hub, mh.VERIFY_RETRY_S + 1)              # due after the retry, not after 600 s
    assert json.loads((tmp_path / "v.json").read_text())["verdict"] == "update_offered"


def test_a_known_catalog_waits_the_full_interval(tmp_path):
    ports, hub = attached_hub(tmp_path)
    _round(hub, 0.0)
    n = len(ports.sent)
    hub.verify_tick(mh.VERIFY_EVERY_S - 1)
    assert len(ports.sent) == n
    hub.verify_tick(mh.VERIFY_EVERY_S + 1)
    assert len(ports.sent) == n + 1


# -- no catalog is not the same as no release on the channel -------------------------------------

def _hub_with(tmp_path, catalog_latest, held):
    ports = FakePorts()
    hub = mh.Hub(ports, lambda s: None, host_info=lambda: "HOST_INFO cpu=1",
                 catalog_latest=catalog_latest, catalog_held=held,
                 versions_path=tmp_path / "v.json")
    hub.open()
    ports.inputs.append(ESP)
    ports.outputs.append(ESP)
    hub.poll()
    return ports, hub


def _verdict_of(tmp_path):
    return json.loads((tmp_path / "v.json").read_text())["verdict"]


def test_verdict_tells_no_catalog_from_no_release_on_the_channel():
    running, latest = {"running": "v1.1.0-rc2"}, {"latest": "-", "update": "0"}
    assert mh.verdict(running, latest, None, catalog_held=False) == "catalog_unknown"
    assert mh.verdict(running, latest, None, catalog_held=True) == "no_release_on_channel"
    assert mh.verdict(running, latest, None) == "catalog_unknown"      # held unknown: none held
    assert mh.verdict({}, latest, None, catalog_held=True) == "unknown"
    assert "no_release_on_channel" not in mh.RECHECK_VERDICTS


def test_a_held_catalog_without_a_release_is_not_polled_every_30_seconds(tmp_path):
    ports, hub = _hub_with(tmp_path, lambda k, c: None, lambda: True)
    _round(hub, 0.0)
    assert _verdict_of(tmp_path) == "no_release_on_channel"
    n = len(ports.sent)
    hub.verify_tick(mh.VERIFY_RETRY_S + 1)
    hub.verify_tick(mh.VERIFY_EVERY_S - 1)
    assert len(ports.sent) == n                       # the normal 600 s pace
    hub.verify_tick(mh.VERIFY_EVERY_S + 1)
    assert len(ports.sent) == n + 1


def test_no_catalog_held_keeps_the_30_second_retry(tmp_path):
    ports, hub = _hub_with(tmp_path, lambda k, c: None, lambda: False)
    _round(hub, 0.0)
    assert _verdict_of(tmp_path) == "catalog_unknown"
    n = len(ports.sent)
    hub.verify_tick(mh.VERIFY_RETRY_S + 1)
    assert len(ports.sent) == n + 1


def test_a_stable_board_against_a_beta_only_catalog_end_to_end(tmp_path):
    cache = mh.fw_catalog.CatalogCache(
        fetch=lambda rev: {"kits": [{"kit": "full", "versions": [
            {"version": "1.1.0-rc4", "tag": "v1.1.0-rc4", "channel": "beta"}]}]},
        clock=lambda: 0.0, background=False)
    stable = STATUS.replace("channel=beta", "channel=stable")
    ports, hub = _hub_with(tmp_path, cache.latest, lambda: cache.has_catalog)
    hub.verify_tick(0.0)
    hub._from_board((reply(stable), 0.0))
    hub.verify_tick(1.0)
    hub._from_board((reply(LATEST), 0.0))
    assert _verdict_of(tmp_path) == "catalog_unknown"     # the first read is what fetches
    assert cache.has_catalog
    n = len(ports.sent)
    hub.verify_tick(mh.VERIFY_RETRY_S + 1)                # the one short retry ...
    hub._from_board((reply(stable), 0.0))
    hub.verify_tick(mh.VERIFY_RETRY_S + 2)
    hub._from_board((reply(LATEST), 0.0))
    assert len(ports.sent) == n + 2
    assert _verdict_of(tmp_path) == "no_release_on_channel"
    hub.verify_tick(2 * mh.VERIFY_RETRY_S + 5)            # ... and then the normal pace
    assert len(ports.sent) == n + 2


# -- strings the code lets through but version_key cannot order ----------------------------------

def test_a_dirty_build_on_a_tag_is_a_dev_build():
    for running in ("v1.1.0-rc4-dirty", "1.1.0-dirty", "v1.1.0-rc4-33-gf6be957-dirty"):
        assert mh.verdict({"running": running}, {}, "1.1.0-rc4") == "dev_build", running


def test_an_unorderable_running_or_catalog_version_never_raises():
    latest = {"latest": "-", "update": "0"}
    assert mh.verdict({"running": "v1.2.0-rc1a"}, latest, "1.1.0-rc4") == "unordered"
    assert mh.verdict({"running": "v1.1.0-rc4"}, latest, "1.2.0-rc1a") == "unordered"
    assert "unordered" not in mh.RECHECK_VERDICTS


def test_the_dirty_tag_reply_is_recorded_end_to_end(tmp_path):
    ports, hub = attached_hub(tmp_path, catalog="1.1.0-rc4")
    hub.verify_tick(0.0)
    hub._from_board((reply(STATUS.replace("running=v1.1.0-rc2", "running=v1.1.0-rc4-dirty")), 0.0))
    hub.verify_tick(1.0)
    hub._from_board((reply(LATEST), 0.0))
    assert _verdict_of(tmp_path) == "dev_build"


def test_a_refused_request_is_asked_again_after_the_short_retry(tmp_path):
    logged: list[str] = []
    ports, hub = attached_hub(tmp_path, log=logged.append)
    hub.verify_tick(0.0)
    hub._from_board((reply("ERR busy"), 0.0))         # the board was mid-job: not an answer
    assert any("reply dropped" in m for m in logged)
    n = len(ports.sent)
    hub.verify_tick(mh.VERIFY_RETRY_S + 1)            # not 600 s
    assert len(ports.sent) == n + 1


# -- a client's companion request beside the hub's own version check ---------

def test_report_runs_no_version_check_while_a_client_request_is_out(tmp_path):
    ports, hub = attached_hub(tmp_path)
    with hub._companion_lock:
        hub.report(0.0)
    assert hub._pending is None and ports.sent == []


def test_client_request_waits_for_the_version_reply(tmp_path):
    ports, hub = attached_hub(tmp_path)
    hub.verify_tick(0.0)                                  # FW_UPD_STATUS out
    request = list(sysex_companion("KIT_STATUS"))
    early: list[bool] = []

    def board() -> None:
        time.sleep(0.1)
        early.append(any(m == request for _, m in ports.sent))
        ports.opened["board-in"].feed(reply(STATUS))
        for _ in range(200):
            if any(m == request for _, m in ports.sent):
                ports.opened["board-in"].feed(reply("KITSTATUS: name=DRUMS"))
                return
            time.sleep(0.01)

    t = threading.Thread(target=board)
    t.start()
    assert hub.companion("KIT_STATUS", 2.0) == "KITSTATUS: name=DRUMS"
    t.join()
    assert early == [False]                               # not sent over the hub's request
    assert hub._round["FW_UPD_STATUS"]["kit"] == "full"   # the version reply was the hub's


def test_an_unanswered_version_request_does_not_hold_a_client_for_good(tmp_path, monkeypatch):
    logged: list[str] = []
    monkeypatch.setattr(mh, "VERIFY_TIMEOUT_S", 0.05)
    ports, hub = attached_hub(tmp_path, log=logged.append)
    hub.verify_tick(0.0)
    t = _answer_when_sent(ports, [reply("KITSTATUS: name=DRUMS")])
    assert hub.companion("KIT_STATUS", 2.0) == "KITSTATUS: name=DRUMS"
    t.join()
    assert hub._pending is None and any("no reply" in m for m in logged)


def test_a_client_deadline_inside_the_version_wait_is_a_timeout(tmp_path):
    ports, hub = attached_hub(tmp_path)
    hub.verify_tick(0.0)
    with pytest.raises(HilError) as e:
        hub.companion("KIT_STATUS", 0.05)
    assert e.value.code == TIMEOUT and "version check" in e.value.message
