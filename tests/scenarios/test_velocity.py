"""Scenario `velocity` against a fake board — no sound card, no MIDI, no device.

The fake closes the same loop the real rig closes: the pad transport records
every press the scheduler makes, and the fake capture renders exactly those
presses into the WAV that the real `analyze_velocity` then reads. So the curve
under test is the one the schedule actually produced — a scenario that sent the
wrong velocities, in the wrong order, or dropped half of them cannot pass here.
"""
from __future__ import annotations

import threading
import time
import wave
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from crosspad_hil.capture import CaptureResult
from crosspad_hil.cdc import CdcLink
from crosspad_hil.devices import Device, MidiPortInfo, Ports, SerialPortInfo, UsbMode
from crosspad_hil.errors import TIMEOUT, HilError
from crosspad_hil.midi import MidiRole
from crosspad_hil.scenarios import base, get, velocity
from tests.fakes import FakeSerial

SR = 48000
CDC = SerialPortInfo(path="/dev/ttyACM0", vid=0x303A, pid=0x3456, serial="CP-1",
                     product="Crosspad", location="1-2.1")
ESP_MIDI = MidiPortInfo(name="Crosspad MIDI 1", rtmidi_out=0, rtmidi_in=0,
                        alsa_hw="hw:4,0,0", rawmidi="/dev/snd/midiC4D0")
STM_MIDI = MidiPortInfo(name="CrossPad MIDI+Serial 2", rtmidi_out=1, rtmidi_in=1,
                        alsa_hw="hw:5,0,0", rawmidi="/dev/snd/midiC5D0")
# A sweep the analyser can judge: 18 dB between the quietest and the loudest.
VELS = [16, 64, 127]
CDC_SCRIPT = [
    ("KIT_LOAD 0", "OK"),
    ("KIT_STATUS", "KITSTATUS: current=0 loading=0 pending=-1 name=Basic"),
    ("APP_LIST", "APPS: Sampler,Mixer running=Sampler"),
    ("APP_START Sampler", "OK"),
]


def make_device(*, stm_midi: MidiPortInfo | None = STM_MIDI) -> Device:
    return Device(id="dev_ab12", serial="CP-1", usb_mode=UsbMode.DEFAULT,
                  ports=Ports(cdc=CDC, esp_midi=ESP_MIDI, stm_midi=stm_midi))


# --------------------------------------------------------------------------- the fake board


class PadTransportFake:
    """What `open_transport` hands back: stamps every press so it can be rendered."""

    name = "sysex_stm"

    def __init__(self) -> None:
        self.presses: list[tuple[float, int, int]] = []
        self.releases: list[tuple[float, int]] = []
        self.closed = False

    def press(self, pad: int, vel: int) -> None:
        self.presses.append((time.monotonic(), pad, vel))

    def release(self, pad: int) -> None:
        self.releases.append((time.monotonic(), pad))

    def close(self) -> None:
        self.closed = True


def render(path: Path, presses: list[tuple[float, int, int]], t0: float, seconds: float, *,
           flat: bool = False, drop: frozenset[int] = frozenset()) -> float:
    """Turn the recorded presses into the take the capture endpoint would have heard.

    A short decaying burst per hit, its peak proportional to velocity, over a
    noise floor quiet enough that the analyser's floor+12 dB threshold sees even
    the quietest hit. Returns the peak in dBFS.
    """
    n = max(int(seconds * SR), SR // 4)
    rng = np.random.default_rng(7)
    x = rng.uniform(-1e-4, 1e-4, n)
    k = np.arange(int(0.08 * SR))
    shape = np.exp(-k / (0.01 * SR)) * np.sin(2 * np.pi * 300 * k / SR)
    for i, (t, _pad, vel) in enumerate(presses):
        if i in drop:
            continue
        start = int((t - t0) * SR)
        room = min(len(k), n - start)
        if room <= 0:
            continue
        x[start:start + room] += (0.5 if flat else 0.5 * vel / 127.0) * shape[:room]
    frames = (np.clip(x, -1.0, 1.0) * 32767).astype("<i2")
    stereo = np.repeat(frames, 2)
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as w:
        w.setnchannels(2)
        w.setsampwidth(2)
        w.setframerate(SR)
        w.writeframes(stereo.tobytes())
    return float(20 * np.log10(max(float(np.abs(x).max()), 1e-12)))


class Board:
    """One fake CrossPad: its pad transport, its capture, and how it misbehaves."""

    def __init__(self) -> None:
        self.transport = PadTransportFake()
        self.session: CaptureFake | None = None
        self.flat = False
        self.drop: frozenset[int] = frozenset()
        self.start_error: HilError | None = None
        self.echo = EchoBus()

    def open_transport(self, device: Device, kind: str, **kw: Any) -> PadTransportFake:
        assert kind == "sysex_stm"      # CDC is gone in the audio profile
        return self.transport

    def capture_session(self, device: Device, **kw: Any) -> CaptureFake:
        self.session = CaptureFake(self, device, **kw)
        return self.session

    def midi(self, device: Device, role: MidiRole) -> FakeMidi:
        return self.echo.port(role)


class CaptureFake:
    """CaptureSession's shape: mode switch, route and restore, minus the hardware."""

    def __init__(self, board: Board, device: Device, *, seconds: float | None = None,
                 preset: str = "", out: Any = None, keep_mode: bool = False,
                 log: Any = None, **kw: Any) -> None:
        self._board = board
        self.device = device
        self.seconds = seconds
        self.preset = preset
        self.keep_mode = keep_mode
        self.path = Path(out)
        self.started_at: float | None = None
        self.result: CaptureResult | None = None
        self.closed = False

    def start(self) -> dict:
        if self._board.start_error is not None:
            raise self._board.start_error
        self.started_at = time.monotonic()
        return {"device": self.device.id, "wav": str(self.path), "preset": self.preset,
                "backend": "fake", "sample_rate": SR, "channels": 2, "seconds": self.seconds,
                "audio_mode_entered": True, "audio_tasks_resumed": True}

    def stop(self) -> CaptureResult:
        if self.result is not None:
            return self.result
        t0 = self.started_at or time.monotonic()
        seconds = time.monotonic() - t0 + 0.1
        peak = render(self.path, self._board.transport.presses, t0, seconds,
                      flat=self._board.flat, drop=self._board.drop)
        self.result = CaptureResult(
            wav=str(self.path), seconds=round(seconds, 3), peak_dbfs=round(peak, 2),
            rms_dbfs=-30.0, overruns=0, silent=peak < -60.0, sample_rate=SR, channels=2,
            frames=int(seconds * SR), device=self.device.id, preset=self.preset,
            backend="fake", health={"ok": True, "detail": "fake"})
        return self.result

    def close(self) -> None:
        self.closed = True


class FakeMidi:
    """MidiIO's shape for the echo probe: the bridge writes, the ESP port answers."""

    def __init__(self, bus: EchoBus, role: MidiRole) -> None:
        self._bus = bus
        self.role = role
        self.queue: deque[bytes] = deque()
        self.opened = False
        self.closed = False

    def open(self) -> None:
        self.opened = True

    def close(self) -> None:
        self.closed = True

    def drain(self) -> int:
        n = len(self.queue)
        self.queue.clear()
        return n

    def send_sysex(self, frame: bytes) -> None:
        self._bus.sent.append(bytes(frame))
        if self._bus.answers and self._bus.esp is not None:
            self._bus.esp.queue.append(bytes(frame))

    def receive(self, timeout_s: float = 1.0) -> bytes | None:
        return self.queue.popleft() if self.queue else None


class EchoBus:
    def __init__(self) -> None:
        self.sent: list[bytes] = []
        self.answers = True
        self.esp: FakeMidi | None = None
        self.stm: FakeMidi | None = None

    def port(self, role: MidiRole) -> FakeMidi:
        io = FakeMidi(self, role)
        if role is MidiRole.ESP:
            self.esp = io
        else:
            self.stm = io
        return io


@dataclass
class FakeContext(base.Context):
    cdc_fake: FakeSerial = None  # type: ignore[assignment]
    cdc_opens: int = 0

    def open_cdc(self) -> CdcLink:
        self.cdc_opens += 1
        fake = self.cdc_fake
        link = CdcLink(self.device.ports.cdc.path, serial_factory=lambda path, **kw: fake)
        link.open()
        self.opened.append(link)
        return link


def make_ctx(tmp_path: Path, *, script: list | None = None,
             stm_midi: MidiPortInfo | None = STM_MIDI) -> tuple[FakeContext, Board, FakeSerial]:
    fake = FakeSerial(CDC_SCRIPT if script is None else script)
    board = Board()
    ctx = FakeContext(device=make_device(stm_midi=stm_midi), workdir=tmp_path,
                      cancelled=threading.Event(), log=lambda s: None, cdc_fake=fake)
    return ctx, board, fake


@pytest.fixture(autouse=True)
def fast(monkeypatch: pytest.MonkeyPatch) -> None:
    """Collapse the human-scale waits; the sweep's own grid stays real."""
    monkeypatch.setattr(velocity, "LEAD_S", 0.2)
    monkeypatch.setattr(velocity, "TAIL_S", 0.2)
    monkeypatch.setattr(velocity, "ROUTE_SETTLE_S", 0.0)
    monkeypatch.setattr(velocity, "APP_RETRY_SETTLE_S", 0.0)
    monkeypatch.setattr(velocity, "ECHO_TIMEOUT_S", 0.05)
    monkeypatch.setattr(velocity, "STIM_TAIL_S", 1.0)


def wire(monkeypatch: pytest.MonkeyPatch, board: Board) -> None:
    monkeypatch.setattr(velocity, "CaptureSession", board.capture_session)
    monkeypatch.setattr(velocity, "open_transport", board.open_transport)
    monkeypatch.setattr(velocity, "MidiIO", board.midi)


def run(ctx: FakeContext, **params: Any) -> base.Report:
    kw: dict[str, Any] = {"vels": list(VELS), "rounds": 1, "period": 0.35,
                          "gate_ms": 40.0, "echo_probes": 0}
    kw.update(params)
    return velocity.SCENARIO.run(ctx, velocity.Params(**kw), base.Progress())


# --------------------------------------------------------------------------- tests


def test_registered_and_defaults() -> None:
    assert get("velocity") is velocity.SCENARIO
    p = velocity.Params()
    assert (p.kit, p.pad, p.rounds, p.period, p.gate_ms) == (0, 0, 3, 1.0, 80.0)
    # the thresholds hil_velocity shipped with
    assert (p.min_range_db, p.max_jitter_ms, p.echo_probes) == (12.0, 40.0, 20)
    assert p.vels == [8, 16, 24, 32, 48, 64, 80, 96, 112, 127]
    assert p.keep_mode is False and p.assume_ready is False


def test_sweep_passes_and_reports_the_curve(tmp_path: Path,
                                            monkeypatch: pytest.MonkeyPatch) -> None:
    ctx, board, fake = make_ctx(tmp_path)
    wire(monkeypatch, board)
    rep = run(ctx, rounds=2)
    assert rep.passed, rep.summary
    assert rep.exit_code == 0

    # the prologue put the device where the sweep needs it
    assert "KIT_LOAD 0" in fake.written and "APP_START Sampler" in fake.written
    assert rep.data["prologue"] == {"assume_ready": False, "kit": 0, "app": "Sampler"}

    # interleaved, not blocked: every velocity once per round, in order
    assert [vel for _t, _pad, vel in board.transport.presses] == VELS * 2
    assert len(board.transport.releases) == 6
    assert board.transport.closed

    body = rep.data["velocity"]
    assert body["expected"] == 6 and body["matched"] == 6 and body["missed"] == []
    assert [c["vel"] for c in body["curve"]] == VELS
    assert all(c["hits"] == 2 for c in body["curve"])
    rel = [c["measured_rel_db"] for c in body["curve"]]
    assert rel == sorted(rel)                       # louder velocity, louder hit
    assert body["dynamic_range_db"] >= 12.0
    assert body["monotonic_inversions"] == 0
    assert body["latency_ms"]["n"] == 6
    assert "PASS" in rep.summary

    roles = {a.role: a.path for a in rep.artifacts}
    assert roles["capture"].endswith("capture.wav")
    assert Path(roles["analysis"]).exists()
    assert board.session is not None and board.session.closed


def test_dropped_trigger_fails(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A hit the host sent that made no sound is the failure the old script led with."""
    ctx, board, _fake = make_ctx(tmp_path)
    board.drop = frozenset({1})
    wire(monkeypatch, board)
    rep = run(ctx)
    assert not rep.passed and rep.exit_code == 1
    body = rep.data["velocity"]
    assert body["missed"] == [1] and body["matched"] == 2
    assert any("never sounded" in r for r in body["reasons"])


def test_flat_response_fails_on_dynamic_range(tmp_path: Path,
                                              monkeypatch: pytest.MonkeyPatch) -> None:
    ctx, board, _fake = make_ctx(tmp_path)
    board.flat = True                    # every hit the same loudness: velocity is dead
    wire(monkeypatch, board)
    rep = run(ctx, rounds=2)
    assert not rep.passed and rep.exit_code == 1
    body = rep.data["velocity"]
    assert body["matched"] == 6          # they all sounded — they just all sounded the same
    assert body["dynamic_range_db"] < 12.0
    assert any("does not change loudness" in r for r in body["reasons"])


def test_min_range_db_is_the_threshold_that_decides(tmp_path: Path,
                                                    monkeypatch: pytest.MonkeyPatch) -> None:
    ctx, board, _fake = make_ctx(tmp_path)
    board.flat = True
    wire(monkeypatch, board)
    # A board that answers every velocity identically measures a range of zero
    # give or take the measurement, so the bar has to be under that noise for
    # "the threshold is what decides" to be what this test proves. Asking for
    # exactly 0.0 dB made it a test of the analyser's rounding instead.
    rep = run(ctx, min_range_db=-0.5)
    assert rep.passed and rep.exit_code == 0
    assert abs(rep.data["velocity"]["dynamic_range_db"]) < 0.5


def test_assume_ready_skips_the_prologue(tmp_path: Path,
                                         monkeypatch: pytest.MonkeyPatch) -> None:
    ctx, board, fake = make_ctx(tmp_path)
    wire(monkeypatch, board)
    rep = run(ctx, assume_ready=True)
    assert rep.passed, rep.summary
    assert ctx.cdc_opens == 0
    assert fake.written == []
    assert rep.data["prologue"] == {"assume_ready": True, "kit": None, "app": None}


def test_echo_probes_measure_the_control_path(tmp_path: Path,
                                              monkeypatch: pytest.MonkeyPatch) -> None:
    ctx, board, _fake = make_ctx(tmp_path)
    wire(monkeypatch, board)
    rep = run(ctx, echo_probes=4)
    assert rep.passed, rep.summary
    echo = rep.data["echo_rtt"]
    assert echo["n"] == 4
    assert echo["min_ms"] <= echo["median_ms"] <= echo["max_ms"]
    # the probes go down the bridge, and the ESP port is what answers
    assert len(board.echo.sent) == 4
    assert board.echo.stm is not None and board.echo.stm.closed
    assert board.echo.esp is not None and board.echo.esp.closed


def test_echo_without_replies_is_reported_as_none(tmp_path: Path,
                                                  monkeypatch: pytest.MonkeyPatch) -> None:
    ctx, board, _fake = make_ctx(tmp_path)
    board.echo.answers = False
    wire(monkeypatch, board)
    rep = run(ctx, echo_probes=2)
    assert rep.passed, rep.summary          # a dead echo does not fail the sweep
    assert rep.data["echo_rtt"] is None


def test_missing_stm_bridge_is_an_environment_error(tmp_path: Path,
                                                    monkeypatch: pytest.MonkeyPatch) -> None:
    ctx, board, _fake = make_ctx(tmp_path, stm_midi=None)
    wire(monkeypatch, board)
    rep = run(ctx)
    assert rep.exit_code == 2 and not rep.passed
    assert "STM bridge" in rep.summary
    assert ctx.cdc_opens == 0


def test_kit_load_failure_is_an_environment_error(tmp_path: Path,
                                                  monkeypatch: pytest.MonkeyPatch) -> None:
    ctx, board, _fake = make_ctx(tmp_path, script=[("KIT_LOAD 0", "ERR bad kit id")])
    wire(monkeypatch, board)
    rep = run(ctx)
    assert rep.exit_code == 2 and not rep.passed
    assert "KIT_LOAD 0 failed" in rep.summary
    assert board.session is None


def test_sampler_missing_from_the_launcher_is_an_environment_error(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    ctx, board, _fake = make_ctx(tmp_path, script=[
        ("KIT_LOAD 0", "OK"),
        ("KIT_STATUS", "KITSTATUS: current=0 loading=0 pending=-1 name=Basic"),
        ("APP_LIST", "APPS: Mixer running=Mixer"),
    ])
    wire(monkeypatch, board)
    rep = run(ctx)
    assert rep.exit_code == 2
    assert "Sampler" in rep.summary


def test_capture_that_never_starts_is_an_environment_error(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """hil_velocity: "audio card did not appear" is a bench problem, not a firmware one."""
    ctx, board, _fake = make_ctx(tmp_path)
    board.start_error = HilError(TIMEOUT, "no CrossPad UAC2 capture endpoint appeared",
                                 hint="usb-mode set audio")
    wire(monkeypatch, board)
    rep = run(ctx)
    assert rep.exit_code == 2 and not rep.passed
    assert "could not start the capture" in rep.summary
    assert board.transport.presses == []


def test_build_sweep_is_a_constant_grid() -> None:
    hits = velocity.build_sweep(3, [16, 127], rounds=2, period=0.5, gate_ms=80.0)
    assert [(h.t_ms, h.pad, h.vel, h.gate_ms) for h in hits] == [
        (0.0, 3, 16, 80.0), (500.0, 3, 127, 80.0),
        (1000.0, 3, 16, 80.0), (1500.0, 3, 127, 80.0)]
