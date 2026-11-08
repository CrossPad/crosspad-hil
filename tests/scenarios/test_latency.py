"""Scenario `latency` against a fake board: no card, no MIDI, no device, no wall clock.

The fake closes the loop the real bench closes, on a virtual monotonic clock:
the STM bridge records every note the scenario sends, the duplex stream
records every burst it schedules, and at the end the fake renders the take
the board's UAC IN would have delivered -- each pad note `pad_ms` after its
send, each host burst `host_ms` after the moment its first frame left the
host, both plus the capture ring the fake console reports in its `SPK #`
lines. The clock logs the scenario fits are generated the way the driver
produces them: positions that only move in 1 ms USB packets, stamped late by
a random scheduling delay, on a device clock 40 ppm off the host's.

So "the scenario measured 13.0 ms" here means it recovered a delay it was
never told, through the same fits, onsets and subtraction it uses on a board.
"""
from __future__ import annotations

import json
import re
import threading
import wave
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest

from crosspad_hil import fixtures
from crosspad_hil.analyze import wav_stats
from crosspad_hil.capture import CaptureDevice, CaptureResult
from crosspad_hil.cdc import CdcLink
from crosspad_hil.devices import Device, MidiPortInfo, Ports, SerialPortInfo, UsbMode
from crosspad_hil.errors import TIMEOUT, HilError
from crosspad_hil.midi import MidiRole
from crosspad_hil.scenarios import base, get, latency
from tests.fakes import FakeSerial

SR = 44100
CDC = SerialPortInfo(path="/dev/ttyACM0", vid=0x303A, pid=0x3456, serial="CP-1",
                     product="Crosspad", location="1-2.1")
CONSOLE = SerialPortInfo(path="/dev/ttyACM1", vid=0x0483, pid=0x5740, serial="STM001",
                         product="CrossPad MIDI+Serial", location="1-2.4:1.2")
ESP_MIDI = MidiPortInfo(name="Crosspad MIDI 1", rtmidi_out=0, rtmidi_in=0,
                        alsa_hw="hw:4,0,0", rawmidi="/dev/snd/midiC4D0")
STM_MIDI = MidiPortInfo(name="CrossPad MIDI+Serial 2", rtmidi_out=1, rtmidi_in=1,
                        alsa_hw="hw:5,0,0", rawmidi="/dev/snd/midiC5D0")
TOL_MS = 0.2


# --------------------------------------------------------------------------- the clock


class VClock:
    """Monotonic time that moves only when the scenario sleeps."""

    def __init__(self, t0: float = 5000.0) -> None:
        self.t = t0

    def now(self) -> float:
        return self.t

    def sleep_until(self, ctx: base.Context, target: float) -> None:
        ctx.check_cancelled()
        self.t = max(self.t, target)


# --------------------------------------------------------------------------- the board


class Board:
    """The fake CrossPad's truth: how late each path really is, and what goes wrong."""

    def __init__(self, clock: VClock) -> None:
        self.clock = clock
        self.pad_ms = 13.0             # STM note -> jack
        self.host_ms = 25.0            # frame out of the host -> jack
        self.cap_ring = 300            # capture ring, frames: reported AND applied
        self.drift = 40e-6             # device clock against the host's
        self.loop_gain = 0.3
        self.drop: set[int] = set()    # pad hits that never sound
        self.double: set[int] = set()  # pad hits that sound twice
        self.underrun_at: float | None = None   # s after the stream started
        self.after_underrun_ms = 9.0   # what an underrun leaves behind on the host path
        self.out_xruns = 0
        self.esp_echo_ms = 2.5
        self.spk_every_s = 1.0
        self.start_error: HilError | None = None
        self.notes: list[tuple[float, bool, int, int]] = []
        self.sysex: list[bytes] = []
        self.rec: FakeDuplex | None = None
        self.session: FakeSession | None = None
        self.t_stream: float | None = None
        self.midi_opened: list[tuple[MidiRole, bool]] = []
        self.midi_closed: list[MidiRole] = []
        self.esp_io: FakeMidi | None = None

    # -- seams the scenario is wired to
    def capture_session(self, device: Device, **kw: Any) -> FakeSession:
        self.session = FakeSession(self, device, **kw)
        return self.session

    def backend(self, p: Any) -> FakeBackend:
        return FakeBackend(self)

    def midi(self, device: Device, role: MidiRole, **kw: Any) -> FakeMidi:
        io = FakeMidi(self, role, listen=kw.get("listen", True))
        if role is MidiRole.ESP:
            self.esp_io = io
        return io

    def echoes(self, io: Any) -> FakeEchoes:
        return FakeEchoes(self)

    def note_ons(self) -> list[tuple[float, int, int]]:
        return [(t, note, vel) for t, on, note, vel in self.notes if on]


class FakeDuplex:
    """`duplex.DuplexRecording`'s shape over the board's true clocks."""

    PACKET = 44                         # frames per 1 ms USB packet: the position's step
    OUT_DELAY_S = 0.004                 # output frame 0 leaves this long after capture frame 0

    def __init__(self, board: Board, path: Path, rate: int) -> None:
        self.board = board
        self.path = Path(path)
        self.sample_rate = rate
        self.t0 = board.clock.now()
        self.spf = (1.0 / rate) * (1.0 + board.drift)
        self.buffer = 1764
        self.info = {"pcm": "fake", "playback_buffer": self.buffer, "playback_period": 441}
        self.bursts: list[tuple[int, list[int]]] = []
        self.t_end: float | None = None
        self.started_at = self.t0
        board.rec = self
        board.t_stream = self.t0

    @property
    def schedule_lead_frames(self) -> int:
        return self.buffer + 2 * 441 + self.sample_rate // 100

    def t_out(self, n: float) -> float:
        return self.t0 + self.OUT_DELAY_S + n * self.spf

    def out_position(self) -> int:
        now = self.board.clock.now()
        return int((now - self.t0 - self.OUT_DELAY_S) / self.spf) + self.buffer

    def schedule(self, frame: int, samples: list[int]) -> None:
        assert frame > self.out_position(), "a burst scheduled into frames already written"
        self.bursts.append((int(frame), list(samples)))

    def _points(self, offset: float, seed: int) -> list[tuple[int, float]]:
        rng = np.random.default_rng(seed)
        end = self.t_end if self.t_end is not None else self.board.clock.now()
        ts = np.arange(self.t0 + offset + 0.005, end, 0.002)
        frame = np.floor((ts - self.t0 - offset) / self.spf / self.PACKET) * self.PACKET
        stamp = ts + rng.exponential(0.0003, ts.size)
        return list(zip(frame.astype(int).tolist(), stamp.tolist(), strict=True))

    def capture_points(self) -> list[tuple[int, float]]:
        return self._points(0.0, 1)

    def playback_points(self) -> list[tuple[int, float]]:
        return self._points(self.OUT_DELAY_S, 2)

    def stop(self) -> None:
        if self.t_end is None:
            self.t_end = self.board.clock.now()
            self._render()

    def _render(self) -> None:
        b = self.board
        n = int((self.t_end - self.t0) / self.spf)
        rng = np.random.default_rng(3)
        sig = np.zeros(n)
        leg = b.cap_ring / self.sample_rate

        def put(t_sound: float, samples: Any) -> None:
            i = int(round((t_sound - self.t0) / self.spf))
            room = min(len(samples), n - i)
            if room > 0 and i >= 0:
                sig[i:i + room] += np.asarray(samples[:room], dtype=float) * b.loop_gain

        for k, (t, note, _vel) in enumerate(b.note_ons()):
            if k in b.drop:
                continue
            click = fixtures.click_samples(fixtures.click_seed(note - 36))
            put(t + b.pad_ms / 1000.0 + leg, click)
            if k in b.double:
                put(t + b.pad_ms / 1000.0 + leg + 0.06, click)
        for frame, samples in self.bursts:
            t_play = self.t_out(frame)
            extra = 0.0
            if b.underrun_at is not None and t_play >= self.t0 + b.underrun_at:
                extra = b.after_underrun_ms
            put(t_play + (b.host_ms + extra) / 1000.0 + leg, [s / 32767.0 for s in samples])
        # The loop lands on one channel; the other carries only the floor.
        left = rng.normal(0.0, 1e-4, n)
        right = rng.normal(0.0, 1e-4, n) + sig
        pcm = (np.clip(np.stack([left, right], axis=1), -1, 1) * 32767).astype("<i2")
        with wave.open(str(self.path), "wb") as w:
            w.setnchannels(2)
            w.setsampwidth(2)
            w.setframerate(self.sample_rate)
            w.writeframes(pcm.tobytes())

    def health(self) -> dict[str, Any]:
        flags = [f"{self.board.out_xruns} playback xrun(s)"] if self.board.out_xruns else []
        return {"ok": not flags, "captured_s": 1.0, "wall_s": 1.0, "overruns": 0,
                "out_xruns": self.board.out_xruns, "max_gap_ms": 2.0, "missed_s": 0.0,
                "exclusive": True, "flags": flags, "detail": "fake"}


class FakeBackend:
    name = "fake-duplex"

    def __init__(self, board: Board) -> None:
        self.board = board
        self.recording: FakeDuplex | None = None

    def start(self, cap: CaptureDevice, path: Path, *, seconds: float | None,
              sample_rate: int, channels: int) -> FakeDuplex:
        self.recording = FakeDuplex(self.board, path, sample_rate)
        return self.recording


class FakeSession:
    """CaptureSession's shape: the profile switch and the route, minus the hardware."""

    def __init__(self, board: Board, device: Device, *, preset: str = "", out: Any = None,
                 backend: Any = None, keep_mode: bool = False, **kw: Any) -> None:
        self.board = board
        self.device = device
        self.preset = preset
        self.path = Path(out)
        self.backend = backend
        self.keep_mode = keep_mode
        self.result: CaptureResult | None = None
        self.closed = False

    def start(self) -> dict:
        if self.board.start_error is not None:
            raise self.board.start_error
        cap = CaptureDevice(name="CrossPad", api="ALSA", alsa_id="Crosspad", sample_rate=SR)
        self.backend.start(cap, self.path, seconds=None, sample_rate=SR, channels=2)
        return {"device": self.device.id, "wav": str(self.path), "preset": self.preset,
                "backend": "fake-duplex", "sample_rate": SR, "channels": 2}

    def stop(self) -> CaptureResult:
        if self.result is None:
            rec = self.backend.recording
            rec.stop()
            st = wav_stats(self.path)
            self.result = CaptureResult(
                wav=str(self.path), seconds=st["seconds"], peak_dbfs=st["peak_dbfs"],
                rms_dbfs=st["rms_dbfs"], overruns=0, silent=False, sample_rate=SR,
                channels=2, frames=st["frames"], device=self.device.id, preset=self.preset,
                backend="fake-duplex", health=rec.health())
        return self.result

    def close(self) -> None:
        self.closed = True


class FakeMidi:
    """MidiIO's shape: the STM bridge takes notes, the ESP port answers SysEx echoes."""

    def __init__(self, board: Board, role: MidiRole, *, listen: bool) -> None:
        self.board = board
        self.role = role
        self.listen = listen
        self.rx: list[bytes] = []

    def open(self) -> None:
        self.board.midi_opened.append((self.role, self.listen))

    def close(self) -> None:
        self.board.midi_closed.append(self.role)

    def send_note(self, on: bool, note: int, vel: int = 100, channel: int = 0) -> None:
        assert self.role is MidiRole.STM
        self.board.notes.append((self.board.clock.now(), on, note, vel))
        self.board.clock.t += 0.00005          # the send itself takes a moment

    def send_sysex(self, frame: bytes) -> None:
        self.board.sysex.append(bytes(frame))
        if self.board.esp_io is not None:
            self.board.esp_io.rx.append(bytes(frame))

    def drain(self) -> int:
        n = len(self.rx)
        self.rx.clear()
        return n

    def receive(self, timeout_s: float = 1.0) -> bytes | None:
        return self.rx.pop(0) if self.rx else None

    def receive_note(self, timeout_s: float = 1.0) -> list[int] | None:
        return None


class FakeEchoes:
    """NoteEchoes' shape: the ESP sends each pad note back `esp_echo_ms` later."""

    def __init__(self, board: Board) -> None:
        self.board = board
        self.notes: list[tuple[float, list[int]]] = []

    def start(self) -> None:
        return None

    def stop(self) -> None:
        self.notes = [(t + self.board.esp_echo_ms / 1000.0, [0x90, note, vel])
                      for t, note, vel in self.board.note_ons()]


class FakeConsole:
    """The STM console: an `SPK #` line every `spk_every_s` once the stream runs."""

    def __init__(self, board: Board) -> None:
        self.board = board
        self.lines: list[str] = []

    def _advance(self) -> None:
        b = self.board
        if b.t_stream is None:
            return
        while True:
            t_j = b.t_stream + 0.8 + len(self.lines) * b.spk_every_s
            if t_j > b.clock.now():
                return
            under = int(b.underrun_at is not None and t_j - b.t_stream >= b.underrun_at)
            self.lines.append(
                f"I ({int(t_j * 1000) % 10_000_000}) usb_audio_bridge: SPK #{len(self.lines) + 1}:"
                f" len=528 taken=528 ring=880 under={under} refused=0 cap_ring={b.cap_ring}"
                " cap_drop=17 ring_min=600 rt_busy_max_us=900 idle1_pct=30"
                " spk_gap_max_us=4100 cap_under=0")

    @property
    def seq(self) -> int:
        self._advance()
        return len(self.lines)

    def read(self, since_seq: int | None = None, wait_ms: int = 0, match: str | None = None,
             limit: int = 2000) -> Any:
        self._advance()
        start = since_seq or 1
        sel = [(i + 1, line) for i, line in enumerate(self.lines)
               if i + 1 >= start and (match is None or re.search(match, line))][:limit]
        return SimpleNamespace(lines=sel, next_seq=sel[-1][0] + 1 if sel else start,
                               lines_lost=0)

    def close(self) -> None:
        return None


class CdcBoard(FakeSerial):
    """The CDC side: kits, apps, the Mixer desk, the STM register, the pad counters."""

    def __init__(self, board: Board, *, kits: dict[int, str] | None = None) -> None:
        super().__init__([], timeout=0.01)
        self.board = board
        self.kits = kits if kits is not None else {0: "Basic", 1: fixtures.CLICK_NAME}
        self.kit = 0
        self.app: str | None = None
        self.desk = 1
        self.desk_undo: list[int] = []
        self.desks = {1: "STREAM", 2: "BEATS", 3: "SAMPLING"}
        self.in2 = 0

    def _desk_line(self, ok: int = 1) -> str:
        slots = " ".join(f"{s}=f:{self.desks[s]}" if s in self.desks else f"{s}=e:"
                         for s in range(1, 7))
        return (f"MIXPRE: ok={ok} undo={int(bool(self.desk_undo))} "
                f"dir=/sdcard/crosspad/mixer {slots}")

    def _answer(self, cmd: str) -> str | list[str] | None:
        if cmd == "KIT_LIST":
            body = ",".join(f"{i}:{n}" for i, n in sorted(self.kits.items()))
            return [f"KITS: {body}", f"KITS: end count={len(self.kits)} current={self.kit}"]
        if cmd == "KIT_STATUS":
            return (f"KITSTATUS: current={self.kit} loading=0 pending=-1 "
                    f"name={self.kits.get(self.kit, '-')}")
        if cmd.startswith("KIT_LOAD "):
            self.kit = int(cmd.split()[1])
            return "OK"
        if cmd == "APP_LIST":
            return f"APPS: Sampler,Mixer running={self.app or '-'}"
        if cmd.startswith("APP_START "):
            self.app = cmd.split()[1]
            return "OK"
        if cmd == "APP_STOP":
            self.app = None
            return "OK"
        if cmd == "STM_REG 033C 1":
            return "REG 033C: 0F"
        if cmd == "PAD_STATS_RESET":
            return "OK"
        if cmd == "PAD_STATS":
            n = len(self.board.note_ons())
            return (f"PADSTATS: press=0 release=0 inj_drop=0 played={n} last=0:0 freeslots=10 "
                    f"stm={n} at=0 rxmsg={n} uart_ovf=0 sbuf_drop=0 usb_tx={n} usb_drop=0")
        if cmd == "SMPL_LAST":
            return ("SMPLLAST: stm_presses=12 stm_note=36 stm_pad=0 engine_total=12 events= "
                    "opens=0/0(-) seeks=0/0(-) reads=0/0(-) closes=0/0(-) defer=0/0 evict=0 "
                    "lat_us last=700 avg=650 max=1500 n=12 stream_staged=0 stream_fb=0 "
                    "stream_kbps=0 stream_read_max_us=0 late_max_ms=0 late_lost=0")
        if cmd == "INGAIN":
            return f"INGAIN: IN1=24 IN2={self.in2}"
        if cmd.startswith("INGAIN 1 "):
            self.in2 = int(cmd.split()[2])
            return f"INGAIN: IN1=24 IN2={self.in2}"
        if cmd == "MIX_PRESET":
            return self._desk_line()
        if cmd.startswith("MIX_PRESET LOAD "):
            self.desk_undo.append(self.desk)
            self.desk = int(cmd.split()[2])
            return self._desk_line()
        if cmd == "MIX_PRESET UNDO":
            if not self.desk_undo:
                return self._desk_line(ok=0)
            self.desk = self.desk_undo.pop()
            return self._desk_line()
        return None

    def write(self, data: bytes) -> int:
        for raw in data.decode("utf-8", "replace").split("\n"):
            cmd = raw.strip()
            if not cmd:
                continue
            self.written.append(cmd)
            reply = self._answer(cmd)
            if reply is None:
                self._rx.append((cmd + "\r\n").encode())     # unknown verb: echoed
            else:
                self.feed([reply] if isinstance(reply, str) else reply)
        return len(data)


@dataclass
class FakeContext(base.Context):
    cdc_board: CdcBoard = None  # type: ignore[assignment]
    console_fake: FakeConsole = None  # type: ignore[assignment]
    consoles_opened: int = 0

    def open_cdc(self) -> CdcLink:
        fake = self.cdc_board
        link = CdcLink(self.device.ports.cdc.path, serial_factory=lambda path, **kw: fake)
        link.open()
        self.opened.append(link)
        return link

    def open_console(self, reset: bool = False) -> Any:
        self.consoles_opened += 1
        return self.console_fake


def make_device(*, stm_midi: MidiPortInfo | None = STM_MIDI) -> Device:
    return Device(id="dev_ab12", serial="CP-1", usb_mode=UsbMode.DEFAULT,
                  ports=Ports(cdc=CDC, console=CONSOLE, esp_midi=ESP_MIDI, stm_midi=stm_midi))


@pytest.fixture
def bench(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Any:
    clock = VClock()
    board = Board(clock)
    monkeypatch.setattr(latency, "_now", clock.now)
    monkeypatch.setattr(latency, "_sleep_until", clock.sleep_until)
    monkeypatch.setattr(latency, "CaptureSession", board.capture_session)
    monkeypatch.setattr(latency, "make_backend", board.backend)
    monkeypatch.setattr(latency, "MidiIO", board.midi)
    monkeypatch.setattr(latency, "NoteEchoes", board.echoes)

    def make(*, kits: dict[int, str] | None = None,
             stm_midi: MidiPortInfo | None = STM_MIDI) -> tuple[FakeContext, Board, CdcBoard]:
        cdc = CdcBoard(board, kits=kits)
        ctx = FakeContext(device=make_device(stm_midi=stm_midi), workdir=tmp_path,
                          cancelled=threading.Event(), log=lambda s: None, cdc_board=cdc,
                          console_fake=FakeConsole(board))
        return ctx, board, cdc

    return make


def run(ctx: FakeContext, **params: Any) -> base.Report:
    kw: dict[str, Any] = {"hits": 12, "seed": 7, "echo_probes": 0, "min_gap_ms": 300.0,
                          "max_gap_ms": 420.0}
    kw.update(params)
    return latency.SCENARIO.run(ctx, latency.Params(**kw), base.Progress())


def headline(rep: base.Report, path: str) -> dict[str, Any]:
    body = rep.data["paths"][path]
    return body[body["headline"]]


# --------------------------------------------------------------------------- the answer


def test_registered_and_defaults() -> None:
    assert get("latency") is latency.SCENARIO
    p = latency.Params()
    assert p.paths == ["pad", "host"] and p.hits == 60
    assert (p.min_gap_ms, p.max_gap_ms, p.window_ms) == (300.0, 700.0, 200.0)
    assert p.kit == "HIL CLICK" and p.preset == "headphone" and p.mixer_desk == "BEATS"
    assert p.max_pad_ms is None and p.max_host_ms is None and p.gate_stat == "p50"
    assert p.console and p.exclusive_card and not p.allow_underrun


def test_known_delays_come_back_within_0_2_ms(bench: Any) -> None:
    ctx, board, cdc = bench()
    rep = run(ctx)
    assert rep.passed, rep.summary
    assert rep.exit_code == 0
    leg_ms = board.cap_ring / SR * 1000.0

    pad = rep.data["paths"]["pad"]
    assert pad["headline"] == "jack_ms" and pad["metric"] == "pad_jack_ms"
    assert pad["measured"] == 12 and pad["missing"] == [] and pad["double"] == []
    assert pad["capture_leg"]["ms"] == pytest.approx(leg_ms, abs=1e-3)
    for stat in ("p50", "p95", "max", "min"):
        assert pad["jack_ms"][stat] == pytest.approx(board.pad_ms, abs=TOL_MS), stat
        assert pad["uac_in_ms"][stat] == pytest.approx(board.pad_ms + leg_ms, abs=TOL_MS)

    host = rep.data["paths"]["host"]
    assert host["measured"] == 12
    for stat in ("p50", "p95", "max", "min"):
        assert host["jack_ms"][stat] == pytest.approx(board.host_ms, abs=TOL_MS), stat

    # the stream's clock was fitted at the device's rate, not the nominal one
    assert rep.data["clock"]["capture"]["rate_hz"] == pytest.approx(SR / (1 + board.drift),
                                                                    abs=0.05)
    assert "PASS" in rep.summary and "pad->jack p50 13.0" in rep.summary
    json.dumps(rep.data)                       # report.json must be able to hold it


def test_the_notes_and_bursts_are_what_was_asked(bench: Any) -> None:
    ctx, board, cdc = bench()
    rep = run(ctx, pads=[0, 3], vel=90, gate_ms=25.0)
    assert rep.passed, rep.summary
    ons = board.note_ons()
    assert [n for _t, n, _v in ons] == [36, 39] * 6
    assert {v for _t, _n, v in ons} == {90}
    offs = [(t, n) for t, on, n, _v in board.notes if not on]
    assert len(offs) == 12
    assert all(off_t - on_t == pytest.approx(0.025, abs=1e-3)
               for (on_t, _n, _v), (off_t, _n2) in zip(ons, offs, strict=True))
    # random gaps inside the bounds asked for, and reproducible from the seed
    gaps = np.diff([t for t, _n, _v in ons])
    assert gaps.min() >= 0.300 - 1e-3 and gaps.max() <= 0.420 + 1e-3
    assert len(set(np.round(gaps, 4))) > 6
    assert rep.data["config"]["seed"] == 7
    assert len(board.rec.bursts) == 12
    # the STM port is a sender only; every port is closed before the switch back
    assert (MidiRole.STM, False) in board.midi_opened
    assert MidiRole.STM in board.midi_closed and MidiRole.ESP in board.midi_closed


def test_the_board_is_put_back(bench: Any) -> None:
    ctx, board, cdc = bench()
    rep = run(ctx)
    assert rep.passed, rep.summary
    w = cdc.written
    # the run's own state: the click kit, the Sampler, the BEATS desk, counters zeroed
    assert "KIT_LOAD 1" in w and "APP_START Sampler" in w and "MIX_PRESET LOAD 2" in w
    assert w.index("PAD_STATS_RESET") < w.index("MIX_PRESET LOAD 2")
    # ...and afterwards, once the counters are read, the board as it was found
    tail = w[w.index("SMPL_LAST"):]
    assert [c for c in tail if c in ("MIX_PRESET UNDO", "KIT_LOAD 0", "APP_STOP")] == [
        "MIX_PRESET UNDO", "KIT_LOAD 0", "APP_STOP"]
    assert cdc.kit == 0 and cdc.app is None and cdc.desk == 1
    assert rep.data["left"] == {"desk": "restored", "kit": "restored", "app": "restored"}
    assert rep.data["prologue"]["found"] == {"kit": 0, "app": None}
    assert rep.data["device_counters"]["pad_stats"]["stm"] == 12
    assert rep.data["device_counters"]["smpl_last"]["lat_us_avg"] == 650
    assert board.session.closed or board.session.result is not None


def test_the_kit_found_is_put_back_by_name_after_a_rescan(bench: Any) -> None:
    # A kit id is a slot in the card's scan. Provisioning pushes a folder and
    # reboots; a new folder can land ahead of the kits already there, so the id
    # read before the push names another kit afterwards.
    ctx, board, cdc = bench(kits={0: "Basic", 1: "Zed", 2: fixtures.CLICK_NAME})
    cdc.kit = 1
    answer = cdc._answer

    def rescanned(cmd: str) -> Any:
        reply = answer(cmd)
        if cmd == "KIT_STATUS" and cdc.kits[1] == "Zed":
            cdc.kits = {0: "Basic", 1: fixtures.CLICK_NAME, 2: "Zed"}
            cdc.kit = 2
        return reply

    cdc._answer = rescanned  # type: ignore[method-assign]
    rep = run(ctx, paths=["pad"])
    assert rep.passed, rep.summary
    assert rep.data["prologue"]["found"]["kit"] == 1
    assert rep.data["prologue"]["found_kit_name"] == "Zed"
    assert cdc.kits[cdc.kit] == "Zed" and rep.data["left"]["kit"] == "restored"


def test_a_refused_desk_undo_is_not_reported_as_restored(bench: Any) -> None:
    # The undo snapshot is RAM, the loaded desk is persisted: a board that reset
    # mid-run answers UNDO with ok=0 and keeps BEATS. That must not read "restored".
    ctx, board, cdc = bench()
    answer = cdc._answer

    def undo_lost(cmd: str) -> Any:
        if cmd == "MIX_PRESET UNDO":
            return cdc._desk_line(ok=0)
        return answer(cmd)

    cdc._answer = undo_lost  # type: ignore[method-assign]
    rep = run(ctx, paths=["host"])
    assert "MIX_PRESET UNDO" in cdc.written and cdc.desk == 2
    assert rep.data["left"]["desk"].startswith("ENV: MIX_PRESET UNDO refused")
    assert "board NOT put back: desk ENV" in rep.summary
    assert any("board NOT put back" in n for n in rep.data["notes"])


def test_artifacts_and_hit_rows(bench: Any, tmp_path: Path) -> None:
    ctx, board, cdc = bench()
    rep = run(ctx)
    roles = {a.role: Path(a.path) for a in rep.artifacts}
    assert {"capture", "hits", "clock", "analysis", "console"} <= set(roles)
    rows = roles["hits"].read_text().splitlines()
    assert rows[0].startswith("path,k,t_ref,expected_frame,onset_frame,status,uac_in_ms,jack_ms")
    assert len(rows) == 1 + 24
    assert json.loads(roles["analysis"].read_text())["paths"]["host"]["measured"] == 12


# --------------------------------------------------------------------------- what goes wrong


def test_a_hit_that_never_sounds_fails(bench: Any) -> None:
    ctx, board, cdc = bench()
    board.drop = {3}
    rep = run(ctx, paths=["pad"])
    assert not rep.passed and rep.exit_code == 1
    pad = rep.data["paths"]["pad"]
    assert pad["missing"] == [3] and pad["measured"] == 11
    assert pad["jack_ms"]["p50"] == pytest.approx(board.pad_ms, abs=TOL_MS)
    assert any("1/12 hits never sounded (k=[3])" in f for f in rep.data["failures"])
    # the device's own count says the note got to the ESP: not a transport loss
    assert "PAD_STATS stm=12 of 12" in rep.summary


def test_max_missing_tolerates_what_it_is_told_to(bench: Any) -> None:
    ctx, board, cdc = bench()
    board.drop = {3}
    rep = run(ctx, paths=["pad"], max_missing=1)
    assert rep.passed, rep.summary


def test_a_hit_that_sounds_twice_fails_and_is_not_scored(bench: Any) -> None:
    ctx, board, cdc = bench()
    board.double = {5}
    rep = run(ctx, paths=["pad"])
    assert not rep.passed and rep.exit_code == 1
    pad = rep.data["paths"]["pad"]
    assert pad["double"] == [5] and pad["measured"] == 11
    assert pad["jack_ms"]["max"] == pytest.approx(board.pad_ms, abs=TOL_MS)
    assert any("sounded twice (k=[5])" in f for f in rep.data["failures"])


def test_nothing_audible_is_the_bench_not_the_firmware(bench: Any) -> None:
    ctx, board, cdc = bench()
    board.drop = set(range(12))
    rep = run(ctx, paths=["pad"])
    assert not rep.passed and rep.exit_code == 2
    assert "none of 12 hits made a sound" in rep.summary
    assert rep.data["left"]["kit"] == "restored"


def test_an_underrun_mid_run_fails_and_splits_the_numbers(bench: Any) -> None:
    ctx, board, cdc = bench()
    board.underrun_at = 1.5 + 0.36 * 8         # mid host phase
    rep = run(ctx, paths=["host"], hits=16)
    assert not rep.passed and rep.exit_code == 1
    assert rep.data["spk"]["delta"]["under"] == 1
    assert any("device glitch during the measurement: under+1" in f
               for f in rep.data["failures"])
    seg = rep.data["paths"]["host"]["segments"]
    assert seg["before"]["n"] >= 3 and seg["after"]["n"] >= 3
    assert seg["before"]["p50"] == pytest.approx(board.host_ms, abs=TOL_MS)
    assert seg["after"]["p50"] == pytest.approx(board.host_ms + board.after_underrun_ms,
                                                abs=TOL_MS)
    assert seg["delta_p50_ms"] == pytest.approx(board.after_underrun_ms, abs=2 * TOL_MS)
    assert seg["before"]["n"] + seg["after"]["n"] + seg["ambiguous"] == 16
    assert seg["guard_s"] == rep.data["spk"]["event_guard_s"] >= latency.SPK_GUARD_S


def test_allow_underrun_reports_instead_of_failing(bench: Any) -> None:
    ctx, board, cdc = bench()
    board.underrun_at = 1.5 + 0.36 * 8
    rep = run(ctx, paths=["host"], hits=16, allow_underrun=True)
    assert rep.passed, rep.summary
    assert any("under+1" in n for n in rep.data["notes"])
    assert "segments" in rep.data["paths"]["host"]


def test_a_host_xrun_is_not_charged_to_the_board(bench: Any) -> None:
    ctx, board, cdc = bench()
    board.out_xruns = 1
    rep = run(ctx, paths=["host"])
    assert not rep.passed and rep.exit_code == 2
    assert "capture not measured" in rep.summary and "playback xrun" in rep.summary


# --------------------------------------------------------------------------- knobs


def test_thresholds_decide(bench: Any) -> None:
    ctx, board, cdc = bench()
    rep = run(ctx, max_pad_ms=12.0, max_host_ms=30.0)
    assert not rep.passed and rep.exit_code == 1
    assert any(f.startswith("pad: jack p50 13.0") and f.endswith("> 12.00 ms")
               for f in rep.data["failures"])
    assert not any(f.startswith("host:") for f in rep.data["failures"])

    ctx, board, cdc = bench()
    rep = run(ctx, max_pad_ms=14.0, max_host_ms=26.0, gate_stat="max")
    assert rep.passed, rep.summary


def test_without_the_console_nothing_is_subtracted(bench: Any) -> None:
    ctx, board, cdc = bench()
    rep = run(ctx, console=False)
    assert rep.passed, rep.summary
    assert ctx.consoles_opened == 0
    pad = rep.data["paths"]["pad"]
    assert pad["headline"] == "uac_in_ms" and pad["jack_ms"] is None
    assert pad["uac_in_ms"]["p50"] == pytest.approx(board.pad_ms + 300 / SR * 1000, abs=TOL_MS)
    assert rep.data["spk"] is None
    assert "pad->UAC IN" in rep.summary


def test_the_esp_echo_and_the_physical_pad_model(bench: Any) -> None:
    ctx, board, cdc = bench()
    rep = run(ctx, paths=["pad"])
    pad = rep.data["paths"]["pad"]
    assert pad["esp_echo_ms"]["p50"] == pytest.approx(board.esp_echo_ms, abs=0.01)
    assert pad["esp_echo_missing"] == 0
    est = pad["physical_est_ms"]
    assert est["press_debounce_ms"] == 15 and est["kind"].startswith("W")
    # 13.0 measured - 0.5 USB OUT + 0.625 scan + (15 - 0.125) window
    assert est["p50"] == pytest.approx(28.0, abs=TOL_MS)
    assert est["range"][0] < est["p50"] < est["range"][1]


def test_echo_probes_go_down_the_bridge(bench: Any) -> None:
    ctx, board, cdc = bench()
    rep = run(ctx, paths=["pad"], echo_probes=3)
    assert rep.passed, rep.summary
    assert len(board.sysex) == 3
    assert rep.data["echo_rtt"]["n"] == 3


def test_host_only_leaves_the_kit_and_app_alone(bench: Any) -> None:
    ctx, board, cdc = bench()
    rep = run(ctx, paths=["host"])
    assert rep.passed, rep.summary
    assert not any(c.startswith(("KIT_LOAD", "APP_START", "PAD_STATS")) for c in cdc.written)
    assert "pad" not in rep.data["paths"]
    assert rep.data["left"] == {"desk": "restored"}


def test_in2_is_pinned_for_the_run_and_put_back(bench: Any) -> None:
    ctx, board, cdc = bench()
    rep = run(ctx, paths=["host"], in2_gain=24)
    assert rep.passed, rep.summary
    assert "INGAIN 1 24" in cdc.written and cdc.written[-1] == "INGAIN 1 0"
    assert cdc.in2 == 0 and rep.data["left"]["in2"] == "restored"
    assert rep.data["ingain_during"] == {"IN1": 24, "IN2": 24}


def test_in2_is_not_touched_on_the_speaker_loop(bench: Any) -> None:
    ctx, board, cdc = bench()
    rep = run(ctx, paths=["host"], in2_gain=24, preset="speaker")
    assert rep.passed, rep.summary
    assert not any(c.startswith("INGAIN") for c in cdc.written)


def test_desk_left_alone_when_asked(bench: Any) -> None:
    ctx, board, cdc = bench()
    rep = run(ctx, paths=["host"], mixer_desk="")
    assert rep.passed, rep.summary
    assert not any(c.startswith("MIX_PRESET") for c in cdc.written)


# --------------------------------------------------------------------------- the bench


def test_missing_stm_bridge_is_an_environment_error(bench: Any) -> None:
    ctx, board, cdc = bench(stm_midi=None)
    rep = run(ctx)
    assert rep.exit_code == 2 and "STM bridge" in rep.summary
    assert cdc.written == []


def test_missing_kit_without_provisioning_is_an_environment_error(bench: Any) -> None:
    ctx, board, cdc = bench(kits={0: "Basic"})
    rep = run(ctx, provision=False)
    assert rep.exit_code == 2 and "HIL CLICK" in rep.summary
    assert board.session is None


def test_a_capture_that_never_starts_still_puts_the_board_back(bench: Any) -> None:
    ctx, board, cdc = bench()
    board.start_error = HilError(TIMEOUT, "no CrossPad UAC2 capture card", hint="usb-mode")
    rep = run(ctx)
    assert rep.exit_code == 2 and "could not start the duplex stream" in rep.summary
    assert board.notes == []
    assert cdc.kit == 0 and cdc.app is None and cdc.desk == 1


def test_bad_params_are_refused() -> None:
    ctx = None
    for kw in ({"paths": ["pad", "nope"]}, {"hits": 0}, {"min_gap_ms": 100.0},
               {"gate_stat": "p99"}, {"preset": "asis"}, {"pads": [16]}):
        with pytest.raises(HilError) as e:
            latency.SCENARIO.run(ctx, latency.Params(**kw), base.Progress())  # type: ignore[arg-type]
        assert e.value.code == "BAD_ARGS", kw


def test_cli_flags_map_onto_params() -> None:
    import argparse

    parser = argparse.ArgumentParser()
    base.params_to_argparse(latency.Params, parser)
    ns = parser.parse_args(["--paths", "pad", "--hits", "120", "--max-pad-ms", "15.5",
                            "--no-console", "--pads", "0,5", "--seed", "3"])
    p = base.argparse_to_params(latency.Params, ns)
    assert p.paths == ["pad"] and p.hits == 120 and p.max_pad_ms == 15.5
    assert p.console is False and p.pads == [0, 5] and p.seed == 3 and p.max_host_ms is None
