"""Scenario `speedtest` against a fake board — no MIDI hardware, no device.

The fake is the whole trigger chain in miniature: pad hits arrive as SysEx on a
stand-in for the STM bridge, a device model decides which of them it can keep up
with, and the CDC endpoint answers PAD_STATS from that model. So the loss the
scenario reports is loss the fake device actually inflicted, and the rate sweep
finds the limit rather than being told it.
"""
from __future__ import annotations

import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest

from crosspad_hil import stim
from crosspad_hil.cdc import CdcLink
from crosspad_hil.devices import Device, MidiPortInfo, Ports, SerialPortInfo, UsbMode
from crosspad_hil.midi import MidiRole
from crosspad_hil.scenarios import base, get, speedtest
from tests.fakes import FakeSerial

CDC = SerialPortInfo(path="/dev/ttyACM0", vid=0x303A, pid=0x3456, serial="CP-1",
                     product="Crosspad", location="1-2.1")
STM_MIDI = MidiPortInfo(name="CrossPad MIDI+Serial 2", rtmidi_out=1, rtmidi_in=1,
                        alsa_hw="hw:5,0,0", rawmidi="/dev/snd/midiC5D0")
PRESS_SUB = 0x07
RELEASE_SUB = 0x08
FREE_SLOTS = 7


def make_device(*, stm_midi: MidiPortInfo | None = STM_MIDI) -> Device:
    return Device(id="dev_ab12", serial="CP-1", usb_mode=UsbMode.DEFAULT,
                  ports=Ports(cdc=CDC, stm_midi=stm_midi))


# --------------------------------------------------------------------------- the fake board


class Engine:
    """What the firmware would count.

    ``max_rate`` is the only misbehaviour modelled: a press arriving sooner than
    the device can take one is lost before it is counted, which is exactly the
    sent-vs-accepted gap the scenario calls transport loss.
    """

    def __init__(self, *, max_rate: float | None = None) -> None:
        self.max_rate = max_rate
        self.press = 0
        self.release = 0
        self.played = 0
        self.seen = 0
        self._last = -1e9

    def on_press(self) -> None:
        self.seen += 1
        now = time.monotonic()
        if self.max_rate is not None and now - self._last < 1.0 / self.max_rate:
            return
        self._last = now
        self.press += 1
        self.played += 1

    def on_release(self) -> None:
        self.release += 1

    def reset(self) -> None:
        self.press = self.release = self.played = 0


class SysexPort:
    """MidiIO's shape for the pad transport: frames in, counters moved."""

    def __init__(self, engine: Engine) -> None:
        self._engine = engine
        self.frames: list[bytes] = []
        self.opened = False
        self.closed = False

    def open(self) -> None:
        self.opened = True

    def close(self) -> None:
        self.closed = True

    def send_sysex(self, frame: bytes) -> None:
        self.frames.append(bytes(frame))
        if frame[3] == PRESS_SUB:
            self._engine.on_press()
        elif frame[3] == RELEASE_SUB:
            self._engine.on_release()


class CdcBoard(FakeSerial):
    """The CDC endpoint: kit, app registry and the pad counters the sweep reads."""

    def __init__(self, engine: Engine, *, answer_pad_stats: bool = True) -> None:
        super().__init__()
        self.engine = engine
        self.answer_pad_stats = answer_pad_stats
        self.running: str | None = None
        self.apps = ["Sampler", "Mixer"]
        self.current_kit = -1
        self.kit_error: str | None = None

    def write(self, data: bytes) -> int:
        for raw in data.decode("utf-8", "replace").split("\n"):
            cmd = raw.strip()
            if not cmd:
                continue
            self.written.append(cmd)
            if cmd.startswith("KIT_LOAD "):
                if self.kit_error is not None:
                    self.feed([self.kit_error])
                    continue
                self.current_kit = int(cmd.split(" ", 1)[1])
                self.feed(["OK"])
            elif cmd == "KIT_STATUS":
                self.feed([f"KITSTATUS: current={self.current_kit} loading=0 "
                           f"pending=-1 name=Basic"])
            elif cmd == "APP_LIST":
                self.feed([f"APPS: {','.join(self.apps)} running={self.running or '-'}"])
            elif cmd.startswith("APP_START "):
                self.running = cmd.split(" ", 1)[1]
                self.feed(["OK"])
            elif cmd == "PAD_STATS_RESET":
                self.engine.reset()
                self.feed(["OK"])
            elif cmd == "PAD_STATS" and self.answer_pad_stats:
                e = self.engine
                self.feed([f"PADSTATS: press={e.press} release={e.release} "
                           f"played={e.played} freeslots={FREE_SLOTS}"])
            elif cmd == "CDC_STATS":
                self.feed(["CDCSTATS: rx=0 cmds=0 drop=0"])
            else:
                # hil_control_handle_cdc() matched nothing: the device echoes it back.
                self._rx.append((cmd + "\r\n").encode())
        return len(data)


@dataclass
class FakeContext(base.Context):
    cdc_fake: CdcBoard = None  # type: ignore[assignment]
    cdc_opens: int = 0

    def open_cdc(self) -> CdcLink:
        self.cdc_opens += 1
        fake = self.cdc_fake
        link = CdcLink(self.device.ports.cdc.path, serial_factory=lambda path, **kw: fake)
        link.open()
        self.opened.append(link)
        return link


class Bench:
    def __init__(self, *, max_rate: float | None = None,
                 answer_pad_stats: bool = True) -> None:
        self.engine = Engine(max_rate=max_rate)
        self.port = SysexPort(self.engine)
        self.cdc = CdcBoard(self.engine, answer_pad_stats=answer_pad_stats)

    def midi_factory(self, device: Device, role: MidiRole) -> SysexPort:
        assert role is MidiRole.STM      # never the ESP port, never CDC
        return self.port


def make_ctx(tmp_path: Path, bench: Bench, *,
             stm_midi: MidiPortInfo | None = STM_MIDI) -> FakeContext:
    return FakeContext(device=make_device(stm_midi=stm_midi), workdir=tmp_path,
                       cancelled=threading.Event(), log=lambda s: None, cdc_fake=bench.cdc)


@pytest.fixture
def wire(monkeypatch: pytest.MonkeyPatch):
    """Let the real stim.sweep run, with the fake MIDI port under it.

    `open_transport`'s midi_factory is a default argument, so it is bound at
    import time and monkeypatching stim.MidiIO would not reach it; the injection
    point is the sweep call the scenario makes.
    """
    def install(bench: Bench) -> None:
        real = stim.sweep

        def wrapper(device: Device, **kw: Any) -> dict:
            kw.setdefault("midi_factory", bench.midi_factory)
            return real(device, **kw)

        monkeypatch.setattr(speedtest, "sweep", wrapper)
    return install


def run(ctx: FakeContext, **params: Any) -> base.Report:
    kw: dict[str, Any] = {"rates": [8], "duration": 0.3, "settle": 0.0, "padsets": "0"}
    kw.update(params)
    return speedtest.SCENARIO.run(ctx, speedtest.Params(**kw), base.Progress())


# --------------------------------------------------------------------------- tests


def test_registered_and_defaults() -> None:
    assert get("speedtest") is speedtest.SCENARIO
    p = speedtest.Params()
    assert p.rates == [4, 8, 12, 16, 24, 32, 48, 64, 96, 128]
    assert p.padsets == "0;0-3;0-15"
    assert (p.kit, p.duration, p.settle, p.vel, p.gate_ms) == (0, 5.0, 1.5, 100, 25.0)
    assert (p.chord, p.loss_threshold, p.full_sweep, p.repeat) == (False, 1.0, False, 1)
    assert p.start_app is True and p.require_rate is None


def test_parse_padsets_and_limit_table() -> None:
    assert speedtest.parse_padsets("0;0-3;0-15") == [[0], [0, 1, 2, 3], list(range(16))]
    assert speedtest.parse_padsets(" 0 ; 1,2 ") == [[0], [1, 2]]
    # the lowest limit across repeats wins: a rate that failed once is not sustained
    assert speedtest.limit_table([
        {"padset": "1pad", "limit_rate_hz": 64.0},
        {"padset": "1pad", "limit_rate_hz": 32.0},
        {"padset": "4pad", "limit_rate_hz": None},
    ]) == {"1pad": 32.0, "4pad": None}


def test_clean_sweep_reports_the_ceiling(tmp_path: Path, wire) -> None:
    bench = Bench()
    wire(bench)
    ctx = make_ctx(tmp_path, bench)
    rep = run(ctx)
    assert rep.passed and rep.exit_code == 0
    assert rep.data["limit_hits_per_s"] == {"1pad": None}
    assert "no loss up to 8 hits/s" in rep.summary

    row = rep.data["rows"][0]
    assert row["transport"] == "sysex_stm" and row["rep"] == 0
    assert row["loss"]["sent"] == row["loss"]["accepted"] == row["loss"]["played"]
    assert row["loss"]["total_pct"] == 0.0
    assert row["freeslots"] == FREE_SLOTS

    # the hits took the bridge, not the 64-deep CDC command queue
    assert bench.port.frames and not any(c.startswith("PAD_PRESS") for c in bench.cdc.written)
    assert bench.port.closed
    assert "KIT_LOAD 0" in bench.cdc.written and "APP_START Sampler" in bench.cdc.written
    assert Path([a.path for a in rep.artifacts if a.role == "analysis"][0]).exists()


def test_sweep_finds_the_limit_and_still_passes(tmp_path: Path, wire) -> None:
    """hil_speedtest reported where the device stops keeping up; it never judged it."""
    bench = Bench(max_rate=20.0)
    wire(bench)
    ctx = make_ctx(tmp_path, bench)
    rep = run(ctx, rates=[8, 40])
    assert rep.passed and rep.exit_code == 0
    assert rep.data["limit_hits_per_s"] == {"1pad": 40.0}
    assert "1pad: 40 hits/s" in rep.summary
    rows = rep.data["rows"]
    assert [r["rate_hz"] for r in rows] == [8.0, 40.0]
    assert rows[0]["loss"]["total_pct"] == 0.0
    assert rows[1]["loss"]["total_pct"] > 1.0
    # the loss is all in front of the firmware: what it accepted, it played
    assert rows[1]["loss"]["engine_pct"] == 0.0


def test_the_sweep_stops_at_the_limit_unless_full_sweep(tmp_path: Path, wire) -> None:
    bench = Bench(max_rate=20.0)
    wire(bench)
    ctx = make_ctx(tmp_path, bench)
    rep = run(ctx, rates=[40, 8])
    # 40 loses first, so 8 is never reached: the shape past the limit costs
    # minutes and usually says the same thing
    assert [r["rate_hz"] for r in rep.data["rows"]] == [40.0]

    bench = Bench(max_rate=20.0)
    wire(bench)
    ctx = make_ctx(tmp_path, bench)
    rep = run(ctx, rates=[40, 8], full_sweep=True)
    assert [r["rate_hz"] for r in rep.data["rows"]] == [40.0, 8.0]
    assert rep.data["limit_hits_per_s"] == {"1pad": 40.0}


def test_require_rate_turns_a_limit_into_a_failure(tmp_path: Path, wire) -> None:
    bench = Bench(max_rate=20.0)
    wire(bench)
    ctx = make_ctx(tmp_path, bench)
    rep = run(ctx, rates=[8, 40], require_rate=64)
    assert not rep.passed and rep.exit_code == 1
    assert "below the required 64 hits/s" in rep.summary
    assert "1pad limited at 40 hits/s" in rep.summary


def test_counters_that_stop_answering_fail_the_run(tmp_path: Path, wire) -> None:
    """A sweep whose counters went quiet measured nothing — the old script still
    exited 0 there, which is the one place its verdict is not preserved."""
    bench = Bench(answer_pad_stats=False)
    wire(bench)
    ctx = make_ctx(tmp_path, bench)
    rep = run(ctx)
    assert not rep.passed and rep.exit_code == 1
    assert "stopped answering mid-sweep" in rep.summary
    assert rep.data["error"]["code"] == "UNKNOWN_VERB"


def test_chord_mode_hits_every_pad_of_the_set(tmp_path: Path, wire) -> None:
    bench = Bench()
    wire(bench)
    ctx = make_ctx(tmp_path, bench)
    rep = run(ctx, padsets="0-1", chord=True)
    assert rep.passed, rep.summary
    assert rep.data["mode"] == "chord"
    assert rep.data["padsets"] == [[0, 1]]
    row = rep.data["rows"][0]
    assert row["padset"] == "2pad-chord"
    # one scheduled beat is two presses when the set is struck as a chord
    assert row["loss"]["sent"] == 2 * int(0.3 * 8)
    assert row["loss"]["accepted"] == row["loss"]["sent"]


def test_repeat_runs_the_whole_sweep_again(tmp_path: Path, wire) -> None:
    bench = Bench()
    wire(bench)
    ctx = make_ctx(tmp_path, bench)
    rep = run(ctx, repeat=2)
    assert rep.passed
    assert [r["rep"] for r in rep.data["rows"]] == [0, 1]
    assert rep.data["limit_hits_per_s"] == {"1pad": None}


def test_no_start_app_leaves_the_launcher_alone(tmp_path: Path, wire) -> None:
    """The sampler is the system instrument: pads sound with no app open."""
    bench = Bench()
    wire(bench)
    ctx = make_ctx(tmp_path, bench)
    rep = run(ctx, start_app=False)
    assert rep.passed
    assert rep.data["app_open"] is False
    assert "KIT_LOAD 0" in bench.cdc.written
    assert not any(c.startswith("APP_START") for c in bench.cdc.written)


def test_missing_stm_bridge_is_an_environment_error(tmp_path: Path, wire) -> None:
    bench = Bench()
    wire(bench)
    ctx = make_ctx(tmp_path, bench, stm_midi=None)
    rep = run(ctx)
    assert rep.exit_code == 2 and not rep.passed
    assert "STM bridge" in rep.summary
    assert ctx.cdc_opens == 0


def test_kit_load_failure_is_an_environment_error(tmp_path: Path, wire) -> None:
    bench = Bench()
    wire(bench)
    ctx = make_ctx(tmp_path, bench)
    bench.cdc.kit_error = "ERR bad kit id"
    rep = run(ctx)
    assert rep.exit_code == 2 and not rep.passed
    assert "KIT_LOAD 0 failed" in rep.summary
    assert bench.port.frames == []


def test_empty_rates_or_padsets_are_environment_errors(tmp_path: Path, wire) -> None:
    bench = Bench()
    wire(bench)
    ctx = make_ctx(tmp_path, bench)
    assert run(ctx, rates=[]).exit_code == 2
    ctx = make_ctx(tmp_path, bench)
    assert run(ctx, padsets=";").exit_code == 2
    ctx = make_ctx(tmp_path, bench)
    rep = run(ctx, padsets="0-99")
    assert rep.exit_code == 2 and "pads must be" in rep.summary
