"""Scenario `usb_mode_cycle` with an injected `set_mode` and a console fake."""
from __future__ import annotations

import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest

from crosspad_hil.cdc import CdcLink
from crosspad_hil.console import Console
from crosspad_hil.devices import AudioCardInfo, Device, MidiPortInfo, Ports, SerialPortInfo, UsbMode
from crosspad_hil.errors import CANCELLED, TIMEOUT, HilError
from crosspad_hil.scenarios import base, get, usb_mode_cycle
from tests.fakes import FakeSerial

CDC = SerialPortInfo(path="/dev/ttyACM0", vid=0x303A, pid=0x3456,
                     serial="CP-1", product="CrossPad", location="1-2.1")
CONSOLE = SerialPortInfo(path="/dev/ttyACM1", vid=0x0483, pid=0x5740,
                         serial="STM-1", product="CrossPad", location="1-2.2")
ESP_MIDI = MidiPortInfo(name="Crosspad MIDI 1", rtmidi_out=2, rtmidi_in=2,
                        alsa_hw="hw:4,0,0", rawmidi="/dev/snd/midiC4D0")
UAC2 = AudioCardInfo(name="Crosspad Audio", sounddevice_index=5, alsa_id="hw:5")


def heap_block(free_internal: int) -> list[str]:
    # the PerfMon block as the STM VCP shows it (parsers test HEAP_BLOCK)
    return [
        "I (10000) PerfMon: === Heap Statistics ===",
        "I (10000) PerfMon: Internal SRAM:",
        f"I (10001) PerfMon:   Free:                {free_internal} bytes",
        "I (10002) PerfMon: DMA-capable:",
        "I (10002) PerfMon:   Free:                70000 bytes",
        "I (10003) PerfMon: PSRAM:",
        "I (10003) PerfMon:   Free:              8123456 bytes",
        "I (10010) PerfMon:   Total tasks: 23",
    ]


def device_in(mode: UsbMode) -> Device:
    if mode == UsbMode.AUDIO:
        ports = Ports(cdc=None, console=CONSOLE, esp_midi=ESP_MIDI, uac2=UAC2)
    else:
        ports = Ports(cdc=CDC, console=CONSOLE, esp_midi=ESP_MIDI, uac2=None)
    return Device(id="dev_ab12", serial="CP-1", usb_mode=mode, ports=ports)


class FakeSetMode:
    """Stands in for usbmode.set_mode: records every call, flips the Device,
    and can fail a given call with TIMEOUT or feed console lines per switch."""

    def __init__(self, console: FakeSerial, cdc: FakeCdc | None = None) -> None:
        self.console = console
        self.cdc = cdc
        self.calls: list[tuple[str, str, bool, float]] = []
        self.fail_call: int | None = None
        self.heap_per_switch: list[int] = []
        self.lines_on_switch: dict[int, list[str]] = {}
        self.mode = UsbMode.DEFAULT

    def __call__(self, device: Device, mode: UsbMode, *, wait: bool = True,
                 timeout_s: float = 20.0, **kw: Any) -> Device:
        n = len(self.calls)
        self.calls.append((device.id, mode.value, wait, timeout_s))
        if self.fail_call == n:
            raise HilError(TIMEOUT, f"{device.id} did not re-enumerate in {mode.value} mode",
                           hint="power-cycle, or send F0 7D 1B 01 F7 by hand")
        self.mode = mode
        if n < len(self.heap_per_switch):
            # Both paths get the same number: the console block for the rounds
            # that still read it, and the CDC MEM reply for the ones that do
            # not, so a test scripting heap_per_switch does not have to know
            # which path the scenario takes this round.
            self.console.feed(heap_block(self.heap_per_switch[n]))
            if self.cdc is not None:
                self.cdc.free = self.heap_per_switch[n]
        self.console.feed(self.lines_on_switch.get(n, []))
        return device_in(mode)


def mem_line(free_internal: int) -> str:
    return (f"MEM: int_free={free_internal} int_largest=32000 int_min=18000 "
            "int_blocks=900 psram_free=8123456 psram_largest=8000000 psram_blocks=12")


class FakeCdc:
    """Answers MEM with whatever the run's heap script currently says.

    Since 16a6f45 the scenario samples the heap over CDC rather than off the
    PerfMon heartbeat, and until this existed the unit test had no fake for
    that path: ctx.open_cdc() fell through to base.Context and opened the real
    board. The tests passed or failed on whether a CrossPad happened to be
    plugged in, and stole its port from whatever else was using it.
    """

    def __init__(self) -> None:
        self.free: int | None = None
        self.opens = 0

    def link(self) -> CdcLink:
        self.opens += 1
        fake = FakeSerial(script=[("MEM", mem_line(self.free))] if self.free is not None else [])
        link = CdcLink("/dev/ttyACM0", serial_factory=lambda path, **kw: fake)
        link.open()
        return link


@dataclass
class FakeContext(base.Context):
    console_fake: FakeSerial = None  # type: ignore[assignment]
    cdc_fake: FakeCdc = None  # type: ignore[assignment]

    def open_cdc(self) -> CdcLink:
        link = self.cdc_fake.link()
        self.opened.append(link)
        return link

    def open_console(self, reset: bool = False) -> Console:
        fake = self.console_fake
        con = Console(self.device.ports.console.path,
                      log_path=self.workdir / "console.log",
                      serial_factory=lambda path, **kw: fake)
        con.open(reset=reset)
        self.opened.append(con)
        return con


def make_ctx(tmp_path: Path, start_mode: UsbMode = UsbMode.DEFAULT,
             heap0: int | None = 80_000) -> tuple[FakeContext, FakeSetMode]:
    con = FakeSerial()
    cdc = FakeCdc()
    if heap0 is not None:
        con.feed(heap_block(heap0))
        cdc.free = heap0
    ctx = FakeContext(device=device_in(start_mode), workdir=tmp_path,
                      cancelled=threading.Event(), log=lambda s: None,
                      console_fake=con, cdc_fake=cdc)
    return ctx, FakeSetMode(con, cdc)


@pytest.fixture(autouse=True)
def isolate(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(usb_mode_cycle, "LOCK_DIR", tmp_path / "locks")
    # The fake feeds a PerfMon block on the switch, so the wait for it returns as
    # soon as the reader thread has parsed it; this only bounds the rounds where
    # no block is fed at all (the fault and enumeration-failure cases).
    monkeypatch.setattr(usb_mode_cycle, "HEAP_SAMPLE_TIMEOUT_S", 0.15)
    # The fake hands its lines over at once, so the reader is caught up long
    # before the hardware-sized quiet window would end.
    monkeypatch.setattr(usb_mode_cycle, "CONSOLE_QUIET_MS", 50)
    monkeypatch.setattr(usb_mode_cycle, "CONSOLE_DRAIN_MAX_S", 0.5)


def run(ctx: FakeContext, sm: FakeSetMode, monkeypatch: pytest.MonkeyPatch,
        **params: object) -> base.Report:
    monkeypatch.setattr(usb_mode_cycle, "set_mode", sm)
    return usb_mode_cycle.SCENARIO.run(ctx, usb_mode_cycle.Params(**params), base.Progress())  # type: ignore[arg-type]


def test_registered_and_defaults() -> None:
    assert get("usb_mode_cycle") is usb_mode_cycle.SCENARIO
    p = usb_mode_cycle.Params()
    assert (p.rounds, p.dwell, p.enum_timeout) == (5, 2.0, 15.0)


def test_heap_drift_rule() -> None:
    assert usb_mode_cycle.heap_drift([]) is None
    assert usb_mode_cycle.heap_drift([100]) == (100, 100, 0.0)
    # base is the second sample (warm-up); the loss is spread over the cycles
    # between that sample and the last one, which is len-1-WARMUP_ROUNDS = 2 —
    # rounds 3 and 4. Here the heap falls 10 a round, and that is what is reported.
    base_, last, per_round = usb_mode_cycle.heap_drift([100, 90, 80, 70])
    assert (base_, last) == (90, 70)
    assert per_round == pytest.approx(10.0)


def test_three_clean_rounds(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    ctx, sm = make_ctx(tmp_path)
    sm.heap_per_switch = [79_000, 79_000, 79_000, 79_000, 79_000, 79_000]
    rep = run(ctx, sm, monkeypatch, rounds=3, dwell=0.0, enum_timeout=7.0)
    assert rep.passed, rep.summary
    assert rep.exit_code == 0
    modes = [c[1] for c in sm.calls]
    assert modes == ["audio", "default"] * 3
    assert all(c[2] is True and c[3] == 7.0 for c in sm.calls)
    d = rep.data
    assert len(d["rounds"]) == 3
    assert d["heap_start"] == 80_000
    assert d["heaps"] == [79_000, 79_000, 79_000]
    assert d["drift"] == {"base": 79_000, "last": 79_000, "per_round": 0.0}
    assert d["leak"] is False
    assert d["failures"] == []
    assert d["restored"] is True and d["final_mode"] == "default"
    # a console log is an artifact of every run
    assert any(a.role == "console" for a in rep.artifacts)
    assert (tmp_path / "console.log").exists()


def test_heap_leak_per_round_fails(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    ctx, sm = make_ctx(tmp_path)
    # after warm-up the heap falls 5 kB every cycle
    sm.heap_per_switch = [78_000, 78_000, 73_000, 73_000, 68_000, 68_000, 63_000, 63_000]
    rep = run(ctx, sm, monkeypatch, rounds=4, dwell=0.0)
    assert not rep.passed
    assert rep.exit_code == 1
    assert rep.data["heaps"] == [78_000, 73_000, 68_000, 63_000]
    assert rep.data["leak"] is True
    assert rep.data["drift"]["per_round"] == pytest.approx(5000.0)
    assert any("heap falls" in f for f in rep.data["failures"])


def test_switch_that_never_enumerates_fails_and_restores(tmp_path: Path,
                                                        monkeypatch: pytest.MonkeyPatch) -> None:
    ctx, sm = make_ctx(tmp_path)
    sm.fail_call = 3            # round 2, audio -> default never comes back
    rep = run(ctx, sm, monkeypatch, rounds=3, dwell=0.0)
    assert not rep.passed and rep.exit_code == 1
    assert "round 2" in rep.data["failures"][0]
    assert "default" in rep.data["failures"][0]
    # the finally clause tried again to restore DEFAULT after the failed leg
    assert [c[1] for c in sm.calls] == ["audio", "default", "audio", "default", "default"]
    assert rep.data["restored"] is True
    assert rep.data["final_mode"] == "default"


def test_first_switch_that_never_enumerates_fails_stayed_in_default(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The device never leaves default (the very first default->audio switch times
    out, e.g. audio init broken in firmware) — the round FAILs with a message that
    says where the device stayed, and the scenario still makes a best-effort attempt
    to (re)send the default switch rather than trusting its own stale model."""
    ctx, sm = make_ctx(tmp_path)
    sm.fail_call = 0            # round 1, default -> audio never comes back
    rep = run(ctx, sm, monkeypatch, rounds=3, dwell=0.0)
    assert not rep.passed and rep.exit_code == 1
    assert "round 1" in rep.data["failures"][0]
    assert "device stayed in default" in rep.data["failures"][0]
    # best-effort restore attempted even though the model never left DEFAULT
    assert [c[1] for c in sm.calls] == ["audio", "default"]
    assert rep.data["restored"] is True
    assert rep.data["final_mode"] == "default"
    # only round 1 was recorded; rounds 2 and 3 were never attempted
    assert len(rep.data["rounds"]) == 1


def test_fatal_on_console_stops_the_run(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    ctx, sm = make_ctx(tmp_path)
    sm.lines_on_switch = {2: ["Guru Meditation Error: Core  0 panic'ed (StoreProhibited).",
                              "ESP-ROM:esp32s3-20210327"]}
    rep = run(ctx, sm, monkeypatch, rounds=4, dwell=0.0)
    assert not rep.passed and rep.exit_code == 1
    assert len(rep.data["rounds"]) == 2
    assert rep.data["rounds"][1]["fatal"] and rep.data["rounds"][1]["reboots"] == 1
    assert rep.data["reboots"] == 1
    assert any("device fault" in f for f in rep.data["failures"])
    # rounds 3 and 4 were never attempted
    assert len(sm.calls) == 4


def test_not_in_default_mode_is_environment(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    ctx, sm = make_ctx(tmp_path, start_mode=UsbMode.AUDIO)
    rep = run(ctx, sm, monkeypatch, rounds=2)
    assert rep.exit_code == 2 and not rep.passed
    assert sm.calls == []
    assert "default" in rep.summary


def test_cdc_lock_is_held_for_the_run(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from crosspad_hil.locks import PortLock

    ctx, sm = make_ctx(tmp_path)
    seen: list[list[dict]] = []
    original = sm.__call__

    def spy(device: Device, mode: UsbMode, **kw: Any) -> Device:
        seen.append(PortLock.holders(tmp_path / "locks"))
        return original(device, mode, **kw)

    monkeypatch.setattr(usb_mode_cycle, "set_mode", spy)
    rep = usb_mode_cycle.SCENARIO.run(ctx, usb_mode_cycle.Params(rounds=1, dwell=0.0),
                                      base.Progress())
    assert rep.passed
    assert seen and all(any(h["port"] == CDC.path and h["purpose"] == "usb_mode_cycle"
                            for h in holders) for holders in seen)
    assert PortLock.holders(tmp_path / "locks") == []      # released at the end


def test_lock_busy_is_environment(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from crosspad_hil.locks import PortLock

    ctx, sm = make_ctx(tmp_path)
    with PortLock(CDC.path, "someone-else", tmp_path / "locks"):
        rep = run(ctx, sm, monkeypatch, rounds=1, dwell=0.0)
    assert rep.exit_code == 2
    assert sm.calls == []
    assert "busy" in rep.summary.lower() or "PORT_BUSY" in rep.summary


def test_cancel_between_legs_still_restores(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    ctx, sm = make_ctx(tmp_path)
    original = sm.__call__

    def cancel_after_audio(device: Device, mode: UsbMode, **kw: Any) -> Device:
        out = original(device, mode, **kw)
        if mode == UsbMode.AUDIO:
            ctx.cancelled.set()
        return out

    monkeypatch.setattr(usb_mode_cycle, "set_mode", cancel_after_audio)
    with pytest.raises(HilError) as ei:
        usb_mode_cycle.SCENARIO.run(ctx, usb_mode_cycle.Params(rounds=3, dwell=0.5),
                                    base.Progress())
    assert ei.value.code == CANCELLED
    # audio (then cancel noticed in the dwell) → restore default in finally
    assert [c[1] for c in sm.calls] == ["audio", "default"]
    assert sm.mode == UsbMode.DEFAULT


# ── heap over CDC ─────────────────────────────────────────────────────────────
def test_heap_is_read_over_cdc_when_the_default_profile_has_one(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The PerfMon heartbeat is off, so the console path only times out; MEM over
    CDC is the live answer, and the step's own port lock must not stand in its way."""
    class CdcContext(FakeContext):
        opened_cdc: int = 0

        def open_cdc(self):  # type: ignore[override]
            self.opened_cdc += 1
            lock_dir = usb_mode_cycle.LOCK_DIR
            # the step's lock must be released while we hold the port
            assert not list(Path(lock_dir).glob("*.lock")), "port lock held during MEM"
            class Link:
                def close(self) -> None:
                    pass
            return Link()

    con = FakeSerial()
    ctx = CdcContext(device=device_in(UsbMode.DEFAULT), workdir=tmp_path,
                     cancelled=threading.Event(), log=lambda s: None, console_fake=con)
    sm = FakeSetMode(con)
    heaps = iter([70_000, 69_000, 69_000, 69_000])
    monkeypatch.setattr(usb_mode_cycle.verbs, "mem", lambda link: {"int_free": next(heaps)})
    rep = run(ctx, sm, monkeypatch, rounds=3, dwell=0.0, enum_timeout=7.0)
    assert rep.passed, rep.summary
    assert rep.data["heap_start"] == 70_000
    assert rep.data["heaps"] == [69_000, 69_000, 69_000]
    assert ctx.opened_cdc == 4
    # and the lock is back in place once the read is over
    assert rep.data["restored"] is not False
