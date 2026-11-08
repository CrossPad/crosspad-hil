"""Flip the USB profile back and forth and check the device survives it.

Port of ``platform-idf/tools/hil_usb_mode_cycle.py``. Switching between
CDC+MIDI and MIDI+UAC2 tears down TinyUSB, installs a different descriptor set
and re-enumerates. Once is not the interesting number: a descriptor leak, an
endpoint that is not released, an audio task left suspended only shows after
several rounds.

Both legs go through ``usbmode.set_mode`` on the ESP MIDI port (SysEx 0x1B) and
return only when the host has re-enumerated the expected interfaces. Heap and
faults are read from the STM VCP console, which is unaffected by the ESP's USB
going away — the whole reason the check is possible at all.
"""
from __future__ import annotations

import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from crosspad_hil import verbs
from crosspad_hil.console import Console
from crosspad_hil.devices import Device, UsbMode
from crosspad_hil.errors import PORT_BUSY, HilError
from crosspad_hil.knowledge import load
from crosspad_hil.locks import PortLock
from crosspad_hil.scenarios.base import Artifact, Context, Progress, Report, register
from crosspad_hil.usbmode import set_mode

LEAK_BYTES_PER_ROUND = 2048    # hil_usb_mode_cycle: `if drift / rounds > 2048`
WARMUP_ROUNDS = 1              # `base = heaps[1]`: the first switch allocates the UAC2 buffers
# The closing line of a PerfMon block (markers.yaml heap_block.end). Once it is
# on the console, the parser's heap snapshot is that block's.
HEAP_BLOCK_END = re.escape(load("markers")["heap_block"]["end"])
HEAP_SAMPLE_TIMEOUT_S = 12.0   # perf_monitor_task prints every 10 s (main.cpp interval_ms)
# A console reader that has handed over nothing for this long is caught up; it
# matches the reader's own read() timeout, which is the longest it can sit on
# bytes it already has.
CONSOLE_QUIET_MS = 200
CONSOLE_DRAIN_MAX_S = 1.0
LOCK_PURPOSE = "usb_mode_cycle"
LOCK_DIR: Path | None = None   # tests point this at a temp dir; None = locks.default_lock_dir()
RECOVERY_HINT = ("device stuck without CDC: send F0 7D 1B 01 F7 to its ESP MIDI port "
                 "(`crosspad-hil usb-mode set default`), or power-cycle")


@dataclass
class Params:
    rounds: int = field(default=5, metadata={"help": "default->audio->default cycles"})
    dwell: float = field(default=2.0, metadata={"help": "seconds to stay in each profile"})
    enum_timeout: float = field(
        default=15.0, metadata={"help": "seconds to wait for the host to re-enumerate"})


def heap_drift(heaps: list[int]) -> tuple[int, int, float] | None:
    """(base, last, bytes lost per round) with the old warm-up rule.

    The first switch allocates the UAC2 buffers every later round reuses, so
    counting it as a leak would be wrong: the baseline is the sample after
    round 2 when there is one.
    """
    if not heaps:
        return None
    base = heaps[WARMUP_ROUNDS] if len(heaps) > WARMUP_ROUNDS else heaps[0]
    last = heaps[-1]
    # The cycles that actually separate the two samples: the baseline is the
    # sample after round WARMUP_ROUNDS+1, the last is after round len(heaps).
    # Dividing by len(heaps)-1 instead (what the old script did) spreads the
    # loss over a round that is not between them and under-reports the leak.
    rounds = max(1, len(heaps) - 1 - WARMUP_ROUNDS)
    return base, last, (base - last) / rounds


def _faults(con: Console, since_seq: int) -> tuple[list[str], int]:
    fatal: list[str] = []
    reboots = 0
    for ev in con.events(since_seq=since_seq):
        if ev.kind == "fatal":
            fatal.append(ev.line)
        elif ev.kind == "reboot":
            reboots += 1
    return fatal, reboots


def _heap_internal(con: Console) -> int | None:
    heap = con.snapshot().get("heap") or {}
    value = heap.get(0)
    return int(value) if value is not None else None


def _heap_via_cdc(ctx: Context, lock: PortLock) -> int | None:
    """Internal free heap over CDC (``MEM``), for the moments the board is in the
    default profile. The PerfMon heartbeat that ``_sample_heap`` waits for has
    been off since 2026-08-31 (it blocked the drain loop), so the console path
    now only ever times out -- 12 s per sample, and no heap drift verdict at
    all. The step holds the CDC port lock for the whole run; it is let go for
    exactly this one read and taken straight back.
    """
    lock.release()
    link = None
    try:
        link = ctx.open_cdc()
        return int(verbs.mem(link)["int_free"])
    except HilError:
        return None
    finally:
        if link is not None:
            link.close()
        lock.acquire()


def _sample_heap(con: Console, since_seq: int,
                 timeout_s: float = HEAP_SAMPLE_TIMEOUT_S) -> int | None:
    """Internal free heap from the first PerfMon block at or after ``since_seq``.

    The console is read by a background thread, so a snapshot taken the instant
    a mode switch returns is still the block from *before* the switch: every
    round would report the heap it started with, and a profile that loses 5 kB
    a cycle would read as flat. Waiting for the block's closing line is what
    makes the sample belong to this round — and it is a wait on data, not a
    sleep, so it costs nothing on a board that prints promptly.

    A board with PerfMon quiet keeps whatever it last printed rather than
    stalling the run; ``None`` until it has printed anything at all.
    """
    con.expect([HEAP_BLOCK_END], timeout_s=timeout_s, since_seq=since_seq)
    # The first block after the mark is not necessarily this round's: a block the
    # *previous* round produced can still be sitting unparsed when this round
    # takes its mark, and it is then numbered into this round. So let the reader
    # finish what it is holding and take the newest block it has — the round
    # wants the freshest heap, not the first one to arrive.
    deadline = time.monotonic() + CONSOLE_DRAIN_MAX_S
    while time.monotonic() < deadline:
        if not con.read(since_seq=con.seq + 1, wait_ms=CONSOLE_QUIET_MS).lines:
            break
    return _heap_internal(con)


def _wait(ctx: Context, seconds: float) -> None:
    if seconds > 0:
        ctx.cancelled.wait(seconds)
    ctx.check_cancelled()


def _env(summary: str, data: dict[str, Any], artifacts: list[Artifact]) -> Report:
    return Report(passed=False, summary=summary, data=data, artifacts=artifacts, exit_code=2)


class UsbModeCycleScenario:
    name = "usb_mode_cycle"
    Params = Params
    description = "cycle CDC+MIDI <-> MIDI+UAC2 and watch enumeration, heap and faults"

    def run(self, ctx: Context, params: Params, progress: Progress) -> Report:
        try:
            p = params
            data: dict[str, Any] = {
                "rounds_requested": p.rounds, "rounds": [], "heap_start": None, "heaps": [],
                "drift": None, "leak": False, "failures": [], "fatal": [], "reboots": 0,
                "restored": True, "final_mode": ctx.device.usb_mode.value,
            }
            artifacts: list[Artifact] = []
            ctx.check_cancelled()
            dev: Device = ctx.device
            if dev.usb_mode != UsbMode.DEFAULT or dev.ports.cdc is None:
                return _env(f"usb_mode_cycle: {dev.id} is in {dev.usb_mode.value} mode — "
                            "start from default (CDC+MIDI)", data, artifacts)
            if dev.ports.esp_midi is None:
                return _env(f"usb_mode_cycle: {dev.id} has no ESP MIDI port; the mode switch "
                            "(SysEx 0x1B) has nothing to travel on", data, artifacts)

            con = ctx.open_console(reset=False)
            if con.log_path is not None:
                artifacts.append(
                    Artifact(path=str(con.log_path), mime="text/plain", role="console"))
            # Console.events() is exclusive (ev.seq > since_seq): the boundary is
            # the last line that is not ours, not the first one that is.
            seq0 = con.seq
            cdc_path = dev.ports.cdc.path

            # Hold the CDC port for the whole run: its endpoint is about to vanish
            # and come back several times, and another session opening it in
            # between would either wedge or measure the wrong thing.
            lock = PortLock(cdc_path, LOCK_PURPOSE, LOCK_DIR)
            try:
                lock.acquire()
            except HilError as e:
                if e.code != PORT_BUSY:
                    raise
                return _env(f"usb_mode_cycle: CDC port busy — {e.message}"
                            + (f" ({e.hint})" if e.hint else ""), data, artifacts)

            try:
                data["heap_start"] = _heap_via_cdc(ctx, lock)
                if data["heap_start"] is None:
                    data["heap_start"] = _sample_heap(con, seq0 + 1)
                ctx.log(f"{p.rounds} rounds, dwell {p.dwell}s, start heap {data['heap_start']}")
                try:
                    self._cycle(ctx, p, con, seq0, data, progress, lock)
                finally:
                    # Whatever happened, do not leave the board without a CDC. A round
                    # that failed to switch to audio leaves ctx.device unchanged — that
                    # model is only ever updated on a *successful* switch, so "still
                    # DEFAULT" there is stale, not proof the real device is fine —
                    # best-effort restore anyway (see the "to audio failed" branch in
                    # _cycle, which sets the marker below).
                    dev = ctx.device
                    if data.pop("_mode_uncertain", False) or dev.usb_mode != UsbMode.DEFAULT:
                        try:
                            dev = set_mode(dev, UsbMode.DEFAULT, wait=True,
                                           timeout_s=p.enum_timeout)
                            ctx.device = dev
                        except HilError as e:
                            data["restored"] = False
                            data["failures"].append(
                                f"restore to default failed: {e.code}: {e.message} "
                                f"— {RECOVERY_HINT}")
                    data["final_mode"] = ctx.device.usb_mode.value
            finally:
                lock.release()

            fatal, reboots = _faults(con, seq0)
            data["fatal"] = fatal
            data["reboots"] = reboots
            return self._verdict(p, data, artifacts)

        finally:
            ctx.close_all()
    def _cycle(self, ctx: Context, p: Params, con: Console, seq0: int,
               data: dict[str, Any], progress: Progress, lock: PortLock) -> None:
        dev = ctx.device
        for rnd in range(1, p.rounds + 1):
            ctx.check_cancelled()
            record: dict[str, Any] = {
                "round": rnd, "to_audio_s": 0.0, "to_default_s": 0.0,
                "heap": None, "fatal": [], "reboots": 0,
            }
            mark = con.seq + 1      # console lines from here belong to this round
            # --- default -> audio ------------------------------------------
            t0 = time.monotonic()
            try:
                dev = set_mode(dev, UsbMode.AUDIO, wait=True, timeout_s=p.enum_timeout)
            except HilError as e:
                data["failures"].append(
                    f"round {rnd}: switch to audio failed — device stayed in default "
                    f"— {e.code}: {e.message}")
                data["rounds"].append(record)
                data["_mode_uncertain"] = True
                return
            ctx.device = dev
            record["to_audio_s"] = round(time.monotonic() - t0, 3)
            _wait(ctx, p.dwell)

            # --- audio -> default (still SysEx: no CDC in this profile) -----
            t0 = time.monotonic()
            try:
                dev = set_mode(dev, UsbMode.DEFAULT, wait=True, timeout_s=p.enum_timeout)
            except HilError as e:
                data["failures"].append(
                    f"round {rnd}: switch to default failed — device stayed in audio "
                    f"— {e.code}: {e.message}")
                data["rounds"].append(record)
                return
            ctx.device = dev
            record["to_default_s"] = round(time.monotonic() - t0, 3)
            _wait(ctx, p.dwell)

            # --- what the console saw --------------------------------------
            # Waiting for this round's own PerfMon block is also what gives the
            # reader thread the chance to deliver a panic the switch provoked:
            # a fault checked the microsecond a switch returns is a fault the
            # console has not printed yet.
            heap_now = _heap_via_cdc(ctx, lock)
            if heap_now is None:
                heap_now = _sample_heap(con, mark)
            record["heap"] = heap_now
            if heap_now is not None:
                data["heaps"].append(heap_now)
            fatal, reboots = _faults(con, seq0)
            already = sum(len(r["fatal"]) for r in data["rounds"])
            already_reb = sum(int(r["reboots"]) for r in data["rounds"])
            record["fatal"] = fatal[already:]
            record["reboots"] = reboots - already_reb
            data["rounds"].append(record)
            if record["fatal"] or record["reboots"]:
                for line in record["fatal"][-5:]:
                    ctx.log(f"   {line}")
                data["failures"].append(
                    f"round {rnd}: device fault — {len(record['fatal'])} fatal line(s), "
                    f"{record['reboots']} reboot(s); see console log")
                return
            start = data["heap_start"]
            delta = f"  ({heap_now - start:+d} vs start)" if (heap_now is not None
                                                             and start is not None) else ""
            ctx.log(f"round {rnd:3d}/{p.rounds}  ok   free={heap_now}{delta}")
            progress(rnd, p.rounds, f"round {rnd}/{p.rounds} ok free={heap_now}")

    def _verdict(self, p: Params, data: dict[str, Any], artifacts: list[Artifact]) -> Report:
        drift = heap_drift(list(data["heaps"]))
        if drift is not None:
            base, last, per_round = drift
            data["drift"] = {"base": base, "last": last, "per_round": round(per_round, 2)}
            if per_round > LEAK_BYTES_PER_ROUND:
                data["leak"] = True
                data["failures"].append(f"heap falls ~{int(per_round)} B per mode cycle")
        failures: list[str] = data["failures"]
        if failures:
            summary = (f"usb_mode_cycle: FAIL — {len(failures)} problem(s): "
                       + "; ".join(failures))
            return Report(passed=False, summary=summary, data=data, artifacts=artifacts,
                          exit_code=1)
        done = len(data["rounds"])
        return Report(passed=True,
                      summary=f"usb_mode_cycle: PASS — {done} clean profile switches",
                      data=data, artifacts=artifacts, exit_code=0)


SCENARIO = register(UsbModeCycleScenario())


def main(argv: list[str] | None = None) -> int:
    """Entry point for the platform-idf shim `tools/hil_usb_mode_cycle.py`."""
    import sys

    from crosspad_hil.cli import main as cli_main

    args = list(sys.argv[1:] if argv is None else argv)
    return cli_main(["run", "usb_mode_cycle", *args])
