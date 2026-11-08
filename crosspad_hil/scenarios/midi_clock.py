"""Drive the board with a host MIDI clock and check it follows, then that it takes over.

For each tempo: CLOCK_STATUS RESET, 0xFA, `beats` beats of 0xF8 at 24 PPQN from a
dedicated thread (time.perf_counter deadlines, drift-free), 0xFC; then CLOCK_STATUS:
src=host, bpm within 0.1 %, ticks_in == sent, running went 1 -> 0. Finally stop
sending and wait `release_s`: src=int within 1.0-1.2 s, ticks_out still growing.
The forwarded leg (STM USB / TRS) is plan 2; this checks what the ESP can see.
"""
from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field
from typing import Any

from crosspad_hil import verbs
from crosspad_hil.midi import MidiIO, MidiRole
from crosspad_hil.scenarios.base import Context, Progress, Report, register

PPQN = 24
BPM_TOL = 0.001          # 0.1 %
RELEASE_MIN_S = 1.0
RELEASE_MAX_S = 1.2
# The board drains USB MIDI every millisecond; a status read issued right after
# 0xFC can land before the byte is handled.
AFTER_STOP_S = 0.1


@dataclass
class Params:
    bpms: str = field(default="90,120,174", metadata={"help": "tempos to send, comma-separated"})
    beats: int = field(default=16, metadata={"help": "beats per tempo"})
    release_s: float = field(default=1.5, metadata={"help": "how long to watch for the hand-back"})


def _send_clock(io: MidiIO, bpm: float, beats: int,
                cancelled: threading.Event) -> tuple[int, float]:
    """Send `beats` beats of 0xF8, drift-free. Returns (ticks sent, monotonic time of the last)."""
    period = 60.0 / (PPQN * bpm)
    t0 = time.perf_counter()
    sent = 0
    last = time.monotonic()
    for n in range(beats * PPQN):
        if cancelled.is_set():
            break
        due = t0 + n * period
        while True:
            left = due - time.perf_counter()
            if left <= 0:
                break
            time.sleep(left if left > 0.002 else 0)
        io.send_realtime(0xF8)
        last = time.monotonic()
        sent += 1
    return sent, last


class MidiClockScenario:
    name = "midi_clock"
    Params = Params
    description = "host MIDI clock at several tempos: the board follows it, then takes over"

    def run(self, ctx: Context, params: Params, progress: Progress) -> Report:
        try:
            bpms = [float(b) for b in params.bpms.split(",") if b.strip()]
            data: dict[str, Any] = {"runs": [], "release": None}
            failures: list[str] = []
            link = ctx.open_cdc()
            io = MidiIO(ctx.device, MidiRole.ESP)
            io.open()
            ctx.opened.append(io)
            steps = len(bpms) + 1
            try:
                for i, bpm in enumerate(bpms):
                    progress(i, steps, f"{bpm:g} BPM")
                    verbs.clock_status(link, reset=True)
                    before = verbs.clock_status(link)
                    io.send_realtime(0xFA)
                    sent, _last_tick = _send_clock(io, bpm, params.beats, ctx.cancelled)
                    during = verbs.clock_status(link)
                    io.send_realtime(0xFC)
                    last_host_byte = time.monotonic()
                    ctx.cancelled.wait(AFTER_STOP_S)
                    after = verbs.clock_status(link)
                    got_in = during["ticks_in"] - before["ticks_in"]
                    run = {"bpm": bpm, "sent": sent, "ticks_in": got_in,
                           "measured": during["bpm"], "src": during["src"],
                           "running": [during["running"], after["running"]],
                           "max_jitter_us": during["max_jitter_us"]}
                    data["runs"].append(run)
                    if during["src"] != "host":
                        failures.append(f"{bpm:g}: src={during['src']} while the host clocked")
                    if abs(during["bpm"] - bpm) > bpm * BPM_TOL:
                        failures.append(f"{bpm:g}: board says {during['bpm']:.2f}")
                    if got_in < sent - 1:   # the status read races the last tick
                        failures.append(f"{bpm:g}: {sent} ticks sent, {got_in} counted")
                    if run["running"] != [True, False]:
                        failures.append(f"{bpm:g}: running {run['running']} (want [True, False])")

                progress(len(bpms), steps, "host silent: hand-back")
                # The host stays the source while it sends anything; its last
                # byte here is the 0xFC, a few CDC reads after the last tick.
                t_stop = last_host_byte
                # A CLOCK_STATUS read takes ~0.2 s (the link waits out the reply),
                # so the hand-back is bracketed: after the start of the last read
                # that still said host, before the end of the first that said int.
                last_host_start: float | None = None
                first_int_end: float | None = None
                out0 = verbs.clock_status(link)["ticks_out"]
                while time.monotonic() - t_stop < params.release_s and first_int_end is None:
                    started = time.monotonic() - t_stop
                    st = verbs.clock_status(link)
                    if st["src"] == "host":
                        last_host_start = started
                    else:
                        first_int_end = time.monotonic() - t_stop
                out1 = verbs.clock_status(link)["ticks_out"]
                data["release"] = {"after_s": [last_host_start, first_int_end],
                                   "ticks_out_grew": out1 - out0}
                early = first_int_end is not None and first_int_end < RELEASE_MIN_S
                late = first_int_end is None or (last_host_start or 0.0) > RELEASE_MAX_S
                if early or late:
                    failures.append(f"hand-back between {last_host_start} and {first_int_end} s "
                                    "(want 1.0-1.2)")
                if out1 <= out0:
                    failures.append("internal clock not ticking after the hand-back")
                gen_late = verbs.clock_status(link)["max_late_us"]
                data["release"]["max_late_us"] = gen_late
                if gen_late > 1000:
                    failures.append(f"internal clock sent a tick {gen_late} us late (want <= 1000)")
            finally:
                io.close()
                link.close()

            progress(steps, steps, "done")
            passed = not failures
            summary = ("midi_clock: PASS — " + ", ".join(f"{r['bpm']:g}→{r['measured']:.2f}"
                                                       for r in data["runs"])
                       if passed else "midi_clock: FAIL — " + "; ".join(failures))
            data["failures"] = failures
            return Report(passed=passed, summary=summary, data=data, artifacts=[],
                          exit_code=0 if passed else 1)
        finally:
            ctx.close_all()


SCENARIO = register(MidiClockScenario())
