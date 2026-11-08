"""Leave a board running for hours and say afterwards whether it stayed up.

Port of ``platform-idf/tools/hil_stability.py``. Nothing is driven here except
(optionally) a MIDI burst every so often: the scenario watches the STM VCP
console, which is the only port that survives ESP resets and USB-profile
switches, and turns what it sees into five counted things.

* **Resets** — every ROM banner is one. The uptime it interrupted, the
  ``rst:`` reason and the 300 console lines that led up to it are recorded,
  because "it restarts every few minutes" is diagnosed from the lines before
  the banner, not after it.
* **Boot health** — after a reset the boot markers must all reappear within
  ``BOOT_WINDOW_S``; a boot that half-completes is its own failure, distinct
  from a crash. The marker list is ``required_stability``, which is the smoke
  list minus ``STM32 ident:`` — an ESP reset mid-soak does not re-ident the
  STM co-processor, so demanding that line would fail every clean reboot.
* **Faults and E-lines** — from the shared console parser, so this scenario
  and ``smoke`` agree on what a panic is. Note that a ``TG0WDT_SYS_RST`` /
  ``TG1WDT`` reset reason does *not* match the fatal set (the digit in ``TG0``
  breaks the ``[A-Z_]*WDT`` pattern in ``markers.yaml``); a watchdog reset is
  therefore counted as a reboot and not as a fault. That is the old behaviour
  and it is deliberate here — the run fails on either, and conflating the two
  would change what the counters mean.
* **Stalls** — PerfMon prints every ~10 s, so a console that says nothing for
  ``--silence-timeout`` seconds is a hung device, and silence is reported once
  per silent stretch rather than once per poll.
* **Heap trend** — the PerfMon ``Free:`` block per slot. A minimum that walks
  steadily down is a leak; the first/last/min per slot is what the report
  carries.

The soak ends when the duration runs out or the run is cancelled. Cancelling is
a normal way to stop a soak, not a failure of one, so a cancelled run still
produces the report and is still judged on what it saw — which is why this
scenario watches ``ctx.cancelled`` itself instead of letting ``check_cancelled``
throw the evidence away.

One thing the old script did is not carried over: when the port itself went
away it reopened it by VID/PID, and tagged any reset landing within five
seconds of that reopen as ``monitor_induced`` rather than spontaneous, because
reopening the bridge VCP pulses the ESP reset line. ``Console`` owns its port
and offers no reopen, so a port that vanishes ends the run's console instead of
restarting it — the counter is still reported, always zero, so a reader of two
reports is not comparing different fields.

Exit code 0 = no resets, fatals, stalls or incomplete boots; 1 = instability
seen; 2 = no console port to watch.
"""
from __future__ import annotations

import json
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from crosspad_hil import audio_state, verbs
from crosspad_hil.cdc import CdcLink
from crosspad_hil.console import Console
from crosspad_hil.devices import Device, discover, select
from crosspad_hil.errors import HilError
from crosspad_hil.knowledge import load as load_knowledge
from crosspad_hil.midi import MidiIO, MidiRole
from crosspad_hil.parsers import ConsoleEvent
from crosspad_hil.scenarios.base import Artifact, Context, Progress, Report, register

CONTEXT_LINES = 300          # hil_stability.CONTEXT_LINES
# hil_stability.BOOT_WINDOW_S = 45, which markers.yaml carries as boot_timeout_s.
BOOT_WINDOW_S = float(load_knowledge("markers").get("boot_timeout_s", 45))
POLL_MS = 250                # how long a console read blocks before the loop looks around
PROGRESS_EVERY_S = 5.0       # a soak must show it is alive long before --status-interval
SECONDS_PER_HOUR = 3600.0
# hil_stability.midi_stim_thread: NoteOn 36..39, then NoteOff, 50 ms apart.
STIM_NOTES: tuple[int, ...] = (36, 37, 38, 39)
STIM_VELOCITY = 0x64
STIM_GAP_S = 0.05


@dataclass
class Params:
    duration_hours: float = field(
        default=0.0, metadata={"help": "run time in hours (0 = until cancelled)"})
    silence_timeout: float = field(
        default=90.0, metadata={"help": "seconds of console silence that count as a stall "
                                        "when there is no CDC port to probe"})
    probe_interval: float = field(
        default=30.0, metadata={"help": "seconds between MEM probes over CDC (0 = judge by "
                                        "console silence only); an unanswered probe is the stall"})
    status_interval: float = field(
        default=300.0, metadata={"help": "seconds between status lines in the log"})
    stim_midi: bool = field(
        default=False, metadata={"help": "send periodic MIDI note bursts while soaking"})
    stim_interval: float = field(
        default=20.0, metadata={"help": "seconds between MIDI bursts"})


class MidiStim(threading.Thread):
    """Periodic note bursts, so the soak is not just measuring an idle board.

    The device is looked up again for every burst: a USB re-enumeration
    reshuffles the ALSA/rtmidi indices, and the port index captured at the start
    of an eight-hour run is not the one that is there at hour three. A burst
    that cannot be sent is counted, never raised — the soak is the point, not
    the stimulus.
    """

    def __init__(self, device: Device, interval_s: float, halt: threading.Event,
                 log: Any) -> None:
        super().__init__(name="stability_midi_stim", daemon=True)
        self._device = device
        self._interval = max(0.05, interval_s)
        self._halt = halt
        self._log = log
        self.sent = 0
        self.errors = 0

    def _fresh(self) -> Device:
        try:
            self._device = select(discover(), self._device.id)
        except HilError:
            pass          # the board may be mid-reset; try the port we know
        return self._device

    def run(self) -> None:
        # wait() first: hil_stability sends its first burst one interval in.
        while not self._halt.wait(self._interval):
            try:
                with MidiIO(self._fresh(), MidiRole.ESP) as io:
                    for note in STIM_NOTES:
                        io.send_note(True, note, STIM_VELOCITY)
                        time.sleep(STIM_GAP_S)
                    for note in STIM_NOTES:
                        io.send_note(False, note)
                        time.sleep(STIM_GAP_S)
            except HilError as e:
                self.errors += 1
                self._log(f"MIDI stimulus burst failed: {e.code}: {e.message}")
            else:
                self.sent += 1


class Monitor:
    """The counters, the context dumps and the events file — hil_stability.Monitor.

    Fed from :class:`~crosspad_hil.console.Console` rather than from the port
    directly: fatals, E-lines, reboots, reset reasons and PerfMon heap blocks
    all come off the shared parser, so this scenario cannot disagree with
    ``smoke`` about what a panic looks like. Only the per-boot marker set is
    tracked here, because it is the stability list, not the smoke one.
    """

    def __init__(self, con: Console, workdir: Path, required: list[str], log: Any) -> None:
        self._con = con
        self._workdir = workdir
        self._required = list(required)
        self._log = log
        self.events_path = workdir / "events.jsonl"
        # monitor_induced_resets is kept at zero and reported: the old script
        # reopened a lost port itself and tagged the reset that reopening caused,
        # and Console has no reopen for this to hang off. See the module note.
        self.stats: dict[str, Any] = {
            "start": time.strftime("%F %T"),
            "resets": 0, "monitor_induced_resets": 0,
            "fatals": 0, "errors": 0, "stalls": 0,
            "incomplete_boots": 0, "lines": 0,
        }
        self.reset_events: list[dict[str, Any]] = []
        self.heap_series: list[tuple[float, list[int]]] = []
        self.contexts: list[str] = []
        self.boot_t = time.monotonic()
        self.boot_pending: set[str] = set()
        self.boot_deadline: float | None = None
        self.first_banner_seen = False
        self._boot_window_s = BOOT_WINDOW_S
        self._dumps = 0
        self._heap_seq: int | None = None
        self._heap_block: list[int] = []

    # -- output ------------------------------------------------------------
    def event(self, kind: str, **kw: Any) -> dict[str, Any]:
        rec = {"ts": time.strftime("%F %T"), "kind": kind, **kw}
        with open(self.events_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(rec, default=str) + "\n")
        return rec

    def dump_context(self, tag: str, seq: int) -> str:
        """The CONTEXT_LINES console lines ending at ``seq``, as their own file.

        The ordinal in the name is not decoration: a reset burst produces a
        banner and a panic inside the same second, and the old script's
        timestamp-only name meant the second dump overwrote the first — losing
        exactly the evidence the file exists for.
        """
        self._dumps += 1
        path = self._workdir / f"context_{tag}_{self._dumps:03d}_{time.strftime('%H%M%S')}.log"
        start = max(1, seq - CONTEXT_LINES + 1)
        lines = [line for s, line in self._con.read(since_seq=start,
                                                    limit=CONTEXT_LINES).lines if s <= seq]
        path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        self.contexts.append(str(path))
        return str(path)

    # -- ingest ------------------------------------------------------------
    def ingest(self, lines: list[tuple[int, str]], events: list[ConsoleEvent],
               now: float) -> None:
        """Take one batch of console lines and its parser events, in line order.

        The two views have to be merged rather than handled one after the other:
        the banner that starts a boot window and the markers that close it can
        land in the same read, and a batch processed events-first would arm the
        window after its own markers had already gone past — every clean reboot
        would then be reported as an incomplete boot.
        """
        self.stats["lines"] += len(lines)
        pos = 0
        for seq, line in lines:
            self._on_line(line, now)
            while pos < len(events) and events[pos].seq <= seq:
                self._on_event(events[pos], now)
                pos += 1
        for ev in events[pos:]:
            self._on_event(ev, now)
        self._flush_heap(now)

    def _on_line(self, line: str, now: float) -> None:
        if not self.boot_pending:
            return
        for marker in list(self.boot_pending):
            if marker in line:
                self.boot_pending.discard(marker)
        if not self.boot_pending:
            self._log(f"boot complete in {round(now - self.boot_t, 1)}s")

    def _on_event(self, ev: ConsoleEvent, now: float) -> None:
        if ev.kind != "heap":
            self._flush_heap(now)
        if ev.kind == "reboot":
            self._on_reset(ev, now)
        elif ev.kind == "reset_reason":
            self._on_reset_reason(ev)
        elif ev.kind == "fatal":
            self.stats["fatals"] += 1
            ctx_path = self.dump_context("fatal", ev.seq)
            self.event("fatal", line=ev.line, context=ctx_path)
            self._log(f"!! FATAL: {ev.line}")
        elif ev.kind == "error_line":
            # The allow-list is already applied by the parser (error_allow).
            self.stats["errors"] += 1
            self.event("error_line", line=ev.line)
        elif ev.kind == "heap":
            # One PerfMon block is one event per slot, all carrying the seq of
            # the line that closed the block.
            if self._heap_seq is not None and ev.seq != self._heap_seq:
                self._flush_heap(now)
            self._heap_seq = ev.seq
            self._heap_block.append(int(ev.data["free"]))

    def _flush_heap(self, now: float) -> None:
        if self._heap_block:
            self.heap_series.append((round(now, 1), self._heap_block))
        self._heap_seq, self._heap_block = None, []

    def _on_reset(self, ev: ConsoleEvent, now: float) -> None:
        if not self.first_banner_seen:
            # Opening the bridge VCP pulses the ESP reset line, so the first
            # banner of a run is one the harness caused.
            self.first_banner_seen = True
            self._log("initial boot detected (port-open reset) — monitoring from here")
        else:
            uptime = round(now - self.boot_t, 1)
            ctx_path = self.dump_context("reset", ev.seq)
            self.stats["resets"] += 1
            self.reset_events.append(
                self.event("reset", uptime_s=uptime, context=ctx_path))
            self._log(f"!! RESET after {uptime}s uptime (context -> {ctx_path})")
        self.boot_t = now
        self.boot_pending = set(self._required)
        self.boot_deadline = now + self._boot_window_s

    def _on_reset_reason(self, ev: ConsoleEvent) -> None:
        reason = f"{ev.data['code']} ({ev.data['name']})"
        if self.reset_events and "reason" not in self.reset_events[-1]:
            self.reset_events[-1]["reason"] = reason
            self.event("reset_reason", reason=reason)
            self._log(f"   reset reason: {reason}")

    # -- timers ------------------------------------------------------------
    def check_boot_deadline(self, now: float) -> None:
        if self.boot_deadline is None or now <= self.boot_deadline:
            return
        if self.boot_pending:
            missing = sorted(self.boot_pending)
            self.stats["incomplete_boots"] += 1
            self.event("incomplete_boot", missing=missing)
            self._log(f"!! INCOMPLETE BOOT — missing: {missing}")
        self.boot_deadline = None
        self.boot_pending = set()

    def stall(self, silent_s: float, seq: int) -> None:
        self.stats["stalls"] += 1
        ctx_path = self.dump_context("stall", seq)
        self.event("stall", silent_s=round(silent_s, 1), context=ctx_path)
        self._log(f"!! STALL: no console output for {round(silent_s)}s "
                  f"(context -> {ctx_path})")

    def probe_stall(self, reason: str, seq: int) -> None:
        self.stats["stalls"] += 1
        ctx_path = self.dump_context("stall", seq)
        self.event("stall", probe=reason, context=ctx_path)
        self._log(f"!! STALL: MEM probe unanswered ({reason}) (context -> {ctx_path})")

    def probe_heap(self, now: float, mem: dict[str, Any]) -> None:
        self.heap_series.append((round(now, 1), [int(mem["int_free"]),
                                                 int(mem.get("psram_free") or 0)]))

    # -- summary -----------------------------------------------------------
    def heap_summary(self) -> list[dict[str, int]] | None:
        """Per heap slot: where it started, where it ended, and its low-water mark."""
        if len(self.heap_series) < 2:
            return None
        first, last = self.heap_series[0][1], self.heap_series[-1][1]
        out: list[dict[str, int]] = []
        for slot in range(min(len(first), len(last))):
            low = min(block[slot] for _t, block in self.heap_series if len(block) > slot)
            out.append({"slot": slot, "first": first[slot], "last": last[slot],
                        "min": low, "delta": last[slot] - first[slot]})
        return out


def _summary_lines(data: dict[str, Any]) -> list[str]:
    s = data["stats"]
    lines = [f"SUMMARY: resets={s['resets']} fatals={s['fatals']} errors={s['errors']} "
             f"stalls={s['stalls']} incomplete_boots={s['incomplete_boots']} "
             f"lines={s['lines']}"]
    for ev in data["reset_events"]:
        lines.append(f"  reset @ {ev['ts']} after {ev['uptime_s']}s "
                     f"reason={ev.get('reason', '?')}")
    if data.get("liveness") == "cdc" and data.get("heap_trend"):
        lines.append("  heap slots: [0] internal free, [1] PSRAM free (MEM over CDC)")
    for h in data["heap_trend"] or []:
        lines.append(f"  heap[{h['slot']}]: first={h['first']} last={h['last']} "
                     f"min={h['min']} delta={h['delta']:+d}")
    return lines


class StabilityScenario:
    name = "stability"
    Params = Params
    description = "soak: watch the console for resets, fatals, stalls, bad boots and heap drift"

    def run(self, ctx: Context, params: Params, progress: Progress) -> Report:
        try:
            p = params
            required = list(load_knowledge("markers")["required_stability"])
            data: dict[str, Any] = {
                "duration_hours": p.duration_hours, "silence_timeout": p.silence_timeout,
                "stim_midi": p.stim_midi, "cancelled": False, "seconds": 0.0,
                "stats": {}, "reset_events": [], "heap_trend": None,
                "stim_bursts": 0, "stim_errors": 0,
            }
            ctx.check_cancelled()
            con = ctx.open_console(reset=False)
            artifacts: list[Artifact] = []
            if con.log_path is not None:
                artifacts.append(
                    Artifact(path=str(con.log_path), mime="text/plain", role="console"))

            mon = Monitor(con, ctx.workdir, required, ctx.log)
            artifacts.append(
                Artifact(path=str(mon.events_path), mime="application/x-ndjson", role="events"))
            # If the reset that opening the port caused is already in the ring,
            # the next banner is a real one. If it is not there yet, the first
            # banner to arrive is still ours. Either way the harness never
            # reports the reboot it caused itself.
            mon.first_banner_seen = any(ev.kind == "reboot" for ev in con.events(since_seq=0))
            if mon.first_banner_seen:
                # open_console already waited that boot out; asking for its
                # markers now would only find lines that have scrolled past.
                ctx.log("console open; the port-open reset is already accounted for")

            # A quiet firmware is not a hung one: since the PerfMon heartbeat went,
            # an idle board prints nothing for minutes. With a CDC port the soak
            # asks MEM instead and treats a missing answer as the stall; the
            # replies double as the heap series the console no longer carries.
            link: CdcLink | None = None
            if p.probe_interval > 0 and ctx.device.ports.cdc is not None:
                try:
                    link = ctx.open_cdc()
                except HilError as e:
                    ctx.log(f"no CDC probe ({e.code}); judging liveness by console silence")
            data["liveness"] = "cdc" if link is not None else "console"
            # Read-only until the end: the soak itself only listens and sends MIDI, but a stim
            # note or a board that reboots under it must not leave the mixer/codec moved.
            audio_snap = audio_state.snapshot(link) if link is not None else None
            halt = threading.Event()
            stim = MidiStim(ctx.device, p.stim_interval, halt, ctx.log) if p.stim_midi else None
            if stim is not None:
                stim.start()
            try:
                self._soak(ctx, p, con, mon, progress, data, link)
            finally:
                halt.set()
                audio_state.put_back(link, audio_snap, ctx.log, data)
                if stim is not None:
                    stim.join(timeout=max(2.0, STIM_GAP_S * 2 * len(STIM_NOTES) + 1.0))
                    data["stim_bursts"] = stim.sent
                    data["stim_errors"] = stim.errors
                mon.stats["end"] = time.strftime("%F %T")
                data["stats"] = dict(mon.stats)
                data["reset_events"] = list(mon.reset_events)
                data["heap_trend"] = mon.heap_summary()
                for path in mon.contexts:
                    artifacts.append(
                        Artifact(path=path, mime="text/plain", role="context"))

            s = mon.stats
            unstable = bool(s["resets"] or s["fatals"] or s["stalls"] or s["incomplete_boots"])
            lines = _summary_lines(data)
            if data["cancelled"]:
                lines.insert(0, f"stability: cancelled after {data['seconds']:.0f}s "
                                "— judged on what it saw")
            lines.insert(0, "stability: " + ("FAIL — the board did not stay up"
                                             if unstable else "PASS — the board stayed up"))
            return Report(passed=not unstable, summary="\n".join(lines), data=data,
                          artifacts=artifacts, exit_code=1 if unstable else 0)

        finally:
            ctx.close_all()

    def _soak(self, ctx: Context, p: Params, con: Console, mon: Monitor,
              progress: Progress, data: dict[str, Any], link: CdcLink | None = None) -> None:
        started = time.monotonic()
        total_s = p.duration_hours * SECONDS_PER_HOUR if p.duration_hours > 0 else None
        t_end = started + total_s if total_s is not None else None
        total = max(1, int(total_s)) if total_s is not None else None
        cursor = con.seq + 1
        ev_cursor = con.seq
        last_line_t = started
        last_status_t = started
        last_progress_t = 0.0
        last_probe_t = started
        stall_reported = False
        how = (f"MEM probe every {p.probe_interval:g}s" if link is not None
               else f"stall after {p.silence_timeout}s of silence")
        ctx.log(f"soaking {'until cancelled' if total_s is None else f'for {total_s:.0f}s'}; {how}")
        while True:
            now = time.monotonic()
            if ctx.cancelled.is_set():
                data["cancelled"] = True
                break
            if t_end is not None and now >= t_end:
                break

            result = con.read(since_seq=cursor, wait_ms=POLL_MS)
            now = time.monotonic()
            if result.lines:
                cursor = result.next_seq
                last_line_t = now
                stall_reported = False
                # Events past the last line of this batch belong to lines the
                # reader has parsed but read() has not handed over yet; they
                # wait, so line and event always arrive together.
                last_seq = result.lines[-1][0]
                events = [ev for ev in con.events(since_seq=ev_cursor) if ev.seq <= last_seq]
                if events:
                    ev_cursor = events[-1].seq
                mon.ingest(result.lines, events, now)
            mon.check_boot_deadline(now)
            if link is not None and now - last_probe_t >= p.probe_interval:
                last_probe_t = now
                try:
                    mem = verbs.mem(link)
                except HilError as e:
                    if not stall_reported:
                        mon.probe_stall(e.code, con.seq)
                        stall_reported = True
                else:
                    mon.probe_heap(now, mem)
                    last_line_t = now
                    stall_reported = False
                now = time.monotonic()

            # One stall per silent stretch, not one per poll: the old script
            # latched it the same way, and a hung board would otherwise fill the
            # events file with the same finding every quarter second. With a
            # CDC probe the silence rule is off: quiet is what this firmware is.
            if link is None and not stall_reported and now - last_line_t > p.silence_timeout:
                mon.stall(now - last_line_t, con.seq)
                stall_reported = True

            if now - last_status_t >= p.status_interval:
                last_status_t = now
                self._status(ctx, mon, now)
            if now - last_progress_t >= PROGRESS_EVERY_S:
                last_progress_t = now
                s = mon.stats
                progress(int(now - started), total,
                         f"up {int(now - started)}s, resets={s['resets']} "
                         f"fatals={s['fatals']} stalls={s['stalls']}")
        data["seconds"] = round(time.monotonic() - started, 1)
        progress(int(data["seconds"]), total, "soak finished")

    def _status(self, ctx: Context, mon: Monitor, now: float) -> None:
        s = mon.stats
        heap = mon.heap_series[-1][1] if mon.heap_series else []
        ctx.log(f"status: up {round((now - mon.boot_t) / 60)}min since boot, "
                f"resets={s['resets']} fatals={s['fatals']} errors={s['errors']} "
                f"stalls={s['stalls']} heap={heap}")


SCENARIO = register(StabilityScenario())


def main(argv: list[str] | None = None) -> int:
    """Entry point for the platform-idf shim `tools/hil_stability.py`."""
    import sys

    from crosspad_hil.cli import main as cli_main

    args = list(sys.argv[1:] if argv is None else argv)
    return cli_main(["run", "stability", *args])
