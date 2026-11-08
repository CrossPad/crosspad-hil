"""TRS MIDI OUT cabled to IN: Type A/B, throughput, and throughput against bit errors per segment.

Needs a loop cable from the OUT jack to the IN jack and STM 1.14 (MIDI_TRS answers a type).
The stream is numbered note-ons on channel 1 (seq % 32 as the note, 1 + seq // 32 as the
velocity), the same encoding the ESP's MIDI_LOOP end speaks, so every leg is checked message
by message:

    host --USB--> STM --TRS OUT--> cable --TRS IN--> STM --USB--> host
                                                         \\--UART--> ESP (MIDI_LOOP capture)
    ESP (MIDI_LOOP SEND) --UART--> STM --TRS OUT--> ... the same way back

Phases: each type (A, B) with a short stream -- with a straight cable exactly one should carry
it -- then the working type again (MIDI IN still receives after the other type's silence or
garbage); throughput at several rates from the host and from the ESP; a TX baud skew sweep
(real bit errors at the receiver); the STM's bit-flip injector on each segment at several
rates; a clean run at the end. The routing the test needs (MIDI IN to USB and ESP, USB and
ESP to MIDI OUT, no thru, no USB to ESP) is set for the run and the saved one put back.
"""
from __future__ import annotations

import math
import threading
import time
from dataclasses import dataclass, field
from typing import Any

from crosspad_hil import verbs
from crosspad_hil.errors import ENV, NO_DEVICE, HilError
from crosspad_hil.midi import _rtmidi_factories, _unregister
from crosspad_hil.scenarios.base import Context, Progress, Report, register

STATUS = 0x90                 # note-on, channel 1 (what the ESP listens on)
NOTES = 32
VELOCITIES = 127
PERIOD = NOTES * VELOCITIES   # the sequence repeats after this many messages
BITS_PER_MSG = 30             # 3 bytes of start + 8 data + stop
LINE_MSGS_PER_S = 31250 / BITS_PER_MSG

# r0 = MIDI2USB | MIDI2ESP | USB2MIDI | ESP2MIDI, r1 = ENABLED (crosspad-core
# STMRoutingSettings). No MIDI thru (the cable would echo forever), no USB to ESP
# (the ESP would count the host's stream before it went round the loop).
LOOP_ROUTE = "4E 04"

SEGMENTS = {  # name -> (STM MIDI_SEG_*, source of the stream, counter of bytes through it)
    "usb_out": (5, "host", "usb_out"),
    "trs_out": (1, "host", "trs_queued"),
    "trs_in": (2, "host", "in_rx"),
    "to_usb": (4, "host", "to_usb"),
    "to_esp": (3, "host", "to_esp"),
    "esp_in": (6, "esp", "esp_in"),
}

DRAIN_QUIET_S = 0.4
DRAIN_MAX_S = 6.0
WORKING_RATIO = 0.95


@dataclass
class Params:
    probe_count: int = field(default=300, metadata={"help": "messages per type in the A/B check"})
    seconds: float = field(default=3.0, metadata={"help": "length of each throughput leg"})
    rates: str = field(default="250,500,900,1200", metadata={"help": "host stream rates, msg/s"})
    esp_rates: str = field(default="500,900", metadata={"help": "ESP stream rates, msg/s"})
    skews: str = field(default="-60,-50,-45,-40,-35,-30,0,30,35,40,45,50,60",
                       metadata={"help": "TX baud skews, 0.1 %% steps"})
    flip_one_in: str = field(default="1000000,100000,10000,1000,100",
                             metadata={"help": "injector rates: a bit flips with 1/N"})
    segments: str = field(default=",".join(SEGMENTS),
                          metadata={"help": "segments for the injector"})
    error_rate: int = field(default=500,
                            metadata={"help": "stream rate for the skew and injector legs"})
    error_seconds: float = field(default=2.0,
                                 metadata={"help": "length of each skew / injector leg"})


def encode(seq: int) -> list[int]:
    p = seq % PERIOD
    return [STATUS, p % NOTES, 1 + p // NOTES]


def decode(msg: list[int]) -> int | None:
    if len(msg) != 3 or msg[0] != STATUS or msg[1] >= NOTES or not 1 <= msg[2] <= VELOCITIES:
        return None
    return (msg[2] - 1) * NOTES + msg[1]


def _pct(sorted_vals: list[float], p: float) -> float:
    if not sorted_vals:
        return math.nan
    return sorted_vals[min(len(sorted_vals) - 1, int(round(p * (len(sorted_vals) - 1))))]


def analyze(sent_t: list[float], received: list[tuple[float, list[int]]],
            timed: bool = True) -> dict[str, Any]:
    """Messages sent at `sent_t[seq]` against what came back.

    `delivered`: distinct sequence numbers that arrived; `spurious`: a test message whose
    number was already seen, never sent, or arrived before it was sent (a corrupted one that
    still decodes); `foreign`: anything else that arrived. Latency is per delivered message,
    from its send time (`timed=False` when the sender's clock is not the host's).
    """
    n = len(sent_t)
    seen: set[int] = set()
    lat: list[float] = []
    foreign = spurious = in_order = gaps = 0
    last: int | None = None
    first_t = last_t = None
    for t, msg in received:
        seq = decode(msg)
        if seq is None:
            foreign += 1
            continue
        if seq >= n or seq in seen or (timed and t < sent_t[seq]):
            spurious += 1
            continue
        seen.add(seq)
        if last is not None:
            if seq == last + 1:
                in_order += 1
            elif seq > last:
                gaps += 1
        last = seq
        first_t = t if first_t is None else first_t
        last_t = t
        if timed:
            lat.append((t - sent_t[seq]) * 1000.0)
    lat.sort()
    delivered = len(seen)
    span = (last_t - first_t) if first_t is not None and last_t is not None else 0.0
    return {
        "sent": n, "delivered": delivered, "lost": n - delivered,
        "loss_pct": round(100.0 * (n - delivered) / n, 3) if n else 0.0,
        "in_order": in_order, "gaps": gaps, "spurious": spurious, "foreign": foreign,
        "goodput": round((delivered - 1) / span, 1) if span > 0 and delivered > 1 else 0.0,
        "lat_p50_ms": round(_pct(lat, 0.5), 2) if lat else None,
        "lat_p99_ms": round(_pct(lat, 0.99), 2) if lat else None,
        "lat_max_ms": round(lat[-1], 2) if lat else None,
    }


def ber_from_mer(errored: int, sent: int) -> float | None:
    """Bit error rate that would give this message error rate, errors independent."""
    if not sent:
        return None
    mer = min(errored / sent, 1.0)
    if mer >= 1.0:
        return 1.0
    return 1.0 - (1.0 - mer) ** (1.0 / BITS_PER_MSG)


class StmPort:
    """The STM's USB MIDI port, both directions (MidiIO opens only its OUT)."""

    def __init__(self, ctx: Context) -> None:
        info = ctx.device.ports.stm_midi
        if info is None or info.rtmidi_out is None or info.rtmidi_in is None:
            raise HilError(NO_DEVICE, "the STM MIDI port is not enumerated in both directions",
                           hint="is the STM bridge on USB? run `crosspad-hil devices`")
        out_f, in_f = _rtmidi_factories()
        self._out = out_f()
        self._in = in_f()
        self._lock = threading.Lock()
        self.rx: list[tuple[float, list[int]]] = []
        self._out.open_port(info.rtmidi_out)
        self._in.open_port(info.rtmidi_in)
        self._in.ignore_types(sysex=True, timing=True, active_sense=True)
        self._in.set_callback(self._on_message, None)

    def _on_message(self, event: tuple[list[int], float], _data: Any) -> None:
        t = time.perf_counter()
        with self._lock:
            self.rx.append((t, list(event[0])))

    def take(self) -> list[tuple[float, list[int]]]:
        with self._lock:
            out, self.rx = self.rx, []
        return out

    def count(self) -> int:
        with self._lock:
            return len(self.rx)

    def send_stream(self, count: int, rate: float, cancelled: threading.Event) -> list[float]:
        t0 = time.perf_counter()
        sent_t: list[float] = []
        for seq in range(count):
            if cancelled.is_set():
                break
            due = t0 + seq / rate
            while True:
                left = due - time.perf_counter()
                if left <= 0:
                    break
                time.sleep(left if left > 0.002 else 0)
            sent_t.append(time.perf_counter())
            self._out.send_message(encode(seq))
        return sent_t

    def close(self) -> None:
        for h in (self._in, self._out):
            try:
                if h is self._in:
                    h.cancel_callback()
                h.close_port()
            finally:
                _unregister(h)


def _floats(text: str) -> list[float]:
    return [float(x) for x in text.split(",") if x.strip()]


class MidiTrsLoopScenario:
    name = "midi_trs_loop"
    Params = Params
    description = ("TRS MIDI OUT looped to IN: Type A/B, throughput, and throughput against "
                   "bit errors per segment")

    def _drain(self, port: StmPort, cancelled: threading.Event) -> None:
        t_end = time.monotonic() + DRAIN_MAX_S
        n, quiet_since = port.count(), time.monotonic()
        while time.monotonic() < t_end and not cancelled.is_set():
            time.sleep(0.05)
            m = port.count()
            if m != n:
                n, quiet_since = m, time.monotonic()
            elif time.monotonic() - quiet_since >= DRAIN_QUIET_S:
                return

    def _leg(self, ctx: Context, link: Any, port: StmPort, source: str, count: int,
             rate: float) -> dict[str, Any]:
        """One stream round the loop; host and ESP ends judged, STM counters read."""
        verbs.midi_trs(link, "RESET")
        verbs.midi_loop(link, "CAPTURE 1")
        port.take()
        if source == "host":
            sent_t = port.send_stream(count, rate, ctx.cancelled)
            timed = True
        else:
            t0 = time.perf_counter()
            verbs.midi_loop(link, f"SEND {count} {int(rate)}")
            while not ctx.cancelled.is_set() and verbs.midi_loop(link).get("sending"):
                time.sleep(0.1)
            sent_t = [t0 + seq / rate for seq in range(count)]
            timed = False
        self._drain(port, ctx.cancelled)
        host = analyze(sent_t, port.take(), timed=timed)
        esp = verbs.midi_loop(link)
        stm = verbs.midi_trs(link)
        esp_ok = int(esp.get("rx", 0)) - int(esp.get("back", 0))
        sent = len(sent_t) if source == "host" else int(esp.get("sent", 0))
        return {
            "source": source, "rate": rate, "host": host,
            "esp": {"rx": esp.get("rx"), "ok": esp_ok, "lost": sent - esp_ok,
                    "loss_pct": round(100.0 * (sent - esp_ok) / sent, 3) if sent else 0.0,
                    "gaps": esp.get("gaps"), "back": esp.get("back"),
                    "send_fail": esp.get("fail") if source == "esp" else None},
            "stm": {k: v for k, v in stm.items()
                    if k not in ("type", "skew", "flip_seg", "flip_n", "route")},
        }

    def run(self, ctx: Context, params: Params, progress: Progress) -> Report:
        failures: list[str] = []
        data: dict[str, Any] = {"types": {}, "throughput": [], "skew": [], "flips": [],
                                "final": None}
        try:
            link = ctx.open_cdc()
            state = verbs.midi_trs(link)
            if state.get("type") not in ("A", "B"):
                raise HilError(ENV, "MIDI_TRS: the STM has no TRS MIDI OUT",
                               hint="flash STM 1.14 or newer (STM_FW_UPDATE)")
            saved_type = state["type"]
            port = StmPort(ctx)
            ctx.opened.append(port)
            rates = _floats(params.rates)
            esp_rates = _floats(params.esp_rates)
            skews = [int(s) for s in _floats(params.skews)]
            ones = [int(n) for n in _floats(params.flip_one_in)]
            segs = [s.strip() for s in params.segments.split(",") if s.strip()]
            unknown = [s for s in segs if s not in SEGMENTS]
            if unknown:
                raise HilError(ENV, f"unknown segment(s) {unknown}",
                               hint=f"one of {list(SEGMENTS)}")
            steps = 3 + len(rates) + len(esp_rates) + len(skews) + len(segs) * len(ones) + 1
            step = 0

            def tick(what: str) -> None:
                nonlocal step
                progress(step, steps, what)
                step += 1

            try:
                verbs.midi_trs(link, f"ROUTE {LOOP_ROUTE}")
                verbs.midi_trs(link, "FLIP 0 0")
                verbs.midi_trs(link, "SKEW 0")

                # -- Type A / B -------------------------------------------------
                probe_rate = 300.0
                for t in ("A", "B"):
                    tick(f"type {t}")
                    verbs.midi_trs(link, f"TYPE {t}")
                    time.sleep(0.1)
                    data["types"][t] = self._leg(ctx, link, port, "host",
                                                 params.probe_count, probe_rate)
                working = [t for t, leg in data["types"].items()
                           if leg["host"]["delivered"] >= WORKING_RATIO * params.probe_count]
                data["working_types"] = working
                if not working:
                    failures.append("neither type carried the stream round the loop")
                    raise _Stop
                use = working[0] if len(working) == 1 else saved_type
                data["type_used"] = use
                tick(f"type {use} again")
                verbs.midi_trs(link, f"TYPE {use}")
                time.sleep(0.1)
                again = self._leg(ctx, link, port, "host", params.probe_count, probe_rate)
                data["type_again"] = again
                if again["host"]["lost"] or again["host"]["spurious"] or again["esp"]["lost"]:
                    failures.append(f"type {use} after the switch: host lost "
                                    f"{again['host']['lost']}, spurious "
                                    f"{again['host']['spurious']}, ESP lost {again['esp']['lost']}")

                # -- throughput -------------------------------------------------
                for r in rates:
                    tick(f"host {r:g}/s")
                    n = min(int(r * params.seconds), PERIOD - 64)
                    leg = self._leg(ctx, link, port, "host", n, r)
                    data["throughput"].append(leg)
                    if r <= 0.9 * LINE_MSGS_PER_S and (leg["host"]["lost"] or leg["esp"]["lost"]):
                        failures.append(f"host {r:g}/s under the line rate: host lost "
                                        f"{leg['host']['lost']}, ESP lost {leg['esp']['lost']}")
                for r in esp_rates:
                    tick(f"ESP {r:g}/s")
                    n = min(int(r * params.seconds), PERIOD - 64)
                    leg = self._leg(ctx, link, port, "esp", n, r)
                    data["throughput"].append(leg)
                    if r <= 0.9 * LINE_MSGS_PER_S and (leg["host"]["lost"] or leg["esp"]["lost"]):
                        failures.append(f"ESP {r:g}/s under the line rate: host lost "
                                        f"{leg['host']['lost']}, ESP lost {leg['esp']['lost']}")

                # -- physical bit errors: TX baud skew ---------------------------
                n_err = min(int(params.error_rate * params.error_seconds), PERIOD - 64)
                for sk in skews:
                    tick(f"skew {sk / 10:+.1f} %")
                    verbs.midi_trs(link, f"SKEW {sk}")
                    leg = self._leg(ctx, link, port, "host", n_err, params.error_rate)
                    errored = leg["host"]["lost"] + leg["host"]["spurious"]
                    leg["skew_pct"] = sk / 10
                    leg["ber_est"] = ber_from_mer(errored, leg["host"]["sent"])
                    data["skew"].append(leg)
                verbs.midi_trs(link, "SKEW 0")

                # -- injected bit errors per segment ----------------------------
                for seg in segs:
                    code, source, counter = SEGMENTS[seg]
                    for one in ones:
                        tick(f"{seg} 1/{one}")
                        verbs.midi_trs(link, f"FLIP {code} {one}")
                        leg = self._leg(ctx, link, port, source, n_err, params.error_rate)
                        bits = 8 * int(leg["stm"].get(counter, 0))
                        flips = int(leg["stm"].get("flips", 0))
                        leg.update({"segment": seg, "one_in": one, "ber_set": 1.0 / one,
                                    "ber_measured": flips / bits if bits else None})
                        data["flips"].append(leg)
                    verbs.midi_trs(link, "FLIP 0 0")

                # -- clean run at the end: MIDI IN survived all of it ------------
                tick("final clean run")
                final = self._leg(ctx, link, port, "host", n_err, params.error_rate)
                data["final"] = final
                if final["host"]["lost"] or final["host"]["spurious"] or final["esp"]["lost"]:
                    failures.append(f"final clean run: host lost {final['host']['lost']}, "
                                    f"spurious {final['host']['spurious']}, "
                                    f"ESP lost {final['esp']['lost']}")
            except _Stop:
                pass
            finally:
                for undo in ("FLIP 0 0", "SKEW 0", f"TYPE {saved_type}", "ROUTE SAVED"):
                    try:
                        verbs.midi_trs(link, undo)
                    except HilError:
                        pass
                try:
                    verbs.midi_loop(link, "CAPTURE 0")
                except HilError:
                    pass

            progress(steps, steps, "done")
            passed = not failures
            ceiling = max((leg["host"]["goodput"] for leg in data["throughput"]
                           if leg["source"] == "host"), default=0.0)
            summary = (f"midi_trs_loop: {'PASS' if passed else 'FAIL'} — types carrying the loop "
                       f"{data.get('working_types')}, host goodput up to {ceiling:g} msg/s "
                       f"(line {LINE_MSGS_PER_S:.0f})"
                       + ("" if passed else " — " + "; ".join(failures)))
            data["failures"] = failures
            return Report(passed=passed, summary=summary, data=data, artifacts=[],
                          exit_code=0 if passed else 1)
        finally:
            ctx.close_all()


class _Stop(Exception):
    """Nothing further can be measured (no type carries the stream)."""


SCENARIO = register(MidiTrsLoopScenario())
