"""Dual-path USB MIDI throughput and latency — ESP native vs the STM bridge.

Port of ``platform-idf/tools/hil_midi_bench.py``. The board takes host MIDI two
ways: straight into TinyUSB on the ESP, or through the STM32 bridge which
relays it over LPUART2. Both converge in the ESP's ``uartLoop``, so the only
way to tell them apart is to tag the traffic — which the echo frame (0x1D 0x09)
already allows: its payload is a 28-bit sequence number, and bit 27 marks the
frames injected through the bridge.

Three phases, in order: ESP alone, STM alone, both at once. The third is the
one that matters — a relay that keeps up on its own and starves when the native
path is busy is a scheduling bug, and it is invisible from either of the first
two phases.

Replies always come back on the ESP native MIDI IN port whichever way the query
went in, so one reader serves all three phases.
"""
from __future__ import annotations

import statistics
import threading
import time
from dataclasses import dataclass, field
from typing import Any

from crosspad_hil import knowledge
from crosspad_hil.errors import NO_DEVICE, HilError
from crosspad_hil.midi import MidiIO, MidiRole, decode_echo_seq, sysex_echo
from crosspad_hil.scenarios.base import Artifact, Context, Progress, Report, register

SRC_ESP = 0
SRC_STM = 1
SRC_BIT = 1 << 27          # hil_midi_bench: set for STM-injected frames
ECHO_FRAME_BYTES = 9       # F0 7D 1D 09 s s s s F7
READER_SETTLE_S = 0.3      # hil_midi_bench.main: time.sleep(0.3) after starting the reader
FLOOD_RATE = 2000.0        # `int(2000 * args.duration)` when --flood is set
PHASES: tuple[tuple[str, bool, bool], ...] = (
    ("esp_only", True, False),
    ("stm_only", False, True),
    ("both", True, True),
)


@dataclass
class Params:
    rate: float = field(default=200.0, metadata={"help": "messages per second per path"})
    duration: float = field(default=10.0, metadata={"help": "seconds per phase"})
    flood: bool = field(
        default=False, metadata={"help": "ignore rate; write as fast as the port accepts"})
    settle: float = field(
        default=2.0, metadata={"help": "seconds to wait for stragglers after each phase"})


class _Reader(threading.Thread):
    """Timestamps every echo reply on the ESP MIDI IN port.

    One reader for the whole run rather than one per phase: replies from a
    phase's tail can arrive after the phase's own sender has stopped, and a
    reader torn down with it would count them as loss.
    """

    def __init__(self, io: MidiIO, route_id: int, echo_sub: int) -> None:
        super().__init__(name="midi_bench_reader", daemon=True)
        self._io = io
        self._route_id = route_id
        self._echo_sub = echo_sub
        self._halt = threading.Event()
        self._lock = threading.Lock()
        self.recv: dict[int, float] = {}

    def run(self) -> None:
        while not self._halt.is_set():
            frame = self._io.receive(0.05)
            if frame is None:
                continue
            now = time.monotonic()
            if len(frame) >= ECHO_FRAME_BYTES and frame[2] == self._route_id \
                    and frame[3] == self._echo_sub:
                seq = decode_echo_seq(frame)
                with self._lock:
                    self.recv.setdefault(seq, now)

    def take(self) -> dict[int, float]:
        with self._lock:
            out = dict(self.recv)
            self.recv.clear()
        return out

    def stop(self) -> None:
        self._halt.set()


class _Sender(threading.Thread):
    """Sequence-stamped echo frames down one MIDI out, at a rate or flat out."""

    def __init__(self, io: MidiIO, src: int, count: int, rate: float, flood: bool) -> None:
        super().__init__(name=f"midi_bench_tx_{src}", daemon=True)
        self._io = io
        self.src = src
        self._count = count
        self._rate = rate
        self._flood = flood
        self.sent: dict[int, float] = {}
        self.errors = 0

    def run(self) -> None:
        period = 0.0 if self._flood else 1.0 / max(self._rate, 1e-6)
        t_next = time.monotonic()
        base = SRC_BIT if self.src == SRC_STM else 0
        for i in range(self._count):
            seq = base | i
            self.sent[seq] = time.monotonic()
            try:
                self._io.send_sysex(sysex_echo(seq))
            except Exception:  # noqa: BLE001 — a refused write is loss, not a crash
                self.errors += 1
            if period:
                t_next += period
                dt = t_next - time.monotonic()
                if dt > 0:
                    time.sleep(dt)


def summarise(sent: dict[int, float], recv: dict[int, float], tx_seconds: float) -> dict[str, Any]:
    """hil_midi_bench.phase's per-path block: loss, tx rate, latency percentiles."""
    lat: list[float] = []
    lost = 0
    for seq, t_send in sent.items():
        t_recv = recv.get(seq)
        if t_recv is None:
            lost += 1
        else:
            lat.append((t_recv - t_send) * 1000.0)
    n = len(sent)
    bps = n * ECHO_FRAME_BYTES / max(tx_seconds, 1e-3)
    res: dict[str, Any] = {
        "sent": n,
        "received": n - lost,
        "lost": lost,
        "loss_pct": round(100.0 * lost / n, 2) if n else 0.0,
        "tx_kbs": round(bps / 1024.0, 1),
    }
    if lat:
        lat.sort()
        res["lat_ms"] = {
            "min": round(lat[0], 1),
            "med": round(statistics.median(lat), 1),
            "p95": round(lat[max(0, int(len(lat) * 0.95) - 1)], 1),
            "max": round(lat[-1], 1),
        }
    return res


class MidiBenchScenario:
    name = "midi_bench"
    Params = Params
    description = "echo-frame throughput and latency on the ESP port, the STM bridge, and both"

    def run(self, ctx: Context, params: Params, progress: Progress) -> Report:
        try:
            p = params
            k = knowledge.load("sysex")
            route_id = int(k["audio_route"]["id"])
            echo_sub = int(k["audio_route"]["subs"]["echo"])
            data: dict[str, Any] = {
                "config": {"rate": p.rate, "duration": p.duration,
                           "flood": p.flood, "settle": p.settle},
                "phases": {}, "worst_loss_pct": 0.0,
            }
            if ctx.device.ports.esp_midi is None or ctx.device.ports.stm_midi is None:
                raise HilError(
                    NO_DEVICE,
                    f"{ctx.device.id}: both MIDI entries are needed "
                    f"(esp={ctx.device.ports.esp_midi is not None}, "
                    f"stm={ctx.device.ports.stm_midi is not None})",
                    hint="the board must be in the default CDC+MIDI profile",
                    device=ctx.device.id)

            count = int((FLOOD_RATE if p.flood else p.rate) * p.duration)
            if count <= 0:
                count = 1

            esp = MidiIO(ctx.device, MidiRole.ESP)
            stm = MidiIO(ctx.device, MidiRole.STM)
            esp.open()
            ctx.opened.append(esp)
            stm.open()
            ctx.opened.append(stm)
            reader = _Reader(esp, route_id, echo_sub)
            try:
                esp.drain()
                reader.start()
                ctx.cancelled.wait(READER_SETTLE_S)
                ctx.check_cancelled()
                for i, (name, use_esp, use_stm) in enumerate(PHASES, start=1):
                    ctx.check_cancelled()
                    progress(i - 1, len(PHASES), f"phase {name}")
                    data["phases"][name] = self._phase(
                        ctx, name, esp, stm, reader, use_esp, use_stm, count, p)
                    progress(i, len(PHASES), f"phase {name} done")
            finally:
                reader.stop()
                reader.join(timeout=2.0)
                esp.close()
                stm.close()

            worst = max((path["loss_pct"] for phase in data["phases"].values()
                         for path in phase.values()), default=0.0)
            data["worst_loss_pct"] = worst
            bench = ctx.workdir / "bench.json"
            bench.parent.mkdir(parents=True, exist_ok=True)
            import json
            bench.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
            artifacts = [Artifact(path=str(bench), mime="application/json", role="bench")]

            passed = worst == 0.0
            summary = (f"midi_bench: {'PASS' if passed else 'FAIL'} — worst loss {worst}% "
                       + "; ".join(
                           f"{ph} {path}={res['loss_pct']}%"
                           for ph, block in data["phases"].items()
                           for path, res in block.items()))
            return Report(passed=passed, summary=summary, data=data, artifacts=artifacts,
                          exit_code=0 if passed else 1)
        finally:
            ctx.close_all()

    def _phase(self, ctx: Context, name: str, esp: MidiIO, stm: MidiIO, reader: _Reader,
               use_esp: bool, use_stm: bool, count: int, p: Params) -> dict[str, Any]:
        ctx.log(f"--- phase {name}: rate={'flood' if p.flood else p.rate}/s "
                f"count={count} per path ---")
        senders: list[_Sender] = []
        if use_esp:
            senders.append(_Sender(esp, SRC_ESP, count, p.rate, p.flood))
        if use_stm:
            senders.append(_Sender(stm, SRC_STM, count, p.rate, p.flood))
        t0 = time.monotonic()
        for s in senders:
            s.start()
        for s in senders:
            s.join()
        tx_seconds = time.monotonic() - t0
        # Stragglers: a reply in flight when the last frame went out is not loss.
        ctx.cancelled.wait(p.settle)
        recv = reader.take()

        out: dict[str, Any] = {}
        for s in senders:
            path = "esp" if s.src == SRC_ESP else "stm"
            res = summarise(s.sent, recv, tx_seconds)
            res["send_errors"] = s.errors
            out[path] = res
            lat = res.get("lat_ms")
            ctx.log(f"  {path}: sent={res['sent']} lost={res['lost']} ({res['loss_pct']}%) "
                    f"tx={res['tx_kbs']} KB/s "
                    + (f"lat med={lat['med']}ms p95={lat['p95']}ms max={lat['max']}ms"
                       if lat else "lat n/a"))
        return out


SCENARIO = register(MidiBenchScenario())


def main(argv: list[str] | None = None) -> int:
    """Entry point for the platform-idf shim `tools/hil_midi_bench.py`."""
    import sys

    from crosspad_hil.cli import main as cli_main

    args = list(sys.argv[1:] if argv is None else argv)
    return cli_main(["run", "midi_bench", *args])
