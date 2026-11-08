"""Blast audio-route QUERY frames at the ESP MIDI port and count what came back.

Port of ``platform-idf/tools/hil_midi_stress.py``. Every query the firmware
parses answers with exactly one 14-byte reply, so

    loss = sent - replies

is end-to-end USB MIDI RX loss: TinyUSB FIFO -> byte queue -> parser. Nothing
else on the device is exercised, which is the point — a pad stimulus that goes
missing could be lost anywhere between the host and the sample engine, and this
narrows it to the receive path alone.

Two things the old script established and this keeps:

* the **sanity query first**. One frame, one reply, before the burst. If that
  does not come back the port or the firmware is wrong and the burst's numbers
  would be noise about the wrong thing — reported as an environment failure
  (exit 2), not as loss.
* the **console is read in parallel**. "USB MIDI receive queue full" is the
  firmware saying which end dropped it, and it only ever appears on the STM
  VCP. A board with no console still runs; it just cannot say why.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

from crosspad_hil import knowledge
from crosspad_hil.errors import ENV, HilError
from crosspad_hil.midi import MidiIO, MidiRole, sysex_query
from crosspad_hil.scenarios.base import Artifact, Context, Progress, Report, register

# hil_midi_stress.main: time.sleep(0.5) after starting the receiver, 1.0 after
# the sanity query.
RX_SETTLE_S: float = 0.5
SANITY_WAIT_S: float = 1.0
# The console lines the old script kept: anything about the MIDI path.
CONSOLE_MATCH: str = r"queue full|MIDI"
CONSOLE_LINES: int = 10
QUERY_BYTES: int = 5


@dataclass
class Params:
    count: int = field(default=200, metadata={"help": "queries in the burst"})
    settle: float = field(
        default=3.0, metadata={"help": "seconds to keep reading replies after the burst"})
    console: bool = field(
        default=True,
        metadata={"help": "watch the STM VCP for 'receive queue full' while the burst runs"})


def _reply_matcher() -> tuple[int, int]:
    k = knowledge.load("sysex")
    return int(k["audio_route"]["id"]), int(k["audio_route"]["subs"]["query"])


def _drain_replies(io: MidiIO, seconds: float, cancelled: Any = None) -> int:
    """Count query replies arriving within ``seconds``. Returns the count."""
    route_id, query_sub = _reply_matcher()
    deadline = time.monotonic() + seconds
    seen = 0
    while True:
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            return seen
        if cancelled is not None and cancelled.is_set():
            return seen
        frame = io.receive(min(remaining, 0.25))
        if frame is None:
            continue
        if len(frame) >= 5 and frame[2] == route_id and frame[3] == query_sub:
            seen += 1


class MidiStressScenario:
    name = "midi_stress"
    Params = Params
    description = "burst SysEx queries at the ESP MIDI port and count the replies (RX loss)"

    def run(self, ctx: Context, params: Params, progress: Progress) -> Report:
        try:
            p = params
            data: dict[str, Any] = {
                "sent": 0, "received": 0, "lost": 0, "loss_pct": 0.0,
                "sanity": 0, "replies_total": 0, "burst_seconds": 0.0, "tx_kbs": 0.0,
                "console": [], "port": None,
            }
            artifacts: list[Artifact] = []
            if p.count <= 0:
                raise HilError(ENV, f"count must be >= 1, got {p.count}")

            console = None
            if p.console and ctx.device.ports.console is not None:
                console = ctx.open_console(reset=False)
                if console.log_path is not None:
                    artifacts.append(
                        Artifact(path=str(console.log_path), mime="text/plain", role="console"))
            console_mark = console.seq if console is not None else 0

            io = MidiIO(ctx.device, MidiRole.ESP)
            io.open()
            ctx.opened.append(io)
            data["port"] = io.port_name
            try:
                io.drain()
                ctx.cancelled.wait(RX_SETTLE_S)
                ctx.check_cancelled()

                # --- sanity: one query must produce exactly one reply -----------
                progress(0, 3, "sanity query")
                frame = sysex_query()
                io.send_sysex(frame)
                sanity = _drain_replies(io, SANITY_WAIT_S, ctx.cancelled)
                data["sanity"] = sanity
                ctx.check_cancelled()
                if sanity < 1:
                    data["replies_total"] = 0
                    return Report(
                        passed=False,
                        summary="midi_stress: no reply to a single query — wrong port, or "
                                "firmware without the 0x1D audio-route command",
                        data=data, artifacts=artifacts, exit_code=2)

                # --- the burst: as fast as the port accepts it ------------------
                progress(1, 3, f"bursting {p.count} queries")
                t0 = time.monotonic()
                for i in range(p.count):
                    io.send_sysex(frame)
                    if (i & 0x3F) == 0:
                        ctx.check_cancelled()
                burst_s = max(time.monotonic() - t0, 1e-3)
                nbytes = p.count * QUERY_BYTES
                data["burst_seconds"] = round(burst_s, 3)
                data["tx_kbs"] = round(nbytes / burst_s / 1024.0, 1)
                ctx.log(f"burst: {p.count} queries ({nbytes} B) written in {burst_s:.3f}s "
                        f"({data['tx_kbs']} KB/s)")

                progress(2, 3, f"collecting replies for {p.settle}s")
                burst_replies = _drain_replies(io, p.settle, ctx.cancelled)
            finally:
                io.close()

            data["sent"] = p.count
            data["received"] = burst_replies
            data["lost"] = p.count - burst_replies
            data["loss_pct"] = round(100.0 * data["lost"] / p.count, 2)
            data["replies_total"] = burst_replies + data["sanity"]

            if console is not None:
                hits = console.read(since_seq=console_mark + 1, match=CONSOLE_MATCH,
                                    limit=CONSOLE_LINES)
                data["console"] = [line for _seq, line in hits.lines]
                for line in data["console"]:
                    ctx.log(f"console: {line.strip()}")

            progress(3, 3, "done")
            passed = data["lost"] <= 0
            summary = (f"midi_stress: {'PASS' if passed else 'FAIL'} — sent={data['sent']} "
                       f"received={data['received']} lost={data['lost']} "
                       f"({data['loss_pct']}%)")
            return Report(passed=passed, summary=summary, data=data, artifacts=artifacts,
                          exit_code=0 if passed else 1)
        finally:
            ctx.close_all()


SCENARIO = register(MidiStressScenario())


def main(argv: list[str] | None = None) -> int:
    """Entry point for the platform-idf shim `tools/hil_midi_stress.py`."""
    import sys

    from crosspad_hil.cli import main as cli_main

    args = list(sys.argv[1:] if argv is None else argv)
    return cli_main(["run", "midi_stress", *args])
