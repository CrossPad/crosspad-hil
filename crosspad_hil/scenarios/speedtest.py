"""How fast can the pads be played before the device stops keeping up?

Port of ``platform-idf/tools/hil_speedtest.py``. The hit rate is ramped and
every step reports how far each hit got along the trigger chain:

    host sent  ->  firmware accepted  ->  the engine started a voice
                   (PAD_STATS press)      (PAD_STATS played)

A gap between *sent* and *accepted* is transport loss (USB MIDI, the STM
bridge, the ESP's LPUART). A gap between *accepted* and *played* is loss inside
the device: EventBus overflow, or no free voice slot. Free slots are sampled
too, so polyphony exhaustion shows up directly instead of being inferred.

Pad sets are swept separately (1 pad, 4 pads, all 16) because the cost is not
per hit alone: each distinct pad holds its own voice and its own LED, so 16
pads at rate R is a different load from 1 pad at rate R.

Hits go over the **STM bridge as SysEx**, never over CDC: a hit is two CDC
commands, ``app_queue`` is 64 deep and drops on overflow, so a CDC sweep would
measure the host's command path rather than the device's pad path. Counters
still come off CDC, so the board has to be in the default USB profile —
which is also why this composes with ``velocity`` rather than replacing it.
That one proves the hits are audible; this one proves they arrive.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

from crosspad_hil import verbs
from crosspad_hil.errors import HilError
from crosspad_hil.scenarios.base import (
    ENV_EXIT_CODES,
    Artifact,
    Context,
    Progress,
    Report,
    register,
)
from crosspad_hil.stim import parse_pads, sweep

SAMPLER_APP = "Sampler"
# hil_speedtest.main: time.sleep(7) after KIT_LOAD, time.sleep(3) after APP_START.
# Both are waits on state here (verbs.kit_load polls KIT_STATUS, app_start polls
# APP_LIST), so the sleeps are gone rather than reproduced.
TRANSPORT = "sysex_stm"
DEFAULT_PADSETS = "0;0-3;0-15"
DEFAULT_RATES = [4, 8, 12, 16, 24, 32, 48, 64, 96, 128]


@dataclass
class Params:
    kit: int = field(default=0, metadata={"help": "kit id to load before playing"})
    padsets: str = field(
        default=DEFAULT_PADSETS,
        metadata={"help": "semicolon-separated pad sets, e.g. '0;0-3;0-15'"})
    rates: list[int] = field(
        default_factory=lambda: list(DEFAULT_RATES),
        metadata={"help": "hits per second to sweep"})
    duration: float = field(default=5.0, metadata={"help": "seconds per rate step"})
    settle: float = field(
        default=1.5, metadata={"help": "seconds to let the queues drain before reading "
                                       "the counters"})
    vel: int = field(default=100, metadata={"help": "velocity of every hit"})
    gate_ms: float = field(default=25.0, metadata={"help": "press->release gap"})
    chord: bool = field(
        default=False,
        metadata={"help": "hit every pad of the set at once (polyphony) instead of "
                          "round-robin (throughput)"})
    loss_threshold: float = field(
        default=1.0, metadata={"help": "percent loss that counts as the device's limit"})
    full_sweep: bool = field(
        default=False,
        metadata={"help": "run every rate even past the limit — the shape of the "
                          "degradation rather than only its onset"})
    repeat: int = field(
        default=1, metadata={"help": "repeat the whole sweep N times (flakiness)"})
    start_app: bool = field(
        default=True,
        metadata={"help": "load the kit and open the Sampler first; --no-start-app leaves "
                          "it closed, because the sampler is the system instrument and the "
                          "pads play without it — that path is worth testing too"})
    require_rate: int | None = field(
        default=None,
        metadata={"help": "fail when a pad set's limit is below this many hits/s; unset "
                          "reports the limits without judging them"})


def _env(summary: str, data: dict[str, Any], artifacts: list[Artifact]) -> Report:
    return Report(passed=False, summary=summary, data=data, artifacts=artifacts, exit_code=2)


def parse_padsets(spec: str) -> list[list[int]]:
    """``"0;0-3;0-15"`` -> three pad sets (hil_speedtest.main splits on ';')."""
    return [parse_pads(part) for part in spec.split(";") if part.strip()]


def limit_table(limits: list[dict[str, Any]]) -> dict[str, float | None]:
    """Pad set -> the lowest rate at which it lost hits, or None if it never did.

    The lowest across repeats, not the last: a rate that failed once is a rate
    the device does not sustain, and averaging that away is how a flaky limit
    gets reported as a solid one.
    """
    out: dict[str, float | None] = {}
    for row in limits:
        label = str(row["padset"])
        rate = row["limit_rate_hz"]
        if label not in out:
            out[label] = rate
        elif rate is not None and (out[label] is None or rate < out[label]):
            out[label] = rate
    return out


class SpeedtestScenario:
    name = "speedtest"
    Params = Params
    description = "ramp the pad hit rate until the device drops hits, per pad set"

    def run(self, ctx: Context, params: Params, progress: Progress) -> Report:
        try:
            p = params
            rates = [float(r) for r in p.rates]
            # base._coerce only special-cases the bare scalar types, so an
            # `int | None` field arrives from the daemon as a string — and a
            # string compares to no rate at all.
            require = int(p.require_rate) if p.require_rate is not None else None
            data: dict[str, Any] = {
                "kit": p.kit, "mode": "chord" if p.chord else "round-robin",
                "transport": TRANSPORT, "app_open": p.start_app,
                "duration_s": p.duration, "velocity": p.vel, "gate_ms": p.gate_ms,
                "loss_threshold_pct": p.loss_threshold, "repeat": p.repeat,
                "rates": rates, "padsets": [], "rows": [], "limits": [],
                "limit_hits_per_s": {}, "require_rate": require,
            }
            artifacts: list[Artifact] = []
            if not rates:
                return _env("speedtest: no rates to sweep", data, artifacts)
            try:
                pad_sets = parse_padsets(p.padsets)
            except HilError as e:
                return _env(f"speedtest: {e.message}", data, artifacts)
            if not pad_sets:
                return _env(f"speedtest: no pad sets in {p.padsets!r}", data, artifacts)
            data["padsets"] = [list(s) for s in pad_sets]
            if ctx.device.ports.stm_midi is None:
                return _env(f"speedtest: {ctx.device.id} has no STM bridge MIDI port — the "
                            "hits (SysEx 0x1D 07) have nothing to travel on", data, artifacts)
            ctx.check_cancelled()

            # Raises NO_CDC_IN_AUDIO_MODE (exit 2) when the board is in the audio
            # profile, which is the honest answer: the counters live on CDC.
            link = ctx.open_cdc()
            try:
                verbs.kit_load(link, p.kit)
            except HilError as e:
                return _env(f"speedtest: KIT_LOAD {p.kit} failed — {e.code}: {e.message} "
                            "(firmware without hil_control, or no SD card)", data, artifacts)
            if p.start_app:
                try:
                    verbs.app_start(link, SAMPLER_APP)
                except HilError as e:
                    return _env(f"speedtest: {SAMPLER_APP} would not start — "
                                f"{e.code}: {e.message}", data, artifacts)
            apps = verbs.app_list(link)
            ctx.log(f"kit {p.kit}, running={apps.get('running')}, "
                    f"{len(pad_sets)} pad set(s), rates {rates[0]:.0f}..{rates[-1]:.0f}/s")

            steps = len(rates) * len(pad_sets)
            total = steps * max(1, p.repeat)
            try:
                for rep in range(max(1, p.repeat)):
                    ctx.check_cancelled()
                    out = sweep(
                        ctx.device, rates=rates, pad_sets=pad_sets, transports=(TRANSPORT,),
                        seconds=p.duration, settle_s=p.settle, velocity=p.vel,
                        gate_ms=p.gate_ms, chord=p.chord,
                        loss_threshold_pct=p.loss_threshold, full_sweep=p.full_sweep,
                        cancelled=ctx.cancelled, link=link,
                        progress=self._progress(progress, rep, steps, total))
                    for row in out["rows"]:
                        row["rep"] = rep
                        data["rows"].append(row)
                        self._log_row(ctx, row)
                    for lim in out["limits"]:
                        lim["rep"] = rep
                        data["limits"].append(lim)
            except HilError as e:
                if e.code in ENV_EXIT_CODES:
                    raise
                # hil_speedtest: "counters unreadable — device may have reset". It
                # kept its exit code at 0; a sweep whose counters stopped answering
                # measured nothing, so it fails here.
                data["error"] = e.to_dict()
                return self._emit(ctx, data, artifacts, passed=False,
                                  summary=f"speedtest: FAIL — the device stopped answering "
                                          f"mid-sweep ({e.code}: {e.message}); it may have "
                                          f"reset")

            data["limit_hits_per_s"] = limit_table(data["limits"])
            return self._verdict(ctx, require, data, artifacts)
        finally:
            ctx.close_all()

    def _progress(self, progress: Progress, rep: int, steps: int, total: int):
        def sink(done: int, _steps: int | None, message: str) -> None:
            progress(rep * steps + done, total, message)
        return sink

    def _log_row(self, ctx: Context, row: dict[str, Any]) -> None:
        # `loss` is None when the counters were unreadable, and printing that as a
        # zero would claim a clean step that was never measured.
        loss = row.get("loss") or {}
        cell = {k: str(loss.get(k, "?")) for k in ("sent", "accepted", "played", "total_pct")}
        ctx.log(f"  {row['padset']:>12} {row['rate_hz']:>6.0f}/s  "
                f"sent={cell['sent']:<6} accepted={cell['accepted']:<6} "
                f"played={cell['played']:<6} loss={cell['total_pct']}%  "
                f"slots={row.get('freeslots')}")

    # --- verdict -------------------------------------------------------------
    def _verdict(self, ctx: Context, require_rate: int | None, data: dict[str, Any],
                 artifacts: list[Artifact]) -> Report:
        table = data["limit_hits_per_s"]
        top = data["rates"][-1]
        parts = [f"{label}: {rate:.0f} hits/s" if rate is not None
                 else f"{label}: no loss up to {top:.0f} hits/s"
                 for label, rate in table.items()]
        # hil_speedtest returned 0 whatever the limits were — it reports where the
        # device stops keeping up, it does not decide that a limit is too low.
        # `require_rate` is the opt-in that turns that judgement on.
        short = [f"{label} limited at {rate:.0f} hits/s" for label, rate in table.items()
                 if rate is not None and require_rate is not None and rate < require_rate]
        if short:
            return self._emit(ctx, data, artifacts, passed=False,
                              summary=f"speedtest: FAIL — below the required "
                                      f"{require_rate} hits/s: " + "; ".join(short))
        return self._emit(ctx, data, artifacts, passed=True,
                          summary="speedtest: PASS — " + ("; ".join(parts) or "nothing swept"))

    def _emit(self, ctx: Context, data: dict[str, Any], artifacts: list[Artifact], *,
              passed: bool, summary: str) -> Report:
        path = ctx.workdir / "speedtest.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(data, indent=2, default=str) + "\n", encoding="utf-8")
        artifacts.append(Artifact(path=str(path), mime="application/json", role="analysis"))
        return Report(passed=passed, summary=summary, data=data, artifacts=artifacts,
                      exit_code=0 if passed else 1)


SCENARIO = register(SpeedtestScenario())


def main(argv: list[str] | None = None) -> int:
    """Entry point for the platform-idf shim `tools/hil_speedtest.py`."""
    import sys

    from crosspad_hil.cli import main as cli_main

    args = list(sys.argv[1:] if argv is None else argv)
    return cli_main(["run", "speedtest", *args])
