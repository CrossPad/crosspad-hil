"""Does every pad get its waveform? Load kits and watch the cache fill.

"Not all waveforms show up" is a bug you can only see with a kit on the
screen and your finger on each pad, which is why it lived for so long. The
firmware now reports the cache itself (WAVE_STATUS: every sample layer of
every pad in the current kit, cached or not, plus the loader's counters), so
this scenario asks the question the way a user would notice it:

  for each kit:  KIT_LOAD -> wait for the kit -> wait for every layer cached

and, with --play, does it while the pads keep firing — the sample streamer and
the waveform loader share the card's FAT descriptor pool, and a load that
only succeeds from silence is not the thing users hit.

A kit passes when `missing` is empty within --timeout. A kit fails when it is
not, and the report says which pad:layer never arrived and what the loader
counted meanwhile (retries, permanent open/read failures, dropped requests) —
enough to tell "the file is not there" from "the loader never got it".

Exit code 0 = every kit fully cached, 1 = at least one layer missing (or a kit
that never became current), 2 = no HIL control / no kits on the card.
"""
from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field
from typing import Any

from crosspad_hil import verbs
from crosspad_hil.cdc import CdcLink
from crosspad_hil.errors import TIMEOUT, HilError
from crosspad_hil.scenarios import register
from crosspad_hil.scenarios.base import Context, Progress, Report

SAMPLER_APP = "Sampler"
PAD_COUNT = 16
POLL_S = 0.5


@dataclass
class Params:
    kits: list[int] | None = field(
        default=None,
        metadata={"help": "comma-separated kit ids; default: every kit KIT_LIST reports"},
    )
    timeout: float = field(
        default=30.0, metadata={"help": "seconds a kit gets to have every layer cached"}
    )
    load_timeout: float = field(default=20.0, metadata={"help": "seconds for KIT_LOAD to land"})
    play: bool = field(
        default=False,
        metadata={"help": "keep the pads firing while the cache fills "
                         "(the streamer competes for the card)"},
    )
    hit_rate: float = field(default=8.0, metadata={"help": "hits/s across the pads with --play"})
    settle: float = field(
        default=1.0, metadata={"help": "seconds after a load before watching the cache"})


class _Player(threading.Thread):
    """PAD_PRESS/PAD_RELEASE round-robin, like kit_churn's."""

    def __init__(self, link: CdcLink, hit_rate: float) -> None:
        super().__init__(name="waveform_cache_player", daemon=True)
        self._link = link
        self._period = 1.0 / max(hit_rate, 0.1)
        self._halt = threading.Event()
        self.hits = 0

    def run(self) -> None:
        pad = 0
        while not self._halt.is_set():
            try:
                verbs.pad_press(self._link, pad, 100)
                verbs.pad_release(self._link, pad)
                self.hits += 1
            except HilError:
                pass
            pad = (pad + 1) % PAD_COUNT
            self._halt.wait(self._period)

    def stop(self) -> None:
        self._halt.set()
        self.join(timeout=2.0)


def _wait_cached(ctx: Context, link: CdcLink, timeout_s: float) -> tuple[dict[str, Any], float]:
    """Poll WAVE_STATUS until nothing is missing or the deadline passes."""
    t0 = time.monotonic()
    deadline = t0 + timeout_s
    st = verbs.wave_status(link)
    while st["missing"] and time.monotonic() < deadline:
        ctx.check_cancelled()
        # Wait on the cancel flag, not the clock: a kit that never finishes
        # caching holds this loop for the whole timeout, and Ctrl-C has to land
        # in a poll interval rather than at the end of it.
        ctx.cancelled.wait(POLL_S)
        st = verbs.wave_status(link)
    return st, time.monotonic() - t0


def _env(summary: str, data: dict[str, Any]) -> Report:
    return Report(passed=False, summary=summary, data=data, artifacts=[], exit_code=2)


class WaveformCacheScenario:
    name = "waveform_cache"
    Params = Params
    description = ("load kits and check every pad layer gets its waveform cached; "
                   "--play keeps the pads firing")

    def run(self, ctx: Context, params: Params, progress: Progress) -> Report:
        p = params
        data: dict[str, Any] = {
            "mode": "play" if p.play else "silent", "kits": [], "failures": [],
            "hits_sent": 0,
        }
        player: _Player | None = None
        try:
            link = ctx.open_cdc()
            try:
                kl = verbs.kit_list(link)
            except HilError as e:
                return _env(f"waveform_cache: KIT_LIST failed ({e.code}: {e.message})", data)
            kits = [{"id": int(k["id"]), "name": str(k["name"])} for k in kl.get("kits", [])]
            if not kits:
                return _env("waveform_cache: no kits on the card", data)
            names = {k["id"]: k["name"] for k in kits}
            cycle = list(p.kits) if p.kits else [k["id"] for k in kits]
            unknown = [k for k in cycle if k not in names]
            if unknown:
                return _env(f"waveform_cache: no such kit id: {unknown}", data)
            try:
                verbs.wave_status(link)
            except HilError as e:
                return _env(
                    f"waveform_cache: WAVE_STATUS unsupported ({e.code}) — "
                    "firmware predates it", data)

            if p.play:
                apps = verbs.app_list(link)
                if apps.get("running") != SAMPLER_APP:
                    # The gate: a kit must be current before the Sampler launches.
                    verbs.kit_load(link, cycle[0], wait_s=p.load_timeout)
                    verbs.app_start(link, SAMPLER_APP)
                player = _Player(link, p.hit_rate)
                player.start()
                ctx.log(f"playing {p.hit_rate:.0f} hits/s across the pads while the cache fills")

            failures: list[str] = []
            for n, kit_id in enumerate(cycle, 1):
                ctx.check_cancelled()
                name = names[kit_id]
                progress(n, len(cycle), f"{name}: loading")
                try:
                    verbs.kit_load(link, kit_id, wait_s=p.load_timeout)
                except HilError as e:
                    if e.code != TIMEOUT:
                        raise
                    failures.append(f"{name}: never became current ({e.message})")
                    data["kits"].append({"id": kit_id, "name": name, "landed": False})
                    continue
                ctx.cancelled.wait(p.settle)
                st, seconds = _wait_cached(ctx, link, p.timeout)
                row = {"id": kit_id, "name": name, "landed": True, "seconds": round(seconds, 2),
                       "total": st["total"], "cached": st["cached"], "missing": st["missing"],
                       "loader": {k: st[k] for k in ("queued", "req", "ok", "retry",
                                                     "fail_info",
                                                     "fail_read", "alloc_fail", "drop")}}
                data["kits"].append(row)
                if st["missing"]:
                    miss = ", ".join(f"{pad}:{layer}" for pad, layer in st["missing"])
                    failures.append(
                        f"{name}: {len(st['missing'])}/{st['total']} layer(s) never cached "
                        f"after {p.timeout:.0f}s — pad:layer {miss}; loader queued={st['queued']} "
                        f"retry={st['retry']} fail_info={st['fail_info']} "
                        f"fail_read={st['fail_read']} "
                        f"drop={st['drop']}")
                    ctx.log(f"  {name:<20} {st['cached']}/{st['total']} MISSING {miss}")
                else:
                    ctx.log(f"  {name:<20} {st['cached']}/{st['total']} in {seconds:.1f}s")

            if player is not None:
                player.stop()
                data["hits_sent"] = player.hits
                player = None
            data["failures"] = failures
            landed = sum(1 for k in data["kits"] if k.get("landed"))
            if failures:
                summary = (f"waveform_cache: FAIL — {len(failures)} of {len(cycle)} kit(s):\n  "
                           + "\n  ".join(failures))
                return Report(passed=False, summary=summary, data=data, artifacts=[], exit_code=1)
            slow = max((k["seconds"] for k in data["kits"] if k.get("landed")), default=0.0)
            summary = (f"waveform_cache: PASS — {landed} kit(s), every layer cached "
                       f"(slowest {slow:.1f}s{', pads firing' if p.play else ''})")
            return Report(passed=True, summary=summary, data=data, artifacts=[], exit_code=0)
        finally:
            if player is not None:
                player.stop()
            ctx.close_all()


SCENARIO = register(WaveformCacheScenario())
