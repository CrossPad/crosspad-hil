"""Every clock a glitch can be named on, and the maps between them.

* **capture** (``CaptureClock``): a frame of the take <-> host monotonic time,
  from the duplex stream's clock log (lower envelope, one fit per stretch
  between xruns). ~1 ms.
* **device log** (``DeviceLogClock``): an ESP log timestamp ``I (12345)`` <->
  host time, from when each console line arrived (minus the time its bytes
  took on the 115200-baud UART), lower envelope per boot. +-1-2 ms.
* **device timer** (``TimerOffset``): esp_timer ms (``req_ms``, ``underrun_at``)
  <-> log ms, the median offset of pairs the board logs both ways.
* **render** (``RenderMap``): the take's frame <-> the frame the engine rendered,
  a constant per block (the path is digital) -- whatever breaks it is a slip.

``Timebase.axes(frame)`` hands a glitch every one of them at once.
"""
from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from crosspad_hil.latency_math import ClockFit, lower_envelope_fit

UART_S_PER_BYTE = 10.0 / 115_200.0     # 8N1
_ANSI = re.compile(r"\x1b\[[0-9;]*m")
_DEV_TS = re.compile(r"^[IWEDV] \((\d+)\)")


def strip_ansi(line: str) -> str:
    return _ANSI.sub("", line)


def dev_ts(line: str) -> int | None:
    m = _DEV_TS.match(strip_ansi(line).lstrip())
    return int(m.group(1)) if m else None


# --------------------------------------------------------------------------- capture

@dataclass
class CaptureClock:
    sr: int
    fits: list[tuple[float, float, ClockFit]] = field(default_factory=list)   # (t0, t1, fit)
    t_start: float | None = None
    note: str = ""

    @staticmethod
    def from_points(points: list[tuple[int, float]], sr: int, *,
                    breaks: list[float] | None = None,
                    t_start: float | None = None) -> CaptureClock:
        """One fit per stretch between ``breaks`` (host times of capture xruns)."""
        clk = CaptureClock(sr=sr, t_start=t_start)
        pts = sorted((int(f), float(t)) for f, t in points)
        if len(pts) < 2:
            clk.note = "fewer than two clock points: frame = start + n / rate"
            return clk
        cuts = sorted(breaks or [])
        edges = [-math.inf, *cuts, math.inf]
        for a, b in zip(edges, edges[1:], strict=False):
            seg = [(f, t) for f, t in pts if a <= t < b]
            if len(seg) < 2:
                continue
            try:
                fit = lower_envelope_fit([f for f, _ in seg], [t for _, t in seg],
                                         nominal_rate=sr)
            except ValueError as e:
                clk.note = f"clock fit: {e}"
                continue
            clk.fits.append((seg[0][1], seg[-1][1], fit))
        return clk

    def _fit_for_t(self, t: float) -> ClockFit | None:
        if not self.fits:
            return None
        for a, b, fit in self.fits:
            if a <= t <= b:
                return fit
        return min(self.fits, key=lambda x: min(abs(t - x[0]), abs(t - x[1])))[2]

    def frame_of(self, t: float) -> float:
        fit = self._fit_for_t(t)
        if fit is None:
            return (t - (self.t_start or 0.0)) * self.sr
        return float(fit.frame(t))

    def t_of(self, frame: float) -> float:
        if not self.fits:
            return (self.t_start or 0.0) + frame / self.sr
        best = None
        for a, b, fit in self.fits:
            t = float(fit.t(frame))
            if a - 1.0 <= t <= b + 1.0:
                return t
            best = t if best is None else best
        return float(best if best is not None else 0.0)

    def to_dict(self) -> dict[str, Any]:
        return {"segments": [{"t0": round(a, 3), "t1": round(b, 3), **fit.to_dict()}
                             for a, b, fit in self.fits], "note": self.note}


# --------------------------------------------------------------------------- device log

@dataclass
class DeviceLogClock:
    """Log ms -> host time, per boot (``epoch``)."""

    epochs: list[dict[str, Any]] = field(default_factory=list)

    @staticmethod
    def fit(rows: list[tuple[int, float, float, str]]) -> DeviceLogClock:
        """``rows``: (seq, t_arr, t_wall, line) from ``Console.read_timed`` or console.tsv."""
        clk = DeviceLogClock()
        cur: list[tuple[int, float, int]] = []
        last_ms = -1
        epochs: list[list[tuple[int, float, int]]] = []
        for seq, t_arr, _tw, line in rows:
            ms = dev_ts(line)
            if ms is None:
                continue
            if ms < last_ms - 2000:              # a reboot: the log clock started over
                if cur:
                    epochs.append(cur)
                cur = []
            last_ms = ms
            t_emit = float(t_arr) - (len(line) + 2) * UART_S_PER_BYTE
            cur.append((ms, t_emit, seq))
        if cur:
            epochs.append(cur)
        for e in epochs:
            seqs = (e[0][2], e[-1][2])
            if len(e) < 2 or e[-1][0] - e[0][0] < 1000:
                clk.epochs.append({"seq": seqs, "ms": (e[0][0], e[-1][0]),
                                   "fit": None, "offset": e[0][1] - e[0][0] / 1000.0})
                continue
            try:
                fit = lower_envelope_fit([m for m, _, _ in e], [t for _, t, _ in e],
                                         nominal_rate=1000.0, bin_s=10.0)
            except ValueError:
                fit = None
            off = float(np.median([t - m / 1000.0 for m, t, _ in e]))
            clk.epochs.append({"seq": seqs, "ms": (e[0][0], e[-1][0]), "fit": fit,
                               "offset": off})
        return clk

    def epoch_of(self, seq: int | None = None, t_host: float | None = None) -> dict | None:
        if not self.epochs:
            return None
        if seq is not None:
            for e in self.epochs:
                if e["seq"][0] <= seq <= e["seq"][1]:
                    return e
        if t_host is not None:
            best = None
            for e in self.epochs:
                t0 = self.to_host(e["ms"][0], e)
                if t0 is not None and t0 <= t_host + 1.0:
                    best = e
            return best or self.epochs[0]
        return self.epochs[-1]

    def to_host(self, ms: float, epoch: dict | None = None) -> float | None:
        e = epoch or (self.epochs[-1] if self.epochs else None)
        if e is None:
            return None
        fit = e.get("fit")
        if fit is not None:
            return float(fit.t(ms))
        return float(e["offset"] + ms / 1000.0)

    def to_dev(self, t_host: float) -> float | None:
        e = self.epoch_of(t_host=t_host)
        if e is None:
            return None
        fit = e.get("fit")
        if fit is not None:
            return float(fit.frame(t_host))
        return float((t_host - e["offset"]) * 1000.0)

    def to_dict(self) -> dict[str, Any]:
        out = []
        for e in self.epochs:
            fit = e.get("fit")
            out.append({"seq": list(e["seq"]), "ms": list(e["ms"]),
                        "rate": round(fit.rate, 4) if fit else None,
                        "resid_ms": fit.resid_ms if fit else None})
        return {"epochs": out}


@dataclass
class TimerOffset:
    """esp_timer ms - log ms: ``timer_ms = log_ms - offset``."""

    offset_ms: float = 760.0
    pairs: int = 0

    @staticmethod
    def fit(pairs: list[tuple[float, float]], default: float = 760.0) -> TimerOffset:
        """``pairs``: (log ms of a "KIT_LOAD n started" line, req_ms of its swap row)."""
        if not pairs:
            return TimerOffset(default, 0)
        return TimerOffset(float(np.median([a - b for a, b in pairs])), len(pairs))

    def log_ms(self, timer_ms: float) -> float:
        return float(timer_ms) + self.offset_ms


# --------------------------------------------------------------------------- all of them

@dataclass
class Timebase:
    sr: int
    capture: CaptureClock
    log: DeviceLogClock | None = None
    wall_offset: float = 0.0              # t_wall - t_mono
    steps: list[dict[str, Any]] = field(default_factory=list)   # [{id, t0, t1, block}]
    runs: list[dict[str, Any]] = field(default_factory=list)    # model runs (passes)
    bpm: float = 120.0
    bars: list[int] = field(default_factory=list)

    def frame_of(self, t: float) -> float:
        return self.capture.frame_of(t)

    def axes(self, frame: float) -> dict[str, Any]:
        t = self.capture.t_of(frame)
        out: dict[str, Any] = {"frame": int(round(frame)),
                               "wav_s": round(frame / self.sr, 6),
                               "t_host": round(t, 6),
                               "t_wall": round(t + self.wall_offset, 3),
                               "clock_s": "+-0.002"}
        import time as _time

        out["wall"] = _time.strftime("%H:%M:%S", _time.localtime(t + self.wall_offset)) + \
            f".{int(((t + self.wall_offset) % 1) * 1000):03d}"
        if self.log is not None:
            ms = self.log.to_dev(t)
            out["dev_log_ms"] = None if ms is None else round(ms, 1)
        step = self.step_at(t)
        if step is not None:
            out["step"] = step["id"]
            out["step_s"] = round(t - step["t0"], 3)
        pos = self.position(frame)
        if pos:
            out.update(pos)
        return out

    def step_at(self, t: float) -> dict[str, Any] | None:
        best = None
        for s in self.steps:
            if s["t0"] <= t <= s.get("t1", s["t0"]):
                if best is None or s["t0"] >= best["t0"]:
                    best = s
        return best

    def position(self, frame: float) -> dict[str, Any]:
        """Scene, pass and bar.beat.tick of the sequencer at ``frame`` (fitted runs)."""
        spb = int(round(self.sr * 60.0 / self.bpm))
        for r in reversed(self.runs):
            if r.get("f_from", 0) <= frame < r.get("f_stop", 0):
                sched = r.get("schedule") or [[0, 0]]
                scene = int(sched[0][0])
                bars = self.bars[scene] if scene < len(self.bars) and self.bars[scene] else 1
                loop = bars * 4 * spb
                rel = frame - float(r.get("f_anchor", 0.0))
                k = math.floor(rel / loop)
                pos = rel - k * loop
                beat = int(pos // spb)
                tick = int((pos - beat * spb) / spb * 96)
                return {"scene": scene, "pass": int(k), "bar_beat":
                        f"{beat // 4 + 1}.{beat % 4 + 1}.{tick:02d}"}
        return {}
