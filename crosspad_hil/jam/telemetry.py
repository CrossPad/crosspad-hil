"""What the board said about itself while the jam ran: console lines and readbacks.

Every parser here takes one line (or one reply) and returns a plain dict, and
none of them minds a line in an older or newer format: a field the firmware
does not print is simply absent. ``Telemetry`` indexes the whole console by
host time so a glitch can ask "what did the board print around me?".
"""
from __future__ import annotations

import bisect
import re
from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from crosspad_hil.jam.clocks import DeviceLogClock, dev_ts, strip_ansi

_NUM = re.compile(r"^-?\d+(\.\d+)?$")


def _val(v: str) -> Any:
    if _NUM.match(v):
        return float(v) if "." in v else int(v)
    return v


def kv(text: str) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for tok in text.replace(":", " ").split():
        k, sep, v = tok.partition("=")
        if sep and k:
            out[k] = _val(v.rstrip(","))
    return out


# --------------------------------------------------------------------------- line kinds

RING_MIN_NONE = 4294967295       # ring_min sentinel: nothing written yet


def parse_spk(line: str) -> dict[str, Any] | None:
    """``SPK #n: len= taken= ring= under= refused= cap_ring= cap_drop= ring_min=
    [ev_at= hole_ms= cap_under=]`` (old lines lack the tail)."""
    clean = strip_ansi(line)
    i = clean.find("SPK #")
    if i < 0:
        return None
    m = re.match(r"SPK #(\d+)", clean[i:])
    out = {"n": int(m.group(1)) if m else None, **kv(clean[i + 4:].split(":", 1)[-1])}
    if out.get("ring_min") == RING_MIN_NONE:
        out["ring_min"] = None
    return out


def parse_underrun(line: str) -> dict[str, Any] | None:
    """``engine underrun_at=Nms bg_pass|bg_idle=Nus read=Nus yld= fb= stage=c p1= p2=``."""
    clean = strip_ansi(line)
    i = clean.find("underrun_at=")
    if i < 0:
        return None
    d = kv(clean[i:].replace("ms ", " ").replace("us ", " "))
    at = re.search(r"underrun_at=(\d+)ms", clean)
    if at:
        d["underrun_at"] = int(at.group(1))
    return d


def parse_pool(line: str) -> dict[str, Any] | None:
    clean = strip_ansi(line)
    i = clean.find("engine pool:")
    return None if i < 0 else kv(clean[i + len("engine pool:"):])


_KIT_LOAD = re.compile(r"KIT_LOAD (\d+) (started|queued|refused)")
_BOUNCE = re.compile(r"BOUNCE (\d+) (\d+) (started|refused)")
_BOUNCED = re.compile(r"bounced scene (\d+) to (\S+) \((\d+) frames\)")
_BOUNCE_FAIL = re.compile(r"bounce failed: (.*)$")
_MODE = re.compile(r"USB mode switched to: (\S+)(?: \(rt_gap_max_us=(\d+)\))?")


def classify(line: str) -> tuple[str, dict[str, Any]]:
    """(kind, fields) of one console line."""
    clean = strip_ansi(line)
    if "SPK #" in clean:
        return "spk", parse_spk(clean) or {}
    if "underrun_at=" in clean:
        return "underrun", parse_underrun(clean) or {}
    if "engine pool:" in clean:
        return "pool", parse_pool(clean) or {}
    m = _KIT_LOAD.search(clean)
    if m:
        return "kit_load", {"kit": int(m.group(1)), "state": m.group(2)}
    m = _BOUNCE.search(clean)
    if m:
        return "bounce", {"scene": int(m.group(1)), "track": int(m.group(2)),
                          "state": m.group(3)}
    m = _BOUNCED.search(clean)
    if m:
        return "bounced", {"scene": int(m.group(1)), "file": m.group(2),
                           "frames": int(m.group(3))}
    m = _BOUNCE_FAIL.search(clean)
    if m:
        return "bounce_failed", {"why": m.group(1)}
    m = _MODE.search(clean)
    if m:
        return "mode", {"mode": m.group(1), "rt_gap_max_us": int(m.group(2) or 0)}
    if "Speaker opened" in clean or "speaker opened" in clean.lower():
        return "spk_open", {}
    if "Speaker closed" in clean or "SPK closed" in clean:
        return "spk_closed", kv(clean)
    if "Guru Meditation" in clean or "abort()" in clean or "panic" in clean.lower() \
            or "task_wdt" in clean:
        return "fatal", {}
    if "rst:0x" in clean or "ESP-ROM:" in clean:
        return "reboot", {}
    lower = clean.lower()
    if "sequences.json" in lower or "seq_store" in lower or ("saved" in lower and "seq" in lower):
        return "sd_write", {"what": "sequences"}
    if "preset" in lower and ("save" in lower or "wrote" in lower):
        return "sd_write", {"what": "preset"}
    if "nvs" in lower and ("commit" in lower or "save" in lower):
        return "nvs", {}
    return "line", {}


# --------------------------------------------------------------------------- readbacks

def parse_seq(line: str) -> dict[str, Any]:
    head, _, path = line.partition(" file=")
    out = kv(head.split("SEQ:", 1)[-1])
    for key in ("bars", "ev"):
        if isinstance(out.get(key), str):
            out[key] = [int(v) for v in str(out[key]).split("/") if v.strip().isdigit()]
        elif isinstance(out.get(key), int):
            out[key] = [out[key]]
    out["file"] = path.strip()
    return out


def parse_seq_dump(lines: Iterable[str]) -> list[dict[str, int]]:
    out = []
    for line in lines:
        d = kv(line.split("SEQDUMP:", 1)[-1])
        if {"seq", "i", "pos", "pad", "vel"} <= set(d):
            out.append({k: int(d[k]) for k in ("seq", "i", "pos", "pad", "vel")})
    return out


def parse_click(line: str) -> dict[str, Any]:
    d = kv(line.split("CLICK:", 1)[-1].replace("/", " loop="))
    return d


def parse_song(line: str) -> dict[str, Any]:
    head, _, blocks = line.partition("blocks=")
    d = kv(head.split("SONG:", 1)[-1])
    d["blocks"] = [[int(x) for x in b.split(":")] for b in blocks.strip().split(",") if ":" in b]
    return d


def parse_mixlvl(lines: Iterable[str]) -> dict[str, Any]:
    chans, outs = [], []
    for line in lines:
        d = kv(line.split("MIXLVL:", 1)[-1].replace("/", " pk_r="))
        if "ch" in d:
            d["route"] = [int(c) for c in str(d.get("route", "")) if c in "01"]
            chans.append(d)
        elif "out" in d:
            outs.append(d)
    return {"channels": chans, "outputs": outs}


def parse_clock(line: str) -> dict[str, Any]:
    return kv(line.split("CLOCK:", 1)[-1])


def parse_smpl_last(line: str) -> dict[str, Any]:
    return kv(line.split("SMPLLAST:", 1)[-1])


def parse_smpl_pool(line: str) -> dict[str, Any]:
    return kv(line.split("SMPLPOOL:", 1)[-1])


# --------------------------------------------------------------------------- the index

@dataclass
class Row:
    seq: int
    t_arr: float
    t_wall: float
    line: str
    dev_ms: int | None
    kind: str
    fields: dict[str, Any] = field(default_factory=dict)
    t_host: float | None = None       # when the board said it was (log ms -> host), else t_arr


@dataclass
class Telemetry:
    rows: list[Row] = field(default_factory=list)
    clock: DeviceLogClock | None = None
    _t: list[float] = field(default_factory=list)

    @staticmethod
    def from_rows(raw: list[tuple[int, float, float, str]]) -> Telemetry:
        clk = DeviceLogClock.fit(raw)
        tel = Telemetry(clock=clk)
        for seq, ta, tw, line in raw:
            kind, fields = classify(line)
            ms = dev_ts(line)
            t_host = None
            if ms is not None:
                t_host = clk.to_host(ms, clk.epoch_of(seq=seq))
            tel.rows.append(Row(seq=seq, t_arr=ta, t_wall=tw, line=line, dev_ms=ms, kind=kind,
                                fields=fields, t_host=t_host if t_host is not None else ta))
        tel.rows.sort(key=lambda r: r.t_host or r.t_arr)
        tel._t = [r.t_host or r.t_arr for r in tel.rows]
        return tel

    @staticmethod
    def load_tsv(path: Path) -> Telemetry:
        raw: list[tuple[int, float, float, str]] = []
        try:
            text = Path(path).read_text(encoding="utf-8", errors="replace")
        except OSError:
            return Telemetry()
        for line in text.splitlines():
            parts = line.split("\t", 3)
            if len(parts) < 4:
                continue
            try:
                raw.append((int(parts[0]), float(parts[1]), float(parts[2]), parts[3]))
            except ValueError:
                continue
        return Telemetry.from_rows(raw)

    def around(self, t: float, before: float, after: float,
               kinds: tuple[str, ...] | None = None) -> list[Row]:
        i = bisect.bisect_left(self._t, t - before)
        j = bisect.bisect_right(self._t, t + after)
        rows = self.rows[i:j]
        return [r for r in rows if kinds is None or r.kind in kinds]

    def of_kind(self, *kinds: str, t0: float | None = None,
                t1: float | None = None) -> list[Row]:
        return [r for r in self.rows if r.kind in kinds
                and (t0 is None or (r.t_host or r.t_arr) >= t0)
                and (t1 is None or (r.t_host or r.t_arr) <= t1)]

    def spk_covering(self, t: float, lag: float = 0.2) -> tuple[Row | None, Row | None]:
        """The SPK lines either side of ``t``: their difference is what happened in between
        (one line a second at best -- a stream fault is placed to that second). The line
        after must come ``lag`` later: one printed in the same instant has not counted it."""
        spk = self.of_kind("spk")
        before = [r for r in spk if (r.t_host or r.t_arr) <= t]
        after = [r for r in spk if (r.t_host or r.t_arr) > t + lag]
        return (before[-1] if before else None, after[0] if after else None)
