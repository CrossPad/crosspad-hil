"""Workflows: a user's whole task as data, run by more than one driver.

A workflow file (``<name>.yaml`` beside this module) lists steps, each tagged
with the shallowest depth it runs at. The board driver (a scenario) maps each
step's ``op`` to CDC verbs and checks; the Mucha performer plays each step's
``sim`` script on the PC simulator. Both read the same tempo, fixture and
patterns from here, so the fly plays the beat the board is tested with.

Stdlib + PyYAML only (the performer imports this from outside the venv).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

DEPTHS: tuple[str, ...] = ("quick", "medium", "full")
HERE = Path(__file__).resolve().parent


@dataclass(frozen=True)
class Hit:
    step: int
    pad: int
    vel: int


@dataclass(frozen=True)
class Pattern:
    name: str
    bars: int
    hits: tuple[Hit, ...]


@dataclass
class Step:
    id: str
    level: str
    op: str
    args: dict[str, Any] = field(default_factory=dict)
    caption: tuple[str, str] = ("", "")
    sim: list[dict[str, Any]] = field(default_factory=list)


@dataclass
class Workflow:
    raw: dict[str, Any]
    steps: list[Step]
    patterns: dict[str, Pattern]

    @property
    def bpm(self) -> float:
        return float(self.raw["bpm"])

    @property
    def steps_per_bar(self) -> int:
        return int(self.raw.get("steps_per_bar", 16))

    @property
    def step_s(self) -> float:
        """One grid step (a 1/16 at 4/4) in seconds."""
        return 60.0 / self.bpm * 4.0 / self.steps_per_bar

    @property
    def bar_s(self) -> float:
        return self.step_s * self.steps_per_bar

    def for_depth(self, depth: str) -> list[Step]:
        """The steps a run at `depth` takes, in file order."""
        if depth not in DEPTHS:
            raise ValueError(f"depth must be one of {', '.join(DEPTHS)}, not {depth!r}")
        upto = DEPTHS.index(depth)
        return [s for s in self.steps if DEPTHS.index(s.level) <= upto]

    def edits(self, depth: str) -> list[dict[str, Any]]:
        """Pad-editor changes a run at `depth` makes (cumulative over depths)."""
        out: list[dict[str, Any]] = []
        for d in DEPTHS[: DEPTHS.index(depth) + 1]:
            out += list((self.raw.get("edits") or {}).get(d) or [])
        return out

    def names(self, tag: str) -> dict[str, str]:
        """The run's own names, `tag` filled in."""

        def fill(s: str) -> str:
            return s.format(tag=tag.lower(), TAG=tag.upper())

        fx = self.raw["fixture"]
        return {
            "file": fill(fx["file"]),
            "dir": fx["dir"],
            "remote": f"{fx['dir'].rstrip('/')}/{fill(fx['file'])}",
            "project": fill(self.raw["project"]),
            "kit": fill(self.raw["kit"]),
        }


def _pattern(name: str, raw: dict[str, Any]) -> Pattern:
    hits = tuple(Hit(int(s), int(p), int(v)) for s, p, v in raw["hits"])
    return Pattern(name=name, bars=int(raw["bars"]), hits=hits)


def _step(raw: dict[str, Any]) -> Step:
    level = raw.get("level", "quick")
    if level not in DEPTHS:
        raise ValueError(f"step {raw.get('id')!r}: level {level!r} is not one of {DEPTHS}")
    cap = raw.get("caption") or ["", ""]
    return Step(
        id=str(raw["id"]),
        level=level,
        op=str(raw.get("op", raw["id"])),
        args=dict(raw.get("args") or {}),
        caption=(str(cap[0]), str(cap[1])),
        sim=list(raw.get("sim") or []),
    )


def load(name_or_path: str | Path) -> Workflow:
    """A workflow by name (``beat``) or by path to its YAML."""
    p = Path(name_or_path)
    if not p.suffix:
        p = HERE / f"{name_or_path}.yaml"
    raw = yaml.safe_load(p.read_text(encoding="utf-8"))
    patterns = {k: _pattern(k, v) for k, v in (raw.get("patterns") or {}).items()}
    steps = [_step(s) for s in raw.get("steps") or []]
    ids = [s.id for s in steps]
    if len(set(ids)) != len(ids):
        raise ValueError(f"{p.name}: step ids repeat")
    return Workflow(raw=raw, steps=steps, patterns=patterns)


def expected_scene(
    wf: Workflow, depth: str, ops: set[str] | None = None
) -> tuple[int, list[Hit]]:
    """Scene 1 as the run at `depth` leaves it before the bounce: (bars, hits).

    quick: pattern A. medium: A + B overdubbed, then doubled. full: as medium,
    without the erased pad's notes. `ops`: the step ops a driver actually ran
    (the simulator has no script for some), default every op at `depth`.
    """
    a = wf.patterns["A"]
    hits = list(a.hits)
    bars = a.bars
    run = ops if ops is not None else {s.op for s in wf.for_depth(depth)}
    if "overdub" in run:
        hits += list(wf.patterns["B"].hits)
    if "erase" in run:
        gone = int(wf.raw.get("erase_pad", -1))
        hits = [h for h in hits if h.pad != gone]
    if "double" in run:
        n = bars * wf.steps_per_bar
        hits = hits + [Hit(h.step + n, h.pad, h.vel) for h in hits]
        bars *= 2
    return bars, sorted(hits, key=lambda h: (h.step, h.pad))
