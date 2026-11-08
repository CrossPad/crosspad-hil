"""Every action the jam took, on one host clock, one JSON line each.

The analysis rebuilds what the take should hold from this log alone (with the
board's readbacks), so it records what was *done*, when it was handed to the
OS (``t_mono`` just before the write, ``t_mono_end`` just after), not what the
script meant to do. ``time.monotonic`` is the clock the capture's clock log is
on, so an action maps onto a frame of the take without any other help.
"""
from __future__ import annotations

import json
import threading
import time
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class Action:
    t_mono: float
    t_mono_end: float
    t_wall: float
    block: str
    step: str
    kind: str
    args: dict[str, Any] = field(default_factory=dict)
    console_seq: int | None = None
    ok: bool = True
    err: str | None = None
    aid: int = 0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @staticmethod
    def from_dict(d: dict[str, Any]) -> Action:
        return Action(t_mono=float(d["t_mono"]), t_mono_end=float(d.get("t_mono_end",
                                                                         d["t_mono"])),
                      t_wall=float(d.get("t_wall", 0.0)), block=str(d.get("block", "")),
                      step=str(d.get("step", "")), kind=str(d["kind"]),
                      args=dict(d.get("args") or {}), console_seq=d.get("console_seq"),
                      ok=bool(d.get("ok", True)), err=d.get("err"), aid=int(d.get("aid", 0)))


class ActionLog:
    """Thread-safe append-only log; ``emit`` stamps and writes one action."""

    def __init__(self, path: Path | None, *, clock: Callable[[], float] = time.monotonic,
                 wall: Callable[[], float] = time.time,
                 console_seq: Callable[[], int | None] | None = None) -> None:
        self.path = Path(path) if path is not None else None
        self._clock = clock
        self._wall = wall
        self._console_seq = console_seq
        self._lock = threading.Lock()
        self.actions: list[Action] = []
        self.block = ""
        self.step = ""
        self._fh = None
        if self.path is not None:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self._fh = open(self.path, "a", buffering=1, encoding="utf-8")  # noqa: SIM115

    def close(self) -> None:
        with self._lock:
            if self._fh is not None:
                self._fh.close()
                self._fh = None

    def _write(self, a: Action) -> None:
        with self._lock:
            a.aid = len(self.actions) + 1
            self.actions.append(a)
            if self._fh is not None:
                self._fh.write(json.dumps(a.to_dict(), default=str) + "\n")

    def emit(self, kind: str, *, t0: float | None = None, t1: float | None = None,
             ok: bool = True, err: str | None = None, block: str | None = None,
             step: str | None = None, **args: Any) -> Action:
        now = self._clock()
        a = Action(t_mono=now if t0 is None else t0, t_mono_end=now if t1 is None else t1,
                   t_wall=self._wall(), block=self.block if block is None else block,
                   step=self.step if step is None else step, kind=kind, args=args,
                   console_seq=self._console_seq() if self._console_seq else None,
                   ok=ok, err=err)
        self._write(a)
        return a

    def run(self, kind: str, fn: Callable[[], Any], **args: Any) -> tuple[Action, Any]:
        """Stamp ``fn()`` before and after; an exception is logged (ok=False) and re-raised."""
        t0 = self._clock()
        try:
            out = fn()
        except Exception as e:  # noqa: BLE001 - logged, then the caller decides
            self.emit(kind, t0=t0, t1=self._clock(), ok=False, err=f"{type(e).__name__}: {e}",
                      **args)
            raise
        return self.emit(kind, t0=t0, t1=self._clock(), **args), out

    @contextmanager
    def span(self, kind: str, **args: Any) -> Iterator[dict[str, Any]]:
        """``with log.span("step_x"):`` -- one action with the span's start and end."""
        t0 = self._clock()
        extra: dict[str, Any] = {}
        try:
            yield extra
        except Exception as e:
            self.emit(kind, t0=t0, t1=self._clock(), ok=False, err=f"{type(e).__name__}: {e}",
                      **args, **extra)
            raise
        self.emit(kind, t0=t0, t1=self._clock(), **args, **extra)


def load_actions(path: Path) -> list[Action]:
    out: list[Action] = []
    try:
        lines = Path(path).read_text(encoding="utf-8").splitlines()
    except OSError:
        return out
    for i, line in enumerate(lines):
        line = line.strip()
        if not line:
            continue
        try:
            a = Action.from_dict(json.loads(line))
        except (ValueError, KeyError):
            continue
        if not a.aid:
            a.aid = i + 1
        out.append(a)
    return out
