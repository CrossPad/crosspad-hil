"""Bench lease: one holder per board, a queue behind it, and what firmware was left on it.

Several sessions share one CrossPad. Opening its CDC port for a moment, or
flashing it, breaks whoever is in the middle of a test — and the port locks in
``locks.py`` only cover the instant a port is open, not the test around it. A
lease covers the whole test: claim the board, run, release it saying what
firmware you left on it.

    claim    -> granted, or the current holder and your place in the queue
    renew    -> push your lease's expiry out by its ttl
    release  -> give the board back, recording ``firmware_left``
    check    -> every op that touches the board asks this first; a board
                claimed by someone else is refused with BENCH_BUSY

A lease that is not renewed within its ttl expires, so a session that died
does not keep the board. Every board op by the holder renews it. When a lease
ends with sessions queued, the head of the queue has ``HANDOFF_S`` to claim
before the board is first come, first served again; a queued session that has
not asked again within ``QUEUE_TTL_S`` drops out of the queue.

State is one JSON file shared by every process on the machine — each
crosspad-mcp session runs a daemon of its own — so every operation is a
read-modify-write under an exclusive lock on a sidecar file, and the JSON is
replaced atomically. ``force`` overrides a refusal and is written into the
device's history, so the override is never silent.
"""
from __future__ import annotations

import json
import logging
import os
import sys
import tempfile
import threading
import time
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

from .errors import BAD_ARGS, BENCH_BUSY, HilError

_log = logging.getLogger("crosspad_hil.bench")

BENCH_ENV = "CROSSPAD_HIL_BENCH"
BENCH_MD_ENV = "CROSSPAD_BENCH_MD"
HOLDER_ENV = "CROSSPAD_BENCH_HOLDER"
# enforce (default): refuse; warn: log the refusal and let the op through, for a
# first rollout next to the hand-kept BENCH.md; off: no checks at all.
MODE_ENV = "CROSSPAD_BENCH_MODE"
MODES = ("enforce", "warn", "off")

DEFAULT_TTL_S = 30 * 60.0
MIN_TTL_S = 60.0
MAX_TTL_S = 12 * 3600.0
# A queued session that has not asked again for an hour has gone away.
QUEUE_TTL_S = 60 * 60.0
# After a lease ends, the head of the queue has this long to claim the board.
HANDOFF_S = 10 * 60.0
HISTORY_N = 30
# A renew that would move the expiry by less than this is not worth a write.
_RENEW_SLACK_S = 60.0

MD_BEGIN = "<!-- bench:status -->"
MD_END = "<!-- /bench:status -->"

_FILE_VERSION = 1


def default_bench_path() -> Path:
    """``$CROSSPAD_HIL_BENCH``, else ``$XDG_STATE_HOME/crosspad-hil/bench.json``
    (``~/.local/state/...``). Not the runtime dir: the history of what was left
    on the board is worth keeping across a reboot."""
    env = os.environ.get(BENCH_ENV)
    if env:
        return Path(env)
    state = os.environ.get("XDG_STATE_HOME")
    base = Path(state) if state else Path.home() / ".local" / "state"
    return base / "crosspad-hil" / "bench.json"


def _hhmm(ts: float | None) -> str:
    return "?" if ts is None else time.strftime("%Y-%m-%d %H:%M", time.localtime(ts))


def _mins(seconds: float) -> str:
    m = max(0, round(seconds / 60.0))
    return f"{m} min" if m < 120 else f"{m / 60.0:.1f} h"


def _text(value: Any, name: str, *, required: bool = True) -> str:
    if value is None or (isinstance(value, str) and not value.strip()):
        if required:
            raise HilError(BAD_ARGS, f"{name!r} is required",
                           hint=f"{name}: free text, e.g. the session name")
        return ""
    if not isinstance(value, str):
        raise HilError(BAD_ARGS, f"{name!r} must be a string")
    return value.strip()


def _ttl(ttl_s: float | int | None) -> float:
    if ttl_s is None:
        return DEFAULT_TTL_S
    if isinstance(ttl_s, bool) or not isinstance(ttl_s, (int, float)):
        raise HilError(BAD_ARGS, "ttl_s must be a number of seconds")
    return float(min(max(float(ttl_s), MIN_TTL_S), MAX_TTL_S))


@contextmanager
def _exclusive(path: Path) -> Iterator[None]:
    """Exclusive advisory lock on ``path`` (flock on POSIX, msvcrt on Windows)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a+b") as fh:
        if sys.platform == "win32":  # pragma: no cover - exercised on Windows only
            import msvcrt

            fh.seek(0)
            while True:
                try:
                    msvcrt.locking(fh.fileno(), msvcrt.LK_LOCK, 1)
                    break
                except OSError:
                    continue  # LK_LOCK gives up after ~10 s; keep waiting
            try:
                yield
            finally:
                fh.seek(0)
                msvcrt.locking(fh.fileno(), msvcrt.LK_UNLCK, 1)
            return
        import fcntl

        fcntl.flock(fh.fileno(), fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(fh.fileno(), fcntl.LOCK_UN)


class Bench:
    """The lease table. Every public method is one locked read-modify-write."""

    def __init__(self, path: Path | str | None = None, *,
                 clock: Callable[[], float] = time.time,
                 md_path: Path | str | None = None,
                 mode: str | None = None,
                 log: Callable[[str], None] | None = None) -> None:
        self.path = Path(path) if path is not None else default_bench_path()
        self._lock_path = self.path.with_name(self.path.name + ".lock")
        md = md_path if md_path is not None else os.environ.get(BENCH_MD_ENV)
        self.md_path = Path(md) if md else None
        self._clock = clock
        self._log = log or _log.warning
        mode = (mode or os.environ.get(MODE_ENV) or "enforce").strip().lower()
        if mode not in MODES:
            self._log(f"bench: {MODE_ENV}={mode!r} is not one of {MODES}; enforcing")
            mode = "enforce"
        self.mode = mode
        self._guard = threading.Lock()

    # ----------------------------------------------------------------- storage

    def _load(self) -> dict:
        try:
            raw = self.path.read_text(encoding="utf-8")
        except FileNotFoundError:
            return {"version": _FILE_VERSION, "devices": {}}
        try:
            data = json.loads(raw)
            if not isinstance(data, dict) or not isinstance(data.get("devices"), dict):
                raise ValueError("no 'devices' object")
        except ValueError as e:
            aside = self.path.with_name(f"{self.path.name}.corrupt-{int(self._clock())}")
            self._log(f"bench: {self.path} is unreadable ({e}); moved to {aside}, starting empty")
            try:
                os.replace(self.path, aside)
            except OSError:
                pass
            return {"version": _FILE_VERSION, "devices": {}}
        return data

    def _save(self, data: dict) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        fd, tmp = tempfile.mkstemp(prefix=self.path.name + ".", dir=str(self.path.parent))
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as fh:
                json.dump(data, fh, indent=2, sort_keys=True)
                fh.write("\n")
            os.replace(tmp, self.path)
        except BaseException:
            try:
                os.unlink(tmp)
            except OSError:
                pass
            raise

    @contextmanager
    def _txn(self) -> Iterator[tuple[dict, float, list[bool]]]:
        """Yield ``(state, now, dirty)``; set ``dirty[0]`` to write the state back.
        Stale leases and queue entries are expired on the way in."""
        with self._guard, _exclusive(self._lock_path):
            data = self._load()
            now = self._clock()
            dirty = [False]
            for dev in data["devices"].values():
                if self._expire(dev, now):
                    dirty[0] = True
            yield data, now, dirty
            if dirty[0]:
                self._save(data)
                self._write_md(data, now)

    @staticmethod
    def _dev(data: dict, device: str) -> dict:
        dev = data["devices"].setdefault(device, {})
        dev.setdefault("lease", None)
        dev.setdefault("queue", [])
        dev.setdefault("history", [])
        dev.setdefault("firmware", None)
        dev.setdefault("free_since", None)
        return dev

    # ----------------------------------------------------------------- rules

    def _expire(self, dev: dict, now: float) -> bool:
        changed = False
        lease = dev.get("lease")
        if lease is not None and now >= float(lease["expires"]):
            until = float(lease["expires"])
            self._history(dev, {"kind": "expired", "holder": lease["holder"],
                                "purpose": lease.get("purpose", ""), "since": lease["since"],
                                "until": until, "firmware_left": None})
            dev["firmware"] = {"desc": None, "by": lease["holder"], "at": until,
                               "note": "lease expired without a release: unknown"}
            dev["lease"] = None
            dev["free_since"] = until
            changed = True
        queue = dev.get("queue") or []
        alive = [q for q in queue if now < float(q["expires"])]
        if len(alive) != len(queue):
            dev["queue"] = alive
            changed = True
        return changed

    @staticmethod
    def _history(dev: dict, entry: dict) -> None:
        hist = dev.setdefault("history", [])
        hist.append(entry)
        del hist[:-HISTORY_N]

    @staticmethod
    def _reserved_for(dev: dict, now: float) -> tuple[str, float] | None:
        """The queue head a just-freed board is held for, and until when."""
        if dev.get("lease") is not None or not dev.get("queue"):
            return None
        free_since = dev.get("free_since")
        if free_since is None:
            return None
        until = float(free_since) + HANDOFF_S
        if now >= until:
            return None
        return str(dev["queue"][0]["holder"]), until

    @staticmethod
    def _grant(dev: dict, holder: str, purpose: str, ttl: float, now: float) -> dict:
        lease = {"holder": holder, "purpose": purpose, "since": now, "ttl_s": ttl,
                 "renewed": now, "expires": now + ttl}
        dev["lease"] = lease
        dev["queue"] = [q for q in dev["queue"] if q["holder"] != holder]
        dev["free_since"] = None
        return lease

    @staticmethod
    def _vacate(dev: dict, now: float) -> None:
        dev["lease"] = None
        dev["free_since"] = now

    @staticmethod
    def _touch(lease: dict, now: float, ttl: float | None = None) -> bool:
        if ttl is not None:
            lease["ttl_s"] = ttl
        target = now + float(lease["ttl_s"])
        if ttl is None and target - float(lease["expires"]) < _RENEW_SLACK_S:
            return False
        lease["expires"] = target
        lease["renewed"] = now
        return True

    # ----------------------------------------------------------------- views

    @staticmethod
    def _lease_view(lease: dict | None, now: float) -> dict | None:
        if lease is None:
            return None
        return {**lease, "since_at": _hhmm(lease["since"]), "expires_at": _hhmm(lease["expires"]),
                "expires_in_s": round(float(lease["expires"]) - now, 1)}

    def _device_view(self, device: str, dev: dict, now: float, history: int) -> dict:
        reserved = self._reserved_for(dev, now)
        queue = [{**q, "position": i + 1, "since_at": _hhmm(q["since"])}
                 for i, q in enumerate(dev.get("queue") or [])]
        view: dict[str, Any] = {
            "device": device,
            "lease": self._lease_view(dev.get("lease"), now),
            "queue": queue,
            "firmware": dev.get("firmware"),
            "history": list(dev.get("history") or [])[-history:] if history > 0 else [],
        }
        if reserved is not None:
            view["reserved_for"] = {"holder": reserved[0], "until": reserved[1],
                                    "until_at": _hhmm(reserved[1])}
        return view

    @staticmethod
    def _busy_message(device: str, lease: dict, now: float) -> str:
        purpose = f" ({lease['purpose']})" if lease.get("purpose") else ""
        return (f"{device} is claimed by {lease['holder']}{purpose} since "
                f"{_hhmm(lease['since'])[-5:]}; the lease runs to {_hhmm(lease['expires'])[-5:]} "
                f"(in {_mins(float(lease['expires']) - now)}) unless renewed")

    # ----------------------------------------------------------------- operations

    def claim(self, device: str, holder: str, purpose: str = "", ttl_s: float | None = None,
              *, force: bool = False) -> dict:
        """Take the board, or join the queue behind whoever has it.

        Claiming again while holding renews (and may change ttl/purpose);
        claiming again while queued keeps the place and refreshes it.
        ``force`` takes the board from the current holder; that is logged.
        """
        device = _text(device, "device")
        holder = _text(holder, "holder")
        purpose = _text(purpose, "purpose", required=False)
        ttl = _ttl(ttl_s)
        with self._txn() as (data, now, dirty):
            dev = self._dev(data, device)
            dirty[0] = True
            lease = dev["lease"]
            reserved = self._reserved_for(dev, now)
            if lease is not None and lease["holder"] == holder:
                self._touch(lease, now, ttl)
                if purpose:
                    lease["purpose"] = purpose
                granted = True
            elif lease is None and (reserved is None or reserved[0] == holder):
                lease = self._grant(dev, holder, purpose, ttl, now)
                granted = True
            elif force:
                taken_from = lease["holder"] if lease is not None else None
                if lease is not None:
                    self._history(dev, {"kind": "taken", "holder": lease["holder"],
                                        "purpose": lease.get("purpose", ""),
                                        "since": lease["since"], "until": now,
                                        "by": holder, "firmware_left": None})
                else:
                    assert reserved is not None
                    self._history(dev, {"kind": "force", "holder": holder, "over": reserved[0],
                                        "op": "bench.claim", "at": now})
                over = taken_from if taken_from is not None else f"{reserved[0]} (queue head)"
                self._log(f"bench: {holder} force-claimed {device} from {over}")
                lease = self._grant(dev, holder, purpose, ttl, now)
                granted = True
            else:
                granted = False
                entry = next((q for q in dev["queue"] if q["holder"] == holder), None)
                if entry is None:
                    entry = {"holder": holder, "purpose": purpose, "since": now, "ttl_s": ttl}
                    dev["queue"].append(entry)
                elif purpose:
                    entry["purpose"] = purpose
                entry["ttl_s"] = ttl
                entry["seen"] = now
                entry["expires"] = now + QUEUE_TTL_S
            view = self._device_view(device, dev, now, history=0)
        out: dict[str, Any] = {"granted": granted, "holder": holder, **view}
        if granted:
            lv = view["lease"]
            out["message"] = (f"{holder} holds {device} until {lv['expires_at'][-5:]} "
                              f"(in {_mins(lv['expires_in_s'])}); every board op renews it")
        else:
            pos = next(q["position"] for q in view["queue"] if q["holder"] == holder)
            out["position"] = pos
            if view["lease"] is not None:
                out["message"] = (self._busy_message(device, dev["lease"], now)
                                  + f"; you are #{pos} in the queue")
            else:
                r = view["reserved_for"]
                out["message"] = (f"{device} is free but held for {r['holder']} (first in the "
                                  f"queue) until {r['until_at'][-5:]}; you are #{pos}")
        return out

    def renew(self, device: str, holder: str, ttl_s: float | None = None) -> dict:
        """Push the expiry of ``holder``'s lease out to now + ttl."""
        device = _text(device, "device")
        holder = _text(holder, "holder")
        with self._txn() as (data, now, dirty):
            dev = self._dev(data, device)
            lease = dev["lease"]
            if lease is None or lease["holder"] != holder:
                self._refuse(device, dev, holder, now, op="bench.renew")
            self._touch(lease, now, _ttl(ttl_s) if ttl_s is not None else float(lease["ttl_s"]))
            dirty[0] = True
            return {"renewed": True, **self._device_view(device, dev, now, history=0)}

    def release(self, device: str, holder: str, firmware_left: str | None = None,
                *, force: bool = False) -> dict:
        """Give the board back. ``firmware_left`` says what is on it now.

        A queued holder that releases leaves the queue. ``force`` releases
        somebody else's lease (a session that died holding it); logged.
        """
        device = _text(device, "device")
        holder = _text(holder, "holder")
        fw = _text(firmware_left, "firmware_left", required=False) or None
        with self._txn() as (data, now, dirty):
            dev = self._dev(data, device)
            lease = dev["lease"]
            if lease is not None and (lease["holder"] == holder or force):
                kind = "released" if lease["holder"] == holder else "forced-release"
                entry = {"kind": kind, "holder": lease["holder"],
                         "purpose": lease.get("purpose", ""), "since": lease["since"],
                         "until": now, "firmware_left": fw}
                if kind == "forced-release":
                    entry["by"] = holder
                    self._log(f"bench: {holder} force-released {device} from {lease['holder']}")
                self._history(dev, entry)
                dev["firmware"] = {"desc": fw, "by": holder, "at": now}
                if fw is None:
                    dev["firmware"]["note"] = "released without saying: unknown"
                self._vacate(dev, now)
                dirty[0] = True
                released = True
            elif any(q["holder"] == holder for q in dev["queue"]):
                dev["queue"] = [q for q in dev["queue"] if q["holder"] != holder]
                dirty[0] = True
                released = False
            else:
                if lease is None:
                    raise HilError(BAD_ARGS, f"{holder} holds no lease on {device} and is not "
                                             "queued for it", hint="bench.status shows who is",
                                   device=device)
                self._refuse(device, dev, holder, now, op="bench.release",
                             hint="force=true releases a lease whose holder is gone (logged)")
            view = self._device_view(device, dev, now, history=1)
        return {"released": released, "left_queue": not released, **view}

    def check(self, device: str, holder: str | None, op: str, *, force: bool = False) -> dict:
        """May ``holder`` touch ``device`` now? Raises BENCH_BUSY when not.

        Free board: yes (nothing is claimed, nobody is waiting for it). The
        holder: yes, and the lease is renewed. ``force``: yes, logged.
        """
        device = _text(device, "device")
        holder = (holder or "").strip() or None
        if self.mode == "off":
            return {"allowed": True, "device": device, "mode": "off"}
        with self._txn() as (data, now, dirty):
            dev = data["devices"].get(device)
            if dev is None:
                return {"allowed": True, "device": device, "lease": None}
            dev = self._dev(data, device)
            lease = dev["lease"]
            if lease is not None and holder is not None and lease["holder"] == holder:
                if self._touch(lease, now):
                    dirty[0] = True
                return {"allowed": True, "device": device, "lease": self._lease_view(lease, now)}
            reserved = self._reserved_for(dev, now)
            blocked = lease is not None or (reserved is not None and reserved[0] != holder)
            if not blocked:
                return {"allowed": True, "device": device,
                        "lease": self._lease_view(lease, now)}
            if force:
                over = lease["holder"] if lease is not None else reserved[0]  # type: ignore[index]
                self._history(dev, {"kind": "force", "holder": holder or "?", "over": over,
                                    "op": op, "at": now})
                self._log(f"bench: {holder or 'an anonymous caller'} forced {op} on {device} "
                          f"held by {over}")
                dirty[0] = True
                return {"allowed": True, "forced": True, "device": device,
                        "lease": self._lease_view(lease, now)}
            if self.mode == "warn":
                try:
                    self._refuse(device, dev, holder, now, op=op)
                except HilError as e:
                    self._log(f"bench (warn only): {e.message}")
                    return {"allowed": True, "device": device, "warning": e.message,
                            "lease": self._lease_view(lease, now)}
            self._refuse(device, dev, holder, now, op=op)
        raise AssertionError("unreachable")  # pragma: no cover

    def check_all(self, holder: str | None, op: str, *, force: bool = False) -> dict:
        """``check`` against every device that has a lease: for ops (an ST-Link
        flash) that cannot say which board they will touch."""
        with self._txn() as (data, _now, _dirty):
            devices = sorted(data["devices"])
        results = [self.check(d, holder, op, force=force) for d in devices]
        return {"allowed": True, "devices": results}

    def touch(self, device: str, holder: str | None) -> None:
        """Renew ``holder``'s lease if it has one; otherwise nothing. Never raises."""
        if not holder:
            return
        try:
            with self._txn() as (data, now, dirty):
                dev = data["devices"].get(device)
                lease = None if dev is None else dev.get("lease")
                if lease is not None and lease["holder"] == holder and self._touch(lease, now):
                    dirty[0] = True
        except OSError as e:
            self._log(f"bench: renewing {holder} on {device}: {e}")

    def status(self, device: str | None = None, history: int = 10) -> dict:
        with self._txn() as (data, now, _dirty):
            names = [device] if device is not None else sorted(data["devices"])
            # _dev() may add an empty entry for an unknown device; nothing is written back.
            devices = [self._device_view(n, self._dev(data, n), now, history) for n in names]
        return {"now": now, "now_at": _hhmm(now), "path": str(self.path), "devices": devices}

    def known_devices(self) -> list[str]:
        with self._txn() as (data, _now, _dirty):
            return sorted(data["devices"])

    # ----------------------------------------------------------------- errors

    def _refuse(self, device: str, dev: dict, holder: str | None, now: float, *, op: str,
                hint: str | None = None) -> None:
        lease = dev.get("lease")
        queue = [q["holder"] for q in dev.get("queue") or []]
        if lease is not None:
            msg = self._busy_message(device, lease, now)
            owner = lease["holder"]
            details: dict[str, Any] = {"holder": owner, "purpose": lease.get("purpose", ""),
                                       "since": lease["since"], "expires": lease["expires"],
                                       "expires_in_s": round(float(lease["expires"]) - now, 1)}
        else:
            reserved = self._reserved_for(dev, now)
            assert reserved is not None
            owner = reserved[0]
            msg = (f"{device} is free but held for {owner}, first in the queue, until "
                   f"{_hhmm(reserved[1])[-5:]} (in {_mins(reserved[1] - now)})")
            details = {"holder": owner, "reserved_until": reserved[1]}
        who = f"you ({holder})" if holder else "this caller (no holder given)"
        raise HilError(
            BENCH_BUSY, f"{msg}; {who} may not run {op}",
            hint=hint or ("claim it to join the queue (crosspad_bench_claim / `crosspad-hil "
                          "bench claim`), crosspad_bench_status shows the queue; force=true "
                          f"overrides and is logged in {device}'s history"),
            device=device, op=op, queue=queue, caller=holder, **details)

    # ----------------------------------------------------------------- markdown

    def _write_md(self, data: dict, now: float) -> None:
        if self.md_path is None:
            return
        try:
            views = [self._device_view(n, data["devices"][n], now, history=8)
                     for n in sorted(data["devices"])]
            write_markdown(self.md_path, {"now": now, "now_at": _hhmm(now), "devices": views})
        except OSError as e:
            self._log(f"bench: writing {self.md_path}: {e}")


# --------------------------------------------------------------------------- markdown view

def render_markdown(status: dict) -> str:
    """The status as the human-readable block BENCH.md carries between its markers."""
    lines = [f"_Generated by crosspad-hil from the bench lease at {status['now_at']}; "
             "do not edit by hand._", ""]
    if not status["devices"]:
        lines.append("No board has been claimed yet.")
    for d in status["devices"]:
        lines.append(f"### {d['device']}")
        lines.append("")
        fw = d.get("firmware")
        if fw:
            desc = fw.get("desc") or fw.get("note") or "unknown"
            lines.append(f"On the board: {desc} (left by {fw.get('by')}, {_hhmm(fw.get('at'))})")
            lines.append("")
        lines.append("**Now**")
        lines.append("")
        lv = d.get("lease")
        if lv:
            purpose = f" — {lv['purpose']}" if lv.get("purpose") else ""
            lines.append(f"- {lv['holder']}{purpose}; since {lv['since_at']}, "
                         f"lease to {lv['expires_at']}")
        elif d.get("reserved_for"):
            r = d["reserved_for"]
            lines.append(f"- free, held for {r['holder']} until {r['until_at']}")
        else:
            lines.append("- free")
        lines.append("")
        lines.append("**Queue**")
        lines.append("")
        if d["queue"]:
            for q in d["queue"]:
                purpose = f" — {q['purpose']}" if q.get("purpose") else ""
                lines.append(f"{q['position']}. {q['holder']}{purpose} (since {q['since_at']})")
        else:
            lines.append("- empty")
        lines.append("")
        lines.append("**History**")
        lines.append("")
        if d["history"]:
            for h in reversed(d["history"]):
                lines.append("- " + _history_line(h))
        else:
            lines.append("- none")
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def _history_line(h: dict) -> str:
    kind = h.get("kind")
    if kind in ("force",):
        return f"{_hhmm(h.get('at'))} {h.get('holder')} forced {h.get('op')} over {h.get('over')}"
    fw = h.get("firmware_left")
    left = f"; left: {fw}" if fw else "; left: unknown"
    by = f" by {h['by']}" if h.get("by") else ""
    purpose = f" ({h['purpose']})" if h.get("purpose") else ""
    return (f"{_hhmm(h.get('until'))} {h.get('holder')}{purpose}: {kind}{by}"
            f" (held since {_hhmm(h.get('since'))[-5:]}){left}")


def write_markdown(path: Path | str, status: dict) -> None:
    """Replace the block between ``MD_BEGIN``/``MD_END`` in ``path`` (append one when
    the file has none), leaving the human-written text around it alone."""
    path = Path(path)
    block = f"{MD_BEGIN}\n{render_markdown(status)}{MD_END}"
    try:
        text = path.read_text(encoding="utf-8")
    except FileNotFoundError:
        text = ""
    start, end = text.find(MD_BEGIN), text.find(MD_END)
    if start != -1 and end > start:
        new = text[:start] + block + text[end + len(MD_END):]
    else:
        new = (text.rstrip("\n") + "\n\n" if text else "") + block + "\n"
    if new != text:
        tmp = path.with_name(path.name + ".tmp")
        tmp.write_text(new, encoding="utf-8")
        os.replace(tmp, path)
