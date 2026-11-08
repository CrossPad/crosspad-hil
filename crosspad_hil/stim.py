"""Pad stimulus: play the pads on a schedule, and account for what did not arrive.

Ported from ``platform-idf/tools/hil_speedtest.py`` (the scheduler, the rate
sweep, the transport comparison and the loss accounting against ``PAD_STATS``),
``hil_velocity.py`` (gate and velocity sets) and the ``Player`` thread of
``hil_kit_churn.py`` (fire-and-forget writes that never stop for the thing
being tested).

Three facts shape this module.

* **A hit is two commands, and the queue is 64 deep.** On the ``cdc`` transport
  every hit costs a ``PAD_PRESS`` and a ``PAD_RELEASE`` in the firmware's
  ``app_queue``, which the priority-1 main loop drains and which *drops on
  overflow*. The requested rate is therefore clamped against that queue
  (:data:`CDC_MAX_CMD_RATE_HZ`) and the clamp is reported in the plan — a
  stimulus that quietly ran at a third of the rate you asked for is worse than
  one that refused.
* **Drops are reported, never hidden.** ``CDC_STATS`` drop is sampled before
  the run and while it runs; every increase is timestamped into
  ``drop_events``, and (unless ``backoff=False``) the remaining schedule is
  stretched so the run stops digging. The final status always carries
  ``cdc_drops``.
* **The device's own counters are the truth.** ``PAD_STATS`` says how many
  presses the firmware accepted (``press``) and how many started a voice
  (``played``). The gap between what the host sent and ``press`` is transport
  loss; the gap between ``press`` and ``played`` is loss inside the device.
  Both are reported separately, because they have different causes and
  different fixes.

The SysEx transports do not go through ``app_queue`` at all — they reach the
pad path over the STM bridge (``sysex_stm``) or the ESP's own MIDI port
(``sysex_esp``) — so they are not clamped. Counters still come off CDC, which
is why a sweep comparing transports needs the board in the default USB profile.
"""
from __future__ import annotations

import heapq
import random
import statistics
import threading
import time
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Protocol

from . import verbs
from .cdc import CdcLink
from .devices import Device, UsbMode
from .errors import BAD_ARGS, NO_CDC_IN_AUDIO_MODE, NO_DEVICE, HilError
from .midi import MidiIO, MidiRole, sysex_pad
from .serial_open import open_serial

TRANSPORTS: tuple[str, ...] = ("cdc", "sysex_stm", "sysex_esp")
PAD_COUNT = 16

DEFAULT_RATE_HZ = 8.0            # design §5: "default 8 Hz per pad, burst 16"
DEFAULT_VELOCITY = 100           # hil_speedtest --vel
DEFAULT_GATE_MS = 25.0           # hil_speedtest --gate-ms; hil_velocity uses 80
DEFAULT_SECONDS = 10.0

# app_queue is 64 deep and drops on overflow (main/main.cpp); a hit is two
# commands. 32 commands/s keeps the queue under half depth even if the prio-1
# main loop stalls for a second, which is what a heavy console burst does.
CDC_QUEUE_DEPTH = 64
CMDS_PER_HIT = 2
CDC_MAX_CMD_RATE_HZ = 32.0
CDC_MAX_HIT_RATE_HZ = CDC_MAX_CMD_RATE_HZ / CMDS_PER_HIT

MIN_RATE_HZ = 0.1
DROP_POLL_S = 2.0                # how often the watcher reads CDC_STATS
BACKOFF_FACTOR = 2.0             # schedule stretch applied when drops appear
SETTLE_S = 1.0                   # drain before believing the counters (hil_speedtest --settle)
SWEEP_SETTLE_S = 1.5
SPIN_S = 0.002                   # hil_speedtest.precise_sleep_until: sleep, then spin ~1 ms


# --------------------------------------------------------------------------- schedule

@dataclass
class Hit:
    """One pad hit on the schedule. ``t_ms`` is relative to the start of the run."""

    t_ms: float
    pad: int
    vel: int = DEFAULT_VELOCITY
    gate_ms: float = DEFAULT_GATE_MS

    def to_dict(self) -> dict:
        return {"t_ms": self.t_ms, "pad": self.pad, "vel": self.vel, "gate_ms": self.gate_ms}


def parse_pads(spec: str | Sequence[int]) -> list[int]:
    """``"0,2,4-7"`` -> ``[0, 2, 4, 5, 6, 7]`` (hil_speedtest.parse_pads)."""
    if not isinstance(spec, str):
        pads = [int(p) for p in spec]
    else:
        pads = []
        for part in spec.split(","):
            part = part.strip()
            if not part:
                continue
            try:
                if "-" in part[1:]:
                    a, b = part.split("-", 1)
                    pads.extend(range(int(a), int(b) + 1))
                else:
                    pads.append(int(part))
            except ValueError as e:
                raise HilError(BAD_ARGS, f"bad pad spec {spec!r} (near {part!r})",
                               hint="pads look like '0,2,4-7'") from e
    bad = [p for p in pads if not 0 <= p < PAD_COUNT]
    if bad:
        raise HilError(BAD_ARGS, f"pads must be 0..{PAD_COUNT - 1}, got {bad}", pads=bad)
    if not pads:
        raise HilError(BAD_ARGS, f"no pads in {spec!r}", hint="pads look like '0,2,4-7'")
    return pads


def _check_range(name: str, value: float, lo: float, hi: float) -> float:
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise HilError(BAD_ARGS, f"{name} must be a number, got {value!r}")
    v = float(value)
    if not lo <= v <= hi:
        raise HilError(BAD_ARGS, f"{name} must be {lo}..{hi}, got {v}")
    return v


def build_schedule(pads: Sequence[int], *, rate_hz: float = DEFAULT_RATE_HZ,
                   seconds: float = DEFAULT_SECONDS,
                   velocity: int | Sequence[int] = DEFAULT_VELOCITY,
                   gate_ms: float = DEFAULT_GATE_MS, humanize_ms: float = 0.0,
                   chord: bool = False, seed: int | None = None) -> list[Hit]:
    """Round-robin the pads at ``rate_hz`` for ``seconds`` (``chord`` hits all at once).

    ``velocity`` may be a set of velocities (hil_velocity's ``--vels``), cycled
    across the hits so consecutive hits take different paths through the engine.
    ``humanize_ms`` jitters each press by ±that many milliseconds, seeded so a
    run is reproducible.
    """
    pads = parse_pads(pads)
    rate_hz = _check_range("rate_hz", rate_hz, MIN_RATE_HZ, 1000.0)
    seconds = _check_range("seconds", seconds, 0.0, 24 * 3600.0)
    gate_ms = _check_range("gate_ms", gate_ms, 0.0, 10_000.0)
    humanize_ms = _check_range("humanize_ms", humanize_ms, 0.0, 1000.0)
    vels = [velocity] if isinstance(velocity, int) else list(velocity)
    if not vels:
        raise HilError(BAD_ARGS, "velocity set is empty")
    for v in vels:
        _check_range("velocity", v, 0, 127)

    rnd = random.Random(seed)
    period_ms = 1000.0 / rate_hz
    # The gate collapses toward the period at high rates, which is what fast
    # fingerdrumming actually does (hil_speedtest.run_step).
    gate = min(gate_ms, period_ms * 0.4) if gate_ms > period_ms * 0.4 else gate_ms
    beats = int(seconds * rate_hz + 1e-9)
    out: list[Hit] = []
    for n in range(beats):
        t = n * period_ms
        if humanize_ms:
            t = max(0.0, t + rnd.uniform(-humanize_ms, humanize_ms))
        vel = int(vels[n % len(vels)])
        if chord:
            out.extend(Hit(t_ms=t, pad=p, vel=vel, gate_ms=gate) for p in pads)
        else:
            out.append(Hit(t_ms=t, pad=pads[n % len(pads)], vel=vel, gate_ms=gate))
    out.sort(key=lambda h: h.t_ms)
    return out


def pattern_to_hits(pattern: Iterable[Mapping[str, Any]]) -> list[Hit]:
    """``[{t_ms, pad, vel?, gate_ms?}, …]`` -> hits, validated before a byte is written."""
    hits: list[Hit] = []
    for i, item in enumerate(pattern):
        if not isinstance(item, Mapping):
            raise HilError(BAD_ARGS, f"pattern[{i}] must be an object, got {type(item).__name__}")
        unknown = set(item) - {"t_ms", "pad", "vel", "gate_ms"}
        if unknown:
            raise HilError(BAD_ARGS, f"pattern[{i}] has unknown keys {sorted(unknown)}",
                           hint="a step is {t_ms, pad, vel?, gate_ms?}")
        if "t_ms" not in item or "pad" not in item:
            raise HilError(BAD_ARGS, f"pattern[{i}] needs t_ms and pad")
        pad = item["pad"]
        if not isinstance(pad, int) or isinstance(pad, bool) or not 0 <= pad < PAD_COUNT:
            raise HilError(BAD_ARGS, f"pattern[{i}].pad must be 0..{PAD_COUNT - 1}, got {pad!r}")
        hits.append(Hit(
            t_ms=_check_range(f"pattern[{i}].t_ms", item["t_ms"], 0.0, 24 * 3600_000.0),
            pad=pad,
            vel=int(_check_range(f"pattern[{i}].vel", item.get("vel", DEFAULT_VELOCITY), 0, 127)),
            gate_ms=_check_range(f"pattern[{i}].gate_ms", item.get("gate_ms", DEFAULT_GATE_MS),
                                 0.0, 10_000.0),
        ))
    if not hits:
        raise HilError(BAD_ARGS, "pattern is empty", hint="a step is {t_ms, pad, vel?, gate_ms?}")
    hits.sort(key=lambda h: h.t_ms)
    return hits


def throttle(rate_hz: float, transport: str) -> tuple[float, dict | None]:
    """Clamp ``rate_hz`` against the transport's queue. Returns (rate, report|None)."""
    if transport != "cdc" or rate_hz <= CDC_MAX_HIT_RATE_HZ:
        return rate_hz, None
    return CDC_MAX_HIT_RATE_HZ, {
        "requested_rate_hz": rate_hz,
        "applied_rate_hz": CDC_MAX_HIT_RATE_HZ,
        "reason": (f"a hit is {CMDS_PER_HIT} CDC commands and app_queue is "
                   f"{CDC_QUEUE_DEPTH} deep and drops on overflow; "
                   f"capped at {CDC_MAX_CMD_RATE_HZ:.0f} commands/s"),
        "hint": "use transport=sysex_stm to drive the pads faster than CDC sustains",
    }


# --------------------------------------------------------------------------- transports

class PadTransport(Protocol):
    """What the scheduler needs from a way of hitting a pad."""

    name: str

    def press(self, pad: int, vel: int) -> None: ...
    def release(self, pad: int) -> None: ...
    def close(self) -> None: ...


class CdcPadTransport:
    """``PAD_PRESS``/``PAD_RELEASE`` over CDC, fire-and-forget.

    Writes go through :meth:`CdcLink.send`, never ``transact`` — waiting for the
    ``OK`` would starve the stimulus, and the ``OK`` is not this command's
    anyway (``PAD_PRESS`` answers exactly like ``KIT_LOAD`` does).
    """

    name = "cdc"

    def __init__(self, link: CdcLink, *, owns: bool = False) -> None:
        self._link = link
        self._owns = owns

    def press(self, pad: int, vel: int) -> None:
        self._link.send(f"PAD_PRESS {pad} {vel}")

    def release(self, pad: int) -> None:
        self._link.send(f"PAD_RELEASE {pad}")

    def close(self) -> None:
        if self._owns:
            self._link.close()


class SysexPadTransport:
    """SysEx ``0x1D 0x07/0x08`` on the STM bridge or the ESP MIDI port."""

    def __init__(self, io: Any, role: MidiRole, *, owns: bool = True) -> None:
        self._io = io
        self._owns = owns
        self.name = "sysex_stm" if role == MidiRole.STM else "sysex_esp"

    def press(self, pad: int, vel: int) -> None:
        self._io.send_sysex(sysex_pad(True, pad, vel))

    def release(self, pad: int) -> None:
        self._io.send_sysex(sysex_pad(False, pad))

    def close(self) -> None:
        if self._owns:
            self._io.close()


def _require_cdc(device: Device) -> str:
    if device.ports.cdc is None:
        if device.usb_mode == UsbMode.AUDIO:
            raise HilError(NO_CDC_IN_AUDIO_MODE,
                           f"{device.id} is in the MIDI+UAC2 profile; CDC endpoint absent",
                           hint="usbmode.set mode=default, or transport=sysex_stm",
                           device=device.id)
        raise HilError(NO_DEVICE, f"{device.id} has no CDC port "
                                  f"(usb_mode={device.usb_mode.value})",
                       hint="is the board in bootloader/DFU?", device=device.id)
    return device.ports.cdc.path


def open_transport(device: Device, kind: str = "cdc", *,
                   link: CdcLink | None = None,
                   serial_factory: Callable[..., Any] = open_serial,
                   midi_factory: Callable[..., Any] = MidiIO) -> PadTransport:
    """Open the pad transport named by ``kind``; the caller closes it."""
    if kind not in TRANSPORTS:
        raise HilError(BAD_ARGS, f"unknown transport {kind!r}",
                       hint=f"one of {', '.join(TRANSPORTS)}")
    if kind == "cdc":
        if link is not None:
            return CdcPadTransport(link, owns=False)
        port = _require_cdc(device)
        own = CdcLink(port, serial_factory=serial_factory)
        own.open()
        return CdcPadTransport(own, owns=True)
    role = MidiRole.STM if kind == "sysex_stm" else MidiRole.ESP
    io = midi_factory(device, role)
    io.open()
    return SysexPadTransport(io, role)


# --------------------------------------------------------------------------- counters

@dataclass
class CounterSample:
    press: int
    release: int
    played: int
    freeslots: int
    drop: int

    def to_dict(self) -> dict:
        return {"press": self.press, "release": self.release, "played": self.played,
                "freeslots": self.freeslots, "drop": self.drop}


class Counters:
    """``PAD_STATS`` + ``CDC_STATS``, serialized so the watcher and ``status()`` share them."""

    def __init__(self, link: CdcLink, *, owns: bool = False) -> None:
        self._link = link
        self._owns = owns
        self._lock = threading.Lock()

    def sample(self) -> CounterSample:
        with self._lock:
            pad = verbs.pad_stats(self._link)
            try:
                drop = int(verbs.cdc_stats(self._link)["drop"])
            except HilError:
                drop = 0        # firmware predating CDC_STATS; pad counters still hold
            return CounterSample(press=pad["press"], release=pad["release"],
                                 played=pad["played"], freeslots=pad["freeslots"], drop=drop)

    def reset(self) -> CounterSample:
        with self._lock:
            verbs.pad_stats(self._link, reset=True)
        return self.sample()

    def close(self) -> None:
        if self._owns:
            self._link.close()


def open_counters(device: Device, *, link: CdcLink | None = None,
                  serial_factory: Callable[..., Any] = open_serial) -> Counters | None:
    """A CDC counter source, or ``None`` when the board has no reachable CDC."""
    if link is not None:
        return Counters(link, owns=False)
    if device.ports.cdc is None:
        return None
    own = CdcLink(device.ports.cdc.path, serial_factory=serial_factory)
    own.open()
    return Counters(own, owns=True)


# --------------------------------------------------------------------------- the run

def _precise_sleep_until(target: float, halt: threading.Event,
                         clock: Callable[[], float]) -> bool:
    """Sleep until ``target``; returns False if ``halt`` was set first.

    hil_speedtest.precise_sleep_until: plain ``sleep()`` overshoots badly at
    short periods, so the last ~1 ms is spun.
    """
    while True:
        remaining = target - clock()
        if remaining <= 0:
            return not halt.is_set()
        if remaining > SPIN_S:
            if halt.wait(remaining - SPIN_S / 2):
                return False
        elif halt.is_set():
            return False


@dataclass
class StimPlan:
    """What the stimulus is going to do, including what it refused to do."""

    transport: str
    pads: list[int]
    hits: int
    seconds: float
    rate_hz: float | None
    velocity: Any
    gate_ms: float | None
    humanize_ms: float
    throttled: dict | None = None
    counters: str | None = None      # None = available, else why not

    def to_dict(self) -> dict:
        return {"transport": self.transport, "pads": self.pads, "hits": self.hits,
                "seconds": self.seconds, "rate_hz": self.rate_hz, "velocity": self.velocity,
                "gate_ms": self.gate_ms, "humanize_ms": self.humanize_ms,
                "throttled": self.throttled, "counters_unavailable": self.counters}


class Stimulus:
    """A schedule, a transport and the device's own counters around it.

    ``start()`` returns immediately; ``status()`` is safe to call while it runs
    and re-reads the counters; ``stop()`` halts, releases every pad it pressed,
    lets the queue drain and returns the final status.
    """

    def __init__(self, schedule: Sequence[Hit], transport: PadTransport, *,
                 plan: StimPlan | None = None, counters: Counters | None = None,
                 reset_counters: bool = True, backoff: bool = True,
                 drop_poll_s: float = DROP_POLL_S, settle_s: float = SETTLE_S,
                 clock: Callable[[], float] = time.monotonic) -> None:
        self._schedule = list(schedule)
        self._transport = transport
        self._counters = counters
        self._reset_counters = reset_counters
        self._backoff = backoff
        self._drop_poll_s = drop_poll_s
        self._settle_s = settle_s
        self._clock = clock
        self.plan = plan or StimPlan(
            transport=transport.name,
            pads=sorted({h.pad for h in self._schedule}),
            hits=len(self._schedule), seconds=self._span_s(), rate_hz=None,
            velocity=sorted({h.vel for h in self._schedule}), gate_ms=None, humanize_ms=0.0)

        self._halt = threading.Event()
        self._finished = threading.Event()
        self._lock = threading.Lock()
        self._thread: threading.Thread | None = None
        self._watcher: threading.Thread | None = None
        self._t0: float | None = None
        self._t_end: float | None = None
        self._sent = 0
        self._presses = 0
        self._releases = 0
        self._errors: list[str] = []
        self._sched_err_ms: list[float] = []
        self._held: set[int] = set()
        self._baseline: CounterSample | None = None
        self._last: CounterSample | None = None
        self._drop_events: list[dict] = []
        self._stretch = 1.0
        self._anchor_t = 0.0
        self._anchor_ms = 0.0
        self._closed = False

    # -- lifecycle -----------------------------------------------------
    def _span_s(self) -> float:
        return (max(h.t_ms + h.gate_ms for h in self._schedule) / 1000.0
                if self._schedule else 0.0)

    def start(self) -> StimPlan:
        if self._thread is not None:
            raise HilError(BAD_ARGS, "this stimulus has already been started")
        if self._counters is not None:
            self._baseline = (self._counters.reset() if self._reset_counters
                              else self._counters.sample())
            self._last = self._baseline
        self._thread = threading.Thread(target=self._run, name="stim", daemon=True)
        self._thread.start()
        if self._counters is not None and self._drop_poll_s > 0:
            self._watcher = threading.Thread(target=self._watch, name="stim-drops", daemon=True)
            self._watcher.start()
        return self.plan

    @property
    def running(self) -> bool:
        return self._thread is not None and not self._finished.is_set()

    def wait(self, timeout_s: float | None = None) -> bool:
        """Wait for the schedule to finish. Returns False on timeout."""
        return self._finished.wait(timeout_s)

    def stop(self, timeout_s: float = 5.0) -> dict:
        """Halt, release what is held, drain, read the counters one last time."""
        self._halt.set()
        if self._thread is not None:
            self._thread.join(timeout_s)
        if self._watcher is not None:
            self._watcher.join(timeout_s)
        self._release_held()
        if self._counters is not None and self._settle_s > 0:
            time.sleep(self._settle_s)
        st = self.status(refresh=True)
        st["running"] = False
        self._close()
        return st

    def _close(self) -> None:
        if self._closed:
            return
        self._closed = True
        for closer in (self._transport.close,
                       self._counters.close if self._counters else lambda: None):
            try:
                closer()
            except Exception as e:  # noqa: BLE001 - closing must not mask the run's result
                self._errors.append(f"close: {e}")

    # -- the scheduler -------------------------------------------------
    def _target(self, t_ms: float) -> float:
        return self._anchor_t + (t_ms - self._anchor_ms) / 1000.0 * self._stretch

    def _run(self) -> None:
        try:
            t0 = self._clock()
            with self._lock:
                self._t0 = t0
            self._anchor_t = t0
            self._anchor_ms = 0.0
            # (release time, seq, pad): a release is queued when its press
            # actually fires, so a halted or stretched run still lets go.
            pending: list[tuple[float, int, int]] = []
            seq = 0
            for i, hit in enumerate(self._schedule):
                target = self._target(hit.t_ms)
                while pending and pending[0][0] <= target:
                    release_at, _, pad = heapq.heappop(pending)
                    if not _precise_sleep_until(release_at, self._halt, self._clock):
                        return
                    self._do_release(pad)
                if not _precise_sleep_until(target, self._halt, self._clock):
                    return
                self._sched_err_ms.append((self._clock() - target) * 1000.0)
                self._do_press(hit.pad, hit.vel)
                seq += 1
                heapq.heappush(pending, (target + hit.gate_ms / 1000.0, seq, hit.pad))
                if self._backoff:
                    self._maybe_backoff(i)
            while pending:
                release_at, _, pad = heapq.heappop(pending)
                if not _precise_sleep_until(release_at, self._halt, self._clock):
                    return
                self._do_release(pad)
        finally:
            with self._lock:
                self._t_end = self._clock()
            self._finished.set()

    def _do_press(self, pad: int, vel: int) -> None:
        try:
            self._transport.press(pad, vel)
        except Exception as e:  # noqa: BLE001 - a dead port must not kill the thread silently
            self._errors.append(f"press {pad}: {e}")
            self._halt.set()
            return
        with self._lock:
            self._sent += 1
            self._presses += 1
            self._held.add(pad)

    def _do_release(self, pad: int) -> None:
        try:
            self._transport.release(pad)
        except Exception as e:  # noqa: BLE001
            self._errors.append(f"release {pad}: {e}")
            self._halt.set()
            return
        with self._lock:
            self._sent += 1
            self._releases += 1
            self._held.discard(pad)

    def _release_held(self) -> None:
        with self._lock:
            held = sorted(self._held)
        for pad in held:
            self._do_release(pad)

    # -- drops ---------------------------------------------------------
    def _watch(self) -> None:
        while not self._halt.wait(self._drop_poll_s) and not self._finished.is_set():
            try:
                sample = self._counters.sample() if self._counters else None
            except HilError as e:
                self._errors.append(f"counters: {e}")
                continue
            if sample is None:
                continue
            with self._lock:
                previous = self._last
                self._last = sample
            if previous is not None and sample.drop > previous.drop:
                self._note_drops(sample.drop - previous.drop, sample.drop)

    def _note_drops(self, delta: int, total: int) -> None:
        with self._lock:
            self._drop_events.append({
                "at_s": round((self._clock() - (self._t0 or self._clock())), 3),
                "dropped": delta, "total": total, "stretch": self._stretch})

    def _maybe_backoff(self, index: int) -> None:
        """Stretch what is left of the schedule once the device says it is dropping."""
        with self._lock:
            unacted = [e for e in self._drop_events if not e.get("backoff")]
            if not unacted:
                return
            for e in unacted:
                e["backoff"] = True
            self._anchor_t = self._clock()
            self._anchor_ms = (self._schedule[index + 1].t_ms
                               if index + 1 < len(self._schedule) else 0.0)
            self._stretch *= BACKOFF_FACTOR

    # -- reporting -----------------------------------------------------
    def status(self, refresh: bool = True) -> dict:
        sample = None
        if refresh and self._counters is not None:
            try:
                sample = self._counters.sample()
                with self._lock:
                    self._last = sample
            except HilError as e:
                self._errors.append(f"counters: {e}")
        with self._lock:
            sample = sample or self._last
            t0, t_end = self._t0, self._t_end
            sent, presses, releases = self._sent, self._presses, self._releases
            errs = list(self._sched_err_ms)
            drop_events = [dict(e) for e in self._drop_events]
            errors = list(self._errors)
        elapsed = 0.0 if t0 is None else ((t_end or self._clock()) - t0)
        base = self._baseline
        delta = None
        drops = None
        if sample is not None and base is not None:
            delta = {"press": sample.press - base.press,
                     "release": sample.release - base.release,
                     "played": sample.played - base.played}
            drops = max(0, sample.drop - base.drop)
        out: dict[str, Any] = {
            "running": self.running,
            "plan": self.plan.to_dict(),
            "elapsed_s": round(elapsed, 3),
            "sent": sent,
            "presses_sent": presses,
            "releases_sent": releases,
            "hits_planned": len(self._schedule),
            "schedule_error_ms": _percentiles(errs),
            "pad_stats": sample.to_dict() if sample else None,
            "pad_stats_baseline": base.to_dict() if base else None,
            "pad_stats_delta": delta,
            "cdc_drops": drops,
            "drop_events": drop_events,
            "loss": loss_accounting(presses, delta) if delta else None,
            "errors": errors,
        }
        return out


def _percentiles(values: Sequence[float]) -> dict | None:
    if not values:
        return None
    ordered = sorted(values)
    return {"p50": round(statistics.median(ordered), 2),
            "p90": round(ordered[min(len(ordered) - 1, int(len(ordered) * 0.9))], 2),
            "max": round(ordered[-1], 2)}


def loss_accounting(sent_presses: int, delta: Mapping[str, int]) -> dict:
    """hil_speedtest's three losses: transport, engine, and the two together.

    ``sent -> press`` is the host-to-firmware leg (USB, the STM bridge, the ESP
    LPUART, ``app_queue``); ``press -> played`` is inside the device (EventBus
    overflow, or no free voice slot).
    """
    press = int(delta.get("press", 0))
    played = int(delta.get("played", 0))
    tx = 100.0 * (sent_presses - press) / sent_presses if sent_presses else 0.0
    engine = 100.0 * (press - played) / press if press else 0.0
    total = 100.0 * (sent_presses - played) / sent_presses if sent_presses else 0.0
    return {"sent": sent_presses, "accepted": press, "played": played,
            "transport_pct": round(max(0.0, tx), 2),
            "engine_pct": round(max(0.0, engine), 2),
            "total_pct": round(max(0.0, total), 2)}


# --------------------------------------------------------------------------- entry points

def plan_stimulus(*, pads: str | Sequence[int] | None = None, rate_hz: float | None = None,
                  pattern: Sequence[Mapping[str, Any]] | None = None,
                  seconds: float = DEFAULT_SECONDS,
                  velocity: int | Sequence[int] = DEFAULT_VELOCITY,
                  gate_ms: float = DEFAULT_GATE_MS, humanize_ms: float = 0.0,
                  chord: bool = False, transport: str = "cdc",
                  seed: int | None = None) -> tuple[list[Hit], StimPlan]:
    """Turn CLI/daemon arguments into a schedule plus the plan that describes it."""
    if transport not in TRANSPORTS:
        raise HilError(BAD_ARGS, f"unknown transport {transport!r}",
                       hint=f"one of {', '.join(TRANSPORTS)}")
    if pattern is not None:
        if rate_hz is not None:
            raise HilError(BAD_ARGS, "pass rate_hz or pattern, not both",
                           hint="a pattern already carries its own timing")
        hits = pattern_to_hits(pattern)
        plan = StimPlan(transport=transport, pads=sorted({h.pad for h in hits}), hits=len(hits),
                        seconds=round(max(h.t_ms + h.gate_ms for h in hits) / 1000.0, 3),
                        rate_hz=None, velocity=sorted({h.vel for h in hits}),
                        gate_ms=None, humanize_ms=humanize_ms)
        return hits, plan
    pad_list = parse_pads(pads if pads is not None else list(range(PAD_COUNT)))
    applied, throttled = throttle(DEFAULT_RATE_HZ if rate_hz is None else float(rate_hz),
                                  transport)
    hits = build_schedule(pad_list, rate_hz=applied, seconds=seconds, velocity=velocity,
                          gate_ms=gate_ms, humanize_ms=humanize_ms, chord=chord, seed=seed)
    plan = StimPlan(transport=transport, pads=pad_list, hits=len(hits), seconds=float(seconds),
                    rate_hz=applied, velocity=velocity, gate_ms=gate_ms,
                    humanize_ms=humanize_ms, throttled=throttled)
    return hits, plan


def start_stimulus(device: Device, *, pads: str | Sequence[int] | None = None,
                   rate_hz: float | None = None,
                   pattern: Sequence[Mapping[str, Any]] | None = None,
                   seconds: float = DEFAULT_SECONDS,
                   velocity: int | Sequence[int] = DEFAULT_VELOCITY,
                   gate_ms: float = DEFAULT_GATE_MS, humanize_ms: float = 0.0,
                   chord: bool = False, transport: str = "cdc", seed: int | None = None,
                   link: CdcLink | None = None, backoff: bool = True,
                   drop_poll_s: float = DROP_POLL_S, settle_s: float = SETTLE_S,
                   serial_factory: Callable[..., Any] = open_serial,
                   midi_factory: Callable[..., Any] = MidiIO) -> Stimulus:
    """Open the transport and the counters, then start playing. Caller calls ``stop()``."""
    hits, plan = plan_stimulus(pads=pads, rate_hz=rate_hz, pattern=pattern, seconds=seconds,
                               velocity=velocity, gate_ms=gate_ms, humanize_ms=humanize_ms,
                               chord=chord, transport=transport, seed=seed)
    tr = open_transport(device, transport, link=link, serial_factory=serial_factory,
                        midi_factory=midi_factory)
    counters: Counters | None = None
    try:
        counter_link = link
        if counter_link is None and isinstance(tr, CdcPadTransport):
            counter_link = tr._link      # the pad link answers status queries too
        counters = open_counters(device, link=counter_link, serial_factory=serial_factory)
    except HilError as e:
        plan.counters = f"{e.code}: {e.message}"
    if counters is None and plan.counters is None:
        plan.counters = ("no CDC port on this device — PAD_STATS and CDC_STATS "
                         "are unreadable, so loss cannot be accounted")
    stim = Stimulus(hits, tr, plan=plan, counters=counters, backoff=backoff,
                    drop_poll_s=drop_poll_s, settle_s=settle_s)
    stim.start()
    return stim


def run_stimulus(device: Device, *, timeout_s: float | None = None, **kw: Any) -> dict:
    """Start, play to the end of the schedule, stop — the foreground CLI path."""
    stim = start_stimulus(device, **kw)
    try:
        stim.wait(timeout_s if timeout_s is not None else stim.plan.seconds + 30.0)
    finally:
        status = stim.stop()
    return status


# --------------------------------------------------------------------------- sweep

def sweep(device: Device, *, rates: Sequence[float] = (4, 8, 12, 16, 24, 32),
          pad_sets: Sequence[Sequence[int]] = ((0,),),
          transports: Sequence[str] = ("cdc",), seconds: float = 5.0,
          settle_s: float = SWEEP_SETTLE_S, velocity: int = DEFAULT_VELOCITY,
          gate_ms: float = DEFAULT_GATE_MS, chord: bool = False,
          loss_threshold_pct: float = 1.0, full_sweep: bool = False,
          progress: Callable[[int, int | None, str], None] | None = None,
          cancelled: threading.Event | None = None,
          **kw: Any) -> dict:
    """Ramp the hit rate until the device stops keeping up (hil_speedtest).

    Each step reports where the hits got to: sent -> accepted (``PAD_STATS
    press``) -> played. The first rate whose total loss exceeds
    ``loss_threshold_pct`` is that pad set's limit; without ``full_sweep`` the
    sweep stops there, because the shape past the limit costs minutes to
    measure and usually says the same thing.
    """
    rows: list[dict] = []
    limits: list[dict] = []
    steps = len(rates) * len(pad_sets) * len(transports)
    done = 0
    for tr_name in transports:
        for pads in pad_sets:
            pad_list = parse_pads(pads)
            label = f"{len(pad_list)}pad" + ("-chord" if chord else "")
            limit: float | None = None
            for rate in rates:
                if cancelled is not None and cancelled.is_set():
                    raise HilError("CANCELLED", "sweep cancelled")
                done += 1
                if progress:
                    progress(done, steps, f"{tr_name} {label} @ {rate} Hz")
                status = run_stimulus(device, pads=pad_list, rate_hz=rate, seconds=seconds,
                                      velocity=velocity, gate_ms=gate_ms, chord=chord,
                                      transport=tr_name, settle_s=settle_s, backoff=False, **kw)
                row = {"transport": tr_name, "padset": label, "pads": len(pad_list),
                       "rate_hz": float(rate), "requested_rate_hz": float(rate),
                       "applied_rate_hz": status["plan"]["rate_hz"],
                       "throttled": status["plan"]["throttled"],
                       "sent": status["presses_sent"],
                       "loss": status["loss"], "cdc_drops": status["cdc_drops"],
                       "schedule_error_ms": status["schedule_error_ms"],
                       "freeslots": (status["pad_stats"] or {}).get("freeslots")}
                rows.append(row)
                total = (status["loss"] or {}).get("total_pct")
                if total is not None and total > loss_threshold_pct and limit is None:
                    limit = float(rate)
                    limits.append({"transport": tr_name, "padset": label, "limit_rate_hz": rate,
                                   "loss_pct": total})
                    if not full_sweep:
                        break
            if limit is None:
                limits.append({"transport": tr_name, "padset": label, "limit_rate_hz": None,
                               "loss_pct": None})
    return {"rows": rows, "limits": limits, "loss_threshold_pct": loss_threshold_pct,
            "seconds_per_step": seconds}
