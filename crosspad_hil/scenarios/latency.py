"""How late is a pad, and how late is the host: measured on the board's own loop.

Two paths, each struck ``--hits`` times with a random gap (so nothing the
host or the board does on a fixed grid lines up with the hits):

``pad``
    A note-on on the **STM bridge's USB MIDI port** (note ``36 + pad``,
    channel 1). The STM forwards it byte for byte onto the UART to the ESP,
    which takes it through ``onNoteOn2Audio`` -- the very path a physical pad
    takes from the STM onward: UART, ``uartLoop``, PadManager, the event
    queue, the engine, the I2S DMA ring, the DAC. It is the earliest point a
    host can inject in the USB-audio profile; what it skips is the STM's own
    ADC scan and its debounce/velocity window (``STM_VPAD`` skips those too,
    and needs CDC, which this profile does not have).
``host``
    A noise burst written into the board's UAC OUT: the host's own audio,
    through TinyUSB's FIFO, the playback ring, the mixer's "USB" channel, the
    DMA ring and the DAC.

Both are heard on the PCB's DAC->ADC loop (capture preset ``headphone`` =
codec 1, the phones' DAC, by default) through the board's UAC IN, recorded by
one duplex stream on the card (``crosspad_hil.duplex``) whose clock log maps
every captured frame -- and every played one -- to the monotonic clock the
MIDI sends are stamped on. A hit's latency is ``onset time - send time`` (pad)
or ``onset time - the time its first frame left the host`` (host).

What the board reports of its capture path is subtracted: the capture ring's
level (``cap_ring`` in the console's ``SPK #`` lines, which the firmware prints
only while a UAC OUT stream plays -- the duplex stream keeps one playing).
What it does not report stays in the number and is listed under
``excludes``/``u_sys_ms``: the ADC's group delay, the I2S RX DMA queue (0-3
blocks, fixed per session), TinyUSB's IN FIFO. The headline is therefore
``*_jack_ms`` = "at the jack, give or take the unreported capture parts",
and ``*_uac_in_ms`` (nothing subtracted) is what a DAW recording the board
sees.

``pad_physical_est_ms`` adds the part no host stimulus reaches -- the STM's
scan and its press window, read from ``STM_REG 33C`` -- as a model (marked
"W": computed, not measured).

The board is put back: the kit, the app and the Mixer desk (``MIX_PRESET
UNDO``) it had, the capture route, the default USB profile. The SD card keeps
the ``HIL CLICK`` fixture kit once provisioned. Opening the STM console (for
the ``SPK #`` lines) resets the ESP once, as in every console-reading
scenario; ``--no-console`` skips it and with it the capture-ring subtraction
and the underrun watch.
"""
from __future__ import annotations

import csv
import json
import math
import random
import re
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from crosspad_hil import card_hold, fixtures, verbs
from crosspad_hil.capture import PRESETS, CaptureSession
from crosspad_hil.devices import UsbMode
from crosspad_hil.duplex import DEFAULT_LATENCY_MS, AlsaDuplexBackend
from crosspad_hil.errors import BAD_ARGS, ENV, NOT_SUPPORTED, TIMEOUT, HilError
from crosspad_hil.midi import MidiIO, MidiRole
from crosspad_hil.scenarios import _audio
from crosspad_hil.scenarios.base import (
    ENV_EXIT_CODES,
    Artifact,
    Context,
    Progress,
    Report,
    register,
)
from crosspad_hil.scenarios.velocity import echo_rtt
from crosspad_hil.usbmode import set_mode

PATHS: tuple[str, ...] = ("pad", "host")
SAMPLER_APP = "Sampler"
GATE_STATS: tuple[str, ...] = ("p50", "p95", "max")
# A prologue that fails with one of these is the bench's problem (exit 2), not the firmware's.
SETUP_CODES = ENV_EXIT_CODES | {TIMEOUT, NOT_SUPPORTED}

LEAD_S = 1.5             # stream settled, first SPK line in, a stretch of floor before hit 0
PHASE_PAUSE_S = 0.5
TAIL_S = 1.0             # the last hit's sound, and one more SPK line, after the last hit
PRE_TOL_MS = 3.0         # an onset this far *before* its reference is still found (clock error)
RUNUP_MS = 25.0          # what was sounding just before a window: decides "masked"
NEXT_GUARD_MS = 10.0     # a window stops this far short of the next hit's reference
EARLY_SHARE_ENV = 0.1    # more early onsets than this share: the clock map is wrong, not the board
SPK_GUARD_S = 1.5        # an SPK line follows its event by up to ~1 s (1 Hz worker) plus polling
SPK_WORKER_S = 1.0       # the worker prints a moved counter at its next pass, once a second
SPK_SEEN_MARGIN_S = 0.1  # the print's busy wait, the UART to the STM, the console's reader
ECHO_WINDOW_S = 0.1      # a note echo later than this is not the hit's

# The model for what a physical pad adds (W: from the STM's code, not measured).
# (min, typical, max) in ms; the window is relative to pressDebounceMs (STM_REG 33C).
USB_OUT_MS = (0.0, 0.5, 1.0)          # host -> STM bulk OUT: the bridge path has it, a pad not
STM_SCAN_MS = (0.25, 0.625, 1.0)      # TIM15 250 us scan, threshold crossing
WINDOW_OFFSET_MS = (-1.0, -0.125, 0.75)  # HAL tick 1 ms, pad updated every 0.25-0.75 ms
REG_PRESS_DEBOUNCE = 0x033C
# Capture-path parts the firmware does not report (S): ADC group delay ~0.3,
# I2S RX DMA queue 0-3 x 1.45 ms, TinyUSB IN FIFO and host scheduling ~0-1.5.
UNREPORTED_CAPTURE_MS = (0.3, 6.2)

GLITCH_KEYS: tuple[str, ...] = ("under", "refused", "cap_drop", "cap_under")
HOST_BURST_SEED = fixtures.click_seed(fixtures.SINE_PAD_COUNT)   # past every pad's seed
INT16 = 32767

PAD_INCLUDES = ["host MIDI TX (ALSA seq) and USB bulk OUT to the STM (0-1 ms)",
                "STM USB->UART relay, 460800 baud", "ESP UART RX timeout and uartLoop poll",
                "PadManager, crosspad_events queue, engine voice start (block quantisation)",
                "I2S TX DMA ring", "DAC", "PCB loop to the ADC"]
PAD_EXCLUDES = ["the STM's ADC scan and press/velocity window (see physical_est_ms)",
                "the capture ring (subtracted from jack_ms, from SPK cap_ring)"]
HOST_INCLUDES = ["USB isochronous OUT from the moment ALSA reports the frame played",
                 "TinyUSB OUT FIFO", "playback ring (SPK ring)", "mixer USB channel",
                 "I2S TX DMA ring", "DAC", "PCB loop to the ADC"]
HOST_EXCLUDES = ["the host's own output buffer (PipeWire, a DAW): measured from the card",
                 "the capture ring (subtracted from jack_ms, from SPK cap_ring)"]
NOT_SUBTRACTED = ["ADC group delay", "I2S RX DMA queue (0-3 blocks, fixed per session)",
                  "TinyUSB IN FIFO and USB IN"]


@dataclass
class Params:
    paths: list[str] = field(default_factory=lambda: list(PATHS),
                             metadata={"help": "what to measure: pad (note on the STM bridge -> "
                                               "jack), host (UAC OUT -> jack)"})
    hits: int = field(default=60, metadata={"help": "hits per path"})
    min_gap_ms: float = field(default=300.0, metadata={"help": "shortest random gap between "
                                                               "hits"})
    max_gap_ms: float = field(default=700.0, metadata={"help": "longest random gap"})
    seed: int | None = field(default=None, metadata={"help": "seed of the random gaps; a "
                                                             "fresh one when omitted, recorded "
                                                             "in the report either way"})
    pads: list[int] = field(default_factory=lambda: [0],
                            metadata={"help": "pads struck round-robin on the pad path"})
    vel: int = field(default=100, metadata={"help": "note velocity"})
    gate_ms: float = field(default=30.0, metadata={"help": "note-on -> note-off"})
    note_base: int = field(default=36, metadata={"help": "the STM's note of pad 0 (36 + "
                                                         "Settings' noteOffset)"})
    channel: int = field(default=0, metadata={"help": "MIDI channel 0-15 of the notes (the "
                                                      "ESP's STM input listens on channel 1 = 0)"})
    kit: str = field(default=fixtures.CLICK_NAME,
                     metadata={"help": "kit the pad path plays; HIL CLICK starts at full level "
                                       "on its first frame, a kit that fades in reads late"})
    provision: bool = field(default=True, metadata={"help": "build and push HIL CLICK when "
                                                            "the card lacks it"})
    mixer_desk: str = field(default="BEATS",
                            metadata={"help": "Mixer desk (MIX_PRESET slot name) for the run, "
                                              "MIX_PRESET UNDO afterwards: BEATS puts the pads "
                                              "and the computer on both outputs; '' leaves "
                                              "the desk alone"})
    preset: str = field(default="headphone",
                        metadata={"help": "capture path: headphone (codec 1, the phones' "
                                          "DAC->ADC loop) or speaker (codec 0)"})
    window_ms: float = field(default=200.0, metadata={"help": "latest onset searched for "
                                                              "after a hit"})
    host_level: float = field(default=0.8, metadata={"help": "peak of the host burst, 0..1 "
                                                             "(the level does not move an "
                                                             "onset; a quiet loop needs it)"})
    in2_gain: int | None = field(
        default=None,
        metadata={"help": "pin codec 1's line preamp (INGAIN IN2) at this many dB for the run "
                          "and put the old value back (the board keeps it in NVS); for a "
                          "headphone loop too quiet to time. Default: leave it alone"})
    free_loop: bool = field(
        default=True,
        metadata={"help": "on an input pair that hears Line 3/4 (which switches the jack "
                          "loop's driver off), select a pair without it for the run and put "
                          "the old one back (the board keeps it in NVS); for the LINE1 presets"})
    max_pad_ms: float | None = field(default=None,
                                     metadata={"help": "FAIL when the pad path's headline "
                                                       "(--gate-stat) is above this"})
    max_host_ms: float | None = field(default=None,
                                      metadata={"help": "FAIL when the host path's headline "
                                                        "(--gate-stat) is above this"})
    gate_stat: str = field(default="p50", metadata={"help": "statistic the thresholds judge: "
                                                            "p50, p95 or max"})
    max_missing: int = field(default=0, metadata={"help": "hits per path allowed to make no "
                                                          "sound"})
    allow_underrun: bool = field(default=False,
                                 metadata={"help": "report SPK under/refused/cap_drop/cap_under "
                                                   "during the run instead of failing on them"})
    console: bool = field(default=True,
                          metadata={"help": "read the console's SPK lines (capture ring, "
                                            "underruns); opening it resets the ESP once"})
    esp_echo: bool = field(default=True, metadata={"help": "time the ESP's USB MIDI echo of "
                                                           "each pad note (send2USB)"})
    echo_probes: int = field(default=20, metadata={"help": "SysEx echo round trips first "
                                                           "(USB transport baseline); 0 skips"})
    alsa_latency_ms: float = field(default=DEFAULT_LATENCY_MS,
                                   metadata={"help": "host buffer of the duplex stream; "
                                                     "accounted for, not measured"})
    exclusive_card: bool = field(default=True,
                                 metadata={"help": "pause the audio-nodes service for the run "
                                                   "(bench holder only); the duplex stream "
                                                   "needs the card itself"})
    enum_timeout: float = field(default=20.0, metadata={"help": "seconds to wait for a USB "
                                                                "profile switch"})


@dataclass
class PadHit:
    k: int
    pad: int
    note: int
    t_send: float
    send_ms: float


@dataclass
class HostHit:
    k: int
    frame: int
    t_sched: float


# --------------------------------------------------------------------------- seams


def _now() -> float:
    """The clock every send is stamped with (the duplex log's, too)."""
    return time.monotonic()


def _sleep_until(ctx: Context, target: float) -> None:
    while True:
        remaining = target - _now()
        if remaining <= 0:
            ctx.check_cancelled()
            return
        if ctx.cancelled.wait(remaining):
            ctx.check_cancelled()


def make_backend(p: Params) -> Any:
    return AlsaDuplexBackend(latency_ms=p.alsa_latency_ms)


def _math() -> Any:
    try:
        from crosspad_hil import latency_math
    except ImportError as exc:
        raise HilError(NOT_SUPPORTED, f"the latency analysis needs numpy: {exc}",
                       hint="pip install 'crosspad-hil[audio]'") from exc
    return latency_math


def _read_wav(path: str) -> tuple[Any, int]:
    from crosspad_hil.analyze import read_wav_ch

    x, sr, _ch = read_wav_ch(path, "auto")
    return x, sr


# --------------------------------------------------------------------------- the console


_SPK_RE = re.compile(r"SPK #(\d+):(.*)")


def parse_spk_line(line: str) -> dict[str, int] | None:
    """``SPK #n: len= taken= ring= under= refused= cap_ring= cap_drop= ...`` as ints."""
    m = _SPK_RE.search(line)
    if not m:
        return None
    row: dict[str, int] = {"n": int(m.group(1))}
    for tok in m.group(2).split():
        k, sep, v = tok.partition("=")
        if sep:
            try:
                row[k] = int(v)
            except ValueError:
                pass
    return row


class SpkWatch:
    """The ``SPK #`` lines of this session, each stamped when the host saw it.

    ``_t`` is the poll that read a line and ``_t_prev`` the one before it: a
    line read at ``_t`` was printed after ``_t_prev``, so what it reports
    happened after ``_t_prev - SPK_WORKER_S``.
    """

    def __init__(self, con: Any | None) -> None:
        self._con = con
        self.rows: list[dict[str, Any]] = []
        self._next = (con.seq + 1) if con is not None else 0
        self._last = _now()

    def poll(self) -> None:
        if self._con is None:
            return
        res = self._con.read(since_seq=self._next, match="SPK #", limit=10000)
        t = _now()
        for _seq, line in res.lines:
            row = parse_spk_line(line)
            if row is not None:
                self.rows.append({**row, "_t": t, "_t_prev": self._last})
        self._next = res.next_seq
        self._last = t


def spk_summary(rows: list[dict[str, Any]], rate: int) -> dict[str, Any]:
    """Counters over the session (deltas from its first line) and when one first moved."""
    if not rows:
        return {"lines": 0}
    first, last = rows[0], rows[-1]
    delta = {k: max(0, int(last.get(k, 0)) - int(first.get(k, 0)))
             for k in GLITCH_KEYS if k in last}
    base = sum(int(first.get(k, 0)) for k in GLITCH_KEYS)
    event = next((r for r in rows[1:]
                  if sum(int(r.get(k, 0)) for k in GLITCH_KEYS) > base), None)
    event_t = event["_t"] if event is not None else None
    caps = [r["cap_ring"] for r in rows if "cap_ring" in r]
    rings = [r["ring"] for r in rows if "ring" in r]
    out: dict[str, Any] = {"lines": len(rows), "delta": delta,
                           "totals": {k: last.get(k) for k in GLITCH_KEYS if k in last},
                           "event_seen_at": event_t}
    if event is not None:
        out["event_guard_s"] = event_guard(event)
    if caps:
        out["cap_ring_frames"] = {"min": min(caps), "max": max(caps),
                                  "mean": round(sum(caps) / len(caps), 1)}
    if rings:
        out["ring_frames"] = {"min": min(rings), "max": max(rings),
                              "mean": round(sum(rings) / len(rings), 1)}
    out["rate"] = rate
    return out


def event_guard(row: dict[str, Any]) -> float:
    """How long before ``row["_t"]`` (when it was seen) its event can have happened.

    The line was printed after the previous poll (else that poll would have
    read it) and at most one worker pass after the event. Polls are ~a gap
    apart during the hits but over a second apart across the pause between
    the paths, so a fixed guard lets a hit after the event count as before it.
    """
    prev = row.get("_t_prev")
    if prev is None:
        return SPK_GUARD_S
    span = float(row["_t"]) - float(prev) + SPK_WORKER_S + SPK_SEEN_MARGIN_S
    return round(max(SPK_GUARD_S, span), 3)


def capture_leg(rows: list[dict[str, Any]], t0: float, t1: float,
                rate: int) -> dict[str, Any] | None:
    """The capture ring's mean level over a phase, in ms -- what gets subtracted.

    The lines of the phase (seen between its first hit and a guard after its
    last); a phase that printed none borrows every line of the session.
    """
    inside = [r for r in rows if t0 <= r["_t"] <= t1 + SPK_GUARD_S and "cap_ring" in r]
    used = inside or [r for r in rows if "cap_ring" in r]
    if not used:
        return None
    frames = [int(r["cap_ring"]) for r in used]
    mean = sum(frames) / len(frames)
    spread = (max(frames) - min(frames)) / 2.0
    return {"ms": round(mean / rate * 1000.0, 3), "frames_mean": round(mean, 1),
            "frames_min": min(frames), "frames_max": max(frames), "lines": len(frames),
            "from_phase": bool(inside),
            # A line is one snapshot of a level that swings by a mic interval
            # between reads; the mean of a few is good to about the half swing.
            "u_ms": round(max(spread, 0.0) / rate * 1000.0 / math.sqrt(len(frames)), 3)}


# --------------------------------------------------------------------------- the ESP echo


class NoteEchoes:
    """The ESP's USB MIDI note-ons (send2USB echoes of the pad notes), stamped on arrival."""

    def __init__(self, io: Any) -> None:
        self._io = io
        self.notes: list[tuple[float, list[int]]] = []
        self._halt = threading.Event()
        self._thread = threading.Thread(target=self._run, name="note-echoes", daemon=True)

    def start(self) -> None:
        self._thread.start()

    def stop(self) -> None:
        self._halt.set()
        self._thread.join(2.0)

    def _run(self) -> None:
        while not self._halt.is_set():
            try:
                msg = self._io.receive_note(0.05)
            except HilError:
                return
            if msg:
                self.notes.append((_now(), list(msg)))


def match_echoes(hits: list[PadHit], notes: list[tuple[float, list[int]]]) -> list[float | None]:
    """Per hit, ms from its send to the first note-on of its note that came back."""
    ons = [(t, m[1]) for t, m in notes
           if len(m) >= 3 and (m[0] & 0xF0) == 0x90 and m[2] > 0]
    out: list[float | None] = []
    for h in hits:
        hit = next((t for t, note in ons
                    if note == h.note and h.t_send <= t <= h.t_send + ECHO_WINDOW_S), None)
        out.append(None if hit is None else round((hit - h.t_send) * 1000.0, 3))
    return out


# --------------------------------------------------------------------------- CDC helpers


_DESK_RE = re.compile(r"(\d+)=([sfe]):(.*?)(?=\s+\d+=[sfe]:|$)")


def parse_desks(line: str) -> dict[str, int]:
    """``MIXPRE: ok=1 undo=0 dir=... 1=f:STREAM 2=f:BEATS ...`` -> {name: slot}."""
    out: dict[str, int] = {}
    for slot, kind, name in _DESK_RE.findall(line):
        if kind != "e" and name.strip():
            out.setdefault(name.strip(), int(slot))
    return out


def _kv(line: str) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for tok in line.split(":", 1)[-1].split():
        k, sep, v = tok.partition("=")
        if sep:
            try:
                out[k] = int(v)
            except ValueError:
                out[k] = v
    return out


def _smpl_last(line: str) -> dict[str, Any]:
    m = re.search(r"lat_us last=(\d+) avg=(\d+) max=(\d+) n=(\d+)", line)
    return ({"lat_us_last": int(m.group(1)), "lat_us_avg": int(m.group(2)),
             "lat_us_max": int(m.group(3)), "lat_n": int(m.group(4))} if m else {})


def _env(summary: str, data: dict[str, Any], artifacts: list[Artifact]) -> Report:
    return Report(passed=False, summary=summary, data=data, artifacts=artifacts, exit_code=2)


def _validate(p: Params) -> None:
    bad = [x for x in p.paths if x not in PATHS]
    if bad or not p.paths:
        raise HilError(BAD_ARGS, f"--paths: {', '.join(bad) or 'empty'}",
                       hint="one or both of: " + ", ".join(PATHS))
    if p.hits < 1:
        raise HilError(BAD_ARGS, f"--hits must be >= 1, got {p.hits}")
    if not 0 < p.min_gap_ms <= p.max_gap_ms:
        raise HilError(BAD_ARGS, "need 0 < --min-gap-ms <= --max-gap-ms")
    if p.min_gap_ms < p.window_ms + fixtures.CLICK_SECONDS * 1000.0 + NEXT_GUARD_MS:
        raise HilError(BAD_ARGS, f"--min-gap-ms {p.min_gap_ms:.0f} leaves no room for a "
                                 f"{p.window_ms:.0f} ms window plus the click",
                       hint="raise --min-gap-ms or lower --window-ms")
    if p.gate_ms >= p.min_gap_ms:
        raise HilError(BAD_ARGS, "--gate-ms must be shorter than --min-gap-ms")
    if p.gate_stat not in GATE_STATS:
        raise HilError(BAD_ARGS, f"--gate-stat {p.gate_stat!r}", hint="p50, p95 or max")
    if p.preset not in PRESETS:
        raise HilError(BAD_ARGS, f"--preset {p.preset!r}", hint=", ".join(sorted(PRESETS)))
    if not p.pads or any(not 0 <= x < fixtures.SINE_PAD_COUNT for x in p.pads):
        raise HilError(BAD_ARGS, f"--pads must be 0..15, got {p.pads}")
    if not 0 <= p.note_base or p.note_base + max(p.pads) > 127:
        raise HilError(BAD_ARGS, f"--note-base {p.note_base} puts a pad off the MIDI range")
    if not 0 <= p.channel <= 15 or not 1 <= p.vel <= 127:
        raise HilError(BAD_ARGS, "--channel 0..15, --vel 1..127")
    if not 0.0 < p.host_level <= 1.0:
        raise HilError(BAD_ARGS, f"--host-level must be in (0, 1], got {p.host_level}")


# --------------------------------------------------------------------------- scenario


class LatencyScenario:
    name = "latency"
    Params = Params
    description = ("pad (note on the STM bridge) and host (UAC OUT) -> jack latency on the "
                   "board's own DAC->ADC loop: median/p95/max, capture ring subtracted")

    def run(self, ctx: Context, params: Params, progress: Progress) -> Report:
        p = params
        _validate(p)
        seed = p.seed if p.seed is not None else random.SystemRandom().randrange(2 ** 31)
        rng = random.Random(seed)
        gaps = {path: [rng.uniform(p.min_gap_ms, p.max_gap_ms) / 1000.0
                       for _ in range(p.hits)] for path in PATHS}
        data: dict[str, Any] = {
            "config": {"paths": list(p.paths), "hits": p.hits, "seed": seed,
                       "gap_ms": [p.min_gap_ms, p.max_gap_ms], "pads": list(p.pads),
                       "vel": p.vel, "note_base": p.note_base, "channel": p.channel,
                       "kit": p.kit, "preset": p.preset, "mixer_desk": p.mixer_desk,
                       "window_ms": p.window_ms, "gate_stat": p.gate_stat,
                       "host_level": p.host_level, "in2_gain": p.in2_gain,
                       "max_pad_ms": p.max_pad_ms, "max_host_ms": p.max_host_ms},
            "prologue": {}, "firmware": {}, "stm": {}, "capture": None, "clock": None,
            "spk": None, "echo_rtt": None, "paths": {}, "device_counters": {},
            "left": {}, "failures": [],
        }
        ctx.partial = data
        artifacts: list[Artifact] = []
        state: dict[str, Any] = {}
        try:
            try:
                _math()
            except HilError as e:
                return _env(f"latency: {e.message}", data, artifacts)
            if "pad" in p.paths and ctx.device.ports.stm_midi is None:
                return _env(f"latency: {ctx.device.id} has no STM bridge MIDI port -- the pad "
                            "notes have nothing to travel on", data, artifacts)
            con = None
            if p.console:
                progress(0, 5, "opening the console (resets the ESP)")
                con = ctx.open_console(reset=False)
                artifacts.append(Artifact(path=str(ctx.workdir / "console.log"),
                                          mime="text/plain", role="console"))
            progress(1, 5, "prologue: kit, app, desk")
            try:
                self._prologue(ctx, p, data, state, con)
            except HilError as e:
                if e.code in SETUP_CODES:
                    return _env(f"latency: prologue -- {e.code}: {e.message}"
                                + (f" ({e.hint})" if e.hint else ""), data, artifacts)
                raise
            measured = self._measure(ctx, p, data, artifacts, state, con, gaps, progress)
            if isinstance(measured, Report):
                return measured
            self._finish(ctx, p, data, state)
            state["finished"] = True
            progress(5, 5, "analysing")
            return self._analyse(ctx, p, data, artifacts, measured)
        finally:
            if not state.get("finished"):
                self._finish(ctx, p, data, state)
            ctx.close_all()

    # -- before ------------------------------------------------------------------
    def _prologue(self, ctx: Context, p: Params, data: dict[str, Any],
                  state: dict[str, Any], con: Any) -> None:
        if ctx.device.usb_mode is UsbMode.AUDIO or ctx.device.ports.cdc is None:
            ctx.log("device is in the USB-audio profile -- the default one first (CDC)")
            ctx.device = set_mode(ctx.device, UsbMode.DEFAULT, wait=True,
                                  timeout_s=p.enum_timeout)
        link = ctx.open_cdc()
        try:
            for key, fn in (("app_versions", verbs.app_versions),
                            ("stm_fw", verbs.stm_fw_info)):
                try:
                    data["firmware"][key] = fn(link)
                except HilError as e:
                    data["firmware"][key] = f"{e.code}: {e.message}"
            try:
                db = verbs.stm_reg(link, REG_PRESS_DEBOUNCE, 1)["bytes"][0]
                data["stm"]["press_debounce_ms"] = int(db)
            except HilError as e:
                data["stm"]["press_debounce_ms"] = None
                data["stm"]["note"] = f"STM_REG 33C: {e.code}: {e.message}"
            found = {"kit": None, "app": None}
            try:
                ks = verbs.kit_status(link)
                found["kit"] = ks["current"]
                state["kit_name"] = data["prologue"]["found_kit_name"] = ks.get("name")
            except HilError as e:
                data["prologue"]["kit_status"] = f"{e.code}: {e.message}"
            found["app"] = verbs.app_list(link).get("running")
            state["found"] = data["prologue"]["found"] = found
            if "pad" in p.paths:
                builder = (fixtures.write_click_kits
                           if p.kit.lower() == fixtures.CLICK_NAME.lower() else _no_fixture)
                link, provisioned = _audio.ensure_kits(
                    ctx, link, [p.kit], builder, console=con, provision=p.provision,
                    missing_hint_suffix=" -- HIL CLICK is built by crosspad_hil.fixtures")
                data["prologue"]["provisioned"] = provisioned
                kid, _name, current = _audio.resolve_kit(link, kit_name=p.kit)
                state["kit_changed"] = kid != current
                _audio.load_kit_waiting(ctx, link, kid)
                data["prologue"]["kit"] = {"id": kid, "name": p.kit}
                state["app_changed"] = found["app"] != SAMPLER_APP
                _audio.start_app(ctx, link, SAMPLER_APP)
                data["prologue"]["app"] = SAMPLER_APP
                try:
                    verbs.pad_stats(link, reset=True)
                    state["pad_stats_reset"] = True
                except HilError as e:
                    data["prologue"]["pad_stats"] = f"{e.code}: {e.message}"
            if p.mixer_desk:
                data["prologue"]["desk"] = self._load_desk(ctx, link, p.mixer_desk, state)
            if p.free_loop and PRESETS[p.preset][1] == _audio.LOOP_ADC_INPUT:
                # A pair that hears Line 3/4 leaves the loop undriven; the pair
                # first, so its switch cannot undo the IN2 pin below.
                state["pair_previous"] = _audio.free_jack_loop(ctx, link, data)
            if p.in2_gain is not None and PRESETS[p.preset][0] == _audio.JACK_CODEC:
                # INGAIN is CDC only, so here, before the audio profile; the journal
                # holds the old value until _finish has put it back.
                state["in2_previous"] = _audio.set_in2(ctx, link, int(p.in2_gain), data)
        finally:
            link.close()

    def _load_desk(self, ctx: Context, link: Any, name: str, state: dict[str, Any]) -> Any:
        try:
            listing = link.transact("MIX_PRESET", "MIXPRE:").line
        except HilError as e:
            ctx.log(f"MIX_PRESET: {e.code}: {e.message}; the desk is left as it is")
            return f"not available: {e.code}"
        desks = parse_desks(listing)
        slot = next((s for n, s in desks.items() if n.lower() == name.lower()), None)
        if slot is None:
            ctx.log(f"no Mixer desk named {name!r} (have {sorted(desks)}); left as it is")
            return f"no desk named {name!r}"
        reply = link.transact(f"MIX_PRESET LOAD {slot}", "MIXPRE:").line
        if not reply.startswith("MIXPRE:") or _kv(reply).get("ok") != 1:
            return f"LOAD {slot} refused: {reply}"
        state["desk_loaded"] = True
        ctx.log(f"Mixer desk {name} (slot {slot}) for the run; UNDO afterwards")
        return {"name": name, "slot": slot}

    # -- the run -----------------------------------------------------------------
    def _measure(self, ctx: Context, p: Params, data: dict[str, Any],
                 artifacts: list[Artifact], state: dict[str, Any], con: Any,
                 gaps: dict[str, list[float]], progress: Progress) -> Any:
        span = LEAD_S + TAIL_S + sum(sum(gaps[x]) for x in p.paths) + 30.0
        with card_hold.for_run(ctx, data, enabled=p.exclusive_card, expected_s=span) as hold:
            if hold.get("refused") and card_hold.service_active():
                return _env("latency: the CrossPad card is held by the audio-nodes service and "
                            f"this run may not pause it ({hold['refused']}) -- the duplex "
                            "stream needs the card itself; claim the bench lease "
                            "(crosspad-hil bench claim) and run again", data, artifacts)
            progress(2, 5, "USB-audio profile, duplex stream on the card")
            wav = ctx.workdir / "capture.wav"
            backend = make_backend(p)
            try:
                session = CaptureSession(ctx.device, seconds=None, preset=p.preset, out=wav,
                                         backend=backend, keep_mode=False, log=ctx.log,
                                         logdir=ctx.workdir)
                info = session.start()
            except HilError as e:
                return _env(f"latency: could not start the duplex stream -- {e.code}: "
                            f"{e.message}" + (f" ({e.hint})" if e.hint else ""),
                            data, artifacts)
            ctx.opened.append(session)
            ctx.device = session.device
            rec = backend.recording
            data["capture"] = {**info, "stream": dict(getattr(rec, "info", {}))}
            artifacts.append(Artifact(path=str(wav), mime="audio/wav", role="capture"))
            watch = SpkWatch(con)
            run: dict[str, Any] = {"rec": rec, "watch": watch, "pad": [], "host": [],
                                   "phases": {}, "echoes": None}
            try:
                _sleep_until(ctx, _now() + LEAD_S)
                watch.poll()
                if "pad" in p.paths:
                    self._pad_path(ctx, p, data, gaps["pad"], run, progress)
                if "host" in p.paths:
                    _sleep_until(ctx, _now() + PHASE_PAUSE_S)
                    progress(4, 5, f"host path: {p.hits} bursts into UAC OUT")
                    t0 = _now()
                    run["host"] = self._host_hits(ctx, p, rec, watch, gaps["host"])
                    run["phases"]["host"] = (t0, _now())
                _sleep_until(ctx, _now() + TAIL_S + p.window_ms / 1000.0)
                watch.poll()
            finally:
                try:
                    result = session.stop()
                except HilError as e:
                    result = None
                    data["failures"].append(f"capture stop: {e.code}: {e.message}")
                ctx.device = session.device
            if result is None:
                return _env("latency: the capture did not finish", data, artifacts)
            data["capture"]["result"] = result.to_dict()
            run["result"] = result
            return run

    def _pad_path(self, ctx: Context, p: Params, data: dict[str, Any], gaps: list[float],
                  run: dict[str, Any], progress: Progress) -> None:
        stm = MidiIO(ctx.device, MidiRole.STM, listen=False)
        stm.open()
        esp = None
        echoes = None
        try:
            if (p.echo_probes or p.esp_echo) and ctx.device.ports.esp_midi is not None:
                esp = MidiIO(ctx.device, MidiRole.ESP)
                esp.open()
            if esp is not None and p.echo_probes:
                progress(3, 5, f"{p.echo_probes} SysEx echo round trips")
                data["echo_rtt"] = echo_rtt(stm, esp, p.echo_probes, ctx)
            if esp is not None and p.esp_echo:
                echoes = NoteEchoes(esp)
                echoes.start()
            progress(3, 5, f"pad path: {p.hits} notes on the STM bridge")
            t0 = _now()
            run["pad"] = self._pad_hits(ctx, p, stm, run["watch"], gaps)
            run["phases"]["pad"] = (t0, _now())
            _sleep_until(ctx, _now() + p.window_ms / 1000.0)
        finally:
            if echoes is not None:
                echoes.stop()
                run["echoes"] = echoes.notes
            # The ESP port goes away with the profile switch back; an rtmidi
            # input still open across it hangs the process (midi.MidiIO).
            if esp is not None:
                esp.close()
            stm.close()

    def _pad_hits(self, ctx: Context, p: Params, stm: Any, watch: SpkWatch,
                  gaps: list[float]) -> list[PadHit]:
        hits: list[PadHit] = []
        t_next = _now() + gaps[0]
        for k in range(p.hits):
            _sleep_until(ctx, t_next)
            pad = p.pads[k % len(p.pads)]
            note = p.note_base + pad
            t0 = _now()
            stm.send_note(True, note, p.vel, p.channel)
            t1 = _now()
            hits.append(PadHit(k=k, pad=pad, note=note, t_send=t0,
                               send_ms=round((t1 - t0) * 1000.0, 3)))
            _sleep_until(ctx, t0 + p.gate_ms / 1000.0)
            stm.send_note(False, note, 0, p.channel)
            watch.poll()
            t_next = t0 + (gaps[k + 1] if k + 1 < len(gaps) else 0.0)
        return hits

    def _host_hits(self, ctx: Context, p: Params, rec: Any, watch: SpkWatch,
                   gaps: list[float]) -> list[HostHit]:
        burst = [int(round(v * INT16)) for v in fixtures.click_samples(
            HOST_BURST_SEED, rate=int(rec.sample_rate), peak=p.host_level)]
        lead = int(rec.schedule_lead_frames)
        hits: list[HostHit] = []
        t_next = _now() + gaps[0]
        for k in range(p.hits):
            _sleep_until(ctx, t_next)
            t0 = _now()
            frame = int(rec.out_position()) + lead
            rec.schedule(frame, burst)
            hits.append(HostHit(k=k, frame=frame, t_sched=t0))
            watch.poll()
            t_next = t0 + (gaps[k + 1] if k + 1 < len(gaps) else 0.0)
        return hits

    # -- after -------------------------------------------------------------------
    def _finish(self, ctx: Context, p: Params, data: dict[str, Any],
                state: dict[str, Any]) -> None:
        """Counters, then put back what the prologue changed. Never raises."""
        if "found" not in state:
            return
        left = data["left"]
        try:
            if ctx.device.usb_mode is not UsbMode.DEFAULT or ctx.device.ports.cdc is None:
                ctx.device = set_mode(ctx.device, UsbMode.DEFAULT, wait=True,
                                      timeout_s=p.enum_timeout)
            link = ctx.open_cdc()
        except HilError as e:
            left["error"] = f"{e.code}: {e.message}"
            ctx.log(f"could not get back to CDC to put the board back: {e.message}")
            return
        try:
            if state.get("pad_stats_reset"):
                for key, cmd, prefix in (("pad_stats", "PAD_STATS", "PADSTATS:"),
                                         ("smpl_last", "SMPL_LAST", "SMPLLAST:")):
                    try:
                        line = link.transact(cmd, prefix).line
                        data["device_counters"][key] = (_kv(line) if key == "pad_stats"
                                                        else _smpl_last(line))
                    except HilError as e:
                        data["device_counters"][key] = f"{e.code}: {e.message}"
            found = state["found"]
            steps = (
                ("desk", state.get("desk_loaded"), lambda: _desk_undo(link)),
                ("kit", state.get("kit_changed") and found.get("kit") is not None
                 and found["kit"] >= 0,
                 lambda: verbs.kit_load(link, _home_kit(link, int(found["kit"]),
                                                        state.get("kit_name")), wait_s=30.0)),
                ("app", state.get("app_changed"),
                 lambda: (verbs.app_stop(link) if found.get("app") is None
                          else _audio.start_app(ctx, link, str(found["app"])))),
            )
            for name, needed, fn in steps:
                if not needed:
                    continue
                try:
                    fn()
                    left[name] = "restored"
                except HilError as e:
                    left[name] = f"{e.code}: {e.message}"
                    ctx.log(f"{name} NOT restored: {e.message}")
        finally:
            link.close()
        if state.get("in2_previous") is not None or state.get("pair_previous") is not None:
            _audio.restore_in2(ctx, state.get("in2_previous"), data,
                               previous_pair=state.get("pair_previous"),
                               enum_timeout=p.enum_timeout)
            if state.get("in2_previous") is not None:
                left["in2"] = "restored" if data.get("ingain_restored") else "NOT restored"
            if state.get("pair_previous") is not None:
                left["in_pair"] = "restored" if data.get("in_pair_restored") else "NOT restored"

    # -- verdict -----------------------------------------------------------------
    def _analyse(self, ctx: Context, p: Params, data: dict[str, Any],
                 artifacts: list[Artifact], run: dict[str, Any]) -> Report:
        lm = _math()
        rec, result, watch = run["rec"], run["result"], run["watch"]
        health = dict(getattr(result, "health", None) or {})
        why = _audio.host_loss(data, health)
        if not why and health.get("out_xruns"):
            why = f"{health['out_xruns']} host playback xrun(s): the board starved by the host"
            data["capture_untrusted"] = why
        if not why and health.get("error"):
            why = f"the duplex stream stopped: {health['error']}"
            data["capture_untrusted"] = why
        if why:
            return _audio.not_measured("latency", why, data, artifacts)

        rate = int(rec.sample_rate)
        try:
            cap_pts = rec.capture_points()
            cap = lm.lower_envelope_fit([f for f, _ in cap_pts], [t for _, t in cap_pts],
                                        nominal_rate=rate)
            play = None
            if run["host"]:
                pts = rec.playback_points()
                play = lm.lower_envelope_fit([f for f, _ in pts], [t for _, t in pts],
                                             nominal_rate=rate)
        except ValueError as e:
            return _env(f"latency: the stream's clock log cannot be fitted -- {e}",
                        data, artifacts)
        data["clock"] = {"capture": cap.to_dict(),
                         "playback": play.to_dict() if play is not None else None}
        x, sr = _read_wav(result.wav)
        if sr != rate:
            return _env(f"latency: the take is {sr} Hz, the stream {rate} Hz", data, artifacts)
        noise = lm.noise_floor(x, sr)
        data["noise_dbfs"] = round(20.0 * math.log10(noise), 1)
        spk = spk_summary(watch.rows, rate) if watch.rows else (
            {"lines": 0} if p.console else None)
        data["spk"] = spk
        if spk and spk.get("event_seen_at") is not None and rec.started_at is not None:
            spk["event_seen_after_start_s"] = round(spk["event_seen_at"] - rec.started_at, 3)
        if spk is not None and not spk["lines"]:
            data.setdefault("notes", []).append(
                "no SPK # line on the console: the capture ring is not subtracted and "
                "underruns are not watched (firmware log level, or no UAC OUT stream)")

        rows: list[dict[str, Any]] = []
        envs: list[str] = []
        fails: list[str] = []
        for path in p.paths:
            hits = run[path]
            if not hits:
                continue
            if path == "pad":
                refs = [(h.k, h.t_send) for h in hits]
            else:
                refs = [(h.k, float(play.t(h.frame))) for h in hits]  # type: ignore[union-attr]
            t0, t1 = run["phases"][path]
            leg = capture_leg(watch.rows, t0, t1, rate) if watch.rows else None
            body, hit_rows = self._score_path(lm, path, x, sr, noise, cap, refs, leg, p)
            if path == "pad":
                self._pad_extras(lm, p, data, body, hits, run.get("echoes"), hit_rows)
            self._segments(lm, body, hit_rows, spk)
            data["paths"][path] = body
            rows.extend(hit_rows)
            env, fail = self._judge(path, body, p, data)
            envs.extend(env)
            fails.extend(fail)

        hits_csv = ctx.workdir / "hits.csv"
        _write_csv(hits_csv, rows)
        artifacts.append(Artifact(path=str(hits_csv), mime="text/csv", role="hits"))
        clock_csv = ctx.workdir / "clock.csv"
        _write_clock(clock_csv, rec)
        artifacts.append(Artifact(path=str(clock_csv), mime="text/csv", role="clock"))

        glitch = self._glitches(spk, p)
        if glitch:
            (fails if not p.allow_underrun else data.setdefault("notes", [])).append(glitch)
        data["failures"].extend(fails)
        out = ctx.workdir / "latency.json"
        out.write_text(json.dumps(data, indent=2, default=str) + "\n", encoding="utf-8")
        artifacts.append(Artifact(path=str(out), mime="application/json", role="analysis"))

        head = self._headline(data)
        left = _not_put_back(data["left"])
        if left:
            data.setdefault("notes", []).append(left)
        if envs:
            return _env(f"latency: not measured -- {'; '.join(envs)}" + (f" ({head})"
                                                                         if head else "")
                        + (f" -- {left}" if left else ""),
                        data, artifacts)
        passed = not fails
        summary = f"latency: {'PASS' if passed else 'FAIL'} -- {head}"
        if fails:
            summary += " -- " + "; ".join(fails)
        if left:
            summary += f" -- {left}"
        ctx.log(summary)
        return Report(passed=passed, summary=summary, data=data, artifacts=artifacts,
                      exit_code=0 if passed else 1)

    def _score_path(self, lm: Any, path: str, x: Any, sr: int, noise: float, cap: Any,
                    refs: list[tuple[int, float]], leg: dict[str, Any] | None,
                    p: Params) -> tuple[dict[str, Any], list[dict[str, Any]]]:
        expected = [float(cap.frame(t)) for _k, t in refs]
        pre = PRE_TOL_MS / 1000.0 * sr
        guard = NEXT_GUARD_MS / 1000.0 * sr
        win = p.window_ms / 1000.0 * sr
        hit_rows: list[dict[str, Any]] = []
        lists: dict[str, list[int]] = {"missing": [], "masked": [], "double": [], "early": []}
        good_t: list[float] = []
        raw: list[float] = []
        leg_ms = leg["ms"] if leg else None
        for idx, ((k, t_ref), e) in enumerate(zip(refs, expected, strict=True)):
            stop = e + win
            if idx + 1 < len(expected):
                stop = min(stop, expected[idx + 1] - guard)
            onset = lm.find_onset(x, e - pre, stop, sr=sr, noise=noise, pre_ms=RUNUP_MS)
            lat = None
            status = onset.status
            if onset.index is not None:
                lat = (float(cap.t(onset.index)) - t_ref) * 1000.0
                if lat < 0 and status == "ok":
                    status = "early"
            if status in lists:
                lists[status].append(k)
            if status == "ok" and lat is not None:
                raw.append(lat)
                good_t.append(t_ref)
            hit_rows.append({"path": path, "k": k, "t_ref": round(t_ref, 6),
                             "expected_frame": round(e, 1),
                             "onset_frame": None if onset.index is None
                             else round(onset.index, 2),
                             "status": status,
                             "uac_in_ms": None if lat is None else round(lat, 3),
                             "jack_ms": None if lat is None or leg_ms is None
                             else round(lat - leg_ms, 3),
                             "peak_dbfs": round(20 * math.log10(max(onset.peak, 1e-9)), 1)})
        uac = lm.describe(raw)
        jack = lm.describe([v - leg_ms for v in raw]) if leg_ms is not None and raw else None
        u_sys: dict[str, Any] = {"not_subtracted_ms": list(UNREPORTED_CAPTURE_MS),
                                 "clock_map_ms": 0.5}
        if leg:
            u_sys["capture_ring_ms"] = leg["u_ms"]
        if path == "pad":
            u_sys["usb_out_ms"] = [USB_OUT_MS[0], USB_OUT_MS[2]]
        body: dict[str, Any] = {
            "metric": f"{path}_jack_ms" if jack else f"{path}_uac_in_ms",
            "hits": len(refs), "measured": len(raw),
            **lists,
            "uac_in_ms": uac,
            "capture_leg": leg,
            "jack_ms": jack,
            "headline": "jack_ms" if jack else "uac_in_ms",
            "includes": PAD_INCLUDES if path == "pad" else HOST_INCLUDES,
            "excludes": PAD_EXCLUDES if path == "pad" else HOST_EXCLUDES,
            "not_subtracted": NOT_SUBTRACTED,
            "u_sys_ms": u_sys,
            "_t": good_t, "_raw": raw,
        }
        return body, hit_rows

    def _pad_extras(self, lm: Any, p: Params, data: dict[str, Any], body: dict[str, Any],
                    hits: list[PadHit], notes: Any, rows: list[dict[str, Any]]) -> None:
        body["send_ms"] = lm.describe([h.send_ms for h in hits], ci=False)
        if notes is not None:
            echo = match_echoes(hits, notes)
            got = [v for v in echo if v is not None]
            body["esp_echo_ms"] = lm.describe(got) if got else None
            body["esp_echo_missing"] = sum(1 for v in echo if v is None)
            for row, v in zip(rows, echo, strict=True):
                row["esp_echo_ms"] = v
        db = data["stm"].get("press_debounce_ms")
        head = body.get(body["headline"]) or {}
        if db is not None and head.get("n"):
            base = float(head["p50"])
            lo = base - USB_OUT_MS[2] + STM_SCAN_MS[0] + db + WINDOW_OFFSET_MS[0]
            typ = base - USB_OUT_MS[1] + STM_SCAN_MS[1] + db + WINDOW_OFFSET_MS[1]
            hi = base - USB_OUT_MS[0] + STM_SCAN_MS[2] + db + WINDOW_OFFSET_MS[2]
            body["physical_est_ms"] = {
                "kind": "W (model, not measured)", "from": body["headline"] + ".p50",
                "press_debounce_ms": db, "p50": round(typ, 2),
                "range": [round(lo, 2), round(hi, 2)],
                "adds": "STM scan 0.25-1 ms + press window (db-1 .. db+0.75 ms), "
                        "minus the bridge's USB OUT (0-1 ms)"}

    def _segments(self, lm: Any, body: dict[str, Any], rows: list[dict[str, Any]],
                  spk: dict[str, Any] | None) -> None:
        times, raw = body.pop("_t"), body.pop("_raw")
        t_event = (spk or {}).get("event_seen_at")
        if t_event is None:
            return
        leg = body["capture_leg"]["ms"] if body.get("capture_leg") else 0.0
        guard = float((spk or {}).get("event_guard_s", SPK_GUARD_S))
        before, after, ambiguous = lm.split_at(times, [v - leg for v in raw], t_event, guard)
        seg: dict[str, Any] = {"event_seen_at": t_event, "guard_s": guard,
                               "before": lm.describe(before), "after": lm.describe(after),
                               "ambiguous": ambiguous}
        if before and after:
            seg["delta_p50_ms"] = round(seg["after"]["p50"] - seg["before"]["p50"], 3)
        body["segments"] = seg

    def _judge(self, path: str, body: dict[str, Any], p: Params,
               data: dict[str, Any]) -> tuple[list[str], list[str]]:
        envs: list[str] = []
        fails: list[str] = []
        n = body["hits"]
        if not body["measured"] and len(body["missing"]) == n:
            envs.append(f"{path}: none of {n} hits made a sound on the {p.preset} loop -- "
                        "check the Mixer desk (--mixer-desk BEATS), the capture preset and "
                        "the kit")
            return envs, fails
        if len(body["early"]) > max(1, EARLY_SHARE_ENV * n):
            envs.append(f"{path}: {len(body['early'])} onsets before their own send -- the "
                        "stream's clock map is wrong")
        if not body["measured"]:
            envs.append(f"{path}: no hit could be measured "
                        f"({len(body['masked'])} masked, {len(body['double'])} double)")
            return envs, fails
        if len(body["missing"]) > p.max_missing:
            why = f"{path}: {len(body['missing'])}/{n} hits never sounded (k={body['missing']})"
            ps = data["device_counters"].get("pad_stats")
            if path == "pad" and isinstance(ps, dict) and "stm" in ps:
                why += f" (PAD_STATS stm={ps['stm']} of {n})"
            fails.append(why)
        if body["double"]:
            fails.append(f"{path}: {len(body['double'])} hit(s) sounded twice "
                         f"(k={body['double']})")
        limit = p.max_pad_ms if path == "pad" else p.max_host_ms
        head = body[body["headline"]]
        if limit is not None and head.get("n") and head[p.gate_stat] > limit:
            fails.append(f"{path}: {body['headline'].replace('_ms', '')} {p.gate_stat} "
                         f"{head[p.gate_stat]:.2f} ms > {limit:.2f} ms")
        return envs, fails

    def _glitches(self, spk: dict[str, Any] | None, p: Params) -> str | None:
        if not spk or not spk.get("delta"):
            return None
        moved = {k: v for k, v in spk["delta"].items() if v}
        if not moved:
            return None
        return ("device glitch during the measurement: "
                + " ".join(f"{k}+{v}" for k, v in moved.items())
                + " (SPK #; latency before/after in paths.*.segments)")

    def _headline(self, data: dict[str, Any]) -> str:
        parts = []
        for path, body in data["paths"].items():
            h = body.get(body.get("headline", ""), None) or {}
            if not h.get("n"):
                parts.append(f"{path}: nothing measured")
                continue
            label = f"{path}->jack" if body["headline"] == "jack_ms" else f"{path}->UAC IN"
            parts.append(f"{label} p50 {h['p50']:.2f} p95 {h['p95']:.2f} max {h['max']:.2f} ms "
                         f"(n={h['n']}/{body['hits']})")
            leg = body.get("capture_leg")
            if leg:
                parts[-1] += f", capture ring {leg['ms']:.2f} ms subtracted"
        return "; ".join(parts)


def _desk_undo(link: Any) -> None:
    """``MIX_PRESET UNDO``, checked: the reply is a listing whether or not it undid.

    The undo snapshot lives in the ESP's RAM while the loaded desk is persisted,
    so a board that reset mid-run answers ``ok=0`` and keeps the run's desk.
    """
    reply = link.transact("MIX_PRESET UNDO", "MIXPRE:").line
    if not reply.startswith("MIXPRE:") or _kv(reply).get("ok") != 1:
        raise HilError(ENV, f"MIX_PRESET UNDO refused: {reply}",
                       hint="the run's desk stays (the board keeps it): load the old one "
                            "in the Mixer")


def _home_kit(link: Any, kit_id: int, name: str | None) -> int:
    """The id the kit found before the run has now.

    A kit id is a position in the card's scan, not a name: provisioning HIL CLICK
    pushes a folder and reboots, and on FAT a new folder can take a slot ahead of
    kits already there, so the id read before the push may now be another kit.
    """
    if not name:
        return kit_id
    kits = {int(k["id"]): str(k["name"]) for k in verbs.kit_list(link).get("kits", [])}
    if kits.get(kit_id, "").lower() == name.lower():
        return kit_id
    match = sorted(k for k, n in kits.items() if n.lower() == name.lower())
    if not match:
        raise HilError(ENV, f"the kit found before the run ({name!r}, id {kit_id}) is not "
                            "in KIT_LIST any more", kits=sorted(kits.values()))
    return match[0]


def _not_put_back(left: dict[str, Any]) -> str | None:
    bad = {k: v for k, v in left.items() if v != "restored"}
    if not bad:
        return None
    return "board NOT put back: " + "; ".join(f"{k} {v}" for k, v in bad.items())


def _no_fixture(root: Path) -> dict[str, Path]:
    return {}


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    keys: list[str] = []
    for r in rows:
        keys.extend(k for k in r if k not in keys)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=keys or ["path"])
        w.writeheader()
        w.writerows(rows)


def _write_clock(path: Path, rec: Any) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["side", "frame", "t_monotonic"])
        for side, pts in (("capture", rec.capture_points()), ("playback", rec.playback_points())):
            for frame, t in pts[:: max(1, len(pts) // 20000)]:
                w.writerow([side, frame, f"{t:.6f}"])


SCENARIO = register(LatencyScenario())
