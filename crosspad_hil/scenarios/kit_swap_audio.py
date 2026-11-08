"""Swap the kit while the pads sound, and let the recording say whether it was seamless.

`kit_churn` proves a swap does not crash and does not get stuck. It cannot hear
anything: `SMPL_PEAK` says "something rendered", not "pad 5 kept its tone all
the way through the swap". This scenario records the board through its own UAC2
endpoint while pads fire and kits swap under them, and scores the *audio*
against three requirements a live set actually has:

1. **Every hit is audible.** A pad struck during or after a swap must sound.
2. **No hit has a hole in it.** A tone that stops for a block or two mid-sample
   is the swap stealing render time or the loader freeing a buffer that was
   still being read; `max_gap_ms` is the number that separates a scheduling
   hiccup from an audible dropout (one 64-frame block at 44.1 kHz is 1.45 ms,
   so the default gate of 5 ms is ~3 blocks).
3. **A sounding pad finishes its sample before it is reloaded.** A pad whose
   tone changes frequency *mid-window* was reloaded while it was still playing.
   The verdict reads `switch_run_frames`, not the raw `switch_frames`: clean
   captures flip a single frame to the other candidate on a harmonic, and a pad
   cannot be reloaded for 20 ms and back again, so a switch has to hold for
   three frames (60 ms) to be one.
4. **A hit starts when it was struck.** A tone that arrives late has no hole in
   it — the voice waited for its first refill — so `late_start_ms` is reported
   and gated separately (`--max-late-ms`), and the leading quiet run is excluded
   from the dropout numbers. Measured on the bench: a pad hit at t=5.50 whose
   tone began at ~5.72. Folding that into `max_gap_ms` would have blamed the
   swap for a 220 ms dropout in a tone that then played perfectly.

The measurement rests on the fixture kits: `HIL SINE A` and `HIL SINE B` give
every pad a different pure sine, interleaved 111 Hz apart, so "which kit is pad
p currently loaded with" is a question about two named frequency bands rather
than about timbre. :func:`crosspad_hil.analyze.analyze_tone_tracks` is the
kernel; this module owns the stimulus, the swap schedule and the verdict.

The frequency formulas are duplicated here rather than imported from
`tools/make_sine_kits.py`: crosspad-hil does not depend on platform-idf. Change
one, change the other.

Deferral is expected, not penalised. A pad struck just before a swap keeps
playing its old kit's tone, and the *next* hit on that pad may still be on the
old kit if the loader honoured the sounding voice — that is the behaviour under
test. What must never happen is a pad going *back*: once a pad has been heard
on the new kit, an older kit on a later hit means the reload was undone or the
swap order was not respected. `kit_sequence_problems` is that rule, and it is a
pure function so it can be tested without a board.

The kit-load request goes over SysEx (`F0 7D 1D 0B <kit>`), not CDC: the
USB-audio profile has no CDC endpoint, and the recording cannot be paused to go
back and get one. `KIT_SWAP_LOG` is read afterwards, over CDC, as evidence for
the report — never as a gate, so a firmware that predates the verb still gets a
verdict from the audio.
"""
from __future__ import annotations

import bisect
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from crosspad_hil import card_hold, fixtures, verbs
from crosspad_hil.cdc import CdcLink
from crosspad_hil.console import Console
from crosspad_hil.devices import UsbMode
from crosspad_hil.errors import (
    BAD_ARGS,
    ENV,
    NO_CDC_IN_AUDIO_MODE,
    NOT_SUPPORTED,
    TIMEOUT,
    HilError,
)
from crosspad_hil.midi import sysex_kit_load
from crosspad_hil.scenarios import _audio, _deps
from crosspad_hil.scenarios.base import Artifact, Context, Progress, Report, register
from crosspad_hil.usbmode import set_mode

SAMPLER_APP = "Sampler"
PAD_COUNT = 16

# The fixture's tone length. The scoring window is cut out of it, and `dwell`
# has to exceed it or a swap lands inside a tone that is still being scored.
TONE_S = 2.0

STREAM_SETTLE_S = 0.5         # let the capture stream settle before the first hit
TAIL_S = 2.5                  # the last tone has to finish inside the recording
APP_START_SETTLE_S = 2.0
KIT_LOAD_TIMEOUT_S = 20.0
KIT_SETTLE_S = 2.0            # after the run, before asking KIT_STATUS what it settled on

# --- the fixture's two frequency ladders (tools/make_sine_kits.py) -----------
STEP_HZ = 222.0
BASE_A_HZ = 300.0
BASE_B_HZ = 411.0

SETUP_CODES: frozenset[str] = frozenset(
    {ENV, NOT_SUPPORTED, TIMEOUT, NO_CDC_IN_AUDIO_MODE}
)


def freq_a(pad: int) -> float:
    """Kit A, pad p: 300 + 222*p Hz. Duplicated from `tools/make_sine_kits.py`."""
    return BASE_A_HZ + STEP_HZ * pad


def freq_b(pad: int) -> float:
    """Kit B, pad p: 411 + 222*p Hz. Duplicated from `tools/make_sine_kits.py`."""
    return BASE_B_HZ + STEP_HZ * pad


def candidate_freqs(pads: list[int]) -> dict[int, list[float]]:
    """pad -> [kit A Hz, kit B Hz]; the index into that list is the "kit" the
    scoring reports, so 0 is always A and 1 is always B."""
    return {p: [freq_a(p), freq_b(p)] for p in pads}


@dataclass
class Params:
    kit_a: str = field(default="HIL SINE A",
                       metadata={"help": "first kit, by name (case-insensitive)"})
    kit_b: str = field(default="HIL SINE B",
                       metadata={"help": "second kit, by name (case-insensitive)"})
    swaps: int = field(default=3,
                       metadata={"help": "kit swaps to perform: A->B->A->B..."})
    hit_rate: float = field(default=4.0, metadata={"help": "pad hits per second"})
    dwell: float = field(
        default=6.0,
        metadata={"help": "seconds between swaps; must exceed the 2.0 s tone plus a load"})
    lead: float = field(default=3.0,
                        metadata={"help": "seconds of playing before the first swap"})
    pads: list[int] | None = field(
        default=None, metadata={"help": "pads to round-robin (comma-separated); default 0-15"})
    velocity: int = field(default=100, metadata={"help": "velocity of every injected hit"})
    gate_ms: int = field(default=60, metadata={"help": "how long each pad is held"})
    preset: str = field(
        default="headphone",
        metadata={"help": "capture path: headphone (unity DAC->ADC loop), speaker, mics, line-in"})
    volume: int | None = field(default=None, metadata={"help": "capture codec volume 0..100"})
    dropout_db: float = field(
        default=10.0,
        metadata={"help": "dB below a tone's median level that counts as a dropout frame"})
    max_gap_ms: float = field(
        default=5.0,
        metadata={"help": "longest dropout a hit may carry; a 64-frame block at 44.1 kHz "
                          "is 1.45 ms, so ~3 ms and up is a real hole"})
    max_late_ms: float = field(
        default=20.0,
        metadata={"help": "how long after the hit a tone may take to start; a voice that "
                          "waited for its first refill starts late, which is not a hole"})
    no_provision: bool = field(
        default=False,
        metadata={"help": "do not build and push the fixture kits when the card lacks "
                          "them; fail with 'no kit named' instead"})
    exclusive_card: bool = field(
        default=True,
        metadata={"help": "pause the audio-nodes service for the run, so the capture opens "
                          "the card directly instead of through PipeWire (which lost seconds)"})
    keep_audio_mode: bool = field(
        default=False, metadata={"help": "stay in the USB audio profile when done"})
    out: str | None = field(
        default=None, metadata={"help": "output WAV (default: kit_swap_audio.wav in the work dir)"})


# --------------------------------------------------------------------------- seams
# Same shape as pitched_kit's: one place a test can replace the audio machinery
# without a sound card, and one place that turns "the optional extra isn't
# installed" into an ENV (exit 2) rather than a plain FAIL.


def _as_env(exc: HilError) -> HilError:
    if exc.code != NOT_SUPPORTED:
        return exc
    return HilError(ENV, exc.message, hint=exc.hint, **exc.details)


def open_capture(device: Any, **kw: Any) -> Any:
    try:
        return _deps.module("capture").CaptureSession(device, **kw)
    except HilError as e:
        raise _as_env(e) from None


def open_pad_transport(device: Any, kind: str) -> Any:
    return _deps.module("stim").open_transport(device, kind)


def analyze_tone_tracks(*args: Any, **kw: Any) -> dict:
    try:
        return dict(_deps.module("analyze").analyze_tone_tracks(*args, **kw))
    except HilError as e:
        raise _as_env(e) from None


class ForgivingTransport:
    """Pad writes that count a refusal instead of raising it, plus the swap frames.

    Two threads share one MIDI out here — the stimulus pressing pads and the run
    thread asking for a kit — so every write goes through one lock. rtmidi's
    ALSA sender is not re-entrant, and a swap frame interleaved into a pad frame
    is a malformed SysEx that the device drops silently: the swap would simply
    never happen, and the recording would show a perfectly seamless run.

    A single lost message is worth losing one hit over, not the whole recording
    and the USB-profile restore that has to run in `finally` (pitched_kit's twin).
    """

    def __init__(self, inner: Any) -> None:
        self._inner = inner
        self._lock = threading.Lock()
        self.name = getattr(inner, "name", "?")
        self.failures = 0

    def press(self, pad: int, vel: int) -> None:
        with self._lock:
            try:
                self._inner.press(pad, vel)
            except Exception:  # noqa: BLE001 - a lost message is counted, not raised
                self.failures += 1

    def release(self, pad: int) -> None:
        with self._lock:
            try:
                self._inner.release(pad)
            except Exception:  # noqa: BLE001
                self.failures += 1

    def send_kit_load(self, kit_id: int) -> bool:
        with self._lock:
            try:
                self._inner._io.send_sysex(sysex_kit_load(kit_id))
                return True
            except Exception:  # noqa: BLE001
                self.failures += 1
                return False

    def close(self) -> None:
        self._inner.close()


# --------------------------------------------------------------------------- schedule


def build_swaps(swaps: int, lead: float, dwell: float) -> list[dict[str, Any]]:
    """The swap requests: `lead`, `lead+dwell`, … alternating B, A, B, ….

    Each entry carries `stage` (1-based: stage 0 is the kit the run started on),
    `kit_index` (0 = kit A, 1 = kit B — the same index `analyze_tone_tracks`
    reports) and `t`, seconds after the stimulus started. Pure, so the schedule
    and the sequence rule below can be tested without a board.
    """
    return [{"stage": i + 1, "kit_index": 1 if i % 2 == 0 else 0,
             "t": round(lead + i * dwell, 4)}
            for i in range(swaps)]


def stage_kits(swaps: list[dict[str, Any]]) -> list[int]:
    """Kit index per stage, stage 0 being the kit loaded before the run: [0, 1, 0, 1, …]."""
    return [0] + [int(s["kit_index"]) for s in swaps]


def kit_sequence_problems(hits: list[dict[str, Any]], swaps: list[dict[str, Any]]
                          ) -> list[str]:
    """Per pad, the measured kits must only ever move forward through the swap order.

    Each hit is assigned the *latest* stage, at or before the stage its timestamp
    falls in, whose kit matches what was measured. A hit on the old kit after a
    swap request therefore lands on the earlier stage and is perfectly legal —
    that is a deferred reload, the thing this scenario wants to see happen. Two
    things are not legal and each gets a line:

    * a kit that had not been requested yet at that time (no matching stage) —
      a pad cannot play a kit before it was asked for, so the timeline is wrong;
    * a stage lower than one already observed on that pad — the pad went back to
      an older kit, i.e. a reload was lost or applied out of order.

    Hits with `kit is None` (nothing audible) are skipped; they are already a
    FAIL under the audible rule, and guessing a stage for them would double-count.
    """
    times = [float(s["t"]) for s in swaps]
    kits = stage_kits(swaps)
    problems: list[str] = []
    by_pad: dict[int, list[dict[str, Any]]] = {}
    for h in hits:
        by_pad.setdefault(int(h["pad"]), []).append(h)
    for pad in sorted(by_pad):
        last_stage = -1
        for h in sorted(by_pad[pad], key=lambda r: float(r["t"])):
            if h.get("kit") is None:
                continue
            kit = int(h["kit"])
            reached = bisect.bisect_right(times, float(h["t"]))
            allowed = [s for s in range(reached + 1) if kits[s] == kit]
            if not allowed:
                problems.append(
                    f"pad {pad} played kit {'AB'[kit]} at t={float(h['t']):.2f}s, before it "
                    "was ever requested")
                continue
            stage = allowed[-1]
            if stage < last_stage:
                problems.append(
                    f"pad {pad} went back to kit {'AB'[kit]} (stage {stage}) at "
                    f"t={float(h['t']):.2f}s after it had reached stage {last_stage}")
            last_stage = max(last_stage, stage)
    return problems


class Player(threading.Thread):
    """Round-robin fingerdrumming that does not stop for the swap.

    One hit is a press, `gate_ms` of hold, and a release; the sample is a
    one-shot, so the tone runs its full 2 s whatever the gate does — which is
    what makes a fixed scoring window possible. Every press is stamped with
    `time.monotonic()`, because the scoring window of a hit is defined by when
    the *host* sent it, not by when the schedule said it would.
    """

    def __init__(self, transport: ForgivingTransport, hit_rate: float, pads: list[int],
                 velocity: int, gate_ms: int) -> None:
        super().__init__(name="kit_swap_audio_player", daemon=True)
        self._transport = transport
        self._period = 1.0 / max(0.1, hit_rate)
        self._pads = list(pads)
        self._velocity = velocity
        self._gate_s = max(0.0, gate_ms / 1000.0)
        self._halt = threading.Event()
        self._lock = threading.Lock()
        self._hits: list[dict[str, Any]] = []

    def run(self) -> None:
        i = 0
        next_t = time.monotonic()
        while not self._halt.is_set():
            delay = next_t - time.monotonic()
            if delay > 0 and self._halt.wait(delay):
                break
            pad = self._pads[i % len(self._pads)]
            i += 1
            self._transport.press(pad, self._velocity)
            stamp = time.monotonic()
            with self._lock:
                self._hits.append({"t": stamp, "pad": pad})
            if self._halt.wait(self._gate_s):
                break
            self._transport.release(pad)
            next_t += self._period

    def hits(self) -> list[dict[str, Any]]:
        with self._lock:
            return [dict(h) for h in self._hits]

    def stop(self) -> None:
        self._halt.set()


# --------------------------------------------------------------------------- verdict


def swap_verdict(p: Params, data: dict[str, Any]) -> list[str]:
    """The FAIL reasons for one scored run — pure, so a fabricated result set tests it."""
    problems: list[str] = list(data.get("failures", []))
    if data.get("fatal") or data.get("reboots"):
        lines = list(data.get("fatal") or [])[:6] or ["(reset with no panic line)"]
        problems.append("crash during the run:\n    " + "\n    ".join(lines))
    status = data.get("kit_status")
    want = data.get("last_kit_id")
    if status is None:
        problems.append("KIT_STATUS unavailable after the run")
    elif want is not None and (status.get("current") != want or status.get("loading")
                               or status.get("pending", -1) != -1):
        problems.append(
            f"settled on kit {status.get('current')} (loading={int(bool(status.get('loading')))} "
            f"pending={status.get('pending')}), not the last one requested ({want})")
    scored: list[dict[str, Any]] = data.get("scored", [])
    missing = [r for r in scored if r.get("missing")]
    if missing:
        problems.append(f"{len(missing)} hit(s) made no sound: "
                        + ", ".join(f"pad {r['pad']}@{float(r['t']):.2f}s" for r in missing[:8]))
    # `switch_run_frames`, never the raw `switch_frames`: a lone frame on the
    # other candidate is a harmonic, not a reload (see analyze_tone_tracks).
    switched = [r for r in scored if int(r.get("switch_run_frames", 0)) > 0]
    if switched:
        problems.append(
            f"{len(switched)} hit(s) changed kit mid-tone (reloaded while still sounding): "
            + ", ".join(f"pad {r['pad']}@{float(r['t']):.2f}s" for r in switched[:8]))
    late = [r for r in scored if float(r.get("late_start_ms", 0.0)) > p.max_late_ms]
    if late:
        worst = max(float(r["late_start_ms"]) for r in late)
        problems.append(f"{len(late)} hit(s) started late (up to {worst:.0f} ms)")
    holed = [r for r in scored if float(r.get("max_gap_ms", 0.0)) > p.max_gap_ms]
    if holed:
        worst = max(float(r["max_gap_ms"]) for r in holed)
        problems.append(
            f"{len(holed)} hit(s) with holes up to {worst:.0f} ms (limit {p.max_gap_ms:.0f} ms)")
    problems += list(data.get("sequence_problems", []))
    return problems


def _faults(con: Console, since_seq: int) -> tuple[list[str], int]:
    """(new fatal lines, reboot count) from the lines after `since_seq` — kit_churn's."""
    fatal: list[str] = []
    reboots = 0
    for ev in con.events(since_seq=since_seq):
        if ev.kind == "fatal":
            fatal.append(ev.line)
        elif ev.kind == "reboot":
            reboots += 1
    return fatal, reboots


def _env(summary: str, data: dict[str, Any], artifacts: list[Artifact]) -> Report:
    return Report(passed=False, summary=summary, data=data, artifacts=artifacts, exit_code=2)


def _wait(ctx: Context, seconds: float) -> None:
    if seconds > 0:
        ctx.cancelled.wait(seconds)
    ctx.check_cancelled()


class KitSwapAudioScenario:
    name = "kit_swap_audio"
    Params = Params
    description = "swap kits while pads fire, recorded through UAC2, scored per tone"

    def run(self, ctx: Context, params: Params, progress: Progress) -> Report:
        try:
            p = params
            pads = list(p.pads) if p.pads else list(range(PAD_COUNT))
            if not pads or any(x < 0 or x > 15 for x in pads):
                raise HilError(BAD_ARGS, f"pads must be 0..15, got {p.pads}", pads=p.pads)
            if p.swaps < 1:
                raise HilError(BAD_ARGS, f"swaps must be >= 1, got {p.swaps}", swaps=p.swaps)
            if p.dwell <= TONE_S:
                raise HilError(
                    BAD_ARGS,
                    f"dwell must exceed the {TONE_S:g}s tone (got {p.dwell:g}s)",
                    hint="a swap inside a tone that is still being scored cannot be "
                         "told from a pad reloaded mid-sample", dwell=p.dwell)
            swaps = build_swaps(p.swaps, p.lead, p.dwell)
            data: dict[str, Any] = {
                "kits": {"a": p.kit_a, "b": p.kit_b}, "kit_ids": {}, "provisioned": [],
                "pads": pads, "swaps": swaps, "last_kit_id": None,
                "wav": None, "capture": None, "capture_health": None,
                "hits_sent": 0, "send_failures": 0, "swaps_sent": 0,
                "scored": [], "totals": None, "sequence_problems": [],
                "kit_swap_log": [], "kit_status": None, "pad_stats": None,
                "fatal": [], "reboots": 0,
                "audio_mode_kept": bool(p.keep_audio_mode), "failures": [],
            }
            artifacts: list[Artifact] = []
            try:
                with card_hold.for_run(ctx, data, enabled=p.exclusive_card,
                                       expected_s=p.lead + len(swaps) * p.dwell + 30):
                    return self._run(ctx, p, progress, pads, swaps, data, artifacts)
            except HilError as e:
                if e.code not in SETUP_CODES:
                    raise
                data["failures"].append(f"{e.code}: {e.message}")
                hint = f" ({e.hint})" if e.hint else ""
                return _env(f"kit_swap_audio: {e.message}{hint}", data, artifacts)
        finally:
            ctx.close_all()

    # -- the run --------------------------------------------------------
    def _run(self, ctx: Context, p: Params, progress: Progress, pads: list[int],
             swaps: list[dict[str, Any]], data: dict[str, Any],
             artifacts: list[Artifact]) -> Report:
        ctx.check_cancelled()
        progress(0, 5, "setting the device up over CDC")

        if ctx.device.usb_mode is not UsbMode.DEFAULT or ctx.device.ports.cdc is None:
            ctx.log("no CDC port — assuming the USB audio profile, switching back to default")
            ctx.device = set_mode(ctx.device, UsbMode.DEFAULT)

        # The console outlives the profile switch (it is the STM bridge VCP), so
        # it is the only witness to a panic that happens while CDC is gone.
        con = ctx.open_console(reset=False)
        link = ctx.open_cdc()
        try:
            # ensure_kits can hand back a *different* link: provisioning closes
            # this one to push, and the reboot after re-enumerates the CDC node.
            link, provisioned = self._provision(ctx, p, link, con)
            data["provisioned"] = provisioned
            ids = self._prepare(ctx, p, link, pads)
            data["kit_ids"] = {"a": ids[0], "b": ids[1]}
        finally:
            link.close()   # the CDC endpoint goes away with the profile switch below

        kit_ids = [ids[0], ids[1]]
        data["last_kit_id"] = kit_ids[int(swaps[-1]["kit_index"])]
        seq0 = con.seq
        total_s = p.lead + p.swaps * p.dwell + TAIL_S
        ctx.log(f"kit_swap_audio: {p.swaps} swap(s) every {p.dwell:.1f}s under "
                f"{p.hit_rate:.1f} hits/s across {len(pads)} pads, ~{total_s:.0f}s")

        wav = Path(p.out) if p.out else ctx.workdir / "kit_swap_audio.wav"
        data["wav"] = str(wav)
        session, info, artifact = _audio.start_capture(
            ctx, open_capture, wav, preset=p.preset, volume=p.volume,
            keep_mode=p.keep_audio_mode,
            on_open=lambda: progress(1, 5, "switching to the USB audio profile and "
                                           "opening the capture"))
        data["capture_start"] = info
        artifacts.append(artifact)

        transport: ForgivingTransport | None = None
        player: Player | None = None
        try:
            transport = ForgivingTransport(open_pad_transport(ctx.device, "sysex_esp"))
            _wait(ctx, STREAM_SETTLE_S)
            progress(2, 5, f"playing and swapping ({total_s:.0f}s)")
            # Hit and swap stamps are absolute monotonic; the scoring windows are
            # relative to the WAV's own t=0, which is session.started_at (set once
            # the stream reported active), not "now". Re-basing happens once, below.
            player = Player(transport, p.hit_rate, pads, p.velocity, p.gate_ms)
            player.start()
            self._play_swaps(ctx, transport, swaps, kit_ids, time.monotonic())
            _wait(ctx, p.dwell)
        finally:
            if player is not None:
                player.stop()
                player.join(timeout=3.0)
            try:
                _wait(ctx, TAIL_S)   # the last tone has to land inside the recording
            finally:
                if transport is not None:
                    transport.close()
                _audio.stop_capture(ctx, session, data)
                data["send_failures"] = transport.failures if transport is not None else 0

        raw_hits = player.hits() if player is not None else []
        data["hits_sent"] = len(raw_hits)

        why = _audio.host_loss(data, data.get("capture_health"))
        if why:
            return _audio.not_measured("kit_swap_audio", why, data, artifacts)
        reasons = _audio.capture_problems(data.get("capture") or {},
                                          data.get("capture_health") or {})
        if reasons:
            reason = "; ".join(reasons)
            data["failures"].append(f"capture: {reason}")
            return _env(f"kit_swap_audio: {reason}", data, artifacts)

        progress(3, 5, f"scoring {len(raw_hits)} hits")
        # Without the stream's own t=0 every window is scored off the wrong slice,
        # which would read as sixteen silent pads rather than as the bench fault
        # it is (pitched_kit's re-basing, and its reason for existing).
        if session.started_at is None:
            data["failures"].append("capture: the stream never reported a start time")
            return _env("kit_swap_audio: the capture never started; nothing to score against",
                        data, artifacts)
        started_at = float(session.started_at)
        hits = [{"t": h["t"] - started_at, "pad": h["pad"]} for h in raw_hits]
        for s in swaps:
            s["t_wav"] = round(float(s["t_sent"]) - started_at, 4) if "t_sent" in s else None
        result = analyze_tone_tracks(wav, hits, tone_s=TONE_S, freqs=candidate_freqs(pads),
                                     dropout_db=p.dropout_db)
        data["scored"] = result["hits"]
        data["totals"] = result["totals"]
        data["sequence_problems"] = kit_sequence_problems(
            result["hits"], [{"t": s["t_wav"], "kit_index": s["kit_index"]}
                             for s in swaps if s.get("t_wav") is not None])

        # -- the device's own account: profile is back, CDC is back -------
        progress(4, 5, "reading the device's swap log")
        _wait(ctx, KIT_SETTLE_S)
        link = ctx.open_cdc()
        try:
            data["kit_swap_log"] = verbs.kit_swap_log(link, log=ctx.log)
            try:
                data["kit_status"] = verbs.kit_status(link)
            except HilError as e:
                ctx.log(f"KIT_STATUS: {e.code}: {e.message}")
            try:
                data["pad_stats"] = verbs.pad_stats(link)
            except HilError as e:
                ctx.log(f"PAD_STATS: {e.code}: {e.message}")
        finally:
            link.close()

        data["fatal"], data["reboots"] = _faults(con, seq0)
        progress(5, 5, "done")
        return self._verdict(p, data, artifacts)

    def _provision(self, ctx: Context, p: Params, link: CdcLink,
                   con: Console) -> tuple[CdcLink, list[str]]:
        """Build and push the two sine fixtures if the card does not have them.

        A bench whose card was reflashed, or a board that has never seen these
        kits, should not cost a run: the generator is in this package
        (`crosspad_hil.fixtures`), the push is `FS_PUT`, and the reboot is what
        makes the kit scan notice. `--no-provision` keeps the old behaviour.
        """
        return _audio.ensure_kits(
            ctx, link, [p.kit_a, p.kit_b], fixtures.write_sine_kits, console=con,
            provision=not p.no_provision,
            missing_hint_suffix=" — push it with tools/make_sine_kits.py")

    def _prepare(self, ctx: Context, p: Params, link: CdcLink,
                 pads: list[int]) -> tuple[int, int]:
        hint = " — push both with tools/make_sine_kits.py"
        id_a, name_a, current = _audio.resolve_kit(link, kit_name=p.kit_a,
                                                   missing_hint_suffix=hint)
        id_b, name_b, _current = _audio.resolve_kit(link, kit_name=p.kit_b,
                                                    missing_hint_suffix=hint)
        if id_a == id_b:
            raise HilError(ENV, f"{p.kit_a!r} and {p.kit_b!r} are the same kit ({id_a})",
                           hint="the two fixtures must differ or nothing is being swapped")
        _audio.load_kit(ctx, link, id_a, name_a, current, wait_s=KIT_LOAD_TIMEOUT_S)
        _audio.ensure_app_running(
            ctx, link, SAMPLER_APP,
            not_installed_reason=f"{SAMPLER_APP} is not in the launcher — nothing routes "
                                 f"NoteOn to the sample engine",
            settle_s=APP_START_SETTLE_S)
        # Reset here, so PAD_STATS afterwards counts this run's hits only.
        verbs.pad_stats(link, reset=True)
        ctx.log(f"kit A = {name_a!r} ({id_a}), kit B = {name_b!r} ({id_b})")
        return int(id_a), int(id_b)

    def _play_swaps(self, ctx: Context, transport: ForgivingTransport,
                    swaps: list[dict[str, Any]], kit_ids: list[int], t_ref: float) -> None:
        """Fire each swap at its scheduled moment while the player keeps hitting pads."""
        for s in swaps:
            delta = (t_ref + float(s["t"])) - time.monotonic()
            if delta > 0:
                _wait(ctx, delta)
            else:
                ctx.check_cancelled()
            kit_id = kit_ids[int(s["kit_index"])]
            sent = transport.send_kit_load(kit_id)
            s["t_sent"] = time.monotonic()
            s["kit_id"] = kit_id
            s["sent"] = sent
            ctx.log(f"  swap {s['stage']}/{len(swaps)} -> kit {'AB'[s['kit_index']]} "
                    f"({kit_id}) at t={float(s['t']):.1f}s")

    # -- verdict --------------------------------------------------------
    def _verdict(self, p: Params, data: dict[str, Any], artifacts: list[Artifact]) -> Report:
        data["swaps_sent"] = sum(1 for s in data["swaps"] if s.get("sent"))
        problems = swap_verdict(p, data)
        totals = data.get("totals") or {}
        if problems:
            return Report(passed=False,
                          summary="kit_swap_audio: FAIL — " + "; ".join(problems),
                          data=data, artifacts=artifacts, exit_code=1)
        note = ""
        if float(totals.get("dropout_ms", 0.0)) > 0:
            note = (f", WARN {totals['dropout_ms']:.0f} ms of dropout under the "
                    f"{p.max_gap_ms:.0f} ms gap limit")
        if int(totals.get("late_starts", 0)):
            note += (f", WARN {totals['late_starts']} late start(s) up to "
                     f"{totals['late_start_max_ms']:.0f} ms under the "
                     f"{p.max_late_ms:.0f} ms limit")
        deferred = sum(1 for row in data.get("kit_swap_log", []) if row.get("deferred"))
        deferred_note = (", deferred pads honoured" if deferred
                         else ", no pad needed deferring")
        provisioned = data.get("provisioned") or []
        pushed = f" (provisioned {', '.join(provisioned)})" if provisioned else ""
        summary = (f"kit_swap_audio: PASS — {data['swaps_sent']} swaps under "
                   f"{p.hit_rate:g} hits/s, {totals.get('audible', 0)}/"
                   f"{totals.get('hits', 0)} hits audible, "
                   f"{totals.get('hits_with_dropouts', 0)} dropouts"
                   f"{deferred_note}{note}{pushed}")
        return Report(passed=True, summary=summary, data=data, artifacts=artifacts,
                      exit_code=0)


SCENARIO = register(KitSwapAudioScenario())


def main(argv: list[str] | None = None) -> int:
    """Entry point matching the other scenarios' `main()`."""
    import sys

    from crosspad_hil.cli import main as cli_main

    args = list(sys.argv[1:] if argv is None else argv)
    return cli_main(["run", "kit_swap_audio", *args])
