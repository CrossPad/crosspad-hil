"""Record the sampler through the CrossPad's own USB audio interface — one cable.

Port of ``platform-idf/tools/hil_sampler_record.py``. The board carries everything
this needs on its native USB port: CDC to set the device up, MIDI to trigger the
pads, and a UAC2 capture endpoint to record from. No audio interface, no second
cable, no microphone in the room.

Four facts about that path decide the order everything happens in, and all four
are load-bearing:

* **The USB-audio profile has no CDC endpoint.** Everything the run needs from
  CDC — the kit, the Sampler, the BLE role, the pad→note map, the counter reset —
  has to be asked for *before* the switch, and the pads then have to be driven
  over SysEx (``0x1D 07/08``, the ESP MIDI port) or over the BLE link.
* **Entering that profile parks the RT mixer** (it fights the UAC bridge for
  I2S), so ``0x1D 06 01`` has to resume it or the capture is three seconds of
  digital silence. :class:`crosspad_hil.capture.CaptureSession` owns that
  prologue, and this scenario must not repeat it.
* **What is recorded is the DAC→ADC loop, not the mics.** ``headphone`` is
  codec1 LINE1, near unity and with the better S/N; ``mics`` (codec0 LINE2)
  hears the room, which is not what the sample engine put on the bus.
* **The default profile must come back whatever happens**, or the next run finds
  a board it cannot talk to. The capture session restores it from its own
  ``finally``, and this scenario stops the session from one too.

``speedtest`` ramps the hit rate in steps and answers, per step, whether the hits
were *audible* — onsets in the recording against onsets injected — which is the
half a counter test cannot see. ``live`` records until ``duration`` (or until the
run is cancelled) and leaves the pads to a human.

One deliberate difference from the old script: the onsets are scored by
:func:`crosspad_hil.analyze.analyze_onset`, whose kernel anchors on the measured
noise floor and matches against the send grid, rather than by the old flux-rise
window. Same question, better-tested kernel — but its default refractory window
(150 ms) would swallow every hit of a 25/s step, so the window and the match
tolerance are narrowed to half the shortest period the schedule contains.
"""
from __future__ import annotations

import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from crosspad_hil import card_hold, verbs
from crosspad_hil.cdc import CdcLink
from crosspad_hil.devices import UsbMode
from crosspad_hil.errors import (
    BAD_ARGS,
    ENV,
    NO_CDC_IN_AUDIO_MODE,
    NO_DEVICE,
    NOT_SUPPORTED,
    TIMEOUT,
    HilError,
)
from crosspad_hil.scenarios import _audio, _deps
from crosspad_hil.scenarios.base import Artifact, Context, Progress, Report, register
from crosspad_hil.usbmode import set_mode

MODES: tuple[str, ...] = ("speedtest", "live")
STIMULI: tuple[str, ...] = ("usb", "ble")
SAMPLER_APP = "Sampler"

# --- timing, every one the old script's literal -------------------------------
LEAD_S = 0.5                 # build_schedule(..., lead_s=0.5): first hit of the take
LEAD_GAP_S = 1.0             # `t += 1.0` after the lead-in hit, before the first step
STREAM_SETTLE_S = 0.4        # time.sleep(0.4) between rec.start() and the first hit
TAIL_S = 1.5                 # "let the last tail decay into the recording"
APP_START_SETTLE_S = 2.0     # time.sleep(2.0) after APP_START Sampler
KIT_LOAD_TIMEOUT_S = 20.0    # the old 40 × 0.5 s poll for `current == kit`
STIM_GRACE_S = 10.0          # slack on top of the schedule before the stimulus is abandoned
STIM_POLL_S = 0.1            # how often the run looks at a stimulus it is waiting out
MAX_LIVE_S = 900.0           # capture.MAX_CAPTURE_S; a live take without a duration stops here

# The pad→note map a kit has not redefined (PadManager's default, note 36 + pad).
BLE_BASE_NOTE = 36

# HilError codes that mean the bench is wrong rather than the firmware: the old
# script's die() paths, which all exited 2. A capture stream that will not open
# and a UAC2 endpoint that never appears are bench problems too.
SETUP_CODES: frozenset[str] = frozenset(
    {ENV, NO_DEVICE, NO_CDC_IN_AUDIO_MODE, NOT_SUPPORTED, TIMEOUT}
)


@dataclass
class Params:
    mode: str = field(default="speedtest",
                      metadata={"help": "speedtest (drive the pads and score them) or live"})
    kit: int | None = field(default=None, metadata={"help": "kit id to load first"})
    kit_name: str | None = field(
        default=None, metadata={"help": "kit by name, case-insensitive (wins over --kit)"})
    preset: str = field(
        default="headphone",
        metadata={"help": "capture path: headphone (the unity DAC->ADC loop), "
                          "speaker, mics, line-in"})
    # Left at the board's own volume the DAC->ADC loop this scenario records
    # through clips: measured 9.09 on BLACK TESLA, peak 0.0 dBFS at the stock 64,
    # and a clipped transient is a transient the onset analysis cannot find — the
    # 2 hits/s step scored 0.667 with the device reporting played=280/280. At 20
    # the take peaks at -21 dBFS and that step scores 1.0. The session puts the
    # board's volume back afterwards.
    volume: int | None = field(default=20,
                               metadata={"help": "capture codec volume 0..100 "
                                                 "(default 20: the loop clips above it)"})
    rates: list[int] = field(
        default_factory=lambda: [2, 4, 6, 8, 12, 16, 20, 25],
        metadata={"help": "speedtest hit rates, hits/s"})
    step: float = field(default=3.0, metadata={"help": "seconds per rate step"})
    gap: float = field(default=1.0, metadata={"help": "silence between steps"})
    pads: str = field(default="0-15", metadata={"help": "pads to cycle, e.g. 0-3,8"})
    velocity: int = field(default=110, metadata={"help": "velocity of every injected hit"})
    gate_ms: int = field(default=30, metadata={"help": "how long each pad is held"})
    stim: str = field(
        default="usb",
        metadata={"help": "speedtest: drive the pads over USB MIDI SysEx (usb) or over the "
                          "BLE MIDI link (ble) — the latter records the sampler while the "
                          "radio, the UAC2 stream and the audio engine all run at once"})
    duration: float = field(
        default=0.0, metadata={"help": "live mode: seconds (0 = until the run is cancelled)"})
    min_audible: float = field(
        default=0.9, metadata={"help": "onset ratio below which a step FAILs"})
    fail_rate: float = field(
        default=16.0, metadata={"help": "only steps at or below this rate must pass"})
    keep_audio_mode: bool = field(
        default=False, metadata={"help": "stay in the USB audio profile when done"})
    exclusive_card: bool = field(
        default=True,
        metadata={"help": "pause the audio-nodes service for the run (bench holder only), so "
                          "the capture opens the card directly instead of through PipeWire"})
    out: str | None = field(
        default=None, metadata={"help": "output WAV (default: capture.wav in the work dir)"})


# --------------------------------------------------------------------------- seams
# Thin wrappers so the run has one place to reach the audio, stimulus and BLE
# machinery — and one place a test can replace it without a sound card.


def _as_env(exc: HilError) -> HilError:
    """A host with no audio stack is a bench problem, not a failed firmware.

    ``capture``/``analyze`` report a missing sounddevice, PortAudio or numpy as
    NOT_SUPPORTED, which the runner would score as a plain FAIL (exit 1). This
    run cannot happen at all without them, so it is ENV — exit 2.
    """
    if exc.code != NOT_SUPPORTED:
        return exc
    return HilError(ENV, exc.message, hint=exc.hint, **exc.details)


def open_capture(device: Any, **kw: Any) -> Any:
    """A :class:`crosspad_hil.capture.CaptureSession` — mode switch, route, WAV, restore."""
    try:
        return _deps.module("capture").CaptureSession(device, **kw)
    except HilError as e:
        raise _as_env(e) from None


def open_pad_transport(device: Any, kind: str) -> Any:
    """A pad transport from :mod:`crosspad_hil.stim`; ``sysex_esp`` in the audio profile."""
    return _deps.module("stim").open_transport(device, kind)


def open_ble_link(address: str | None = None, **kw: Any) -> Any:
    """This PC's radio, connected to the board — :func:`crosspad_hil.ble.connect`."""
    return _deps.module("ble").connect(address, **kw)


def ble_self_address(link: CdcLink) -> str | None:
    """The board's own BLE address, so the radio connects to *this* board.

    ``BLE_STATUS self=`` is the only sound match: the service UUID is shared by
    every BLE MIDI peripheral in range and the name by every CrossPad on the
    bench. It has to be read here, over CDC, because CDC is gone by the time the
    stimulus connects.
    """
    return _deps.module("ble").self_address(link)


def run_stimulus(hits: Any, transport: Any, *, cancelled: Any = None,
                 grace_s: float = STIM_GRACE_S) -> dict:
    """Play ``hits`` through ``transport`` and return the stimulus' final status.

    The counters live on CDC, which this profile does not have, so the stimulus
    is built without them: what the device made of the hits is read back over
    CDC once the default profile is restored.
    """
    stim = _deps.module("stim").Stimulus(hits, transport, counters=None, settle_s=0.0)
    stim.start()
    try:
        deadline = time.monotonic() + stim.plan.seconds + grace_s
        while stim.running and time.monotonic() < deadline:
            if cancelled is not None:
                if cancelled.wait(STIM_POLL_S):
                    break
            else:
                stim.wait(STIM_POLL_S)
    finally:
        status = stim.stop()
    return status


def analyze_wav(wav: Path, expected: list[float], **kw: Any) -> dict:
    """Onset verdict for the take: which injected hits were audible, and how late."""
    try:
        return dict(_deps.module("analyze").analyze("onset", str(wav), expected=expected, **kw))
    except HilError as e:
        raise _as_env(e) from None


# --------------------------------------------------------------------------- transports


class ForgivingTransport:
    """Counts a refused write instead of raising it, so the take survives one.

    From hil_sampler_record.Midi.sysex: WinMM refuses the write when the device
    leaves the bus, and a take is worth more than the message that was lost —
    raising here throws away the audio recorded so far *and* skips the switch
    back to the default USB profile, leaving the board somewhere the next run
    has to recover from. :class:`crosspad_hil.stim.Stimulus` halts the schedule
    on a transport exception, which is exactly what must not happen.
    """

    def __init__(self, inner: Any) -> None:
        self._inner = inner
        self.name = getattr(inner, "name", "?")
        self.failures = 0

    def press(self, pad: int, vel: int) -> None:
        try:
            self._inner.press(pad, vel)
        except Exception:  # noqa: BLE001 - a lost message is counted, not raised
            self.failures += 1

    def release(self, pad: int) -> None:
        try:
            self._inner.release(pad)
        except Exception:  # noqa: BLE001
            self.failures += 1

    def close(self) -> None:
        self._inner.close()


class BlePadTransport:
    """Pads over the BLE MIDI link, exercised while USB streams the audio back.

    Notes map to pads the way any external keyboard's do (``PAD_NOTES``, which a
    kit redefines — so it is asked for over CDC before the profile switch, not
    assumed to be 36+pad).
    """

    name = "ble"

    def __init__(self, ble: Any, notes: dict[int, int] | None = None,
                 base_note: int = BLE_BASE_NOTE) -> None:
        self._ble = ble
        self._notes = dict(notes or {})
        self._base = base_note

    def note_for(self, pad: int) -> int:
        return self._notes.get(pad, self._base + pad) & 0x7F

    def press(self, pad: int, vel: int) -> None:
        self._ble.send_note(True, self.note_for(pad), vel & 0x7F)

    def release(self, pad: int) -> None:
        self._ble.send_note(False, self.note_for(pad))

    def close(self) -> None:
        self._ble.disconnect()


# --------------------------------------------------------------------------- schedule


def build_schedule(rates: list[float], *, step_s: float, gap_s: float, pads: list[int],
                   gate_ms: float, velocity: int,
                   lead_s: float = LEAD_S) -> tuple[list[dict], list[dict], float]:
    """(pattern, steps, total seconds) — hil_sampler_record.build_schedule.

    The pattern is what :mod:`crosspad_hil.stim` plays; ``steps`` carries each
    rate's window and the indices its hits occupy in the schedule, which is how
    a per-step ratio is read back out of one whole-take onset analysis. The
    lead-in hit is index 0: it is the recording's time zero, and including it in
    the analysed grid is what lets the matcher find the capture's own latency.
    """
    pattern: list[dict] = []
    steps: list[dict] = []

    def add(t_s: float, pad: int) -> int:
        pattern.append({"t_ms": round(t_s * 1000.0, 3), "pad": pad,
                        "vel": int(velocity), "gate_ms": float(gate_ms)})
        return len(pattern) - 1

    t = lead_s
    add(t, pads[0])
    t += LEAD_GAP_S
    k = 0
    for rate in rates:
        start = t
        n = max(1, int(round(step_s * rate)))
        times: list[float] = []
        first = len(pattern)
        for i in range(n):
            ti = start + i / rate
            add(ti, pads[k % len(pads)])
            k += 1
            times.append(ti)
        steps.append({"rate": float(rate), "start": start, "end": start + n / rate,
                      "n": n, "times": times, "indices": list(range(first, first + n))})
        t = start + n / rate + gap_s
    return pattern, steps, t


def onset_windows(rates: list[float]) -> dict[str, float]:
    """Refractory and match tolerance for a schedule whose fastest step is ``rates``.

    The library's defaults (150 ms both) are sized for a velocity ramp; at 25
    hits/s the period is 40 ms and they would score all but the first hit of a
    step as missed. Half the shortest period lands within a millisecond or two
    of the old script's ±window, which is the geometry that was believed on
    hardware.
    """
    analyze = _deps.module("analyze")
    fastest = max([float(r) for r in rates] or [1.0])
    half_period = 0.5 / max(fastest, 1e-6)
    return {"refractory_s": min(analyze.ONSET_REFRACTORY_S, half_period),
            "tol_s": min(analyze.ONSET_TOL_S, half_period)}


# --------------------------------------------------------------------------- helpers


def _env(summary: str, data: dict[str, Any], artifacts: list[Artifact]) -> Report:
    return Report(passed=False, summary=summary, data=data, artifacts=artifacts, exit_code=2)


def _wait(ctx: Context, seconds: float) -> None:
    """Sleep that returns early on cancel and then raises CANCELLED."""
    if seconds > 0:
        ctx.cancelled.wait(seconds)
    ctx.check_cancelled()


def _safe(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]", "_", name or "kit")


class SamplerRecordScenario:
    name = "sampler_record"
    Params = Params
    description = "record the sampler through the board's own UAC2 endpoint and score it"

    def run(self, ctx: Context, params: Params, progress: Progress) -> Report:
        try:
            p = params
            if p.mode not in MODES:
                raise HilError(BAD_ARGS, f"unknown mode {p.mode!r}",
                               hint="one of: " + ", ".join(MODES))
            if p.stim not in STIMULI:
                raise HilError(BAD_ARGS, f"unknown stimulus {p.stim!r}",
                               hint="one of: " + ", ".join(STIMULI))
            data: dict[str, Any] = {
                "mode": p.mode, "stim": p.stim if p.mode == "speedtest" else None,
                "preset": p.preset, "kit": None, "kit_name": None, "wav": None,
                "capture": None, "capture_health": None, "steps": [], "failed_rates": [],
                "onset": None, "audible": 0, "hits_planned": 0, "hits_sent": 0,
                "send_failures": 0,
                "schedule_error_ms": None, "pad_stats": None, "pad_notes": None,
                "ble_self": None,
                "audio_mode_kept": bool(p.keep_audio_mode), "interrupted": False,
                "failures": [],
            }
            artifacts: list[Artifact] = []
            try:
                with card_hold.for_run(ctx, data, enabled=p.exclusive_card,
                                       expected_s=p.duration or None):
                    return self._run(ctx, p, progress, data, artifacts)
            except HilError as e:
                if e.code not in SETUP_CODES:
                    raise
                # Every one of these was a die() in the old script: exit 2, and
                # keep whatever the run had already learned.
                data["failures"].append(f"{e.code}: {e.message}")
                hint = f" ({e.hint})" if e.hint else ""
                return _env(f"sampler_record: {e.message}{hint}", data, artifacts)
        finally:
            ctx.close_all()

    # -- the run ------------------------------------------------------------
    def _run(self, ctx: Context, p: Params, progress: Progress, data: dict[str, Any],
             artifacts: list[Artifact]) -> Report:
        ctx.check_cancelled()
        progress(0, 4, "setting the device up over CDC")

        # No CDC means the board is already in the audio profile, and SysEx is
        # the only way back (hil_sampler_record.ensure_default_mode).
        if ctx.device.usb_mode is not UsbMode.DEFAULT or ctx.device.ports.cdc is None:
            ctx.log("no CDC port — assuming the USB audio profile, switching back to default")
            ctx.device = set_mode(ctx.device, UsbMode.DEFAULT)

        link = ctx.open_cdc()
        ble_notes: dict[int, int] | None = None
        ble_addr: str | None = None
        try:
            kit_id, kit_name = self._prepare(ctx, p, link)
            data["kit"], data["kit_name"] = kit_id, kit_name
            if p.mode == "speedtest" and p.stim == "ble":
                ble_notes, ble_addr = self._prepare_ble(ctx, link)
                data["pad_notes"] = dict(ble_notes)
                data["ble_self"] = ble_addr
            verbs.pad_stats(link, reset=True)
        finally:
            # The CDC endpoint goes away with the profile switch; a handle held
            # across it points at a node that no longer exists.
            link.close()

        wav = Path(p.out) if p.out else ctx.workdir / f"{p.mode}-{_safe(kit_name)}.wav"
        session, info, artifact = _audio.start_capture(
            ctx, open_capture, wav, preset=p.preset, volume=p.volume,
            keep_mode=p.keep_audio_mode,
            on_open=lambda: progress(1, 4, "switching to the USB audio profile and "
                                            "opening the capture"))
        data["capture_start"] = info
        artifacts.append(artifact)
        steps: list[dict] = []
        schedule: list[float] = []
        try:
            _wait(ctx, STREAM_SETTLE_S)     # let the stream settle before the first hit
            progress(2, 4, f"recording ({p.mode})")
            if p.mode == "live":
                self._record_live(ctx, p, data)
            else:
                steps, schedule = self._record_speedtest(ctx, p, data, ble_notes, ble_addr)
        finally:
            # Whatever happened, finalize the WAV and put the profile back: the
            # session's stop() does both, and its restore runs even if the
            # measurement below raises.
            _audio.stop_capture(ctx, session, data)

        progress(3, 4, "scoring the take")
        if p.mode == "speedtest" and _audio.host_loss(data, data.get("capture_health")) is None:
            self._score(ctx, p, wav, data, steps, schedule)
        self._read_counters(ctx, data)
        progress(4, 4, "done")
        return self._verdict(p, data, artifacts)

    # -- setup, all of it over CDC and all of it before the switch ----------
    def _prepare(self, ctx: Context, p: Params, link: CdcLink) -> tuple[int, str]:
        kit_id, name, current = _audio.resolve_kit(link, kit_id=p.kit, kit_name=p.kit_name)
        _audio.load_kit(ctx, link, kit_id, name, current, wait_s=KIT_LOAD_TIMEOUT_S)
        # app_start polls APP_LIST itself and raises TIMEOUT if the launch never
        # lands; the settle wait is the old script's, for the app's first LVGL
        # frames rather than the launch (ensure_app_running's docstring).
        _audio.ensure_app_running(
            ctx, link, SAMPLER_APP,
            not_installed_reason=f"{SAMPLER_APP} is not in the launcher — nothing to record",
            settle_s=APP_START_SETTLE_S)
        return int(kit_id), name

    def _prepare_ble(self, ctx: Context, link: CdcLink) -> tuple[dict[int, int], str | None]:
        """Server mode and the note map, both of them last-chance: CDC is about to go.

        The stimulus connects *to* the board, so the board has to be the
        peripheral whatever its persisted role is; and a kit redefines which
        note each pad answers to, so the map is asked for rather than assumed.
        """
        verbs.ble_start(link, 1)
        ctx.log(f"BLE: {verbs.ble_status(link)}")
        notes = dict(verbs.pad_notes(link).get("notes") or {})
        if not notes:
            raise HilError(ENV, "PAD_NOTES gave nothing; cannot map BLE notes to pads",
                           hint="firmware predating PAD_NOTES cannot be driven over BLE here")
        return notes, ble_self_address(link)

    # -- the two modes ------------------------------------------------------
    def _record_live(self, ctx: Context, p: Params, data: dict[str, Any]) -> None:
        seconds = p.duration if p.duration > 0 else MAX_LIVE_S
        ctx.log(f"recording — play the pads ({seconds:.0f} s, or cancel to stop)")
        # A cancelled live take is a finished take, not a failed one: the old
        # script caught KeyboardInterrupt, kept the audio and still reported.
        # ctx.check_cancelled() is deliberately not called here.
        data["interrupted"] = bool(ctx.cancelled.wait(seconds))

    def _record_speedtest(self, ctx: Context, p: Params, data: dict[str, Any],
                          ble_notes: dict[int, int] | None,
                          ble_addr: str | None) -> tuple[list[dict], list[float]]:
        stim_mod = _deps.module("stim")
        pads = stim_mod.parse_pads(p.pads)
        rates = [float(r) for r in p.rates]
        if not rates:
            raise HilError(BAD_ARGS, "no rates to test", hint="e.g. --rates 2,4,8,16")
        pattern, steps, total = build_schedule(
            rates, step_s=p.step, gap_s=p.gap, pads=pads, gate_ms=float(p.gate_ms),
            velocity=p.velocity, lead_s=LEAD_S)
        hits = stim_mod.pattern_to_hits(pattern)
        data["hits_planned"] = len(hits)
        data["steps"] = [{"rate": s["rate"], "start": s["start"], "end": s["end"],
                          "injected": s["n"], "audible": 0, "ratio": 0.0}
                         for s in steps]
        schedule = [h.t_ms / 1000.0 for h in hits]
        ctx.log(f"speedtest: {len(rates)} steps, {len(pads)} pads, ~{total:.0f} s, "
                f"{sum(s['n'] for s in steps)} hits, stimulus over {p.stim}")

        if p.stim == "ble":
            transport: Any = BlePadTransport(open_ble_link(ble_addr), ble_notes)
        else:
            # No CDC in this profile: the pads travel as SysEx 0x1D 07/08 on the
            # ESP's own MIDI port, which the audio profile keeps.
            transport = open_pad_transport(ctx.device, "sysex_esp")
        forgiving = ForgivingTransport(transport)
        try:
            status = run_stimulus(hits, forgiving, cancelled=ctx.cancelled)
        finally:
            data["send_failures"] = forgiving.failures
        data["hits_sent"] = int(status.get("presses_sent", 0))
        data["schedule_error_ms"] = status.get("schedule_error_ms")
        for err in status.get("errors", []):
            data["failures"].append(f"stimulus: {err}")
        _wait(ctx, TAIL_S)                     # let the last tail decay into the recording
        return steps, schedule

    # -- verdict ------------------------------------------------------------
    def _score(self, ctx: Context, p: Params, wav: Path, data: dict[str, Any],
               steps: list[dict], schedule: list[float]) -> None:
        if not steps or not schedule:
            return
        onset = analyze_wav(wav, schedule, **onset_windows([s["rate"] for s in steps]))
        data["onset"] = onset
        missed = set(int(i) for i in onset.get("missed", []))
        total_audible = 0
        ctx.log("   rate   injected  audible   ratio")
        for record, step in zip(data["steps"], steps, strict=True):
            got = sum(1 for i in step["indices"] if i not in missed)
            total_audible += got
            record["audible"] = got
            record["ratio"] = round(got / step["n"], 3) if step["n"] else 0.0
            flag = ""
            if step["rate"] <= p.fail_rate and record["ratio"] < p.min_audible:
                flag = "  <-- FAIL"
                data["failed_rates"].append(step["rate"])
            ctx.log(f"  {step['rate']:5.0f}/s  {step['n']:8d}  {got:7d}  "
                    f"{record['ratio']:6.2f}{flag}")
        data["audible"] = total_audible
        if total_audible == 0:
            data["failures"].append(
                "nothing audible at all — wrong capture route, or no kit loaded")

    def _read_counters(self, ctx: Context, data: dict[str, Any]) -> None:
        """What the device made of the hits, read once CDC is back.

        Reported, never asserted on: the audible ratio is this scenario's
        verdict, and the counters are how you tell a hit that was refused from
        one that played and could not be heard.
        """
        if data["audio_mode_kept"] or ctx.device.ports.cdc is None:
            return
        try:
            link = ctx.open_cdc()
            data["pad_stats"] = verbs.pad_stats(link)
        except HilError as e:
            ctx.log(f"PAD_STATS: {e.code}: {e.message}")
            return
        stats = data["pad_stats"]
        if stats["press"]:
            pct = 100.0 * stats["played"] / stats["press"]
            ctx.log(f"device: {stats['press']} hits accepted, {stats['played']} reached "
                    f"the sample engine ({pct:.1f}%)")

    def _verdict(self, p: Params, data: dict[str, Any], artifacts: list[Artifact]) -> Report:
        problems: list[str] = list(data["failures"])
        capture = data.get("capture") or {}
        health = data.get("capture_health") or {}
        # The holes the host made are not the device's: the take is not scored
        # (no missed hits, no silence), only what the device itself said counts.
        why = _audio.host_loss(data, health)
        if why is None:
            problems += _audio.capture_problems(capture, health)
        if data["failed_rates"]:
            rates = ", ".join(f"{r:.0f}/s" for r in data["failed_rates"])
            problems.append(f"hits went missing at or below {p.fail_rate:.0f}/s: {rates}")
        if data["send_failures"]:
            problems.append(f"{data['send_failures']} pad message(s) could not be sent — "
                            "the device left the bus mid-run")
        seconds = float(capture.get("seconds") or 0.0)
        peak = capture.get("peak_dbfs")
        if why is not None and not problems:
            return _audio.not_measured("sampler_record", why, data, artifacts)
        if why is not None:
            problems.append(f"capture not measured: {why}")
        if problems:
            return Report(passed=False,
                          summary="sampler_record: FAIL — " + "; ".join(problems),
                          data=data, artifacts=artifacts, exit_code=1)
        if p.mode == "live":
            note = " (interrupted)" if data["interrupted"] else ""
            summary = (f"sampler_record: PASS — {seconds:.1f} s recorded, peak {peak} dBFS"
                       + note)
        else:
            audible = data.get("audible", 0)
            summary = (f"sampler_record: PASS — {audible}/{data['hits_planned'] - 1} injected "
                       f"hits audible across {len(data['steps'])} rate steps, peak {peak} dBFS")
        return Report(passed=True, summary=summary, data=data, artifacts=artifacts, exit_code=0)


SCENARIO = register(SamplerRecordScenario())


def main(argv: list[str] | None = None) -> int:
    """Entry point for the platform-idf shim `tools/hil_sampler_record.py`."""
    import sys

    from crosspad_hil.cli import main as cli_main

    args = list(sys.argv[1:] if argv is None else argv)
    return cli_main(["run", "sampler_record", *args])
