"""Does velocity change loudness, and does every hit sound at all?

Port of ``platform-idf/tools/hil_velocity.py``. Pad hits are injected through
the *physical* pad path (SysEx 0x1D/0x07 on the STM bridge -> PadManager ->
pad logic -> sample engine) and the result is measured on the PCB's own
DAC->ADC loop over the UAC2 capture endpoint, so nothing but the board is in
the loop and no human has to listen.

Four questions, in the old script's order: does velocity change loudness (peak
dBFS against velocity), is the response monotonic, how consistently late is a
hit, and were any hits dropped.

Two things about the measurement are deliberate and easy to get wrong:

* **The tap is codec 0 LINE1** (capture preset ``speaker``). The loop through
  codec 1 is near unity but compresses above ~0.2 amplitude, which flattens the
  top of a velocity sweep and turns a real dynamic range into a fake one.
  Codec 0 is ~24 dB down with a -85 dB floor and stays linear across the sweep.
* **The absolute latency number means nothing on its own.** It contains the
  host's MIDI TX stack and the whole USB capture pipeline. Compare it across
  firmware revisions on one rig; the rig-independent figures are the jitter
  spread and the SysEx echo round-trip, which is why ``echo_probes`` exists.
"""
from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from typing import Any

from crosspad_hil import card_hold, knowledge, verbs
from crosspad_hil.analyze import analyze_velocity
from crosspad_hil.capture import CaptureResult, CaptureSession
from crosspad_hil.devices import UsbMode
from crosspad_hil.errors import HilError
from crosspad_hil.midi import MidiIO, MidiRole, decode_echo_seq, sysex_echo
from crosspad_hil.scenarios import _audio
from crosspad_hil.scenarios.base import Artifact, Context, Progress, Report, register
from crosspad_hil.stim import Hit, Stimulus, open_transport
from crosspad_hil.usbmode import set_mode

# hil_velocity.main: lead_s = 1.5, and total_s carries another 1.5 s of tail.
LEAD_S = 1.5
TAIL_S = 1.5
# hil_velocity's SYSEX_ROUTE_LOOP is codec0 ADC -> LINE1 plus mic source -> codec0,
# which is exactly capture.PRESETS["speaker"]. The route, the RT-mixer resume and
# the restore all live in CaptureSession, so they cannot drift apart from cap_*.
CAPTURE_PRESET = "speaker"
ROUTE_SETTLE_S = 1.0        # time.sleep(1) after the route frames land
APP_RETRY_SETTLE_S = 8.0    # "Sampler not running (attempt 1) — retrying"
APP_ATTEMPTS = 2            # for attempt in (1, 2)
ECHO_TIMEOUT_S = 0.5        # measure_echo_rtt: deadline = time.time() + 0.5
ECHO_GAP_S = 0.02           # time.sleep(0.02) between probes
STIM_TAIL_S = 5.0           # slack on top of the schedule before giving up on it
SAMPLER_APP = "Sampler"
DEFAULT_VELS = [8, 16, 24, 32, 48, 64, 80, 96, 112, 127]


@dataclass
class Params:
    kit: int = field(default=0, metadata={"help": "kit id to load before playing"})
    pad: int = field(default=0, metadata={"help": "pad to strike"})
    vels: list[int] = field(
        default_factory=lambda: list(DEFAULT_VELS),
        metadata={"help": "velocity sweep, ascending"})
    rounds: int = field(
        default=3, metadata={"help": "sweep repetitions, interleaved rather than blocked"})
    period: float = field(default=1.0, metadata={"help": "seconds between hits"})
    gate_ms: float = field(default=80.0, metadata={"help": "press->release gap"})
    min_range_db: float = field(
        default=12.0, metadata={"help": "required dB spread between the quietest and "
                                        "loudest velocity"})
    max_jitter_ms: float = field(
        default=40.0, metadata={"help": "largest allowed latency spread between the "
                                        "median hit and the slow tail"})
    echo_probes: int = field(
        default=20, metadata={"help": "SysEx echo round-trip probes; 0 skips them"})
    keep_mode: bool = field(
        default=False, metadata={"help": "stay in the USB-audio profile afterwards, "
                                         "for back-to-back runs"})
    assume_ready: bool = field(
        default=False, metadata={"help": "skip the kit/app/mode prologue — the device is "
                                         "already set up"})
    exclusive_card: bool = field(
        default=True,
        metadata={"help": "pause the audio-nodes service for the run (bench holder only), so "
                          "the capture opens the card directly instead of through PipeWire"})


def _env(summary: str, data: dict[str, Any], artifacts: list[Artifact]) -> Report:
    return Report(passed=False, summary=summary, data=data, artifacts=artifacts, exit_code=2)


def _wait(ctx: Context, seconds: float) -> None:
    if seconds > 0:
        ctx.cancelled.wait(seconds)
    ctx.check_cancelled()


def echo_rtt(stm: Any, esp: Any, probes: int, ctx: Context) -> dict[str, Any] | None:
    """Round-trip echo frames down the bridge and back out the ESP's own MIDI IN.

    This is the control path a pad hit takes with the audio pipeline left out —
    host MIDI TX, STM relay, ESP parse, ESP MIDI TX — so the sweep's latency can
    be read as "control path plus engine plus audio path" instead of as one
    number nobody can attribute. It is also the only latency figure here that
    does not depend on the recording rig.

    ``None`` when nothing came back at all, which is what hil_velocity reported.
    """
    k = knowledge.load("sysex")
    route_id = int(k["audio_route"]["id"])
    echo_sub = int(k["audio_route"]["subs"]["echo"])
    esp.drain()
    rtts: list[float] = []
    for i in range(probes):
        ctx.check_cancelled()
        seq = i & 0x7F
        t0 = time.monotonic()
        stm.send_sysex(sysex_echo(seq))
        deadline = t0 + ECHO_TIMEOUT_S
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                break
            frame = esp.receive(remaining)
            if frame is None:
                continue
            # hil_velocity took the next reply of any kind; matching the sequence
            # instead cannot turn a healthy probe into a lost one, and it stops a
            # late reply to probe N-1 being timed as probe N's.
            if len(frame) >= 9 and frame[2] == route_id and frame[3] == echo_sub \
                    and decode_echo_seq(frame) == seq:
                rtts.append((time.monotonic() - t0) * 1000.0)
                break
        _wait(ctx, ECHO_GAP_S)
    if not rtts:
        return None
    ordered = sorted(rtts)
    mid = len(ordered) // 2
    median = ordered[mid] if len(ordered) % 2 else (ordered[mid - 1] + ordered[mid]) / 2.0
    return {"n": len(rtts), "mean_ms": round(sum(rtts) / len(rtts), 2),
            "median_ms": round(median, 2), "min_ms": round(ordered[0], 2),
            "max_ms": round(ordered[-1], 2)}


def build_sweep(pad: int, vels: list[int], rounds: int, period: float,
                gate_ms: float) -> list[Hit]:
    """One hit per velocity per round, interleaved on a constant grid.

    Interleaved and not blocked: a run that plays every quiet hit first and
    every loud hit last cannot tell a velocity curve from the sample engine
    warming up.
    """
    hits: list[Hit] = []
    for rnd in range(rounds):
        for i, vel in enumerate(vels):
            t = (rnd * len(vels) + i) * period
            hits.append(Hit(t_ms=t * 1000.0, pad=pad, vel=int(vel), gate_ms=gate_ms))
    return hits


class VelocityScenario:
    name = "velocity"
    Params = Params
    description = "sweep pad velocity and measure loudness, monotonicity and trigger jitter"

    def run(self, ctx: Context, params: Params, progress: Progress) -> Report:
        try:
            p = params
            vels = [int(v) for v in p.vels]
            hits = build_sweep(p.pad, vels, p.rounds, p.period, p.gate_ms)
            data: dict[str, Any] = {
                "config": {"kit": p.kit, "pad": p.pad, "velocities": vels,
                           "rounds": p.rounds, "period": p.period, "gate_ms": p.gate_ms,
                           "min_range_db": p.min_range_db, "max_jitter_ms": p.max_jitter_ms,
                           "preset": CAPTURE_PRESET},
                "expected_hits": len(hits), "prologue": None, "echo_rtt": None,
                "capture": None, "stimulus": None, "velocity": None,
            }
            artifacts: list[Artifact] = []
            if not vels:
                return _env("velocity: the velocity sweep is empty", data, artifacts)
            if ctx.device.ports.stm_midi is None:
                return _env(f"velocity: {ctx.device.id} has no STM bridge MIDI port — the pad "
                            "hits (SysEx 0x1D 07) have nothing to travel on", data, artifacts)
            ctx.check_cancelled()

            report = self._prologue(ctx, p, data, artifacts)
            if report is not None:
                return report

            with card_hold.for_run(ctx, data, enabled=p.exclusive_card,
                                   expected_s=len(hits) * p.period + 30):
                progress(1, 4, "entering the USB-audio profile and starting the capture")
                session = CaptureSession(
                    ctx.device, seconds=None, preset=CAPTURE_PRESET,
                    out=ctx.workdir / "capture.wav", keep_mode=p.keep_mode, log=ctx.log)
                try:
                    info = session.start()
                except HilError as e:
                    # hil_velocity: "FATAL: audio card did not appear" is exit 2 — the
                    # bench never got into a state where the firmware could be judged.
                    return _env(f"velocity: could not start the capture — {e.code}: {e.message}"
                                + (f" ({e.hint})" if e.hint else ""), data, artifacts)
                ctx.opened.append(session)
                data["capture"] = dict(info)
                artifacts.append(
                    Artifact(path=str(info["wav"]), mime="audio/wav", role="capture"))
                ctx.device = session.device
                t_capture = session.started_at or time.monotonic()
                _wait(ctx, ROUTE_SETTLE_S)

                try:
                    if p.echo_probes:
                        progress(2, 4, f"{p.echo_probes} control-path echo probes")
                        data["echo_rtt"] = self._echo(ctx, p)
                        ctx.log(f"control RTT: {data['echo_rtt']}")

                    progress(3, 4, f"sweeping {len(vels)} velocities x {p.rounds} rounds")
                    t_first = self._sweep(ctx, p, hits, data)
                    result = session.stop()
                finally:
                    # Whatever happened, do not leave the board in the audio profile
                    # with the RT mixer resumed and the mics unrouted.
                    session.close()
                data["capture"]["result"] = result.to_dict()
                ctx.log(f"capture: {result.seconds:.1f}s peak {result.peak_dbfs} dBFS "
                        f"({result.backend}, {result.health.get('detail')})")

                progress(4, 4, "analysing")
                return self._analyse(ctx, p, hits, t_first - t_capture, result, data, artifacts)
        finally:
            ctx.close_all()

    # --- prologue -----------------------------------------------------------
    def _prologue(self, ctx: Context, p: Params, data: dict[str, Any],
                  artifacts: list[Artifact]) -> Report | None:
        """Kit, app and USB profile. Returns a Report only when the bench is wrong."""
        state: dict[str, Any] = {"assume_ready": p.assume_ready, "kit": None, "app": None}
        data["prologue"] = state
        if p.assume_ready:
            ctx.log("assume-ready: leaving the kit, the app and the USB profile alone")
            return None
        if ctx.device.usb_mode is UsbMode.AUDIO:
            # The kit and the app are loaded over CDC, which the audio profile
            # does not have. Back-to-back runs start here.
            ctx.log("device is in the USB-audio profile — restoring the default one first")
            try:
                ctx.device = set_mode(ctx.device, UsbMode.DEFAULT, wait=True)
            except HilError as e:
                return _env(f"velocity: could not leave the audio profile — "
                            f"{e.code}: {e.message}", data, artifacts)
        link = ctx.open_cdc()
        try:
            try:
                verbs.kit_load(link, p.kit)
            except HilError as e:
                return _env(f"velocity: KIT_LOAD {p.kit} failed — {e.code}: {e.message} "
                            "(firmware without hil_control, or no SD card)", data, artifacts)
            state["kit"] = p.kit
            ctx.log(f"kit {p.kit} loaded")

            apps = verbs.app_list(link)
            if SAMPLER_APP not in apps.get("apps", []):
                return _env(f"velocity: {SAMPLER_APP} is not in the launcher — nothing owns "
                            "pad logic, so a hit would make no sound", data, artifacts)
            for attempt in range(1, APP_ATTEMPTS + 1):
                try:
                    verbs.app_start(link, SAMPLER_APP)
                except HilError as e:
                    ctx.log(f"APP_START {SAMPLER_APP} (attempt {attempt}): "
                            f"{e.code}: {e.message}")
                if verbs.app_list(link).get("running") == SAMPLER_APP:
                    state["app"] = SAMPLER_APP
                    break
                ctx.log(f"{SAMPLER_APP} not running (attempt {attempt}) — retrying")
                _wait(ctx, APP_RETRY_SETTLE_S)
            else:
                return _env(f"velocity: {SAMPLER_APP} would not start after "
                            f"{APP_ATTEMPTS} attempts", data, artifacts)
        finally:
            # The audio profile is about to take this endpoint away; a link left
            # open here holds a port lock on a node that no longer exists.
            link.close()
        return None

    # --- the run ------------------------------------------------------------
    def _echo(self, ctx: Context, p: Params) -> dict[str, Any] | None:
        """Echo probes go down the bridge; replies only ever come back on the ESP port."""
        if ctx.device.ports.esp_midi is None:
            ctx.log("no ESP MIDI port: the control-path round trip cannot be read")
            return None
        stm = MidiIO(ctx.device, MidiRole.STM)
        esp = MidiIO(ctx.device, MidiRole.ESP)
        stm.open()
        try:
            esp.open()
            try:
                return echo_rtt(stm, esp, p.echo_probes, ctx)
            finally:
                esp.close()
        finally:
            stm.close()

    def _sweep(self, ctx: Context, p: Params, hits: list[Hit],
               data: dict[str, Any]) -> float:
        """Play the sweep. Returns the monotonic time the first hit was scheduled for."""
        _wait(ctx, LEAD_S)      # quiet stretch the noise floor is measured on
        transport = open_transport(ctx.device, "sysex_stm")
        # No counters: PAD_STATS lives on CDC, and this profile has no CDC. What
        # arrived is read off the audio instead, which is the point of the test.
        stim = Stimulus(hits, transport, counters=None, backoff=False)
        t_first = time.monotonic()
        stim.start()
        try:
            span = hits[-1].t_ms / 1000.0 + p.gate_ms / 1000.0
            stim.wait(span + STIM_TAIL_S)
        finally:
            data["stimulus"] = stim.stop()      # also closes the transport
        _wait(ctx, TAIL_S)                      # let the last decay into the capture
        return t_first

    # --- verdict ------------------------------------------------------------
    def _analyse(self, ctx: Context, p: Params, hits: list[Hit], offset_s: float,
                 result: CaptureResult, data: dict[str, Any],
                 artifacts: list[Artifact]) -> Report:
        # Send times relative to the start of the capture. The analyser estimates
        # and removes the constant pipeline offset from the first onset, so what
        # matters here is the spacing, not the absolute anchor.
        why = _audio.host_loss(data, getattr(result, "health", None))
        if why:
            return _audio.not_measured("velocity", why, data, artifacts)
        expected = [{"t": round(offset_s + h.t_ms / 1000.0, 4), "vel": h.vel} for h in hits]
        try:
            verdict = analyze_velocity(result.wav, expected=expected,
                                       min_range_db=p.min_range_db,
                                       max_jitter_ms=p.max_jitter_ms)
        except HilError as e:
            return _env(f"velocity: the capture could not be analysed — "
                        f"{e.code}: {e.message}", data, artifacts)
        body = verdict.to_dict()
        data["velocity"] = body
        path = ctx.workdir / "velocity.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(data, indent=2, default=str) + "\n", encoding="utf-8")
        artifacts.append(Artifact(path=str(path), mime="application/json", role="analysis"))

        for c in body["curve"]:
            ctx.log(f"  vel {c['vel']:>3}  peak {c['peak_mean_dbfs']:>7} dBFS  "
                    f"std {c['peak_std_db']:>4}  measured {c['measured_rel_db']:>7}  "
                    f"expected {c['expected_rel_db']:>7}")
        ctx.log(f"noise floor {body['noise_floor_dbfs']} dBFS, "
                f"{body['matched']}/{body['expected']} hits matched, "
                f"range {body['dynamic_range_db']} dB, "
                f"{body['monotonic_inversions']} inversion(s)")

        passed = verdict.passed
        head = (f"velocity: {'PASS' if passed else 'FAIL'} — "
                f"{body['matched']}/{body['expected']} hits sounded, "
                f"range {body['dynamic_range_db']} dB over vel "
                f"{p.vels[0]}..{p.vels[-1]}")
        if body["reasons"]:
            head += " — " + "; ".join(body["reasons"])
        if result.silent:
            # Every hit missing and a silent capture is the prologue's fault far
            # more often than the firmware's, so say so rather than only counting.
            head += f" (capture is silent: {result.hint})"
        return Report(passed=passed, summary=head, data=data, artifacts=artifacts,
                      exit_code=0 if passed else 1)


SCENARIO = register(VelocityScenario())


def main(argv: list[str] | None = None) -> int:
    """Entry point for the platform-idf shim `tools/hil_velocity.py`."""
    import sys

    from crosspad_hil.cli import main as cli_main

    args = list(sys.argv[1:] if argv is None else argv)
    return cli_main(["run", "velocity", *args])
