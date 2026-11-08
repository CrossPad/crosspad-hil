"""Prove the speaker amp gate acoustically, through the CrossPad's own microphones.

Port of ``platform-idf/tools/hil_speaker_acoustic.py``. A loopback only shows
that samples reach the codec; it says nothing about the amplifier, which sits
after the DAC. This records the built-in mics while the sampler plays, so what
is measured is air rather than wiring: silence -> pad hits -> silence, and a
working gate raises the mic floor when the pads fire.

The input map, measured by sweeping both codecs against a forced-off amp:

    codec 0 LINE2   built-in mics, stereo single-ended   <- the only acoustic path
    codec 0 LINE1   DAC loopback
    codec 1 LINE1   DAC loopback, clips
    codec 1 LINE2   line-in jack, disconnected — noise floor only

Anything but codec 0 LINE2 (capture preset ``mics``) reads the DAC electrically
and will happily report a working speaker with the amp powered down. That is
the opposite of the trap ``audio_loopback`` has, and the reason the preset is a
parameter with a loud default rather than something the caller has to know.

Setup runs over CDC first and the capture second, in that order: the USB-audio
profile has no CDC endpoint (MIDI + UAC2 already claim every IN endpoint the
ESP32-S3 has), but a running app survives the re-enumeration. Everything needed
while capturing therefore travels as SysEx 0x1D.

Recording is ``crosspad_hil.capture``, the pad stimulus is ``crosspad_hil.stim``
and the verdict comes from ``crosspad_hil.analyze``.
"""
from __future__ import annotations

import statistics
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from crosspad_hil import card_hold, verbs
from crosspad_hil.devices import UsbMode
from crosspad_hil.errors import BAD_ARGS, NOT_SUPPORTED, HilError
from crosspad_hil.knowledge import load as load_knowledge
from crosspad_hil.midi import MidiIO, MidiRole, sysex_audio_route
from crosspad_hil.scenarios import _audio, _deps
from crosspad_hil.scenarios.base import Artifact, Context, Progress, Report, register
from crosspad_hil.usbmode import set_mode

# --- the old script's literals -----------------------------------------------
HIT_PERIOD_S = 0.5          # hil_speaker_acoustic: hit_period, not a knob there either
TAIL_SETTLE_S = 5.5         # post window starts at pre + play + 5.5 (the amp hold)
HIT_WINDOW_S = 0.30         # "+3 dB in the 300 ms after each trigger"
SYSEX_GAP_S = 0.15          # time.sleep(0.15) between route frames
APP_START_WAIT_S = 2.5      # cdc_command('APP_START Sampler', wait=2.5)
STIM_TIMEOUT_PAD_S = 10.0
SAMPLER_APP = "Sampler"
ACOUSTIC_PRESET = "mics"    # codec 0 LINE2
MAX_VOLUME = 100


@dataclass
class Params:
    pad: int = field(default=0, metadata={"help": "pad index to hit"})
    kit: int = field(default=0, metadata={"help": "kit id to load before playing"})
    hits: int = field(default=12, metadata={"help": "number of pad hits"})
    velocity: int = field(default=120, metadata={"help": "velocity of each hit"})
    gate_ms: float = field(default=60.0, metadata={"help": "press-to-release time per hit"})
    pre_s: float = field(default=3.0, metadata={"help": "silence recorded before the hits"})
    post_s: float = field(default=7.0, metadata={"help": "silence recorded after the hits"})
    preset: str = field(
        default=ACOUSTIC_PRESET,
        metadata={"help": "capture path; only 'mics' (codec 0 LINE2) hears the speaker — "
                          "the others read the DAC electrically"})
    rise_db: float = field(
        default=3.0, metadata={"help": "dB over the mic floor that counts as a hit"})
    hit_ratio: float = field(
        default=0.75, metadata={"help": "fraction of hits that must be located to pass"})
    transport: str = field(
        default="sysex_esp",
        metadata={"help": "pad transport while capturing (sysex_esp|sysex_stm); there is "
                          "no CDC in the audio profile"})
    load_timeout: float = field(default=15.0, metadata={"help": "seconds a kit load may take"})
    enum_timeout: float = field(
        default=20.0, metadata={"help": "seconds to wait for a USB profile to enumerate"})
    keep_mode: bool = field(
        default=False, metadata={"help": "leave the device in the audio profile on exit"})
    exclusive_card: bool = field(
        default=True,
        metadata={"help": "pause the audio-nodes service for the run (bench holder only), so "
                          "the capture opens the card directly instead of through PipeWire"})


def _sub(name: str) -> int:
    return int(load_knowledge("sysex")["audio_route"]["subs"][name])


def _env(summary: str, data: dict[str, Any], artifacts: list[Artifact]) -> Report:
    return Report(passed=False, summary=summary, data=data, artifacts=artifacts, exit_code=2)


def open_both_codecs(device: Any, midi_factory: Any = None, sleep: Any = None) -> None:
    """Volume up and unmute on *both* codecs, the frames capture does not send.

    The session's route prologue picks the ADC input, the mic source and resumes
    the RT mixer, but nothing unmutes: a muted DAC is silence that looks exactly
    like a dead amplifier, which is the one verdict this scenario must not get
    wrong for the wrong reason.
    """
    # Resolved here rather than as default arguments so a caller (and a test) can
    # swap them, and so the module attribute is what is used.
    factory = midi_factory or MidiIO
    pause = sleep or time.sleep
    io = factory(device, MidiRole.ESP)
    io.open()
    try:
        for codec in (0, 1):
            io.send_sysex(sysex_audio_route(_sub("volume"), codec, MAX_VOLUME))
            pause(SYSEX_GAP_S)
            io.send_sysex(sysex_audio_route(_sub("mute"), codec, 0))
            pause(SYSEX_GAP_S)
    finally:
        io.close()


def verdict_from(analysis: dict[str, Any], hits: int, *, rise_db: float,
                 hit_ratio: float) -> dict[str, Any]:
    """The old script's two-part rule, over the velocity kernel's per-hit levels.

    Legacy compared a p90 of the 100 ms RMS envelope during playback against the
    median of the pre-roll. Here the floor is the kernel's own noise floor (the
    same quiet stretch) and the rise is the median of the per-hit peaks, which
    is the quantity the +3 dB threshold was always about — and unlike a
    percentile over the whole playback phase it cannot be dragged up by one
    thump. The pass rule itself is unchanged: three quarters of the hits
    located, and a rise of at least 3 dB.
    """
    floor = float(analysis.get("noise_floor_dbfs", -120.0))
    per_hit = [h for h in analysis.get("per_hit", []) if h.get("peak_dbfs") is not None]
    rises = [round(float(h["peak_dbfs"]) - floor, 2) for h in per_hit]
    located = int(analysis.get("matched", len(rises)))
    needed = max(1, int(hits * hit_ratio))
    rise = round(statistics.median(rises), 2) if rises else 0.0
    return {
        "floor_dbfs": round(floor, 2),
        "per_hit_db": rises,
        "rise_db": rise,
        "hits_detected": located,
        "hits_required": needed,
        "extra_onsets": list(analysis.get("extra", [])),
        "latency_ms": dict(analysis.get("latency_ms", {})),
        "passed": located >= needed and rise >= rise_db,
    }


class SpeakerAcousticScenario:
    name = "speaker_acoustic"
    Params = Params
    description = "record the built-in mics while the sampler plays; did the speaker sound?"

    def run(self, ctx: Context, params: Params, progress: Progress) -> Report:
        try:
            p = params
            play_s = p.hits * HIT_PERIOD_S
            data: dict[str, Any] = {
                "preset": p.preset, "pad": p.pad, "kit": p.kit, "hits": p.hits,
                "pre_s": p.pre_s, "play_s": play_s, "post_s": p.post_s,
                "reasons": [], "restored": True, "final_mode": ctx.device.usb_mode.value,
            }
            artifacts: list[Artifact] = []
            try:
                capture_mod = _deps.module("capture")
                analyze_mod = _deps.module("analyze")
                stim_mod = _deps.module("stim")
            except HilError as e:
                return _env(f"speaker_acoustic: {e.message}", data, artifacts)

            presets: dict[str, tuple[int, int]] = dict(capture_mod.PRESETS)
            if p.preset not in presets:
                raise HilError(BAD_ARGS, f"unknown capture preset {p.preset!r}",
                               hint="one of: " + ", ".join(sorted(presets)))
            data["codec"], data["adc_input"] = presets[p.preset]
            if p.preset != ACOUSTIC_PRESET:
                # Kept as a warning rather than an error: sweeping the other inputs is
                # how the map above was measured in the first place.
                ctx.log(f"preset {p.preset!r} is not the acoustic path — this reads the DAC "
                        "electrically and will pass with the amp powered down")

            progress(0, 3, "setting the sampler up over CDC")
            try:
                running = self._setup(ctx, p)
            except HilError as e:
                return _env(f"speaker_acoustic: setup failed — {e.code}: {e.message}"
                            + (f" ({e.hint})" if e.hint else ""), data, artifacts)
            data["running_app"] = running
            if running is None or running.lower() != SAMPLER_APP.lower():
                return _env(f"speaker_acoustic: {SAMPLER_APP} is not running "
                            f"(APP_LIST running={running or '-'}) — the test would measure "
                            "silence", data, artifacts)

            wav = ctx.workdir / "mic.wav"
            progress(1, 3, f"recording {p.pre_s + play_s + p.post_s:.0f} s of the mics")
            try:
                with card_hold.for_run(ctx, data, enabled=p.exclusive_card,
                                       expected_s=p.pre_s + play_s + p.post_s + 30):
                    capture, stim_status = self._record(ctx, p, play_s, wav, capture_mod,
                                                        stim_mod)
            except HilError as e:
                self._note_mode(ctx, data)
                return _env(f"speaker_acoustic: capture failed — {e.code}: {e.message}",
                            data, artifacts)
            self._note_mode(ctx, data)
            artifacts.append(Artifact(path=str(wav), mime="audio/wav", role="capture"))
            data["capture"] = capture
            data["stimulus"] = stim_status
            why = _audio.host_loss(data, capture.get("health"))
            if why:
                return _audio.not_measured("speaker_acoustic", why, data, artifacts)

            progress(2, 3, "analysing")
            schedule = [{"t": round(p.pre_s + i * HIT_PERIOD_S, 4), "vel": p.velocity}
                        for i in range(p.hits)]
            try:
                # The velocity kernel is the only one that reports a level per hit,
                # which is what the acoustic rise is measured from. Its velocity-ladder
                # checks are inert here because every hit is sent at one velocity.
                analysis = analyze_mod.analyze("velocity", str(wav), expected=schedule,
                                               rise_db=p.rise_db, tol_s=HIT_WINDOW_S)
                tail = analyze_mod.analyze(
                    "silence", str(wav),
                    expected={"skip_s": p.pre_s + play_s + TAIL_SETTLE_S})
            except HilError as e:
                if e.code == NOT_SUPPORTED:
                    return _env(f"speaker_acoustic: {e.message}", data, artifacts)
                return Report(passed=False, data=data, artifacts=artifacts, exit_code=1,
                              summary=f"speaker_acoustic: analysis failed — {e.message}")

            measured = verdict_from(analysis, p.hits, rise_db=p.rise_db,
                                    hit_ratio=p.hit_ratio)
            data.update(measured)
            data["tail_rms_dbfs"] = tail.get("rms_dbfs")
            data["analysis"] = analysis
            progress(3, 3, "done")
            return self._verdict(p, data, artifacts, capture)

        finally:
            ctx.close_all()

    # -- phases --------------------------------------------------------

    def _setup(self, ctx: Context, p: Params) -> str | None:
        """Kit, app and RT mixer, over CDC — which only the default profile has."""
        if ctx.device.usb_mode is not UsbMode.DEFAULT or ctx.device.ports.cdc is None:
            ctx.log("leaving the audio profile to reach CDC for setup")
            ctx.device = set_mode(ctx.device, UsbMode.DEFAULT, wait=True,
                                  timeout_s=p.enum_timeout)
        link = ctx.open_cdc()
        try:
            verbs.audio_tasks(link, True)
            verbs.kit_load(link, p.kit, wait_s=p.load_timeout)
            verbs.app_start(link, SAMPLER_APP, wait_s=APP_START_WAIT_S)
            return verbs.app_list(link)["running"]
        finally:
            # The CDC endpoint is about to vanish with the profile switch; a link
            # left open holds a port lock on a node that no longer exists.
            link.close()

    def _record(self, ctx: Context, p: Params, play_s: float, wav: Path,
                capture_mod: Any, stim_mod: Any) -> tuple[dict[str, Any], dict[str, Any]]:
        """silence -> hits -> silence, recorded in one take."""
        total = p.pre_s + play_s + p.post_s
        session = capture_mod.CaptureSession(
            ctx.device, seconds=total, preset=p.preset, out=wav,
            resume_audio_tasks=True, keep_mode=p.keep_mode, log=ctx.log)
        session.start()
        stim_status: dict[str, Any] = {}
        try:
            open_both_codecs(ctx.device)
            ctx.cancelled.wait(p.pre_s)
            ctx.check_cancelled()
            ctx.log(f"--- {p.hits} pad hits on pad {p.pad} ---")
            stim = stim_mod.start_stimulus(
                ctx.device, pads=[p.pad], rate_hz=1.0 / HIT_PERIOD_S, seconds=play_s,
                velocity=p.velocity, gate_ms=p.gate_ms, transport=p.transport)
            try:
                stim.wait(play_s + STIM_TIMEOUT_PAD_S)
            finally:
                stim_status = stim.stop()
            ctx.log("--- silence ---")
            ctx.cancelled.wait(p.post_s)
        finally:
            # Stopping the session is also what puts the USB profile back, so the
            # device it holds afterwards is the current one and ctx's is stale.
            result = session.stop()
            ctx.device = getattr(session, "device", ctx.device)
        return _capture_dict(result), stim_status

    def _note_mode(self, ctx: Context, data: dict[str, Any]) -> None:
        mode = getattr(ctx.device, "usb_mode", None)
        data["final_mode"] = mode.value if mode is not None else "unknown"
        data["restored"] = data["final_mode"] != UsbMode.AUDIO.value

    def _verdict(self, p: Params, data: dict[str, Any], artifacts: list[Artifact],
                 capture: dict[str, Any]) -> Report:
        reasons: list[str] = data["reasons"]
        if capture.get("silent"):
            # Deliberately a FAIL and not an environment error: a dead amplifier
            # produces exactly this, and it is the thing being tested. The hint
            # from capture names the other explanations.
            reasons.append("capture is silent — " + str(capture.get("hint") or "no signal"))
        if data["hits_detected"] < data["hits_required"]:
            reasons.append(f"only {data['hits_detected']}/{p.hits} hits located "
                           f"(need {data['hits_required']}, >= +{p.rise_db:g} dB within "
                           f"{HIT_WINDOW_S * 1000:.0f} ms of each trigger)")
        if data["rise_db"] < p.rise_db:
            reasons.append(f"mics rose {data['rise_db']:.1f} dB over the floor, "
                           f"need {p.rise_db:.1f} dB")
        body = (f"floor {data['floor_dbfs']:.1f} dBFS, rise {data['rise_db']:+.1f} dB, "
                f"{data['hits_detected']}/{p.hits} hits located, "
                f"tail {data.get('tail_rms_dbfs')} dBFS")
        if reasons:
            return Report(passed=False, exit_code=1, data=data, artifacts=artifacts,
                          summary=f"speaker_acoustic: FAIL — {body}; " + "; ".join(reasons))
        return Report(passed=True, exit_code=0, data=data, artifacts=artifacts,
                      summary=f"speaker_acoustic: PASS — {body}")


def _capture_dict(result: Any) -> dict[str, Any]:
    if hasattr(result, "to_dict"):
        return dict(result.to_dict())
    return dict(result)


SCENARIO = register(SpeakerAcousticScenario())


def main(argv: list[str] | None = None) -> int:
    """Entry point for the platform-idf shim ``tools/hil_speaker_acoustic.py``."""
    import sys

    from crosspad_hil.cli import main as cli_main

    args = list(sys.argv[1:] if argv is None else argv)
    return cli_main(["run", "speaker_acoustic", *args])
