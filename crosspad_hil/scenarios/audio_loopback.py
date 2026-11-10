"""USB audio loopback: play a known multitone into the board and judge what comes back.

Port of ``platform-idf/tools/hil_audio_loopback.py``. One cycle plays the
reference into the UAC2 playback endpoint while recording the capture endpoint,
then asks the multitone kernel whether every tone came back, at level, in tune,
and without glitch bursts or dropouts. Run for hours it doubles as a stability
probe: when the audio card disappears (the board reset back into CDC+MIDI) the
event is counted and the profile re-entered instead of ending the run.

Two things about this path are not obvious, and both produce a capture that is
three seconds of silence when they are got wrong:

* the USB-audio profile parks the RT mixer, so it has to be resumed
  (SysEx ``0x1D 06 01``) after the mode switch — ``capture`` does that;
* the path worth recording is the DAC->ADC loop (``headphone`` = codec1 LINE1,
  near unity, ~110 dB S/N), not the built-in mics, which hear the room rather
  than what the codec put on the bus.

Recording is ``crosspad_hil.capture`` and the verdict is
``crosspad_hil.analyze``'s ``multitone`` kernel, both of which were ported from
this same script. Playback lives here because it is the only thing in the
library that ever needs to push audio *into* the device.
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import time
import wave
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from crosspad_hil import card_hold
from crosspad_hil import host_gain as hostgain
from crosspad_hil.analyze import MULTITONE_MIN_RMS_DBFS
from crosspad_hil.audio_nodes import OUT_NODE
from crosspad_hil.capture import PIPEWIRE_DEVICE, pipewire_target_env
from crosspad_hil.devices import (
    Device,
    UsbMode,
    discover,
    is_uac2_name,
    parse_arecord_list,
    select,
)
from crosspad_hil.errors import BAD_ARGS, ENV, NOT_SUPPORTED, HilError
from crosspad_hil.scenarios import _audio, _deps
from crosspad_hil.scenarios.base import Artifact, Context, Progress, Report, register
from crosspad_hil.usbmode import set_mode

# --- timings and levels, every one the old script's literal -------------------
PLAY_DELAY_S = 0.8          # run_cycle: time.sleep(0.8) between arecord and aplay
CAPTURE_PAD_S = 3.0         # rec_len = int(tone_seconds + 3)
PLAY_TIMEOUT_PAD_S = 15.0   # aplay timeout = tone_seconds + 15
MODE_SETTLE_S = 1.5         # ensure_audio_mode: let ALSA settle once the card is back
REF_SR = 44100        # the board's engine rate; the reference is generated at it so
                      # aplay does not resample
# Reference level per capture path. LINE1 on either codec is the near-unity
# DAC->ADC loop and compresses above ~0.2 FS, so it is fed 0.15; the mic path is
# an attenuated copy (~-25 dB) and needs the full 0.7.
LOOP_ADC_INPUT = 1
LOOP_REF_LEVEL = 0.15
MIC_REF_LEVEL = 0.7
AUDIO_EXTRA_HINT = "pip install 'crosspad-hil[audio]' (Linux also needs libportaudio2)"
# The jack loop (codec 1, INGAIN's IN2) is measured at a pinned preamp gain and
# the level floor follows it: see _audio's "the jack's preamp".


@dataclass
class Params:
    duration_hours: float = field(
        default=1.0, metadata={"help": "total run time in hours (0 = one cycle and stop)"})
    cycles: int | None = field(
        default=None, metadata={"help": "stop after this many cycles (default: duration only)"})
    tone_seconds: float = field(
        default=10.0, metadata={"help": "reference tone length per cycle"})
    interval: float = field(default=15.0, metadata={"help": "pause between cycles in seconds"})
    freqs: list[int] = field(
        default_factory=lambda: [440, 997, 3181],
        metadata={"help": "test tones in Hz (comma-separated)"})
    warn_glitch_per_s: float = field(
        default=5.0,
        metadata={"help": "WARN above this many glitch bursts per second; the fixed "
                          "i2s1_tx underrun measured ~0/s, the old regression ~250/s"})
    preset: str = field(
        default="headphone",
        metadata={"help": "capture path: headphone|speaker|mics|line-in "
                          "(headphone = codec1 LINE1, the DAC->ADC loop)"})
    ref_level: float | None = field(
        default=None, metadata={"help": "reference amplitude 0..1 (default: from the preset)"})
    enum_timeout: float = field(
        default=20.0, metadata={"help": "seconds to wait for the UAC2 profile to enumerate"})
    keep_mode: bool = field(
        default=False, metadata={"help": "leave the device in the audio profile on exit"})
    in2_gain: int | None = field(
        default=_audio.IN2_DEFAULT_DB,
        metadata={"help": "line-in (IN2, codec 1) preamp gain in dB for the jack-loop presets; "
                          "set over CDC before the run, the previous value put back after "
                          "(the board keeps it in NVS); the level floor follows it. "
                          "Omit to leave the board's gain alone"})
    free_loop: bool = field(
        default=True,
        metadata={"help": "on an input pair that hears Line 3/4 (which switches the jack "
                          "loop's driver off), select a pair without it for the run and put "
                          "the old one back (the board keeps it in NVS); for the LINE1 presets"})
    exclusive_card: bool = field(
        default=True,
        metadata={"help": "pause the audio-nodes service for the run, so the capture opens "
                          "the card directly instead of through PipeWire (which lost seconds)"})
    host_gain: bool = field(
        default=True,
        metadata={"help": "put the host's ALSA mixer and PipeWire volumes for the card at "
                          "0 dB each time it enumerates, and record them"})


# --------------------------------------------------------------------------- playback


def _sounddevice() -> Any:
    try:
        import sounddevice  # type: ignore[import-not-found]
    except Exception as exc:  # noqa: BLE001 - missing PortAudio raises OSError, not ImportError
        raise HilError(NOT_SUPPORTED, f"sounddevice is unavailable: {exc}",
                       hint=AUDIO_EXTRA_HINT) from exc
    return sounddevice


def _api_rank() -> dict[str, int]:
    """Host-API preference, borrowed from ``capture`` so both directions of the
    loopback settle on the same 48 kHz endpoint instead of one of them landing
    on MME and resampling for nothing."""
    return dict(getattr(_deps.module("capture"), "API_RANK", {}))


class SoundDevicePlayer:
    """Push the reference WAV at the board's UAC2 playback endpoint.

    Frames go out as raw int16 exactly as they sit in the file, which keeps
    playback free of numpy — a missing analysis extra should not stop the
    device from being driven.
    """

    name = "sounddevice"
    block_frames = 1024

    def __init__(self, sd: Any | None = None) -> None:
        self._sd = sd

    def _mod(self) -> Any:
        if self._sd is None:
            self._sd = _sounddevice()
        return self._sd

    def _index(self) -> int:
        sd = self._mod()
        rank = _api_rank()
        best: tuple[int, int] | None = None
        for idx, d in enumerate(sd.query_devices()):
            if not is_uac2_name(str(d["name"])) or int(d.get("max_output_channels", 0)) < 2:
                continue
            api = str(sd.query_hostapis(d["hostapi"])["name"])
            score = rank.get(api, 2)
            if best is None or score < best[0]:
                best = (score, idx)
        if best is None:
            raise HilError(ENV, "no CrossPad UAC2 playback endpoint",
                           hint="the capture side enumerated but the speaker side did not; "
                                "`crosspad-hil devices` and check the audio profile")
        return best[1]

    def play(self, device: Device, wav: Path, timeout_s: float) -> None:
        sd = self._mod()
        with wave.open(str(wav), "rb") as w:
            rate, channels = w.getframerate(), w.getnchannels()
            frames = w.readframes(w.getnframes())
        stream = sd.RawOutputStream(samplerate=rate, channels=channels, dtype="int16",
                                    device=self._index())
        stream.start()
        try:
            step = self.block_frames * channels * 2
            for off in range(0, len(frames), step):
                stream.write(frames[off:off + step])
        finally:
            stream.stop()
            stream.close()


class AplayPlayer:
    """Linux fallback, and what the old script used exclusively."""

    name = "aplay"

    def __init__(self, run: Any = subprocess.run) -> None:
        self._run = run

    def _card(self, device: Device) -> str:
        if device.ports.uac2 is not None and device.ports.uac2.alsa_id:
            return device.ports.uac2.alsa_id
        # `aplay -l` prints the same table `arecord -l` does, so the same parser reads it.
        out = self._run(["aplay", "-l"], capture_output=True, text=True, timeout=5.0).stdout
        for _card, alsa_id, human in parse_arecord_list(out or ""):
            if is_uac2_name(human):
                return alsa_id
        raise HilError(ENV, "no CrossPad playback card in `aplay -l`",
                       hint="usb-mode set audio, then `crosspad-hil devices`",
                       device=device.id)

    def _aplay(self, device_arg: str, wav: Path, timeout_s: float) -> tuple[int, str]:
        argv = ["aplay", "-q", "-D", device_arg, str(wav)]
        # Through PipeWire, the CrossPad's node -- not the desktop's speakers.
        target = pipewire_target_env(OUT_NODE) if device_arg == PIPEWIRE_DEVICE else {}
        try:
            # LC_ALL=C so "busy" below is not a localised message.
            done = self._run(argv, capture_output=True, timeout=timeout_s,
                             env={**os.environ, "LC_ALL": "C", **target})
        except (OSError, subprocess.SubprocessError) as exc:
            raise HilError(ENV, f"aplay failed to run: {exc}",
                           hint="install alsa-utils, or " + AUDIO_EXTRA_HINT) from exc
        return done.returncode, (done.stderr or b"").decode(errors="replace").strip()

    def _pw_play(self, wav: Path, timeout_s: float) -> tuple[int, str]:
        # `aplay -D pipewire` goes through ALSA's pipewire plugin, which stalls
        # when the CrossPad sink churns during a USB-profile switch (a 25 s
        # timeout in the gate, no sound out). pw-play talks to PipeWire natively
        # and does not, so it is the fallback when the card is held by the DAW
        # nodes. A stall here still surfaces as ENV, not a firmware fault.
        argv = ["pw-play", "--target", OUT_NODE, str(wav)]
        try:
            done = self._run(argv, capture_output=True, timeout=timeout_s,
                             env={**os.environ, "LC_ALL": "C"})
        except (OSError, subprocess.SubprocessError) as exc:
            raise HilError(ENV, f"pw-play failed to run: {exc}",
                           hint="install pipewire (pw-play), or free the card "
                                "(crosspad-hil audio nodes) and retry") from exc
        return done.returncode, (done.stderr or b"").decode(errors="replace").strip()

    def play(self, device: Device, wav: Path, timeout_s: float) -> None:
        rc, err = self._aplay(f"plughw:{self._card(device)}", wav, timeout_s)
        if rc != 0 and "busy" in err.lower():
            # The card is owned exclusively -- on this bench by PipeWire holding
            # the CrossPad DAW nodes. Play through PipeWire natively (pw-play)
            # rather than aplay's pipewire plugin, which stalled the whole gate.
            if shutil.which("pw-play"):
                rc, err = self._pw_play(wav, timeout_s)
            else:
                rc, err = self._aplay(PIPEWIRE_DEVICE, wav, timeout_s)
        if rc != 0:
            raise HilError(ENV, f"aplay failed: {err}")


def default_player() -> Any:
    """sounddevice when it imports, aplay when alsa-utils is there, else NOT_SUPPORTED."""
    try:
        return SoundDevicePlayer(_sounddevice())
    except HilError:
        if sys.platform.startswith("linux") and shutil.which("aplay"):
            return AplayPlayer()
        raise


# --------------------------------------------------------------------------- helpers


def ref_level(preset: str, presets: dict[str, tuple[int, int]]) -> float:
    """Amplitude the reference is written at for a capture path."""
    _codec, adc = presets[preset]
    return LOOP_REF_LEVEL if adc == LOOP_ADC_INPUT else MIC_REF_LEVEL


def _refresh(ctx: Context) -> Device | None:
    """This device as it looks right now, or None if it is not enumerated at all."""
    try:
        return select(discover(), ctx.device.id)
    except HilError:
        return None


def _audio_ready(device: Device | None) -> bool:
    """hil_audio_loopback.audio_card_present(), asked of one device."""
    return device is not None and device.ports.uac2 is not None


def _env(summary: str, data: dict[str, Any], artifacts: list[Artifact]) -> Report:
    return Report(passed=False, summary=summary, data=data, artifacts=artifacts, exit_code=2)


class AudioLoopbackScenario:
    name = "audio_loopback"
    Params = Params
    description = "play a multitone through the UAC2 endpoints and judge the return signal"

    def run(self, ctx: Context, params: Params, progress: Progress) -> Report:
        try:
            p = params
            data: dict[str, Any] = {
                "cycles": 0, "pass": 0, "warn": 0, "fail": 0, "host_lost": 0, "mode_reentries": 0,
                "records": [], "failures": [], "cancelled": False,
                "preset": p.preset, "freqs": [float(f) for f in p.freqs],
                "restored": True, "final_mode": ctx.device.usb_mode.value,
                "host_gain": [], "host_gain_final": None,
                "ingain_before": None, "ingain_during": None, "ingain_after": None,
                "ingain_restored": None, "min_rms_dbfs": MULTITONE_MIN_RMS_DBFS,
                "in_pair_before": None, "in_pair_during": None, "in_pair_after": None,
                "in_pair_restored": None,
            }
            artifacts: list[Artifact] = []
            try:
                capture_mod = _deps.module("capture")
                analyze_mod = _deps.module("analyze")
                player = default_player()
            except HilError as e:
                return _env(f"audio_loopback: {e.message}"
                            + (f" ({e.hint})" if e.hint else ""), data, artifacts)
            data["player"] = player.name

            presets: dict[str, tuple[int, int]] = dict(capture_mod.PRESETS)
            if p.preset not in presets:
                raise HilError(BAD_ARGS, f"unknown capture preset {p.preset!r}",
                               hint="one of: " + ", ".join(sorted(presets)))
            data["codec"], data["adc_input"] = presets[p.preset]
            level = p.ref_level if p.ref_level is not None else ref_level(p.preset, presets)
            data["ref_level"] = level

            ref_path = ctx.workdir / "reference.wav"
            try:
                analyze_mod.write_reference(ref_path, data["freqs"], p.tone_seconds,
                                            level=level, sr=REF_SR)
            except HilError as e:
                if e.code == NOT_SUPPORTED:
                    return _env(f"audio_loopback: {e.message}", data, artifacts)
                raise
            artifacts.append(Artifact(path=str(ref_path), mime="audio/wav", role="reference"))
            events = ctx.workdir / "cycles.jsonl"
            artifacts.append(Artifact(path=str(events), mime="application/x-ndjson", role="log"))

            try:
                previous_in2, previous_pair = self._set_in2(ctx, p, data)
            except HilError as e:
                if e.code != ENV:
                    raise
                return _env(f"audio_loopback: {e.message}"
                            + (f" ({e.hint})" if e.hint else ""), data, artifacts)
            try:
                report = self._measure(ctx, p, data, artifacts, ref_path, events, player,
                                       capture_mod, progress)
            finally:
                self._restore_in2(ctx, p, data, previous_in2, previous_pair)
            for note in (_audio.in2_not_restored(data, previous_in2),
                         _audio.in_pair_not_restored(data, previous_pair)):
                if note is not None:
                    report = Report(passed=False, summary=f"{report.summary}; {note}",
                                    data=data, artifacts=artifacts,
                                    exit_code=report.exit_code or 2)
            return report

        finally:
            ctx.close_all()

    def _measure(self, ctx: Context, p: Params, data: dict[str, Any],
                 artifacts: list[Artifact], ref_path: Path, events: Path, player: Any,
                 capture_mod: Any, progress: Progress) -> Report:
        if not self._ensure_audio(ctx, p, data):
            return _env("audio_loopback: cannot get the device into the MIDI+UAC2 profile — "
                        + "; ".join(data["failures"]), data, artifacts)
        try:
            with card_hold.for_run(ctx, data, enabled=p.exclusive_card,
                                   expected_s=(p.duration_hours * 3600.0
                                               or (p.cycles or 1) * (p.tone_seconds + p.interval))):
                self._loop(ctx, p, data, ref_path, events, player, capture_mod, progress)
            if p.host_gain:
                # Read again before the card goes away: a session that restored
                # its stored volume mid-run explains a level failure.
                final = hostgain.read(_card(ctx.device))
                data["host_gain_final"] = final.to_dict()
        except HilError as e:
            if e.code != ENV:
                raise
            # The host lost its playback or capture path mid-run (a card
            # taken by something else, a tool gone missing): the bench's
            # problem, reported once, with the cycles so far kept.
            return _env(f"audio_loopback: {e.message}", data, artifacts)
        finally:
            self._restore(ctx, p, data)

        return self._verdict(data, artifacts)

    # -- the jack's preamp gain -----------------------------------------

    def _set_in2(self, ctx: Context, p: Params,
                 data: dict[str, Any]) -> tuple[int | None, int | None]:
        """Drive the jack loop and pin IN2 at ``p.in2_gain`` for the run.

        Returns ``(IN2 gain, input pair)`` to put back, each None when untouched.
        INGAIN and IN_PAIR are CDC only (not on the companion list), so this
        happens in the default profile, before the switch to USB audio. The pair
        goes first: a pair that hears Line 3/4 leaves the LINE1 presets
        recording a floating jack (see ``_audio.free_jack_loop``).
        """
        pin_gain = p.in2_gain is not None and data.get("codec") == _audio.JACK_CODEC
        free_loop = p.free_loop and data.get("adc_input") == _audio.LOOP_ADC_INPUT
        if not (pin_gain or free_loop):
            return None, None
        if ctx.device.usb_mode is not UsbMode.DEFAULT or ctx.device.ports.cdc is None:
            ctx.device = set_mode(ctx.device, UsbMode.DEFAULT, wait=True,
                                  timeout_s=p.enum_timeout)
        link = _open_cdc(ctx)
        previous = previous_pair = None
        try:
            if free_loop:
                previous_pair = _audio.free_jack_loop(ctx, link, data)
            if pin_gain:
                try:
                    previous = _audio.set_in2(ctx, link, p.in2_gain, data)
                except BaseException:
                    # Nobody restores a pair that was never handed to the caller.
                    if previous_pair is not None:
                        _audio.put_pair_back(ctx, link, previous_pair, data)
                    raise
        finally:
            link.close()
        if previous is not None:
            data["min_rms_dbfs"] = _audio.level_floor(MULTITONE_MIN_RMS_DBFS, p.in2_gain)
            ctx.log(f"level floor {data['min_rms_dbfs']:.0f} dBFS at IN2 {p.in2_gain} dB")
        return previous, previous_pair

    def _restore_in2(self, ctx: Context, p: Params, data: dict[str, Any],
                     previous: int | None, previous_pair: int | None = None) -> None:
        """Put IN2 and the input pair back, whatever the run did; read back. The
        board keeps both in NVS."""
        _audio.restore_in2(ctx, previous, data, previous_pair=previous_pair,
                           open_cdc=_open_cdc, set_mode_fn=set_mode,
                           enum_timeout=p.enum_timeout,
                           back_to_audio=p.keep_mode
                           and ctx.device.usb_mode is UsbMode.AUDIO)
        data["final_mode"] = ctx.device.usb_mode.value

    # -- mode ----------------------------------------------------------

    def _ensure_audio(self, ctx: Context, p: Params, data: dict[str, Any]) -> bool:
        """Enter the audio profile if the card is not already there."""
        fresh = _refresh(ctx)
        if _audio_ready(fresh):
            ctx.device = fresh  # type: ignore[assignment]
            return True
        try:
            ctx.device = set_mode(ctx.device, UsbMode.AUDIO, wait=True, timeout_s=p.enum_timeout)
        except HilError as e:
            data["failures"].append(f"switch to audio failed — {e.code}: {e.message}")
            return False
        ctx.cancelled.wait(MODE_SETTLE_S)
        return True

    def _restore(self, ctx: Context, p: Params, data: dict[str, Any]) -> None:
        """Never leave the board without a CDC endpoint, whatever the run did."""
        if not p.keep_mode and ctx.device.usb_mode is not UsbMode.DEFAULT:
            try:
                ctx.device = set_mode(ctx.device, UsbMode.DEFAULT, wait=True,
                                      timeout_s=p.enum_timeout)
            except HilError as e:
                data["restored"] = False
                data["failures"].append(
                    f"restore to default failed: {e.code}: {e.message} — send "
                    "F0 7D 1B 01 F7 to the ESP MIDI port, or power-cycle")
        data["final_mode"] = ctx.device.usb_mode.value

    # -- cycles --------------------------------------------------------

    def _set_host_gain(self, ctx: Context, p: Params, data: dict[str, Any],
                       when: str) -> None:
        """0 dB on the host side, read back; ENV when it will not stay there.

        The session restores the card's stored mixer and node volumes on every
        enumeration (-20 dB on the bench), so this runs each time the card
        appears, not once.
        """
        if not p.host_gain:
            return
        before, after = hostgain.apply(_card(ctx.device))
        data["host_gain"].append({"when": when, "before": before.to_dict(),
                                  "after": after.to_dict()})
        was = before.deviations()
        if was:
            ctx.log(f"host gain ({when}): {', '.join(was)} -> set to 0 dB")
        for note in after.notes:
            ctx.log(f"host gain: {note}")
        still = after.deviations()
        if still:
            raise HilError(ENV, f"host gain {', '.join(still)} (not 0 dB after setting it)",
                           hint="the level this run measures would be the host's; check "
                                "the card's mixer (alsamixer) and wpctl for the CrossPad nodes")

    def _loop(self, ctx: Context, p: Params, data: dict[str, Any], ref_path: Path,
              events: Path, player: Any, capture_mod: Any, progress: Progress) -> None:
        deadline = time.monotonic() + p.duration_hours * 3600.0
        total = p.cycles
        self._set_host_gain(ctx, p, data, "start")
        while True:
            if ctx.cancelled.is_set():
                data["cancelled"] = True
                break
            if not _audio_ready(_refresh(ctx)):
                ctx.log("audio card vanished (device reset?) — re-entering the audio profile")
                data["mode_reentries"] += 1
                if not self._ensure_audio(ctx, p, data):
                    data["failures"].append("cannot recover the audio profile")
                    break
                self._set_host_gain(ctx, p, data, f"re-entry {data['mode_reentries']}")

            verdict, reasons, metrics = self._cycle(ctx, p, ref_path, player, capture_mod,
                                                    data["min_rms_dbfs"])
            data["cycles"] += 1
            data[verdict] += 1
            record = {"ts": time.strftime("%F %T"), "cycle": data["cycles"],
                      "verdict": verdict, "reasons": reasons, "metrics": metrics}
            data["records"].append(record)
            _append_jsonl(events, record)
            ctx.log(f"cycle {data['cycles']}: {verdict.upper()}{_brief(metrics)}"
                    + (f" — {'; '.join(reasons)}" if reasons else ""))
            progress(data["cycles"], total,
                     f"cycle {data['cycles']}: {verdict}")

            if p.duration_hours == 0 or time.monotonic() >= deadline:
                break
            if p.cycles is not None and data["cycles"] >= p.cycles:
                break
            # Cancellation ends the run at the cycle boundary and still reports what
            # the cycles said, which is what Ctrl-C did to the old script.
            if ctx.cancelled.wait(p.interval):
                data["cancelled"] = True
                break

    def _cycle(self, ctx: Context, p: Params, ref_path: Path, player: Any,
               capture_mod: Any, min_rms_dbfs: float = MULTITONE_MIN_RMS_DBFS
               ) -> tuple[str, list[str], dict[str, Any] | None]:
        """One take: record, play the reference into it, judge the return."""
        wav = ctx.workdir / "cycle_rec.wav"
        session = capture_mod.CaptureSession(
            ctx.device, seconds=p.tone_seconds + CAPTURE_PAD_S, preset=p.preset, out=wav,
            # The scenario owns the profile across every cycle: letting the session
            # restore it would re-enumerate the board between takes. The RT mixer
            # stays parked: the host is the source here, and a resumed mixer writes
            # both I2S TX channels every block alongside the UAC bridge -- the
            # return came back 1.8 % flat with a glitch every DMA block.
            keep_mode=True, resume_audio_tasks=False, log=ctx.log)
        try:
            session.start()
        except HilError as e:
            return "fail", [f"capture did not start: {e.message}"], None
        try:
            ctx.cancelled.wait(PLAY_DELAY_S)
            player.play(ctx.device, ref_path, p.tone_seconds + PLAY_TIMEOUT_PAD_S)
            session.wait()
        except HilError as e:
            if e.code == ENV:
                # The host could not play or record at all: that is the bench,
                # and one such cycle is the whole answer. Counting it as a
                # failed cycle produced "227 cycles / 227 FAIL" on a board that
                # was fine.
                raise
            return "fail", [e.message], None
        finally:
            try:
                result = session.stop()
            except HilError:
                result = None
        why = _audio.host_loss({}, getattr(result, "health", None))
        if why:
            # Holes the host made: not this cycle's verdict on the firmware.
            return "host_lost", [why], None
        try:
            out = _deps.analyze("multitone", wav, expected={
                "freqs": [float(f) for f in p.freqs],
                "warn_glitch_per_s": p.warn_glitch_per_s,
                "min_rms_dbfs": min_rms_dbfs})
        except HilError as e:
            return "fail", [f"analysis failed: {e.message}"], None
        verdict = str(out.get("verdict", "fail"))
        if verdict not in ("pass", "warn", "fail"):
            verdict = "fail"
        return verdict, list(out.get("reasons", [])), out

    def _verdict(self, data: dict[str, Any], artifacts: list[Artifact]) -> Report:
        cycles, failed = data["cycles"], data["fail"]
        lost = data.get("host_lost", 0)
        body = (f"{cycles} cycles — {data['pass']} PASS / {data['warn']} WARN / "
                f"{failed} FAIL" + (f" / {lost} not measured (host lost capture)" if lost else "")
                + f", {data['mode_reentries']} mode re-entries")
        drift = (data.get("host_gain_final") or {}).get("deviations") or []
        level_fails = sum(1 for r in data["records"] if r["verdict"] == "fail"
                          and any("level too low" in x for x in r.get("reasons") or []))
        if failed and drift and level_fails:
            # The session put its stored volume back during the run: the low
            # level is the host's, and saying FAIL would blame the firmware.
            return _env(f"audio_loopback: host gain {', '.join(drift)} at the end of the run "
                        f"— {level_fails} 'level too low' cycle(s) are the host's, not the "
                        f"firmware's ({body})", data, artifacts)
        if lost and not failed:
            return _audio.not_measured("audio_loopback", f"{lost} cycle(s) the host lost "
                                       f"capture of ({body})", data, artifacts)
        if cycles and not failed:
            return Report(passed=True, summary=f"audio_loopback: PASS — {body}",
                          data=data, artifacts=artifacts, exit_code=0)
        extra = "; ".join(data["failures"])
        return Report(passed=False,
                      summary=f"audio_loopback: FAIL — {body}" + (f" ({extra})" if extra else ""),
                      data=data, artifacts=artifacts, exit_code=1)


def _open_cdc(ctx: Context) -> Any:
    """CDC for the INGAIN reads (tests replace this)."""
    return ctx.open_cdc()


def _card(device: Device) -> str | None:
    """The ALSA card id of the board's UAC2 card, as amixer -c takes it."""
    return device.ports.uac2.alsa_id if device.ports.uac2 is not None else None


def _brief(metrics: dict[str, Any] | None) -> str:
    """The one-line cycle summary the old script logged."""
    if not metrics or not metrics.get("signal_present"):
        return ""
    offs = ",".join(f"{t['offset_pct']:+.1f}%" for t in metrics.get("tones", {}).values())
    return (f" rms={metrics.get('rms_dbfs')}dBFS "
            f"glitch/s={metrics.get('glitch_bursts_per_s')} offs=[{offs}]")


def _append_jsonl(path: Path, record: dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, default=str) + "\n")


SCENARIO = register(AudioLoopbackScenario())


def main(argv: list[str] | None = None) -> int:
    """Entry point for the platform-idf shim ``tools/hil_audio_loopback.py``."""
    import sys as _sys

    from crosspad_hil.cli import main as cli_main

    args = list(_sys.argv[1:] if argv is None else argv)
    return cli_main(["run", "audio_loopback", *args])
