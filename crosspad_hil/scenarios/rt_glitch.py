"""Does the LED/pad storm put clicks in what the RT mixer renders?

Port of ``platform-idf/tools/hil_rt_glitch.py`` — the LED-coalescing baseline,
and the one experiment here that needs nothing but the CrossPad's own cable.

The device is put into the MIDI+UAC2 profile with the capture endpoint on the
DAC→ADC loop, and one take is recorded across three phases:

* **A silence** — nothing is touched, so the noise floor is on record;
* **B sustain** — a single pad held down: voices render, LED traffic is minimal;
* **C hammer** — a pad storm, which is heavy LED/RMT traffic on top of the same
  voices.

B is the control for C. If C has clicks that B does not, something on the LED
path is stealing time from the RT mixer, which is the condition LED coalescing
exists for. Equal click rates = the baseline is clean.

Two things are load-bearing and both are why a naive version of this records
silence. The UAC2 profile **parks the RT mixer** (it fights the UAC bridge for
I2S), so ``0x1D 06 01`` has to resume it — :class:`~crosspad_hil.capture.CaptureSession`
does that. And the recorded path has to be the **DAC→ADC loop** (``headphone``
= codec1 LINE1), not the mics, or what comes back is the room.

Pads are hit over SysEx ``0x1D 07/08`` on the STM bridge, which is the physical
pad path: pad logic, sample playback and LEDs. A MIDI NoteOn would light the
LEDs and never reach pad logic, so it would test the wrong half.

A click count means nothing on a take the pads were not heard in, so the
verdict first insists the capture is not ``silent`` and that both the held pad
(B) and the storm (C) stand ``MIN_PHASE_ABOVE_FLOOR_DB`` over phase A's floor.
2026-10-09: a Sampler left in "16 LEVELS" by an earlier UI-driving scenario
played pad 0 at level 1 of 16, the take was silent, and the run said CLEAN. The
prologue therefore switches the Sampler's performance modes off first.

The same day, a Sampler mixer channel muted (and saved) by a UI walk, and a codec
output volume turned down to 28 by the volume overlay, did the same. The prologue
therefore reads the mixer and codec state (``audio_state``): a muted Sampler channel
(``MIX_MUTE``), one with no send (``MIX_ROUTE``) or a codec volume under 40
(``AUDIO_VOL``) is put right, logged and recorded in ``data["audio_state_fixed"]``.
Only a fix the board refuses (a firmware without the verb) or one that does not
stick makes the run MISCONFIGURED rather than measure a board it knows is mute.

Exit code 0 = CLEAN, 1 = clicks under LED load, 2 = not measured: the prologue
never got the Sampler playing, the board is MISCONFIGURED (a fix refused or not
sticking), the
capture endpoint never appeared, or the take is SILENT.
"""
from __future__ import annotations

import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from crosspad_hil import audio_state, card_hold, verbs
from crosspad_hil.analyze import click_metric, db, read_wav
from crosspad_hil.capture import CaptureSession
from crosspad_hil.devices import UsbMode
from crosspad_hil.errors import CANCELLED, TIMEOUT, HilError
from crosspad_hil.knowledge import load as load_knowledge
from crosspad_hil.midi import MidiIO, MidiRole, sysex_audio_route
from crosspad_hil.scenarios import _audio
from crosspad_hil.scenarios.base import Artifact, Context, Progress, Report, register
from crosspad_hil.snapshot import ref_to_delta, take_snapshot
from crosspad_hil.stim import open_transport
from crosspad_hil.usbmode import set_mode

SAMPLER_APP = "Sampler"
# --- timings, every one the old script's literal --------------------------------
KIT_SETTLE_S = 6.0          # hil_rt_glitch: sleep(6) — kit load + waveform top-up
APP_START_SETTLE_S = 3.0    # sleep(3) after APP_START Sampler
RETRY_SETTLE_S = 8.0        # sleep(8) before the second attempt at the prologue
PROLOGUE_ATTEMPTS = 2       # `for attempt in (1, 2)`
ROUTE_SETTLE_S = 1.0        # sleep(1) after the route + resume SysEx
SYSEX_GAP_S = 0.06          # capture.py's spacing between route frames
CAPTURE_TAIL_S = 2.0        # total_s = silence + sustain + hammer + 2
CAPTURE_PRESET = "headphone"  # codec1 LINE1 — the same pair as SYSEX_ROUTE_LOOP
HAMMER_VELOCITY = 110       # pad_press(pad) default vel
# Phases B (held pad) and C (storm) must stand this far over phase A's floor, or
# the take recorded something other than the pads and its click counts say
# nothing. A real kit at HAMMER_VELOCITY is tens of dB over the floor; a pad at
# level 1 of 16, or a capture not on the DAC loop, is not.
MIN_PHASE_ABOVE_FLOOR_DB = 20.0
PHASE_C_PADS = 4            # phase C cycles pad..pad+3
PAD_TRANSPORT = "sysex_stm"  # the physical-pad path, not MIDI NoteOn

# Analysis window trims, from hil_rt_glitch's `bounds` dict. B starts half a
# second in because the attack of the held pad is not a glitch.
TRIM_S = 0.2
SUSTAIN_LEAD_S = 0.5
# The verdict rule: C is allowed the clicks B already had, plus this much.
CLEAN_MARGIN_PER_S = 0.1
# The Sampler's performance switches (one button on the panel that cycles
# Off -> 16 LEVELS (velocity) -> 16 PITCH -> Off). The label reads
# "16 LEVELS: PAD n" / "16 PITCH: PAD n" while a mode is on, plain "16 LEVELS"
# while off. The mode lives in RAM: it survives an app restart, not a reboot.
PERF_BUTTON_RE = re.compile(r"\b16\s+(LEVELS|PITCH)\b(?:\s*:\s*PAD\s*(\d+))?", re.IGNORECASE)
PERF_PRESS_LIMIT = 3        # Off -> Velocity -> Pitch -> Off: three presses always get to off
UI_SETTLE_S = 0.4           # after an encoder turn or press, before the next snapshot
ENC_ROTATE_MAX = 16         # the firmware clamps ENC_ROTATE here
ENGINE_POLL_S = 0.25        # SMPL_ENGINE poll while the native take runs (state=1)


def level_problem(capture_silent: bool, a_db: float, b_db: float, c_db: float) -> str | None:
    """Why the take cannot be scored (it is silent / the pads are not above the floor), or None.

    A: the noise floor, B: one held pad, C: the pad storm. Every phase the pads
    play in has to clear A by ``MIN_PHASE_ABOVE_FLOOR_DB``; the capture's own
    ``silent`` flag (nothing on the endpoint at all) is enough on its own.
    """
    why = []
    if capture_silent:
        why.append("the capture reports silence")
    for name, level in (("B (sustain)", b_db), ("C (hammer)", c_db)):
        if level < a_db + MIN_PHASE_ABOVE_FLOOR_DB:
            why.append(f"{name} at {level} dBFS is not {MIN_PHASE_ABOVE_FLOOR_DB:.0f} dB over "
                       f"the A floor of {a_db} dBFS")
    return "; ".join(why) or None


def perf_button(ui: dict[str, Any] | None) -> dict[str, Any] | None:
    """The Sampler's performance button in a snapshot's ``ui`` section, or None.

    The entry is the group item plus ``on`` (the label carries ": PAD n") and
    ``mode`` (LEVELS | PITCH). An already-on button wins if there are several.
    """
    found: list[dict[str, Any]] = []
    for item in (ui or {}).get("group") or []:
        m = PERF_BUTTON_RE.search(str(item.get("label") or ""))
        if m:
            found.append({**item, "on": m.group(2) is not None, "mode": m.group(1).upper()})
    return next((b for b in found if b["on"]), found[0] if found else None)


def _sub(name: str) -> int:
    return int(load_knowledge("sysex")["audio_route"]["subs"][name])


@dataclass
class Params:
    kit: int = field(default=0, metadata={"help": "kit id to load (KIT_LIST names them)"})
    pad: int = field(
        default=0, metadata={"help": "base pad index; phase C cycles pad..pad+3"})
    silence_s: float = field(default=5.0, metadata={"help": "phase A: nothing playing"})
    sustain_s: float = field(default=10.0, metadata={"help": "phase B: one pad held down"})
    hammer_s: float = field(default=15.0, metadata={"help": "phase C: the pad storm"})
    hammer_rate: float = field(default=12.0, metadata={"help": "pad hits per second in phase C"})
    keep_mode: bool = field(
        default=False, metadata={"help": "leave the board in the MIDI+UAC2 profile afterwards"})
    exclusive_card: bool = field(
        default=True,
        metadata={"help": "pause the audio-nodes service for the run (bench holder only), so "
                          "the capture opens the card directly instead of through PipeWire"})


def _wait(ctx: Context, seconds: float) -> None:
    """Sleep that returns early on cancel and then raises CANCELLED."""
    if seconds > 0:
        ctx.cancelled.wait(seconds)
    ctx.check_cancelled()


def _env(summary: str, data: dict[str, Any], artifacts: list[Artifact]) -> Report:
    return Report(passed=False, summary=summary, data=data, artifacts=artifacts, exit_code=2)


def _phase(x: Any, sr: int, start_s: float, end_s: float) -> dict[str, Any]:
    """Click count, click rate and level of one slice of the take."""
    seg = x[int(max(0.0, start_s) * sr):int(max(0.0, end_s) * sr)]
    dur = max(end_s - start_s, 1e-3)
    if len(seg) == 0:
        return {"rms_dbfs": -120.0, "clicks": 0, "clicks_per_s": 0.0,
                "first_click_offsets_s": [], "start_s": round(start_s, 3),
                "end_s": round(end_s, 3), "seconds": 0.0}
    clicks, positions = click_metric(seg, sr)
    rms = db(float((seg ** 2).mean() ** 0.5))
    return {"rms_dbfs": round(rms, 1), "clicks": clicks,
            "clicks_per_s": round(clicks / dur, 2), "first_click_offsets_s": positions,
            "start_s": round(start_s, 3), "end_s": round(end_s, 3),
            "seconds": round(len(seg) / float(sr), 3)}


class RtGlitchScenario:
    name = "rt_glitch"
    Params = Params
    description = "phased UAC2 capture: clicks under pad/LED load that a held pad does not have"

    def run(self, ctx: Context, params: Params, progress: Progress) -> Report:
        try:
            p = params
            data: dict[str, Any] = {
                "kit": p.kit, "pad": p.pad, "preset": CAPTURE_PRESET,
                "phases": {}, "verdict": None, "capture": None, "capture_health": None,
                "hits": 0, "prologue_attempts": 0, "failures": [],
            }
            artifacts: list[Artifact] = []
            ctx.check_cancelled()

            # The board has to be in the default profile to take CDC commands at
            # all; set_mode waits for the CDC node instead of the old script's
            # blind sleep(5).
            if ctx.device.usb_mode is UsbMode.AUDIO:
                ctx.log(f"{ctx.device.id} is in the MIDI+UAC2 profile — restoring default first")
                ctx.device = set_mode(ctx.device, UsbMode.DEFAULT)

            progress(0, 4, "loading the kit and starting the Sampler")
            env = self._prologue(ctx, p, data)
            if env is not None:
                return _env(env, data, artifacts)

            total_s = p.silence_s + p.sustain_s + p.hammer_s + CAPTURE_TAIL_S
            with card_hold.for_run(ctx, data, enabled=p.exclusive_card, expected_s=total_s):
                return self._take(ctx, p, progress, data, artifacts)

        finally:
            ctx.close_all()

    def _take(self, ctx: Context, p: Params, progress: Progress, data: dict[str, Any],
              artifacts: list[Artifact]) -> Report:
        """Record A/B/C in the audio profile and score the take."""
        progress(1, 4, "entering the MIDI+UAC2 profile and recording")
        total_s = p.silence_s + p.sustain_s + p.hammer_s + CAPTURE_TAIL_S
        wav = ctx.workdir / "capture.wav"
        session = CaptureSession(
            ctx.device, seconds=total_s, preset=CAPTURE_PRESET, out=wav,
            resume_audio_tasks=True, keep_mode=p.keep_mode, log=ctx.log)
        try:
            data["capture"] = session.start()
        except HilError as e:
            # No capture endpoint, no take: the bench is wrong, not the
            # firmware (hil_rt_glitch returned 2 here too).
            session.close()
            return _env(f"rt_glitch: could not start the capture — {e.code}: {e.message}",
                        data, artifacts)
        ctx.device = session.device
        artifacts.append(Artifact(path=str(wav), mime="audio/wav", role="capture"))
        try:
            _wait(ctx, ROUTE_SETTLE_S)
            marks = self._phases(ctx, p, progress, data)
            # The recording is duration-bounded; let it run itself out so the
            # tail of phase C is on the take.
            session.wait(timeout_s=total_s + 10.0)
        finally:
            # Restore in the old script's order — mixer suspended, routing
            # back to defaults, then the USB profile, which stop() owns —
            # and do it on the way out of a cancelled run too.
            self._restore_route(ctx)
            result = session.stop()
            data["capture"] = {**(data["capture"] or {}), **result.to_dict()}
            data["capture_health"] = getattr(result, "health", None)
            ctx.device = session.device

        why = _audio.host_loss(data, data["capture_health"])
        if why:
            return _audio.not_measured("rt_glitch", why, data, artifacts)

        progress(3, 4, "measuring clicks per phase")
        return self._verdict(ctx, p, marks, wav, data, artifacts)

    # --- prologue -------------------------------------------------------------
    def _prologue(self, ctx: Context, p: Params, data: dict[str, Any]) -> str | None:
        """Kit + Sampler over CDC. Returns an environment message, or None on success.

        The retry is not defensive dressing: a kit load intermittently
        crash-reboots the device during sample preload (a known null-deref), and
        the old script gave the prologue exactly one more go before giving up.
        """
        link = ctx.open_cdc()
        try:
            for attempt in range(1, PROLOGUE_ATTEMPTS + 1):
                ctx.check_cancelled()
                data["prologue_attempts"] = attempt
                try:
                    verbs.kit_load(link, p.kit, wait_s=KIT_SETTLE_S)
                except HilError as e:
                    if e.code != TIMEOUT:
                        return (f"rt_glitch: KIT_LOAD {p.kit} refused ({e.code}: {e.message}) "
                                "— old firmware without hil_control?")
                    ctx.log(f"KIT_LOAD {p.kit}: {e.message}")
                else:
                    try:
                        verbs.app_start(link, SAMPLER_APP, wait_s=APP_START_SETTLE_S)
                    except HilError as e:
                        ctx.log(f"APP_START {SAMPLER_APP}: {e.code}: {e.message}")
                    else:
                        why = self._engine_holds(ctx, link, p.kit, data)
                        if why is None:
                            self._performance_off(ctx, link, data)
                            return self._audio_state(ctx, link, data)
                        ctx.log(why)
                if attempt < PROLOGUE_ATTEMPTS:
                    ctx.log(f"Sampler not running (attempt {attempt}) — the device likely "
                            "crash-rebooted during the kit load; retrying")
                    _wait(ctx, RETRY_SETTLE_S)
            return f"rt_glitch: {SAMPLER_APP} not running after {PROLOGUE_ATTEMPTS} attempts"
        finally:
            # CDC disappears the moment the UAC2 profile is entered; a link left
            # open would hold the lock on a node that no longer exists.
            link.close()

    def _engine_holds(self, ctx: Context, link: Any, kit: int,
                      data: dict[str, Any]) -> str | None:
        """None when the native engine holds ``kit`` (SMPL_ENGINE ``kit=``); else why not.

        KIT_STATUS is the loader's word; the engine takes the kit afterwards on a
        task of its own, and a Sampler with no kit in it renders nothing for the
        pads to be measured by. A firmware without ``kit=`` (or without the verb)
        is not judged.
        """
        try:
            deadline = time.monotonic() + max(KIT_SETTLE_S, 1.0)
            eng = verbs.smpl_engine(link)
            while eng.get("state") == 1 and time.monotonic() < deadline:
                _wait(ctx, ENGINE_POLL_S)
                eng = verbs.smpl_engine(link)
        except HilError as e:
            data["engine_kit"] = {"unsupported": True, "error": e.code}
            return None
        data["engine_kit"] = {k: eng.get(k) for k in ("engine", "state", "why", "kit")}
        if "kit" not in eng or eng.get("kit") == kit:
            return None
        return (f"rt_glitch: the native engine holds kit {eng.get('kit')} "
                f"(state={eng.get('state')} why={eng.get('why')}), not kit {kit}: "
                f"the pads would not sound")

    # --- mixer / codec state -----------------------------------------------------
    def _audio_state(self, ctx: Context, link: Any, data: dict[str, Any]) -> str | None:
        """Is a pad audible on this board at all; None when it is (or was made so).

        A UI-driving scenario can leave the Sampler's mixer channel muted or without a
        send, or a codec output volume turned right down (28 of 100 after the volume
        overlay), all of it surviving a reboot. The take is then silence, and so is
        every run until someone finds it. The mute (``MIX_MUTE``), the sends and the
        volumes are put right (logged, ``data["audio_state_fixed"]``); only a fix the
        board refuses or one that does not stick makes the run MISCONFIGURED instead of
        measuring a board it knows is mute. A board that will not say (firmware without
        MIX_LVL) is measured as before.
        """
        data["audio_state_fixed"] = []
        data["audio_state_unread"] = []
        try:
            res = audio_state.ensure_audible(link, ctx.log)
        except HilError as e:
            if e.code == CANCELLED:
                raise
            ctx.log(f"WARNING: could not check the audio state: {e.code}: {e.message}")
            data["audio_state_unread"] = [f"{e.code}: {e.message}"]
            return None
        data["audio_state_fixed"] = res["fixed"]
        data["audio_state_unread"] = res["unread"]
        if res["unread"]:
            ctx.log("audio state not fully readable (" + ", ".join(res["unread"])
                    + ") -- measuring anyway")
        if not res["unfixable"]:
            return None
        data["verdict"] = "MISCONFIGURED"
        data["audio_state_unfixable"] = res["unfixable"]
        data["failures"].extend(res["unfixable"])
        return ("rt_glitch: MISCONFIGURED -- the board cannot be heard and the verbs cannot fix "
                "it: " + "; ".join(res["unfixable"]))

    # --- Sampler performance modes ---------------------------------------------
    def _ui(self, ctx: Context, link: Any) -> dict[str, Any] | None:
        """The UI section of a snapshot, or None when the verbs did not answer."""
        ui = take_snapshot(ctx.device, link, include=("ui",)).ui
        return None if not ui or "error" in ui else ui

    def _focus_button(self, ctx: Context, link: Any) -> dict[str, Any] | None:
        """Rotate until the performance button holds the focus; its snapshot, or None.

        ``ui focus eN`` is a guess: the group lists widgets the encoder skips, so
        the turn count from the indices can land elsewhere. First turn by the
        index distance, then one detent at a time, and believe only a snapshot
        whose focus is the button.
        """
        for step in range(64):
            ui = self._ui(ctx, link)
            btn = perf_button(ui)
            if ui is None or btn is None:
                return None
            focus = ui.get("focus") or {}
            if focus.get("ref") == btn["ref"]:
                return ui
            delta = 1
            if step == 0 and focus.get("ref") is not None:
                delta = ref_to_delta(ui["group"], int(focus["index"]), btn["ref"])
            delta = max(-ENC_ROTATE_MAX, min(ENC_ROTATE_MAX, delta)) or 1
            verbs.enc_rotate(link, delta)
            _wait(ctx, UI_SETTLE_S)
            if step >= len(ui["group"]) + 2:
                break
        return None

    def _performance_off(self, ctx: Context, link: Any, data: dict[str, Any]) -> None:
        """Switch the Sampler's 16 LEVELS / 16 PITCH off; never fails the run.

        A pad in 16 LEVELS plays level (pad+1) of 16 -- pad 0 is near silent --
        and the mode outlives the app. The button is not found -> warn and go on
        (the verdict still refuses a take it cannot hear).
        """
        info: dict[str, Any] = {"found": False, "presses": 0}
        data["performance_mode"] = info
        try:
            btn = perf_button(self._ui(ctx, link))
            if btn is None:
                ctx.log("WARNING: no '16 LEVELS' / '16 PITCH' button on the Sampler panel — "
                        "performance modes not checked")
                return
            info.update(found=True, before=btn["label"])
            if not btn["on"]:
                info["after"] = btn["label"]
                ctx.log(f"Sampler performance modes already off ('{btn['label']}')")
                return
            while btn["on"] and info["presses"] < PERF_PRESS_LIMIT:
                if self._focus_button(ctx, link) is None:
                    ctx.log("WARNING: could not focus the performance button — leaving it")
                    break
                verbs.enc_press(link)
                info["presses"] += 1
                _wait(ctx, UI_SETTLE_S)
                btn = perf_button(self._ui(ctx, link))
                if btn is None:
                    ctx.log("WARNING: the performance button vanished after a press")
                    return
            info["after"] = btn["label"]
            if btn["on"]:
                ctx.log(f"WARNING: Sampler performance mode still on after {info['presses']} "
                        f"press(es): '{btn['label']}'")
            else:
                ctx.log(f"Sampler performance mode was on ('{info['before']}'): pressed "
                        f"{info['presses']}x, now '{btn['label']}'")
        except HilError as e:
            if e.code == CANCELLED:
                raise
            ctx.log(f"WARNING: could not check the Sampler performance modes: "
                    f"{e.code}: {e.message}")

    # --- stimulation ----------------------------------------------------------
    def _phases(self, ctx: Context, p: Params, progress: Progress,
                data: dict[str, Any]) -> dict[str, float]:
        """Run A/B/C against the running capture; return the phase start marks."""
        pads = open_transport(ctx.device, PAD_TRANSPORT)
        t0 = time.monotonic()
        hits = 0
        try:
            ctx.log(f"phase A: silence {p.silence_s}s")
            progress(2, 4, "phase A: silence")
            _wait(ctx, p.silence_s)

            ctx.log(f"phase B: sustain {p.sustain_s}s (pad {p.pad})")
            progress(2, 4, "phase B: sustain")
            t_b = time.monotonic()
            pads.press(p.pad, HAMMER_VELOCITY)
            hits += 1
            try:
                _wait(ctx, p.sustain_s)
            finally:
                # A pad left held would keep a voice open across phase C.
                pads.release(p.pad)

            ctx.log(f"phase C: hammering {p.hammer_s}s @ {p.hammer_rate}/s")
            progress(2, 4, "phase C: hammering")
            t_c = time.monotonic()
            period = 1.0 / max(0.1, p.hammer_rate)
            i = 0
            while time.monotonic() - t_c < p.hammer_s:
                pad = p.pad + (i % PHASE_C_PADS)
                pads.press(pad, HAMMER_VELOCITY)
                hits += 1
                _wait(ctx, period / 2)
                pads.release(pad)
                _wait(ctx, period / 2)
                i += 1
            t_end = time.monotonic()
        finally:
            data["hits"] = hits
            pads.close()
        return {"t0": t0, "b": t_b - t0, "c": t_c - t0, "end": t_end - t0}

    # --- restore --------------------------------------------------------------
    def _restore_route(self, ctx: Context) -> None:
        """Suspend the mixer and put the codec routing back the way it boots.

        ``CaptureSession`` restores the USB profile but not the routing, and the
        old script did restore it — a board left recording its own DAC loop is a
        confusing thing to pick up next.
        """
        io = MidiIO(ctx.device, MidiRole.ESP)
        try:
            with io:
                for frame in (sysex_audio_route(_sub("audio_tasks"), 0),
                              sysex_audio_route(_sub("adc_input"), 1, 0),
                              sysex_audio_route(_sub("mic_src"), 0),
                              sysex_audio_route(_sub("dac_output"), 1, 3)):
                    io.send_sysex(frame)
                    time.sleep(SYSEX_GAP_S)
        except HilError as e:  # restoring must not mask the measurement
            ctx.log(f"could not restore the default audio route: {e.code}: {e.message}")

    # --- verdict --------------------------------------------------------------
    def _verdict(self, ctx: Context, p: Params, marks: dict[str, float], wav: Path,
                 data: dict[str, Any], artifacts: list[Artifact]) -> Report:
        try:
            # Both channels averaged: the loop feeds the same signal to each, and
            # averaging halves the converter noise the click metric sits on.
            frames, sr = read_wav(wav, channel=None)
        except HilError as e:
            return _env(f"rt_glitch: nothing to analyse — {e.code}: {e.message}", data, artifacts)
        x = frames.mean(axis=1)
        bounds = {
            "A_silence": (TRIM_S, p.silence_s - TRIM_S),
            "B_sustain": (marks["b"] + SUSTAIN_LEAD_S, marks["b"] + p.sustain_s - TRIM_S),
            "C_hammer": (marks["c"] + TRIM_S, marks["end"] - TRIM_S),
        }
        for name, (a, b) in bounds.items():
            phase = _phase(x, sr, a, b)
            data["phases"][name] = phase
            ctx.log(f"{name}: rms={phase['rms_dbfs']} dBFS clicks={phase['clicks']} "
                    f"({phase['clicks_per_s']}/s)")

        a_db = float(data["phases"]["A_silence"]["rms_dbfs"])
        b_db = float(data["phases"]["B_sustain"]["rms_dbfs"])
        c_db = float(data["phases"]["C_hammer"]["rms_dbfs"])
        silent = bool((data.get("capture") or {}).get("silent"))
        why = level_problem(silent, a_db, b_db, c_db)
        if why:
            data["verdict"] = "SILENT"
            data["failures"].append(why)
            hint = (data.get("capture") or {}).get("hint")
            return _env(f"rt_glitch: SILENT — {why}: no kit in the engine, a Sampler performance "
                        "mode (16 LEVELS) left on, or the capture is not on the DAC loop"
                        + (f" ({hint})" if hint else ""), data, artifacts)

        c_rate = float(data["phases"]["C_hammer"]["clicks_per_s"])
        b_rate = float(data["phases"]["B_sustain"]["clicks_per_s"])
        clean = c_rate <= b_rate + CLEAN_MARGIN_PER_S
        data["verdict"] = "CLEAN" if clean else "CLICKS-UNDER-LED-LOAD"
        summary = (f"rt_glitch: {data['verdict']} — sustain {b_rate}/s, "
                   f"hammer {c_rate}/s over {data['hits']} pad hits")
        if not clean:
            data["failures"].append(
                f"the hammer phase clicks {c_rate}/s against the sustain phase's {b_rate}/s: "
                "the LED path is interfering with the RT mixer")
        return Report(passed=clean, summary=summary, data=data, artifacts=artifacts,
                      exit_code=0 if clean else 1)


SCENARIO = register(RtGlitchScenario())


def main(argv: list[str] | None = None) -> int:
    """Entry point for the platform-idf shim `tools/hil_rt_glitch.py`."""
    import sys

    from crosspad_hil.cli import main as cli_main

    args = list(sys.argv[1:] if argv is None else argv)
    return cli_main(["run", "rt_glitch", *args])
