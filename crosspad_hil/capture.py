"""UAC2 capture: record what the board is actually playing (spec §3.2 `crosspad_capture`).

Ported from platform-idf tools: hil_sampler_record.py (find_capture / API_RANK /
Recorder / PRESETS / the route prologue), hil_audio_loopback.py (ensure_audio_mode,
arecord fallback) and hil_velocity.py (the codec0 measurement tap).

Two requirements here are not obvious and both make the difference between a take
and three seconds of digital silence:

* **The USB-audio profile parks the RT mixer** (it fights the UAC bridge for I2S), so
  ``0x1D 06 01`` has to resume it after the mode switch or the pads are silent on the
  capture endpoint. Resume it **only when the pads are the source**: with the host
  streaming (``audio_loopback``) a resumed mixer writes the same I2S TX channels
  block by block and the return comes back 1.8 % flat with a glitch every DMA
  block -- ``resume_audio_tasks=False`` there.
* **What you record is the DAC→ADC loop, not the mics.** ``headphone`` = codec1 LINE1,
  near unity with the better S/N; ``mics`` (codec0 LINE2) hears the room, which is not
  what the sample engine put on the bus.

Backends are injectable: ``SoundDeviceBackend`` (cross-platform, per-callback health)
first, ``ArecordBackend`` (Linux, alsa-utils) as the fallback the spec asks for, and a
fake in the tests — nothing here needs audio hardware to be unit-tested.
"""
from __future__ import annotations

import math
import os
import re
import shutil
import subprocess
import sys
import threading
import time
import wave
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Protocol

from .audio_nodes import MIC_NODE
from .devices import Device, UsbMode, discover, is_uac2_name, parse_arecord_list
from .errors import BAD_ARGS, ENV, NOT_SUPPORTED, TIMEOUT, HilError
from .knowledge import load as load_knowledge
from .midi import MidiIO, MidiRole, query_route, sysex_audio_route
from .usbmode import set_mode

SR_DEFAULT = 44100    # the engine rate the firmware ships with (kits are 44.1 kHz);
                      # the card's own default wins when the backend reports one
CHANNELS_DEFAULT = 2
MAX_CAPTURE_S = 900.0           # spec §3.7: cap_* handles live 15 min at most
SILENT_PEAK_DBFS = -60.0        # below this nothing played; the mode/route prologue failed
AUDIO_EXTRA_HINT = "pip install 'crosspad-hil[audio]' (Linux also needs libportaudio2)"

# capture codec, that codec's ADC input (hil_sampler_record.py PRESETS).
# LINE1 on either codec is the DAC->ADC loop; codec1 is near unity and has ~12 dB
# better S/N than codec0 (measured), which is why it is the default.
# mic_src value for "both codecs, summed" (audio_route_control.h AUDIO_ROUTE_CAPTURE_MIX)
CAPTURE_MIX = 2
# mic_src value for the pads-only mixer bus "USB rec" (AUDIO_ROUTE_CAPTURE_SAMPLER):
# digital, no DAC/ADC in the path, so a frame of the take is a frame of the render.
# No ADC input to set (None): nothing analogue is recorded.
CAPTURE_SAMPLER = 3
PRESETS: dict[str, tuple[int, int | None]] = {
    "headphone": (1, 1),
    "speaker": (0, 1),
    "mics": (0, 2),
    "line-in": (1, 2),
    "mix": (CAPTURE_MIX, 2),   # both codecs' jacks summed on the board
    "sampler": (CAPTURE_SAMPLER, None),   # the mixer's recording bus, sample-exact
}
DEFAULT_PRESET = "headphone"
# Record whatever the board routes to the capture endpoint, unchanged: what a
# phone on the USB-C port would receive with the product's own routing.
AS_IS = "asis"

# MME and DirectSound cap at 44.1 kHz and would resample a 48 kHz UAC2 endpoint for
# nothing (hil_sampler_record.py API_RANK). Anything unlisted ranks between the two.
API_RANK = {"Windows WASAPI": 0, "ALSA": 0, "Core Audio": 0,
            "Windows WDM-KS": 1, "Windows DirectSound": 3, "MME": 4}


# --------------------------------------------------------------------------- model


@dataclass(frozen=True)
class CaptureDevice:
    """The UAC2 capture endpoint as one backend sees it."""

    name: str
    api: str
    index: int | None = None        # sounddevice device index
    alsa_id: str | None = None      # `arecord -l` card id, e.g. "Audio"
    sample_rate: int = SR_DEFAULT
    channels: int = CHANNELS_DEFAULT

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class CaptureResult:
    wav: str
    seconds: float
    peak_dbfs: float
    rms_dbfs: float
    overruns: int
    silent: bool
    sample_rate: int
    channels: int
    frames: int
    device: str
    preset: str
    backend: str
    health: dict = field(default_factory=dict)
    hint: str | None = None

    def to_dict(self) -> dict:
        return asdict(self)


class Recording(Protocol):
    """A capture in flight. ``stop()`` finalizes the WAV; ``health()`` explains it."""

    path: Path

    def stop(self, timeout_s: float = 10.0) -> None: ...

    def wait(self, timeout_s: float | None = None) -> bool: ...

    def health(self) -> dict: ...


class AudioBackend(Protocol):
    name: str

    def find_capture(self, device: Device, *, timeout_s: float = 15.0) -> CaptureDevice: ...

    def start(self, cap: CaptureDevice, path: Path, *, seconds: float | None,
              sample_rate: int, channels: int) -> Recording: ...


# --------------------------------------------------------------------------- sounddevice


def _sounddevice() -> Any:
    try:
        import sounddevice  # type: ignore[import-not-found]
    except Exception as exc:  # noqa: BLE001 - missing PortAudio raises OSError, not ImportError
        raise HilError(NOT_SUPPORTED, f"sounddevice is unavailable: {exc}",
                       hint=AUDIO_EXTRA_HINT) from exc
    return sounddevice


class _SoundDeviceRecording:
    """Streams straight to the WAV file, so a long take costs no RAM.

    The gap between callbacks is tracked because a capture that simply stops looks
    identical to a quiet recording once it is on disk (hil_sampler_record.py Recorder).
    """

    def __init__(self, sd: Any, cap: CaptureDevice, path: Path, *, seconds: float | None,
                 sample_rate: int, channels: int) -> None:
        self.path = path
        self._sd = sd
        self._seconds = seconds
        self._sample_rate = sample_rate
        self._channels = channels
        self.overruns = 0
        self.frames_in = 0
        self.max_gap = 0.0
        self.status_flags = 0
        self.exclusive = False
        self._t0: float | None = None
        self._last_cb = 0.0
        self._wall = 0.0
        self._was_active = False
        self._done = threading.Event()
        self._stopped = False
        self._wav = wave.open(str(path), "wb")
        self._wav.setnchannels(channels)
        self._wav.setsampwidth(2)
        self._wav.setframerate(sample_rate)
        common = {"samplerate": sample_rate, "channels": channels, "dtype": "int16",
                  "device": cap.index, "callback": self._cb, "latency": "low"}
        try:
            # WASAPI's shared path runs the stream through the Windows capture mixer,
            # whose endpoint volume sits between the device and the file.
            self._stream = sd.InputStream(extra_settings=sd.WasapiSettings(exclusive=True),
                                          **common)
            self.exclusive = True
        except Exception:  # noqa: BLE001 - no WASAPI here, or exclusive refused
            self._stream = sd.InputStream(**common)

    def _cb(self, indata: Any, frames: int, time_info: Any, status: Any) -> None:
        now = time.perf_counter()
        if status:
            if getattr(status, "input_overflow", False):
                self.overruns += 1
            self.status_flags |= int(status)
        if self._t0 is None:
            self._t0 = now
        else:
            self.max_gap = max(self.max_gap, now - self._last_cb)
        self._last_cb = now
        self.frames_in += frames
        self._wav.writeframes(bytes(indata))
        if self._seconds is not None and self.frames_in >= self._seconds * self._sample_rate:
            self._done.set()

    def start(self) -> None:
        self._stream.start()
        deadline = time.perf_counter() + 5.0
        while self._t0 is None and time.perf_counter() < deadline:
            time.sleep(0.002)

    def wait(self, timeout_s: float | None = None) -> bool:
        if self._seconds is None:
            return False
        return self._done.wait(timeout_s if timeout_s is not None else self._seconds + 5.0)

    def stop(self, timeout_s: float = 10.0) -> None:
        if self._stopped:
            return
        self._stopped = True
        if self._seconds is not None:
            self._done.wait(min(timeout_s, self._seconds + 5.0))
        self._was_active = bool(getattr(self._stream, "active", False))
        self._wall = (time.perf_counter() - self._t0) if self._t0 else 0.0
        try:
            self._stream.stop()
            self._stream.close()
        finally:
            self._wav.close()

    def health(self) -> dict:
        """Captured seconds against the wall clock is the honest check: a stalled
        stream keeps its file valid and its peak respectable, and only the missing
        seconds give it away."""
        got = self.frames_in / float(self._sample_rate)
        missed = max(0.0, self._wall - got)
        flags: list[str] = []
        if self._stopped and not self._was_active:
            flags.append("stream went inactive")
        if self.overruns:
            flags.append(f"{self.overruns} overruns")
        if self.max_gap > 0.5:
            flags.append(f"{self.max_gap * 1000:.0f} ms callback gap")
        if missed > 0.25:
            flags.append(f"{missed:.1f} s never arrived")
        return {"ok": not flags, "captured_s": round(got, 3), "wall_s": round(self._wall, 3),
                "overruns": self.overruns, "max_gap_ms": round(self.max_gap * 1000, 1),
                "missed_s": round(missed, 3), "exclusive": self.exclusive,
                "flags": flags,
                "detail": f"captured {got:.1f} s of {self._wall:.1f} s"
                          + (" - " + ", ".join(flags) if flags else " - clean")}


class SoundDeviceBackend:
    """Primary backend: cross-platform, and the only one that can report overruns."""

    name = "sounddevice"

    def __init__(self, sd: Any | None = None,
                 sleep: Callable[[float], None] = time.sleep,
                 clock: Callable[[], float] = time.monotonic) -> None:
        self._sd = sd
        self._sleep = sleep
        self._clock = clock

    def _mod(self) -> Any:
        if self._sd is None:
            self._sd = _sounddevice()
        return self._sd

    def find_capture(self, device: Device, *, timeout_s: float = 15.0) -> CaptureDevice:
        """The endpoint registers with each host API in its own time after the mode
        switch, so a first sighting on a lesser API is not the final answer."""
        sd = self._mod()
        deadline = self._clock() + timeout_s
        settle: float | None = None
        while True:
            try:
                sd._terminate()
                sd._initialize()
            except Exception:  # noqa: BLE001 - re-scan is best effort
                pass
            best: tuple[int, int, dict, str] | None = None
            for idx, d in enumerate(sd.query_devices()):
                if not is_uac2_name(str(d["name"])) or int(d.get("max_input_channels", 0)) < 2:
                    continue
                api = str(sd.query_hostapis(d["hostapi"])["name"])
                rank = API_RANK.get(api, 2)
                if best is None or rank < best[0]:
                    best = (rank, idx, dict(d), api)
            if best is not None and (best[0] == 0 or (settle is not None
                                                      and self._clock() >= settle)):
                _rank, idx, info, api = best
                return CaptureDevice(name=str(info["name"]), api=api, index=idx,
                                     sample_rate=int(info.get("default_samplerate")
                                                     or SR_DEFAULT),
                                     channels=min(2, int(info["max_input_channels"])))
            if best is not None and settle is None:
                settle = self._clock() + 2.0
            elif best is None and self._clock() >= deadline:
                raise HilError(TIMEOUT, "no CrossPad UAC2 capture endpoint appeared "
                                        f"within {timeout_s:.0f} s",
                               hint="usb-mode set audio, then `crosspad-hil devices`",
                               device=device.id)
            self._sleep(0.5)

    def start(self, cap: CaptureDevice, path: Path, *, seconds: float | None,
              sample_rate: int, channels: int) -> Recording:
        rec = _SoundDeviceRecording(self._mod(), cap, path, seconds=seconds,
                                    sample_rate=sample_rate, channels=channels)
        rec.start()
        return rec


# --------------------------------------------------------------------------- arecord


def _run_text(argv: list[str], timeout: float = 5.0) -> str:
    """stdout of ``argv`` under LC_ALL=C — `arecord -l` is localized, its parser is not."""
    try:
        return subprocess.run(argv, capture_output=True, text=True, timeout=timeout,
                              env={**os.environ, "LC_ALL": "C"}).stdout
    except (OSError, subprocess.SubprocessError, ValueError):
        return ""


# ALSA's PipeWire plugin: the way in when the card itself is taken.
PIPEWIRE_DEVICE = "pipewire"


def pipewire_target_env(node: str) -> dict[str, str]:
    """The environment that points ALSA's PipeWire plugin at `node`. Without it
    the plugin opens the session default -- the desktop's own microphone or
    speakers, not the CrossPad."""
    return {"PIPEWIRE_NODE": node}


class _ArecordRecording:
    def __init__(self, argv: list[str], path: Path, *, seconds: float | None,
                 sample_rate: int, channels: int, device_arg: str = "",
                 env: dict[str, str] | None = None) -> None:
        self.path = path
        self.device_arg = device_arg
        self._seconds = seconds
        self._sample_rate = sample_rate
        self._channels = channels
        self._t0 = time.perf_counter()
        # The WAV's t=0, on the clock the scenarios schedule hits with. Taken
        # here and not after busy_now(): that waits 0.7 s with arecord already
        # recording, and a session stamped after it put every event 0.7 s late.
        self.started_at = time.monotonic()
        self._wall = 0.0
        self._stopped = False
        self._hung = False
        self._rc: int | None = None
        self._stderr = ""
        try:
            self._proc = subprocess.Popen(argv, stdout=subprocess.DEVNULL,
                                          stderr=subprocess.PIPE,
                                          env={**os.environ, "LC_ALL": "C", **(env or {})})
        except OSError as exc:
            raise HilError(ENV, f"could not start arecord: {exc}",
                           hint="install alsa-utils, or " + AUDIO_EXTRA_HINT) from exc

    def busy_now(self, wait_s: float = 0.7) -> bool:
        """True when arecord already gave up because something else owns the card.

        PipeWire holds the UAC2 capture whenever the CrossPad DAW nodes are up
        (`crosspad-hil audio nodes run`), and an exclusive plughw: open then
        fails instantly. Without this the scenario surfaces it as "cannot read
        <file>.wav", which says nothing about the cause.
        """
        try:
            self._proc.wait(wait_s)
        except subprocess.TimeoutExpired:
            return False
        self._rc = self._proc.returncode
        if self._proc.stderr is not None:
            self._stderr = self._proc.stderr.read().decode(errors="replace").strip()
        self._stopped = True
        return self._rc != 0 and "busy" in self._stderr.lower()


    def wait(self, timeout_s: float | None = None) -> bool:
        if self._seconds is None:
            return False
        try:
            self._proc.wait(timeout_s if timeout_s is not None else self._seconds + 5.0)
            return True
        except subprocess.TimeoutExpired:
            return False

    def stop(self, timeout_s: float = 10.0) -> None:
        if self._stopped:
            return
        self._stopped = True
        if self._seconds is not None:
            try:
                self._proc.wait(min(timeout_s, self._seconds + 5.0))
            except subprocess.TimeoutExpired:
                pass
        if self._proc.poll() is None:
            # arecord finalizes the WAV header on SIGINT/SIGTERM; SIGKILL leaves it lying.
            self._proc.terminate()
            try:
                self._proc.wait(3.0)
            except subprocess.TimeoutExpired:
                self._proc.kill()
                try:
                    self._proc.wait(2.0)
                except subprocess.TimeoutExpired:
                    # Not even SIGKILL: arecord is stuck in the kernel, on a USB
                    # device that stopped answering. Leave it; its stderr would
                    # block a read forever.
                    self._hung = True
        self._wall = time.perf_counter() - self._t0
        self._rc = self._proc.returncode
        if self._proc.stderr is not None and not self._hung:
            self._stderr = self._proc.stderr.read().decode(errors="replace").strip()

    def health(self) -> dict:
        got = 0.0
        try:
            with wave.open(str(self.path), "rb") as w:
                got = w.getnframes() / float(w.getframerate() or self._sample_rate)
        except (OSError, wave.Error):
            pass
        missed = max(0.0, self._wall - got)
        faults: list[str] = []
        if self._hung:
            faults.append("arecord did not exit even on SIGKILL (the card stopped answering)")
        # arecord returns non-zero when it is signalled, which is how we stop it.
        if self._rc not in (0, None, -2, -15) and self._stderr:
            faults.append(f"arecord: {self._stderr}")
        if missed > 0.5:
            faults.append(f"{missed:.1f} s never arrived")
        notes: list[str] = []
        if self.device_arg == PIPEWIRE_DEVICE:
            # Not a fault, so it must not decide `ok`: the take went through
            # PipeWire's graph rather than straight off the card, so anything
            # comparing engine rates or measuring glitches is measuring that
            # graph too. It stays in the detail for whoever reads the numbers.
            notes.append("recorded through PipeWire (the card was busy)")
        flags = faults + notes
        return {"ok": not faults, "captured_s": round(got, 3), "wall_s": round(self._wall, 3),
                "overruns": 0, "max_gap_ms": None, "missed_s": round(missed, 3),
                "returncode": self._rc, "recorder_hung": self._hung, "flags": flags,
                "detail": f"captured {got:.1f} s of {self._wall:.1f} s"
                          + (" - " + ", ".join(flags) if flags else " - clean")}


class ArecordBackend:
    """Linux fallback (spec §2.1). No overrun counter, but no PortAudio either."""

    name = "arecord"

    def __init__(self, run_text: Callable[[list[str]], str] = _run_text,
                 sleep: Callable[[float], None] = time.sleep,
                 clock: Callable[[], float] = time.monotonic) -> None:
        self._run = run_text
        self._sleep = sleep
        self._clock = clock

    def _card_rate(self, card: int) -> int:
        """The one rate the card offers, from ``/proc/asound/cardN/stream0``.

        ``plughw:`` would happily resample a 48 kHz card to whatever ``-r`` says
        and hand back a WAV that looks native -- an A/B between engine rates
        recorded through that was comparing ALSA's linear resampler, not the
        board. Falls back to SR_DEFAULT when the proc file cannot be read.
        """
        try:
            text = self._run(["cat", f"/proc/asound/card{card}/stream0"])
        except Exception:  # noqa: BLE001 - any failure means "unknown", not a bench error
            return SR_DEFAULT
        capture = text.split("Capture:", 1)[1] if "Capture:" in text else text
        m = re.search(r"Rates:\s*(\d+)", capture)
        return int(m.group(1)) if m else SR_DEFAULT

    def find_capture(self, device: Device, *, timeout_s: float = 15.0) -> CaptureDevice:
        deadline = self._clock() + timeout_s
        while True:
            cards = parse_arecord_list(self._run(["arecord", "-l"]))
            if device.ports.uac2 is not None and device.ports.uac2.alsa_id:
                card = next((c for c, aid, _h in cards if aid == device.ports.uac2.alsa_id), None)
                rate = self._card_rate(card) if card is not None else SR_DEFAULT
                return CaptureDevice(name=device.ports.uac2.name, api="ALSA",
                                     alsa_id=device.ports.uac2.alsa_id, sample_rate=rate)
            for card, alsa_id, human in cards:
                if is_uac2_name(human):
                    return CaptureDevice(name=human, api="ALSA", alsa_id=alsa_id,
                                         sample_rate=self._card_rate(card))
            if self._clock() >= deadline:
                raise HilError(TIMEOUT, "no CrossPad UAC2 capture card in `arecord -l` "
                                        f"within {timeout_s:.0f} s",
                               hint="usb-mode set audio, then `crosspad-hil devices`",
                               device=device.id)
            self._sleep(0.5)

    def start(self, cap: CaptureDevice, path: Path, *, seconds: float | None,
              sample_rate: int, channels: int) -> Recording:
        def argv_for(device_arg: str) -> list[str]:
            argv = ["arecord", "-q", "-D", device_arg, "-f", "S16_LE",
                    "-c", str(channels), "-r", str(sample_rate), str(path)]
            if seconds is not None:
                # `-d` is a hard stop, so it is the requested length rounded up and no
                # more: a spare second on the end is a second of the take that nobody
                # asked for and that every measurement then averages over.
                argv[1:1] = ["-d", str(max(1, math.ceil(seconds)))]
            return argv

        def recording(device_arg: str, env: dict[str, str] | None = None) -> _ArecordRecording:
            return _ArecordRecording(argv_for(device_arg), path, seconds=seconds,
                                     sample_rate=sample_rate, channels=channels,
                                     device_arg=device_arg, env=env)

        rec = recording(f"plughw:{cap.alsa_id}")
        if not rec.busy_now():
            return rec
        # Something owns the card exclusively — on this bench that is PipeWire,
        # holding the CrossPad DAW nodes. Recording through PipeWire lets both
        # coexist instead of making the DAW setup and the HIL suite an either/or.
        return recording(PIPEWIRE_DEVICE, pipewire_target_env(MIC_NODE))


def default_backend() -> AudioBackend:
    """sounddevice when it imports, arecord when alsa-utils is there, else NOT_SUPPORTED."""
    try:
        return SoundDeviceBackend(_sounddevice())
    except HilError:
        if sys.platform.startswith("linux") and shutil.which("arecord"):
            return ArecordBackend()
        raise


# --------------------------------------------------------------------------- session


def _sub(name: str) -> int:
    return int(load_knowledge("sysex")["audio_route"]["subs"][name])


class CaptureSession:
    """One `cap_*` capture: mode switch, route prologue, recording, restore.

    ``start()`` is the whole non-obvious part — see the module docstring. ``stop()``
    finalizes the WAV, measures it and puts the device back the way it was found.
    """

    def __init__(self, device: Device, *, seconds: float | None = None,
                 preset: str = DEFAULT_PRESET, out: str | Path | None = None,
                 resume_audio_tasks: bool = True, keep_mode: bool = False,
                 sample_rate: int | None = None, channels: int = CHANNELS_DEFAULT,
                 volume: int | None = None, backend: AudioBackend | None = None,
                 midi_factory: Callable[..., Any] = MidiIO,
                 set_mode_fn: Callable[..., Device] = set_mode,
                 discover_fn: Callable[[], list[Device]] = discover,
                 logdir: str | Path = "hil_logs",
                 silent_peak_dbfs: float = SILENT_PEAK_DBFS,
                 find_timeout_s: float = 15.0,
                 sleep: Callable[[float], None] = time.sleep,
                 log: Callable[[str], None] | None = None) -> None:
        if preset not in PRESETS and preset != AS_IS:
            raise HilError(BAD_ARGS, f"unknown capture preset {preset!r}",
                           hint="one of: " + ", ".join(sorted(PRESETS)) + f", {AS_IS}")
        if seconds is not None:
            if seconds <= 0:
                raise HilError(BAD_ARGS, f"seconds must be > 0, got {seconds!r}")
            if seconds > MAX_CAPTURE_S:
                raise HilError(BAD_ARGS,
                               f"seconds must be <= {MAX_CAPTURE_S:.0f} (cap_* handle TTL)",
                               hint="record several takes instead of one long one")
        if volume is not None and not 0 <= volume <= 100:
            raise HilError(BAD_ARGS, f"volume must be 0..100, got {volume!r}")
        self.device = device
        self.seconds = seconds
        self.preset = preset
        self.resume_audio_tasks = resume_audio_tasks
        self.keep_mode = keep_mode
        self.channels = channels
        self.volume = volume
        self._sample_rate = sample_rate
        self._backend = backend
        self._midi_factory = midi_factory
        self._set_mode_fn = set_mode_fn
        self._discover_fn = discover_fn
        self._silent_peak = silent_peak_dbfs
        self._find_timeout = find_timeout_s
        self._sleep = sleep
        self._log = log or (lambda _m: None)
        self._out = Path(out) if out is not None else None
        self._logdir = Path(logdir)
        self._entered_audio = False
        self._route_home: dict | None = None
        self._rec: Recording | None = None
        self._cap: CaptureDevice | None = None
        self._result: CaptureResult | None = None
        self.started_at: float | None = None

    # -- helpers -------------------------------------------------------

    @property
    def running(self) -> bool:
        return self._rec is not None and self._result is None

    @property
    def wav_path(self) -> Path:
        if self._out is not None:
            return self._out
        stamp = time.strftime("%Y%m%d_%H%M%S")
        return self._logdir / f"capture_{stamp}" / "capture.wav"

    def _apply_route(self, dev: Device) -> None:
        if self.preset == AS_IS:
            self._read_route_as_is(dev)
            return
        codec, adc = PRESETS[self.preset]
        io = self._midi_factory(dev, MidiRole.ESP)
        io.open()
        try:
            # The capture preset *persists* on the device, so a capture run used
            # to leave the board pointing wherever it had aimed it — and two of
            # the presets aim at a DAC->ADC bus that rails on codec 1. Nothing
            # here notices; the next audio step in the same gate fails instead,
            # on hardware that is fine. Remember what to put back.
            if self._route_home is None:
                try:
                    # ms in practice; a bench that will not answer must not cost a second.
                    self._route_home = query_route(io, timeout_s=0.6)
                except HilError as e:
                    self._log(f"could not read the capture route to restore later: {e}")
            if self.resume_audio_tasks:
                # Audio mode parks the RT mixer; without this the pads are silent.
                io.send_sysex(sysex_audio_route(_sub("audio_tasks"), 1))
                self._sleep(0.2)
            io.send_sysex(sysex_audio_route(_sub("mic_src"), codec))
            self._sleep(0.06)
            # The mix reads both codecs, so both ADCs must sit on the jack. The
            # sampler bus is digital: no ADC is involved, none is moved.
            for c in (() if adc is None else (0, 1) if codec == CAPTURE_MIX else (codec,)):
                io.send_sysex(sysex_audio_route(_sub("adc_input"), c, adc))
                self._sleep(0.06)
            if self.volume is not None and codec in (0, 1):
                io.send_sysex(sysex_audio_route(_sub("volume"), codec, self.volume))
                self._sleep(0.06)
        finally:
            io.close()

    def _read_route_as_is(self, dev: Device) -> None:
        """Read the route for the report and send nothing: nothing to restore."""
        io = self._midi_factory(dev, MidiRole.ESP)
        io.open()
        try:
            self.route_seen = query_route(io, timeout_s=0.6)
        except HilError as e:
            self.route_seen = None
            self._log(f"could not read the capture route: {e}")
        finally:
            io.close()

    # -- lifecycle -----------------------------------------------------

    def start(self) -> dict:
        if self._rec is not None:
            raise HilError(BAD_ARGS, "capture already started",
                           hint="stop() this session before starting another")
        dev = self.device
        if dev.usb_mode is not UsbMode.AUDIO:
            self._log(f"switching {dev.id} into the MIDI+UAC2 profile")
            dev = self._set_mode_fn(dev, UsbMode.AUDIO, discover_fn=self._discover_fn,
                                    midi_factory=self._midi_factory)
            self._entered_audio = True
            self.device = dev
        try:
            self._apply_route(dev)

            backend = self._backend or default_backend()
            self._backend = backend
            self._cap = backend.find_capture(dev, timeout_s=self._find_timeout)
            rate = self._sample_rate or self._cap.sample_rate or SR_DEFAULT
            channels = min(self.channels, self._cap.channels or self.channels)
            path = self.wav_path
            path.parent.mkdir(parents=True, exist_ok=True)
        except BaseException:
            # The route persists on the board, and a session that never started
            # never reaches stop(): put the route (and the profile) back now.
            self._restore()
            raise
        try:
            self._rec = backend.start(self._cap, path, seconds=self.seconds,
                                      sample_rate=rate, channels=channels)
        except HilError:
            self._restore()
            raise
        except Exception as exc:  # noqa: BLE001 - a failed stream must still restore the mode
            self._restore()
            raise HilError(ENV, f"could not open the capture stream at {rate} Hz: {exc}",
                           hint=f"the endpoint publishes {self._cap.sample_rate} Hz") from exc
        self._sample_rate = rate
        self.channels = channels
        self.started_at = getattr(self._rec, "started_at", None) or time.monotonic()
        codec, adc = PRESETS.get(self.preset, (None, None))
        return {"device": dev.id, "wav": str(path), "preset": self.preset,
                "route_seen": getattr(self, "route_seen", None),
                "codec": codec, "adc_input": adc, "backend": backend.name,
                "capture_device": self._cap.to_dict(), "sample_rate": rate,
                "channels": channels, "seconds": self.seconds,
                "audio_mode_entered": self._entered_audio,
                "audio_tasks_resumed": self.resume_audio_tasks}

    def health(self) -> dict:
        if self._rec is None:
            raise HilError(BAD_ARGS, "capture not started", hint="capture.start first")
        h = dict(self._rec.health())
        h["running"] = self.running
        h["elapsed_s"] = round(time.monotonic() - self.started_at, 3) if self.started_at else 0.0
        return h

    def wait(self, timeout_s: float | None = None) -> bool:
        """True once a duration-bounded capture has finished on its own."""
        if self._rec is None:
            raise HilError(BAD_ARGS, "capture not started", hint="capture.start first")
        return self._rec.wait(timeout_s)

    def stop(self) -> CaptureResult:
        if self._rec is None:
            raise HilError(BAD_ARGS, "capture not started", hint="capture.start first")
        if self._result is not None:
            return self._result
        try:
            self._rec.stop()
        finally:
            self._restore()
        health = self._rec.health()
        from .analyze import wav_stats  # local: analysis needs numpy, capture does not

        stats = wav_stats(self._rec.path)
        silent = stats["peak_dbfs"] < self._silent_peak
        hint = None
        if silent:
            hint = ("nothing on the capture endpoint: check that the RT mixer was resumed "
                    "(SysEx 0x1D 06 01) and that the preset is the DAC->ADC loop "
                    "(headphone = codec1 LINE1), not the mics")
        self._result = CaptureResult(
            wav=str(self._rec.path), seconds=stats["seconds"], peak_dbfs=stats["peak_dbfs"],
            rms_dbfs=stats["rms_dbfs"], overruns=int(health.get("overruns") or 0),
            silent=silent, sample_rate=stats["sample_rate"], channels=stats["channels"],
            frames=stats["frames"], device=self.device.id, preset=self.preset,
            backend=getattr(self._backend, "name", "?"), health=health, hint=hint)
        return self._result

    def _restore_route(self) -> None:
        """Put back the route _apply_route changed, before the profile goes back.

        MIDI is reachable in both profiles, but the port this session already
        used is the audio-mode one, so do it here rather than after the switch.
        """
        home = self._route_home
        if home is None:
            return
        self._route_home = None
        io = self._midi_factory(self.device, MidiRole.ESP)
        try:
            io.open()
        except Exception as exc:  # noqa: BLE001 - restoring must not mask the result
            self._log(f"could not reopen MIDI to restore the capture route: {exc}")
            return
        try:
            # Inputs before the source, the order audio_route_apply_preset uses.
            for codec, value in enumerate(home.get("adc", [])[:2]):
                io.send_sysex(sysex_audio_route(_sub("adc_input"), codec, value))
                self._sleep(0.06)
            io.send_sysex(sysex_audio_route(_sub("mic_src"), home["mic_src"]))
            self._sleep(0.06)
            # Volume too, and only when this session moved it: a take that turned
            # a codec down to stop a DAC->ADC loop clipping must not leave the
            # speakers quiet for whoever picks the board up next.
            if self.volume is not None:
                for codec, value in enumerate(home.get("vol", [])[:2]):
                    io.send_sysex(sysex_audio_route(_sub("volume"), codec, value))
                    self._sleep(0.06)
        except Exception as exc:  # noqa: BLE001
            self._log(f"could not restore the capture route: {exc}")
        finally:
            io.close()

    def _restore(self) -> None:
        self._restore_route()
        if self._entered_audio and not self.keep_mode:
            self._entered_audio = False
            try:
                self.device = self._set_mode_fn(self.device, UsbMode.DEFAULT,
                                                discover_fn=self._discover_fn,
                                                midi_factory=self._midi_factory)
            except HilError as e:            # restoring must not mask the capture result
                self._log(f"could not restore the default USB profile: {e}")

    def close(self) -> None:
        """Handle-registry teardown: stop if still running, always leave the mode sane."""
        if self.running:
            try:
                self.stop()
                return
            except HilError:
                pass
        self._restore()


def capture(device: Device, *, seconds: float = 3.0, **kw: Any) -> CaptureResult:
    """Record `seconds` and return the measured result — the CLI's one-shot path."""
    session = CaptureSession(device, seconds=seconds, **kw)
    session.start()
    try:
        session.wait()
    finally:
        result = session.stop()
    return result
