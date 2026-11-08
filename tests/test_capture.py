"""Capture sessions with a faked audio backend — no PortAudio, no board.

What is actually being pinned down here is the prologue and the epilogue, because
those are what make the difference between a take and three seconds of silence: the
RT mixer has to be resumed after the USB profile switch, the recorded path has to be
the DAC→ADC loop rather than the mics, and the device has to end up back in the
default profile whatever happened in between.
"""
from __future__ import annotations

import sys
import wave
from pathlib import Path

import numpy as np
import pytest

from crosspad_hil.capture import (
    PRESETS,
    ArecordBackend,
    CaptureDevice,
    CaptureSession,
    SoundDeviceBackend,
    capture,
    default_backend,
)
from crosspad_hil.devices import (
    AudioCardInfo,
    Device,
    MidiPortInfo,
    Ports,
    SerialPortInfo,
    UsbMode,
)
from crosspad_hil.errors import BAD_ARGS, ENV, NOT_SUPPORTED, TIMEOUT, HilError
from crosspad_hil.midi import MidiIO
from tests.fakes import FakeMidiBus

SR = 48000
CDC = SerialPortInfo(path="/dev/ttyACM0", vid=0x303A, pid=0x3456, serial="ABCD1234",
                     product="Crosspad", location="1-2:1.0")
UAC = AudioCardInfo(name="Crosspad Audio", sounddevice_index=3, alsa_id="Audio")
MIDI = MidiPortInfo(name="Crosspad MIDI 1", rtmidi_out=0, rtmidi_in=0, alsa_hw=None, rawmidi=None)


def dev_default() -> Device:
    return Device(id="dev_1234", serial="ABCD1234", usb_mode=UsbMode.DEFAULT,
                  ports=Ports(cdc=CDC, esp_midi=MIDI))


def dev_audio() -> Device:
    return Device(id="dev_1234", serial="ABCD1234", usb_mode=UsbMode.AUDIO,
                  ports=Ports(uac2=UAC, esp_midi=MIDI))


def midi_factory_for(bus: FakeMidiBus):
    def factory(device: Device, role=None, **kw) -> MidiIO:
        return MidiIO(device, role or "esp", out_factory=bus.out_factory,
                      in_factory=bus.in_factory)
    return factory


class FakeSetMode:
    """Stands in for usbmode.set_mode; records the modes it was asked for."""

    def __init__(self) -> None:
        self.calls: list[UsbMode] = []

    def __call__(self, device: Device, mode: UsbMode, **kw) -> Device:
        self.calls.append(mode)
        return dev_audio() if mode is UsbMode.AUDIO else dev_default()


class FakeRecording:
    """Writes `signal` to the WAV the moment it is stopped, like a real finalize."""

    def __init__(self, path: Path, signal: np.ndarray, sample_rate: int, channels: int,
                 health: dict | None = None) -> None:
        self.path = path
        self._signal = signal
        self._sr = sample_rate
        self._channels = channels
        self._health = health or {}
        self.stopped = False

    def _write(self) -> None:
        data = self._signal
        if data.ndim == 1:
            data = np.column_stack([data] * self._channels)
        pcm = (np.clip(data, -1.0, 0.999969) * 32768.0).astype("<i2")
        with wave.open(str(self.path), "wb") as w:
            w.setnchannels(self._channels)
            w.setsampwidth(2)
            w.setframerate(self._sr)
            w.writeframes(pcm.tobytes())

    def stop(self, timeout_s: float = 10.0) -> None:
        if not self.stopped:
            self.stopped = True
            self._write()

    def wait(self, timeout_s: float | None = None) -> bool:
        return True

    def health(self) -> dict:
        return {"ok": True, "overruns": 0, "captured_s": len(self._signal) / self._sr,
                "flags": [], **self._health}


class FakeBackend:
    name = "fake"

    def __init__(self, signal: np.ndarray | None = None, *, health: dict | None = None,
                 cap: CaptureDevice | None = None, fail: Exception | None = None) -> None:
        t = np.arange(int(2.0 * SR)) / SR
        self.signal = signal if signal is not None else 0.4 * np.sin(2 * np.pi * 440 * t)
        self.health = health
        self.cap = cap or CaptureDevice(name="Crosspad Audio", api="ALSA", index=3,
                                        alsa_id="Audio", sample_rate=SR, channels=2)
        self.fail = fail
        self.started: list[dict] = []
        self.recordings: list[FakeRecording] = []

    def find_capture(self, device: Device, *, timeout_s: float = 15.0) -> CaptureDevice:
        return self.cap

    def start(self, cap: CaptureDevice, path: Path, *, seconds, sample_rate, channels):
        if self.fail is not None:
            raise self.fail
        self.started.append({"cap": cap, "path": path, "seconds": seconds,
                             "sample_rate": sample_rate, "channels": channels})
        rec = FakeRecording(path, self.signal, sample_rate, channels, self.health)
        self.recordings.append(rec)
        return rec


def session(device: Device, bus: FakeMidiBus, backend: FakeBackend, set_mode: FakeSetMode,
            tmp_path: Path, **kw) -> CaptureSession:
    return CaptureSession(device, backend=backend, midi_factory=midi_factory_for(bus),
                          set_mode_fn=set_mode, discover_fn=lambda: [device],
                          logdir=tmp_path, sleep=lambda _s: None,
                          out=kw.pop("out", tmp_path / "capture.wav"), **kw)


def sysex_frames(bus: FakeMidiBus) -> list[list[int]]:
    return [m for m in bus.sent if m and m[0] == 0xF0]


def route_frames(bus: FakeMidiBus) -> list[list[int]]:
    """Audio-route frames that change something, in order.

    The query (sub 0x10) is filtered out: a session reads the route before it
    changes anything so it can put it back, and that read must not make every
    positional assertion in here shift by one.
    """
    return [m for m in sysex_frames(bus) if len(m) > 3 and m[2] == 0x1D and m[3] != 0x10]


# --------------------------------------------------------------------------- prologue


def test_start_resumes_the_rt_mixer_then_routes_the_dac_adc_loop(tmp_path: Path) -> None:
    """The two requirements the spec calls out, in the order they have to happen."""
    bus, backend, set_mode = FakeMidiBus(), FakeBackend(), FakeSetMode()
    s = session(dev_default(), bus, backend, set_mode, tmp_path, seconds=2.0)
    info = s.start()
    frames = route_frames(bus)
    # F0 7D 1D <sub> <args…> F7 — 06 01 resumes the audio tasks the UAC bridge parked.
    assert frames[0] == [0xF0, 0x7D, 0x1D, 0x06, 0x01, 0xF7]
    assert frames[1] == [0xF0, 0x7D, 0x1D, 0x02, 0x01, 0xF7]        # mic_src = codec1
    assert frames[2] == [0xF0, 0x7D, 0x1D, 0x01, 0x01, 0x01, 0xF7]  # codec1 ADC = LINE1
    # …and the route was read before any of it, so stop() can put it back.
    assert sysex_frames(bus)[0][:4] == [0xF0, 0x7D, 0x1D, 0x10]
    assert info["preset"] == "headphone" and (info["codec"], info["adc_input"]) == (1, 1)
    assert info["audio_tasks_resumed"] is True and info["audio_mode_entered"] is True
    assert set_mode.calls == [UsbMode.AUDIO]
    s.stop()


def test_headphone_is_the_default_preset_and_mics_is_not() -> None:
    assert PRESETS["headphone"] == (1, 1)          # codec1 LINE1: the DAC->ADC loop
    assert PRESETS["mics"] == (0, 2)               # codec0 LINE2 hears the room instead


def test_preset_selects_the_codec_and_input(tmp_path: Path) -> None:
    bus, backend, set_mode = FakeMidiBus(), FakeBackend(), FakeSetMode()
    s = session(dev_audio(), bus, backend, set_mode, tmp_path, seconds=1.0, preset="mics")
    info = s.start()
    frames = route_frames(bus)
    assert frames[1] == [0xF0, 0x7D, 0x1D, 0x02, 0x00, 0xF7]        # mic_src = codec0
    assert frames[2] == [0xF0, 0x7D, 0x1D, 0x01, 0x00, 0x02, 0xF7]  # codec0 ADC = LINE2
    assert (info["codec"], info["adc_input"]) == (0, 2)
    s.stop()


def test_no_mode_switch_when_the_board_is_already_in_audio_mode(tmp_path: Path) -> None:
    bus, backend, set_mode = FakeMidiBus(), FakeBackend(), FakeSetMode()
    s = session(dev_audio(), bus, backend, set_mode, tmp_path, seconds=1.0)
    assert s.start()["audio_mode_entered"] is False
    s.stop()
    assert set_mode.calls == []          # entered nothing, so restore nothing


def test_resume_audio_tasks_can_be_turned_off(tmp_path: Path) -> None:
    bus, backend, set_mode = FakeMidiBus(), FakeBackend(), FakeSetMode()
    s = session(dev_audio(), bus, backend, set_mode, tmp_path, seconds=1.0,
                resume_audio_tasks=False)
    s.start()
    assert all(f[3] != 0x06 for f in sysex_frames(bus))
    s.stop()


def test_volume_is_sent_when_asked(tmp_path: Path) -> None:
    bus, backend, set_mode = FakeMidiBus(), FakeBackend(), FakeSetMode()
    s = session(dev_audio(), bus, backend, set_mode, tmp_path, seconds=1.0, volume=80)
    s.start()
    assert [0xF0, 0x7D, 0x1D, 0x04, 0x01, 80, 0xF7] in sysex_frames(bus)
    s.stop()


# --------------------------------------------------------------------------- result


def test_stop_measures_the_wav_and_restores_the_default_profile(tmp_path: Path) -> None:
    bus, backend, set_mode = FakeMidiBus(), FakeBackend(), FakeSetMode()
    s = session(dev_default(), bus, backend, set_mode, tmp_path, seconds=2.0)
    s.start()
    r = s.stop().to_dict()
    assert Path(r["wav"]).is_file() and r["wav"].endswith("capture.wav")
    assert r["seconds"] == pytest.approx(2.0, abs=0.05)
    assert r["peak_dbfs"] == pytest.approx(-8.0, abs=0.5)      # 0.4 FS
    assert r["rms_dbfs"] < r["peak_dbfs"]
    assert r["silent"] is False and r["hint"] is None
    assert r["overruns"] == 0 and r["sample_rate"] == SR and r["channels"] == 2
    assert r["device"] == "dev_1234" and r["preset"] == "headphone" and r["backend"] == "fake"
    assert set_mode.calls == [UsbMode.AUDIO, UsbMode.DEFAULT]


def test_silence_is_reported_with_the_reason_it_is_usually_silent(tmp_path: Path) -> None:
    quiet = np.zeros(int(1.0 * SR))
    bus, backend, set_mode = FakeMidiBus(), FakeBackend(quiet), FakeSetMode()
    s = session(dev_audio(), bus, backend, set_mode, tmp_path, seconds=1.0)
    s.start()
    r = s.stop().to_dict()
    assert r["silent"] is True and r["peak_dbfs"] == -120.0
    assert "0x1D 06 01" in r["hint"] and "headphone" in r["hint"]


def test_stop_is_idempotent_and_keeps_the_first_result(tmp_path: Path) -> None:
    bus, backend, set_mode = FakeMidiBus(), FakeBackend(), FakeSetMode()
    s = session(dev_audio(), bus, backend, set_mode, tmp_path, seconds=1.0)
    s.start()
    first = s.stop()
    assert s.stop() is first
    assert backend.recordings[0].stopped is True


def test_health_explains_a_stalled_stream(tmp_path: Path) -> None:
    bus, set_mode = FakeMidiBus(), FakeSetMode()
    backend = FakeBackend(health={"ok": False, "overruns": 3, "missed_s": 1.4,
                                  "flags": ["1.4 s never arrived"]})
    s = session(dev_audio(), bus, backend, set_mode, tmp_path, seconds=2.0)
    s.start()
    h = s.health()
    assert h["running"] is True and h["overruns"] == 3 and h["ok"] is False
    r = s.stop().to_dict()
    assert r["overruns"] == 3 and r["health"]["flags"] == ["1.4 s never arrived"]


def test_keep_mode_leaves_the_board_in_audio(tmp_path: Path) -> None:
    bus, backend, set_mode = FakeMidiBus(), FakeBackend(), FakeSetMode()
    s = session(dev_default(), bus, backend, set_mode, tmp_path, seconds=1.0, keep_mode=True)
    s.start()
    s.stop()
    assert set_mode.calls == [UsbMode.AUDIO]


def test_close_stops_a_running_capture(tmp_path: Path) -> None:
    bus, backend, set_mode = FakeMidiBus(), FakeBackend(), FakeSetMode()
    s = session(dev_default(), bus, backend, set_mode, tmp_path, seconds=1.0)
    s.start()
    s.close()
    assert backend.recordings[0].stopped is True
    assert set_mode.calls[-1] is UsbMode.DEFAULT


def test_a_failed_stream_still_restores_the_profile(tmp_path: Path) -> None:
    bus, set_mode = FakeMidiBus(), FakeSetMode()
    backend = FakeBackend(fail=OSError("Invalid sample rate"))
    s = session(dev_default(), bus, backend, set_mode, tmp_path, seconds=1.0)
    with pytest.raises(HilError) as e:
        s.start()
    assert e.value.code == ENV and "sample rate" in e.value.message
    assert set_mode.calls == [UsbMode.AUDIO, UsbMode.DEFAULT]


def test_a_capture_endpoint_that_never_appears_still_restores_the_route(tmp_path: Path) -> None:
    """start() fails after the route was applied: it puts the route back itself."""
    def responder(msg: list[int]) -> list[list[int]]:
        if msg[:4] == [0xF0, 0x7D, 0x1D, 0x10]:     # mic_src=0, adc=[2, 1]
            return [[0xF0, 0x7D, 0x1D, 0x10, 0, 2, 1, 1, 2, 70, 70, 0, 0, 0xF7]]
        return []

    class NoEndpoint(FakeBackend):
        def find_capture(self, device: Device, *, timeout_s: float = 15.0) -> CaptureDevice:
            raise HilError(TIMEOUT, "no capture endpoint")

    bus, backend, set_mode = FakeMidiBus(responder=responder), NoEndpoint(), FakeSetMode()
    s = session(dev_default(), bus, backend, set_mode, tmp_path, seconds=1.0)
    with pytest.raises(HilError) as e:
        s.start()
    assert e.value.code == TIMEOUT
    frames = route_frames(bus)
    assert frames[1:3] == [[0xF0, 0x7D, 0x1D, 0x02, 0x01, 0xF7],          # applied: codec1
                           [0xF0, 0x7D, 0x1D, 0x01, 0x01, 0x01, 0xF7]]
    assert frames[3:] == [[0xF0, 0x7D, 0x1D, 0x01, 0x00, 0x02, 0xF7],     # restored, inputs
                          [0xF0, 0x7D, 0x1D, 0x01, 0x01, 0x01, 0xF7],     # first...
                          [0xF0, 0x7D, 0x1D, 0x02, 0x00, 0xF7]]           # ...then the source
    assert set_mode.calls == [UsbMode.AUDIO, UsbMode.DEFAULT]
    assert backend.started == []


def test_capture_one_shot_records_and_returns_the_result(tmp_path: Path) -> None:
    bus, backend, set_mode = FakeMidiBus(), FakeBackend(), FakeSetMode()
    r = capture(dev_default(), seconds=2.0, backend=backend,
                midi_factory=midi_factory_for(bus), set_mode_fn=set_mode,
                discover_fn=lambda: [dev_default()], logdir=tmp_path,
                out=tmp_path / "one.wav", sleep=lambda _s: None)
    assert r.silent is False and Path(r.wav).is_file()
    assert set_mode.calls == [UsbMode.AUDIO, UsbMode.DEFAULT]


def test_the_wav_lands_under_the_logdir_by_default(tmp_path: Path) -> None:
    bus, backend, set_mode = FakeMidiBus(), FakeBackend(), FakeSetMode()
    s = CaptureSession(dev_audio(), backend=backend, midi_factory=midi_factory_for(bus),
                       set_mode_fn=set_mode, discover_fn=lambda: [dev_audio()],
                       logdir=tmp_path, sleep=lambda _s: None, seconds=1.0)
    info = s.start()
    assert Path(info["wav"]).parent.parent == tmp_path
    assert Path(info["wav"]).name == "capture.wav"
    s.stop()


# --------------------------------------------------------------------------- validation


@pytest.mark.parametrize("kw, needle", [
    ({"preset": "onboard-mic"}, "preset"),
    ({"seconds": 0}, "seconds"),
    ({"seconds": 1200.0}, "seconds"),
    ({"volume": 120}, "volume"),
])
def test_bad_arguments_are_refused_before_anything_is_touched(tmp_path: Path, kw, needle) -> None:
    with pytest.raises(HilError) as e:
        CaptureSession(dev_audio(), logdir=tmp_path, **kw)
    assert e.value.code == BAD_ARGS and needle in e.value.message


def test_health_and_stop_before_start_are_errors(tmp_path: Path) -> None:
    s = CaptureSession(dev_audio(), logdir=tmp_path)
    for call in (s.health, s.stop, s.wait):
        with pytest.raises(HilError) as e:
            call()
        assert e.value.code == BAD_ARGS


def test_starting_twice_is_refused(tmp_path: Path) -> None:
    bus, backend, set_mode = FakeMidiBus(), FakeBackend(), FakeSetMode()
    s = session(dev_audio(), bus, backend, set_mode, tmp_path, seconds=1.0)
    s.start()
    with pytest.raises(HilError) as e:
        s.start()
    assert e.value.code == BAD_ARGS
    s.stop()


# --------------------------------------------------------------------------- backends


class FakeSd:
    """Just enough of sounddevice to exercise the host-API ranking."""

    def __init__(self, devices: list[dict], hostapis: list[str]) -> None:
        self._devices = devices
        self._hostapis = hostapis

    def _terminate(self) -> None:
        pass

    def _initialize(self) -> None:
        pass

    def query_devices(self) -> list[dict]:
        return self._devices

    def query_hostapis(self, index: int) -> dict:
        return {"name": self._hostapis[index]}


def test_sounddevice_backend_prefers_the_native_host_api() -> None:
    sd = FakeSd(
        devices=[{"name": "Crosspad Audio", "hostapi": 0, "max_input_channels": 2,
                  "default_samplerate": 44100.0},
                 {"name": "Crosspad Audio", "hostapi": 1, "max_input_channels": 2,
                  "default_samplerate": 48000.0},
                 {"name": "HDA Intel", "hostapi": 1, "max_input_channels": 2,
                  "default_samplerate": 48000.0}],
        hostapis=["MME", "Windows WASAPI"])
    cap = SoundDeviceBackend(sd).find_capture(dev_audio())
    assert cap.api == "Windows WASAPI" and cap.index == 1 and cap.sample_rate == 48000


def test_sounddevice_backend_times_out_with_a_usable_hint() -> None:
    sd = FakeSd(devices=[{"name": "HDA Intel", "hostapi": 0, "max_input_channels": 2}],
                hostapis=["ALSA"])
    backend = SoundDeviceBackend(sd, sleep=lambda _s: None, clock=_ticking())
    with pytest.raises(HilError) as e:
        backend.find_capture(dev_audio(), timeout_s=1.0)
    assert e.value.code == TIMEOUT and "usb-mode set audio" in (e.value.hint or "")


def _ticking(step: float = 0.6):
    state = {"t": 0.0}

    def clock() -> float:
        state["t"] += step
        return state["t"]
    return clock


def test_missing_sounddevice_is_reported_as_not_supported(
        monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(sys.modules, "sounddevice", None)
    monkeypatch.setattr(sys, "platform", "win32")       # no arecord fallback there
    with pytest.raises(HilError) as e:
        default_backend()
    assert e.value.code == NOT_SUPPORTED and "crosspad-hil[audio]" in (e.value.hint or "")


def test_arecord_backend_uses_the_card_discovery_already_did() -> None:
    cap = ArecordBackend(run_text=lambda _argv: "").find_capture(dev_audio())
    assert cap.api == "ALSA" and cap.alsa_id == "Audio"


def test_arecord_backend_falls_back_to_arecord_l() -> None:
    listing = ("card 1: Audio [Crosspad Audio], device 0: USB Audio [USB Audio]\n"
               "  Subdevices: 1/1\n")
    dev = Device(id="dev_1234", serial=None, usb_mode=UsbMode.AUDIO, ports=Ports(esp_midi=MIDI))
    cap = ArecordBackend(run_text=lambda _argv: listing).find_capture(dev)
    assert cap.alsa_id == "Audio" and "Crosspad Audio" in cap.name


def test_arecord_backend_times_out_when_the_card_never_appears() -> None:
    dev = Device(id="dev_1234", serial=None, usb_mode=UsbMode.AUDIO, ports=Ports(esp_midi=MIDI))
    backend = ArecordBackend(run_text=lambda _argv: "", sleep=lambda _s: None,
                             clock=_ticking())
    with pytest.raises(HilError) as e:
        backend.find_capture(dev, timeout_s=1.0)
    assert e.value.code == TIMEOUT


def test_arecord_argv_asks_for_exactly_the_requested_length(monkeypatch, tmp_path: Path) -> None:
    seen: dict = {}

    class FakeProc:
        stderr = None
        returncode = 0

        def __init__(self, argv, **kw) -> None:
            seen["argv"] = argv

        def wait(self, timeout=None) -> int:
            return 0

        def poll(self) -> int:
            return 0

    monkeypatch.setattr("crosspad_hil.capture.subprocess.Popen", FakeProc)
    cap = CaptureDevice(name="Crosspad Audio", api="ALSA", alsa_id="Audio")
    rec = ArecordBackend().start(cap, tmp_path / "a.wav", seconds=3.0,
                                 sample_rate=48000, channels=2)
    argv = seen["argv"]
    assert argv[:2] == ["arecord", "-d"] and argv[2] == "3"
    assert "plughw:Audio" in argv and "S16_LE" in argv and "48000" in argv
    assert argv[-1].endswith("a.wav")
    rec.stop()


def test_arecord_records_open_ended_without_a_duration(monkeypatch, tmp_path: Path) -> None:
    seen: dict = {}

    class FakeProc:
        stderr = None
        returncode = 0

        def __init__(self, argv, **kw) -> None:
            seen["argv"] = argv

        def wait(self, timeout=None) -> int:
            return 0

        def poll(self) -> int:
            return 0

        def terminate(self) -> None:
            pass

    monkeypatch.setattr("crosspad_hil.capture.subprocess.Popen", FakeProc)
    cap = CaptureDevice(name="Crosspad Audio", api="ALSA", alsa_id="Audio")
    ArecordBackend().start(cap, tmp_path / "a.wav", seconds=None, sample_rate=48000, channels=2)
    assert "-d" not in seen["argv"]


def test_arecord_through_pipewire_is_a_note_not_a_fault(monkeypatch, tmp_path: Path) -> None:
    """The card was busy, so the take went through PipeWire: the health says so,
    but a complete take is still ok — pitched_kit scored nothing on this bench
    because the note alone flipped `ok`."""
    class FakeProc:
        stderr = None
        returncode = 0
        calls = 0

        def __init__(self, argv, **kw) -> None:
            pass

        def wait(self, timeout=None) -> int:
            # The plughw: attempt "finishes" at once — busy_now() reads a
            # non-zero rc with "busy" on stderr; the PipeWire one runs.
            FakeProc.calls += 1
            if FakeProc.calls == 1:
                self.returncode = 1
                return 1
            return 0

        def poll(self) -> int:
            return 0

    class BusyProc(FakeProc):
        class _Err:
            @staticmethod
            def read() -> bytes:
                return b"audio open error: Device or resource busy"
        stderr = _Err()

    procs = iter([BusyProc, FakeProc])
    envs: list[dict] = []

    def popen(argv, **kw):
        envs.append(kw.get("env") or {})
        return next(procs)(argv, **kw)

    monkeypatch.setattr("crosspad_hil.capture.subprocess.Popen", popen)
    path = tmp_path / "a.wav"
    with wave.open(str(path), "wb") as w:
        w.setnchannels(2)
        w.setsampwidth(2)
        w.setframerate(48000)
        w.writeframes(b"\0" * 4 * 48000 * 2)
    cap = CaptureDevice(name="Crosspad Audio", api="ALSA", alsa_id="Audio")
    rec = ArecordBackend().start(cap, path, seconds=2.0, sample_rate=48000, channels=2)
    rec.stop()
    health = rec.health()
    assert health["ok"] is True
    assert any("PipeWire" in f for f in health["flags"])
    assert "PipeWire" in health["detail"]
    # Through PipeWire it records the CrossPad's node, never the desktop's microphone.
    assert "PIPEWIRE_NODE" not in envs[0]
    assert envs[1]["PIPEWIRE_NODE"] == "crosspad_mic"


def test_an_arecord_stuck_on_a_hung_card_does_not_take_the_run_down(monkeypatch,
                                                                     tmp_path: Path) -> None:
    """2026-10-06: the board hung in the audio profile; arecord ignored SIGKILL (stuck
    in the kernel on the USB device) and stop() raised TimeoutExpired."""
    import subprocess

    class Stuck:
        def read(self) -> bytes:
            raise AssertionError("a hung process's stderr would block forever")

    class FakeProc:
        stderr = Stuck()
        returncode = None
        killed = False

        def __init__(self, argv, **kw) -> None:
            pass

        def wait(self, timeout=None) -> int:
            raise subprocess.TimeoutExpired(["arecord"], timeout)

        def poll(self) -> None:
            return None

        def terminate(self) -> None:
            pass

        def kill(self) -> None:
            FakeProc.killed = True

    monkeypatch.setattr("crosspad_hil.capture.subprocess.Popen", FakeProc)
    cap = CaptureDevice(name="Crosspad Audio", api="ALSA", alsa_id="Audio")
    rec = ArecordBackend().start(cap, tmp_path / "a.wav", seconds=None, sample_rate=48000,
                                 channels=2)
    rec.stop()
    health = rec.health()
    assert FakeProc.killed and health["recorder_hung"] is True and not health["ok"]
    assert any("did not exit even on SIGKILL" in f for f in health["flags"])


# --------------------------------------------------------------------------- the sampler bus

def test_sampler_preset_records_the_digital_bus_and_moves_no_adc(tmp_path: Path) -> None:
    """mic_src 3 = the pads-only recording bus ("USB rec"): no ADC is involved, so the
    preset must not move either codec's input (live_jam's capture)."""
    bus, backend, set_mode = FakeMidiBus(), FakeBackend(), FakeSetMode()
    s = session(dev_audio(), bus, backend, set_mode, tmp_path, seconds=1.0, preset="sampler",
                volume=80)
    info = s.start()
    frames = route_frames(bus)
    assert frames[0] == [0xF0, 0x7D, 0x1D, 0x06, 0x01, 0xF7]
    assert frames[1] == [0xF0, 0x7D, 0x1D, 0x02, 0x03, 0xF7]        # mic_src = the bus
    assert all(f[3] not in (0x01, 0x04) for f in frames)             # no ADC, no codec volume
    assert (info["codec"], info["adc_input"]) == (3, None)
    s.stop()
    assert PRESETS["sampler"] == (3, None)
