"""Daemon ops for capture and analyze: handles, expiry, and the offline verdict path."""
from __future__ import annotations

import io
import wave
from pathlib import Path

import numpy as np
import pytest

from crosspad_hil import serve
from crosspad_hil.devices import Backends, SerialPortInfo, discover
from crosspad_hil.errors import BAD_ARGS, HANDLE_EXPIRED
from tests.fakes import FakeSerial

CDC = "/dev/ttyACM0"
CON = "/dev/ttyACM1"
SR = 48000


def _backends() -> Backends:
    ports = [
        SerialPortInfo(path=CDC, vid=0x303A, pid=0x3456, serial="AB12CD34",
                       product="CrossPad", location="1-1.2:1.0"),
        SerialPortInfo(path=CON, vid=0x0483, pid=0x5740, serial="STM0001",
                       product="CrossPad MIDI+Serial", location="1-1.3:1.0"),
    ]
    return Backends(list_serial=lambda: list(ports), list_midi=lambda: [],
                    list_audio=lambda: [])


class Clock:
    def __init__(self) -> None:
        self.t = 1000.0

    def __call__(self) -> float:
        return self.t


class FakeCapture:
    """Stands in for CaptureSession: records the arguments and the lifecycle order."""

    instances: list[FakeCapture] = []

    def __init__(self, device, **kw) -> None:
        self.device = device
        self.kw = kw
        self.events: list[str] = []
        self.running = False
        self.closed = False
        FakeCapture.instances.append(self)

    def start(self) -> dict:
        self.events.append("start")
        self.running = True
        return {"device": self.device.id, "wav": "/tmp/cap.wav", "preset": self.kw.get("preset"),
                "backend": "fake", "seconds": self.kw.get("seconds"), "channels": 2}

    def health(self) -> dict:
        return {"ok": True, "running": self.running, "overruns": 0}

    def stop(self):
        self.events.append("stop")
        self.running = False

        class R:
            @staticmethod
            def to_dict() -> dict:
                return {"wav": "/tmp/cap.wav", "seconds": 2.0, "peak_dbfs": -8.0,
                        "rms_dbfs": -14.0, "overruns": 0, "silent": False}
        return R()

    def close(self) -> None:
        self.events.append("close")
        self.closed = True
        self.running = False


@pytest.fixture(autouse=True)
def _reset_instances():
    FakeCapture.instances = []


@pytest.fixture
def lock_dir(monkeypatch, tmp_path):
    monkeypatch.setenv("XDG_RUNTIME_DIR", str(tmp_path))
    return tmp_path


def _daemon(**kw):
    fakes = {CDC: FakeSerial([]), CON: FakeSerial([])}
    out = io.StringIO()
    clock = Clock()
    d = serve.Daemon(io.StringIO(), out, backends=_backends(),
                     serial_factory=lambda path, **_: fakes[path], clock=clock,
                     handle_ttl_s=60.0, capture_factory=FakeCapture, **kw)
    return d, clock, discover(_backends())[0].id


def _ok(resp: dict) -> dict:
    assert resp["ok"] is True, resp
    return resp["result"]


def _err(resp: dict) -> dict:
    assert resp["ok"] is False, resp
    return resp["error"]


def write_wav(path: Path, data: np.ndarray, sr: int = SR) -> Path:
    pcm = (np.clip(data, -1.0, 0.999969) * 32768.0).astype("<i2")
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sr)
        w.writeframes(pcm.tobytes())
    return path


# --------------------------------------------------------------------------- capture ops


def test_capture_start_mints_a_cap_handle(lock_dir) -> None:
    d, _clock, dev = _daemon()
    r = _ok(d.handle({"id": 1, "op": "capture.start",
                      "args": {"device": dev, "seconds": 2.0, "preset": "headphone"}}))
    assert r["handle"] == "cap_1" and r["preset"] == "headphone" and r["device"] == dev
    sess = FakeCapture.instances[0]
    assert sess.events == ["start"]
    assert sess.kw["resume_audio_tasks"] is True      # the RT mixer resume is the default
    assert sess.kw["seconds"] == 2.0


def test_capture_stop_returns_the_measured_result_and_frees_the_handle(lock_dir) -> None:
    d, _clock, dev = _daemon()
    h = _ok(d.handle({"id": 1, "op": "capture.start", "args": {"device": dev}}))["handle"]
    r = _ok(d.handle({"id": 2, "op": "capture.stop", "args": {"handle": h}}))
    assert r["peak_dbfs"] == -8.0 and r["silent"] is False
    assert FakeCapture.instances[0].events == ["start", "stop"]
    assert _err(d.handle({"id": 3, "op": "capture.health",
                          "args": {"handle": h}}))["code"] == HANDLE_EXPIRED


def test_capture_health_while_recording(lock_dir) -> None:
    d, _clock, dev = _daemon()
    h = _ok(d.handle({"id": 1, "op": "capture.start", "args": {"device": dev}}))["handle"]
    r = _ok(d.handle({"id": 2, "op": "capture.health", "args": {"handle": h}}))
    assert r["running"] is True and r["ok"] is True


def test_an_expired_cap_handle_puts_the_usb_profile_back(lock_dir) -> None:
    """A forgotten capture must not leave the board in UAC2 with no CDC endpoint."""
    d, clock, dev = _daemon()
    _ok(d.handle({"id": 1, "op": "capture.start", "args": {"device": dev}}))
    clock.t += d.CAPTURE_TTL_S + 1
    assert d._handles.sweep() == ["cap_1"]
    assert FakeCapture.instances[0].events == ["start", "stop"]


def test_capture_start_rejects_a_non_string_device(lock_dir) -> None:
    d, _clock, _dev = _daemon()
    assert _err(d.handle({"id": 1, "op": "capture.start",
                          "args": {"device": 5}}))["code"] == BAD_ARGS


def test_capture_options_reach_the_session(lock_dir) -> None:
    d, _clock, dev = _daemon()
    _ok(d.handle({"id": 1, "op": "capture.start",
                  "args": {"device": dev, "preset": "mics", "resume_audio_tasks": False,
                           "keep_mode": True, "volume": 70, "channels": 1,
                           "sample_rate": 44100, "out": "/tmp/x.wav"}}))
    kw = FakeCapture.instances[0].kw
    assert kw["preset"] == "mics" and kw["resume_audio_tasks"] is False
    assert kw["keep_mode"] is True and kw["volume"] == 70 and kw["channels"] == 1
    assert kw["sample_rate"] == 44100 and kw["out"] == "/tmp/x.wav"
    assert "seconds" not in kw                    # unset stays unset: an open-ended capture


# --------------------------------------------------------------------------- analyze op


def test_analyze_wav_returns_a_verdict(lock_dir, tmp_path: Path) -> None:
    d, _clock, _dev = _daemon()
    t = np.arange(int(2.0 * SR)) / SR
    wav = write_wav(tmp_path / "tone.wav", 0.4 * np.sin(2 * np.pi * 440 * t))
    r = _ok(d.handle({"id": 1, "op": "analyze.wav", "args": {"kind": "click", "wav": str(wav)}}))
    assert r["kind"] == "click" and r["verdict"] == "pass" and r["passed"] is True


def test_analyze_wav_passes_expected_and_options(lock_dir, tmp_path: Path) -> None:
    d, _clock, _dev = _daemon()
    rng = np.random.default_rng(3)
    x = 1e-4 * rng.standard_normal(int(3.0 * SR))
    env = np.exp(-np.arange(int(0.1 * SR)) / (0.02 * SR))
    for t in (0.5, 1.5, 2.5):
        i = int(t * SR)
        x[i:i + len(env)] += 0.5 * env * np.sin(2 * np.pi * 220 * np.arange(len(env)) / SR)
    wav = write_wav(tmp_path / "hits.wav", x)
    r = _ok(d.handle({"id": 1, "op": "analyze.wav",
                      "args": {"kind": "onset", "wav": str(wav), "expected": [0.5, 1.5, 2.5]}}))
    assert r["expected"] == 3 and r["matched"] == 3 and r["missed"] == []
    r2 = _ok(d.handle({"id": 2, "op": "analyze.wav",
                       "args": {"kind": "click", "wav": str(wav),
                                "options": {"max_clicks_per_s": 0.0}}}))
    assert r2["kind"] == "click"


def test_analyze_wav_rejects_an_unknown_kind(lock_dir, tmp_path: Path) -> None:
    d, _clock, _dev = _daemon()
    wav = write_wav(tmp_path / "q.wav", np.zeros(SR))
    e = _err(d.handle({"id": 1, "op": "analyze.wav",
                       "args": {"kind": "spectrogram", "wav": str(wav)}}))
    assert e["code"] == BAD_ARGS and "multitone" in (e["hint"] or "")


def test_analyze_wav_needs_kind_and_wav(lock_dir) -> None:
    d, _clock, _dev = _daemon()
    assert _err(d.handle({"id": 1, "op": "analyze.wav", "args": {}}))["code"] == BAD_ARGS
