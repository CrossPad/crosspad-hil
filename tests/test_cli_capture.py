"""`crosspad-hil capture` and `crosspad-hil analyze` — argparse tree, exit codes, rendering."""
from __future__ import annotations

import io
import json
import wave
from pathlib import Path

import numpy as np
import pytest

from crosspad_hil import cli
from crosspad_hil.devices import Backends, SerialPortInfo

SR = 48000
CDC = "/dev/ttyACM0"
CON = "/dev/ttyACM1"


def _backends() -> Backends:
    ports = [
        SerialPortInfo(path=CDC, vid=0x303A, pid=0x3456, serial="AB12CD34",
                       product="CrossPad", location="1-1.2:1.0"),
        SerialPortInfo(path=CON, vid=0x0483, pid=0x5740, serial="STM0001",
                       product="CrossPad MIDI+Serial", location="1-1.3:1.0"),
    ]
    return Backends(list_serial=lambda: list(ports), list_midi=lambda: [],
                    list_audio=lambda: [])


@pytest.fixture
def lock_dir(monkeypatch, tmp_path):
    monkeypatch.setenv("XDG_RUNTIME_DIR", str(tmp_path))
    return tmp_path


class FakeCapture:
    """Records what the CLI asked for and hands back a canned result."""

    last: FakeCapture | None = None

    def __init__(self, device, **kw) -> None:
        self.device = device
        self.kw = kw
        self.events: list[str] = []
        self.result = {"wav": "/tmp/cap.wav", "seconds": 3.0, "peak_dbfs": -8.0,
                       "rms_dbfs": -15.0, "overruns": 0, "silent": False, "sample_rate": SR,
                       "channels": 2, "frames": 3 * SR, "device": device.id,
                       "preset": kw.get("preset"), "backend": "fake",
                       "health": {"detail": "captured 3.0 s of 3.0 s - clean"}, "hint": None}
        FakeCapture.last = self

    def start(self) -> dict:
        self.events.append("start")
        return {"device": self.device.id, "wav": self.result["wav"],
                "preset": self.kw.get("preset"), "codec": 1, "adc_input": 1, "backend": "fake",
                "capture_device": {"name": "Crosspad Audio", "api": "ALSA"},
                "sample_rate": SR, "channels": 2, "seconds": self.kw.get("seconds"),
                "audio_mode_entered": True, "audio_tasks_resumed": True}

    def wait(self, timeout_s: float | None = None) -> bool:
        self.events.append("wait")
        return True

    def stop(self):
        self.events.append("stop")
        outer = self

        class R:
            @staticmethod
            def to_dict() -> dict:
                return dict(outer.result)
        return R()


def _deps(**kw) -> tuple[cli.CliDeps, io.StringIO, io.StringIO]:
    out, err = io.StringIO(), io.StringIO()
    return (cli.CliDeps(backends=_backends(), stdout=out, stderr=err, sleep=lambda _s: None,
                        capture_factory=FakeCapture, **kw), out, err)


def write_wav(path: Path, data: np.ndarray) -> Path:
    pcm = (np.clip(data, -1.0, 0.999969) * 32768.0).astype("<i2")
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(SR)
        w.writeframes(pcm.tobytes())
    return path


def hits_wav(path: Path, times, amps=None) -> Path:
    rng = np.random.default_rng(11)
    total = max(times) + 1.0
    x = 1e-4 * rng.standard_normal(int(total * SR))
    env = np.exp(-np.arange(int(0.12 * SR)) / (0.02 * SR))
    for i, t in enumerate(times):
        amp = 0.5 if amps is None else amps[i]
        j = int(t * SR)
        x[j:j + len(env)] += amp * env * np.sin(2 * np.pi * 220 * np.arange(len(env)) / SR)
    return write_wav(path, x)


# --------------------------------------------------------------------------- capture


def test_capture_runs_the_session_and_prints_the_measurement(lock_dir) -> None:
    deps, out, err = _deps()
    assert cli.main(["capture", "--seconds", "3"], deps=deps) == 0
    assert FakeCapture.last.events == ["start", "wait", "stop"]
    assert FakeCapture.last.kw["preset"] == "headphone"          # the DAC->ADC loop
    assert FakeCapture.last.kw["resume_audio_tasks"] is True     # SysEx 0x1D 06 01
    assert FakeCapture.last.kw["keep_mode"] is False
    text = out.getvalue()
    assert "-8.0 dBFS" in text and "3.00s" in text and "headphone" in text
    assert "recording 3.0s" in err.getvalue()


def test_capture_json(lock_dir) -> None:
    deps, out, _err = _deps()
    assert cli.main(["capture", "--seconds", "1", "--json"], deps=deps) == 0
    data = json.loads(out.getvalue())
    assert data["wav"] == "/tmp/cap.wav" and data["silent"] is False


def test_a_silent_capture_exits_1_and_says_why(lock_dir) -> None:
    deps, out, _err = _deps()

    class Silent(FakeCapture):
        def __init__(self, device, **kw) -> None:
            super().__init__(device, **kw)
            self.result.update(silent=True, peak_dbfs=-120.0,
                               hint="resume the RT mixer (SysEx 0x1D 06 01)")
    deps.capture_factory = Silent
    assert cli.main(["capture"], deps=deps) == 1
    assert "SILENT" in out.getvalue() and "0x1D 06 01" in out.getvalue()


def test_capture_flags_reach_the_session(lock_dir) -> None:
    deps, _out, _err = _deps()
    assert cli.main(["capture", "--seconds", "5", "--preset", "mics", "--volume", "70",
                     "--channels", "1", "--sample-rate", "44100", "--no-resume-tasks",
                     "--keep-mode", "--out", "/tmp/x.wav"], deps=deps) == 0
    kw = FakeCapture.last.kw
    assert kw["seconds"] == 5.0 and kw["preset"] == "mics" and kw["volume"] == 70
    assert kw["channels"] == 1 and kw["sample_rate"] == 44100
    assert kw["resume_audio_tasks"] is False and kw["keep_mode"] is True
    assert kw["out"] == "/tmp/x.wav"


def test_capture_can_analyze_the_take_it_just_made(lock_dir, tmp_path: Path) -> None:
    wav = write_wav(tmp_path / "cap.wav", 0.4 * np.sin(2 * np.pi * 440
                                                       * np.arange(2 * SR) / SR))
    deps, out, _err = _deps()

    class WithWav(FakeCapture):
        def __init__(self, device, **kw) -> None:
            super().__init__(device, **kw)
            self.result["wav"] = str(wav)
    deps.capture_factory = WithWav
    assert cli.main(["capture", "--analyze", "click", "--json"], deps=deps) == 0
    data = json.loads(out.getvalue())
    assert data["analysis"]["kind"] == "click" and data["analysis"]["passed"] is True


def test_capture_rejects_an_unknown_preset(lock_dir) -> None:
    deps, _out, _err = _deps()
    assert cli.main(["capture", "--preset", "onboard-mic"], deps=deps) == 2


# --------------------------------------------------------------------------- analyze


def test_analyze_click_passes_on_a_clean_tone(lock_dir, tmp_path: Path) -> None:
    wav = write_wav(tmp_path / "t.wav", 0.4 * np.sin(2 * np.pi * 440 * np.arange(2 * SR) / SR))
    deps, out, _err = _deps()
    assert cli.main(["analyze", "click", str(wav)], deps=deps) == 0
    assert "click: PASS" in out.getvalue()


def test_analyze_exits_1_on_a_failing_verdict(lock_dir, tmp_path: Path) -> None:
    rng = np.random.default_rng(5)
    wav = write_wav(tmp_path / "n.wav", 0.05 * rng.standard_normal(3 * SR))
    deps, out, _err = _deps()
    assert cli.main(["analyze", "silence", str(wav)], deps=deps) == 1
    text = out.getvalue()
    assert "silence: FAIL" in text and "noise floor" in text


def test_analyze_onset_takes_a_schedule(lock_dir, tmp_path: Path) -> None:
    wav = hits_wav(tmp_path / "h.wav", [0.5, 1.0, 1.5, 2.0])
    deps, out, _err = _deps()
    rc = cli.main(["analyze", "onset", str(wav), "--expected", "[0.5, 1.0, 1.5, 2.0]", "--json"],
                  deps=deps)
    assert rc == 0
    data = json.loads(out.getvalue())
    assert data["matched"] == 4 and data["missed"] == []


def test_analyze_reads_the_schedule_from_a_file(lock_dir, tmp_path: Path) -> None:
    wav = hits_wav(tmp_path / "h.wav", [0.5, 1.0, 1.5])
    sched = tmp_path / "hits.json"
    sched.write_text(json.dumps([{"t": 0.5, "vel": 40}, {"t": 1.0, "vel": 80},
                                 {"t": 1.5, "vel": 127}]), encoding="utf-8")
    deps, out, _err = _deps()
    rc = cli.main(["analyze", "velocity", str(wav), "--expected", f"@{sched}", "--json"],
                  deps=deps)
    data = json.loads(out.getvalue())
    assert [c["vel"] for c in data["curve"]] == [40, 80, 127]
    assert rc in (0, 1)          # the synthetic hits are equal-amplitude on purpose


def test_analyze_velocity_prints_the_curve(lock_dir, tmp_path: Path) -> None:
    vels = [16, 64, 127]
    wav = hits_wav(tmp_path / "v.wav", [0.5, 1.2, 1.9], [v / 127 * 0.7 for v in vels])
    deps, out, _err = _deps()
    sched = json.dumps([{"t": 0.5, "vel": 16}, {"t": 1.2, "vel": 64}, {"t": 1.9, "vel": 127}])
    assert cli.main(["analyze", "velocity", str(wav), "--expected", sched], deps=deps) == 0
    text = out.getvalue()
    assert "velocity: PASS" in text and "vel  16" in text and "vel 127" in text


def test_analyze_opt_passes_extra_options(lock_dir, tmp_path: Path) -> None:
    wav = write_wav(tmp_path / "s.wav", 0.5 * np.sin(2 * np.pi * 440 * np.arange(6 * SR) / SR))
    deps, out, _err = _deps()
    assert cli.main(["analyze", "psd", str(wav), "--opt", "freq=440", "--json"], deps=deps) == 0
    data = json.loads(out.getvalue())
    assert data["freq_hz"] == 440 and data["verdict"] == "pass"


def test_analyze_rejects_a_malformed_opt(lock_dir, tmp_path: Path) -> None:
    wav = write_wav(tmp_path / "s.wav", np.zeros(SR))
    deps, _out, err = _deps()
    assert cli.main(["analyze", "click", str(wav), "--opt", "freq"], deps=deps) == 2
    assert "key=value" in err.getvalue()


def test_analyze_rejects_non_json_expected(lock_dir, tmp_path: Path) -> None:
    wav = write_wav(tmp_path / "s.wav", np.zeros(SR))
    deps, _out, err = _deps()
    assert cli.main(["analyze", "onset", str(wav), "--expected", "four hits"], deps=deps) == 2
    assert "not JSON" in err.getvalue()


def test_analyze_rejects_an_unknown_kind(lock_dir, tmp_path: Path) -> None:
    wav = write_wav(tmp_path / "s.wav", np.zeros(SR))
    deps, _out, _err = _deps()
    assert cli.main(["analyze", "spectrogram", str(wav)], deps=deps) == 2


def test_analyze_missing_file_exits_2(lock_dir, tmp_path: Path) -> None:
    deps, _out, err = _deps()
    assert cli.main(["analyze", "click", str(tmp_path / "nope.wav")], deps=deps) == 2
    assert "no such WAV" in err.getvalue()
