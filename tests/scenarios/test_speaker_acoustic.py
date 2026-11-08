"""Scenario `speaker_acoustic` with the audio layer faked — no mics, no board.

The CDC setup runs against a scripted fake serial (the real CdcLink and the real
verbs on top of it), while the capture session, the pad stimulus and the
analysis kernels are stand-ins. What is being tested is the order of operations
— set the sampler up while CDC still exists, then switch profiles — and the
legacy pass rule over the numbers the kernel reports.
"""
from __future__ import annotations

import threading
import wave
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pytest

import crosspad_hil.analyze as analyze_mod
import crosspad_hil.capture as capture_mod
import crosspad_hil.stim as stim_mod
from crosspad_hil.cdc import CdcLink
from crosspad_hil.devices import AudioCardInfo, Device, Ports, SerialPortInfo, UsbMode
from crosspad_hil.errors import NOT_SUPPORTED, TIMEOUT, HilError
from crosspad_hil.scenarios import base, get, speaker_acoustic
from tests.fakes import FakeSerial

CDC = SerialPortInfo(path="/dev/ttyACM0", vid=0x303A, pid=0x3456, serial="CP-1",
                     product="Crosspad", location="1-2.1")
CONSOLE = SerialPortInfo(path="/dev/ttyACM1", vid=0x0483, pid=0x5740, serial="STM-1",
                         product="CrossPad MIDI+Serial", location="1-2.2")
UAC2 = AudioCardInfo(name="Crosspad Audio", sounddevice_index=3, alsa_id="Audio")

SETUP_SCRIPT = [
    ("AUDIO_TASKS 1", "OK"),
    ("KIT_LOAD 0", "OK"),
    ("KIT_STATUS", "KITSTATUS: current=0 loading=0 pending=-1 name=Basic"),
    ("APP_START Sampler", "OK"),
    ("APP_LIST", "APPS: Sampler,Mixer running=Sampler"),
]


def make_device(mode: UsbMode = UsbMode.DEFAULT) -> Device:
    ports = (Ports(console=CONSOLE, uac2=UAC2) if mode is UsbMode.AUDIO
             else Ports(cdc=CDC, console=CONSOLE))
    return Device(id="dev_ab12", serial="CP-1", usb_mode=mode, ports=ports)


def write_silence(path: Path, seconds: float = 0.2, sr: int = 48000) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as w:
        w.setnchannels(2)
        w.setsampwidth(2)
        w.setframerate(sr)
        w.writeframes(b"\0\0\0\0" * int(sr * seconds))


@dataclass
class FakeResult:
    wav: str
    silent: bool = False
    hint: str | None = None

    def to_dict(self) -> dict:
        return {"wav": self.wav, "silent": self.silent, "hint": self.hint,
                "peak_dbfs": -50.0 if self.silent else -12.0}


@dataclass
class Bench:
    sessions: list[dict] = field(default_factory=list)
    stims: list[dict] = field(default_factory=list)
    frames: list[bytes] = field(default_factory=list)
    analyses: list[dict] = field(default_factory=list)
    modes: list[str] = field(default_factory=list)
    silent: bool = False
    start_error: HilError | None = None
    analyze_error: HilError | None = None
    floor_dbfs: float = -70.0
    hit_dbfs: float = -50.0
    matched: int | None = None
    device: Device = field(default_factory=lambda: make_device(UsbMode.DEFAULT))

    def velocity_verdict(self, hits: int) -> dict:
        matched = hits if self.matched is None else self.matched
        per_hit = [{"i": i, "vel": 120, "onset_s": 3.0 + i * 0.5,
                    "peak_dbfs": self.hit_dbfs if i < matched else None}
                   for i in range(hits)]
        return {"kind": "velocity", "verdict": "pass", "reasons": [],
                "noise_floor_dbfs": self.floor_dbfs, "matched": matched,
                "missed": list(range(matched, hits)), "per_hit": per_hit, "extra": [],
                "latency_ms": {"p50": 18.0, "p90": 24.0, "max": 30.0, "n": matched}}


class FakeSession:
    def __init__(self, bench: Bench, device: Device, **kw: Any) -> None:
        self._bench = bench
        self.device = device
        self.kw = kw
        bench.sessions.append({"device": device.id, **kw})

    def start(self) -> dict:
        if self._bench.start_error is not None:
            raise self._bench.start_error
        write_silence(Path(self.kw["out"]))
        # the session is what enters the audio profile
        self._bench.modes.append("audio")
        self.device = make_device(UsbMode.AUDIO)
        return {"wav": str(self.kw["out"])}

    def wait(self, timeout_s: float | None = None) -> bool:
        return True

    def stop(self) -> FakeResult:
        if not self.kw.get("keep_mode"):
            self._bench.modes.append("default")
            self.device = make_device(UsbMode.DEFAULT)
        hint = ("nothing on the capture endpoint: check that the RT mixer was resumed"
                if self._bench.silent else None)
        return FakeResult(wav=str(self.kw["out"]), silent=self._bench.silent, hint=hint)


class FakeStim:
    def __init__(self, bench: Bench, **kw: Any) -> None:
        self._bench = bench
        self.kw = kw
        bench.stims.append(dict(kw))

    def wait(self, timeout_s: float | None = None) -> bool:
        return True

    def stop(self, timeout_s: float = 5.0) -> dict:
        hits = int(self.kw["seconds"] * self.kw["rate_hz"])
        return {"running": False, "sent": hits, "presses": hits}


class FakeMidi:
    def __init__(self, bench: Bench, device: Device, role: Any) -> None:
        self._bench = bench
        self.closed = False

    def open(self) -> None:
        return None

    def send_sysex(self, frame: bytes) -> None:
        self._bench.frames.append(bytes(frame))

    def close(self) -> None:
        self.closed = True


@pytest.fixture
def bench(monkeypatch: pytest.MonkeyPatch) -> Bench:
    b = Bench()
    monkeypatch.setattr(capture_mod, "CaptureSession",
                        lambda device, **kw: FakeSession(b, device, **kw))
    monkeypatch.setattr(stim_mod, "start_stimulus",
                        lambda device, **kw: FakeStim(b, **kw))
    monkeypatch.setattr(speaker_acoustic, "MidiIO",
                        lambda device, role: FakeMidi(b, device, role))
    monkeypatch.setattr(speaker_acoustic, "SYSEX_GAP_S", 0.0)
    monkeypatch.setattr(speaker_acoustic, "APP_START_WAIT_S", 0.0)

    def fake_analyze(kind, wav, *, expected=None, **kw):
        b.analyses.append({"kind": kind, "expected": expected, **kw})
        if b.analyze_error is not None:
            raise b.analyze_error
        if kind == "silence":
            return {"kind": "silence", "verdict": "pass", "rms_dbfs": -68.0}
        return b.velocity_verdict(len(expected))

    monkeypatch.setattr(analyze_mod, "analyze", fake_analyze)

    def fake_set_mode(device, mode, **kw):
        b.modes.append(mode.value)
        return make_device(UsbMode(mode))

    monkeypatch.setattr(speaker_acoustic, "set_mode", fake_set_mode)
    return b


@dataclass
class FakeContext(base.Context):
    cdc_fake: FakeSerial = None  # type: ignore[assignment]

    def open_cdc(self) -> CdcLink:
        link = CdcLink(self.device.ports.cdc.path, serial_factory=lambda path, **kw: self.cdc_fake)
        link.open()
        self.opened.append(link)
        return link


def make_ctx(tmp_path: Path, script: list | None = None,
             mode: UsbMode = UsbMode.DEFAULT) -> FakeContext:
    return FakeContext(device=make_device(mode), workdir=tmp_path,
                       cancelled=threading.Event(), log=lambda s: None,
                       cdc_fake=FakeSerial(SETUP_SCRIPT if script is None else script))


def run(ctx: FakeContext, **params: Any) -> base.Report:
    kw: dict[str, Any] = {"hits": 12, "pre_s": 0.0, "post_s": 0.0}
    kw.update(params)
    return speaker_acoustic.SCENARIO.run(ctx, speaker_acoustic.Params(**kw), base.Progress())


def test_registered_and_defaults() -> None:
    assert get("speaker_acoustic") is speaker_acoustic.SCENARIO
    p = speaker_acoustic.Params()
    assert (p.pad, p.kit, p.hits, p.velocity, p.gate_ms, p.pre_s, p.post_s, p.preset,
            p.rise_db, p.hit_ratio, p.transport, p.keep_mode) == (
        0, 0, 12, 120, 60.0, 3.0, 7.0, "mics", 3.0, 0.75, "sysex_esp", False)


def test_a_speaker_that_sounds_passes(tmp_path: Path, bench: Bench) -> None:
    ctx = make_ctx(tmp_path)
    rep = run(ctx)
    assert rep.passed, rep.summary
    assert rep.exit_code == 0
    d = rep.data
    assert d["hits_detected"] == 12
    assert d["hits_required"] == 9          # max(1, int(12 * 0.75))
    assert d["floor_dbfs"] == -70.0
    assert d["rise_db"] == 20.0
    assert d["tail_rms_dbfs"] == -68.0
    assert d["preset"] == "mics"
    assert (d["codec"], d["adc_input"]) == (0, 2)   # the only acoustic path
    assert any(a.role == "capture" for a in rep.artifacts)


def test_setup_runs_over_cdc_before_the_profile_switch(tmp_path: Path, bench: Bench) -> None:
    ctx = make_ctx(tmp_path)
    run(ctx)
    written = ctx.cdc_fake.written
    assert written[0] == "AUDIO_TASKS 1"
    assert "KIT_LOAD 0" in written
    assert "APP_START Sampler" in written
    # the audio profile is entered by the capture session, i.e. after the setup
    assert bench.modes == ["audio", "default"]


def test_the_pads_are_driven_over_sysex_while_capturing(tmp_path: Path, bench: Bench) -> None:
    run(make_ctx(tmp_path), pad=3, velocity=100, hits=8)
    stim = bench.stims[0]
    assert stim["pads"] == [3]
    assert stim["rate_hz"] == 1.0 / speaker_acoustic.HIT_PERIOD_S
    assert stim["seconds"] == 8 * speaker_acoustic.HIT_PERIOD_S
    assert stim["velocity"] == 100
    assert stim["transport"] == "sysex_esp"


def test_both_codecs_are_unmuted_and_turned_up(tmp_path: Path, bench: Bench) -> None:
    run(make_ctx(tmp_path))
    # volume(codec, 100) and mute(codec, 0) for codec 0 and 1: F0 7D 1D 04|05 …
    assert bench.frames == [
        bytes([0xF0, 0x7D, 0x1D, 0x04, 0x00, 100, 0xF7]),
        bytes([0xF0, 0x7D, 0x1D, 0x05, 0x00, 0x00, 0xF7]),
        bytes([0xF0, 0x7D, 0x1D, 0x04, 0x01, 100, 0xF7]),
        bytes([0xF0, 0x7D, 0x1D, 0x05, 0x01, 0x00, 0xF7]),
    ]


def test_the_analysis_gets_the_send_grid_and_the_legacy_windows(tmp_path: Path,
                                                                bench: Bench) -> None:
    run(make_ctx(tmp_path), hits=4, pre_s=0.3, post_s=0.2, rise_db=3.0)
    vel = bench.analyses[0]
    assert vel["kind"] == "velocity"
    assert [h["t"] for h in vel["expected"]] == [0.3, 0.8, 1.3, 1.8]
    assert {h["vel"] for h in vel["expected"]} == {120}
    assert vel["rise_db"] == 3.0
    assert vel["tol_s"] == speaker_acoustic.HIT_WINDOW_S
    # the tail is the post-roll after the amp hold has expired
    assert bench.analyses[1]["kind"] == "silence"
    assert bench.analyses[1]["expected"]["skip_s"] == 0.3 + 2.0 + speaker_acoustic.TAIL_SETTLE_S


def test_too_few_hits_located_fails(tmp_path: Path, bench: Bench) -> None:
    bench.matched = 5                     # 5 of 12 is under the 75 % rule
    rep = run(make_ctx(tmp_path))
    assert not rep.passed
    assert rep.exit_code == 1
    assert "5/12 hits located" in rep.summary


def test_a_rise_under_three_db_fails(tmp_path: Path, bench: Bench) -> None:
    bench.hit_dbfs = -68.5                # 1.5 dB over the floor
    rep = run(make_ctx(tmp_path))
    assert not rep.passed
    assert "need 3.0 dB" in rep.summary


def test_a_silent_capture_fails_and_keeps_the_hint(tmp_path: Path, bench: Bench) -> None:
    bench.silent = True
    rep = run(make_ctx(tmp_path))
    assert not rep.passed
    assert rep.exit_code == 1             # a dead amplifier looks exactly like this
    assert "RT mixer was resumed" in rep.summary


def test_sampler_that_never_starts_is_an_environment_error(tmp_path: Path, bench: Bench) -> None:
    script = [*SETUP_SCRIPT[:-1], ("APP_LIST", "APPS: Sampler,Mixer running=-")]
    rep = run(make_ctx(tmp_path, script))
    assert rep.exit_code == 2
    assert "APP_START Sampler: not running" in rep.summary


def test_another_app_running_is_an_environment_error(tmp_path: Path, bench: Bench,
                                                     monkeypatch: pytest.MonkeyPatch) -> None:
    """The belt-and-braces check: app_start was happy but something else owns the screen."""
    monkeypatch.setattr(speaker_acoustic.verbs, "app_start", lambda link, name, wait_s=3.0: {})
    script = [*SETUP_SCRIPT[:-1], ("APP_LIST", "APPS: Sampler,Mixer running=Mixer")]
    rep = run(make_ctx(tmp_path, script))
    assert rep.exit_code == 2
    assert "would measure silence" in rep.summary


def test_a_failed_setup_is_an_environment_error(tmp_path: Path, bench: Bench,
                                                monkeypatch: pytest.MonkeyPatch) -> None:
    def boom(link, kit_id, wait_s=15.0):
        raise HilError(TIMEOUT, f"KIT_LOAD {kit_id}: not current after {wait_s:.1f}s")

    monkeypatch.setattr(speaker_acoustic.verbs, "kit_load", boom)
    rep = run(make_ctx(tmp_path))
    assert rep.exit_code == 2
    assert "setup failed" in rep.summary


def test_a_capture_that_never_starts_is_an_environment_error(tmp_path: Path,
                                                             bench: Bench) -> None:
    bench.start_error = HilError(NOT_SUPPORTED, "sounddevice is unavailable")
    rep = run(make_ctx(tmp_path))
    assert rep.exit_code == 2
    assert "capture failed" in rep.summary


def test_a_missing_analysis_extra_is_an_environment_error(tmp_path: Path,
                                                          bench: Bench) -> None:
    bench.analyze_error = HilError(NOT_SUPPORTED, "audio analysis needs numpy")
    rep = run(make_ctx(tmp_path))
    assert rep.exit_code == 2
    assert "numpy" in rep.summary


def test_an_audio_mode_device_is_brought_back_for_setup(tmp_path: Path, bench: Bench) -> None:
    ctx = make_ctx(tmp_path, mode=UsbMode.AUDIO)
    ctx.device = make_device(UsbMode.AUDIO)
    rep = run(ctx)
    assert bench.modes[0] == "default"     # CDC first, capture second
    assert rep.passed, rep.summary


def test_keep_mode_leaves_the_audio_profile_alone(tmp_path: Path, bench: Bench) -> None:
    rep = run(make_ctx(tmp_path), keep_mode=True)
    assert bench.sessions[0]["keep_mode"] is True
    assert bench.modes == ["audio"]
    assert rep.data["final_mode"] == "audio"
    assert rep.data["restored"] is False


def test_unknown_preset_is_bad_args(tmp_path: Path, bench: Bench) -> None:
    with pytest.raises(HilError) as e:
        run(make_ctx(tmp_path), preset="nope")
    assert e.value.code == "BAD_ARGS"


def test_verdict_helper_applies_the_old_two_part_rule() -> None:
    analysis = {"noise_floor_dbfs": -70.0, "matched": 9, "extra": [],
                "per_hit": [{"peak_dbfs": -60.0}] * 9 + [{"peak_dbfs": None}] * 3}
    out = speaker_acoustic.verdict_from(analysis, 12, rise_db=3.0, hit_ratio=0.75)
    assert out["hits_required"] == 9
    assert out["rise_db"] == 10.0
    assert out["passed"] is True
    assert speaker_acoustic.verdict_from(analysis, 12, rise_db=12.0,
                                         hit_ratio=0.75)["passed"] is False
