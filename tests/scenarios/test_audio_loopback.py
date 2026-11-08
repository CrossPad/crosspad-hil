"""Scenario `audio_loopback` with the audio layer faked — no sound card, no board.

The capture session, the analysis kernel and the playback backend are all
replaced here: what is under test is the cycle loop, the mode handling and the
verdict, not PortAudio. The fakes keep the real signatures, so a change to
capture.CaptureSession or analyze.analyze that this scenario would trip over
shows up as a failure here.
"""
from __future__ import annotations

import json
import threading
import wave
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest

import crosspad_hil.analyze as analyze_mod
import crosspad_hil.capture as capture_mod
from crosspad_hil import host_gain as hostgain
from crosspad_hil.devices import AudioCardInfo, Device, Ports, SerialPortInfo, UsbMode
from crosspad_hil.errors import ENV, NOT_SUPPORTED, HilError
from crosspad_hil.scenarios import audio_loopback, base, get

CDC = SerialPortInfo(path="/dev/ttyACM0", vid=0x303A, pid=0x3456, serial="CP-1",
                     product="Crosspad", location="1-2.1")
CONSOLE = SerialPortInfo(path="/dev/ttyACM1", vid=0x0483, pid=0x5740, serial="STM-1",
                         product="CrossPad MIDI+Serial", location="1-2.2")
UAC2 = AudioCardInfo(name="Crosspad Audio", sounddevice_index=3, alsa_id="Audio")


def make_device(mode: UsbMode = UsbMode.AUDIO) -> Device:
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
    seconds: float = 3.0
    peak_dbfs: float = -12.0
    rms_dbfs: float = -20.0
    silent: bool = False


class Bench:
    """Everything the scenario reaches for outside itself, recorded."""

    def __init__(self) -> None:
        self.sessions: list[dict] = []
        self.played: list[tuple[str, float]] = []
        self.modes: list[str] = []
        self.references: list[dict] = []
        self.verdicts: list[dict] = []
        self.start_error: HilError | None = None
        self.play_error: HilError | None = None
        self.device = make_device(UsbMode.AUDIO)
        # discover() answers with `visible`; `sightings` overrides it one call at a
        # time, which is how a board that falls back to CDC+MIDI mid-run is staged.
        self.visible = make_device(UsbMode.AUDIO)
        self.sightings: list[Device] = []
        # the host side: PCM in dB as the session left it, what apply() can reach
        self.pcm_db = 0.0
        self.pcm_settable = True
        self.gain_calls: list[str | None] = []

    def gain(self) -> hostgain.HostGain:
        return hostgain.HostGain(card="Audio", controls=[
            {"name": "'PCM',0", "direction": "playback", "raw": 0, "percent": 0,
             "db": self.pcm_db, "on": True}])

    def seen(self) -> Device:
        return self.sightings.pop(0) if self.sightings else self.visible


class FakeSession:
    """capture.CaptureSession's contract: start(), wait(), stop() -> a result."""

    def __init__(self, bench: Bench, device: Device, **kw: Any) -> None:
        self._bench = bench
        self.device = device
        self.kw = kw
        bench.sessions.append({"device": device.id, **kw})

    def start(self) -> dict:
        if self._bench.start_error is not None:
            raise self._bench.start_error
        write_silence(Path(self.kw["out"]))
        return {"wav": str(self.kw["out"]), "preset": self.kw.get("preset")}

    def wait(self, timeout_s: float | None = None) -> bool:
        return True

    def stop(self) -> FakeResult:
        return FakeResult(wav=str(self.kw["out"]))


class FakePlayer:
    name = "fake"

    def __init__(self, bench: Bench) -> None:
        self._bench = bench

    def play(self, device: Device, wav: Path, timeout_s: float) -> None:
        if self._bench.play_error is not None:
            raise self._bench.play_error
        self._bench.played.append((str(wav), timeout_s))


@pytest.fixture(autouse=True)
def fast(monkeypatch: pytest.MonkeyPatch) -> None:
    """Collapse the human-scale waits the old script carried."""
    monkeypatch.setattr(audio_loopback, "MODE_SETTLE_S", 0.0)
    monkeypatch.setattr(audio_loopback, "PLAY_DELAY_S", 0.0)


@pytest.fixture
def bench(monkeypatch: pytest.MonkeyPatch) -> Bench:
    b = Bench()

    monkeypatch.setattr(capture_mod, "CaptureSession",
                        lambda device, **kw: FakeSession(b, device, **kw))
    monkeypatch.setattr(audio_loopback, "default_player", lambda: FakePlayer(b))

    def fake_reference(path, freqs, seconds, *, level=0.7, sr=48000, channels=2, **kw):
        b.references.append({"path": str(path), "freqs": list(freqs), "seconds": seconds,
                             "level": level, "sr": sr})
        write_silence(Path(path))
        return Path(path)

    monkeypatch.setattr(analyze_mod, "write_reference", fake_reference)

    def fake_analyze(kind, wav, *, expected=None, **kw):
        b.verdicts.append({"kind": kind, "wav": str(wav), "expected": expected})
        return dict(b.next_verdict())

    monkeypatch.setattr(analyze_mod, "analyze", fake_analyze)

    def fake_set_mode(device, mode, **kw):
        b.modes.append(mode.value)
        b.visible = make_device(UsbMode(mode))
        b.device = b.visible
        return b.visible

    monkeypatch.setattr(audio_loopback, "set_mode", fake_set_mode)

    def fake_apply(card):
        b.gain_calls.append(card)
        before = b.gain()
        if b.pcm_settable:
            b.pcm_db = 0.0
        return before, b.gain()

    monkeypatch.setattr(hostgain, "apply", fake_apply)
    monkeypatch.setattr(hostgain, "read", lambda card: b.gain())
    monkeypatch.setattr(audio_loopback, "discover", lambda: [b.seen()])
    monkeypatch.setattr(audio_loopback, "select", lambda devices, _id=None: devices[0])

    b.queue: list[dict] = []
    b.next_verdict = lambda: b.queue.pop(0) if len(b.queue) > 1 else b.queue[0]
    b.queue.append({"verdict": "pass", "reasons": [], "signal_present": True,
                    "rms_dbfs": -18.0, "glitch_bursts_per_s": 0.0,
                    "tones": {"440": {"offset_pct": 0.1}}})
    return b


def make_ctx(tmp_path: Path, mode: UsbMode = UsbMode.AUDIO) -> base.Context:
    return base.Context(device=make_device(mode), workdir=tmp_path,
                        cancelled=threading.Event(), log=lambda s: None)


def run(ctx: base.Context, **params: Any) -> base.Report:
    # in2_gain off unless a test asks: the IN2 tests below stage their own CDC.
    kw = {"duration_hours": 0.0, "tone_seconds": 1.0, "interval": 0.0, "in2_gain": None}
    kw.update(params)
    return audio_loopback.SCENARIO.run(ctx, audio_loopback.Params(**kw), base.Progress())


def test_registered_and_defaults() -> None:
    assert get("audio_loopback") is audio_loopback.SCENARIO
    p = audio_loopback.Params()
    assert (p.duration_hours, p.cycles, p.tone_seconds, p.interval, p.freqs,
            p.warn_glitch_per_s, p.preset, p.ref_level, p.keep_mode) == (
        1.0, None, 10.0, 15.0, [440, 997, 3181], 5.0, "headphone", None, False)


def test_one_clean_cycle_passes(tmp_path: Path, bench: Bench) -> None:
    ctx = make_ctx(tmp_path)
    rep = run(ctx)
    assert rep.passed, rep.summary
    assert rep.exit_code == 0
    d = rep.data
    assert (d["cycles"], d["pass"], d["warn"], d["fail"], d["mode_reentries"]) == (1, 1, 0, 0, 0)
    assert "1 cycles" in rep.summary
    # the reference is written once, at the loop level for this preset
    assert bench.references[0]["level"] == audio_loopback.LOOP_REF_LEVEL
    assert bench.references[0]["freqs"] == [440.0, 997.0, 3181.0]
    # capture length is the tone plus the old script's 3 s of headroom
    assert bench.sessions[0]["seconds"] == 1.0 + audio_loopback.CAPTURE_PAD_S
    assert bench.sessions[0]["preset"] == "headphone"
    assert bench.sessions[0]["keep_mode"] is True
    # The host is the source: a resumed RT mixer would write the same I2S TX
    # beside the UAC bridge (2026-09-02 gate #6: 1.8 % flat, a glitch per DMA
    # block), so the scenario leaves it parked.
    assert bench.sessions[0]["resume_audio_tasks"] is False
    assert bench.played and bench.played[0][0].endswith("reference.wav")


def test_cycle_records_land_in_jsonl_and_artifacts(tmp_path: Path, bench: Bench) -> None:
    rep = run(make_ctx(tmp_path), cycles=2, duration_hours=1.0)
    roles = {a.role for a in rep.artifacts}
    assert {"reference", "log"} <= roles
    lines = [json.loads(x) for x in (tmp_path / "cycles.jsonl").read_text().splitlines()]
    assert [r["cycle"] for r in lines] == [1, 2]
    assert rep.data["cycles"] == 2


def test_multitone_fail_fails_the_run(tmp_path: Path, bench: Bench) -> None:
    bench.queue[0] = {"verdict": "fail", "reasons": ["tone 440 Hz missing (band power 0.0)"],
                      "signal_present": False, "rms_dbfs": -120.0}
    rep = run(make_ctx(tmp_path))
    assert not rep.passed
    assert rep.exit_code == 1
    assert "FAIL" in rep.summary
    assert rep.data["fail"] == 1


def test_warn_does_not_fail_the_run(tmp_path: Path, bench: Bench) -> None:
    bench.queue[0] = {"verdict": "warn", "reasons": ["glitches 7.0/s"], "signal_present": True,
                      "rms_dbfs": -20.0, "glitch_bursts_per_s": 7.0, "tones": {}}
    rep = run(make_ctx(tmp_path))
    assert rep.passed
    assert (rep.data["warn"], rep.data["fail"]) == (1, 0)


def test_warn_threshold_reaches_the_kernel(tmp_path: Path, bench: Bench) -> None:
    run(make_ctx(tmp_path), warn_glitch_per_s=2.5, freqs=[440])
    expected = bench.verdicts[0]["expected"]
    assert expected == {"freqs": [440.0], "warn_glitch_per_s": 2.5, "min_rms_dbfs": -45.0}


def test_mics_preset_uses_the_loud_reference(tmp_path: Path, bench: Bench) -> None:
    run(make_ctx(tmp_path), preset="mics")
    assert bench.references[0]["level"] == audio_loopback.MIC_REF_LEVEL


def test_explicit_ref_level_wins(tmp_path: Path, bench: Bench) -> None:
    run(make_ctx(tmp_path), ref_level=0.5)
    assert bench.references[0]["level"] == 0.5


def test_default_profile_is_switched_to_audio_and_back(tmp_path: Path, bench: Bench) -> None:
    bench.visible = make_device(UsbMode.DEFAULT)
    ctx = make_ctx(tmp_path, UsbMode.DEFAULT)
    rep = run(ctx)
    assert bench.modes == ["audio", "default"]
    assert rep.data["final_mode"] == "default"
    assert rep.data["restored"] is True


def test_keep_mode_leaves_the_board_in_the_audio_profile(tmp_path: Path, bench: Bench) -> None:
    rep = run(make_ctx(tmp_path), keep_mode=True)
    assert bench.modes == []
    assert rep.data["final_mode"] == "audio"


def test_a_vanished_card_is_counted_and_re_entered(tmp_path: Path, bench: Bench) -> None:
    # in the profile at the start, gone by the time the first cycle looks
    bench.sightings = [make_device(UsbMode.AUDIO)] + [make_device(UsbMode.DEFAULT)] * 2
    rep = run(make_ctx(tmp_path))
    assert rep.data["mode_reentries"] == 1
    assert bench.modes[0] == "audio"


def test_the_profile_is_restored_even_when_playback_fails(tmp_path: Path, bench: Bench) -> None:
    """A host that cannot play is the bench, not the board: one ENV verdict, not
    a failed cycle per attempt (a gate once logged 227 of them), and the profile
    still goes back to default on the way out."""
    bench.play_error = HilError(ENV, "aplay failed: device busy")
    ctx = make_ctx(tmp_path, UsbMode.DEFAULT)
    bench.visible = make_device(UsbMode.DEFAULT)
    rep = run(ctx)
    assert not rep.passed
    assert rep.exit_code == 2
    assert "device busy" in rep.summary
    assert bench.modes[-1] == "default"


def test_a_capture_that_never_starts_is_a_failed_cycle(tmp_path: Path, bench: Bench) -> None:
    bench.start_error = HilError(NOT_SUPPORTED, "sounddevice is unavailable")
    rep = run(make_ctx(tmp_path))
    assert rep.exit_code == 1
    assert rep.data["records"][0]["verdict"] == "fail"
    assert "capture did not start" in rep.data["records"][0]["reasons"][0]


def test_a_missing_audio_extra_is_an_environment_error(tmp_path: Path,
                                                       monkeypatch: pytest.MonkeyPatch,
                                                       bench: Bench) -> None:
    def no_player():
        raise HilError(NOT_SUPPORTED, "sounddevice is unavailable: no PortAudio",
                       hint="pip install 'crosspad-hil[audio]'")

    monkeypatch.setattr(audio_loopback, "default_player", no_player)
    rep = run(make_ctx(tmp_path))
    assert rep.exit_code == 2
    assert not rep.passed
    assert "sounddevice" in rep.summary


def test_unknown_preset_is_bad_args(tmp_path: Path, bench: Bench) -> None:
    with pytest.raises(HilError) as e:
        run(make_ctx(tmp_path), preset="nope")
    assert e.value.code == "BAD_ARGS"


def test_cancellation_stops_at_a_cycle_boundary(tmp_path: Path, bench: Bench) -> None:
    ctx = make_ctx(tmp_path)
    ctx.cancelled.set()
    rep = run(ctx, duration_hours=1.0)
    assert rep.data["cancelled"] is True
    assert rep.data["cycles"] == 0
    # nothing ran, so nothing passed: the run is not silently green
    assert not rep.passed


def test_ref_level_helper_follows_the_adc_input() -> None:
    presets = dict(capture_mod.PRESETS)
    assert audio_loopback.ref_level("headphone", presets) == audio_loopback.LOOP_REF_LEVEL
    assert audio_loopback.ref_level("speaker", presets) == audio_loopback.LOOP_REF_LEVEL
    assert audio_loopback.ref_level("mics", presets) == audio_loopback.MIC_REF_LEVEL
    assert audio_loopback.ref_level("line-in", presets) == audio_loopback.MIC_REF_LEVEL


def test_aplay_player_uses_pw_play_when_the_card_is_busy(tmp_path: Path,
                                                        monkeypatch: pytest.MonkeyPatch) -> None:
    """plughw is busy (PipeWire holds the CrossPad sink), so playback falls to
    pw-play — native PipeWire, not aplay's stalling pipewire plugin."""
    wav = tmp_path / "ref.wav"
    write_silence(wav)
    calls: list[list[str]] = []

    class Done:
        def __init__(self, rc: int, err: bytes = b"") -> None:
            self.returncode = rc
            self.stderr = err

    def fake_run(argv, **kw):
        calls.append(argv)
        if argv[0] == "aplay" and "plughw:Audio" in argv:
            return Done(1, b"aplay: audio open error: Device or resource busy")
        if argv[0] == "pw-play":
            return Done(0)
        return Done(0)

    monkeypatch.setattr(audio_loopback.shutil, "which", lambda name: "/usr/bin/" + name)
    player = audio_loopback.AplayPlayer(run=fake_run)
    player.play(make_device(), wav, timeout_s=5.0)   # must not raise
    assert ["pw-play", "--target", "crosspad_out"] == next(c for c in calls if c[0] == "pw-play")[:3]
    assert not any(c[0] == "aplay" and audio_loopback.PIPEWIRE_DEVICE in c for c in calls)


def test_aplay_player_falls_back_to_aplay_pipewire_without_pw_play(tmp_path: Path,
                                                                  monkeypatch: pytest.MonkeyPatch) -> None:
    """No pw-play on the host → the old aplay -D pipewire path still runs."""
    wav = tmp_path / "ref.wav"
    write_silence(wav)
    calls: list[list[str]] = []

    class Done:
        def __init__(self, rc: int, err: bytes = b"") -> None:
            self.returncode = rc
            self.stderr = err

    envs: list[dict] = []

    def fake_run(argv, **kw):
        calls.append(argv)
        envs.append(kw.get("env") or {})
        if argv[0] == "aplay" and "plughw:Audio" in argv:
            return Done(1, b"Device or resource busy")
        return Done(0)

    monkeypatch.setattr(audio_loopback.shutil, "which", lambda name: None)
    player = audio_loopback.AplayPlayer(run=fake_run)
    player.play(make_device(), wav, timeout_s=5.0)
    i = next(i for i, c in enumerate(calls) if c[0] == "aplay" and audio_loopback.PIPEWIRE_DEVICE in c)
    # The CrossPad's node, never the desktop's speakers.
    assert envs[i]["PIPEWIRE_NODE"] == "crosspad_out"
    assert "PIPEWIRE_NODE" not in envs[0]


# --------------------------------------------------------------------------- host gain

def test_host_gain_is_set_to_zero_db_and_recorded(tmp_path: Path, bench: Bench) -> None:
    """The bench: the session left the card's PCM at -20 dB; the run puts it at 0."""
    bench.pcm_db = -20.0
    rep = run(make_ctx(tmp_path))
    assert rep.passed, rep.summary
    assert bench.gain_calls == ["Audio"]
    first = rep.data["host_gain"][0]
    assert first["when"] == "start"
    assert first["before"]["deviations"] == ["ALSA 'PCM',0 playback -20.0 dB"]
    assert first["after"]["deviations"] == []
    assert rep.data["host_gain_final"]["worst_db"] == 0.0


def test_a_host_gain_that_will_not_reach_zero_db_is_exit_2(tmp_path: Path,
                                                         bench: Bench) -> None:
    bench.pcm_db, bench.pcm_settable = -20.0, False
    rep = run(make_ctx(tmp_path))
    assert rep.exit_code == 2 and not rep.passed
    assert "host gain ALSA 'PCM',0 playback -20.0 dB" in rep.summary
    assert rep.data["cycles"] == 0          # nothing measured against a wrong gain
    assert bench.modes == ["default"]      # the profile still goes back on the way out


def test_level_failures_under_a_drifted_host_gain_are_the_hosts(tmp_path: Path,
                                                              monkeypatch: pytest.MonkeyPatch,
                                                              bench: Bench) -> None:
    """32/32 'level too low' at -45.7 dBFS with a clean signal: the session put
    the stored -20 dB back mid-run. Exit 2 naming the gain, not a firmware FAIL."""
    bench.queue[0] = {"verdict": "fail", "reasons": ["level too low: -45.7 dBFS < -45"],
                      "signal_present": True, "rms_dbfs": -45.7, "glitch_bursts_per_s": 0.0}
    monkeypatch.setattr(hostgain, "read", lambda card: hostgain.HostGain(card="Audio", controls=[
        {"name": "'PCM',0", "direction": "playback", "raw": 30, "percent": 60, "db": -20.0,
         "on": True}]))
    rep = run(make_ctx(tmp_path))
    assert rep.exit_code == 2
    assert "host gain ALSA 'PCM',0 playback -20.0 dB at the end of the run" in rep.summary
    assert "1 'level too low' cycle(s) are the host's" in rep.summary


def test_a_level_failure_at_unity_gain_is_still_the_firmwares(tmp_path: Path,
                                                            bench: Bench) -> None:
    bench.queue[0] = {"verdict": "fail", "reasons": ["level too low: -45.7 dBFS < -45"],
                      "signal_present": True, "rms_dbfs": -45.7, "glitch_bursts_per_s": 0.0}
    rep = run(make_ctx(tmp_path))
    assert rep.exit_code == 1 and "FAIL" in rep.summary


def test_host_gain_is_set_again_after_a_re_entry(tmp_path: Path, bench: Bench) -> None:
    bench.sightings = [make_device(UsbMode.AUDIO)] + [make_device(UsbMode.DEFAULT)] * 2
    rep = run(make_ctx(tmp_path))
    assert rep.data["mode_reentries"] == 1
    assert [h["when"] for h in rep.data["host_gain"]] == ["start", "re-entry 1"]


def test_no_host_gain_touches_nothing(tmp_path: Path, bench: Bench) -> None:
    bench.pcm_db = -20.0
    rep = run(make_ctx(tmp_path), host_gain=False)
    assert bench.gain_calls == [] and rep.data["host_gain"] == []
    assert rep.data["host_gain_final"] is None


# --------------------------------------------------------------------------- IN2 (jack) gain

class Board:
    """INGAIN over a fake CDC: the board keeps what it was last told (NVS)."""

    def __init__(self, in2: int = 0, clamp_to: int | None = None) -> None:
        self.gain = {"IN1": 24, "IN2": in2}
        self.clamp_to = clamp_to
        self.calls: list[tuple] = []
        self.fail_restore = False
        self.supported = True

    def ingain(self, link: Any, input_index: int | None = None, db: int | None = None) -> dict:
        self.calls.append((input_index, db))
        if not self.supported:
            raise HilError(NOT_SUPPORTED, "INGAIN: device replied ERR INGAIN none")
        if input_index is not None:
            if self.fail_restore and len(self.calls) > 2:
                raise HilError(ENV, "no CDC")
            self.gain[f"IN{input_index + 1}"] = db if self.clamp_to is None else min(db, self.clamp_to)
        return dict(self.gain)


class Link:
    def close(self) -> None:
        pass


@pytest.fixture
def board(monkeypatch: pytest.MonkeyPatch, bench: Bench) -> Board:
    b = Board()
    from crosspad_hil import verbs
    monkeypatch.setattr(verbs, "ingain", b.ingain)
    monkeypatch.setattr(audio_loopback, "_open_cdc", lambda ctx: Link())
    return b


def test_in2_is_set_for_the_run_and_put_back(tmp_path: Path, bench: Bench, board: Board) -> None:
    """dcfac16 brought the jack up at 0 dB: the loop came back at -45.7 dBFS. The run
    measures at +24 dB (-20.6 on the bench) and leaves the board's 0 dB as it was."""
    bench.visible = make_device(UsbMode.DEFAULT)
    rep = run(make_ctx(tmp_path, UsbMode.DEFAULT), in2_gain=24)
    assert rep.passed, rep.summary
    d = rep.data
    assert (d["ingain_before"]["IN2"], d["ingain_during"]["IN2"], d["ingain_after"]["IN2"]) \
        == (0, 24, 0)
    assert d["ingain_restored"] is True
    assert board.calls == [(None, None), (1, 24), (1, 0)]
    assert d["min_rms_dbfs"] == -45.0
    assert bench.verdicts[0]["expected"]["min_rms_dbfs"] == -45.0
    assert bench.modes == ["audio", "default"]


def test_the_level_floor_follows_the_gain(tmp_path: Path, bench: Bench, board: Board) -> None:
    bench.visible = make_device(UsbMode.DEFAULT)
    rep = run(make_ctx(tmp_path, UsbMode.DEFAULT), in2_gain=12)
    assert rep.data["min_rms_dbfs"] == -57.0
    assert bench.verdicts[0]["expected"]["min_rms_dbfs"] == -57.0


def test_starting_in_the_audio_profile_sets_it_over_cdc_first(tmp_path: Path, bench: Bench,
                                                             board: Board) -> None:
    rep = run(make_ctx(tmp_path), in2_gain=24)
    assert rep.passed, rep.summary
    assert bench.modes == ["default", "audio", "default"]
    assert rep.data["ingain_after"]["IN2"] == 0


def test_in2_is_put_back_even_when_the_run_fails(tmp_path: Path, bench: Bench,
                                                 board: Board) -> None:
    bench.play_error = HilError(ENV, "aplay failed: device busy")
    bench.visible = make_device(UsbMode.DEFAULT)
    rep = run(make_ctx(tmp_path, UsbMode.DEFAULT), in2_gain=24)
    assert rep.exit_code == 2 and "device busy" in rep.summary
    assert board.gain["IN2"] == 0 and rep.data["ingain_restored"] is True


def test_in2_is_put_back_when_cancelled(tmp_path: Path, bench: Bench, board: Board) -> None:
    bench.visible = make_device(UsbMode.DEFAULT)
    ctx = make_ctx(tmp_path, UsbMode.DEFAULT)
    ctx.cancelled.set()
    run(ctx, in2_gain=24)
    assert board.gain["IN2"] == 0


def test_a_restore_that_fails_is_reported_not_swallowed(tmp_path: Path, bench: Bench,
                                                        board: Board) -> None:
    board.fail_restore = True
    bench.visible = make_device(UsbMode.DEFAULT)
    rep = run(make_ctx(tmp_path, UsbMode.DEFAULT), in2_gain=24)
    assert rep.exit_code == 2 and not rep.passed
    assert "IN2 gain left at 24 dB, not restored to 0 dB" in rep.summary
    assert rep.data["ingain_restored"] is False


def test_keep_mode_goes_back_to_audio_after_the_restore(tmp_path: Path, bench: Bench,
                                                        board: Board) -> None:
    rep = run(make_ctx(tmp_path), in2_gain=24, keep_mode=True)
    assert bench.modes == ["default", "audio", "default", "audio"]
    assert rep.data["final_mode"] == "audio" and board.gain["IN2"] == 0


def test_a_gain_the_board_clamps_is_exit_2(tmp_path: Path, bench: Bench, board: Board) -> None:
    board.clamp_to = 21
    bench.visible = make_device(UsbMode.DEFAULT)
    rep = run(make_ctx(tmp_path, UsbMode.DEFAULT), in2_gain=24)
    assert rep.exit_code == 2 and "IN2 gain is 21 dB after asking for 24 dB" in rep.summary
    assert rep.data["cycles"] == 0
    # the clamped value was written all the same, and is taken back
    assert board.calls[-1] == (1, 0) and rep.data["ingain_restored"] is True


def test_firmware_without_ingain_measures_at_the_default_floor(tmp_path: Path, bench: Bench,
                                                               board: Board) -> None:
    board.supported = False
    bench.visible = make_device(UsbMode.DEFAULT)
    rep = run(make_ctx(tmp_path, UsbMode.DEFAULT), in2_gain=24)
    assert rep.passed and "INGAIN" in rep.data["ingain_note"]
    assert rep.data["min_rms_dbfs"] == -45.0 and rep.data["ingain_restored"] is None


def test_the_mic_path_leaves_the_jack_alone(tmp_path: Path, bench: Bench, board: Board) -> None:
    run(make_ctx(tmp_path), in2_gain=24, preset="mics")
    assert board.calls == []


def test_a_pin_a_killed_run_left_is_kept_first(tmp_path: Path, bench: Bench, board: Board) -> None:
    """A run killed at +24 dB left IN2 there and its promise in the journal:
    the next run puts 0 back before it reads "before", and leaves 0 behind."""
    from crosspad_hil import pins
    board.gain["IN2"] = 24
    pins.record("dev_ab12", pins.IN2, 0)
    bench.visible = make_device(UsbMode.DEFAULT)
    rep = run(make_ctx(tmp_path, UsbMode.DEFAULT), in2_gain=24)
    assert rep.data["ingain_recovered"] == {"in2": 0}
    assert rep.data["ingain_before"]["IN2"] == 0 and board.gain["IN2"] == 0
    assert pins.pending("dev_ab12") == {}


def test_a_cycle_the_host_lost_capture_of_is_not_a_fail(tmp_path: Path, bench: Bench,
                                                         monkeypatch: pytest.MonkeyPatch) -> None:
    def lossy(self: FakeSession) -> FakeResult:
        result = FakeResult(wav=str(self.kw["out"]))
        result.health = {"ok": False, "missed_s": 1.2, "max_gap_ms": 20.0, "overruns": 0}
        return result

    monkeypatch.setattr(FakeSession, "stop", lossy)
    rep = run(make_ctx(tmp_path))
    assert not rep.passed and rep.exit_code == 2, rep.summary
    assert (rep.data["fail"], rep.data["host_lost"]) == (0, 1)
    assert "capture not measured" in rep.summary and "host lost 1.2 s" in rep.data["records"][0]["reasons"][0]
