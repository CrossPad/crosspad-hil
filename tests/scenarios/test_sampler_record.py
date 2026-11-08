"""Scenario `sampler_record` against fakes — no board, no sound card, no radio.

The CDC fake models the parts of hil_control.cpp the setup leans on; the capture
session, the pad transport and the onset analysis are replaced at their module
seams, so what is actually exercised here is the scenario's own logic: the order
of the setup (everything over CDC *before* the profile switch), the per-step
scoring, the verdict rules, and the promise that the capture is stopped — and
therefore the USB profile restored — whatever goes wrong.
"""
from __future__ import annotations

import threading
import wave
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

import pytest

from crosspad_hil import cdc as cdc_mod
from crosspad_hil.cdc import CdcLink
from crosspad_hil.devices import AudioCardInfo, Device, MidiPortInfo, Ports, SerialPortInfo, UsbMode
from crosspad_hil.errors import ENV, NO_DEVICE, NOT_SUPPORTED, HilError
from crosspad_hil.scenarios import base, get, sampler_record
from tests.fakes import FakeSerial

KITS = ["Basic", "DRUMS", "Perc"]
CDC_PORT = SerialPortInfo(path="/dev/ttyACM0", vid=0x303A, pid=0x3456,
                          serial="CP-1", product="CrossPad", location="1-2.1")
CONSOLE_PORT = SerialPortInfo(path="/dev/ttyACM1", vid=0x0483, pid=0x5740,
                              serial="STM-1", product="CrossPad", location="1-2.2")
ESP_MIDI = MidiPortInfo(name="Crosspad MIDI 1", rtmidi_out=2, rtmidi_in=2,
                        alsa_hw="hw:4,0,0", rawmidi="/dev/snd/midiC4D0")
UAC2 = AudioCardInfo(name="Crosspad Audio", sounddevice_index=5, alsa_id="hw:5")


class SamplerDevice(FakeSerial):
    """CDC endpoint: kits, the launcher, the pad counters, the BLE role."""

    def __init__(self) -> None:
        super().__init__(timeout=0.01)      # the reader thread sleeps this long per idle read
        self.current = 1
        self.apps = ["Sampler", "Mixer"]
        self.running: str | None = "Sampler"
        self.press = 0
        self.played = 0
        self.notes = {p: 36 + p for p in range(16)}
        self.ble_mode: int | None = None
        self.ble_self = "AA:BB:CC:DD:EE:FF"

    def write(self, data: bytes) -> int:
        for raw in data.decode("utf-8", "replace").split("\n"):
            cmd = raw.strip()
            if not cmd:
                continue
            self.written.append(cmd)
            if cmd == "APP_LIST":
                self.feed([f"APPS: {','.join(self.apps)} running={self.running or '-'}"])
            elif cmd.startswith("APP_START "):
                self.running = cmd.split(" ", 1)[1]
                self.feed(["OK"])
            elif cmd == "KIT_LIST":
                body = ",".join(f"{i}:{n}" for i, n in enumerate(KITS))
                self.feed([f"KITS: {body} current={self.current}"])
            elif cmd == "KIT_STATUS":
                name = KITS[self.current] if self.current >= 0 else "-"
                self.feed([f"KITSTATUS: current={self.current} loading=0 pending=-1 name={name}"])
            elif cmd.startswith("KIT_LOAD "):
                self.current = int(cmd.split(" ", 1)[1])
                self.feed(["OK"])
            elif cmd == "PAD_STATS_RESET":
                self.press = self.played = 0
                self.feed(["OK"])
            elif cmd == "PAD_STATS":
                self.feed([f"PADSTATS: press={self.press} release={self.press} "
                           f"played={self.played} freeslots=7"])
            elif cmd == "PAD_NOTES":
                body = " ".join(f"{p}:{n}" for p, n in sorted(self.notes.items()))
                self.feed([f"PADNOTES: {body}"])
            elif cmd.startswith("BLE_START"):
                parts = cmd.split()
                self.ble_mode = int(parts[1]) if len(parts) > 1 else None
                self.feed(["OK"])
            elif cmd == "BLE_STATUS":
                self.feed([f"BLE: supported=1 running=1 state=connected mode=server "
                           f"self={self.ble_self} peer=- itvl=12 rx_msg=0 tx_err=0"])
            else:
                self._rx.append((cmd + "\r\n").encode())
        return len(data)


def device_in(mode: UsbMode) -> Device:
    if mode is UsbMode.AUDIO:
        ports = Ports(cdc=None, console=CONSOLE_PORT, esp_midi=ESP_MIDI, uac2=UAC2)
    else:
        ports = Ports(cdc=CDC_PORT, console=CONSOLE_PORT, esp_midi=ESP_MIDI, uac2=None)
    return Device(id="dev_ab12", serial="CP-1", usb_mode=mode, ports=ports)


@dataclass
class FakeContext(base.Context):
    cdc_fake: SamplerDevice = None  # type: ignore[assignment]

    def open_cdc(self) -> CdcLink:
        if self.device.ports.cdc is None:
            raise HilError(NO_DEVICE, "no CDC in this profile")
        fake = self.cdc_fake
        link = CdcLink(self.device.ports.cdc.path, serial_factory=lambda path, **kw: fake)
        link.open()
        self.opened.append(link)
        return link


@dataclass
class FakeResult:
    """What crosspad_hil.capture.CaptureResult gives the scenario back."""

    wav: str
    seconds: float = 4.0
    peak_dbfs: float = -6.0
    silent: bool = False
    hint: str | None = None
    health: dict = field(default_factory=lambda: {"ok": True, "detail": "clean"})

    def to_dict(self) -> dict:
        return {"wav": self.wav, "seconds": self.seconds, "peak_dbfs": self.peak_dbfs,
                "silent": self.silent, "hint": self.hint}


class FakeCapture:
    """Stands in for CaptureSession: the profile switch, the WAV and the restore."""

    def __init__(self, device: Device, **kw: Any) -> None:
        self.device = device
        self.kw = kw
        self.started = False
        self.stopped = False
        self.start_error: HilError | None = None
        self.result_kw: dict[str, Any] = {}

    def start(self) -> dict:
        if self.start_error is not None:
            raise self.start_error
        self.started = True
        path = Path(self.kw["out"])
        path.parent.mkdir(parents=True, exist_ok=True)
        with wave.open(str(path), "wb") as w:      # a real (silent) WAV: analyze is faked
            w.setnchannels(2)
            w.setsampwidth(2)
            w.setframerate(48000)
            w.writeframes(b"\x00" * 4)
        self.device = device_in(UsbMode.AUDIO)
        return {"wav": str(path), "preset": self.kw["preset"], "backend": "fake"}

    def stop(self) -> FakeResult:
        self.stopped = True
        if not self.kw.get("keep_mode"):
            self.device = device_in(UsbMode.DEFAULT)
        return FakeResult(wav=str(self.kw["out"]), **self.result_kw)


class FakeTransport:
    """A pad transport that records what the scheduler asked for."""

    name = "fake"

    def __init__(self, fail_on_press: bool = False) -> None:
        self.presses: list[tuple[int, int]] = []
        self.releases: list[int] = []
        self.closed = False
        self.fail_on_press = fail_on_press

    def press(self, pad: int, vel: int) -> None:
        if self.fail_on_press:
            raise OSError("the device left the bus")
        self.presses.append((pad, vel))

    def release(self, pad: int) -> None:
        self.releases.append(pad)

    def close(self) -> None:
        self.closed = True


class CancelInsideTheTake(threading.Event):
    """Cancels the run from inside the recording, which is where a Ctrl+C lands.

    A cancel raised anywhere earlier is a cancelled run and reported as one; the
    long wait is the take, and only there does the old script's "keep the audio
    and report" behaviour apply.
    """

    def wait(self, timeout: float | None = None) -> bool:
        if timeout is not None and timeout >= 1.0:
            self.set()
            return True
        return super().wait(timeout)


class FakeBle:
    """Just enough BleMidiLink for BlePadTransport."""

    def __init__(self) -> None:
        self.notes: list[tuple[bool, int, int]] = []
        self.disconnected = False

    def send_note(self, on: bool, note: int, vel: int = 100) -> None:
        self.notes.append((on, note, vel))

    def disconnect(self) -> None:
        self.disconnected = True


@pytest.fixture(autouse=True)
def fast(monkeypatch: pytest.MonkeyPatch) -> None:
    """Collapse every human-scale wait the old script carried."""
    for name in ("LEAD_S", "LEAD_GAP_S", "STREAM_SETTLE_S", "TAIL_S", "APP_START_SETTLE_S"):
        monkeypatch.setattr(sampler_record, name, 0.0)
    # …and the settle a real CDC endpoint needs after enumerating, twice per run.
    monkeypatch.setattr(cdc_mod, "OPEN_SETTLE_S", 0.0)


@pytest.fixture
def bench(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """Context + the three seams, all wired to fakes."""

    class Bench:
        def __init__(self) -> None:
            self.cdc = SamplerDevice()
            self.ctx = FakeContext(device=device_in(UsbMode.DEFAULT), workdir=tmp_path,
                                   cancelled=threading.Event(), log=lambda s: None,
                                   cdc_fake=self.cdc)
            self.capture: FakeCapture | None = None
            self.capture_error: HilError | None = None
            self.transport = FakeTransport()
            self.transport_error: HilError | None = None
            self.ble = FakeBle()
            self.ble_address: str | None = None
            self.onset: dict[str, Any] = {"missed": [], "matched": 0, "verdict": "pass"}
            self.analyzed: list[tuple[Path, list[float], dict]] = []

        def make_capture(self, device: Device, **kw: Any) -> FakeCapture:
            self.capture = FakeCapture(device, **kw)
            if self.capture_error is not None:
                self.capture.start_error = self.capture_error
            return self.capture

        def make_transport(self, device: Device, kind: str) -> FakeTransport:
            if self.transport_error is not None:
                raise self.transport_error
            self.transport.kind = kind  # type: ignore[attr-defined]
            return self.transport

        def make_ble(self, address: str | None = None, **kw: Any) -> FakeBle:
            self.ble_address = address
            return self.ble

        def analyze(self, wav: Path, expected: list[float], **kw: Any) -> dict:
            self.analyzed.append((wav, list(expected), dict(kw)))
            return dict(self.onset)

        def run(self, **params: Any) -> base.Report:
            return sampler_record.SCENARIO.run(
                self.ctx, sampler_record.Params(**params), base.Progress())  # type: ignore[arg-type]

    b = Bench()
    monkeypatch.setattr(sampler_record, "open_capture", b.make_capture)
    monkeypatch.setattr(sampler_record, "open_pad_transport", b.make_transport)
    monkeypatch.setattr(sampler_record, "open_ble_link", b.make_ble)
    monkeypatch.setattr(sampler_record, "analyze_wav", b.analyze)
    return b


# --------------------------------------------------------------------------- unit


def test_registered_and_defaults() -> None:
    assert get("sampler_record") is sampler_record.SCENARIO
    p = sampler_record.Params()
    assert (p.mode, p.stim, p.preset, p.min_audible, p.fail_rate) == (
        "speedtest", "usb", "headphone", 0.9, 16.0)
    assert p.rates == [2, 4, 6, 8, 12, 16, 20, 25]
    assert (p.step, p.gap, p.pads, p.velocity, p.gate_ms) == (3.0, 1.0, "0-15", 110, 30)
    assert p.keep_audio_mode is False and p.duration == 0.0


def test_build_schedule_lays_out_the_lead_in_then_the_ramp(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(sampler_record, "LEAD_GAP_S", 1.0)   # the `fast` fixture zeroed it
    pattern, steps, total = sampler_record.build_schedule(
        [2.0, 4.0], step_s=1.0, gap_s=0.5, pads=[0, 1], gate_ms=30.0, velocity=110, lead_s=0.5)
    # index 0 is the lead-in hit: the recording's time zero, and part of the grid
    assert pattern[0]["t_ms"] == 500.0
    assert [s["n"] for s in steps] == [2, 4]
    assert steps[0]["start"] == pytest.approx(1.5)          # lead_s + LEAD_GAP_S
    assert steps[0]["times"] == pytest.approx([1.5, 2.0])
    assert steps[1]["start"] == pytest.approx(3.0)          # end of step 1 + gap
    # indices are what maps one whole-take analysis back onto per-step ratios
    assert steps[0]["indices"] == [1, 2] and steps[1]["indices"] == [3, 4, 5, 6]
    assert len(pattern) == 7
    assert total == pytest.approx(4.5)
    # the pads round-robin across step boundaries, as the old script's k did
    assert [e["pad"] for e in pattern] == [0, 0, 1, 0, 1, 0, 1]


def test_onset_windows_shrink_for_a_fast_ramp() -> None:
    from crosspad_hil import analyze

    slow = sampler_record.onset_windows([1.0, 2.0])        # 250 ms is wider than the defaults
    assert slow == {"refractory_s": analyze.ONSET_REFRACTORY_S, "tol_s": analyze.ONSET_TOL_S}
    quick = sampler_record.onset_windows([2.0, 25.0])
    assert quick["refractory_s"] == pytest.approx(0.02)    # half of a 40 ms period
    assert quick["tol_s"] == pytest.approx(0.02)


def test_forgiving_transport_counts_a_refused_write() -> None:
    inner = FakeTransport(fail_on_press=True)
    tr = sampler_record.ForgivingTransport(inner)
    tr.press(1, 100)
    tr.release(1)
    tr.close()
    assert tr.failures == 1
    assert inner.releases == [1] and inner.closed


def test_ble_pad_transport_uses_the_kit_note_map() -> None:
    ble = FakeBle()
    tr = sampler_record.BlePadTransport(ble, {0: 60})
    tr.press(0, 90)
    tr.press(1, 90)          # not in the map: falls back to 36 + pad
    tr.release(0)
    assert ble.notes == [(True, 60, 90), (True, 37, 90), (False, 60, 100)]
    tr.close()
    assert ble.disconnected


# --------------------------------------------------------------------------- speedtest


def test_speedtest_passes_and_leaves_the_board_where_it_found_it(bench) -> None:
    rep = bench.run(kit_name="drums", rates=[8], step=0.25, gap=0.0, pads="0-1", gate_ms=10)
    assert rep.passed, rep.summary
    assert rep.exit_code == 0
    d = rep.data
    assert (d["kit"], d["kit_name"]) == (1, "DRUMS")
    assert d["steps"] == [{"rate": 8.0, "start": 0.0, "end": 0.25, "injected": 2,
                           "audible": 2, "ratio": 1.0}]
    assert d["audible"] == 2
    assert d["hits_planned"] == 3           # the lead-in plus two step hits
    assert d["hits_sent"] == 3
    assert d["failed_rates"] == []
    # the setup all happened over CDC, before the profile switch
    assert bench.cdc.written.index("PAD_STATS_RESET") < len(bench.cdc.written) - 1
    assert bench.capture.started and bench.capture.stopped
    assert bench.capture.kw["preset"] == "headphone"
    assert bench.capture.kw["keep_mode"] is False
    assert bench.transport.closed and len(bench.transport.presses) == 3
    # the counters are read once CDC is back, and reported rather than asserted on
    assert d["pad_stats"] == {"press": 0, "release": 0, "played": 0, "freeslots": 7}
    assert [a.role for a in rep.artifacts] == ["capture"]
    assert Path(rep.artifacts[0].path).is_file()


def test_analysis_gets_the_whole_grid_and_the_narrowed_windows(bench) -> None:
    bench.run(rates=[4, 20], step=0.25, gap=0.0, pads="0-1", gate_ms=10)
    wav, expected, kw = bench.analyzed[0]
    assert Path(wav).is_file()
    assert expected[0] == 0.0                       # the lead-in is index 0 of the grid
    assert len(expected) == 1 + 1 + 5
    assert kw["refractory_s"] == pytest.approx(0.025)
    assert kw["tol_s"] == pytest.approx(0.025)


@pytest.mark.parametrize("rates, missed_step_rate, expect_pass", [
    ([8, 20], 8.0, False),      # a step at or below --fail-rate must sound
    ([8, 20], 20.0, True),      # above it, missing hits are reported and tolerated
])
def test_only_steps_at_or_below_fail_rate_have_to_sound(bench, rates, missed_step_rate,
                                                        expect_pass) -> None:
    # index 0 is the lead-in; the 8/s step is 1..2, the 20/s step is 3..7
    bench.onset = {"missed": [1] if missed_step_rate == 8.0 else [3, 4, 5, 6, 7]}
    rep = bench.run(rates=rates, step=0.25, gap=0.0, pads="0-1", gate_ms=10, fail_rate=16.0)
    assert rep.passed is expect_pass
    failed = rep.data["failed_rates"]
    assert failed == ([] if expect_pass else [8.0])
    if not expect_pass:
        assert rep.exit_code == 1
        assert "8/s" in rep.summary


def test_nothing_audible_at_all_names_the_route(bench) -> None:
    bench.onset = {"missed": [0, 1, 2]}
    rep = bench.run(rates=[8], step=0.25, gap=0.0, pads="0-1", gate_ms=10)
    assert not rep.passed and rep.exit_code == 1
    assert "wrong capture route" in rep.summary
    assert rep.data["audible"] == 0


def test_a_silent_capture_fails_with_the_capture_hint(bench) -> None:
    bench.onset = {"missed": []}

    def silent(device, **kw):
        cap = bench.make_capture(device, **kw)
        cap.result_kw = {"silent": True, "peak_dbfs": -99.0,
                         "hint": "check that the RT mixer was resumed (SysEx 0x1D 06 01)"}
        return cap

    sampler_record.open_capture = silent  # type: ignore[assignment]
    try:
        rep = bench.run(rates=[8], step=0.25, gap=0.0, pads="0-1", gate_ms=10)
    finally:
        sampler_record.open_capture = bench.make_capture  # type: ignore[assignment]
    assert not rep.passed and rep.exit_code == 1
    assert "0x1D 06 01" in rep.summary


def test_an_unhealthy_capture_fails_even_when_every_hit_sounded(bench) -> None:
    def stalled(device, **kw):
        cap = bench.make_capture(device, **kw)
        cap.result_kw = {"health": {"ok": False, "detail": "captured 1.0 s of 4.0 s"}}
        return cap

    sampler_record.open_capture = stalled  # type: ignore[assignment]
    try:
        rep = bench.run(rates=[8], step=0.25, gap=0.0, pads="0-1", gate_ms=10)
    finally:
        sampler_record.open_capture = bench.make_capture  # type: ignore[assignment]
    assert not rep.passed and rep.exit_code == 1
    assert "capture unhealthy" in rep.summary


def test_a_refused_pad_message_is_counted_not_raised(bench) -> None:
    bench.transport.fail_on_press = True
    rep = bench.run(rates=[8], step=0.25, gap=0.0, pads="0-1", gate_ms=10)
    # The take survives — and so does the profile restore, which is the point.
    assert bench.capture.stopped
    assert rep.data["send_failures"] == 3
    assert not rep.passed and "could not be sent" in rep.summary


# --------------------------------------------------------------------------- BLE stimulus


def test_ble_stimulus_asks_cdc_for_the_role_and_the_note_map_first(bench) -> None:
    bench.cdc.notes = {p: 48 + p for p in range(16)}
    rep = bench.run(stim="ble", rates=[8], step=0.25, gap=0.0, pads="0-1", gate_ms=10)
    assert rep.passed, rep.summary
    # Server mode and PAD_NOTES are the last chance: CDC is gone after the switch.
    assert "BLE_START 1" in bench.cdc.written
    assert bench.cdc.written.index("BLE_START 1") < bench.cdc.written.index("PAD_STATS_RESET")
    assert "PAD_NOTES" in bench.cdc.written
    assert rep.data["pad_notes"][0] == 48
    # …and the radio connects to *this* board, by the address it reported itself
    assert bench.ble_address == "AA:BB:CC:DD:EE:FF"
    assert [n for on, n, _v in bench.ble.notes if on] == [48, 48, 49]
    assert bench.ble.disconnected


def test_ble_stimulus_without_a_note_map_is_an_environment_failure(bench) -> None:
    bench.cdc.notes = {}
    rep = bench.run(stim="ble", rates=[8], step=0.25, gap=0.0, pads="0-1", gate_ms=10)
    assert not rep.passed and rep.exit_code == 2
    assert "PAD_NOTES" in rep.summary
    assert bench.capture is None          # nothing was recorded, nothing to restore


# --------------------------------------------------------------------------- live


def test_live_mode_records_for_the_duration_and_leaves_the_pads_alone(bench) -> None:
    rep = bench.run(mode="live", duration=0.05)
    assert rep.passed, rep.summary
    assert rep.data["steps"] == [] and rep.data["hits_sent"] == 0
    assert bench.transport.presses == []
    assert bench.analyzed == []           # live mode does not score anything
    assert bench.capture.started and bench.capture.stopped


def test_live_mode_treats_a_cancel_as_the_end_of_the_take(bench) -> None:
    """The old script caught Ctrl+C, kept the audio and still reported."""
    bench.ctx.cancelled = CancelInsideTheTake()
    rep = bench.run(mode="live", duration=30.0)
    assert rep.passed and rep.data["interrupted"] is True
    assert "interrupted" in rep.summary
    assert bench.capture.stopped


def test_keep_audio_mode_skips_the_counter_read(bench) -> None:
    rep = bench.run(mode="live", duration=0.05, keep_audio_mode=True)
    assert rep.passed
    assert bench.capture.kw["keep_mode"] is True
    assert rep.data["pad_stats"] is None      # no CDC to ask, by request


# --------------------------------------------------------------------------- environment


def test_a_host_without_an_audio_stack_is_exit_2(bench) -> None:
    bench.capture_error = HilError(NOT_SUPPORTED, "sounddevice is unavailable",
                                   hint="pip install 'crosspad-hil[audio]'")
    rep = bench.run(mode="live", duration=0.05)
    assert not rep.passed and rep.exit_code == 2
    assert "sounddevice" in rep.summary


def test_no_sampler_in_the_launcher_is_exit_2(bench) -> None:
    bench.cdc.apps = ["Mixer"]
    bench.cdc.running = None
    rep = bench.run(mode="live", duration=0.05)
    assert not rep.passed and rep.exit_code == 2
    assert "Sampler" in rep.summary
    assert bench.capture is None


def test_an_unknown_kit_name_lists_what_is_there(bench) -> None:
    rep = bench.run(kit_name="nope")
    assert not rep.passed and rep.exit_code == 2
    assert "nope" in rep.summary and "DRUMS" in (rep.data["failures"][0] + rep.summary)


def test_a_transport_that_cannot_open_still_stops_the_capture(bench) -> None:
    bench.transport_error = HilError(NO_DEVICE, "no ESP MIDI port")
    rep = bench.run(rates=[8], step=0.25, gap=0.0, pads="0-1", gate_ms=10)
    assert not rep.passed and rep.exit_code == 2
    assert bench.capture.started and bench.capture.stopped
    assert bench.ctx.device.usb_mode is UsbMode.DEFAULT


def test_a_bad_mode_is_bad_args(bench) -> None:
    with pytest.raises(HilError) as e:
        bench.run(mode="nonsense")
    assert e.value.code == "BAD_ARGS"


def test_the_audio_profile_is_left_when_the_run_starts_in_it(bench, monkeypatch) -> None:
    """No CDC means the board is already in the audio profile — SysEx is the way back."""
    calls: list[tuple[str, str]] = []

    def fake_set_mode(device: Device, mode: UsbMode, **kw: Any) -> Device:
        calls.append((device.id, mode.value))
        return device_in(mode)

    monkeypatch.setattr(sampler_record, "set_mode", fake_set_mode)
    bench.ctx.device = device_in(UsbMode.AUDIO)
    rep = bench.run(mode="live", duration=0.05)
    assert rep.passed, rep.summary
    assert calls == [("dev_ab12", "default")]


def test_env_report_keeps_what_the_run_already_learned(bench) -> None:
    bench.capture_error = HilError(ENV, "no CrossPad UAC2 capture endpoint appeared")
    rep = bench.run(kit_name="DRUMS", mode="live", duration=0.05)
    assert rep.exit_code == 2
    assert rep.data["kit_name"] == "DRUMS"
    assert rep.data["failures"] == ["ENV: no CrossPad UAC2 capture endpoint appeared"]
    assert replace(bench.ctx.device, board_rev=None).usb_mode is UsbMode.DEFAULT


def test_hits_the_host_lost_are_not_charged_to_the_device(bench) -> None:
    """Every hit "missed" -- because the host dropped the seconds they were in."""
    def lossy(device, **kw):
        cap = bench.make_capture(device, **kw)
        cap.result_kw = {"health": {"ok": False, "missed_s": 2.0, "max_gap_ms": 30.0,
                                    "overruns": 0, "detail": "captured 2.0 s of 4.0 s"}}
        return cap

    bench.onset = {"missed": [0, 1, 2, 3, 4, 5, 6, 7]}
    sampler_record.open_capture = lossy  # type: ignore[assignment]
    try:
        rep = bench.run(rates=[8], step=0.25, gap=0.0, pads="0-1", gate_ms=10)
    finally:
        sampler_record.open_capture = bench.make_capture  # type: ignore[assignment]
    assert not rep.passed and rep.exit_code == 2, rep.summary
    assert "capture not measured — host lost 2.0 s" in rep.summary
    assert bench.analyzed == [] and rep.data["failed_rates"] == []
