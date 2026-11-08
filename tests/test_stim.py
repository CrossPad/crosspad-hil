"""Pad stimulus: the schedule, the throttle, and the loss accounting. No hardware.

The transport and the counters are both injectable, so a whole run — schedule, backoff,
release of held pads, PAD_STATS delta — is exercised in a fraction of a second.
"""
from __future__ import annotations

import io
import json
import threading
import uuid
from pathlib import Path

import pytest

from crosspad_hil import stim
from crosspad_hil.cdc import CdcLink
from crosspad_hil.cli import CliDeps, render_stim
from crosspad_hil.cli import main as cli_main
from crosspad_hil.devices import Backends, Device, Ports, SerialPortInfo, UsbMode, discover
from crosspad_hil.errors import BAD_ARGS, NO_CDC_IN_AUDIO_MODE, NO_DEVICE, HilError
from crosspad_hil.serve import Daemon
from tests.fakes import FakeSerial

CDC = "/dev/ttyACM0"
CON = "/dev/ttyACM1"


class FakeTransport:
    """Records every press/release with the moment it happened."""

    name = "fake"

    def __init__(self, fail_on: int | None = None) -> None:
        self.events: list[tuple[str, int, int]] = []
        self.closed = False
        self.fail_on = fail_on
        self._lock = threading.Lock()

    def press(self, pad: int, vel: int) -> None:
        with self._lock:
            if self.fail_on is not None and len(self.events) >= self.fail_on:
                raise OSError("port went away")
            self.events.append(("press", pad, vel))

    def release(self, pad: int) -> None:
        with self._lock:
            self.events.append(("release", pad, 0))

    def close(self) -> None:
        self.closed = True

    @property
    def presses(self) -> list[tuple[str, int, int]]:
        return [e for e in self.events if e[0] == "press"]


class FakeCounters:
    """PAD_STATS / CDC_STATS that follow what the transport was told, plus optional drops."""

    def __init__(self, transport: FakeTransport, *, lose: int = 0, drops: list[int] | None = None):
        self._t = transport
        self._lose = lose
        self._drops = list(drops or [])
        self._base = 0
        self.closed = False
        self.samples = 0

    def _drop(self) -> int:
        return self._drops.pop(0) if len(self._drops) > 1 else (self._drops[0]
                                                                if self._drops else 0)

    def sample(self) -> stim.CounterSample:
        self.samples += 1
        presses = max(0, len(self._t.presses) - self._base - self._lose)
        return stim.CounterSample(press=presses, release=presses, played=presses,
                                  freeslots=12, drop=self._drop())

    def reset(self) -> stim.CounterSample:
        self._base = len(self._t.presses)
        return self.sample()

    def close(self) -> None:
        self.closed = True


def run(schedule, transport=None, counters=None, **kw) -> tuple[stim.Stimulus, dict]:
    tr = transport or FakeTransport()
    s = stim.Stimulus(schedule, tr, counters=counters, drop_poll_s=kw.pop("drop_poll_s", 0.02),
                      settle_s=kw.pop("settle_s", 0.0), **kw)
    s.start()
    s.wait(10.0)
    return s, s.stop()


# --------------------------------------------------------------------------- pad specs

@pytest.mark.parametrize("spec,expect", [
    ("0,2,4-7", [0, 2, 4, 5, 6, 7]),
    ("3", [3]),
    ("0-15", list(range(16))),
    ([1, 2], [1, 2]),
])
def test_parse_pads(spec, expect):
    assert stim.parse_pads(spec) == expect


@pytest.mark.parametrize("spec", ["", "16", "0-99", "a,b", "-1"])
def test_parse_pads_rejects_nonsense(spec):
    with pytest.raises(HilError) as e:
        stim.parse_pads(spec)
    assert e.value.code == BAD_ARGS


# --------------------------------------------------------------------------- schedule

def test_round_robin_rate_and_count():
    hits = stim.build_schedule([0, 1, 2], rate_hz=10, seconds=1.0, gate_ms=10)
    assert len(hits) == 10
    assert [h.pad for h in hits[:4]] == [0, 1, 2, 0]
    assert hits[1].t_ms - hits[0].t_ms == pytest.approx(100.0)


def test_chord_hits_every_pad_at_the_same_instant():
    hits = stim.build_schedule([0, 5, 9], rate_hz=4, seconds=1.0, chord=True)
    assert len(hits) == 12
    assert {h.pad for h in hits[:3]} == {0, 5, 9}
    assert len({h.t_ms for h in hits[:3]}) == 1


def test_velocity_set_is_cycled():
    hits = stim.build_schedule([0], rate_hz=4, seconds=1.0, velocity=[10, 20])
    assert [h.vel for h in hits] == [10, 20, 10, 20]


def test_humanize_is_reproducible_with_a_seed():
    a = stim.build_schedule([0], rate_hz=8, seconds=1.0, humanize_ms=5, seed=7)
    b = stim.build_schedule([0], rate_hz=8, seconds=1.0, humanize_ms=5, seed=7)
    c = stim.build_schedule([0], rate_hz=8, seconds=1.0, humanize_ms=5, seed=8)
    assert [h.t_ms for h in a] == [h.t_ms for h in b]
    assert [h.t_ms for h in a] != [h.t_ms for h in c]
    assert all(h.t_ms >= 0 for h in a)


def test_the_gate_collapses_toward_the_period_at_speed():
    """A 25 ms gate at 40 Hz would overlap the next hit; fast fingerdrumming does not."""
    hits = stim.build_schedule([0], rate_hz=40, seconds=0.2, gate_ms=25)
    assert hits[0].gate_ms == pytest.approx(10.0)      # 40 % of the 25 ms period
    slow = stim.build_schedule([0], rate_hz=4, seconds=1.0, gate_ms=25)
    assert slow[0].gate_ms == 25.0


def test_pattern_steps_are_validated_before_anything_is_written():
    with pytest.raises(HilError) as e:
        stim.pattern_to_hits([{"t_ms": 0, "pad": 99}])
    assert e.value.code == BAD_ARGS
    with pytest.raises(HilError):
        stim.pattern_to_hits([{"t_ms": 0}])
    with pytest.raises(HilError):
        stim.pattern_to_hits([{"t_ms": 0, "pad": 1, "colour": "red"}])
    with pytest.raises(HilError):
        stim.pattern_to_hits([])


def test_pattern_is_sorted_and_defaulted():
    hits = stim.pattern_to_hits([{"t_ms": 30, "pad": 2}, {"t_ms": 0, "pad": 1, "vel": 40}])
    assert [(h.t_ms, h.pad, h.vel) for h in hits] == [
        (0.0, 1, 40), (30.0, 2, stim.DEFAULT_VELOCITY)]


# --------------------------------------------------------------------------- throttle

def test_cdc_rate_is_clamped_against_the_64_deep_queue():
    rate, report = stim.throttle(40.0, "cdc")
    assert rate == stim.CDC_MAX_HIT_RATE_HZ
    assert report["requested_rate_hz"] == 40.0
    assert "app_queue" in report["reason"] and "64" in report["reason"]
    assert "sysex_stm" in report["hint"]


def test_a_rate_the_queue_sustains_is_left_alone():
    assert stim.throttle(8.0, "cdc") == (8.0, None)


def test_sysex_is_not_clamped_because_it_does_not_use_app_queue():
    assert stim.throttle(60.0, "sysex_stm") == (60.0, None)


def test_the_plan_reports_the_clamp():
    _hits, plan = stim.plan_stimulus(pads="0,1", rate_hz=50, seconds=0.1, transport="cdc")
    assert plan.rate_hz == stim.CDC_MAX_HIT_RATE_HZ
    assert plan.throttled["applied_rate_hz"] == stim.CDC_MAX_HIT_RATE_HZ


def test_rate_and_pattern_are_mutually_exclusive():
    with pytest.raises(HilError) as e:
        stim.plan_stimulus(pattern=[{"t_ms": 0, "pad": 0}], rate_hz=8)
    assert e.value.code == BAD_ARGS


def test_unknown_transport_is_bad_args():
    with pytest.raises(HilError) as e:
        stim.plan_stimulus(pads="0", transport="carrier-pigeon")
    assert e.value.code == BAD_ARGS


# --------------------------------------------------------------------------- running

def test_every_press_is_matched_by_a_release():
    hits = stim.build_schedule([0, 1], rate_hz=200, seconds=0.05, gate_ms=2)
    tr = FakeTransport()
    s, status = run(hits, tr)
    presses = [e for e in tr.events if e[0] == "press"]
    releases = [e for e in tr.events if e[0] == "release"]
    assert len(presses) == len(hits) == status["presses_sent"]
    assert len(releases) == len(presses)
    assert sorted(p[1] for p in presses) == sorted(r[1] for r in releases)
    assert status["running"] is False
    # stop() closes the transport object; whether that closes a *link* is the transport's
    # own business (CdcPadTransport(owns=False) leaves a caller-supplied link alone).
    assert tr.closed is True
    del s


def test_stop_releases_a_pad_that_is_still_held():
    hits = [stim.Hit(t_ms=0, pad=3, vel=100, gate_ms=60_000)]   # a gate that outlives the test
    tr = FakeTransport()
    s = stim.Stimulus(hits, tr, drop_poll_s=0, settle_s=0)
    s.start()
    for _ in range(200):
        if tr.presses:
            break
        threading.Event().wait(0.01)
    status = s.stop(timeout_s=1.0)
    assert ("release", 3, 0) in tr.events
    assert status["releases_sent"] >= 1


def test_pad_stats_delta_and_loss_accounting():
    hits = stim.build_schedule([0], rate_hz=200, seconds=0.05, gate_ms=1)
    tr = FakeTransport()
    counters = FakeCounters(tr, lose=2)          # two presses never reach the firmware
    _s, status = run(hits, tr, counters)
    sent = status["presses_sent"]
    assert status["pad_stats_delta"]["press"] == sent - 2
    assert status["loss"]["sent"] == sent
    assert status["loss"]["accepted"] == sent - 2
    assert status["loss"]["transport_pct"] == pytest.approx(100.0 * 2 / sent, abs=0.01)
    assert status["loss"]["engine_pct"] == 0.0
    assert counters.closed is True


def test_loss_accounting_separates_transport_from_engine():
    r = stim.loss_accounting(100, {"press": 90, "played": 80})
    assert r == {"sent": 100, "accepted": 90, "played": 80,
                 "transport_pct": 10.0, "engine_pct": 11.11, "total_pct": 20.0}


def test_drops_are_reported_not_hidden():
    hits = stim.build_schedule([0], rate_hz=50, seconds=0.6, gate_ms=1)
    tr = FakeTransport()
    counters = FakeCounters(tr, drops=[0, 0, 7, 7])
    _s, status = run(hits, tr, counters, drop_poll_s=0.02)
    assert status["cdc_drops"] == 7
    assert status["drop_events"], "a drop the device reported must be timestamped"
    assert status["drop_events"][0]["dropped"] == 7


def test_backoff_stretches_what_is_left_after_a_drop():
    hits = stim.build_schedule([0], rate_hz=50, seconds=0.6, gate_ms=1)
    tr = FakeTransport()
    counters = FakeCounters(tr, drops=[0, 0, 3, 3])
    _s, backed_off = run(hits, tr, counters, drop_poll_s=0.02, backoff=True)
    assert backed_off["drop_events"][0]["backoff"] is True
    # The run is stretched, so it does not finish all of the schedule in the same wall time.
    assert backed_off["presses_sent"] <= len(hits)


def test_no_counters_says_why_rather_than_reporting_zero():
    hits = stim.build_schedule([0], rate_hz=100, seconds=0.05)
    _s, status = run(hits)
    assert status["pad_stats"] is None
    assert status["pad_stats_delta"] is None
    assert status["cdc_drops"] is None


def test_a_dead_port_stops_the_run_and_is_reported():
    hits = stim.build_schedule([0], rate_hz=200, seconds=0.1, gate_ms=1)
    tr = FakeTransport(fail_on=3)
    _s, status = run(hits, tr)
    assert status["errors"] and "port went away" in status["errors"][0]
    assert status["presses_sent"] < len(hits)


def test_schedule_error_is_measured():
    hits = stim.build_schedule([0], rate_hz=50, seconds=0.2, gate_ms=1)
    _s, status = run(hits)
    err = status["schedule_error_ms"]
    assert err is not None and err["p50"] >= 0
    assert err["max"] < 200.0      # a scheduler that drifted by more than a period is broken


def test_starting_twice_is_refused():
    s = stim.Stimulus([stim.Hit(t_ms=0, pad=0)], FakeTransport(), drop_poll_s=0, settle_s=0)
    s.start()
    with pytest.raises(HilError) as e:
        s.start()
    assert e.value.code == BAD_ARGS
    s.stop()


# --------------------------------------------------------------------------- transports

def _device(usb_mode=UsbMode.DEFAULT, cdc=True) -> Device:
    port = SerialPortInfo(path=CDC, vid=0x303A, pid=0x3456, serial="AB", product="CrossPad",
                          location="1-1:1.0")
    return Device(id="dev_test", serial="AB", usb_mode=usb_mode,
                  ports=Ports(cdc=port if cdc else None))


def test_cdc_transport_is_fire_and_forget():
    sent: list[str] = []

    class Link:
        def send(self, cmd: str) -> None:
            sent.append(cmd)

        def close(self) -> None:
            sent.append("closed")

    tr = stim.CdcPadTransport(Link(), owns=False)
    tr.press(3, 90)
    tr.release(3)
    tr.close()
    assert sent == ["PAD_PRESS 3 90", "PAD_RELEASE 3"]     # never closes a link it does not own


def test_sysex_transport_sends_pad_frames():
    frames: list[bytes] = []

    class Io:
        def send_sysex(self, frame: bytes) -> None:
            frames.append(frame)

        def close(self) -> None:
            frames.append(b"closed")

    from crosspad_hil.midi import MidiRole

    tr = stim.SysexPadTransport(Io(), MidiRole.STM)
    assert tr.name == "sysex_stm"
    tr.press(2, 77)
    tr.release(2)
    assert frames[0] == bytes([0xF0, 0x7D, 0x1D, 0x07, 2, 77, 0xF7])
    assert frames[1][3] == 0x08


def test_audio_mode_says_the_cdc_endpoint_is_absent():
    with pytest.raises(HilError) as e:
        stim.open_transport(_device(UsbMode.AUDIO, cdc=False), "cdc")
    assert e.value.code == NO_CDC_IN_AUDIO_MODE
    assert "sysex_stm" in (e.value.hint or "")


def test_no_cdc_port_at_all_is_no_device():
    with pytest.raises(HilError) as e:
        stim.open_transport(_device(UsbMode.BOOTLOADER, cdc=False), "cdc")
    assert e.value.code == NO_DEVICE


# --------------------------------------------------------------------------- end to end

def _stats_serial() -> FakeSerial:
    fake = FakeSerial([], timeout=0.01)
    real_write = fake.write

    def write(data: bytes) -> int:
        cmd = data.decode().strip()
        if cmd == "PAD_STATS_RESET":
            fake.feed(["OK"])
            fake.written.append(cmd)
            return len(data)
        if cmd.startswith("PAD_STATS"):
            fake.feed(["PADSTATS: press=8 release=8 played=8 freeslots=12"])
            fake.written.append(cmd)
            return len(data)
        if cmd == "CDC_STATS":
            fake.feed(["CDCSTATS: rx=100 cmds=40 drop=0"])
            fake.written.append(cmd)
            return len(data)
        if cmd.startswith("PAD_"):
            fake.written.append(cmd)
            return len(data)
        return real_write(data)

    fake.write = write  # type: ignore[method-assign]
    return fake


def test_start_stimulus_shares_one_link_for_pads_and_counters():
    fake = _stats_serial()
    link = CdcLink(f"fake-stim-{uuid.uuid4().hex[:6]}", serial_factory=lambda p, **kw: fake)
    link.open()
    try:
        s = stim.start_stimulus(_device(), pads="0,1", rate_hz=8, seconds=0.3, gate_ms=5,
                                link=link, drop_poll_s=0, settle_s=0)
        s.wait(5.0)
        status = s.stop()
    finally:
        link.close()
    assert any(c.startswith("PAD_PRESS") for c in fake.written)
    assert any(c.startswith("PAD_RELEASE") for c in fake.written)
    assert status["pad_stats"]["played"] == 8
    assert status["cdc_drops"] == 0


# --------------------------------------------------------------------------- CLI and daemon

def _backends() -> Backends:
    ports = [
        SerialPortInfo(path=CDC, vid=0x303A, pid=0x3456, serial="AB12", product="CrossPad",
                       location="1-1.2:1.0"),
        SerialPortInfo(path=CON, vid=0x0483, pid=0x5740, serial="STM1",
                       product="CrossPad MIDI+Serial", location="1-1.3:1.0"),
    ]
    return Backends(list_serial=lambda: list(ports), list_midi=lambda: [], list_audio=lambda: [])


def test_cli_stim_reports_the_delta_and_the_drops(tmp_path, monkeypatch):
    monkeypatch.setenv("XDG_RUNTIME_DIR", str(tmp_path))
    fake = _stats_serial()
    out = io.StringIO()
    deps = CliDeps(backends=_backends(), serial_factory=lambda p, **kw: fake,
                   stdout=out, stderr=io.StringIO())
    code = cli_main(["stim", "--pads", "0,1", "--rate", "8", "--seconds", "0.3",
                     "--gate", "5", "--json"], deps=deps)
    assert code == 0
    data = json.loads(out.getvalue())
    assert data["plan"]["pads"] == [0, 1]
    assert data["pad_stats_delta"] is not None
    assert data["cdc_drops"] == 0


def test_cli_stim_takes_a_pattern_file(tmp_path, monkeypatch):
    monkeypatch.setenv("XDG_RUNTIME_DIR", str(tmp_path))
    pattern = tmp_path / "groove.json"
    pattern.write_text(json.dumps([{"t_ms": 0, "pad": 4, "vel": 90, "gate_ms": 5},
                                   {"t_ms": 20, "pad": 5, "vel": 60, "gate_ms": 5}]))
    fake = _stats_serial()
    out = io.StringIO()
    deps = CliDeps(backends=_backends(), serial_factory=lambda p, **kw: fake,
                   stdout=out, stderr=io.StringIO())
    assert cli_main(["stim", "--pattern", str(pattern), "--json"], deps=deps) == 0
    data = json.loads(out.getvalue())
    assert data["plan"]["pads"] == [4, 5]
    assert data["plan"]["hits"] == 2
    assert data["presses_sent"] == 2


def test_render_stim_names_the_throttle():
    status = {"plan": {"transport": "cdc", "pads": [0], "throttled":
                       {"requested_rate_hz": 40, "applied_rate_hz": 16, "reason": "app_queue"},
                       "counters_unavailable": None},
              "presses_sent": 10, "elapsed_s": 1.0, "cdc_drops": 3, "drop_events": [{}],
              "loss": {"sent": 10, "accepted": 9, "played": 9, "transport_pct": 10.0,
                       "engine_pct": 0.0, "total_pct": 10.0},
              "pad_stats_delta": {"press": 9}, "schedule_error_ms": None, "errors": []}
    text = render_stim(status)
    assert "throttled: 40 -> 16 Hz" in text
    assert "cdc_drops: 3" in text
    assert "loss transport 10.0%" in text


def test_daemon_stim_start_status_stop(tmp_path, monkeypatch):
    monkeypatch.setenv("XDG_RUNTIME_DIR", str(tmp_path))
    fake = _stats_serial()
    daemon = Daemon(io.StringIO(), io.StringIO(), backends=_backends(),
                    serial_factory=lambda p, **kw: fake)
    dev = discover(_backends())[0]
    started = daemon.handle({"id": 1, "op": "stim.start",
                             "args": {"device": dev.id, "pads": "0,1", "rate_hz": 8,
                                      "seconds": 0.4, "gate_ms": 5}})
    assert started["ok"], started
    handle = started["result"]["handle"]
    assert handle.startswith("stim_")
    assert started["result"]["plan"]["pads"] == [0, 1]
    status = daemon.handle({"id": 2, "op": "stim.status", "args": {"handle": handle}})
    assert status["ok"] and "sent" in status["result"]
    stopped = daemon.handle({"id": 3, "op": "stim.stop", "args": {"handle": handle}})
    assert stopped["ok"] and stopped["result"]["running"] is False
    # The handle is spent: a second stop cannot find it.
    again = daemon.handle({"id": 4, "op": "stim.stop", "args": {"handle": handle}})
    assert not again["ok"]


def test_daemon_stim_throttles_and_says_so(tmp_path, monkeypatch):
    monkeypatch.setenv("XDG_RUNTIME_DIR", str(tmp_path))
    fake = _stats_serial()
    daemon = Daemon(io.StringIO(), io.StringIO(), backends=_backends(),
                    serial_factory=lambda p, **kw: fake)
    r = daemon.handle({"id": 1, "op": "stim.start",
                       "args": {"pads": "0", "rate_hz": 60, "seconds": 0.2, "gate_ms": 2}})
    assert r["ok"], r
    assert r["result"]["plan"]["throttled"]["applied_rate_hz"] == stim.CDC_MAX_HIT_RATE_HZ
    daemon.handle({"id": 2, "op": "stim.stop", "args": {"handle": r["result"]["handle"]}})


def test_daemon_rejects_a_bad_pattern(tmp_path, monkeypatch):
    monkeypatch.setenv("XDG_RUNTIME_DIR", str(tmp_path))
    daemon = Daemon(io.StringIO(), io.StringIO(), backends=_backends(),
                    serial_factory=lambda p, **kw: _stats_serial())
    r = daemon.handle({"id": 1, "op": "stim.start",
                       "args": {"pattern": [{"t_ms": 0, "pad": 99}]}})
    assert not r["ok"] and r["error"]["code"] == BAD_ARGS


def test_pattern_file_that_is_not_a_list_is_refused(tmp_path, monkeypatch):
    monkeypatch.setenv("XDG_RUNTIME_DIR", str(tmp_path))
    bad = tmp_path / "bad.json"
    bad.write_text(json.dumps({"pattern": {"t_ms": 0}}))
    deps = CliDeps(backends=_backends(), serial_factory=lambda p, **kw: _stats_serial(),
                   stdout=io.StringIO(), stderr=io.StringIO())
    assert cli_main(["stim", "--pattern", str(bad)], deps=deps) == 2
    assert Path(bad).is_file()
