"""Scenario `perf`: the measurement parts as pure functions, and a whole run against
fakes — the regression the release gate let through (kit swap 2.8 s over CDC,
4.0 s in USB audio, engine init 3.3 s) must FAIL with its numbers; the healthy
build (1.6 s, 1.7 s, 130 ms) must pass. No board."""
from __future__ import annotations

import threading
import wave
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pytest

from crosspad_hil import verbs
from crosspad_hil.devices import Device, Ports, SerialPortInfo, UsbMode
from crosspad_hil.errors import BAD_ARGS, ENV, TIMEOUT, HilError
from crosspad_hil.parsers import parse_cdc_reply
from crosspad_hil.scenarios import _audio, base, get, perf, release_gate
from crosspad_hil.scenarios.perf import (
    Params,
    engine_init_ms,
    is_native_engine,
    phase_stats,
    pick_sink,
    rows_after,
    summary,
    swap_plan,
    verdict,
    write_tone,
)

# Real lines, bench dev_1ede 2026-10-05 (release_gate smoke console.log and
# bench_uac_20261005/A_swap.txt).
BLOCK = "I (2011) EngineHandoff: control block 1432 B at 0x3c7563c8"
STARTED = "I (5280) EngineHandoff: started: Init took 4255616 B in 43 blocks; arena 4252/4605 kB used"
ROW = ("KITSWAP: seq=22 kit=4 ok=1 req_ms=1287693 parse_ms=15 done_ms=2817 gated_blocks=0 "
       "gate_max_us=0 gate_total_us=0 gate_count=0 underruns=0 noblock=0 deferred=0x0000 "
       "forced=0x0000 dropped_hits=0 late_hits=0 rt_gap_max_us=1651 stream_fopens=0 "
       "bg_pass_max_us=214 fopen_max_us=7556 stream_kb=0 loader_kb=1538 "
       "pads=329,498,666,826,993,1161,1328,1488,1655,1819,1984,2149,2315,2484,2649,2817")


def row(seq: int, done_ms: int, ok: bool = True) -> dict[str, Any]:
    return {"seq": seq, "kit": 3 + seq % 2, "ok": ok, "done_ms": done_ms, "underruns": 0,
            "rt_gap_max_us": 1500}


# --------------------------------------------------------------------------- pure parts

def test_registered_and_in_the_release_gate_after_kit_churn() -> None:
    assert get("perf") is perf.SCENARIO
    order = release_gate.STEP_ORDER
    assert order.index("perf") == order.index("kit_churn") + 1
    p = Params()
    assert (p.swaps, p.max_cdc_ms, p.max_uac_ratio, p.max_init_ms, p.uac, p.stream, p.init) \
        == (6, 2000.0, 1.2, 500.0, True, True, True)


def test_engine_init_from_the_boot_stamps() -> None:
    assert engine_init_ms(["junk", BLOCK, "I (3000) other: x", STARTED]) == 3269
    assert engine_init_ms([BLOCK]) is None
    assert engine_init_ms([STARTED, BLOCK]) is None      # a started before the block is not it
    assert engine_init_ms([]) is None


def test_the_real_kitswap_row_parses_into_what_phase_stats_reads() -> None:
    parsed = parse_cdc_reply(ROW)
    assert parsed is not None
    st = phase_stats([parsed])
    assert (st["median_ms"], st["ok"], st["rt_gap_max_us"]) == (2817, 1, 1651)


def test_rows_after_takes_only_this_phase() -> None:
    rows = [row(s, 1000) for s in range(20, 28)] + [{"end": True, "count": 8}]
    assert [r["seq"] for r in rows_after(rows, 23)] == [24, 25, 26, 27]


def test_phase_stats_median_and_failed_swaps() -> None:
    st = phase_stats([row(1, 2808), row(2, 2846), row(3, 2817), row(4, 9999, ok=False)])
    assert st["median_ms"] == 2817 and st["failed_seq"] == [4] and st["swaps"] == 4


def test_swap_plan_alternates_starting_away_from_a() -> None:
    assert swap_plan(6, 3, 4) == [4, 3, 4, 3, 4, 3]


def _data(cdc: list[int], uac: list[int] | None, init: int | None) -> dict[str, Any]:
    d: dict[str, Any] = {"cdc": phase_stats([row(i, v) for i, v in enumerate(cdc)]),
                         "uac": None, "init_ms": init, "stuck": []}
    if uac is not None:
        d["uac"] = phase_stats([row(100 + i, v) for i, v in enumerate(uac)])
    return d


def test_verdict_healthy_build_passes() -> None:
    d = _data([1594, 1596, 1595, 1594, 1596, 1595], [1700] * 6, 130)
    assert verdict(Params(), d) == ([], [])
    assert d["uac_ratio"] == pytest.approx(1700 / 1595, abs=1e-3)


def test_verdict_regression_fails_on_all_three_with_numbers() -> None:
    d = _data([2808, 2846, 2817, 2820, 2830, 2812], [3971, 4082, 4060, 4052, 4113, 4070], 3269)
    failures, unmeasured = verdict(Params(), d)
    assert unmeasured == []
    assert failures[0] == "CDC kit swap: median 2818 ms > 2000 ms"
    assert failures[1].startswith("USB-audio kit swap: median 4065 ms = 1.44x CDC 2818 ms")
    assert failures[2] == "engine init at boot: 3269 ms > 500 ms"
    text = summary(Params(), d, failures, unmeasured)
    assert text.startswith("perf: FAIL") and "median 2818 ms over 6 swaps" in text
    assert "= 1.44x CDC" in text and "3269 ms" in text


def test_uac_over_ratio_but_within_the_fixed_extra_passes() -> None:
    # v1.1.3 on dev_1ede: CDC got faster, the USB tasks' ~120 ms did not.
    d = _data([477] * 6, [597] * 6, None)
    d["init_native"] = True
    assert verdict(Params(), d) == ([], [])
    failures, _ = verdict(Params(max_uac_extra_ms=100), d)
    assert failures[0].startswith("USB-audio kit swap: median 597 ms = 1.25x CDC 477 ms")
    assert failures[0].endswith("+120 ms > +100 ms")


def test_thresholds_are_parameters() -> None:
    d = _data([2818] * 6, [4065] * 6, 3269)
    assert verdict(Params(max_cdc_ms=3000, max_uac_ratio=1.5, max_init_ms=4000), d) == ([], [])


def test_missing_rows_and_init_are_unmeasured_not_passed() -> None:
    d = _data([], None, None)
    failures, unmeasured = verdict(Params(uac=False), d)
    assert failures == []
    assert any("KIT_SWAP_LOG has no rows" in u for u in unmeasured)
    assert any("EngineHandoff" in u for u in unmeasured)


NATIVE = "I (1944) EngineArena: dynamic: engines take PSRAM as they need it, 4596 kB free now"


def test_native_engine_is_recognised_from_its_arena_line() -> None:
    assert is_native_engine(["junk", NATIVE])
    assert not is_native_engine([BLOCK, NATIVE])          # the prebuilt streamer ran
    assert not is_native_engine([])


def test_native_engine_has_no_init_and_is_not_unmeasured() -> None:
    d = _data([474] * 6, None, None)
    d["init_native"] = True
    failures, unmeasured = verdict(Params(uac=False), d)
    assert (failures, unmeasured) == ([], [])
    assert "none (native engine)" in summary(Params(uac=False), d, failures, unmeasured)


def test_short_or_stuck_phases_fail() -> None:
    d = _data([1500, 1500], None, 100)
    d["stuck"] = ["CDC: KIT_LOAD 4: not current after 30.0s"]
    failures, _ = verdict(Params(uac=False), d)
    assert "kit swap stuck: CDC: KIT_LOAD 4: not current after 30.0s" in failures
    assert "CDC kit swap: only 2 of 6 swaps logged" in failures


def test_write_tone_is_a_stereo_16_bit_wav(tmp_path: Path) -> None:
    path = write_tone(tmp_path / "t.wav", seconds=1.5)
    with wave.open(str(path)) as w:
        assert (w.getnchannels(), w.getsampwidth(), w.getframerate()) == (2, 2, 44_100)
        assert w.getnframes() == int(1.5 * 44_100)


def test_pick_sink_prefers_the_target_then_the_cards_alsa_output() -> None:
    names = {"alsa_output.usb-Espressif_CrossPad-00.analog-stereo", "crosspad_out", "x"}
    assert pick_sink(names, "crosspad_out") == "crosspad_out"
    names.discard("crosspad_out")
    assert pick_sink(names, "crosspad_out").startswith("alsa_output.usb-Espressif_CrossPad")
    assert pick_sink({"alsa_output.pci-0000"}, "crosspad_out") is None


def test_release_gate_passes_the_thresholds_and_shrinks_for_quick() -> None:
    p = release_gate.Params(perf_max_cdc_ms=1800, perf_max_uac_ratio=1.1,
                            perf_max_init_ms=300, perf_stream=False)
    sp = release_gate._step_params("perf", p)
    assert (sp.swaps, sp.max_cdc_ms, sp.max_uac_ratio, sp.max_init_ms, sp.stream) \
        == (6, 1800, 1.1, 300, False)
    assert release_gate._step_params("perf", release_gate.Params(quick=True)).swaps == 2
    budget = release_gate._step_timeout_s("perf", sp, p.step_timeout_factor)
    assert 300 <= budget <= 900


# --------------------------------------------------------------------------- a whole run on fakes

@dataclass
class Bench:
    """What the fake board does: device-side done_ms per phase and the boot lines."""
    cdc_ms: list[int]
    uac_ms: list[int]
    boot_lines: list[str]
    rows: list[dict[str, Any]] = field(default_factory=list)
    seq: int = 10
    current: int = 3
    mode: UsbMode = UsbMode.DEFAULT
    calls: list[str] = field(default_factory=list)

    def swap(self, kit: int, ms: int) -> None:
        self.seq += 1
        self.current = kit
        self.rows.append(row(self.seq, ms))


class FakeConsole:
    def __init__(self, bench: Bench) -> None:
        self.bench = bench
        self.seq = 0

    def wait_boot(self, timeout_s: float = 0, settle_s: float = 0) -> Any:
        return type("Boot", (), {"complete": True})()

    def expect(self, patterns: list[str], timeout_s: float = 0, since_seq: int = 0) -> Any:
        hit = any("started" in line for line in self.bench.boot_lines)
        return type("Hit", (), {"hit": patterns[0] if hit else None})()

    def read(self, since_seq: int = 0, match: str | None = None, limit: int = 0) -> Any:
        lines = [(i, s) for i, s in enumerate(self.bench.boot_lines) if not match or match in s]
        return type("Read", (), {"lines": lines})()

    def close(self) -> None:
        pass


class FakeCtx(base.Context):
    bench: Bench

    def open_console(self, reset: bool = False) -> Any:
        self.bench.calls.append(f"console reset={reset}")
        return FakeConsole(self.bench)

    def open_cdc(self) -> Any:
        if self.bench.mode is not UsbMode.DEFAULT:
            raise AssertionError("CDC opened in the audio profile")
        return type("Link", (), {"close": lambda self: None})()


def rig(monkeypatch: pytest.MonkeyPatch, tmp_path: Path, bench: Bench,
        stream_ok: bool = True) -> tuple[FakeCtx, list[str]]:
    cdc_ms = iter(bench.cdc_ms)
    uac_ms = iter(bench.uac_ms)
    pending: dict[str, int] = {}

    monkeypatch.setattr(_audio, "ensure_kits", lambda ctx, link, names, b, **kw: (link, []))
    monkeypatch.setattr(_audio, "resolve_kit",
                        lambda link, kit_name, **kw: ((3, kit_name, 3) if kit_name.endswith("A")
                                                      else (4, kit_name, 3)))
    monkeypatch.setattr(perf, "start_sampler", lambda ctx, link: None)
    monkeypatch.setattr(_audio, "load_kit", lambda *a, **kw: None)
    monkeypatch.setattr(verbs, "kit_swap_log", lambda link, log=None: list(bench.rows))

    def kit_load(link: Any, kit: int, wait_s: float = 0) -> dict:
        bench.swap(kit, next(cdc_ms))
        return {}

    monkeypatch.setattr(verbs, "kit_load", kit_load)

    def set_mode(device: Device, mode: UsbMode, **kw: Any) -> Device:
        bench.mode = mode
        bench.calls.append(f"mode {mode.value}")
        ports = Ports() if mode is UsbMode.AUDIO else DEV.ports
        return Device(id=device.id, serial=device.serial, usb_mode=mode, ports=ports)

    monkeypatch.setattr(perf, "set_mode", set_mode)

    class Stream:
        argv = ["pw-play", "--target", "crosspad_out", "tone.wav"]

        def alive(self) -> bool:
            return stream_ok

        def stop(self) -> str:
            bench.calls.append("stream stop")
            return "" if stream_ok else "pw-play: target not found"

    def start_stream(wav: Path, target: str, log: Any) -> Stream:
        assert bench.mode is UsbMode.AUDIO and wav.exists()
        bench.calls.append("stream start")
        return Stream()

    monkeypatch.setattr(perf, "start_host_stream", start_stream)
    monkeypatch.setattr(perf, "open_companion",
                        lambda device: type("IO", (), {"close": lambda self: None})())

    def companion(io: Any, verb: str, timeout_s: float = 5.0) -> str:
        assert bench.mode is UsbMode.AUDIO
        if verb.startswith("KIT_LOAD"):
            pending["kit"] = int(verb.split()[1])
            bench.calls.append(verb)
            return "OK"
        if "kit" in pending:
            bench.swap(pending.pop("kit"), next(uac_ms))
        return f"KITSTATUS: current={bench.current} loading=0 pending=-1 name=HIL SINE"

    monkeypatch.setattr(perf, "companion_command", companion)
    monkeypatch.setattr(perf, "STREAM_SETTLE_S", 0.0)
    logs: list[str] = []
    ctx = FakeCtx(device=DEV, workdir=tmp_path, cancelled=threading.Event(), log=logs.append)
    ctx.bench = bench
    return ctx, logs


DEV = Device(id="dev_1ede", serial="CP0048", usb_mode=UsbMode.DEFAULT,
             ports=Ports(cdc=SerialPortInfo(path="/dev/ttyACM2", vid=0x303A, pid=0x3456,
                                            serial="CP0048", product="CrossPad",
                                            location="1-1.2:1.0")))


def test_regression_build_fails_with_its_numbers(monkeypatch, tmp_path) -> None:
    bench = Bench(cdc_ms=[2808, 2846, 2817, 2820, 2830, 2812],
                  uac_ms=[3971, 4082, 4060, 4052, 4113, 4070], boot_lines=[BLOCK, STARTED])
    ctx, _ = rig(monkeypatch, tmp_path, bench)
    rep = perf.SCENARIO.run(ctx, Params(settle=0.0), base.Progress())
    assert rep.exit_code == 1 and not rep.passed
    assert "CDC kit swap: median 2818 ms > 2000 ms" in rep.summary
    assert "1.44x CDC" in rep.summary and "engine init at boot: 3269 ms > 500 ms" in rep.summary
    # reset first, swaps over CDC, then audio + stream + companion, then back to default
    assert bench.calls[0] == "console reset=True"
    assert bench.calls[1:4] == ["mode audio", "stream start", "KIT_LOAD 4"]
    assert bench.calls[-2:] == ["stream stop", "mode default"]
    assert rep.data["cdc"]["swaps"] == 6 and rep.data["uac"]["swaps"] == 6
    assert len(rep.data["uac_host_s"]) == 6


def test_healthy_build_passes(monkeypatch, tmp_path) -> None:
    bench = Bench(cdc_ms=[1594, 1596, 1595, 1594, 1596, 1595], uac_ms=[1700] * 6,
                  boot_lines=["I (1864) EngineHandoff: control block 1432 B",
                              "I (1994) EngineHandoff: started: Init took 1 B"])
    ctx, _ = rig(monkeypatch, tmp_path, bench)
    rep = perf.SCENARIO.run(ctx, Params(settle=0.0), base.Progress())
    assert rep.exit_code == 0 and rep.passed, rep.summary
    assert rep.data["init_ms"] == 130
    assert rep.summary.startswith("perf: PASS")


def test_a_host_stream_that_dies_is_a_bench_problem_and_the_board_goes_back(
        monkeypatch, tmp_path) -> None:
    bench = Bench(cdc_ms=[1500] * 6, uac_ms=[1600] * 6, boot_lines=[BLOCK, STARTED])
    ctx, _ = rig(monkeypatch, tmp_path, bench, stream_ok=False)
    rep = perf.SCENARIO.run(ctx, Params(settle=0.0, max_init_ms=5000), base.Progress())
    assert rep.exit_code == 2, rep.summary
    assert "pw-play exited before the swaps" in rep.summary
    assert bench.mode is UsbMode.DEFAULT        # restored even though the phase failed


def test_a_stuck_uac_swap_fails(monkeypatch, tmp_path) -> None:
    bench = Bench(cdc_ms=[1500] * 6, uac_ms=[1600] * 6, boot_lines=[BLOCK, STARTED])
    ctx, _ = rig(monkeypatch, tmp_path, bench)

    def never_lands(ctx_: Any, io: Any, kit: int, timeout_s: float) -> None:
        raise HilError(TIMEOUT, f"companion KIT_LOAD {kit}: not current after 30s")

    monkeypatch.setattr(perf.PerfScenario, "_wait_landed", staticmethod(never_lands))
    rep = perf.SCENARIO.run(ctx, Params(settle=0.0, max_init_ms=5000), base.Progress())
    assert rep.exit_code == 1
    assert "kit swap stuck: USB audio: companion KIT_LOAD 4" in rep.summary


def test_no_stream_and_no_uac_flags(monkeypatch, tmp_path) -> None:
    bench = Bench(cdc_ms=[1500] * 6, uac_ms=[1600] * 6, boot_lines=[BLOCK, STARTED])
    ctx, _ = rig(monkeypatch, tmp_path, bench)
    rep = perf.SCENARIO.run(ctx, Params(settle=0.0, stream=False, max_init_ms=5000),
                            base.Progress())
    assert rep.exit_code == 0 and "stream start" not in bench.calls
    bench2 = Bench(cdc_ms=[1500] * 6, uac_ms=[], boot_lines=[BLOCK, STARTED])
    ctx2, _ = rig(monkeypatch, tmp_path / "2", bench2)
    rep2 = perf.SCENARIO.run(ctx2, Params(settle=0.0, uac=False, init=False),
                             base.Progress())
    assert rep2.exit_code == 0 and "mode audio" not in bench2.calls
    assert bench2.calls[0] == "console reset=False"


# --------------------------------------------------------------------------- starting the Sampler

class Launcher:
    """APP_LIST / APP_START / APP_STOP of a board whose launches can be dropped."""

    def __init__(self, running: str | None = None, dropped_starts: int = 0,
                 apps: tuple[str, ...] = ("Sampler", "Settings")) -> None:
        self.running = running
        self.dropped = dropped_starts
        self.apps = list(apps)
        self.starts: list[float] = []
        self.stops = 0
        self.lists_until_closed = 0

    def install(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(verbs, "app_list", self.app_list)
        monkeypatch.setattr(verbs, "app_start", self.app_start)
        monkeypatch.setattr(verbs, "app_stop", self.app_stop)

    def app_list(self, link: Any) -> dict:
        if self.lists_until_closed:
            self.lists_until_closed -= 1
            if not self.lists_until_closed:
                self.running = None
        return {"apps": self.apps, "running": self.running}

    def app_stop(self, link: Any) -> dict:
        self.stops += 1
        self.lists_until_closed = 3      # the teardown takes a few polls
        return {}

    def app_start(self, link: Any, name: str, wait_s: float = 3.0) -> dict:
        self.starts.append(wait_s)
        if self.dropped:
            self.dropped -= 1
            raise HilError(TIMEOUT, f"APP_START {name}: not running after {wait_s:.1f}s "
                                    "(APP_LIST running=-)", running=None, name=name)
        self.running = name
        return {"running": name}


def _ctx(tmp_path: Path) -> base.Context:
    return base.Context(device=DEV, workdir=tmp_path, cancelled=threading.Event(),
                        log=lambda s: None)


def test_sampler_start_survives_a_dropped_launch_after_the_reset(monkeypatch, tmp_path) -> None:
    """The bench run on dev_1ede: right after the reset, APP_START answered OK
    and nothing ran within 3 s (running=-). A longer window and a second request."""
    board = Launcher(running=None, dropped_starts=1)
    board.install(monkeypatch)
    perf.start_sampler(_ctx(tmp_path), "LINK", settle_s=0.0)
    assert board.running == "Sampler"
    assert board.starts == [perf.APP_START_WAIT_S, perf.APP_START_WAIT_S]
    assert perf.APP_START_WAIT_S > 3.0


def test_sampler_start_closes_the_previous_app_first(monkeypatch, tmp_path) -> None:
    board = Launcher(running="Settings")
    board.install(monkeypatch)
    perf.start_sampler(_ctx(tmp_path), "LINK", settle_s=0.0)
    assert board.stops == 1 and board.running == "Sampler" and len(board.starts) == 1


def test_sampler_start_gives_up_after_every_attempt(monkeypatch, tmp_path) -> None:
    board = Launcher(running=None, dropped_starts=5)
    board.install(monkeypatch)
    with pytest.raises(HilError) as ei:
        perf.start_sampler(_ctx(tmp_path), "LINK", settle_s=0.0)
    assert ei.value.code == TIMEOUT and len(board.starts) == perf.APP_START_ATTEMPTS


def test_sampler_already_running_or_missing(monkeypatch, tmp_path) -> None:
    board = Launcher(running="Sampler")
    board.install(monkeypatch)
    perf.start_sampler(_ctx(tmp_path), "LINK", settle_s=0.0)
    assert board.starts == []
    Launcher(apps=("Settings",)).install(monkeypatch)
    with pytest.raises(HilError) as ei:
        perf.start_sampler(_ctx(tmp_path), "LINK", settle_s=0.0)
    assert ei.value.code == ENV


def test_swaps_beyond_the_ring_are_refused(tmp_path) -> None:
    ctx = base.Context(device=DEV, workdir=tmp_path, cancelled=threading.Event(),
                       log=lambda s: None)
    with pytest.raises(HilError) as ei:
        perf.SCENARIO.run(ctx, Params(swaps=9), base.Progress())
    assert ei.value.code == BAD_ARGS


def test_env_codes_are_known() -> None:
    assert ENV in perf.SETUP_CODES and TIMEOUT in perf.SETUP_CODES
