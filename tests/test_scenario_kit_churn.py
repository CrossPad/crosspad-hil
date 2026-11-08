"""Scenario `kit_churn` against a stateful fake device — no hardware.

The fake models exactly the parts of hil_control.cpp the scenario leans on:
KIT_LIST / KIT_STATUS / KIT_LOAD with the "queued vs started" console line,
PAD_PRESS / PAD_RELEASE answering OK like everything else, SMPL_PEAK that is
read-and-clear and only non-zero when pads were hit, PAD_STATS, APP_LIST.
"""
from __future__ import annotations

import threading
from dataclasses import dataclass
from pathlib import Path

import pytest

from crosspad_hil.cdc import CdcLink
from crosspad_hil.console import Console
from crosspad_hil.devices import Device, Ports, SerialPortInfo, UsbMode
from crosspad_hil.scenarios import base, get, kit_churn
from tests.fakes import FakeSerial

KITS = ["Basic", "Drums", "Perc", "Synth"]
PANIC = "Guru Meditation Error: Core  1 panic'ed (LoadProhibited)."


class KitDevice(FakeSerial):
    """CDC endpoint with the kit loader's state machine.

    A load takes `load_polls` KIT_STATUS polls to finish (so a swap window has
    a measurable length in the test). A KIT_LOAD that arrives mid-load is held
    as `pending` and started when the running one completes — unless
    `drop_pending` models the old bug where it was thrown away. `stuck` never
    finishes a load. `crash_on_kit` feeds a panic line into the console fake
    when that kit is requested. `silent` makes SMPL_PEAK report 0 forever.
    """

    def __init__(self, console: FakeSerial, *, load_polls: int = 2) -> None:
        super().__init__()
        self.console = console
        self.lose_console_lines = 0   # >0: every n-th KIT_LOAD console line is lost
        self._console_seq = 0
        self.load_polls = load_polls
        self.current = -1
        self.target = -1
        self.pending = -1
        self.loading_left = 0
        self.running: str | None = "Sampler"
        self.apps = ["Sampler", "Mixer"]
        self.presses = 0
        self.releases = 0
        self.presses_since_peak = 0
        self.silent = False
        self.free = 7
        self.stuck = False
        self.drop_pending = False
        self.crash_on_kit: int | None = None
        # The native engine behind SMPL_ENGINE. "follow": it takes every kit
        # that becomes current (after `take_polls` SMPL_ENGINE polls in
        # state=1). "stale": it keeps the kit before -- the merge-review C2
        # bug, a take refused while one ran and never retried. "old_fw": a
        # reply without kit=. "pitched": the bank holds the kit, the native
        # engine holds none. "failed": the take failed for `why`.
        self.engine_mode = "follow"
        self.take_polls = 0
        self.why = "-"
        self._engine_kit = -1
        self._take_left = 0
        self._took_for = -1

    def _smpleng(self) -> str:
        if self.loading_left == 0 and self.current != self._took_for:
            # A load just finished: its end asks for a take.
            self._took_for = self.current
            self._take_left = self.take_polls
            pending = self.current
            # "stale" keeps the seed kit (0) whatever the board settles on.
            self._engine_kit = 0 if self.engine_mode == "stale" else pending
        state = 2
        if self._take_left > 0:
            self._take_left -= 1
            state = 1
        kit = self._engine_kit
        engine = "native"
        if self.engine_mode in ("pitched", "failed"):
            kit, engine = -1, "none"
            state = 3 if self.engine_mode == "failed" else 0
        why = self.why if self.engine_mode == "failed" else "-"
        kit_field = "" if self.engine_mode == "old_fw" else f"kit={kit} "
        return (f"SMPLENG: engine={engine} state={state} why={why} {kit_field}pitch=0,0,0 "
                f"slots=16 complete=16 head_kb=256 heads_kb=4096 pool_kb=64 pool_blocks=0/4/4 "
                f"voices=0/4 lim=0/16 notes=0 drop=0 steal=0 norings=0 underrun=0 quiet=0 "
                f"rt_max_us=0 noteon_max_us=0 read_kb=0 bg_stack_free=1800 engines=native")

    def _log(self, text: str) -> None:
        if self.lose_console_lines and text.startswith("KIT_LOAD") and \
                self._console_seq % self.lose_console_lines == self.lose_console_lines - 1:
            self._console_seq += 1
            return  # the UART dropped this one (pad storm + kit load)
        self._console_seq += 1
        self.console.feed([f"I (4242) hil_control: {text}"])

    def _status(self) -> str:
        if self.loading_left > 0 and not self.stuck:
            self.loading_left -= 1
            if self.loading_left == 0:
                self.current = self.target
                if self.pending != -1 and not self.drop_pending:
                    self.target = self.pending
                    self.loading_left = self.load_polls
                self.pending = -1
        name = KITS[self.current] if self.current >= 0 else "-"
        loading = 1 if self.loading_left > 0 else 0
        return f"KITSTATUS: current={self.current} loading={loading} pending={self.pending} name={name}"

    def write(self, data: bytes) -> int:
        for raw in data.decode("utf-8", "replace").split("\n"):
            cmd = raw.strip()
            if not cmd:
                continue
            self.written.append(cmd)
            if cmd == "APP_LIST":
                self.feed([f"APPS: {','.join(self.apps)} running={self.running or '-'}"])
            elif cmd in ("APP_DESTROY", "APP_STOP"):
                self.running = None
                self.feed(["OK"])
            elif cmd.startswith("APP_START "):
                self.running = cmd.split(" ", 1)[1]
                self.feed(["OK"])
            elif cmd == "KIT_LIST":
                body = ",".join(f"{i}:{n}" for i, n in enumerate(KITS))
                self.feed([f"KITS: {body} current={self.current}"])
            elif cmd == "KIT_STATUS":
                self.feed([self._status()])
            elif cmd.startswith("KIT_LOAD "):
                kit = int(cmd.split(" ", 1)[1])
                if kit >= len(KITS):
                    self.feed(["ERR bad kit id"])
                    continue
                if self.crash_on_kit == kit:
                    self.console.feed([PANIC])
                if self.loading_left > 0:
                    self.pending = kit
                    self._log(f"KIT_LOAD {kit} queued")
                else:
                    self.target = kit
                    self.loading_left = self.load_polls
                    self._log(f"KIT_LOAD {kit} started")
                self.feed(["OK"])
            elif cmd.startswith("PAD_PRESS "):
                self.presses += 1
                self.presses_since_peak += 1
                self.feed(["OK"])
            elif cmd.startswith("PAD_RELEASE "):
                self.releases += 1
                self.feed(["OK"])
            elif cmd == "SMPL_ENGINE":
                self.feed([self._smpleng()])
            elif cmd == "PITCHED_STATUS":
                active = 1 if self.engine_mode == "pitched" else 0
                self.feed([f"PITCHED: active={active} zones=0 roots= voices=0/24 bank_kb=0 "
                           f"rt_max_us=0 steals=0"])
            elif cmd == "PAD_STATS":
                self.feed([f"PADSTATS: press={self.presses} release={self.releases} "
                           f"played={self.presses} freeslots=7"])
            elif cmd == "SMPL_PEAK":
                peak = 0 if self.silent or self.presses_since_peak == 0 else 12000
                self.presses_since_peak = 0
                self.feed([f"SMPLPEAK: {peak} free={self.free}"])
            else:
                self._rx.append((cmd + "\r\n").encode())
        return len(data)


def make_device() -> Device:
    return Device(
        id="dev_ab12", serial="CP-1", usb_mode=UsbMode.DEFAULT,
        ports=Ports(
            cdc=SerialPortInfo(path="/dev/ttyACM0", vid=0x303A, pid=0x3456,
                               serial="CP-1", product="CrossPad", location="1-2.1"),
            console=SerialPortInfo(path="/dev/ttyACM1", vid=0x0483, pid=0x5740,
                                   serial="STM-1", product="CrossPad", location="1-2.2"),
        ),
    )


@dataclass
class FakeContext(base.Context):
    cdc_fake: KitDevice = None  # type: ignore[assignment]
    console_fake: FakeSerial = None  # type: ignore[assignment]

    def open_console(self, reset: bool = False) -> Console:
        fake = self.console_fake
        con = Console(self.device.ports.console.path,
                      log_path=self.workdir / "console.log",
                      serial_factory=lambda path, **kw: fake)
        con.open(reset=reset)
        self.opened.append(con)
        return con

    def open_cdc(self) -> CdcLink:
        fake = self.cdc_fake
        link = CdcLink(self.device.ports.cdc.path, serial_factory=lambda path, **kw: fake)
        link.open()
        self.opened.append(link)
        return link


def make_ctx(tmp_path: Path, **dev_kw: object) -> tuple[FakeContext, KitDevice]:
    con = FakeSerial()
    dev = KitDevice(con, **dev_kw)  # type: ignore[arg-type]
    ctx = FakeContext(device=make_device(), workdir=tmp_path,
                      cancelled=threading.Event(), log=lambda s: None,
                      cdc_fake=dev, console_fake=con)
    return ctx, dev


@pytest.fixture(autouse=True)
def fast(monkeypatch: pytest.MonkeyPatch) -> None:
    """Collapse every human-scale wait the old script carried."""
    monkeypatch.setattr(kit_churn, "APP_DESTROY_SETTLE_S", 0.0)
    monkeypatch.setattr(kit_churn, "APP_START_SETTLE_S", 0.0)
    monkeypatch.setattr(kit_churn, "MIN_DWELL_S", 0.0)
    monkeypatch.setattr(kit_churn, "RAPID_DRAIN_S", 0.0)
    monkeypatch.setattr(kit_churn, "RAPID_QUIET_POLL_S", 0.01)
    monkeypatch.setattr(kit_churn, "RAPID_SOUND_SETTLE_S", 0.0)
    monkeypatch.setattr(kit_churn, "PEAK_WINDOW_S", 0.05)


def run(ctx: FakeContext, **params: object) -> base.Report:
    return kit_churn.SCENARIO.run(ctx, kit_churn.Params(**params), base.Progress())  # type: ignore[arg-type]


def test_registered_and_defaults() -> None:
    assert get("kit_churn") is kit_churn.SCENARIO
    p = kit_churn.Params()
    assert (p.rounds, p.kits, p.dwell, p.load_timeout, p.hit_rate, p.pads,
            p.rapid, p.no_play, p.silence_fails) == (20, None, 2.0, 15.0, 8.0, None, None, False, False)


def test_normal_mode_swaps_under_play_and_counts_hits_in_window(tmp_path: Path) -> None:
    ctx, dev = make_ctx(tmp_path, load_polls=2)
    rep = run(ctx, rounds=4, dwell=0.0, hit_rate=50.0, load_timeout=5.0)
    assert rep.passed, rep.summary
    assert rep.exit_code == 0
    d = rep.data
    assert d["mode"] == "normal"
    assert d["cycle"] == [0, 1, 2, 3]
    assert d["pads"] == list(range(16))
    assert len(d["rounds"]) == 4
    # round r loads cycle[r % len(cycle)] — the seed already loaded kit 0
    assert [r["kit"] for r in d["rounds"]] == [1, 2, 3, 0]
    assert all(r["landed"] for r in d["rounds"])
    # each swap window spans two KIT_STATUS polls (~0.8 s) at 50 hits/s
    assert all(r["hits_in_window"] >= kit_churn.MIN_HITS_IN_WINDOW for r in d["rounds"])
    assert d["hits_in_windows"] == sum(r["hits_in_window"] for r in d["rounds"])
    assert d["false_negative"] is False
    assert d["starved"] == []
    assert d["silent"] == []
    assert d["failures"] == []
    assert d["pad_stats"]["press"] == dev.presses
    assert d["hits_sent"] == dev.presses
    assert d["reboots"] == 0 and d["fatal"] == []
    # the device saw the seed plus four swaps, all from an idle loader
    assert dev.written.count("KIT_LOAD 0") == 2
    assert "KIT_LOAD 1" in dev.written and "KIT_LOAD 3" in dev.written


def test_false_negative_guard_fails_when_stimulus_never_fires(tmp_path: Path,
                                                             monkeypatch: pytest.MonkeyPatch) -> None:
    """The stimulus thread is started but never sends: every swap window is
    quiet, and the run must FAIL rather than report a green swap from silence."""
    monkeypatch.setattr(kit_churn.Player, "run", lambda self: None)
    ctx, dev = make_ctx(tmp_path, load_polls=2)
    rep = run(ctx, rounds=3, dwell=0.0, hit_rate=50.0, load_timeout=5.0)
    assert not rep.passed
    assert rep.exit_code == 1
    assert rep.data["false_negative"] is True
    assert [r["hits_in_window"] for r in rep.data["rounds"]] == [0, 0, 0]
    assert len(rep.data["starved"]) == 3
    assert dev.presses == 0
    assert "false negative" in rep.summary


def test_no_play_control_case_skips_the_guard(tmp_path: Path) -> None:
    ctx, dev = make_ctx(tmp_path, load_polls=1)
    rep = run(ctx, rounds=2, no_play=True, dwell=0.0, load_timeout=5.0)
    assert rep.passed
    assert rep.data["false_negative"] is False
    assert rep.data["hits_sent"] == 0
    assert rep.data["silent"] == []          # silence is not judged without stimulus
    assert dev.presses == 0
    assert not any(c.startswith("PAD_PRESS") for c in dev.written)


@pytest.mark.parametrize("silence_fails, expect_pass", [(False, True), (True, False)])
def test_silent_kit_is_warn_or_fail(tmp_path: Path, silence_fails: bool, expect_pass: bool) -> None:
    ctx, dev = make_ctx(tmp_path, load_polls=2)
    dev.silent = True
    rep = run(ctx, rounds=2, dwell=0.0, hit_rate=50.0, load_timeout=5.0,
              silence_fails=silence_fails)
    assert rep.passed is expect_pass
    assert len(rep.data["silent"]) == 2
    assert all(r["silent"] and r["peak"] == 0 for r in rep.data["rounds"])
    assert rep.data["silence_fails"] is silence_fails


def test_crash_during_swap_fails_with_console_line(tmp_path: Path) -> None:
    ctx, dev = make_ctx(tmp_path, load_polls=1)
    dev.crash_on_kit = 2
    rep = run(ctx, rounds=4, dwell=0.0, hit_rate=50.0, load_timeout=5.0)
    assert not rep.passed
    assert rep.exit_code == 1
    assert rep.data["failures"] and "CRASH" in rep.data["failures"][0]
    assert any("Guru Meditation" in line for line in rep.data["fatal"])
    # the run stops at the crash: kits 1 and 2 were requested, 3 never was
    assert len(rep.data["rounds"]) == 2
    assert "KIT_LOAD 3" not in dev.written


def test_stuck_load_fails_as_never_current(tmp_path: Path) -> None:
    ctx, dev = make_ctx(tmp_path, load_polls=1)
    rep = run(ctx, rounds=2, dwell=0.0, hit_rate=50.0, load_timeout=5.0)
    assert rep.passed
    dev.stuck = True
    rep = run(ctx, rounds=2, dwell=0.0, hit_rate=50.0, load_timeout=0.6)
    assert rep.exit_code == 2 or not rep.passed
    # a stuck *seed* is an environment error (2); a stuck *swap* is a failure (1)
    if rep.exit_code == 1:
        assert "never became current" in rep.data["failures"][0]


def test_environment_errors_exit_2(tmp_path: Path) -> None:
    ctx, dev = make_ctx(tmp_path)
    dev.apps = ["Mixer"]
    dev.running = "Mixer"
    rep = run(ctx, rounds=1)
    assert rep.exit_code == 2 and not rep.passed
    assert "Sampler" in rep.summary

    ctx, dev = make_ctx(tmp_path)
    rep = run(ctx, rounds=1, kits=[0, 9])
    assert rep.exit_code == 2
    assert "9" in rep.summary


def test_sampler_is_started_when_not_running(tmp_path: Path) -> None:
    ctx, dev = make_ctx(tmp_path, load_polls=1)
    dev.running = "Mixer"
    rep = run(ctx, rounds=1, no_play=True, dwell=0.0, load_timeout=5.0)
    assert rep.passed
    assert "APP_DESTROY" in dev.written
    assert "APP_START Sampler" in dev.written
    # ...and the board is handed on the way it was found: Mixer up again.
    assert dev.written.index("APP_START Mixer") > dev.written.index("APP_START Sampler")
    assert dev.running == "Mixer"


def test_the_kit_and_the_launcher_are_put_back(tmp_path: Path) -> None:
    """The gate's next step (usb_mode_cycle) measures heap drift: it must not
    inherit the Sampler and whatever kit the burst settled on."""
    ctx, dev = make_ctx(tmp_path, load_polls=1)
    dev.current, dev.running = 2, None
    rep = run(ctx, rounds=3, no_play=True, dwell=0.0, load_timeout=5.0)
    assert rep.passed, rep.summary
    assert rep.data["found"] == {"kit": 2, "app": None}
    assert rep.data["left_as_found"] is True
    assert dev.current == 2 and dev.running is None


def test_a_stopped_player_releases_the_pad_it_was_holding() -> None:
    sent: list[str] = []

    from crosspad_hil.scenarios.kit_churn import Player

    class Link:
        def send(self, line: str) -> None:
            sent.append(line)
            player._halt.set()   # halted the moment the first press goes out

    player = Player(Link(), hit_rate=50.0, pads=[3])  # type: ignore[arg-type]
    player.run()
    assert sent == ["PAD_PRESS 3 77", "PAD_RELEASE 3"]


def test_the_seed_kit_is_loaded_before_the_sampler_is_started(tmp_path: Path) -> None:
    """A board with no current kit shows the kit-selector gate.

    APP_LIST then reports running=- because the overlay is not the
    orchestrator's running app, so APP_START times out — and the scenario, which
    started the Sampler before seeding, could never pass from a cold boot.
    """
    ctx, dev = make_ctx(tmp_path, load_polls=1)
    dev.running = None
    rep = run(ctx, rounds=1, no_play=True, dwell=0.0, load_timeout=5.0)
    assert rep.passed, rep.summary
    first_load = next(i for i, c in enumerate(dev.written) if c.startswith("KIT_LOAD"))
    first_start = next(i for i, c in enumerate(dev.written) if c.startswith("APP_START"))
    assert first_load < first_start


def test_rapid_mode_settles_on_last_logged_kit(tmp_path: Path) -> None:
    ctx, dev = make_ctx(tmp_path, load_polls=1)
    rep = run(ctx, rounds=6, rapid=0.02, hit_rate=50.0, dwell=0.0, load_timeout=1.0)
    assert rep.passed, rep.summary
    d = rep.data
    assert d["mode"] == "rapid"
    assert d["rounds"] == []
    r = d["rapid"]
    assert r["interval_s"] == 0.02
    # round n asks for cycle[n % 4]: 1,2,3,0,1,2
    assert r["requested"] == [1, 2, 3, 0, 1, 2]
    assert r["device_logged"] == [1, 2, 3, 0, 1, 2]
    assert r["last_logged"] == 2
    assert r["settled"]["current"] == 2
    assert r["settled"]["loading"] is False and r["settled"]["pending"] == -1
    assert r["peak"] == 12000
    # first request met an idle loader, the other five were queued
    assert dev.written.count("KIT_LOAD 1") == 2


def test_rapid_mode_fails_when_pending_request_is_dropped(tmp_path: Path) -> None:
    ctx, dev = make_ctx(tmp_path, load_polls=1)
    dev.drop_pending = True
    rep = run(ctx, rounds=5, rapid=0.02, hit_rate=50.0, dwell=0.0, load_timeout=1.0)
    assert not rep.passed
    assert rep.exit_code == 1
    r = rep.data["rapid"]
    assert r["last_logged"] == 1          # requested 1,2,3,0,1 → last logged is 1
    assert r["settled"]["current"] == 1   # …but only because the first one landed
    # the coalescing check is on the *sequence*: the device logged 2,3,0 and
    # never loaded them; the failure text names the mismatch when it exists
    assert rep.data["failures"]


def test_rapid_mode_without_stimulus_does_not_judge_silence(tmp_path: Path) -> None:
    ctx, dev = make_ctx(tmp_path, load_polls=1)
    rep = run(ctx, rounds=3, rapid=0.02, no_play=True, load_timeout=1.0)
    assert rep.passed
    assert rep.data["rapid"]["peak"] == 0
    assert rep.data["hits_sent"] == 0


def test_cancel_stops_the_run_and_the_player(tmp_path: Path) -> None:
    ctx, dev = make_ctx(tmp_path, load_polls=2)
    ctx.cancelled.set()
    from crosspad_hil.errors import CANCELLED, HilError
    with pytest.raises(HilError) as ei:
        run(ctx, rounds=5, dwell=0.0, hit_rate=50.0, load_timeout=5.0)
    assert ei.value.code == CANCELLED
    # every thread the scenario started is gone
    assert not any(t.name.startswith("kit_churn_player") for t in threading.enumerate())


def test_rapid_mode_tolerates_console_lines_lost_when_it_settled_on_the_last_request(
    tmp_path: Path,
) -> None:
    """Gate #4 (2026-09-02): the board settled on request 20 of 20, but the console
    had logged 16 -- lines lost under 8 hits/s plus a kit load, not swaps lost."""
    ctx, dev = make_ctx(tmp_path, load_polls=1)
    dev.lose_console_lines = 3
    rep = run(ctx, rounds=6, rapid=0.02, hit_rate=50.0, dwell=0.0, load_timeout=1.0)
    assert rep.passed, rep.summary
    r = rep.data["rapid"]
    assert r["requested"] == [1, 2, 3, 0, 1, 2]
    assert len(r["device_logged"]) < 6
    assert r["settled"]["current"] == 2
    assert r["console_lost_requests"] == 6 - len(r["device_logged"])


def test_rapid_mode_does_not_fail_on_a_kit_with_no_samples(tmp_path: Path) -> None:
    """A kit whose kit.json has empty samples[] (e.g. the bench's Kraken Blue)
    loads nothing and reads SMPL_PEAK free at the empty mark. Silence is then
    the kit, not the loader — the burst must not FAIL on it."""
    ctx, dev = make_ctx(tmp_path, load_polls=1)
    dev.silent = True
    dev.free = 23                     # empty player: no samples loaded
    rep = run(ctx, rounds=6, rapid=0.02, hit_rate=50.0, dwell=0.0, load_timeout=1.0)
    assert rep.passed, rep.summary
    assert rep.exit_code == 0
    assert not any("silent after the burst" in f for f in rep.data["failures"])
    assert rep.data.get("empty_kits")


# ── The native engine holds the kit the board settled on (merge review C2) ──

def test_normal_mode_checks_the_engine_holds_each_settled_kit(tmp_path: Path) -> None:
    ctx, dev = make_ctx(tmp_path, load_polls=1)
    dev.take_polls = 2                       # each take is still running at the first look
    rep = run(ctx, rounds=4, dwell=0.0, hit_rate=50.0, load_timeout=5.0)
    assert rep.passed, rep.summary
    assert [r["engine_kit"]["kit"] for r in rep.data["rounds"]] == [1, 2, 3, 0]
    assert all(r["engine_kit"]["state"] == 2 for r in rep.data["rounds"])


def test_normal_mode_fails_when_the_engine_keeps_the_old_kit(tmp_path: Path) -> None:
    """The take asked for during another was refused and nothing retried it:
    KIT_STATUS says the new kit, the engine still plays the old one."""
    ctx, dev = make_ctx(tmp_path, load_polls=1)
    dev.engine_mode = "stale"
    rep = run(ctx, rounds=3, dwell=0.0, hit_rate=50.0, load_timeout=5.0)
    assert not rep.passed
    assert rep.exit_code == 1
    assert any("native engine holds kit 0" in f and "settled on 1" in f
               for f in rep.data["failures"]), rep.data["failures"]


def test_rapid_mode_checks_the_engine_holds_the_settled_kit(tmp_path: Path) -> None:
    ctx, dev = make_ctx(tmp_path, load_polls=1)
    rep = run(ctx, rounds=6, rapid=0.02, hit_rate=50.0, dwell=0.0, load_timeout=1.0)
    assert rep.passed, rep.summary
    assert rep.data["rapid"]["engine_kit"]["kit"] == 2

    ctx, dev = make_ctx(tmp_path, load_polls=1)
    dev.engine_mode = "stale"
    rep = run(ctx, rounds=6, rapid=0.02, hit_rate=50.0, dwell=0.0, load_timeout=1.0)
    assert not rep.passed
    assert any("native engine holds kit" in f for f in rep.data["failures"]), rep.data["failures"]


def test_engine_check_tolerates_old_firmware_a_pitched_bank_and_an_empty_kit(
    tmp_path: Path,
) -> None:
    for mode, why in (("old_fw", "-"), ("pitched", "-"), ("failed", "no_sample_readable")):
        ctx, dev = make_ctx(tmp_path, load_polls=1)
        dev.engine_mode = mode
        dev.why = why
        rep = run(ctx, rounds=2, dwell=0.0, hit_rate=50.0, load_timeout=5.0)
        assert rep.passed, (mode, rep.summary)


def test_engine_check_fails_when_a_sampler_kit_was_never_taken(tmp_path: Path) -> None:
    ctx, dev = make_ctx(tmp_path, load_polls=1)
    dev.engine_mode = "failed"
    dev.why = "pool_refused"
    rep = run(ctx, rounds=2, dwell=0.0, hit_rate=50.0, load_timeout=5.0)
    assert not rep.passed
    assert any("holds no kit" in f and "pool_refused" in f for f in rep.data["failures"])


def test_engine_check_fails_when_a_take_never_finishes(tmp_path: Path) -> None:
    ctx, dev = make_ctx(tmp_path, load_polls=1)
    dev.take_polls = 10_000
    rep = run(ctx, rounds=1, dwell=0.0, hit_rate=50.0, load_timeout=0.3)
    assert not rep.passed
    assert any("never finished" in f for f in rep.data["failures"])
