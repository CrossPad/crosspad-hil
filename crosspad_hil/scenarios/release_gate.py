"""Release RC gate for platform-idf: chain the whole checklist, one report, one exit code.

Composes the existing scenarios in-process, back to back, on one device:

    smoke -> power_state -> app_churn -> back_churn -> kit_churn -> perf -> beat
          -> usb_mode_cycle -> ble_midi -> audio_loopback -> stability

``beat`` is the user's own task end to end: a prepared loop sliced into a kit,
a beat sequenced on it, bounced and checked hit by hit (``--beat-depth``,
medium by default, quick under ``--quick``).

``perf`` is the one step that judges speed rather than survival: kit swap time
(CDC, and USB audio under a host stream) and the sample engine's init, against
thresholds. Without it the gate passed a release whose kit swap had gone from
1.6 s to 2.8 s (4.0 s in the product's audio profile).

Each step runs through its own :class:`~crosspad_hil.scenarios.base.Scenario`
exactly as ``crosspad-hil run <step>`` would, in its own sub-workdir (so its
console log / events / report.json land where every other run of that
scenario would put them) — but *without* going through
:func:`~crosspad_hil.scenarios.base.run_scenario` for the step itself. That
function double-checks a lot that a top-level run needs (device discovery,
the cross-process port lock, report.json for a standalone invocation), and
the port lock is keyed on the device id: a nested ``run_scenario`` call for
step 2 while the gate's own top-level ``run_scenario("release_gate", ...)``
still holds that same lock would see its own pid on the lock file and fail
with ``PORT_BUSY``. So ``_run_step`` below is a trimmed copy of
``run_scenario``'s try/except/report.json body — same exception handling,
same report shape — minus the locking and device discovery, which the gate
already did once for the whole run.

MEM (int_free / psram_free) is sampled over CDC immediately before and after
every step, best-effort: a step that leaves the board mid audio-profile
switch has no CDC to read, and that is logged rather than failed.

Exit code: 0 every step passed, 1 any step FAILed (a real firmware problem —
takes priority even if another step also reported a bench issue), 2 no step
FAILed but at least one hit an environment problem (missing test hardware /
optional dependency, device unreachable, ...). ``--keep-going`` runs every
step regardless of an earlier FAIL, for the full picture in a release
write-up; the default stops at the first FAIL so a bad build fails fast.

``--quick`` shrinks every step to a CI-length dry run of the *gate itself* —
proof the chain wires together and the device survives one lap, not a release
verdict. A real go/no-go needs the full run (``audio_loopback`` for an hour,
``stability`` for two, by default).
"""
from __future__ import annotations

import dataclasses
import faulthandler
import json
import threading
import time
import traceback
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from crosspad_hil import board_state, verbs
from crosspad_hil.devices import Device, UsbMode, discover
from crosspad_hil.errors import BAD_ARGS, CANCELLED, HilError
from crosspad_hil.scenarios import get, register
from crosspad_hil.scenarios.base import ENV_EXIT_CODES, Artifact, Context, Progress, Report
from crosspad_hil.usbmode import set_mode

STEP_ORDER: tuple[str, ...] = (
    "smoke", "power_state", "app_churn", "back_churn", "kit_churn", "perf", "beat",
    "usb_mode_cycle",
    "ble_midi", "audio_loopback", "stability",
)

# --quick: enough of each step to prove the chain runs end to end and the
# board survives one lap of it — not enough to certify a release. A real
# go/no-go still needs the full-length run.
QUICK_SMOKE_TIMEOUT_S = 12
QUICK_POWER_SETTLE_S = 0.0          # power_state: one sample, profile only — no watchdog verdict
QUICK_APP_CHURN_ROUNDS = 4          # app_churn's own floor for a measurable leak slope
QUICK_BACK_CHURN_ROUNDS = 3          # back_churn: enough visits for a slope, quick lap
QUICK_KIT_CHURN_ROUNDS = 6
QUICK_PERF_SWAPS = 2
QUICK_USB_MODE_ROUNDS = 2
QUICK_BLE_SCAN_TIMEOUT_S = 6.0
QUICK_BLE_CONNECT_TIMEOUT_S = 8.0
QUICK_AUDIO_LOOPBACK_HOURS = 0.0    # audio_loopback's own "0 = one cycle and stop"
QUICK_STABILITY_HOURS = 2.0 / 60.0  # 2 minutes
QUICK_STABILITY_SILENCE_S = 30.0    # short enough to still mean something in 2 minutes

# Per-step watchdog. A step run in-process that never returns (a firmware wedge, or a
# blocking call in a library the step depends on with no timeout of its own — exactly
# what happened once in usb_mode_cycle) used to hang the whole gate forever, since
# nothing ever bounded scenario.run() itself. STEP_TIMEOUT_MIN_S is the floor for every
# short step; steps that carry their own duration (audio_loopback, stability) size their
# budget off that instead, times --step-timeout-factor, so a real multi-hour run is not
# mistaken for a hang.
STEP_TIMEOUT_MIN_S = 120.0
# After the budget, a stuck step gets a cooperative cancel and this long to wind down
# (close its links, release the port locks) before it is written off as abandoned.
STEP_CANCEL_GRACE_S = 30.0
# The gate's own thread is supposed to report a step this long after its budget
# (cancel grace plus the report writing); if it has not, the gate itself is stuck,
# and the only useful thing left is a stack dump of every thread.
STALL_MARGIN_S = 120.0
# Per-visit / per-round costs measured on the bench, used to size a step's budget.
APP_CHURN_VISIT_OVERHEAD_S = 2.0    # APP_START + APP_LIST + APP_STOP on top of dwell+settle
APP_CHURN_APPS_GUESS = 8            # visits per round when the launcher list is not known yet
# One lap of every app, --depth levels deep, clicked back out. Measured 9.09 on
# the bench with ten targets at depth 3: 2/6/8 rounds took 132/133/139 s per
# round. The old figure of 60 s gave eight rounds a 17-minute budget, and the
# gate wrote TIMEOUT over a run that was passing.
BACK_CHURN_ROUND_S = 140.0
KIT_CHURN_SWAP_OVERHEAD_S = 4.0     # a load takes 1.5-3.5 s on top of the dwell (4-bit card)
USB_MODE_CYCLE_S = 40.0             # two re-enumerations per cycle, ~15 s each on a slow hub
STEP_SETUP_S = 30.0                 # discovery, kit seeding, the MEM reads around a step
BEAT_STEP_S = {"quick": 240.0, "medium": 480.0, "full": 900.0}   # beat, by depth
PERF_BOOT_S = 60.0                  # perf resets the board to time the engine init
PERF_SWAP_OVERHEAD_S = 5.0          # one measured swap (2.8-4 s on the slow builds) + polling


@dataclass
class Params:
    quick: bool = field(
        default=False,
        metadata={"help": "shrink every step to a CI-length dry run of the gate itself "
                          "(not a release verdict)"},
    )
    keep_going: bool = field(
        default=False,
        metadata={"help": "run every remaining step after one FAILs, instead of stopping "
                          "at the first"},
    )
    beat_depth: str = field(
        default="medium", metadata={"help": "beat: quick | medium | full (quick under --quick)"})
    app_churn_rounds: int = field(
        default=10, metadata={"help": "app_churn: visits per app"})
    kit_churn_rounds: int = field(
        default=20, metadata={"help": "kit_churn: kit swaps to perform"})
    kit_churn_rapid: float = field(
        default=0.4,
        metadata={"help": "kit_churn: seconds between kit requests fired without waiting "
                          "for the previous load (the selector-gate coalescing path)"},
    )
    kit_churn_load_timeout: float | None = field(
        default=None,
        metadata={"help": "kit_churn: seconds a single kit load may take (default: the "
                          "scenario's own); a 1-bit SD card needs ~30, and the step's "
                          "budget grows with it"},
    )
    perf_swaps: int = field(
        default=6, metadata={"help": "perf: kit swaps per USB profile (1..8)"})
    perf_max_cdc_ms: float = field(
        default=2000.0, metadata={"help": "perf: FAIL when the CDC median kit swap exceeds "
                                          "this (ms)"})
    perf_max_uac_ratio: float = field(
        default=1.2, metadata={"help": "perf: FAIL when the USB-audio median exceeds this "
                                       "times the CDC median"})
    perf_max_init_ms: float = field(
        default=500.0, metadata={"help": "perf: FAIL when the sample engine's init at boot "
                                         "exceeds this (ms)"})
    perf_stream: bool = field(
        default=True, metadata={"help": "perf: stream a tone into the board (pw-play) during "
                                        "the USB-audio swaps"})
    usb_mode_rounds: int = field(
        default=5, metadata={"help": "usb_mode_cycle: default<->audio cycles"})
    ble_midi_kit: int | None = field(
        default=None,
        metadata={"help": "ble_midi: load this kit first so the note-to-sample path is "
                          "exercised too"},
    )
    audio_loopback_hours: float = field(
        default=1.0, metadata={"help": "audio_loopback: total run time in hours"})
    stability_hours: float = field(
        default=2.0, metadata={"help": "stability: soak run time in hours"})
    strict_state: bool = field(
        default=False,
        metadata={"help": "a step that leaves persistent board state changed (BLE, USB guard, "
                          "WiFi, capture route, input gain) FAILs instead of a WARN"})
    step_timeout_factor: float = field(
        default=2.0,
        metadata={"help": "per-step watchdog: multiplies a step's expected duration "
                          f"(floor {STEP_TIMEOUT_MIN_S:.0f}s); a step stuck past that is "
                          "abandoned and reported as TIMEOUT instead of hanging the gate"},
    )


def _step_params(step: str, p: Params) -> Any:
    """Build the Params dataclass instance for one composed step."""
    cls = get(step).Params
    if step == "smoke":
        return cls(timeout=QUICK_SMOKE_TIMEOUT_S) if p.quick else cls()
    if step == "power_state":
        return cls(settle_s=QUICK_POWER_SETTLE_S) if p.quick else cls()
    if step == "app_churn":
        return cls(rounds=QUICK_APP_CHURN_ROUNDS if p.quick else p.app_churn_rounds)
    if step == "back_churn":
        return cls(rounds=QUICK_BACK_CHURN_ROUNDS if p.quick else 8)
    if step == "kit_churn":
        kw: dict[str, Any] = {}
        if p.kit_churn_load_timeout is not None:
            kw["load_timeout"] = p.kit_churn_load_timeout
        return cls(
            rounds=QUICK_KIT_CHURN_ROUNDS if p.quick else p.kit_churn_rounds,
            rapid=p.kit_churn_rapid,
            **kw,
        )
    if step == "perf":
        return cls(swaps=QUICK_PERF_SWAPS if p.quick else p.perf_swaps,
                   max_cdc_ms=p.perf_max_cdc_ms, max_uac_ratio=p.perf_max_uac_ratio,
                   max_init_ms=p.perf_max_init_ms, stream=p.perf_stream)
    if step == "beat":
        return cls(depth="quick" if p.quick else p.beat_depth)
    if step == "usb_mode_cycle":
        return cls(rounds=QUICK_USB_MODE_ROUNDS if p.quick else p.usb_mode_rounds)
    if step == "ble_midi":
        if p.quick:
            return cls(kit=p.ble_midi_kit, scan_timeout=QUICK_BLE_SCAN_TIMEOUT_S,
                       connect_timeout=QUICK_BLE_CONNECT_TIMEOUT_S)
        return cls(kit=p.ble_midi_kit)
    if step == "audio_loopback":
        return cls(duration_hours=QUICK_AUDIO_LOOPBACK_HOURS if p.quick
                  else p.audio_loopback_hours)
    if step == "stability":
        if p.quick:
            return cls(duration_hours=QUICK_STABILITY_HOURS,
                       silence_timeout=QUICK_STABILITY_SILENCE_S)
        return cls(duration_hours=p.stability_hours)
    raise HilError(BAD_ARGS, f"release_gate: unknown step {step!r}")


def _expected_step_seconds(step: str, step_params: Any) -> float:
    """Rough wall-clock budget a step needs, used only to size its watchdog timeout.

    Steps carrying an explicit ``duration_hours`` (audio_loopback, stability) dominate
    the runtime by design. The churn steps scale with their rounds: ten rounds of
    app_churn are ~50 visits at ~3.5 s each, which the old flat floor of two minutes
    cut off at round seven -- and an abandoned step kept the ports, so every step
    after it reported PORT_BUSY as if the board had a problem.
    """
    hours = getattr(step_params, "duration_hours", None)
    if hours is not None:
        return float(hours) * 3600.0
    rounds = float(getattr(step_params, "rounds", 0) or 0)
    dwell = float(getattr(step_params, "dwell", 0.0) or 0.0)
    settle = float(getattr(step_params, "settle", 0.0) or 0.0)
    if step == "app_churn":
        apps = getattr(step_params, "apps", None)
        per_round = len(apps) if apps else APP_CHURN_APPS_GUESS
        return STEP_SETUP_S + rounds * per_round * (dwell + settle + APP_CHURN_VISIT_OVERHEAD_S)
    if step == "back_churn":
        return STEP_SETUP_S + rounds * BACK_CHURN_ROUND_S
    if step == "kit_churn":
        rapid = getattr(step_params, "rapid", None)
        # A swap may legitimately take up to load_timeout before the scenario
        # calls it stuck, so the watchdog has to outlast that, not the typical
        # load — otherwise a slow card trips the gate before kit_churn can
        # report anything.
        load_timeout = float(getattr(step_params, "load_timeout", 0.0) or 0.0)
        per_swap = (float(rapid) if rapid else dwell) \
            + max(KIT_CHURN_SWAP_OVERHEAD_S, load_timeout)
        return STEP_SETUP_S + rounds * per_swap
    if step == "perf":
        swaps = float(getattr(step_params, "swaps", 0) or 0)
        phases = 2.0 if getattr(step_params, "uac", True) else 1.0
        per_swap = PERF_SWAP_OVERHEAD_S + float(getattr(step_params, "settle", 0.0) or 0.0)
        return STEP_SETUP_S + PERF_BOOT_S + USB_MODE_CYCLE_S + phases * swaps * per_swap
    if step == "beat":
        return STEP_SETUP_S + BEAT_STEP_S.get(getattr(step_params, "depth", "quick"), 900.0)
    if step == "usb_mode_cycle":
        return STEP_SETUP_S + rounds * (USB_MODE_CYCLE_S + 2.0 * dwell)
    if step == "power_state":
        # Waits for STM uptime past the chip watchdog, bounded by max_wait_s.
        return STEP_SETUP_S + float(getattr(step_params, "max_wait_s", 0.0) or 0.0)
    return 0.0


def _step_timeout_s(step: str, step_params: Any, factor: float) -> float:
    return max(STEP_TIMEOUT_MIN_S, _expected_step_seconds(step, step_params) * factor)


def _redetect_device(device_id: str, log: Callable[[str], None]) -> Device | None:
    """Best-effort: is the device even still enumerated after a step timed out?

    A step abandoned mid-hang may still be running on its own thread (see _run_step)
    and holding a port lock or a serial handle — starting the next step on the same
    device before checking is how one hang becomes a chain of PORT_BUSY failures that
    say nothing new. This is read-only and itself bounded by discover()'s own
    subprocess timeouts.
    """
    try:
        found = [d for d in discover() if d.id == device_id]
    except Exception as exc:  # noqa: BLE001 - a discovery hiccup must not crash the gate
        log(f"device recheck after timeout failed: {exc!r}")
        return None
    if not found:
        log(f"device {device_id} not found after timeout — treating it as lost")
        return None
    log(f"device {device_id} still enumerated (mode={found[0].usb_mode.value}) after timeout")
    return found[0]


def _ensure_default_profile(device: Device, log: Callable[[str], None]) -> Device:
    """Put the board back in the CDC-bearing profile before a step starts.

    A gate killed inside audio_loopback -- or any earlier session -- leaves the
    board in MIDI+UAC2, which has no CDC endpoint. Every CDC step then reports
    NO_CDC_IN_AUDIO_MODE as ENV, one after another, and the run says nothing
    about the firmware. The device record is re-read here because the one the
    gate holds is a snapshot from before the previous step.
    """
    try:
        found = [d for d in discover() if d.id == device.id]
    except Exception as exc:  # noqa: BLE001 - a discovery hiccup must not crash the gate
        log(f"profile check skipped, discovery failed: {exc!r}")
        return device
    if not found:
        return device
    fresh = found[0]
    if fresh.usb_mode is not UsbMode.AUDIO:
        return fresh
    log(f"device {device.id} is in the {fresh.usb_mode.value} profile "
        "(no CDC) -- switching back to default before the step")
    try:
        return set_mode(fresh, UsbMode.DEFAULT)
    except HilError as exc:
        log(f"could not restore the default profile: {exc.code}: {exc.message}")
        return fresh


class _StallWatch:
    """Dump every thread's stack when the gate stops making progress.

    A step is bounded by ``_run_step``'s watchdog -- but that watchdog runs on the
    gate's own thread. Once, a run sat 55 minutes inside usb_mode_cycle with an
    8-minute budget, every thread parked on a futex, no report and no log line:
    the watchdog thread was the stuck one. This timer lives on its own thread and
    writes ``stall_<step>.txt`` (faulthandler, all threads) into the gate's
    workdir when a step is still unreported ``STALL_MARGIN_S`` past its budget
    and grace, then re-arms so a long hang leaves several snapshots.
    """

    def __init__(self, workdir: Path, log: Callable[[str], None]) -> None:
        self._workdir = workdir
        self._log = log
        self._timer: threading.Timer | None = None
        self._lock = threading.Lock()

    def arm(self, step: str, after_s: float) -> None:
        self.disarm()
        t = threading.Timer(after_s, self._fire, args=(step, after_s))
        t.daemon = True
        t.name = f"gate-stall-{step}"
        with self._lock:
            self._timer = t
        t.start()

    def disarm(self) -> None:
        with self._lock:
            t, self._timer = self._timer, None
        if t is not None:
            t.cancel()

    def _fire(self, step: str, after_s: float) -> None:
        path = self._workdir / f"stall_{step}.txt"
        try:
            with path.open("a", encoding="utf-8") as fh:
                fh.write(f"--- {time.strftime('%Y-%m-%d %H:%M:%S')} step={step} "
                         f"unreported {after_s:.0f}s past its budget ---\n")
                fh.flush()
                faulthandler.dump_traceback(file=fh, all_threads=True)
        except OSError as exc:
            self._log(f"[{step}] stall detected but the stack dump failed: {exc!r}")
            return
        self._log(f"[{step}] STALL: still unreported {after_s:.0f}s past its budget -- "
                  f"stacks of every thread in {path}")
        # Keep sampling: a hang that moves (or does not) between dumps says more.
        self.arm(step, STALL_MARGIN_S * 5)


def _prefixed_log(log: Callable[[str], None], step: str) -> Callable[[str], None]:
    def inner(line: str) -> None:
        log(f"[{step}] {line}")
    return inner


def _read_mem(step_ctx: Context) -> dict[str, Any] | None:
    """Best-effort MEM snapshot; None when CDC is unreachable right now.

    A step that leaves the board mid USB-profile switch (or failed to restore
    it) has no CDC to read from — that is logged on the step's own context,
    not raised, so a MEM sample never turns into a gate failure of its own.
    """
    try:
        link = step_ctx.open_cdc()
    except HilError as e:
        step_ctx.log(f"MEM read skipped: {e.code}: {e.message}")
        return None
    try:
        return verbs.mem(link)
    except HilError as e:
        step_ctx.log(f"MEM read failed: {e.code}: {e.message}")
        return None
    finally:
        link.close()


def _read_state(step_ctx: Context) -> dict[str, Any] | None:
    """Best-effort board_state reading; None when CDC is unreachable right now."""
    try:
        link = step_ctx.open_cdc()
    except HilError as e:
        step_ctx.log(f"state read skipped: {e.code}: {e.message}")
        return None
    try:
        return board_state.read(link)
    finally:
        link.close()


def _partial_snapshot(step_ctx: Context) -> dict[str, Any]:
    """What the step had collected, safe to read while it may still be writing."""
    try:
        return dict(step_ctx.partial)
    except RuntimeError:          # changed size during iteration
        return {}


def _run_step(
    name: str, scenario: Any, step_ctx: Context, step_params: Any,
    sink: Progress, log: Callable[[str], None], timeout_s: float,
    gate_cancelled: threading.Event,
) -> tuple[Report, float, float, bool]:
    """One step of the gate: same exception handling and report.json shape as
    ``run_scenario``, minus the device lock and discovery the gate already did
    once for the whole run (see the module docstring for why nesting
    ``run_scenario`` itself is not an option here).

    ``scenario.run()`` executes on a daemon thread that this function bounds with
    ``timeout_s``: a step is expected to enforce its own internal timeouts, but the
    gate runs overnight and unattended, so one library call somewhere below it with
    no timeout of its own (it has happened — see usbmode.py's SEND_TIMEOUT_S) must
    not be able to hang the whole run. A thread cannot be killed, only abandoned: if
    it never returns, this reports TIMEOUT and moves on. Before giving up on it the
    step gets a cooperative cancel on its *own* event (``step_ctx.cancelled`` is
    private to the step; the gate's cancel is forwarded to it while it runs) and
    ``STEP_CANCEL_GRACE_S`` to wind down -- the churn scenarios check it every
    visit, so they normally exit here and release their ports. Only a step that
    ignores that is abandoned: the thread — daemon, so it will not keep the
    process alive — keeps running and does its own ``step_ctx.close_all()``
    whenever (if ever) it finishes, since the main thread must not touch that
    context's resources while the abandoned thread still might.
    """
    started = time.time()
    log(f"[{name}] device={step_ctx.device.id} workdir={step_ctx.workdir} "
        f"timeout={_fmt_seconds(timeout_s)}")
    box: dict[str, Report] = {}

    def worker() -> None:
        try:
            report = scenario.run(step_ctx, step_params, sink)
            if not report.passed and report.exit_code == 0:
                report.exit_code = 1
        except HilError as e:
            if e.code == CANCELLED:
                report = Report(passed=False, summary=f"cancelled: {e.message}",
                                data={**step_ctx.partial, "cancelled": True, "error": e.to_dict()},
                                artifacts=[], exit_code=1)
            elif e.code in ENV_EXIT_CODES:
                report = Report(passed=False, summary=f"environment: {e.code}: {e.message}",
                                data={**step_ctx.partial, "error": e.to_dict()},
                                artifacts=[], exit_code=2)
            else:
                report = Report(passed=False, summary=f"{e.code}: {e.message}",
                                data={**step_ctx.partial, "error": e.to_dict()},
                                artifacts=[], exit_code=1)
            log(f"[{name}] {report.summary}")
        except Exception as exc:  # noqa: BLE001 - a step crash must still produce a report
            report = Report(
                passed=False,
                summary=f"crashed: {exc!r}",
                data={**step_ctx.partial,
                      "error": {"code": "INTERNAL", "message": repr(exc), "hint": None,
                                "details": {}, "traceback": traceback.format_exc()}},
                artifacts=[],
                exit_code=1,
            )
            log(f"[{name}] {report.summary}")
        finally:
            try:
                step_ctx.close_all()
            except Exception:  # noqa: BLE001 - teardown must not mask the step result
                pass
        box["report"] = report

    thread = threading.Thread(target=worker, name=f"gate-step-{name}", daemon=True)
    thread.start()
    deadline = time.monotonic() + timeout_s
    while thread.is_alive() and time.monotonic() < deadline:
        thread.join(0.5)
        if gate_cancelled.is_set():
            # A real task.cancel() on the gate reaches the running step this way.
            step_ctx.cancelled.set()
    timed_out = thread.is_alive()
    if timed_out:
        step_ctx.cancelled.set()
        thread.join(STEP_CANCEL_GRACE_S)
        abandoned = thread.is_alive()
        if abandoned:
            tail = ("abandoned: it ignored the cancel, keeps running in the background "
                    "and still holds its ports, so later steps may report PORT_BUSY")
        else:
            tail = "stopped on cancel and released its ports"
        report = Report(
            passed=False,
            summary=f"TIMEOUT: step exceeded its {_fmt_seconds(timeout_s)} budget; {tail} "
                    "(this TIMEOUT report.json is final)",
            # Copied, not splatted: an abandoned step is still running and may
            # still be writing into partial, and iterating a dict another thread
            # is growing raises. A step that hangs is the case with the most to
            # lose — whatever it had measured before it stopped answering is all
            # the evidence there will ever be.
            data={**_partial_snapshot(step_ctx), "timeout": True,
                  "timeout_s": timeout_s, "abandoned": abandoned},
            artifacts=[],
            exit_code=1,
        )
        log(f"[{name}] {report.summary}")
    else:
        report = box["report"]
    finished = time.time()

    report_path = step_ctx.workdir / "report.json"
    if not any(a.role == "report" for a in report.artifacts):
        report.artifacts.append(
            Artifact(path=str(report_path), mime="application/json", role="report"))
    doc = {
        "scenario": name,
        "device": step_ctx.device.id,
        "passed": report.passed,
        "summary": report.summary,
        "exit_code": report.exit_code,
        "data": report.data,
        "artifacts": [dataclasses.asdict(a) for a in report.artifacts],
        "params": dataclasses.asdict(step_params) if dataclasses.is_dataclass(step_params)
                  and not isinstance(step_params, type) else {},
        "workdir": str(step_ctx.workdir),
        "started": started,
        "finished": finished,
        "seconds": round(finished - started, 3),
    }
    report_path.write_text(json.dumps(doc, indent=2, default=str) + "\n", encoding="utf-8")
    log(f"[{name}] {'TIMEOUT' if timed_out else 'PASS' if report.passed else 'FAIL'} "
        f"exit={report.exit_code} report={report_path}")
    return report, started, finished, timed_out


def _combine_exit(current: int, new: int) -> int:
    """A real FAIL (1) always wins; otherwise the worse of the two (ENV > pass)."""
    if current == 1 or new == 1:
        return 1
    return max(current, new)


def _fmt_mem(m: dict[str, Any] | None) -> str:
    if not m:
        return "n/a"
    return f"int={m.get('int_free', '?')} psram={m.get('psram_free', '?')}"


def _fmt_seconds(s: float) -> str:
    if s >= 3600:
        return f"{s / 3600:.2f}h"
    if s >= 60:
        return f"{s / 60:.1f}m"
    return f"{s:.1f}s"


def _table(results: list[dict[str, Any]]) -> str:
    header = f"{'step':<16} {'result':<7} {'time':>8}  {'mem before':<22} {'mem after':<22}"
    lines = [header, "-" * len(header)]
    for r in results:
        lines.append(
            f"{r['step']:<16} {r['result']:<7} {_fmt_seconds(r['seconds']):>8}  "
            f"{_fmt_mem(r['mem_before']):<22} {_fmt_mem(r['mem_after']):<22}"
        )
    return "\n".join(lines)


def _summary(results: list[dict[str, Any]], stopped_at: str | None, stop_reason: str,
            overall_passed: bool, params: Params) -> str:
    head = f"release_gate: {'PASS' if overall_passed else 'FAIL'}"
    if stopped_at:
        head += f" — stopped after {stopped_at} {stop_reason}"
    if params.quick:
        head += "  [--quick: CI-length dry run of the gate, not a release verdict]"
    lines = [head, "", _table(results)]
    for r in results:
        if r["result"] not in ("PASS", "SKIP"):
            lines.append(f"\n{r['step']}: {r['summary']}")
    for r in results:
        if r.get("state_left") and r["result"] == "PASS":
            lines.append(f"\nWARN {r['step']}: left the board changed: "
                         f"{board_state.describe(r['state_left'])}"
                         " (--strict-state makes this a FAIL)")
    return "\n".join(lines)


class ReleaseGateScenario:
    name = "release_gate"
    Params = Params
    description = ("RC gate: smoke -> app_churn -> back_churn -> kit_churn -> perf -> beat -> "
                   "usb_mode_cycle -> ble_midi -> audio_loopback -> stability, one report, "
                   "one exit code")

    def run(self, ctx: Context, params: Params, progress: Progress) -> Report:
        try:
            current_device = ctx.device
            results: list[dict[str, Any]] = []
            artifacts: list[Artifact] = []
            stopped_at: str | None = None
            stop_reason = "FAILed and --keep-going was not set"
            overall_passed = True
            worst_code = 0
            total = len(STEP_ORDER)
            stall = _StallWatch(ctx.workdir, ctx.log)

            for i, step in enumerate(STEP_ORDER, start=1):
                ctx.check_cancelled()
                current_device = _ensure_default_profile(current_device, ctx.log)
                step_dir = ctx.workdir / step
                step_dir.mkdir(parents=True, exist_ok=True)
                # The step's own cancel event: a timed-out step can be told to stop
                # without cancelling the steps still to come. _run_step forwards the
                # gate's cancel into it while the step runs.
                step_ctx = Context(
                    device=current_device,
                    workdir=step_dir,
                    cancelled=threading.Event(),
                    log=_prefixed_log(ctx.log, step),
                    serial_factory=ctx.serial_factory,
                    holder=ctx.holder,
                )
                progress(i - 1, total, f"[{i}/{total}] {step}: starting")
                mem_before = _read_mem(step_ctx)
                state_before = _read_state(step_ctx)

                scenario = get(step)
                step_params = _step_params(step, params)
                timeout_s = _step_timeout_s(step, step_params, params.step_timeout_factor)
                stall.arm(step, timeout_s + STEP_CANCEL_GRACE_S + STALL_MARGIN_S)
                try:
                    report, started, finished, timed_out = _run_step(
                        step, scenario, step_ctx, step_params, progress, ctx.log, timeout_s,
                        ctx.cancelled)
                finally:
                    stall.disarm()

                device_lost = False
                if timed_out:
                    # The abandoned thread may still be touching step_ctx's device
                    # (a lock, a serial handle) — a fresh, read-only discover() is
                    # the honest way to learn what state the board is actually in
                    # before pointing the next step at it.
                    fresh = _redetect_device(step_ctx.device.id, ctx.log)
                    if fresh is None:
                        device_lost = True
                    else:
                        current_device = fresh
                    mem_after = None
                    state_after = None
                else:
                    current_device = step_ctx.device
                    mem_after = _read_mem(step_ctx)
                    state_after = _read_state(step_ctx)

                # A step is supposed to hand the board on the way it found it.
                changes = board_state.diff(state_before, state_after)
                left = [c for c in changes if c["persistent"]]
                step_passed = report.passed
                step_code = report.exit_code
                if left:
                    ctx.log(f"[{step}] left the board changed: {board_state.describe(left)}")
                    if params.strict_state and step_passed:
                        step_passed, step_code = False, 1

                result = ("TIMEOUT" if timed_out
                          else "PASS" if step_passed
                          else "ENV" if step_code == 2 else "FAIL")
                if result == "FAIL" and report.passed:
                    summary_line = f"left the board changed: {board_state.describe(left)}"
                else:
                    summary_line = ((report.summary or "").splitlines()[0]
                                    if report.summary else "")
                results.append({
                    "step": step, "result": result, "passed": step_passed,
                    "exit_code": step_code, "seconds": round(finished - started, 3),
                    "summary": summary_line,
                    "mem_before": mem_before, "mem_after": mem_after,
                    "state_before": state_before, "state_after": state_after,
                    "state_changes": changes, "state_left": left,
                    "workdir": str(step_dir), "timed_out": timed_out,
                })
                step_report = step_dir / "report.json"
                artifacts.append(Artifact(path=str(step_report), mime="application/json",
                                         role=f"report:{step}"))
                step_console = step_dir / "console.log"
                if step_console.exists():
                    artifacts.append(Artifact(path=str(step_console), mime="text/plain",
                                             role=f"console:{step}"))

                progress(i, total, f"[{i}/{total}] {step}: {result}")
                worst_code = _combine_exit(worst_code, step_code)
                if not step_passed:
                    overall_passed = False
                    # A step that timed out *and* left the device unreachable is
                    # worse than an ordinary FAIL: nothing downstream has a device
                    # to run against, so this stops the gate even under
                    # --keep-going.
                    if device_lost:
                        stopped_at = step
                        stop_reason = "TIMEOUT left the device unreachable"
                        break
                    if not params.keep_going:
                        stopped_at = step
                        break

            if stopped_at is not None:
                idx = STEP_ORDER.index(stopped_at)
                for step in STEP_ORDER[idx + 1:]:
                    results.append({
                        "step": step, "result": "SKIP", "passed": None, "exit_code": None,
                        "seconds": 0.0, "summary": f"skipped — {stopped_at} {stop_reason}",
                        "mem_before": None, "mem_after": None, "workdir": None,
                    })

            data: dict[str, Any] = {
                "steps": results, "stopped_at": stopped_at, "quick": params.quick,
                "keep_going": params.keep_going,
            }
            return Report(
                passed=overall_passed,
                summary=_summary(results, stopped_at, stop_reason, overall_passed, params),
                data=data,
                artifacts=artifacts,
                exit_code=worst_code,
            )
        finally:
            ctx.close_all()


SCENARIO = register(ReleaseGateScenario())
