"""Open and close every app in the launcher, over and over, and watch what leaks.

Port of platform-idf/tools/hil_app_churn.py. This is the demo-floor test: a
visitor picks an app, plays with it, backs out, picks another. Two failure
modes only show up here:

  * an app that never frees what create() allocated — internal heap trends
    down one step per visit, and after enough visits allocation starts failing
  * an app that leaves a dangling encoder group or widget pointer behind —
    the crash lands on the SECOND visit, not the first

The device is driven over CDC (APP_LIST / APP_START / APP_STOP), the internal
heap is sampled with MEM after every close, and the STM VCP console — opened
WITHOUT resetting the ESP — is watched for fatals and reboots.

Exit code 0 = PASS, 1 = leak / crash / stuck, 2 = device without HIL control
or an unknown app name.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from crosspad_hil import audio_state, verbs
from crosspad_hil.cdc import CdcLink
from crosspad_hil.console import Console
from crosspad_hil.errors import HilError
from crosspad_hil.scenarios import register
from crosspad_hil.scenarios.base import Artifact, Context, Progress, Report

# from hil_app_churn.DEFAULT_SKIP: apps that are not a "visit" — they power the
# device down, re-enumerate USB, or flash firmware. Churning them tests the
# harness, not the launcher.
DEFAULT_SKIP: tuple[str, ...] = ("Power OFF", "Firmware", "Settings")
KIT_SEED_WAIT_S = 20.0         # a cold kit load off the SD card
# How long verbs.app_start polls APP_LIST before a launch counts as refused.
START_WAIT_S: float = 3.0
# After a close, the board may still be moving: leaving Pitch Play gives its
# instrument back and the kit reloads, and a heap sampled mid-reload reads as
# a loss of that app. The sample waits for the kit to stand still (two quiet
# readings this far apart), up to KIT_SETTLE_S.
KIT_SETTLE_S = 10.0
KIT_QUIET_S = 0.5


def _wait_kit_settled(link: CdcLink, cancelled: Any) -> None:
    """Return once KIT_STATUS shows no load and none pending twice in a row,
    or after KIT_SETTLE_S."""
    quiet = 0
    waited = 0.0
    while waited < KIT_SETTLE_S and quiet < 2:
        ks = verbs.kit_status(link)
        quiet = quiet + 1 if not ks["loading"] and ks["pending"] < 0 else 0
        if quiet < 2:
            cancelled.wait(KIT_QUIET_S)
            waited += KIT_QUIET_S


@dataclass
class Params:
    # Four is the floor at which a leak is measurable at all: the head and tail
    # windows are centred 1.5 visits from each end, so three rounds leave them
    # zero visits apart and every app reports 0 B/visit however badly it leaks.
    # Eight puts a ~56 B/visit leak — the launcher's un-freed lv_group, real and
    # missed by this scenario for exactly this reason — at ~390 B of separation,
    # which is clear of the heap's own jitter.
    rounds: int = field(
        default=8, metadata={"help": "visits per app (4 is the measurable minimum)"})
    apps: list[str] | None = field(
        default=None,
        metadata={"help": "comma-separated subset; default is everything APP_LIST reports"},
    )
    skip: list[str] | None = field(
        default=None,
        metadata={"help": "comma-separated apps to leave alone "
                          "(default: Power OFF,Firmware,Settings)"},
    )
    dwell: float = field(
        default=1.0, metadata={"help": "seconds to stay inside an app before closing"}
    )
    settle: float = field(
        default=1.2, metadata={"help": "seconds after closing before sampling the heap"}
    )
    leak_bytes: int = field(
        default=32,
        # 2048 was the ported default, and it is not a threshold an app ever
        # crosses: the launcher's un-freed lv_group lost ~56 B per visit and sat
        # under it indefinitely. The bar has to be below what real leaks look
        # like, so it is set below that measurement. At eight rounds a 56 B/visit
        # loss shows as ~390 B between the head and tail windows; anything under
        # ~32 B/visit is closer to the heap's own jitter than to a leak, and a
        # bench that sees that as noise should raise this rather than have the
        # default hide the next 56 B one.
        metadata={"help": "per-visit internal-heap loss that counts as a leak"},
    )


def slope(series: list[int]) -> tuple[float, float, int, float] | None:
    """(head, tail, visits, drop) of a free-heap series, or None if too short.

    from hil_app_churn.report: compare the tail to the head rather than
    first-to-last — the first visit pays one-off costs (font glyph cache, lazily
    built tables) that are not a leak, and averaging both ends rejects jitter.

    ``drop`` is the raw fall from one window to the other. It is not a rate:
    the head window is centred on visit 2.5 and the tail on visit len-1.5, so
    the fall happened over ``visits - 1`` visits — see :func:`per_visit`.
    """
    if len(series) < 3:
        return None
    head = sum(series[1:3]) / 2
    tail = sum(series[-2:]) / 2
    visits = len(series) - 2
    return head, tail, visits, head - tail


def per_visit(drop: float, visits: int) -> float:
    """Bytes lost per visit: the drop spread over the visits it happened across.

    The two windows are centred 1.5 visits in from each end, so ``visits - 1``
    visits separate them — dividing by ``visits`` (what the old script did)
    under-reports a real leak by a third at five rounds, which is the difference
    between naming the app that leaks and shrugging at it.
    """
    span = visits - 1
    return drop / span if span > 0 else 0.0


class _ConsoleWatch:
    """New fatal / reboot events since the last look, from Console.events()."""

    def __init__(self, console: Console) -> None:
        self._console = console
        # Start from what the console has already seen. Opening the bridge VCP
        # resets the board, so a run that counted from zero would report the
        # reboot it caused itself as a device fault.
        self._handled = len(console.events(since_seq=0))

    def poll(self) -> tuple[list[str], int]:
        events = self._console.events(since_seq=0)
        fresh, self._handled = events[self._handled:], len(events)
        fatal = [ev.line for ev in fresh if ev.kind == "fatal"]
        reboots = sum(1 for ev in fresh if ev.kind == "reboot")
        return fatal, reboots



def _seed_kit_if_needed(ctx: Context, link: CdcLink) -> None:
    """Load a kit when none is current, so kit-backed apps can launch.

    Silent on a board with no kit manager and no SD card: that board has no
    gate either, so there is nothing to prepare and nothing to report.
    """
    try:
        if int(verbs.kit_status(link).get("current", -1)) >= 0:
            return
        kits = verbs.kit_list(link).get("kits", [])
    except HilError:
        return
    if not kits:
        return
    first = int(kits[0]["id"])
    ctx.log(f"no kit loaded — seeding with {kits[0].get('name', first)} so the Sampler can start")
    try:
        verbs.kit_load(link, first, wait_s=KIT_SEED_WAIT_S)
    except HilError as e:
        ctx.log(f"kit seed failed ({e.code}: {e.message}) — kit-backed apps may refuse to launch")


def _report(data: dict[str, Any], console_log: str, leak_bytes: int) -> Report:
    lines: list[str] = ["--- per-app free-heap trend " + "-" * 32]
    leaks: list[str] = []
    unmeasured: list[str] = []
    slopes: dict[str, dict[str, Any]] = {}
    for app, series in data["trend"].items():
        s = slope(series)
        if s is None:
            lines.append(f"  {app:<20} too few samples ({len(series)})")
            continue
        head, tail, visits, drop = s
        rate = per_visit(drop, visits)
        # With the windows zero visits apart there is no rate to compute, and
        # reporting 0 B/visit would be an answer rather than the absence of one.
        measurable = visits > 1
        leak = measurable and rate > leak_bytes
        slopes[app] = {
            "head": head, "tail": tail, "visits": visits,
            "per_visit": rate, "leak": leak, "measurable": measurable,
        }
        if not measurable:
            unmeasured.append(app)
            lines.append(f"  {app:<20} {int(head):>8} -> {int(tail):>8}  "
                         "(not measurable at this many rounds)")
            continue
        verdict = ""
        if leak:
            verdict = f"  <-- LEAK ~{int(rate)} B/visit"
            leaks.append(
                f"{app}: ~{int(rate)} B lost per visit "
                f"({int(drop)} B from head to tail of {visits} visits)"
            )
        lines.append(
            f"  {app:<20} {int(head):>8} -> {int(tail):>8}  "
            f"({int(rate):+6} B/visit){verdict}"
        )
    data["slopes"] = slopes
    data["leaks"] = leaks
    data["unmeasured"] = unmeasured
    failures = list(data["failures"]) + leaks
    lines.append("")
    if failures:
        lines.append(f"FAIL — {len(failures)} problem(s):")
        for f in failures:
            lines.append(f"  * {f}")
        passed = False
    else:
        if unmeasured:
            # Silence is not a pass. A run too short to measure a leak must say
            # so, or "no leak found" reads as "no leak", which is how a real one
            # survived this scenario.
            lines.append(f"PASS — no crashes; leak check INCONCLUSIVE for "
                         f"{len(unmeasured)} app(s) at these rounds (needs 4+): "
                         + ", ".join(unmeasured))
        else:
            lines.append(f"PASS — no crashes, no per-app heap slope over {leak_bytes} B/visit")
        passed = True
    lines.append(f"console log: {console_log}")
    return Report(
        passed=passed,
        summary="\n".join(lines),
        data=data,
        artifacts=[Artifact(path=console_log, mime="text/plain", role="console")],
        exit_code=0 if passed else 1,
    )


class AppChurnScenario:
    name = "app_churn"
    Params = Params
    description = "open/close every app repeatedly; per-app internal-heap slope + crash watch"

    def run(self, ctx: Context, params: Params, progress: Progress) -> Report:
        try:
            console_log = str(ctx.workdir / "console.log")
            data: dict[str, Any] = {
                "apps": [], "targets": [], "skipped": [], "rounds": params.rounds,
                "trend": {}, "slopes": {}, "leaks": [], "failures": [],
                "launch_fail": {}, "fatal": [], "reboots": 0, "leak_bytes": params.leak_bytes,
            }

            # Opened deasserted: the old script reset the ESP just by opening the
            # STM VCP, which threw away the state the churn was supposed to test.
            console = ctx.open_console(reset=False)
            link = ctx.open_cdc()
            watch = _ConsoleWatch(console)
            open_app: str | None = None
            # An app visited can leave audio state behind (a muted mixer channel, a codec
            # volume from the overlay); the next audio scenario would measure silence.
            audio_snap: dict[str, Any] | None = None
            try:
                audio_snap = audio_state.snapshot(link)
                listing = verbs.app_list(link)
                apps: list[str] = list(listing["apps"])
                running: str | None = listing["running"]
                data["apps"] = apps
                if not apps:
                    return Report(
                        passed=False,
                        summary="APP_LIST returned nothing — firmware without HIL control?",
                        data=data, artifacts=[], exit_code=2,
                    )
                if running:
                    verbs.app_stop(link)
                    ctx.cancelled.wait(params.settle)

                # A board with no current kit puts the kit-selector gate in
                # front of the Sampler, and APP_LIST then reports running=-
                # because the overlay is not the orchestrator's running app. The
                # launch is refused, and the churn reports that as the app
                # failing when it is the bench that is unprepared. Seed a kit so
                # every app in the rotation is actually launchable.
                _seed_kit_if_needed(ctx, link)

                skip = set(DEFAULT_SKIP if params.skip is None else params.skip)
                if params.apps:
                    wanted = [a for a in params.apps if a]
                    unknown = [a for a in wanted if a not in apps]
                    if unknown:
                        return Report(
                            passed=False,
                            summary=f"not in the launcher: {', '.join(unknown)}",
                            data=data, artifacts=[], exit_code=2,
                        )
                    targets = wanted
                    skipped: list[str] = []
                else:
                    targets = [a for a in apps if a not in skip]
                    skipped = [a for a in apps if a in skip]
                data["targets"] = targets
                data["skipped"] = skipped
                ctx.log(
                    f"launcher has {len(apps)} apps; churning {len(targets)}: {', '.join(targets)}"
                )
                if skipped:
                    ctx.log(f"skipped: {', '.join(skipped)}")
                ctx.log(f"{params.rounds} rounds, dwell {params.dwell}s")

                # (app -> [int_free after each visit]); failures are collected, not
                # raised, so one broken app does not hide the rest.
                trend: dict[str, list[int]] = {a: [] for a in targets}
                failures: list[str] = []
                launch_fail: dict[str, int] = {a: 0 for a in targets}
                data["trend"] = trend
                data["failures"] = failures
                data["launch_fail"] = launch_fail
                total = params.rounds * len(targets)
                visit = 0

                for rnd in range(1, params.rounds + 1):
                    for app in targets:
                        ctx.check_cancelled()
                        visit += 1
                        progress(visit, total, f"round {rnd}/{params.rounds} {app}")
                        # One MEM before the launch and one after the close, so
                        # what this visit failed to hand back is measured on its
                        # own. A sample taken only after the close also carries
                        # everything the apps visited in between lost: with one
                        # global heap and N apps in the rotation, the single
                        # leaking app's loss is smeared over all N, and no app
                        # crosses the threshold that would name it.
                        free_before = int(verbs.mem(link)["int_free"])
                        try:
                            verbs.app_start(link, app, wait_s=START_WAIT_S)
                        except HilError as e:
                            launch_fail[app] += 1
                            failures.append(
                                f"{app}: APP_START refused ({e.code}: {e.message}) on round {rnd}"
                            )
                            continue
                        open_app = app
                        ctx.cancelled.wait(params.dwell)
                        verbs.app_stop(link)
                        open_app = None
                        ctx.cancelled.wait(params.settle)
                        _wait_kit_settled(link, ctx.cancelled)

                        fatal, reboots = watch.poll()
                        if fatal or reboots:
                            data["fatal"].extend(fatal)
                            data["reboots"] += reboots
                            for f in fatal[-5:]:
                                ctx.log(f"   {f}")
                            what = "device fault" if fatal else "device rebooted"
                            failures.append(f"{app}: {what} on round {rnd} — see {console_log}")
                            return _report(data, console_log, params.leak_bytes)

                        free_after = int(verbs.mem(link)["int_free"])
                        series = trend[app]
                        series.append((series[-1] if series else free_before)
                                      - (free_before - free_after))

                    # A round that never got the device back to the launcher means
                    # the state machine is stuck, and every later measurement is noise.
                    running = verbs.app_list(link)["running"]
                    m = verbs.mem(link)
                    heap_now = int(m["int_free"])
                    psram = (f"  psram_free={m.get('psram_free')}"
                             f" psram_largest={m.get('psram_largest')}")
                    data.setdefault("psram", []).append(
                        {"round": rnd, "free": m.get("psram_free"),
                         "largest": m.get("psram_largest")})
                    stuck = f" STUCK in {running}" if running else ""
                    ctx.log(f"round {rnd:3d}/{params.rounds}  int_free={heap_now}{stuck}{psram}")
                    if running:
                        failures.append(
                            f"round {rnd}: device did not return to the launcher (in {running})"
                        )
                        verbs.app_stop(link)
                        ctx.cancelled.wait(params.settle)

                return _report(data, console_log, params.leak_bytes)
            finally:
                if open_app is not None:
                    # An error between APP_START and APP_STOP left this visit's
                    # app on screen: close it so the board ends in the launcher,
                    # as found. Best-effort; the original error is what is raised.
                    try:
                        verbs.app_stop(link)
                    except HilError as e:
                        ctx.log(f"{open_app} left open: APP_STOP failed ({e.code}: {e.message})")
                audio_state.put_back(link, audio_snap, ctx.log, data)
                link.close()
                console.close()


        finally:
            ctx.close_all()
SCENARIO = register(AppChurnScenario())
