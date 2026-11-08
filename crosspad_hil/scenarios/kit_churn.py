"""Keep playing while the kit is swapped underneath — the live-set test.

Port of ``platform-idf/tools/hil_kit_churn.py``. A pad stimulus runs from its
own thread and never stops for the kit change; the swap loop asks the loader's
own state (``KIT_STATUS``) whether a swap landed, because ``PAD_PRESS`` answers
``OK`` exactly like ``KIT_LOAD`` does and an ``OK`` off the wire proves nothing.

Two modes. Normal waits for every swap to land and measures how long it took
under load. ``rapid`` fires requests faster than a load can finish — the
coalescing path — and asserts the burst settles on the last kit the *device*
logged receiving, then that the kit still sounds.

The stimulus-in-window rule: a run where no pad hit landed inside any swap
window has not tested the thing this scenario exists for, however many rounds
it survived. That is reported as ``false_negative`` and is a FAIL.

The engine rule: once a kit has settled, the native sampler engine must hold
*that* kit (``SMPL_ENGINE`` ``kit=``). KIT_STATUS is the loader's account; the
engine takes the kit afterwards on its own task, and a take asked for while
another ran used to be refused and never retried -- the board then played the
old kit under the new one's name, which no other check here can hear.
"""
from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field
from typing import Any

from crosspad_hil import verbs
from crosspad_hil.cdc import CdcLink
from crosspad_hil.console import Console
from crosspad_hil.errors import TIMEOUT, HilError
from crosspad_hil.scenarios.base import Context, Progress, Report, register

# --- timing constants, every one the old script's literal ---------------------
APP_DESTROY_SETTLE_S = 1.0     # hil_kit_churn.main: time.sleep(1.0) after APP_DESTROY
APP_START_SETTLE_S = 2.5       # time.sleep(2.5) after APP_START Sampler
MIN_DWELL_S = 0.3              # time.sleep(max(0.3, args.dwell))
SLOW_FRACTION = 0.5            # elapsed > load_timeout * 0.5 → "slow"
MIN_HITS_IN_WINDOW = 2         # `if player and during < 2: starved.append(...)`
STATUS_TIMEOUT_S = 1.5         # cdc.request("KIT_STATUS", "KITSTATUS:", 1.5)
RAPID_DRAIN_S = 1.5            # run_rapid: time.sleep(1.5) before believing KIT_STATUS
RAPID_QUIET_POLL_S = 0.4       # run_rapid: time.sleep(0.4) between quiet polls
RAPID_QUIET_NEEDED = 3         # three quiescent samples in a row
RAPID_SOUND_SETTLE_S = 1.5     # "Let the new kit actually sound"
PEAK_WINDOW_S = 1.0            # run_rapid: SMPL_PEAK clear, sleep(1.0), SMPL_PEAK read
RAPID_STATUS_EVERY = 10        # `if rnd % 10 == 0:` progress poll during the burst
PAD_COUNT = 16
# A kit that loaded almost no samples cannot be judged for sound: peak=0 is
# then "nothing to play", not "firmware rendered nothing". A real 16-pad kit
# reports free~7; an empty/broken kit (e.g. a kit.json with all samples:[])
# reports free~23. Above this, a silent read is an empty kit, not a fault.
EMPTY_KIT_FREE_SLOTS = 20
SAMPLER_APP = "Sampler"
ENGINE_POLL_S = 0.25           # SMPL_ENGINE poll while the native take runs (state=1)
# A take the engine could not make because the kit has nothing to play: the
# same "empty kit" EMPTY_KIT_FREE_SLOTS describes, said by the engine.
ENGINE_EMPTY_WHY = ("no_sample_readable",)
SEED_LOG_WAIT_S = 2.0          # how long to wait for the seed's own console line
# markers.yaml kit_request, narrowed to one kit id.
KIT_REQ_LINE = r"hil_control: KIT_LOAD {kit} (queued|started)"


@dataclass
class Params:
    rounds: int = field(default=20, metadata={"help": "kit swaps to perform"})
    kits: list[int] | None = field(
        default=None, metadata={"help": "kit ids to cycle (comma-separated); default every kit"})
    dwell: float = field(default=2.0, metadata={"help": "seconds of playing between swaps"})
    load_timeout: float = field(
        default=15.0, metadata={"help": "seconds a single kit load may take"})
    hit_rate: float = field(default=8.0, metadata={"help": "pad hits per second"})
    pads: list[int] | None = field(
        default=None, metadata={"help": "pads to cycle (comma-separated); default 0-15"})
    rapid: float | None = field(
        default=None,
        metadata={"help": "seconds between kit requests fired without waiting for the "
                          "previous load (e.g. 0.4); unset = one-at-a-time mode"})
    no_play: bool = field(default=False, metadata={"help": "swap from silence (control case)"})
    silence_fails: bool = field(
        default=False, metadata={"help": "treat a kit that renders nothing as a failure"})


class Player(threading.Thread):
    """Fingerdrumming that never stops, least of all during the swap.

    From hil_kit_churn.Player: round-robins the pads so several voices stream
    off the card at once, varies velocity so the engine takes a different path
    per hit, and stamps every hit so the run can count how many landed inside
    each swap window instead of assuming they did. Writes are fire-and-forget
    through ``CdcLink.send`` — a blocking pad write would starve the stimulus
    exactly across the seconds that matter.
    """

    def __init__(self, link: CdcLink, hit_rate: float, pads: list[int]) -> None:
        super().__init__(name="kit_churn_player", daemon=True)
        self._link = link
        self._period = 1.0 / max(1.0, hit_rate)
        self._pads = list(pads)
        self.hits = 0
        self._stamps: list[float] = []
        self._lock = threading.Lock()
        self._halt = threading.Event()

    def run(self) -> None:
        i = 0
        while not self._halt.is_set():
            pad = self._pads[i % len(self._pads)]
            i += 1
            vel = 60 + (i * 17) % 67                       # hil_kit_churn.Player.run
            self._link.send(f"PAD_PRESS {pad} {vel}")
            with self._lock:
                self.hits += 1
                self._stamps.append(time.monotonic())
            if self._halt.wait(self._period * 0.5):
                self._link.send(f"PAD_RELEASE {pad}")   # never leave a pad held
                break
            self._link.send(f"PAD_RELEASE {pad}")
            if self._halt.wait(self._period * 0.5):
                break

    def hits_between(self, t0: float, t1: float) -> int:
        with self._lock:
            return sum(1 for t in self._stamps if t0 <= t <= t1)

    def stop(self) -> None:
        self._halt.set()


def _faults(con: Console, since_seq: int) -> tuple[list[str], int]:
    """(new fatal lines, reboot count) from the lines after ``since_seq``."""
    fatal: list[str] = []
    reboots = 0
    for ev in con.events(since_seq=since_seq):
        if ev.kind == "fatal":
            fatal.append(ev.line)
        elif ev.kind == "reboot":
            reboots += 1
    return fatal, reboots


def _kit_requests(con: Console, since_seq: int) -> list[tuple[int, str]]:
    """(kit id, queued|started) the firmware logged receiving, in arrival order.

    ``started`` means the request met an idle loader, ``queued`` that it arrived
    on top of a load already running — which is the whole point of rapid mode,
    and the only way the host can tell the two apart.
    """
    return [(int(ev.data["kit"]), str(ev.data["state"]))
            for ev in con.events(since_seq=since_seq) if ev.kind == "kit"]


def _cdc_drops(con: Console) -> int:
    return int(con.snapshot().get("cdc_drops", 0) or 0)


def _heap_internal(con: Console) -> int | None:
    heap = con.snapshot().get("heap") or {}
    value = heap.get(0)
    return int(value) if value is not None else None


def _engine_kit(ctx: Context, link: CdcLink, settled: int,
                timeout_s: float) -> tuple[str | None, dict[str, Any]]:
    """(why not, what SMPL_ENGINE said) -- why not is None when the native
    engine holds ``settled``.

    Waits out a take still running (state=1) for ``timeout_s``. Also fine: a
    firmware whose SMPLENG has no ``kit=`` (not judged), a pitched bank holding
    the kit (the native engine rightly holds none) and a kit the engine could
    not take because it has nothing to play. Anything else is a failure: the
    engine holding another kit, or none, after the board settled.
    """
    deadline = time.monotonic() + max(0.0, timeout_s)
    eng = verbs.smpl_engine(link)
    while eng.get("state") == 1 and time.monotonic() < deadline:
        _wait(ctx, ENGINE_POLL_S)
        eng = verbs.smpl_engine(link)
    info: dict[str, Any] = {"engine": eng.get("engine"), "state": eng.get("state"),
                            "why": eng.get("why"), "kit": eng.get("kit")}
    if "kit" not in eng:
        info["unsupported"] = True
        return None, info
    if eng.get("state") == 1:
        return (f"the native take of kit {settled} never finished ({timeout_s:.1f}s; "
                f"engine still holds {eng.get('kit')})"), info
    kit = eng.get("kit")
    if kit == settled:
        return None, info
    if kit != -1:
        return f"the native engine holds kit {kit} but the board settled on {settled}", info
    try:
        pitched = bool(int(verbs.pitched_status(link).get("active", 0)))
    except (HilError, ValueError, TypeError):
        pitched = False
    if pitched:
        info["pitched"] = True
        return None, info
    if eng.get("why") in ENGINE_EMPTY_WHY:
        info["empty"] = True
        return None, info
    return (f"the native engine holds no kit after the board settled on {settled} "
            f"(state={eng.get('state')} why={eng.get('why')})"), info


def _env(summary: str, data: dict[str, Any]) -> Report:
    return Report(passed=False, summary=summary, data=data, artifacts=[], exit_code=2)


def _wait(ctx: Context, seconds: float) -> None:
    """Sleep that returns early on cancel and then raises CANCELLED."""
    if seconds > 0:
        ctx.cancelled.wait(seconds)
    ctx.check_cancelled()


class KitChurnScenario:
    name = "kit_churn"
    Params = Params
    description = "swap kits while the pads keep firing; --rapid spins the selector"

    def run(self, ctx: Context, params: Params, progress: Progress) -> Report:
        try:
            p = params
            pads = list(p.pads) if p.pads else list(range(PAD_COUNT))
            data: dict[str, Any] = {
                "mode": "rapid" if p.rapid else "normal",
                "kits": [], "cycle": [], "pads": pads,
                "rounds_requested": p.rounds, "rounds": [],
                "hits_sent": 0, "hits_per_s": 0.0, "hits_in_windows": 0,
                "starved": [], "silent": [], "slow": [], "failures": [],
                "fatal": [], "reboots": 0, "cdc_drops": 0, "pad_stats": None,
                "false_negative": False, "silence_fails": p.silence_fails, "rapid": None,
                "found": None, "left_as_found": None,
            }
            ctx.check_cancelled()
            con = ctx.open_console(reset=False)
            link = ctx.open_cdc()
            home: dict[str, Any] = {}
            try:
                return self._run(ctx, p, pads, data, con, link, home, progress)
            finally:
                self._leave_as_found(ctx, p, link, home, data)
        finally:
            ctx.close_all()

    def _leave_as_found(self, ctx: Context, p: Params, link: CdcLink, home: dict[str, Any],
                        data: dict[str, Any]) -> None:
        """Put back the kit and the app the board had before the run; never raises.

        The next gate step (usb_mode_cycle measures heap drift) otherwise starts
        with the Sampler up and whatever kit the burst settled on.
        """
        if not home:
            return
        try:
            kit = home.get("kit", -1)
            if kit >= 0 and verbs.kit_status(link)["current"] != kit:
                verbs.kit_load(link, kit, wait_s=p.load_timeout)
            app = home.get("app")
            running = verbs.app_list(link).get("running")
            if app is None and running is not None:
                verbs.app_stop(link)
            elif app is not None and running != app:
                verbs.app_start(link, app)
            data["left_as_found"] = True
        except HilError as e:
            data["left_as_found"] = False
            ctx.log(f"could not put the board back as found ({home}): {e.code}: {e.message}")

    def _run(self, ctx: Context, p: Params, pads: list[int], data: dict[str, Any],
             con: Console, link: CdcLink, home: dict[str, Any], progress: Progress) -> Report:

        # --- what is on the card ----------------------------------------------
        try:
            kl = verbs.kit_list(link)
        except HilError as e:
            return _env(f"kit_churn: KIT_LIST failed ({e.code}: {e.message}) — "
                        "no kit manager, or no SD card", data)
        kits = [{"id": int(k["id"]), "name": str(k["name"])} for k in kl.get("kits", [])]
        if not kits:
            return _env("kit_churn: KIT_LIST returned nothing — no kit manager, or no SD card",
                        data)
        names = {k["id"]: k["name"] for k in kits}
        data["kits"] = kits
        home["kit"] = int(verbs.kit_status(link)["current"])
        if p.kits:
            unknown = [k for k in p.kits if k not in names]
            if unknown:
                return _env(f"kit_churn: no such kit id: {unknown}", data)
            cycle = list(p.kits)
        else:
            cycle = [k["id"] for k in kits]
        data["cycle"] = cycle
        ctx.log(f"{len(kits)} kits on the card; cycling {len(cycle)}: "
                + ", ".join(names[k] for k in cycle))

        # --- seed the kit BEFORE starting the Sampler -------------------------
        # A board with no current kit puts the kit-selector gate in front of
        # the Sampler, and APP_LIST then reports running=- because the
        # overlay is not the orchestrator's running app — so app_start times
        # out and the scenario could never pass from a cold boot. Loading a
        # kit first removes the gate instead of trying to press through it.
        # It also seeds from silence, so round 1 is a swap like the rest.
        ctx.log(f"seeding with {names[cycle[0]]} ...")
        seed_mark = con.seq + 1
        try:
            verbs.kit_load(link, cycle[0], wait_s=p.load_timeout)
        except HilError as e:
            if e.code != TIMEOUT:
                raise
            return _env("kit_churn: seed kit never finished loading — "
                        "SD card or sampler state is bad", data)

        # --- the Sampler has to own pad logic for a hit to reach the engine ---
        apps = verbs.app_list(link)
        home["app"] = apps.get("running")
        data["found"] = dict(home)
        if SAMPLER_APP not in apps.get("apps", []):
            return _env("kit_churn: Sampler is not in the launcher — nothing to churn", data)
        if apps.get("running") != SAMPLER_APP:
            try:
                verbs.app_destroy(link)
            except HilError as e:
                ctx.log(f"APP_DESTROY: {e.code}: {e.message} (nothing running?)")
            _wait(ctx, APP_DESTROY_SETTLE_S)
            verbs.app_start(link, SAMPLER_APP)
            _wait(ctx, APP_START_SETTLE_S)

        player: Player | None = None
        if not p.no_play:
            player = Player(link, p.hit_rate, pads)
            player.start()
            ctx.log(f"playing {p.hit_rate:.0f} hits/s across pads {pads[0]}..{pads[-1]} "
                    "— this does not stop for the swap")

        # Console.events() is exclusive (ev.seq > since_seq), so the boundary
        # has to be the last line that is NOT ours — the seed's own
        # "KIT_LOAD n started". Waiting for that line is also the only way to
        # know the reader thread has caught up with the seed: a boundary taken
        # before it has is one line short, the run's first swap falls off the
        # front of the log, and every rapid-mode check then judges the wrong
        # request. A firmware that does not log the line at all keeps the old
        # behaviour, minus at most that one line.
        seed_line = con.expect([KIT_REQ_LINE.format(kit=cycle[0])],
                               timeout_s=SEED_LOG_WAIT_S, since_seq=seed_mark)
        seq0 = seed_line.seq if seed_line.seq is not None else con.seq
        drops0 = _cdc_drops(con)
        t_play0 = time.monotonic()
        try:
            if p.rapid:
                self._run_rapid(ctx, p, link, con, player, cycle, names, seq0, drops0, data,
                                progress)
            else:
                self._run_normal(ctx, p, link, con, player, cycle, names, seq0, data, progress)
        finally:
            if player is not None:
                player.stop()
                player.join(timeout=2.0)
                data["hits_sent"] = player.hits
                span = max(0.001, time.monotonic() - t_play0)
                data["hits_per_s"] = round(player.hits / span, 2)

        fatal, reboots = _faults(con, seq0)
        data["fatal"] = fatal
        data["reboots"] = reboots
        data["cdc_drops"] = max(0, _cdc_drops(con) - drops0)
        try:
            data["pad_stats"] = verbs.pad_stats(link)
        except HilError as e:
            ctx.log(f"PAD_STATS: {e.code}: {e.message}")

        return self._verdict(p, player is not None, data)

    # --- normal mode ---------------------------------------------------------
    def _run_normal(self, ctx: Context, p: Params, link: CdcLink, con: Console,
                    player: Player | None, cycle: list[int], names: dict[int, str],
                    seq0: int, data: dict[str, Any], progress: Progress) -> None:
        for rnd in range(1, p.rounds + 1):
            ctx.check_cancelled()
            target = cycle[rnd % len(cycle)]
            t0 = time.monotonic()
            landed = True
            load_err: HilError | None = None
            try:
                # KIT_LOAD + KIT_STATUS polling: the id flips when changeKit()
                # succeeds, loading=0 once the pads are reassigned, pending=-1
                # when nothing is queued behind it. All three, or it did not land.
                verbs.kit_load(link, target, wait_s=p.load_timeout)
            except HilError as e:
                landed = False
                load_err = e
            t1 = time.monotonic()
            elapsed = t1 - t0
            during = player.hits_between(t0, t1) if player is not None else 0

            record: dict[str, Any] = {
                "round": rnd, "kit": target, "name": names[target],
                "seconds": round(elapsed, 3), "landed": landed,
                "hits_in_window": during, "peak": None, "free": None,
                "silent": False, "heap": _heap_internal(con), "engine_kit": None,
            }
            data["rounds"].append(record)

            fatal, reboots = _faults(con, seq0)
            if fatal or reboots:
                lines = fatal[:6] or ["(reset with no panic line)"]
                data["failures"].append(
                    f"round {rnd}: CRASH swapping to {names[target]} after {elapsed:.1f}s\n    "
                    + "\n    ".join(lines))
                break
            if not landed:
                detail = f" ({load_err.code}: {load_err.message})" if load_err else ""
                data["failures"].append(
                    f"round {rnd}: kit {names[target]} never became current "
                    f"({p.load_timeout:.0f}s) — load stuck or dropped{detail}")
                break
            if elapsed > p.load_timeout * SLOW_FRACTION:
                data["slow"].append({"round": rnd, "kit": target, "seconds": round(elapsed, 2)})

            # SMPL_PEAK is read-and-clear: clear, let the new kit play, read.
            verbs.smpl_peak(link)
            _wait(ctx, max(MIN_DWELL_S, p.dwell))
            pk = verbs.smpl_peak(link)
            record["peak"] = pk.get("peak")
            record["free"] = pk.get("free")
            _free = pk.get("free")
            _empty = _free is not None and _free >= EMPTY_KIT_FREE_SLOTS
            if player is not None and pk.get("peak") == 0 and not _empty:
                record["silent"] = True
                data["silent"].append({"round": rnd, "kit": target})
            elif _empty and pk.get("peak") == 0:
                record["empty_kit"] = True
                data.setdefault("empty_kits", []).append({"round": rnd, "kit": target})
            if player is not None and during < MIN_HITS_IN_WINDOW:
                data["starved"].append({"round": rnd, "kit": target, "hits": during})

            engine_err, record["engine_kit"] = _engine_kit(ctx, link, target, p.load_timeout)
            if engine_err:
                data["failures"].append(f"round {rnd}: {engine_err}")

            ctx.log(f"  [{rnd:>3}/{p.rounds}] {names[target]:<16} {elapsed:5.1f}s  "
                    f"hits={during:<3} peak={record['peak']} free={record['free']}"
                    f"{'  <- SILENT' if record['silent'] else ''}  heap={record['heap']}")
            progress(rnd, p.rounds,
                     f"round {rnd}/{p.rounds} kit={target} hits_in_window={during}")

    # --- rapid mode ----------------------------------------------------------
    def _run_rapid(self, ctx: Context, p: Params, link: CdcLink, con: Console,
                   player: Player | None, cycle: list[int], names: dict[int, str],
                   seq0: int, drops0: int, data: dict[str, Any], progress: Progress) -> None:
        interval = float(p.rapid or 0.0)
        rapid: dict[str, Any] = {
            "interval_s": interval, "requested": [], "device_logged": [],
            "last_logged": None, "last_state": None, "loader_seen_busy": False,
            "settled": None, "peak": None, "free": None, "drops_this_run": 0,
            "engine_kit": None,
        }
        data["rapid"] = rapid
        ctx.log(f"rapid mode: a kit request every {interval:.2f}s for {p.rounds} requests, "
                "without waiting for any of them")

        for rnd in range(1, p.rounds + 1):
            ctx.check_cancelled()
            target = cycle[rnd % len(cycle)]
            # Fire and forget: the point is to meet a busy loader, and an OK
            # here would not be this command's anyway.
            link.send(f"KIT_LOAD {target}")
            rapid["requested"].append(target)
            fatal, reboots = _faults(con, seq0)
            if fatal or reboots:
                data["failures"].append(
                    "crash during the burst:\n    "
                    + "\n    ".join(fatal[:6] or ["(reset with no panic line)"]))
                return
            if rnd % RAPID_STATUS_EVERY == 0:
                st = verbs.kit_status(link)
                ctx.log(f"  [{rnd:>3}/{p.rounds}] asked {names[target]:<16} "
                        f"current={st['current']} loading={st['loading']} "
                        f"pending={st['pending']}")
            progress(rnd, p.rounds, f"request {rnd}/{p.rounds} kit={target}")
            _wait(ctx, interval)

        ctx.log(f"burst done; host sent {len(rapid['requested'])}. "
                "waiting for the loader to drain ...")
        # A request travels CDC -> lv_async_call -> the LVGL thread; give that
        # queue time to drain before believing KIT_STATUS, or the first poll
        # catches the gap between one load finishing and the next being picked
        # up — which reads as "settled", one or two kits early.
        _wait(ctx, RAPID_DRAIN_S)

        # Require the quiet to hold: RAPID_QUIET_NEEDED quiescent samples in a row.
        # Whether the loader was ever seen with work left is recorded too: a
        # loader that is quiescent from the very first poll after a burst it
        # could not have finished did not coalesce the queue, it dropped it.
        deadline = time.monotonic() + p.load_timeout * 2
        settled: dict[str, Any] | None = None
        quiet = 0
        while time.monotonic() < deadline:
            settled = verbs.kit_status(link)
            busy = settled["loading"] is not False or settled["pending"] != -1
            if busy:
                rapid["loader_seen_busy"] = True
            quiet = 0 if busy else quiet + 1
            if quiet >= RAPID_QUIET_NEEDED:
                break
            _wait(ctx, RAPID_QUIET_POLL_S)
        rapid["settled"] = settled

        # Let the new kit actually sound before asking whether it does.
        _wait(ctx, RAPID_SOUND_SETTLE_S)
        verbs.smpl_peak(link)
        _wait(ctx, PEAK_WINDOW_S)
        pk = verbs.smpl_peak(link)
        rapid["peak"] = pk.get("peak")
        rapid["free"] = pk.get("free")
        engine_err: str | None = None
        if settled is not None:
            engine_err, rapid["engine_kit"] = _engine_kit(ctx, link, settled["current"],
                                                          p.load_timeout)

        # Now the console has caught up: judge the device on the requests it
        # logged receiving, not on the ones the host sent.
        requests = _kit_requests(con, seq0)
        logged = [kit for kit, _state in requests]
        rapid["device_logged"] = logged
        last = logged[-1] if logged else (rapid["requested"][-1] if rapid["requested"] else None)
        host_last = rapid["requested"][-1] if rapid["requested"] else None
        # Lines the console lost under the pad storm; the settle rule below does
        # not hold that against the loader when it ended on the right kit.
        rapid["console_lost_requests"] = max(0, len(rapid["requested"]) - len(logged))
        rapid["last_logged"] = last
        rapid["last_state"] = requests[-1][1] if requests else None
        rapid["drops_this_run"] = max(0, _cdc_drops(con) - drops0)
        ctx.log(f"requests: host sent {len(rapid['requested'])}, device logged {len(logged)}; "
                f"last it saw was {names.get(last)} ({last})")
        ctx.log(f"CDC commands dropped in transit: {rapid['drops_this_run']} this run")

        fatal, reboots = _faults(con, seq0)
        cur = settled["current"] if settled else None
        if engine_err and not (fatal or reboots):
            data["failures"].append(engine_err)
        if fatal or reboots:
            data["failures"].append("crash during or after the burst")
        elif settled is None or settled["loading"] is not False or settled["pending"] != -1:
            data["failures"].append("the loader never drained; a request is stuck")
        elif cur != last and cur != host_last:
            # Coalescing keeps the newest request, so the burst must settle on
            # the last one the device received.
            data["failures"].append(
                f"settled on {names.get(cur)} ({cur}) but {names.get(last)} ({last}) "
                "was the last request the device logged")
        elif cur != last:
            # Settled on the last request the *host* sent, while the console's
            # tally stops earlier: under 8 pad hits/s plus a kit load the UART
            # drops lines, and a missing "KIT_LOAD n queued" line is a console
            # artefact, not a swap that went nowhere. The device did the right
            # thing; say what the console lost rather than fail on it.
            ctx.log(f"settled on the last request sent ({names.get(cur)}); the console "
                    f"logged {len(logged)} of {len(rapid['requested'])} requests, "
                    f"last {names.get(last)} ({last}) -- lines lost on the console, "
                    "not requests lost by the loader")
        elif (rapid["last_state"] == "queued" and not rapid["loader_seen_busy"]
              and len(logged) > 1 and logged.count(last) > 1):
            # The last request the device logged arrived on top of a running
            # load, so the loader still owed one swap when the burst ended — yet
            # it reported nothing loading and nothing pending on the very first
            # poll. That is the dropped-pending bug wearing a passing disguise:
            # the burst wrapped back onto the kit that was already loading, so
            # `cur == last` holds while 2, 3, 0 went nowhere. Checked only for a
            # trailing `queued`, so a `--rapid` interval long enough for every
            # request to land on an idle loader is not accused of it. Only a
            # burst that wraps back onto an earlier kit can hide a dropped
            # pending behind `cur == last`; a burst of distinct kits that ends
            # on the last one simply finished (a fast loader, e.g. the native
            # engine, is idle by the first poll).
            data["failures"].append(
                f"the loader was idle at the first poll after {len(logged)} requests, "
                f"the last of them queued behind a running load: the queue behind "
                f"{names.get(cur)} ({cur}) went nowhere — device logged {logged}")
        elif (player is not None and rapid["peak"] == 0
              and (rapid["free"] is None or rapid["free"] < EMPTY_KIT_FREE_SLOTS)):
            data["failures"].append("silent after the burst: pads accepted, nothing rendered")
        elif player is not None and rapid["peak"] == 0:
            # The kit the burst settled on loaded ~no samples (free at the empty
            # mark): it cannot make sound, so silence is the kit, not the loader.
            data.setdefault("empty_kits", []).append(
                {"round": "rapid", "kit": rapid.get("last_logged")})
            ctx.log(f"final kit loaded no samples (free={rapid['free']}); "
                    "silence is the kit, not the firmware — not a failure")

    # --- verdict -------------------------------------------------------------
    def _verdict(self, p: Params, stimulus: bool, data: dict[str, Any]) -> Report:
        rounds: list[dict[str, Any]] = data["rounds"]
        data["hits_in_windows"] = sum(int(r["hits_in_window"]) for r in rounds)
        if data["mode"] == "normal":
            data["false_negative"] = bool(
                stimulus and rounds
                and all(int(r["hits_in_window"]) < MIN_HITS_IN_WINDOW for r in rounds))
        failures: list[str] = data["failures"]
        problems: list[str] = list(failures)
        if data["false_negative"]:
            problems.append(
                f"false negative: {len(rounds)} rounds and not one swap window saw "
                f">= {MIN_HITS_IN_WINDOW} pad hits — the swap was never tested under play")
        if data["silent"] and p.silence_fails:
            problems.append("kits that rendered nothing: "
                            + ", ".join(f"{s['kit']} (round {s['round']})" for s in data["silent"]))
        if problems:
            return Report(passed=False, summary="kit_churn: FAIL — " + "; ".join(problems),
                          data=data, artifacts=[], exit_code=1)
        note = ""
        if data["silent"]:
            note = f" (WARN: {len(data['silent'])} silent round(s))"
        if data["mode"] == "rapid":
            summary = "kit_churn: PASS — the burst coalesced onto the last kit and it plays" + note
        else:
            summary = (f"kit_churn: PASS — {len(rounds)} swaps under continuous play, "
                       f"{data['hits_in_windows']} hits inside swap windows, no crash" + note)
        return Report(passed=True, summary=summary, data=data, artifacts=[], exit_code=0)


SCENARIO = register(KitChurnScenario())


def main(argv: list[str] | None = None) -> int:
    """Entry point for the platform-idf shim `tools/hil_kit_churn.py`."""
    import sys

    from crosspad_hil.cli import main as cli_main

    args = list(sys.argv[1:] if argv is None else argv)
    return cli_main(["run", "kit_churn", *args])
