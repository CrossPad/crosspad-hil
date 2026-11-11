"""Shared setup for the two "record the engine through UAC2" scenarios.

`sampler_record` and `pitched_kit` both do the same four things over CDC, in
the same order, for the same reason (see `sampler_record`'s module docstring):
name a kit and load it if it is not current, make sure the Sampler app is the
one running, open/close the capture session around the profile switch, and
judge whether a finished capture is even worth scoring. Everything here is
CDC- and capture-shaped, not scenario-shaped — the two callers keep their own
`Params`, schedules and verdict rules.

`ensure_kits()` is the fifth thing, added for `kit_swap_audio` and available to
the other two: a fixture kit the card does not have is a bench problem the host
can *fix* — build it, push it over `FS_PUT`, reboot so the kit scan sees it —
rather than a run that dies several seconds in with "no kit named". It lives
here because nothing about it is scenario-shaped either.
"""
from __future__ import annotations

from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

from crosspad_hil import fs, pins, verbs
from crosspad_hil.cdc import CdcLink
from crosspad_hil.console import Console
from crosspad_hil.devices import UsbMode
from crosspad_hil.errors import ENV, NOT_SUPPORTED, TIMEOUT, UNKNOWN_VERB, HilError
from crosspad_hil.kits import KITS_ROOT
from crosspad_hil.kits import mkdir_tolerant as _mkdir
from crosspad_hil.kits import push_kit_dir as _push_kit
from crosspad_hil.kits import reboot_for_kit_scan as _reboot_for_the_kit_scan
from crosspad_hil.kits import wait_for_kit_scan as _wait_for_the_kit_scan
from crosspad_hil.scenarios.base import Artifact, Context, Report

KIT_LOAD_TIMEOUT_S = 20.0
APP_START_SETTLE_S = 2.0


def resolve_kit(link: CdcLink, *, kit_id: int | None = None, kit_name: str | None = None,
                missing_hint_suffix: str = "") -> tuple[int, str, int]:
    """(kit_id, name, current) from `KIT_LIST`; `kit_name` (case-insensitive) wins.

    With neither `kit_id` nor `kit_name`, falls back to whatever `KIT_LIST`
    reports as current, or the lowest id on a card it has never loaded from.
    `missing_hint_suffix` is appended to a "no kit named" hint — a caller with
    one fixture kit in mind can point at how to provision it.
    """
    listing = verbs.kit_list(link)
    kits = {int(k["id"]): str(k["name"]) for k in listing.get("kits", [])}
    current = int(listing.get("current", -1))
    if kit_name:
        match = [k for k, n in kits.items() if n.lower() == kit_name.lower()]
        if not match:
            raise HilError(ENV, f"no kit named {kit_name!r}",
                           hint="have: " + (", ".join(sorted(kits.values())) or "none")
                                + missing_hint_suffix,
                           kits=sorted(kits.values()))
        kit_id = match[0]
    if kit_id is None:
        kit_id = current if current >= 0 else (min(kits) if kits else 0)
    return kit_id, kits.get(kit_id, "?"), current


def load_kit(ctx: Context, link: CdcLink, kit_id: int, name: str, current: int,
            wait_s: float = KIT_LOAD_TIMEOUT_S) -> None:
    """`KIT_LOAD` only when `kit_id` is not already current — a reload is not free."""
    if kit_id != current:
        ctx.log(f"loading kit {kit_id} {name!r}")
        verbs.kit_load(link, kit_id, wait_s=wait_s)
    ctx.log(f"kit {kit_id} {name!r} loaded")


def ensure_app_running(ctx: Context, link: CdcLink, app_name: str, *,
                       not_installed_reason: str,
                       settle_s: float = APP_START_SETTLE_S) -> None:
    """Start `app_name` if it is not already the running app; ENV if it is not installed.

    The settle wait is the app's first LVGL frames coming up, not the launch
    itself — `app_start` already polls `APP_LIST` and raises TIMEOUT if the
    launch never lands.
    """
    apps = verbs.app_list(link)
    if app_name not in apps.get("apps", []):
        raise HilError(ENV, not_installed_reason,
                       hint="install the sampler app and reflash", apps=apps.get("apps", []))
    if apps.get("running") != app_name:
        ctx.log(f"starting {app_name}")
        verbs.app_start(link, app_name)
        ctx.cancelled.wait(settle_s)
        ctx.check_cancelled()


def start_capture(ctx: Context, open_capture_fn: Any, wav: Path, *, preset: str,
                  volume: int | None, on_open: Callable[[], None] | None = None,
                  **extra: Any) -> tuple[Any, dict, Artifact]:
    """Open the capture session, start it, and hand back the artifact for the take.

    `on_open` runs after the (possibly slow) capture device is opened but before
    the (also slow) profile switch in `session.start()` — the caller's hook for
    a progress message that should land between the two, not after both.

    `ctx.device` is updated to the profile the session actually switched to
    (the USB-audio profile on success) before the caller does anything else
    with it.
    """
    session = open_capture_fn(ctx.device, seconds=None, preset=preset, out=wav,
                              volume=volume, logdir=ctx.workdir, log=ctx.log, **extra)
    if on_open is not None:
        on_open()
    info = session.start()
    ctx.device = session.device
    return session, info, Artifact(path=str(wav), mime="audio/wav", role="capture")


def stop_capture(ctx: Context, session: Any, data: dict[str, Any]) -> Any:
    """Finalize the WAV, restore the USB profile (the session's own `finally`), and
    record the result and its health in `data`. Runs even when scoring above it raises."""
    result = session.stop()
    ctx.device = session.device
    data["capture"] = result.to_dict()
    data["capture_health"] = result.health
    return result


def capture_problems(capture: dict[str, Any], health: dict[str, Any]) -> list[str]:
    """Reasons a finished capture is not worth scoring: silence, or a stalled stream.

    Reported, never raised — callers decide the severity: `sampler_record` folds
    it into its usual FAIL verdict (partial results already exist), `pitched_kit`
    treats it as a bench problem and scores nothing (see its module docstring).
    """
    problems: list[str] = []
    if capture.get("silent"):
        problems.append(capture.get("hint") or "the capture is silent")
    if health and not health.get("ok", True):
        problems.append(f"capture unhealthy: {health.get('detail')}")
    return problems


# --------------------------------------------------------------------------- host losses
#
# Whatever the host failed to record is not the device's: a take with holes the
# capture made itself is not scored at all ("capture not measured", exit 2),
# rather than read as clicks, dropouts or silent pads.

HOST_LOSS_S = 0.5
HOST_GAP_MS = 500.0


def host_capture_verdict(health: dict[str, Any]) -> tuple[float, str | None]:
    """``(seconds the host lost, why the take cannot be trusted | None)``.

    A capture that never received part of the stream (``missed_s``), stalled
    (callback gap) or overran has holes the device did not make; scoring them
    would charge the host's loss to the firmware.
    """
    missed = float(health.get("missed_s") or 0.0)
    gap = float(health.get("max_gap_ms") or 0.0)
    overruns = int(health.get("overruns") or 0)
    why = []
    if missed > HOST_LOSS_S:
        why.append(f"host lost {missed:.1f} s of the capture")
    if gap > HOST_GAP_MS:
        why.append(f"a {gap:.0f} ms gap in the host's capture callbacks")
    if overruns:
        why.append(f"{overruns} host overrun(s)")
    if not why:
        return missed, None
    if health.get("exclusive") is False:
        why.append("recorded through PipeWire, the card was busy")
    return missed, "; ".join(why)


def host_loss(data: dict[str, Any], health: dict[str, Any] | None) -> str | None:
    """Note what the host lost of a take in ``data``; why it cannot be scored, or None."""
    lost, why = host_capture_verdict(health or {})
    data["host_lost_s"] = round(lost, 3)
    data["capture_untrusted"] = why
    return why


def not_measured(name: str, why: str, data: dict[str, Any],
                 artifacts: list[Artifact]) -> Report:
    data.setdefault("failures", []).append(f"capture not measured: {why}")
    return Report(passed=False, summary=f"{name}: capture not measured — {why}",
                  data=data, artifacts=artifacts, exit_code=2)


# --------------------------------------------------------------------------- provisioning
#
# mkdir/push/reboot/scan-wait now live in crosspad_hil.kits (imported above as
# _mkdir / _push_kit / _reboot_for_the_kit_scan / _wait_for_the_kit_scan) so
# `crosspad-hil kit push` can reuse them without importing a scenario module.


def _missing_kits(link: CdcLink, names: Sequence[str]) -> tuple[list[str], list[str]]:
    """(names not on the card, every name the card has) — case-insensitive."""
    have = [str(k["name"]) for k in verbs.kit_list(link).get("kits", [])]
    lowered = {n.lower() for n in have}
    return [n for n in names if n.lower() not in lowered], have


def ensure_kits(ctx: Context, link: CdcLink, names: Sequence[str],
                builder: Callable[[Path], dict[str, Path]], *,
                console: Console | None = None,
                provision: bool = True, kits_root: str = KITS_ROOT,
                push: Callable[..., dict] = fs.push,
                missing_hint_suffix: str = "") -> tuple[CdcLink, list[str]]:
    """Make sure every kit in `names` is on the card, building and pushing the missing ones.

    Returns `(link, provisioned)` — the link because **the one handed in is
    closed and replaced**: pushing needs the raw port and the reboot afterwards
    re-enumerates the CDC node, possibly onto a different path. Callers must use
    the returned link from here on. `provisioned` is the list of kit names
    actually pushed, empty when the card already had them.

    `builder(root) -> {kit name: directory}` writes the fixtures locally; the
    directory basename is what the card gets. With `provision=False` a missing
    kit is the old ENV failure, unchanged.

    Everything that can go wrong here is a bench problem, not a firmware one:
    a refused push, a board that does not come back, a scan that never ends —
    all ENV (exit 2), because none of them is evidence about the thing the
    scenario measures.
    """
    # A run started seconds after boot meets the card scan, which refuses every
    # reader rather than serving a short list — asking once would read "no kit
    # named" off a listing that is not final yet and provision kits the card has.
    _wait_for_the_kit_scan(ctx, link)
    missing, have = _missing_kits(link, names)
    if not missing:
        return link, []
    if not provision:
        raise HilError(ENV, "no kit named " + ", ".join(repr(n) for n in missing),
                       hint="have: " + (", ".join(sorted(have)) or "none")
                            + missing_hint_suffix + " (or drop --no-provision)",
                       kits=sorted(have))

    ctx.log(f"provisioning {len(missing)} missing kit(s): " + ", ".join(missing))
    root = ctx.workdir / "fixtures"
    built = builder(root)
    unknown = [n for n in missing if n not in built]
    if unknown:
        raise HilError(ENV, "the fixture builder does not know " + ", ".join(unknown),
                       hint="it built: " + ", ".join(sorted(built)) + missing_hint_suffix)

    for name in missing:
        remote_dir = f"{kits_root}/{built[name].name}"
        _mkdir(link, remote_dir)
        _mkdir(link, f"{remote_dir}/SAMPLES")
    # The console is the reset line and the only witness to the boot that
    # follows. A caller that already has one must hand it over: opening a second
    # Console on the same port takes a PortLock the first one is holding.
    if console is None:
        console = ctx.open_console(reset=False)
    link.close()                       # fs.push needs the raw port, unframed

    pushed = 0
    try:
        for name in missing:
            remote_dir = f"{kits_root}/{built[name].name}"
            ctx.log(f"  pushing {name!r} -> {remote_dir}")
            pushed += _push_kit(ctx, built[name], remote_dir, push)
    except HilError as exc:
        raise HilError(ENV, f"pushing the fixture kits failed: {exc.message}",
                       hint=exc.hint or "the STM VCP console carries the FX log lines; "
                                        "BLE_STOP first if the fx_worker cannot start",
                       **exc.details) from exc
    ctx.log(f"  {pushed} bytes pushed; rebooting so the kit scan picks them up")

    _reboot_for_the_kit_scan(ctx, console)
    link = ctx.open_cdc()
    # `Context.open_cdc` only probes on its first real link, and this one is not
    # the first — but the board did just reboot, so the probe is the point.
    link.wait_ready()
    _wait_for_the_kit_scan(ctx, link)

    still, have = _missing_kits(link, names)
    if still:
        raise HilError(ENV, "the pushed kits are still not in KIT_LIST: "
                            + ", ".join(still),
                       hint="have: " + (", ".join(sorted(have)) or "none")
                            + "; the card may be full, or the kit scan rejected the JSON",
                       kits=sorted(have))
    ctx.log("provisioned: " + ", ".join(missing))
    return link, list(missing)


# ---------------------------------------------------------------- the jack's preamp (IN2)
# The bench loop is DAC -> jack -> codec 1's ADC, whose preamp is INGAIN's IN2.
# platform-idf dcfac16 brings it up at 0 dB, where the loop returned -45.7 dBFS;
# at +24 dB (what it effectively had before) -20.6. A scenario that judges a
# level through the jack pins IN2 for the run and puts the old value back: the
# board keeps it in NVS. Level floors were set for +24 dB and move with it.

JACK_CODEC = 1
IN2_INDEX = 1
IN2_DEFAULT_DB = 24
LEVEL_FLOOR_GAIN_DB = 24


def level_floor(floor_at_24db: float, gain_db: int | None) -> float:
    """A level floor set for IN2 at +24 dB, moved to ``gain_db`` (unchanged for None)."""
    return floor_at_24db if gain_db is None else floor_at_24db - (LEVEL_FLOOR_GAIN_DB - gain_db)


def set_in2(ctx: Context, link: CdcLink, gain_db: int, data: dict[str, Any]) -> int | None:
    """Pin IN2 at ``gain_db`` over ``link`` (CDC: INGAIN is not on the companion list).

    Returns the value to put back, or None when the firmware has no INGAIN (noted
    in ``data["ingain_note"]``). Records ``ingain_before`` / ``ingain_during``;
    ENV when the board will not take the value.
    """
    try:
        # A run killed with IN2 pinned left a promise in the journal: keep it
        # first, so "before" is the board's own value and not the last pin.
        data["ingain_recovered"] = pins.restore_pending(link, ctx.device.id, ctx.log,
                                                        keys=(pins.IN2,)) or None
        before = verbs.ingain(link)
    except HilError as e:
        if e.code != NOT_SUPPORTED:
            raise
        data["ingain_note"] = f"{e.message}: the jack's gain is the board's own"
        ctx.log(f"INGAIN: {e.message}; measuring at whatever gain the board has")
        return None
    data["ingain_before"] = before
    # Written before the board is touched: a kill from here on leaves the journal
    # holding the value to put back.
    pins.record(ctx.device.id, pins.IN2, before.get("IN2"))
    during = verbs.ingain(link, IN2_INDEX, gain_db)
    data["ingain_during"] = during
    if during.get("IN2") != gain_db:
        # It was written all the same: put the old value back before giving up,
        # since nobody restores a pin that was never handed to the caller.
        try:
            data["ingain_after"] = verbs.ingain(link, IN2_INDEX, int(before.get("IN2", 0)))
            data["ingain_restored"] = data["ingain_after"].get("IN2") == before.get("IN2")
            if data["ingain_restored"]:
                pins.clear(ctx.device.id, pins.IN2)
        except HilError as e:
            data["ingain_restored"] = False
            ctx.log(f"IN2 gain NOT restored after the refused pin: {e.code}: {e.message}")
        raise HilError(ENV, f"IN2 gain is {during.get('IN2')} dB after asking for "
                            f"{gain_db} dB", hint="the board clamps it to 0..48 in 3 dB steps")
    ctx.log(f"IN2 gain {before.get('IN2')} -> {gain_db} dB for the run")
    return before.get("IN2")


def restore_in2(ctx: Context, previous: int | None, data: dict[str, Any], *,
                previous_pair: int | None = None,
                open_cdc: Callable[[Context], Any] | None = None,
                set_mode_fn: Callable[..., Any] | None = None,
                enum_timeout: float = 20.0, back_to_audio: bool = False) -> None:
    """Put IN2 back to ``previous`` and the input pair to ``previous_pair`` (each
    only when not None), reading both back; never raises.

    Switches to the default profile first when the board has no CDC, and back to
    USB audio afterwards when ``back_to_audio``. Records ``ingain_after`` /
    ``ingain_restored`` and ``in_pair_after`` / ``in_pair_restored``; a failure
    also lands in ``data["failures"]``.
    """
    if previous is None and previous_pair is None:
        return
    from crosspad_hil.usbmode import set_mode as _set_mode

    switch = set_mode_fn or _set_mode
    opener = open_cdc or (lambda c: c.open_cdc())
    try:
        was_audio = ctx.device.usb_mode is UsbMode.AUDIO or ctx.device.ports.cdc is None
        if was_audio:
            ctx.device = switch(ctx.device, UsbMode.DEFAULT, wait=True, timeout_s=enum_timeout)
        link = opener(ctx)
        try:
            if previous is not None:
                try:
                    after = verbs.ingain(link, IN2_INDEX, previous)
                    data["ingain_after"] = after
                    data["ingain_restored"] = after.get("IN2") == previous
                    if data["ingain_restored"]:
                        pins.clear(ctx.device.id, pins.IN2)
                except HilError as e:
                    data["ingain_restored"] = False
                    data.setdefault("failures", []).append(
                        f"IN2 gain not restored to {previous} dB: {e.code}: {e.message}")
            if previous_pair is not None:
                put_pair_back(ctx, link, previous_pair, data)
        finally:
            link.close()
        if was_audio and back_to_audio:
            ctx.device = switch(ctx.device, UsbMode.AUDIO, wait=True, timeout_s=enum_timeout)
    except HilError as e:
        # Only what was not already put back: a failed switch back to audio
        # after both reads does not undo them.
        if previous is not None and data.get("ingain_restored") is None:
            data["ingain_restored"] = False
            data.setdefault("failures", []).append(
                f"IN2 gain not restored to {previous} dB: {e.code}: {e.message}")
        if previous_pair is not None and data.get("in_pair_restored") is None:
            data["in_pair_restored"] = False
            data.setdefault("failures", []).append(
                f"input pair not restored to {previous_pair}: {e.code}: {e.message}")
    if previous is not None and not data.get("ingain_restored"):
        ctx.log(f"IN2 gain NOT restored to {previous} dB (the board keeps it)")
    if previous_pair is not None and not data.get("in_pair_restored"):
        ctx.log(f"input pair NOT restored to {previous_pair} (the board keeps it)")


def in2_not_restored(data: dict[str, Any], previous: int | None) -> str | None:
    """The summary note for a restore that failed, or None."""
    if data.get("ingain_restored") is not False:
        return None
    return (f"IN2 gain left at {(data.get('ingain_during') or {}).get('IN2')} dB, "
            f"not restored to {previous} dB (the board keeps it in NVS)")


def in_pair_not_restored(data: dict[str, Any], previous: int | None) -> str | None:
    """The summary note for an input pair that was not put back, or None."""
    if previous is None or data.get("in_pair_restored") is not False:
        return None
    return (f"input pair left at {data.get('in_pair_during')}, not restored to {previous} "
            "(the board keeps it)")


# ------------------------------------------------------ the input pair (the loop's driver)
# The bench loop is codec 1 LOUT1/ROUT1 driving the Line 3/4 jack net, which
# LINE1 of both codecs reads. Since platform-idf PR #158/#159 (2026-11-04) the
# firmware powers that driver down, hi-Z, whenever the Mixer's input pair hears
# Line 3/4 ("Mic + 3/4", "1/2 + 3/4"). A capture on LINE1 then records a
# floating jack -- the 689 Hz / 2 kHz block-rate spur and hiss at about
# -34 dBFS, every tone "missing" -- on a board that is fine: the v1.1.3 gate's
# audio_loopback 0/164 and 0/126 (2026-11-08/09) on a board left on pair 2.
# A scenario that measures through the loop selects a pair without Line 3/4
# for the run and puts the old one back: the board keeps it.

LOOP_ADC_INPUT = 1          # LINE1, on either codec: the jack net the loop drives
LINE34 = "3/4"


def pair_hears_line34(pair: int, names: dict[int, str]) -> bool:
    return LINE34 in names.get(pair, "")


def put_pair_back(ctx: Context, link: CdcLink, previous: int, data: dict[str, Any]) -> bool:
    """IN_PAIR ``previous``, read back; records ``in_pair_after`` / ``in_pair_restored``."""
    try:
        after = verbs.in_pair(link, previous)
    except HilError as e:
        data["in_pair_restored"] = False
        data.setdefault("failures", []).append(
            f"input pair not restored to {previous}: {e.code}: {e.message}")
        return False
    data["in_pair_after"] = after["pair"]
    data["in_pair_restored"] = after["pair"] == previous
    if data["in_pair_restored"]:
        pins.clear(ctx.device.id, pins.IN_PAIR)
    return data["in_pair_restored"]


def free_jack_loop(ctx: Context, link: CdcLink, data: dict[str, Any]) -> int | None:
    """Make sure the jack loop is driven: on an input pair that hears Line 3/4,
    select the first pair that does not for the run.

    Returns the pair to put back, or None when nothing was changed (the pair is
    fine, or the firmware has no IN_PAIR -- noted in ``data["in_pair_note"]``).
    Records ``in_pair_before`` / ``in_pair_names`` / ``in_pair_during``. ENV
    (exit 2) naming the pair when no pair without Line 3/4 exists or the board
    refuses it: the loop would be measured floating, which says nothing about
    the firmware.
    """
    try:
        data["in_pair_recovered"] = pins.restore_pending(link, ctx.device.id, ctx.log,
                                                         keys=(pins.IN_PAIR,)) or None
        before = verbs.in_pair(link)
    except HilError as e:
        # ERR IN_PAIR none, or a firmware older than the verb (it echoes it).
        if e.code not in (NOT_SUPPORTED, UNKNOWN_VERB):
            raise
        data["in_pair_note"] = f"{e.message}: no input pairs, the loop is the firmware's"
        ctx.log(f"IN_PAIR: {e.message}; the jack loop is left as the board has it")
        return None
    pair, names = before["pair"], before["names"]
    data["in_pair_before"] = pair
    data["in_pair_names"] = {str(k): v for k, v in names.items()}
    if not pair_hears_line34(pair, names):
        return None
    label = f"input pair {pair} ({names.get(pair)})"
    free = [i for i in sorted(names) if not pair_hears_line34(i, names)]
    if not free:
        raise HilError(ENV, f"{label} hears Line 3/4, which turns the jack loop's driver off, "
                            "and every pair does", hint="pairs: " + ", ".join(
                                f"{i}={n}" for i, n in sorted(names.items())))
    target = free[0]
    # Written before the board is touched: a kill from here on leaves the journal
    # holding the pair to put back.
    pins.record(ctx.device.id, pins.IN_PAIR, pair)
    during = verbs.in_pair(link, target)
    data["in_pair_during"] = during["pair"]
    if not during["ok"] or during["pair"] != target:
        if during["pair"] != pair:
            put_pair_back(ctx, link, pair, data)
        else:
            pins.clear(ctx.device.id, pins.IN_PAIR)
        raise HilError(ENV, f"{label} hears Line 3/4, which turns the jack loop's driver off, "
                            f"and the board refused IN_PAIR {target}",
                       hint="set a pair without Line 3/4 in Mixer > IN (or IN_PAIR over "
                            "CDC) and rerun")
    ctx.log(f"{label} hears Line 3/4 and switches the jack loop's driver off: "
            f"pair {target} ({names.get(target)}) for the run")
    return pair


# ------------------------------------------------------------- a kit, then the app
# After an ESP reset the card mounts ~15 s late and the kit scan later still;
# until then KIT_LOAD answers "bad kit id" or "scanning". With no kit loaded the
# Sampler does not start (APP_LIST stays running=-; the firmware logs "App
# already started"). So a scenario loads a kit, waiting for the card, and only
# then starts the app -- with a window long enough for its first kit load.

CARD_WAIT_S = 45.0
APP_START_WAIT_S = 15.0
LAUNCHER_WAIT_S = 10.0


def load_kit_waiting(ctx: Context, link: CdcLink, kit_id: int, *,
                     wait_s: float = CARD_WAIT_S, load_timeout: float = 30.0) -> None:
    """KIT_LOAD that waits out a card still mounting; anything else is raised."""
    import time as _time

    deadline = _time.monotonic() + wait_s
    while True:
        try:
            verbs.kit_load(link, kit_id, wait_s=load_timeout)
            return
        except HilError as e:
            text = f"{e.message} {e.hint or ''}".lower()
            not_yet = "bad kit id" in text or "scan" in text
            if not not_yet or _time.monotonic() >= deadline:
                raise
            ctx.log(f"kit {kit_id}: the card is not listed yet ({e.message}); waiting")
            ctx.cancelled.wait(3.0)
            ctx.check_cancelled()


def start_app(ctx: Context, link: CdcLink, name: str, *, wait_s: float = APP_START_WAIT_S,
              attempts: int = 2, launcher_s: float = LAUNCHER_WAIT_S) -> None:
    """Get ``name`` running: close another app first, then APP_START with a window
    for its kit load, once more when the first request went nowhere."""
    import time as _time

    apps = verbs.app_list(link)
    if name not in apps.get("apps", []):
        raise HilError(ENV, f"{name} is not in the launcher", apps=apps.get("apps", []))
    running = apps.get("running")
    if running == name:
        return
    if running is not None:
        verbs.app_stop(link)
        deadline = _time.monotonic() + launcher_s
        while verbs.app_list(link).get("running") is not None:
            if _time.monotonic() >= deadline:
                raise HilError(TIMEOUT, f"{running} still running {launcher_s:.0f}s after APP_STOP")
            ctx.cancelled.wait(0.25)
    for attempt in range(1, attempts + 1):
        try:
            verbs.app_start(link, name, wait_s=wait_s)
            return
        except HilError as e:
            if e.code != TIMEOUT or attempt == attempts \
                    or e.details.get("running") not in (None, name):
                raise
            ctx.log(f"APP_START {name} attempt {attempt}/{attempts}: {e.message}")
