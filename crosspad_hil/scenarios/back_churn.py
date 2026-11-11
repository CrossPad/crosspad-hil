"""Descend into every app, then click the power button back out — repeatedly.

`app_churn` opens an app and closes it with APP_STOP, which rebuilds the main
screen and papers over what the teardown left behind. This scenario leaves the
app to close *itself* the way a finger does: the power button's short click,
resolved by crosspad_gui::powerBack(). That is a different code path, and it is
the one that crashed.

The shape of the bug it was written for: the Sampler's file explorer, opened
over the pad editor, opened over the sampler panel. A click three levels down
used to skip every level and destroy the app, and the nested layers then
restored encoder groups their owner had already deleted — the next encoder read
dereferenced a freed lv_group and the board panicked in lv_group_get_focused().
Nothing in `app_churn` goes three levels deep, so nothing in `app_churn` saw it.

So each visit: enter the app, walk *down* (press the first focusable thing that
is not a way out, up to --depth times), then click back until the launcher is
up again. A visit fails if the device panics, reboots, writes a core dump,
needs more than --clicks clicks, or comes back to something that is not the
launcher. Between visits the heap is sampled the way app_churn samples it, so a
level that leaks its lv_group or its lv_timer shows up as a per-app slope.

Exit code 0 = PASS, 1 = crash / stuck / leak, 2 = a bench that cannot run it.
"""
from __future__ import annotations

import re
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from crosspad_hil import audio_state, verbs
from crosspad_hil.cdc import CdcLink
from crosspad_hil.errors import ENV, HilError
from crosspad_hil.scenarios import register
from crosspad_hil.scenarios.app_churn import _ConsoleWatch, per_visit, slope
from crosspad_hil.scenarios.base import Artifact, Context, Progress, Report

# Apps that are not a "visit": they power the device down or reflash it.
# Settings is *not* skipped here (app_churn skips it) — it is the other
# multi-level app, and levels are the point.
DEFAULT_SKIP: tuple[str, ...] = ("Power OFF", "Firmware")

# Labels a blind descent must never press. This scenario presses things it has
# not read, so the deny-list is what keeps it from erasing a kit or powering the
# board off in the middle of a run.
# "usb" is here because a control with that word re-enumerates the device: the
# DAW Control app's "USB audio" button and Settings > USB both switch the USB
# profile to MIDI+UAC2, which has no CDC endpoint. A blind press there does not
# fail the app under test — it removes the channel this scenario is driving, and
# every later step reports "no reply" from a board that is perfectly healthy.
DENY = re.compile(
    r"(?i)(power|shut|off\b|delete|erase|format|wipe|reset|reflash|update|flash|dfu|"
    r"bootloader|factory|remove|save|overwrite|record|usb|"
    # REC: a take -- the Recorder's, or the top bar's on the pattern -- writes
    # the card, and a growing recordings list read as the Recorder leaking.
    r"\brec\b|"
    # DAW Control's five capture-preset buttons. Pressing one *persists* the
    # route, and two of them point the capture at a DAC->ADC bus that rails on
    # codec 1 — so a blind press here does not fail back_churn, it fails every
    # audio step that runs after it in the same gate. That is how a release_gate
    # came back with audio_loopback 227 cycles / 227 FAIL on hardware whose pads
    # rendered fine (AUDIOLVL 0.05, SMPLPEAK 5692) once the preset was put back.
    # \bmix\b deliberately does not match the Mixer app.
    r"mics|line in|spk bus|hp bus|\bmix\b|"
    # Saved radio and USB settings. A blind press on a switch that opens nothing
    # is kept as a toggle, and these persist: BLE turned on costs ~30 kB of
    # internal RAM for every later step; the USB serial guard decides whether a
    # profile switch needs a person at the board.
    r"bluetooth|\bble\b|wireless|wi-?fi|guard|"
    # The Sampler's 16 LEVELS switch: left on, every pad plays the source pad's
    # note at a velocity ramp until a restart, so the pitched_kit run after this
    # one hears one pitch on all sixteen pads.
    r"16 levels|16 pitch|full level|"
    # + NEW (a song in Arrange's picker, a kit in the kit browser) writes the
    # card and leaves the new one open: the Arrange heap trend then compares a
    # different song each visit and read +50..+130 B/visit as a leak.
    r"\bnew\b)"
)

# Labels that go *up* a level: pressing one is not a descent, and pressing it
# by accident would make the click count lie about how deep the visit got.
UP = {"back", "close", "exit", "cancel", "go back...", "go back", "<", "abort"}

# The same, drawn instead of spelled: a button whose whole label is an LVGL
# glyph. LV_SYMBOL_LEFT is the launcher-bound arrow the first sweep kept
# pressing — it took the walk *out* of the app, and everything it pressed after
# that was the launcher, which is how "descended 3 levels into KitSelector"
# came to mean "started three apps".
UP_GLYPHS = ("\uf053", "\uf060", "\uf00d", "\uf104", "\uf191")


def is_way_out(label: str) -> bool:
    text = label.strip().lower()
    return text in UP or any(g in label for g in UP_GLYPHS)

KIT_SEED_WAIT_S = 25.0
HEAP_SETTLE_POLL_S = 0.5   # MEM cadence while waiting for the heap to stop moving
HEAP_SETTLE_STABLE = 3     # consecutive equal reads (1 s) that count as settled
HEAP_SETTLE_MAX_S = 10.0
START_WAIT_S = 4.0


@dataclass
class Params:
    rounds: int = field(
        default=8,
        metadata={"help": "descend/unwind cycles per target (8 makes a leak measurable)"},
    )
    depth: int = field(
        default=3, metadata={"help": "levels to try to descend before clicking back out"}
    )
    clicks: int = field(
        default=8, metadata={"help": "power clicks allowed to reach the launcher again"}
    )
    apps: list[str] | None = field(
        default=None, metadata={"help": "comma-separated subset of APP_LIST; default is all"}
    )
    skip: list[str] | None = field(
        default=None,
        metadata={"help": "comma-separated apps to leave alone (default: Power OFF,Firmware)"},
    )
    cases: list[str] | None = field(
        default=None,
        metadata={"help": "scripted deep states to add: sampler_explorer,settings_deep,drawer "
                          "(default: all of them; pass 'none' for none)"},
    )
    dwell: float = field(default=0.9, metadata={"help": "seconds to settle after each press"})
    settle: float = field(
        default=1.2, metadata={"help": "seconds after the last click before sampling the heap"}
    )
    leak_bytes: int = field(
        default=32, metadata={"help": "per-visit internal-heap loss that counts as a leak"}
    )
    pads: int = field(
        default=2,
        metadata={"help": "pad hits to fire inside an app that plays them, per visit "
                          "(0 = walk the UI silently, as this scenario used to)"},
    )


# ── UI walking ───────────────────────────────────────────────────────────────


def group_labels(link: CdcLink) -> list[str]:
    return [str(e.get("label") or "") for e in verbs.enc_group(link)["group"]]


def group_ptrs(link: CdcLink) -> list[str]:
    return [str(e.get("ptr") or "") for e in verbs.enc_group(link)["group"]]


def is_launcher(labels: list[str], apps: list[str]) -> bool:
    """The launcher is up when its own tiles are what the encoder group holds.

    Counting objects is not enough — a screen with the same number of
    focusables would pass — and requiring every app is too strict, since the
    group is rebuilt and a tile can be missing while the launcher is fine.
    """
    present = sum(1 for a in apps if a in labels)
    return present >= min(3, len(apps))


def focus_label(link: CdcLink, label: str, limit: int) -> bool:
    """Rotate until `label` is focused. False if it never comes round."""
    return focus_match(link, lambda lb: lb == label, limit)


def focus_match(link: CdcLink, match: Callable[[str], bool], limit: int) -> bool:
    """Rotate until a label `match` accepts is focused. False if it never comes round."""
    for _ in range(limit):
        if match(str(verbs.enc_focus(link).get("label") or "")):
            return True
        verbs.enc_rotate(link, 1)
    return match(str(verbs.enc_focus(link).get("label") or ""))


def descend(ctx: Context, link: CdcLink, depth: int, dwell: float, pick: int,
            inside: str | None = None) -> list[str]:
    """Press into up to `depth` levels; return the label pressed at each.

    `pick` rotates which candidate is chosen from round to round, so eight
    rounds cover eight different ways into the app rather than the same one
    eight times.

    `inside` is the app the walk must stay in. A press that ends the app is not
    a level, and everything pressed after it would be the launcher — counted as
    depth, that turns "three levels deep" into "three apps started", which is
    exactly what the first sweep reported.
    """
    path: list[str] = []
    for _level in range(depth):
        ctx.check_cancelled()
        entries = verbs.enc_group(link)["group"]
        labels = [str(e.get("label") or "") for e in entries]
        before = [str(e.get("ptr") or "") for e in entries]
        candidates = [lb for lb in labels if lb and not is_way_out(lb) and not DENY.search(lb)]
        if not candidates:
            break
        target = candidates[pick % len(candidates)]
        if not focus_label(link, target, len(labels) + 2):
            break
        verbs.enc_press(link)
        ctx.cancelled.wait(dwell)
        if inside is not None and verbs.app_list(link)["running"] != inside:
            break
        if group_ptrs(link) == before:
            # Nothing opened: that press was a toggle or a no-op, and the app is
            # no deeper than it was. Reporting it as a level would make the
            # unwind assertion expect a click that has nothing to close.
            break
        path.append(target)
    return path


# Targets whose visit should make a sound. Walking the Sampler with the engine
# idle exercises its widgets and nothing else; a finger that opens it plays it,
# and a voice allocated under a half-torn-down screen is a different code path
# from a voice allocated at rest.
PAD_TARGETS = ("Sampler", "sampler_explorer")
PAD_APP = "Sampler"
PAD_COUNT = 16
PAD_HOLD_S = 0.12
PAD_GAP_S = 0.15
PAD_QUIET_POLL_S = 0.25
PAD_QUIET_MAX_S = 4.0


def play_pads(ctx: Context, link: CdcLink, hits: int, rnd: int) -> dict[str, Any]:
    """Hit `hits` pads inside the running app, then wait for the voices to end.

    PAD_PRESS answers OK the way KIT_LOAD does, so the proof a hit reached the
    engine is the PAD_STATS delta, not the reply. The heap is not sampled until
    the sample slots come back, or the hits would be charged to the visit as a
    leak that frees itself one visit later.
    """
    out: dict[str, Any] = {"fired": 0, "played": 0, "quiet": True}
    try:
        before = verbs.pad_stats(link)
        free0 = verbs.smpl_peak(link)["free"]
        for i in range(hits):
            ctx.check_cancelled()
            pad = (rnd * hits + i) % PAD_COUNT
            try:
                verbs.pad_press(link, pad, 100)
                ctx.cancelled.wait(PAD_HOLD_S)
            finally:
                verbs.pad_release(link, pad)
            out["fired"] += 1
            ctx.cancelled.wait(PAD_GAP_S)
        out["played"] = verbs.pad_stats(link)["played"] - before["played"]
        deadline = time.monotonic() + PAD_QUIET_MAX_S
        while verbs.smpl_peak(link)["free"] < free0 and time.monotonic() < deadline:
            ctx.cancelled.wait(PAD_QUIET_POLL_S)
        out["quiet"] = verbs.smpl_peak(link)["free"] >= free0
    except HilError as exc:
        out["error"] = f"{exc.code}: {exc.message}"
    return out


def settled_int_free(link: CdcLink, cancelled: threading.Event) -> int:
    """int_free once it has stopped moving.

    An async teardown or the kit loader's tail can still be running when the
    fixed settle expires; sampled then, its bytes are charged to this target and
    refunded to whichever one is measured next -- gate #6 read that as
    KitSelector +662 B/visit against Settings -397 B/visit, net zero.
    """
    last = int(verbs.mem(link)["int_free"])
    same = 1
    deadline = time.monotonic() + HEAP_SETTLE_MAX_S
    while same < HEAP_SETTLE_STABLE and time.monotonic() < deadline:
        cancelled.wait(HEAP_SETTLE_POLL_S)
        cur = int(verbs.mem(link)["int_free"])
        same = same + 1 if cur == last else 1
        last = cur
    return last


# ADC input names AUDIO_ADC takes, by the value AUDIO_STATUS reports
# (audio_route_control.cpp parse_adc_input_name).
_ADC_NAMES = {0: "DIFF", 1: "LINE1", 2: "LINE2"}
_AUDIO_RE = re.compile(r"mic_src=(-?\d+)\s+adc=\[(-?\d+),(-?\d+)\]")


def _audio_route(link: CdcLink) -> tuple[int, int, int] | None:
    """(mic_src, adc0, adc1), or None when the board would not say."""
    try:
        reply = link.transact("AUDIO_STATUS", expect="AUDIO:")
    except HilError:
        return None
    m = _AUDIO_RE.search(reply.line)
    return (int(m.group(1)), int(m.group(2)), int(m.group(3))) if m else None


def _restore_audio(ctx: Context, link: CdcLink, home: tuple[int, int, int] | None) -> bool:
    """Put the capture route the run started with back.

    A blind descent can press a capture-preset button, and the preset
    *persists* — two of them point capture at a DAC->ADC bus that rails on
    codec 1. Nothing in this scenario notices, and then every audio step that
    runs after it in the same release_gate fails on a healthy board. The
    deny-list keeps those labels unpressed; this is the belt to that brace, for
    the next control that grows the same power under a name nobody denied.

    True when the route is (or was put back) where it started.
    """
    if home is None:
        return True
    now = _audio_route(link)
    if now == home:
        return True
    mic_src, adc0, adc1 = home
    ok = True
    # Inputs before the capture source, the order audio_route_apply_preset uses:
    # switching the source first streams one buffer of the old input.
    for codec, value in ((0, adc0), (1, adc1)):
        name = _ADC_NAMES.get(value)
        if name is None:
            continue
        try:
            link.transact(f"AUDIO_ADC {codec} {name}", expect="OK")
        except HilError:
            ok = False
    try:
        link.transact(f"AUDIO_MIC_SRC {mic_src}", expect="OK")
    except HilError:
        ok = False
    if ok and _audio_route(link) == home:
        return True
    ctx.log(f"capture route not restored (wanted {home}, got {_audio_route(link)}) — "
            "audio steps after this run may fail on a healthy board")
    return False


SONG_OPEN_WAIT_S = 15.0


def open_song(link: CdcLink) -> str | None:
    """The song Arrange has open, or None when the board cannot say."""
    try:
        return verbs.asong(link).get("project") or None
    except HilError:
        return None


def _restore_song(ctx: Context, link: CdcLink, home: str | None) -> bool:
    """Open the song the visit started with again, before the kit and the heap.

    Arrange's first control is the open song's name, which opens the song
    picker: a walk that presses a row there leaves another song open, and the
    open song (its tracks, its kit, its Pitch instrument) is internal RAM that
    differs from song to song -- the Arrange trend read that as a leak. Before
    the kit: a song names its kit and opening it loads that one. False when the
    song could not be opened again, and the caller counts that against the trend.
    """
    if home is None or open_song(link) == home:
        return True
    try:
        if verbs.asong(link, f"OPEN {home}").get("ok") != 1:
            ctx.log(f"song {home!r} not reopened (refused) — heap after this visit is noisy")
            return False
        deadline = time.monotonic() + SONG_OPEN_WAIT_S
        while time.monotonic() < deadline:
            st = verbs.asong(link)
            if st.get("project") == home and st.get("open") == 2:
                return True
            if st.get("open") == 3:
                break
            ctx.cancelled.wait(0.5)
    except HilError as e:
        ctx.log(f"song {home!r} not reopened ({e.code}) — heap after this visit is noisy")
        return False
    ctx.log(f"song {home!r} not reopened in {SONG_OPEN_WAIT_S:.0f} s "
            "— heap after this visit is noisy")
    return False


def _restore_kit(ctx: Context, link: CdcLink, home: int) -> bool:
    """Put the kit the run started with back before the heap is sampled.

    A walk into the kit selector loads whichever row it pressed, and a loaded
    kit holds ~12 kB of internal RAM that differs from kit to kit. Sampled as
    it lands, the selector read as a 328 B/visit leak that app_churn (which
    never presses a row) could not see. False when the kit could not be put
    back, and the caller counts that against the trend.
    """
    try:
        status = verbs.kit_status(link)
    except HilError:
        return False
    if (status["current"] == home and not status["loading"]
            and status["pending"] == -1):
        return True
    if home < 0:
        return False
    try:
        verbs.kit_load(link, home, wait_s=KIT_SEED_WAIT_S)
        return True
    except HilError as e:
        ctx.log(f"kit {home} not restored ({e.code}) — heap after this visit is noisy")
        return False


def _settings(link: CdcLink) -> dict[str, Any]:
    """BLE running/mode, the USB serial guard and WiFi enabled; None where unknown."""
    home: dict[str, Any] = {"ble_running": None, "ble_mode": None,
                            "usb_guard": None, "wifi_enabled": None}
    try:
        st = verbs.ble_status(link)
        home["ble_running"] = int(st["running"]) == 1
        home["ble_mode"] = st.get("mode")
    except (HilError, KeyError, TypeError, ValueError):
        pass
    try:
        home["usb_guard"] = verbs.usb_guard(link)
    except HilError:
        pass
    try:
        home["wifi_enabled"] = int(verbs.wifi_status(link)["enabled"]) == 1
    except (HilError, KeyError, TypeError, ValueError):
        pass
    return home


def _restore_settings(ctx: Context, link: CdcLink, home: dict[str, Any] | None) -> list[str]:
    """Put BLE, the USB guard and WiFi back where the run found them.

    The deny-list keeps the radio and USB toggles unpressed, but the guard and
    WiFi persist, and a BLE left up leaves the board at ~8 kB internal free for
    whatever the gate runs next. Best-effort: what could not be put back is
    returned (and logged), never raised.
    """
    if home is None or all(v is None for v in home.values()):
        return []
    now = _settings(link)
    failed: list[str] = []

    def put(what: str, fn: Any) -> None:
        try:
            fn()
        except HilError as e:
            ctx.log(f"{what} not restored ({e.code}: {e.message})")
            failed.append(what)

    if home["ble_running"] is not None and now["ble_running"] is not None:
        mode = 0 if home["ble_mode"] == "host" else 1
        if home["ble_running"] and not now["ble_running"]:
            put("ble", lambda: verbs.ble_start(link, mode))
        elif not home["ble_running"] and now["ble_running"]:
            put("ble", lambda: verbs.ble_stop(link))
    if home["usb_guard"] is not None and now["usb_guard"] not in (None, home["usb_guard"]):
        put("usb_guard", lambda: verbs.usb_guard(link, home["usb_guard"]))
    wifi = home["wifi_enabled"]
    if wifi is not None and now["wifi_enabled"] not in (None, wifi):
        put("wifi", lambda: verbs.wifi_enable(link, wifi))
    return failed


# ── Scripted deep states ─────────────────────────────────────────────────────


def _ensure_kit(ctx: Context, link: CdcLink) -> bool:
    try:
        if int(verbs.kit_status(link).get("current", -1)) >= 0:
            return True
        kits = verbs.kit_list(link).get("kits", [])
    except HilError:
        return False
    if not kits:
        return False
    try:
        verbs.kit_load(link, int(kits[0]["id"]), wait_s=KIT_SEED_WAIT_S)
        return True
    except HilError as e:
        ctx.log(f"kit seed failed ({e.code}) — the sampler case will be skipped")
        return False


def is_edit(label: str) -> bool:
    return re.search(r"\bedit\b", label, re.IGNORECASE) is not None


def is_sample_button(label: str) -> bool:
    """The pad editor's file button: the sample's name behind a note, or a folder to pick one."""
    return label.startswith((SYMBOL_AUDIO, SYMBOL_DIRECTORY))


SYMBOL_AUDIO = "\uf001"      # LV_SYMBOL_AUDIO
SYMBOL_DIRECTORY = "\uf07b"  # LV_SYMBOL_DIRECTORY
SAMPLER_STEPS: tuple[tuple[str, Callable[[str], bool]], ...] = (
    ("EDIT", is_edit), ("the sample button", is_sample_button))
SAMPLER_STEPS_ADVANCED: tuple[tuple[str, Callable[[str], bool]], ...] = tuple(
    (want, lambda lb, want=want: lb == want) for want in ("Advanced", "Edit", "Browse"))


def case_sampler_explorer(ctx: Context, link: CdcLink, p: Params) -> list[str]:
    """Sampler → pad editor → file explorer → inside a folder. The reported bug."""
    if not _ensure_kit(ctx, link):
        raise HilError(ENV, "no kit on the card; the sampler explorer case needs one")
    verbs.app_start(link, "Sampler", wait_s=START_WAIT_S)
    ctx.cancelled.wait(p.dwell)
    path = []
    # EDIT sits on the panel since the device-look layout (10.2026) and the pad
    # editor's file button is named after the sample; from 13.09 until then, Edit
    # was inside Advanced and the editor had a Browse button. The transport rows
    # make the group longer than any fixed rotation budget.
    new_layout = any(is_edit(lb) for lb in group_labels(link))
    steps = SAMPLER_STEPS if new_layout else SAMPLER_STEPS_ADVANCED
    for label, match in steps:
        if not focus_match(link, match, len(group_labels(link)) + 2):
            raise HilError(ENV, f"the Sampler has no '{label}' entry — UI changed?")
        verbs.enc_press(link)
        ctx.cancelled.wait(p.dwell + 1.2)
        path.append(label)
        if new_layout and label == "EDIT":
            # The editor opens on no pad until one is hit; the sample button does
            # nothing without one.
            verbs.pad_press(link, 0, 100)
            verbs.pad_release(link, 0)
            ctx.cancelled.wait(p.dwell)
    # The first entry that is a folder, i.e. not a file and not a way out. A
    # label of icons alone (the explorer's back arrow) has no letter or digit.
    labels = group_labels(link)
    folders = [
        lb for lb in labels
        if any(ch.isalnum() for ch in lb) and "." not in lb and lb.strip().lower() not in UP
        and lb not in ("Load",) and not DENY.search(lb)
    ]
    if not folders:
        raise HilError(ENV, "no folder in the sample browser to descend into")
    if focus_label(link, folders[0], len(labels) + 2):
        verbs.enc_press(link)
        ctx.cancelled.wait(p.dwell + 0.8)
        path.append(folders[0])
    return path


def case_settings_deep(ctx: Context, link: CdcLink, p: Params) -> list[str]:
    """Settings, two categories deep, with whatever the second press opened."""
    verbs.app_start(link, "Settings", wait_s=START_WAIT_S)
    ctx.cancelled.wait(p.dwell)
    return ["Settings"] + descend(ctx, link, 2, p.dwell, pick=0, inside="Settings")


def case_drawer(ctx: Context, link: CdcLink, p: Params) -> list[str]:
    """No app: the quick-settings drawer, opened by the hold gesture."""
    verbs.pwr_gesture(link, hold=True)
    ctx.cancelled.wait(p.dwell)
    return ["drawer"]


CASES = {
    "sampler_explorer": case_sampler_explorer,
    "settings_deep": case_settings_deep,
    "drawer": case_drawer,
}


# ── Report ───────────────────────────────────────────────────────────────────


def _report(data: dict[str, Any], console_log: str, leak_bytes: int) -> Report:
    lines: list[str] = ["--- per-target free-heap trend " + "-" * 28]
    leaks: list[str] = []
    unmeasured: list[str] = []
    slopes: dict[str, dict[str, Any]] = {}
    # The launcher's heap at the end of every round is the ground truth. A
    # target that hands memory back only once the next app starts (the kit
    # restore leaves ~650 B until then) reads as a leak in its own column and
    # as a gain in the next one, while the round series stays flat.
    # A round is one visit per target, so the launcher series is denominated in
    # rounds while leak_bytes is denominated in visits. Comparing them raw made
    # the gate len(trend) times too strict: gate #7 read 37 B/round -- 3.7 B per
    # visit across ten targets, 188 B over the whole run -- as "over 32 B",
    # disarmed the refund for every target, and failed the run on KitSelector's
    # -695 B/visit against Settings' +656 B/visit while every other target sat
    # at exactly 60727 in all eight rounds.
    round_rate: float | None = None
    round_per_visit: float | None = None
    r_head = r_tail = 0.0
    visits_per_round = max(1, len(data["trend"]))
    rs = slope(data.get("heap_rounds", []))
    if rs is not None:
        r_head, r_tail, r_n, r_drop = rs
        round_rate = per_visit(r_drop, r_n)
        round_per_visit = round_rate / visits_per_round
        data["round_slope"] = {"head": r_head, "tail": r_tail, "rounds": r_n,
                               "per_round": round_rate,
                               "per_visit": round_per_visit,
                               "visits_per_round": visits_per_round}
    # A target that really loses `rate` bytes every visit is visited once per
    # round, so the launcher must lose at least that much per round too. When
    # the round loses substantially less, the bytes came back inside the round
    # and the column is an accounting artifact, not a leak.
    def is_refunded(rate: float) -> bool:
        return round_rate is not None and round_rate + leak_bytes < rate
    for name, series in data["trend"].items():
        s = slope(series)
        if s is None:
            lines.append(f"  {name:<20} too few samples ({len(series)})")
            continue
        head, tail, visits, drop = s
        rate = per_visit(drop, visits)
        measurable = visits > 1
        over = measurable and rate > leak_bytes
        refunded = over and is_refunded(rate)
        leak = over and not refunded
        slopes[name] = {"head": head, "tail": tail, "visits": visits,
                        "per_visit": rate, "leak": leak, "measurable": measurable,
                        "refunded": refunded}
        if not measurable:
            unmeasured.append(name)
            lines.append(f"  {name:<20} {int(head):>8} -> {int(tail):>8}  "
                         "(not measurable at this many rounds)")
            continue
        if leak:
            verdict = f"  <-- LEAK ~{int(rate)} B/visit"
        elif over:
            verdict = "  (handed back before the round ended)"
        else:
            verdict = ""
        if leak:
            leaks.append(f"{name}: ~{int(rate)} B lost per visit "
                         f"({int(drop)} B from head to tail of {visits} visits)")
        lines.append(f"  {name:<20} {int(head):>8} -> {int(tail):>8}  "
                     f"({int(rate):+6} B/visit){verdict}")
    if round_rate is not None:
        lines.append(f"  {'launcher, per round':<20} {int(r_head):>8} -> {int(r_tail):>8}  "
                     f"({int(round_rate):+6} B/round = {int(round_per_visit):+4} B/visit "
                     f"over {visits_per_round} targets)")
        if round_per_visit > leak_bytes and not leaks:
            leaks.append(f"launcher heap falls ~{int(round_rate)} B per round "
                         f"({int(round_per_visit)} B/visit over {visits_per_round} targets) "
                         f"with no single target over {leak_bytes} B/visit")
    data["slopes"] = slopes
    data["leaks"] = leaks
    data["unmeasured"] = unmeasured

    lines.append("")
    lines.append("--- levels reached / clicks to unwind " + "-" * 21)
    for name, depths in data["depths"].items():
        clicks = data["clicks"].get(name, [])
        lines.append(f"  {name:<20} depth {min(depths, default=0)}..{max(depths, default=0)}"
                     f"   clicks {min(clicks, default=0)}..{max(clicks, default=0)}")

    pad_problems: list[str] = []
    plays = data.get("pad_play") or {}
    if plays:
        lines.append("")
        lines.append("--- pads played inside the app " + "-" * 28)
        for name, rows in plays.items():
            fired = sum(int(r.get("fired", 0)) for r in rows)
            played = sum(int(r.get("played", 0)) for r in rows)
            errs = [r["error"] for r in rows if r.get("error")]
            noisy = sum(1 for r in rows if not r.get("quiet", True))
            left = [r["left"] for r in rows if "left" in r]
            note = ""
            if errs:
                note = f"   ({len(errs)} visit(s) could not drive the pads: {errs[0]})"
            elif noisy:
                note = f"   ({noisy} visit(s) still sounding when the heap was sampled)"
            elif left:
                note = (f"   ({len(left)} visit(s) left for "
                        f"{', '.join(sorted(set(map(str, left))))})")
            lines.append(f"  {name:<20} {fired:>4} hit(s), {played:>4} played{note}")
            if fired and not played and not errs:
                pad_problems.append(
                    f"{name}: {fired} pad hit(s) fired and the engine played none")

    if data.get("audio_restores_failed"):
        lines.append(f"  (capture route not restored after "
                     f"{data['audio_restores_failed']} visit(s): audio steps after "
                     f"this run may fail on healthy hardware)")
    if data.get("song_restores_failed"):
        lines.append(f"  (song not reopened after {data['song_restores_failed']} visit(s): "
                     "the Arrange slope includes song-footprint noise)")
    if data.get("kit_restores_failed"):
        lines.append(f"  (kit not restored after {data['kit_restores_failed']} visit(s): "
                     "the KitSelector slope includes kit-footprint noise)")
    failures = list(data["failures"]) + leaks + pad_problems
    lines.append("")
    if failures:
        lines.append(f"FAIL — {len(failures)} problem(s):")
        for f in failures:
            lines.append(f"  * {f}")
        passed = False
    else:
        # A run that never got past level 1 anywhere proves nothing about the
        # thing this scenario exists to test, so it says so instead of passing
        # quietly.
        deepest = max((max(d, default=0) for d in data["depths"].values()), default=0)
        if deepest < 2:
            lines.append("PASS — no crashes, but nothing went deeper than one level: "
                         "the multi-level path was NOT exercised")
        elif unmeasured:
            lines.append("PASS — no crashes; leak check INCONCLUSIVE for "
                         f"{len(unmeasured)} target(s) at these rounds (needs 4+): "
                         + ", ".join(unmeasured))
        else:
            lines.append(f"PASS — every target unwound to the launcher; "
                         f"no heap slope over {leak_bytes} B/visit")
        passed = True
    lines.append(f"console log: {console_log}")
    return Report(
        passed=passed,
        summary="\n".join(lines),
        data=data,
        artifacts=[Artifact(path=console_log, mime="text/plain", role="console")],
        exit_code=0 if passed else 1,
    )


class BackChurnScenario:
    name = "back_churn"
    Params = Params
    description = ("descend into every app and click the power button back out; "
                   "crash watch + per-target heap slope")

    def run(self, ctx: Context, params: Params, progress: Progress) -> Report:
        try:
            console_log = str(ctx.workdir / "console.log")
            data: dict[str, Any] = {
                "apps": [], "targets": [], "skipped": [], "rounds": params.rounds,
                "depth": params.depth, "clicks_allowed": params.clicks,
                "trend": {}, "slopes": {}, "leaks": [], "failures": [],
                "depths": {}, "clicks": {}, "paths": {},
                "fatal": [], "reboots": 0, "coredump": None,
                "leak_bytes": params.leak_bytes,
                "kit_home": -1, "kit_restores_failed": 0, "song_restores_failed": 0,
                "audio_home": None, "audio_restores_failed": 0,
                "settings_home": None, "settings_restore_failed": [],
            }
            # A CDC timeout six rounds in used to throw all of this away: the
            # report kept the error and nothing else. Handed over by reference,
            # so whatever the loop has filled in by then survives the failure.
            ctx.partial = data
            console = ctx.open_console(reset=False)
            link = ctx.open_cdc()
            watch = _ConsoleWatch(console)
            kit_home: int = -1
            audio_home: tuple[int, int, int] | None = None
            settings_home: dict[str, Any] | None = None
            # Mixer mutes/volumes, codec volumes and the ADC routing outlive a run, and a
            # blind press can land on any of them (the volume overlay, a muted Sampler).
            audio_snap: dict[str, Any] | None = None
            try:
                audio_snap = audio_state.snapshot(link)
                listing = verbs.app_list(link)
                apps: list[str] = list(listing["apps"])
                data["apps"] = apps
                if not apps:
                    return Report(
                        passed=False,
                        summary="APP_LIST returned nothing — firmware without HIL control?",
                        data=data, artifacts=[], exit_code=2)
                if listing["running"]:
                    verbs.app_stop(link)
                    ctx.cancelled.wait(params.settle)

                try:
                    dump0 = verbs.coredump_info(link)
                except HilError:
                    dump0 = {"present": False, "seq": None}
                data["coredump_before"] = dump0.get("seq")

                skip = set(DEFAULT_SKIP if params.skip is None else params.skip)
                if params.apps:
                    unknown = [a for a in params.apps if a not in apps]
                    if unknown:
                        return Report(passed=False,
                                      summary=f"not in the launcher: {', '.join(unknown)}",
                                      data=data, artifacts=[], exit_code=2)
                    targets = list(params.apps)
                    skipped: list[str] = []
                else:
                    targets = [a for a in apps if a not in skip]
                    skipped = [a for a in apps if a in skip]
                if params.cases is None:
                    # --apps names exactly what to visit; the scripted states
                    # are not among those names, so they stay out unless asked
                    # for by --cases.
                    cases = [] if params.apps else list(CASES)
                else:
                    wanted = [c.strip() for c in params.cases if c.strip()]
                    if wanted == ["none"]:
                        wanted = []
                    unknown = [c for c in wanted if c not in CASES]
                    if unknown:
                        return Report(passed=False,
                                      summary=f"unknown case(s): {', '.join(unknown)}; "
                                              f"known: {', '.join(CASES)}",
                                      data=data, artifacts=[], exit_code=2)
                    cases = wanted
                if "Sampler" not in apps and "sampler_explorer" in cases:
                    cases.remove("sampler_explorer")
                if "Settings" not in apps and "settings_deep" in cases:
                    cases.remove("settings_deep")
                names = targets + cases
                data["targets"] = names
                data["skipped"] = skipped
                if ("Sampler" in targets or "KitSelector" in targets
                        or "sampler_explorer" in cases):
                    # The Sampler refuses to launch with no current kit (the
                    # kit-selector gate owns the screen instead and APP_LIST
                    # still reports running=-), and that reads as the app
                    # failing when it is the bench that is unprepared. The
                    # selector loads a kit on its first press, so starting with
                    # one loaded keeps its visits comparable to each other.
                    _ensure_kit(ctx, link)
                try:
                    kit_home = int(verbs.kit_status(link)["current"])
                except HilError:
                    kit_home = -1
                data["kit_home"] = kit_home
                audio_home = _audio_route(link)
                data["audio_home"] = list(audio_home) if audio_home else None
                # A blind press can flip a saved radio/USB setting the deny-list
                # did not name; the finally puts back whatever moved.
                settings_home = _settings(link)
                data["settings_home"] = settings_home
                ctx.log(f"{len(targets)} apps + {len(cases)} scripted states, "
                        f"{params.rounds} rounds, depth {params.depth}, "
                        f"{params.clicks} clicks allowed")
                if skipped:
                    ctx.log(f"skipped: {', '.join(skipped)}")

                trend: dict[str, list[int]] = {n: [] for n in names}
                failures: list[str] = []
                depths: dict[str, list[int]] = {n: [] for n in names}
                clicks_used: dict[str, list[int]] = {n: [] for n in names}
                paths: dict[str, list[str]] = {}
                pad_play: dict[str, list[dict[str, Any]]] = {}
                data.update({"trend": trend, "failures": failures,
                             "depths": depths, "clicks": clicks_used, "paths": paths,
                             "pad_play": pad_play})

                total = params.rounds * len(names)
                visit = 0
                for rnd in range(1, params.rounds + 1):
                    for name in names:
                        ctx.check_cancelled()
                        visit += 1
                        progress(visit, total, f"round {rnd}/{params.rounds} {name}")
                        free_before = settled_int_free(link, ctx.cancelled)
                        song_before = open_song(link)

                        # ── in ──────────────────────────────────────────────
                        try:
                            if name in CASES:
                                path = CASES[name](ctx, link, params)
                            else:
                                verbs.app_start(link, name, wait_s=START_WAIT_S)
                                ctx.cancelled.wait(params.dwell)
                                path = descend(ctx, link, params.depth, params.dwell,
                                               pick=rnd - 1, inside=name)
                        except HilError as e:
                            failures.append(
                                f"{name}: could not get in ({e.code}: {e.message}) on round {rnd}")
                            self._recover(ctx, link, params)
                            continue
                        depths[name].append(len(path))
                        paths.setdefault(name, path)

                        if params.pads and name in PAD_TARGETS:
                            # A press deeper in can hand the pad to another app
                            # (a Sampler pad's menu opens the Slicer or Pitch
                            # Play): hits there are not the Sampler's to count.
                            running = verbs.app_list(link)["running"]
                            pad_play.setdefault(name, []).append(
                                play_pads(ctx, link, params.pads, rnd)
                                if running == PAD_APP else {"left": running})

                        # ── back out, one click at a time ───────────────────
                        used = 0
                        home = False
                        dead = None
                        for _ in range(params.clicks):
                            used += 1
                            try:
                                verbs.pwr_gesture(link)
                                ctx.cancelled.wait(params.dwell)
                                running = verbs.app_list(link)["running"]
                                labels = group_labels(link)
                            except HilError as e:
                                dead = f"{e.code}: {e.message}"
                                break
                            if not running and is_launcher(labels, apps):
                                home = True
                                break
                        clicks_used[name].append(used)

                        fatal, reboots = watch.poll()
                        if fatal or reboots:
                            data["fatal"].extend(fatal)
                            data["reboots"] += reboots
                            for f in fatal[-5:]:
                                ctx.log(f"   {f}")
                            what = "device fault" if fatal else "device rebooted"
                            failures.append(
                                f"{name}: {what} clicking back out of {path or '[level 0]'} "
                                f"on round {rnd} — see {console_log}")
                            data["coredump"] = self._dump(link)
                            return _report(data, console_log, params.leak_bytes)
                        if dead is not None:
                            failures.append(
                                f"{name}: device stopped answering after {used} click(s) "
                                f"out of {path} on round {rnd} ({dead})")
                            data["coredump"] = self._dump(link)
                            return _report(data, console_log, params.leak_bytes)
                        if not home:
                            failures.append(
                                f"{name}: {params.clicks} clicks did not reach the launcher "
                                f"from {path or '[level 0]'} on round {rnd}")
                            self._recover(ctx, link, params)
                            continue

                        dump = self._dump(link)
                        if dump and dump.get("seq") != data["coredump_before"]:
                            data["coredump"] = dump
                            failures.append(
                                f"{name}: a core dump appeared on round {rnd} "
                                f"({dump.get('line', '')})")
                            return _report(data, console_log, params.leak_bytes)

                        if not _restore_song(ctx, link, song_before):
                            data["song_restores_failed"] += 1
                        if not _restore_kit(ctx, link, kit_home):
                            data["kit_restores_failed"] += 1
                        if not _restore_audio(ctx, link, audio_home):
                            data["audio_restores_failed"] += 1
                        ctx.cancelled.wait(params.settle)
                        free_after = settled_int_free(link, ctx.cancelled)
                        series = trend[name]
                        series.append((series[-1] if series else free_before)
                                      - (free_before - free_after))

                    heap_now = settled_int_free(link, ctx.cancelled)
                    m = verbs.mem(link)
                    data.setdefault("heap_rounds", []).append(heap_now)
                    psram = (f"  psram_free={m.get('psram_free')}"
                             f" psram_largest={m.get('psram_largest')}")
                    data.setdefault("psram", []).append(
                        {"round": rnd, "free": m.get("psram_free"),
                         "largest": m.get("psram_largest")})
                    ctx.log(f"round {rnd:3d}/{params.rounds}  int_free={heap_now}{psram}")

                return _report(data, console_log, params.leak_bytes)
            finally:
                # Each clean visit puts the kit and the route back, but a visit
                # that ends early (a crash, a cancel, an early return) skipped it;
                # the next steps of a gate measure audio on whatever was left.
                try:
                    _restore_kit(ctx, link, kit_home)
                    _restore_audio(ctx, link, audio_home)
                except HilError as e:
                    ctx.log(f"leaving the kit/route as found: {e.code}: {e.message}")
                data["settings_restore_failed"] = _restore_settings(ctx, link, settings_home)
                audio_state.put_back(link, audio_snap, ctx.log, data)
                link.close()
                console.close()
        finally:
            ctx.close_all()

    @staticmethod
    def _dump(link: CdcLink) -> dict[str, Any] | None:
        try:
            d = verbs.coredump_info(link)
        except HilError:
            return None
        return d if d.get("present") else {"present": False, "seq": None}

    @staticmethod
    def _recover(ctx: Context, link: CdcLink, params: Params) -> None:
        """Put the device back in the launcher after a failed visit.

        APP_STOP, not more clicks: it rebuilds the main screen, so whatever
        state the visit ended in cannot leak into the next target's measurement.
        """
        try:
            verbs.app_stop(link)
        except HilError:
            return
        ctx.cancelled.wait(params.settle)


SCENARIO = register(BackChurnScenario())
