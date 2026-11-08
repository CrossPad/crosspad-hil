"""The audio state a UI-driving scenario can leave behind, read and put back.

A scenario that presses and turns the encoder through the whole UI (back_churn,
app_churn, firmware_app, pad_edit_sticky, the stability soak) can land on a
control that changes audio state which *outlives the run*, and the next audio
scenario then measures silence. On 2026-10-09 each of these cost hours:

* the Mixer channel "Sampler" muted and saved (``MIX_LVL ch=0 name=Sampler ...
  mute=1``) -- survives a reboot (``MIX_MUTE`` puts it back);
* the codec output volumes moved by the volume overlay (``AUDIO_STATUS
  vol=[28,50]``, factory 64/64);
* the ADC routing changed (``adc=[2,1]`` is the good default, ``adc=[1,2]``
  raised the floor).

(The Sampler's "16 LEVELS" performance mode is RAM state with no verb; rt_glitch
presses it back off itself.)

:func:`snapshot` reads all of it in one go, :func:`restore` puts back what moved
and says what it did. :func:`ensure_audible` is the other direction, for an
audio scenario's prologue: is this board in a state where a pad can be heard.

Verbs used, all in ``hil_control.cpp`` / ``audio_route_control.cpp``:

====================  =========================================================
reads                 ``MIX_LVL``, ``AUDIO_STATUS``, ``IN_PAIR``
writes (restored)     ``IN_PAIR n``, ``AUDIO_ADC c DIFF|LINE1|LINE2``,
                      ``AUDIO_MIC_SRC c``, ``AUDIO_OUT c LINE1|LINE2|ALL``,
                      ``AUDIO_VOL c 0..100`` (also saved to NVS, as the overlay
                      does), ``AUDIO_MUTE c 0|1``, ``MIX_ROUTE ch out 0|1``,
                      ``MIX_MUTE ch 0|1``, ``MIX_VOL ch pct``, ``MIX_OUT_MUTE out
                      0|1``, ``MIX_OUT_VOL out pct`` (the four saved with the desk,
                      as the Mixer's close does)
**unrestorable**      a mixer level above unity (``vol`` > 1.00: the verbs take
                      0..100 %), and the mixer's mute/level on a firmware that
                      predates ``MIX_MUTE``: those are reported, never guessed.
                      ``MIX_PRESET LOAD n`` replaces the whole desk, which is not
                      "putting back what changed".
====================  =========================================================

Every read is best-effort: a firmware without a verb (a build without the Mixer
has no ``MIX_LVL``) leaves that part of the snapshot ``None``, and a ``None`` on
either side is not a difference -- the same rule as :mod:`board_state`.
"""
from __future__ import annotations

import re
from collections.abc import Callable
from typing import Any

from crosspad_hil import verbs
from crosspad_hil.cdc import CdcLink
from crosspad_hil.errors import BAD_ARGS, CANCELLED, NOT_SUPPORTED, TIMEOUT, UNKNOWN_VERB, HilError

# AUDIO_STATUS reports these by number; the verbs take them by name
# (audio_route_control.cpp parse_adc_input_name / parse_dac_output_name).
ADC_NAMES = {0: "DIFF", 1: "LINE1", 2: "LINE2"}
DAC_NAMES = {1: "LINE1", 2: "LINE2", 3: "ALL"}

SAMPLER_CHANNEL = "Sampler"
# Codec output volumes below this make a pad inaudible at the capture; factory is 64.
MIN_OUT_VOL = 40
FACTORY_VOL = 64
VOL_TOLERANCE = 0.005         # MIX_LVL volumes are printed to two decimals

NO_MIXER_VERB = ("the firmware has no MIX_MUTE / MIX_VOL / MIX_OUT_MUTE / MIX_OUT_VOL "
                 "(it predates them)")
ABOVE_UNITY = "a level above unity has no verb (MIX_VOL / MIX_OUT_VOL take 0..100 %)"
MIXER_KINDS = ("ch_mute", "ch_vol", "out_mute", "out_vol")


def _pct(vol: Any) -> int | None:
    """``MIX_LVL vol=0.80`` as the verbs' percent (80); None above unity or unreadable."""
    try:
        v = float(vol)
    except (TypeError, ValueError):
        return None
    if v < 0.0 or v > 1.0 + VOL_TOLERANCE:
        return None
    return min(100, int(round(v * 100.0)))

_LIST = re.compile(r"(\w+)=\[([^\]]*)\]")
_SCALAR = re.compile(r"(\w+)=(-?\d+)\b")
_INPAIR = re.compile(r"pair=(-?\d+)")

Log = Callable[[str], None]


def _ints(text: str) -> list[int]:
    return [int(x) for x in text.replace(" ", "").split(",") if x.lstrip("-").isdigit()]


def parse_audio_status(line: str) -> dict[str, Any]:
    """``AUDIO: mic_src=1 adc=[2,1] out=[3,3] vol=[64,64] mute=[0,0]`` as lists of ints.

    A scalar (``vol=80``, an older firmware) becomes a one-item list.
    """
    body = line.split(":", 1)[1] if ":" in line else line
    out: dict[str, Any] = {}
    for key, items in _LIST.findall(body):
        out[key] = _ints(items)
    rest = _LIST.sub("", body)
    for key, value in _SCALAR.findall(rest):
        out.setdefault(key, [int(value)])
    if "mic_src" in out:
        out["mic_src"] = out["mic_src"][0]
    return out


def snapshot(dev: CdcLink) -> dict[str, Any]:
    """Everything below, as plain JSON-able data; a part the board will not give is None.

    ``{"mixer": {"channels": [{ch, name, vol, mute, route:[0|1..]}],
                 "outputs":  [{out, vol, mute}]} | None,
       "audio": {"mic_src": int, "adc": [..], "out": [..], "vol": [..], "mute": [..]} | None,
       "in_pair": int | None}``
    """
    snap: dict[str, Any] = {"mixer": None, "audio": None, "in_pair": None}

    def silent(e: HilError) -> bool:
        """A board that answers nothing at all (crashed, rebooting) gets asked once, not
        three times: an echo or an ERR is a firmware without the verb, a TIMEOUT is no board."""
        if e.code == CANCELLED:
            raise e
        return e.code == TIMEOUT

    try:
        desk = verbs.mix_lvl(dev)
        if desk["channels"] or desk["outputs"]:
            snap["mixer"] = {
                "channels": [{k: c.get(k) for k in ("ch", "name", "vol", "mute", "route")}
                             for c in desk["channels"]],
                "outputs": [{k: o.get(k) for k in ("out", "vol", "mute")}
                            for o in desk["outputs"]],
            }
    except HilError as e:
        if silent(e):
            return snap
    try:
        line = dev.transact("AUDIO_STATUS", "AUDIO:").line or ""
        status = parse_audio_status(line)
        if status:
            snap["audio"] = status
    except HilError as e:
        if silent(e):
            return snap
    try:
        m = _INPAIR.search(dev.transact("IN_PAIR", "INPAIR:").line or "")
        if m:
            snap["in_pair"] = int(m.group(1))
    except HilError as e:
        silent(e)
    return snap


# --- what moved ----------------------------------------------------------------

def _channel(snap: dict[str, Any], ch: int) -> dict[str, Any] | None:
    for c in ((snap.get("mixer") or {}).get("channels") or []):
        if c.get("ch") == ch:
            return c
    return None


def _same_vol(a: Any, b: Any) -> bool:
    try:
        return abs(float(a) - float(b)) <= VOL_TOLERANCE
    except (TypeError, ValueError):
        return a == b


def diff(before: dict[str, Any] | None, after: dict[str, Any] | None) -> list[dict[str, Any]]:
    """``[{key, what, before, after, fixable}]`` for every field that moved.

    ``key`` is a tuple the restore dispatches on; ``what`` is the sentence for the
    log. Parts either snapshot could not read are skipped, and a mixer channel is
    only compared against the channel of the same number *and* name.
    """
    if not before or not after:
        return []
    out: list[dict[str, Any]] = []

    def add(key: tuple, what: str, a: Any, b: Any, fixable: bool = True) -> None:
        out.append({"key": key, "what": what, "before": a, "after": b, "fixable": fixable})

    if before.get("in_pair") is not None and after.get("in_pair") is not None \
            and before["in_pair"] != after["in_pair"]:
        add(("in_pair",), "IN pair", before["in_pair"], after["in_pair"])

    ba, aa = before.get("audio") or {}, after.get("audio") or {}
    if "mic_src" in ba and "mic_src" in aa and ba["mic_src"] != aa["mic_src"]:
        add(("mic_src",), "capture source codec", ba["mic_src"], aa["mic_src"])
    for field, label in (("adc", "ADC input"), ("out", "DAC output"),
                         ("vol", "codec volume"), ("mute", "codec mute")):
        for i, (x, y) in enumerate(zip(ba.get(field) or [], aa.get(field) or [], strict=False)):
            if x != y:
                fixable = field != "out" or x in DAC_NAMES
                add((field, i), f"{label} codec {i}", x, y, fixable)

    for c0 in ((before.get("mixer") or {}).get("channels") or []):
        c1 = _channel(after, c0.get("ch", -1))
        if c1 is None or c1.get("name") != c0.get("name"):
            continue
        tag = f"mixer ch {c0['ch']} {c0.get('name')}"
        if c0.get("mute") != c1.get("mute"):
            add(("ch_mute", c0["ch"]), f"{tag} mute", c0.get("mute"), c1.get("mute"))
        if not _same_vol(c0.get("vol"), c1.get("vol")):
            add(("ch_vol", c0["ch"]), f"{tag} volume", c0.get("vol"), c1.get("vol"),
                _pct(c0.get("vol")) is not None)
        r0, r1 = c0.get("route") or [], c1.get("route") or []
        for o, (x, y) in enumerate(zip(r0, r1, strict=False)):
            if x != y:
                add(("route", c0["ch"], o), f"{tag} send to out {o}", x, y)
    outs1 = {o.get("out"): o for o in ((after.get("mixer") or {}).get("outputs") or [])}
    for o0 in ((before.get("mixer") or {}).get("outputs") or []):
        o1 = outs1.get(o0.get("out"))
        if o1 is None:
            continue
        tag = f"mixer out {o0['out']}"
        if o0.get("mute") != o1.get("mute"):
            add(("out_mute", o0["out"]), f"{tag} mute", o0.get("mute"), o1.get("mute"))
        if not _same_vol(o0.get("vol"), o1.get("vol")):
            add(("out_vol", o0["out"]), f"{tag} volume", o0.get("vol"), o1.get("vol"),
                _pct(o0.get("vol")) is not None)
    return out


# --- putting it back -------------------------------------------------------------

def _ok(dev: CdcLink, cmd: str) -> None:
    """Send an OK/ERR verb; an ERR (or an echo, a timeout) is a HilError."""
    line = (dev.transact(cmd, "OK").line or "").strip()
    if line.startswith("ERR"):
        raise HilError(BAD_ARGS, f"{cmd}: device replied {line}")


def _predates(e: HilError | KeyError | ValueError) -> bool:
    """A firmware without the verb: it echoes the line (``UNKNOWN_VERB``; TIMEOUT if the
    echo was missed) or answers ERR with no reason of the verb's own. The verbs' own
    refusals name themselves (``ERR MIX_MUTE ch 0|1``)."""
    if not isinstance(e, HilError):
        return False
    if e.code in (UNKNOWN_VERB, TIMEOUT):
        return True
    return e.code == BAD_ARGS and "MIX_" not in e.message.split("replied", 1)[-1]


def _apply(dev: CdcLink, change: dict[str, Any]) -> None:
    key, want = change["key"], change["before"]
    kind = key[0]
    if kind == "in_pair":
        # Reads back "INPAIR: ok=0 ..." on a refused pair; the write itself is the query form.
        line = dev.transact(f"IN_PAIR {int(want)}", "INPAIR:").line or ""
        if "ok=0" in line:
            raise HilError(BAD_ARGS, f"IN_PAIR {want}: device replied {line.strip()}")
    elif kind == "mic_src":
        _ok(dev, f"AUDIO_MIC_SRC {int(want)}")
    elif kind == "adc":
        _ok(dev, f"AUDIO_ADC {key[1]} {ADC_NAMES[int(want)]}")
    elif kind == "out":
        _ok(dev, f"AUDIO_OUT {key[1]} {DAC_NAMES[int(want)]}")
    elif kind == "vol":
        _ok(dev, f"AUDIO_VOL {key[1]} {int(want)}")
    elif kind == "mute":
        _ok(dev, f"AUDIO_MUTE {key[1]} {int(want)}")
    elif kind == "route":
        _ok(dev, f"MIX_ROUTE {key[1]} {key[2]} {int(want)}")
    elif kind == "ch_mute":
        _ok(dev, f"MIX_MUTE {key[1]} {int(want)}")
    elif kind == "ch_vol":
        _ok(dev, f"MIX_VOL {key[1]} {_pct(want)}")
    elif kind == "out_mute":
        _ok(dev, f"MIX_OUT_MUTE {key[1]} {int(want)}")
    elif kind == "out_vol":
        _ok(dev, f"MIX_OUT_VOL {key[1]} {_pct(want)}")
    else:
        raise HilError(NOT_SUPPORTED, f"no verb puts back {kind}")


# Inputs before the capture source (the order audio_route_apply_preset uses:
# switching the source first streams one buffer of the old input); IN_PAIR first
# because choosing a pair moves the ADCs under it.
_ORDER = ("in_pair", "adc", "mic_src", "out", "vol", "mute", "route", *MIXER_KINDS)


def restore(dev: CdcLink, snap: dict[str, Any] | None, log: Log | None = None) -> dict[str, Any]:
    """Put the audio state back to ``snap``; say what moved and what could not be put back.

    Returns ``{"restored": [str], "unrestorable": [str], "failed": [str],
    "unread": bool}``. Each entry reads like ``"codec volume codec 0: 28 -> 64"``
    (current -> wanted). Every difference is also logged. Never raises on a verb
    that fails (a restore runs in a ``finally`` and must not mask the run's own
    error); a cancelled run still restores.
    """
    say = log or (lambda _s: None)
    result: dict[str, Any] = {"restored": [], "unrestorable": [], "failed": [], "unread": False}
    if not snap:
        result["unread"] = True
        return result
    try:
        now = snapshot(dev)
    except HilError as e:
        say(f"audio_state: could not read the board to compare: {e.code}: {e.message}")
        result["unread"] = True
        return result
    changes = diff(snap, now)
    if not changes:
        return result

    def sort_key(c: dict[str, Any]) -> int:
        return _ORDER.index(c["key"][0]) if c["key"][0] in _ORDER else len(_ORDER)

    mixer_verbs: bool | None = None       # False once the board shows it has no MIX_MUTE
    for change in sorted(changes, key=sort_key):
        line = f"{change['what']}: {change['after']} -> {change['before']}"
        if not change["fixable"]:
            result["unrestorable"].append(line)
            say(f"WARNING: audio_state: {line} NOT restored -- {ABOVE_UNITY}; "
                "put it back in the Mixer app")
            continue
        if change["key"][0] in MIXER_KINDS and mixer_verbs is False:
            result["unrestorable"].append(line)
            continue                       # said once below, when the first one was refused
        try:
            _apply(dev, change)
        except (HilError, KeyError, ValueError) as e:
            if change["key"][0] in MIXER_KINDS and _predates(e):
                mixer_verbs = False
                result["unrestorable"].append(line)
                say(f"WARNING: audio_state: {line} NOT restored -- {NO_MIXER_VERB}; "
                    "put it back in the Mixer app")
                continue
            result["failed"].append(line)
            say(f"WARNING: audio_state: {line} could not be restored: {e}")
            continue
        result["restored"].append(line)
        say(f"audio_state: restored {line}")

    if result["restored"]:
        # Believe the board, not the OKs: an ERR-free verb that the firmware then overrode
        # (an ADC claim holds LINE2 while the Recorder listens) did not put anything back.
        try:
            still = {c["key"] for c in diff(snap, snapshot(dev)) if c["fixable"]}
        except HilError:
            still = set()
        for change in changes:
            if change["key"] in still and change["fixable"]:
                line = f"{change['what']}: {change['after']} -> {change['before']}"
                if line in result["unrestorable"]:
                    continue                  # already said: no verb for it
                if line in result["restored"]:
                    result["restored"].remove(line)
                if line not in result["failed"]:
                    result["failed"].append(line)
                    say(f"WARNING: audio_state: {line} did not stick")
    return result


def put_back(dev: CdcLink | None, snap: dict[str, Any] | None, log: Log | None = None,
             data: dict[str, Any] | None = None) -> dict[str, Any] | None:
    """:func:`restore` for a scenario's ``finally``: never raises, records into ``data``.

    ``data["audio_state_restored"]`` gets the restore result, and
    ``data["audio_state_unrestorable"]`` / ``["audio_state_restore_failed"]`` what
    was left changed, so a report shows it. ``dev`` or ``snap`` None (the run never
    got far enough to read) is a no-op.
    """
    if dev is None or not snap:
        return None
    say = log or (lambda _s: None)
    try:
        result = restore(dev, snap, say)
    except HilError as e:
        say(f"audio_state: left as found: {e.code}: {e.message}")
        return None
    if data is not None:
        data["audio_state_restored"] = result["restored"]
        data["audio_state_unrestorable"] = result["unrestorable"]
        data["audio_state_restore_failed"] = result["failed"]
    return result


# --- is a pad audible --------------------------------------------------------------

def problems(snap: dict[str, Any], channel: str = SAMPLER_CHANNEL,
             min_out_vol: int = MIN_OUT_VOL) -> list[dict[str, Any]]:
    """What in ``snap`` keeps ``channel``'s pads from being heard: ``[{text, cmd}]``.

    ``cmd`` is the verb that fixes it, None when there is none (a muted channel).
    A part the snapshot could not read, or a desk with no ``channel``, says nothing:
    that is not a misconfiguration, the take's own level check judges it.
    """
    found: list[dict[str, Any]] = []
    ch = next((c for c in ((snap.get("mixer") or {}).get("channels") or [])
               if str(c.get("name")).lower() == channel.lower()), None)
    if ch is not None:
        if ch.get("mute") == 1:
            found.append({"text": f"mixer channel {ch['ch']} '{channel}' was muted (mute=1, "
                                  f"saved across reboots); unmuted",
                          "cmd": f"MIX_MUTE {ch['ch']} 0"})
        route = list(ch.get("route") or [])
        if route and not any(route):
            for o in range(len(route)):
                found.append({"text": f"mixer channel {ch['ch']} '{channel}' has no send; "
                                      f"send to out {o} switched on",
                              "cmd": f"MIX_ROUTE {ch['ch']} {o} 1"})
    for codec, vol in enumerate((snap.get("audio") or {}).get("vol") or []):
        if vol < min_out_vol:
            found.append({"text": f"codec {codec} output volume {vol} (< {min_out_vol}) raised "
                                  f"to {FACTORY_VOL}", "cmd": f"AUDIO_VOL {codec} {FACTORY_VOL}"})
    return found


def ensure_audible(dev: CdcLink, log: Log | None = None, *, channel: str = SAMPLER_CHANNEL,
                   min_out_vol: int = MIN_OUT_VOL,
                   snap: dict[str, Any] | None = None) -> dict[str, Any]:
    """Fix what keeps ``channel``'s pads from being heard, as far as the verbs reach.

    Checks the channel's mute and sends and the codec output volumes
    (``AUDIO_STATUS vol=``), unmutes the channel (``MIX_MUTE``), fixes the sends
    (``MIX_ROUTE``, every output, as the factory desk has them) and raises a codec
    volume below ``min_out_vol`` to ``FACTORY_VOL`` (``AUDIO_VOL``). A verb the board
    refuses (a firmware without ``MIX_MUTE``) leaves that problem unfixable. The board
    is read again afterwards, and anything still wrong is unfixable.

    Returns ``{"fixed": [str], "unfixable": [str], "unread": [str]}``. ``unread``
    names what the board would not say (firmware without ``MIX_LVL``): not a
    misconfiguration, the caller measures and lets the verdict judge.
    """
    say = log or (lambda _s: None)
    out: dict[str, Any] = {"fixed": [], "unfixable": [], "unread": []}
    snap = snap if snap is not None else snapshot(dev)
    if snap.get("mixer") is None:
        out["unread"].append("mixer (MIX_LVL)")
    if snap.get("audio") is None:
        out["unread"].append("codec volumes (AUDIO_STATUS)")

    fixed_any = False
    for p in problems(snap, channel, min_out_vol):
        if p["cmd"] is None:
            out["unfixable"].append(p["text"])
            say(f"WARNING: audio_state: {p['text']}")
            continue
        try:
            _ok(dev, p["cmd"])
        except HilError as e:
            out["unfixable"].append(f"{p['text']} ({p['cmd']}: {e.message})")
            say(f"WARNING: audio_state: could not fix: {p['text']} ({p['cmd']}: {e.message})")
            continue
        fixed_any = True
        out["fixed"].append(p["text"])
        say(f"audio_state: fixed: {p['text']}")
    if fixed_any:
        for p in problems(snapshot(dev), channel, min_out_vol):
            if p["cmd"] is not None:
                if p["text"] in out["fixed"]:
                    out["fixed"].remove(p["text"])
                out["unfixable"].append(f"{p['text']} (still wrong after {p['cmd']})")
                say(f"WARNING: audio_state: still wrong after {p['cmd']}: {p['text']}")
    return out
