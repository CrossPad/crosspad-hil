"""Console-line and CDC-reply parsers.

Pure functions plus one small stateful class; no I/O, no threads, no global state.
Console rules are ported from platform-idf/tools/hil_smoke.py, hil_stability.py and
hil_kit_churn.py; the CDC reply grammar is platform-idf/main/hil_control.cpp — each parser
below quotes the firmware printf format it decodes.
"""
from __future__ import annotations

import copy
import re
import shlex
from dataclasses import dataclass, field
from typing import Any

from crosspad_hil.knowledge import load


@dataclass
class ConsoleEvent:
    kind: str
    seq: int
    line: str
    data: dict[str, Any] = field(default_factory=dict)


class ConsoleParser:
    """Feed console lines one at a time; get typed events back; ask for a summary.

    Boot-scoped state (markers, pending required set, boot_complete) is cleared by
    reset_boot_tracking(); session counters (fatals, errors, reboots, heap, kit requests,
    cdc drops) accumulate for the life of the parser.
    """

    def __init__(self, knowledge: dict | None = None, required: list[str] | None = None) -> None:
        k = knowledge if knowledge is not None else load("markers")
        # hil_smoke.BOOT_MARKERS / REQUIRED_MARKERS / OPTIONAL_MARKERS
        self._boot_markers: list[str] = list(k["boot_markers"])
        self._required: list[str] = list(required) if required is not None else list(k["required"])
        self._optional: list[str] = list(k.get("optional", []))
        # hil_smoke: re.match(r"^E \(\d+\)", line) and ERROR_ALLOWLIST substrings
        self._error_re = re.compile(k["error_line"])
        self._error_allow: list[str] = list(k.get("error_allow", []))
        # hil_stability.FATAL_PATTERNS ∪ hil_kit_churn.FATAL_RE
        self._fatal_res: list[re.Pattern[str]] = [re.compile(p) for p in k["fatal_patterns"]]
        # hil_stability.RST_REASON_RE
        self._reset_re = re.compile(k["reset_reason"])
        # hil_stability / hil_kit_churn: "Heap Statistics" … HEAP_FREE_RE … "Total tasks:"
        heap = k["heap_block"]
        self._heap_start: str = heap["start"]
        self._heap_line = re.compile(heap["line"])
        self._heap_end: str = heap["end"]
        # hil_kit_churn.KIT_REQ_RE / CDC_DROP_RE
        self._kit_re = re.compile(k["kit_request"])
        self._drops_re = re.compile(k["cdc_drops"])
        # hil_kit_churn: "ESP-ROM:esp32s3" in line → reboots += 1
        self._reboot: str = k["reboot"]

        self._markers_seen: dict[str, int] = {}
        self._pending: set[str] = set(self._required)
        self._boot_complete = False
        self._reset_armed = True
        self._heap_block: list[int] = []
        self._in_heap_block = False

        self._reboots = 0
        self._reset_reasons: list[str] = []
        self._fatals: list[dict[str, Any]] = []
        self._errors: list[dict[str, Any]] = []
        self._heap: dict[int, int] = {}
        self._kit_requests: list[dict[str, Any]] = []
        self._cdc_drops = 0

    def reset_boot_tracking(self) -> None:
        self._markers_seen = {}
        self._pending = set(self._required)
        self._boot_complete = False
        self._reset_armed = True
        self._heap_block = []
        self._in_heap_block = False

    def feed(self, seq: int, line: str) -> list[ConsoleEvent]:
        events: list[ConsoleEvent] = []
        if not line:
            return events

        # Reboot banner (hil_kit_churn) — also re-arms the per-boot pending set (hil_stability).
        if self._reboot in line:
            self._reboots += 1
            self._pending = set(self._required)
            self._boot_complete = False
            events.append(ConsoleEvent("reboot", seq, line, {"count": self._reboots}))
        # Boot-loop counting (hil_smoke.evaluate): every boot marker occurrence is counted.
        for marker in self._boot_markers:
            if marker in line:
                self._markers_seen[marker] = self._markers_seen.get(marker, 0) + 1
                if marker == "ESP-ROM:":
                    self._reset_armed = True

        # Reset reason: first "rst:" line after a banner (hil_stability last_reset_reason rule).
        if self._reset_armed:
            m = self._reset_re.search(line)
            if m:
                self._reset_armed = False
                code, name = m.group(1), m.group(2)
                self._reset_reasons.append(f"{code} ({name})")
                events.append(ConsoleEvent("reset_reason", seq, line, {"code": code, "name": name}))

        # Required / optional markers (substring match, hil_smoke).
        for marker in self._required + self._optional:
            if marker in line:
                self._markers_seen[marker] = self._markers_seen.get(marker, 0) + 1
                events.append(ConsoleEvent("boot_marker", seq, line, {"marker": marker}))
                self._pending.discard(marker)
        if self._pending == set() and not self._boot_complete:
            self._boot_complete = True
            events.append(ConsoleEvent("boot_complete", seq, line, {"missing": []}))

        # Fatal: first pattern wins (hil_stability breaks after the first hit).
        for rx in self._fatal_res:
            if rx.search(line):
                self._fatals.append({"seq": seq, "pattern": rx.pattern, "line": line})
                events.append(ConsoleEvent("fatal", seq, line, {"pattern": rx.pattern}))
                break

        # E-level line outside the allowlist (hil_smoke).
        if self._error_re.match(line) and not any(a in line for a in self._error_allow):
            self._errors.append({"seq": seq, "line": line})
            events.append(ConsoleEvent("error_line", seq, line, {}))

        # PerfMon heap block: "Heap Statistics" opens, Free: lines collect, "Total tasks:" closes.
        hm = self._heap_line.search(line)
        if hm:
            if self._in_heap_block:
                self._heap_block.append(int(hm.group(1)))
        elif self._heap_start in line:
            self._heap_block = []
            self._in_heap_block = True
        elif self._heap_end in line and self._in_heap_block:
            for slot, free in enumerate(self._heap_block):
                self._heap[slot] = free
                events.append(ConsoleEvent("heap", seq, line, {"slot": slot, "free": free}))
            self._heap_block = []
            self._in_heap_block = False

        # Kit requests the firmware actually saw (hil_kit_churn KIT_REQ_RE).
        km = self._kit_re.search(line)
        if km:
            kit, state = int(km.group(1)), km.group(2)
            self._kit_requests.append({"kit": kit, "state": state, "seq": seq})
            events.append(ConsoleEvent("kit", seq, line, {"kit": kit, "state": state}))

        # Cumulative CDC drop counter (hil_kit_churn CDC_DROP_RE; main.cpp prints a running total).
        dm = self._drops_re.search(line)
        if dm:
            self._cdc_drops = int(dm.group(1))
            events.append(ConsoleEvent("cdc_drops", seq, line, {"dropped": self._cdc_drops}))

        return events

    def forget_reboots(self) -> None:
        """Zero the reboot/reset-reason counters (a caller-induced reset)."""
        self._reboots = 0
        self._reset_reasons.clear()
        self._markers_seen = {}

    def snapshot(self) -> dict:
        boots = max((self._markers_seen.get(m, 0) for m in self._boot_markers), default=0)
        return {
            "fatals": copy.deepcopy(self._fatals),
            "reboots": self._reboots,
            "reset_reasons": list(self._reset_reasons),
            "errors": copy.deepcopy(self._errors),
            "markers_seen": dict(self._markers_seen),
            "boot_complete": self._boot_complete,
            "missing_markers": [m for m in self._required if m in self._pending],
            "bootloops": max(0, boots - 1),
            "heap": dict(self._heap),
            "kit_requests": copy.deepcopy(self._kit_requests),
            "cdc_drops": self._cdc_drops,
        }


_INT_RE = re.compile(r"-?\d+$")
# Underscores and digits: WIFI_AP:, WIFI_NET:, AUDIO_RATE: ... -- [A-Z]+ left every such reply
# unknown.
_PREFIX_RE = re.compile(r"^([A-Z][A-Z0-9_]*):\s?(.*)$")
# WIFI_STATUS puts the SSID last before rssi=, quoted; greedy, so a quote inside the name survives.
_WIFI_SSID_RE = re.compile(r'ssid="(.*)" rssi=')
# The escapes wifi_verbs.cpp escape_ssid() writes: backslash + one of \ " | r n, or \xHH.
_SSID_ESCAPE_RE = re.compile(r"\\(x[0-9a-fA-F]{2}|.)")
_SSID_ESCAPES = {"r": "\r", "n": "\n"}
_FAMILY_ERR_RE = re.compile(r"^([A-Z][A-Z0-9]*)_ERR(?:OR)?(?:\s+(.*))?$")
_KITSTATUS_RE = re.compile(r"current=(-?\d+)\s+loading=(\d+)\s+pending=(-?\d+)\s+name=(.*)$")
# Newer firmware appends fields after running= ("running=Sampler selector=3").
_APPS_RE = re.compile(r"^(.*?)\s*running=(\S*)(?:\s+(.*))?$")
_BUCKET_RE = re.compile(r"<=(\S+)\s+u=(\d+)/(\d+)B\s+f=(\d+)/(\d+)B")
_MEMBIG_RE = re.compile(r"@([0-9a-fA-F]+)=(\d+)")
_ENCFOCUS_RE = re.compile(r"obj=(\S+)\s+idx=(-?\d+)(?:\s+edit=([01]))?\s+text=(.*)$")
_AUDIOLVL_RE = re.compile(r"L=(-?[\d.]+)\s+R=(-?[\d.]+)\s+amp=(\d)\s+allowed=(\d)")
_SMPLPEAK_RE = re.compile(r"(-?\d+)\s+free=(-?\d+)")  # hil_kit_churn parse_peak
# "PITCHED: active=%d zones=%u roots=%s voices=%u/%u bank_kb=%u rt_max_us=%u steals=%u [err=why]"
# err= appears only after a failed load; the device already replaced its spaces
# with underscores, so it is one \S+ field.
_PITCHED_RE = re.compile(
    r"active=(\d)\s+zones=(\d+)\s+roots=(\S*)\s+voices=(\d+)/(\d+)\s+"
    r"bank_kb=(\d+)\s+rt_max_us=(\d+)\s+steals=(\d+)(?:\s+err=(\S+))?"
)
_APPVER_RE = re.compile(r"^(\S+)\s+id=(\S+)\s+commit=(\S+)\s+ref=(\S+)\s+dirty=(\d)$")
_APPVER_END_RE = re.compile(r"^end count=(\d+)$")
_PADINFO_SLOT_RE = re.compile(r"\[(\d+)\]\s+'(.*)'\s+(\d+)\.\.(\d+)")
# "PADMIX: kit=%d max=%d loading=%d name=%s" / "PADMIX: pad=%d vol=%d pan=%d sample=%s" /
# "PADMIX: end count=16" (hil_control.cpp PAD_MIX); name and sample run to the end of the line.
_PADMIX_HEAD_RE = re.compile(r"^kit=(-?\d+)\s+max=(\d+)\s+loading=(\d)\s+name=(.*)$")
_PADMIX_PAD_RE = re.compile(r"^pad=(\d+)\s+vol=(-?\d+)\s+pan=(-?\d+)\s+sample=(.*)$")
_PADMIX_END_RE = re.compile(r"^end count=(\d+)$")


def _scalar(value: str) -> Any:
    """key=value scalar rule: decimal → int, "-" → None, else the string itself."""
    if _INT_RE.match(value):
        return int(value)
    if value == "-":
        return None
    return value


def _kv(body: str) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for tok in body.split():
        key, sep, value = tok.partition("=")
        if sep:
            out[key] = _scalar(value)
    return out


def _dash_none(value: str) -> str | None:
    return None if value == "-" else value


def _parse_fwupd(body: str) -> dict:
    # "FWUPD: state=%s running=%s kit=%s latest=%s channel=%s progress=%u last_check=%lld
    #  err=%s stack_free=%u inst_stack_free=%u"
    # (platform-idf main/fw_updater_verbs.cpp); "-" reads as None.
    return {"kind": "fwupd", **_kv(body)}


def _parse_fwcat(body: str) -> dict:
    # "FWCAT: count=%u total=%u" + " | <kit> <version> <channel> <tier> <on_card 0|1>"
    parts = [p.strip() for p in body.split("|")]
    head = _kv(parts[0]) if parts else {}
    rows: list[dict[str, Any]] = []
    for part in parts[1:]:
        toks = part.split()
        if len(toks) != 5 or toks[4] not in ("0", "1"):
            continue
        rows.append({"kit": toks[0], "version": toks[1], "channel": toks[2], "tier": toks[3],
                     "on_card": toks[4] == "1"})
    count = head.get("count")
    count = count if isinstance(count, int) else len(rows)
    total = head.get("total")
    return {"kind": "fwcat", "count": count, "total": total if isinstance(total, int) else count,
            "versions": rows}


def _parse_kitstatus(body: str) -> dict | None:
    # "KITSTATUS: current=%d loading=%d pending=%d name=%s"
    m = _KITSTATUS_RE.search(body)
    if not m:
        return None
    return {
        "kind": "kitstatus",
        "current": int(m.group(1)),
        "loading": m.group(2) != "0",
        "pending": int(m.group(3)),
        "name": _dash_none(m.group(4).strip()),
    }


def _parse_apps(body: str) -> dict | None:
    # "APPS: " + names joined by "," + " running=%s"
    m = _APPS_RE.match(body)
    if not m:
        return None
    apps = [a for a in m.group(1).split(",") if a]
    out = {"kind": "apps", "apps": apps, "running": _dash_none(m.group(2))}
    if m.group(3):
        out["extra"] = _kv(m.group(3))
    return out


_KITS_END_RE = re.compile(r"^end count=(\d+) current=(-?\d+)$")


def _parse_kits(body: str) -> dict | None:
    # One "KITS: <id>:<name>,<id>:<name>…" per batch, then
    # "KITS: end count=N current=M" (hil_control.cpp handle_kit_list).
    end = _KITS_END_RE.match(body)
    if end:
        return {"kind": "kits", "end": True,
                "count": int(end.group(1)), "current": int(end.group(2))}
    kits: list[dict[str, Any]] = []
    for item in body.split(","):
        kit_id, sep, name = item.strip().partition(":")
        if not sep or not _INT_RE.match(kit_id):
            continue
        kits.append({"id": int(kit_id), "name": name})
    if not kits:
        return None
    return {"kind": "kits", "kits": kits}


_KITSWAP_END_RE = re.compile(r"^end count=(\d+)$")
_KITSWAP_MASK_KEYS = ("deferred", "forced")


def _parse_kitswap(body: str) -> dict | None:
    # One row per swap, oldest first (the device keeps a ring of the last 8):
    #   "KITSWAP: seq=N kit=I ok=0|1 req_ms=N parse_ms=N done_ms=N gated_blocks=N
    #    gate_max_us=N gate_total_us=N gate_count=N deferred=0xMMMM forced=0xMMMM
    #    dropped_hits=N pads=a,b,…,p"
    # then "KITSWAP: end count=N".
    #
    # The two masks arrive as 0x-prefixed hex, which the shared key=value scalar
    # rule leaves as a string (it only promotes decimals) — a caller comparing
    # `deferred & (1 << pad)` would silently be doing string work, so they are
    # converted here, and the raw text is kept beside them for the report.
    end = _KITSWAP_END_RE.match(body)
    if end:
        return {"kind": "kitswap", "end": True, "count": int(end.group(1))}
    out: dict[str, Any] = {"kind": "kitswap", "end": False, **_kv(body)}
    if "seq" not in out or "kit" not in out:
        return None
    for key in _KITSWAP_MASK_KEYS:
        raw = out.get(key)
        if isinstance(raw, str):
            try:
                out[key] = int(raw, 16)
            except ValueError:
                return None
            out[f"{key}_hex"] = raw
        elif isinstance(raw, int):
            out[f"{key}_hex"] = f"0x{raw:04X}"
    pads_raw = out.get("pads")
    if isinstance(pads_raw, str):
        try:
            out["pads"] = [int(v) for v in pads_raw.split(",") if v != ""]
        except ValueError:
            return None
    elif isinstance(pads_raw, int):
        out["pads"] = [pads_raw]
    if "ok" in out:
        out["ok"] = bool(out["ok"])
    return out


def _parse_padstats(body: str) -> dict:
    # "PADSTATS: press=%u release=%u played=%u freeslots=%d"
    return {"kind": "padstats", **_kv(body)}


def _parse_padnotes(body: str) -> dict:
    # "PADNOTES:" + " %u:%u" per pad
    notes: dict[int, int] = {}
    for tok in body.split():
        pad, sep, note = tok.partition(":")
        if sep and _INT_RE.match(pad) and _INT_RE.match(note):
            notes[int(pad)] = int(note)
    return {"kind": "padnotes", "notes": notes}


def _parse_leds(body: str) -> dict:
    # "LEDS: bri=%u anim=%d coalesce=%d cfgbri=%u pwr=0x%02X pwrN=%u txfail=%u colors=RRGGBB,…"
    raw: dict[str, str] = {}
    for tok in body.split():
        key, sep, value = tok.partition("=")
        if sep:
            raw[key] = value
    colors = [c for c in raw.get("colors", "").split(",") if c]
    return {
        "kind": "leds",
        "brightness": int(raw.get("bri", "0")),
        "anim": raw.get("anim", "0") != "0",
        "coalesce": raw.get("coalesce", "0") != "0",
        "cfgbri": int(raw.get("cfgbri", "0")),
        "pwr": int(raw.get("pwr", "0"), 16),
        "pwr_count": int(raw.get("pwrN", "0")),
        "txfail": int(raw.get("txfail", "0")),
        "colors": colors,
    }


def _parse_mem(body: str) -> dict:
    # "MEM: int_free=%u int_largest=%u int_min=%u int_blocks=%u psram_free=%u psram_largest=%u
    #  psram_blocks=%u"
    return {"kind": "mem", **_kv(body)}


def _parse_clock(body: str) -> dict:
    # "CLOCK: src=host|int bpm=%.2f running=%d ticks_in=%u ticks_out=%u last_gap_us=%d
    #  max_jitter_us=%d max_late_us=%d"
    raw = _kv(body)
    return {"kind": "clock", **raw, "bpm": float(raw.get("bpm", 0)),
            "running": raw.get("running") == 1}


def _parse_memblk(body: str) -> dict:
    # "MEMBLK: biggest_used=%u" + " | <=%s u=%u/%uB f=%u/%uB" per bucket
    parts = [p.strip() for p in body.split("|")]
    head = _kv(parts[0]) if parts else {}
    buckets: list[dict[str, Any]] = []
    for part in parts[1:]:
        m = _BUCKET_RE.search(part)
        if m:
            buckets.append(
                {
                    "le": m.group(1),
                    "used_n": int(m.group(2)),
                    "used_b": int(m.group(3)),
                    "free_n": int(m.group(4)),
                    "free_b": int(m.group(5)),
                }
            )
    return {"kind": "memblk", "biggest_used": int(head.get("biggest_used", 0)), "buckets": buckets}


def _parse_membig(body: str) -> dict:
    # "MEMBIG:" + " @%08x=%u" per block >= 8 kB
    blocks = [{"addr": int(a, 16), "size": int(s)} for a, s in _MEMBIG_RE.findall(body)]
    return {"kind": "membig", "blocks": blocks}


def _parse_wave(body: str) -> dict | None:
    # "WAVE: kit=%d total=%d cached=%d loader=%d queued=%u req=%u ok=%u retry=%u
    #        fail_info=%u fail_read=%u alloc_fail=%u drop=%u missing=p:l,p:l|-"
    kv = _kv(body)
    if "total" not in kv or "cached" not in kv:
        return None
    missing: list[tuple[int, int]] = []
    raw = str(kv.get("missing", "-"))
    if raw != "-":
        for tok in raw.split(","):
            pad, _, layer = tok.partition(":")
            if pad.isdigit() and layer.isdigit():
                missing.append((int(pad), int(layer)))
    out = {"kind": "wave", "missing": missing}
    for key in ("kit", "total", "cached", "loader", "queued", "req", "ok", "retry",
                "fail_info", "fail_read", "alloc_fail", "drop"):
        out[key] = int(kv.get(key, 0))
    out["loader"] = bool(out["loader"])
    return out


def _parse_cdcstats(body: str) -> dict:
    # "CDCSTATS: rx=%u cmds=%u drop=%u"
    return {"kind": "cdcstats", **_kv(body)}


def _parse_lvglstats(body: str) -> dict:
    # "LVGLSTATS: refr=%u refr_max_us=%u refr_avg_us=%u render_us=%u render_max_us=%u
    #  flush_wait_us=%u flush_wait_max_us=%u flushes=%u" -- read-and-clear on the device
    return {"kind": "lvglstats", **_kv(body)}


def _parse_ble(body: str) -> dict:
    # "BLE: supported=%d running=%d state=%s mode=%s self=%s peer=%s itvl=%u txoff=%d rxoff=%d …"
    return {"kind": "ble", **_kv(body)}


def _parse_bledev(body: str) -> dict:
    # "BLEDEV: count=%u" + " | %s %s %d" (addr, name with spaces, rssi)
    parts = [p.strip() for p in body.split("|")]
    head = _kv(parts[0]) if parts else {}
    devices: list[dict[str, Any]] = []
    for part in parts[1:]:
        toks = part.split()
        if len(toks) < 2 or not _INT_RE.match(toks[-1]):
            continue
        devices.append({"addr": toks[0], "name": " ".join(toks[1:-1]), "rssi": int(toks[-1])})
    return {"kind": "bledev", "count": int(head.get("count", 0)), "devices": devices}


def _unescape_ssid(text: str) -> str:
    def one(m: re.Match[str]) -> str:
        code = m.group(1)
        if len(code) == 3:
            return chr(int(code[1:], 16))
        return _SSID_ESCAPES.get(code, code)
    return _SSID_ESCAPE_RE.sub(one, text)


def _parse_wifi(body: str) -> dict:
    # 'WIFI: state=%s ssid="%s" rssi=%d ip=%s leases=%u fail=%s reason=%u heap_int=%u heap_init=%u
    #  heap_low=%u … stack_free=%u enabled=%u' -- the SSID is escaped (see _unescape_ssid);
    # none is ssid=""
    m = _WIFI_SSID_RE.search(body)
    rest = _WIFI_SSID_RE.sub("rssi=", body, count=1) if m else body
    return {"kind": "wifi", **_kv(rest),
            "ssid": (_unescape_ssid(m.group(1)) or None) if m else None}


def _parse_wifi_ap(body: str) -> dict:
    # "WIFI_AP: count=%u" + " | <rssi> <auth> <escaped ssid>" per network. A literal "|" in an
    # SSID arrives as "\|", so " | " only ever separates rows.
    parts = body.split(" | ")
    head = _kv(parts[0]) if parts else {}
    aps: list[dict[str, Any]] = []
    for part in parts[1:]:
        toks = part.split(" ", 2)
        if len(toks) < 2 or not _INT_RE.match(toks[0]):
            continue
        aps.append({"rssi": int(toks[0]), "auth": toks[1],
                    "ssid": _unescape_ssid(toks[2]) if len(toks) > 2 else ""})
    return {"kind": "wifi_ap", "count": int(head.get("count", 0)), "aps": aps}


def _parse_wifi_net(body: str) -> dict:
    # "WIFI_NET: count=%u" + " | <escaped ssid>" per saved network, in connect order
    parts = body.split(" | ")
    head = _kv(parts[0]) if parts else {}
    return {"kind": "wifi_net", "count": int(head.get("count", 0)),
            "nets": [_unescape_ssid(p) for p in parts[1:]]}


def _parse_wifi_ping(body: str) -> dict:
    # "WIFI_PING: ok ms=%u status=%d"
    toks = body.split(" ", 1)
    return {"kind": "wifi_ping", "ok": toks[0] == "ok", **_kv(toks[1] if len(toks) > 1 else "")}


def _parse_net_mode(body: str) -> dict:
    # "NET_MODE: active=0|1 tab=wifi|store|updates|-"
    return {"kind": "net_mode", **_kv(body)}


def _parse_ui(body: str) -> dict:
    # "UI: display=%s touch=%s drawer=%d lcd=%d rgb=%d theme=%u bt_icon=%s app=%s"
    return {"kind": "ui", **_kv(body)}


def _parse_daw(body: str) -> dict | None:
    # "DAW: running=%d connected=%d daw=%u mode=%u transport=%u beat=%u beats=%lu
    #      row0="%s" row1="%s" row2="%s"" — the rows are quoted because a DAW
    # sends channel and pattern names with spaces in them.
    try:
        toks = shlex.split(body)
    except ValueError:
        return None
    out: dict[str, Any] = {"kind": "daw"}
    for tok in toks:
        key, sep, value = tok.partition("=")
        if sep:
            out[key] = value if key.startswith("row") else _scalar(value)
    if "running" not in out:
        return None
    out["running"] = out["running"] == 1
    out["connected"] = out.get("connected") == 1
    out["rows"] = [out.pop(f"row{i}", "") for i in range(3)]
    return out


def _parse_enc(body: str) -> dict:
    # "ENC: group=%p launcher=%p owner=%s"
    return {"kind": "enc", **_kv(body)}


def _parse_encfocus(body: str) -> dict | None:
    # "ENCFOCUS: obj=%p idx=%d edit=%d text=%s"; edit= is absent before 2026-10-06
    m = _ENCFOCUS_RE.search(body)
    if not m:
        return None
    out: dict[str, Any] = {"kind": "encfocus", "index": int(m.group(2)), "label": m.group(4),
                           "ptr": m.group(1)}
    if m.group(3) is not None:
        out["editing"] = m.group(3) == "1"
    return out


def _parse_encgroup_head(body: str) -> dict | None:
    # "ENCGROUP: count=%u" — the indexed lines that follow are parse_enc_group's job
    head = _kv(body)
    if "count" not in head:
        return None
    return {"kind": "encgroup", "count": int(head["count"])}


_AUDIORATE_RE = re.compile(r"running=(\d+)\s+configured=(\d+)")


def _parse_audiorate(body: str) -> dict | None:
    # "AUDIO_RATE: running=%lu configured=%lu"
    m = _AUDIORATE_RE.search(body)
    if not m:
        return None
    return {"kind": "audiorate", "running": int(m.group(1)), "configured": int(m.group(2))}


def _parse_audiolvl(body: str) -> dict | None:
    # "AUDIOLVL: L=%.5f R=%.5f amp=%d allowed=%d"
    m = _AUDIOLVL_RE.search(body)
    if not m:
        return None
    return {
        "kind": "audiolvl",
        "left": float(m.group(1)),
        "right": float(m.group(2)),
        "amp": m.group(3) != "0",
        "allowed": m.group(4) != "0",
    }


def _parse_smplpeak(body: str) -> dict | None:
    # "SMPLPEAK: %d free=%d poly=%u rt_max_us=%u rxskip=%u noblock=%u underrun=%u
    #  unpaced=%u rt_busy_max_us=%u overruns=%u" (platform-idf hil_control.cpp,
    # the native engine's merge, 2026-10). Only peak and free are required: older
    # firmware ends earlier, and the prebuilt engine's load_main=/load_bknd=/
    # maxc_main=/maxc_bknd= (always nan) are gone -- whatever key=value follows is
    # kept as it comes.
    m = _SMPLPEAK_RE.search(body)
    if not m:
        return None
    extra = {k: v for k, v in _kv(body).items() if k not in ("free",)}
    return {"kind": "smplpeak", **extra, "peak": int(m.group(1)), "free": int(m.group(2))}


def _split_pair(value: Any) -> tuple[int, int] | None:
    """"a/b" of two ints, or None."""
    a, sep, b = str(value).partition("/")
    if sep and _INT_RE.match(a) and _INT_RE.match(b):
        return int(a), int(b)
    return None


def _parse_smpleng(body: str) -> dict:
    # "SMPLENG: engine=native|marcel|none state=%u why=%s kit=%d pitch=%d,%d,%d slots=%u
    #  complete=%u head_kb=%u heads_kb=%u pool_kb=%u pool_blocks=%u/%u peak=%u
    #  voices=%u/%u lim=%u/%u notes=%u drop=%u steal=%u norings=%u underrun=%u
    #  quiet=%u rt_max_us=%u noteon_max_us=%u read_kb=%u bg_stack_free=%u
    #  preview=%u prev_pk=%d engines=native[,marcel]"
    # (kit=, bg_stack_free=, preview= and prev_pk= from the 2026-10-09 merge fix;
    # an older line lacks them and the keys are simply absent.)
    # (platform-idf SamplerEnginePort.cpp native_sampler_status). The pairs are
    # split into <key> and <key>_max / the named halves; why "-" reads as None.
    kv = _kv(body)
    out: dict[str, Any] = {"kind": "smpleng", **kv}
    for key, lo, hi in (("pool_blocks", "pool_blocks", "pool_blocks_total"),
                        ("voices", "voices", "voices_peak"),
                        ("lim", "streaming", "limit")):
        pair = _split_pair(kv.get(key))
        if pair:
            out[lo], out[hi] = pair
    pitch = kv.get("pitch")
    if isinstance(pitch, str):
        try:
            out["pitch"] = [int(x) for x in pitch.split(",") if x]
        except ValueError:
            pass
    elif isinstance(pitch, int):
        out["pitch"] = [pitch]
    engines = kv.get("engines")
    out["engines"] = str(engines).split(",") if engines else ["native", "marcel"]
    return out


def _parse_smplpitch(body: str) -> dict:
    # "SMPLPITCH: pad=%d semi=%d engine=native|marcel|none"
    return {"kind": "smplpitch", **_kv(body)}


def _parse_smplpoly(body: str) -> dict:
    # "SMPLPOLY: max_voices=%u max_streaming=%u" (the native engine's caps)
    return {"kind": "smplpoly", **_kv(body)}


def _parse_pitched(body: str) -> dict | None:
    # "PITCHED: active=%d zones=%u roots=%s voices=%u/%u bank_kb=%u rt_max_us=%u
    #           steals=%u [err=why]"
    m = _PITCHED_RE.search(body)
    if not m:
        return None
    roots_raw = m.group(3)
    try:
        roots = [float(r) for r in roots_raw.split(",") if r] if roots_raw else []
    except ValueError:
        # A non-numeric roots token is a bad reply, not a crash — parse_cdc_reply's
        # contract is None for anything it cannot make sense of.
        return None
    return {
        "kind": "pitched",
        "active": m.group(1) != "0",
        "zones": int(m.group(2)),
        "roots": roots,
        "voices_active": int(m.group(4)),
        "voices_max": int(m.group(5)),
        "bank_kb": int(m.group(6)),
        "rt_max_us": int(m.group(7)),
        "steals": int(m.group(8)),
        # None on a board that never failed a load, and on firmware that
        # predates the field — both mean "no reason to report". Named
        # load_err, not err: "kind": "err" is a whole different reply shape
        # elsewhere in this module, and this is a field on a healthy "pitched"
        # reply, not one.
        "load_err": m.group(9),
    }


def _parse_appver(body: str) -> dict | None:
    # "APPVER: %s id=%s commit=%s ref=%s dirty=%d"  |  "APPVER: end count=%u"
    end = _APPVER_END_RE.match(body)
    if end:
        return {"kind": "appver", "end": True, "count": int(end.group(1))}
    m = _APPVER_RE.match(body)
    if not m:
        return None
    return {
        "kind": "appver",
        "component": m.group(1),
        "id": _dash_none(m.group(2)),
        "commit": m.group(3),
        "ref": _dash_none(m.group(4)),
        "dirty": m.group(5) != "0",
    }


def _parse_padinfo(body: str, line: str) -> dict:
    # "PADINFO: idx=%d num=%u kit=%s vol=%d pan=%d mode=%u dirty=%d count=%u"
    #   + " | [%u] '%s' %u..%u" per sample slot
    parts = [p.strip() for p in body.split("|")]
    out: dict[str, Any] = {"kind": "padinfo", **(_kv(parts[0]) if parts else {})}
    slots: list[dict[str, Any]] = []
    for part in parts[1:]:
        m = _PADINFO_SLOT_RE.search(part)
        if m:
            slots.append(
                {
                    "index": int(m.group(1)),
                    "file": m.group(2),
                    "lo": int(m.group(3)),
                    "hi": int(m.group(4)),
                }
            )
    out["slots"] = slots
    out["raw"] = line
    return out


# "STM_FW running=1.4/0x000F[(down)] board=r20 fw_pcb=r20 bundled=1.4/0x000F pcb=r20 size=112276"
# or, with no image in LittleFS, "STM_FW running=1.4/0x000F bundled=none(ESP_ERR_NOT_FOUND)".
# No colon after the verb — the reply predates the NAME: convention.
_STM_FW_VER_RE = re.compile(r"^(\d+\.\d+)/(0x[0-9A-Fa-f]+)(\(down\))?$")


def _parse_padmix(body: str) -> dict | None:
    m = _PADMIX_END_RE.match(body)
    if m:
        return {"kind": "padmix", "end": True, "count": int(m.group(1))}
    m = _PADMIX_HEAD_RE.match(body)
    if m:
        return {"kind": "padmix", "kit": int(m.group(1)), "max": int(m.group(2)),
                "loading": m.group(3) != "0", "name": m.group(4).strip()}
    m = _PADMIX_PAD_RE.match(body)
    if m:
        return {"kind": "padmix", "pad": int(m.group(1)), "vol": int(m.group(2)),
                "pan": int(m.group(3)), "sample": _dash_none(m.group(4).strip())}
    return None


def _parse_stm_fw(body: str) -> dict:
    out: dict = {"kind": "stm_fw", "linked": True}
    for token in body.split():
        if "=" not in token:
            continue
        key, value = token.split("=", 1)
        if key in ("running", "bundled"):
            m = _STM_FW_VER_RE.match(value)
            if m:
                out[key] = m.group(1)
                out[f"{key}_proto"] = int(m.group(2), 16)
                if key == "running" and m.group(3):
                    out["linked"] = False
            else:
                # "none(ESP_ERR_NOT_FOUND)"
                out[key] = None
                out[f"{key}_error"] = value[len("none("):-1] if value.startswith("none(") else value
        elif key == "size":
            out[key] = int(value)
        else:
            out[key] = value
    return out


def _split_tail(body: str, key: str) -> tuple[str, str | None]:
    """`body` up to " key=" and what follows it -- a value that may hold spaces."""
    marker = f" {key}="
    at = body.find(marker)
    if at < 0:
        if body.startswith(f"{key}="):
            return "", body[len(key) + 1:]
        return body, None
    return body[:at], body[at + len(marker):]


def _slashed(value: Any) -> list[int]:
    return [int(x) for x in str(value).split("/") if x != ""]


def _parse_asong(body: str) -> dict:
    # "ASONG: ok= open= bpm= len= sync= play= pos= under= openfail= bounce= done= total=
    #  clips=a/b/c/d file=<clip|-> project=<name, may have spaces>[ err=<text>]"
    # Names last on purpose (hil_control.cpp): everything before is plain k=v.
    head, err = _split_tail(body, "err")
    head, project = _split_tail(head, "project")
    out: dict[str, Any] = {"kind": "asong", **_kv(head)}
    out["clips"] = _slashed(out.get("clips", ""))
    if "pos" in out:
        out["pos"] = float(out["pos"])
    out["file"] = out.get("file")          # "-" is None already
    out["project"] = project if project is not None else ""
    out["err"] = err
    return out


def _parse_song(body: str) -> dict:
    # "SONG: mode= loop= len= pos= pass= scene= blocks=s:r,s:r,..."
    head, blocks = _split_tail(body, "blocks")
    out: dict[str, Any] = {"kind": "song", **_kv(head)}
    pairs = []
    for item in (blocks or "").split(","):
        scene, sep, reps = item.partition(":")
        if sep:
            pairs.append((int(scene), int(reps)))
    out["blocks"] = pairs
    return out


def _parse_seq(body: str) -> dict:
    # "SEQ: mode= state= scenes= scene=A>P bars=a/b/c/d ev=a/b/c/d bpm= quant= pos=
    #  dirty= erase= countin= file=<path, may have spaces>"
    head, path = _split_tail(body, "file")
    out: dict[str, Any] = {"kind": "seq", **_kv(head)}
    for key in ("bars", "ev"):
        if key in out:
            out[key] = _slashed(out[key])
    for key in ("bpm", "pos"):
        if key in out:
            out[key] = float(out[key])
    out["file"] = path
    return out


def parse_cdc_reply(line: str) -> dict | None:
    """Decode one CDC reply line into a dict keyed by "kind"; None when it is not a known reply."""
    text = line.strip()
    if not text:
        return None
    if text == "OK":
        return {"kind": "ok"}
    if text == "ERR" or text.startswith("ERR "):
        return {"kind": "err", "message": text[3:].strip()}
    # The bulk verbs answer "FS_ERR <reason>" / "ASSETS_ERROR <reason>" — an error
    # with its family in front, so a stream's failure is never mistaken for a
    # rejected text command. It is still an error reply.
    m = _FAMILY_ERR_RE.match(text)
    if m:
        return {"kind": "err", "message": (m.group(2) or "").strip(), "family": m.group(1)}
    if text.startswith("STM_FW "):
        return _parse_stm_fw(text[len("STM_FW "):])
    m = _PREFIX_RE.match(text)
    if not m:
        return None
    prefix, body = m.group(1), m.group(2).strip()
    if body == "ERR" or body.startswith("ERR "):
        return {"kind": "err", "message": text}
    if prefix == "KITSTATUS":
        return _parse_kitstatus(body)
    if prefix == "APPS":
        return _parse_apps(body)
    if prefix == "KITS":
        return _parse_kits(body)
    if prefix == "KITSWAP":
        return _parse_kitswap(body)
    if prefix == "PADSTATS":
        return _parse_padstats(body)
    if prefix == "PADNOTES":
        return _parse_padnotes(body)
    if prefix == "LEDS":
        return _parse_leds(body)
    if prefix == "MEM":
        return _parse_mem(body)
    if prefix == "CLOCK":
        return _parse_clock(body)
    if prefix == "MEMBLK":
        return _parse_memblk(body)
    if prefix == "MEMBIG":
        return _parse_membig(body)
    if prefix == "CDCSTATS":
        return _parse_cdcstats(body)
    if prefix == "LVGLSTATS":
        return _parse_lvglstats(body)
    if prefix == "WAVE":
        return _parse_wave(body)
    if prefix == "BLE":
        return _parse_ble(body)
    if prefix == "BLEDEV":
        return _parse_bledev(body)
    if prefix == "WIFI":
        return _parse_wifi(body)
    if prefix == "WIFI_AP":
        return _parse_wifi_ap(body)
    if prefix == "WIFI_NET":
        return _parse_wifi_net(body)
    if prefix == "WIFI_PING":
        return _parse_wifi_ping(body)
    if prefix == "NET_MODE":
        return _parse_net_mode(body)
    if prefix == "UI":
        return _parse_ui(body)
    if prefix == "DAW":
        return _parse_daw(body)
    if prefix == "ENC":
        return _parse_enc(body)
    if prefix == "ENCFOCUS":
        return _parse_encfocus(body)
    if prefix == "ENCGROUP":
        return _parse_encgroup_head(body)
    if prefix == "AUDIOLVL":
        return _parse_audiolvl(body)
    if prefix == "AUDIO_RATE":
        return _parse_audiorate(body)
    if prefix == "SMPLPEAK":
        return _parse_smplpeak(body)
    if prefix == "SMPLENG":
        return _parse_smpleng(body)
    if prefix == "SMPLPITCH":
        return _parse_smplpitch(body)
    if prefix == "SMPLPOLY":
        return _parse_smplpoly(body)
    if prefix == "PITCHED":
        return _parse_pitched(body)
    if prefix == "APPVER":
        return _parse_appver(body)
    if prefix == "PADINFO":
        return _parse_padinfo(body, text)
    if prefix == "PADMIX":
        return _parse_padmix(body)
    if prefix == "KITSAVE":
        # "KITSAVE: busy=0|1 ok=-1|0|1 kit=N"
        return {"kind": "kitsave", **_kv(body)}
    if prefix == "ASONG":
        return _parse_asong(body)
    if prefix == "SONG":
        return _parse_song(body)
    if prefix == "SEQ":
        return _parse_seq(body)
    if prefix == "RSTLOG":
        return {"kind": "rstlog", **_kv(body)}
    if prefix == "FWUPD":
        return _parse_fwupd(body)
    if prefix == "FWUI":
        # "FWUI: tab=%s sel=%s card=%s items=%u selkey=%s host=%s store=%s busy=%u"
        return {"kind": "fwui", **_kv(body)}
    if prefix == "FWCAT":
        return _parse_fwcat(body)
    return None


# "  [%u] %p %s" — the label is everything after the pointer and one separating space; it may be
# empty (widget_label returns "" for an object with no caption) or contain spaces and brackets.
# Firmware since platform-idf 35f09ef puts the widget's box between pointer and label.
_ENC_ENTRY_RE = re.compile(
    r"^\s*\[(\d+)\]\s+(\S+)(?: x=(-?\d+) y=(-?\d+) w=(-?\d+) h=(-?\d+))?(?: (.*))?$")


def parse_enc_group(lines: list[str]) -> list[dict]:
    """Decode an ENC_GROUP dump: head line optional, ERR/noise skipped, index is the device's."""
    entries: list[dict] = []
    for raw in lines:
        line = raw.rstrip("\r\n")
        m = _ENC_ENTRY_RE.match(line)
        if not m:
            continue
        index = int(m.group(1))
        label = m.group(7) if m.group(7) is not None else ""
        entry = {"ref": f"e{index}", "index": index, "ptr": m.group(2), "label": label}
        if m.group(3) is not None:
            entry.update(x=int(m.group(3)), y=int(m.group(4)), w=int(m.group(5)), h=int(m.group(6)))
        entries.append(entry)
    return entries
