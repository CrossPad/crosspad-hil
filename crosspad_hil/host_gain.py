"""The host's own gain on the CrossPad UAC2 card: read it, put it at 0 dB, say what it was.

A level the board is judged on (``audio_loopback``'s "level too low") passes
through two host-side volumes before it means anything about the firmware:

* the card's ALSA mixer — the UAC feature units, ``PCM`` and friends. On the
  bench it sat at 30/50 = -20 dB, and a clean loop came back at -45.7 dBFS
  against a -45 floor: 32 FAILs that were the host's;
* the PipeWire node volumes (``crosspad_out`` / ``crosspad_mic`` and the
  card's own ALSA nodes), which apply when playback goes through PipeWire.

Setting either by hand does not last: the session (WirePlumber's stored
routes, alsactl's saved state) restores them every time the card enumerates,
and a USB profile switch is an enumeration. So a scenario sets them each time
the card appears, reads them back, and records both readings in its report.

Every command goes through ``run`` (``subprocess.run``-like) so tests can
replay real tool output. Missing tools are reported, not raised: a host
without ``amixer`` or ``wpctl`` simply has nothing to say about that half.
"""
from __future__ import annotations

import math
import re
import shutil
import subprocess
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from crosspad_hil import audio_nodes

TARGET_DB = 0.0
TOLERANCE_DB = 0.5

Run = Callable[[list[str]], subprocess.CompletedProcess]

_SCONTROL = re.compile(r"^Simple mixer control '(.+)',(\d+)\s*$")
_CHANNEL = re.compile(r"^\s*[^:]+:\s*(Playback|Capture)\s+(-?\d+)\s+\[(\d+)%\]"
                      r"(?:\s+\[(-?\d+(?:\.\d+)?)dB\])?(?:\s+\[(on|off)\])?")
_WPCTL = re.compile(r"Volume:\s*([0-9.]+)(\s*\[MUTED\])?")


def _run(argv: list[str]) -> subprocess.CompletedProcess:
    return subprocess.run(argv, capture_output=True, text=True, timeout=10.0, check=False)


# --------------------------------------------------------------------------- parsing

def parse_scontrols(text: str) -> list[str]:
    """``amixer scontrols`` -> control specs ``"'PCM',0"`` as amixer takes them back."""
    out = []
    for line in text.splitlines():
        m = _SCONTROL.match(line.strip())
        if m:
            out.append(f"'{m.group(1)}',{m.group(2)}")
    return out


def parse_sget(text: str) -> list[dict[str, Any]]:
    """``amixer sget`` -> one entry per direction with a volume: the quietest channel
    (``{"direction", "raw", "percent", "db", "on"}``); ``db`` None when the card
    publishes no dB range."""
    worst: dict[str, dict[str, Any]] = {}
    for line in text.splitlines():
        m = _CHANNEL.match(line)
        if not m:
            continue
        direction = m.group(1).lower()
        entry = {"direction": direction, "raw": int(m.group(2)), "percent": int(m.group(3)),
                 "db": float(m.group(4)) if m.group(4) is not None else None,
                 "on": None if m.group(5) is None else m.group(5) == "on"}
        have = worst.get(direction)
        if have is None or entry["percent"] < have["percent"]:
            worst[direction] = entry
    return [worst[d] for d in ("playback", "capture") if d in worst]


def parse_wpctl_volume(text: str) -> tuple[float, bool] | None:
    """``wpctl get-volume`` -> (volume, muted)."""
    m = _WPCTL.search(text)
    if not m:
        return None
    return float(m.group(1)), bool(m.group(2))


def wpctl_db(volume: float) -> float:
    """wpctl prints the cubic volume (the UI scale); the gain is its cube."""
    return -math.inf if volume <= 0 else 60.0 * math.log10(volume)


def crosspad_nodes(names: dict[str, int]) -> dict[str, int]:
    """The PipeWire nodes that carry the CrossPad's audio, by name -> id."""
    return {n: i for n, i in names.items()
            if n in (audio_nodes.OUT_NODE, audio_nodes.MIC_NODE)
            or (n.startswith("alsa_") and "crosspad" in n.lower())}


# --------------------------------------------------------------------------- the reading

@dataclass
class HostGain:
    card: str | None
    controls: list[dict[str, Any]] = field(default_factory=list)
    nodes: list[dict[str, Any]] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)

    def deviations(self, tolerance_db: float = TOLERANCE_DB) -> list[str]:
        """Every volume that is not at unity, as "<what> <dB>"; muted counts."""
        out = []
        for c in self.controls:
            what = f"ALSA {c['name']} {c['direction']}"
            if c.get("on") is False:
                out.append(f"{what} muted")
            elif c["db"] is not None:
                if abs(c["db"] - TARGET_DB) > tolerance_db:
                    out.append(f"{what} {c['db']:+.1f} dB")
            elif c["percent"] < 100:
                out.append(f"{what} {c['percent']}%")
        for n in self.nodes:
            what = f"PipeWire {n['name']}"
            if n["muted"]:
                out.append(f"{what} muted")
            elif abs(n["db"] - TARGET_DB) > tolerance_db:
                out.append(f"{what} {n['db']:+.1f} dB")
        return out

    def worst_db(self) -> float | None:
        dbs = [c["db"] for c in self.controls if c["db"] is not None]
        dbs += [n["db"] for n in self.nodes]
        return min(dbs) if dbs else None

    def to_dict(self) -> dict[str, Any]:
        return {"card": self.card, "controls": self.controls, "nodes": self.nodes,
                "notes": self.notes, "deviations": self.deviations(),
                "worst_db": self.worst_db()}


def _have(tool: str) -> bool:
    return shutil.which(tool) is not None


def read(card: str | None, *, run: Run = _run,
         have: Callable[[str], bool] = _have) -> HostGain:
    g = HostGain(card=card)
    if card is None:
        g.notes.append("no ALSA card for the CrossPad: mixer not read")
    elif not have("amixer"):
        g.notes.append("amixer not installed: mixer not read")
    else:
        listing = run(["amixer", "-c", card, "scontrols"])
        if listing.returncode != 0:
            g.notes.append(f"amixer -c {card}: {(listing.stderr or '').strip()}")
        for spec in parse_scontrols(listing.stdout or ""):
            got = run(["amixer", "-c", card, "sget", spec])
            for entry in parse_sget(got.stdout or ""):
                g.controls.append({"name": spec, **entry})
    if not have("wpctl") or not have("pw-dump"):
        g.notes.append("wpctl/pw-dump not installed: PipeWire volumes not read")
        return g
    names = audio_nodes._node_ids(audio_nodes._pw_dump(run))
    for name, node_id in sorted(crosspad_nodes(names).items()):
        parsed = parse_wpctl_volume(run(["wpctl", "get-volume", str(node_id)]).stdout or "")
        if parsed is None:
            continue
        vol, muted = parsed
        g.nodes.append({"name": name, "id": node_id, "volume": vol, "muted": muted,
                        "db": round(wpctl_db(vol), 2)})
    return g


def apply(card: str | None, *, run: Run = _run,
          have: Callable[[str], bool] = _have) -> tuple[HostGain, HostGain]:
    """Put every CrossPad volume at 0 dB (100 % where the card has no dB scale)
    and unmuted. Returns (before, after); ``after`` is read back, not assumed."""
    before = read(card, run=run, have=have)
    for c in before.controls:
        spec = c["name"]
        if c["db"] is not None:
            run(["amixer", "-c", str(card), "sset", spec, f"{TARGET_DB:g}dB"])
        else:
            run(["amixer", "-c", str(card), "sset", spec, "100%"])
        if c.get("on") is False:
            run(["amixer", "-c", str(card), "sset", spec,
                 "unmute" if c["direction"] == "playback" else "cap"])
    for n in before.nodes:
        run(["wpctl", "set-volume", str(n["id"]), "1.0"])
        if n["muted"]:
            run(["wpctl", "set-mute", str(n["id"]), "0"])
    return before, read(card, run=run, have=have)
