"""What the MIDI hub tells the board about this computer, as companion verb lines.

`HOST_INFO cpu=<pct> ram=<used_mb>/<total_mb> time=<unix_s> tz=<min_east>
daws=<installed>/<running>` keeps the board's USB glyph lit and its clock set,
and tells DAW Control which DAWs it may offer to launch (daw_launcher); `FW_LATEST <x.y.z>` is the
newest firmware for the board's kit and channel in crosspad.app's catalog
(fw_catalog), or ~/.config/crosspad/latest.json (`{"version": "x.y.z"}`) when
that file exists (an explicit override). Linux only, like the hub: /proc/stat, /proc/meminfo.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

LATEST_PATH = Path.home() / ".config" / "crosspad" / "latest.json"
PROC_STAT = Path("/proc/stat")
PROC_MEMINFO = Path("/proc/meminfo")
KB_PER_MB = 1024
SECONDS_PER_MINUTE = 60


class CpuMeter:
    """Whole-machine load between two readings of /proc/stat."""

    def __init__(self, stat: Path = PROC_STAT) -> None:
        self._stat = stat
        self._last = self._times()

    def _times(self) -> tuple[int, int]:
        fields = [int(x) for x in self._stat.read_text().splitlines()[0].split()[1:]]
        return fields[3] + fields[4], sum(fields)      # idle + iowait, total

    def percent(self) -> int:
        idle, total = self._times()
        d_idle, d_total = idle - self._last[0], total - self._last[1]
        self._last = (idle, total)
        return 0 if d_total <= 0 else round(100 * (d_total - d_idle) / d_total)


def ram_mb(meminfo: Path = PROC_MEMINFO) -> tuple[int, int]:
    """(used, total) in MB; used = total - MemAvailable, what `free` calls used."""
    info: dict[str, int] = {}
    for row in meminfo.read_text().splitlines():
        key, _, value = row.partition(":")
        if value.strip():
            info[key] = int(value.split()[0])          # kB
    total = info["MemTotal"]
    avail = info.get("MemAvailable", info["MemFree"])
    return (total - avail) // KB_PER_MB, total // KB_PER_MB


def host_info_line(cpu: CpuMeter, *, meminfo: Path = PROC_MEMINFO, now: float | None = None,
                   daws: tuple[int, int] | None = None) -> str:
    """`daws` = (installed, running) bitmasks by DAW daw_id (daw_launcher.DawLauncher.masks)."""
    now = time.time() if now is None else now
    used, total = ram_mb(meminfo)
    tz = time.localtime(now).tm_gmtoff // SECONDS_PER_MINUTE
    line = f"HOST_INFO cpu={cpu.percent()} ram={used}/{total} time={int(now)} tz={tz}"
    return line if daws is None else f"{line} daws={daws[0]}/{daws[1]}"


def latest_version(path: Path = LATEST_PATH) -> str | None:
    try:
        version = json.loads(path.read_text()).get("version")
    except (OSError, ValueError, AttributeError):
        return None
    return version if isinstance(version, str) and version else None


def latest_for(cache, kit: str, channel: str, override=latest_version) -> str | None:
    """The version FW_LATEST names: ~/.config/crosspad/latest.json when present
    (an offline bench's explicit override), else crosspad.app's catalog."""
    return override() or cache.latest(kit, channel)
