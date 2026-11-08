"""The board's SD card as the host sees it in USB transfer mode (MSC).

Settings → USB → "SD card as USB drive" restarts the board as a USB drive
(platform-idf usb_transfer.cpp; the MSC + CDC profile, so CDC stays): its SCSI
inquiry says vendor "CrossPad", product "SD Card". This finds that disk, mounts
it through UDisks2 the way a desktop does (or finds where the desktop mounted
it), and ejects it with ``Drive.Eject`` -- a SCSI START STOP UNIT that leaves
the board on USB. ``udisksctl power-off`` would take the whole board off
USB with the card (platform-idf CLAUDE.md, bench traps).

Linux only; everything goes through ``lsblk``, ``udisksctl`` and ``gdbus``.
"""
from __future__ import annotations

import json
import re
import subprocess
import time
from collections.abc import Callable, Sequence
from pathlib import Path

from crosspad_hil.errors import ENV, TIMEOUT, HilError

VENDOR = "CrossPad"
MODEL = "SD Card"

Runner = Callable[[Sequence[str]], str]


def _run(cmd: Sequence[str]) -> str:
    try:
        out = subprocess.run(list(cmd), capture_output=True, text=True, timeout=30)
    except FileNotFoundError as e:
        raise HilError(ENV, f"{cmd[0]} is not installed", hint="udisks2 and lsblk are needed "
                       "for the USB drive steps") from e
    if out.returncode != 0:
        raise HilError(ENV, f"{' '.join(cmd)} failed: {out.stderr.strip() or out.stdout.strip()}",
                       hint="is udisks2 running, and may this user mount removable drives?")
    return out.stdout


def find_disk(blockdevices: list[dict]) -> tuple[str, str] | None:
    """(disk, what to mount) of the board in @p blockdevices (lsblk -J): its first
    partition, or the disk itself when the card has no partition table."""
    for d in blockdevices:
        if d.get("type") != "disk":
            continue
        if (d.get("vendor") or "").strip() != VENDOR or \
                not (d.get("model") or "").strip().startswith(MODEL):
            continue
        parts = [c for c in d.get("children") or [] if c.get("type") == "part"]
        disk = f"/dev/{d['name']}"
        return disk, f"/dev/{parts[0]['name']}" if parts else disk
    return None


def wait_disk(timeout_s: float = 60.0, *, run: Runner = _run,
              wait: Callable[[float], object] = time.sleep) -> tuple[str, str]:
    """Poll lsblk until the board's card is a disk; TIMEOUT when it never shows."""
    deadline = time.monotonic() + timeout_s
    while True:
        tree = json.loads(run(["lsblk", "-J", "-o", "NAME,TYPE,VENDOR,MODEL,TRAN"]))
        found = find_disk(tree.get("blockdevices", []))
        if found:
            return found
        if time.monotonic() >= deadline:
            raise HilError(TIMEOUT, "the board's SD card never showed up as a USB disk",
                           hint=f"lsblk lists no {VENDOR} {MODEL}; did the board restart into "
                                "USB transfer mode?")
        wait(1.0)


_MOUNTED = re.compile(r"Mounted \S+ at (.+?)\.?$")
_DRIVE = re.compile(r"Drive:\s+'(/org/freedesktop/UDisks2/drives/[^']+)'")


def mount(part: str, *, run: Runner = _run, attempts: int = 10,
          wait: Callable[[float], object] = time.sleep) -> Path:
    """Mount @p part as the desktop would (udisksctl, no password prompt). UDisks2
    learns of a new disk a moment after the kernel does: until then it answers
    "Error looking up object", and the mount is asked again."""
    for left in range(attempts - 1, -1, -1):
        try:
            out = run(["udisksctl", "mount", "-b", part, "--no-user-interaction"]).strip()
            break
        except HilError as e:
            if "AlreadyMounted" in e.message or "already mounted" in e.message:
                return mountpoint(part, run=run)       # the desktop got there first
            if "looking up object" not in e.message or not left:
                raise
            wait(1.0)
    m = _MOUNTED.search(out)
    if not m:
        raise HilError(ENV, f"udisksctl did not say where {part} went: {out!r}")
    return Path(m.group(1))


def mountpoint(part: str, *, run: Runner = _run) -> Path:
    """Where @p part is mounted now (a desktop mounts a new drive by itself)."""
    where = run(["lsblk", "-no", "MOUNTPOINT", part]).strip().splitlines()
    if not where or not where[0]:
        raise HilError(ENV, f"{part} is not mounted anywhere")
    return Path(where[0])


def unmount(part: str, *, run: Runner = _run) -> None:
    run(["udisksctl", "unmount", "-b", part, "--no-user-interaction"])


def eject(disk: str, *, run: Runner = _run) -> None:
    """``Drive.Eject``: the medium out, the board still on USB."""
    m = _DRIVE.search(run(["udisksctl", "info", "-b", disk]))
    if not m:
        raise HilError(ENV, f"udisksctl info names no drive for {disk}")
    run(["gdbus", "call", "--system", "--dest", "org.freedesktop.UDisks2",
         "--object-path", m.group(1), "--method", "org.freedesktop.UDisks2.Drive.Eject", "{}"])
