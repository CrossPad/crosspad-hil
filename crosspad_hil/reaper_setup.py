"""Put the CrossPad control script into REAPER and start it with REAPER.

The script itself lives with the app (crosspad-dawcontrol `host/reaper/CrossPad.lua`);
this copies it to `<resource>/Scripts/CrossPad/` and adds one marked line to
`<resource>/Scripts/__startup.lua`, which REAPER runs on every start. A copy,
not a link: REAPER keeps running what was installed even while the app repo
moves or checks out another branch — reinstall to update.

On Linux it also enables the hub's virtual raw MIDI card ("hw:VirMIDI", see
midi_hub) for input and output. REAPER keeps enabled devices as bitmasks by
device index in reaper.ini — `midiins` / `midiouts` for devices 0-31, `_h`,
`_x`, `_x_h` for the next three 32 — and the index of a name in its device
cache (reaper-midihw-alsa.ini, `in<N>=` / `on<N>=`). REAPER rewrites
reaper.ini when it quits, so this runs only while REAPER is closed, and only
after REAPER has run once with the card present (it is not in the cache before).
"""
from __future__ import annotations

import os
import re
import sys
from collections.abc import Callable
from pathlib import Path

from crosspad_hil.errors import BAD_ARGS, HilError

SCRIPT_NAME = "CrossPad.lua"
SCRIPT_DIR = Path("Scripts") / "CrossPad"
STARTUP = Path("Scripts") / "__startup.lua"
STARTUP_MARK = "-- CrossPad DAW Control (crosspad-hil reaper install)"
STARTUP_LINE = 'dofile(reaper.GetResourcePath() .. "/Scripts/CrossPad/CrossPad.lua")'
# Where the script sits in a platform checkout; the default when --script is not given.
REPO_SCRIPT = Path("components/crosspad-dawcontrol/host/reaper") / SCRIPT_NAME
VIRMIDI_DEVICE = "hw:VirMIDI"
DEVICE_CACHE = "reaper-midihw-alsa.ini"
MASK_WORDS = ("", "_h", "_x", "_x_h")   # 32 devices per word, index 0..127
BITS_PER_WORD = 32


def reaper_running(proc: Path = Path("/proc")) -> bool:
    for entry in proc.iterdir():
        if not entry.name.isdigit():
            continue
        try:
            argv0 = (entry / "cmdline").read_bytes().split(b"\0", 1)[0]
        except OSError:
            continue
        if argv0.rsplit(b"/", 1)[-1] == b"reaper":
            return True
    return False


def device_index(cache_text: str, direction: str, name: str) -> int | None:
    """Index of `name` among REAPER's cached MIDI devices; direction "i" or "o"."""
    for m in re.finditer(rf"^{direction}n(\d+)=(.*)$", cache_text, re.MULTILINE):
        if m.group(2).strip() == name:
            return int(m.group(1))
    return None


def _section_bounds(lines: list[str], section: str) -> tuple[int, int] | None:
    start = next((i for i, row in enumerate(lines) if row.strip().lower() == f"[{section}]"), None)
    if start is None:
        return None
    end = next((i for i in range(start + 1, len(lines)) if lines[i].startswith("[")), len(lines))
    return start, end


def set_enabled_bit(ini_text: str, key: str, index: int) -> str:
    """reaper.ini with device `index` set in the `key` bitmask; everything else untouched."""
    word = f"{key}{MASK_WORDS[index // BITS_PER_WORD]}"
    bit = 1 << (index % BITS_PER_WORD)
    lines = ini_text.splitlines()
    bounds = _section_bounds(lines, "reaper")
    if bounds is None:
        lines += ["[reaper]"]
        bounds = (len(lines) - 1, len(lines))
    start, end = bounds
    for i in range(start + 1, end):
        k, sep, v = lines[i].partition("=")
        if sep and k == word:
            lines[i] = f"{word}={int(v or 0) | bit}"
            break
    else:
        while end > start + 1 and not lines[end - 1].strip():
            end -= 1                      # before the blank line that closes the section
        lines.insert(end, f"{word}={bit}")
    return "\n".join(lines) + "\n"


def enable_virmidi(resource: Path, running: Callable[[], bool] = reaper_running) -> dict:
    """Enable hw:VirMIDI for input and output; {"enabled": bool, "why": str}."""
    cache = resource / DEVICE_CACHE
    text = cache.read_text() if cache.exists() else ""
    idx_in = device_index(text, "i", VIRMIDI_DEVICE)
    idx_out = device_index(text, "o", VIRMIDI_DEVICE)
    if idx_in is None or idx_out is None:
        return {"enabled": False, "why": f"{VIRMIDI_DEVICE} not in REAPER's device list yet: "
                                          "load snd-virmidi, start REAPER once, close it, "
                                          "install again"}
    if running():
        return {"enabled": False, "why": "REAPER is running and would overwrite reaper.ini "
                                         "when it quits: close it and install again"}
    ini = resource / "reaper.ini"
    body = ini.read_text() if ini.exists() else ""
    body = set_enabled_bit(set_enabled_bit(body, "midiins", idx_in), "midiouts", idx_out)
    ini.write_text(body)
    return {"enabled": True, "why": f"{VIRMIDI_DEVICE} input {idx_in}, output {idx_out}"}


def default_resource_dir() -> Path:
    """REAPER's resource path for a normal (non-portable) install."""
    if sys.platform == "darwin":
        return Path.home() / "Library/Application Support/REAPER"
    if sys.platform == "win32":
        return Path(os.environ.get("APPDATA", Path.home())) / "REAPER"
    return Path.home() / ".config/REAPER"


def find_script(cwd: Path | None = None) -> Path | None:
    candidate = (cwd or Path.cwd()) / REPO_SCRIPT
    return candidate if candidate.is_file() else None


def _startup_has_line(text: str) -> bool:
    return STARTUP_MARK in text


def install(resource: Path, script: Path, *, midi: bool = sys.platform.startswith("linux"),
            running: Callable[[], bool] = reaper_running) -> dict:
    if not resource.is_dir():
        raise HilError(BAD_ARGS, f"no REAPER resource folder at {resource}",
                       hint="start REAPER once, or pass --resource")
    if not script.is_file():
        raise HilError(BAD_ARGS, f"no control script at {script}",
                       hint=f"pass --script, or run from a platform-idf checkout ({REPO_SCRIPT})")
    target = resource / SCRIPT_DIR / SCRIPT_NAME
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(script.read_bytes())
    startup = resource / STARTUP
    text = startup.read_text() if startup.exists() else ""
    if not _startup_has_line(text):
        if text and not text.endswith("\n"):
            text += "\n"
        startup.write_text(f"{text}{STARTUP_MARK}\n{STARTUP_LINE}\n")
    result = status(resource)
    if midi:
        result["midi"] = enable_virmidi(resource, running)
    return result


def status(resource: Path) -> dict:
    startup = resource / STARTUP
    return {
        "resource": str(resource),
        "script": str(resource / SCRIPT_DIR / SCRIPT_NAME),
        "script_installed": (resource / SCRIPT_DIR / SCRIPT_NAME).is_file(),
        "starts_with_reaper": startup.exists() and _startup_has_line(startup.read_text()),
    }
