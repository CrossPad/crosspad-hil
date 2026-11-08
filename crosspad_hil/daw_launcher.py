"""Which DAWs this computer can start, which already run, and starting one.

The MIDI hub reports both as bitmasks by DAW-protocol daw_id in its HOST_INFO
(`daws=<installed>/<running>`), and DAW Control on the board offers a button
for every DAW that is installed and not running; its `launch` frame comes back
through the hub, which calls DawLauncher.launch(). A DAW that runs but whose
script does not answer is not started again — the board names it instead.

FL Studio and Ableton Live live in a Wine prefix (the same search as the FL
template: Bottles, then ~/.wine) and start through that prefix's own runner
(wine_command), REAPER is native. Nothing is launchable
without a graphical session: the hub runs under `systemd --user`, whose
environment has DISPLAY / WAYLAND_DISPLAY only once the desktop imported them,
so they are read from the manager when the hub's own environment lacks them.
The DAW opens whatever it opens by itself; no project is ever picked here.
"""
from __future__ import annotations

import glob
import os
import shutil
import subprocess
import threading
import time
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from pathlib import Path

from crosspad_hil import fl_template
from crosspad_hil.midi import daw_ids

SESSION_VARS = ("DISPLAY", "WAYLAND_DISPLAY", "XAUTHORITY", "XDG_RUNTIME_DIR",
                "DBUS_SESSION_BUS_ADDRESS")
RESCAN_S = 30.0          # installs change rarely; a glob over the prefix every report would not
RELAUNCH_GUARD_S = 30.0  # FL under Wine shows no process for a while after `wine` returns
REAPER_PATHS = ("/opt/REAPER/reaper", "~/opt/REAPER/reaper")
Which = Callable[[str], "str | None"]
Command = tuple[tuple[str, ...], dict[str, str]]    # argv, environment on top of the session's
FLATPAK_BOTTLES = "com.usebottles.bottles"


@dataclass(frozen=True)
class Daw:
    daw_id: int
    name: str
    command: tuple[str, ...]
    env: dict[str, str] = field(default_factory=dict)
    # argv[0] basenames (lower case) of the running DAW; a Wine process shows
    # its Windows path there.
    process_names: frozenset[str] = frozenset()


def _basename(arg: str) -> str:
    return arg.replace("\\", "/").rsplit("/", 1)[-1].lower()


def bottle_name(bottle: Path) -> str:
    """The name bottles-cli knows a bottle by (`Name:` in bottle.yml), not its folder."""
    try:
        for row in (bottle / "bottle.yml").read_text().splitlines():
            if row.startswith("Name:"):
                return row.partition(":")[2].strip() or bottle.name
    except OSError:
        pass
    return bottle.name


def wine_command(drive_c: Path, exe: Path, which: Which = shutil.which) -> Command | None:
    """How to start `exe` in its own prefix: a bottle through its own runner
    (Bottles pins one per bottle, e.g. kron4ek wine 11), anything else through
    the system wine. None when the tool for that prefix is missing."""
    bottle = drive_c.parent
    if FLATPAK_BOTTLES in bottle.parts:
        flatpak = which("flatpak")
        return None if flatpak is None else (
            (flatpak, "run", "--command=bottles-cli", FLATPAK_BOTTLES,
             "run", "-b", bottle_name(bottle), "-e", str(exe)), {})
    if bottle.parent.name == "bottles" and bottle.parent.parent.name == "bottles":
        cli = which("bottles-cli")
        return None if cli is None else (
            (cli, "run", "-b", bottle_name(bottle), "-e", str(exe)), {})
    wine = which("wine") or which("wine-stable")
    return None if wine is None else ((wine, str(exe)), {"WINEPREFIX": str(bottle)})


def _wine_daw(daw_id: int, name: str, drive_c: Path, exe: Path,
              process_names: frozenset[str], which: Which) -> Daw | None:
    how = wine_command(drive_c, exe, which)
    return None if how is None else Daw(daw_id, name, how[0], how[1], process_names)


def find_fl(drive_c: Path | None, which: Which = shutil.which) -> Daw | None:
    if drive_c is None:
        return None
    hits = sorted(drive_c.glob("Program Files/Image-Line/FL Studio*/FL64.exe"))
    if not hits:
        return None
    return _wine_daw(daw_ids()["fl_studio"], "FL Studio", drive_c, hits[-1],
                     frozenset({"fl64.exe", "fl64 (scaled).exe"}), which)


def find_ableton(prefixes: Iterable[Path], which: Which = shutil.which) -> Daw | None:
    for drive_c in prefixes:
        hits = sorted(drive_c.glob("ProgramData/Ableton/Live*/Program/Ableton Live*.exe"))
        if hits:
            return _wine_daw(daw_ids()["ableton"], "Ableton Live", drive_c, hits[-1],
                             frozenset({hits[-1].name.lower()}), which)
    return None


def find_reaper(which: Which = shutil.which,
                candidates: Iterable[str] = REAPER_PATHS) -> Daw | None:
    exe = which("reaper")
    if exe is None:
        exe = next((p for p in (os.path.expanduser(c) for c in candidates)
                    if os.access(p, os.X_OK)), None)
    if exe is None:
        return None
    return Daw(daw_ids()["reaper"], "REAPER", (exe,), {}, frozenset({"reaper"}))


def wine_prefixes() -> list[Path]:
    return [Path(hit) for pattern in fl_template.PREFIX_GLOBS
            for hit in sorted(glob.glob(os.path.expanduser(pattern)))]


def find_daws() -> dict[int, Daw]:
    found = [find_fl(fl_template.default_drive_c()),
             find_ableton(wine_prefixes()),
             find_reaper()]
    return {d.daw_id: d for d in found if d is not None}


def running_argv0(proc: Path = Path("/proc")) -> set[str]:
    """argv[0] basename (lower case) of every process we can read."""
    names: set[str] = set()
    for entry in proc.iterdir():
        if not entry.name.isdigit():
            continue
        try:
            argv0 = (entry / "cmdline").read_bytes().split(b"\0", 1)[0]
        except OSError:
            continue
        if argv0:
            names.add(_basename(argv0.decode(errors="replace")))
    return names


def manager_environment() -> dict[str, str]:
    """`systemctl --user show-environment`, the session variables a desktop imported."""
    try:
        out = subprocess.run(["systemctl", "--user", "show-environment"], capture_output=True,
                             text=True, check=False, timeout=5).stdout
    except (OSError, subprocess.SubprocessError):
        return {}
    env: dict[str, str] = {}
    for row in out.splitlines():
        key, sep, value = row.partition("=")
        if sep:
            env[key] = value
    return env


def session_env(own: dict[str, str],
                manager: Callable[[], dict[str, str]]) -> dict[str, str] | None:
    """The graphical session's variables, or None when there is no display to start a DAW on."""
    env = {k: own[k] for k in SESSION_VARS if own.get(k)}
    if not (env.get("DISPLAY") or env.get("WAYLAND_DISPLAY")):
        env = {k: v for k, v in manager().items() if k in SESSION_VARS and v}
    return env if env.get("DISPLAY") or env.get("WAYLAND_DISPLAY") else None


class DawLauncher:
    def __init__(self, *, find: Callable[[], dict[int, Daw]] = find_daws,
                 running: Callable[[], set[str]] = running_argv0,
                 environ: Callable[[], dict[str, str]] = lambda: dict(os.environ),
                 manager_env: Callable[[], dict[str, str]] = manager_environment,
                 popen: Callable[..., subprocess.Popen] = subprocess.Popen,
                 clock: Callable[[], float] = time.monotonic,
                 log: Callable[[str], None] = print) -> None:
        self._find = find
        self._running = running
        self._environ = environ
        self._manager_env = manager_env
        self._popen = popen
        self._clock = clock
        self._log = log
        self._daws: dict[int, Daw] = {}
        self._session: dict[str, str] | None = None
        self._next_scan = 0.0
        self._launched_at: dict[int, float] = {}

    def _rescan(self) -> None:
        now = self._clock()
        if now < self._next_scan:
            return
        self._next_scan = now + RESCAN_S
        self._daws = self._find()
        self._session = session_env(self._environ(), self._manager_env)

    def masks(self) -> tuple[int, int]:
        """(installed, running) by daw_id; installed is 0 without a graphical session."""
        self._rescan()
        if self._session is None:
            return 0, 0
        names = self._running()
        installed = running = 0
        for daw in self._daws.values():
            installed |= 1 << daw.daw_id
            if daw.process_names & names:
                running |= 1 << daw.daw_id
        return installed, running

    def launch(self, daw_id: int) -> str:
        """'started', or why not: 'unknown', 'no_session', 'running', 'recent', 'failed'."""
        self._rescan()
        daw = self._daws.get(daw_id)
        if daw is None:
            return "unknown"
        if self._session is None:
            return "no_session"
        now = self._clock()
        if now - self._launched_at.get(daw_id, -RELAUNCH_GUARD_S) < RELAUNCH_GUARD_S:
            return "recent"
        if daw.process_names & self._running():
            return "running"
        env = {**self._environ(), **self._session, **daw.env}
        try:
            child = self._popen(list(daw.command), env=env, cwd=str(Path(daw.command[-1]).parent),
                        stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL,
                        stderr=subprocess.DEVNULL, start_new_session=True)
        except OSError as exc:
            self._log(f"launch {daw.name}: {exc}")
            return "failed"
        # Reaped off the hub's threads: `wine` returns at once, REAPER not at all.
        threading.Thread(target=child.wait, daemon=True).start()
        self._launched_at[daw_id] = now
        self._log(f"launched {daw.name}")
        return "started"
