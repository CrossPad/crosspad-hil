"""daw_launcher: which DAWs the hub may offer, which run, and starting one."""
from __future__ import annotations

from pathlib import Path

from crosspad_hil import daw_launcher as dl

FL = dl.Daw(1, "FL Studio", ("/usr/bin/wine", "C:/FL/FL64.exe"), {"WINEPREFIX": "/p"},
            frozenset({"fl64.exe"}))
REAPER = dl.Daw(3, "REAPER", ("/opt/REAPER/reaper",), {}, frozenset({"reaper"}))
DISPLAY = {"DISPLAY": ":1", "PATH": "/usr/bin"}


class FakeChild:
    def wait(self) -> int:
        return 0


class Clock:
    def __init__(self) -> None:
        self.t = 1000.0

    def __call__(self) -> float:
        return self.t


def _launcher(*, running: set[str] | None = None, environ: dict | None = None,
              manager: dict | None = None, calls: list | None = None, clock: Clock | None = None):
    running = set() if running is None else running
    calls = [] if calls is None else calls

    def popen(cmd, **kw):
        calls.append((cmd, kw))
        return FakeChild()

    return dl.DawLauncher(find=lambda: {1: FL, 3: REAPER}, running=lambda: set(running),
                          environ=lambda: dict(DISPLAY if environ is None else environ),
                          manager_env=lambda: dict(manager or {}), popen=popen,
                          clock=clock or Clock(), log=lambda _: None)


def test_masks_are_bitmasks_by_daw_id() -> None:
    assert _launcher().masks() == (0b1010, 0)
    assert _launcher(running={"fl64.exe"}).masks() == (0b1010, 0b0010)


def test_nothing_is_launchable_without_a_display() -> None:
    launcher = _launcher(environ={"PATH": "/usr/bin"})
    assert launcher.masks() == (0, 0)
    assert launcher.launch(1) == "no_session"


def test_display_comes_from_the_service_manager_when_the_hub_has_none() -> None:
    calls: list = []
    launcher = _launcher(environ={"PATH": "/usr/bin"}, manager={"WAYLAND_DISPLAY": "wayland-0",
                                                                 "SHELL": "/bin/zsh"}, calls=calls)
    assert launcher.masks() == (0b1010, 0)
    assert launcher.launch(3) == "started"
    env = calls[0][1]["env"]
    assert env["WAYLAND_DISPLAY"] == "wayland-0" and "SHELL" not in env


def test_launch_starts_the_command_detached_with_its_prefix() -> None:
    calls: list = []
    assert _launcher(calls=calls).launch(1) == "started"
    cmd, kw = calls[0]
    assert cmd == ["/usr/bin/wine", "C:/FL/FL64.exe"]
    assert kw["env"]["WINEPREFIX"] == "/p" and kw["env"]["DISPLAY"] == ":1"
    assert kw["start_new_session"] is True


def test_a_running_daw_is_not_started_again() -> None:
    calls: list = []
    assert _launcher(running={"reaper"}, calls=calls).launch(3) == "running"
    assert calls == []


def test_a_second_press_within_the_guard_is_ignored() -> None:
    calls: list = []
    clock = Clock()
    launcher = _launcher(calls=calls, clock=clock)
    assert launcher.launch(1) == "started"
    clock.t += dl.RELAUNCH_GUARD_S - 1
    assert launcher.launch(1) == "recent"
    clock.t += 2
    assert launcher.launch(1) == "started"
    assert len(calls) == 2


def test_unknown_daw_is_refused() -> None:
    assert _launcher().launch(2) == "unknown"


def test_wine_command_picks_the_prefix_runner(tmp_path: Path) -> None:
    exe = Path("x/FL64.exe")
    which = {"flatpak": "/usr/bin/flatpak", "bottles-cli": "/usr/bin/bottles-cli",
             "wine": "/usr/bin/wine"}.get
    flat = tmp_path / ".var/app/com.usebottles.bottles/data/bottles/bottles/FL-Studio"
    flat.mkdir(parents=True)
    (flat / "bottle.yml").write_text("Arch: win64\nName: FL Studio\nPath: FL-Studio\n")
    cmd, env = dl.wine_command(flat / "drive_c", exe, which)
    assert cmd == ("/usr/bin/flatpak", "run", "--command=bottles-cli", "com.usebottles.bottles",
                   "run", "-b", "FL Studio", "-e", str(exe)) and env == {}
    native = tmp_path / ".local/share/bottles/bottles/Live"
    native.mkdir(parents=True)
    cmd, _ = dl.wine_command(native / "drive_c", exe, which)
    assert cmd[:4] == ("/usr/bin/bottles-cli", "run", "-b", "Live")
    cmd, env = dl.wine_command(tmp_path / ".wine/drive_c", exe, which)
    assert cmd == ("/usr/bin/wine", str(exe)) and env == {"WINEPREFIX": str(tmp_path / ".wine")}
    assert dl.wine_command(flat / "drive_c", exe, lambda _: None) is None


def test_find_fl_and_ableton_in_a_prefix(tmp_path: Path) -> None:
    drive_c = tmp_path / ".wine/drive_c"
    fl = drive_c / "Program Files/Image-Line/FL Studio 2026/FL64.exe"
    live = drive_c / "ProgramData/Ableton/Live 12 Suite/Program/Ableton Live 12 Suite.exe"
    for p in (fl, live):
        p.parent.mkdir(parents=True)
        p.write_bytes(b"")
    which = {"wine": "/usr/bin/wine"}.get
    assert dl.find_fl(drive_c, which).command == ("/usr/bin/wine", str(fl))
    ab = dl.find_ableton([drive_c], which)
    assert ab.daw_id == 2 and ab.process_names == frozenset({"ableton live 12 suite.exe"})


def test_find_reaper_on_path_or_in_opt(tmp_path: Path) -> None:
    exe = tmp_path / "reaper"
    exe.write_text("")
    exe.chmod(0o755)
    assert dl.find_reaper(lambda _: None, [str(exe)]).command == (str(exe),)
    assert dl.find_reaper(lambda _: None, [str(tmp_path / "none")]) is None


def test_running_names_read_argv0_including_windows_paths(tmp_path: Path) -> None:
    for pid, argv in (("10", b"C:\\Program Files\\Image-Line\\FL Studio 2026\\FL64.exe\0"),
                      ("11", b"/opt/REAPER/reaper\0-nosplash\0"), ("12", b"")):
        (tmp_path / pid).mkdir()
        (tmp_path / pid / "cmdline").write_bytes(argv)
    (tmp_path / "self").mkdir()
    assert dl.running_argv0(tmp_path) == {"fl64.exe", "reaper"}
