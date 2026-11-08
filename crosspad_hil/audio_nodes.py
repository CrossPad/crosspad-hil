"""CrossPad UAC2 as two PipeWire nodes, created on hotplug in any order.

WirePlumber (0.4.17 / PipeWire 1.0.5 on this bench) enumerates the CrossPad
sound card fine when it is already there at start-up, but never creates a
device for it on hotplug: the card comes back under the same ALSA index every
time the USB profile flips (`usb-mode set audio`), and the udev add/change
events for it are dropped. Restarting WirePlumber picks the card up and tears
every client's streams down with it.

This module side-steps the session manager: it watches udev for the CrossPad
audio card and creates the two nodes straight in the PipeWire daemon through
`pw-cli -m create-node` (monitor mode keeps the client, and so the node,
alive), then destroys them when the card goes away. The nodes get stable,
human names — "CrossPad Microphone" and "CrossPad Output" — so a mixer or a
DAW sees the pads' capture path and the speaker path as two separate devices
whatever the ALSA index is today.

`install()` writes a `systemd --user` unit around `run()` and a WirePlumber
rule that disables the `alsa_card.usb-Espressif_Crosspad_Audio*` device, so a
WirePlumber restart cannot add a second, differently named copy.
"""
from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import sys
import time
from collections.abc import Callable, Iterable, Iterator
from dataclasses import dataclass, field
from pathlib import Path
from shlex import quote as shlex_quote

from crosspad_hil.errors import BAD_ARGS, HilError

USB_VID = "303a"
CARD_MODEL = "Crosspad_Audio"

MIC_NODE = "crosspad_mic"
OUT_NODE = "crosspad_out"
MIC_DESCRIPTION = "CrossPad Microphone"
OUT_DESCRIPTION = "CrossPad Output"

# systemd-logind grants the seat user its ACL on /dev/snd/controlC* a few tens
# of milliseconds after udev announces the card (35 ms measured); a node
# created before that fails to open the device.
ACCESS_TIMEOUT_S = 3.0
ACCESS_POLL_S = 0.05

UNIT_NAME = "crosspad-audio-nodes.service"
WP_RULE_NAME = "51-crosspad-audio.lua"

_CARD_RE = re.compile(r"/sound/card(\d+)$")


@dataclass
class UdevEvent:
    action: str
    devpath: str
    props: dict[str, str] = field(default_factory=dict)

    @property
    def card(self) -> int | None:
        m = _CARD_RE.search(self.devpath)
        return int(m.group(1)) if m else None


def parse_udev_monitor(lines: Iterable[str]) -> Iterator[UdevEvent]:
    """Turn `udevadm monitor -u -p` output into card-level events.

    Blocks are separated by blank lines; the header is
    `UDEV  [ts] <action> <devpath> (<subsystem>)`, the rest `KEY=value`.
    """
    header: tuple[str, str] | None = None
    props: dict[str, str] = {}
    for raw in lines:
        line = raw.rstrip("\n")
        if line.startswith("UDEV  ["):
            # `UDEV  [ts] <action> <devpath> (<subsystem>)`; the preamble's
            # "UDEV - the event which udev sends out…" line has no bracket.
            parts = line.split()
            header = (parts[2], parts[3]) if len(parts) >= 4 else None
            props = {}
        elif "=" in line and header is not None:
            key, _, value = line.partition("=")
            props[key.strip()] = value.strip()
        elif not line.strip() and header is not None:
            yield UdevEvent(header[0], header[1], props)
            header, props = None, {}
    if header is not None:
        yield UdevEvent(header[0], header[1], props)


def is_crosspad_card(props: dict[str, str]) -> bool:
    return (props.get("ID_VENDOR_ID", "").lower() == USB_VID
            and props.get("ID_MODEL") == CARD_MODEL)


def udev_props(syspath: str) -> dict[str, str]:
    out = subprocess.run(["udevadm", "info", "-q", "property", "-p", syspath],
                         capture_output=True, text=True, check=False).stdout
    return dict(line.partition("=")[::2] for line in out.splitlines() if "=" in line)


def existing_cards() -> list[int]:
    """CrossPad audio cards present right now (normally zero or one)."""
    cards: list[int] = []
    for path in sorted(Path("/sys/class/sound").glob("card[0-9]*")):
        props = udev_props(str(path))
        if is_crosspad_card(props) and props.get("SOUND_INITIALIZED") == "1":
            cards.append(int(path.name[len("card"):]))
    return cards


def wait_accessible(card: int, timeout_s: float = ACCESS_TIMEOUT_S,
                    sleep: Callable[[float], None] = time.sleep) -> bool:
    node = f"/dev/snd/controlC{card}"
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        if os.access(node, os.R_OK | os.W_OK):
            return True
        sleep(ACCESS_POLL_S)
    return False


def node_command(card: int, *, capture: bool) -> list[str]:
    """`pw-cli -m create-node …` for one direction of `hw:<card>,0`.

    The stream is stereo either way: which physical input it carries (mics,
    line in, a DAC bus, or both codecs summed) is the board's route preset.
    """
    factory = "api.alsa.pcm.source" if capture else "api.alsa.pcm.sink"
    name = MIC_NODE if capture else OUT_NODE
    description = MIC_DESCRIPTION if capture else OUT_DESCRIPTION
    media_class = "Audio/Source" if capture else "Audio/Sink"
    props = (f"{{ factory.name={factory} node.name={name} "
             f'node.description="{description}" media.class={media_class} '
             f'api.alsa.path="hw:{card},0" audio.channels=2 audio.position=[FL,FR] }}')
    return ["pw-cli", "-m", "create-node", "adapter", props]


class NodeSet:
    """The two pw-cli holders for one card; idle until `start()`."""

    def __init__(self, spawn: Callable[[list[str]], subprocess.Popen] | None = None) -> None:
        self._spawn = spawn or (lambda argv: subprocess.Popen(
            argv, stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL))
        self.card: int | None = None
        self._procs: list[subprocess.Popen] = []

    @property
    def running(self) -> bool:
        return bool(self._procs) and all(p.poll() is None for p in self._procs)

    def start(self, card: int) -> None:
        if self.running and self.card == card:
            return
        self.stop()
        self.card = card
        self._procs = [self._spawn(node_command(card, capture=True)),
                       self._spawn(node_command(card, capture=False))]

    def stop(self) -> None:
        for p in self._procs:
            if p.poll() is None:
                p.terminate()
        for p in self._procs:
            try:
                p.wait(timeout=2.0)
            except subprocess.TimeoutExpired:
                p.kill()
        self._procs = []
        self.card = None


DEFAULT_APPS = ("FL Studio",)

HOLD_DIR = "defaults-holds"     # under the lock dir, this user's alone (0700)
DEFAULT_KEYS = ("default.audio.sink", "default.audio.source")


def _hold_dir(lock_dir: Path | None) -> Path:
    from crosspad_hil.locks import default_lock_dir
    return (lock_dir if lock_dir is not None else default_lock_dir()) / HOLD_DIR


def hold_defaults(log: Callable[[str], None] | None = None, *, lock_dir: Path | None = None,
                  run_cmd: Callable[[list[str]], subprocess.CompletedProcess] | None = None,
                  pid: int | None = None) -> bool:
    """This process is about to switch a board into MIDI+UAC2 for a test: the
    desktop keeps its default source and sink until the process exits.

    Two things would move them otherwise, each time the card enumerates:
    `take_defaults` (which `defaults_held` now stops) and WirePlumber itself,
    which brings back a configured default the moment that node reappears --
    and `take_defaults` configured the CrossPad's last time. So the devices in
    use now are made the configured ones first; a CrossPad that already is the
    default stays it. True when this call took the hold.
    """
    if not sys.platform.startswith("linux"):
        return False
    pid = pid if pid is not None else os.getpid()
    d = _hold_dir(lock_dir)
    from crosspad_hil.locks import private_lock_dir
    try:
        private_lock_dir(d.parent)
        try:
            d.mkdir(mode=0o700)
        except FileExistsError:
            pass
    except (OSError, HilError):
        return False
    dfd = _open_private_dir(d)
    if dfd is None:
        return False
    try:
        fd = os.open(str(pid), os.O_CREAT | os.O_EXCL | os.O_WRONLY | os.O_NOFOLLOW,
                     0o600, dir_fd=dfd)
    except OSError:          # already held by this pid, a link, no access
        os.close(dfd)
        return False
    with os.fdopen(fd, "w") as f:
        f.write(json.dumps({"pid": pid, "ts": time.time()}))
    if pid == os.getpid():
        import atexit
        atexit.register(_drop_hold, dfd, str(pid))
    else:
        os.close(dfd)
    run_cmd = run_cmd or _run_quiet
    try:
        objects = _pw_dump(run_cmd)
        ids = _node_ids(objects)
        for name in _defaults(objects).values():
            if name in (MIC_NODE, OUT_NODE) or name not in ids:
                continue
            run_cmd(["wpctl", "set-default", str(ids[name])])
            if log:
                log(f"crosspad audio: desktop default {name} kept for this run")
    except (OSError, subprocess.SubprocessError):
        pass   # no PipeWire tools here: nothing to keep
    return True


def _drop_hold(dfd: int, name: str) -> None:
    try:
        os.unlink(name, dir_fd=dfd)
    except OSError:
        pass
    os.close(dfd)


def defaults_held(lock_dir: Path | None = None) -> bool:
    """True while a live process holds the desktop defaults (`hold_defaults`);
    the hold of a process that is gone is removed."""
    if not sys.platform.startswith("linux"):
        return False
    from crosspad_hil.locks import pid_alive
    dfd = _open_private_dir(_hold_dir(lock_dir))
    if dfd is None:
        return False
    held = False
    try:
        for name in os.listdir(dfd):
            try:
                fd = os.open(name, os.O_RDONLY | os.O_NOFOLLOW, dir_fd=dfd)
                with os.fdopen(fd) as f:
                    pid = int(json.loads(f.read())["pid"])
            except (OSError, ValueError, KeyError, TypeError):
                continue
            if pid_alive(pid):
                held = True
            else:
                try:
                    os.unlink(name, dir_fd=dfd)
                except OSError:
                    pass
    finally:
        os.close(dfd)
    return held


def _open_private_dir(d: Path) -> int | None:
    """`d` opened once, never through a link, as a directory of this user that
    nobody else may write into; every hold is made, read and removed relative
    to that descriptor, so the path cannot be swapped under it. None otherwise
    -- a hold there is not trusted, either way."""
    import stat
    try:
        dfd = os.open(str(d), os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
    except OSError:
        return None
    st = os.fstat(dfd)
    if st.st_uid != os.getuid() or st.st_mode & (stat.S_IWGRP | stat.S_IWOTH):
        os.close(dfd)
        return None
    return dfd


def _defaults(objects: list[dict]) -> dict[str, str]:
    """`default.audio.sink` / `.source` -> node name, from the "default" metadata."""
    out: dict[str, str] = {}
    for o in objects:
        if not str(o.get("type", "")).endswith("Metadata"):
            continue
        if (o.get("props") or {}).get("metadata.name") != "default":
            continue
        for entry in o.get("metadata") or []:
            value = entry.get("value")
            if entry.get("key") in DEFAULT_KEYS and isinstance(value, dict) and value.get("name"):
                out[entry["key"]] = str(value["name"])
    return out


def take_defaults(log: Callable[[str], None], *, sleep: Callable[[float], None] = time.sleep,
                  run_cmd: Callable[[list[str]], subprocess.CompletedProcess] | None = None,
                  apps: Iterable[str] = DEFAULT_APPS,
                  held: Callable[[], bool] | None = None) -> bool:
    """Make the two nodes the session defaults once PipeWire lists them, and
    move the named applications' open streams onto them.

    A DAW under Wine sees the pulse device list only at its own start-up; the
    one thing that follows a card plugged in later is "[Default input]" /
    "[Default output]" — and a stream it already opened stays where it was
    until retargeted, hence the per-application move. PipeWire falls back to
    another device by itself when the nodes go away, so nothing is undone.
    Not while a HIL run holds the defaults: its profile switches are not a
    card being plugged in, and the desktop's sound stays where it is.
    """
    if (held or defaults_held)():
        log("crosspad audio: a HIL run holds the desktop defaults, left as they are")
        return False
    run_cmd = run_cmd or _run_quiet
    deadline = time.monotonic() + ACCESS_TIMEOUT_S
    while time.monotonic() < deadline:
        objects = _pw_dump(run_cmd)
        ids = _node_ids(objects)
        if MIC_NODE in ids and OUT_NODE in ids:
            # wpctl wants object ids, not names; a name is accepted and ignored.
            run_cmd(["wpctl", "set-default", str(ids[MIC_NODE])])
            run_cmd(["wpctl", "set-default", str(ids[OUT_NODE])])
            moved = 0
            for node_id, is_input in _app_streams(objects, set(apps)):
                run_cmd(["pw-metadata", str(node_id), "target.object",
                         MIC_NODE if is_input else OUT_NODE])
                moved += 1
            log(f"crosspad audio: default source/sink -> CrossPad, {moved} stream(s) moved")
            return True
        sleep(ACCESS_POLL_S * 4)
    log("crosspad audio: nodes not listed in time, defaults untouched")
    return False


def _pw_dump(run_cmd: Callable[[list[str]], subprocess.CompletedProcess]) -> list[dict]:
    try:
        objects = json.loads(run_cmd(["pw-dump"]).stdout or "[]")
    except json.JSONDecodeError:
        return []
    return [o for o in objects if isinstance(o, dict)]


def _node_ids(objects: list[dict]) -> dict[str, int]:
    """node.name -> object id for every Node in a pw-dump."""
    ids: dict[str, int] = {}
    for o in objects:
        if str(o.get("type", "")).endswith("Node"):
            name = ((o.get("info") or {}).get("props") or {}).get("node.name")
            if name:
                ids[str(name)] = int(o["id"])
    return ids


def _app_streams(objects: list[dict], apps: set[str]) -> list[tuple[int, bool]]:
    """(node id, is capture) for every audio stream owned by one of `apps`."""
    out: list[tuple[int, bool]] = []
    for o in objects:
        if not str(o.get("type", "")).endswith("Node"):
            continue
        props = (o.get("info") or {}).get("props") or {}
        media = str(props.get("media.class", ""))
        if props.get("application.name") in apps and media.startswith("Stream/"):
            out.append((int(o["id"]), media.startswith("Stream/Input")))
    return out


def run(log: Callable[[str], None] = print, *, nodes: NodeSet | None = None,
        events: Iterable[UdevEvent] | None = None,
        wait: Callable[[int], bool] = wait_accessible,
        defaults: bool = False, apps: Iterable[str] | None = None,
        on_up: Callable[[Callable[[str], None]], bool] | None = None) -> None:
    """Keep the two nodes in step with the card, for as long as we live.

    `events` defaults to a live `udevadm monitor`; tests hand in a list.
    With `defaults`, every time the nodes come up they become the session's
    default source and sink and `apps`' streams move over (`on_up` overrides
    that hook for tests).
    """
    nodes = nodes or NodeSet()
    app_names = tuple(apps) if apps else DEFAULT_APPS
    up_hook = on_up or ((lambda log: take_defaults(log, apps=app_names)) if defaults else None)
    for card in existing_cards():
        if wait(card):
            nodes.start(card)
            log(f"crosspad audio card {card}: nodes up")
            if up_hook:
                up_hook(log)
    proc: subprocess.Popen | None = None
    if events is None:
        proc = subprocess.Popen(["udevadm", "monitor", "-u", "-p", "-s", "sound"],
                                stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True)
        assert proc.stdout is not None
        events = parse_udev_monitor(proc.stdout)
    try:
        for ev in events:
            card = ev.card
            if card is None:
                continue
            if ev.action == "remove" and nodes.card == card:
                nodes.stop()
                log(f"crosspad audio card {card}: gone, nodes down")
            elif (ev.action in ("add", "change") and is_crosspad_card(ev.props)
                  and ev.props.get("SOUND_INITIALIZED") == "1"):
                if nodes.running and nodes.card == card:
                    continue
                if wait(card):
                    nodes.start(card)
                    log(f"crosspad audio card {card}: nodes up")
                    if up_hook:
                        up_hook(log)
                else:
                    log(f"crosspad audio card {card}: no access to controlC{card}, skipped")
    finally:
        nodes.stop()
        if proc is not None and proc.poll() is None:
            proc.terminate()


# -- install ---------------------------------------------------------------

WP_RULE = """-- Written by `crosspad-hil audio nodes install`.
-- The CrossPad UAC2 card is exposed by crosspad-audio-nodes.service as
-- "CrossPad Microphone" / "CrossPad Output"; keep WirePlumber from adding
-- a second copy under its own name when it does notice the card.
table.insert(alsa_monitor.rules, {
  matches = {
    {
      { "device.name", "matches", "alsa_card.usb-Espressif_Crosspad_Audio*" },
    },
  },
  apply_properties = {
    ["device.disabled"] = true,
  },
})
"""


def unit_text(exe: str, take_defaults: bool = False, apps: Iterable[str] | None = None) -> str:
    flag = " --take-defaults" if take_defaults else ""
    if apps:
        flag += " --apps " + " ".join(shlex_quote(a) for a in apps)
    return f"""[Unit]
Description=CrossPad UAC2 as PipeWire nodes (hotplug watcher)
After=pipewire.service wireplumber.service
Wants=pipewire.service

[Service]
ExecStart={exe} audio nodes run{flag}
Restart=on-failure
RestartSec=2

[Install]
WantedBy=default.target
"""


def config_paths(home: Path | None = None) -> tuple[Path, Path]:
    home = home or Path.home()
    return (home / ".config/systemd/user" / UNIT_NAME,
            home / ".config/wireplumber/main.lua.d" / WP_RULE_NAME)


def install(exe: str | None = None, *, home: Path | None = None,
            systemctl: Callable[[list[str]], None] | None = None,
            take_defaults: bool = False, apps: Iterable[str] | None = None) -> dict:
    exe = exe or shutil.which("crosspad-hil") or sys.argv[0]
    if not exe or not Path(exe).is_absolute():
        raise HilError(BAD_ARGS, f"cannot resolve an absolute crosspad-hil path from {exe!r}",
                       hint="pass --exe /full/path/to/crosspad-hil")
    unit, rule = config_paths(home)
    unit.parent.mkdir(parents=True, exist_ok=True)
    rule.parent.mkdir(parents=True, exist_ok=True)
    unit.write_text(unit_text(exe, take_defaults, apps))
    rule.write_text(WP_RULE)
    run_systemctl = systemctl or _systemctl_user
    run_systemctl(["daemon-reload"])
    run_systemctl(["enable", "--now", UNIT_NAME])
    return {"unit": str(unit), "wireplumber_rule": str(rule), "exe": exe}


def uninstall(*, home: Path | None = None,
              systemctl: Callable[[list[str]], None] | None = None) -> dict:
    unit, rule = config_paths(home)
    run_systemctl = systemctl or _systemctl_user
    if unit.exists():
        run_systemctl(["disable", "--now", UNIT_NAME])
        unit.unlink()
    if rule.exists():
        rule.unlink()
    run_systemctl(["daemon-reload"])
    return {"unit": str(unit), "wireplumber_rule": str(rule)}


def _systemctl_user(args: list[str]) -> None:
    subprocess.run(["systemctl", "--user", *args], check=True)


def status() -> dict:
    active = subprocess.run(["systemctl", "--user", "is-active", UNIT_NAME],
                            capture_output=True, text=True, check=False).stdout.strip()
    nodes = _pipewire_node_names()
    return {"service": active or "unknown",
            "cards": existing_cards(),
            "mic": MIC_NODE in nodes, "out": OUT_NODE in nodes}


def _run_quiet(argv: list[str]) -> subprocess.CompletedProcess:
    return subprocess.run(argv, capture_output=True, text=True, check=False)


def _pipewire_node_names(run_cmd: Callable[[list[str]], subprocess.CompletedProcess] | None = None
                         ) -> set[str]:
    run_cmd = run_cmd or _run_quiet
    out = run_cmd(["pw-dump"]).stdout or ""
    return set(re.findall(r'"node\.name":\s*"([^"]+)"', out))
