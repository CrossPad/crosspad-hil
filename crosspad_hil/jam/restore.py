"""What the board was before the jam, and putting it back -- even after a killed run.

The jam changes things the board keeps: the sequencer's scenes (saved to
``sequences.json`` by itself within ~500 ms of any change), the mixer desk
(NVS, on every preset load and when the Mixer closes), preset slots on the
card, the open audio-song project. ``HomeState.snapshot`` reads all of it --
and fetches the files -- before anything changes, and the snapshot goes into
the pins journal first, so a run that dies mid-way leaves a promise the next
run keeps (``restore_pending``).

Putting back goes in this order, each step on its own (one failing does not
stop the others), each reported in ``left``:

1. the sequencer and the song stop; the click as it was;
2. the kit (by name: provisioning can renumber kits) and the app;
3. the desk: the home desk (saved as slot 6 before the jam) loaded -- which
   writes it to NVS again -- then the slot files put back as they were (or
   removed when they did not exist);
4. the audio-song project: the jam's deleted (only when it is the one open),
   the home project opened again;
5. ``sequences.json``: once the sequencer reports ``dirty=0``, the copy taken
   before the jam is written back and the board is restarted so the
   sequencer reads it (this also clears what lives only in RAM: swing, the
   codec volume a SysEx moved, the preset undo);
6. the board's state compared with the snapshot.
"""
from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from crosspad_hil import pins, verbs
from crosspad_hil.errors import ENV, HilError

PIN_KEY = "jam_home"
PROJECT = "hil_live_jam"
# What the board answers FS_GET for a file that is not on the card: stat() (or open())
# -> ENOENT (file_transfer.cpp). Anything else -- a timeout, a stalled pull, another
# errno -- is not proof of absence.
NOT_THERE = ("FS_ERR stat_2", "FS_ERR open_2")


@dataclass
class HomeState:
    seq: dict[str, Any] = field(default_factory=dict)
    song: dict[str, Any] = field(default_factory=dict)
    click: dict[str, Any] = field(default_factory=dict)
    kit: dict[str, Any] = field(default_factory=dict)
    app: str | None = None
    mix_slots: dict[str, Any] = field(default_factory=dict)
    mix_dir: str = ""
    mix_desk: dict[str, Any] = field(default_factory=dict)
    project: str | None = None
    route: dict[str, Any] = field(default_factory=dict)
    files: dict[str, str | None] = field(default_factory=dict)   # remote -> local copy | None
    seq_file: str = ""
    scenes0: int = 0
    usb_guard: bool | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @staticmethod
    def from_dict(d: dict[str, Any]) -> HomeState:
        h = HomeState()
        for k, v in (d or {}).items():
            if hasattr(h, k):
                setattr(h, k, v)
        return h


def _try(fn: Any, default: Any = None) -> Any:
    try:
        return fn()
    except HilError:
        return default


def snapshot(link: Any, *, log: Any = print, need_file: bool = True) -> HomeState:
    """Everything the jam may change, read over CDC (no files yet: see ``fetch_files``).

    The sequencer read is not optional: ``scenes0`` is the line between the user's
    scenes and the jam's, and putting back deletes every scene from it upward. A
    failed read raises (the jam stops before it changes anything) instead of
    guessing 0 and later deleting the user's scenes."""
    h = HomeState()
    h.seq = verbs.seq(link, timeout_s=10.0) or {}
    if "scenes" not in h.seq:
        raise HilError(ENV, f"SEQ gave no scene count ({h.seq!r}): cannot tell the user's "
                            "scenes from the jam's", hint="is the sequencer up? retry")
    h.seq_file = str(h.seq.get("file") or "")
    if not h.seq_file or h.seq_file == "-":
        # Before the sequence store has read the card, SEQ shows the empty sequencer
        # it booted with (1 scene) -- not the user's scenes, and no file to copy.
        # The store reads the card only once a kit has been loaded after the boot.
        if need_file:
            raise HilError(ENV, "the sequencer has not loaded its file from the card yet "
                                f"({h.seq!r})", hint="load a kit after a boot, then retry")
        h.seq = {}                      # nothing to compare: the card's file is checked instead
    h.scenes0 = int(h.seq.get("scenes") or 0)
    h.song = _try(lambda: verbs.song_text(link), {}) or {}
    h.click = _try(lambda: verbs.click(link), {}) or {}
    ks = _try(lambda: verbs.kit_status(link), {}) or {}
    h.kit = {"id": ks.get("current"), "name": ks.get("name")}
    h.app = (_try(lambda: verbs.app_list(link), {}) or {}).get("running")
    pr = _try(lambda: verbs.mix_preset(link), {}) or {}
    h.mix_slots = {str(k): v for k, v in (pr.get("slots") or {}).items()}
    h.mix_dir = str(pr.get("dir") or "")
    h.mix_desk = _try(lambda: verbs.mix_lvl(link), {}) or {}
    a = _try(lambda: verbs.asong(link), {}) or {}
    h.project = a.get("project")
    h.usb_guard = _try(lambda: verbs.usb_guard(link))
    return h


def wait_seq_store(link: Any, timeout_s: float = 90.0, *, sleep: Any = time.sleep,
                   clock: Any = time.monotonic) -> dict[str, Any]:
    """SEQ once the sequence store has read the card (``file=`` set); ENV past the timeout.

    After a boot the card mounts in ~15 s and the store loads after it; until then SEQ
    reports the empty sequencer the board booted with."""
    deadline = clock() + timeout_s
    st: dict[str, Any] = {}
    while True:
        st = _try(lambda: verbs.seq(link, timeout_s=10.0), {}) or {}
        if str(st.get("file") or "") not in ("", "-"):
            return st
        if clock() >= deadline:
            raise HilError(ENV, f"the sequencer did not load its file within {timeout_s:.0f} s "
                                f"({st!r})", hint="is the SD card in, with sequences.json?")
        sleep(1.0)


def preset_path(home: HomeState, slot: int) -> str:
    return f"{home.mix_dir.rstrip('/')}/preset_{slot}.json"


def fetch_files(home: HomeState, pull: Any, dest: Path, slots: tuple[int, ...] = (4, 5, 6)
                ) -> None:
    """Copies of the files the jam will overwrite. A file the board reports as not on
    the card (``FS_ERR stat_2``, ENOENT) is recorded as None (remove it when putting back);
    any other failure raises -- a timeout is not proof the file is absent, and
    recording it as absent would delete the user's file when putting back."""
    dest.mkdir(parents=True, exist_ok=True)
    wanted = [preset_path(home, s) for s in slots] if home.mix_dir else []
    if home.seq_file and home.seq_file not in ("-", ""):
        wanted.append(home.seq_file)
    for remote in wanted:
        local = dest / remote.strip("/").replace("/", "__")
        try:
            pull(remote, local)
            home.files[remote] = str(local)
        except HilError as e:
            if not str(e.details.get("reply") or "").startswith(NOT_THERE):
                raise
            home.files[remote] = None


def journal(device_id: str, home: HomeState, path: Path | None = None) -> None:
    pins.record(device_id, PIN_KEY, home.to_dict(), path)


def unjournal(device_id: str, path: Path | None = None) -> None:
    pins.clear(device_id, PIN_KEY, path)


def _seq_quiet(link: Any, timeout_s: float = 5.0) -> bool:
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        st = _try(lambda: verbs.seq(link, timeout_s=10.0), {}) or {}
        if int(st.get("dirty", 1) or 0) == 0:
            return True
        time.sleep(0.25)
    return False


def put_back_ram(link: Any, home: HomeState, *, jam_scenes: list[int] | None = None,
                 kit_id_by_name: Any = None, log: Any = print) -> dict[str, str]:
    """Steps 1-4 over one CDC link. ``kit_id_by_name(name) -> id`` resolves the home kit."""
    left: dict[str, str] = {}

    def step(name: str, fn: Any) -> None:
        try:
            r = fn()
            left[name] = "restored" if r is not False else "unchanged"
        except HilError as e:
            left[name] = f"{e.code}: {e.message}"
            log(f"put back {name}: {e.code}: {e.message}")

    step("seq_stop", lambda: verbs.seq(link, "STOP", timeout_s=10.0))
    step("song_stop", lambda: verbs.asong(link, "STOP", timeout_s=10.0))
    if home.click and not int(home.click.get("running") or 0):
        step("click", lambda: verbs.click(link, 0))
    # Without the sequencer as it was, scenes0 means nothing: delete no scene at all.
    for s in sorted(jam_scenes or [], reverse=True) if "scenes" in home.seq else []:
        def delete(s: int = s) -> Any:
            st = verbs.seq(link, timeout_s=10.0)
            if int(st.get("scenes") or 0) <= s or s < home.scenes0:
                return False                       # never a scene that was there before
            return verbs.seq(link, f"DEL {s}", timeout_s=10.0)
        step(f"scene_{s}", delete)
    for key, cmd in (("bpm", "BPM"), ("quant", "QUANT"), ("countin", "COUNTIN")):
        if key in home.seq:
            v = home.seq[key]
            step(key, lambda cmd=cmd, v=v: verbs.seq(link, f"{cmd} {int(round(float(v)))}",
                                                     timeout_s=10.0))
    if home.song.get("blocks") is not None:
        def song() -> Any:
            verbs.song_text(link, "CLEAR")
            for sc, rep in home.song.get("blocks") or []:
                verbs.song_text(link, f"ADD {sc} {rep}")
            return verbs.song_text(link, f"LOOP {int(home.song.get('loop') or 0)}")
        step("song", song)
    if home.kit.get("name") and kit_id_by_name is not None:
        def kit() -> Any:
            kid = kit_id_by_name(home.kit["name"])
            if kid is None:
                raise HilError("ENV", f"no kit named {home.kit['name']!r} any more")
            if verbs.kit_status(link).get("current") == kid:
                return False
            return verbs.kit_load(link, int(kid), wait_s=30.0)
        step("kit", kit)
    def app() -> Any:
        running = verbs.app_list(link).get("running")
        if running == home.app:
            return False
        if home.app is None:
            return verbs.app_stop(link)
        return verbs.app_start(link, str(home.app), wait_s=15.0)
    step("app", app)
    if "6" in {str(k) for k in home.mix_slots} or home.mix_dir:
        step("desk", lambda: verbs.mix_preset(link, "LOAD 6"))
    if home.project and home.project != PROJECT:
        def project() -> Any:
            st = verbs.asong(link, timeout_s=10.0)
            if st.get("project") == PROJECT:
                verbs.asong(link, "DELETE", timeout_s=10.0)
                time.sleep(1.0)
            return verbs.asong(link, f"OPEN {home.project}", timeout_s=10.0)
        step("project", project)
    _seq_quiet(link)
    return left


def put_back_files(home: HomeState, push: Any, delete: Any, *, log: Any = print
                   ) -> dict[str, str]:
    """Step 5's files: each copy back where it was; a file that was not there, removed."""
    left: dict[str, str] = {}
    for remote, local in home.files.items():
        try:
            if local is None:
                delete(remote)
                left[remote] = "removed (was not there)"
            else:
                push(Path(local), remote)
                left[remote] = "restored"
        except HilError as e:
            left[remote] = f"{e.code}: {e.message}"
            log(f"put back {remote}: {e.code}: {e.message}")
    return left


def all_put_back(left: dict[str, str]) -> bool:
    """True when every step of ``put_back_ram``/``put_back_files`` succeeded."""
    return all(str(v) in ("restored", "unchanged") or str(v).startswith("removed")
               for v in left.values())


def board_diff(home: HomeState, now: HomeState) -> dict[str, Any]:
    """What still differs after putting back (empty: the board is as it was)."""
    out: dict[str, Any] = {}
    for key in ("scenes", "bpm", "quant", "countin"):
        a, b = home.seq.get(key), now.seq.get(key)
        if a is not None and b is not None and str(a) != str(b):
            out[f"seq.{key}"] = [a, b]
    if home.kit.get("name") and home.kit.get("name") != now.kit.get("name"):
        out["kit"] = [home.kit.get("name"), now.kit.get("name")]
    if home.app != now.app:
        out["app"] = [home.app, now.app]
    if home.project and home.project != now.project:
        out["project"] = [home.project, now.project]
    return out


def pending_home(device_id: str, path: Path | None = None) -> HomeState | None:
    """The home state an unfinished jam left in the journal, if any."""
    pend = pins.pending(device_id, path).get(PIN_KEY)
    if not pend:
        return None
    return HomeState.from_dict(pend if isinstance(pend, dict) else json.loads(pend))


def restore_pending(link: Any, device_id: str, log: Any, *, pull: Any = None, push: Any = None,
                    delete: Any = None, kit_id_by_name: Any = None,
                    path: Path | None = None) -> dict[str, Any] | None:
    """A jam that never put the board back: do it now, from the journal."""
    pend = pins.pending(device_id, path).get(PIN_KEY)
    if not pend:
        return None
    home = HomeState.from_dict(pend if isinstance(pend, dict) else json.loads(pend))
    log("pins: a live_jam run did not put the board back -- doing it now")
    left = put_back_ram(link, home, jam_scenes=list(range(home.scenes0, 8)),
                        kit_id_by_name=kit_id_by_name, log=log)
    if push is not None and delete is not None:
        left.update(put_back_files(home, push, delete, log=log))
        if all_put_back(left):
            unjournal(device_id, path)        # the promise is kept only when it was kept
        else:
            log("pins: putting back was incomplete -- the journal stays for the next run")
    return left


def _ram_restorer(link: Any, value: Any) -> bool:
    """For ``pins.restore_pending`` (other scenarios' normalising step): the RAM-level
    part. The files need the device itself: the jam's next run puts them back."""
    home = HomeState.from_dict(value if isinstance(value, dict) else {})
    left = put_back_ram(link, home, jam_scenes=list(range(home.scenes0, 8)))
    return all(not str(v).startswith(("ERR", "NOT_", "TIMEOUT")) for v in left.values()) \
        and not home.files


pins.RESTORERS[PIN_KEY] = _ram_restorer
