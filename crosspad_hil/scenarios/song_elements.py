"""Nested song elements, played on the screen: song -> element -> element, and back.

An element is a song inside a song (core `ElementNav`, Arrange's CLIP EDIT enters
one, a power click leaves it). This run builds the three-level song the design asks
for (spec 2026-10-24 §12) the way a person does and then judges the three promises
the feature makes: what you hear inside the element is what you edit, going back up
bakes it without a gap, and a cold cache costs time, never sound.

It drives the screen (`UI_TREE` + `TOUCH_TAP`, the encoder, the pads, the power
button) and reads results with state verbs only: `ASONG`, `ASONG_DBG` (`seg=` the
segments per track, `ctx=` the context stem: segments / descriptor open / solo),
`ASONG_SEQ`, song.json and the card's listings.

The steps:

  1  a new song (Arrange -> the songs -> NEW)
  2  an audio track recorded (ADD TRACK -> AUDIO, REC, REC)
  3  a kit track recorded (ADD TRACK -> KITS -> PERF -> a kit, REC, pads, REC)
  4  element from region (EDIT -> loop region -> Make element from region)
  4b another audio track recorded over it: the song around the element
  5  CLIP EDIT enters it: the song around it is a context stem
  6  element from region once more, inside it, and in: a third level
  7  the pads play the element's kit
  8  REC changes the element's notes
  9  two power clicks up, each level kept and baked
 10  PLAY at once: sound within one ring prime, no ring underrun
 11  the block's bake is another file than it was
 12  in again: SOLO silences the context stem, and turned off brings it back
 13  a second song places the element; an edit inside asks "used in 1 other place",
     Make a copy forks it and leaves the first song's block alone
 14  a cold cache (the run's bakes deleted): the block says "no bake" and the baker
     brings it back
 15  and the song plays at once

The speakers stay at `OUTVOL 2 <outvol>` from the first step to the last: nothing
here is meant to be loud. The run's own songs, elements, bakes and take go off the
card at the end (only what it made: the listings are taken before and compared
after), the kit that was loaded comes back and the song that was open is reopened.
The picker of Arrange lists 24 songs, so a "Song N" on a full card (and the song
that was open before) can be reached only by `ASONG OPEN`, and its folder goes with
the file verbs: both are noted where they happen.
"""
from __future__ import annotations

import json
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from crosspad_hil import audio_state, fs, verbs
from crosspad_hil.cdc import CdcLink
from crosspad_hil.errors import ENV, HilError
from crosspad_hil.scenarios.base import Artifact, Context, Progress, Report, register
from crosspad_hil.ui import TILE_REC_X, Ui, UiMissing

SONGS_ROOT = "/sdcard/SONGS"
ELEMENTS_DIR = f"{SONGS_ROOT}/.elements"
BAKE_DIR = f"{SONGS_ROOT}/.bake"
RECORDINGS = "/sdcard/crosspad/recordings"
ELEMENT_PATH = ".elements/"      # ASONG project= inside an element
ELEMENT_TITLE = "ELEMENT"        # what an element's block and crumb say before it is named

# Arrange's top bar tiles are 38 wide on a 40 pitch: REC from 124, the mode tile, PLAY
# (ui.TILE_PLAY_X, 200, lands on the mode tile in it and toggles REC/DUB instead).
TILE_W = 38
TILE_PITCH = 40
TILE_REC_LEFT = 124
TILE_PLAY_X = TILE_REC_LEFT + 2 * TILE_PITCH + TILE_W // 2
DEFAULT_REGION = (0, 4)  # what "Set a loop region" offers on a song with none
REGION = (1, 3)          # the song's element: two bars with the take around it, the context
INNER_REGION = (0, 2)    # inside it: the whole element
TIMELINE_Y = 66          # blocks are below the context row, the crumbs above
VIEW_X = (88, 302)       # the part of the timeline a finger reaches
BUSY_WORDS = ("COPYING", "KEEPING", "BAKING", "FREEZING", "LINKING")
BAKE_MISSING = "no bake"
QUESTION = "used in"

EDIT_DONE = 2            # ASONG edit=: 0 idle, 1 running, 2 done
PLAYING = 1
PAD_GAP_S = 0.5          # a pad hit a beat apart: four notes in one bar
COUNT_IN_S = 2.5         # REC's count-in on the song's grid, waited out before the pads
ASK_S = 8.0              # one ask of the board that is only a CDC round trip
PLAY_START_S = 3.0       # PLAY to a moving position: a ring prime and two CDC round trips
UNDER_WATCH_S = 3.0      # played this long without a ring underrun
POLL_S = 0.25
POWER_SETTLE_S = 0.8     # a power click's redraw
TILE_SETTLE_S = 0.3      # a top-bar tap's
MENU_SETTLE_S = 0.6      # an overlay opening
KNOB_SETTLE_S = 0.4      # an encoder turn or click
BLOCK_SETTLE_S = 0.8     # a block selected, its keys changed
JOB_START_S = 1.0        # an element job's progress box up after the tap that started it
LIST_READ_S = 1.5        # the library list read off the card
LIST_TRIES = 5           # an FS_LIST asked again while the card is busy
LIST_RETRY_S = 2.0
PAD_RING_S = 0.2         # between pads hit for the engine count
ENGINE_SETTLE_S = 0.5    # the engine's counter after the last pad
SOLO_SETTLE_S = 1.0      # a SOLO tap heard in the stem's descriptor
OPEN_WAIT_S = 6.0        # an app opening from the home map
NEW_SONG_WAIT_S = 10.0   # NEW opening an empty song
SONG_REOPEN_WAIT_S = 15.0
LAST_NOTE_S = 1.0        # after the last pad, before REC ends the take
MOVING_POS_S = 0.05      # a song position past this is a playing song
CLICK_WAIT_S = 3.0       # a power click that changes nothing was spent on the focus
UP_CLICKS = 4            # clicks one level may take: the focus, a swallowed one, the leave


@dataclass
class Params:
    kit: str = field(default="HIL SINE A", metadata={"help": "the kit of the kit track"})
    outvol: int = field(default=15, metadata={
        "help": "OUTVOL 2 level (speakers and phones) for the whole run"})
    take_s: float = field(default=9.0, metadata={"help": "seconds of the audio take"})
    hits: int = field(default=4, metadata={"help": "pads hit into each take"})
    keep: bool = field(default=False, metadata={
        "help": "leave the songs, elements, bakes and take on the card"})
    job_timeout_s: float = field(default=60.0, metadata={
        "help": "for an element job (copy, bake, freeze) or the baker to finish"})


# --------------------------------------------------------------------------- host side

def element_blocks(song: dict[str, Any]) -> list[dict[str, Any]]:
    """The blocks of a song.json that are elements, track by track: `file` is the bake
    they play, `element` the library path they stand for."""
    return [c for t in song.get("tracks", []) for c in t.get("clips", []) if c.get("element")]


def new_names(before: list[str], after: list[str]) -> list[str]:
    """What @p after has that @p before had not, sorted: what a run made."""
    return sorted(set(after) - set(before))


def is_new_song(project: str, before: list[str]) -> bool:
    """A project the run made: one the card did not list before it. A NEW that timed out
    leaves the user's own song open, and that one is never the run's to delete."""
    return bool(project) and project not in before


def seq_shape(blocks: list[dict[str, Any]]) -> list[tuple[int, int, int, int]]:
    """A seq track's blocks as (bar, bars, from, notes): two takes differ here."""
    return [(int(b["bar"]), int(b["bars"]), int(b["from"]), int(b["notes"])) for b in blocks]


def says_busy(labels: list[str]) -> bool:
    """An element job's progress box is up: its stage words are on the screen."""
    return any(w in label for label in labels for w in BUSY_WORDS)


# --------------------------------------------------------------------------- the run

class _Stop(Exception):
    """A step the rest of the run depends on failed: report what was checked so far."""


class _Run:
    """One pass: the link, the screen, what has been checked and what to undo."""

    def __init__(self, ctx: Context, params: Params) -> None:
        self.ctx = ctx
        self.p = params
        self.link: CdcLink | None = None
        self.data: dict[str, Any] = {"steps": {}}
        ctx.partial = self.data
        self.failures: list[str] = []
        self.artifacts: list[Artifact] = []
        self._ui: Ui | None = None
        self.audio_snap: dict[str, Any] | None = None
        self.song0 = ""                 # the song open before the run
        self.kit0 = -1                  # the kit loaded before the run
        self.songs: list[str] = []      # the run's own songs, in the order made
        self.before: dict[str, list[str]] = {}
        self.element = ""               # the first element's library path
        self.inner = ""                 # the element made inside it
        self.bake0 = ""                 # the song's block of it, as first baked
        self.title = ""                 # what the crumb and the library call it
        self.rec_ok = False

    # ----------------------------------------------------------------- the hands

    @property
    def ui(self) -> Ui:
        if self._ui is None:
            self._ui = Ui(self.link, wait=self.ctx.cancelled.wait)
        self._ui.link = self.link             # the link changes across pulls
        return self._ui

    def wait(self, s: float) -> None:
        self.ctx.cancelled.wait(s)

    def power(self) -> None:
        verbs.pwr_gesture(self.link)
        self.wait(POWER_SETTLE_S)

    def hit(self, pad: int, vel: int = 100) -> None:
        verbs.pad_press(self.link, pad, vel)
        verbs.pad_release(self.link, pad)

    def labels(self) -> list[str]:
        return self.ui.screen().labels()

    def until(self, ask: Callable[[], Any], done: Callable[[Any], bool],
              timeout_s: float | None = None) -> Any:
        deadline = time.monotonic() + (timeout_s or ASK_S)
        got = ask()
        while not done(got) and time.monotonic() < deadline and not self.ctx.cancelled.is_set():
            self.wait(POLL_S)
            got = ask()
        return got

    def song(self) -> dict[str, Any]:
        return verbs.asong(self.link)

    def project(self) -> str:
        return str(self.song().get("project", ""))

    def home(self) -> None:
        """Power clicks back to the home map."""
        for _ in range(8):
            sc = self.ui.screen()
            if sc.running == "-" and sc.find("Kits") and sc.find("Sampler"):
                return
            self.power()
        raise _Stop("power clicks did not bring the home map back: "
                    + ", ".join(self.labels()[:20]))

    def open_arrange(self) -> None:
        sc = self.ui.screen()
        if sc.running == "Arrange":
            return
        self.home()
        self.ui.tap("Arrange")
        sc = self.until(self.ui.screen, lambda s: s.running == "Arrange", timeout_s=OPEN_WAIT_S)
        if sc.running != "Arrange":
            raise UiMissing("TIMEOUT", f"Arrange opened {sc.running}",
                            hint="could press: " + ", ".join(sc.labels()[:20]))

    def tile(self, x: int) -> None:
        self.ui.tap_tile(x)
        self.wait(TILE_SETTLE_S)

    def stop(self) -> None:
        """PLAY while the song plays: it stops."""
        if self.song().get("play") == PLAYING:
            self.tile(TILE_PLAY_X)
            self.until(self.song, lambda r: r.get("play") != PLAYING)

    def stem(self) -> dict[str, Any]:
        """ASONG_DBG once the context stem has segments: the baker makes it in the
        background after an enter, and on a cold card that takes a few seconds."""
        return self.until(lambda: verbs.asong_dbg(self.link), lambda d: d["ctx"][0] >= 1,
                          self.p.job_timeout_s)

    def wait_job(self, timeout_s: float | None = None) -> dict[str, Any]:
        """Until an element job, a take or a copy is over: the progress box gone for two
        looks running one after the other, the engine's edit job not running."""
        quiet = 0

        def calm() -> bool:
            nonlocal quiet
            st = self.song()
            idle = st.get("edit") != 1 and not says_busy(self.labels())
            quiet = quiet + 1 if idle else 0
            return quiet >= 2

        self.until(lambda: None, lambda _: calm(), timeout_s or self.p.job_timeout_s)
        return self.song()

    # ----------------------------------------------------------------- the screen

    def edit_menu(self) -> None:
        """EDIT on the timeline, or its "..." inside an element (EDIT is hidden there)."""
        if self.ui.screen().find("EDIT"):
            self.ui.tap("EDIT")
        else:
            self.ui.tap("...")

    def menu_item(self, text: str) -> None:
        """A row of the open Edit menu, scrolled to: the list is longer than the sheet."""
        self.ui.tap_node(self.ui.seek(text, contains=True))

    def element_block(self) -> Any:
        """The element block on the timeline, whatever it is called: the tappable object
        saying ELEMENT below the context row (the crumbs above it say it too)."""
        for n in self.ui.screen().nodes:
            if n.interactive and n.cls == "obj" and n.y >= TIMELINE_Y \
                    and n.text.startswith(ELEMENT_TITLE):
                return n
        raise UiMissing("TIMEOUT", "no element block on the timeline",
                        hint="on screen: " + ", ".join(self.labels()[:30]))

    def tap_block(self, node: Any) -> None:
        """A finger on the part of a block the timeline shows (its middle may be off screen)."""
        x0, x1 = max(node.x, VIEW_X[0]), min(node.x + node.w, VIEW_X[1])
        verbs.touch_tap(self.link, (x0 + x1) // 2, node.y + node.h // 2)
        self.wait(BLOCK_SETTLE_S)

    def set_region(self, start: int, end: int) -> str:
        """EDIT -> Set a loop region: the knob moves L from the default four bars, a click
        takes it, the knob moves R, a click takes that. The region the engine holds."""
        self.edit_menu()
        self.ui.tap("Set a loop region", contains=True)
        self.wait(MENU_SETTLE_S)
        for delta in (start - DEFAULT_REGION[0], end - DEFAULT_REGION[1]):
            if delta:
                verbs.enc_rotate(self.link, delta)
                self.wait(KNOB_SETTLE_S)
            verbs.enc_press(self.link)
            self.wait(KNOB_SETTLE_S)
        return str(self.song()["rgn"])

    def make_element(self) -> None:
        """The loop region as an element, and the job waited out."""
        self.edit_menu()
        self.menu_item("Make element from region")
        self.wait(JOB_START_S)
        self.wait_job()

    def enter(self) -> str:
        """The element block selected, CLIP EDIT: the library path the song engine is in now."""
        before = self.project()
        self.tap_block(self.element_block())
        self.ui.tap("CLIP EDIT")
        st = self.until(self.song, lambda r: str(r.get("project")) != before,
                        timeout_s=self.p.job_timeout_s)
        self.wait_job()
        return str(st["project"])

    def up(self, levels: int) -> str:
        """Power clicks out of @p levels elements. A click that changed nothing (the focus
        was spent on the first one) is repeated; it is never clicked at the top, where
        it would leave Arrange."""
        project = self.project()
        for _ in range(levels):
            level = project
            for _ in range(UP_CLICKS):
                if not level.startswith(ELEMENT_PATH):
                    break
                self.power()
                spent = time.monotonic() + CLICK_WAIT_S
                hard = time.monotonic() + self.p.job_timeout_s
                while time.monotonic() < hard and not self.ctx.cancelled.is_set():
                    project = self.project()
                    if project != level:
                        break
                    if time.monotonic() >= spent and not says_busy(self.labels()):
                        break
                    self.wait(POLL_S)
                if project != level:
                    break
            if project == level:
                raise UiMissing("TIMEOUT", f"power clicks did not leave {level}",
                                hint="on screen: " + ", ".join(self.labels()[:30]))
            self.wait_job()
        return project

    def record(self) -> list[tuple[int, int, int, int]]:
        """REC on the top bar, the count-in waited out, the pads hit, REC again: the kit
        track's blocks after the take (the song keeps playing)."""
        self.tile(TILE_REC_X)
        self.wait(COUNT_IN_S)
        for pad in range(self.p.hits):
            self.hit(pad)
            self.wait(PAD_GAP_S)
        self.wait(LAST_NOTE_S)
        self.tile(TILE_REC_X)
        self.wait_job()
        return seq_shape(verbs.asong_seq(self.link)["blocks"])

    def pull(self, remote: str, name: str) -> dict[str, Any]:
        """A song.json off the card over the raw port (the link is closed and opened again)."""
        local = self.ctx.workdir / name
        self.link.close()
        try:
            fs.pull(self.ctx.device, remote, local)
        finally:
            self.link = self.ctx.open_cdc()
        return json.loads(local.read_text(encoding="utf-8"))

    def entries(self, path: str) -> list[dict[str, Any]]:
        """A folder's entries, asked again while the card is busy (a bake running after the
        song was reopened): a listing that timed out was taken for an empty folder, and
        the cleanup left the run's songs on the card. A folder that is not there is none."""
        for attempt in range(LIST_TRIES):
            try:
                return list(verbs.fs_list(self.link, path)["entries"])
            except HilError as e:
                if "opendir" in e.message:
                    return []
                if attempt == LIST_TRIES - 1:
                    raise
                self.wait(LIST_RETRY_S)
        return []

    def listing(self, path: str) -> list[str]:
        return sorted(str(e["name"]) for e in self.entries(path))

    def song_file(self, song: str) -> str:
        return f"{SONGS_ROOT}/{song}/song.json"

    # ----------------------------------------------------------------- bookkeeping

    def step(self, n: int | str, ok: bool, checked: str, *, why: str = "",
             manual: str = "") -> bool:
        entry: dict[str, Any] = {"status": "pass" if ok else "fail", "checked": checked}
        if manual:
            entry["manual"] = manual
        if not ok:
            entry["why"] = why
            self.failures.append(f"step {n}: {why}")
        self.data["steps"][str(n)] = entry
        return ok

    def missing(self, n: int | str, what: str, e: UiMissing) -> None:
        """A control a person would press was not there: the step fails with the screen."""
        self.step(n, False, what, why=f"{e.message}; {e.hint}"[:400])

    def need(self, n: int | str, ok: bool, checked: str, why: str) -> None:
        """A step the rest cannot go on without."""
        if not self.step(n, ok, checked, why=why):
            raise _Stop(f"step {n}: {why}")

    # ----------------------------------------------------------------- setup

    def setup(self) -> None:
        self.link = self.ctx.open_cdc()
        if verbs.fs_roots(self.link)["roots"].get("/sdcard", {}).get("total") is None:
            raise HilError(ENV, "no SD card mounted on the board",
                           hint="FS_ROOTS shows /sdcard unmounted; put a FAT card in and reset")
        kits = verbs.kit_list(self.link)["kits"]
        if self.p.kit not in {str(k["name"]) for k in kits}:
            raise HilError(ENV, f"no kit {self.p.kit!r} on the card",
                           hint="KIT_LIST names them; --kit picks another")
        # Quiet before anything can sound.
        verbs.outvol(self.link, 2, self.p.outvol)
        self.data["outvol0"] = {k: v for k, v in verbs.outvol(self.link).items() if k != "line"}
        self.audio_snap = audio_state.snapshot(self.link)
        self.song0 = self.project()
        self.kit0 = int(verbs.kit_status(self.link)["current"])
        self.before = {"songs": self.listing(SONGS_ROOT), "elements": self.listing(ELEMENTS_DIR),
                       "bake": self.listing(BAKE_DIR), "recordings": self.listing(RECORDINGS)}
        self.data["before"] = {k: len(v) for k, v in self.before.items()}
        self.data["resets0"] = verbs.reset_count(self.link)

    # ----------------------------------------------------------------- 1-3: the song

    def new_song(self) -> None:
        """Arrange, the songs, NEW."""
        try:
            self.open_arrange()
            self.ui.tap(self.project())
            self.ui.tap("NEW")
            st = self.until(self.song, lambda r: r["open"] == 2
                            and is_new_song(str(r["project"]), self.before["songs"]),
                            timeout_s=NEW_SONG_WAIT_S)
        except UiMissing as e:
            self.missing(1, "Arrange -> the songs -> NEW", e)
            raise _Stop("no new song") from e
        self.need(1, is_new_song(str(st["project"]), self.before["songs"]) and st["open"] == 2,
                  "Arrange, the song's name, NEW: an empty song opens",
                  f"project={st['project']} open={st['open']} (the song open before: {self.song0})")
        self.songs.append(str(st["project"]))

    def record_audio(self, track: int) -> dict[str, Any]:
        """ADD TRACK -> AUDIO arms the new track on the mics; REC, the take, REC: the clip
        on @p track (0-based), once its job is done."""
        self.ui.tap("ADD TRACK", contains=True)
        self.ui.tap("AUDIO")
        self.tile(TILE_REC_X)
        self.wait(self.p.take_s)
        self.tile(TILE_REC_X)
        return self.until(self.song, lambda r: r["clips"][track] >= 1 and r["edit"] == EDIT_DONE,
                          timeout_s=self.p.job_timeout_s)

    def audio_take(self) -> None:
        try:
            st = self.record_audio(0)
        except UiMissing as e:
            self.missing(2, "ADD TRACK -> AUDIO, REC, REC", e)
            raise _Stop("no audio take") from e
        self.need(2, st["clips"][0] == 1 and st["len"] >= 4,
                  "ADD TRACK -> AUDIO, REC, wait, REC: a clip on track 1, four bars or more",
                  f"clips {st['clips']} len={st['len']}")

    def bed_take(self) -> None:
        """Another audio track recorded over the element: what plays around it, the song's
        context. The element took everything inside its bars from the tracks it was made
        of, so without this the stem it is heard against would be empty."""
        try:
            st = self.record_audio(3)
        except UiMissing as e:
            self.missing("4b", "ADD TRACK -> AUDIO, REC, REC over the element", e)
            raise _Stop("no bed under the element") from e
        self.need("4b", st["clips"][3] == 1,
                  "ADD TRACK -> AUDIO, REC, REC over the element: a clip on track 4, the context "
                  "the element is heard against",
                  f"clips {st['clips']}")

    def kit_take(self) -> None:
        """ADD TRACK -> KITS -> PERF -> the kit, PICK; REC, pads, REC; PLAY to stop."""
        try:
            self.ui.tap("ADD TRACK", contains=True)
            self.ui.tap("KITS")
            self.ui.tap("PERF")
            self.ui.tap_node(self.ui.seek(self.p.kit))
            self.ui.tap("PICK")
            self.until(lambda: verbs.kit_status(self.link),
                       lambda r: r["name"] == self.p.kit and not r["loading"],
                       timeout_s=self.p.job_timeout_s)
            blocks = self.record()
            self.stop()
        except UiMissing as e:
            self.missing(3, "ADD TRACK -> KITS -> PERF -> kit -> REC, pads, REC", e)
            raise _Stop("no kit take") from e
        notes = sum(b[3] for b in blocks)
        self.need(3, notes >= self.p.hits and self.song().get("play") != PLAYING,
                  f"ADD TRACK -> KITS -> PERF -> {self.p.kit}, REC, {self.p.hits} pads, REC: "
                  "the kit track holds the notes",
                  f"seq blocks {blocks}, notes {notes} of {self.p.hits}")

    # ----------------------------------------------------------------- 4-6: the elements

    def element_from_region(self) -> None:
        """A loop region, Make element from region: one element block, its bake on the card."""
        try:
            rgn = self.set_region(*REGION)
            self.make_element()
            block = self.element_block()
        except UiMissing as e:
            self.missing(4, "EDIT -> Set a loop region -> Make element from region", e)
            raise _Stop("no element made") from e
        song = self.pull(self.song_file(self.songs[0]), "song1_made.json")
        blocks = element_blocks(song)
        self.data["made"] = {"region": rgn, "title": block.text, "blocks": blocks}
        bakes = self.listing(BAKE_DIR)
        ok = rgn == f"{REGION[0]}-{REGION[1]}" and len(blocks) == 1 and blocks[0]["file"] in bakes
        self.need(4, ok, "EDIT -> loop region -> Make element from region: one element block "
                         "playing a bake that is on the card",
                  f"region {rgn}, element blocks {blocks}, "
                  f"bakes made {new_names(self.before['bake'], bakes)}")
        self.element = blocks[0]["element"]
        self.bake0 = blocks[0]["file"]

    def enter_element(self) -> None:
        """CLIP EDIT on the block: the engine is in the element, the song a context stem."""
        try:
            project = self.enter()
        except UiMissing as e:
            self.missing(5, "the block, CLIP EDIT", e)
            raise _Stop("did not enter the element") from e
        dbg = self.stem()
        self.data["entered"] = {"project": project, "ctx": dbg["ctx"], "seg": dbg["seg"]}
        crumbs = [lb for lb in self.labels() if lb.startswith(ELEMENT_TITLE) or lb == "SONG"]
        self.title = next((c for c in crumbs if c != "SONG"), "")
        self.need(5, project == self.element and dbg["ctx"][0] >= 1 and bool(self.title),
                  "CLIP EDIT: the engine is in the element, the crumbs say SONG, the context "
                  "stem has segments",
                  f"project {project} (want {self.element}), ctx {dbg['ctx']}, crumbs {crumbs}")

    def nested_element(self) -> None:
        """The same inside: a region, Make element from region, and in: the third level."""
        try:
            self.set_region(*INNER_REGION)
            self.make_element()
            project = self.enter()
        except UiMissing as e:
            self.missing(6, "inside the element: region, Make element from region, CLIP EDIT", e)
            raise _Stop("no third level") from e
        self.inner = project
        dbg = self.stem()
        self.data["nested"] = {"project": project, "ctx": dbg["ctx"]}
        self.need(6, project.startswith(ELEMENT_PATH) and project != self.element
                  and dbg["ctx"][0] >= 1,
                  "inside the element: Make element from region, CLIP EDIT: a third level "
                  "with its own context stem",
                  f"project {project} (the first element {self.element}), ctx {dbg['ctx']}")

    # ----------------------------------------------------------------- 7-8: play, edit

    def play_pads(self) -> None:
        """The pads play the element's kit."""
        e0 = int(verbs.smpl_last(self.link)["engine_total"])
        for pad in range(self.p.hits):
            self.hit(pad)
            self.wait(PAD_RING_S)
        self.wait(ENGINE_SETTLE_S)
        played = int(verbs.smpl_last(self.link)["engine_total"]) - e0
        self.step(7, played >= self.p.hits,
                  f"{self.p.hits} pads hit inside the element: the engine plays them",
                  why=f"{played} notes played of {self.p.hits}",
                  manual="the pads sound like the kit of the element's track")

    def change_notes(self) -> None:
        """REC, pads, REC: the element's kit track holds other notes. A take in Arrange is
        blocks of their own notes (ASONG_SEQ), not the scenes' events (SEQ_DUMP): those
        stay as they were, and the step passed on them only while an element was given
        the card's old shared scenes."""
        before = seq_shape(verbs.asong_seq(self.link)["blocks"])
        try:
            after = self.record()
            self.stop()
        except UiMissing as e:
            self.missing(8, "REC, pads, REC inside the element", e)
            return
        self.data["notes"] = {"before": before, "after": after}
        self.step(8, sum(b[3] for b in after) >= self.p.hits and after != before,
                  "REC, pads, REC inside the element: the kit track's notes are another take",
                  why=f"seq blocks (bar, bars, from, notes) {before} -> {after}")

    # ----------------------------------------------------------------- 9-11: up, play, bake

    def go_up(self) -> None:
        try:
            project = self.up(2)
        except UiMissing as e:
            self.missing(9, "two power clicks up", e)
            raise _Stop("did not get back to the song") from e
        self.need(9, project == self.songs[0] and not self.labels().count("SONG"),
                  "two power clicks (each waits for its KEEPING / BAKING): back in the song",
                  f"project {project} (want {self.songs[0]})")

    def play_at_once(self, n: int, what: str) -> None:
        """PLAY at the song's level at once: the position moves within one ring prime with
        the ring filled and nothing late, and keeps going without an underrun."""
        u0 = int(self.song().get("under", 0))
        t0 = time.monotonic()
        self.tile(TILE_PLAY_X)
        st = self.until(self.song, lambda r: r.get("play") == PLAYING
                        and float(r["pos"]) > MOVING_POS_S,
                        timeout_s=PLAY_START_S)
        started = time.monotonic() - t0
        dbg = verbs.asong_dbg(self.link)
        self.wait(UNDER_WATCH_S)
        late = verbs.asong_dbg(self.link)
        under = int(self.song().get("under", 0)) - u0
        self.stop()
        self.data.setdefault("play", {})[str(n)] = {
            "started_s": round(started, 2), "ringfill": dbg["ringfill"], "seg": dbg["seg"],
            "under": under, "pos": st["pos"]}
        ok = (st.get("play") == PLAYING and started <= PLAY_START_S and dbg["ringfill"] > 0
              and sum(dbg["seg"]) >= 1 and under == 0 and late["ringfill"] > 0)
        self.step(n, ok, what,
                  why=f"play={st.get('play')} after {started:.1f} s (limit {PLAY_START_S}), "
                      f"ringfill {dbg['ringfill']} -> {late['ringfill']}, segments {dbg['seg']}, "
                      f"{under} underruns",
                  manual="the song sounds at once, the element as it was last edited")

    def bake_changed(self) -> None:
        """The song's block of the element is not the first bake: the leave baked it."""
        song = self.pull(self.song_file(self.songs[0]), "song1_after.json")
        blocks = element_blocks(song)
        bakes = self.listing(BAKE_DIR)
        now = blocks[0]["file"] if blocks else ""
        self.data["rebaked"] = {"first": self.bake0, "now": now}
        self.step(11, len(blocks) == 1 and now != self.bake0 and now in bakes,
                  "the block's bake after the edit is another file, on the card",
                  why=f"blocks {blocks}; first bake {self.bake0}, "
                      f"bakes made {new_names(self.before['bake'], bakes)}")

    # ----------------------------------------------------------------- 12: SOLO

    def solo(self) -> None:
        """In again, PLAY: the context stem sounds; SOLO silences it; SOLO again brings it back."""
        try:
            self.enter()
            self.tile(TILE_PLAY_X)
            self.until(self.song, lambda r: r.get("play") == PLAYING)
            self.wait(SOLO_SETTLE_S)
            off = verbs.asong_dbg(self.link)["ctx"]
            self.ui.tap("SOLO", contains=True)
            self.wait(SOLO_SETTLE_S)
            on = verbs.asong_dbg(self.link)["ctx"]
            self.ui.tap("SOLO", contains=True)
            self.wait(SOLO_SETTLE_S)
            again = verbs.asong_dbg(self.link)["ctx"]
            self.stop()
            self.up(1)
        except UiMissing as e:
            self.missing(12, "in again, PLAY, SOLO, SOLO", e)
            return
        self.data["solo"] = {"off": off, "on": on, "off_again": again}
        # ctx= is (segments, descriptor open, solo): the stem is read while it is not solo.
        self.step(12, off[2] == 0 and off[1] == 1 and on[2] == 1 and on[1] == 0
                  and again[2] == 0 and again[1] == 1,
                  "in the element, PLAY: the stem is read; SOLO: it is not; SOLO again: it is",
                  why=f"ctx (segments/open/solo) SOLO off {off}, on {on}, off again {again}",
                  manual="SOLO on: only the element is heard; off: the song around it too")

    # ----------------------------------------------------------------- 13: copy or replace

    def copy_question(self) -> None:
        """A second song places the element; an edit inside asks; Make a copy forks it."""
        try:
            self.ui.tap(self.project())
            self.ui.tap("NEW")
            st = self.until(self.song, lambda r: r["open"] == 2
                            and is_new_song(str(r["project"]), self.before["songs"])
                            and r["project"] != self.songs[0], timeout_s=NEW_SONG_WAIT_S)
            if not is_new_song(str(st["project"]), self.before["songs"]):
                raise UiMissing("TIMEOUT", f"NEW opened {st['project']}, not a new song")
            self.songs.append(str(st["project"]))
            self.edit_menu()
            self.menu_item("Element from library")
            self.wait(LIST_READ_S)
            self.ui.tap(self.title, contains=True)
            self.wait(JOB_START_S)
            project = self.enter()
            self.tap_block(self.element_block())
            self.ui.tap("DUP")
            self.wait(JOB_START_S)
            self.power()
            deadline = time.monotonic() + self.p.job_timeout_s
            while not any(QUESTION in lb for lb in self.labels()) and time.monotonic() < deadline:
                self.wait(POLL_S)
            asked = next((lb for lb in self.labels() if QUESTION in lb), "")
            self.ui.tap("Make a copy")
            self.wait(JOB_START_S)
            self.wait_job()
        except UiMissing as e:
            self.missing(13, "a second song: Element from library, CLIP EDIT, DUP, power, "
                         "Make a copy", e)
            return
        second = self.pull(self.song_file(self.songs[1]), "song2.json")
        first = self.pull(self.song_file(self.songs[0]), "song1_end.json")
        forked = [b["element"] for b in element_blocks(second)]
        kept = [b["element"] for b in element_blocks(first)]
        elements = new_names(self.before["elements"], self.listing(ELEMENTS_DIR))
        self.data["copy"] = {"entered": project, "asked": asked, "second": forked,
                             "first": kept, "elements": elements}
        self.step(13, project == self.element and "1 other place" in asked
                  and forked and self.element not in forked and kept == [self.element]
                  and len(elements) == 3,
                  "second song: Element from library, CLIP EDIT, DUP inside, power click: "
                  "'used in 1 other place'; Make a copy: this song has a new element, the first "
                  "song's block is still the old one",
                  why=f"entered {project} (want {self.element}), asked {asked!r}, second song's "
                      f"blocks {forked}, first song's {kept}, elements made {elements}")

    # ----------------------------------------------------------------- 14: cold cache

    def cold_cache(self) -> None:
        """The run's bakes deleted, the first song opened again: its block says "no bake" until
        the baker has made the file, and then the song plays."""
        mine = new_names(self.before["bake"], self.listing(BAKE_DIR))
        for name in mine:
            verbs.fs_delete(self.link, f"{BAKE_DIR}/{name}")
        verbs.asong(self.link, f"OPEN {self.songs[0]}")      # not in the picker's 24
        saw = False
        deadline = time.monotonic() + self.p.job_timeout_s
        while time.monotonic() < deadline and not self.ctx.cancelled.is_set():
            labels = self.labels()
            saw = saw or any(BAKE_MISSING in lb or "BAKING" in lb for lb in labels)
            if saw and not any(BAKE_MISSING in lb for lb in labels) and not says_busy(labels):
                break
            self.wait(POLL_S)
        song = self.pull(self.song_file(self.songs[0]), "song1_cold.json")
        blocks = element_blocks(song)
        bakes = self.listing(BAKE_DIR)
        back = bool(blocks) and blocks[0]["file"] in bakes
        self.data["cold"] = {"deleted": mine, "saw_missing": saw, "block": blocks,
                             "bakes": new_names(self.before["bake"], bakes)}
        ok = saw and back
        self.step(14, ok, "the run's bakes deleted, the song opened: the block says 'no bake', "
                          "then its bake is back on the card",
                  why=f"{len(mine)} bakes deleted; 'no bake' seen={saw}; block {blocks} "
                      f"bake on the card={back}")
        if ok:
            self.play_at_once(15, "after the cold cache: PLAY at once, the ring fills, no underrun")

    # ----------------------------------------------------------------- cleanup

    def rm_tree(self, path: str) -> None:
        for e in self.entries(path):
            child = f"{path}/{e['name']}"
            if e["dir"]:
                self.rm_tree(child)
            else:
                verbs.fs_delete(self.link, child)
        verbs.fs_delete(self.link, path)

    def rm_new(self, root: str, key: str, what: str) -> None:
        """What @p root holds beyond its listing from before the run (tree by tree)."""
        for name in new_names(self.before[key], self.listing(root)):
            self.ctx.log(f"cleanup: {what} {name}")
            self.rm_tree(f"{root}/{name}")

    def reopen_song0(self) -> None:
        """The song that was open before the run, open again: before a song of the run
        is deleted, and before the kit goes back -- a kit loaded is written into the
        song that is open, a kept song of the run included."""
        if self.song0 and self.project() != self.song0:
            verbs.asong(self.link, f"OPEN {self.song0}")
            self.until(self.song, lambda r: r["project"] == self.song0 and r["open"] == 2,
                       timeout_s=SONG_REOPEN_WAIT_S)

    def delete_songs(self) -> None:
        """The run's songs, off the card: a "Song N" on a card of 50 songs is not in the
        picker's 24, so Arrange's own Delete cannot reach it. The song that was open
        before is reopened first, so none of them is the one the engine holds."""
        self.reopen_song0()
        # Only the run's own: a song is never taken off the card for being missing from
        # a listing made before (a listing cut off once made every song look new).
        for name in new_names(self.before["songs"], self.listing(SONGS_ROOT)):
            if name not in self.songs:
                self.ctx.log(f"cleanup: song {name} is not the run's; left alone")
                continue
            self.ctx.log(f"cleanup: song {name}")
            self.rm_tree(f"{SONGS_ROOT}/{name}")

    def checks_after(self) -> None:
        """The card as found: every listing the run changed is as it was, the volume is
        quiet and the speaker amp is off."""
        after = {"songs": self.listing(SONGS_ROOT), "elements": self.listing(ELEMENTS_DIR),
                 "bake": self.listing(BAKE_DIR), "recordings": self.listing(RECORDINGS)}
        left = {k: new_names(self.before[k], after[k]) for k in after}
        gone = {k: new_names(after[k], self.before[k]) for k in after}
        out = verbs.outvol(self.link)
        self.data["cleanup"] = {"left": left, "gone": gone, "spk_amp": out["spk_amp"],
                                "spk_amp0": self.data["outvol0"]["spk_amp"]}
        extra = {k: v for k, v in left.items() if v}
        # Not the bakes: the cache evicts old ones by itself, and one that left during the
        # run is expected.
        lost = {k: v for k, v in gone.items() if v and k != "bake"}
        if extra and not self.p.keep:
            self.failures.append(f"cleanup left {extra} on the card")
        if lost:
            self.failures.append(f"cleanup: the card lost {lost}")
        if out["spk_amp"] != "off":
            self.failures.append(f"the speaker amp is {out['spk_amp']} after the run")

    def speakers_off(self) -> None:
        """The speaker amp off, as the brief asks, whatever it was at the start: Setup ->
        Audio -> the Speakers switch (OUTVOL only reads the amp's gate)."""
        if verbs.outvol(self.link)["spk_amp"] == "off":
            return
        self.data["spk_amp_was_on"] = True
        self.home()
        self.ui.tap("Setup")
        self.ui.tap("Audio")
        row = self.ui.seek("Speakers (auto", contains=True)
        switch = next(n for n in self.ui.screen().nodes
                      if n.cls == "switch" and abs(n.y - row.y) < row.h)
        if switch.checked:
            self.ui.tap_node(switch)
        self.until(lambda: verbs.outvol(self.link), lambda o: o["spk_amp"] == "off")

    def cleanup(self) -> None:
        undo: list[tuple[str, Callable[[], Any]]] = [
            ("stop", self.stop), ("home", self.home), ("speakers", self.speakers_off),
            ("home", self.home)]
        if not self.p.keep:
            undo += [("songs", self.delete_songs),
                     ("elements", lambda: self.rm_new(ELEMENTS_DIR, "elements", "element")),
                     ("bakes", lambda: self.rm_new(BAKE_DIR, "bake", "bake")),
                     ("takes", lambda: self.rm_new(RECORDINGS, "recordings", "take"))]
        if self.kit0 >= 0:
            undo += [("song", self.reopen_song0), ("kit", self.put_kit_back)]
        for what, fn in undo:
            try:
                fn()
            except Exception as e:  # noqa: BLE001 - one undo failing must not stop the rest
                self.ctx.log(f"cleanup: {what}: {type(e).__name__}: {e}")
        try:
            self.checks_after()
        except HilError as e:
            self.failures.append(f"cleanup could not be checked: {e.message}")
        audio_state.put_back(self.link, self.audio_snap, self.ctx.log, self.data)

    def put_kit_back(self) -> None:
        if int(verbs.kit_status(self.link)["current"]) != self.kit0:
            verbs.kit_load(self.link, self.kit0, wait_s=self.p.job_timeout_s)


class SongElementsScenario:
    name = "song_elements"
    Params = Params
    description = "nested song elements played on the screen: make, enter, edit, leave, bake"

    def run(self, ctx: Context, params: Params, progress: Progress) -> Report:
        run = _Run(ctx, params)
        phases: list[tuple[str, Callable[[], None]]] = [
            ("new song", run.new_song), ("audio take", run.audio_take),
            ("kit take", run.kit_take), ("element from region", run.element_from_region),
            ("bed", run.bed_take),
            ("enter", run.enter_element), ("nested element", run.nested_element),
            ("pads", run.play_pads), ("change notes", run.change_notes),
            ("up twice", run.go_up),
            ("play at once", lambda: run.play_at_once(
                10, "PLAY at once: the position moves within one ring prime, the ring is "
                    "filled, no underrun")),
            ("bake changed", run.bake_changed), ("SOLO", run.solo),
            ("copy or replace", run.copy_question), ("cold cache", run.cold_cache)]
        try:
            progress(0, len(phases), "prepare the bench")
            run.setup()
            for i, (what, phase) in enumerate(phases):
                ctx.check_cancelled()
                progress(i, len(phases), what)
                try:
                    phase()
                except UiMissing as e:
                    run.failures.append(f"{what}: {e.message}; {e.hint}"[:400])
                    raise _Stop(what) from e
            progress(len(phases), len(phases), "done")
        except _Stop:
            pass
        except HilError as e:
            if e.code == ENV and not run.data["steps"]:
                return self._bench(e.message, run.data)
            run.failures.append(f"{e.code}: {e.message}")
        finally:
            if run.link is not None:
                try:
                    delta = verbs.reset_count(run.link) - run.data.get("resets0", 0)
                    if delta:
                        run.failures.append(f"{delta} reset(s) during the run")
                except HilError as e:
                    run.failures.append(f"the board stopped answering: {e.message}")
                run.cleanup()
            ctx.close_all()
        run.data["failures"] = run.failures
        run.data["ui_busy"] = run.ui.busy if run._ui else 0
        return self._report(run)

    @staticmethod
    def _bench(why: str, data: dict[str, Any]) -> Report:
        data["failures"] = [why]
        return Report(passed=False, summary=f"song_elements: BENCH — {why}", data=data,
                      artifacts=[], exit_code=2)

    @staticmethod
    def _report(run: _Run) -> Report:
        passed = not run.failures
        steps = run.data["steps"]
        done = sum(1 for s in steps.values() if s["status"] == "pass")
        to_listen = sorted(int(n) for n, s in steps.items() if s.get("manual"))
        summary = (f"song_elements: PASS ({done} steps done on the screen; to listen to: "
                   f"{', '.join(map(str, to_listen))})" if passed
                   else "song_elements: FAIL — " + "; ".join(run.failures))
        return Report(passed=passed, summary=summary, data=run.data, artifacts=run.artifacts,
                      exit_code=0 if passed else 1)


SCENARIO = register(SongElementsScenario())
