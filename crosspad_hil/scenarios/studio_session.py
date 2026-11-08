"""The studio session of 21.10.2026, sample to Ableton project, played on the screen.

Jakub Jan Bryndal's first studio test of the CrossPad S3 (Instagram reel
DewkqQaMMkv) walked one path through the firmware: play a kit, trim and copy
pads, record a pattern, bounce it into a song, record a synth through line in
onto the arrangement, chop a sample brought over USB, mix the song and export it
as an Ableton Live Set. The fixture (`--fixture`, or CROSSPAD_STUDIO_FIXTURE) is
the folder written from that reel: `scenario.md` numbers the 22 steps, `sd/`
holds the three kits of the session and `sd.sha256` their sums.

The run does what the player did, with the controls the player used: it finds a
button on the screen by what it says and taps it (`UI_TREE` + `TOUCH_TAP`,
crosspad_hil.ui), turns the knob (`ENC_ROTATE`/`ENC_PRESS`), hits the pads
(`PAD_PRESS`) and clicks the power button (`PWR_GESTURE`). State verbs only
read what came of it: PAD_INFO, SEQ, ASONG, song.json, the card. A control
that is missing, covered, disabled or says something else fails its step with
what the screen offered instead. The top bar's drawn tiles (REC, PLAY) are
found by their place, as a person finds them by their shape.

What stays for a person is listed per step under "manual": what it sounds like
and what the LEDs do.

The song is a new one (Arrange → song → NEW), so nobody's song is touched; it
is deleted the same way at the end with the slice kit, the uploaded sample and
the take unless `--keep`. The fixture kit is saved by the run (as the player
saved it) and put back from the fixture afterwards. song.json and the .als are
copied into the work dir either way. `--stop-after N` ends the run after the
phase holding step N, for working on the scenario itself.
"""
from __future__ import annotations

import gzip
import hashlib
import json
import os
import re
import struct
import time
import xml.etree.ElementTree as ET
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from crosspad_hil import audio_state, fs, msc, parsers, verbs
from crosspad_hil.cdc import CdcLink
from crosspad_hil.devices import discover, select
from crosspad_hil.errors import ENV, HilError
from crosspad_hil.kits import KITS_ROOT, reboot_for_kit_scan, wait_for_kit_scan
from crosspad_hil.scenarios._audio import ensure_kits
from crosspad_hil.scenarios.base import Artifact, Context, Progress, Report, register
from crosspad_hil.ui import ICON_COPY, ICON_PLAY, TILE_PLAY_X, TILE_REC_X, Ui, UiMissing

FIXTURE_ENV = "CROSSPAD_STUDIO_FIXTURE"
SONGS_ROOT = "/sdcard/SONGS"
RECORDINGS = "/sdcard/crosspad/recordings"
SAMPLE_RATE = 44100
BEATS_PER_BAR = 4
DEFAULT_BPM = 120
START_DETENTS = 5
VOL_DETENTS = -3
PAN_DETENTS = 2
PITCH_DETENTS = 2
TAP_GAP_S = 0.3             # a press and a release are a CDC round trip each
SEQ_RUNNING = 4             # SEQ state=: the take is past its count-in
DRIVE_RESTARTS = 2          # into USB transfer mode and out of it

BOUNCE_RUNNING, BOUNCE_DONE, BOUNCE_FAILED = 1, 2, 3
OP_READY, OP_FAILED = 2, 3

_PLAYS = re.compile(r"plays ([\d.]+) (?:-|back to) ([\d.]+) s")
_BARS = re.compile(r"^\d+b$")
_SAVED_KIT = re.compile(r"Saved the Slice kit (\S+)")
_NUMBER = re.compile(r"[-+]?\d+")


def _number(text: str) -> int | None:
    """The last number a label shows ("PITCH +2" -> 2), None when it shows none."""
    found = _NUMBER.findall(text)
    return int(found[-1]) if found else None


@dataclass
class Params:
    fixture: str = field(default="", metadata={
        "help": f"the fixture folder (scenario.md, sd/, sd.sha256); default ${FIXTURE_ENV}"})
    kit: str = field(default="ŚWIĘTE SŁOWA", metadata={"help": "the kit the session played"})
    bpm: int = field(default=93, metadata={"help": "the session's tempo"})
    hits: int = field(default=4, metadata={"help": "pads hit into each scene"})
    take_s: float = field(default=6.0, metadata={"help": "seconds of the line-in take"})
    grid_tol: int = field(default=1, metadata={
        "help": "frames a bounce may differ from its bars on the song's grid"})
    reboot: bool = field(default=True, metadata={
        "help": "reset the board and check the song and the pads came back the same"})
    keep: bool = field(default=False, metadata={
        "help": "leave the song, the slice kit, the upload and the take on the card"})
    stop_after: int = field(default=0, metadata={
        "help": "end after the phase holding this step (0: run them all)"})
    poll_s: float = field(default=0.25, metadata={"help": "between polls of a job's state"})
    job_timeout_s: float = field(default=45.0, metadata={
        "help": "for a kit load, a bounce, putting a take in the song, an export"})


# --------------------------------------------------------------------------- host side

def verify_fixture(root: Path) -> tuple[dict[str, Path], list[str]]:
    """The fixture's kits by their kit.json name, and what is wrong with the folder.

    Every line of sd.sha256 must match its file under sd/: a kit that changed since
    the session is not the session's kit.
    """
    problems: list[str] = []
    sums = root / "sd.sha256"
    sd = root / "sd"
    if not sums.is_file() or not sd.is_dir():
        return {}, [f"{root} has no sd/ or sd.sha256 -- not the studio fixture"]
    for line in sums.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        want, _, rel = line.partition("  ")
        path = sd / rel.strip().removeprefix("./")
        if not path.is_file():
            problems.append(f"missing {rel.strip()}")
        elif hashlib.sha256(path.read_bytes()).hexdigest() != want.strip():
            problems.append(f"sha256 differs: {rel.strip()}")
    kits: dict[str, Path] = {}
    for kit_json in sorted((sd / "crosspad" / "kits").glob("*/kit.json")):
        kits[str(json.loads(kit_json.read_text(encoding="utf-8"))["name"])] = kit_json.parent
    if not kits:
        problems.append("no kits under sd/crosspad/kits")
    return kits, problems


def wav_frames(head: bytes) -> tuple[int, int, int]:
    """(frames, channels, rate) from the start of a PCM WAV: the data chunk's size."""
    if head[:4] != b"RIFF" or head[8:12] != b"WAVE":
        raise ValueError("not a RIFF/WAVE file")
    pos, channels, rate, width = 12, 0, 0, 0
    while pos + 8 <= len(head):
        cid, size = head[pos:pos + 4], struct.unpack_from("<I", head, pos + 4)[0]
        if cid == b"fmt ":
            _, channels, rate, _, _, bits = struct.unpack_from("<HHIIHH", head, pos + 8)
            width = bits // 8
        elif cid == b"data":
            if not channels or not width:
                raise ValueError("data chunk before fmt")
            return size // (channels * width), channels, rate
        pos += 8 + size + (size & 1)
    raise ValueError("no data chunk in the header read")


def bar_frames(bars: int, bpm: float, rate: int = SAMPLE_RATE) -> float:
    """Frames `bars` bars of 4/4 last at `bpm`, exactly."""
    return bars * BEATS_PER_BAR * 60.0 * rate / bpm


def song_bar_frames(bars: int, bpm: float, rate: int = SAMPLE_RATE) -> int:
    """Frames `bars` bars last on the song's grid: the beat truncated first, as core
    AudioSong::samplesPerBar and the metronome count it, so a bounce is whole beats."""
    return bars * BEATS_PER_BAR * int(rate * 60.0 / bpm)


def parse_live_set(als: bytes) -> dict[str, Any]:
    """What a Live Set says about the song: tempo and, per AudioTrack, name, volume and
    clips (start in beats, the file relative to the project)."""
    root = ET.fromstring(gzip.decompress(als) if als[:2] == b"\x1f\x8b" else als)
    tempo = root.find(".//MainTrack//Mixer/Tempo/Manual")
    tracks = []
    for tr in root.iter("AudioTrack"):
        name = tr.find("./Name/EffectiveName")
        vol = tr.find(".//Mixer/Volume/Manual")
        clips = []
        for clip in tr.iter("AudioClip"):
            rel = clip.find(".//SampleRef/FileRef/RelativePath")
            clips.append({"time": float(clip.get("Time", "0")),
                          "path": rel.get("Value", "") if rel is not None else ""})
        tracks.append({"name": name.get("Value", "") if name is not None else "",
                       "volume": float(vol.get("Value", "nan")) if vol is not None else None,
                       "clips": clips})
    return {"tempo": float(tempo.get("Value", "nan")) if tempo is not None else None,
            "tracks": tracks}


def compare_set(song: dict[str, Any], live: dict[str, Any], bpm: float) -> list[str]:
    """How the Set differs from song.json: tempo, track count, and each audio clip's bar.

    Seq blocks (clips with notes and no file) are not audio and are not in the Set.
    """
    out: list[str] = []
    if live["tempo"] is None or abs(live["tempo"] - bpm) > 0.01:
        out.append(f"the Set's tempo is {live['tempo']} (want {bpm})")
    tracks = song.get("tracks", [])
    if len(live["tracks"]) != len(tracks):
        out.append(f"the Set has {len(live['tracks'])} audio tracks, song.json {len(tracks)}")
    for t, (st, lt) in enumerate(zip(tracks, live["tracks"], strict=False)):
        want = sorted(float(c.get("bar", 0)) * BEATS_PER_BAR
                      for c in st.get("clips", []) if c.get("file"))
        have = sorted(c["time"] for c in lt["clips"])
        if want != have:
            out.append(f"track {t + 1}: clips at beats {have} in the Set, {want} in song.json")
    return out


# --------------------------------------------------------------------------- the run

class _Stop(Exception):
    """A step the rest of the run depends on failed: report what was checked so far."""


class _Run:
    """One pass: the link, the screen, what has been checked and what to undo."""

    def __init__(self, ctx: Context, params: Params, kits: dict[str, Path]) -> None:
        self.ctx = ctx
        self.p = params
        self.kits = kits
        self.link: CdcLink | None = None
        self.data: dict[str, Any] = {"steps": {}}
        ctx.partial = self.data
        self.failures: list[str] = []
        self.artifacts: list[Artifact] = []
        self.song = ""              # the run's own song, once NEW made it
        self.song0 = ""             # the song open before the run
        self.kit_saved = False
        self.take = ""
        self.uploaded = False
        self.chop_kit = ""
        # A name of the run's own: the Slicer names its kit after the file, and a
        # kit of that name left by a run that died would refuse this one's.
        self.upload_name = time.strftime("hil_studio_%H%M%S.wav")
        self.console: Any = None
        self.resets_before_drive = 0
        self._ui: Ui | None = None
        self.audio_snap: dict[str, Any] | None = None

    # ----------------------------------------------------------------- the hands

    @property
    def ui(self) -> Ui:
        if self._ui is None:
            self._ui = Ui(self.link, wait=self.ctx.cancelled.wait)
        self._ui.link = self.link             # the link changes across restarts
        return self._ui

    @property
    def song_dir(self) -> str:
        return f"{SONGS_ROOT}/{self.song}"

    def wait(self, s: float) -> None:
        self.ctx.cancelled.wait(s)

    def knob(self, detents: int) -> None:
        """Turn the encoder, a detent at a time as a hand does (ENC_ROTATE caps at 16)."""
        while detents:
            step = max(-16, min(16, detents))
            verbs.enc_rotate(self.link, step)
            detents -= step
            self.wait(0.15)
        self.wait(0.3)

    def click(self) -> None:
        verbs.enc_press(self.link)
        self.wait(0.4)

    def power(self) -> None:
        verbs.pwr_gesture(self.link)
        self.wait(0.8)

    def hit(self, pad: int, vel: int = 100) -> None:
        verbs.pad_press(self.link, pad, vel)
        verbs.pad_release(self.link, pad)

    def turn(self, label: str, detents: int) -> str:
        """Tap a value control, turn the knob, click it: what the control says after."""
        return self.turned(label, detents)[1]

    def turned(self, label: str, detents: int) -> tuple[str, str]:
        """turn(), and what the control said before it."""
        sc = self.ui.screen()
        node = self.ui.tap(label, contains=True)
        before = " ".join(dict.fromkeys(sc.words(node)))
        self.knob(detents)
        self.click()
        sc = self.ui.screen()
        after = sc.at(*node.center)
        return before, " ".join(dict.fromkeys(sc.words(after))) if after else ""

    def set_number(self, label: str, want: int) -> str:
        """A numbered value control (TRACK 3) brought to @p want: what it said before
        decides the way to turn; untouched when it already says it."""
        sc = self.ui.screen()
        node = sc.find(label, contains=True)
        now = _number(" ".join(sc.words(node))) if node else None
        if node is None or now is None:
            raise UiMissing("TIMEOUT", f"no {label} value on screen",
                            hint="could press: " + ", ".join(sc.labels()[:20]))
        if now != want:
            self.turn(label, want - now)
        return " ".join(dict.fromkeys(self.ui.screen().words(
            self.ui.screen().find(label, contains=True))))

    def says(self, text: str) -> str:
        """The first label on screen holding @p text, all of it ("" when none)."""
        for n in self.ui.screen().nodes:
            if n.cls == "label" and text.casefold() in n.text.casefold():
                return n.text
        return ""

    def until(self, ask: Callable[[], Any], done: Callable[[Any], bool],
              timeout_s: float | None = None) -> Any:
        deadline = time.monotonic() + (timeout_s or self.p.job_timeout_s)
        got = ask()
        while not done(got) and time.monotonic() < deadline and not self.ctx.cancelled.is_set():
            self.wait(self.p.poll_s)
            got = ask()
        return got

    def home(self) -> None:
        """Power clicks back to the home map, saving the kit when the Sampler asks."""
        for _ in range(8):
            sc = self.ui.screen()
            if sc.running == "-" and sc.find("Kits") and sc.find("Sampler"):
                return
            if sc.find("Unsaved changes", interactive=False) and sc.find("Save"):
                self.ui.tap("Save")
                self.kit_saved = True
                self.wait(1.0)
                continue
            self.power()
        raise _Stop("power clicks did not bring the home map back: "
                    + ", ".join(self.ui.screen().labels()[:20]))

    def open(self, label: str, app: str) -> None:
        """From the home map, the cell saying @p label; the app must be @p app."""
        self.home()
        self.ui.tap(label)
        sc = self.until(self.ui.screen, lambda s: s.running == app, timeout_s=6.0)
        if sc.running != app:
            raise UiMissing("TIMEOUT", f"{label!r} opened {sc.running}, not {app}",
                            hint="could press: " + ", ".join(sc.labels()[:20]))

    def pad_shape(self, pad: int) -> dict[str, Any]:
        info = parsers.parse_cdc_reply(verbs.pad_info(self.link, pad)["raw"]) or {}
        shape = {k: info.get(k) for k in ("vol", "pan", "mode", "count")}
        shape["slots"] = [{k: s[k] for k in ("file", "lo", "hi")} for s in info.get("slots", [])]
        return shape

    def engine(self) -> int:
        return int(verbs.smpl_last(self.link)["engine_total"])

    def pull(self, remote: str, local: Path, length: int | None = None) -> bytes:
        """A file of the card, over the raw port: the link is closed and opened again."""
        self.link.close()
        try:
            fs.pull(self.ctx.device, remote, local, length=length)
        finally:
            self.link = self.ctx.open_cdc()
        return local.read_bytes()

    def song_json(self, name: str = "song.json") -> dict[str, Any]:
        return json.loads(self.pull(f"{self.song_dir}/song.json",
                                    self.ctx.workdir / name).decode("utf-8"))

    def clip_path(self, file: str) -> str:
        return file if file.startswith("/") else f"{self.song_dir}/clips/{file}"

    # ----------------------------------------------------------------- bookkeeping

    def step(self, n: int, ok: bool, checked: str, *, why: str = "", manual: str = "") -> bool:
        entry: dict[str, Any] = {"status": "pass" if ok else "fail", "checked": checked}
        if manual:
            entry["manual"] = manual
        if not ok:
            entry["why"] = why
            self.failures.append(f"step {n}: {why}")
        self.data["steps"][str(n)] = entry
        return ok

    def missing(self, n: int, what: str, e: UiMissing) -> None:
        """A control a person would press was not there: the step fails with the screen."""
        self.step(n, False, what, why=f"{e.message}; {e.hint}"[:400])

    # ----------------------------------------------------------------- setup

    def setup(self) -> None:
        self.link = self.ctx.open_cdc()
        if verbs.fs_roots(self.link)["roots"].get("/sdcard", {}).get("total") is None:
            raise HilError(ENV, "no SD card mounted on the board",
                           hint="FS_ROOTS shows /sdcard unmounted; put a FAT card in and reset")
        self.link, pushed = ensure_kits(self.ctx, self.link, list(self.kits),
                                        lambda _root: self.kits)
        self.data["kits_pushed"] = pushed
        self.away_from_kit()      # first: leaving a dirty kit saves it over the card's
        self.put_kit_back()
        self.reboot_board()       # the kit list keeps a previous run's edits until a scan
        self.audio_snap = audio_state.snapshot(self.link)
        self.data["resets0"] = verbs.reset_count(self.link)
        self.song0 = str(verbs.asong(self.link).get("project", ""))
        self.under0 = int(verbs.smpl_counters(self.link).get("underrun", 0))

    def reboot_board(self) -> None:
        """Reset through the STM bridge and wait for the kit scan: a fresh kit list.
        One console for the run: a second one would wait on the first one's lock."""
        if self.console is None:
            self.console = self.ctx.open_console(reset=False)
        if self.link is not None:
            self.link.close()
        reboot_for_kit_scan(self.ctx, self.console)
        self.relink()
        wait_for_kit_scan(self.ctx, self.link)
        if self.data.get("resets0") is not None:      # a reset the run made: not counted
            self.data["resets0"] = verbs.reset_count(self.link)

    def away_from_kit(self) -> None:
        """Bench preparation: another fixture kit loaded, so step 1 loads the session's
        from the card -- a tap on the kit already loaded only opens the Sampler, and
        a previous run's edits would still be in it."""
        other = next(n for n in self.kits if n != self.p.kit)
        kits = verbs.kit_list(self.link)["kits"]
        kid = next(int(k["id"]) for k in kits if str(k["name"]) == other)
        verbs.kit_load(self.link, kid, wait_s=self.p.job_timeout_s)

    # ----------------------------------------------------------------- 1: the kit

    def load_kit(self) -> tuple[dict[str, Any], Any]:
        """Kits on the map, the kit's row scrolled to and tapped (LOAD when the
        browser stays up), the Sampler from the map if the load did not open it."""
        self.home()
        self.ui.tap("Kits")
        self.ui.tap_node(self.ui.seek(self.p.kit))
        self.wait(1.0)
        if self.ui.screen().find("LOAD"):                # picked, not loaded yet
            self.ui.tap("LOAD")
        st = self.until(lambda: verbs.kit_status(self.link),
                        lambda r: r["name"] == self.p.kit and not r["loading"])
        if self.ui.screen().running != "Sampler":
            self.open("Sampler", "Sampler")
        return st, self.ui.screen()

    def kit(self) -> None:
        """Kits on the map, the kit's row scrolled to and tapped (LOAD when the browser
        stays up); the Sampler from the map if the load did not open it. Then every
        pad, once."""
        try:
            st, sc = self.load_kit()
        except UiMissing as e:
            self.missing(1, "Kits → the kit's row", e)
            raise _Stop("the kit could not be loaded from the screen") from e
        e0 = self.engine()
        for pad in range(16):
            self.hit(pad)
            self.wait(0.1)
        played = self.engine() - e0
        under = int(verbs.smpl_counters(self.link).get("underrun", 0)) - self.under0
        self.data["kit"] = {"loaded": st["name"], "app": sc.running, "played": played,
                            "under": under}
        self.step(1, st["name"] == self.p.kit and sc.running == "Sampler" and played >= 16
                  and under == 0,
                  "Kits → scroll to the kit → tap it: loaded, Sampler open; 16 pads played",
                  why=f"kit {st['name']!r}, app {sc.running}, {played} notes played of 16, "
                      f"{under} underruns",
                  manual="every pad sounds at once, its LED lights, no clicks")

    # ----------------------------------------------------------------- 2-4: the pad editor

    def editor(self) -> None:
        """Pad 1, EDIT: the waveform's region; START, VOL, PAN, PITCH by tap and knob;
        the region auditioned with PLAY."""
        try:
            self.ui.tap("EDIT", contains=True)
            self.ui.wait_for("PAD", contains=True)
            self.hit(0)                          # the editor follows the pad hit
            self.ui.wait_for("PAD 1")
        except UiMissing as e:
            self.missing(2, "Sampler → EDIT on pad 1", e)
            raise _Stop("the pad editor did not open") from e
        info = self.says("plays")
        m = _PLAYS.search(info)
        self.step(2, bool(m), "the editor shows pad 1's sample and the region it plays",
                  why=f"no 'plays a - b s' on the editor: {info!r}",
                  manual="the waveform is drawn whole")
        before = self.pad_shape(0)
        try:
            start_label = self.turn("START", START_DETENTS)
        except UiMissing as e:
            self.missing(3, "START, turned", e)
            return
        after = self.pad_shape(0)
        lo0 = before["slots"][0]["lo"] if before["slots"] else -1
        lo1 = after["slots"][0]["lo"] if after["slots"] else -1
        info2 = self.says("plays")
        self.data["editor"] = {"start": [lo0, lo1], "info": [info, info2], "label": start_label}
        self.step(3, lo1 > lo0 and info2 != info,
                  f"START tapped, {START_DETENTS} detents, clicked: the region starts later",
                  why=f"start {lo0} -> {lo1}, editor {info!r} -> {info2!r}",
                  manual="the pad plays only the region")
        got: dict[str, tuple[str, str]] = {}
        try:
            for label, d in (("VOL", VOL_DETENTS), ("PAN", PAN_DETENTS), ("PITCH", PITCH_DETENTS)):
                got[label] = self.turned(label, d)
        except UiMissing as e:
            self.missing(4, "VOL / PAN / PITCH, turned", e)
            return
        shape = self.pad_shape(0)
        self.ui.tap(ICON_PLAY)
        self.data["editor"].update(values=got, vol=shape["vol"], pan=shape["pan"])
        want_vol = (before["vol"] or 0) + VOL_DETENTS
        pitch = [_number(s) for s in got.get("PITCH", ("", ""))]
        ok = (shape["vol"] == want_vol and shape["pan"] != before["pan"]
              and None not in pitch and pitch[1] - pitch[0] == PITCH_DETENTS)
        self.step(4, ok, "VOL, PAN, PITCH by tap and knob; PAD_INFO and the labels follow",
                  why=f"labels {got}, PAD_INFO vol {shape['vol']} (want {want_vol}) "
                      f"pan {before['pan']} -> {shape['pan']}",
                  manual="the change is heard from the next hit; PLAY auditions the region")

    # ----------------------------------------------------------------- 5: COPY

    def copy(self) -> None:
        """The copy key, pad 2 hit, Copy in the question: pad 2 is pad 1."""
        src = self.pad_shape(0)
        try:
            self.ui.tap(ICON_COPY)
            prompt = self.says("COPY:")
            self.hit(1)
            self.ui.wait_for("Copy PAD 1 onto PAD 2?", interactive=False)
            self.ui.tap("Copy")
        except UiMissing as e:
            self.missing(5, "copy key → pad 2 → Copy", e)
            return
        self.wait(0.5)
        dst = self.pad_shape(1)
        self.data["copy"] = {"prompt": prompt, "src": src, "dst": dst}
        same = {k: dst[k] for k in ("vol", "pan", "slots")} == {k: src[k] for k in
                                                               ("vol", "pan", "slots")}
        self.step(5, bool(prompt) and same,
                  "copy key, pad 2 hit, Copy confirmed: pad 2 has pad 1's sample, region, "
                  "level and pan",
                  why=f"prompt {prompt!r}; pad 1 {src} / pad 2 {dst}")

    # ----------------------------------------------------------------- the run's song

    def new_song(self) -> None:
        """Arrange → the song's name → NEW, then its tempo in the song's sheet."""
        self.open("Arrange", "Arrange")
        self.ui.tap(self.song0)
        self.ui.tap("NEW")
        st = self.until(lambda: verbs.asong(self.link),
                        lambda r: r["open"] == OP_READY and r["project"] != self.song0)
        if st["project"] == self.song0:
            raise _Stop("NEW did not open a new song")
        self.song = st["project"]
        self.ui.tap(self.song)
        self.ui.tap("EDIT")
        tempo = self.turn("Tempo", self.p.bpm - DEFAULT_BPM)
        bpm = self.until(lambda: verbs.asong(self.link), lambda r: r["bpm"] == self.p.bpm,
                         timeout_s=5.0)["bpm"]
        self.data["song"] = {"name": self.song, "tempo": tempo, "bpm": bpm}
        if bpm != self.p.bpm:
            raise _Stop(f"the song's Tempo turned to {tempo!r}, ASONG bpm={bpm}")
        self.home()

    # ----------------------------------------------------------------- 6-9: Seq

    def knob_cycle_to(self, pattern: re.Pattern[str], want: str, limit: int = 8) -> str:
        """A key only the knob reaches that steps through choices on each click (the
        Sequencer's bars: 1b 2b 4b 8b): turn to it, click until it says @p want."""
        group = [str(e.get("label") or "") for e in verbs.enc_group(self.link)["group"]]
        now = next((g for g in group if pattern.match(g.strip())), None)
        if now is None:
            raise UiMissing("TIMEOUT", f"no {pattern.pattern} in the knob's reach",
                            hint="the knob reaches: " + ", ".join(group)[:300])
        self.ui.knob_to(now)
        got = now
        for _ in range(limit):
            if got.strip().casefold() == want.casefold():
                break
            self.click()
            got = str(verbs.enc_focus(self.link).get("label", ""))
        return got

    def record(self, scene_label: str) -> int:
        """REC on the top bar, the count-in waited out, the hits played, REC again to
        end the take, PLAY to stop: the scene's events."""
        self.ui.tap_tile(TILE_REC_X)
        self.until(lambda: verbs.seq(self.link), lambda r: r["state"] == SEQ_RUNNING,
                   timeout_s=8.0)
        for pad in range(self.p.hits):
            self.hit(pad)
            self.wait(TAP_GAP_S)
        self.wait(1.0)
        self.ui.tap_tile(TILE_REC_X)
        self.ui.tap_tile(TILE_PLAY_X)
        sq = verbs.seq(self.link)
        scene = int(str(sq["scene"]).split(">")[0])
        self.data.setdefault("takes", {})[scene_label] = {"scene": scene, "ev": sq["ev"][scene],
                                                          "bars": sq["bars"][scene]}
        return sq["ev"][scene]

    def seq(self) -> None:
        try:
            self.open("Seq", "Sequencer")
        except UiMissing as e:
            self.missing(6, "Seq on the map", e)
            raise _Stop("the Sequencer did not open") from e
        self.step(6, True, "Seq on the map opens the Sequencer",
                  manual="the knob walks the main menu without losing focus")
        try:
            bars = self.knob_cycle_to(_BARS, "1b")
            ev = self.record("scene 1")
        except UiMissing as e:
            self.missing(7, "bars, REC, pads, REC, PLAY", e)
            raise _Stop("nothing was recorded") from e
        self.step(7, ev == self.p.hits and bars.startswith("1"),
                  f"one bar ({bars!r}), REC, {self.p.hits} pads, REC, PLAY: every hit kept",
                  why=f"{ev} events of {self.p.hits}, bars control {bars!r}",
                  manual="the loop plays the hits back on the beat")
        try:
            self.ui.tap("x2")
        except UiMissing as e:
            self.missing(9, "x2", e)
            return
        sq = verbs.seq(self.link)
        scene = int(str(sq["scene"]).split(">")[0])
        self.data["doubled"] = {"bars": sq["bars"][scene], "ev": sq["ev"][scene]}
        self.step(9, sq["bars"][scene] == 2 and sq["ev"][scene] == 2 * ev,
                  "x2: twice the bars, the hits twice over",
                  why=f"bars {sq['bars'][scene]} events {sq['ev'][scene]} (want 2 and {2 * ev})")

    def fader(self, pad: int):
        """Pad Mix's fader of @p pad (1-based): the control reading "<level>" "<pad>"."""
        sc = self.ui.screen()
        return next(c for c in sc.controls()
                    if (w := sc.words(c)) and w[-1] == str(pad) and w[0].isdigit())

    def padmix(self) -> None:
        """Pad Mix on the map: a finger low on pad 1's fader brings the pad's level
        down there -- PAD_INFO reads it -- and the fader shows it."""
        before = self.pad_shape(0)["vol"]
        try:
            self.open("Pad Mix", "PadMixer")
            fader = self.fader(1)
        except (UiMissing, StopIteration) as e:
            self.step(8, False, "Pad Mix: pad 1's fader",
                      why=f"no fader for pad 1 on {self.ui.screen().labels()}: {e}"[:400])
            return
        shown = self.ui.screen().words(fader)[0]
        x, y = fader.center[0], fader.y + fader.h * 4 // 5
        verbs.touch_tap(self.link, x, y)
        self.wait(0.6)
        turned = self.pad_shape(0)["vol"]
        label = self.ui.screen().words(self.fader(1))[0]
        self.data["padmix"] = {"shown": shown, "label": label, "vol": [before, turned]}
        self.step(8, turned < before and label == str(turned),
                  "Pad Mix: a finger low on pad 1's fader brings the pad's level down",
                  why=f"fader showed {shown}, then {label!r}; PAD_INFO vol {before} -> {turned}",
                  manual="the fader changes the pad while the pattern plays")

    # ----------------------------------------------------------------- 10-11: bounce, Arrange

    def bounce(self, n: int, track_label: str) -> dict[str, Any]:
        """BOUNCE, then the track in the question; ASONG once the bounce ends."""
        self.ui.tap("BOUNCE")
        self.ui.wait_for("on which track?", contains=True, interactive=False)
        self.ui.tap(track_label)
        return self.until(lambda: verbs.asong(self.link),
                          lambda r: r["bounce"] in (BOUNCE_DONE, BOUNCE_FAILED))

    def bounce1(self) -> None:
        try:
            self.open("Seq", "Sequencer")
            clips0 = verbs.asong(self.link)["clips"]
            st = self.bounce(10, "Track 1")
        except UiMissing as e:
            self.missing(10, "BOUNCE → Track 1", e)
            raise _Stop("the first bounce did not start") from e
        self.data["bounce1"] = {"state": st["bounce"], "file": st["file"], "err": st["err"]}
        if st["bounce"] != BOUNCE_DONE or st["clips"][0] != clips0[0] + 1:
            self.step(10, False, "BOUNCE → Track 1",
                      why=f"bounce={st['bounce']} clips {clips0} -> {st['clips']} "
                          f"{st['err'] or ''}".rstrip())
            raise _Stop("the first bounce did not finish")
        song = self.until(self.song_json, lambda s: len(s["tracks"][0].get("clips", [])) >= 1,
                          timeout_s=5.0)
        clip = song["tracks"][0]["clips"][-1]
        head = self.pull(self.clip_path(clip["file"]), self.ctx.workdir / "bounce1_head.bin",
                         length=512)
        frames, _, rate = wav_frames(head)
        want = song_bar_frames(clip.get("bars", 2), self.p.bpm, rate)
        self.data["bounce1"].update(frames=frames, want=want, bars=clip.get("bars"),
                                    short_of_exact=round(bar_frames(clip.get("bars", 2),
                                                                    self.p.bpm, rate) - want, 2))
        self.step(10, abs(frames - want) <= self.p.grid_tol,
                  f"BOUNCE → Track 1: a clip of {clip.get('bars')} bars, {frames} frames "
                  f"against the song's {want}",
                  why=f"the bounce is {frames} frames, {abs(frames - want)} off the song's grid")

    def arrange(self) -> None:
        try:
            self.open("Arrange", "Arrange")
            zoom0 = self.ui.screen().labels()
            self.turn("ZOOM", 2)
            zoom1 = self.ui.screen().labels()
        except UiMissing as e:
            self.missing(11, "Arrange, ZOOM", e)
            return
        clips = verbs.asong(self.link)["clips"]
        moved = "changed" if zoom0 != zoom1 else "did not change"
        self.step(11, clips[0] >= 1 and zoom0 != zoom1,
                  "Arrange: the clip on track 1; ZOOM turned changes the view",
                  why=f"clips {clips}, the screen {moved} on ZOOM")

    # ----------------------------------------------------------------- 12-13: line in

    def line_take(self) -> None:
        """Record on the map: LINE, SYNC, REC (the song plays), STOP; ARR, TRACK 2,
        BAR 1, PUT."""
        clips0 = verbs.asong(self.link)["clips"]
        try:
            self.open("Record", "Recorder")
            for _ in range(5):                       # the input key steps MIC, LINE, ...
                if self.ui.screen().find("LINE"):
                    break
                self.ui.tap("MIC", contains=True) if self.ui.screen().find("MIC") else \
                    self.ui.tap_node(self.input_key())
            self.ui.wait_for("LINE")
            sync = self.ui.screen().find("SYNC")
            if sync is not None and not sync.checked:
                self.ui.tap_node(sync)
            self.ui.tap("REC")
            st = self.until(lambda: verbs.stool(self.link), lambda r: r.get("rec") == 1,
                            timeout_s=10.0)
            self.wait(self.p.take_s)
            self.ui.tap("STOP")
            st = self.until(lambda: verbs.stool(self.link), lambda r: r.get("rec") == 0)
        except UiMissing as e:
            self.missing(12, "Record: LINE, SYNC, REC, STOP", e)
            return
        self.take = str(st.get("file", ""))
        self.data["take"] = {"file": self.take, "frames": st.get("frames"),
                             "song": st.get("song"), "from": st.get("from"), "in": st.get("in")}
        self.step(12, bool(self.take) and int(st.get("frames", 0)) > 0 and st.get("song") == 1,
                  "Record: LINE, SYNC, REC, STOP: a take from line in, with the song",
                  why=f"take {self.take or '-'} frames {st.get('frames')} song={st.get('song')} "
                      f"input={st.get('in')} err={st.get('err') or '-'}",
                  manual="the take starts on the bar, in time with the song")
        try:
            self.ui.tap("ARR", contains=True)
            head = self.ui.wait_for("To Arrange", contains=True, interactive=False).text
            track = self.set_number("TRACK", 2)
            self.ui.tap("PUT")
            st = self.until(lambda: verbs.asong(self.link),
                            lambda r: r["clips"][1] == clips0[1] + 1)
        except UiMissing as e:
            self.missing(13, "ARR → TRACK 2 → PUT", e)
            return
        song = self.song_json()
        clip = (song["tracks"][1].get("clips") or [{}])[-1]
        self.data["arr"] = {"head": head, "track": track, "clips": st["clips"], "clip": clip}
        self.step(13, f"{self.p.bpm} BPM" in head and track.strip() == "TRACK 2"
                  and st["clips"][1] == clips0[1] + 1 and clip.get("bar") == 0,
                  "ARR: 'To Arrange: <song>, <bpm> BPM', TRACK 2, BAR 1, PUT: the take on "
                  "track 2 at bar 1",
                  why=f"{head!r}, {track!r}, clips {clips0} -> {st['clips']}, clip {clip}",
                  manual="no dropout heard in the take")

    # ----------------------------------------------------------------- 14-15: Seq 2, DUP

    def seq2(self) -> None:
        """Seq: the scene key (knob), ADD in the scene sheet, Done; one bar recorded;
        BOUNCE, 'Track 4': the clip on the lowest free track (UX 2000 7.1)."""
        try:
            self.open("Seq", "Sequencer")
            scenes0 = verbs.seq(self.link)["scenes"]
            self.ui.knob_to("Seq ", contains=True)
            self.click()
            self.ui.tap("ADD", contains=True)
            self.ui.tap("Done", contains=True)
            scenes = verbs.seq(self.link)["scenes"]
            self.knob_cycle_to(_BARS, "1b")
            ev = self.record("scene 2")
            clips0 = verbs.asong(self.link)["clips"]
            st = self.bounce(14, "Track 4")
        except UiMissing as e:
            self.missing(14, "scene sheet ADD, REC, BOUNCE → Track 4", e)
            return
        target = 3 if clips0[3] else clips0.index(0)
        want = [c + (t == target) for t, c in enumerate(clips0)]
        self.data["bounce4"] = {"scenes": [scenes0, scenes], "ev": ev, "state": st["bounce"],
                                "target": target, "clips": [clips0, st["clips"]]}
        self.step(14, scenes == scenes0 + 1 and ev == self.p.hits
                  and st["bounce"] == BOUNCE_DONE and st["clips"] == want,
                  f"a scene added, {self.p.hits} hits, BOUNCE → 'Track 4': on track "
                  f"{target + 1} (the lowest free), the others untouched",
                  why=f"scenes {scenes0} -> {scenes}, {ev} events, bounce={st['bounce']}, "
                      f"clips {clips0} -> {st['clips']} (want {want})")

    def dup(self) -> None:
        """Arrange: track 1's clip tapped, DUP: a copy right after it, on nothing."""
        song0 = self.song_json("song_before_dup.json")
        clip = song0["tracks"][0]["clips"][0]
        stem = Path(clip["file"]).stem
        try:
            self.open("Arrange", "Arrange")
            self.ui.tap(stem)
            self.ui.tap("DUP")
        except UiMissing as e:
            self.missing(15, "Arrange: the clip, DUP", e)
            return
        song = self.until(self.song_json,
                          lambda s: len(s["tracks"][0].get("clips", [])) == 2, timeout_s=5.0)
        clips = sorted(song["tracks"][0].get("clips", []), key=lambda c: c.get("bar", 0))
        overlaps = [(a.get("bar"), b.get("bar")) for a, b in zip(clips, clips[1:], strict=False)
                    if a.get("bar", 0) + a.get("bars", 0) > b.get("bar", 0)]
        copy = clips[1] if len(clips) == 2 else {}
        self.data["dup"] = {"clips": clips, "overlaps": overlaps}
        self.step(15, len(clips) == 2 and copy.get("bar") == clip["bar"] + clip["bars"]
                  and copy.get("file") == clip["file"] and not overlaps,
                  "Arrange: the clip tapped, DUP: the copy right after it, nothing overlapping",
                  why=f"track 1 {clips}, overlaps {overlaps}")

    # ----------------------------------------------------------------- 16, 21-22: the USB drive

    def usb_drive(self, n: int) -> tuple[str, str, Path]:
        """Setup → USB → 'SD card as USB drive' → Start → Restart: the board comes
        back as a disk (and CDC: the MSC + CDC profile); mounted on the host.
        (disk, partition, mount point)."""
        self.open("Setup", "Settings")
        self.ui.tap("USB")
        self.ui.tap_node(self.ui.seek("SD card as USB drive", contains=True))
        self.ui.tap("Start")
        self.resets_before_drive = verbs.reset_count(self.link)
        self.ui.tap("Restart")
        self.link.close()
        disk = part = ""
        try:
            disk, part = msc.wait_disk(wait=self.wait)
            root = msc.mount(part, wait=self.wait)
            self.relink(apps=False)
            return disk, part, root
        except HilError:
            self.leave_drive(disk, part, mounted=False)
            raise

    def refresh_device(self) -> None:
        """Where the board is now: a restart can bring its CDC back as another
        /dev/ttyACM, and fs.push opens the port it is given."""
        self.ctx.device = select(discover(), self.ctx.device.id)

    def relink(self, timeout_s: float = 45.0, *, apps: bool = True) -> None:
        """The CDC link again after the board re-enumerated (a restart, a profile).
        @p apps: wait for the app registry; USB transfer mode boots none, and is
        ready once USB_MSC answers."""
        deadline = time.monotonic() + timeout_s
        while True:
            try:
                self.refresh_device()
                self.link = self.ctx.open_cdc()
                if apps:
                    self.link.wait_ready()
                else:
                    verbs.usb_msc_status(self.link)
                return
            except HilError:
                if time.monotonic() >= deadline:
                    raise
                self.wait(1.0)

    def leave_drive(self, disk: str, part: str, *, mounted: bool = True) -> None:
        """Unmount and eject on the host (the board stays on USB), then "Exit and
        restart" on the board's screen. Should the screen not answer, the power
        cycle a person would give it: the STM bridge resets the ESP."""
        try:
            if part and mounted:
                msc.unmount(part)
            if disk:
                msc.eject(disk)
        except HilError as e:
            self.ctx.log(f"leaving the USB drive: {e.message}")
        try:
            if self.link is not None:
                self.link.close()
            self.relink(apps=False)
            self.ui.tap("Exit and restart", contains=True)
            self.link.close()
            self.wait(3.0)
            self.relink()
            wait_for_kit_scan(self.ctx, self.link)
        except HilError as e:
            self.ctx.log(f"Exit and restart did not bring the board back ({e.message}); "
                         "resetting it")
            self.reboot_board()
            return
        # Into the drive and out of it: two restarts the run asked for. Any more is
        # the board's own.
        now = verbs.reset_count(self.link)
        extra = now - self.resets_before_drive - DRIVE_RESTARTS
        if extra > 0:
            self.failures.append(f"{extra} reset(s) besides the USB drive's two restarts")
        self.data["resets0"] = self.data["resets0"] + now - self.resets_before_drive - max(extra, 0)

    def upload(self) -> None:
        """A sample copied onto the card from the computer, through the USB drive."""
        sample = max((self.kits[self.p.kit] / "SAMPLES").glob("*.wav"),
                     key=lambda f: f.stat().st_size)
        try:
            disk, part, root = self.usb_drive(16)
        except (UiMissing, HilError) as e:
            self.step(16, False, "Setup → USB → SD card as USB drive → Start → Restart",
                      why=f"{e.message}; {getattr(e, 'hint', '') or ''}"[:400])
            raise _Stop("the card did not come up as a USB drive") from e
        dest = root / RECORDINGS.removeprefix("/sdcard/") / self.upload_name
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(sample.read_bytes())
        os.sync()
        same = hashlib.sha256(dest.read_bytes()).hexdigest() == \
            hashlib.sha256(sample.read_bytes()).hexdigest()
        self.uploaded = True
        try:
            self.leave_drive(disk, part)
        except HilError as e:
            self.step(16, False, "unmount, eject, restart", why=e.message)
            raise _Stop("the board did not come back from the USB drive") from e
        size = verbs.fs_stat(self.link, f"{RECORDINGS}/{self.upload_name}")["size"]
        self.load_kit()                                    # a boot loads no kit
        self.data["upload"] = {"disk": disk, "mount": str(root), "same": same, "size": size}
        self.step(16, same and size == sample.stat().st_size,
                  "Setup → USB → SD card as USB drive: mounted on the computer, a sample "
                  "copied in, ejected, the board back",
                  why=f"read back {'same' if same else 'different'}, on the card {size} B "
                      f"(sent {sample.stat().st_size})",
                  manual="the speaker plays the sample")

    # ----------------------------------------------------------------- 17-18: chop, cancel

    def kit_names(self) -> list[str]:
        return [str(k["name"]) for k in verbs.kit_list(self.link)["kits"]]

    def chop(self) -> None:
        """Slicer: the uploaded sample picked, EQUAL (eight cuts), the second slice
        selected and moved, KIT: a kit of slices."""
        kits0 = set(self.kit_names())
        try:
            self.open("Slicer", "Slicer")
            self.ui.tap("pick a t", contains=True)
            self.ui.tap_node(self.ui.seek(self.upload_name))
            self.ui.wait_for("Slice 1/", contains=True, interactive=False)
            if self.ui.screen().find("HITS"):          # one hit finds one slice: equal cuts
                self.ui.tap("HITS")
            first = self.ui.wait_for("Slice 1/8", contains=True, interactive=False).text
            sel = self.set_number("SEL", 2)
            before = self.says("Slice 2/")
            self.turn("MOVE", 3)
            after = self.says("Slice 2/")
            self.ui.tap("KIT", contains=True)
            self.ui.wait_for("Kit name", interactive=False)
            self.ui.keyboard_ok()                       # the suggested name, as it is
            # The kit list is the evidence; the "Saved the Slice kit" line can be
            # gone again before the next look.
            new = self.until(lambda: set(self.kit_names()) - kits0, bool)
            saved = self.says("Saved the Slice kit") or self.says("Not saved")
            if m := _SAVED_KIT.search(saved):
                self.chop_kit = m.group(1)
        except UiMissing as e:
            self.missing(17, "Slicer: pick, SEL, MOVE, KIT", e)
            return
        self.chop_kit = sorted(new)[0] if new else self.chop_kit
        self.data["chop"] = {"first": first, "sel": sel, "slice2": [before, after],
                             "saved": saved, "kit": self.chop_kit}
        self.step(17, bool(new) and before != after and sel.endswith("2"),
                  "Slicer: the upload picked, EQUAL, SEL 2, MOVE turned, KIT, the name taken "
                  "with the keyboard's OK: a new Slice kit",
                  why=f"{first!r}; {sel!r}; slice 2 {before!r} -> {after!r}; {saved!r}; "
                      f"new kits {new}",
                  manual="each slice on its own pad (LED colours), the moved start heard")

    def cancel_bounce(self) -> None:
        """BOUNCE → 'Track 4', Cancel while it runs; Kits shows the slice kit."""
        try:
            self.open("Seq", "Sequencer")
            clips0 = verbs.asong(self.link)["clips"]
            self.ui.tap("BOUNCE")
            self.ui.tap("Track 4")
            self.ui.tap("Cancel", timeout_s=3.0)
            st = self.until(lambda: verbs.asong(self.link),
                            lambda r: r["bounce"] != BOUNCE_RUNNING, timeout_s=10.0)
            self.home()
            self.ui.tap("Kits")
            self.ui.tap("SLICE")                       # where the Slicer said it went
            listed = self.chop_kit and self.ui.seek(self.chop_kit) is not None
        except UiMissing as e:
            self.missing(18, "BOUNCE → Track 4 → Cancel; Kits", e)
            return
        early = st["bounce"] == BOUNCE_DONE
        self.data["cancel"] = {"state": st["bounce"], "clips": [clips0, st["clips"]],
                               "finished_first": early, "kit_listed": bool(listed)}
        self.step(18, st["bounce"] != BOUNCE_RUNNING and (early or st["clips"] == clips0)
                  and bool(listed),
                  "BOUNCE → Track 4 → Cancel while it runs; the slice kit in Kits, SLICE",
                  why=f"bounce={st['bounce']} clips {clips0} -> {st['clips']}, "
                      f"kit {self.chop_kit or '-'} listed={bool(listed)}")

    # ----------------------------------------------------------------- 19-20: mix, export

    def mix(self) -> None:
        """Arrange: MIX, a finger on each track's fader, Done. (A header held mutes
        its track since Arrange 0.34; MIX is the key beside the song's name.)"""
        try:
            self.open("Arrange", "Arrange")
            self.ui.tap("MIX")
            self.ui.wait_for("Mix", interactive=False)
            sliders = sorted((n for n in self.ui.screen().controls() if n.cls == "slider"),
                             key=lambda n: n.x)
            used = [i for i, tr in enumerate(self.song_json()["tracks"]) if tr.get("clips")]
            for i, s in enumerate(sliders):
                if i in used:
                    verbs.touch_tap(self.link, s.center[0], s.y + s.h * (2 + i) // 6)
                    self.wait(0.4)
            shown = [w for w in self.ui.screen().labels() if "dB" in w]
            self.ui.tap("Done", contains=True)
        except UiMissing as e:
            self.missing(19, "Arrange: MIX → faders → Done", e)
            return
        vols = [round(tr.get("volume", 1.0), 3) for tr in
                self.until(self.song_json, lambda s: any(abs(tr.get("volume", 1.0) - 1) > 0.01
                                                         for tr in s["tracks"]), timeout_s=5.0)
                ["tracks"]]
        moved = [i for i in used if abs(vols[i] - 1.0) > 0.01]
        self.data["mix"] = {"shown": shown, "volumes": vols, "used": used}
        self.step(19, moved == used and len(set(vols[i] for i in used)) == len(used),
                  "Arrange: MIX opens the mix; a finger on each used track's fader; "
                  "Done: the levels in song.json",
                  why=f"faders show {shown}; song.json volumes {vols} (tracks {used})")

    def export(self) -> None:
        """The song's sheet: Export to Ableton Live .als; the Set beside song.json."""
        try:
            self.open("Arrange", "Arrange")
            self.ui.tap(self.song)
            self.ui.tap("EDIT")
            self.ui.tap("Export to Ableton Live", contains=True)
            done = self.until(lambda: self.says(".als"), bool)
        except UiMissing as e:
            self.missing(20, "the song's sheet → Export to Ableton Live", e)
            return
        als = f"{self.song_dir}/{self.song}.als"
        try:
            size = verbs.fs_stat(self.link, als)["size"]
        except HilError:
            size = 0
        self.data["export"] = {"said": done, "als": als, "size": size}
        self.step(20, size > 0, f"Export to Ableton Live .als: {self.song}.als beside "
                                "song.json",
                  why=f"the screen said {done!r}; {als} is {size} bytes")
        self.home()

    def ableton(self) -> None:
        """The card as a USB drive again: the Set and its clips read on the computer,
        as Ableton opens them; then, the board back, the song and the pads as before."""
        before_pads = [self.pad_shape(p) for p in range(16)]
        before_song = self.song_json("song_before_drive.json")
        try:
            disk, part, root = self.usb_drive(21)
        except (UiMissing, HilError) as e:
            self.step(21, False, "Setup → USB → SD card as USB drive → Start → Restart",
                      why=f"{e.message}; {getattr(e, 'hint', '') or ''}"[:400])
            raise _Stop("the card did not come up as a USB drive") from e
        folder = root / SONGS_ROOT.removeprefix("/sdcard/") / self.song
        als = folder / f"{self.song}.als"
        self.step(21, als.is_file(), "Setup → USB → SD card as USB drive: the computer "
                                     "mounts the card",
                  why=f"{als} is not on the mounted card")
        if als.is_file():
            local = self.ctx.workdir / als.name
            local.write_bytes(als.read_bytes())
            (self.ctx.workdir / "song.json").write_bytes((folder / "song.json").read_bytes())
            self.artifacts += [Artifact(str(local), "application/gzip", "live_set"),
                               Artifact(str(self.ctx.workdir / "song.json"), "application/json",
                                        "song")]
            live = parse_live_set(local.read_bytes())
            song = json.loads((folder / "song.json").read_text(encoding="utf-8"))
            diffs = compare_set(song, live, self.p.bpm)
            missing = [c["path"] for tr in live["tracks"] for c in tr["clips"]
                       if not (folder / c["path"]).is_file()
                       or (folder / c["path"]).stat().st_size <= 44]
            self.data["live_set"] = {"tempo": live["tempo"],
                                     "tracks": [len(tr["clips"]) for tr in live["tracks"]],
                                     "diffs": diffs, "missing": missing}
            self.step(22, not diffs and not missing,
                      "on the computer: the Set's tempo, tracks and clip bars as song.json, "
                      "every clip's file beside it",
                      why="; ".join(diffs + [f"missing media {m}" for m in missing]),
                      manual="Ableton Live opens the Set with no missing media")
        try:
            self.leave_drive(disk, part)
        except HilError as e:
            raise _Stop(f"the board did not come back from the USB drive: {e.message}") from e
        after_song = self.song_json("song_after_drive.json")
        self.load_kit()                                    # a boot loads no kit
        after_pads = [self.pad_shape(p) for p in range(16)]
        diffs = []
        if after_song != before_song:
            diffs.append("song.json differs after the restart")
        if after_pads != before_pads:
            pairs = zip(before_pads, after_pads, strict=False)
            diffs.append(f"pads {[i + 1 for i, (a, b) in enumerate(pairs) if a != b]} differ")
        self.data["restart"] = {"diffs": diffs}
        self.step(5, not diffs, "after the restart: song.json and every pad's sample, region, "
                                "level and pan as saved",
                  why="; ".join(diffs))

    def input_key(self):
        """The Recorder's input key, whatever it shows (MIC, LINE, PADS, USB)."""
        sc = self.ui.screen()
        for c in sc.controls():
            if c.text.strip() in ("MIC", "LINE", "PADS", "USB"):
                return c
        raise UiMissing("TIMEOUT", "no input key on the Recorder",
                        hint="could press: " + ", ".join(sc.labels()[:20]))

    # ----------------------------------------------------------------- cleanup

    def delete_song(self) -> None:
        """Arrange → the song's name → EDIT → Delete song → Delete, as a person does."""
        self.open("Arrange", "Arrange")
        self.ui.tap(self.song)
        self.ui.tap("EDIT")
        self.ui.tap("Delete song")
        self.ui.tap("Delete")
        self.until(lambda: verbs.asong(self.link), lambda r: r["project"] != self.song,
                   timeout_s=10.0)

    def reopen(self, name: str) -> None:
        self.open("Arrange", "Arrange")
        self.ui.tap(str(verbs.asong(self.link)["project"]))
        self.ui.tap_node(self.ui.seek(name))
        self.until(lambda: verbs.asong(self.link), lambda r: r["project"] == name,
                   timeout_s=10.0)

    def put_kit_back(self) -> None:
        """The run saved the fixture kit from the Sampler; the fixture's kit.json goes back."""
        kit_dir = self.kits[self.p.kit]
        self.link.close()
        self.refresh_device()
        try:
            fs.push(self.ctx.device, kit_dir / "kit.json",
                    f"{KITS_ROOT}/{kit_dir.name}/kit.json")
        finally:
            self.link = self.ctx.open_cdc()

    def delete_kit(self, name: str) -> None:
        """The Slicer's kit off the card (its folder holds the kit's name)."""
        kit_dir = next((f"{KITS_ROOT}/{e['name']}" for e in
                        verbs.fs_list(self.link, KITS_ROOT)["entries"]
                        if e["dir"] and e["name"].casefold() == name.casefold()), None)
        if kit_dir is None:
            return
        for sub in (f"{kit_dir}/SAMPLES", kit_dir):
            for e in verbs.fs_list(self.link, sub)["entries"]:
                if not e["dir"]:
                    verbs.fs_delete(self.link, f"{sub}/{e['name']}")
        verbs.fs_delete(self.link, f"{kit_dir}/SAMPLES")
        verbs.fs_delete(self.link, kit_dir)

    def cleanup(self) -> None:
        undo: list[tuple[str, Callable[[], Any]]] = [
            ("stop", lambda: verbs.seq(self.link, "STOP")),
            ("home", self.home)]
        if self.song and not self.p.keep:
            undo.append(("delete the song", self.delete_song))
        if self.song0 and self.song0 != self.song:
            undo.append(("reopen " + self.song0, lambda: self.reopen(self.song0)))
        if self.take and not self.p.keep:
            undo.append(("delete the take", lambda: verbs.fs_delete(self.link, self.take)))
        if self.uploaded and not self.p.keep:
            undo.append(("delete the upload",
                         lambda: verbs.fs_delete(self.link, f"{RECORDINGS}/{self.upload_name}")))
        if self.chop_kit and not self.p.keep:
            undo.append(("delete the slice kit", lambda: self.delete_kit(self.chop_kit)))
        if self.kit_saved:
            undo.append(("put the kit back", self.put_kit_back))
        for what, fn in undo:
            try:
                fn()
            except Exception as e:  # noqa: BLE001 - one undo failing must not stop the rest
                self.ctx.log(f"cleanup: {what}: {type(e).__name__}: {e}")
        audio_state.put_back(self.link, self.audio_snap, self.ctx.log, self.data)


class StudioSessionScenario:
    name = "studio_session"
    Params = Params
    description = "the 21.10.2026 studio session played on the screen: kit, pattern, song, .als"

    def run(self, ctx: Context, params: Params, progress: Progress) -> Report:
        root = Path(params.fixture or os.environ.get(FIXTURE_ENV, "")).expanduser()
        if not str(root) or str(root) == ".":
            return self._bench(f"no fixture: --fixture or ${FIXTURE_ENV}", {})
        kits, problems = verify_fixture(root)
        if problems:
            return self._bench(f"fixture {root}: " + "; ".join(problems[:5]), {})
        if params.kit not in kits:
            return self._bench(f"the fixture has no kit {params.kit!r} (has {sorted(kits)})", {})

        run = _Run(ctx, params, kits)
        # (last step the phase covers, what it is, the phase)
        phases: list[tuple[int, str, Callable[[], None]]] = [
            (1, "kit and pads", run.kit), (4, "pad editor", run.editor), (5, "COPY", run.copy),
            (6, "a song of the run", run.new_song), (9, "Seq", run.seq),
            (8, "Pad Mix", run.padmix), (10, "bounce to track 1", run.bounce1),
            (11, "Arrange", run.arrange), (13, "line-in take", run.line_take),
            (14, "Seq 2 bounced", run.seq2), (15, "DUP", run.dup),
            (16, "a sample over the USB drive", run.upload), (17, "chop", run.chop),
            (18, "cancel a bounce", run.cancel_bounce), (19, "song mix", run.mix),
            (20, "export .als", run.export), (22, "the Set on the computer", run.ableton),
        ]
        if params.stop_after:
            last = next((i for i, (n, _, _) in enumerate(phases) if n >= params.stop_after),
                        len(phases) - 1)
            phases = phases[:last + 1]
        try:
            progress(0, len(phases), "provision the fixture kits")
            run.setup()
            for i, (_, what, phase) in enumerate(phases):
                ctx.check_cancelled()
                progress(i, len(phases), what)
                phase()
            progress(len(phases), len(phases), "done")
        except _Stop as e:
            run.failures.append(str(e))
        except HilError as e:
            # The board did not answer in time: what kept its CDC loop is the
            # evidence (CDC_STATS maxcmd= names the verb; read-and-cleared).
            run.failures.append(f"{e.code}: {e.message}")
            try:
                run.data["cdc_stats"] = run.link.transact("CDC_STATS", "CDCSTATS:").line
            except HilError as e2:
                run.data["cdc_stats"] = f"unreadable: {e2.message}"
        finally:
            if run.link is not None:
                if run.data.get("resets0") is not None and not ctx.cancelled.is_set():
                    try:
                        delta = verbs.reset_count(run.link) - run.data["resets0"]
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
        return Report(passed=False, summary=f"studio_session: BENCH — {why}", data=data,
                      artifacts=[], exit_code=2)

    @staticmethod
    def _report(run: _Run) -> Report:
        passed = not run.failures
        steps = run.data["steps"]
        auto = sum(1 for s in steps.values() if s["status"] == "pass")
        to_watch = sorted(int(n) for n, s in steps.items() if s.get("manual"))
        tail = (f" ({auto} steps done on the screen; to watch or listen to: "
                f"{', '.join(map(str, to_watch))})")
        summary = ("studio_session: PASS" + tail if passed
                   else "studio_session: FAIL — " + "; ".join(run.failures))
        return Report(passed=passed, summary=summary, data=run.data, artifacts=run.artifacts,
                      exit_code=0 if passed else 1)


SCENARIO = register(StudioSessionScenario())
