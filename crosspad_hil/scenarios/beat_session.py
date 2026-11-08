"""A beat made from nothing on the screen, as a player makes one, every wait timed.

A new song; a drum kit track, a groove recorded with the song playing on; a second kit
picked while it plays and a part recorded over the groove; STOP and the tracks ARM left
freeze by themselves; PLAY and both are heard from the first bar. Nothing is asked of a
deep menu that the player would not tap anyway: what the run measures is what a player
waits for, and each wait has a budget it fails over.

It drives the screen (`UI_TREE` + `TOUCH_TAP`, the pads) and reads results with state
verbs only: `ASONG`, `ASONG_SEQ`, `KIT_STATUS`, `KIT2`, `SMPL_ENGINE` (the native
engine's `notes=`, the voices it started), `MIX_LVL` (per channel peaks) and song.json.
While the song plays a sampler thread logs every ~0.3 s: position, the engines' note
counters and the Sampler / Kit 2 / Song peaks, saved as an artifact; a pass of the
groove that started no voice of the drum kit (or, frozen, put nothing on the Song
channel) is a gap the player heard as silence.

The steps:
  1  a new song (Arrange -> the songs -> NEW): the playhead on bar one
  2  ADD TRACK -> KITS -> DRUMS -> the drum kit, PICK: the kit the row said, loaded
  3  REC, the groove, REC: the song plays on and the groove is heard on the next pass
  4  ADD TRACK -> KITS -> PERF -> a second kit, PICK, the song playing: the groove goes on
  5  REC punched in anywhere over the looping groove: the part lands on it (the take goes
     round the song), the song as long as before; both heard on the next pass
  6  STOP: the track ARM left freezes by itself, within its budget
  7  PLAY: the frozen groove from its file, the part live, no ring underrun
  8  ADD TRACK -> KITS -> PITCH PLAY -> a sample, back: a melody recorded on the loop,
     heard with the rest
  9  the top bar's BPM turned: on the stop every track ARM is not on freezes again
 10  EDIT -> a loop region over the song -> Duplicate x2: eight bars, the renders found
 11  another song and back: it plays at once, nothing rendered again

The speakers stay at `OUTVOL 2 <outvol>`. The run's song goes off the card at the end,
the kit that was loaded comes back and the song that was open is reopened.
"""
from __future__ import annotations

import json
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from crosspad_hil import verbs
from crosspad_hil.errors import ENV, HilError
from crosspad_hil.scenarios.base import Artifact, Context, Progress, Report, register
from crosspad_hil.scenarios.song_elements import PLAYING, TILE_PLAY_X, _Stop
from crosspad_hil.scenarios.song_elements import _Run as _StudioRun
from crosspad_hil.ui import TILE_REC_X, UiMissing

BEAT_S = 0.5             # a beat at the song's 120 BPM
BAR_S = 4 * BEAT_S
COUNT_IN_S = BAR_S + 0.3  # REC's one-bar count-in, and the CDC round trip
SAMPLE_S = 0.3           # the listening thread's period
KIT_WAIT_S = 30.0        # the longest any kit load is waited for before the run gives up
HEADER_H = 28            # a track header's height: the column holding it is taller

# What a player may wait for, each a budget in seconds the run fails over.
BUDGET = {
    "new song": 3.0,
    "drum kit": 6.0,          # PICK to the kit loaded
    "second kit": 6.0,        # PICK to the second engine holding it
    "groove heard": 2 * BAR_S + 0.5,   # REC out to the first voice of the take
    "freeze start": 2.0,      # STOP to FREEZING on the track's header
    "freeze": 10.0,           # STOP to the freeze written
    "play": 1.5,              # PLAY to the position moving
    "instrument": 6.0,        # a sample picked to the instrument ready
    "refreeze": 15.0,         # a tempo turned, the stop, to every track frozen again
    "duplicate": 15.0,        # Duplicate x2 to every track frozen again
    "reopen": 8.0,            # the song opened to the first voice after PLAY
}


@dataclass
class Params:
    kit: str = field(default="808_DISTORTED", metadata={
        "help": "the drum kit (the browser's DRUMS list)"})
    second_kit: str = field(default="BLACK TESLA", metadata={
        "help": "the second kit (the browser's PERF list)"})
    sample: str = field(default="808_Clean", metadata={
        "help": "Pitch Play's sample for the melody track"})
    bpm_turn: int = field(default=-10, metadata={"help": "detents the top bar's BPM is turned"})
    outvol: int = field(default=15, metadata={
        "help": "OUTVOL 2 level (speakers and phones) for the whole run"})
    keep: bool = field(default=False, metadata={"help": "leave the song on the card"})
    job_timeout_s: float = field(default=60.0, metadata={
        "help": "for a freeze or a save to finish"})
    # The studio's hands read these; this run records no audio track.
    take_s: float = 0.0
    hits: int = 0


def _peaks(link: Any) -> dict[str, float]:
    return {str(c["name"]): max(float(c["pk_l"]), float(c["pk_r"]))
            for c in verbs.mix_lvl(link)["channels"]}


class _Listener:
    """A thread that logs what the board plays while the song runs."""

    def __init__(self, run: _BeatRun) -> None:
        self.run = run
        self.rows: list[dict[str, Any]] = []
        self._stop = threading.Event()
        self._th: threading.Thread | None = None

    def __enter__(self) -> _Listener:
        self._th = threading.Thread(target=self._loop, daemon=True)
        self._th.start()
        return self

    def __exit__(self, *exc: Any) -> None:
        self._stop.set()
        if self._th:
            self._th.join(5.0)

    def _loop(self) -> None:
        while not self._stop.is_set() and not self.run.ctx.cancelled.is_set():
            link = self.run.link          # a file pull reopens it
            try:
                s = verbs.asong(link)
                e = verbs.smpl_engine(link)
                k = verbs.kit2(link)
                pk = _peaks(link)
                self.rows.append({
                    "t": round(time.monotonic() - self.run.t0, 2), "play": s["play"],
                    "pos": float(s["pos"]), "under": int(s.get("under", 0)),
                    "notes": int(e.get("notes", 0)), "notes2": int(k.get("notes", 0)),
                    "smp": round(pk.get("Sampler", 0.0), 4), "kit2": round(pk.get("Kit 2", 0.0), 4),
                    "pitch": round(pk.get("Pitched", 0.0), 4),
                    "song": round(pk.get("Song", 0.0), 4)})
            except HilError as err:
                self.rows.append({"t": round(time.monotonic() - self.run.t0, 2),
                                  "err": err.message})
            self._stop.wait(SAMPLE_S)

    def since(self, t: float) -> list[dict[str, Any]]:
        return [r for r in self.rows if "err" not in r and r["t"] >= t]


class _BeatRun(_StudioRun):
    def __init__(self, ctx: Context, params: Params) -> None:
        super().__init__(ctx, params)
        self.t0 = time.monotonic()
        self.waits: dict[str, float] = {}
        self.data["waits"] = self.waits
        self.listen: _Listener | None = None
        self.groove_notes = 0
        self.groove_bars = 2

    # ----------------------------------------------------------------- measuring

    def timed(self, name: str, ask: Callable[[], Any], done: Callable[[Any], bool],
              limit_s: float) -> tuple[Any, float]:
        """Polls @p ask until @p done, at most @p limit_s; the time is a wait the player
        had, judged against its budget."""
        t = time.monotonic()
        got = self.until(ask, done, timeout_s=limit_s)
        took = round(time.monotonic() - t, 2)
        self.waits[name] = took
        if not done(got):
            self.failures.append(f"{name}: not done after {limit_s:.0f}s")
        elif name in BUDGET and took > BUDGET[name]:
            self.failures.append(f"{name}: {took:.2f}s, over its {BUDGET[name]:.1f}s budget")
        return got, took

    def header(self, track: int) -> Any:
        """Track @p track's header on the timeline (0-based)."""
        sc = self.ui.screen()
        want = f"{track + 1:02d}"
        for n in sc.controls():
            if n.h == HEADER_H and sc.words(n)[:1] == [want]:
                return n
        raise UiMissing("TIMEOUT", f"no header for track {track + 1}",
                        hint="could press: " + ", ".join(sc.labels()[:20]))

    def header_says(self, track: int) -> str:
        return " ".join(self.ui.screen().words(self.header(track)))

    def to_bar_one(self) -> None:
        """Until the playing song is back on its first bar."""
        self.until(self.song, lambda r: float(r["pos"]) < 0.3,
                   timeout_s=3 * self.groove_bars * BAR_S)

    def listen_passes(self, n: int) -> None:
        """Long enough for @p n whole passes of the groove's block, from wherever the song is."""
        self.wait((n + 1) * self.groove_bars * BAR_S)

    def hit_bars(self, pattern: list[tuple[float, int, int]], bars: int) -> int:
        """Plays @p pattern ((beat, pad, velocity) in a bar) for @p bars bars from now, on
        the beat grid of the song: the take starts on the bar the count-in ended on."""
        start = time.monotonic()
        hits = sorted((b * BAR_S + beat * BEAT_S, pad, vel)
                      for b in range(bars) for beat, pad, vel in pattern)
        for at, pad, vel in hits:
            delay = start + at - time.monotonic()
            if delay > 0:
                self.wait(delay)
            self.hit(pad, vel)
        return len(hits)

    # ----------------------------------------------------------------- the steps

    def song_start(self) -> None:
        t = time.monotonic()
        self.new_song()
        self.waits["new song"] = round(time.monotonic() - t, 2)
        bar = self.ui.screen()
        self.need(1, "STOP 1.1" in " ".join(bar.labels()),
                  "NEW: the playhead of the empty song on bar one",
                  "top bar: " + ", ".join(bar.labels()[:6]))

    def pick_kit(self, step: int, tab: str, kit: str) -> None:
        self.ui.tap("ADD TRACK", contains=True)
        self.ui.tap("KITS")
        self.ui.tap(tab)
        self.ui.tap_node(self.ui.seek(kit))      # a tap on the row picks it
        self.wait(0.5)
        if self.ui.screen().find("PICK"):
            self.ui.tap("PICK")

    def drum_track(self) -> None:
        try:
            self.pick_kit(2, "DRUMS", self.p.kit)
        except UiMissing as e:
            self.missing(2, "ADD TRACK -> KITS -> DRUMS -> kit, PICK", e)
            raise _Stop("no drum track") from e
        k, took = self.timed("drum kit", lambda: verbs.kit_status(self.link),
                             lambda r: not r["loading"] and r["name"] == self.p.kit, KIT_WAIT_S)
        self.need(2, k["name"] == self.p.kit,
                  f"ADD TRACK -> KITS -> DRUMS -> {self.p.kit}, PICK: that kit loaded",
                  f"loaded {k['name']!r} in {took:.2f}s")

    GROOVE = [(0.0, 0, 115), (0.5, 2, 80), (1.0, 1, 110), (1.5, 2, 80),
              (2.0, 0, 115), (2.5, 2, 80), (3.0, 1, 110), (3.5, 2, 80)]

    def groove(self) -> None:
        self.tile(TILE_REC_X)
        self.wait(COUNT_IN_S)
        self.groove_notes = self.hit_bars(self.GROOVE, 2)
        self.wait(0.2)
        self.tile(TILE_REC_X)                   # REC again: out, the song plays on
        t_out = time.monotonic() - self.t0
        blocks = verbs.asong_seq(self.link, 0)["blocks"]
        playing = self.song().get("play") == PLAYING
        # A take shorter than the song goes round to its end (a two-bar groove in the
        # empty song's four bars is one block of four): the block, not its length.
        self.need(3, playing and len(blocks) == 1 and blocks[0]["bar"] == 0
                  and blocks[0]["bars"] >= 2 and blocks[0]["notes"] == self.groove_notes,
                  "REC, two bars of groove, REC: one block on bar one holding the groove, "
                  "the song playing on", f"blocks {blocks}, play={playing}")
        self.groove_bars = int(blocks[0]["bars"]) if blocks else 2
        # Heard: the drum engine starts voices again after the punch-out.
        n0 = int(verbs.smpl_engine(self.link)["notes"])
        _, took = self.timed("groove heard", lambda: int(verbs.smpl_engine(self.link)["notes"]),
                             lambda n: n >= n0 + self.groove_notes // 2, 3 * BAR_S)
        self.listen_passes(2)
        rows = self.listen.since(t_out) if self.listen else []
        passes = self.passes(rows, 0, self.groove_bars)
        silent = [p for p in passes if p["notes"] == 0]
        self.need(3, bool(passes) and not silent,
                  "the groove heard on every pass after REC, without a stop",
                  f"passes {passes}")

    @staticmethod
    def passes(rows: list[dict[str, Any]], bar: int, bars: int) -> list[dict[str, Any]]:
        """Each whole pass of the bars [bar, bar + bars) in @p rows: the voices the drum
        engine started over it and the loudest Song channel peak."""
        out: list[dict[str, Any]] = []
        cur: list[dict[str, Any]] = []
        for r in rows:
            inside = bar <= r["pos"] < bar + bars and r["play"] == PLAYING
            # The song going round starts the next pass, even when the bars are all of it.
            if cur and (not inside or r["pos"] < cur[-1]["pos"]):
                out.append(cur)
                cur = []
            if inside:
                cur.append(r)
        if cur:
            out.append(cur)                 # the look ended in a pass: whole or not, below
        # A pass is whole when it was seen from its first bar to its last.
        # A sample is ~1 s apart (four verbs): within three quarters of a bar of each edge.
        whole = [p for p in out if p[0]["pos"] < bar + 0.75 and p[-1]["pos"] > bar + bars - 0.75]
        return [{"from": p[0]["t"], "notes": p[-1]["notes"] - p[0]["notes"],
                 "song": max(r["song"] for r in p), "smp": max(r["smp"] for r in p)}
                for p in whole]

    def second_kit(self) -> None:
        t_add = time.monotonic() - self.t0
        try:
            self.pick_kit(4, "PERF", self.p.second_kit)
        except UiMissing as e:
            self.missing(4, "ADD TRACK -> KITS -> PERF -> kit, PICK", e)
            raise _Stop("no second kit") from e
        k, took = self.timed("second kit", lambda: verbs.kit2(self.link),
                             lambda r: int(r.get("state", 0)) == 2 and int(r.get("active", 0)) == 1,
                             KIT_WAIT_S)
        self.listen_passes(2)
        rows = self.listen.since(t_add) if self.listen else []
        passes = self.passes(rows, 0, self.groove_bars)
        silent = [p for p in passes if p["notes"] == 0]
        head = self.header_says(1)
        self.need(4, self.p.second_kit in head, f"the new track's header names {self.p.second_kit}",
                  f"track 2 says {head!r}, KIT2 have={k.get('have')}")
        self.need(4, int(k.get("state", 0)) == 2 and not silent and bool(passes),
                  f"ADD TRACK -> KITS -> PERF -> {self.p.second_kit}, PICK while it plays: "
                  "the second engine holds it and the groove goes on",
                  f"KIT2 have={k.get('have')} in {took:.2f}s; passes {passes}")

    PART = [(0.0, 5, 100), (1.5, 6, 90), (2.0, 7, 100), (3.0, 6, 90)]

    def part(self) -> None:
        """REC punched in while it plays, onto the armed (new) track."""
        # Punched in wherever the loop is, the part from the next bar line: with LOOP
        # on the take goes round the song, so the part lands on the groove wherever
        # it falls -- across the song's end too -- and the song stays as long.
        # The hard case: the last beat of the loop, the part from bar one again.
        self.until(self.song, lambda r: float(r["pos"]) > self.groove_bars - 0.4,
                   timeout_s=(self.groove_bars + 1) * BAR_S)
        self.tile(TILE_REC_X)
        pos = float(self.song()["pos"])
        self.wait((int(pos) + 1 - pos) * BAR_S)
        part_notes = self.hit_bars(self.PART, 2)
        self.tile(TILE_REC_X)
        t_out = time.monotonic() - self.t0
        blocks = verbs.asong_seq(self.link, 1)["blocks"]
        notes = sum(int(b["notes"]) for b in blocks)
        length = int(self.song()["len"])
        self.need(5, notes == part_notes and length == self.groove_bars,
                  "REC punched in anywhere over the looping groove, REC: the part on the second "
                  "kit's track, the song as long as before",
                  f"blocks {blocks} ({notes} of {part_notes} notes), len {length} "
                  f"(groove {self.groove_bars})")
        k0 = int(verbs.kit2(self.link).get("notes", 0))
        self.listen_passes(2)
        k1 = int(verbs.kit2(self.link).get("notes", 0))
        rows = self.listen.since(t_out) if self.listen else []
        drums = self.passes(rows, 0, self.groove_bars)
        self.need(5, k1 > k0 and drums and not [p for p in drums if p["notes"] == 0],
                  "both heard on the passes after: the part on Kit 2, the groove on the kit",
                  f"Kit 2 voices {k1 - k0}; groove passes {drums}")

    def stop_and_freeze(self) -> None:
        self.tile(TILE_PLAY_X)                  # PLAY while playing: STOP
        t = time.monotonic()
        self.until(self.song, lambda r: r.get("play") != PLAYING)
        start = self.until(lambda: self.header_says(0), lambda h: "FREEZ" in h,
                           timeout_s=BUDGET["freeze start"] * 3)
        self.waits["freeze start"] = round(time.monotonic() - t, 2)
        if "FREEZ" not in start:
            self.failures.append(f"freeze start: the groove's header still says {start!r}")
        elif self.waits["freeze start"] > BUDGET["freeze start"]:
            self.failures.append(f"freeze start: {self.waits['freeze start']:.2f}s, over "
                                 f"its {BUDGET['freeze start']:.1f}s budget")
        st = self.until(self.song, lambda r: r.get("bounce") in (2, 3),
                        timeout_s=self.p.job_timeout_s)
        self.waits["freeze"] = round(time.monotonic() - t, 2)
        song = self.until(lambda: self.pull(self.song_file(self.songs[-1]), "song.json"),
                          lambda d: bool(d["tracks"][0].get("freeze")), timeout_s=15.0)
        fz = song["tracks"][0].get("freeze") or {}
        self.need(6, st.get("bounce") == 2 and bool(fz.get("clips"))
                  and self.waits["freeze"] <= BUDGET["freeze"],
                  "STOP: the groove's track (ARM left it) freezes by itself",
                  f"bounce={st.get('bounce')} in {self.waits['freeze']:.2f}s, "
                  f"freeze {json.dumps(fz)[:200]}")

    def play_back(self) -> None:
        under0 = int(self.song().get("under", 0))
        self.tile(TILE_PLAY_X)
        t_play = time.monotonic() - self.t0
        self.timed("play", self.song, lambda r: float(r.get("pos", 0)) > 0.05, 5.0)
        k0 = int(verbs.kit2(self.link).get("notes", 0))
        self.listen_passes(2)
        k1 = int(verbs.kit2(self.link).get("notes", 0))
        rows = self.listen.since(t_play) if self.listen else []
        drums = self.passes(rows, 0, self.groove_bars)
        under = int(self.song().get("under", 0)) - under0
        self.need(7, bool(drums) and all(p["song"] > 0.005 for p in drums) and k1 > k0
                  and under == 0,
                  "PLAY: the frozen groove from its file, the part live on Kit 2, no underrun",
                  f"groove passes {drums}; Kit 2 voices {k1 - k0}; underruns {under}")
        self.stop()


class _BeatSteps(_BeatRun):
    """Steps 8-11: the song grows past the first loop."""

    MELODY = [(0.0, 0, 100), (1.0, 2, 90), (2.0, 4, 100), (3.0, 7, 90)]

    def frozen(self, tracks: list[int]) -> bool:
        heads = [self.header_says(t) for t in tracks]
        return all("FRZ" in h for h in heads)

    def melody(self) -> None:
        self.data["mem_before_pitch"] = verbs.mem(self.link)
        try:
            self.ui.tap("ADD TRACK", contains=True)
            self.ui.tap("KITS")
            self.ui.tap("PITCH PLAY")
            t = time.monotonic()
            self.wait(1.0)
            # With no sample picked yet, Pitch Play opens to pick one; with one, the
            # track plays it and Arrange stays.
            in_pitch_play = self.ui.screen().running == "PitchPlay"
            if in_pitch_play and self.ui.screen().find("Pick a sample"):
                self.ui.tap("Pick a sample")
                self.ui.tap_node(self.ui.seek(self.p.sample, contains=True))
                t = time.monotonic()
            st = self.until(lambda: verbs.inst_status(self.link),
                            lambda r: r.get("state") == "ready", timeout_s=KIT_WAIT_S)
            self.waits["instrument"] = round(time.monotonic() - t, 2)
            if in_pitch_play:
                self.power()                    # back to Arrange
            self.until(self.ui.screen, lambda sc: sc.running == "Arrange", timeout_s=5.0)
        except UiMissing as e:
            self.missing(8, "ADD TRACK -> KITS -> PITCH PLAY -> sample, back", e)
            raise _Stop("no melody track") from e
        head = self.header_says(2)
        self.need(8, st.get("state") == "ready" and "PITCH" in head
                  and self.waits["instrument"] <= BUDGET["instrument"],
                  f"PITCH PLAY -> {self.p.sample}, a power click back: a Pitch track, armed",
                  f"INST {st}, track 3 says {head!r}, {self.waits['instrument']:.2f}s")
        self.tile(TILE_PLAY_X)
        self.until(self.song, lambda r: r.get("play") == PLAYING, timeout_s=5.0)
        self.to_bar_one()
        self.tile(TILE_REC_X)
        notes = self.hit_bars(self.MELODY, 2)
        self.tile(TILE_REC_X)
        t_out = time.monotonic() - self.t0
        blocks = verbs.asong_seq(self.link, 2)["blocks"]
        got = sum(int(b["notes"]) for b in blocks)
        self.listen_passes(2)
        rows = self.listen.since(t_out) if self.listen else []
        pitched = max((r.get("pitch", 0.0) for r in rows), default=0.0)
        drums = self.passes(rows, 0, self.groove_bars)
        self.need(8, got == notes and pitched > 0.005 and drums
                  and all(p["song"] > 0.005 for p in drums),
                  "REC on the loop, a melody, REC: on the Pitch track and heard with the rest",
                  f"blocks {blocks} ({got} of {notes}); Pitched peak {pitched}; groove {drums}")
        self.stop()

    def tempo(self) -> None:
        self.stop()
        bpm0 = float(self.song()["bpm"])
        sc = self.ui.screen()
        tile = next(n for n in sc.controls() if sc.words(n)[:1] == ["BPM"] and n.y < 40)
        self.ui.tap_node(tile)
        verbs.enc_rotate(self.link, self.p.bpm_turn)
        self.wait(0.4)
        verbs.enc_press(self.link)
        t = time.monotonic()
        bpm = float(self.until(self.song, lambda r: float(r["bpm"]) != bpm0, timeout_s=5.0)["bpm"])
        done = self.until(lambda: self.frozen([0, 1]), bool, timeout_s=BUDGET["refreeze"] * 2)
        self.waits["refreeze"] = round(time.monotonic() - t, 2)
        self.need(9, bpm == bpm0 + self.p.bpm_turn and done
                  and self.waits["refreeze"] <= BUDGET["refreeze"],
                  "the BPM turned on the top bar: the kit tracks freeze again by themselves",
                  f"bpm {bpm0} -> {bpm}; frozen={done} after {self.waits['refreeze']:.2f}s")

    def duplicate(self) -> None:
        bars0 = int(self.song()["len"])
        rgn = self.set_region(0, bars0)
        self.edit_menu()
        self.menu_item("Duplicate the region")
        t = time.monotonic()
        song = self.until(self.song, lambda r: int(r["len"]) >= 2 * bars0, timeout_s=10.0)
        self.edit_menu()
        self.menu_item("Clear the loop region")
        done = self.until(lambda: self.frozen([0, 1]), bool, timeout_s=BUDGET["duplicate"] * 2)
        self.waits["duplicate"] = round(time.monotonic() - t, 2)
        blocks = [len(verbs.asong_seq(self.link, k)["blocks"]) for k in range(3)]
        self.need(10, int(song["len"]) == 2 * bars0 and done
                  and self.waits["duplicate"] <= BUDGET["duplicate"],
                  f"a loop region over the song ({rgn}), Duplicate x2: twice as long, frozen again",
                  f"len {bars0} -> {song['len']}, blocks per track {blocks}, frozen={done} "
                  f"after {self.waits['duplicate']:.2f}s")

    def pick_song(self, name: str) -> None:
        """The song list a page at a time until a row says @p name: back to its first page
        ("Back ..."), then on ("More ..."). It opens on the open song's page."""
        def page(word: str) -> bool:
            # The row is at an end of the list: scrolled to, as a finger does.
            try:
                self.ui.tap_node(self.ui.seek(word, contains=True))
            except UiMissing:
                return False
            self.wait(0.8)
            return True

        for _ in range(20):
            if not page("Back ..."):
                break
        for _ in range(20):
            try:
                self.ui.tap_node(self.ui.seek(name))
                return
            except UiMissing:
                if not page("More ..."):
                    raise
        raise UiMissing("TIMEOUT", f"{name!r} on no page of the song list")

    def reopen(self) -> None:
        mine = self.songs[-1]
        verbs.asong(self.link, f"OPEN {self.song0}")
        self.until(self.song, lambda r: r["project"] == self.song0 and r["open"] == 2,
                   timeout_s=15.0)
        self.ui.tap(self.song0)
        self.pick_song(mine)
        t = time.monotonic()
        self.until(self.song, lambda r: r["project"] == mine and r["open"] == 2, timeout_s=15.0)
        n0 = int(verbs.smpl_engine(self.link)["notes"])
        self.tile(TILE_PLAY_X)
        rows0 = time.monotonic() - self.t0
        self.until(lambda: self.listen.since(rows0) if self.listen else [],
                   lambda rs: any(r["song"] > 0.005 or r["notes"] > n0 for r in rs),
                   timeout_s=BUDGET["reopen"] * 2)
        self.waits["reopen"] = round(time.monotonic() - t, 2)
        busy = [h for h in (self.header_says(k) for k in range(2)) if "FREEZ" in h]
        self.need(11, self.waits["reopen"] <= BUDGET["reopen"] and not busy,
                  "another song and back: sound within its budget, no track rendered again",
                  f"{self.waits['reopen']:.2f}s to the first voice; freezing {busy}")
        self.stop()


class BeatSessionScenario:
    name = "beat_session"
    Params = Params
    description = "a beat made from nothing on the screen: kits, takes, freeze, every wait timed"

    def run(self, ctx: Context, params: Params, progress: Progress) -> Report:
        run = _BeatSteps(ctx, params)
        phases: list[tuple[str, Callable[[], None]]] = [
            ("new song", run.song_start), ("drum track", run.drum_track),
            ("groove", run.groove), ("second kit", run.second_kit), ("part", run.part),
            ("stop and freeze", run.stop_and_freeze), ("play back", run.play_back),
            ("melody", run.melody), ("tempo", run.tempo), ("duplicate", run.duplicate),
            ("reopen", run.reopen)]
        return run_phases(self.name, run, ctx, phases, progress)


def run_phases(name: str, run: _BeatRun, ctx: Context,
               phases: list[tuple[str, Callable[[], None]]], progress: Progress) -> Report:
    """The phases in order with the listener running, then the cleanup and the report: a
    phase that cannot go on (_Stop, a control missing) ends the run where it is."""
    try:
        progress(0, len(phases), "prepare the bench")
        run.setup()
        run.open_arrange()
        with _Listener(run) as listen:
            run.listen = listen
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
            return _bench(name, e.message, run.data)
        run.failures.append(f"{e.code}: {e.message}")
    finally:
        if run.listen is not None:
            path = ctx.workdir / f"{name}_listen.json"
            path.write_text(json.dumps(run.listen.rows))
            run.artifacts.append(Artifact(str(path), "application/json", "listen"))
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
    passed = not run.failures
    waits = ", ".join(f"{k} {v:.2f}s" for k, v in run.waits.items())
    summary = (f"{name}: PASS (waits: {waits})" if passed
               else f"{name}: FAIL — " + "; ".join(run.failures) + f" (waits: {waits})")
    return Report(passed=passed, summary=summary, data=run.data, artifacts=run.artifacts,
                  exit_code=0 if passed else 1)


def _bench(name: str, why: str, data: dict[str, Any]) -> Report:
    data["failures"] = [why]
    return Report(passed=False, summary=f"{name}: BENCH — {why}", data=data,
                  artifacts=[], exit_code=2)


SCENARIO = register(BeatSessionScenario())
