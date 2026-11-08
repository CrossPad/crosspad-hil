"""The beat grown into a song: its loop an element, the element twice, an edit inside it
heard in both, a tempo turned -- on the screen, every wait timed.

beat_session's first five steps make the beat (a drum groove and a part on a second
kit, four bars); then:

  6  STOP: the groove's track freezes (ARM is on the part)
  7  EDIT -> a loop region over the beat -> Make element from region: one element block
     plays the beat, and it is heard
  8  the block selected, DUP: the element again right after it, the song eight bars,
     both blocks heard from bar one
  9  CLIP EDIT on the second block: inside, DUB a few hits onto the part, a power click
     out: back without a wait, and both blocks bake again by themselves
 10  the top bar's BPM turned: the blocks play on (varispeed), then bake again at the
     new tempo by themselves
 11  PLAY: the whole song, no ring underrun

Elements of audio recordings are song_elements' run; these are of kit tracks, whose
freezes the bake is made from. The speakers stay at `OUTVOL 2 <outvol>`; the song,
the element and its bakes go off the card at the end.
"""
from __future__ import annotations

import time
from collections.abc import Callable
from typing import Any

from crosspad_hil import verbs
from crosspad_hil.scenarios import beat_session as bs
from crosspad_hil.scenarios.base import Context, Progress, Report, register
from crosspad_hil.scenarios.song_elements import (
    ELEMENT_TITLE,
    PLAYING,
    TILE_PLAY_X,
    TIMELINE_Y,
    _Stop,
    says_busy,
)
from crosspad_hil.ui import UiMissing

NOT_FRESH = ("old", "no bake", "not baked")   # what an element block says until baked
BUSY_SHOWN = ("COPYING", "KEEPING", "BAKING", "FREEZING", "LINKING", "OPENING")
REWIND_TILE_X = 44 + 19     # the top bar's STOP (STOP n.n): stopped already, back to bar one
DUB_TILE_X = 164 + 19

BUDGET = dict(bs.BUDGET)
BUDGET.update({
    "element": 20.0,          # Make element from region to the block baked
    "enter": 8.0,             # CLIP EDIT to the element open
    "leave": 10.0,            # the click to the song open again: an edited kit track frozen, baked
    "rebake": 20.0,           # the edit left, to every block of it baked again
    "tempo rebake": 25.0,     # the BPM turned, to every block baked at the new tempo
})


class _SongRun(bs._BeatSteps):
    def element_blocks(self) -> list[Any]:
        """The element blocks on the timeline, left to right."""
        out = [n for n in self.ui.screen().nodes if n.interactive and n.cls == "obj"
               and n.y >= TIMELINE_Y and n.text.startswith(ELEMENT_TITLE)]
        return sorted(out, key=lambda n: n.x)

    def blocks_fresh(self) -> bool:
        texts = [n.text for n in self.element_blocks()]
        return bool(texts) and not any(w in t for t in texts for w in NOT_FRESH)

    def timed_until(self, name: str, done: Callable[[], bool], limit_s: float) -> bool:
        t = time.monotonic()
        ok = bool(self.until(done, bool, timeout_s=limit_s))
        self.waits[name] = round(time.monotonic() - t, 2)
        if not ok:
            self.failures.append(f"{name}: not done after {limit_s:.0f}s")
        elif self.waits[name] > BUDGET[name]:
            self.failures.append(f"{name}: {self.waits[name]:.2f}s, over its "
                                 f"{BUDGET[name]:.1f}s budget")
        return ok

    def heard_passes(self, t_from: float, bar: int, bars: int) -> list[dict[str, Any]]:
        return self.passes(self.listen.since(t_from) if self.listen else [], bar, bars)

    # ----------------------------------------------------------------- the steps

    def freeze_beat(self) -> None:
        self.stop()
        ok = self.until(lambda: self.frozen([0]), bool, timeout_s=self.p.job_timeout_s)
        self.need(6, ok, "STOP: the groove's track freezes",
                  f"track 1 says {self.header_says(0)!r}")

    def chorus(self) -> None:
        self.beat_bars = int(self.song()["len"])
        rgn = self.set_region(0, self.beat_bars)
        self.make_element()
        self.timed_until("element", self.blocks_fresh, BUDGET["element"] * 2)
        blocks = self.element_blocks()
        self.need(7, len(blocks) == 1,
                  f"a loop region over the beat ({rgn}), Make element from region: one element "
                  "block, baked", f"blocks {[b.text for b in blocks]}")
        self.edit_menu()
        self.menu_item("Clear the loop region")
        under0 = int(self.song().get("under", 0))
        self.tile(TILE_PLAY_X)
        t_play = time.monotonic() - self.t0
        self.wait((2 * self.beat_bars + 1) * bs.BAR_S)
        heard = self.heard_passes(t_play, 0, self.beat_bars)
        self.stop()
        self.need(7, bool(heard) and all(p["song"] > 0.005 for p in heard)
                  and int(self.song().get("under", 0)) == under0,
                  "PLAY: the element block heard on every pass, no underrun", f"passes {heard}")

    def again(self) -> None:
        """The block selected, DUP: the chorus twice, as a player repeats a block."""
        try:
            self.tap_block(self.element_blocks()[0])
            self.ui.tap("DUP")
        except (UiMissing, IndexError) as e:
            self.failures.append(f"step 8: no element block to DUP: {e}")
            raise _Stop("no DUP") from None
        self.wait_job()
        song = self.until(self.song, lambda r: int(r["len"]) >= 2 * self.beat_bars, timeout_s=10.0)
        blocks = self.element_blocks()
        self.need(8, int(song["len"]) == 2 * self.beat_bars and len(blocks) == 2,
                  "the element block selected, DUP: the element again, right after it",
                  f"len {song['len']}, blocks {[(b.x, b.text) for b in blocks]}")
        self.tile(REWIND_TILE_X)                # STOP while stopped: back to bar one
        under0 = int(self.song().get("under", 0))
        self.tile(TILE_PLAY_X)
        t_play = time.monotonic() - self.t0
        self.wait((2 * 2 * self.beat_bars + 1) * bs.BAR_S)
        whole = self.heard_passes(t_play, 0, 2 * self.beat_bars)
        rows = self.listen.since(t_play) if self.listen else []
        silent = [r["pos"] for r in rows if r["play"] == PLAYING and r["song"] <= 0.0005]
        self.stop()
        self.need(8, bool(whole) and int(self.song().get("under", 0)) == under0,
                  "PLAY from bar one: both blocks heard, no underrun",
                  f"passes {whole}; positions with a silent Song channel {silent[:12]}")

    def edit_inside(self) -> None:
        blocks = self.element_blocks()
        if len(blocks) < 2:
            raise _Stop("no second block")
        self.tap_block(blocks[1])
        t = time.monotonic()
        self.ui.tap("CLIP EDIT")
        inside = self.until(self.song, lambda r: str(r.get("project", "")).startswith(".elements/"),
                            timeout_s=BUDGET["enter"] * 3)
        self.waits["enter"] = round(time.monotonic() - t, 2)
        # What the screen says until the element is ready to play into: each change, timed.
        shown: list[tuple[float, str]] = []

        def busy_now() -> bool:
            words = [w for w in self.labels() if any(b in w for b in BUSY_SHOWN)]
            text = " | ".join(words)
            if not shown or shown[-1][1] != text:
                shown.append((round(time.monotonic() - t, 2), text))
            return bool(words)

        self.until(lambda: None, lambda _: not busy_now(), timeout_s=self.p.job_timeout_s)
        self.waits["enter ready"] = round(time.monotonic() - t, 2)
        self.data["enter_shown"] = shown
        self.need(9, str(inside.get("project", "")).startswith(".elements/")
                  and self.waits["enter"] <= BUDGET["enter"],
                  "CLIP EDIT on the second block: inside the element",
                  f"project {inside.get('project')} after {self.waits['enter']:.2f}s")
        seq0 = sum(int(b["notes"]) for b in verbs.asong_seq(self.link, 1)["blocks"])
        self.tile(TILE_PLAY_X)
        self.until(self.song, lambda r: r.get("play") == PLAYING, timeout_s=5.0)
        self.to_bar_one()
        self.tile(DUB_TILE_X)
        dubbed = self.hit_bars([(0.5, 9, 100), (2.5, 9, 100)], 2)
        self.tile(DUB_TILE_X)
        self.stop()
        seq1 = sum(int(b["notes"]) for b in verbs.asong_seq(self.link, 1)["blocks"])
        self.need(9, seq1 >= seq0 + dubbed, "DUB inside the element: the hits on the part",
                  f"notes {seq0} -> {seq1} ({dubbed} hit)")
        song, took = self.leave_element()
        self.waits["leave"] = took
        self.need(9, not song.startswith(".elements/") and took <= BUDGET["leave"],
                  "a power click out: the song again (an edited kit track is frozen and "
                  "the element baked on the way)", f"project {song} after {took:.2f}s")
        self.timed_until("rebake", self.blocks_fresh, BUDGET["rebake"] * 2)
        self.need(9, self.blocks_fresh(), "both blocks of the element baked again by themselves",
                  f"blocks {[b.text for b in self.element_blocks()]}")

    def leave_element(self) -> tuple[str, float]:
        """Power clicks until the song is open again, timed from the click that started
        the leave: a click the top bar's focus takes back first (a tile was tapped) is the
        player's too, but no wait of the element's."""
        level = self.project()
        for _ in range(4):
            t = time.monotonic()
            self.power()
            moved = self.until(lambda: self.project() != level or says_busy(self.labels()),
                               bool, timeout_s=1.5)
            if moved:
                break
        song = self.until(self.project, lambda p: p != level, timeout_s=self.p.job_timeout_s)
        took = round(time.monotonic() - t, 2)
        self.wait_job()
        return song, took

    def tempo_elements(self) -> None:
        bpm0 = float(self.song()["bpm"])
        sc = self.ui.screen()
        tile = next(n for n in sc.controls() if sc.words(n)[:1] == ["BPM"] and n.y < 40)
        self.ui.tap_node(tile)
        verbs.enc_rotate(self.link, self.p.bpm_turn)
        self.wait(0.4)
        verbs.enc_press(self.link)
        self.until(self.song, lambda r: float(r["bpm"]) != bpm0, timeout_s=5.0)
        # The blocks go old first (the bake is of the old tempo); then baked again.
        went_old = self.until(lambda: not self.blocks_fresh(), bool, timeout_s=5.0)
        self.data["tempo_went_old"] = bool(went_old)
        self.timed_until("tempo rebake", self.blocks_fresh, BUDGET["tempo rebake"] * 2)
        blocks = [b.text for b in self.element_blocks()]
        self.need(10, self.blocks_fresh(),
                  "the BPM turned: every block baked again at the new tempo",
                  f"bpm {bpm0} -> {self.song()['bpm']}; blocks {blocks}")

    def whole_song(self) -> None:
        under0 = int(self.song().get("under", 0))
        self.tile(TILE_PLAY_X)
        t_play = time.monotonic() - self.t0
        bars = int(self.song()["len"])
        bpm = float(self.song()["bpm"])
        self.wait((bars + 1) * 4 * 60.0 / bpm)
        rows = self.listen.since(t_play) if self.listen else []
        loud = [r for r in rows if r["play"] == PLAYING and r["song"] > 0.005]
        self.stop()
        self.need(11, bool(loud) and int(self.song().get("under", 0)) == under0,
                  "PLAY: the whole song heard, no underrun",
                  f"{len(loud)} of {len(rows)} looks with the Song channel up")


class BeatToSongScenario:
    name = "beat_to_song"
    Params = bs.Params
    description = "the beat grown into a song of elements: twice, edited inside, tempo turned"

    def run(self, ctx: Context, params: bs.Params, progress: Progress) -> Report:
        run = _SongRun(ctx, params)
        phases: list[tuple[str, Callable[[], None]]] = [
            ("new song", run.song_start), ("drum track", run.drum_track),
            ("groove", run.groove), ("second kit", run.second_kit), ("part", run.part),
            ("freeze", run.freeze_beat), ("chorus", run.chorus), ("again", run.again),
            ("edit inside", run.edit_inside), ("tempo", run.tempo_elements),
            ("whole song", run.whole_song)]
        return bs.run_phases("beat_to_song", run, ctx, phases, progress)


SCENARIO = register(BeatToSongScenario())
