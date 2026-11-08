"""Record a scene, bounce it to the audio song, play the song back, and restart it hard.

1. ASONG OPEN `project`; poll until open=2 (ready). open=3 is a song the board could
   not read -- a failure, never "empty". The scenes are the open song's: SEQ's
   file= turns to the project's once they are in.
2. SEQ: a scene of its own in the project (SEQ ADD), count-in off, one bar at
   120 BPM, REC, `pads` hits, PLAY. Recorded before the OPEN it was the song open
   at the time's, and gone from the project once the pattern stopped.
3. ASONG BOUNCE 0 0; poll until bounce=2 (done) with one more clip on track 1.
   bounce=3 fails with the board's reason.
4. ASONG PLAY 0; play=1 within `open_timeout_s`; after `play_s`, under=0 and
   openfail=0 (the card kept up, every clip opened).
5. `rounds` of STOP then PLAY with no wait between -- the window in which the old
   engine freed a ring the render was still reading. RESET_LOG must not move and
   the board must still answer.
6. PLAY, STOP and BOUNCE back to back: the bounce must still finish. A stop or play
   landing beside it once overwrote the queued bounce and left the engine busy
   until reboot.

Every ASONG verb only asks; the scenario polls the bare verb for the answer. The
scene is deleted and count-in put back whatever happened.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

from crosspad_hil import verbs
from crosspad_hil.scenarios.base import Context, Progress, Report, register

MAX_SCENES = 8
OPEN_READY = 2
OPEN_FAILED = 3
BOUNCE_DONE = 2
BOUNCE_FAILED = 3


@dataclass
class Params:
    project: str = field(default="hil_song_bounce",
                         metadata={"help": "song project to open (created if missing)"})
    pads: int = field(default=4, metadata={"help": "pads hit into the one-bar scene"})
    rounds: int = field(default=200, metadata={"help": "STOP/PLAY pairs with no wait between"})
    play_s: float = field(default=4.0,
                          metadata={"help": "how long the song plays before underruns are read"})
    bar_s: float = field(default=2.0,
                         metadata={"help": "one bar at 120 BPM, waited out after the hits"})
    poll_s: float = field(default=0.2, metadata={"help": "between polls of ASONG"})
    open_timeout_s: float = field(default=15.0, metadata={"help": "for OPEN and for PLAY to start"})
    bounce_timeout_s: float = field(default=30.0, metadata={"help": "for the bounce to finish"})


def _poll(ctx: Context, link: Any, params: Params, timeout_s: float, done) -> dict[str, Any]:
    """ASONG until `done(reply)` or the timeout; the last reply either way."""
    deadline = time.monotonic() + timeout_s
    st = verbs.asong(link)
    while not done(st) and time.monotonic() < deadline and not ctx.cancelled.is_set():
        ctx.cancelled.wait(params.poll_s)
        st = verbs.asong(link)
    return st


class SongBounceScenario:
    name = "song_bounce"
    Params = Params
    description = "record a scene, bounce it, play the song, then STOP/PLAY it hard"

    def run(self, ctx: Context, params: Params, progress: Progress) -> Report:
        data: dict[str, Any] = {}
        ctx.partial = data
        failures: list[str] = []
        steps = 6
        link = None
        scene = None
        countin = 1
        try:
            link = ctx.open_cdc()
            resets0 = verbs.reset_count(link)

            progress(0, steps, f"open {params.project}")
            countin = verbs.seq(link, "STOP")["countin"]   # nothing plays: the scenes can switch
            t0 = time.monotonic()
            verbs.asong(link, f"OPEN {params.project}")
            st = _poll(ctx, link, params, params.open_timeout_s,
                       lambda r: r["open"] in (OPEN_READY, OPEN_FAILED))
            data["open"] = {"state": st["open"], "s": round(time.monotonic() - t0, 2)}
            if st["open"] != OPEN_READY:
                failures.append(f"open={st['open']} for {params.project} (want {OPEN_READY})")
                return self._report(failures, data)
            mine = f"/{params.project}/"
            deadline = time.monotonic() + params.open_timeout_s
            sq = verbs.seq(link)
            while mine not in sq["file"] and time.monotonic() < deadline \
                    and not ctx.cancelled.is_set():
                ctx.cancelled.wait(params.poll_s)
                sq = verbs.seq(link)
            if mine not in sq["file"]:
                failures.append(f"the scenes stayed {sq['file']} after opening {params.project}")
                return self._report(failures, data)

            progress(1, steps, "record a one-bar scene of its own")
            before = sq["scenes"]
            if before >= MAX_SCENES:
                data["failures"] = [f"all {MAX_SCENES} scenes are in use; free one"]
                return Report(passed=False, summary=f"song_bounce: BENCH — {data['failures'][0]}",
                              data=data, artifacts=[], exit_code=2)
            verbs.seq(link, "ADD")
            scene = before
            for action in (f"SCENE {scene}", "COUNTIN 0", f"BARS {scene} 1", "BPM 120", "REC"):
                verbs.seq(link, action)
            for pad in range(params.pads):
                verbs.pad_press(link, pad, 100)
                verbs.pad_release(link, pad)
            ctx.cancelled.wait(params.bar_s)
            rec = verbs.seq(link, "PLAY")
            data["scene"] = scene
            data["events"] = rec["ev"][scene] if scene < len(rec["ev"]) else 0
            if not data["events"]:
                failures.append(f"SEQ recorded nothing into scene {scene + 1}")

            progress(2, steps, f"bounce scene {scene + 1} to track 1")
            clips0 = verbs.asong(link)["clips"][0]
            verbs.asong(link, f"BOUNCE {scene} 0")
            st = _poll(ctx, link, params, params.bounce_timeout_s,
                       lambda r: r["bounce"] in (BOUNCE_DONE, BOUNCE_FAILED))
            data["bounce"] = {"state": st["bounce"], "file": st["file"], "err": st["err"]}
            if st["bounce"] != BOUNCE_DONE:
                failures.append(f"bounce={st['bounce']}: {st['err'] or 'no reason given'}")
                return self._report(failures, data)
            if st["clips"][0] != clips0 + 1:
                failures.append(f"track 1 has {st['clips'][0]} clips after the bounce "
                                f"(want {clips0 + 1})")

            progress(3, steps, "play the song")
            verbs.asong(link, "PLAY 0")
            st = _poll(ctx, link, params, params.open_timeout_s, lambda r: r["play"] == 1)
            if st["play"] != 1:
                failures.append("song never started playing")
                return self._report(failures, data)
            ctx.cancelled.wait(params.play_s)
            st = verbs.asong(link)
            data["play"] = {"pos": st["pos"], "under": st["under"], "openfail": st["openfail"]}
            if st["under"] or st["openfail"]:
                failures.append(f"while playing: under={st['under']} openfail={st['openfail']}")

            progress(4, steps, f"{params.rounds} x STOP/PLAY")
            for _ in range(params.rounds):
                if ctx.cancelled.is_set():
                    break
                verbs.asong(link, "STOP")
                verbs.asong(link, "PLAY 0")
            verbs.asong(link, "STOP")
            resets1 = verbs.reset_count(link)
            data["resets"] = resets1 - resets0
            if resets1 != resets0:
                failures.append(f"{resets1 - resets0} reset(s) during STOP/PLAY")

            progress(5, steps, "PLAY, STOP, BOUNCE back to back")
            clips0 = verbs.asong(link)["clips"][0]
            verbs.asong(link, "PLAY 0")
            verbs.asong(link, "STOP")
            verbs.asong(link, f"BOUNCE {scene} 0")
            st = _poll(ctx, link, params, params.bounce_timeout_s,
                       lambda r: r["bounce"] in (BOUNCE_DONE, BOUNCE_FAILED))
            data["bounce_after_stop"] = {"state": st["bounce"], "err": st["err"]}
            if st["bounce"] != BOUNCE_DONE or st["clips"][0] != clips0 + 1:
                failures.append(f"bounce right after STOP: bounce={st['bounce']} "
                                f"clips {clips0}->{st['clips'][0]} {st['err'] or ''}".rstrip())

            progress(steps, steps, "done")
            return self._report(failures, data)
        finally:
            if link is not None and scene is not None:
                verbs.seq(link, "STOP")
                verbs.seq(link, f"DEL {scene}")
                verbs.seq(link, f"COUNTIN {countin}")
            ctx.close_all()

    @staticmethod
    def _report(failures: list[str], data: dict[str, Any]) -> Report:
        passed = not failures
        data["failures"] = failures
        summary = ("song_bounce: PASS" if passed else "song_bounce: FAIL — " + "; ".join(failures))
        return Report(passed=passed, summary=summary, data=data, artifacts=[],
                      exit_code=0 if passed else 1)


SCENARIO = register(SongBounceScenario())
