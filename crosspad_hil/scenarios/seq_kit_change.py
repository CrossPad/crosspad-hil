"""A take survives picking another kit: the sequencer's scenes are not the kit's.

1. SEQ ADD a scene of its own (the player's scenes are never touched), one bar,
   count-in off, REC `pads` hits, PLAY.
2. Load another kit while the take plays. The scene must still hold every hit,
   and the scenes must be stored outside the kits folder.
3. Load the first kit back: still every hit.

The scene is deleted again whatever happened, so a run leaves the player's
scenes and count-in as it found them. Until 26.09.2026 a kit change saved the scenes into
the outgoing kit's folder and loaded the incoming kit's -- a player who picked
a kit mid-groove saw the groove gone.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from crosspad_hil import verbs
from crosspad_hil.scenarios.base import Context, Progress, Report, register

MAX_SCENES = 8


@dataclass
class Params:
    pads: int = field(default=3, metadata={"help": "pads hit into the take"})
    bar_s: float = field(default=2.0,
                         metadata={"help": "one bar at 120 BPM, waited out after the hits"})
    load_timeout_s: float = field(default=30.0, metadata={"help": "for each kit load"})
    settle_s: float = field(
        default=1.5,
        metadata={"help": "after a load: the scene store acts on its own 0.5 s tick"})


class SeqKitChangeScenario:
    name = "seq_kit_change"
    Params = Params
    description = "record a take, load another kit and back; the take must stay"

    def run(self, ctx: Context, params: Params, progress: Progress) -> Report:
        data: dict[str, Any] = {}
        ctx.partial = data
        failures: list[str] = []
        steps = 3
        scene = None
        link = None
        countin = 1
        try:
            link = ctx.open_cdc()
            kits = verbs.kit_list(link)["kits"]
            home = verbs.kit_status(link)["current"]
            other = next((k["id"] for k in kits if k["id"] != home), None)
            if home < 0 or other is None:
                return self._bench("needs a kit loaded and a second one on the card", data)

            progress(0, steps, "record a take into a scene of its own")
            st = verbs.seq(link, "STOP")
            before, countin = st["scenes"], st["countin"]
            if before >= MAX_SCENES:
                return self._bench(f"all {MAX_SCENES} scenes are in use; free one", data)
            if verbs.seq(link, "ADD")["scenes"] != before + 1:
                return self._bench("SEQ ADD made no scene", data)
            scene = before
            for action in (f"SCENE {scene}", "COUNTIN 0", f"BARS {scene} 1", "BPM 120", "REC"):
                verbs.seq(link, action)
            for pad in range(params.pads):
                verbs.pad_press(link, pad, 100)
                verbs.pad_release(link, pad)
            ctx.cancelled.wait(params.bar_s)
            st = verbs.seq(link, "PLAY")
            taken = st["ev"][scene]
            data.update(scene=scene, events=taken, file=st["file"])
            if not taken:
                failures.append(f"SEQ recorded nothing into scene {scene + 1}")
                return self._report(failures, data)
            if "/kits/" in st["file"].lower():
                failures.append(f"scenes are stored in a kit's folder: {st['file']}")

            for step, kit in ((1, other), (2, home)):
                progress(step, steps, f"load kit {kit}")
                verbs.kit_load(link, kit, wait_s=params.load_timeout_s)
                ctx.cancelled.wait(params.settle_s)
                st = verbs.seq(link)
                data[f"after_kit_{kit}"] = {"scenes": st["scenes"], "events": st["ev"][scene]}
                if st["scenes"] != before + 1 or st["ev"][scene] != taken:
                    failures.append(f"after kit {kit}: scenes={st['scenes']} (want {before + 1}), "
                                    f"scene {scene + 1} has {st['ev'][scene]} hits (want {taken})")

            progress(steps, steps, "done")
            return self._report(failures, data)
        finally:
            if link is not None and scene is not None:
                verbs.seq(link, "STOP")
                verbs.seq(link, f"DEL {scene}")
                verbs.seq(link, f"COUNTIN {countin}")
            ctx.close_all()

    @staticmethod
    def _bench(why: str, data: dict[str, Any]) -> Report:
        data["failures"] = [why]
        return Report(passed=False, summary=f"seq_kit_change: BENCH — {why}", data=data,
                      artifacts=[], exit_code=2)

    @staticmethod
    def _report(failures: list[str], data: dict[str, Any]) -> Report:
        passed = not failures
        data["failures"] = failures
        summary = ("seq_kit_change: PASS" if passed
                   else "seq_kit_change: FAIL — " + "; ".join(failures))
        return Report(passed=passed, summary=summary, data=data, artifacts=[],
                      exit_code=0 if passed else 1)


SCENARIO = register(SeqKitChangeScenario())
