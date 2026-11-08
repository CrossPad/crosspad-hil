"""A setting being turned in the Sampler's pad editor stays turned on the next pad.

1. Sampler → EDIT (Advanced → Edit before 10.2026): the pad editor, on the
   first pad of `pads`.
2. Focus CHOKE and click it into edit mode (ENC_FOCUS edit=1).
3. For every other pad of `pads`: hit it. The editor must show that pad, still
   be in edit mode on CHOKE, and one detent must change the value with no
   click in between; a detent back puts it where it was.

Until 06.10.2026 every pad hit dropped the editor out of edit mode, so
retuning one setting across pads took an extra click per pad.

The kit is loaded again from the card at the end: the turns are put back, but
the pads stay marked modified, and the Sampler would ask to save them.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

from crosspad_hil import audio_state, verbs
from crosspad_hil.cdc import CdcLink
from crosspad_hil.errors import HilError
from crosspad_hil.scenarios.base import Context, Progress, Report, register

PARAM = "CHOKE"
CHOKE_MAX = 7
START_WAIT_S = 5.0
_CHOKE_RE = re.compile(r"^CHOKE\s+(\d+|-)$")
_EDIT_RE = re.compile(r"\bedit\b", re.IGNORECASE)


@dataclass
class Params:
    pads: str = field(default="0,5,10,15", metadata={
        "help": "pads hit in turn (0-based); the first one opens the editor"})
    settle_s: float = field(default=0.5, metadata={
        "help": "after a pad hit or a press: the editor picks hits up on a 50 ms tick"})
    load_timeout_s: float = field(default=30.0, metadata={"help": "for the kit reload at the end"})


def _choke(label: str) -> int | None:
    """CHOKE's value off its label, 0 for '-'; None when the label is not CHOKE's."""
    m = _CHOKE_RE.match(label.strip())
    if not m:
        return None
    return 0 if m.group(1) == "-" else int(m.group(1))


def _is_edit(label: str) -> bool:
    return _EDIT_RE.search(label) is not None


def _labels(link: CdcLink) -> list[str]:
    return [str(e.get("label") or "") for e in verbs.enc_group(link)["group"]]


def _focus(link: CdcLink, match, limit: int) -> bool:
    """Rotate until a label `match` accepts is focused. False if it never comes round."""
    for _ in range(limit):
        if match(str(verbs.enc_focus(link).get("label") or "")):
            return True
        verbs.enc_rotate(link, 1)
    return match(str(verbs.enc_focus(link).get("label") or ""))


class PadEditStickyScenario:
    name = "pad_edit_sticky"
    Params = Params
    description = "pad editor: a setting in edit mode stays in edit mode across pad hits"

    def run(self, ctx: Context, params: Params, progress: Progress) -> Report:
        data: dict[str, Any] = {"hops": []}
        ctx.partial = data
        failures: list[str] = []
        link = None
        kit = -1
        audio_snap: dict[str, Any] | None = None
        try:
            pads = [int(x) for x in params.pads.split(",") if x.strip()]
        except ValueError:
            return self._bench(
                f"--pads must be comma-separated pad numbers, got {params.pads!r}", data)
        if len(pads) < 2 or any(not 0 <= p < 16 for p in pads):
            return self._bench("--pads needs at least two pads, 0..15", data)
        steps = len(pads) + 1
        try:
            link = ctx.open_cdc()
            audio_snap = audio_state.snapshot(link)
            kit = int(verbs.kit_status(link).get("current", -1))
            if kit < 0:
                return self._bench("no kit loaded (bench.py ready)", data)

            progress(0, steps, "Sampler → Advanced → Edit")
            verbs.app_start(link, "Sampler", wait_s=START_WAIT_S)
            ctx.cancelled.wait(params.settle_s)
            # EDIT sits on the panel itself since the 10.2026 layout; before it,
            # inside Advanced.
            path = ("Edit",) if any(_is_edit(lb) for lb in _labels(link)) else ("Advanced", "Edit")
            for label in path:
                match = _is_edit if label == "Edit" else (lambda lb, want=label: lb == want)
                if not _focus(link, match, len(_labels(link)) + 2):
                    return self._bench(f"the Sampler has no '{label}' entry — UI changed?", data)
                verbs.enc_press(link)
                ctx.cancelled.wait(params.settle_s + 0.7)
            if not any(_choke(lb) is not None for lb in _labels(link)):
                return self._bench(
                    "no CHOKE in the encoder group — the pad editor did not open", data)

            self._hit(ctx, link, pads[0], params.settle_s)
            if f"PAD {pads[0] + 1}" not in _labels(link):
                return self._bench(f"the editor did not move to pad {pads[0] + 1}", data)
            if not _focus(link, lambda lb: _choke(lb) is not None, len(_labels(link)) + 2):
                return self._bench("CHOKE never took the focus", data)
            verbs.enc_press(link)
            ctx.cancelled.wait(params.settle_s)
            f = verbs.enc_focus(link)
            if "editing" not in f:
                return self._bench("ENC_FOCUS has no edit= — firmware predates 2026-10-06", data)
            if not f["editing"]:
                return self._bench("a click on CHOKE did not enter edit mode", data)

            for step, pad in enumerate(pads[1:], start=1):
                ctx.check_cancelled()
                progress(step, steps, f"hit pad {pad + 1}, then turn {PARAM}")
                hop = self._hop(ctx, link, pad, params.settle_s)
                data["hops"].append(hop)
                failures += hop["failures"]

            progress(steps, steps, "done")
            return self._report(failures, data)
        finally:
            if link is not None:
                try:
                    if verbs.enc_focus(link).get("editing"):
                        verbs.enc_press(link)
                    verbs.app_stop(link)
                    if kit >= 0:
                        verbs.kit_load(link, kit, wait_s=params.load_timeout_s)
                except HilError as e:
                    ctx.log(f"cleanup: {e.code} {e.message}")
                audio_state.put_back(link, audio_snap, ctx.log, data)
            ctx.close_all()

    @staticmethod
    def _hit(ctx: Context, link: CdcLink, pad: int, settle_s: float) -> None:
        verbs.pad_press(link, pad, 100)
        verbs.pad_release(link, pad)
        ctx.cancelled.wait(settle_s)

    def _hop(self, ctx: Context, link: CdcLink, pad: int, settle_s: float) -> dict[str, Any]:
        """Hit `pad`, then one detent there and one back. The hop's own failures in it."""
        hop: dict[str, Any] = {"pad": pad + 1, "failures": []}
        fail = hop["failures"].append
        self._hit(ctx, link, pad, settle_s)
        hop["moved"] = f"PAD {pad + 1}" in _labels(link)
        if not hop["moved"]:
            fail(f"pad {pad + 1}: the editor did not move to it")
            return hop
        f = verbs.enc_focus(link)
        hop["editing"] = bool(f.get("editing"))
        hop["label"] = f["label"]
        before = _choke(f["label"])
        if before is None or not hop["editing"]:
            fail(f"pad {pad + 1}: edit mode dropped by the hit "
                 f"(focus '{f['label']}', edit={int(hop['editing'])})")
            if hop["editing"]:
                verbs.enc_press(link)
            if not _focus(link, lambda lb: _choke(lb) is not None, len(_labels(link)) + 2):
                return hop
            verbs.enc_press(link)             # back in edit mode, so the next hop tests something
            ctx.cancelled.wait(settle_s)
            return hop
        d = -1 if before >= CHOKE_MAX else 1
        verbs.enc_rotate(link, d)
        ctx.cancelled.wait(settle_s)
        turned = _choke(verbs.enc_focus(link)["label"])
        verbs.enc_rotate(link, -d)
        ctx.cancelled.wait(settle_s)
        back = _choke(verbs.enc_focus(link)["label"])
        hop.update(before=before, turned=turned, back=back)
        if turned != before + d:
            fail(f"pad {pad + 1}: a detent after the hit gave CHOKE {turned} (want {before + d})")
        if back != before:
            fail(f"pad {pad + 1}: a detent back gave CHOKE {back} (want {before})")
        return hop

    @staticmethod
    def _bench(why: str, data: dict[str, Any]) -> Report:
        data["failures"] = [why]
        return Report(passed=False, summary=f"pad_edit_sticky: BENCH — {why}", data=data,
                      artifacts=[], exit_code=2)

    @staticmethod
    def _report(failures: list[str], data: dict[str, Any]) -> Report:
        passed = not failures
        data["failures"] = failures
        summary = ("pad_edit_sticky: PASS" if passed
                   else "pad_edit_sticky: FAIL — " + "; ".join(failures))
        return Report(passed=passed, summary=summary, data=data, artifacts=[],
                      exit_code=0 if passed else 1)


SCENARIO = register(PadEditStickyScenario())
