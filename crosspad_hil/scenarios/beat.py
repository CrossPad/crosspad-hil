"""Make a beat: the core user workflow, from a prepared loop to a bounced song.

The steps, what each one checks and at which depth it runs are data:
``crosspad_hil/workflows/beat.yaml`` (the Mucha performer plays the same file
on the PC simulator). This module is the board's driver: one method per step
``op``, CDC verbs to act, CDC verbs and the bounced WAV to judge.

No microphone anywhere: the run starts from a loop it writes itself
(``workflows/beat_fixture``) -- eight hits, one per eighth note, each at its
own pitch -- so the Slicer's cuts, the kit, the sequenced pattern and the
bounce can all be checked against known numbers.

The run works in a song of its own (``HIL_BEAT_<TAG>``): the sequencer's
scenes are the open song's, so the user's are never opened for writing. It
makes a kit of its own (``BEAT_<TAG>``) and deletes it, and puts back the
song, the kit, the app, the mixer desk and the theme it found. A board that
is being played (pad presses without the run's stimulus) is a bench problem
(exit 2), not a run.

Depths: ``quick`` (default, ~3 min) slices, loads, records, bounces, checks
the theme; ``medium`` adds the pad editor, overdub, x2, the mixer and the
song edits; ``full`` adds erase, a second scene and a song of scenes, a take
onto the song, an import, a Live-set export.
"""

from __future__ import annotations

import json
import re
import struct
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

from crosspad_hil import fs, verbs, workflows
from crosspad_hil.cdc import CdcLink
from crosspad_hil.errors import ENV, HilError
from crosspad_hil.scenarios.base import Artifact, Context, Progress, Report, register
from crosspad_hil.workflows import beat_fixture

OPEN_READY, OPEN_FAILED = 2, 3
BOUNCE_DONE, BOUNCE_FAILED = 2, 3
MAX_SCENES = 8
SR = 44100
PRESS_HOLD_S = 0.06
# Hits leave 20 ms early: the line still crosses the CDC loop, which the
# Sequencer's UI keeps busy; 1/16 quantisation takes +-75 ms back.
LEAD_S = -0.02
IDLE_WATCH_S = 2.0
SCREENSHOT = "/sdcard/crosspad/screen.bmp"
BACKUP = "/sdcard/crosspad/hil_beat_settings_{tag}.json"

_PADINFO_KIT = re.compile(r"\bkit=(\S+/kit\.json)")


def kit_json_from_padinfo(raw: str) -> str | None:
    """The card path of the kit a PADINFO line names (the Slicer's KIT writes to
    /crosspad/slicer, not /crosspad/kits), or None when the line names none."""
    m = _PADINFO_KIT.search(raw)
    if not m:
        return None
    path = m.group(1)
    return path if path.startswith("/sdcard/") else "/sdcard" + path


# The launcher's tiles carry short names (gui.cpp's app labels), not the app ids.
LAUNCHER_TILES = {
    "Sequencer": "Seq",
    "Settings": "Setup",
    "Recorder": "Record",
    "PadMixer": "Pad Mix",
    "KitSelector": "Kits",
    "DawControl": "DAW",
}
DISPLAY_ICON = "\uf06e"  # LV_SYMBOL_EYE_OPEN: Settings -> Display (settings_ui.cpp)
THEMES = ("Neon Green", "Yellow", "Cyan", "Purple", "Orange", "Pink", "Blue", "Red")
# A fall in internal RAM is a leak; PSRAM moves with what the first song, kit and
# bounce of a boot allocate for good (rings, caches), so it is reported, and
# fails only under the floor the UI needs.
MEM_TOLERANCE = {"int_free": 12 * 1024}
PSRAM_FLOOR = 64 * 1024


@dataclass
class Params:
    depth: str = field(
        default="quick", metadata={"help": "quick (default, a few minutes) | medium | full"}
    )
    workflow: str = field(default="beat", metadata={"help": "workflow name or path to its YAML"})
    tag: str = field(
        default="", metadata={"help": "the run's own name in files/kit/song (default: the time)"}
    )
    keep: bool = field(
        default=False,
        metadata={"help": "leave the song, kit and fixture on the card (for a look afterwards)"},
    )
    poll_s: float = field(default=0.2, metadata={"help": "between polls"})
    timeout_s: float = field(default=30.0, metadata={"help": "for opens, exports, loads, bounces"})


class StepFailed(Exception):
    """A check of the current step failed; the run goes on to cleanup."""


class BeatRun:
    def __init__(self, ctx: Context, params: Params) -> None:
        self.ctx = ctx
        self.p = params
        self.wf = workflows.load(params.workflow)
        self.tag = params.tag or datetime.now().strftime("%m%d%H%M")
        self.names = self.wf.names(self.tag)
        self.link: CdcLink | None = None
        self.data: dict[str, Any] = {
            "depth": params.depth,
            "tag": self.tag,
            "names": self.names,
            "steps": {},
            "left": {},
        }
        self.failures: list[str] = []
        self.home: dict[str, Any] = {}
        self.scene: int | None = None
        self.kit_id: int | None = None
        self.kit_json: str | None = None
        self.loop: beat_fixture.Loop | None = None
        self.semis: dict[int, int] = {}
        self.reversed: list[int] = []
        self.artifacts: list[Artifact] = []
        self.opened_project = False
        self.made_kit = False
        self.pushed: list[str] = []
        self.warnings: list[str] = []
        self.rebooted = False

    # ── plumbing ────────────────────────────────────────────────────────
    @property
    def L(self) -> CdcLink:
        assert self.link is not None
        return self.link

    def log(self, msg: str) -> None:
        self.ctx.log(f"beat: {msg}")

    def wait(self, s: float) -> None:
        self.ctx.cancelled.wait(s)

    def check(self, ok: bool, msg: str) -> None:
        if not ok:
            raise StepFailed(msg)

    def poll(
        self, fn: Callable[[], dict], done: Callable[[dict], bool], timeout_s: float | None = None
    ) -> dict:
        deadline = time.monotonic() + (timeout_s or self.p.timeout_s)
        st = fn()
        while not done(st) and time.monotonic() < deadline and not self.ctx.cancelled.is_set():
            self.wait(self.p.poll_s)
            st = fn()
        return st

    def raw(self, cmd: str, expect: str, timeout_s: float | None = None) -> str:
        return (self.L.transact(cmd, expect, timeout_s=timeout_s).line or "").strip()

    def with_raw_port(self, fn: Callable[[], Any]) -> Any:
        """fs.push/pull need the port unframed: close the link, run, reopen."""
        self.L.close()
        try:
            return fn()
        finally:
            self.link = self.ctx.open_cdc()

    def pull(self, remote: str, local: Path) -> Path:
        self.with_raw_port(
            lambda: fs.pull(self.ctx.device, remote, local, serial_factory=self.ctx.serial_factory)
        )
        return local

    def push(self, local: Path, remote: str) -> None:
        self.with_raw_port(
            lambda: fs.push(self.ctx.device, local, remote, serial_factory=self.ctx.serial_factory)
        )

    def delete(self, remote: str) -> bool:
        try:
            verbs.fs_delete(self.L, remote)
            return True
        except HilError:
            return False

    def kit_by_name(self, name: str) -> int | None:
        for k in verbs.kit_list(self.L)["kits"]:
            if k["name"] == name:
                return k["id"]
        return None

    def press(self, pad: int, vel: int) -> None:
        verbs.pad_press(self.L, pad, vel)

    def release(self, pad: int) -> None:
        verbs.pad_release(self.L, pad)

    def play_hits(self, hits: list[workflows.Hit], t0: float) -> None:
        """Press each hit at t0 + its grid time (+LEAD_S), release PRESS_HOLD_S later."""
        events = []
        for h in hits:
            t = t0 + LEAD_S + h.step * self.wf.step_s
            events.append((t, 0, h.pad, h.vel))
            events.append((t + PRESS_HOLD_S, 1, h.pad, 0))
        events.sort()
        for t, kind, pad, vel in events:
            now = time.monotonic()
            if t > now:
                self.wait(t - now)
            # Sent, not transacted: under the Sequencer's UI a reply takes up to
            # ~100 ms, and waiting for each one piled the later hits up late.
            self.L.send(f"PAD_PRESS {pad} {vel}" if kind == 0 else f"PAD_RELEASE {pad}")

    # ── the encoder: real UI paths ──────────────────────────────────────
    def labels(self) -> list[str]:
        return [str(e.get("label") or "") for e in verbs.enc_group(self.L)["group"]]

    def focus_on(self, match: Callable[[str], bool], limit: int = 48) -> str:
        """Turn the encoder until a label `match` accepts is focused; returns that label."""
        for _ in range(limit + 1):
            label = verbs.enc_focus(self.L)["label"]
            if match(label):
                return label
            verbs.enc_rotate(self.L, 1)
            self.wait(0.05)
        raise StepFailed(
            f"encoder: nothing matching came round in {limit} detents (group: {self.labels()[:24]})"
        )

    def click_on(self, match: Callable[[str], bool], settle_s: float = 0.4) -> str:
        label = self.focus_on(match)
        verbs.enc_press(self.L)
        self.wait(settle_s)
        return label

    def launch(self, app: str) -> str:
        """The app through the launcher, as a player opens it; APP_START if the
        launcher does not show it to the encoder."""
        try:
            verbs.app_stop(self.L)
        except HilError:
            pass
        self.wait(0.8)
        try:
            tile = LAUNCHER_TILES.get(app, app).lower()
            self.click_on(lambda s: s.strip().lower() == tile, settle_s=0.2)
            st = self.poll(
                lambda: verbs.app_list(self.L),
                lambda r: (r.get("running") or "").lower() == app.lower(),
                5.0,
            )
            if (st.get("running") or "").lower() == app.lower():
                return "launcher"
        except (StepFailed, HilError):
            pass
        verbs.app_start(self.L, app, wait_s=5.0)
        return "APP_START"

    # ── steps ───────────────────────────────────────────────────────────
    def op_preflight(self, step: workflows.Step) -> dict:
        L = self.L
        a = verbs.pad_stats(L)
        self.wait(IDLE_WATCH_S)
        b = verbs.pad_stats(L)
        if b["press"] != a["press"]:
            raise HilError(
                ENV,
                f"someone is playing the board ({b['press'] - a['press']} "
                f"presses in {IDLE_WATCH_S:.0f} s)",
                hint="wait until the pads are quiet, then run again",
            )
        seq = verbs.seq(L)
        if seq.get("state", 0) not in (0,):
            raise HilError(
                ENV,
                f"the sequencer is running (state={seq.get('state')})",
                hint="stop it on the board first; the run never stops your music",
            )
        if seq.get("dirty"):
            # Not a stop: opening the run's song writes the open song's pending
            # scenes into its own folder first (SongEngine doOpen). Give the
            # store its own moment anyway.
            seq = self.poll(lambda: verbs.seq(L), lambda r: not r.get("dirty"), 20.0)
        radio = self.wait_radio_off()
        song = verbs.asong(L)
        kit = verbs.kit_status(L)
        ui = verbs.ui_state(L)
        app = verbs.app_list(L).get("running")
        eq = self.mix_eq()
        desk = verbs.mix_lvl(L)
        self.home = {
            "project": song.get("project") or "",
            "kit_id": kit["current"],
            "kit": kit.get("name") or "",
            "app": app,
            "theme": ui.get("theme"),
            "eq0": eq.get(0, ""),
            "desk": {
                c["name"]: {"vol": c.get("vol"), "mute": c.get("mute")}
                for c in desk["channels"]
                if "name" in c
            },
            "seq_file": seq.get("file"),
        }
        self.base = {
            "resets": verbs.reset_count(L),
            "mem": verbs.mem(L),
            "smpl": verbs.smpl_counters(L),
        }
        backup = BACKUP.format(tag=self.tag.lower())
        line = self.raw(f"SETTINGS_EXPORT {backup}", "OK")
        self.check(line.startswith("OK"), f"SETTINGS_EXPORT: {line}")
        st = self.poll(lambda: self.settings_status(), lambda r: r.get("busy") == 0, 10.0)
        self.check(st.get("ok") == 1, f"settings backup failed: {st}")
        self.pushed.append(backup)
        self.home["backup"] = backup
        return {"home": self.home, "base": self.base, "radio": radio}

    def settings_status(self) -> dict:
        return verbs._kv_line(self.raw("SETTINGS_STATUS", "SETTINGS:"), "SETTINGS:")

    def mix_eq(self) -> dict[int, str]:
        lines = self.L.transact_multi("MIX_EQ", "MIXEQ:", end="MIXEQ: end")
        out = {}
        for ln in lines:
            m = re.match(r"MIXEQ: out=(\d+) (.*)", ln.strip())
            if m:  # live= is the processing state, not a setting
                out[int(m.group(1))] = re.sub(r"\s*live=\d+", "", m.group(2))
        return out

    def op_fixture(self, step: workflows.Step) -> dict:
        hits = beat_fixture.hits_from_spec(self.wf.raw["fixture"]["hits"])
        local = self.ctx.workdir / self.names["file"]
        self.loop = beat_fixture.build(local, self.wf.bpm, hits)
        self.push(local, self.names["remote"])
        self.pushed.append(self.names["remote"])
        st = verbs.fs_stat(self.L, self.names["remote"])
        return {
            "local": str(local),
            "remote": self.names["remote"],
            "frames": self.loop.frames,
            "stat": st,
        }

    def op_project(self, step: workflows.Step) -> dict:
        name = self.names["project"]
        verbs.asong(self.L, f"OPEN {name}")
        self.opened_project = True
        st = self.poll(
            lambda: verbs.asong(self.L),
            lambda r: r["open"] in (OPEN_READY, OPEN_FAILED) and r["project"] == name,
        )
        self.check(
            st["open"] == OPEN_READY and st["project"] == name,
            f"ASONG OPEN {name}: open={st['open']} project={st['project']!r}",
        )
        seq = self.poll(lambda: verbs.seq(self.L), lambda r: name in (r.get("file") or ""), 10.0)
        self.check(
            name in (seq.get("file") or ""),
            f"the sequencer still reads {seq.get('file')!r}, not the run's song",
        )
        return {"project": name, "seq_file": seq.get("file"), "scenes": seq.get("scenes")}

    def op_slice(self, step: workflows.Step) -> dict:
        assert self.loop is not None
        out: dict[str, Any] = {"launch": self.launch("Slicer")}
        self.wait(1.0)
        # The file: the Slicer's own list, by the encoder.
        try:
            self.click_on(
                lambda s: "pick a take" in s or ".wav" in s.lower() or s.strip().endswith(".."),
                settle_s=1.0,
            )
            self.click_on(lambda s: self.names["file"] in s, settle_s=1.5)
            out["open"] = "file list"
        except StepFailed as exc:
            out["open"] = f"STOOL OV ({exc})"
            verbs.stool(self.L, f"OV {self.names['remote']}")
        st = self.poll(
            lambda: verbs.stool(self.L),
            lambda r: str(r.get("ov", "")).endswith(f"/{self.loop.frames}"),
            15.0,
        )
        out["ov"] = st.get("ov")
        self.check(
            str(st.get("ov", "")).endswith(f"/{self.loop.frames}"),
            f"the Slicer did not open the loop (ov={st.get('ov')}, want .../{self.loop.frames})",
        )
        sens = int(step.args.get("sens", 50))
        st = verbs.stool(self.L, f"ONSETS {sens}")
        m = re.search(r"slices=(\d+):([\d,]*)", st["line"])
        slices = [int(v) for v in (m.group(2).split(",") if m and m.group(2) else [])]
        out["slices"] = slices
        want = self.loop.onsets_frames
        self.check(len(slices) == len(want), f"{len(slices)} slices, want {len(want)}")
        lo, hi = step.args.get("tolerance_ms", [-14, 2])
        sr = self.loop.sample_rate
        errs = [round((s - w) * 1000.0 / sr, 2) for s, w in zip(slices, want, strict=True)]
        out["error_ms"] = errs
        bad = [e for e in errs if not (lo <= e <= hi)]
        self.check(not bad, f"slices off their hits by {bad} ms (allowed {lo}..{hi})")
        return out

    def op_export(self, step: workflows.Step) -> dict:
        name = self.names["kit"]
        # ATK and PITCH are the verb's own and last until a reboot: as the Slicer saves them.
        verbs.stool(self.L, "ATK 0")
        verbs.stool(self.L, "PITCH 0")
        verbs.stool(self.L, f"EXPORT {name}")
        self.made_kit = True
        st = self.poll(
            lambda: verbs.stool(self.L),
            lambda r: f"result={name}" in r["line"] or (r.get("err") not in (None, "", "-")),
            self.p.timeout_s + 30,
        )
        out: dict[str, Any] = {"kit": name, "stool": st["line"][-200:]}
        self.check(f"result={name}" in st["line"], f"EXPORT {name}: {st['line'][-160:]}")
        kid = self.kit_by_name(name)
        if kid is None and "after a restart" in st["line"]:
            # The Slicer says so on screen too: the list had no room left for it
            # (PortableKitLoader::addKit). A finding, not a stop: the kit is
            # on the card, a boot's scan lists it.
            self.warnings.append(
                f"the Slicer's kit {name} joined the kit list only after a restart "
                f"(addKit had no spare slot; {len(verbs.kit_list(self.L)['kits'])} kits listed)"
            )
            out["restart"] = self.reboot()
            kid = self.kit_by_name(name)
        self.check(kid is not None, f"kit {name} not in KIT_LIST after the export")
        self.kit_id = kid
        out["id"] = kid
        return out

    def wait_radio_off(self) -> str:
        st = self.poll(
            lambda: {"line": self.raw("WIFI_STATUS", "WIFI:")},
            lambda r: "state=off" in r["line"] and "leases=0" in r["line"],
            120.0,
        )
        return st["line"][:60]

    def reboot(self) -> float:
        """Reset through the STM bridge (crosspad_hil.kits, as the fixture pushes do)."""
        from crosspad_hil import kits as kitsmod

        t0 = time.monotonic()
        self.L.close()
        con = self.ctx.open_console(reset=False)
        try:
            kitsmod.reboot_for_kit_scan(self.ctx, con)
        finally:
            con.close()
        self.link = self.ctx.open_cdc()
        self.link.wait_ready()
        kitsmod.wait_for_kit_scan(self.ctx, self.link)
        # A boot with WiFi allowed checks for updates on a lease, and the radio
        # holds ~100 kB of PSRAM until 30 s after it: a kit loaded and the
        # Sampler opened in that window ran LVGL out of memory and hung the
        # board (TWDT on taskLVGL, 18.10). Wait for the radio to go.
        self.wait_radio_off()
        self.rebooted = True
        return round(time.monotonic() - t0, 1)

    def op_load_kit(self, step: workflows.Step) -> dict:
        assert self.kit_id is not None
        try:
            verbs.app_stop(self.L)
        except HilError:
            pass
        st = verbs.kit_load(self.L, self.kit_id, wait_s=self.p.timeout_s)
        pads = []
        for i in range(16):
            pads.append(verbs.pad_info(self.L, i)["raw"])
        has = [i for i, r in enumerate(pads) if re.search(r"\bcount=[1-9]", r)]
        self.check(has == list(range(8)), f"pads with a sample: {[i + 1 for i in has]}, want 1..8")
        self.kit_json = kit_json_from_padinfo(pads[0])
        return {"status": st, "pads": pads[:9], "kit_json": self.kit_json}

    def op_audition(self, step: workflows.Step) -> dict:
        verbs.smpl_counters(self.L)  # read-clear the peak
        for pad in range(8):
            self.press(pad, 110)
            self.wait(0.12)
            self.release(pad)
            self.wait(0.1)
        pk = verbs.smpl_counters(self.L)
        self.check(
            int(pk.get("peak", 0)) > 500, f"the pads are silent (SMPL_PEAK {pk.get('peak')})"
        )
        return {"peak": pk.get("peak")}

    def op_pad_edit(self, step: workflows.Step) -> dict:
        edits = self.wf.edits(self.p.depth)
        out: dict[str, Any] = {"launch": self.launch("Sampler"), "edits": []}
        self.wait(1.0)
        first = edits[0]["pad"]
        self.press(first, 100)
        self.wait(0.1)
        self.release(first)
        self.wait(0.3)
        self.click_on(lambda s: s.strip().startswith("EDIT"), settle_s=1.0)
        for e in edits:
            pad, ctl = int(e["pad"]), str(e["control"])
            self.press(pad, 100)
            self.wait(0.1)
            self.release(pad)
            self.wait(0.4)  # the editor follows a hit on its 50 ms tick
            if ctl == "REV":
                self.click_on(lambda s: s.strip() == "REV", settle_s=0.6)
                self.reversed.append(pad)
                out["edits"].append({"pad": pad + 1, "REV": True})
                continue
            label = self.focus_on(lambda s, c=ctl: s.strip().startswith(c + " "))
            cur = 0
            m = re.search(r"([+-]?\d+)", label)
            if m:
                cur = int(m.group(1))
            delta = int(e["value"]) - cur
            verbs.enc_press(self.L)
            self.wait(0.2)
            while delta:
                d = max(-16, min(16, delta))
                verbs.enc_rotate(self.L, d)
                delta -= d
                self.wait(0.1)
            now = verbs.enc_focus(self.L)["label"]
            verbs.enc_press(self.L)
            self.wait(0.3)
            out["edits"].append({"pad": pad + 1, ctl: now})
            if ctl == "PITCH":
                self.semis[pad] = int(e["value"])
            self.check(str(abs(int(e["value"]))) in now, f"{ctl} on pad {pad + 1} reads {now!r}")
        # Out of the editor and the Sampler: "Unsaved changes" -> Save.
        verbs.pwr_gesture(self.L)
        self.wait(0.6)
        verbs.pwr_gesture(self.L)
        self.wait(0.8)
        try:
            self.click_on(lambda s: s.strip().lower() == "save", settle_s=1.5)
            out["saved_by"] = "dialog"
        except StepFailed:
            out["saved_by"] = "none"
        local = self.pull(self.kit_json_path(), self.ctx.workdir / "kit_after_edit.json")
        kit = json.loads(local.read_text())
        pads = kit["pads"]
        for e in edits:
            p = pads[int(e["pad"])]
            if e["control"] == "PITCH":
                self.check(
                    p.get("pitch") == e["value"],
                    f"kit.json pad {e['pad'] + 1} pitch={p.get('pitch')}, want {e['value']}",
                )
            elif e["control"] == "REV":
                self.check(p.get("reverse") is True, f"kit.json pad {e['pad'] + 1} not reversed")
            elif e["control"] == "VOL":
                self.check(
                    p.get("volume") == e["value"],
                    f"kit.json pad {e['pad'] + 1} volume={p.get('volume')}, want {e['value']}",
                )
        out["kit_json"] = [
            {k: v for k, v in pads[int(e["pad"])].items() if k != "samples"} for e in edits
        ]
        # A reversed pad plays a rendered copy; give the render a moment.
        if self.reversed:
            self.wait(3.0)
        return out

    def kit_json_path(self) -> str:
        if self.kit_json:
            return self.kit_json
        return f"/sdcard/crosspad/slicer/{self.names['kit']}/kit.json"

    def new_scene(self) -> int:
        st = verbs.seq(self.L, "STOP")
        if st["scenes"] >= MAX_SCENES:
            raise StepFailed(f"all {MAX_SCENES} scenes in use in the run's own song")
        verbs.seq(self.L, "ADD")
        return int(st["scenes"])

    def select_scene(self, s: int) -> None:
        verbs.seq(self.L, "STOP")
        verbs.seq(self.L, f"SCENE {s}")
        st = self.poll(
            lambda: verbs.seq(self.L), lambda r: str(r.get("scene", "")).startswith(f"{s}>"), 5.0
        )
        self.check(
            str(st.get("scene", "")).startswith(f"{s}>"),
            f"scene {s} not selected: {st.get('scene')}",
        )

    def record_pattern(self, scene: int, pat: workflows.Pattern) -> None:
        for a in (
            "COUNTIN 0",
            f"BARS {scene} {pat.bars}",
            f"BPM {int(self.wf.bpm)}",
            f"QUANT {int(self.wf.raw.get('quant', 2))}",
        ):
            verbs.seq(self.L, a)
        verbs.seq(self.L, "REC")
        # Where the take's loop starts, read off the board rather than assumed:
        # REC starts the clock from wherever it holds.
        # The pos is taken when the request lands; the reply comes ~200 ms later
        # (measured 18.10), so the send time is the clock to subtract it from.
        t_ask = time.monotonic()
        st = verbs.seq(self.L)
        loop_s = pat.bars * self.wf.bar_s
        t0 = t_ask - float(st.get("pos", 0.0)) * loop_s
        if t0 + LEAD_S < time.monotonic() + 0.05:
            t0 += loop_s
        self.play_hits(list(pat.hits), t0)
        end = t0 + pat.bars * self.wf.bar_s + 0.15
        self.wait(max(0.0, end - time.monotonic()))
        verbs.seq(self.L, "PLAY")

    def dump_scene(self, scene: int) -> list[tuple[int, int, int]]:
        step_frames = self.wf.step_s * SR
        return sorted(
            (int(round(e["pos"] / step_frames)), e["pad"], e["vel"])
            for e in verbs.seq_dump(self.L)
            if e["seq"] == scene
        )

    def compare(
        self, got: list[tuple[int, int, int]], want: list[workflows.Hit], what: str
    ) -> None:
        w = sorted((h.step, h.pad, h.vel) for h in want)
        if got != w:
            missing = sorted(set(w) - set(got))
            extra = sorted(set(got) - set(w))
            raise StepFailed(
                f"{what}: {len(got)} events, want {len(w)}; missing "
                f"{missing[:6]} extra {extra[:6]} (step, pad, vel)"
            )

    def op_record(self, step: workflows.Step) -> dict:
        pat = self.wf.patterns[step.args.get("pattern", "A")]
        # The pads record in the Sequencer, not under the launcher's MODE map.
        launched = self.launch("Sequencer")
        self.wait(0.8)
        self.scene = self.new_scene()
        self.select_scene(self.scene)
        self.record_pattern(self.scene, pat)
        got = self.dump_scene(self.scene)
        self.compare(got, list(pat.hits), f"scene {self.scene + 1} after REC")
        st = verbs.seq(self.L)
        return {
            "launch": launched,
            "scene": self.scene,
            "events": len(got),
            "bars": st["bars"][self.scene],
            "bpm": st.get("bpm"),
            "quant": st.get("quant"),
        }

    def op_overdub(self, step: workflows.Step) -> dict:
        assert self.scene is not None
        pat = self.wf.patterns[step.args.get("pattern", "B")]
        a = self.wf.patterns["A"]
        t_ask = time.monotonic()
        st = verbs.seq(self.L, "OVERDUB")
        loop_s = a.bars * self.wf.bar_s
        t0 = t_ask + (1.0 - float(st.get("pos", 0.0))) * loop_s
        if t0 - time.monotonic() < 0.25:  # too close to the wrap: the pass after
            t0 += loop_s
        self.play_hits(list(pat.hits), t0)
        self.wait(max(0.0, t0 + loop_s + 0.1 - time.monotonic()))
        verbs.seq(self.L, "PLAY")
        got = self.dump_scene(self.scene)
        self.compare(got, list(a.hits) + list(pat.hits), "scene after OVERDUB")
        return {"events": len(got)}

    def op_double(self, step: workflows.Step) -> dict:
        assert self.scene is not None
        before = self.dump_scene(self.scene)
        bars0 = verbs.seq(self.L)["bars"][self.scene]
        st = verbs.seq(self.L, "DOUBLE")
        self.check(
            st["bars"][self.scene] == 2 * bars0, f"x2: bars {bars0} -> {st['bars'][self.scene]}"
        )
        n = bars0 * self.wf.steps_per_bar
        want = [workflows.Hit(s, p, v) for s, p, v in before] + [
            workflows.Hit(s + n, p, v) for s, p, v in before
        ]
        self.compare(self.dump_scene(self.scene), want, "scene after x2")
        return {"bars": st["bars"][self.scene], "events": len(want)}

    def op_erase(self, step: workflows.Step) -> dict:
        assert self.scene is not None
        pad = int(self.wf.raw["erase_pad"])
        before = self.dump_scene(self.scene)
        verbs.seq(self.L, "ERASE 1")
        self.press(pad, 100)
        self.wait(0.1)
        self.release(pad)
        self.wait(0.3)
        verbs.seq(self.L, "ERASE 0")
        want = [workflows.Hit(s, p, v) for s, p, v in before if p != pad]
        self.compare(self.dump_scene(self.scene), want, f"scene after ERASE pad {pad + 1}")
        return {"erased": pad + 1, "events": len(want)}

    def op_scene2(self, step: workflows.Step) -> dict:
        assert self.scene is not None
        pat = self.wf.patterns[step.args.get("pattern", "C")]
        s2 = self.new_scene()
        self.select_scene(s2)
        self.record_pattern(s2, pat)
        self.compare(self.dump_scene(s2), list(pat.hits), f"scene {s2 + 1}")
        verbs.seq(self.L, "STOP")
        verbs.song(self.L, "CLEAR")
        verbs.song(self.L, f"ADD {self.scene} 1")
        st = verbs.song(self.L, f"ADD {s2} 1")
        self.check(
            list(map(tuple, st["blocks"])) == [(self.scene, 1), (s2, 1)],
            f"song blocks {st['blocks']}",
        )
        st = verbs.song(self.L, "PLAY 0")
        self.wait(1.0)
        playing = verbs.song(self.L)
        verbs.seq(self.L, "STOP")
        verbs.song(self.L, "CLEAR")
        self.select_scene(self.scene)
        self.check(playing.get("mode") == 1, f"song mode did not start: {playing}")
        return {"scene2": s2, "blocks": st["blocks"]}

    def op_mixer(self, step: workflows.Step) -> dict:
        desk = verbs.mix_lvl(self.L)
        ch = next((c for c in desk["channels"] if c.get("name") == "Sampler"), None)
        self.check(ch is not None, "no Sampler channel in MIX_LVL")
        i = int(ch["ch"])
        out: dict[str, Any] = {"ch": i, "before": {"vol": ch.get("vol"), "mute": ch.get("mute")}}
        try:
            verbs.mix_vol(self.L, i, 50)
            c = next(c for c in verbs.mix_lvl(self.L)["channels"] if c.get("ch") == i)
            self.check(
                abs(float(c.get("vol", 0)) - 0.5) < 0.02, f"MIX_VOL 50 reads vol={c.get('vol')}"
            )
            verbs.mix_mute(self.L, i, True)
            c = next(c for c in verbs.mix_lvl(self.L)["channels"] if c.get("ch") == i)
            self.check(int(c.get("mute", 0)) == 1, f"MIX_MUTE reads mute={c.get('mute')}")
            self.raw("MIX_EQ 0 0 PEAK 1000 60 100", "OK")
            eq = self.mix_eq().get(0, "")
            out["eq_set"] = eq
            self.check("b0=peak/1000/60/100" in eq, f"MIX_EQ band 0 reads {eq!r}")
        finally:
            verbs.mix_vol(self.L, i, int(round(float(ch.get("vol", 1.0)) * 100)))
            verbs.mix_mute(self.L, i, bool(int(ch.get("mute", 0))))
            self.restore_eq0()
        c = next(c for c in verbs.mix_lvl(self.L)["channels"] if c.get("ch") == i)
        out["after"] = {"vol": c.get("vol"), "mute": c.get("mute")}
        self.check(
            abs(float(c.get("vol", 0)) - float(ch.get("vol", 0))) < 0.011
            and int(c.get("mute", 0)) == int(ch.get("mute", 0)),
            f"desk not back: {out['after']} vs {out['before']}",
        )
        self.check(self.mix_eq().get(0) == self.home["eq0"], "EQ on output 1 not back")
        return out

    def restore_eq0(self) -> None:
        home = self.home.get("eq0", "")
        m = re.search(r"on=(\d)", home)
        for band, kind, hz, db, q in re.findall(r"b(\d)=(\w+)/(-?\d+)/(-?\d+)/(-?\d+)", home):
            self.raw(f"MIX_EQ 0 {band} {kind.upper()} {hz} {db} {q}", "OK")
        if m:
            self.raw(f"MIX_EQ 0 {'ON' if m.group(1) == '1' else 'OFF'}", "OK")

    def op_bounce(self, step: workflows.Step) -> dict:
        from crosspad_hil.workflows import beat_audio

        assert self.scene is not None and self.loop is not None
        track = int(step.args.get("track", 0))
        clips0 = verbs.asong(self.L)["clips"][track]
        verbs.asong(self.L, f"BOUNCE {self.scene} {track}")
        st = self.poll(
            lambda: verbs.asong(self.L),
            lambda r: r["bounce"] in (BOUNCE_DONE, BOUNCE_FAILED),
            self.p.timeout_s + 20,
        )
        self.check(
            st["bounce"] == BOUNCE_DONE, f"bounce={st['bounce']}: {st.get('err') or 'no reason'}"
        )
        self.check(
            st["clips"][track] == clips0 + 1,
            f"track {track + 1}: {clips0} -> {st['clips'][track]} clips",
        )
        remote = st["file"] or ""
        if not remote.startswith("/"):
            remote = f"/sdcard/SONGS/{self.names['project']}/clips/{Path(remote).name}"
        local = self.pull(remote, self.ctx.workdir / Path(remote).name)
        self.artifacts.append(Artifact(path=str(local), mime="audio/wav", role="bounce"))
        bars, hits = workflows.expected_scene(self.wf, self.p.depth)
        exp = beat_audio.expected_hits(
            hits, self.wf.step_s, self.loop.freqs, self.semis, self.reversed
        )
        res = beat_audio.analyze_bounce(
            local,
            exp,
            want_s=bars * self.wf.bar_s,
            onset_tol_s=float(step.args.get("onset_tol_ms", 15)) / 1000,
        )
        res["remote"] = remote
        res["bars"] = bars
        if res["failures"]:
            self.data["steps"]["bounce"] = res
            raise StepFailed("bounce: " + "; ".join(res["failures"][:6]))
        return res

    def op_arrange(self, step: workflows.Step) -> dict:
        L = self.L
        out: dict[str, Any] = {}
        st = verbs.asong(L, "PLAY 0")
        st = self.poll(lambda: verbs.asong(L), lambda r: r["play"] == 1, 10.0)
        self.check(st["play"] == 1, "the song never started")
        self.wait(self.wf.bar_s * 2 + 0.5)
        st = verbs.asong(L)
        verbs.asong(L, "STOP")
        out["play"] = {"under": st["under"], "openfail": st["openfail"], "pos": st["pos"]}
        self.check(
            not st["under"] and not st["openfail"],
            f"while playing: under={st['under']} openfail={st['openfail']}",
        )
        clips0 = verbs.asong(L)["clips"][0]
        st = verbs.asong(L, "CUT 0 1")
        self.check(st.get("ok") == 1, "ASONG CUT 0 1 refused (no block across bar 2?)")
        st = self.poll(lambda: verbs.asong(L), lambda r: r["clips"][0] == clips0 + 1, 5.0)
        self.check(st["clips"][0] == clips0 + 1, f"cut: track 1 {clips0} -> {st['clips'][0]} clips")
        len0 = int(st["len"])
        verbs.asong(L, "REGION 0 2")
        st = verbs.asong(L, "DUPREGION")
        self.check(st.get("ok") == 1, "ASONG DUPREGION refused")
        st = self.poll(lambda: verbs.asong(L), lambda r: int(r["len"]) == len0 + 2, 5.0)
        out["len"] = [len0, int(st["len"])]
        self.check(int(st["len"]) == len0 + 2, f"region x2: length {len0} -> {st['len']} bars")
        verbs.asong(L, "REGION 0 0")
        return out

    def op_arrange_full(self, step: workflows.Step) -> dict:
        L = self.L
        out: dict[str, Any] = {}
        pat = self.wf.patterns[step.args.get("pattern", "A")]
        bar = int(step.args.get("rec_bar", 4))
        st = verbs.asong(L, f"REC {bar} 0")
        self.check(st.get("ok") == 1, f"ASONG REC {bar} refused")
        st = self.poll(lambda: verbs.asong(L), lambda r: r["play"] == 1, 10.0)
        # The take starts at `bar`; the transport is there now.
        pos = float(st.get("pos", bar))
        t0 = time.monotonic() + max(0.0, (bar - pos)) * self.wf.bar_s
        self.play_hits(list(pat.hits), t0)
        self.wait(max(0.0, t0 + pat.bars * self.wf.bar_s + 0.2 - time.monotonic()))
        verbs.asong(L, "STOP")
        self.wait(0.5)
        rows = L.transact_multi("ASONG_SEQ", "ASEQ:", end=None)
        own = sum(int(verbs._kv_line(r, "ASEQ:").get("notes", 0)) for r in rows if "own=1" in r)
        out["asong_seq"] = rows
        self.check(
            own == len(pat.hits), f"take onto the song: {own} own notes, want {len(pat.hits)}"
        )
        assert self.loop is not None
        # An empty track asked for becomes the first free one (SongEngine
        # audioTarget): the clip is counted over the whole song.
        clips0 = sum(verbs.asong(L)["clips"])
        st = verbs.asong(L, f"IMPORT {self.names['remote']} 0 {self.loop.frames} 3 0 0")
        self.check(st.get("ok") == 1, "ASONG IMPORT refused")
        st = self.poll(lambda: verbs.asong(L), lambda r: sum(r["clips"]) == clips0 + 1, 20.0)
        out["import_clips"] = st["clips"]
        self.check(
            sum(st["clips"]) == clips0 + 1,
            f"import: {clips0} -> {sum(st['clips'])} clips "
            f"(edit={st.get('edit')} err={st.get('err')})",
        )
        # The Live set: edit= 2 done, 3 failed; REPORT has the export's lines
        # (none before the first export of a song).
        verbs.asong(L, "EXPORT")
        deadline = time.monotonic() + 45.0
        rep: list[str] = []
        while time.monotonic() < deadline:
            self.wait(0.5)
            rep = [r for r in L.transact_multi("ASONG REPORT", "REPORT:", end=None)
                   if r.startswith("REPORT:")]
            if rep and verbs.asong(L).get("edit") in (2, 3):
                break
        st = verbs.asong(L)
        out["export"] = {"edit": st.get("edit"), "report": rep[:12]}
        if st.get("edit") != 2:
            why = "; ".join(r[len("REPORT: "):] for r in rep[:3]) or "no report"
            if "memory" in why.lower():
                # This board's PSRAM (an 88-kit card, a kit and a song open)
                # is short for the Set: a finding to report, not this test's
                # verdict on the arrangement.
                self.warnings.append(f"Live-set EXPORT failed: {why}")
            else:
                raise StepFailed(f"Live-set EXPORT failed (edit={st.get('edit')}): {why}")
        verbs.asong(L, "PLAY 0")
        self.poll(lambda: verbs.asong(L), lambda r: r["play"] == 1, 10.0)
        self.wait(int(verbs.asong(L)["len"]) * self.wf.bar_s + 0.5)
        st = verbs.asong(L)
        verbs.asong(L, "STOP")
        out["play"] = {"under": st["under"], "openfail": st["openfail"]}
        self.check(
            not st["under"] and not st["openfail"],
            f"whole song: under={st['under']} openfail={st['openfail']}",
        )
        return out

    def screenshot_top(self, name: str) -> tuple[int, int, int]:
        line = self.raw("SCREENSHOT", "SCREENSHOT:", timeout_s=10.0)
        self.check(line.startswith("SCREENSHOT: /"), f"SCREENSHOT: {line}")
        local = self.pull(SCREENSHOT, self.ctx.workdir / f"{name}.bmp")
        self.artifacts.append(Artifact(path=str(local), mime="image/bmp", role=name))
        return header_colour(local)

    def op_settings(self, step: workflows.Step) -> dict:
        out: dict[str, Any] = {"launch": self.launch("Settings")}
        self.wait(1.0)
        theme0 = int(verbs.ui_state(self.L).get("theme", 0))
        self.click_on(lambda s: DISPLAY_ICON in s or "display" in s.lower(), settle_s=0.8)
        before = self.screenshot_top("settings_before")
        self.click_on(lambda s: s.strip() == "Theme Color", settle_s=0.6)
        verbs.enc_rotate(self.L, 1)
        self.wait(0.3)
        verbs.enc_press(self.L)
        self.wait(1.0)
        theme1 = int(verbs.ui_state(self.L).get("theme", 0))
        after = self.screenshot_top("settings_changed")
        out.update(theme=[theme0, theme1], colour=[before, after])
        self.check(theme1 != theme0, f"Theme Color did not change (theme={theme1})")
        self.check(
            colour_dist(before, after) > 40,
            f"the header did not change colour: {before} -> {after}",
        )
        self.raw(f"SETTINGS_IMPORT {self.home['backup']}", "OK")
        st = self.poll(lambda: self.settings_status(), lambda r: r.get("busy") == 0, 10.0)
        self.check(st.get("ok") == 1, f"SETTINGS_IMPORT of the backup: {st}")
        self.wait(0.8)
        theme2 = int(verbs.ui_state(self.L).get("theme", 0))
        back = self.screenshot_top("settings_restored")
        out.update(restored=theme2, colour_back=back)
        self.check(theme2 == theme0, f"theme {theme2} after the backup, want {theme0}")
        self.check(colour_dist(before, back) <= 12, f"the header is {back}, was {before}")
        return out

    # ── putting back ────────────────────────────────────────────────────
    def cleanup(self) -> dict:
        left: dict[str, str] = {}
        L = self.L

        def step(name: str, fn: Callable[[], Any]) -> None:
            try:
                fn()
            except Exception as exc:  # noqa: BLE001 - every step on its own
                left[name] = f"{type(exc).__name__}: {exc}"

        step("seq_stop", lambda: verbs.seq(L, "STOP"))
        step("song_stop", lambda: verbs.asong(L, "STOP"))
        step("app", lambda: verbs.app_stop(L))
        if self.home.get("kit"):

            def home_kit() -> None:
                # Before the songs change: this load goes through the kit task,
                # which releases the run's kit (removeKit refuses a "retired"
                # one), and it lands in the run's song, not the user's.
                self.poll(lambda: verbs.kit_status(L), lambda r: not r["loading"], 20.0)
                if verbs.kit_status(L).get("name") != self.home["kit"]:
                    kid = self.kit_by_name(self.home["kit"])
                    if kid is None:
                        raise RuntimeError(f"kit {self.home['kit']!r} not on the list")
                    verbs.kit_load(L, kid, wait_s=self.p.timeout_s)

            step("home_kit", home_kit)
        if self.made_kit and not self.p.keep:

            def drop_kit() -> None:
                kid = self.kit_by_name(self.names["kit"])
                if kid is None:
                    return

                def ask() -> bool:
                    line = self.raw(f"KIT_DELETE {kid} {self.names['kit']}", "OK")
                    if not line.startswith("OK"):
                        raise RuntimeError(line)
                    self.wait(1.5)
                    return self.kit_by_name(self.names["kit"]) is None

                if ask():
                    return
                # The players keep the kit's files open after a swap, and a
                # file that is open cannot be unlinked (removeKit fails, the
                # kit stays): another kit's load lets go of them. Still in the
                # run's own song, so no song of the user's records the swap.
                other = next(
                    k["id"]
                    for k in verbs.kit_list(L)["kits"]
                    if k["name"] not in (self.names["kit"], self.home.get("kit"))
                )
                verbs.kit_load(L, other, wait_s=self.p.timeout_s)
                gone = ask()
                home = self.kit_by_name(self.home.get("kit") or "")
                if home is not None:
                    verbs.kit_load(L, home, wait_s=self.p.timeout_s)
                if not gone:
                    raise RuntimeError("still on the list after a swap released its files")

            step("kit", drop_kit)
        if self.opened_project and not self.p.keep:

            def drop_project() -> None:
                if verbs.asong(L).get("project") == self.names["project"]:
                    verbs.asong(L, "DELETE")
                    self.poll(
                        lambda: verbs.asong(L),
                        lambda r: (
                            r.get("project") != self.names["project"]
                            and r["open"] in (OPEN_READY, OPEN_FAILED)
                        ),
                        15.0,
                    )

            step("project", drop_project)
        if self.home.get("project"):

            def home_project() -> None:
                if verbs.asong(L).get("project") != self.home["project"]:
                    verbs.asong(L, f"OPEN {self.home['project']}")
                st = self.poll(
                    lambda: verbs.asong(L),
                    lambda r: (
                        r.get("project") == self.home["project"]
                        and r["open"] in (OPEN_READY, OPEN_FAILED)
                    ),
                    15.0,
                )
                if st.get("project") != self.home["project"]:
                    raise RuntimeError(f"open song is {st.get('project')!r}")

            step("home_project", home_project)
        if self.home.get("kit"):

            def home_kit_after() -> None:
                self.wait(1.5)  # the home song brings its own kit (song_sounds)
                self.poll(lambda: verbs.kit_status(L), lambda r: not r["loading"], 20.0)
                if verbs.kit_status(L).get("name") != self.home["kit"]:
                    kid = self.kit_by_name(self.home["kit"])
                    if kid is None:
                        raise RuntimeError(f"kit {self.home['kit']!r} not on the list")
                    verbs.kit_load(L, kid, wait_s=self.p.timeout_s)

            step("home_kit_after", home_kit_after)
        if self.home.get("eq0") is not None and "eq0" in self.home:
            step("eq", lambda: self.mix_eq().get(0) == self.home["eq0"] or self.restore_eq0())
        if self.home.get("theme") is not None:

            def theme() -> None:
                if int(verbs.ui_state(L).get("theme", -1)) != int(self.home["theme"]):
                    self.raw(f"SETTINGS_IMPORT {self.home['backup']}", "OK")
                    self.poll(lambda: self.settings_status(), lambda r: r.get("busy") == 0, 10.0)

            step("theme", theme)
        if not self.p.keep:
            for remote in self.pushed:
                step(f"rm {remote}", lambda r=remote: verbs.fs_delete(L, r))
        if self.home.get("app") and self.home["app"] not in ("-", "Launcher"):
            step("home_app", lambda: verbs.app_start(L, self.home["app"], wait_s=5.0))
        return left

    def verdict_after(self) -> list[str]:
        out: list[str] = []
        L = self.L
        if not getattr(self, "base", None):
            return out
        resets = verbs.reset_count(L)
        if resets != self.base["resets"] + (1 if self.rebooted else 0):
            out.append(f"{resets - self.base['resets']} reset(s) during the run")
        smpl = verbs.smpl_counters(L)
        for key in ("underrun", "overruns"):
            a, b = self.base["smpl"].get(key), smpl.get(key)
            if isinstance(a, int) and isinstance(b, int) and b != a and not self.rebooted:
                out.append(f"SMPL {key} {a} -> {b}")
        mem = verbs.mem(L)
        self.data["mem"] = {"before": self.base["mem"], "after": mem}
        if self.rebooted:
            self.data["mem"]["note"] = "the run rebooted the board: no before/after comparison"
            mem_checks: dict[str, int] = {}
        else:
            mem_checks = MEM_TOLERANCE
        for key, tol in mem_checks.items():
            a, b = self.base["mem"].get(key), mem.get(key)
            if isinstance(a, int) and isinstance(b, int) and a - b > tol:
                out.append(f"MEM {key} {a} -> {b} (-{(a - b) // 1024} kB)")
        pf = mem.get("psram_free")
        if isinstance(pf, int) and pf < PSRAM_FLOOR:
            out.append(f"MEM psram_free {pf} after the run, under {PSRAM_FLOOR // 1024} kB")
        return out


def header_colour(bmp: Path, row: int = 60) -> tuple[int, int, int]:
    """Median colour of one row of a 24-bit BMP: by default the middle of the
    settings page's header bar (rows 48..76 under the top bar, which ends at
    38), painted in the theme colour (the transport bar is not themed)."""
    data = bmp.read_bytes()
    off = struct.unpack_from("<I", data, 10)[0]
    w, h = struct.unpack_from("<ii", data, 18)
    bpp = struct.unpack_from("<H", data, 28)[0]
    step = bpp // 8
    stride = (w * step + 3) & ~3
    y = row if h < 0 else abs(h) - 1 - row
    base = off + y * stride
    px = [tuple(data[base + x * step + 2 - c] for c in range(3)) for x in range(abs(w))]
    px.sort(key=lambda p: sum(p))
    return px[len(px) // 2]


def colour_dist(a: tuple[int, int, int], b: tuple[int, int, int]) -> int:
    return sum(abs(x - y) for x, y in zip(a, b, strict=True))


class BeatScenario:
    name = "beat"
    Params = Params
    description = "make a beat: slice a prepared loop, sequence it, bounce it, check every hit"

    def run(self, ctx: Context, params: Params, progress: Progress) -> Report:
        run = BeatRun(ctx, params)
        ctx.partial = run.data
        steps = run.wf.for_depth(params.depth)
        run.data["plan"] = [s.id for s in steps]
        t_start = time.monotonic()
        failed_at: str | None = None
        bench: HilError | None = None
        try:
            run.link = ctx.open_cdc()
            for i, step in enumerate(steps):
                if step.op == "cleanup":
                    continue
                if ctx.cancelled.is_set():
                    break
                progress(i, len(steps), f"{step.id}: {step.caption[1] or step.caption[0]}")
                t0 = time.monotonic()
                try:
                    res = getattr(run, f"op_{step.op}")(step)
                    run.data["steps"][step.id] = {
                        "ok": True,
                        "s": round(time.monotonic() - t0, 1),
                        **(res or {}),
                    }
                except StepFailed as exc:
                    run.data["steps"].setdefault(step.id, {})
                    run.data["steps"][step.id].update(
                        ok=False, s=round(time.monotonic() - t0, 1), why=str(exc)
                    )
                    run.failures.append(f"{step.id}: {exc}")
                    failed_at = step.id
                    break
        except HilError as exc:
            if exc.code != ENV:
                raise
            bench = exc
        finally:
            if run.link is not None:
                try:
                    progress(len(steps) - 1, len(steps), "cleanup: everything put back")
                    t0 = time.monotonic()
                    run.data["left"] = run.cleanup()
                    run.data["steps"]["cleanup"] = {
                        "s": round(time.monotonic() - t0, 1),
                        "left": run.data["left"],
                    }
                    run.failures += [
                        f"not put back: {k} ({v})" for k, v in run.data["left"].items()
                    ]
                    if bench is None:
                        run.failures += run.verdict_after()
                except Exception as exc:  # noqa: BLE001 - the report must still be written
                    run.failures.append(f"cleanup: {type(exc).__name__}: {exc}")
            ctx.close_all()
        if bench is not None:
            run.data["failures"] = [bench.message] + run.failures
            return Report(
                passed=False,
                summary=f"beat: BENCH — {bench.message}",
                data=run.data,
                artifacts=run.artifacts,
                exit_code=2,
            )
        run.data["seconds"] = round(time.monotonic() - t_start, 1)
        run.data["failures"] = run.failures
        run.data["warnings"] = run.warnings
        passed = not run.failures
        took = f"{run.data['seconds']:.0f} s"
        warn = ""
        if run.warnings:
            warn = f"; {len(run.warnings)} warning(s): " + "; ".join(run.warnings)
        summary = (
            f"beat ({params.depth}): PASS in {took}, {len(steps)} steps{warn}"
            if passed
            else f"beat ({params.depth}): FAIL at {failed_at or 'the end'} — "
            + "; ".join(run.failures[:4])
        )
        return Report(
            passed=passed,
            summary=summary,
            data=run.data,
            artifacts=run.artifacts,
            exit_code=0 if passed else 1,
        )


SCENARIO = register(BeatScenario())
