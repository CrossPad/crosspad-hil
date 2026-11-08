"""Plays one block of the script on time, and notices when the board stops answering.

``Performer`` turns a block's steps into one time-ordered list of sends (pad
presses and releases, swaps, previews, the codec's negative control, a host
note, the stream pausing, a bounce, MIDI clock and transport, app switches)
and plays it from a single thread: wait until the moment (sleeping, never
spinning -- a spin holds the GIL the duplex stream needs), send, log the real
send time. The model uses the logged time, not the planned one.

``Prober`` runs beside it: an echo a second, KIT_STATUS four times a second
around swaps, UI_STATE while another app is in front -- through the companion,
the only state readable in the audio profile.

The watchdog (a callable the scenario supplies) is asked between sends; when
it answers with a reason (no capture frames for 2 s, the duplex stream died,
three echoes or companion requests unanswered, the console silent for 15 s)
the block stops and is reported as a hang.
"""
from __future__ import annotations

import threading
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from crosspad_hil.jam.actions import ActionLog
from crosspad_hil.jam.control import UacLink
from crosspad_hil.jam.script import Block, Step

CLOCK_PPQN = 24
WATCH_EVERY_S = 0.25


@dataclass(order=True)
class Event:
    t: float
    order: int
    kind: str = field(compare=False)
    args: dict[str, Any] = field(compare=False, default_factory=dict)


@dataclass
class BlockRun:
    sent: int = 0
    failed: int = 0
    hung: str | None = None
    t_hung: float | None = None
    swaps: list[dict[str, Any]] = field(default_factory=list)
    bounces: list[dict[str, Any]] = field(default_factory=list)
    planned: int = 0


class Performer:
    def __init__(self, link: UacLink, log: ActionLog, *, kits: dict[str, int],
                 scenes: dict[str, int], clock: Callable[[], float] = time.monotonic,
                 sleep: Callable[[float], None] = time.sleep,
                 cancelled: threading.Event | None = None,
                 watchdog: Callable[[], str | None] | None = None,
                 recording: Any = None, route_home: dict[str, Any] | None = None) -> None:
        self.link = link
        self.log = log
        self.kits = kits
        self.scenes = scenes
        self.clock = clock
        self.sleep = sleep
        self.cancelled = cancelled or threading.Event()
        self.watchdog = watchdog or (lambda: None)
        self.recording = recording
        self.route_home = route_home or {}

    # -- the plan ------------------------------------------------------------
    def plan(self, block: Block, t0: float) -> list[Event]:
        ev: list[Event] = []
        n = 0

        def add(t: float, kind: str, **args: Any) -> None:
            nonlocal n
            n += 1
            ev.append(Event(t0 + t, n, kind, args))

        for st in block.steps:
            if st.skip:
                add(st.t_rel, "skip", step=st.id, why=st.skip)
                continue
            add(st.t_rel, "step", step=st)
            self._plan_step(st, add)
        ev.sort()
        return ev

    def _plan_step(self, st: Step, add: Callable[..., None]) -> None:
        a = st.args
        if st.action == "pattern":
            for h in st.hits:
                add(st.t_rel + h.t, "press", step=st.id, pad=h.pad, vel=h.vel,
                    hold_ms=h.hold_ms, via=h.via)
                add(st.t_rel + h.t + h.hold_ms / 1000.0, "release", step=st.id, pad=h.pad,
                    via=h.via)
        elif st.action == "swap":
            add(st.t_rel, "swap", step=st.id, name=a["kit"], via=a.get("via", "sysex"))
        elif st.action == "preview":
            for i, pad in enumerate(a.get("pads", [])):
                add(st.t_rel + i * float(a.get("gap", 1.0)), "preview", step=st.id, pad=pad)
        elif st.action == "codec_neg":
            vol = self.route_home.get("vol") or [80, 80]
            mute = self.route_home.get("mute") or [0, 0]
            add(st.t_rel, "codec", step=st.id, sub=0x04, values=[0, max(0, int(vol[0]) - 30)])
            add(st.t_rel + 0.8, "codec", step=st.id, sub=0x04, values=[0, int(vol[0])])
            add(st.t_rel + 1.6, "codec", step=st.id, sub=0x05, values=[1, 1])
            add(st.t_rel + 2.4, "codec", step=st.id, sub=0x05, values=[1, int(mute[1])])
        elif st.action == "note_neg":
            note = int(a.get("note", 60))
            for k in range(3):
                add(st.t_rel + k * 0.8, "note_on", step=st.id, note=note + k)
                add(st.t_rel + k * 0.8 + 0.3, "note_off", step=st.id, note=note + k)
        elif st.action == "stream_pause":
            add(st.t_rel, "pause", step=st.id, seconds=float(a.get("seconds", 2.0)),
                mode=a.get("mode", "drop"))
        elif st.action == "bounce":
            add(st.t_rel, "bounce", step=st.id, scene=a["scene"], track=int(a["track"]))
        elif st.action == "clock":
            bpm = float(a.get("bpm", 120))
            tick = 60.0 / bpm / CLOCK_PPQN
            plan = {str(k): float(t) for k, t in a.get("plan", [])}
            t_end = min(st.dur, plan.get("F8_stop", st.dur))
            for k, t in sorted(plan.items(), key=lambda x: x[1]):
                if k in ("FA", "FB", "FC"):
                    add(st.t_rel + t, "realtime", step=st.id, byte={"FA": 0xFA, "FB": 0xFB,
                                                                   "FC": 0xFC}[k], what=k)
            stopped = sorted((plan[k], k) for k in ("FC", "FB", "FA") if k in plan)
            t = 0.0
            while t < t_end:
                # ticks keep running through Stop (a DAW keeps its clock up), as hosts do
                add(st.t_rel + t, "tick", step=st.id)
                t += tick
            del stopped
            add(st.t_rel + t_end, "clock_end", step=st.id)
        elif st.action == "ui":
            apps = list(a.get("apps", ["Mixer", "Sampler"]))
            for i, app in enumerate(apps):
                add(st.t_rel + i * 4.0, "app", step=st.id, name=app)

    # -- the play --------------------------------------------------------------
    def run(self, events: list[Event]) -> BlockRun:
        out = BlockRun(planned=len(events))
        last_watch = -1e9
        for e in events:
            while True:
                if self.cancelled.is_set():
                    return out
                now = self.clock()
                if now - last_watch >= WATCH_EVERY_S:
                    last_watch = now
                    why = self.watchdog()
                    if why:
                        out.hung, out.t_hung = why, now
                        self.log.emit("hung", reason=why)
                        return out
                if now >= e.t:
                    break
                self.sleep(min(e.t - now, 0.02))
            self._do(e, out)
        return out

    def _do(self, e: Event, out: BlockRun) -> None:
        a = e.args
        link = self.link
        act = None
        if e.kind == "step":
            st: Step = a["step"]
            self.log.step = st.id
            act = self.log.emit("step", id=st.id, dur=st.dur, flags=st.flags,
                                action=st.action)
        elif e.kind == "skip":
            act = self.log.emit("step_skipped", id=a["step"], why=a["why"])
        elif e.kind == "press":
            act = link.pad(a["pad"], a["vel"], hold_ms=a["hold_ms"], via=a["via"])
        elif e.kind == "release":
            act = link.release(a["pad"], via=a["via"])
        elif e.kind == "swap":
            kid = self.kits.get(a["name"])
            if kid is None:
                act = self.log.emit("kit", ok=False, err=f"no kit {a['name']!r}", name=a["name"])
            else:
                act = link.kit(kid, a["name"], via=a["via"])
                out.swaps.append({"aid": act.aid, "name": a["name"], "kit": kid,
                                  "t": act.t_mono, "via": a["via"]})
        elif e.kind == "preview":
            act = link.preview(a["pad"])
        elif e.kind == "codec":
            act = link.route(a["sub"], *a["values"], what="codec_neg")
        elif e.kind == "note_on":
            act = link.note(True, a["note"])
        elif e.kind == "note_off":
            act = link.note(False, a["note"], 0)
        elif e.kind == "pause":
            t0 = self.clock()
            err = None
            try:
                if self.recording is not None and hasattr(self.recording, "pause_out"):
                    self.recording.pause_out(a["seconds"], a["mode"])
                else:
                    err = "no duplex stream to pause"
            except Exception as ex:  # noqa: BLE001
                err = f"{type(ex).__name__}: {ex}"
            act = self.log.emit("stream_pause", t0=t0, ok=err is None, err=err,
                                seconds=a["seconds"], mode=a["mode"])
        elif e.kind == "bounce":
            scene = self.scenes.get(str(a["scene"]))
            if scene is None:
                act = self.log.emit("bounce", ok=False, err=f"no scene {a['scene']}",
                                    label=a["scene"])
            else:
                act = link.bounce(scene, a["track"], str(a["scene"]))
                out.bounces.append({"aid": act.aid, "scene": scene, "label": a["scene"],
                                    "track": a["track"], "t": act.t_mono, "via": "sysex"})
        elif e.kind == "realtime":
            act = link.realtime(a["byte"], a["what"])
        elif e.kind == "tick":
            act = link.realtime(0xF8, "F8")
        elif e.kind == "clock_end":
            act = self.log.emit("clock_end")
        elif e.kind == "app":
            act = link.notify(f"APP_START {a['name']}", kind="app", name=a["name"])
        if act is not None:
            out.sent += 1
            if not act.ok:
                out.failed += 1


class Prober(threading.Thread):
    """Echo once a second; KIT_STATUS at 4 Hz inside ``swap_windows``; UI_STATE at 1 Hz
    inside ``ui_windows`` (host monotonic spans). Everything goes to the action log."""

    def __init__(self, link: UacLink, log: ActionLog, *, swap_windows: list[tuple[float, float]],
                 ui_windows: list[tuple[float, float]], clock: Callable[[], float] = time.monotonic,
                 echo_every_s: float = 1.0, companion: bool = True) -> None:
        super().__init__(name="jam_prober", daemon=True)
        self.link = link
        self.log = log
        self.swap_windows = swap_windows
        self.ui_windows = ui_windows
        self.clock = clock
        self.echo_every_s = echo_every_s
        self.companion = companion
        self._halt = threading.Event()
        self.kit_status: list[tuple[float, str]] = []
        self.ui_state: list[tuple[float, str]] = []

    def stop(self) -> None:
        self._halt.set()

    def run(self) -> None:
        last_echo = last_ui = last_kit = -1e9
        while not self._halt.wait(0.05):
            now = self.clock()
            if now - last_echo >= self.echo_every_s:
                last_echo = now
                self.link.echo()
            if not self.companion:
                continue
            if any(a <= now <= b for a, b in self.swap_windows) and now - last_kit >= 0.25:
                last_kit = now
                r = self.link.companion("KIT_STATUS", timeout_s=0.5)
                if r is not None:
                    self.kit_status.append((now, r.strip()))
                    self.log.emit("probe", verb="KIT_STATUS", reply=r.strip()[:160])
            if any(a <= now <= b for a, b in self.ui_windows) and now - last_ui >= 1.0:
                last_ui = now
                r = self.link.companion("UI_STATE", timeout_s=0.5)
                if r is not None:
                    self.ui_state.append((now, r.strip()))
                    self.log.emit("probe", verb="UI_STATE", reply=r.strip()[:160])
