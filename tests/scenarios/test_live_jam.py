"""Scenario ``live_jam`` against a fake board: the whole set, on a virtual clock.

The fake board answers every verb the jam uses, keeps a sequencer that records
what it is played (quantised as the firmware does), swaps kits and logs the
swaps, bounces, keeps preset slots and files. Its capture renders the take a
perfect board would have recorded from what the scenario actually did (its
own action log and readbacks, through the same model the analysis uses) --
so a clean set must come back a ZERO-GLITCH PASS, a fault the fake puts into
a take must come back as a FAIL naming its moment and step, a board that
stops answering must be a hang that is recovered, and everything the jam
changed must be put back.
"""
from __future__ import annotations

import json
import queue
import threading
import wave
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from crosspad_hil import card_hold, verbs
from crosspad_hil.devices import AudioCardInfo, Device, MidiPortInfo, Ports, SerialPortInfo, UsbMode
from crosspad_hil.errors import HilError
from crosspad_hil.jam import spec as S
from crosspad_hil.jam.actions import load_actions
from crosspad_hil.jam.calibration import Calibration
from crosspad_hil.jam.model import BlockInput, build_expectation, quantise, samples_per_beat
from crosspad_hil.jam.render import Fault, render_expected, write_wav
from crosspad_hil.scenarios import _audio, base, get, live_jam
from crosspad_hil.scenarios.live_jam import Params

SR = 44_100
CDC = SerialPortInfo(path="/dev/ttyACM0", vid=0x303A, pid=0x3456, serial="CP-1",
                     product="Crosspad", location="1-2.1")
UAC2 = AudioCardInfo(name="Crosspad Audio", sounddevice_index=3, alsa_id="Audio")
ESP = MidiPortInfo(name="Crosspad MIDI 1", rtmidi_out=0, rtmidi_in=0, alsa_hw=None, rawmidi=None)
KITS = {"User Kit": 7, S.JAM_A: 20, S.JAM_B: 21, S.JAM_FX: 22, S.JAM_P: 23}


def device(mode: UsbMode) -> Device:
    ports = Ports(uac2=UAC2, esp_midi=ESP) if mode is UsbMode.AUDIO else Ports(cdc=CDC, esp_midi=ESP)
    return Device(id="dev_jam1", serial="CP-1", usb_mode=mode, ports=ports)


class VClock:
    def __init__(self) -> None:
        self.t = 5_000.0
        self.lock = threading.Lock()

    def now(self) -> float:
        with self.lock:
            return self.t

    def sleep(self, s: float) -> None:
        with self.lock:
            self.t += max(0.0, s) + 1e-5


def desk_json(name: str) -> dict:
    chans = []
    for i, n in enumerate(("Sampler", "Pitched", "USB", "Click", "IN1", "IN2", "Song", "Preview")):
        chans.append({"id": i, "name": n, "volume": 1.0, "muted": False, "soloed": False,
                      "pan": 0.0, "routes": [{"out": o, "enabled": n in ("Sampler", "Pitched")
                                              or o < 2, "volume": 1.0} for o in range(3)]})
    return {"version": 1, "name": name, "gains": [24, 0],
            "desk": {"version": 3, "numOutputs": 3, "channels": chans,
                     "outputs": [{"volume": 1.0, "muted": False} for _ in range(3)]}}


class World:
    """The fake board."""

    def __init__(self, clock: VClock, tmp: Path | None = None) -> None:
        self.c = clock
        self.tmp = tmp or Path(".")
        self.calls: list[str] = []
        self.mode = UsbMode.DEFAULT
        self.kit = 7
        self.app: str | None = None
        self.scenes = 2
        self.bars = [1, 2, 1, 1, 1, 1, 1, 1]
        self.events: dict[int, list[dict]] = {0: [{"pos": 0, "pad": 0, "vel": 100}], 1: []}
        self.active = 0
        self.pending = 0
        self.seq_mode = 0          # 0 none 1 playback 2 record 3 overdub
        self.seq_state = 0         # 0 idle 3 pre-beat 4 running
        self.t0 = self.c.now()     # the metronome's pass start
        self.bpm = 100.0
        self.quant = 2
        self.countin = 1
        self.erase = False
        self.song: list[tuple[int, int]] = []
        self.song_mode = 0
        self.song_loop = 0
        self.project = "My Song"
        self.bounce_state = 0
        self.bounce_file = "-"
        self.clips = [0, 0, 0, 0]
        self.presets = {1: ("f", "STREAM"), 2: ("f", "BEATS"), 3: ("f", "SAMPLING"),
                        4: ("e", "Empty"), 5: ("e", "Empty"), 6: ("s", "Mine")}
        self.files: dict[str, bytes] = {
            "/sdcard/crosspad/sequences.json": b'{"scenes": "the user\'s"}',
            "/sdcard/crosspad/mixer/preset_6.json": json.dumps(desk_json("Mine")).encode()}
        self.swap_rows: list[dict] = []
        self.counters = {"underrun": 0, "noblock": 0}
        self.pads = {"press": 0, "release": 0, "played": 0, "freeslots": 8, "inj_drop": 0,
                     "uart_ovf": 0, "sbuf_drop": 0, "usb_drop": 0}
        self.resets = 0
        self.rx: queue.Queue[bytes] = queue.Queue()
        self.hang_at: float | None = None       # virtual time the board stops answering
        self.faults: dict[str, list[Fault]] = {}
        self.stool = {"rec": 0, "file": "-", "frames": 0}
        self.cdc_after_reboot: list[str] = []

    # -- the metronome / sequencer ------------------------------------------------
    @property
    def spb(self) -> int:
        return samples_per_beat(self.bpm, SR)

    def loop(self, scene: int | None = None) -> int:
        s = self.active if scene is None else scene
        return self.bars[s] * 4 * self.spb

    def pos(self) -> int:
        return int(((self.c.now() - self.t0) * SR) % self.loop())

    def seq(self, link: Any, action: str | None = None, timeout_s: float | None = None) -> dict:
        if action:
            self.calls.append(f"SEQ {action}")
            a = action.split()
            if a[0] == "ADD":
                self.scenes += 1
                self.events[self.scenes - 1] = []
            elif a[0] == "SCENE":
                self.active = self.pending = int(a[1])
                self.t0 = self.c.now()
            elif a[0] == "BARS" and len(a) == 3:
                self.bars[int(a[1])] = int(a[2])
            elif a[0] == "BPM":
                self.bpm = float(a[1])
            elif a[0] == "QUANT":
                self.quant = int(a[1])
            elif a[0] == "COUNTIN":
                self.countin = int(a[1])
            elif a[0] == "REC":
                self.events[self.active] = []
                self.seq_mode = 2
                if self.seq_state != 4:
                    self.t0 = self.c.now() + (4 * self.spb / SR if self.countin else 0.0)
                self.seq_state = 4
            elif a[0] == "OVERDUB":
                self.seq_mode = 3
            elif a[0] == "PLAY":
                self.seq_mode, self.seq_state = 1, 4
            elif a[0] == "STOP":
                self.seq_mode, self.seq_state = 0, 0
            elif a[0] == "DOUBLE":
                n = self.loop()
                self.events[self.active] += [{**e, "pos": e["pos"] + n}
                                             for e in self.events[self.active]]
                self.bars[self.active] *= 2
            elif a[0] == "ERASE":
                self.erase = a[1] == "1"
            elif a[0] == "DEL":
                s = int(a[1])
                if s < self.scenes:
                    self.scenes -= 1
                    self.events.pop(s, None)
        return {"mode": self.seq_mode, "state": self.seq_state, "scenes": self.scenes,
                "scene": f"{self.active}>{self.pending}", "bars": list(self.bars),
                "bpm": self.bpm, "quant": self.quant, "countin": self.countin, "dirty": 0,
                "file": "/sdcard/crosspad/sequences.json"}

    def pad_press(self, link: Any, pad: int, vel: int = 127, at: float | None = None) -> dict:
        self.pads["press"] += 1
        self.pads["played"] += 1
        if self.erase:
            self.events[self.active] = [e for e in self.events[self.active] if e["pad"] != pad]
        elif self.seq_mode in (2, 3):
            per = {0: 1, 1: 2, 2: 4, 3: 8, 4: 3, 5: 6}.get(self.quant, 0)
            rel = ((self.c.now() if at is None else at) - self.t0) * SR - 64
            if rel < 0:
                pos = 0 if -rel <= self.spb / per / 2 else None
            else:
                pos = quantise(int(rel % self.loop()), self.spb, per, self.loop())
            if pos is not None:
                self.events[self.active].append({"pos": pos, "pad": pad, "vel": vel})
        return {"ok": True}

    def seq_dump(self, link: Any, timeout_s: float = 5.0) -> list[dict]:
        out = []
        for s in range(self.scenes):
            for i, e in enumerate(sorted(self.events.get(s, []), key=lambda e: e["pos"])):
                out.append({"seq": s, "i": i, "pos": e["pos"], "pad": e["pad"], "vel": e["vel"]})
        return out

    def click(self, link: Any, on: int | None = None, bpm: int | None = None,
              bars: int | None = None) -> dict:
        return {"running": int(self.seq_state == 4), "bpm": self.bpm, "bars": self.bars[self.active],
                "pos": self.pos() if self.seq_state == 4 else 0, "loop": self.loop()}

    def song_text(self, link: Any, action: str | None = None) -> dict:
        if action:
            self.calls.append(f"SONG {action}")
            a = action.split()
            if a[0] == "CLEAR":
                self.song, self.song_mode = [], 0
            elif a[0] == "ADD":
                self.song.append((int(a[1]), int(a[2])))
            elif a[0] == "LOOP":
                self.song_loop = int(a[1])
            elif a[0] == "PLAY":
                self.song_mode, self.seq_state, self.seq_mode = 1, 4, 1
                self.active = self.song[0][0]
                self.t0 = self.c.now()
        return {"mode": self.song_mode, "loop": self.song_loop, "len": len(self.song), "pos": 0,
                "pass": 0, "scene": self.active, "blocks": [list(b) for b in self.song]}

    def asong(self, link: Any, action: str | None = None, timeout_s: float | None = None) -> dict:
        if action:
            self.calls.append(f"ASONG {action}")
            a = action.split(maxsplit=1)
            if a[0] == "OPEN":
                self.project = a[1]
            elif a[0] == "BOUNCE":
                sc, tr = (int(v) for v in a[1].split())
                self._bounce(sc, tr)
            elif a[0] == "DELETE":
                self.project = "Song 1"
        return {"open": 2, "bounce": self.bounce_state, "clips": list(self.clips),
                "file": self.bounce_file, "project": self.project, "under": 0}

    def _bounce(self, scene: int, track: int) -> None:
        self.clips[track] += 1
        self.bounce_state = 2
        self.bounce_file = f"HIL_S{scene + 1}_{self.clips[track]:02d}.wav"
        self.seq_state, self.seq_mode = 0, 0
        # a clean clip; a bounce in a USB-audio block is replaced by the take's own pads
        t = np.arange(2 * SR) / SR
        x = 0.05 * np.sin(2 * np.pi * 470.0 * t) * np.minimum(1.0, np.minimum(t, t[-1] - t) / 0.005)
        path = Path(self.tmp) / f"clip_{len(self.files)}.wav"
        write_wav(path, np.column_stack([x, x]), SR)
        self.files[f"/sdcard/SONGS/{self.project}/clips/{self.bounce_file}"] = path.read_bytes()
        self.last_bounce = (scene, track, self.c.now())

    # -- kits ---------------------------------------------------------------------
    def _swap(self, kit: int) -> None:
        self.swap_rows.append({"seq": len(self.swap_rows) + 1, "kit": kit, "ok": True,
                               "req_ms": 1000, "parse_ms": 30, "done_ms": 90,
                               "dropped_hits": 0, "late_hits": 0, "underruns": 0, "noblock": 0,
                               "deferred": 0, "forced": 0,
                               "pads": [30 + 4 * i for i in range(16)], "rt_gap_max_us": 900})
        self.kit = kit

    def kit_load(self, link: Any, kit: int, wait_s: float = 0) -> dict:
        self.calls.append(f"KIT_LOAD {kit}")
        self._swap(kit)
        return {}

    # -- the MIDI side -------------------------------------------------------------
    def on_sysex(self, frame: bytes) -> None:
        if self.hang_at is not None and self.c.now() >= self.hang_at:
            return
        f = list(frame)
        if f[2] == 0x1D:
            sub = f[3]
            if sub == 0x07:
                self.pad_press(None, f[4], f[5], at=self.c.now() + 0.008)
            elif sub == 0x0B:
                self._swap(f[4])
            elif sub == 0x0C:
                self._bounce(f[4], f[5])
            elif sub == 0x09:
                self.rx.put(bytes(frame))
            elif sub == 0x10:
                self.rx.put(bytes([0xF0, 0x7D, 0x1D, 0x10, 3, 1, 1, 3, 3, 80, 80, 0, 0, 0xF7]))
        elif f[2] == 0x1F:
            verb = bytes(f[4:-1]).decode()
            if f[3] == 0x01:
                reply = {"CP_CAPS": "CP_CAPS 1 control=1 files=0",
                         "KIT_STATUS": f"KITSTATUS: current={self.kit} loading=0 pending=-1",
                         "UI_STATE": "UI: app=Sampler"}.get(verb,
                                                            "ERR unsupported_companion_command")
                self.rx.put(bytes([0xF0, 0x7D, 0x1F, 0x03, *reply.encode(), 0xF7]))
            elif verb.startswith("KIT_LOAD "):
                self._swap(int(verb.split()[1]))
            elif verb.startswith("APP_START "):
                self.app = verb.split()[1]


class FakeIO:
    def __init__(self, world: World) -> None:
        self.w = world

    def open(self) -> None: ...
    def close(self) -> None: ...

    def send_sysex(self, frame: bytes) -> None:
        self.w.on_sysex(frame)

    def send_note(self, on: bool, note: int, vel: int = 100, channel: int = 0) -> None: ...
    def send_realtime(self, status: int) -> None: ...

    def drain(self) -> int:
        n = 0
        while not self.w.rx.empty():
            self.w.rx.get_nowait()
            n += 1
        return n

    def receive(self, timeout_s: float = 1.0) -> bytes | None:
        try:
            return self.w.rx.get(timeout=min(timeout_s, 0.02))
        except queue.Empty:
            return None


class FakeRec:
    """The duplex stream's recording: frames advance with the virtual clock."""

    def __init__(self, world: World, path: Path) -> None:
        self.w = world
        self.path = path
        self.sample_rate = SR
        self.started_at = world.c.now()
        self.error = None
        self.events: list[dict] = []
        self.info = {"pcm": "fake"}

    @property
    def frames_in(self) -> int:
        end = self.w.c.now() if self.w.hang_at is None else min(self.w.c.now(), self.w.hang_at)
        return int((end - self.started_at) * SR)

    def pause_out(self, seconds: float, mode: str = "drop") -> None:
        t = self.w.c.now()
        self.events += [{"kind": "out_mute", "t": t, "seconds": seconds},
                        {"kind": "out_unmute", "t": t + seconds}]

    def capture_points(self) -> list[tuple[int, float]]:
        return [(k, self.started_at + k / SR + 2e-4) for k in range(0, self.frames_in, SR // 50)]

    def health(self) -> dict:
        return {"ok": True, "missed_s": 0.0, "overruns": 0, "out_xruns": 0, "events": self.events}


class FakeSession:
    """CaptureSession over the fake: at stop() the take is rendered from what the
    scenario did -- the same model the analysis will hold it against."""

    def __init__(self, world: World, out: Path, ctx_dir: Path) -> None:
        self.w = world
        self.out = Path(out)
        self.dir = ctx_dir
        self.device = device(UsbMode.AUDIO)
        self._backend = type("B", (), {})()

    def start(self) -> dict:
        self.started_at = self.w.c.now()
        self._backend.recording = FakeRec(self.w, self.out)
        return {"route_seen": {"mic_src": 3}, "preset": "sampler"}

    def stop(self) -> dict:
        rec = self._backend.recording
        name = self.out.parent.name
        blk = json.loads((self.out.parent / "block.json").read_text())
        acts = [a for a in load_actions(self.dir / "actions.jsonl") if a.block == name]
        n = max(SR // 2, rec.frames_in)
        rb = blk["readback_before"]
        rows = [r for r in self.w.swap_rows if r["seq"] > blk.get("swap_seq0", -1)]
        b = BlockInput(name=name, sr=SR, n=n, frame_of=lambda t: (t - rec.started_at) * SR,
                       actions=acts, kits=S.jam_kits(), kit0=blk["kit0"],
                       seq_before=rb.get("seq") or {}, dump_before=rb.get("seq_dump") or [],
                       dump_after=self.w.seq_dump(None), song=rb.get("song") or {},
                       click=rb.get("click") or {}, swap_rows=rows, desk=blk.get("desk") or {},
                       stream_events=rec.events, t_stream_start=rec.started_at,
                       t_stream_stop=self.w.c.now(), fx_first_load=blk.get("fx_first_load"))
        exp = build_expectation(b, Calibration())
        x = render_expected(exp, pilot=S.PilotSpec(), delay=1500,
                            faults=self.w.faults.get(name, []), noise=2e-5)
        write_wav(self.out, x, SR)
        if getattr(self.w, "last_bounce", None):
            sc, tr, t_b = self.w.last_bounce
            pads = render_expected(exp, pilot=None, n=n)
            a = int((t_b - rec.started_at) * SR)
            if 0 <= a < n:
                key = next(k for k in self.w.files if k.endswith(self.w.bounce_file))
                path = self.dir / "fake_bounce.wav"
                write_wav(path, pads[a:a + 4 * SR], SR)
                self.w.files[key] = path.read_bytes()
            self.w.last_bounce = None
        return {"wav": str(self.out), "seconds": n / SR}


class FakeConsole:
    def __init__(self, world: World) -> None:
        self.w = world
        self.resets = 0

    @property
    def seq(self) -> int:
        if self.w.hang_at is not None and self.w.c.now() >= self.w.hang_at:
            return int(self.w.hang_at)
        return int(self.w.c.now())

    def start_tsv(self, path: Path) -> None:
        Path(path).write_text("")

    def read(self, since_seq: int = 0, match: str | None = None, limit: int = 0) -> Any:
        lines = ["E (1) task_wdt: Task watchdog got triggered"] if self.w.hang_at else []
        return type("R", (), {"lines": list(enumerate(lines))})()

    def wait_boot(self, timeout_s: float = 0, settle_s: float = 0) -> Any:
        return type("B", (), {"complete": True})()

    def reset(self) -> None:
        self.resets += 1
        self.w.calls.append("console reset")
        self.w.hang_at = None
        self.w.mode = UsbMode.DEFAULT

    def events(self, since_seq: int = 0) -> list:
        return []

    def close(self) -> None: ...


class FakeCtx(base.Context):
    world: World

    def open_console(self, reset: bool = False) -> Any:
        self.con = FakeConsole(self.world)
        return self.con

    def open_cdc(self) -> Any:
        assert self.world.mode is UsbMode.DEFAULT, "CDC opened in the audio profile"
        return type("Link", (), {"close": lambda self: None})()


def install(mp: pytest.MonkeyPatch, w: World, tmp_path: Path) -> None:
    c = w.c
    w.tmp = tmp_path
    mp.setenv("CROSSPAD_HIL_PINS", str(tmp_path / "pins.json"))
    mp.setattr(live_jam, "_now", c.now)
    mp.setattr(live_jam, "_sleep", c.sleep)
    for name in ("seq", "pad_press", "seq_dump", "click", "song_text", "asong", "kit_load"):
        mp.setattr(verbs, name, getattr(w, name))
    mp.setattr(verbs, "pad_release", lambda link, p: {"ok": True})
    mp.setattr(verbs, "kit_status", lambda link: {"current": w.kit, "name": next(
        n for n, i in KITS.items() if i == w.kit), "loading": False, "pending": -1})
    mp.setattr(verbs, "kit_list", lambda link: {"kits": [{"id": i, "name": n}
                                                         for n, i in KITS.items()],
                                               "current": w.kit})
    mp.setattr(verbs, "kit_swap_log", lambda link, log=None: list(w.swap_rows[-8:]))
    mp.setattr(verbs, "smpl_counters", lambda link: dict(w.counters))
    mp.setattr(verbs, "pad_stats", lambda link, reset=False: dict(w.pads))
    mp.setattr(verbs, "pitched_status", lambda link: {"active": 0, "shed": 0, "steals": 0})
    mp.setattr(verbs, "smpl_pool", lambda link: {"esc": 0, "fixed": 0, "leak": 0})
    mp.setattr(verbs, "smpl_engine", lambda link: {"engine": "native", "underrun": 0,
                                                   "norings": 0, "drop": 0})
    mp.setattr(verbs, "smpl_last", lambda link: {"late_max_ms": 0})
    mp.setattr(verbs, "reset_count", lambda link: w.resets)
    mp.setattr(verbs, "clock_status", lambda link, reset=False: {"src": "int", "bpm": w.bpm})
    mp.setattr(verbs, "app_sha", lambda link: {"sha256": "f" * 64})
    mp.setattr(verbs, "stm_fw_info", lambda link: {"running": "1.0"})
    mp.setattr(verbs, "usb_guard", lambda link, on=None: True)
    mp.setattr(verbs, "mem", lambda link: {"int_free": 1})
    mp.setattr(verbs, "worker", lambda link: [])
    mp.setattr(verbs, "arena", lambda link: {"owner": "marcel"})
    mp.setattr(verbs, "app_list", lambda link: {"apps": ["Sampler", "Mixer"], "running": w.app})
    mp.setattr(verbs, "app_stop", lambda link: setattr(w, "app", None))
    mp.setattr(verbs, "fs_delete", lambda link, path: w.files.pop(path, None) or {"ok": True})

    def mix_preset(link: Any, action: str | None = None) -> dict:
        if action:
            w.calls.append(f"MIX_PRESET {action}")
            a = action.split(maxsplit=2)
            if a[0] == "SAVE":
                w.presets[int(a[1])] = ("s", a[2])
                w.files[f"/sdcard/crosspad/mixer/preset_{a[1]}.json"] = json.dumps(
                    desk_json(a[2])).encode()
        for k in (4, 5):
            body = w.files.get(f"/sdcard/crosspad/mixer/preset_{k}.json")
            w.presets[k] = ("s", json.loads(body)["name"]) if body else ("e", "Empty")
        kinds = {"s": "saved", "f": "factory", "e": "empty"}
        return {"ok": True, "undo": False, "dir": "/sdcard/crosspad/mixer",
                "slots": {k: {"kind": kinds[v[0]], "name": v[1]} for k, v in w.presets.items()},
                "line": "MIXPRE: ok=1"}

    mp.setattr(verbs, "mix_preset", mix_preset)
    mp.setattr(verbs, "mix_lvl", lambda link: {"channels": desk_json("x")["desk"]["channels"],
                                               "outputs": [{"out": o, "vol": 1.0}
                                                           for o in range(3)]})

    def stool(link: Any, action: str | None = None, timeout_s: float = 5.0) -> dict:
        w.calls.append(f"STOOL {action}")
        if action == "REC":
            w.stool.update(rec=1, file="/sdcard/REC/rec_0001.wav")
        elif action == "STOP":
            w.stool["rec"] = 0
        return {"rec": w.stool["rec"], "file": w.stool["file"], "frames": 0, "sr": SR}

    mp.setattr(verbs, "stool", stool)
    mp.setattr(_audio, "ensure_kits", lambda ctx, link, names, b, **kw: (link, []))
    mp.setattr(_audio, "resolve_kit", lambda link, kit_name, **kw: (KITS[kit_name], kit_name,
                                                                    w.kit))
    mp.setattr(_audio, "load_kit_waiting", lambda ctx, link, kid, **kw: w.kit_load(link, kid))

    def start_app(ctx: Any, link: Any, name: str, **kw: Any) -> None:
        w.calls.append(f"APP_START {name}")
        w.app = name

    mp.setattr(_audio, "start_app", start_app)
    mp.setattr(verbs, "app_start", lambda link, name, wait_s=3.0: start_app(None, link, name))

    def set_mode(dev: Device, mode: UsbMode, **kw: Any) -> Device:
        w.calls.append(f"mode {mode.value}")
        if mode is UsbMode.DEFAULT and w.hang_at is not None and w.c.now() >= w.hang_at:
            raise HilError("TIMEOUT", "did not re-enumerate")
        w.mode = mode
        return device(mode)

    mp.setattr(live_jam, "set_mode", set_mode)
    mp.setattr(live_jam.LiveJamScenario, "_refresh", staticmethod(lambda ctx: None))

    @contextmanager
    def hold(ctx: Any, data: Any = None, **kw: Any):
        yield {"paused": False}

    mp.setattr(card_hold, "for_run", hold)
    mp.setattr(live_jam, "open_capture", lambda dev, **kw: FakeSession(w, kw["out"],
                                                                       tmp_path / "run"))
    mp.setattr(live_jam, "open_midi", lambda dev: FakeIO(w))
    mp.setattr(live_jam, "query_route", lambda io, timeout_s=0.6: {
        "mic_src": 3, "adc": [1, 1], "out": [3, 3], "vol": [80, 80], "mute": [0, 0]})

    def pull(ctx: Any, remote: str, local: Path, **kw: Any) -> dict:
        w.calls.append(f"pull {remote}")
        if remote not in w.files:
            raise HilError("FLASH_FAILED", "device reported FS_ERR stat_2",   # ENOENT
                           reply="FS_ERR stat_2")
        Path(local).parent.mkdir(parents=True, exist_ok=True)
        Path(local).write_bytes(w.files[remote])
        return {}

    def push(ctx: Any, local: Path, remote: str) -> dict:
        w.calls.append(f"push {remote}")
        w.files[remote] = Path(local).read_bytes()
        return {}

    mp.setattr(live_jam, "pull_file", pull)
    mp.setattr(live_jam, "push_file", push)


def run_jam(tmp_path: Path, mp: pytest.MonkeyPatch, w: World | None = None,
            **kw: Any) -> tuple[base.Report, World]:
    w = w or World(VClock())
    install(mp, w, tmp_path)
    ctx = FakeCtx(device=device(UsbMode.DEFAULT), workdir=tmp_path / "run",
                  cancelled=threading.Event(), log=lambda s: None)
    ctx.world = w
    params = Params(minutes=kw.pop("minutes", 0.6), probes=False, monitor=False,
                    recorder=kw.pop("recorder", False), snippets=2, allow_partial=True, **kw)
    return live_jam.SCENARIO.run(ctx, params, base.Progress()), w


# --------------------------------------------------------------------------- the tests

def test_registered_and_dry_run(tmp_path: Path) -> None:
    assert get("live_jam") is live_jam.SCENARIO
    ctx = base.Context(device=device(UsbMode.DEFAULT), workdir=tmp_path,
                       cancelled=threading.Event(), log=lambda s: None)
    rep = live_jam.SCENARIO.run(ctx, Params(dry_run=True), base.Progress())
    assert rep.passed and "[B1] UAC" in rep.summary and "B2.clock" in rep.summary


def test_a_clean_jam_passes_and_puts_everything_back(tmp_path: Path, monkeypatch) -> None:
    rep, w = run_jam(tmp_path, monkeypatch)
    assert rep.exit_code == 0, rep.summary
    assert "ZERO-GLITCH PASS" in rep.summary
    c = w.calls
    # the board was read and journalled before anything changed, the desks built from it
    assert c.index("MIX_PRESET SAVE 6 HILHOME") < c.index("push /sdcard/crosspad/mixer/preset_4.json")
    assert "ASONG OPEN hil_live_jam" in c
    # S1 recorded with count-in into the jam's own scene (2 user scenes before it)
    assert "SEQ BARS 2 2" in c and "SEQ REC" in c
    assert "SONG ADD 2 2" in c and "SONG ADD 3 1" in c and "SEQ DOUBLE" in c
    # a profile switch per block, back to default after each
    assert c.count("mode audio") == 4
    # put back: the desk, the preset slots that were not there removed, the project,
    # sequences.json, a restart
    assert "MIX_PRESET LOAD 6" in c
    assert "/sdcard/crosspad/mixer/preset_4.json" not in w.files
    assert w.files["/sdcard/crosspad/sequences.json"] == b'{"scenes": "the user\'s"}'
    assert json.loads(w.files["/sdcard/crosspad/mixer/preset_6.json"])["name"] == "Mine"
    assert w.project == "My Song" and w.kit == 7 and w.scenes == 2
    assert "console reset" in c                       # the restart after sequences.json
    rd = tmp_path / "run"
    doc = json.loads((rd / "jam_report.json").read_text())
    assert doc["verdict"]["status"] == "ZERO-GLITCH PASS"
    assert {b["name"] for b in doc["blocks"]} == {"B1", "B1b", "B2", "B3"}
    recs = json.loads((rd / "recs.json").read_text())
    assert [r["window"] for r in recs] == ["W0", "W1", "W2"] and all(
        not r["missing"] and not r["extra"] for r in recs)
    assert json.loads((rd / "bounces.json").read_text())[0]["state"] == 2
    assert not json.loads((tmp_path / "pins.json").read_text() or "{}").get("dev_jam1")


def test_a_hole_in_a_take_is_a_fail_with_its_moment_and_step(tmp_path: Path,
                                                             monkeypatch) -> None:
    w = World(VClock())
    w.faults["B2"] = [Fault("zero_block", frame=int(4.0 * SR), ms=20)]
    rep, _w = run_jam(tmp_path, monkeypatch, w)
    assert rep.exit_code == 1 and "FAIL" in rep.summary
    doc = json.loads((tmp_path / "run" / "jam_report.json").read_text())
    g = [g for g in doc["glitches"] if g["status"] == "glitch"]
    assert len(g) == 1 and g[0]["block"] == "B2" and "PILOT_GAP" in g[0]["classes"]
    assert abs(g[0]["axes"]["wav_s"] - 4.0) < 0.005
    assert g[0]["step"].startswith("B2.")


def test_a_board_that_hangs_in_a_block_is_recovered_and_the_set_goes_on(
        tmp_path: Path, monkeypatch) -> None:
    w = World(VClock())
    real = World.on_sysex

    def arm(self, frame):  # noqa: ANN001
        if self.hang_at is None and getattr(self, "_armed", False) is False \
                and self.mode is UsbMode.AUDIO and list(frame)[3] == 0x0B:
            self._armed = True
            self.hang_at = self.c.now() + 0.5        # the first swap of the set: dead
        return real(self, frame)

    monkeypatch.setattr(World, "on_sysex", arm)
    rep, w = run_jam(tmp_path, monkeypatch, w)
    assert rep.exit_code == 1
    doc = json.loads((tmp_path / "run" / "jam_report.json").read_text())
    hung = [c for c in doc["counters"] if c["counter"] == "hung"]
    assert hung and "capture stopped: no frames for 2 s" in hung[0]["note"]
    assert "console reset" in w.calls
    assert {b["name"] for b in doc["blocks"]} >= {"B1", "B2", "B3"}   # the rest still ran
    assert w.kit == 7 and w.project == "My Song"


def test_a_failure_in_prep_still_puts_the_board_back(tmp_path: Path, monkeypatch) -> None:
    w = World(VClock())
    install(monkeypatch, w, tmp_path)

    def no_open(link: Any, action: str | None = None, timeout_s: float | None = None) -> dict:
        if action and action.startswith("OPEN"):
            raise HilError("ENV", "ASONG OPEN: no card")
        return w.asong(link, action, timeout_s)

    monkeypatch.setattr(verbs, "asong", no_open)
    ctx = FakeCtx(device=device(UsbMode.DEFAULT), workdir=tmp_path / "run",
                  cancelled=threading.Event(), log=lambda s: None)
    ctx.world = w
    rep = live_jam.SCENARIO.run(ctx, Params(minutes=0.3, probes=False, monitor=False,
                                            recorder=False), base.Progress())
    assert rep.exit_code == 2 and "no card" in rep.summary
    assert "MIX_PRESET LOAD 6" in w.calls and w.kit == 7
    assert json.loads(w.files["/sdcard/crosspad/mixer/preset_6.json"])["name"] == "Mine"


def test_a_killed_run_s_journal_puts_the_board_back_on_the_next_start(
        tmp_path: Path, monkeypatch) -> None:
    from crosspad_hil import pins
    from crosspad_hil.jam import restore as RS

    w = World(VClock())
    install(monkeypatch, w, tmp_path)
    keep = tmp_path / "keep.json"
    keep.write_bytes(b'{"scenes": "before the killed run"}')
    home = RS.HomeState(seq={"scenes": 2, "bpm": 100.0, "quant": 2, "countin": 1},
                        kit={"id": 7, "name": "User Kit"}, scenes0=2, project="My Song",
                        mix_dir="/sdcard/crosspad/mixer",
                        files={"/sdcard/crosspad/sequences.json": str(keep),
                               "/sdcard/crosspad/mixer/preset_4.json": None})
    pins.record("dev_jam1", RS.PIN_KEY, home.to_dict())
    w.files["/sdcard/crosspad/mixer/preset_4.json"] = json.dumps(desk_json("JAM-DUCK")).encode()
    w.scenes = 4
    w.project = "hil_live_jam"
    rep, w = run_jam(tmp_path, monkeypatch, w, minutes=0.3)
    back = rep.data["restored_from_journal"]
    assert back["/sdcard/crosspad/sequences.json"] == "restored"
    assert back["/sdcard/crosspad/mixer/preset_4.json"].startswith("removed")
    first = w.calls.index("SEQ DEL 3")
    assert w.calls[first + 1] == "SEQ DEL 2"           # the dead run's scenes, newest first
    assert first < w.calls.index("SEQ ADD")            # before this run touched anything


def test_analyze_only_reads_a_run_without_a_board(tmp_path: Path, monkeypatch) -> None:
    rep, _w = run_jam(tmp_path, monkeypatch)
    ctx = base.Context(device=device(UsbMode.DEFAULT), workdir=tmp_path / "again",
                       cancelled=threading.Event(), log=lambda s: None)
    again = live_jam.SCENARIO.run(ctx, Params(analyze_only=str(tmp_path / "run"),
                                              allow_partial=True), base.Progress())
    assert again.exit_code == rep.exit_code == 0


def test_the_wav_window_of_the_recorder_comes_back_as_a_wav(tmp_path: Path) -> None:
    src = tmp_path / "rec.wav"
    x = (np.sin(np.arange(SR * 3) / 10.0) * 8000).astype("<i2")
    with wave.open(str(src), "wb") as fh:
        fh.setnchannels(1)
        fh.setsampwidth(2)
        fh.setframerate(SR)
        fh.writeframes(x.tobytes())
    head = src.read_bytes()[:4096]
    ch, sr, align, data = live_jam.wav_layout(head)
    assert (ch, sr, align, data) == (1, SR, 2, 44)
