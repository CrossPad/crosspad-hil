"""Scenario `live_set`, without a board: the pure parts on synthesized audio, and
the whole set against a fake world (profiles, SEQ, bounces, kit swaps, counters)."""
from __future__ import annotations

import math
import struct
import threading
import time
import wave
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from crosspad_hil import fixtures, host_gain, pins, verbs
from crosspad_hil.devices import AudioCardInfo, Device, Ports, SerialPortInfo, UsbMode
from crosspad_hil.scenarios import _audio, base, get, live_set
from crosspad_hil.scenarios.live_set import (
    KIT_A,
    KIT_B,
    KIT_P,
    Params,
    expected_in,
    late_from_timestamps,
    parse_spk,
    pattern,
    phase_problems,
    score_take,
    swap_totals,
)

SR = 44_100


def write_wav(path: Path, seconds: float, tones: list[tuple[float, float, float]] = (),
              *, hole: tuple[float, float] | None = None,
              click_at: float | None = None) -> Path:
    """Mono-in-stereo 16-bit; `tones` = (Hz, start s, end s)."""
    n = int(seconds * SR)
    frames = bytearray()
    for i in range(n):
        t = i / SR
        v = sum(0.1 * math.sin(2 * math.pi * f * t) for f, a, b in tones if a <= t < b)
        if hole and hole[0] <= t < hole[1]:
            v = 0.0
        if click_at is not None and abs(t - click_at) < 0.5 / SR:
            v = 0.9
        s = int(max(-1.0, min(1.0, v)) * 32767)
        frames += struct.pack("<hh", s, s)
    with wave.open(str(path), "wb") as w:
        w.setnchannels(2)
        w.setsampwidth(2)
        w.setframerate(SR)
        w.writeframes(bytes(frames))
    return path


# ------------------------------------------------------------------ pure parts

def test_registered() -> None:
    assert get("live_set") is live_set.SCENARIO
    assert Params().phases == ["one", "four", "many", "pitched"]


def test_spk_lines_old_and_new_format() -> None:
    old = ["I (1334737) usb_audio_bridge: SPK #1: len=176 taken=44 ring=44 under=0 refused=0 "
           "cap_ring=334 cap_drop=0",
           "I (1336737) usb_audio_bridge: SPK #2: len=176 taken=44 ring=40 under=2 refused=1 "
           "cap_ring=330 cap_drop=0"]
    assert parse_spk(old) == {"lines": 2, "under": 2, "refused": 1, "cap_drop": 0,
                              "cap_drop_total": 0}
    new = [old[0] + " ring_min=12", old[1] + " ring_min=3"]
    assert parse_spk(new)["ring_min"] == 3
    assert parse_spk(["nothing here"]) == {"lines": 0}


def test_patterns() -> None:
    assert {tuple(pattern("one", t, chord_every=0)) for t in range(40)} == {(0,)}
    four = [pattern("four", t, chord_every=0)[0] for t in range(64)]
    assert set(four[:16]) == {0, 1, 2, 3} and set(four[16:32]) == {4, 5, 6, 7}
    many = [pattern("many", t, chord_every=8) for t in range(16)]
    assert len(many[7]) == 16 and len(many[15]) == 8 and many[0] == [0]
    assert live_set.bounce_pads("one") == [0, 0, 0, 0]
    assert live_set.phase_kits("pitched") == [KIT_P, KIT_A]


def test_expected_follows_the_kit_in_force() -> None:
    hits = [(1.0, 3), (6.0, 3)]
    kits = [(-1e9, KIT_A), (5.0, KIT_B)]
    assert expected_in(hits, kits, 0.0, 2.0) == {3: [round(fixtures.freq_a(3), 2)]}
    # after the swap the old voice may still sound: both kits' tones count
    assert expected_in(hits, kits, 5.0, 10.0) == {
        3: sorted([round(fixtures.freq_a(3), 2), round(fixtures.freq_b(3), 2)])}


def test_score_take_audible_clean(tmp_path: Path) -> None:
    f0, f1 = fixtures.freq_a(0), fixtures.freq_a(5)
    wav = write_wav(tmp_path / "t.wav", 4.0, [(f0, 0, 4), (f1, 0, 4), (441.0, 0, 4)])
    hits = [(0.5, 0), (1.0, 5), (2.5, 0)]
    take = score_take(wav, slice_s=2.0,
                      expect=lambda a, b: expected_in(hits, [(-1e9, KIT_A)], a, b))
    assert take["slices"] == 2 and take["inaudible"] == 0
    assert take["clicks"] == 0 and take["dropouts"] == 0 and take["silent_slices"] == 0


def test_score_take_finds_an_inaudible_pad_a_hole_and_a_click(tmp_path: Path) -> None:
    f0 = fixtures.freq_a(0)
    wav = write_wav(tmp_path / "t.wav", 4.0, [(f0, 0, 4)], hole=(1.0, 1.01), click_at=3.0)
    hits = [(0.5, 0), (0.6, 7)]               # pad 7 was struck but never sounds
    take = score_take(wav, slice_s=2.0,
                      expect=lambda a, b: expected_in(hits, [(-1e9, KIT_A)], a, b))
    assert take["inaudible"] == 1 and take["worst"][0]["inaudible_pads"] == [7]
    assert take["dropouts"] == 1
    assert take["clicks"] >= 1


def test_swap_totals_and_problems() -> None:
    rows = [{"seq": 5, "ok": True, "underruns": 0, "noblock": 2, "dropped_hits": 0,
             "late_hits": 1, "rt_gap_max_us": 1651}]
    totals = swap_totals(rows)
    assert totals == {"swaps": 1, "underruns": 0, "noblock": 2, "dropped_hits": 0,
                      "late_hits": 1, "late_hits_max": 1, "rt_gap_max_us": 1651, "failed": 0}
    probs = phase_problems({"spk": {"under": 3}, "swaps": totals, "engine": {"underrun": 1},
                            "pads": {"usb_drop": 2}, "take": {"inaudible": 4},
                            "bounce": {"ok": False, "why": "the bounced file is silent"},
                            "seq_stopped": True, "resets": 1})
    assert probs == [
        "usb_audio_bridge under=3", "engine underrun +1", "kit swap noblock=2",
        "1 late hit(s) the host could not time: no timing",
        "rt_gap_max_us=1651 > one block (1451) with an underrun or a click",
        "pad path usb_drop +2", "capture: 4 pad hit(s) inaudible on USB IN",
        "bounce: the bounced file is silent", "the scene was not playing after the live block",
        "1 reset(s)"]
    assert phase_problems({"spk": {"under": 0}, "swaps": swap_totals([])}) == []


def test_rt_gap_up_to_two_blocks_passes_on_a_clean_phase() -> None:
    def probs(gap: int, **ph) -> list[str]:
        rows = [{"seq": 1, "rt_gap_max_us": gap}]
        return phase_problems({"swaps": swap_totals(rows), "late": {"late_max_ms": 0.0}, **ph})

    assert probs(2902) == []
    assert probs(2903) == ["rt_gap_max_us=2903 > two blocks (2902)"]
    assert probs(2000, take={"clicks": 1}) == [
        "rt_gap_max_us=2000 > one block (1451) with an underrun or a click", "capture: 1 click(s)"]
    assert probs(2000, engine={"underrun": 2}) == [
        "engine underrun +2", "rt_gap_max_us=2000 > one block (1451) with an underrun or a click"]


def test_late_hits_one_per_swap_and_never_later_than_50_ms() -> None:
    def probs(late_hits: list[int], late_ms: float | None) -> list[str]:
        rows = [{"seq": i, "late_hits": n} for i, n in enumerate(late_hits)]
        return phase_problems({"swaps": swap_totals(rows),
                               "late": {"late_max_ms": late_ms, "why": "unpaired"}})

    assert probs([1, 1, 0], 50.0) == []
    assert probs([2], 10.0) == ["kit swap late_hits=2 in one swap (> 1)"]
    assert probs([1], 51.0) == ["a held hit sounded 51 ms late (> 50 ms)"]
    assert probs([1], None) == ["1 late hit(s) the host could not time: unpaired"]


def test_late_from_timestamps_times_hits_inside_a_pad_s_reload_window() -> None:
    """Pad 2 reloads 30..70 ms after the request, pad 5 70..120 ms (sequentially)."""
    pads = [-1] * 16
    pads[2], pads[5] = 70, 120
    rows = [{"seq": 9, "parse_ms": 30, "late_hits": 1, "pads": pads},
            {"seq": 10, "parse_ms": 30, "late_hits": 0, "pads": pads}]
    t0, t1 = 100.0, 200.0
    hits = [(t0 + 0.040, 2),     # inside pad 2's window: held 30 ms
            (t0 + 0.100, 2),     # pad 2 is back: on time
            (t0 + 0.065, 5),     # before pad 5's reload began: the old sample, on time
            (t1 + 0.040, 2)]     # the second swap logs no late hit: not timed
    got = late_from_timestamps(rows, [t0, t1], hits)
    assert got["late_max_ms"] == 30.0
    assert [s["late_max_ms"] for s in got["per_swap"]] == [30.0, 0.0]
    assert got["per_swap"][0]["candidates"] == 1
    unpaired = late_from_timestamps(rows, [t0], hits)
    assert unpaired["late_max_ms"] is None and "cannot be paired" in unpaired["why"]


def test_a_pitched_chord_expects_only_the_notes_that_keep_a_voice() -> None:
    chord = [(5.0, pad) for pad in range(16)]
    on_p = expected_in(chord, [(0.0, KIT_P)], 0.0, 10.0)
    assert sorted(on_p) == list(range(6, 16))          # 0..5 were stolen
    on_a = expected_in(chord, [(0.0, KIT_A)], 0.0, 10.0)
    assert sorted(on_a) == list(range(16))             # the sine kits have no such limit
    eight = [(5.0, pad) for pad in range(0, 16, 2)]
    assert sorted(expected_in(eight, [(0.0, KIT_P)], 0.0, 10.0)) == list(range(0, 16, 2))


# ------------------------------------------------------------------ the whole set, faked

CDC = SerialPortInfo(path="/dev/ttyACM0", vid=0x303A, pid=0x3456, serial="CP-1",
                     product="Crosspad", location="1-2.1")
UAC2 = AudioCardInfo(name="Crosspad Audio", sounddevice_index=3, alsa_id="Audio")


def device(mode: UsbMode) -> Device:
    ports = Ports(uac2=UAC2) if mode is UsbMode.AUDIO else Ports(cdc=CDC)
    return Device(id="dev_ab12", serial="CP-1", usb_mode=mode, ports=ports)


class World:
    """The board as the scenario sees it, every call recorded."""

    def __init__(self) -> None:
        self.calls: list[str] = []
        self.mode = UsbMode.DEFAULT
        self.kit = 7
        self.scenes = 2
        self.seq_state = 0
        self.bounce = 0
        self.clips = [0, 0, 0, 0]
        self.swap_rows: list[dict] = []
        self.spk_lines: list[str] = []
        self.bounce_silent = False
        self.seq_dies_in_uac = False
        self.capture_health: dict = {"missed_s": 0.0, "max_gap_ms": 12.0, "overruns": 0,
                                      "exclusive": True}
        self.timeouts: set = set()
        self.running: str | None = None
        self.stuck_in_audio = 0          # set_mode(default) refusals still to come
        self.unrecoverable = False
        self.console_resets = 0
        self.boot_lines = ["I (812) bsp_sdmmc: SDMMC host prepared: slot=0 max_freq=40000kHz",
                           "I (815) bsp_sdmmc: SDMMC SLOT_0 pin map: CLK=14 CMD=15 D0=2 D1=4 "
                           "D2=12 D3=13 width=4",
                           "I (2905) bsp_sdmmc: SD Card mounted successfully at /sdcard"]

    # verbs ---------------------------------------------------------------
    def seq(self, link: Any, action: str | None = None, timeout_s: float | None = None) -> dict:
        self.timeouts.add(timeout_s)
        if action:
            self.calls.append(f"SEQ {action}")
            if action == "ADD":
                self.scenes += 1
            elif action == "PLAY":
                self.seq_state = 4
            elif action == "STOP":
                self.seq_state = 0
            elif action.startswith("DEL"):
                self.scenes -= 1
        return {"scenes": self.scenes, "countin": 1, "state": self.seq_state}

    def asong(self, link: Any, action: str | None = None,
              timeout_s: float | None = None) -> dict:
        if action:
            self.calls.append(f"ASONG {action}")
            if action.startswith("BOUNCE"):
                self.seq_state = 0                 # a bounce stops the sequencer
                self.clips[int(action.split()[2])] += 1
                self.bounce = 2
        return {"open": 2, "bounce": self.bounce, "clips": list(self.clips),
                "file": "HIL_S3_01.wav", "project": "hil_live_set", "err": None}

    def kit_load(self, link: Any, kit: int, wait_s: float = 0) -> dict:
        self.calls.append(f"KIT_LOAD {kit}")
        self.kit = kit
        return {}

    def install(self, mp: pytest.MonkeyPatch) -> None:
        mp.setattr(verbs, "seq", self.seq)
        mp.setattr(verbs, "asong", self.asong)
        mp.setattr(verbs, "kit_load", self.kit_load)
        mp.setattr(verbs, "kit_status", lambda link: {"current": self.kit})
        mp.setattr(verbs, "kit_swap_log", lambda link, log=None: list(self.swap_rows))
        mp.setattr(verbs, "smpl_counters", lambda link: {"underrun": 0, "noblock": 0})
        mp.setattr(verbs, "pitched_status", lambda link: {"active": True, "voices_max": 10,
                                                          "steals": 6, "rt_max_us": 900})
        mp.setattr(verbs, "pad_stats", lambda link: {"press": 0, "played": 0, "usb_drop": 0})
        mp.setattr(verbs, "pad_press", lambda link, p, v=100: None)
        mp.setattr(verbs, "pad_release", lambda link, p: None)
        mp.setattr(verbs, "app_list", lambda link: {"apps": ["Sampler"], "running": self.running})

        def app_start(link: Any, name: str, wait_s: float = 3.0) -> dict:
            self.calls.append(f"APP_START {name} {wait_s:g}")
            self.running = name
            return {"running": name}

        mp.setattr(verbs, "app_start", app_start)
        def app_stop(link: Any) -> None:
            self.calls.append("APP_STOP")
            self.running = None

        mp.setattr(verbs, "app_stop", app_stop)
        mp.setattr(verbs, "reset_count", lambda link: 0)
        mp.setattr(verbs, "mem", lambda link: {"int_free": 120000})
        mp.setattr(_audio, "ensure_kits", lambda ctx, link, names, b, **kw: (link, []))
        ids = {KIT_A: 3, KIT_B: 4, KIT_P: 5}
        mp.setattr(_audio, "resolve_kit", lambda link, kit_name, **kw: (ids[kit_name], kit_name, 7))
        mp.setattr(pins, "restore_pending", lambda *a, **kw: {})

        def set_mode(dev: Device, mode: UsbMode, **kw: Any) -> Device:
            self.calls.append(f"mode {mode.value}")
            if mode is UsbMode.DEFAULT and (self.unrecoverable or self.stuck_in_audio):
                self.stuck_in_audio = max(0, self.stuck_in_audio - 1)
                raise live_set.HilError("TIMEOUT", "dev_ab12 did not re-enumerate in mode "
                                                   "'default' within 45 s")
            if mode is UsbMode.DEFAULT and self.seq_dies_in_uac:
                self.seq_state = 0
            self.mode = mode
            return device(mode)

        mp.setattr(live_set, "set_mode", set_mode)
        mp.setattr(live_set.LiveSetScenario, "_refresh", staticmethod(lambda ctx: None))
        mp.setattr(host_gain, "apply", lambda card: (host_gain.HostGain(card), host_gain.HostGain(card)))
        world = self

        class Capture:
            started_at = time.monotonic()

            def __init__(self, dev: Device, **kw: Any) -> None:
                world.calls.append(f"capture {kw['preset']}")
                self.out = kw["out"]

            def start(self) -> dict:
                write_wav(Path(self.out), 0.2)
                return {"route_seen": {"mic_src": 3}}

            def stop(self) -> dict:
                return {"wav": str(self.out), "silent": False, "seconds": 0.2}

            def health(self) -> dict:
                return dict(world.capture_health)

        class Transport:
            class _io:
                @staticmethod
                def send_sysex(frame: bytes) -> None:
                    world.calls.append(f"sysex kit {frame[4]}")
                    world.swap_rows.append({"seq": len(world.swap_rows) + 10, "ok": True,
                                            "underruns": 0, "noblock": 0, "dropped_hits": 0,
                                            "late_hits": 0, "rt_gap_max_us": 900})

            def press(self, pad: int, vel: int) -> None: ...
            def release(self, pad: int) -> None: ...
            def close(self) -> None: ...

        class Stream:
            argv = ["pw-play"]
            def alive(self) -> bool: return True
            def stop(self) -> str:
                world.calls.append("stream stop")
                return ""

        mp.setattr(live_set, "open_capture", Capture)
        mp.setattr(live_set, "open_pad_transport", lambda dev, kind: Transport())
        mp.setattr(live_set, "start_host_stream", lambda wav, target, log: (
            world.calls.append("stream start"), Stream())[1])

        def pull(ctx: Any, remote: str, local: Path) -> dict:
            world.calls.append(f"pull {remote}")
            write_wav(local, 0.5, [] if world.bounce_silent else [(440.0, 0, 0.5)])
            return {}

        mp.setattr(live_set, "pull_file", pull)
        mp.setattr(live_set, "score_take", self.score_take)
        mp.setattr(live_set, "STREAM_SETTLE_S", 0.0)
        mp.setattr(live_set, "POLL_S", 0.0)

    def score_take(self, path: Path, *, slice_s: float = 10.0, expect: Any = None) -> dict:
        silent = self.bounce_silent and "bounce" in path.name
        return {"slices": 1, "clicks": 0, "dropouts": 0, "inaudible": 0,
                "silent_slices": 1 if silent else 0, "peak_dbfs": -20.0, "worst": []}


class Console:
    def __init__(self, world: World) -> None:
        self.world = world
        self.seq = 0

    def read(self, since_seq: int = 0, match: str | None = None, limit: int = 0) -> Any:
        src = self.world.boot_lines if match == "bsp_sdmmc" else self.world.spk_lines
        return type("R", (), {"lines": [(i, s) for i, s in enumerate(src)]})()

    def wait_boot(self, timeout_s: float = 0, settle_s: float = 0) -> Any:
        return type("B", (), {"complete": True})()

    def reset(self) -> None:
        self.world.calls.append("console reset")
        self.world.console_resets += 1
        if not self.world.unrecoverable:
            self.world.stuck_in_audio = 0

    def events(self, since_seq: int = 0) -> list:
        return []

    def close(self) -> None: ...


class FakeCtx(base.Context):
    world: World

    def open_console(self, reset: bool = False) -> Any:
        return Console(self.world)

    def open_cdc(self) -> Any:
        assert self.world.mode is UsbMode.DEFAULT, "CDC opened in the audio profile"
        return type("Link", (), {"close": lambda self: None})()


def run_set(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, world: World,
            **kw: Any) -> base.Report:
    world.install(monkeypatch)
    ctx = FakeCtx(device=device(UsbMode.DEFAULT), workdir=tmp_path,
                  cancelled=threading.Event(), log=lambda s: None)
    ctx.world = world
    params = Params(phases=kw.pop("phases", ["one", "pitched"]), block_s=0.3,
                    swap_every_s=0.1, bpm=6000, **kw)
    return live_set.SCENARIO.run(ctx, params, base.Progress())


def test_a_clean_set_passes_and_puts_everything_back(tmp_path, monkeypatch) -> None:
    w = World()
    rep = run_set(tmp_path, monkeypatch, w)
    assert rep.passed, rep.summary
    assert "zero glitches in 2 phase(s)" in rep.summary
    c = w.calls
    # each phase: a bounce window in default (scene, bounce, SEQ PLAY again), then a block
    first_bounce = c.index("ASONG BOUNCE 2 0")
    assert c[first_bounce + 1] == "SEQ PLAY"            # a bounce stops it by design
    assert "ASONG BOUNCE 3 1" in c                       # the next phase: its own track
    assert c.index("mode audio") > first_bounce
    assert "capture asis" in c and "stream start" in c and "stream stop" in c
    assert any(x.startswith("sysex kit") for x in c)      # kit swaps inside the block
    assert "KIT_LOAD 5" in c                               # the pitched phase's kit
    assert any(x.startswith("pull /sdcard/SONGS/hil_live_set/clips/") for x in c)
    # put back: scenes deleted newest first, count-in, the kit found at the start
    assert c[-7:] == ["SEQ STOP", "SEQ DEL 3", "SEQ DEL 2", "SEQ COUNTIN 1", "ASONG STOP",
                      "KIT_LOAD 7", "APP_STOP"]
    assert rep.data["route_seen"] == {"mic_src": 3}
    # a kit before the app, and the app with a window for its first kit load
    assert c.index("KIT_LOAD 3") < c.index("APP_START Sampler 15")
    assert w.timeouts == {live_set.SEQ_TIMEOUT_S}
    assert w.running is None                    # found in the launcher, left there


def test_usb_bridge_underruns_fail_the_phase_with_the_number(tmp_path, monkeypatch) -> None:
    w = World()
    w.spk_lines = ["I (1) usb_audio_bridge: SPK #9: len=176 taken=44 ring=0 under=3 "
                   "refused=0 cap_ring=330 cap_drop=0 ring_min=0"]
    rep = run_set(tmp_path, monkeypatch, w, phases=["one"])
    assert rep.exit_code == 1 and "usb_audio_bridge under=3" in rep.summary


def test_a_silent_bounce_fails(tmp_path, monkeypatch) -> None:
    w = World()
    w.bounce_silent = True
    rep = run_set(tmp_path, monkeypatch, w, phases=["many"])
    assert rep.exit_code == 1 and "bounce: the bounced file is silent" in rep.summary


def test_a_scene_that_did_not_survive_the_block_fails(tmp_path, monkeypatch) -> None:
    w = World()
    w.seq_dies_in_uac = True
    rep = run_set(tmp_path, monkeypatch, w, phases=["four"])
    assert "the scene was not playing after the live block" in rep.summary


def test_the_board_is_put_back_even_when_a_block_blows_up(tmp_path, monkeypatch) -> None:
    w = World()

    def no_stream(*a: Any, **kw: Any) -> Any:
        raise live_set.HilError("ENV", "pw-play is not installed: no host stream")

    w.install(monkeypatch)
    monkeypatch.setattr(live_set, "start_host_stream", no_stream)
    ctx = FakeCtx(device=device(UsbMode.DEFAULT), workdir=tmp_path,
                  cancelled=threading.Event(), log=lambda s: None)
    ctx.world = w
    rep = live_set.SCENARIO.run(ctx, Params(phases=["one"], block_s=0.3, bpm=6000),
                                base.Progress())
    assert rep.exit_code == 2 and "pw-play is not installed" in rep.summary
    assert w.mode is UsbMode.DEFAULT
    assert "SEQ DEL 2" in w.calls and w.calls[-2:] == ["KIT_LOAD 7", "APP_STOP"]


def test_unknown_phase_is_bad_args(tmp_path) -> None:
    ctx = base.Context(device=device(UsbMode.DEFAULT), workdir=tmp_path,
                       cancelled=threading.Event(), log=lambda s: None)
    with pytest.raises(live_set.HilError):
        live_set.SCENARIO.run(ctx, Params(phases=["jazz"]), base.Progress())


def test_kit_zero_found_at_the_start_is_put_back(tmp_path, monkeypatch) -> None:
    w = World()
    w.kit = 0
    run_set(tmp_path, monkeypatch, w, phases=["one"])
    assert w.calls[-2:] == ["KIT_LOAD 0", "APP_STOP"]


def test_a_reset_during_the_phase_is_counted(tmp_path, monkeypatch) -> None:
    w = World()
    counts = iter([0, 0, 1, 1, 1, 1])
    w.install(monkeypatch)
    monkeypatch.setattr(verbs, "reset_count", lambda link: next(counts))
    ctx = FakeCtx(device=device(UsbMode.DEFAULT), workdir=tmp_path,
                  cancelled=threading.Event(), log=lambda s: None)
    ctx.world = w
    monkeypatch.setattr(World, "install", lambda self, mp: None)
    rep = live_set.SCENARIO.run(ctx, Params(phases=["one"], block_s=0.2, swap_every_s=0.1,
                                            bpm=6000), base.Progress())
    assert "1 reset(s)" in rep.summary


def test_cap_drop_counts_from_boot_so_the_block_takes_the_difference() -> None:
    """On the bench cap_drop read 33 735 / 33 764 all day: not this block's drops."""
    line = ("I (1) usb_audio_bridge: SPK #{n}: len=176 taken=44 ring=40 under=0 refused=0 "
            "cap_ring=330 cap_drop={d}")
    before = [line.format(n=1, d=33735)]
    block = [line.format(n=2, d=33764), line.format(n=3, d=33764)]
    assert parse_spk(block, before=before)["cap_drop"] == 29
    assert parse_spk(block)["cap_drop"] == 0           # no earlier line: the block's own first
    assert parse_spk(block)["cap_drop_total"] == 33764


def test_a_kit_load_waits_for_a_card_still_mounting(monkeypatch, tmp_path) -> None:
    answers = iter([HilErr("BAD_ARGS", "KIT_LOAD 3: device replied ERR bad kit id"),
                    HilErr("TIMEOUT", "KIT_LOAD 3: the kit scan is still walking the card"),
                    None])

    def kit_load(link: Any, kit: int, wait_s: float = 0) -> dict:
        a = next(answers)
        if a is not None:
            raise a
        return {}

    monkeypatch.setattr(verbs, "kit_load", kit_load)
    ctx = base.Context(device=device(UsbMode.DEFAULT), workdir=tmp_path,
                       cancelled=threading.Event(), log=lambda s: None)
    monkeypatch.setattr(ctx.cancelled, "wait", lambda s: False)
    _audio.load_kit_waiting(ctx, "LINK", 3)
    with pytest.raises(live_set.HilError):     # anything else is not waited out
        monkeypatch.setattr(verbs, "kit_load", lambda *a, **k: (_ for _ in ()).throw(
            HilErr("BAD_ARGS", "KIT_LOAD 3: kit_id must be a non-negative int")))
        _audio.load_kit_waiting(ctx, "LINK", 3)


def HilErr(code: str, message: str) -> live_set.HilError:  # noqa: N802
    return live_set.HilError(code, message)


def test_sd_mode_from_the_boot_log() -> None:
    four = ["I (812) bsp_sdmmc: SDMMC host prepared: slot=0 max_freq=40000kHz",
            "I (815) bsp_sdmmc: SDMMC SLOT_0 pin map: CLK=14 CMD=15 D0=2 D1=4 D2=12 D3=13 width=4",
            "I (2905) bsp_sdmmc: SD Card mounted successfully at /sdcard"]
    assert live_set.parse_sd_boot(four) == {"seen": True, "width": 4, "max_freq_khz": 40000,
                                           "retried_1bit": False, "mounted": True}
    one = four[:2] + ["W (1900) bsp_sdmmc: SD 4-bit init failed (ESP_ERR_TIMEOUT), retrying "
                      "1-bit"] + four[1:]
    sd = live_set.parse_sd_boot(one)
    assert sd["width"] == 1 and sd["retried_1bit"] is True
    assert live_set.parse_sd_boot([])["seen"] is False


def test_the_sd_mode_is_in_the_report(tmp_path, monkeypatch) -> None:
    w = World()
    rep = run_set(tmp_path, monkeypatch, w, phases=["one"])
    assert rep.data["sd"]["width"] == 4
    assert "SD card at boot: width=4-bit max_freq=40000 kHz" in rep.summary


def test_a_board_stuck_in_audio_is_brought_back_by_a_console_reset(tmp_path, monkeypatch) -> None:
    """The full run of 5.10 stopped after MANY: the board stayed in UAC."""
    w = World()
    w.install(monkeypatch)
    real = live_set.LiveSetScenario._live_block

    def block(self, ctx, p, phase, ph, data, artifacts, con):  # noqa: ANN001
        real(self, ctx, p, phase, ph, data, artifacts, con)
        if phase == "many":
            w.stuck_in_audio = 5                 # two asks fail, then the reset clears it

    monkeypatch.setattr(live_set.LiveSetScenario, "_live_block", block)
    monkeypatch.setattr(World, "install", lambda self, mp: None)
    rep = run_set(tmp_path, monkeypatch, w, phases=["many", "pitched"])
    assert rep.passed, rep.summary
    assert w.console_resets == 1
    many = rep.data["phases"][0]
    assert many["recovered_by_reset"] is True and "reset(s)" not in rep.summary
    assert rep.data["phases"][1]["phase"] == "pitched" and "KIT_LOAD 5" in w.calls


def test_a_board_that_hangs_in_the_uac_block_is_a_fail_with_its_console_tail(
        tmp_path, monkeypatch) -> None:
    """2026-10-06: the board hung in the audio profile; arecord's stop raised
    TimeoutExpired, the scenario crashed and left the board in UAC without CDC."""
    import subprocess

    w = World()
    w.install(monkeypatch)
    w.spk_lines = ["I (387218) usb_audio_bridge: SPK #34698: len=704 ring=600 under=0",
                   "E (390000) task_wdt: Task watchdog got triggered"]
    stop = live_set.open_capture.stop

    def hung_stop(self):  # noqa: ANN001
        if "one" in str(self.out):
            w.stuck_in_audio = 5                 # it will not switch back by itself
            raise subprocess.TimeoutExpired(["arecord"], 2.0)
        return stop(self)

    monkeypatch.setattr(live_set.open_capture, "stop", hung_stop)
    monkeypatch.setattr(World, "install", lambda self, mp: None)
    rep = run_set(tmp_path, monkeypatch, w, phases=["one", "four"])
    assert rep.exit_code == 1 and not rep.passed
    one, four = rep.data["phases"]
    assert "board hung in UAC block at t=" in rep.summary
    assert one["hung"]["reason"].startswith("capture stop: TimeoutExpired")
    assert one["hung"]["console_tail"][-1].endswith("Task watchdog got triggered")
    assert one["recovered_by_reset"] is True and w.console_resets == 1
    # straight to the reset: no 45 s profile-switch waits on a board that hung
    reset_at = w.calls.index("console reset")
    assert "mode default" not in w.calls[reset_at - 1:reset_at]
    assert w.stuck_in_audio == 0                  # the reset brought it back
    assert four.get("not_run") is None and four.get("problems") == []


def test_a_phase_that_aborts_does_not_cost_the_next(tmp_path, monkeypatch) -> None:
    w = World()
    w.install(monkeypatch)
    calls = {"n": 0}

    def flaky_stream(wav, target, log):  # noqa: ANN001
        calls["n"] += 1
        if calls["n"] == 1:
            raise live_set.HilError("ENV", "pw-play exited before the block")
        return type("S", (), {"argv": [], "alive": lambda self: True,
                              "stop": lambda self: ""})()

    monkeypatch.setattr(live_set, "start_host_stream", flaky_stream)
    monkeypatch.setattr(World, "install", lambda self, mp: None)
    rep = run_set(tmp_path, monkeypatch, w, phases=["one", "four"])
    one, four = rep.data["phases"]
    assert "pw-play exited" in one["aborted"] and four.get("aborted") is None
    assert four["problems"] == [] and "engine" in four
    assert rep.exit_code == 2 and "aborted (ENV: pw-play exited" in rep.summary


def test_a_board_that_cannot_be_recovered_stops_the_set_cleanly(tmp_path, monkeypatch) -> None:
    w = World()
    w.install(monkeypatch)
    real = live_set.LiveSetScenario._live_block

    def block(self, ctx, p, phase, ph, data, artifacts, con):  # noqa: ANN001
        real(self, ctx, p, phase, ph, data, artifacts, con)
        w.unrecoverable = True

    monkeypatch.setattr(live_set.LiveSetScenario, "_live_block", block)
    monkeypatch.setattr(World, "install", lambda self, mp: None)
    rep = run_set(tmp_path, monkeypatch, w, phases=["one", "four", "many"])
    assert [ph.get("not_run") for ph in rep.data["phases"][1:]] == \
        ["the board could not be recovered"] * 2
    assert rep.exit_code == 2 and "four     not run" in rep.summary


def test_host_capture_verdict() -> None:
    v = live_set.host_capture_verdict
    assert v({"missed_s": 0.1, "max_gap_ms": 30, "overruns": 0}) == (0.1, None)
    lost, why = v({"missed_s": 22.3, "max_gap_ms": 40, "overruns": 0, "exclusive": False})
    assert lost == 22.3 and why == ("host lost 22.3 s of the capture; recorded through "
                                    "PipeWire, the card was busy")
    assert "host overrun" in v({"missed_s": 0, "overruns": 2})[1]
    assert "gap" in v({"missed_s": 0, "max_gap_ms": 900})[1]


def test_a_take_the_host_lost_seconds_of_is_not_charged_to_the_device(tmp_path, monkeypatch) -> None:
    """MANY lost 22.3 s through PipeWire on the bench: the clicks, holes and
    inaudible pads scored from that take are the host's, so the phase is not
    measured (exit 2), not a firmware FAIL."""
    w = World()
    w.capture_health = {"missed_s": 22.3, "max_gap_ms": 40.0, "overruns": 0, "exclusive": False}
    w.install(monkeypatch)
    clean = w.score_take

    def lossy(path, slice_s=10.0, expect=None):  # noqa: ANN001
        if "bounce" in Path(path).name:
            return clean(path, slice_s=slice_s, expect=expect)
        return {"slices": 30, "clicks": 41, "dropouts": 7, "inaudible": 120,
                "silent_slices": 2, "peak_dbfs": -20.0, "worst": []}

    monkeypatch.setattr(live_set, "score_take", lossy)
    monkeypatch.setattr(World, "install", lambda self, mp: None)
    rep = run_set(tmp_path, monkeypatch, w, phases=["many"])
    many = rep.data["phases"][0]
    assert many["host_lost_s"] == 22.3 and many["problems"] == []
    assert rep.exit_code == 2
    assert "capture not measured: host lost 22.3 s of the capture" in rep.summary
    assert "click" not in rep.summary


def test_the_card_is_taken_for_the_block_and_streamed_to_directly(tmp_path, monkeypatch) -> None:
    from contextlib import contextmanager

    w = World()
    w.install(monkeypatch)
    held: list[str] = []

    @contextmanager
    def hold(log, enabled=True, **kw):  # noqa: ANN001
        held.append("pause")
        yield {"paused": True}
        held.append("resume")

    monkeypatch.setattr(live_set.card_hold, "exclusive_card", hold)
    monkeypatch.setattr(live_set, "start_direct_stream", lambda wav, card, log: (
        w.calls.append(f"direct stream {card}"), type("S", (), {
            "argv": ["aplay"], "alive": lambda self: True, "stop": lambda self: ""})())[1])
    monkeypatch.setattr(World, "install", lambda self, mp: None)
    rep = run_set(tmp_path, monkeypatch, w, phases=["one"])
    assert rep.passed, rep.summary
    assert held == ["pause", "resume"] and "direct stream Audio" in w.calls
    assert "stream start" not in w.calls and rep.data["phases"][0]["card_exclusive"] is True


# ------------------------------------------------------------------ end of the host stream
def _spk(n: int, ln: int, ring: int, under: int) -> str:
    return (f"I ({n * 10}) usb_audio_bridge: SPK #{n}: len={ln} taken={ln // 4} ring={ring} "
            f"under={under} refused=0 cap_ring=249 cap_drop=0 ring_min=500")


def test_the_stream_running_dry_at_the_end_is_not_an_underrun() -> None:
    """2026-10-06, ea83f96 / 41e384a: the host stream stops before the switch out of
    the audio profile, and the block's last line shows the ring running dry."""
    lines = [_spk(1, 532, 682, 0), _spk(2, 704, 745, 0), _spk(3, 348, 0, 1)]
    spk = parse_spk(lines)
    assert spk["under"] == 0 and spk["end_of_stream_under"] == 1
    assert phase_problems({"spk": spk}) == []


def test_an_underrun_while_the_stream_ran_still_fails() -> None:
    mid = parse_spk([_spk(1, 532, 682, 0), _spk(2, 704, 120, 1), _spk(3, 524, 653, 1)])
    assert mid["under"] == 1 and "end_of_stream_under" not in mid
    assert phase_problems({"spk": mid}) == ["usb_audio_bridge under=1"]
    both = parse_spk([_spk(1, 532, 682, 0), _spk(2, 704, 120, 1), _spk(3, 348, 0, 2)])
    assert (both["under"], both["end_of_stream_under"]) == (1, 1)
    # a full-size packet with an empty ring is not the stream ending
    full = parse_spk([_spk(1, 532, 682, 0), _spk(2, 704, 0, 1)])
    assert full["under"] == 1 and "end_of_stream_under" not in full


def test_a_deferred_pad_s_wait_is_not_counted_as_late() -> None:
    """01:41 on ea83f96: pad 0, played 12/s, was deferred until its voice ended
    (1150 ms); hits during the wait played the old sample on time. The estimate took
    the whole wait for a reload window: 14 candidates, 1133 ms 'late'."""
    pads = [-1] * 16
    pads[1:16] = [30 + 20 * i for i in range(15)]      # the others: 20 ms each, from 30 ms
    pads[0] = 1150
    rows = [{"seq": 3, "parse_ms": 30, "late_hits": 1, "pads": pads, "deferred": 0x0001}]
    t0 = 100.0
    hits = [(t0 + k / 12.0, 0) for k in range(16)]      # 0..1.25 s, every 83 ms
    got = late_from_timestamps(rows, [t0], hits)
    assert got["per_swap"][0]["candidates"] <= 1 and got["late_max_ms"] <= 20.0
    only = [{"seq": 4, "parse_ms": 30, "late_hits": 1, "pads": [1150] + [-1] * 15,
             "deferred": 0x0001}]
    untimed = late_from_timestamps(only, [t0], hits)
    assert untimed["late_max_ms"] is None and "only deferred pads" in untimed["why"]


# ------------------------------------------------------------------ envelope holes
TAKES = Path(__file__).resolve().parents[1] / "fixtures" / "takes"


def test_a_hole_that_kept_its_noise_is_still_a_hole() -> None:
    """The bounce of 2026-10-06 01:41 (ea83f96): 50 ms under the envelope, ~20 LSB of
    noise left in it, so not a single exact-zero dropout."""
    take = score_take(TAKES / "bounce_hole_ea83f96.wav")
    assert take["dropouts"] == 0
    assert take["holes"] == 1 and 40.0 <= take["hole_max_ms"] <= 60.0
    hole = take["worst"][0]["holes"][0]
    assert 0.19 <= hole["t"] <= 0.215
    assert phase_problems({"take": take})[-1].startswith("capture: 1 hole(s) of 10 ms or more")


def test_a_clean_bounce_has_no_hole() -> None:
    take = score_take(TAKES / "bounce_clean_c510199.wav")
    assert (take["holes"], take["dropouts"], take["clicks"]) == (0, 0, 0)


def _tone_take(tmp_path: Path, spans: list[tuple[float, float]], seconds: float,
               name: str = "t.wav") -> Path:
    """A 440 Hz tone sounding in ``spans``, ~20 LSB of noise elsewhere."""
    sr = 44100
    t = np.arange(int(sr * seconds)) / sr
    x = np.random.default_rng(1).normal(0.0, 20.0, len(t))
    for a, b in spans:
        on = (t >= a) & (t < b)
        x[on] += 8000.0 * np.sin(2 * np.pi * 440.0 * t[on])
    path = tmp_path / name
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sr)
        w.writeframes(np.clip(x, -32768, 32767).astype("<i2").tobytes())
    return path


def test_envelope_holes_ignore_the_take_s_silent_ends_and_short_dips(tmp_path: Path) -> None:
    # sound 0.3..1.7 s with a 6 ms dip (too short) and a 25 ms hole
    take = score_take(_tone_take(tmp_path, [(0.3, 0.8), (0.806, 1.2), (1.225, 1.7)], 2.0))
    assert take["holes"] == 1 and 20.0 <= take["hole_max_ms"] <= 30.0


def test_a_hole_across_two_slices_is_one_hole(tmp_path: Path) -> None:
    take = score_take(_tone_take(tmp_path, [(0.1, 0.995), (1.02, 1.9)], 2.0), slice_s=1.0)
    assert take["holes"] == 1 and 20.0 <= take["hole_max_ms"] <= 30.0
    silent = score_take(_tone_take(tmp_path, [], 1.0, "silent.wav"))
    assert silent["holes"] == 0
