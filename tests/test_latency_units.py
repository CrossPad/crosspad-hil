"""The parts of `latency` that carry its numbers, one at a time.

`latency_math` (clock maps, onsets, statistics), the HIL CLICK fixture, the
console and desk parsers, the ESP echo listener, and `duplex.DuplexRecording`
over a fake libasound -- the ctypes layer itself only ever meets a real card.
"""
from __future__ import annotations

import json
import queue
import threading
import time
import wave
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from crosspad_hil import duplex, fixtures, latency_math
from crosspad_hil.errors import ENV, HilError
from crosspad_hil.scenarios import latency

SR = 44100


# --------------------------------------------------------------------------- lines and clocks


def test_robust_line_ignores_outliers() -> None:
    x = np.arange(500, dtype=float)
    y = 3.0 + 0.5 * x
    y[::37] += 40.0                            # a few wild points
    slope, icpt = latency_math.robust_line(x, y)
    assert slope == pytest.approx(0.5, abs=1e-9) and icpt == pytest.approx(3.0, abs=1e-6)
    small = latency_math.robust_line(x[:50], y[:50])
    assert small[0] == pytest.approx(0.5, abs=1e-9)
    with pytest.raises(ValueError):
        latency_math.robust_line([1.0], [2.0])


def test_lower_envelope_recovers_a_drifting_clock_under_late_stamps() -> None:
    rng = np.random.default_rng(0)
    spf = (1.0 / SR) * (1 + 120e-6)            # device 120 ppm slow against the host
    t0 = 123456.789
    ts = np.arange(t0, t0 + 60.0, 0.001)
    frames = np.floor((ts - t0) / spf / 44) * 44           # moves in 1 ms packets
    stamps = ts + rng.exponential(0.0005, ts.size)         # always late
    stamps[::997] += 0.03                                   # a stalled thread now and then
    fit = latency_math.lower_envelope_fit(frames, stamps, nominal_rate=SR)
    for f in (0, SR * 10, SR * 59):
        assert float(fit.t(f)) == pytest.approx(t0 + f * spf, abs=5e-5)
    assert fit.rate == pytest.approx(1 / spf, abs=0.02)
    assert float(fit.frame(fit.t(12345.0))) == pytest.approx(12345.0)
    assert fit.resid_ms["min"] > -0.1 and fit.resid_ms["p50"] > 0
    json.dumps(fit.to_dict())


def test_lower_envelope_refuses_a_log_at_the_wrong_rate() -> None:
    ts = np.arange(0.0, 10.0, 0.01)
    with pytest.raises(ValueError, match="not the stream's"):
        latency_math.lower_envelope_fit(ts * 48000, ts, nominal_rate=SR)
    with pytest.raises(ValueError):
        latency_math.lower_envelope_fit([1.0], [1.0], nominal_rate=SR)


# --------------------------------------------------------------------------- onsets


def _take(n: int = SR, noise: float = 1e-4, seed: int = 1) -> np.ndarray:
    return np.random.default_rng(seed).normal(0.0, noise, n)


def _click(level: float = 0.1, pad: int = 0) -> np.ndarray:
    return np.asarray(fixtures.click_samples(fixtures.click_seed(pad)), dtype=float) * level / 0.25


def test_onset_is_the_first_frame_of_the_click() -> None:
    x = _take()
    x[10000:10000 + 2205] += _click()
    floor = latency_math.noise_floor(x, SR)
    assert floor == pytest.approx(1e-4, rel=0.3)
    on = latency_math.find_onset(x, 9000, 18000, sr=SR, noise=floor)
    assert on.status == "ok" and on.events == 1
    assert 9999.0 <= on.index <= 10000.0
    assert on.to_dict()["status"] == "ok"


def test_a_quiet_window_is_missing() -> None:
    x = _take()
    on = latency_math.find_onset(x, 9000, 18000, sr=SR, noise=latency_math.noise_floor(x, SR))
    assert on.status == "missing" and on.index is None


def test_a_window_opening_on_sound_is_masked() -> None:
    x = _take()
    x[8000:20000] += 0.05 * np.sin(np.arange(12000) * 2 * np.pi * 440 / SR)   # a long tone
    x[10000:10000 + 2205] += _click()
    on = latency_math.find_onset(x, 9000, 18000, sr=SR, noise=latency_math.noise_floor(x, SR))
    assert on.status == "masked"


def test_a_sound_that_stops_and_starts_again_is_double() -> None:
    x = _take()
    x[10000:10000 + 2205] += _click()
    x[13000:13000 + 2205] += _click()          # 68 ms later, after a quiet gap
    on = latency_math.find_onset(x, 9000, 18000, sr=SR, noise=latency_math.noise_floor(x, SR))
    assert on.status == "double" and on.events == 2
    assert 9999.0 <= on.index <= 10000.0


def test_a_ringing_tail_is_one_sound() -> None:
    x = _take()
    x[10000:10000 + 2205] += _click()
    k = np.arange(4000)
    x[10900:14900] += 0.004 * np.exp(-k / 800) * np.sin(k * 2 * np.pi * 900 / SR)
    on = latency_math.find_onset(x, 9000, 18000, sr=SR, noise=latency_math.noise_floor(x, SR))
    assert on.status == "ok" and on.events == 1


# --------------------------------------------------------------------------- statistics


def test_describe_and_split() -> None:
    v = list(np.linspace(10.0, 12.0, 21))
    d = latency_math.describe(v)
    assert d["n"] == 21 and d["p50"] == 11.0 and d["min"] == 10.0 and d["max"] == 12.0
    assert d["p95"] == pytest.approx(11.9) and d["p50_ci95"][0] <= 11.0 <= d["p50_ci95"][1]
    assert latency_math.describe(v) == d                    # seeded: the same twice
    assert latency_math.describe([]) == {"n": 0}
    assert "p50_ci95" not in latency_math.describe([1.0, 2.0])
    before, after, amb = latency_math.split_at([1.0, 2.0, 3.0, 4.0, 5.0],
                                               [10, 20, 30, 40, 50], 3.5, 1.0)
    assert before == [10, 20] and after == [40, 50] and amb == 1


# --------------------------------------------------------------------------- HIL CLICK


def test_click_starts_at_full_level_and_is_deterministic() -> None:
    a = fixtures.click_samples(fixtures.click_seed(0))
    assert len(a) == round(SR * fixtures.CLICK_SECONDS)
    assert a[0] == fixtures.CLICK_PEAK and a[1] == -fixtures.CLICK_PEAK
    n_noise = round(SR * fixtures.CLICK_NOISE_S)
    assert max(abs(v) for v in a[:n_noise]) <= fixtures.CLICK_PEAK
    assert all(v == 0.0 for v in a[n_noise:])
    assert abs(a[n_noise - 1]) < 1e-12                      # faded to nothing
    assert a == fixtures.click_samples(fixtures.click_seed(0))
    assert a[2:50] != fixtures.click_samples(fixtures.click_seed(1))[2:50]


def test_click_kit_on_disk(tmp_path: Path) -> None:
    built = fixtures.write_click_kits(tmp_path)
    kit_dir = built["HIL CLICK"]
    assert kit_dir.name == "HIL_CLICK"
    kit = json.loads((kit_dir / "kit.json").read_text())
    assert kit["name"] == "HIL CLICK" and len(kit["pads"]) == 16
    assert all("playbackMode" not in p for p in kit["pads"])   # one-shot
    files = fixtures.kit_files(kit_dir)
    assert files[-1].name == "kit.json" and len(files) == 17
    with wave.open(str(kit_dir / "SAMPLES" / "click05.wav"), "rb") as w:
        assert (w.getnchannels(), w.getframerate(), w.getsampwidth()) == (2, SR, 2)
        first = np.frombuffer(w.readframes(1), dtype="<i2")
    assert list(first) == [int(fixtures.CLICK_PEAK * 32767)] * 2


# --------------------------------------------------------------------------- parsers


SPK = ("I (123456) usb_audio_bridge: SPK #7: len=528 taken=528 ring=880 under={u} refused=0 "
       "cap_ring={c} cap_drop=17 ring_min=600 rt_busy_max_us=900 idle1_pct=30 "
       "spk_gap_max_us=4100 cap_under=0")


def test_spk_lines_and_their_summary() -> None:
    row = latency.parse_spk_line(SPK.format(u=0, c=300))
    assert row["n"] == 7 and row["cap_ring"] == 300 and row["ring"] == 880
    assert latency.parse_spk_line("I (1) main: nothing here") is None
    rows = [{**latency.parse_spk_line(SPK.format(u=u, c=c)), "_t": t}
            for t, u, c in ((10.0, 0, 290), (11.0, 0, 310), (12.0, 2, 300), (13.0, 2, 300))]
    s = latency.spk_summary(rows, SR)
    assert s["delta"] == {"under": 2, "refused": 0, "cap_drop": 0, "cap_under": 0}
    assert s["event_seen_at"] == 12.0 and s["cap_ring_frames"]["mean"] == 300.0
    leg = latency.capture_leg(rows, 10.5, 11.2, SR)
    assert leg["frames_mean"] == 305.0 and leg["from_phase"] and leg["lines"] == 2
    assert leg["ms"] == pytest.approx(305 / SR * 1000, abs=1e-3)
    assert latency.capture_leg(rows, 100.0, 101.0, SR)["from_phase"] is False
    assert latency.capture_leg([], 0.0, 1.0, SR) is None


def test_the_event_guard_spans_the_poll_gap(monkeypatch: pytest.MonkeyPatch) -> None:
    # A line read at _t was printed after the poll before it, and the worker prints
    # at most a pass after the event: the event is no earlier than that poll minus
    # a pass. A fixed 1.5 s is too short once polls are over half a second apart.
    def row(t: float, prev: float, under: int) -> dict[str, Any]:
        return {**latency.parse_spk_line(SPK.format(u=under, c=300)), "_t": t, "_t_prev": prev}

    s = latency.spk_summary([row(10.0, 9.9, 0), row(12.4, 10.0, 1)], SR)
    assert s["event_seen_at"] == 12.4
    assert s["event_guard_s"] == pytest.approx(
        2.4 + latency.SPK_WORKER_S + latency.SPK_SEEN_MARGIN_S)
    s = latency.spk_summary([row(10.0, 9.9, 0), row(10.3, 10.0, 1)], SR)
    assert s["event_guard_s"] == latency.SPK_GUARD_S          # never under the old floor

    # SpkWatch stamps each line with the poll that read it and the one before
    clock = iter([100.0, 100.4, 101.9])
    monkeypatch.setattr(latency, "_now", lambda: next(clock))

    class Con:
        seq = 0
        lines: list[tuple[int, str]] = []

        def read(self, since_seq: int, match: str, limit: int) -> Any:
            sel = [(s, ln) for s, ln in self.lines if s >= since_seq]
            nxt = sel[-1][0] + 1 if sel else since_seq
            return type("R", (), {"lines": sel, "next_seq": nxt})()

    con = Con()
    watch = latency.SpkWatch(con)                              # t = 100.0
    con.lines = [(1, SPK.format(u=0, c=300))]
    watch.poll()                                               # t = 100.4
    con.lines.append((2, SPK.format(u=1, c=300)))
    watch.poll()                                               # t = 101.9
    assert [(r["_t"], r["_t_prev"]) for r in watch.rows] == [(100.4, 100.0), (101.9, 100.4)]


def test_desk_listing_with_names_that_have_spaces() -> None:
    line = ("MIXPRE: ok=1 undo=0 dir=/sdcard/crosspad/mixer 1=s:My Desk 2=f:BEATS "
            "3=f:SAMPLING 4=e: 5=e: 6=s:LIVE 2")
    assert latency.parse_desks(line) == {"My Desk": 1, "BEATS": 2, "SAMPLING": 3, "LIVE 2": 6}


def test_echo_matching() -> None:
    hits = [latency.PadHit(k=0, pad=0, note=36, t_send=1.0, send_ms=0.0),
            latency.PadHit(k=1, pad=1, note=37, t_send=2.0, send_ms=0.0)]
    notes = [(1.0025, [0x90, 36, 100]), (1.5, [0x80, 36, 0]), (2.0031, [0x90, 36, 100]),
             (2.2, [0x90, 37, 100])]
    assert latency.match_echoes(hits, notes) == [2.5, None]   # 37 came back too late


class _NoteQueue:
    def __init__(self) -> None:
        self.q: queue.Queue[list[int]] = queue.Queue()

    def receive_note(self, timeout_s: float = 1.0) -> list[int] | None:
        try:
            return self.q.get(timeout=timeout_s)
        except queue.Empty:
            return None


def test_note_echoes_are_stamped_as_they_arrive() -> None:
    io = _NoteQueue()
    echoes = latency.NoteEchoes(io)
    echoes.start()
    t0 = time.monotonic()
    io.q.put([0x90, 36, 100])
    deadline = time.monotonic() + 2.0
    while not echoes.notes and time.monotonic() < deadline:
        time.sleep(0.005)
    echoes.stop()
    assert echoes.notes and echoes.notes[0][1] == [0x90, 36, 100]
    assert 0.0 <= echoes.notes[0][0] - t0 < 0.5


# --------------------------------------------------------------------------- duplex


class FakeAlsa:
    """libasound's shape at `AlsaLib`'s level, over a card running in real time."""

    def __init__(self, *, buffer: int = 1764, period: int = 441, busy: bool = False,
                 xrun_reads: int = 0) -> None:
        self.buffer, self.period = buffer, period
        self.busy = busy
        self.xrun_reads = xrun_reads
        self.out = bytearray()
        self.read = 0
        self.t_start: dict[str, float] = {}
        self.closed: list[str] = []

    def open(self, name: str, stream: int, *, nonblock: bool = True) -> str:
        if self.busy:
            raise duplex.AlsaError("snd_pcm_open", -duplex.EBUSY, "Device or resource busy")
        return "cap" if stream == duplex.SND_PCM_STREAM_CAPTURE else "play"

    def set_params(self, h: str, *, rate: int, channels: int, latency_us: int) -> None:
        self.rate = rate

    def get_params(self, h: str) -> tuple[int, int]:
        return self.buffer, self.period

    def link(self, a: str, b: str) -> bool:
        return True

    def state(self, h: str) -> int:
        return duplex.SND_PCM_STATE_RUNNING if h in self.t_start else \
            duplex.SND_PCM_STATE_PREPARED

    def start(self, h: str) -> int:
        now = time.monotonic()
        self.t_start.setdefault("cap", now)       # linked: one starts both
        self.t_start.setdefault("play", now)
        return 0

    def _elapsed(self, h: str) -> int:
        t = self.t_start.get(h)
        return 0 if t is None else int((time.monotonic() - t) * self.rate)

    def avail_delay(self, h: str) -> tuple[int, int]:
        if h == "cap":
            n = self._elapsed("cap") - self.read
            return n, n
        queued = len(self.out) // 4 - self._elapsed("play")
        return self.buffer - queued, queued

    def readi(self, h: str, frames: int, channels: int) -> bytes:
        if self.xrun_reads:
            self.xrun_reads -= 1
            raise duplex.AlsaError("snd_pcm_readi", -duplex.EPIPE, "Broken pipe")
        n = min(frames, self._elapsed("cap") - self.read)
        self.read += n
        return bytes(n * channels * 2)

    def writei(self, h: str, data: bytes, channels: int) -> int:
        frames = len(data) // (channels * 2)
        space = self.buffer - (len(self.out) // 4 - self._elapsed("play"))
        n = max(0, min(frames, space))
        self.out += data[: n * channels * 2]
        return n

    def recover(self, h: str, errno: int) -> None:
        return None

    def drop(self, h: str) -> None:
        return None

    def close(self, h: str) -> None:
        self.closed.append(h)


def test_duplex_records_plays_schedules_and_logs(tmp_path: Path) -> None:
    alsa = FakeAlsa()
    rec = duplex.DuplexRecording(alsa, "plughw:Fake", tmp_path / "take.wav", rate=SR)
    rec.start()
    assert rec.info["linked"] and rec.info["playback_buffer"] == 1764
    time.sleep(0.15)
    frame = rec.out_position() + rec.schedule_lead_frames
    rec.schedule(frame, [1000, -1000, 500])
    time.sleep(0.35)
    rec.stop()
    out = np.frombuffer(bytes(alsa.out), dtype="<i2").reshape(-1, 2)
    assert list(out[frame:frame + 3, 0]) == [1000, -1000, 500]
    assert list(out[frame:frame + 3, 1]) == [1000, -1000, 500]
    assert not out[:frame].any() and not out[frame + 3:].any()
    with wave.open(str(tmp_path / "take.wav"), "rb") as w:
        assert w.getframerate() == SR and w.getnchannels() == 2
        assert w.getnframes() == rec.frames_in > 0.3 * SR
    cap, play = rec.capture_points(), rec.playback_points()
    assert len(cap) > 100 and len(play) > 100
    fit = latency_math.lower_envelope_fit([f for f, _ in cap], [t for _, t in cap],
                                          nominal_rate=SR)
    assert float(fit.t(0)) == pytest.approx(alsa.t_start["cap"], abs=0.002)
    h = rec.health()
    assert h["ok"], h
    assert h["out_xruns"] == 0 and h["overruns"] == 0
    assert alsa.closed == ["play", "cap"]


def test_the_playback_log_maps_a_scheduled_frame_to_when_it_played(tmp_path: Path) -> None:
    # The host path's reference is play.t(frame): the playback log must say when a
    # frame left, not when it was written (frames_out - delay, not frames_out).
    # The fake card plays frame f at t_start + f / rate and writes a full buffer
    # ahead of that, so a sign or a missing delay is off by the whole buffer.
    alsa = FakeAlsa()
    rec = duplex.DuplexRecording(alsa, "plughw:Fake", tmp_path / "take.wav", rate=SR)
    rec.start()
    time.sleep(0.2)
    frame = rec.out_position() + rec.schedule_lead_frames
    rec.schedule(frame, [1000] * 4)
    time.sleep(0.35)
    rec.stop()
    pts = rec.playback_points()
    play = latency_math.lower_envelope_fit([f for f, _ in pts], [t for _, t in pts],
                                           nominal_rate=SR)
    assert float(play.t(frame)) == pytest.approx(alsa.t_start["play"] + frame / SR, abs=0.002)
    assert alsa.buffer / SR > 0.03                            # what a wrong log would be off by


def test_a_capture_xrun_is_counted_and_survived(tmp_path: Path) -> None:
    alsa = FakeAlsa(xrun_reads=1)
    rec = duplex.DuplexRecording(alsa, "plughw:Fake", tmp_path / "take.wav", rate=SR)
    rec.start()
    time.sleep(0.15)
    rec.stop()
    h = rec.health()
    assert rec.in_xruns == 1 and not h["ok"] and "1 capture xrun(s)" in h["flags"]


def test_a_busy_card_names_the_bench_lease(tmp_path: Path) -> None:
    rec = duplex.DuplexRecording(FakeAlsa(busy=True), "plughw:Fake", tmp_path / "t.wav",
                                 rate=SR)
    with pytest.raises(HilError) as e:
        rec.start()
    assert e.value.code == ENV and "busy" in e.value.message
    assert "bench claim" in (e.value.hint or "")


def test_backend_hands_back_its_recording(tmp_path: Path) -> None:
    from crosspad_hil.capture import CaptureDevice

    alsa = FakeAlsa()
    backend = duplex.AlsaDuplexBackend(alsa=alsa, finder=object())
    cap = CaptureDevice(name="CrossPad", api="ALSA", alsa_id="Crosspad", sample_rate=SR)
    rec = backend.start(cap, tmp_path / "t.wav", seconds=None, sample_rate=SR, channels=2)
    try:
        assert backend.recording is rec and rec.pcm == "plughw:Crosspad"
    finally:
        rec.stop()
    with pytest.raises(HilError):
        backend.start(CaptureDevice(name="x", api="ALSA"), tmp_path / "u.wav", seconds=None,
                      sample_rate=SR, channels=2)


def test_alsa_lib_binds_its_calls() -> None:
    """The ctypes signatures against a stand-in library object: names and types only."""

    class Fn:
        def __call__(self, *a: Any) -> int:
            return 0

    class Lib:
        def __getattr__(self, name: str) -> Fn:
            fn = Fn()
            object.__setattr__(self, name, fn)
            return fn

    lib = Lib()
    duplex.AlsaLib(lib)
    for name in ("snd_pcm_open", "snd_pcm_set_params", "snd_pcm_readi", "snd_pcm_writei",
                 "snd_pcm_avail_delay", "snd_pcm_link", "snd_strerror"):
        assert getattr(lib, name).argtypes is not None, name


def test_thread_safety_of_the_schedule(tmp_path: Path) -> None:
    alsa = FakeAlsa()
    rec = duplex.DuplexRecording(alsa, "plughw:Fake", tmp_path / "take.wav", rate=SR)
    rec.start()
    frames = []

    def scheduler() -> None:
        for _ in range(20):
            f = rec.out_position() + rec.schedule_lead_frames
            rec.schedule(f, [7] * 10)
            frames.append(f)
            time.sleep(0.01)

    th = threading.Thread(target=scheduler)
    th.start()
    th.join()
    time.sleep(0.12)
    rec.stop()
    out = np.frombuffer(bytes(alsa.out), dtype="<i2").reshape(-1, 2)[:, 0]
    for f in frames:
        assert (out[f:f + 10] == 7).all(), f
