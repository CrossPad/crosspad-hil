"""One full-duplex PCM stream on the board's UAC2 card, with a clock log.

The `latency` scenario has to know *when* a frame was at the converter, on
the same clock it stamps its MIDI sends with. `arecord` cannot say (its start
is a process start, tens of ms either way), and the bench has no PortAudio, so
this talks to libasound directly through ctypes -- nothing to install.

One thread owns both directions of the card, non-blocking:

* capture: read what is there, append it to the WAV, and log
  ``(frames read + snd_pcm_delay, monotonic)`` -- the frame being digitised
  right now;
* playback: keep the buffer topped up with silence plus any burst the caller
  scheduled at an absolute output frame, and log
  ``(frames written - snd_pcm_delay, monotonic)`` -- the frame being played
  right now.

Both logs are what `latency_math.lower_envelope_fit` turns into frame <->
time maps. Keeping the playback stream running also keeps the firmware
printing its `SPK #` lines (only a live UAC OUT stream does), which is where
the capture ring's level -- the part of the capture path the board reports --
comes from.

`AlsaDuplexBackend` has `capture.AudioBackend`'s shape, so `CaptureSession`
does the profile switch, the route and the restore exactly as for every other
take. A stream xrun on either side is counted, recovered and reported in
`health()`: the clock log is broken across it, so the scenario does not score
such a take.
"""
from __future__ import annotations

import array
import ctypes
import ctypes.util
import threading
import time
import wave
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

from .capture import ArecordBackend, CaptureDevice
from .devices import Device
from .errors import ENV, NOT_SUPPORTED, HilError

SND_PCM_STREAM_PLAYBACK = 0
SND_PCM_STREAM_CAPTURE = 1
SND_PCM_NONBLOCK = 1
SND_PCM_FORMAT_S16_LE = 2
SND_PCM_ACCESS_RW_INTERLEAVED = 3
SND_PCM_STATE_PREPARED = 2
SND_PCM_STATE_RUNNING = 3
EAGAIN = 11
EBUSY = 16
EPIPE = 32
EBADFD = 77
ESTRPIPE = 86

DEFAULT_LATENCY_MS = 40.0       # each direction's buffer; periods are a quarter of it
IDLE_SLEEP_S = 0.0005           # nothing to read and nothing to write: wait this long
MAX_POINTS = 2_000_000          # ~15 min of points at the loop's rate; a cap, not a target
BYTES_PER_SAMPLE = 2


class AlsaError(HilError):
    """A negative return from libasound; ``errno`` is its absolute value."""

    def __init__(self, what: str, err: int, text: str) -> None:
        super().__init__(ENV, f"{what}: {text} ({err})", errno=abs(err))
        self.errno = abs(err)


class AlsaLib:
    """The dozen libasound calls the stream needs, with Python types at the edges."""

    def __init__(self, lib: Any | None = None) -> None:
        if lib is None:
            name = ctypes.util.find_library("asound") or "libasound.so.2"
            try:
                lib = ctypes.CDLL(name)
            except OSError as exc:
                raise HilError(NOT_SUPPORTED, f"libasound is not loadable: {exc}",
                               hint="the latency scenario needs alsa-lib (libasound2)") from exc
        self._lib = lib
        pcm = ctypes.c_void_p
        sig = {
            "snd_pcm_open": ([ctypes.POINTER(pcm), ctypes.c_char_p, ctypes.c_int, ctypes.c_int],
                             ctypes.c_int),
            "snd_pcm_set_params": ([pcm, ctypes.c_int, ctypes.c_int, ctypes.c_uint,
                                    ctypes.c_uint, ctypes.c_int, ctypes.c_uint], ctypes.c_int),
            "snd_pcm_get_params": ([pcm, ctypes.POINTER(ctypes.c_ulong),
                                    ctypes.POINTER(ctypes.c_ulong)], ctypes.c_int),
            "snd_pcm_link": ([pcm, pcm], ctypes.c_int),
            "snd_pcm_prepare": ([pcm], ctypes.c_int),
            "snd_pcm_start": ([pcm], ctypes.c_int),
            "snd_pcm_drop": ([pcm], ctypes.c_int),
            "snd_pcm_close": ([pcm], ctypes.c_int),
            "snd_pcm_state": ([pcm], ctypes.c_int),
            "snd_pcm_readi": ([pcm, ctypes.c_void_p, ctypes.c_ulong], ctypes.c_long),
            "snd_pcm_writei": ([pcm, ctypes.c_void_p, ctypes.c_ulong], ctypes.c_long),
            "snd_pcm_avail_delay": ([pcm, ctypes.POINTER(ctypes.c_long),
                                     ctypes.POINTER(ctypes.c_long)], ctypes.c_int),
            "snd_pcm_recover": ([pcm, ctypes.c_int, ctypes.c_int], ctypes.c_int),
            "snd_strerror": ([ctypes.c_int], ctypes.c_char_p),
        }
        for fn, (args, res) in sig.items():
            f = getattr(lib, fn)
            f.argtypes = args
            f.restype = res
        # Optional: only the jam's "the phone stops its stream" needs it.
        self._has_unlink = hasattr(lib, "snd_pcm_unlink")
        if self._has_unlink:
            lib.snd_pcm_unlink.argtypes = [pcm]
            lib.snd_pcm_unlink.restype = ctypes.c_int

    def _check(self, what: str, rc: int) -> int:
        if rc < 0:
            raise AlsaError(what, rc, self.strerror(rc))
        return rc

    def strerror(self, err: int) -> str:
        raw = self._lib.snd_strerror(int(err))
        return raw.decode(errors="replace") if raw else f"error {err}"

    def open(self, name: str, stream: int, *, nonblock: bool = True) -> Any:
        handle = ctypes.c_void_p()
        self._check(f"snd_pcm_open {name}", self._lib.snd_pcm_open(
            ctypes.byref(handle), name.encode(), stream, SND_PCM_NONBLOCK if nonblock else 0))
        return handle

    def set_params(self, h: Any, *, rate: int, channels: int, latency_us: int) -> None:
        self._check("snd_pcm_set_params", self._lib.snd_pcm_set_params(
            h, SND_PCM_FORMAT_S16_LE, SND_PCM_ACCESS_RW_INTERLEAVED, channels, rate, 0,
            latency_us))

    def get_params(self, h: Any) -> tuple[int, int]:
        buf, per = ctypes.c_ulong(), ctypes.c_ulong()
        self._check("snd_pcm_get_params",
                    self._lib.snd_pcm_get_params(h, ctypes.byref(buf), ctypes.byref(per)))
        return int(buf.value), int(per.value)

    def link(self, a: Any, b: Any) -> bool:
        return self._lib.snd_pcm_link(a, b) >= 0

    def unlink(self, h: Any) -> bool:
        """Undo snd_pcm_link for ``h``; False when the library cannot."""
        if not getattr(self, "_has_unlink", False):
            return False
        return self._lib.snd_pcm_unlink(h) >= 0

    def prepare(self, h: Any) -> None:
        self._check("snd_pcm_prepare", self._lib.snd_pcm_prepare(h))

    def start(self, h: Any) -> int:
        return int(self._lib.snd_pcm_start(h))

    def state(self, h: Any) -> int:
        return int(self._lib.snd_pcm_state(h))

    def drop(self, h: Any) -> None:
        self._lib.snd_pcm_drop(h)

    def close(self, h: Any) -> None:
        self._lib.snd_pcm_close(h)

    def readi(self, h: Any, frames: int, channels: int) -> bytes:
        """Up to ``frames`` frames; b"" when nothing is ready (EAGAIN)."""
        buf = ctypes.create_string_buffer(frames * channels * BYTES_PER_SAMPLE)
        n = int(self._lib.snd_pcm_readi(h, buf, frames))
        if n == -EAGAIN:
            return b""
        self._check("snd_pcm_readi", n)
        return buf.raw[: n * channels * BYTES_PER_SAMPLE]

    def writei(self, h: Any, data: bytes, channels: int) -> int:
        """Frames accepted (0 when the buffer is full)."""
        frames = len(data) // (channels * BYTES_PER_SAMPLE)
        n = int(self._lib.snd_pcm_writei(h, data, frames))
        if n == -EAGAIN:
            return 0
        return self._check("snd_pcm_writei", n)

    def avail_delay(self, h: Any) -> tuple[int, int]:
        avail, delay = ctypes.c_long(), ctypes.c_long()
        self._check("snd_pcm_avail_delay", self._lib.snd_pcm_avail_delay(
            h, ctypes.byref(avail), ctypes.byref(delay)))
        return int(avail.value), int(delay.value)

    def recover(self, h: Any, errno: int) -> None:
        self._lib.snd_pcm_recover(h, -abs(int(errno)), 1)


class PointLog:
    """``(frame, t)`` clock points, decimated to the earliest-looking one per ``bin_s``.

    The I/O loop logs a point every pass (~2000 a second); a 20-minute jam
    would hold millions. What a lower-envelope clock fit uses is the least-late
    point of each short bin, so that is all that is kept: the point whose
    ``t - frame / rate`` is smallest in each ``bin_s`` slice of time.
    """

    def __init__(self, rate: float, bin_s: float = 0.010, cap: int = MAX_POINTS) -> None:
        self._spf = 1.0 / float(rate)
        self._bin_s = float(bin_s)
        self._cap = int(cap)
        self.points: list[tuple[int, float]] = []
        self._bin: int | None = None
        self._best: tuple[int, float] | None = None
        self._best_r = 0.0
        self.seen = 0

    def add(self, frame: int, t: float) -> None:
        self.seen += 1
        b = int(t // self._bin_s)
        r = t - frame * self._spf
        if self._bin is None or b != self._bin:
            self._flush()
            self._bin, self._best, self._best_r = b, (int(frame), float(t)), r
        elif r < self._best_r:
            self._best, self._best_r = (int(frame), float(t)), r

    def _flush(self) -> None:
        if self._best is not None and len(self.points) < self._cap:
            self.points.append(self._best)
        self._best = None

    def snapshot(self) -> list[tuple[int, float]]:
        out = list(self.points)
        if self._best is not None:
            out.append(self._best)
        return out


class DuplexRecording:
    """A running duplex stream: the WAV of what came in, the clock logs, the schedule.

    Has `capture.Recording`'s shape (``path``, ``stop``, ``wait``, ``health``)
    plus what the latency scenario needs: ``out_position()``, ``schedule()``,
    ``capture_points()`` and ``playback_points()``.
    """

    def __init__(self, alsa: Any, pcm: str, path: Path, *, rate: int, channels: int = 2,
                 latency_ms: float = DEFAULT_LATENCY_MS,
                 clock: Callable[[], float] = time.monotonic,
                 source: Callable[[int, int], bytes] | None = None,
                 decimate_s: float | None = None) -> None:
        self.path = Path(path)
        self.pcm = pcm
        self.sample_rate = int(rate)
        self.channels = int(channels)
        self._alsa = alsa
        self._latency_us = int(latency_ms * 1000)
        self._clock = clock
        self._cap: Any = None
        self._play: Any = None
        self._wav: Any = None
        self._thread: threading.Thread | None = None
        self._halt = threading.Event()
        self._lock = threading.Lock()
        self._bursts: list[tuple[int, list[int]]] = []
        self._cap_pts: list[tuple[int, float]] = []
        self._play_pts: list[tuple[int, float]] = []
        # A continuous signal to play (``source(first_frame, n) -> int16 bytes``,
        # interleaved), bursts on top; None plays silence.
        self._source = source
        # Decimated clock logs (PointLog) instead of every pass, for long takes.
        self._cap_log = PointLog(rate, decimate_s) if decimate_s else None
        self._play_log = PointLog(rate, decimate_s) if decimate_s else None
        # Timestamped stream events: xruns on either side, the output paused and
        # resumed -- where on the clock the take is broken.
        self.events: list[dict[str, Any]] = []
        self._pause_req: tuple[str, float] | None = None
        self._out_paused_until: float | None = None
        self._mute_until: float | None = None
        self._linked = False
        self.frames_in = 0
        self.frames_out = 0
        self.in_xruns = 0
        self.out_xruns = 0
        self.max_gap_s = 0.0
        self.error: BaseException | None = None
        self.info: dict[str, Any] = {"pcm": pcm, "rate": self.sample_rate,
                                     "channels": self.channels}
        self.started_at: float | None = None
        self._t_start = 0.0
        self._wall = 0.0
        self._stopped = False

    # -- lifecycle -----------------------------------------------------------
    def start(self) -> None:
        a = self._alsa
        try:
            self._cap = a.open(self.pcm, SND_PCM_STREAM_CAPTURE)
        except AlsaError as e:
            raise self._busy_or(e) from e
        try:
            self._play = a.open(self.pcm, SND_PCM_STREAM_PLAYBACK)
            for h in (self._cap, self._play):
                a.set_params(h, rate=self.sample_rate, channels=self.channels,
                             latency_us=self._latency_us)
            cbuf, cper = a.get_params(self._cap)
            pbuf, pper = a.get_params(self._play)
            linked = a.link(self._cap, self._play)
            self._linked = bool(linked)
            self.info.update(capture_buffer=cbuf, capture_period=cper, playback_buffer=pbuf,
                             playback_period=pper, linked=linked)
            self._pbuf = pbuf
            self._chunk = max(1, cper)
            self._wav = wave.open(str(self.path), "wb")
            self._wav.setnchannels(self.channels)
            self._wav.setsampwidth(BYTES_PER_SAMPLE)
            self._wav.setframerate(self.sample_rate)
            # Prime the playback buffer with silence; reaching its start
            # threshold starts it (and, when linked, the capture with it).
            self._write_out(pbuf)
            for h in (self._play, self._cap):
                if a.state(h) == SND_PCM_STATE_PREPARED:
                    a.start(h)
            states = (a.state(self._cap), a.state(self._play))
            if states != (SND_PCM_STATE_RUNNING, SND_PCM_STATE_RUNNING):
                raise HilError(ENV, f"the duplex stream on {self.pcm} did not start "
                                    f"(capture state {states[0]}, playback state {states[1]})",
                               hint="snd_pcm_state: 2 prepared, 3 running, 4 xrun")
        except BaseException:
            self._close_handles()
            raise
        self._t_start = time.perf_counter()
        self.started_at = self._clock()
        self._thread = threading.Thread(target=self._run, name="duplex", daemon=True)
        self._thread.start()

    def _busy_or(self, e: AlsaError) -> HilError:
        if e.errno == EBUSY:
            return HilError(ENV, f"the CrossPad card ({self.pcm}) is busy",
                            hint="something else holds it -- on the bench the "
                                 "crosspad-audio-nodes service: hold the bench lease "
                                 "(crosspad-hil bench claim) so the run can pause it")
        return e

    def stop(self, timeout_s: float = 10.0) -> None:
        if self._stopped:
            return
        self._stopped = True
        self._halt.set()
        if self._thread is not None:
            self._thread.join(timeout_s)
        self._wall = time.perf_counter() - self._t_start if self._t_start else 0.0
        self._close_handles()

    def _close_handles(self) -> None:
        for h in (self._play, self._cap):
            if h is not None:
                try:
                    self._alsa.drop(h)
                finally:
                    self._alsa.close(h)
        self._play = self._cap = None
        if self._wav is not None:
            self._wav.close()
            self._wav = None

    def wait(self, timeout_s: float | None = None) -> bool:
        return False            # unbounded: it runs until stop()

    # -- what the scenario uses ------------------------------------------------
    def out_position(self) -> int:
        """Output frames handed to the card so far; a burst must be scheduled past this."""
        return self.frames_out

    @property
    def schedule_lead_frames(self) -> int:
        """How far past ``out_position()`` a burst is safe: one buffer and a period."""
        return int(self.info.get("playback_buffer", 0)) + 2 * int(self.info.get(
            "playback_period", 0)) + self.sample_rate // 100

    def schedule(self, frame: int, samples: Sequence[int]) -> None:
        """Play ``samples`` (int16, the same on every channel) from output frame ``frame``."""
        with self._lock:
            self._bursts.append((int(frame), [int(s) for s in samples]))
            self._bursts.sort(key=lambda b: b[0])

    def capture_points(self) -> list[tuple[int, float]]:
        with self._lock:
            if self._cap_log is not None:
                return self._cap_log.snapshot()
            return list(self._cap_pts)

    def playback_points(self) -> list[tuple[int, float]]:
        with self._lock:
            if self._play_log is not None:
                return self._play_log.snapshot()
            return list(self._play_pts)

    def pause_out(self, seconds: float, mode: str = "drop") -> None:
        """The phone stops its stream for ``seconds``: ``drop`` stops the playback PCM
        (unlinked from the capture first) and starts it again; ``silence`` keeps it
        running on digital zeros. Done by the I/O thread; ``events`` records both ends."""
        if mode not in ("drop", "silence"):
            raise HilError(ENV, f"pause mode {mode!r}: drop or silence")
        with self._lock:
            self._pause_req = (mode, float(seconds))

    def health(self) -> dict[str, Any]:
        got = self.frames_in / float(self.sample_rate)
        missed = max(0.0, self._wall - got)
        flags: list[str] = []
        if self.in_xruns:
            flags.append(f"{self.in_xruns} capture xrun(s)")
        if self.out_xruns:
            flags.append(f"{self.out_xruns} playback xrun(s)")
        if self.max_gap_s > 0.5:
            flags.append(f"{self.max_gap_s * 1000:.0f} ms loop gap")
        if missed > 0.25:
            flags.append(f"{missed:.1f} s never arrived")
        if self.error is not None:
            flags.append(f"stream error: {self.error}")
        return {"ok": not flags, "captured_s": round(got, 3), "wall_s": round(self._wall, 3),
                "overruns": self.in_xruns, "out_xruns": self.out_xruns,
                "events": list(self.events),
                "max_gap_ms": round(self.max_gap_s * 1000, 1), "missed_s": round(missed, 3),
                "exclusive": True, "flags": flags,
                "error": None if self.error is None else repr(self.error),
                "detail": f"captured {got:.1f} s of {self._wall:.1f} s"
                          + (" - " + ", ".join(flags) if flags else " - clean")}

    # -- the I/O thread --------------------------------------------------------
    def _render(self, n0: int, n: int) -> bytes:
        muted = self._mute_until is not None and self._clock() < self._mute_until
        if self._source is not None and not muted:
            out = array.array("h")
            out.frombytes(self._source(n0, n))
        else:
            out = array.array("h", bytes(n * self.channels * BYTES_PER_SAMPLE))
        with self._lock:
            # Bursts that ended before this chunk are done with.
            self._bursts = [b for b in self._bursts if b[0] + len(b[1]) > n0]
            live = [b for b in self._bursts if b[0] < n0 + n]
        for start, samples in live:
            for j in range(max(start, n0), min(start + len(samples), n0 + n)):
                v = samples[j - start]
                base = (j - n0) * self.channels
                for c in range(self.channels):
                    out[base + c] = v
        return out.tobytes()

    def _write_out(self, frames: int) -> int:
        data = self._render(self.frames_out, frames)
        w = self._alsa.writei(self._play, data, self.channels)
        self.frames_out += w
        return w

    def _log(self, pts: list[tuple[int, float]], frame: int, t: float) -> None:
        dec = self._cap_log if pts is self._cap_pts else self._play_log
        if dec is not None:
            with self._lock:
                dec.add(frame, t)
            return
        if len(pts) < MAX_POINTS:
            with self._lock:
                pts.append((frame, t))

    def _event(self, kind: str, **kw: Any) -> None:
        self.events.append({"kind": kind, "t": self._clock(), "frames_in": self.frames_in,
                            "frames_out": self.frames_out, **kw})

    def _pause_step(self) -> bool:
        """Handle a requested output pause/resume; True while the output is stopped."""
        a = self._alsa
        req = None
        with self._lock:
            req, self._pause_req = self._pause_req, None
        now = self._clock()
        if req is not None:
            mode, seconds = req
            unlink = getattr(a, "unlink", None)
            if mode == "drop" and self._linked and not (unlink and unlink(self._play)):
                self._event("out_pause_fallback", why="snd_pcm_unlink not available")
                mode = "silence"
            if mode == "drop":
                self._linked = False
                a.drop(self._play)
                self._out_paused_until = now + seconds
                self._event("out_stop", mode="drop", seconds=seconds)
            else:
                self._mute_until = now + seconds
                self._event("out_mute", mode="silence", seconds=seconds)
        if self._out_paused_until is not None:
            if now < self._out_paused_until:
                return True
            self._out_paused_until = None
            a.prepare(self._play)
            self._write_out(int(self._pbuf))
            if a.state(self._play) == SND_PCM_STATE_PREPARED:
                a.start(self._play)
            self._event("out_start")
        if self._mute_until is not None and now >= self._mute_until:
            self._mute_until = None
            self._event("out_unmute")
        return False

    def _xrun(self, h: Any, e: AlsaError, side: str) -> bool:
        if e.errno not in (EPIPE, ESTRPIPE):
            return False
        self._event("xrun", side=side)
        if side == "in":
            self.in_xruns += 1
        else:
            self.out_xruns += 1
        self._alsa.recover(h, e.errno)
        if side == "out":
            self._write_out(int(self._pbuf))
        if self._alsa.state(h) == SND_PCM_STATE_PREPARED:
            self._alsa.start(h)
        return True

    def _run(self) -> None:
        a = self._alsa
        last = time.perf_counter()
        try:
            while not self._halt.is_set():
                busy = False
                try:
                    avail, delay = a.avail_delay(self._cap)
                    t = self._clock()
                    self._log(self._cap_pts, self.frames_in + delay, t)
                    if avail >= self._chunk:
                        data = a.readi(self._cap, avail, self.channels)
                        n = len(data) // (self.channels * BYTES_PER_SAMPLE)
                        if n:
                            self._wav.writeframes(data)
                            self.frames_in += n
                            busy = True
                except AlsaError as e:
                    if not self._xrun(self._cap, e, "in"):
                        raise
                    continue
                if self._pause_step():
                    time.sleep(IDLE_SLEEP_S)
                    continue
                try:
                    avail_p, delay_p = a.avail_delay(self._play)
                    t = self._clock()
                    self._log(self._play_pts, self.frames_out - delay_p, t)
                    if avail_p >= self._chunk:
                        busy = self._write_out(avail_p) > 0 or busy
                except AlsaError as e:
                    if not self._xrun(self._play, e, "out"):
                        raise
                    continue
                now = time.perf_counter()
                if busy:
                    self.max_gap_s = max(self.max_gap_s, now - last)
                    last = now
                else:
                    time.sleep(IDLE_SLEEP_S)
        except BaseException as exc:  # noqa: BLE001 - reported through health(), not lost
            self.error = exc


class AlsaDuplexBackend:
    """`capture.AudioBackend` for a duplex take: the card from `arecord -l`, the
    stream through libasound. ``recording`` is the last one started."""

    name = "alsa-duplex"

    def __init__(self, *, latency_ms: float = DEFAULT_LATENCY_MS, alsa: Any | None = None,
                 finder: Any | None = None,
                 source_factory: Callable[[int, int], Callable[[int, int], bytes]] | None = None,
                 decimate_s: float | None = None) -> None:
        self._latency_ms = latency_ms
        self._alsa = alsa
        self._finder = finder or ArecordBackend()
        # ``source_factory(rate, channels) -> source``: the signal the playback
        # side streams (the jam's pilot), built once the card's rate is known.
        self._source_factory = source_factory
        self._decimate_s = decimate_s
        self.recording: DuplexRecording | None = None

    def find_capture(self, device: Device, *, timeout_s: float = 15.0) -> CaptureDevice:
        return self._finder.find_capture(device, timeout_s=timeout_s)

    def start(self, cap: CaptureDevice, path: Path, *, seconds: float | None,
              sample_rate: int, channels: int) -> DuplexRecording:
        if not cap.alsa_id:
            raise HilError(ENV, f"no ALSA card id for {cap.name}",
                           hint="the duplex stream opens plughw:<card>; "
                                "`crosspad-hil devices` should list it")
        alsa = self._alsa if self._alsa is not None else AlsaLib()
        source = (self._source_factory(int(sample_rate), int(channels))
                  if self._source_factory is not None else None)
        rec = DuplexRecording(alsa, f"plughw:{cap.alsa_id}", path, rate=sample_rate,
                              channels=channels, latency_ms=self._latency_ms, source=source,
                              decimate_s=self._decimate_s)
        rec.start()
        self.recording = rec
        return rec
