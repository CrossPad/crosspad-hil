"""The full live setup, the way the 10.10 release is accepted: fast and without a glitch.

A phone on the USB-C port streams into the board and listens to what it sends
back, a sequencer scene runs at 120 BPM, someone plays the pads and builds a
beat, bounces it, swaps kits. Every counter that can see a glitch has to stay
at zero, for the whole set.

The set runs in phases, one per playing style:

    one      one pad retriggered fast (--one-rate, 12 hits/s)
    four     at most four pads, alternating
    many     all sixteen pads round-robin, plus a chord every few seconds
    pitched  the same on the pitched kit (HIL PITCHED)

Each phase is a **bounce window** in the default profile followed by a **live
block** in the USB-audio profile:

* bounce window -- record the phase's pads into a scene of its own (SEQ, 120
  BPM), bounce it while the pads keep playing, check the bounce completed and
  that the file is non-silent and click-free, restart the scene;
* live block (``--block-s``) -- the host plays a continuous tone into the board
  (``pw-play`` on ``crosspad_out``, one stream: every restart would count as an
  underrun) and records the capture endpoint *with the product's routing*
  (preset ``asis``: what a phone would receive), the pads play the phase's
  pattern over SysEx, the kit is swapped every ``--swap-every-s`` (A/B, and
  pitched <-> sampler in the pitched phase), the console's ``SPK #`` lines carry
  the USB bridge's counters;
* back in the default profile -- KIT_SWAP_LOG rows of the phase, SMPL_PEAK's
  engine counters, PAD_STATS, MEM, the reset log, and whether the scene was
  still playing.

Why alternate rather than run all of it in the audio profile: ``ASONG`` (bounce)
and ``SEQ`` answer only over CDC, which the audio profile does not have, and a
bounce stops the sequencer (``SongEngine::restore`` = ``seq.stop()``) until a
``SEQ PLAY``. A user does the same from the device's own UI; that path is not
reachable from the host.

The capture is USB IN with the board's mixer as shipped (preset ``asis``):
whatever the mixer sends there is what a phone receives -- on a live stream the
viewers heard exactly what the speakers played. The pads and the scene have to
be audible in it: every pad struck in a 10-s slice has to show up at its own
frequency (the fixture kits give every pad a pure tone, and the kit in force
at that moment decides which), and a slice with nothing in it while the set
plays is a hole. Whether the host's stream reaches USB IN depends on how the
mixer routes the USB channel; it is reported, not required.

A bounce stops the sequencer by design; every bounce window restarts it.

Before anything is measured the bench is normalised (default profile, any pin
an unfinished run left in the journal put back, host gain at 0 dB on every
entry into the audio profile); afterwards everything the run changed is put
back in a ``finally``: scenes, count-in, kit, app, profile.
"""
from __future__ import annotations

import math
import shutil
import struct
import subprocess
import threading
import time
import wave
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from crosspad_hil import card_hold, fixtures, fs, pins, verbs
from crosspad_hil import host_gain as hostgain
from crosspad_hil.audio_nodes import OUT_NODE, _pipewire_node_names
from crosspad_hil.cdc import CdcLink
from crosspad_hil.console import Console
from crosspad_hil.devices import Device, UsbMode
from crosspad_hil.errors import (
    BAD_ARGS,
    CANCELLED,
    ENV,
    NO_CDC_IN_AUDIO_MODE,
    NOT_SUPPORTED,
    TIMEOUT,
    HilError,
)
from crosspad_hil.midi import sysex_kit_load
from crosspad_hil.scenarios import _audio, _deps
from crosspad_hil.scenarios._audio import HOST_LOSS_S, host_capture_verdict  # noqa: F401
from crosspad_hil.scenarios.base import Artifact, Context, Progress, Report, register
from crosspad_hil.usbmode import set_mode

SAMPLER_APP = "Sampler"
PROJECT = "hil_live_set"
PHASES: tuple[str, ...] = ("one", "four", "many", "pitched")
KIT_A, KIT_B, KIT_P = "HIL SINE A", "HIL SINE B", fixtures.PITCHED_NAME

BLOCK_US = 1451               # one 64-frame block at 44.1 kHz
# The bench's acceptance (2026-10): a swap may cost the RT loop up to two
# blocks, as long as the engine never underran and the take has no click --
# one block failed swaps that played clean.
RT_GAP_PASS_US = 2 * BLOCK_US
# A hit that lands on a pad while its sample reloads is held and played when
# the pad is back (the firmware's late_hits). At most one per swap, and none
# later than this -- in the audio profile too, no allowance for UAC.
LATE_HITS_PER_SWAP = 1
LATE_MAX_MS = 50.0
# The pitched engine's voice limit: of a 16-note chord on the pitched kit the
# first 6 pressed are stolen (a 5 ms fade), so only the last 10 must sound.
PITCHED_VOICES = 10
SETUP_CODES = frozenset({ENV, NOT_SUPPORTED, TIMEOUT, NO_CDC_IN_AUDIO_MODE})
STREAM_SETTLE_S = 2.0
BOUNCE_TIMEOUT_S = 60.0
POLL_S = 0.25
SLICE_S = 10.0
DROPOUT_FRAMES = 64           # a run of digital zeros this long inside sound is a hole
# ...but an underrun in a bounced or captured take rarely leaves exact zeros:
# a 57 ms hole in a bounce kept ~20 LSB of noise and scored "0 dropouts". So a
# hole is also the envelope falling under HOLE_FRAC of the take's level for
# HOLE_MIN_S, measured in HOLE_WIN_S RMS windows.
HOLE_WIN_S = 0.005
HUNG_TAIL_LINES = 40          # console lines kept in the report when the board hangs
HOLE_FRAC = 0.10
HOLE_MIN_S = 0.010
SILENT_DBFS = -60.0
TONE_HZ = 441.0
TONE_AMPLITUDE = 3000

OPEN_READY, BOUNCE_DONE, BOUNCE_FAILED = 2, 2, 3
# SequencerState (crosspad-core SequencerTypes.hpp): PreBeat 3 is the last
# beat before the downbeat, Running 4; anything else is not playing.
SEQ_PLAYING = (3, 4)
# A kit load saves a dirty sequence to the card first, and the CDC loop waits
# for that write: SEQ/ASONG must not give up after the usual 2 s (seen: no reply
# to SEQ PLAY right after KIT_LOAD of the pitched kit).
SEQ_TIMEOUT_S = 10.0


def _seq(link: CdcLink, action: str | None = None) -> dict[str, Any]:
    return verbs.seq(link, action, timeout_s=SEQ_TIMEOUT_S)


def _asong(link: CdcLink, action: str | None = None) -> dict[str, Any]:
    return verbs.asong(link, action, timeout_s=SEQ_TIMEOUT_S)


@dataclass
class Params:
    phases: list[str] = field(default_factory=lambda: list(PHASES),
                              metadata={"help": "phases to run, in order: one,four,many,pitched"})
    block_s: float = field(default=300.0,
                           metadata={"help": "seconds of each live block in the audio profile"})
    swap_every_s: float = field(default=90.0,
                                metadata={"help": "seconds between kit swaps inside a block"})
    one_rate: float = field(default=12.0, metadata={"help": "phase one: hits/s on one pad"})
    four_rate: float = field(default=8.0, metadata={"help": "phase four: hits/s over 4 pads"})
    many_rate: float = field(default=16.0, metadata={"help": "phases many/pitched: hits/s"})
    chord_every_s: float = field(default=4.0,
                                 metadata={"help": "phases many/pitched: a chord this often"})
    bpm: int = field(default=120, metadata={"help": "sequencer tempo"})
    stream: bool = field(default=True,
                         metadata={"help": "host plays a continuous tone into the board"})
    capture: bool = field(default=True,
                          metadata={"help": "record the capture endpoint over every block"})
    bounce: bool = field(default=True, metadata={"help": "bounce a scene in every phase"})
    stream_target: str = field(default=OUT_NODE,
                               metadata={"help": "PipeWire node the host stream plays to"})
    no_provision: bool = field(default=False,
                               metadata={"help": "fail instead of pushing missing fixture kits"})
    enum_timeout: float = field(default=45.0,
                                metadata={"help": "seconds for a USB profile switch"})
    exclusive_card: bool = field(
        default=True,
        metadata={"help": "pause the audio-nodes service for each live block and record / "
                          "stream straight on the card's ALSA device (through PipeWire the "
                          "capture lost seconds under load)"})
    reset_first: bool = field(
        default=True,
        metadata={"help": "reset the board first: a clean boot, and its log says how the SD "
                          "card came up (bus width, clock, a 1-bit retry)"})


# ------------------------------------------------------------------ seams (tests replace these)

def open_capture(device: Any, **kw: Any) -> Any:
    return _deps.module("capture").CaptureSession(device, **kw)


def open_pad_transport(device: Any, kind: str) -> Any:
    return _deps.module("stim").open_transport(device, kind)


def pull_file(ctx: Context, remote: str, local: Path) -> dict[str, Any]:
    return fs.pull(ctx.device, remote, local, serial_factory=ctx.serial_factory)


class HostStream:
    """``pw-play`` of one long file: the phone's stream, never restarted mid-block."""

    def __init__(self, argv: list[str]) -> None:
        self.argv = argv
        self.proc = subprocess.Popen(argv, stdout=subprocess.DEVNULL,
                                     stderr=subprocess.PIPE, text=True)

    def alive(self) -> bool:
        return self.proc.poll() is None

    hung = False      # did not exit even on SIGKILL: stuck on a card that stopped answering

    def stop(self) -> str:
        if self.proc.poll() is None:
            self.proc.terminate()
            try:
                self.proc.wait(timeout=3.0)
            except subprocess.TimeoutExpired:
                self.proc.kill()
                try:
                    self.proc.wait(timeout=3.0)
                except subprocess.TimeoutExpired:
                    self.hung = True      # its stderr would block a read forever
                    return f"{self.argv[0]} did not exit even on SIGKILL"
        return ((self.proc.stderr.read() if self.proc.stderr else "") or "").strip()


def start_direct_stream(wav: Path, card: str | None, log: Any) -> Any:
    """``aplay`` on the card's own ALSA device: the stream when the card is ours alone."""
    if card is None:
        raise HilError(ENV, "no ALSA card for the CrossPad to stream to")
    if shutil.which("aplay") is None:
        raise HilError(ENV, "aplay is not installed: no host stream",
                       hint="install alsa-utils, or run with --no-stream")
    log(f"host stream: aplay -D plughw:{card} {wav.name}")
    return HostStream(["aplay", "-q", "-D", f"plughw:{card}", str(wav)])


# More of the capture than this missing is the host's failure, not a measurement.
def start_host_stream(wav: Path, target: str, log: Any, wait_s: float = 15.0) -> Any:
    if shutil.which("pw-play") is None:
        raise HilError(ENV, "pw-play is not installed: no host stream",
                       hint="install pipewire (pw-play), or run with --no-stream")
    deadline = time.monotonic() + wait_s
    sink = None
    while sink is None:
        names = _pipewire_node_names()
        sink = target if target in names else next(
            (n for n in sorted(names) if n.startswith("alsa_output") and "crosspad" in n.lower()),
            None)
        if sink is not None or time.monotonic() >= deadline:
            break
        time.sleep(0.3)
    if sink is None:
        raise HilError(ENV, f"no PipeWire sink for the CrossPad within {wait_s:.0f}s",
                       hint="is PipeWire running, and the board in the audio profile?")
    log(f"host stream: pw-play --target {sink} {wav.name}")
    return HostStream(["pw-play", "--target", sink, str(wav)])


# ------------------------------------------------------------------ pure parts

def write_tone(path: Path, seconds: float) -> Path:
    """A stereo 16-bit 441 Hz sine (10 whole periods repeated, so it is seamless)."""
    rate = 44_100
    period = rate / TONE_HZ
    one = b"".join(struct.pack("<hh", s, s) for s in
                   (int(TONE_AMPLITUDE * math.sin(2 * math.pi * i / period))
                    for i in range(int(round(period * 10)))))
    frames = int(seconds * rate)
    reps, rest = divmod(frames, len(one) // 4)
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as w:
        w.setnchannels(2)
        w.setsampwidth(2)
        w.setframerate(rate)
        w.writeframes(one * reps + one[:rest * 4])
    return path


def _spk_rows(lines: list[str]) -> list[dict[str, int]]:
    rows = []
    for line in lines:
        i = line.find("SPK #")
        if i < 0:
            continue
        kv = {}
        for tok in line[i:].replace(":", " ").split():
            k, sep, v = tok.partition("=")
            if sep:
                try:
                    kv[k] = int(v)
                except ValueError:
                    pass
        rows.append(kv)
    return rows


def parse_spk(lines: list[str], before: list[str] | None = None) -> dict[str, Any]:
    """The USB bridge's counters for one block, from the console's ``SPK #`` lines.

    ``SPK #<n>: len= taken= ring= under= refused= cap_ring= cap_drop= [ring_min=]``.
    ``under``/``refused`` restart at every entry into the audio profile, so the
    block's last line holds its totals. ``cap_drop`` counts from boot (33 764
    all day on the bench): the block's share is the difference from the last
    line before the block (``before``), or from the block's own first line.
    """
    rows = _spk_rows(lines)
    if not rows:
        return {"lines": 0}
    last = rows[-1]
    prior = _spk_rows(before or [])
    base = prior[-1] if prior else rows[0]
    drop0 = base.get("cap_drop", 0)
    drop = last.get("cap_drop", 0)
    out: dict[str, Any] = {"lines": len(rows), "under": last.get("under", 0),
                           "refused": last.get("refused", 0),
                           "cap_drop": drop - drop0 if drop >= drop0 else drop,
                           "cap_drop_total": drop}
    # The host stream stopped before the profile switch: the block's last line
    # often carries the ring running dry (ring=0, a short packet, under+1).
    # That is the end of the stream, not a glitch -- unless the lines while
    # the stream ran already had under.
    if len(rows) >= 2 and _end_of_stream(rows):
        during = rows[-2].get("under", 0)
        out["end_of_stream_under"] = out["under"] - during
        out["under"] = during
    mins = [r["ring_min"] for r in rows if "ring_min" in r]
    if mins:
        out["ring_min"] = min(mins)
    return out


def _end_of_stream(rows: list[dict[str, int]]) -> bool:
    """The last SPK line is the stream running out: more under than the line before,
    the ring empty, and a packet shorter than every one while the stream ran."""
    last, prev = rows[-1], rows[-2]
    lens = [r["len"] for r in rows[:-1] if r.get("len")]
    return (last.get("under", 0) > prev.get("under", 0) and last.get("ring", -1) == 0
            and bool(lens) and 0 <= last.get("len", 0) < min(lens))


def parse_sd_boot(lines: list[str]) -> dict[str, Any]:
    """How the SD card came up, from the boot log's ``bsp_sdmmc`` lines."""
    import re

    sd: dict[str, Any] = {"seen": False, "width": None, "max_freq_khz": None,
                          "retried_1bit": False, "mounted": None}
    for line in lines:
        if "bsp_sdmmc" not in line:
            continue
        sd["seen"] = True
        m = re.search(r"max_freq=(\d+)kHz", line)
        if m:
            sd["max_freq_khz"] = int(m.group(1))
        m = re.search(r"pin map:.*width=(\d+)", line)
        if m and not sd["retried_1bit"]:
            sd["width"] = int(m.group(1))
        if "retrying 1-bit" in line:
            sd["retried_1bit"] = True
            sd["width"] = 1
        if "SD Card mounted successfully" in line:
            sd["mounted"] = True
        elif "SD Card mount failed" in line:
            sd["mounted"] = False
    return sd


def pattern(phase: str, tick: int, *, chord_every: int) -> list[int]:
    """The pads one tick of a phase presses together."""
    if phase == "one":
        return [0]
    if phase == "four":
        group = (tick // 16) % 4          # move to another four-pad group now and then
        return [group * 4 + tick % 4]
    if chord_every and tick % chord_every == chord_every - 1:
        return list(range(0, 16, 2)) if (tick // chord_every) % 2 else list(range(16))
    return [tick % 16]


def bounce_pads(phase: str) -> list[int]:
    """What the phase's bounced scene holds: one sample, four, or many."""
    return {"one": [0, 0, 0, 0], "four": [0, 1, 2, 3]}.get(phase, list(range(16)))


def phase_kits(phase: str) -> list[str]:
    """The kit a phase starts on, then what its swaps alternate with."""
    return [KIT_P, KIT_A] if phase == "pitched" else [KIT_A, KIT_B]


AUDIBLE_SNR_DB = 12.0        # a pad's tone this far above the slice's floor
RELATIVE_FLOOR_DB = 40.0     # ...and the floor is never more than this under the loudest tone
BAND_HZ = 12.0


def pad_freqs(kit: str, pad: int) -> float:
    """The pure tone a fixture kit gives a pad."""
    if kit == KIT_A:
        return fixtures.freq_a(pad)
    if kit == KIT_B:
        return fixtures.freq_b(pad)
    return fixtures.note_to_hz(fixtures.PITCHED_BASE_NOTE + pad)


def expected_in(hits: list[tuple[float, int]], kits: list[tuple[float, str]],
                t0: float, t1: float) -> dict[int, list[float]]:
    """``{pad: [candidate Hz]}`` for the pads struck in ``[t0, t1)``: the tone of the
    kit in force when it was struck (and the one before a swap, for a voice that
    was still sounding). A chord note the pitched engine stole a voice from (the
    first ones pressed of a chord larger than PITCHED_VOICES) is not expected."""
    out: dict[int, set[float]] = {}
    chords: dict[float, list[int]] = {}
    for t, pad in hits:
        chords.setdefault(t, []).append(pad)
    for t, pad in hits:
        if not t0 <= t < t1:
            continue
        current = [k for ts, k in kits if ts <= t]
        group = chords[t]
        if current and current[-1] == KIT_P and len(group) > PITCHED_VOICES \
                and group.index(pad) < len(group) - PITCHED_VOICES:
            continue
        cands = out.setdefault(pad, set())
        for kit in current[-2:]:
            cands.add(round(pad_freqs(kit, pad), 2))
    return {p: sorted(c) for p, c in out.items()}


class EnvelopeHoles:
    """Holes in the envelope of a take, fed one slice at a time.

    A slice is cut into HOLE_WIN_S windows; its threshold is HOLE_FRAC of the
    median window RMS between the slice's first and last loud window (loud =
    over half the median of the non-zero windows). A hole is a run of windows
    under the threshold, at least HOLE_MIN_S long, that sound closes: the
    take's leading silence (before anything was loud) and its trailing silence
    (never closed) are not holes. A run may span slices.
    """

    def __init__(self, np: Any) -> None:
        self._np = np
        self.holes: list[tuple[float, float]] = []      # (start s, length ms)
        self._started = False
        self._run_from: float | None = None

    def feed(self, x: Any, sr: int, t0: float) -> list[tuple[float, float]]:
        """Windows of slice ``x`` (starting at ``t0`` s); the holes it closed."""
        np = self._np
        win = max(1, int(sr * HOLE_WIN_S))
        m = len(x) // win
        if m == 0:
            return []
        rms = np.sqrt(np.mean(x[: m * win].reshape(m, win) ** 2, axis=1))
        nz = rms[rms > 0]
        if not len(nz):
            self._run_from = None
            return []
        loud = np.flatnonzero(rms > 0.5 * float(np.median(nz)))
        if not len(loud):
            return []
        thr = HOLE_FRAC * float(np.median(rms[loud[0]:loud[-1] + 1]))
        closed: list[tuple[float, float]] = []
        dt = win / float(sr)
        for i, below in enumerate(rms < thr):
            t = t0 + i * dt
            if not self._started:
                self._started = not below
                continue
            if below:
                if self._run_from is None:
                    self._run_from = t
            elif self._run_from is not None:
                if t - self._run_from >= HOLE_MIN_S - 1e-9:
                    closed.append((round(self._run_from, 3),
                                   round((t - self._run_from) * 1000.0, 1)))
                self._run_from = None
        self.holes += closed
        return closed

    def drop_run(self) -> None:
        """A silent slice is reported on its own; a run does not carry over it."""
        self._run_from = None


def score_take(path: Path, *, slice_s: float = SLICE_S,
               expect: Any = None) -> dict[str, Any]:
    """Clicks, holes and (with ``expect(t0, t1) -> {pad: [Hz]}``) inaudible pads over a
    long capture, one slice at a time (read whole, 30 min is >1 GB of float64).

    ``dropouts`` are runs of exact digital zero; ``holes`` are envelope holes
    (EnvelopeHoles), which an underrun's leftover noise does not hide."""
    np = _deps.module("analyze")._np()
    click_metric = _deps.module("analyze").click_metric
    out: dict[str, Any] = {"slices": 0, "clicks": 0, "dropouts": 0, "holes": 0,
                           "hole_max_ms": 0.0, "silent_slices": 0,
                           "inaudible": 0, "peak_dbfs": None, "worst": []}
    envelope = EnvelopeHoles(np)
    peak = 0.0
    with wave.open(str(path), "rb") as w:
        sr, ch, width = w.getframerate(), w.getnchannels(), w.getsampwidth()
        if width != 2:
            raise HilError(ENV, f"capture is {8 * width}-bit; the scorer reads 16-bit")
        n = int(sr * slice_s)
        t = 0.0
        while True:
            raw = w.readframes(n)
            if not raw:
                break
            x = np.frombuffer(raw, dtype="<i2").reshape(-1, ch).astype(np.float64) / 32768.0
            mono = x[:, int(np.argmax((x ** 2).sum(axis=0)))]
            out["slices"] += 1
            rms = float(np.sqrt(np.mean(mono ** 2))) if len(mono) else 0.0
            p = float(np.max(np.abs(mono))) if len(mono) else 0.0
            peak = max(peak, p)
            if rms == 0.0 or 20 * math.log10(max(rms, 1e-12)) < SILENT_DBFS:
                out["silent_slices"] += 1
                envelope.drop_run()
            else:
                clicks, _ = click_metric(mono, sr)
                holes = _zero_runs(np, mono, DROPOUT_FRAMES)
                gaps = envelope.feed(mono, sr, t)
                missing = _inaudible(np, mono, sr, expect(t, t + len(mono) / sr)) \
                    if expect is not None else []
                out["clicks"] += int(clicks)
                out["dropouts"] += holes
                out["inaudible"] += len(missing)
                if clicks or holes or gaps or missing:
                    out["worst"].append({"t": round(t, 1), "clicks": int(clicks),
                                         "dropouts": holes, "inaudible_pads": missing,
                                         **({"holes": [{"t": s, "ms": ms} for s, ms in gaps]}
                                            if gaps else {})})
            t += len(mono) / float(sr)
    out["seconds"] = round(t, 1)
    out["peak_dbfs"] = round(20 * math.log10(peak), 1) if peak > 0 else None
    out["holes"] = len(envelope.holes)
    out["hole_max_ms"] = max((ms for _s, ms in envelope.holes), default=0.0)
    out["worst"] = out["worst"][:20]
    return out


def _inaudible(np: Any, x: Any, sr: int, expected: dict[int, list[float]]) -> list[int]:
    """Pads whose tone is not ``AUDIBLE_SNR_DB`` above the slice's median bin."""
    if not expected or len(x) < sr // 10:
        return []
    spec = np.abs(np.fft.rfft(x * np.hanning(len(x))))
    hz = np.fft.rfftfreq(len(x), 1.0 / sr)
    # The slice's own noise, but never below 40 dB under its loudest tone: a
    # clean take has a median near zero, and splatter would then pass as a tone.
    floor = max(float(np.median(spec)), float(spec.max()) * 10 ** (-RELATIVE_FLOOR_DB / 20),
                1e-12)
    missing = []
    for pad, cands in sorted(expected.items()):
        best = 0.0
        for f in cands:
            band = spec[(hz >= f - BAND_HZ) & (hz <= f + BAND_HZ)]
            if len(band):
                best = max(best, float(band.max()))
        if 20 * math.log10(max(best, 1e-12) / floor) < AUDIBLE_SNR_DB:
            missing.append(pad)
    return missing


def _zero_runs(np: Any, x: Any, min_len: int) -> int:
    """Runs of exact digital zero at least ``min_len`` long, not at either edge."""
    z = np.concatenate(([0], (x == 0.0).astype(np.int8), [0]))
    d = np.diff(z)
    starts, ends = np.flatnonzero(d == 1), np.flatnonzero(d == -1)
    lengths = ends - starts
    inner = (starts > 0) & (ends < len(x))
    return int(np.count_nonzero((lengths >= min_len) & inner))


def phase_problems(ph: dict[str, Any]) -> list[str]:
    """Zero tolerance: every counter that saw a glitch, by name and value."""
    out: list[str] = []
    hung = ph.get("hung")
    if hung:
        out.append(f"board hung in UAC block at t={hung['t_s']:.0f} s ({hung['reason']})")
    spk = ph.get("spk") or {}
    for key in ("under", "refused", "cap_drop"):
        if spk.get(key):
            out.append(f"usb_audio_bridge {key}={spk[key]}")
    eng = ph.get("engine") or {}
    for key in ("underrun", "noblock"):
        if (eng.get(key) or 0) > 0:
            out.append(f"engine {key} +{eng[key]}")
    swaps = ph.get("swaps") or {}
    for key in ("underruns", "noblock", "dropped_hits"):
        if swaps.get(key):
            out.append(f"kit swap {key}={swaps[key]}")
    if (swaps.get("late_hits_max") or 0) > LATE_HITS_PER_SWAP:
        out.append(f"kit swap late_hits={swaps['late_hits_max']} in one swap "
                   f"(> {LATE_HITS_PER_SWAP})")
    late_ms = (ph.get("late") or {}).get("late_max_ms")
    if late_ms is not None and late_ms > LATE_MAX_MS:
        out.append(f"a held hit sounded {late_ms:.0f} ms late (> {LATE_MAX_MS:.0f} ms)")
    elif late_ms is None and swaps.get("late_hits"):
        out.append(f"{swaps['late_hits']} late hit(s) the host could not time: "
                   + ((ph.get("late") or {}).get("why") or "no timing"))
    gap = swaps.get("rt_gap_max_us") or 0
    clicks = 0 if ph.get("capture_untrusted") else int((ph.get("take") or {}).get("clicks") or 0)
    if gap > RT_GAP_PASS_US:
        out.append(f"rt_gap_max_us={gap} > two blocks ({RT_GAP_PASS_US})")
    elif gap > BLOCK_US and (swaps.get("underruns") or clicks
                             or ((ph.get("engine") or {}).get("underrun") or 0) > 0):
        out.append(f"rt_gap_max_us={gap} > one block ({BLOCK_US}) with an underrun or a click")
    for key in ("inj_drop", "sbuf_drop", "usb_drop", "uart_ovf"):
        if ((ph.get("pads") or {}).get(key) or 0) > 0:
            out.append(f"pad path {key} +{ph['pads'][key]}")
    take = {} if ph.get("capture_untrusted") else (ph.get("take") or {})
    if take.get("inaudible"):
        out.append(f"capture: {take['inaudible']} pad hit(s) inaudible on USB IN")
    if take.get("silent_slices"):
        out.append(f"capture: {take['silent_slices']} silent 10-s slice(s) while the set played")
    if take.get("clicks"):
        out.append(f"capture: {take['clicks']} click(s)")
    if take.get("dropouts"):
        out.append(f"capture: {take['dropouts']} dropout(s)")
    if take.get("holes"):
        out.append(f"capture: {take['holes']} hole(s) of {HOLE_MIN_S * 1000:.0f} ms or more "
                   f"(longest {take['hole_max_ms']:.0f} ms)")
    if not ph.get("capture_untrusted"):
        for p in ph.get("capture_problems") or []:
            out.append(f"capture: {p}")
    b = ph.get("bounce")
    if b is not None and not b.get("ok"):
        out.append(f"bounce: {b.get('why')}")
    if ph.get("seq_stopped"):
        out.append("the scene was not playing after the live block")
    resets = (ph.get("resets") or 0) - (ph.get("recovery_resets") or 0)
    if resets > 0:
        out.append(f"{resets} reset(s)")
    if ph.get("fatal"):
        out.append("crash: " + "; ".join(ph["fatal"][:3]))
    for f in ph.get("failures") or []:
        out.append(f)
    return out


def _delta(after: dict[str, Any], before: dict[str, Any], keys: tuple[str, ...]) -> dict[str, int]:
    out = {}
    for k in keys:
        a, b = after.get(k), before.get(k)
        if isinstance(a, int) and isinstance(b, int):
            out[k] = a - b
    return out


def swap_totals(rows: list[dict[str, Any]]) -> dict[str, int]:
    out: dict[str, int] = {"swaps": len(rows)}
    for k in ("underruns", "noblock", "dropped_hits", "late_hits"):
        out[k] = sum(int(r.get(k) or 0) for r in rows)
    out["late_hits_max"] = max((int(r.get("late_hits") or 0) for r in rows), default=0)
    out["rt_gap_max_us"] = max((int(r.get("rt_gap_max_us") or 0) for r in rows), default=0)
    out["failed"] = sum(1 for r in rows if not r.get("ok", True))
    return out


def late_from_timestamps(rows: list[dict[str, Any]], swap_times: list[float],
                         hits: list[tuple[float, int]]) -> dict[str, Any]:
    """How late the held hits of each swap sounded, from the host's own clock.

    The firmware counts its late hits (a hit on a pad while that pad reloads is
    held and played when the pad is back) but does not time them. The swap row
    does say when each pad was back (``pads``: ms from the request, -1 =
    untouched); the pads reload one after another, so a pad's reload began when
    the previous one ended (or when parsing did). A hit the host sent inside
    that window waited until its end -- except on a pad the row marks ``deferred``,
    which waited for its voice to end first (hits meanwhile played the old
    sample, on time): its window is as long as the slowest pad that did not
    wait. Only swaps whose row counts a late hit are
    timed, and the worst candidate is taken: an estimate that errs late, never
    early. Host and device latencies are the same path for the swap request and
    the hits, so they cancel.
    """
    if len(rows) != len(swap_times):
        return {"late_max_ms": None, "per_swap": [],
                "why": f"{len(swap_times)} swap(s) sent, {len(rows)} logged: rows and "
                       "requests cannot be paired"}
    by_pad: dict[int, list[float]] = {}
    for t, pad in hits:
        by_pad.setdefault(pad, []).append(t)
    per: list[dict[str, Any]] = []
    for row, t_req in zip(rows, swap_times, strict=True):
        done = [int(d) for d in (row.get("pads") or []) if isinstance(d, int)]
        n_late = int(row.get("late_hits") or 0)
        entry: dict[str, Any] = {"seq": row.get("seq"), "late_hits": n_late, "late_max_ms": 0.0}
        if n_late:
            parse = float(row.get("parse_ms") or 0)
            deferred = int(row.get("deferred") or 0)
            ends = sorted(d for d in done if d >= 0)
            starts = {pad: max([parse] + [e for e in ends if e < end])
                      for pad, end in enumerate(done) if end >= 0}
            # A deferred pad waited for its voice to end before its reload began,
            # and a hit meanwhile played the old sample -- on time. Its reload is
            # taken to last as long as the slowest pad that did not wait.
            spans = [done[p] - starts[p] for p in starts if not deferred >> p & 1]
            worst, candidates = 0.0, 0
            for pad, end in enumerate(done):
                if end < 0:
                    continue
                if deferred >> pad & 1:
                    if not spans:
                        entry.update(late_max_ms=None, why="only deferred pads were reloaded: "
                                                            "their reload start is unknown")
                        break
                    start = max(starts[pad], end - max(spans))
                else:
                    start = starts[pad]
                for t in by_pad.get(pad, []):
                    rel = (t - t_req) * 1000.0
                    if start <= rel < end:
                        candidates += 1
                        worst = max(worst, end - rel)
            else:
                entry.update(late_max_ms=round(worst, 1), candidates=candidates)
        per.append(entry)
    untimed = [e for e in per if e["late_max_ms"] is None]
    if untimed:
        return {"late_max_ms": None, "per_swap": per, "estimated": True,
                "why": f"swap seq {untimed[0]['seq']}: {untimed[0]['why']}"}
    return {"late_max_ms": max((e["late_max_ms"] for e in per), default=0.0),
            "per_swap": per, "estimated": True}


# ------------------------------------------------------------------ the pad player

class PadPlayer(threading.Thread):
    """Plays a phase's pattern until stopped; counts what it sent and what was refused."""

    def __init__(self, transport: Any, phase: str, rate: float, chord_every_s: float) -> None:
        super().__init__(name=f"live_set_{phase}", daemon=True)
        self._t = transport
        self._phase = phase
        self._period = 1.0 / max(0.5, rate)
        self._chord_every = max(1, int(round(chord_every_s * rate))) if chord_every_s else 0
        self._halt = threading.Event()
        self._lock = threading.Lock()
        self.hits = 0
        self.failures = 0
        self.log: list[tuple[float, int]] = []       # (monotonic, pad) per press
        self.kits: list[tuple[float, str]] = []      # (monotonic, kit name) per swap

    def run(self) -> None:
        tick = 0
        next_t = time.monotonic()
        while not self._halt.is_set():
            delay = next_t - time.monotonic()
            if delay > 0 and self._halt.wait(delay):
                break
            pads = pattern(self._phase, tick, chord_every=self._chord_every)
            now = time.monotonic()
            for pad in pads:
                self._send("press", pad)
            with self._lock:
                self.log.extend((now, pad) for pad in pads)
            self._halt.wait(min(0.04, self._period / 2))
            for pad in pads:            # always released: a held pad outlives the run
                self._send("release", pad)
            with self._lock:
                self.hits += len(pads)
            tick += 1
            next_t += self._period

    def _send(self, what: str, pad: int) -> None:
        with self._lock:
            try:
                if what == "press":
                    self._t.press(pad, 100)
                else:
                    self._t.release(pad)
            except Exception:  # noqa: BLE001 - a lost message is counted, not raised
                self.failures += 1

    def kit(self, kit_id: int, name: str = "") -> bool:
        with self._lock:
            try:
                self._t._io.send_sysex(sysex_kit_load(kit_id))
                self.kits.append((time.monotonic(), name))
                return True
            except Exception:  # noqa: BLE001
                self.failures += 1
                return False

    def stop(self) -> None:
        self._halt.set()


# ------------------------------------------------------------------ the scenario

def _wait(ctx: Context, seconds: float) -> None:
    if seconds > 0:
        ctx.cancelled.wait(seconds)
    ctx.check_cancelled()


class LiveSetScenario:
    name = "live_set"
    Params = Params
    description = ("the full live setup: host stream + capture over USB audio, a 120 BPM "
                   "scene, pads in four styles, bounces, kit swaps; zero glitches")

    def run(self, ctx: Context, params: Params, progress: Progress) -> Report:
        p = params
        unknown = [ph for ph in p.phases if ph not in PHASES]
        if unknown or not p.phases:
            raise HilError(BAD_ARGS, f"unknown phase(s) {unknown}",
                           hint="phases: " + ",".join(PHASES))
        data: dict[str, Any] = {"phases": [], "normalised": {}, "kits": {}, "found": {},
                                "route_seen": None, "failures": [], "left": {}}
        ctx.partial = data
        artifacts: list[Artifact] = []
        state: dict[str, Any] = {"scenes": [], "countin": None, "con": None}
        try:
            try:
                self._run(ctx, p, progress, data, artifacts, state)
            except HilError as e:
                if e.code not in SETUP_CODES:
                    raise
                data["failures"].append(f"{e.code}: {e.message}")
                return self._verdict(data, artifacts, env=f"{e.message}"
                                     + (f" ({e.hint})" if e.hint else ""))
            finally:
                self._put_back(ctx, p, data, state)
            return self._verdict(data, artifacts)
        finally:
            ctx.close_all()

    # -- the set ---------------------------------------------------------------
    def _run(self, ctx: Context, p: Params, progress: Progress, data: dict[str, Any],
             artifacts: list[Artifact], state: dict[str, Any]) -> None:
        total = 2 + 3 * len(p.phases)
        step = 0
        progress(step, total, "normalising the bench")
        self._default(ctx, p)
        con = ctx.open_console(reset=p.reset_first)
        state["con"] = con
        artifacts.append(Artifact(path=str(ctx.workdir / "console.log"), mime="text/plain",
                                  role="console"))
        if p.reset_first:
            boot = con.wait_boot(timeout_s=60.0, settle_s=2.0)
            data["boot_complete"] = bool(getattr(boot, "complete", False))
            data["sd"] = parse_sd_boot([line for _s, line in
                                        con.read(since_seq=0, match="bsp_sdmmc",
                                                 limit=200).lines])
            ctx.log(f"SD card: {data['sd']}")
        link = ctx.open_cdc()
        try:
            data["normalised"]["pins"] = pins.restore_pending(link, ctx.device.id, ctx.log)
            # ...and the host's own: an audio-nodes service an unfinished run paused.
            data["normalised"]["host_pins"] = pins.restore_pending(None, card_hold.PIN_DEVICE,
                                                                   ctx.log)
            link, data["provisioned"] = _audio.ensure_kits(
                ctx, link, [KIT_A, KIT_B, KIT_P], fixtures.write_live_set_kits, console=con,
                provision=not p.no_provision,
                missing_hint_suffix=" — the fixtures are built by crosspad_hil.fixtures")
            for name in (KIT_A, KIT_B, KIT_P):
                kid, _n, _cur = _audio.resolve_kit(link, kit_name=name)
                data["kits"][name] = int(kid)
            data["found"] = {"kit": verbs.kit_status(link)["current"],
                             "app": verbs.app_list(link).get("running")}
            st = _seq(link, "STOP")
            state["countin"] = st.get("countin")
            state["scenes0"] = int(st.get("scenes") or 0)
            if p.bounce:
                _asong(link, f"OPEN {PROJECT}")
                self._poll_asong(ctx, link, lambda r: r.get("open") in (OPEN_READY, 3), 60.0)
            # A kit before the app: with none loaded (a reset, the card mounting late)
            # the Sampler does not start at all -- APP_LIST stays at running=-.
            _audio.load_kit_waiting(ctx, link, data["kits"][KIT_A])
            _audio.start_app(ctx, link, SAMPLER_APP)
            data["resets0"] = verbs.reset_count(link)
        finally:
            link.close()
        step += 1

        for i, phase in enumerate(p.phases):
            ph: dict[str, Any] = {"phase": phase, "failures": []}
            data["phases"].append(ph)
            # One phase that aborts does not cost the others: the board is brought
            # back (a console reset if it has to be) and the counters are read anyway.
            try:
                progress(step, total, f"{phase}: bounce window")
                self._bounce_window(ctx, p, phase, ph, state, artifacts)
                step += 1
                progress(step, total, f"{phase}: live block ({p.block_s:.0f}s)")
                self._live_block(ctx, p, phase, ph, data, artifacts, con)
                step += 1
            except HilError as e:
                if e.code == CANCELLED:
                    raise
                ph["aborted"] = f"{e.code}: {e.message}"
                ctx.log(f"{phase}: aborted — {ph['aborted']}")
            progress(step, total, f"{phase}: reading the counters")
            try:
                self._recover_default(ctx, p, ph, con)
                self._counters(ctx, p, ph, con)
            except HilError as e:
                if e.code == CANCELLED:
                    raise
                ph["aborted"] = "; ".join(x for x in (ph.get("aborted"),
                                                      f"after the block: {e.message}") if x)
                for later in p.phases[i + 1:]:
                    data["phases"].append({"phase": later, "failures": [],
                                           "not_run": "the board could not be recovered"})
                ph["problems"] = phase_problems(ph)
                break
            ph["problems"] = phase_problems(ph)
            said = list(ph["problems"])
            if ph.get("aborted"):
                said.append(f"aborted: {ph['aborted']}")
            ctx.log(f"{phase}: " + ("; ".join(said) or "clean"))
            step += 1
        progress(total, total, "done")

    def _default(self, ctx: Context, p: Params) -> None:
        if ctx.device.usb_mode is not UsbMode.DEFAULT or ctx.device.ports.cdc is None:
            ctx.log("normalising: back to the default profile")
            ctx.device = set_mode(ctx.device, UsbMode.DEFAULT, wait=True,
                                  timeout_s=p.enum_timeout)

    def _recover_default(self, ctx: Context, p: Params, ph: dict[str, Any],
                         con: Console | None) -> None:
        """Back to the default profile, the way a person at the bench would: ask
        again (up to --enum-timeout each, the USB serial guard needs ~45 s), then a
        hard reset over the STM bridge and once more; raises when it still is not."""
        last: HilError | None = None
        # A board that hung in the audio profile does not answer a profile switch:
        # straight to the reset, not two --enum-timeout waits first.
        hung = bool(ph.get("hung")) and con is not None
        for _attempt in range(0 if hung else 2):
            try:
                self._refresh(ctx)
                self._default(ctx, p)
                return
            except HilError as e:
                if e.code == CANCELLED:
                    raise
                last = e
                ctx.log(f"back to default failed: {e.code}: {e.message}")
        if con is None:
            raise last  # type: ignore[misc]
        ctx.log(("the board hung: " if hung else "")
                + "hard reset over the STM bridge, then back to default")
        con.reset()
        con.wait_boot(timeout_s=60.0, settle_s=2.0)
        ph["recovery_resets"] = (ph.get("recovery_resets") or 0) + 1
        ph["recovered_by_reset"] = True
        self._refresh(ctx)
        self._default(ctx, p)

    @staticmethod
    def _refresh(ctx: Context) -> None:
        """The board as it enumerates now: a timed-out switch leaves ctx.device stale."""
        try:
            from crosspad_hil.devices import discover, select

            ctx.device = select(discover(), ctx.device.id)
        except HilError as e:
            ctx.log(f"rediscovery: {e.code}: {e.message}")

    @staticmethod
    def _poll_asong(ctx: Context, link: CdcLink, done: Any, timeout_s: float) -> dict:
        deadline = time.monotonic() + timeout_s
        while True:
            st = _asong(link)
            if done(st) or time.monotonic() >= deadline:
                return st
            _wait(ctx, POLL_S)

    # -- default profile: a scene, a bounce, the scene again -----------------------
    def _bounce_window(self, ctx: Context, p: Params, phase: str, ph: dict[str, Any],
                       state: dict[str, Any], artifacts: list[Artifact]) -> None:
        self._default(ctx, p)
        kits = phase_kits(phase)
        link = ctx.open_cdc()
        player: PadPlayer | None = None
        transport = None
        bounced: tuple[dict[str, Any], int, int] | None = None
        try:
            verbs.kit_load(link, self._kit(ph, kits[0], ctx), wait_s=30.0)
            ph["kit_swaps_log_seq"] = max((int(r.get("seq", -1)) for r in
                                           verbs.kit_swap_log(link, log=ctx.log)), default=-1)
            ph["engine0"] = verbs.smpl_counters(link)
            ph["pads0"] = verbs.pad_stats(link)
            ph["resets_seen"] = verbs.reset_count(link)
            # The phase's own scene: its pads, at the set's tempo.
            st = _seq(link, "STOP")
            scene = int(st.get("scenes") or 0)
            _seq(link, "ADD")
            state["scenes"].append(scene)
            for action in (f"SCENE {scene}", "COUNTIN 0", f"BARS {scene} 1", f"BPM {p.bpm}",
                           "REC"):
                _seq(link, action)
            for pad in bounce_pads(phase):
                verbs.pad_press(link, pad, 100)
                verbs.pad_release(link, pad)
            _wait(ctx, 60.0 / p.bpm * 4)
            _seq(link, "PLAY")
            ph["scene"] = scene
            if not p.bounce:
                return
            transport = open_pad_transport(ctx.device, "sysex_esp")
            player = PadPlayer(transport, phase, self._rate(p, phase), p.chord_every_s)
            player.start()
            track = (len(state["scenes"]) - 1) % 4      # one track per phase
            clips = _asong(link).get("clips") or []
            clips0 = clips[track] if track < len(clips) else 0
            _asong(link, f"BOUNCE {scene} {track}")
            st = self._poll_asong(ctx, link, lambda r: r.get("bounce") in
                                  (BOUNCE_DONE, BOUNCE_FAILED), BOUNCE_TIMEOUT_S)
            _seq(link, "PLAY")       # a bounce stops the sequencer, by design
            bounced = (st, clips0, track)
        finally:
            if player is not None:
                player.stop()
                player.join(timeout=3.0)
            if transport is not None:
                transport.close()
            link.close()
        if bounced is not None:
            # After the link is closed: FS_GET opens the CDC port itself.
            ph["bounce"] = self._judge_bounce(ctx, *bounced, phase, artifacts)

    def _judge_bounce(self, ctx: Context, st: dict[str, Any], clips0: int, track: int,
                      phase: str, artifacts: list[Artifact]) -> dict[str, Any]:
        b: dict[str, Any] = {"state": st.get("bounce"), "file": st.get("file"),
                             "err": st.get("err"), "ok": False}
        if st.get("bounce") != BOUNCE_DONE:
            b["why"] = f"bounce={st.get('bounce')} {st.get('err') or 'did not complete'}"
            return b
        clips = st.get("clips") or []
        if track < len(clips) and clips[track] != clips0 + 1:
            b["why"] = f"clips on track {track + 1} went {clips0} -> {clips[track]}"
            return b
        remote = f"/sdcard/SONGS/{st.get('project') or PROJECT}/clips/{st.get('file')}"
        local = ctx.workdir / f"bounce_{phase}.wav"
        try:
            pull_file(ctx, remote, local)
        except HilError as e:
            b["why"] = f"could not fetch {remote}: {e.code}: {e.message}"
            return b
        artifacts.append(Artifact(path=str(local), mime="audio/wav", role=f"bounce:{phase}"))
        take = score_take(local, slice_s=SLICE_S)
        b["take"] = take
        if take["silent_slices"] == take["slices"]:
            b["why"] = "the bounced file is silent"
        elif take["clicks"] or take["dropouts"] or take.get("holes"):
            b["why"] = f"the bounced file has {take['clicks']} click(s), " \
                       f"{take['dropouts']} dropout(s), {take.get('holes', 0)} hole(s)" \
                       + (f" (longest {take['hole_max_ms']:.0f} ms)" if take.get("holes") else "")
        else:
            b["ok"] = True
        return b

    # -- audio profile: the phone's view of the set ---------------------------------
    def _live_block(self, ctx: Context, p: Params, phase: str, ph: dict[str, Any],
                    data: dict[str, Any], artifacts: list[Artifact], con: Console) -> None:
        ctx.device = set_mode(ctx.device, UsbMode.AUDIO, wait=True, timeout_s=p.enum_timeout)
        with card_hold.for_run(ctx, enabled=p.exclusive_card, expected_s=p.block_s) as hold:
            ph["card_exclusive"] = hold["paused"]
            ph["card_hold"] = hold
            self._live_block_on_card(ctx, p, phase, ph, data, artifacts, con, hold["paused"])

    def _live_block_on_card(self, ctx: Context, p: Params, phase: str, ph: dict[str, Any],
                            data: dict[str, Any], artifacts: list[Artifact], con: Console,
                            direct: bool) -> None:
        before, after = hostgain.apply(_card(ctx.device))
        ph["host_gain"] = {"before": before.deviations(), "after": after.deviations()}
        seq0 = con.seq
        session = stream = transport = player = None
        names = phase_kits(phase)
        kits = [self._kit(ph, k, ctx) for k in names]
        t_block0 = time.monotonic()
        hung: str | None = None
        try:
            if p.capture:
                wav = ctx.workdir / f"live_{phase}.wav"
                session = open_capture(ctx.device, seconds=None, preset="asis", out=wav,
                                       keep_mode=True, resume_audio_tasks=False, log=ctx.log)
                info = session.start()
                data["route_seen"] = info.get("route_seen")
                artifacts.append(Artifact(path=str(wav), mime="audio/wav", role=f"live:{phase}"))
            if p.stream:
                tone = write_tone(ctx.workdir / "stream.wav", p.block_s + 60)
                stream = (start_direct_stream(tone, _card(ctx.device), ctx.log) if direct
                          else start_host_stream(tone, p.stream_target, ctx.log))
                _wait(ctx, STREAM_SETTLE_S)
                if not stream.alive():
                    raise HilError(ENV, "the host stream exited before the block: "
                                   + stream.stop())
            transport = open_pad_transport(ctx.device, "sysex_esp")
            player = PadPlayer(transport, phase, self._rate(p, phase), p.chord_every_s)
            player.kits.append((0.0, names[0]))     # the window loaded it before the block
            player.start()
            t_end = time.monotonic() + p.block_s
            next_swap = time.monotonic() + p.swap_every_s
            swaps = 0
            while time.monotonic() < t_end:
                _wait(ctx, max(0.0, min(next_swap, t_end) - time.monotonic()))
                if time.monotonic() >= next_swap and time.monotonic() < t_end:
                    swaps += 1
                    player.kit(kits[swaps % len(kits)], names[swaps % len(kits)])
                    next_swap += p.swap_every_s
            ph["swaps_sent"] = swaps
        except (subprocess.SubprocessError, OSError) as e:
            # arecord/aplay timing out, a MIDI port gone: the board stopped answering.
            hung = f"{type(e).__name__}: {e}"
        finally:
            if player is not None:
                player.stop()
                player.join(timeout=3.0)
                ph["hits_sent"], ph["send_failures"] = player.hits, player.failures
                # For timing the late hits against the swap log, back in the
                # default profile; dropped from the report there.
                ph["_hit_log"] = list(player.log)
                ph["_swap_times"] = [t for t, _k in player.kits if t]
            if transport is not None:
                transport.close()
            if stream is not None:
                ph["stream_stderr"] = stream.stop()[-300:]
                if getattr(stream, "hung", False):
                    hung = hung or "the host stream did not exit: the card stopped answering"
            if session is not None:
                try:
                    result = session.stop()
                    ph["capture"] = result.to_dict() if hasattr(result, "to_dict") else result
                    health = session.health() if hasattr(session, "health") else {}
                    ph["host_capture"] = health
                    ph["host_lost_s"], untrusted = host_capture_verdict(health or {})
                    if untrusted:
                        ph["capture_untrusted"] = untrusted
                        ctx.log(f"{phase}: capture not trusted -- {untrusted}")
                    ph["capture_problems"] = _audio.capture_problems(ph["capture"] or {},
                                                                     health or {})
                    if (health or {}).get("recorder_hung"):
                        hung = hung or "the capture did not exit: the card stopped answering"
                except HilError as e:
                    ph["failures"].append(f"capture stop: {e.code}: {e.message}")
                except (subprocess.SubprocessError, OSError) as e:
                    hung = hung or f"capture stop: {type(e).__name__}: {e}"
            spk = con.read(since_seq=0, match="SPK #", limit=100000).lines
            ph["spk"] = parse_spk([line for s, line in spk if s >= seq0],
                                  before=[line for s, line in spk if s < seq0])
            ph["console_seq"] = (seq0, con.seq)
            if hung:
                tail = con.read(since_seq=max(seq0, con.seq - HUNG_TAIL_LINES + 1),
                                limit=HUNG_TAIL_LINES).lines
                ph["hung"] = {"t_s": round(time.monotonic() - t_block0, 1),
                              "at": time.strftime("%H:%M:%S"), "reason": hung,
                              "console_tail": [line for _s, line in tail]}
                ctx.log(f"{phase}: board hung in the UAC block at "
                        f"t={ph['hung']['t_s']:.0f} s: {hung}")
        if hung:
            raise HilError(TIMEOUT, f"board hung in UAC block at t={ph['hung']['t_s']:.0f} s: "
                           f"{hung}", hint="the phase recovers it with a reset over the STM "
                                           "bridge; the console tail is in the report")
        if session is not None and p.capture and player is not None:
            t_wav0 = float(getattr(session, "started_at", None) or 0.0)
            hits = [(t - t_wav0, pad) for t, pad in player.log]
            kit_times = [(t - t_wav0 if t else -1e9, k) for t, k in player.kits]
            try:
                ph["take"] = score_take(ctx.workdir / f"live_{phase}.wav",
                                        expect=lambda a, b: expected_in(hits, kit_times, a, b))
            except HilError as e:
                ph["failures"].append(f"scoring the capture: {e.message}")

    # -- default profile: what the board counted ------------------------------
    def _counters(self, ctx: Context, p: Params, ph: dict[str, Any], con: Console) -> None:
        link = ctx.open_cdc()
        try:
            rows = [r for r in verbs.kit_swap_log(link, log=ctx.log)
                    if int(r.get("seq", -1)) > ph.get("kit_swaps_log_seq", -1)]
            ph["swaps"] = swap_totals(rows)
            ph["swap_rows"] = rows           # the evidence behind late / rt_gap
            ph["late"] = late_from_timestamps(rows, ph.pop("_swap_times", []),
                                              ph.pop("_hit_log", []))
            ph["swaps"]["late_max_ms"] = ph["late"]["late_max_ms"]
            if ph.get("swaps_sent", 0) > len(rows) and ph.get("swaps_sent", 0) <= 8:
                ph["failures"].append(f"{ph['swaps_sent']} swaps sent, {len(rows)} logged")
            ph["engine"] = _delta(verbs.smpl_counters(link), ph.get("engine0") or {},
                                  ("underrun", "noblock"))
            ph["pads"] = _delta(verbs.pad_stats(link), ph.get("pads0") or {},
                                ("press", "played", "inj_drop", "sbuf_drop", "usb_drop",
                                 "uart_ovf"))
            ph["mem"] = verbs.mem(link)
            if phase_kits(ph.get("phase", ""))[0] == KIT_P:
                try:     # evidence for the voice limit (steals), not a gate here
                    ph["pitched_status"] = verbs.pitched_status(link)
                except HilError as e:
                    ph["pitched_status"] = {"error": f"{e.code}: {e.message}"}
            st = _seq(link)
            ph["seq_state"] = st.get("state")
            ph["seq_stopped"] = st.get("state") not in SEQ_PLAYING
            resets = verbs.reset_count(link)
            ph["resets"] = resets - int(ph.get("resets_seen", resets))
        finally:
            link.close()
        seq0, seq1 = ph.get("console_seq", (0, con.seq))
        fatal = [ev.line for ev in con.events(since_seq=seq0)
                 if ev.kind in ("fatal",) and ev.seq <= seq1]
        reboots = sum(1 for ev in con.events(since_seq=seq0) if ev.kind == "reboot")
        ph["fatal"] = fatal
        if reboots:
            ph["resets"] = max(ph.get("resets") or 0, reboots)

    @staticmethod
    def _kit(ph: dict[str, Any], name: str, ctx: Context) -> int:
        kits = ctx.partial.get("kits") or {}
        if name not in kits:
            raise HilError(ENV, f"no kit {name!r} on the card")
        return int(kits[name])

    @staticmethod
    def _rate(p: Params, phase: str) -> float:
        return {"one": p.one_rate, "four": p.four_rate}.get(phase, p.many_rate)

    # -- put everything back ----------------------------------------------------
    def _put_back(self, ctx: Context, p: Params, data: dict[str, Any],
                  state: dict[str, Any]) -> None:
        left = data["left"]
        try:
            self._default(ctx, p)
            link = ctx.open_cdc()
        except HilError as e:
            left["error"] = f"{e.code}: {e.message}"
            ctx.log(f"could not get back to CDC to put the board back: {e.message}")
            return
        try:
            for name, fn in (
                ("seq", lambda: _seq(link, "STOP")),
                ("scenes", lambda: [_seq(link, f"DEL {s}")
                                    for s in sorted(state["scenes"], reverse=True)]),
                ("countin", lambda: state["countin"] is not None
                 and _seq(link, f"COUNTIN {state['countin']}")),
                ("song", lambda: p.bounce and _asong(link, "STOP")),
                ("kit", lambda: data["found"].get("kit") is not None
                 and data["found"]["kit"] >= 0
                 and verbs.kit_status(link)["current"] != data["found"]["kit"]
                 and verbs.kit_load(link, data["found"]["kit"], wait_s=30.0)),
                ("app", lambda: data["found"] and data["found"].get("app") is None
                 and verbs.app_list(link).get("running") is not None and verbs.app_stop(link)),
            ):
                try:
                    fn()
                    left[name] = "restored"
                except HilError as e:
                    left[name] = f"{e.code}: {e.message}"
        finally:
            link.close()

    # -- verdict ----------------------------------------------------------------
    def _verdict(self, data: dict[str, Any], artifacts: list[Artifact],
                 env: str | None = None) -> Report:
        lines = []
        for ph in data["phases"]:
            ph.pop("_hit_log", None)        # a phase that died before its counters
            ph.pop("_swap_times", None)
            if ph.get("not_run"):
                lines.append(f"  {ph['phase']:<8} not run: {ph['not_run']}")
                continue
            probs = ph.get("problems")
            if probs is None:
                probs = phase_problems(ph)
            text = "; ".join(probs) if probs else ""
            if ph.get("aborted"):
                text = "; ".join(x for x in (text, f"aborted ({ph['aborted']})") if x)
            if ph.get("capture_untrusted"):
                text = "; ".join(x for x in (text, "capture not measured: "
                                             + ph["capture_untrusted"]) if x)
            lines.append(f"  {ph['phase']:<8} " + (text or "clean"))
        sd = data.get("sd")
        if sd:
            lines.append(f"  SD card at boot: width={sd.get('width')}-bit "
                         f"max_freq={sd.get('max_freq_khz')} kHz"
                         + (" (4-bit failed, retried 1-bit)" if sd.get("retried_1bit") else "")
                         + ("" if sd.get("mounted") is not False else " MOUNT FAILED"))
        route = data.get("route_seen")
        if route is not None:
            lines.append(f"  capture route (what the phone receives): {route}")
        bad = [ph for ph in data["phases"] if not ph.get("not_run")
               and (ph.get("problems") or phase_problems(ph))]
        unmeasured = [ph for ph in data["phases"]
                      if ph.get("not_run") or ph.get("aborted") or ph.get("capture_untrusted")]
        if env is None and not bad and unmeasured:
            env = "phase(s) not measured: " + ", ".join(ph["phase"] for ph in unmeasured)
        if env is not None and not bad:
            return Report(passed=False, summary="live_set: not measured — " + env + "\n"
                          + "\n".join(lines), data=data, artifacts=artifacts, exit_code=2)
        if bad or env is not None:
            head = "live_set: FAIL — " + ", ".join(ph["phase"] for ph in bad)
            if env is not None:
                head += f" (then: {env})"
            return Report(passed=False, summary=head + "\n" + "\n".join(lines), data=data,
                          artifacts=artifacts, exit_code=1)
        return Report(passed=True, summary="live_set: PASS — zero glitches in "
                      f"{len(data['phases'])} phase(s)\n" + "\n".join(lines), data=data,
                      artifacts=artifacts, exit_code=0)


def _card(device: Device) -> str | None:
    return device.ports.uac2.alsa_id if device.ports.uac2 is not None else None


SCENARIO = register(LiveSetScenario())
