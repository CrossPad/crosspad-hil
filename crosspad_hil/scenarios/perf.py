"""How fast the board does the two things a player waits for: swapping a kit, and
bringing the sample engine up.

The release gate proved the build does not crash, leak or drop audio — and let a
release through whose kit swap had gone from 1.6 s to 2.8 s over CDC and 4.0 s
in the USB-audio profile (the product's default), and whose sample engine took
3.3 s to initialise instead of 130 ms. Nothing measured either. This does:

1. **Kit swap, CDC profile.** ``HIL SINE A`` <-> ``HIL SINE B``, ``--swaps``
   times (default 6), no pads played, ``KIT_LOAD`` over CDC and ``--settle``
   seconds between swaps. The number is the device's own: the median
   ``done_ms`` of those swaps in ``KIT_SWAP_LOG`` (ms from the request to the
   last pad reloaded), not a host stopwatch that includes USB round trips.
2. **Kit swap, USB-audio profile, with the host streaming into the board.** The
   same swaps, requested over the companion SysEx channel (``KIT_LOAD k`` — the
   profile has no CDC) while ``pw-play`` plays a tone into the board's output,
   because a board that renders a host stream is the product's normal state and
   the slow case. ``KIT_SWAP_LOG`` is a ring of 8 in PSRAM that survives the
   profile switch, so it is read over CDC once the board is back in the default
   profile; rows are told apart by ``seq``.
   Nothing is recorded: the tone goes through PipeWire to ``crosspad_out``, the
   node the audio-nodes service makes of the card, so a card that service
   holds (which makes ``kit_swap_audio``'s direct capture fail as busy) is
   exactly the path this uses. No sink at all is exit 2.
3. **Sample-engine init at boot.** A reset, then the console's two
   ``EngineHandoff`` lines — ``control block`` and ``started`` — whose
   ``(ms since boot)`` stamps bracket ``SampleStreamPlayer_Init``.

FAIL, with the numbers, when the CDC median exceeds ``--max-cdc-ms`` (2000),
the USB-audio median exceeds ``--max-uac-ratio`` (1.2) times the CDC median of
the same run, or the init exceeds ``--max-init-ms`` (500). A measurement the
bench could not take (no ``KIT_SWAP_LOG`` on an old firmware, no ``pw-play``,
no ``EngineHandoff`` lines) is exit 2, not a pass.
"""
from __future__ import annotations

import math
import re
import shutil
import statistics
import struct
import subprocess
import time
import wave
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from crosspad_hil import audio_nodes, fixtures, verbs
from crosspad_hil.cdc import CdcLink
from crosspad_hil.console import Console
from crosspad_hil.devices import Device, UsbMode
from crosspad_hil.errors import (
    BAD_ARGS,
    ENV,
    NO_CDC_IN_AUDIO_MODE,
    NOT_SUPPORTED,
    TIMEOUT,
    HilError,
)
from crosspad_hil.midi import MidiIO, MidiRole, companion_command
from crosspad_hil.scenarios import _audio
from crosspad_hil.scenarios.base import Artifact, Context, Progress, Report, register
from crosspad_hil.usbmode import set_mode

SAMPLER_APP = "Sampler"
# Starting the Sampler loads its kit, and right after the reset this scenario
# makes, the launcher is still being built: on the bench (dev_1ede, both
# images) APP_START's default 3 s window ran out with nothing running.
APP_START_WAIT_S = 15.0
APP_START_ATTEMPTS = 2
LAUNCHER_TIMEOUT_S = 10.0     # APP_STOP tears the app down and rebuilds the main screen
APP_START_SETTLE_S = 2.0
KIT_RING = 8                  # KIT_SWAP_LOG keeps the last 8 swaps (hil_control.cpp)
STATUS_POLL_S = 0.2
STREAM_SETTLE_S = 2.0         # the host stream reaching the board before the first swap
STREAM_NODE_WAIT_S = 15.0     # the PipeWire sink appearing after the profile switch
TONE_HZ = 441.0
TONE_AMPLITUDE = 3000
TONE_RATE = 44_100

SETUP_CODES: frozenset[str] = frozenset({ENV, NOT_SUPPORTED, TIMEOUT, NO_CDC_IN_AUDIO_MODE})

_ENGINE_BLOCK = re.compile(r"^[IWE] \((\d+)\) EngineHandoff: control block")
_ENGINE_STARTED = re.compile(r"^[IWE] \((\d+)\) EngineHandoff: started")
_KITSTATUS = re.compile(r"current=(-?\d+)\s+loading=(\d+)(?:\s+pending=(-?\d+))?")
_SPK_LINE = r"usb_audio_bridge: SPK #"


@dataclass
class Params:
    kit_a: str = field(default="HIL SINE A", metadata={"help": "first kit of the pair"})
    kit_b: str = field(default="HIL SINE B", metadata={"help": "second kit of the pair"})
    swaps: int = field(default=6, metadata={"help": f"swaps per profile (1..{KIT_RING}: "
                                                    "the device keeps the last "
                                                    f"{KIT_RING})"})
    settle: float = field(default=3.0, metadata={"help": "seconds between swaps"})
    load_timeout: float = field(default=30.0,
                                metadata={"help": "seconds one swap may take before the run "
                                                  "calls it stuck"})
    max_cdc_ms: float = field(default=2000.0,
                              metadata={"help": "FAIL when the CDC median done_ms exceeds this"})
    max_uac_ratio: float = field(default=1.2,
                                 metadata={"help": "FAIL when the USB-audio median exceeds "
                                                   "this times the CDC median"})
    max_init_ms: float = field(default=500.0,
                               metadata={"help": "FAIL when the sample engine's init at boot "
                                                 "exceeds this"})
    uac: bool = field(default=True, metadata={"help": "measure in the USB-audio profile too"})
    stream: bool = field(default=True,
                         metadata={"help": "USB-audio: play a tone into the board with pw-play "
                                           "while swapping"})
    stream_target: str = field(default=audio_nodes.OUT_NODE,
                               metadata={"help": "PipeWire node pw-play targets (falls back "
                                                 "to the card's own ALSA sink)"})
    init: bool = field(default=True,
                       metadata={"help": "reset the board and time the sample engine's init"})
    init_timeout: float = field(default=60.0,
                                metadata={"help": "seconds after the reset to wait for "
                                                  "'EngineHandoff: started'"})
    no_provision: bool = field(default=False,
                               metadata={"help": "fail instead of pushing missing HIL SINE "
                                                 "kits to the card"})


# --------------------------------------------------------------------------- pure parts

def engine_init_ms(lines: list[str]) -> int | None:
    """ms between the first ``EngineHandoff: control block`` and the ``started``
    after it, from the ``(ms since boot)`` stamps; None when either is missing."""
    block: int | None = None
    for line in lines:
        text = line.strip()
        if block is None:
            m = _ENGINE_BLOCK.search(text)
            if m:
                block = int(m.group(1))
            continue
        m = _ENGINE_STARTED.search(text)
        if m:
            return int(m.group(1)) - block
    return None


def last_seq(rows: list[dict[str, Any]]) -> int:
    seqs = [int(r["seq"]) for r in rows if "seq" in r]
    return max(seqs) if seqs else -1


def rows_after(rows: list[dict[str, Any]], seq: int) -> list[dict[str, Any]]:
    """The swap rows a phase produced: every row newer than the phase's start."""
    return [r for r in rows if "seq" in r and int(r["seq"]) > seq]


def phase_stats(rows: list[dict[str, Any]]) -> dict[str, Any]:
    done = [int(r["done_ms"]) for r in rows if r.get("ok") and "done_ms" in r]
    return {
        "swaps": len(rows),
        "ok": len(done),
        "failed_seq": [int(r["seq"]) for r in rows if not r.get("ok")],
        "done_ms": done,
        "median_ms": statistics.median(done) if done else None,
        "min_ms": min(done) if done else None,
        "max_ms": max(done) if done else None,
        "underruns": sum(int(r.get("underruns", 0) or 0) for r in rows),
        "rt_gap_max_us": max((int(r.get("rt_gap_max_us", 0) or 0) for r in rows), default=0),
    }


def swap_plan(swaps: int, id_a: int, id_b: int) -> list[int]:
    """B, A, B, A, ... starting from A already loaded."""
    return [id_b if i % 2 == 0 else id_a for i in range(swaps)]


def verdict(p: Params, data: dict[str, Any]) -> tuple[list[str], list[str]]:
    """(failures, unmeasured): failures are the firmware's, unmeasured the bench's."""
    failures: list[str] = [f"kit swap stuck: {s}" for s in data.get("stuck") or []]
    unmeasured: list[str] = []

    def phase(key: str, label: str) -> dict[str, Any] | None:
        st = data.get(key)
        if st is None:
            return None
        if st["swaps"] == 0:
            unmeasured.append(f"{label}: KIT_SWAP_LOG has no rows for this run "
                              "(firmware without the verb?)")
            return None
        if st["swaps"] < p.swaps:
            failures.append(f"{label}: only {st['swaps']} of {p.swaps} swaps logged")
        if st["failed_seq"]:
            failures.append(f"{label}: swaps failed (ok=0) at seq {st['failed_seq']}")
        return st if st["median_ms"] is not None else None

    cdc = phase("cdc", "CDC kit swap")
    if cdc is not None and cdc["median_ms"] > p.max_cdc_ms:
        failures.append(f"CDC kit swap: median {cdc['median_ms']:.0f} ms > "
                        f"{p.max_cdc_ms:.0f} ms")
    uac = phase("uac", "USB-audio kit swap")
    if uac is not None and cdc is not None:
        ratio = uac["median_ms"] / cdc["median_ms"] if cdc["median_ms"] else math.inf
        data["uac_ratio"] = round(ratio, 3)
        if ratio > p.max_uac_ratio:
            failures.append(f"USB-audio kit swap: median {uac['median_ms']:.0f} ms = "
                            f"{ratio:.2f}x CDC {cdc['median_ms']:.0f} ms > "
                            f"{p.max_uac_ratio:.2f}x")
    if p.init:
        init = data.get("init_ms")
        if init is None:
            unmeasured.append("engine init: no 'EngineHandoff: control block' / 'started' "
                              "pair on the console after the reset")
        elif init > p.max_init_ms:
            failures.append(f"engine init at boot: {init} ms > {p.max_init_ms:.0f} ms")
    return failures, unmeasured


def _fmt_phase(label: str, st: dict[str, Any] | None) -> str:
    if not st or st.get("median_ms") is None:
        return f"  {label}: not measured"
    return (f"  {label}: median {st['median_ms']:.0f} ms over {st['ok']} swaps "
            f"({st['min_ms']}..{st['max_ms']} ms)")


def summary(p: Params, data: dict[str, Any], failures: list[str],
            unmeasured: list[str]) -> str:
    head = "perf: FAIL" if failures else "perf: not measured" if unmeasured else "perf: PASS"
    lines = [head, _fmt_phase(f"CDC kit swap (limit {p.max_cdc_ms:.0f} ms)", data.get("cdc"))]
    if p.uac:
        uac = _fmt_phase(f"USB-audio kit swap{' + host stream' if p.stream else ''} "
                         f"(limit {p.max_uac_ratio:.2f}x CDC)", data.get("uac"))
        if data.get("uac_ratio") is not None:
            uac += f" = {data['uac_ratio']:.2f}x CDC"
        lines.append(uac)
    if p.init:
        init = data.get("init_ms")
        lines.append(f"  engine init at boot (limit {p.max_init_ms:.0f} ms): "
                     + ("not measured" if init is None else f"{init} ms"))
    lines += [f"FAIL: {f}" for f in failures] + [f"not measured: {u}" for u in unmeasured]
    return "\n".join(lines)


def write_tone(path: Path, seconds: float, *, hz: float = TONE_HZ,
               amplitude: int = TONE_AMPLITUDE, rate: int = TONE_RATE) -> Path:
    """A stereo 16-bit sine, the stream the board renders while it swaps."""
    frames = int(seconds * rate)
    period = rate / hz
    one = b"".join(struct.pack("<hh", s, s) for s in
                   (int(amplitude * math.sin(2 * math.pi * i / period))
                    for i in range(int(round(period * 10)))))   # 10 whole periods
    cycle_frames = len(one) // 4
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as w:
        w.setnchannels(2)
        w.setsampwidth(2)
        w.setframerate(rate)
        reps, rest = divmod(frames, cycle_frames)
        w.writeframes(one * reps + one[:rest * 4])
    return path


# ------------------------------------------------- seams (tests replace these)

class HostStream:
    """``pw-play`` running in the background; ``stop()`` ends it."""

    def __init__(self, argv: list[str]) -> None:
        self.argv = argv
        self.proc = subprocess.Popen(argv, stdout=subprocess.DEVNULL,
                                     stderr=subprocess.PIPE, text=True)

    def alive(self) -> bool:
        return self.proc.poll() is None

    def stop(self) -> str:
        if self.proc.poll() is None:
            self.proc.terminate()
            try:
                self.proc.wait(timeout=3.0)
            except subprocess.TimeoutExpired:
                self.proc.kill()
                self.proc.wait(timeout=3.0)
        err = self.proc.stderr.read() if self.proc.stderr else ""
        return (err or "").strip()


def pick_sink(names: set[str], target: str) -> str | None:
    """``target`` when PipeWire has it, else the CrossPad's own ALSA sink."""
    if target in names:
        return target
    own = sorted(n for n in names if n.startswith("alsa_output") and "crosspad" in n.lower())
    return own[0] if own else None


def start_host_stream(wav: Path, target: str, log: Any,
                      wait_s: float = STREAM_NODE_WAIT_S) -> HostStream:
    if shutil.which("pw-play") is None:
        raise HilError(ENV, "pw-play is not installed: no host stream to render",
                       hint="install pipewire (pw-play), or run with --no-stream")
    deadline = time.monotonic() + wait_s
    sink = None
    while sink is None:
        sink = pick_sink(audio_nodes._pipewire_node_names(), target)
        if sink is not None or time.monotonic() >= deadline:
            break
        time.sleep(0.3)
    if sink is None:
        raise HilError(ENV, f"no PipeWire sink for the CrossPad ({target!r} or its ALSA "
                            f"output) within {wait_s:.0f}s",
                       hint="is the board in the audio profile and PipeWire running?")
    log(f"host stream: pw-play --target {sink} {wav.name}")
    return HostStream(["pw-play", "--target", sink, str(wav)])


def open_companion(device: Device) -> Any:
    io = MidiIO(device, MidiRole.ESP)
    io.open()
    return io


# --------------------------------------------------------------------------- scenario

def start_sampler(ctx: Context, link: CdcLink, *, wait_s: float = APP_START_WAIT_S,
                  attempts: int = APP_START_ATTEMPTS,
                  launcher_s: float = LAUNCHER_TIMEOUT_S,
                  settle_s: float = APP_START_SETTLE_S) -> None:
    """Get the Sampler running: close whatever else runs, wait for the launcher,
    then APP_START with a window long enough for its kit load — once more if the
    first request went nowhere (``running=-``: dropped by a launcher still being
    built)."""
    apps = verbs.app_list(link)
    if SAMPLER_APP not in apps.get("apps", []):
        raise HilError(ENV, f"{SAMPLER_APP} is not in the launcher; kit swaps load nothing "
                            "without it", hint="install the sampler app and reflash",
                       apps=apps.get("apps", []))
    running = apps.get("running")
    if running == SAMPLER_APP:
        return
    if running is not None:
        ctx.log(f"closing {running} before starting {SAMPLER_APP}")
        verbs.app_stop(link)
        deadline = time.monotonic() + launcher_s
        while verbs.app_list(link).get("running") is not None:
            if time.monotonic() >= deadline:
                raise HilError(TIMEOUT, f"{running} still running {launcher_s:.0f}s after "
                                        "APP_STOP", hint="the app does not tear down")
            _wait(ctx, 0.25)
    last: HilError | None = None
    for attempt in range(1, attempts + 1):
        try:
            verbs.app_start(link, SAMPLER_APP, wait_s=wait_s)
            break
        except HilError as e:
            if e.code != TIMEOUT:
                raise
            last = e
            ctx.log(f"APP_START {SAMPLER_APP} attempt {attempt}/{attempts}: {e.message}")
            if e.details.get("running") not in (None, SAMPLER_APP):
                raise   # something else came up: not a dropped request
    else:
        assert last is not None
        raise last
    _wait(ctx, settle_s)


def _wait(ctx: Context, seconds: float) -> None:
    if seconds > 0:
        ctx.cancelled.wait(seconds)
    ctx.check_cancelled()


class PerfScenario:
    name = "perf"
    Params = Params
    description = ("kit swap A<->B timed by the device (CDC, then USB audio with a host "
                   "stream) and the sample engine's init at boot, against thresholds")

    def run(self, ctx: Context, params: Params, progress: Progress) -> Report:
        p = params
        if not 1 <= p.swaps <= KIT_RING:
            raise HilError(BAD_ARGS, f"swaps must be 1..{KIT_RING}, got {p.swaps}",
                           hint=f"KIT_SWAP_LOG keeps the last {KIT_RING} swaps only")
        data: dict[str, Any] = {
            "params": {"swaps": p.swaps, "max_cdc_ms": p.max_cdc_ms,
                       "max_uac_ratio": p.max_uac_ratio, "max_init_ms": p.max_init_ms},
            "kit_ids": {}, "provisioned": [], "init_ms": None, "init_lines": [],
            "cdc": None, "uac": None, "uac_ratio": None, "uac_host_s": [], "cdc_host_s": [],
            "uac_stream": None, "rows": [], "setup_errors": [], "stuck": [],
        }
        ctx.partial = data
        artifacts: list[Artifact] = []
        try:
            try:
                self._run(ctx, p, progress, data, artifacts)
            except HilError as e:
                if e.code not in SETUP_CODES:
                    raise
                data["setup_errors"].append(f"{e.code}: {e.message}")
                hint = f" ({e.hint})" if e.hint else ""
                failures, _unmeasured = verdict(p, data)
                text = summary(p, data, failures, [f"{e.code}: {e.message}{hint}"])
                return Report(passed=False, summary=text, data=data, artifacts=artifacts,
                              exit_code=1 if failures else 2)
            failures, unmeasured = verdict(p, data)
            data["failures"], data["unmeasured"] = failures, unmeasured
            return Report(passed=not failures and not unmeasured,
                          summary=summary(p, data, failures, unmeasured), data=data,
                          artifacts=artifacts,
                          exit_code=1 if failures else 2 if unmeasured else 0)
        finally:
            ctx.close_all()

    # -- phases -------------------------------------------------------------
    def _run(self, ctx: Context, p: Params, progress: Progress, data: dict[str, Any],
             artifacts: list[Artifact]) -> None:
        total = 4
        if ctx.device.usb_mode is not UsbMode.DEFAULT or ctx.device.ports.cdc is None:
            ctx.log("no CDC port — switching back to the default profile first")
            ctx.device = set_mode(ctx.device, UsbMode.DEFAULT)

        con: Console | None = None
        if p.init:
            progress(0, total, "resetting the board to time the sample engine's init")
            con = ctx.open_console(reset=True)
            self._measure_init(ctx, p, con, data)
        else:
            con = ctx.open_console(reset=False)
        console_log = ctx.workdir / "console.log"
        artifacts.append(Artifact(path=str(console_log), mime="text/plain", role="console"))

        progress(1, total, "CDC: kit swaps")
        link = ctx.open_cdc()
        try:
            link, data["provisioned"] = _audio.ensure_kits(
                ctx, link, [p.kit_a, p.kit_b], fixtures.write_sine_kits, console=con,
                provision=not p.no_provision,
                missing_hint_suffix=" — push it with tools/make_sine_kits.py")
            id_a, id_b = self._prepare(ctx, p, link)
            data["kit_ids"] = {"a": id_a, "b": id_b}
            seq0 = last_seq(verbs.kit_swap_log(link, log=ctx.log))
            for k in swap_plan(p.swaps, id_a, id_b):
                t0 = time.monotonic()
                try:
                    verbs.kit_load(link, k, wait_s=p.load_timeout)
                except HilError as e:
                    if e.code != TIMEOUT:
                        raise
                    data["stuck"].append(f"CDC: {e.message}")   # the firmware's, not the bench's
                    break
                data["cdc_host_s"].append(round(time.monotonic() - t0, 3))
                _wait(ctx, p.settle)
            rows = verbs.kit_swap_log(link, log=ctx.log)
        finally:
            link.close()
        data["rows"] = rows
        cdc_rows = rows_after(rows, seq0)
        data["cdc"] = phase_stats(cdc_rows)
        ctx.log(_fmt_phase("CDC kit swap", data["cdc"]).strip())

        if p.uac:
            progress(2, total, "USB audio: kit swaps under a host stream")
            self._uac_phase(ctx, p, con, data, last_seq(cdc_rows) if cdc_rows else seq0,
                            id_a, id_b)
        progress(total, total, "done")

    def _measure_init(self, ctx: Context, p: Params, con: Console, data: dict[str, Any]
                      ) -> None:
        boot = con.wait_boot(timeout_s=p.init_timeout, settle_s=1.0)
        data["boot_complete"] = bool(getattr(boot, "complete", False))
        hit = con.expect([r"EngineHandoff: started"], timeout_s=p.init_timeout,
                         since_seq=0)
        lines = [line for _s, line in con.read(since_seq=0, match="EngineHandoff",
                                               limit=50).lines]
        data["init_lines"] = lines
        data["init_ms"] = engine_init_ms(lines)
        if hit.hit is None:
            ctx.log("no 'EngineHandoff: started' on the console after the reset")
        else:
            ctx.log(f"sample engine init at boot: {data['init_ms']} ms")

    def _prepare(self, ctx: Context, p: Params, link: CdcLink) -> tuple[int, int]:
        hint = " — push both with tools/make_sine_kits.py"
        id_a, name_a, current = _audio.resolve_kit(link, kit_name=p.kit_a,
                                                   missing_hint_suffix=hint)
        id_b, _name_b, _ = _audio.resolve_kit(link, kit_name=p.kit_b,
                                              missing_hint_suffix=hint)
        if id_a == id_b:
            raise HilError(ENV, f"{p.kit_a!r} and {p.kit_b!r} are the same kit ({id_a})")
        start_sampler(ctx, link)
        _audio.load_kit(ctx, link, id_a, name_a, current, wait_s=p.load_timeout)
        _wait(ctx, p.settle)
        return int(id_a), int(id_b)

    def _uac_phase(self, ctx: Context, p: Params, con: Console | None, data: dict[str, Any],
                   seq0: int, id_a: int, id_b: int) -> None:
        ctx.device = set_mode(ctx.device, UsbMode.AUDIO)
        spk_seq = con.seq if con is not None else 0
        stream: HostStream | None = None
        io: Any = None
        try:
            if p.stream:
                wav = write_tone(ctx.workdir / "tone.wav",
                                 seconds=p.swaps * (p.load_timeout + p.settle) + 30.0)
                stream = start_host_stream(wav, p.stream_target, ctx.log)
                _wait(ctx, STREAM_SETTLE_S)
                if not stream.alive():
                    raise HilError(ENV, "pw-play exited before the swaps: "
                                        + (stream.stop() or "no error output"))
            io = open_companion(ctx.device)
            for k in swap_plan(p.swaps, id_a, id_b):
                t0 = time.monotonic()
                reply = companion_command(io, f"KIT_LOAD {k}")
                if reply.strip().upper().startswith("ERR"):
                    raise HilError(ENV, f"companion KIT_LOAD {k}: {reply.strip()}",
                                   hint="the firmware refused the swap over SysEx 0x1F")
                try:
                    self._wait_landed(ctx, io, k, p.load_timeout)
                except HilError as e:
                    if e.code != TIMEOUT:
                        raise
                    data["stuck"].append(f"USB audio: {e.message}")
                    break
                data["uac_host_s"].append(round(time.monotonic() - t0, 3))
                _wait(ctx, p.settle)
        finally:
            if io is not None:
                try:
                    io.close()
                except Exception:  # noqa: BLE001 - teardown must not mask the result
                    pass
            if stream is not None:
                err = stream.stop()
                data["uac_stream"] = {"argv": stream.argv, "stderr": err[-500:]}
            if con is not None:
                spk = con.read(since_seq=spk_seq, match=_SPK_LINE, limit=2000).lines
                data["uac_spk_lines"] = len(spk)
            try:
                ctx.device = set_mode(ctx.device, UsbMode.DEFAULT)
            except HilError as e:
                ctx.log(f"could not return to the default profile: {e.code}: {e.message}")
                raise
        link = ctx.open_cdc()
        try:
            rows = verbs.kit_swap_log(link, log=ctx.log)
        finally:
            link.close()
        data["rows"] = rows
        data["uac"] = phase_stats(rows_after(rows, seq0))
        ctx.log(_fmt_phase("USB-audio kit swap", data["uac"]).strip())

    @staticmethod
    def _wait_landed(ctx: Context, io: Any, kit: int, timeout_s: float) -> None:
        deadline = time.monotonic() + timeout_s
        last = ""
        while True:
            last = companion_command(io, "KIT_STATUS")
            m = _KITSTATUS.search(last)
            if m and int(m.group(1)) == kit and int(m.group(2)) == 0 \
                    and (m.group(3) is None or int(m.group(3)) == -1):
                return
            if time.monotonic() >= deadline:
                raise HilError(TIMEOUT, f"companion KIT_LOAD {kit}: not current after "
                                        f"{timeout_s:.0f}s (last: {last.strip()!r})",
                               kit_id=kit)
            _wait(ctx, STATUS_POLL_S)


SCENARIO = register(PerfScenario())
