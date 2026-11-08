"""``live_jam``: a whole live set as a performer plays it, listened to sample by sample.

A phone on the USB-C port streams into the board (here: the host plays a
pilot tone and a timecode into UAC OUT, routed by the board's mixer onto the
recording bus) while someone builds a beat and performs it: a sequencer scene
recorded with count-in and grown by overdub, scenes doubled, a pad erased, a
song of scenes, pads struck in every style (one pad retriggered, four
alternating, all sixteen with chords), every per-pad feature of a kit (pitch,
reverse, filters, pan, volume, choke, hold, loop, layers, slices), kit swaps
between sampler kits and into the pitched engine, previews, the desk switched
between presets, MIDI clock with Stop/Continue/Start from the host, an app
switch, the phone pausing its stream, bounces into the audio song. The board's
recording bus is captured digitally over USB IN for the whole time (preset
``sampler``: a frame of the take is a frame of the render).

Every sound of the fixture kits (``jam.spec``) has a frequency of its own, so
the take says which pad played, from which kit, when -- and ``jam.analysis``
holds it against what the model (``jam.model``) says the board should have
played: a missing or late hit, a hole, a cut, a click, a ghost, a slipped or
broken stream, each named with its moment on every clock (take, host, wall,
the board's log), the script step and the action it belongs to, what the
board printed around it, and a cause. A board counter that moved (underruns,
drops, a reset) is a failure even when the take does not show it.

The set (``jam.script``) runs in USB-audio blocks with short CDC windows in
between, because in the audio profile only SysEx reaches the board: the
sequencer's transport, song, desk and readbacks are CDC-only (the companion
allowlist has no SEQ/ASONG/MIX). Everything the set changes on the board is
read first, journalled (``jam.restore``) and put back at the end -- the
sequencer file (``sequences.json`` saves itself), the desk (NVS), the preset
slots, the song project -- and the board is restarted so nothing of the jam
stays in RAM.

Verdict: ``ZERO-GLITCH PASS`` (exit 0), ``FAIL`` (exit 1, every glitch with its
moment and cause) or ``not measured`` (exit 2: the host lost audio, a
capability is missing, a clock could not be fitted). ``--analyze-only DIR``
re-analyses a run; ``--dry-run`` prints the set without a board.
"""
from __future__ import annotations

import json
import math
import struct
import subprocess
import sys
import time
import wave
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from crosspad_hil import card_hold, fs, pins, verbs
from crosspad_hil.devices import UsbMode
from crosspad_hil.errors import (
    BAD_ARGS,
    CANCELLED,
    ENV,
    NO_CDC_IN_AUDIO_MODE,
    NOT_SUPPORTED,
    TIMEOUT,
    HilError,
)
from crosspad_hil.jam import restore as RS
from crosspad_hil.jam import script as J
from crosspad_hil.jam import spec as S
from crosspad_hil.jam.actions import ActionLog
from crosspad_hil.jam.calibration import Calibration
from crosspad_hil.jam.control import Capabilities, UacLink
from crosspad_hil.jam.performer import BlockRun, Performer, Prober
from crosspad_hil.midi import MidiIO, MidiRole, query_route
from crosspad_hil.scenarios import _audio
from crosspad_hil.scenarios.base import Artifact, Context, Progress, Report, register
from crosspad_hil.usbmode import set_mode

SETUP_CODES = frozenset({ENV, NOT_SUPPORTED, TIMEOUT, NO_CDC_IN_AUDIO_MODE})
SEQ_TIMEOUT_S = 10.0
SAMPLER_APP = "Sampler"
PROJECT = RS.PROJECT
CAPTURE_BUS = 2
STREAM_SETTLE_S = 1.5
TAIL_S = 2.0
HUNG_TAIL_LINES = 40
DESK_SLOTS = {"JAM-DUCK": 4, "JAM-FULL": 5}
HOME_SLOT = 6
CDC_LAT_S = 0.003
MAX_SCENES = 8
# Console lines that mean the board is hung even though the console still talks.
HANG_SIGNS = r"task_wdt|no free block|Guru Meditation|abort\(\)|LoadProhibited|StoreProhibited"


# ------------------------------------------------------------------ seams (tests replace these)

def _now() -> float:
    return time.monotonic()


def _sleep(s: float) -> None:
    if s > 0:
        time.sleep(s)


def open_capture(device: Any, **kw: Any) -> Any:
    from crosspad_hil.capture import CaptureSession
    from crosspad_hil.duplex import AlsaDuplexBackend
    from crosspad_hil.jam.pilot import PilotSource

    spec: S.PilotSpec = kw.pop("pilot")
    backend = AlsaDuplexBackend(latency_ms=kw.pop("latency_ms"),
                                source_factory=lambda rate, ch: PilotSource(spec, rate, ch).render,
                                decimate_s=0.01)
    return CaptureSession(device, backend=backend, **kw)


def open_midi(device: Any) -> Any:
    return MidiIO(device, MidiRole.ESP)


def pull_file(ctx: Context, remote: str, local: Path, *, offset: int = 0,
              length: int | None = None) -> dict[str, Any]:
    return fs.pull(ctx.device, remote, local, offset=offset, length=length,
                   serial_factory=ctx.serial_factory)


def push_file(ctx: Context, local: Path, remote: str) -> dict[str, Any]:
    return fs.push(ctx.device, local, remote, serial_factory=ctx.serial_factory)


def start_monitor(wav: Path, out: Path) -> Any:
    """The live monitor, a process of its own (the duplex thread keeps the GIL)."""
    return subprocess.Popen([sys.executable, "-m", "crosspad_hil.jam.monitor", str(wav),
                             str(out)], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


# ------------------------------------------------------------------ parameters

@dataclass
class Params:
    minutes: float = field(default=12.5, metadata={
        "help": "minutes of USB-audio blocks (the set scales to it; rates do not)"})
    seed: int = field(default=1, metadata={"help": "the set's seed: same seed, same set"})
    sections: list[str] = field(default_factory=lambda: ["all"], metadata={
        "help": "all, or some of: build,stream,bounce,clock,ui (the rest is skipped, "
                "and says so)"})
    soak_min: float = field(default=0.0, metadata={
        "help": "a continuous USB-audio block this long at the end (the known underrun "
                "shows after 5-6 min of one session); 0 = none"})
    bpm: int = field(default=120, metadata={"help": "the scenes' tempo"})
    pilot_hz: float = field(default=12_000.0, metadata={"help": "the stream's pilot tone"})
    pilot_dbfs: float = field(default=-26.0, metadata={"help": "its level"})
    timecode: bool = field(default=True, metadata={
        "help": "a timecode under the pilot: lost/repeated samples counted exactly"})
    duplex_latency_ms: float = field(default=80.0, metadata={
        "help": "host buffer of the duplex stream (each direction)"})
    stream_pause_mode: str = field(default="drop", metadata={
        "help": "how the phone pauses its stream: drop (stop the OUT PCM) or silence"})
    strict: bool = field(default=False, metadata={
        "help": "no hit held by a swap's reload at all (default: 1 per swap, <= 50 ms)"})
    late_ms: float = field(default=10.0, metadata={"help": "a hit later than the path's "
                                                           "median by this is LATE"})
    seq_late_ms: float = field(default=8.0, metadata={"help": "the same for sequencer events"})
    ghost_db: float = field(default=-35.0, metadata={"help": "sound where none is due"})
    allow_partial: bool = field(default=False, metadata={
        "help": "a feature the board cannot do is skipped, not 'not measured'"})
    analyze_only: str = field(default="", metadata={
        "help": "re-analyse this run directory; no board"})
    dry_run: bool = field(default=False, metadata={"help": "print the set; no board"})
    calibrate: bool = field(default=False, metadata={
        "help": "write the measured latencies/levels as this firmware's calibration"})
    snippets: int = field(default=40, metadata={"help": "glitch sound snippets to cut"})
    monitor: bool = field(default=True, metadata={"help": "the live monitor process"})
    probes: bool = field(default=True, metadata={
        "help": "echo / KIT_STATUS / UI_STATE probes during the blocks"})
    audible: bool = field(default=False, metadata={
        "help": "leave Out 1/2 on in the jam desks (default: muted, the take is unchanged)"})
    recorder: bool = field(default=True, metadata={
        "help": "the Recorder (STOOL IN PADS) takes block B2 to the card as well"})
    restore_sequences: bool = field(default=True, metadata={
        "help": "put sequences.json back from the copy and restart the board at the end"})
    exclusive_card: bool = field(default=True, metadata={
        "help": "pause the audio-nodes service for each block (bench holder only)"})
    reset_first: bool = field(default=True, metadata={"help": "reset the board first"})
    enum_timeout: float = field(default=45.0, metadata={"help": "seconds per profile switch"})
    no_provision: bool = field(default=False, metadata={
        "help": "fail instead of pushing missing fixture kits"})


# ------------------------------------------------------------------ the run's state

@dataclass
class JamState:
    script: J.JamScript
    log: ActionLog
    workdir: Path = Path(".")
    params: Params = field(default_factory=Params)
    con: Any = None
    home: RS.HomeState | None = None
    journalled: bool = False
    kits: dict[str, int] = field(default_factory=dict)
    scenes: dict[str, int] = field(default_factory=dict)
    caps: Capabilities = field(default_factory=Capabilities)
    desk: dict[str, Any] = field(default_factory=dict)       # the desk JSON of the jam's slots
    desk_now: str = ""
    kit_now: str = S.JAM_A
    fx_loaded: bool = False
    bounces: list[dict[str, Any]] = field(default_factory=list)
    pending_pulls: list[dict[str, Any]] = field(default_factory=list)
    rec_checks: list[dict[str, Any]] = field(default_factory=list)
    recorder: dict[str, Any] = field(default_factory=dict)
    blocks_done: list[str] = field(default_factory=list)
    out_vol: float = 1.0
    project0: str | None = None


def _seq(link: Any, action: str | None = None) -> dict[str, Any]:
    return verbs.seq(link, action, timeout_s=SEQ_TIMEOUT_S)


def _scene_now(st: dict[str, Any]) -> tuple[int, int]:
    a, _, b = str(st.get("scene", "0>0")).partition(">")
    try:
        return int(a), int(b or a)
    except ValueError:
        return -1, -1


def _wait(ctx: Context, seconds: float) -> None:
    end = _now() + max(0.0, seconds)
    while True:
        ctx.check_cancelled()
        left = end - _now()
        if left <= 0:
            return
        _sleep(min(left, 0.05))


# ------------------------------------------------------------------ the scenario

class LiveJamScenario:
    name = "live_jam"
    Params = Params
    description = ("a whole live set: the phone's stream + pads in every style, a beat built "
                   "and performed (REC/overdub/scenes/song/MIDI clock), every kit feature, "
                   "swaps, bounces; every glitch with its moment, step and cause")

    def run(self, ctx: Context, params: Params, progress: Progress) -> Report:
        p = params
        if p.stream_pause_mode not in ("drop", "silence"):
            raise HilError(BAD_ARGS, f"--stream-pause-mode {p.stream_pause_mode!r}",
                           hint="drop or silence")
        if p.analyze_only:
            return self._analyse(ctx, p, Path(p.analyze_only), [], data={})
        script = J.build_script(p.seed, minutes=p.minutes, bpm=p.bpm, soak_min=p.soak_min,
                                sections=p.sections, stream_pause_mode=p.stream_pause_mode)
        ctx.workdir.mkdir(parents=True, exist_ok=True)
        if p.dry_run:
            table = J.script_table(script)
            (ctx.workdir / "script.txt").write_text(table + "\n", encoding="utf-8")
            return Report(passed=True, summary=table, data={"script": script.to_dict()},
                          artifacts=[], exit_code=0)
        log = ActionLog(ctx.workdir / "actions.jsonl", clock=_now)
        st = JamState(script=script, log=log, workdir=ctx.workdir, params=p)
        data: dict[str, Any] = {"seed": p.seed, "minutes": p.minutes, "blocks": {},
                                "windows": {}, "failures": [], "left": {}, "env": None}
        ctx.partial = data
        artifacts: list[Artifact] = []
        env: str | None = None
        try:
            try:
                self._prep(ctx, p, st, data, progress)
                st.script = J.build_script(p.seed, minutes=p.minutes, bpm=p.bpm,
                                           soak_min=p.soak_min, sections=p.sections,
                                           caps=st.caps.as_needs(),
                                           stream_pause_mode=p.stream_pause_mode)
                self._write_script(ctx, st)
                total = len(st.script.order)
                for i, name in enumerate(st.script.order[1:], start=1):
                    progress(i, total, name)
                    if name in st.script.windows and name != "FINAL":
                        self._window(ctx, p, st, data, name)
                    elif name == "FINAL":
                        self._final(ctx, p, st, data)
                    else:
                        self._block(ctx, p, st, data, st.script.block(name))
            except HilError as e:
                if e.code == CANCELLED or e.code not in SETUP_CODES:
                    raise
                env = f"{e.code}: {e.message}" + (f" ({e.hint})" if e.hint else "")
                data["env"] = env
                ctx.log(f"live_jam: {env}")
            finally:
                self._put_back(ctx, p, st, data)
                log.close()
                self._write_side_files(ctx, st, data)
            return self._analyse(ctx, p, ctx.workdir, artifacts, data=data, env=env)
        finally:
            ctx.close_all()

    # ---------------------------------------------------------------- PREP
    def _prep(self, ctx: Context, p: Params, st: JamState, data: dict[str, Any],
              progress: Progress) -> None:
        progress(0, None, "PREP: normalising the bench")
        log = st.log
        log.block, log.step = "PREP", "PREP"
        self._default(ctx, p)
        con = ctx.open_console(reset=p.reset_first)
        st.con = con
        if hasattr(con, "start_tsv"):
            con.start_tsv(ctx.workdir / "console.tsv")
        log._console_seq = lambda: getattr(con, "seq", None)
        if p.reset_first:
            boot = con.wait_boot(timeout_s=60.0, settle_s=2.0)
            data["boot_complete"] = bool(getattr(boot, "complete", False))
            from crosspad_hil.scenarios.live_set import parse_sd_boot

            data["sd"] = parse_sd_boot([line for _s, line in con.read(
                since_seq=0, match="bsp_sdmmc", limit=200).lines])
        stale: RS.HomeState | None = None
        link = ctx.open_cdc()
        try:
            data["pins"] = pins.restore_pending(link, ctx.device.id, ctx.log,
                                                keys=tuple(k for k in pins.pending(ctx.device.id)
                                                           if k != RS.PIN_KEY))
            data["host_pins"] = pins.restore_pending(None, card_hold.PIN_DEVICE, ctx.log)
            stale = RS.pending_home(ctx.device.id)
            if stale is not None:
                ctx.log("pins: a live_jam run did not put the board back -- doing it now")
                data["restored_from_journal"] = RS.put_back_ram(
                    link, stale, jam_scenes=list(range(stale.scenes0, MAX_SCENES)),
                    kit_id_by_name=lambda n: self._kit_id(link, n), log=ctx.log)
        finally:
            link.close()
        if stale is not None:
            data["restored_from_journal"].update(RS.put_back_files(
                stale, lambda loc, r: push_file(ctx, loc, r), lambda r: self._delete(ctx, r),
                log=ctx.log))
            if not RS.all_put_back(data["restored_from_journal"]):
                # a new snapshot now would take the last jam's leftovers for "home"
                raise HilError(ENV, "the last live_jam run's board state could not be put "
                                    f"back: {data['restored_from_journal']}",
                               hint="the journal is kept; fix the step that failed and rerun")
            RS.unjournal(ctx.device.id)
            con.reset()
            con.wait_boot(timeout_s=60.0, settle_s=2.0)
            self._refresh(ctx)
        # The jam's kits first: provisioning reboots the board, and a snapshot taken
        # before that reboot would describe a sequencer that has not read the card.
        link = ctx.open_cdc()
        try:
            link, data["provisioned"] = _audio.ensure_kits(
                ctx, link, list(S.KIT_NAMES), S.write_jam_kits, console=con,
                provision=not p.no_provision,
                missing_hint_suffix=" -- built by crosspad_hil.jam.spec")
        finally:
            link.close()
        link = ctx.open_cdc()
        try:
            # The sequence store reads sequences.json only once a kit has been loaded
            # after the boot. With no kit up, the jam's own kit A is loaded (the home
            # kit, none, is what gets put back), then the snapshot waits for the file.
            home_kit = verbs.kit_status(link)
            if str(verbs.seq(link, timeout_s=SEQ_TIMEOUT_S).get("file") or "") in ("", "-"):
                kid, _n, _cur = _audio.resolve_kit(link, kit_name=S.JAM_A)
                _audio.load_kit_waiting(ctx, link, int(kid))
            RS.wait_seq_store(link)
            home = RS.snapshot(link, log=ctx.log)
            home.kit = {"id": home_kit.get("current"), "name": home_kit.get("name")}
            st.home = home
            if home.scenes0 + 3 > MAX_SCENES:
                raise HilError(ENV, f"the sequencer holds {home.scenes0} scenes: the jam "
                                    "needs 3 free of 8", hint="delete a scene you do not need")
            log.emit("window_op", op="home snapshot", scenes0=home.scenes0,
                     kit=home.kit, app=home.app, project=home.project)
            fw = {}
            for key, fn in (("app_sha", verbs.app_sha), ("stm_fw", verbs.stm_fw_info)):
                try:
                    fw[key] = fn(link)
                except HilError as e:
                    fw[key] = f"{e.code}: {e.message}"
            data["firmware"] = fw
            st.caps.fw_sha = _fw_sha(fw)
        finally:
            link.close()
        # copies of what will be overwritten, journalled before anything changes
        RS.fetch_files(home, lambda r, loc: pull_file(ctx, r, loc), ctx.workdir / "home")
        RS.journal(ctx.device.id, home)
        st.journalled = True
        (ctx.workdir / "home.json").write_text(json.dumps(home.to_dict(), indent=1,
                                                          default=str), encoding="utf-8")
        link = ctx.open_cdc()
        try:
            for name in S.KIT_NAMES:
                kid, _n, _cur = _audio.resolve_kit(link, kit_name=name)
                st.kits[name] = int(kid)
            # the FX kit renders its .fx files on its first load: make that happen in B2
            try:
                verbs.fs_delete(link, f"{_kits_root()}/{S.KIT_DIRS[S.JAM_FX]}/.fx")
                st.fx_loaded = False
            except HilError:
                pass
            self._probe(ctx, link, st)
            self._desks(ctx, p, link, st)
            verbs.asong(link, f"OPEN {PROJECT}", timeout_s=SEQ_TIMEOUT_S)
            self._poll_asong(ctx, link, lambda r: r.get("open") in (2, 3), 60.0)
            _audio.load_kit_waiting(ctx, link, st.kits[S.JAM_A])
            st.kit_now = S.JAM_A
            _audio.start_app(ctx, link, SAMPLER_APP)
            n = home.scenes0
            st.scenes = {"S1": n, "S2": n + 1, "S3": n + 2}
            log.emit("window_op", op="prep done", kits=st.kits, scenes=st.scenes)
        finally:
            link.close()
        self._pulls(ctx, st)

    def _probe(self, ctx: Context, link: Any, st: JamState) -> None:
        caps = st.caps
        caps.stm_midi = ctx.device.ports.stm_midi is not None
        caps.sysex_bounce = True          # 1D 0C is in 724d3ad; refused is logged on the console
        caps.companion = ctx.device.ports.esp_midi is not None
        io = None
        try:
            if ctx.device.ports.esp_midi is not None:
                io = open_midi(ctx.device)
                io.open()
                from crosspad_hil.midi import companion_command

                r = companion_command(io, "CP_CAPS", timeout_s=2.0)
                caps.companion = r.startswith("CP_CAPS")
                r = companion_command(io, "SEQ", timeout_s=2.0)
                caps.companion_seq = r.startswith("SEQ:")
        except HilError as e:
            caps.notes.append(f"companion probe: {e.code}: {e.message}")
            caps.companion = False
        finally:
            if io is not None:
                io.close()
        st.log.emit("window_op", op="capabilities", **caps.as_needs())

    def _desks(self, ctx: Context, p: Params, link: Any, st: JamState) -> None:
        """HILHOME (the desk as found) into slot 6; JAM-DUCK/-FULL built from it into 4/5."""
        home = st.home
        assert home is not None
        verbs.mix_preset(link, f"SAVE {HOME_SLOT} HILHOME")
        lvl = verbs.mix_lvl(link)
        outs = {int(o["out"]): o for o in lvl.get("outputs", [])}
        st.out_vol = float((outs.get(CAPTURE_BUS) or {}).get("vol", 1.0))
        if st.out_vol > 1.5:
            raise HilError(ENV, f"the recording bus gain is {20 * math.log10(st.out_vol):+.1f} dB "
                                "(Settings' DAW gain): the fixtures would clip the take",
                           hint="set the DAW gain to 0 dB for the jam")
        st.desk["levels"] = lvl
        st.log.emit("window_op", op="desk HILHOME saved", out_vol=st.out_vol)
        st.pending_pulls.append({"remote": RS.preset_path(home, HOME_SLOT),
                                 "local": str(ctx.workdir / "home" / "HILHOME.json"),
                                 "then": "desks"})

    def _build_desks(self, ctx: Context, p: Params, st: JamState) -> None:
        """After HILHOME arrived: the two jam desks, pushed to their slots."""
        src = ctx.workdir / "home" / "HILHOME.json"
        try:
            doc = json.loads(src.read_text(encoding="utf-8"))
        except (OSError, ValueError) as e:
            raise HilError(ENV, f"the home desk did not arrive: {e}") from e
        for name, slot in DESK_SLOTS.items():
            d = json.loads(json.dumps(doc))
            d.pop("gains", None)                    # a preset with gains moves INGAIN
            d["name"] = name
            desk = d.get("desk") or {}
            for c in desk.get("channels", []):
                cname = str(c.get("name"))
                c["soloed"] = False
                c["muted"] = False
                c["pan"] = 0.0
                c["volume"] = 1.0
                if name == "JAM-DUCK":
                    c["volume"] = {"USB": 0.5, "Sampler": 0.8}.get(cname, 1.0)
                    c["pan"] = -0.5 if cname == "Pitched" else 0.0
                for r in c.get("routes", []):
                    if int(r.get("out", -1)) == CAPTURE_BUS:
                        r["enabled"] = cname in ("Sampler", "Pitched", "USB")
                        r["volume"] = 1.0
            for i, o in enumerate(desk.get("outputs", [])):
                if i != CAPTURE_BUS and not p.audible:
                    o["muted"] = True
            out = ctx.workdir / "desks" / f"preset_{slot}.json"
            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_text(json.dumps(d), encoding="utf-8")
            push_file(ctx, out, RS.preset_path(st.home, slot))   # type: ignore[arg-type]
            st.desk[name] = desk
            st.log.emit("window_op", op=f"desk {name} -> slot {slot}")

    def _pulls(self, ctx: Context, st: JamState) -> None:
        """Files to fetch now that the CDC link is closed (FS_GET opens the port itself)."""
        todo, st.pending_pulls = st.pending_pulls, []
        for item in todo:
            local = Path(item["local"])
            local.parent.mkdir(parents=True, exist_ok=True)
            try:
                if item.get("then") == "wav_window":
                    _pull_wav_window(ctx, item["remote"], local, int(item["start"]),
                                     int(item["frames"]))
                else:
                    pull_file(ctx, item["remote"], local)
                item["ok"] = True
            except HilError as e:
                item["ok"] = False
                item["err"] = f"{e.code}: {e.message}"
                st.log.emit("pull", ok=False, err=item["err"], remote=item["remote"])
            if item.get("then") == "desks":
                self._build_desks(ctx, st.params, st)
            if item.get("then") == "wav_window":
                st.recorder["window"] = {k: item.get(k) for k in ("local", "start", "frames",
                                                                  "ok", "err")}
            for b in st.bounces:
                if item.get("bounce") and b.get("name") == item["bounce"]:
                    b["local"] = str(local.relative_to(ctx.workdir)) if item.get("ok") else None
                    b["fetch_err"] = item.get("err")

    # ---------------------------------------------------------------- CDC windows
    def _window(self, ctx: Context, p: Params, st: JamState, data: dict[str, Any],
                name: str) -> None:
        self._default(ctx, p)
        st.log.block, st.log.step = name, name
        rec: dict[str, Any] = {"ops": []}
        data["windows"][name] = rec
        link = ctx.open_cdc()
        try:
            # after a reset (a hang) the board is in the launcher: the pads play nothing there
            if verbs.app_list(link).get("running") != SAMPLER_APP:
                if st.kit_now in st.kits:
                    _audio.load_kit_waiting(ctx, link, st.kits[st.kit_now])
                _audio.start_app(ctx, link, SAMPLER_APP)
                rec["app_restarted"] = True
            fn = getattr(self, f"_w_{name.lower()}", None)
            if fn is not None:
                fn(ctx, p, st, link, rec)
        except HilError as e:
            if e.code == CANCELLED:
                raise
            rec["error"] = f"{e.code}: {e.message}"
            st.log.emit("window_op", op=f"{name} aborted", ok=False, err=rec["error"])
            ctx.log(f"{name}: {rec['error']}")
            if e.code in (NO_CDC_IN_AUDIO_MODE,):
                raise
        finally:
            link.close()
        self._pulls(ctx, st)

    def _op(self, st: JamState, rec: dict[str, Any], op: str, fn: Any, **args: Any) -> Any:
        t0 = _now()
        try:
            out = fn()
        except HilError as e:
            st.log.emit("window_op", t0=t0, ok=False, err=f"{e.code}: {e.message}", op=op, **args)
            rec["ops"].append({"op": op, "ok": False, "err": f"{e.code}: {e.message}"})
            raise
        st.log.emit("window_op", t0=t0, op=op, **args)
        rec["ops"].append({"op": op, "ok": True})
        return out

    def _guard_scene(self, link: Any, scene: int, what: str) -> dict[str, Any]:
        """REC, ERASE and DOUBLE act on the sounding scene: it must be the jam's."""
        stt = _seq(link)
        active, _pending = _scene_now(stt)
        if active != scene:
            raise HilError(ENV, f"{what}: the sounding scene is {active}, not the jam's {scene}"
                                " -- refusing to touch it",
                           hint="the user's scenes are never recorded over")
        return stt

    def _select_scene(self, ctx: Context, link: Any, scene: int, timeout_s: float = 12.0) -> None:
        _seq(link, f"SCENE {scene}")
        deadline = _now() + timeout_s
        while True:
            active, _ = _scene_now(_seq(link))
            if active == scene:
                return
            if _now() >= deadline:
                raise HilError(TIMEOUT, f"scene {scene} did not become the sounding one")
            _wait(ctx, 0.1)

    def _anchor(self, link: Any) -> dict[str, Any]:
        t0 = _now()
        c = verbs.click(link)
        t1 = _now()
        c["t_host"] = (t0 + t1) / 2.0
        return c

    def _record(self, ctx: Context, p: Params, st: JamState, link: Any, scene: int, bars: int,
                hits: list[tuple[float, int]], *, count_in: bool, early: tuple[int, ...] = ()
                ) -> dict[str, Any]:
        """Record ``hits`` ((beat, pad), from the pass start) into ``scene`` by PAD_PRESS,
        on the host's clock against the CLICK anchor; returns what was meant to land."""
        sr_beat = 60.0 / p.bpm
        self._guard_scene(link, scene, "REC")
        t_rec = _now()
        _seq(link, "REC")
        st.log.emit("window_op", op=f"REC scene {scene}", t0=t_rec)
        if count_in:
            # one bar of count-in; a hit a 32nd early for beat one is beat one (grace):
            # every hit on beat one goes then, the anchor is read only after the downbeat
            early = tuple(dict.fromkeys([*early, *(pad for beat, pad in hits if beat == 0)]))
            t_down = t_rec + 4 * sr_beat
            for pad in early:
                _wait(ctx, t_down - 0.03 - _now())
                self._press(st, link, pad)
            _wait(ctx, t_down + 0.15 - _now())
        c = self._anchor(link)
        loop = int(c.get("loop") or 0)
        pos = int(c.get("pos") or 0)
        sr = loop / (bars * 4 * sr_beat) if loop else 44_100.0
        t_pass = c["t_host"] - pos / sr
        if count_in:
            if pos > 0.5 * loop:
                t_pass += loop / sr                 # still before the first downbeat
        elif pos / sr > 0.05:
            t_pass += loop / sr                     # into a running scene: from its next pass
        intended = []
        for beat, pad in hits:
            if count_in and beat == 0 and pad in early:
                intended.append({"pad": pad, "beat": beat})
                continue
            _wait(ctx, t_pass + beat * sr_beat - CDC_LAT_S - _now())
            self._press(st, link, pad)
            intended.append({"pad": pad, "beat": beat})
        _wait(ctx, t_pass + bars * 4 * sr_beat + 0.1 - _now())
        _seq(link, "PLAY")
        st.log.emit("window_op", op=f"REC scene {scene} closed")
        return {"scene": scene, "intended": intended, "spb_s": sr_beat}

    def _press(self, st: JamState, link: Any, pad: int) -> None:
        t0 = _now()
        try:
            verbs.pad_press(link, pad, 100)
            verbs.pad_release(link, pad)
            st.log.emit("pad_cdc", t0=t0, pad=pad, vel=100, via="cdc", block=st.log.block)
        except HilError as e:
            st.log.emit("pad_cdc", t0=t0, ok=False, err=e.message, pad=pad)

    def _check_rec(self, st: JamState, link: Any, meant: dict[str, Any], quant: int,
                   window: str) -> None:
        """What the take recorded against what was played (grid lines of the scene)."""
        from crosspad_hil.jam.model import grid_per_beat

        dump = [e for e in verbs.seq_dump(link) if e["seq"] == meant["scene"]]
        c = verbs.click(link)
        bars_loop = int(c.get("loop") or 0)
        spb = bars_loop // max(1, int(c.get("bars") or 1) * 4) if bars_loop else 22_050
        per = grid_per_beat(quant) or 1
        want = sorted((h["pad"], int(round(h["beat"] * per)) * spb // per)
                      for h in meant["intended"])
        got = sorted((e["pad"], e["pos"]) for e in dump)
        missing = [w for w in want if w not in got]
        extra = [g for g in got if g not in want]
        st.rec_checks.append({"window": window, "scene": meant["scene"], "want": len(want),
                              "got": len(got), "missing": missing[:20], "extra": extra[:20]})

    def _w_w0(self, ctx: Context, p: Params, st: JamState, link: Any, rec: dict) -> None:
        """S1: count-in, the base groove by PAD_PRESS (kick a 32nd early for beat one)."""
        s1 = st.scenes["S1"]
        self._op(st, rec, "SEQ STOP", lambda: _seq(link, "STOP"))
        got = self._op(st, rec, "SEQ ADD", lambda: _seq(link, "ADD"))
        if int(got.get("scenes") or 0) != s1 + 1:
            raise HilError(ENV, f"SEQ ADD gave {got.get('scenes')} scenes, {s1 + 1} expected")
        self._op(st, rec, "SCENE S1", lambda: self._select_scene(ctx, link, s1))
        self._guard_scene(link, s1, "BARS")
        for op in (f"BARS {s1} 2", f"BPM {p.bpm}", "QUANT 2", "COUNTIN 1"):
            self._op(st, rec, f"SEQ {op}", lambda op=op: _seq(link, op))
        hits = [(0.0, 0), (2.0, 0), (4.0, 0), (6.0, 0)]                 # kick on 1 and 3
        hits += [(b + 1.0, 1) for b in (0, 2, 4, 6)]                    # snare on 2 and 4
        hits += [(k * 0.5, 2) for k in range(16)]                       # hat on the 8ths
        hits += [(3.5, 3), (7.5, 3)]                                    # open hat, the 'and' of 4
        hits.sort()
        meant = self._op(st, rec, "REC S1", lambda: self._record(
            ctx, p, st, link, s1, 2, hits, count_in=True, early=(0,)))
        self._check_rec(st, link, meant, 2, "W0")
        _audio.load_kit_waiting(ctx, link, st.kits[S.JAM_A])
        st.kit_now = S.JAM_A
        self._load_desk(st, rec, link, "JAM-FULL")

    def _load_desk(self, st: JamState, rec: dict, link: Any, name: str) -> None:
        slot = DESK_SLOTS[name]
        # a slot pushed over FS_PUT is unknown to the Mixer until a refresh: bare first
        listing = verbs.mix_preset(link)
        if (listing.get("slots") or {}).get(slot, {}).get("name") != name:
            raise HilError(ENV, f"preset slot {slot} is not {name} on the card",
                           hint=str(listing.get("line")))
        r = self._op(st, rec, f"MIX_PRESET LOAD {slot} ({name})",
                     lambda: verbs.mix_preset(link, f"LOAD {slot}"))
        if not r.get("ok"):
            raise HilError(ENV, f"MIX_PRESET LOAD {slot} refused: {r.get('line')}")
        st.desk_now = name

    def _w_w1a(self, ctx: Context, p: Params, st: JamState, link: Any, rec: dict) -> None:
        """Arm the overdub on S1 for the short block B1b (no hit of B1 may land in S1)."""
        s1 = st.scenes["S1"]
        self._op(st, rec, "SEQ PLAY", lambda: _seq(link, "PLAY"))
        self._op(st, rec, "SCENE S1", lambda: self._select_scene(ctx, link, s1))
        self._guard_scene(link, s1, "OVERDUB")
        self._op(st, rec, "SEQ OVERDUB", lambda: _seq(link, "OVERDUB"))

    def _w_w1(self, ctx: Context, p: Params, st: JamState, link: Any, rec: dict) -> None:
        s1, s2 = st.scenes["S1"], st.scenes["S2"]
        self._op(st, rec, "SEQ PLAY (closes the overdub)", lambda: _seq(link, "PLAY"))
        self._op(st, rec, "SEQ ADD", lambda: _seq(link, "ADD"))
        self._op(st, rec, "SCENE S2", lambda: self._select_scene(ctx, link, s2))
        self._guard_scene(link, s2, "BARS")
        self._op(st, rec, f"SEQ BARS {s2} 2", lambda: _seq(link, f"BARS {s2} 2"))
        groove = [(b * 1.0, 8 + (b % 4)) for b in range(8)] + \
                 [(b + 0.75, 10) for b in (1, 3, 5, 7)]
        groove.sort()
        meant = self._op(st, rec, "REC S2", lambda: self._record(
            ctx, p, st, link, s2, 2, groove, count_in=False))
        self._check_rec(st, link, meant, 2, "W1")
        self._guard_scene(link, s2, "DOUBLE")
        self._op(st, rec, "SEQ DOUBLE", lambda: _seq(link, "DOUBLE"))
        self._op(st, rec, "SCENE S1", lambda: self._select_scene(ctx, link, s1))
        self._guard_scene(link, s1, "ERASE")
        self._op(st, rec, "ERASE pad 3 of S1", lambda: (
            _seq(link, "ERASE 1"), self._press(st, link, 3), _wait(ctx, 0.1),
            _seq(link, "ERASE 0")))
        for op in ("CLEAR", f"ADD {s1} 2", f"ADD {s2} 1", "LOOP 1", "PLAY"):
            self._op(st, rec, f"SONG {op}", lambda op=op: verbs.song_text(link, op))
        self._load_desk(st, rec, link, "JAM-DUCK")
        if p.recorder:
            try:
                self._op(st, rec, "STOOL IN PADS", lambda: verbs.stool(link, "IN PADS"))
                r = self._op(st, rec, "STOOL REC", lambda: verbs.stool(link, "REC"))
                st.recorder = {"started": True, "t": _now(), "sr": r.get("sr")}
            except HilError as e:
                st.recorder = {"started": False, "err": f"{e.code}: {e.message}"}

    def _w_w2(self, ctx: Context, p: Params, st: JamState, link: Any, rec: dict) -> None:
        s1, s3 = st.scenes["S1"], st.scenes["S3"]
        if st.recorder.get("started"):
            try:
                r = self._op(st, rec, "STOOL STOP", lambda: verbs.stool(link, "STOP"))
                st.recorder.update(stopped=True, file=r.get("file"), frames=r.get("frames"),
                                   sr=r.get("sr") or st.recorder.get("sr"))
                frames = int(r.get("frames") or 0)
                if r.get("file") and frames > 0:
                    # a probe, not the whole take: 10 s from the middle, against B2's capture
                    st.pending_pulls.append({"remote": str(r["file"]), "then": "wav_window",
                                             "local": str(st.workdir / "recorder"
                                                          / "window.wav"),
                                             "start": frames // 2,
                                             "frames": int(10 * (r.get("sr") or 44_100))})
            except HilError as e:
                st.recorder["stop_err"] = f"{e.code}: {e.message}"
        # bounce #3 over CDC, while CDC pads play
        a = verbs.asong(link, timeout_s=SEQ_TIMEOUT_S)
        if a.get("project") == PROJECT:
            clips0 = list(a.get("clips") or [])
            t0 = _now()
            self._op(st, rec, f"ASONG BOUNCE {s1} 2", lambda: verbs.asong(
                link, f"BOUNCE {s1} 2", timeout_s=SEQ_TIMEOUT_S))
            for k in range(8):
                self._press(st, link, 8 + k % 4)
                _wait(ctx, 0.25)
            done = self._poll_asong(ctx, link, lambda r: r.get("bounce") in (2, 3), 60.0)
            self._bounce_entry(st, "bounce3", None, s1, 2, "cdc", clips0, done, t0)
            self._op(st, rec, "SEQ PLAY", lambda: _seq(link, "PLAY"))
        else:
            rec["bounce3"] = f"not bounced: the open project is {a.get('project')!r}"
        self._op(st, rec, "SONG CLEAR", lambda: verbs.song_text(link, "CLEAR"))
        self._op(st, rec, "SEQ ADD", lambda: _seq(link, "ADD"))
        self._op(st, rec, "SCENE S3", lambda: self._select_scene(ctx, link, s3))
        self._guard_scene(link, s3, "BARS")
        for op in (f"BARS {s3} 1", "QUANT 4"):
            self._op(st, rec, f"SEQ {op}", lambda op=op: _seq(link, op))
        trip = [(k / 3.0, k % 4) for k in range(12)]
        meant = self._op(st, rec, "REC S3", lambda: self._record(
            ctx, p, st, link, s3, 1, trip, count_in=False))
        self._check_rec(st, link, meant, 4, "W2")
        self._load_desk(st, rec, link, "JAM-FULL")

    def _w_w3(self, ctx: Context, p: Params, st: JamState, link: Any, rec: dict) -> None:
        s1 = st.scenes["S1"]
        self._op(st, rec, "SONG CLEAR", lambda: verbs.song_text(link, "CLEAR"))
        self._op(st, rec, "SCENE S1", lambda: self._select_scene(ctx, link, s1))
        self._op(st, rec, "SEQ PLAY", lambda: _seq(link, "PLAY"))
        if st.kit_now != S.JAM_A:
            _audio.load_kit_waiting(ctx, link, st.kits[S.JAM_A])
            st.kit_now = S.JAM_A

    def _final(self, ctx: Context, p: Params, st: JamState, data: dict[str, Any]) -> None:
        self._default(ctx, p)
        st.log.block, st.log.step = "FINAL", "FINAL"
        link = ctx.open_cdc()
        out: dict[str, Any] = {}
        try:
            for key, fn in (("reset_log", verbs.reset_count), ("mem", verbs.mem),
                            ("worker", verbs.worker), ("pitched", verbs.pitched_status),
                            ("arena", verbs.arena), ("clock", verbs.clock_status),
                            ("native", verbs.smpl_engine), ("pool", verbs.smpl_pool)):
                try:
                    out[key] = fn(link)
                except HilError as e:
                    out[key] = f"{e.code}: {e.message}"
        finally:
            link.close()
        data["final"] = out
        self._pulls(ctx, st)

    # ---------------------------------------------------------------- USB-audio blocks
    def _readback(self, ctx: Context, link: Any, *, anchor: bool) -> dict[str, Any]:
        rb: dict[str, Any] = {"counters": {}}
        for key, fn in (("seq", lambda: _seq(link)), ("seq_dump", lambda: verbs.seq_dump(link)),
                        ("song", lambda: verbs.song_text(link)),
                        ("kit", lambda: verbs.kit_status(link)),
                        ("mix", lambda: verbs.mix_lvl(link)),
                        ("clock_status", lambda: verbs.clock_status(link)),
                        ("smpl_last", lambda: verbs.smpl_last(link)),
                        ("swap_log", lambda: verbs.kit_swap_log(link, log=ctx.log)),
                        ("asong", lambda: verbs.asong(link, timeout_s=SEQ_TIMEOUT_S))):
            try:
                rb[key] = fn()
            except HilError as e:
                rb[key] = None
                rb.setdefault("errors", {})[key] = f"{e.code}: {e.message}"
        for key, fn in (("smpl", verbs.smpl_counters), ("pads", verbs.pad_stats),
                        ("pitched", verbs.pitched_status), ("native", verbs.smpl_engine),
                        ("pool", verbs.smpl_pool), ("resets", verbs.reset_count)):
            try:
                rb["counters"][key] = fn(link)
            except HilError as e:
                rb.setdefault("errors", {})[f"counters.{key}"] = f"{e.code}: {e.message}"
        rb["counters"]["asong"] = rb.get("asong") or {}
        if anchor:
            try:
                rb["click"] = self._anchor(link)       # last: closest to the switch
            except HilError as e:
                rb.setdefault("errors", {})["click"] = f"{e.code}: {e.message}"
        return rb

    def _block(self, ctx: Context, p: Params, st: JamState, data: dict[str, Any],
               block: J.Block) -> None:
        name = block.name
        bdir = ctx.workdir / "blocks" / name
        bdir.mkdir(parents=True, exist_ok=True)
        st.log.block, st.log.step = name, f"{name}.prep"
        doc: dict[str, Any] = {"name": name, "wav": f"blocks/{name}/capture.wav",
                               "kit0": st.kit_now, "fx_first_load": not st.fx_loaded,
                               "pilot": self._pilot(p).to_dict()}
        data["blocks"][name] = doc
        self._default(ctx, p)
        link = ctx.open_cdc()
        try:
            if any(s.action == "bounce" and not s.skip for s in block.steps):
                a = verbs.asong(link, timeout_s=SEQ_TIMEOUT_S)
                doc["asong_before"] = a
                if a.get("project") != PROJECT:
                    # a bounce would land in someone's song: not this block's bounce
                    for s in block.steps:
                        if s.action == "bounce":
                            s.skip = f"the open project is {a.get('project')!r}, not {PROJECT}"
            rows0 = verbs.kit_swap_log(link, log=ctx.log)
            doc["swap_seq0"] = max((int(r.get("seq", -1)) for r in rows0), default=-1)
            route = None
            doc["readback_before"] = rb = self._readback(ctx, link, anchor=True)
            kit_name = str((rb.get("kit") or {}).get("name") or "")
            if kit_name in S.KIT_NAMES:
                doc["kit0"] = st.kit_now = kit_name      # what the board says, after a reset too
        finally:
            link.close()
        doc["desk"] = self._desk_doc(st)
        (bdir / "block.json").write_text(json.dumps(doc, default=str), encoding="utf-8")
        run = BlockRun()
        hung: str | None = None
        ctx.device = set_mode(ctx.device, UsbMode.AUDIO, wait=True, timeout_s=p.enum_timeout)
        con_seq0 = getattr(st.con, "seq", 0)
        with card_hold.for_run(ctx, enabled=p.exclusive_card,
                               expected_s=block.duration + 30) as hold:
            doc["card_hold"] = hold
            session = io = link_u = prober = mon = None
            try:
                session = open_capture(ctx.device, seconds=None, preset="sampler",
                                       out=bdir / "capture.wav", keep_mode=True,
                                       log=ctx.log, logdir=ctx.workdir, pilot=self._pilot(p),
                                       latency_ms=p.duplex_latency_ms)
                info = session.start()
                ctx.opened.append(session)
                rec = getattr(getattr(session, "_backend", None), "recording", None)
                doc["capture"] = {k: v for k, v in info.items() if k != "capture_device"}
                doc["t_capture_start"] = getattr(session, "started_at", None)
                doc["t_stream_start"] = getattr(rec, "started_at", doc["t_capture_start"])
                if p.monitor:
                    try:
                        mon = start_monitor(bdir / "capture.wav", bdir / "live_events.jsonl")
                    except OSError as e:
                        doc["monitor"] = f"not started: {e}"
                io = open_midi(ctx.device)
                io.open()
                try:
                    route = query_route(io, timeout_s=0.6)
                except HilError:
                    route = None
                link_u = UacLink(io, st.log, clock=_now)
                link_u.start()
                t0 = _now() + STREAM_SETTLE_S
                perf = Performer(link_u, st.log, kits=st.kits, scenes=st.scenes, clock=_now,
                                 sleep=_sleep, cancelled=ctx.cancelled,
                                 watchdog=self._watchdog(st, rec, link_u),
                                 recording=rec, route_home=route)
                events = perf.plan(block, t0)
                if p.probes:
                    swaps = [(t0 + s.t_rel - 3, t0 + s.t_rel + 3) for s in block.steps
                             if s.action == "swap"]
                    uis = [(t0 + s.t_rel, t0 + s.t_rel + s.dur + 2) for s in block.steps
                           if s.action == "ui"]
                    prober = Prober(link_u, st.log, swap_windows=swaps, ui_windows=uis,
                                    clock=_now, companion=st.caps.companion)
                    prober.start()
                run = perf.run(events)
                _wait(ctx, max(0.0, t0 + block.duration - _now()) + TAIL_S
                      if not run.hung else 0.0)
                hung = run.hung
            except (subprocess.SubprocessError, OSError) as e:
                hung = f"{type(e).__name__}: {e}"
            except HilError as e:
                if e.code == CANCELLED or e.code not in SETUP_CODES:
                    raise
                doc["not_measured"] = f"{e.code}: {e.message}" + (f" ({e.hint})" if e.hint
                                                                   else "")
                ctx.log(f"{name}: not measured -- {doc['not_measured']}")
            finally:
                if prober is not None:
                    prober.stop()
                    prober.join(timeout=2.0)
                if link_u is not None:
                    doc["echo"] = _echo_summary(link_u)
                    link_u.stop()
                if io is not None:
                    try:
                        io.close()
                    except Exception:  # noqa: BLE001
                        pass
                if session is not None:
                    doc["t_stream_stop"] = _now()
                    try:
                        result = session.stop()
                        doc["capture_result"] = (result.to_dict() if hasattr(result, "to_dict")
                                                 else result)
                    except HilError as e:
                        doc["capture_error"] = f"{e.code}: {e.message}"
                    except (subprocess.SubprocessError, OSError) as e:
                        hung = hung or f"capture stop: {type(e).__name__}: {e}"
                    rec = getattr(getattr(session, "_backend", None), "recording", None)
                    if rec is not None:
                        doc["capture_points"] = [list(x) for x in rec.capture_points()]
                        doc["duplex"] = {"health": rec.health(), "events": list(rec.events),
                                         "info": getattr(rec, "info", {})}
                        doc["sr"] = int(getattr(rec, "sample_rate", 0) or 0)
                if mon is not None:
                    try:
                        mon.terminate()
                    except OSError:
                        pass
        if hung:
            tail = []
            if st.con is not None:
                tail = [ln for _s, ln in st.con.read(
                    since_seq=max(con_seq0, st.con.seq - HUNG_TAIL_LINES + 1),
                    limit=HUNG_TAIL_LINES).lines]
            doc["hung"] = {"t_s": None if run.t_hung is None else
                           round(run.t_hung - doc.get("t_stream_start", run.t_hung), 2),
                           "reason": hung, "console_tail": tail}
            ctx.log(f"{name}: the board hung -- {hung}")
        doc["route_home"] = route
        self._after_block(ctx, p, st, doc, block, run)
        (bdir / "block.json").write_text(json.dumps(doc, default=str), encoding="utf-8")
        st.blocks_done.append(name)

    def _after_block(self, ctx: Context, p: Params, st: JamState, doc: dict[str, Any],
                     block: J.Block, run: BlockRun) -> None:
        try:
            self._recover(ctx, p, st, bool(doc.get("hung")), doc)
        except HilError as e:
            doc["not_measured"] = f"could not get back to CDC: {e.code}: {e.message}"
            raise
        link = ctx.open_cdc()
        try:
            rb = self._readback(ctx, link, anchor=False)
            doc["readback_after"] = rb
            rows = [r for r in (rb.get("swap_log") or [])
                    if int(r.get("seq", -1)) > int(doc.get("swap_seq0", -1))]
            doc["swap_rows"] = rows
            if run.swaps:
                st.kit_now = run.swaps[-1]["name"]
                if any(s["name"] == S.JAM_FX for s in run.swaps):
                    st.fx_loaded = True
            for bo in run.bounces:
                a = rb.get("asong") or {}
                clips0 = (doc.get("asong_before") or {}).get("clips") or []
                self._bounce_entry(st, f"bounce_{block.name}", bo, bo["scene"], bo["track"],
                                   "sysex", list(clips0), a, bo["t"], block=block.name)
        finally:
            link.close()
        self._pulls(ctx, st)

    def _bounce_entry(self, st: JamState, name: str, bo: dict[str, Any] | None, scene: int,
                      track: int, via: str, clips0: list[int], status: dict[str, Any],
                      t_req: float, block: str | None = None) -> None:
        clips = status.get("clips") or []
        entry = {"name": name, "block": block, "aid": (bo or {}).get("aid"), "via": via,
                 "scene": scene, "track": track, "state": status.get("bounce"),
                 "file": status.get("file"), "err": status.get("err"),
                 "project": status.get("project"), "t_req": t_req,
                 "clips_before": clips0[track] if track < len(clips0) else None,
                 "clips_after": clips[track] if track < len(clips) else None}
        if status.get("bounce") == 2 and status.get("file") not in (None, "-"):
            remote = f"/sdcard/SONGS/{status.get('project') or PROJECT}/clips/{status.get('file')}"
            entry["remote"] = remote
            st.pending_pulls.append({"remote": remote, "bounce": name,
                                     "local": str(st.workdir / "bounces" / f"{name}.wav")})
        st.bounces.append(entry)

    def _watchdog(self, st: JamState, rec: Any, link: UacLink) -> Any:
        seq0 = getattr(st.con, "seq", None) if st.con is not None else None
        box = {"frames": -1, "t_frames": _now(), "con": None, "t_con": _now()}

        def alive() -> bool:
            """The console spoke in the last 15 s and shows no hang signature this block.
            (A hung render still logs -- the watchdog's lines, "no free block" -- so
            a talking console alone is not enough.)"""
            if st.con is None or _now() - box["t_con"] > 15.0:
                return False
            try:
                signs = st.con.read(since_seq=seq0, match=HANG_SIGNS, limit=3).lines
            except Exception:  # noqa: BLE001 -- no console, no proof the board is alive
                return False
            return not signs

        def check() -> str | None:
            now = _now()
            if rec is not None:
                if getattr(rec, "error", None) is not None:
                    return f"the duplex stream failed: {rec.error}"
                f = int(getattr(rec, "frames_in", 0))
                if f != box["frames"]:
                    box["frames"], box["t_frames"] = f, now
                elif now - box["t_frames"] > 2.0:
                    return "capture stopped: no frames for 2 s"
            if st.con is not None:
                s = getattr(st.con, "seq", None)
                if s != box["con"]:
                    box["con"], box["t_con"] = s, now
                elif now - box["t_con"] > 15.0:
                    return "the console is silent for 15 s"
            lost_echoes = link.echo_pending_over(2.0)
            if lost_echoes >= 3 or link.companion_misses >= 3:
                if alive():
                    # 7.10 run 3: the board went on playing and taking SysEx while its
                    # replies to the host were dropped (usb_drop); resetting it then cost
                    # the rest of the block. Counted, not treated as a hang.
                    st.log.emit("midi_in_lost", echoes=lost_echoes,
                                companion=link.companion_misses)
                    for k in [k for k, t in link.echo_sent.items() if now - t > 2.0]:
                        link.echo_sent.pop(k, None)
                    link.companion_misses = 0
                    return None
                return ("3 echoes unanswered" if lost_echoes >= 3
                        else "3 companion requests unanswered")
            return None

        return check

    def _desk_doc(self, st: JamState) -> dict[str, Any]:
        desk = json.loads(json.dumps(st.desk.get(st.desk_now) or {}))
        if desk.get("outputs") and len(desk["outputs"]) > CAPTURE_BUS:
            desk["outputs"][CAPTURE_BUS]["volume"] = st.out_vol
        return desk

    @staticmethod
    def _pilot(p: Params) -> S.PilotSpec:
        return S.PilotSpec(hz=p.pilot_hz, dbfs=p.pilot_dbfs, timecode=p.timecode)

    # ---------------------------------------------------------------- recovery
    def _default(self, ctx: Context, p: Params) -> None:
        if ctx.device.usb_mode is not UsbMode.DEFAULT or ctx.device.ports.cdc is None:
            ctx.device = set_mode(ctx.device, UsbMode.DEFAULT, wait=True,
                                  timeout_s=p.enum_timeout)

    def _recover(self, ctx: Context, p: Params, st: JamState, hung: bool,
                 doc: dict[str, Any]) -> None:
        """Back to the default profile; a board that hung is reset over the STM bridge."""
        last: HilError | None = None
        for _ in range(0 if hung else 2):
            try:
                self._refresh(ctx)
                self._default(ctx, p)
                return
            except HilError as e:
                if e.code == CANCELLED:
                    raise
                last = e
        if st.con is None:
            raise last or HilError(TIMEOUT, "no way back to the default profile")
        ctx.log("hard reset over the STM bridge, then the default profile")
        st.con.reset()
        st.con.wait_boot(timeout_s=60.0, settle_s=2.0)
        doc["recovered_by_reset"] = True
        self._refresh(ctx)
        self._default(ctx, p)

    @staticmethod
    def _refresh(ctx: Context) -> None:
        try:
            from crosspad_hil.devices import discover, select

            ctx.device = select(discover(), ctx.device.id)
        except HilError as e:
            ctx.log(f"rediscovery: {e.code}: {e.message}")

    @staticmethod
    def _poll_asong(ctx: Context, link: Any, done: Any, timeout_s: float) -> dict[str, Any]:
        deadline = _now() + timeout_s
        while True:
            stt = verbs.asong(link, timeout_s=SEQ_TIMEOUT_S)
            if done(stt) or _now() >= deadline:
                return stt
            _wait(ctx, 0.25)

    @staticmethod
    def _kit_id(link: Any, name: str) -> int | None:
        try:
            for k in verbs.kit_list(link).get("kits", []):
                if str(k["name"]).lower() == str(name).lower():
                    return int(k["id"])
        except HilError:
            return None
        return None

    # ---------------------------------------------------------------- put back
    def _put_back(self, ctx: Context, p: Params, st: JamState, data: dict[str, Any]) -> None:
        home = st.home
        left = data["left"]
        if home is None:
            return
        try:
            self._default(ctx, p)
            link = ctx.open_cdc()
        except HilError as e:
            left["error"] = f"{e.code}: {e.message} (left in the pins journal)"
            return
        try:
            left.update(RS.put_back_ram(link, home, jam_scenes=list(st.scenes.values()),
                                        kit_id_by_name=lambda n: self._kit_id(link, n),
                                        log=ctx.log))
        finally:
            link.close()
        files = {k: v for k, v in home.files.items()
                 if p.restore_sequences or k != home.seq_file}
        fh = RS.HomeState.from_dict({**home.to_dict(), "files": files})
        left.update(RS.put_back_files(
            fh, lambda loc, r: push_file(ctx, loc, r),
            lambda r: self._delete(ctx, r), log=ctx.log))
        if p.restore_sequences and home.seq_file in files and st.con is not None:
            try:
                st.con.reset()
                st.con.wait_boot(timeout_s=60.0, settle_s=2.0)
                self._refresh(ctx)
                left["reboot"] = "restored"
            except HilError as e:
                left["reboot"] = f"{e.code}: {e.message}"
        try:
            link = ctx.open_cdc()
            try:
                # With a home kit back the store reads the card again; with none it
                # does not, and the sequences are checked as the file on the card.
                if home.kit.get("name"):
                    RS.wait_seq_store(link)
                now = RS.snapshot(link, need_file=bool(home.kit.get("name")))
                if not now.seq and home.seq_file in home.files and home.files[home.seq_file]:
                    back = ctx.workdir / "home" / "sequences_after.json"
                    pull_file(ctx, home.seq_file, back)
                    same = back.read_bytes() == Path(home.files[home.seq_file]).read_bytes()
                    left["sequences_file"] = "restored" if same else "differs from the copy"
                diff = RS.board_diff(home, now)
                data["board_diff"] = diff
                if diff:
                    left["board"] = f"differs: {diff}"
                try:
                    verbs.clock_status(link, reset=True)
                except HilError:
                    pass
            finally:
                link.close()
        except HilError as e:
            left["board"] = f"could not be read back: {e.code}: {e.message}"
        bad = [k for k, v in left.items() if v not in ("restored", "unchanged")
               and not str(v).startswith("removed") and k != "board"]
        if not bad and not data.get("board_diff"):
            RS.unjournal(ctx.device.id)
        else:
            ctx.log(f"LEFT CHANGED: {bad} (the pins journal keeps the home state)")

    def _delete(self, ctx: Context, remote: str) -> None:
        link = ctx.open_cdc()
        try:
            verbs.fs_delete(link, remote)
        finally:
            link.close()

    # ---------------------------------------------------------------- files and verdict
    def _write_script(self, ctx: Context, st: JamState) -> None:
        (ctx.workdir / "script.json").write_text(json.dumps(st.script.to_dict(), default=str),
                                                 encoding="utf-8")
        (ctx.workdir / "script.txt").write_text(J.script_table(st.script) + "\n",
                                                encoding="utf-8")

    def _write_side_files(self, ctx: Context, st: JamState, data: dict[str, Any]) -> None:
        wd = ctx.workdir
        (wd / "bounces.json").write_text(json.dumps(st.bounces, default=str), encoding="utf-8")
        (wd / "caps.json").write_text(json.dumps(st.caps.to_dict()), encoding="utf-8")
        (wd / "recs.json").write_text(json.dumps(st.rec_checks), encoding="utf-8")
        (wd / "restore.json").write_text(json.dumps({"left": data.get("left"),
                                                     "board_diff": data.get("board_diff")},
                                                    default=str), encoding="utf-8")
        env = {"fw": data.get("firmware"), "sd": data.get("sd"), "caps": st.caps.to_dict(),
               "kits": st.kits, "scenes": st.scenes, "recorder": st.recorder,
               "env": data.get("env")}
        (wd / "env.json").write_text(json.dumps(env, default=str), encoding="utf-8")

    def _analyse(self, ctx: Context, p: Params, run_dir: Path, artifacts: list[Artifact], *,
                 data: dict[str, Any], env: str | None = None) -> Report:
        from crosspad_hil.jam import analysis as AN
        from crosspad_hil.jam import calibration as CAL
        from crosspad_hil.jam.detect import Tolerances

        tol = Tolerances(late_ms=p.late_ms, seq_late_ms=p.seq_late_ms, ghost_db=p.ghost_db,
                         strict=p.strict)
        fw = _fw_sha(data.get("firmware") or {})
        cal = CAL.load_for(fw) if fw else Calibration()
        (run_dir / "calibration.json").write_text(json.dumps(cal.to_dict()), encoding="utf-8")
        ja = AN.analyze_run(run_dir, tolerances=tol, calibration=cal, snippets=p.snippets,
                            allow_partial=p.allow_partial)
        if p.calibrate:
            new = CAL.calibrate_from(ja, cal)
            new.fw_sha = fw or "unknown"
            data["calibration_written"] = str(CAL.save_for(new))
        doc = ja.doc
        status, code = doc["verdict"]["status"], doc["verdict"]["exit"]
        if env is not None and code == 0:
            status, code = "not measured", 2
        head = f"live_jam: {status}"
        if env:
            head += f" -- {env}"
        lines = [head] + [f"  {ln}" for ln in doc["verdict"]["lines"][:40]]
        s = doc["summary"]
        lines.append(f"  {s['glitches']} glitch(es), {s['counters']} counter(s), "
                     f"{s['unverifiable']} unverifiable; by cause {s['by_cause']}")
        lines.append(f"  report: {run_dir / AN.REPORT}  timeline: {run_dir / 'timeline.html'}")
        for name in ("timeline.html", "timeline.md", AN.REPORT):
            artifacts.append(Artifact(path=str(run_dir / name), mime="text/plain",
                                      role=f"jam:{name}"))
        data["jam"] = {"status": status, "summary": s, "report": str(run_dir / AN.REPORT)}
        return Report(passed=code == 0, summary="\n".join(lines), data=data,
                      artifacts=artifacts, exit_code=code)


def _echo_summary(link: UacLink) -> dict[str, Any]:
    rtts = sorted(r for _t, r in link.echo_rtt_ms)
    if not rtts:
        return {"n": 0, "misses": link.echo_misses}
    return {"n": len(rtts), "p50": round(rtts[len(rtts) // 2], 2), "max": round(rtts[-1], 2),
            "misses": link.echo_misses, "pending": len(link.echo_sent)}


def _fw_sha(fw: dict[str, Any]) -> str:
    sha = fw.get("app_sha")
    return str(sha.get("sha256", "")) if isinstance(sha, dict) else ""


def _kits_root() -> str:
    from crosspad_hil.kits import KITS_ROOT

    return KITS_ROOT


def wav_layout(head: bytes) -> tuple[int, int, int, int]:
    """(channels, rate, bytes per frame, offset of the data) from a WAV file's head."""
    if head[:4] != b"RIFF" or head[8:12] != b"WAVE":
        raise HilError(ENV, "not a WAV file")
    i = 12
    ch = sr = align = 0
    while i + 8 <= len(head):
        cid, size = head[i:i + 4], struct.unpack("<I", head[i + 4:i + 8])[0]
        if cid == b"fmt ":
            _fmt, ch, sr, _br, align = struct.unpack("<HHIIH", head[i + 8:i + 22])
        elif cid == b"data":
            return ch, sr, align, i + 8
        i += 8 + size + (size & 1)
    raise HilError(ENV, "no data chunk in the first bytes of the WAV")


def _pull_wav_window(ctx: Context, remote: str, local: Path, start: int, frames: int) -> None:
    """``frames`` of the card's WAV from frame ``start``, as a WAV of its own: the header
    and the window are fetched (FS_GET offset/length), not the whole take."""
    head = local.with_suffix(".head")
    pull_file(ctx, remote, head, offset=0, length=4096)
    ch, sr, align, data = wav_layout(head.read_bytes())
    raw = local.with_suffix(".raw")
    pull_file(ctx, remote, raw, offset=data + start * align, length=frames * align)
    with wave.open(str(local), "wb") as w:
        w.setnchannels(ch)
        w.setsampwidth(align // max(1, ch))
        w.setframerate(sr)
        w.writeframes(raw.read_bytes())
    head.unlink(missing_ok=True)
    raw.unlink(missing_ok=True)


SCENARIO = register(LiveJamScenario())
