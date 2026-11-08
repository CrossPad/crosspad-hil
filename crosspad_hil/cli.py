"""crosspad-hil command line: `main(argv) -> int`.

Every subcommand builds the same dict the daemon returns for the equivalent op;
`--json` prints that dict, otherwise a one-screen human summary. Exit codes:
0 ok, 1 fail (scenario failed / device replied ERR / expect missed), 2 env or
usage (any HilError, argparse error).
"""
from __future__ import annotations

import argparse
import dataclasses
import faulthandler
import importlib
import json
import os
import platform
import shutil
import signal
import sys
import threading
import time
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, TextIO

from . import __version__, hub_link, verbs
from . import ble as ble_host
from . import capture as cap_mod
from . import diagnose as diag
from . import fpc as fpc_mod
from . import kits as kits_mod
from . import stim as stim_mod
from .analyze import KINDS as ANALYZE_KINDS
from .analyze import analyze as analyze_wav
from .bench import HOLDER_ENV, Bench, render_markdown, write_markdown
from .capture import CaptureSession
from .cdc import CdcLink, Reply
from .console import Console
from .devices import Backends, Device, UsbMode, discover, select
from .errors import BAD_ARGS, NO_DEVICE, HilError
from .fs import push as fs_push
from .knowledge import load as load_knowledge
from .locks import PortLock
from .midi import MidiIO, MidiRole, companion_command, echo_rtt, query_route, sysex_audio_route
from .ota import flash as ota_flash_and_boot
from .ota import request_bootloader
from .record import RecordingSerial
from .scenarios import base as scen
from .scenarios import get as get_scenario

# Import every scenario module so the catalog is complete before the parser
# (or the daemon's scenario.list) is built. Naming them one by one here is
# how midi_bench and midi_stress shipped written, registered by nobody and
# reachable from nothing.
from .scenarios import load_builtin as _load_builtin_scenarios
from .scenarios import names as scenario_names
from .serial_open import open_serial
from .snapshot import ref_to_delta, take_snapshot
from .usbmode import set_mode

_load_builtin_scenarios()

# Multi-line verbs: cdc.yaml says reply "multi" but not the prefix (hil_control.cpp).
_DOCTOR_REQUIRED = ("python", "import:serial", "import:rtmidi", "import:yaml")


@dataclass
class CliDeps:
    """Everything the CLI touches that a test wants to replace."""

    backends: Backends | None = None
    serial_factory: Callable[..., Any] = open_serial
    midi_factory: Callable[..., Any] = MidiIO
    run_scenario_fn: Callable[..., scen.Report] = scen.run_scenario
    flash_fn: Callable[..., dict] = ota_flash_and_boot
    bootloader_fn: Callable[..., dict] = request_bootloader
    set_mode_fn: Callable[..., Device] = set_mode
    capture_factory: Callable[..., Any] = CaptureSession
    analyze_fn: Callable[..., dict] = analyze_wav
    push_fn: Callable[..., dict] = fs_push
    bench: Bench | None = None
    sleep: Callable[[float], None] = time.sleep
    stdin: TextIO | None = None
    stdout: TextIO | None = None
    stderr: TextIO | None = None

    def inp(self) -> TextIO:
        return self.stdin or sys.stdin

    def out(self) -> TextIO:
        return self.stdout or sys.stdout

    def err(self) -> TextIO:
        return self.stderr or sys.stderr

    def lease(self) -> Bench:
        if self.bench is None:
            self.bench = Bench()
        return self.bench


@dataclass
class CmdResult:
    data: dict
    text: str
    exit_code: int = 0


Handler = Callable[[CliDeps, argparse.Namespace], CmdResult]


# --------------------------------------------------------------------------- helpers

def _holder(ns: argparse.Namespace) -> str | None:
    return getattr(ns, "holder", None) or os.environ.get(HOLDER_ENV) or None


def _resolve(deps: CliDeps, ns: argparse.Namespace) -> Device:
    """The board a command is about to touch, after checking the bench lease."""
    dev = select(discover(deps.backends), getattr(ns, "device", None))
    deps.lease().check(dev.id, _holder(ns), f"cli:{ns.command}",
                       force=bool(getattr(ns, "bench_force", False)))
    return dev


@contextmanager
def _keep_lease(deps: CliDeps, ns: argparse.Namespace, device_id: str,
                every_s: float = 60.0) -> Iterator[None]:
    """Renew the caller's lease while a long command (a scenario) runs."""
    holder = _holder(ns)
    if not holder:
        yield
        return
    stop = threading.Event()

    def loop() -> None:
        while not stop.wait(every_s):
            deps.lease().touch(device_id, holder)

    t = threading.Thread(target=loop, name="bench-keepalive", daemon=True)
    t.start()
    try:
        yield
    finally:
        stop.set()


@contextmanager
def _cdc(deps: CliDeps, dev: Device) -> Iterator[CdcLink | hub_link.HubLink]:
    """A link for verbs: through the hub, or direct (`--no-hub`, or inside an exclusive op)."""
    link = hub_link.open_link(dev, purpose="cli", serial_factory=deps.serial_factory)
    try:
        yield link
    finally:
        link.close()


@contextmanager
def _console(deps: CliDeps, dev: Device, *, reset: bool = False,
             log_path: Path | None = None) -> Iterator[Console]:
    if dev.ports.console is None:
        raise HilError(NO_DEVICE, f"{dev.id} has no STM32 bridge VCP (0x0483:0x5740)",
                       hint="the console comes off the STM bridge, not the ESP CDC", device=dev.id)
    con = Console(dev.ports.console.path, log_path=log_path, serial_factory=deps.serial_factory)
    con.open(reset=reset)
    try:
        yield con
    finally:
        con.close()


def _reply_dict(reply: Reply) -> dict:
    return {"line": reply.line, "parsed": reply.parsed, "rtt_ms": reply.rtt_ms,
            "more_lines": reply.more_lines, "extra_lines": reply.extra_lines}


def _reply_exit(reply: Reply) -> int:
    return 1 if (reply.parsed or {}).get("kind") == "err" else 0


def _kv(d: dict, keys: tuple[str, ...] | None = None) -> str:
    items = d.items() if keys is None else ((k, d.get(k)) for k in keys)
    return "  ".join(f"{k}={v}" for k, v in items)


def _int_arg(value: Any, name: str, lo: int, hi: int) -> int:
    try:
        v = int(value)
    except (TypeError, ValueError) as e:
        raise HilError(BAD_ARGS, f"{name} must be an integer, got {value!r}") from e
    if not lo <= v <= hi:
        raise HilError(BAD_ARGS, f"{name} must be {lo}..{hi}, got {v}")
    return v


# --------------------------------------------------------------------------- renderers

def render_devices(devices: list[dict]) -> str:
    if not devices:
        return "no CrossPad found (ESP CDC 0x303A:0x3456 / STM bridge 0x0483:0x5740)"
    rows = ["ID        MODE        CDC              CONSOLE          ESP MIDI / STM MIDI / UAC2"]
    for d in devices:
        p = d["ports"]

        def path(k: str, p: dict = p) -> str:
            v = p.get(k)
            return (v.get("path") or v.get("name") or "-") if v else "-"

        rows.append(f"{d['id']:<9} {d['usb_mode']:<11} {path('cdc'):<16} {path('console'):<16} "
                    f"{path('esp_midi')} / {path('stm_midi')} / {path('uac2')}")
    return "\n".join(rows)


def render_led_grid(leds: dict) -> str:
    """Port of hil_led_state.render: 4x4 as it sits under your hands, pads 12..15 on top."""
    colors = list(leds.get("colors") or [])
    lines = [
        f"  brightness {leds.get('brightness')}   settings {leds.get('cfgbri')}"
        f"   animating {leds.get('anim')}   coalesced {leds.get('coalesce')}",
        f"  last power state 0x{int(leds.get('pwr') or 0):02X}"
        f"   notifications {leds.get('pwr_count')}"
        f"   (0x00 AWAKE · 0x03 WOKE · 0x04 LIGHT)",
    ]
    txfail = leds.get("txfail") or 0
    if txfail:
        lines.append(f"  {txfail} frame(s) never reached the strip — the model is right "
                     f"and the wire is not")
    lines.append("")
    for row in range(3, -1, -1):
        cells = []
        for col in range(4):
            idx = row * 4 + col
            c = colors[idx] if idx < len(colors) else "??????"
            cells.append(f"{idx:2d}:{c}")
        lines.append("   " + "  ".join(cells))
    if colors and all(c == "000000" for c in colors):
        lines.append("")
        lines.append("  every pad is black in the model — nothing repainted them")
    return "\n".join(lines)


def render_hub_control(control: dict | None) -> str:
    """The hub's control view on one line: clients, the exclusive op, the CDC."""
    if control is None:
        return "control=down"
    ex = control.get("lease")
    held = f"{ex['client']} ({ex['purpose']}, pid {ex['pid']})" if ex else "-"
    return (f"clients={len(control.get('clients') or [])} exclusive={held} "
            f"cdc={'open' if control.get('cdc_open') else 'closed'}")


def render_ui(ui: dict) -> str:
    focus = ui.get("focus") or {}
    focus_ref = focus.get("ref")
    lines = [f"UI  drawer={ui.get('drawer')}  theme={ui.get('theme')}  app={ui.get('app') or '-'}"]
    for item in ui.get("group") or []:
        mark = ">" if item.get("ref") == focus_ref else " "
        lines.append(f"{mark} {item.get('ref'):<3} {item.get('label')}")
    if not ui.get("group"):
        lines.append("  (encoder group empty)")
    return "\n".join(lines)


def render_snapshot(snap: dict) -> str:
    out = [f"{snap.get('snapshot_id')}  {snap.get('device')}  usb_mode={snap.get('usb_mode')}"]
    apps = snap.get("apps")
    if apps:
        out.append(f"apps: running={apps.get('running') or '-'}  "
                   f"[{', '.join(apps.get('apps') or [])}]")
    kit = snap.get("kit")
    if kit:
        out.append(f"kit: {kit.get('current')} {kit.get('name') or ''}  "
                   f"loading={kit.get('loading')} pending={kit.get('pending')}")
    ui = snap.get("ui")
    if ui:
        out.append(render_ui(ui))
    leds = snap.get("leds")
    if leds:
        out.append("leds:")
        out.append(render_led_grid(leds))
    pads = snap.get("pads")
    if pads:
        out.append("pads: " + _kv(pads))
    mem = snap.get("mem")
    if mem:
        out.append("mem: " + _kv(mem))
    ble = snap.get("ble")
    if ble:
        out.append("ble: " + _kv(ble))
    con = snap.get("console")
    if con:
        out.append(f"console: fatals={len(con.get('fatals') or [])} reboots={con.get('reboots')} "
                   f"cdc_drops={con.get('cdc_drops')} since_seq={con.get('since_seq')}")
    if snap.get("changed"):
        out.append("changed: " + ", ".join(snap["changed"]))
    return "\n".join(out)


def render_report(report: scen.Report) -> str:
    head = "PASS" if report.passed else "FAIL"
    lines = [f"{head}: {report.summary}"]
    for k, v in report.data.items():
        if isinstance(v, (int, float, str, bool)) or v is None:
            lines.append(f"  {k}: {v}")
        else:
            lines.append(f"  {k}: {json.dumps(v)[:200]}")
    for a in report.artifacts:
        lines.append(f"  artifact[{a.role}]: {a.path}")
    return "\n".join(lines)


# --------------------------------------------------------------------------- doctor

def doctor_checks(backends: Backends | None = None) -> list[dict]:
    checks: list[dict] = []

    def add(name: str, ok: bool, detail: str, fix: str | None = None) -> None:
        checks.append({"name": name, "ok": bool(ok), "detail": detail, "fix": fix})

    add("python", sys.version_info >= (3, 10), platform.python_version(),
        None if sys.version_info >= (3, 10) else "install Python >= 3.10")
    for mod, pkg, required in (("serial", "pyserial", True),
                               ("rtmidi", "python-rtmidi", True),
                               ("yaml", "PyYAML", True),
                               ("sounddevice", "crosspad-hil[audio]", False),
                               ("bleak", "crosspad-hil[ble]", False)):
        try:
            importlib.import_module(mod)
            add(f"import:{mod}", True, "present")
        except Exception as e:  # noqa: BLE001 — a broken optional dep is a finding, not a crash
            add(f"import:{mod}", not required, f"missing ({type(e).__name__}: {e})",
                f"pip install '{pkg}'")
    holders = PortLock.holders()
    live = [h for h in holders if h.get("alive")]
    add("locks", True, "no live port locks" if not live else
        "; ".join(f"{h['port']} held by pid {h['pid']} ({h['purpose']})" for h in live),
        None if not live else "close the other crosspad-hil / script, or wait")
    try:
        devs = discover(backends)
        add("devices", len(devs) > 0,
            ", ".join(f"{d.id}:{d.usb_mode.value}" for d in devs) or "none found",
            None if devs else "connect the board; if only the STM VCP shows, the ESP may be in "
                              "bootloader or UAC2 mode")
    except Exception as e:  # noqa: BLE001
        add("devices", False, f"discovery failed: {e}", "run with --json for details")
    if sys.platform.startswith("linux"):
        try:
            import grp

            groups = {grp.getgrgid(g).gr_name for g in os.getgroups()}
            ok = bool(groups & {"dialout", "uucp", "plugdev"})
            add("serial-group", ok, ", ".join(sorted(groups & {"dialout", "uucp", "plugdev"})) or
                "not in dialout/uucp", None if ok else "sudo usermod -aG dialout $USER; re-login")
        except Exception as e:  # noqa: BLE001
            add("serial-group", True, f"could not read groups ({e})")
    a2l = shutil.which("xtensa-esp32s3-elf-addr2line")
    add("addr2line", True, a2l or "not on PATH (only diagnose needs it)",
        None if a2l else ". $IDF_PATH/export.sh")
    return checks


# --------------------------------------------------------------------------- commands

def cmd_devices(deps: CliDeps, ns: argparse.Namespace) -> CmdResult:
    def once() -> tuple[dict, str]:
        devs = [d.to_dict() for d in discover(deps.backends)]
        return {"devices": devs}, render_devices(devs)

    data, text = once()
    if not ns.watch:
        return CmdResult(data, text)
    out = deps.out()
    try:
        while True:
            out.write((json.dumps(data) if ns.as_json else text) + "\n")
            out.flush()
            deps.sleep(1.0)
            data, text = once()
    except KeyboardInterrupt:
        pass
    return CmdResult(data, text)


def cmd_doctor(deps: CliDeps, ns: argparse.Namespace) -> CmdResult:
    checks = doctor_checks(deps.backends)
    bad = [c for c in checks if c["name"] in _DOCTOR_REQUIRED and not c["ok"]]
    lines = []
    for c in checks:
        lines.append(f"[{'ok' if c['ok'] else '!!'}] {c['name']:<14} {c['detail']}")
        if not c["ok"] and c["fix"]:
            lines.append(f"     fix: {c['fix']}")
    return CmdResult({"checks": checks}, "\n".join(lines), 2 if bad else 0)


def cmd_console(deps: CliDeps, ns: argparse.Namespace) -> CmdResult:
    dev = _resolve(deps, ns)
    log_path = Path(ns.log) if ns.log else None
    with _console(deps, dev, reset=False, log_path=log_path) as con:
        if ns.reset:
            with hub_link.exclusive("console reset"):
                con.reset()
        if ns.expect:
            res = con.expect(list(ns.expect), list(ns.reject or []), timeout_s=ns.timeout)
            data = dataclasses.asdict(res)
            data["port"] = dev.ports.console.path
            if res.hit:
                text = f"HIT {res.hit!r} at seq {res.seq} after {res.elapsed_s:.1f}s\n" + \
                       "\n".join(res.context)
                return CmdResult(data, text, 0)
            if res.rejected:
                text = f"REJECTED {res.rejected!r} at seq {res.seq}\n" + "\n".join(res.context)
            else:
                text = f"no match within {ns.timeout:.0f}s"
            return CmdResult(data, text, 1)
        out = deps.out()
        deadline = time.monotonic() + ns.timeout if ns.timeout > 0 else None
        seq: int | None = None
        count = 0
        try:
            while deadline is None or time.monotonic() < deadline:
                r = con.read(since_seq=seq, wait_ms=500)
                seq = r.next_seq
                for _, line in r.lines:
                    count += 1
                    if not ns.as_json:
                        out.write(line + "\n")
                out.flush()
        except KeyboardInterrupt:
            pass
        snap = con.snapshot()
        return CmdResult({"lines": count, "snapshot": snap}, f"{count} lines", 0)


def cmd_cdc(deps: CliDeps, ns: argparse.Namespace) -> CmdResult:
    cmd = " ".join([ns.verb, *ns.args])
    verb = ns.verb.upper()
    dev = _resolve(deps, ns)
    entry = (load_knowledge("cdc").get("verbs") or {}).get(verb)
    with _cdc(deps, dev) as link:
        if entry and entry.get("reply") == "multi":
            # cdc.yaml carries the prefix for every multi-line verb; a second
            # copy here shadowed it and only listed three, so the other five
            # (KIT_LIST among them) waited for a line starting with the verb
            # name and always timed out. A list means the reply mixes prefixes.
            expect = ns.expect or entry.get("prefix") or verb
            if isinstance(expect, list):
                expect = tuple(expect)
            lines = link.transact_multi(cmd, expect,
                                        end=entry.get("end"),
                                        timeout_s=(ns.timeout or 2000.0) / 1000.0)
            return CmdResult({"lines": lines}, "\n".join(lines), 0)
        # A person reading the reply wants all of it: the lines it continues on
        # (" bt:", " exc:", RESET_LOG's " #n" rows) whatever cdc.yaml says.
        reply = link.transact(cmd, expect=ns.expect, follow=True,
                              timeout_s=None if ns.timeout is None else ns.timeout / 1000.0)
    text = "\n".join([reply.line, *(" " + m for m in reply.more_lines), *reply.extra_lines])
    return CmdResult(_reply_dict(reply), text, _reply_exit(reply))


def cmd_app(deps: CliDeps, ns: argparse.Namespace) -> CmdResult:
    dev = _resolve(deps, ns)
    with _cdc(deps, dev) as link:
        if ns.app_cmd == "list":
            r = verbs.app_list(link)
            return CmdResult(r, f"running={r['running'] or '-'}\n" + "\n".join(r["apps"]))
        if ns.app_cmd == "start":
            r = verbs.app_start(link, ns.name, wait_s=ns.wait)
            return CmdResult(r, f"running={r['running']}")
        if ns.app_cmd == "stop":
            r = verbs.app_stop(link)
            return CmdResult(r, "stopped")
        r = verbs.app_versions(link)
        # A field the firmware did not send is None, and a format spec on None
        # raises rather than printing anything — an app installed as a local
        # directory has no manifest id, which was enough to make the whole
        # command a TypeError instead of a listing.
        def _f(v: Any) -> str:
            return "-" if v is None else str(v)
        lines = [f"{_f(c['component']):<24} {_f(c['id']):<12} {_f(c['commit'])} {_f(c['ref'])}"
                 f"{' dirty' if c['dirty'] else ''}" for c in r["components"]]
        return CmdResult(r, "\n".join(lines) or "(no app components)")


@dataclass
class _PushCtx:
    """The sliver of `scenarios.base.Context` that `kits.push_kit` needs.

    A full `Context` also does the "did opening the console just reset the
    board" wait (1.5 s, real hardware only) before handing it back — useful
    when a scenario races that reset, irrelevant here since nothing has reset
    yet. Keeping this local avoids that wait and any dependency on the
    scenarios package from a CLI verb that (per the design) has no business
    importing one.
    """

    device: Device
    log: Callable[[str], None]
    cancelled: threading.Event
    serial_factory: Callable[..., Any]

    def open_cdc(self) -> CdcLink:
        link = CdcLink(self.device.ports.cdc.path, serial_factory=self.serial_factory)
        link.open()
        return link

    def check_cancelled(self) -> None:
        return None


def _push_built_kit(deps: CliDeps, dev: Device, kit_dir: Path, workdir: Path,
                    load: bool) -> dict[str, Any]:
    """Send an already-built kit directory: the mkdir/push/reboot/scan-wait dance."""
    err = deps.err()
    ctx = _PushCtx(device=dev, log=lambda m: err.write(str(m) + "\n"),
                  cancelled=threading.Event(), serial_factory=deps.serial_factory)
    with hub_link.exclusive("kit push"), \
            _console(deps, dev, reset=False, log_path=workdir / "console.log") as console:
        link = ctx.open_cdc()
        return kits_mod.push_kit(ctx, dev, link, kit_dir, console=console, load=load,
                                 push=deps.push_fn)


def _pushed_text(result: dict[str, Any]) -> str:
    return (f"pushed {result['name']!r} -> {result['dir']} ({result['bytes']} B), "
            f"kit_id={result['kit_id']}" + (" (loaded)" if result["loaded"] else ""))


def _workdir(ns: argparse.Namespace, prefix: str) -> Path:
    workdir = Path(ns.logdir) / f"{prefix}_{time.strftime('%Y%m%d_%H%M%S')}"
    workdir.mkdir(parents=True, exist_ok=True)
    return workdir


def _cmd_kit_push(deps: CliDeps, dev: Device, ns: argparse.Namespace) -> CmdResult:
    """Build a kit from a directory of WAVs and push it onto the card.

    Builds locally first (no port touched yet — a bad WAV fails before
    anything is opened), then reuses the same mkdir/push/reboot/scan-wait
    `kits.push_kit` shares with `scenarios._audio.ensure_kits`.
    """
    src = Path(ns.dir)
    if not src.is_dir():
        raise HilError(BAD_ARGS, f"not a directory: {src}")
    name = ns.name or src.name
    err = deps.err()
    workdir = _workdir(ns, "kit_push")
    kit_dir = kits_mod.build_kit(src, workdir / "build", name, rate=ns.rate)
    n_samples = len(list((kit_dir / "SAMPLES").glob("*.wav")))
    err.write(f"built {kit_dir} ({n_samples} sample(s), {ns.rate} Hz)\n")

    result = _push_built_kit(deps, dev, kit_dir, workdir, ns.load)
    return CmdResult(result, _pushed_text(result), 0)


def _cmd_kit_from_fpc(deps: CliDeps, ns: argparse.Namespace) -> CmdResult:
    """An FL Studio FPC preset -> a CrossPad kit directory, optionally pushed.

    Nothing here touches a port unless `--push` is given: the conversion is
    pure host work and a preset whose samples cannot be found should fail
    before a device is even looked for.
    """
    preset_path = Path(ns.preset)
    if not preset_path.is_file():
        raise HilError(BAD_ARGS, f"not a file: {preset_path}")
    preset = fpc_mod.parse_fpc_preset(preset_path)

    drive_c = Path(ns.drive_c) if ns.drive_c else fpc_mod.find_drive_c(preset_path.resolve())
    if drive_c is None:
        raise HilError(BAD_ARGS, f"{preset_path} is not inside a Wine prefix's drive_c",
                       hint="pass --drive-c with the prefix's drive_c directory")
    factory = Path(ns.fl_data) if ns.fl_data else fpc_mod.default_factory_data(drive_c)

    err = deps.err()
    kit_dir = fpc_mod.kit_from_fpc(preset, Path(ns.out_dir), drive_c=drive_c,
                                   factory_data=factory, name=ns.name,
                                   bank=("A", "B").index(ns.bank.upper()), rate=ns.rate,
                                   log=lambda m: err.write(str(m) + "\n"))
    n_samples = len(list((kit_dir / "SAMPLES").glob("*.wav")))
    result: dict[str, Any] = {"preset": preset.name, "dir": str(kit_dir),
                              "samples": n_samples, "pushed": None}
    text = f"{preset.name!r} -> {kit_dir} ({n_samples}/{kits_mod.MAX_PADS} pad(s))"
    if ns.push:
        dev = _resolve(deps, ns)
        pushed = _push_built_kit(deps, dev, kit_dir, _workdir(ns, "kit_from_fpc"), ns.load)
        result["pushed"] = pushed
        text += "\n" + _pushed_text(pushed)
    return CmdResult(result, text, 0)


def _cmd_kit_to_fpc(deps: CliDeps, ns: argparse.Namespace) -> CmdResult:
    """A CrossPad kit directory -> an FPC preset pointing at its WAVs."""
    kit_dir = Path(ns.dir)
    if not (kit_dir / "kit.json").is_file():
        raise HilError(BAD_ARGS, f"{kit_dir} has no kit.json",
                       hint="point this at a kit directory, not a directory of WAVs")
    out = fpc_mod.fpc_from_kit(kit_dir, Path(ns.out), win_prefix=ns.win_prefix, name=ns.name)
    prefix = ns.win_prefix or fpc_mod.default_win_prefix(kit_dir)
    result = {"fst": str(out), "bytes": out.stat().st_size, "win_prefix": prefix}
    return CmdResult(result, f"{kit_dir} -> {out} ({result['bytes']} B), pads point at "
                             f"{prefix}\\SAMPLES", 0)


def cmd_kit(deps: CliDeps, ns: argparse.Namespace) -> CmdResult:
    if ns.kit_cmd == "from-fpc":       # host-only conversions: no device unless --push
        return _cmd_kit_from_fpc(deps, ns)
    if ns.kit_cmd == "to-fpc":
        return _cmd_kit_to_fpc(deps, ns)
    dev = _resolve(deps, ns)
    if ns.kit_cmd == "push":
        return _cmd_kit_push(deps, dev, ns)
    with _cdc(deps, dev) as link:
        if ns.kit_cmd == "list":
            r = verbs.kit_list(link)
            lines = [f"{'*' if k['id'] == r['current'] else ' '} {k['id']:>3}  {k['name']}"
                     for k in r["kits"]]
            return CmdResult(r, "\n".join(lines))
        if ns.kit_cmd == "status":
            r = verbs.kit_status(link)
            return CmdResult(r, _kv(r))
        kit_id = _int_arg(ns.id, "ID", 0, 255)
        if ns.wait:
            r = verbs.kit_load(link, kit_id, wait_s=ns.timeout)
        else:
            link.transact(f"KIT_LOAD {kit_id}")
            r = verbs.kit_status(link)
        return CmdResult(r, _kv(r))


def cmd_padmix(deps: CliDeps, ns: argparse.Namespace) -> CmdResult:
    """The Pad Mixer app on the host: the current kit's faders, live, into kit.json."""
    from . import padmix

    dev = _resolve(deps, ns)
    with _keep_lease(deps, ns, dev.id), _cdc(deps, dev) as link:
        r = padmix.run(link)
    saved = ", ".join(r["saved"]) or "nothing"
    text = f"saved: {saved}" + ("  (last kit NOT saved)" if r["unsaved"] else "")
    return CmdResult(r, text, 1 if r["unsaved"] else 0)


def cmd_pad(deps: CliDeps, ns: argparse.Namespace) -> CmdResult:
    dev = _resolve(deps, ns)
    with _cdc(deps, dev) as link:
        if ns.pad_cmd == "press":
            r = verbs.pad_press(link, _int_arg(ns.idx, "I", 0, 15), _int_arg(ns.vel, "VEL", 0, 127))
            return CmdResult(r or {"ok": True}, f"pad {ns.idx} pressed vel={ns.vel}")
        if ns.pad_cmd == "release":
            r = verbs.pad_release(link, _int_arg(ns.idx, "I", 0, 15))
            return CmdResult(r or {"ok": True}, f"pad {ns.idx} released")
        if ns.pad_cmd == "pressure":
            r = verbs.pad_pressure(link, _int_arg(ns.idx, "I", 0, 15),
                                   _int_arg(ns.val, "V", 0, 255))
            return CmdResult(r or {"ok": True}, f"pad {ns.idx} pressure={ns.val}")
        if ns.pad_cmd == "stats":
            r = verbs.pad_stats(link, reset=ns.reset)
            return CmdResult(r, _kv(r))
        r = verbs.pad_notes(link)
        notes = r["notes"]
        rows = []
        for row in range(3, -1, -1):
            rows.append("   " + "  ".join(f"{row * 4 + c:2d}:{notes.get(row * 4 + c, '?'):>3}"
                                          for c in range(4)))
        return CmdResult({"notes": {str(k): v for k, v in notes.items()}}, "\n".join(rows))


def cmd_ui(deps: CliDeps, ns: argparse.Namespace) -> CmdResult:
    dev = _resolve(deps, ns)
    with _cdc(deps, dev) as link:
        if ns.ui_cmd == "snapshot":
            snap = take_snapshot(dev, link, include=("apps", "ui")).to_dict()
            return CmdResult(snap, render_snapshot(snap))
        if ns.ui_cmd == "press":
            verbs.enc_press(link, ms=ns.ms)
            r = verbs.enc_focus(link)
            return CmdResult(r, "pressed; focus " + _kv(r))
        if ns.ui_cmd == "rotate":
            verbs.enc_rotate(link, int(ns.n))
            r = verbs.enc_focus(link)
            return CmdResult(r, f"rotated {ns.n}; focus " + _kv(r))
        group = verbs.enc_group(link)["group"]
        focus = verbs.enc_focus(link)
        delta = ref_to_delta(group, focus["index"], ns.ref)
        if delta:
            verbs.enc_rotate(link, delta)
        r = verbs.enc_focus(link)
        r["delta"] = delta
        return CmdResult(r, f"focus {ns.ref} (delta {delta}) -> {r['label']}")


def cmd_led(deps: CliDeps, ns: argparse.Namespace) -> CmdResult:
    dev = _resolve(deps, ns)
    with _cdc(deps, dev) as link:
        r = verbs.led_state(link)
    return CmdResult(r, render_led_grid(r))


def cmd_mem(deps: CliDeps, ns: argparse.Namespace) -> CmdResult:
    dev = _resolve(deps, ns)
    with _cdc(deps, dev) as link:
        if ns.blocks:
            r = verbs.mem_blocks(link)
            s = r["summary"]
            lines = [f"biggest used block {s.get('biggest_used')} B"]
            for b in s.get("buckets", []):
                lines.append(f"  <= {b['le']:>7}: used {b['used_n']:>4} ({b['used_b']:>7} B)  "
                             f"free {b['free_n']:>4} ({b['free_b']:>7} B)")
            for blk in r["big"]:
                lines.append(f"  big {blk['addr']} {blk['size']} B")
            return CmdResult(r, "\n".join(lines))
        r = verbs.mem(link)
    return CmdResult(r, _kv(r))


def cmd_ble(deps: CliDeps, ns: argparse.Namespace) -> CmdResult:
    if ns.ble_cmd.startswith("host-"):
        return cmd_ble_host(deps, ns)
    dev = _resolve(deps, ns)
    with _cdc(deps, dev) as link:
        c = ns.ble_cmd
        if c == "status":
            r = verbs.ble_status(link)
        elif c == "start":
            mode = None if ns.role is None else {"server": 1, "host": 0}[ns.role]
            r = verbs.ble_start(link, mode) or {"ok": True}
        elif c == "stop":
            r = verbs.ble_stop(link) or {"ok": True}
        elif c == "enable":
            r = verbs.ble_enable(link, ns.state == "on") or {"ok": True}
        elif c == "scan":
            r = verbs.ble_scan(link, ms=int(ns.ms)) or {"ok": True}
        elif c == "devices":
            r = verbs.ble_devices(link)
        elif c == "connect":
            r = verbs.ble_connect(link, ns.addr) or {"ok": True}
        elif c == "disconnect":
            r = verbs.ble_disconnect(link) or {"ok": True}
        else:
            r = verbs.ble_send(link, _int_arg(ns.note, "NOTE", 0, 127),
                               _int_arg(ns.vel, "VEL", 0, 127)) or {"ok": True}
    if c == "devices":
        lines = [f"{d['addr']}  rssi={d['rssi']}  {d['name']}" for d in r.get("devices", [])]
        return CmdResult(r, "\n".join(lines) or "(no BLE MIDI devices)")
    return CmdResult(r, _kv(r))


def cmd_ble_host(deps: CliDeps, ns: argparse.Namespace) -> CmdResult:
    """The other half of the link: this PC's own radio (`ble host-*`).

    Everything here needs bleak. `host-scan` is the only one that does not also need the
    board's CDC port — the rest read `BLE_STATUS self=` so they connect to *this* board and
    not to whatever else in the room advertises BLE MIDI.
    """
    c = ns.ble_cmd[len("host-"):]
    if c == "scan":
        found = ble_host.scan(float(ns.timeout))
        data = {"count": len(found), "peripherals": [p.to_dict() for p in found]}
        text = "\n".join(f"{p.address}  rssi={p.rssi}  {p.name!r}" for p in found)
        return CmdResult(data, text or "(no BLE MIDI peripherals in range)")

    dev = _resolve(deps, ns)
    with _cdc(deps, dev) as link:
        ble = ble_host.connect(getattr(ns, "address", None) or None, link=link,
                               scan_timeout_s=float(ns.timeout))
        try:
            if c == "listen":
                r = ble.listen(float(ns.seconds))
                r["status"] = ble.status()
                text = (f"{r['count']} messages in {r['seconds']:.1f}s "
                        f"({r['notes_on']} note-on)")
            elif c == "send":
                note = _int_arg(ns.note, "NOTE", 0, 127)
                vel = _int_arg(ns.vel, "VEL", 0, 127)
                before = int(verbs.ble_status(link).get("rx_msg", 0) or 0)
                ble.send_hit(note, vel, gate_s=float(ns.gate) / 1000.0)
                deps.sleep(0.5)
                after = int(verbs.ble_status(link).get("rx_msg", 0) or 0)
                r = {"note": note, "vel": vel, "rx_msg_before": before, "rx_msg_after": after,
                     "device_received": after - before}
                text = (f"sent note {note} vel {vel}; the device parsed "
                        f"{after - before} more BLE MIDI messages")
            else:                                    # latency
                r = ble_host.latency(ble, link, n=int(ns.n))
                rtt = r["rtt_ms"] or {}
                text = (f"device -> host: {r['received']}/{r['sent']} arrived"
                        + (f", rtt p50 {rtt.get('p50')} ms p90 {rtt.get('p90')} ms "
                           f"max {rtt.get('max')} ms" if rtt else ""))
        finally:
            ble.disconnect()
    r["address"] = ble.address
    return CmdResult(r, text)


def cmd_stim(deps: CliDeps, ns: argparse.Namespace) -> CmdResult:
    """Play the pads on a schedule and account for every hit that did not arrive."""
    dev = _resolve(deps, ns)
    pattern = None
    if ns.pattern:
        raw = json.loads(Path(ns.pattern).read_text(encoding="utf-8"))
        pattern = raw["pattern"] if isinstance(raw, dict) else raw
        if not isinstance(pattern, list):
            raise HilError(BAD_ARGS, f"{ns.pattern} must hold a list of "
                                     "{t_ms, pad, vel?, gate_ms?} steps")
    velocity: Any = ([int(v) for v in str(ns.velocity).split(",")]
                     if "," in str(ns.velocity) else _int_arg(ns.velocity, "--velocity", 0, 127))
    with hub_link.exclusive("stim"):
        status = stim_mod.run_stimulus(
            dev, pads=(None if pattern else ns.pads),
            rate_hz=(None if pattern else float(ns.rate)), pattern=pattern,
            seconds=float(ns.seconds), velocity=velocity, gate_ms=float(ns.gate),
            humanize_ms=float(ns.humanize), chord=bool(ns.chord), transport=ns.transport,
            seed=ns.seed, backoff=not ns.no_backoff, serial_factory=deps.serial_factory,
            midi_factory=deps.midi_factory)
    return CmdResult(status, render_stim(status), 0 if not status.get("errors") else 1)


def render_stim(status: dict) -> str:
    plan = status.get("plan") or {}
    lines = [f"{plan.get('transport')}: {status.get('presses_sent')} presses on pads "
             f"{plan.get('pads')} over {status.get('elapsed_s')}s"]
    thr = plan.get("throttled")
    if thr:
        lines.append(f"throttled: {thr['requested_rate_hz']} -> {thr['applied_rate_hz']} Hz "
                     f"({thr['reason']})")
    if plan.get("counters_unavailable"):
        lines.append(f"counters: {plan['counters_unavailable']}")
    loss = status.get("loss")
    if loss:
        lines.append(f"pads: sent={loss['sent']} accepted={loss['accepted']} "
                     f"played={loss['played']}  loss transport {loss['transport_pct']}% / "
                     f"engine {loss['engine_pct']}% / total {loss['total_pct']}%")
    delta = status.get("pad_stats_delta")
    if delta:
        lines.append("PAD_STATS delta: " + _kv(delta))
    drops = status.get("cdc_drops")
    if drops is not None:
        lines.append(f"cdc_drops: {drops}" + (f" in {len(status['drop_events'])} bursts"
                                              if status.get("drop_events") else ""))
    err = status.get("schedule_error_ms")
    if err:
        lines.append(f"schedule error ms: p50 {err['p50']} p90 {err['p90']} max {err['max']}")
    for e in status.get("errors") or []:
        lines.append(f"error: {e}")
    return "\n".join(lines)


def cmd_capture(deps: CliDeps, ns: argparse.Namespace) -> CmdResult:
    """Record what the board is playing, then put the USB profile back.

    The two things that decide whether this returns audio or silence are defaults
    here, not flags to remember: the RT mixer is resumed after the profile switch
    (SysEx 0x1D 06 01), and the recorded path is the DAC->ADC loop (`headphone` =
    codec1 LINE1) rather than the built-in mics.
    """
    dev = _resolve(deps, ns)
    session = deps.capture_factory(
        dev, seconds=float(ns.seconds), preset=ns.preset, out=ns.out, volume=ns.volume,
        resume_audio_tasks=not ns.no_resume_tasks, keep_mode=ns.keep_mode,
        sample_rate=ns.sample_rate, channels=int(ns.channels), logdir=ns.logdir,
        midi_factory=deps.midi_factory, set_mode_fn=deps.set_mode_fn,
        discover_fn=lambda: discover(deps.backends),
        log=lambda m: deps.err().write(m.rstrip("\n") + "\n"))
    with hub_link.exclusive("capture"):
        info = session.start()
        deps.err().write(f"recording {ns.seconds:.1f}s from "
                         f"{info['capture_device']['name']} ({info['backend']}), "
                         f"preset {info['preset']} -> codec{info['codec']} "
                         f"in{info['adc_input']}\n")
        try:
            session.wait()
        finally:
            result = session.stop()
    data = result.to_dict()
    if ns.analyze:
        data["analysis"] = deps.analyze_fn(ns.analyze, data["wav"])
    return CmdResult(data, render_capture(data), 1 if data["silent"] else 0)


def render_capture(r: dict) -> str:
    lines = [f"{r['wav']}  {r['seconds']:.2f}s  {r['sample_rate']} Hz x{r['channels']}",
             f"peak {r['peak_dbfs']} dBFS  rms {r['rms_dbfs']} dBFS  "
             f"overruns {r['overruns']}  preset {r['preset']}"]
    health = r.get("health") or {}
    if health.get("detail"):
        lines.append(f"health: {health['detail']}")
    if r.get("silent"):
        lines.append("SILENT: " + (r.get("hint") or "nothing on the capture endpoint"))
    if r.get("analysis"):
        lines.append(render_analysis(r["analysis"]))
    return "\n".join(lines)


def cmd_analyze(deps: CliDeps, ns: argparse.Namespace) -> CmdResult:
    """Offline: a WAV in, one typed verdict out. Touches no device."""
    expected = _json_value(ns.expected, "--expected") if ns.expected else None
    options = {}
    for item in ns.opt or []:
        if "=" not in item:
            raise HilError(BAD_ARGS, f"--opt takes key=value, got {item!r}")
        key, _, raw = item.partition("=")
        options[key.strip()] = _json_value(raw, f"--opt {key}", literal_ok=True)
    if ns.ref:
        options["ref"] = ns.ref
    data = deps.analyze_fn(ns.kind, ns.wav, expected=expected, **options)
    return CmdResult(data, render_analysis(data), 0 if data["passed"] else 1)


def cmd_jam(deps: CliDeps, ns: argparse.Namespace) -> CmdResult:
    """live_jam without a board: the set as a table, a run re-analysed, the kits built,
    a synthetic run to try the analysis on."""
    from crosspad_hil.jam import script as jscript

    if ns.jam_cmd == "script":
        s = jscript.build_script(ns.seed, minutes=ns.minutes, bpm=ns.bpm, soak_min=ns.soak_min,
                                 sections=ns.sections.split(",") if ns.sections else None)
        cov = jscript.coverage(s)
        return CmdResult({"script": s.to_dict(), "coverage": cov}, jscript.script_table(s), 0)
    if ns.jam_cmd == "analyze":
        from crosspad_hil.jam import analysis as jan
        from crosspad_hil.jam.detect import Tolerances

        tol = Tolerances(strict=ns.strict, late_ms=ns.late_ms)
        ja = jan.analyze_run(Path(ns.run_dir), tolerances=tol, snippets=ns.snippets,
                             allow_partial=ns.allow_partial)
        v = ja.doc["verdict"]
        text = "\n".join([f"live_jam: {v['status']}", *[f"  {ln}" for ln in v["lines"][:60]],
                          f"  report: {Path(ns.run_dir) / jan.REPORT}",
                          f"  timeline: {Path(ns.run_dir) / 'timeline.html'}"])
        return CmdResult({"verdict": v, "summary": ja.doc["summary"]}, text, v["exit"])
    if ns.jam_cmd == "kits":
        from crosspad_hil.jam import spec as jspec

        out = jspec.write_jam_kits(Path(ns.out))
        return CmdResult({k: str(v) for k, v in out.items()},
                         "\n".join(f"{k}: {v}" for k, v in out.items()), 0)
    if ns.jam_cmd == "synth":
        from crosspad_hil.jam import analysis as jan
        from crosspad_hil.jam import synth
        from crosspad_hil.jam.render import Fault

        sr = 44_100

        def faults(exp: Any, _sr: int) -> list:
            out = []
            for spec in ns.fault or []:
                kind, _, at = spec.partition("@")
                t = float(at or 5.0)
                out.append({"hole": Fault("zero_block", frame=int(t * sr), ms=20),
                            "click": Fault("click", frame=int(t * sr), amp=0.2),
                            "pilot": Fault("pilot_gap", frame=int(t * sr), ms=4),
                            "slip": Fault("capture_drop", frame=int(t * sr), samples=37)}[kind])
            return out

        synth.synth_run(Path(ns.out), seed=ns.seed, faults=faults)
        ja = jan.analyze_run(Path(ns.out), allow_partial=True)
        return CmdResult({"verdict": ja.doc["verdict"]},
                         f"synthetic run in {ns.out}: {ja.doc['verdict']['status']}",
                         ja.doc["verdict"]["exit"])
    raise HilError(BAD_ARGS, f"unknown jam command {ns.jam_cmd!r}")


def _json_value(raw: str, what: str, *, literal_ok: bool = False) -> Any:
    """`@file` reads JSON from a file; otherwise the string is parsed as JSON."""
    text = Path(raw[1:]).read_text(encoding="utf-8") if raw.startswith("@") else raw
    try:
        return json.loads(text)
    except ValueError as e:
        if literal_ok:
            return text
        raise HilError(BAD_ARGS, f"{what} is not JSON: {e}",
                       hint='e.g. --expected "[0.5, 1.0, 1.5]" or --expected @hits.json') from e


def render_analysis(r: dict) -> str:
    head = f"{r['kind']}: {r['verdict'].upper()}"
    detail = {
        "multitone": lambda: f"rms {r.get('rms_dbfs')} dBFS  glitches "
                             f"{r.get('glitch_bursts_per_s')}/s  dropouts {r.get('dropouts')}  "
                             + " ".join(f"{f}Hz={t['band_power']}"
                                        for f, t in (r.get("tones") or {}).items()),
        "click": lambda: f"{r.get('clicks')} clicks ({r.get('clicks_per_s')}/s)  "
                         f"rms {r.get('rms_dbfs')} dBFS",
        "onset": lambda: f"{r.get('matched')}/{r.get('expected')} matched  "
                         f"missed {r.get('missed')}  extra {len(r.get('extra') or [])}  "
                         f"latency p50/p90/max {_lat(r.get('latency_ms'))}",
        "velocity": lambda: f"{r.get('matched')}/{r.get('expected')} matched  range "
                            f"{r.get('dynamic_range_db')} dB  inversions "
                            f"{r.get('monotonic_inversions')}",
        "psd": lambda: f"carrier {r.get('dom_freq_hz')} Hz  sidebands "
                       + " ".join(f"{k}={v}dBc" for k, v in (r.get("sidebands") or {}).items()
                                  if v is not None),
        "silence": lambda: f"floor {r.get('rms_dbfs')} dBFS  peak {r.get('peak_dbfs')} dBFS  "
                           f"{r.get('glitches')} artifacts ({r.get('glitch_per_s')}/s)",
        # Slips are the number a level meter cannot show: a stream that drops or
        # repeats samples measures perfectly on level, glitch count and spectrum.
        "sine": lambda: f"{r.get('est_freq_hz')} Hz ({r.get('freq_offset_pct')}%)  rms "
                        f"{r.get('rms_dbfs')} dBFS  {r.get('timing_slips')} slips  drift "
                        f"{r.get('net_drift_samples')} samples  {r.get('glitches')} glitches "
                        f"({r.get('glitch_per_s')}/s)",
    }.get(r["kind"])
    lines = [head + ("  " + detail() if detail else "")]
    if r["kind"] == "velocity":
        for c in r.get("curve") or []:
            lines.append(f"  vel {c['vel']:>3}  peak {c['peak_mean_dbfs']:>7} dBFS  "
                         f"rel {c.get('measured_rel_db')} (expected {c['expected_rel_db']})")
    lines += [f"  - {reason}" for reason in r.get("reasons") or []]
    return "\n".join(lines)


def _lat(lat: dict | None) -> str:
    lat = lat or {}
    return "/".join("-" if lat.get(k) is None else str(lat[k]) for k in ("p50", "p90", "max"))


def cmd_diagnose(deps: CliDeps, ns: argparse.Namespace) -> CmdResult:
    """Crash evidence from a log file, or from the console the board is printing to now."""
    kw: dict[str, Any] = {"elf": ns.elf, "addr2line": ns.addr2line, "root": ns.idf_root,
                          "context_dir": ns.logdir, "context_lines": int(ns.context),
                          "decode": not ns.no_decode}
    if ns.log_file:
        bundle = diag.diagnose(log_file=ns.log_file, **kw)
    else:
        dev = _resolve(deps, ns)
        with _console(deps, dev) as con:
            deps.sleep(float(ns.seconds))
            lines = [line for _seq, line in con.read(since_seq=0, limit=50_000).lines]
        bundle = diag.diagnose(lines=lines, source=f"console:{dev.id}", **kw)
    return CmdResult(bundle, diag.render(bundle), 1 if bundle.get("found") else 0)


def cmd_midi(deps: CliDeps, ns: argparse.Namespace) -> CmdResult:
    if ns.midi_cmd == "hub":
        # Host only: the hub follows whatever board appears, it never needs one.
        from crosspad_hil import midi_hub
        if ns.hub_cmd == "run":
            midi_hub.run(lambda line: print(line, file=sys.stderr, flush=True))
            return CmdResult({}, "midi hub stopped")
        if ns.hub_cmd == "install":
            r = midi_hub.install(ns.exe)
            return CmdResult(r, f"installed {r['unit']}; open '{r['ports']}' as the DAW's "
                                "MIDI input and output (restart the DAW once to see it)")
        if ns.hub_cmd == "uninstall":
            return CmdResult(midi_hub.uninstall(), "midi hub service removed")
        if ns.hub_cmd == "versions":
            try:
                data = json.loads(midi_hub.VERSIONS_PATH.read_text())
                line = (f"{data['board']}: {data['verdict']} running={data['running']} "
                        f"board_latest={data['board_latest']} catalog={data['catalog_latest']} "
                        f"kit={data['kit']} channel={data['channel']}")
            except FileNotFoundError:
                return CmdResult({}, "no reading yet (is the hub running with a board "
                                     "attached?)", 2)
            except (OSError, ValueError, KeyError, TypeError) as exc:
                return CmdResult({}, f"unreadable {midi_hub.VERSIONS_PATH}: {exc}", 2)
            return CmdResult(data, line)
        r = midi_hub.status()
        hub_up = r["hub_readable"] and r["hub_writable"]
        return CmdResult(r, f"service={r['service']} hub={'up' if hub_up else 'down'} "
                            f"board={r['board_ports'] or '-'} "
                            f"{render_hub_control(r['control'])}")
    dev = _resolve(deps, ns)
    role = MidiRole.STM if getattr(ns, "port", "esp") == "stm" else MidiRole.ESP
    io = deps.midi_factory(dev, role)
    io.open()
    try:
        if ns.midi_cmd == "sysex":
            try:
                frame = bytes.fromhex(ns.hex.replace(" ", "").replace(",", ""))
            except ValueError as e:
                raise HilError(BAD_ARGS, f"not a hex string: {ns.hex!r}") from e
            io.send_sysex(frame)
            return CmdResult({"sent": len(frame)}, f"sent {len(frame)} bytes on {role.value}")
        if ns.midi_cmd == "note":
            note = _int_arg(ns.note, "N", 0, 127)
            vel = _int_arg(ns.vel, "VEL", 0, 127)
            io.send_note(ns.state == "on", note, vel)
            return CmdResult({"sent": 1, "on": ns.state == "on", "note": note, "vel": vel},
                             f"note {ns.state} {note} vel={vel}")
        r = echo_rtt(io, n=int(ns.n))
        rtt = r.get("rtt_ms", {})
        return CmdResult(r, f"sent={r['sent']} received={r['received']} lost={r['lost']}  "
                            f"rtt p50={rtt.get('p50')} p90={rtt.get('p90')} "
                            f"max={rtt.get('max')} ms")
    finally:
        io.close()


def cmd_companion(deps: CliDeps, ns: argparse.Namespace) -> CmdResult:
    """One companion verb over SysEx 0x1F — the audio profile's control plane."""
    dev = _resolve(deps, ns)
    verb = " ".join(ns.verb)
    io = deps.midi_factory(dev, MidiRole.ESP)
    io.open()
    try:
        text = companion_command(io, verb, timeout_s=ns.timeout)
    finally:
        io.close()
    return CmdResult({"verb": verb, "reply": text, "error": text.startswith("ERR")}, text.rstrip())


def cmd_usb_mode(deps: CliDeps, ns: argparse.Namespace) -> CmdResult:
    dev = _resolve(deps, ns)
    if ns.usb_cmd == "get":
        return CmdResult({"device": dev.id, "usb_mode": dev.usb_mode.value}, dev.usb_mode.value)
    mode = UsbMode(ns.mode)
    with hub_link.exclusive("usb-mode set"):
        new = deps.set_mode_fn(dev, mode, wait=not ns.no_wait,
                               discover_fn=lambda: discover(deps.backends),
                               midi_factory=deps.midi_factory)
    return CmdResult(new.to_dict(), f"{new.id} usb_mode={new.usb_mode.value}")


def cmd_fl(deps: CliDeps, ns: argparse.Namespace) -> CmdResult:
    """FL Studio host-side helpers; nothing here touches a board."""
    from crosspad_hil import fl_template
    if ns.fl_cmd == "template":
        drive_c = Path(ns.drive_c) if ns.drive_c else fl_template.default_drive_c()
        if drive_c is None:
            raise HilError(BAD_ARGS, "no Wine prefix with FL Studio found",
                           hint="pass --drive-c with the prefix's drive_c directory")
        if ns.template_cmd == "install":
            input_index = fl_template.CROSSPAD_INPUT if ns.input is None else ns.input
            r = fl_template.install(drive_c, tracks=ns.tracks, input_index=input_index,
                                    empty=Path(ns.empty) if ns.empty else None)
            return CmdResult(r, f"{fl_template.TEMPLATE_NAME}.flp ({r['tracks']} insert(s) on "
                                f"input {r['input']}) -> " + ", ".join(r["written"]) +
                                "\nFL Studio: File > New from template > "
                                f"{fl_template.TEMPLATE_CATEGORY} > {fl_template.TEMPLATE_NAME}")
        r = fl_template.status(drive_c)
        lines = [f"{p}: inputs={r['inputs'][p]}" for p in r["installed"]] or ["not installed"]
        return CmdResult(r, "\n".join(lines))
    raise HilError(BAD_ARGS, f"unknown fl command {ns.fl_cmd}")


def cmd_reaper(deps: CliDeps, ns: argparse.Namespace) -> CmdResult:
    """REAPER host-side helpers; nothing here touches a board."""
    from crosspad_hil import reaper_setup
    resource = Path(ns.resource) if ns.resource else reaper_setup.default_resource_dir()
    if ns.reaper_cmd == "install":
        script = Path(ns.script) if ns.script else reaper_setup.find_script()
        if script is None:
            raise HilError(BAD_ARGS, "no CrossPad.lua to install",
                           hint=f"pass --script, or run from a platform-idf checkout "
                                f"({reaper_setup.REPO_SCRIPT})")
        r = reaper_setup.install(resource, script)
        midi = r.get("midi")
        tail = "" if midi is None else (f"\nMIDI: {midi['why']}" if midi["enabled"]
                                        else f"\nMIDI not enabled: {midi['why']}")
        return CmdResult(r, f"{r['script']} installed, runs with every REAPER start{tail}")
    r = reaper_setup.status(resource)
    return CmdResult(r, f"script: {'yes' if r['script_installed'] else 'no'}, "
                        f"starts with REAPER: {'yes' if r['starts_with_reaper'] else 'no'}")


def cmd_audio(deps: CliDeps, ns: argparse.Namespace) -> CmdResult:
    """Audio routing and levels.

    `route` goes over SysEx (it works in both USB profiles, unlike CDC);
    `level` and `tasks` are CDC verbs; `nodes` never touches the board — it
    keeps the host's PipeWire nodes for the UAC2 card alive across hotplug.
    """
    if ns.audio_cmd == "nodes":
        from crosspad_hil import audio_nodes
        if ns.nodes_cmd == "run":
            audio_nodes.run(lambda line: print(line, file=sys.stderr, flush=True),
                            defaults=ns.take_defaults, apps=ns.apps)
            return CmdResult({}, "audio nodes watcher stopped")
        if ns.nodes_cmd == "install":
            r = audio_nodes.install(ns.exe, take_defaults=ns.take_defaults, apps=ns.apps)
            return CmdResult(r, f"installed {r['unit']} (+ {r['wireplumber_rule']}); "
                                "restart wireplumber once so the rule applies")
        if ns.nodes_cmd == "uninstall":
            r = audio_nodes.uninstall()
            return CmdResult(r, "audio nodes service removed")
        r = audio_nodes.status()
        return CmdResult(r, f"service={r['service']} cards={r['cards']} "
                            f"mic={'up' if r['mic'] else 'down'} "
                            f"out={'up' if r['out'] else 'down'}")
    if ns.audio_cmd == "gain":
        from crosspad_hil import host_gain
        cards = [d.ports.uac2.alsa_id for d in discover(deps.backends)
                 if d.ports.uac2 is not None and d.ports.uac2.alsa_id]
        card = cards[0] if cards else None
        if ns.set:
            before, after = host_gain.apply(card)
            r = {"before": before.to_dict(), "after": after.to_dict()}
            dev = after.deviations()
            text = (f"card {card}: was {', '.join(before.deviations()) or '0 dB'}; now "
                    f"{', '.join(dev) or '0 dB'}")
            return CmdResult(r, text, 2 if dev else 0)
        g = host_gain.read(card).to_dict()
        text = (f"card {card}: {', '.join(g['deviations']) or '0 dB everywhere'}"
                + "".join(f"\n  note: {n}" for n in g["notes"]))
        return CmdResult(g, text)
    dev = _resolve(deps, ns)
    if ns.audio_cmd == "route":
        io = deps.midi_factory(dev, MidiRole.ESP)
        io.open()
        try:
            if ns.sub is None:
                state = query_route(io, timeout_s=2.0)
                text = (f"mic_src=codec{state['mic_src']}  "
                        f"adc={state['adc']}  out={state['out']}  "
                        f"vol={state['vol']}  mute={state['mute']}")
                return CmdResult(state, text)
            sub = int(load_knowledge("sysex")["audio_route"]["subs"][ns.sub])
            args = [int(a) for a in ns.args]
            io.send_sysex(sysex_audio_route(sub, *args))
            return CmdResult({"sent": ns.sub, "args": args},
                             f"{ns.sub} {' '.join(map(str, args))} sent")
        finally:
            io.close()
    with _cdc(deps, dev) as link:
        if ns.audio_cmd == "level":
            r = verbs.audio_level(link)
            return CmdResult(r, f"L={r['left']:.3f} R={r['right']:.3f} amp={r['amp']} "
                                f"allowed={r['allowed']}")
        on = ns.state == "on"
        verbs.audio_tasks(link, on)
        return CmdResult({"audio_tasks": on}, f"audio tasks {'resumed' if on else 'suspended'}")


def cmd_snapshot(deps: CliDeps, ns: argparse.Namespace) -> CmdResult:
    dev = _resolve(deps, ns)
    with _cdc(deps, dev) as link:
        snap = take_snapshot(dev, link).to_dict()
    return CmdResult(snap, render_snapshot(snap))


def cmd_flash(deps: CliDeps, ns: argparse.Namespace) -> CmdResult:
    dev = _resolve(deps, ns)
    fw = Path(ns.firmware)
    if not fw.is_file():
        raise HilError(BAD_ARGS, f"firmware not found: {fw}", hint="idf.py build first")
    if ns.delta and not ns.base_fw:
        raise HilError(BAD_ARGS, "--delta requires --base-fw (firmware currently on the device)")
    transport = "uart" if ns.uart else "ota"
    err = deps.err()

    def progress(done: int, total: int) -> None:
        if not ns.as_json and total:
            err.write(f"\rProgress: {done * 100 // total}% ({done}/{total})")
            err.flush()

    kw: dict[str, Any] = {"transport": transport, "wait_boot": ns.wait_boot, "console": None,
                          "progress": progress}
    if ns.delta:
        kw["delta_base"] = Path(ns.base_fw)
    with hub_link.exclusive("ota flash"):
        r = deps.flash_fn(dev, fw, **kw)
    if not ns.as_json:
        err.write("\n")
    f = r.get("flash") or {}
    text = (f"flashed {f.get('bytes')} B in {f.get('seconds', 0):.1f}s "
            f"({f.get('kbps', 0):.0f} kB/s) "
            f"mode={f.get('mode')} version={f.get('version')}")
    boot = r.get("boot")
    code = 0
    if boot is not None:
        ok = boot.get("complete") and not boot.get("fatal") and not boot.get("bootloops")
        text += f"\nboot: {'complete' if ok else 'FAILED'} in {boot.get('seconds', 0):.1f}s"
        if boot.get("missing"):
            text += "\n  missing: " + ", ".join(boot["missing"])
        if boot.get("fatal"):
            text += "\n  fatal: " + json.dumps(boot["fatal"])[:200]
        code = 0 if ok else 1
    return CmdResult(r, text, code)


def cmd_bootloader(deps: CliDeps, ns: argparse.Namespace) -> CmdResult:
    dev = _resolve(deps, ns)
    target = "stm" if ns.stm else "esp"
    with hub_link.exclusive("bootloader"):
        r = deps.bootloader_fn(dev, target, method=ns.method, timeout_s=ns.timeout)
    port = r.get("bootloader_port")
    return CmdResult(r, f"{target} bootloader port: {port}" if port else
                     f"{target} bootloader request sent; no bootloader port seen yet")


class CliProgress(scen.Progress):
    def __init__(self, stream: TextIO, quiet: bool) -> None:
        self._stream = stream
        self._quiet = quiet

    def __call__(self, progress: int, total: int | None, message: str) -> None:
        if self._quiet:
            return
        tot = f"/{total}" if total is not None else ""
        self._stream.write(f"[{progress}{tot}] {message}\n")
        self._stream.flush()


def cmd_run(deps: CliDeps, ns: argparse.Namespace) -> CmdResult:
    name = ns.scenario
    s = get_scenario(name)
    params = scen.argparse_to_params(s.Params, ns)
    dev = _resolve(deps, ns)
    workdir = Path(ns.logdir) / f"{name}_{time.strftime('%Y%m%d_%H%M%S')}"
    holder = _holder(ns)
    with _keep_lease(deps, ns, dev.id):
        report = deps.run_scenario_fn(name, params, device=dev, workdir=workdir,
                                      progress=CliProgress(deps.err(), ns.as_json),
                                      log=lambda m: deps.err().write(str(m) + "\n"),
                                      **({"holder": holder} if holder else {}))
    data = dataclasses.asdict(report)
    data["scenario"] = name
    data["workdir"] = str(workdir)
    return CmdResult(data, render_report(report), report.exit_code)


def cmd_record(deps: CliDeps, ns: argparse.Namespace) -> CmdResult:
    dev = _resolve(deps, ns)
    out_path = Path(ns.out)

    def factory(path: str, **kw: Any) -> Any:
        return RecordingSerial(deps.serial_factory(path, **kw), out_path)

    rec_deps = dataclasses.replace(deps, serial_factory=factory)
    n = 0
    out = deps.out()
    with hub_link.exclusive("record"), _cdc(rec_deps, dev) as link:
        for raw in deps.inp():
            cmd = raw.strip()
            if not cmd:
                continue
            n += 1
            try:
                reply = link.transact(cmd, timeout_s=2.0)
                out.write(reply.line + "\n")
                for extra in reply.extra_lines:
                    out.write(extra + "\n")
            except HilError as e:
                out.write(f"error: {e.code}: {e.message}\n")
            out.flush()
    return CmdResult({"out": str(out_path), "commands": n}, f"{n} commands recorded to {out_path}")


def _bench_device(deps: CliDeps, ns: argparse.Namespace) -> str:
    """A lease is keyed by device id: an id is used as given (an unplugged board can
    still be claimed or released), anything else goes through discovery."""
    device = getattr(ns, "device", None)
    if device and device.startswith("dev_"):
        return device
    try:
        return select(discover(deps.backends), device).id
    except HilError:
        known = deps.lease().known_devices()
        if device or len(known) != 1:
            raise
        return known[0]


def cmd_bench(deps: CliDeps, ns: argparse.Namespace) -> CmdResult:
    bench = deps.lease()
    sub = ns.bench_cmd
    if sub == "status" or sub == "md":
        device = getattr(ns, "device", None)
        if device and not device.startswith("dev_"):
            device = _bench_device(deps, ns)
        status = bench.status(device, history=ns.history)
        if sub == "md" and ns.write:
            write_markdown(ns.write, status)
            return CmdResult({"written": ns.write, **status}, f"wrote {ns.write}")
        return CmdResult(status, render_markdown(status).rstrip())
    holder = _holder(ns)
    if not holder:
        raise HilError(BAD_ARGS, "no holder: pass --holder NAME or set "
                                 f"{HOLDER_ENV}", hint="the holder is your session's name")
    device = _bench_device(deps, ns)
    force = bool(getattr(ns, "bench_force", False))
    if sub == "claim":
        r = bench.claim(device, holder, ns.purpose or "",
                        None if ns.ttl is None else ns.ttl * 60.0, force=force)
        return CmdResult(r, r["message"], 0 if r["granted"] else 1)
    if sub == "renew":
        r = bench.renew(device, holder, None if ns.ttl is None else ns.ttl * 60.0)
        return CmdResult(r, f"{holder} holds {device} until {r['lease']['expires_at']}")
    r = bench.release(device, holder, ns.firmware_left, force=force)
    text = (f"{holder} released {device}" if r["released"]
            else f"{holder} left the queue for {device}")
    return CmdResult(r, text)


def cmd_serve(deps: CliDeps, ns: argparse.Namespace) -> CmdResult:
    from .serve import Daemon

    Daemon(deps.inp(), deps.out(), backends=deps.backends, serial_factory=deps.serial_factory).run()
    return CmdResult({"ok": True}, "", 0)


# --------------------------------------------------------------------------- parser

def _common(suppress: bool) -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(add_help=False)
    dflt = argparse.SUPPRESS if suppress else None
    p.add_argument("-d", "--device", dest="device", default=dflt,
                   help="device id (dev_xxxx) or a port path; implicit when one board is attached")
    p.add_argument("--json", dest="as_json", action="store_true",
                   default=argparse.SUPPRESS if suppress else False,
                   help="print the same object the daemon returns")
    p.add_argument("--holder", dest="holder", default=dflt, metavar="NAME",
                   help=f"bench lease holder (your session's name; default ${HOLDER_ENV})")
    p.add_argument("--bench-force", dest="bench_force", action="store_true",
                   default=argparse.SUPPRESS if suppress else False,
                   help="touch the board even though another holder has claimed it (logged)")
    p.add_argument("--no-hub", dest="no_hub", action="store_true",
                   default=argparse.SUPPRESS if suppress else False,
                   help="open the board's port directly, not through the CrossPad hub "
                        f"(also ${hub_link.NO_HUB_ENV}=1)")
    return p


def build_parser() -> argparse.ArgumentParser:
    top = _common(False)
    sub_common = _common(True)
    ap = argparse.ArgumentParser(prog="crosspad-hil", parents=[top],
                                 description="Talk to a CrossPad: discovery, console, CDC verbs, "
                                             "MIDI, USB mode, OTA, snapshots, HIL scenarios.")
    ap.add_argument("--version", action="version", version=f"crosspad-hil {__version__}")
    sub = ap.add_subparsers(dest="command", required=True, metavar="COMMAND")

    def add(name: str, func: Handler, help_: str) -> argparse.ArgumentParser:
        p = sub.add_parser(name, help=help_, parents=[sub_common])
        p.set_defaults(func=func)
        return p

    p = add("devices", cmd_devices, "list attached CrossPads and their ports")
    p.add_argument("--watch", action="store_true", help="re-list every second until Ctrl-C")

    add("doctor", cmd_doctor, "check deps, permissions, port locks, devices")

    p = add("console", cmd_console, "STM bridge console: tail, or --expect patterns")
    p.add_argument("--reset", action="store_true", help="pulse DTR/RTS (reboots the ESP) first")
    p.add_argument("--expect", nargs="+", metavar="P", help="regexes; exit 0 on first hit")
    p.add_argument("--reject", nargs="+", metavar="P", help="regexes that fail the expect")
    p.add_argument("--timeout", type=float, default=30.0, help="seconds (tail: 0 = until Ctrl-C)")
    p.add_argument("--log", metavar="FILE", help="also append every line to FILE")

    p = add("cdc", cmd_cdc, "send a raw CDC verb and print the reply")
    p.add_argument("verb", metavar="VERB")
    p.add_argument("args", nargs="*", metavar="ARGS")
    p.add_argument("--expect", metavar="PREFIX", help="reply prefix (default: from cdc.yaml)")
    p.add_argument("--timeout", type=float, default=None,
                   help="milliseconds (default: the verb's own from cdc.yaml, else 2000)")

    p = add("app", cmd_app, "list / start NAME / stop / versions")
    s = p.add_subparsers(dest="app_cmd", required=True)
    s.add_parser("list", parents=[sub_common])
    q = s.add_parser("start", parents=[sub_common])
    q.add_argument("name", metavar="NAME")
    q.add_argument("--wait", type=float, default=3.0, help="seconds to wait for APP_LIST running=")
    s.add_parser("stop", parents=[sub_common])
    s.add_parser("versions", parents=[sub_common])

    p = add("kit", cmd_kit,
            "list / load ID [--wait] / status / push DIR / from-fpc FST DIR / to-fpc DIR FST")
    s = p.add_subparsers(dest="kit_cmd", required=True)
    s.add_parser("list", parents=[sub_common])
    q = s.add_parser("load", parents=[sub_common])
    q.add_argument("id", metavar="ID")
    q.add_argument("--wait", action="store_true", help="wait until KIT_STATUS reports the kit")
    q.add_argument("--timeout", type=float, default=15.0, help="seconds for --wait")
    s.add_parser("status", parents=[sub_common])
    q = s.add_parser("push", parents=[sub_common],
                     help="build a kit from a directory of WAVs and push it onto the card")
    q.add_argument("dir", metavar="DIR", help="directory of *.wav files, one per pad (sorted)")
    q.add_argument("--name", metavar="NAME", help="kit name (default: DIR's basename)")
    q.add_argument("--load", action="store_true", help="KIT_LOAD it once the scan finds it")
    q.add_argument("--rate", type=int, default=44100, help="kit sample rate (default: 44100)")
    q.add_argument("--logdir", default="hil_logs", help="where the build + console log go")

    q = s.add_parser("from-fpc", parents=[sub_common],
                     help="convert an FL Studio FPC preset (.fst) into a kit directory")
    q.add_argument("preset", metavar="PRESET.fst", help="the FPC preset to read")
    q.add_argument("out_dir", metavar="OUT_DIR", help="where the kit directory is created")
    q.add_argument("--name", metavar="NAME", help="kit name (default: the preset's own)")
    q.add_argument("--bank", default="A", choices=["A", "B", "a", "b"],
                   help="which FPC bank becomes the 16 pads (default: A)")
    q.add_argument("--drive-c", metavar="PATH", dest="drive_c",
                   help="the Wine prefix's drive_c (default: found above the preset)")
    q.add_argument("--fl-data", metavar="PATH", dest="fl_data",
                   help="what %%FLStudioFactoryData%% means (default: the FL install found "
                        "under drive_c)")
    q.add_argument("--rate", type=int, default=44100, help="kit sample rate (default: 44100)")
    q.add_argument("--push", action="store_true", help="push the kit onto the card afterwards")
    q.add_argument("--load", action="store_true", help="with --push, KIT_LOAD it once scanned")
    q.add_argument("--logdir", default="hil_logs", help="where the console log goes with --push")

    q = s.add_parser("to-fpc", parents=[sub_common],
                     help="write an FPC preset whose pads point at a kit directory's WAVs")
    q.add_argument("dir", metavar="KIT_DIR", help="a kit directory (kit.json + SAMPLES/)")
    q.add_argument("out", metavar="OUT.fst", help="the preset to write")
    q.add_argument("--win-prefix", metavar="PATH", dest="win_prefix",
                   help="the kit directory as Windows sees it, e.g. "
                        r"'C:\users\me\Documents\CrossPad\MY_KIT'")
    q.add_argument("--name", metavar="NAME", help="preset name (default: the kit's own)")

    add("padmix", cmd_padmix, "TUI: the current kit's 16 pad volumes, live on the board "
        "and saved into its kit.json; Enter picks another kit")

    p = add("pad", cmd_pad, "press I [VEL] / release I / pressure I V / stats [--reset] / notes")
    s = p.add_subparsers(dest="pad_cmd", required=True)
    q = s.add_parser("press", parents=[sub_common])
    q.add_argument("idx", metavar="I")
    q.add_argument("vel", nargs="?", default=127, metavar="VEL")
    q = s.add_parser("release", parents=[sub_common])
    q.add_argument("idx", metavar="I")
    q = s.add_parser("pressure", parents=[sub_common])
    q.add_argument("idx", metavar="I")
    q.add_argument("val", metavar="V")
    q = s.add_parser("stats", parents=[sub_common])
    q.add_argument("--reset", action="store_true")
    s.add_parser("notes", parents=[sub_common])

    p = add("ui", cmd_ui, "snapshot / focus REF / press / rotate N")
    s = p.add_subparsers(dest="ui_cmd", required=True)
    s.add_parser("snapshot", parents=[sub_common])
    q = s.add_parser("focus", parents=[sub_common])
    q.add_argument("ref", metavar="REF", help="e<i> from ui snapshot")
    q = s.add_parser("press", parents=[sub_common])
    q.add_argument("--ms", type=int, default=80)
    q = s.add_parser("rotate", parents=[sub_common])
    q.add_argument("n", type=int, metavar="N")

    p = add("led", cmd_led, "state: the LED controller's model as a 4x4 grid")
    s = p.add_subparsers(dest="led_cmd", required=True)
    s.add_parser("state", parents=[sub_common])

    p = add("mem", cmd_mem, "free/largest/min internal + PSRAM; --blocks for the histogram")
    p.add_argument("--blocks", action="store_true")

    p = add("ble", cmd_ble, "status / start [server|host] / stop / scan [MS] / devices / "
                            "enable on|off / connect ADDR / disconnect / send NOTE [VEL]")
    s = p.add_subparsers(dest="ble_cmd", required=True)
    s.add_parser("status", parents=[sub_common])
    q = s.add_parser("start", parents=[sub_common])
    q.add_argument("role", nargs="?", choices=["server", "host"], default=None)
    s.add_parser("stop", parents=[sub_common])
    q = s.add_parser("enable", parents=[sub_common],
                     help="persist BLE on/off (wireless.enableBLE) and apply it now")
    q.add_argument("state", choices=["on", "off"])
    q = s.add_parser("scan", parents=[sub_common])
    q.add_argument("ms", nargs="?", default=5000, metavar="MS")
    s.add_parser("devices", parents=[sub_common])
    q = s.add_parser("connect", parents=[sub_common])
    q.add_argument("addr", metavar="ADDR")
    s.add_parser("disconnect", parents=[sub_common])
    q = s.add_parser("send", parents=[sub_common])
    q.add_argument("note", metavar="NOTE")
    q.add_argument("vel", nargs="?", default=100, metavar="VEL")
    # host-*: this PC's own radio (needs bleak), the other half of every BLE leg.
    q = s.add_parser("host-scan", parents=[sub_common],
                     help="scan this PC's radio for BLE MIDI peripherals")
    q.add_argument("--timeout", type=float, default=ble_host.DEFAULT_SCAN_TIMEOUT_S)
    for name, helptext in (("host-listen", "connect and print what the board sends"),
                           ("host-send", "write a note from this PC to the board"),
                           ("host-latency", "device -> host round trip over BLE_SEND")):
        q = s.add_parser(name, parents=[sub_common], help=helptext)
        q.add_argument("--address", metavar="ADDR",
                       help="skip the scan; default is BLE_STATUS self=")
        q.add_argument("--timeout", type=float, default=ble_host.DEFAULT_SCAN_TIMEOUT_S,
                       help="scan seconds")
        if name == "host-listen":
            q.add_argument("--seconds", type=float, default=5.0)
        elif name == "host-send":
            q.add_argument("note", metavar="NOTE")
            q.add_argument("vel", nargs="?", default=100, metavar="VEL")
            q.add_argument("--gate", type=float, default=50.0, help="note-on to note-off, ms")
        else:
            q.add_argument("--n", type=int, default=8, help="notes to time")

    p = add("midi", cmd_midi, "sysex HEX [--port esp|stm] / note on|off N [VEL] / echo [--n N]")
    s = p.add_subparsers(dest="midi_cmd", required=True)
    q = s.add_parser("sysex", parents=[sub_common])
    q.add_argument("hex", metavar="HEX", help='e.g. "F0 7D 1D 10 F7"')
    q.add_argument("--port", dest="port", choices=["esp", "stm"], default="esp")
    q = s.add_parser("note", parents=[sub_common])
    q.add_argument("state", choices=["on", "off"])
    q.add_argument("note", metavar="N")
    q.add_argument("vel", nargs="?", default=100, metavar="VEL")
    q = s.add_parser("echo", parents=[sub_common])
    q.add_argument("--n", type=int, default=20)
    q = s.add_parser("hub", help="host only: fixed 'CrossPad Hub' ALSA ports a DAW under Wine "
                                 "opens instead of the board; bridged to whichever CrossPad "
                                 "appears")
    n = q.add_subparsers(dest="hub_cmd", required=True)
    n.add_parser("run", help="hold the hub ports and follow the board (what the service runs)")
    i = n.add_parser("install", help="systemd --user unit, enabled now")
    i.add_argument("--exe", default=None, help="absolute crosspad-hil path for the unit")
    n.add_parser("uninstall")
    n.add_parser("status")
    n.add_parser("versions", help="what the hub last read from the board over SysEx, "
                                  "against crosspad.app")

    p = add("companion", cmd_companion,
            "VERB [ARG...] — one companion verb over SysEx 0x1F (works in the audio profile)")
    p.add_argument("verb", nargs="+", metavar="VERB",
                   help='e.g. STM_FW_INFO, "APP_START Sampler", "PAD_PRESS 3 100"')
    p.add_argument("--timeout", type=float, default=5.0, help="seconds to wait for the reply")

    p = add("usb-mode", cmd_usb_mode, "get / set default|audio [--no-wait]")
    s = p.add_subparsers(dest="usb_cmd", required=True)
    s.add_parser("get", parents=[sub_common])
    q = s.add_parser("set", parents=[sub_common])
    q.add_argument("mode", choices=["default", "audio"])
    q.add_argument("--no-wait", action="store_true", help="do not wait for re-enumeration")

    p = add("audio", cmd_audio, "route (query or set over SysEx), level, tasks on|off")
    s = p.add_subparsers(dest="audio_cmd", required=True)
    q = s.add_parser("route", parents=[sub_common])
    q.add_argument("sub", nargs="?", choices=sorted(load_knowledge("sysex")["audio_route"]["subs"]),
                   help="omit to query the current routing")
    q.add_argument("args", nargs="*", metavar="ARG", help="sub-command arguments, e.g. 1 2")
    s.add_parser("level", parents=[sub_common])
    q = s.add_parser("gain", parents=[sub_common],
                     help="host only: the ALSA mixer and PipeWire volumes on the CrossPad card")
    q.add_argument("--set", action="store_true",
                   help="put them at 0 dB and read back (exit 2 when one will not move)")
    q = s.add_parser("tasks", parents=[sub_common])
    q.add_argument("state", choices=["on", "off"])
    q = s.add_parser("nodes", help="host only: CrossPad UAC2 as 'CrossPad Microphone' / "
                                   "'CrossPad Output' PipeWire nodes, re-created on every hotplug")
    n = q.add_subparsers(dest="nodes_cmd", required=True)
    r = n.add_parser("run", help="watch udev and hold the nodes (what the service runs)")
    r.add_argument("--take-defaults", action="store_true",
                   help="make the nodes the default source/sink whenever they come up "
                        "(a DAW under Wine only follows '[Default input/output]')")
    r.add_argument("--apps", nargs="*", default=None, metavar="APP",
                   help="applications whose open streams move onto the nodes with "
                        "--take-defaults (default: 'FL Studio')")
    i = n.add_parser("install", help="systemd --user unit + WirePlumber rule, enabled now")
    i.add_argument("--exe", default=None, help="absolute crosspad-hil path for the unit")
    i.add_argument("--take-defaults", action="store_true",
                   help="bake --take-defaults into the unit")
    i.add_argument("--apps", nargs="*", default=None, metavar="APP",
                   help="bake --apps into the unit")
    n.add_parser("uninstall")
    n.add_parser("status")

    p = add("fl", cmd_fl, "host only: FL Studio helpers — template install|status")
    s = p.add_subparsers(dest="fl_cmd", required=True)
    q = s.add_parser("template", help="a project template whose first insert(s) record the "
                                      "CrossPad (FL's API cannot pick a track's audio input)")
    n = q.add_subparsers(dest="template_cmd", required=True)
    i = n.add_parser("install", help="build it from FL's Empty.flp and put it where "
                                     "File > New from template looks")
    i.add_argument("--tracks", type=int, default=1, help="how many inserts to route (default: 1)")
    i.add_argument("--input", type=int, default=None,
                   help="index in the driver's input list (default: 0 = 'In 1 - In 2', the "
                        "[Default input] pair the CrossPad Microphone node feeds)")
    i.add_argument("--drive-c", metavar="PATH", dest="drive_c",
                   help="the Wine prefix's drive_c (default: the Bottles/Wine prefix holding FL)")
    i.add_argument("--empty", metavar="EMPTY.flp", help="FL's empty template (default: found "
                                                        "under the FL install)")
    st = n.add_parser("status", help="which profiles have the template and what it routes")
    st.add_argument("--drive-c", metavar="PATH", dest="drive_c")

    p = add("reaper", cmd_reaper, "host only: REAPER control script — install|status")
    s = p.add_subparsers(dest="reaper_cmd", required=True)
    for name, help_text in (("install", "copy CrossPad.lua into REAPER and start it with REAPER"),
                            ("status", "is the script there, and does REAPER start it")):
        q = s.add_parser(name, help=help_text)
        q.add_argument("--resource", metavar="PATH",
                       help="REAPER's resource folder (default: ~/.config/REAPER on Linux)")
        if name == "install":
            q.add_argument("--script", metavar="CrossPad.lua",
                           help="the control script (default: from the platform-idf checkout "
                                "in the current directory)")

    add("snapshot", cmd_snapshot, "one-call device state: apps, ui, kit, leds, pads, mem, ble")

    p = add("stim", cmd_stim, "play the pads on a schedule; reports PAD_STATS delta and drops")
    p.add_argument("--pads", default="0-15", metavar="SPEC", help="e.g. 0,2,4-7")
    p.add_argument("--rate", type=float, default=stim_mod.DEFAULT_RATE_HZ, metavar="HZ",
                   help=f"hits per second (CDC is capped at "
                        f"{stim_mod.CDC_MAX_HIT_RATE_HZ:.0f} Hz: app_queue is 64 deep)")
    p.add_argument("--seconds", type=float, default=stim_mod.DEFAULT_SECONDS)
    p.add_argument("--velocity", default=stim_mod.DEFAULT_VELOCITY, metavar="V",
                   help="0..127, or a comma-separated set cycled across the hits")
    p.add_argument("--gate", type=float, default=stim_mod.DEFAULT_GATE_MS, metavar="MS",
                   help="note-on to note-off")
    p.add_argument("--humanize", type=float, default=0.0, metavar="MS",
                   help="jitter each press by +/- this much")
    p.add_argument("--chord", action="store_true", help="hit every pad at once, not round-robin")
    p.add_argument("--pattern", metavar="FILE",
                   help="JSON list of {t_ms, pad, vel?, gate_ms?}; replaces --rate")
    p.add_argument("--transport", choices=list(stim_mod.TRANSPORTS), default="cdc")
    p.add_argument("--seed", type=int, default=None, help="make --humanize reproducible")
    p.add_argument("--no-backoff", action="store_true",
                   help="do not stretch the schedule when the device reports drops")

    p = add("diagnose", cmd_diagnose, "crash bundle: reset reason, decoded backtrace, heap, "
                                      "300 lines of context")
    p.add_argument("--log", "--log-file", dest="log_file", metavar="FILE",
                   help="a console log; without it, listen to the board for --seconds")
    p.add_argument("--elf", metavar="FILE",
                   help="default: the build whose SHA256 matches what the firmware printed")
    p.add_argument("--addr2line", metavar="FILE", help="override the toolchain binary")
    p.add_argument("--idf-root", dest="idf_root", metavar="DIR",
                   help="where platform-idf lives (default $CROSSPAD_IDF_ROOT, ~/GIT/platform-idf)")
    p.add_argument("--seconds", type=float, default=3.0, help="console listen time, no --log")
    p.add_argument("--context", type=int, default=diag.DEFAULT_CONTEXT_LINES,
                   help="lines of console around the panic")
    p.add_argument("--logdir", default="hil_logs", help="where the context file is written")
    p.add_argument("--no-decode", action="store_true",
                   help="report raw backtrace addresses; needs no toolchain")

    p = add("capture", cmd_capture, "record the UAC2 capture endpoint (enters audio mode, "
                                    "resumes the RT mixer, restores the profile)")
    p.add_argument("--seconds", type=float, default=3.0, metavar="S")
    p.add_argument("--preset", choices=sorted(cap_mod.PRESETS), default=cap_mod.DEFAULT_PRESET,
                   help="signal path: headphone/speaker = the DAC->ADC loop, mics = the room")
    p.add_argument("--out", metavar="FILE", help="WAV path (default: LOGDIR/capture_<ts>/)")
    p.add_argument("--logdir", default="hil_logs", metavar="DIR")
    p.add_argument("--volume", type=int, metavar="V", help="codec volume 0..100 before recording")
    p.add_argument("--channels", type=int, default=cap_mod.CHANNELS_DEFAULT, metavar="N")
    p.add_argument("--sample-rate", dest="sample_rate", type=int, metavar="R")
    p.add_argument("--no-resume-tasks", dest="no_resume_tasks", action="store_true",
                   help="do not send SysEx 0x1D 06 01 (the capture will be silent)")
    p.add_argument("--keep-mode", dest="keep_mode", action="store_true",
                   help="stay in the MIDI+UAC2 profile afterwards")
    p.add_argument("--analyze", choices=sorted(ANALYZE_KINDS), metavar="KIND",
                   help="run one analysis on the recording: " + ", ".join(sorted(ANALYZE_KINDS)))

    p = add("analyze", cmd_analyze, "offline verdict on a WAV: " + ", ".join(sorted(ANALYZE_KINDS)))
    p.add_argument("kind", choices=sorted(ANALYZE_KINDS), metavar="KIND")
    p.add_argument("wav", metavar="WAV")
    p.add_argument("--expected", metavar="JSON",
                   help="onset/velocity: a hit count, [t, …] seconds, or "
                        "[{\"t\": 0.5, \"vel\": 96}, …]; @file reads it from JSON")
    p.add_argument("--ref", metavar="FILE", help="psd: reference WAV to cross-correlate against")
    p.add_argument("--opt", action="append", metavar="K=V",
                   help="extra analysis option, e.g. --opt freq=997 --opt channel=1")

    p = add("jam", cmd_jam, "live_jam without a board: the set, a re-analysis, the kits, "
                            "a synthetic run")
    s = p.add_subparsers(dest="jam_cmd", required=True)
    q = s.add_parser("script", parents=[sub_common], help="the set as a table (deterministic)")
    q.add_argument("--seed", type=int, default=1)
    q.add_argument("--minutes", type=float, default=12.5)
    q.add_argument("--bpm", type=int, default=120)
    q.add_argument("--soak-min", dest="soak_min", type=float, default=0.0)
    q.add_argument("--sections", default="", help="comma list; default all")
    q = s.add_parser("analyze", parents=[sub_common], help="re-analyse a run directory")
    q.add_argument("run_dir", metavar="RUN_DIR")
    q.add_argument("--strict", action="store_true")
    q.add_argument("--late-ms", dest="late_ms", type=float, default=10.0)
    q.add_argument("--snippets", type=int, default=40)
    q.add_argument("--allow-partial", dest="allow_partial", action="store_true")
    q = s.add_parser("kits", parents=[sub_common], help="write the four jam kits")
    q.add_argument("out", metavar="DIR")
    q = s.add_parser("synth", parents=[sub_common],
                     help="a synthetic run (no board), optionally with faults, analysed")
    q.add_argument("out", metavar="DIR")
    q.add_argument("--seed", type=int, default=1)
    q.add_argument("--fault", action="append", metavar="KIND@S",
                   help="hole|click|pilot|slip, at S seconds of the take (repeatable)")

    p = add("flash", cmd_flash, "OTA-flash FW over CDC (switches out of UAC2 mode if needed)")
    p.add_argument("firmware", nargs="?", default="build/CrossPad.bin", metavar="FW")
    g = p.add_mutually_exclusive_group()
    g.add_argument("--ota", action="store_true", help="OTA over CDC (default)")
    g.add_argument("--uart", action="store_true", help="UART/esptool (not in P0: use idf.py flash)")
    p.add_argument("--delta", action="store_true", help="send a delta patch (needs --base-fw)")
    p.add_argument("--base-fw", dest="base_fw", metavar="F", help="firmware currently on device")
    p.add_argument("--wait-boot", dest="wait_boot", action="store_true", default=True)
    p.add_argument("--no-wait-boot", dest="wait_boot", action="store_false")

    p = add("bootloader", cmd_bootloader, "request ROM bootloader (--esp, default) or STM DFU")
    g = p.add_mutually_exclusive_group()
    g.add_argument("--esp", action="store_true")
    g.add_argument("--stm", action="store_true")
    p.add_argument("--method", default="cdc,midi", help="order of transports to try")
    p.add_argument("--timeout", type=float, default=10.0, help="seconds to wait for the port")

    p = add("run", cmd_run, "run a HIL scenario: " + ", ".join(scenario_names()))
    rs = p.add_subparsers(dest="scenario", required=True, metavar="SCENARIO")
    for name in scenario_names():
        s_obj = get_scenario(name)
        q = rs.add_parser(name, help=s_obj.description, parents=[sub_common])
        scen.params_to_argparse(s_obj.Params, q)
        q.add_argument("--logdir", default="hil_logs", help="where <name>_<ts>/ is created")

    p = add("bench", cmd_bench, "bench lease: claim / renew / release the board, status, md")
    s = p.add_subparsers(dest="bench_cmd", required=True)
    q = s.add_parser("claim", parents=[sub_common], help="take the board or join the queue "
                                                        "(exit 1 when queued)")
    q.add_argument("purpose", nargs="?", default="", help="what you are about to do")
    q.add_argument("--ttl", type=float, metavar="MIN",
                   help="minutes before an unrenewed lease expires (default 30)")
    q = s.add_parser("renew", parents=[sub_common])
    q.add_argument("--ttl", type=float, metavar="MIN")
    q = s.add_parser("release", parents=[sub_common], help="give the board back")
    q.add_argument("firmware_left", nargs="?", default=None, metavar="FIRMWARE_LEFT",
                   help="what is on the board now (repo, branch, commit)")
    q = s.add_parser("status", parents=[sub_common])
    q.add_argument("--history", type=int, default=10, metavar="N")
    q = s.add_parser("md", parents=[sub_common], help="status as markdown (BENCH.md's block)")
    q.add_argument("--history", type=int, default=8, metavar="N")
    q.add_argument("--write", metavar="FILE",
                   help="replace the generated block in FILE (e.g. hil_logs/BENCH.md)")

    p = add("record", cmd_record, "record a CDC transcript (stdin commands) for replay tests")
    p.add_argument("--out", default="transcript.ndjson", metavar="FILE")

    p = add("serve", cmd_serve, "NDJSON daemon on stdio (used by crosspad-mcp)")
    p.add_argument("--client", default=None, metavar="NAME",
                   help="the name other clients of the hub see while this one holds the port")
    return ap


# --------------------------------------------------------------------------- main

def _print_error(deps: CliDeps, as_json: bool, err: HilError) -> None:
    if as_json:
        deps.out().write(json.dumps({"ok": False, "error": err.to_dict()}) + "\n")
        deps.out().flush()
        return
    e = deps.err()
    e.write(f"error: {err.code}: {err.message}\n")
    if err.hint:
        e.write(f"hint: {err.hint}\n")
    if err.details.get("candidates"):
        for c in err.details["candidates"]:
            e.write(f"  candidate: {c.get('id')} usb_mode={c.get('usb_mode')}\n")
    e.flush()


def _install_stack_dump_signal() -> None:
    """``kill -USR1 <pid>`` prints every thread's Python stack to stderr.

    A multi-hour run that stops making progress with every thread parked on a
    futex cannot be diagnosed after the fact (ptrace is off for non-children on
    stock Ubuntu, so py-spy needs root). This costs nothing and turns "it hung
    overnight" into a traceback.
    """
    if not hasattr(signal, "SIGUSR1"):
        return
    try:
        faulthandler.register(signal.SIGUSR1, all_threads=True, chain=False)
    except (RuntimeError, ValueError):  # not the main thread / already registered
        pass


def _sigterm_as_interrupt(signum: int, frame: object) -> None:
    raise KeyboardInterrupt(f"signal {signum}")


def _install_sigterm_handler() -> None:
    """``kill <pid>`` runs the scenario's ``finally`` blocks, like Ctrl-C does.

    By default SIGTERM ends Python on the spot: an ``audio_loopback`` killed
    that way left IN2 at +24 dB in the board's NVS and the board in the audio
    profile. (``kill -9`` still cannot be caught; ``pins`` covers that.)
    """
    if threading.current_thread() is not threading.main_thread():
        return
    try:
        signal.signal(signal.SIGTERM, _sigterm_as_interrupt)
    except (ValueError, OSError):
        pass


def _apply_global_flags(ns: argparse.Namespace) -> None:
    """Flags that act through the environment, so child processes inherit them."""
    if getattr(ns, "no_hub", False):
        hub_link.set_no_hub(True)
    client = getattr(ns, "client", None)
    if client:
        os.environ[hub_link.CLIENT_ENV] = str(client)


def main(argv: list[str] | None = None, *, deps: CliDeps | None = None) -> int:
    deps = deps or CliDeps()
    _install_stack_dump_signal()
    _install_sigterm_handler()
    parser = build_parser()
    args = list(sys.argv[1:] if argv is None else argv)
    # argparse writes to the real sys.std* — redirect so tests can capture --help / usage.
    saved = (sys.stdout, sys.stderr)
    sys.stdout, sys.stderr = deps.out(), deps.err()
    try:
        ns = parser.parse_args(args)
    except SystemExit as e:
        return int(e.code or 0) if isinstance(e.code, int) else 2
    finally:
        sys.stdout, sys.stderr = saved
    _apply_global_flags(ns)
    as_json = bool(getattr(ns, "as_json", False))
    try:
        res: CmdResult = ns.func(deps, ns)
    except HilError as e:
        _print_error(deps, as_json, e)
        return 2
    except KeyboardInterrupt:
        deps.err().write("interrupted\n")
        return 130
    if ns.command == "serve":
        return res.exit_code
    out = deps.out()
    if as_json:
        out.write(json.dumps(res.data, default=str) + "\n")
    elif res.text:
        out.write(res.text + "\n")
    out.flush()
    return res.exit_code


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
