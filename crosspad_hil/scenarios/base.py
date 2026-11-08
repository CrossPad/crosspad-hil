"""Scenario framework: the protocol every `crosspad-hil run <name>` implements.

A scenario is a module-level object with `name`, `Params` (a dataclass), `description`
and `run(ctx, params, progress) -> Report`. The CLI derives argparse flags from `Params`
(`params_to_argparse`), the daemon derives the JSON schema (`params_schema`), and both
call `run_scenario()`, which owns the work directory, `report.json` and the exit code.
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import threading
import time
import traceback
import types
import typing
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

from crosspad_hil import hub_link
from crosspad_hil.cdc import CdcLink
from crosspad_hil.console import Console
from crosspad_hil.devices import Device, UsbMode, discover, select
from crosspad_hil.errors import (
    AMBIGUOUS_DEVICE,
    BAD_ARGS,
    CANCELLED,
    ENV,
    NO_CDC_IN_AUDIO_MODE,
    NO_DEVICE,
    PORT_BUSY,
    HilError,
)
from crosspad_hil.locks import PortLock
from crosspad_hil.serial_open import open_serial

# HilError codes that mean "the bench is wrong", not "the firmware failed": exit 2.
ENV_EXIT_CODES: frozenset[str] = frozenset(
    {ENV, NO_DEVICE, AMBIGUOUS_DEVICE, PORT_BUSY, NO_CDC_IN_AUDIO_MODE}
)


@dataclass
class Artifact:
    path: str
    mime: str
    role: str


@dataclass
class Report:
    passed: bool
    summary: str
    data: dict
    artifacts: list[Artifact]
    exit_code: int  # 0 pass, 1 fail, 2 env

    def to_dict(self) -> dict:
        return {
            "passed": self.passed,
            "summary": self.summary,
            "data": self.data,
            "artifacts": [dataclasses.asdict(a) for a in self.artifacts],
            "exit_code": self.exit_code,
        }


class Progress:
    """Progress sink. Default: no-op. The CLI prints; the daemon emits task.progress."""

    def __call__(self, progress: int, total: int | None, message: str) -> None:
        return None


@dataclass
class Context:
    device: Device
    workdir: Path
    cancelled: threading.Event
    log: Callable[[str], None]
    serial_factory: Callable[..., Any] = open_serial
    # The bench-lease holder this run is for (the caller's session name), or None.
    holder: str | None = None
    # Sessions this context handed out, so a scenario (or its caller) can close
    # everything it opened without tracking each one itself.
    opened: list[Any] = dataclasses.field(default_factory=list)
    # What the scenario has collected so far. A scenario that keeps a data dict
    # assigns it here once; run_scenario merges it into the report when the run
    # dies mid-way, so twelve minutes of bench time do not reduce to one error
    # line. Assigned by reference — later mutations are seen.
    partial: dict[str, Any] = dataclasses.field(default_factory=dict)
    # Set when this context opened the console, which resets the board.
    _console_opened: bool = False
    # Set once a CDC link of this context has seen the firmware answer.
    _cdc_readied: bool = False

    def close_all(self) -> None:
        """Close every session this context opened (idempotent, never raises)."""
        while self.opened:
            item = self.opened.pop()
            try:
                item.close()
            except Exception:  # noqa: BLE001 - teardown must not mask the real failure
                pass

    def _resolve_cdc_port(self) -> str | None:
        """Where this device's CDC node lives now.

        A reset can move the board from /dev/ttyACM1 to /dev/ttyACM2, so the
        path captured at discovery time goes stale mid-run. The device id is
        stable, so look it up again.
        """
        try:
            from crosspad_hil.devices import discover, select

            fresh = select(discover(), self.device.id)
        except HilError:
            return None
        self.device = fresh
        return fresh.ports.cdc.path if fresh.ports.cdc else None

    def open_console(self, reset: bool = False) -> Console:
        port = self.device.ports.console
        if port is None:
            raise HilError(
                NO_DEVICE,
                f"{self.device.id} has no STM VCP console port",
                hint="the STM32 bridge (0x0483:0x5740) carries the ESP console; is it enumerated?",
                device=self.device.id,
            )
        self.workdir.mkdir(parents=True, exist_ok=True)
        con = Console(
            port.path,
            log_path=self.workdir / "console.log",
            serial_factory=self.serial_factory,
        )
        con.open(reset=reset)
        self.opened.append(con)
        self._console_opened = True
        if not reset:
            # Opening the STM bridge VCP resets the ESP on this hardware even
            # with DTR/RTS deasserted. A scenario must not count the reboot it
            # caused itself, so ride it out before handing the console over.
            # The banner can arrive before we start watching, so look at what
            # the parser already saw rather than waiting for an event.
            time.sleep(1.5)
            if con.snapshot().get("reboots"):
                con.wait_boot(timeout_s=45.0, settle_s=1.0)
                con.forget_boot_history()
        return con

    def open_cdc(self) -> CdcLink | hub_link.HubLink:
        """Direct inside the run's exclusive op (every scenario that does not set
        ``exclusive = False``); through the hub otherwise."""
        port = self.device.ports.cdc
        if port is None and hub_link.direct():
            if self.device.usb_mode == UsbMode.AUDIO:
                raise HilError(
                    NO_CDC_IN_AUDIO_MODE,
                    f"{self.device.id} is in MIDI+UAC2 profile; CDC endpoint absent",
                    hint="usb-mode set default (SysEx 0x1B on the ESP MIDI port) then retry",
                    device=self.device.id,
                )
            raise HilError(
                NO_DEVICE,
                f"{self.device.id} has no ESP CDC port",
                hint="expected VID 0x303A PID 0x3456; is the device in bootloader/DFU?",
                device=self.device.id,
            )
        link = hub_link.open_link(self.device, purpose="scenario",
                                  serial_factory=self.serial_factory,
                                  port_resolver=self._resolve_cdc_port)
        self.opened.append(link)
        first_real = not self._cdc_readied and self.serial_factory is open_serial
        if self._console_opened or first_real:
            # Opening the console resets the ESP on this hardware, so the
            # firmware is still booting when the CDC node comes back. The
            # first link of a run on real hardware waits too: a scenario
            # started seconds after `smoke` (which resets the board) found the
            # node up and the firmware not yet answering, and its first MEM
            # timed out. Injected serial factories (tests) skip the probe.
            link.wait_ready()
            self._console_opened = False
            self._cdc_readied = True
        return link

    def check_cancelled(self) -> None:
        if self.cancelled.is_set():
            raise HilError(CANCELLED, "scenario cancelled", hint="task.cancel was requested")


@runtime_checkable
class Scenario(Protocol):
    name: str
    Params: type
    description: str

    def run(self, ctx: Context, params: Any, progress: Progress) -> Report: ...


# The one explicit registry the package allows. Populated by `register()` calls at
# import time of each scenario module (see scenarios/__init__.py load_builtin()).
_REGISTRY: dict[str, Scenario] = {}


def register(s: Scenario) -> Scenario:
    _REGISTRY[s.name] = s
    return s


def get(name: str) -> Scenario:
    try:
        return _REGISTRY[name]
    except KeyError:
        raise HilError(
            BAD_ARGS,
            f"unknown scenario {name!r}",
            hint="known scenarios: " + ", ".join(names()),
            scenario=name,
        ) from None


def names() -> list[str]:
    return sorted(_REGISTRY)


# ── dataclass ↔ argparse ─────────────────────────────────────────────────────


def _unwrap_optional(tp: Any) -> tuple[Any, bool]:
    """`X | None` / `Optional[X]` → (X, True); anything else → (tp, False)."""
    origin = typing.get_origin(tp)
    if origin is typing.Union or origin is types.UnionType:
        args = [a for a in typing.get_args(tp) if a is not type(None)]
        if len(args) == 1 and len(typing.get_args(tp)) == 2:
            return args[0], True
    return tp, False


def _type_name(tp: Any) -> str:
    inner, optional = _unwrap_optional(tp)
    origin = typing.get_origin(inner)
    if origin is list:
        (item,) = typing.get_args(inner)
        base = f"list[{item.__name__}]"
    else:
        base = getattr(inner, "__name__", str(inner))
    return f"{base} | None" if optional else base


def _list_parser(item: type) -> Callable[[str], list]:
    def parse(text: str) -> list:
        text = text.strip()
        if not text:
            return []
        return [item(part.strip()) for part in text.split(",")]

    parse.__name__ = f"list[{item.__name__}]"
    return parse


def _field_default(f: dataclasses.Field) -> Any:
    if f.default is not dataclasses.MISSING:
        return f.default
    if f.default_factory is not dataclasses.MISSING:  # type: ignore[misc]
        return f.default_factory()  # type: ignore[misc]
    return None


def params_to_argparse(params_cls: type, parser: argparse.ArgumentParser) -> None:
    """Add one option per dataclass field.

    Rules: `bool` → `--x` / `--no-x`; `int|float|str` → typed option; `X | None` → same
    with default None; `list[int]` / `list[str]` → comma-separated; help from
    `metadata["help"]`, default appended.
    """
    if not dataclasses.is_dataclass(params_cls):
        raise TypeError(f"{params_cls!r} is not a dataclass")
    hints = typing.get_type_hints(params_cls)
    for f in dataclasses.fields(params_cls):
        tp = hints[f.name]
        inner, _optional = _unwrap_optional(tp)
        flag = "--" + f.name.replace("_", "-")
        default = _field_default(f)
        help_text = str(f.metadata.get("help", ""))
        suffix = f"(default: {default})" if default is not None else "(default: none)"
        full_help = f"{help_text} {suffix}".strip()
        if inner is bool:
            group = parser.add_mutually_exclusive_group()
            group.add_argument(flag, dest=f.name, action="store_true", default=default,
                               help=full_help)
            group.add_argument("--no-" + f.name.replace("_", "-"), dest=f.name,
                               action="store_false", help=f"negate {flag}")
            continue
        origin = typing.get_origin(inner)
        if origin is list:
            (item,) = typing.get_args(inner)
            if item not in (int, str, float):
                raise TypeError(f"field {f.name}: unsupported list item type {item!r}")
            parser.add_argument(flag, dest=f.name, type=_list_parser(item), default=default,
                                metavar="A,B,...", help=full_help)
            continue
        if inner in (int, float, str):
            parser.add_argument(flag, dest=f.name, type=inner, default=default,
                                metavar=inner.__name__.upper(), help=full_help)
            continue
        raise TypeError(f"field {f.name}: unsupported annotation {tp!r}")


def argparse_to_params(params_cls: type, ns: argparse.Namespace) -> Any:
    kwargs = {f.name: getattr(ns, f.name) for f in dataclasses.fields(params_cls)}
    return params_cls(**kwargs)


def params_from_dict(params_cls: type, data: dict[str, Any]) -> Any:
    """Build a Params from JSON (the daemon's scenario.run args).

    Unknown keys and values of the wrong type are BAD_ARGS rather than a
    TypeError from the dataclass, so the caller gets a usable message.
    """
    fields = {f.name: f for f in dataclasses.fields(params_cls)}
    unknown = sorted(set(data) - set(fields))
    if unknown:
        raise HilError(
            BAD_ARGS,
            f"unknown parameter(s): {', '.join(unknown)}",
            hint=f"known: {', '.join(sorted(fields))}",
            unknown=unknown,
            known=sorted(fields),
        )
    hints = typing.get_type_hints(params_cls)
    kwargs: dict[str, Any] = {}
    for name, value in data.items():
        expected = _type_name(hints[name])
        try:
            kwargs[name] = _coerce(value, expected)
        except (TypeError, ValueError) as exc:
            raise HilError(
                BAD_ARGS,
                f"{name}: expected {expected}, got {value!r}",
                hint=str(exc) or None,
                param=name,
                expected=expected,
            ) from None
    return params_cls(**kwargs)


def _coerce(value: Any, expected: str) -> Any:
    """JSON value -> the type the dataclass field declares."""
    if value is None:
        return None
    if expected.startswith("list"):
        if isinstance(value, str):
            value = [v for v in value.split(",") if v != ""]
        if not isinstance(value, list):
            raise TypeError("expected a list or a comma-separated string")
        item = "int" if "int" in expected else "str"
        return [int(v) for v in value] if item == "int" else [str(v) for v in value]
    if expected == "bool":
        if isinstance(value, bool):
            return value
        raise TypeError("expected true or false")
    if expected == "int":
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise TypeError("expected a number")
        return int(value)
    if expected == "float":
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise TypeError("expected a number")
        return float(value)
    return str(value)


def params_schema(params_cls: type) -> list[dict]:
    """[{name, type, default, help}] — what the daemon's scenario.list reports."""
    hints = typing.get_type_hints(params_cls)
    return [
        {
            "name": f.name,
            "type": _type_name(hints[f.name]),
            "default": _field_default(f),
            "help": str(f.metadata.get("help", "")),
        }
        for f in dataclasses.fields(params_cls)
    ]


# ── runner ───────────────────────────────────────────────────────────────────


def _params_dict(params: Any) -> dict:
    if dataclasses.is_dataclass(params) and not isinstance(params, type):
        return dataclasses.asdict(params)
    if isinstance(params, dict):
        return dict(params)
    return {}


def run_scenario(
    name: str,
    params: Any,
    *,
    device: Device | None = None,
    workdir: Path | None = None,
    progress: Progress | None = None,
    cancelled: threading.Event | None = None,
    log: Callable[[str], None] = print,
    holder: str | None = None,
) -> Report:
    """Run one registered scenario, write `<workdir>/report.json`, return its Report.

    Exit codes: 0 pass, 1 fail (including CANCELLED and any non-environment error),
    2 for HilError codes in ENV_EXIT_CODES. A crash inside the scenario is reported,
    never propagated — report.json must exist for every run that started.
    """
    scenario = get(name)
    if device is None:
        device = select(discover(), None)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    if workdir is None:
        workdir = (Path("hil_logs") / f"{name}_{stamp}").resolve()
    workdir.mkdir(parents=True, exist_ok=True)
    ctx = Context(
        device=device,
        workdir=workdir,
        cancelled=cancelled if cancelled is not None else threading.Event(),
        log=log,
        holder=holder,
    )
    sink = progress if progress is not None else Progress()
    started = time.time()
    log(f"[{name}] device={device.id} workdir={workdir}")
    # One scenario at a time per board. Port locks are not enough: a scenario
    # that switches the USB profile changes what ports the board even has, so
    # two runs can collide while sharing no port at that instant. That is not a
    # theoretical race — a scenario started while another still held the board
    # restored the profile underneath it and then failed looking for a MIDI
    # port that was mid-re-enumeration, reporting a bench error for a board
    # that was fine.
    device_lock = PortLock(f"device:{device.id}", f"scenario {name}")
    # The hub steps aside for the whole run unless the scenario says it only
    # sends verbs (``exclusive = False``): one that resets the board or switches
    # its USB profile must not have other clients' verbs interleaved.
    hold: hub_link.ExclusiveHold | None = None
    try:
        # Inside the try on purpose: PORT_BUSY is an ENV code, so a board that
        # is already busy reports exit 2 with a report.json like any other
        # bench problem, rather than a different shape of failure.
        device_lock.acquire()
        if getattr(scenario, "exclusive", True):
            hold = hub_link.hold_exclusive(f"scenario {name}")
        report = scenario.run(ctx, params, sink)
        if not report.passed and report.exit_code == 0:
            report.exit_code = 1
    except HilError as e:
        if e.code == CANCELLED:
            report = Report(passed=False, summary=f"cancelled: {e.message}",
                            data={**ctx.partial, "cancelled": True, "error": e.to_dict()},
                            artifacts=[], exit_code=1)
        elif e.code in ENV_EXIT_CODES:
            report = Report(passed=False, summary=f"environment: {e.code}: {e.message}",
                            data={**ctx.partial, "error": e.to_dict()}, artifacts=[], exit_code=2)
        else:
            report = Report(passed=False, summary=f"{e.code}: {e.message}",
                            data={**ctx.partial, "error": e.to_dict()}, artifacts=[], exit_code=1)
        log(f"[{name}] {report.summary}")
    except Exception as exc:  # noqa: BLE001 — a scenario crash must still produce a report
        report = Report(
            passed=False,
            summary=f"crashed: {exc!r}",
            data={**ctx.partial,
                  "error": {"code": "INTERNAL", "message": repr(exc),
                            "hint": None, "details": {},
                            "traceback": traceback.format_exc()}},
            artifacts=[],
            exit_code=1,
        )
        log(f"[{name}] {report.summary}")
    finally:
        # Whatever the scenario opened stays locked until it is closed, and the
        # next run on the same board would fail with PORT_BUSY.
        ctx.close_all()
        if hold is not None:
            hold.release()
        device_lock.release()
    finished = time.time()

    report_path = workdir / "report.json"
    report_artifact = Artifact(path=str(report_path), mime="application/json", role="report")
    if not any(a.role == "report" for a in report.artifacts):
        report.artifacts.append(report_artifact)
    doc = {
        "scenario": name,
        "device": device.id,
        "passed": report.passed,
        "summary": report.summary,
        "exit_code": report.exit_code,
        "data": report.data,
        "artifacts": [dataclasses.asdict(a) for a in report.artifacts],
        "params": _params_dict(params),
        "workdir": str(workdir),
        "started": started,
        "finished": finished,
        "seconds": round(finished - started, 3),
    }
    report_path.write_text(json.dumps(doc, indent=2, default=str) + "\n", encoding="utf-8")
    log(f"[{name}] {'PASS' if report.passed else 'FAIL'} exit={report.exit_code} "
        f"report={report_path}")
    return report
