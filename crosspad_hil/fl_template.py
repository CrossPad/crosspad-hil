"""An FL Studio project template whose mixer inserts already record the CrossPad.

FL Studio's MIDI-scripting API cannot select a mixer track's audio input
(there is no `mixer.setTrackInput`, and the REC event range that would carry
it has none; probing it with `getEventIDName` crashes FL), and under Wine the
selector is a mouse-only popup.  What FL *does* persist per project is the
input, as the insert's `Input` event in the `.flp` stream.  So the input is set
where FL keeps it: a project template, built from FL's own `Empty.flp` with the
first inserts named and pointed at the CrossPad stereo pair, installed where
FL lists user templates (File > New from template).

`.flp` is the `FLhd`/`FLdt` container `fpc.py` already reads.  A mixer insert
is a run of events ending in `Input` (DWORD+26 = 154, `-1` = none) and
`Output` (DWORD+19 = 147); an optional `Name` (TEXT+12 = 204) opens the run and
a `Color` (DWORD+21 = 149) closes it.  Insert 0 is the master.  Everything
else in the file is carried through verbatim.
"""
from __future__ import annotations

import glob
import os
import struct
from dataclasses import dataclass
from pathlib import Path

from crosspad_hil.errors import BAD_ARGS, HilError
from crosspad_hil.fpc import _encode_event, _events, _utf16z

EV_INPUT = 154          # DWORD + 26, i32: index into the driver's input list, -1 = none
EV_OUTPUT = 147         # DWORD + 19
EV_NAME = 204           # TEXT + 12, UTF-16 zero-terminated
EV_COLOR = 149          # DWORD + 21, 0x00BBGGRR
EV_SLOT_HEAD = 236      # DATA + 44: opens an insert's slot list

TEMPLATE_NAME = "CrossPad"
TEMPLATE_CATEGORY = "CrossPad"
CROSSPAD_INPUT = 0      # the driver's first stereo pair: "In 1 - In 2" = [Default input]
CROSSPAD_COLOR = 0x002E8AE8   # 0x00BBGGRR: the pad orange, RGB e8 8a 2e
TEMPLATE_DIR = Path("Documents/Image-Line/FL Studio/Projects/Templates")
EMPTY_REL = Path("Data/Templates/Empty/Empty.flp")


@dataclass
class InsertSpec:
    index: int                 # 1-based mixer insert
    name: str
    input: int
    color: int | None = None


def _split(data: bytes) -> tuple[bytes, bytes, bytes]:
    """`(head chunk incl. its 8-byte header, FLdt body, trailing bytes)`."""
    if data[:4] != b"FLhd":
        raise HilError(BAD_ARGS, "not an FLP file: missing FLhd")
    hl = struct.unpack_from("<I", data, 4)[0]
    off = 8 + hl
    if data[off:off + 4] != b"FLdt":
        raise HilError(BAD_ARGS, "not an FLP file: missing FLdt")
    dl = struct.unpack_from("<I", data, off + 4)[0]
    return data[:off], data[off + 8:off + 8 + dl], data[off + 8 + dl:]


def _join(head: bytes, body: bytes, tail: bytes) -> bytes:
    return head + b"FLdt" + struct.pack("<I", len(body)) + body + tail


def insert_inputs(data: bytes) -> list[int]:
    """Each insert's input, master first — what the file says, for tests and `status`."""
    _, body, _ = _split(data)
    return [struct.unpack("<i", payload)[0] for ev, payload in _events(body) if ev == EV_INPUT]


def build_template(empty: bytes, inserts: list[InsertSpec]) -> bytes:
    """`Empty.flp` with the given inserts named and pointed at their inputs.

    Insert runs are located by their `Input` events: the n-th one belongs to
    insert n (0 = master).  The name goes in front of the run's slot list, the
    colour after `Output` — where FL itself writes them.
    """
    head, body, tail = _split(empty)
    events = _events(body)
    by_index = {spec.index: spec for spec in inserts}
    if any(i < 1 for i in by_index):
        raise HilError(BAD_ARGS, "insert 0 is the master; inserts start at 1")
    out = bytearray()
    run_start = 0                 # first event of the insert currently being scanned
    run_no = 0
    pending: list[tuple[int, bytes]] = []
    for ev, payload in events:
        if ev == EV_SLOT_HEAD and run_no in by_index and not pending:
            pending.append((EV_NAME, _utf16z(by_index[run_no].name)))
        if pending and ev == EV_SLOT_HEAD:
            for pev, ppayload in pending:
                out += _encode_event(pev, ppayload)
            pending = []
        if ev == EV_INPUT and run_no in by_index:
            payload = struct.pack("<i", by_index[run_no].input)
        out += _encode_event(ev, payload)
        if ev == EV_OUTPUT:
            spec = by_index.get(run_no)
            if spec is not None and spec.color is not None:
                out += _encode_event(EV_COLOR, struct.pack("<I", spec.color))
            run_no += 1
            run_start = len(out)
    missing = [i for i in by_index if i >= run_no]
    if missing:
        raise HilError(BAD_ARGS, f"the template has {run_no} inserts, none numbered {missing}")
    del run_start
    return _join(head, bytes(out), tail)


def crosspad_inserts(tracks: int, input_index: int = CROSSPAD_INPUT) -> list[InsertSpec]:
    if tracks < 1:
        raise HilError(BAD_ARGS, "at least one insert")
    if tracks == 1:
        return [InsertSpec(1, TEMPLATE_NAME, input_index, CROSSPAD_COLOR)]
    return [InsertSpec(i, f"{TEMPLATE_NAME} {i}", input_index, CROSSPAD_COLOR)
            for i in range(1, tracks + 1)]


# --------------------------------------------------------------------------- the Wine prefix

PREFIX_GLOBS = (
    "~/.var/app/com.usebottles.bottles/data/bottles/bottles/*/drive_c",   # Bottles (flatpak)
    "~/.local/share/bottles/bottles/*/drive_c",                           # Bottles (native)
    "~/.wine/drive_c",
)


def default_drive_c() -> Path | None:
    """The Wine prefix on this machine that holds an FL Studio install.

    A prefix whose FL has been started (it has user template folders) wins
    over one that merely has the program files.
    """
    fallback: Path | None = None
    for pattern in PREFIX_GLOBS:
        for hit in sorted(glob.glob(os.path.expanduser(pattern))):
            drive_c = Path(hit)
            if find_empty_flp(drive_c) is None:
                continue
            if user_template_dirs(drive_c):
                return drive_c
            fallback = fallback or drive_c
    return fallback


def find_empty_flp(drive_c: Path) -> Path | None:
    """FL's own empty template under `Program Files/Image-Line/FL Studio*`."""
    hits = sorted(drive_c.glob(f"Program Files/Image-Line/FL Studio*/{EMPTY_REL.as_posix()}"))
    return hits[-1] if hits else None


def user_template_dirs(drive_c: Path) -> list[Path]:
    """Every user profile's template folder that exists (Bottles keeps two)."""
    return sorted(p for p in drive_c.glob(f"users/*/{TEMPLATE_DIR.as_posix()}") if p.is_dir())


def install(drive_c: Path, *, tracks: int = 1, input_index: int = CROSSPAD_INPUT,
            empty: Path | None = None) -> dict:
    empty_path = empty or find_empty_flp(drive_c)
    if empty_path is None or not empty_path.is_file():
        raise HilError(BAD_ARGS, f"no FL Studio install under {drive_c}",
                       hint="pass --empty with FL's Data/Templates/Empty/Empty.flp")
    targets = user_template_dirs(drive_c)
    if not targets:
        raise HilError(BAD_ARGS, f"no Projects/Templates folder under {drive_c}/users",
                       hint="start FL Studio once so it creates its user data folders")
    data = build_template(empty_path.read_bytes(), crosspad_inserts(tracks, input_index))
    written = []
    for base in targets:
        dest = base / TEMPLATE_CATEGORY / f"{TEMPLATE_NAME}.flp"
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(data)
        written.append(str(dest))
    return {"empty": str(empty_path), "written": written, "tracks": tracks,
            "input": input_index, "bytes": len(data)}


def status(drive_c: Path) -> dict:
    """Where the template is installed and what each copy routes."""
    installed: list[str] = []
    inputs: dict[str, list[int]] = {}
    for base in user_template_dirs(drive_c):
        dest = base / TEMPLATE_CATEGORY / f"{TEMPLATE_NAME}.flp"
        if dest.is_file():
            installed.append(str(dest))
            inputs[str(dest)] = insert_inputs(dest.read_bytes())
    return {"drive_c": str(drive_c), "installed": installed, "inputs": inputs}
