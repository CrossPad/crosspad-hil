"""Crash evidence in one call: reset reason, panic registers, decoded backtrace, heap, context.

What this replaces is the manual ritual after a board dies: scroll the console back to the
``Guru Meditation`` banner, copy the ``Backtrace:`` line by hand, remember which build
directory the flashed firmware came from, source the IDF export script for
``xtensa-esp32s3-elf-addr2line``, and then still not have the heap numbers from the sample
before the crash. Every one of those steps is a place to get it wrong; this module does all
of them from one console log.

Three details are load-bearing and are the reason this is not a five-line addr2line wrapper:

* **The log names the ELF it came from.** ESP-IDF prints ``ELF file SHA256: <prefix>`` both at
  boot (``app_init``) and in the panic dump, and that prefix is the first bytes of
  ``sha256sum CrossPad.elf``. Decoding against the wrong build directory does not fail — it
  silently prints plausible, wrong source lines, which is worse than printing nothing. So the
  candidate ELFs (``build/``, ``build_v1/``, ``build_v2/``) are hashed and the one the device
  was actually running is chosen; a mismatch is reported rather than hidden.
* **The first backtrace frame is usually ROM.** ``0x40056f5d`` is ROM ``memcpy``; addr2line
  answers ``?? ??:0`` and that is the correct answer, not an error. The frame is kept with
  ``func: null`` so the numbering matches what the device printed.
* **A missing toolchain or ELF is an environment problem, not a crash.** Both raise
  ``HilError(ENV)`` naming what to install or build, because "diagnose the crash" failing with
  a traceback of its own is the least useful outcome available.

Reset reason, fatal patterns and the PerfMon heap block are the same rules
``crosspad_hil.parsers`` uses (knowledge/markers.yaml); the heap parser here is section-aware
(Internal SRAM / DMA-capable / PSRAM) because "which pool ran out" is the whole question when
an allocation failed.
"""
from __future__ import annotations

import hashlib
import os
import re
import subprocess
import time
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .errors import BAD_ARGS, ENV, HilError

# 300 lines of console around the panic (spec §3.2); most of it is what led up to the crash.
DEFAULT_CONTEXT_LINES = 300
CONTEXT_AFTER_FRACTION = 1 / 3

ADDR2LINE_EXE = "xtensa-esp32s3-elf-addr2line"
ADDR2LINE_TIMEOUT_S = 60.0
ELF_NAME = "CrossPad.elf"

# ── console patterns ───────────────────────────────────────────────────────────────────
# "Guru Meditation Error: Core  0 panic'ed (StoreProhibited). Exception was unhandled."
_GURU_RE = re.compile(r"Guru Meditation Error:\s*Core\s+(\d+)\s+panic'ed\s*\(([^)]*)\)")
# "abort() was called at PC 0x420... on core 0"
_ABORT_RE = re.compile(r"abort\(\) was called at PC (0x[0-9a-fA-F]+) on core (\d+)")
_ASSERT_RE = re.compile(r"assert failed:\s*(.+)$")
_WDT_RE = re.compile(r"Task watchdog got triggered|task_wdt:\s")
_IWDT_RE = re.compile(r"Interrupt wdt timeout on CPU(\d+)")
_CANARY_RE = re.compile(r"Stack canary watchpoint triggered \(([^)]*)\)")
_SMASH_RE = re.compile(r"Stack smashing protect failure")
_HEAP_CORRUPT_RE = re.compile(r"CORRUPT HEAP|heap corruption|Heap corruption")
_REG_RE = re.compile(r"\b(PC|EXCCAUSE|EXCVADDR|A1)\s*:\s*(0x[0-9a-fA-F]+)")
_BACKTRACE_RE = re.compile(r"^\s*Backtrace:\s*(.+)$")
_BT_PAIR_RE = re.compile(r"(0x[0-9a-fA-F]{8}):(0x[0-9a-fA-F]{8})")
_ELF_SHA_RE = re.compile(r"ELF file SHA256:\s*([0-9a-fA-F]+)")
_RESET_RE = re.compile(r"rst:(0x[0-9a-fA-F]+)\s*\(([^)]+)\)")
_REBOOT_MARK = "ESP-ROM:esp32s3"
_WDT_TASK_RE = re.compile(r"-\s*(\S+)\s*\(CPU\s*\d+\)")

# PerfMon heap block: "=== Heap Statistics ===", then one section header per pool.
_HEAP_START = "Heap Statistics"
_HEAP_SECTION_RE = re.compile(r"PerfMon:\s*(Internal SRAM|DMA-capable|PSRAM):\s*$")
_HEAP_FIELD_RE = re.compile(
    r"PerfMon:\s+(Total|Used|Free|Largest free block|Min free ever|"
    r"Allocated blocks|Free blocks|Total blocks):\s+(\d+)"
)
_HEAP_FIELDS = {
    "Total": "total",
    "Used": "used",
    "Free": "free",
    "Largest free block": "largest_free",
    "Min free ever": "min_free",
    "Allocated blocks": "allocated_blocks",
    "Free blocks": "free_blocks",
    "Total blocks": "total_blocks",
}
_HEAP_SECTIONS = {"Internal SRAM": "internal", "DMA-capable": "dma", "PSRAM": "psram"}

# EXCCAUSE values that reach a CrossPad panic (Xtensa ISA table, the ones IDF names).
EXCCAUSE_NAMES: dict[int, str] = {
    0x00: "IllegalInstruction",
    0x01: "Syscall",
    0x02: "InstrFetchError",
    0x03: "LoadStoreError",
    0x05: "Alloca",
    0x06: "IntegerDivideByZero",
    0x08: "Privileged",
    0x09: "LoadStoreAlignment",
    0x1C: "LoadProhibited",
    0x1D: "StoreProhibited",
    0x1E: "InstrFetchProhibited",
    0x1F: "InstrFetchProhibited",
}

# ESP32-S3 address map, only what a data pointer may legitimately be (technical reference
# manual §"System and Memory"). Anything outside is a wild pointer, and saying so is most of
# the diagnosis for a Load/StoreProhibited.
_DATA_RANGES: tuple[tuple[int, int, str], ...] = (
    (0x3C000000, 0x3E000000, "PSRAM / flash-mapped data"),
    (0x3FC88000, 0x3FD00000, "internal SRAM (DRAM)"),
    (0x50000000, 0x50002000, "RTC fast memory"),
    (0x600FE000, 0x60100000, "RTC slow memory"),
)
# Frames in these directories are library plumbing; the interesting frame is the first one
# below them (the app's own code), which is what likely_cause names.
_LIBRARY_HINTS = ("managed_components/", "lvgl/", "esp_lvgl_port/", "freertos", "/components/esp")


@dataclass
class Frame:
    """One decoded backtrace entry. ``func``/``file`` are None for ROM or stripped code."""

    addr: str
    func: str | None = None
    file: str | None = None
    line: int | None = None
    inlined: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {"addr": self.addr, "func": self.func, "file": self.file, "line": self.line,
                "inlined": self.inlined}

    def where(self) -> str:
        if self.func is None:
            return f"{self.addr} (no symbol — ROM or stripped)"
        loc = f" at {self.file}:{self.line}" if self.file else ""
        return f"{self.func}{loc}"


@dataclass
class Panic:
    """The register-dump facts. ``cause`` is the name the panic handler printed."""

    core: int | None = None
    cause: str | None = None
    exccause: str | None = None
    pc: str | None = None
    excvaddr: str | None = None
    message: str | None = None
    tasks: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {"core": self.core, "cause": self.cause, "exccause": self.exccause,
                "pc": self.pc, "excvaddr": self.excvaddr, "message": self.message,
                "tasks": list(self.tasks)}


Runner = Callable[[Path, Path, Sequence[str]], str]


# ── toolchain and ELF discovery ────────────────────────────────────────────────────────

def _espressif_root() -> Path:
    return Path(os.environ.get("IDF_TOOLS_PATH") or (Path.home() / ".espressif"))


def find_addr2line(explicit: str | Path | None = None) -> Path:
    """Locate ``xtensa-esp32s3-elf-addr2line``: explicit, env, PATH, then the IDF tools tree.

    Raises ``HilError(ENV)`` naming what to install — the toolchain is not a dependency of
    this package and pretending otherwise would make the failure look like a bug here.
    """
    if explicit is not None:
        p = Path(explicit)
        if not p.is_file():
            raise HilError(ENV, f"addr2line not found at {p}",
                           hint=f"install the IDF toolchain or drop --addr2line to search "
                                f"PATH and {_espressif_root()}")
        return p
    env = os.environ.get("CROSSPAD_ADDR2LINE")
    if env and Path(env).is_file():
        return Path(env)
    import shutil

    which = shutil.which(ADDR2LINE_EXE)
    if which:
        return Path(which)
    root = _espressif_root() / "tools" / "xtensa-esp-elf"
    found = sorted(root.glob(f"*/xtensa-esp-elf/bin/{ADDR2LINE_EXE}"))
    if found:
        return found[-1]  # newest toolchain release (directory names sort by version+date)
    raise HilError(
        ENV,
        f"{ADDR2LINE_EXE} not found on PATH, in $CROSSPAD_ADDR2LINE, or under {root}",
        hint="source the IDF environment (. $IDF_PATH/export.sh) or install the toolchain "
             "(python $IDF_PATH/tools/idf_tools.py install xtensa-esp-elf)",
    )


def _idf_roots(root: str | Path | None = None) -> list[Path]:
    """Where a platform-idf checkout may live; same env names crosspad-mcp config.ts uses."""
    out: list[Path] = []
    if root is not None:
        out.append(Path(root))
    env = os.environ.get("CROSSPAD_IDF_ROOT")
    if env:
        out.append(Path(env))
    git_dir = Path(os.environ.get("CROSSPAD_GIT_DIR") or (Path.home() / "GIT"))
    out.append(git_dir / "platform-idf")
    out.append(Path.cwd())
    seen: set[Path] = set()
    uniq: list[Path] = []
    for p in out:
        rp = p.expanduser()
        if rp not in seen:
            seen.add(rp)
            uniq.append(rp)
    return uniq


def elf_candidates(root: str | Path | None = None) -> list[Path]:
    """Every ``build*/CrossPad.elf`` under the known roots, newest first."""
    found: list[Path] = []
    for r in _idf_roots(root):
        if not r.is_dir():
            continue
        for build in sorted(r.glob("build*")):
            elf = build / ELF_NAME
            if elf.is_file() and elf not in found:
                found.append(elf)
    found.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return found


def elf_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def find_elf(explicit: str | Path | None = None, sha_prefix: str | None = None,
             root: str | Path | None = None) -> tuple[Path, str, bool | None]:
    """Return ``(elf, sha256, matches_log)`` for the build the device was running.

    ``matches_log`` is None when the log carried no ELF SHA to compare against. An explicit
    ELF is used even when it does not match — the caller asked for it — but the mismatch is
    still reported, because wrong line numbers look exactly like right ones.
    """
    if explicit is not None:
        p = Path(explicit)
        if not p.is_file():
            raise HilError(ENV, f"ELF not found: {p}",
                           hint="idf.py build (per-rev: idf.py -B build_v2 -DSDKCONFIG="
                                "sdkconfig.v2 build)")
        sha = elf_sha256(p)
        return p, sha, (sha.startswith(sha_prefix.lower()) if sha_prefix else None)
    candidates = elf_candidates(root)
    if not candidates:
        roots = ", ".join(str(r) for r in _idf_roots(root))
        raise HilError(ENV, f"no {ELF_NAME} found under any build dir of: {roots}",
                       hint="idf.py build, or pass --elf / set CROSSPAD_IDF_ROOT")
    hashed = [(c, elf_sha256(c)) for c in candidates]
    if sha_prefix:
        want = sha_prefix.lower().rstrip(".")
        for cand, sha in hashed:
            if sha.startswith(want):
                return cand, sha, True
        return hashed[0][0], hashed[0][1], False
    return hashed[0][0], hashed[0][1], None


# ── addr2line ──────────────────────────────────────────────────────────────────────────

def _run_addr2line(exe: Path, elf: Path, addrs: Sequence[str]) -> str:
    cmd = [str(exe), "-pfiaC", "-e", str(elf), *addrs]
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=ADDR2LINE_TIMEOUT_S)
    except OSError as e:
        raise HilError(ENV, f"could not run {exe}: {e}",
                       hint="is the IDF toolchain installed for this architecture?") from e
    except subprocess.TimeoutExpired as e:
        raise HilError(ENV, f"{exe.name} timed out after {ADDR2LINE_TIMEOUT_S:.0f}s",
                       hint="is the ELF on a slow or disconnected filesystem?") from e
    if proc.returncode != 0:
        raise HilError(ENV, f"{exe.name} failed (rc={proc.returncode}): "
                            f"{(proc.stderr or '').strip()[:200]}",
                       hint="does the ELF match the chip (xtensa-esp32s3)?")
    return proc.stdout


_A2L_HEAD_RE = re.compile(r"^(0x[0-9a-fA-F]+):\s*(.*)$")
_A2L_INLINE_RE = re.compile(r"^\s*\(inlined by\)\s*(.*)$")
_A2L_BODY_RE = re.compile(r"^(.*?)\s+at\s+(.*?):(\d+|\?+)(?:\s*\(discriminator \d+\))?\s*$")


# Where a CrossPad source tree starts, for an ELF built somewhere other than this checkout
# (a downloaded artifact, a colleague's build): the path is cut at the leftmost of these.
_PROJECT_MARKERS = ("/managed_components/", "/components/", "/main/", "/tools/")


def _shorten(path: str, elf: Path) -> str:
    """Trim absolute build paths to something a human can read in a report.

    Relative to the project root when the ELF sits in it, and otherwise cut at the first
    CrossPad directory in the path — the ELF is often not where it was built. Then
    ``managed_components/ns__name/`` → ``name/`` so an LVGL frame reads
    ``lvgl/src/misc/lv_event.c`` instead of forty columns of checkout path.
    """
    p = path.replace("\\", "/")
    root = str(elf.parent.parent).replace("\\", "/").rstrip("/")
    if root and p.startswith(root + "/"):
        p = p[len(root) + 1:]
    elif p.startswith("/") or (len(p) > 2 and p[1] == ":"):
        cuts = [i for i in (p.find(m) for m in _PROJECT_MARKERS) if i != -1]
        if cuts:
            p = p[min(cuts) + 1:]
    m = re.match(r"^managed_components/(?:[^/]+__)?([^/]+)/(.*)$", p)
    if m:
        p = f"{m.group(1)}/{m.group(2)}"
    return p


def _parse_addr2line(text: str) -> list[Frame]:
    frames: list[Frame] = []
    current: str | None = None
    for raw in text.splitlines():
        line = raw.rstrip()
        if not line:
            continue
        head = _A2L_HEAD_RE.match(line)
        inline = None if head else _A2L_INLINE_RE.match(line)
        if head:
            current = head.group(1)
            body, is_inline = head.group(2), False
        elif inline and current is not None:
            body, is_inline = inline.group(1), True
        else:
            continue
        frames.append(_frame_from_body(current, body, is_inline))
    return frames


def _frame_from_body(addr: str, body: str, inlined: bool) -> Frame:
    m = _A2L_BODY_RE.match(body.strip())
    if not m:
        return Frame(addr=addr, inlined=inlined)
    func, file, line = m.group(1).strip(), m.group(2).strip(), m.group(3)
    if func in ("??", ""):
        func = ""
    if file in ("??", ""):
        file = ""
    return Frame(
        addr=addr,
        func=func or None,
        file=file or None,
        line=int(line) if line.isdigit() and line != "0" else None,
        inlined=inlined,
    )


def decode_backtrace(addrs: Sequence[str], elf: Path, addr2line: Path,
                     runner: Runner = _run_addr2line) -> list[Frame]:
    """Decode PC addresses to ``Frame``s; ``runner`` is injectable so tests need no toolchain."""
    if not addrs:
        return []
    frames = _parse_addr2line(runner(addr2line, elf, list(addrs)))
    for f in frames:
        if f.file:
            f.file = _shorten(f.file, elf)
    return frames


# ── console parsing ────────────────────────────────────────────────────────────────────

def _parse_heap_blocks(lines: Sequence[str]) -> list[dict[str, Any]]:
    """Every PerfMon heap block, as ``{at, internal, dma, psram}`` (``at`` = line index)."""
    blocks: list[dict[str, Any]] = []
    current: dict[str, Any] | None = None
    section: str | None = None
    for i, line in enumerate(lines):
        if _HEAP_START in line:
            if current is not None:
                blocks.append(current)
            current = {"at": i}
            section = None
            continue
        if current is None:
            continue
        sm = _HEAP_SECTION_RE.search(line)
        if sm:
            section = _HEAP_SECTIONS[sm.group(1)]
            current.setdefault(section, {})
            continue
        fm = _HEAP_FIELD_RE.search(line)
        if fm and section is not None:
            current[section][_HEAP_FIELDS[fm.group(1)]] = int(fm.group(2))
            continue
        if "Total tasks:" in line or "Task Stack Usage" in line:
            blocks.append(current)
            current = None
            section = None
    if current is not None:
        blocks.append(current)
    return [b for b in blocks if len(b) > 1]


# An assertion prints its message on the line *before* ``abort() was called``, so the dump
# starts a few lines above the banner or the one fact naming the failure is lost.
_ASSERT_LOOKBACK = 5


def _panic_index(lines: Sequence[str]) -> int | None:
    """Index of the LAST panic banner — a soak log holds several and the newest is the one."""
    hit: int | None = None
    for i, line in enumerate(lines):
        if (_GURU_RE.search(line) or _ABORT_RE.search(line) or _WDT_RE.search(line)
                or _IWDT_RE.search(line) or _CANARY_RE.search(line) or _SMASH_RE.search(line)
                or _HEAP_CORRUPT_RE.search(line)):
            hit = i
    if hit is None:
        return None
    for j in range(hit - 1, max(-1, hit - 1 - _ASSERT_LOOKBACK), -1):
        if _ASSERT_RE.search(lines[j]) and _REBOOT_MARK not in lines[j]:
            return j
    return hit


def _collect_panic(lines: Sequence[str], start: int) -> tuple[Panic, list[str], int]:
    """Registers, backtrace addresses and the end of the dump, reading forward from ``start``.

    The dump stops at the reboot banner: everything after it belongs to the next boot and
    reading further would attribute the fresh boot's registers to the crash.
    """
    panic = Panic()
    addrs: list[str] = []
    end = len(lines)
    regs: dict[str, str] = {}
    for i in range(start, len(lines)):
        line = lines[i]
        if _REBOOT_MARK in line and i > start:
            end = i
            break
        g = _GURU_RE.search(line)
        if g and panic.cause is None:
            panic.core, panic.cause = int(g.group(1)), g.group(2)
        a = _ABORT_RE.search(line)
        if a:
            panic.cause = panic.cause or "abort"
            panic.pc = panic.pc or a.group(1)
            panic.core = panic.core if panic.core is not None else int(a.group(2))
        asrt = _ASSERT_RE.search(line)
        if asrt and panic.message is None:
            panic.message = asrt.group(1).strip()
        iw = _IWDT_RE.search(line)
        if iw:
            panic.cause = panic.cause or "InterruptWdtTimeout"
            panic.core = panic.core if panic.core is not None else int(iw.group(1))
        if _WDT_RE.search(line):
            panic.cause = panic.cause or "TaskWatchdog"
            for t in _WDT_TASK_RE.findall(line):
                if t not in panic.tasks:
                    panic.tasks.append(t)
        c = _CANARY_RE.search(line)
        if c:
            panic.cause = panic.cause or "StackCanary"
            panic.message = panic.message or c.group(1)
        if _SMASH_RE.search(line):
            panic.cause = panic.cause or "StackSmashing"
        if _HEAP_CORRUPT_RE.search(line):
            panic.cause = panic.cause or "CorruptHeap"
            panic.message = panic.message or line.strip()
        for name, value in _REG_RE.findall(line):
            regs.setdefault(name, value)
        bt = _BACKTRACE_RE.search(line)
        if bt and not addrs:
            addrs = [pc for pc, _sp in _BT_PAIR_RE.findall(bt.group(1))]
    if "PC" in regs:
        panic.pc = regs["PC"]
    if "EXCVADDR" in regs:
        panic.excvaddr = regs["EXCVADDR"]
    if "EXCCAUSE" in regs:
        code = int(regs["EXCCAUSE"], 16)
        panic.exccause = f"0x{code:02x}"
        if panic.cause is None:
            panic.cause = EXCCAUSE_NAMES.get(code)
    return panic, addrs, end


def _reset_reason(lines: Sequence[str], after: int | None) -> dict[str, str] | None:
    """The reset the panic caused, i.e. the first ``rst:`` line after it (else the last one)."""
    last: dict[str, str] | None = None
    for i, line in enumerate(lines):
        m = _RESET_RE.search(line)
        if not m:
            continue
        reason = {"code": m.group(1), "name": m.group(2)}
        if after is not None and i > after:
            return reason
        last = reason
    return last


def _address_kind(addr: str | None) -> str | None:
    """Human name for what an EXCVADDR points at — or that it points at nothing at all."""
    if not addr:
        return None
    try:
        value = int(addr, 16)
    except ValueError:
        return None
    if value < 0x1000:
        return "a null pointer" if value == 0 else f"a null pointer + 0x{value:x}"
    for lo, hi, name in _DATA_RANGES:
        if lo <= value < hi:
            return name
    return "no mapped data memory"


def _is_library(frame: Frame) -> bool:
    path = (frame.file or "").replace("\\", "/")
    return any(h in path for h in _LIBRARY_HINTS)


def first_app_frame(frames: Sequence[Frame]) -> Frame | None:
    """The topmost frame that is CrossPad code — where to start reading, not LVGL's insides."""
    for f in frames:
        if f.file and not _is_library(f):
            return f
    for f in frames:
        if f.func:
            return f
    return None


def _likely_cause(panic: Panic, frames: Sequence[Frame], heap: dict | None) -> str:
    """One sentence a human can act on: what kind of fault, where, and the memory context."""
    cause = panic.cause or "unknown fault"
    bits: list[str] = []
    if cause in ("StoreProhibited", "LoadProhibited", "LoadStoreError", "InstrFetchProhibited"):
        verb = "write to" if cause.startswith("Store") else "read from"
        kind = _address_kind(panic.excvaddr)
        where = f" ({kind})" if kind else ""
        stale = " — a stale or uninitialised pointer" if kind == "no mapped data memory" else ""
        bits.append(f"{cause}: a {verb} {panic.excvaddr}{where}{stale}")
    elif cause == "TaskWatchdog":
        who = ", ".join(panic.tasks) if panic.tasks else "a task"
        bits.append(f"task watchdog: {who} did not yield — something slow ran on it "
                    f"(SD or flash I/O on the LVGL or RT thread is the usual one)")
    elif cause == "InterruptWdtTimeout":
        bits.append("interrupt watchdog: interrupts were disabled or an ISR ran too long")
    elif cause in ("StackCanary", "StackSmashing"):
        bits.append(f"stack overflow ({panic.message or 'canary overwritten'}) — raise that "
                    f"task's stack or stop putting large buffers on it")
    elif cause == "CorruptHeap":
        bits.append("heap corruption — a write past the end of an allocation, or a use "
                    "after free; enable CONFIG_HEAP_POISONING_COMPREHENSIVE to catch the writer")
    elif cause == "abort":
        bits.append(f"abort(): {panic.message or 'no assertion message on the console'}")
    else:
        bits.append(cause if panic.exccause is None else f"{cause} (EXCCAUSE {panic.exccause})")

    app = first_app_frame(frames)
    if app is not None:
        bits.append(f"first CrossPad frame: {app.where()}")
    if heap:
        internal = (heap.get("internal") or {})
        free = internal.get("free")
        largest = internal.get("largest_free")
        if free is not None and (free < 20_000 or (largest is not None and largest < 8_192)):
            bits.append(f"internal heap was tight ({free} B free, largest block {largest} B) — "
                        f"an allocation that failed unchecked would look exactly like this")
    return "; ".join(bits)


# ── context file ───────────────────────────────────────────────────────────────────────

def write_context(lines: Sequence[str], center: int | None, path: Path,
                  count: int = DEFAULT_CONTEXT_LINES) -> dict[str, Any]:
    """Write ~``count`` lines around ``center`` and return where they came from.

    Two thirds before the panic and one third after: the cause is upstream of the banner,
    but the reboot that follows says whether the board came back.
    """
    total = len(lines)
    if center is None:
        start = max(0, total - count)
        end = total
    else:
        after = max(1, int(count * CONTEXT_AFTER_FRACTION))
        start = max(0, center - (count - after))
        end = min(total, center + after)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", errors="replace") as fh:
        for line in lines[start:end]:
            fh.write(line.rstrip("\r\n") + "\n")
    return {"path": str(path), "lines": end - start, "first_line": start + 1, "last_line": end}


# ── entry points ───────────────────────────────────────────────────────────────────────

def read_log(path: str | Path) -> list[str]:
    """Console logs carry UTF-8 box drawing and a panic dump; never die on the decode."""
    p = Path(path)
    if not p.is_file():
        raise HilError(BAD_ARGS, f"log file not found: {p}")
    with open(p, encoding="utf-8", errors="replace") as fh:
        return [line.rstrip("\r\n") for line in fh]


def diagnose(
    *,
    lines: Iterable[str] | None = None,
    log_file: str | Path | None = None,
    elf: str | Path | None = None,
    addr2line: str | Path | None = None,
    root: str | Path | None = None,
    context_dir: str | Path = "hil_logs",
    context_path: str | Path | None = None,
    context_lines: int = DEFAULT_CONTEXT_LINES,
    decode: bool = True,
    runner: Runner | None = None,
    source: str | None = None,
) -> dict[str, Any]:
    """One crash bundle from console lines or a log file.

    Exactly one of ``lines`` / ``log_file``. ``elf`` defaults to the build directory whose
    ELF SHA256 matches the one the firmware printed. ``decode=False`` returns the raw
    backtrace addresses without needing a toolchain at all. ``runner`` is resolved when it is
    called, not when this is defined, so a test can replace ``_run_addr2line`` wholesale.
    """
    if (lines is None) == (log_file is None):
        raise HilError(BAD_ARGS, "diagnose needs exactly one of lines= or log_file=")
    text_lines = read_log(log_file) if log_file is not None else [
        str(line).rstrip("\r\n") for line in lines or []
    ]
    kind = source or ("log_file" if log_file is not None else "lines")

    idx = _panic_index(text_lines)
    warnings: list[str] = []
    panic: Panic | None = None
    addrs: list[str] = []
    if idx is not None:
        panic, addrs, _end = _collect_panic(text_lines, idx)
        if not addrs:
            warnings.append("panic found but no Backtrace: line — the dump was truncated, or "
                            "the build has CONFIG_ESP_SYSTEM_USE_EH_FRAME (use espcoredump)")

    heaps = _parse_heap_blocks(text_lines)
    heap_before = next((b for b in reversed(heaps) if idx is None or b["at"] <= idx), None)
    heap_after = next((b for b in heaps if idx is not None and b["at"] > idx), None)
    if idx is not None and heap_after is None:
        warnings.append("no PerfMon heap block after the panic; heap_before is the last "
                        "sample taken before it")

    sha_prefix = None
    for line in reversed(text_lines[: (idx if idx is not None else len(text_lines)) + 40]):
        m = _ELF_SHA_RE.search(line)
        if m:
            sha_prefix = m.group(1)
            break

    frames: list[Frame] = []
    elf_path: Path | None = None
    elf_sha: str | None = None
    elf_match: bool | None = None
    if addrs and decode:
        elf_path, elf_sha, elf_match = find_elf(elf, sha_prefix, root)
        if elf_match is False:
            warnings.append(
                f"the device was running ELF SHA256 {sha_prefix}…, {elf_path} is {elf_sha[:9]}… "
                f"— line numbers below may be from a different build"
            )
        frames = decode_backtrace(addrs, elf_path, find_addr2line(addr2line),
                                  runner=runner or _run_addr2line)
    elif addrs:
        frames = [Frame(addr=a) for a in addrs]

    ctx_path = (Path(context_path) if context_path is not None
                else Path(context_dir) / f"diagnose_{time.strftime('%Y%m%d_%H%M%S')}.log")
    context = write_context(text_lines, idx, ctx_path, context_lines)

    reboots = sum(1 for line in text_lines if _REBOOT_MARK in line)
    likely = (_likely_cause(panic, frames, heap_before) if panic is not None
              else "no panic in the lines examined")
    return {
        "found": idx is not None,
        "source": {"kind": kind,
                   "path": str(log_file) if log_file is not None else None,
                   "lines": len(text_lines)},
        "reset_reason": _reset_reason(text_lines, idx),
        "panic": panic.to_dict() if panic is not None else None,
        "backtrace": [f.to_dict() for f in frames],
        "backtrace_addrs": addrs,
        "heap_after": heap_after,
        "heap_before": heap_before,
        "context": context["path"],
        "context_range": context,
        "likely_cause": likely,
        "elf": str(elf_path) if elf_path is not None else None,
        "elf_sha256": elf_sha,
        "elf_sha256_log": sha_prefix,
        "elf_matches_log": elf_match,
        "reboots": reboots,
        "warnings": warnings,
    }


def render(bundle: dict[str, Any]) -> str:
    """One-screen human summary — the same facts the JSON carries, in reading order."""
    out: list[str] = []
    src = bundle.get("source") or {}
    if not bundle.get("found"):
        out.append(f"no panic in {src.get('path') or src.get('kind')} "
                   f"({src.get('lines', 0)} lines, {bundle.get('reboots', 0)} boots)")
        rst = bundle.get("reset_reason")
        if rst:
            out.append(f"last reset: {rst['code']} ({rst['name']})")
        out.append(f"context: {bundle.get('context')}")
        return "\n".join(out)
    p = bundle.get("panic") or {}
    rst = bundle.get("reset_reason")
    head = f"PANIC core {p.get('core')}: {p.get('cause')}"
    if p.get("exccause"):
        head += f" (EXCCAUSE {p['exccause']})"
    out.append(head)
    out.append(f"  pc={p.get('pc')} excvaddr={p.get('excvaddr')}"
               + (f"  reset={rst['code']} ({rst['name']})" if rst else ""))
    if p.get("message"):
        out.append(f"  message: {p['message']}")
    out.append("backtrace:")
    for i, f in enumerate(bundle.get("backtrace") or []):
        mark = "  (inlined)" if f.get("inlined") else ""
        loc = f"{f.get('file')}:{f.get('line')}" if f.get("file") else "??"
        out.append(f"  #{i:<2} {f['addr']}  {f.get('func') or '??'}  {loc}{mark}")
    if not bundle.get("backtrace"):
        out.append("  (none — see warnings)")
    heap = bundle.get("heap_after") or bundle.get("heap_before")
    if heap:
        when = "after" if bundle.get("heap_after") else "before"
        internal = heap.get("internal") or {}
        psram = heap.get("psram") or {}
        out.append(f"heap ({when} the panic): internal free={internal.get('free')} "
                   f"largest={internal.get('largest_free')}  psram free={psram.get('free')}")
    if bundle.get("elf"):
        match = bundle.get("elf_matches_log")
        tag = {True: "matches the log", False: "DOES NOT match the log", None: "unverified"}[match]
        out.append(f"elf: {bundle['elf']} ({tag})")
    for w in bundle.get("warnings") or []:
        out.append(f"warning: {w}")
    out.append(f"likely cause: {bundle.get('likely_cause')}")
    out.append(f"context: {bundle.get('context')} ({bundle.get('context_range', {}).get('lines')} "
               f"lines around the panic)")
    return "\n".join(out)


__all__ = [
    "DEFAULT_CONTEXT_LINES",
    "EXCCAUSE_NAMES",
    "Frame",
    "Panic",
    "decode_backtrace",
    "diagnose",
    "elf_candidates",
    "elf_sha256",
    "find_addr2line",
    "find_elf",
    "first_app_frame",
    "read_log",
    "render",
    "write_context",
]
