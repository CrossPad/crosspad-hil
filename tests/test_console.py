"""Console: STM VCP tail with a ring buffer, parser events, expect and boot detection.

Every hygiene invariant from spec §2.4 that concerns the console is a test here:
opening never asserts DTR/RTS, and the only reset is Console.reset().
"""
from __future__ import annotations

import threading
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest

from crosspad_hil import console as console_mod
from crosspad_hil.console import BootResult, Console, ExpectResult, ReadResult
from crosspad_hil.parsers import ConsoleEvent
from crosspad_hil.serial_open import open_serial
from tests.fakes import FakeSerial

# Same shape as knowledge/markers.yaml (contract §knowledge), inline so this file does not
# depend on the YAML being finished.
MARKERS: dict = {
    "boot_markers": ["ESP-ROM:", "main_task: Started on CPU0"],
    "required": [
        "Platform fully initialized",
        "STM32 ident:",
        "Crosspad initialization complete",
        "All systems operational",
        "LVGL setup done successfully",
        "App registry initialized",
        "LoadMainScreen completed successfully",
    ],
    "required_stability": [
        "Platform fully initialized",
        "Crosspad initialization complete",
        "All systems operational",
        "LVGL setup done successfully",
        "App registry initialized",
        "LoadMainScreen completed successfully",
    ],
    "optional": ["SD Card mounted successfully", "ES8388 [1] started", "DRV2605 found"],
    "error_line": r"^E \(\d+\)",
    "error_allow": ["file not found"],
    "fatal_patterns": [
        r"Guru Meditation",
        r"abort\(\) was called",
        r"assert failed",
        r"CORRUPT HEAP",
        r"Stack smashing",
        r"Task watchdog",
        r"Interrupt wdt timeout",
        r"LoadProhibited",
        r"StoreProhibited",
        r"InstrFetchProhibited",
        r"IllegalInstruction",
        r"Stack canary watchpoint",
        r"task_wdt",
        r"rst:0x[0-9a-f]+ \(([A-Z_]*(PANIC|WDT|BROWNOUT))",
    ],
    "reset_reason": r"rst:(0x[0-9a-fA-F]+)\s*\(([^)]+)\)",
    "heap_block": {
        "start": "Heap Statistics",
        "line": r"PerfMon:\s+Free:\s+(\d+) bytes",
        "end": "Total tasks:",
    },
    "kit_request": r"hil_control: KIT_LOAD (\d+) (queued|started)",
    "cdc_drops": r"CDC: (\d+) commands dropped",
    "reboot": "ESP-ROM:esp32s3",
    "boot_timeout_s": 45,
}

BANNER = "ESP-ROM:esp32s3-20210327"
BOOT_OK: list[str] = [
    BANNER,
    "Build:Mar 27 2021",
    "rst:0x1 (POWERON),boot:0x8 (SPI_FAST_FLASH_BOOT)",
    "I (312) main_task: Started on CPU0",
    "I (900) CrosspadPlatform: Platform fully initialized",
    "I (950) main: STM32 ident: 0x42 fw=2.0",
    "I (1200) main: Crosspad initialization complete",
    "I (1210) main: All systems operational",
    "E (1300) audio: file not found: /littlefs/STARTUP.wav",
    "I (1500) display: LVGL setup done successfully",
    "I (1600) gui: App registry initialized",
    "I (1800) gui: LoadMainScreen completed successfully",
]


def wait_until(pred: Callable[[], bool], timeout_s: float = 2.0) -> bool:
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        if pred():
            return True
        time.sleep(0.005)
    return pred()


@pytest.fixture
def make_console(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> Callable[..., tuple[Console, FakeSerial]]:
    """Open a Console on a FakeSerial. Locks go to tmp_path; every console is closed."""
    monkeypatch.setenv("XDG_RUNTIME_DIR", str(tmp_path))
    opened: list[Console] = []

    def _make(
        lines: list[str] | None = None, *, reset: bool = False, **kw: Any
    ) -> tuple[Console, FakeSerial]:
        fake = FakeSerial([], timeout=0.01)
        if lines:
            fake.feed(lines)
        con = Console(
            "/dev/fake-console",
            knowledge=kw.pop("knowledge", MARKERS),
            serial_factory=lambda path, **_kw: fake,
            **kw,
        )
        con.open(reset=reset)
        opened.append(con)
        return con, fake

    yield _make
    for con in opened:
        con.close()


# ── hygiene ─────────────────────────────────────────────────────────────────
def test_open_never_asserts_dtr_or_rts(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Through the real open_serial: only deassertions happen on open."""
    monkeypatch.setenv("XDG_RUNTIME_DIR", str(tmp_path))
    fake = FakeSerial([], timeout=0.01)
    con = Console(
        "/dev/fake-console",
        knowledge=MARKERS,
        serial_factory=lambda path, **kw: open_serial(path, serial_cls=lambda: fake, **kw),
    )
    con.open()
    try:
        assert fake.is_open
        assert fake.control_history, "open_serial must deassert DTR/RTS explicitly"
        assert all(value is False for _name, value in fake.control_history)
    finally:
        con.close()


def test_reset_only_via_reset(make_console: Callable[..., Any]) -> None:
    con, fake = make_console()
    assert ("rts", True) not in fake.control_history
    con.reset()
    # reset_pulse: dtr False, rts True, 0.1 s, rts False -- DTR rewritten after each RTS write
    assert fake.control_history[-5:] == [
        ("dtr", False), ("rts", True), ("dtr", False), ("rts", False), ("dtr", False)]
    assert fake.rts is False and fake.dtr is False
    assert sum(1 for name, value in fake.control_history if name == "rts" and value) == 1


def test_open_with_reset_pulses_exactly_once(make_console: Callable[..., Any]) -> None:
    con, fake = make_console(reset=True)
    assert sum(1 for name, value in fake.control_history if name == "rts" and value) == 1
    assert fake.rts is False


def test_open_takes_port_lock(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    from crosspad_hil.errors import PORT_BUSY, HilError
    from crosspad_hil.locks import PortLock

    monkeypatch.setenv("XDG_RUNTIME_DIR", str(tmp_path))
    fake = FakeSerial([], timeout=0.01)
    con = Console("/dev/fake-console", knowledge=MARKERS, serial_factory=lambda p, **kw: fake)
    con.open()
    try:
        holders = PortLock.holders()
        assert [(h["port"], h["purpose"]) for h in holders] == [("/dev/fake-console", "console")]
        with pytest.raises(HilError) as ei:
            PortLock("/dev/fake-console", "cdc").acquire()
        assert ei.value.code == PORT_BUSY
    finally:
        con.close()
    assert PortLock.holders() == []


# ── ring, seq, read ─────────────────────────────────────────────────────────
def test_lines_are_numbered_from_one_and_crlf_stripped(make_console: Callable[..., Any]) -> None:
    con, fake = make_console(["first", "second"])
    assert wait_until(lambda: con.seq == 2)
    res = con.read()
    assert isinstance(res, ReadResult)
    assert res.lines == [(1, "first"), (2, "second")]
    assert res.next_seq == 3
    assert res.lines_lost == 0


def test_ring_wrap_counts_lines_lost(make_console: Callable[..., Any]) -> None:
    con, fake = make_console([f"line {i}" for i in range(1, 9)], ring_size=5)
    assert wait_until(lambda: con.seq == 8)
    res = con.read()
    assert [s for s, _ in res.lines] == [4, 5, 6, 7, 8]
    assert res.lines_lost == 0  # since_seq None: "from the oldest kept line"
    res = con.read(since_seq=1)
    assert [s for s, _ in res.lines] == [4, 5, 6, 7, 8]
    assert res.lines_lost == 3  # seqs 1..3 were evicted
    assert con.snapshot()["lines_lost"] == 3


def test_read_since_seq_limit_and_match(make_console: Callable[..., Any]) -> None:
    con, fake = make_console(["I (1) a", "W (2) b", "I (3) c", "I (4) d"])
    assert wait_until(lambda: con.seq == 4)
    res = con.read(since_seq=2, limit=2)
    assert res.lines == [(2, "W (2) b"), (3, "I (3) c")]
    assert res.next_seq == 4
    res = con.read(match=r"^I \(")
    assert [line for _, line in res.lines] == ["I (1) a", "I (3) c", "I (4) d"]
    res = con.read(since_seq=99)
    assert res.lines == [] and res.next_seq == 99 and res.lines_lost == 0


def test_read_wait_ms_blocks_until_a_line_arrives(make_console: Callable[..., Any]) -> None:
    con, fake = make_console()
    threading.Timer(0.05, lambda: fake.feed(["late"])).start()
    t0 = time.monotonic()
    res = con.read(wait_ms=2000)
    assert res.lines == [(1, "late")]
    assert time.monotonic() - t0 < 1.5


def test_read_wait_ms_returns_empty_on_timeout(make_console: Callable[..., Any]) -> None:
    con, _ = make_console()
    t0 = time.monotonic()
    res = con.read(wait_ms=50)
    assert res.lines == [] and res.next_seq == 1
    assert 0.04 <= time.monotonic() - t0 < 1.0


def test_log_file_receives_every_line(make_console: Callable[..., Any], tmp_path: Path) -> None:
    log = tmp_path / "logs" / "console.log"
    con, fake = make_console(["alpha", "beta"], log_path=log)
    assert wait_until(lambda: con.seq == 2)
    con.close()
    assert log.read_text(encoding="utf-8") == "alpha\nbeta\n"
    assert con.snapshot()["log_path"] == str(log)


# ── expect ──────────────────────────────────────────────────────────────────
def test_expect_hit_on_new_lines_only(make_console: Callable[..., Any]) -> None:
    con, fake = make_console(["I (1) Platform fully initialized"])
    assert wait_until(lambda: con.seq == 1)
    # Default window: lines after the call. The marker is already in the ring → no hit.
    miss = con.expect(["Platform fully initialized"], timeout_s=0.1)
    assert miss.hit is None and miss.rejected is None and miss.seq is None
    # since_seq=0 widens the window to the whole ring.
    hit = con.expect(["Platform fully initialized"], timeout_s=0.1, since_seq=0)
    assert isinstance(hit, ExpectResult)
    assert hit.hit == "Platform fully initialized" and hit.seq == 1
    # A line that arrives after the call is seen without since_seq.
    threading.Timer(0.05, lambda: fake.feed(["I (2) LoadMainScreen completed"])).start()
    hit = con.expect([r"LoadMainScreen \w+"], timeout_s=2.0)
    assert hit.hit == r"LoadMainScreen \w+" and hit.seq == 2
    assert hit.context[-1] == "I (2) LoadMainScreen completed"
    assert 0.0 <= hit.elapsed_s < 2.0


def test_expect_reject_wins_and_reports_pattern(make_console: Callable[..., Any]) -> None:
    con, fake = make_console()
    threading.Timer(
        0.02, lambda: fake.feed(["I (1) booting", "Guru Meditation Error: Core 0 panic'ed",
                                  "I (9) LoadMainScreen completed successfully"])
    ).start()
    res = con.expect(["LoadMainScreen"], reject=["Guru Meditation", "abort\\(\\)"], timeout_s=2.0)
    assert res.hit is None
    assert res.rejected == "Guru Meditation"
    assert res.seq == 2
    assert res.context == ["I (1) booting", "Guru Meditation Error: Core 0 panic'ed"]


def test_expect_context_is_twenty_lines_ending_at_hit(make_console: Callable[..., Any]) -> None:
    con, fake = make_console([f"line {i}" for i in range(1, 31)])
    assert wait_until(lambda: con.seq == 30)
    res = con.expect([r"^line 25$"], since_seq=0, timeout_s=0.5)
    assert res.seq == 25
    assert len(res.context) == 20
    assert res.context[0] == "line 6" and res.context[-1] == "line 25"


def test_expect_timeout_returns_tail_context(make_console: Callable[..., Any]) -> None:
    con, fake = make_console(["a", "b"])
    assert wait_until(lambda: con.seq == 2)
    res = con.expect(["never"], timeout_s=0.05)
    assert res.hit is None and res.rejected is None and res.seq is None
    assert res.context == ["a", "b"]
    assert res.elapsed_s >= 0.05


# ── wait_boot ───────────────────────────────────────────────────────────────
def test_wait_boot_complete(make_console: Callable[..., Any]) -> None:
    con, fake = make_console(BOOT_OK)
    t0 = time.monotonic()
    boot = con.wait_boot(timeout_s=5.0, settle_s=0.05)
    assert isinstance(boot, BootResult)
    assert boot.complete is True
    assert boot.missing == []
    assert boot.fatal == []
    assert boot.errors == []  # "file not found" is allow-listed (hil_smoke ERROR_ALLOWLIST)
    assert boot.bootloops == 0
    assert 0.05 <= boot.seconds <= time.monotonic() - t0 + 0.01
    assert boot.seconds < 2.0  # returned at completion + settle, not at the deadline


def test_wait_boot_missing_markers_at_timeout(make_console: Callable[..., Any]) -> None:
    partial = [line for line in BOOT_OK
               if "LoadMainScreen" not in line and "STM32 ident" not in line]
    con, fake = make_console(partial)
    boot = con.wait_boot(timeout_s=0.3, settle_s=0.05)
    assert boot.complete is False
    assert boot.missing == ["STM32 ident:", "LoadMainScreen completed successfully"]
    assert boot.seconds >= 0.3


def test_wait_boot_reports_fatal_and_error_lines(make_console: Callable[..., Any]) -> None:
    lines = list(BOOT_OK)
    lines.insert(4, "E (500) sd: mount failed")
    lines.insert(5, "Guru Meditation Error: Core  1 panic'ed (LoadProhibited)")
    con, fake = make_console(lines)
    boot = con.wait_boot(timeout_s=5.0, settle_s=0.05)
    assert boot.complete is True
    assert [f["line"] for f in boot.fatal] == [
        "Guru Meditation Error: Core  1 panic'ed (LoadProhibited)"
    ]
    assert boot.fatal[0]["pattern"] == "Guru Meditation"
    assert boot.fatal[0]["seq"] == 6
    assert [e["line"] for e in boot.errors] == ["E (500) sd: mount failed"]


def test_wait_boot_counts_bootloops(make_console: Callable[..., Any]) -> None:
    # Banner twice = the board started booting twice (hil_smoke evaluate: max(count) - 1).
    lines = [BANNER, "I (312) main_task: Started on CPU0", BANNER] + BOOT_OK[1:]
    con, fake = make_console(lines)
    boot = con.wait_boot(timeout_s=5.0, settle_s=0.05)
    assert boot.complete is True
    assert boot.bootloops == 1


def test_wait_boot_default_timeout_from_knowledge(make_console: Callable[..., Any]) -> None:
    con, fake = make_console(BOOT_OK, knowledge={**MARKERS, "boot_timeout_s": 0.2})
    boot = con.wait_boot(settle_s=0.01)
    assert boot.complete is True


def test_reset_clears_boot_tracking_and_fatal_baseline(make_console: Callable[..., Any]) -> None:
    con, fake = make_console(BOOT_OK + ["Guru Meditation Error: Core 0 panic'ed"])
    assert wait_until(lambda: con.seq == len(BOOT_OK) + 1)
    assert con.snapshot()["boot_complete"] is True
    con.reset()
    snap = con.snapshot()
    assert snap["boot_complete"] is False
    assert snap["missing_markers"] == MARKERS["required"]
    fake.feed(BOOT_OK)
    boot = con.wait_boot(timeout_s=5.0, settle_s=0.05)
    assert boot.complete is True
    assert boot.fatal == []  # the pre-reset fatal is not this boot's


# ── events ──────────────────────────────────────────────────────────────────
def test_events_emitted_to_callback_and_ring(make_console: Callable[..., Any]) -> None:
    seen: list[ConsoleEvent] = []
    con, fake = make_console(BOOT_OK, on_event=seen.append)
    assert wait_until(lambda: any(e.kind == "boot_complete" for e in seen))
    kinds = [e.kind for e in seen]
    assert "reboot" in kinds
    assert "reset_reason" in kinds
    assert kinds.count("boot_marker") == len(MARKERS["required"])
    assert kinds[-1] == "boot_complete"
    assert seen[-1].data == {"missing": []}
    assert seen[-1].seq == len(BOOT_OK)
    assert con.events(since_seq=0) == seen
    assert con.events(since_seq=seen[-1].seq) == []
    assert con.events(since_seq=seen[-1].seq - 1) == [e for e in seen if e.seq == seen[-1].seq]


def test_on_event_exception_does_not_kill_reader(
    make_console: Callable[..., Any], capsys: pytest.CaptureFixture[str]
) -> None:
    def boom(ev: ConsoleEvent) -> None:
        raise RuntimeError("callback bug")

    con, fake = make_console([BANNER, "after"], on_event=boom)
    assert wait_until(lambda: con.seq == 2)
    assert con.read().lines[-1] == (2, "after")
    assert "on_event raised" in capsys.readouterr().err


def test_snapshot_merges_parser_and_session_fields(make_console: Callable[..., Any]) -> None:
    con, fake = make_console(BOOT_OK + ["CDC: 4 commands dropped"])
    assert wait_until(lambda: con.seq == len(BOOT_OK) + 1)
    snap = con.snapshot()
    assert snap["seq"] == len(BOOT_OK) + 1
    assert snap["lines_lost"] == 0
    assert snap["port"] == "/dev/fake-console"
    assert snap["log_path"] is None
    assert snap["reboots"] == 1
    assert snap["reset_reasons"] == ["0x1 (POWERON)"]
    assert snap["cdc_drops"] == 4
    assert snap["boot_complete"] is True


def test_close_is_idempotent_and_stops_reader(make_console: Callable[..., Any]) -> None:
    con, fake = make_console(["x"])
    assert wait_until(lambda: con.seq == 1)
    con.close()
    con.close()
    assert fake.is_open is False
    fake.feed(["after close"])
    time.sleep(0.05)
    assert con.seq == 1


def test_module_constants() -> None:
    assert console_mod.READ_CHUNK == 512
    assert console_mod.CONTEXT_LINES == 20
