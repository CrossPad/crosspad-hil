"""ConsoleParser: ESP-IDF console lines → typed events (ported hil_smoke / hil_stability rules)."""
from __future__ import annotations

import copy

import pytest

from crosspad_hil.parsers import ConsoleEvent, ConsoleParser

# Identical to knowledge/markers.yaml (contract §knowledge); kept inline so this file has no
# dependency on the knowledge task.
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
        r"LoadProhibited",
        r"StoreProhibited",
        r"InstrFetchProhibited",
        r"IllegalInstruction",
        r"Stack canary watchpoint",
        r"task_wdt",
        r"Task watchdog",
        r"Interrupt wdt timeout",
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

# A complete, healthy boot as the STM VCP bridge shows it.
GOOD_BOOT = [
    "ESP-ROM:esp32s3-20210327",
    "rst:0x1 (POWERON),boot:0x8 (SPI_FAST_FLASH_BOOT)",
    "I (525) main_task: Started on CPU0",
    "I (1234) CrosspadPlatform: Platform fully initialized",
    "I (1300) stm32_i2c: STM32 ident: proto=1.2 fw=1.4 pcb=20",
    "I (1500) bsp: SD Card mounted successfully",
    "I (2000) main: Crosspad initialization complete",
    "I (2100) main: All systems operational",
    "I (2500) display: LVGL setup done successfully",
    "I (2600) gui: App registry initialized",
    "I (2700) gui: LoadMainScreen completed successfully",
]


def feed_all(parser: ConsoleParser, lines: list[str], start: int = 0) -> list[ConsoleEvent]:
    out: list[ConsoleEvent] = []
    for i, line in enumerate(lines, start):
        out.extend(parser.feed(i, line))
    return out


def kinds(events: list[ConsoleEvent]) -> list[str]:
    return [e.kind for e in events]


def test_event_dataclass_shape() -> None:
    ev = ConsoleEvent(kind="fatal", seq=3, line="x", data={"pattern": "y"})
    assert (ev.kind, ev.seq, ev.line, ev.data) == ("fatal", 3, "x", {"pattern": "y"})


def test_good_boot_emits_markers_then_boot_complete() -> None:
    p = ConsoleParser(knowledge=MARKERS)
    events = feed_all(p, GOOD_BOOT)
    markers = [e.data["marker"] for e in events if e.kind == "boot_marker"]
    assert markers == (
        MARKERS["required"][:2] + ["SD Card mounted successfully"] + MARKERS["required"][2:]
    )
    complete = [e for e in events if e.kind == "boot_complete"]
    assert len(complete) == 1
    assert complete[0].seq == 10
    assert complete[0].data == {"missing": []}
    snap = p.snapshot()
    assert snap["boot_complete"] is True
    assert snap["missing_markers"] == []
    assert snap["bootloops"] == 0
    assert snap["reboots"] == 1
    assert snap["reset_reasons"] == ["0x1 (POWERON)"]
    assert snap["errors"] == [] and snap["fatals"] == []
    assert snap["markers_seen"]["ESP-ROM:"] == 1
    assert snap["markers_seen"]["main_task: Started on CPU0"] == 1
    assert snap["markers_seen"]["LoadMainScreen completed successfully"] == 1


def test_incomplete_boot_reports_missing_in_required_order() -> None:
    p = ConsoleParser(knowledge=MARKERS)
    lines = [ln for ln in GOOD_BOOT if "STM32 ident" not in ln and "App registry" not in ln]
    events = feed_all(p, lines)
    assert "boot_complete" not in kinds(events)
    snap = p.snapshot()
    assert snap["boot_complete"] is False
    assert snap["missing_markers"] == ["STM32 ident:", "App registry initialized"]


def test_required_override_uses_stability_set() -> None:
    p = ConsoleParser(knowledge=MARKERS, required=MARKERS["required_stability"])
    lines = [ln for ln in GOOD_BOOT if "STM32 ident" not in ln]
    events = feed_all(p, lines)
    assert "boot_complete" in kinds(events)
    assert p.snapshot()["missing_markers"] == []


def test_reboot_event_and_bootloop_count() -> None:
    p = ConsoleParser(knowledge=MARKERS)
    events = feed_all(p, GOOD_BOOT + GOOD_BOOT + GOOD_BOOT[:3])
    reboots = [e for e in events if e.kind == "reboot"]
    assert [e.data["count"] for e in reboots] == [1, 2, 3]
    assert reboots[2].line == "ESP-ROM:esp32s3-20210327"
    snap = p.snapshot()
    assert snap["reboots"] == 3
    # hil_smoke.evaluate: boots = max(count per BOOT_MARKER); bootloops = boots - 1
    assert snap["bootloops"] == 2
    # the second boot completed again
    assert kinds(events).count("boot_complete") == 2


def test_reset_reason_first_after_banner_only() -> None:
    p = ConsoleParser(knowledge=MARKERS)
    events = feed_all(
        p,
        [
            "rst:0x1 (POWERON),boot:0x8 (SPI_FAST_FLASH_BOOT)",  # banner missed: still accepted
            "rst:0x3 (RTC_SW_SYS_RST),boot:0x8 (SPI_FAST_FLASH_BOOT)",  # ignored until next banner
            "ESP-ROM:esp32s3-20210327",
            "rst:0x10 (RTCWDT_RTC_RST),boot:0x8 (SPI_FAST_FLASH_BOOT)",
        ],
    )
    reasons = [e for e in events if e.kind == "reset_reason"]
    assert [e.data for e in reasons] == [
        {"code": "0x1", "name": "POWERON"},
        {"code": "0x10", "name": "RTCWDT_RTC_RST"},
    ]
    assert p.snapshot()["reset_reasons"] == ["0x1 (POWERON)", "0x10 (RTCWDT_RTC_RST)"]
    # a WDT reset reason is also a fatal (hil_kit_churn FATAL_RE). Note the ported regex's
    # [A-Z_]* cannot cross a digit, so TG1WDT_SYS_RST / TG0WDT_SYS_RST are *not* fatals —
    # that is the firmware scripts' behaviour and is kept as is.
    fatal = [e for e in events if e.kind == "fatal"]
    assert len(fatal) == 1 and fatal[0].seq == 3
    assert fatal[0].data["pattern"] == r"rst:0x[0-9a-f]+ \(([A-Z_]*(PANIC|WDT|BROWNOUT))"


@pytest.mark.parametrize(
    ("line", "pattern"),
    [
        (
            "Guru Meditation Error: Core  1 panic'ed (LoadProhibited). Exception was unhandled.",
            r"Guru Meditation",
        ),
        ("abort() was called at PC 0x4037a1b2 on core 0", r"abort\(\) was called"),
        ("assert failed: xQueueGenericSend queue.c:832 (pxQueue)", r"assert failed"),
        (
            "CORRUPT HEAP: Bad head at 0x3fcb1234. Expected 0xabba1234 got 0x00000000",
            r"CORRUPT HEAP",
        ),
        ("Stack smashing protect failure!", r"Stack smashing"),
        # matches both "task_wdt" and "Task watchdog": the first pattern in the list wins
        ("E (91234) task_wdt: Task watchdog got triggered.", r"task_wdt"),
        ("Interrupt wdt timeout on CPU1", r"Interrupt wdt timeout"),
        (
            "Debug exception reason: Stack canary watchpoint triggered (audio_rt)",
            r"Stack canary watchpoint",
        ),
    ],
)
def test_fatal_patterns_first_match_wins(line: str, pattern: str) -> None:
    p = ConsoleParser(knowledge=MARKERS)
    events = p.feed(7, line)
    fatal = [e for e in events if e.kind == "fatal"]
    assert len(fatal) == 1
    assert fatal[0].data == {"pattern": pattern}
    assert p.snapshot()["fatals"] == [{"seq": 7, "pattern": pattern, "line": line}]


def test_guru_meditation_line_is_one_fatal_not_two() -> None:
    # Matches both "Guru Meditation" and "LoadProhibited": hil_stability breaks on the first hit.
    p = ConsoleParser(knowledge=MARKERS)
    events = p.feed(1, "Guru Meditation Error: Core  1 panic'ed (LoadProhibited).")
    assert kinds(events) == ["fatal"]


@pytest.mark.parametrize(
    ("line", "is_error"),
    [
        ("E (3100) sdmmc_cmd: sdmmc_read_sectors_dma: returned 0x107", True),
        ("E (3000) audio: file not found: /sdcard/STARTUP.wav", False),  # ERROR_ALLOWLIST
        ("W (3200) main: CDC: 0 commands dropped (app_queue full)", False),
        ("I (3300) main: E (1) is not at line start", False),
    ],
)
def test_error_line_with_allowlist(line: str, is_error: bool) -> None:
    p = ConsoleParser(knowledge=MARKERS)
    events = p.feed(42, line)
    errs = [e for e in events if e.kind == "error_line"]
    assert (len(errs) == 1) is is_error
    if is_error:
        assert errs[0].data == {}
        assert p.snapshot()["errors"] == [{"seq": 42, "line": line}]
    else:
        assert p.snapshot()["errors"] == []


HEAP_BLOCK = [
    "I (10000) PerfMon: === Heap Statistics ===",
    "I (10000) PerfMon: Internal SRAM:",
    "I (10000) PerfMon:   Total:               398964 bytes",
    "I (10001) PerfMon:   Free:                76544 bytes",
    "I (10001) PerfMon:   Largest free block:   65536 bytes",
    "I (10002) PerfMon: DMA-capable:",
    "I (10002) PerfMon:   Free:                70000 bytes",
    "I (10003) PerfMon: PSRAM:",
    "I (10003) PerfMon:   Free:              8123456 bytes",
    "I (10004) PerfMon:   Allocated blocks:       1201",
    "I (10010) PerfMon:   Total tasks: 23",
]


def test_heap_block_emits_one_event_per_slot_at_block_end() -> None:
    p = ConsoleParser(knowledge=MARKERS)
    events = feed_all(p, HEAP_BLOCK, start=100)
    heap = [e for e in events if e.kind == "heap"]
    assert [(e.data["slot"], e.data["free"]) for e in heap] == [
        (0, 76544),
        (1, 70000),
        (2, 8123456),
    ]
    assert all(e.seq == 110 for e in heap)  # all emitted on the "Total tasks:" line
    assert p.snapshot()["heap"] == {0: 76544, 1: 70000, 2: 8123456}


def test_heap_block_without_end_is_discarded_by_next_start() -> None:
    p = ConsoleParser(knowledge=MARKERS)
    truncated = HEAP_BLOCK[:4]  # start + one Free line, no "Total tasks:"
    events = feed_all(p, truncated + HEAP_BLOCK)
    heap = [e for e in events if e.kind == "heap"]
    assert len(heap) == 3
    assert p.snapshot()["heap"] == {0: 76544, 1: 70000, 2: 8123456}


def test_heap_latest_value_wins() -> None:
    p = ConsoleParser(knowledge=MARKERS)
    feed_all(p, HEAP_BLOCK)
    second = [ln.replace("76544", "70120") for ln in HEAP_BLOCK]
    feed_all(p, second, start=50)
    assert p.snapshot()["heap"][0] == 70120


def test_kit_requests_and_cdc_drops() -> None:
    p = ConsoleParser(knowledge=MARKERS)
    events = feed_all(
        p,
        [
            "I (5000) hil_control: KIT_LOAD 3 started",
            "I (5100) hil_control: KIT_LOAD 4 queued",
            "W (6000) main: CDC: 12 commands dropped (app_queue full)",
            "W (7000) main: CDC: 15 commands dropped (app_queue full)",
        ],
        start=200,
    )
    kit = [e for e in events if e.kind == "kit"]
    assert [e.data for e in kit] == [{"kit": 3, "state": "started"}, {"kit": 4, "state": "queued"}]
    drops = [e for e in events if e.kind == "cdc_drops"]
    assert [e.data for e in drops] == [{"dropped": 12}, {"dropped": 15}]
    snap = p.snapshot()
    assert snap["kit_requests"] == [
        {"kit": 3, "state": "started", "seq": 200},
        {"kit": 4, "state": "queued", "seq": 201},
    ]
    assert snap["cdc_drops"] == 15  # cumulative firmware counter: latest, not summed


def test_reset_boot_tracking_clears_boot_state_keeps_session_counters() -> None:
    p = ConsoleParser(knowledge=MARKERS)
    feed_all(p, GOOD_BOOT + ["E (9000) x: broken"])
    p.feed(99, "Guru Meditation Error: Core  0 panic'ed (StoreProhibited).")
    before = p.snapshot()
    assert before["boot_complete"] is True and before["bootloops"] == 0
    p.reset_boot_tracking()
    snap = p.snapshot()
    assert snap["boot_complete"] is False
    assert snap["markers_seen"] == {}
    assert snap["missing_markers"] == MARKERS["required"]
    assert snap["bootloops"] == 0
    # session counters survive
    assert snap["reboots"] == 1
    assert len(snap["errors"]) == 1
    assert len(snap["fatals"]) == 1
    assert snap["reset_reasons"] == ["0x1 (POWERON)"]
    # and a fresh boot after the reset completes again
    events = feed_all(p, GOOD_BOOT, start=300)
    assert "boot_complete" in kinds(events)
    assert p.snapshot()["bootloops"] == 0


def test_snapshot_is_a_copy() -> None:
    p = ConsoleParser(knowledge=MARKERS)
    feed_all(p, GOOD_BOOT)
    snap = p.snapshot()
    snap["markers_seen"].clear()
    snap["reset_reasons"].append("bogus")
    fresh = p.snapshot()
    assert fresh["markers_seen"]["ESP-ROM:"] == 1
    assert fresh["reset_reasons"] == ["0x1 (POWERON)"]


def test_constructor_does_not_mutate_knowledge() -> None:
    k = copy.deepcopy(MARKERS)
    p = ConsoleParser(knowledge=k, required=["only this"])
    p.feed(0, "only this")
    assert k == MARKERS


def test_plain_line_yields_no_events() -> None:
    p = ConsoleParser(knowledge=MARKERS)
    assert p.feed(1, "I (100) main: hello") == []
    assert p.feed(2, "") == []
