"""Crash decode. The acceptance case is a real panic captured from a rev2 board.

`tests/fixtures/kit_selector_panic.log` is the console of an `app_churn` run that crashed the
board on 2026-08-26 (platform-idf/hil_findings/2026-08-26-kit-selector-panic.md records what
the toolchain made of it). `kit_selector_panic.addr2line.txt` is what
`xtensa-esp32s3-elf-addr2line -pfiaC` actually printed for those addresses, so the decode is
tested against real toolchain output with no toolchain installed.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from crosspad_hil import diagnose as diag
from crosspad_hil.errors import BAD_ARGS, ENV, HilError

FIXTURES = Path(__file__).parent / "fixtures"
PANIC_LOG = FIXTURES / "kit_selector_panic.log"
A2L_OUT = (FIXTURES / "kit_selector_panic.addr2line.txt").read_text()

# The addresses the device printed on its Backtrace: line, in order.
EXPECTED_ADDRS = [
    "0x40056f5d", "0x40379149", "0x420aa97d", "0x4208e2e7", "0x4207b0c1", "0x4201438b",
    "0x4205979a", "0x42014e9a", "0x4208df07", "0x4208fd62", "0x4208febd", "0x42031c91",
]


@pytest.fixture
def fake_elf(tmp_path: Path) -> Path:
    p = tmp_path / "CrossPad.elf"
    p.write_bytes(b"\x7fELF not really")
    return p


@pytest.fixture
def fake_addr2line(tmp_path: Path) -> Path:
    p = tmp_path / "xtensa-esp32s3-elf-addr2line"
    p.write_text("#!/bin/sh\n")
    return p


def canned_runner(_exe: Path, _elf: Path, addrs) -> str:
    assert list(addrs) == EXPECTED_ADDRS
    return A2L_OUT


def run_fixture(tmp_path: Path, elf: Path, a2l: Path, **kw) -> dict:
    return diag.diagnose(log_file=PANIC_LOG, elf=elf, addr2line=a2l, runner=canned_runner,
                         context_dir=tmp_path, **kw)


# --------------------------------------------------------------------------- acceptance

def test_decodes_the_recorded_frames(tmp_path, fake_elf, fake_addr2line):
    """The frames in hil_findings/2026-08-26-kit-selector-panic.md, in order."""
    b = run_fixture(tmp_path, fake_elf, fake_addr2line)
    frames = b["backtrace"]
    # An inlined frame repeats its parent's address, so only the real ones line up 1:1.
    assert [f["addr"] for f in frames if not f["inlined"]] == EXPECTED_ADDRS
    named = [(f["func"], f["file"], f["line"]) for f in frames]
    assert named[0] == (None, None, None)          # ROM memcpy: no symbol is the right answer
    assert named[1] == ("lv_memcpy", "lvgl/src/stdlib/clib/lv_string_clib.c", 40)
    assert named[2] == ("lv_array_push_back", "lvgl/src/misc/lv_array.c", 189)
    assert named[3] == ("lv_event_add", "lvgl/src/misc/lv_event.c", 135)
    assert named[4] == ("lv_obj_add_event_cb", "lvgl/src/core/lv_obj_event.c", 105)
    assert named[5] == ("kit_selector_pre_launch(crosspad_gui::ILvglApp*, _lv_obj_t*)",
                        "main/gui/gui.cpp", 89)
    assert named[6] == ("crosspad_gui::AppOrchestrator::onAppSelected(void*, _lv_obj_t*)",
                        "components/crosspad-gui/src/components/app_orchestrator.cpp", 100)
    assert named[7] == ("async_app_start(void*)", "main/hil_control.cpp", 79)
    assert [f["func"] for f in frames[8:]] == [
        "lv_async_timer_cb", "lv_timer_exec", "lv_timer_handler", "lv_timer_handler",
        "lvgl_port_task"]
    assert any(f["inlined"] for f in frames)       # lv_timer_handler is inlined into lv_timer_exec


def test_panic_registers_and_reset_reason(tmp_path, fake_elf, fake_addr2line):
    b = run_fixture(tmp_path, fake_elf, fake_addr2line)
    assert b["found"] is True
    assert b["panic"] == {
        "core": 0, "cause": "StoreProhibited", "exccause": "0x1d",
        "pc": "0x40056f60", "excvaddr": "0xd00f0748", "message": None, "tasks": [],
    }
    assert b["reset_reason"] == {"code": "0xc", "name": "RTC_SW_CPU_RST"}


def test_likely_cause_names_the_app_frame_and_the_wild_pointer(tmp_path, fake_elf,
                                                               fake_addr2line):
    b = run_fixture(tmp_path, fake_elf, fake_addr2line)
    cause = b["likely_cause"]
    assert "StoreProhibited" in cause
    assert "no mapped data memory" in cause          # 0xd00f0748 is not RAM on an ESP32-S3
    assert "main/gui/gui.cpp:89" in cause            # not the LVGL frames above it
    assert "kit_selector_pre_launch" in cause


def test_first_app_frame_skips_library_plumbing(tmp_path, fake_elf, fake_addr2line):
    b = run_fixture(tmp_path, fake_elf, fake_addr2line)
    frames = [diag.Frame(**{k: f[k] for k in ("addr", "func", "file", "line", "inlined")})
              for f in b["backtrace"]]
    assert diag.first_app_frame(frames).file == "main/gui/gui.cpp"


def test_heap_and_context_are_part_of_the_bundle(tmp_path, fake_elf, fake_addr2line):
    b = run_fixture(tmp_path, fake_elf, fake_addr2line)
    heap = b["heap_after"] or b["heap_before"]
    assert heap["internal"]["free"] > 0 and heap["psram"]["free"] > 0
    ctx = Path(b["context"])
    assert ctx.is_file()
    assert 0 < b["context_range"]["lines"] <= diag.DEFAULT_CONTEXT_LINES
    # The banner itself must be inside the context, or it is context for nothing.
    assert "Guru Meditation" in ctx.read_text(encoding="utf-8")


def test_context_line_count_is_configurable(tmp_path, fake_elf, fake_addr2line):
    b = run_fixture(tmp_path, fake_elf, fake_addr2line, context_lines=40)
    assert b["context_range"]["lines"] == 40


def test_elf_that_does_not_match_the_log_is_reported_not_hidden(tmp_path, fake_elf,
                                                               fake_addr2line):
    """Wrong line numbers look exactly like right ones, so the mismatch has to be loud."""
    b = run_fixture(tmp_path, fake_elf, fake_addr2line)
    assert b["elf_sha256_log"].startswith("14bf581c7")
    assert b["elf_matches_log"] is False
    assert any("may be from a different build" in w for w in b["warnings"])


def test_decode_false_needs_no_toolchain(tmp_path):
    b = diag.diagnose(log_file=PANIC_LOG, decode=False, context_dir=tmp_path)
    assert b["backtrace_addrs"] == EXPECTED_ADDRS
    assert [f["func"] for f in b["backtrace"]] == [None] * len(EXPECTED_ADDRS)
    assert b["elf"] is None


# --------------------------------------------------------------------------- parsing rules

def test_no_panic_is_not_an_error(tmp_path):
    log = tmp_path / "quiet.log"
    log.write_text("I (100) app_init: boot\nI (200) MAIN: hello\n")
    b = diag.diagnose(log_file=log, decode=False, context_dir=tmp_path)
    assert b["found"] is False
    assert b["panic"] is None
    assert b["likely_cause"] == "no panic in the lines examined"
    assert Path(b["context"]).is_file()


def test_the_last_panic_wins(tmp_path):
    lines = [
        "Guru Meditation Error: Core  0 panic'ed (LoadProhibited). Exception was unhandled.",
        "EXCVADDR: 0x00000000",
        "Backtrace: 0x40000001:0x3fc00000",
        "ESP-ROM:esp32s3-20210327",
        "Guru Meditation Error: Core  1 panic'ed (StoreProhibited). Exception was unhandled.",
        "EXCVADDR: 0xdeadbeef",
        "Backtrace: 0x40000002:0x3fc00000",
    ]
    b = diag.diagnose(lines=lines, decode=False, context_dir=tmp_path)
    assert b["panic"]["core"] == 1
    assert b["panic"]["excvaddr"] == "0xdeadbeef"
    assert b["backtrace_addrs"] == ["0x40000002"]


def test_the_dump_stops_at_the_reboot_banner(tmp_path):
    """Registers printed after the reset belong to the next boot, not to this crash."""
    lines = [
        "Guru Meditation Error: Core  0 panic'ed (StoreProhibited). Exception was unhandled.",
        "EXCVADDR: 0xdeadbeef",
        "Backtrace: 0x40000002:0x3fc00000",
        "ESP-ROM:esp32s3-20210327",
        "EXCVADDR: 0x11111111",
        "Backtrace: 0x49999999:0x3fc00000",
    ]
    b = diag.diagnose(lines=lines, decode=False, context_dir=tmp_path)
    assert b["panic"]["excvaddr"] == "0xdeadbeef"
    assert b["backtrace_addrs"] == ["0x40000002"]


def test_task_watchdog_names_the_task(tmp_path):
    lines = [
        "E (5000) task_wdt: Task watchdog got triggered. The following tasks/users did not "
        "reset the watchdog in time:",
        "E (5000) task_wdt:  - LVGL (CPU 0)",
    ]
    b = diag.diagnose(lines=lines, decode=False, context_dir=tmp_path)
    assert b["panic"]["cause"] == "TaskWatchdog"
    assert b["panic"]["tasks"] == ["LVGL"]
    assert "did not yield" in b["likely_cause"]


def test_abort_carries_the_assertion_message(tmp_path):
    lines = [
        "assert failed: xQueueSemaphoreTake queue.c:1653 (pxQueue)",
        "abort() was called at PC 0x42001234 on core 0",
        "Backtrace: 0x40000002:0x3fc00000",
    ]
    b = diag.diagnose(lines=lines, decode=False, context_dir=tmp_path)
    assert b["panic"]["cause"] == "abort"
    assert b["panic"]["pc"] == "0x42001234"
    assert "xQueueSemaphoreTake" in b["panic"]["message"]
    assert "xQueueSemaphoreTake" in b["likely_cause"]


def test_stack_canary_is_a_stack_overflow(tmp_path):
    lines = ["Stack canary watchpoint triggered (audio_bg)",
             "Backtrace: 0x40000002:0x3fc00000"]
    b = diag.diagnose(lines=lines, decode=False, context_dir=tmp_path)
    assert b["panic"]["cause"] == "StackCanary"
    assert "stack overflow" in b["likely_cause"]


def test_panic_without_a_backtrace_warns(tmp_path):
    lines = ["Guru Meditation Error: Core  0 panic'ed (IllegalInstruction). Exception was "
             "unhandled.", "EXCVADDR: 0x00000000"]
    b = diag.diagnose(lines=lines, decode=False, context_dir=tmp_path)
    assert b["backtrace_addrs"] == []
    assert any("no Backtrace:" in w for w in b["warnings"])


def test_a_tight_heap_is_added_to_the_cause(tmp_path):
    lines = [
        "I (1) PerfMon: === Heap Statistics ===",
        "I (1) PerfMon: Internal SRAM:",
        "I (1) PerfMon:   Free:                    9000",
        "I (1) PerfMon:   Largest free block:      4000",
        "I (1) PerfMon: Total tasks: 20",
        "Guru Meditation Error: Core  0 panic'ed (StoreProhibited). Exception was unhandled.",
        "EXCVADDR: 0x00000000",
        "Backtrace: 0x40000002:0x3fc00000",
    ]
    b = diag.diagnose(lines=lines, decode=False, context_dir=tmp_path)
    assert b["heap_before"]["internal"] == {"free": 9000, "largest_free": 4000}
    assert "internal heap was tight" in b["likely_cause"]


def test_a_null_pointer_is_named_as_one(tmp_path):
    lines = ["Guru Meditation Error: Core  0 panic'ed (LoadProhibited). Exception was unhandled.",
             "EXCVADDR: 0x00000008", "Backtrace: 0x40000002:0x3fc00000"]
    b = diag.diagnose(lines=lines, decode=False, context_dir=tmp_path)
    assert "null pointer + 0x8" in b["likely_cause"]


def test_render_is_one_screen(tmp_path, fake_elf, fake_addr2line):
    text = diag.render(run_fixture(tmp_path, fake_elf, fake_addr2line))
    assert "PANIC core 0: StoreProhibited" in text
    assert "kit_selector_pre_launch" in text
    assert "likely cause:" in text and "context:" in text
    assert len(text.splitlines()) < 40


# --------------------------------------------------------------------------- environment

def test_missing_log_file_is_bad_args(tmp_path):
    with pytest.raises(HilError) as e:
        diag.diagnose(log_file=tmp_path / "nope.log", decode=False)
    assert e.value.code == BAD_ARGS


def test_lines_or_log_file_but_not_both(tmp_path):
    with pytest.raises(HilError) as e:
        diag.diagnose(lines=["x"], log_file=PANIC_LOG, context_dir=tmp_path)
    assert e.value.code == BAD_ARGS
    with pytest.raises(HilError):
        diag.diagnose(context_dir=tmp_path)


def test_missing_toolchain_names_the_fix(monkeypatch, tmp_path):
    monkeypatch.delenv("CROSSPAD_ADDR2LINE", raising=False)
    monkeypatch.setenv("PATH", str(tmp_path))
    monkeypatch.setenv("IDF_TOOLS_PATH", str(tmp_path / "no-espressif"))
    with pytest.raises(HilError) as e:
        diag.find_addr2line()
    assert e.value.code == ENV
    assert "export.sh" in (e.value.hint or "")


def test_explicit_toolchain_that_is_not_there_is_env(tmp_path):
    with pytest.raises(HilError) as e:
        diag.find_addr2line(tmp_path / "nope")
    assert e.value.code == ENV


def test_addr2line_from_the_idf_tools_tree(monkeypatch, tmp_path):
    root = tmp_path / "tools" / "xtensa-esp-elf" / "esp-14.2.0" / "xtensa-esp-elf" / "bin"
    root.mkdir(parents=True)
    exe = root / diag.ADDR2LINE_EXE
    exe.write_text("#!/bin/sh\n")
    monkeypatch.delenv("CROSSPAD_ADDR2LINE", raising=False)
    monkeypatch.setenv("PATH", str(tmp_path / "empty"))
    monkeypatch.setenv("IDF_TOOLS_PATH", str(tmp_path))
    assert diag.find_addr2line() == exe


def test_missing_elf_names_the_build_command(monkeypatch, tmp_path):
    monkeypatch.setenv("CROSSPAD_IDF_ROOT", str(tmp_path / "nowhere"))
    monkeypatch.setenv("CROSSPAD_GIT_DIR", str(tmp_path / "nowhere"))
    monkeypatch.chdir(tmp_path)
    with pytest.raises(HilError) as e:
        diag.find_elf()
    assert e.value.code == ENV
    assert "idf.py build" in (e.value.hint or "")


def test_elf_is_picked_by_the_sha_the_firmware_printed(monkeypatch, tmp_path):
    root = tmp_path / "platform-idf"
    for name, body in (("build_v1", b"one"), ("build_v2", b"two")):
        (root / name).mkdir(parents=True)
        (root / name / "CrossPad.elf").write_bytes(body)
    monkeypatch.setenv("CROSSPAD_IDF_ROOT", str(root))
    want = diag.elf_sha256(root / "build_v2" / "CrossPad.elf")
    elf, sha, match = diag.find_elf(sha_prefix=want[:9])
    assert elf == root / "build_v2" / "CrossPad.elf" and sha == want and match is True


def test_addr2line_failure_is_env(tmp_path, fake_elf, fake_addr2line):
    def boom(_exe, _elf, _addrs):
        raise HilError(ENV, "addr2line failed (rc=1)")

    with pytest.raises(HilError) as e:
        diag.diagnose(log_file=PANIC_LOG, elf=fake_elf, addr2line=fake_addr2line, runner=boom,
                      context_dir=tmp_path)
    assert e.value.code == ENV


def test_shorten_trims_managed_components(tmp_path):
    elf = tmp_path / "build" / "CrossPad.elf"
    elf.parent.mkdir()
    elf.write_bytes(b"x")
    path = str(tmp_path / "managed_components" / "lvgl__lvgl" / "src" / "misc" / "lv_array.c")
    assert diag._shorten(path, elf) == "lvgl/src/misc/lv_array.c"


# --------------------------------------------------------------------------- CLI and daemon

def test_cli_diagnose_decodes_a_log_file(tmp_path, monkeypatch):
    """`crosspad-hil diagnose --log-file …` — the acceptance path, end to end."""
    import io

    from crosspad_hil.cli import CliDeps
    from crosspad_hil.cli import main as cli_main

    monkeypatch.setattr(diag, "_run_addr2line", canned_runner)
    monkeypatch.setattr(diag, "find_addr2line", lambda explicit=None: Path("/bin/true"))
    monkeypatch.setattr(diag, "find_elf",
                        lambda explicit=None, sha_prefix=None, root=None:
                        (Path("/tmp/CrossPad.elf"), "0" * 64, True))
    out, err = io.StringIO(), io.StringIO()
    code = cli_main(["diagnose", "--log-file", str(PANIC_LOG), "--logdir", str(tmp_path)],
                    deps=CliDeps(stdout=out, stderr=err))
    text = out.getvalue()
    assert code == 1, "a decoded panic is a failure, not a clean run"
    assert "kit_selector_pre_launch" in text and "main/gui/gui.cpp:89" in text


def test_cli_diagnose_json_carries_the_whole_bundle(tmp_path, monkeypatch):
    import io
    import json

    from crosspad_hil.cli import CliDeps
    from crosspad_hil.cli import main as cli_main

    out = io.StringIO()
    code = cli_main(["diagnose", "--log-file", str(PANIC_LOG), "--logdir", str(tmp_path),
                     "--no-decode", "--json"],
                    deps=CliDeps(stdout=out, stderr=io.StringIO()))
    data = json.loads(out.getvalue())
    assert code == 1
    assert data["panic"]["cause"] == "StoreProhibited"
    assert data["backtrace_addrs"] == EXPECTED_ADDRS
    assert Path(data["context"]).is_file()


def test_cli_diagnose_of_a_clean_log_exits_zero(tmp_path):
    import io

    from crosspad_hil.cli import CliDeps
    from crosspad_hil.cli import main as cli_main

    log = tmp_path / "clean.log"
    log.write_text("I (1) app_init: boot\n")
    code = cli_main(["diagnose", "--log", str(log), "--logdir", str(tmp_path), "--no-decode"],
                    deps=CliDeps(stdout=io.StringIO(), stderr=io.StringIO()))
    assert code == 0


def test_daemon_diagnose_crash_from_a_log_file(tmp_path):
    import io

    from crosspad_hil.serve import Daemon

    daemon = Daemon(io.StringIO(), io.StringIO())
    r = daemon.handle({"id": 1, "op": "diagnose.crash",
                       "args": {"log_file": str(PANIC_LOG), "decode": False,
                                "context_dir": str(tmp_path)}})
    assert r["ok"], r
    assert r["result"]["panic"]["excvaddr"] == "0xd00f0748"
    assert r["result"]["reset_reason"]["name"] == "RTC_SW_CPU_RST"
