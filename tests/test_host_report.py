"""host_report: what the hub tells the board about this computer."""
from __future__ import annotations

from pathlib import Path

from crosspad_hil import host_report as hr

MEMINFO = "MemTotal:       63133696 kB\nMemFree:         1000000 kB\nMemAvailable:   30000000 kB\n"


def _stat(path: Path, user: int, idle: int) -> None:
    # cpu  user nice system idle iowait irq softirq steal
    path.write_text(f"cpu  {user} 0 0 {idle} 0 0 0 0\ncpu0 1 2 3 4\n")


def test_cpu_percent_is_the_busy_share_between_readings(tmp_path: Path) -> None:
    stat = tmp_path / "stat"
    _stat(stat, 100, 900)
    cpu = hr.CpuMeter(stat)
    _stat(stat, 130, 970)          # +30 busy, +70 idle
    assert cpu.percent() == 30
    assert cpu.percent() == 0      # nothing moved since


def test_ram_used_is_total_minus_available(tmp_path: Path) -> None:
    meminfo = tmp_path / "meminfo"
    meminfo.write_text(MEMINFO)
    assert hr.ram_mb(meminfo) == ((63133696 - 30000000) // 1024, 63133696 // 1024)


def test_host_info_line_matches_the_firmware_verb(tmp_path: Path) -> None:
    stat, meminfo = tmp_path / "stat", tmp_path / "meminfo"
    _stat(stat, 0, 100)
    meminfo.write_text(MEMINFO)
    line = hr.host_info_line(hr.CpuMeter(stat), meminfo=meminfo, now=1789214622.7)
    assert line.startswith("HOST_INFO cpu=0 ram=32357/61654 time=1789214622 tz=")
    assert len(line) <= 190


def test_host_info_line_carries_the_daw_masks(tmp_path: Path) -> None:
    stat, meminfo = tmp_path / "stat", tmp_path / "meminfo"
    _stat(stat, 0, 100)
    meminfo.write_text(MEMINFO)
    line = hr.host_info_line(hr.CpuMeter(stat), meminfo=meminfo, now=1789214622.7, daws=(10, 2))
    assert line.endswith(" daws=10/2")


def test_latest_version_reads_the_json_document(tmp_path: Path) -> None:
    path = tmp_path / "latest.json"
    assert hr.latest_version(path) is None
    path.write_text('{"version": "1.0.2"}')
    assert hr.latest_version(path) == "1.0.2"
    for bad in ("not json", "[1]", '{"version": 3}', '{"version": ""}'):
        path.write_text(bad)
        assert hr.latest_version(path) is None
