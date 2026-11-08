"""The console stamps every line with when it reached the host (live_jam's device clock)."""
from __future__ import annotations

import time
from pathlib import Path

from crosspad_hil.console import Console
from tests.fakes import FakeSerial


class WaitingSerial(FakeSerial):
    """FakeSerial that says how much it holds, like pyserial's in_waiting."""

    @property
    def in_waiting(self) -> int:
        return len(self._rx) if hasattr(self, "_rx") else 0


def test_lines_carry_arrival_times_and_go_to_the_tsv(tmp_path: Path) -> None:
    ser = FakeSerial([], timeout=0.01)
    clock = iter(float(i) for i in range(100, 10_000))
    con = Console("/dev/fake", serial_factory=lambda *a, **k: ser,
                  tsv_path=tmp_path / "console.tsv", clock=lambda: next(clock))
    con._ingest("I (1000) a: one", 10.5, 1.7e9)
    con._ingest("I (2000) b: two", 11.25, 1.7e9 + 1)
    rows = con.read_timed()
    assert [(s, t) for s, t, _w, _l in rows] == [(1, 10.5), (2, 11.25)]
    assert rows[1][3] == "I (2000) b: two"
    con.start_tsv(tmp_path / "late.tsv")
    con._ingest("I (3000) c: three", 12.0, 1.7e9 + 2)
    text = (tmp_path / "late.tsv").read_text().splitlines()
    assert text[0].split("\t")[:2] == ["1", "10.500000"] and len(text) == 3
    assert text[2].endswith("I (3000) c: three")
    assert [r[0] for r in con.read_timed(since_seq=2, match="c:")] == [3]


def test_arrival_time_is_when_the_newline_came(tmp_path: Path) -> None:
    con = Console("/dev/fake", serial_factory=lambda *a, **k: FakeSerial([], timeout=0.01))
    t0 = time.monotonic()
    con._ingest("x")
    (_s, t, _w, _l), = con.read_timed()
    assert t0 <= t <= time.monotonic()
