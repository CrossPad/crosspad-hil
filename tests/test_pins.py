"""pins: the journal that outlives a killed run."""
from __future__ import annotations

from typing import Any

import pytest

from crosspad_hil import pins, verbs
from crosspad_hil.errors import HilError


def test_record_clear_pending_and_the_oldest_value_wins() -> None:
    pins.record("dev_1", pins.IN2, 0)
    pins.record("dev_1", pins.IN2, 24)      # a second pin on top of an unfinished one
    assert pins.pending("dev_1") == {"in2": 0}
    assert pins.pending("dev_2") == {}
    pins.clear("dev_1", pins.IN2)
    assert pins.pending("dev_1") == {}


class Board:
    def __init__(self, in2: int = 24, fail: bool = False) -> None:
        self.in2, self.fail = in2, fail

    def ingain(self, link: Any, input_index: int | None = None, db: int | None = None) -> dict:
        if self.fail:
            raise HilError("TIMEOUT", "no reply")
        if input_index is not None:
            self.in2 = db
        return {"IN1": 24, "IN2": self.in2}


def test_a_killed_run_is_put_right_by_the_next(monkeypatch: pytest.MonkeyPatch) -> None:
    """audio_loopback killed mid-run left IN2 at 24 dB in NVS."""
    board = Board(in2=24)
    monkeypatch.setattr(verbs, "ingain", board.ingain)
    pins.record("dev_1", pins.IN2, 0)
    logs: list[str] = []
    assert pins.restore_pending("LINK", "dev_1", logs.append) == {"in2": 0}
    assert board.in2 == 0 and pins.pending("dev_1") == {}
    assert "put back to 0" in logs[0]


def test_an_entry_that_cannot_be_restored_stays(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(verbs, "ingain", Board(fail=True).ingain)
    pins.record("dev_1", pins.IN2, 0)
    assert pins.restore_pending("LINK", "dev_1", lambda m: None) == {}
    assert pins.pending("dev_1") == {"in2": 0}
