"""beat's closing verdict: an underrun fails, a long render pass or two does not."""
from __future__ import annotations

from typing import Any

import pytest

from crosspad_hil import verbs
from crosspad_hil.scenarios import beat

MEM = {"int_free": 54000, "psram_free": 3_000_000}


def verdict(monkeypatch: pytest.MonkeyPatch, before: dict[str, int],
            after: dict[str, int]) -> tuple[list[str], dict[str, Any]]:
    run = object.__new__(beat.BeatRun)
    run.link = object()
    run.rebooted = False
    run.data = {}
    run.base = {"resets": 3, "smpl": before, "mem": MEM}
    monkeypatch.setattr(verbs, "reset_count", lambda link: 3)
    monkeypatch.setattr(verbs, "smpl_counters", lambda link: after)
    monkeypatch.setattr(verbs, "mem", lambda link: MEM)
    return run.verdict_after(), run.data


def test_one_long_render_pass_is_recorded_not_failed(monkeypatch: pytest.MonkeyPatch) -> None:
    """2026-11-26 on main: overruns 1 -> 2, underrun 0, every hit of the bounce in place."""
    out, data = verdict(monkeypatch, {"underrun": 0, "overruns": 1}, {"underrun": 0, "overruns": 2})
    assert out == [] and data["smpl_delta"] == {"underrun": 0, "overruns": 1}


def test_more_than_the_budget_fails(monkeypatch: pytest.MonkeyPatch) -> None:
    out, _ = verdict(monkeypatch, {"underrun": 0, "overruns": 1},
                     {"underrun": 0, "overruns": 2 + beat.OVERRUN_BUDGET})
    assert out == [f"SMPL overruns 1 -> {2 + beat.OVERRUN_BUDGET} "
                   f"(more than {beat.OVERRUN_BUDGET} in one run)"]


def test_an_underrun_fails(monkeypatch: pytest.MonkeyPatch) -> None:
    out, _ = verdict(monkeypatch, {"underrun": 0, "overruns": 1}, {"underrun": 1, "overruns": 1})
    assert out == ["SMPL underrun 0 -> 1"]
