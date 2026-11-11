"""back_churn's DENY list: buttons a blind press must never hit, because the
state they leave behind fails the scenarios that run after it."""
from __future__ import annotations

import pytest

from crosspad_hil.scenarios import back_churn


@pytest.mark.parametrize("label", ["16 LEVELS", "16 LEVELS: PAD 6", "Full level",
                                   "USB audio", "Bluetooth", "Delete",
                                   "\uf067 NEW", "+ NEW KIT", "+ NEW DRUMS"])
def test_deny_skips_buttons_that_leave_state_behind(label: str) -> None:
    assert back_churn.DENY.search(label)


@pytest.mark.parametrize("label", ["Mixer", "Kits", "Pad Mixer", "Settings", "NEWS", "Song 20"])
def test_deny_lets_ordinary_buttons_through(label: str) -> None:
    assert not back_churn.DENY.search(label)


class _Ctx:
    def __init__(self) -> None:
        import threading
        self.cancelled = threading.Event()
        self.logged: list[str] = []

    def log(self, line: str) -> None:
        self.logged.append(line)


def test_restore_song_reopens_the_song_the_visit_started_with(monkeypatch) -> None:
    state = {"project": "Song 21", "open": 2}
    sent: list[str | None] = []

    def asong(link, action=None, timeout_s=None):
        sent.append(action)
        if action and action.startswith("OPEN "):
            state["project"] = action[5:]
            return {"ok": 1, **state}
        return dict(state)

    monkeypatch.setattr(back_churn.verbs, "asong", asong)
    assert back_churn._restore_song(_Ctx(), None, "Song 20")
    assert "OPEN Song 20" in sent and state["project"] == "Song 20"


def test_restore_song_leaves_the_same_song_alone(monkeypatch) -> None:
    sent: list[str | None] = []

    def asong(link, action=None, timeout_s=None):
        sent.append(action)
        return {"project": "Song 20", "open": 2}

    monkeypatch.setattr(back_churn.verbs, "asong", asong)
    assert back_churn._restore_song(_Ctx(), None, "Song 20")
    assert sent == [None]


def test_restore_song_counts_a_refusal(monkeypatch) -> None:
    def asong(link, action=None, timeout_s=None):
        return {"ok": 0, "project": "Song 21", "open": 2}

    monkeypatch.setattr(back_churn.verbs, "asong", asong)
    ctx = _Ctx()
    assert not back_churn._restore_song(ctx, None, "Song 20")
    assert ctx.logged
