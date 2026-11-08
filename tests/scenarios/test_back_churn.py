"""back_churn's DENY list: buttons a blind press must never hit, because the
state they leave behind fails the scenarios that run after it."""
from __future__ import annotations

import pytest

from crosspad_hil.scenarios import back_churn


@pytest.mark.parametrize("label", ["16 LEVELS", "16 LEVELS: PAD 6", "Full level",
                                   "USB audio", "Bluetooth", "Delete"])
def test_deny_skips_buttons_that_leave_state_behind(label: str) -> None:
    assert back_churn.DENY.search(label)


@pytest.mark.parametrize("label", ["Mixer", "Kits", "Pad Mixer", "Settings"])
def test_deny_lets_ordinary_buttons_through(label: str) -> None:
    assert not back_churn.DENY.search(label)
