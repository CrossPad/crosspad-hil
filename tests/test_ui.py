"""crosspad_hil.ui without a board: matching, owners, spots, hits and the verbs it sends."""
from __future__ import annotations

import dataclasses
from typing import Any

import pytest

from crosspad_hil import ui as ui_mod
from crosspad_hil import verbs
from crosspad_hil.ui import ICON_COPY, Node, Screen, Ui, UiMissing, norm


def node(i: int, parent: int, cls: str, x: int, y: int, w: int, h: int, text: str = "", *,
         interactive: bool = False, hit: bool = True, disabled: bool = False) -> Node:
    return Node(i=i, parent=parent, cls=cls, x=x, y=y, w=w, h=h, interactive=interactive,
                hit=hit, checked=False, disabled=disabled, focused=False, text=text)


# A Settings-like page: a full-screen container, a row (icon label, word label),
# a list whose rows are labels, a button, and a map button drawn under it all.
SCREEN = Screen(running="Settings", nodes=(
    node(0, -1, "obj", 0, 40, 320, 200, interactive=True),
    node(1, 0, "obj", 4, 50, 300, 30, "", interactive=True),
    node(2, 1, "label", 10, 55, 12, 12, ""),
    node(3, 1, "label", 30, 55, 40, 12, "USB"),
    node(4, 0, "editable_button", 80, 90, 230, 100, "BASIC", interactive=True),
    node(5, 4, "label", 90, 95, 100, 12, "BASIC"),
    node(6, 4, "label", 90, 120, 100, 12, "ŚWIĘTE SŁOWA"),
    node(7, 0, "button", 4, 200, 60, 28, "", interactive=True),
    node(8, 0, "button", 70, 200, 60, 28, "Save", interactive=True, disabled=True),
    node(9, 0, "button", 4, 90, 70, 40, "Kits", interactive=True, hit=False),
))


def test_norm_drops_icons_case_and_spaces():
    assert norm("  Play ") == "play"


def test_a_row_says_the_words_of_its_labels():
    row = SCREEN.find("USB")
    assert row is not None and row.i == 1
    assert SCREEN.spot(row, "USB") == SCREEN.nodes[3].center


def test_a_list_row_is_tapped_on_its_label_not_the_list_middle():
    lst = SCREEN.find("ŚWIĘTE SŁOWA")
    assert lst is not None and lst.i == 4
    assert SCREEN.spot(lst, "ŚWIĘTE SŁOWA") == SCREEN.nodes[6].center


def test_a_list_row_seeked_is_its_label_when_the_list_says_its_first_row():
    # The list's own text is its first row's (BASIC): its middle is another row.
    row = Ui._row(SCREEN, "BASIC", False)
    assert row is not None and row.i == 5


def test_a_container_does_not_say_its_controls_words():
    assert SCREEN.owner(SCREEN.nodes[3]).i == 1
    assert "USB" not in SCREEN.words(SCREEN.nodes[0])


def test_an_icon_only_control_is_found_by_its_glyph():
    assert SCREEN.find(ICON_COPY).i == 7


def test_a_covered_control_is_not_a_control():
    assert SCREEN.find("Kits") is None
    assert SCREEN.find("Kits", interactive=False).i == 9


def test_at_picks_the_smallest_reachable_control():
    assert SCREEN.at(20, 60).i == 1


class FakeVerbs:
    """ui_tree answers from a list of screens; taps and drags are recorded."""

    def __init__(self, screens: list[Screen]) -> None:
        self.screens = screens
        self.calls: list[tuple] = []

    def ui_tree(self, _link: Any) -> dict[str, Any]:
        sc = self.screens[0] if len(self.screens) == 1 else self.screens.pop(0)
        return {"running": sc.running, "nodes": [dataclasses.asdict(n) for n in sc.nodes]}

    def touch_tap(self, _link: Any, x: int, y: int, hold_ms: int = 80) -> dict:
        self.calls.append(("tap", x, y, hold_ms))
        return {}


@pytest.fixture
def fake(monkeypatch: pytest.MonkeyPatch):
    def install(screens: list[Screen]) -> FakeVerbs:
        f = FakeVerbs(screens)
        monkeypatch.setattr(verbs, "ui_tree", f.ui_tree)
        monkeypatch.setattr(verbs, "touch_tap", f.touch_tap)
        return f
    return install


def test_tap_puts_the_finger_on_the_words(fake):
    f = fake([SCREEN])
    Ui(object(), wait=lambda s: None).tap("USB")
    assert f.calls == [("tap", *SCREEN.nodes[3].center, 80)]


def test_a_disabled_control_is_waited_for_then_reported(fake, monkeypatch):
    f = fake([SCREEN])
    monkeypatch.setattr(ui_mod, "POLL_S", 0.0)
    with pytest.raises(UiMissing) as ei:
        Ui(object(), wait=lambda s: None, timeout_s=0.05).tap("Save")
    assert "disabled" in ei.value.message
    assert f.calls == []


def test_missing_control_names_what_could_be_pressed(fake, monkeypatch):
    fake([SCREEN])
    monkeypatch.setattr(ui_mod, "POLL_S", 0.0)
    with pytest.raises(UiMissing) as ei:
        Ui(object(), wait=lambda s: None, timeout_s=0.05).tap("Export")
    assert "USB" in ei.value.hint and "Kits" not in ei.value.hint
