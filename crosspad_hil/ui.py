"""Drive the board's screen the way a person does: find a control by what it says, tap it.

`UI_TREE` lists every visible widget with its text and rectangle (crosspad-gui
`ui_tree_probe`); `TOUCH_TAP` puts a finger on the panel through its own read
path. A test built on these presses the buttons a player presses -- a missing,
hidden, disabled or mislabelled control fails the test where a state verb would
have walked past it.

Matching ignores case, runs of spaces and the FontAwesome symbols LVGL draws
from its private-use range, so ``"Play"`` finds ``"\\uf04b Play"``.
"""
from __future__ import annotations

import re
import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from crosspad_hil import verbs
from crosspad_hil.errors import TIMEOUT, HilError

# LVGL's FontAwesome glyphs (LV_SYMBOL_*) the firmware's icon-only buttons carry.
ICON_PLAY = "\uf04b"
ICON_COPY = "\uf0c5"

# The top bar's drawn tiles carry no text (crosspad-gui transport_bar.cpp, the
# UX 2000 row): a person finds them by shape and place, the test by place.
TOP_BAR_Y = 20
TILE_REC_X = 141 + 19
TILE_PLAY_X = 181 + 19

CONTENT_AREA = (0, 44, 320, 192)   # under the top bar: x, y, w, h
BUSY_RETRIES = 20                  # UI_TREE refused for LVGL's lock, each ~1 s

SETTLE_S = 0.4          # a tap's 80 ms contact, the event, the redraw
POLL_S = 0.25
WAIT_S = 6.0

_SYMBOLS = re.compile("[-]")
_SPACES = re.compile(r"\s+")


def norm(text: str) -> str:
    """What a label says, for matching: no icon glyphs, single spaces, case-folded."""
    return _SPACES.sub(" ", _SYMBOLS.sub("", text)).strip().casefold()


def _matches(have: str, want: str, contains: bool) -> bool:
    """@p have says @p want; an icon-only @p want is matched glyph for glyph."""
    w, h = norm(want), norm(have)
    if not w:
        w, h = want.strip(), have.strip()
    return bool(w) and (w in h if contains else w == h)


@dataclass(frozen=True)
class Node:
    i: int
    parent: int
    cls: str
    x: int
    y: int
    w: int
    h: int
    interactive: bool
    hit: bool
    checked: bool
    disabled: bool
    focused: bool
    text: str

    @property
    def center(self) -> tuple[int, int]:
        return self.x + self.w // 2, self.y + self.h // 2

    def says(self, text: str, *, contains: bool = False) -> bool:
        return _matches(self.text, text, contains)


@dataclass(frozen=True)
class Screen:
    running: str
    nodes: tuple[Node, ...]

    def controls(self) -> list[Node]:
        """What a finger can reach: interactive and not under another screen or a dialog."""
        return [n for n in self.nodes if n.interactive and n.hit]

    def owner(self, node: Node) -> Node | None:
        """The nearest interactive ancestor of @p node: the control it belongs to."""
        p = node.parent
        while 0 <= p < len(self.nodes):
            if self.nodes[p].interactive:
                return self.nodes[p]
            p = self.nodes[p].parent
        return None

    def words(self, control: Node) -> list[str]:
        """What a control says: its own text and the labels that belong to it (not to
        a control inside it) -- a settings row reads "\uf287" then "USB"."""
        out = [control.text] if control.text else []
        for n in self.nodes:
            if n.cls == "label" and n.text and n.i != control.i:
                o = self.owner(n)
                if o is not None and o.i == control.i:
                    out.append(n.text)
        return out

    def _says(self, node: Node, text: str, contains: bool) -> bool:
        return any(_matches(w, text, contains) for w in self.words(node))

    def find(self, text: str, *, contains: bool = False, interactive: bool = True) -> Node | None:
        """The smallest widget that says @p text -- a button rather than the panel
        holding it. With @p interactive, only a control a finger reaches (``hit``)
        whose text or a label inside it says so; without, any widget drawn,
        covered or not."""
        if interactive:
            hits = [n for n in self.controls() if self._says(n, text, contains)]
        else:
            hits = [n for n in self.nodes if n.says(text, contains=contains)]
        return min(hits, key=lambda n: n.w * n.h) if hits else None

    def spot(self, control: Node, text: str, contains: bool = False) -> tuple[int, int]:
        """Where to put the finger for @p text on @p control: its own text means its
        middle; a label of it (a row of a list, a word on a settings row) means
        that label's middle -- a list's middle is some other row."""
        if control.says(text, contains=contains):
            return control.center
        for n in self.nodes:
            if n.cls == "label" and n.says(text, contains=contains):
                o = self.owner(n)
                if o is not None and o.i == control.i:
                    return n.center
        return control.center

    def at(self, x: int, y: int) -> Node | None:
        """The smallest control a finger reaches whose rectangle holds (x, y)."""
        hits = [n for n in self.controls() if n.x <= x < n.x + n.w and n.y <= y < n.y + n.h]
        return min(hits, key=lambda n: n.w * n.h) if hits else None

    def shows(self, text: str, *, contains: bool = False) -> bool:
        return self.find(text, contains=contains, interactive=False) is not None

    def labels(self) -> list[str]:
        """What a person could press here: each control's words, its own or inside it."""
        seen: list[str] = []
        for n in self.controls():
            words = list(dict.fromkeys(self.words(n)))
            t = " ".join(_SPACES.sub(" ", _SYMBOLS.sub("", w)).strip() for w in words)
            t = _SPACES.sub(" ", t).strip()
            if t and t not in seen:
                seen.append(t[:60])
        return seen


class UiMissing(HilError):
    """A control a person would press is not on the screen (or never became tappable)."""


class Ui:
    """The screen of one CDC link. @p wait is the run's cancellable sleep."""

    def __init__(self, link: Any, *, wait: Callable[[float], Any] = time.sleep,
                 settle_s: float = SETTLE_S, timeout_s: float = WAIT_S) -> None:
        self.link = link
        self._wait = wait
        self.settle_s = settle_s
        self.timeout_s = timeout_s
        self.busy = 0             # UI_TREE answers "lvgl_busy": the UI held LVGL over 1 s

    def screen(self) -> Screen:
        """What is on screen now. A UI busy for over a second (the probe could not
        take LVGL's lock) is asked again: the busy span is recorded in ``busy``."""
        for left in range(BUSY_RETRIES - 1, -1, -1):
            try:
                tree = verbs.ui_tree(self.link)
                break
            except HilError as e:
                if "lvgl_busy" not in str(e.details.get("reply", e.message)) or not left:
                    raise
                self.busy += 1
                self._wait(POLL_S)
        return Screen(running=str(tree["running"]),
                      nodes=tuple(Node(i=n["i"], parent=n["parent"], cls=n["cls"], x=n["x"],
                                       y=n["y"], w=n["w"], h=n["h"],
                                       interactive=n["interactive"], hit=n["hit"],
                                       checked=n["checked"],
                                       disabled=n["disabled"], focused=n["focused"],
                                       text=n["text"]) for n in tree["nodes"]))

    def wait_for(self, text: str, *, contains: bool = False, interactive: bool = True,
                 enabled: bool = True, timeout_s: float | None = None) -> Node:
        """Poll until a widget saying @p text is on screen (and enabled); UiMissing
        with what the screen offered instead."""
        deadline = time.monotonic() + (self.timeout_s if timeout_s is None else timeout_s)
        while True:
            sc = self.screen()
            node = sc.find(text, contains=contains, interactive=interactive)
            if node and not (enabled and node.disabled):
                return node
            if time.monotonic() >= deadline:
                state = "disabled" if node else "not on screen"
                raise UiMissing(TIMEOUT, f"{text!r} {state} (app {sc.running})",
                                hint="could press: " + ", ".join(sc.labels()[:30]),
                                wanted=text, running=sc.running, labels=sc.labels())
            self._wait(POLL_S)

    def tap_node(self, node: Node, hold_ms: int = 80) -> Node:
        x, y = node.center
        verbs.touch_tap(self.link, x, y, hold_ms)
        self._wait(self.settle_s + hold_ms / 1000)
        return node

    def seek(self, text: str, *, contains: bool = False,
             area: tuple[int, int, int, int] | None = None,
             swipes: int = 25, down: bool = True) -> Node:
        """Drag a list with a finger until a line saying @p text shows, as a person
        scrolls for a row; return the widget saying it (a control, or the label of
        a row the list itself handles). @p area (x, y, w, h) is where to drag,
        else the largest control on screen; @p down first drags upwards to see
        what is further down, then the other way. UiMissing once the list has
        stopped moving both ways."""
        last: tuple[str, ...] | None = None
        still = 0
        turned = False
        for _ in range(swipes * 2):
            sc = self.screen()
            node = self._row(sc, text, contains)
            if node is not None:
                return self._settled(node, text, contains)
            box = area or self._list_area(sc)
            seen = tuple(n.text for n in sc.nodes if n.cls == "label" and self._in(n, box))
            # The first drag on a list can only give it the focus; two that move
            # nothing are its end -- then the other way, once.
            still = still + 1 if seen == last else 0
            if still >= 2:
                if turned:
                    break
                turned, down, still = True, not down, 0
            last = seen
            x = box[0] + box[2] // 2
            top, bottom = box[1] + box[3] // 6, box[1] + box[3] * 5 // 6
            y1, y2 = (bottom, top) if down else (top, bottom)
            verbs.touch_drag(self.link, x, y1, x, y2, 300, 50)
            self._wait(0.35 + 0.35)
        sc = self.screen()
        raise UiMissing(TIMEOUT, f"{text!r} not found scrolling the list (app {sc.running})",
                        hint="last lines: " + ", ".join(t for t in (last or ()) if t)[:300],
                        wanted=text, running=sc.running)

    def _settled(self, node: Node, text: str, contains: bool) -> Node:
        """@p node once the list has stopped gliding: a drag leaves the list moving,
        and a row tapped where it was a moment ago is the row below it."""
        for _ in range(12):
            self._wait(POLL_S)
            again = self._row(self.screen(), text, contains)
            if again is None:
                break
            if again.center == node.center:
                return again
            node = again
        return node

    @staticmethod
    def _in(n: Node, box: tuple[int, int, int, int]) -> bool:
        x, y, w, h = box
        return x <= n.x + n.w // 2 < x + w and y <= n.y + n.h // 2 < y + h

    @staticmethod
    def _row(sc: Screen, text: str, contains: bool) -> Node | None:
        """What to tap for @p text in a list: the control saying it itself, else the
        label saying it whose control a finger reaches (a row the list handles)."""
        control = sc.find(text, contains=contains)
        if control is None:
            return None
        label = None
        for n in sc.nodes:
            if n.cls == "label" and n.says(text, contains=contains):
                o = sc.owner(n)
                if o is not None and o.i == control.i:
                    label = n
                    break
        # A list the size of many rows says what its first row says: its middle is
        # another row (a tap there picked the kit two rows down), so the row's label.
        if control.says(text, contains=contains) and (label is None or control.h < 2 * label.h):
            return control
        return label or control

    @staticmethod
    def _list_area(sc: Screen) -> tuple[int, int, int, int]:
        """The biggest list-sized control, else the screen under the top bar: a page
        that scrolls as a whole (a settings page) is dragged anywhere on it."""
        lists = [n for n in sc.controls() if n.h >= 80 and n.w * n.h < 320 * 200]
        if not lists:
            return CONTENT_AREA
        n = max(lists, key=lambda n: n.w * n.h)
        return n.x, n.y, n.w, n.h

    def knob_to(self, text: str, *, contains: bool = False, steps: int = 40) -> str:
        """Turn the encoder a detent at a time until the focused control says @p text,
        as a hand reaches a control the finger cannot (one in the encoder's group
        only). Returns what it said; UiMissing after a full turn round the group."""
        seen: list[str] = []
        for _ in range(steps):
            label = str(verbs.enc_focus(self.link).get("label", ""))
            if _matches(label, text, contains):
                return label
            seen.append(label)
            verbs.enc_rotate(self.link, 1)
            self._wait(0.2)
        raise UiMissing(TIMEOUT, f"the knob never reached {text!r}",
                        hint="focus went: " + " > ".join(dict.fromkeys(seen))[:300],
                        wanted=text)

    def reach(self, text: str, *, contains: bool = False) -> Node | None:
        """Use the control saying @p text the way a person can: tap it, or -- when
        no finger reaches it but the knob does -- turn to it and click. Returns the
        node tapped, None when it was clicked with the knob."""
        sc = self.screen()
        if sc.find(text, contains=contains) is not None:
            return self.tap(text, contains=contains)
        drawn = [n for n in sc.nodes if n.interactive and _matches(n.text, text, contains)]
        if not drawn:
            return self.tap(text, contains=contains)       # waits, then says what was there
        self.knob_to(text, contains=contains)
        verbs.enc_press(self.link)
        self._wait(self.settle_s)
        return None

    def keyboard_ok(self) -> Node:
        """The on-screen keyboard's OK key (LVGL's: bottom right, the tick), which
        takes what the text box says -- a suggested name accepted as it is."""
        sc = self.screen()
        kb = next((n for n in sc.controls() if n.cls == "keyboard"), None)
        if kb is None:
            raise UiMissing(TIMEOUT, f"no keyboard on screen (app {sc.running})",
                            hint="could press: " + ", ".join(sc.labels()[:20]))
        x, y = kb.x + kb.w - kb.w // 14, kb.y + kb.h - kb.h // 8
        verbs.touch_tap(self.link, x, y)
        self._wait(self.settle_s)
        return kb

    def tap_tile(self, x: int, hold_ms: int = 80) -> Node:
        """A top-bar tile by its place (TILE_*_X); the row dragged back to the
        transport first when the status page is showing, as a finger does.
        UiMissing when nothing a finger reaches is there."""
        if str(verbs.topbar(self.link).get("transport")) != "1":
            verbs.touch_drag(self.link, 40, TOP_BAR_Y, 300, TOP_BAR_Y, 250, 50)
            self._wait(0.8)
        node = self.screen().at(x, TOP_BAR_Y)
        if node is None or node.w > 80:
            raise UiMissing(TIMEOUT, f"no top-bar tile at x={x}",
                            hint="is the transport page showing? TOPBAR transport=1")
        return self.tap_node(node, hold_ms)

    def tap(self, text: str, *, contains: bool = False, hold_ms: int = 80,
            timeout_s: float | None = None) -> Node:
        """Tap where @p text is, once its control is there and enabled: the control's
        middle, or the label's when the words are a label of a bigger control."""
        node = self.wait_for(text, contains=contains, timeout_s=timeout_s)
        x, y = self.screen().spot(node, text, contains)
        verbs.touch_tap(self.link, x, y, hold_ms)
        self._wait(self.settle_s + hold_ms / 1000)
        return node
