"""`crosspad-hil padmix`: the Pad Mixer app on the host, for mastering kits in bulk.

Sixteen faders on the board's current kit, moved live (`PAD_MIX pad vol`, the
app's own fader) and written into the kit's kit.json (`KIT_SAVE`) a moment
after the last move, before another kit is loaded and on the way out. Enter
swaps the faders for the kit list and back; a kit picked there is loaded on
the board and its faders come up.

Escape takes back the last action -- a fader move, or a kit change, which
loads the previous kit again -- and in the kit list it leaves the list.

`PadMixSession` is the logic and talks to the board only through `verbs`;
`run()` is the curses screen around it, keys and mouse.
"""
from __future__ import annotations

import math
import os
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from . import verbs
from .errors import ENV, HilError

PADS = 16
# A fader is written to the board at most this often while a key repeats.
SEND_EVERY_S = 0.03
# kit.json is written once the faders have been still this long.
AUTOSAVE_AFTER_S = 1.5
# Moves of one pad closer together than this are one step of undo.
MERGE_WITHIN_S = 0.35
# How long an audition holds the pad before releasing it.
AUDITION_HOLD_S = 0.25
BIG_STEP = 8
DOUBLE_CLICK_S = 0.4


def volume_db(vol: int) -> str:
    """The engines play a pad at vol/100 linear: 100 is unity."""
    if vol <= 0:
        return "-inf dB"
    return f"{20.0 * math.log10(vol / 100.0):+.1f} dB"


@dataclass
class Action:
    kind: str                 # "gain" | "kit"
    kit: int                  # the kit the action happened in (gain) / left (kit)
    pad: int = -1
    old: int = 0
    new: int = 0
    to_kit: int = -1
    t: float = 0.0


@dataclass
class PadMixSession:
    link: Any
    clock: Callable[[], float] = time.monotonic
    kits: list[dict[str, Any]] = field(default_factory=list)
    kit: int = -1
    name: str = ""
    max: int = 127
    vols: list[int] = field(default_factory=lambda: [100] * PADS)
    samples: list[str | None] = field(default_factory=lambda: [None] * PADS)
    selected: int = 0
    history: list[Action] = field(default_factory=list)
    pending: dict[int, int] = field(default_factory=dict)
    # What the kit.json on the card holds, as far as this session knows.
    baseline: list[int] = field(default_factory=lambda: [100] * PADS)
    saved_kits: list[str] = field(default_factory=list)
    status: str = ""
    error: bool = False
    running_app: str | None = None
    _last_send: float = 0.0
    _last_change: float = 0.0
    _release_at: float = 0.0
    _release_pad: int = -1

    # -- reading the board --------------------------------------------------

    def start(self) -> None:
        self.kits = verbs.kit_list(self.link)["kits"]
        self._read_mix()
        self._read_app()
        self.say(f"kit {self.kit} {self.name}")

    def _read_mix(self) -> None:
        mix = verbs.pad_mix(self.link)
        self.kit, self.name, self.max = mix["kit"], mix["name"] or "", mix["max"]
        self.vols = [p["vol"] for p in mix["pads"]]
        self.samples = [p["sample"] for p in mix["pads"]]
        self.pending.clear()
        self.baseline = list(self.vols)

    @property
    def unsaved(self) -> bool:
        return self.vols != self.baseline

    def _read_app(self) -> None:
        self.running_app = verbs.app_list(self.link)["running"]

    def say(self, text: str, error: bool = False) -> None:
        self.status, self.error = text, error

    def kit_index(self, kit_id: int) -> int:
        for i, k in enumerate(self.kits):
            if k["id"] == kit_id:
                return i
        return -1

    # -- faders -------------------------------------------------------------

    def set_volume(self, pad: int, vol: int, *, record: bool = True) -> None:
        vol = max(0, min(self.max, vol))
        old = self.vols[pad]
        if vol == old:
            return
        now = self.clock()
        if record:
            top = self.history[-1] if self.history else None
            if (top and top.kind == "gain" and top.kit == self.kit and top.pad == pad
                    and now - top.t < MERGE_WITHIN_S):
                top.new, top.t = vol, now
                if top.new == top.old:
                    self.history.pop()
            else:
                self.history.append(Action("gain", self.kit, pad=pad, old=old, new=vol, t=now))
        self.vols[pad] = vol
        self.pending[pad] = vol
        self._last_change = now

    def nudge(self, delta: int) -> None:
        self.set_volume(self.selected, self.vols[self.selected] + delta)

    def select(self, pad: int) -> None:
        self.selected = max(0, min(PADS - 1, pad))

    # -- undo ---------------------------------------------------------------

    def undo(self) -> None:
        if not self.history:
            self.say("nothing to undo")
            return
        action = self.history.pop()
        if action.kind == "gain":
            if action.kit != self.kit:
                self.say(f"undo skipped: that move was in kit {action.kit}", error=True)
                return
            self.set_volume(action.pad, action.old, record=False)
            self.select(action.pad)
            self.say(f"undo: pad {action.pad + 1} back to {action.old}")
            return
        self.load_kit(action.kit, record=False)
        if self.kit == action.kit:
            self.say(f"undo: back to kit {self.kit} {self.name}")

    # -- the board ----------------------------------------------------------

    def flush(self) -> None:
        """Send the moved faders; write kit.json once they have settled."""
        now = self.clock()
        if self.pending and now - self._last_send >= SEND_EVERY_S:
            for pad, vol in sorted(self.pending.items()):
                verbs.pad_mix_set(self.link, pad, vol)
            self.pending.clear()
            self._last_send = now
        if self._release_pad >= 0 and now >= self._release_at:
            verbs.pad_release(self.link, self._release_pad)
            self._release_pad = -1
        if self.unsaved and not self.pending and now - self._last_change >= AUTOSAVE_AFTER_S:
            self.save()

    def save(self) -> bool:
        if self.pending:
            for pad, vol in sorted(self.pending.items()):
                verbs.pad_mix_set(self.link, pad, vol)
            self.pending.clear()
        if not self.unsaved:
            return True
        try:
            verbs.kit_save(self.link)
        except HilError as exc:
            self.say(f"kit.json NOT saved: {exc.message}", error=True)
            return False
        self.baseline = list(self.vols)
        if self.name not in self.saved_kits:
            self.saved_kits.append(self.name)
        self.say(f"saved {self.name}/kit.json")
        return True

    def load_kit(self, kit_id: int, *, record: bool = True) -> None:
        if kit_id == self.kit:
            return
        if not self.save():
            self.say(self.status + " -- s retries, Esc takes the moves back", error=True)
            return
        left = self.kit
        try:
            verbs.kit_load(self.link, kit_id, wait_s=30.0)
            self._read_mix()
            self._read_app()
        except HilError as exc:
            self.say(f"kit {kit_id} did not load: {exc.message}", error=True)
            return
        if record:
            self.history.append(Action("kit", left, to_kit=kit_id, t=self.clock()))
        self.say(f"kit {self.kit} {self.name}")

    def step_kit(self, delta: int) -> None:
        if not self.kits:
            return
        i = self.kit_index(self.kit)
        i = (i + delta) % len(self.kits) if i >= 0 else 0
        self.load_kit(self.kits[i]["id"])

    def audition(self, pad: int) -> None:
        # On the home screen's pad map a pad opens an app instead of playing.
        if self.running_app is None:
            self._read_app()
        if self.running_app is None:
            self.say("audition needs an app open on the board (the pads pick apps "
                     "on the home screen)", error=True)
            return
        if self._release_pad >= 0:
            verbs.pad_release(self.link, self._release_pad)
        verbs.pad_press(self.link, pad, 110)
        self._release_pad, self._release_at = pad, self.clock() + AUDITION_HOLD_S


# ── the screen ──────────────────────────────────────────────────────────

HELP_MIX = ("←→ pad  ↑↓ ±1  PgUp/PgDn ±8  space play  Enter kits  [ ] prev/next kit  "
            "Esc undo  s save  q quit  · mouse: drag a fader, wheel ±1, click the number plays")
HELP_KITS = ("↑↓ choose  Enter load  type to filter  Esc back  · mouse: click choose, "
             "double-click load, wheel scroll")


@dataclass
class _Layout:
    left: int
    col_w: int
    top: int          # value row
    track_top: int
    track_h: int
    num_row: int
    name_row: int


class Screen:
    """Curses drawing and input around a session."""

    def __init__(self, stdscr: Any, session: PadMixSession, curses: Any) -> None:
        self.scr, self.s, self.c = stdscr, session, curses
        self.mode = "mix"
        self.cursor = 0
        self.scroll = 0
        self.filter = ""
        self.drag_pad = -1
        self._last_click = (0.0, -1)
        self._layout: _Layout | None = None
        c = curses
        c.curs_set(0)
        stdscr.keypad(True)
        stdscr.timeout(25)
        c.mousemask(c.ALL_MOUSE_EVENTS | c.REPORT_MOUSE_POSITION)
        c.mouseinterval(0)
        # Button-event tracking: the terminal reports motion while a button is
        # held, which is what dragging a fader needs.
        print("\033[?1002h", end="", flush=True)
        if c.has_colors():
            c.start_color()
            c.use_default_colors()
            c.init_pair(1, c.COLOR_CYAN, -1)     # fader
            c.init_pair(2, c.COLOR_YELLOW, -1)   # selected fader
            c.init_pair(3, c.COLOR_BLUE, -1)     # empty track
            c.init_pair(4, c.COLOR_RED, -1)      # error
            c.init_pair(5, c.COLOR_GREEN, -1)    # saved
            c.init_pair(6, c.COLOR_MAGENTA, -1)  # unity line

    def close(self) -> None:
        print("\033[?1002l", end="", flush=True)

    # -- helpers --------------------------------------------------------------

    def _put(self, y: int, x: int, text: str, attr: int = 0) -> None:
        h, w = self.scr.getmaxyx()
        if y < 0 or y >= h or x >= w:
            return
        try:
            self.scr.addstr(y, x, text[: max(0, w - x - (1 if y == h - 1 else 0))], attr)
        except self.c.error:
            pass

    def _pair(self, n: int) -> int:
        return self.c.color_pair(n) if self.c.has_colors() else 0

    def visible_kits(self) -> list[dict[str, Any]]:
        f = self.filter.lower()
        return [k for k in self.s.kits if f in k["name"].lower()] if f else list(self.s.kits)

    # -- drawing --------------------------------------------------------------

    def draw(self) -> None:
        self.scr.erase()
        h, w = self.scr.getmaxyx()
        s = self.s
        state = (("unsaved", self._pair(2)) if s.unsaved
                 else ("saved", self._pair(5)))
        head = f" Pad Mix  kit {s.kit}: {s.name}  (0..{s.max})"
        self._put(0, 0, head, self.c.A_BOLD)
        self._put(0, max(len(head) + 2, w - 10), state[0], state[1] | self.c.A_BOLD)
        if self.mode == "mix":
            self._draw_mix(h, w)
            help_ = HELP_MIX
        else:
            self._draw_kits(h, w)
            help_ = HELP_KITS
        self._put(h - 2, 0, s.status, self._pair(4) | self.c.A_BOLD if s.error else 0)
        self._put(h - 1, 0, help_, self.c.A_DIM)
        self.scr.refresh()

    def _draw_mix(self, h: int, w: int) -> None:
        s = self.s
        col_w = max(3, min(6, (w - 6) // PADS))
        left = 5
        top = 2
        track_top = top + 1
        track_h = max(3, h - track_top - 6)
        num_row = track_top + track_h
        name_row = num_row + 1
        self._layout = _Layout(left, col_w, top, track_top, track_h, num_row, name_row)
        bar_w = max(1, col_w - 1)
        # The row the fill reaches at 100, unity gain -- a sample kit's 127 goes past it.
        unity = track_top + track_h - round(100 * track_h / s.max) if s.max > 100 else -1
        for row, label in ((track_top, str(s.max)), (track_top + track_h - 1, "0")):
            self._put(row, 0, f"{label:>4}", self.c.A_DIM)
        if unity >= track_top:
            self._put(unity, 0, " 100", self._pair(6))
        for pad in range(PADS):
            x = left + pad * col_w
            vol = s.vols[pad]
            sel = pad == s.selected
            fill = round(vol * track_h / s.max) if s.max else 0
            self._put(top, x, f"{vol:>{bar_w}}", self.c.A_BOLD if sel else 0)
            for r in range(track_h):
                y = track_top + track_h - 1 - r
                if r < fill:
                    self._put(y, x, "█" * bar_w, self._pair(2 if sel else 1))
                elif y == unity:
                    self._put(y, x, "╌" * bar_w, self._pair(6))
                else:
                    self._put(y, x, ("┃" if sel else "│").center(bar_w), self._pair(3))
            num_attr = self.c.A_REVERSE | self.c.A_BOLD if sel else 0
            self._put(num_row, x, f"{pad + 1:>{bar_w}}", num_attr)
        pad = s.selected
        sample = s.samples[pad] or "-"
        self._put(name_row, left,
                  f"pad {pad + 1}: {sample}   vol {s.vols[pad]}  {volume_db(s.vols[pad])}",
                  self.c.A_BOLD)
        undo = f"undo steps: {len(s.history)}"
        self._put(name_row + 1, left, undo, self.c.A_DIM)

    def _draw_kits(self, h: int, w: int) -> None:
        kits = self.visible_kits()
        rows = max(1, h - 5)
        self.cursor = max(0, min(self.cursor, len(kits) - 1)) if kits else 0
        if self.cursor < self.scroll:
            self.scroll = self.cursor
        if self.cursor >= self.scroll + rows:
            self.scroll = self.cursor - rows + 1
        self._put(1, 1, f"filter: {self.filter}_" if self.filter else
                  f"{len(kits)} kits on the card", self.c.A_DIM)
        for i, k in enumerate(kits[self.scroll:self.scroll + rows]):
            idx = self.scroll + i
            mark = "*" if k["id"] == self.s.kit else " "
            attr = self.c.A_REVERSE if idx == self.cursor else 0
            self._put(2 + i, 1, f"{mark} {k['id']:>3}  {k['name']}".ljust(w - 2), attr)

    # -- input ----------------------------------------------------------------

    def handle(self, ch: int) -> bool:
        """One key or mouse event; False to quit."""
        if ch == self.c.KEY_MOUSE:
            self._mouse()
            return True
        if ch == self.c.KEY_RESIZE:
            return True
        if ch == 27:
            ch = self._escape()
            if ch is None:
                return True
        return self._key_mix(ch) if self.mode == "mix" else self._key_kits(ch)

    def _escape(self) -> int | None:
        """A lone Esc, or a key sequence curses did not decode.

        A terminal that ignores keypad mode sends the arrows as ESC [ A, not
        the ESC O A terminfo names, and curses hands them over byte by byte --
        the '[' then read as "previous kit". Those are decoded here; any other
        sequence (Alt+key, an unknown function key) is dropped, never taken
        for Esc, which is undo.
        """
        seq = ""
        while len(seq) < 8:
            nxt = self.scr.getch()
            if nxt == -1:
                break
            seq += chr(nxt) if 0 <= nxt < 256 else "?"
            if seq[-1].isalpha() or seq[-1] == "~":
                break
        if not seq:
            return 27
        c = self.c
        keys = {"[A": c.KEY_UP, "[B": c.KEY_DOWN, "[C": c.KEY_RIGHT, "[D": c.KEY_LEFT,
                "OA": c.KEY_UP, "OB": c.KEY_DOWN, "OC": c.KEY_RIGHT, "OD": c.KEY_LEFT,
                "[5~": c.KEY_PPAGE, "[6~": c.KEY_NPAGE, "[1;2A": c.KEY_SR, "[1;2B": c.KEY_SF,
                "OM": c.KEY_ENTER}
        return keys.get(seq)

    def _enter_kits(self) -> None:
        self.mode = "kits"
        self.filter = ""
        self.cursor = max(0, self.s.kit_index(self.s.kit))

    def _key_mix(self, ch: int) -> bool:
        c, s = self.c, self.s
        if ch in (ord("q"), ord("Q")):
            return False
        if ch == c.KEY_LEFT:
            s.select(s.selected - 1)
        elif ch == c.KEY_RIGHT:
            s.select(s.selected + 1)
        elif ch == c.KEY_UP:
            s.nudge(1)
        elif ch == c.KEY_DOWN:
            s.nudge(-1)
        elif ch in (c.KEY_PPAGE, c.KEY_SR):
            s.nudge(BIG_STEP)
        elif ch in (c.KEY_NPAGE, c.KEY_SF):
            s.nudge(-BIG_STEP)
        elif ch == ord(" "):
            s.audition(s.selected)
        elif ch in (10, 13, c.KEY_ENTER):
            self._enter_kits()
        elif ch == 27:
            s.undo()
        elif ch == ord("s"):
            s.save()
        elif ch == ord("]"):
            self._busy("loading the next kit…")
            s.step_kit(1)
        elif ch == ord("["):
            self._busy("loading the previous kit…")
            s.step_kit(-1)
        return True

    def _key_kits(self, ch: int) -> bool:
        c = self.c
        kits = self.visible_kits()
        if ch == c.KEY_UP:
            self.cursor -= 1
        elif ch == c.KEY_DOWN:
            self.cursor += 1
        elif ch == c.KEY_PPAGE:
            self.cursor -= 10
        elif ch == c.KEY_NPAGE:
            self.cursor += 10
        elif ch in (10, 13, c.KEY_ENTER):
            if kits:
                self._load(kits[max(0, min(self.cursor, len(kits) - 1))]["id"])
        elif ch == 27:
            if self.filter:
                self.filter = ""
                self.cursor = max(0, self.s.kit_index(self.s.kit))
                return True
            else:
                self.mode = "mix"
        elif ch in (c.KEY_BACKSPACE, 127, 8):
            self.filter = self.filter[:-1]
        elif 32 <= ch < 127:
            self.filter += chr(ch)
            self.cursor = 0
        if kits:
            self.cursor = max(0, min(self.cursor, len(kits) - 1))
        return True

    def _busy(self, text: str) -> None:
        self.s.say(text)
        self.draw()

    def _load(self, kit_id: int) -> None:
        self._busy(f"loading kit {kit_id}…")
        self.s.load_kit(kit_id)
        self.mode = "mix"

    def _mouse(self) -> None:
        c = self.c
        try:
            _, x, y, _, bstate = c.getmouse()
        except c.error:
            return
        wheel_up = bstate & c.BUTTON4_PRESSED
        press = bstate & c.BUTTON1_PRESSED
        release = bstate & c.BUTTON1_RELEASED
        motion = bstate & c.REPORT_MOUSE_POSITION
        button5 = getattr(c, "BUTTON5_PRESSED", 0)
        # An ncurses without a fifth button (macOS's 6.0) reports the wheel's
        # downward turn as REPORT_MOUSE_POSITION -- motion's own bit. Motion
        # arrives only with a button held (mode 1002), so without a drag that
        # bit is the wheel.
        wheel_down = bstate & button5 if button5 else (motion and self.drag_pad < 0)
        if wheel_down:
            motion = 0
        if self.mode == "kits":
            self._mouse_kits(y, bool(press), bool(wheel_up), bool(wheel_down),
                             bool(bstate & c.BUTTON3_PRESSED))
            return
        lay = self._layout
        if lay is None:
            return
        if release:
            self.drag_pad = -1
            return
        if y == 0 and press:
            self._enter_kits()
            return
        pad = (x - lay.left) // lay.col_w if x >= lay.left else -1
        if motion and self.drag_pad >= 0:
            self._fader_to(self.drag_pad, y)
            return
        if not 0 <= pad < PADS:
            return
        if wheel_up or wheel_down:
            self.s.select(pad)
            self.s.nudge(1 if wheel_up else -1)
        elif press:
            self.s.select(pad)
            if y == lay.num_row:
                self.s.audition(pad)
            elif lay.top <= y < lay.num_row:
                self.drag_pad = pad
                self._fader_to(pad, y)

    def _fader_to(self, pad: int, y: int) -> None:
        lay, s = self._layout, self.s
        assert lay is not None
        bottom = lay.track_top + lay.track_h - 1
        if y < lay.track_top:
            vol = s.max
        elif y > bottom:
            vol = 0      # dragged below the track
        else:
            # The row under the pointer becomes the fader's top filled row.
            vol = round((bottom - y + 1) * s.max / lay.track_h)
        s.set_volume(pad, vol)

    def _mouse_kits(self, y: int, press: bool, up: bool, down: bool, right: bool) -> None:
        if right:
            self.mode = "mix"
            return
        if up or down:
            self.cursor += -3 if up else 3
            return
        if not press or y < 2:
            return
        idx = self.scroll + (y - 2)
        kits = self.visible_kits()
        if not 0 <= idx < len(kits):
            return
        now = time.monotonic()
        last_t, last_idx = self._last_click
        self._last_click = (now, idx)
        if idx == last_idx and now - last_t < DOUBLE_CLICK_S:
            self._load(kits[idx]["id"])
        else:
            self.cursor = idx


def _loop(stdscr: Any, session: PadMixSession, curses: Any) -> None:
    screen = Screen(stdscr, session, curses)
    try:
        session.start()
        while True:
            screen.draw()
            try:
                ch = stdscr.getch()
                if ch != -1 and not screen.handle(ch):
                    break
                # Drain a key repeat in one go, so the board sees the last value.
                while (ch := stdscr.getch()) != -1:
                    if not screen.handle(ch):
                        return
                session.flush()
            except HilError as exc:
                # A refusal (a kit loading on the board, a reply lost) is shown,
                # not fatal: the moves stay pending and go out on the next pass.
                session.say(f"{exc.code}: {exc.message}", error=True)
    except KeyboardInterrupt:
        pass       # Ctrl-C quits like q: what was moved is still saved below
    finally:
        try:
            session.save()
        finally:
            screen.close()


def run(link: Any) -> dict[str, Any]:
    """The TUI on `link` until q; returns what it saved."""
    try:
        import curses
    except ImportError as exc:      # Windows without the wheel
        raise HilError(ENV, "padmix needs curses", hint="pip install windows-curses") from exc
    import locale

    locale.setlocale(locale.LC_ALL, "")
    os.environ.setdefault("ESCDELAY", "25")   # Esc is undo: no second's wait for a sequence
    session = PadMixSession(link)
    curses.wrapper(_loop, session, curses)
    return {"kit": session.kit, "name": session.name, "saved": session.saved_kits,
            "unsaved": session.unsaved, "status": session.status}
