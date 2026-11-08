"""padmix: PAD_MIX / KIT_SAVE verbs and the session behind the TUI. No hardware, no curses."""
from __future__ import annotations

import pytest

from crosspad_hil import padmix, verbs
from crosspad_hil.errors import HilError
from crosspad_hil.parsers import parse_cdc_reply
from tests.test_verbs import make_link  # noqa: F401  (fixture)

_MIX = ["PADMIX: kit=3 max=127 loading=0 name=LIVE ROOTS"] + [
    f"PADMIX: pad={i} vol={100 - i} pan=0 sample={'-' if i == 15 else f'{i:02}.wav'}"
    for i in range(16)
] + ["PADMIX: end count=16"]


# ── parsers / verbs ─────────────────────────────────────────────────────

def test_parse_padmix_rows() -> None:
    assert parse_cdc_reply("PADMIX: kit=3 max=100 loading=1 name=A B") == {
        "kind": "padmix", "kit": 3, "max": 100, "loading": True, "name": "A B"}
    assert parse_cdc_reply("PADMIX: pad=4 vol=90 pan=-3 sample=kick 1.wav") == {
        "kind": "padmix", "pad": 4, "vol": 90, "pan": -3, "sample": "kick 1.wav"}
    assert parse_cdc_reply("PADMIX: end count=16") == {"kind": "padmix", "end": True, "count": 16}
    assert parse_cdc_reply("KITSAVE: busy=0 ok=1 kit=3") == {
        "kind": "kitsave", "busy": 0, "ok": 1, "kit": 3}


def test_pad_mix_reads_sixteen_pads(make_link) -> None:  # noqa: F811
    link = make_link([("PAD_MIX", _MIX)])
    mix = verbs.pad_mix(link)
    assert (mix["kit"], mix["name"], mix["max"], mix["loading"]) == (3, "LIVE ROOTS", 127, False)
    assert [p["vol"] for p in mix["pads"]] == [100 - i for i in range(16)]
    assert mix["pads"][15]["sample"] is None


def test_pad_mix_short_listing_is_an_error(make_link) -> None:  # noqa: F811
    link = make_link([("PAD_MIX", _MIX[:5] + ["PADMIX: end count=16"])])
    with pytest.raises(HilError) as ei:
        verbs.pad_mix(link)
    assert ei.value.code == "NOT_SUPPORTED"


def test_pad_mix_set_and_its_refusal(make_link) -> None:  # noqa: F811
    link = make_link([("PAD_MIX 2 90", "OK"), ("PAD_MIX 2 1", "ERR loading")])
    assert verbs.pad_mix_set(link, 2, 90) == {"ok": True}
    with pytest.raises(HilError) as ei:
        verbs.pad_mix_set(link, 2, 1)
    assert ei.value.code == "BAD_ARGS"
    with pytest.raises(HilError):
        verbs.pad_mix_set(link, 16, 1)          # checked before a byte is written
    assert link.fake.written == ["PAD_MIX 2 90", "PAD_MIX 2 1"]


def test_kit_save_waits_for_the_write(make_link) -> None:  # noqa: F811
    link = make_link([("KIT_SAVE", "OK")], sequences={"KIT_SAVE_STATUS": [
        "KITSAVE: busy=1 ok=-1 kit=3",
        "KITSAVE: busy=0 ok=1 kit=3",
    ]})
    assert verbs.kit_save(link) == {"busy": 0, "ok": 1, "kit": 3}


def test_kit_save_failure(make_link) -> None:  # noqa: F811
    link = make_link([("KIT_SAVE", "OK"), ("KIT_SAVE_STATUS", "KITSAVE: busy=0 ok=0 kit=3")])
    with pytest.raises(HilError):
        verbs.kit_save(link)


# ── the session ─────────────────────────────────────────────────────────

class FakeBoard:
    """The verbs padmix calls, against a model of two kits."""

    def __init__(self) -> None:
        self.kits = {3: [100] * 16, 4: [80] * 16}
        self.card = {k: list(v) for k, v in self.kits.items()}
        self.current = 3
        self.calls: list[tuple] = []

    def install(self, mp: pytest.MonkeyPatch) -> None:
        mp.setattr(verbs, "kit_list", lambda link: {
            "kits": [{"id": 3, "name": "A"}, {"id": 4, "name": "B"}], "current": self.current})
        mp.setattr(verbs, "app_list", lambda link: {"apps": [], "running": "Sampler"})
        mp.setattr(verbs, "pad_mix", self.pad_mix)
        mp.setattr(verbs, "pad_mix_set", self.pad_mix_set)
        mp.setattr(verbs, "kit_save", self.kit_save)
        mp.setattr(verbs, "kit_load", self.kit_load)
        mp.setattr(verbs, "pad_press", lambda link, p, v=127: self.calls.append(("press", p)))
        mp.setattr(verbs, "pad_release", lambda link, p: self.calls.append(("release", p)))

    def pad_mix(self, link):
        return {"kit": self.current, "name": "AB"[self.current - 3], "max": 127,
                "loading": False,
                "pads": [{"pad": i, "vol": v, "pan": 0, "sample": f"{i}.wav"}
                         for i, v in enumerate(self.kits[self.current])]}

    def pad_mix_set(self, link, pad, vol):
        self.calls.append(("set", pad, vol))
        self.kits[self.current][pad] = vol

    def kit_save(self, link, wait_s=5.0):
        self.calls.append(("save", self.current))
        self.card[self.current] = list(self.kits[self.current])
        return {"busy": 0, "ok": 1, "kit": self.current}

    def kit_load(self, link, kit_id, wait_s=15.0):
        self.calls.append(("load", kit_id))
        self.kits[self.current] = list(self.card[self.current])   # unsaved moves are lost
        self.current = kit_id


class Clock:
    def __init__(self) -> None:
        self.t = 100.0

    def __call__(self) -> float:
        return self.t


@pytest.fixture
def board(monkeypatch: pytest.MonkeyPatch) -> FakeBoard:
    b = FakeBoard()
    b.install(monkeypatch)
    return b


def _session(clock: Clock) -> padmix.PadMixSession:
    s = padmix.PadMixSession(link=None, clock=clock)
    s.start()
    return s


def test_a_held_key_is_one_undo_step_and_one_write(board: FakeBoard) -> None:
    clock = Clock()
    s = _session(clock)
    for _ in range(5):
        s.nudge(-1)
        clock.t += 0.05
    assert s.vols[0] == 95 and len(s.history) == 1
    s.flush()
    assert board.calls == [("set", 0, 95)]           # coalesced: the last value only
    s.undo()
    clock.t += padmix.SEND_EVERY_S
    s.flush()
    assert s.vols[0] == 100 and board.kits[3][0] == 100
    assert not s.unsaved                             # back to what the card holds


def test_autosave_after_the_faders_settle(board: FakeBoard) -> None:
    clock = Clock()
    s = _session(clock)
    s.select(2)
    s.nudge(10)
    s.flush()
    assert ("save", 3) not in board.calls
    clock.t += padmix.AUTOSAVE_AFTER_S + 0.1
    s.flush()
    assert board.card[3][2] == 110 and not s.unsaved and s.saved_kits == ["A"]


def test_volume_is_clamped_to_the_kit_range(board: FakeBoard) -> None:
    s = _session(Clock())
    s.nudge(500)
    assert s.vols[0] == 127
    s.nudge(-500)
    assert s.vols[0] == 0


def test_kit_change_saves_first_and_undo_goes_back(board: FakeBoard) -> None:
    clock = Clock()
    s = _session(clock)
    s.nudge(-20)
    s.load_kit(4)
    assert board.card[3][0] == 80                    # saved before leaving
    assert s.kit == 4 and s.vols == [80] * 16
    clock.t += 5
    s.nudge(5)
    s.undo()                                         # the move in kit 4
    assert s.vols[0] == 80 and s.kit == 4
    s.undo()                                         # the kit change
    assert s.kit == 3 and s.vols[0] == 80
    s.undo()                                         # the move in kit 3
    assert s.vols[0] == 100
    s.save()
    assert board.card[3][0] == 100


def test_step_kit_wraps(board: FakeBoard) -> None:
    s = _session(Clock())
    s.step_kit(1)
    assert s.kit == 4
    s.step_kit(1)
    assert s.kit == 3


def test_audition_presses_then_releases(board: FakeBoard) -> None:
    clock = Clock()
    s = _session(clock)
    s.audition(5)
    clock.t += padmix.AUDITION_HOLD_S + 0.01
    s.flush()
    assert board.calls == [("press", 5), ("release", 5)]


def test_audition_refused_on_the_home_screen(board: FakeBoard,
                                             monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(verbs, "app_list", lambda link: {"apps": [], "running": None})
    s = _session(Clock())
    s.audition(5)
    assert board.calls == [] and s.error


def test_volume_db() -> None:
    assert padmix.volume_db(100) == "+0.0 dB"
    assert padmix.volume_db(50) == "-6.0 dB"
    assert padmix.volume_db(0) == "-inf dB"
