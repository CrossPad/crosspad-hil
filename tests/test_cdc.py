"""CdcLink: reader thread + prefix waiters (hil_kit_churn.Cdc), knowledge-derived expects,
echo → UNKNOWN_VERB, timeout hints, multi-line replies, burst drop accounting.

Reply formats are copied from platform-idf/main/hil_control.cpp snprintf strings.
"""
from __future__ import annotations

import itertools
import threading
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest

from crosspad_hil import cdc as cdc_mod
from crosspad_hil.cdc import BurstResult, CdcLink, Reply
from crosspad_hil.errors import BAD_ARGS, NO_DEVICE, NOT_SUPPORTED, TIMEOUT, UNKNOWN_VERB, HilError
from tests.fakes import FakeSerial

_PORT_SEQ = itertools.count()

KITSTATUS = "KITSTATUS: current=3 loading=0 pending=-1 name=DRUMS"
CDCSTATS_0 = "CDCSTATS: rx=100 cmds=98 drop=1"
CDCSTATS_1 = "CDCSTATS: rx=140 cmds=130 drop=3"
ENCGROUP = [
    "ENCGROUP: count=3",
    "  [0] 0x3fcb0001 Back",
    "  [1] 0x3fcb0002 Load",
    "  [2] 0x3fcb1234 Kit: DRUMS",
]
APPVER = [
    "APPVER: crosspad-sampler id=sampler commit=abc1234 ref=v1.2 dirty=0",
    "APPVER: crosspad-fishtank id=- commit=0123abc ref=- dirty=0",
    "APPVER: end count=2",
]
MEMBLOCKS = [
    "MEMBLK: biggest_used=4096 | <=64 u=120/5000B f=3/100B | <=big u=1/16384B f=0/0B",
    "MEMBIG: @3fc9a000=16384",
]

# Minimal cdc.yaml-shaped knowledge so the expect-derivation tests do not depend on Task 2.
CDC_KNOWLEDGE: dict = {
    "verbs": {
        "KIT_STATUS": {"args": [], "reply": "KITSTATUS:", "end": None, "profile": "default"},
        "KIT_LOAD": {"args": ["kit_id:int"], "reply": "OK", "end": None, "profile": "default"},
        "PAD_PRESS": {"args": ["idx:int", "vel:int?"], "reply": "OK", "end": None,
                      "profile": "default"},
        "ENC_GROUP": {"args": [], "reply": "multi", "prefix": "ENCGROUP:", "end": None,
                      "profile": "default"},
        "APP_VERSIONS": {"args": [], "reply": "multi", "prefix": "APPVER:", "end": "APPVER: end",
                         "profile": "default"},
        "MEM_BLOCKS": {"args": [], "reply": "multi", "prefix": "MEMBLK:", "end": "MEMBIG:",
                       "profile": "default"},
        "CDC_STATS": {"args": [], "reply": "CDCSTATS:", "end": None, "profile": "default"},
        "USB_AUDIO": {"args": [], "reply": None, "end": None, "profile": "default"},
    }
}


class SequencedFake(FakeSerial):
    """FakeSerial whose replies for one command are consumed in order (last one repeats).

    FakeSerial maps a command to one fixed reply; CDC_STATS before/after a burst needs two.
    """

    def __init__(self, sequences: dict[str, list[str]]) -> None:
        super().__init__([], timeout=0.01)
        self._sequences = {k: list(v) for k, v in sequences.items()}

    def write(self, data: bytes) -> int:
        text = data.decode("utf-8", errors="replace")
        for raw in text.split("\n"):
            cmd = raw.rstrip("\r")
            if not cmd:
                continue
            self.written.append(cmd)
            seq = self._sequences.get(cmd)
            if seq is None:
                self._rx.append((cmd + "\r\n").encode())
                continue
            reply = seq.pop(0) if len(seq) > 1 else seq[0]
            self._rx.append((reply + "\r\n").encode())
        return len(data)


class SilentFake(FakeSerial):
    """A port that swallows everything — the CDC endpoint of a board mid-reboot."""

    def write(self, data: bytes) -> int:
        self.written.append(data.decode("utf-8", errors="replace").rstrip("\r\n"))
        return len(data)


@pytest.fixture(autouse=True)


def fast_constants(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv("XDG_RUNTIME_DIR", str(tmp_path))
    monkeypatch.setattr(cdc_mod, "OPEN_SETTLE_S", 0.0)
    monkeypatch.setattr(cdc_mod, "BURST_SETTLE_S", 0.01)


@pytest.fixture
def make_link() -> Callable[..., tuple[CdcLink, FakeSerial]]:
    opened: list[CdcLink] = []

    def _make(
        script: list[tuple[str, str | list[str]]] | None = None,
        *,
        fake: FakeSerial | None = None,
        knowledge: dict | None = CDC_KNOWLEDGE,
    ) -> tuple[CdcLink, FakeSerial]:
        ser = fake if fake is not None else FakeSerial(script or [], timeout=0.01)
        link = CdcLink(
            f"/dev/fake-cdc-{next(_PORT_SEQ)}",
            serial_factory=lambda path, **kw: ser,
            knowledge=knowledge,
        )
        link.open()
        opened.append(link)
        return link, ser

    yield _make
    for link in opened:
        link.close()


# ── hygiene / lifecycle ─────────────────────────────────────────────────────
def test_open_deasserts_dtr_rts_and_locks_port(tmp_path: Path) -> None:
    from crosspad_hil.locks import PortLock
    from crosspad_hil.serial_open import open_serial

    fake = FakeSerial([], timeout=0.01)
    link = CdcLink(
        "/dev/fake-cdc",
        serial_factory=lambda path, **kw: open_serial(path, serial_cls=lambda: fake, **kw),
        knowledge=CDC_KNOWLEDGE,
    )
    link.open()
    try:
        assert fake.is_open
        assert all(value is False for _n, value in fake.control_history)
        assert [(h["port"], h["purpose"]) for h in PortLock.holders()] == [("/dev/fake-cdc", "cdc")]
    finally:
        link.close()
    assert PortLock.holders() == []
    assert fake.is_open is False


def test_send_is_fire_and_forget_with_newline(make_link: Callable[..., Any]) -> None:
    link, fake = make_link([("PAD_PRESS 3 100", "OK")])
    link.send("PAD_PRESS 3 100")
    assert fake.written == ["PAD_PRESS 3 100"]
    time.sleep(0.05)
    assert link.unmatched() == ["OK"]  # nobody waited for it; it is not lost, just unclaimed


# ── "OK is not your ack" ────────────────────────────────────────────────────
def test_reply_matched_by_prefix_while_unrelated_ok_lines_interleave(
    make_link: Callable[..., Any],
) -> None:
    link, fake = make_link([("KIT_STATUS", KITSTATUS)])
    # Pad traffic that arrived just before our command's reply.
    fake.feed(["OK", "OK", "SMPLPEAK: 1200 free=7"])
    reply = link.transact("KIT_STATUS")
    assert isinstance(reply, Reply)
    assert reply.line == KITSTATUS
    assert reply.parsed is not None and reply.parsed["current"] == 3
    assert reply.parsed["loading"] is False and reply.parsed["pending"] == -1
    assert reply.rtt_ms >= 0.0
    assert reply.extra_lines == ["OK", "OK", "SMPLPEAK: 1200 free=7"]
    assert link.unmatched() == ["OK", "OK", "SMPLPEAK: 1200 free=7"]


def test_ok_verb_accepts_ok_or_err(make_link: Callable[..., Any]) -> None:
    link, fake = make_link([("KIT_LOAD 3", "OK"), ("KIT_LOAD 99", "ERR bad kit id")])
    assert link.transact("KIT_LOAD 3").parsed == {"kind": "ok"}
    err = link.transact("KIT_LOAD 99")
    assert err.line == "ERR bad kit id"
    assert err.parsed is not None and err.parsed["kind"] == "err"


def test_concurrent_transacts_each_get_their_own_reply(make_link: Callable[..., Any]) -> None:
    link, fake = make_link([("KIT_STATUS", KITSTATUS), ("CDC_STATS", CDCSTATS_0)])
    out: dict[str, str] = {}

    def worker(cmd: str) -> None:
        out[cmd] = link.transact(cmd).line

    threads = [threading.Thread(target=worker, args=(c,)) for c in ("KIT_STATUS", "CDC_STATS")]
    for t in threads:
        t.start()
    for t in threads:
        t.join(2.0)
    assert out == {"KIT_STATUS": KITSTATUS, "CDC_STATS": CDCSTATS_0}


# ── expect derivation from knowledge ────────────────────────────────────────
def test_transact_derives_expect_from_knowledge(make_link: Callable[..., Any]) -> None:
    link, fake = make_link([("KIT_STATUS", KITSTATUS)])
    assert link.transact("KIT_STATUS").line == KITSTATUS
    assert link.transact("KIT_STATUS", expect="KITSTATUS:").line == KITSTATUS


def test_transact_multi_verb_without_expect_uses_prefix(make_link: Callable[..., Any]) -> None:
    link, fake = make_link([("ENC_GROUP", ENCGROUP)])
    reply = link.transact("ENC_GROUP")
    assert reply.line == "ENCGROUP: count=3"


def test_transact_no_reply_verb_is_fire_and_forget(make_link: Callable[..., Any]) -> None:
    link, fake = make_link(fake=SilentFake([], timeout=0.01))
    reply = link.transact("USB_AUDIO")
    assert reply == Reply(line="", parsed=None, rtt_ms=0.0, extra_lines=[])
    assert fake.written == ["USB_AUDIO"]


def test_transact_verb_absent_from_knowledge_raises_before_sending(
    make_link: Callable[..., Any],
) -> None:
    link, fake = make_link([])
    with pytest.raises(HilError) as ei:
        link.transact("NOT_A_VERB 1")
    assert ei.value.code == UNKNOWN_VERB
    assert "expect=" in (ei.value.hint or "")
    assert fake.written == []


def test_default_knowledge_is_cdc_yaml(make_link: Callable[..., Any]) -> None:
    link, fake = make_link([("LED_STATE", "LEDS: bri=80 anim=0 coalesce=1 cfgbri=80 pwr=0x00 "
                              "pwrN=3 txfail=0 colors=" + ",".join(["000000"] * 16))],
                           knowledge=None)
    assert link.transact("LED_STATE").line.startswith("LEDS:")


# ── echo → UNKNOWN_VERB ─────────────────────────────────────────────────────
def test_echo_of_sent_line_is_unknown_verb(make_link: Callable[..., Any]) -> None:
    # FakeSerial echoes unscripted commands, exactly like cdc_dispatch_one() when no
    # handler matched.
    link, fake = make_link([])
    with pytest.raises(HilError) as ei:
        link.transact("KIT_STATUS", timeout_s=1.0)
    assert ei.value.code == UNKNOWN_VERB
    assert ei.value.details["cmd"] == "KIT_STATUS"
    assert fake.written == ["KIT_STATUS"]


def test_echo_with_explicit_expect_is_still_unknown_verb(make_link: Callable[..., Any]) -> None:
    link, fake = make_link([])
    with pytest.raises(HilError) as ei:
        link.transact("FOO_BAR 7", expect="OK", timeout_s=1.0)
    assert ei.value.code == UNKNOWN_VERB


# ── timeouts ────────────────────────────────────────────────────────────────
def test_timeout_with_zero_bytes_hints_uac2(make_link: Callable[..., Any]) -> None:
    link, fake = make_link(fake=SilentFake([], timeout=0.01))
    t0 = time.monotonic()
    with pytest.raises(HilError) as ei:
        link.transact("KIT_STATUS", timeout_s=0.1)
    assert ei.value.code == TIMEOUT
    assert "UAC2" in (ei.value.hint or "")
    assert ei.value.details["rx_bytes"] == 0
    assert 0.09 <= time.monotonic() - t0 < 1.0  # 10ms slack: Windows time.sleep() undershoots


def test_timeout_with_traffic_does_not_blame_uac2(make_link: Callable[..., Any]) -> None:
    fake = SilentFake([], timeout=0.01)
    link, _ = make_link(fake=fake)
    threading.Timer(0.02, lambda: fake.feed(["OK", "OK"])).start()
    with pytest.raises(HilError) as ei:
        link.transact("KIT_STATUS", timeout_s=0.2)
    assert ei.value.code == TIMEOUT
    assert "UAC2" not in (ei.value.hint or "")
    assert ei.value.details["rx_bytes"] > 0
    assert link.unmatched() == ["OK", "OK"]


# ── transact_multi ──────────────────────────────────────────────────────────
def test_transact_multi_enc_group_ends_on_idle(make_link: Callable[..., Any]) -> None:
    link, fake = make_link([("ENC_GROUP", ENCGROUP)])
    t0 = time.monotonic()
    lines = link.transact_multi("ENC_GROUP", "ENCGROUP:", end=None, idle_ms=100)
    assert lines == [
        "ENCGROUP: count=3",
        "[0] 0x3fcb0001 Back",
        "[1] 0x3fcb0002 Load",
        "[2] 0x3fcb1234 Kit: DRUMS",
    ]
    assert 0.09 <= time.monotonic() - t0 < 1.0  # 10ms slack: Windows time.sleep() undershoots
    from crosspad_hil.parsers import parse_enc_group

    assert [e["label"] for e in parse_enc_group(lines)] == ["Back", "Load", "Kit: DRUMS"]


def test_parse_enc_group_reads_the_label_past_the_widget_box() -> None:
    from crosspad_hil.parsers import parse_enc_group

    lines = [
        "ENCGROUP: count=2",
        "  [0] 0x3fcb0001 x=4 y=40 w=72 h=-1 Sampler",
        "  [1] 0x3fcb0002 x=-12 y=0 w=320 h=36 Kit: DRUMS",
    ]
    entries = parse_enc_group(lines)
    assert [e["label"] for e in entries] == ["Sampler", "Kit: DRUMS"]
    assert (entries[1]["x"], entries[1]["w"], entries[0]["h"]) == (-12, 320, -1)


def test_transact_multi_with_end_pattern_filters_interleaved_traffic(
    make_link: Callable[..., Any],
) -> None:
    link, fake = make_link([("APP_VERSIONS", [APPVER[0], "OK", APPVER[1], APPVER[2]])])
    t0 = time.monotonic()
    lines = link.transact_multi("APP_VERSIONS", "APPVER:", end="APPVER: end", idle_ms=500)
    assert lines == APPVER
    assert time.monotonic() - t0 < 0.4  # returned on the end line, not on idle


def test_fs_list_of_an_empty_directory_is_its_end_line_alone(make_link: Callable[..., Any]) -> None:
    from crosspad_hil import verbs
    link, _ = make_link([("FS_LIST /sdcard/SONGS/a/clips", ["FS_END"])])
    assert verbs.fs_list(link, "/sdcard/SONGS/a/clips")["entries"] == []


def test_fs_list_cut_off_before_its_end_is_a_timeout_not_a_short_listing(
    make_link: Callable[..., Any],
) -> None:
    from crosspad_hil import verbs
    link, _ = make_link([("FS_LIST /sdcard/SONGS", ["FS_LIST: D 0 a", "FS_LIST: D 0 b"])])
    with pytest.raises(HilError) as ei:
        verbs.fs_list(link, "/sdcard/SONGS", timeout_s=0.3)
    assert ei.value.code == TIMEOUT


def test_transact_multi_mem_blocks_two_lines(make_link: Callable[..., Any]) -> None:
    link, fake = make_link([("MEM_BLOCKS", MEMBLOCKS)])
    assert link.transact_multi("MEM_BLOCKS", "MEMBLK:", end="MEMBIG:") == MEMBLOCKS


def test_transact_multi_echo_and_timeout(make_link: Callable[..., Any]) -> None:
    link, fake = make_link([])
    with pytest.raises(HilError) as ei:
        link.transact_multi("ENC_GROUP", "ENCGROUP:", timeout_s=0.5)
    assert ei.value.code == UNKNOWN_VERB
    link2, _ = make_link(fake=SilentFake([], timeout=0.01))
    with pytest.raises(HilError) as ei:
        link2.transact_multi("ENC_GROUP", "ENCGROUP:", timeout_s=0.1)
    assert ei.value.code == TIMEOUT


# ── burst ───────────────────────────────────────────────────────────────────
def test_burst_accounts_drops_from_cdc_stats(make_link: Callable[..., Any]) -> None:
    fake = SequencedFake({"CDC_STATS": [CDCSTATS_0, CDCSTATS_1], "PAD_PRESS 0 100": ["OK"],
                          "PAD_RELEASE 0": ["OK"]})
    link, _ = make_link(fake=fake)
    cmds = ["PAD_PRESS 0 100", "PAD_RELEASE 0"] * 3
    t0 = time.monotonic()
    res = link.burst(cmds, rate_hz=100.0)
    elapsed = time.monotonic() - t0
    assert isinstance(res, BurstResult)
    assert res.sent == 6
    assert res.drops_before == 1 and res.drops_after == 3 and res.dropped == 2
    assert 0.045 <= res.seconds <= elapsed  # 5ms slack: Windows time.sleep() undershoots
    assert fake.written[0] == "CDC_STATS" and fake.written[-1] == "CDC_STATS"
    assert fake.written[1:-1] == cmds


def test_burst_paces_at_rate(make_link: Callable[..., Any]) -> None:
    fake = SequencedFake({"CDC_STATS": [CDCSTATS_0, CDCSTATS_0]})
    link, _ = make_link(fake=fake)
    res = link.burst(["PAD_PRESS 1 90"] * 5, rate_hz=50.0)  # 4 intervals of 20 ms
    assert res.dropped == 0
    assert 0.075 <= res.seconds < 0.5  # 5ms slack: Windows time.sleep() undershoots


def test_burst_rejects_non_positive_rate(make_link: Callable[..., Any]) -> None:
    link, _ = make_link([])
    with pytest.raises(HilError) as ei:
        link.burst(["PAD_PRESS 0"], rate_hz=0)
    assert ei.value.code == BAD_ARGS


# ── unmatched ring ──────────────────────────────────────────────────────────
def test_unmatched_is_bounded(make_link: Callable[..., Any]) -> None:
    link, fake = make_link([])
    fake.feed([f"noise {i}" for i in range(250)])
    time.sleep(0.1)
    lines = link.unmatched()
    assert len(lines) == cdc_mod.UNMATCHED_RING == 200
    assert lines[0] == "noise 50" and lines[-1] == "noise 249"


def test_close_is_idempotent(make_link: Callable[..., Any]) -> None:
    link, fake = make_link([])
    link.close()
    link.close()
    assert fake.is_open is False


def test_open_waits_for_the_port_to_come_back_after_a_reset() -> None:
    """The ESP re-enumerates after a reset, so its CDC node is briefly absent.

    Opening the STM bridge VCP is itself a reset on this hardware (observed:
    'rst:0x1 (POWERON)' in the console right before the CDC open failed), so a
    scenario that opens the console and then the CDC link hits this every run.
    """
    attempts: list[int] = []
    ser = FakeSerial([("MEM", "MEM: int_free=1")], timeout=0.01)

    def flaky_factory(path: str, **kw: object) -> FakeSerial:
        attempts.append(1)
        if len(attempts) < 3:
            raise FileNotFoundError(2, "No such file or directory", path)
        return ser

    link = CdcLink(f"/dev/fake-cdc-{next(_PORT_SEQ)}", serial_factory=flaky_factory)
    link.open(wait_s=5.0)
    try:
        assert len(attempts) == 3
        assert link.transact("MEM", timeout_s=1.0).parsed["int_free"] == 1
    finally:
        link.close()


def test_open_gives_up_with_no_device_and_a_hint() -> None:
    def never(path: str, **kw: object) -> FakeSerial:
        raise FileNotFoundError(2, "No such file or directory", path)

    link = CdcLink(f"/dev/fake-cdc-{next(_PORT_SEQ)}", serial_factory=never)
    with pytest.raises(HilError) as ei:
        link.open(wait_s=0.5)
    assert ei.value.code == NO_DEVICE
    assert "re-enumerates" in (ei.value.hint or "")


# ── family error replies (FS_ERR / ASSETS_ERROR, main/file_transfer.cpp) ─────
def test_transact_claims_family_error_as_the_answer(make_link: Callable[..., Any]) -> None:
    """FS_STAT expects "FS_STAT …"; "FS_ERR …" is the refusal, not silence."""
    link, _ = make_link([("FS_STAT /sdcard/x", "FS_ERR stat_2")])
    reply = link.transact("FS_STAT /sdcard/x", "FS_STAT", timeout_s=0.5)
    assert reply.line == "FS_ERR stat_2"
    assert reply.parsed == {"kind": "err", "family": "FS", "message": "stat_2"}


def test_transact_multi_raises_on_family_error_instead_of_timing_out(
    make_link: Callable[..., Any],
) -> None:
    link, _ = make_link([("FS_ROOTS", "FS_ERR no_worker int_largest=3700 need=5120")])
    t0 = time.monotonic()
    with pytest.raises(HilError) as ei:
        link.transact_multi("FS_ROOTS", "FS_ROOTS:", end="FS_END", timeout_s=2.0)
    assert ei.value.code == NOT_SUPPORTED
    assert "no_worker" in str(ei.value)
    assert time.monotonic() - t0 < 1.0  # the refusal ends the wait, the timeout does not


def test_transact_multi_error_after_first_row_is_someone_elses(
    make_link: Callable[..., Any],
) -> None:
    rows = ["FS_LIST: F 10 a.wav", "ERR unknown", "FS_LIST: D 0 kits", "FS_END"]
    link, _ = make_link([("FS_LIST /sdcard", rows)])
    lines = link.transact_multi("FS_LIST /sdcard", "FS_LIST:", end="FS_END", timeout_s=1.0)
    assert lines == ["FS_LIST: F 10 a.wav", "FS_LIST: D 0 kits", "FS_END"]


def test_a_verb_that_blocks_has_its_own_timeout() -> None:
    """STM_FW_UPDATE reflashes the STM and answers after ~10 s: 2 s gave up early."""
    link = CdcLink("fake-timeout", serial_factory=lambda path, **kw: FakeSerial([]))
    assert link._timeout_for("STM_FW_UPDATE", None) == 30.0
    assert link._timeout_for("KIT_STATUS", None) == 2.0
    assert link._timeout_for("STM_FW_UPDATE", 5.0) == 5.0          # the caller still wins


# ── continuation lines (sent starting with a space) ─────────────────────────
COREDUMP = ["COREDUMP: seq=3 task=main pc=0x40381234 cause=28 vaddr=0x00000000 depth=3",
            " bt: 0x40381234 0x40382000 0x40383000",
            " exc: tcb=0x3fc9a000 a0=0x80381234 a1=0x3fcbbb00"]
RSTLOG = ["RSTLOG: n=2 coredump_seq=3",
          " #2 PANIC stm_last=0x00 stm_total=0 stm_wd=0 prev=12m app=Sampler COREDUMP",
          " #1 POWERON stm_last=0x00 stm_total=0 stm_wd=0 prev=0m app=-"]


def test_a_reply_s_continuation_lines_are_its_own(make_link: Callable[..., Any]) -> None:
    """COREDUMP_INFO's " bt:" / " exc:" and RESET_LOG's " #n" rows came after the
    reply's first line, and were lost to whoever was not waiting any more."""
    link, _ = make_link([("COREDUMP_INFO", COREDUMP), ("RESET_LOG", RSTLOG)], knowledge=None)
    r = link.transact("COREDUMP_INFO")
    assert r.line == COREDUMP[0]
    assert r.more_lines == ["bt: 0x40381234 0x40382000 0x40383000",
                            "exc: tcb=0x3fc9a000 a0=0x80381234 a1=0x3fcbbb00"]
    assert r.extra_lines == [] and link.unmatched() == []
    r = link.transact("RESET_LOG")
    assert r.parsed["n"] == 2 and [m.split()[0] for m in r.more_lines] == ["#2", "#1"]


def test_continuation_lines_are_collected_only_when_asked(make_link: Callable[..., Any]) -> None:
    link, _ = make_link([("KIT_STATUS", ["KITSTATUS: current=1 loading=0 pending=-1 name=A",
                                         " stray"])])
    assert link.transact("KIT_STATUS").more_lines == []      # cdc.yaml: no continuation
    r = link.transact("KIT_STATUS", follow=True)
    assert r.more_lines == ["stray"]


def test_a_space_line_after_someone_else_s_reply_is_not_taken(
        make_link: Callable[..., Any]) -> None:
    link, fake = make_link([("COREDUMP_INFO", COREDUMP[:1])], knowledge=None)
    fake.feed(["I (123) console: boot", " indented console text"])
    r = link.transact("COREDUMP_INFO")
    assert r.more_lines == []


def test_coredump_info_still_finds_the_backtrace(make_link: Callable[..., Any]) -> None:
    from crosspad_hil import verbs

    link, _ = make_link([("COREDUMP_INFO", COREDUMP)], knowledge=None)
    info = verbs.coredump_info(link)
    assert info["present"] and info["seq"] == 3
    assert info["backtrace"] == "bt: 0x40381234 0x40382000 0x40383000"


# ── shared with the hub: resolve_expect, verb_timeout_s, purpose, reader_alive ──

def test_resolve_expect_derives_from_knowledge() -> None:
    verbs = CDC_KNOWLEDGE["verbs"]
    assert cdc_mod.resolve_expect(verbs, "KIT_STATUS", None) == ("KITSTATUS:", "ERR")
    assert cdc_mod.resolve_expect(verbs, "KIT_LOAD 3", None) == ("OK", "ERR")
    assert cdc_mod.resolve_expect(verbs, "APP_VERSIONS", None) == ("APPVER:", "ERR")
    assert cdc_mod.resolve_expect(verbs, "USB_AUDIO", None) is None
    assert cdc_mod.resolve_expect(verbs, "WHATEVER", "FOO:") == ("FOO:", "ERR")


def test_resolve_expect_unknown_verb_raises() -> None:
    with pytest.raises(HilError) as e:
        cdc_mod.resolve_expect(CDC_KNOWLEDGE["verbs"], "NOPE", None)
    assert e.value.code == UNKNOWN_VERB


def test_verb_timeout_s_prefers_the_caller() -> None:
    verbs = {"SLOW": {"reply": "OK", "timeout_s": 30}}
    assert cdc_mod.verb_timeout_s(verbs, "SLOW", None) == 30.0
    assert cdc_mod.verb_timeout_s(verbs, "SLOW", 5) == 5.0
    assert cdc_mod.verb_timeout_s(verbs, "FAST", None) == cdc_mod.DEFAULT_TIMEOUT_S


def test_purpose_lands_in_lock_file(monkeypatch) -> None:
    from crosspad_hil.locks import PortLock

    monkeypatch.setattr(cdc_mod, "OPEN_SETTLE_S", 0)
    port = f"/dev/fake{next(_PORT_SEQ)}"
    link = CdcLink(port, serial_factory=lambda *a, **k: FakeSerial(), knowledge=CDC_KNOWLEDGE,
                   purpose="hub")
    link.open()
    try:
        assert [r["purpose"] for r in PortLock.holders() if r["port"] == port] == ["hub"]
    finally:
        link.close()


class _DyingSerial(FakeSerial):
    def read(self, size: int = 1) -> bytes:
        raise OSError("device reports readiness to read but returned no data")


def test_reader_alive_goes_false_when_the_port_dies(monkeypatch) -> None:
    monkeypatch.setattr(cdc_mod, "OPEN_SETTLE_S", 0)
    link = CdcLink(f"/dev/fake{next(_PORT_SEQ)}", serial_factory=lambda *a, **k: _DyingSerial(),
                   knowledge=CDC_KNOWLEDGE)
    assert link.reader_alive() is False
    link.open()
    try:
        deadline = time.monotonic() + 2.0
        while link.reader_alive() and time.monotonic() < deadline:
            time.sleep(0.01)
        assert link.reader_alive() is False
    finally:
        link.close()


def test_wait_ready_gives_up_on_an_empty_launcher(make_link: Callable[..., Any]) -> None:
    """A board that answers APP_LIST with no apps (USB transfer mode boots no app
    registry) used to keep wait_ready() polling for good."""
    link, _ = make_link([("APP_LIST", "APPS:  running=-")], knowledge=None)
    t0 = time.monotonic()
    with pytest.raises(HilError) as ei:
        link.wait_ready(timeout_s=1.0)
    assert ei.value.code == TIMEOUT
    assert "no apps" in ei.value.details["cause"]
    assert time.monotonic() - t0 < 5.0

