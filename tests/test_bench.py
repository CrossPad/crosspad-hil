"""Bench lease: claim / queue / expiry / refusal, the file shared between processes,
and the daemon and CLI refusing board ops while somebody else holds the board."""
from __future__ import annotations

import io
import json
import threading

import pytest

from crosspad_hil import bench as bench_mod
from crosspad_hil import cli, serve
from crosspad_hil.bench import HANDOFF_S, MD_BEGIN, MD_END, QUEUE_TTL_S, Bench
from crosspad_hil.devices import Backends, SerialPortInfo, discover
from crosspad_hil.errors import BAD_ARGS, BENCH_BUSY, HilError
from tests.fakes import FakeSerial

DEV = "dev_1ede"
MIN = 60.0


class Clock:
    def __init__(self) -> None:
        self.t = 1_800_000_000.0

    def __call__(self) -> float:
        return self.t

    def advance(self, s: float) -> None:
        self.t += s


@pytest.fixture
def clock() -> Clock:
    return Clock()


@pytest.fixture
def path(tmp_path):
    return tmp_path / "bench.json"


@pytest.fixture
def b(path, clock) -> Bench:
    return Bench(path, clock=clock, log=lambda m: None)


def _busy(fn, *a, **kw) -> HilError:
    with pytest.raises(HilError) as ei:
        fn(*a, **kw)
    assert ei.value.code == BENCH_BUSY
    return ei.value


# ---------------------------------------------------------------- claim and queue

def test_claim_free_board_is_granted(b):
    r = b.claim(DEV, "pidf", "firmware_app test", ttl_s=20 * MIN)
    assert r["granted"] is True
    assert r["lease"]["holder"] == "pidf" and r["lease"]["purpose"] == "firmware_app test"
    assert r["lease"]["expires_in_s"] == 20 * MIN
    assert "pidf holds dev_1ede" in r["message"]


def test_queue_keeps_arrival_order(b):
    b.claim(DEV, "pidf", "test")
    r1 = b.claim(DEV, "twin", "smoke")
    r2 = b.claim(DEV, "stm", "dfu")
    assert (r1["granted"], r1["position"]) == (False, 1)
    assert (r2["granted"], r2["position"]) == (False, 2)
    assert "claimed by pidf (test)" in r2["message"] and "#2 in the queue" in r2["message"]
    # asking again keeps the place
    assert b.claim(DEV, "twin")["position"] == 1
    st = b.status(DEV)["devices"][0]
    assert [q["holder"] for q in st["queue"]] == ["twin", "stm"]
    assert st["lease"]["holder"] == "pidf"


def test_claim_again_while_holding_renews(b, clock):
    b.claim(DEV, "pidf", ttl_s=10 * MIN)
    clock.advance(8 * MIN)
    r = b.claim(DEV, "pidf", ttl_s=30 * MIN)
    assert r["granted"] is True and r["lease"]["expires_in_s"] == 30 * MIN


def test_ttl_is_clamped(b):
    assert b.claim(DEV, "a", ttl_s=1)["lease"]["ttl_s"] == bench_mod.MIN_TTL_S
    b.release(DEV, "a")
    assert b.claim(DEV, "b", ttl_s=10**9)["lease"]["ttl_s"] == bench_mod.MAX_TTL_S


def test_holder_is_required(b):
    with pytest.raises(HilError) as ei:
        b.claim(DEV, "  ")
    assert ei.value.code == BAD_ARGS


# ---------------------------------------------------------------- refusal (check)

def test_check_refuses_another_holder_naming_holder_and_eta(b, clock):
    b.claim(DEV, "pidf", "firmware_app", ttl_s=30 * MIN)
    clock.advance(5 * MIN)
    e = _busy(b.check, DEV, "twin", "cdc.transact")
    assert "claimed by pidf (firmware_app)" in e.message
    assert "in 25 min" in e.message and "you (twin) may not run cdc.transact" in e.message
    assert e.details["holder"] == "pidf" and e.details["expires_in_s"] == 25 * MIN
    assert e.details["op"] == "cdc.transact"
    # no holder at all is refused just the same
    e = _busy(b.check, DEV, None, "ota.flash")
    assert "no holder given" in e.message


def test_check_free_board_and_unknown_board_are_allowed(b):
    assert b.check(DEV, None, "cdc.open")["allowed"] is True
    b.claim(DEV, "pidf")
    b.release(DEV, "pidf", "fw")
    assert b.check(DEV, "anyone", "cdc.open")["allowed"] is True


def test_check_by_holder_renews_the_lease(b, clock):
    b.claim(DEV, "pidf", ttl_s=10 * MIN)
    clock.advance(9 * MIN)
    r = b.check(DEV, "pidf", "cdc.verb")
    assert r["allowed"] is True and r["lease"]["expires_in_s"] == 10 * MIN
    # survives past the original expiry because it was renewed
    clock.advance(9 * MIN)
    assert b.check(DEV, "pidf", "cdc.verb")["allowed"] is True


def test_force_overrides_and_is_logged(b):
    logged: list[str] = []
    b._log = logged.append
    b.claim(DEV, "pidf", "test")
    r = b.check(DEV, "twin", "ota.flash", force=True)
    assert r["allowed"] is True and r["forced"] is True
    hist = b.status(DEV)["devices"][0]["history"]
    assert hist[-1]["kind"] == "force" and hist[-1]["over"] == "pidf"
    assert hist[-1]["op"] == "ota.flash" and hist[-1]["holder"] == "twin"
    assert logged and "forced ota.flash" in logged[0]
    # the lease itself is untouched
    assert b.status(DEV)["devices"][0]["lease"]["holder"] == "pidf"


def test_force_claim_takes_the_board(b):
    b.claim(DEV, "dead-session")
    r = b.claim(DEV, "twin", force=True)
    assert r["granted"] is True and r["lease"]["holder"] == "twin"
    hist = b.status(DEV)["devices"][0]["history"]
    assert hist[-1]["kind"] == "taken" and hist[-1]["by"] == "twin"


def test_warn_mode_logs_instead_of_refusing(path, clock, monkeypatch):
    monkeypatch.setenv("CROSSPAD_BENCH_MODE", "warn")
    logged: list[str] = []
    b = Bench(path, clock=clock, log=logged.append)
    b.claim(DEV, "pidf", "test")
    r = b.check(DEV, "twin", "cdc.open")
    assert r["allowed"] is True and "claimed by pidf" in r["warning"]
    assert logged and "warn only" in logged[0]
    # claims still queue: the lease itself is unchanged by the mode
    assert b.claim(DEV, "twin")["granted"] is False


def test_off_mode_checks_nothing(path, clock):
    b = Bench(path, clock=clock, mode="off", log=lambda m: None)
    b.claim(DEV, "pidf")
    assert b.check(DEV, "twin", "ota.flash")["allowed"] is True


def test_unknown_mode_enforces(path, clock):
    logged: list[str] = []
    b = Bench(path, clock=clock, mode="maybe", log=logged.append)
    b.claim(DEV, "pidf")
    _busy(b.check, DEV, "twin", "cdc.open")
    assert "enforcing" in logged[0]


# ---------------------------------------------------------------- expiry

def test_lease_expires_after_ttl_without_renew(b, clock):
    b.claim(DEV, "pidf", ttl_s=10 * MIN)
    clock.advance(10 * MIN + 1)
    st = b.status(DEV)["devices"][0]
    assert st["lease"] is None
    assert st["history"][-1]["kind"] == "expired" and st["history"][-1]["holder"] == "pidf"
    assert st["firmware"]["desc"] is None and "unknown" in st["firmware"]["note"]
    assert b.check(DEV, "twin", "cdc.open")["allowed"] is True
    assert b.claim(DEV, "twin")["granted"] is True


def test_expired_lease_hands_the_board_to_the_queue_head(b, clock):
    b.claim(DEV, "pidf", ttl_s=10 * MIN)
    b.claim(DEV, "twin")
    b.claim(DEV, "stm")
    clock.advance(11 * MIN)
    # the head may claim; anybody else is held off during the handoff window
    _busy(b.check, DEV, "stm", "cdc.open")
    r = b.claim(DEV, "stm")
    assert r["granted"] is False and "held for twin" in r["message"]
    assert b.claim(DEV, "twin")["granted"] is True


def test_handoff_window_lapses(b, clock):
    b.claim(DEV, "pidf")
    b.claim(DEV, "twin")
    b.claim(DEV, "stm")
    b.release(DEV, "pidf", "fw")
    clock.advance(HANDOFF_S + 1)
    # twin never came back: first come, first served
    assert b.claim(DEV, "stm")["granted"] is True
    assert [q["holder"] for q in b.status(DEV)["devices"][0]["queue"]] == ["twin"]


def test_queue_entries_expire_when_not_refreshed(b, clock):
    b.claim(DEV, "pidf", ttl_s=bench_mod.MAX_TTL_S)
    b.claim(DEV, "twin")
    clock.advance(QUEUE_TTL_S / 2)
    b.claim(DEV, "stm")
    clock.advance(QUEUE_TTL_S / 2 + 1)
    assert [q["holder"] for q in b.status(DEV)["devices"][0]["queue"]] == ["stm"]


# ---------------------------------------------------------------- release

def test_release_records_firmware_left_and_history(b, clock):
    b.claim(DEV, "pidf", "plan C")
    clock.advance(20 * MIN)
    r = b.release(DEV, "pidf", "platform-idf feat/fw-homebrew-r2 eac76e8")
    assert r["released"] is True and r["lease"] is None
    st = b.status(DEV)["devices"][0]
    assert st["firmware"]["desc"] == "platform-idf feat/fw-homebrew-r2 eac76e8"
    h = st["history"][-1]
    assert h["kind"] == "released" and h["purpose"] == "plan C"
    assert h["until"] - h["since"] == 20 * MIN


def test_release_by_another_holder_is_refused_unless_forced(b):
    b.claim(DEV, "pidf")
    _busy(b.release, DEV, "twin", "x")
    r = b.release(DEV, "twin", None, force=True)
    assert r["released"] is True
    h = b.status(DEV)["devices"][0]["history"][-1]
    assert h["kind"] == "forced-release" and h["by"] == "twin" and h["holder"] == "pidf"


def test_release_while_queued_leaves_the_queue(b):
    b.claim(DEV, "pidf")
    b.claim(DEV, "twin")
    r = b.release(DEV, "twin")
    assert r["released"] is False and r["left_queue"] is True
    assert b.status(DEV)["devices"][0]["queue"] == []


def test_renew_needs_the_lease(b, clock):
    b.claim(DEV, "pidf", ttl_s=10 * MIN)
    clock.advance(5 * MIN)
    assert b.renew(DEV, "pidf")["lease"]["expires_in_s"] == 10 * MIN
    _busy(b.renew, DEV, "twin")


# ---------------------------------------------------------------- the shared file

def test_state_survives_a_restart(path, clock, b):
    b.claim(DEV, "pidf", "test", ttl_s=30 * MIN)
    b.claim(DEV, "twin")
    again = Bench(path, clock=clock, log=lambda m: None)
    st = again.status(DEV)["devices"][0]
    assert st["lease"]["holder"] == "pidf" and st["queue"][0]["holder"] == "twin"
    _busy(again.check, DEV, "twin", "cdc.open")


def test_corrupt_file_is_moved_aside(path, clock):
    path.write_text("{not json")
    logged: list[str] = []
    b = Bench(path, clock=clock, log=logged.append)
    assert b.claim(DEV, "pidf")["granted"] is True
    assert list(path.parent.glob("bench.json.corrupt-*")) and "unreadable" in logged[0]


def test_concurrent_claims_grant_exactly_one(path):
    results: list[dict] = []
    guard = threading.Lock()

    def claim(i: int) -> None:
        r = Bench(path, log=lambda m: None).claim(DEV, f"s{i}")  # one Bench per "process"
        with guard:
            results.append(r)

    threads = [threading.Thread(target=claim, args=(i,)) for i in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert sum(r["granted"] for r in results) == 1
    st = Bench(path).status(DEV)["devices"][0]
    assert len(st["queue"]) == 7


def test_markdown_block_is_replaced_in_place(tmp_path, path, clock):
    md = tmp_path / "BENCH.md"
    md.write_text(f"# Bench\n\nrules\n\n{MD_BEGIN}\nold\n{MD_END}\n\ntrailer\n", encoding="utf-8")
    b = Bench(path, clock=clock, md_path=md, log=lambda m: None)
    b.claim(DEV, "pidf", "firmware_app")
    b.claim(DEV, "twin", "smoke")
    text = md.read_text(encoding="utf-8")
    assert text.startswith("# Bench\n\nrules\n\n" + MD_BEGIN) and text.endswith("trailer\n")
    assert "old" not in text
    assert "- pidf — firmware_app; since" in text and "1. twin — smoke" in text
    b.release(DEV, "pidf", "pidf eac76e8")
    text = md.read_text(encoding="utf-8")
    assert "On the board: pidf eac76e8" in text and "released" in text


# ---------------------------------------------------------------- daemon and CLI

CDC = "/dev/ttyACM0"
CON = "/dev/ttyACM1"


def _backends() -> Backends:
    ports = [
        SerialPortInfo(path=CDC, vid=0x303A, pid=0x3456, serial="AB12CD34",
                       product="CrossPad", location="1-1.2:1.0"),
        SerialPortInfo(path=CON, vid=0x0483, pid=0x5740, serial="STM0001",
                       product="CrossPad MIDI+Serial", location="1-1.3:1.0"),
    ]
    return Backends(list_serial=lambda: list(ports), list_midi=lambda: [],
                    list_audio=lambda: [])


@pytest.fixture
def lock_dir(monkeypatch, tmp_path):
    monkeypatch.setenv("XDG_RUNTIME_DIR", str(tmp_path / "run"))
    return tmp_path


def test_daemon_refuses_board_ops_of_another_holder(lock_dir, b):
    fake = FakeSerial([("APP_LIST", "APPS: Sampler running=-")] * 2)
    d = serve.Daemon(io.StringIO(), io.StringIO(), backends=_backends(),
                     serial_factory=lambda path, **_: fake, bench=b)
    dev = discover(_backends())[0].id

    def call(name: str, **args) -> dict:
        return d.handle({"id": 1, "op": name, "args": args})

    r = call("bench.claim", device=dev, holder="pidf", purpose="firmware_app")
    assert r["ok"] and r["result"]["granted"] is True
    r = call("cdc.transact", device=dev, cmd="APP_LIST", holder="twin")
    assert r["ok"] is False and r["error"]["code"] == BENCH_BUSY
    assert r["error"]["details"]["holder"] == "pidf"
    assert r["error"]["details"]["op"] == "cdc.transact"
    assert "APP_LIST" not in fake.written  # the port was never opened
    r = call("bench.check", holder="twin", op="idf.py flash")
    assert r["ok"] is False and r["error"]["code"] == BENCH_BUSY
    assert call("cdc.transact", device=dev, cmd="APP_LIST", holder="pidf")["ok"] is True
    assert call("cdc.transact", device=dev, cmd="APP_LIST", holder="twin",
                bench_force=True)["ok"] is True
    r = call("bench.release", device=dev, holder="pidf", firmware_left="pidf abc123")
    assert r["ok"] and r["result"]["released"] is True
    st = call("bench.status", device=dev)["result"]["devices"][0]
    assert st["firmware"]["desc"] == "pidf abc123"
    assert [h["kind"] for h in st["history"]] == ["force", "released"]


def test_daemon_bench_check_without_a_board_uses_the_only_known_lease(lock_dir, b):
    empty = Backends(list_serial=lambda: [], list_midi=lambda: [], list_audio=lambda: [])
    d = serve.Daemon(io.StringIO(), io.StringIO(), backends=empty, bench=b)
    b.claim(DEV, "pidf")
    r = d.handle({"id": 1, "op": "bench.check", "args": {"holder": "stm", "op": "stm swd"}})
    assert r["ok"] is False and r["error"]["details"]["holder"] == "pidf"
    r = d.handle({"id": 2, "op": "bench.check", "args": {"holder": "pidf", "op": "stm swd"}})
    assert r["ok"] is True


def test_cli_bench_round_trip_and_refusal(lock_dir, b, monkeypatch):
    fake = FakeSerial([("APP_LIST", "APPS: Sampler running=-")])
    out, err = io.StringIO(), io.StringIO()
    deps = cli.CliDeps(backends=_backends(), serial_factory=lambda path, **_: fake,
                       stdout=out, stderr=err, sleep=lambda s: None, bench=b)
    assert cli.main(["bench", "claim", "firmware_app", "--holder", "pidf", "--json"],
                    deps=deps) == 0
    assert json.loads(out.getvalue())["granted"] is True
    monkeypatch.setenv("CROSSPAD_BENCH_HOLDER", "twin")
    assert cli.main(["bench", "claim", "smoke"], deps=deps) == 1  # queued
    assert cli.main(["cdc", "APP_LIST"], deps=deps) == 2
    assert "BENCH_BUSY" in err.getvalue() and "claimed by pidf" in err.getvalue()
    assert cli.main(["cdc", "APP_LIST", "--holder", "pidf"], deps=deps) == 0
    assert cli.main(["bench", "release", "pidf abc123", "--holder", "pidf"], deps=deps) == 0
    out.truncate(0)
    out.seek(0)
    assert cli.main(["bench", "status"], deps=deps) == 0
    text = out.getvalue()
    assert "On the board: pidf abc123" in text and "1. twin — smoke" in text
