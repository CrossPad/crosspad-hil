from __future__ import annotations

from crosspad_hil.scenarios import release_check as rc


def test_status_on_an_older_image_offers_the_update():
    f = {"state": "idle", "latest": "1.1.0-rc3", "channel": "beta", "err": "-", "running": "v1.1.0-rc2"}
    assert rc.judge_status(f, "1.1.0-rc3", want_beta=True, running_is_tag=False) == []


def test_status_names_wrong_latest_channel_or_error():
    f = {"latest": "1.1.0-rc2", "channel": "stable", "err": "dns"}
    problems = rc.judge_status(f, "1.1.0-rc3", want_beta=True, running_is_tag=False)
    assert any("latest=1.1.0-rc2" in p for p in problems)
    assert any("channel=stable" in p for p in problems)
    assert any("err=dns" in p for p in problems)


def test_latest_update_flag_follows_the_running_image():
    assert rc.judge_latest({"latest": "1.1.0-rc3", "update": "1"}, "1.1.0-rc3", running_is_tag=False) == []
    assert rc.judge_latest({"latest": "1.1.0-rc3", "update": "0"}, "1.1.0-rc3", running_is_tag=True) == []
    assert rc.judge_latest({"latest": "1.1.0-rc3", "update": "1"}, "1.1.0-rc3", running_is_tag=True)


def test_the_release_itself_has_no_newer_update_to_name():
    f = {"state": "up_to_date", "latest": "-", "channel": "beta", "err": "-"}
    assert rc.judge_status(f, "1.1.0-rc3", want_beta=True, running_is_tag=True) == []
    assert rc.judge_status(f, "1.1.0-rc3", want_beta=True, running_is_tag=False)
    assert rc.judge_latest({"latest": "-", "update": "0"}, "1.1.0-rc3", running_is_tag=True) == []
    assert rc.judge_latest({"latest": "-", "update": "0"}, "1.1.0-rc3", running_is_tag=False)


def test_components_match_by_prefix():
    comps = [{"component": "crosspad-mixer", "commit": "1950a9e"}, {"component": "crosspad-song", "commit": "deadbee"}]
    release = {"components": {"components/crosspad-mixer": "1950a9e654b20e5e96ecf86efb2a9b73b2eb394f",
                              "components/crosspad-song": "04e57581720148fdac2722e540980f3e881d24ac"}}
    [p] = rc.components_problems(comps, release)
    assert "crosspad-song" in p


def test_catalog_lists():
    cat = {"kits": [{"kit": "full", "versions": [{"tag": "v1.1.0-rc3"}]}]}
    assert rc.catalog_lists(cat, "full", "v1.1.0-rc3")
    assert not rc.catalog_lists(cat, "lite", "v1.1.0-rc3")


def test_registered():
    from crosspad_hil.scenarios import get, load_builtin
    load_builtin()
    assert get("release_check").name == "release_check"


# -- components_problems: an equality check ---------------------------------------------------

_REL = {"components": {"components/crosspad-mixer": "1950a9e654b20e5e96ecf86efb2a9b73b2eb394f",
                       "components/crosspad-song": "04e57581720148fdac2722e540980f3e881d24ac"}}
_RUN = [{"component": "crosspad-mixer", "commit": "1950a9e"},
        {"component": "crosspad-song", "commit": "04e5758"}]


def test_components_equal_sets_pass():
    assert rc.components_problems(_RUN, _REL) == []


def test_components_missing_from_app_versions_fails():
    [p] = rc.components_problems(_RUN[:1], _REL)
    assert "crosspad-song" in p and "missing" in p


def test_components_extra_in_app_versions_fails():
    extra = _RUN + [{"component": "crosspad-new", "commit": "abcdef1"}]
    [p] = rc.components_problems(extra, _REL)
    assert "crosspad-new" in p and "not in the release" in p


def test_components_short_or_empty_commit_fails():
    run = [{"component": "crosspad-mixer", "commit": "1950"}, {"component": "crosspad-song", "commit": ""}]
    assert len(rc.components_problems(run, _REL)) == 2


def test_components_malformed_release_is_a_problem_not_an_exception():
    for bad in ({}, {"components": {}}, {"components": []}, {"components": None}, [], "x", None):
        assert rc.components_problems(_RUN, bad)
    assert rc.components_problems(_RUN, {"components": {"nosep": "abc1234", "a/b": 5}})
    assert rc.components_problems(None, _REL)
    assert rc.components_problems([{"commit": "x"}], _REL)


def test_non_json_components_file_is_a_bench_problem(monkeypatch):
    class Done:
        stdout = "<html>not json</html>"
    monkeypatch.setattr(rc.subprocess, "run", lambda *a, **k: Done())
    s = rc.ReleaseCheckScenario()
    s.p = rc.Params(tag="v1.1.0-rc3")
    try:
        s._release_components()
    except rc._Stop as e:
        assert e.exit_code == 2 and "not JSON" in e.summary
    else:
        raise AssertionError("no _Stop")


# -- phase B with a fake board: the way back --------------------------------------------------

import threading  # noqa: E402
from types import SimpleNamespace  # noqa: E402

import pytest  # noqa: E402

from crosspad_hil.errors import CANCELLED, TIMEOUT, HilError  # noqa: E402
from crosspad_hil.scenarios.fw_switch import FwSwitchScenario  # noqa: E402


class Board:
    def __init__(self):
        self.running = "A"
        self.state = {"A": "confirmed", "B": "confirmed"}
        self.installed = False
        self.switches = 0
        self.link_gen = 0
        self.after_switch = "confirmed"      # what the slot booted by FW_SWITCH reports
        self.crash = None                    # None | "stay" | "rollback" | "stuck_trial"
        self.new_confirm_lies = False        # _wait_confirmed(new slot) -> None
        self.status_script: list = []        # states FW_UPD_STATUS answers first (a callable runs)
        self.status_calls = 0

    def switch(self):
        self.switches += 1
        self.running = "B" if self.running == "A" else "A"
        self.state[self.running] = self.after_switch
        self.link_gen += 1                   # old link dies with the reboot

    def die(self):
        self.link_gen += 1


class Link:
    def __init__(self, board):
        self.board, self.gen = board, board.link_gen

    def dead(self):
        return self.gen != self.board.link_gen

    def close(self):
        pass

    def wait_ready(self):
        pass


@pytest.fixture
def bench(monkeypatch):
    board = Board()
    cancelled = threading.Event()

    def check_cancelled():
        if cancelled.is_set():
            raise HilError(CANCELLED, "scenario cancelled")
    ctx = SimpleNamespace(device=SimpleNamespace(id="dev"), cancelled=cancelled, partial={},
                          log=lambda *_: None, open_cdc=lambda: Link(board),
                          check_cancelled=check_cancelled)
    real_sleep = rc.time.sleep
    monkeypatch.setattr(rc.time, "sleep", lambda s: real_sleep(0.001))
    monkeypatch.setattr(rc, "discover", lambda: [])
    monkeypatch.setattr(rc, "select", lambda d, i: ctx.device)
    monkeypatch.setattr(rc, "MidiIO", lambda *a, **k: SimpleNamespace(
        open=lambda: None, close=lambda: None))
    monkeypatch.setattr(rc.ReleaseCheckScenario, "_wait_catalog", lambda self, rev, kit: None)
    monkeypatch.setattr(rc.ReleaseCheckScenario, "_phase_a", lambda self, *a: None)
    monkeypatch.setattr(rc.ReleaseCheckScenario, "_companion", lambda self, verb: (
        "FW_LATEST: latest=1.1.0-rc3 running=v1.1.0-rc3 update=0" if verb == "FW_LATEST" else "OK"))
    monkeypatch.setattr(rc.ReleaseCheckScenario, "_release_components", lambda self: dict(_REL))

    def alive(link):
        if link.dead():
            raise HilError(TIMEOUT, "link is dead")

    def fw_slots(link):
        alive(link)
        return {"slots": [{"slot": n, "running": n == board.running, "state": board.state[n]}
                          for n in "AB"], "rollback": True}

    def fw_switch(link):
        alive(link)
        board.switch()
        return {}

    def fw_check(link):
        alive(link)
        if board.crash:
            board.die()
            if board.crash == "rollback":
                board.running = "A"
                board.state["A"] = "confirmed"
            elif board.crash == "stuck_trial":
                board.state[board.running] = "trial"
            raise HilError(TIMEOUT, "board went away during FW_CHECK")

    def fw_upd_status(link):
        alive(link)
        board.status_calls += 1
        if board.status_script:
            step = board.status_script.pop(0)
            state = step() if callable(step) else step
            return {"state": state, "kit": "full", "running": "v1.1.0-rc2", "channel": "beta",
                    "err": "-"}
        ready = board.installed and board.running == "A"
        return {"state": "ready_to_restart" if ready else "up_to_date", "kit": "full",
                "running": "v1.1.0-rc2", "channel": "beta", "err": "-"}

    def fw_download(link, *a, **k):
        alive(link)
        board.installed = True

    def reopen(ctx_, link):
        if ctx_.cancelled.is_set():
            raise HilError(CANCELLED, "scenario cancelled")
        return Link(board)

    def wait_confirmed(ctx_, link, label, timeout, data):
        if ctx_.cancelled.is_set():
            raise HilError(CANCELLED, "scenario cancelled")
        if label != "A" and board.new_confirm_lies:
            return None
        if board.running == label and board.state[label] == "confirmed":
            return {"slot": label}
        return None

    v = rc.verbs
    monkeypatch.setattr(v, "fw_slots", fw_slots)
    monkeypatch.setattr(v, "fw_switch", fw_switch)
    monkeypatch.setattr(v, "fw_check", fw_check)
    monkeypatch.setattr(v, "fw_upd_status", fw_upd_status)
    monkeypatch.setattr(v, "fw_download", fw_download)
    monkeypatch.setattr(v, "reset_count", lambda link: 3)
    monkeypatch.setattr(v, "cdc_stats", lambda link: {"maxcmd_us": 10})
    monkeypatch.setattr(v, "app_versions", lambda link: {"board": {"fw": "v1"}, "components": _RUN})
    monkeypatch.setattr(FwSwitchScenario, "_reopen", staticmethod(reopen))
    monkeypatch.setattr(FwSwitchScenario, "_wait_confirmed", staticmethod(wait_confirmed))
    ctx.board = board
    return ctx


def _run(ctx, **kw):
    s = rc.ReleaseCheckScenario()
    params = rc.Params(tag="v1.1.0-rc3", install=True, restore_timeout=0.5, **kw)
    ctx.link0 = Link(ctx.board)
    ctx.open_cdc = lambda: Link(ctx.board)
    report = s.run(ctx, params, lambda *a: None)
    return report


def test_a_clean_install_check_and_way_back(bench):
    r = _run(bench)
    assert r.passed and r.exit_code == 0, r.summary
    assert bench.board.running == "A" and bench.board.switches == 2
    assert r.data["back_on_start_slot"] is True


def test_a_reboot_during_the_check_is_recovered(bench):
    bench.board.crash = "stay"
    r = _run(bench)
    assert r.exit_code == 1 and "FW_CHECK over cdc" in r.summary
    assert r.data["back_on_start_slot"] is True
    assert bench.board.running == "A" and bench.board.state["A"] == "confirmed"
    assert bench.board.switches == 2          # in, and back


def test_a_board_that_rolled_back_by_itself_is_not_switched_again(bench):
    bench.board.crash = "rollback"
    r = _run(bench)
    assert r.exit_code == 1 and r.data["back_on_start_slot"] is True
    assert bench.board.running == "A"
    assert bench.board.switches == 1          # only the switch into the release


def test_a_failed_recovery_is_exit_2_whatever_the_body_said(bench):
    bench.board.crash = "stuck_trial"
    r = _run(bench)
    assert r.exit_code == 2 and r.passed is False
    assert r.data["back_on_start_slot"] is False and r.data["recovery_error"]
    assert "FW_CHECK over cdc" in r.summary            # the body's own summary is kept
    assert "WRONG IMAGE" in r.summary
    assert bench.board.running == "B"


def test_a_failure_right_after_the_first_switch_still_recovers(bench):
    bench.board.new_confirm_lies = True
    r = _run(bench)
    assert r.exit_code == 1 and "never confirmed" in r.summary     # the body's own code, kept
    assert r.data["back_on_start_slot"] is True
    assert bench.board.running == "A" and bench.board.switches == 2


def test_cancellation_during_phase_b_still_completes_the_recovery(bench):
    board, orig = bench.board, FwSwitchScenario._wait_confirmed

    def cancel_after_switch(ctx_, link, label, timeout, data):
        ctx_.cancelled.set()
        raise HilError(CANCELLED, "scenario cancelled")
    FwSwitchScenario._wait_confirmed = staticmethod(cancel_after_switch)
    try:
        s = rc.ReleaseCheckScenario()
        params = rc.Params(tag="v1.1.0-rc3", install=True, restore_timeout=0.5)
        with pytest.raises(HilError) as exc:
            s.run(bench, params, lambda *a: None)
    finally:
        FwSwitchScenario._wait_confirmed = orig
    assert exc.value.code == CANCELLED and bench.cancelled.is_set()
    assert s.data["back_on_start_slot"] is True
    assert board.running == "A" and board.state["A"] == "confirmed"


def _reset_count_kills_link(monkeypatch, bench, state):
    board = bench.board

    def reset_count(link):
        if board.running == "B":
            board.die()
            board.state["B"] = state
            raise HilError(TIMEOUT, "no reply to RESET_LOG")
        return 3
    monkeypatch.setattr(rc.verbs, "reset_count", reset_count)


def test_f_hilerror_escaping_the_body_with_a_failed_recovery_is_exit_2(bench, monkeypatch):
    _reset_count_kills_link(monkeypatch, bench, "trial")      # B never confirms: stuck
    r = _run(bench)
    assert r.passed is False and r.exit_code == 2
    assert "no reply to RESET_LOG" in r.summary and "WRONG IMAGE" in r.summary
    assert r.data["back_on_start_slot"] is False and r.data["cancelled"] is False
    assert r.data["error"]["code"] == TIMEOUT
    assert bench.board.running == "B"


def test_g_hilerror_escaping_the_body_with_a_good_recovery_still_propagates(bench, monkeypatch):
    _reset_count_kills_link(monkeypatch, bench, "confirmed")
    s = rc.ReleaseCheckScenario()
    with pytest.raises(HilError) as exc:
        s.run(bench, rc.Params(tag="v1.1.0-rc3", install=True, restore_timeout=0.5), lambda *a: None)
    assert exc.value.code == TIMEOUT
    assert s.data["back_on_start_slot"] is True and bench.board.running == "A"


def test_h_cancellation_with_a_failed_recovery_is_an_exit_2_report(bench):
    orig = FwSwitchScenario._wait_confirmed

    def cancel_after_switch(ctx_, link, label, timeout, data):
        ctx_.cancelled.set()
        bench.board.state[bench.board.running] = "trial"        # and it never confirms
        raise HilError(CANCELLED, "scenario cancelled")
    FwSwitchScenario._wait_confirmed = staticmethod(cancel_after_switch)
    try:
        r = rc.ReleaseCheckScenario().run(
            bench, rc.Params(tag="v1.1.0-rc3", install=True, restore_timeout=0.3), lambda *a: None)
    finally:
        FwSwitchScenario._wait_confirmed = orig
    assert r.exit_code == 2 and r.data["cancelled"] is True
    assert "scenario cancelled" in r.summary and "WRONG IMAGE" in r.summary


def test_i_restore_swallows_any_exception(bench, monkeypatch):
    s = rc.ReleaseCheckScenario()
    s.ctx, s.p, s.data = bench, rc.Params(tag="v1.1.0-rc3", restore_timeout=0.5), {}
    s.link = Link(bench.board)
    monkeypatch.setattr(rc.verbs, "fw_slots", lambda link: {"rows": []})      # KeyError "slots"
    s._restore("A")
    assert s.data["back_on_start_slot"] is False and "slots" in s.data["recovery_error"]


# -- M1: a companion reply must be the verb's own ----------------------------------------------

def _frame(text: str, last: bool = True) -> bytes:
    return bytes([0xF0, 0x7D, 0x1F, 0x03 if last else 0x02, *text.encode("ascii"), 0xF7])


class FakeIO:
    def __init__(self, replies):
        self._replies, self.sent = list(replies), []

    def drain(self):
        return 0

    def send_sysex(self, frame):
        self.sent.append(bytes(frame))

    def receive(self, timeout_s):
        if self._replies:
            return self._replies.pop(0)
        time.sleep(0.001)
        return None


import time  # noqa: E402

_STATUS = "FWUPD: state=idle running=v1.1.0-rc2 kit=full latest=1.1.0-rc3 channel=beta err=-"
_LATEST = "FW_LATEST: latest=1.1.0-rc3 running=v1.1.0-rc2 update=1"


def _scenario(io, **params):
    s = rc.ReleaseCheckScenario()
    s.io, s.p = io, rc.Params(tag="v1.1.0-rc3", **params)
    return s


@pytest.mark.parametrize("verb, wrong, right", [
    ("FW_UPD_STATUS", _LATEST, _STATUS),
    ("FW_LATEST", _STATUS, _LATEST),
    ("FW_CHECK", _STATUS, "OK"),
    ("FW_CHECK", _LATEST, "ERR no_mem"),
])
def test_a_reply_of_another_verb_is_not_taken_for_the_answer(verb, wrong, right):
    s = _scenario(FakeIO([_frame(wrong), _frame(right)]))
    assert s._companion(verb) == right


def test_only_foreign_replies_end_in_a_timeout_not_in_a_wrong_answer(monkeypatch):
    monkeypatch.setattr(rc, "OK_WITHIN_S", -2.9)              # a 0.1 s read
    s = _scenario(FakeIO([_frame(_STATUS)]))
    with pytest.raises(HilError) as exc:
        s._companion("FW_LATEST")
    assert exc.value.code == TIMEOUT


def test_a_multi_chunk_reply_is_matched_whole():
    s = _scenario(FakeIO([_frame("FWUPD: state=idle ", last=False), _frame("running=v1")]))
    assert s._companion("FW_UPD_STATUS") == "FWUPD: state=idle running=v1"


def test_err_busy_on_a_query_verb_is_asked_again(monkeypatch):
    monkeypatch.setattr(rc.time, "sleep", lambda s: None)
    s = _scenario(FakeIO([_frame("ERR busy"), _frame("ERR busy"), _frame(_LATEST)]))
    assert s._companion("FW_LATEST") == _LATEST
    assert len(s.io.sent) == 3


# -- M2: ERR busy on FW_CHECK is a wait, not a firmware failure --------------------------------

def _checker(bench, monkeypatch, **params):
    monkeypatch.setattr(rc, "POLL_S", 0.001)
    s = rc.ReleaseCheckScenario()
    s.ctx, s.p, s.data = bench, rc.Params(tag="v1.1.0-rc3", **params), {"steps": {}}
    s.link = Link(bench.board)
    return s


def _busy():
    return HilError("NOT_SUPPORTED", "FW_CHECK: device replied ERR busy")


def test_fw_check_busy_over_cdc_waits_for_the_background_check_and_retries(bench, monkeypatch):
    s = _checker(bench, monkeypatch, check_timeout=5.0)
    calls = {"n": 0}

    def fw_check(link):
        calls["n"] += 1
        if calls["n"] <= 2:
            raise _busy()
    monkeypatch.setattr(rc.verbs, "fw_check", fw_check)
    bench.board.status_script = ["checking", "checking", "up_to_date"]
    s._fw_check_ok("cdc", lambda: (rc.verbs.fw_check(s.link), "OK")[1])
    assert calls["n"] == 3
    assert s.data["steps"]["fw_check_cdc"]["status"]["state"] == "up_to_date"
    assert bench.board.status_calls >= 3          # it waited while FW_UPD_STATUS said checking


def test_fw_check_busy_over_sysex_waits_and_retries(bench, monkeypatch):
    s = _checker(bench, monkeypatch, check_timeout=5.0)
    answers = ["ERR busy", "ERR busy", "OK"]
    bench.board.status_script = ["checking", "up_to_date", "up_to_date"]
    s._fw_check_ok("sysex", lambda: answers.pop(0))
    assert answers == []
    assert s.data["steps"]["fw_check_sysex"]["answer"] == "OK"


def test_a_check_that_stays_busy_fails_after_check_timeout_as_a_firmware_fault(bench, monkeypatch):
    s = _checker(bench, monkeypatch, check_timeout=0.05)
    with pytest.raises(rc._Stop) as exc:
        s._fw_check_ok("sysex", lambda: "ERR busy")
    assert exc.value.exit_code == 1 and "still ERR busy" in exc.value.summary


def test_any_other_err_is_still_a_failure_at_once(bench, monkeypatch):
    s = _checker(bench, monkeypatch, check_timeout=5.0)
    with pytest.raises(rc._Stop) as exc:
        s._fw_check_ok("sysex", lambda: "ERR no_mem")
    assert "ERR no_mem" in exc.value.summary
    with pytest.raises(rc._Stop) as exc:
        s._fw_check_ok("cdc", lambda: (_ for _ in ()).throw(
            HilError("NOT_SUPPORTED", "FW_CHECK: device replied ERR disabled")))
    assert "ERR disabled" in exc.value.summary


# -- I4: cancellation stops the loops, and an install that was cancelled is never switched in ---

def test_a_cancel_during_the_install_poll_raises_and_never_switches(bench, monkeypatch):
    monkeypatch.setattr(rc, "POLL_S", 0.001)
    bench.board.status_script = ["installing", lambda: (bench.cancelled.set(), "installing")[1],
                                 "installing", "installing"]
    s = rc.ReleaseCheckScenario()
    with pytest.raises(HilError) as exc:
        s.run(bench, rc.Params(tag="v1.1.0-rc3", install=True, restore_timeout=0.5), lambda *a: None)
    assert exc.value.code == CANCELLED
    assert bench.board.switches == 0 and bench.board.running == "A"
    assert bench.board.status_calls < 8            # it did not spin on the cancelled event


def test_a_cancel_on_the_last_poll_does_not_go_on_to_fw_switch(bench, monkeypatch):
    monkeypatch.setattr(rc, "POLL_S", 0.001)
    bench.board.status_script = [lambda: (bench.cancelled.set(), "ready_to_restart")[1]]
    s = rc.ReleaseCheckScenario()
    with pytest.raises(HilError) as exc:
        s.run(bench, rc.Params(tag="v1.1.0-rc3", install=True, restore_timeout=0.5), lambda *a: None)
    assert exc.value.code == CANCELLED
    assert bench.board.switches == 0


def test_settle_stops_on_cancel_instead_of_spinning(bench, monkeypatch):
    s = _checker(bench, monkeypatch, check_timeout=30.0)
    seen = {"n": 0}

    def read():
        seen["n"] += 1
        if seen["n"] == 3:
            bench.cancelled.set()
        return {"state": "checking"}
    with pytest.raises(HilError) as exc:
        s._settle(read)
    assert exc.value.code == CANCELLED and seen["n"] == 3
