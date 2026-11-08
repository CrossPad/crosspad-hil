"""Typed CDC verbs over a scripted FakeSerial. No hardware."""
from __future__ import annotations

import uuid
from collections.abc import Iterator

import pytest

from crosspad_hil import verbs
from crosspad_hil.cdc import CdcLink
from crosspad_hil.errors import BAD_ARGS, NOT_SUPPORTED, HilError
from tests.fakes import FakeSerial


class SeqFakeSerial(FakeSerial):
    """FakeSerial whose reply to a repeated command changes between calls.

    `sequences` maps a command to a list of replies (each reply is a str or a
    list[str]); every write pops the next one, the last one stays sticky.
    Commands not in `sequences` fall through to the plain FakeSerial script.
    """

    def __init__(self, script=(), sequences: dict[str, list] | None = None) -> None:
        super().__init__(list(script))
        self.sequences = {k: list(v) for k, v in (sequences or {}).items()}

    def write(self, data: bytes) -> int:
        cmd = data.decode("utf-8", "replace").strip()
        seq = self.sequences.get(cmd)
        if seq is None:
            return super().write(data)
        reply = seq.pop(0) if len(seq) > 1 else seq[0]
        self.written.append(cmd)
        self.feed([reply] if isinstance(reply, str) else list(reply))
        return len(data)


def _open_link(fake: FakeSerial) -> CdcLink:
    link = CdcLink(f"fake-verbs-{uuid.uuid4().hex[:8]}",
                   serial_factory=lambda path, **kw: fake)
    link.open()
    return link


@pytest.fixture
def make_link() -> Iterator:
    opened: list[CdcLink] = []

    def _make(script=(), sequences=None) -> CdcLink:
        fake = SeqFakeSerial(script, sequences)
        link = _open_link(fake)
        link.fake = fake  # type: ignore[attr-defined]
        opened.append(link)
        return link

    yield _make
    for link in opened:
        link.close()


# ── app_* ──────────────────────────────────────────────────────────────

def test_app_list_parses_apps_and_running(make_link) -> None:
    link = make_link([("APP_LIST", "APPS: Sampler,Sequencer,Fishtank running=-")])
    assert verbs.app_list(link) == {
        "apps": ["Sampler", "Sequencer", "Fishtank"], "running": None}


def test_app_start_success_polls_app_list(make_link) -> None:
    link = make_link(
        [("APP_START Sampler", "OK")],
        sequences={"APP_LIST": ["APPS: Sampler,Fishtank running=-",
                                "APPS: Sampler,Fishtank running=Sampler"]},
    )
    assert verbs.app_start(link, "Sampler", wait_s=2.0) == {"running": "Sampler"}
    assert link.fake.written[0] == "APP_START Sampler"
    assert link.fake.written.count("APP_LIST") >= 2


def test_app_start_err_raises_bad_args(make_link) -> None:
    link = make_link([("APP_START Nope", "ERR unknown app")])
    with pytest.raises(HilError) as ei:
        verbs.app_start(link, "Nope")
    assert ei.value.code == "BAD_ARGS"
    assert "unknown app" in ei.value.message
    assert "APP_LIST" not in link.fake.written


def test_app_start_never_running_times_out(make_link) -> None:
    link = make_link([("APP_START Sampler", "OK"),
                      ("APP_LIST", "APPS: Sampler running=-")])
    with pytest.raises(HilError) as ei:
        verbs.app_start(link, "Sampler", wait_s=0.5)
    assert ei.value.code == "TIMEOUT"


@pytest.mark.parametrize("name", ["", "x" * 32, "has space"])
def test_app_start_rejects_bad_name(make_link, name: str) -> None:
    link = make_link()
    with pytest.raises(HilError) as ei:
        verbs.app_start(link, name)
    assert ei.value.code == "BAD_ARGS"
    assert link.fake.written == []


@pytest.mark.parametrize("fn,cmd", [
    (verbs.app_stop, "APP_STOP"),
    (verbs.app_destroy, "APP_DESTROY"),
    (verbs.app_self_close, "APP_SELF_CLOSE"),
])
def test_app_ok_verbs(make_link, fn, cmd: str) -> None:
    link = make_link([(cmd, "OK")])
    assert fn(link) == {"ok": True}
    assert link.fake.written == [cmd]


def test_app_versions_reads_until_end(make_link) -> None:
    link = make_link([("APP_VERSIONS", [
        "APPVER: crosspad-core id=- commit=2d7c54b ref=main dirty=0",
        "APPVER: crosspad-sampler id=sampler commit=af6dbd0 ref=v1.2 dirty=1",
        "APPVER: end count=2",
    ])])
    out = verbs.app_versions(link)
    assert out["count"] == 2
    assert len(out["components"]) == 2
    assert out["components"][1] == {
        "component": "crosspad-sampler", "id": "sampler", "commit": "af6dbd0",
        "ref": "v1.2", "dirty": True}


def test_app_versions_reads_board_row(make_link) -> None:
    link = make_link([("APP_VERSIONS", [
        "APPVER: crosspad-core id=- commit=2d7c54b ref=main dirty=0",
        "BOARD: fw=v2 pcb=20",
        "APPVER: end count=1",
    ])])
    assert verbs.app_versions(link)["board"] == {"fw": "v2", "pcb": "20"}


# ── kit_* ──────────────────────────────────────────────────────────────

def test_kit_list_reads_until_end(make_link) -> None:
    link = make_link([("KIT_LIST", [
        "KITS: 0:Basic,1:Drums",
        "KITS: 2:808",
        "KITS: end count=3 current=1",
    ])])
    assert verbs.kit_list(link) == {
        "kits": [{"id": 0, "name": "Basic"}, {"id": 1, "name": "Drums"},
                 {"id": 2, "name": "808"}],
        "current": 1, "count": 3}


def test_kit_list_rejects_a_listing_shorter_than_its_trailer(make_link) -> None:
    # A dropped line and a card with fewer kits look identical without the count.
    link = make_link([("KIT_LIST", [
        "KITS: 0:Basic",
        "KITS: end count=2 current=0",
    ])])
    with pytest.raises(HilError) as ei:
        verbs.kit_list(link)
    assert ei.value.code == "NOT_SUPPORTED"
    assert ei.value.details == {"got": 1, "want": 2}


def test_kit_list_of_an_empty_card(make_link) -> None:
    link = make_link([("KIT_LIST", ["KITS: end count=0 current=-1"])])
    assert verbs.kit_list(link) == {"kits": [], "current": -1, "count": 0}


def test_kit_status(make_link) -> None:
    link = make_link([("KIT_STATUS", "KITSTATUS: current=3 loading=0 pending=-1 name=DRUMS")])
    assert verbs.kit_status(link) == {
        "current": 3, "loading": False, "pending": -1, "name": "DRUMS"}


def test_kit_load_waits_until_landed_with_pending(make_link) -> None:
    # Two polls in flight (loading=1, then the request is pending behind another
    # load), third poll shows the kit landed. OK alone proves nothing
    # (hil_kit_churn: KIT_STATUS is the honest answer).
    link = make_link(
        [("KIT_LOAD 5", "OK")],
        sequences={"KIT_STATUS": [
            "KITSTATUS: current=2 loading=1 pending=-1 name=OLD",
            "KITSTATUS: current=2 loading=1 pending=5 name=OLD",
            "KITSTATUS: current=5 loading=0 pending=-1 name=DRUMS",
        ]},
    )
    out = verbs.kit_load(link, 5, wait_s=5.0)
    assert out == {"current": 5, "loading": False, "pending": -1, "name": "DRUMS"}
    assert link.fake.written[0] == "KIT_LOAD 5"
    assert link.fake.written.count("KIT_STATUS") == 3


def test_kit_load_current_but_still_loading_is_not_landed(make_link) -> None:
    link = make_link(
        [("KIT_LOAD 5", "OK")],
        sequences={"KIT_STATUS": [
            "KITSTATUS: current=5 loading=1 pending=-1 name=DRUMS",
            "KITSTATUS: current=5 loading=0 pending=-1 name=DRUMS",
        ]},
    )
    out = verbs.kit_load(link, 5, wait_s=5.0)
    assert out["loading"] is False
    assert link.fake.written.count("KIT_STATUS") == 2


def test_kit_load_timeout_reports_last_status(make_link) -> None:
    link = make_link([("KIT_LOAD 5", "OK"),
                      ("KIT_STATUS", "KITSTATUS: current=2 loading=1 pending=5 name=OLD")])
    with pytest.raises(HilError) as ei:
        verbs.kit_load(link, 5, wait_s=0.6)
    assert ei.value.code == "TIMEOUT"
    assert ei.value.details["status"]["pending"] == 5


def test_kit_load_err_bad_id(make_link) -> None:
    link = make_link([("KIT_LOAD 99", "ERR bad kit id")])
    with pytest.raises(HilError) as ei:
        verbs.kit_load(link, 99)
    assert ei.value.code == "BAD_ARGS"


def test_kit_load_negative_rejected_locally(make_link) -> None:
    link = make_link()
    with pytest.raises(HilError) as ei:
        verbs.kit_load(link, -1)
    assert ei.value.code == "BAD_ARGS"
    assert link.fake.written == []


def test_kit_list_scanning_is_a_retryable_timeout(make_link) -> None:
    # listKits() is still walking the card; the vector it is growing must not
    # be iterated, so this is TIMEOUT (retry) rather than a parse failure.
    link = make_link([("KIT_LIST", "ERR scanning")])
    with pytest.raises(HilError) as ei:
        verbs.kit_list(link)
    assert ei.value.code == "TIMEOUT"


def test_kit_load_scanning_is_a_retryable_timeout(make_link) -> None:
    # The id came from a listing the scan has not finished; also TIMEOUT, not
    # the BAD_ARGS a genuinely unknown id gets (test_kit_load_err_bad_id).
    link = make_link([("KIT_LOAD 5", "ERR scanning")])
    with pytest.raises(HilError) as ei:
        verbs.kit_load(link, 5)
    assert ei.value.code == "TIMEOUT"
    assert ei.value.details["kit_id"] == 5


# ── pad_* ──────────────────────────────────────────────────────────────

@pytest.mark.parametrize("idx,vel,ok", [
    (0, 127, True), (15, 1, True), (7, 0, True),
    (16, 100, False), (-1, 100, False), (3, 128, False), (3, -1, False),
])
def test_pad_press_range_check(make_link, idx: int, vel: int, ok: bool) -> None:
    link = make_link([(f"PAD_PRESS {idx} {vel}", "OK")])
    if ok:
        assert verbs.pad_press(link, idx, vel) == {"ok": True}
        assert link.fake.written == [f"PAD_PRESS {idx} {vel}"]
    else:
        with pytest.raises(HilError) as ei:
            verbs.pad_press(link, idx, vel)
        assert ei.value.code == "BAD_ARGS"
        assert link.fake.written == []


@pytest.mark.parametrize("idx,ok", [(0, True), (15, True), (16, False), (-1, False)])
def test_pad_release_range_check(make_link, idx: int, ok: bool) -> None:
    link = make_link([(f"PAD_RELEASE {idx}", "OK")])
    if ok:
        assert verbs.pad_release(link, idx) == {"ok": True}
    else:
        with pytest.raises(HilError) as ei:
            verbs.pad_release(link, idx)
        assert ei.value.code == "BAD_ARGS"


@pytest.mark.parametrize("idx,val,ok", [(0, 0, True), (15, 255, True), (2, 256, False), (16, 1, False)])
def test_pad_pressure_range_check(make_link, idx: int, val: int, ok: bool) -> None:
    link = make_link([(f"PAD_PRESSURE {idx} {val}", "OK")])
    if ok:
        assert verbs.pad_pressure(link, idx, val) == {"ok": True}
    else:
        with pytest.raises(HilError) as ei:
            verbs.pad_pressure(link, idx, val)
        assert ei.value.code == "BAD_ARGS"


def test_pad_stats_and_reset(make_link) -> None:
    link = make_link([("PAD_STATS", "PADSTATS: press=12 release=11 played=12 freeslots=6"),
                      ("PAD_STATS_RESET", "OK")])
    assert verbs.pad_stats(link) == {"press": 12, "release": 11, "played": 12, "freeslots": 6}
    assert "PAD_STATS_RESET" not in link.fake.written
    verbs.pad_stats(link, reset=True)
    assert link.fake.written[-2:] == ["PAD_STATS_RESET", "PAD_STATS"]


def test_pad_notes_int_keys(make_link) -> None:
    line = "PADNOTES:" + "".join(f" {i}:{36 + i}" for i in range(16))
    link = make_link([("PAD_NOTES", line)])
    notes = verbs.pad_notes(link)["notes"]
    assert len(notes) == 16
    assert notes[0] == 36 and notes[15] == 51
    assert all(isinstance(k, int) for k in notes)


def test_pad_info_raw(make_link) -> None:
    raw = "PADINFO: idx=3 num=39 kit=/sd/kits/DRUMS vol=100 pan=0 mode=0 dirty=0 count=1 | [0] 'kick.wav' 0..0"
    link = make_link([("PAD_INFO 3", raw)])
    assert verbs.pad_info(link, 3) == {"raw": raw}


# ── enc_* / ui ─────────────────────────────────────────────────────────

@pytest.mark.parametrize("delta,ok", [(1, True), (-16, True), (16, True), (0, False), (17, False)])
def test_enc_rotate_range(make_link, delta: int, ok: bool) -> None:
    link = make_link([(f"ENC_ROTATE {delta}", "OK")])
    if ok:
        assert verbs.enc_rotate(link, delta) == {"ok": True}
    else:
        with pytest.raises(HilError) as ei:
            verbs.enc_rotate(link, delta)
        assert ei.value.code == "BAD_ARGS"


@pytest.mark.parametrize("ms,ok", [(80, True), (1, True), (5000, True), (0, False), (5001, False)])
def test_enc_press_range(make_link, ms: int, ok: bool) -> None:
    link = make_link([(f"ENC_PRESS {ms}", "OK")])
    if ok:
        assert verbs.enc_press(link, ms) == {"ok": True}
        assert link.fake.written == [f"ENC_PRESS {ms}"]
    else:
        with pytest.raises(HilError):
            verbs.enc_press(link, ms)


def test_enc_group_via_transact_multi(make_link) -> None:
    link = make_link([("ENC_GROUP", [
        "ENCGROUP: count=3",
        "  [0] 0x3fcb1234 Sampler",
        "  [1] 0x3fcb5678 Sequencer",
        "  [2] 0x3fcb9abc Settings",
    ])])
    out = verbs.enc_group(link)
    assert [g["ref"] for g in out["group"]] == ["e0", "e1", "e2"]
    assert out["group"][2] == {"ref": "e2", "index": 2, "ptr": "0x3fcb9abc", "label": "Settings"}


def test_enc_group_empty(make_link) -> None:
    link = make_link([("ENC_GROUP", ["ENCGROUP: count=0"])])
    assert verbs.enc_group(link) == {"group": []}


def test_enc_focus_and_state(make_link) -> None:
    link = make_link([("ENC_FOCUS", "ENCFOCUS: obj=0x3fcb5678 idx=1 text=Sequencer"),
                      ("ENC_STATE", "ENC: group=0x3fcb0001 launcher=0x3fcb0001 owner=launcher")])
    assert verbs.enc_focus(link) == {"index": 1, "label": "Sequencer", "ptr": "0x3fcb5678"}
    st = verbs.enc_state(link)
    assert st["owner"] == "launcher"


def test_enc_focus_editing(make_link) -> None:
    link = make_link([("ENC_FOCUS", "ENCFOCUS: obj=0x3fcb5678 idx=12 edit=1 text=CHOKE 3")])
    assert verbs.enc_focus(link) == {"index": 12, "label": "CHOKE 3", "ptr": "0x3fcb5678",
                                     "editing": True}


def test_ui_state(make_link) -> None:
    link = make_link([("UI_STATE",
                       "UI: display=yes touch=yes drawer=1 lcd=80 rgb=60 theme=2 bt_icon=hidden app=-")])
    ui = verbs.ui_state(link)
    assert ui["display"] == "yes"
    assert int(ui["theme"]) == 2


# ── led / mem / stats ──────────────────────────────────────────────────

def test_led_state_colors_length_16(make_link) -> None:
    colors = ",".join(f"{i:02X}00FF" for i in range(16))
    link = make_link([("LED_STATE",
                       f"LEDS: bri=40 anim=0 coalesce=1 cfgbri=60 pwr=0x00 pwrN=3 txfail=0 colors={colors}")])
    out = verbs.led_state(link)
    assert len(out["colors"]) == 16
    assert out["colors"][0] == "0000FF" and out["colors"][15] == "0F00FF"
    assert out["brightness"] == 40
    assert out["txfail"] == 0


def test_mem_ints(make_link) -> None:
    link = make_link([("MEM", "MEM: int_free=76000 int_largest=32000 int_min=18000 int_blocks=900 "
                              "psram_free=6000000 psram_largest=4000000 psram_blocks=300")])
    out = verbs.mem(link)
    assert out["int_free"] == 76000 and out["psram_blocks"] == 300


def test_imu_ths_sends_both_thresholds(make_link) -> None:
    link = make_link([("IMU_THS 12 2", "OK")])
    verbs.imu_ths(link, 12, 2)


def test_imu_ths_rejects_out_of_range() -> None:
    with pytest.raises(HilError):
        verbs.imu_ths(None, 32, 0)  # type: ignore[arg-type]


def test_ulp_counts(make_link) -> None:
    link = make_link([("ULP", "ULP: running=1 period_ms=100 heartbeat=617")])
    out = verbs.ulp(link)
    assert out == {"running": 1, "period_ms": 100, "heartbeat": 617}


def test_mem_blocks_two_lines(make_link) -> None:
    link = make_link([("MEM_BLOCKS", [
        "MEMBLK: biggest_used=16384 | <=64 u=400/12000B f=10/300B | <=128 u=100/9000B f=2/200B "
        "| <=256 u=50/9000B f=1/200B | <=512 u=20/7000B f=0/0B | <=1k u=10/8000B f=1/700B "
        "| <=2k u=5/8000B f=0/0B | <=8k u=4/20000B f=1/5000B | <=big u=2/30000B f=1/20000B",
        "MEMBIG: @3fcb0000=16384 @3fcc0000=12288",
    ])])
    out = verbs.mem_blocks(link)
    assert out["summary"]["biggest_used"] == 16384
    assert len(out["summary"]["buckets"]) == 8
    assert out["big"] == [{"addr": 0x3FCB0000, "size": 16384},
                          {"addr": 0x3FCC0000, "size": 12288}]


def test_cdc_stats(make_link) -> None:
    link = make_link([("LVGL_STATS", "LVGLSTATS: refr=3 refr_max_us=90 refr_avg_us=50 "
                                     "render_us=100 render_max_us=60 flush_wait_us=40 "
                                     "flush_wait_max_us=30 flushes=9")])
    assert verbs.lvgl_stats(link)["refr_max_us"] == 90
    link = make_link([("CDC_STATS", "CDCSTATS: rx=120 cmds=118 drop=2")])
    assert verbs.cdc_stats(link) == {"rx": 120, "cmds": 118, "drop": 2, "maxcmd_us": 0}
    link = make_link([("CDC_STATS", "CDCSTATS: rx=120 cmds=118 drop=2 maxcmd_us=412 maxgap_us=9")])
    assert verbs.cdc_stats(link)["maxcmd_us"] == 412


def test_audio_level_and_smpl_peak(make_link) -> None:
    link = make_link([("AUDIO_LEVEL", "AUDIOLVL: L=0.01234 R=0.00500 amp=1 allowed=1"),
                      ("SMPL_PEAK", "SMPLPEAK: 1234 free=7")])
    lvl = verbs.audio_level(link)
    assert lvl["amp"] is True and abs(lvl["left"] - 0.01234) < 1e-9
    assert verbs.smpl_peak(link) == {"peak": 1234, "free": 7}


def test_ingain_reads_sets_and_refuses(make_link) -> None:
    link = make_link([("INGAIN 1 24", "INGAIN: IN1=24 IN2=24"),
                      ("INGAIN", "INGAIN: IN1=24 IN2=0")])
    assert verbs.ingain(link) == {"IN1": 24, "IN2": 0}
    assert verbs.ingain(link, 1, 24) == {"IN1": 24, "IN2": 24}
    with pytest.raises(HilError) as ei:
        verbs.ingain(link, 1)
    assert ei.value.code == "BAD_ARGS"
    none = make_link([("INGAIN", "ERR INGAIN none")])
    with pytest.raises(HilError) as ei:
        verbs.ingain(none)
    assert ei.value.code == "NOT_SUPPORTED"


# ── ble_* ──────────────────────────────────────────────────────────────

def test_ble_status_ints(make_link) -> None:
    link = make_link([("BLE_STATUS",
                       "BLE: supported=1 running=1 state=connected mode=server self=aa:bb:cc:dd:ee:ff "
                       "peer=11:22:33:44:55:66 itvl=15 txoff=36 rxoff=0 tx_msg=181 tx_pkt=90 "
                       "tx_drop=0 tx_err=0 rx_msg=3 rx_pkt=3")])
    st = verbs.ble_status(link)
    assert st["supported"] == 1 and st["running"] == 1
    assert st["itvl"] == 15 and st["txoff"] == 36 and st["tx_msg"] == 181
    assert st["state"] == "connected" and st["mode"] == "server"
    assert st["peer"] == "11:22:33:44:55:66"


def test_ble_start_modes(make_link) -> None:
    link = make_link([("BLE_START", "OK"), ("BLE_START 0", "OK"), ("BLE_START 1", "OK")])
    assert verbs.ble_start(link) == {"ok": True}
    assert verbs.ble_start(link, 0) == {"ok": True}
    assert verbs.ble_start(link, 1) == {"ok": True}
    assert link.fake.written == ["BLE_START", "BLE_START 0", "BLE_START 1"]
    with pytest.raises(HilError) as ei:
        verbs.ble_start(link, 2)
    assert ei.value.code == "BAD_ARGS"


def test_ble_start_and_enable_deferred_while_wifi_holds_the_radio(make_link) -> None:
    link = make_link([("BLE_START", "OK deferred=wifi"), ("BLE_ENABLE 1", "OK deferred=wifi")])
    assert verbs.ble_start(link) == {"ok": True, "deferred": True}
    assert verbs.ble_enable(link, True) == {"ok": True, "deferred": True}


def test_ble_start_err_is_not_supported(make_link) -> None:
    link = make_link([("BLE_START", "ERR")])
    with pytest.raises(HilError) as ei:
        verbs.ble_start(link)
    assert ei.value.code == "NOT_SUPPORTED"


_BLE_ST = "BLE: supported=1 running={run} state={state} mode=server self=- peer=- itvl=0 txoff=0 rxoff=0 " \
          "tx_msg=0 tx_pkt=0 tx_drop=0 tx_err=0 rx_msg=0 rx_pkt=0 req={req}"


def test_ble_start_queued_waits_for_the_outcome(make_link, monkeypatch) -> None:
    """The board runs the stack job off the CDC task: OK queued, then BLE_STATUS req=."""
    monkeypatch.setattr(verbs, "BLE_JOB_POLL_S", 0.0)
    link = make_link(sequences={
        "BLE_START 1": ["OK queued"],
        "BLE_STATUS": [_BLE_ST.format(run=0, state="off", req="pending"),
                       _BLE_ST.format(run=1, state="advertising", req="ok")],
    })
    assert verbs.ble_start(link, 1) == {"ok": True}
    assert link.fake.written == ["BLE_START 1", "BLE_STATUS", "BLE_STATUS"]


def test_ble_queued_deferred_and_refused(make_link, monkeypatch) -> None:
    monkeypatch.setattr(verbs, "BLE_JOB_POLL_S", 0.0)
    link = make_link(sequences={
        "BLE_ENABLE 1": ["OK queued"],
        "BLE_START": ["OK queued"],
        "BLE_STATUS": [_BLE_ST.format(run=0, state="off", req="deferred"),
                       _BLE_ST.format(run=0, state="off", req="err")],
    })
    assert verbs.ble_enable(link, True) == {"ok": True, "deferred": True}
    with pytest.raises(HilError) as ei:
        verbs.ble_start(link)
    assert ei.value.code == NOT_SUPPORTED


def test_ble_stop_queued(make_link, monkeypatch) -> None:
    monkeypatch.setattr(verbs, "BLE_JOB_POLL_S", 0.0)
    link = make_link(sequences={
        "BLE_STOP": ["OK queued"],
        "BLE_STATUS": [_BLE_ST.format(run=0, state="off", req="ok")],
    })
    assert verbs.ble_stop(link) == {"ok": True}


def test_ble_stop_legacy_ok(make_link) -> None:
    link = make_link([("BLE_STOP", "OK")])
    assert verbs.ble_stop(link) == {"ok": True}


def test_ble_scan_devices(make_link) -> None:
    link = make_link([("BLE_SCAN 3000", "OK"),
                      ("BLE_DEVICES", "BLEDEV: count=2 | aa:bb:cc:dd:ee:01 Piano -60 | aa:bb:cc:dd:ee:02 - -71")])
    assert verbs.ble_scan(link, 3000) == {"ok": True}
    dev = verbs.ble_devices(link)
    assert dev["count"] == 2
    assert dev["devices"][0] == {"addr": "aa:bb:cc:dd:ee:01", "name": "Piano", "rssi": -60}


@pytest.mark.parametrize("addr,ok", [
    ("aa:bb:cc:dd:ee:ff", True), ("AA:BB:CC:DD:EE:FF", True),
    ("aa-bb-cc-dd-ee-ff", False), ("aabbccddeeff", False), ("", False),
])
def test_ble_connect_addr_format(make_link, addr: str, ok: bool) -> None:
    link = make_link([(f"BLE_CONNECT {addr}", "OK")])
    if ok:
        assert verbs.ble_connect(link, addr) == {"ok": True}
    else:
        with pytest.raises(HilError) as ei:
            verbs.ble_connect(link, addr)
        assert ei.value.code == "BAD_ARGS"


def test_ble_send_txoff_disconnect_stop(make_link) -> None:
    link = make_link([("BLE_SEND 60 100", "OK"), ("BLE_TXOFF 36", "OK"),
                      ("BLE_DISCONNECT", "OK"), ("BLE_STOP", "OK")])
    assert verbs.ble_send(link, 60) == {"ok": True}
    assert verbs.ble_txoff(link, 36) == {"ok": True}
    assert verbs.ble_disconnect(link) == {"ok": True}
    assert verbs.ble_stop(link) == {"ok": True}
    with pytest.raises(HilError):
        verbs.ble_send(link, 128)
    with pytest.raises(HilError):
        verbs.ble_txoff(link, 128)


# ── misc ───────────────────────────────────────────────────────────────

def test_audio_tasks(make_link) -> None:
    link = make_link([("AUDIO_TASKS 1", "OK"), ("AUDIO_TASKS 0", "OK")])
    assert verbs.audio_tasks(link, True) == {"ok": True}
    assert verbs.audio_tasks(link, False) == {"ok": True}
    assert link.fake.written == ["AUDIO_TASKS 1", "AUDIO_TASKS 0"]


def test_bootloader_request_is_fire_and_forget(make_link) -> None:
    link = make_link()  # device never replies: it reboots into ROM download mode
    assert verbs.bootloader_request(link) == {"sent": True}
    assert link.fake.written == ["BOOTLOADER_REQUEST"]


def test_stm_dfu(make_link) -> None:
    link = make_link([("STM_DFU", "OK")])
    assert verbs.stm_dfu(link) == {"sent": True}


class _CapturingLink:
    """Enough of `CdcLink` for `pitched_bench`: records the `timeout_s` it was given."""

    def __init__(self, lines: list[str]) -> None:
        self.calls: list[dict] = []
        self._lines = lines

    def transact_multi(self, cmd: str, prefix: str, *, end: str, timeout_s: float) -> list[str]:
        self.calls.append({"cmd": cmd, "timeout_s": timeout_s})
        return self._lines


def test_pitched_bench_scales_timeout_with_iters() -> None:
    link = _CapturingLink(["BENCH: ifft1024_us=12", "OK"])
    verbs.pitched_bench(link, 5000)
    assert link.calls[0]["timeout_s"] == pytest.approx(10.0 + 5000 * 0.05)


def test_pitched_bench_caps_timeout_at_300s() -> None:
    link = _CapturingLink(["OK"])
    verbs.pitched_bench(link, 1_000_000)
    assert link.calls[0]["timeout_s"] == pytest.approx(300.0)


def test_pitched_bench_respects_an_explicit_timeout() -> None:
    link = _CapturingLink(["OK"])
    verbs.pitched_bench(link, 100, timeout_s=2.0)
    assert link.calls[0]["timeout_s"] == pytest.approx(2.0)


def test_pitched_bench_default_iters_still_gets_the_base_timeout() -> None:
    link = _CapturingLink(["OK"])
    verbs.pitched_bench(link)
    assert link.calls[0]["timeout_s"] == pytest.approx(10.0 + 1 * 0.05)


# ── firmware library / A/B switch ────────────────────────────────────────

def test_fw_slots_reads_both_rows_and_the_trailer(make_link) -> None:
    link = make_link([("FW_SLOTS", [
        "FWSLOT: slot=A running=1 state=confirmed pkg=bench-current ver=1.0.0 kit=full",
        "FWSLOT: slot=B running=0 state=empty pkg=- ver=- kit=-",
        "FWSLOT: rollback=1",
    ])])
    assert verbs.fw_slots(link) == {
        "slots": [
            {"slot": "A", "running": True, "state": "confirmed", "pkg": "bench-current",
             "ver": "1.0.0", "kit": "full"},
            {"slot": "B", "running": False, "state": "empty", "pkg": None, "ver": None,
             "kit": None},
        ],
        "rollback": True,
    }


def test_fw_slots_unparseable_reply_is_not_supported(make_link) -> None:
    link = make_link([("FW_SLOTS", ["FWSLOT: garbage"])])
    with pytest.raises(HilError) as ei:
        verbs.fw_slots(link)
    assert ei.value.code == "NOT_SUPPORTED"


def test_fw_list_reads_rows_and_the_count_trailer(make_link) -> None:
    link = make_link([("FW_LIST", [
        "FWPKG: name=bench-current kit=full ver=1.0.0 rev=1 size=2300000 assets=1 "
        "settings=0 verdict=ok",
        "FWPKG: name=old-v1 kit=full ver=0.9.0 rev=3 size=2100000 assets=0 settings=0 "
        "verdict=built for another board revision",
        "FWPKG: count=2",
    ])])
    out = verbs.fw_list(link)
    assert out["count"] == 2
    assert out["packages"][0] == {
        "name": "bench-current", "kit": "full", "ver": "1.0.0", "rev": 1, "size": 2300000,
        "assets": True, "settings": False, "verdict": "ok",
    }
    # a verdict with spaces is the last field on the row and must not be truncated
    assert out["packages"][1]["verdict"] == "built for another board revision"


def test_fw_list_trailer_mismatch_is_not_supported(make_link) -> None:
    link = make_link([("FW_LIST", [
        "FWPKG: name=a kit=full ver=1.0.0 rev=1 size=1 assets=0 settings=0 verdict=ok",
        "FWPKG: count=2",
    ])])
    with pytest.raises(HilError) as ei:
        verbs.fw_list(link)
    assert ei.value.code == "NOT_SUPPORTED"
    assert ei.value.details == {"got": 1, "want": 2}


def test_fw_list_busy_error_keeps_the_fwpkg_prefix(make_link) -> None:
    # Unlike KIT_LIST's bare "ERR scanning", FW_LIST's busy/no-task refusals
    # keep the "FWPKG:" prefix (main/hil_control.cpp handle_fw_list), so the
    # generic multi-line error detection in cdc.py never sees them — this call
    # has to recognise the single "FWPKG: ERR busy" line for itself.
    link = make_link([("FW_LIST", ["FWPKG: ERR busy"])])
    with pytest.raises(HilError) as ei:
        verbs.fw_list(link)
    assert ei.value.code == "NOT_SUPPORTED"
    assert "ERR busy" in ei.value.message


def test_fw_install_ok(make_link) -> None:
    link = make_link([("FW_INSTALL bench-current", "OK")])
    assert verbs.fw_install(link, "bench-current") == {"ok": True, "warn_stm_proto_older": False}
    assert link.fake.written == ["FW_INSTALL bench-current"]


def test_fw_install_force_and_warn(make_link) -> None:
    link = make_link([("FW_INSTALL bench-current FORCE", "OK warn=stm_proto_older")])
    out = verbs.fw_install(link, "bench-current", force=True)
    assert out == {"ok": True, "warn_stm_proto_older": True}
    assert link.fake.written == ["FW_INSTALL bench-current FORCE"]


def test_fw_install_refused(make_link) -> None:
    link = make_link([("FW_INSTALL missing", "ERR no such package")])
    with pytest.raises(HilError) as ei:
        verbs.fw_install(link, "missing")
    assert ei.value.code == "NOT_SUPPORTED"


def test_fw_install_rejects_bad_names_locally(make_link) -> None:
    link = make_link()
    with pytest.raises(HilError) as ei:
        verbs.fw_install(link, "has space")
    assert ei.value.code == "BAD_ARGS"
    assert link.fake.written == []


def test_fw_install_status_polls_bare(make_link) -> None:
    link = make_link([("FW_INSTALL", "FWINST: phase=writing percent=42 error=-")])
    assert verbs.fw_install_status(link) == {"phase": "writing", "percent": 42, "error": None}


def test_fw_install_status_carries_the_error_text(make_link) -> None:
    link = make_link([("FW_INSTALL", "FWINST: phase=failed percent=10 error=checksum")])
    assert verbs.fw_install_status(link) == {"phase": "failed", "percent": 10, "error": "checksum"}


def test_fw_switch_ok(make_link) -> None:
    link = make_link([("FW_SWITCH", "OK reboot=1")])
    assert verbs.fw_switch(link) == {"ok": True, "reboot": True}


def test_fw_switch_refused(make_link) -> None:
    link = make_link([("FW_SWITCH", "ERR running slot not confirmed yet")])
    with pytest.raises(HilError) as ei:
        verbs.fw_switch(link)
    assert ei.value.code == "NOT_SUPPORTED"
    assert "not confirmed" in ei.value.message


def test_fw_confirm(make_link) -> None:
    link = make_link([("FW_CONFIRM", "OK")])
    assert verbs.fw_confirm(link) == {"ok": True}


def test_app_sha(make_link) -> None:
    sha = "ab" * 32
    link = make_link([("APP_SHA", f"APPSHA: {sha}")])
    assert verbs.app_sha(link) == {"sha256": sha}


def test_app_sha_unparseable_is_not_supported(make_link) -> None:
    link = make_link([("APP_SHA", "APPSHA: not-hex")])
    with pytest.raises(HilError) as ei:
        verbs.app_sha(link)
    assert ei.value.code == "NOT_SUPPORTED"


def test_usb_msc_status(make_link) -> None:
    link = make_link([("USB_MSC", "USB_MSC: active=0 card=1 state=waiting size_mb=15000 "
                       "read_kb=0 write_kb=0")])
    assert verbs.usb_msc_status(link) == {
        "active": False, "card": True, "state": "waiting", "size_mb": 15000,
        "read_kb": 0, "write_kb": 0,
    }


def test_cdc_stats_reads_the_main_task_stack_fields(make_link) -> None:
    link = make_link([("CDC_STATS", "CDCSTATS: rx=5 cmds=5 drop=0 maxcmd_us=412 maxgap_us=9 "
                                    "stack_min=1536 stack_verb=MIX_PRESET stack_last=2600 "
                                    "maxcmd=KIT_LOAD 3")])
    stats = verbs.cdc_stats(link)
    assert (stats["stack_min"], stats["stack_verb"], stats["stack_last"]) == (1536, "MIX_PRESET", 2600)


def test_ota_prefs_reads_auto_accept(make_link) -> None:
    link = make_link([("OTA_PREFS", "OTA_PREFS: auto_accept=1")])
    assert verbs.ota_prefs(link) is True
    link = make_link([("OTA_PREFS", "OTA_PREFS: garbage")])
    with pytest.raises(HilError) as ei:
        verbs.ota_prefs(link)
    assert ei.value.code == "NOT_SUPPORTED"


def test_usb_guard_reads_and_sets(make_link) -> None:
    link = make_link([("USB_GUARD", "USB_GUARD: 1"), ("USB_GUARD 0", "USB_GUARD: 0")])
    assert verbs.usb_guard(link) is True
    assert verbs.usb_guard(link, False) is False
    assert link.fake.written == ["USB_GUARD", "USB_GUARD 0"]


def test_unparseable_reply_is_not_supported(make_link) -> None:
    link = make_link([("KIT_STATUS", "KITSTATUS: garbage")])
    with pytest.raises(HilError) as ei:
        verbs.kit_status(link)
    assert ei.value.code == "NOT_SUPPORTED"


# ── net_mode_* / wifi_* ────────────────────────────────────────────────

def test_net_mode_status(make_link) -> None:
    link = make_link([("NET_MODE", "NET_MODE: active=1 tab=wifi")])
    assert verbs.net_mode_status(link) == {"active": True, "tab": "wifi"}


def test_net_mode_set_reboot_and_unchanged(make_link) -> None:
    link = make_link([("NET_MODE 1 TAB=wifi", "OK reboot=1")])
    assert verbs.net_mode_set(link, True, tab="wifi") == {"ok": True, "reboot": True}
    link = make_link([("NET_MODE 0", "OK unchanged")])
    assert verbs.net_mode_set(link, False) == {"ok": True, "reboot": False}


def test_net_mode_set_rejects_unknown_tab(make_link) -> None:
    with pytest.raises(HilError) as ei:
        verbs.net_mode_set(make_link(), True, tab="nope")
    assert ei.value.code == "BAD_ARGS"


def test_wifi_set_quotes_and_short_pass_is_an_error(make_link) -> None:
    link = make_link([('WIFI_SET "Dom Kowalskich" "short"', "ERR short_pass")])
    with pytest.raises(HilError) as ei:
        verbs.wifi_set(link, "Dom Kowalskich", "short")
    assert "short_pass" in ei.value.message


def test_wifi_set_refuses_a_quote_in_the_ssid(make_link) -> None:
    with pytest.raises(HilError) as ei:
        verbs.wifi_set(make_link(), 'a"b', "password1")
    assert ei.value.code == "BAD_ARGS"


def test_wifi_list_and_scan(make_link) -> None:
    link = make_link([("WIFI_LIST", r"WIFI_NET: count=2 | a \| b | c"),
                      ("WIFI_SCAN", "WIFI_AP: count=1 | -48 wpa2 Dom")])
    assert verbs.wifi_list(link) == ["a | b", "c"]
    assert verbs.wifi_scan(link) == [{"rssi": -48, "auth": "wpa2", "ssid": "Dom"}]


def test_verbs_table_covers_every_public_function() -> None:
    public = {n for n in dir(verbs) if not n.startswith("_")
              and callable(getattr(verbs, n)) and getattr(verbs, n).__module__ == verbs.__name__}
    assert set(verbs.VERBS) == public
    assert verbs.VERBS["kit_load"] is verbs.kit_load


def test_clock_status_parses_types(make_link) -> None:
    link = make_link([("CLOCK_STATUS",
                       "CLOCK: src=host bpm=120.04 running=1 ticks_in=480 ticks_out=480 "
                       "last_gap_us=20833 max_jitter_us=812 max_late_us=0")])
    out = verbs.clock_status(link)
    assert out["src"] == "host" and out["running"] is True
    assert abs(out["bpm"] - 120.04) < 1e-6
    assert out["ticks_in"] == 480 and out["max_jitter_us"] == 812


def test_clock_status_reset_sends_the_flag(make_link) -> None:
    link = make_link([("CLOCK_STATUS RESET", "CLOCK: src=int bpm=120.00 running=0 ticks_in=0 "
                                             "ticks_out=10 last_gap_us=0 max_jitter_us=0 "
                                             "max_late_us=0")])
    assert verbs.clock_status(link, reset=True)["src"] == "int"


# ── firmware updates ──────────────────────────────────────────────────────

def test_fw_upd_status_reads_the_line(make_link) -> None:
    link = make_link([("FW_UPD_STATUS", "FWUPD: state=up_to_date running=v1.3.0 kit=full latest=- "
                                        "channel=beta progress=0 last_check=0 err=-")])
    st = verbs.fw_upd_status(link)
    assert st["state"] == "up_to_date" and st["latest"] is None and st["channel"] == "beta"


def test_fw_catalog_with_a_kit_filter(make_link) -> None:
    link = make_link([("FW_CATALOG full", "FWCAT: count=1 total=1 | full 1.3.0 stable official 0")])
    out = verbs.fw_catalog(link, "full")
    assert out["versions"][0]["version"] == "1.3.0" and out["versions"][0]["on_card"] is False


def test_fw_download_sends_install_and_reports_busy(make_link) -> None:
    link = make_link([("FW_DOWNLOAD full 1.3.0 INSTALL", "OK"), ("FW_DOWNLOAD full 1.3.0", "ERR busy")])
    assert verbs.fw_download(link, "full", "1.3.0", install=True) == {"ok": True}
    with pytest.raises(HilError) as ei:
        verbs.fw_download(link, "full", "1.3.0")
    assert "busy" in ei.value.message


def test_fw_download_refuses_a_token_with_a_space(make_link) -> None:
    link = make_link([])
    with pytest.raises(HilError) as ei:
        verbs.fw_download(link, "full kit", "1.3.0")
    assert ei.value.code == "BAD_ARGS"


def test_fw_tokens_take_what_the_device_holds(make_link) -> None:
    v24 = "v1.0.0-" + "a" * 17                       # 'v' + a 23-character version
    link = make_link([(f"FW_DOWNLOAD {'k' * 15} {v24}", "OK")])
    assert verbs.fw_download(link, "k" * 15, v24) == {"ok": True}
    for kit, version in (("k" * 16, "1.3.0"), ("full", v24 + "a")):
        with pytest.raises(HilError) as ei:
            verbs.fw_download(make_link([]), kit, version)
        assert ei.value.code == "BAD_ARGS"


def test_fw_upd_cfg_builds_its_tokens(make_link) -> None:
    link = make_link([("FW_UPD_CFG boot=1 gc_mb=40", "OK"), ("FW_UPD_CFG gc_mb=0", "OK")])
    assert verbs.fw_upd_cfg(link, boot=True, gc_mb=40) == {"ok": True}
    assert verbs.fw_upd_cfg(link, gc_mb=0) == {"ok": True}
    with pytest.raises(HilError):
        verbs.fw_upd_cfg(link)
    with pytest.raises(HilError):
        verbs.fw_upd_cfg(link, gc_mb=-1)


def test_fw_upd_channel_and_url(make_link) -> None:
    link = make_link([("FW_UPD_CHANNEL full beta", "OK"), ("FW_UPD_URL http://10.0.0.5:8765", "OK"),
                      ("FW_UPD_URL DEFAULT", "OK")])
    verbs.fw_upd_channel(link, "full", "beta")
    verbs.fw_upd_url(link, "http://10.0.0.5:8765")
    verbs.fw_upd_url(link, None)
    with pytest.raises(HilError):
        verbs.fw_upd_channel(link, "full", "nightly")


def test_fw_upd_url_refuses_whitespace(make_link) -> None:
    link = make_link([])
    with pytest.raises(HilError) as ei:
        verbs.fw_upd_url(link, "http://10.0.0.5:8765 x")
    assert ei.value.code == "BAD_ARGS"


def test_fw_ui_reads_the_line(make_link) -> None:
    link = make_link([("FW_UI", "FWUI: tab=updates sel=- card=- items=1 selkey=- host=app store=missing busy=0")])
    ui = verbs.fw_ui(link)
    assert ui["tab"] == "updates" and ui["sel"] is None and ui["store"] == "missing" and ui["items"] == 1


# ── live_jam verbs ───────────────────────────────────────────────────────

def test_pad_stats_keeps_the_loss_counters(make_link) -> None:
    link = make_link([("PAD_STATS", "PADSTATS: press=10 release=10 inj_drop=2 played=9 last=3:100 "
                                    "freeslots=7 stm=0 at=0 rxmsg=5 uart_ovf=1 sbuf_drop=0 "
                                    "usb_tx=9 usb_drop=3")])
    st = verbs.pad_stats(link)
    assert st["inj_drop"] == 2 and st["uart_ovf"] == 1 and st["usb_drop"] == 3
    assert st["press"] == 10 and st["played"] == 9


def test_seq_dump_click_and_mix(make_link) -> None:
    link = make_link([
        ("SEQ_DUMP", ["SEQDUMP: seq=0 i=0 pos=0 pad=0 vel=100",
                      "SEQDUMP: seq=0 i=1 pos=22050 pad=1 vel=90", "SEQDUMP: end"]),
        ("CLICK", "CLICK: running=1 bpm=120.0 beats=4 bars=2 bar=1 beat=2 pos=33075/176400"),
        ("MIX_LVL", ["MIXLVL: ch=0 name=Sampler pk=0.1000/0.0500 vol=0.80 mute=0 route=111",
                     "MIXLVL: ch=3 name=USB pk=0.0500/0.0500 vol=1.00 mute=0 route=110",
                     "MIXLVL: ch=4 name=Click pk=0.0000/0.0000 vol=1.00 mute=0 route=001",
                     "MIXLVL: ch=5 name=Song pk=0.0000/0.0000 vol=1.00 mute=1 route=000",
                     "MIXLVL: out=2 pk=0.1000/0.1000 vol=1.00 mute=0", "MIXLVL: end"]),
    ])
    assert verbs.seq_dump(link) == [{"seq": 0, "i": 0, "pos": 0, "pad": 0, "vel": 100},
                                    {"seq": 0, "i": 1, "pos": 22050, "pad": 1, "vel": 90}]
    c = verbs.click(link)
    assert c["running"] == 1 and c["pos"] == 33075 and c["loop"] == 176400
    desk = verbs.mix_lvl(link)
    assert desk["channels"][1]["route"] == [1, 1, 0] and desk["outputs"][0]["out"] == 2
    # leading zeros are sends that are off, not digits to drop
    assert desk["channels"][2]["route"] == [0, 0, 1] and desk["channels"][3]["route"] == [0, 0, 0]
    assert desk["channels"][0]["pk_l"] == pytest.approx(0.1)


def test_mix_preset_slots_and_route(make_link) -> None:
    link = make_link([
        ("MIX_PRESET LOAD 4", "MIXPRE: ok=1 undo=1 dir=/sdcard/crosspad/mixer 1=f:STREAM "
                              "2=f:BEATS 3=f:SAMPLING 4=s:JAM-DUCK 5=e:Empty 6=s:HILHOME"),
        ("MIX_ROUTE 3 2 1", "OK"),
    ])
    pr = verbs.mix_preset(link, "LOAD 4")
    assert pr["ok"] and pr["undo"] and pr["slots"][4] == {"kind": "saved", "name": "JAM-DUCK"}
    assert pr["slots"][1]["kind"] == "factory" and pr["slots"][5]["kind"] == "empty"
    assert verbs.mix_route(link, 3, 2, True) == {"ok": True}


def test_stool_names_come_last(make_link) -> None:
    link = make_link([("STOOL REC", "STOOL: ok=1 in=2 pk=10/10 rec=1 song=0 wait=0 from=0 "
                                    "frames=4410 sr=44100 job=0 prog=0 prev=0 at=0 ov=0..0/0 "
                                    "cols=0 hops=0 recs=3 slices=0: n=0 folders=0 more=0 "
                                    "file=/sdcard/REC/rec_0007.wav err= result= first=a.wav "
                                    "dir=- names=")])
    st = verbs.stool(link, "REC")
    assert st["rec"] == 1 and st["sr"] == 44100 and st["file"] == "/sdcard/REC/rec_0007.wav"


def test_smpl_peak_and_counters_on_the_native_engine(make_link) -> None:
    line = ("SMPLPEAK: 5692 free=48 poly=16 rt_max_us=1122 rxskip=3 noblock=0 underrun=2 "
            "unpaced=0 rt_busy_max_us=1301 overruns=1")
    link = make_link([("SMPL_PEAK", line), ("SMPL_PEAK", line)])
    assert verbs.smpl_peak(link) == {"peak": 5692, "free": 48}
    c = verbs.smpl_counters(link)
    assert c["poly"] == 16 and c["rxskip"] == 3 and c["unpaced"] == 0 and c["underrun"] == 2
    assert "load_main" not in c


def test_smpl_engine_status_and_select(make_link) -> None:
    link = make_link([("SMPL_ENGINE", "SMPLENG: engine=none state=0 why=- pitch=0,0,0 slots=0 "
                                      "pool_blocks=0/0 voices=0/0 lim=0/0 engines=native"),
                      ("SMPL_ENGINE native", "OK loading"),
                      ("SMPL_ENGINE marcel", "ERR only_native")])
    st = verbs.smpl_engine(link)
    assert st["engine"] == "none" and st["engines"] == ["native"] and "kind" not in st
    assert verbs.smpl_engine_select(link, "native") == {"ok": True, "reply": "OK loading"}
    with pytest.raises(HilError) as e:
        verbs.smpl_engine_select(link, "marcel")
    assert e.value.code == NOT_SUPPORTED


def test_smpl_engine_select_refuses_an_unknown_engine(make_link) -> None:
    with pytest.raises(HilError) as e:
        verbs.smpl_engine_select(make_link([]), "plaits")
    assert e.value.code == BAD_ARGS


def test_smpl_pitch(make_link) -> None:
    link = make_link([("SMPL_PITCH 0 12", "SMPLPITCH: pad=0 semi=12 engine=native")])
    assert verbs.smpl_pitch(link, 0, 12) == {"pad": 0, "semi": 12, "engine": "native"}
    with pytest.raises(HilError):
        verbs.smpl_pitch(link, 16, 0)
    with pytest.raises(HilError):
        verbs.smpl_pitch(link, 0, 25)


def test_mix_mute_and_vol_verbs(make_link) -> None:
    link = make_link([("MIX_MUTE 0 1", "OK"), ("MIX_VOL 0 80", "OK"),
                      ("MIX_OUT_MUTE 2 0", "OK"), ("MIX_OUT_VOL 1 100", "OK"),
                      ("MIX_MUTE 9 1", "ERR MIX_MUTE ch 0|1")])
    assert verbs.mix_mute(link, 0, True) == {"ok": True}
    assert verbs.mix_vol(link, 0, 80) == {"ok": True}
    assert verbs.mix_out_mute(link, 2, False) == {"ok": True}
    assert verbs.mix_out_vol(link, 1, 100) == {"ok": True}
    assert link.fake.written == ["MIX_MUTE 0 1", "MIX_VOL 0 80", "MIX_OUT_MUTE 2 0", "MIX_OUT_VOL 1 100"]
    with pytest.raises(HilError) as ei:
        verbs.mix_mute(link, 9, True)
    assert ei.value.code == BAD_ARGS
    with pytest.raises(HilError) as ei:
        verbs.mix_vol(link, 0, 101)
    assert ei.value.code == BAD_ARGS


def test_smpl_poly_retries_while_a_take_holds_the_locks(make_link) -> None:
    """The board answers ERR busy instead of waiting on the CDC task; the helper retries.

    Real sleeps (2 x 0.25 s): patching verbs.time.sleep patches the time module
    itself, which also starves the fake link's reader thread -- a flaky TIMEOUT."""
    link = make_link(sequences={
        "SMPL_POLY 20 10": ["ERR busy", "ERR busy", "SMPLPOLY: max_voices=20 max_streaming=10"],
    })
    assert verbs.smpl_poly(link, 20, 10) == {"max_voices": 20, "max_streaming": 10}
    assert link.fake.written.count("SMPL_POLY 20 10") == 3


def test_smpl_poly_other_err_is_not_retried(make_link) -> None:
    link = make_link([("SMPL_POLY 20", "ERR no_kit")])
    with pytest.raises(HilError):
        verbs.smpl_poly(link, 20)
    assert link.fake.written == ["SMPL_POLY 20"]


def test_smpl_poly_reads_and_sets_the_native_caps(make_link) -> None:
    link = make_link([("SMPL_POLY", "SMPLPOLY: max_voices=16 max_streaming=8"),
                      ("SMPL_POLY 20 10", "SMPLPOLY: max_voices=20 max_streaming=10")])
    assert verbs.smpl_poly(link) == {"max_voices": 16, "max_streaming": 8}
    assert verbs.smpl_poly(link, 20, 10) == {"max_voices": 20, "max_streaming": 10}
    with pytest.raises(HilError):
        verbs.smpl_poly(link, 0)
    with pytest.raises(HilError):
        verbs.smpl_poly(link, streaming=4)


def test_smpl_pool_on_a_native_only_image(make_link) -> None:
    link = make_link([("SMPL_POOL", "ERR only_native")])
    with pytest.raises(HilError) as e:
        verbs.smpl_pool(link)
    assert e.value.code == NOT_SUPPORTED


def test_smpl_pool_splits_the_fractions(make_link) -> None:
    link = make_link([("SMPL_POOL", "SMPLPOOL: layout=1 busy=3/20 peak=7 voices=4 used=8/16 "
                                    "audits=10 torn=0 late=0 esc=0 fixed=0 leak=0 unheld=0 "
                                    "shared=0 bkg=port bkg_port=5 bkg_lib=0 bkg_stale=0")])
    p = verbs.smpl_pool(link)
    assert p["busy"] == 3 and p["busy_max"] == 20 and p["used_max"] == 16 and p["bkg"] == "port"


# ── settings backup (SETTINGS_EXPORT / IMPORT / STATUS) ───────────────────────

def test_settings_export_default_and_given_path(make_link) -> None:
    link = make_link([("SETTINGS_EXPORT", "OK /sdcard/crosspad-settings.json"),
                      ("SETTINGS_EXPORT /spiflash/s.json", "OK /spiflash/s.json")])
    assert verbs.settings_export(link) == {"started": True,
                                           "path": "/sdcard/crosspad-settings.json"}
    assert verbs.settings_export(link, "/spiflash/s.json")["path"] == "/spiflash/s.json"


def test_settings_export_refusal_raises(make_link) -> None:
    link = make_link([("SETTINGS_EXPORT /tmp/x", "ERR path")])
    with pytest.raises(HilError) as ei:
        verbs.settings_export(link, "/tmp/x")
    assert ei.value.code == "BAD_ARGS" and "path" in ei.value.message


def test_settings_import_starts(make_link) -> None:
    link = make_link([("SETTINGS_IMPORT /sdcard/crosspad-settings.json",
                       "OK /sdcard/crosspad-settings.json")])
    assert verbs.settings_import(link, "/sdcard/crosspad-settings.json")["started"] is True


def test_settings_status_parses_done_and_running(make_link) -> None:
    link = make_link(sequences={"SETTINGS_STATUS": [
        "SETTINGS: busy=1 op=i ok=-1 err=- path=/sdcard/crosspad-settings.json",
        "SETTINGS: busy=0 op=i ok=0 err=bad json at 12 path=/sdcard/crosspad-settings.json",
        "SETTINGS: busy=0 op=- ok=-1 err=- path=-",
    ]})
    assert verbs.settings_status(link) == {"busy": True, "op": "import", "ok": None,
                                           "err": None,
                                           "path": "/sdcard/crosspad-settings.json"}
    assert verbs.settings_status(link)["err"] == "bad json at 12"
    assert verbs.settings_status(link) == {"busy": False, "op": None, "ok": None,
                                           "err": None, "path": None}


# ── UX 2000: ASONG ARM/SEQTRACK/KIT2, ASONG_SEQ, KIT2 ───────────────────

_ASONG_OK = ("ASONG: ok=1 open=2 edit=0 eprog=0 rgn=0-0 bpm=120 len=8 sync=0 play=0 pos=0.00 "
             "under=0 openfail=0 bounce=0 done=0 total=0 clips=0/0/0/0 file=- project=Demo")


def test_asong_arm_sends_the_track(make_link) -> None:
    link = make_link([("ASONG ARM 2", _ASONG_OK)])
    assert verbs.asong_arm(link, 2)["ok"] == 1


def test_asong_arm_rejects_a_track_out_of_range(make_link) -> None:
    with pytest.raises(HilError) as ei:
        verbs.asong_arm(make_link(), 4)
    assert ei.value.code == "BAD_ARGS"


def test_asong_seqtrack_and_kit2(make_link) -> None:
    link = make_link([("ASONG SEQTRACK", _ASONG_OK),
                      ("ASONG KIT2 /sdcard/kits/Drums/kit.json", _ASONG_OK)])
    assert verbs.asong_seqtrack(link)["ok"] == 1
    assert verbs.asong_kit2(link, "/sdcard/kits/Drums/kit.json")["ok"] == 1


def test_asong_seqtrack_refused_when_no_track_is_free(make_link) -> None:
    link = make_link([("ASONG SEQTRACK", _ASONG_OK.replace("ok=1", "ok=0"))])
    with pytest.raises(HilError) as ei:
        verbs.asong_seqtrack(link)
    assert ei.value.code == "BAD_ARGS"


def test_asong_kit2_refuses_an_empty_or_multiline_path(make_link) -> None:
    for bad in ("", "a\nb"):
        with pytest.raises(HilError) as ei:
            verbs.asong_kit2(make_link(), bad)
        assert ei.value.code == "BAD_ARGS"


def test_asong_seq_rows_and_trailer(make_link) -> None:
    link = make_link([("ASONG_SEQ 1", [
        "ASEQ: bar=0 bars=2 from=0 notes=12 own=1",
        "ASEQ: bar=4 bars=1 from=0 notes=5 own=0",
        "ASEQ: track=1 take=0 punchable=1 edit=4 route=1 armed=1",
    ])])
    r = verbs.asong_seq(link, 1)
    assert r["blocks"] == [{"bar": 0, "bars": 2, "from": 0, "notes": 12, "own": 1},
                           {"bar": 4, "bars": 1, "from": 0, "notes": 5, "own": 0}]
    assert r["track"] == 1 and r["route"] == 1 and r["armed"] == 1 and r["punchable"] == 1


def test_asong_seq_without_a_seq_track(make_link) -> None:
    link = make_link([("ASONG_SEQ", "ASEQ: track=-1 take=0 punchable=0 edit=0 route=-1 armed=-1")])
    r = verbs.asong_seq(link)
    assert r["blocks"] == [] and r["track"] == -1


def test_asong_seq_no_engine_is_not_supported(make_link) -> None:
    link = make_link([("ASONG_SEQ", "ASEQ: no engine")])
    with pytest.raises(HilError) as ei:
        verbs.asong_seq(link)
    assert ei.value.code == "NOT_SUPPORTED"


def test_kit2_status_and_notes(make_link) -> None:
    link = make_link([
        ("KIT2", "KIT2: state=2 active=1 want=/sdcard/k/kit.json have=/sdcard/k/kit.json why=- "
                 "slots=16 complete=16 head_kb=64 heads_kb=1024 pool_kb=512 voices=1 stream=0 "
                 "lim=8 notes=3 drop=0 steal=0 norings=0 underrun=0 pair=1 kit1_lim=8/4 engine=1"),
        ("KIT2 NOTE 3 100", "OK"),
        ("KIT2 OFF 3", "OK"),
    ])
    st = verbs.kit2(link)
    assert st["state"] == 2 and st["active"] == 1 and st["slots"] == 16 and st["engine"] == 1
    assert st["kit1_lim"] == "8/4"
    assert verbs.kit2_note(link, 3, 100) == {"ok": True}
    assert verbs.kit2_off(link, 3) == {"ok": True}


def test_kit2_without_a_second_kit_is_not_supported(make_link) -> None:
    link = make_link([("KIT2", "ERR no_second_kit"), ("KIT2 NOTE 0 100", "ERR no_second_kit")])
    with pytest.raises(HilError) as ei:
        verbs.kit2(link)
    assert ei.value.code == "NOT_SUPPORTED"
    with pytest.raises(HilError) as ei:
        verbs.kit2_note(link, 0)
    assert "no_second_kit" in ei.value.message


def test_kit2_note_rejects_a_pad_out_of_range(make_link) -> None:
    with pytest.raises(HilError) as ei:
        verbs.kit2_note(make_link(), 16)
    assert ei.value.code == "BAD_ARGS"
