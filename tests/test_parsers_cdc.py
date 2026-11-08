"""parse_cdc_reply: one dict per reply prefix, formats copied from main/hil_control.cpp."""
from __future__ import annotations

import pytest

from crosspad_hil.parsers import parse_cdc_reply

BLE_LINE = (
    "BLE: supported=1 running=1 state=connected mode=server "
    "self=7c:df:a1:00:11:22 peer=a4:c1:38:aa:bb:cc itvl=15 txoff=36 rxoff=0 "
    "tx_msg=181 tx_pkt=120 tx_drop=0 tx_err=0 rx_msg=4 rx_pkt=4"
)

LEDS_LINE = (
    "LEDS: bri=80 anim=1 coalesce=0 cfgbri=50 pwr=0x1F pwrN=3 txfail=0 colors="
    "FF0000,00FF00,0000FF,000000,000000,000000,000000,000000,"
    "000000,000000,000000,000000,000000,000000,000000,FFFFFF"
)

MEMBLK_LINE = (
    "MEMBLK: biggest_used=32768 | <=64 u=812/31240B f=40/1800B | <=128 u=210/19800B f=12/1100B"
    " | <=256 u=90/16000B f=3/600B | <=512 u=40/14000B f=2/900B | <=1k u=20/15000B f=1/800B"
    " | <=2k u=10/16000B f=0/0B | <=8k u=6/30000B f=1/4096B | <=big u=3/70000B f=1/18000B"
)

PADINFO_LINE = (
    "PADINFO: idx=3 num=39 kit=DRUMS vol=100 pan=0 mode=1 dirty=0 count=2"
    " | [0] 'kick.wav' 0..127 | [1] 'kick2.wav' 64..127"
)

CASES: list[tuple[str, dict]] = [
    # "KITSTATUS: current=%d loading=%d pending=%d name=%s"
    (
        "KITSTATUS: current=3 loading=1 pending=7 name=DRUMS",
        {"kind": "kitstatus", "current": 3, "loading": True, "pending": 7, "name": "DRUMS"},
    ),
    (
        "KITSTATUS: current=-1 loading=0 pending=-1 name=-",
        {"kind": "kitstatus", "current": -1, "loading": False, "pending": -1, "name": None},
    ),
    # "APPS: " + "%s%s" joined by "," + " running=%s"
    (
        "APPS: Sampler,Sequencer,Settings running=Sampler",
        {"kind": "apps", "apps": ["Sampler", "Sequencer", "Settings"], "running": "Sampler"},
    ),
    ("APPS:  running=-", {"kind": "apps", "apps": [], "running": None}),
    # newer firmware appends fields after running=
    (
        "APPS: Sampler,Settings running=Sampler selector=3",
        {"kind": "apps", "apps": ["Sampler", "Settings"], "running": "Sampler",
         "extra": {"selector": 3}},
    ),
    # "KITS: <id>:<name>,…" per batch, then "KITS: end count=%d current=%d"
    (
        "KITS: 0:Basic,1:Drums,2:Deep House",
        {
            "kind": "kits",
            "kits": [
                {"id": 0, "name": "Basic"},
                {"id": 1, "name": "Drums"},
                {"id": 2, "name": "Deep House"},
            ],
        },
    ),
    ("KITS: end count=3 current=2",
     {"kind": "kits", "end": True, "count": 3, "current": 2}),
    ("KITS: end count=0 current=-1",
     {"kind": "kits", "end": True, "count": 0, "current": -1}),
    # "PADSTATS: press=%u release=%u played=%u freeslots=%d"
    (
        "PADSTATS: press=181 release=181 played=181 freeslots=12",
        {"kind": "padstats", "press": 181, "release": 181, "played": 181, "freeslots": 12},
    ),
    # "PADNOTES:" + " %u:%u" x16
    (
        "PADNOTES: 0:36 1:37 2:38 3:39 4:40 5:41 6:42 7:43 8:44 9:45 10:46 11:47 12:48 13:49"
        " 14:50 15:51",
        {"kind": "padnotes", "notes": {i: 36 + i for i in range(16)}},
    ),
    # "LEDS: bri=%u anim=%d coalesce=%d cfgbri=%u pwr=0x%02X pwrN=%u txfail=%u colors="
    #   + "%02X%02X%02X" per pad, comma-separated
    (
        LEDS_LINE,
        {
            "kind": "leds",
            "brightness": 80,
            "anim": True,
            "coalesce": False,
            "cfgbri": 50,
            "pwr": 0x1F,
            "pwr_count": 3,
            "txfail": 0,
            "colors": ["FF0000", "00FF00", "0000FF"] + ["000000"] * 12 + ["FFFFFF"],
        },
    ),
    # "MEM: int_free=%u int_largest=%u int_min=%u int_blocks=%u psram_free=%u psram_largest=%u
    #  psram_blocks=%u"
    (
        "MEM: int_free=76544 int_largest=65536 int_min=18700 int_blocks=1201 "
        "psram_free=8123456 psram_largest=4194304 psram_blocks=300",
        {
            "kind": "mem",
            "int_free": 76544,
            "int_largest": 65536,
            "int_min": 18700,
            "int_blocks": 1201,
            "psram_free": 8123456,
            "psram_largest": 4194304,
            "psram_blocks": 300,
        },
    ),
    # "MEMBLK: biggest_used=%u" + " | <=%s u=%u/%uB f=%u/%uB" x8
    (
        MEMBLK_LINE,
        {
            "kind": "memblk",
            "biggest_used": 32768,
            "buckets": [
                {"le": "64", "used_n": 812, "used_b": 31240, "free_n": 40, "free_b": 1800},
                {"le": "128", "used_n": 210, "used_b": 19800, "free_n": 12, "free_b": 1100},
                {"le": "256", "used_n": 90, "used_b": 16000, "free_n": 3, "free_b": 600},
                {"le": "512", "used_n": 40, "used_b": 14000, "free_n": 2, "free_b": 900},
                {"le": "1k", "used_n": 20, "used_b": 15000, "free_n": 1, "free_b": 800},
                {"le": "2k", "used_n": 10, "used_b": 16000, "free_n": 0, "free_b": 0},
                {"le": "8k", "used_n": 6, "used_b": 30000, "free_n": 1, "free_b": 4096},
                {"le": "big", "used_n": 3, "used_b": 70000, "free_n": 1, "free_b": 18000},
            ],
        },
    ),
    # "MEMBIG:" + " @%08x=%u" per block >= 8 kB
    (
        "MEMBIG: @3fc9a000=32768 @3fcb2000=8196",
        {
            "kind": "membig",
            "blocks": [{"addr": 0x3FC9A000, "size": 32768}, {"addr": 0x3FCB2000, "size": 8196}],
        },
    ),
    ("MEMBIG:", {"kind": "membig", "blocks": []}),
    # "CDCSTATS: rx=%u cmds=%u drop=%u"
    (
        "CDCSTATS: rx=1200 cmds=1180 drop=20",
        {"kind": "cdcstats", "rx": 1200, "cmds": 1180, "drop": 20},
    ),
    # "BLE: supported=%d running=%d state=%s mode=%s self=%s peer=%s itvl=%u txoff=%d rxoff=%d
    #  tx_msg=%u tx_pkt=%u tx_drop=%u tx_err=%u rx_msg=%u rx_pkt=%u"
    (
        BLE_LINE,
        {
            "kind": "ble",
            "supported": 1,
            "running": 1,
            "state": "connected",
            "mode": "server",
            "self": "7c:df:a1:00:11:22",
            "peer": "a4:c1:38:aa:bb:cc",
            "itvl": 15,
            "txoff": 36,
            "rxoff": 0,
            "tx_msg": 181,
            "tx_pkt": 120,
            "tx_drop": 0,
            "tx_err": 0,
            "rx_msg": 4,
            "rx_pkt": 4,
        },
    ),
    (
        "BLE: supported=0 running=0 state=off mode=server self=- peer=- itvl=0 txoff=0 rxoff=0 "
        "tx_msg=0 tx_pkt=0 tx_drop=0 tx_err=0 rx_msg=0 rx_pkt=0",
        {
            "kind": "ble",
            "supported": 0,
            "running": 0,
            "state": "off",
            "mode": "server",
            "self": None,
            "peer": None,
            "itvl": 0,
            "txoff": 0,
            "rxoff": 0,
            "tx_msg": 0,
            "tx_pkt": 0,
            "tx_drop": 0,
            "tx_err": 0,
            "rx_msg": 0,
            "rx_pkt": 0,
        },
    ),
    # "BLEDEV: count=%u" + " | %s %s %d" (addr name rssi) — names may contain spaces
    (
        "BLEDEV: count=2 | a4:c1:38:aa:bb:cc Studio iPad Pro -61 | 11:22:33:44:55:66 CrossPad -80",
        {
            "kind": "bledev",
            "count": 2,
            "devices": [
                {"addr": "a4:c1:38:aa:bb:cc", "name": "Studio iPad Pro", "rssi": -61},
                {"addr": "11:22:33:44:55:66", "name": "CrossPad", "rssi": -80},
            ],
        },
    ),
    ("BLEDEV: count=0", {"kind": "bledev", "count": 0, "devices": []}),
    # "DAW: running=%d connected=%d daw=%u mode=%u transport=%u beat=%u beats=%lu
    #      row0="%s" row1="%s" row2="%s"" — rows are quoted, they carry spaces
    (
        'DAW: running=1 connected=1 daw=1 mode=0 transport=1 beat=0 beats=14 '
        'row0="808 Kick" row1="Pattern 1" row2="130.0 BPM"',
        {
            "kind": "daw",
            "running": True,
            "connected": True,
            "daw": 1,
            "mode": 0,
            "transport": 1,
            "beat": 0,
            "beats": 14,
            "rows": ["808 Kick", "Pattern 1", "130.0 BPM"],
        },
    ),
    (
        'DAW: running=0 connected=0 daw=0 mode=0 transport=0 beat=0 beats=0 row0="" row1="" row2=""',
        {
            "kind": "daw",
            "running": False,
            "connected": False,
            "daw": 0,
            "mode": 0,
            "transport": 0,
            "beat": 0,
            "beats": 0,
            "rows": ["", "", ""],
        },
    ),
    # "UI: display=%s touch=%s drawer=%d lcd=%d rgb=%d theme=%u bt_icon=%s app=%s"
    (
        "UI: display=yes touch=no drawer=1 lcd=80 rgb=50 theme=2 bt_icon=hidden app=-",
        {
            "kind": "ui",
            "display": "yes",
            "touch": "no",
            "drawer": 1,
            "lcd": 80,
            "rgb": 50,
            "theme": 2,
            "bt_icon": "hidden",
            "app": None,
        },
    ),
    (
        "UI: display=none touch=no drawer=0 lcd=-1 rgb=-1 theme=0 bt_icon=solid app=Sampler",
        {
            "kind": "ui",
            "display": "none",
            "touch": "no",
            "drawer": 0,
            "lcd": -1,
            "rgb": -1,
            "theme": 0,
            "bt_icon": "solid",
            "app": "Sampler",
        },
    ),
    # "ENC: group=%p launcher=%p owner=%s"
    (
        "ENC: group=0x3fca1234 launcher=0x3fca1234 owner=launcher",
        {"kind": "enc", "group": "0x3fca1234", "launcher": "0x3fca1234", "owner": "launcher"},
    ),
    (
        "ENC: group=0 launcher=0x3fca1234 owner=none",
        {"kind": "enc", "group": 0, "launcher": "0x3fca1234", "owner": "none"},
    ),
    # "ENCFOCUS: obj=%p idx=%d text=%s" — text may contain spaces or be empty
    (
        "ENCFOCUS: obj=0x3fcb0010 idx=4 text=Send transpose",
        {"kind": "encfocus", "index": 4, "label": "Send transpose", "ptr": "0x3fcb0010"},
    ),
    ("ENCFOCUS: obj=0 idx=-1 text=", {"kind": "encfocus", "index": -1, "label": "", "ptr": "0"}),
    # edit= since 2026-10-06: whether a turn changes the focused value
    (
        "ENCFOCUS: obj=0x3fcb0020 idx=12 edit=1 text=CHOKE 2",
        {"kind": "encfocus", "index": 12, "label": "CHOKE 2", "ptr": "0x3fcb0020", "editing": True},
    ),
    (
        "ENCFOCUS: obj=0x3fcb0020 idx=12 edit=0 text=CHOKE -",
        {"kind": "encfocus", "index": 12, "label": "CHOKE -", "ptr": "0x3fcb0020", "editing": False},
    ),
    # "ENCGROUP: count=%u" (head line only; the body is parse_enc_group's job)
    ("ENCGROUP: count=5", {"kind": "encgroup", "count": 5}),
    # "AUDIOLVL: L=%.5f R=%.5f amp=%d allowed=%d"
    (
        "AUDIOLVL: L=0.12345 R=0.00000 amp=1 allowed=1",
        {"kind": "audiolvl", "left": 0.12345, "right": 0.0, "amp": True, "allowed": True},
    ),
    # "SMPLPEAK: %d free=%d"
    ("SMPLPEAK: 18234 free=12", {"kind": "smplpeak", "peak": 18234, "free": 12}),
    ("SMPLPEAK: 0 free=-1", {"kind": "smplpeak", "peak": 0, "free": -1}),
    # the native engine's merge (2026-10): load_*/maxc_* gone; poly, rxskip, unpaced in
    (
        "SMPLPEAK: 5692 free=48 poly=16 rt_max_us=1122 rxskip=0 noblock=0 underrun=2 "
        "unpaced=0 rt_busy_max_us=1301 overruns=0",
        {"kind": "smplpeak", "peak": 5692, "free": 48, "poly": 16, "rt_max_us": 1122,
         "rxskip": 0, "noblock": 0, "underrun": 2, "unpaced": 0, "rt_busy_max_us": 1301,
         "overruns": 0},
    ),
    # "SMPLPITCH: pad=%d semi=%d engine=%s"
    ("SMPLPITCH: pad=0 semi=12 engine=native",
     {"kind": "smplpitch", "pad": 0, "semi": 12, "engine": "native"}),
    ("SMPLPITCH: pad=7 semi=-5 engine=none",
     {"kind": "smplpitch", "pad": 7, "semi": -5, "engine": "none"}),
    # "SMPLPOLY: max_voices=%u max_streaming=%u"
    ("SMPLPOLY: max_voices=16 max_streaming=8",
     {"kind": "smplpoly", "max_voices": 16, "max_streaming": 8}),
    # "PITCHED: active=%d zones=%u roots=%s voices=%u/%u bank_kb=%u rt_max_us=%u
    #           steals=%u [err=why]"
    (
        "PITCHED: active=1 zones=2 roots=57.00,69.03 voices=3/24 bank_kb=180 "
        "rt_max_us=812 steals=0",
        {"kind": "pitched", "active": True, "zones": 2, "roots": [57.0, 69.03],
         "voices_active": 3, "voices_max": 24, "bank_kb": 180, "rt_max_us": 812,
         "steals": 0, "load_err": None},
    ),
    (
        "PITCHED: active=0 zones=0 roots= voices=0/24 bank_kb=0 rt_max_us=0 steals=0 "
        "err=zone_0:_no_PSRAM_(88_kB)",
        {"kind": "pitched", "active": False, "zones": 0, "roots": [],
         "voices_active": 0, "voices_max": 24, "bank_kb": 0, "rt_max_us": 0,
         "steals": 0, "load_err": "zone_0:_no_PSRAM_(88_kB)"},
    ),
    ("PITCHED: nonsense", None),
    (
        # the rest of the line matches; only roots is garbled — None, not a raised
        # ValueError (parse_cdc_reply's contract for anything it cannot make sense of)
        "PITCHED: active=1 zones=1 roots=not_a_number voices=1/24 bank_kb=10 "
        "rt_max_us=100 steals=0",
        None,
    ),
    # "APPVER: %s id=%s commit=%s ref=%s dirty=%d" / "APPVER: end count=%u"
    (
        "APPVER: crosspad-sampler id=sampler commit=a1b2c3d ref=v1.4.0 dirty=1",
        {
            "kind": "appver",
            "component": "crosspad-sampler",
            "id": "sampler",
            "commit": "a1b2c3d",
            "ref": "v1.4.0",
            "dirty": True,
        },
    ),
    (
        "APPVER: crosspad-core id=- commit=9f8e7d6 ref=- dirty=0",
        {
            "kind": "appver",
            "component": "crosspad-core",
            "id": None,
            "commit": "9f8e7d6",
            "ref": None,
            "dirty": False,
        },
    ),
    ("APPVER: end count=4", {"kind": "appver", "end": True, "count": 4}),
    # "PADINFO: idx=%d num=%u kit=%s vol=%d pan=%d mode=%u dirty=%d count=%u"
    #   + " | [%u] '%s' %u..%u" per sample slot
    (
        PADINFO_LINE,
        {
            "kind": "padinfo",
            "idx": 3,
            "num": 39,
            "kit": "DRUMS",
            "vol": 100,
            "pan": 0,
            "mode": 1,
            "dirty": 0,
            "count": 2,
            "slots": [
                {"index": 0, "file": "kick.wav", "lo": 0, "hi": 127},
                {"index": 1, "file": "kick2.wav", "lo": 64, "hi": 127},
            ],
            "raw": PADINFO_LINE,
        },
    ),
    # bare acknowledgements
    ("OK", {"kind": "ok"}),
    ("OK\r\n", {"kind": "ok"}),
    ("ERR", {"kind": "err", "message": ""}),
    ("ERR bad kit id", {"kind": "err", "message": "bad kit id"}),
    ("ERR unknown app\r\n", {"kind": "err", "message": "unknown app"}),
    # prefixed error bodies
    ("ENCGROUP: ERR lock", {"kind": "err", "message": "ENCGROUP: ERR lock"}),
    ("ENCFOCUS: ERR lock", {"kind": "err", "message": "ENCFOCUS: ERR lock"}),
    # family errors from the bulk verbs (main/file_transfer.cpp)
    ("FS_ERR no_worker int_largest=3700 need=5120",
     {"kind": "err", "family": "FS", "message": "no_worker int_largest=3700 need=5120"}),
    ("ASSETS_ERROR not_littlefs", {"kind": "err", "family": "ASSETS", "message": "not_littlefs"}),
    ("FS_ERR", {"kind": "err", "family": "FS", "message": ""}),
    # STM_FW_INFO — no colon after the verb
    (
        "STM_FW running=1.4/0x000F board=r20 fw_pcb=r20 bundled=1.4/0x000F pcb=r20 size=112276",
        {"kind": "stm_fw", "linked": True, "running": "1.4", "running_proto": 0x000F,
         "board": "r20", "fw_pcb": "r20", "bundled": "1.4", "bundled_proto": 0x000F,
         "pcb": "r20", "size": 112276},
    ),
    (
        "STM_FW running=1.3/0x000E(down) bundled=none(ESP_ERR_NOT_FOUND)",
        {"kind": "stm_fw", "linked": False, "running": "1.3", "running_proto": 0x000E,
         "bundled": None, "bundled_error": "ESP_ERR_NOT_FOUND"},
    ),
    # "FWUPD: state=%s running=%s kit=%s latest=%s channel=%s progress=%u last_check=%lld err=%s stack_free=%u inst_stack_free=%u"
    (
        "FWUPD: state=downloading running=v1.1.0-rc1-64-gc6d227d kit=full latest=1.3.0 "
        "channel=stable progress=42 last_check=1790964000 err=- stack_free=2100 inst_stack_free=2900",
        {"kind": "fwupd", "state": "downloading", "running": "v1.1.0-rc1-64-gc6d227d", "kit": "full",
         "latest": "1.3.0", "channel": "stable", "progress": 42, "last_check": 1790964000, "err": None,
         "stack_free": 2100, "inst_stack_free": 2900},
    ),
    # "FWCAT: count=%u total=%u" + " | <kit> <version> <channel> <tier> <on_card>"
    (
        "FWCAT: count=2 total=3 | full 1.3.0 stable official 1 | lite 1.4.0-rc1 beta community 0",
        {"kind": "fwcat", "count": 2, "total": 3, "versions": [
            {"kit": "full", "version": "1.3.0", "channel": "stable", "tier": "official", "on_card": True},
            {"kit": "lite", "version": "1.4.0-rc1", "channel": "beta", "tier": "community", "on_card": False},
        ]},
    ),
    ("FWCAT: count=0 total=0", {"kind": "fwcat", "count": 0, "total": 0, "versions": []}),
    # "FWUI: tab=%s sel=%s card=%s items=%u selkey=%s host=%s store=%s busy=%u"
    (
        "FWUI: tab=library sel=3 card=- items=31 selkey=zzui03@1.0.0 host=app store=- busy=0",
        {"kind": "fwui", "tab": "library", "sel": 3, "card": None, "items": 31, "selkey": "zzui03@1.0.0",
         "host": "app", "store": None, "busy": 0},
    ),
    (
        "FWUI: tab=store sel=- card=full@1.1.0-fwstore.2 items=2 selkey=- host=net store=ok busy=1",
        {"kind": "fwui", "tab": "store", "sel": None, "card": "full@1.1.0-fwstore.2", "items": 2, "selkey": None,
         "host": "net", "store": "ok", "busy": 1},
    ),
]


@pytest.mark.parametrize(("line", "expected"), CASES, ids=[c[0][:24] for c in CASES])
def test_parse_cdc_reply(line: str, expected: dict) -> None:
    assert parse_cdc_reply(line) == expected


@pytest.mark.parametrize(
    "line",
    [
        "",
        "   ",
        "KIT_LOAD 3",  # an echoed command, not a reply
        "I (1234) main: some log line",
        "lowercase: nope",
        "TOTALLYNEW: x=1",
    ],
)
def test_parse_cdc_reply_unknown_returns_none(line: str) -> None:
    assert parse_cdc_reply(line) is None


def test_trailing_crlf_is_tolerated_on_prefixed_replies() -> None:
    assert parse_cdc_reply("LVGLSTATS: refr=12 refr_max_us=150000 refr_avg_us=40000 "
                           "render_us=300000 render_max_us=90000 flush_wait_us=180000 "
                           "flush_wait_max_us=60000 flushes=96") == {
        "kind": "lvglstats", "refr": 12, "refr_max_us": 150000, "refr_avg_us": 40000,
        "render_us": 300000, "render_max_us": 90000, "flush_wait_us": 180000,
        "flush_wait_max_us": 60000, "flushes": 96}
    assert parse_cdc_reply("CDCSTATS: rx=1 cmds=1 drop=0\r\n") == {
        "kind": "cdcstats",
        "rx": 1,
        "cmds": 1,
        "drop": 0,
    }


def test_leds_short_color_list_is_not_padded() -> None:
    # A truncated reply (buffer cap) yields whatever colours arrived; callers check len().
    out = parse_cdc_reply(
        "LEDS: bri=1 anim=0 coalesce=0 cfgbri=1 pwr=0x00 pwrN=0 txfail=0 colors=FF0000,00FF00"
    )
    assert out is not None
    assert out["colors"] == ["FF0000", "00FF00"]
    assert out["pwr"] == 0


def test_kitstatus_name_with_spaces() -> None:
    out = parse_cdc_reply("KITSTATUS: current=2 loading=0 pending=-1 name=Deep House")
    assert out == {
        "kind": "kitstatus",
        "current": 2,
        "loading": False,
        "pending": -1,
        "name": "Deep House",
    }


def test_prefix_with_underscore_parses():
    # _PREFIX_RE took [A-Z]+ only, so every NAME_WITH_UNDERSCORE: reply was unknown --
    # AUDIO_RATE: among them, not just the WiFi rows.
    assert parse_cdc_reply("AUDIO_RATE: running=44100 configured=44100")["kind"] == "audiorate"


def test_wifi_status_quoted_ssid_with_spaces():
    p = parse_cdc_reply('WIFI: state=up ssid="Dom Kowalskich" rssi=-54 ip=192.168.1.23 leases=1 '
                        'fail=none reason=0 heap_int=61234 heap_init=90112 heap_low=34816 stack_free=1800')
    assert p["kind"] == "wifi"
    assert p["ssid"] == "Dom Kowalskich"
    assert p["state"] == "up" and p["rssi"] == -54 and p["leases"] == 1
    assert p["reason"] == 0 and p["heap_init"] == 90112 and p["heap_low"] == 34816


def test_wifi_status_empty_ssid_is_none():
    p = parse_cdc_reply('WIFI: state=failed ssid="" rssi=0 ip=0.0.0.0 leases=0 fail=auth reason=202 '
                        'heap_int=99279 heap_init=0 heap_low=0 stack_free=0')
    assert p["ssid"] is None and p["fail"] == "auth" and p["reason"] == 202


def test_wifi_ap_rows_keep_spaces_and_utf8():
    p = parse_cdc_reply("WIFI_AP: count=2 | -48 wpa2 Dom Kowalskich | -70 open Żółta sieć")
    assert p["kind"] == "wifi_ap" and p["count"] == 2
    assert p["aps"][0] == {"rssi": -48, "auth": "wpa2", "ssid": "Dom Kowalskich"}
    assert p["aps"][1] == {"rssi": -70, "auth": "open", "ssid": "Żółta sieć"}


def test_wifi_net_list():
    p = parse_cdc_reply("WIFI_NET: count=2 | Dom Kowalskich | Bench")
    assert p == {"kind": "wifi_net", "count": 2, "nets": ["Dom Kowalskich", "Bench"]}


def test_wifi_ping():
    p = parse_cdc_reply("WIFI_PING: ok ms=231 status=404")
    assert p["kind"] == "wifi_ping" and p["ok"] is True and p["ms"] == 231 and p["status"] == 404


def test_net_mode():
    assert parse_cdc_reply("NET_MODE: active=1 tab=wifi") == {"kind": "net_mode", "active": 1, "tab": "wifi"}
    assert parse_cdc_reply("NET_MODE: active=0 tab=-")["tab"] is None


def test_wifi_status_escaped_ssid_is_unescaped():
    # wifi_verbs.cpp escape_ssid(): backslash, quote and pipe get a backslash, CR/LF become \r \n
    p = parse_cdc_reply(r'WIFI: state=up ssid="a \"q\" b\\c \| d\r\n\x01" rssi=-50 ip=1.2.3.4 leases=0 '
                        r'fail=none reason=0 heap_int=1 heap_init=0 heap_low=0 stack_free=0')
    assert p["ssid"] == 'a "q" b\\c | d\r\n\x01'
    assert p["rssi"] == -50 and p["state"] == "up"


def test_wifi_status_ssid_that_looks_like_the_next_field():
    p = parse_cdc_reply(r'WIFI: state=up ssid="x\" rssi=1 ip=" rssi=-50 ip=1.2.3.4 leases=0 fail=none reason=0')
    assert p["ssid"] == 'x" rssi=1 ip='
    assert p["rssi"] == -50


def test_wifi_ap_row_with_separator_inside_ssid():
    p = parse_cdc_reply(r"WIFI_AP: count=2 | -48 wpa2 a \| b | -70 open c\\ | d")
    assert p["aps"][0] == {"rssi": -48, "auth": "wpa2", "ssid": "a | b"}
    assert p["aps"][1] == {"rssi": -70, "auth": "open", "ssid": "c\\"}


def test_wifi_ap_row_with_line_break_in_ssid():
    p = parse_cdc_reply(r"WIFI_AP: count=1 | -48 wpa2 x\r\ny")
    assert p["aps"][0]["ssid"] == "x\r\ny"


def test_wifi_net_escaped_names():
    p = parse_cdc_reply(r'WIFI_NET: count=2 | say \"hi\" | back\\slash \| pipe')
    assert p["nets"] == ['say "hi"', "back\\slash | pipe"]


def test_widened_prefix_does_not_claim_errors_or_log_lines() -> None:
    assert parse_cdc_reply("ERR foo: bar")["kind"] == "err"
    assert parse_cdc_reply("I (12) TAG: msg") is None
    assert parse_cdc_reply("AUDIO_RATE: ERR x")["kind"] == "err"


def test_fwcat_tolerates_a_malformed_head() -> None:
    out = parse_cdc_reply("FWCAT: count=x total=- | full 1.3.0 stable official 1")
    assert out["count"] == 1 and out["total"] == 1 and out["versions"][0]["kit"] == "full"


SMPLENG_LINE = ("SMPLENG: engine=native state=2 why=- pitch=12,1,-5 slots=16 complete=16 head_kb=256 "
                "heads_kb=1437 pool_kb=512 pool_blocks=3/128 peak=9 voices=2/11 lim=8/16 notes=337 "
                "drop=0 steal=0 norings=0 underrun=0 quiet=12 rt_max_us=1279 noteon_max_us=640 "
                "read_kb=46 engines=native")


def test_smpleng_splits_its_pairs() -> None:
    p = parse_cdc_reply(SMPLENG_LINE)
    assert p["kind"] == "smpleng"
    assert p["engine"] == "native" and p["state"] == 2 and p["why"] is None
    assert p["pitch"] == [12, 1, -5]
    assert (p["pool_blocks"], p["pool_blocks_total"]) == (3, 128)
    assert (p["voices"], p["voices_peak"]) == (2, 11)
    assert (p["streaming"], p["limit"]) == (8, 16)
    assert p["underrun"] == 0 and p["norings"] == 0 and p["read_kb"] == 46
    assert p["engines"] == ["native"]


def test_smpleng_reports_both_engines_and_a_failed_take() -> None:
    p = parse_cdc_reply("SMPLENG: engine=marcel state=3 why=pool_refused pitch=0,0,0 slots=0 "
                        "pool_blocks=0/0 voices=0/0 lim=0/0 engines=native,marcel")
    assert p["engine"] == "marcel" and p["state"] == 3 and p["why"] == "pool_refused"
    assert p["engines"] == ["native", "marcel"]


def test_smpleng_from_firmware_before_engines_field() -> None:
    # An older branch printed no engines=: that image had both engines.
    p = parse_cdc_reply("SMPLENG: engine=native state=2 why=- pitch=0,1,7 slots=1")
    assert p["engines"] == ["native", "marcel"]


def test_smpleng_carries_the_held_kit_the_bg_stack_and_the_audition() -> None:
    # 2026-10-09 merge fix: kit= (the kit the engine holds, -1 with none),
    # bg_stack_free=, preview= and prev_pk=.
    p = parse_cdc_reply("SMPLENG: engine=none state=0 why=- kit=-1 pitch=0,0,0 slots=0 "
                        "pool_blocks=0/4/4 voices=0/0 lim=0/16 bg_stack_free=1812 preview=2 "
                        "prev_pk=1203 engines=native")
    assert p["kit"] == -1
    assert p["bg_stack_free"] == 1812
    assert p["preview"] == 2 and p["prev_pk"] == 1203
    p = parse_cdc_reply(SMPLENG_LINE.replace("why=- ", "why=- kit=7 "))
    assert p["kit"] == 7
