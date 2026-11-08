"""Console lines and readbacks, old and new formats."""
from __future__ import annotations

from crosspad_hil.jam import telemetry as T


def test_spk_lines_old_new_and_sentinel() -> None:
    old = "I (1) usb_audio_bridge: SPK #3: len=176 taken=44 ring=44 under=0 refused=0 cap_ring=3 cap_drop=0"
    new = old + " ring_min=4294967295 ev_at=12345 hole_ms=2"
    assert T.parse_spk(old)["under"] == 0 and T.parse_spk(old)["n"] == 3
    d = T.parse_spk(new)
    assert d["ring_min"] is None and d["ev_at"] == 12345 and d["hole_ms"] == 2


def test_engine_lines_without_a_log_prefix() -> None:
    line = ("\x1b[31mengine underrun_at=123456ms bg_pass=900us read=2100us yld=1 fb=0 stage=r "
            "p1=10us/1/0/2000us p2=-\x1b[0m")
    kind, d = T.classify(line)
    assert kind == "underrun" and d["underrun_at"] == 123456 and d["stage"] == "r"
    kind, d = T.classify("\x1b[31mengine pool: esc=1 fixed=2 leak=0 unheld=0 shared=0 late=0 "
                         "busy=3 voices=4\x1b[0m")
    assert kind == "pool" and d["fixed"] == 2


def test_the_board_s_own_events() -> None:
    assert T.classify("I (5) hil_control: KIT_LOAD 21 started") == \
        ("kit_load", {"kit": 21, "state": "started"})
    assert T.classify("I (5) audio_route: BOUNCE 2 0 refused")[1]["state"] == "refused"
    k, d = T.classify("I (9) song: bounced scene 3 to HIL_S3_01.wav (176400 frames)")
    assert k == "bounced" and d["frames"] == 176400
    k, d = T.classify("I (9) usbcfg: USB mode switched to: audio (rt_gap_max_us=1500)")
    assert k == "mode" and d["rt_gap_max_us"] == 1500
    assert T.classify("E (1) task_wdt: Task watchdog got triggered")[0] == "fatal"


def test_readbacks() -> None:
    s = T.parse_seq("SEQ: mode=1 state=4 scenes=3 scene=1>2 bars=2/4/1/1/1/1/1/1 ev=20/16/0/0/0/0/0/0 "
                    "bpm=120.0 quant=2 pos=0.250 dirty=0 erase=0 countin=1 file=/sdcard/x y.json")
    assert s["bars"][:3] == [2, 4, 1] and s["scene"] == "1>2" and s["file"] == "/sdcard/x y.json"
    assert T.parse_song("SONG: mode=1 loop=1 len=2 pos=0 pass=1 scene=0 blocks=0:2,1:1")["blocks"] \
        == [[0, 2], [1, 1]]
    c = T.parse_click("CLICK: running=1 bpm=120.0 beats=4 bars=2 bar=1 beat=3 pos=40000/176400")
    assert c["pos"] == 40000 and c["loop"] == 176400
    m = T.parse_mixlvl(["MIXLVL: ch=3 name=USB pk=0.05/0.05 vol=1.00 mute=0 route=111",
                        "MIXLVL: out=2 pk=0.1/0.1 vol=1.00 mute=0"])
    assert m["channels"][0]["route"] == [1, 1, 1] and m["outputs"][0]["out"] == 2


def test_the_index_answers_around_a_moment_and_the_covering_spk_lines() -> None:
    raw = []
    for k in range(5):
        raw.append((k + 1, 100.0 + k, 0.0, f"I ({1000 + 1000 * k}) usb_audio_bridge: SPK #{k}: "
                                            f"len=176 under={0 if k < 3 else 2} refused=0"))
    tel = T.Telemetry.from_rows(raw)
    a, b = tel.spk_covering(102.5)
    assert a.fields["n"] == 2 and b.fields["n"] == 3
    assert len(tel.around(102.0, 0.6, 0.6)) == 1
