"""knowledge/*.yaml is firmware-coupled data; these tests pin its shape."""
from __future__ import annotations

import re

import pytest

from crosspad_hil import errors
from crosspad_hil.errors import HilError
from crosspad_hil.knowledge import KNOWLEDGE_NAMES, load

# From platform-idf/tools/hil_smoke.py REQUIRED_MARKERS.
SEVEN = [
    "Platform fully initialized",
    "STM32 ident:",
    "Crosspad initialization complete",
    "All systems operational",
    "LVGL setup done successfully",
    "App registry initialized",
    "LoadMainScreen completed successfully",
]


def test_names() -> None:
    assert KNOWLEDGE_NAMES == ("markers", "sysex", "cdc")


def test_unknown_name_is_env_error() -> None:
    with pytest.raises(HilError) as ei:
        load("nope")
    assert ei.value.code == errors.ENV
    assert ei.value.details["available"] == ["markers", "sysex", "cdc"]


def test_load_is_cached_but_returns_independent_copies() -> None:
    a = load("markers")
    a["required"].append("MUTATED")
    b = load("markers")
    assert "MUTATED" not in b["required"]


# -- markers.yaml -------------------------------------------------------

def test_seven_required_markers_in_order() -> None:
    assert load("markers")["required"] == SEVEN


def test_stability_required_is_the_seven_minus_stm32_ident() -> None:
    m = load("markers")
    assert m["required_stability"] == [x for x in SEVEN if x != "STM32 ident:"]
    assert len(m["required_stability"]) == 6


def test_boot_markers_and_reboot() -> None:
    m = load("markers")
    assert m["boot_markers"] == ["ESP-ROM:", "main_task: Started on CPU0"]
    assert m["reboot"] == "ESP-ROM:esp32s3"
    assert m["boot_timeout_s"] == 45


def test_optional_markers() -> None:
    assert load("markers")["optional"] == [
        "SD Card mounted successfully", "ES8388 [1] started", "DRV2605 found",
    ]


def test_fatal_patterns_compile_and_hit_known_lines() -> None:
    pats = [re.compile(p) for p in load("markers")["fatal_patterns"]]
    assert len(pats) >= 7
    samples = [
        "Guru Meditation Error: Core  1 panic'ed (LoadProhibited).",
        "abort() was called at PC 0x4037a8c1 on core 0",
        "assert failed: xQueueGenericSend queue.c:832",
        "CORRUPT HEAP: Bad head at 0x3fc9e4f0.",
        "Stack smashing protect failure!",
        "E (12345) task_wdt: Task watchdog got triggered.",
        "Interrupt wdt timeout on CPU1",
        "rst:0xc (RTC_SW_CPU_RST),boot:0x8 (SPI_FAST_FLASH_BOOT)",
    ]
    for s in samples[:-1]:
        assert any(p.search(s) for p in pats), s
    # A software reset is not fatal; a PANIC/WDT/BROWNOUT reset reason is.
    assert not any(p.search(samples[-1]) for p in pats)
    # TG0/TG1 WDT resets carry a digit and are plain reboots, not faults
    assert not any(p.search("rst:0x7 (TG0WDT_SYS_RST),boot:0x8") for p in pats)
    assert any(p.search("rst:0x10 (RTCWDT_RTC_RST),boot:0x8") for p in pats)
    assert any(p.search("rst:0xf (BROWNOUT_RST),boot:0x8") for p in pats)


def test_error_line_and_allowlist() -> None:
    m = load("markers")
    rx = re.compile(m["error_line"])
    assert rx.match("E (1234) sdmmc: file not found")
    assert not rx.match("W (1234) sdmmc: file not found")
    assert m["error_allow"] == ["file not found"]


def test_reset_reason_two_groups() -> None:
    rx = re.compile(load("markers")["reset_reason"])
    mt = rx.search("rst:0x7 (TG0WDT_SYS_RST),boot:0x8 (SPI_FAST_FLASH_BOOT)")
    assert mt is not None
    assert mt.groups() == ("0x7", "TG0WDT_SYS_RST")


def test_heap_block() -> None:
    hb = load("markers")["heap_block"]
    assert hb["start"] == "Heap Statistics"
    assert hb["end"] == "Total tasks:"
    mt = re.compile(hb["line"]).search("I (91234) PerfMon:   Free: 187432 bytes")
    assert mt is not None and mt.group(1) == "187432"


def test_kit_request_and_cdc_drops() -> None:
    m = load("markers")
    kr = re.compile(m["kit_request"]).search("I (5) hil_control: KIT_LOAD 3 queued")
    assert kr is not None and kr.groups() == ("3", "queued")
    cd = re.compile(m["cdc_drops"]).search("W (9) main: CDC: 12 commands dropped (app_queue full)")
    assert cd is not None and cd.group(1) == "12"


# -- sysex.yaml ---------------------------------------------------------

def test_sysex_ids() -> None:
    s = load("sysex")
    assert s["manufacturer"] == 0x7D
    assert s["usb_mode"] == {"id": 0x1B, "default": 0x01, "audio": 0x02}
    assert s["audio_route"]["id"] == 0x1D
    assert s["audio_route"]["subs"] == {
        "adc_input": 0x01, "mic_src": 0x02, "dac_output": 0x03, "volume": 0x04,
        "mute": 0x05, "audio_tasks": 0x06, "pad_press": 0x07, "pad_release": 0x08,
        "echo": 0x09, "preview": 0x0A, "kit_load": 0x0B, "bounce": 0x0C, "query": 0x10,
    }
    assert s["bootloader"] == {"id": 0x19, "esp": 0x00}


def test_companion_ids_and_allowlist() -> None:
    c = load("sysex")["companion"]
    assert c["id"] == 0x1F
    assert c["subs"] == {"request": 0x01, "reply_more": 0x02, "reply_last": 0x03, "notify": 0x04}
    assert c["max_request"] == 190
    assert "USB_GUARD" in c["allowed"] and "FS_PUT" not in c["allowed"]
    assert "HOST_INFO" in c["allowed"] and "FW_LATEST" in c["allowed"]


def test_companion_allows_the_ux2000_forms_whole_only() -> None:
    allowed = load("sysex")["companion"]["allowed"]
    for entry in ("ASONG", "ASONG PLAY", "ASONG STOP", "KIT2", "STOOL"):
        assert entry in allowed, entry
    # Arguments that act, and the seq report, stay CDC-only.
    for entry in ("ASONG KIT2", "ASONG ARM", "ASONG_SEQ", "ASONG SEQTRACK"):
        assert entry not in allowed, entry


def test_ux2000_verbs_in_cdc_yaml() -> None:
    v = load("cdc")["verbs"]
    assert v["KIT2"]["reply"] == "KIT2:" and v["KIT2"]["args"] == ["action:str?"]
    assert v["STOOL"]["reply"] == "STOOL:"
    assert v["ASONG_SEQ"] == {"args": ["track:int?"], "reply": "multi", "prefix": "ASEQ:",
                              "end": None, "profile": "default"}


def test_host_denylist_contains_stm_bootloader_request() -> None:
    assert [0x19, 0x01] in load("sysex")["host_denylist"]


def test_daw_ids_and_subs() -> None:
    d = load("sysex")["daw"]
    assert d["id"] == 0x1E
    assert d["proto_ver"] == 1
    assert d["subs"] == {
        "hello": 0x01, "led": 0x02, "led_all": 0x03, "text": 0x04, "beat": 0x05,
        "transport": 0x06, "mode": 0x07, "config": 0x08, "pad": 0x09, "ping": 0x0A,
        "pan": 0x0B, "launch": 0x0C, "bye": 0x7F,
    }
    assert d["daw_ids"] == {"other": 0, "fl_studio": 1, "ableton": 2, "reaper": 3}
    assert d["caps"] == {"rgb": 0, "text": 1, "encoder": 2, "aftertouch": 3, "control": 4}


# -- cdc.yaml -----------------------------------------------------------

EXPECTED_VERBS = {
    "ASSETS_BEGIN", "ASSETS_DUMP", "FS_PUT", "FS_GET", "FS_LIST", "FS_ROOTS", "FS_STAT",
    "FS_MKDIR", "FS_DELETE",
    "APP_LIST", "APP_VERSIONS", "APP_START", "APP_STOP", "APP_DESTROY", "APP_SELF_CLOSE",
    "KIT_LIST", "KIT_STATUS", "KIT_LOAD", "KIT_DELETE", "KIT_SWAP_LOG", "WAVE_STATUS",
    "SETTINGS_EXPORT", "SETTINGS_IMPORT", "SETTINGS_STATUS",
    "HOST_INFO", "FW_LATEST",
    "CLICK", "SEQ", "SEQ_DUMP", "SONG", "ASONG",
    "STM_RESET",
    "PAD_PRESS", "PAD_RELEASE", "PAD_PRESSURE", "PAD_PAN", "PAD_STATS", "PAD_STATS_RESET",
    "PAD_NOTES", "PAD_INFO", "STM_VPAD", "MIDI_TRS", "MIDI_LOOP",
    "ENC_ROTATE", "ENC_PRESS", "ENC_FOCUS", "ENC_GROUP", "ENC_STATE", "UI_STATE",
    "LED_STATE", "MEM", "MEM_BLOCKS", "CDC_STATS", "LVGL_STATS", "LVGL_INV", "LVGL_PROF", "TASKS", "CLOCK_STATUS",
    "PWR_GESTURE",
    "RESET_LOG", "COREDUMP_INFO", "RESET_TEST", "STM_REG", "SETTINGS_RESET",
    "IMU", "IMU_MIDI", "IMU_GESTURE", "IMU_THS", "IMU_REG", "IMU_REGW", "TOUCH_TRACE", "TOUCH_DRAG", "TOUCH_TAP", "TOPBAR", "UI_TREE", "WORKER",
    "ULP", "HEAP_TRACE",
    "AUDIO_LEVEL", "INGAIN", "MIX_LVL", "AUDIO_STATUS", "AUDIO_ADC", "AUDIO_MIC_SRC", "AUDIO_OUT", "AUDIO_VOL",
    "AUDIO_MUTE", "AUDIO_TASKS", "AUDIO_RATE", "SYSEX_ECHO", "DAW_STATE",
    "SMPL_PEAK", "SMPL_LAST", "SMPL_POLY", "SMPL_PREVIEW",
    "SMPL_ENGINE", "SMPL_PITCH", "SMPL_POOL", "SMPL_BKG",
    "INST_LOAD", "INST_RELEASE", "INST_STATUS",
    "ASONG_SEQ", "KIT2", "STOOL",
    "RT_PROF", "OUTVOL", "MIX_EQ", "HEAP_CHECK", "TBAR", "CLIP", "ASONG_DBG",
    "TWIN", "TWIN_HELLO", "TWIN_STATE", "MOTION",
    "PITCHED_STATUS", "PITCHED_BENCH",
    "ARENA",
    "BLE_STATUS", "BLE_DEVICES", "BLE_START", "BLE_STOP", "BLE_SCAN", "BLE_CONNECT",
    "BLE_DISCONNECT", "BLE_TXOFF", "BLE_SEND", "BLE_ENABLE", "USB_GUARD", "OTA_PREFS",
    "NET_MODE", "WIFI_STATUS", "WIFI_ENABLE", "WIFI_SET", "WIFI_LIST", "WIFI_FORGET", "WIFI_FORGET_ALL",
    "WIFI_UP", "WIFI_DOWN", "WIFI_SCAN", "WIFI_PING", "REMOTE_STATUS",
    "OTA_BEGIN", "OTA_DELTA", "USB_AUDIO", "USB_DEFAULT", "BOOTLOADER_REQUEST", "STM_DFU",
    "STM_FW_INFO", "STM_FW_UPDATE",
    "FW_SLOTS", "FW_LIST", "FW_INSTALL", "FW_SWITCH", "FW_CONFIRM", "APP_SHA", "USB_MSC",
    "FW_UPD_STATUS", "FW_CHECK", "FW_CATALOG", "FW_DOWNLOAD", "FW_UPD_CFG", "FW_UPD_CHANNEL", "FW_UPD_URL", "FW_UI",
    "MIX_MUTE", "MIX_VOL", "MIX_OUT_MUTE", "MIX_OUT_VOL",
    "UAC_STATS", "SD_TRACE", "SD_BOUNCE",
    "PAD_MIX", "KIT_SAVE", "KIT_SAVE_STATUS",
}


def test_every_hil_control_verb_is_present() -> None:
    assert set(load("cdc")["verbs"]) == EXPECTED_VERBS


def test_every_verb_has_args_reply_end_profile() -> None:
    for name, v in load("cdc")["verbs"].items():
        assert set(v) >= {"args", "reply", "end", "profile"}, name
        assert isinstance(v["args"], list), name
        for a in v["args"]:
            assert re.fullmatch(r"[a-z_]+:(int|str)\??", a), (name, a)
        assert v["profile"] == "default", name
        assert v["reply"] is None or isinstance(v["reply"], str), name
        if v["reply"] == "multi":
            # A reply that mixes prefixes lists them all (APP_VERSIONS ends its
            # APPVER: rows with a LIBVER: summary).
            prefixes = v["prefix"] if isinstance(v["prefix"], list) else [v["prefix"]]
            assert prefixes, name
            for prefix in prefixes:
                assert prefix.endswith(":"), (name, prefix)


def test_multi_verbs() -> None:
    v = load("cdc")["verbs"]
    assert v["ENC_GROUP"] == {"args": [], "reply": "multi", "prefix": "ENCGROUP:", "end": None,
                              "profile": "default"}
    assert v["APP_VERSIONS"]["end"] == "APPVER: end"
    assert v["APP_VERSIONS"]["prefix"] == ["APPVER:", "LIBVER:", "BOARD:"]
    assert v["MEM_BLOCKS"]["end"] == "MEMBIG:"
    assert v["MEM_BLOCKS"]["prefix"] == "MEMBLK:"


def test_reply_prefixes_from_hil_control_cpp() -> None:
    v = load("cdc")["verbs"]
    assert v["KIT_STATUS"]["reply"] == "KITSTATUS:"
    assert v["APP_LIST"]["reply"] == "APPS:"
    # KIT_LIST is batched: prefix on every line, count in the trailer.
    assert v["KIT_LIST"]["reply"] == "multi"
    assert v["KIT_LIST"]["prefix"] == "KITS:"
    assert v["KIT_LIST"]["end"] == "KITS: end"
    assert v["PAD_STATS"]["reply"] == "PADSTATS:"
    assert v["PAD_NOTES"]["reply"] == "PADNOTES:"
    assert v["PAD_INFO"]["reply"] == "PADINFO:"
    assert v["ENC_FOCUS"]["reply"] == "ENCFOCUS:"
    assert v["ENC_STATE"]["reply"] == "ENC:"
    assert v["UI_STATE"]["reply"] == "UI:"
    assert v["LED_STATE"]["reply"] == "LEDS:"
    assert v["MEM"]["reply"] == "MEM:"
    assert v["CDC_STATS"]["reply"] == "CDCSTATS:"
    assert v["AUDIO_LEVEL"]["reply"] == "AUDIOLVL:"
    assert v["AUDIO_STATUS"]["reply"] == "AUDIO:"
    assert v["SMPL_PEAK"]["reply"] == "SMPLPEAK:"
    assert v["SMPL_ENGINE"]["reply"] == "SMPLENG:"
    assert v["SMPL_PITCH"]["reply"] == "SMPLPITCH:"
    assert v["SMPL_POLY"]["reply"] == "SMPLPOLY:"
    assert v["SMPL_PITCH"]["args"] == ["pad:int", "semitones:int"]
    assert v["SMPL_POLY"]["args"] == ["total:int?", "streaming:int?"]
    assert v["BLE_STATUS"]["reply"] == "BLE:"
    assert v["BLE_DEVICES"]["reply"] == "BLEDEV:"
    assert v["OTA_BEGIN"]["reply"] == "OTA_"
    for ok_verb in ("PAD_PRESS", "KIT_LOAD", "APP_START", "ENC_ROTATE", "BLE_SEND", "STM_DFU",
                    "AUDIO_TASKS", "SMPL_PREVIEW"):
        assert v[ok_verb]["reply"] == "OK", ok_verb
    for silent in ("BOOTLOADER_REQUEST", "USB_AUDIO", "USB_DEFAULT"):
        assert v[silent]["reply"] is None, silent


def test_arg_specs_match_the_sscanf_formats() -> None:
    v = load("cdc")["verbs"]
    assert v["PAD_PRESS"]["args"] == ["idx:int", "vel:int?"]
    assert v["PAD_PRESSURE"]["args"] == ["idx:int", "val:int"]
    assert v["KIT_LOAD"]["args"] == ["kit_id:int"]
    assert v["APP_START"]["args"] == ["name:str"]
    assert v["ENC_PRESS"]["args"] == ["ms:int?"]
    assert v["BLE_SCAN"]["args"] == ["ms:int?"]
    assert v["BLE_START"]["args"] == ["mode:int?"]
    assert v["AUDIO_ADC"]["args"] == ["codec:int", "input:str"]
    assert v["OTA_BEGIN"]["args"] == ["size:int", "version:str"]
