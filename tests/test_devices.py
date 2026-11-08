"""devices: discovery from injected Backends, grouping, usb_mode inference, select()."""
from __future__ import annotations

import sys

import pytest

from crosspad_hil import devices
from crosspad_hil.devices import (
    AudioCardInfo,
    Backends,
    Device,
    MidiPortInfo,
    Ports,
    SerialPortInfo,
    UsbMode,
    device_id_for,
    discover,
    select,
)
from crosspad_hil.errors import AMBIGUOUS_DEVICE, NO_DEVICE, HilError

# ---------------------------------------------------------------- factories


def esp_cdc(path: str = "/dev/ttyACM0", serial: str | None = "A1B2C3",
            location: str | None = "1-3.1:1.0", pid: int = 0x3456) -> SerialPortInfo:
    return SerialPortInfo(path=path, vid=0x303A, pid=pid, serial=serial,
                          product="Crosspad", location=location)


def esp_boot(path: str = "/dev/ttyACM0", serial: str | None = "A1B2C3",
             location: str | None = "1-3.1:1.0") -> SerialPortInfo:
    return SerialPortInfo(path=path, vid=0x303A, pid=0x1001, serial=serial,
                          product="USB JTAG/serial debug unit", location=location)


def stm_vcp(path: str = "/dev/ttyACM1", serial: str | None = "STM001",
            location: str | None = "1-3.2:1.0",
            product: str | None = "CrossPad MIDI+Serial") -> SerialPortInfo:
    return SerialPortInfo(path=path, vid=0x0483, pid=0x5740, serial=serial,
                          product=product, location=location)


def esp_midi(name: str = "Crosspad MIDI 1", out: int | None = 1,
             inp: int | None = 1) -> MidiPortInfo:
    return MidiPortInfo(name=name, rtmidi_out=out, rtmidi_in=inp,
                        alsa_hw="hw:2,0,0", rawmidi="/dev/snd/midiC2D0")


def stm_midi(name: str = "CrossPad MIDI+Serial MIDI 1", out: int | None = 2,
             inp: int | None = 2) -> MidiPortInfo:
    return MidiPortInfo(name=name, rtmidi_out=out, rtmidi_in=inp,
                        alsa_hw="hw:3,0,0", rawmidi="/dev/snd/midiC3D0")


def uac2(name: str = "Crosspad Audio: USB Audio (hw:2,0)", index: int = 4) -> AudioCardInfo:
    return AudioCardInfo(name=name, sounddevice_index=index, alsa_id="Audio")


def backends(serial=(), midi=(), audio=()) -> Backends:
    return Backends(list_serial=lambda: list(serial),
                    list_midi=lambda: list(midi),
                    list_audio=lambda: list(audio))


# ---------------------------------------------------------------- ids


def test_device_id_for_is_stable_sha1_prefix() -> None:
    assert device_id_for("A1B2C3", "/dev/x") == device_id_for("A1B2C3", "/dev/y")
    assert device_id_for("A1B2C3", "/dev/x").startswith("dev_")
    assert len(device_id_for("A1B2C3", "/dev/x")) == 8
    assert device_id_for(None, "/dev/x") == device_id_for(None, "/dev/x")
    assert device_id_for(None, "/dev/x") != device_id_for(None, "/dev/y")


# ---------------------------------------------------------------- predicates


def test_name_predicates() -> None:
    assert devices.is_esp_midi_name("Crosspad MIDI 1")
    assert devices.is_esp_midi_name("CROSSPAD MIDI 1:Crosspad MIDI 1 MIDI 1 24:0")
    assert not devices.is_esp_midi_name("CrossPad MIDI+Serial MIDI 1")
    # midi_hub's own virtual ports are not a board (a phantom "Board found" otherwise)
    assert not devices.is_esp_midi_name("RtMidiIn Client:CrossPad Hub 131:0")
    assert devices.is_stm_midi_name("CrossPad MIDI+Serial MIDI 1")
    assert not devices.is_stm_midi_name("Crosspad MIDI 1")
    assert devices.is_uac2_name("Crosspad Audio: USB Audio (hw:2,0)")
    assert not devices.is_uac2_name("HDA Intel PCH: ALC (hw:0,0)")


def test_bcd_decoders() -> None:
    assert devices.fw_rev_from_bcd(0x0521) == "v2"
    assert devices.fw_rev_from_bcd(0x0510) == "v1"
    assert devices.fw_rev_from_bcd(0x0501) is None      # pre-revision image
    assert devices.fw_rev_from_bcd(0x0101) is None      # ROM download mode
    assert devices.fw_rev_from_bcd(None) is None
    assert devices.pcb_from_bcd(0x3020) == 20
    assert devices.pcb_from_bcd(0x3019) == 19
    assert devices.pcb_from_bcd(0x0200) is None         # STM image without the PCB
    assert devices.pcb_from_bcd(0x302A) is None         # not BCD
    assert devices.board_rev_from_pcb(20) == "v2"
    assert devices.board_rev_from_pcb(19) == "v1"
    assert devices.board_rev_from_pcb(16) == "v1"
    assert devices.board_rev_from_pcb(99) is None


def test_discover_reports_hardware_and_firmware_revision() -> None:
    esp = esp_cdc()
    esp.bcd = 0x0511
    stm = stm_vcp()
    stm.bcd = 0x3020
    d = discover(backends(serial=[esp, stm], midi=[], audio=[]))[0]
    assert (d.board_rev, d.pcb, d.fw_rev) == ("v2", 20, "v1")
    assert d.to_dict()["pcb"] == 20 and d.to_dict()["fw_rev"] == "v1"


def test_serial_predicates_and_stm_product_rule() -> None:
    assert devices.is_esp_cdc(esp_cdc())
    assert devices.is_esp_cdc(esp_cdc(pid=0x4001))
    assert not devices.is_esp_cdc(esp_boot())
    assert devices.is_esp_bootloader(esp_boot())
    assert devices.is_esp_bootloader(SerialPortInfo("/dev/x", 0x303A, 0x0009, None, None, None))
    assert devices.is_stm_console(stm_vcp())
    assert devices.is_stm_console(stm_vcp(product=None))
    assert not devices.is_stm_console(stm_vcp(product="STM32 Virtual ComPort"))
    assert not devices.is_stm_console(SerialPortInfo("/dev/x", 0x1A86, 0x7523, None, None, None))


# ---------------------------------------------------------------- discover


def test_one_board_default_mode() -> None:
    devs = discover(backends(serial=[esp_cdc(), stm_vcp()],
                             midi=[esp_midi(), stm_midi()], audio=[]))
    assert len(devs) == 1
    d = devs[0]
    assert d.id == device_id_for("STM001", "/dev/ttyACM1")  # bridge serial, not the ESP's
    assert d.serial == "A1B2C3"
    assert d.usb_mode is UsbMode.DEFAULT
    assert d.ports.cdc is not None and d.ports.cdc.path == "/dev/ttyACM0"
    assert d.ports.console is not None and d.ports.console.path == "/dev/ttyACM1"
    assert d.ports.esp_midi is not None and d.ports.esp_midi.rtmidi_out == 1
    assert d.ports.stm_midi is not None and d.ports.stm_midi.name.endswith("MIDI 1")
    assert d.ports.uac2 is None
    assert d.ports.bootloader is None
    assert d.board_rev is None


def test_one_board_audio_mode_has_no_cdc_but_uac2() -> None:
    devs = discover(backends(serial=[stm_vcp()], midi=[esp_midi(), stm_midi()],
                             audio=[uac2()]))
    assert len(devs) == 1
    d = devs[0]
    assert d.usb_mode is UsbMode.AUDIO
    assert d.ports.cdc is None
    assert d.ports.console is not None
    assert d.ports.uac2 is not None and d.ports.uac2.sounddevice_index == 4
    assert d.ports.esp_midi is not None
    assert d.id == device_id_for("STM001", "/dev/ttyACM1")


def test_one_board_bootloader() -> None:
    devs = discover(backends(serial=[esp_boot(), stm_vcp()]))
    assert len(devs) == 1
    d = devs[0]
    assert d.usb_mode is UsbMode.BOOTLOADER
    assert d.ports.bootloader is not None and d.ports.bootloader.path == "/dev/ttyACM0"
    assert d.ports.cdc is None
    assert d.ports.console is not None


def test_two_boards_paired_by_location_prefix() -> None:
    ports = [
        esp_cdc("/dev/ttyACM0", "AAAA", "1-3.1:1.0"),
        stm_vcp("/dev/ttyACM3", "STMB", "1-4.2:1.0"),
        esp_cdc("/dev/ttyACM2", "BBBB", "1-4.1:1.0"),
        stm_vcp("/dev/ttyACM1", "STMA", "1-3.2:1.0"),
    ]
    devs = discover(backends(serial=ports, midi=[esp_midi("Crosspad MIDI 1", 1, 1),
                                                  esp_midi("Crosspad MIDI 2", 2, 2)]))
    assert len(devs) == 2
    by_serial = {d.serial: d for d in devs}
    assert by_serial["AAAA"].ports.console.path == "/dev/ttyACM1"
    assert by_serial["BBBB"].ports.console.path == "/dev/ttyACM3"
    assert by_serial["AAAA"].ports.esp_midi.name == "Crosspad MIDI 1"
    assert by_serial["BBBB"].ports.esp_midi.name == "Crosspad MIDI 2"
    assert all(d.usb_mode is UsbMode.DEFAULT for d in devs)


def test_unpaired_stm_console_is_its_own_unknown_device() -> None:
    ports = [esp_cdc("/dev/ttyACM0", "AAAA", "1-3.1:1.0"),
             stm_vcp("/dev/ttyACM1", "STMA", "1-3.2:1.0"),
             stm_vcp("/dev/ttyACM5", "STMZ", "2-1.2:1.0")]
    devs = discover(backends(serial=ports))
    assert len(devs) == 2
    lonely = [d for d in devs if d.serial == "STMZ"][0]
    assert lonely.usb_mode is UsbMode.UNKNOWN
    assert lonely.ports.console.path == "/dev/ttyACM5"
    assert lonely.ports.cdc is None and lonely.ports.esp_midi is None
    paired = [d for d in devs if d.serial == "AAAA"][0]
    assert paired.ports.console.path == "/dev/ttyACM1"


def test_no_locations_two_of_each_stay_unpaired() -> None:
    ports = [esp_cdc("COM3", "AAAA", None), esp_cdc("COM5", "BBBB", None),
             stm_vcp("COM4", "STMA", None), stm_vcp("COM6", "STMB", None)]
    devs = discover(backends(serial=ports))
    assert len(devs) == 4
    assert sum(1 for d in devs if d.ports.cdc) == 2
    assert sum(1 for d in devs if d.ports.console and not d.ports.cdc) == 2


def test_nothing_connected() -> None:
    assert discover(backends()) == []


def test_midi_only_synthesizes_device() -> None:
    devs = discover(backends(midi=[esp_midi()]))
    assert len(devs) == 1
    assert devs[0].usb_mode is UsbMode.UNKNOWN
    assert devs[0].serial is None
    assert devs[0].ports.esp_midi is not None


def test_to_dict_shape() -> None:
    d = discover(backends(serial=[esp_cdc(), stm_vcp()], midi=[esp_midi()]))[0]
    dd = d.to_dict()
    assert dd["id"] == d.id
    assert dd["usb_mode"] == "default"
    assert dd["serial"] == "A1B2C3"
    assert dd["board_rev"] is None
    assert dd["ports"]["cdc"] == {"path": "/dev/ttyACM0", "vid": 0x303A, "pid": 0x3456,
                                  "serial": "A1B2C3", "product": "Crosspad",
                                  "location": "1-3.1:1.0", "bcd": None}
    assert dd["ports"]["uac2"] is None
    assert dd["ports"]["esp_midi"]["rtmidi_out"] == 1


def test_ports_paths() -> None:
    p = Ports(cdc=esp_cdc(), console=stm_vcp())
    assert p.paths() == {"cdc": "/dev/ttyACM0", "console": "/dev/ttyACM1"}
    assert Ports(bootloader=esp_boot("/dev/ttyACM9")).paths() == {"bootloader": "/dev/ttyACM9"}


# ---------------------------------------------------------------- select


def _two() -> list[Device]:
    return discover(backends(serial=[
        esp_cdc("/dev/ttyACM0", "AAAA", "1-3.1:1.0"), stm_vcp("/dev/ttyACM1", "STMA", "1-3.2:1.0"),
        esp_cdc("/dev/ttyACM2", "BBBB", "1-4.1:1.0"), stm_vcp("/dev/ttyACM3", "STMB", "1-4.2:1.0"),
    ]))


def test_select_implicit_single() -> None:
    devs = discover(backends(serial=[esp_cdc(), stm_vcp()]))
    assert select(devs) is devs[0]


def test_select_implicit_ignores_lonely_stm() -> None:
    devs = discover(backends(serial=[esp_cdc("/dev/ttyACM0", "AAAA", "1-3.1:1.0"),
                                     stm_vcp("/dev/ttyACM1", "STMA", "1-3.2:1.0"),
                                     stm_vcp("/dev/ttyACM5", "STMZ", "2-1.2:1.0")]))
    assert select(devs).serial == "AAAA"


def test_select_none_raises_no_device_with_hint() -> None:
    with pytest.raises(HilError) as ei:
        select([])
    assert ei.value.code == NO_DEVICE
    assert "bootloader/DFU" in ei.value.message


def test_select_audio_mode_board_counts_as_esp_side() -> None:
    devs = discover(backends(serial=[stm_vcp()], midi=[esp_midi()], audio=[uac2()]))
    assert select(devs).usb_mode is UsbMode.AUDIO


def test_select_ambiguous_lists_candidates() -> None:
    devs = _two()
    with pytest.raises(HilError) as ei:
        select(devs)
    err = ei.value
    assert err.code == AMBIGUOUS_DEVICE
    assert err.hint is not None and "device=" in err.hint
    ids = sorted(c["id"] for c in err.details["candidates"])
    assert ids == sorted(d.id for d in devs)
    assert all("ports" in c and "usb_mode" in c for c in err.details["candidates"])


def test_select_by_id() -> None:
    devs = _two()
    want = devs[1]
    assert select(devs, want.id) is want


def test_select_by_port_path_any_role() -> None:
    devs = _two()
    assert select(devs, "/dev/ttyACM3").serial == "BBBB"   # console path
    assert select(devs, "/dev/ttyACM0").serial == "AAAA"   # cdc path


def test_select_unknown_path_hint_names_roles() -> None:
    devs = _two()
    with pytest.raises(HilError) as ei:
        select(devs, "/dev/ttyUSB9")
    err = ei.value
    assert err.code == NO_DEVICE
    assert err.hint is not None
    assert "/dev/ttyACM1 (console)" in err.hint
    assert "/dev/ttyACM0 (cdc)" in err.hint
    assert err.details["device"] == "/dev/ttyUSB9"


# ---------------------------------------------------------------- Linux parsers


AMIDI_L = """Dir Device    Name
IO  hw:2,0,0  Crosspad MIDI 1
IO  hw:3,0,0  CrossPad MIDI+Serial MIDI 1
"""

ASOUND_CARDS = """ 0 [PCH            ]: HDA-Intel - HDA Intel PCH
                      HDA Intel PCH at 0xa1210000 irq 148
 2 [MIDI           ]: USB-Audio - Crosspad MIDI
                      TinyUSB Crosspad MIDI at usb-0000:00:14.0-3.1, full speed
 3 [MIDISerial     ]: USB-Audio - CrossPad MIDI+Serial
                      STMicroelectronics CrossPad MIDI+Serial at usb-0000:00:14.0-3.2, full speed
"""

ARECORD_L = """**** List of CAPTURE Hardware Devices ****
card 0: PCH [HDA Intel PCH], device 0: ALC236 Analog [ALC236 Analog]
  Subdevices: 1/1
  Subdevice #0: subdevice #0
card 2: Audio [Crosspad Audio], device 0: USB Audio [USB Audio]
  Subdevices: 1/1
  Subdevice #0: subdevice #0
"""


def test_parse_amidi_list() -> None:
    assert devices.parse_amidi_list(AMIDI_L) == [
        ("hw:2,0,0", "Crosspad MIDI 1"),
        ("hw:3,0,0", "CrossPad MIDI+Serial MIDI 1"),
    ]
    assert devices.parse_amidi_list("") == []


def test_parse_asound_cards() -> None:
    assert devices.parse_asound_cards(ASOUND_CARDS) == [
        (0, "PCH", "HDA Intel PCH"),
        (2, "MIDI", "Crosspad MIDI"),
        (3, "MIDISerial", "CrossPad MIDI+Serial"),
    ]


def test_parse_arecord_list() -> None:
    assert devices.parse_arecord_list(ARECORD_L) == [
        (0, "PCH", "HDA Intel PCH"),
        (2, "Audio", "Crosspad Audio"),
    ]


def test_rawmidi_node() -> None:
    assert devices.rawmidi_node("hw:3,0,0") == "/dev/snd/midiC3D0"
    assert devices.rawmidi_node("hw:12,1") == "/dev/snd/midiC12D1"


def test_location_pairing_helpers() -> None:
    assert devices.location_key("1-3.2:1.0") == "1-3.2"
    assert devices.location_key(None) == ""
    assert devices.common_prefix_len("1-3.1", "1-3.2") == 4
    assert devices.common_prefix_len("1-3.1", "2-1.2") == 0


def test_enrich_midi_from_alsa_fills_hw_and_rawmidi() -> None:
    ports = [esp_midi(out=0, inp=0), stm_midi(out=1, inp=1)]
    for p in ports:
        p.alsa_hw = None
        p.rawmidi = None
    devices.enrich_midi_from_alsa(ports, AMIDI_L, ASOUND_CARDS)
    assert ports[0].alsa_hw == "hw:2,0,0" and ports[0].rawmidi == "/dev/snd/midiC2D0"
    assert ports[1].alsa_hw == "hw:3,0,0" and ports[1].rawmidi == "/dev/snd/midiC3D0"


def test_enrich_midi_from_asound_cards_without_amidi() -> None:
    ports = [stm_midi(out=1, inp=1)]
    ports[0].alsa_hw = None
    ports[0].rawmidi = None
    devices.enrich_midi_from_alsa(ports, "", ASOUND_CARDS)
    # hil_speedtest.py stm_midi_card: card number from /proc/asound/cards, device 0
    assert ports[0].rawmidi == "/dev/snd/midiC3D0"
    assert ports[0].alsa_hw == "hw:3,0,0"


def test_midi_ports_from_amidi_when_rtmidi_missing() -> None:
    ports = devices.midi_ports_from_amidi(AMIDI_L)
    assert [p.name for p in ports] == ["Crosspad MIDI 1", "CrossPad MIDI+Serial MIDI 1"]
    assert ports[0].rtmidi_out is None and ports[0].rtmidi_in is None
    assert ports[0].alsa_hw == "hw:2,0,0"


def test_audio_cards_from_arecord() -> None:
    cards = devices.audio_cards_from_arecord(ARECORD_L)
    assert cards == [AudioCardInfo(name="Crosspad Audio", sounddevice_index=None, alsa_id="Audio")]


def test_default_backends_without_optional_deps(monkeypatch: pytest.MonkeyPatch) -> None:
    # importing a module set to None in sys.modules raises ImportError
    monkeypatch.setitem(__import__("sys").modules, "rtmidi", None)
    monkeypatch.setitem(__import__("sys").modules, "sounddevice", None)
    monkeypatch.setattr(devices, "_run", lambda argv, timeout=5.0: "")
    monkeypatch.setattr(devices, "_read_text", lambda path: "")
    b = devices.default_backends()
    assert b.list_midi() == []
    assert b.list_audio() == []
    assert isinstance(b.list_serial(), list)


def test_default_backends_rtmidi_names(monkeypatch: pytest.MonkeyPatch) -> None:
    import sys
    import types

    class _Out:
        def get_ports(self) -> list[str]:
            return ["Midi Through 14:0", "Crosspad MIDI 1 24:0", "CrossPad MIDI+Serial MIDI 1 28:0"]

    class _In:
        def get_ports(self) -> list[str]:
            return ["Crosspad MIDI 1 24:0", "CrossPad MIDI+Serial MIDI 1 28:0"]

    fake = types.ModuleType("rtmidi")
    fake.MidiOut = _Out  # type: ignore[attr-defined]
    fake.MidiIn = _In  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "rtmidi", fake)
    monkeypatch.setattr(devices, "_run", lambda argv, timeout=5.0: "")
    monkeypatch.setattr(devices, "_read_text", lambda path: "")
    ports = devices.default_backends().list_midi()
    assert [(p.name, p.rtmidi_out, p.rtmidi_in) for p in ports] == [
        ("Crosspad MIDI 1 24:0", 1, 0),
        ("CrossPad MIDI+Serial MIDI 1 28:0", 2, 1),
    ]


@pytest.mark.skipif(not sys.platform.startswith("linux"),
                     reason="alsa_id merge only runs on Linux (devices.py gates arecord there)")
def test_default_backends_sounddevice(monkeypatch: pytest.MonkeyPatch) -> None:
    import sys
    import types

    fake = types.ModuleType("sounddevice")
    fake.query_devices = lambda: [  # type: ignore[attr-defined]
        {"name": "HDA Intel PCH: ALC236 (hw:0,0)", "max_input_channels": 2},
        {"name": "Crosspad Audio: USB Audio (hw:2,0)", "max_input_channels": 0},
        {"name": "Crosspad Audio: USB Audio (hw:2,0)", "max_input_channels": 2},
    ]
    monkeypatch.setitem(sys.modules, "sounddevice", fake)
    monkeypatch.setattr(devices, "_run", lambda argv, timeout=5.0: ARECORD_L)
    cards = devices.default_backends().list_audio()
    assert cards == [AudioCardInfo(name="Crosspad Audio: USB Audio (hw:2,0)",
                                   sounddevice_index=2, alsa_id="Audio")]


# ---------------------------------------------------------------- rtmidi teardown

def test_rtmidi_list_unregisters_both_handles(monkeypatch) -> None:
    import types

    made: list = []

    class _FakeRt:
        def __init__(self) -> None:
            self.deleted = False
            made.append(self)

        def get_ports(self) -> list[str]:
            return ["Crosspad MIDI 1"]

        def delete(self) -> None:
            self.deleted = True

    monkeypatch.setitem(sys.modules, "rtmidi",
                        types.SimpleNamespace(MidiOut=_FakeRt, MidiIn=_FakeRt))
    monkeypatch.setattr(devices, "_run", lambda *a, **k: "")
    monkeypatch.setattr(devices, "_read_text", lambda *a, **k: "")
    devices._rtmidi_list()
    assert len(made) == 2 and all(h.deleted for h in made)


def test_rtmidi_list_unregisters_even_when_get_ports_raises(monkeypatch) -> None:
    import types

    made: list = []

    class _FakeRt:
        def __init__(self) -> None:
            self.deleted = False
            made.append(self)

        def get_ports(self) -> list[str]:
            raise RuntimeError("backend is wedged")

        def delete(self) -> None:
            self.deleted = True

    monkeypatch.setitem(sys.modules, "rtmidi",
                        types.SimpleNamespace(MidiOut=_FakeRt, MidiIn=_FakeRt))
    monkeypatch.setattr(devices, "_run", lambda *a, **k: "")
    monkeypatch.setattr(devices, "_read_text", lambda *a, **k: "")
    assert devices._rtmidi_list() == []
    assert made and all(h.deleted for h in made)


def test_windows_hardware_ids_give_the_bcd_the_revision_comes_from() -> None:
    # Get-CimInstance Win32_PnPEntity output as seen on a clean Windows 11 with a v2 board.
    text = (
        "USB\\VID_303A&PID_3456\\44:1B:F6:8B:22:38|USB\\VID_303A&PID_3456&REV_0521,USB\\VID_303A&PID_3456\n"
        "USB\\VID_303A&PID_3456&MI_00\\6&35D700E4&0&0000|USB\\VID_303A&PID_3456&REV_0521&MI_00\n"
        "USB\\VID_0483&PID_5740\\205F37A3564B|USB\\VID_0483&PID_5740&REV_3020,USB\\VID_0483&PID_5740\n"
        "USB\\ROOT_HUB30\\4&1|USB\\ROOT_HUB30\n")
    bcds = devices.parse_windows_usb_ids(text)
    assert bcds == {(0x303A, 0x3456, "44:1B:F6:8B:22:38"): 0x0521,
                    (0x0483, 0x5740, "205F37A3564B"): 0x3020}
    assert devices.fw_rev_from_bcd(bcds[(0x303A, 0x3456, "44:1B:F6:8B:22:38")]) == "v2"
    assert devices.pcb_from_bcd(bcds[(0x0483, 0x5740, "205F37A3564B")]) == 20


# ---------------------------------------------------------------- other ESP32s


def backends_with_macs(serial=(), known=()):
    store = {"macs": set(known), "remembered": set()}
    b = Backends(list_serial=lambda: list(serial), list_midi=lambda: [], list_audio=lambda: [],
                 known_macs=lambda: set(store["macs"]),
                 remember_macs=lambda macs: store["remembered"].update(macs))
    return b, store


def test_an_unknown_esp32_in_rom_beside_a_crosspad_is_not_a_crosspad() -> None:
    # The CrossPad in its CDC profile and two other ESP32-S3 projects in ROM
    # (USB JTAG/serial, 303a:1001) on other hubs: one board, not three.
    cp = esp_cdc("/dev/ttyACM2", serial="44:1B:F6:8B:22:38", location="1-3:1.0")
    other1 = esp_boot("/dev/ttyACM0", serial="74:4D:BD:87:84:34", location="1-4:1.0")
    other2 = esp_boot("/dev/ttyACM1", serial="64:E8:33:4B:58:40", location="7-2:1.0")
    b, _ = backends_with_macs([cp, other1, other2])
    devs = discover(b)
    assert len(devs) == 1
    assert devs[0].ports.cdc is not None and devs[0].ports.cdc.path == "/dev/ttyACM2"


def test_a_lone_unknown_esp32_in_rom_is_not_a_crosspad() -> None:
    b, _ = backends_with_macs([esp_boot(serial="74:4D:BD:87:84:34")])
    assert discover(b) == []


def test_a_crosspad_in_rom_is_known_by_its_mac() -> None:
    boot = esp_boot(serial="44:1B:F6:8B:22:38")
    b, _ = backends_with_macs([boot, esp_boot("/dev/ttyACM5", serial="74:4D:BD:87:84:34",
                                              location="7-2:1.0")],
                              known={"44:1B:F6:8B:22:38"})
    devs = discover(b)
    assert len(devs) == 1
    assert devs[0].usb_mode is UsbMode.BOOTLOADER
    assert devs[0].ports.bootloader is not None and devs[0].ports.bootloader.serial == "44:1B:F6:8B:22:38"


def test_a_crosspad_seen_in_cdc_is_remembered_by_its_mac() -> None:
    b, store = backends_with_macs([esp_cdc(serial="44:1B:F6:8B:22:38")])
    discover(b)
    assert "44:1B:F6:8B:22:38" in store["remembered"]


def test_a_rom_port_paired_with_the_stm_bridge_needs_no_known_mac() -> None:
    b, _ = backends_with_macs([esp_boot(serial="AA:BB:CC:DD:EE:FF"), stm_vcp()])
    devs = discover(b)
    assert len(devs) == 1 and devs[0].ports.bootloader is not None


def test_the_stm_bridge_pairs_with_the_crosspad_not_with_other_esp32s() -> None:
    # CrossPad CDC on bus 1, its STM bridge plugged into another bus, two
    # unrelated ESP32-S3s in ROM: one board, with its console.
    cp = esp_cdc("/dev/ttyACM2", serial="44:1B:F6:8B:22:38", location="1-3:1.0")
    other1 = esp_boot("/dev/ttyACM0", serial="74:4D:BD:87:84:34", location="1-4:1.0")
    other2 = esp_boot("/dev/ttyACM1", serial="64:E8:33:4B:58:40", location="7-2:1.0")
    stm = stm_vcp("/dev/ttyACM3", location="3-1:1.0")
    b, _ = backends_with_macs([cp, other1, other2, stm])
    devs = discover(b)
    assert len(devs) == 1
    assert devs[0].ports.cdc is not None and devs[0].ports.console is not None


def test_an_unknown_rom_port_still_pairs_with_a_lone_stm_bridge() -> None:
    # Recovery: the CrossPad's ESP is in ROM and its MAC was never seen; the
    # STM bridge beside it on the same hub vouches for it.
    boot = esp_boot("/dev/ttyACM0", serial="AA:BB:CC:DD:EE:FF", location="1-3.1:1.0")
    other = esp_boot("/dev/ttyACM1", serial="64:E8:33:4B:58:40", location="7-2:1.0")
    b, _ = backends_with_macs([boot, other, stm_vcp("/dev/ttyACM3", location="1-3.2:1.0")])
    devs = discover(b)
    assert len(devs) == 1
    assert devs[0].ports.bootloader is not None and devs[0].ports.bootloader.path == "/dev/ttyACM0"


def test_an_unknown_esp32_in_rom_does_not_take_the_bridge_of_a_crosspad_in_its_audio_profile() -> None:
    # The CrossPad in MIDI+UAC2 (no CDC, so no serial port of its own), its
    # STM bridge, and another project's ESP32-S3 in ROM on another bus. The
    # bridge is not spare: its ESP is the one behind the MIDI port and the
    # card. It read as dev_xxxx in usb_mode=bootloader, and bench.py ready
    # then refused the board.
    serial = [stm_vcp("/dev/ttyACM0", location="1-3:1.0"),
              esp_boot("/dev/ttyACM2", serial="64:E8:33:4B:58:40", location="7-2:1.0")]
    b = Backends(list_serial=lambda: list(serial),
                 list_midi=lambda: [esp_midi("Crosspad Audio:Crosspad Audio MIDI 1 36:0"), stm_midi()],
                 list_audio=lambda: [uac2()],
                 known_macs=lambda: {"44:1B:F6:8B:22:38"}, remember_macs=lambda macs: None)
    devs = discover(b)
    assert len(devs) == 1
    d = devs[0]
    assert d.usb_mode is UsbMode.AUDIO
    assert d.ports.bootloader is None
    assert d.ports.esp_midi is not None and d.ports.uac2 is not None and d.ports.console is not None
