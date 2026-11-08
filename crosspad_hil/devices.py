"""Discovery of CrossPad boards and the ports each one exposes (spec §2.3).

A CrossPad shows up on the host as several USB devices at once:

* ESP native-USB CDC ``0x303A:0x3456`` (or ``0x4001``) — the HIL command port.
  Gone in the MIDI+UAC2 profile.
* ESP ROM bootloader ``0x303A:0x1001`` / ``0x0009`` — after BOOTLOADER_REQUEST.
* STM32 bridge VCP ``0x0483:0x5740`` — the ESP console, survives ESP resets and
  USB mode switches (product string contains "crosspad").
* USB MIDI: ``Crosspad MIDI*`` (ESP, both profiles) and
  ``CrossPad MIDI+Serial`` (STM bridge).
* UAC2 ``Crosspad Audio`` — only in the audio profile.

The STM bridge's ``bcdDevice`` carries the PCB revision; the ESP's carries the
firmware image's revision.

One pass over three injectable backends (pyserial / rtmidi / sounddevice, with
ALSA text enrichment on Linux) merges those into ``Device`` records, infers the
USB mode, and ``select()`` applies the one-board-implicit rule.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import subprocess
import sys
from collections.abc import Callable
from dataclasses import asdict, dataclass, field, fields
from enum import Enum

from crosspad_hil.errors import AMBIGUOUS_DEVICE, NO_DEVICE, HilError

# ---------------------------------------------------------------- constants

ESP_VID = 0x303A                       # hil_smoke.py ESPRESSIF_VID
ESP_CDC_PIDS = {0x3456, 0x4001}        # requestBootloader.py ESPRESSIF_NORMAL_PIDS
ESP_BOOT_PIDS = {0x1001, 0x0009}       # requestBootloader.py ESPRESSIF_BOOTLOADER_PIDS
STM_VID = 0x0483                       # hil_smoke.py STM_BRIDGE_VID
STM_PID = 0x5740                       # hil_smoke.py STM_BRIDGE_PID

ESP_MIDI_NAME_MATCH = "crosspad"       # contains, case-insensitive, and not STM
HUB_MIDI_NAME = "CrossPad Hub"         # midi_hub's own virtual ports, never a board
STM_MIDI_NAME_MATCH = "MIDI+Serial"    # hil_audio_loopback.py STM_MIDI_NAME
UAC2_NAME_MATCH = "Crosspad Audio"     # hil_sampler_record.py AUDIO_DEV_MATCH

ESP_BCD_MAJOR = 0x05                  # platform-idf usb_descriptors.h USB_BCD_BASE >> 8
ESP_BCD_REVS = {1: "v1", 2: "v2"}     # BSP_BOARD_REV_NUM
STM_BCD_PCB_MARK = 0x30               # CrossPad_STM32_r2 usbd_composite_cdc_midi.c
PCB_REVS = {16: "v1", 19: "v1", 20: "v2"}   # PCB_REVISION of each STM line

_SUBPROCESS_TIMEOUT_S = 5.0


# ---------------------------------------------------------------- model


class UsbMode(str, Enum):
    DEFAULT = "default"
    AUDIO = "audio"
    BOOTLOADER = "bootloader"
    UNKNOWN = "unknown"


@dataclass
class SerialPortInfo:
    path: str
    vid: int
    pid: int
    serial: str | None
    product: str | None
    location: str | None
    bcd: int | None = None


@dataclass
class MidiPortInfo:
    name: str
    rtmidi_out: int | None
    rtmidi_in: int | None
    alsa_hw: str | None
    rawmidi: str | None


@dataclass
class AudioCardInfo:
    name: str
    sounddevice_index: int | None
    alsa_id: str | None


@dataclass
class Ports:
    cdc: SerialPortInfo | None = None
    console: SerialPortInfo | None = None
    esp_midi: MidiPortInfo | None = None
    stm_midi: MidiPortInfo | None = None
    uac2: AudioCardInfo | None = None
    bootloader: SerialPortInfo | None = None

    def paths(self) -> dict[str, str]:
        """Serial paths by role — the things a lock or a `select()` path can name."""
        out: dict[str, str] = {}
        for role in ("cdc", "console", "bootloader"):
            port = getattr(self, role)
            if port is not None:
                out[role] = port.path
        return out


@dataclass
class Device:
    id: str
    serial: str | None
    usb_mode: UsbMode
    ports: Ports
    board_rev: str | None = None
    pcb: int | None = None
    fw_rev: str | None = None

    def to_dict(self) -> dict:
        ports = {
            f.name: (None if getattr(self.ports, f.name) is None
                     else asdict(getattr(self.ports, f.name)))
            for f in fields(Ports)
        }
        return {
            "id": self.id,
            "serial": self.serial,
            "usb_mode": self.usb_mode.value,
            "board_rev": self.board_rev,
            "pcb": self.pcb,
            "fw_rev": self.fw_rev,
            "ports": ports,
        }

    def has_esp_side(self) -> bool:
        """True when the ESP is reachable: CDC, bootloader, UAC2 or its own MIDI port."""
        p = self.ports
        return any(x is not None for x in (p.cdc, p.bootloader, p.uac2, p.esp_midi))


@dataclass
class Backends:
    list_serial: Callable[[], list[SerialPortInfo]]
    list_midi: Callable[[], list[MidiPortInfo]]
    list_audio: Callable[[], list[AudioCardInfo]]
    # MACs of the ESPs that have shown up as a CrossPad (their CDC serial): an
    # ESP32 in ROM (303a:1001) is a CrossPad only if its MAC is one of these or
    # it sits beside a CrossPad's STM bridge -- any other ESP32-S3 on the desk
    # enumerates exactly the same way.
    known_macs: Callable[[], set[str]] = field(default=lambda: set())
    remember_macs: Callable[[set[str]], None] = field(default=lambda macs: None)


def fw_rev_from_bcd(bcd: int | None) -> str | None:
    """The revision the ESP image was built for; None for older images and the ROM."""
    if bcd is None or bcd >> 8 != ESP_BCD_MAJOR:
        return None
    return ESP_BCD_REVS.get((bcd >> 4) & 0xF)


def pcb_from_bcd(bcd: int | None) -> int | None:
    """The PCB the STM bridge reads from OTP; None when the image does not say."""
    if bcd is None or bcd >> 8 != STM_BCD_PCB_MARK:
        return None
    tens, ones = (bcd >> 4) & 0xF, bcd & 0xF
    if tens > 9 or ones > 9:
        return None
    return tens * 10 + ones


def board_rev_from_pcb(pcb: int | None) -> str | None:
    return PCB_REVS.get(pcb) if pcb is not None else None


def device_id_for(serial: str | None, fallback: str) -> str:
    """``"dev_" + sha1(serial or fallback)[:4]`` — stable across re-enumeration."""
    key = serial if serial else fallback
    return "dev_" + hashlib.sha1(key.encode("utf-8")).hexdigest()[:4]


# ---------------------------------------------------------------- predicates


def is_esp_cdc(p: SerialPortInfo) -> bool:
    return p.vid == ESP_VID and p.pid in ESP_CDC_PIDS


def is_esp_bootloader(p: SerialPortInfo) -> bool:
    return p.vid == ESP_VID and p.pid in ESP_BOOT_PIDS


def is_stm_console(p: SerialPortInfo) -> bool:
    # hil_smoke.py candidate_ports: STM VCP whose product names a CrossPad.
    # A None product (Windows) is accepted; another product on ST's VID:PID is not.
    if p.vid != STM_VID or p.pid != STM_PID:
        return False
    return p.product is None or "crosspad" in p.product.lower()


def is_stm_midi_name(name: str) -> bool:
    return STM_MIDI_NAME_MATCH in name


def is_esp_midi_name(name: str) -> bool:
    low = name.lower()
    return (ESP_MIDI_NAME_MATCH in low and not is_stm_midi_name(name)
            and HUB_MIDI_NAME.lower() not in low)


def is_uac2_name(name: str) -> bool:
    return UAC2_NAME_MATCH in name


# ---------------------------------------------------------------- Linux text parsers

# amidi -l:  "IO  hw:3,0,0  CrossPad MIDI+Serial MIDI 1"   (hil_midi_bench.py rawmidi_dev)
_AMIDI_RE = re.compile(r"^\s*[IO]+\s+(hw:\d+,\d+(?:,\d+)?)\s+(.*\S)\s*$")
# /proc/asound/cards:  " 3 [MIDISerial     ]: USB-Audio - CrossPad MIDI+Serial"
_ASOUND_CARD_RE = re.compile(r"^\s*(\d+)\s+\[([^\]]*?)\s*\]:\s*(?:\S+\s+-\s+)?(.*\S)\s*$")
# arecord -l:  "card 2: Audio [Crosspad Audio], device 0: USB Audio [USB Audio]"
_ARECORD_RE = re.compile(r"^card (\d+): (\S+) \[([^\]]*)\], device \d+:")
_HW_RE = re.compile(r"hw:(\d+),(\d+)")


def parse_amidi_list(text: str) -> list[tuple[str, str]]:
    """``[(hw:X,Y,Z, name)]`` in listing order."""
    out: list[tuple[str, str]] = []
    for line in text.splitlines():
        m = _AMIDI_RE.match(line)
        if m:
            out.append((m.group(1), m.group(2)))
    return out


def parse_asound_cards(text: str) -> list[tuple[int, str, str]]:
    """``[(card_number, card_id, long_name)]`` from /proc/asound/cards
    (hil_speedtest.py stm_midi_card)."""
    out: list[tuple[int, str, str]] = []
    for line in text.splitlines():
        m = _ASOUND_CARD_RE.match(line)
        if m:
            out.append((int(m.group(1)), m.group(2), m.group(3)))
    return out


def parse_arecord_list(text: str) -> list[tuple[int, str, str]]:
    """``[(card_number, alsa_id, human_name)]`` from ``arecord -l``
    (hil_audio_loopback.py CARD_NAME / hil_usb_mode_cycle.py uac_present)."""
    out: list[tuple[int, str, str]] = []
    for line in text.splitlines():
        m = _ARECORD_RE.match(line)
        if m:
            out.append((int(m.group(1)), m.group(2), m.group(3)))
    return out


def rawmidi_node(hw: str) -> str:
    """``hw:3,0,0`` → ``/dev/snd/midiC3D0`` (hil_midi_bench.py rawmidi_dev)."""
    m = _HW_RE.search(hw)
    if not m:
        raise ValueError(f"not an ALSA hw spec: {hw!r}")
    return f"/dev/snd/midiC{m.group(1)}D{m.group(2)}"


def _midi_role(name: str) -> str | None:
    if is_stm_midi_name(name):
        return "stm"
    if is_esp_midi_name(name):
        return "esp"
    return None


def enrich_midi_from_alsa(ports: list[MidiPortInfo], amidi_text: str, cards_text: str) -> None:
    """Fill ``alsa_hw``/``rawmidi`` on rtmidi-found ports from ALSA listings.

    ``amidi -l`` gives the exact ``hw:`` spec; when amidi is absent the card
    number from /proc/asound/cards with device 0 is the fallback
    (hil_speedtest.py stm_midi_card).
    """
    by_role_hw: dict[str, list[str]] = {"esp": [], "stm": []}
    for hw, name in parse_amidi_list(amidi_text):
        role = _midi_role(name)
        if role:
            by_role_hw[role].append(hw)
    if not by_role_hw["esp"] and not by_role_hw["stm"]:
        for card, card_id, long_name in parse_asound_cards(cards_text):
            role = "stm" if card_id == "MIDISerial" else _midi_role(long_name)
            if role:
                by_role_hw[role].append(f"hw:{card},0,0")
    used: dict[str, int] = {"esp": 0, "stm": 0}
    for p in ports:
        if p.alsa_hw is not None:
            continue
        role = _midi_role(p.name)
        if role is None:
            continue
        candidates = by_role_hw[role]
        i = used[role]
        if i < len(candidates):
            p.alsa_hw = candidates[i]
            p.rawmidi = rawmidi_node(candidates[i])
            used[role] = i + 1


def midi_ports_from_amidi(amidi_text: str) -> list[MidiPortInfo]:
    """CrossPad MIDI ports from ``amidi -l`` alone (no rtmidi installed)."""
    out: list[MidiPortInfo] = []
    for hw, name in parse_amidi_list(amidi_text):
        if _midi_role(name) is None:
            continue
        out.append(MidiPortInfo(name=name, rtmidi_out=None, rtmidi_in=None,
                                alsa_hw=hw, rawmidi=rawmidi_node(hw)))
    return out


def audio_cards_from_arecord(arecord_text: str) -> list[AudioCardInfo]:
    """CrossPad UAC2 cards from ``arecord -l`` alone (no sounddevice installed)."""
    return [
        AudioCardInfo(name=human, sounddevice_index=None, alsa_id=alsa_id)
        for _card, alsa_id, human in parse_arecord_list(arecord_text)
        if is_uac2_name(human)
    ]


# ---------------------------------------------------------------- default backends


def _run(argv: list[str], timeout: float = _SUBPROCESS_TIMEOUT_S) -> str:
    """stdout of ``argv`` under LC_ALL=C, or "" when the tool is missing, fails or times out.

    The locale is not optional: `arecord -l` is translated, so on a Polish host it
    prints "karta 4: Audio [Crosspad Audio]" and the card regex matches nothing —
    the UAC2 endpoint then never appears, `usb_mode` never becomes `audio`, and every
    capture on that host times out waiting for a device that is already enumerated.
    """
    try:
        return subprocess.run(argv, capture_output=True, text=True, timeout=timeout,
                              env={**os.environ, "LC_ALL": "C"}).stdout
    except (OSError, subprocess.SubprocessError, ValueError):
        return ""


def _read_text(path: str) -> str:
    try:
        with open(path, encoding="utf-8", errors="replace") as fh:
            return fh.read()
    except OSError:
        return ""


def _read_bcd(usb_device_path: str | None) -> int | None:
    if not usb_device_path:
        return None
    try:
        return int(_read_text(os.path.join(usb_device_path, "bcdDevice")).strip(), 16)
    except ValueError:
        return None


# Windows keeps bcdDevice in the device's hardware IDs ("USB\\VID_303A&PID_3456&REV_0521"),
# not anywhere pyserial reads. The instance ID of the USB device — the composite
# parent of a CDC interface — ends in the serial number pyserial reports.
_WIN_DEVICE_ID_RE = re.compile(r"^USB\\VID_([0-9A-F]{4})&PID_([0-9A-F]{4})\\(.+)$", re.IGNORECASE)
_WIN_REV_RE = re.compile(r"&REV_([0-9A-F]{4})", re.IGNORECASE)


def parse_windows_usb_ids(text: str) -> dict[tuple[int, int, str], int]:
    """``DeviceID|HardwareID,HardwareID…`` lines -> {(vid, pid, SERIAL): bcdDevice}."""
    out: dict[tuple[int, int, str], int] = {}
    for line in text.splitlines():
        device_id, _, hardware_ids = line.strip().partition("|")
        m = _WIN_DEVICE_ID_RE.match(device_id)
        rev = _WIN_REV_RE.search(hardware_ids)
        if m and rev:
            out[(int(m.group(1), 16), int(m.group(2), 16),
                 m.group(3).upper())] = int(rev.group(1), 16)
    return out


def _windows_bcds() -> dict[tuple[int, int, str], int]:
    if sys.platform != "win32":
        return {}
    return parse_windows_usb_ids(_run([
        "powershell", "-NoProfile", "-Command",
        "Get-CimInstance Win32_PnPEntity -Filter \"DeviceID LIKE 'USB%'\" | "
        "ForEach-Object { $_.DeviceID + '|' + ($_.HardwareID -join ',') }"]))


def _pyserial_list() -> list[SerialPortInfo]:
    try:
        from serial.tools import list_ports
    except ImportError:
        return []
    out: list[SerialPortInfo] = []
    win_bcds: dict[tuple[int, int, str], int] | None = None
    for p in list_ports.comports():
        bcd = _read_bcd(getattr(p, "usb_device_path", None))
        if bcd is None and sys.platform == "win32" and p.vid is not None:
            if win_bcds is None:
                win_bcds = _windows_bcds()        # one PowerShell call for all ports
            bcd = win_bcds.get((p.vid, p.pid, (p.serial_number or "").upper()))
        out.append(SerialPortInfo(
            path=p.device,
            vid=p.vid if p.vid is not None else 0,
            pid=p.pid if p.pid is not None else 0,
            serial=p.serial_number,
            product=p.product,
            location=p.location,
            bcd=bcd,
        ))
    return out


# ── macOS: ask a child, because this process is told the truth only once ─────
#
# CoreMIDI and PortAudio both snapshot the device list the first time a process
# touches them and never refresh it. A long-lived process therefore cannot see a
# board that re-enumerated: `usb-mode set audio` sat through its whole 20 s
# timeout while a fresh `crosspad-hil devices` saw the new profile in 4 s -- the
# switch had worked, the waiter was blind. Every scenario that enters the audio
# profile failed that way, which is what made the profile look broken.
#
# Linux does not need this (ALSA is read from /proc and the text listings are
# re-run anyway), so the cost is paid only where the problem is.
_PROBE_TIMEOUT_S = 10.0

_MIDI_PROBE = """
import json, rtmidi
mo, mi = rtmidi.MidiOut(), rtmidi.MidiIn()
try:
    print(json.dumps({"out": list(mo.get_ports()), "in": list(mi.get_ports())}))
finally:
    mo.delete(); mi.delete()
"""

_AUDIO_PROBE = """
import json, sounddevice
print(json.dumps([{"name": str(d["name"]), "in": int(d.get("max_input_channels", 0))}
                  for d in sounddevice.query_devices()]))
"""


def _probe_json(code: str):
    """Run a snippet in a fresh interpreter and parse its JSON, or None."""
    try:
        p = subprocess.run([sys.executable, "-c", code], capture_output=True,
                           text=True, timeout=_PROBE_TIMEOUT_S, check=False)
    except Exception:  # noqa: BLE001 - a probe must never stop discovery
        return None
    if p.returncode != 0:
        return None
    try:
        return json.loads(p.stdout)
    except ValueError:
        return None


def _midi_names_fresh() -> tuple[list[str], list[str]] | None:
    got = _probe_json(_MIDI_PROBE)
    if not isinstance(got, dict):
        return None
    return list(got.get("out") or []), list(got.get("in") or [])


def _rtmidi_list() -> list[MidiPortInfo]:
    amidi_text = _run(["amidi", "-l"]) if sys.platform.startswith("linux") else ""
    cards_text = _read_text("/proc/asound/cards") if sys.platform.startswith("linux") else ""
    try:
        import rtmidi
    except ImportError:
        return midi_ports_from_amidi(amidi_text) if amidi_text else []
    # Both handles must be closed explicitly: constructing one registers an ALSA
    # sequencer client, and python-rtmidi only unregisters it on delete(). Left
    # to the garbage collector they accumulate, and at ~64 clients the whole
    # machine stops opening sequencers — the PC simulator dies at startup with
    # "snd_seq_hw_open ... Cannot allocate memory", nowhere near the leak.
    fresh = _midi_names_fresh() if sys.platform == "darwin" else None
    mo = mi = None
    try:
        if fresh is not None:
            out_names, in_names = fresh
        else:
            mo, mi = rtmidi.MidiOut(), rtmidi.MidiIn()
            out_names = list(mo.get_ports())
            in_names = list(mi.get_ports())
    except Exception:  # noqa: BLE001 - a broken backend must not stop discovery
        return midi_ports_from_amidi(amidi_text) if amidi_text else []
    finally:
        for h in (mo, mi):
            try:
                if h is not None:
                    h.delete()
            except Exception:  # noqa: BLE001 - teardown must not mask discovery
                pass
    ports: list[MidiPortInfo] = []
    seen_in: set[int] = set()
    for oi, name in enumerate(out_names):
        if _midi_role(name) is None:
            continue
        # hil_sampler_record.py Midi.open: OUT and IN indices found independently
        ii = next((n for n, s in enumerate(in_names) if s == name and n not in seen_in), None)
        if ii is None:
            role = _midi_role(name)
            ii = next((n for n, s in enumerate(in_names)
                       if _midi_role(s) == role and n not in seen_in), None)
        if ii is not None:
            seen_in.add(ii)
        ports.append(MidiPortInfo(name=name, rtmidi_out=oi, rtmidi_in=ii,
                                  alsa_hw=None, rawmidi=None))
    for ii, name in enumerate(in_names):
        if ii in seen_in or _midi_role(name) is None:
            continue
        ports.append(MidiPortInfo(name=name, rtmidi_out=None, rtmidi_in=ii,
                                  alsa_hw=None, rawmidi=None))
    if amidi_text or cards_text:
        enrich_midi_from_alsa(ports, amidi_text, cards_text)
    return ports


def _sounddevice_list() -> list[AudioCardInfo]:
    arecord_text = _run(["arecord", "-l"]) if sys.platform.startswith("linux") else ""
    alsa_ids = [alsa_id for _c, alsa_id, human in parse_arecord_list(arecord_text)
                if is_uac2_name(human)]
    try:
        import sounddevice
    except Exception:  # noqa: BLE001 - a missing PortAudio raises OSError, not ImportError
        return audio_cards_from_arecord(arecord_text)
    infos: list = []
    if sys.platform == "darwin":
        got = _probe_json(_AUDIO_PROBE)
        # A child that could not answer is not proof of no cards; fall through.
        infos = [{"name": d.get("name", ""), "max_input_channels": d.get("in", 0)}
                 for d in got] if isinstance(got, list) else []
    if not infos:
        try:
            infos = list(sounddevice.query_devices())
        except Exception:  # noqa: BLE001 - PortAudio can fail to init without a sound server
            return audio_cards_from_arecord(arecord_text)
    cards: list[AudioCardInfo] = []
    for idx, d in enumerate(infos):
        # hil_sampler_record.py find_capture: capture endpoint, at least stereo
        if not is_uac2_name(str(d["name"])) or int(d.get("max_input_channels", 0)) < 2:
            continue
        alsa_id = alsa_ids[len(cards)] if len(cards) < len(alsa_ids) else None
        cards.append(AudioCardInfo(name=str(d["name"]), sounddevice_index=idx, alsa_id=alsa_id))
    if not cards:
        return audio_cards_from_arecord(arecord_text)
    return cards


_MAC_RE = re.compile(r"^[0-9A-Fa-f]{2}(:[0-9A-Fa-f]{2}){5}$")


def is_mac(serial: str | None) -> bool:
    """An ESP's USB serial is its MAC in both its own CDC and the ROM's JTAG port."""
    return bool(serial) and bool(_MAC_RE.match(serial or ""))


def _mac_store_path() -> str:
    base = os.environ.get("XDG_STATE_HOME") or os.path.join(os.path.expanduser("~"), ".local",
                                                            "state")
    return os.path.join(base, "crosspad-hil", "crosspad_macs.json")


def _load_known_macs() -> set[str]:
    try:
        with open(_mac_store_path(), encoding="utf-8") as f:
            return {m.upper() for m in json.load(f) if is_mac(m)}
    except (OSError, ValueError, TypeError):
        return set()


def _remember_macs(macs: set[str]) -> None:
    merged = _load_known_macs() | {m.upper() for m in macs}
    path = _mac_store_path()
    try:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        tmp = path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(sorted(merged), f)
        os.replace(tmp, path)
    except OSError:
        pass   # recognition by MAC is a convenience; discovery must not fail on it


def default_backends() -> Backends:
    """pyserial + rtmidi + sounddevice, each optional, ALSA text enrichment on Linux."""
    return Backends(list_serial=_pyserial_list, list_midi=_rtmidi_list,
                    list_audio=_sounddevice_list,
                    known_macs=_load_known_macs, remember_macs=_remember_macs)


# ---------------------------------------------------------------- grouping


def location_key(location: str | None) -> str:
    """USB topology without the interface suffix: ``1-3.2:1.0`` → ``1-3.2``."""
    if not location:
        return ""
    return location.split(":", 1)[0]


def common_prefix_len(a: str, b: str) -> int:
    n = 0
    for x, y in zip(a, b, strict=False):
        if x != y:
            break
        n += 1
    return n


@dataclass
class _Group:
    esp: SerialPortInfo | None = None       # cdc or bootloader port
    stm: SerialPortInfo | None = None
    esp_midi: MidiPortInfo | None = None
    stm_midi: MidiPortInfo | None = None
    uac2: AudioCardInfo | None = None
    fallback_key: str = field(default="")


def _pair(esp_ports: list[SerialPortInfo], stm_ports: list[SerialPortInfo]) -> list[_Group]:
    groups: list[_Group] = []
    if len(esp_ports) == 1 and len(stm_ports) == 1:
        return [_Group(esp=esp_ports[0], stm=stm_ports[0])]
    scored: list[tuple[int, int, int]] = []
    for ei, e in enumerate(esp_ports):
        for si, s in enumerate(stm_ports):
            ek, sk = location_key(e.location), location_key(s.location)
            if not ek or not sk:
                continue
            n = common_prefix_len(ek, sk)
            if n > 0:
                scored.append((n, ei, si))
    scored.sort(key=lambda t: (-t[0], t[1], t[2]))
    used_e: set[int] = set()
    used_s: set[int] = set()
    for _n, ei, si in scored:
        if ei in used_e or si in used_s:
            continue
        used_e.add(ei)
        used_s.add(si)
        groups.append(_Group(esp=esp_ports[ei], stm=stm_ports[si]))
    for ei, e in enumerate(esp_ports):
        if ei not in used_e:
            groups.append(_Group(esp=e))
    for si, s in enumerate(stm_ports):
        if si not in used_s:
            groups.append(_Group(stm=s))
    return groups


def _infer_mode(ports: Ports) -> UsbMode:
    if ports.bootloader is not None:
        return UsbMode.BOOTLOADER
    if ports.cdc is not None:
        return UsbMode.DEFAULT
    if ports.uac2 is not None:
        return UsbMode.AUDIO
    return UsbMode.UNKNOWN


def discover(backends: Backends | None = None) -> list[Device]:
    """One pass over the backends → ``Device`` records, ESP-side devices first."""
    b = backends if backends is not None else default_backends()
    serial_ports = b.list_serial()
    midi_ports = b.list_midi()
    audio_cards = b.list_audio()

    known = {m.upper() for m in b.known_macs()}
    seen = {p.serial.upper() for p in serial_ports if is_esp_cdc(p) and is_mac(p.serial)}
    if seen - known:
        b.remember_macs(seen)
    known |= seen

    esp_serial = [p for p in serial_ports if is_esp_cdc(p) or is_esp_bootloader(p)]
    stm_serial = [p for p in serial_ports if is_stm_console(p)]
    esp_midis = [m for m in midi_ports if is_esp_midi_name(m.name)]
    stm_midis = [m for m in midi_ports if is_stm_midi_name(m.name)]
    uac2s = [a for a in audio_cards if is_uac2_name(a.name)]

    # The ESPs known to be CrossPads pair with the STM bridges first; only a
    # bridge left over may vouch for an ESP32 in ROM whose MAC was never seen
    # (a board being recovered). Any other ESP32 in ROM is some other
    # project's board -- and must not take a CrossPad's bridge from it.
    def _known(p: SerialPortInfo) -> bool:
        return is_esp_cdc(p) or (p.serial or "").upper() in known
    groups = _pair([p for p in esp_serial if _known(p)], stm_serial)
    spare = [g for g in groups if g.esp is None]
    # A CrossPad in its audio profile has no serial port, only its MIDI port
    # and card: as many bridges as there are of those already have their ESP
    # (in enumeration order, as the ports are attached below) and are not
    # spare. Otherwise another project's ESP32 in ROM took such a bridge and
    # the board read as bootloader.
    spare = spare[max(len(esp_midis), len(uac2s)):]
    unknown = [p for p in esp_serial if not _known(p)]
    if spare and unknown:
        vouched = [g for g in _pair(unknown, [g.stm for g in spare])
                   if g.esp is not None and g.stm is not None]
        taken = {id(g.stm) for g in vouched}
        groups = [g for g in groups if not (g.esp is None and id(g.stm) in taken)] + vouched
    esp_groups = [g for g in groups if g.esp is not None]
    stm_only = [g for g in groups if g.esp is None]

    # MIDI ports / UAC2 cards carry no USB serial: attach in enumeration order.
    if esp_groups:
        targets = esp_groups
    elif len(stm_only) == 1:
        targets = stm_only
    else:
        targets = []
    for g, m in zip(targets, esp_midis, strict=False):
        g.esp_midi = m
    for g, a in zip(targets, uac2s, strict=False):
        g.uac2 = a
    for g, m in zip([g for g in groups if g.stm is not None], stm_midis, strict=False):
        g.stm_midi = m

    if not groups and (esp_midis or uac2s):
        # audio profile with the STM bridge absent (or Windows without the VCP driver)
        key = esp_midis[0].name if esp_midis else uac2s[0].name
        groups.append(_Group(esp_midi=esp_midis[0] if esp_midis else None,
                             uac2=uac2s[0] if uac2s else None, fallback_key=key))

    devices: list[Device] = []
    for g in groups:
        ports = Ports(
            cdc=g.esp if g.esp is not None and is_esp_cdc(g.esp) else None,
            bootloader=g.esp if g.esp is not None and is_esp_bootloader(g.esp) else None,
            console=g.stm,
            esp_midi=g.esp_midi,
            stm_midi=g.stm_midi,
            uac2=g.uac2,
        )
        # Identity comes from the STM bridge when there is one: it survives every
        # ESP reset, while the ESP's own node disappears and comes back (and the
        # stock ESP serial is the placeholder "123456" on every board). Falling
        # back to the ESP means the same hardware answers to two different ids
        # depending on whether it happens to be mid-reset.
        if g.stm is not None:
            id_serial, id_fallback = g.stm.serial, g.stm.path
        elif g.esp is not None:
            id_serial, id_fallback = g.esp.serial, g.esp.path
        else:
            id_serial, id_fallback = None, g.fallback_key
        # Reported serial still describes the board itself (the ESP when present);
        # only the identity key prefers the bridge.
        serial = g.esp.serial if g.esp is not None else id_serial
        pcb = pcb_from_bcd(g.stm.bcd) if g.stm is not None else None
        fw_rev = fw_rev_from_bcd(g.esp.bcd) if g.esp is not None and is_esp_cdc(g.esp) else None
        devices.append(Device(id=device_id_for(id_serial, id_fallback), serial=serial,
                              usb_mode=_infer_mode(ports), ports=ports,
                              board_rev=board_rev_from_pcb(pcb), pcb=pcb, fw_rev=fw_rev))
    devices.sort(key=lambda d: (0 if d.has_esp_side() else 1))
    return devices


# ---------------------------------------------------------------- selection


def select(devices: list[Device], device: str | None = None) -> Device:
    """The one board, or the one named by id / port path; else NO_DEVICE / AMBIGUOUS_DEVICE."""
    if device is None:
        with_esp = [d for d in devices if d.has_esp_side()]
        if len(with_esp) == 1:
            return with_esp[0]
        if not with_esp:
            raise HilError(
                NO_DEVICE,
                "no CrossPad found; is it in bootloader/DFU?",
                hint=(
                    "check the cable, `crosspad-hil devices`, and udev permissions; "
                    + (f"{len(devices)} STM console(s) seen without an ESP side"
                       if devices else "nothing with VID 0x303A or 0x0483 enumerated")
                ),
                candidates=[d.to_dict() for d in devices],
            )
        raise HilError(
            AMBIGUOUS_DEVICE,
            f"{len(with_esp)} CrossPads connected; pick one",
            hint="pass device=<id> (or a port path): "
            + ", ".join(f"{d.id} [{d.usb_mode.value}]" for d in with_esp),
            candidates=[d.to_dict() for d in with_esp],
        )
    for d in devices:
        if d.id == device:
            return d
    for d in devices:
        if device in d.ports.paths().values():
            return d
    known = [f"{path} ({role})" for d in devices for role, path in d.ports.paths().items()]
    raise HilError(
        NO_DEVICE,
        f"no CrossPad matches {device!r}",
        hint=("known ports: " + ", ".join(known)) if known else "no CrossPad ports enumerated",
        device=device,
        candidates=[d.to_dict() for d in devices],
    )
