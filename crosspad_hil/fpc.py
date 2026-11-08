"""FL Studio FPC presets (`.fst`) <-> CrossPad kits, host side, stdlib only.

# What a `.fst` is

An `.fst` is *not* an FPC file format — it is FL Studio's generic "channel
state" container, the same `FLhd`/`FLdt` chunk pair a `.flp` project uses,
holding one channel instead of a project:

    "FLhd"  u32 len=6   u16 format=0x30  u16 nChannels  u16 ppq=96
    "FLdt"  u32 len     <event stream>

The event stream is the ordinary FLP event encoding: one id byte, then a
payload whose size the id decides — ids 0..63 carry one byte, 64..127 a u16,
128..191 a u32, and 192..255 a varint length followed by that many bytes.
Every FPC preset in FL Studio 2026's factory set is the same ten events:

    199 version text, 159 build number, 28, 37,
    201 plugin name ("FPC", UTF-16LE),
    212 plugin wrapper (52 bytes: window geometry and flags),
    203 preset name (UTF-16LE),
    155, 128,
    213 plugin state  <- everything below lives in here

So the container is FL's, and FPC's own format is only what event 213 holds.

# Event 213: FPC's state

Four magic bytes (`03 e4 84 1e`) followed by a flat stream of records:

    u32 id   u64 size   <size bytes>

Records nest — a record's payload is often another stream of the same shape.
The top level is exactly 32 `id 2` records (pad A1..A16 then B1..B16) followed
by three globals (`id 1` a demo-loop MIDI path, `id 3`, `id 4`). A pad is:

    100  64 bytes, sixteen i32 — the fields this module names in `FpcPad`
    101  pad name, UTF-16LE, *not* NUL-terminated (the events around the
         plugin state are; the records inside it are not)
    102  32 bytes, opaque (first i32 is 1 for a single-layer pad, 3 when the
         pad has velocity layers)
    105  one per sample layer, in velocity order; contains
           300  sample path, UTF-16LE ("" when the pad is empty)
           301  28 bytes, seven i32: level, pan, vel_low, vel_high, enabled, _, _
           302  empty
    103/104 pairs, then 106 (empty) — envelopes and the pad's mod matrix,
         opaque here

Of the sixteen i32 in record 100, seven were identified from the factory set
(see `_P_*` below): level, pan, note, cut group, cut-by group, colour and icon.
The note fields decode exactly to the General MIDI drum map (pad A2 "Kick Drum"
is note 36, A3 "Closed Hat" is 42), which is what pins the layout down.

Colour is stored the way FL stores every colour, `0x00BBGGRR`, while a CrossPad
kit stores `0xRRGGBB`; `_swap_rgb` is that one byte swap in both directions.
Level (0..128) and pan (-64..63) land on CrossPad's own 0..127 volume and
-64..63 pan with no scaling, so both survive a round trip unchanged.

# What this module keeps opaque

Everything not named above is carried through as raw bytes (`FpcPad.flags`,
`FpcPad.tail`, `FpcPreset.globals_`, `FpcPreset.events`), so parsing a factory
preset and writing it back reproduces the file byte for byte — the check
`tests/test_fpc.py` runs over every preset it can find. A preset built from a
CrossPad kit fills those blobs from constants captured from FL's own
`Empty.fst`.
"""
from __future__ import annotations

import getpass
import json
import re
import shutil
import struct
import subprocess
import tempfile
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path
from typing import Any

from crosspad_hil.errors import BAD_ARGS, HilError
from crosspad_hil.kits import MAX_PADS, read_wav_stereo16, write_stereo16_wav

# --------------------------------------------------------------------------- container

FLHD = b"FLhd"
FLDT = b"FLdt"
FLHD_PAYLOAD = bytes.fromhex("300001006000")     # format 0x30, 1 channel, 96 ppq

EV_VERSION = 199
EV_PLUGIN_NAME = 201
EV_PLUGIN_WRAPPER = 212
EV_PRESET_NAME = 203
EV_PLUGIN_PARAMS = 213

# The wrapper events around 213, verbatim from FL Studio 2026's `Empty.fst`.
# Event 212 is the plugin window's geometry and flags and 128 a per-preset
# cookie: FL rewrites both when it saves, and neither changes what is loaded.
TEMPLATE_EVENTS: tuple[tuple[int, bytes], ...] = (
    (EV_VERSION, b"20.0.4.634\x00"),
    (159, bytes.fromhex("7a020000")),
    (28, bytes.fromhex("03")),
    (37, bytes.fromhex("01")),
    (EV_PLUGIN_NAME, "FPC\0".encode("utf-16-le")),   # events NUL-terminate, records do not
    (EV_PLUGIN_WRAPPER, bytes.fromhex(
        "00000000000000000200000000000000510100000000000000000000000000000000"
        "00004c000000d0000000170300006f010000")),
    (EV_PRESET_NAME, b""),                       # filled in with the preset name
    (155, bytes.fromhex("00000000")),
    (128, bytes.fromhex("48515600")),
)

# --------------------------------------------------------------------------- FPC records

FPC_MAGIC = bytes.fromhex("03e4841e")

REC_GLOBAL_DEMO_LOOP = 1
REC_PAD = 2
REC_PAD_PARAMS = 100
REC_PAD_NAME = 101
REC_PAD_FLAGS = 102
REC_LAYER = 105
REC_LAYER_PATH = 300
REC_LAYER_PARAMS = 301

_HEADER = struct.Struct("<IQ")                   # u32 id, u64 size

# Indices into record 100's sixteen i32.
_P_LEVEL = 0
_P_PAN = 1
_P_NOTE = 3
_P_CUT = 5                                       # choke group this pad belongs to, -1 = none
_P_CUT_BY = 6                                    # choke group that silences this pad
_P_COLOR = 8                                     # 0x00BBGGRR
_P_ICON = 9
_PAD_PARAM_COUNT = 16

# Indices into record 301's seven i32.
_L_LEVEL = 0
_L_PAN = 1
_L_VEL_LOW = 2
_L_VEL_HIGH = 3
_L_ENABLED = 4
_LAYER_PARAM_COUNT = 7

FPC_BANK_PADS = 16                               # pads per bank; a preset holds banks A and B
FPC_PAD_COUNT = 32
FPC_LEVEL_DEFAULT = 99                           # what FL's own Empty.fst puts on every pad
MAX_VELOCITY = 127

# First note DAW Control sends for pad 0 (crosspad-dawcontrol, DawProtocol.hpp).
DAW_BASE_NOTE = 36

# Default note map, read out of FL's `Empty.fst`: bank A is the General MIDI
# drum map in FPC's pad order, bank B is a chromatic run from middle C.
FPC_DEFAULT_NOTES: tuple[int, ...] = (
    37, 36, 42, 54, 40, 38, 46, 44, 48, 47, 45, 43, 49, 55, 51, 53,
    60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75,
)

# Opaque blobs from `Empty.fst`: a pad's flags record, the envelope/mod-matrix
# records that follow its layers, and the three global records after the pads.
DEFAULT_PAD_FLAGS = bytes.fromhex("01" + "00" * 31)
DEFAULT_PAD_TAIL = bytes.fromhex(
    "6700000008000000000000000000000000000000680000009000000000000000010000000300000004000000"
    "000000000000000000000000000000000000000000000000000000000000e83f000000000000f03fcdcc4c3e"
    "00000001000000000000f83f000000000000e03fcdcc4c3e00000001000000000000d03f0000000000000000"
    "cdcc4c3e00000001000000000000000001000000ffffffff0200000080000000800000000000000080000000"
    "6700000008000000000000000100000000000000680000004800000000000000010000000300000001000000"
    "0000000000000000000000000000e03fcdcc4c3e000000000000000000000000ffffffffffffffffffffffff"
    "800000008000000000000000800000006a0000000000000000000000")
DEFAULT_LAYER_TAIL = bytes.fromhex("2e0100000000000000000000")     # record 302, always empty
DEFAULT_GLOBALS: tuple[tuple[int, bytes], ...] = (
    (REC_GLOBAL_DEMO_LOOP, "World Loops\\Polka\\fpc_polka_10.mid".encode("utf-16-le")),
    (3, bytes.fromhex("01")),
    (4, bytes(32)),
)

# --------------------------------------------------------------------------- CrossPad side

KIT_MAX_VOLUME = 127
KIT_PAN_MIN, KIT_PAN_MAX = -64, 63
FL_PATH_VARS = ("%FLStudioFactoryData%",)

# FL ships its factory packs as Ogg Vorbis inside a RIFF WAVE wrapper (fmt tag
# 0x674F), which stdlib `wave` refuses and no dependency of this package can
# decode. ffmpeg, when the bench has it, is the way out; without it the pad is
# reported rather than silently dropped.
WAV_FMT_VORBIS = 0x674F
FFMPEG_TIMEOUT_S = 60.0


def _wav_format_tag(path: Path) -> int | None:
    """The `fmt ` chunk's format tag, or None when the file is not a RIFF WAVE."""
    with path.open("rb") as fh:
        head = fh.read(40)
    if head[:4] != b"RIFF" or head[8:12] != b"WAVE" or head[12:16] != b"fmt ":
        return None
    return struct.unpack_from("<H", head, 20)[0]


def _transcode_with_ffmpeg(src: Path, rate: int) -> list[tuple[int, int]] | None:
    """`src` -> stereo int16 frames via ffmpeg, or None when ffmpeg is not installed."""
    if shutil.which("ffmpeg") is None:
        return None
    with tempfile.TemporaryDirectory() as tmp:
        dst = Path(tmp) / "decoded.wav"
        try:
            proc = subprocess.run(
                ["ffmpeg", "-nostdin", "-loglevel", "error", "-y", "-i", str(src),
                 "-acodec", "pcm_s16le", "-ar", str(rate), "-ac", "2", str(dst)],
                capture_output=True, timeout=FFMPEG_TIMEOUT_S, check=False)
        except subprocess.TimeoutExpired as exc:
            raise HilError(BAD_ARGS, f"ffmpeg timed out decoding {src.name}",
                           hint=f"gave up after {FFMPEG_TIMEOUT_S:.0f}s") from exc
        if proc.returncode != 0 or not dst.is_file():
            raise HilError(BAD_ARGS, f"ffmpeg could not decode {src.name}",
                           hint=proc.stderr.decode("utf-8", "replace").strip()[:200])
        return read_wav_stereo16(dst, rate)


def load_sample_frames(src: Path, rate: int) -> list[tuple[int, int]]:
    """One sample file -> stereo int16 frames at `rate`, ffmpeg only if it has to."""
    try:
        return read_wav_stereo16(src, rate)
    except HilError:
        if _wav_format_tag(src) != WAV_FMT_VORBIS:
            raise
        frames = _transcode_with_ffmpeg(src, rate)
        if frames is None:
            raise HilError(BAD_ARGS,
                           f"{src.name} is Ogg Vorbis in a WAV wrapper, which needs ffmpeg",
                           hint="install ffmpeg, or export the pad from FL as PCM first"
                           ) from None
        return frames


def _swap_rgb(value: int) -> int:
    """0x00BBGGRR <-> 0x00RRGGBB — FL's colour order against CrossPad's."""
    return ((value & 0xFF) << 16) | (value & 0xFF00) | ((value >> 16) & 0xFF)


def _utf16(text: str) -> bytes:
    """UTF-16LE as an FPC *record* stores it: no terminator, empty string = no bytes."""
    return text.encode("utf-16-le")


def _utf16z(text: str) -> bytes:
    """UTF-16LE as an FLP *event* stores it: NUL-terminated."""
    return (text + "\0").encode("utf-16-le")


def _from_utf16(raw: bytes) -> str:
    return raw.decode("utf-16-le", "replace").rstrip("\0")


# --------------------------------------------------------------------------- model

@dataclass
class FpcLayer:
    """One velocity layer of a pad: a sample path plus its velocity window."""

    sample_path: str = ""
    params: list[int] = field(default_factory=lambda: [MAX_VELOCITY, 0, 0, MAX_VELOCITY, 1, 0, -1])
    tail: bytes = DEFAULT_LAYER_TAIL     # the records after 300/301 (302, always empty)

    @property
    def vel_low(self) -> int:
        return self.params[_L_VEL_LOW]

    @property
    def vel_high(self) -> int:
        return self.params[_L_VEL_HIGH]

    @property
    def level(self) -> int:
        return self.params[_L_LEVEL]

    @property
    def enabled(self) -> bool:
        return bool(self.params[_L_ENABLED])


@dataclass
class FpcPad:
    """One FPC pad. `params` is record 100 whole; the properties name its fields."""

    index: int
    name: str = ""
    params: list[int] = field(default_factory=lambda: [0] * _PAD_PARAM_COUNT)
    flags: bytes = DEFAULT_PAD_FLAGS
    layers: list[FpcLayer] = field(default_factory=list)
    tail: bytes = DEFAULT_PAD_TAIL

    @property
    def note(self) -> int:
        return self.params[_P_NOTE]

    @property
    def level(self) -> int:
        return self.params[_P_LEVEL]

    @property
    def pan(self) -> int:
        return self.params[_P_PAN]

    @property
    def color(self) -> int:
        """Pad colour as 0xRRGGBB, the order a CrossPad kit.json uses."""
        return _swap_rgb(self.params[_P_COLOR])

    @property
    def icon(self) -> int:
        return self.params[_P_ICON]

    @property
    def choke_group(self) -> int:
        """FPC's cut group, or 0 when the pad chokes nothing (FPC spells that -1)."""
        cut = self.params[_P_CUT]
        return cut if cut > 0 else 0

    @property
    def sample_path(self) -> str:
        """The path a full-strength hit plays — empty when the pad has no sample."""
        layer = self.playing_layer()
        return layer.sample_path if layer else ""

    def playing_layer(self) -> FpcLayer | None:
        """The layer a velocity-127 hit reaches, else the last one with a sample.

        A CrossPad pad holds one sample per slot with no velocity switching, so a
        multi-layer FPC pad has to collapse to one layer; the loudest is the one
        a drummer recognises the kit by.
        """
        usable = [ly for ly in self.layers if ly.sample_path and ly.enabled]
        if not usable:
            return None
        for layer in usable:
            if layer.vel_low <= MAX_VELOCITY <= layer.vel_high:
                return layer
        return usable[-1]


@dataclass
class FpcPreset:
    """A parsed `.fst`. `events` and `globals_` carry what this module does not model."""

    name: str = ""
    pads: list[FpcPad] = field(default_factory=list)
    globals_: list[tuple[int, bytes]] = field(default_factory=lambda: list(DEFAULT_GLOBALS))
    events: list[tuple[int, bytes]] = field(default_factory=lambda: list(TEMPLATE_EVENTS))
    header: bytes = FLHD_PAYLOAD

    def bank(self, index: int = 0) -> list[FpcPad]:
        """Pads of bank A (0) or B (1)."""
        start = index * FPC_BANK_PADS
        return self.pads[start:start + FPC_BANK_PADS]


# --------------------------------------------------------------------------- parsing

def _events(body: bytes) -> list[tuple[int, bytes]]:
    """The FLP event stream: id byte, then a size the id decides."""
    out: list[tuple[int, bytes]] = []
    i = 0
    while i < len(body):
        ev = body[i]
        i += 1
        if ev < 64:
            size = 1
        elif ev < 128:
            size = 2
        elif ev < 192:
            size = 4
        else:
            size = 0
            shift = 0
            while True:
                if i >= len(body):
                    raise HilError(BAD_ARGS, "truncated varint in the FLdt event stream")
                b = body[i]
                i += 1
                size |= (b & 0x7F) << shift
                shift += 7
                if not b & 0x80:
                    break
        if i + size > len(body):
            raise HilError(BAD_ARGS, f"event {ev} runs past the end of the FLdt chunk")
        out.append((ev, body[i:i + size]))
        i += size
    return out


def _encode_event(ev: int, payload: bytes) -> bytes:
    if ev < 192:
        return bytes([ev]) + payload
    size = len(payload)
    varint = bytearray()
    while True:
        b = size & 0x7F
        size >>= 7
        varint.append(b | 0x80 if size else b)
        if not size:
            break
    return bytes([ev]) + bytes(varint) + payload


def _records(blob: bytes) -> list[tuple[int, bytes]]:
    out: list[tuple[int, bytes]] = []
    i = 0
    while i + _HEADER.size <= len(blob):
        rec_id, size = _HEADER.unpack_from(blob, i)
        end = i + _HEADER.size + size
        if end > len(blob):
            raise HilError(BAD_ARGS, f"FPC record {rec_id} claims {size} bytes, past the end")
        out.append((rec_id, blob[i + _HEADER.size:end]))
        i = end
    if i != len(blob):
        raise HilError(BAD_ARGS, f"{len(blob) - i} trailing bytes after the last FPC record")
    return out


def _encode_record(rec_id: int, payload: bytes) -> bytes:
    return _HEADER.pack(rec_id, len(payload)) + payload


def _ints(payload: bytes, count: int, what: str) -> list[int]:
    if len(payload) != count * 4:
        raise HilError(BAD_ARGS, f"{what}: expected {count * 4} bytes, got {len(payload)}")
    return list(struct.unpack(f"<{count}i", payload))


def _parse_layer(payload: bytes) -> FpcLayer:
    layer = FpcLayer(params=[])
    tail = bytearray()
    for rec_id, data in _records(payload):
        if rec_id == REC_LAYER_PATH:
            layer.sample_path = _from_utf16(data)
        elif rec_id == REC_LAYER_PARAMS:
            layer.params = _ints(data, _LAYER_PARAM_COUNT, "FPC layer params")
        else:
            tail += _encode_record(rec_id, data)
    layer.tail = bytes(tail)
    return layer


def _parse_pad(index: int, payload: bytes) -> FpcPad:
    pad = FpcPad(index=index, tail=b"")
    tail = bytearray()
    for rec_id, data in _records(payload):
        if rec_id == REC_PAD_PARAMS:
            pad.params = _ints(data, _PAD_PARAM_COUNT, "FPC pad params")
        elif rec_id == REC_PAD_NAME:
            pad.name = _from_utf16(data)
        elif rec_id == REC_PAD_FLAGS:
            pad.flags = data
        elif rec_id == REC_LAYER:
            pad.layers.append(_parse_layer(data))
        else:
            tail += _encode_record(rec_id, data)
    pad.tail = bytes(tail)
    return pad


def parse_fpc_preset(path: Path | str) -> FpcPreset:
    """Read one `.fst` FPC preset. Raises `HilError(BAD_ARGS)` on anything else."""
    path = Path(path)
    data = path.read_bytes()
    if data[:4] != FLHD:
        raise HilError(BAD_ARGS, f"{path.name}: not an FL Studio .fst (no FLhd chunk)",
                       hint="FPC presets live under Data/Patches/Plugin presets/Generators/FPC")
    header_len = struct.unpack_from("<I", data, 4)[0]
    off = 8 + header_len
    if data[off:off + 4] != FLDT:
        raise HilError(BAD_ARGS, f"{path.name}: no FLdt chunk after the FLhd header")
    body_len = struct.unpack_from("<I", data, off + 4)[0]
    body = data[off + 8:off + 8 + body_len]

    events = _events(body)
    by_id = dict(events)
    plugin = by_id.get(EV_PLUGIN_NAME, b"")
    if _from_utf16(plugin) != "FPC":
        raise HilError(BAD_ARGS,
                       f"{path.name}: plugin is {_from_utf16(plugin)!r}, not FPC",
                       hint="only FPC presets carry a pad layout this can read")
    blob = by_id.get(EV_PLUGIN_PARAMS)
    if blob is None:
        raise HilError(BAD_ARGS, f"{path.name}: no plugin-state event ({EV_PLUGIN_PARAMS})")
    if blob[:4] != FPC_MAGIC:
        raise HilError(BAD_ARGS, f"{path.name}: FPC state starts with {blob[:4].hex()}, "
                                 f"expected {FPC_MAGIC.hex()}")

    preset = FpcPreset(name=_from_utf16(by_id.get(EV_PRESET_NAME, b"")) or path.stem,
                       pads=[], globals_=[], header=data[8:8 + header_len],
                       events=[(ev, b"" if ev == EV_PRESET_NAME else raw)
                               for ev, raw in events if ev != EV_PLUGIN_PARAMS])
    for rec_id, payload in _records(blob[len(FPC_MAGIC):]):
        if rec_id == REC_PAD:
            preset.pads.append(_parse_pad(len(preset.pads), payload))
        else:
            preset.globals_.append((rec_id, payload))
    return preset


# --------------------------------------------------------------------------- writing

def _encode_layer(layer: FpcLayer) -> bytes:
    return _encode_record(REC_LAYER,
                          _encode_record(REC_LAYER_PATH, _utf16(layer.sample_path)
                                         if layer.sample_path else b"")
                          + _encode_record(REC_LAYER_PARAMS,
                                           struct.pack(f"<{_LAYER_PARAM_COUNT}i", *layer.params))
                          + layer.tail)


def _encode_pad(pad: FpcPad) -> bytes:
    body = _encode_record(REC_PAD_PARAMS, struct.pack(f"<{_PAD_PARAM_COUNT}i", *pad.params))
    body += _encode_record(REC_PAD_NAME, _utf16(pad.name) if pad.name else b"")
    body += _encode_record(REC_PAD_FLAGS, pad.flags)
    for layer in pad.layers:
        body += _encode_layer(layer)
    return _encode_record(REC_PAD, body + pad.tail)


def write_fpc_preset(preset: FpcPreset, path: Path | str) -> Path:
    """Serialise `preset` back to an `.fst`. Round-trips a parsed file byte for byte."""
    blob = bytearray(FPC_MAGIC)
    for pad in preset.pads:
        blob += _encode_pad(pad)
    for rec_id, payload in preset.globals_:
        blob += _encode_record(rec_id, payload)

    body = bytearray()
    for ev, payload in preset.events:
        if ev == EV_PRESET_NAME:
            payload = _utf16z(preset.name)
        body += _encode_event(ev, payload)
    body += _encode_event(EV_PLUGIN_PARAMS, bytes(blob))

    out = bytearray(FLHD)
    out += struct.pack("<I", len(preset.header)) + preset.header
    out += FLDT + struct.pack("<I", len(body)) + bytes(body)
    path = Path(path)
    path.write_bytes(bytes(out))
    return path


# --------------------------------------------------------------------------- path resolution

def _case_insensitive(root: Path, parts: list[str]) -> Path | None:
    """Walk `parts` under `root` ignoring case — Windows paths on a Linux box."""
    here = root
    for part in parts:
        candidate = here / part
        if candidate.exists():
            here = candidate
            continue
        try:
            matches = [c for c in here.iterdir() if c.name.lower() == part.lower()]
        except OSError:
            return None
        if not matches:
            return None
        here = matches[0]
    return here


def resolve_sample_path(win_path: str, *, drive_c: Path, factory_data: Path | None = None
                        ) -> Path | None:
    """A Windows path out of a preset -> a readable file under the Wine prefix.

    Handles the two shapes the factory presets use: `%FLStudioFactoryData%\\...`
    (which points at the FL install directory, not `drive_c`) and a plain
    `C:\\...`. Returns None when nothing matches, which is how an empty pad and
    a missing sample both look to the caller.
    """
    if not win_path:
        return None
    text = win_path.replace("/", "\\")
    for var in FL_PATH_VARS:
        if text.lower().startswith(var.lower()):
            if factory_data is None:
                return None
            rest = text[len(var):].lstrip("\\")
            return _case_insensitive(factory_data, rest.split("\\"))
    match = re.match(r"^([A-Za-z]):\\(.*)$", text)
    if match:
        return _case_insensitive(drive_c, match.group(2).split("\\"))
    return _case_insensitive(drive_c, text.lstrip("\\").split("\\"))


def default_factory_data(drive_c: Path) -> Path | None:
    """`C:\\Program Files\\Image-Line\\FL Studio <year>` — what %FLStudioFactoryData% means."""
    image_line = _case_insensitive(drive_c, ["Program Files", "Image-Line"])
    if image_line is None or not image_line.is_dir():
        return None
    # "FL Studio ASIO" sits in the same directory and sorts after the install
    # itself, so the sample tree has to be what picks the winner, not the name.
    installs = sorted((d for d in image_line.iterdir()
                       if d.is_dir() and d.name.lower().startswith("fl studio")
                       and _case_insensitive(d, ["Data", "Patches"]) is not None),
                      key=lambda d: d.name)
    return installs[-1] if installs else None


def find_drive_c(start: Path) -> Path | None:
    """The `drive_c` a preset was found under, by walking up from its own path."""
    for parent in [start, *start.parents]:
        if parent.name.lower() == "drive_c":
            return parent
    return None


# --------------------------------------------------------------------------- FPC -> kit

def _kit_dirname(name: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9]+", "_", name).strip("_").upper()
    return cleaned or "KIT"


def kit_from_fpc(preset: FpcPreset, out_dir: Path | str, *, drive_c: Path,
                 factory_data: Path | None = None, name: str | None = None,
                 bank: int = 0, rate: int = 44100,
                 log: Callable[[str], None] | None = None) -> Path:
    """One FPC bank -> a CrossPad kit directory (`out_dir/<DIRNAME>`).

    Each pad's playing layer is converted to the kit's own 16-bit stereo WAV the
    way `kits.build_kit` does, so the result is what `kit push` already knows how
    to send. A pad whose sample cannot be found or decoded is left empty and
    reported through `log`; only a kit with no usable pad at all fails.

    FPC's note, colour, level and pan land on the kit's `note`/`color`/`volume`/
    `pan` unchanged (both sides use 0..127 and -64..63); the pad name is kept
    as a `name` key the firmware's
    kit parser ignores and `fpc_from_kit` reads back.
    """
    kit_name = name or preset.name
    kit_dir = Path(out_dir) / _kit_dirname(kit_name)
    samples_dir = kit_dir / "SAMPLES"
    samples_dir.mkdir(parents=True, exist_ok=True)

    pads: list[dict[str, Any]] = []
    skipped: list[str] = []
    for slot, pad in enumerate(preset.bank(bank)):
        samples: list[str] = []
        layer = pad.playing_layer()
        if layer is not None:
            src = resolve_sample_path(layer.sample_path, drive_c=drive_c,
                                      factory_data=factory_data)
            if src is None or not src.is_file():
                skipped.append(f"{layer.sample_path} (not found)")
            else:
                # One unreadable sample must not lose the other fifteen pads.
                try:
                    frames = load_sample_frames(src, rate)
                except HilError as exc:
                    skipped.append(f"{layer.sample_path} ({exc.message})")
                else:
                    sample_name = f"pad{slot:02d}.wav"
                    write_stereo16_wav(samples_dir / sample_name, frames, rate)
                    samples = [sample_name]
        pads.append({
            "name": pad.name, "note": pad.note,
            "color": pad.color, "color_pressed": pad.color,
            "choke_group": pad.choke_group, "cycle": False, "pitch": 0,
            "volume": max(0, min(KIT_MAX_VOLUME, pad.level)),
            "pan": max(KIT_PAN_MIN, min(KIT_PAN_MAX, pad.pan)),
            "samples": samples,
        })

    for note in skipped:
        if log:
            log(f"  skipped: {note}")
    if skipped and not any(p["samples"] for p in pads):
        raise HilError(BAD_ARGS,
                       f"not one of {len(skipped)} sample(s) in {kit_name!r} could be converted",
                       hint=f"first: {skipped[0]}; pass --drive-c / --fl-data so the "
                            "preset's Windows paths resolve",
                       skipped=skipped[:MAX_PADS])

    kit = {
        "name": kit_name, "author": "crosspad-hil",
        "description": [f"converted from the FPC preset {preset.name!r}"],
        "version": "1.0", "created_at": date.today().isoformat(), "pads": pads,
    }
    (kit_dir / "kit.json").write_text(json.dumps(kit))
    return kit_dir


# --------------------------------------------------------------------------- kit -> FPC

def default_win_prefix(kit_dir: Path) -> str:
    """Where a pushed kit is expected to sit inside the Wine prefix."""
    return f"C:\\users\\{getpass.getuser()}\\Documents\\CrossPad\\{Path(kit_dir).name}"


def _empty_pad(index: int) -> FpcPad:
    params = [0] * _PAD_PARAM_COUNT
    params[_P_LEVEL] = FPC_LEVEL_DEFAULT
    params[_P_NOTE] = FPC_DEFAULT_NOTES[index]
    params[_P_CUT] = params[_P_CUT_BY] = -1
    return FpcPad(index=index, params=params, layers=[FpcLayer()])


def fpc_from_kit(kit_dir: Path | str, out_fst: Path | str, *, win_prefix: str | None = None,
                 name: str | None = None) -> Path:
    """A CrossPad kit directory -> an FPC preset whose pads point at its WAVs.

    Bank A gets the kit's 16 pads, bank B stays empty. Sample paths are built as
    `<win_prefix>\\SAMPLES\\<file>`, so `win_prefix` is the kit directory as
    Windows sees it inside the Wine prefix — the WAVs have to be reachable from
    there for FL to load them.
    """
    kit_dir = Path(kit_dir)
    try:
        kit = json.loads((kit_dir / "kit.json").read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise HilError(BAD_ARGS, f"{kit_dir}: no readable kit.json ({exc})",
                       hint="point this at a kit directory built by `kit push`/build_kit"
                       ) from exc

    prefix = (win_prefix or default_win_prefix(kit_dir)).rstrip("\\")
    preset = FpcPreset(name=name or str(kit.get("name") or kit_dir.name),
                       pads=[_empty_pad(i) for i in range(FPC_PAD_COUNT)])

    for slot, pad_json in enumerate(kit.get("pads", [])[:MAX_PADS]):
        pad = preset.pads[slot]
        pad.name = str(pad_json.get("name") or "")
        # DAW Control plays pad N as note DAW_BASE_NOTE + N, so a kit without
        # explicit notes lands pad-for-pad on FPC instead of on the GM drum map.
        pad.params[_P_NOTE] = int(pad_json.get("note", DAW_BASE_NOTE + slot))
        pad.params[_P_LEVEL] = int(pad_json.get("volume", FPC_LEVEL_DEFAULT))
        pad.params[_P_PAN] = int(pad_json.get("pan", 0))
        pad.params[_P_COLOR] = _swap_rgb(int(pad_json.get("color", 0)))
        choke = int(pad_json.get("choke_group", 0))
        pad.params[_P_CUT] = pad.params[_P_CUT_BY] = choke if choke > 0 else -1
        samples = pad_json.get("samples") or []
        if samples:
            first = samples[0]
            filename = first if isinstance(first, str) else str(first.get("sample", ""))
            if filename:
                pad.layers[0].sample_path = f"{prefix}\\SAMPLES\\{filename}"

    return write_fpc_preset(preset, out_fst)
