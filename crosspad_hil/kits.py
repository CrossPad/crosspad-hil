"""Build and push a kit for the sampler from plain WAV files, stdlib only.

`build_kit` is the host half of "record 16 samples, hand them to the pads":
every WAV in a directory becomes one pad, resampled and downmixed to the
kit's own rate so a phone recording or a synth render slots in without
manual conversion. `push_kit` is the device half — the same
mkdir/push/reboot/scan dance `scenarios/_audio.ensure_kits` already does for
fixture kits, generalised so `crosspad-hil kit push` can drive it from the
CLI without importing a scenario module (see that module's docstring for why
the CDC link must be closed before `fs.push` touches the raw port).

Everything here is `wave`, `struct`, `json`, `math`, `pathlib`, `re`,
`datetime` — a bench box with no numpy still builds a kit.
"""
from __future__ import annotations

import json
import re
import struct
import time
import wave
from collections.abc import Callable, Sequence
from datetime import date
from pathlib import Path
from typing import Any

from crosspad_hil import fixtures, verbs
from crosspad_hil.cdc import CdcLink
from crosspad_hil.console import Console
from crosspad_hil.errors import BAD_ARGS, ENV, HilError
from crosspad_hil.fs import push as fs_push

# --------------------------------------------------------------------------- build_kit

MAX_PADS = 16
DEFAULT_RATE = 44100

# 16 distinct hues, one per pad — cycled if a kit somehow has fewer/more via
# a caller that does not go through build_kit's own 16-pad loop.
PAD_COLOR_PALETTE: tuple[int, ...] = (
    0xE6194B, 0x3CB44B, 0xFFE119, 0x4363D8, 0xF58231, 0x911EB4, 0x46F0F0, 0xF032E6,
    0xBCF60C, 0xFABEBE, 0x008080, 0xE6BEFF, 0x9A6324, 0xFFFAC8, 0x800000, 0xAAFFC3,
)


def _dirname(name: str) -> str:
    """`name` -> the card's directory name: upper-cased, non-alnum runs -> `_`."""
    cleaned = re.sub(r"[^A-Za-z0-9]+", "_", name).strip("_").upper()
    return cleaned or "KIT"


def _clip16(v: float) -> int:
    return max(-32768, min(32767, int(round(v))))


def _decode_pcm(raw: bytes, channels: int, sampwidth: int, filename: str) -> list[tuple[int, ...]]:
    """Raw PCM bytes -> list of per-frame tuples of int16-range ints."""
    if sampwidth == 2:
        fmt = "<" + "h" * channels
        return [tuple(vals) for vals in struct.iter_unpack(fmt, raw)]
    if sampwidth == 1:
        # WAV 8-bit PCM is unsigned; centre it and scale to int16 range.
        return [tuple((b - 128) * 256 for b in chunk)
                for chunk in zip(*([iter(raw)] * channels), strict=False)]
    if sampwidth == 3:
        step = 3 * channels
        n = len(raw) // step
        out: list[tuple[int, ...]] = []
        for i in range(n):
            base = i * step
            frame = []
            for c in range(channels):
                off = base + c * 3
                val = raw[off] | (raw[off + 1] << 8) | (raw[off + 2] << 16)
                if val & 0x800000:
                    val -= 0x1000000
                frame.append(val >> 8)      # 24-bit -> 16-bit
            out.append(tuple(frame))
        return out
    if sampwidth == 4:
        fmt = "<" + "i" * channels
        return [tuple(v >> 16 for v in vals) for vals in struct.iter_unpack(fmt, raw)]
    raise HilError(BAD_ARGS, f"{filename}: unsupported sample width ({sampwidth * 8}-bit)",
                   hint="8/16/24/32-bit PCM only")


def _resample_linear(frames: list[tuple[int, ...]], src_rate: int,
                     dst_rate: int) -> list[tuple[int, ...]]:
    """Linear-interpolate `frames` from `src_rate` to `dst_rate`. stdlib only."""
    if src_rate == dst_rate or not frames:
        return frames
    channels = len(frames[0])
    n_src = len(frames)
    n_dst = max(1, round(n_src * dst_rate / src_rate))
    ratio = src_rate / dst_rate
    out: list[tuple[int, ...]] = []
    last = n_src - 1
    for i in range(n_dst):
        pos = i * ratio
        i0 = min(int(pos), last)
        i1 = min(i0 + 1, last)
        frac = pos - i0
        f0, f1 = frames[i0], frames[i1]
        out.append(tuple(_clip16(f0[c] + (f1[c] - f0[c]) * frac) for c in range(channels)))
    return out


def read_wav_stereo16(path: Path, rate: int) -> list[tuple[int, int]]:
    """One WAV file -> stereo int16 frames at `rate`.

    Accepts 8/16/24/32-bit PCM (rejects compressed/float — `wave` itself
    refuses to open those, so the failure is caught and renamed to the
    file it came from). Mono is duplicated to stereo, >2 channels keeps the
    first two, and a rate mismatch is linearly resampled.
    """
    path = Path(path)
    try:
        with wave.open(str(path), "rb") as w:
            channels = w.getnchannels()
            sampwidth = w.getsampwidth()
            src_rate = w.getframerate()
            raw = w.readframes(w.getnframes())
    except wave.Error as exc:
        raise HilError(BAD_ARGS, f"{path.name}: not a readable PCM WAV file ({exc})",
                       hint="8/16/24/32-bit PCM only, no float/compressed WAVs") from exc
    frames = _decode_pcm(raw, channels, sampwidth, path.name)
    if channels == 1:
        frames = [(v, v) for (v,) in frames]
    elif channels > 2:
        frames = [f[:2] for f in frames]
    if src_rate != rate:
        frames = _resample_linear(frames, src_rate, rate)
    return frames


def write_stereo16_wav(path: Path, frames: Sequence[tuple[int, int]], rate: int) -> None:
    data = bytearray()
    for left, right in frames:
        data += struct.pack("<hh", left, right)
    with wave.open(str(path), "wb") as w:
        w.setnchannels(2)
        w.setsampwidth(2)
        w.setframerate(rate)
        w.writeframes(bytes(data))


def _wav_files(src_dir: Path) -> list[Path]:
    seen: set[str] = set()
    out: list[Path] = []
    for f in sorted((*src_dir.glob("*.wav"), *src_dir.glob("*.WAV")),
                    key=lambda p: p.name.lower()):
        low = f.name.lower()
        if low not in seen:      # a case-insensitive filesystem globs both patterns
            seen.add(low)
            out.append(f)
    return out


def build_kit(src_dir: Path, out_dir: Path, name: str, *, rate: int = DEFAULT_RATE) -> Path:
    """Every `*.wav`/`*.WAV` in `src_dir`, sorted by name, becomes one pad.

    More than 16 files is `BAD_ARGS`; fewer leaves the remaining pads with
    `"samples": []`. Returns the kit directory (`out_dir/<DIRNAME>`), which
    is what `kit.json`'s neighbours (`SAMPLES/*.wav`) live under — the
    directory basename is what the card gets, `name` is what `KIT_LIST`
    reports.
    """
    src_dir = Path(src_dir)
    files = _wav_files(src_dir)
    if len(files) > MAX_PADS:
        raise HilError(BAD_ARGS,
                       f"{len(files)} WAV files in {src_dir}, max {MAX_PADS} pads",
                       hint="drop files or split into more than one kit",
                       count=len(files))

    kit_dir = Path(out_dir) / _dirname(name)
    samples_dir = kit_dir / "SAMPLES"
    samples_dir.mkdir(parents=True, exist_ok=True)

    pads: list[dict[str, Any]] = []
    for pad in range(MAX_PADS):
        color = PAD_COLOR_PALETTE[pad % len(PAD_COLOR_PALETTE)]
        samples: list[str] = []
        if pad < len(files):
            frames = read_wav_stereo16(files[pad], rate)
            sample_name = f"pad{pad:02d}.wav"
            write_stereo16_wav(samples_dir / sample_name, frames, rate)
            samples = [sample_name]
        pads.append({
            "color": color, "color_pressed": color, "choke_group": 0, "cycle": False,
            "pitch": 0, "volume": 100, "pan": 0, "samples": samples,
        })

    kit = {
        "name": name, "author": "crosspad-hil",
        "description": [f"built by crosspad-hil kits.build_kit from {src_dir.name}"],
        "version": "1.0", "created_at": date.today().isoformat(), "pads": pads,
    }
    (kit_dir / "kit.json").write_text(json.dumps(kit))
    return kit_dir


# --------------------------------------------------------------------------- provisioning
#
# Shared with scenarios/_audio.py, which used to own these under private
# names — a CLI verb (`kit push`) has no business importing a scenario
# module, so they live here and `_audio.py` imports them back.

KITS_ROOT = "/sdcard/crosspad/kits"
MKDIR_EEXIST = "mkdir_17"        # FS_ERR mkdir_17: EEXIST, the directory is already there
BOOT_TIMEOUT_S = 60.0            # a cold boot with an SD scan behind it
BOOT_SETTLE_S = 1.0
SCAN_TIMEOUT_S = 60.0            # KIT_LIST answers "ERR scanning" while listKits() walks the card
SCAN_POLL_S = 1.0
PUSH_TIMEOUT_S = 300.0


def mkdir_tolerant(link: CdcLink, path: str) -> None:
    """FS_MKDIR that tolerates an existing directory (the device has no -p either)."""
    try:
        verbs.fs_mkdir(link, path)
    except HilError as exc:
        if MKDIR_EEXIST not in exc.message:
            raise


def push_kit_dir(ctx: Any, kit_dir: Path, remote_dir: str,
                 push: Callable[..., dict]) -> int:
    """Every file of one kit directory to `remote_dir`; returns the byte count.

    The CDC link must already be closed: `fs.push` takes the pipe out of newline
    framing for an announced byte count, so it opens the port itself and a
    `CdcLink` still holding it would lose the payload to the reader thread — the
    same reason `ota.py` and `fs.py` live beside `verbs.py` rather than inside it.
    """
    total = 0
    for local in fixtures.kit_files(kit_dir):
        remote = f"{remote_dir}/SAMPLES/{local.name}" if local.parent.name == "SAMPLES" \
            else f"{remote_dir}/{local.name}"
        result = push(ctx.device, local, remote, timeout_s=PUSH_TIMEOUT_S)
        total += int(result.get("bytes", 0))
    return total


def reboot_for_kit_scan(ctx: Any, console: Console) -> None:
    """Reset the board through the STM bridge and wait for the banner.

    The kit scan runs once, after the mount: a kit pushed into a live filesystem
    is not in `KIT_LIST` until the next boot. Opening the bridge VCP with
    DTR/RTS asserted resets the ESP — the trap every other scenario is written
    to avoid — and here it is exactly the mechanism wanted, so it is asked for
    deliberately through `Console.reset()` rather than fallen into.
    """
    ctx.log("  resetting the board through the STM bridge (the kit scan runs once, at boot)")
    console.reset()
    boot = console.wait_boot(timeout_s=BOOT_TIMEOUT_S, settle_s=BOOT_SETTLE_S)
    console.forget_boot_history()      # the reboot is ours; no later check may count it
    if boot.missing:
        raise HilError(ENV, "the board did not finish booting after the fixture push",
                       hint="missing boot markers: " + ", ".join(boot.missing)
                            + "; check the console log in the work dir",
                       missing=list(boot.missing))


def wait_for_kit_scan(ctx: Any, link: CdcLink) -> None:
    """Poll KIT_LIST until it stops answering "ERR scanning".

    `verbs.kit_list` turns that into TIMEOUT with its own hint; the scan grows
    the kit vector in place, so a reader during it is refused rather than served
    a short list. The boot scan starts only once the launcher is up, which is
    why this can still be running well after the banner.
    """
    deadline = time.monotonic() + SCAN_TIMEOUT_S
    last: HilError | None = None
    while True:
        try:
            verbs.kit_list(link)
            return
        except HilError as exc:
            last = exc
            if time.monotonic() >= deadline:
                raise HilError(ENV, "the kit scan never finished",
                               hint=f"KIT_LIST still failing after {SCAN_TIMEOUT_S:.0f}s "
                                    f"({exc.code}: {exc.message}); is the SD card readable?"
                               ) from last
            ctx.cancelled.wait(SCAN_POLL_S)
            ctx.check_cancelled()


def push_kit(ctx_like: Any, device: Any, link: CdcLink, kit_dir: Path, *,
            console: Console, load: bool = False,
            push: Callable[..., dict] = fs_push,
            kits_root: str = KITS_ROOT,
            load_wait_s: float = 15.0) -> dict[str, Any]:
    """Push one kit built by `build_kit` onto the card and (optionally) load it.

    Mirrors `scenarios._audio.ensure_kits`'s provisioning half for a single,
    already-built kit: mkdir the kit dir and its `SAMPLES` dir, close `link`
    (pushing needs the raw port), push every file, reboot so the kit scan
    picks it up, reopen CDC through `ctx_like.open_cdc()`, wait for the scan,
    and look the new kit up by name in `KIT_LIST` — a card's kit id is
    assigned by scan order, never by the caller.

    Returns `{name, dir, bytes, kit_id, loaded}`.
    """
    kit_name = str(json.loads((kit_dir / "kit.json").read_text())["name"])
    remote_dir = f"{kits_root}/{kit_dir.name}"
    mkdir_tolerant(link, remote_dir)
    mkdir_tolerant(link, f"{remote_dir}/SAMPLES")

    link.close()                       # fs.push needs the raw port, unframed
    pushed = 0
    try:
        for local in fixtures.kit_files(kit_dir):
            remote = f"{remote_dir}/SAMPLES/{local.name}" if local.parent.name == "SAMPLES" \
                else f"{remote_dir}/{local.name}"
            result = push(device, local, remote, timeout_s=PUSH_TIMEOUT_S)
            pushed += int(result.get("bytes", 0))
    except HilError as exc:
        raise HilError(ENV, f"pushing kit {kit_name!r} failed: {exc.message}",
                       hint=exc.hint or "the STM VCP console carries the FX log lines; "
                                        "BLE_STOP first if the fx_worker cannot start",
                       **exc.details) from exc
    ctx_like.log(f"  {pushed} bytes pushed; rebooting so the kit scan picks up {kit_name!r}")

    reboot_for_kit_scan(ctx_like, console)
    link = ctx_like.open_cdc()
    link.wait_ready()
    wait_for_kit_scan(ctx_like, link)

    listing = verbs.kit_list(link)
    match = [k for k in listing.get("kits", []) if str(k["name"]).lower() == kit_name.lower()]
    if not match:
        have = sorted(str(k["name"]) for k in listing.get("kits", []))
        raise HilError(ENV, f"pushed kit {kit_name!r} is not in KIT_LIST after reboot",
                       hint="have: " + (", ".join(have) or "none")
                            + "; the card may be full, or the kit scan rejected the JSON",
                       kits=have)
    kit_id = int(match[0]["id"])

    loaded = False
    if load:
        verbs.kit_load(link, kit_id, wait_s=load_wait_s)
        loaded = True
    return {"name": kit_name, "dir": kit_dir.name, "bytes": pushed, "kit_id": kit_id,
            "loaded": loaded}
