# Host-side tools: kits, pad mix, companion, MIDI hub, PipeWire nodes, FL template

One line per command is in
[Command-line tools § crosspad-hil](https://crosspad.app/docs/reference/cli/#crosspad-hil).
This page is how they work. `midi hub`, `audio nodes` and `fl template` are
written for Linux: they use systemd --user, the ALSA sequencer, PipeWire and a
Wine prefix.

## Kit sync

Build a kit from a directory of WAVs and push it onto the card:

```bash
crosspad-hil kit push ./my-drum-kit [--name "My Kit"] [--rate 44100] [--load]
```

Every `*.wav` in the directory becomes one pad (sorted by name, max 16),
converted to 16-bit PCM at the kit's rate; `--load` calls `KIT_LOAD` once the
post-push reboot's kit scan finds it. `crosspad_hil.kits.build_kit` does the
conversion (stdlib `wave`/`struct`/`math` only, no numpy) and can be called
directly to build a kit without touching a device.

### FL Studio FPC presets

FPC (FL's drum sampler) presets go both ways, host side only — no device is
touched unless `--push` is given:

```bash
crosspad-hil kit from-fpc "…/Generators/FPC/Electronic/808.fst" ./kits [--push] [--load]
crosspad-hil kit to-fpc ./kits/MY_KIT ./MY_KIT.fst --win-prefix 'C:\users\me\Documents\CrossPad\MY_KIT'
```

`from-fpc` reads bank A's 16 pads (`--bank B` for the other bank), resolves each
pad's sample path against the Wine prefix (`--drive-c`, guessed by walking up
from the preset; `--fl-data` for what `%FLStudioFactoryData%` means) and
converts the WAVs the way `kit push` does. Pad name, MIDI note, colour, volume
and pan carry over. `to-fpc` does the reverse: `--win-prefix` is the kit
directory as Windows sees it, because that is where FL will look for the WAVs.

**The `.fst` format.** An `.fst` is not an FPC format — it is FL's generic
channel-state container, the same `FLhd`/`FLdt` chunk pair a `.flp` uses, whose
event stream ends in event 213, the plugin's own state. FPC's state is four
magic bytes (`03 e4 84 1e`) then a tree of `u32 id / u64 size / payload`
records: 32 pad records (bank A then B), each holding a 64-byte parameter block
(level, pan, MIDI note, cut group, colour as `0x00BBGGRR`, icon), a UTF-16 name,
one record per velocity layer with the sample's Windows path, and opaque
envelope blocks. `crosspad_hil/fpc.py`'s module docstring has the full map.
Everything the module does not model is carried through verbatim, so parsing a
preset and writing it back reproduces the file byte for byte — `tests/test_fpc.py`
checks that against every FPC preset installed on the machine.

**Two things to know.** FL's own factory packs are Ogg Vorbis inside a WAV
wrapper, which stdlib `wave` cannot read: `from-fpc` uses `ffmpeg` when it is on
PATH and otherwise names the pads it had to skip. And a CrossPad pad plays one
sample, so a pad with velocity layers collapses to the layer a velocity-127 hit
would reach.

## Mastering kits: `padmix`

```bash
crosspad-hil padmix
```

How to use it, step by step: [padmix.md](padmix.md).

The Pad Mixer app on the host, for levelling many kits in one sitting: sixteen
faders on the board's current kit. A move is heard at once (`PAD_MIX pad vol`
sets the pad in the kit and in the engine, as the app's fader does) and lands
in the kit's `kit.json` once the faders have been still for 1.5 s, before
another kit is loaded and on the way out (`KIT_SAVE`, the Sampler's own save:
the whole kit is rewritten from what the board parsed).

| Faders | |
|---|---|
| ← → | pad |
| ↑ ↓ / PgUp PgDn | ±1 / ±8 (volume in the kit's units: 0..127, 100 = unity; a pitched kit 0..100) |
| space | play the pad (needs an app open on the board: on the home screen the pads open apps) |
| Enter | the kit list |
| `[` `]` | previous / next kit, saved first |
| Esc | undo: a fader move (a held key is one step), or a kit change -- the previous kit comes back |
| `s` / `q` | save now / quit (saved) |
| mouse | drag a fader, wheel ±1 on the pad under the pointer, click a pad number to play it, click the title for the kit list |

In the kit list ↑ ↓ choose, typing filters by name, Enter (or a double click)
loads, Esc clears the filter and then goes back to the faders. CDC only: the
verbs are not on the companion allowlist, so the board must be in the CDC
profile (`crosspad-hil usb-mode set default`).

## The audio profile's control plane: `companion`

MIDI + UAC2 (the USB-audio profile: the one a DAW sees, and the one a fresh
board boots into) has no CDC endpoint. The companion verb set — the
remote-control and read-only subset the board's BLE companion service allows —
rides SysEx `0x1F` instead, in both profiles:

```bash
crosspad-hil companion UI_STATE
crosspad-hil companion "APP_START Sampler"
crosspad-hil companion "PAD_PRESS 3 100" && crosspad-hil companion "PAD_RELEASE 3"
crosspad-hil companion "FS_LIST /sdcard"      # ERR unsupported_companion_command
```

The catalog (`crosspad_hil/knowledge/sysex.yaml` `companion`) lists the allowed verbs; the
reply is the CDC text, chunked by the device and joined here; `ERR ...` is
printed, not raised. Bulk work (files, OTA) still needs the CDC profile:

```bash
crosspad-hil usb-mode set default
```

**That request is guarded on the device.** With the USB serial guard on (the
default) the board shows *USB serial mode? Allow / Deny* for 30 s before it
re-enumerates, so `set default` waits 45 s and says so when it times out. A
bench board turns the guard off once, persisted, over CDC: `crosspad-hil cdc
"USB_GUARD 0"`. The prompt cannot be pressed from the host on purpose —
`companion ENC_PRESS` answers `ERR guard_prompt_active` while it is up.

## A DAW under Wine: `midi hub`

Wine lists ALSA MIDI ports once, when the DAW starts, matches incoming
events to the device it opened by their ALSA source address, and never
releases an input that vanished. A CrossPad that shows up later, flips its
USB profile or takes an OTA is invisible or "already allocated" until the
DAW restarts. The hub is the fix:

```bash
crosspad-hil midi hub install     # systemd --user unit, ports "CrossPad Hub"
crosspad-hil midi hub status
crosspad-hil midi hub run         # foreground
```

It is an ALSA sequencer client alive from login with one readable and one
writable port, both named **CrossPad Hub**. Point the DAW's MIDI input and
output at those (FL: Options → MIDI Settings, controller type CrossPad, same
port number in and out) and forget about the board: the hub attaches to the
CrossPad ESP port whenever one is present, in any plug order, and detaches
when it goes. Board → hub‑readable for the DAW; hub‑writable → board for
what the DAW sends. Writes into the hub are also mirrored to the readable
side, so a host tool that shares the DAW's script (the fl‑studio‑mcp bridge)
uses the same two ports — set its MIDI port names to `CrossPad Hub` and drop
the board relay (`SYSEX_ECHO`).

The hub is also the board's host endpoint. While a board is attached it sends
`HOST_INFO` (this computer's CPU, RAM and clock) every 2 s and `FW_LATEST`
every minute, as companion notify frames (`0x1F 04`: run, never answered) —
the board shows a USB glyph in place of the charging bolt while they arrive,
and the numbers in DAW Control. The newest firmware version is read from
`~/.config/crosspad/latest.json` (`{"version": "x.y.z"}`). A hub restart
drops a running Wine DAW's subscription like any port change does: relink with
`aconnect` or restart the DAW.

The same process holds the board's CDC port for every crosspad-hil client (one
port, many clients, exclusive ops): [bench.md § Sharing one port: the
hub](bench.md#sharing-one-port-the-hub).

## CrossPad UAC2 on the host: `audio nodes`

The USB-audio profile shows up on Linux as an ALSA card that comes and goes
with every profile switch, always under the same card index. WirePlumber
(0.4.17 / PipeWire 1.0.5) enumerates it at start-up but drops the hotplug
events, so nothing appears in the mixer until WirePlumber is restarted — which
tears down every other client's streams. `crosspad-hil audio nodes` keeps the
card usable without the session manager:

```bash
crosspad-hil audio nodes install     # systemd --user unit + WirePlumber rule
crosspad-hil audio nodes status      # service, card index, node state
crosspad-hil audio nodes run         # foreground, what the unit runs
crosspad-hil audio nodes uninstall
```

The watcher follows `udevadm monitor` for the CrossPad audio card, waits for
logind's ACL on `controlC<n>` (it lands ~35 ms after the card), then creates two
nodes straight in the PipeWire daemon through `pw-cli -m create-node` — a
source **CrossPad Microphone** and a sink **CrossPad Output** — and destroys
them when the card goes away. The names never change, whatever `hw:N` the card
got this time, so a DAW or `pavucontrol` shows the pads' capture path and the
speaker path as two fixed devices in any plug order. The WirePlumber rule
(`~/.config/wireplumber/main.lua.d/51-crosspad-audio.lua`) disables the
`alsa_card.usb-Espressif_Crosspad_Audio*` device so a later WirePlumber restart
cannot add a second, differently named copy; restart WirePlumber once after
`install` for it to apply.

`--take-defaults` makes the nodes the session's default source and sink each
time they come up and moves the streams of the applications named with
`--apps` (default "FL Studio") onto them — a DAW under Wine only ever sees
"[Default input]" / "[Default output]", and a stream it already opened does
not follow a default change by itself.

The stream is stereo; what it carries is the board's route preset, set with
`crosspad-hil audio route`, Settings → USB, or the DAW Control bar.
`mic_src 0|1|2|3` picks the codec whose ADC is recorded (2 = **Mix**, both
codecs summed on the board; 3 = the pads-only "USB rec" mixer bus, digital, no
DAC/ADC), and `adc_input <codec> <input>` picks that codec's input: the
microphones, the line jack, or the DAC→ADC loop. `capture --preset` sets both
(`headphone`, `speaker`, `mics`, `line-in`, `mix`, `sampler`).

A four-channel stream was tried and dropped: the ESP32-S3 USB FIFO cannot fit a 4-channel isochronous
IN endpoint beside MIDI and feedback, TinyUSB refuses the configuration.
`capture --preset mix` records the sum.

## FL Studio mixer input: `fl template`

A DAW under Wine sees the CrossPad as `[Default input]`, so FL's mixer lists it
as the driver's "In 1 - In 2", never by name. FL's MIDI-scripting API cannot
select a track's audio input (no `mixer` call carries it, no REC event does,
and probing that id range crashes FL), so the input is set where FL keeps it —
in the project:

```bash
crosspad-hil fl template install [--tracks N] [--input I]
crosspad-hil fl template status
```

`install` takes FL's own `Data/Templates/Empty/Empty.flp`, names the first
insert(s) "CrossPad", points them at input pair 0 (the default stereo pair) and
writes the result to every user profile's `Projects/Templates/CrossPad/` in the
Wine prefix holding FL (Bottles and `~/.wine` are searched; `--drive-c` to
choose). FL lists it under File → New from template → CrossPad after its next
start; the insert is ready to arm. **An armed insert monitors its input**: with
the board's capture preset on a DAC→ADC bus (`headphone` / `speaker`) that is
an analog loop — mic → FL → CrossPad Output → mixer → DAC → ADC → mic — and it
rails within a second. Record through `mics`, `line-in` or `mix` while
monitoring; the bus presets are for listening to what the board plays. `crosspad_hil.fl_template` carries the
`.flp` insert layout (`Input` event 154, `-1` = none) and rewrites nothing
else in the file.
