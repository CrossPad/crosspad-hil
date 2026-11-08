# padmix — levelling kits from the computer

`crosspad-hil padmix` is the Pad Mixer app on your computer: sixteen faders on
the kit the CrossPad is playing. You hear every move on the board at once, and
it is written into that kit's `kit.json` on the card by itself. Use it to level
many kits in one sitting.

## Before you start

- The board is plugged in over USB and in the **CDC profile**
  (`crosspad-hil usb-mode set default`; the MIDI+UAC2 profile has no CDC).
- To hear a pad from the computer (space), **open an app on the board** — the
  Sampler, for instance. On the home screen the pads open apps instead of
  playing.
- Firmware with `PAD_MIX` / `KIT_SAVE` (platform-idf after PR #130).

```bash
crosspad-hil padmix
```

## The screen

```
 Pad Mix  kit 15: LIVE_ROOTS  (0..127)                          saved
      100  100   97  100 ...                     <- value of each pad
 127  │    │    ┃    │                           <- the selected pad is yellow
 100 ████ ████ ████ ████ ...                     <- 100 = unity (0 dB)
   0 ████ ████ ████ ████
        1    2    3    4 ...  16                 <- pad numbers
     pad 3: 03.wav   vol 97  -0.3 dB
```

Top right: **saved** / **unsaved** — whether `kit.json` on the card matches what
you hear. Volume is in the kit's own units: 0..127 with 100 = unity (127 =
+2.1 dB); a pitched kit goes 0..100.

## Keys

| Faders | |
|---|---|
| ← → | choose a pad |
| ↑ ↓ | volume ±1 |
| PgUp PgDn | volume ±8 |
| space | play the pad |
| Enter | the kit list |
| `[` `]` | previous / next kit (the current one is saved first) |
| Esc | undo the last action |
| `s` | save now |
| `q` or Ctrl-C | quit (saves first) |

| Kit list | |
|---|---|
| ↑ ↓, PgUp PgDn | choose |
| letters | filter by name |
| Enter | load the kit on the board, back to the faders |
| Esc | clear the filter; again: back to the faders |

## Mouse

- **Drag** a fader up or down.
- **Wheel** over a pad: ±1.
- **Click a pad number** to play it.
- **Click the title** to open the kit list.
- In the kit list: click to choose, **double-click** to load, wheel to scroll,
  right-click to go back.

## Saving and undo

- `kit.json` is written **1.5 s after your last move**, before another kit is
  loaded, and when you quit. You never have to press `s`.
- **Esc undoes one action at a time**: a fader move (moves of one pad less
  than 0.35 s apart — a held key, a drag — are one step), or a **kit change** — the previous kit is loaded again, and the
  next Esc undoes your moves in it.
- The whole kit is rewritten, the same way the Sampler's own save does it.

## Workflow for many kits

1. Open the Sampler on the board, start `crosspad-hil padmix`.
2. Level the sixteen pads (space to listen, arrows or the mouse to set).
3. `]` — saved, the next kit comes up. Repeat.

## When something goes wrong

| You see | Why / what to do |
|---|---|
| `audition needs an app open on the board` | the board is on the home screen — open the Sampler |
| `ERR loading` in the status line | a kit is still loading; the move goes out on its own once it has |
| `kit.json NOT saved` | the card refused the write; `s` retries, Esc takes the moves back |
| no reply / CDC errors at start | the board is in the USB-audio profile: `crosspad-hil usb-mode set default` |
| Esc does nothing in the kit list | it clears the filter first; press it again |
