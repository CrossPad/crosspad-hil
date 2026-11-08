# `latency`: pad -> jack and host -> jack, measured on the board

`crosspad-hil run latency` strikes two paths `--hits` times each (60 by
default) with a random 300-700 ms gap and times every hit from the moment it
left the host to its onset on the PCB's own DAC->ADC loop:

| path | stimulus | what it covers |
|---|---|---|
| `pad` | note-on `36+pad` on the **STM bridge's USB MIDI port** (channel 1) | USB OUT to the STM, the STM's relay onto the UART, the ESP's UART RX and `uartLoop`, `onNoteOn2Audio` -> PadManager -> event queue -> engine voice start, I2S TX DMA, DAC. From the STM onward this is the physical pad's own path. |
| `host` | a 20 ms noise burst written into the board's **UAC OUT** | USB isochronous OUT, TinyUSB's FIFO, the playback ring, the mixer's USB channel, I2S TX DMA, DAC. |

Both are recorded through the board's UAC IN (capture preset `headphone` =
codec 1, the phones' DAC looped back to its ADC). One duplex PCM stream on the
card (`crosspad_hil/duplex.py`, libasound through ctypes -- no PortAudio
needed) records UAC IN, plays UAC OUT, and logs which frame was at the
converter at which `CLOCK_MONOTONIC` time; the MIDI sends are stamped on the
same clock. `crosspad_hil/latency_math.py` fits those logs (a lower envelope:
every error in such a log is late), finds each onset (first frame over a
threshold set by the hit's own peak, the take's noise and the run-up) and
summarises.

## Run it

Hold the bench lease (the duplex stream needs the card itself, so the run
pauses `crosspad-audio-nodes` -- only the lease holder may):

```bash
export CROSSPAD_BENCH_HOLDER=<you>
crosspad-hil bench claim "latency" --ttl 30
crosspad-hil run latency -d dev_xxxx --json                      # both paths, report only
crosspad-hil run latency --paths pad --hits 120 --max-pad-ms 15  # a gate on one path
crosspad-hil run latency --max-pad-ms 15 --max-host-ms 28 --gate-stat p95
```

About 2 minutes for the default 60+60 hits, plus a kit push the first time
(`HIL CLICK`, built by `crosspad_hil.fixtures`: per pad a burst of noise at
full level from its first frame, so the onset is the start and not a ramp).

The prologue (CDC) reads the firmware versions and `STM_REG 33C`
(`pressDebounceMs`), loads `HIL CLICK`, starts the Sampler, loads the Mixer
desk `BEATS` (pads and computer on both outputs; `--mixer-desk ''` leaves the
desk alone) and zeroes `PAD_STATS`. Afterwards it reads `PAD_STATS` and
`SMPL_LAST` (`lat_us`) and puts back the desk (`MIX_PRESET UNDO`), the kit and
the app it found; the capture route and the default USB profile are restored
by the capture session. `--in2-gain 24` pins codec 1's preamp for a quiet
loop and puts the old value back. Opening the console (for the `SPK #` lines)
resets the ESP once; `--no-console` avoids it. With `BEATS` the speakers
click along with every hit (`--mixer-desk ''` keeps the loaded desk, if that
one routes the pads to the phones).

Exit codes as everywhere: 0 pass, 1 the firmware failed (a hit that never
sounded or sounded twice, an underrun during the run, a threshold exceeded),
2 the bench (no lease, card busy, a host xrun, nothing audible at all).

## What the report says

`report.json` -> `data.paths.pad` / `data.paths.host`:

* `uac_in_ms` -- send -> sound in the board's UAC IN: what a DAW recording the
  board sees. Nothing subtracted.
* `jack_ms` -- the headline when the console is read: `uac_in_ms` minus the
  capture ring's mean level from the `SPK #` lines (`capture_leg`). The board
  prints those only while a UAC OUT stream plays; the duplex stream keeps one
  playing (silence on the pad path).
* `n/min/p5/p50/p95/max/mean/std` and a seeded bootstrap `p50_ci95`;
  `missing`, `double`, `masked`, `early` list the hits left out and why.
* `includes`, `excludes`, `not_subtracted`, `u_sys_ms`: what is in the number.
* `pad.physical_est_ms` -- **a model, not a measurement (W)**: `p50` plus the
  STM's scan (0.25-1 ms) and its press/velocity window (`STM_REG 33C` - 1 ..
  + 0.75 ms), minus the bridge's USB OUT (0-1 ms). This is the L1 lever
  (`pressDebounceMs` 15 -> 6).
* `pad.esp_echo_ms` -- send -> the ESP's own USB MIDI echo of the note
  (`send2USB`): the control path without the audio. `echo_rtt` is the SysEx
  round trip (USB transport alone).
* `segments` (when an `SPK #` counter moved): the numbers before and after the
  first underrun and their `delta_p50_ms` -- the L3 effect on the host path.
  A line is seen up to a worker pass plus a poll gap after its event, so the
  hits in the `guard_s` before it was seen count in neither (`ambiguous`).
* `data.spk`, `data.device_counters` (`PAD_STATS stm=` must equal the hits
  sent), `data.clock` (the fits' residuals), `data.left` (what was put back).

Artifacts: `capture.wav`, `hits.csv` (one row per hit), `clock.csv`,
`latency.json`, `console.log`.

## What it does not measure

* **The STM's press window (L1) and ADC scan.** No host stimulus enters
  before them: a note on the bridge enters at the STM's UART TX, and so does
  `STM_VPAD` (which also needs CDC, absent in this profile). Hence
  `physical_est_ms`. A finger on a pad is the only full measurement.
* **The absolute jack.** Of the capture path the board reports only the ring
  level. The ADC's group delay, the I2S RX DMA queue (0-3 blocks of 1.45 ms,
  fixed per session) and TinyUSB's IN FIFO stay in `jack_ms`: it reads up to
  ~6 ms late (`u_sys_ms.not_subtracted_ms`), the same amount for both paths
  within one session. An independent recorder on the jack (the PC's line in)
  is what removes it.
* **The host's own buffers.** `host` starts when ALSA reports the frame
  played on the card (+-1 ms for snd-usb-audio); PipeWire's or a DAW's buffer
  in front of that is not in it.
* `cap_ring` is a snapshot per `SPK #` line (every 10 s, or 1 s after an
  event); on firmware before 3a2d89f (L5) the capture ring's level is random
  per session (1.5-63.6 ms on c510199), so compare `jack_ms`, not
  `uac_in_ms`, across those builds -- and even then expect the unsubtracted
  RX queue to move between sessions.

## Before / after

Same bench, same session, a few runs per build: `--hits 120 --seed 1` on the
old build twice and the new one twice, compare `paths.*.jack_ms.p50` (and
`p50_ci95`) and `paths.host.segments.delta_p50_ms`. For L1 compare
`physical_est_ms` at `pressDebounceMs` 15 and 6.
