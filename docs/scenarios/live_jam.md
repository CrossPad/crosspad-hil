# `live_jam`: a whole live set, every glitch with its moment and its cause

`crosspad-hil run live_jam` plays the set a performer plays with a phone on the
USB-C port (a live setup): the phone streams into the board, a beat is
built and performed on the pads and the sequencer, kits are swapped (into the
pitched engine too), every per-pad feature of a kit is used, the desk changes,
a DAW's MIDI clock stops and starts the transport, an app comes to the front,
the phone pauses its stream, scenes are bounced into the audio song. The
board's recording bus is captured **digitally** over USB IN the whole time
(capture preset `sampler`, `mic_src 3`: a frame of the take is a frame of the
render), and the take is held, voice by voice, against what the board should
have played.

Tag criterion: **zero glitches** -- no underrun, hole, click, missing, late or
cut note, no break in the host's stream -- and no board counter that moved.

## Run it

```bash
export CROSSPAD_BENCH_HOLDER=<you>
crosspad-hil bench claim "live_jam" --ttl 60
PYTHONPATH=~/GIT/_wt/hil-jam crosspad-hil run live_jam -d dev_xxxx --holder <you>
#   --minutes 12.5      USB-audio time (the set scales; rates and grooves do not)
#   --soak-min 10       one more continuous 10-min block (the known underrun after 5-6 min)
#   --seed 1            the set is deterministic per seed (re-run = glitch determinism)
#   --sections all      or some of build,stream,bounce,clock,ui
#   --calibrate         write the measured latencies/levels as this firmware's calibration
#   --audible           leave Out 1/2 on (default: muted in the jam desks; the take is unchanged)
crosspad-hil jam script --seed 1                 # the set as a table, no board
crosspad-hil jam analyze hil_logs/live_jam_<ts>  # re-analyse a run, no board
crosspad-hil jam synth /tmp/j --fault hole@5.2   # a synthetic run with a fault, analysed
crosspad-hil jam kits /tmp/kits                  # the four fixture kits
```

About 20 min at the defaults (12.5 min of USB audio in four blocks, five CDC
windows, four profile switches each way, the put-back with a restart), 30 min
with `--soak-min 10`. The first run on a card pushes the kits (~19 MB) and
renders the FX kit's `.fx/` files on its first load (inside block B2, on
purpose).

## The set

| | profile | what happens | what it proves |
|---|---|---|---|
| PREP | CDC | default profile, pins journal (an earlier jam's promise kept first), console reset + boot (SD mode), **home snapshot** (SEQ, SONG, CLICK, kit, app, desk, preset slots 4-6 and `sequences.json` fetched) **journalled before any change**, kits provisioned, `.fx` deleted, capabilities probed (companion, companion SEQ), desk HILHOME saved to slot 6 and the jam desks JAM-DUCK / JAM-FULL built from it (USB->rec on, no `gains`, Out 1/2 muted), `ASONG OPEN hil_live_jam` | nothing of the user's is touched before it is copied |
| W0 | CDC | S1 recorded by PAD_PRESS with count-in: kick on 1 and 3, snare on 2 and 4, hat on the 8ths, open hat on the 'and' of 4; beat one played a 32nd early (count-in grace); `SEQ_DUMP` checked against what was played | REC, count-in grace, 1/16 quantisation |
| **B1** | UAC | the stream; **one** (pad 12 at 12 Hz, 45 s); **four** (pads 8-11, 6 s samples, 8 Hz, 75 s) with swaps A->B->A; preview of 3 pads; codec volume/mute moved (negative control); a velocity ramp; pads through the companion verb; host MIDI notes (negative: they play nothing); the phone's stream stops for 2 s; **bounce #1** (SysEx 1D 0C) of S1 over "one" | retrigger, the streamer under long samples, swaps (free pads at once, held hits <= 1 per swap <= 50 ms), preview as a voice, the digital bus ignoring the codec, a stream pause the pads do not notice, a bounce that stops the sequencer |
| W1a | CDC | `SEQ OVERDUB` on S1 (guarded: the sounding scene must be the jam's) | |
| **B1b** | UAC | the hi-perc layer on pads 4-7 in 16ths, +-20 ms off the grid | overdub: each hit's `quantise()` vs `SEQ_DUMP`, heard from the next pass |
| W1 | CDC | `SEQ PLAY` closes the overdub; S2 recorded (groove on 8-11), `DOUBLE`; pad 3 erased from S1; song S1 x2, S2 x1, loop; JAM-DUCK; Recorder `IN PADS` + `REC` | scenes, DOUBLE, ERASE, song, a desk preset |
| **B2** | UAC | song plays; **many** (16 pads at 16 Hz + 8-pad chords, 60 s); swap A->FX (first load renders `.fx` on the card mid-set), the **FX tour** (pitch +7/-5, reverse, low/high pass, pan L/R, volume 50, choke, hold, loop on/off, two layers, a slice); swap back; **MIDI clock** F8 at 120 BPM, FC, FB, FC, FA, then silence (hand-back); **UI**: `APP_START Mixer` / `Sampler` through the companion (closing the Mixer writes its desk to NVS); four with swaps through the companion; **bounce #2** of S2 (4 bars) | the 8-voice budget (oldest stolen: STOLEN is fine, EARLY_CUT is not), every kit feature, the FX render under load, transport following the host, an app switch, the companion swap path |
| W2 | CDC | Recorder `STOP` and a 10 s window of its take fetched (header + range, not the whole file); **bounce #3 over CDC** while CDC pads play; S3 (1 bar, `QUANT 4` = 1/8T) recorded; JAM-FULL | the Recorder's take vs the capture, bounces from both paths, the triplet grid |
| **B3** | UAC | S3 plays; swap A->P (engine switch: the sequencer's notes become pitched blips); a legato melody; 12-note chords (cap 10: two steals each); one note retriggered; P->A->P; **bounce #4** of S3 over chords | the pitched engine: notes, release, cap and steal order, retrigger, engine switches |
| SOAK | UAC | `--soak-min`: four/many alternating with A<->B swaps, the stream continuous | a long session (the known 5-6 min underrun) |
| FINAL | CDC | reset log, MEM, WORKER, PITCHED_STATUS, ARENA, CLOCK_STATUS, SMPL_ENGINE, SMPL_POOL (ERR only_native on an image without the prebuilt engine) | |
| put back | CDC | sequencer and song stopped, the jam's scenes deleted (never one that existed), BPM/QUANT/COUNTIN/song back, the kit **by name**, the app, desk HILHOME loaded (NVS), preset slots put back or removed, the jam's project deleted and the user's reopened, `sequences.json` written back once the sequencer is quiet, **a restart** (RAM-only state: swing, codec volume, preset undo), the board's state compared with the snapshot; the journal entry cleared only when all of it came back | |

Every step names its features, what it should do and how it is checked
(`crosspad-hil jam script`), and `jam_report.json`'s `coverage` says, per
feature, `verified`, `exercised` (ran in a block that was not measured),
`failed` or `not_run`. A step the board cannot do (no companion, no bounce
SysEx) is **skipped with its reason**; a missing feature makes the verdict
`not measured` unless `--allow-partial`.

## What is listened for

The fixture kits (`crosspad_hil/jam/spec.py`) give every pad of every kit a
frequency of its own (A `230+240p`, B `350+240p`, FX 4.3-7.9 kHz one feature
per pad, P whole tones 85..115), the host streams a **pilot** (12 kHz, -26
dBFS) and a **timecode** (band-limited noise, 15-19 kHz, period 2^18), and two
**quiet bands** (9-10.5, 13.5-14.5 kHz) carry nothing. The metronome's 1000 /
1500 Hz are kept 50 Hz clear (60 Hz is geometrically impossible with any
120 Hz interleave).

| class | how | resolution |
|---|---|---|
| `PILOT_GAP` | the host's own signal subtracted from the take (the path is bit-exact): what is left where the stream had a hole is the stream itself | a sample (0.5 ms holes found) |
| `SLIP` / `CAPTURE_SLIP` | the timecode's delay changes by exactly the samples lost/repeated; a pad tone's phase across it says whether the capture lost them (pads and stream jump together) or the stream did | a sample |
| `CLICK` | energy in the quiet bands 20 dB over their floor, not explained by an edge the model expects (a steal, a restart, a hard loop start, a pilot edge) | ~0.1 ms |
| `MISSING` / `SEQ_MISS` | no voice started in the band: from the band's complex envelope (a restart inside a sounding tone is read too: the sample comes back at phase 0) | per voice |
| `LATE` / `LATE_SWAP` | onset later than the path's own median by 10 ms (8 ms for the sequencer); a hit held by a swap's reload: <= 1 per swap, <= 50 ms (`--strict`: none) | ~1 ms |
| `HOLE` | a voice's band 10 dB under its own level inside it | 2 ms columns |
| `EARLY_CUT` | a voice ended before its time **with no other voice starting** (an end at another voice's start while the engine is at its limit is a steal: fine) | ~2 ms |
| `WRONG_KIT`, `GHOST`, `SEQ_EXTRA`, `CLICK_LEAK` | sound in a band where nothing is due (40 ms windows: the 120 Hz neighbour leaks under -45 dB) | 5 ms |
| `REC_*`, `rec` counter | overdub/REC hits against `SEQ_DUMP` | per hit |
| `BOUNCE` | state, clips+1, the file's holes/clicks, the file vs the capture of the same render (8.5 kHz low-passed, gain fitted): which side is lower says card write or USB IN | 10 ms |
| counters | SPK `under`/`refused`/`cap_drop`, engine `underrun`/`noblock`, swap `dropped_hits`/`underruns`, pitched `shed`, pad path `inj_drop`/`uart_ovf`/`sbuf_drop`/`usb_drop`, pool, resets, fatal lines, a hang | per block |

The model (`crosspad_hil/jam/model.py`) runs the sequencer from the CLICK
anchor read just before the block (re-fitted to the take), the song order,
the overdub, bounce and MIDI-transport windows, the sampler's budget (8 stereo
voices, the one furthest into its sample stolen; ties inside one render block
left open), the pitched cap (a releasing voice first, then the oldest; 5 ms
fade), choke/hold/loop, swaps timed by `KIT_SWAP_LOG`, the desk's gains. After
the onsets are read it is **rebuilt on them**, so steals, ends and clicks are
judged against what actually started when. What it cannot vouch for is
`unverifiable`, never PASS: a restart that landed at the same phase, a chord
landing on a full engine (which voices survive is decided inside a render
block), another app in front, a reversed sample's start, a swap without its
log row. The report says how many voices were judged and how many were not.

## Every glitch, on every clock

`jam_report.json` lists every glitch with:

* **when**: the take's frame and second, host monotonic, wall clock, the
  board's log ms (console lines stamped on arrival, the UART time taken off,
  lower envelope per boot), the script step and seconds into it, the
  sequencer's scene, pass and bar.beat.tick;
* **what**: its classes (a 20 ms hole is one glitch: a pilot gap, a hole in
  every sounding voice, a click at each edge), what was measured and expected;
* **around it**: the action it belongs to (the hit, the sequencer event), the
  actions within 250 ms and the last 10, the model's state (voices sounding,
  kits, swap/bounce/fx-render/ui flags), the SPK lines either side and their
  delta, engine reports, console lines from -1.5 to +0.5 s, the host stream's
  xruns and pauses, the firmware and SD card;
* **why**: a cause with its evidence and a confidence -- `host_playback`,
  `host_capture`, `usb_stream`, `usb_capture`, `engine_underrun`, `swap`,
  `bounce`, `fx_render_io`, `render_overload`, `polyphony`, `ui`, `nvs_commit`,
  `sd_writer`, `log_busywait`, `inject_queue`, `seq_poll_stall`, `unknown`;
* `glitches/gNNNN_{cap,exp,diff}.wav`: 0.5 s of the take, of the expectation
  and of their difference.

`by_flag` counts glitches per minute in each condition (swap, bounce,
fx_render, ui, many, pitched, clock, ...): where they cluster.
`timeline.html` draws it (one file, no external library); `labels_<block>.txt`
are Audacity labels for each `blocks/<block>/capture.wav`;
`expected_<block>.wav` is what the take should have been.

| what the stream shows | the pads | telemetry | cause |
|---|---|---|---|
| hole / slip | unchanged | SPK `under`/`refused` up, host `out_xrun` | `usb_stream` / `host_playback` |
| slip of N samples | the same N (phase) | `cap_drop`/`cap_under` up, host `in_xrun` | `usb_capture` / `host_capture` |
| unchanged | hole / click | `underrun_at`, swap/bounce flag | `engine_underrun`, `swap`, `bounce` |
| hole | hole | -- | `render_overload` |

## Verdict

* `live_jam: ZERO-GLITCH PASS` (exit 0): no glitch, no counter, coverage
  complete (or `--allow-partial`), clocks fitted;
* `FAIL` (exit 1): every glitch with its moment, step and cause, every counter;
* `not measured` (exit 2): the host lost audio, a capability is missing, a
  clock could not be fitted, the bench was wrong. A board that hangs is a FAIL
  (`hung` with the console tail), recovered over the STM bridge; the set goes
  on.
* `LEFT CHANGED: ...` when something could not be put back; the pins journal
  keeps the home state and the next jam (or any scenario's normalising step)
  puts it back.

## Known limits

* Never run on the board from this branch's workflow: the board-side code is
  exercised against a fake board (`tests/scenarios/test_live_jam.py`) on a
  virtual clock; the analysis against synthetic takes with known faults
  (`tests/analyze/test_jam_golden.py`). First bench runs: `--minutes 3` (a
  smoke jam), then `--calibrate`, then the full set, then the same seed again.
* `--calibrate` exists for what the model cannot read off the firmware: the
  real SysEx/companion latency, the steal/retrigger fades, the low-pass
  attenuation, the velocity curve, the pan law. Until calibrated, LEVEL is
  informational and the onset judgement uses each path's own median.
* A stream fault is placed to the sample but attributed to a second: SPK lines
  come at most once a second. Without a firmware event ring (`ev_at=` in SPK,
  `t0_ms` of a bounce, `KITSWAP` lines) `usb_stream` means "the device or the
  host/link".
* In the USB-audio profile the companion allowlist has no SEQ/ASONG/MIX: the
  transport and desk move only in the CDC windows; overdub is armed and closed
  there (block B1b). With companion SEQ the script would not need the windows.
* The pitched kit's notes in a scene are ~100 ms blips (a sequencer note is
  released on the next 2 ms poll): reported as a product gap, not a glitch.
* Host MIDI notes play nothing (`SamplerPadLogic::onNoteOn` is empty):
  verified as such, reported under `product_gaps`.
* SWING (UI only, not readable), `masterFX.reverb` (no DSP), STM_VPAD (which pad
  a `mode` hits is not mapped), SD hot-plug and USB replugging (need hands) are
  not covered.
* The MIDI clock with a host tempo other than the scene's is not modelled (the
  host tempo is written to the metronome without rescaling the events); the
  clock section runs at the scene's tempo.
