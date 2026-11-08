# crosspad-hil

Python library, CLI and NDJSON daemon that talks to a CrossPad (ESP32-S3 pad
controller): device discovery, the STM32-bridge console with boot/fatal
parsers, CDC verbs, MIDI SysEx, USB profile switching, OTA flashing, a lease
for sharing one board, and the hardware-in-the-loop scenario suite. Every
hardware trap (DTR/RTS reboots the ESP, `OK` is not your ack, CDC vanishes in
UAC2 mode, `F0 7D 19 01 F7` hangs the STM bridge) is encoded once here and
inherited by the CLI, the scenarios and crosspad-mcp.

Not on PyPI yet; install from git (Python 3.10 or newer; the Windows commands
are on the site):

```bash
python3 -m venv .venv && . .venv/bin/activate
pip install "crosspad-hil[all] @ git+https://github.com/CrossPad/crosspad-hil"
crosspad-hil doctor
crosspad-hil devices
crosspad-hil run smoke
```

## Documentation

The docs live at **https://crosspad.app/docs/**:

- Install and first smoke test: [Set up the toolchain](https://crosspad.app/docs/start-here/1-toolchain/), [Check it works](https://crosspad.app/docs/start-here/4-check/)
- Run scenarios, read reports, share a bench: [Test on hardware (HIL)](https://crosspad.app/docs/how-to/hil-testing/)
- Every command and its common options: [Command-line tools § crosspad-hil](https://crosspad.app/docs/reference/cli/#crosspad-hil)
- When it fails: [Troubleshooting § Checks and tests](https://crosspad.app/docs/troubleshooting/#checks-and-tests)
- The firmware side: [Hardware-in-the-loop testing](https://crosspad.app/docs/subsystems/hil-testing/)

In this repo: [docs/bench.md](docs/bench.md) (the bench lease and the hub in full),
[docs/padmix.md](docs/padmix.md) (levelling kits from the computer),
[docs/host-tools.md](docs/host-tools.md) (`kit push`, FPC presets,
`companion`, `midi hub`, `audio nodes`, `fl template`) and the scenario notes
under [docs/](docs/). Design: [the crosspad-mcp v10 spec](https://github.com/CrossPad/crosspad-mcp/blob/master/docs/superpowers/specs/2026-08-25-crosspad-hil-and-mcp-v10-design.md).

## Scenarios

This list is the reference for `crosspad-hil run`, so it stays here: it
mirrors the registry in `crosspad_hil/scenarios/__init__.py`, and a new
scenario adds its row in the same change. `crosspad-hil run <name> --help`
lists a scenario's flags. Exit code **0** passed, **1** the firmware failed,
**2** the bench is wrong; audio, BLE and raw-ALSA scenarios need the `[all]`
extras and exit 2 naming what is missing without them.
`platform-idf/tools/hil_*.py` are thin shims over these.

| Scenario | What it does |
|---|---|
| `app_churn` | open/close every app repeatedly; per-app internal-heap slope + crash watch |
| `audio_loopback` | play a multitone through the UAC2 endpoints and judge the return signal; it sets the host gain and the jack's preamp itself, see [docs/scenarios/audio_loopback.md](docs/scenarios/audio_loopback.md) |
| `back_churn` | descend into every app (and the sampler's file explorer, Settings, the drawer) and click the power button back out, one level per click; crash / core-dump watch + per-target heap slope |
| `beat` | make a beat end to end from one YAML (`crosspad_hil/workflows/beat.yaml`, also played on the simulator): a loop it writes itself is sliced in the Slicer, exported as a kit, loaded, recorded as a pattern in a song of its own, bounced and checked hit by hit, plus the theme; `--depth quick` (a few minutes) / `medium` (pad editor, overdub, x2, mixer, song edits) / `full` (erase, scenes, a song of scenes, an import, a Live-set export); the song, kit, app, desk and theme are put back |
| `ble_midi` | drive BLE MIDI both ways from this PC's radio, in both of the board's roles |
| `daw_control` | DAW Control app over the ESP MIDI port: hello reply, `led_all` painting matches `LED_STATE`, a simulated pad press/release turns into NoteOn/NoteOff, `text` doesn't upset the app |
| `firmware_app` | the Firmware app end to end, driven with the encoder and read through `FW_UI`: 30 test packages pushed to the card and listed in the Library, a kit card (pin, unpin, delete from card), a Store download in a normal boot and in network mode, the PSRAM budget, churn |
| `fs_transfer` | round-trip a file over CDC: proves host and device CRCs agree and that framing returns to text after a binary stream; `--assets` reflashes the assets partition |
| `fw_switch` | install a package for the running build from the SD card (`FW_INSTALL`), switch OTA slots and back (`FW_SWITCH`), and check the same image comes back (`APP_SHA`); the package must already be on the card |
| `fw_update` | firmware updates end to end: serves CI-signed packages of the board's kit and revision from this machine (`crosspad_hil.fw_server`), points the board at it with `FW_UPD_URL`, and checks the catalog, resumable verified downloads, signature refusals, cleanup, install and the network-mode fallback; the bench WiFi comes from `$CROSSPAD_WIFI_SSID` / `$CROSSPAD_WIFI_PASS` |
| `kit_churn` | swap kits while the pads keep firing; `--rapid` spins the selector |
| `kit_swap_audio` | swap kits while pads fire, recorded through UAC2, scored per tone: every hit audible, no hole over `--max-gap-ms`, no pad reloaded mid-sample; builds and pushes its own sine fixtures when the card lacks them (`--no-provision` to refuse) |
| `latency` | pad -> jack and host -> jack: notes on the STM bridge and noise bursts into UAC OUT, timed on the board's own DAC->ADC loop by a duplex stream with a clock log; median/p95/max per path, the capture ring the board reports (`SPK #` cap_ring) subtracted, underruns split out; `--max-pad-ms` / `--max-host-ms` gate. Puts back the kit, app and Mixer desk. See [docs/latency.md](docs/latency.md) for what it does and does not measure |
| `led_state` | dump the LED controller model (colours, brightness, anim, power state) |
| `live_jam` | A live setup, listened to sample by sample (~20 min): the host streams a pilot + timecode through the board's mixer while a beat is built and performed (REC with count-in, overdub, scenes, DOUBLE/ERASE, song, MIDI clock Stop/Continue/Start), pads in every style, every kit feature (pitch, reverse, filters, pan, volume, choke, hold, loop, layers, slice), sampler and pitched swaps, previews, desk presets, an app switch, a stream pause, bounces over SysEx and CDC, the Recorder; USB IN recorded digitally (`sampler` bus) and every voice held against a model of the engines: each glitch with its frame, wall clock, board log ms, step, action and cause; counters gated at zero; everything put back (sequences.json, desk/NVS, preset slots, project) and journalled first. `crosspad-hil jam script|analyze|synth|kits` without a board. See [docs/scenarios/live_jam.md](docs/scenarios/live_jam.md) |
| `live_set` | the full live setup (~30 min): per phase (one pad fast / four pads / all 16 + chords / pitched) a bounce window in the default profile (a scene of the phase's pads at 120 BPM, bounced while the pads play, the file pulled and scored, the scene restarted) and a live block in USB audio (host stream on `crosspad_out`, USB IN recorded with the board's own routing and checked for clicks, holes and every struck pad audible, kit swaps incl. sampler↔pitched); zero tolerance on SPK under/refused, engine underrun/noblock, swap late/dropped hits and rt_gap, pad-path drops, resets; normalises the bench first and puts everything back |
| `midi_bench` | echo-frame throughput and latency on the ESP port, the STM bridge, and both |
| `midi_clock` | send a host MIDI clock at several tempos (`--bpms`, default 90,120,174): the board follows it (`CLOCK_STATUS` src=host, bpm within 0.1 %, every tick counted), then takes over on its own clock when the host stops |
| `midi_stress` | burst SysEx queries at the ESP MIDI port and count the replies (RX loss) |
| `pad_edit_sticky` | the Sampler's pad editor: a setting in edit mode (CHOKE) stays in edit mode across pad hits, and one detent changes it on the next pad |
| `perf` | kit swap HIL SINE A↔B timed by the device (`KIT_SWAP_LOG` median `done_ms`): over CDC, then in the USB-audio profile with `pw-play` streaming into the board (swaps via companion `KIT_LOAD`); plus the sample engine's init at boot (`EngineHandoff` control block → started). FAIL with the numbers above `--max-cdc-ms` (2000), `--max-uac-ratio` (1.2× CDC) or `--max-init-ms` (500) |
| `pitched_kit` | load the pitched-instrument fixture kit, record it through UAC2 and check each pad's pitch, a 16-pad chord and `PITCHED_STATUS` (steals, RT budget, root detection) |
| `power_state` | STM power path without a person: BQ25895 profile held past its 80 s watchdog, no profile re-apply, no STM reset, no stuck USB_SUSPENDED / SHIP_PENDING, LED1 in AUTO (STM 1.9 power debug words via `STM_REG`) |
| `release_check` | a published release seen from the bench board: crosspad.app lists the tag for its revision and kit, `FW_CHECK` over CDC and SysEx, `FW_UPD_STATUS` / `FW_LATEST` name it; `--install` (or a board already on that tag) downloads it into the other slot, switches in, checks `APP_VERSIONS` against the release and switches back |
| `release_gate` | the whole chain in one report: smoke → power_state → app_churn → back_churn → kit_churn → perf → usb_mode_cycle → ble_midi → audio_loopback → stability; per-step budgets from the rounds, a stuck step is cancelled then abandoned, the USB profile is restored before every step; `--quick` for a CI-length dry run, `--keep-going` overnight |
| `rt_glitch` | phased UAC2 capture: clicks under pad/LED load that a held pad does not have |
| `sampler_record` | record the sampler through the board's own UAC2 endpoint and score it |
| `seq_kit_change` | record a take in a scene of its own, load another kit while it plays and then the first one back: every hit stays, and the scenes are stored outside the kits folder |
| `smoke` | reset, capture the boot log, check markers / E-lines / fatals / boot loops |
| `song_bounce` | record a scene, bounce it to the audio song (`ASONG BOUNCE`), play the song, then STOP/PLAY it hard: no underruns, every clip opens, no reset (`RESET_LOG`) |
| `speaker_acoustic` | record the built-in mics while the sampler plays; did the speaker sound? |
| `speedtest` | ramp the pad hit rate until the device drops hits, per pad set |
| `stability` | soak: watch the console for resets, fatals, stalls, bad boots and heap drift |
| `studio_session` | the studio session of 21.10.2026 played on the screen (`--fixture`): kit, pad trims and copies, a pattern, a bounce into a new song, line-in onto the arrangement, a chopped sample, the mix and an Ableton Live Set export; every control found with `UI_TREE` and tapped, the knob turned, the pads hit; state verbs only read the result, and the song, kit and take are deleted afterwards |
| `usb_mode_cycle` | cycle CDC+MIDI <-> MIDI+UAC2 and watch enumeration, heap and faults |
| `velocity` | sweep pad velocity and measure loudness, monotonicity and trigger jitter |
| `waveform_cache` | load kits and check every pad layer gets its waveform cached (`WAVE_STATUS`); `--play` keeps the pads firing meanwhile |
| `wifi` | WiFi on demand in a normal boot: join, ping, a wrong password fails cleanly, the memory comes back, BLE hands over and back; then one pass through network mode (`NET_MODE`); the bench network comes from `$CROSSPAD_WIFI_SSID` / `$CROSSPAD_WIFI_PASS` |

## Development

```bash
python3 -m venv .venv && . .venv/bin/activate
pip install -e ".[dev,audio]"
ruff check . && pytest
```

Tests need no hardware: `tests/fakes.py` provides `FakeSerial`, `FakeMidiOut`
and `FakeMidiIn`. A change that breaks a docs page (command, output, path)
opens an issue in CrossPad/crosspad-docs.

## License

MIT, see [LICENSE](LICENSE).
