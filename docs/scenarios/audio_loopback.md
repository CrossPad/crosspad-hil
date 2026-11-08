# `audio_loopback`: host gain and the jack's preamp

`crosspad-hil run audio_loopback` plays a multitone through the UAC2 endpoints and
judges the return signal. Two things between the board and the verdict are
set by the scenario itself, and both are in its report.

## Host gain

A level measured through the UAC2 card first passes the host's
own volumes: the card's ALSA mixer (`PCM`, found at 30/50 = -20 dB on the
bench) and the PipeWire node volumes. The session (WirePlumber, alsactl)
restores its stored values every time the card enumerates, so a setting made
by hand does not survive a profile switch. `audio_loopback` therefore sets
both to 0 dB each time the card appears, reads them back into its report
(`host_gain`, `host_gain_final`), exits 2 naming the volume when one will not
reach 0 dB, and turns "level too low" FAILs into exit 2 when the gain drifted
during the run. `--no-host-gain` leaves the host alone.
`crosspad-hil audio gain` shows the reading; `--set` applies it.

## The jack's preamp (IN2)

The bench loop goes DAC -> jack -> codec 1's ADC,
whose preamp (`INGAIN`, IN2) the firmware brings up at 0 dB since
platform-idf dcfac16; the loop returned -45.7 dBFS there and -20.6 at +24 dB.
For the jack presets `audio_loopback` reads `INGAIN` over CDC, sets IN2 to
`--in2-gain` (24) before entering USB audio, and puts the old value back
afterwards, on failure and cancel too — the board keeps it in NVS. The report
carries `ingain_before` / `ingain_during` / `ingain_after`; the level floor
(-45 dBFS at +24 dB) moves with the gain; a value the board will not take, or
will not take back, is exit 2.
