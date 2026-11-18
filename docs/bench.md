# Sharing one board: the bench lease

The everyday steps (claim, run, put the board back, release) are in
[Test on hardware (HIL)](https://crosspad.app/docs/how-to/hil-testing/), and
what to do when another holder has the board is in its section
[The board is claimed by another holder](https://crosspad.app/docs/how-to/hil-testing/#the-board-is-claimed-by-another-holder).
This page is the full behaviour.

Several sessions (people, Claude sessions, CI) share one CrossPad, and opening
its CDC port for a moment, or flashing it, breaks whoever is in the middle of a
test. The port locks only cover the instant a port is open; the **bench lease**
covers the whole test:

```bash
export CROSSPAD_BENCH_HOLDER=pidf-cf            # or --holder on every command
crosspad-hil bench claim "firmware_app test, ~40 min" --ttl 45
crosspad-hil run smoke                          # refused with BENCH_BUSY for anyone else
crosspad-hil bench release "platform-idf feat/x eac76e8"   # what is on the board now
crosspad-hil bench status                       # holder, ETA, queue, firmware left, history
```

One holder per board, a queue behind it, a note of the firmware the holder
left, and expiry: a lease not renewed within its ttl (default 30 min) ends, so
a session that died does not keep the board. Every board op by the holder
renews it, and a running scenario or OTA keeps it alive. `claim` exits 1 when
you were queued instead; ask again later — a queued place not refreshed for an
hour lapses, and when the board is released the head of the queue has 10
minutes to claim it.

Every command and daemon op that touches the board checks the lease and is
refused with `BENCH_BUSY` (exit 2) — naming the holder and their ETA — while
somebody else holds it; a board nobody claimed is not gated. `--bench-force`
(daemon: `bench_force`) overrides, and the override is written into the
board's history. `CROSSPAD_BENCH_MODE=warn` logs the refusal and lets the op
through instead — for a first rollout next to a hand-kept queue — and `off`
checks nothing; claims, releases and the queue work the same in every mode. The daemon's `bench.claim` / `renew` / `release` / `status`
/ `check` ops are what crosspad-mcp's `crosspad_bench_*` tools call;
`bench.check` is for tools that touch the board outside the daemon (idf.py, an
ST-Link).

The state is one JSON file every process shares —
`$XDG_STATE_HOME/crosspad-hil/bench.json` (`~/.local/state/...`), or
`$CROSSPAD_HIL_BENCH` — read and rewritten under an exclusive lock. With
`CROSSPAD_BENCH_MD=path/to/BENCH.md` every change also rewrites the block
between `<!-- bench:status -->` markers in that file; `crosspad-hil bench md
--write FILE` does it once.

## A quiet run

```bash
CROSSPAD_HIL_QUIET=1 crosspad-hil run rt_glitch
```

With `CROSSPAD_HIL_QUIET=1` a scenario refuses (exit 2) a board whose speakers
are saved louder than 5 %, and nothing in the run writes codec 0's volume. Set
the board up once:

```bash
crosspad-hil cdc AUDIO_VOL 0 5      # saved as the speakers' level
crosspad-hil console --reset        # a restart gives it to the speakers' pair alone
```

The speakers are one output pair of codec 0; the Phones 1/2 pair, which the
loop captures read, keeps its own level after the restart. `AUDIO_VOL` and
`OUTVOL` set both pairs at once, so lowering them during a run silences the
capture: `kit_swap_audio` then reports hits that made no sound and `rt_glitch`
SILENT. Put the speakers back with `AUDIO_VOL 0 64` and a restart. A scenario
that listens through the board's microphones (`audio_loopback`,
`speaker_acoustic`) has nothing to hear in a quiet run.

## Sharing one port: the hub

The bench lease says *who* is testing; the hub decides who has the board's
**port** at this instant. `crosspad-hil midi hub run` (the systemd unit, see
[host-tools.md](host-tools.md#a-daw-under-wine-midi-hub)) also serves a control
socket (`$XDG_RUNTIME_DIR/crosspad-hil/hub.sock`; `127.0.0.1` + token on
Windows). Every crosspad-hil client sends its verbs through it, so a CP Tools
screen, an MCP query and a scenario's verbs share one CDC; the hub opens it on
the first verb and closes it 5 s after the last client, which is when a browser
(crosspad.app Web Serial) can have it. In the MIDI+UAC2 profile the same verbs
go over SysEx `0x1F`. OTA, file transfer, USB profile switches, capture,
stimulus, `console --reset` and scenarios take an **exclusive op**: the hub
steps aside and everyone else gets `PORT_BUSY` naming the holder until it ends
(or its process dies). A scenario class that only sends verbs sets
`exclusive = False`. The bench lease is checked first, the hub second.

```bash
crosspad-hil midi hub status              # ... clients=2 exclusive=- cdc=open
crosspad-hil --no-hub cdc LED_STATE       # direct, as before (also CROSSPAD_HIL_NO_HUB=1)
crosspad-hil serve --client cp-tools      # the name others see while it holds the port
systemctl --user restart crosspad-midi-hub.service   # after upgrading crosspad-hil
```

A client that finds no hub starts one (`midi hub run`, detached) and waits up
to 3 s for it.
