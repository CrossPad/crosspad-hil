# Serial transcripts

Each `*.ndjson` file here is a recorded conversation with a real CrossPad, replayed in
tests through `crosspad_hil.record.ReplaySerial` so scenarios and typed verbs run in CI
without hardware. The reply lines are what `platform-idf/main/hil_control.cpp` actually
printed — never hand-edit a reply to make a test pass; re-record instead.

## Format

One JSON object per line, in order of occurrence:

| `dir`  | fields                        | meaning                                                     |
|--------|-------------------------------|-------------------------------------------------------------|
| `meta` | `version`, `port`, `baudrate` | first line; `version` is `record.TRANSCRIPT_VERSION` (1)   |
| `ctl`  | `attr` (`dtr`/`rts`), `value` | the host set a control line (hygiene: both should be false) |
| `tx`   | `data` (hex)                  | bytes the host wrote                                        |
| `rx`   | `data` (hex)                  | bytes the host read (one `read()`/`readline()` result)      |

`t` is seconds since the recorder was created. Empty reads (timeouts) are not recorded.
`ReplaySerial(path, speed=1.0)` reproduces the recorded gaps; `speed=0.0` (default in tests)
replays instantly.

## Producing one

