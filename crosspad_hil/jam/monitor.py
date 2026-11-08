"""The jam's live ear: ``python -m crosspad_hil.jam.monitor <take.wav> <events.jsonl>``.

A process of its own (the duplex thread that writes the take must keep the
GIL), reading the take as it grows -- Python's ``wave`` patches the header on
every write -- half a second at a time: the pilot's level and the energy in
the quiet bands. A pilot that drops 10 dB or a quiet band 20 dB over its floor
is printed at once (``LIVE t=12.345 event=pilot_drop ...``) and appended to
``events.jsonl``. It is a first look while the set plays; the verdict is the
offline analysis.
"""
from __future__ import annotations

import json
import sys
import time
import wave
from pathlib import Path
from typing import Any

import numpy as np

from crosspad_hil.jam import spec as S

CHUNK_S = 0.5
IDLE_EXIT_S = 5.0


class LiveMonitor:
    def __init__(self, sr: int, pilot_hz: float = S.PILOT_HZ) -> None:
        self.sr = sr
        self.pilot_hz = pilot_hz
        self.pilot_level: float | None = None
        self.q_floor: float | None = None
        self.events: list[dict[str, Any]] = []

    def feed(self, x: np.ndarray, t0: float) -> list[dict[str, Any]]:
        """One chunk (mono float) starting at take time ``t0``; returns new events."""
        out: list[dict[str, Any]] = []
        n = len(x)
        if n < 64:
            return out
        k = np.arange(n)
        ph = np.exp(-2j * np.pi * self.pilot_hz * k / self.sr)
        hop = max(1, int(0.002 * self.sr))
        m = n // hop
        y = (x[: m * hop] * ph[: m * hop]).reshape(m, hop).mean(axis=1)
        amp = 2 * np.abs(y)
        level = float(np.median(amp))
        if self.pilot_level is None and level > 1e-4:
            self.pilot_level = level
        if self.pilot_level:
            low = np.flatnonzero(amp < self.pilot_level * 10 ** (-10 / 20))
            if len(low):
                out.append({"t": round(t0 + low[0] * hop / self.sr, 3), "event": "pilot_drop",
                            "ms": round(len(low) * hop / self.sr * 1000, 1)})
            self.pilot_level = 0.9 * self.pilot_level + 0.1 * max(level, 1e-6)
        spec = np.abs(np.fft.rfft(x * np.hanning(n)))
        hz = np.fft.rfftfreq(n, 1 / self.sr)
        q = sum(float(spec[(hz >= a) & (hz <= b)].mean()) for a, b in S.QUIET_BANDS
                if b < self.sr / 2)
        if self.q_floor is None:
            self.q_floor = q
        elif q > 10 * max(self.q_floor, 1e-9):
            out.append({"t": round(t0, 3), "event": "quiet_band_energy",
                        "db_over_floor": round(20 * np.log10(q / max(self.q_floor, 1e-9)), 1)})
        else:
            self.q_floor = 0.95 * self.q_floor + 0.05 * q
        self.events += out
        return out


def follow(wav: Path, out: Path, *, idle_exit_s: float = IDLE_EXIT_S,
           sleep: Any = time.sleep) -> int:
    deadline = time.monotonic() + 30.0
    while not wav.exists():
        if time.monotonic() > deadline:
            return 1
        sleep(0.2)
    pos = 0
    mon: LiveMonitor | None = None
    idle_since = time.monotonic()
    with open(out, "a", encoding="utf-8") as fh:
        while True:
            try:
                with wave.open(str(wav), "rb") as w:
                    sr, ch, n = w.getframerate(), w.getnchannels(), w.getnframes()
                    if mon is None:
                        mon = LiveMonitor(sr)
                    want = int(CHUNK_S * sr)
                    if n - pos >= want:
                        w.setpos(pos)
                        raw = w.readframes(want)
                        x = np.frombuffer(raw, dtype="<i2").reshape(-1, ch)[:, 0] / 32768.0
                        for ev in mon.feed(x, pos / sr):
                            print(f"LIVE t={ev['t']:.3f} event={ev['event']} "
                                  + " ".join(f"{k}={v}" for k, v in ev.items()
                                             if k not in ("t", "event")), flush=True)
                            fh.write(json.dumps(ev) + "\n")
                        pos += want
                        idle_since = time.monotonic()
                        continue
            except (EOFError, wave.Error, OSError, ValueError):
                pass
            if time.monotonic() - idle_since > idle_exit_s:
                return 0
            sleep(0.2)


def main(argv: list[str] | None = None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    if len(args) != 2:
        print("usage: python -m crosspad_hil.jam.monitor <take.wav> <events.jsonl>",
              file=sys.stderr)
        return 2
    return follow(Path(args[0]), Path(args[1]))


if __name__ == "__main__":
    sys.exit(main())
