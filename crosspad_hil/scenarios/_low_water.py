"""Internal and DMA-capable heap low-water marks from MEM, judged against a floor.

``int_min`` and ``dma_min`` in the MEM reply are ESP-IDF's
``minimum_free_bytes``: the lowest free heap since boot, not now. A session can
leave ``int_free`` comfortable and still have passed through a moment with
almost nothing left, and that moment is when an SD sector buffer, a WiFi frame
or BLE's controller fails to allocate — which a customer sees as a random
failure. Free-heap slopes never see it, so it is judged on its own.

Two readings of the numbers:

* the marks only fall until the next boot, so the lowest value seen over a run
  is its low-water even across a reboot (a reboot fails the run anyway);
* ``heap_caps_get_info`` adds up each region's own minimum, which can be lower
  than the free total ever was at one instant. The mark errs low, never high.

``dma_*`` is the internal RAM a DMA buffer can come from; ``int_*`` also counts
~8 kB of RTC fast memory no DMA can reach. Firmware before ``dma_min`` existed
(2026-10-07) is reported as not measured, not judged.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

# Measured on dev_1ede with v1.1.1 (2026-11-05): ready board dma_min=16968
# int_min=24648; after beat_session x2, every app once and app_churn x8,
# dma_min=952 int_min=8632. The floors sit between, with room for one more
# sector buffer and a BLE start's worth of controller allocations.
DMA_FLOOR = 4 * 1024
INT_FLOOR = 16 * 1024

# (MEM key, what the line calls it)
MARKS: tuple[tuple[str, str], ...] = (
    ("dma_min", "DMA-capable internal"),
    ("int_min", "internal"),
)


@dataclass
class Drop:
    """One reading that set a new low-water mark, and what was happening."""

    key: str
    before: int
    after: int
    where: str

    @property
    def size(self) -> int:
        return self.before - self.after


@dataclass
class LowWater:
    """Lowest ``int_min`` / ``dma_min`` over a run, the first reading, and who lowered them."""

    int_floor: int = INT_FLOOR
    dma_floor: int = DMA_FLOOR
    first: dict[str, int] = field(default_factory=dict)
    low: dict[str, int] = field(default_factory=dict)
    drops: list[Drop] = field(default_factory=list)

    def floor(self, key: str) -> int:
        return self.dma_floor if key == "dma_min" else self.int_floor

    def sample(self, mem: dict[str, Any], where: str = "") -> None:
        """Take one MEM reply; a fall in a mark is booked against ``where``."""
        for key, _name in MARKS:
            raw = mem.get(key)
            if raw is None:
                continue
            value = int(raw)
            if key not in self.low:
                self.first[key] = value
                self.low[key] = value
                continue
            if value < self.low[key]:
                if where:
                    self.drops.append(Drop(key, self.low[key], value, where))
                self.low[key] = value

    def judge(self) -> tuple[list[str], list[str]]:
        """(report lines, failures). A floor of 0 reports the mark without judging it."""
        lines = ["--- heap low-water (lowest free since boot) " + "-" * 16]
        failures: list[str] = []
        for key, name in MARKS:
            floor = self.floor(key)
            if key not in self.low:
                lines.append(f"  {key:<8} not reported by this firmware — not judged")
                continue
            first, low = self.first[key], self.low[key]
            verdict = ""
            if floor and low < floor:
                when = ("fell below it during this run" if first >= floor
                        else "was already below it when this run started")
                verdict = f"  <-- under the {floor} B floor"
                failures.append(
                    f"{name} heap low-water {key}={low} B is under the {floor} B floor "
                    f"({when}; {first} B at the start)")
            judged = f"floor {floor}" if floor else "not judged"
            lines.append(f"  {key:<8} {first:>8} -> {low:>8}  ({judged}){verdict}")
        biggest = sorted(self.drops, key=lambda d: d.size, reverse=True)[:5]
        if biggest:
            lines.append("  largest falls of a mark, and where:")
            for d in biggest:
                lines.append(f"    {d.key:<8} -{d.size:>6} B ({d.before} -> {d.after})  {d.where}")
        return lines, failures

    def as_data(self) -> dict[str, Any]:
        return {
            "floors": {"int_min": self.int_floor, "dma_min": self.dma_floor},
            "first": dict(self.first),
            "low": dict(self.low),
            "drops": [{"key": d.key, "before": d.before, "after": d.after, "where": d.where}
                      for d in self.drops],
        }
