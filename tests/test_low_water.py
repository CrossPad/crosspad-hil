"""The heap low-water check app_churn and stability share."""
from __future__ import annotations

from crosspad_hil.scenarios._low_water import DMA_FLOOR, INT_FLOOR, LowWater


def test_tracks_first_lowest_and_who_lowered_it() -> None:
    lw = LowWater()
    lw.sample({"int_min": 30_000, "dma_min": 20_000})
    lw.sample({"int_min": 29_000, "dma_min": 20_000}, "Sampler visit")
    lw.sample({"int_min": 29_500, "dma_min": 5_000}, "Pitch Play visit")
    assert lw.first == {"int_min": 30_000, "dma_min": 20_000}
    assert lw.low == {"int_min": 29_000, "dma_min": 5_000}
    assert [(d.key, d.size, d.where) for d in lw.drops] == [
        ("int_min", 1_000, "Sampler visit"), ("dma_min", 15_000, "Pitch Play visit")]
    lines, failures = lw.judge()
    assert failures == []
    # Largest fall first.
    falls = lines[lines.index("  largest falls of a mark, and where:") + 1:]
    assert "Pitch Play visit" in falls[0] and "Sampler visit" in falls[1]


def test_under_the_floor_fails_and_says_when() -> None:
    lw = LowWater()
    lw.sample({"int_min": INT_FLOOR - 1, "dma_min": 16_968})
    lw.sample({"int_min": 8_632, "dma_min": 952}, "app_churn")
    _lines, failures = lw.judge()
    assert failures == [
        f"DMA-capable internal heap low-water dma_min=952 B is under the {DMA_FLOOR} B floor "
        "(fell below it during this run; 16968 B at the start)",
        f"internal heap low-water int_min=8632 B is under the {INT_FLOOR} B floor "
        f"(was already below it when this run started; {INT_FLOOR - 1} B at the start)",
    ]


def test_exactly_at_the_floor_passes() -> None:
    lw = LowWater()
    lw.sample({"int_min": INT_FLOOR, "dma_min": DMA_FLOOR})
    assert lw.judge()[1] == []


def test_firmware_without_dma_min_is_not_judged() -> None:
    lw = LowWater()
    lw.sample({"int_free": 50_000, "int_min": 40_000})
    lines, failures = lw.judge()
    assert failures == []
    assert any("dma_min" in line and "not reported by this firmware" in line for line in lines)
    assert lw.as_data()["low"] == {"int_min": 40_000}


def test_a_reboot_does_not_raise_the_low_water() -> None:
    # After a reboot the marks start over higher; the run's low-water stays
    # the lowest one seen.
    lw = LowWater()
    lw.sample({"dma_min": 3_000})
    lw.sample({"dma_min": 40_000}, "after reboot")
    assert lw.low["dma_min"] == 3_000
    assert lw.judge()[1] != []
