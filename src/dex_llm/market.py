from __future__ import annotations

from collections.abc import Sequence

from dex_llm.models import Candle


def compute_atr(candles: Sequence[Candle], period: int = 14) -> float:
    if not candles:
        raise ValueError("candles are required to compute ATR")

    if len(candles) == 1:
        candle = candles[0]
        return candle.high - candle.low

    true_ranges: list[float] = []
    for previous, current in zip(candles, candles[1:], strict=False):
        true_ranges.append(
            max(
                current.high - current.low,
                abs(current.high - previous.close),
                abs(current.low - previous.close),
            )
        )

    window = true_ranges[-period:] if len(true_ranges) >= period else true_ranges
    return sum(window) / len(window)
