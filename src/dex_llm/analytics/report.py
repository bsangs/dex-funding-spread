from __future__ import annotations

from collections.abc import Sequence

from dex_llm.models import TradeOutcome


def summarize_outcomes(outcomes: Sequence[TradeOutcome]) -> dict[str, float]:
    if not outcomes:
        return {
            "count": 0.0,
            "win_rate": 0.0,
            "avg_pnl": 0.0,
            "avg_hold_minutes": 0.0,
        }

    wins = sum(1 for outcome in outcomes if outcome.pnl > 0)
    total_pnl = sum(outcome.pnl for outcome in outcomes)
    total_hold = sum(outcome.hold_minutes for outcome in outcomes)
    return {
        "count": float(len(outcomes)),
        "win_rate": wins / len(outcomes),
        "avg_pnl": total_pnl / len(outcomes),
        "avg_hold_minutes": total_hold / len(outcomes),
    }

