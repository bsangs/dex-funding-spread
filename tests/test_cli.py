from __future__ import annotations

from datetime import UTC, datetime

from dex_llm.cli import _apply_rest_private_bootstrap
from dex_llm.models import (
    Candle,
    HyperliquidClearinghouseState,
    HyperliquidMarginSummary,
    LiveOrderState,
    LiveStateSnapshot,
    OrderBookSnapshot,
    OrderState,
    PriceLevel,
)


def test_apply_rest_private_bootstrap_marks_snapshot_and_preserves_private_state() -> None:
    now = datetime.now(tz=UTC)
    snapshot = LiveStateSnapshot(
        symbol="ETH",
        order_book=OrderBookSnapshot(
            symbol="ETH",
            captured_at=now,
            best_bid=2100.0,
            best_ask=2101.0,
            mid_price=2100.5,
            bids=[PriceLevel(price=2100.0, size=1.0, orders=1)],
            asks=[PriceLevel(price=2101.0, size=1.0, orders=1)],
        ),
        candles_5m=[
            Candle(ts=now, open=2100.0, high=2102.0, low=2099.0, close=2101.0, volume=1.0)
        ],
        candles_15m=[
            Candle(ts=now, open=2100.0, high=2102.0, low=2099.0, close=2101.0, volume=1.0)
        ],
    )
    state = HyperliquidClearinghouseState(
        asset_positions=[],
        cross_margin_summary=HyperliquidMarginSummary(account_value=1000.0),
        margin_summary=HyperliquidMarginSummary(account_value=1000.0),
        withdrawable=900.0,
        time=now,
    )
    open_orders = [
        LiveOrderState(
            coin="ETH",
            side="B",
            limit_price=2095.0,
            size=0.1,
            reduce_only=False,
            is_trigger=False,
            order_type="limit",
            oid=1,
            status=OrderState.OPEN,
        )
    ]

    updated = _apply_rest_private_bootstrap(
        snapshot,
        clearinghouse_state=state,
        open_orders=open_orders,
        fills=[],
        bootstrapped_at=now,
    )

    assert updated.clearinghouse_state == state
    assert updated.open_orders == open_orders
    assert updated.channel_timestamps["restPrivateBootstrap"] == now
    assert updated.channel_snapshot_flags["restPrivateBootstrap"] is False
