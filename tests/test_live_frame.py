from __future__ import annotations

from collections.abc import Mapping
from datetime import UTC, datetime, timedelta

from dex_llm.live_frame import LiveFrameBuilder
from dex_llm.models import (
    BboSnapshot,
    Candle,
    Cluster,
    ClusterShape,
    ClusterSide,
    HeatmapSnapshot,
    HyperliquidActiveAssetContext,
    HyperliquidClearinghouseState,
    HyperliquidFrontendOrder,
    HyperliquidMarginSummary,
    HyperliquidPerpPosition,
    HyperliquidUserFill,
    LiveStateSnapshot,
    MapQuality,
    OrderBookSnapshot,
    PriceLevel,
    TradeSide,
)


class FakeHyperliquidClient:
    def __init__(
        self,
        *,
        book: OrderBookSnapshot | None = None,
        candles_5m: list[Candle] | None = None,
        candles_15m: list[Candle] | None = None,
        clearinghouse_state: HyperliquidClearinghouseState | None = None,
        open_orders: list[HyperliquidFrontendOrder] | None = None,
        fills: list[HyperliquidUserFill] | None = None,
    ) -> None:
        now = datetime.now(tz=UTC)
        self.book = book or OrderBookSnapshot(
            symbol="BTC",
            captured_at=now,
            best_bid=70100.0,
            best_ask=70110.0,
            mid_price=70105.0,
            bids=[PriceLevel(price=70100.0, size=8.0, orders=12)],
            asks=[PriceLevel(price=70110.0, size=9.0, orders=14)],
        )
        self.candles_5m = candles_5m or self._build_candles(
            now=now,
            interval_minutes=5,
            length=24,
            start_price=70000.0,
        )
        self.candles_15m = candles_15m or self._build_candles(
            now=now,
            interval_minutes=15,
            length=24,
            start_price=69950.0,
        )
        self.clearinghouse_state = clearinghouse_state or HyperliquidClearinghouseState(
            asset_positions=[],
            cross_margin_summary=HyperliquidMarginSummary(account_value=10_000.0),
            margin_summary=HyperliquidMarginSummary(account_value=10_000.0),
            withdrawable=9_500.0,
            time=now,
        )
        self.open_orders = open_orders or []
        self.fills = fills or []

    def fetch_l2_book(self, coin: str) -> OrderBookSnapshot:
        return self.book

    def fetch_candles(self, coin: str, interval: str, limit: int) -> list[Candle]:
        candles = self.candles_5m if interval == "5m" else self.candles_15m
        return candles[-limit:]

    def fetch_clearinghouse_state(
        self,
        user: str,
        dex: str = "",
    ) -> HyperliquidClearinghouseState:
        return self.clearinghouse_state

    def fetch_frontend_open_orders(
        self,
        user: str,
        dex: str = "",
    ) -> list[HyperliquidFrontendOrder]:
        return self.open_orders

    def fetch_user_fills_by_time(
        self,
        user: str,
        start_time: int,
        end_time: int | None = None,
        aggregate_by_time: bool = False,
    ) -> list[HyperliquidUserFill]:
        return self.fills

    def _build_candles(
        self,
        *,
        now: datetime,
        interval_minutes: int,
        length: int,
        start_price: float,
    ) -> list[Candle]:
        candles: list[Candle] = []
        current = start_price
        for index in range(length):
            ts = now - timedelta(minutes=(length - index) * interval_minutes)
            candles.append(
                Candle(
                    ts=ts,
                    open=current,
                    high=current + 80.0,
                    low=current - 70.0,
                    close=current + 20.0,
                    volume=10.0 + index,
                )
            )
            current += 12.0
        return candles


class FakeHeatmapClient:
    def __init__(
        self,
        *,
        image_path: str | None = None,
        captured_at: datetime | None = None,
    ) -> None:
        self.image_path = image_path
        self.captured_at = captured_at or datetime.now(tz=UTC)

    def fetch_heatmap(
        self,
        symbol: str,
        extra_params: Mapping[str, str] | None = None,
    ) -> HeatmapSnapshot:
        return HeatmapSnapshot(
            provider="coinglass",
            symbol=symbol,
            captured_at=self.captured_at,
            image_path=self.image_path,
            clusters_above=[
                Cluster(
                    side=ClusterSide.ABOVE,
                    price=70300.0,
                    size=15.0,
                    shape=ClusterShape.SINGLE_WALL,
                )
            ],
            clusters_below=[
                Cluster(
                    side=ClusterSide.BELOW,
                    price=69900.0,
                    size=12.0,
                    shape=ClusterShape.STAIRCASE,
                )
            ],
        )


def test_live_frame_builder_uses_real_heatmap_when_available() -> None:
    builder = LiveFrameBuilder(FakeHyperliquidClient(), FakeHeatmapClient())

    frame = builder.build("BTC", allow_synthetic=False)

    assert frame.map_quality == MapQuality.CLEAN
    assert frame.metadata["heatmap_provider"] == "coinglass"
    assert frame.clusters_above[0].price == 70300.0
    assert frame.heatmap_image_path is None
    assert frame.heatmap_image_url is None
    assert frame.kill_switch.allow_new_trades is True


def test_live_frame_builder_can_fallback_to_synthetic_clusters() -> None:
    builder = LiveFrameBuilder(FakeHyperliquidClient(), None)

    frame = builder.build("BTC", allow_synthetic=True)

    assert frame.map_quality == MapQuality.MIXED
    assert frame.metadata["heatmap_provider"] == "synthetic-observe-only"
    assert frame.clusters_above[0].side == ClusterSide.ABOVE
    assert frame.kill_switch.allow_new_trades is False
    assert "synthetic heatmap fallback active" in frame.kill_switch.reasons


def test_live_frame_builder_marks_upper_sweep_reclaim_from_recent_candles() -> None:
    now = datetime.now(tz=UTC)
    candles_5m = [
        Candle(
            ts=now - timedelta(minutes=25),
            open=70120.0,
            high=70190.0,
            low=70090.0,
            close=70170.0,
            volume=11.0,
        ),
        Candle(
            ts=now - timedelta(minutes=20),
            open=70170.0,
            high=70240.0,
            low=70130.0,
            close=70210.0,
            volume=12.0,
        ),
        Candle(
            ts=now - timedelta(minutes=15),
            open=70210.0,
            high=70320.0,
            low=70190.0,
            close=70270.0,
            volume=13.0,
        ),
        Candle(
            ts=now - timedelta(minutes=10),
            open=70270.0,
            high=70295.0,
            low=70180.0,
            close=70210.0,
            volume=14.0,
        ),
        Candle(
            ts=now - timedelta(minutes=5),
            open=70210.0,
            high=70250.0,
            low=70160.0,
            close=70195.0,
            volume=15.0,
        ),
    ]
    candles_15m = [
        Candle(
            ts=now - timedelta(minutes=45),
            open=70020.0,
            high=70180.0,
            low=69980.0,
            close=70120.0,
            volume=50.0,
        ),
        Candle(
            ts=now - timedelta(minutes=30),
            open=70120.0,
            high=70260.0,
            low=70090.0,
            close=70210.0,
            volume=54.0,
        ),
        Candle(
            ts=now - timedelta(minutes=15),
            open=70210.0,
            high=70320.0,
            low=70130.0,
            close=70195.0,
            volume=58.0,
        ),
    ]
    builder = LiveFrameBuilder(
        FakeHyperliquidClient(
            candles_5m=candles_5m,
            candles_15m=candles_15m,
        ),
        FakeHeatmapClient(),
    )

    frame = builder.build("BTC")

    assert frame.sweep.touched_cluster_side == ClusterSide.ABOVE
    assert frame.sweep.cluster_price == 70300.0
    assert frame.sweep.wick_only is True
    assert frame.sweep.body_reclaimed is True


def test_live_frame_builder_loads_private_position_state() -> None:
    now = datetime.now(tz=UTC)
    clearinghouse_state = HyperliquidClearinghouseState(
        asset_positions=[
            HyperliquidPerpPosition(
                coin="BTC",
                entry_price=69950.0,
                liquidation_price=65000.0,
                leverage_type="isolated",
                leverage_value=6.0,
                raw_usd=1_200.0,
                margin_used=220.0,
                position_value=1_240.0,
                size=0.4,
                unrealized_pnl=42.0,
            )
        ],
        cross_margin_summary=HyperliquidMarginSummary(account_value=5_500.0),
        margin_summary=HyperliquidMarginSummary(account_value=5_500.0),
        withdrawable=4_900.0,
        time=now,
    )
    open_orders = [
        HyperliquidFrontendOrder(
            coin="BTC",
            side="B",
            limit_price=69800.0,
            size=0.1,
            reduce_only=False,
            is_trigger=False,
            order_type="limit",
            oid=1,
            timestamp=now,
        ),
        HyperliquidFrontendOrder(
            coin="BTC",
            side="A",
            limit_price=70500.0,
            size=0.1,
            reduce_only=True,
            is_trigger=False,
            order_type="takeProfit",
            oid=2,
            timestamp=now,
        ),
    ]
    fills = [
        HyperliquidUserFill(
            coin="BTC",
            closed_pnl=-12.0,
            direction="Close Long",
            price=70020.0,
            size=0.1,
            time=now - timedelta(hours=1),
        ),
        HyperliquidUserFill(
            coin="BTC",
            closed_pnl=8.0,
            direction="Close Long",
            price=70110.0,
            size=0.1,
            time=now - timedelta(hours=2),
        ),
    ]
    builder = LiveFrameBuilder(
        FakeHyperliquidClient(
            clearinghouse_state=clearinghouse_state,
            open_orders=open_orders,
            fills=fills,
        ),
        FakeHeatmapClient(),
    )

    frame = builder.build("BTC", user_address="0xabc")

    assert frame.position.side == TradeSide.LONG
    assert frame.position.quantity == 0.4
    assert frame.position.open_orders == 2
    assert frame.position.consecutive_losses_today == 1
    assert frame.position.liquidation_price == 65000.0
    assert frame.kill_switch.allow_new_trades is True


def test_live_frame_builder_preserves_heatmap_image_metadata() -> None:
    builder = LiveFrameBuilder(
        FakeHyperliquidClient(),
        FakeHeatmapClient(image_path="data/heatmaps/images/coinglass-eth.png"),
    )

    frame = builder.build("BTC", allow_synthetic=False)

    assert frame.heatmap_path == "data/heatmaps/images/coinglass-eth.png"
    assert frame.heatmap_image_path == "data/heatmaps/images/coinglass-eth.png"


def test_live_frame_builder_uses_webdata3_and_userevents_for_private_freshness() -> None:
    now = datetime.now(tz=UTC)
    snapshot = LiveStateSnapshot(
        symbol="BTC",
        order_book=OrderBookSnapshot(
            symbol="BTC",
            captured_at=now,
            best_bid=70100.0,
            best_ask=70110.0,
            mid_price=70105.0,
            bids=[PriceLevel(price=70100.0, size=8.0, orders=12)],
            asks=[PriceLevel(price=70110.0, size=9.0, orders=14)],
        ),
        candles_5m=FakeHyperliquidClient().candles_5m,
        candles_15m=FakeHyperliquidClient().candles_15m,
        bbo=BboSnapshot(symbol="BTC", captured_at=now, bid=70100.0, ask=70110.0, mid=70105.0),
        active_asset_ctx=HyperliquidActiveAssetContext(
            coin="BTC",
            mark_price=70105.0,
            oracle_price=70103.0,
            mid_price=70105.0,
            max_leverage=20.0,
            timestamp=now,
        ),
        clearinghouse_state=HyperliquidClearinghouseState(
            asset_positions=[],
            cross_margin_summary=HyperliquidMarginSummary(account_value=10_000.0),
            margin_summary=HyperliquidMarginSummary(account_value=10_000.0),
            withdrawable=9_500.0,
            time=now,
        ),
        channel_timestamps={
            "l2Book": now,
            "candle:5m": now,
            "candle:15m": now,
            "bbo": now,
            "activeAssetCtx": now,
            "webData2": now - timedelta(seconds=20),
            "webData3": now,
            "orderUpdates": now,
            "userFills": now,
            "userEvents": now,
        },
        channel_snapshot_flags={},
    )
    builder = LiveFrameBuilder(FakeHyperliquidClient(), FakeHeatmapClient(captured_at=now))

    frame = builder.build_from_snapshot(snapshot, allow_synthetic=False)

    assert frame.kill_switch.allow_new_trades is True
