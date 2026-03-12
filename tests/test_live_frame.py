from __future__ import annotations

from collections.abc import Mapping
from datetime import UTC, datetime, timedelta

from dex_llm.live_frame import LiveFrameBuilder
from dex_llm.models import (
    Candle,
    Cluster,
    ClusterShape,
    ClusterSide,
    HeatmapSnapshot,
    MapQuality,
    OrderBookSnapshot,
    PriceLevel,
)


class FakeHyperliquidClient:
    def __init__(self) -> None:
        now = datetime.now(tz=UTC)
        self.book = OrderBookSnapshot(
            symbol="BTC",
            captured_at=now,
            best_bid=70100.0,
            best_ask=70110.0,
            mid_price=70105.0,
            bids=[PriceLevel(price=70100.0, size=8.0, orders=12)],
            asks=[PriceLevel(price=70110.0, size=9.0, orders=14)],
        )
        self.candles = [
            Candle(
                ts=now - timedelta(minutes=index * 15),
                open=70000.0 + index,
                high=70100.0 + index,
                low=69900.0 + index,
                close=70050.0 + index,
                volume=10.0 + index,
            )
            for index in range(16, 0, -1)
        ]

    def fetch_l2_book(self, coin: str) -> OrderBookSnapshot:
        return self.book

    def fetch_candles(self, coin: str, interval: str, limit: int) -> list[Candle]:
        return self.candles[-limit:]


class FakeHeatmapClient:
    def fetch_heatmap(
        self,
        symbol: str,
        extra_params: Mapping[str, str] | None = None,
    ) -> HeatmapSnapshot:
        return HeatmapSnapshot(
            provider="coinglass",
            symbol=symbol,
            captured_at=datetime.now(tz=UTC),
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


def test_live_frame_builder_can_fallback_to_synthetic_clusters() -> None:
    builder = LiveFrameBuilder(FakeHyperliquidClient(), None)

    frame = builder.build("BTC", allow_synthetic=True)

    assert frame.map_quality == MapQuality.MIXED
    assert frame.metadata["heatmap_provider"] == "synthetic-orderbook"
    assert frame.clusters_above[0].side == ClusterSide.ABOVE
