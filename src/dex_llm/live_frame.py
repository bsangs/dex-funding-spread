from __future__ import annotations

from collections.abc import Mapping
from datetime import datetime
from typing import Protocol

from dex_llm.market import compute_atr
from dex_llm.models import (
    Candle,
    Cluster,
    ClusterShape,
    ClusterSide,
    HeatmapSnapshot,
    MapQuality,
    MarketFrame,
    OrderBookSnapshot,
    PositionState,
    PriceLevel,
    SweepObservation,
)


class HyperliquidClientProtocol(Protocol):
    def fetch_l2_book(self, coin: str) -> OrderBookSnapshot: ...

    def fetch_candles(self, coin: str, interval: str, limit: int) -> list[Candle]: ...


class HeatmapClientProtocol(Protocol):
    def fetch_heatmap(
        self,
        symbol: str,
        extra_params: Mapping[str, str] | None = None,
    ) -> HeatmapSnapshot: ...


class SyntheticHeatmapProvider:
    def from_orderbook(
        self,
        symbol: str,
        book_snapshot_time: datetime,
        asks: list[PriceLevel],
        bids: list[PriceLevel],
    ) -> HeatmapSnapshot:
        clusters_above = [
            self._cluster_from_level(level, ClusterSide.ABOVE) for level in asks[:3]
        ]
        clusters_below = [
            self._cluster_from_level(level, ClusterSide.BELOW) for level in bids[:3]
        ]
        return HeatmapSnapshot(
            provider="synthetic-orderbook",
            symbol=symbol,
            captured_at=book_snapshot_time,
            clusters_above=clusters_above,
            clusters_below=clusters_below,
            metadata={
                "warning": "Synthetic fallback uses order book liquidity, not liquidation data.",
            },
        )

    def _cluster_from_level(self, level: PriceLevel, side: ClusterSide) -> Cluster:
        orders = level.orders
        size = level.size
        shape = ClusterShape.SINGLE_WALL if orders >= 10 else ClusterShape.STAIRCASE
        return Cluster(side=side, price=level.price, size=size, shape=shape)


class LiveFrameBuilder:
    def __init__(
        self,
        hyperliquid_client: HyperliquidClientProtocol,
        heatmap_client: HeatmapClientProtocol | None = None,
    ) -> None:
        self.hyperliquid_client = hyperliquid_client
        self.heatmap_client = heatmap_client
        self.synthetic_provider = SyntheticHeatmapProvider()

    def build(
        self,
        symbol: str,
        heatmap_params: Mapping[str, str] | None = None,
        allow_synthetic: bool = False,
    ) -> MarketFrame:
        book = self.hyperliquid_client.fetch_l2_book(symbol)
        candles_5m = self.hyperliquid_client.fetch_candles(symbol, "5m", limit=30)
        candles_15m = self.hyperliquid_client.fetch_candles(symbol, "15m", limit=30)

        heatmap_errors: list[str] = []
        heatmap_snapshot: HeatmapSnapshot | None = None
        if self.heatmap_client is not None:
            try:
                heatmap_snapshot = self.heatmap_client.fetch_heatmap(
                    symbol=symbol,
                    extra_params=heatmap_params,
                )
            except Exception as exc:
                heatmap_errors.append(str(exc))

        if heatmap_snapshot is None:
            if not allow_synthetic:
                raise RuntimeError(
                    "Heatmap provider failed and synthetic fallback is disabled: "
                    + (heatmap_errors[0] if heatmap_errors else "no provider configured")
                )
            heatmap_snapshot = self.synthetic_provider.from_orderbook(
                symbol=symbol,
                book_snapshot_time=book.captured_at,
                asks=book.asks,
                bids=book.bids,
            )
            map_quality = MapQuality.MIXED
        else:
            map_quality = MapQuality.CLEAN

        metadata: dict[str, object] = {
            "best_bid": book.best_bid,
            "best_ask": book.best_ask,
            "heatmap_provider": heatmap_snapshot.provider,
        }
        if heatmap_errors:
            metadata["heatmap_error"] = heatmap_errors[0]
        if heatmap_snapshot.raw_path is not None:
            metadata["heatmap_raw_path"] = heatmap_snapshot.raw_path

        return MarketFrame(
            timestamp=book.captured_at,
            exchange="hyperliquid",
            symbol=symbol,
            current_price=book.mid_price,
            candles_5m=candles_5m[-30:],
            candles_15m=candles_15m[-30:],
            clusters_above=heatmap_snapshot.clusters_above[:3],
            clusters_below=heatmap_snapshot.clusters_below[:3],
            atr=max(compute_atr(candles_15m), 1.0),
            heatmap_path=heatmap_snapshot.image_path or heatmap_snapshot.image_url,
            map_quality=map_quality,
            sweep=self._infer_sweep_state(book.mid_price, heatmap_snapshot),
            position=PositionState(),
            metadata=metadata,
        )

    def _infer_sweep_state(
        self,
        current_price: float,
        heatmap_snapshot: HeatmapSnapshot,
    ) -> SweepObservation:
        above_prices = [cluster.price for cluster in heatmap_snapshot.clusters_above]
        below_prices = [cluster.price for cluster in heatmap_snapshot.clusters_below]
        touched_side: ClusterSide | None = None
        if above_prices and current_price >= min(above_prices):
            touched_side = ClusterSide.ABOVE
        if below_prices and current_price <= max(below_prices):
            touched_side = ClusterSide.BELOW
        return SweepObservation(
            touched_cluster_side=touched_side,
            wick_only=False,
            body_reclaimed=False,
        )
