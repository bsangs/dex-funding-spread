from __future__ import annotations

from collections.abc import Mapping
from datetime import UTC, datetime
from time import perf_counter
from typing import Protocol

from dex_llm.market import compute_atr
from dex_llm.models import (
    Candle,
    Cluster,
    ClusterShape,
    ClusterSide,
    HeatmapSnapshot,
    HyperliquidClearinghouseState,
    HyperliquidFrontendOrder,
    HyperliquidUserFill,
    MapQuality,
    MarketFrame,
    OrderBookSnapshot,
    PositionState,
    PriceLevel,
    SweepObservation,
    TradeSide,
)
from dex_llm.risk.kill_switch import KillSwitchPolicy


class HyperliquidClientProtocol(Protocol):
    def fetch_l2_book(self, coin: str) -> OrderBookSnapshot: ...

    def fetch_candles(self, coin: str, interval: str, limit: int) -> list[Candle]: ...

    def fetch_clearinghouse_state(
        self,
        user: str,
        dex: str = "",
    ) -> HyperliquidClearinghouseState: ...

    def fetch_frontend_open_orders(
        self,
        user: str,
        dex: str = "",
    ) -> list[HyperliquidFrontendOrder]: ...

    def fetch_user_fills_by_time(
        self,
        user: str,
        start_time: int,
        end_time: int | None = None,
    ) -> list[HyperliquidUserFill]: ...


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
        kill_switch_policy: KillSwitchPolicy | None = None,
    ) -> None:
        self.hyperliquid_client = hyperliquid_client
        self.heatmap_client = heatmap_client
        self.synthetic_provider = SyntheticHeatmapProvider()
        self.kill_switch_policy = kill_switch_policy or KillSwitchPolicy()

    def build(
        self,
        symbol: str,
        heatmap_params: Mapping[str, str] | None = None,
        allow_synthetic: bool = False,
        user_address: str | None = None,
        dex: str = "",
    ) -> MarketFrame:
        public_started = perf_counter()
        book = self.hyperliquid_client.fetch_l2_book(symbol)
        candles_5m = self.hyperliquid_client.fetch_candles(symbol, "5m", limit=30)
        candles_15m = self.hyperliquid_client.fetch_candles(symbol, "15m", limit=30)
        public_latency_ms = (perf_counter() - public_started) * 1000

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

        private_errors: list[str] = []
        position = PositionState()
        private_state_loaded = False
        private_latency_ms: float | None = None
        clearinghouse_state: HyperliquidClearinghouseState | None = None
        open_orders: list[HyperliquidFrontendOrder] = []
        fills: list[HyperliquidUserFill] = []

        if user_address:
            private_started = perf_counter()
            try:
                clearinghouse_state = self.hyperliquid_client.fetch_clearinghouse_state(
                    user=user_address,
                    dex=dex,
                )
            except Exception as exc:
                private_errors.append(f"clearinghouseState failed: {exc}")

            try:
                open_orders = self.hyperliquid_client.fetch_frontend_open_orders(
                    user=user_address,
                    dex=dex,
                )
            except Exception as exc:
                private_errors.append(f"frontendOpenOrders failed: {exc}")

            try:
                day_start = book.captured_at.astimezone(UTC).replace(
                    hour=0,
                    minute=0,
                    second=0,
                    microsecond=0,
                )
                fills = self.hyperliquid_client.fetch_user_fills_by_time(
                    user=user_address,
                    start_time=int(day_start.timestamp() * 1000),
                    end_time=int(book.captured_at.timestamp() * 1000),
                )
            except Exception as exc:
                private_errors.append(f"userFillsByTime failed: {exc}")

            private_latency_ms = (perf_counter() - private_started) * 1000
            private_state_loaded = not private_errors
            position = self._build_position_state(
                symbol=symbol,
                clearinghouse_state=clearinghouse_state,
                open_orders=open_orders,
                fills=fills,
            )

        frame_timestamp = min(book.captured_at, heatmap_snapshot.captured_at)
        component_timestamps: dict[str, str] = {
            "book": book.captured_at.isoformat(),
            "heatmap": heatmap_snapshot.captured_at.isoformat(),
        }
        metadata: dict[str, object] = {
            "best_bid": book.best_bid,
            "best_ask": book.best_ask,
            "heatmap_provider": heatmap_snapshot.provider,
            "public_data_latency_ms": round(public_latency_ms, 2),
            "component_timestamps": component_timestamps,
        }
        if heatmap_errors:
            metadata["heatmap_error"] = heatmap_errors[0]
        if heatmap_snapshot.raw_path is not None:
            metadata["heatmap_raw_path"] = heatmap_snapshot.raw_path
        if user_address:
            metadata["user_address"] = user_address
            metadata["private_state_latency_ms"] = round(private_latency_ms or 0.0, 2)
            metadata["private_state_loaded"] = private_state_loaded
            if private_errors:
                metadata["private_state_error"] = "; ".join(private_errors)
            if clearinghouse_state is not None:
                metadata["account_value"] = clearinghouse_state.margin_summary.account_value
                metadata["withdrawable"] = clearinghouse_state.withdrawable
                component_timestamps["private_state"] = clearinghouse_state.time.isoformat()

        kill_switch = self.kill_switch_policy.evaluate(
            frame_timestamp=frame_timestamp,
            position=position,
            info_latency_ms=public_latency_ms,
            private_state_latency_ms=private_latency_ms,
            private_state_required=user_address is not None,
            private_state_loaded=private_state_loaded or user_address is None,
            heatmap_provider=heatmap_snapshot.provider,
            heatmap_error=heatmap_errors[0] if heatmap_errors else None,
        )
        metadata["kill_switch"] = kill_switch.model_dump(mode="json")

        return MarketFrame(
            timestamp=frame_timestamp,
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
            sweep=self._infer_sweep_state(candles_5m, heatmap_snapshot),
            position=position,
            kill_switch=kill_switch,
            metadata=metadata,
        )

    def _infer_sweep_state(
        self,
        candles_5m: list[Candle],
        heatmap_snapshot: HeatmapSnapshot,
    ) -> SweepObservation:
        if not candles_5m:
            return SweepObservation()

        candidates: list[SweepObservation] = []
        for cluster in heatmap_snapshot.clusters_above[:3]:
            candidate = self._analyze_cluster_sweep(candles_5m, cluster)
            if candidate is not None:
                candidates.append(candidate)
        for cluster in heatmap_snapshot.clusters_below[:3]:
            candidate = self._analyze_cluster_sweep(candles_5m, cluster)
            if candidate is not None:
                candidates.append(candidate)

        if not candidates:
            return SweepObservation()

        candidates.sort(
            key=lambda candidate: (
                candidate.body_reclaimed,
                candidate.trigger_candle_ts or datetime.min.replace(tzinfo=UTC),
            ),
            reverse=True,
        )
        return candidates[0]

    def _analyze_cluster_sweep(
        self,
        candles: list[Candle],
        cluster: Cluster,
    ) -> SweepObservation | None:
        latest = candles[-1]
        touch_index: int | None = None
        for index, candle in enumerate(candles):
            if cluster.side == ClusterSide.ABOVE and candle.high >= cluster.price:
                touch_index = index
            if cluster.side == ClusterSide.BELOW and candle.low <= cluster.price:
                touch_index = index

        if touch_index is None:
            return None

        trigger = candles[touch_index]
        touched_window = candles[touch_index:]
        if cluster.side == ClusterSide.ABOVE:
            wick_only = max(trigger.open, trigger.close) < cluster.price
            body_reclaimed = (
                latest.close < cluster.price
                and any(candle.close < cluster.price for candle in touched_window)
            )
        else:
            wick_only = min(trigger.open, trigger.close) > cluster.price
            body_reclaimed = (
                latest.close > cluster.price
                and any(candle.close > cluster.price for candle in touched_window)
            )

        return SweepObservation(
            touched_cluster_side=cluster.side,
            wick_only=wick_only,
            body_reclaimed=body_reclaimed,
            cluster_price=cluster.price,
            trigger_candle_ts=trigger.ts,
        )

    def _build_position_state(
        self,
        symbol: str,
        clearinghouse_state: HyperliquidClearinghouseState | None,
        open_orders: list[HyperliquidFrontendOrder],
        fills: list[HyperliquidUserFill],
    ) -> PositionState:
        position = PositionState(
            open_orders=sum(1 for order in open_orders if order.coin == symbol),
            consecutive_losses_today=self._count_consecutive_losses(fills, symbol),
        )
        if clearinghouse_state is None:
            return position

        matched_position = next(
            (
                item
                for item in clearinghouse_state.asset_positions
                if item.coin == symbol and abs(item.size) > 0
            ),
            None,
        )
        if matched_position is None:
            return position

        side = TradeSide.FLAT
        if matched_position.size > 0:
            side = TradeSide.LONG
        elif matched_position.size < 0:
            side = TradeSide.SHORT

        return PositionState(
            side=side,
            entry_price=matched_position.entry_price,
            quantity=abs(matched_position.size),
            open_orders=position.open_orders,
            consecutive_losses_today=position.consecutive_losses_today,
            liquidation_price=matched_position.liquidation_price,
            unrealized_pnl=matched_position.unrealized_pnl,
            margin_used=matched_position.margin_used,
        )

    def _count_consecutive_losses(
        self,
        fills: list[HyperliquidUserFill],
        symbol: str,
    ) -> int:
        closed_fills = [
            fill
            for fill in sorted(fills, key=lambda item: item.time, reverse=True)
            if fill.coin == symbol and abs(fill.closed_pnl) > 0
        ]
        streak = 0
        for fill in closed_fills:
            if fill.closed_pnl < 0:
                streak += 1
                continue
            break
        return streak
