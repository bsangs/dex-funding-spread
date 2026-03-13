from __future__ import annotations

from collections.abc import Mapping
from datetime import UTC, datetime
from time import perf_counter
from typing import Protocol

from dex_llm.executor.safety import extract_role_from_cloid
from dex_llm.market import compute_atr
from dex_llm.models import (
    Candle,
    Cluster,
    ClusterShape,
    ClusterSide,
    HeatmapSnapshot,
    HyperliquidClearinghouseState,
    HyperliquidFrontendOrder,
    HyperliquidUserEvent,
    HyperliquidUserFill,
    LiveOrderState,
    LiveStateSnapshot,
    MapQuality,
    MarginMode,
    MarketFrame,
    OrderBookSnapshot,
    OrderState,
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
        aggregate_by_time: bool = False,
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
            provider="synthetic-observe-only",
            symbol=symbol,
            captured_at=book_snapshot_time,
            clusters_above=clusters_above,
            clusters_below=clusters_below,
            metadata={
                "warning": (
                    "Synthetic fallback is observe-only and uses order book liquidity, "
                    "not liquidation data."
                ),
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
            if not heatmap_snapshot.clusters_above and not heatmap_snapshot.clusters_below:
                can_merge_synthetic = allow_synthetic or heatmap_snapshot.provider.startswith(
                    "coinglass-web-scrape"
                )
                if not can_merge_synthetic:
                    raise RuntimeError(
                        "Heatmap provider returned no clusters and synthetic fallback is disabled"
                    )
                heatmap_snapshot = self._merge_synthetic_clusters(
                    heatmap_snapshot=heatmap_snapshot,
                    synthetic_snapshot=self.synthetic_provider.from_orderbook(
                        symbol=symbol,
                        book_snapshot_time=book.captured_at,
                        asks=book.asks,
                        bids=book.bids,
                    ),
                )
                map_quality = MapQuality.MIXED

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
                if hasattr(self.hyperliquid_client, "fetch_user_fills_window"):
                    fills, fills_safe = self.hyperliquid_client.fetch_user_fills_window(
                        user=user_address,
                        start_time=int(day_start.timestamp() * 1000),
                        end_time=int(book.captured_at.timestamp() * 1000),
                    )
                    if not fills_safe:
                        private_errors.append("userFillsByTime exceeded safe backfill limit")
                else:
                    fills = self.hyperliquid_client.fetch_user_fills_by_time(
                        user=user_address,
                        start_time=int(day_start.timestamp() * 1000),
                        end_time=int(book.captured_at.timestamp() * 1000),
                        aggregate_by_time=True,
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
        if heatmap_snapshot.metadata:
            metadata["heatmap_metadata"] = dict(heatmap_snapshot.metadata)
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
            heatmap_path=(
                heatmap_snapshot.heatmap_image_path
                or heatmap_snapshot.heatmap_image_url
                or heatmap_snapshot.image_path
                or heatmap_snapshot.image_url
            ),
            heatmap_image_path=heatmap_snapshot.heatmap_image_path or heatmap_snapshot.image_path,
            heatmap_image_url=heatmap_snapshot.heatmap_image_url or heatmap_snapshot.image_url,
            map_quality=map_quality,
            sweep=self._infer_sweep_state(candles_5m, heatmap_snapshot),
            position=position,
            kill_switch=kill_switch,
            metadata=metadata,
        )

    def build_from_snapshot(
        self,
        snapshot: LiveStateSnapshot,
        *,
        heatmap_params: Mapping[str, str] | None = None,
        allow_synthetic: bool = False,
        fills: list[HyperliquidUserFill] | None = None,
    ) -> MarketFrame:
        book = snapshot.order_book
        heatmap_errors: list[str] = []
        heatmap_snapshot: HeatmapSnapshot | None = None
        if self.heatmap_client is not None:
            try:
                heatmap_snapshot = self.heatmap_client.fetch_heatmap(
                    symbol=snapshot.symbol,
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
                symbol=snapshot.symbol,
                book_snapshot_time=book.captured_at,
                asks=book.asks,
                bids=book.bids,
            )
            map_quality = MapQuality.MIXED
        else:
            map_quality = MapQuality.CLEAN
            if not heatmap_snapshot.clusters_above and not heatmap_snapshot.clusters_below:
                can_merge_synthetic = allow_synthetic or heatmap_snapshot.provider.startswith(
                    "coinglass-web-scrape"
                )
                if not can_merge_synthetic:
                    raise RuntimeError(
                        "Heatmap provider returned no clusters and synthetic fallback is disabled"
                    )
                heatmap_snapshot = self._merge_synthetic_clusters(
                    heatmap_snapshot=heatmap_snapshot,
                    synthetic_snapshot=self.synthetic_provider.from_orderbook(
                        symbol=snapshot.symbol,
                        book_snapshot_time=book.captured_at,
                        asks=book.asks,
                        bids=book.bids,
                    ),
                )
                map_quality = MapQuality.MIXED

        position = self._build_position_state(
            symbol=snapshot.symbol,
            clearinghouse_state=snapshot.clearinghouse_state,
            open_orders=[
                HyperliquidFrontendOrder(
                    coin=order.coin,
                    side=order.side,
                    limit_price=order.limit_price,
                    size=order.size,
                    reduce_only=order.reduce_only,
                    is_trigger=order.is_trigger,
                    order_type=order.order_type,
                    oid=order.oid,
                    cloid=order.cloid,
                    trigger_price=order.trigger_price,
                    timestamp=order.timestamp or book.captured_at,
                )
                for order in snapshot.open_orders
            ],
            fills=fills or snapshot.recent_fills,
            last_user_event=(
                snapshot.recent_user_events[-1]
                if snapshot.recent_user_events
                else None
            ),
        )
        public_age_ms = _channel_age_ms(
            snapshot,
            ("l2Book", "candle", "bbo", "activeAssetCtx"),
        )
        private_age_ms = _channel_age_ms(
            snapshot,
            (
                "webData3",
                "orderUpdates",
                "userFills",
                "userEvents",
                "restPrivateBootstrap",
            ),
        )

        frame_timestamp = min(book.captured_at, heatmap_snapshot.captured_at)
        metadata: dict[str, object] = {
            "best_bid": book.best_bid,
            "best_ask": book.best_ask,
            "heatmap_provider": heatmap_snapshot.provider,
            "component_timestamps": {
                key: value.isoformat() for key, value in snapshot.channel_timestamps.items()
            },
            "channel_snapshot_flags": dict(snapshot.channel_snapshot_flags),
        }
        if snapshot.bbo is not None:
            metadata["bbo_mid"] = snapshot.bbo.mid
        if snapshot.active_asset_ctx is not None:
            metadata["oracle_price"] = snapshot.active_asset_ctx.oracle_price
            metadata["mark_price"] = snapshot.active_asset_ctx.mark_price
        if heatmap_errors:
            metadata["heatmap_error"] = heatmap_errors[0]
        if heatmap_snapshot.raw_path is not None:
            metadata["heatmap_raw_path"] = heatmap_snapshot.raw_path
        if heatmap_snapshot.metadata:
            metadata["heatmap_metadata"] = dict(heatmap_snapshot.metadata)

        kill_switch = self.kill_switch_policy.evaluate(
            frame_timestamp=frame_timestamp,
            position=position,
            info_latency_ms=public_age_ms,
            private_state_latency_ms=private_age_ms,
            private_state_required=(
                snapshot.clearinghouse_state is not None or bool(snapshot.open_orders)
            ),
            private_state_loaded=snapshot.clearinghouse_state is not None,
            heatmap_provider=heatmap_snapshot.provider,
            heatmap_error=heatmap_errors[0] if heatmap_errors else None,
        )
        metadata["kill_switch"] = kill_switch.model_dump(mode="json")

        return MarketFrame(
            timestamp=frame_timestamp,
            exchange="hyperliquid",
            symbol=snapshot.symbol,
            current_price=book.mid_price,
            candles_5m=snapshot.candles_5m[-30:],
            candles_15m=snapshot.candles_15m[-30:],
            clusters_above=heatmap_snapshot.clusters_above[:3],
            clusters_below=heatmap_snapshot.clusters_below[:3],
            atr=max(compute_atr(snapshot.candles_15m), 1.0),
            heatmap_path=(
                heatmap_snapshot.heatmap_image_path
                or heatmap_snapshot.heatmap_image_url
                or heatmap_snapshot.image_path
                or heatmap_snapshot.image_url
            ),
            heatmap_image_path=heatmap_snapshot.heatmap_image_path or heatmap_snapshot.image_path,
            heatmap_image_url=heatmap_snapshot.heatmap_image_url or heatmap_snapshot.image_url,
            map_quality=map_quality,
            sweep=self._infer_sweep_state(snapshot.candles_5m, heatmap_snapshot),
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
        last_user_event: object | None = None,
    ) -> PositionState:
        position = PositionState(
            open_orders=sum(1 for order in open_orders if order.coin == symbol),
            active_orders=[
                self._to_live_order(order)
                for order in open_orders
                if order.coin == symbol
            ],
            consecutive_losses_today=self._count_consecutive_losses(fills, symbol),
            last_user_event=(
                last_user_event
                if isinstance(last_user_event, HyperliquidUserEvent)
                else None
            ),
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
            active_orders=position.active_orders,
            consecutive_losses_today=position.consecutive_losses_today,
            fills_cursor=fills[-1].fill_hash if fills else None,
            last_user_event=position.last_user_event,
            liquidation_price=matched_position.liquidation_price,
            unrealized_pnl=matched_position.unrealized_pnl,
            margin_used=matched_position.margin_used,
            live_leverage=matched_position.leverage_value,
            target_leverage=matched_position.leverage_value,
            margin_mode=MarginMode(matched_position.leverage_type)
            if matched_position.leverage_type in {MarginMode.CROSS.value, MarginMode.ISOLATED.value}
            else None,
            entries_blocked_reduce_only=any(
                not order.reduce_only for order in position.active_orders
            ) and side == TradeSide.FLAT,
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
        dedupe: set[str] = set()
        streak = 0
        for fill in closed_fills:
            identity = fill.fill_hash or (
                f"{fill.oid}:{int(fill.time.timestamp() * 1000)}:{fill.closed_pnl}:{fill.size}"
            )
            if identity in dedupe:
                continue
            dedupe.add(identity)
            if fill.closed_pnl < 0:
                streak += 1
                continue
            break
        return streak

    def _to_live_order(self, order: HyperliquidFrontendOrder) -> LiveOrderState:
        role = extract_role_from_cloid(order.cloid)
        return LiveOrderState(
            coin=order.coin,
            side=order.side,
            limit_price=order.limit_price,
            size=order.size,
            reduce_only=order.reduce_only,
            is_trigger=order.is_trigger,
            order_type=order.order_type,
            oid=order.oid,
            cloid=order.cloid,
            status=OrderState.OPEN,
            role=role,
            timestamp=order.timestamp,
            trigger_price=order.trigger_price,
        )

    def _merge_synthetic_clusters(
        self,
        *,
        heatmap_snapshot: HeatmapSnapshot,
        synthetic_snapshot: HeatmapSnapshot,
    ) -> HeatmapSnapshot:
        metadata = dict(heatmap_snapshot.metadata)
        metadata["cluster_source"] = "synthetic-orderbook"
        metadata["cluster_provider"] = synthetic_snapshot.provider
        return heatmap_snapshot.model_copy(
            update={
                "provider": f"{heatmap_snapshot.provider}+synthetic",
                "clusters_above": synthetic_snapshot.clusters_above,
                "clusters_below": synthetic_snapshot.clusters_below,
                "metadata": metadata,
            }
        )


def _channel_age_ms(snapshot: LiveStateSnapshot, channels: tuple[str, ...]) -> float | None:
    matched = [
        timestamp
        for key, timestamp in snapshot.channel_timestamps.items()
        if any(key.startswith(channel) for channel in channels)
    ]
    if not matched:
        return None
    newest = max(matched)
    reference_time = snapshot.captured_at or datetime.now(tz=UTC)
    return max(0.0, (reference_time - newest).total_seconds() * 1000)
