from __future__ import annotations

from typing import Protocol

from dex_llm.models import (
    ClusterSide,
    FeatureSnapshot,
    MapQuality,
    MarketFrame,
    OrderRole,
    Playbook,
    TradePlan,
    TradeSide,
)


class RouterProtocol(Protocol):
    def route(
        self,
        frame: MarketFrame,
        features: FeatureSnapshot,
        previous_plan: TradePlan | None = None,
    ) -> TradePlan: ...


class HeuristicPlaybookRouter:
    def __init__(self, dominant_ratio_threshold: float = 1.6) -> None:
        self.dominant_ratio_threshold = dominant_ratio_threshold

    def route(
        self,
        frame: MarketFrame,
        features: FeatureSnapshot,
        previous_plan: TradePlan | None = None,
    ) -> TradePlan:
        _ = previous_plan
        if not frame.kill_switch.allow_new_trades and frame.position.side == TradeSide.FLAT:
            reason = (
                frame.kill_switch.reasons[0]
                if frame.kill_switch.reasons
                else "kill switch active"
            )
            return self._flat_plan(reason)

        if frame.position.side != TradeSide.FLAT:
            return self._manage_open_position(frame)

        has_pending_entry = any(
            order.coin == frame.symbol and not order.reduce_only
            for order in frame.position.active_orders
        )
        if has_pending_entry or frame.position.entries_blocked_reduce_only:
            return self._flat_plan("existing entry workflow detected; reconcile open orders first")

        if frame.map_quality == MapQuality.DIRTY:
            return self._flat_plan("map is too noisy for a high-confidence playbook")

        if features.sweep_reclaim_ready and frame.sweep.touched_cluster_side is not None:
            return self._sweep_reclaim(frame)

        if (
            features.dominant_cluster_side is not None
            and features.dominant_ratio >= self.dominant_ratio_threshold
            and features.directional_vacuum
        ):
            return self._magnet_follow(frame, features.dominant_cluster_side)

        if features.double_sweep_ready:
            return self._flat_plan("double sweep watch only; wait for one side to resolve first")

        if features.cluster_fade_ready:
            return self._cluster_fade(frame)

        return self._flat_plan("cluster map does not offer a clean directional or reclaim setup")

    def _magnet_follow(self, frame: MarketFrame, dominant_side: ClusterSide) -> TradePlan:
        if dominant_side == ClusterSide.ABOVE:
            top_cluster = max(frame.clusters_above, key=lambda cluster: cluster.size)
            return TradePlan(
                playbook=Playbook.MAGNET_FOLLOW,
                side=TradeSide.LONG,
                entry_band=(
                    frame.current_price - frame.atr * 0.1,
                    frame.current_price + frame.atr * 0.1,
                ),
                invalid_if=frame.current_price - frame.atr * 0.6,
                tp1=frame.current_price + frame.atr * 0.7,
                tp2=top_cluster.price,
                ttl_min=20,
                reason="upside cluster dominates and sits behind a clean vacuum",
                touch_confidence=0.72,
                expected_touch_minutes=20,
            )

        top_cluster = max(frame.clusters_below, key=lambda cluster: cluster.size)
        return TradePlan(
            playbook=Playbook.MAGNET_FOLLOW,
            side=TradeSide.SHORT,
            entry_band=(
                frame.current_price - frame.atr * 0.1,
                frame.current_price + frame.atr * 0.1,
            ),
            invalid_if=frame.current_price + frame.atr * 0.6,
            tp1=frame.current_price - frame.atr * 0.7,
            tp2=top_cluster.price,
            ttl_min=20,
            reason="downside cluster dominates and sits behind a clean vacuum",
            touch_confidence=0.72,
            expected_touch_minutes=20,
        )

    def _sweep_reclaim(self, frame: MarketFrame) -> TradePlan:
        if frame.sweep.touched_cluster_side == ClusterSide.ABOVE:
            if frame.sweep.cluster_price is None:
                return self._flat_plan("sweep reclaim missing touched cluster price")
            swept_price = frame.sweep.cluster_price
            return TradePlan(
                playbook=Playbook.SWEEP_RECLAIM,
                side=TradeSide.SHORT,
                entry_band=(
                    frame.current_price - frame.atr * 0.08,
                    frame.current_price + frame.atr * 0.05,
                ),
                invalid_if=swept_price + frame.atr * 0.15,
                tp1=frame.current_price - frame.atr * 0.6,
                tp2=frame.current_price - frame.atr * 1.2,
                ttl_min=12,
                reason="price swept the upper cluster and reclaimed back inside the prior range",
                touch_confidence=0.82,
                expected_touch_minutes=12,
            )

        if frame.sweep.cluster_price is None:
            return self._flat_plan("sweep reclaim missing touched cluster price")
        swept_price = frame.sweep.cluster_price
        return TradePlan(
            playbook=Playbook.SWEEP_RECLAIM,
            side=TradeSide.LONG,
            entry_band=(
                frame.current_price - frame.atr * 0.05,
                frame.current_price + frame.atr * 0.08,
            ),
            invalid_if=swept_price - frame.atr * 0.15,
            tp1=frame.current_price + frame.atr * 0.6,
            tp2=frame.current_price + frame.atr * 1.2,
            ttl_min=12,
            reason="price swept the lower cluster and reclaimed back inside the prior range",
            touch_confidence=0.82,
            expected_touch_minutes=12,
        )

    def _flat_plan(self, reason: str) -> TradePlan:
        return TradePlan(
            playbook=Playbook.NO_TRADE,
            side=TradeSide.FLAT,
            entry_band=(0.0, 0.0),
            invalid_if=0.0,
            tp1=0.0,
            tp2=0.0,
            ttl_min=0,
            reason=reason,
            touch_confidence=0.0,
            expected_touch_minutes=None,
        )

    def _manage_open_position(self, frame: MarketFrame) -> TradePlan:
        side = frame.position.side
        entry_reference = frame.current_price
        band_half_width = max(frame.atr * 0.05, 1.0)
        existing_exits = {
            order.role: (order.trigger_price or order.limit_price)
            for order in frame.position.active_orders
            if order.coin == frame.symbol and order.reduce_only
        }
        if side == TradeSide.LONG:
            stop_price = existing_exits.get(
                OrderRole.STOP_LOSS,
                frame.current_price - frame.atr * 0.45,
            )
            tp1 = max(
                existing_exits.get(OrderRole.TAKE_PROFIT_1, frame.current_price + frame.atr * 0.6),
                frame.current_price + frame.atr * 0.3,
            )
            tp2 = max(
                existing_exits.get(OrderRole.TAKE_PROFIT_2, frame.current_price + frame.atr * 1.2),
                tp1 + frame.atr * 0.4,
            )
        else:
            stop_price = existing_exits.get(
                OrderRole.STOP_LOSS,
                frame.current_price + frame.atr * 0.45,
            )
            tp1 = min(
                existing_exits.get(OrderRole.TAKE_PROFIT_1, frame.current_price - frame.atr * 0.6),
                frame.current_price - frame.atr * 0.3,
            )
            tp2 = min(
                existing_exits.get(OrderRole.TAKE_PROFIT_2, frame.current_price - frame.atr * 1.2),
                tp1 - frame.atr * 0.4,
            )
        return TradePlan(
            playbook=Playbook.MAGNET_FOLLOW,
            side=side,
            entry_band=(
                entry_reference - band_half_width,
                entry_reference + band_half_width,
            ),
            invalid_if=stop_price,
            tp1=tp1,
            tp2=tp2,
            ttl_min=5,
            reason="fallback position review keeps the open trade and updates progressive exits",
            touch_confidence=1.0,
            expected_touch_minutes=5,
        )

    def _cluster_fade(self, frame: MarketFrame) -> TradePlan:
        upper_cluster = max(frame.clusters_above, key=lambda cluster: cluster.size)
        lower_cluster = max(frame.clusters_below, key=lambda cluster: cluster.size)
        band_half_width = max(frame.atr * 0.05, 1.0)
        if lower_cluster.size >= upper_cluster.size:
            return TradePlan(
                playbook=Playbook.CLUSTER_FADE,
                side=TradeSide.LONG,
                entry_band=(
                    lower_cluster.price - band_half_width,
                    lower_cluster.price + band_half_width,
                ),
                invalid_if=lower_cluster.price - frame.atr * 0.25,
                tp1=frame.current_price,
                tp2=min(upper_cluster.price, frame.current_price + frame.atr),
                ttl_min=30,
                reason="fade the lower wall with a single long limit order",
                touch_confidence=0.68,
                expected_touch_minutes=30,
            )
        return TradePlan(
            playbook=Playbook.CLUSTER_FADE,
            side=TradeSide.SHORT,
            entry_band=(
                upper_cluster.price - band_half_width,
                upper_cluster.price + band_half_width,
            ),
            invalid_if=upper_cluster.price + frame.atr * 0.25,
            tp1=frame.current_price,
            tp2=max(lower_cluster.price, frame.current_price - frame.atr),
            ttl_min=30,
            reason="fade the upper wall with a single short limit order",
            touch_confidence=0.68,
            expected_touch_minutes=30,
        )
