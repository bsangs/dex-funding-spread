from __future__ import annotations

from typing import Protocol

from dex_llm.models import (
    ClusterSide,
    FeatureSnapshot,
    MapQuality,
    MarketFrame,
    Playbook,
    RestingOrderPlan,
    TradePlan,
    TradeSide,
)


class RouterProtocol(Protocol):
    def route(self, frame: MarketFrame, features: FeatureSnapshot) -> TradePlan: ...


class HeuristicPlaybookRouter:
    def __init__(self, dominant_ratio_threshold: float = 1.6) -> None:
        self.dominant_ratio_threshold = dominant_ratio_threshold

    def route(self, frame: MarketFrame, features: FeatureSnapshot) -> TradePlan:
        if not frame.kill_switch.allow_new_trades:
            reason = (
                frame.kill_switch.reasons[0]
                if frame.kill_switch.reasons
                else "kill switch active"
            )
            return self._flat_plan(reason)

        if frame.position.side != TradeSide.FLAT:
            return self._flat_plan("existing position detected; no averaging down")

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

        if frame.clusters_above and frame.clusters_below:
            return self._cluster_fade(frame)

        if (
            features.dominant_cluster_side is not None
            and features.dominant_ratio >= self.dominant_ratio_threshold
            and features.directional_vacuum
        ):
            return self._magnet_follow(frame, features.dominant_cluster_side)

        if features.double_sweep_ready:
            return TradePlan(
                playbook=Playbook.DOUBLE_SWEEP,
                side=TradeSide.FLAT,
                entry_band=(frame.current_price, frame.current_price),
                invalid_if=frame.current_price,
                tp1=frame.current_price,
                tp2=frame.current_price,
                ttl_min=20,
                reason=(
                    "both nearby clusters are active, so wait for the first sweep "
                    "and only trade the second move"
                ),
            )

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
        )

    def _cluster_fade(self, frame: MarketFrame) -> TradePlan:
        upper_cluster = max(frame.clusters_above, key=lambda cluster: cluster.size)
        lower_cluster = max(frame.clusters_below, key=lambda cluster: cluster.size)
        band_half_width = max(frame.atr * 0.05, 1.0)

        lower_long = RestingOrderPlan(
            side=TradeSide.LONG,
            entry_band=(
                lower_cluster.price - band_half_width,
                lower_cluster.price + band_half_width,
            ),
            invalid_if=lower_cluster.price - frame.atr * 0.25,
            tp1=frame.current_price,
            tp2=min(upper_cluster.price, frame.current_price + frame.atr),
            ttl_min=30,
            reason="rest a long fade at the lower liquidation wall",
            cluster_price=lower_cluster.price,
        )
        upper_short = RestingOrderPlan(
            side=TradeSide.SHORT,
            entry_band=(
                upper_cluster.price - band_half_width,
                upper_cluster.price + band_half_width,
            ),
            invalid_if=upper_cluster.price + frame.atr * 0.25,
            tp1=frame.current_price,
            tp2=max(lower_cluster.price, frame.current_price - frame.atr),
            ttl_min=30,
            reason="rest a short fade at the upper liquidation wall",
            cluster_price=upper_cluster.price,
        )
        return TradePlan(
            playbook=Playbook.CLUSTER_FADE,
            side=TradeSide.FLAT,
            entry_band=(0.0, 0.0),
            invalid_if=0.0,
            tp1=0.0,
            tp2=0.0,
            ttl_min=30,
            reason="arm both upper-short and lower-long fade bands around major clusters",
            resting_orders=[lower_long, upper_short],
        )
