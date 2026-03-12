from __future__ import annotations

from dex_llm.models import (
    ClusterSide,
    FeatureSnapshot,
    MapQuality,
    MarketFrame,
    Playbook,
    TradePlan,
    TradeSide,
)


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
            swept_price = max(cluster.price for cluster in frame.clusters_above)
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

        swept_price = min(cluster.price for cluster in frame.clusters_below)
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
