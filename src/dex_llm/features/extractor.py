from __future__ import annotations

from dex_llm.models import (
    Cluster,
    ClusterShape,
    ClusterSide,
    EntryCandidate,
    FeatureSnapshot,
    MarketFrame,
    TradeSide,
)


class FeatureExtractor:
    def __init__(
        self,
        dominant_ratio_threshold: float = 1.6,
        double_sweep_atr_multiple: float = 0.8,
        cluster_fade_balance_threshold: float = 0.65,
        cluster_fade_distance_atr_multiple: float = 1.25,
    ) -> None:
        self.dominant_ratio_threshold = dominant_ratio_threshold
        self.double_sweep_atr_multiple = double_sweep_atr_multiple
        self.cluster_fade_balance_threshold = cluster_fade_balance_threshold
        self.cluster_fade_distance_atr_multiple = cluster_fade_distance_atr_multiple

    def extract(self, frame: MarketFrame) -> FeatureSnapshot:
        top_above = self._largest_cluster(frame.clusters_above)
        top_below = self._largest_cluster(frame.clusters_below)
        above_total = sum(cluster.size for cluster in frame.clusters_above[:3])
        below_total = sum(cluster.size for cluster in frame.clusters_below[:3])
        dominant_ratio = self._dominant_ratio(above_total, below_total)
        cluster_balance_ratio = self._cluster_balance_ratio(above_total, below_total)
        dominant_side = self._dominant_side(above_total, below_total, dominant_ratio)
        closest_above_distance = self._closest_distance(frame.current_price, frame.clusters_above)
        closest_below_distance = self._closest_distance(frame.current_price, frame.clusters_below)
        directional_vacuum = self._directional_vacuum(
            frame=frame,
            dominant_side=dominant_side,
            top_above=top_above,
            top_below=top_below,
        )
        sweep_reclaim_ready = (
            frame.sweep.touched_cluster_side is not None and frame.sweep.body_reclaimed
        )
        double_sweep_ready = (
            closest_above_distance is not None
            and closest_below_distance is not None
            and closest_above_distance <= frame.atr * self.double_sweep_atr_multiple
            and closest_below_distance <= frame.atr * self.double_sweep_atr_multiple
        )
        cluster_fade_ready = (
            top_above is not None
            and top_below is not None
            and cluster_balance_ratio >= self.cluster_fade_balance_threshold
            and closest_above_distance is not None
            and closest_below_distance is not None
            and closest_above_distance <= frame.atr * self.cluster_fade_distance_atr_multiple
            and closest_below_distance <= frame.atr * self.cluster_fade_distance_atr_multiple
            and not directional_vacuum
        )
        entry_candidates = self._entry_candidates(
            frame=frame,
            top_above=top_above,
            top_below=top_below,
        )

        notes: list[str] = []
        if dominant_side is not None:
            notes.append(f"dominant_{dominant_side.value}:{dominant_ratio:.2f}")
        notes.append(f"cluster_balance:{cluster_balance_ratio:.2f}")
        if sweep_reclaim_ready:
            notes.append("sweep_reclaim_ready")
        if double_sweep_ready:
            notes.append("double_sweep_ready")
        if cluster_fade_ready:
            notes.append("cluster_fade_ready")
        if entry_candidates:
            notes.append(f"entry_candidates:{len(entry_candidates)}")
        if frame.map_quality.value != "clean":
            notes.append(f"map_quality:{frame.map_quality.value}")

        return FeatureSnapshot(
            dominant_cluster_side=dominant_side,
            dominant_ratio=dominant_ratio,
            cluster_balance_ratio=cluster_balance_ratio,
            closest_above_distance=closest_above_distance,
            closest_below_distance=closest_below_distance,
            top_above=top_above,
            top_below=top_below,
            sweep_reclaim_ready=sweep_reclaim_ready,
            double_sweep_ready=double_sweep_ready,
            cluster_fade_ready=cluster_fade_ready,
            directional_vacuum=directional_vacuum,
            entry_candidates=entry_candidates,
            notes=notes,
        )

    def _largest_cluster(self, clusters: list[Cluster]) -> Cluster | None:
        if not clusters:
            return None
        return max(clusters, key=lambda cluster: cluster.size)

    def _dominant_ratio(self, above_total: float, below_total: float) -> float:
        smaller = min(above_total, below_total)
        if smaller == 0:
            return float("inf")
        return max(above_total, below_total) / smaller

    def _dominant_side(
        self,
        above_total: float,
        below_total: float,
        dominant_ratio: float,
    ) -> ClusterSide | None:
        if dominant_ratio < self.dominant_ratio_threshold:
            return None
        if above_total > below_total:
            return ClusterSide.ABOVE
        if below_total > above_total:
            return ClusterSide.BELOW
        return None

    def _cluster_balance_ratio(self, above_total: float, below_total: float) -> float:
        larger = max(above_total, below_total)
        if larger == 0:
            return 0.0
        return min(above_total, below_total) / larger

    def _closest_distance(self, current_price: float, clusters: list[Cluster]) -> float | None:
        distances: list[float] = []
        for cluster in clusters:
            if cluster.side == ClusterSide.ABOVE and cluster.price > current_price:
                distances.append(cluster.price - current_price)
            if cluster.side == ClusterSide.BELOW and cluster.price < current_price:
                distances.append(current_price - cluster.price)
        if not distances:
            return None
        return min(distances)

    def _directional_vacuum(
        self,
        frame: MarketFrame,
        dominant_side: ClusterSide | None,
        top_above: Cluster | None,
        top_below: Cluster | None,
    ) -> bool:
        if dominant_side == ClusterSide.ABOVE and top_above is not None:
            return (
                top_above.shape == ClusterShape.SINGLE_WALL
                and top_above.price - frame.current_price >= frame.atr * 0.5
            )
        if dominant_side == ClusterSide.BELOW and top_below is not None:
            return (
                top_below.shape == ClusterShape.SINGLE_WALL
                and frame.current_price - top_below.price >= frame.atr * 0.5
            )
        return False

    def _entry_candidates(
        self,
        *,
        frame: MarketFrame,
        top_above: Cluster | None,
        top_below: Cluster | None,
    ) -> list[EntryCandidate]:
        candidates: list[EntryCandidate] = []
        top_clusters = frame.clusters_above[:3] + frame.clusters_below[:3]
        total_cluster_size = sum(cluster.size for cluster in top_clusters)
        htf_levels = self._higher_timeframe_levels(frame)
        band_half_width = max(frame.atr * 0.08, 1.0)

        if top_below is not None:
            candidates.append(
                self._cluster_candidate(
                    side=TradeSide.LONG,
                    cluster=top_below,
                    current_price=frame.current_price,
                    atr=frame.atr,
                    total_cluster_size=total_cluster_size,
                    band_half_width=band_half_width,
                    htf_levels=htf_levels,
                )
            )
        if top_above is not None:
            candidates.append(
                self._cluster_candidate(
                    side=TradeSide.SHORT,
                    cluster=top_above,
                    current_price=frame.current_price,
                    atr=frame.atr,
                    total_cluster_size=total_cluster_size,
                    band_half_width=band_half_width,
                    htf_levels=htf_levels,
                )
            )

        if frame.sweep.cluster_price is not None and frame.sweep.touched_cluster_side is not None:
            sweep_side = (
                TradeSide.SHORT
                if frame.sweep.touched_cluster_side == ClusterSide.ABOVE
                else TradeSide.LONG
            )
            distance_atr = (
                abs(frame.current_price - frame.sweep.cluster_price)
                / max(frame.atr, 1.0)
            )
            alignment = self._htf_alignment_score(
                price=frame.sweep.cluster_price,
                atr=frame.atr,
                htf_levels=htf_levels,
            )
            candidates.append(
                EntryCandidate(
                    side=sweep_side,
                    entry_band=(
                        frame.sweep.cluster_price - band_half_width,
                        frame.sweep.cluster_price + band_half_width,
                    ),
                    anchor_price=frame.sweep.cluster_price,
                    anchor_type="sweep_retest",
                    distance_atr=round(distance_atr, 2),
                    persistence_score=min(1.0, round(0.6 + alignment, 2)),
                    expected_wait_minutes=self._expected_wait_minutes(
                        distance_atr=distance_atr,
                        persistence_score=min(1.0, 0.6 + alignment),
                    ),
                    reason="wait for a retest of the swept liquidity wall instead of chasing",
                )
            )

        candidates.sort(
            key=lambda candidate: (
                candidate.persistence_score,
                candidate.distance_atr,
            ),
            reverse=True,
        )
        return candidates[:3]

    def _cluster_candidate(
        self,
        *,
        side: TradeSide,
        cluster: Cluster,
        current_price: float,
        atr: float,
        total_cluster_size: float,
        band_half_width: float,
        htf_levels: list[float],
    ) -> EntryCandidate:
        distance_atr = abs(current_price - cluster.price) / max(atr, 1.0)
        size_share = (cluster.size / total_cluster_size) if total_cluster_size > 0 else 0.0
        shape_bonus = 0.12 if cluster.shape == ClusterShape.SINGLE_WALL else 0.05
        alignment = self._htf_alignment_score(price=cluster.price, atr=atr, htf_levels=htf_levels)
        persistence_score = min(1.0, round(0.35 + size_share + shape_bonus + alignment, 2))
        return EntryCandidate(
            side=side,
            entry_band=(
                cluster.price - band_half_width,
                cluster.price + band_half_width,
            ),
            anchor_price=cluster.price,
            anchor_type="cluster_retest",
            distance_atr=round(distance_atr, 2),
            persistence_score=persistence_score,
            expected_wait_minutes=self._expected_wait_minutes(
                distance_atr=distance_atr,
                persistence_score=persistence_score,
            ),
            reason=(
                "prefer a retest of the persistent liquidation wall "
                "instead of the current price"
            ),
        )

    @staticmethod
    def _higher_timeframe_levels(frame: MarketFrame) -> list[float]:
        levels: list[float] = []
        if frame.candles_1h:
            window_1h = frame.candles_1h[-24:]
            levels.extend(
                [
                    max(candle.high for candle in window_1h),
                    min(candle.low for candle in window_1h),
                ]
            )
        if frame.candles_4h:
            window_4h = frame.candles_4h[-12:]
            levels.extend(
                [
                    max(candle.high for candle in window_4h),
                    min(candle.low for candle in window_4h),
                ]
            )
        return levels

    @staticmethod
    def _htf_alignment_score(
        *,
        price: float,
        atr: float,
        htf_levels: list[float],
    ) -> float:
        if not htf_levels:
            return 0.0
        tolerance = max(atr * 0.35, 1.0)
        matched = sum(1 for level in htf_levels if abs(level - price) <= tolerance)
        return min(0.25, matched * 0.08)

    @staticmethod
    def _expected_wait_minutes(
        *,
        distance_atr: float,
        persistence_score: float,
    ) -> int:
        estimate = int(25 + distance_atr * 18 + (1 - persistence_score) * 25)
        return max(20, min(360, estimate))
