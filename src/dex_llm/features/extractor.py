from __future__ import annotations

from dex_llm.models import Cluster, ClusterShape, ClusterSide, FeatureSnapshot, MarketFrame


class FeatureExtractor:
    def __init__(
        self,
        dominant_ratio_threshold: float = 1.6,
        double_sweep_atr_multiple: float = 0.8,
    ) -> None:
        self.dominant_ratio_threshold = dominant_ratio_threshold
        self.double_sweep_atr_multiple = double_sweep_atr_multiple

    def extract(self, frame: MarketFrame) -> FeatureSnapshot:
        top_above = self._largest_cluster(frame.clusters_above)
        top_below = self._largest_cluster(frame.clusters_below)
        above_total = sum(cluster.size for cluster in frame.clusters_above[:3])
        below_total = sum(cluster.size for cluster in frame.clusters_below[:3])
        dominant_ratio = self._dominant_ratio(above_total, below_total)
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

        notes: list[str] = []
        if dominant_side is not None:
            notes.append(f"dominant_{dominant_side.value}:{dominant_ratio:.2f}")
        if sweep_reclaim_ready:
            notes.append("sweep_reclaim_ready")
        if double_sweep_ready:
            notes.append("double_sweep_ready")
        if frame.map_quality.value != "clean":
            notes.append(f"map_quality:{frame.map_quality.value}")

        return FeatureSnapshot(
            dominant_cluster_side=dominant_side,
            dominant_ratio=dominant_ratio,
            closest_above_distance=closest_above_distance,
            closest_below_distance=closest_below_distance,
            top_above=top_above,
            top_below=top_below,
            sweep_reclaim_ready=sweep_reclaim_ready,
            double_sweep_ready=double_sweep_ready,
            directional_vacuum=directional_vacuum,
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

