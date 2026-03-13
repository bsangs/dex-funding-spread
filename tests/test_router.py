from __future__ import annotations

import json
from pathlib import Path

from dex_llm.features.extractor import FeatureExtractor
from dex_llm.llm.router import HeuristicPlaybookRouter
from dex_llm.models import ClusterShape, ClusterSide, MarketFrame, Playbook, TradeSide


def load_sample_frame() -> MarketFrame:
    return MarketFrame.model_validate(json.loads(Path("examples/sample_frame.json").read_text()))


def test_router_prefers_magnet_follow_for_directional_vacuum_sample() -> None:
    frame = load_sample_frame()
    features = FeatureExtractor().extract(frame)

    plan = HeuristicPlaybookRouter().route(frame, features)

    assert plan.playbook == Playbook.MAGNET_FOLLOW
    assert plan.side == TradeSide.LONG
    assert plan.touch_confidence > 0.5
    assert plan.expected_touch_minutes == 60


def test_router_returns_cluster_fade_only_when_clusters_are_balanced() -> None:
    frame = load_sample_frame()
    frame.clusters_above[0].shape = ClusterShape.STAIRCASE
    frame.clusters_above[0].price = frame.current_price + frame.atr
    frame.clusters_above[0].size = 122.0
    frame.clusters_above[1].size = 110.0
    frame.clusters_above[2].size = 95.0
    frame.clusters_below[0].price = frame.current_price - frame.atr
    frame.clusters_below[0].size = 120.0
    frame.clusters_below[1].size = 108.0
    frame.clusters_below[2].size = 96.0
    features = FeatureExtractor().extract(frame)

    plan = HeuristicPlaybookRouter().route(frame, features)

    assert features.cluster_fade_ready is True
    assert plan.playbook == Playbook.CLUSTER_FADE
    assert plan.side == TradeSide.SHORT
    assert plan.resting_orders == []
    assert plan.touch_confidence > 0.5


def test_router_prefers_double_sweep_before_cluster_fade() -> None:
    frame = load_sample_frame()
    frame.clusters_above[0].shape = ClusterShape.STAIRCASE
    frame.clusters_above[0].price = frame.current_price + (frame.atr * 0.5)
    frame.clusters_above[0].size = 118.0
    frame.clusters_above[1].size = 102.0
    frame.clusters_above[2].size = 90.0
    frame.clusters_below[0].price = frame.current_price - (frame.atr * 0.5)
    frame.clusters_below[0].size = 120.0
    frame.clusters_below[1].size = 105.0
    frame.clusters_below[2].size = 92.0
    features = FeatureExtractor().extract(frame)

    plan = HeuristicPlaybookRouter().route(frame, features)

    assert features.cluster_fade_ready is True
    assert features.double_sweep_ready is True
    assert plan.playbook == Playbook.NO_TRADE
    assert plan.side == TradeSide.FLAT


def test_router_returns_sweep_reclaim_when_reclaim_is_visible() -> None:
    frame = load_sample_frame()
    frame.sweep.touched_cluster_side = ClusterSide.ABOVE
    frame.sweep.body_reclaimed = True
    frame.sweep.cluster_price = max(cluster.price for cluster in frame.clusters_above)
    features = FeatureExtractor().extract(frame)

    plan = HeuristicPlaybookRouter().route(frame, features)

    assert plan.playbook == Playbook.SWEEP_RECLAIM
    assert plan.side == TradeSide.SHORT


def test_router_returns_no_trade_when_kill_switch_is_active() -> None:
    frame = load_sample_frame()
    frame.kill_switch.allow_new_trades = False
    frame.kill_switch.reasons = ["synthetic heatmap fallback active"]
    features = FeatureExtractor().extract(frame)

    plan = HeuristicPlaybookRouter().route(frame, features)

    assert plan.playbook == Playbook.NO_TRADE
    assert "synthetic heatmap" in plan.reason
