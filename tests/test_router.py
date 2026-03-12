from __future__ import annotations

import json
from pathlib import Path

from dex_llm.features.extractor import FeatureExtractor
from dex_llm.llm.router import HeuristicPlaybookRouter
from dex_llm.models import ClusterSide, MarketFrame, Playbook, TradeSide


def load_sample_frame() -> MarketFrame:
    return MarketFrame.model_validate(json.loads(Path("examples/sample_frame.json").read_text()))


def test_router_returns_magnet_follow_for_sample_frame() -> None:
    frame = load_sample_frame()
    features = FeatureExtractor().extract(frame)

    plan = HeuristicPlaybookRouter().route(frame, features)

    assert plan.playbook == Playbook.MAGNET_FOLLOW
    assert plan.side == TradeSide.LONG


def test_router_returns_sweep_reclaim_when_reclaim_is_visible() -> None:
    frame = load_sample_frame()
    frame.sweep.touched_cluster_side = ClusterSide.ABOVE
    frame.sweep.body_reclaimed = True
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
