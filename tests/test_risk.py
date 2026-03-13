from __future__ import annotations

import json
from pathlib import Path

from dex_llm.features.extractor import FeatureExtractor
from dex_llm.llm.router import HeuristicPlaybookRouter
from dex_llm.models import (
    AccountState,
    MarketFrame,
    Playbook,
    PositionState,
    RestingOrderPlan,
    TradePlan,
    TradeSide,
)
from dex_llm.risk.policy import RiskPolicy


def load_sample_frame() -> MarketFrame:
    return MarketFrame.model_validate(json.loads(Path("examples/sample_frame.json").read_text()))


def test_risk_policy_sizes_allowed_trade() -> None:
    frame = load_sample_frame()
    plan = HeuristicPlaybookRouter().route(frame, FeatureExtractor().extract(frame))
    account = AccountState(equity=10_000.0, available_margin=10_000.0, max_leverage=10.0)

    assessment = RiskPolicy().assess(plan, account, frame.position, frame.kill_switch)

    assert assessment.allowed is True
    assert assessment.recommended_quantity > 0
    assert "risk checks passed" in assessment.reason


def test_risk_policy_blocks_after_two_losses() -> None:
    frame = load_sample_frame()
    frame.position = PositionState(side=TradeSide.FLAT, consecutive_losses_today=2)
    plan = HeuristicPlaybookRouter().route(frame, FeatureExtractor().extract(frame))
    account = AccountState(equity=10_000.0, available_margin=10_000.0, max_leverage=10.0)

    assessment = RiskPolicy().assess(plan, account, frame.position, frame.kill_switch)

    assert assessment.allowed is False
    assert "loss streak" in assessment.reason


def test_risk_policy_blocks_when_kill_switch_is_active() -> None:
    frame = load_sample_frame()
    frame.kill_switch.allow_new_trades = False
    frame.kill_switch.reasons = ["private account state unavailable"]
    plan = HeuristicPlaybookRouter().route(frame, FeatureExtractor().extract(frame))
    account = AccountState(equity=10_000.0, available_margin=10_000.0, max_leverage=10.0)

    assessment = RiskPolicy().assess(plan, account, frame.position, frame.kill_switch)

    assert assessment.allowed is False
    assert "private account state unavailable" in assessment.reason


def test_risk_policy_allocates_asymmetric_cluster_fade_quantities() -> None:
    account = AccountState(equity=10_000.0, available_margin=10_000.0, max_leverage=10.0)
    plan = TradePlan(
        playbook=Playbook.CLUSTER_FADE,
        side=TradeSide.FLAT,
        entry_band=(0.0, 0.0),
        invalid_if=0.0,
        tp1=0.0,
        tp2=0.0,
        ttl_min=30,
        reason="arm both bands",
        resting_orders=[
            RestingOrderPlan(
                side=TradeSide.LONG,
                entry_band=(3998.0, 4002.0),
                invalid_if=3970.0,
                tp1=4025.0,
                tp2=4040.0,
                ttl_min=30,
                reason="lower long fade",
            ),
            RestingOrderPlan(
                side=TradeSide.SHORT,
                entry_band=(4048.0, 4052.0),
                invalid_if=4075.0,
                tp1=4025.0,
                tp2=4010.0,
                ttl_min=30,
                reason="upper short fade",
            ),
        ],
    )

    assessment = RiskPolicy().assess(plan, account, PositionState(), None)

    assert assessment.allowed is True
    assert len(assessment.resting_order_quantities) == 2
    assert assessment.resting_order_quantities[0] > assessment.resting_order_quantities[1]
