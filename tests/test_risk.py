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
    assert "side-based sizing checks passed" in assessment.reason


def test_risk_policy_does_not_block_after_two_losses() -> None:
    frame = load_sample_frame()
    frame.position = PositionState(side=TradeSide.FLAT, consecutive_losses_today=2)
    plan = HeuristicPlaybookRouter().route(frame, FeatureExtractor().extract(frame))
    account = AccountState(equity=10_000.0, available_margin=10_000.0, max_leverage=10.0)

    assessment = RiskPolicy().assess(plan, account, frame.position, frame.kill_switch)

    assert assessment.allowed is True
    assert assessment.recommended_quantity > 0


def test_risk_policy_blocks_when_kill_switch_is_active() -> None:
    frame = load_sample_frame()
    frame.kill_switch.allow_new_trades = False
    frame.kill_switch.reasons = ["private account state unavailable"]
    plan = HeuristicPlaybookRouter().route(frame, FeatureExtractor().extract(frame))
    account = AccountState(equity=10_000.0, available_margin=10_000.0, max_leverage=10.0)

    assessment = RiskPolicy().assess(plan, account, frame.position, frame.kill_switch)

    assert assessment.allowed is False
    assert "private account state unavailable" in assessment.reason


def test_risk_policy_allocates_more_notional_to_longs_than_shorts() -> None:
    account = AccountState(equity=10_000.0, available_margin=10_000.0, max_leverage=10.0)
    long_plan = TradePlan(
        playbook=Playbook.MAGNET_FOLLOW,
        side=TradeSide.LONG,
        entry_band=(3998.0, 4002.0),
        invalid_if=3970.0,
        tp1=4025.0,
        tp2=4040.0,
        ttl_min=30,
        reason="long entry",
    )
    short_plan = TradePlan(
        playbook=Playbook.MAGNET_FOLLOW,
        side=TradeSide.SHORT,
        entry_band=(4048.0, 4052.0),
        invalid_if=4075.0,
        tp1=4025.0,
        tp2=4010.0,
        ttl_min=30,
        reason="short entry",
    )

    long_assessment = RiskPolicy().assess(long_plan, account, PositionState(), None)
    short_assessment = RiskPolicy().assess(short_plan, account, PositionState(), None)

    assert long_assessment.allowed is True
    assert short_assessment.allowed is True
    assert long_assessment.recommended_notional == 90_000.0
    assert short_assessment.recommended_notional == 40_000.0
    assert long_assessment.recommended_quantity > short_assessment.recommended_quantity
