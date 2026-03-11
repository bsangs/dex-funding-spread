from __future__ import annotations

import json
from pathlib import Path

from dex_llm.features.extractor import FeatureExtractor
from dex_llm.llm.router import HeuristicPlaybookRouter
from dex_llm.models import AccountState, MarketFrame, PositionState, TradeSide
from dex_llm.risk.policy import RiskPolicy


def load_sample_frame() -> MarketFrame:
    return MarketFrame.model_validate(json.loads(Path("examples/sample_frame.json").read_text()))


def test_risk_policy_sizes_allowed_trade() -> None:
    frame = load_sample_frame()
    plan = HeuristicPlaybookRouter().route(frame, FeatureExtractor().extract(frame))
    account = AccountState(equity=10_000.0, available_margin=10_000.0, max_leverage=10.0)

    assessment = RiskPolicy().assess(plan, account, frame.position)

    assert assessment.allowed is True
    assert assessment.recommended_quantity > 0


def test_risk_policy_blocks_after_two_losses() -> None:
    frame = load_sample_frame()
    frame.position = PositionState(side=TradeSide.FLAT, consecutive_losses_today=2)
    plan = HeuristicPlaybookRouter().route(frame, FeatureExtractor().extract(frame))
    account = AccountState(equity=10_000.0, available_margin=10_000.0, max_leverage=10.0)

    assessment = RiskPolicy().assess(plan, account, frame.position)

    assert assessment.allowed is False
    assert "loss streak" in assessment.reason

