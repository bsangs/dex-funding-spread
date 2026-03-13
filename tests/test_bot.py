from __future__ import annotations

from datetime import UTC, datetime, timedelta

from dex_llm.bot import BotRuntime, StrategyState
from dex_llm.models import (
    FeatureSnapshot,
    Playbook,
    PositionState,
    RestingOrderPlan,
    RiskAssessment,
    TradePlan,
    TradeSide,
)


def test_bot_runtime_expires_directional_entries_after_ttl() -> None:
    runtime = object.__new__(BotRuntime)
    plan = TradePlan(
        playbook=Playbook.MAGNET_FOLLOW,
        side=TradeSide.LONG,
        entry_band=(2100.0, 2102.0),
        invalid_if=2090.0,
        tp1=2110.0,
        tp2=2120.0,
        ttl_min=20,
        reason="follow upside liquidity",
    )
    state = StrategyState(
        frame=None,  # type: ignore[arg-type]
        features=FeatureSnapshot(
            dominant_cluster_side=None,
            dominant_ratio=1.0,
            cluster_balance_ratio=1.0,
            closest_above_distance=None,
            closest_below_distance=None,
            top_above=None,
            top_below=None,
            sweep_reclaim_ready=False,
            double_sweep_ready=False,
            cluster_fade_ready=False,
            directional_vacuum=False,
            notes=[],
        ),
        plan=plan,
        risk=RiskAssessment(allowed=True, reason="ok"),
        updated_at=datetime.now(tz=UTC) - timedelta(minutes=21),
    )

    effective = runtime._effective_plan(
        strategy_state=state,
        position=PositionState(),
        current_price=2101.0,
        now=datetime.now(tz=UTC),
    )

    assert effective.playbook == Playbook.NO_TRADE
    assert "expired" in effective.reason


def test_bot_runtime_keeps_only_non_invalidated_resting_orders() -> None:
    runtime = object.__new__(BotRuntime)
    plan = TradePlan(
        playbook=Playbook.CLUSTER_FADE,
        side=TradeSide.FLAT,
        entry_band=(0.0, 0.0),
        invalid_if=0.0,
        tp1=0.0,
        tp2=0.0,
        ttl_min=30,
        reason="rest both sides",
        resting_orders=[
            RestingOrderPlan(
                side=TradeSide.LONG,
                entry_band=(2090.0, 2092.0),
                invalid_if=2085.0,
                tp1=2100.0,
                tp2=2110.0,
                ttl_min=30,
                reason="long wall",
            ),
            RestingOrderPlan(
                side=TradeSide.SHORT,
                entry_band=(2120.0, 2122.0),
                invalid_if=2128.0,
                tp1=2110.0,
                tp2=2100.0,
                ttl_min=30,
                reason="short wall",
            ),
        ],
    )
    state = StrategyState(
        frame=None,  # type: ignore[arg-type]
        features=FeatureSnapshot(
            dominant_cluster_side=None,
            dominant_ratio=1.0,
            cluster_balance_ratio=1.0,
            closest_above_distance=None,
            closest_below_distance=None,
            top_above=None,
            top_below=None,
            sweep_reclaim_ready=False,
            double_sweep_ready=False,
            cluster_fade_ready=True,
            directional_vacuum=False,
            notes=[],
        ),
        plan=plan,
        risk=RiskAssessment(allowed=True, reason="ok"),
        updated_at=datetime.now(tz=UTC),
    )

    effective = runtime._effective_plan(
        strategy_state=state,
        position=PositionState(),
        current_price=2084.5,
        now=datetime.now(tz=UTC),
    )

    assert effective.playbook == Playbook.CLUSTER_FADE
    assert len(effective.resting_orders) == 1
    assert effective.resting_orders[0].side == TradeSide.SHORT
