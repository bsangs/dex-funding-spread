from __future__ import annotations

from datetime import UTC, datetime, timedelta

from dex_llm.bot import BotRuntime, StrategyState
from dex_llm.models import (
    AccountState,
    FeatureSnapshot,
    HyperliquidUserFill,
    LiveOrderState,
    OrderRole,
    OrderState,
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


def test_bot_runtime_expires_resting_order_when_touch_window_passes() -> None:
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
                touch_confidence=0.8,
                expected_touch_minutes=5,
                reason="long wall",
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
        updated_at=datetime.now(tz=UTC) - timedelta(minutes=6),
    )

    effective = runtime._effective_plan(
        strategy_state=state,
        position=PositionState(),
        current_price=2091.0,
        now=datetime.now(tz=UTC),
    )

    assert effective.playbook == Playbook.NO_TRADE
    assert "invalidated" in effective.reason or "expired" in effective.reason


def test_bot_runtime_raises_when_live_leverage_preflight_is_invalid() -> None:
    runtime = object.__new__(BotRuntime)
    runtime.live = True
    runtime.symbol = "ETH"

    class StubExecutor:
        target_leverage = 10
        margin_mode = "isolated"

        def build_orders_from_plan(self, **_: object) -> list[object]:
            return [type("Order", (), {"reduce_only": False})()]

        def apply_leverage_preflight(self, **_: object) -> RiskAssessment:
            return type(
                "ValidationResult",
                (),
                {
                    "valid": False,
                    "reason": "leverage preflight update failed: rounding",
                    "model_dump": lambda self, mode="json": {
                        "valid": False,
                        "reason": "leverage preflight update failed: rounding",
                    },
                },
            )()

    runtime.executor = StubExecutor()

    try:
        runtime._sync_orders(
            snapshot=type(
                "Snapshot",
                (),
                {
                    "order_book": type(
                        "Book",
                        (),
                        {
                            "best_bid": 2100.0,
                            "best_ask": 2101.0,
                        },
                    )(),
                    "active_asset_ctx": type("Ctx", (), {"oracle_price": 2100.5})(),
                    "captured_at": datetime.now(tz=UTC),
                },
            )(),
            position=PositionState(),
            account=AccountState(equity=200.0, available_margin=200.0, max_leverage=10.0),
            plan=TradePlan(
                playbook=Playbook.SWEEP_RECLAIM,
                side=TradeSide.SHORT,
                entry_band=(2099.0, 2100.0),
                invalid_if=2105.0,
                tp1=2090.0,
                tp2=2080.0,
                ttl_min=12,
                reason="test",
            ),
            risk=RiskAssessment(
                allowed=True,
                reason="ok",
                recommended_quantity=0.1,
                recommended_notional=100.0,
                risk_budget=1.0,
            ),
        )
    except RuntimeError as exc:
        assert "leverage preflight update failed" in str(exc)
    else:
        raise AssertionError("expected leverage preflight failure to raise")


def test_bot_runtime_refreshes_when_entry_order_state_changes() -> None:
    runtime = object.__new__(BotRuntime)
    runtime._strategy_state = StrategyState(
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
        plan=TradePlan(
            playbook=Playbook.NO_TRADE,
            side=TradeSide.FLAT,
            entry_band=(0.0, 0.0),
            invalid_if=0.0,
            tp1=0.0,
            tp2=0.0,
            ttl_min=0,
            reason="idle",
        ),
        risk=RiskAssessment(allowed=False, reason="idle"),
        updated_at=datetime.now(tz=UTC),
    )
    runtime.strategy_interval_s = 300
    runtime._previous_strategy_signature = (
        TradeSide.FLAT,
        0.0,
        0,
        (),
        False,
        None,
    )
    position = PositionState(
        side=TradeSide.FLAT,
        open_orders=1,
        active_orders=[
            LiveOrderState(
                coin="ETH",
                side="B",
                limit_price=2100.0,
                size=0.1,
                reduce_only=False,
                is_trigger=False,
                order_type="limit",
                oid=1,
                cloid="entry-1",
                status=OrderState.OPEN,
                role=OrderRole.ENTRY,
            )
        ],
    )
    signature = runtime._strategy_refresh_signature(position=position, fills=[])

    assert runtime._should_refresh_strategy(
        now=datetime.now(tz=UTC),
        position=position,
        refresh_signature=signature,
    )


def test_bot_runtime_refreshes_when_fill_state_changes() -> None:
    runtime = object.__new__(BotRuntime)
    runtime._strategy_state = StrategyState(
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
        plan=TradePlan(
            playbook=Playbook.NO_TRADE,
            side=TradeSide.FLAT,
            entry_band=(0.0, 0.0),
            invalid_if=0.0,
            tp1=0.0,
            tp2=0.0,
            ttl_min=0,
            reason="idle",
        ),
        risk=RiskAssessment(allowed=False, reason="idle"),
        updated_at=datetime.now(tz=UTC),
    )
    runtime.strategy_interval_s = 300
    runtime._previous_strategy_signature = (
        TradeSide.FLAT,
        0.0,
        0,
        (),
        False,
        None,
    )
    fill = HyperliquidUserFill(
        coin="ETH",
        closed_pnl=0.0,
        direction="Open Long",
        price=2100.0,
        size=0.1,
        time=datetime.now(tz=UTC),
        oid=2,
        fill_hash="fill-1",
    )
    signature = runtime._strategy_refresh_signature(
        position=PositionState(),
        fills=[fill],
    )

    assert runtime._should_refresh_strategy(
        now=datetime.now(tz=UTC),
        position=PositionState(),
        refresh_signature=signature,
    )
