from __future__ import annotations

import io
import json
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest
from rich.console import Console

from dex_llm.bot import BotRuntime, StrategyState
from dex_llm.models import (
    AccountState,
    Candle,
    FeatureSnapshot,
    LiveOrderState,
    KillSwitchStatus,
    LiveStateSnapshot,
    OrderBookSnapshot,
    OrderState,
    PriceLevel,
    HyperliquidUserFill,
    MarketFrame,
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


def test_bot_runtime_invalidates_single_resting_order() -> None:
    runtime = object.__new__(BotRuntime)
    plan = TradePlan(
        playbook=Playbook.CLUSTER_FADE,
        side=TradeSide.FLAT,
        entry_band=(0.0, 0.0),
        invalid_if=0.0,
        tp1=0.0,
        tp2=0.0,
        ttl_min=30,
        reason="rest one side",
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

    assert effective.playbook == Playbook.NO_TRADE
    assert "invalidated" in effective.reason


def test_bot_runtime_pauses_entry_after_sticky_exchange_rejection() -> None:
    runtime = object.__new__(BotRuntime)
    updated_at = datetime.now(tz=UTC)
    runtime._entry_rejection_block = type(
        "EntryRejectionBlock",
        (),
        {
            "strategy_updated_at": updated_at,
            "reason": "Insufficient margin to place order. asset=1",
        },
    )()
    plan = TradePlan(
        playbook=Playbook.CLUSTER_FADE,
        side=TradeSide.SHORT,
        entry_band=(2152.0, 2154.0),
        invalid_if=2160.0,
        tp1=2140.0,
        tp2=2128.0,
        ttl_min=40,
        reason="short fade",
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
        updated_at=updated_at,
    )

    effective = runtime._effective_plan(
        strategy_state=state,
        position=PositionState(),
        current_price=2100.0,
        now=datetime.now(tz=UTC),
    )

    assert effective.playbook == Playbook.NO_TRADE
    assert "entry paused after exchange rejection" in effective.reason


def test_bot_runtime_ensure_ws_connection_does_not_raise_on_reconnect_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runtime = object.__new__(BotRuntime)
    runtime.sync_interval_s = 120
    events: list[tuple[str, int | None, str]] = []

    class StubWsClient:
        def connection_alive(self) -> bool:
            return False

        def reconnect(self) -> None:
            raise RuntimeError("temporary 502")

        def wait_until_public_ready(self, *, timeout_s: float) -> None:
            raise AssertionError("should not reach wait_until_public_ready after failed reconnect")

    runtime.ws_client = StubWsClient()
    runtime._emit_runtime_error = lambda *, phase, cycle, exc: events.append(
        (phase, cycle, str(exc))
    )
    monkeypatch.setattr("dex_llm.bot.time.sleep", lambda _: None)

    connected = runtime._ensure_ws_connection(timeout_s=10.0, cycle=7)

    assert connected is False
    assert events == [("reconnect", 7, "temporary 502")]


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


def test_bot_runtime_returns_receipt_when_live_leverage_preflight_is_invalid() -> None:
    runtime = object.__new__(BotRuntime)
    runtime.live = True
    runtime.symbol = "ETH"

    class StubExecutor:
        target_leverage = 10
        margin_mode = "isolated"

        def build_orders_from_plan(self, **_: object) -> list[object]:
            return [type("Order", (), {"reduce_only": False})()]

        def target_leverage_for_side(self, _: TradeSide) -> int:
            return self.target_leverage

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

    receipts, leverage_preflight = runtime._sync_orders(
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

    assert leverage_preflight is not None
    assert leverage_preflight["valid"] is False
    assert "leverage preflight update failed" in leverage_preflight["reason"]
    assert len(receipts) == 1
    assert receipts[0]["action"] == "leverage_preflight"
    assert receipts[0]["status"] == "rejected"


def test_bot_runtime_keeps_active_entry_workflow_on_soft_no_trade() -> None:
    runtime = object.__new__(BotRuntime)
    runtime.live = True
    runtime.symbol = "ETH"

    class StubExecutor:
        def _signed_position_size(self, position: PositionState) -> float:
            raise AssertionError("existing entry workflow should be preserved, not reconciled")

    runtime.executor = StubExecutor()

    receipts, leverage_preflight = runtime._sync_orders(
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
        position=PositionState(
            side=TradeSide.FLAT,
            active_orders=[
                LiveOrderState(
                    coin="ETH",
                    side="B",
                    limit_price=2039.0,
                    size=0.1,
                    reduce_only=False,
                    is_trigger=False,
                    order_type="limit",
                    oid=1,
                    status=OrderState.OPEN,
                ),
                LiveOrderState(
                    coin="ETH",
                    side="A",
                    limit_price=2092.0,
                    size=0.1,
                    reduce_only=True,
                    is_trigger=True,
                    order_type="trigger",
                    oid=2,
                    status=OrderState.OPEN,
                ),
            ],
        ),
        account=AccountState(equity=200.0, available_margin=200.0, max_leverage=10.0),
        plan=TradePlan(
            playbook=Playbook.NO_TRADE,
            side=TradeSide.FLAT,
            entry_band=(0.0, 0.0),
            invalid_if=0.0,
            tp1=0.0,
            tp2=0.0,
            ttl_min=0,
            reason="existing entry workflow detected; reconcile open orders first",
        ),
        risk=RiskAssessment(allowed=False, reason="plan requests hold/close only"),
    )

    assert receipts == []
    assert leverage_preflight is None


def test_bot_runtime_keeps_active_entry_workflow_even_on_hard_no_trade_reason() -> None:
    runtime = object.__new__(BotRuntime)
    runtime.live = True
    runtime.symbol = "ETH"

    class StubExecutor:
        def _signed_position_size(self, position: PositionState) -> float:
            raise AssertionError("flat active entry workflow should never cancel-only")

    runtime.executor = StubExecutor()

    receipts, leverage_preflight = runtime._sync_orders(
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
        position=PositionState(
            side=TradeSide.FLAT,
            active_orders=[
                LiveOrderState(
                    coin="ETH",
                    side="B",
                    limit_price=2039.0,
                    size=0.1,
                    reduce_only=False,
                    is_trigger=False,
                    order_type="limit",
                    oid=1,
                    status=OrderState.OPEN,
                ),
            ],
        ),
        account=AccountState(equity=200.0, available_margin=200.0, max_leverage=10.0),
        plan=TradePlan(
            playbook=Playbook.NO_TRADE,
            side=TradeSide.FLAT,
            entry_band=(0.0, 0.0),
            invalid_if=0.0,
            tp1=0.0,
            tp2=0.0,
            ttl_min=0,
            reason="directional entry invalidated before fill",
        ),
        risk=RiskAssessment(allowed=False, reason="plan requests hold/close only"),
    )

    assert receipts == []
    assert leverage_preflight is None


def test_bot_runtime_keeps_active_entry_workflow_when_risk_blocks_new_entry() -> None:
    runtime = object.__new__(BotRuntime)
    runtime.live = True
    runtime.symbol = "ETH"

    class StubExecutor:
        def build_orders_from_plan(self, **kwargs: object):
            raise AssertionError("blocked active entry workflow should not rebuild desired orders")

    runtime.executor = StubExecutor()

    receipts, leverage_preflight = runtime._sync_orders(
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
        position=PositionState(
            side=TradeSide.FLAT,
            active_orders=[
                LiveOrderState(
                    coin="ETH",
                    side="B",
                    limit_price=2039.0,
                    size=0.1,
                    reduce_only=False,
                    is_trigger=False,
                    order_type="limit",
                    oid=1,
                    status=OrderState.OPEN,
                ),
                LiveOrderState(
                    coin="ETH",
                    side="A",
                    limit_price=2092.0,
                    size=0.1,
                    reduce_only=True,
                    is_trigger=True,
                    order_type="trigger",
                    oid=2,
                    status=OrderState.OPEN,
                ),
            ],
        ),
        account=AccountState(equity=200.0, available_margin=200.0, max_leverage=10.0),
        plan=TradePlan(
            playbook=Playbook.CLUSTER_FADE,
            side=TradeSide.LONG,
            entry_band=(2038.0, 2040.0),
            invalid_if=2017.0,
            tp1=2077.0,
            tp2=2092.0,
            ttl_min=300,
            reason="preserve existing active entry workflow",
        ),
        risk=RiskAssessment(
            allowed=False,
            reason="entry workflow already exists; reconcile live orders first",
        ),
    )

    assert receipts == []
    assert leverage_preflight is None


def test_bot_runtime_clamps_target_leverage_to_account_max() -> None:
    runtime = object.__new__(BotRuntime)
    runtime.live = True
    runtime.symbol = "ETH"

    class StubExecutor:
        target_leverage = 20
        margin_mode = "isolated"

        def build_orders_from_plan(self, **_: object) -> list[object]:
            return [type("Order", (), {"reduce_only": False})()]

        def target_leverage_for_side(self, _: TradeSide) -> int:
            return self.target_leverage

        def apply_leverage_preflight(self, **kwargs: object):
            assert kwargs["target_leverage"] == 10
            return type(
                "ValidationResult",
                (),
                {
                    "valid": True,
                    "reason": "ok",
                    "model_dump": lambda self, mode="json": {"valid": True, "reason": "ok"},
                },
            )()

        def execute_plan(self, **_: object):
            return []

    runtime.executor = StubExecutor()

    receipts, leverage_preflight = runtime._sync_orders(
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

    assert receipts == []
    assert leverage_preflight == {"valid": True, "reason": "ok"}


def test_bot_runtime_only_refreshes_after_strategy_interval() -> None:
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
    assert runtime._should_refresh_strategy(now=datetime.now(tz=UTC)) is False


def test_bot_runtime_refreshes_after_strategy_interval() -> None:
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
        updated_at=datetime.now(tz=UTC) - timedelta(minutes=6),
    )
    runtime.strategy_interval_s = 300
    assert runtime._should_refresh_strategy(now=datetime.now(tz=UTC)) is True


def test_bot_runtime_waits_for_next_review_after_recent_fill() -> None:
    runtime = object.__new__(BotRuntime)
    updated_at = datetime.now(tz=UTC) - timedelta(minutes=1)
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
        updated_at=updated_at,
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
    effective = runtime._effective_plan(
        strategy_state=state,
        position=PositionState(),
        current_price=2101.0,
        now=datetime.now(tz=UTC),
        fills=[fill],
    )

    assert effective.playbook == Playbook.NO_TRADE
    assert "recent fill" in effective.reason


def test_bot_runtime_skips_router_when_position_is_open() -> None:
    runtime = object.__new__(BotRuntime)
    runtime.heatmap_params = {}
    runtime.allow_synthetic = False
    runtime._strategy_state = None

    class StubBuilder:
        def build_from_snapshot(self, *_: object, **__: object) -> MarketFrame:
            frame = MarketFrame.model_validate(
                json.loads(Path("examples/sample_frame.json").read_text())
            )
            return frame.model_copy(
                update={"position": PositionState(side=TradeSide.SHORT, quantity=1.0)}
            )

    class StubRouter:
        def route(self, *_: object, **__: object) -> TradePlan:
            raise AssertionError("router should not be called for open-position management")

    class StubRiskPolicy:
        def assess(self, *_: object, **__: object) -> RiskAssessment:
            return RiskAssessment(
                allowed=False,
                reason="single-position mode blocks averaging down",
            )

    runtime.builder = StubBuilder()
    runtime.router = StubRouter()
    runtime.risk_policy = StubRiskPolicy()
    runtime._account_from_snapshot = lambda snapshot: AccountState(
        equity=100.0,
        available_margin=100.0,
        max_leverage=20.0,
    )

    state = runtime._compute_strategy_state(
        snapshot=object(),  # type: ignore[arg-type]
        fills=None,
        position=PositionState(side=TradeSide.SHORT, quantity=1.0),
    )

    assert state.plan.side == TradeSide.SHORT
    assert "code-managed" in state.plan.reason


def test_account_from_snapshot_clamps_to_venue_max_leverage() -> None:
    runtime = object.__new__(BotRuntime)
    runtime.live = True
    runtime.max_leverage = 20.0
    now = datetime.now(tz=UTC)

    account = runtime._account_from_snapshot(
        type(
            "Snapshot",
            (),
            {
                "clearinghouse_state": type(
                    "State",
                    (),
                    {
                        "margin_summary": type("Margin", (), {"account_value": 500.0})(),
                        "withdrawable": 450.0,
                    },
                )(),
                "active_asset_ctx": type("Ctx", (), {"max_leverage": 10.0})(),
                "order_book": type("Book", (), {"mid_price": 2100.0})(),
            },
        )()
    )

    assert account.max_leverage == 10.0


def test_bot_runtime_emit_cycle_suppresses_idle_cycles_after_boot() -> None:
    buffer = io.StringIO()
    runtime = _logging_runtime(console=Console(file=buffer, force_terminal=False, width=120))
    now = datetime.now(tz=UTC)
    snapshot = _snapshot(now=now)
    plan = TradePlan(
        playbook=Playbook.NO_TRADE,
        side=TradeSide.FLAT,
        entry_band=(0.0, 0.0),
        invalid_if=0.0,
        tp1=0.0,
        tp2=0.0,
        ttl_min=0,
        reason="idle",
    )

    runtime._emit_cycle(
        cycle=1,
        snapshot=snapshot,
        position=PositionState(),
        plan=plan,
        risk=RiskAssessment(allowed=False, reason="plan requests hold/close only"),
        kill_switch=KillSwitchStatus(),
        receipts=[],
        meta=_meta(),
    )

    assert "BOOT" in buffer.getvalue()
    assert "PLAN" in buffer.getvalue()

    buffer.seek(0)
    buffer.truncate(0)

    runtime._emit_cycle(
        cycle=2,
        snapshot=_snapshot(now=now + timedelta(minutes=1)),
        position=PositionState(),
        plan=plan,
        risk=RiskAssessment(allowed=False, reason="plan requests hold/close only"),
        kill_switch=KillSwitchStatus(),
        receipts=[],
        meta=_meta(),
    )

    assert buffer.getvalue() == ""


def test_bot_runtime_emit_cycle_groups_eventful_cycle_into_review_block() -> None:
    buffer = io.StringIO()
    runtime = _logging_runtime(console=Console(file=buffer, force_terminal=False, width=120))
    now = datetime.now(tz=UTC)
    plan = TradePlan(
        playbook=Playbook.CLUSTER_FADE,
        side=TradeSide.LONG,
        entry_band=(2090.0, 2092.0),
        invalid_if=2085.0,
        tp1=2100.0,
        tp2=2110.0,
        ttl_min=90,
        reason="fade the lower wall with a single long limit order",
    )

    runtime._emit_cycle(
        cycle=1,
        snapshot=_snapshot(now=now),
        position=PositionState(),
        plan=plan,
        risk=RiskAssessment(allowed=True, reason="side-based sizing checks passed"),
        kill_switch=KillSwitchStatus(),
        receipts=[
            {
                "action": "place",
                "cloid": "0x1100feedfacefeedfacefeedfacefeed",
                "oid": 12345,
                "decision": "place",
                "success": True,
                "status": "open",
                "message": "",
            }
        ],
        meta=_meta(),
    )

    output = buffer.getvalue()
    assert "REVIEW R0001" in output
    assert "PLAN" in output
    assert "WHY" in output
    assert "ORDER" in output
    assert "place entry oid=12345 status=open" in output


def test_bot_runtime_treats_grouped_submit_as_order_not_error() -> None:
    buffer = io.StringIO()
    runtime = _logging_runtime(console=Console(file=buffer, force_terminal=False, width=120))
    now = datetime.now(tz=UTC)
    plan = TradePlan(
        playbook=Playbook.CLUSTER_FADE,
        side=TradeSide.SHORT,
        entry_band=(2152.0, 2154.0),
        invalid_if=2160.0,
        tp1=2140.0,
        tp2=2128.0,
        ttl_min=40,
        reason="short fade",
    )

    runtime._emit_cycle(
        cycle=1,
        snapshot=_snapshot(now=now),
        position=PositionState(),
        plan=plan,
        risk=RiskAssessment(allowed=True, reason="ok"),
        kill_switch=KillSwitchStatus(),
        receipts=[
            {
                "action": "place",
                "cloid": "0x2200feedfacefeedfacefeedfacefeed",
                "oid": None,
                "decision": "place",
                "success": True,
                "status": "unknown",
                "message": "grouped order submitted",
            }
        ],
        meta=_meta(),
    )

    output = buffer.getvalue()
    assert "ORDER" in output
    assert "grouped order submitted" not in output or "ERROR" not in output


def test_bot_runtime_emit_cycle_keeps_printing_errors_when_error_persists() -> None:
    buffer = io.StringIO()
    runtime = _logging_runtime(console=Console(file=buffer, force_terminal=False, width=120))
    now = datetime.now(tz=UTC)
    plan = TradePlan(
        playbook=Playbook.NO_TRADE,
        side=TradeSide.FLAT,
        entry_band=(0.0, 0.0),
        invalid_if=0.0,
        tp1=0.0,
        tp2=0.0,
        ttl_min=0,
        reason="private account state unavailable",
    )
    kill_switch = KillSwitchStatus(
        allow_new_trades=False,
        reduce_only=False,
        reasons=["private account state unavailable"],
    )

    runtime._emit_cycle(
        cycle=1,
        snapshot=_snapshot(now=now),
        position=PositionState(),
        plan=plan,
        risk=RiskAssessment(allowed=False, reason="private account state unavailable"),
        kill_switch=kill_switch,
        receipts=[],
        meta=_meta(),
    )

    buffer.seek(0)
    buffer.truncate(0)

    runtime._emit_cycle(
        cycle=2,
        snapshot=_snapshot(now=now + timedelta(minutes=1)),
        position=PositionState(),
        plan=plan,
        risk=RiskAssessment(allowed=False, reason="private account state unavailable"),
        kill_switch=kill_switch,
        receipts=[],
        meta=_meta(),
    )
    second_output = buffer.getvalue()
    assert 'ERROR    kill switch active | reason="private account state unavailable"' in second_output

    buffer.seek(0)
    buffer.truncate(0)

    runtime._emit_cycle(
        cycle=3,
        snapshot=_snapshot(now=now + timedelta(minutes=2)),
        position=PositionState(),
        plan=plan,
        risk=RiskAssessment(allowed=False, reason="private account state unavailable"),
        kill_switch=kill_switch,
        receipts=[],
        meta=_meta(),
    )
    third_output = buffer.getvalue()
    assert 'ERROR    kill switch active | reason="private account state unavailable"' in third_output


def _logging_runtime(*, console: Console) -> BotRuntime:
    runtime = object.__new__(BotRuntime)
    runtime.symbol = "ETH"
    runtime.user_address = "0x1234567890abcdef1234567890abcdef12345678"
    runtime.strategy_interval_s = 300
    runtime.sync_interval_s = 60
    runtime.live = True
    runtime.console = console
    runtime._event_block_count = 0
    runtime._boot_logged = False
    runtime._last_plan_signature = None
    runtime._last_position_signature = None
    runtime._last_reduce_only_signature = None
    runtime._last_entry_block_reason = None
    runtime._last_active_order_signature = None
    runtime._seen_fill_keys = set()
    runtime._seen_user_event_keys = set()
    runtime._entry_rejection_block = None
    return runtime


def _snapshot(
    *,
    now: datetime,
    fills: list[HyperliquidUserFill] | None = None,
) -> LiveStateSnapshot:
    return LiveStateSnapshot(
        symbol="ETH",
        order_book=OrderBookSnapshot(
            symbol="ETH",
            captured_at=now,
            best_bid=2100.0,
            best_ask=2101.0,
            mid_price=2100.5,
            bids=[PriceLevel(price=2100.0, size=1.0, orders=1)],
            asks=[PriceLevel(price=2101.0, size=1.0, orders=1)],
        ),
        candles_5m=[
            Candle(ts=now, open=2100.0, high=2102.0, low=2098.0, close=2101.0, volume=1.0)
        ],
        candles_15m=[
            Candle(ts=now, open=2100.0, high=2103.0, low=2097.0, close=2101.0, volume=1.0)
        ],
        recent_fills=fills or [],
    )


def _meta() -> dict[str, object]:
    return {
        "private_state_source": "ws",
        "private_ws_ready": True,
        "fills_safe_complete": True,
    }
