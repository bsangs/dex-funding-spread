from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

from dex_llm.cli import _build_executor
from dex_llm.config import AppSettings
from dex_llm.executor.live import DesiredOrder, HyperliquidExchangeExecutor
from dex_llm.executor.nonces import NonceManager
from dex_llm.executor.safety import (
    AmbiguousStateResolver,
    AssetMetadata,
    PreSubmitValidator,
    RateLimitBudgeter,
    build_deterministic_cloid,
    extract_role_from_cloid,
)
from dex_llm.models import (
    LiveOrderState,
    MarginMode,
    OrderRole,
    OrderState,
    PendingActionState,
    Playbook,
    PositionState,
    ReconciliationDecision,
    RestingOrderPlan,
    RiskAssessment,
    TradePlan,
    TradeSide,
)


class FakeExchange:
    def __init__(self) -> None:
        self.schedule_cancel_calls: list[int | None] = []
        self.modified: list[dict[str, object]] = []
        self.ordered: list[dict[str, object]] = []
        self.canceled: list[tuple[str, object]] = []
        self.expires_after: int | None = None

    def schedule_cancel(self, deadline: int | None) -> dict[str, object]:
        self.schedule_cancel_calls.append(deadline)
        return {"status": "ok", "response": {"status": "scheduledCancel"}}

    def update_leverage(self, leverage: int, name: str, is_cross: bool = True) -> dict[str, object]:
        return {
            "status": "ok",
            "response": {
                "status": "open",
                "leverage": leverage,
                "name": name,
                "isCross": is_cross,
            },
        }

    def update_isolated_margin(self, amount: float, name: str) -> dict[str, object]:
        return {"status": "ok", "response": {"status": "open", "amount": amount, "name": name}}

    def set_expires_after(self, expires_after: int | None) -> None:
        self.expires_after = expires_after

    def order(
        self,
        *,
        name: str,
        is_buy: bool,
        sz: float,
        limit_px: float,
        order_type: dict[str, object],
        reduce_only: bool,
        cloid: object,
    ) -> dict[str, object]:
        self.ordered.append(
            {
                "name": name,
                "is_buy": is_buy,
                "sz": sz,
                "limit_px": limit_px,
                "order_type": order_type,
                "reduce_only": reduce_only,
                "cloid": str(cloid),
            }
        )
        return {"status": "ok", "response": {"status": "open", "oid": 1}}

    def modify_order(
        self,
        oid: object,
        *,
        name: str,
        is_buy: bool,
        sz: float,
        limit_px: float,
        order_type: dict[str, object],
        reduce_only: bool,
        cloid: object,
    ) -> dict[str, object]:
        self.modified.append(
            {
                "oid": oid,
                "name": name,
                "is_buy": is_buy,
                "sz": sz,
                "limit_px": limit_px,
                "order_type": order_type,
                "reduce_only": reduce_only,
                "cloid": str(cloid),
            }
        )
        return {"status": "ok", "response": {"status": "open", "oid": 2}}

    def cancel_by_cloid(self, symbol: str, cloid: object) -> dict[str, object]:
        self.canceled.append((symbol, cloid))
        return {"status": "ok", "response": {"status": "canceled"}}

    def cancel(self, symbol: str, oid: int) -> dict[str, object]:
        self.canceled.append((symbol, oid))
        return {"status": "ok", "response": {"status": "canceled"}}


class FilledEntryExchange(FakeExchange):
    def __init__(self) -> None:
        super().__init__()
        self.order_statuses = ["filled", "open", "open", "open"]

    def order(
        self,
        *,
        name: str,
        is_buy: bool,
        sz: float,
        limit_px: float,
        order_type: dict[str, object],
        reduce_only: bool,
        cloid: object,
    ) -> dict[str, object]:
        response = super().order(
            name=name,
            is_buy=is_buy,
            sz=sz,
            limit_px=limit_px,
            order_type=order_type,
            reduce_only=reduce_only,
            cloid=cloid,
        )
        status = self.order_statuses[len(self.ordered) - 1]
        response["response"]["status"] = status
        response["response"]["oid"] = len(self.ordered)
        return response


def test_build_deterministic_cloid_is_revisioned() -> None:
    frame_ts = datetime(2026, 3, 12, 12, 0, tzinfo=UTC)

    first = build_deterministic_cloid("strategy", "BTC", frame_ts, OrderRole.ENTRY, 1)
    repeat = build_deterministic_cloid("strategy", "BTC", frame_ts, OrderRole.ENTRY, 1)
    revised = build_deterministic_cloid("strategy", "BTC", frame_ts, OrderRole.ENTRY, 2)

    assert first == repeat
    assert first != revised
    assert first.startswith("0x")
    assert len(first) == 34
    assert extract_role_from_cloid(first) == OrderRole.ENTRY
    assert extract_role_from_cloid(
        build_deterministic_cloid("strategy", "BTC", frame_ts, OrderRole.TAKE_PROFIT_1, 1)
    ) == OrderRole.TAKE_PROFIT_1


def test_nonce_manager_seeds_and_increments(tmp_path: Path) -> None:
    watermark = tmp_path / "nonce.txt"
    watermark.write_text("1000", encoding="utf-8")
    now_values = iter([900, 5000])
    manager = NonceManager(
        "0xsigner",
        watermark_path=watermark,
        now_ms=lambda: next(now_values),
    )

    assert manager.seed() == 1001
    assert manager.next_nonce() == 5000
    assert manager.current() == 5000


def test_pre_submit_validator_quantizes_and_blocks_bad_reduce_only() -> None:
    validator = PreSubmitValidator(
        {"BTC": AssetMetadata(symbol="BTC", asset_index=0, size_decimals=3, max_leverage=20.0)},
        max_price_deviation_bps=500.0,
    )

    result = validator.validate_order(
        symbol="BTC",
        side=TradeSide.LONG,
        price=70_100.9876,
        size=0.12349,
        reduce_only=False,
        best_bid=70_000.0,
        best_ask=70_010.0,
    )

    assert result.valid is True
    assert result.size == 0.123
    assert result.price == 70101.0

    bad_result = validator.validate_order(
        symbol="BTC",
        side=TradeSide.LONG,
        price=70_100.0,
        size=0.123,
        reduce_only=True,
        current_position_size=0.0,
        best_bid=70_000.0,
        best_ask=70_010.0,
    )
    assert bad_result.valid is False
    assert "reduce-only" in bad_result.reason


def test_pre_submit_validator_blocks_cross_entries_without_liquidation_guard() -> None:
    validator = PreSubmitValidator(
        {"ETH": AssetMetadata(symbol="ETH", asset_index=0, size_decimals=3, max_leverage=20.0)}
    )

    result = validator.validate_order(
        symbol="ETH",
        side=TradeSide.LONG,
        price=4000.0,
        size=0.2,
        reduce_only=False,
        best_bid=3999.0,
        best_ask=4001.0,
        margin_mode=MarginMode.CROSS,
        target_leverage=10,
        stop_reference_price=3970.0,
    )

    assert result.valid is False
    assert "cross-mode" in result.reason


def test_pre_submit_validator_blocks_when_stop_is_too_close_to_liquidation() -> None:
    validator = PreSubmitValidator(
        {"ETH": AssetMetadata(symbol="ETH", asset_index=0, size_decimals=3, max_leverage=20.0)}
    )

    result = validator.validate_order(
        symbol="ETH",
        side=TradeSide.LONG,
        price=4000.0,
        size=0.2,
        reduce_only=False,
        best_bid=3999.0,
        best_ask=4001.0,
        margin_mode=MarginMode.ISOLATED,
        target_leverage=10,
        stop_reference_price=3740.0,
    )

    assert result.valid is False
    assert "liquidation buffer" in result.reason


def test_rate_limit_budgeter_soft_degrades_on_pressure() -> None:
    budgeter = RateLimitBudgeter(soft_open_order_limit=900)
    budgeter.update_address_budget(used=85, limit=100)

    status = budgeter.evaluate(open_order_count=901)

    assert status.degrade is True
    assert status.suspend_llm is True
    assert status.reduce_only_only is True


def test_ambiguous_state_resolver_prefers_order_status() -> None:
    resolver = AmbiguousStateResolver(
        query_order_by_cloid=lambda _: {"status": "open", "oid": 1},
        fetch_open_orders=lambda: [],
        fetch_historical_orders=lambda: [],
        sleep_fn=lambda _: None,
    )

    outcome = resolver.resolve(
        PendingActionState(
            symbol="BTC",
            cloid="0x1234567890abcdef1234567890abcdef",
            first_seen_at=datetime.now(tz=UTC),
        )
    )

    assert outcome.decision == ReconciliationDecision.KEEP
    assert outcome.status == OrderState.OPEN


def test_exchange_executor_reconciles_modify_and_cancel_place() -> None:
    exchange = FakeExchange()
    validator = PreSubmitValidator(
        {"BTC": AssetMetadata(symbol="BTC", asset_index=0, size_decimals=3, max_leverage=20.0)}
    )
    executor = HyperliquidExchangeExecutor(
        base_url="https://api.hyperliquid.xyz",
        signer_private_key="0x59c6995e998f97a5a0044966f094538b2924c92f6e7e0c0c7f3d4e3cbb0dbe4a",
        signer_agent_address="0x0000000000000000000000000000000000000000",
        trading_user_address="0x1111111111111111111111111111111111111111",
        validator=validator,
        nonce_manager=NonceManager("0xsigner", now_ms=lambda: 1_000),
        exchange_client=exchange,
    )
    desired_orders = [
        DesiredOrder(
            symbol="BTC",
            side=TradeSide.LONG,
            price=70010.0,
            size=0.1,
            role=OrderRole.ENTRY,
            reduce_only=False,
            order_type={"limit": {"tif": "Gtc"}},
            cloid="0x1234567890abcdef1234567890abcdef",
        ),
        DesiredOrder(
            symbol="BTC",
            side=TradeSide.SHORT,
            price=70600.0,
            size=0.05,
            role=OrderRole.TAKE_PROFIT_1,
            reduce_only=True,
            order_type={"limit": {"tif": "Gtc"}},
            cloid="0xabcdef1234567890abcdef1234567890",
        ),
    ]
    current_orders = [
        LiveOrderState(
            coin="BTC",
            side="B",
            limit_price=69900.0,
            size=0.1,
            reduce_only=False,
            is_trigger=False,
            order_type="limit",
            oid=1,
            cloid="0x11111111111111111111111111111111",
            status=OrderState.OPEN,
            role=OrderRole.ENTRY,
        ),
        LiveOrderState(
            coin="BTC",
            side="A",
            limit_price=70600.0,
            size=0.05,
            reduce_only=False,
            is_trigger=False,
            order_type="limit",
            oid=2,
            cloid="0x22222222222222222222222222222222",
            status=OrderState.OPEN,
            role=OrderRole.TAKE_PROFIT_1,
        ),
    ]

    receipts = executor.reconcile_orders(
        symbol="BTC",
        desired_orders=desired_orders,
        current_orders=current_orders,
        current_position_size=0.1,
        best_bid=70000.0,
        best_ask=70010.0,
        oracle_price=70005.0,
    )

    assert any(receipt.decision == ReconciliationDecision.MODIFY for receipt in receipts)
    assert any(receipt.decision == ReconciliationDecision.CANCEL for receipt in receipts)
    assert any(receipt.decision == ReconciliationDecision.PLACE for receipt in receipts)
    assert len(exchange.modified) == 1
    assert len(exchange.canceled) == 1
    assert len(exchange.ordered) == 1


def test_exchange_executor_keeps_unchanged_orders() -> None:
    exchange = FakeExchange()
    validator = PreSubmitValidator(
        {"BTC": AssetMetadata(symbol="BTC", asset_index=0, size_decimals=3, max_leverage=20.0)}
    )
    executor = HyperliquidExchangeExecutor(
        base_url="https://api.hyperliquid.xyz",
        signer_private_key="0x59c6995e998f97a5a0044966f094538b2924c92f6e7e0c0c7f3d4e3cbb0dbe4a",
        signer_agent_address="0x0000000000000000000000000000000000000000",
        trading_user_address="0x1111111111111111111111111111111111111111",
        validator=validator,
        nonce_manager=NonceManager("0xsigner", now_ms=lambda: 1_000),
        exchange_client=exchange,
    )
    desired = DesiredOrder(
        symbol="BTC",
        side=TradeSide.LONG,
        price=70010.0,
        size=0.1,
        role=OrderRole.ENTRY,
        reduce_only=False,
        order_type={"limit": {"tif": "Gtc"}},
        cloid="0x11111111111111111111111111111111",
    )
    current = LiveOrderState(
        coin="BTC",
        side="B",
        limit_price=70010.0,
        size=0.1,
        reduce_only=False,
        is_trigger=False,
        order_type="limit",
        oid=1,
        cloid="0x11111111111111111111111111111111",
        status=OrderState.OPEN,
        role=OrderRole.ENTRY,
    )

    receipts = executor.reconcile_orders(
        symbol="BTC",
        desired_orders=[desired],
        current_orders=[current],
        current_position_size=0.0,
        best_bid=70000.0,
        best_ask=70010.0,
        oracle_price=70005.0,
    )

    assert len(receipts) == 1
    assert receipts[0].decision == ReconciliationDecision.KEEP
    assert not exchange.modified
    assert not exchange.canceled
    assert not exchange.ordered


def test_exchange_executor_keeps_orders_within_price_and_size_tolerance() -> None:
    exchange = FakeExchange()
    validator = PreSubmitValidator(
        {"BTC": AssetMetadata(symbol="BTC", asset_index=0, size_decimals=3, max_leverage=20.0)}
    )
    executor = HyperliquidExchangeExecutor(
        base_url="https://api.hyperliquid.xyz",
        signer_private_key="0x59c6995e998f97a5a0044966f094538b2924c92f6e7e0c0c7f3d4e3cbb0dbe4a",
        signer_agent_address="0x0000000000000000000000000000000000000000",
        trading_user_address="0x1111111111111111111111111111111111111111",
        validator=validator,
        nonce_manager=NonceManager("0xsigner", now_ms=lambda: 1_000),
        exchange_client=exchange,
    )
    desired = DesiredOrder(
        symbol="BTC",
        side=TradeSide.LONG,
        price=70010.0,
        size=0.1000,
        role=OrderRole.ENTRY,
        reduce_only=False,
        order_type={"limit": {"tif": "Gtc"}},
        cloid="0x11111111111111111111111111111111",
    )
    current = LiveOrderState(
        coin="BTC",
        side="B",
        limit_price=70010.8,
        size=0.10005,
        reduce_only=False,
        is_trigger=False,
        order_type="limit",
        oid=1,
        cloid="0x11111111111111111111111111111111",
        status=OrderState.OPEN,
        role=OrderRole.ENTRY,
    )

    receipts = executor.reconcile_orders(
        symbol="BTC",
        desired_orders=[desired],
        current_orders=[current],
        current_position_size=0.0,
        best_bid=70000.0,
        best_ask=70010.0,
        oracle_price=70005.0,
    )

    assert len(receipts) == 1
    assert receipts[0].decision == ReconciliationDecision.KEEP
    assert not exchange.modified
    assert not exchange.canceled
    assert not exchange.ordered


def test_exchange_executor_replaces_orders_outside_keep_tolerance() -> None:
    exchange = FakeExchange()
    validator = PreSubmitValidator(
        {"BTC": AssetMetadata(symbol="BTC", asset_index=0, size_decimals=3, max_leverage=20.0)}
    )
    executor = HyperliquidExchangeExecutor(
        base_url="https://api.hyperliquid.xyz",
        signer_private_key="0x59c6995e998f97a5a0044966f094538b2924c92f6e7e0c0c7f3d4e3cbb0dbe4a",
        signer_agent_address="0x0000000000000000000000000000000000000000",
        trading_user_address="0x1111111111111111111111111111111111111111",
        validator=validator,
        nonce_manager=NonceManager("0xsigner", now_ms=lambda: 1_000),
        exchange_client=exchange,
    )
    desired = DesiredOrder(
        symbol="BTC",
        side=TradeSide.LONG,
        price=70010.0,
        size=0.1000,
        role=OrderRole.ENTRY,
        reduce_only=False,
        order_type={"limit": {"tif": "Gtc"}},
        cloid="0x11111111111111111111111111111111",
    )
    current = LiveOrderState(
        coin="BTC",
        side="B",
        limit_price=70030.0,
        size=0.1020,
        reduce_only=False,
        is_trigger=False,
        order_type="limit",
        oid=1,
        cloid="0x11111111111111111111111111111111",
        status=OrderState.OPEN,
        role=OrderRole.ENTRY,
    )

    receipts = executor.reconcile_orders(
        symbol="BTC",
        desired_orders=[desired],
        current_orders=[current],
        current_position_size=0.0,
        best_bid=70000.0,
        best_ask=70010.0,
        oracle_price=70005.0,
    )

    assert any(receipt.decision == ReconciliationDecision.MODIFY for receipt in receipts)
    assert not any(receipt.decision == ReconciliationDecision.KEEP for receipt in receipts)
    assert exchange.modified


def test_execute_plan_places_protection_after_filled_cluster_fade_entry() -> None:
    exchange = FilledEntryExchange()
    validator = PreSubmitValidator(
        {"BTC": AssetMetadata(symbol="BTC", asset_index=0, size_decimals=3, max_leverage=20.0)}
    )
    executor = HyperliquidExchangeExecutor(
        base_url="https://api.hyperliquid.xyz",
        signer_private_key="0x59c6995e998f97a5a0044966f094538b2924c92f6e7e0c0c7f3d4e3cbb0dbe4a",
        signer_agent_address="0x0000000000000000000000000000000000000000",
        trading_user_address="0x1111111111111111111111111111111111111111",
        validator=validator,
        nonce_manager=NonceManager("0xsigner", now_ms=lambda: 1_000),
        exchange_client=exchange,
    )
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
                entry_band=(69990.0, 70010.0),
                invalid_if=69850.0,
                tp1=70100.0,
                tp2=70200.0,
                ttl_min=30,
                reason="lower long fade",
            ),
            RestingOrderPlan(
                side=TradeSide.SHORT,
                entry_band=(70590.0, 70610.0),
                invalid_if=70750.0,
                tp1=70450.0,
                tp2=70350.0,
                ttl_min=30,
                reason="upper short fade",
            ),
        ],
    )

    receipts = executor.execute_plan(
        plan=plan,
        risk=RiskAssessment(
            allowed=True,
            reason="risk checks passed",
            recommended_quantity=0.12,
            resting_order_quantities=[0.12, 0.05],
            recommended_notional=1193.0,
            risk_budget=35.0,
        ),
        symbol="BTC",
        frame_timestamp=datetime(2026, 3, 13, 0, 0, tzinfo=UTC),
        position=PositionState(),
        best_bid=70000.0,
        best_ask=70010.0,
        oracle_price=70005.0,
    )

    assert len(exchange.ordered) == 5
    assert receipts[0].status == OrderState.FILLED
    assert any(receipt.action == "cancel" for receipt in receipts)
    assert exchange.canceled
    assert exchange.ordered[0]["sz"] == 0.12
    assert exchange.ordered[1]["sz"] == 0.05
    assert sum(1 for receipt in receipts if receipt.action == "place") >= 4


def test_exchange_executor_does_not_arm_dead_man_switch_for_resting_entries() -> None:
    exchange = FakeExchange()
    validator = PreSubmitValidator(
        {"BTC": AssetMetadata(symbol="BTC", asset_index=0, size_decimals=3)}
    )
    executor = HyperliquidExchangeExecutor(
        base_url="https://api.hyperliquid.xyz",
        signer_private_key="0x59c6995e998f97a5a0044966f094538b2924c92f6e7e0c0c7f3d4e3cbb0dbe4a",
        signer_agent_address="0x0000000000000000000000000000000000000000",
        trading_user_address="0x1111111111111111111111111111111111111111",
        validator=validator,
        nonce_manager=NonceManager("0xsigner", now_ms=lambda: 1_000),
        exchange_client=exchange,
    )

    executor.schedule_dead_man_switch(
        has_resting_entry=True,
        position_open=False,
        now=datetime(2026, 3, 12, 12, 0, tzinfo=UTC),
    )
    executor.schedule_dead_man_switch(
        has_resting_entry=True,
        position_open=True,
        now=datetime(2026, 3, 12, 12, 0, tzinfo=UTC),
    )

    assert exchange.schedule_cancel_calls[0] is None
    assert exchange.schedule_cancel_calls[1] is None


def test_build_executor_uses_requested_symbol() -> None:
    class DummyGateway:
        def __init__(self) -> None:
            self.symbols: list[str] = []

        def fetch_asset_meta(self, symbol: str) -> AssetMetadata:
            self.symbols.append(symbol)
            return AssetMetadata(symbol=symbol, asset_index=0, size_decimals=3)

        def user_rate_limit(self, user: str) -> dict[str, int]:
            return {"nRequestsUsed": 0, "nRequestsCap": 100}

        def query_order_by_cloid(self, user: str, cloid: str) -> dict[str, object]:
            return {}

        def open_orders(self, user: str) -> list[dict[str, object]]:
            return []

        def historical_orders(self, user: str) -> list[dict[str, object]]:
            return []

    settings = AppSettings(
        signer_private_key="0x59c6995e998f97a5a0044966f094538b2924c92f6e7e0c0c7f3d4e3cbb0dbe4a",
        signer_agent_address="0x0000000000000000000000000000000000000000",
    )
    gateway = DummyGateway()

    _build_executor(
        settings,
        symbol="ETH",
        rest_gateway=gateway,
        user_address="0x1111111111111111111111111111111111111111",
    )

    assert gateway.symbols == ["ETH"]


def test_leverage_preflight_rejects_insufficient_isolated_margin() -> None:
    validator = PreSubmitValidator(
        {
            "BTC": AssetMetadata(
                symbol="BTC",
                asset_index=0,
                size_decimals=3,
                max_leverage=20.0,
            )
        }
    )

    result = validator.validate_leverage_preflight(
        symbol="BTC",
        target_leverage=10,
        margin_mode=MarginMode.ISOLATED,
        current_leverage=None,
        max_leverage=20.0,
        recommended_notional=10_000.0,
        available_margin=500.0,
    )

    assert result.valid is False
    assert "isolated margin" in result.reason
