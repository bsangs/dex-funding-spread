from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from typing import Any

from eth_account import Account
from hyperliquid.exchange import Exchange
from hyperliquid.utils.types import Cloid

from dex_llm.executor.nonces import ClockDriftError, NonceManager
from dex_llm.executor.safety import (
    AmbiguousStateResolver,
    BudgetStatus,
    PreSubmitValidator,
    RateLimitBudgeter,
    ValidationResult,
    build_deterministic_cloid,
)
from dex_llm.models import (
    ExecutionReceipt,
    LiveOrderState,
    MarginMode,
    OrderRole,
    OrderState,
    PendingActionState,
    PositionState,
    ReconciliationDecision,
    TradePlan,
    TradeSide,
)


@dataclass(slots=True)
class DesiredOrder:
    symbol: str
    side: TradeSide
    price: float
    size: float
    role: OrderRole
    reduce_only: bool
    order_type: dict[str, object]
    cloid: str
    expires_after: int | None = None


class HyperliquidExchangeExecutor:
    def __init__(
        self,
        *,
        base_url: str,
        signer_private_key: str,
        signer_agent_address: str,
        trading_user_address: str,
        validator: PreSubmitValidator,
        nonce_manager: NonceManager,
        budgeter: RateLimitBudgeter | None = None,
        vault_address: str | None = None,
        exchange_client: Exchange | None = None,
        ambiguous_resolver: AmbiguousStateResolver | None = None,
    ) -> None:
        self.wallet = Account.from_key(signer_private_key)
        self.signer_agent_address = signer_agent_address.lower()
        self.trading_user_address = trading_user_address
        self.validator = validator
        self.nonce_manager = nonce_manager
        self.budgeter = budgeter or RateLimitBudgeter()
        self.exchange = exchange_client or Exchange(
            wallet=self.wallet,
            base_url=base_url,
            account_address=trading_user_address,
            vault_address=vault_address,
        )
        self.ambiguous_resolver = ambiguous_resolver

    def verify_signer(self) -> None:
        derived = self.wallet.address.lower()
        if derived != self.signer_agent_address:
            raise ValueError("signer_agent_address does not match signer_private_key")

    def seed_nonce(self, reference_ms: int | None = None) -> int:
        return self.nonce_manager.seed(reference_ms)

    def schedule_dead_man_switch(
        self,
        *,
        has_resting_entry: bool,
        position_open: bool,
        now: datetime,
    ) -> Any:
        if position_open or not has_resting_entry:
            return self.exchange.schedule_cancel(None)
        deadline = int((now + timedelta(seconds=30)).timestamp() * 1000)
        return self.exchange.schedule_cancel(deadline)

    def apply_leverage_preflight(
        self,
        *,
        symbol: str,
        target_leverage: int,
        margin_mode: MarginMode,
        current_leverage: float | None,
        max_leverage: float,
        recommended_notional: float,
        available_margin: float,
    ) -> ValidationResult:
        result = self.validator.validate_leverage_preflight(
            symbol=symbol,
            target_leverage=target_leverage,
            margin_mode=margin_mode,
            current_leverage=current_leverage,
            max_leverage=max_leverage,
            recommended_notional=recommended_notional,
            available_margin=available_margin,
        )
        if not result.valid:
            return result
        if current_leverage is None or int(current_leverage) != target_leverage:
            self.exchange.update_leverage(
                leverage=target_leverage,
                name=symbol,
                is_cross=margin_mode == MarginMode.CROSS,
            )
        if margin_mode == MarginMode.ISOLATED:
            required_margin = (recommended_notional / target_leverage) * 1.1
            self.exchange.update_isolated_margin(amount=required_margin, name=symbol)
        return result

    def build_orders_from_plan(
        self,
        *,
        symbol: str,
        plan: TradePlan,
        quantity: float,
        frame_timestamp: datetime,
        revision: int,
        strategy_id: str = "dex-llm",
    ) -> list[DesiredOrder]:
        if plan.side not in {TradeSide.LONG, TradeSide.SHORT}:
            return []
        entry_price = sum(plan.entry_band) / 2
        entry_side = plan.side
        exit_side = TradeSide.SHORT if entry_side == TradeSide.LONG else TradeSide.LONG
        entry_expires_after = None
        if plan.ttl_min <= 1:
            entry_expires_after = int(
                (frame_timestamp + timedelta(minutes=plan.ttl_min)).timestamp() * 1000
            )
        return [
            DesiredOrder(
                symbol=symbol,
                side=entry_side,
                price=entry_price,
                size=quantity,
                role=OrderRole.ENTRY,
                reduce_only=False,
                order_type={"limit": {"tif": "Gtc"}},
                cloid=build_deterministic_cloid(
                    strategy_id,
                    symbol,
                    frame_timestamp,
                    OrderRole.ENTRY,
                    revision,
                ),
                expires_after=entry_expires_after,
            ),
            DesiredOrder(
                symbol=symbol,
                side=exit_side,
                price=plan.invalid_if,
                size=quantity,
                role=OrderRole.STOP_LOSS,
                reduce_only=True,
                order_type={
                    "trigger": {
                        "triggerPx": plan.invalid_if,
                        "isMarket": True,
                        "tpsl": "sl",
                    }
                },
                cloid=build_deterministic_cloid(
                    strategy_id,
                    symbol,
                    frame_timestamp,
                    OrderRole.STOP_LOSS,
                    revision,
                ),
            ),
            DesiredOrder(
                symbol=symbol,
                side=exit_side,
                price=plan.tp1,
                size=quantity / 2,
                role=OrderRole.TAKE_PROFIT_1,
                reduce_only=True,
                order_type={"limit": {"tif": "Gtc"}},
                cloid=build_deterministic_cloid(
                    strategy_id,
                    symbol,
                    frame_timestamp,
                    OrderRole.TAKE_PROFIT_1,
                    revision,
                ),
            ),
            DesiredOrder(
                symbol=symbol,
                side=exit_side,
                price=plan.tp2,
                size=quantity / 2,
                role=OrderRole.TAKE_PROFIT_2,
                reduce_only=True,
                order_type={"limit": {"tif": "Gtc"}},
                cloid=build_deterministic_cloid(
                    strategy_id,
                    symbol,
                    frame_timestamp,
                    OrderRole.TAKE_PROFIT_2,
                    revision,
                ),
            ),
        ]

    def execute_plan(
        self,
        *,
        plan: TradePlan,
        symbol: str,
        quantity: float,
        frame_timestamp: datetime,
        position: PositionState,
        best_bid: float,
        best_ask: float,
        oracle_price: float | None,
        revision: int = 0,
    ) -> list[ExecutionReceipt]:
        budget = self.budgeter.evaluate(open_order_count=position.open_orders)
        if budget.reduce_only_only and position.side == TradeSide.FLAT:
            return [
                self._receipt(
                    symbol=symbol,
                    cloid="budget-blocked",
                    action="budget",
                    decision=ReconciliationDecision.AWAIT_RESOLUTION,
                    success=False,
                    status=OrderState.UNKNOWN,
                    message="rate-limit budget degraded; new entries suspended",
                )
            ]

        desired_orders = self.build_orders_from_plan(
            symbol=symbol,
            plan=plan,
            quantity=quantity,
            frame_timestamp=frame_timestamp,
            revision=revision,
        )
        if position.side == TradeSide.FLAT:
            desired_orders = [order for order in desired_orders if order.role == OrderRole.ENTRY]
        else:
            desired_orders = [order for order in desired_orders if order.role != OrderRole.ENTRY]

        return self.reconcile_orders(
            symbol=symbol,
            desired_orders=desired_orders,
            current_orders=position.active_orders,
            current_position_size=quantity if position.side != TradeSide.FLAT else 0.0,
            best_bid=best_bid,
            best_ask=best_ask,
            oracle_price=oracle_price,
        )

    def reconcile_orders(
        self,
        *,
        symbol: str,
        desired_orders: list[DesiredOrder],
        current_orders: Iterable[LiveOrderState],
        current_position_size: float,
        best_bid: float,
        best_ask: float,
        oracle_price: float | None,
    ) -> list[ExecutionReceipt]:
        current_by_role = {order.role: order for order in current_orders if order.coin == symbol}
        receipts: list[ExecutionReceipt] = []

        for desired in desired_orders:
            current = current_by_role.get(desired.role)
            if current is None:
                receipts.append(
                    self.place_order(
                        desired,
                        current_position_size=current_position_size,
                        best_bid=best_bid,
                        best_ask=best_ask,
                        oracle_price=oracle_price,
                    )
                )
                continue
            if self._can_keep(current, desired):
                receipts.append(
                    self._receipt(
                        symbol=symbol,
                        cloid=current.cloid or desired.cloid,
                        action="keep",
                        decision=ReconciliationDecision.KEEP,
                        success=True,
                        status=current.status,
                        message="existing order already matches desired state",
                    )
                )
                continue
            if self._can_modify(current, desired):
                receipts.append(
                    self.modify_order(
                        current,
                        desired,
                        current_position_size=current_position_size,
                        best_bid=best_bid,
                        best_ask=best_ask,
                        oracle_price=oracle_price,
                    )
                )
                continue
            receipts.append(self.cancel_order(symbol, current))
            receipts.append(
                self.place_order(
                    desired,
                    current_position_size=current_position_size,
                    best_bid=best_bid,
                    best_ask=best_ask,
                    oracle_price=oracle_price,
                )
            )

        desired_roles = {order.role for order in desired_orders}
        for current in current_orders:
            if current.coin != symbol or current.role in desired_roles:
                continue
            receipts.append(self.cancel_order(symbol, current))

        return receipts

    def place_order(
        self,
        desired: DesiredOrder,
        *,
        current_position_size: float,
        best_bid: float,
        best_ask: float,
        oracle_price: float | None,
    ) -> ExecutionReceipt:
        validation = self.validator.validate_order(
            symbol=desired.symbol,
            side=desired.side,
            price=desired.price,
            size=desired.size,
            reduce_only=desired.reduce_only,
            current_position_size=current_position_size,
            best_bid=best_bid,
            best_ask=best_ask,
            oracle_price=oracle_price,
        )
        if not validation.valid:
            return self._receipt(
                symbol=desired.symbol,
                cloid=desired.cloid,
                action="place",
                decision=ReconciliationDecision.PLACE,
                success=False,
                status=OrderState.REJECTED,
                message=validation.reason,
            )

        try:
            self._set_expires_after(desired.expires_after)
            self.nonce_manager.next_nonce()
            response = self.exchange.order(
                name=desired.symbol,
                is_buy=desired.side == TradeSide.LONG,
                sz=validation.size,
                limit_px=validation.price,
                order_type=desired.order_type,
                reduce_only=desired.reduce_only,
                cloid=Cloid.from_str(desired.cloid),
            )
            return self._receipt_from_response(
                symbol=desired.symbol,
                cloid=desired.cloid,
                action="place",
                decision=ReconciliationDecision.PLACE,
                response=response,
            )
        except ClockDriftError as exc:
            return self._receipt(
                symbol=desired.symbol,
                cloid=desired.cloid,
                action="place",
                decision=ReconciliationDecision.AWAIT_RESOLUTION,
                success=False,
                status=OrderState.UNKNOWN,
                message=str(exc),
            )
        except Exception as exc:
            return self._resolve_or_fail(desired.symbol, desired.cloid, "place", exc)
        finally:
            self._set_expires_after(None)

    def modify_order(
        self,
        current: LiveOrderState,
        desired: DesiredOrder,
        *,
        current_position_size: float,
        best_bid: float,
        best_ask: float,
        oracle_price: float | None,
    ) -> ExecutionReceipt:
        validation = self.validator.validate_order(
            symbol=desired.symbol,
            side=desired.side,
            price=desired.price,
            size=desired.size,
            reduce_only=desired.reduce_only,
            current_position_size=current_position_size,
            best_bid=best_bid,
            best_ask=best_ask,
            oracle_price=oracle_price,
        )
        if not validation.valid:
            return self._receipt(
                symbol=desired.symbol,
                cloid=desired.cloid,
                action="modify",
                decision=ReconciliationDecision.MODIFY,
                success=False,
                status=OrderState.REJECTED,
                message=validation.reason,
            )

        try:
            self.nonce_manager.next_nonce()
            response = self.exchange.modify_order(
                current.oid if current.oid else Cloid.from_str(current.cloid or desired.cloid),
                name=desired.symbol,
                is_buy=desired.side == TradeSide.LONG,
                sz=validation.size,
                limit_px=validation.price,
                order_type=desired.order_type,
                reduce_only=desired.reduce_only,
                cloid=Cloid.from_str(desired.cloid),
            )
            return self._receipt_from_response(
                symbol=desired.symbol,
                cloid=desired.cloid,
                action="modify",
                decision=ReconciliationDecision.MODIFY,
                response=response,
            )
        except Exception as exc:
            return self._resolve_or_fail(desired.symbol, desired.cloid, "modify", exc)

    def cancel_order(self, symbol: str, current: LiveOrderState) -> ExecutionReceipt:
        try:
            self.nonce_manager.next_nonce()
            if current.cloid is not None:
                response = self.exchange.cancel_by_cloid(symbol, Cloid.from_str(current.cloid))
            else:
                response = self.exchange.cancel(symbol, current.oid)
            return self._receipt_from_response(
                symbol=symbol,
                cloid=current.cloid or str(current.oid),
                action="cancel",
                decision=ReconciliationDecision.CANCEL,
                response=response,
            )
        except Exception as exc:
            return self._resolve_or_fail(symbol, current.cloid or str(current.oid), "cancel", exc)

    def reserve_request_weight(self) -> BudgetStatus:
        self.budgeter.note_rest_weight(0)
        return self.budgeter.evaluate(open_order_count=0)

    def noop(self) -> Any:
        nonce = self.nonce_manager.next_nonce()
        return self.exchange.noop(nonce)

    def _resolve_or_fail(
        self,
        symbol: str,
        cloid: str,
        action: str,
        exc: Exception,
    ) -> ExecutionReceipt:
        if self.ambiguous_resolver is None:
            return self._receipt(
                symbol=symbol,
                cloid=cloid,
                action=action,
                decision=ReconciliationDecision.AWAIT_RESOLUTION,
                success=False,
                status=OrderState.UNKNOWN,
                message=str(exc),
            )
        pending = PendingActionState(
            symbol=symbol,
            cloid=cloid,
            first_seen_at=datetime.now(tz=UTC),
            last_error=str(exc),
        )
        outcome = self.ambiguous_resolver.resolve(pending)
        return self._receipt(
            symbol=symbol,
            cloid=cloid,
            action=action,
            decision=outcome.decision,
            success=outcome.decision != ReconciliationDecision.AWAIT_RESOLUTION,
            status=outcome.status,
            oid=outcome.oid,
            message=outcome.message or str(exc),
            raw_response=outcome.raw_response,
        )

    @staticmethod
    def _can_modify(current: LiveOrderState, desired: DesiredOrder) -> bool:
        current_is_buy = current.side.upper() in {"B", "BUY"}
        desired_is_buy = desired.side == TradeSide.LONG
        current_trigger = current.order_type.startswith("trigger")
        desired_trigger = "trigger" in desired.order_type
        return (
            current.reduce_only == desired.reduce_only
            and current_is_buy == desired_is_buy
            and current_trigger == desired_trigger
        )

    @staticmethod
    def _can_keep(current: LiveOrderState, desired: DesiredOrder) -> bool:
        if not HyperliquidExchangeExecutor._can_modify(current, desired):
            return False
        if abs(current.limit_price - desired.price) > 1e-9:
            return False
        if abs(current.size - desired.size) > 1e-9:
            return False
        return True

    def _set_expires_after(self, expires_after: int | None) -> None:
        self.exchange.set_expires_after(expires_after)

    def _receipt_from_response(
        self,
        *,
        symbol: str,
        cloid: str,
        action: str,
        decision: ReconciliationDecision,
        response: Any,
    ) -> ExecutionReceipt:
        payload = response if isinstance(response, Mapping) else {"response": response}
        status = OrderState.OPEN
        success = True
        message = ""
        oid = None
        if isinstance(payload.get("status"), str) and payload["status"].lower() == "err":
            success = False
            status = OrderState.REJECTED
            message = str(payload.get("response", "exchange error"))
        if isinstance(payload.get("response"), Mapping):
            data = payload["response"]
            oid = _extract_oid(data)
            raw_status = data.get("status")
            if isinstance(raw_status, str):
                status = _status_from_string(raw_status)
        return self._receipt(
            symbol=symbol,
            cloid=cloid,
            action=action,
            decision=decision,
            success=success,
            status=status,
            oid=oid,
            message=message,
            raw_response=dict(payload),
        )

    def _receipt(
        self,
        *,
        symbol: str,
        cloid: str,
        action: str,
        decision: ReconciliationDecision,
        success: bool,
        status: OrderState,
        message: str,
        oid: int | None = None,
        raw_response: dict[str, object] | None = None,
    ) -> ExecutionReceipt:
        return ExecutionReceipt(
            symbol=symbol,
            action=action,
            cloid=cloid,
            oid=oid,
            decision=decision,
            success=success,
            status=status,
            message=message,
            raw_response=raw_response or {},
            submitted_at=datetime.now(tz=UTC),
        )


def _status_from_string(value: str) -> OrderState:
    lowered = value.lower()
    if lowered == "scheduledcancel":
        return OrderState.SCHEDULED_CANCEL
    for state in OrderState:
        if state.value.lower() == lowered:
            return state
    return OrderState.UNKNOWN


def _extract_oid(payload: Mapping[str, object]) -> int | None:
    oid = payload.get("oid")
    if isinstance(oid, int):
        return oid
    if isinstance(oid, str) and oid.isdigit():
        return int(oid)
    order = payload.get("order")
    if isinstance(order, Mapping):
        return _extract_oid(order)
    return None
