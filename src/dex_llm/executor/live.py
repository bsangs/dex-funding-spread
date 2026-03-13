from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from decimal import ROUND_DOWN, Decimal
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
    ExecutionMode,
    ExecutionReceipt,
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
    stop_reference_price: float | None = None


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
        keep_price_tolerance_bps: float = 2.0,
        keep_size_tolerance_fraction: float = 0.001,
        margin_mode: MarginMode = MarginMode.ISOLATED,
        target_leverage: int = 10,
        enable_stop_loss: bool = True,
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
        self.keep_price_tolerance_bps = keep_price_tolerance_bps
        self.keep_size_tolerance_fraction = keep_size_tolerance_fraction
        self.margin_mode = margin_mode
        self.target_leverage = target_leverage
        self.enable_stop_loss = enable_stop_loss
        self._grouped_entry_cloids: set[str] = set()

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
        # Resting entries are meant to stay on the book until they fill or a
        # later reconciliation pass explicitly replaces or cancels them.
        _ = has_resting_entry, position_open, now
        return {"status": "skipped", "response": {"status": "disabled"}}

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
        try:
            if current_leverage is None or int(current_leverage) != target_leverage:
                self.exchange.update_leverage(
                    leverage=target_leverage,
                    name=symbol,
                    is_cross=margin_mode == MarginMode.CROSS,
                )
            if margin_mode == MarginMode.ISOLATED:
                required_margin = _quantize_usd_amount(
                    (recommended_notional / target_leverage)
                    * (1 + self.validator.leverage_buffer_fraction)
                )
                self.exchange.update_isolated_margin(amount=required_margin, name=symbol)
        except Exception as exc:
            return ValidationResult(
                valid=False,
                reason=f"leverage preflight update failed: {exc}",
            )
        return result

    def build_orders_from_plan(
        self,
        *,
        symbol: str,
        plan: TradePlan,
        risk: RiskAssessment,
        frame_timestamp: datetime,
        revision: int,
        strategy_id: str = "dex-llm",
        quantity_override: float | None = None,
    ) -> list[DesiredOrder]:
        if plan.resting_orders:
            return self.build_orders_from_resting_orders(
                symbol=symbol,
                resting_orders=plan.resting_orders,
                quantities=self._resting_quantities(
                    resting_orders=plan.resting_orders,
                    risk=risk,
                    quantity_override=quantity_override,
                ),
                frame_timestamp=frame_timestamp,
                revision=revision,
                strategy_id=strategy_id,
            )
        if plan.side not in {TradeSide.LONG, TradeSide.SHORT}:
            return []
        quantity = self._entry_quantity(risk, quantity_override=quantity_override)
        if quantity <= 0:
            return []
        entry_price = sum(plan.entry_band) / 2
        entry_side = plan.side
        exit_side = TradeSide.SHORT if entry_side == TradeSide.LONG else TradeSide.LONG
        entry_expires_after = None
        if plan.ttl_min <= 1:
            entry_expires_after = int(
                (frame_timestamp + timedelta(minutes=plan.ttl_min)).timestamp() * 1000
            )
        desired_orders = [
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
                stop_reference_price=plan.invalid_if,
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
        if self.enable_stop_loss:
            desired_orders.insert(
                1,
                DesiredOrder(
                    symbol=symbol,
                    side=exit_side,
                    price=self._aggressive_market_limit_price(
                        symbol=symbol,
                        side=exit_side,
                        reference_price=plan.invalid_if,
                    ),
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
            )
        return desired_orders

    def build_orders_from_resting_orders(
        self,
        *,
        symbol: str,
        resting_orders: list[RestingOrderPlan],
        quantities: list[float],
        frame_timestamp: datetime,
        revision: int,
        strategy_id: str = "dex-llm",
        entry_only: bool = True,
        side_filter: TradeSide | None = None,
    ) -> list[DesiredOrder]:
        if len(quantities) != len(resting_orders):
            raise ValueError("resting order quantities must align with resting orders")
        desired_orders: list[DesiredOrder] = []
        for index, (resting_order, quantity) in enumerate(
            zip(resting_orders, quantities, strict=False)
        ):
            if side_filter is not None and resting_order.side != side_filter:
                continue
            if quantity <= 0:
                continue
            if entry_only:
                desired_orders.append(
                    DesiredOrder(
                        symbol=symbol,
                        side=resting_order.side,
                        price=sum(resting_order.entry_band) / 2,
                        size=quantity,
                        role=OrderRole.ENTRY,
                        reduce_only=False,
                        order_type={"limit": {"tif": "Gtc"}},
                        cloid=build_deterministic_cloid(
                            strategy_id,
                            symbol,
                            frame_timestamp,
                            OrderRole.ENTRY,
                            revision + index,
                        ),
                        expires_after=None,
                        stop_reference_price=resting_order.invalid_if,
                    )
                )
                continue

            exit_side = TradeSide.SHORT if resting_order.side == TradeSide.LONG else TradeSide.LONG
            if self.enable_stop_loss:
                desired_orders.append(
                    DesiredOrder(
                        symbol=symbol,
                        side=exit_side,
                        price=self._aggressive_market_limit_price(
                            symbol=symbol,
                            side=exit_side,
                            reference_price=resting_order.invalid_if,
                        ),
                        size=quantity,
                        role=OrderRole.STOP_LOSS,
                        reduce_only=True,
                        order_type={
                            "trigger": {
                                "triggerPx": resting_order.invalid_if,
                                "isMarket": True,
                                "tpsl": "sl",
                            }
                        },
                        cloid=build_deterministic_cloid(
                            strategy_id,
                            symbol,
                            frame_timestamp,
                            OrderRole.STOP_LOSS,
                            revision + index,
                        ),
                    )
                )
            desired_orders.extend(
                [
                    DesiredOrder(
                        symbol=symbol,
                        side=exit_side,
                        price=resting_order.tp1,
                        size=quantity / 2,
                        role=OrderRole.TAKE_PROFIT_1,
                        reduce_only=True,
                        order_type={"limit": {"tif": "Gtc"}},
                        cloid=build_deterministic_cloid(
                            strategy_id,
                            symbol,
                            frame_timestamp,
                            OrderRole.TAKE_PROFIT_1,
                            revision + index,
                        ),
                    ),
                    DesiredOrder(
                        symbol=symbol,
                        side=exit_side,
                        price=resting_order.tp2,
                        size=quantity / 2,
                        role=OrderRole.TAKE_PROFIT_2,
                        reduce_only=True,
                        order_type={"limit": {"tif": "Gtc"}},
                        cloid=build_deterministic_cloid(
                            strategy_id,
                            symbol,
                            frame_timestamp,
                            OrderRole.TAKE_PROFIT_2,
                            revision + index,
                        ),
                    ),
                ]
            )
        return desired_orders

    def build_grouped_entry_orders(
        self,
        *,
        symbol: str,
        plan: TradePlan,
        risk: RiskAssessment,
        frame_timestamp: datetime,
        revision: int,
        strategy_id: str = "dex-llm",
    ) -> list[DesiredOrder]:
        target_order: RestingOrderPlan | TradePlan
        quantity: float
        if plan.resting_orders:
            target_order = plan.resting_orders[0]
            quantity = self._resting_quantities(resting_orders=plan.resting_orders, risk=risk)[0]
        else:
            target_order = plan
            quantity = self._entry_quantity(risk)
        if quantity <= 0:
            return []
        entry_price = sum(target_order.entry_band) / 2
        exit_side = TradeSide.SHORT if target_order.side == TradeSide.LONG else TradeSide.LONG
        desired_orders = [
            DesiredOrder(
                symbol=symbol,
                side=target_order.side,
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
                stop_reference_price=target_order.invalid_if,
            ),
            DesiredOrder(
                symbol=symbol,
                side=exit_side,
                price=target_order.tp2,
                size=quantity,
                role=OrderRole.TAKE_PROFIT_2,
                reduce_only=True,
                order_type={
                    "trigger": {
                        "triggerPx": target_order.tp2,
                        "isMarket": True,
                        "tpsl": "tp",
                    }
                },
                cloid=build_deterministic_cloid(
                    strategy_id,
                    symbol,
                    frame_timestamp,
                    OrderRole.TAKE_PROFIT_2,
                    revision,
                ),
            ),
        ]
        if self.enable_stop_loss:
            desired_orders.append(
                DesiredOrder(
                    symbol=symbol,
                    side=exit_side,
                    price=self._aggressive_market_limit_price(
                        symbol=symbol,
                        side=exit_side,
                        reference_price=target_order.invalid_if,
                    ),
                    size=quantity,
                    role=OrderRole.STOP_LOSS,
                    reduce_only=True,
                    order_type={
                        "trigger": {
                            "triggerPx": target_order.invalid_if,
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
                )
            )
        return desired_orders

    def execute_plan(
        self,
        *,
        plan: TradePlan,
        risk: RiskAssessment,
        symbol: str,
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

        if plan.playbook == Playbook.NO_TRADE:
            signed_position_size = self._signed_position_size(position)
            receipts = self.reconcile_orders(
                symbol=symbol,
                desired_orders=[],
                current_orders=position.active_orders,
                current_position_size=signed_position_size,
                best_bid=best_bid,
                best_ask=best_ask,
                oracle_price=oracle_price,
            )
            if position.side != TradeSide.FLAT:
                receipts.append(
                    self.close_position(
                        symbol=symbol,
                        signed_position_size=signed_position_size,
                        reason=plan.reason,
                    )
                )
            return receipts

        protection_error = self._protection_plan_error(plan)
        if protection_error is not None:
            return [
                self._receipt(
                    symbol=symbol,
                    cloid="invalid-protection-plan",
                    action="plan_guard",
                    decision=ReconciliationDecision.AWAIT_RESOLUTION,
                    success=False,
                    status=OrderState.REJECTED,
                    message=protection_error,
                )
            ]

        if position.side == TradeSide.FLAT:
            return self._submit_grouped_entry_workflow(
                symbol=symbol,
                plan=plan,
                risk=risk,
                frame_timestamp=frame_timestamp,
                revision=revision,
                current_orders=position.active_orders,
                best_bid=best_bid,
                best_ask=best_ask,
                oracle_price=oracle_price,
            )

        desired_orders = self.build_orders_from_plan(
            symbol=symbol,
            plan=plan,
            risk=risk,
            frame_timestamp=frame_timestamp,
            revision=revision,
            quantity_override=position.quantity,
        )
        if plan.resting_orders:
            if position.side != TradeSide.FLAT:
                desired_orders = self.build_orders_from_resting_orders(
                    symbol=symbol,
                    resting_orders=plan.resting_orders,
                    quantities=[position.quantity for _ in plan.resting_orders],
                    frame_timestamp=frame_timestamp,
                    revision=revision,
                    entry_only=False,
                    side_filter=position.side,
                )
        elif position.side == TradeSide.FLAT:
            desired_orders = [order for order in desired_orders if order.role == OrderRole.ENTRY]
        else:
            desired_orders = [order for order in desired_orders if order.role != OrderRole.ENTRY]
            desired_orders = self._preserve_existing_take_profit_orders(
                symbol=symbol,
                desired_orders=desired_orders,
                current_orders=position.active_orders,
            )

        receipts = self.reconcile_orders(
            symbol=symbol,
            desired_orders=desired_orders,
            current_orders=position.active_orders,
            current_position_size=self._signed_position_size(position),
            best_bid=best_bid,
            best_ask=best_ask,
            oracle_price=oracle_price,
        )
        if position.side == TradeSide.FLAT:
            exit_receipts = self._sync_protection_after_fill(
                symbol=symbol,
                plan=plan,
                risk=risk,
                frame_timestamp=frame_timestamp,
                revision=revision,
                receipts=receipts,
                current_orders=position.active_orders,
                best_bid=best_bid,
                best_ask=best_ask,
                oracle_price=oracle_price,
            )
            receipts.extend(exit_receipts)
        return receipts

    def _submit_grouped_entry_workflow(
        self,
        *,
        symbol: str,
        plan: TradePlan,
        risk: RiskAssessment,
        frame_timestamp: datetime,
        revision: int,
        current_orders: Iterable[LiveOrderState],
        best_bid: float,
        best_ask: float,
        oracle_price: float | None,
    ) -> list[ExecutionReceipt]:
        desired_orders = self.build_grouped_entry_orders(
            symbol=symbol,
            plan=plan,
            risk=risk,
            frame_timestamp=frame_timestamp,
            revision=revision,
        )
        if not desired_orders:
            return self.reconcile_orders(
                symbol=symbol,
                desired_orders=[],
                current_orders=current_orders,
                current_position_size=0.0,
                best_bid=best_bid,
                best_ask=best_ask,
                oracle_price=oracle_price,
            )

        parent_order = desired_orders[0]
        current_entry_orders = [
            order
            for order in current_orders
            if order.coin == symbol and not order.reduce_only
        ]
        receipts = [
            self.cancel_order(symbol, order)
            for order in current_orders
            if order.coin == symbol and order.reduce_only
        ]

        if len(current_entry_orders) == 1:
            current_entry = current_entry_orders[0]
            if (
                current_entry.cloid in self._grouped_entry_cloids
                and self._can_keep(current_entry, parent_order)
            ):
                receipts.append(
                    self._receipt(
                        symbol=symbol,
                        cloid=current_entry.cloid or parent_order.cloid,
                        action="keep",
                        decision=ReconciliationDecision.KEEP,
                        success=True,
                        status=current_entry.status,
                        message="existing grouped entry already matches desired state",
                    )
                )
                return receipts

        for current_entry in current_entry_orders:
            receipts.append(self.cancel_order(symbol, current_entry))

        receipts.extend(
            self._place_grouped_entry_orders(
                desired_orders=desired_orders,
                best_bid=best_bid,
                best_ask=best_ask,
                oracle_price=oracle_price,
            )
        )
        return receipts

    def _preserve_existing_take_profit_orders(
        self,
        *,
        symbol: str,
        desired_orders: list[DesiredOrder],
        current_orders: Iterable[LiveOrderState],
    ) -> list[DesiredOrder]:
        current_take_profits = [
            order
            for order in current_orders
            if order.coin == symbol
            and order.reduce_only
            and order.role in {OrderRole.TAKE_PROFIT_1, OrderRole.TAKE_PROFIT_2}
        ]
        if not current_take_profits:
            return desired_orders

        preserved_orders = [
            self._desired_order_from_live_take_profit(order)
            for order in current_take_profits
        ]
        return [
            order
            for order in desired_orders
            if order.role not in {OrderRole.TAKE_PROFIT_1, OrderRole.TAKE_PROFIT_2}
        ] + preserved_orders

    def _desired_order_from_live_take_profit(
        self,
        order: LiveOrderState,
    ) -> DesiredOrder:
        order_type: dict[str, object]
        if order.is_trigger or order.trigger_price is not None:
            order_type = {
                "trigger": {
                    "triggerPx": order.trigger_price or order.limit_price,
                    "isMarket": True,
                    "tpsl": "tp",
                }
            }
        else:
            order_type = {"limit": {"tif": "Gtc"}}
        return DesiredOrder(
            symbol=order.coin,
            side=self._trade_side_from_order(order),
            price=order.limit_price,
            size=order.size,
            role=order.role,
            reduce_only=True,
            order_type=order_type,
            cloid=order.cloid or str(order.oid),
        )

    def _place_grouped_entry_orders(
        self,
        *,
        desired_orders: list[DesiredOrder],
        best_bid: float,
        best_ask: float,
        oracle_price: float | None,
    ) -> list[ExecutionReceipt]:
        if len(desired_orders) not in {2, 3}:
            raise ValueError("grouped entry workflow expects entry+tp or entry+tp+sl orders")
        entry_order = desired_orders[0]
        child_orders = desired_orders[1:]
        entry_validation = self.validator.validate_order(
            symbol=entry_order.symbol,
            side=entry_order.side,
            price=entry_order.price,
            size=entry_order.size,
            reduce_only=False,
            current_position_size=0.0,
            best_bid=best_bid,
            best_ask=best_ask,
            oracle_price=oracle_price,
            margin_mode=self.margin_mode,
            target_leverage=self.target_leverage,
            stop_reference_price=entry_order.stop_reference_price,
        )
        if not entry_validation.valid:
            return [
                self._receipt(
                    symbol=entry_order.symbol,
                    cloid=entry_order.cloid,
                    action="place",
                    decision=ReconciliationDecision.PLACE,
                    success=False,
                    status=OrderState.REJECTED,
                    message=entry_validation.reason,
                )
            ]

        child_validations = []
        for child_order in child_orders:
            validation = self._normalize_grouped_child_order(child_order)
            if not validation.valid:
                return [
                    self._receipt(
                        symbol=child_order.symbol,
                        cloid=child_order.cloid,
                        action="place",
                        decision=ReconciliationDecision.PLACE,
                        success=False,
                        status=OrderState.REJECTED,
                        message=validation.reason,
                    )
                ]
            child_validations.append(validation)

        order_requests = [self._order_request(entry_order, entry_validation)] + [
            self._order_request(child_order, validation)
            for child_order, validation in zip(child_orders, child_validations, strict=False)
        ]
        try:
            self._set_expires_after(entry_order.expires_after)
            self.nonce_manager.next_nonce()
            response = self.exchange.bulk_orders(order_requests, grouping="normalTpsl")
            receipts = self._receipts_from_grouped_response(
                desired_orders=desired_orders,
                response=response,
            )
            if any(
                receipt.cloid == entry_order.cloid
                and receipt.success
                and receipt.status in {OrderState.OPEN, OrderState.FILLED}
                for receipt in receipts
            ):
                self._grouped_entry_cloids.add(entry_order.cloid)
            return receipts
        except ClockDriftError as exc:
            return [
                self._receipt(
                    symbol=entry_order.symbol,
                    cloid=entry_order.cloid,
                    action="place",
                    decision=ReconciliationDecision.AWAIT_RESOLUTION,
                    success=False,
                    status=OrderState.UNKNOWN,
                    message=str(exc),
                )
            ]
        except Exception as exc:
            return [
                self._resolve_or_fail(entry_order.symbol, entry_order.cloid, "place", exc)
            ]
        finally:
            self._set_expires_after(None)

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
        current_by_key = {
            self._order_key(order.role, self._trade_side_from_order(order)): order
            for order in current_orders
            if order.coin == symbol
        }
        receipts: list[ExecutionReceipt] = []

        for desired in desired_orders:
            current = current_by_key.get(self._order_key(desired.role, desired.side))
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

        desired_keys = {self._order_key(order.role, order.side) for order in desired_orders}
        for current in current_orders:
            current_key = self._order_key(current.role, self._trade_side_from_order(current))
            if current.coin != symbol or current_key in desired_keys:
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
            margin_mode=self.margin_mode,
            target_leverage=self.target_leverage,
            stop_reference_price=desired.stop_reference_price,
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
                order_type=self._normalized_order_type(
                    desired.symbol,
                    desired.order_type,
                ),
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
            margin_mode=self.margin_mode,
            target_leverage=self.target_leverage,
            stop_reference_price=desired.stop_reference_price,
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
                order_type=self._normalized_order_type(
                    desired.symbol,
                    desired.order_type,
                ),
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

    def _can_keep(self, current: LiveOrderState, desired: DesiredOrder) -> bool:
        if not self._can_modify(current, desired):
            return False
        if desired.price <= 0:
            return False
        price_delta_bps = abs(current.limit_price - desired.price) / desired.price * 10_000
        if price_delta_bps > self.keep_price_tolerance_bps:
            return False
        if desired.size <= 0:
            return False
        size_delta_fraction = abs(current.size - desired.size) / desired.size
        if size_delta_fraction > self.keep_size_tolerance_fraction:
            return False
        return True

    @staticmethod
    def _order_key(role: OrderRole, side: TradeSide) -> tuple[OrderRole, TradeSide]:
        return role, side

    @staticmethod
    def _trade_side_from_order(order: LiveOrderState) -> TradeSide:
        return TradeSide.LONG if order.side.upper() in {"B", "BUY"} else TradeSide.SHORT

    def _sync_protection_after_fill(
        self,
        *,
        symbol: str,
        plan: TradePlan,
        risk: RiskAssessment,
        frame_timestamp: datetime,
        revision: int,
        receipts: list[ExecutionReceipt],
        current_orders: Iterable[LiveOrderState],
        best_bid: float,
        best_ask: float,
        oracle_price: float | None,
    ) -> list[ExecutionReceipt]:
        filled_entries = [
            receipt
            for receipt in receipts
            if receipt.action in {"place", "modify"}
            and receipt.status == OrderState.FILLED
            and receipt.decision in {ReconciliationDecision.PLACE, ReconciliationDecision.MODIFY}
        ]
        if not filled_entries:
            return []

        side_filter = None
        if plan.resting_orders:
            resting_quantities = self._resting_quantities(
                resting_orders=plan.resting_orders,
                risk=risk,
            )
            desired_map = {
                order.cloid: order
                for order in self.build_orders_from_resting_orders(
                    symbol=symbol,
                    resting_orders=plan.resting_orders,
                    quantities=resting_quantities,
                    frame_timestamp=frame_timestamp,
                    revision=revision,
                )
            }
            filled_sides = {
                desired_map[receipt.cloid].side
                for receipt in filled_entries
                if receipt.cloid in desired_map
            }
            if len(filled_sides) > 1:
                return [
                    self._receipt(
                        symbol=symbol,
                        cloid="cluster-fade-safe-fail",
                        action="safe_fail",
                        decision=ReconciliationDecision.AWAIT_RESOLUTION,
                        success=False,
                        status=OrderState.UNKNOWN,
                        message=(
                            "multiple cluster_fade entries filled in one cycle; "
                            "skip new orders until next sync"
                        ),
                    )
                ]
            for receipt in filled_entries:
                desired = desired_map.get(receipt.cloid)
                if desired is not None:
                    side_filter = desired.side
                    break
            if side_filter is None:
                return []
            sibling_cancels = self._cancel_opposing_entries(
                symbol=symbol,
                receipts=receipts,
                desired_map=desired_map,
                keep_side=side_filter,
            )
            desired_orders = self.build_orders_from_resting_orders(
                symbol=symbol,
                resting_orders=plan.resting_orders,
                quantities=resting_quantities,
                frame_timestamp=frame_timestamp,
                revision=revision,
                entry_only=False,
                side_filter=side_filter,
            )
            protection_size = next(
                (order.size for order in desired_orders if order.role == OrderRole.STOP_LOSS),
                next(
                    (
                        order.size
                        for order in desired_orders
                        if order.role == OrderRole.TAKE_PROFIT_2
                    ),
                    0.0,
                ),
            )
        else:
            if plan.side == TradeSide.FLAT:
                return []
            full_orders = self.build_orders_from_plan(
                symbol=symbol,
                plan=plan,
                risk=risk,
                frame_timestamp=frame_timestamp,
                revision=revision,
            )
            desired_orders = [order for order in full_orders if order.role != OrderRole.ENTRY]
            sibling_cancels = []
            protection_size = self._entry_quantity(risk)

        protection_receipts = self.reconcile_orders(
            symbol=symbol,
            desired_orders=desired_orders,
            current_orders=current_orders,
            current_position_size=self._signed_size(side_filter or plan.side, protection_size),
            best_bid=best_bid,
            best_ask=best_ask,
            oracle_price=oracle_price,
        )
        combined_receipts = sibling_cancels + protection_receipts
        if desired_orders and not self._protection_orders_secured(
            desired_orders=desired_orders,
            current_orders=current_orders,
            receipts=protection_receipts,
        ):
            signed_position_size = self._signed_size(side_filter or plan.side, protection_size)
            combined_receipts.append(
                self._emergency_close_position(
                    symbol=symbol,
                    signed_position_size=signed_position_size,
                    reason="filled entry was not fully protected; fail closed",
                )
            )
        return combined_receipts

    @staticmethod
    def _protection_plan_error(plan: TradePlan) -> str | None:
        if plan.resting_orders:
            if len(plan.resting_orders) > 1:
                return "single-position mode allows at most one pending entry"
            for index, order in enumerate(plan.resting_orders, start=1):
                error = HyperliquidExchangeExecutor._protection_level_error(
                    side=order.side,
                    entry_band=order.entry_band,
                    invalid_if=order.invalid_if,
                    tp1=order.tp1,
                    tp2=order.tp2,
                )
                if error is not None:
                    return f"resting order {index} is missing valid exits: {error}"
            return None

        if plan.side == TradeSide.FLAT:
            return None

        error = HyperliquidExchangeExecutor._protection_level_error(
            side=plan.side,
            entry_band=plan.entry_band,
            invalid_if=plan.invalid_if,
            tp1=plan.tp1,
            tp2=plan.tp2,
        )
        if error is None:
            return None
        return f"actionable trade plan is missing valid exits: {error}"

    @staticmethod
    def _protection_level_error(
        *,
        side: TradeSide,
        entry_band: tuple[float, float],
        invalid_if: float,
        tp1: float,
        tp2: float,
    ) -> str | None:
        if side == TradeSide.FLAT:
            return None

        band_low, band_high = entry_band
        if band_low <= 0 or band_high <= 0:
            return "entry band must be positive"
        if invalid_if <= 0 or tp1 <= 0 or tp2 <= 0:
            return "invalidation level and both take-profit targets must be positive"

        mid_entry = (band_low + band_high) / 2
        if side == TradeSide.LONG:
            if invalid_if >= mid_entry:
                return "invalidation level must sit below the entry band"
            if tp1 <= mid_entry or tp2 <= mid_entry:
                return "take-profit targets must sit above the entry band"
            if tp2 < tp1:
                return "tp2 must be at or above tp1"
            return None

        if invalid_if <= mid_entry:
            return "invalidation level must sit above the entry band"
        if tp1 >= mid_entry or tp2 >= mid_entry:
            return "take-profit targets must sit below the entry band"
        if tp2 > tp1:
            return "tp2 must be at or below tp1"
        return None

    def _protection_orders_secured(
        self,
        *,
        desired_orders: list[DesiredOrder],
        current_orders: Iterable[LiveOrderState],
        receipts: list[ExecutionReceipt],
    ) -> bool:
        current_by_key = {
            self._order_key(order.role, self._trade_side_from_order(order)): order
            for order in current_orders
            if order.coin
        }
        for desired in desired_orders:
            current = current_by_key.get(self._order_key(desired.role, desired.side))
            if current is None:
                if not self._has_successful_receipt(
                    receipts,
                    cloid=desired.cloid,
                    decisions={ReconciliationDecision.PLACE},
                ):
                    return False
                continue

            if self._can_keep(current, desired):
                if not self._has_successful_receipt(
                    receipts,
                    cloid=current.cloid or desired.cloid,
                    decisions={ReconciliationDecision.KEEP},
                ):
                    return False
                continue

            if self._can_modify(current, desired):
                if not self._has_successful_receipt(
                    receipts,
                    cloid=desired.cloid,
                    decisions={ReconciliationDecision.MODIFY},
                ):
                    return False
                continue

            if not self._has_successful_receipt(
                receipts,
                cloid=desired.cloid,
                decisions={ReconciliationDecision.PLACE},
            ):
                return False
        return True

    @staticmethod
    def _has_successful_receipt(
        receipts: list[ExecutionReceipt],
        *,
        cloid: str,
        decisions: set[ReconciliationDecision],
    ) -> bool:
        terminal_failures = {OrderState.REJECTED, OrderState.UNKNOWN, OrderState.CANCELED}
        return any(
            receipt.cloid == cloid
            and receipt.decision in decisions
            and receipt.success
            and receipt.status not in terminal_failures
            for receipt in receipts
        )

    def _emergency_close_position(
        self,
        *,
        symbol: str,
        signed_position_size: float,
        reason: str,
    ) -> ExecutionReceipt:
        if signed_position_size == 0:
            return self._receipt(
                symbol=symbol,
                cloid="emergency-close-skipped",
                action="emergency_close",
                decision=ReconciliationDecision.AWAIT_RESOLUTION,
                success=False,
                status=OrderState.UNKNOWN,
                message=reason,
            )

        try:
            self.nonce_manager.next_nonce()
            response = self.exchange.market_close(symbol, sz=abs(signed_position_size))
            receipt = self._receipt_from_response(
                symbol=symbol,
                cloid="emergency-close",
                action="emergency_close",
                decision=ReconciliationDecision.CANCEL_PLACE,
                response=response,
            )
            return receipt.model_copy(update={"message": reason})
        except Exception as exc:
            return self._resolve_or_fail(symbol, "emergency-close", "emergency_close", exc)

    def close_position(
        self,
        *,
        symbol: str,
        signed_position_size: float,
        reason: str,
    ) -> ExecutionReceipt:
        return self._emergency_close_position(
            symbol=symbol,
            signed_position_size=signed_position_size,
            reason=reason,
        )

    def _normalize_grouped_child_order(self, desired: DesiredOrder) -> ValidationResult:
        meta = self.validator.asset_metadata.get(desired.symbol)
        if meta is None:
            return ValidationResult(
                valid=False,
                reason=f"missing asset metadata for {desired.symbol}",
            )
        if desired.price <= 0 or desired.size <= 0:
            return ValidationResult(valid=False, reason="price and size must be positive")
        price = self.validator.quantize_price(desired.symbol, desired.price, asset_meta=meta)
        size = self.validator.quantize_size(desired.symbol, desired.size, asset_meta=meta)
        if size <= 0:
            return ValidationResult(valid=False, reason="size rounds to zero")
        notional = price * size
        if notional < self.validator.min_notional:
            return ValidationResult(valid=False, reason="minimum order notional not met")
        return ValidationResult(
            valid=True,
            reason="ok",
            price=price,
            size=size,
            notional=notional,
        )

    def _order_request(
        self,
        desired: DesiredOrder,
        validation: ValidationResult,
    ) -> dict[str, object]:
        return {
            "coin": desired.symbol,
            "is_buy": desired.side == TradeSide.LONG,
            "sz": validation.size,
            "limit_px": validation.price,
            "order_type": self._normalized_order_type(
                desired.symbol,
                desired.order_type,
            ),
            "reduce_only": desired.reduce_only,
            "cloid": Cloid.from_str(desired.cloid),
        }

    def _normalized_order_type(
        self,
        symbol: str,
        order_type: dict[str, object],
    ) -> dict[str, object]:
        if "trigger" not in order_type:
            return order_type
        trigger = dict(order_type["trigger"])
        raw_trigger_price = float(trigger["triggerPx"])
        meta = self.validator.asset_metadata.get(symbol)
        if meta is None:
            raise ValueError(f"missing asset metadata for {symbol}")
        trigger["triggerPx"] = self.validator.quantize_price(
            symbol,
            raw_trigger_price,
            asset_meta=meta,
        )
        return {
            "trigger": {
                "isMarket": trigger["isMarket"],
                "triggerPx": trigger["triggerPx"],
                "tpsl": trigger["tpsl"],
            }
        }

    def _aggressive_market_limit_price(
        self,
        *,
        symbol: str,
        side: TradeSide,
        reference_price: float,
    ) -> float:
        slippage = getattr(self.exchange, "DEFAULT_SLIPPAGE", 0.05)
        if side == TradeSide.LONG:
            raw_price = reference_price * (1 + slippage)
        else:
            raw_price = reference_price * (1 - slippage)
        meta = self.validator.asset_metadata.get(symbol)
        if meta is None:
            raise ValueError(f"missing asset metadata for {symbol}")
        return self.validator.quantize_price(symbol, raw_price, asset_meta=meta)

    def _receipts_from_grouped_response(
        self,
        *,
        desired_orders: list[DesiredOrder],
        response: Any,
    ) -> list[ExecutionReceipt]:
        payload = response if isinstance(response, Mapping) else {"response": response}
        if isinstance(payload.get("status"), str) and payload["status"].lower() == "err":
            message = str(payload.get("response", "exchange error"))
            return [
                self._receipt(
                    symbol=desired.symbol,
                    cloid=desired.cloid,
                    action="place",
                    decision=ReconciliationDecision.PLACE,
                    success=False,
                    status=OrderState.REJECTED,
                    message=message,
                    raw_response=dict(payload),
                )
                for desired in desired_orders
            ]

        statuses: list[object] = []
        response_payload = payload.get("response")
        if isinstance(response_payload, Mapping):
            data = response_payload.get("data")
            if isinstance(data, Mapping):
                raw_statuses = data.get("statuses")
                if isinstance(raw_statuses, list):
                    statuses = raw_statuses

        if (
            len(statuses) == 1
            and len(desired_orders) > 1
            and isinstance(statuses[0], Mapping)
            and "error" in statuses[0]
        ):
            message = str(statuses[0]["error"])
            return [
                self._receipt(
                    symbol=desired.symbol,
                    cloid=desired.cloid,
                    action="place",
                    decision=ReconciliationDecision.PLACE,
                    success=False,
                    status=OrderState.REJECTED,
                    message=message,
                    raw_response=dict(payload),
                )
                for desired in desired_orders
            ]

        receipts: list[ExecutionReceipt] = []
        for index, desired in enumerate(desired_orders):
            status_payload = statuses[index] if index < len(statuses) else None
            receipts.append(
                self._receipt_from_grouped_status(
                    desired=desired,
                    status_payload=status_payload,
                    response_payload=dict(payload),
                )
            )
        return receipts

    def _receipt_from_grouped_status(
        self,
        *,
        desired: DesiredOrder,
        status_payload: object,
        response_payload: dict[str, object],
    ) -> ExecutionReceipt:
        if isinstance(status_payload, Mapping):
            if "error" in status_payload:
                return self._receipt(
                    symbol=desired.symbol,
                    cloid=desired.cloid,
                    action="place",
                    decision=ReconciliationDecision.PLACE,
                    success=False,
                    status=OrderState.REJECTED,
                    message=str(status_payload["error"]),
                    raw_response=response_payload,
                )
            if "resting" in status_payload and isinstance(status_payload["resting"], Mapping):
                return self._receipt(
                    symbol=desired.symbol,
                    cloid=desired.cloid,
                    action="place",
                    decision=ReconciliationDecision.PLACE,
                    success=True,
                    status=OrderState.OPEN,
                    oid=_extract_oid(status_payload["resting"]),
                    message="grouped order armed",
                    raw_response=response_payload,
                )
            if "filled" in status_payload and isinstance(status_payload["filled"], Mapping):
                return self._receipt(
                    symbol=desired.symbol,
                    cloid=desired.cloid,
                    action="place",
                    decision=ReconciliationDecision.PLACE,
                    success=True,
                    status=OrderState.FILLED,
                    oid=_extract_oid(status_payload["filled"]),
                    message="grouped order filled immediately",
                    raw_response=response_payload,
                )

        return self._receipt(
            symbol=desired.symbol,
            cloid=desired.cloid,
            action="place",
            decision=ReconciliationDecision.PLACE,
            success=True,
            status=OrderState.UNKNOWN,
            message="grouped order submitted",
            raw_response=response_payload,
        )

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
            nested_status = _first_nested_status(data)
            if isinstance(nested_status, Mapping):
                if "error" in nested_status:
                    error_message = str(nested_status["error"])
                    if action == "cancel" and _is_terminal_cancel_error(error_message):
                        success = True
                        status = OrderState.CANCELED
                        message = error_message
                    else:
                        success = False
                        status = OrderState.REJECTED
                        message = error_message
                elif "resting" in nested_status and isinstance(nested_status["resting"], Mapping):
                    oid = _extract_oid(nested_status["resting"]) or oid
                    status = OrderState.OPEN
                elif "filled" in nested_status and isinstance(nested_status["filled"], Mapping):
                    oid = _extract_oid(nested_status["filled"]) or oid
                    status = OrderState.FILLED
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
            mode=ExecutionMode.LIVE,
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

    @staticmethod
    def _entry_quantity(
        risk: RiskAssessment,
        *,
        quantity_override: float | None = None,
    ) -> float:
        if quantity_override is not None:
            return quantity_override
        return risk.recommended_quantity

    def _resting_quantities(
        self,
        *,
        resting_orders: list[RestingOrderPlan],
        risk: RiskAssessment,
        quantity_override: float | None = None,
    ) -> list[float]:
        if quantity_override is not None:
            return [quantity_override for _ in resting_orders]
        if len(risk.resting_order_quantities) == len(resting_orders):
            return list(risk.resting_order_quantities)
        return [risk.recommended_quantity for _ in resting_orders]

    def _cancel_opposing_entries(
        self,
        *,
        symbol: str,
        receipts: list[ExecutionReceipt],
        desired_map: Mapping[str, DesiredOrder],
        keep_side: TradeSide,
    ) -> list[ExecutionReceipt]:
        cancellations: list[ExecutionReceipt] = []
        for receipt in receipts:
            desired = desired_map.get(receipt.cloid)
            if desired is None or desired.role != OrderRole.ENTRY or desired.side == keep_side:
                continue
            if receipt.status not in {OrderState.OPEN, OrderState.TRIGGERED, OrderState.UNKNOWN}:
                continue
            current = LiveOrderState(
                coin=symbol,
                side="B" if desired.side == TradeSide.LONG else "A",
                limit_price=desired.price,
                size=desired.size,
                reduce_only=False,
                is_trigger=False,
                order_type="limit",
                oid=receipt.oid or 0,
                cloid=receipt.cloid,
                status=receipt.status,
                role=OrderRole.ENTRY,
            )
            cancellations.append(self.cancel_order(symbol, current))
        return cancellations

    @staticmethod
    def _signed_position_size(position: PositionState) -> float:
        return HyperliquidExchangeExecutor._signed_size(position.side, position.quantity)

    @staticmethod
    def _signed_size(side: TradeSide, quantity: float) -> float:
        if side == TradeSide.SHORT:
            return -abs(quantity)
        if side == TradeSide.LONG:
            return abs(quantity)
        return 0.0


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


def _first_nested_status(payload: Mapping[str, object]) -> Mapping[str, object] | None:
    data = payload.get("data")
    if not isinstance(data, Mapping):
        return None
    statuses = data.get("statuses")
    if not isinstance(statuses, list) or not statuses:
        return None
    first = statuses[0]
    if isinstance(first, Mapping):
        return first
    return None


def _is_terminal_cancel_error(message: str) -> bool:
    lowered = message.lower()
    return (
        "already canceled" in lowered
        or "never placed" in lowered
        or "filled" in lowered
    )


def _quantize_usd_amount(value: float) -> float:
    return float(Decimal(str(value)).quantize(Decimal("0.000001"), rounding=ROUND_DOWN))
