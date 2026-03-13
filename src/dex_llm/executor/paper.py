from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime

from dex_llm.models import (
    AccountState,
    Candle,
    ExecutionMode,
    ExecutionReceipt,
    LiveOrderState,
    OrderRole,
    OrderState,
    PaperOrderTicket,
    Playbook,
    PositionState,
    ReconciliationDecision,
    RiskAssessment,
    TradeOutcome,
    TradePlan,
    TradeSide,
)


class PaperExecutor:
    def build_ticket(
        self,
        plan: TradePlan,
        risk: RiskAssessment,
        account: AccountState,
    ) -> PaperOrderTicket:
        if not risk.allowed:
            raise ValueError(risk.reason)

        entry_price = sum(plan.entry_band) / 2
        leverage = min(account.max_leverage, risk.recommended_notional / account.equity)
        return PaperOrderTicket(
            side=plan.side,
            entry_price=entry_price,
            quantity=risk.recommended_quantity,
            invalid_if=plan.invalid_if,
            take_profit_1=plan.tp1,
            take_profit_2=plan.tp2,
            ttl_min=plan.ttl_min,
            leverage=leverage,
            playbook=plan.playbook,
        )


@dataclass(slots=True)
class PaperPendingEntry:
    id: str
    side: TradeSide
    entry_price: float
    quantity: float
    invalid_if: float
    tp1: float
    tp2: float
    ttl_min: int
    playbook: Playbook
    placed_at: datetime
    reason: str


@dataclass(slots=True)
class PaperOpenPosition:
    side: TradeSide
    entry_price: float
    quantity: float
    initial_quantity: float
    invalid_if: float
    tp1: float
    tp2: float
    playbook: Playbook
    opened_at: datetime
    tp1_filled: bool = False
    realized_pnl: float = 0.0


class PaperBroker:
    def __init__(self, *, enable_stop_loss: bool = True) -> None:
        self.pending_entries: list[PaperPendingEntry] = []
        self.position: PaperOpenPosition | None = None
        self.realized_pnl = 0.0
        self.outcomes: list[TradeOutcome] = []
        self.enable_stop_loss = enable_stop_loss

    def sync_plan(
        self,
        *,
        symbol: str,
        plan: TradePlan,
        risk: RiskAssessment,
        frame_timestamp: datetime,
    ) -> list[ExecutionReceipt]:
        if self.position is not None:
            if self.pending_entries:
                return self._clear_pending(symbol, "position open; clear stale paper entries")
            return []

        if plan.playbook == Playbook.NO_TRADE:
            return self._clear_pending(symbol, "no-trade plan; clear paper entries")

        desired_entries = self._build_desired_entries(plan=plan, risk=risk, now=frame_timestamp)
        receipts: list[ExecutionReceipt] = []
        current_by_id = {entry.id: entry for entry in self.pending_entries}
        desired_ids = {entry.id for entry in desired_entries}

        updated_entries: list[PaperPendingEntry] = []
        for desired in desired_entries:
            current = current_by_id.get(desired.id)
            if current is None:
                receipts.append(
                    self._receipt(
                        symbol=symbol,
                        action="paper_place",
                        cloid=desired.id,
                        decision=ReconciliationDecision.PLACE,
                        status=OrderState.OPEN,
                        message="paper entry armed",
                    )
                )
                updated_entries.append(desired)
                continue
            if self._can_keep(current, desired):
                receipts.append(
                    self._receipt(
                        symbol=symbol,
                        action="paper_keep",
                        cloid=desired.id,
                        decision=ReconciliationDecision.KEEP,
                        status=OrderState.OPEN,
                        message="paper entry unchanged",
                    )
                )
                updated_entries.append(current)
                continue
            receipts.append(
                self._receipt(
                    symbol=symbol,
                    action="paper_modify",
                    cloid=desired.id,
                    decision=ReconciliationDecision.MODIFY,
                    status=OrderState.OPEN,
                    message="paper entry updated",
                )
            )
            updated_entries.append(desired)

        for current in self.pending_entries:
            if current.id in desired_ids:
                continue
            receipts.append(
                self._receipt(
                    symbol=symbol,
                    action="paper_cancel",
                    cloid=current.id,
                    decision=ReconciliationDecision.CANCEL,
                    status=OrderState.CANCELED,
                    message="paper entry removed",
                )
            )

        self.pending_entries = updated_entries
        return receipts

    def close_position_market(
        self,
        *,
        symbol: str,
        price: float,
        now: datetime,
        reason: str,
    ) -> ExecutionReceipt:
        if self.position is None:
            return self._receipt(
                symbol=symbol,
                action="paper_hold",
                cloid="paper-position",
                decision=ReconciliationDecision.KEEP,
                status=OrderState.UNKNOWN,
                message="no paper position to close",
            )
        pnl = self._close_position(
            self.position,
            quantity=self.position.quantity,
            price=price,
            now=now,
        )
        return self._receipt(
            symbol=symbol,
            action="paper_close",
            cloid="paper-position",
            decision=ReconciliationDecision.CANCEL_PLACE,
            status=OrderState.FILLED,
            message=f"paper position closed by strategy review ({pnl:.2f}); {reason}",
        )

    def mark_market(
        self,
        *,
        symbol: str,
        price_candle: Candle | None,
        best_bid: float,
        best_ask: float,
        now: datetime,
    ) -> list[ExecutionReceipt]:
        receipts: list[ExecutionReceipt] = []
        if self.position is not None:
            receipts.extend(
                self._update_open_position(
                    symbol=symbol,
                    price_candle=price_candle,
                    best_bid=best_bid,
                    best_ask=best_ask,
                    now=now,
                )
            )
            if self.position is not None:
                return receipts

        for entry in list(self.pending_entries):
            if self._entry_touched(entry, best_bid=best_bid, best_ask=best_ask):
                self.position = PaperOpenPosition(
                    side=entry.side,
                    entry_price=entry.entry_price,
                    quantity=entry.quantity,
                    initial_quantity=entry.quantity,
                    invalid_if=entry.invalid_if,
                    tp1=entry.tp1,
                    tp2=entry.tp2,
                    playbook=entry.playbook,
                    opened_at=now,
                )
                self.pending_entries = [
                    pending for pending in self.pending_entries if pending.id != entry.id
                ]
                receipts.append(
                    self._receipt(
                        symbol=symbol,
                        action="paper_fill_entry",
                        cloid=entry.id,
                        decision=ReconciliationDecision.PLACE,
                        status=OrderState.FILLED,
                        message="paper entry filled",
                    )
                )
                sibling_receipts = self._clear_pending(
                    symbol,
                    "entry filled; cancel sibling paper entries",
                )
                receipts.extend(sibling_receipts)
                break
        return receipts

    def paper_position_state(self, *, symbol: str) -> PositionState:
        active_orders = [
            LiveOrderState(
                coin=symbol,
                side="B" if entry.side == TradeSide.LONG else "A",
                limit_price=entry.entry_price,
                size=entry.quantity,
                reduce_only=False,
                is_trigger=False,
                order_type="limit",
                oid=0,
                cloid=entry.id,
                status=OrderState.OPEN,
                role=OrderRole.ENTRY,
                timestamp=entry.placed_at,
            )
            for entry in self.pending_entries
        ]
        if self.position is None:
            return PositionState(
                side=TradeSide.FLAT,
                open_orders=len(active_orders),
                active_orders=active_orders,
            )
        return PositionState(
            side=self.position.side,
            entry_price=self.position.entry_price,
            quantity=self.position.quantity,
            open_orders=0,
            active_orders=[],
        )

    def account_state(self, base_account: AccountState, *, mark_price: float) -> AccountState:
        unrealized = self.unrealized_pnl(mark_price=mark_price)
        equity = base_account.equity + self.realized_pnl + unrealized
        available_margin = base_account.available_margin + self.realized_pnl
        if available_margin <= 0:
            available_margin = equity
        if equity <= 0:
            equity = 1.0
        return AccountState(
            equity=equity,
            available_margin=max(1.0, available_margin),
            max_leverage=base_account.max_leverage,
        )

    def unrealized_pnl(self, *, mark_price: float) -> float:
        if self.position is None:
            return 0.0
        if self.position.side == TradeSide.LONG:
            return (mark_price - self.position.entry_price) * self.position.quantity
        return (self.position.entry_price - mark_price) * self.position.quantity

    def state_payload(self, *, mark_price: float) -> dict[str, object]:
        return {
            "pending_entries": [
                {
                    "id": entry.id,
                    "side": entry.side,
                    "entry_price": entry.entry_price,
                    "quantity": entry.quantity,
                    "invalid_if": entry.invalid_if,
                    "tp1": entry.tp1,
                    "tp2": entry.tp2,
                    "playbook": entry.playbook,
                    "placed_at": entry.placed_at.isoformat(),
                }
                for entry in self.pending_entries
            ],
            "position": None
            if self.position is None
            else {
                "side": self.position.side,
                "entry_price": self.position.entry_price,
                "quantity": self.position.quantity,
                "invalid_if": self.position.invalid_if,
                "tp1": self.position.tp1,
                "tp2": self.position.tp2,
                "tp1_filled": self.position.tp1_filled,
                "playbook": self.position.playbook,
                "opened_at": self.position.opened_at.isoformat(),
            },
            "realized_pnl": self.realized_pnl,
            "unrealized_pnl": self.unrealized_pnl(mark_price=mark_price),
            "closed_trades": len(self.outcomes),
        }

    def _build_desired_entries(
        self,
        *,
        plan: TradePlan,
        risk: RiskAssessment,
        now: datetime,
    ) -> list[PaperPendingEntry]:
        if plan.resting_orders:
            entries: list[PaperPendingEntry] = []
            for index, order in enumerate(plan.resting_orders):
                quantity = (
                    risk.resting_order_quantities[index]
                    if index < len(risk.resting_order_quantities)
                    else risk.recommended_quantity
                )
                entries.append(
                    PaperPendingEntry(
                        id=f"paper-resting-{order.side.value}-{index}",
                        side=order.side,
                        entry_price=sum(order.entry_band) / 2,
                        quantity=quantity,
                        invalid_if=order.invalid_if,
                        tp1=order.tp1,
                        tp2=order.tp2,
                        ttl_min=order.ttl_min,
                        playbook=plan.playbook,
                        placed_at=now,
                        reason=order.reason,
                    )
                )
            return entries
        if plan.side not in {TradeSide.LONG, TradeSide.SHORT} or risk.recommended_quantity <= 0:
            return []
        return [
            PaperPendingEntry(
                id=f"paper-entry-{plan.side.value}",
                side=plan.side,
                entry_price=sum(plan.entry_band) / 2,
                quantity=risk.recommended_quantity,
                invalid_if=plan.invalid_if,
                tp1=plan.tp1,
                tp2=plan.tp2,
                ttl_min=plan.ttl_min,
                playbook=plan.playbook,
                placed_at=now,
                reason=plan.reason,
            )
        ]

    def _update_open_position(
        self,
        *,
        symbol: str,
        price_candle: Candle | None,
        best_bid: float,
        best_ask: float,
        now: datetime,
    ) -> list[ExecutionReceipt]:
        assert self.position is not None
        position = self.position
        receipts: list[ExecutionReceipt] = []
        _ = price_candle
        stop_hit = self.enable_stop_loss and self._stop_triggered(
            position,
            best_bid=best_bid,
            best_ask=best_ask,
        )
        if stop_hit:
            pnl = self._close_position(
                position,
                quantity=position.quantity,
                price=position.invalid_if,
                now=now,
            )
            receipts.append(
                self._receipt(
                    symbol=symbol,
                    action="paper_stop",
                    cloid="paper-position",
                    decision=ReconciliationDecision.CANCEL,
                    status=OrderState.FILLED,
                    message=f"paper stop-loss filled ({pnl:.2f})",
                )
            )
            return receipts

        if not position.tp1_filled and self._take_profit_triggered(
            position.side,
            price=position.tp1,
            best_bid=best_bid,
            best_ask=best_ask,
        ):
            qty = position.initial_quantity / 2
            pnl = self._close_position(
                position,
                quantity=qty,
                price=position.tp1,
                now=now,
                final=False,
            )
            position.tp1_filled = True
            receipts.append(
                self._receipt(
                    symbol=symbol,
                    action="paper_tp1",
                    cloid="paper-position",
                    decision=ReconciliationDecision.MODIFY,
                    status=OrderState.FILLED,
                    message=f"paper take-profit 1 filled ({pnl:.2f})",
                )
            )

        if self.position is not None and self._take_profit_triggered(
            position.side,
            price=position.tp2,
            best_bid=best_bid,
            best_ask=best_ask,
        ):
            pnl = self._close_position(
                position,
                quantity=position.quantity,
                price=position.tp2,
                now=now,
            )
            receipts.append(
                self._receipt(
                    symbol=symbol,
                    action="paper_tp2",
                    cloid="paper-position",
                    decision=ReconciliationDecision.CANCEL,
                    status=OrderState.FILLED,
                    message=f"paper take-profit 2 filled ({pnl:.2f})",
                )
            )

        return receipts

    def _close_position(
        self,
        position: PaperOpenPosition,
        *,
        quantity: float,
        price: float,
        now: datetime,
        final: bool = True,
    ) -> float:
        if position.side == TradeSide.LONG:
            pnl = (price - position.entry_price) * quantity
        else:
            pnl = (position.entry_price - price) * quantity
        self.realized_pnl += pnl
        position.realized_pnl += pnl
        position.quantity = max(0.0, position.quantity - quantity)
        if final or position.quantity <= 0:
            hold_minutes = max(0, int((now - position.opened_at).total_seconds() / 60))
            self.outcomes.append(
                TradeOutcome(
                    playbook=position.playbook,
                    pnl=position.realized_pnl,
                    hold_minutes=hold_minutes,
                )
            )
            self.position = None
        return pnl

    def _clear_pending(self, symbol: str, message: str) -> list[ExecutionReceipt]:
        receipts = [
            self._receipt(
                symbol=symbol,
                action="paper_cancel",
                cloid=entry.id,
                decision=ReconciliationDecision.CANCEL,
                status=OrderState.CANCELED,
                message=message,
            )
            for entry in self.pending_entries
        ]
        self.pending_entries = []
        return receipts

    @staticmethod
    def _entry_touched(entry: PaperPendingEntry, *, best_bid: float, best_ask: float) -> bool:
        if entry.side == TradeSide.LONG:
            return best_ask <= entry.entry_price
        return best_bid >= entry.entry_price

    @staticmethod
    def _stop_triggered(position: PaperOpenPosition, *, best_bid: float, best_ask: float) -> bool:
        if position.side == TradeSide.LONG:
            return best_bid <= position.invalid_if
        return best_ask >= position.invalid_if

    @staticmethod
    def _take_profit_triggered(
        side: TradeSide,
        *,
        price: float,
        best_bid: float,
        best_ask: float,
    ) -> bool:
        if side == TradeSide.LONG:
            return best_bid >= price
        return best_ask <= price

    @staticmethod
    def _can_keep(current: PaperPendingEntry, desired: PaperPendingEntry) -> bool:
        return (
            current.side == desired.side
            and abs(current.entry_price - desired.entry_price) <= 1e-9
            and abs(current.quantity - desired.quantity) <= 1e-9
            and abs(current.invalid_if - desired.invalid_if) <= 1e-9
            and abs(current.tp1 - desired.tp1) <= 1e-9
            and abs(current.tp2 - desired.tp2) <= 1e-9
        )

    @staticmethod
    def _receipt(
        *,
        symbol: str,
        action: str,
        cloid: str,
        decision: ReconciliationDecision,
        status: OrderState,
        message: str,
    ) -> ExecutionReceipt:
        return ExecutionReceipt(
            mode=ExecutionMode.PAPER,
            symbol=symbol,
            action=action,
            cloid=cloid,
            decision=decision,
            success=True,
            status=status,
            message=message,
            submitted_at=datetime.now(tz=UTC),
        )
