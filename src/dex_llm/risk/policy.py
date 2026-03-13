from __future__ import annotations

from dex_llm.models import (
    AccountState,
    KillSwitchStatus,
    Playbook,
    PositionState,
    RestingOrderPlan,
    RiskAssessment,
    TradePlan,
    TradeSide,
)


class RiskPolicy:
    def __init__(
        self,
        long_notional_fraction: float = 1.0,
        short_notional_fraction: float = 0.4,
        long_target_leverage: int = 20,
        short_target_leverage: int = 15,
    ) -> None:
        if long_notional_fraction <= 0 or short_notional_fraction <= 0:
            raise ValueError("side notional fractions must be positive")
        if long_target_leverage <= 0 or short_target_leverage <= 0:
            raise ValueError("side target leverage must be positive")
        self.side_notional_fraction = {
            TradeSide.LONG: long_notional_fraction,
            TradeSide.SHORT: short_notional_fraction,
        }
        self.side_target_leverage = {
            TradeSide.LONG: long_target_leverage,
            TradeSide.SHORT: short_target_leverage,
        }

    def assess(
        self,
        plan: TradePlan,
        account: AccountState,
        position: PositionState,
        kill_switch: KillSwitchStatus | None = None,
    ) -> RiskAssessment:
        if (
            kill_switch is not None
            and not kill_switch.allow_new_trades
            and position.side == TradeSide.FLAT
        ):
            reason = kill_switch.reasons[0] if kill_switch.reasons else "kill switch active"
            return RiskAssessment(allowed=False, reason=reason)

        if plan.playbook == Playbook.NO_TRADE or plan.side == TradeSide.FLAT:
            return RiskAssessment(allowed=False, reason="plan requests hold/close only")

        has_pending_entry = any(not order.reduce_only for order in position.active_orders)
        if position.entries_blocked_reduce_only or has_pending_entry:
            return RiskAssessment(
                allowed=False,
                reason="entry workflow already exists; reconcile live orders first",
            )

        if position.side != TradeSide.FLAT or position.quantity > 0:
            return RiskAssessment(
                allowed=False,
                reason="single-position mode blocks averaging down",
            )

        order = self._target_order(plan)
        if order is None:
            return RiskAssessment(allowed=False, reason="plan does not contain an actionable entry")

        quantity, notional = self._size_order(order=order, account=account)
        if quantity <= 0 or notional <= 0:
            return RiskAssessment(allowed=False, reason="no margin available")

        assessment = RiskAssessment(
            allowed=True,
            reason="side-based sizing checks passed",
            recommended_quantity=quantity,
            recommended_notional=notional,
            risk_budget=notional,
        )
        if plan.resting_orders:
            assessment.resting_order_quantities = [quantity]
        return assessment

    @staticmethod
    def _target_order(plan: TradePlan) -> RestingOrderPlan | TradePlan | None:
        if not plan.resting_orders:
            return plan
        if len(plan.resting_orders) != 1:
            return None
        return plan.resting_orders[0]

    def _size_order(
        self,
        *,
        order: RestingOrderPlan | TradePlan,
        account: AccountState,
    ) -> tuple[float, float]:
        entry_price = sum(order.entry_band) / 2
        if entry_price <= 0:
            return 0.0, 0.0
        leverage = min(account.max_leverage, float(self.side_target_leverage[order.side]))
        notional = account.available_margin * leverage * self.side_notional_fraction[order.side]
        if notional <= 0:
            return 0.0, 0.0
        return notional / entry_price, notional
