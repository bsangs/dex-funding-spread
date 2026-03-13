from __future__ import annotations

from dex_llm.models import (
    AccountState,
    KillSwitchStatus,
    Playbook,
    PositionState,
    RiskAssessment,
    TradePlan,
    TradeSide,
)


class RiskPolicy:
    def __init__(
        self,
        risk_per_trade_pct: float = 0.35,
        max_consecutive_losses: int = 2,
        notional_buffer_fraction: float = 0.35,
        cluster_fade_long_weight: float = 0.8,
        cluster_fade_short_weight: float = 0.3,
    ) -> None:
        self.risk_per_trade_pct = risk_per_trade_pct
        self.max_consecutive_losses = max_consecutive_losses
        self.notional_buffer_fraction = notional_buffer_fraction
        if cluster_fade_long_weight <= 0 or cluster_fade_short_weight <= 0:
            raise ValueError("cluster_fade side weights must be positive")
        self.cluster_fade_weights = {
            TradeSide.LONG: cluster_fade_long_weight,
            TradeSide.SHORT: cluster_fade_short_weight,
        }

    def assess(
        self,
        plan: TradePlan,
        account: AccountState,
        position: PositionState,
        kill_switch: KillSwitchStatus | None = None,
    ) -> RiskAssessment:
        if kill_switch is not None and not kill_switch.allow_new_trades:
            reason = kill_switch.reasons[0] if kill_switch.reasons else "kill switch active"
            return RiskAssessment(allowed=False, reason=reason)

        if plan.resting_orders:
            return self._assess_resting_orders(plan, account, position)

        if (
            plan.playbook in {Playbook.NO_TRADE, Playbook.DOUBLE_SWEEP}
            or plan.side == TradeSide.FLAT
        ):
            return RiskAssessment(allowed=False, reason="playbook is observational only")

        has_pending_entry = any(not order.reduce_only for order in position.active_orders)
        if position.entries_blocked_reduce_only or has_pending_entry:
            return RiskAssessment(
                allowed=False,
                reason="entry workflow already exists; reconcile live orders first",
            )

        if position.side != TradeSide.FLAT or position.quantity > 0:
            return RiskAssessment(allowed=False, reason="averaging down is disabled")

        if position.consecutive_losses_today >= self.max_consecutive_losses:
            return RiskAssessment(allowed=False, reason="daily loss streak limit reached")

        mid_entry = sum(plan.entry_band) / 2
        stop_distance = abs(mid_entry - plan.invalid_if)
        if stop_distance <= 0:
            return RiskAssessment(allowed=False, reason="invalid stop distance")

        risk_budget = account.equity * (self.risk_per_trade_pct / 100)
        quantity = risk_budget / stop_distance
        max_notional = (
            account.available_margin * account.max_leverage * self.notional_buffer_fraction
        )
        notional = quantity * mid_entry
        if notional > max_notional:
            quantity = max_notional / mid_entry
            notional = max_notional

        if quantity <= 0:
            return RiskAssessment(allowed=False, reason="no margin available")

        return RiskAssessment(
            allowed=True,
            reason="risk checks passed",
            recommended_quantity=quantity,
            recommended_notional=notional,
            risk_budget=risk_budget,
        )

    def _assess_resting_orders(
        self,
        plan: TradePlan,
        account: AccountState,
        position: PositionState,
    ) -> RiskAssessment:
        has_pending_entry = any(not order.reduce_only for order in position.active_orders)
        if position.entries_blocked_reduce_only or has_pending_entry:
            return RiskAssessment(
                allowed=False,
                reason="entry workflow already exists; reconcile live orders first",
            )
        if position.side != TradeSide.FLAT or position.quantity > 0:
            return RiskAssessment(allowed=False, reason="averaging down is disabled")
        if position.consecutive_losses_today >= self.max_consecutive_losses:
            return RiskAssessment(allowed=False, reason="daily loss streak limit reached")

        stop_distances = [
            abs((sum(order.entry_band) / 2) - order.invalid_if)
            for order in plan.resting_orders
        ]
        if not stop_distances or min(stop_distances) <= 0:
            return RiskAssessment(allowed=False, reason="invalid stop distance")

        risk_budget = account.equity * (self.risk_per_trade_pct / 100)
        total_weight = sum(self.cluster_fade_weights[order.side] for order in plan.resting_orders)
        if total_weight <= 0:
            return RiskAssessment(allowed=False, reason="invalid cluster fade side weights")

        quantities: list[float] = []
        notionals: list[float] = []
        for order in plan.resting_orders:
            weight = self.cluster_fade_weights[order.side]
            allocated_budget = risk_budget * (weight / total_weight)
            mid_entry = sum(order.entry_band) / 2
            stop_distance = abs(mid_entry - order.invalid_if)
            quantity = allocated_budget / stop_distance
            quantities.append(quantity)
            notionals.append(quantity * mid_entry)

        total_notional = sum(notionals)
        max_notional = (
            account.available_margin * account.max_leverage * self.notional_buffer_fraction
        )
        if total_notional > max_notional:
            scale = max_notional / total_notional
            quantities = [quantity * scale for quantity in quantities]
            notionals = [notional * scale for notional in notionals]
            total_notional = max_notional
        if not quantities or min(quantities) <= 0:
            return RiskAssessment(allowed=False, reason="no margin available")

        return RiskAssessment(
            allowed=True,
            reason="risk checks passed for resting cluster fade orders",
            recommended_quantity=max(quantities),
            resting_order_quantities=quantities,
            recommended_notional=total_notional,
            risk_budget=risk_budget,
        )
