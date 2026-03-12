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
    ) -> None:
        self.risk_per_trade_pct = risk_per_trade_pct
        self.max_consecutive_losses = max_consecutive_losses
        self.notional_buffer_fraction = notional_buffer_fraction

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
