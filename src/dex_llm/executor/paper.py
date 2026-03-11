from __future__ import annotations

from dex_llm.models import AccountState, PaperOrderTicket, RiskAssessment, TradePlan


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

