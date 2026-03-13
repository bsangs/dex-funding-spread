from __future__ import annotations

from datetime import UTC, datetime

from dex_llm.executor.paper import PaperBroker
from dex_llm.models import Playbook, RestingOrderPlan, RiskAssessment, TradePlan, TradeSide


def test_paper_broker_fills_one_resting_entry_and_cancels_sibling() -> None:
    broker = PaperBroker()
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
                entry_band=(2098.0, 2100.0),
                invalid_if=2089.0,
                tp1=2110.0,
                tp2=2120.0,
                ttl_min=30,
                reason="lower long fade",
            ),
            RestingOrderPlan(
                side=TradeSide.SHORT,
                entry_band=(2120.0, 2122.0),
                invalid_if=2128.0,
                tp1=2110.0,
                tp2=2100.0,
                ttl_min=30,
                reason="upper short fade",
            ),
        ],
    )
    risk = RiskAssessment(
        allowed=True,
        reason="ok",
        recommended_quantity=0.1,
        resting_order_quantities=[0.1, 0.05],
        recommended_notional=300.0,
        risk_budget=10.0,
    )

    receipts = broker.sync_plan(
        symbol="ETH",
        plan=plan,
        risk=risk,
        frame_timestamp=datetime(2026, 3, 13, 0, 0, tzinfo=UTC),
    )
    assert len(receipts) == 2

    fill_receipts = broker.mark_market(
        symbol="ETH",
        price_candle=None,
        best_bid=2098.8,
        best_ask=2098.5,
        now=datetime(2026, 3, 13, 0, 1, tzinfo=UTC),
    )

    assert broker.position is not None
    assert broker.position.side == TradeSide.LONG
    assert broker.position.quantity == 0.1
    assert broker.pending_entries == []
    assert any(receipt.action == "paper_fill_entry" for receipt in fill_receipts)
    assert any(receipt.action == "paper_cancel" for receipt in fill_receipts)


def test_paper_broker_tracks_total_trade_pnl_across_tp1_and_tp2() -> None:
    broker = PaperBroker()
    plan = TradePlan(
        playbook=Playbook.MAGNET_FOLLOW,
        side=TradeSide.LONG,
        entry_band=(2100.0, 2100.0),
        invalid_if=2090.0,
        tp1=2110.0,
        tp2=2120.0,
        ttl_min=20,
        reason="follow upside",
    )
    risk = RiskAssessment(
        allowed=True,
        reason="ok",
        recommended_quantity=1.0,
        recommended_notional=2100.0,
        risk_budget=10.0,
    )

    broker.sync_plan(
        symbol="ETH",
        plan=plan,
        risk=risk,
        frame_timestamp=datetime(2026, 3, 13, 0, 0, tzinfo=UTC),
    )
    broker.mark_market(
        symbol="ETH",
        price_candle=None,
        best_bid=2100.0,
        best_ask=2100.0,
        now=datetime(2026, 3, 13, 0, 1, tzinfo=UTC),
    )
    broker.mark_market(
        symbol="ETH",
        price_candle=None,
        best_bid=2110.0,
        best_ask=2110.1,
        now=datetime(2026, 3, 13, 0, 2, tzinfo=UTC),
    )
    broker.mark_market(
        symbol="ETH",
        price_candle=None,
        best_bid=2120.0,
        best_ask=2120.1,
        now=datetime(2026, 3, 13, 0, 3, tzinfo=UTC),
    )

    assert broker.position is None
    assert len(broker.outcomes) == 1
    assert broker.outcomes[0].pnl == 15.0
    assert broker.realized_pnl == 15.0
