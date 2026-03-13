from __future__ import annotations

import json
import time
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any

from rich.console import Console

from dex_llm.analytics.report import summarize_outcomes
from dex_llm.executor import HyperliquidExchangeExecutor
from dex_llm.executor.paper import PaperBroker
from dex_llm.features.extractor import FeatureExtractor
from dex_llm.integrations.hyperliquid_live import HyperliquidRestGateway, HyperliquidWsStateClient
from dex_llm.live_frame import LiveFrameBuilder, _channel_age_ms
from dex_llm.llm.router import RouterProtocol
from dex_llm.models import (
    AccountState,
    FeatureSnapshot,
    HyperliquidFrontendOrder,
    KillSwitchStatus,
    LiveStateSnapshot,
    MarketFrame,
    Playbook,
    PositionState,
    RestingOrderPlan,
    RiskAssessment,
    TradePlan,
    TradeSide,
)
from dex_llm.risk.policy import RiskPolicy


@dataclass(slots=True)
class StrategyState:
    frame: MarketFrame
    features: FeatureSnapshot
    plan: TradePlan
    risk: RiskAssessment
    updated_at: datetime


class BotRuntime:
    def __init__(
        self,
        *,
        symbol: str,
        user_address: str,
        heatmap_params: dict[str, str],
        allow_synthetic: bool,
        router: RouterProtocol,
        builder: LiveFrameBuilder,
        rest_gateway: HyperliquidRestGateway,
        ws_client: HyperliquidWsStateClient,
        executor: HyperliquidExchangeExecutor,
        risk_policy: RiskPolicy,
        max_leverage: float,
        strategy_interval_s: int = 1_800,
        sync_interval_s: int = 120,
        live: bool = False,
        execution_mode_live: bool = False,
        dex: str = "",
        console: Console | None = None,
    ) -> None:
        self.symbol = symbol
        self.user_address = user_address
        self.heatmap_params = heatmap_params
        self.allow_synthetic = allow_synthetic
        self.router = router
        self.builder = builder
        self.rest_gateway = rest_gateway
        self.ws_client = ws_client
        self.executor = executor
        self.risk_policy = risk_policy
        self.max_leverage = max_leverage
        self.strategy_interval_s = strategy_interval_s
        self.sync_interval_s = sync_interval_s
        self.live = live or execution_mode_live
        self.dex = dex
        self.console = console or Console()
        self.paper_broker = PaperBroker()
        self._strategy_state: StrategyState | None = None
        self._previous_position_side = TradeSide.FLAT

    def run(self, *, max_cycles: int | None = None) -> None:
        cycle = 0
        self.ws_client.connect(self.symbol, user_address=self.user_address)
        self.ws_client.wait_until_public_ready(timeout_s=self.rest_gateway.timeout_s)
        try:
            while max_cycles is None or cycle < max_cycles:
                cycle += 1
                cycle_started = datetime.now(tz=UTC)
                snapshot, fills, meta = self._capture_snapshot()
                pre_strategy_receipts: list[dict[str, object]] = []
                if not self.live:
                    pre_strategy_receipts = self._paper_mark_market(snapshot, cycle_started)
                position = self._position_state(snapshot, fills)
                refresh_strategy = self._should_refresh_strategy(cycle_started, position)
                if refresh_strategy:
                    self._strategy_state = self._compute_strategy_state(snapshot, fills, position)

                if self._strategy_state is None:
                    raise RuntimeError("strategy state was not initialized")

                effective_plan = self._effective_plan(
                    strategy_state=self._strategy_state,
                    position=position,
                    current_price=snapshot.order_book.mid_price,
                    now=cycle_started,
                )
                account = self._account_from_snapshot(snapshot)
                kill_switch = self._kill_switch_from_snapshot(
                    snapshot=snapshot,
                    position=position,
                    strategy_state=self._strategy_state,
                )
                risk = self.risk_policy.assess(
                    effective_plan,
                    account,
                    position,
                    kill_switch,
                )
                receipts, leverage_preflight = self._sync_orders(
                    snapshot=snapshot,
                    position=position,
                    account=account,
                    plan=effective_plan,
                    risk=risk,
                )
                receipts = pre_strategy_receipts + receipts
                dms = None
                if self.live:
                    dms = self.executor.schedule_dead_man_switch(
                        has_resting_entry=bool(effective_plan.resting_orders),
                        position_open=position.side != TradeSide.FLAT,
                        now=cycle_started,
                    )
                self._emit_cycle(
                    cycle=cycle,
                    position=position,
                    account=account,
                    plan=effective_plan,
                    risk=risk,
                    receipts=receipts,
                    leverage_preflight=leverage_preflight,
                    dms=dms,
                    strategy_refreshed=refresh_strategy,
                    meta=meta,
                )
                self._previous_position_side = position.side
                if max_cycles is not None and cycle >= max_cycles:
                    break
                sleep_for = self.sync_interval_s - (
                    datetime.now(tz=UTC) - cycle_started
                ).total_seconds()
                if sleep_for > 0:
                    self.ws_client.send_heartbeat_if_idle(idle_s=max(5.0, sleep_for / 2))
                    time.sleep(sleep_for)
        finally:
            self.ws_client.close()
            self.rest_gateway.close()
            if (
                self.builder.heatmap_client is not None
                and hasattr(self.builder.heatmap_client, "close")
            ):
                self.builder.heatmap_client.close()

    def _capture_snapshot(self) -> tuple[LiveStateSnapshot, list[object] | None, dict[str, object]]:
        snapshot = self.ws_client.snapshot().model_copy(
            update={"captured_at": datetime.now(tz=UTC)}
        )
        fills: list[object] | None = None
        fills_safe_complete = True
        private_source = "ws" if self.ws_client.private_state_ready() else "pending"
        private_bootstrap_error: str | None = None
        day_start = snapshot.order_book.captured_at.astimezone(UTC).replace(
            hour=0,
            minute=0,
            second=0,
            microsecond=0,
        )
        paginated_fills, fills_safe_complete = self.rest_gateway.paginate_user_fills_by_time(
            user=self.user_address,
            start_time=int(day_start.timestamp() * 1000),
            end_time=int(snapshot.order_book.captured_at.timestamp() * 1000),
        )
        fills = paginated_fills
        if not self.ws_client.private_state_ready():
            try:
                clearinghouse_state = self.rest_gateway.fetch_clearinghouse_state(
                    user=self.user_address,
                    dex=self.dex,
                )
                open_orders = self.rest_gateway.fetch_open_orders(
                    user=self.user_address,
                    dex=self.dex,
                )
                snapshot = self._apply_rest_private_bootstrap(
                    snapshot,
                    clearinghouse_state=clearinghouse_state,
                    open_orders=open_orders,
                    fills=fills,
                    bootstrapped_at=datetime.now(tz=UTC),
                )
                private_source = "rest_bootstrap"
            except Exception as exc:
                private_bootstrap_error = str(exc)
        elif fills is not None:
            snapshot = snapshot.model_copy(update={"recent_fills": fills})
        meta: dict[str, object] = {
            "snapshot": snapshot.model_dump(mode="json"),
            "fills_safe_complete": fills_safe_complete,
            "private_state_source": private_source,
            "private_ws_ready": self.ws_client.private_state_ready(),
        }
        if private_bootstrap_error is not None:
            meta["private_bootstrap_error"] = private_bootstrap_error
        return snapshot, fills, meta

    def _compute_strategy_state(
        self,
        snapshot: LiveStateSnapshot,
        fills: list[object] | None,
        position: PositionState,
    ) -> StrategyState:
        frame = self.builder.build_from_snapshot(
            snapshot,
            heatmap_params=self.heatmap_params,
            allow_synthetic=self.allow_synthetic,
            fills=fills,
        )
        frame = frame.model_copy(update={"position": position})
        features = FeatureExtractor().extract(frame)
        plan = self.router.route(frame, features)
        account = self._account_from_snapshot(snapshot)
        risk = self.risk_policy.assess(plan, account, position, frame.kill_switch)
        return StrategyState(
            frame=frame,
            features=features,
            plan=plan,
            risk=risk,
            updated_at=datetime.now(tz=UTC),
        )

    def _should_refresh_strategy(self, now: datetime, position: PositionState) -> bool:
        if self._strategy_state is None:
            return True
        if (now - self._strategy_state.updated_at).total_seconds() >= self.strategy_interval_s:
            return True
        return self._previous_position_side != TradeSide.FLAT and position.side == TradeSide.FLAT

    def _effective_plan(
        self,
        *,
        strategy_state: StrategyState,
        position: PositionState,
        current_price: float,
        now: datetime,
    ) -> TradePlan:
        plan = strategy_state.plan
        if position.side != TradeSide.FLAT:
            return plan
        if plan.playbook.value == "no_trade":
            return plan
        if plan.resting_orders:
            active_resting_orders = [
                order
                for order in plan.resting_orders
                if not self._resting_order_invalidated(order, current_price)
            ]
            if not active_resting_orders:
                return self._flat_plan("all resting entries invalidated before fill")
            if len(active_resting_orders) != len(plan.resting_orders):
                return plan.model_copy(
                    update={
                        "resting_orders": active_resting_orders,
                        "reason": "drop invalidated resting entries and keep the remaining bands",
                    }
                )
            return plan
        age_s = (now - strategy_state.updated_at).total_seconds()
        if age_s >= plan.ttl_min * 60:
            return self._flat_plan("directional entry expired before fill")
        if self._directional_entry_invalidated(plan, current_price):
            return self._flat_plan("directional entry invalidated before fill")
        return plan

    def _sync_orders(
        self,
        *,
        snapshot: LiveStateSnapshot,
        position: PositionState,
        account: AccountState,
        plan: TradePlan,
        risk: RiskAssessment,
    ) -> tuple[list[dict[str, object]], dict[str, object] | None]:
        best_bid = snapshot.order_book.best_bid
        best_ask = snapshot.order_book.best_ask
        oracle_price = snapshot.active_asset_ctx.oracle_price if snapshot.active_asset_ctx else None
        frame_timestamp = snapshot.captured_at or datetime.now(tz=UTC)

        if not self.live:
            receipts = self.paper_broker.sync_plan(
                symbol=self.symbol,
                plan=plan,
                risk=risk,
                frame_timestamp=frame_timestamp,
            )
            return [receipt.model_dump(mode="json") for receipt in receipts], None

        if plan.playbook.value == "no_trade" and position.side == TradeSide.FLAT:
            receipts = self.executor.reconcile_orders(
                symbol=self.symbol,
                desired_orders=[],
                current_orders=position.active_orders,
                current_position_size=0.0,
                best_bid=best_bid,
                best_ask=best_ask,
                oracle_price=oracle_price,
            )
            return [receipt.model_dump(mode="json") for receipt in receipts], None

        desired_orders = self.executor.build_orders_from_plan(
            symbol=self.symbol,
            plan=plan,
            risk=risk,
            frame_timestamp=frame_timestamp,
            revision=0,
        )
        needs_entry_work = any(not order.reduce_only for order in desired_orders)
        leverage_preflight = None
        if self.live and needs_entry_work and position.side == TradeSide.FLAT:
            leverage_result = self.executor.apply_leverage_preflight(
                symbol=self.symbol,
                target_leverage=self.executor.target_leverage,
                margin_mode=self.executor.margin_mode,
                current_leverage=position.live_leverage,
                max_leverage=account.max_leverage,
                recommended_notional=risk.recommended_notional,
                available_margin=account.available_margin,
            )
            leverage_preflight = leverage_result.model_dump(mode="json")
            if not leverage_result.valid:
                receipts = self.executor.reconcile_orders(
                    symbol=self.symbol,
                    desired_orders=[],
                    current_orders=position.active_orders,
                    current_position_size=0.0,
                    best_bid=best_bid,
                    best_ask=best_ask,
                    oracle_price=oracle_price,
                )
                return [receipt.model_dump(mode="json") for receipt in receipts], leverage_preflight

        receipts = self.executor.execute_plan(
            plan=plan,
            risk=risk,
            symbol=self.symbol,
            frame_timestamp=frame_timestamp,
            position=position,
            best_bid=best_bid,
            best_ask=best_ask,
            oracle_price=oracle_price,
        )
        return [receipt.model_dump(mode="json") for receipt in receipts], leverage_preflight

    def _position_state(
        self,
        snapshot: LiveStateSnapshot,
        fills: list[object] | None,
    ) -> PositionState:
        if not self.live:
            return self.paper_broker.paper_position_state(symbol=snapshot.symbol)
        return self.builder._build_position_state(
            symbol=snapshot.symbol,
            clearinghouse_state=snapshot.clearinghouse_state,
            open_orders=[
                HyperliquidFrontendOrder(
                    coin=order.coin,
                    side=order.side,
                    limit_price=order.limit_price,
                    size=order.size,
                    reduce_only=order.reduce_only,
                    is_trigger=order.is_trigger,
                    order_type=order.order_type,
                    oid=order.oid,
                    cloid=order.cloid,
                    trigger_price=order.trigger_price,
                    timestamp=order.timestamp or snapshot.order_book.captured_at,
                )
                for order in snapshot.open_orders
            ],
            fills=fills or snapshot.recent_fills,
            last_user_event=(
                snapshot.recent_user_events[-1]
                if snapshot.recent_user_events
                else None
            ),
        )

    def _account_from_snapshot(self, snapshot: LiveStateSnapshot) -> AccountState:
        equity = (
            snapshot.clearinghouse_state.margin_summary.account_value
            if snapshot.clearinghouse_state is not None
            else 10_000.0
        )
        if equity <= 0:
            equity = 10_000.0
        available_margin = (
            snapshot.clearinghouse_state.withdrawable
            if snapshot.clearinghouse_state is not None
            else equity
        )
        if available_margin <= 0:
            available_margin = equity
        base_account = AccountState(
            equity=equity,
            available_margin=available_margin,
            max_leverage=self.max_leverage,
        )
        if not self.live:
            return self.paper_broker.account_state(
                base_account,
                mark_price=snapshot.order_book.mid_price,
            )
        return base_account

    def _kill_switch_from_snapshot(
        self,
        *,
        snapshot: LiveStateSnapshot,
        position: PositionState,
        strategy_state: StrategyState,
    ) -> KillSwitchStatus:
        info_latency_ms = _channel_age_ms(snapshot, ("l2Book", "candle", "bbo", "activeAssetCtx"))
        private_latency_ms = _channel_age_ms(
            snapshot,
            ("webData3", "orderUpdates", "userFills", "userEvents", "restPrivateBootstrap"),
        )
        heatmap_provider = strategy_state.frame.metadata.get("heatmap_provider", "unknown")
        heatmap_error = strategy_state.frame.metadata.get("heatmap_error")
        return self.builder.kill_switch_policy.evaluate(
            frame_timestamp=snapshot.order_book.captured_at,
            position=position,
            info_latency_ms=info_latency_ms,
            private_state_latency_ms=private_latency_ms,
            private_state_required=True,
            private_state_loaded=snapshot.clearinghouse_state is not None,
            heatmap_provider=str(heatmap_provider),
            heatmap_error=str(heatmap_error) if heatmap_error is not None else None,
        )

    def _emit_cycle(
        self,
        *,
        cycle: int,
        position: PositionState,
        account: AccountState,
        plan: TradePlan,
        risk: RiskAssessment,
        receipts: list[dict[str, object]],
        leverage_preflight: dict[str, object] | None,
        dms: Any,
        strategy_refreshed: bool,
        meta: dict[str, object],
    ) -> None:
        payload = {
            "cycle": cycle,
            "mode": "live" if self.live else "paper",
            "strategy_refreshed": strategy_refreshed,
            "position_side": position.side,
            "open_orders": position.open_orders,
            "account_equity": account.equity,
            "plan": plan.model_dump(mode="json"),
            "risk": risk.model_dump(mode="json"),
            "leverage_preflight": leverage_preflight,
            "execution_receipts": receipts,
            "dead_man_switch": dms,
            "live_state": {
                "private_state_source": meta.get("private_state_source"),
                "private_ws_ready": meta.get("private_ws_ready"),
                "fills_safe_complete": meta.get("fills_safe_complete"),
            },
        }
        if not self.live:
            payload["paper_state"] = self.paper_broker.state_payload(
                mark_price=self._strategy_state.frame.current_price
                if self._strategy_state is not None
                else 0.0
            )
            payload["paper_summary"] = summarize_outcomes(self.paper_broker.outcomes)
        self.console.print_json(json.dumps(payload))

    def _paper_mark_market(
        self,
        snapshot: LiveStateSnapshot,
        now: datetime,
    ) -> list[dict[str, object]]:
        receipts = self.paper_broker.mark_market(
            symbol=self.symbol,
            price_candle=snapshot.candles_5m[-1] if snapshot.candles_5m else None,
            best_bid=snapshot.order_book.best_bid,
            best_ask=snapshot.order_book.best_ask,
            now=now,
        )
        return [receipt.model_dump(mode="json") for receipt in receipts]

    @staticmethod
    def _apply_rest_private_bootstrap(
        snapshot: LiveStateSnapshot,
        *,
        clearinghouse_state: object,
        open_orders: list[object],
        fills: list[object] | None,
        bootstrapped_at: datetime,
    ) -> LiveStateSnapshot:
        channel_timestamps = dict(snapshot.channel_timestamps)
        channel_timestamps["restPrivateBootstrap"] = bootstrapped_at
        channel_snapshot_flags = dict(snapshot.channel_snapshot_flags)
        channel_snapshot_flags["restPrivateBootstrap"] = False
        return snapshot.model_copy(
            update={
                "clearinghouse_state": clearinghouse_state,
                "open_orders": open_orders,
                "recent_fills": fills if fills is not None else snapshot.recent_fills,
                "channel_timestamps": channel_timestamps,
                "channel_snapshot_flags": channel_snapshot_flags,
            }
        )

    @staticmethod
    def _resting_order_invalidated(order: RestingOrderPlan, current_price: float) -> bool:
        if order.side == TradeSide.LONG:
            return current_price <= order.invalid_if
        return current_price >= order.invalid_if

    @staticmethod
    def _directional_entry_invalidated(plan: TradePlan, current_price: float) -> bool:
        if plan.side == TradeSide.LONG:
            return current_price <= plan.invalid_if
        if plan.side == TradeSide.SHORT:
            return current_price >= plan.invalid_if
        return False

    @staticmethod
    def _flat_plan(reason: str) -> TradePlan:
        return TradePlan(
            playbook=Playbook.NO_TRADE,
            side=TradeSide.FLAT,
            entry_band=(0.0, 0.0),
            invalid_if=0.0,
            tp1=0.0,
            tp2=0.0,
            ttl_min=0,
            reason=reason,
            resting_orders=[],
        )
