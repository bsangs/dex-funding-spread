from __future__ import annotations

import time
from dataclasses import dataclass
from datetime import UTC, datetime

from rich.console import Console

from dex_llm.executor import HyperliquidExchangeExecutor
from dex_llm.executor.paper import PaperBroker
from dex_llm.executor.safety import extract_role_from_cloid
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
    OrderRole,
    Playbook,
    PositionState,
    RestingOrderPlan,
    RiskAssessment,
    TradePlan,
    TradeSide,
    UserEventType,
)
from dex_llm.risk.policy import RiskPolicy

_LOG_DIVIDER = "=" * 70


@dataclass(slots=True)
class StrategyState:
    frame: MarketFrame
    features: FeatureSnapshot
    plan: TradePlan
    risk: RiskAssessment
    updated_at: datetime


@dataclass(slots=True)
class EntryRejectionBlock:
    strategy_updated_at: datetime
    reason: str


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
        strategy_interval_s: int = 900,
        sync_interval_s: int = 120,
        live: bool = False,
        execution_mode_live: bool = False,
        dex: str = "",
        enable_stop_loss: bool = True,
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
        self.paper_broker = PaperBroker(enable_stop_loss=enable_stop_loss)
        self._strategy_state: StrategyState | None = None
        self._event_block_count = 0
        self._boot_logged = False
        self._last_plan_signature: tuple[object, ...] | None = None
        self._last_position_signature: tuple[object, ...] | None = None
        self._last_reduce_only_signature: tuple[str, ...] | None = None
        self._last_entry_block_reason: str | None = None
        self._last_active_order_signature: tuple[tuple[object, ...], ...] | None = None
        self._seen_fill_keys: set[str] = set()
        self._seen_user_event_keys: set[str] = set()
        self._entry_rejection_block: EntryRejectionBlock | None = None

    def run(self, *, max_cycles: int | None = None) -> None:
        cycle = 0
        try:
            self.ws_client.connect(self.symbol, user_address=self.user_address)
            self.ws_client.wait_until_public_ready(timeout_s=self.rest_gateway.timeout_s)
            while max_cycles is None or cycle < max_cycles:
                if not self.ws_client.connection_alive():
                    self.ws_client.reconnect()
                    self.ws_client.wait_until_public_ready(timeout_s=self.rest_gateway.timeout_s)
                cycle += 1
                cycle_started = datetime.now(tz=UTC)
                snapshot, fills, meta = self._capture_snapshot()
                pre_strategy_receipts: list[dict[str, object]] = []
                if not self.live:
                    pre_strategy_receipts = self._paper_mark_market(snapshot, cycle_started)
                position = self._position_state(snapshot, fills)
                if self._should_refresh_strategy(now=cycle_started):
                    self._strategy_state = self._compute_strategy_state(snapshot, fills, position)
                    self._entry_rejection_block = None

                if self._strategy_state is None:
                    raise RuntimeError("strategy state was not initialized")

                effective_plan = self._effective_plan(
                    strategy_state=self._strategy_state,
                    position=position,
                    current_price=snapshot.order_book.mid_price,
                    now=cycle_started,
                    fills=fills or snapshot.recent_fills,
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
                receipts, _ = self._sync_orders(
                    snapshot=snapshot,
                    position=position,
                    account=account,
                    plan=effective_plan,
                    risk=risk,
                )
                receipts = pre_strategy_receipts + receipts
                self._update_entry_rejection_block(
                    strategy_state=self._strategy_state,
                    position=position,
                    receipts=receipts,
                )
                if self.live:
                    self.executor.schedule_dead_man_switch(
                        has_resting_entry=bool(effective_plan.resting_orders),
                        position_open=position.side != TradeSide.FLAT,
                        now=cycle_started,
                    )
                self._emit_cycle(
                    cycle=cycle,
                    snapshot=snapshot,
                    position=position,
                    plan=effective_plan,
                    risk=risk,
                    kill_switch=kill_switch,
                    receipts=receipts,
                    meta=meta,
                )
                if max_cycles is not None and cycle >= max_cycles:
                    break
                sleep_for = self.sync_interval_s - (
                    datetime.now(tz=UTC) - cycle_started
                ).total_seconds()
                if sleep_for > 0:
                    sent = self.ws_client.send_heartbeat_if_idle(
                        idle_s=max(5.0, sleep_for / 2)
                    )
                    if not sent and not self.ws_client.connection_alive():
                        self.ws_client.reconnect()
                        self.ws_client.wait_until_public_ready(
                            timeout_s=self.rest_gateway.timeout_s
                        )
                    time.sleep(sleep_for)
        except Exception as exc:
            self._emit_runtime_error(
                phase="startup" if cycle == 0 else "runtime",
                cycle=cycle if cycle > 0 else None,
                exc=exc,
            )
            raise
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
        snapshot = snapshot.model_copy(
            update={
                "candles_1h": self.rest_gateway.fetch_candles(self.symbol, "1h", limit=48),
                "candles_4h": self.rest_gateway.fetch_candles(self.symbol, "4h", limit=30),
            }
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
        previous_plan = self._strategy_state.plan if self._strategy_state is not None else None
        if position.side != TradeSide.FLAT:
            plan = self._position_management_plan(frame=frame, previous_plan=previous_plan)
        else:
            plan = self.router.route(
                frame,
                features,
                previous_plan=previous_plan,
            )
            if previous_plan is not None:
                plan = self._stabilize_plan(
                    frame=frame,
                    previous_plan=previous_plan,
                    candidate_plan=plan,
                )
        account = self._account_from_snapshot(snapshot)
        risk = self.risk_policy.assess(plan, account, position, frame.kill_switch)
        return StrategyState(
            frame=frame,
            features=features,
            plan=plan,
            risk=risk,
            updated_at=datetime.now(tz=UTC),
        )

    def _should_refresh_strategy(
        self,
        *,
        now: datetime,
    ) -> bool:
        if self._strategy_state is None:
            return True
        if (now - self._strategy_state.updated_at).total_seconds() >= self.strategy_interval_s:
            return True
        return False

    def _effective_plan(
        self,
        *,
        strategy_state: StrategyState,
        position: PositionState,
        current_price: float,
        now: datetime,
        fills: list[object] | None = None,
    ) -> TradePlan:
        plan = strategy_state.plan
        if position.side != TradeSide.FLAT:
            if plan.side not in {position.side, TradeSide.FLAT}:
                return self._flat_plan("position review flipped side; close instead of reversing")
            return plan
        if plan.playbook.value == "no_trade":
            return plan
        entry_rejection_block = getattr(self, "_entry_rejection_block", None)
        if (
            entry_rejection_block is not None
            and entry_rejection_block.strategy_updated_at == strategy_state.updated_at
        ):
            return self._flat_plan(
                f"entry paused after exchange rejection: {entry_rejection_block.reason}"
            )
        if self._has_recent_fill_since(strategy_state.updated_at, fills):
            return self._flat_plan("recent fill detected; wait for next 5m strategy review")
        if plan.resting_orders:
            active_resting_orders = [
                order
                for order in plan.resting_orders
                if not self._resting_order_invalidated(order, current_price)
                and not self._resting_order_expired(order, strategy_state.updated_at, now)
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
        expiry_minutes = plan.expected_touch_minutes or plan.ttl_min
        if age_s >= expiry_minutes * 60:
            return self._flat_plan("directional entry expired before fill")
        if self._directional_entry_invalidated(plan, current_price):
            return self._flat_plan("directional entry invalidated before fill")
        return plan

    @staticmethod
    def _stabilize_plan(
        *,
        frame: MarketFrame,
        previous_plan: TradePlan,
        candidate_plan: TradePlan,
    ) -> TradePlan:
        if (
            previous_plan.playbook == Playbook.NO_TRADE
            or candidate_plan.playbook == Playbook.NO_TRADE
            or previous_plan.side != candidate_plan.side
            or previous_plan.resting_orders
            or candidate_plan.resting_orders
        ):
            return candidate_plan

        previous_mid = sum(previous_plan.entry_band) / 2
        candidate_mid = sum(candidate_plan.entry_band) / 2
        entry_shift = abs(previous_mid - candidate_mid)
        tp_shift = abs(previous_plan.tp2 - candidate_plan.tp2)
        if entry_shift <= frame.atr * 0.25 and tp_shift <= frame.atr * 0.35:
            return previous_plan.model_copy(
                update={
                    "reason": (
                        "keep prior resting idea; "
                        "new cycle does not materially improve the level"
                    ),
                }
            )
        return candidate_plan

    @staticmethod
    def _position_management_plan(
        *,
        frame: MarketFrame,
        previous_plan: TradePlan | None,
    ) -> TradePlan:
        side = frame.position.side
        if (
            previous_plan is not None
            and previous_plan.playbook != Playbook.NO_TRADE
            and previous_plan.side == side
        ):
            return previous_plan.model_copy(
                update={"reason": "code-managed open position keeps the existing take-profit plan"}
            )

        default_tp = (
            frame.current_price + frame.atr
            if side == TradeSide.LONG
            else frame.current_price - frame.atr
        )
        invalid_if = (
            frame.current_price - frame.atr * 0.25
            if side == TradeSide.LONG
            else frame.current_price + frame.atr * 0.25
        )
        return TradePlan(
            playbook=Playbook.MAGNET_FOLLOW,
            side=side,
            entry_band=(frame.current_price, frame.current_price),
            invalid_if=invalid_if,
            tp1=default_tp,
            tp2=default_tp,
            ttl_min=5,
            reason="code-managed open position uses fallback take-profit placeholders only",
            touch_confidence=1.0,
            expected_touch_minutes=5,
        )

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
            if plan.playbook == Playbook.NO_TRADE and self.paper_broker.position is not None:
                receipt = self.paper_broker.close_position_market(
                    symbol=self.symbol,
                    price=snapshot.order_book.mid_price,
                    now=frame_timestamp,
                    reason=plan.reason,
                )
                return [receipt.model_dump(mode="json")], None
            receipts = self.paper_broker.sync_plan(
                symbol=self.symbol,
                plan=plan,
                risk=risk,
                frame_timestamp=frame_timestamp,
            )
            return [receipt.model_dump(mode="json") for receipt in receipts], None

        if plan.playbook == Playbook.NO_TRADE:
            signed_position_size = self.executor._signed_position_size(position)
            receipts = self.executor.reconcile_orders(
                symbol=self.symbol,
                desired_orders=[],
                current_orders=position.active_orders,
                current_position_size=signed_position_size,
                best_bid=best_bid,
                best_ask=best_ask,
                oracle_price=oracle_price,
            )
            if position.side != TradeSide.FLAT:
                receipts.append(
                    self.executor.close_position(
                        symbol=self.symbol,
                        signed_position_size=signed_position_size,
                        reason=plan.reason,
                    )
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
            target_leverage = min(
                self.executor.target_leverage_for_side(plan.side),
                max(1, int(account.max_leverage)),
            )
            leverage_result = self.executor.apply_leverage_preflight(
                symbol=self.symbol,
                target_leverage=target_leverage,
                margin_mode=self.executor.margin_mode,
                current_leverage=position.live_leverage,
                max_leverage=account.max_leverage,
                recommended_notional=risk.recommended_notional,
                available_margin=account.available_margin,
            )
            leverage_preflight = leverage_result.model_dump(mode="json")
            if not leverage_result.valid:
                return [
                    {
                        "mode": "live",
                        "symbol": self.symbol,
                        "action": "leverage_preflight",
                        "cloid": "leverage-preflight",
                        "oid": None,
                        "decision": "await_resolution",
                        "success": False,
                        "status": "rejected",
                        "message": leverage_result.reason,
                        "raw_response": leverage_preflight,
                    }
                ], leverage_preflight

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

    @staticmethod
    def _has_recent_fill_since(updated_at: datetime, fills: list[object] | None) -> bool:
        if not fills:
            return False
        fill_time = getattr(fills[-1], "time", None)
        return isinstance(fill_time, datetime) and fill_time > updated_at

    def _update_entry_rejection_block(
        self,
        *,
        strategy_state: StrategyState,
        position: PositionState,
        receipts: list[dict[str, object]],
    ) -> None:
        if position.side != TradeSide.FLAT:
            self._entry_rejection_block = None
            return
        for receipt in receipts:
            reason = self._sticky_entry_rejection_reason(receipt)
            if reason is None:
                continue
            self._entry_rejection_block = EntryRejectionBlock(
                strategy_updated_at=strategy_state.updated_at,
                reason=reason,
            )
            return

    @staticmethod
    def _sticky_entry_rejection_reason(receipt: dict[str, object]) -> str | None:
        action = str(receipt.get("action", ""))
        message = str(receipt.get("message", "")).strip()
        if not message:
            return None
        if action == "leverage_preflight":
            return message
        if not BotRuntime._receipt_is_error(receipt):
            return None
        cloid = receipt.get("cloid")
        role = extract_role_from_cloid(cloid if isinstance(cloid, str) else None)
        if role not in {OrderRole.ENTRY, OrderRole.UNKNOWN}:
            return None
        sticky_markers = (
            "Insufficient margin",
            "insufficient isolated margin",
            "target leverage exceeds venue max",
            "target leverage outside",
            "minimum order notional not met",
            "size rounds to zero",
            "price and size must be positive",
        )
        if any(marker in message for marker in sticky_markers):
            return message
        return None

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
        venue_max_leverage = (
            snapshot.active_asset_ctx.max_leverage
            if snapshot.active_asset_ctx is not None
            else None
        )
        resolved_max_leverage = self.max_leverage
        if isinstance(venue_max_leverage, (int, float)) and venue_max_leverage > 0:
            resolved_max_leverage = min(self.max_leverage, float(venue_max_leverage))
        base_account = AccountState(
            equity=equity,
            available_margin=available_margin,
            max_leverage=resolved_max_leverage,
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
        snapshot: LiveStateSnapshot,
        position: PositionState,
        plan: TradePlan,
        risk: RiskAssessment,
        kill_switch: KillSwitchStatus,
        receipts: list[dict[str, object]],
        meta: dict[str, object],
    ) -> None:
        first_cycle = not self._boot_logged
        lines: list[str] = []
        has_local_order_event = False

        if first_cycle:
            lines.append(
                self._event_line(
                    "BOOT",
                    (
                        f"mode={'live' if self.live else 'paper'} "
                        f"strategy={self.strategy_interval_s}s sync={self.sync_interval_s}s "
                        f"user={self._short_address(self.user_address)} "
                        f"private={meta.get('private_state_source', 'unknown')}"
                    ),
                )
            )
            self._seed_seen_events(snapshot)

        lines.extend(self._render_plan_events(plan=plan, first_cycle=first_cycle))
        lines.extend(
            self._render_reduce_only_events(
                kill_switch=kill_switch,
                first_cycle=first_cycle,
            )
        )
        lines.extend(
            self._render_entry_block_events(
                plan=plan,
                position=position,
                risk=risk,
                first_cycle=first_cycle,
            )
        )
        lines.extend(
            self._render_platform_error_events(
                kill_switch=kill_switch,
                meta=meta,
            )
        )
        receipt_lines, has_local_order_event = self._render_receipt_events(
            receipts=receipts,
            position=position,
        )
        lines.extend(receipt_lines)
        if not first_cycle:
            lines.extend(self._render_fill_events(snapshot=snapshot))
            lines.extend(self._render_user_events(snapshot=snapshot))
        lines.extend(
            self._render_external_order_events(
                position=position,
                first_cycle=first_cycle,
                has_local_order_event=has_local_order_event,
            )
        )
        lines.extend(
            self._render_position_events(
                position=position,
                first_cycle=first_cycle,
            )
        )

        if not lines:
            return

        self._print_event_block(
            cycle=cycle,
            snapshot=snapshot,
            position=position,
            lines=lines,
        )
        self._boot_logged = True

    def _render_plan_events(
        self,
        *,
        plan: TradePlan,
        first_cycle: bool,
    ) -> list[str]:
        signature = self._plan_signature(plan)
        should_log = first_cycle or signature != self._last_plan_signature
        self._last_plan_signature = signature
        if not should_log:
            return []

        summary, reason = self._describe_plan(plan)
        return [
            self._event_line("PLAN", summary),
            self._event_line("WHY", reason),
        ]

    def _render_reduce_only_events(
        self,
        *,
        kill_switch: KillSwitchStatus,
        first_cycle: bool,
    ) -> list[str]:
        current = tuple(kill_switch.reasons) if kill_switch.reduce_only else None
        previous = self._last_reduce_only_signature
        self._last_reduce_only_signature = current
        if first_cycle:
            if current is None:
                return []
            return [
                self._event_line(
                    "BLOCK",
                    f"reduce-only safeguards active | why=\"{kill_switch.reasons[0]}\"",
                )
            ]
        if previous == current:
            return []
        if current is None and previous is not None:
            return [self._event_line("RECOVERED", "reduce-only safeguards cleared")]
        if current is None:
            return []
        return [
            self._event_line(
                "BLOCK",
                f"reduce-only safeguards active | why=\"{kill_switch.reasons[0]}\"",
            )
        ]

    def _render_entry_block_events(
        self,
        *,
        plan: TradePlan,
        position: PositionState,
        risk: RiskAssessment,
        first_cycle: bool,
    ) -> list[str]:
        current = None
        if (
            position.side == TradeSide.FLAT
            and self._plan_is_actionable(plan)
            and not risk.allowed
        ):
            current = risk.reason
        previous = self._last_entry_block_reason
        self._last_entry_block_reason = current
        if first_cycle:
            if current is None:
                return []
            return [self._event_line("BLOCK", f"entry blocked | why=\"{current}\"")]
        if previous == current:
            return []
        if current is None and previous is not None:
            return [self._event_line("RECOVERED", "entry gate reopened")]
        if current is None:
            return []
        return [self._event_line("BLOCK", f"entry blocked | why=\"{current}\"")]

    def _render_platform_error_events(
        self,
        *,
        kill_switch: KillSwitchStatus,
        meta: dict[str, object],
    ) -> list[str]:
        lines: list[str] = []
        for reason in kill_switch.reasons:
            lines.append(self._event_line("ERROR", f"kill switch active | reason=\"{reason}\""))
        if meta.get("fills_safe_complete") is False:
            lines.append(
                self._event_line(
                    "ERROR",
                    "user fill pagination incomplete; reconciliation may miss older fills",
                )
            )
        private_bootstrap_error = meta.get("private_bootstrap_error")
        if isinstance(private_bootstrap_error, str) and private_bootstrap_error:
            lines.append(
                self._event_line(
                    "ERROR",
                    f"private bootstrap failed | reason=\"{private_bootstrap_error}\"",
                )
            )
        return lines

    def _render_receipt_events(
        self,
        *,
        receipts: list[dict[str, object]],
        position: PositionState,
    ) -> tuple[list[str], bool]:
        lines: list[str] = []
        has_local_order_event = False
        for receipt in receipts:
            rendered = self._render_receipt_event(receipt=receipt, position=position)
            if rendered is None:
                continue
            if rendered.startswith("ORDER"):
                has_local_order_event = True
            lines.append(rendered)
        return lines, has_local_order_event

    def _render_receipt_event(
        self,
        *,
        receipt: dict[str, object],
        position: PositionState,
    ) -> str | None:
        action = str(receipt.get("action", "unknown"))
        if action in {"keep", "paper_keep", "paper_hold"}:
            return None

        if action in {"paper_fill_entry", "paper_tp1", "paper_tp2", "paper_stop"}:
            label = action.replace("paper_", "").replace("_", " ")
            return self._event_line("FILL", label)
        if action == "paper_close":
            message = str(receipt.get("message", "")).strip()
            suffix = f" | why=\"{message}\"" if message else ""
            return self._event_line("POSITION", f"paper close{suffix}")

        role = self._receipt_role_label(receipt=receipt, position=position)
        oid = receipt.get("oid")
        status = str(receipt.get("status", "unknown"))
        message = str(receipt.get("message", "")).strip()
        detail_parts = [action]
        if role != "unknown":
            detail_parts.append(role)
        if oid is not None:
            detail_parts.append(f"oid={oid}")
        if status and status != "unknown":
            detail_parts.append(f"status={status}")

        if self._receipt_is_error(receipt):
            if message:
                detail_parts.append(f"reason=\"{message}\"")
            return self._event_line("ERROR", " | ".join([detail_parts[0], " ".join(detail_parts[1:])]).strip())

        return self._event_line("ORDER", " ".join(detail_parts))

    def _render_fill_events(self, *, snapshot: LiveStateSnapshot) -> list[str]:
        lines: list[str] = []
        for fill in snapshot.recent_fills:
            key = self._fill_key(fill)
            if key in self._seen_fill_keys:
                continue
            self._seen_fill_keys.add(key)
            lines.append(self._event_line("FILL", self._describe_fill(fill)))
        return lines

    def _render_user_events(self, *, snapshot: LiveStateSnapshot) -> list[str]:
        lines: list[str] = []
        for event in snapshot.recent_user_events:
            if event.coin not in {None, self.symbol}:
                continue
            key = self._user_event_key(event)
            if key in self._seen_user_event_keys:
                continue
            self._seen_user_event_keys.add(key)
            if event.event_type == UserEventType.OTHER:
                continue
            lines.append(self._event_line("ERROR", self._describe_user_event(event)))
        return lines

    def _render_external_order_events(
        self,
        *,
        position: PositionState,
        first_cycle: bool,
        has_local_order_event: bool,
    ) -> list[str]:
        current = self._active_order_signature(position)
        previous = self._last_active_order_signature
        self._last_active_order_signature = current
        if first_cycle or has_local_order_event or previous == current:
            return []
        return [
            self._event_line(
                "STATE",
                f"open-order set changed without local action | count={position.open_orders}",
            )
        ]

    def _render_position_events(
        self,
        *,
        position: PositionState,
        first_cycle: bool,
    ) -> list[str]:
        signature = self._position_signature(position)
        previous = self._last_position_signature
        self._last_position_signature = signature
        if first_cycle:
            if position.side == TradeSide.FLAT and position.open_orders == 0:
                return []
            return [self._event_line("POSITION", self._describe_position(position))]
        if previous == signature:
            return []
        return [self._event_line("POSITION", self._describe_position(position))]

    def _print_event_block(
        self,
        *,
        cycle: int,
        snapshot: LiveStateSnapshot,
        position: PositionState,
        lines: list[str],
    ) -> None:
        self._event_block_count += 1
        review_id = f"R{self._event_block_count:04d}"
        self.console.print(_LOG_DIVIDER)
        self.console.print(
            " | ".join(
                [
                    f"REVIEW {review_id}",
                    f"cycle={cycle}",
                    self.symbol,
                    self._fmt_time(snapshot.order_book.captured_at),
                    f"mid={self._fmt_price(snapshot.order_book.mid_price)}",
                    f"pos={position.side.value}",
                    f"oo={position.open_orders}",
                ]
            )
        )
        for line in lines:
            self.console.print(line)
        self.console.print(_LOG_DIVIDER)

    def _emit_runtime_error(
        self,
        *,
        phase: str,
        cycle: int | None,
        exc: Exception,
    ) -> None:
        self._event_block_count += 1
        review_id = f"R{self._event_block_count:04d}"
        self.console.print(_LOG_DIVIDER)
        header = [
            f"REVIEW {review_id}",
            self.symbol,
            self._fmt_time(datetime.now(tz=UTC)),
            f"phase={phase}",
        ]
        if cycle is not None:
            header.insert(1, f"cycle={cycle}")
        self.console.print(" | ".join(header))
        self.console.print(
            self._event_line(
                "ERROR",
                f"{type(exc).__name__}: {exc}",
            )
        )
        self.console.print(_LOG_DIVIDER)

    def _seed_seen_events(self, snapshot: LiveStateSnapshot) -> None:
        self._seen_fill_keys = {self._fill_key(fill) for fill in snapshot.recent_fills}
        self._seen_user_event_keys = {
            self._user_event_key(event) for event in snapshot.recent_user_events
        }

    @staticmethod
    def _event_line(label: str, message: str) -> str:
        return f"{label:<9}{message}"

    @staticmethod
    def _plan_is_actionable(plan: TradePlan) -> bool:
        if plan.resting_orders:
            return True
        return plan.playbook != Playbook.NO_TRADE and plan.side != TradeSide.FLAT

    def _describe_plan(self, plan: TradePlan) -> tuple[str, str]:
        if plan.resting_orders:
            order = plan.resting_orders[0]
            summary = (
                f"{plan.playbook.value} rest-{order.side.value} "
                f"entry={self._fmt_price(order.entry_band[0])}-{self._fmt_price(order.entry_band[1])} "
                f"sl={self._fmt_price(order.invalid_if)} "
                f"tp={self._fmt_price(order.tp1)}/{self._fmt_price(order.tp2)} "
                f"ttl={order.ttl_min}m conf={order.touch_confidence:.2f}"
            )
            reason = plan.reason
            if order.reason != plan.reason:
                reason = f"{plan.reason}; order={order.reason}"
            return summary, reason
        if plan.playbook == Playbook.NO_TRADE or plan.side == TradeSide.FLAT:
            return "no_trade", plan.reason
        summary = (
            f"{plan.playbook.value} {plan.side.value} "
            f"entry={self._fmt_price(plan.entry_band[0])}-{self._fmt_price(plan.entry_band[1])} "
            f"sl={self._fmt_price(plan.invalid_if)} "
            f"tp={self._fmt_price(plan.tp1)}/{self._fmt_price(plan.tp2)} "
            f"ttl={plan.ttl_min}m conf={plan.touch_confidence:.2f}"
        )
        return summary, plan.reason

    @staticmethod
    def _receipt_is_error(receipt: dict[str, object]) -> bool:
        success = bool(receipt.get("success", False))
        decision = str(receipt.get("decision", ""))
        status = str(receipt.get("status", ""))
        return (
            not success
            or decision == "await_resolution"
            or status == "rejected"
        )

    def _receipt_role_label(
        self,
        *,
        receipt: dict[str, object],
        position: PositionState,
    ) -> str:
        cloid = receipt.get("cloid")
        role = extract_role_from_cloid(cloid if isinstance(cloid, str) else None)
        if role == OrderRole.UNKNOWN:
            oid = receipt.get("oid")
            for order in position.active_orders:
                if isinstance(oid, int) and order.oid == oid:
                    role = order.role
                    break
                if isinstance(cloid, str) and order.cloid == cloid:
                    role = order.role
                    break
        return role.value

    def _describe_fill(self, fill: object) -> str:
        direction = str(getattr(fill, "direction", "fill")).strip() or "fill"
        price = self._fmt_price(getattr(fill, "price", None))
        size = self._fmt_size(getattr(fill, "size", None))
        parts = [direction, f"qty={size}", f"px={price}"]
        closed_pnl = getattr(fill, "closed_pnl", None)
        if isinstance(closed_pnl, (int, float)) and closed_pnl != 0:
            parts.append(f"pnl={closed_pnl:+.2f}")
        oid = getattr(fill, "oid", None)
        if oid is not None:
            parts.append(f"oid={oid}")
        return " ".join(parts)

    def _describe_user_event(self, event: object) -> str:
        if getattr(event, "event_type", UserEventType.OTHER) == UserEventType.LIQUIDATION:
            label = "liquidation reported"
        elif getattr(event, "event_type", UserEventType.OTHER) == UserEventType.NON_USER_CANCEL:
            label = "non-user cancel reported"
        else:
            label = "user event reported"
        coin = getattr(event, "coin", None)
        timestamp = getattr(event, "timestamp", None)
        parts = [label]
        if coin:
            parts.append(f"coin={coin}")
        if isinstance(timestamp, datetime):
            parts.append(f"at={self._fmt_time(timestamp)}")
        return " ".join(parts)

    def _describe_position(self, position: PositionState) -> str:
        if position.side == TradeSide.FLAT:
            return f"flat qty=0 open_orders={position.open_orders}"
        parts = [
            f"{position.side.value}",
            f"qty={self._fmt_size(position.quantity)}",
        ]
        if position.entry_price is not None:
            parts.append(f"entry={self._fmt_price(position.entry_price)}")
        if position.unrealized_pnl is not None:
            parts.append(f"uPnL={position.unrealized_pnl:+.2f}")
        if position.liquidation_price is not None:
            parts.append(f"liq={self._fmt_price(position.liquidation_price)}")
        parts.append(f"open_orders={position.open_orders}")
        return " ".join(parts)

    def _plan_signature(self, plan: TradePlan) -> tuple[object, ...]:
        resting_signature = tuple(
            (
                order.side.value,
                self._fmt_price(order.entry_band[0]),
                self._fmt_price(order.entry_band[1]),
                self._fmt_price(order.invalid_if),
                self._fmt_price(order.tp1),
                self._fmt_price(order.tp2),
                order.ttl_min,
                round(order.touch_confidence, 2),
                order.reason,
            )
            for order in plan.resting_orders
        )
        return (
            plan.playbook.value,
            plan.side.value,
            self._fmt_price(plan.entry_band[0]),
            self._fmt_price(plan.entry_band[1]),
            self._fmt_price(plan.invalid_if),
            self._fmt_price(plan.tp1),
            self._fmt_price(plan.tp2),
            plan.ttl_min,
            round(plan.touch_confidence, 2),
            plan.reason,
            resting_signature,
        )

    def _position_signature(self, position: PositionState) -> tuple[object, ...]:
        return (
            position.side.value,
            self._fmt_size(position.quantity),
            self._fmt_price(position.entry_price),
            self._fmt_price(position.liquidation_price),
        )

    def _active_order_signature(self, position: PositionState) -> tuple[tuple[object, ...], ...]:
        return tuple(
            sorted(
                (
                    order.cloid or f"oid:{order.oid}",
                    order.role.value,
                    order.side,
                    self._fmt_price(order.limit_price),
                    self._fmt_price(order.trigger_price),
                    self._fmt_size(order.size),
                    order.reduce_only,
                    order.status.value,
                )
                for order in position.active_orders
            )
        )

    @staticmethod
    def _fill_key(fill: object) -> str:
        fill_hash = getattr(fill, "fill_hash", None)
        if isinstance(fill_hash, str) and fill_hash:
            return fill_hash
        oid = getattr(fill, "oid", None)
        time_value = getattr(fill, "time", None)
        price = getattr(fill, "price", None)
        size = getattr(fill, "size", None)
        direction = getattr(fill, "direction", None)
        closed_pnl = getattr(fill, "closed_pnl", None)
        return "|".join(str(part) for part in (oid, time_value, price, size, direction, closed_pnl))

    @staticmethod
    def _user_event_key(event: object) -> str:
        return "|".join(
            str(part)
            for part in (
                getattr(event, "event_type", None),
                getattr(event, "coin", None),
                getattr(event, "timestamp", None),
                getattr(event, "oid", None),
                getattr(event, "cloid", None),
            )
        )

    @staticmethod
    def _fmt_time(value: datetime) -> str:
        return value.astimezone(UTC).strftime("%Y-%m-%d %H:%M:%SZ")

    @staticmethod
    def _fmt_price(value: float | None) -> str:
        if value is None:
            return "-"
        magnitude = abs(value)
        if magnitude >= 1_000:
            return f"{value:.1f}"
        if magnitude >= 100:
            return f"{value:.2f}"
        if magnitude >= 1:
            return f"{value:.3f}".rstrip("0").rstrip(".")
        return f"{value:.6f}".rstrip("0").rstrip(".")

    @staticmethod
    def _fmt_size(value: float | None) -> str:
        if value is None:
            return "-"
        return f"{value:.4f}".rstrip("0").rstrip(".")

    @staticmethod
    def _short_address(value: str) -> str:
        if len(value) <= 12:
            return value
        return f"{value[:6]}...{value[-4:]}"

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
    def _resting_order_expired(
        order: RestingOrderPlan,
        updated_at: datetime,
        now: datetime,
    ) -> bool:
        if order.expected_touch_minutes is None:
            return False
        return (now - updated_at).total_seconds() >= order.expected_touch_minutes * 60

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
