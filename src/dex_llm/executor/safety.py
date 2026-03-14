from __future__ import annotations

import hashlib
import math
import time
from collections.abc import Callable, Iterable, Mapping
from datetime import UTC, datetime
from typing import Any

from pydantic import BaseModel, Field

from dex_llm.models import (
    ExecutionMode,
    LiveOrderState,
    MarginMode,
    OrderRole,
    OrderState,
    PendingActionState,
    ReconciliationDecision,
    TradeSide,
)


class AssetMetadata(BaseModel):
    symbol: str
    asset_index: int
    size_decimals: int
    max_leverage: float | None = None


class BudgetStatus(BaseModel):
    degrade: bool = False
    reasons: list[str] = Field(default_factory=list)
    suspend_llm: bool = False
    reduce_only_only: bool = False
    degrade_mode: str = "off"


class ValidationResult(BaseModel):
    valid: bool
    reason: str
    price: float = 0.0
    size: float = 0.0
    notional: float = 0.0
    normalized_intent: Any | None = None


class ResolutionOutcome(BaseModel):
    decision: ReconciliationDecision
    status: OrderState = OrderState.UNKNOWN
    oid: int | None = None
    message: str = ""
    raw_response: dict[str, object] = Field(default_factory=dict)


def build_deterministic_cloid(
    strategy_id: str,
    symbol: str,
    frame_ts: datetime,
    role: OrderRole,
    revision: int,
) -> str:
    role_prefix = {
        OrderRole.ENTRY: "11",
        OrderRole.TAKE_PROFIT_1: "21",
        OrderRole.TAKE_PROFIT_2: "22",
        OrderRole.STOP_LOSS: "31",
        OrderRole.UNKNOWN: "ff",
    }[role]
    revision_prefix = f"{max(0, min(revision, 255)):02x}"
    payload = f"{strategy_id}:{symbol}:{frame_ts.isoformat()}:{role.value}:{revision}"
    digest = hashlib.blake2b(payload.encode("utf-8"), digest_size=14).hexdigest()
    return f"0x{role_prefix}{revision_prefix}{digest}"


def extract_role_from_cloid(cloid: str | None) -> OrderRole:
    if not cloid or not cloid.startswith("0x") or len(cloid) < 6:
        return OrderRole.UNKNOWN
    prefix = cloid[2:4].lower()
    return {
        "11": OrderRole.ENTRY,
        "21": OrderRole.TAKE_PROFIT_1,
        "22": OrderRole.TAKE_PROFIT_2,
        "31": OrderRole.STOP_LOSS,
    }.get(prefix, OrderRole.UNKNOWN)


def canonical_fill_key(fill: Mapping[str, object]) -> str:
    fill_hash = fill.get("hash") or fill.get("fill_hash")
    if isinstance(fill_hash, str) and fill_hash:
        return fill_hash
    parts = (
        fill.get("oid"),
        fill.get("coin"),
        fill.get("time"),
        fill.get("px") or fill.get("price"),
        fill.get("sz") or fill.get("size"),
        fill.get("closedPnl") or fill.get("closed_pnl"),
        fill.get("dir") or fill.get("direction"),
    )
    return "|".join("" if part is None else str(part) for part in parts)


def dedupe_fills(fills: Iterable[Mapping[str, object]]) -> list[Mapping[str, object]]:
    seen: set[str] = set()
    deduped: list[Mapping[str, object]] = []
    for fill in fills:
        key = canonical_fill_key(fill)
        if key in seen:
            continue
        seen.add(key)
        deduped.append(fill)
    return deduped


def has_active_entry_orders(orders: Iterable[LiveOrderState], symbol: str) -> bool:
    return any(
        order.coin == symbol and not order.reduce_only and order.status == OrderState.OPEN
        for order in orders
    )


class RateLimitBudgeter:
    def __init__(
        self,
        *,
        soft_open_order_limit: int = 900,
        open_order_soft_limit: int | None = None,
        address_utilization_limit: float = 0.8,
    ) -> None:
        self.soft_open_order_limit = open_order_soft_limit or soft_open_order_limit
        self.address_utilization_limit = address_utilization_limit
        self.rest_weight_used = 0
        self.ws_messages = 0
        self.ws_inflight_posts = 0
        self.address_limit: int | None = None
        self.address_used: int | None = None
        self._last_status = BudgetStatus()

    def note_rest_weight(self, weight: int = 1) -> None:
        self.rest_weight_used += weight

    def note_ws_message(self, count: int = 1) -> None:
        self.ws_messages += count

    def set_inflight_posts(self, count: int) -> None:
        self.ws_inflight_posts = max(0, count)

    def update_address_budget(self, *, used: int, limit: int) -> None:
        self.address_used = used
        self.address_limit = limit

    def evaluate(self, *, open_order_count: int) -> BudgetStatus:
        reasons: list[str] = []
        suspend_llm = False
        reduce_only_only = False
        if open_order_count >= self.soft_open_order_limit:
            reasons.append(f"soft open-order limit reached ({open_order_count})")
            suspend_llm = True
            reduce_only_only = True
        if self.address_used is not None and self.address_limit:
            utilization = self.address_used / self.address_limit
            if utilization >= self.address_utilization_limit:
                reasons.append(
                    f"address action utilization too high ({utilization * 100:.0f}%)"
                )
                suspend_llm = True
                reduce_only_only = True
        status = BudgetStatus(
            degrade=bool(reasons),
            reasons=reasons,
            suspend_llm=suspend_llm,
            reduce_only_only=reduce_only_only,
            degrade_mode="soft" if reasons else "off",
        )
        self._last_status = status
        return status

    def sync(
        self,
        user: str,
        snapshot: Mapping[str, object],
        *,
        open_orders: int,
    ) -> BudgetStatus:
        used = int(snapshot.get("nRequestsUsed", 0))
        limit = int(snapshot.get("nRequestsCap", 0))
        if limit > 0:
            self.update_address_budget(used=used, limit=limit)
        return self.evaluate(open_order_count=open_orders)

    def should_degrade(self, *, required_actions: int = 1) -> bool:
        if self._last_status.degrade:
            return True
        if self.address_limit is None or self.address_used is None:
            return False
        return (self.address_used + required_actions) >= self.address_limit


class PreSubmitValidator:
    def __init__(
        self,
        asset_metadata: Mapping[str, AssetMetadata] | None = None,
        *,
        min_notional: float = 10.0,
        leverage_buffer_fraction: float = 0.0,
    ) -> None:
        self.asset_metadata = asset_metadata or {}
        self.min_notional = min_notional
        self.leverage_buffer_fraction = leverage_buffer_fraction

    def validate(
        self,
        intent: Any,
        *,
        asset_meta: AssetMetadata,
        bbo: Any,
        active_asset_ctx: Any,
        position: Any,
        margin_mode: MarginMode | None = None,
        target_leverage: int | None = None,
        stop_reference_price: float | None = None,
    ) -> ValidationResult:
        side = TradeSide.LONG if getattr(intent, "is_buy", False) else TradeSide.SHORT
        best_bid = getattr(bbo, "bid", None)
        best_ask = getattr(bbo, "ask", None)
        oracle_price = (
            getattr(active_asset_ctx, "oracle_price", None)
            if active_asset_ctx is not None
            else None
        )
        current_position_size = getattr(position, "quantity", 0.0) or 0.0
        if getattr(position, "side", None) == TradeSide.SHORT:
            current_position_size = -abs(current_position_size)
        result = self.validate_order(
            symbol=intent.symbol,
            side=side,
            price=intent.limit_price,
            size=intent.size,
            reduce_only=getattr(intent, "reduce_only", False),
            current_position_size=current_position_size,
            best_bid=best_bid,
            best_ask=best_ask,
            oracle_price=oracle_price,
            asset_meta=asset_meta,
            margin_mode=margin_mode,
            target_leverage=target_leverage,
            stop_reference_price=stop_reference_price,
        )
        if not result.valid:
            return result
        return result.model_copy(
            update={
                "normalized_intent": intent.model_copy(
                    update={
                        "limit_price": result.price,
                        "size": result.size,
                    }
                )
            }
        )

    def validate_order(
        self,
        *,
        symbol: str,
        side: TradeSide,
        price: float,
        size: float,
        reduce_only: bool,
        current_position_size: float = 0.0,
        best_bid: float | None = None,
        best_ask: float | None = None,
        oracle_price: float | None = None,
        asset_meta: AssetMetadata | None = None,
        margin_mode: MarginMode | None = None,
        target_leverage: int | None = None,
        stop_reference_price: float | None = None,
        current_liquidation_price: float | None = None,
    ) -> ValidationResult:
        _ = margin_mode, target_leverage, stop_reference_price, current_liquidation_price
        meta = asset_meta or self.asset_metadata.get(symbol)
        if meta is None:
            return ValidationResult(valid=False, reason=f"missing asset metadata for {symbol}")
        if price <= 0 or size <= 0:
            return ValidationResult(valid=False, reason="price and size must be positive")

        quantized_price = self.quantize_price(symbol, price, asset_meta=meta)
        quantized_size = self.quantize_size(symbol, size, asset_meta=meta)
        if quantized_size <= 0:
            return ValidationResult(valid=False, reason="size rounds to zero")

        notional = quantized_price * quantized_size
        if notional < self.min_notional:
            return ValidationResult(valid=False, reason="minimum order notional not met")

        if reduce_only:
            if current_position_size == 0:
                return ValidationResult(valid=False, reason="reduce-only order has no position")
            if current_position_size > 0 and side != TradeSide.SHORT:
                return ValidationResult(valid=False, reason="reduce-only long position must sell")
            if current_position_size < 0 and side != TradeSide.LONG:
                return ValidationResult(valid=False, reason="reduce-only short position must buy")

        return ValidationResult(
            valid=True,
            reason="ok",
            price=quantized_price,
            size=quantized_size,
            notional=notional,
        )

    def validate_leverage_preflight(
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
        if target_leverage < 10 or target_leverage > 20:
            return ValidationResult(valid=False, reason="target leverage outside 10x-20x range")
        if target_leverage > max_leverage:
            return ValidationResult(valid=False, reason="target leverage exceeds venue max")
        if margin_mode not in {MarginMode.CROSS, MarginMode.ISOLATED}:
            return ValidationResult(valid=False, reason="unsupported margin mode")
        required_margin = recommended_notional / target_leverage
        buffered_margin = required_margin * (1 + self.leverage_buffer_fraction)
        if margin_mode == MarginMode.ISOLATED and available_margin < buffered_margin:
            return ValidationResult(valid=False, reason="insufficient isolated margin")
        if current_leverage is not None and math.isclose(
            current_leverage,
            target_leverage,
            rel_tol=0.0,
        ):
            return ValidationResult(valid=True, reason="already aligned")
        return ValidationResult(valid=True, reason="preflight ok")

    def quantize_price(
        self,
        symbol: str,
        price: float,
        *,
        asset_meta: AssetMetadata | None = None,
    ) -> float:
        meta = asset_meta or self.asset_metadata[symbol]
        decimals = max(0, 6 - meta.size_decimals)
        return round(float(f"{price:.5g}"), decimals)

    def quantize_size(
        self,
        symbol: str,
        size: float,
        *,
        asset_meta: AssetMetadata | None = None,
    ) -> float:
        meta = asset_meta or self.asset_metadata[symbol]
        return round(size, meta.size_decimals)

    @staticmethod
    def trim_trailing_zeros(value: float) -> str:
        text = f"{value:.12f}".rstrip("0").rstrip(".")
        return text if text else "0"

    @staticmethod
    def _mid(best_bid: float | None, best_ask: float | None) -> float | None:
        if best_bid is None or best_ask is None:
            return None
        return (best_bid + best_ask) / 2


class AmbiguousStateResolver:
    def __init__(
        self,
        lookup_client: object | None = None,
        *,
        query_order_by_cloid: Callable[[str], object] | None = None,
        fetch_open_orders: Callable[[], list[Mapping[str, object]]] | None = None,
        fetch_historical_orders: Callable[[], list[Mapping[str, object]]] | None = None,
        sleep_fn: Callable[[float], None] | None = None,
        sleeper: Callable[[float], None] | None = None,
        max_attempts: int = 3,
        max_wait_s: float = 3.0,
    ) -> None:
        if lookup_client is not None:
            self._lookup_client = lookup_client
            self.query_order_by_cloid = lambda cloid: lookup_client.query_order_by_cloid("", cloid)
            self.fetch_open_orders = lambda: lookup_client.open_orders("")
            self.fetch_historical_orders = lambda: lookup_client.historical_orders("")
        else:
            self._lookup_client = None
            if (
                query_order_by_cloid is None
                or fetch_open_orders is None
                or fetch_historical_orders is None
            ):
                raise ValueError("lookup_client or explicit lookup callables are required")
            self.query_order_by_cloid = query_order_by_cloid
            self.fetch_open_orders = fetch_open_orders
            self.fetch_historical_orders = fetch_historical_orders
        self.sleep_fn = sleeper or sleep_fn or time.sleep
        self.max_attempts = max_attempts
        self.max_wait_s = max_wait_s

    def resolve(self, pending: PendingActionState, user: str | None = None) -> ResolutionOutcome:
        started_at = time.monotonic()
        attempts = 0
        while attempts < self.max_attempts and (time.monotonic() - started_at) <= self.max_wait_s:
            attempts += 1
            if self._lookup_client is not None:
                status_payload = self._lookup_client.query_order_by_cloid(user or "", pending.cloid)
                open_orders = self._lookup_client.open_orders(user or "")
                historical_orders = self._lookup_client.historical_orders(user or "")
            else:
                status_payload = self.query_order_by_cloid(pending.cloid)
                open_orders = self.fetch_open_orders()
                historical_orders = self.fetch_historical_orders()
            status = self._extract_status(status_payload)
            if status is not None:
                return status

            open_status = self._search_orders(open_orders, pending.cloid)
            if open_status is not None:
                return open_status

            historical_status = self._search_orders(historical_orders, pending.cloid)
            if historical_status is not None:
                return historical_status

            if attempts < self.max_attempts:
                self.sleep_fn(self.max_wait_s / self.max_attempts)

        return ResolutionOutcome(
            decision=ReconciliationDecision.AWAIT_RESOLUTION,
            status=OrderState.UNKNOWN,
            message="order state remained ambiguous after bounded retries",
        )

    def _extract_status(self, payload: object) -> ResolutionOutcome | None:
        if not isinstance(payload, Mapping):
            return None
        status = self._normalize_status(payload)
        if status is None:
            return None
        decision = (
            ReconciliationDecision.KEEP
            if status in {OrderState.OPEN, OrderState.FILLED, OrderState.TRIGGERED}
            else ReconciliationDecision.CANCEL
        )
        return ResolutionOutcome(
            decision=decision,
            status=status,
            oid=self._extract_oid(payload),
            raw_response=dict(payload),
        )

    def _search_orders(
        self,
        orders: Iterable[Mapping[str, object]],
        cloid: str,
    ) -> ResolutionOutcome | None:
        for order in orders:
            raw_cloid = order.get("cloid") or order.get("oid")
            if raw_cloid is not None and str(raw_cloid) != cloid:
                continue
            status = self._normalize_status(order)
            if status is None:
                status = OrderState.OPEN
            decision = (
                ReconciliationDecision.KEEP
                if status in {OrderState.OPEN, OrderState.FILLED, OrderState.TRIGGERED}
                else ReconciliationDecision.CANCEL
            )
            return ResolutionOutcome(
                decision=decision,
                status=status,
                oid=self._extract_oid(order),
                raw_response=dict(order),
            )
        return None

    @staticmethod
    def _extract_oid(payload: Mapping[str, object]) -> int | None:
        oid = payload.get("oid")
        if isinstance(oid, int):
            return oid
        if isinstance(oid, str) and oid.isdigit():
            return int(oid)
        status_payload = payload.get("status")
        if isinstance(status_payload, Mapping):
            return AmbiguousStateResolver._extract_oid(status_payload)
        order_payload = payload.get("order")
        if isinstance(order_payload, Mapping):
            return AmbiguousStateResolver._extract_oid(order_payload)
        return None

    @staticmethod
    def _normalize_status(payload: Mapping[str, object]) -> OrderState | None:
        raw_status = payload.get("status")
        if isinstance(raw_status, Mapping):
            return AmbiguousStateResolver._normalize_status(raw_status)
        if isinstance(raw_status, str):
            return _status_from_string(raw_status)
        order_status = payload.get("orderStatus")
        if isinstance(order_status, str):
            return _status_from_string(order_status)
        if "filled" in payload:
            return OrderState.FILLED if payload.get("filled") else OrderState.OPEN
        return None


def _status_from_string(value: str) -> OrderState:
    lowered = value.lower()
    if lowered == "scheduledcancel":
        return OrderState.SCHEDULED_CANCEL
    for state in OrderState:
        if state.value.lower() == lowered:
            return state
    return OrderState.UNKNOWN


def utc_now() -> datetime:
    return datetime.now(tz=UTC)


def execution_mode_allows_live(mode: ExecutionMode) -> bool:
    return mode == ExecutionMode.LIVE
