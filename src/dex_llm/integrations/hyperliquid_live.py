from __future__ import annotations

import threading
import time
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime

from hyperliquid.info import Info
from hyperliquid.utils.types import Cloid

from dex_llm.executor.safety import AssetMetadata, extract_role_from_cloid
from dex_llm.integrations.hyperliquid import HyperliquidInfoClient
from dex_llm.models import (
    BboSnapshot,
    Candle,
    HyperliquidActiveAssetContext,
    HyperliquidClearinghouseState,
    HyperliquidFrontendOrder,
    HyperliquidUserEvent,
    HyperliquidUserFill,
    LiveOrderState,
    LiveStateSnapshot,
    OrderBookSnapshot,
    OrderRole,
    OrderState,
    UserEventType,
)


class HyperliquidRestGateway:
    def __init__(
        self,
        *,
        base_url: str = "https://api.hyperliquid.xyz",
        timeout_s: float = 10.0,
    ) -> None:
        self.base_url = base_url
        self.timeout_s = timeout_s
        self.info = Info(base_url=base_url, skip_ws=True, timeout=timeout_s)
        self.legacy = HyperliquidInfoClient(base_url=base_url, timeout_s=timeout_s)

    def close(self) -> None:
        self.legacy.close()

    def fetch_asset_meta(self, symbol: str) -> AssetMetadata:
        payload = self.info.meta()
        universe = payload.get("universe", []) if isinstance(payload, dict) else []
        for index, item in enumerate(universe):
            if not isinstance(item, dict) or str(item.get("name")) != symbol:
                continue
            return AssetMetadata(
                symbol=symbol,
                asset_index=index,
                size_decimals=int(item.get("szDecimals", 0)),
                max_leverage=_coerce_float(item.get("maxLeverage")),
            )
        raise ValueError(f"Unknown Hyperliquid symbol: {symbol}")

    def fetch_active_asset_ctx(self, symbol: str) -> HyperliquidActiveAssetContext:
        payload = self.info.meta_and_asset_ctxs()
        if not isinstance(payload, list) or len(payload) != 2:
            raise ValueError("Unexpected metaAndAssetCtxs payload")
        meta_payload, contexts = payload
        universe = meta_payload.get("universe", []) if isinstance(meta_payload, dict) else []
        if not isinstance(contexts, list):
            raise ValueError("Unexpected asset context payload")
        for item, ctx in zip(universe, contexts, strict=False):
            if not isinstance(item, dict) or str(item.get("name")) != symbol:
                continue
            if not isinstance(ctx, dict):
                continue
            return HyperliquidActiveAssetContext(
                coin=symbol,
                mark_price=_coerce_float(ctx.get("markPx")),
                oracle_price=_coerce_float(ctx.get("oraclePx")),
                mid_price=_coerce_float(ctx.get("midPx")),
                max_leverage=_coerce_float(item.get("maxLeverage")),
                funding=_coerce_float(ctx.get("funding")),
                open_interest=_coerce_float(ctx.get("openInterest")),
                timestamp=_extract_datetime(ctx.get("time")),
            )
        raise ValueError(f"Missing asset context for {symbol}")

    def fetch_clearinghouse_state(
        self,
        *,
        user: str,
        dex: str = "",
    ) -> HyperliquidClearinghouseState:
        return self.legacy.fetch_clearinghouse_state(user=user, dex=dex)

    def fetch_frontend_open_orders(
        self,
        *,
        user: str,
        dex: str = "",
    ) -> list[HyperliquidFrontendOrder]:
        return self.legacy.fetch_frontend_open_orders(user=user, dex=dex)

    def fetch_open_orders(
        self,
        *,
        user: str,
        dex: str = "",
    ) -> list[LiveOrderState]:
        payload = self.info.open_orders(user, dex)
        return [parse_live_order(item) for item in _coerce_items(payload)]

    def fetch_l2_book(self, symbol: str) -> OrderBookSnapshot:
        return self.legacy.fetch_l2_book(symbol)

    def fetch_candles(self, symbol: str, interval: str, limit: int) -> list[Candle]:
        return self.legacy.fetch_candles(symbol, interval, limit)

    def fetch_bbo(self, symbol: str) -> BboSnapshot:
        book = self.fetch_l2_book(symbol)
        return BboSnapshot(
            symbol=symbol,
            captured_at=book.captured_at,
            bid=book.best_bid,
            ask=book.best_ask,
            mid=book.mid_price,
        )

    def query_order_by_cloid(self, user: str, cloid: str) -> object:
        return self.info.query_order_by_cloid(user, Cloid.from_str(cloid))

    def historical_orders(self, user: str) -> object:
        return self.info.historical_orders(user)

    def open_orders(self, user: str) -> object:
        return self.info.open_orders(user)

    def user_rate_limit(self, user: str) -> object:
        return self.info.user_rate_limit(user)

    def fetch_user_fills_by_time(
        self,
        *,
        user: str,
        start_time: int,
        end_time: int | None = None,
        aggregate_by_time: bool = True,
    ) -> list[HyperliquidUserFill]:
        payload = self.info.user_fills_by_time(
            user,
            start_time,
            end_time=end_time,
            aggregate_by_time=aggregate_by_time,
        )
        return HyperliquidInfoClient.parse_user_fills(payload)

    def paginate_user_fills_by_time(
        self,
        *,
        user: str,
        start_time: int,
        end_time: int,
        aggregate_by_time: bool = True,
        max_pages: int = 5,
    ) -> tuple[list[HyperliquidUserFill], bool]:
        dedupe: dict[str, HyperliquidUserFill] = {}
        cursor_end = end_time
        safe_complete = True
        for _ in range(max_pages):
            batch = self.fetch_user_fills_by_time(
                user=user,
                start_time=start_time,
                end_time=cursor_end,
                aggregate_by_time=aggregate_by_time,
            )
            for fill in batch:
                dedupe[fill_identity(fill)] = fill
            if len(batch) < 2000:
                break
            oldest_ms = min(int(fill.time.timestamp() * 1000) for fill in batch)
            if oldest_ms <= start_time:
                break
            cursor_end = oldest_ms - 1
        else:
            safe_complete = False
        ordered = sorted(dedupe.values(), key=lambda item: item.time)
        return ordered, safe_complete


class HyperliquidWsStateClient:
    def __init__(
        self,
        *,
        base_url: str = "https://api.hyperliquid.xyz",
        timeout_s: float = 10.0,
        user_address: str | None = None,
    ) -> None:
        self.info = Info(base_url=base_url, skip_ws=False, timeout=timeout_s)
        self.user_address = user_address
        self._symbol: str | None = None
        self._lock = threading.Lock()
        self._order_book: OrderBookSnapshot | None = None
        self._candles_5m: list[Candle] = []
        self._candles_15m: list[Candle] = []
        self._bbo: BboSnapshot | None = None
        self._active_asset_ctx: HyperliquidActiveAssetContext | None = None
        self._clearinghouse_state: HyperliquidClearinghouseState | None = None
        self._open_orders: dict[str, LiveOrderState] = {}
        self._recent_fills: list[HyperliquidUserFill] = []
        self._recent_user_events: list[HyperliquidUserEvent] = []
        self._channel_timestamps: dict[str, datetime] = {}
        self._channel_snapshot_flags: dict[str, bool] = {}
        self._subscription_ids: list[tuple[dict[str, object], int]] = []
        self._last_ping_at: datetime | None = None
        self._last_pong_at: datetime | None = None

    def close(self) -> None:
        for subscription, subscription_id in self._subscription_ids:
            self.info.unsubscribe(subscription, subscription_id)
        self._subscription_ids.clear()
        self.info.disconnect_websocket()

    def connect(self, symbol: str, *, user_address: str | None = None) -> None:
        self._symbol = symbol
        self.user_address = user_address or self.user_address
        subscriptions: list[dict[str, object]] = [
            {"type": "l2Book", "coin": symbol},
            {"type": "candle", "coin": symbol, "interval": "5m"},
            {"type": "candle", "coin": symbol, "interval": "15m"},
            {"type": "bbo", "coin": symbol},
            {"type": "activeAssetCtx", "coin": symbol},
        ]
        if self.user_address:
            subscriptions.extend(
                [
                    {"type": "webData3", "user": self.user_address},
                    {"type": "orderUpdates", "user": self.user_address},
                    {"type": "userEvents", "user": self.user_address},
                    {"type": "userFills", "user": self.user_address},
                ]
            )
        for subscription in subscriptions:
            subscription_id = self.info.subscribe(subscription, self.ingest_message)
            self._subscription_ids.append((subscription, subscription_id))

    def wait_until_ready(self, *, timeout_s: float = 5.0) -> None:
        deadline = time.monotonic() + timeout_s
        while time.monotonic() < deadline:
            if self._is_ready():
                return
            time.sleep(0.05)
        raise TimeoutError("Timed out waiting for Hyperliquid websocket snapshots")

    def send_heartbeat_if_idle(self, *, idle_s: float = 30.0) -> bool:
        manager = getattr(self.info, "ws_manager", None)
        if manager is None:
            return False
        last_activity = max(self._channel_timestamps.values(), default=datetime.now(tz=UTC))
        if (datetime.now(tz=UTC) - last_activity).total_seconds() < idle_s:
            return False
        manager.ws.send('{"method":"ping"}')
        self._last_ping_at = datetime.now(tz=UTC)
        return True

    def snapshot(self) -> LiveStateSnapshot:
        if self._symbol is None or self._order_book is None:
            raise RuntimeError("Websocket state has not been initialized")
        return LiveStateSnapshot(
            symbol=self._symbol,
            order_book=self._order_book,
            candles_5m=list(self._candles_5m),
            candles_15m=list(self._candles_15m),
            bbo=self._bbo,
            active_asset_ctx=self._active_asset_ctx,
            clearinghouse_state=self._clearinghouse_state,
            open_orders=list(self._open_orders.values()),
            recent_fills=list(self._recent_fills),
            recent_user_events=list(self._recent_user_events),
            channel_timestamps=dict(self._channel_timestamps),
            channel_snapshot_flags=dict(self._channel_snapshot_flags),
        )

    def ingest_message(self, ws_msg: Mapping[str, object]) -> None:
        channel = str(ws_msg.get("channel", ""))
        if channel == "pong":
            self._last_pong_at = datetime.now(tz=UTC)
            return
        data = ws_msg.get("data")
        is_snapshot = bool(ws_msg.get("isSnapshot", False))
        with self._lock:
            if channel == "l2Book":
                symbol = str((data or {}).get("coin", self._symbol or ""))
                self._order_book = HyperliquidInfoClient.parse_l2_book(symbol, data)
            elif channel == "candle":
                self._merge_candles(data, is_snapshot=is_snapshot)
            elif channel == "bbo":
                self._bbo = parse_bbo_message(data, fallback_symbol=self._symbol or "")
            elif channel == "activeAssetCtx":
                self._active_asset_ctx = parse_active_asset_ctx_message(
                    data,
                    fallback_symbol=self._symbol or "",
                )
            elif channel in {"webData2", "webData3"}:
                self._apply_web_data2(data)
            elif channel == "orderUpdates":
                self._apply_order_updates(data)
            elif channel == "userFills":
                self._apply_user_fills(data)
            elif channel in {"user", "userEvents"}:
                self._apply_user_events(data)
            self._channel_timestamps[channel] = datetime.now(tz=UTC)
            self._channel_snapshot_flags[channel] = is_snapshot

    def _is_ready(self) -> bool:
        public_ready = (
            self._order_book is not None
            and bool(self._candles_5m)
            and bool(self._candles_15m)
            and self._bbo is not None
            and self._active_asset_ctx is not None
        )
        if not public_ready:
            return False
        if self.user_address is None:
            return True
        return self._clearinghouse_state is not None

    def _merge_candles(self, data: object, *, is_snapshot: bool) -> None:
        if isinstance(data, list):
            candles = HyperliquidInfoClient.parse_candles(data)
            interval = _detect_interval(data)
        elif isinstance(data, dict):
            candles = HyperliquidInfoClient.parse_candles([data])
            interval = str(data.get("i", ""))
        else:
            return
        if interval == "5m":
            self._candles_5m = (
                candles
                if is_snapshot
                else _dedupe_candles(self._candles_5m + candles)
            )
        elif interval == "15m":
            self._candles_15m = (
                candles
                if is_snapshot
                else _dedupe_candles(self._candles_15m + candles)
            )

    def _apply_web_data2(self, data: object) -> None:
        if not isinstance(data, dict):
            return
        candidate = data.get("clearinghouseState") or data.get("marginState") or data
        if isinstance(candidate, dict) and "assetPositions" in candidate:
            self._clearinghouse_state = HyperliquidInfoClient.parse_clearinghouse_state(candidate)
        open_orders = data.get("openOrders")
        if isinstance(open_orders, list):
            self._open_orders = {
                order_key(order): order
                for order in (parse_live_order(item) for item in open_orders)
            }

    def _apply_order_updates(self, data: object) -> None:
        for item in _coerce_items(data):
            order = parse_live_order(item)
            key = order_key(order)
            if order.status in {OrderState.CANCELED, OrderState.FILLED, OrderState.REJECTED}:
                self._open_orders.pop(key, None)
                continue
            self._open_orders[key] = order

    def _apply_user_fills(self, data: object) -> None:
        payload = data.get("fills") if isinstance(data, dict) else data
        fills = HyperliquidInfoClient.parse_user_fills(payload)
        merged = {fill_identity(fill): fill for fill in self._recent_fills}
        for fill in fills:
            merged[fill_identity(fill)] = fill
        self._recent_fills = sorted(merged.values(), key=lambda item: item.time)[-500:]

    def _apply_user_events(self, data: object) -> None:
        events = _coerce_items(data)
        for item in events:
            self._recent_user_events.append(parse_user_event(item))
        self._recent_user_events = self._recent_user_events[-100:]


def parse_bbo_message(payload: object, *, fallback_symbol: str) -> BboSnapshot:
    if not isinstance(payload, dict):
        raise ValueError("Unexpected bbo payload")
    bbo = payload.get("bbo")
    if isinstance(bbo, list) and len(bbo) >= 2:
        bid, ask = bbo[0], bbo[1]
    else:
        bid = payload.get("bid") or payload.get("bidPx") or payload.get("b")
        ask = payload.get("ask") or payload.get("askPx") or payload.get("a")
    bid_px = _extract_price(bid)
    ask_px = _extract_price(ask)
    timestamp = _extract_datetime(payload.get("time")) or datetime.now(tz=UTC)
    symbol = str(payload.get("coin", fallback_symbol))
    return BboSnapshot(
        symbol=symbol,
        captured_at=timestamp,
        bid=bid_px,
        ask=ask_px,
        mid=(bid_px + ask_px) / 2,
    )


def parse_active_asset_ctx_message(
    payload: object,
    *,
    fallback_symbol: str,
) -> HyperliquidActiveAssetContext:
    if not isinstance(payload, dict):
        raise ValueError("Unexpected activeAssetCtx payload")
    ctx = payload.get("ctx") if isinstance(payload.get("ctx"), dict) else payload
    return HyperliquidActiveAssetContext(
        coin=str(payload.get("coin", fallback_symbol)),
        mark_price=_coerce_float(ctx.get("markPx")),
        oracle_price=_coerce_float(ctx.get("oraclePx")),
        mid_price=_coerce_float(ctx.get("midPx")),
        max_leverage=_coerce_float(ctx.get("maxLeverage")),
        funding=_coerce_float(ctx.get("funding")),
        open_interest=_coerce_float(ctx.get("openInterest")),
        timestamp=_extract_datetime(ctx.get("time")),
    )


def parse_live_order(payload: Mapping[str, object]) -> LiveOrderState:
    side = str(payload.get("side", ""))
    status_raw = str(payload.get("status", "open"))
    cloid = _coerce_optional_str(payload.get("cloid") or payload.get("clientOrderId"))
    role_hint = str(payload.get("role") or payload.get("clientTag") or cloid or "unknown")
    return LiveOrderState(
        coin=str(payload.get("coin", "")),
        side=side,
        limit_price=_coerce_float(payload.get("limitPx", payload.get("limit_price"))) or 0.0,
        size=_coerce_float(payload.get("sz", payload.get("origSz", payload.get("size")))) or 0.0,
        reduce_only=bool(payload.get("reduceOnly", False)),
        is_trigger=bool(payload.get("isTrigger", False)),
        order_type=str(payload.get("orderType", payload.get("order_type", "unknown"))),
        oid=int(payload.get("oid", 0)),
        cloid=cloid,
        status=_parse_order_state(status_raw),
        role=_parse_order_role(role_hint),
        timestamp=_extract_datetime(payload.get("timestamp")),
        trigger_price=_coerce_float(payload.get("triggerPx")),
    )


def parse_user_event(payload: Mapping[str, object]) -> HyperliquidUserEvent:
    event_type = UserEventType.OTHER
    if payload.get("liquidation") or str(payload.get("type", "")).lower() == "liquidation":
        event_type = UserEventType.LIQUIDATION
    elif payload.get("nonUserCancel") or (
        _coerce_optional_str(payload.get("reason")) == "nonUserCancel"
    ):
        event_type = UserEventType.NON_USER_CANCEL
    return HyperliquidUserEvent(
        event_type=event_type,
        coin=_coerce_optional_str(payload.get("coin")),
        timestamp=_extract_datetime(payload.get("time")),
        payload=dict(payload),
    )


def fill_identity(fill: HyperliquidUserFill) -> str:
    if fill.fill_hash:
        return fill.fill_hash
    oid = fill.oid if fill.oid is not None else 0
    return "|".join(
        [
            str(oid),
            fill.coin,
            str(int(fill.time.timestamp() * 1000)),
            str(fill.price),
            str(fill.size),
            str(fill.closed_pnl),
            fill.direction,
        ]
    )


def order_key(order: LiveOrderState) -> str:
    if order.cloid:
        return order.cloid
    return f"oid:{order.oid}"


def _parse_order_state(status: str) -> OrderState:
    lowered = status.lower()
    if "filled" in lowered:
        return OrderState.FILLED
    if "canceled" in lowered:
        return OrderState.CANCELED
    if "trigger" in lowered:
        return OrderState.TRIGGERED
    if "reject" in lowered:
        return OrderState.REJECTED
    if "scheduledcancel" in lowered:
        return OrderState.SCHEDULED_CANCEL
    if "open" in lowered or not lowered:
        return OrderState.OPEN
    return OrderState.UNKNOWN


def _parse_order_role(value: str) -> OrderRole:
    lowered = value.lower()
    role_from_cloid = extract_role_from_cloid(value)
    if role_from_cloid != OrderRole.UNKNOWN:
        return role_from_cloid
    if lowered in {OrderRole.ENTRY.value, "entry"}:
        return OrderRole.ENTRY
    if lowered in {OrderRole.TAKE_PROFIT_1.value, "take_profit_1"}:
        return OrderRole.TAKE_PROFIT_1
    if lowered in {OrderRole.TAKE_PROFIT_2.value, "take_profit_2"}:
        return OrderRole.TAKE_PROFIT_2
    if lowered in {OrderRole.STOP_LOSS.value, "stop_loss"}:
        return OrderRole.STOP_LOSS
    return OrderRole.UNKNOWN


def _detect_interval(payload: Sequence[object]) -> str:
    for item in payload:
        if isinstance(item, dict) and isinstance(item.get("i"), str):
            return str(item["i"])
    return ""


def _dedupe_candles(candles: Sequence[Candle]) -> list[Candle]:
    dedupe = {candle.ts: candle for candle in candles}
    return [dedupe[key] for key in sorted(dedupe)]


def _extract_price(payload: object) -> float:
    if isinstance(payload, dict):
        for key in ("px", "price"):
            value = payload.get(key)
            if value is not None:
                return float(value)
    if isinstance(payload, (int, float, str)):
        return float(payload)
    raise ValueError("Missing price")


def _extract_datetime(payload: object) -> datetime | None:
    if isinstance(payload, (int, float)):
        seconds = payload / 1000 if payload > 10_000_000_000 else payload
        return datetime.fromtimestamp(seconds, tz=UTC)
    return None


def _coerce_float(value: object) -> float | None:
    if value is None or value == "":
        return None
    if isinstance(value, bool):
        return float(value)
    if isinstance(value, (int, float, str)):
        return float(value)
    return None


def _coerce_optional_str(value: object) -> str | None:
    if value is None:
        return None
    return str(value)


def _coerce_int(value: object) -> int | None:
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, (int, float, str)):
        return int(value)
    return None


def _coerce_items(payload: object) -> list[dict[str, object]]:
    if isinstance(payload, list):
        return [item for item in payload if isinstance(item, dict)]
    if isinstance(payload, dict):
        for key in ("orders", "fills", "events", "data"):
            value = payload.get(key)
            if isinstance(value, list):
                return [item for item in value if isinstance(item, dict)]
        return [payload]
    return []


HyperliquidRestBootstrapClient = HyperliquidRestGateway
