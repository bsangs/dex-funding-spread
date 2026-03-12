from __future__ import annotations

import time
from collections.abc import Mapping
from datetime import UTC, datetime

import httpx
from hyperliquid.info import Info
from hyperliquid.utils.types import Cloid

from dex_llm.models import (
    Candle,
    HyperliquidClearinghouseState,
    HyperliquidFrontendOrder,
    HyperliquidMarginSummary,
    HyperliquidPerpPosition,
    HyperliquidUserFill,
    OrderBookSnapshot,
    PriceLevel,
)


class HyperliquidInfoClient:
    def __init__(
        self,
        base_url: str = "https://api.hyperliquid.xyz",
        timeout_s: float = 10.0,
        transport: httpx.BaseTransport | None = None,
    ) -> None:
        self._client = httpx.Client(
            base_url=base_url,
            timeout=timeout_s,
            headers={"Content-Type": "application/json"},
            transport=transport,
        )
        self._sdk_info = (
            None
            if transport is not None
            else Info(base_url=base_url, skip_ws=True, timeout=timeout_s)
        )

    def close(self) -> None:
        self._client.close()

    def fetch_l2_book(self, coin: str) -> OrderBookSnapshot:
        payload: dict[str, object] = {"type": "l2Book", "coin": coin}
        return self.parse_l2_book(coin, self._post_info(payload))

    def fetch_candles(self, coin: str, interval: str, limit: int) -> list[Candle]:
        interval_ms = self._interval_to_ms(interval)
        now_ms = int(time.time() * 1000)
        start_ms = now_ms - limit * interval_ms
        payload: dict[str, object] = {
            "type": "candleSnapshot",
            "req": {
                "coin": coin,
                "interval": interval,
                "startTime": start_ms,
                "endTime": now_ms,
            },
        }
        return self.parse_candles(self._post_info(payload))

    def fetch_clearinghouse_state(
        self,
        user: str,
        dex: str = "",
    ) -> HyperliquidClearinghouseState:
        payload: dict[str, object] = {"type": "clearinghouseState", "user": user}
        if dex:
            payload["dex"] = dex
        return self.parse_clearinghouse_state(self._post_info(payload))

    def fetch_frontend_open_orders(
        self,
        user: str,
        dex: str = "",
    ) -> list[HyperliquidFrontendOrder]:
        payload: dict[str, object] = {"type": "frontendOpenOrders", "user": user}
        if dex:
            payload["dex"] = dex
        return self.parse_frontend_open_orders(self._post_info(payload))

    def fetch_user_fills_by_time(
        self,
        user: str,
        start_time: int,
        end_time: int | None = None,
        aggregate_by_time: bool = False,
    ) -> list[HyperliquidUserFill]:
        payload: dict[str, object] = {
            "type": "userFillsByTime",
            "user": user,
            "startTime": start_time,
            "aggregateByTime": aggregate_by_time,
        }
        if end_time is not None:
            payload["endTime"] = end_time
        return self.parse_user_fills(self._post_info(payload))

    def fetch_historical_orders(self, user: str) -> object:
        if self._sdk_info is None:
            raise RuntimeError("historicalOrders requires SDK-backed HyperliquidInfoClient")
        return self._sdk_info.historical_orders(user)

    def fetch_open_orders(self, user: str, dex: str = "") -> object:
        if self._sdk_info is None:
            raise RuntimeError("openOrders requires SDK-backed HyperliquidInfoClient")
        return self._sdk_info.open_orders(user, dex)

    def query_order_by_cloid(self, user: str, cloid: str) -> object:
        if self._sdk_info is None:
            raise RuntimeError("orderStatus requires SDK-backed HyperliquidInfoClient")
        return self._sdk_info.query_order_by_cloid(user, Cloid.from_str(cloid))

    def fetch_user_rate_limit(self, user: str) -> object:
        if self._sdk_info is None:
            raise RuntimeError("userRateLimit requires SDK-backed HyperliquidInfoClient")
        return self._sdk_info.user_rate_limit(user)

    @staticmethod
    def parse_l2_book(coin: str, payload: object) -> OrderBookSnapshot:
        if not isinstance(payload, dict):
            raise ValueError("Unexpected Hyperliquid l2Book payload")

        levels = payload.get("levels")
        if not isinstance(levels, list) or len(levels) != 2:
            raise ValueError("Missing Hyperliquid levels")

        bids = [HyperliquidInfoClient._parse_price_level(level) for level in levels[0]]
        asks = [HyperliquidInfoClient._parse_price_level(level) for level in levels[1]]
        if not bids or not asks:
            raise ValueError("Empty Hyperliquid order book")

        best_bid = bids[0].price
        best_ask = asks[0].price
        return OrderBookSnapshot(
            symbol=coin,
            captured_at=datetime.fromtimestamp(float(payload["time"]) / 1000, tz=UTC),
            best_bid=best_bid,
            best_ask=best_ask,
            mid_price=(best_bid + best_ask) / 2,
            bids=bids,
            asks=asks,
        )

    @staticmethod
    def parse_candles(payload: object) -> list[Candle]:
        if not isinstance(payload, list):
            raise ValueError("Unexpected Hyperliquid candle payload")
        candles: list[Candle] = []
        for item in payload:
            if not isinstance(item, dict):
                continue
            candles.append(
                Candle(
                    ts=datetime.fromtimestamp(float(item["t"]) / 1000, tz=UTC),
                    open=float(item["o"]),
                    high=float(item["h"]),
                    low=float(item["l"]),
                    close=float(item["c"]),
                    volume=float(item["v"]),
                )
            )
        return candles

    @staticmethod
    def parse_clearinghouse_state(payload: object) -> HyperliquidClearinghouseState:
        if not isinstance(payload, dict):
            raise ValueError("Unexpected Hyperliquid clearinghouseState payload")

        asset_positions_payload = payload.get("assetPositions", [])
        if not isinstance(asset_positions_payload, list):
            raise ValueError("Unexpected Hyperliquid assetPositions payload")

        asset_positions: list[HyperliquidPerpPosition] = []
        for item in asset_positions_payload:
            if not isinstance(item, dict):
                continue
            position_payload = item.get("position")
            if not isinstance(position_payload, dict):
                continue
            leverage_payload = position_payload.get("leverage", {})
            leverage_type = "cross"
            leverage_value = 0.0
            raw_usd: float | None = None
            if isinstance(leverage_payload, dict):
                leverage_type = str(leverage_payload.get("type", leverage_type))
                leverage_value = _coerce_float(leverage_payload.get("value", 0.0))
                if leverage_payload.get("rawUsd") is not None:
                    raw_usd = _coerce_float(leverage_payload["rawUsd"])

            asset_positions.append(
                HyperliquidPerpPosition(
                    coin=str(position_payload["coin"]),
                    entry_price=_optional_float(position_payload.get("entryPx")),
                    liquidation_price=_optional_float(position_payload.get("liquidationPx")),
                    leverage_type=leverage_type,
                    leverage_value=leverage_value,
                    raw_usd=raw_usd,
                    margin_used=_coerce_float(position_payload.get("marginUsed", 0.0)),
                    max_leverage=_coerce_float(position_payload.get("maxLeverage", 0.0)),
                    position_value=_coerce_float(position_payload.get("positionValue", 0.0)),
                    size=_coerce_float(position_payload.get("szi", 0.0)),
                    unrealized_pnl=_coerce_float(position_payload.get("unrealizedPnl", 0.0)),
                )
            )

        return HyperliquidClearinghouseState(
            asset_positions=asset_positions,
            cross_maintenance_margin_used=_coerce_float(
                payload.get("crossMaintenanceMarginUsed", 0.0)
            ),
            cross_margin_summary=HyperliquidInfoClient._parse_margin_summary(
                payload.get("crossMarginSummary")
            ),
            margin_summary=HyperliquidInfoClient._parse_margin_summary(payload.get("marginSummary")),
            withdrawable=_coerce_float(payload.get("withdrawable", 0.0)),
            time=datetime.fromtimestamp(_coerce_float(payload["time"]) / 1000, tz=UTC),
        )

    @staticmethod
    def parse_frontend_open_orders(payload: object) -> list[HyperliquidFrontendOrder]:
        if not isinstance(payload, list):
            raise ValueError("Unexpected Hyperliquid frontendOpenOrders payload")

        orders: list[HyperliquidFrontendOrder] = []
        for item in payload:
            if not isinstance(item, dict):
                continue
            orders.append(
                HyperliquidFrontendOrder(
                    coin=str(item["coin"]),
                    side=str(item.get("side", "")),
                    limit_price=_coerce_float(item.get("limitPx", 0.0)),
                    size=_coerce_float(item.get("sz", item.get("origSz", 0.0))),
                    reduce_only=bool(item.get("reduceOnly", False)),
                    is_trigger=bool(item.get("isTrigger", False)),
                    order_type=str(item.get("orderType", "unknown")),
                    oid=int(item.get("oid", 0)),
                    cloid=_optional_str(item.get("cloid") or item.get("clientOrderId")),
                    trigger_price=_optional_float(item.get("triggerPx")),
                    timestamp=datetime.fromtimestamp(
                        _coerce_float(item["timestamp"]) / 1000,
                        tz=UTC,
                    ),
                )
            )
        return orders

    @staticmethod
    def parse_user_fills(payload: object) -> list[HyperliquidUserFill]:
        if not isinstance(payload, list):
            raise ValueError("Unexpected Hyperliquid userFills payload")

        fills: list[HyperliquidUserFill] = []
        for item in payload:
            if not isinstance(item, dict):
                continue
            fills.append(
                HyperliquidUserFill(
                    coin=str(item["coin"]),
                    closed_pnl=_coerce_float(item.get("closedPnl", 0.0)),
                    direction=str(item.get("dir", "")),
                    price=_coerce_float(item.get("px", 0.0)),
                    size=_coerce_float(item.get("sz", 0.0)),
                    oid=_optional_int(item.get("oid")),
                    fill_hash=_optional_str(item.get("hash")),
                    side=_optional_str(item.get("side")),
                    crossed=bool(item.get("crossed", False)),
                    start_position=_optional_float(item.get("startPosition")),
                    time=datetime.fromtimestamp(_coerce_float(item["time"]) / 1000, tz=UTC),
                )
            )
        return fills

    @staticmethod
    def _parse_price_level(payload: object) -> PriceLevel:
        if not isinstance(payload, dict):
            raise ValueError("Unexpected price level payload")
        return PriceLevel(
            price=_coerce_float(payload["px"]),
            size=_coerce_float(payload["sz"]),
            orders=int(payload["n"]),
        )

    @staticmethod
    def _interval_to_ms(interval: str) -> int:
        mapping = {
            "1m": 60_000,
            "3m": 180_000,
            "5m": 300_000,
            "15m": 900_000,
            "30m": 1_800_000,
            "1h": 3_600_000,
            "2h": 7_200_000,
            "4h": 14_400_000,
            "8h": 28_800_000,
            "12h": 43_200_000,
            "1d": 86_400_000,
            "3d": 259_200_000,
            "1w": 604_800_000,
            "1M": 2_592_000_000,
        }
        if interval not in mapping:
            raise ValueError(f"Unsupported interval: {interval}")
        return mapping[interval]

    @staticmethod
    def _parse_margin_summary(payload: object) -> HyperliquidMarginSummary:
        if not isinstance(payload, dict):
            return HyperliquidMarginSummary()
        return HyperliquidMarginSummary(
            account_value=_coerce_float(payload.get("accountValue", 0.0)),
            total_margin_used=_coerce_float(payload.get("totalMarginUsed", 0.0)),
            total_ntl_pos=_coerce_float(payload.get("totalNtlPos", 0.0)),
            total_raw_usd=_coerce_float(payload.get("totalRawUsd", 0.0)),
        )

    def _post_info(self, payload: Mapping[str, object]) -> object:
        response = self._client.post("/info", json=payload)
        response.raise_for_status()
        return response.json()


def _optional_float(value: object) -> float | None:
    if value is None:
        return None
    return _coerce_float(value)


def _optional_str(value: object) -> str | None:
    if value is None:
        return None
    return str(value)


def _optional_int(value: object) -> int | None:
    if value is None:
        return None
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, (float, int, str)):
        return int(value)
    raise ValueError(f"Cannot convert {value!r} to int")


def _coerce_float(value: object) -> float:
    if isinstance(value, bool):
        return float(value)
    if isinstance(value, (float, int, str)):
        return float(value)
    raise ValueError(f"Cannot convert {value!r} to float")
