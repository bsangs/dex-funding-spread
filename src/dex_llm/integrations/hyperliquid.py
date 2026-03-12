from __future__ import annotations

import time
from datetime import UTC, datetime

import httpx

from dex_llm.models import Candle, OrderBookSnapshot, PriceLevel


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

    def close(self) -> None:
        self._client.close()

    def fetch_l2_book(self, coin: str) -> OrderBookSnapshot:
        payload = {"type": "l2Book", "coin": coin}
        response = self._client.post("/info", json=payload)
        response.raise_for_status()
        return self.parse_l2_book(coin, response.json())

    def fetch_candles(self, coin: str, interval: str, limit: int) -> list[Candle]:
        interval_ms = self._interval_to_ms(interval)
        now_ms = int(time.time() * 1000)
        start_ms = now_ms - limit * interval_ms
        payload = {
            "type": "candleSnapshot",
            "req": {
                "coin": coin,
                "interval": interval,
                "startTime": start_ms,
                "endTime": now_ms,
            },
        }
        response = self._client.post("/info", json=payload)
        response.raise_for_status()
        return self.parse_candles(response.json())

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
    def _parse_price_level(payload: object) -> PriceLevel:
        if not isinstance(payload, dict):
            raise ValueError("Unexpected price level payload")
        return PriceLevel(
            price=float(payload["px"]),
            size=float(payload["sz"]),
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
            "4h": 14_400_000,
            "1d": 86_400_000,
        }
        if interval not in mapping:
            raise ValueError(f"Unsupported interval: {interval}")
        return mapping[interval]

