from __future__ import annotations

import threading
from datetime import UTC, datetime, timedelta

from dex_llm.integrations.hyperliquid_live import (
    HyperliquidRestGateway,
    HyperliquidWsStateClient,
    fill_identity,
    parse_active_asset_ctx_message,
    parse_bbo_message,
    parse_live_order,
    parse_user_event,
)
from dex_llm.models import (
    HyperliquidUserFill,
    OrderRole,
    OrderState,
    UserEventType,
)


class StubRestGateway(HyperliquidRestGateway):
    def __init__(self, batches: list[list[HyperliquidUserFill]]) -> None:
        self._batches = batches
        self._calls = 0

    def fetch_user_fills_by_time(
        self,
        *,
        user: str,
        start_time: int,
        end_time: int | None = None,
        aggregate_by_time: bool = True,
    ) -> list[HyperliquidUserFill]:
        batch = self._batches[self._calls]
        self._calls += 1
        return batch


def test_paginate_user_fills_dedupes_inclusive_boundaries() -> None:
    now = datetime.now(tz=UTC)
    shared = HyperliquidUserFill(
        coin="BTC",
        closed_pnl=-12.0,
        direction="Close Long",
        price=70000.0,
        size=0.1,
        time=now - timedelta(minutes=10),
        oid=1,
        fill_hash="fill-1",
    )
    gateway = StubRestGateway(
        [
            [shared] + [
                HyperliquidUserFill(
                    coin="BTC",
                    closed_pnl=-1.0,
                    direction="Close Long",
                    price=69900.0,
                    size=0.1,
                    time=now - timedelta(minutes=11 + index),
                    oid=10 + index,
                )
                for index in range(1999)
            ],
            [shared],
        ]
    )

    fills, safe_complete = gateway.paginate_user_fills_by_time(
        user="0xuser",
        start_time=int((now - timedelta(hours=1)).timestamp() * 1000),
        end_time=int(now.timestamp() * 1000),
    )

    assert safe_complete is True
    assert len([fill for fill in fills if fill_identity(fill) == "fill-1"]) == 1


def test_parse_helpers_cover_bbo_asset_ctx_and_user_event() -> None:
    bbo = parse_bbo_message(
        {"coin": "BTC", "bid": {"px": "70000"}, "ask": {"px": "70010"}, "time": 1_773_276_479_505},
        fallback_symbol="BTC",
    )
    asset_ctx = parse_active_asset_ctx_message(
        {"coin": "BTC", "ctx": {"markPx": "70005", "oraclePx": "70002", "midPx": "70006"}},
        fallback_symbol="BTC",
    )
    user_event = parse_user_event({"type": "liquidation", "coin": "BTC", "time": 1_773_276_479_505})

    assert bbo.mid == 70005.0
    assert asset_ctx.oracle_price == 70002.0
    assert user_event.event_type == UserEventType.LIQUIDATION


def test_parse_live_order_maps_role_and_status() -> None:
    cloid = "0x2101abcdefabcdefabcdefabcdefabcd"
    order = parse_live_order(
        {
            "coin": "BTC",
            "side": "A",
            "limitPx": "70600",
            "sz": "0.1",
            "reduceOnly": True,
            "isTrigger": False,
            "orderType": "limit",
            "oid": 5,
            "cloid": cloid,
            "status": "filled",
            "timestamp": 1_773_276_479_505,
        }
    )

    assert order.status == OrderState.FILLED
    assert order.role == OrderRole.TAKE_PROFIT_1


def test_ws_state_client_ingests_messages_and_dedupes_fills() -> None:
    client = object.__new__(HyperliquidWsStateClient)
    client.info = None
    client.user_address = "0xuser"
    client._symbol = "BTC"
    client._lock = threading.Lock()
    client._order_book = None
    client._candles_5m = []
    client._candles_15m = []
    client._bbo = None
    client._active_asset_ctx = None
    client._clearinghouse_state = None
    client._open_orders = {}
    client._recent_fills = []
    client._recent_user_events = []
    client._channel_timestamps = {}
    client._channel_snapshot_flags = {}
    client._subscription_ids = []
    client._last_ping_at = None
    client._last_pong_at = None

    client.ingest_message(
        {
            "channel": "l2Book",
            "data": {
                "coin": "BTC",
                "time": 1_773_276_479_505,
                "levels": [
                    [{"px": "70000", "sz": "1.0", "n": 1}],
                    [{"px": "70010", "sz": "1.2", "n": 2}],
                ],
            },
            "isSnapshot": True,
        }
    )
    client.ingest_message(
        {
            "channel": "candle",
            "data": {
                "t": 1_773_272_700_000,
                "s": "BTC",
                "i": "5m",
                "o": "70196.0",
                "c": "70184.0",
                "h": "70234.0",
                "l": "70068.0",
                "v": "94.61712",
            },
            "isSnapshot": True,
        }
    )
    client.ingest_message(
        {
            "channel": "candle",
            "data": {
                "t": 1_773_272_700_000,
                "s": "BTC",
                "i": "15m",
                "o": "70196.0",
                "c": "70184.0",
                "h": "70234.0",
                "l": "70068.0",
                "v": "94.61712",
            },
            "isSnapshot": True,
        }
    )
    client.ingest_message(
        {
            "channel": "bbo",
            "data": {
                "coin": "BTC",
                "bbo": [{"px": "70000"}, {"px": "70010"}],
                "time": 1_773_276_479_505,
            },
        }
    )
    client.ingest_message(
        {
            "channel": "activeAssetCtx",
            "data": {
                "coin": "BTC",
                "ctx": {"oraclePx": "70001", "markPx": "70003", "midPx": "70005"},
            },
        }
    )
    client.ingest_message(
        {
            "channel": "userFills",
            "data": {
                "fills": [
                    {
                        "coin": "BTC",
                        "px": "70020.0",
                        "sz": "0.1",
                        "time": 1_773_276_479_505,
                        "dir": "Close Long",
                        "closedPnl": "-8.4",
                        "hash": "fill-1",
                    },
                    {
                        "coin": "BTC",
                        "px": "70020.0",
                        "sz": "0.1",
                        "time": 1_773_276_479_505,
                        "dir": "Close Long",
                        "closedPnl": "-8.4",
                        "hash": "fill-1",
                    },
                ]
            },
        }
    )
    client.ingest_message(
        {
            "channel": "userEvents",
            "data": {"type": "liquidation", "coin": "BTC", "time": 1_773_276_479_505},
        }
    )

    snapshot = client.snapshot()

    assert snapshot.order_book.mid_price == 70005.0
    assert snapshot.bbo is not None and snapshot.bbo.mid == 70005.0
    assert (
        snapshot.active_asset_ctx is not None
        and snapshot.active_asset_ctx.oracle_price == 70001.0
    )
    assert len(snapshot.recent_fills) == 1
    assert snapshot.recent_user_events[-1].event_type == UserEventType.LIQUIDATION


def test_ws_state_client_public_ready_does_not_require_private_snapshot() -> None:
    client = object.__new__(HyperliquidWsStateClient)
    client.info = None
    client.user_address = "0xuser"
    client._symbol = "BTC"
    client._lock = threading.Lock()
    client._order_book = object()
    client._candles_5m = [object()]
    client._candles_15m = [object()]
    client._bbo = object()
    client._active_asset_ctx = object()
    client._clearinghouse_state = None
    client._open_orders = {}
    client._recent_fills = []
    client._recent_user_events = []
    client._channel_timestamps = {}
    client._channel_snapshot_flags = {}
    client._subscription_ids = []
    client._last_ping_at = None
    client._last_pong_at = None

    client.wait_until_public_ready(timeout_s=0.01)

    assert client.private_state_ready() is False
