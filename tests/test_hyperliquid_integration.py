from __future__ import annotations

from datetime import UTC, datetime

from dex_llm.integrations.hyperliquid import HyperliquidInfoClient


def test_parse_l2_book_payload() -> None:
    payload = {
        "coin": "BTC",
        "time": 1_773_276_479_505,
        "levels": [
            [
                {"px": "70196.0", "sz": "1.0", "n": 3},
                {"px": "70195.0", "sz": "2.0", "n": 4},
            ],
            [
                {"px": "70197.0", "sz": "1.5", "n": 5},
                {"px": "70198.0", "sz": "2.5", "n": 6},
            ],
        ],
    }

    snapshot = HyperliquidInfoClient.parse_l2_book("BTC", payload)

    assert snapshot.best_bid == 70196.0
    assert snapshot.best_ask == 70197.0
    assert snapshot.mid_price == 70196.5


def test_parse_candles_payload() -> None:
    payload = [
        {
            "t": 1_773_272_700_000,
            "T": 1_773_272_999_999,
            "s": "BTC",
            "i": "5m",
            "o": "70196.0",
            "c": "70184.0",
            "h": "70234.0",
            "l": "70068.0",
            "v": "94.61712",
            "n": 2218,
        }
    ]

    candles = HyperliquidInfoClient.parse_candles(payload)

    assert len(candles) == 1
    assert candles[0].close == 70184.0


def test_parse_clearinghouse_state_payload() -> None:
    payload = {
        "assetPositions": [
            {
                "type": "oneWay",
                "position": {
                    "coin": "BTC",
                    "entryPx": "70196.0",
                    "leverage": {"type": "isolated", "value": 6, "rawUsd": "1200.0"},
                    "liquidationPx": "65100.0",
                    "marginUsed": "210.5",
                    "maxLeverage": "10",
                    "positionValue": "1220.0",
                    "szi": "0.4",
                    "unrealizedPnl": "18.2",
                },
            }
        ],
        "crossMaintenanceMarginUsed": "15.0",
        "crossMarginSummary": {
            "accountValue": "5500.0",
            "totalMarginUsed": "210.5",
            "totalNtlPos": "1220.0",
            "totalRawUsd": "5500.0",
        },
        "marginSummary": {
            "accountValue": "5500.0",
            "totalMarginUsed": "210.5",
            "totalNtlPos": "1220.0",
            "totalRawUsd": "5500.0",
        },
        "withdrawable": "4900.0",
        "time": 1_773_276_479_505,
    }

    state = HyperliquidInfoClient.parse_clearinghouse_state(payload)

    assert state.margin_summary.account_value == 5500.0
    assert state.asset_positions[0].coin == "BTC"
    assert state.asset_positions[0].size == 0.4
    assert state.asset_positions[0].leverage_type == "isolated"


def test_parse_frontend_open_orders_payload() -> None:
    payload = [
        {
            "coin": "BTC",
            "limitPx": "70190.0",
            "oid": 123,
            "side": "B",
            "sz": "0.2",
            "origSz": "0.2",
            "reduceOnly": False,
            "isTrigger": False,
            "orderType": "limit",
            "timestamp": 1_773_276_479_505,
        }
    ]

    orders = HyperliquidInfoClient.parse_frontend_open_orders(payload)

    assert len(orders) == 1
    assert orders[0].coin == "BTC"
    assert orders[0].limit_price == 70190.0
    assert orders[0].timestamp == datetime.fromtimestamp(1_773_276_479_505 / 1000, tz=UTC)


def test_parse_user_fills_payload() -> None:
    payload = [
        {
            "coin": "BTC",
            "px": "70120.0",
            "sz": "0.1",
            "time": 1_773_276_479_505,
            "dir": "Close Long",
            "closedPnl": "-8.4",
        }
    ]

    fills = HyperliquidInfoClient.parse_user_fills(payload)

    assert len(fills) == 1
    assert fills[0].coin == "BTC"
    assert fills[0].closed_pnl == -8.4
