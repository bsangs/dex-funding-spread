from __future__ import annotations

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

