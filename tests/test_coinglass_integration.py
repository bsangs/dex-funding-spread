from __future__ import annotations

from dex_llm.integrations.coinglass import CoinGlassHeatmapClient
from dex_llm.models import ClusterSide


def test_parse_heatmap_payload_from_levels_shape() -> None:
    client = CoinGlassHeatmapClient(api_key="test")
    payload = {
        "data": {
            "time": 1_773_276_479_505,
            "imageUrl": "https://example.com/heatmap.png",
            "levels": [
                [
                    {"px": "70000", "sz": "12.5", "n": 22},
                    {"px": "69950", "sz": "8.0", "n": 11},
                ],
                [
                    {"px": "70250", "sz": "14.0", "n": 25},
                    {"px": "70300", "sz": "7.5", "n": 6},
                ],
            ],
        }
    }

    snapshot = client.parse_heatmap_payload("BTC", payload)

    assert snapshot.provider == "coinglass"
    assert snapshot.image_url == "https://example.com/heatmap.png"
    assert snapshot.clusters_below[0].side == ClusterSide.BELOW
    assert snapshot.clusters_above[0].side == ClusterSide.ABOVE
    client.close()


def test_parse_heatmap_payload_from_longs_and_shorts() -> None:
    client = CoinGlassHeatmapClient(api_key="test")
    payload = {
        "data": {
            "longs": [
                {"price": "69900", "liquidationAmount": "5.5", "orders": "3"},
            ],
            "shorts": [
                {"price": "70350", "liquidationAmount": "6.5", "orders": "21"},
            ],
        }
    }

    snapshot = client.parse_heatmap_payload("BTC", payload)

    assert snapshot.clusters_below[0].price == 69900.0
    assert snapshot.clusters_above[0].price == 70350.0
    client.close()

