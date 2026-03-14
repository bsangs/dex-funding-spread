from __future__ import annotations

import base64
import gzip
import json
from datetime import UTC, datetime
from pathlib import Path

import httpx
from Crypto.Cipher import AES
from Crypto.Util.Padding import pad

from dex_llm.integrations.coinglass import (
    CoinGlassFallbackHeatmapClient,
    CoinGlassHeatmapClient,
    CoinGlassHyperliquidLiqMapClient,
    CoinGlassLiquidationsPageClient,
    _prune_cache_dir,
)
from dex_llm.models import ClusterSide, HeatmapSnapshot


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


def test_fallback_client_uses_playwright_snapshot_when_api_fails() -> None:
    class BrokenApiClient:
        def fetch_heatmap(
            self,
            symbol: str,
            extra_params=None,
            cache_image: bool = True,
        ) -> HeatmapSnapshot:
            raise ValueError("Upgrade plan")

        def close(self) -> None:
            return None

    class FakeWebClient:
        def fetch_heatmap(
            self,
            symbol: str,
            extra_params=None,
            cache_image: bool = True,
        ) -> HeatmapSnapshot:
            return HeatmapSnapshot(
                provider="coinglass-web-scrape",
                symbol=symbol,
                captured_at=datetime.now(tz=UTC),
                clusters_above=[],
                clusters_below=[],
                heatmap_image_path="data/heatmaps/images/coinglass-web-eth.png",
                metadata={"symbol_visible": True},
            )

        def close(self) -> None:
            return None

    client = CoinGlassFallbackHeatmapClient(BrokenApiClient(), FakeWebClient())

    snapshot = client.fetch_heatmap("ETH")

    assert snapshot.provider == "coinglass-web-scrape"
    assert snapshot.metadata["api_error"] == "Upgrade plan"


def test_web_client_extract_symbol_context() -> None:
    lines = CoinGlassLiquidationsPageClient._extract_symbol_context(
        "청산 히트맵\nBTC\n$100K\nETH\n$200K\nSOL\n$50K",
        "ETH",
    )

    assert lines == ["BTC", "$100K", "ETH", "$200K", "SOL", "$50K"]


def test_prune_cache_dir_keeps_only_latest_ten_files(tmp_path: Path) -> None:
    for index in range(12):
        path = tmp_path / f"coinglass-eth-20260314T120{index:02d}Z.json"
        path.write_text("{}", encoding="utf-8")

    _prune_cache_dir(tmp_path)

    remaining = sorted(path.name for path in tmp_path.iterdir())
    assert len(remaining) == 10
    assert remaining[0] == "coinglass-eth-20260314T12002Z.json"
    assert remaining[-1] == "coinglass-eth-20260314T12011Z.json"


def test_hyperliquid_liqmap_client_decodes_encrypted_payload(tmp_path: Path) -> None:
    decoded_payload = {
        "price": 2133.9,
        "list": [
            {
                "liquidationPrice": 1560.6,
                "positionUsd": 149_416_436.305,
                "updateTime": 1_773_421_324_000,
            },
            {
                "liquidationPrice": 2300.0,
                "positionUsd": 10_000_000.0,
                "updateTime": 1_773_421_325_000,
            },
            {
                "liquidationPrice": 2300.0,
                "positionUsd": 12_000_000.0,
                "updateTime": 1_773_421_326_000,
            },
        ],
    }
    time_header = "1773421366426"
    seed_key = base64.b64encode(time_header.encode("utf-8")).decode("ascii")[:16]
    data_key = "ae8c3261aa204775"
    user_header = _encrypt_and_b64(gzip.compress(data_key.encode("utf-8")), seed_key)
    data_body = _encrypt_and_b64(
        gzip.compress(json.dumps(decoded_payload, separators=(",", ":")).encode("utf-8")),
        data_key,
    )

    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.params["symbol"] == "ETH"
        assert request.headers["encryption"] == "true"
        return httpx.Response(
            200,
            json={
                "code": "0",
                "msg": "success",
                "data": data_body,
                "success": True,
            },
            headers={
                "time": time_header,
                "user": user_header,
                "encryption": "true",
                "v": "2",
            },
        )

    client = CoinGlassHyperliquidLiqMapClient(
        cache_dir=tmp_path,
        transport=httpx.MockTransport(handler),
    )

    snapshot = client.fetch_heatmap("ETH")

    assert snapshot.provider == "coinglass-hyperliquid-liqmap"
    assert snapshot.metadata["price"] == 2133.9
    assert len(snapshot.metadata["positions"]) == 3
    assert len(snapshot.metadata["levels_above"]) == 1
    assert len(snapshot.metadata["levels_below"]) == 1
    assert snapshot.raw_path is not None
    assert snapshot.clusters_below[0].price == 1560.6
    assert snapshot.clusters_above[0].price == 2300.0
    assert snapshot.clusters_above[0].size == 22_000_000.0
    client.close()


def test_hyperliquid_liqmap_client_decodes_with_cache_ts_when_time_header_missing(
    tmp_path: Path,
) -> None:
    decoded_payload = {
        "price": 2133.9,
        "list": [
            {
                "liquidationPrice": 1560.6,
                "positionUsd": 149_416_436.305,
                "updateTime": 1_773_421_324_000,
            }
        ],
    }
    cache_ts_header = "1773453486514"
    seed_key = base64.b64encode(cache_ts_header.encode("utf-8")).decode("ascii")[:16]
    data_key = "96539205e5364856"
    user_header = _encrypt_and_b64(gzip.compress(data_key.encode("utf-8")), seed_key)
    data_body = _encrypt_and_b64(
        gzip.compress(json.dumps(decoded_payload, separators=(",", ":")).encode("utf-8")),
        data_key,
    )

    def handler(request: httpx.Request) -> httpx.Response:
        assert request.headers["cache-ts-v2"] == cache_ts_header
        return httpx.Response(
            200,
            json={
                "code": "0",
                "msg": "success",
                "data": data_body,
                "success": True,
            },
            headers={
                "user": user_header,
                "encryption": "true",
                "v": "0",
            },
        )

    client = CoinGlassHyperliquidLiqMapClient(
        cache_dir=tmp_path,
        transport=httpx.MockTransport(handler),
    )
    original_request_headers = client._request_headers

    def fixed_request_headers() -> dict[str, str]:
        headers = original_request_headers()
        headers["cache-ts-v2"] = cache_ts_header
        return headers

    client._request_headers = fixed_request_headers  # type: ignore[method-assign]

    snapshot = client.fetch_heatmap("ETH")

    assert snapshot.provider == "coinglass-hyperliquid-liqmap"
    assert snapshot.metadata["price"] == 2133.9
    assert len(snapshot.metadata["positions"]) == 1
    assert snapshot.clusters_below[0].price == 1560.6
    client.close()


def _encrypt_and_b64(payload: bytes, key: str) -> str:
    cipher = AES.new(key.encode("utf-8"), AES.MODE_ECB)
    return base64.b64encode(cipher.encrypt(pad(payload, AES.block_size))).decode("ascii")
