from __future__ import annotations

import base64
import gzip
import json
from collections.abc import Mapping
from datetime import UTC, datetime
from pathlib import Path
from time import time
from urllib.parse import urlparse

import httpx
from Crypto.Cipher import AES
from Crypto.Util.Padding import unpad
from playwright.sync_api import (
    Page,
    sync_playwright,
)
from playwright.sync_api import (
    TimeoutError as PlaywrightTimeoutError,
)

from dex_llm.models import Cluster, ClusterShape, ClusterSide, HeatmapSnapshot

MAX_HEATMAP_CACHE_FILES = 10


class CoinGlassHyperliquidLiqMapClient:
    def __init__(
        self,
        *,
        base_url: str = "https://capi.coinglass.com",
        liq_map_path: str = "/api/hyperliquid/topPosition/liqMap",
        timeout_s: float = 10.0,
        cache_dir: Path = Path("data/heatmaps"),
        transport: httpx.BaseTransport | None = None,
        obe: str | None = None,
        language: str = "en",
    ) -> None:
        self.base_url = base_url
        self.liq_map_path = liq_map_path
        self.cache_dir = cache_dir
        self.obe = obe
        self.language = language
        self.user_agent = (
            "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/145.0.0.0 Safari/537.36"
        )
        self._client = httpx.Client(
            base_url=base_url,
            timeout=timeout_s,
            transport=transport,
        )

    def close(self) -> None:
        self._client.close()

    def fetch_heatmap(
        self,
        symbol: str,
        extra_params: Mapping[str, str] | None = None,
        cache_image: bool = True,
    ) -> HeatmapSnapshot:
        _ = cache_image
        params = dict(extra_params or {})
        params["symbol"] = symbol.upper()
        request_headers = self._request_headers()
        response = self._client.get(
            self.liq_map_path,
            params=params,
            headers=request_headers,
        )
        response.raise_for_status()
        payload = response.json()
        decoded = self._decode_encrypted_response(
            payload,
            response.headers,
            request_headers=request_headers,
        )
        snapshot = self.parse_liq_map_payload(symbol=symbol.upper(), payload=decoded)
        raw_path = self._write_raw_payload(symbol.upper(), decoded, prefix="coinglass-hl-liqmap")
        return snapshot.model_copy(update={"raw_path": raw_path})

    def parse_liq_map_payload(self, *, symbol: str, payload: object) -> HeatmapSnapshot:
        if not isinstance(payload, dict):
            raise ValueError("Unexpected CoinGlass Hyperliquid liqMap payload")
        current_price = _coerce_float(payload.get("price"))
        positions = payload.get("list")
        if not isinstance(positions, list):
            raise ValueError("liqMap payload missing list")
        below_levels: dict[float, dict[str, float]] = {}
        above_levels: dict[float, dict[str, float]] = {}
        detailed_positions: list[dict[str, object]] = []
        latest_timestamp: datetime | None = None
        for item in positions:
            if not isinstance(item, dict):
                continue
            liquidation_price = _coerce_float(item.get("liquidationPrice"))
            if liquidation_price is None or liquidation_price <= 0:
                continue
            weight = abs(
                _coerce_float(item.get("positionUsd"))
                or _coerce_float(item.get("margin"))
                or _coerce_float(item.get("size"))
                or 0.0
            )
            if weight <= 0:
                continue
            update_time = _extract_timestamp_value(item.get("updateTime"))
            if (
                update_time is not None
                and (latest_timestamp is None or update_time > latest_timestamp)
            ):
                latest_timestamp = update_time
            target = below_levels if liquidation_price < current_price else above_levels
            bucket = target.setdefault(liquidation_price, {"size": 0.0, "orders": 0.0})
            bucket["size"] += weight
            bucket["orders"] += 1
            detailed_positions.append(
                {
                    "liquidation_price": liquidation_price,
                    "position_usd": _coerce_float(item.get("positionUsd")) or 0.0,
                    "leverage": _coerce_float(item.get("leverage")) or 0.0,
                    "size": _coerce_float(item.get("size")) or 0.0,
                    "entry_price": _coerce_float(item.get("entryPrice")),
                    "mark_price": _coerce_float(item.get("price")),
                    "position_type": item.get("positionType"),
                    "direction": (
                        "long"
                        if (_coerce_float(item.get("size")) or 0.0) > 0
                        else "short"
                    ),
                    "update_time": item.get("updateTime"),
                }
            )

        return HeatmapSnapshot(
            provider="coinglass-hyperliquid-liqmap",
            symbol=symbol,
            captured_at=latest_timestamp or datetime.now(tz=UTC),
            clusters_above=self._levels_to_clusters(above_levels, ClusterSide.ABOVE),
            clusters_below=self._levels_to_clusters(below_levels, ClusterSide.BELOW),
            metadata={
                "source": "capi-hyperliquid-topPosition-liqMap",
                "price": current_price,
                "positions_count": len(positions),
                "positions": detailed_positions,
                "levels_above": self._levels_to_metadata(above_levels),
                "levels_below": self._levels_to_metadata(below_levels),
            },
        )

    def _request_headers(self) -> dict[str, str]:
        headers = {
            "accept": "application/json",
            "accept-language": "en-US,en;q=0.7",
            "cache-ts-v2": str(int(time() * 1000)),
            "encryption": "true",
            "language": self.language,
            "origin": "https://www.coinglass.com",
            "referer": "https://www.coinglass.com/",
            "sec-ch-ua": '"Chromium";v="145", "Not:A-Brand";v="99"',
            "sec-ch-ua-mobile": "?0",
            "sec-ch-ua-platform": '"macOS"',
            "sec-fetch-dest": "empty",
            "sec-fetch-mode": "cors",
            "sec-fetch-site": "same-site",
            "user-agent": self.user_agent,
        }
        if self.obe:
            headers["obe"] = self.obe
        return headers

    def _decode_encrypted_response(
        self,
        payload: object,
        headers: Mapping[str, str],
        *,
        request_headers: Mapping[str, str] | None = None,
    ) -> dict[str, object]:
        if not isinstance(payload, dict):
            raise ValueError("Unexpected liqMap response envelope")
        encrypted_data = payload.get("data")
        if not isinstance(encrypted_data, str):
            raise ValueError("liqMap response missing encrypted data")
        time_header = headers.get("time")
        if not time_header and request_headers is not None:
            time_header = request_headers.get("cache-ts-v2")
        user_header = headers.get("user")
        if not time_header or not user_header:
            raise ValueError("liqMap response missing decryption headers")
        seed_key = base64.b64encode(time_header.encode("utf-8")).decode("ascii")[:16]
        user_key_gzip = self._aes_ecb_decrypt_base64(user_header, seed_key)
        data_key = gzip.decompress(user_key_gzip).decode("utf-8")
        decoded_gzip = self._aes_ecb_decrypt_base64(encrypted_data, data_key)
        decoded = gzip.decompress(decoded_gzip).decode("utf-8")
        parsed = json.loads(decoded)
        if not isinstance(parsed, dict):
            raise ValueError("Decoded liqMap payload is not a JSON object")
        return parsed

    def _aes_ecb_decrypt_base64(self, cipher_text_base64: str, key: str) -> bytes:
        cipher = AES.new(key.encode("utf-8"), AES.MODE_ECB)
        encrypted = base64.b64decode(cipher_text_base64)
        return unpad(cipher.decrypt(encrypted), AES.block_size)

    def _levels_to_clusters(
        self,
        levels: dict[float, dict[str, float]],
        side: ClusterSide,
    ) -> list[Cluster]:
        clusters = [
            Cluster(
                side=side,
                price=price,
                size=data["size"],
                shape=self._shape_from_orders(int(data["orders"])),
            )
            for price, data in levels.items()
        ]
        return sorted(clusters, key=lambda cluster: cluster.size, reverse=True)[:3]

    def _levels_to_metadata(
        self,
        levels: dict[float, dict[str, float]],
    ) -> list[dict[str, float]]:
        return [
            {
                "price": price,
                "size": data["size"],
                "orders": data["orders"],
            }
            for price, data in sorted(levels.items(), key=lambda item: item[0])
        ]

    def _shape_from_orders(self, orders: int) -> ClusterShape:
        if orders >= 20:
            return ClusterShape.SINGLE_WALL
        if orders >= 5:
            return ClusterShape.STAIRCASE
        return ClusterShape.DIFFUSE

    def _write_raw_payload(
        self,
        symbol: str,
        payload: object,
        *,
        prefix: str,
    ) -> str:
        raw_dir = self.cache_dir / "raw"
        raw_dir.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now(tz=UTC).strftime("%Y%m%dT%H%M%SZ")
        path = raw_dir / f"{prefix}-{symbol.lower()}-{stamp}.json"
        path.write_text(json.dumps(payload, ensure_ascii=True, indent=2), encoding="utf-8")
        _prune_cache_dir(raw_dir)
        return str(path)


class CoinGlassHeatmapClient:
    def __init__(
        self,
        api_key: str,
        base_url: str = "https://open-api-v4.coinglass.com",
        heatmap_path: str = "/api/futures/liquidation/aggregated-heatmap/model1",
        timeout_s: float = 10.0,
        cache_dir: Path = Path("data/heatmaps"),
        transport: httpx.BaseTransport | None = None,
    ) -> None:
        if not api_key:
            raise ValueError("CoinGlass API key is required")
        self.heatmap_path = heatmap_path
        self.cache_dir = cache_dir
        self._client = httpx.Client(
            base_url=base_url,
            timeout=timeout_s,
            headers={"CG-API-KEY": api_key},
            transport=transport,
        )

    def close(self) -> None:
        self._client.close()

    def fetch_heatmap(
        self,
        symbol: str,
        extra_params: Mapping[str, str] | None = None,
        cache_image: bool = True,
    ) -> HeatmapSnapshot:
        params = dict(extra_params or {})
        if "symbol" not in params and "coin" not in params:
            params["symbol"] = symbol

        response = self._client.get(self.heatmap_path, params=params)
        response.raise_for_status()
        payload = response.json()
        snapshot = self.parse_heatmap_payload(symbol=symbol, payload=payload)
        raw_path = self._write_raw_payload(symbol, payload)
        image_path: str | None = None
        if cache_image and snapshot.image_url is not None:
            image_path = self._download_image(symbol, snapshot.image_url)
        return snapshot.model_copy(
            update={
                "raw_path": raw_path,
                "image_path": image_path,
                "heatmap_image_path": image_path,
            }
        )

    def parse_heatmap_payload(self, symbol: str, payload: object) -> HeatmapSnapshot:
        normalized = self._normalize(payload)
        image_url = self._find_image_url(normalized)
        timestamp = self._extract_timestamp(normalized)
        clusters_below, clusters_above = self._extract_clusters(normalized)
        return HeatmapSnapshot(
            provider="coinglass",
            symbol=symbol,
            captured_at=timestamp,
            clusters_above=clusters_above,
            clusters_below=clusters_below,
            heatmap_image_url=image_url,
            image_url=image_url,
            metadata={"response_keys": list(normalized.keys())},
        )

    def _normalize(self, payload: object) -> dict[str, object]:
        if not isinstance(payload, dict):
            raise ValueError("Unexpected CoinGlass payload")
        data = payload.get("data")
        if isinstance(data, dict):
            return data
        return payload

    def _extract_clusters(
        self,
        payload: dict[str, object],
    ) -> tuple[list[Cluster], list[Cluster]]:
        if "levels" in payload and isinstance(payload["levels"], list):
            levels = payload["levels"]
            if len(levels) >= 2 and isinstance(levels[0], list) and isinstance(levels[1], list):
                below = [self._cluster_from_level(item, ClusterSide.BELOW) for item in levels[0]]
                above = [self._cluster_from_level(item, ClusterSide.ABOVE) for item in levels[1]]
                return self._top_three(below), self._top_three(above)

        longs = payload.get("longs")
        shorts = payload.get("shorts")
        if isinstance(longs, list) and isinstance(shorts, list):
            below_clusters = [
                self._cluster_from_mapping(item, ClusterSide.BELOW) for item in longs
            ]
            above_clusters = [
                self._cluster_from_mapping(item, ClusterSide.ABOVE) for item in shorts
            ]
            return self._top_three(below_clusters), self._top_three(above_clusters)

        clusters = payload.get("clusters")
        if isinstance(clusters, list):
            below_clusters = []
            above_clusters = []
            for item in clusters:
                cluster = self._cluster_from_side_mapping(item)
                if cluster.side == ClusterSide.BELOW:
                    below_clusters.append(cluster)
                else:
                    above_clusters.append(cluster)
            return self._top_three(below_clusters), self._top_three(above_clusters)

        raise ValueError("Unsupported CoinGlass response schema")

    def _cluster_from_level(self, payload: object, side: ClusterSide) -> Cluster:
        if not isinstance(payload, dict):
            raise ValueError("Unexpected CoinGlass level payload")
        orders = int(payload.get("n", 0))
        return Cluster(
            side=side,
            price=float(payload["px"]),
            size=float(payload["sz"]),
            shape=self._shape_from_orders(orders),
        )

    def _cluster_from_mapping(self, payload: object, side: ClusterSide) -> Cluster:
        if not isinstance(payload, dict):
            raise ValueError("Unexpected CoinGlass cluster mapping")
        orders = self._to_int(self._pick(payload, ("n", "count", "orders"), default=0))
        return Cluster(
            side=side,
            price=self._to_float(self._pick(payload, ("price", "px", "level"))),
            size=self._to_float(self._pick(payload, ("size", "sz", "amount", "liquidationAmount"))),
            shape=self._shape_from_orders(orders),
        )

    def _cluster_from_side_mapping(self, payload: object) -> Cluster:
        if not isinstance(payload, dict):
            raise ValueError("Unexpected CoinGlass cluster mapping")
        side_value = str(self._pick(payload, ("side",), default="above")).lower()
        side = ClusterSide.ABOVE if side_value == "above" else ClusterSide.BELOW
        return self._cluster_from_mapping(payload, side)

    def _find_image_url(self, payload: dict[str, object]) -> str | None:
        for key in ("imageUrl", "image_url", "heatmapUrl", "heatmap_url", "url"):
            value = payload.get(key)
            if isinstance(value, str) and value.startswith("http"):
                return value
        return None

    def _extract_timestamp(self, payload: dict[str, object]) -> datetime:
        for key in ("time", "timestamp", "updatedAt", "updated_at"):
            value = payload.get(key)
            if isinstance(value, (int, float)):
                seconds = value / 1000 if value > 10_000_000_000 else value
                return datetime.fromtimestamp(seconds, tz=UTC)
            if isinstance(value, str):
                try:
                    return datetime.fromisoformat(value.replace("Z", "+00:00"))
                except ValueError:
                    continue
        return datetime.now(tz=UTC)

    def _write_raw_payload(self, symbol: str, payload: object) -> str:
        raw_dir = self.cache_dir / "raw"
        raw_dir.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now(tz=UTC).strftime("%Y%m%dT%H%M%SZ")
        path = raw_dir / f"coinglass-{symbol.lower()}-{stamp}.json"
        path.write_text(json.dumps(payload, ensure_ascii=True, indent=2), encoding="utf-8")
        _prune_cache_dir(raw_dir)
        return str(path)

    def _download_image(self, symbol: str, image_url: str) -> str:
        parsed = urlparse(image_url)
        suffix = Path(parsed.path).suffix or ".png"
        images_dir = self.cache_dir / "images"
        images_dir.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now(tz=UTC).strftime("%Y%m%dT%H%M%SZ")
        path = images_dir / f"coinglass-{symbol.lower()}-{stamp}{suffix}"
        with self._client.stream("GET", image_url) as response:
            response.raise_for_status()
            path.write_bytes(response.read())
        _prune_cache_dir(images_dir)
        return str(path)

    def _shape_from_orders(self, orders: int) -> ClusterShape:
        if orders >= 20:
            return ClusterShape.SINGLE_WALL
        if orders >= 5:
            return ClusterShape.STAIRCASE
        return ClusterShape.DIFFUSE

    def _top_three(self, clusters: list[Cluster]) -> list[Cluster]:
        return sorted(clusters, key=lambda cluster: cluster.size, reverse=True)[:3]

    def _pick(
        self,
        payload: dict[str, object],
        keys: tuple[str, ...],
        default: object | None = None,
    ) -> object:
        for key in keys:
            if key in payload:
                return payload[key]
        if default is not None:
            return default
        raise KeyError(f"Missing any of {keys}")

    def _to_float(self, value: object) -> float:
        if isinstance(value, (int, float)):
            return float(value)
        if isinstance(value, str):
            return float(value)
        raise ValueError(f"Cannot convert {value!r} to float")

    def _to_int(self, value: object) -> int:
        if isinstance(value, int):
            return value
        if isinstance(value, float):
            return int(value)
        if isinstance(value, str):
            return int(value)
        raise ValueError(f"Cannot convert {value!r} to int")


class CoinGlassLiquidationsPageClient:
    def __init__(
        self,
        *,
        page_url: str = "https://www.coinglass.com/ko/liquidations",
        timeout_s: float = 20.0,
        cache_dir: Path = Path("data/heatmaps"),
    ) -> None:
        self.page_url = page_url
        self.timeout_s = timeout_s
        self.cache_dir = cache_dir
        self.user_agent = (
            "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/145.0.0.0 Safari/537.36"
        )

    def close(self) -> None:
        return None

    def fetch_heatmap(
        self,
        symbol: str,
        extra_params: Mapping[str, str] | None = None,
        cache_image: bool = True,
    ) -> HeatmapSnapshot:
        _ = extra_params, cache_image
        symbol = symbol.upper()
        captured_at = datetime.now(tz=UTC)
        screenshot_path: str | None = None
        body_text = ""
        title = ""
        with sync_playwright() as playwright:
            browser = playwright.chromium.launch(headless=True)
            page = browser.new_page(
                user_agent=self.user_agent,
                locale="ko-KR",
            )
            try:
                page.goto(
                    self.page_url,
                    wait_until="domcontentloaded",
                    timeout=int(self.timeout_s * 1000),
                )
                page.wait_for_timeout(5000)
                title = page.title()
                body_text = page.locator("body").inner_text()
                screenshot_path = self._write_screenshot(symbol, captured_at, page)
            except PlaywrightTimeoutError as exc:
                raise RuntimeError(f"CoinGlass web scrape timed out: {exc}") from exc
            finally:
                browser.close()

        raw_path = self._write_raw_payload(
            symbol,
            {
                "page_url": self.page_url,
                "captured_at": captured_at.isoformat(),
                "title": title,
                "symbol": symbol,
                "symbol_context": self._extract_symbol_context(body_text, symbol),
                "body_excerpt": body_text[:8000],
            },
            prefix="coinglass-web",
        )
        return HeatmapSnapshot(
            provider="coinglass-web-scrape",
            symbol=symbol,
            captured_at=captured_at,
            clusters_above=[],
            clusters_below=[],
            heatmap_image_path=screenshot_path,
            image_path=screenshot_path,
            raw_path=raw_path,
            metadata={
                "page_url": self.page_url,
                "title": title,
                "symbol_context": self._extract_symbol_context(body_text, symbol),
                "symbol_visible": symbol in body_text.split(),
                "cluster_source": "playwright-page",
            },
        )

    def _write_screenshot(self, symbol: str, captured_at: datetime, page: Page) -> str:
        images_dir = self.cache_dir / "images"
        images_dir.mkdir(parents=True, exist_ok=True)
        stamp = captured_at.strftime("%Y%m%dT%H%M%SZ")
        path = images_dir / f"coinglass-web-{symbol.lower()}-{stamp}.png"
        page.screenshot(path=str(path), full_page=True)
        _prune_cache_dir(images_dir)
        return str(path)

    def _write_raw_payload(
        self,
        symbol: str,
        payload: object,
        *,
        prefix: str,
    ) -> str:
        raw_dir = self.cache_dir / "raw"
        raw_dir.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now(tz=UTC).strftime("%Y%m%dT%H%M%SZ")
        path = raw_dir / f"{prefix}-{symbol.lower()}-{stamp}.json"
        path.write_text(json.dumps(payload, ensure_ascii=True, indent=2), encoding="utf-8")
        _prune_cache_dir(raw_dir)
        return str(path)

    @staticmethod
    def _extract_symbol_context(body_text: str, symbol: str) -> list[str]:
        lines = [line.strip() for line in body_text.splitlines() if line.strip()]
        for index, line in enumerate(lines):
            if line.upper() != symbol:
                continue
            start = max(0, index - 2)
            end = min(len(lines), index + 4)
            return lines[start:end]
        return []


class CoinGlassFallbackHeatmapClient:
    def __init__(
        self,
        primary: CoinGlassHeatmapClient | None,
        fallback: CoinGlassLiquidationsPageClient | None,
    ) -> None:
        self.primary = primary
        self.fallback = fallback

    def close(self) -> None:
        if self.primary is not None:
            self.primary.close()
        if self.fallback is not None:
            self.fallback.close()

    def fetch_heatmap(
        self,
        symbol: str,
        extra_params: Mapping[str, str] | None = None,
        cache_image: bool = True,
    ) -> HeatmapSnapshot:
        last_error: Exception | None = None
        if self.primary is not None:
            try:
                return self.primary.fetch_heatmap(
                    symbol,
                    extra_params=extra_params,
                    cache_image=cache_image,
                )
            except Exception as exc:
                last_error = exc
        if self.fallback is None:
            if last_error is None:
                raise ValueError("no CoinGlass provider configured")
            raise last_error
        snapshot = self.fallback.fetch_heatmap(
            symbol,
            extra_params=extra_params,
            cache_image=cache_image,
        )
        if last_error is not None:
            snapshot.metadata["api_error"] = str(last_error)
        return snapshot


def _prune_cache_dir(directory: Path, *, keep_latest: int = MAX_HEATMAP_CACHE_FILES) -> None:
    files = sorted(
        (path for path in directory.iterdir() if path.is_file()),
        key=lambda path: path.name,
        reverse=True,
    )
    for stale_path in files[keep_latest:]:
        stale_path.unlink(missing_ok=True)


def _coerce_float(value: object) -> float | None:
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str) and value:
        return float(value)
    return None


def _extract_timestamp_value(value: object) -> datetime | None:
    if isinstance(value, (int, float)):
        seconds = value / 1000 if value > 10_000_000_000 else value
        return datetime.fromtimestamp(seconds, tz=UTC)
    if isinstance(value, str):
        try:
            return datetime.fromisoformat(value.replace("Z", "+00:00"))
        except ValueError:
            return None
    return None
