from __future__ import annotations

import json
from collections.abc import Mapping
from datetime import UTC, datetime
from pathlib import Path
from urllib.parse import urlparse

import httpx

from dex_llm.models import Cluster, ClusterShape, ClusterSide, HeatmapSnapshot


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
        return snapshot.model_copy(update={"raw_path": raw_path, "image_path": image_path})

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
                return below[:3], above[:3]

        longs = payload.get("longs")
        shorts = payload.get("shorts")
        if isinstance(longs, list) and isinstance(shorts, list):
            below_clusters = [
                self._cluster_from_mapping(item, ClusterSide.BELOW) for item in longs
            ]
            above_clusters = [
                self._cluster_from_mapping(item, ClusterSide.ABOVE) for item in shorts
            ]
            return below_clusters[:3], above_clusters[:3]

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
            return below_clusters[:3], above_clusters[:3]

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
        return str(path)

    def _shape_from_orders(self, orders: int) -> ClusterShape:
        if orders >= 20:
            return ClusterShape.SINGLE_WALL
        if orders >= 5:
            return ClusterShape.STAIRCASE
        return ClusterShape.DIFFUSE

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
