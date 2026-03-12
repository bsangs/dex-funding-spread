from __future__ import annotations

import json
from pathlib import Path

from dex_llm.models import FeatureSnapshot, MarketFrame


def load_prompt_template(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def render_router_prompt(
    frame: MarketFrame,
    features: FeatureSnapshot,
    template: str,
) -> str:
    payload = {
        "exchange": frame.exchange,
        "symbol": frame.symbol,
        "current_price": frame.current_price,
        "atr": frame.atr,
        "map_quality": frame.map_quality.value,
        "heatmap_path": frame.heatmap_path,
        "sweep": frame.sweep.model_dump(mode="json"),
        "position": frame.position.model_dump(mode="json"),
        "kill_switch": frame.kill_switch.model_dump(mode="json"),
        "top_clusters_above": [
            cluster.model_dump(mode="json") for cluster in frame.clusters_above[:3]
        ],
        "top_clusters_below": [
            cluster.model_dump(mode="json") for cluster in frame.clusters_below[:3]
        ],
        "features": features.model_dump(mode="json"),
        "candles_5m": [candle.model_dump(mode="json") for candle in frame.candles_5m[-6:]],
        "candles_15m": [candle.model_dump(mode="json") for candle in frame.candles_15m[-4:]],
    }
    return (
        f"{template.strip()}\n\n"
        "## Structured Context\n\n"
        "```json\n"
        f"{json.dumps(payload, ensure_ascii=True, indent=2)}\n"
        "```"
    )
