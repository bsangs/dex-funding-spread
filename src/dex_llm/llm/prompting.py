from __future__ import annotations

import base64
import json
import mimetypes
from pathlib import Path

from dex_llm.models import FeatureSnapshot, MarketFrame


def load_prompt_template(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def build_router_payload(
    frame: MarketFrame,
    features: FeatureSnapshot,
) -> dict[str, object]:
    return {
        "exchange": frame.exchange,
        "symbol": frame.symbol,
        "current_price": frame.current_price,
        "atr": frame.atr,
        "map_quality": frame.map_quality.value,
        "heatmap_path": frame.heatmap_path,
        "heatmap_image_path": frame.heatmap_image_path,
        "heatmap_image_url": frame.heatmap_image_url,
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


def render_router_prompt(
    frame: MarketFrame,
    features: FeatureSnapshot,
    template: str,
) -> str:
    payload = build_router_payload(frame, features)
    return (
        f"{template.strip()}\n\n"
        "## Structured Context\n\n"
        "```json\n"
        f"{json.dumps(payload, ensure_ascii=True, indent=2)}\n"
        "```"
    )


def build_router_input(
    frame: MarketFrame,
    features: FeatureSnapshot,
    template: str,
    *,
    image_detail: str = "auto",
) -> list[dict[str, object]]:
    payload = build_router_payload(frame, features)
    user_content: list[dict[str, object]] = [
        {
            "type": "input_text",
            "text": (
                "Use the attached heatmap image when present, then classify the scene "
                "from the structured context and return a TradePlan. For every non-flat "
                "setup, choose a passive limit zone where price is likely to tag soon, "
                "not a market entry."
            ),
        }
    ]
    image_input = _build_image_input(frame, image_detail=image_detail)
    if image_input is not None:
        user_content.append(image_input)
    user_content.append(
        {
            "type": "input_text",
            "text": (
                "Structured Context\n"
                f"{json.dumps(payload, ensure_ascii=True, indent=2)}"
            ),
        }
    )
    return [
        {"role": "system", "content": template.strip()},
        {"role": "user", "content": user_content},
    ]


def _build_image_input(
    frame: MarketFrame,
    *,
    image_detail: str,
) -> dict[str, object] | None:
    if frame.heatmap_image_path:
        local_path = Path(frame.heatmap_image_path)
        if local_path.exists():
            mime_type = mimetypes.guess_type(local_path.name)[0] or "image/png"
            encoded = base64.b64encode(local_path.read_bytes()).decode("ascii")
            return {
                "type": "input_image",
                "image_url": f"data:{mime_type};base64,{encoded}",
                "detail": image_detail,
            }
    if frame.heatmap_image_url:
        return {
            "type": "input_image",
            "image_url": frame.heatmap_image_url,
            "detail": image_detail,
        }
    return None
