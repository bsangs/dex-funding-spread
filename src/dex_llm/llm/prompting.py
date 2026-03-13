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
    heatmap_metadata = frame.metadata.get("heatmap_metadata")
    if not isinstance(heatmap_metadata, dict):
        heatmap_metadata = {}
    return {
        "exchange": frame.exchange,
        "symbol": frame.symbol,
        "atr": frame.atr,
        "map_quality": frame.map_quality.value,
        "heatmap_path": frame.heatmap_path,
        "heatmap_image_path": frame.heatmap_image_path,
        "heatmap_image_url": frame.heatmap_image_url,
        "sweep": frame.sweep.model_dump(mode="json"),
        "kill_switch": frame.kill_switch.model_dump(mode="json"),
        "clusters_above": [
            cluster.model_dump(mode="json") for cluster in frame.clusters_above
        ],
        "clusters_below": [
            cluster.model_dump(mode="json") for cluster in frame.clusters_below
        ],
        "heatmap_levels_above_detailed": heatmap_metadata.get("levels_above"),
        "heatmap_levels_below_detailed": heatmap_metadata.get("levels_below"),
        "heatmap_positions": heatmap_metadata.get("positions"),
        "higher_timeframe_levels": frame.metadata.get("higher_timeframe_levels"),
        "entry_candidates": [
            candidate.model_dump(mode="json") for candidate in features.entry_candidates
        ],
        "features": features.model_dump(mode="json"),
        "candles_5m": [candle.model_dump(mode="json") for candle in frame.candles_5m[-6:]],
        "candles_15m": [candle.model_dump(mode="json") for candle in frame.candles_15m[-12:]],
        "candles_1h": [candle.model_dump(mode="json") for candle in frame.candles_1h[-12:]],
        "candles_4h": [candle.model_dump(mode="json") for candle in frame.candles_4h[-8:]],
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
    policy_feedback: list[str] | None = None,
    previous_plan: dict[str, object] | None = None,
) -> list[dict[str, object]]:
    payload = build_router_payload(frame, features)
    if policy_feedback:
        payload["policy_feedback"] = policy_feedback
    if previous_plan is not None:
        payload["previous_plan"] = previous_plan
    user_content: list[dict[str, object]] = [
        {
            "type": "input_text",
            "text": (
                "첨부된 히트맵 이미지가 있으면 우선 참고하고, 구조화된 컨텍스트를 바탕으로 "
                "TradePlan 하나만 반환하세요. 신규 진입은 시장가가 아니라 지정가만 사용하세요. "
                "현재가에 바로 체결되는지보다 다음 10~30분 가격 경로를 먼저 가정하고, 그 경로 안에서 "
                "도달 가능성이 높은 지정가 구간을 고르세요. 현재가 주변에 즉시 체결될 자리가 없다는 "
                "이유만으로 no_trade를 반환하지 마세요."
            ),
        }
    ]
    if policy_feedback:
        user_content.append(
            {
                "type": "input_text",
                "text": (
                    "안전 제약 피드백:\n"
                    + "\n".join(f"- {item}" for item in policy_feedback)
                    + "\n이 제약을 만족하도록 계획을 수정하고, 불가능하면 no_trade를 반환하세요."
                ),
            }
        )
    image_input = _build_image_input(frame, image_detail=image_detail)
    if image_input is not None:
        user_content.append(image_input)
    user_content.append(
        {
            "type": "input_text",
            "text": (
                "구조화된 컨텍스트\n"
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
