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
        "position": _position_payload(frame),
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
                "TradePlan 하나만 반환하세요. 신규 진입은 시장가가 아니라 곧 체결될 가능성이 "
                "높은 지정가 구간만 사용하세요."
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


def _position_payload(frame: MarketFrame) -> dict[str, object]:
    position = frame.position
    active_entry_orders = [
        {
            "price": order.limit_price,
            "side": order.side,
            "status": order.status.value,
        }
        for order in position.active_orders
        if order.coin == frame.symbol and not order.reduce_only
    ]
    active_exit_orders = [
        {
            "role": order.role.value,
            "price": order.trigger_price or order.limit_price,
            "status": order.status.value,
        }
        for order in position.active_orders
        if order.coin == frame.symbol and order.reduce_only
    ]
    payload: dict[str, object] = {
        "side": position.side.value,
        "has_position": position.side.value != "flat",
        "entry_price": position.entry_price,
        "open_orders": position.open_orders,
        "entries_blocked_reduce_only": position.entries_blocked_reduce_only,
        "active_entry_orders": active_entry_orders,
        "active_exit_orders": active_exit_orders,
    }
    if position.last_user_event is not None:
        payload["last_user_event"] = {
            "event_type": position.last_user_event.event_type.value,
            "reason": position.last_user_event.reason,
            "timestamp": position.last_user_event.timestamp.isoformat()
            if position.last_user_event.timestamp is not None
            else None,
        }
    return payload
