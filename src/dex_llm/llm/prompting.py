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
    levels_above = heatmap_metadata.get("levels_above")
    levels_below = heatmap_metadata.get("levels_below")
    positions = heatmap_metadata.get("positions")
    return {
        "exchange": frame.exchange,
        "symbol": frame.symbol,
        "current_price": frame.current_price,
        "position_side": frame.position.side.value,
        "open_orders": frame.position.open_orders,
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
        "heatmap_prompt_window_abs": round(
            _heatmap_prompt_window(current_price=frame.current_price, atr=frame.atr),
            2,
        ),
        "heatmap_levels_above_count": len(levels_above) if isinstance(levels_above, list) else 0,
        "heatmap_levels_below_count": len(levels_below) if isinstance(levels_below, list) else 0,
        "heatmap_positions_count": len(positions) if isinstance(positions, list) else 0,
        "heatmap_levels_above_detailed": _filter_heatmap_levels(
            levels_above,
            current_price=frame.current_price,
            atr=frame.atr,
            side="above",
        ),
        "heatmap_levels_below_detailed": _filter_heatmap_levels(
            levels_below,
            current_price=frame.current_price,
            atr=frame.atr,
            side="below",
        ),
        "heatmap_positions": _filter_heatmap_positions(
            positions,
            current_price=frame.current_price,
            atr=frame.atr,
        ),
        "higher_timeframe_levels": frame.metadata.get("higher_timeframe_levels"),
        "active_orders": [
            _active_order_payload(order, current_price=frame.current_price)
            for order in frame.position.active_orders
            if order.coin == frame.symbol
        ],
        "has_active_entry_workflow": any(
            order.coin == frame.symbol and not order.reduce_only
            for order in frame.position.active_orders
        ),
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


def _heatmap_prompt_window(*, current_price: float, atr: float) -> float:
    return max(atr * 8.0, current_price * 0.03, 25.0)


def _filter_heatmap_levels(
    levels: object,
    *,
    current_price: float,
    atr: float,
    side: str,
    near_limit: int = 12,
    strong_limit: int = 8,
) -> list[dict[str, object]] | None:
    if not isinstance(levels, list):
        return None
    window = _heatmap_prompt_window(current_price=current_price, atr=atr)
    normalized: list[tuple[float, float, dict[str, object]]] = []
    for item in levels:
        if not isinstance(item, dict):
            continue
        price = _as_float(item.get("price"))
        if price is None:
            continue
        size = _as_float(item.get("size")) or 0.0
        normalized.append((price, size, dict(item)))

    near_items = [
        entry for entry in normalized if abs(entry[0] - current_price) <= window
    ]
    near_items.sort(key=lambda entry: (abs(entry[0] - current_price), -entry[1]))

    strong_items = sorted(
        normalized,
        key=lambda entry: (-entry[1], abs(entry[0] - current_price)),
    )

    selected: dict[float, dict[str, object]] = {}
    for price, _, item in near_items[:near_limit]:
        selected[price] = item
    for price, _, item in strong_items[:strong_limit]:
        selected.setdefault(price, item)

    filtered = list(selected.values())
    filtered.sort(
        key=lambda item: _level_sort_key(
            item=item,
            current_price=current_price,
            side=side,
        )
    )
    return filtered


def _filter_heatmap_positions(
    positions: object,
    *,
    current_price: float,
    atr: float,
    near_limit: int = 18,
    strong_per_side: int = 8,
    max_total: int = 28,
) -> list[dict[str, object]] | None:
    if not isinstance(positions, list):
        return None
    window = _heatmap_prompt_window(current_price=current_price, atr=atr)
    normalized: list[tuple[float, float, dict[str, object]]] = []
    for item in positions:
        if not isinstance(item, dict):
            continue
        price = _as_float(item.get("liquidation_price"))
        if price is None:
            continue
        weight = abs(_as_float(item.get("position_usd")) or 0.0)
        normalized.append((price, weight, dict(item)))

    near_items = [
        entry for entry in normalized if abs(entry[0] - current_price) <= window
    ]
    near_items.sort(key=lambda entry: (abs(entry[0] - current_price), -entry[1]))

    above = [entry for entry in normalized if entry[0] >= current_price]
    below = [entry for entry in normalized if entry[0] < current_price]
    above.sort(key=lambda entry: (-entry[1], abs(entry[0] - current_price)))
    below.sort(key=lambda entry: (-entry[1], abs(entry[0] - current_price)))

    selected: dict[tuple[float, float], dict[str, object]] = {}
    for price, weight, item in near_items[:near_limit]:
        selected[(price, weight)] = item
    for bucket in (above[:strong_per_side], below[:strong_per_side]):
        for price, weight, item in bucket:
            selected.setdefault((price, weight), item)

    filtered = list(selected.values())
    filtered.sort(
        key=lambda item: (
            abs((_as_float(item.get("liquidation_price")) or current_price) - current_price),
            -abs(_as_float(item.get("position_usd")) or 0.0),
        )
    )
    return filtered[:max_total]


def _level_sort_key(
    *,
    item: dict[str, object],
    current_price: float,
    side: str,
) -> tuple[float, float]:
    price = _as_float(item.get("price")) or current_price
    if side == "below":
        return (-price, abs(price - current_price))
    return (price, abs(price - current_price))


def _as_float(value: object) -> float | None:
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str) and value:
        return float(value)
    return None


def _active_order_payload(order, *, current_price: float) -> dict[str, object]:
    reference_price = order.trigger_price or order.limit_price
    distance_bps = None
    if reference_price and current_price > 0:
        distance_bps = abs(reference_price - current_price) / current_price * 10_000
    return {
        "role": order.role.value,
        "side": order.side,
        "limit_price": order.limit_price,
        "trigger_price": order.trigger_price,
        "size": order.size,
        "reduce_only": order.reduce_only,
        "status": order.status.value,
        "timestamp": order.timestamp.isoformat() if order.timestamp is not None else None,
        "distance_bps_from_current": round(distance_bps, 1) if distance_bps is not None else None,
    }


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
                "현재가에 바로 체결되는지보다 다음 10분~6시간 가격 경로를 먼저 가정하고, 그 경로 안에서 "
                "도달 가능성이 높은 지정가 구간을 고르세요. 현재가 주변에 즉시 체결될 자리가 없다는 "
                "이유만으로 no_trade를 반환하지 마세요. 다만 현재가에서 지나치게 먼 가격은 특별히 강한 "
                "구조적 경로를 설명할 수 있을 때만 제시하고, 그렇지 않으면 no_trade를 반환하세요. "
                "현재 미체결 주문이 있으면 그 주문을 먼저 인식하고 keep 할지 replace 할지 매우 보수적으로 판단하세요. "
                "분명한 구조 개선이나 무효화 근거가 없으면 기존 미체결 주문을 유지하세요. cancel-only 의도는 만들지 마세요."
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
