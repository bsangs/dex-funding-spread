from __future__ import annotations

import json
from pathlib import Path

from dex_llm.features.extractor import FeatureExtractor
from dex_llm.llm.prompting import build_router_input, render_router_prompt
from dex_llm.models import MarketFrame


def load_sample_frame() -> MarketFrame:
    return MarketFrame.model_validate(json.loads(Path("examples/sample_frame.json").read_text()))


def test_router_prompt_includes_heatmap_path_and_kill_switch() -> None:
    frame = load_sample_frame()
    frame.kill_switch.allow_new_trades = False
    frame.kill_switch.reasons = ["synthetic heatmap fallback active"]

    prompt = render_router_prompt(
        frame=frame,
        features=FeatureExtractor().extract(frame),
        template="# Router",
    )

    assert '"heatmap_path": "heatmaps/sample-eth-20260311T150000Z.png"' in prompt
    assert '"heatmap_image_path": "heatmaps/sample-eth-20260311T150000Z.png"' in prompt
    assert '"kill_switch": {' in prompt
    assert '"allow_new_trades": false' in prompt


def test_router_input_prefers_local_heatmap_image(tmp_path: Path) -> None:
    frame = load_sample_frame()
    local_image = tmp_path / "heatmap.png"
    local_image.write_bytes(b"\x89PNG\r\n\x1a\nfake")
    frame.heatmap_image_path = str(local_image)
    frame.heatmap_path = str(local_image)

    payload = build_router_input(
        frame=frame,
        features=FeatureExtractor().extract(frame),
        template="# Router",
    )

    content = payload[1]["content"]
    assert isinstance(content, list)
    assert content[1]["type"] == "input_image"
    assert str(content[1]["image_url"]).startswith("data:image/png;base64,")
    assert "지정가 구간" in str(content[0]["text"])
    structured_context = str(content[2]["text"])
    assert '"current_price"' not in structured_context
    assert '"position"' not in structured_context
    assert '"entry_candidates"' in structured_context
    assert '"candles_1h"' in structured_context
    assert '"candles_4h"' in structured_context
    assert '"heatmap_positions"' in structured_context
    assert '"heatmap_levels_above_detailed"' in structured_context


def test_router_input_falls_back_to_remote_heatmap_url() -> None:
    frame = load_sample_frame()
    frame.heatmap_image_path = None
    frame.heatmap_image_url = "https://example.com/eth-heatmap.png"
    frame.heatmap_path = frame.heatmap_image_url

    payload = build_router_input(
        frame=frame,
        features=FeatureExtractor().extract(frame),
        template="# Router",
    )

    content = payload[1]["content"]
    assert isinstance(content, list)
    assert content[1]["type"] == "input_image"
    assert content[1]["image_url"] == "https://example.com/eth-heatmap.png"
