from __future__ import annotations

import json
from pathlib import Path

from dex_llm.features.extractor import FeatureExtractor
from dex_llm.llm.prompting import render_router_prompt
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

    assert '"heatmap_path": "heatmaps/sample-btc-20260311T150000Z.png"' in prompt
    assert '"kill_switch": {' in prompt
    assert '"allow_new_trades": false' in prompt
