from __future__ import annotations

import json
from pathlib import Path

from dex_llm.collector.storage import JsonlFrameStore
from dex_llm.models import MarketFrame
from dex_llm.replay.session import ReplaySession


def load_sample_frame() -> MarketFrame:
    return MarketFrame.model_validate(json.loads(Path("examples/sample_frame.json").read_text()))


def test_jsonl_store_round_trip(tmp_path: Path) -> None:
    store = JsonlFrameStore(tmp_path / "frames.jsonl")
    frame = load_sample_frame()

    store.append(frame)
    frames = store.read_all()

    assert len(frames) == 1
    assert frames[0].symbol == frame.symbol
    assert len(ReplaySession(frames).frames) == 1

