from __future__ import annotations

import json
from pathlib import Path

from dex_llm.features.extractor import FeatureExtractor
from dex_llm.models import ClusterSide, MarketFrame


def load_sample_frame() -> MarketFrame:
    return MarketFrame.model_validate(json.loads(Path("examples/sample_frame.json").read_text()))


def test_feature_extractor_marks_magnet_follow_context() -> None:
    frame = load_sample_frame()
    features = FeatureExtractor().extract(frame)

    assert features.dominant_cluster_side == ClusterSide.ABOVE
    assert features.directional_vacuum is True
    assert features.sweep_reclaim_ready is False

