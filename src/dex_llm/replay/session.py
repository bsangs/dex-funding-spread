from __future__ import annotations

from collections.abc import Sequence

from dex_llm.features.extractor import FeatureExtractor
from dex_llm.llm.router import RouterProtocol
from dex_llm.models import FeatureSnapshot, MarketFrame, TradePlan


class ReplaySession:
    def __init__(self, frames: Sequence[MarketFrame]) -> None:
        self.frames = list(frames)

    def route_all(
        self,
        extractor: FeatureExtractor,
        router: RouterProtocol,
    ) -> list[tuple[MarketFrame, FeatureSnapshot, TradePlan]]:
        outputs: list[tuple[MarketFrame, FeatureSnapshot, TradePlan]] = []
        for frame in self.frames:
            features = extractor.extract(frame)
            plan = router.route(frame, features)
            outputs.append((frame, features, plan))
        return outputs
