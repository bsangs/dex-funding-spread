from __future__ import annotations

import json
from pathlib import Path

from openai import APITimeoutError

from dex_llm.features.extractor import FeatureExtractor
from dex_llm.llm.openai_router import OpenAIRouter
from dex_llm.llm.router import HeuristicPlaybookRouter
from dex_llm.models import MarketFrame, Playbook, TradePlan, TradeSide


class FakeParsedResponse:
    def __init__(self, plan: TradePlan | None) -> None:
        self.output_parsed = plan


class FakeResponses:
    def __init__(self, results: list[object]) -> None:
        self._results = results
        self.calls = 0
        self.last_kwargs: dict[str, object] | None = None

    def parse(self, **kwargs: object) -> FakeParsedResponse:
        self.last_kwargs = kwargs
        result = self._results[self.calls]
        self.calls += 1
        if isinstance(result, Exception):
            raise result
        return FakeParsedResponse(result)


class FakeOpenAIClient:
    def __init__(self, results: list[object]) -> None:
        self.responses = FakeResponses(results)


def load_sample_frame() -> MarketFrame:
    return MarketFrame.model_validate(json.loads(Path("examples/sample_frame.json").read_text()))


def test_openai_router_returns_parsed_trade_plan() -> None:
    frame = load_sample_frame()
    features = FeatureExtractor().extract(frame)
    plan = TradePlan(
        playbook=Playbook.MAGNET_FOLLOW,
        side=TradeSide.LONG,
        entry_band=(70000.0, 70010.0),
        invalid_if=69900.0,
        tp1=70100.0,
        tp2=70200.0,
        ttl_min=12,
        reason="dominant upside liquidity",
    )
    router = OpenAIRouter(
        client=FakeOpenAIClient([plan]),
        prompt_template="# Router",
    )

    routed = router.route(frame, features)

    assert routed == plan
    assert router.client.responses.last_kwargs is not None
    assert router.client.responses.last_kwargs["model"] == "gpt-5.4"
    assert router.client.responses.last_kwargs["verbosity"] == "medium"
    assert router.client.responses.last_kwargs["store"] is True
    assert router.client.responses.last_kwargs["reasoning"] == {
        "effort": "medium",
        "summary": "auto",
    }
    assert router.client.responses.last_kwargs["include"] == [
        "reasoning.encrypted_content",
        "web_search_call.action.sources",
    ]


def test_openai_router_falls_back_on_timeout() -> None:
    frame = load_sample_frame()
    features = FeatureExtractor().extract(frame)
    fallback = HeuristicPlaybookRouter()
    router = OpenAIRouter(
        client=FakeOpenAIClient([APITimeoutError(request=None)] * 2),
        prompt_template="# Router",
        fallback_router=fallback,
    )

    routed = router.route(frame, features)

    assert routed.playbook == fallback.route(frame, features).playbook


def test_openai_router_skips_call_when_kill_switch_is_active() -> None:
    frame = load_sample_frame()
    frame.kill_switch.allow_new_trades = False
    frame.kill_switch.reasons = ["private account state unavailable"]
    features = FeatureExtractor().extract(frame)
    client = FakeOpenAIClient([])
    router = OpenAIRouter(
        client=client,
        prompt_template="# Router",
    )

    routed = router.route(frame, features)

    assert routed.playbook == Playbook.NO_TRADE
    assert client.responses.calls == 0


def test_openai_router_accepts_custom_response_settings() -> None:
    frame = load_sample_frame()
    features = FeatureExtractor().extract(frame)
    plan = TradePlan(
        playbook=Playbook.MAGNET_FOLLOW,
        side=TradeSide.LONG,
        entry_band=(70000.0, 70010.0),
        invalid_if=69900.0,
        tp1=70100.0,
        tp2=70200.0,
        ttl_min=12,
        reason="dominant upside liquidity",
    )
    client = FakeOpenAIClient([plan])
    router = OpenAIRouter(
        client=client,
        prompt_template="# Router",
        verbosity="high",
        reasoning_effort="high",
        reasoning_summary="detailed",
        store=False,
        include=["reasoning.encrypted_content"],
    )

    routed = router.route(frame, features)

    assert routed == plan
    assert client.responses.last_kwargs is not None
    assert client.responses.last_kwargs["verbosity"] == "high"
    assert client.responses.last_kwargs["reasoning"] == {
        "effort": "high",
        "summary": "detailed",
    }
    assert client.responses.last_kwargs["store"] is False
    assert client.responses.last_kwargs["include"] == ["reasoning.encrypted_content"]
