from __future__ import annotations

import json
from pathlib import Path

import pytest
from openai import APITimeoutError

from dex_llm.features.extractor import FeatureExtractor
from dex_llm.llm.openai_router import OpenAIRouter, OpenAITradePlan
from dex_llm.models import MarketFrame, Playbook, TradePlan, TradeSide


class FakeParsedResponse:
    def __init__(self, plan: OpenAITradePlan | None) -> None:
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
        client=FakeOpenAIClient([_wire_plan(plan)]),
        prompt_template="# Router",
    )

    routed = router.route(frame, features)

    assert routed == plan
    assert router.client.responses.last_kwargs is not None
    assert router.client.responses.last_kwargs["model"] == "gpt-5.4"
    assert router.client.responses.last_kwargs["text"] == {"verbosity": "medium"}
    assert router.client.responses.last_kwargs["store"] is True
    assert router.client.responses.last_kwargs["reasoning"] == {
        "effort": "medium",
        "summary": "auto",
    }
    assert router.client.responses.last_kwargs["include"] == [
        "reasoning.encrypted_content",
        "web_search_call.action.sources",
    ]
    assert isinstance(router.client.responses.last_kwargs["input"], list)


def test_openai_router_extracts_nested_parsed_trade_plan() -> None:
    plan = _wire_plan(
        TradePlan(
            playbook=Playbook.NO_TRADE,
            side=TradeSide.FLAT,
            entry_band=(0.0, 0.0),
            invalid_if=0.0,
            tp1=0.0,
            tp2=0.0,
            ttl_min=15,
            reason="hold",
        )
    )

    class NestedContent:
        def __init__(self, parsed: OpenAITradePlan) -> None:
            self.parsed = parsed

    class NestedMessage:
        def __init__(self, parsed: OpenAITradePlan) -> None:
            self.content = [NestedContent(parsed)]

    class NestedResponse:
        output_parsed = None

        def __init__(self, parsed: OpenAITradePlan) -> None:
            self.output = [NestedMessage(parsed)]

    extracted = OpenAIRouter._extract_trade_plan(NestedResponse(plan))

    assert extracted == plan


def test_openai_router_sends_multimodal_input_when_local_heatmap_exists(tmp_path: Path) -> None:
    frame = load_sample_frame()
    local_image = tmp_path / "heatmap.png"
    local_image.write_bytes(b"\x89PNG\r\n\x1a\nfake")
    frame.heatmap_image_path = str(local_image)
    frame.heatmap_path = str(local_image)
    features = FeatureExtractor().extract(frame)
    plan = TradePlan(
        playbook=Playbook.MAGNET_FOLLOW,
        side=TradeSide.LONG,
        entry_band=(4000.0, 4002.0),
        invalid_if=3970.0,
        tp1=4025.0,
        tp2=4040.0,
        ttl_min=12,
        reason="dominant upside liquidity",
    )
    client = FakeOpenAIClient([plan])
    router = OpenAIRouter(
        client=FakeOpenAIClient([_wire_plan(plan)]),
        prompt_template="# Router",
        image_detail="high",
    )

    routed = router.route(frame, features)

    assert routed == plan
    assert router.client.responses.last_kwargs is not None
    user_content = router.client.responses.last_kwargs["input"][1]["content"]
    assert user_content[1]["type"] == "input_image"
    assert user_content[1]["detail"] == "high"
    assert str(user_content[1]["image_url"]).startswith("data:image/png;base64,")


def test_openai_router_raises_on_timeout() -> None:
    frame = load_sample_frame()
    features = FeatureExtractor().extract(frame)
    router = OpenAIRouter(
        client=FakeOpenAIClient([APITimeoutError(request=None)] * 2),
        prompt_template="# Router",
    )

    with pytest.raises(RuntimeError, match="OpenAI router failed"):
        router.route(frame, features)


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
        client=FakeOpenAIClient([_wire_plan(plan)]),
        prompt_template="# Router",
        verbosity="high",
        reasoning_effort="high",
        reasoning_summary="detailed",
        store=False,
        include=["reasoning.encrypted_content"],
    )

    routed = router.route(frame, features)

    assert routed == plan
    assert router.client.responses.last_kwargs is not None
    assert router.client.responses.last_kwargs["text"] == {"verbosity": "high"}
    assert router.client.responses.last_kwargs["reasoning"] == {
        "effort": "high",
        "summary": "detailed",
    }
    assert router.client.responses.last_kwargs["store"] is False
    assert router.client.responses.last_kwargs["include"] == ["reasoning.encrypted_content"]


def _wire_plan(plan: TradePlan) -> OpenAITradePlan:
    return OpenAITradePlan(
        playbook=plan.playbook,
        side=plan.side,
        entry_band=list(plan.entry_band),
        invalid_if=plan.invalid_if,
        tp1=plan.tp1,
        tp2=plan.tp2,
        ttl_min=plan.ttl_min,
        reason=plan.reason,
        touch_confidence=plan.touch_confidence,
        expected_touch_minutes=plan.expected_touch_minutes,
        resting_orders=[],
    )
