from __future__ import annotations

from pathlib import Path
from typing import Any

from openai import APIConnectionError, APIStatusError, APITimeoutError, OpenAI
from pydantic import BaseModel, Field

from dex_llm.llm.prompting import build_router_input, load_prompt_template
from dex_llm.models import FeatureSnapshot, MarketFrame, Playbook, TradePlan, TradeSide


class OpenAIRestingOrderPlan(BaseModel):
    side: TradeSide
    entry_band: list[float] = Field(min_length=2, max_length=2)
    invalid_if: float
    tp1: float
    tp2: float
    ttl_min: int
    reason: str
    cluster_price: float | None = None
    touch_confidence: float = Field(default=0.5, ge=0.0, le=1.0)
    expected_touch_minutes: int | None = Field(default=None, ge=1)


class OpenAITradePlan(BaseModel):
    playbook: Playbook
    side: TradeSide
    entry_band: list[float] = Field(min_length=2, max_length=2)
    invalid_if: float
    tp1: float
    tp2: float
    ttl_min: int
    reason: str
    touch_confidence: float = Field(default=0.5, ge=0.0, le=1.0)
    expected_touch_minutes: int | None = Field(default=None, ge=1)
    resting_orders: list[OpenAIRestingOrderPlan] = Field(default_factory=list)

    def to_trade_plan(self) -> TradePlan:
        return TradePlan(
            playbook=self.playbook,
            side=self.side,
            entry_band=(self.entry_band[0], self.entry_band[1]),
            invalid_if=self.invalid_if,
            tp1=self.tp1,
            tp2=self.tp2,
            ttl_min=self.ttl_min,
            reason=self.reason,
            touch_confidence=self.touch_confidence,
            expected_touch_minutes=self.expected_touch_minutes,
            resting_orders=[
                self._resting_order_to_domain(order) for order in self.resting_orders
            ],
        )

    @staticmethod
    def _resting_order_to_domain(order: OpenAIRestingOrderPlan):
        from dex_llm.models import RestingOrderPlan

        return RestingOrderPlan(
            side=order.side,
            entry_band=(order.entry_band[0], order.entry_band[1]),
            invalid_if=order.invalid_if,
            tp1=order.tp1,
            tp2=order.tp2,
            ttl_min=order.ttl_min,
            reason=order.reason,
            cluster_price=order.cluster_price,
            touch_confidence=order.touch_confidence,
            expected_touch_minutes=order.expected_touch_minutes,
        )


class OpenAIRouter:
    def __init__(
        self,
        *,
        model: str = "gpt-5.4",
        api_key: str | None = None,
        entry_prompt_path: Path | None = None,
        position_prompt_path: Path | None = None,
        entry_prompt_template: str | None = None,
        position_prompt_template: str | None = None,
        prompt_path: Path | None = None,
        prompt_template: str | None = None,
        timeout_s: float = 5.0,
        max_attempts: int = 2,
        max_output_tokens: int = 900,
        verbosity: str = "medium",
        reasoning_effort: str = "medium",
        reasoning_summary: str = "auto",
        store: bool = True,
        include: list[str] | None = None,
        image_detail: str = "auto",
        client: OpenAI | None = None,
    ) -> None:
        self.model = model
        self.timeout_s = timeout_s
        self.max_attempts = max_attempts
        self.max_output_tokens = max_output_tokens
        self.verbosity = verbosity
        self.reasoning_effort = reasoning_effort
        self.reasoning_summary = reasoning_summary
        self.store = store
        self.image_detail = image_detail
        self.include = include or [
            "reasoning.encrypted_content",
            "web_search_call.action.sources",
        ]
        self.client = client or OpenAI(api_key=api_key, max_retries=0)
        self.entry_prompt_template = entry_prompt_template or prompt_template
        self.position_prompt_template = position_prompt_template or self.entry_prompt_template
        self.entry_prompt_path = entry_prompt_path or prompt_path
        self.position_prompt_path = position_prompt_path or entry_prompt_path or prompt_path

    def route(
        self,
        frame: MarketFrame,
        features: FeatureSnapshot,
        previous_plan: TradePlan | None = None,
    ) -> TradePlan:
        if not frame.kill_switch.allow_new_trades and frame.position.side.value == "flat":
            reason = (
                frame.kill_switch.reasons[0]
                if frame.kill_switch.reasons
                else "kill switch active"
            )
            return self._flat_plan(reason)

        prompt = build_router_input(
            frame=frame,
            features=features,
            template=self._load_template(has_open_position=frame.position.side.value != "flat"),
            image_detail=self.image_detail,
            previous_plan=(
                previous_plan.model_dump(mode="json")
                if previous_plan is not None
                else None
            ),
        )
        last_error: Exception | None = None
        attempts = max(1, self.max_attempts)
        token_budget = max(self.max_output_tokens, 2_700)
        reasoning_effort = self._effective_reasoning_effort()
        for attempt_index in range(attempts):
            timeout_budget = max(self.timeout_s * (2 + attempt_index * 2), 60)
            try:
                response = self.client.responses.parse(
                    model=self.model,
                    input=prompt,
                    text_format=OpenAITradePlan,
                    text={"verbosity": self.verbosity},
                    timeout=timeout_budget,
                    max_output_tokens=token_budget,
                    reasoning={
                        "effort": reasoning_effort,
                        "summary": self.reasoning_summary,
                    },
                    store=self.store,
                    include=self.include,
                )
                plan = self._extract_trade_plan(response)
                if plan is None:
                    output_text = getattr(response, "output_text", "") or ""
                    if output_text:
                        return OpenAITradePlan.model_validate_json(output_text).to_trade_plan()
                    incomplete_details = getattr(response, "incomplete_details", None)
                    incomplete_reason = getattr(incomplete_details, "reason", None)
                    if incomplete_reason == "max_output_tokens" and attempt_index + 1 < attempts:
                        token_budget = max(token_budget * 2, 5_400)
                        last_error = RuntimeError(
                            "OpenAI response exhausted max_output_tokens before returning a plan"
                        )
                        continue
                    raise ValueError("OpenAI response did not include a parsed TradePlan")
                return plan.to_trade_plan()
            except (APITimeoutError, APIConnectionError, APIStatusError, ValueError) as exc:
                last_error = exc
        if last_error is None:
            raise RuntimeError("OpenAI router failed without a recorded exception")
        raise RuntimeError(f"OpenAI router failed: {last_error}") from last_error

    def _load_template(self, *, has_open_position: bool) -> str:
        if has_open_position:
            if self.position_prompt_template is not None:
                return self.position_prompt_template
            if self.position_prompt_path is None:
                raise ValueError(
                    "position_prompt_path or position_prompt_template is required for OpenAI router"
                )
            self.position_prompt_template = load_prompt_template(self.position_prompt_path)
            return self.position_prompt_template

        if self.entry_prompt_template is not None:
            return self.entry_prompt_template
        if self.entry_prompt_path is None:
            raise ValueError(
                "entry_prompt_path or entry_prompt_template is required for OpenAI router"
            )
        self.entry_prompt_template = load_prompt_template(self.entry_prompt_path)
        return self.entry_prompt_template

    @staticmethod
    def _extract_trade_plan(response: Any) -> OpenAITradePlan | None:
        parsed = getattr(response, "output_parsed", None)
        if isinstance(parsed, OpenAITradePlan):
            return parsed

        for item in getattr(response, "output", []) or []:
            for content in getattr(item, "content", []) or []:
                nested = getattr(content, "parsed", None)
                if isinstance(nested, OpenAITradePlan):
                    return nested
        return None

    def _effective_reasoning_effort(self) -> str:
        if self.reasoning_effort in {"high", "xhigh"}:
            return "low"
        return self.reasoning_effort

    @staticmethod
    def _flat_plan(reason: str) -> TradePlan:
        return TradePlan(
            playbook=Playbook.NO_TRADE,
            side=TradeSide.FLAT,
            entry_band=(0.0, 0.0),
            invalid_if=0.0,
            tp1=0.0,
            tp2=0.0,
            ttl_min=0,
            reason=reason,
            touch_confidence=0.0,
            expected_touch_minutes=None,
        )
