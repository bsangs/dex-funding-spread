from __future__ import annotations

from pathlib import Path

from openai import APIConnectionError, APIStatusError, APITimeoutError, OpenAI

from dex_llm.llm.prompting import build_router_input, load_prompt_template
from dex_llm.llm.router import HeuristicPlaybookRouter, RouterProtocol
from dex_llm.models import FeatureSnapshot, MarketFrame, TradePlan


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
        verbosity: str = "medium",
        reasoning_effort: str = "medium",
        reasoning_summary: str = "auto",
        store: bool = True,
        include: list[str] | None = None,
        image_detail: str = "auto",
        client: OpenAI | None = None,
        fallback_router: RouterProtocol | None = None,
    ) -> None:
        self.model = model
        self.timeout_s = timeout_s
        self.max_attempts = max_attempts
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
        self.fallback_router = fallback_router or HeuristicPlaybookRouter()

    def route(
        self,
        frame: MarketFrame,
        features: FeatureSnapshot,
        previous_plan: TradePlan | None = None,
    ) -> TradePlan:
        if not frame.kill_switch.allow_new_trades and frame.position.side.value == "flat":
            return self.fallback_router.route(frame, features, previous_plan=previous_plan)

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
        for _ in range(attempts):
            try:
                response = self.client.responses.parse(
                    model=self.model,
                    input=prompt,
                    text_format=TradePlan,
                    timeout=self.timeout_s,
                    temperature=0,
                    max_output_tokens=300,
                    verbosity=self.verbosity,
                    reasoning={
                        "effort": self.reasoning_effort,
                        "summary": self.reasoning_summary,
                    },
                    store=self.store,
                    include=self.include,
                )
                plan = response.output_parsed
                if plan is None:
                    raise ValueError("OpenAI response did not include a parsed TradePlan")
                return plan
            except (APITimeoutError, APIConnectionError, APIStatusError, ValueError) as exc:
                last_error = exc
        if self.fallback_router is not None:
            return self.fallback_router.route(frame, features, previous_plan=previous_plan)
        raise RuntimeError("OpenAI router failed without fallback") from last_error

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
