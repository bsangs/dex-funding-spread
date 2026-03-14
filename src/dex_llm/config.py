from __future__ import annotations

from pathlib import Path

from pydantic import field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

from dex_llm.models import ExecutionMode, MarginMode


class AppSettings(BaseSettings):
    dex_name: str = "hyperliquid"
    symbol: str = "ETH"
    heatmap_source: str = "external"
    llm_provider: str = "openai"
    llm_model: str = "gpt-5.4"
    openai_api_key: str | None = None
    llm_timeout_s: float = 5.0
    openai_verbosity: str = "medium"
    openai_reasoning_effort: str = "medium"
    openai_reasoning_summary: str = "auto"
    openai_store: bool = True
    openai_include: str = "reasoning.encrypted_content,web_search_call.action.sources"
    hyperliquid_base_url: str = "https://api.hyperliquid.xyz"
    hyperliquid_ws_url: str = "wss://api.hyperliquid.xyz/ws"
    trading_user_address: str | None = None
    hyperliquid_user_address: str | None = None
    signer_agent_address: str | None = None
    signer_private_key: str | None = None
    hyperliquid_vault_address: str | None = None
    hyperliquid_dex: str = ""
    coinglass_api_key: str | None = None
    coinglass_base_url: str = "https://open-api-v4.coinglass.com"
    coinglass_heatmap_path: str = "/api/futures/liquidation/aggregated-heatmap/model1"
    coinglass_web_url: str = "https://www.coinglass.com/ko/liquidations"
    coinglass_use_playwright_fallback: bool = True
    entry_prompt_path: Path = Path("prompts/entry_router_ko.md")
    position_prompt_path: Path = Path("prompts/position_router_ko.md")
    raw_data_dir: Path = Path("data/raw")
    replay_dir: Path = Path("data/replays")
    heatmap_cache_dir: Path = Path("data/heatmaps")
    request_timeout_s: float = 10.0
    coinglass_scrape_timeout_s: float = 20.0
    bot_strategy_interval_s: int = 900
    bot_sync_interval_s: int = 120
    kill_switch_max_info_latency_ms: float = 1_500.0
    kill_switch_max_private_latency_ms: float = 1_500.0
    long_notional_fraction: float = 0.9
    short_notional_fraction: float = 0.4
    enable_stop_loss: bool = False
    execution_mode: ExecutionMode = ExecutionMode.PAPER
    margin_mode: MarginMode = MarginMode.ISOLATED
    long_target_leverage: int = 20
    short_target_leverage: int = 15
    base_leverage: float = 6.0
    max_leverage: float = 20.0
    openai_image_detail: str = "auto"
    clock_drift_limit_ms: float = 500.0

    @field_validator(
        "openai_api_key",
        "trading_user_address",
        "hyperliquid_user_address",
        "signer_agent_address",
        "signer_private_key",
        "hyperliquid_vault_address",
        "coinglass_api_key",
        mode="before",
    )
    @classmethod
    def empty_string_to_none(cls, value: str | None) -> str | None:
        if isinstance(value, str) and not value.strip():
            return None
        return value

    model_config = SettingsConfigDict(
        env_prefix="DEX_LLM_",
        env_file=".env.local",
        env_file_encoding="utf-8",
        extra="ignore",
    )
