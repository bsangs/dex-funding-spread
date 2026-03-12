from __future__ import annotations

from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

from dex_llm.models import ExecutionMode, MarginMode


class AppSettings(BaseSettings):
    dex_name: str = "hyperliquid"
    symbol: str = "BTC"
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
    prompt_path: Path = Path("prompts/playbook_router.md")
    raw_data_dir: Path = Path("data/raw")
    replay_dir: Path = Path("data/replays")
    heatmap_cache_dir: Path = Path("data/heatmaps")
    request_timeout_s: float = 10.0
    kill_switch_max_info_latency_ms: float = 1_500.0
    kill_switch_max_private_latency_ms: float = 1_500.0
    kill_switch_max_data_age_ms: float = 15_000.0
    kill_switch_max_consecutive_losses: int = 2
    risk_per_trade_pct: float = 0.35
    execution_mode: ExecutionMode = ExecutionMode.PAPER
    margin_mode: MarginMode = MarginMode.ISOLATED
    target_leverage: int = 10
    base_leverage: float = 6.0
    max_leverage: float = 10.0
    max_price_deviation_bps: float = 500.0
    clock_drift_limit_ms: float = 500.0

    model_config = SettingsConfigDict(
        env_prefix="DEX_LLM_",
        env_file=".env.local",
        env_file_encoding="utf-8",
        extra="ignore",
    )
