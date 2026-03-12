from __future__ import annotations

from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class AppSettings(BaseSettings):
    dex_name: str = "hyperliquid"
    symbol: str = "BTC"
    heatmap_source: str = "external"
    llm_provider: str = "openai"
    llm_model: str = "gpt-5-mini"
    hyperliquid_base_url: str = "https://api.hyperliquid.xyz"
    coinglass_api_key: str | None = None
    coinglass_base_url: str = "https://open-api-v4.coinglass.com"
    coinglass_heatmap_path: str = "/api/futures/liquidation/aggregated-heatmap/model1"
    prompt_path: Path = Path("prompts/playbook_router.md")
    raw_data_dir: Path = Path("data/raw")
    replay_dir: Path = Path("data/replays")
    heatmap_cache_dir: Path = Path("data/heatmaps")
    request_timeout_s: float = 10.0
    risk_per_trade_pct: float = 0.35
    base_leverage: float = 6.0
    max_leverage: float = 10.0

    model_config = SettingsConfigDict(
        env_prefix="DEX_LLM_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )
