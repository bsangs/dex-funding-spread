from __future__ import annotations

import json
from pathlib import Path

import typer
from rich.console import Console

from dex_llm.collector.storage import JsonlFrameStore
from dex_llm.config import AppSettings
from dex_llm.executor.paper import PaperExecutor
from dex_llm.features.extractor import FeatureExtractor
from dex_llm.integrations.coinglass import CoinGlassHeatmapClient
from dex_llm.integrations.hyperliquid import HyperliquidInfoClient
from dex_llm.live_frame import LiveFrameBuilder
from dex_llm.llm.prompting import load_prompt_template, render_router_prompt
from dex_llm.llm.router import HeuristicPlaybookRouter
from dex_llm.models import AccountState, MarketFrame
from dex_llm.replay.session import ReplaySession
from dex_llm.risk.kill_switch import KillSwitchPolicy
from dex_llm.risk.policy import RiskPolicy

app = typer.Typer(no_args_is_help=True, add_completion=False)
console = Console()
DEFAULT_FRAME_ARGUMENT = typer.Argument(Path("examples/sample_frame.json"))
DEFAULT_RECORD_PATH_OPTION = typer.Option(Path("data/raw/sample-session.jsonl"))
DEFAULT_REPLAY_ARGUMENT = typer.Argument(Path("data/raw/sample-session.jsonl"))
DEFAULT_SYMBOL_ARGUMENT = typer.Argument("BTC")
DEFAULT_LIVE_OUT_OPTION = typer.Option(Path("data/raw/live-session.jsonl"))
HEATMAP_PARAM_OPTION = typer.Option(
    None,
    "--heatmap-param",
    help="Extra CoinGlass query parameter in key=value format. Repeatable.",
)
ALLOW_SYNTHETIC_OPTION = typer.Option(
    False,
    help="Fallback to synthetic order-book clusters when CoinGlass is unavailable.",
)
NO_WRITE_OPTION = typer.Option(False, help="Print the frame without appending it to JSONL.")
USER_ADDRESS_OPTION = typer.Option(
    None,
    "--user-address",
    help="Hyperliquid account address used for private state and kill-switch checks.",
)


def _load_frame(path: Path) -> MarketFrame:
    return MarketFrame.model_validate_json(path.read_text(encoding="utf-8"))


def _parse_key_value_params(values: list[str] | None) -> dict[str, str]:
    params: dict[str, str] = {}
    for value in values or []:
        if "=" not in value:
            raise typer.BadParameter(f"Expected key=value, got {value!r}")
        key, raw = value.split("=", 1)
        params[key] = raw
    return params


@app.command()
def inspect(
    frame_path: Path = DEFAULT_FRAME_ARGUMENT,
    equity: float = typer.Option(10_000.0, help="Account equity used for paper sizing."),
) -> None:
    settings = AppSettings()
    frame = _load_frame(frame_path)
    extractor = FeatureExtractor()
    router = HeuristicPlaybookRouter()
    risk_policy = RiskPolicy(
        risk_per_trade_pct=settings.risk_per_trade_pct,
        max_consecutive_losses=settings.kill_switch_max_consecutive_losses,
    )
    account = AccountState(
        equity=equity,
        available_margin=equity,
        max_leverage=settings.max_leverage,
    )
    features = extractor.extract(frame)
    plan = router.route(frame, features)
    risk = risk_policy.assess(plan, account, frame.position, frame.kill_switch)
    payload: dict[str, object] = {
        "features": features.model_dump(mode="json"),
        "plan": plan.model_dump(mode="json"),
        "risk": risk.model_dump(mode="json"),
    }
    if risk.allowed:
        ticket = PaperExecutor().build_ticket(plan, risk, account)
        payload["paper_ticket"] = ticket.model_dump(mode="json")
    console.print_json(json.dumps(payload))


@app.command()
def prompt(
    frame_path: Path = DEFAULT_FRAME_ARGUMENT,
) -> None:
    settings = AppSettings()
    frame = _load_frame(frame_path)
    features = FeatureExtractor().extract(frame)
    prompt_text = render_router_prompt(
        frame=frame,
        features=features,
        template=load_prompt_template(settings.prompt_path),
    )
    console.print(prompt_text)


@app.command()
def record_sample(
    frame_path: Path = DEFAULT_FRAME_ARGUMENT,
    out_path: Path = DEFAULT_RECORD_PATH_OPTION,
) -> None:
    frame = _load_frame(frame_path)
    JsonlFrameStore(out_path).append(frame)
    console.print(f"recorded frame to {out_path}")


@app.command()
def replay(
    source: Path = DEFAULT_REPLAY_ARGUMENT,
) -> None:
    session = ReplaySession(JsonlFrameStore(source).read_all())
    outputs = session.route_all(FeatureExtractor(), HeuristicPlaybookRouter())
    console.print_json(
        json.dumps(
            [
                {
                    "timestamp": frame.timestamp.isoformat(),
                    "symbol": frame.symbol,
                    "plan": plan.model_dump(mode="json"),
                    "notes": features.notes,
                }
                for frame, features, plan in outputs
            ]
        )
    )


@app.command("hyperliquid-snapshot")
def hyperliquid_snapshot(symbol: str = DEFAULT_SYMBOL_ARGUMENT) -> None:
    settings = AppSettings()
    client = HyperliquidInfoClient(
        base_url=settings.hyperliquid_base_url,
        timeout_s=settings.request_timeout_s,
    )
    try:
        book = client.fetch_l2_book(symbol)
        candles_5m = client.fetch_candles(symbol, "5m", limit=5)
        candles_15m = client.fetch_candles(symbol, "15m", limit=5)
        console.print_json(
            json.dumps(
                {
                    "symbol": symbol,
                    "best_bid": book.best_bid,
                    "best_ask": book.best_ask,
                    "mid_price": book.mid_price,
                    "bids": [level.model_dump(mode="json") for level in book.bids[:5]],
                    "asks": [level.model_dump(mode="json") for level in book.asks[:5]],
                    "candles_5m": [candle.model_dump(mode="json") for candle in candles_5m],
                    "candles_15m": [candle.model_dump(mode="json") for candle in candles_15m],
                }
            )
        )
    finally:
        client.close()


@app.command("coinglass-preview")
def coinglass_preview(
    symbol: str = DEFAULT_SYMBOL_ARGUMENT,
    heatmap_param: list[str] | None = HEATMAP_PARAM_OPTION,
) -> None:
    settings = AppSettings()
    if not settings.coinglass_api_key:
        raise typer.BadParameter("DEX_LLM_COINGLASS_API_KEY is required for CoinGlass preview")

    client = CoinGlassHeatmapClient(
        api_key=settings.coinglass_api_key,
        base_url=settings.coinglass_base_url,
        heatmap_path=settings.coinglass_heatmap_path,
        timeout_s=settings.request_timeout_s,
        cache_dir=settings.heatmap_cache_dir,
    )
    try:
        snapshot = client.fetch_heatmap(symbol, extra_params=_parse_key_value_params(heatmap_param))
        console.print_json(json.dumps(snapshot.model_dump(mode="json")))
    finally:
        client.close()


@app.command("live-frame")
def live_frame(
    symbol: str = DEFAULT_SYMBOL_ARGUMENT,
    out_path: Path = DEFAULT_LIVE_OUT_OPTION,
    heatmap_param: list[str] | None = HEATMAP_PARAM_OPTION,
    allow_synthetic: bool = ALLOW_SYNTHETIC_OPTION,
    no_write: bool = NO_WRITE_OPTION,
    user_address: str | None = USER_ADDRESS_OPTION,
) -> None:
    settings = AppSettings()
    hyperliquid_client = HyperliquidInfoClient(
        base_url=settings.hyperliquid_base_url,
        timeout_s=settings.request_timeout_s,
    )
    coinglass_client: CoinGlassHeatmapClient | None = None
    if settings.coinglass_api_key:
        coinglass_client = CoinGlassHeatmapClient(
            api_key=settings.coinglass_api_key,
            base_url=settings.coinglass_base_url,
            heatmap_path=settings.coinglass_heatmap_path,
            timeout_s=settings.request_timeout_s,
            cache_dir=settings.heatmap_cache_dir,
        )

    try:
        builder = LiveFrameBuilder(
            hyperliquid_client,
            coinglass_client,
            kill_switch_policy=KillSwitchPolicy(
                max_info_latency_ms=settings.kill_switch_max_info_latency_ms,
                max_private_latency_ms=settings.kill_switch_max_private_latency_ms,
                max_data_age_ms=settings.kill_switch_max_data_age_ms,
                max_consecutive_losses=settings.kill_switch_max_consecutive_losses,
            ),
        )
        frame = builder.build(
            symbol=symbol,
            heatmap_params=_parse_key_value_params(heatmap_param),
            allow_synthetic=allow_synthetic,
            user_address=user_address or settings.hyperliquid_user_address,
            dex=settings.hyperliquid_dex,
        )
        if not no_write:
            JsonlFrameStore(out_path).append(frame)
        console.print_json(json.dumps(frame.model_dump(mode="json")))
    finally:
        hyperliquid_client.close()
        if coinglass_client is not None:
            coinglass_client.close()


if __name__ == "__main__":
    app()
