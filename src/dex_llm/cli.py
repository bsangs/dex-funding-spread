from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

import typer
from rich.console import Console

from dex_llm.bot import BotRuntime
from dex_llm.collector.storage import JsonlFrameStore
from dex_llm.config import AppSettings
from dex_llm.executor import (
    AmbiguousStateResolver,
    HyperliquidExchangeExecutor,
    NonceManager,
    PaperExecutor,
    PreSubmitValidator,
    RateLimitBudgeter,
)
from dex_llm.features.extractor import FeatureExtractor
from dex_llm.integrations.coinglass import (
    CoinGlassFallbackHeatmapClient,
    CoinGlassHeatmapClient,
    CoinGlassLiquidationsPageClient,
)
from dex_llm.integrations.hyperliquid import HyperliquidInfoClient
from dex_llm.integrations.hyperliquid_live import HyperliquidRestGateway, HyperliquidWsStateClient
from dex_llm.live_frame import LiveFrameBuilder
from dex_llm.llm.openai_router import OpenAIRouter
from dex_llm.llm.prompting import load_prompt_template, render_router_prompt
from dex_llm.llm.router import HeuristicPlaybookRouter, RouterProtocol
from dex_llm.models import AccountState, ExecutionMode, LiveStateSnapshot, MarketFrame, TradeSide
from dex_llm.replay.session import ReplaySession
from dex_llm.risk.kill_switch import KillSwitchPolicy
from dex_llm.risk.policy import RiskPolicy

app = typer.Typer(no_args_is_help=True, add_completion=False)
console = Console()
DEFAULT_FRAME_ARGUMENT = typer.Argument(Path("examples/sample_frame.json"))
DEFAULT_RECORD_PATH_OPTION = typer.Option(Path("data/raw/sample-session.jsonl"))
DEFAULT_REPLAY_ARGUMENT = typer.Argument(Path("data/raw/sample-session.jsonl"))
DEFAULT_SYMBOL_ARGUMENT = typer.Argument("ETH")
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


def _resolved_trading_user(settings: AppSettings, user_address: str | None) -> str | None:
    return user_address or settings.trading_user_address or settings.hyperliquid_user_address


def _build_kill_switch_policy(settings: AppSettings) -> KillSwitchPolicy:
    return KillSwitchPolicy(
        max_info_latency_ms=settings.kill_switch_max_info_latency_ms,
        max_private_latency_ms=settings.kill_switch_max_private_latency_ms,
        max_data_age_ms=settings.kill_switch_max_data_age_ms,
    )


def _build_heatmap_client(settings: AppSettings) -> object | None:
    primary = None
    if settings.coinglass_api_key:
        primary = CoinGlassHeatmapClient(
            api_key=settings.coinglass_api_key,
            base_url=settings.coinglass_base_url,
            heatmap_path=settings.coinglass_heatmap_path,
            timeout_s=settings.request_timeout_s,
            cache_dir=settings.heatmap_cache_dir,
        )
    fallback = None
    if settings.coinglass_use_playwright_fallback:
        fallback = CoinGlassLiquidationsPageClient(
            page_url=settings.coinglass_web_url,
            timeout_s=settings.coinglass_scrape_timeout_s,
            cache_dir=settings.heatmap_cache_dir,
        )
    if primary is None and fallback is None:
        return None
    return CoinGlassFallbackHeatmapClient(primary, fallback)


def _build_router(settings: AppSettings) -> RouterProtocol:
    if settings.llm_provider == "openai" and settings.openai_api_key:
        return OpenAIRouter(
            api_key=settings.openai_api_key,
            model=settings.llm_model,
            entry_prompt_path=settings.entry_prompt_path,
            position_prompt_path=settings.position_prompt_path,
            timeout_s=settings.llm_timeout_s,
            verbosity=settings.openai_verbosity,
            reasoning_effort=settings.openai_reasoning_effort,
            reasoning_summary=settings.openai_reasoning_summary,
            store=settings.openai_store,
            include=[
                item.strip()
                for item in settings.openai_include.split(",")
                if item.strip()
            ],
            image_detail=settings.openai_image_detail,
        )
    return HeuristicPlaybookRouter()


def _build_account_state(
    settings: AppSettings,
    *,
    equity: float | None,
    available_margin: float | None,
    max_leverage: float | None,
) -> AccountState:
    resolved_equity = equity or 10_000.0
    resolved_margin = available_margin if available_margin is not None else resolved_equity
    resolved_max_leverage = max_leverage or settings.max_leverage
    return AccountState(
        equity=resolved_equity,
        available_margin=resolved_margin,
        max_leverage=resolved_max_leverage,
    )


def _numeric_metadata(frame: MarketFrame, key: str) -> float | None:
    value = frame.metadata.get(key)
    if isinstance(value, (float, int)):
        return float(value)
    return None


def _paper_preview(plan: object, risk: object, account: AccountState) -> object:
    resting_orders = getattr(plan, "resting_orders", [])
    if resting_orders:
        return [
            {
                "side": order.side,
                "entry_price": sum(order.entry_band) / 2,
                "quantity": (
                    risk.resting_order_quantities[index]
                    if index < len(risk.resting_order_quantities)
                    else getattr(risk, "recommended_quantity", 0.0)
                ),
                "invalid_if": order.invalid_if,
                "take_profit_1": order.tp1,
                "take_profit_2": order.tp2,
                "ttl_min": order.ttl_min,
                "leverage": min(
                    account.max_leverage,
                    (
                        (
                            (
                                risk.resting_order_quantities[index]
                                if index < len(risk.resting_order_quantities)
                                else getattr(risk, "recommended_quantity", 0.0)
                            )
                            * (sum(order.entry_band) / 2)
                        )
                        / account.equity
                    ),
                ),
                "reason": order.reason,
            }
            for index, order in enumerate(resting_orders)
        ]
    return PaperExecutor().build_ticket(plan, risk, account).model_dump(mode="json")


def _build_ws_frame(
    settings: AppSettings,
    *,
    symbol: str,
    heatmap_params: dict[str, str],
    allow_synthetic: bool,
    user_address: str | None,
) -> tuple[MarketFrame, dict[str, object]]:
    rest_gateway = HyperliquidRestGateway(
        base_url=settings.hyperliquid_base_url,
        timeout_s=settings.request_timeout_s,
    )
    ws_client = HyperliquidWsStateClient(
        base_url=settings.hyperliquid_base_url,
        timeout_s=settings.request_timeout_s,
        user_address=user_address,
    )
    heatmap_client = _build_heatmap_client(settings)
    builder = LiveFrameBuilder(
        rest_gateway,
        heatmap_client,
        kill_switch_policy=_build_kill_switch_policy(settings),
    )
    try:
        ws_client.connect(symbol, user_address=user_address)
        ws_client.wait_until_public_ready(timeout_s=settings.request_timeout_s)
        snapshot = ws_client.snapshot().model_copy(update={"captured_at": datetime.now(tz=UTC)})
        snapshot = snapshot.model_copy(
            update={
                "candles_1h": rest_gateway.fetch_candles(symbol, "1h", limit=48),
                "candles_4h": rest_gateway.fetch_candles(symbol, "4h", limit=30),
            }
        )
        fills = None
        fills_safe_complete = True
        private_source = "ws" if ws_client.private_state_ready() else "pending"
        private_bootstrap_error: str | None = None
        if user_address:
            day_start = snapshot.order_book.captured_at.astimezone(UTC).replace(
                hour=0,
                minute=0,
                second=0,
                microsecond=0,
            )
            paginated_fills, fills_safe_complete = rest_gateway.paginate_user_fills_by_time(
                user=user_address,
                start_time=int(day_start.timestamp() * 1000),
                end_time=int(snapshot.order_book.captured_at.timestamp() * 1000),
            )
            fills = paginated_fills
            if not ws_client.private_state_ready():
                try:
                    clearinghouse_state = rest_gateway.fetch_clearinghouse_state(
                        user=user_address,
                        dex=settings.hyperliquid_dex,
                    )
                    open_orders = rest_gateway.fetch_open_orders(
                        user=user_address,
                        dex=settings.hyperliquid_dex,
                    )
                    snapshot = _apply_rest_private_bootstrap(
                        snapshot,
                        clearinghouse_state=clearinghouse_state,
                        open_orders=open_orders,
                        fills=fills,
                        bootstrapped_at=datetime.now(tz=UTC),
                    )
                    private_source = "rest_bootstrap"
                except Exception as exc:
                    private_bootstrap_error = str(exc)
            elif fills is not None:
                snapshot = snapshot.model_copy(update={"recent_fills": fills})
        frame = builder.build_from_snapshot(
            snapshot,
            heatmap_params=heatmap_params,
            allow_synthetic=allow_synthetic,
            fills=fills,
        )
        meta = {
            "snapshot": snapshot.model_dump(mode="json"),
            "fills_safe_complete": fills_safe_complete,
            "private_state_source": private_source,
            "private_ws_ready": ws_client.private_state_ready(),
        }
        if private_bootstrap_error is not None:
            meta["private_bootstrap_error"] = private_bootstrap_error
        if not fills_safe_complete:
            frame.kill_switch.allow_new_trades = False
            frame.kill_switch.reduce_only = True
            frame.kill_switch.reasons.append(
                "fill backfill incomplete; safe-fail kill switch active"
            )
        if user_address and snapshot.clearinghouse_state is None:
            frame.kill_switch.allow_new_trades = False
            frame.kill_switch.reasons.append(
                "private account state unavailable after websocket and REST bootstrap"
            )
        return frame, meta
    finally:
        ws_client.close()
        rest_gateway.close()
        if heatmap_client is not None:
            heatmap_client.close()


def _apply_rest_private_bootstrap(
    snapshot: LiveStateSnapshot,
    *,
    clearinghouse_state: object,
    open_orders: list[object],
    fills: list[object] | None,
    bootstrapped_at: datetime,
) -> LiveStateSnapshot:
    channel_timestamps = dict(snapshot.channel_timestamps)
    channel_timestamps["restPrivateBootstrap"] = bootstrapped_at
    channel_snapshot_flags = dict(snapshot.channel_snapshot_flags)
    channel_snapshot_flags["restPrivateBootstrap"] = False
    return snapshot.model_copy(
        update={
            "clearinghouse_state": clearinghouse_state,
            "open_orders": open_orders,
            "recent_fills": fills if fills is not None else snapshot.recent_fills,
            "channel_timestamps": channel_timestamps,
            "channel_snapshot_flags": channel_snapshot_flags,
        }
    )


def _build_executor(
    settings: AppSettings,
    *,
    symbol: str,
    rest_gateway: HyperliquidRestGateway,
    user_address: str,
) -> HyperliquidExchangeExecutor:
    if not settings.signer_private_key or not settings.signer_agent_address:
        raise typer.BadParameter(
            "DEX_LLM_SIGNER_PRIVATE_KEY and "
            "DEX_LLM_SIGNER_AGENT_ADDRESS are required for live execution"
        )
    asset_meta = rest_gateway.fetch_asset_meta(symbol)
    validator = PreSubmitValidator(
        {symbol: asset_meta},
        max_price_deviation_bps=settings.max_price_deviation_bps,
    )
    budgeter = RateLimitBudgeter()
    rate_limit_payload = rest_gateway.user_rate_limit(user_address)
    used = (
        int(rate_limit_payload.get("nRequestsUsed", 0))
        if isinstance(rate_limit_payload, dict)
        else 0
    )
    limit = (
        int(rate_limit_payload.get("nRequestsCap", 0))
        if isinstance(rate_limit_payload, dict)
        else 0
    )
    if limit > 0:
        budgeter.update_address_budget(used=used, limit=limit)
    resolver = AmbiguousStateResolver(
        query_order_by_cloid=lambda cloid: rest_gateway.query_order_by_cloid(user_address, cloid),
        fetch_open_orders=lambda: list(rest_gateway.open_orders(user_address)),
        fetch_historical_orders=lambda: list(rest_gateway.historical_orders(user_address)),
    )
    return HyperliquidExchangeExecutor(
        base_url=settings.hyperliquid_base_url,
        signer_private_key=settings.signer_private_key,
        signer_agent_address=settings.signer_agent_address,
        trading_user_address=user_address,
        validator=validator,
        nonce_manager=NonceManager(
            settings.signer_agent_address,
            storage_path=settings.raw_data_dir / "nonce-watermark.txt",
            clock_drift_limit_ms=settings.clock_drift_limit_ms,
        ),
        budgeter=budgeter,
        vault_address=settings.hyperliquid_vault_address,
        ambiguous_resolver=resolver,
        margin_mode=settings.margin_mode,
        target_leverage=settings.target_leverage,
        enable_stop_loss=settings.enable_stop_loss,
    )


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
        long_notional_fraction=settings.long_notional_fraction,
        short_notional_fraction=settings.short_notional_fraction,
        target_leverage=settings.target_leverage,
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
        payload["paper_ticket"] = _paper_preview(plan, risk, account)
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
        template=load_prompt_template(
            settings.position_prompt_path
            if frame.position.side != TradeSide.FLAT
            else settings.entry_prompt_path
        ),
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
    client = _build_heatmap_client(settings)
    if client is None:
        raise typer.BadParameter("CoinGlass provider is not configured")
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
    frame, meta = _build_ws_frame(
        settings,
        symbol=symbol,
        heatmap_params=_parse_key_value_params(heatmap_param),
        allow_synthetic=allow_synthetic,
        user_address=_resolved_trading_user(settings, user_address),
    )
    payload = frame.model_dump(mode="json")
    payload["live_state"] = meta
    if not no_write:
        JsonlFrameStore(out_path).append(frame)
    console.print_json(json.dumps(payload))


@app.command("route-live")
def route_live(
    symbol: str = DEFAULT_SYMBOL_ARGUMENT,
    heatmap_param: list[str] | None = HEATMAP_PARAM_OPTION,
    allow_synthetic: bool = ALLOW_SYNTHETIC_OPTION,
    user_address: str | None = USER_ADDRESS_OPTION,
) -> None:
    settings = AppSettings()
    frame, meta = _build_ws_frame(
        settings,
        symbol=symbol,
        heatmap_params=_parse_key_value_params(heatmap_param),
        allow_synthetic=allow_synthetic,
        user_address=_resolved_trading_user(settings, user_address),
    )
    features = FeatureExtractor().extract(frame)
    router = _build_router(settings)
    plan = router.route(frame, features)
    account = _build_account_state(
        settings,
        equity=_numeric_metadata(frame, "account_value"),
        available_margin=_numeric_metadata(frame, "withdrawable"),
        max_leverage=settings.max_leverage,
    )
    risk = RiskPolicy(
        long_notional_fraction=settings.long_notional_fraction,
        short_notional_fraction=settings.short_notional_fraction,
        target_leverage=settings.target_leverage,
    ).assess(plan, account, frame.position, frame.kill_switch)
    console.print_json(
        json.dumps(
            {
                "frame": frame.model_dump(mode="json"),
                "features": features.model_dump(mode="json"),
                "plan": plan.model_dump(mode="json"),
                "risk": risk.model_dump(mode="json"),
                "live_state": meta,
            }
        )
    )


@app.command("sync-live")
def sync_live(
    symbol: str = DEFAULT_SYMBOL_ARGUMENT,
    user_address: str | None = USER_ADDRESS_OPTION,
) -> None:
    settings = AppSettings()
    resolved_user = _resolved_trading_user(settings, user_address)
    if resolved_user is None:
        raise typer.BadParameter("trading user address is required for sync-live")
    rest_gateway = HyperliquidRestGateway(
        base_url=settings.hyperliquid_base_url,
        timeout_s=settings.request_timeout_s,
    )
    try:
        state = rest_gateway.fetch_clearinghouse_state(
            user=resolved_user,
            dex=settings.hyperliquid_dex,
        )
        open_orders = rest_gateway.fetch_open_orders(
            user=resolved_user,
            dex=settings.hyperliquid_dex,
        )
        rate_limit = rest_gateway.user_rate_limit(resolved_user)
        console.print_json(
            json.dumps(
                {
                    "user": resolved_user,
                    "clearinghouse_state": state.model_dump(mode="json"),
                    "open_orders": [order.model_dump(mode="json") for order in open_orders],
                    "user_rate_limit": rate_limit,
                }
            )
        )
    finally:
        rest_gateway.close()


@app.command("execute-live")
def execute_live(
    symbol: str = DEFAULT_SYMBOL_ARGUMENT,
    heatmap_param: list[str] | None = HEATMAP_PARAM_OPTION,
    allow_synthetic: bool = ALLOW_SYNTHETIC_OPTION,
    user_address: str | None = USER_ADDRESS_OPTION,
    live: bool = typer.Option(
        False,
        "--live",
        help="Actually submit live orders instead of paper tickets.",
    ),
) -> None:
    settings = AppSettings()
    resolved_user = _resolved_trading_user(settings, user_address)
    if resolved_user is None:
        raise typer.BadParameter("trading user address is required for execute-live")
    frame, meta = _build_ws_frame(
        settings,
        symbol=symbol,
        heatmap_params=_parse_key_value_params(heatmap_param),
        allow_synthetic=allow_synthetic,
        user_address=resolved_user,
    )
    features = FeatureExtractor().extract(frame)
    plan = _build_router(settings).route(frame, features)
    account = _build_account_state(
        settings,
        equity=_numeric_metadata(frame, "account_value"),
        available_margin=_numeric_metadata(frame, "withdrawable"),
        max_leverage=settings.max_leverage,
    )
    risk = RiskPolicy(
        long_notional_fraction=settings.long_notional_fraction,
        short_notional_fraction=settings.short_notional_fraction,
        target_leverage=settings.target_leverage,
    ).assess(plan, account, frame.position, frame.kill_switch)
    payload: dict[str, object] = {
        "frame": frame.model_dump(mode="json"),
        "features": features.model_dump(mode="json"),
        "plan": plan.model_dump(mode="json"),
        "risk": risk.model_dump(mode="json"),
        "live_state": meta,
    }
    protection_only = frame.position.side != TradeSide.FLAT and frame.position.quantity > 0
    if not risk.allowed and not protection_only:
        console.print_json(json.dumps(payload))
        return

    if not live and settings.execution_mode != ExecutionMode.LIVE:
        payload["paper_ticket"] = _paper_preview(plan, risk, account)
        console.print_json(json.dumps(payload))
        return

    rest_gateway = HyperliquidRestGateway(
        base_url=settings.hyperliquid_base_url,
        timeout_s=settings.request_timeout_s,
    )
    try:
        executor = _build_executor(
            settings,
            symbol=symbol,
            rest_gateway=rest_gateway,
            user_address=resolved_user,
        )
        executor.verify_signer()
        leverage_preflight = executor.apply_leverage_preflight(
            symbol=symbol,
            target_leverage=settings.target_leverage,
            margin_mode=settings.margin_mode,
            current_leverage=frame.position.live_leverage,
            max_leverage=account.max_leverage,
            recommended_notional=(
                risk.recommended_notional
                if risk.allowed
                else frame.position.quantity * frame.current_price
            ),
            available_margin=account.available_margin,
        )
        payload["leverage_preflight"] = leverage_preflight.model_dump(mode="json")
        if not leverage_preflight.valid:
            console.print_json(json.dumps(payload))
            return
        receipts = executor.execute_plan(
            plan=plan,
            risk=risk,
            symbol=symbol,
            frame_timestamp=frame.timestamp,
            position=frame.position,
            best_bid=frame.metadata.get("best_bid", frame.current_price),
            best_ask=frame.metadata.get("best_ask", frame.current_price),
            oracle_price=frame.metadata.get("oracle_price"),
        )
        dms = executor.schedule_dead_man_switch(
            has_resting_entry=any(receipt.action in {"place", "modify"} for receipt in receipts),
            position_open=frame.position.side != TradeSide.FLAT,
            now=datetime.now(tz=frame.timestamp.tzinfo or UTC),
        )
        payload["execution_receipts"] = [receipt.model_dump(mode="json") for receipt in receipts]
        payload["dead_man_switch"] = dms
        console.print_json(json.dumps(payload))
    finally:
        rest_gateway.close()


@app.command("run-bot")
def run_bot(
    symbol: str = DEFAULT_SYMBOL_ARGUMENT,
    heatmap_param: list[str] | None = HEATMAP_PARAM_OPTION,
    allow_synthetic: bool = ALLOW_SYNTHETIC_OPTION,
    user_address: str | None = USER_ADDRESS_OPTION,
    live: bool = typer.Option(
        False,
        "--live",
        help="Actually submit live orders instead of paper output.",
    ),
    strategy_interval_s: int | None = typer.Option(
        None,
        help="How often to refresh the strategy/heatmap decision loop.",
    ),
    sync_interval_s: int | None = typer.Option(
        None,
        help="How often to sync orders, fills, and protections.",
    ),
    max_cycles: int | None = typer.Option(
        None,
        help="Optional test hook to stop after N sync cycles.",
        hidden=True,
    ),
) -> None:
    settings = AppSettings()
    resolved_user = _resolved_trading_user(settings, user_address)
    if resolved_user is None:
        raise typer.BadParameter("trading user address is required for run-bot")
    rest_gateway = HyperliquidRestGateway(
        base_url=settings.hyperliquid_base_url,
        timeout_s=settings.request_timeout_s,
    )
    ws_client = HyperliquidWsStateClient(
        base_url=settings.hyperliquid_base_url,
        timeout_s=settings.request_timeout_s,
        user_address=resolved_user,
    )
    heatmap_client = _build_heatmap_client(settings)
    builder = LiveFrameBuilder(
        rest_gateway,
        heatmap_client,
        kill_switch_policy=_build_kill_switch_policy(settings),
    )
    executor = _build_executor(
        settings,
        symbol=symbol,
        rest_gateway=rest_gateway,
        user_address=resolved_user,
    )
    executor.verify_signer()
    runtime = BotRuntime(
        symbol=symbol,
        user_address=resolved_user,
        heatmap_params=_parse_key_value_params(heatmap_param),
        allow_synthetic=allow_synthetic,
        router=_build_router(settings),
        builder=builder,
        rest_gateway=rest_gateway,
        ws_client=ws_client,
        executor=executor,
        risk_policy=RiskPolicy(
            long_notional_fraction=settings.long_notional_fraction,
            short_notional_fraction=settings.short_notional_fraction,
            target_leverage=settings.target_leverage,
        ),
        max_leverage=settings.max_leverage,
        strategy_interval_s=strategy_interval_s or settings.bot_strategy_interval_s,
        sync_interval_s=sync_interval_s or settings.bot_sync_interval_s,
        live=live,
        execution_mode_live=settings.execution_mode == ExecutionMode.LIVE,
        dex=settings.hyperliquid_dex,
        enable_stop_loss=settings.enable_stop_loss,
        console=console,
    )
    runtime.run(max_cycles=max_cycles)


if __name__ == "__main__":
    app()
