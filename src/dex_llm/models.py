from __future__ import annotations

from datetime import datetime
from enum import StrEnum

from pydantic import BaseModel, Field, model_validator


class ClusterSide(StrEnum):
    ABOVE = "above"
    BELOW = "below"


class ClusterShape(StrEnum):
    SINGLE_WALL = "single_wall"
    STAIRCASE = "staircase"
    DIFFUSE = "diffuse"


class MapQuality(StrEnum):
    CLEAN = "clean"
    MIXED = "mixed"
    DIRTY = "dirty"


class Playbook(StrEnum):
    CLUSTER_FADE = "cluster_fade"
    MAGNET_FOLLOW = "magnet_follow"
    SWEEP_RECLAIM = "sweep_reclaim"
    DOUBLE_SWEEP = "double_sweep"
    NO_TRADE = "no_trade"


class TradeSide(StrEnum):
    LONG = "long"
    SHORT = "short"
    FLAT = "flat"


class ExecutionMode(StrEnum):
    PAPER = "paper"
    LIVE = "live"


class MarginMode(StrEnum):
    CROSS = "cross"
    ISOLATED = "isolated"


class OrderRole(StrEnum):
    ENTRY = "entry"
    TAKE_PROFIT_1 = "tp1"
    TAKE_PROFIT_2 = "tp2"
    STOP_LOSS = "sl"
    UNKNOWN = "unknown"


class OrderState(StrEnum):
    OPEN = "open"
    FILLED = "filled"
    CANCELED = "canceled"
    TRIGGERED = "triggered"
    REJECTED = "rejected"
    SCHEDULED_CANCEL = "scheduledCancel"
    UNKNOWN = "unknown"


class UserEventType(StrEnum):
    LIQUIDATION = "liquidation"
    NON_USER_CANCEL = "nonUserCancel"
    OTHER = "other"


class ReconciliationDecision(StrEnum):
    KEEP = "keep"
    MODIFY = "modify"
    CANCEL = "cancel"
    CANCEL_PLACE = "cancel_place"
    PLACE = "place"
    AWAIT_RESOLUTION = "await_resolution"


class Candle(BaseModel):
    ts: datetime
    open: float
    high: float
    low: float
    close: float
    volume: float = Field(ge=0.0)


class Cluster(BaseModel):
    side: ClusterSide
    price: float = Field(gt=0.0)
    size: float = Field(gt=0.0)
    shape: ClusterShape


class SweepObservation(BaseModel):
    touched_cluster_side: ClusterSide | None = None
    wick_only: bool = False
    body_reclaimed: bool = False
    cluster_price: float | None = None
    trigger_candle_ts: datetime | None = None


class LiveOrderState(BaseModel):
    coin: str
    side: str
    limit_price: float = 0.0
    size: float = 0.0
    reduce_only: bool = False
    is_trigger: bool = False
    order_type: str = "unknown"
    oid: int = 0
    cloid: str | None = None
    status: OrderState = OrderState.UNKNOWN
    role: OrderRole = OrderRole.UNKNOWN
    timestamp: datetime | None = None
    trigger_price: float | None = None


class HyperliquidUserEvent(BaseModel):
    event_type: UserEventType = UserEventType.OTHER
    coin: str | None = None
    reason: str | None = None
    oid: int | None = None
    cloid: str | None = None
    timestamp: datetime | None = None
    payload: dict[str, object] = Field(default_factory=dict)


class PositionState(BaseModel):
    side: TradeSide = TradeSide.FLAT
    entry_price: float | None = None
    quantity: float = 0.0
    open_orders: int = 0
    active_orders: list[LiveOrderState] = Field(default_factory=list)
    consecutive_losses_today: int = 0
    fills_cursor: str | None = None
    last_user_event: HyperliquidUserEvent | None = None
    liquidation_price: float | None = None
    unrealized_pnl: float | None = None
    margin_used: float | None = None
    live_leverage: float | None = None
    target_leverage: float | None = None
    margin_mode: MarginMode | None = None
    entries_blocked_reduce_only: bool = False


class PriceLevel(BaseModel):
    price: float = Field(gt=0.0)
    size: float = Field(ge=0.0)
    orders: int = Field(ge=0)


class OrderBookSnapshot(BaseModel):
    symbol: str
    captured_at: datetime
    best_bid: float = Field(gt=0.0)
    best_ask: float = Field(gt=0.0)
    mid_price: float = Field(gt=0.0)
    bids: list[PriceLevel]
    asks: list[PriceLevel]


class BboSnapshot(BaseModel):
    symbol: str
    captured_at: datetime
    bid: float = Field(gt=0.0)
    ask: float = Field(gt=0.0)
    mid: float = Field(gt=0.0)


class HeatmapSnapshot(BaseModel):
    provider: str
    symbol: str
    captured_at: datetime
    clusters_above: list[Cluster]
    clusters_below: list[Cluster]
    image_url: str | None = None
    image_path: str | None = None
    raw_path: str | None = None
    metadata: dict[str, object] = Field(default_factory=dict)


class KillSwitchStatus(BaseModel):
    allow_new_trades: bool = True
    reduce_only: bool = False
    reasons: list[str] = Field(default_factory=list)
    observed_open_orders: int = 0
    entries_blocked_reduce_only: bool = False
    data_age_ms: float | None = None
    info_latency_ms: float | None = None
    private_state_latency_ms: float | None = None


class HyperliquidMarginSummary(BaseModel):
    account_value: float = 0.0
    total_margin_used: float = 0.0
    total_ntl_pos: float = 0.0
    total_raw_usd: float = 0.0


class HyperliquidPerpPosition(BaseModel):
    coin: str
    entry_price: float | None = None
    liquidation_price: float | None = None
    leverage_type: str = "cross"
    leverage_value: float = 0.0
    raw_usd: float | None = None
    margin_used: float = 0.0
    max_leverage: float = 0.0
    position_value: float = 0.0
    size: float = 0.0
    unrealized_pnl: float = 0.0


class HyperliquidClearinghouseState(BaseModel):
    asset_positions: list[HyperliquidPerpPosition]
    cross_maintenance_margin_used: float = 0.0
    cross_margin_summary: HyperliquidMarginSummary
    margin_summary: HyperliquidMarginSummary
    withdrawable: float = 0.0
    time: datetime


class HyperliquidFrontendOrder(BaseModel):
    coin: str
    side: str
    limit_price: float
    size: float
    reduce_only: bool
    is_trigger: bool
    order_type: str
    oid: int
    cloid: str | None = None
    trigger_price: float | None = None
    timestamp: datetime


class HyperliquidUserFill(BaseModel):
    coin: str
    closed_pnl: float
    direction: str
    price: float
    size: float
    time: datetime
    oid: int | None = None
    fill_hash: str | None = None
    side: str | None = None
    crossed: bool | None = None
    start_position: float | None = None


class HyperliquidActiveAssetContext(BaseModel):
    coin: str
    mark_price: float | None = None
    oracle_price: float | None = None
    mid_price: float | None = None
    max_leverage: float | None = None
    funding: float | None = None
    open_interest: float | None = None
    timestamp: datetime | None = None


class LiveStateSnapshot(BaseModel):
    symbol: str
    captured_at: datetime | None = None
    order_book: OrderBookSnapshot
    candles_5m: list[Candle]
    candles_15m: list[Candle]
    bbo: BboSnapshot | None = None
    active_asset_ctx: HyperliquidActiveAssetContext | None = None
    clearinghouse_state: HyperliquidClearinghouseState | None = None
    open_orders: list[LiveOrderState] = Field(default_factory=list)
    recent_fills: list[HyperliquidUserFill] = Field(default_factory=list)
    recent_user_events: list[HyperliquidUserEvent] = Field(default_factory=list)
    channel_timestamps: dict[str, datetime] = Field(default_factory=dict)
    channel_snapshot_flags: dict[str, bool] = Field(default_factory=dict)


class MarketFrame(BaseModel):
    timestamp: datetime
    exchange: str
    symbol: str
    current_price: float = Field(gt=0.0)
    candles_5m: list[Candle]
    candles_15m: list[Candle]
    clusters_above: list[Cluster]
    clusters_below: list[Cluster]
    atr: float = Field(gt=0.0)
    heatmap_path: str | None = None
    map_quality: MapQuality = MapQuality.CLEAN
    sweep: SweepObservation = Field(default_factory=SweepObservation)
    position: PositionState = Field(default_factory=PositionState)
    kill_switch: KillSwitchStatus = Field(default_factory=KillSwitchStatus)
    metadata: dict[str, object] = Field(default_factory=dict)


class FeatureSnapshot(BaseModel):
    dominant_cluster_side: ClusterSide | None
    dominant_ratio: float
    closest_above_distance: float | None
    closest_below_distance: float | None
    top_above: Cluster | None
    top_below: Cluster | None
    sweep_reclaim_ready: bool
    double_sweep_ready: bool
    directional_vacuum: bool
    notes: list[str]


class RestingOrderPlan(BaseModel):
    side: TradeSide
    entry_band: tuple[float, float]
    invalid_if: float
    tp1: float
    tp2: float
    ttl_min: int
    reason: str
    cluster_price: float | None = None

    @model_validator(mode="after")
    def validate_band(self) -> RestingOrderPlan:
        if self.entry_band[0] > self.entry_band[1]:
            raise ValueError("resting order entry_band must be ascending")
        if self.side == TradeSide.FLAT:
            raise ValueError("resting order side must be long or short")
        return self


class TradePlan(BaseModel):
    playbook: Playbook
    side: TradeSide
    entry_band: tuple[float, float]
    invalid_if: float
    tp1: float
    tp2: float
    ttl_min: int
    reason: str
    resting_orders: list[RestingOrderPlan] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_band(self) -> TradePlan:
        if self.entry_band[0] > self.entry_band[1]:
            raise ValueError("entry_band must be ascending")
        return self


class AccountState(BaseModel):
    equity: float = Field(gt=0.0)
    available_margin: float = Field(gt=0.0)
    max_leverage: float = Field(gt=0.0)


class RiskAssessment(BaseModel):
    allowed: bool
    reason: str
    recommended_quantity: float = 0.0
    recommended_notional: float = 0.0
    risk_budget: float = 0.0


class PaperOrderTicket(BaseModel):
    side: TradeSide
    entry_price: float
    quantity: float
    invalid_if: float
    take_profit_1: float
    take_profit_2: float
    ttl_min: int
    leverage: float
    playbook: Playbook


class PendingActionState(BaseModel):
    symbol: str
    cloid: str
    first_seen_at: datetime
    last_checked_at: datetime | None = None
    attempts: int = 0
    last_error: str | None = None
    decision: ReconciliationDecision = ReconciliationDecision.AWAIT_RESOLUTION


class RateLimitSnapshot(BaseModel):
    address: str
    address_actions_used: int = 0
    address_actions_limit: int = 0
    open_orders: int = 0
    rest_weight_used: int = 0
    ws_messages_used: int = 0
    ws_subscriptions_used: int = 0
    inflight_posts: int = 0
    degrade_mode: str = "normal"


class ExecutionReceipt(BaseModel):
    mode: ExecutionMode = ExecutionMode.PAPER
    symbol: str
    action: str
    cloid: str
    oid: int | None = None
    decision: ReconciliationDecision
    success: bool
    status: OrderState = OrderState.UNKNOWN
    message: str = ""
    raw_response: dict[str, object] = Field(default_factory=dict)
    submitted_at: datetime


class TradeOutcome(BaseModel):
    playbook: Playbook
    pnl: float
    hold_minutes: int
