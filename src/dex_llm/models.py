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
    MAGNET_FOLLOW = "magnet_follow"
    SWEEP_RECLAIM = "sweep_reclaim"
    DOUBLE_SWEEP = "double_sweep"
    NO_TRADE = "no_trade"


class TradeSide(StrEnum):
    LONG = "long"
    SHORT = "short"
    FLAT = "flat"


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


class PositionState(BaseModel):
    side: TradeSide = TradeSide.FLAT
    entry_price: float | None = None
    quantity: float = 0.0
    open_orders: int = 0
    consecutive_losses_today: int = 0
    liquidation_price: float | None = None
    unrealized_pnl: float | None = None
    margin_used: float | None = None


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
    timestamp: datetime


class HyperliquidUserFill(BaseModel):
    coin: str
    closed_pnl: float
    direction: str
    price: float
    size: float
    time: datetime


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


class TradePlan(BaseModel):
    playbook: Playbook
    side: TradeSide
    entry_band: tuple[float, float]
    invalid_if: float
    tp1: float
    tp2: float
    ttl_min: int
    reason: str

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


class TradeOutcome(BaseModel):
    playbook: Playbook
    pnl: float
    hold_minutes: int
