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


class PositionState(BaseModel):
    side: TradeSide = TradeSide.FLAT
    entry_price: float | None = None
    quantity: float = 0.0
    open_orders: int = 0
    consecutive_losses_today: int = 0


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
