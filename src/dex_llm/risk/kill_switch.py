from __future__ import annotations

from datetime import UTC, datetime

from dex_llm.models import KillSwitchStatus, PositionState, TradeSide


class KillSwitchPolicy:
    def __init__(
        self,
        max_info_latency_ms: float = 1_500.0,
        max_private_latency_ms: float = 1_500.0,
        max_data_age_ms: float = 15_000.0,
        max_consecutive_losses: int = 2,
    ) -> None:
        self.max_info_latency_ms = max_info_latency_ms
        self.max_private_latency_ms = max_private_latency_ms
        self.max_data_age_ms = max_data_age_ms
        self.max_consecutive_losses = max_consecutive_losses

    def evaluate(
        self,
        frame_timestamp: datetime,
        position: PositionState,
        *,
        info_latency_ms: float | None,
        private_state_latency_ms: float | None,
        private_state_required: bool,
        private_state_loaded: bool,
        heatmap_provider: str,
        heatmap_error: str | None = None,
    ) -> KillSwitchStatus:
        reasons: list[str] = []
        data_age_ms = max(
            0.0,
            (datetime.now(tz=UTC) - frame_timestamp).total_seconds() * 1000,
        )

        if info_latency_ms is not None and info_latency_ms > self.max_info_latency_ms:
            reasons.append(f"public data latency too high ({info_latency_ms:.0f} ms)")

        if private_state_required and not private_state_loaded:
            reasons.append("private account state unavailable")

        if (
            private_state_required
            and private_state_latency_ms is not None
            and private_state_latency_ms > self.max_private_latency_ms
        ):
            reasons.append(f"private state latency too high ({private_state_latency_ms:.0f} ms)")

        if data_age_ms > self.max_data_age_ms:
            reasons.append(f"market frame stale ({data_age_ms:.0f} ms old)")

        if heatmap_provider == "synthetic-orderbook":
            reasons.append("synthetic heatmap fallback active")

        if heatmap_error is not None:
            reasons.append(f"heatmap provider error: {heatmap_error}")

        if position.consecutive_losses_today >= self.max_consecutive_losses:
            reasons.append("daily loss streak limit reached")

        reduce_only = bool(reasons) and (
            position.side != TradeSide.FLAT
            or position.open_orders > 0
            or (private_state_required and not private_state_loaded)
        )

        return KillSwitchStatus(
            allow_new_trades=not reasons,
            reduce_only=reduce_only,
            reasons=reasons,
            observed_open_orders=position.open_orders,
            data_age_ms=data_age_ms,
            info_latency_ms=info_latency_ms,
            private_state_latency_ms=private_state_latency_ms,
        )
