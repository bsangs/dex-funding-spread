from __future__ import annotations

from datetime import UTC, datetime

from dex_llm.models import KillSwitchStatus, PositionState, TradeSide


class KillSwitchPolicy:
    def __init__(
        self,
        max_info_latency_ms: float = 1_500.0,
        max_private_latency_ms: float = 1_500.0,
        max_data_age_ms: float = 15_000.0,
    ) -> None:
        self.max_info_latency_ms = max_info_latency_ms
        self.max_private_latency_ms = max_private_latency_ms
        self.max_data_age_ms = max_data_age_ms

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

        if heatmap_provider == "synthetic-observe-only":
            reasons.append("synthetic heatmap fallback active")

        if heatmap_error is not None:
            reasons.append(f"heatmap provider error: {heatmap_error}")

        if position.entries_blocked_reduce_only:
            reasons.append("ambiguous live entry state requires reduce-only mode")

        reduce_only = bool(reasons) and (
            position.side != TradeSide.FLAT
            or position.open_orders > 0
            or any(order.reduce_only for order in position.active_orders)
            or (private_state_required and not private_state_loaded)
        )

        return KillSwitchStatus(
            allow_new_trades=not reasons,
            reduce_only=reduce_only,
            reasons=reasons,
            observed_open_orders=position.open_orders,
            entries_blocked_reduce_only=position.entries_blocked_reduce_only,
            data_age_ms=data_age_ms,
            info_latency_ms=info_latency_ms,
            private_state_latency_ms=private_state_latency_ms,
        )
