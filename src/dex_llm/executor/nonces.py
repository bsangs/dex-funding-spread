from __future__ import annotations

import threading
import time
from pathlib import Path


class ClockDriftError(RuntimeError):
    """Raised when the local clock drifts too far from a trusted reference."""


class NonceManager:
    def __init__(
        self,
        signer_address: str,
        *,
        storage_path: Path | None = None,
        watermark_path: Path | None = None,
        clock_drift_limit_ms: float = 500.0,
        now_ms: callable | None = None,
    ) -> None:
        self.signer_address = signer_address.lower()
        self.storage_path = watermark_path or storage_path
        self.clock_drift_limit_ms = clock_drift_limit_ms
        self._now_ms = now_ms or self._default_now_ms
        self._lock = threading.Lock()
        self._watermark = self._load_watermark()

    def seed(self, reference_ms: int | None = None) -> int:
        with self._lock:
            now_ms = self._now_ms()
            target = max(now_ms, (self._watermark or 0) + 1)
            if reference_ms is not None:
                self._ensure_clock_is_safe(reference_ms)
                target = max(target, reference_ms)
            self._watermark = target
            self._persist_watermark()
            return target

    def next_nonce(self, reference_ms: int | None = None) -> int:
        with self._lock:
            now_ms = self._now_ms()
            if reference_ms is not None:
                self._ensure_clock_is_safe(reference_ms)
                now_ms = max(now_ms, reference_ms)
            self._watermark = max(now_ms, (self._watermark or 0) + 1)
            self._persist_watermark()
            return self._watermark

    def current_watermark(self) -> int | None:
        return self._watermark

    def current(self) -> int | None:
        return self.current_watermark()

    def _ensure_clock_is_safe(self, reference_ms: int) -> None:
        drift = abs(self._now_ms() - reference_ms)
        if drift > self.clock_drift_limit_ms:
            raise ClockDriftError(
                f"Clock drift {drift:.0f} ms exceeds limit {self.clock_drift_limit_ms:.0f} ms"
            )

    def _load_watermark(self) -> int | None:
        if self.storage_path is None or not self.storage_path.exists():
            return None
        raw = self.storage_path.read_text(encoding="utf-8").strip()
        if not raw:
            return None
        return int(raw)

    def _persist_watermark(self) -> None:
        if self.storage_path is None or self._watermark is None:
            return
        self.storage_path.parent.mkdir(parents=True, exist_ok=True)
        self.storage_path.write_text(str(self._watermark), encoding="utf-8")

    @staticmethod
    def _default_now_ms() -> int:
        return int(time.time() * 1000)
