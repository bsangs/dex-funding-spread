from __future__ import annotations

from pathlib import Path

from dex_llm.models import MarketFrame


class JsonlFrameStore:
    def __init__(self, path: Path) -> None:
        self.path = path

    def append(self, frame: MarketFrame) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(frame.model_dump_json())
            handle.write("\n")

    def read_all(self) -> list[MarketFrame]:
        if not self.path.exists():
            return []
        frames: list[MarketFrame] = []
        with self.path.open("r", encoding="utf-8") as handle:
            for line in handle:
                stripped = line.strip()
                if not stripped:
                    continue
                frames.append(MarketFrame.model_validate_json(stripped))
        return frames

