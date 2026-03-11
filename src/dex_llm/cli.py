from __future__ import annotations

import json
from pathlib import Path

import typer
from rich.console import Console

from dex_llm.collector.storage import JsonlFrameStore
from dex_llm.config import AppSettings
from dex_llm.executor.paper import PaperExecutor
from dex_llm.features.extractor import FeatureExtractor
from dex_llm.llm.prompting import load_prompt_template, render_router_prompt
from dex_llm.llm.router import HeuristicPlaybookRouter
from dex_llm.models import AccountState, MarketFrame
from dex_llm.replay.session import ReplaySession
from dex_llm.risk.policy import RiskPolicy

app = typer.Typer(no_args_is_help=True, add_completion=False)
console = Console()
DEFAULT_FRAME_ARGUMENT = typer.Argument(Path("examples/sample_frame.json"))
DEFAULT_RECORD_PATH_OPTION = typer.Option(Path("data/raw/sample-session.jsonl"))
DEFAULT_REPLAY_ARGUMENT = typer.Argument(Path("data/raw/sample-session.jsonl"))


def _load_frame(path: Path) -> MarketFrame:
    return MarketFrame.model_validate_json(path.read_text(encoding="utf-8"))


@app.command()
def inspect(
    frame_path: Path = DEFAULT_FRAME_ARGUMENT,
    equity: float = typer.Option(10_000.0, help="Account equity used for paper sizing."),
) -> None:
    settings = AppSettings()
    frame = _load_frame(frame_path)
    extractor = FeatureExtractor()
    router = HeuristicPlaybookRouter()
    risk_policy = RiskPolicy(risk_per_trade_pct=settings.risk_per_trade_pct)
    account = AccountState(
        equity=equity,
        available_margin=equity,
        max_leverage=settings.max_leverage,
    )
    features = extractor.extract(frame)
    plan = router.route(frame, features)
    risk = risk_policy.assess(plan, account, frame.position)
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


if __name__ == "__main__":
    app()
