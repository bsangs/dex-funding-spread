# Dex LLM

Single-DEX liquidation-map trading scaffold for a `Magnet Follow + Sweep Reclaim` playbook router.

This repo is set up for replay-first development:

- collect heatmap and market context as replayable frames
- extract deterministic features before asking an LLM anything
- let the LLM choose a playbook, not invent position sizing
- keep sizing, invalidation, loss limits, and kill switches in code
- pull live market data from Hyperliquid and, when configured, liquidation-map data from CoinGlass

## Why this shape

The strategy discussed in this project is not an indicator bot. It is a scene-classification engine:

- `magnet_follow`
- `sweep_reclaim`
- `double_sweep`
- `no_trade`

That means the most important artifact is not the order engine. It is a replayable dataset of liquidation-map frames plus a strict routing contract.

## Project layout

```text
src/dex_llm/
  collector/    JSONL frame capture
  replay/       replay session helpers
  features/     deterministic feature extraction
  llm/          prompt rendering + heuristic baseline router
  risk/         hard guards and position sizing
  executor/     paper execution tickets
  analytics/    simple trade outcome summaries

.planning/      GSD project tracking docs and quick-task history
configs/        example runtime config
examples/       sample replay frame
prompts/        LLM system prompt contract
tests/          scaffold verification
```

## Quick start

```bash
uv sync --dev
uv run dex-llm inspect examples/sample_frame.json
uv run dex-llm prompt examples/sample_frame.json
uv run dex-llm hyperliquid-snapshot BTC
uv run dex-llm live-frame BTC --allow-synthetic
uv run pytest
```

If you have a CoinGlass API key, set `DEX_LLM_COINGLASS_API_KEY` and pass any endpoint-specific query params with repeated `--heatmap-param key=value` options.

```bash
export DEX_LLM_COINGLASS_API_KEY=...
uv run dex-llm coinglass-preview BTC --heatmap-param symbol=BTC
uv run dex-llm live-frame BTC --heatmap-param symbol=BTC --allow-synthetic
```

## Current defaults

- Python-first implementation
- single DEX focus
- replay-first workflow
- heuristic baseline router included so you can compare LLM behavior against a deterministic fallback
- risk engine blocks averaging down and stops after two consecutive losses
- live frame builder can fall back to synthetic order-book clusters when CoinGlass is unavailable

## Suggested build order

1. Configure CoinGlass query params for the exact liquidation endpoint you want to use.
2. Record JSONL sessions and cached heatmap artifacts from `dex-llm live-frame`.
3. Compare heuristic router vs LLM router in replay runs.
4. Add paper execution stats and private account state.
5. Only then open live order permissions.
