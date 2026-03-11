# Dex LLM

Single-DEX liquidation-map trading scaffold for a `Magnet Follow + Sweep Reclaim` playbook router.

This repo is set up for replay-first development:

- collect heatmap and market context as replayable frames
- extract deterministic features before asking an LLM anything
- let the LLM choose a playbook, not invent position sizing
- keep sizing, invalidation, loss limits, and kill switches in code

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
uv run pytest
```

## Current defaults

- Python-first implementation
- single DEX focus
- replay-first workflow
- heuristic baseline router included so you can compare LLM behavior against a deterministic fallback
- risk engine blocks averaging down and stops after two consecutive losses

## Suggested build order

1. Replace the sample frame collector with your real DEX + heatmap provider adapter.
2. Record JSONL sessions and heatmap image paths for replay.
3. Compare heuristic router vs LLM router in replay runs.
4. Add paper execution stats.
5. Only then open live order permissions.

