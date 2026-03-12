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
  llm/          prompt rendering + heuristic / OpenAI router
  risk/         hard guards and position sizing
  executor/     paper + live execution, nonce, validator, reconciliation
  integrations/ Hyperliquid REST/WS adapters and CoinGlass client
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
uv run dex-llm route-live BTC --allow-synthetic
uv run pytest
```

If you have a CoinGlass API key, set `DEX_LLM_COINGLASS_API_KEY` and pass any endpoint-specific query params with repeated `--heatmap-param key=value` options.

```bash
export DEX_LLM_COINGLASS_API_KEY=...
uv run dex-llm coinglass-preview BTC --heatmap-param symbol=BTC
uv run dex-llm live-frame BTC --heatmap-param symbol=BTC --allow-synthetic
```

If you want live WS state, routing, and execution, separate the trading account from the signer:

```bash
export DEX_LLM_TRADING_USER_ADDRESS=0x...
export DEX_LLM_SIGNER_AGENT_ADDRESS=0x...
export DEX_LLM_SIGNER_PRIVATE_KEY=0x...
export DEX_LLM_OPENAI_API_KEY=...
uv run dex-llm live-frame BTC --user-address 0x... --heatmap-param symbol=BTC
uv run dex-llm route-live BTC --user-address 0x... --allow-synthetic
uv run dex-llm sync-live BTC --user-address 0x...
uv run dex-llm execute-live BTC --user-address 0x... --live
```

## Current defaults

- Python-first implementation
- single DEX focus
- replay-first workflow
- heuristic baseline router remains as the fallback when OpenAI routing times out or fails schema validation
- risk engine blocks averaging down and stops after two consecutive losses
- Hyperliquid live state is WS-first and REST is reserved for bootstrap, pagination, and order-state recovery
- live execution stays paper-by-default; `execute-live --live` is required before any real order submission
- synthetic fallback is observe-only and keeps new trades disabled when CoinGlass is unavailable

## Suggested build order

1. Configure CoinGlass query params for the exact liquidation endpoint you want to use.
2. Record JSONL sessions and cached heatmap artifacts from `dex-llm live-frame`.
3. Feed a Hyperliquid trading address into live frames so the kill-switch can gate new trades from real account state.
4. Compare heuristic router vs OpenAI router with `dex-llm route-live`.
5. Validate sync and reconciliation with `dex-llm sync-live`.
6. Only then open live order permissions with `dex-llm execute-live --live`.
