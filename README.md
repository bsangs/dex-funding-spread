# Dex LLM

Single-DEX liquidation-map trading engine for Hyperliquid with CoinGlass clusters, OpenAI structured routing, and WS-first account state.

The default runtime is now aligned around ETH and a five-playbook contract:

- `cluster_fade`
- `magnet_follow`
- `sweep_reclaim`
- `double_sweep`
- `no_trade`

## Why this shape

This repo is built for replay-first strategy work, but the live path is no longer a paper-only scaffold:

- Hyperliquid is the single execution venue
- CoinGlass provides the liquidation-map clusters and optional cached heatmap images
- OpenAI routing stays schema-bound and playbook-only
- risk, sizing, liquidation buffers, and kill-switches stay in code
- live private state is WS-first with REST reserved for bootstrap, pagination, and recovery

`cluster_fade` is handled as a real two-leg entry workflow:

- lower wall = resting long fade
- upper wall = resting short fade
- per-side sizing defaults to `long=0.8`, `short=0.3` risk weights
- once one side fills, the opposite entry is canceled and only the filled side keeps protection orders

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
examples/       sample ETH replay frame
prompts/        LLM system prompt contract
tests/          regression coverage
```

## Quick start

```bash
uv sync --dev
uv run dex-llm inspect examples/sample_frame.json
uv run dex-llm prompt examples/sample_frame.json
uv run dex-llm hyperliquid-snapshot ETH
uv run dex-llm live-frame ETH --allow-synthetic
uv run dex-llm route-live ETH --allow-synthetic
uv run pytest
```

If you have a CoinGlass API key, set `DEX_LLM_COINGLASS_API_KEY` and pass any endpoint-specific params with repeated `--heatmap-param key=value` options.

```bash
export DEX_LLM_COINGLASS_API_KEY=...
uv run dex-llm coinglass-preview ETH --heatmap-param symbol=ETH
uv run dex-llm live-frame ETH --heatmap-param symbol=ETH --allow-synthetic
```

For live WS state, routing, and execution, keep the signer separate from the trading account:

```bash
export DEX_LLM_TRADING_USER_ADDRESS=0x...
export DEX_LLM_SIGNER_AGENT_ADDRESS=0x...
export DEX_LLM_SIGNER_PRIVATE_KEY=0x...
export DEX_LLM_OPENAI_API_KEY=...
uv run dex-llm live-frame ETH --user-address 0x... --heatmap-param symbol=ETH
uv run dex-llm route-live ETH --user-address 0x... --allow-synthetic
uv run dex-llm sync-live ETH --user-address 0x...
uv run dex-llm execute-live ETH --user-address 0x... --live
```

## Current defaults

- Python-first implementation
- ETH as the default symbol
- single-venue execution on Hyperliquid
- OpenAI Responses API with structured output and multimodal heatmap input when an image is available
- heuristic router remains the fallback when OpenAI times out or fails schema validation
- `cluster_fade` is gated by balance and distance checks instead of firing whenever both sides exist
- liquidation buffers, no-averaging-down, and consecutive-loss limits are enforced in code
- synthetic heatmaps remain observe-only and disable new entries

## Safety notes

- the live entry path is effectively isolated-only in v1
- cross-mode entry validation is intentionally fail-closed and should be treated as unsupported for new entries
- isolated entries must pass both minimum liquidation gap and stop-to-liquidation buffer checks
- private freshness is derived from `webData3`, `orderUpdates`, `userFills`, and `userEvents`
- live order submission still requires `execute-live --live`
