# Architecture

## Intent

This codebase is optimized for replay-first strategy work on a single DEX. The priority is not live execution breadth. The priority is being able to answer:

1. What did the map look like?
2. What structured features were extracted?
3. Which playbook was chosen?
4. Did code-level risk rules allow the trade?

## Pipeline

```text
collector -> frame store -> replay session -> feature extractor
          -> prompt renderer / heuristic or OpenAI router
          -> risk policy -> paper executor or Hyperliquid live executor
          -> analytics
```

Live collection currently looks like this:

```text
Hyperliquid websocket -> l2Book + candle + bbo + activeAssetCtx
Hyperliquid websocket -> webData2 + orderUpdates + userEvents + userFills
Hyperliquid REST      -> meta + userFillsByTime + orderStatus + historicalOrders + rate limits
CoinGlass heatmap API -> liquidation clusters + optional image cache
                     \-> LiveFrameBuilder -> MarketFrame JSONL + kill-switch state
```

## Boundaries

- `collector/`
  Persists typed `MarketFrame` snapshots as JSONL.

- `integrations/`
  Fetches public and private snapshots from Hyperliquid and liquidation-map data from CoinGlass.

- `replay/`
  Replays stored frames so router behavior can be compared over time.

- `features/`
  Converts raw clusters into deterministic context such as dominance ratio, directional vacuum, and reclaim readiness.

- `llm/`
  Renders the prompt contract, ships the heuristic baseline router, and optionally calls OpenAI with structured outputs.
  The router must respect the kill-switch and keep the LLM on playbook selection only.

- `risk/`
  Owns all hard guards: kill-switch evaluation, no averaging down, two-loss stop, and size capping from stop distance and leverage.

- `executor/`
  Turns approved plans into paper tickets or live Hyperliquid orders.
  Nonce tracking, pre-submit validation, ambiguous-state resolution, and reconciliation live here.

- `analytics/`
  Summarizes outcomes by playbook after replay or paper trading.

## Current default playbook contract

The router may only return:

- `magnet_follow`
- `sweep_reclaim`
- `double_sweep`
- `no_trade`

`double_sweep` is intentionally a watch-state output. It can return `side = flat`.

## Current live path

1. `live-frame` builds a frame from WS caches plus CoinGlass heatmap and keeps synthetic fallback observe-only.
2. `route-live` produces features, router output, and risk assessment from the current live frame.
3. `sync-live` inspects account state, open orders, and rate-limit posture before any new entry is attempted.
4. `execute-live` stays paper by default and only submits real Hyperliquid orders when `--live` is present.
