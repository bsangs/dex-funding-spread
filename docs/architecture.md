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
          -> prompt renderer / heuristic router -> risk policy -> paper executor
          -> analytics
```

Live collection currently looks like this:

```text
Hyperliquid info API -> candles + order book
Hyperliquid private info -> account state + open orders + recent fills
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
  Renders the prompt contract and ships a heuristic baseline router.
  The router must respect the kill-switch and keep the LLM on playbook selection only.

- `risk/`
  Owns all hard guards: kill-switch evaluation, no averaging down, two-loss stop, and size capping from stop distance and leverage.

- `executor/`
  Turns approved plans into paper tickets. Live execution should be added here later without changing the playbook contract.

- `analytics/`
  Summarizes outcomes by playbook after replay or paper trading.

## Current default playbook contract

The router may only return:

- `magnet_follow`
- `sweep_reclaim`
- `double_sweep`
- `no_trade`

`double_sweep` is intentionally a watch-state output. It can return `side = flat`.

## Next integration points

1. Add an LLM provider adapter that consumes the rendered prompt and validates JSON output.
2. Add live DEX order placement behind the current paper execution interface.
3. Add richer heatmap-image interpretation if the structured cluster feed is not sufficient for reclaim quality.
