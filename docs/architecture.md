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
CoinGlass heatmap API -> liquidation clusters + optional image cache
                    \-> LiveFrameBuilder -> MarketFrame JSONL
```

## Boundaries

- `collector/`
  Persists typed `MarketFrame` snapshots as JSONL.

- `integrations/`
  Fetches real snapshots from Hyperliquid and CoinGlass.

- `replay/`
  Replays stored frames so router behavior can be compared over time.

- `features/`
  Converts raw clusters into deterministic context such as dominance ratio, directional vacuum, and reclaim readiness.

- `llm/`
  Renders the prompt contract and ships a heuristic baseline router.
  The LLM is expected to replace or sit beside the heuristic router, not the risk policy.

- `risk/`
  Owns all hard guards: no averaging down, two-loss stop, and size capping from stop distance and leverage.

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

1. Add private account state and open-order sync for Hyperliquid positions.
2. Add an LLM provider adapter that consumes the rendered prompt and validates JSON output.
3. Add live DEX order placement behind the current paper execution interface.
