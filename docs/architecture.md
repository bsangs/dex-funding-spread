# Architecture

## Intent

This codebase is optimized for replay-first strategy development on a single DEX, but the live path is designed to survive real execution constraints:

1. What did the liquidation map look like?
2. Which deterministic features were extracted?
3. Which playbook was selected?
4. Did code-level safety rules allow the order?
5. If `cluster_fade` armed both sides, did the opposite entry get removed after the first fill?

## Pipeline

```text
collector -> frame store -> replay session -> feature extractor
          -> prompt renderer / heuristic or OpenAI router
          -> risk policy -> paper executor or Hyperliquid live executor
          -> analytics
```

Live collection and execution currently looks like this:

```text
Hyperliquid websocket -> l2Book + candle + bbo + activeAssetCtx
Hyperliquid websocket -> webData3 + orderUpdates + userEvents + userFills
Hyperliquid REST      -> meta + userFillsByTime + orderStatus + historicalOrders + rate limits
CoinGlass heatmap API -> liquidation clusters + optional image cache
                     \-> LiveFrameBuilder -> MarketFrame JSONL + kill-switch state
```

## Boundaries

- `collector/`
  Persists typed `MarketFrame` snapshots as JSONL.

- `integrations/`
  Fetches public/private Hyperliquid state and CoinGlass liquidation-map data.

- `replay/`
  Replays stored frames so routing and safety behavior can be compared over time.

- `features/`
  Converts raw clusters into deterministic context such as dominance ratio, cluster balance, directional vacuum, reclaim readiness, and cluster-fade readiness.

- `llm/`
  Renders the prompt contract, exposes the heuristic baseline router, and optionally calls OpenAI with structured outputs plus attached heatmap images.
  The LLM chooses a playbook only. It does not choose size or override safety policy.

- `risk/`
  Owns hard guards: kill-switch evaluation, no averaging down, loss-streak stops, per-side cluster-fade sizing, and liquidation buffer enforcement.

- `executor/`
  Turns approved plans into paper tickets or live Hyperliquid orders.
  Nonce tracking, pre-submit validation, ambiguous-state resolution, deterministic cloids, and post-fill cluster-fade cleanup live here.

- `analytics/`
  Summarizes outcomes by playbook after replay or paper trading.

## Playbook contract

The router may only return:

- `cluster_fade`
- `magnet_follow`
- `sweep_reclaim`
- `double_sweep`
- `no_trade`

Priority is intentionally biased toward directional clarity before balanced fades:

1. `sweep_reclaim`
2. `magnet_follow`
3. `double_sweep`
4. `cluster_fade`
5. `no_trade`

`double_sweep` remains a watch-state output and can return `side = flat`.

## Safety contract

- live private freshness is measured from `webData3`, `orderUpdates`, `userFills`, and `userEvents`
- synthetic heatmap fallback disables new entries
- isolated entries must satisfy minimum liquidation gap and stop-to-liquidation buffer checks
- the live entry path should be treated as isolated-only in v1
- cross-mode entry validation is fail-closed for new entries
- `cluster_fade` keeps asymmetric risk weights and cancels the opposite resting entry after the first fill
