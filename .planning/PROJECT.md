# Dex LLM

## What This Is

This project is a Python-first single-DEX trading scaffold for liquidation-map playbook routing. It collects replayable market frames, extracts structured cluster features, lets an LLM choose among four playbooks, and keeps execution risk controls in deterministic code.

## Core Value

Every trade decision must be replayable from stored market context, with the LLM limited to playbook selection and the code retaining full control of risk.

## Requirements

### Validated

(None yet — ship to validate)

### Active

- [ ] Build a replay-first data model for candles, cluster tables, heatmap references, and position state.
- [ ] Build a playbook router that only returns `magnet_follow`, `sweep_reclaim`, `double_sweep`, or `no_trade`.
- [ ] Enforce sizing, invalidation, and loss limits in code instead of in the LLM response.

### Out of Scope

- Multi-DEX hedging and funding spread logic — different project shape than the current single-DEX plan.
- Indicator-led strategies such as RSI/MACD routing — rejected because the strategy is map-interpretation first.
- Averaging down logic — explicitly blocked as a risk rule.

## Context

The strategy premise comes from a conversation about treating liquidation maps as scene classification, not prediction. The preferred operating shape is `Magnet Follow + Sweep Reclaim`, with `double_sweep` and `no_trade` kept as explicit playbooks. The user requested Python-first implementation and delegated stack choices. The repo now includes a live Hyperliquid market-data adapter plus a CoinGlass heatmap adapter, with synthetic order-book clusters available as a plumbing fallback when CoinGlass is unavailable.

## Constraints

- **Tech stack**: Python-first with `uv`, typed models, and a replayable local workflow — fastest path to iteration.
- **Strategy boundary**: One DEX only — simplifies execution and removes cross-venue hedge complexity.
- **Safety boundary**: LLM can choose a playbook, but cannot choose size or override risk rules — avoids prompt drift causing position risk.

## Key Decisions

| Decision | Rationale | Outcome |
|----------|-----------|---------|
| Replay-first architecture | The strategy only improves if scenes can be replayed and compared | — Pending |
| Python-first scaffold | Fastest iteration path for a data-heavy, LLM-assisted system | ✓ Good |
| Heuristic baseline router | Needed as a deterministic control against future LLM routing | ✓ Good |
| Hyperliquid + CoinGlass integration path | Single DEX remains Hyperliquid while liquidation maps come from a dedicated provider | — Pending |
| Synthetic heatmap fallback | Keeps collection and replay plumbing runnable even without a CoinGlass key | ⚠️ Revisit |

---
*Last updated: 2026-03-12 after live integration pass*
