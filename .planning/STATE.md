# Project State

## Project Reference

See: .planning/PROJECT.md (updated 2026-03-12)

**Core value:** Every trade decision must be replayable from stored market context, with the LLM limited to playbook selection and the code retaining full control of risk.
**Current focus:** Phase 2 - Live Safety Wiring

## Current Position

Phase: 2 of 3 (Live Safety Wiring)
Plan: 1 of 2 in current phase
Status: In progress
Last activity: 2026-03-12 - Added Hyperliquid private account sync, reclaim detection, prompt heatmap context, and live kill-switch gating

Progress: [######----] 60%

## Performance Metrics

**Velocity:**
- Total plans completed: 0
- Average duration: 0 min
- Total execution time: 0.0 hours

**By Phase:**

| Phase | Plans | Total | Avg/Plan |
|-------|-------|-------|----------|
| - | - | - | - |

**Recent Trend:**
- Last 5 plans: none
- Trend: N/A

## Accumulated Context

### Decisions

Recent decisions affecting current work:

- Use Python-first scaffolding instead of optimizing for low-latency execution on day one.
- Treat the LLM as a playbook selector; keep size and loss limits in code.
- Use Hyperliquid `info` snapshots as the default live market-data source.
- Keep CoinGlass query params configurable at the CLI because endpoint-specific parameter names may vary.
- Require kill-switch checks to gate routing before any new trade is considered.
- Pull Hyperliquid private state into live frames whenever a user address is configured.

### Pending Todos

- Add an actual LLM provider adapter behind the prompt contract.
- Add live order placement behind the current paper execution path.
- Replace synthetic/orderbook reclaim heuristics with production heatmap image analysis if needed.

### Blockers/Concerns

CoinGlass production params remain environment-specific, and live execution is still paper-only.

### Quick Tasks Completed

| # | Description | Date | Commit | Directory |
|---|-------------|------|--------|-----------|
| 1 | 지금까지 대화 기반 단일 DEX 청산맵 LLM 전략 프로젝트 초기 세팅 | 2026-03-11 | 7fd5217 | [1-dex-llm](./quick/1-dex-llm/) |

## Session Continuity

Last session: 2026-03-12 13:30
Stopped at: Live frame builder now enriches frames with private state, reclaim detection, and kill-switch output
Resume file: None
