# Requirements: Dex LLM

**Defined:** 2026-03-12
**Core Value:** Every trade decision must be replayable from stored market context, with the LLM limited to playbook selection and the code retaining full control of risk.

## v1 Requirements

### Replay Data

- [ ] **DATA-01**: Developer can persist replayable market frames containing candles, cluster tables, ATR, map quality, sweep state, and position state.
- [ ] **DATA-02**: Developer can load stored frames back into a replay session without losing typed structure.
- [ ] **DATA-03**: Developer can render structured LLM context from a stored frame.

### Strategy Routing

- [ ] **ROUT-01**: Router can classify a clean directional cluster imbalance as `magnet_follow`.
- [ ] **ROUT-02**: Router can classify a reclaimed sweep as `sweep_reclaim`.
- [ ] **ROUT-03**: Router can emit `double_sweep` or `no_trade` when the map is balanced or noisy.
- [ ] **ROUT-04**: Router output is constrained to the fixed JSON contract.

### Risk and Execution

- [ ] **RISK-01**: Code blocks averaging down when a position is already open.
- [ ] **RISK-02**: Code blocks new trades after two consecutive losses in a day.
- [ ] **RISK-03**: Code can derive a size recommendation from account equity, stop distance, and leverage cap.
- [ ] **RISK-04**: Paper execution can turn an approved plan into an executable ticket with entry, invalidation, and take-profit levels.

## v2 Requirements

### Integrations

- **INTG-01**: Live DEX adapter streams real-time order book and execution state.
- **INTG-02**: Live heatmap provider adapter stores synchronized image and cluster snapshots.
- **INTG-03**: Analytics compute playbook-level expectancy from recorded paper or live outcomes.

## Out of Scope

| Feature | Reason |
|---------|--------|
| Multi-venue spread execution | Different strategy class than the current single-DEX router |
| Indicator-led entries | Not part of the liquidation-map playbook thesis |
| Autonomous LLM position sizing | Size must stay under deterministic code control |

## Traceability

| Requirement | Phase | Status |
|-------------|-------|--------|
| DATA-01 | Phase 1 | Pending |
| DATA-02 | Phase 1 | Pending |
| DATA-03 | Phase 1 | Pending |
| ROUT-01 | Phase 2 | Pending |
| ROUT-02 | Phase 2 | Pending |
| ROUT-03 | Phase 2 | Pending |
| ROUT-04 | Phase 2 | Pending |
| RISK-01 | Phase 3 | Pending |
| RISK-02 | Phase 3 | Pending |
| RISK-03 | Phase 3 | Pending |
| RISK-04 | Phase 3 | Pending |

**Coverage:**
- v1 requirements: 11 total
- Mapped to phases: 11
- Unmapped: 0

---
*Requirements defined: 2026-03-12*
*Last updated: 2026-03-12 after project bootstrap*

