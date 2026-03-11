# Roadmap: Dex LLM

## Overview

The first milestone builds a replay-first single-DEX trading stack for liquidation-map routing. The project starts by making frames collectible and replayable, then layers on playbook selection, and only after that adds risk-constrained execution and analytics.

## Phases

- [ ] **Phase 1: Replay Foundation** - Typed capture, storage, and replay of liquidation-map market frames
- [ ] **Phase 2: Playbook Routing** - Structured feature extraction and LLM-compatible playbook routing
- [ ] **Phase 3: Risk and Execution** - Deterministic risk policy, paper execution, and performance summaries

## Phase Details

### Phase 1: Replay Foundation
**Goal**: Establish the canonical frame model, local persistence, and replay loop for a single DEX liquidation-map workflow.
**Depends on**: Nothing (first phase)
**Requirements**: [DATA-01, DATA-02, DATA-03]
**Success Criteria** (what must be TRUE):
  1. Developer can store a market frame with candles, clusters, sweep state, and position state.
  2. Stored frames can be replayed through a typed Python API.
  3. A sample frame can be used to render structured prompt context.
**Plans**: 2 plans

Plans:
- [ ] 01-01: Create typed models, storage, and replay utilities
- [ ] 01-02: Add sample data, prompt rendering, and frame inspection CLI

### Phase 2: Playbook Routing
**Goal**: Add deterministic feature extraction and a constrained router interface for four playbooks.
**Depends on**: Phase 1
**Requirements**: [ROUT-01, ROUT-02, ROUT-03, ROUT-04]
**Success Criteria** (what must be TRUE):
  1. Feature extraction summarizes cluster dominance, reclaim behavior, and double-sweep conditions.
  2. Router output always matches the fixed playbook JSON contract.
  3. A heuristic baseline exists for replay comparisons before LLM integration.
**Plans**: 2 plans

Plans:
- [ ] 02-01: Implement feature extractor and heuristic baseline router
- [ ] 02-02: Add prompt template and replay comparison flow

### Phase 3: Risk and Execution
**Goal**: Keep all size and failure controls in code and turn approved plans into paper-executable tickets.
**Depends on**: Phase 2
**Requirements**: [RISK-01, RISK-02, RISK-03, RISK-04]
**Success Criteria** (what must be TRUE):
  1. Risk policy blocks averaging down and third consecutive-loss trades.
  2. Approved plans produce sized paper tickets with entry, stop, and targets.
  3. Outcomes can be summarized by playbook for replay review.
**Plans**: 2 plans

Plans:
- [ ] 03-01: Implement risk policy and paper executor
- [ ] 03-02: Add trade outcome analytics and reporting

## Progress

| Phase | Plans Complete | Status | Completed |
|-------|----------------|--------|-----------|
| 1. Replay Foundation | 0/2 | Not started | - |
| 2. Playbook Routing | 0/2 | Not started | - |
| 3. Risk and Execution | 0/2 | Not started | - |

