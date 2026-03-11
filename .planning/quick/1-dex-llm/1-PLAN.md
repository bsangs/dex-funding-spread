---
task: 1
description: "지금까지 대화 기반 단일 DEX 청산맵 LLM 전략 프로젝트 초기 세팅"
mode: quick
created: 2026-03-12
must_haves:
  truths:
    - "The repo contains a replay-first Python scaffold for a single-DEX liquidation-map playbook router."
    - "The LLM contract is limited to playbook routing and excludes sizing."
    - "Risk controls block averaging down and stop new trades after two consecutive losses."
  artifacts:
    - ".planning/PROJECT.md"
    - ".planning/REQUIREMENTS.md"
    - ".planning/ROADMAP.md"
    - "src/dex_llm/"
    - "tests/"
  key_links:
    - "README.md"
    - "prompts/playbook_router.md"
---

# Quick Task 1 Plan

## Objective

Set up the project from scratch using the conversation-defined architecture: Python-first, single DEX, replay-first, liquidation-map routing, LLM only for playbook selection.

## Tasks

### Task 1
- files: [.planning/PROJECT.md, .planning/REQUIREMENTS.md, .planning/ROADMAP.md, .planning/STATE.md, .planning/quick/1-dex-llm/1-PLAN.md]
- action: Bootstrap GSD planning docs and quick-task tracking for the new project.
- verify: Planning docs describe the replay-first, single-DEX liquidation-map scope and phase breakdown.
- done: Quick task can be tracked even though the repo started without ROADMAP.md.

### Task 2
- files: [pyproject.toml, src/dex_llm/, prompts/playbook_router.md, examples/sample_frame.json]
- action: Create the Python package, typed domain models, feature extraction, router, risk policy, storage, replay flow, and CLI.
- verify: The sample frame can be inspected end-to-end into features, a routed playbook, and an optional paper ticket.
- done: The repo contains a runnable baseline scaffold for replay and routing.

### Task 3
- files: [README.md, configs/app.example.toml, tests/, .env.example]
- action: Add documentation, example configuration, and tests for the baseline workflow.
- verify: Local tests pass and the quick task can be summarized with atomic task commits.
- done: The scaffold is understandable and verifiable for the next implementation pass.

