# Plan Next Version Of 0.1.0

## Goal

Implement BudgetRAG Phase 1C for the benchmark CLI: deterministic adaptive context budgeting that chooses among existing fixed policies and budget sizes after retrieval, while preserving all Phase 1B default behavior.

Status: implemented and validated on `feature/budgetrag-phase1c`.

## Constraints

- Keep `legacy` as the default context policy.
- Implement `adaptive-heuristic` as a deterministic rule-based wrapper over existing fixed policies.
- Do not implement RL, bandits, learned policy training, runtime KV-cache pruning, or measured VRAM accounting.
- Keep built-in chat UI, web search, MiMo routing, dictionary mode, image mode, and existing retrievers functionally unchanged.
- Commit only curated documentation/reports; keep raw benchmark outputs under ignored output directories.

## Implementation Plan

1. Merge Phase 1B.1 baseline
   - Status: done locally.
   - Merge `feature/budgetrag-phase1b1` into local `internship`.
   - Push is blocked in this environment by missing GitHub HTTPS credentials.

2. Add adaptive feature extraction
   - Status: done.
   - Compute query length, estimated query tokens, candidate counts, document length stats, score gap, score dispersion, score entropy, and missing score count.

3. Add deterministic adaptive policy selection
   - Status: done.
   - Add `adaptive-heuristic` as a CLI policy.
   - Select among `char-budget`, `score-density`, `evidence-aware`, and `per-doc-budget`.
   - Use configurable small, medium, and large budgets.

4. Record adaptive metrics
   - Status: done.
   - Add per-query `adaptive_budget` metadata.
   - Aggregate selected policy counts, selected budget counts, reason counts, average query tokens, average score gap, and average score entropy.

5. Update matrix and summary tooling
   - Status: done.
   - Pass matrix budget values as adaptive medium budgets for `adaptive-heuristic`.
   - Add adaptive summary columns to CSV/Markdown outputs.

6. Validate and document
   - Status: done.
   - Run focused tests, full test suite, retrieval smoke, adaptive smoke, and compact adaptive matrix smoke.
   - Add curated `docs/reports/phase1c_adaptive_smoke_results.md`.
