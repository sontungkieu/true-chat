# Plan Next Version Of 0.1.0

## Goal

Implement BudgetRAG Phase 1B.1 for the benchmark CLI: schema hardening, robust summary tooling, lexical/query-aware evidence policy naming, traceable matrix runs, and a small curated smoke result snapshot.

Status: implemented and validated on `feature/budgetrag-phase1b1`; pending final review/merge.

## Constraints

- Preserve the default `legacy` context behavior.
- Keep `--max-context-chars` supported and use it as the default context budget.
- Do not implement runtime KV-cache pruning in this phase.
- Do not expose BudgetRAG controls in the built-in chat UI yet.
- Do not implement adaptive, bandit, RL, or runtime KV-cache pruning policies.
- Preserve web search, MiMo routing, dictionary mode, image mode, and all existing retrievers.
- Do not commit benchmark outputs, datasets, caches, local reports, or secrets.

## Implementation Plan

1. Clarify evidence-aware policy naming
   - Status: done.
   - Keep the CLI policy name `evidence-aware`.
   - Record implementation subtype `lexical-query-aware`.
   - Document that this is not answer-aware verification.

2. Add explicit experiment metadata
   - Status: done.
   - Add run metadata to top-level metrics, aggregate rows, and per-query rows.
   - Include benchmark, retriever, policy, budget, generation mode, and KV profile.

3. Harden summary script
   - Status: done.
   - Summarize every aggregate record.
   - Preserve run metadata in CSV rows.
   - Handle older metrics files with missing experiment metadata.

4. Improve matrix traceability
   - Status: done.
   - Add `--run-name`, `--dry-run`, and `--continue-on-error`.
   - Write a manifest for real matrix runs.

5. Add curated result snapshot
   - Status: done.
   - Run a small SciFact retrieval-only matrix.
   - Commit only `docs/reports/phase1b_smoke_results.md`.

6. Validate and commit
   - Status: done.
   - Run focused tests, full tests, and smoke commands.
   - Create medium Conventional Commits.
