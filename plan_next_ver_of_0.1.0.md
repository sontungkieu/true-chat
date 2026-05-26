# Plan Next Version Of 0.1.0

## Goal

Implement BudgetRAG Phase 1B for the benchmark CLI: context budgeting, context compression policies, evidence-aware retention, context/KV estimate metrics, and reproducible policy matrix tooling.

Status: implemented and validated on `feature/budgetrag-phase1b`; pending final review/merge.

## Constraints

- Preserve the default `legacy` context behavior.
- Keep `--max-context-chars` supported and use it as the default context budget.
- Do not implement runtime KV-cache pruning in this phase.
- Do not expose BudgetRAG controls in the built-in chat UI yet.
- Preserve web search, MiMo routing, dictionary mode, image mode, and all existing retrievers.
- Do not commit benchmark outputs, datasets, caches, local reports, or secrets.

## Implementation Plan

1. Add BudgetRAG context data structures and policies
   - Status: done.
   - Add context item/budget/result dataclasses.
   - Add `legacy`, `char-budget`, `per-doc-budget`, `score-density`, `sentence-trim`, and `evidence-aware`.
   - Keep token counts as documented `ceil(chars / 4)` estimates.

2. Integrate BudgetRAG into benchmark runner
   - Status: done.
   - Apply context policy after retrieval and before prompt construction.
   - Compute budget metrics even when `--skip-generation` is used.
   - Keep prompt construction through a dedicated context-based helper.

3. Add analytical KV-cache estimates
   - Status: done.
   - Estimate KV memory from reduced context token estimates.
   - Document that estimates are analytical and not measured runtime pruning.

4. Expose CLI flags
   - Status: done.
   - Add context policy/budget flags.
   - Add KV profile and disable flags.
   - Validate positive budgets and known enum values.

5. Add matrix and summary scripts
   - Status: done.
   - Add compact policy/budget matrix runner.
   - Add JSON/CSV/Markdown summary helper for local result directories.

6. Update docs
   - Status: done.
   - Add BudgetRAG docs under `docs/`.
   - Update README with usage examples and limitations.
   - Keep `pdf/` as N/A because this repo currently has no PDF source tree.

7. Validate and commit
   - Status: done.
   - Run focused tests, full tests, and smoke commands.
   - Create small Conventional Commits.
