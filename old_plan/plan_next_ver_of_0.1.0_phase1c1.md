# Plan Next Version Of 0.1.0

## Goal

Complete BudgetRAG Phase 1C.1: merge Phase 1C into local `internship`, validate adaptive heuristic behavior on a larger retrieval-only SciFact slice, and document decision distributions before any Phase 1D bandit/RL-lite work.

Status: implemented and validated locally on `feature/budgetrag-phase1c1`.

## Constraints

- Do not implement RL, bandits, runtime KV-cache pruning, new retrievers, or new providers.
- Preserve default `legacy` behavior and existing chat/web search/MiMo/dictionary/image routes.
- Keep raw benchmark outputs ignored under `benchmark_results/budgetrag/`.
- Commit only curated documentation and small diagnostic/docs updates.

## Implementation Plan

1. Merge Phase 1C
   - Status: done locally.
   - Merged local `feature/budgetrag-phase1c` into local `internship`.
   - Push is blocked in this environment by missing GitHub HTTPS credentials.

2. Validate merged baseline
   - Status: done.
   - `uv sync --frozen --group dev` passed.
   - `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 uv run pytest` passed.
   - Plain `uv run pytest` remains blocked by external ROS `launch_testing` missing `lark`.
   - Legacy, fixed BudgetRAG, and adaptive retrieval-only smoke commands passed.

3. Run Phase 1C.1 matrix
   - Status: done.
   - Ran SciFact BM25 retrieval-only matrix with `limit 50`, `top-k 5`, four policies, and budgets `1000,2000,4000`.
   - Summarized ignored raw outputs with `scripts/summarize_budgetrag_results.py`.

4. Document findings
   - Status: done.
   - Added `docs/reports/phase1c1_adaptive_validation.md`.
   - Updated README with a short Phase 1C.1 validation note.
   - Updated adaptive budgeting docs with the observed conservative BM25 behavior.

5. Next decision
   - Status: pending future work.
   - If adaptive remains conservative across retrievers, do Phase 1C.2 threshold calibration.
   - If action distributions become stable and diverse enough, proceed to Phase 1D offline bandit/RL-lite.
