# Plan Next Version Of 0.1.0

## Goal

Complete BudgetRAG Phase 1C.2: preserve the conservative adaptive baseline, add calibrated deterministic profiles, record normalized score diagnostics, and validate profile behavior on a retrieval-only SciFact BM25 matrix.

Status: implemented and validated locally on `feature/budgetrag-phase1c2`.

## Constraints

- Do not implement RL, bandits, runtime KV-cache pruning, local Qwen inference, new retrievers, or new providers.
- Preserve existing `adaptive-heuristic` behavior through the default `conservative` profile.
- Keep raw benchmark outputs ignored under `benchmark_results/budgetrag/`.
- Keep chat UI, web search, MiMo, dictionary, image, and fixed context policies functionally unchanged.

## Implementation Plan

1. Add adaptive profiles
   - Status: done.
   - Added `conservative`, `balanced`, and `aggressive` profiles.
   - Kept `conservative` as the default to preserve Phase 1C behavior.
   - Added CLI support through `--adaptive-profile`.

2. Add normalized diagnostics
   - Status: done.
   - Added normalized score gap, normalized score entropy, and score confidence features.
   - Recorded profile and calibration version in per-query adaptive metadata.
   - Aggregated average/min/max normalized diagnostics.

3. Update matrix and summary tooling
   - Status: done.
   - Added `--adaptive-profiles` to the matrix script.
   - Matrix runs adaptive policies once per requested profile and ignores profiles for fixed policies.
   - Summary output includes profile, calibration version, normalized diagnostics, and adaptive count dictionaries.

4. Validate Phase 1C.2
   - Status: done.
   - Ran smoke commands for old, conservative adaptive, and balanced adaptive paths.
   - Ran profile dry-run matrix.
   - Ran SciFact BM25 retrieval-only matrix with `limit 50`, `top-k 5`, profiles `conservative,balanced,aggressive`, and budgets `1000,2000,4000`.

5. Document findings
   - Status: done.
   - Added `docs/reports/phase1c2_adaptive_threshold_calibration.md`.
   - Updated README, adaptive budgeting docs, limitations, and milestones.

## Next Decision

- Phase 1C.3: run tiny generation-mode validation if Groq quota/keys are intentionally available.
- Phase 1D: start offline bandit/RL-lite once profile metadata and fixed-policy baselines are considered stable.
