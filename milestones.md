# Milestones

## BudgetRAG Phase 1B

Status: implemented and validated on `feature/budgetrag-phase1b`; pending final review/merge.

- Add deterministic context-budget policies between retrieval and prompt construction.
- Preserve default legacy context truncation behavior for existing benchmark commands.
- Record per-query and aggregate context budget metrics.
- Add analytical KV-cache memory estimates derived from estimated context tokens.
- Add compact benchmark matrix and summary tooling for policy/budget comparisons.
- Document that this phase does not implement runtime KV-cache pruning.

## Notes

- Version bump: N/A. This repo currently has no `VERSION` or `versioning.py`.
- PDF rebuild: N/A. This repo currently has no `pdf/` source directory.
- Mindmap update: N/A. No `*.xmind`, `*.drawio`, or `*.mm` files are tracked in this repo.

## BudgetRAG Phase 1B.1

Status: implemented and validated on `feature/budgetrag-phase1b1`; pending final review/merge.

- Harden benchmark outputs with explicit experiment metadata.
- Clarify that `evidence-aware` currently means lexical/query-aware span retention.
- Make BudgetRAG summary generation cover all aggregate records.
- Add traceable matrix runs with run names, dry-run mode, continue-on-error mode, and manifests.
- Add a small committed smoke result report under `docs/reports/` while keeping raw matrix outputs ignored.

## BudgetRAG Phase 1C

Status: implemented and validated locally on `feature/budgetrag-phase1c`; pending final review/merge.

- Add `adaptive-heuristic`, a deterministic rule-based context budget selector.
- Select among existing fixed policies after retrieval without changing default legacy behavior.
- Record per-query adaptive decision metadata and aggregate adaptive counts/feature averages.
- Extend CLI, matrix, and summary tooling for adaptive budget candidates.
- Document that adaptive budgeting is not RL, not a bandit, and not runtime KV-cache pruning.

## BudgetRAG Phase 1C.1

Status: implemented and validated locally on `feature/budgetrag-phase1c1`; pending final review/merge.

- Merge Phase 1C into local `internship` and validate the merged baseline.
- Run a larger SciFact BM25 retrieval-only matrix with 50 queries and `top-k 5`.
- Document adaptive selected policy, selected budget, and reason distributions.
- Record that the BM25 validation remains conservative: all adaptive rows selected the 4000-character budget.
- Leave Phase 1D offline bandit/RL-lite for a later phase.

## BudgetRAG Phase 1C.2

Status: implemented and validated locally on `feature/budgetrag-phase1c2`; pending final review/merge.

- Add deterministic adaptive profiles: `conservative`, `balanced`, and `aggressive`.
- Preserve Phase 1C behavior through the default conservative profile.
- Add normalized score gap, normalized score entropy, and score confidence diagnostics.
- Extend matrix and summary tooling for adaptive profile comparisons.
- Run SciFact BM25 profile calibration matrix with 50 queries and document the threshold calibration results.
