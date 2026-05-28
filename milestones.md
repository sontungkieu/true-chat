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

## BudgetRAG Phase 1C.3

Status: in progress on `feature/budgetrag-phase1c3`.

- Add multi-model generation validation across Groq Llama 8B, Groq Qwen 32B, and MiMo.
- Treat MiMo as a token-rich/long-context upper-bound rather than a constrained deployment baseline.
- Record generation provider, model role, latency, estimated prompt/completion tokens, answer length, and error counts.
- Add generation matrix tooling with model config, MiMo credential skipping, dry-run, manifests, and continue-on-error support.
- Run full SciFact BM25 generation validation and MiMo long-context validation before final report.

## BudgetRAG Phase 1D Presentation

Status: presentation artifact drafted; implementation pending.

- Add a LaTeX Beamer slide deck under `pressentation/` for the Phase 1D RLAIF, offline bandit, and local Qwen/KV roadmap.
- Redesign the deck from default Beamer blocks into a custom minimal academic layout with fewer slides, diagrams, and tighter presenter-facing wording.
- Revise the deck to show RLAIF at context, answer, reward, and downstream DPO/KV touchpoints; add benchmark charts, concrete reward-term definitions, context-policy definitions, and Qwen2.5 KV estimates at 1k/16k/128k tokens.
- Add exact TurboQuant 3.5-bit KV payload math for 16k and 128k contexts, explicitly separating payload-only calculation from unmeasured runtime overhead.
- Polish slide wording and typography for a more professional, consistent academic deck; reserve smaller font sizes for dense tables and captions only.
- Add Phase 1C.3 snapshot data from `benchmark_results/budgetrag/phase1c3_snapshot_summary.md`; keep HotpotQA out of performance tables because no HotpotQA benchmark result is present.
- Use the final slide as a close-out summary of the strategy and next decisions instead of a test coverage slide.
- Reorder the deck so strategy/action space and algorithms are introduced before benchmark data and result slides.
- Keep the deck scoped to AI feedback/RLAIF-style labels, offline contextual bandit/RL-lite, and analytical/profiling-oriented KV work.
- Document boundaries in the deck: no PPO/GRPO, no human preference learning claim, no Qwen14B DPO completion claim, and no production runtime KV-cache pruning claim.
