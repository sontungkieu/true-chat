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

Status: implemented and validated locally on `feature/budgetrag-phase1c3`; pending final review/merge.

- Add multi-model generation validation across Groq Llama 8B, Groq Qwen 32B, and MiMo.
- Treat MiMo as a token-rich/long-context upper-bound rather than a constrained deployment baseline.
- Record generation provider, model role, latency, estimated prompt/completion tokens, answer length, and error counts.
- Add generation matrix tooling with model config, MiMo credential skipping, dry-run, resume-by-default behavior, per-job timeouts, manifests, and continue-on-error support.
- Run full SciFact BM25 generation validation with 50 queries for Groq Llama 8B, Groq Qwen 32B, and MiMo v2.5 Pro.
- Run MiMo long-context validation with 30 queries, `top-k 10`, and budgets up to 32000 characters.
- Document curated results in `docs/reports/phase1c3_multi_model_generation.md` and `docs/reports/phase1c3_mimo_long_context.md`.
- Record that Groq high-context adaptive cells hit provider rate limits; MiMo full and long-context cells completed without generation errors.

## BudgetRAG Phase 1D Presentation

Status: presentation artifact drafted; implementation pending.

- Add a LaTeX Beamer slide deck under `pressentation/` for the Phase 1D RLAIF, offline bandit, and local Qwen/KV roadmap.
- Redesign the deck from default Beamer blocks into a custom minimal academic layout with fewer slides, diagrams, and tighter presenter-facing wording.
- Revise the deck to show RLAIF at context, answer, reward, and downstream DPO/KV touchpoints; add benchmark charts, concrete reward-term definitions, context-policy definitions, and Qwen2.5 KV estimates at 1k/16k/128k tokens.
- Move exact TurboQuant 3.5-bit KV payload math for 16k and 128k contexts into an appendix slide titled `KV quantization payload estimate`, explicitly separating payload-only calculation from unmeasured runtime overhead.
- Polish slide wording and typography for a more professional, consistent academic deck; reserve smaller font sizes for dense tables and captions only.
- Add Phase 1C.3 snapshot data from `benchmark_results/budgetrag/phase1c3_snapshot_summary.md`; keep HotpotQA out of performance tables because no HotpotQA benchmark result is present.
- Refresh the generation result slide from `docs/reports/phase1c3_multi_model_generation.md` and `docs/reports/phase1c3_mimo_long_context.md`, including Groq Qwen3-32B and MiMo long-context aggregates.
- Use the final slide as a close-out summary of the strategy and next decisions instead of a test coverage slide.
- Reorder the deck so strategy/action space and algorithms are introduced before benchmark data and result slides.
- Add explicit bandit explanation slides covering state/action/reward/policy mapping and the offline strategies to evaluate: fixed, cheapest, best-average, oracle, epsilon-greedy replay, UCB/LinUCB, and linear reward model.
- Add an explicit RLAIF-to-signal slide showing how structured judge labels become scalar rewards for bandit learning or same-query preference pairs for ranking/DPO smoke.
- Add an `Acc / quality` slide that separates retrieval accuracy proxies from answer-quality smoke metrics and clarifies that full generation strategy accuracy still needs RAGAS metrics on the full logged matrix.
- Revise the Phase 1C.3 context slide itself to include retrieval acc (`hit@k/nDCG@k`) beside cost/KV, while marking answer acc as `N/A` until RAGAS is run post-hoc on cached answers.
- Run a MiMo-backed RAGAS post-hoc judge on 5 cached answers for each slide-14 run and fill the slide with answer-relevancy scores, while noting faithfulness timeout and context-source limitations.
- Expand the slide-14 context matrix into four full action-matrix slides covering all 64 model/action/budget rows from the full Llama, full Qwen, full MiMo, and MiMo long-context runs.
- Replace the diagnostic `n=1` RAGAS fill with a proper MiMo-backed stratified RAGAS answer-relevancy pass: 64/64 action rows, 3 valid cached answers per row, deterministic spread sampling, no noncommittal filtering, and reference-dependent metrics explicitly excluded because cached SciFact rows do not include gold free-text answers.
- Add a Kaggle-first HotpotQA eval path for Phase 1C.3: build BM25 once, cache retrieval, replay the 16 MiMo context-policy action rows, join `hotpotqa/hotpot_qa` references for EM/token-F1, and run MiMo-backed RAGAS samples per action. Local full matrix remains out of scope because HotpotQA BM25 indexing is too heavy for iterative local runs.
- Extend the HotpotQA Kaggle path to support Groq generation with a single injected key, including `qwen/qwen3-32b` smoke runs, while keeping RAGAS judging on MiMo via a separate `--ragas-model` setting.
- Add Kaggle policy-sharding flags for HotpotQA (`--context-policies`, `--context-budgets`, `--adaptive-profiles`) so MiMo jobs can be distributed across multiple accounts by fixed policy and adaptive profile.
- Keep the deck scoped to AI feedback/RLAIF-style labels, offline contextual bandit/RL-lite, and analytical/profiling-oriented KV work.
- Document boundaries in the deck: no PPO/GRPO, no human preference learning claim, no Qwen14B DPO completion claim, and no production runtime KV-cache pruning claim.
