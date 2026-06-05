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
- PDF report: `pdf/main.pdf` is generated from `pdf/main.tex` and `pdf/references.bib`; LaTeX intermediates must be cleaned after each rebuild.
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

Status: started on `feature/rlaif-retrieval-context-v0`.

- Add a normalized answer-quality and context-evidence feedback layer for BudgetRAG action rows.
- Add schema records for retrieval-context actions, answer feedback, context feedback, rewards, and preferences.
- Add deterministic action ids that include retriever, fusion strategy, context policy, budget, adaptive profile, selected context action, and generator model while excluding source run ids.
- Add `rlaif-build` to convert BudgetRAG `query_results.jsonl` outputs into `rlaif_actions.jsonl`, `rlaif_feedback.jsonl`, and `rlaif_feedback_summary.md`.
- Preserve answer text, retrieved records, context metrics, retrieval metrics, latency, token usage, KV estimates, feedback provenance, and missing-label reasons in offline datasets.
- Keep gold metrics, RAGAS/AI judge scores, missing labels, and ambiguous judge results separate.
- Harden AI judge provenance so MiMo, DeepSeek, Groq, and future judge rows use auditable provider/model fields instead of falling back to `heuristic`.
- Allow legacy/full-context action rows without an explicit budget by preserving `budget_chars=null` as a stable action dimension.
- Add explicit answer and context judge labeling paths with dry-run/resume support.
- Label minimal evidence chunks, redundant chunks, irrelevant chunks, missing evidence, and context sufficiency.
- Produce auditable feedback artifacts that can be reused by Phase 1D RLAIF.
- Do not treat missing answer feedback as zero accuracy.
- Add multi-model generation validation across Groq Llama 8B, Groq Qwen 32B, and MiMo as a token-rich/long-context upper-bound.
- Add `configs/budgetrag_models.json`, `src/rag_bench/generation_models.py`, and `scripts/run_budgetrag_generation_matrix.py` to run resumable generation matrices with provider/model metadata.
- Add a Kaggle-first HotpotQA sampled evaluation path with cached BM25 retrieval, MiMo/Groq generation support, policy/profile sharding, reference joins for EM/token-F1, and MiMo-backed RAGAS samples.
- Add HotpotQA failed-row retry tooling so quota-contaminated Groq runs can be resumed without rebuilding BM25.
- Document Phase 1C.3 generation and HotpotQA results in `docs/reports/phase1c3_multi_model_generation.md`, `docs/reports/phase1c3_mimo_long_context.md`, and `docs/reports/phase1c3_hotpotqa_kaggle_eval.md`.

## BudgetRAG Phase 1D

Status: started on `feature/rlaif-retrieval-context-v0`.

- Implement offline RLAIF reward and preference builders over Phase 1C.3 feedback.
- Include retrieval strategy, fusion strategy, context policy, budget, adaptive profile, and generator model in the action space.
- Keep answer quality and evidence support as primary reward terms while token, latency, and estimated KV costs remain bounded penalties.
- Add quality guardrails so a cheaper context policy cannot win when it clearly harms answer quality.
- Build both context-only preferences and retrieval-context preferences.
- Add `rlaif-reward` to write `rlaif_rewards.jsonl`, `rlaif_preferences.jsonl`, and `rlaif_reward_summary.md` from normalized RLAIF action/feedback files.
- Preserve missing or ambiguous feedback as `reward=null` instead of converting absent quality into score zero.
- Add pairwise preference skip reasons for missing quality, ambiguous feedback, small reward deltas, and quality guardrail failures.
- Add `rlaif-train` to write an offline `rlaif_policy.json` artifact with fixed, cheapest, best-average, `family_smoothed_best_average`, `shrinkage_smoothed_best_average`, `linear_reward_model`, `smoothed_linear_selector`, and oracle-logged selector baselines.
- Add `rlaif-eval` to report mean reward, mean quality, normalized token/latency/KV cost, selected action distribution, coverage, and oracle gap.
- Keep selector artifacts offline-only with `runtime_default_replacement=false`; they do not replace `adaptive-heuristic` in runtime defaults.
- Run a Phase 1D selector smoke on real Phase 1C.3 outputs joined with RAGAS post-hoc answer relevancy and document it in `docs/reports/phase1d_rlaif_selector_smoke.md`.
- Add `rlaif-split` for deterministic held-out splits by `benchmark + query_id`, keeping all actions for the same query in one split and dropping cross-split preferences.
- Run held-out selector evaluation with `--split-manifest` and document it in `docs/reports/phase1d_rlaif_heldout_eval.md`.
- Add `rlaif-label-answers` for resumable AI-judge answer labels over normalized action rows, with JSON repair, progress logging, incremental writes, and ambiguous/null handling for invalid or missing labels.
- Harden MiMo key loading for RLAIF labelers so private Kaggle jobs can use an injected `MIMO_API_KEY` process environment variable without requiring `.secrets/.env` in the cloned repo.
- Add `scripts/summarize_rlaif_labels.py` and `docs/reports/phase1d_rlaif_answer_labels_template.md` so MiMo/Groq/DeepSeek judge runs can be summarized as soon as outputs finish.
- Add `rlaif-reward --answer-labels` so valid AI-judge labels can replace RAGAS feedback while invalid/ambiguous labels fall back cleanly instead of becoming score zero.
- Add `scripts/estimate_local_qwen_kv_cache.py` and `docs/reports/local_qwen_kv_estimates.md` for analytical Qwen2.5 KV-cache estimates without loading model weights.
- Add `rlaif-label-contexts` for resumable context-level RLAIF labels over normalized action rows, including selected/redundant/irrelevant chunk ids, context sufficiency, missing evidence, JSON repair, progress logging, and null-score handling for invalid or missing labels.
- Add `scripts/summarize_rlaif_context_labels.py` and `docs/reports/phase1d_rlaif_context_labels_template.md` so context-label runs can be summarized for sufficiency, redundancy, missing evidence, dropped chunk-id hallucinations, and context quality.
- Run the first real MiMo context-level RLAIF subset with 50 action rows, merge two safe parallel shards, summarize sufficiency/evidence/chunk-selection labels, and document the result in `docs/reports/phase1d_rlaif_context_labels_mimo50.md`.
- Add non-default `rlaif-reward --context-labels` so clean non-ambiguous context labels can adjust reward diagnostics while ambiguous/invalid/missing context labels fall back cleanly, then document the MiMo50 context reward candidate in `docs/reports/phase1d_rlaif_context_reward_candidate.md`.
- Add context reward ablation knobs for quality/support blending and insufficient-context penalty weight, plus `scripts/compare_rlaif_reward_sets.py` to report reward-delta distributions before using context candidates for selector training.
- Add `scripts/validate_rlaif_context_labels.py` for sharded context-label merge diagnostics, including duplicate action ids, missing/unknown action ids, ambiguous/invalid labels, dropped unknown chunk ids, and deterministic merged output.
- Add `scripts/run_context_reward_ablation_pipeline.py` and `docs/reports/phase1d_rlaif_full_context_reward_ablation_template.md` so full context-label outputs can be validated, merged, summarized, rebuilt into reward candidates, compared, and selector-swept as soon as Kaggle jobs finish.
- Complete the full MiMo context-label ablation for all 192 Phase 1D action rows, merge existing labels with Kaggle shards 51-121 and 122-192, validate 177 clean usable labels with no missing/unknown/duplicate action ids, run penalty `0.25/0.50/1.00` reward ablations and six-seed selector sweeps, and document the result in `docs/reports/phase1d_rlaif_full_context_reward_ablation.md`.
- Add `scripts/select_rlaif_multijudge_audit_cases.py` to select high-impact context audit rows from MiMo insufficiency, large context-reward deltas, high answer quality with low context support, many irrelevant chunks, selector disagreement metadata, and optional pairwise disagreement labels.
- Add deterministic multi-judge audit sharding so targeted DeepSeek/Groq jobs can run in parallel without overlapping action ids, while preserving full action rows for existing `rlaif-label-contexts` consumption.
- Add `scripts/aggregate_rlaif_multijudge_audit.py` and `docs/reports/phase1d_rlaif_multijudge_audit_template.md` to summarize MiMo/DeepSeek/Groq agreement, MiMo-harsh rows, consensus-insufficient rows, and high-disagreement rows without averaging judges or changing reward defaults.
- Complete the first targeted multi-judge audit with 60 high-impact rows: DeepSeek v4 Flash completed a full 192-action secondary context audit, Groq Qwen3 32B labeled the 60-row targeted subset across local/Kaggle shards, and `docs/reports/phase1d_rlaif_multijudge_audit.md` records 51 consensus-insufficient rows plus 6 MiMo-harsh/high-disagreement rows.
- Add `docs/reports/phase1d_retriever_diversity_run_plan.md` and `scripts/run_retriever_diversity_budgetrag_matrix.sh` to plan `bm25`, `graph-bm25`, and `hybrid-rrf` logged-action coverage before claiming retrieval-strategy allocation.
- Merge the vLLM model-benchmark workflow into `internship` as an operator/profiling path, including `rag-bench model-bench`, `MODEL_BENCH.md`, Vast AI RTX 5060 Ti setup/run scripts, vLLM serving metadata, speculative-decoding sweep helpers, and hardware aggregate summaries under ignored `runs/model_bench/` outputs.
- Merge `feature/rlaif-retrieval-context-v0` into `internship` and rewrite the internship report as an English modular LaTeX report: `pdf/main.tex` is a short driver, `pdf/sections/en/` holds section files and expanded appendices, and `pdf/main.pdf` covers Phase 1A through Phase 1D with claim boundaries for offline RLAIF-style evaluation and future KV work.
- Expand the submission report with a clearer experimental setup, qualitative error analysis, threats to validity, planned HotpotQA/retriever-diversity evaluations, and an implementation map, while removing the internal English/Vietnamese translation-trace appendix from the submitted PDF.
- Add the first learned offline selector baseline, `linear_reward_model`, using non-leaking retrieval-context action/cost features and document its held-out result in `docs/reports/phase1d_rlaif_v2_linear_selector_heldout.md`.
- Add `scripts/run_rlaif_split_sweep.py` for multi-seed held-out selector evaluation and document the six-seed result in `docs/reports/phase1d_rlaif_v2_multiseed_selector_eval.md`.
- Add `scripts/inspect_rlaif_action_coverage.py` for action signature sparsity and train/eval coverage diagnostics, then document the result in `docs/reports/phase1d_rlaif_action_coverage.md`.
- Add `family_smoothed_best_average` to repair exact-signature coverage loss via retrieval-context-family and context-policy backoff, then document the six-seed result in `docs/reports/phase1d_rlaif_v2_family_smoothed_selector_eval.md`.
- Add `smoothed_linear_selector` with train-only aggregate reward features for exact signatures, retrieval-context families, context policies, and retrievers, then document the six-seed result in `docs/reports/phase1d_rlaif_v2_smoothed_linear_selector_eval.md`.
- Add `shrinkage_smoothed_best_average` to score each row with empirical-Bayes shrinkage from exact signature to retrieval-context family, context policy, and global train means, then document the six-seed result in `docs/reports/phase1d_rlaif_v2_shrinkage_selector_eval.md`.
- Add `rlaif-label-pairs` for direct pairwise AI-judge comparisons of reward-derived retrieval-context action pairs, with A/B/tie/ambiguous decisions, quality/support/efficiency winners, resume support, JSON repair, and null-score handling for invalid or missing pair data.
- Add `scripts/summarize_rlaif_pairwise_labels.py` and `docs/reports/phase1d_rlaif_pairwise_labels_template.md` so direct pairwise labels can be summarized for reward-preference agreement, disagreement, tie/ambiguous rates, confidence, quality regret, and unsupported-claim risk.
- Complete the full MiMo answer-label Kaggle run for the Phase 1D selector smoke, rebuild rewards with `rlaif-reward --answer-labels`, rerun held-out split/train/eval, and document the result in `docs/reports/phase1d_rlaif_ai_judge_heldout_eval.md`.
- Run a 50-pair direct MiMo pairwise audit over AI-judge reward-derived preferences, summarize reward/preference agreement, and document the result in `docs/reports/phase1d_rlaif_pairwise_mimo50.md`.
- Add pairwise-calibrated reward diagnostics to detect small quality/support deltas where direct pairwise judge prefers lower resource cost, and document the MiMo-50 result in `docs/reports/phase1d_rlaif_pairwise_calibration_diagnostics.md`.
- Add opt-in `pairwise_tie_v1` reward calibration for preference construction, keep scalar reward/default behavior unchanged, and document the calibrated candidate in `docs/reports/phase1d_rlaif_pairwise_calibrated_reward_candidate.md`.
- Keep larger held-out evaluation pending until answer/context labels are richer than RAGAS answer relevancy.
- Train/evaluate a lightweight offline contextual bandit/selector before considering runtime use.
