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
- Add `rlaif-train` to write an offline `rlaif_policy.json` artifact with fixed, cheapest, best-average, and oracle-logged selector baselines.
- Add `rlaif-eval` to report mean reward, mean quality, normalized token/latency/KV cost, selected action distribution, coverage, and oracle gap.
- Keep selector artifacts offline-only with `runtime_default_replacement=false`; they do not replace `adaptive-heuristic` in runtime defaults.
- Run a Phase 1D selector smoke on real Phase 1C.3 outputs joined with RAGAS post-hoc answer relevancy and document it in `docs/reports/phase1d_rlaif_selector_smoke.md`.
- Add `rlaif-split` for deterministic held-out splits by `benchmark + query_id`, keeping all actions for the same query in one split and dropping cross-split preferences.
- Run held-out selector evaluation with `--split-manifest` and document it in `docs/reports/phase1d_rlaif_heldout_eval.md`.
- Add `rlaif-label-answers` for resumable AI-judge answer labels over normalized action rows, with JSON repair, progress logging, incremental writes, and ambiguous/null handling for invalid or missing labels.
- Add `scripts/summarize_rlaif_labels.py` and `docs/reports/phase1d_rlaif_answer_labels_template.md` so MiMo/Groq/DeepSeek judge runs can be summarized as soon as outputs finish.
- Add `rlaif-reward --answer-labels` so valid AI-judge labels can replace RAGAS feedback while invalid/ambiguous labels fall back cleanly instead of becoming score zero.
- Add `scripts/estimate_local_qwen_kv_cache.py` and `docs/reports/local_qwen_kv_estimates.md` for analytical Qwen2.5 KV-cache estimates without loading model weights.
- Add `rlaif-label-contexts` for resumable context-level RLAIF labels over normalized action rows, including selected/redundant/irrelevant chunk ids, context sufficiency, missing evidence, JSON repair, progress logging, and null-score handling for invalid or missing labels.
- Add `scripts/summarize_rlaif_context_labels.py` and `docs/reports/phase1d_rlaif_context_labels_template.md` so context-label runs can be summarized for sufficiency, redundancy, missing evidence, dropped chunk-id hallucinations, and context quality.
- Add `rlaif-label-pairs` for direct pairwise AI-judge comparisons of reward-derived retrieval-context action pairs, with A/B/tie/ambiguous decisions, quality/support/efficiency winners, resume support, JSON repair, and null-score handling for invalid or missing pair data.
- Add `scripts/summarize_rlaif_pairwise_labels.py` and `docs/reports/phase1d_rlaif_pairwise_labels_template.md` so direct pairwise labels can be summarized for reward-preference agreement, disagreement, tie/ambiguous rates, confidence, quality regret, and unsupported-claim risk.
- Complete the full MiMo answer-label Kaggle run for the Phase 1D selector smoke, rebuild rewards with `rlaif-reward --answer-labels`, rerun held-out split/train/eval, and document the result in `docs/reports/phase1d_rlaif_ai_judge_heldout_eval.md`.
- Run a 50-pair direct MiMo pairwise audit over AI-judge reward-derived preferences, summarize reward/preference agreement, and document the result in `docs/reports/phase1d_rlaif_pairwise_mimo50.md`.
- Add pairwise-calibrated reward diagnostics to detect small quality/support deltas where direct pairwise judge prefers lower resource cost, and document the MiMo-50 result in `docs/reports/phase1d_rlaif_pairwise_calibration_diagnostics.md`.
- Add opt-in `pairwise_tie_v1` reward calibration for preference construction, keep scalar reward/default behavior unchanged, and document the calibrated candidate in `docs/reports/phase1d_rlaif_pairwise_calibrated_reward_candidate.md`.
- Keep larger held-out evaluation pending until answer/context labels are richer than RAGAS answer relevancy.
- Train/evaluate a lightweight offline contextual bandit/selector before considering runtime use.
