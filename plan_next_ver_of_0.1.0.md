# Plan Next Version Of 0.1.0

## Goal

Implement RLAIF for BudgetRAG on branch `internship` without breaking the Phase 1B/1C context-budgeting baseline.

The work is split into two connected phases:

- **Phase 1C.3: answer/context feedback layer.** Produce reliable per-action answer feedback and context-evidence feedback from gold metrics, RAGAS/MiMo judges, and existing BudgetRAG matrix outputs.
- **Phase 1D: RLAIF retrieval-context policy layer.** Convert that feedback into scalar reward rows and pairwise preference rows, then use them to train and evaluate a lightweight offline contextual bandit/selector.

## Current Phase Alignment

| Phase | Status on `internship` | Role |
| --- | --- | --- |
| Phase 1B | Implemented | Fixed context-budget policies and KV/efficiency metrics. |
| Phase 1B.1 | Implemented | Traceable matrix outputs and reportable benchmark summaries. |
| Phase 1C | Implemented | Deterministic `adaptive-heuristic` selector. |
| Phase 1C.1 | Implemented | Larger retrieval-only validation for adaptive behavior. |
| Phase 1C.2 | Implemented | Calibrated adaptive profiles and normalized retrieval diagnostics. |
| Phase 1C.3 | Next required foundation | Generation/judge feedback plus context-evidence labels so actions can be compared by quality, sufficiency, and efficiency. |
| Phase 1D | Main target | RLAIF data builder and offline retrieval-context allocation policy. |

## Why Phase 1C.3 Must Come Before Phase 1D

RLAIF needs a feedback signal. The existing Phase 1C.2 data mostly explains **which context policy was selected** and **how efficient it was**. It does not yet provide enough comparable answer-quality or context-evidence labels across action rows.

Phase 1C.3 should therefore standardize both answer-quality feedback and context-evidence feedback first:

- Same query, same retriever, multiple BudgetRAG actions.
- Each action has answer text, retrieved/context documents, context metrics, latency, token/KV estimates, and quality feedback.
- Quality feedback may come from gold answers when available, RAGAS/MiMo judge scores when gold is missing, or explicit insufficiency labels when neither is reliable.
- Context feedback should judge the retrieved/context set before answer generation. It should identify the minimal evidence subset, redundant chunks, irrelevant chunks, missing evidence, and context sufficiency. These labels will be used both for reward construction and future evidence-token masks for a KV-cache pruning proof of concept.

Only after that can Phase 1D safely ask: "Which retrieval-context action should the system prefer for this query?"

## Phase 1C.3 Implementation Plan: Feedback Layer

1. Normalize action rows
   - Status: dataset builder implemented on `feature/rlaif-retrieval-context-v0`.
   - Read one or more `query_results.jsonl` files from BudgetRAG runs.
   - Extract stable keys: benchmark, query id, question, retrieval strategy, fusion strategy, top-k, context policy, optional budget, adaptive profile, selected adaptive action, generator model, answer, references, context metrics, latency, and token usage.
   - Assign an `action_id` that is deterministic across runs.
   - Represent actions with both retrieval and context dimensions:

```json
{
  "retrieval_strategy": "bm25",
  "fusion_strategy": null,
  "context_policy": "evidence-aware",
  "budget_chars": 2000,
  "adaptive_profile": null,
  "generator_model": "mimo-v2.5-pro"
}
```

2. Normalize feedback sources
   - Status: dataset builder implemented for gold, existing RAGAS fields, existing AI judge fields, and missing-label reasons.
   - Use gold metrics when present: exact match and token F1.
   - Use RAGAS/AI judge fields when present: answer relevancy, faithfulness, answer correctness, and judge rationale.
   - Record feedback provenance explicitly: `gold`, `ragas`, `ai_judge`, `mimo_judge` for backward compatibility, `heuristic`, or `missing`.
   - Store concrete AI judge identity in `judge_provider` and `judge_model` so MiMo, DeepSeek, and Groq rows are auditable without being mislabeled as heuristic.
   - Do not silently treat missing feedback as zero accuracy.

   Note: full-context or legacy baseline rows without an explicit context budget are valid and use `budget_chars: null` in the action identity payload.

3. Add answer-labeling when feedback is absent
   - Status: implemented on `feature/rlaif-retrieval-context-v0`.
   - If judge fields are not present, implement an explicit answer-labeling path instead of leaving everything as `missing`.
   - The labeling path must support `--dry-run`, `--resume`, `--limit`, `--max-errors`, `--judge-provider`, and `--judge-model`.
   - It should write incrementally so a long MiMo judge run can resume safely.
   - Invalid JSON, empty completions, missing answers, and missing contexts become ambiguous labels with `quality_score: null`, never score zero.
   - MiMo V2.5 can spend hidden reasoning tokens before emitting JSON, so the default answer-judge completion budget is `4096`.

```bash
uv run rag-bench rlaif-label-answers \
  --actions benchmark_results/rlaif/<run-name>/rlaif_actions.jsonl \
  --judge-provider mimo \
  --judge-model mimo-v2.5-pro \
  --output benchmark_results/rlaif/<run-name>/rlaif_answer_labels.jsonl \
  --limit 50 \
  --resume
```

Summarize answer labels and compare with RAGAS answer relevancy:

```bash
uv run python scripts/summarize_rlaif_labels.py \
  --labels benchmark_results/rlaif/<run-name>/rlaif_answer_labels.jsonl \
  --ragas-feedback benchmark_results/rlaif/<run-name>/rlaif_feedback.jsonl \
  --out-md benchmark_results/rlaif/<run-name>/rlaif_answer_labels_summary.md \
  --out-json benchmark_results/rlaif/<run-name>/rlaif_answer_labels_summary.json
```

4. Add context-level RLAIF feedback
   - Status: schema baseline and `rlaif-label-contexts` CLI implemented; first real MiMo subset completed on 50 action rows and documented in `docs/reports/phase1d_rlaif_context_labels_mimo50.md`.
   - Judge candidate retrieved/context chunks before generation.
   - Identify selected evidence chunks, redundant chunks, irrelevant chunks, missing evidence, and sufficiency.
   - The labeling path supports `--dry-run`, `--resume`, `--limit`, `--max-errors`, `--judge-provider`, `--judge-model`, JSON repair, progress logging, and incremental writes.
   - MiMo key loading now prefers the process environment before local `--env-file`, so private Kaggle jobs can inject `MIMO_API_KEY` without creating `.secrets/.env` in the cloned repo.
   - Missing contexts, invalid JSON, empty completions, and judge errors become ambiguous labels with null scores, never score zero.
   - Summarize context labels with `scripts/summarize_rlaif_context_labels.py` to track sufficiency, missing evidence, selected/redundant/irrelevant chunk counts, dropped unknown chunk ids, and context quality statistics.
   - Current subset result: 50 valid JSON labels, 0 invalid JSON, 0 errors, 4 ambiguous rows, 18 sufficient rows, 29 insufficient rows, mean context quality 0.478, mean evidence support 0.410, and mean selected chunks 1.38.
   - Implemented non-default `rlaif-reward --context-labels` merge path and filter `ambiguous=true` rows out of clean context supervision.
   - Current context reward candidate result: clean context labels used 46/50; ambiguous context rows fallback 4/50; context labels changed 36 reward rows, with 31 reward decreases and 5 increases; preference count rose from 722 to 822 because context evidence labels sharpened some action-pair gaps.
   - Added ablation knobs: `--context-quality-blend-weight`, `--context-support-blend-weight`, and `--context-insufficient-penalty-weight`.
   - Added `scripts/compare_rlaif_reward_sets.py` to report reward delta distribution, clipped reward counts, and changed rows by context sufficiency.
   - MiMo50 penalty-weight ablation: insufficient penalty `0.25/0.50/1.00` keeps 36 changed rows but changes mean changed-only reward delta from `-0.397` to `-0.558` to `-0.880`, confirming that weight `1.0` is aggressive.
   - Added `scripts/validate_rlaif_context_labels.py` to validate and dedupe parallel context-label shards before reward construction.
   - Added `scripts/run_context_reward_ablation_pipeline.py` to orchestrate full-label postprocess: validate/merge, summarize, rebuild answer-only baseline, rebuild context reward candidates, compare reward deltas, and optionally run multi-seed selector sweeps.
   - Next implementation change: label the remaining 142 action rows, rerun reward-delta ablations, then run multi-seed held-out selector evaluation with full context-label coverage.
   - Build context preference pairs between full, evidence-aware, aggressive, fixed-budget, and adaptive contexts.
   - Keep this independent from answer scoring so the system can learn context allocation directly.

4b. Add direct pairwise RLAIF labels
   - Status: `rlaif-label-pairs` CLI implemented; a 50-pair MiMo audit has been run and documented in `docs/reports/phase1d_rlaif_pairwise_mimo50.md`.
   - Use `rlaif_preferences.jsonl` only as a source of comparable action pairs.
   - Present Action A as the reward-derived chosen action and Action B as the rejected action, but instruct the judge to decide independently.
   - Judge only from logged question, answers, retrieved contexts, and resource costs; do not browse or use external knowledge.
   - Output `chosen=A|B|null`, `tie`, `ambiguous`, answer-quality winner, evidence-support winner, efficiency winner, quality-regret flag, unsupported-claim risk, confidence, and short rationale.
   - Invalid JSON, missing actions, missing rewards, missing answers, and missing contexts become ambiguous labels with null confidence, never score zero.
   - Summarize pairwise labels with `scripts/summarize_rlaif_pairwise_labels.py` to track A/B/tie/ambiguous counts, reward-preference agreement, disagreement, confidence, quality regret, and unsupported-claim risk.
   - This is for reward/preference calibration and selector analysis only; do not train DPO/reward model or replace runtime defaults in v1.
   - Initial MiMo-50 audit result: 50 valid JSON labels, 2 ambiguous timeout labels, 41 A wins, 7 B wins, 0 ties, and 0.854 agreement over non-ambiguous decisions.
   - The observed disagreements cluster around cases where scalar reward prefers slightly higher quality/support scores while the direct judge treats both answers as acceptable or correct abstentions and then favors lower resource cost.
   - Pairwise-calibrated diagnostics are implemented in `scripts/diagnose_rlaif_pairwise_calibration.py`.
   - Initial diagnostic result with candidate thresholds `quality=0.10` and `support=0.20`: 38 small quality/support delta pairs, 35 cheaper-wins-when-tied, and 5 scalar-over-quality disagreements, all in `query_id=128`.
   - Opt-in `pairwise_tie_v1` reward calibration is implemented for preference construction. Default remains `none`; scalar reward rows are unchanged. Initial calibrated candidate creates 1270 preferences, including 900 `pairwise_tie_v1_efficiency` preferences, and is documented in `docs/reports/phase1d_rlaif_pairwise_calibrated_reward_candidate.md`.

```bash
uv run rag-bench rlaif-label-pairs \
  --actions benchmark_results/rlaif/<run-name>/rlaif_actions.jsonl \
  --rewards benchmark_results/rlaif/<run-name>/rlaif_rewards.jsonl \
  --preferences benchmark_results/rlaif/<run-name>/rlaif_preferences.jsonl \
  --output benchmark_results/rlaif/<run-name>/rlaif_pairwise_labels_mimo.jsonl \
  --judge-provider mimo \
  --judge-model mimo-v2.5-pro \
  --limit 50 \
  --resume \
  --sleep-seconds 0.5
```

```bash
uv run rag-bench rlaif-label-contexts \
  --actions benchmark_results/rlaif/<run-name>/rlaif_actions.jsonl \
  --judge-provider mimo \
  --judge-model mimo-v2.5-pro \
  --output benchmark_results/rlaif/<run-name>/rlaif_context_labels.jsonl \
  --limit 50 \
  --resume

uv run python scripts/summarize_rlaif_context_labels.py \
  --labels benchmark_results/rlaif/<run-name>/rlaif_context_labels.jsonl \
  --out-md benchmark_results/rlaif/<run-name>/rlaif_context_labels_summary.md \
  --out-json benchmark_results/rlaif/<run-name>/rlaif_context_labels_summary.json
```

Context label schema:

```json
{
  "query_id": "...",
  "action_id": "...",
  "sufficient": true,
  "selected_chunk_ids": ["doc-1", "doc-3"],
  "redundant_chunk_ids": ["doc-2"],
  "irrelevant_chunk_ids": ["doc-5"],
  "missing_evidence": false,
  "minimality_score": 0.8,
  "evidence_support_score": 0.9,
  "context_quality_score": 0.85,
  "judge_provider": "mimo",
  "judge_model": "mimo-v2.5-pro",
  "provenance": "ai_judge"
}
```

5. Add validation guards
   - Reject comparisons when query ids do not match.
   - Reject comparisons when answers are missing or generation failed.
   - Mark feedback as `ambiguous` when judge output is invalid, conflicting, or below confidence threshold.
   - Keep raw judge rationale separate from model inputs so it can be audited later.

6. Outputs
   - Status: `rlaif-build` writes the first three files below; context labeling outputs remain pending.
   - `rlaif_actions.jsonl`: one normalized candidate action per query/run.
   - `rlaif_feedback.jsonl`: one feedback record per candidate action.
   - `rlaif_answer_labels.jsonl`: explicit judge labels for answers when source runs do not already contain judge fields.
   - `rlaif_pairwise_labels.jsonl`: direct pairwise AI-judge labels for reward/preference calibration.
   - `rlaif_context_labels.jsonl`: context-level RLAIF labels per action.
   - `rlaif_context_preferences.jsonl`: context sufficiency/minimality preference rows.
   - `rlaif_feedback_summary.md`: coverage, missing-label reasons, quality distributions, and judge source counts.

## Phase 1D Implementation Plan: RLAIF Reward And Preference Layer

1. Build scalar rewards
   - Status: dataset-level reward builder implemented on `feature/rlaif-retrieval-context-v0`.
   - Status: optional `--answer-labels` merge path implemented; valid AI-judge labels override original feedback, while invalid/ambiguous/missing labels fall back to original feedback when available and never become score zero.
   - Convert quality, efficiency, and latency into a bounded scalar reward.
   - Default priority: answer quality dominates efficiency.
   - Implemented default formula:
     - `quality = token_f1` when gold exists.
     - Else `quality = weighted judge score` from answer correctness, answer relevancy, and faithfulness.
     - `evidence_support = context evidence support score` when context labels exist.
     - `token_cost_norm = bounded normalized kept/requested token cost`.
     - `latency_norm = bounded answer/query latency cost`.
     - `kv_cost_norm = bounded estimated KV-cache cost`.
     - `error_penalty = 1.0` for generation or judge errors, otherwise `0.0`.
     - `unsupported_claim_penalty = 1.0` when judge detects unsupported claims, otherwise `0.0`.
     - `reward = w_quality * quality + w_support * evidence_support - w_token * token_cost_norm - w_latency * latency_norm - w_kv * kv_cost_norm - w_error * error_penalty - w_unsupported * unsupported_claim_penalty`.
   - Proposed default weights:
     - `w_quality = 0.75`
     - `w_support = 0.10`
     - `w_token = 0.05`
     - `w_latency = 0.05`
     - `w_kv = 0.05`
     - `w_error = 1.0`
     - `w_unsupported = 1.0`
   - Store all components, weights, provenance, and `reward_mode` in every reward row.
   - Missing or ambiguous quality writes a reward row with `reward = null` and `reward_mode = missing_quality` or `ambiguous_feedback`; it is not converted to score zero.
   - Do not use KV savings as the only positive efficiency reward. Very short contexts must still lose when quality or support drops.

2. Build pairwise preferences
   - Status: dataset-level preference builder implemented on `feature/rlaif-retrieval-context-v0`.
   - Build two preference sets:
     - `context_policy_preference`: group by benchmark + query id + retriever + top-k + generator model, then compare context policies/budgets inside the same retriever.
     - `retrieval_context_preference`: group by benchmark + query id + top-k + generator model, then compare retrieval strategy + context policy combinations across retrievers.
   - Compare only actions that answered the same query under comparable generation settings.
   - Prefer the higher reward action only when the reward gap exceeds a threshold.
   - Refuse to create a preference if the higher-efficiency action has a meaningful quality regression.
   - Store:
     - preference type
     - chosen action
     - rejected action
     - reward gap
     - quality gap
     - efficiency gap
     - reason code

3. Train the first offline policy
   - Status: offline selector baselines implemented on `feature/rlaif-retrieval-context-v0`.
   - V1 should be lightweight and auditable, not PPO/DPO fine-tuning.
   - Implemented baselines:
     - `fixed`: choose the most common scored action signature when it is available for a query group.
     - `cheapest`: choose the lowest normalized token + latency + KV cost in the logged query group.
     - `best_average`: choose the available action signature with the highest training mean reward.
     - `family_smoothed_best_average`: back off from exact signature mean reward to retrieval-context-family and context-policy mean reward.
     - `shrinkage_smoothed_best_average`: score each row with empirical-Bayes shrinkage from exact signature to retrieval-context family, context policy, and global train means.
     - `linear_reward_model`: learned offline ridge-regression selector over retrieval-context action/cost features, excluding reward/quality/support labels and preference outcomes.
     - `smoothed_linear_selector`: learned offline ridge-regression selector with train-only aggregate reward means/counts for exact signatures, retrieval-context families, context policies, and retrievers.
     - `oracle_logged`: offline upper bound that chooses the highest observed reward in each logged query group.
   - Output `rlaif_policy.json` as an offline selector artifact before any runtime use.
   - Keep pairwise ranker, contextual bandit table, and LinUCB as later steps once the reward dataset is large enough.

4. Evaluate against existing baselines
   - Status: offline selector evaluator implemented on `feature/rlaif-retrieval-context-v0`.
   - Status: Phase 1D selector smoke completed and documented in `docs/reports/phase1d_rlaif_selector_smoke.md`.
   - Status: held-out query split/eval implemented and documented in `docs/reports/phase1d_rlaif_heldout_eval.md`.
   - Status: full MiMo answer-label run completed on Kaggle, rewards rebuilt with `--answer-labels`, held-out split/train/eval rerun, and documented in `docs/reports/phase1d_rlaif_ai_judge_heldout_eval.md`.
   - Status: first learned offline selector baseline implemented and documented in `docs/reports/phase1d_rlaif_v2_linear_selector_heldout.md`.
   - Status: multi-seed held-out selector sweep implemented and documented in `docs/reports/phase1d_rlaif_v2_multiseed_selector_eval.md`.
   - Status: action coverage/signature sparsity diagnostics implemented and documented in `docs/reports/phase1d_rlaif_action_coverage.md`.
   - Status: family-smoothed selector baseline implemented and documented in `docs/reports/phase1d_rlaif_v2_family_smoothed_selector_eval.md`.
   - Status: smoothed linear selector baseline implemented and documented in `docs/reports/phase1d_rlaif_v2_smoothed_linear_selector_eval.md`.
   - Status: shrinkage-smoothed selector baseline implemented and documented in `docs/reports/phase1d_rlaif_v2_shrinkage_selector_eval.md`.
   - Compare learned policy against:
     - `legacy`
     - fixed budget policies
     - `evidence-aware`
     - `adaptive-heuristic` conservative/balanced/aggressive
   - Required metrics:
     - answer quality: EM/F1 or judge quality
     - efficiency: kept tokens/chars, compression, KV savings
     - latency
     - preference win rate
     - abstention/ambiguous rate
   - Implemented first-pass metrics:
     - mean reward
     - mean quality
     - normalized token, latency, and KV cost
     - selected action distribution
     - scored coverage and selection coverage
     - oracle gap
   - The learned policy must not replace `adaptive-heuristic` by default until it beats baseline under quality guardrails.
   - The learned RLAIF/bandit policy must not replace `adaptive-heuristic` as the default runtime policy until it passes offline evaluation and quality guardrails.
   - Current smoke caveat: the first selector smoke used the same RAGAS-joined logged rewards for train/eval, so it is a resubstitution/offline sanity check rather than held-out generalization.
   - Current held-out result: `rlaif-split` writes deterministic `benchmark + query_id` train/eval files and `rlaif-eval --split-manifest` records `held_out_query_eval=true`.
   - Current multi-seed result: `linear_reward_model` beats `cheapest` on average reward and oracle gap while keeping full coverage, but does not consistently beat `best_average` across seeds.
   - Current action coverage result: exact signatures cover about 0.911 held-out eval groups, matching `best_average` coverage; collapsed retrieval-context families cover 1.000, suggesting family-level backoff before a pairwise ranker.
   - Current family-smoothed result: coverage reaches 1.000 and oracle gap improves over `best_average`, but mean reward/quality still trail exact-signature `best_average`.
   - Current smoothed-linear result: coverage stays 1.000 and reward/oracle gap improve slightly over `linear_reward_model`, but it still does not beat `best_average` or `family_smoothed_best_average` on the six-seed mean.
   - Current shrinkage-smoothed result: coverage stays 1.000, reward improves to 0.602, quality to 0.773, and oracle gap to 0.070; this is the strongest full-coverage non-oracle baseline so far, but still trails `best_average` on reward/quality.
   - Pairwise calibration limitation: `pairwise_tie_v1` currently changes preference construction and diagnostics, but reward-based selectors still train on scalar reward rows; pairwise preferences affect selection only after a pairwise ranker or calibrated scalar reward path is added.
   - Cost-feature limitation: current selector cost features are logged/offline normalized costs; runtime deployment needs estimated token/KV costs and predicted latency features before pre-generation selection.
   - Next evaluator change: merge context-level labels into non-default reward diagnostics, expand logged query groups and retriever diversity, then add a pairwise ranker.

5. Outputs
   - Status: `rlaif-reward`, `rlaif-train`, `rlaif-eval`, and `rlaif-label-contexts` write the implemented files below; a 50-action real context-label subset and non-default context reward candidate are complete, while a full 192-action context-label run remains pending.
   - `rlaif_rewards.jsonl`
   - `rlaif_preferences.jsonl`
   - `rlaif_reward_summary.md`
   - `rlaif_context_preferences.jsonl`
   - `split_manifest.json`: deterministic query-level split manifest.
   - `split_summary.md`: query-level split summary.
   - `rlaif_policy.json`: fixed, cheapest, best-average, `family_smoothed_best_average`, `shrinkage_smoothed_best_average`, `linear_reward_model`, `smoothed_linear_selector`, and oracle-logged offline selector baselines.
   - `rlaif_eval_summary.md`: selector reward/quality/cost/coverage/oracle-gap report.
   - `selector_sweep_summary.md`: multi-seed selector mean/std report.
   - `rlaif_action_coverage.md`: action signature sparsity and split coverage diagnostics.
   - `rlaif_answer_labels_summary.md`: judge label coverage, score distribution, and RAGAS correlation summary.
   - `rlaif_context_labels_mimo50_summary.md`: context-label sufficiency, evidence support, minimality, and chunk-selection summary.
   - `rlaif_reward_summary.md` with `--context-labels`: non-default context reward candidate summary with clean/fallback context-label merge counts.
   - `reward_delta_summary.md`: reward-delta distribution comparing answer-only and context-label candidates.
   - `rlaif_pairwise_labels.jsonl`: direct pairwise AI-judge labels for reward-derived action pairs.
   - `rlaif_pairwise_labels_summary.md`: direct pairwise label agreement and risk summary.
   - `docs/reports/phase1d_rlaif_selector_smoke.md`: curated smoke report over real Phase 1C.3 outputs joined with RAGAS answer relevancy.
   - `docs/reports/phase1d_rlaif_heldout_eval.md`: curated held-out query eval report.
   - `docs/reports/phase1d_rlaif_ai_judge_heldout_eval.md`: curated held-out query eval report after full MiMo answer labels and `rlaif-reward --answer-labels`.
   - `docs/reports/phase1d_rlaif_v2_linear_selector_heldout.md`: curated held-out query eval report for the first learned offline selector baseline.
   - `docs/reports/phase1d_rlaif_v2_multiseed_selector_eval.md`: curated multi-seed selector robustness report.
   - `docs/reports/phase1d_rlaif_action_coverage.md`: curated action coverage and signature sparsity report.
   - `docs/reports/phase1d_rlaif_v2_family_smoothed_selector_eval.md`: curated family-smoothed selector report.
   - `docs/reports/phase1d_rlaif_v2_smoothed_linear_selector_eval.md`: curated smoothed-linear selector report.
   - `docs/reports/phase1d_rlaif_v2_shrinkage_selector_eval.md`: curated shrinkage-smoothed selector report.
   - `docs/reports/phase1d_rlaif_context_labels_mimo50.md`: curated first real context-level RLAIF subset report.
   - `docs/reports/phase1d_rlaif_context_reward_candidate.md`: curated non-default reward candidate report after merging clean MiMo50 context labels.
   - `docs/reports/local_qwen_kv_estimates.md`: analytical Qwen2.5 KV-cache memory table for local deployment planning.
   - `docs/reports/phase1d_rlaif_context_labels_template.md`: context-label report template for sufficiency, redundancy, missing evidence, dropped unknown chunk ids, and context quality.
   - `docs/reports/phase1d_rlaif_pairwise_labels_template.md`: pairwise-label report template for reward-preference agreement, disagreement examples, quality regret, and unsupported-claim risk.
   - Optional CSV summary for slides/reports.

## Non-Blocking KV/Qwen Scaffold

Phase 1D should not implement full KV pruning, but it should leave a clear path for BudgetRAG/MemAlign-Qwen reporting:

- Add a roadmap document such as `docs/local_qwen_kv_plan.md`.
- Add an experiment placeholder such as `experiments/kv_pruning/README.md` if experiment directories are used.
- Keep estimated KV metrics in reward rows as `kv_cost_norm`, not as unchecked positive savings.
- Treat context-level RLAIF labels as future evidence-token masks for a KV-cache pruning proof of concept.

## CLI And File Layout

Current reward/preference CLI:

```bash
uv run rag-bench rlaif-reward \
  --actions benchmark_results/rlaif/<run-name>/rlaif_actions.jsonl \
  --feedback benchmark_results/rlaif/<run-name>/rlaif_feedback.jsonl \
  --answer-labels benchmark_results/rlaif/<run-name>/rlaif_answer_labels.jsonl \
  --output-dir benchmark_results/rlaif/<run-name> \
  --quality-weight 0.75 \
  --support-weight 0.10 \
  --token-weight 0.05 \
  --latency-weight 0.05 \
  --kv-weight 0.05 \
  --min-reward-delta 0.03 \
  --max-quality-regret 0.02
```

Answer labeling:

```bash
uv run rag-bench rlaif-label-answers \
  --actions benchmark_results/rlaif/<run-name>/rlaif_actions.jsonl \
  --judge-provider mimo \
  --judge-model mimo-v2.5-pro \
  --output benchmark_results/rlaif/<run-name>/rlaif_answer_labels.jsonl \
  --resume
```

Context labeling:

```bash
uv run rag-bench rlaif-label-contexts \
  --actions benchmark_results/rlaif/<run-name>/rlaif_actions.jsonl \
  --judge-provider mimo \
  --judge-model mimo-v2.5-pro \
  --output benchmark_results/rlaif/<run-name>/rlaif_context_labels.jsonl \
  --resume

uv run python scripts/summarize_rlaif_context_labels.py \
  --labels benchmark_results/rlaif/<run-name>/rlaif_context_labels.jsonl \
  --out-md benchmark_results/rlaif/<run-name>/rlaif_context_labels_summary.md \
  --out-json benchmark_results/rlaif/<run-name>/rlaif_context_labels_summary.json
```

Pairwise preference labeling:

```bash
uv run rag-bench rlaif-label-pairs \
  --actions benchmark_results/rlaif/<run-name>/rlaif_actions.jsonl \
  --rewards benchmark_results/rlaif/<run-name>/rlaif_rewards.jsonl \
  --preferences benchmark_results/rlaif/<run-name>/rlaif_preferences.jsonl \
  --output benchmark_results/rlaif/<run-name>/rlaif_pairwise_labels_mimo.jsonl \
  --judge-provider mimo \
  --judge-model mimo-v2.5-pro \
  --limit 50 \
  --resume \
  --sleep-seconds 0.5

uv run python scripts/summarize_rlaif_pairwise_labels.py \
  --input benchmark_results/rlaif/<run-name>/rlaif_pairwise_labels_mimo.jsonl \
  --out-md benchmark_results/rlaif/<run-name>/rlaif_pairwise_labels_mimo_summary.md \
  --out-json benchmark_results/rlaif/<run-name>/rlaif_pairwise_labels_mimo_summary.json
```

After the current Kaggle answer-label job completes, use this auditable path before writing a curated report:

```bash
uv run python scripts/summarize_rlaif_labels.py \
  --labels benchmark_results/rlaif/<run-name>/rlaif_answer_labels_mimo.jsonl \
  --ragas-feedback benchmark_results/rlaif/<run-name>/rlaif_feedback.jsonl \
  --out-md benchmark_results/rlaif/<run-name>/rlaif_answer_labels_mimo_summary.md \
  --out-json benchmark_results/rlaif/<run-name>/rlaif_answer_labels_mimo_summary.json

uv run rag-bench rlaif-reward \
  --actions benchmark_results/rlaif/<run-name>/rlaif_actions.jsonl \
  --feedback benchmark_results/rlaif/<run-name>/rlaif_feedback.jsonl \
  --answer-labels benchmark_results/rlaif/<run-name>/rlaif_answer_labels_mimo.jsonl \
  --output-dir benchmark_results/rlaif/<run-name>_ai_judge

uv run rag-bench rlaif-split \
  --rewards benchmark_results/rlaif/<run-name>_ai_judge/rlaif_rewards.jsonl \
  --preferences benchmark_results/rlaif/<run-name>_ai_judge/rlaif_preferences.jsonl \
  --output-dir benchmark_results/rlaif/<run-name>_ai_judge/split_seed42 \
  --train-ratio 0.8 \
  --seed 42

uv run rag-bench rlaif-train \
  --rewards benchmark_results/rlaif/<run-name>_ai_judge/split_seed42/train_rewards.jsonl \
  --preferences benchmark_results/rlaif/<run-name>_ai_judge/split_seed42/train_preferences.jsonl \
  --output benchmark_results/rlaif/<run-name>_ai_judge/split_seed42/rlaif_policy.json

uv run rag-bench rlaif-eval \
  --rewards benchmark_results/rlaif/<run-name>_ai_judge/split_seed42/eval_rewards.jsonl \
  --policy benchmark_results/rlaif/<run-name>_ai_judge/split_seed42/rlaif_policy.json \
  --split-manifest benchmark_results/rlaif/<run-name>_ai_judge/split_seed42/split_manifest.json \
  --out-md benchmark_results/rlaif/<run-name>_ai_judge/split_seed42/rlaif_eval_summary.md
```

Offline selector baselines:

```bash
uv run rag-bench rlaif-split \
  --rewards benchmark_results/rlaif/<run-name>/rlaif_rewards.jsonl \
  --preferences benchmark_results/rlaif/<run-name>/rlaif_preferences.jsonl \
  --output-dir benchmark_results/rlaif/<run-name>/split_seed42 \
  --train-ratio 0.8 \
  --seed 42

uv run rag-bench rlaif-train \
  --rewards benchmark_results/rlaif/<run-name>/split_seed42/train_rewards.jsonl \
  --preferences benchmark_results/rlaif/<run-name>/split_seed42/train_preferences.jsonl \
  --output benchmark_results/rlaif/<run-name>/split_seed42/rlaif_policy.json

uv run rag-bench rlaif-eval \
  --rewards benchmark_results/rlaif/<run-name>/split_seed42/eval_rewards.jsonl \
  --policy benchmark_results/rlaif/<run-name>/split_seed42/rlaif_policy.json \
  --out-md benchmark_results/rlaif/<run-name>/split_seed42/rlaif_eval_summary.md \
  --split-manifest benchmark_results/rlaif/<run-name>/split_seed42/split_manifest.json
```

Default output directory:

```text
benchmark_results/rlaif/<timestamp>/
```

Raw benchmark matrices remain ignored. Small synthetic fixtures and compact summary reports may be committed.

## Test Plan

1. Unit tests
   - Parse BudgetRAG `query_results.jsonl` into normalized action records.
   - Extract gold, RAGAS, and MiMo judge feedback without confusing missing labels with zero scores.
   - Build scalar rewards with stable, bounded values.
   - Build context-only pairwise preferences only inside the same query/retriever group.
   - Build retrieval-context preferences across retrievers for the same query/model.
   - Train fixed, cheapest, best-average, `family_smoothed_best_average`, `shrinkage_smoothed_best_average`, `linear_reward_model`, `smoothed_linear_selector`, and oracle-logged offline selector baselines.
   - Split reward/preference rows by `benchmark + query_id` so action rows for the same query cannot leak across train/eval.
   - Evaluate selector reward, quality, cost, coverage, selected-action distribution, and oracle gap.
   - Enforce quality guardrails so efficiency cannot win over a clearly worse answer.
   - Enforce context sufficiency guardrails so minimality cannot win over missing evidence.
   - Reject ambiguous or invalid judge rows.

2. CLI tests
   - `rag-bench rlaif-build` writes all expected files on a tiny fixture.
   - `rag-bench rlaif-label-answers --dry-run` and `rag-bench rlaif-label-contexts --dry-run` write valid placeholder labels without network calls.
   - `rag-bench rlaif-label-contexts` filters unknown judge-returned chunk ids and preserves null scores for missing/ambiguous labels.
   - `rag-bench rlaif-label-pairs --dry-run` writes ambiguous placeholder labels without network calls.
   - `rag-bench rlaif-label-pairs` preserves null confidence for missing/ambiguous pair data and records A/B/tie decisions without trusting reward-derived preferences.
   - `scripts/summarize_rlaif_pairwise_labels.py` reports reward-preference agreement, disagreement, ties, ambiguity, confidence, quality regret, and unsupported-claim risk.
   - `scripts/summarize_rlaif_context_labels.py` reports sufficiency, missing evidence, chunk selection counts, dropped unknown chunk ids, and score statistics.
   - Labeling commands support `--resume` without duplicating completed action ids.
   - Invalid inputs fail with actionable errors.
   - Output records do not include secrets or provider API keys.

3. Integration smoke
   - Run a tiny local matrix with at least two context actions and mocked generation/judge feedback.
   - Build rewards/preferences.
   - Confirm every preference points to existing action ids.

4. Documentation checks
   - Update `README.md` when code lands.
   - Update `milestones.md`.
   - PDF rebuild: N/A unless a `pdf/` directory exists on the branch at implementation time.
   - Mindmap update: N/A unless a tracked `*.xmind`, `*.drawio`, or `*.mm` file appears.

## Non-Goals For This Implementation

- No PPO, DPO, or model weight fine-tuning in the first RLAIF commit.
- No runtime replacement of `adaptive-heuristic` before offline evaluation.
- No HotpotQA full benchmark in this phase.
- No Kaggle deployment unless explicitly requested after local smoke passes.
- Web-search actions are live stress-test actions and should not be mixed with reproducible BEIR benchmark claims.
- No private document/model flow in this phase; private/trusted-model gating remains separate work.
- No full local Qwen inference or runtime KV-cache pruning in this phase; only roadmap/scaffold and estimated KV reward fields.

## Definition Of Done

- Phase 1C.3 feedback schema is implemented or explicitly stubbed with tests.
- Context-level RLAIF labels are represented in schema/tests and have a first real MiMo-50 subset report.
- Phase 1D reward/preference builder is implemented and covered by tests.
- Both context-only and retrieval-context preference modes are represented.
- CLI can build rewards/preferences from existing BudgetRAG output folders.
- CLI can rebuild rewards with optional answer-label files without treating invalid labels as zero quality.
- CLI can rebuild rewards with optional context-label files without treating ambiguous/invalid context labels as zero quality.
- CLI exposes context-label blend and insufficient-penalty weights so aggressive candidates can be ablated before selector training.
- Summary markdown explains reward coverage, preference coverage, and tradeoffs.
- Answer-label and local Qwen KV-cache summary scripts are documented and covered by tests.
- `README.md` and `milestones.md` are updated when implementation lands.
- Existing test suite still passes.
