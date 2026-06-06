# Plan Next Version Of 0.1.0

## Goal

Implement RLAIF for BudgetRAG on branch `internship` without breaking the Phase 1B/1C context-budgeting baseline.

## Current Framing And Guardrails

Current RLAIF/BudgetRAG infrastructure is strong enough to support end-to-end offline experiments:

```text
RLAIF labels -> reward/preference -> held-out eval -> selector baselines -> diagnostics
```

The current result bottlenecks are still data and coverage, not algorithmic sophistication:

- full MiMo context labels exist for the first 192-action Phase 1D dataset, but they are still small and AI-generated;
- logged query/action data is still small;
- retriever-diverse retrieval-only action coverage now exists for a 50-query SciFact sample, and a first 10-query retriever-diverse MiMo V2.5 generation/answer-label subset now exists, but it is still too small for selector generalization claims;
- exact action signatures remain sparse.

Future MiMo 2.5 runs should use standard `mimo-v2.5`, not `mimo-v2.5-pro`,
because the Pro endpoint did not show enough quality difference to justify the
extra cost. Existing reports may still name `mimo-v2.5-pro` when describing
historical runs; do not rewrite historical provenance.
Every new judge/generation summary must preserve `judge_provider`,
`judge_model`, and `generator_model` columns so historical Pro labels are not
silently aggregated with future standard-v2.5 labels. Merged result tables are
allowed when the drift is explicitly annotated.

Therefore the next bottleneck is richer supervision and broader logged action coverage, not a more complex RL algorithm. Do not add DPO, PPO, GRPO, runtime KV pruning, or a complex reward model in this phase. Current results should be framed as evidence that the infrastructure works, while learned selectors remain data-limited.

The immediate next experiment is to rerun or expand the small retriever-diverse
generation/judge subset with a larger MiMo completion cap, not to add a larger
algorithmic selector. The retrieval-only coverage run has already verified that
sampled queries can carry all three retrievers (`bm25`, `graph-bm25`,
`hybrid-rrf`) and 45 action rows. The first 10-query generation subset verified
the answer-label/reward/preference path, but `MAX_COMPLETION_TOKENS=256`
produced 77 empty answer strings, so full generation should not use that cap.

Do not overclaim:

- This is RLAIF-style AI feedback, not human preference learning.
- The selector is offline logged-candidate evaluation, not online RL.
- Pairwise RLAIF currently audits and calibrates scalar rewards; it does not train the selector unless a pairwise ranker is implemented later.
- Context reward remains a non-default candidate until full-label ablation and multi-seed evaluation support it.

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
| Phase 1C.3 | Implemented first-pass | Generation/judge feedback plus context-evidence labels so actions can be compared by quality, sufficiency, and efficiency. |
| Phase 1D | Implemented first-pass, still data-limited | RLAIF data builder and offline retrieval-context allocation policy. |

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
  "generator_model": "mimo-v2.5"
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
   - First retriever-diverse MiMo V2.5 generation/answer-label subset: 300 action rows over 10 SciFact queries, 300 answer labels, 299 valid JSON labels, 222 labels with numeric diagnostics, 186 clean AI-judge reward rows, and 1559 preference pairs. The run is documented in `docs/reports/phase1d_retriever_diversity_generation_mimo10.md`.
   - Added separate retriever-diversity diagnostics for the subset:
     `docs/reports/phase1d_retriever_diversity_generation_subset_validation.md`,
     `docs/reports/phase1d_retriever_diversity_answer_labels.md`,
     `docs/reports/phase1d_retriever_diversity_action_coverage.md`, and
     `docs/reports/phase1d_retriever_diversity_answer_quality.md`.
   - Completed full retriever-diverse context labeling for the same 300-row
     subset: 300 valid MiMo V2.5 context labels, 253 clean usable labels, 47
     ambiguous labels, 134 sufficient contexts, 158 insufficient contexts,
     mean context quality `0.505`, and mean evidence support `0.436`.
   - Added retriever-diversity context/evidence reports:
     `docs/reports/phase1d_retriever_diversity_context_label_validation.md`,
     `docs/reports/phase1d_retriever_diversity_context_labels.md`,
     `docs/reports/phase1d_retriever_diversity_evidence_quality.md`,
     `docs/reports/phase1d_retriever_diversity_reward_ablation.md`, and
     `docs/reports/phase1d_retriever_diversity_selector_eval.md`.
   - The non-default retriever-diverse context reward candidate changed
     156/300 reward rows, mostly downward, with mean changed-only delta
     `-0.301` and preference count rising from 1559 to 2412. Treat it as
     calibration supervision, not as a default reward or selector target.
   - Added `docs/reports/phase1d_retriever_diversity_next_run_decision.md`.
     The subset has enough retriever-level signal to keep branch A alive, but
     the low-cap empty-answer issue means the next run should be A1-medium:
     50 queries, three retrievers, five policy/profile variants, budgets
     `1000,4000`, and standard MiMo V2.5 with `MAX_COMPLETION_TOKENS=2048`.
     Do not run local Qwen profiling in this branch.
   - Selected 100 high-impact targeted DeepSeek audit rows from the
     retriever-diverse subset, sharded into two 50-row files, to check MiMo
     context-sufficiency harshness before trusting context reward as a selector
     target.
   - Completed the 100-row targeted DeepSeek v4 Flash context audit: 100 valid
     labels, 12 ambiguous rows, 0 invalid JSON/errors, MiMo-vs-DeepSeek
     sufficiency agreement `80/83 = 0.964`, 76 consensus-insufficient rows, 3
     high-disagreement rows, and 1 MiMo-harsh row. This supports the MiMo
     context-insufficiency signal on high-risk rows but does not change reward
     defaults.
   - Completed the A1-medium retriever-diverse generation gate:
     50 SciFact queries x 3 retrievers x 5 policy/profile variants x 2 budgets
     produced 1500/1500 non-empty standard MiMo V2.5 generations with
     `MAX_COMPLETION_TOKENS=2048`, 0 generation errors, and 1500 normalized
     RLAIF action rows. This closes the generation-coverage issue from the
     earlier 256-token-cap subset, but feedback provenance remains `missing`
     until answer/context labels are run. Report:
     `docs/reports/phase1d_retriever_diversity_a1_medium_generation_validation.md`.
   - Added A1 answer-label postprocess tooling while Kaggle shards run:
     `scripts/validate_rlaif_answer_labels.py` validates and merges sharded
     MiMo answer-label JSONL files, skips corrupted partial lines, reports
     missing/unknown/duplicate action ids, and preserves null/missing labels
     instead of turning them into zero scores.
   - Added `scripts/select_stratified_rlaif_actions.py` to prepare the next
     600-row context-label subset, balanced by
     `retrieval_strategy x context_policy/profile x budget`, with optional
     answer-label-priority sampling for ambiguous, low-support, unsupported,
     high-quality-low-support, and high-cost cases.
   - Added
     `docs/reports/phase1d_retriever_diversity_a1_mimo_v25_eval_template.md`
     and extended `rlaif-eval` with selected retriever/context-policy/
     adaptive-profile/budget distributions so A1 can answer whether offline
     selectors actually allocate across retrievers.
   - Consolidated the detailed Markdown experiment reports into the internship
     PDF: the main results now include a data-rich Key Findings table, a phase
     timeline, full context-label statistics, context reward ablations,
     pairwise and DeepSeek audits, answer-only selector sweeps, retriever-
     diversity status, and an appendix mapping Markdown reports to PDF
     sections. A1 answer/context labels remain explicitly marked as pending.
   - Do not use `MAX_COMPLETION_TOKENS=256` for future standard MiMo V2.5 generation runs; the low cap produced 77 empty generated answers with no request errors. Keep the larger cap and validate empty-answer coverage before spending judge budget.

```bash
uv run rag-bench rlaif-label-answers \
  --actions benchmark_results/rlaif/<run-name>/rlaif_actions.jsonl \
  --judge-provider mimo \
  --judge-model mimo-v2.5 \
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
   - Status: schema baseline and `rlaif-label-contexts` CLI implemented; full 192-row first-pass MiMo context labels have been merged, validated, ablated, and documented in `docs/reports/phase1d_rlaif_full_context_reward_ablation.md`.
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
   - Full MiMo context-label run completed for 192/192 action rows by merging existing labels 1-50 with Kaggle shards 51-121 and 122-192.
   - Full-label validation result: 192 labels, 177 clean usable labels, 15 ambiguous labels, 0 invalid JSON, 0 missing action ids, 0 unknown action ids, 0 duplicate action ids, and 0 dropped unknown chunk ids.
   - Full context summary: 110 sufficient contexts, 76 insufficient contexts, sufficiency rate 0.591, mean selected chunks 1.276, mean irrelevant chunks 3.708, mean context quality 0.602, and mean evidence support 0.556.
   - Full context reward ablation result: context candidates changed 140/192 reward rows. Penalty `0.25/0.50/1.00` has mean changed-only deltas `-0.212/-0.316/-0.525` and preference counts `952/946/944`. Penalty `0.25` is the conservative candidate; penalty `1.00` remains diagnostic.
   - Full context ablation report: `docs/reports/phase1d_rlaif_full_context_reward_ablation.md`.
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
   - Multi-judge audit should wait until full MiMo context reward ablation is available. Use DeepSeek/Groq only on targeted subsets first: MiMo context-insufficient rows, strong reward deltas, selector disagreement cases, and pairwise reward-vs-judge disagreement cases. This is an audit step, not the main blocking path.

```bash
uv run rag-bench rlaif-label-pairs \
  --actions benchmark_results/rlaif/<run-name>/rlaif_actions.jsonl \
  --rewards benchmark_results/rlaif/<run-name>/rlaif_rewards.jsonl \
  --preferences benchmark_results/rlaif/<run-name>/rlaif_preferences.jsonl \
  --output benchmark_results/rlaif/<run-name>/rlaif_pairwise_labels_mimo.jsonl \
  --judge-provider mimo \
  --judge-model mimo-v2.5 \
  --limit 50 \
  --resume \
  --sleep-seconds 0.5
```

```bash
uv run rag-bench rlaif-label-contexts \
  --actions benchmark_results/rlaif/<run-name>/rlaif_actions.jsonl \
  --judge-provider mimo \
  --judge-model mimo-v2.5 \
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
  "judge_model": "mimo-v2.5",
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
   - Current full-context result: all 192 MiMo context labels have been merged and ablated; penalty `0.25` is the conservative non-default context reward candidate, while `1.00` remains diagnostic.
   - Current audit result: the first targeted multi-judge audit is complete on 60 high-impact rows, with MiMo/DeepSeek/Groq labels, 51 consensus-insufficient rows, 6 MiMo-harsh/high-disagreement rows, and no invalid JSON or judge errors in the secondary judge outputs.
   - Current HotpotQA result: the Kaggle-first sampled evaluation path has been merged, including cached BM25 retrieval, MiMo/Groq generation support, policy sharding, retry tooling for failed Groq rows, and a curated report in `docs/reports/phase1c3_hotpotqa_kaggle_eval.md`. The report treats clean MiMo sampled shards as usable HotpotQA evidence and keeps quota-contaminated Groq rows separate.
   - Current model-bench result: the vLLM benchmark workflow has been merged as an operator/profiling workflow. It adds `rag-bench model-bench`, `MODEL_BENCH.md`, Vast AI RTX 5060 Ti setup/run scripts, vLLM serving metadata, speculative-decoding sweep scripts, and hardware aggregate summaries. These artifacts support local deployment planning; they are not part of the RLAIF selector benchmark claim.
   - Current reporting result: `feature/rlaif-retrieval-context-v0` has been merged into `internship`, and the internship report is now an English modular LaTeX report. `pdf/main.tex` is a short driver, `pdf/sections/en/` contains section files plus expanded appendices, and `pdf/main.pdf` covers Phase 1A through Phase 1D with added experimental setup, qualitative error analysis, threats to validity, planned HotpotQA/retriever-diversity evaluations, and an implementation map. The internal English/Vietnamese translation trace is no longer included in the submission PDF.
   - Next evaluator change: inspect the 6 MiMo-harsh rows and 51 consensus-insufficient rows as calibration examples, then expand logged query groups and retriever diversity before reassessing whether a pairwise ranker is justified.
   - Retrieval-strategy selection claims require broader logged actions with multiple retrievers such as `bm25`, `graph-bm25`, and `hybrid-rrf`. Current evidence mostly supports context-budget/action selection, not robust retrieval-strategy allocation. Web-search actions remain live stress tests only and must not be mixed with reproducible BEIR benchmark claims.

5. Outputs
   - Status: `rlaif-reward`, `rlaif-train`, `rlaif-eval`, and `rlaif-label-contexts` write the implemented files below; the full 192-action MiMo context-label run, context reward ablations, and six-seed selector sweeps are complete.
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
   - `configs/budgetrag_models.json`: generation model roles for fast Groq baseline, stronger Groq baseline, and MiMo long-context upper-bound rows.
   - `scripts/run_budgetrag_generation_matrix.py`: resumable generation matrix runner for Phase 1C.3 model/action coverage.
   - `scripts/run_hotpotqa_cached_budgetrag_eval.py`: cached HotpotQA evaluation entry point for Kaggle runs.
   - `scripts/run_hotpotqa_retry_failed_rows.py`: retry-only HotpotQA failed-row runner that reuses downloaded outputs and cached retrieval.
   - `docs/reports/phase1c3_multi_model_generation.md`: curated Phase 1C.3 multi-model generation report.
   - `docs/reports/phase1c3_mimo_long_context.md`: curated MiMo long-context report.
   - `docs/reports/phase1c3_hotpotqa_kaggle_eval.md`: curated sampled HotpotQA Kaggle report.
   - `MODEL_BENCH.md`: operator guide for vLLM/Vast AI model benchmark runs.
   - `rag-bench model-bench`: CLI path for benchmarking a local or existing OpenAI-compatible vLLM endpoint.
   - `scripts/setup_vllm_bench*.sh`, `scripts/bench_vast_5060ti*.sh`, and `scripts/vast_bench_lib.sh`: setup and run helpers for Vast AI RTX 5060 Ti CUDA 12.9/13.0 model benchmarking.
   - `runs/model_bench/`: ignored raw output root for model-bench artifacts.
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
   - `docs/reports/phase1d_rlaif_full_context_reward_ablation.md`: curated full 192-action context-label validation, reward ablation, and multi-seed selector sweep report.
   - `docs/reports/phase1d_rlaif_multijudge_audit_template.md`: targeted multi-judge audit template for MiMo/DeepSeek/Groq agreement, MiMo-harsh rows, and consensus-insufficient rows.
   - `docs/reports/phase1d_rlaif_multijudge_audit.md`: curated targeted multi-judge audit report over 60 high-impact rows using MiMo, DeepSeek v4 Flash, and Groq Qwen3 32B.
   - `docs/reports/phase1d_retriever_diversity_run_plan.md`: retriever-diversity run plan for `bm25`, `graph-bm25`, and `hybrid-rrf` logged action coverage.
   - `docs/reports/local_qwen_kv_estimates.md`: analytical Qwen2.5 KV-cache memory table for local deployment planning.
   - `pdf/main.tex`, `pdf/sections/en/*.tex`, and `pdf/main.pdf`: internship-wide English report for BudgetRAG / MemAlign-Qwen, including foundation, BudgetRAG, adaptive heuristic, experimental setup, RLAIF labels, reward/preference construction, selector baselines, qualitative error analysis, multi-judge audit, KV-cache motivation, threats to validity, planned evaluations, implementation map, limitations, future work, and expanded appendices.
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
  --judge-model mimo-v2.5 \
  --output benchmark_results/rlaif/<run-name>/rlaif_answer_labels.jsonl \
  --resume
```

Context labeling:

```bash
uv run rag-bench rlaif-label-contexts \
  --actions benchmark_results/rlaif/<run-name>/rlaif_actions.jsonl \
  --judge-provider mimo \
  --judge-model mimo-v2.5 \
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
  --judge-model mimo-v2.5 \
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
- HotpotQA remains sampled/Kaggle-first in this phase; clean MiMo sampled shards may be reported, but quota-contaminated Groq rows must stay separated from benchmark claims.
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
