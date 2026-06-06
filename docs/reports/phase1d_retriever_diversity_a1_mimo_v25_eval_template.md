# Phase 1D Retriever-Diversity A1 MiMo V2.5 Evaluation Template

This template is for the 50-query A1-medium retriever-diverse run. Fill it only after the sharded Kaggle answer-label jobs have been downloaded, validated, merged, and rebuilt into reward artifacts.

## 1. Setup

- Benchmark: SciFact sampled 50 queries.
- Retrievers: BM25, graph-BM25, hybrid-RRF.
- Policy/profile variants: legacy, evidence-aware, score-density, adaptive-balanced, adaptive-aggressive.
- Budgets: 1000 and 4000 characters.
- Generator: standard MiMo V2.5.
- Max completion tokens: 2048.
- Logged rows: 1500 action rows.
- Claim boundary: offline logged-candidate evaluation only; no runtime default replacement.

## 2. Generation Validation

Use `docs/reports/phase1d_retriever_diversity_a1_medium_generation_validation.md` as the source of record.

Expected gate:

- 1500/1500 generated answers are non-empty.
- 0 generation errors.
- 30 query-result files.
- 1500 normalized RLAIF action rows.

## 3. Answer-Label Coverage

After Kaggle shards finish:

```bash
uv run python scripts/validate_rlaif_answer_labels.py \
  --actions benchmark_results/rlaif/retriever_diversity_a1_medium_mimo50_cap2048/rlaif_actions.jsonl \
  --labels \
    benchmark_results/rlaif/retriever_diversity_a1_medium_mimo50_cap2048/rlaif_answer_labels_mimo_v25_kaggle_part1_1_500.jsonl \
    benchmark_results/rlaif/retriever_diversity_a1_medium_mimo50_cap2048/rlaif_answer_labels_mimo_v25_kaggle_part2_501_1000.jsonl \
    benchmark_results/rlaif/retriever_diversity_a1_medium_mimo50_cap2048/rlaif_answer_labels_mimo_v25_kaggle_part3_1001_1500.jsonl \
  --merged-output benchmark_results/rlaif/retriever_diversity_a1_medium_mimo50_cap2048/rlaif_answer_labels_mimo_v25_merged.jsonl \
  --out-md benchmark_results/rlaif/retriever_diversity_a1_medium_mimo50_cap2048/answer_label_validation_summary.md
```

Report:

- label rows;
- invalid JSON line count;
- duplicate action ids;
- unknown action ids;
- missing action ids;
- ambiguous count;
- clean usable labels;
- merged scored labels.

## 4. Answer Quality By Retriever

Run after `rlaif-reward --answer-labels`:

```bash
uv run python scripts/analyze_retriever_diversity_answer_quality.py \
  --actions benchmark_results/rlaif/retriever_diversity_a1_medium_mimo50_cap2048/rlaif_actions.jsonl \
  --answer-labels benchmark_results/rlaif/retriever_diversity_a1_medium_mimo50_cap2048/rlaif_answer_labels_mimo_v25_merged.jsonl \
  --rewards benchmark_results/rlaif/retriever_diversity_a1_medium_mimo50_cap2048/answer_only/rlaif_rewards.jsonl \
  --out-csv benchmark_results/rlaif/retriever_diversity_a1_medium_mimo50_cap2048/answer_quality_by_retriever_policy.csv \
  --out-md benchmark_results/rlaif/retriever_diversity_a1_medium_mimo50_cap2048/answer_quality_by_retriever_policy.md
```

Answer questions:

- Does graph-BM25 still improve support/quality over BM25?
- Is hybrid-RRF still noisier after the 50-query run?
- Are unsupported-claim penalties concentrated in one retriever or context policy?
- Does budget 4000 improve quality enough to justify higher token/KV cost?

## 5. Context-Label Stratified Subset

Prepare a 600-row subset balanced by retriever, policy/profile, and budget:

```bash
uv run python scripts/select_stratified_rlaif_actions.py \
  --actions benchmark_results/rlaif/retriever_diversity_a1_medium_mimo50_cap2048/rlaif_actions.jsonl \
  --answer-labels benchmark_results/rlaif/retriever_diversity_a1_medium_mimo50_cap2048/rlaif_answer_labels_mimo_v25_merged.jsonl \
  --output benchmark_results/rlaif/retriever_diversity_a1_medium_mimo50_cap2048/rlaif_actions_context_stratified600.jsonl \
  --per-cell 20 \
  --seed 42
```

This gives about 20 rows per `retriever x context_policy/profile x budget` cell. It should be used before spending context-judge budget on all 1500 rows.

## 6. Evidence Quality By Retriever/Policy

After context labels exist, run:

```bash
uv run python scripts/analyze_context_policy_evidence_quality.py \
  --actions benchmark_results/rlaif/retriever_diversity_a1_medium_mimo50_cap2048/rlaif_actions_context_stratified600.jsonl \
  --context-labels benchmark_results/rlaif/retriever_diversity_a1_medium_mimo50_cap2048/rlaif_context_labels_stratified600.jsonl \
  --out-csv benchmark_results/rlaif/retriever_diversity_a1_medium_mimo50_cap2048/context_policy_evidence_quality.csv \
  --out-md benchmark_results/rlaif/retriever_diversity_a1_medium_mimo50_cap2048/context_policy_evidence_quality.md
```

Report:

- context sufficiency rate;
- selected/redundant/irrelevant chunk counts;
- context quality and evidence support by retriever/policy/budget;
- rows where answer quality is high but context support is low.

## 7. Reward Ablation

Keep context reward non-default. Compare:

- answer-only reward;
- context candidate with insufficient-context penalty 0.25;
- optional context candidates 0.50 and 1.00 if the 600-row subset is clean.

Report reward deltas, changed rows, clipped rewards, preference counts, and quality guardrail skips.

## 8. Selector Evaluation

Run query-level held-out sweeps after reward artifacts exist. Include selected retriever, selected context-policy, selected adaptive-profile, and selected budget distributions from `rlaif-eval`.

Policies to compare:

- cheapest;
- best_average;
- family_smoothed_best_average;
- shrinkage_smoothed_best_average;
- linear_reward_model;
- smoothed_linear_selector;
- oracle_logged.

## 9. Decision Gate

Proceed to A2 only if A1-medium is clean and shows stable signal:

- low invalid/ambiguous answer-label rate;
- enough clean reward rows per retriever/policy/budget cell;
- graph-BM25/hybrid-RRF differences are meaningful, not just noise;
- selector results improve over coverage baselines or reveal interpretable retriever choices.

If the signal is weak, prefer a smaller two-retriever matrix over more queries instead of running the full 2250-row generation matrix.

## 10. Limitations

- AI-judge labels are RLAIF-style supervision, not human labels.
- This is logged-candidate offline evaluation, not online RL.
- Context reward is a non-default candidate until full ablation supports it.
- Historical MiMo V2.5 Pro labels and future standard MiMo V2.5 labels must keep model provenance visible.
