# Phase 1D RLAIF AI-Judge Held-Out Evaluation

Date: 2026-06-04

Branch: `feature/rlaif-retrieval-context-v0`

## Scope

This report reruns the Phase 1D held-out selector evaluation after replacing RAGAS-only quality with full MiMo answer-level RLAIF labels where valid.

Pipeline:

```text
Phase 1C.3 BudgetRAG outputs
-> rlaif-build
-> rlaif-label-answers on Kaggle with MiMo
-> summarize_rlaif_labels.py
-> rlaif-reward --answer-labels
-> rlaif-split by benchmark + query_id
-> rlaif-train
-> rlaif-eval --split-manifest
```

This is still an offline held-out query evaluation. It does not replace runtime `adaptive-heuristic`, and it is not a human-label claim.

## Inputs

Base artifacts:

```text
benchmark_results/rlaif/phase1d_selector_smoke/rlaif_actions.jsonl
benchmark_results/rlaif/phase1d_selector_smoke/rlaif_feedback.jsonl
benchmark_results/rlaif/phase1d_selector_smoke/mimo_answer_labels/input/rlaif_answer_labels_mimo.jsonl
```

MiMo answer labels were produced by the private Kaggle job:

```text
codemaivanngu/rlaif-mimo-codemaivanngu-20260604-1819
status: COMPLETE
```

Raw outputs remain under ignored `benchmark_results/`; this file is the committed curated report.

## Commands

```bash
uv run --frozen python scripts/summarize_rlaif_labels.py \
  --labels benchmark_results/rlaif/phase1d_selector_smoke/mimo_answer_labels/input/rlaif_answer_labels_mimo.jsonl \
  --ragas-feedback benchmark_results/rlaif/phase1d_selector_smoke/rlaif_feedback.jsonl \
  --out-md benchmark_results/rlaif/phase1d_selector_smoke/mimo_answer_labels/answer_label_summary.md \
  --out-json benchmark_results/rlaif/phase1d_selector_smoke/mimo_answer_labels/answer_label_summary.json

uv run --frozen rag-bench rlaif-reward \
  --actions benchmark_results/rlaif/phase1d_selector_smoke/rlaif_actions.jsonl \
  --feedback benchmark_results/rlaif/phase1d_selector_smoke/rlaif_feedback.jsonl \
  --answer-labels benchmark_results/rlaif/phase1d_selector_smoke/mimo_answer_labels/input/rlaif_answer_labels_mimo.jsonl \
  --output-dir benchmark_results/rlaif/phase1d_selector_smoke_ai_judge

uv run --frozen rag-bench rlaif-split \
  --rewards benchmark_results/rlaif/phase1d_selector_smoke_ai_judge/rlaif_rewards.jsonl \
  --preferences benchmark_results/rlaif/phase1d_selector_smoke_ai_judge/rlaif_preferences.jsonl \
  --output-dir benchmark_results/rlaif/phase1d_selector_smoke_ai_judge/split_seed42 \
  --train-ratio 0.8 \
  --seed 42

uv run --frozen rag-bench rlaif-train \
  --rewards benchmark_results/rlaif/phase1d_selector_smoke_ai_judge/split_seed42/train_rewards.jsonl \
  --preferences benchmark_results/rlaif/phase1d_selector_smoke_ai_judge/split_seed42/train_preferences.jsonl \
  --output benchmark_results/rlaif/phase1d_selector_smoke_ai_judge/split_seed42/rlaif_policy.json

uv run --frozen rag-bench rlaif-eval \
  --rewards benchmark_results/rlaif/phase1d_selector_smoke_ai_judge/split_seed42/eval_rewards.jsonl \
  --policy benchmark_results/rlaif/phase1d_selector_smoke_ai_judge/split_seed42/rlaif_policy.json \
  --split-manifest benchmark_results/rlaif/phase1d_selector_smoke_ai_judge/split_seed42/split_manifest.json \
  --out-md benchmark_results/rlaif/phase1d_selector_smoke_ai_judge/split_seed42/rlaif_eval_summary.md
```

## Answer Label Coverage

| Metric | Value |
| --- | ---: |
| Label count | 192 |
| Valid JSON | 192 |
| Invalid JSON | 0 |
| Ambiguous labels | 25 |
| Judge errors | 1 |
| Scored labels | 191 |
| Judge provider/model | MiMo / `mimo-v2.5-pro` |

Score summary:

| Score | N | Mean | Std | Min | Max |
| --- | ---: | ---: | ---: | ---: | ---: |
| `overall_quality` | 191 | 0.823 | 0.307 | 0.000 | 1.000 |
| `answer_correctness` | 191 | 0.857 | 0.313 | 0.000 | 1.000 |
| `evidence_support` | 191 | 0.825 | 0.340 | 0.000 | 1.000 |
| `faithfulness` | 189 | 0.847 | 0.317 | 0.000 | 1.000 |
| `unsupported_claim_penalty` | 191 | 0.090 | 0.280 | 0.000 | 1.000 |

The Pearson correlation between MiMo `quality_score` and RAGAS answer relevancy is `0.277`. This is low enough that absolute reward values should not be compared as if RAGAS-only and AI-judge runs share the same label scale.

## Reward Rebuild

| Metric | RAGAS-only previous | MiMo AI-judge + fallback |
| --- | ---: | ---: |
| Actions | 192 | 192 |
| Rewards | 192 | 192 |
| Scored rewards | 192 | 192 |
| Preferences | 578 | 722 |
| Context-policy preferences | 289 | 361 |
| Retrieval-context preferences | 289 | 361 |
| Quality guardrail skips | N/A | 4 |
| Small reward delta skips | N/A | 548 |

Answer-label merge:

| Merge status | Count |
| --- | ---: |
| Used MiMo AI-judge label | 167 |
| Fallback to original feedback | 25 |
| Invalid/ambiguous answer label | 25 |

Interpretation:

- The labeler produced valid JSON for all 192 rows, so parser/repair did not fail on this full run.
- 25 labels were ambiguous or otherwise unusable for reward, so `rlaif-reward --answer-labels` correctly fell back to existing feedback instead of treating them as zero.
- Preference count increased from 578 to 722 because MiMo labels separate more action rewards beyond the `min_reward_delta` threshold.

## Held-Out Split

Split rule:

```text
benchmark + query_id
```

This is intentionally not a random action-row split. Every action for the same SciFact query stays in the same split.

| Metric | RAGAS-only previous | MiMo AI-judge + fallback |
| --- | ---: | ---: |
| Train ratio | 0.8 | 0.8 |
| Seed | 42 | 42 |
| Train query count | 27 | 27 |
| Eval query count | 7 | 7 |
| Train reward rows | 178 | 178 |
| Eval reward rows | 14 | 14 |
| Train preferences | 572 | 714 |
| Eval preferences | 6 | 8 |
| Dropped cross-split preferences | 0 | 0 |

The selector evaluator reports 9 eval query groups because policy evaluation groups include action-comparison dimensions such as model/top-k. The held-out unit remains the original `benchmark + query_id`.

## Held-Out Selector Metrics

The eval summary reports:

```text
held_out_query_eval = true
runtime_default_replacement = false
```

| Policy | Coverage | Mean reward | Mean quality | Token cost | Latency | KV cost | Paired oracle gap |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `fixed` | 0.000 | N/A | N/A | N/A | N/A | N/A | N/A |
| `cheapest` | 1.000 | 0.704 | 0.861 | 0.202 | 0.075 | 0.284 | 0.083 |
| `best_average` | 0.889 | 0.707 | 0.869 | 0.228 | 0.082 | 0.335 | 0.085 |
| `oracle_logged` | 1.000 | 0.788 | 0.961 | 0.221 | 0.074 | 0.325 | 0.000 |

Comparison with the earlier RAGAS-only held-out run:

| Policy | RAGAS-only reward | AI-judge reward | RAGAS-only quality | AI-judge quality |
| --- | ---: | ---: | ---: | ---: |
| `fixed` | N/A | N/A | N/A | N/A |
| `cheapest` | 0.377 | 0.704 | 0.476 | 0.861 |
| `best_average` | 0.425 | 0.707 | 0.538 | 0.869 |
| `oracle_logged` | 0.383 | 0.788 | 0.488 | 0.961 |

The AI-judge numbers are on a different quality scale than RAGAS answer relevancy, so the main comparison should be policy ordering and trade-offs, not the absolute reward increase.

## Interpretation

- `best_average` remains the best non-oracle policy by reward and quality on the covered eval groups.
- `cheapest` has full coverage and lower token/KV/latency cost, but slightly lower quality and reward than `best_average`.
- The gap between `best_average` and `cheapest` is much narrower under MiMo labels than under RAGAS-only labels: reward `0.707` vs `0.704`, quality `0.869` vs `0.861`.
- `oracle_logged` remains substantially higher at reward `0.788` and quality `0.961`, so there is still room for a learned selector or better ranking baseline.
- `fixed` still has zero held-out coverage, confirming that a single fixed action signature is not robust across this split.
- The low RAGAS/MiMo correlation suggests RAGAS answer relevancy was not a strong proxy for the richer answer correctness/support/unsupported-claim judgment.

## Selected Action Distribution

Top non-oracle selections from the eval summary:

| Policy | Selected action signatures |
| --- | --- |
| `cheapest` | `019f1e96dbcf`: 1; `087ead473649`: 1; `1ea6f674fa40`: 1; `4d49e6d9bd85`: 2; `74ecde043492`: 2; `934db6f2ca9e`: 1; `ea2a25656b73`: 1 |
| `best_average` | `1ea6f674fa40`: 1; `4d49e6d9bd85`: 2; `74ecde043492`: 2; `934db6f2ca9e`: 1; `ea2a25656b73`: 1; `f807f764c378`: 1 |
| `oracle_logged` | `019f1e96dbcf`: 1; `1ea6f674fa40`: 1; `2e6730856b00`: 1; `4d49e6d9bd85`: 2; `74ecde043492`: 2; `ea2a25656b73`: 1; `f807f764c378`: 1 |

## Limitations

- This is AI-judge feedback, not human annotation.
- Held-out eval is still small: 7 held-out queries, 14 reward rows, 9 selector query groups.
- MiMo labels and RAGAS labels use different scales; cross-source absolute reward values should not be overinterpreted.
- Context-level RLAIF labels are implemented but not yet populated in this run.
- Direct pairwise RLAIF labels are implemented but not yet populated in this run.
- Raw Kaggle/downloaded outputs remain ignored under `benchmark_results/`.

## Next Steps

1. Run `rlaif-label-pairs --limit 50` on this AI-judge reward/preference set to measure direct judge agreement with reward-derived preferences.
2. Run `rlaif-label-contexts` on a small subset to validate evidence sufficiency and selected/redundant chunk behavior.
3. Use pairwise agreement/disagreement to calibrate reward weights before adding any learned ranking selector.
4. Keep `runtime_default_replacement=false` until held-out quality guardrails pass on larger data.
