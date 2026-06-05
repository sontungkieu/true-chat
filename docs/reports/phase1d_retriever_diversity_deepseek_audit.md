# Phase 1D RLAIF Multi-Judge Targeted Audit

This report aggregates secondary judge labels for a targeted RLAIF audit subset. It is an audit/confidence layer, not a reward-default replacement.

## Inputs

- Actions: `benchmark_results/rlaif/retriever_diversity_generation_mimo10_20260605T194500Z/multijudge_deepseek_audit/targeted_cases_100.jsonl`
- MiMo labels: `benchmark_results/rlaif/retriever_diversity_generation_mimo10_20260605T194500Z/rlaif_context_labels_mimo_v25.jsonl`
- DeepSeek labels: `benchmark_results/rlaif/retriever_diversity_generation_mimo10_20260605T194500Z/multijudge_deepseek_audit/deepseek_context_part1_1_50.jsonl, benchmark_results/rlaif/retriever_diversity_generation_mimo10_20260605T194500Z/multijudge_deepseek_audit/deepseek_context_part2_51_100.jsonl`
- Groq labels: `N/A`

## Judge Counts

| Judge | Labels | Valid | Ambiguous | Invalid JSON | Errors | Clean sufficiency |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| mimo | 100 | 100 | 3 | 0 | 0 | 95 |
| deepseek | 100 | 100 | 12 | 0 | 0 | 88 |

## Sufficiency Agreement

| Pair | Compared | Agree | Disagree | Agreement rate |
| --- | ---: | ---: | ---: | ---: |
| mimo_vs_deepseek | 83 | 80 | 3 | 0.964 |

## Numeric Score Correlation

| Pair / Score | N | Pearson |
| --- | ---: | ---: |
| mimo_vs_deepseek_context_quality_score | 85 | 0.476 |
| mimo_vs_deepseek_evidence_support_score | 85 | 0.905 |

## Audit Signals

| Signal | Count |
| --- | ---: |
| high disagreement cases | 3 |
| MiMo harsh cases | 1 |
| consensus insufficient cases | 76 |
| majority vote `insufficient` | 93 |
| majority vote `sufficient` | 4 |
| majority vote `tie` | 3 |

## Interpretation

The targeted audit supports the MiMo context-label signal on the high-risk rows:

```text
MiMo vs DeepSeek sufficiency agreement: 80/83 = 0.964
Consensus-insufficient rows: 76/100
High-disagreement rows: 3/100
MiMo-harsh rows: 1/100
```

This does not make AI feedback equivalent to human labels, but it reduces the
concern that MiMo is uniquely harsh on this subset. The strongest agreement is
on evidence support (`Pearson = 0.905`), while context quality is only
moderately correlated (`Pearson = 0.476`). For reward construction, this means
context sufficiency/support are safer signals than raw context-quality score.

The audit remains targeted and non-default. It should inform context-reward
calibration and example selection, not replace reward defaults by itself.

## High-Disagreement Examples

| Action | Query | Vote | Judge sufficiency | Selection reason |
| --- | --- | --- | --- | --- |
| `rlaif-action-v1-e2dabbd6c6eae54a` | `51` | tie | mimo=False, deepseek=True | mimo_context_insufficient |
| `rlaif-action-v1-6ce609964f65e8a9` | `51` | tie | mimo=True, deepseek=False | large_negative_context_reward_delta |
| `rlaif-action-v1-8a9499f9af8131e2` | `51` | tie | mimo=True, deepseek=False | large_negative_context_reward_delta |

## MiMo-Harsh Examples

| Action | Query | Vote | Judge sufficiency | Selection reason |
| --- | --- | --- | --- | --- |
| `rlaif-action-v1-e2dabbd6c6eae54a` | `51` | tie | mimo=False, deepseek=True | mimo_context_insufficient |

Rows with strong judge disagreement should be manually inspected before being
used as clean context supervision.
