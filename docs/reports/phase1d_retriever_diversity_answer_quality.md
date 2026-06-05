# Retriever-Diversity Answer Quality

- Actions: `benchmark_results/rlaif/retriever_diversity_generation_mimo10_20260605T194500Z/rlaif_actions.jsonl`
- Answer labels: `benchmark_results/rlaif/retriever_diversity_generation_mimo10_20260605T194500Z/rlaif_answer_labels_mimo_v25.jsonl`
- Rewards: `benchmark_results/rlaif/retriever_diversity_generation_mimo10_20260605T194500Z/reward_mimo_answer/rlaif_rewards.jsonl`
- Action rows: 300
- Label rows: 300
- Reward rows: 300

This report groups answer-level AI-judge labels and optional rewards by retriever, context policy, retriever-policy pair, adaptive profile, and budget. Ambiguous unscored labels are excluded by default; they are counted in the source label summary, not forced to zero.

## By `budget_chars`

| Group | Rows | Scored labels | Missing answers | Quality | Correctness | Support | Unsupported | Reward | Token cost | Latency | KV cost |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `1000` | 125 | 125 | 0 | 0.793 | 0.794 | 0.780 | 0.181 | 0.520 | 0.401 | 0.432 | 0.308 |
| `4000` | 97 | 97 | 0 | 0.861 | 0.860 | 0.867 | 0.072 | 0.652 | 0.772 | 0.421 | 0.894 |

## By `context_policy`

| Group | Rows | Scored labels | Missing answers | Quality | Correctness | Support | Unsupported | Reward | Token cost | Latency | KV cost |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `adaptive-heuristic` | 88 | 88 | 0 | 0.860 | 0.853 | 0.833 | 0.114 | 0.596 | 0.575 | 0.430 | 0.561 |
| `evidence-aware` | 47 | 47 | 0 | 0.853 | 0.866 | 0.870 | 0.091 | 0.603 | 0.596 | 0.405 | 0.597 |
| `legacy` | 44 | 44 | 0 | 0.802 | 0.805 | 0.805 | 0.151 | 0.625 | 0.528 | 0.426 | 0.538 |
| `score-density` | 43 | 43 | 0 | 0.733 | 0.733 | 0.735 | 0.200 | 0.464 | 0.540 | 0.446 | 0.560 |

## By `retrieval_strategy`

| Group | Rows | Scored labels | Missing answers | Quality | Correctness | Support | Unsupported | Reward | Token cost | Latency | KV cost |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `bm25` | 72 | 72 | 0 | 0.847 | 0.856 | 0.824 | 0.117 | 0.613 | 0.568 | 0.410 | 0.580 |
| `graph-bm25` | 71 | 71 | 0 | 0.855 | 0.859 | 0.872 | 0.148 | 0.613 | 0.510 | 0.436 | 0.479 |
| `hybrid-rrf` | 79 | 79 | 0 | 0.771 | 0.761 | 0.761 | 0.135 | 0.514 | 0.607 | 0.435 | 0.626 |

## By `retrieval_strategy,context_policy`

| Group | Rows | Scored labels | Missing answers | Quality | Correctness | Support | Unsupported | Reward | Token cost | Latency | KV cost |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `bm25 / adaptive-heuristic` | 29 | 29 | 0 | 0.941 | 0.962 | 0.886 | 0.034 | 0.679 | 0.564 | 0.431 | 0.556 |
| `bm25 / evidence-aware` | 14 | 14 | 0 | 0.871 | 0.871 | 0.900 | 0.077 | 0.621 | 0.606 | 0.385 | 0.621 |
| `bm25 / legacy` | 15 | 15 | 0 | 0.753 | 0.753 | 0.740 | 0.233 | 0.611 | 0.538 | 0.400 | 0.548 |
| `bm25 / score-density` | 14 | 14 | 0 | 0.729 | 0.729 | 0.714 | 0.200 | 0.465 | 0.570 | 0.403 | 0.621 |
| `graph-bm25 / adaptive-heuristic` | 29 | 29 | 0 | 0.834 | 0.834 | 0.828 | 0.128 | 0.624 | 0.475 | 0.421 | 0.402 |
| `graph-bm25 / evidence-aware` | 15 | 15 | 0 | 0.893 | 0.927 | 0.933 | 0.133 | 0.536 | 0.596 | 0.428 | 0.596 |
| `graph-bm25 / legacy` | 14 | 14 | 0 | 0.964 | 0.971 | 0.986 | 0.071 | 0.760 | 0.515 | 0.448 | 0.516 |
| `graph-bm25 / score-density` | 13 | 13 | 0 | 0.738 | 0.715 | 0.764 | 0.292 | 0.510 | 0.486 | 0.464 | 0.477 |
| `hybrid-rrf / adaptive-heuristic` | 30 | 30 | 0 | 0.807 | 0.767 | 0.790 | 0.177 | 0.488 | 0.682 | 0.439 | 0.720 |
| `hybrid-rrf / evidence-aware` | 18 | 18 | 0 | 0.806 | 0.811 | 0.788 | 0.065 | 0.646 | 0.588 | 0.400 | 0.579 |
| `hybrid-rrf / legacy` | 15 | 15 | 0 | 0.700 | 0.700 | 0.700 | 0.143 | 0.501 | 0.530 | 0.431 | 0.549 |
| `hybrid-rrf / score-density` | 16 | 16 | 0 | 0.731 | 0.750 | 0.733 | 0.125 | 0.427 | 0.558 | 0.469 | 0.574 |

## By `retrieval_strategy,context_policy,adaptive_profile`

| Group | Rows | Scored labels | Missing answers | Quality | Correctness | Support | Unsupported | Reward | Token cost | Latency | KV cost |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `bm25 / adaptive-heuristic / aggressive` | 17 | 17 | 0 | 0.900 | 0.935 | 0.812 | 0.059 | 0.633 | 0.452 | 0.436 | 0.380 |
| `bm25 / adaptive-heuristic / balanced` | 12 | 12 | 0 | 1.000 | 1.000 | 1.000 | 0.000 | 0.752 | 0.721 | 0.424 | 0.807 |
| `bm25 / evidence-aware / conservative` | 14 | 14 | 0 | 0.871 | 0.871 | 0.900 | 0.077 | 0.621 | 0.606 | 0.385 | 0.621 |
| `bm25 / legacy / conservative` | 15 | 15 | 0 | 0.753 | 0.753 | 0.740 | 0.233 | 0.611 | 0.538 | 0.400 | 0.548 |
| `bm25 / score-density / conservative` | 14 | 14 | 0 | 0.729 | 0.729 | 0.714 | 0.200 | 0.465 | 0.570 | 0.403 | 0.621 |
| `graph-bm25 / adaptive-heuristic / aggressive` | 15 | 15 | 0 | 0.847 | 0.860 | 0.860 | 0.113 | 0.632 | 0.373 | 0.394 | 0.248 |
| `graph-bm25 / adaptive-heuristic / balanced` | 14 | 14 | 0 | 0.821 | 0.807 | 0.793 | 0.143 | 0.616 | 0.584 | 0.451 | 0.567 |
| `graph-bm25 / evidence-aware / conservative` | 15 | 15 | 0 | 0.893 | 0.927 | 0.933 | 0.133 | 0.536 | 0.596 | 0.428 | 0.596 |
| `graph-bm25 / legacy / conservative` | 14 | 14 | 0 | 0.964 | 0.971 | 0.986 | 0.071 | 0.760 | 0.515 | 0.448 | 0.516 |
| `graph-bm25 / score-density / conservative` | 13 | 13 | 0 | 0.738 | 0.715 | 0.764 | 0.292 | 0.510 | 0.486 | 0.464 | 0.477 |
| `hybrid-rrf / adaptive-heuristic / aggressive` | 15 | 15 | 0 | 0.700 | 0.613 | 0.667 | 0.200 | 0.478 | 0.527 | 0.422 | 0.497 |
| `hybrid-rrf / adaptive-heuristic / balanced` | 15 | 15 | 0 | 0.913 | 0.920 | 0.913 | 0.153 | 0.496 | 0.838 | 0.456 | 0.943 |
| `hybrid-rrf / evidence-aware / conservative` | 18 | 18 | 0 | 0.806 | 0.811 | 0.788 | 0.065 | 0.646 | 0.588 | 0.400 | 0.579 |
| `hybrid-rrf / legacy / conservative` | 15 | 15 | 0 | 0.700 | 0.700 | 0.700 | 0.143 | 0.501 | 0.530 | 0.431 | 0.549 |
| `hybrid-rrf / score-density / conservative` | 16 | 16 | 0 | 0.731 | 0.750 | 0.733 | 0.125 | 0.427 | 0.558 | 0.469 | 0.574 |

## Interpretation Notes

- Use `scored_labels` and `missing_answer` together; high quality over a small clean subset can hide generation failures.
- `unsupported_claim_penalty` is a risk score where higher is worse.
- Reward and cost columns are present only when `--rewards` is provided.
