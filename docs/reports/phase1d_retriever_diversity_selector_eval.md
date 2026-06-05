# Phase 1D Retriever-Diversity Selector Diagnostics

## Summary

This note keeps the selector results from the retriever-diverse 10-query subset
separate from the label and reward-ablation reports. The split has only 10 query
groups, so every six-seed held-out evaluation uses 8 train queries and 2 eval
queries per seed. These results verify the pipeline, not model generalization.

Two reward settings were evaluated:

1. **Answer-only MiMo V2.5 reward:** answer labels are merged into the scalar
   reward without context-label penalties.
2. **Context-candidate reward:** answer labels plus non-default context labels
   with insufficient-context penalty 0.25 and quality/support blend 0.50/0.50.

Both selector artifacts keep `runtime_default_replacement=false`.

## Answer-Only Reward Sweep

| Policy | Coverage | Reward | Quality | Oracle gap |
| --- | ---: | ---: | ---: | ---: |
| `cheapest` | 0.833 +/- 0.236 | 0.658 +/- 0.331 | 0.925 +/- 0.168 | 0.148 +/- 0.331 |
| `best_average` | 0.500 +/- 0.289 | 0.625 +/- 0.351 | 0.910 +/- 0.180 | 0.179 +/- 0.351 |
| `family_smoothed_best_average` | 0.500 +/- 0.289 | 0.625 +/- 0.351 | 0.910 +/- 0.180 | 0.179 +/- 0.351 |
| `shrinkage_smoothed_best_average` | 0.667 +/- 0.373 | 0.263 +/- 0.716 | 0.710 +/- 0.395 | 0.544 +/- 0.717 |
| `linear_reward_model` | 0.667 +/- 0.373 | 0.555 +/- 0.321 | 0.890 +/- 0.174 | 0.251 +/- 0.321 |
| `smoothed_linear_selector` | 0.583 +/- 0.186 | 0.355 +/- 0.645 | 0.792 +/- 0.311 | 0.451 +/- 0.643 |
| `oracle_logged` | 1.000 +/- 0.000 | 0.804 +/- 0.003 | 1.000 +/- 0.000 | 0.000 +/- 0.000 |

The answer-only sweep is the cleaner selector smoke. `cheapest` performs well
because the sample is small and many high-quality rows are also low-cost. This
does not mean efficiency-only selection is generally sufficient; it means the
current eval split is not broad enough to separate selectors robustly.

## Context-Candidate Reward Sweep

| Policy | Coverage | Reward | Quality | Oracle gap |
| --- | ---: | ---: | ---: | ---: |
| `cheapest` | 0.833 +/- 0.236 | 0.325 +/- 0.307 | 0.683 +/- 0.180 | 0.335 +/- 0.260 |
| `best_average` | 1.000 +/- 0.000 | 0.004 +/- 0.324 | 0.483 +/- 0.219 | 0.655 +/- 0.266 |
| `family_smoothed_best_average` | 1.000 +/- 0.000 | 0.004 +/- 0.324 | 0.483 +/- 0.219 | 0.655 +/- 0.266 |
| `shrinkage_smoothed_best_average` | 1.000 +/- 0.000 | 0.004 +/- 0.324 | 0.483 +/- 0.219 | 0.655 +/- 0.266 |
| `linear_reward_model` | 0.667 +/- 0.373 | 0.147 +/- 0.286 | 0.570 +/- 0.213 | 0.444 +/- 0.309 |
| `smoothed_linear_selector` | 1.000 +/- 0.000 | -0.123 +/- 0.448 | 0.433 +/- 0.221 | 0.782 +/- 0.448 |
| `oracle_logged` | 1.000 +/- 0.000 | 0.659 +/- 0.136 | 0.925 +/- 0.075 | 0.000 +/- 0.000 |

The context-candidate sweep shows that the stricter reward changes the ranking
problem substantially. It is useful as a calibration diagnostic, but the current
candidate is too harsh to use as the default selector target.

## What This Means

The retriever-diverse run closes a logged-action coverage gap:

```text
same query
-> BM25 rows
-> graph-BM25 rows
-> hybrid-RRF rows
-> answer labels
-> context labels
-> reward/preference rows
-> held-out selector diagnostics
```

It does not yet close the quality-supervision or data-volume gap. The next
bottleneck is a larger-cap generation subset with more query groups, not a more
complex selector.
