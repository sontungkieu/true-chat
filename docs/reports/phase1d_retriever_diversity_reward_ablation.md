# Phase 1D Retriever-Diversity Context Reward Ablation

## Summary

This report compares the answer-only MiMo V2.5 reward set against a non-default
context-label reward candidate for the 300-row retriever-diverse SciFact subset:

```text
retriever_diversity_generation_mimo10_20260605T194500Z
```

The candidate uses full MiMo V2.5 context labels over all 300 action rows with:

```text
context_quality_blend_weight = 0.50
context_support_blend_weight = 0.50
context_insufficient_penalty_weight = 0.25
```

This is a calibration experiment, not a default reward change. The runtime
retrieval policy is unchanged.

## Input Coverage

| Metric | Value |
| --- | ---: |
| action rows | 300 |
| answer labels | 300 |
| valid answer JSON labels | 299 |
| context labels | 300 |
| valid context JSON labels | 300 |
| clean usable context labels | 253 |
| ambiguous context labels | 47 |
| missing action ids after merge | 0 |
| unknown action ids after merge | 0 |
| duplicate context-label rows | 0 |

## Reward Build

| Metric | Answer-only | Context candidate |
| --- | ---: | ---: |
| reward rows | 300 | 300 |
| scored reward rows | 186 | 212 |
| AI-judge reward rows | 186 | 186 |
| ambiguous feedback rows | 77 | 77 |
| missing-quality rows | 37 | 37 |
| preference pairs | 1559 | 2412 |
| context-policy preferences | 370 | 571 |
| retrieval-context preferences | 1189 | 1841 |
| quality-guardrail skips | 8 | 123 |
| small-delta skips | 873 | 493 |

The candidate increases the number of scored rewards because context labels can
provide support/context evidence even when answer labels alone are not enough
for a clean answer-quality row. At the same time, the much larger
quality-guardrail skip count shows that the context penalty changes many pair
comparisons and should remain non-default until calibrated on more queries.

## Reward Delta

| Metric | Value |
| --- | ---: |
| shared action rows | 300 |
| changed rewards | 156 |
| negative deltas | 138 |
| positive deltas | 18 |
| zero deltas | 30 |
| mean delta over changed rows | -0.301 |
| median delta over all scored rows | -0.121 |
| minimum delta | -0.725 |
| maximum delta | 0.369 |

Changed rows by context sufficiency:

| Context sufficiency | Changed rows |
| --- | ---: |
| sufficient | 79 |
| insufficient | 74 |
| missing | 3 |

The negative mean delta is expected: the context candidate penalizes noisy or
insufficient evidence, while the answer-only reward can still score an answer
well if the final text looks acceptable. This confirms that context-level RLAIF
adds a different supervision signal from answer-level RLAIF.

## Selector Sweep On Context Candidate

The six-seed query-level held-out sweep uses 8 train queries and 2 eval queries
per seed. The eval set is therefore intentionally small and should be read as a
diagnostic, not as a learned-selector generalization claim.

| Policy | Coverage | Reward | Quality | Oracle gap |
| --- | ---: | ---: | ---: | ---: |
| `cheapest` | 0.833 +/- 0.236 | 0.325 +/- 0.307 | 0.683 +/- 0.180 | 0.335 +/- 0.260 |
| `best_average` | 1.000 +/- 0.000 | 0.004 +/- 0.324 | 0.483 +/- 0.219 | 0.655 +/- 0.266 |
| `family_smoothed_best_average` | 1.000 +/- 0.000 | 0.004 +/- 0.324 | 0.483 +/- 0.219 | 0.655 +/- 0.266 |
| `shrinkage_smoothed_best_average` | 1.000 +/- 0.000 | 0.004 +/- 0.324 | 0.483 +/- 0.219 | 0.655 +/- 0.266 |
| `linear_reward_model` | 0.667 +/- 0.373 | 0.147 +/- 0.286 | 0.570 +/- 0.213 | 0.444 +/- 0.309 |
| `smoothed_linear_selector` | 1.000 +/- 0.000 | -0.123 +/- 0.448 | 0.433 +/- 0.221 | 0.782 +/- 0.448 |
| `oracle_logged` | 1.000 +/- 0.000 | 0.659 +/- 0.136 | 0.925 +/- 0.075 | 0.000 +/- 0.000 |

Compared with the answer-only reward, the context candidate is much harsher. It
sharpens evidence penalties but makes simple smoothed selectors fragile on the
10-query split. The safest interpretation is that full context labels are now
available and useful for calibration, but this candidate reward is not ready to
replace answer-only reward or runtime selection.

## Interpretation

The ablation supports three conclusions:

1. Context-level labels materially change supervision: 156/300 reward rows
   changed after adding context labels.
2. The context candidate exposes evidence-quality risk that answer-only reward
   hides, especially when an answer is acceptable despite noisy retrieved
   context.
3. The current 10-query subset is too small for selector conclusions under the
   stricter reward. More query groups and a larger-generation-cap rerun are
   needed before training a learned retrieval-context selector from this signal.

## Next Step

Do not scale from this exact run to a full 2250-row generation matrix, because
the original generation cap produced 77 empty answers. The next useful run is a
larger-cap retriever-diverse generation subset, followed by the same
answer/context label and reward-ablation pipeline.
