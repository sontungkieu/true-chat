# Phase 1D RLAIF Full Context Reward Ablation

## Summary

This run completes the first full MiMo context-label pass for the Phase 1D
selector-smoke action set:

```text
existing MiMo context labels 1-50
+ Kaggle shard 51-121
+ Kaggle shard 122-192
-> validate / merge / dedupe
-> rebuild answer-only baseline
-> rebuild context reward candidates
-> compare reward deltas
-> six-seed held-out selector sweeps
```

This is still an offline reward-candidate audit. It does not replace the runtime
`adaptive-heuristic` policy and does not claim online RL.

The main result is clear: full context-level RLAIF labels materially change the
reward landscape. They expose many insufficient/noisy contexts, but the resulting
reward candidate is harsher and lowers absolute reward scale. The next step is
calibration and broader logged coverage, not a more complex RL algorithm.

## Inputs

| Input | Value |
| --- | --- |
| actions | `benchmark_results/rlaif/phase1d_selector_smoke/rlaif_actions.jsonl` |
| feedback | `benchmark_results/rlaif/phase1d_selector_smoke/rlaif_feedback.jsonl` |
| answer labels | `benchmark_results/rlaif/phase1d_selector_smoke/mimo_answer_labels/input/rlaif_answer_labels_mimo.jsonl` |
| context labels 1-50 | `benchmark_results/rlaif/phase1d_selector_smoke/rlaif_context_labels_mimo50.jsonl` |
| context labels 51-121 | `benchmark_results/rlaif/phase1d_selector_smoke/kaggle_context_envfix_20260605_135353/51_121/rlaif_context_labels_mimo_51_121.jsonl` |
| context labels 122-192 | `benchmark_results/rlaif/phase1d_selector_smoke/kaggle_context_envfix_20260605_135353/122_192/rlaif_context_labels_mimo_122_192.jsonl` |
| output root | `benchmark_results/rlaif/phase1d_full_context_ablation_mimo192` |

The Kaggle output initially included the cloned repo and `.venv` under
`true-chat/`; those downloaded folders were deleted locally. Only JSONL and
summary outputs were retained under ignored `benchmark_results/`.

## Context Label Validation

| Metric | Value |
| --- | ---: |
| action count | 192 |
| label rows | 192 |
| merged labels | 192 |
| missing actions | 0 |
| unknown action ids | 0 |
| duplicate action ids | 0 |
| duplicate conflicts | 0 |
| clean usable labels | 177 |
| ambiguous labels | 15 |
| invalid JSON labels | 0 |
| dropped unknown chunk ids | 0 |

Shard-level clean usable labels:

| Shard | Rows | Clean usable |
| --- | ---: | ---: |
| 1-50 | 50 | 46 |
| 51-121 | 71 | 66 |
| 122-192 | 71 | 65 |

There was no shard overlap, no missing action id, and no unknown action id. This
means the merge is usable for reward ablation without manual row editing.

## Context Label Summary

| Metric | Value |
| --- | ---: |
| labels | 192 |
| valid JSON | 192 |
| invalid JSON | 0 |
| ambiguous | 15 |
| scored context quality labels | 191 |
| sufficient contexts | 110 |
| insufficient contexts | 76 |
| sufficiency rate | 0.591 |
| missing evidence true | 2 |
| dropped unknown chunk ids | 0 |
| mean selected chunks | 1.276 |
| mean redundant chunks | 0.286 |
| mean irrelevant chunks | 3.708 |
| mean context quality | 0.602 |
| mean evidence support | 0.556 |
| mean minimality | 0.947 |

Interpretation:

- The retriever/context layer often returns many irrelevant chunks: mean
  irrelevant chunk count is 3.708.
- The judge usually selects only a small evidence subset: mean selected chunk
  count is 1.276.
- Context sufficiency is not universally poor, but 76/186 non-ambiguous
  sufficiency decisions are insufficient, so answer-only reward was hiding
  context-quality variation.

## Reward Candidate Ablation

Default answer-only reward remains the baseline. Context reward candidates are
non-default and use clean non-ambiguous context labels only; the 15 ambiguous or
errored context labels fall back to answer-level feedback.

| Setting | Rewards | Preferences |
| --- | ---: | ---: |
| answer-only | 192 | 722 |
| context penalty 0.25 | 192 | 952 |
| context penalty 0.50 | 192 | 946 |
| context penalty 1.00 | 192 | 944 |

Reward delta versus answer-only:

| Penalty | Changed rewards | Negative deltas | Positive deltas | Mean all delta | Mean changed delta | Candidate clipped at -1 |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0.25 | 140 | 108 | 32 | -0.155 | -0.212 | 2 |
| 0.50 | 140 | 108 | 32 | -0.231 | -0.316 | 2 |
| 1.00 | 140 | 108 | 32 | -0.383 | -0.525 | 3 |

Rows changed by context sufficiency:

| Penalty | Sufficient false | Sufficient true | Missing sufficiency |
| ---: | ---: | ---: | ---: |
| 0.25 | 64 | 72 | 4 |
| 0.50 | 64 | 72 | 4 |
| 1.00 | 64 | 72 | 4 |

The context labels sharpen preference construction substantially: preference
count rises from 722 to roughly 950. However, most reward changes are negative,
and penalty 1.00 has large negative deltas. Penalty 1.00 should remain
diagnostic unless a downstream selector result justifies it.

## Multi-Seed Held-Out Selector Sweep

All sweeps use deterministic query-level splits with seeds `1,2,3,4,5,42` and
keep `runtime_default_replacement=false`.

Mean reward / quality / oracle gap:

| Candidate | Policy | Coverage | Reward | Quality | Oracle gap |
| --- | --- | ---: | ---: | ---: | ---: |
| answer-only | cheapest | 1.000 | 0.577 | 0.770 | 0.094 |
| answer-only | best_average | 0.911 | 0.618 | 0.794 | 0.080 |
| answer-only | shrinkage_smoothed_best_average | 1.000 | 0.602 | 0.773 | 0.070 |
| answer-only | smoothed_linear_selector | 1.000 | 0.595 | 0.767 | 0.076 |
| answer-only | oracle_logged | 1.000 | 0.671 | 0.834 | 0.000 |
| penalty 0.25 | cheapest | 1.000 | 0.451 | 0.704 | 0.093 |
| penalty 0.25 | best_average | 0.911 | 0.478 | 0.701 | 0.085 |
| penalty 0.25 | shrinkage_smoothed_best_average | 1.000 | 0.449 | 0.683 | 0.095 |
| penalty 0.25 | smoothed_linear_selector | 1.000 | 0.472 | 0.689 | 0.072 |
| penalty 0.25 | oracle_logged | 1.000 | 0.544 | 0.755 | 0.000 |
| penalty 0.50 | cheapest | 1.000 | 0.386 | 0.704 | 0.104 |
| penalty 0.50 | best_average | 0.911 | 0.420 | 0.701 | 0.091 |
| penalty 0.50 | shrinkage_smoothed_best_average | 1.000 | 0.389 | 0.683 | 0.100 |
| penalty 0.50 | smoothed_linear_selector | 1.000 | 0.415 | 0.689 | 0.074 |
| penalty 0.50 | oracle_logged | 1.000 | 0.489 | 0.755 | 0.000 |
| penalty 1.00 | cheapest | 1.000 | 0.255 | 0.704 | 0.132 |
| penalty 1.00 | best_average | 0.911 | 0.303 | 0.701 | 0.109 |
| penalty 1.00 | shrinkage_smoothed_best_average | 1.000 | 0.273 | 0.689 | 0.114 |
| penalty 1.00 | smoothed_linear_selector | 1.000 | 0.329 | 0.703 | 0.058 |
| penalty 1.00 | oracle_logged | 1.000 | 0.387 | 0.748 | 0.000 |

The context reward candidates lower absolute reward and quality because the
reward now explicitly penalizes insufficient/noisy context. That is expected;
the result should be read as calibration evidence, not as a sign that the
pipeline is broken.

Important observations:

- Penalty 0.25 is the least aggressive candidate and keeps the highest absolute
  reward among context candidates.
- Penalty 1.00 is too harsh for a default candidate; it heavily compresses
  reward scale even though it gives `smoothed_linear_selector` the smallest
  oracle gap among context candidates.
- Selector performance is still unstable and data-limited. There is no strong
  evidence yet that a learned selector robustly beats simple baselines.

## Research Interpretation

This is a stronger result than the MiMo50 subset because full context labels
remove the main coverage caveat:

```text
context labels: 192/192 action rows
clean usable: 177/192
missing action ids: 0
unknown action ids: 0
duplicate action ids: 0
```

But the right conclusion remains conservative:

> Current results show that the infrastructure works, but the learned selector
> is data-limited. The next step is not a more complex RL algorithm, but richer
> supervision and broader logged action coverage: full context-level RLAIF
> labels, more query groups, and multiple retrieval strategies.

## Limitations

- Context labels are MiMo AI-judge labels, not human labels.
- The evaluation remains logged-candidate offline evaluation, not online RL.
- Context reward is still a non-default candidate.
- Current logged actions have low retriever diversity. The evidence mostly
  supports context-budget/action selection, not robust retrieval-strategy
  allocation.
- Web-search actions should remain live stress tests and must not be mixed with
  reproducible BEIR benchmark claims.

## Next

1. Treat penalty 0.25 as the conservative context reward candidate for further
   audit, but do not make it default.
2. Run a targeted multi-judge audit on MiMo-insufficient rows and strong reward
   deltas to check whether MiMo is too harsh.
3. Expand logged runs with `bm25`, `graph-bm25`, and `hybrid-rrf` before making
   retrieval-strategy allocation claims.
4. Add more query groups before considering pairwise ranker or more complex
   learned selectors.
