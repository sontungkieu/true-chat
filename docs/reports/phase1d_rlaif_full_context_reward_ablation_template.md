# Phase 1D RLAIF Full Context Reward Ablation

Status: template. Fill this after the remaining context-label shards finish and
`scripts/run_context_reward_ablation_pipeline.py` has produced a manifest.

## Scope

This report covers the full context-label postprocess path:

```text
download context-label shards
-> validate / merge / dedupe
-> summarize context labels
-> rebuild answer-only reward baseline
-> rebuild context reward candidates
-> compare reward deltas
-> run multi-seed held-out selector sweeps
```

This is an offline reward-candidate audit. It does not replace the runtime
`adaptive-heuristic` policy and does not claim online RL.

Current framing:

```text
Current results show that the infrastructure works, but the learned selector is
data-limited. The next step is not a more complex RL algorithm, but richer
supervision and broader logged action coverage: full context-level RLAIF labels,
more query groups, and multiple retrieval strategies.
```

Do not use this report to justify DPO, PPO, GRPO, runtime KV pruning, or a
complex reward model. The immediate bottleneck is supervision/action coverage.

## Inputs

| Input | Value |
| --- | --- |
| actions | `TODO` |
| feedback | `TODO` |
| answer labels | `TODO` |
| context label shards | `TODO` |
| merged context labels | `TODO` |
| output root | `TODO` |

## Context Label Validation

| Metric | Value |
| --- | ---: |
| action count | TODO |
| label rows | TODO |
| merged labels | TODO |
| missing actions | TODO |
| unknown action ids | TODO |
| duplicate action ids | TODO |
| duplicate conflicts | TODO |
| clean usable labels | TODO |

Notes:

- Duplicate labels should be resolved by the validator merge rule, not by manual editing.
- Unknown action ids should be investigated before using the merged file.
- Ambiguous/invalid labels should remain explicit and must not become score zero.

## Context Label Summary

| Metric | Value |
| --- | ---: |
| valid JSON labels | TODO |
| invalid JSON labels | TODO |
| ambiguous labels | TODO |
| sufficient contexts | TODO |
| insufficient contexts | TODO |
| sufficiency rate | TODO |
| dropped unknown chunk ids | TODO |
| mean selected chunks | TODO |
| mean irrelevant chunks | TODO |
| mean context quality | TODO |
| mean evidence support | TODO |
| mean minimality | TODO |

## Reward Candidate Ablation

Default answer-only reward remains the baseline. Context reward candidates are
non-default and should be compared against that baseline.

| Penalty | Reward rows | Preferences | Changed rewards | Mean changed delta | Negative deltas | Positive deltas |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0.25 | TODO | TODO | TODO | TODO | TODO | TODO |
| 0.50 | TODO | TODO | TODO | TODO | TODO | TODO |
| 1.00 | TODO | TODO | TODO | TODO | TODO | TODO |

Interpretation:

- If penalty `1.00` creates many clipped rewards or very large negative deltas,
  it should remain diagnostic only.
- If penalty `0.25` changes too few preferences, it may be too weak for selector
  training.
- Candidate selection should use held-out selector sweeps, not only reward delta
  counts.

## Multi-Seed Held-Out Selector Sweep

| Candidate | Policy | Coverage | Reward | Quality | Oracle gap |
| --- | --- | ---: | ---: | ---: | ---: |
| answer-only | cheapest | TODO | TODO | TODO | TODO |
| answer-only | best_average | TODO | TODO | TODO | TODO |
| answer-only | shrinkage_smoothed_best_average | TODO | TODO | TODO | TODO |
| answer-only | smoothed_linear_selector | TODO | TODO | TODO | TODO |
| answer-only | oracle_logged | TODO | TODO | TODO | TODO |
| penalty 0.25 | cheapest | TODO | TODO | TODO | TODO |
| penalty 0.25 | best_average | TODO | TODO | TODO | TODO |
| penalty 0.25 | shrinkage_smoothed_best_average | TODO | TODO | TODO | TODO |
| penalty 0.25 | smoothed_linear_selector | TODO | TODO | TODO | TODO |
| penalty 0.25 | oracle_logged | TODO | TODO | TODO | TODO |
| penalty 0.50 | cheapest | TODO | TODO | TODO | TODO |
| penalty 0.50 | best_average | TODO | TODO | TODO | TODO |
| penalty 0.50 | shrinkage_smoothed_best_average | TODO | TODO | TODO | TODO |
| penalty 0.50 | smoothed_linear_selector | TODO | TODO | TODO | TODO |
| penalty 0.50 | oracle_logged | TODO | TODO | TODO | TODO |
| penalty 1.00 | cheapest | TODO | TODO | TODO | TODO |
| penalty 1.00 | best_average | TODO | TODO | TODO | TODO |
| penalty 1.00 | shrinkage_smoothed_best_average | TODO | TODO | TODO | TODO |
| penalty 1.00 | smoothed_linear_selector | TODO | TODO | TODO | TODO |
| penalty 1.00 | oracle_logged | TODO | TODO | TODO | TODO |

Interpretation:

- If results are unstable across seeds, report instability directly.
- `runtime_default_replacement` must remain `false`.
- Do not claim online generalization from logged-candidate offline evaluation.

## Multi-Judge Audit Plan

Run this only after full MiMo context reward ablation is complete. Multi-judge
labeling is an audit/validation step, not the main blocking path.

Target a small DeepSeek/Groq subset first:

- MiMo context-insufficient rows.
- Rows with strong context reward deltas.
- Selector disagreement cases.
- Pairwise reward-vs-judge disagreement cases.

Goal: check whether MiMo is too harsh or whether other judges agree on the same
failure modes.

## Qualitative Examples

Add a few examples after running an example sampler or manual audit:

1. High answer score but context insufficient: TODO.
2. Many irrelevant chunks but one useful evidence chunk: TODO.
3. Acceptable abstention where cheaper context should win: TODO.

## Limitations

- Context labels are AI-judge labels, not human labels.
- The evaluation remains logged-candidate offline evaluation.
- Selector metrics depend on available action rows; they do not imply online
  deployment quality.
- Context reward candidates are opt-in and must not replace runtime defaults.
- Current logged actions have low retriever diversity. The evidence mainly
  supports context-budget/action selection, not robust retrieval-strategy
  allocation.
- Retrieval-strategy claims require broader logged runs with `bm25`,
  `graph-bm25`, and `hybrid-rrf`. Web-search actions are live stress tests only
  and should not be mixed with reproducible BEIR benchmark claims.

## Next

1. Choose a conservative context penalty candidate if multi-seed metrics support it.
2. Run a targeted multi-judge audit if MiMo-only full-label results reveal strong
   or surprising deltas.
3. Expand query groups and retriever diversity before claiming retrieval-strategy
   selection.
4. Use context labels to design evidence-mask/KV pruning supervision only after
   the reward-candidate behavior is stable.
