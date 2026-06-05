# Phase 1D RLAIF v2 Shrinkage-Smoothed Selector Eval

## Summary

This report evaluates `shrinkage_smoothed_best_average`, a row-wise empirical-Bayes alternative to
the hard-backoff `family_smoothed_best_average` selector.

The motivation was a weakness in hard backoff:

```text
if any candidate has an exact-signature train stat,
  family_smoothed_best_average only compares exact-signature candidates
```

That can ignore an exact-unseen candidate whose retrieval-context family has a much stronger train
mean. `shrinkage_smoothed_best_average` instead scores every candidate by the best available train
statistic:

```text
exact signature mean
  shrunk toward retrieval-context family mean
    shrunk toward context-policy mean
      shrunk toward global train mean
```

The selector is still offline-only and writes `runtime_default_replacement=false`.

## Inputs

```text
benchmark_results/rlaif/phase1d_selector_smoke_ai_judge_calibrated/rlaif_rewards.jsonl
benchmark_results/rlaif/phase1d_selector_smoke_ai_judge_calibrated/rlaif_preferences.jsonl
```

Sweep output:

```text
benchmark_results/rlaif/phase1d_selector_smoke_ai_judge_calibrated/split_sweep_shrinkage_seeds_1_2_3_4_5_42/
```

Seeds:

```text
1, 2, 3, 4, 5, 42
```

Split rule:

```text
held-out by benchmark + query_id
```

## Six-Seed Mean/Std

| Policy | Coverage | Reward | Quality | Token cost | Latency | KV cost | Oracle gap |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `cheapest` | 1.000 +/- 0.000 | 0.577 +/- 0.159 | 0.770 +/- 0.152 | 0.162 +/- 0.024 | 0.081 +/- 0.011 | 0.213 +/- 0.034 | 0.094 +/- 0.058 |
| `best_average` | 0.911 +/- 0.050 | 0.618 +/- 0.147 | 0.794 +/- 0.125 | 0.178 +/- 0.034 | 0.081 +/- 0.017 | 0.240 +/- 0.049 | 0.080 +/- 0.050 |
| `family_smoothed_best_average` | 1.000 +/- 0.000 | 0.598 +/- 0.138 | 0.767 +/- 0.122 | 0.176 +/- 0.033 | 0.093 +/- 0.022 | 0.238 +/- 0.046 | 0.074 +/- 0.047 |
| `shrinkage_smoothed_best_average` | 1.000 +/- 0.000 | 0.602 +/- 0.117 | 0.773 +/- 0.097 | 0.183 +/- 0.033 | 0.125 +/- 0.033 | 0.255 +/- 0.048 | 0.070 +/- 0.040 |
| `linear_reward_model` | 1.000 +/- 0.000 | 0.586 +/- 0.153 | 0.774 +/- 0.110 | 0.197 +/- 0.031 | 0.126 +/- 0.033 | 0.269 +/- 0.045 | 0.086 +/- 0.042 |
| `smoothed_linear_selector` | 1.000 +/- 0.000 | 0.595 +/- 0.137 | 0.767 +/- 0.122 | 0.188 +/- 0.041 | 0.126 +/- 0.033 | 0.250 +/- 0.046 | 0.076 +/- 0.046 |
| `oracle_logged` | 1.000 +/- 0.000 | 0.671 +/- 0.134 | 0.834 +/- 0.138 | 0.166 +/- 0.030 | 0.081 +/- 0.012 | 0.221 +/- 0.048 | 0.000 +/- 0.000 |

## Interpretation

`shrinkage_smoothed_best_average` is the strongest full-coverage non-oracle selector so far:

```text
family_smoothed_best_average:
  reward 0.598, quality 0.767, oracle gap 0.074

shrinkage_smoothed_best_average:
  reward 0.602, quality 0.773, oracle gap 0.070
```

It improves over `family_smoothed_best_average`, `linear_reward_model`, and
`smoothed_linear_selector` on mean reward and paired oracle gap, while keeping coverage at `1.000`.

It still does not beat `best_average` on mean reward/quality:

```text
best_average:
  reward 0.618, quality 0.794, coverage 0.911
```

This is a useful trade-off: shrinkage gives full coverage and the lowest non-oracle gap, but exact
signature averaging remains stronger when it can select.

## Pairwise Calibration Limitation

This selector still trains from scalar reward rows. `pairwise_tie_v1` currently affects preference
construction and diagnostics only. Direct pairwise RLAIF preferences will affect selector behavior
only after one of these is implemented:

```text
pairwise_ranker
calibrated_scalar_reward_v1
```

So this report should not be described as a pairwise-trained selector result.

## Cost Feature Limitation

The current selector metrics use logged/offline normalized cost fields. Before runtime deployment,
policy features should be split into:

```text
estimated_token_cost_norm
estimated_kv_cost_norm
predicted_latency_norm
observed_latency_norm  # eval/analysis only
```

Only estimated or predicted costs are available before generation.

## Next Step

Do not move to DPO/PPO/GRPO or runtime KV pruning yet. The next useful steps are:

```text
1. Run context-level RLAIF labels on a small subset.
2. Increase logged query groups and retriever diversity.
3. Add a preference-aware pairwise_ranker after richer labels exist.
```

Current diagnostics also show `retriever` diversity is still too small to support a strong claim
that the policy learned retrieval-strategy selection.
