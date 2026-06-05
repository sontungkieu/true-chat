# Phase 1D RLAIF v2 Smoothed Linear Selector Eval

## Summary

This report evaluates `smoothed_linear_selector`, a learned offline selector that extends
`linear_reward_model` with train-only aggregate reward features.

The goal is to combine three signals:

- normal retrieval-context action/cost features;
- exact-signature, retrieval-context-family, context-policy, and retriever train means/counts;
- full coverage when an eval exact signature was unseen during train.

The aggregate reward features are computed only inside `rlaif-train` from the train split. During
`rlaif-eval`, missing exact-signature aggregates fall back through family/context/retriever/global
features already stored in the policy artifact. Eval reward/quality labels are never used to fill
these features.

The policy remains an offline logged-candidate selector and has
`runtime_default_replacement=false`.

## Inputs

```text
benchmark_results/rlaif/phase1d_selector_smoke_ai_judge_calibrated/rlaif_rewards.jsonl
benchmark_results/rlaif/phase1d_selector_smoke_ai_judge_calibrated/rlaif_preferences.jsonl
```

Sweep output:

```text
benchmark_results/rlaif/phase1d_selector_smoke_ai_judge_calibrated/split_sweep_smoothed_linear_seeds_1_2_3_4_5_42/
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

`smoothed_linear_selector` keeps full coverage and improves over the plain learned selector:

```text
linear_reward_model:
  reward 0.586, oracle gap 0.086

smoothed_linear_selector:
  reward 0.595, oracle gap 0.076
```

This confirms that train-only aggregate action-family statistics are useful features.

However, the gain is modest. `smoothed_linear_selector` still trails:

```text
family_smoothed_best_average:
  reward 0.598, oracle gap 0.074

best_average:
  reward 0.618, quality 0.794
```

So the result is not a robust learned-selector win. The current data remains small, and
`shrinkage_smoothed_best_average` is currently the stronger full-coverage non-oracle selector.

The practical conclusion is:

> Aggregate train means help repair the learned selector, but in this small logged-candidate regime
> they do not yet beat simple action-family smoothing or exact-signature averaging.

## Per-Seed Notes

`smoothed_linear_selector` is closest to the family-smoothed baseline on seeds where exact signature
coverage is the main bottleneck:

```text
seed 3:
  family_smoothed reward 0.478
  smoothed_linear reward 0.475
  linear_reward_model reward 0.336

seed 42:
  family_smoothed reward 0.713
  smoothed_linear reward 0.711
  linear_reward_model reward 0.713
```

It is weaker on seeds where the learned model overweights aggregate/cost features relative to exact
signature quality:

```text
seed 1:
  best_average reward 0.664
  linear_reward_model reward 0.637
  smoothed_linear reward 0.610

seed 4:
  linear_reward_model reward 0.416
  smoothed_linear reward 0.355
```

## Next Step

Do not move to DPO/PPO/GRPO or runtime KV pruning from this result.

The next useful steps are:

```text
1. Run context-level RLAIF labels on a small subset.
2. Add more logged query groups so selector train/eval splits are less tiny.
3. Re-evaluate whether pairwise ranker features help after richer context labels exist.
```

If more data is not available, `family_smoothed_best_average` remains the cleaner v2 baseline than
`smoothed_linear_selector`.

## Limitations

- The evaluation is held out by query id, but each eval split is still small.
- All policies choose among logged candidates only.
- The reward set depends on MiMo answer labels plus RAGAS fallback.
- Aggregate reward features are train-only and eval-safe, but they are still target-derived summary
  features; this is an offline selector baseline, not an online RL result.
