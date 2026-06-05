# Phase 1D RLAIF v2 Family-Smoothed Selector Evaluation

## Summary

This report evaluates `family_smoothed_best_average`, a selector designed after the action
coverage diagnostics showed exact-signature sparsity.

Backoff order:

```text
1. exact action signature mean reward
2. retrieval-context family mean reward
3. context-policy mean reward
4. cost tie-break
```

It is an offline logged-candidate selector. It does not replace the runtime
`adaptive-heuristic` policy.

Run:

```text
benchmark_results/rlaif/phase1d_selector_smoke_ai_judge_calibrated/split_sweep_family_smoothed_seeds_1_2_3_4_5_42/
```

Setup:

```text
label source: full MiMo answer labels + fallback feedback
reward calibration: pairwise_tie_v1 candidate preferences
split rule: benchmark + query_id
train ratio: 0.8
seeds: 1, 2, 3, 4, 5, 42
runtime_default_replacement: false
```

## Mean/Std Across Seeds

| Policy | Coverage | Reward | Quality | Token cost | Latency | KV cost | Oracle gap |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `cheapest` | 1.000 +/- 0.000 | 0.577 +/- 0.159 | 0.770 +/- 0.152 | 0.162 +/- 0.024 | 0.081 +/- 0.011 | 0.213 +/- 0.034 | 0.094 +/- 0.058 |
| `best_average` | 0.911 +/- 0.050 | 0.618 +/- 0.147 | 0.794 +/- 0.125 | 0.178 +/- 0.034 | 0.081 +/- 0.017 | 0.240 +/- 0.049 | 0.080 +/- 0.050 |
| `family_smoothed_best_average` | 1.000 +/- 0.000 | 0.598 +/- 0.138 | 0.767 +/- 0.122 | 0.176 +/- 0.033 | 0.093 +/- 0.022 | 0.238 +/- 0.046 | 0.074 +/- 0.047 |
| `linear_reward_model` | 1.000 +/- 0.000 | 0.586 +/- 0.153 | 0.774 +/- 0.110 | 0.197 +/- 0.031 | 0.126 +/- 0.033 | 0.269 +/- 0.045 | 0.086 +/- 0.042 |
| `oracle_logged` | 1.000 +/- 0.000 | 0.671 +/- 0.134 | 0.834 +/- 0.138 | 0.166 +/- 0.030 | 0.081 +/- 0.012 | 0.221 +/- 0.048 | 0.000 +/- 0.000 |

## Interpretation

`family_smoothed_best_average` does what it was designed to do: it fixes the exact-signature
coverage loss. Coverage rises from `0.911` for `best_average` to `1.000`.

The quality trade-off is mixed:

- It improves average reward over `cheapest` (`0.598` vs `0.577`).
- It improves average reward and oracle gap over `linear_reward_model` (`0.598` vs `0.586`,
  oracle gap `0.074` vs `0.086`).
- It does not beat `best_average` on mean reward or quality (`0.598/0.767` vs `0.618/0.794`),
  but it has full coverage and a lower paired oracle gap (`0.074` vs `0.080`).

The result suggests that action-family smoothing is useful, but family means are still a coarse
signal. In the current small-data regime, exact signatures remain high-value when available, and
family backoff is best understood as a coverage repair rather than a complete replacement for
better ranking.

## Per-Seed Notes

`family_smoothed_best_average` is strongest when `best_average` loses coverage:

```text
seed 3:
  best_average coverage 0.909, reward 0.441
  family_smoothed coverage 1.000, reward 0.478

seed 42:
  best_average coverage 0.889, reward 0.707
  family_smoothed coverage 1.000, reward 0.713
```

It is weaker when exact-signature averages already cover the relevant eval groups:

```text
seed 1:
  best_average reward 0.664
  family_smoothed reward 0.613
```

## Follow-Up Result

The follow-up selector combined exact-signature ranking, family smoothing, and light learned
scoring:

```text
smoothed_linear_selector
```

Implemented behavior:

```text
features:
  exact_signature_train_mean_reward_if_seen
  retrieval_context_family_mean_reward
  context_policy_mean_reward
  retriever_mean_reward
  token_cost_norm / latency_norm / kv_cost_norm

guardrail:
  no direct reward/quality label from held-out rows
  no runtime default replacement
```

Six-seed result:

```text
smoothed_linear_selector:
  coverage 1.000
  reward 0.595
  quality 0.767
  oracle gap 0.076
```

This improves over `linear_reward_model` (`0.586` reward, `0.086` oracle gap), but it still trails
`family_smoothed_best_average` and `best_average`. Do not move to DPO/PPO/GRPO or runtime KV
pruning yet. The next bottleneck is still small logged data and missing context-level RLAIF labels.
