# Phase 1D RLAIF v2 Linear Selector Held-Out Report

## Summary

This report records the first learned offline selector baseline for Phase 1D. The new
`linear_reward_model` policy is trained only on train-split logged reward rows and evaluated
on held-out query groups. It is an offline policy artifact and does not replace the runtime
`adaptive-heuristic` policy.

The run uses the existing AI-judge reward artifacts:

```text
benchmark_results/rlaif/phase1d_selector_smoke_ai_judge_calibrated/split_seed42/
```

The split is by `benchmark + query_id`, not by random action rows.

## Training Artifact

```text
train reward rows: 178
train preferences: 1254
train query groups: 46
action signatures: 75
policies: fixed, cheapest, best_average, linear_reward_model, oracle_logged
runtime_default_replacement: false
```

`linear_reward_model` uses a ridge-regression style linear reward predictor over action and
cost features. The feature table excludes reward, quality, evidence-support labels, and
preference outcomes, so the selector cannot directly read the held-out target label.

## Held-Out Eval

Eval set:

```text
query groups: 9
held_out_query_eval: true
```

| Policy | Coverage | Reward | Quality | Token cost | Latency | KV cost | Oracle gap |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `fixed` | 0.000 | N/A | N/A | N/A | N/A | N/A | N/A |
| `cheapest` | 1.000 | 0.704 | 0.861 | 0.202 | 0.075 | 0.284 | 0.083 |
| `best_average` | 0.889 | 0.707 | 0.869 | 0.228 | 0.082 | 0.335 | 0.085 |
| `linear_reward_model` | 1.000 | 0.713 | 0.872 | 0.217 | 0.074 | 0.319 | 0.075 |
| `oracle_logged` | 1.000 | 0.788 | 0.961 | 0.221 | 0.074 | 0.325 | 0.000 |

## Interpretation

`linear_reward_model` is the strongest non-oracle selector in this small held-out run. It keeps
full coverage like `cheapest`, improves reward and quality over both `cheapest` and
`best_average`, and reduces paired oracle gap from about `0.083-0.085` to `0.075`.

The result is useful as an offline sanity check for learned selection, not as a generalization
claim. The eval set has only 9 query groups and 14 reward rows. The next validation step should
be a multi-seed split sweep and, after context labels are populated, a larger held-out run with
context-sufficiency signals.

## Guardrails

- No runtime selector default changed.
- No online RL is claimed.
- No DPO/PPO/GRPO or runtime KV pruning is implemented in this phase.
- `adaptive-heuristic` remains the runtime default until an offline selector passes larger
  held-out evaluations and quality guardrails.
