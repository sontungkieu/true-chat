# Phase 1D RLAIF v2 Multi-Seed Selector Evaluation

## Summary

This report checks whether the seed-42 held-out result for `linear_reward_model` is stable
across several deterministic query-level splits.

Run:

```text
benchmark_results/rlaif/phase1d_selector_smoke_ai_judge_calibrated/split_sweep_seeds_1_2_3_4_5_42/
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

This is still a logged-candidate offline evaluation. It does not replace `adaptive-heuristic`
and does not claim online RL or online generalization.

## Mean/Std Across Seeds

| Policy | Coverage | Reward | Quality | Token cost | Latency | KV cost | Oracle gap |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `fixed` | 0.026 +/- 0.057 | 0.829 +/- 0.000 | 1.000 +/- 0.000 | 0.146 +/- 0.000 | 0.098 +/- 0.000 | 0.184 +/- 0.000 | 0.007 +/- 0.000 |
| `cheapest` | 1.000 +/- 0.000 | 0.577 +/- 0.159 | 0.770 +/- 0.152 | 0.162 +/- 0.024 | 0.081 +/- 0.011 | 0.213 +/- 0.034 | 0.094 +/- 0.058 |
| `best_average` | 0.911 +/- 0.050 | 0.618 +/- 0.147 | 0.794 +/- 0.125 | 0.178 +/- 0.034 | 0.081 +/- 0.017 | 0.240 +/- 0.049 | 0.080 +/- 0.050 |
| `linear_reward_model` | 1.000 +/- 0.000 | 0.586 +/- 0.153 | 0.774 +/- 0.110 | 0.197 +/- 0.031 | 0.126 +/- 0.033 | 0.269 +/- 0.045 | 0.086 +/- 0.042 |
| `oracle_logged` | 1.000 +/- 0.000 | 0.671 +/- 0.134 | 0.834 +/- 0.138 | 0.166 +/- 0.030 | 0.081 +/- 0.012 | 0.221 +/- 0.048 | 0.000 +/- 0.000 |

## Interpretation

The seed-42 result was positive but somewhat optimistic. Across six seeds:

- `linear_reward_model` keeps full coverage and improves over `cheapest` on mean reward
  (`0.586` vs `0.577`) and oracle gap (`0.086` vs `0.094`).
- `best_average` remains the strongest non-oracle selector by average reward and quality, but
  it has lower coverage (`0.911`) because some eval groups do not contain a train-ranked action
  signature.
- `linear_reward_model` does not yet consistently beat `best_average`; it mainly trades a small
  quality/reward drop for full coverage.
- `fixed` is not meaningful despite one high selected reward because its coverage is only
  `0.026`.

This is still useful: the learned selector has a real positive signal over the efficiency-only
baseline, but it needs more data, better context-level labels, and stronger feature diagnostics
before it can be treated as a stable selector.

## Per-Seed Notes

Seed 42 gave the strongest simple narrative:

```text
linear_reward_model reward 0.713, quality 0.872, coverage 1.000
cheapest reward 0.704, quality 0.861, coverage 1.000
best_average reward 0.707, quality 0.869, coverage 0.889
```

Other seeds were more mixed. For example, seed 2 strongly favored `cheapest`/`oracle_logged`,
while seeds 1, 4, 5 were more favorable to `linear_reward_model` than the cheapest baseline.

## Next Step

The next engineering step should be action coverage diagnostics:

```text
scripts/inspect_rlaif_action_coverage.py
```

This should identify sparse action signatures, train/eval signature mismatch, and whether
collapsing parts of the action signature would make `best_average` and learned policies more
robust. After that, a pairwise ranker can be considered as v2.1.
