# Phase 1D Pairwise-Tie Reward Calibration Candidate

Date: 2026-06-04

Branch: `feature/rlaif-retrieval-context-v0`

## Scope

This report evaluates the opt-in `pairwise_tie_v1` reward calibration candidate.

The candidate does **not** change scalar reward rows. It changes only pairwise preference construction:

```text
if quality/support gaps are inside the configured tie band
and --tie-break-by-efficiency is enabled:
    choose the lower token + latency + estimated-KV cost action
else:
    use the normal scalar reward preference path
```

Runtime behavior is unchanged. `adaptive-heuristic` remains the default runtime policy.

## Command

```bash
uv run --frozen rag-bench rlaif-reward \
  --actions benchmark_results/rlaif/phase1d_selector_smoke/rlaif_actions.jsonl \
  --feedback benchmark_results/rlaif/phase1d_selector_smoke/rlaif_feedback.jsonl \
  --answer-labels benchmark_results/rlaif/phase1d_selector_smoke/mimo_answer_labels/input/rlaif_answer_labels_mimo.jsonl \
  --output-dir benchmark_results/rlaif/phase1d_selector_smoke_ai_judge_calibrated \
  --reward-calibration pairwise_tie_v1 \
  --quality-tie-threshold 0.10 \
  --support-tie-threshold 0.20 \
  --tie-break-by-efficiency
```

Raw outputs remain ignored under `benchmark_results/`.

## Preference Construction Impact

| Metric | Default AI-judge reward | `pairwise_tie_v1` candidate |
| --- | ---: | ---: |
| Actions | 192 | 192 |
| Rewards | 192 | 192 |
| Scored rewards | 192 | 192 |
| Preferences | 722 | 1270 |
| Context-policy preferences | 361 | 635 |
| Retrieval-context preferences | 361 | 635 |
| Quality guardrail skips | 4 | 4 |
| Small reward delta skips | 548 | 0 |
| Higher-reward preference reason | 722 | 370 |
| Pairwise-tie efficiency reason | 0 | 900 |

Calibration settings:

| Setting | Value |
| --- | ---: |
| `reward_calibration` | `pairwise_tie_v1` |
| `quality_tie_threshold` | 0.10 |
| `support_tie_threshold` | 0.20 |
| `tie_break_by_efficiency` | `true` |

## Held-Out Selector Smoke

Because scalar reward rows are intentionally unchanged, the current fixed/cheapest/best-average/oracle selector metrics remain the same as the default AI-judge held-out run.

| Policy | Coverage | Mean reward | Mean quality | Token cost | Latency | KV cost | Oracle gap |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `fixed` | 0.000 | N/A | N/A | N/A | N/A | N/A | N/A |
| `cheapest` | 1.000 | 0.704 | 0.861 | 0.202 | 0.075 | 0.284 | 0.083 |
| `best_average` | 0.889 | 0.707 | 0.869 | 0.228 | 0.082 | 0.335 | 0.085 |
| `oracle_logged` | 1.000 | 0.788 | 0.961 | 0.221 | 0.074 | 0.325 | 0.000 |

Split summary:

| Metric | Default AI-judge reward | `pairwise_tie_v1` candidate |
| --- | ---: | ---: |
| Train query count | 27 | 27 |
| Eval query count | 7 | 7 |
| Train reward rows | 178 | 178 |
| Eval reward rows | 14 | 14 |
| Train preferences | 714 | 1254 |
| Eval preferences | 8 | 16 |
| Dropped cross-split preferences | 0 | 0 |

## Interpretation

- The candidate does not move scalar reward metrics by design.
- It converts many small-gap comparisons that would have been skipped or decided only by scalar reward into explicit efficiency tie-break preferences.
- This matches the MiMo-50 pairwise audit insight: when quality/support are close enough, the direct judge often prefers the cheaper action.
- The next useful effect will appear in learned or preference-aware selectors, not in `cheapest` or `best_average` scalar baselines.

## Guardrails

- Default remains `--reward-calibration none`.
- `--tie-break-by-efficiency` is rejected unless `--reward-calibration pairwise_tie_v1` is set.
- Historical reports are not overwritten.
- Runtime default replacement remains false.
- This is not DPO/PPO/GRPO and not online RL.

## Next Steps

1. Add `linear_reward_model` offline selector baseline that can use the richer preference/reward training set without leaking labels into features.
2. Compare default AI-judge reward vs calibrated candidate on the same held-out query split.
3. Add context-level labels before increasing claims about evidence-mask or KV-retention behavior.
