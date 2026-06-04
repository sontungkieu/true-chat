# Phase 1D Pairwise-Calibrated Reward Diagnostics

Date: 2026-06-04

Branch: `feature/rlaif-retrieval-context-v0`

## Scope

This report adds diagnostics over the MiMo-50 direct pairwise audit to identify when scalar reward may overweight small quality/support differences relative to pairwise judge preference.

This is analysis-only. It does not change reward defaults, does not retrain selector policies, and does not replace runtime `adaptive-heuristic`.

## Command

```bash
uv run --frozen python scripts/diagnose_rlaif_pairwise_calibration.py \
  --labels benchmark_results/rlaif/phase1d_selector_smoke_ai_judge/rlaif_pairwise_labels_mimo_50.jsonl \
  --rewards benchmark_results/rlaif/phase1d_selector_smoke_ai_judge/rlaif_rewards.jsonl \
  --actions benchmark_results/rlaif/phase1d_selector_smoke/rlaif_actions.jsonl \
  --quality-tie-threshold 0.10 \
  --support-tie-threshold 0.20 \
  --out-md benchmark_results/rlaif/phase1d_selector_smoke_ai_judge/pairwise_calibration_diagnostics_50.md \
  --out-json benchmark_results/rlaif/phase1d_selector_smoke_ai_judge/pairwise_calibration_diagnostics_50.json
```

Raw diagnostics remain ignored under `benchmark_results/`; this file is the committed curated report.

## Diagnostics

| Metric | Value |
| --- | ---: |
| Pair labels | 50 |
| Valid non-ambiguous decisions | 48 |
| Small quality/support delta pairs | 38 |
| Cheaper wins when quality/support tied | 35 |
| Scalar-over-quality disagreements | 5 |
| Scalar-over-quality disagreement rate | 0.104 |
| Cheaper-win rate when quality/support tied | 0.921 |
| Mean abs quality gap in scalar-over-quality disagreements | 0.100 |
| Mean abs support gap in scalar-over-quality disagreements | 0.200 |

Configured thresholds:

| Threshold | Value |
| --- | ---: |
| Quality tie threshold | 0.10 |
| Support tie threshold | 0.20 |

Suggested candidate threshold source:

```text
max_abs_gap_among_scalar_over_quality_disagreements
quality ~= 0.10
support ~= 0.20
```

## Pattern

All 5 matched scalar-over-quality disagreements are in `query_id=128`.

The pattern is consistent:

```text
reward formula:
  chooses Action A because MiMo scalar quality/support are slightly higher

direct pairwise judge:
  treats both answers as acceptable or both as correct abstentions
  then chooses Action B because it is cheaper
```

Representative matched cases:

| Preference | Reward-chosen A | Judge-chosen B | Quality gap | Support gap | Rationale |
| --- | --- | --- | ---: | ---: | --- |
| `d2daba5b7c5c6a3d` | `evidence-aware`, budget 8000, reward 0.815 | `adaptive-heuristic`, budget 16000 balanced, reward 0.735 | 0.100 | 0.200 | Both correctly indicate insufficient evidence; B has lower token/KV cost. |
| `af59ade654e591cd` | `adaptive-heuristic`, budget 4000 aggressive, reward 0.829 | `adaptive-heuristic`, budget 16000 balanced, reward 0.735 | 0.100 | 0.200 | Both are correct and supported; B is more efficient. |
| `5a218c7263336226` | `legacy`, budget 32000, reward 0.784 | `adaptive-heuristic`, budget 16000 balanced, reward 0.735 | 0.100 | 0.200 | Both correctly note lack of evidence; B uses fewer resources. |

## Interpretation

- Pairwise agreement is already high, so the current scalar reward is not arbitrary.
- The mismatch is narrow and interpretable: correct abstention or equally acceptable answers should not overpay for small scalar quality/support gains.
- A candidate reward v2 should test a tie band around small quality/support gaps before applying efficiency trade-offs.
- This should remain a candidate mode, not the default, until rerun through held-out evaluation.

## Candidate Calibration Rule

Do not enable this as default yet:

```text
if direct-pairwise-calibrated mode is enabled
and abs(quality_gap) <= quality_tie_threshold
and abs(support_gap) <= support_tie_threshold
and no unsupported-claim risk difference:
    downweight the quality/support gap
    let token + latency + KV efficiency decide more strongly
```

Initial candidate thresholds from this audit:

```text
quality_tie_threshold = 0.10
support_tie_threshold = 0.20
```

These thresholds are audit-derived from a small 50-pair subset. They should be re-estimated after context-level labels and a more diverse pairwise sample.

## Next Steps

1. Add an optional `reward_calibration_v1_candidate` mode behind an explicit flag.
2. Rebuild rewards in a separate output directory and rerun `rlaif-split/train/eval`.
3. Compare original AI-judge reward vs calibrated candidate on held-out query eval.
4. Run context-label subset before making any runtime or default-policy claim.
