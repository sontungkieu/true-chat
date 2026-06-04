# Phase 1D Direct Pairwise RLAIF MiMo-50 Audit

Date: 2026-06-04

Branch: `feature/rlaif-retrieval-context-v0`

## Scope

This report audits whether scalar reward-derived preferences align with direct pairwise MiMo judge preferences.

Pipeline:

```text
Phase 1C.3 BudgetRAG outputs
-> full MiMo answer labels
-> rlaif-reward --answer-labels
-> rlaif_preferences.jsonl
-> rlaif-label-pairs --limit 50
-> summarize_rlaif_pairwise_labels.py
```

This is an offline calibration audit. It does not train DPO, does not train a reward model, and does not replace runtime `adaptive-heuristic`.

## Commands

The AI-judge reward run writes rewards/preferences under `phase1d_selector_smoke_ai_judge`, while normalized action rows remain in the original `phase1d_selector_smoke` artifact:

```bash
uv run --frozen rag-bench rlaif-label-pairs \
  --actions benchmark_results/rlaif/phase1d_selector_smoke/rlaif_actions.jsonl \
  --rewards benchmark_results/rlaif/phase1d_selector_smoke_ai_judge/rlaif_rewards.jsonl \
  --preferences benchmark_results/rlaif/phase1d_selector_smoke_ai_judge/rlaif_preferences.jsonl \
  --output benchmark_results/rlaif/phase1d_selector_smoke_ai_judge/rlaif_pairwise_labels_mimo_50.jsonl \
  --judge-provider mimo \
  --judge-model mimo-v2.5-pro \
  --limit 50 \
  --resume \
  --sleep-seconds 0.5

uv run --frozen python scripts/summarize_rlaif_pairwise_labels.py \
  --input benchmark_results/rlaif/phase1d_selector_smoke_ai_judge/rlaif_pairwise_labels_mimo_50.jsonl \
  --out-md benchmark_results/rlaif/phase1d_selector_smoke_ai_judge/pairwise_label_summary_50.md \
  --out-json benchmark_results/rlaif/phase1d_selector_smoke_ai_judge/pairwise_label_summary_50.json
```

The local run was sharded into two 25-pair output files and merged afterward to keep at most two concurrent MiMo calls:

```text
pairwise_shards/preferences_001_025.jsonl
pairwise_shards/preferences_026_050.jsonl
pairwise_shards/labels_001_025.jsonl
pairwise_shards/labels_026_050.jsonl
```

Raw outputs remain ignored under `benchmark_results/`; this file is the committed curated report.

## Label Quality

| Metric | Value |
| --- | ---: |
| Pair labels | 50 |
| Valid JSON | 50 |
| Invalid JSON | 0 |
| Ambiguous labels | 2 |
| Judge/API errors | 2 |
| Missing input rows | 0 |
| Ties | 0 |
| Judge provider/model | MiMo / `mimo-v2.5-pro` |

The two ambiguous labels were read timeouts. They were kept as ambiguous labels with null confidence, not converted into losses or score zero.

## Agreement With Scalar Reward Preferences

In this setup Action A is the scalar reward-derived chosen action and Action B is the rejected action. The direct judge is still asked to decide independently.

| Metric | Value |
| --- | ---: |
| A wins | 41 |
| B wins | 7 |
| Tie | 0 |
| Ambiguous/error | 2 |
| Agreement with reward preference | 41 |
| Disagreement with reward preference | 7 |
| Agreement rate over non-ambiguous decisions | 0.854 |

Confidence over the 48 non-ambiguous decisions:

| N | Mean | Std | Min | Max |
| ---: | ---: | ---: | ---: | ---: |
| 48 | 0.914 | 0.038 | 0.800 | 1.000 |

## Winner Breakdown

| Judge dimension | Winner counts |
| --- | --- |
| Overall chosen | A: 41; B: 7; missing/ambiguous: 2 |
| Answer quality | A: 24; tie: 24; missing: 2 |
| Evidence support | A: 13; tie: 35; missing: 2 |
| Efficiency | A: 38; B: 8; tie: 2; missing: 2 |
| Unsupported-claim risk | B-risk: 5; neither: 43; missing: 2 |

## Disagreement Pattern

All 7 direct-judge disagreements come from the same query group (`query_id=128`). The scalar reward formula preferred Action A because its MiMo-derived quality/support scores were higher. The direct pairwise judge instead chose Action B because it considered both answers substantively acceptable or equally abstaining, then favored the lower resource cost.

Representative cases:

| Case | Reward-chosen A | Judge-chosen B | Judge rationale |
| --- | --- | --- | --- |
| `d2daba5b7c5c6a3d` | `evidence-aware`, budget 8000, reward 0.815, quality 1.000 | `adaptive-heuristic`, budget 16000 balanced, reward 0.735, quality 0.900 | Both correctly indicate insufficient evidence; B has lower token/KV cost. |
| `af59ade654e591cd` | `adaptive-heuristic`, budget 4000 aggressive, reward 0.829, quality 1.000 | `adaptive-heuristic`, budget 16000 balanced, reward 0.735, quality 0.900 | Both are correct and supported; B is more efficient. |
| `5a218c7263336226` | `legacy`, budget 32000, reward 0.784, quality 1.000 | `adaptive-heuristic`, budget 16000 balanced, reward 0.735, quality 0.900 | Both correctly note lack of evidence; B uses fewer resources. |

This is useful rather than bad: it shows scalar reward calibration may overvalue small absolute quality/support differences when both answers are effectively acceptable or both are correct abstentions.

## Interpretation

- The scalar reward-derived preference agrees with direct MiMo pairwise preference for most valid comparisons: 41/48 non-ambiguous decisions.
- The 7 disagreements are not random parser failures; they cluster around a cost-vs-quality calibration issue in one query group.
- The direct judge often treats answer quality/evidence support as tied, then picks the lower-cost action. The scalar reward formula can still prefer the higher-scored answer because quality weight dominates.
- Unsupported-claim risk appears in 5 pairs, always associated with Action B in this subset. This supports keeping unsupported-claim penalties explicit instead of relying on answer relevancy alone.
- The run has zero invalid JSON, so the pairwise prompt/parser path is stable on this subset.

## Limitations

- This is a 50-pair subset, not the full 722-preference set.
- Pairs are sampled from reward-derived preferences, so the audit measures alignment with the current reward construction rather than discovering every possible pairwise ordering.
- MiMo is an AI judge, not a human labeler.
- The two timeout rows are counted explicitly as ambiguous and excluded from agreement rate.
- The disagreement cluster is small and query-local; it should guide reward calibration, not a broad policy claim.

## Next Steps

1. Add a calibration experiment that lowers the effect of small quality/support deltas when the direct pairwise judge says answer quality is tied. Initial diagnostics are now documented in `docs/reports/phase1d_rlaif_pairwise_calibration_diagnostics.md`.
2. Run `rlaif-label-contexts` on a small subset to check context sufficiency and selected/redundant chunks.
3. Repeat pairwise labeling on a more diverse preference sample after context labels are populated.
4. Keep `runtime_default_replacement=false` until larger held-out and pairwise audits pass quality guardrails.
