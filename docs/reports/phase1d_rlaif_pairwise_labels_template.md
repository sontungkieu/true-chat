# Phase 1D RLAIF Pairwise Labels Template

This report template should be filled after a real `rlaif-label-pairs` run.

## Goal

Direct pairwise RLAIF labels test whether reward-derived preferences agree with an AI judge that compares two retrieval-context actions side by side.

The judge sees:

- the same query;
- Action A as the reward-derived chosen action;
- Action B as the reward-derived rejected action;
- each action's retriever, context policy, budget, answer, retrieved context, token cost, latency, and estimated KV cost.

The judge should not browse or use external knowledge. Quality and evidence support dominate efficiency; lower cost should only win when answer quality and support are similar.

## Commands

```bash
uv run --frozen rag-bench rlaif-label-pairs \
  --actions benchmark_results/rlaif/<run-name>/rlaif_actions.jsonl \
  --rewards benchmark_results/rlaif/<run-name>/rlaif_rewards.jsonl \
  --preferences benchmark_results/rlaif/<run-name>/rlaif_preferences.jsonl \
  --output benchmark_results/rlaif/<run-name>/rlaif_pairwise_labels_mimo.jsonl \
  --judge-provider mimo \
  --judge-model mimo-v2.5-pro \
  --limit 50 \
  --resume \
  --sleep-seconds 0.5

uv run --frozen python scripts/summarize_rlaif_pairwise_labels.py \
  --input benchmark_results/rlaif/<run-name>/rlaif_pairwise_labels_mimo.jsonl \
  --out-md benchmark_results/rlaif/<run-name>/rlaif_pairwise_labels_mimo_summary.md \
  --out-json benchmark_results/rlaif/<run-name>/rlaif_pairwise_labels_mimo_summary.json
```

## Required Summary

| Metric | Value |
| --- | ---: |
| label_count | TODO |
| valid_json_count | TODO |
| invalid_json_count | TODO |
| ambiguous_count | TODO |
| tie_count | TODO |
| A_win_count | TODO |
| B_win_count | TODO |
| agreement_with_reward_preference | TODO |
| disagreement_with_reward_preference | TODO |
| agreement_rate | TODO |
| avg_confidence | TODO |
| quality_regret_count | TODO |
| unsupported_claim_risk_count | TODO |

## Reward Preference vs Judge Preference

| Bucket | Count | Notes |
| --- | ---: | --- |
| judge agrees with reward-derived preference | TODO | Judge chose A. |
| judge disagrees with reward-derived preference | TODO | Judge chose B. |
| tie or ambiguous | TODO | Not treated as agreement. |

## Disagreement Examples

Add compact examples after the run:

| Query | Reward-chosen action | Judge-chosen action | Quality winner | Efficiency winner | Rationale |
| --- | --- | --- | --- | --- | --- |
| TODO | TODO | TODO | TODO | TODO | TODO |

## Interpretation

- High agreement means the scalar reward formula roughly matches direct pairwise AI preference on this sample.
- High disagreement means the reward formula, quality guardrail, or efficiency weights need calibration.
- High tie count may mean action differences are too small or the judge prompt is under-specified.
- High unsupported-claim risk indicates answer-level labels should penalize hallucination more strongly.

## Limitations

- These are AI-judge preferences, not human labels.
- This is offline calibration data, not a runtime policy.
- Do not train DPO, reward models, or replace `adaptive-heuristic` until held-out evaluation and quality guardrails pass.
- Do not mix live web-search action labels with reproducible BEIR benchmark claims.
