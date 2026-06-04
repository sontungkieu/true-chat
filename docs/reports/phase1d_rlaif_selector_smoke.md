# Phase 1D RLAIF Selector Smoke

Date: 2026-06-04

Branch: `feature/rlaif-retrieval-context-v0`

Commit under test: `d46c179 feat(rlaif): add offline selector baselines`

## Scope

This is an offline selector smoke / resubstitution sanity check. It verifies that the current pipeline can run end-to-end on real BudgetRAG outputs:

```text
rlaif-build
-> rlaif-reward
-> rlaif-train
-> rlaif-eval
```

It is not a held-out generalization result. `rlaif-train` and `rlaif-eval` used the same logged reward rows, so the numbers below should be read as a pipeline sanity check and selector diagnostic.

## Input Data

Source data:

- BudgetRAG Phase 1C.3 SciFact generation outputs under `benchmark_results/budgetrag/`.
- Post-hoc RAGAS answer relevancy samples from `benchmark_results/budgetrag/phase1c3_ragas_mimo_posthoc/20260529T033825Z_all64_ragas_mimo_answer_relevancy_stratified_n3.per_sample.csv`.

Join method:

- Joined RAGAS rows back into their original `query_results.jsonl` source by `(source, query_id)`.
- Kept only completed RAGAS rows.
- Wrote local ignored staging input at `benchmark_results/rlaif/phase1d_selector_smoke_input/query_results.jsonl`.

Coverage:

| Item | Count |
| --- | ---: |
| RAGAS source action files | 64 |
| Completed RAGAS samples | 192 |
| Joined action-query rows | 192 |
| Missing source files | 0 |

Action coverage in the joined rows:

| Dimension | Distribution |
| --- | --- |
| Models | `mimo-v2.5-pro`: 96, `llama-3.1-8b-instant`: 48, `qwen/qwen3-32b`: 48 |
| Context policies | `adaptive-heuristic`: 96, `evidence-aware`: 48, `legacy`: 48 |
| Budgets | `4000`: 48, `8000`: 48, `1000`: 36, `2000`: 36, `16000`: 12, `32000`: 12 |

## Commands

```bash
uv run --frozen rag-bench rlaif-build \
  --inputs benchmark_results/rlaif/phase1d_selector_smoke_input/query_results.jsonl \
  --output-dir benchmark_results/rlaif/phase1d_selector_smoke

uv run --frozen rag-bench rlaif-reward \
  --actions benchmark_results/rlaif/phase1d_selector_smoke/rlaif_actions.jsonl \
  --feedback benchmark_results/rlaif/phase1d_selector_smoke/rlaif_feedback.jsonl \
  --output-dir benchmark_results/rlaif/phase1d_selector_smoke \
  --quality-weight 0.75 \
  --support-weight 0.10 \
  --token-weight 0.05 \
  --latency-weight 0.05 \
  --kv-weight 0.05 \
  --min-reward-delta 0.03 \
  --max-quality-regret 0.02

uv run --frozen rag-bench rlaif-train \
  --rewards benchmark_results/rlaif/phase1d_selector_smoke/rlaif_rewards.jsonl \
  --preferences benchmark_results/rlaif/phase1d_selector_smoke/rlaif_preferences.jsonl \
  --output benchmark_results/rlaif/phase1d_selector_smoke/rlaif_policy.json

uv run --frozen rag-bench rlaif-eval \
  --rewards benchmark_results/rlaif/phase1d_selector_smoke/rlaif_rewards.jsonl \
  --policy benchmark_results/rlaif/phase1d_selector_smoke/rlaif_policy.json \
  --out-md benchmark_results/rlaif/phase1d_selector_smoke/rlaif_eval_summary.md
```

## Build Summary

| Artifact | Count |
| --- | ---: |
| `rlaif_actions.jsonl` rows | 192 |
| `rlaif_feedback.jsonl` rows | 192 |
| `rlaif_rewards.jsonl` rows | 192 |
| `rlaif_preferences.jsonl` rows | 578 |
| Query groups | 55 |
| Action signatures | 77 |
| Invalid rows | 0 |

Feedback and reward coverage:

| Field | Value |
| --- | --- |
| Feedback provenance | `ragas`: 192 |
| Reward modes | `ragas`: 192 |
| Scored rewards | 192 / 192 |
| Missing or ambiguous feedback | 0 |

Preference coverage:

| Preference type | Count |
| --- | ---: |
| `context_policy_preference` | 289 |
| `retrieval_context_preference` | 289 |

Skipped preference reasons:

| Reason | Count |
| --- | ---: |
| `small_reward_delta` | 694 |
| `quality_guardrail_failed` | 2 |

## Selector Metrics

The policy artifact has `runtime_default_replacement=false`; none of these selectors replaces the runtime `adaptive-heuristic` default.

| Policy | Coverage | Mean reward | Mean quality | Token cost | Latency | KV cost | Oracle gap |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `fixed` | 0.109 | 0.429 | 0.543 | 0.225 | 0.082 | 0.351 | 0.035 |
| `cheapest` | 1.000 | 0.403 | 0.501 | 0.151 | 0.113 | 0.195 | 0.062 |
| `best_average` | 1.000 | 0.430 | 0.540 | 0.189 | 0.140 | 0.240 | 0.034 |
| `oracle_logged` | 1.000 | 0.464 | 0.576 | 0.171 | 0.114 | 0.220 | 0.000 |

Interpretation:

- `best_average` is the best non-oracle baseline in this smoke: it keeps full scored coverage and cuts oracle gap to about `0.034`.
- `cheapest` lowers token and KV cost, but quality and reward drop visibly. This is expected and confirms why efficiency-only selection should not be the default.
- `fixed` has low coverage because the single most common action signature is not available in most query groups. It is useful as a baseline, not as a realistic selector.
- `oracle_logged` is an offline upper bound over logged actions, not a deployable policy.

Top selected action labels:

| Policy | Top selections |
| --- | --- |
| `fixed` | `mimo-v2.5-pro / bm25 / legacy / conservative / budget8000`: 6 |
| `cheapest` | `mimo-v2.5-pro / bm25 / legacy / conservative / budget8000`: 4; `llama-3.1-8b-instant / bm25 / adaptive-heuristic / aggressive / budget1000`: 3; `mimo-v2.5-pro / bm25 / legacy / conservative / budget4000`: 3 |
| `best_average` | `mimo-v2.5-pro / bm25 / legacy / conservative / budget8000`: 4; `mimo-v2.5-pro / bm25 / evidence-aware / conservative / budget32000`: 3; `mimo-v2.5-pro / bm25 / adaptive-heuristic / balanced / budget2000`: 3; `qwen/qwen3-32b / bm25 / evidence-aware / conservative / budget2000`: 3 |
| `oracle_logged` | `mimo-v2.5-pro / bm25 / legacy / conservative / budget8000`: 4; `mimo-v2.5-pro / bm25 / evidence-aware / conservative / budget4000`: 3; `mimo-v2.5-pro / bm25 / adaptive-heuristic / aggressive / budget1000`: 3; `mimo-v2.5-pro / bm25 / evidence-aware / conservative / budget8000`: 3; `llama-3.1-8b-instant / bm25 / adaptive-heuristic / aggressive / budget1000`: 3 |

## Limitations

- This is not held-out evaluation. Train and eval used the same reward rows.
- Quality is based on RAGAS `answer_relevancy` only. There is no answer correctness, faithfulness, EM, or token-F1 in this smoke.
- The RAGAS data is sampled at `n=3/action`, not full-row coverage for every query in every matrix output.
- Query grouping is by benchmark, query id, top-k, and generator model. This produced 55 comparable query groups from 192 joined rows.
- Raw artifacts remain under ignored `benchmark_results/rlaif/phase1d_selector_smoke*`; this committed report is the curated record.

## Next Steps

1. Completed follow-up: held-out query evaluation is now documented in `docs/reports/phase1d_rlaif_heldout_eval.md`.
2. Add `rlaif-label-answers` and `rlaif-label-contexts` with `--dry-run`, `--resume`, and missing-key skip behavior.
3. Extend quality labels beyond answer relevancy: answer correctness, faithfulness, unsupported claims, and context evidence sufficiency.
4. Only after held-out quality guardrails pass, consider a simple reward model or contextual bandit baseline.
