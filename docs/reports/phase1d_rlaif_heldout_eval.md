# Phase 1D RLAIF Held-Out Query Evaluation

Date: 2026-06-04

Branch: `feature/rlaif-retrieval-context-v0`

## Scope

This report upgrades the previous selector smoke from resubstitution evaluation to a deterministic held-out split by query id.

The split is by:

```text
benchmark + query_id
```

This is intentionally not a random action-row split. Every action for the same SciFact query stays in the same train or eval split, so the offline selector cannot train on one action for a query and evaluate on another action for that same query.

## Input

Starting artifacts:

```text
benchmark_results/rlaif/phase1d_selector_smoke/rlaif_rewards.jsonl
benchmark_results/rlaif/phase1d_selector_smoke/rlaif_preferences.jsonl
```

These came from the Phase 1D selector smoke over real Phase 1C.3 BudgetRAG outputs joined with post-hoc RAGAS answer relevancy.

## Commands

```bash
uv run --frozen rag-bench rlaif-split \
  --rewards benchmark_results/rlaif/phase1d_selector_smoke/rlaif_rewards.jsonl \
  --preferences benchmark_results/rlaif/phase1d_selector_smoke/rlaif_preferences.jsonl \
  --output-dir benchmark_results/rlaif/phase1d_selector_smoke/split_seed42 \
  --train-ratio 0.8 \
  --seed 42

uv run --frozen rag-bench rlaif-train \
  --rewards benchmark_results/rlaif/phase1d_selector_smoke/split_seed42/train_rewards.jsonl \
  --preferences benchmark_results/rlaif/phase1d_selector_smoke/split_seed42/train_preferences.jsonl \
  --output benchmark_results/rlaif/phase1d_selector_smoke/split_seed42/rlaif_policy.json

uv run --frozen rag-bench rlaif-eval \
  --rewards benchmark_results/rlaif/phase1d_selector_smoke/split_seed42/eval_rewards.jsonl \
  --policy benchmark_results/rlaif/phase1d_selector_smoke/split_seed42/rlaif_policy.json \
  --out-md benchmark_results/rlaif/phase1d_selector_smoke/split_seed42/rlaif_eval_summary.md \
  --split-manifest benchmark_results/rlaif/phase1d_selector_smoke/split_seed42/split_manifest.json
```

## Split Summary

| Metric | Value |
| --- | ---: |
| Train ratio | 0.8 |
| Seed | 42 |
| Train query count | 27 |
| Eval query count | 7 |
| Train reward rows | 178 |
| Eval reward rows | 14 |
| Train preferences | 572 |
| Eval preferences | 6 |
| Dropped cross-split preferences | 0 |
| Dropped missing-action preferences | 0 |

Note: the split rule is `benchmark + query_id`. The selector evaluator reports 9 eval query groups because policy evaluation groups also include generation model/top-k dimensions. This is expected: the held-out unit is the original query id, while selector comparison groups are action-comparison groups.

## Held-Out Selector Metrics

The eval summary reports:

```text
held_out_query_eval = true
runtime_default_replacement = false
```

The selector artifact remains offline-only and does not replace the runtime `adaptive-heuristic` default.

| Policy | Coverage | Mean reward | Mean quality | Token cost | Latency | KV cost | Paired oracle gap |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `fixed` | 0.000 | N/A | N/A | N/A | N/A | N/A | N/A |
| `cheapest` | 1.000 | 0.377 | 0.476 | 0.202 | 0.075 | 0.284 | 0.006 |
| `best_average` | 0.889 | 0.425 | 0.538 | 0.239 | 0.083 | 0.335 | 0.008 |
| `oracle_logged` | 1.000 | 0.383 | 0.488 | 0.251 | 0.075 | 0.321 | 0.000 |

The oracle gap is computed pairwise over query groups where the evaluated policy selected a scored action. This avoids the misleading negative gap that can appear when a policy has lower coverage than the oracle. The aggregate mean reward columns are coverage-dependent; a lower-coverage policy can show a higher mean than `oracle_logged` if it skips lower-reward query groups, so paired oracle gap is the safer comparison.

## Comparison With Resubstitution Smoke

| Policy | Resubstitution reward | Held-out reward | Resubstitution quality | Held-out quality |
| --- | ---: | ---: | ---: | ---: |
| `fixed` | 0.429 | N/A | 0.543 | N/A |
| `cheapest` | 0.403 | 0.377 | 0.501 | 0.476 |
| `best_average` | 0.430 | 0.425 | 0.540 | 0.538 |
| `oracle_logged` | 0.464 | 0.383 | 0.576 | 0.488 |

Interpretation:

- `best_average` remains the strongest non-oracle baseline by held-out reward and quality, but coverage is `0.889` rather than full coverage.
- `cheapest` still has full coverage and lower cost, but quality/reward are lower than `best_average`.
- `fixed` has no held-out coverage for this split because its single trained fixed signature does not appear in eval query groups.
- `oracle_logged` is an eval-set upper bound over logged actions, not a deployable selector. Its held-out mean is lower than the resubstitution mean because the held-out query set is small and harder/different.

## Top Held-Out Selections

| Policy | Top selected actions |
| --- | --- |
| `cheapest` | `mimo-v2.5-pro / bm25 / evidence-aware / conservative / budget4000`: 2; `mimo-v2.5-pro / bm25 / legacy / conservative / budget8000`: 2; `qwen/qwen3-32b / bm25 / adaptive-heuristic / aggressive / budget8000`: 1 |
| `best_average` | `mimo-v2.5-pro / bm25 / evidence-aware / conservative / budget16000`: 2; `mimo-v2.5-pro / bm25 / evidence-aware / conservative / budget4000`: 2; `mimo-v2.5-pro / bm25 / legacy / conservative / budget8000`: 2 |
| `oracle_logged` | `mimo-v2.5-pro / bm25 / evidence-aware / conservative / budget4000`: 2; `mimo-v2.5-pro / bm25 / legacy / conservative / budget8000`: 2; `qwen/qwen3-32b / bm25 / adaptive-heuristic / aggressive / budget8000`: 1 |

## Limitations

- Held-out eval is by query id, but the dataset is still small: 7 held-out queries and 14 held-out reward rows.
- Quality is still based on RAGAS `answer_relevancy`; correctness, faithfulness, unsupported claims, and context sufficiency labels are not yet included.
- `best_average` is still a table baseline, not a learned generalizing reward model.
- Raw split/eval artifacts remain under ignored `benchmark_results/rlaif/phase1d_selector_smoke/split_seed42/`; this report is the committed curated record.

## Next Steps

1. Add larger held-out runs once richer feedback is available.
2. Add `rlaif-label-answers` for answer correctness, faithfulness, and unsupported-claim labels.
3. Add `rlaif-label-contexts` for evidence sufficiency, redundancy, and missing evidence.
4. After held-out quality guardrails are stable, add a simple reward model or contextual bandit baseline.
