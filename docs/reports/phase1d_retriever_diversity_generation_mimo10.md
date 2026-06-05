# Phase 1D Retriever-Diversity MiMo Generation Smoke

## Summary

This run is the first retriever-diverse generation and answer-label smoke for
Phase 1D. It follows the retrieval-only coverage run, but turns generation on
for a small 10-query subset so the pipeline can exercise answer labels,
AI-judge rewards, preferences, and selector diagnostics across multiple
retrievers.

It is still a small logged-candidate diagnostic, not a benchmark claim. The
main result is that retriever-diverse rows now have non-empty reward/preference
supervision, while the selector results remain too small and unstable to claim
generalization.

## Run Setup

The matrix was run under:

```bash
RUN_NAME=retriever_diversity_generation_mimo10_20260605T194500Z
LIMIT=10
SKIP_GENERATION=0
MODELS=mimo_v25
RETRIEVERS=bm25,graph-bm25,hybrid-rrf
CONTEXT_BUDGETS=1000,4000
CONTEXT_POLICIES=legacy,evidence-aware,score-density,adaptive-heuristic
ADAPTIVE_PROFILES=balanced,aggressive
MAX_COMPLETION_TOKENS=256
CONTINUE_ON_ERROR=1
scripts/run_retriever_diversity_budgetrag_matrix.sh
```

After the first few sequential cells completed, the remaining work was
parallelized into two non-overlapping MiMo shards:

```text
shard A: bm25 + graph-bm25
shard B: hybrid-rrf
```

The two shards wrote to the same run name and reused completed cells. The final
matrix contains:

```text
3 retrievers x 5 policy/profile variants x 2 budgets = 30 jobs
30 jobs x 10 queries = 300 generated action rows
```

Model provenance is standard MiMo V2.5:

```text
generator_model = mimo-v2.5
generation_model_role = long-context-judge-generator
judge_model = mimo-v2.5
```

This run does not use `mimo-v2.5-pro`.

## Output Paths

Raw outputs are ignored and stored under:

```text
benchmark_results/budgetrag/phase1d_retriever_diversity_smoke/
  retriever_diversity_generation_mimo10_20260605T194500Z/
```

Normalized RLAIF outputs are ignored and stored under:

```text
benchmark_results/rlaif/retriever_diversity_generation_mimo10_20260605T194500Z/
```

Generated local summaries include:

```text
budgetrag_summary.md
rlaif_actions.jsonl
rlaif_feedback.jsonl
rlaif_answer_labels_mimo_v25.jsonl
rlaif_context_labels_mimo_v25.jsonl
rlaif_answer_label_summary_mimo_v25.md
rlaif_context_label_summary_mimo_v25.md
reward_mimo_answer/rlaif_rewards.jsonl
reward_mimo_answer/rlaif_preferences.jsonl
context_penalty_025/rlaif_rewards.jsonl
context_penalty_025/rlaif_preferences.jsonl
reward_mimo_answer/split_sweep_seeds_1_2_3_4_5_42/
reward_mimo_answer/action_coverage.md
```

## Generation Result Counts

| Metric | Value |
| --- | ---: |
| BudgetRAG metrics files | 30 |
| BudgetRAG query result files | 30 |
| RLAIF actions | 300 |
| Query groups | 10 |
| Retrievers per query | 3 / 3 |
| Action rows per query | 30 / 30 |
| Non-empty generated answers | 223 |
| Empty generated answers | 77 |
| Generation errors | 0 |

The 77 empty answers are not request errors. The API reported completion token
usage, but `message.content` was empty. This matches the known MiMo V2.5
behavior where low completion caps can be consumed by hidden reasoning before
visible content is emitted. For this run, empty answers are treated as explicit
`missing_answer` rows and are not converted to zero-quality labels.

For future full generation runs, `MAX_COMPLETION_TOKENS=256` is too low for
standard MiMo V2.5. Use a larger cap before spending on a full 2250-row
generation matrix.

## Answer Labeling

The first attempt to judge answers with `--max-completion-tokens 768` produced
mostly empty judge content and was discarded into `.bad_cap768.jsonl` files.
The accepted run used standard MiMo V2.5 with `--max-completion-tokens 2048`
over two shards.

| Metric | Value |
| --- | ---: |
| Answer labels | 300 |
| Valid JSON labels | 299 |
| Invalid JSON labels | 1 |
| Ambiguous labels | 114 |
| Missing-answer labels | 77 |
| Labels with numeric diagnostics | 222 |
| Mean `overall_quality` over numeric labels | 0.823 |
| Request errors | 0 |

The distinction between numeric diagnostics and clean reward supervision matters.
`rlaif-reward` filters ambiguous labels instead of treating them as score zero.

## Context Labeling

The follow-up context-level judge run used standard MiMo V2.5 over two
non-overlapping 150-action shards and then merged/deduped the outputs:

| Metric | Value |
| --- | ---: |
| Context labels | 300 |
| Valid JSON labels | 300 |
| Invalid JSON labels | 0 |
| Ambiguous labels | 47 |
| Clean usable labels | 253 |
| Missing action ids | 0 |
| Unknown action ids | 0 |
| Duplicate rows | 0 |
| Sufficient contexts | 134 |
| Insufficient contexts | 158 |
| Missing-evidence rows | 10 |
| Mean selected chunks | 0.963 |
| Mean irrelevant chunks | 3.817 |
| Mean context quality | 0.505 |
| Mean evidence support | 0.436 |

This confirms that the retriever-diverse subset now has both answer-level and
context-level RLAIF supervision. It also shows why context supervision matters:
the judge usually selected about one useful chunk while marking nearly four
chunks as irrelevant. Answer-level reward alone would not expose that context
noise.

## Reward And Preference Build

```bash
uv run --frozen rag-bench rlaif-reward \
  --actions benchmark_results/rlaif/retriever_diversity_generation_mimo10_20260605T194500Z/rlaif_actions.jsonl \
  --feedback benchmark_results/rlaif/retriever_diversity_generation_mimo10_20260605T194500Z/rlaif_feedback.jsonl \
  --answer-labels benchmark_results/rlaif/retriever_diversity_generation_mimo10_20260605T194500Z/rlaif_answer_labels_mimo_v25.jsonl \
  --output-dir benchmark_results/rlaif/retriever_diversity_generation_mimo10_20260605T194500Z/reward_mimo_answer
```

| Metric | Value |
| --- | ---: |
| Reward rows | 300 |
| Scored reward rows | 186 |
| Reward mode `ai_judge` | 186 |
| Reward mode `ambiguous_feedback` | 77 |
| Reward mode `missing_quality` | 37 |
| Preferences | 1559 |
| Context-policy preferences | 370 |
| Retrieval-context preferences | 1189 |
| Quality guardrail skips | 8 |
| Small reward-delta skips | 873 |

The reward build turns this subset from a coverage-only run into a small
retriever-diverse quality-supervision run.

## Context Reward Candidate

A non-default context reward candidate was built with:

```text
context_quality_blend_weight = 0.50
context_support_blend_weight = 0.50
context_insufficient_penalty_weight = 0.25
```

| Metric | Answer-only | Context candidate |
| --- | ---: | ---: |
| Reward rows | 300 | 300 |
| Scored reward rows | 186 | 212 |
| Preferences | 1559 | 2412 |
| Context-policy preferences | 370 | 571 |
| Retrieval-context preferences | 1189 | 1841 |
| Quality-guardrail skips | 8 | 123 |
| Small reward-delta skips | 873 | 493 |

The candidate changed 156/300 reward rows. Of the changed rows, 138 decreased
and 18 increased, with mean changed-only delta `-0.301`. This is the expected
direction for a context-evidence penalty, but the size of the shift means the
candidate must remain non-default until more query groups and calibration data
are available.

## Reward By Retriever

| Retriever | Rows | Scored rewards | Mean reward | Mean quality component | Mean token cost |
| --- | ---: | ---: | ---: | ---: | ---: |
| `bm25` | 100 | 62 | 0.613 | 0.908 | 0.576 |
| `graph-bm25` | 100 | 58 | 0.613 | 0.926 | 0.513 |
| `hybrid-rrf` | 100 | 66 | 0.514 | 0.862 | 0.603 |

In this small subset, `bm25` and `graph-bm25` have similar reward means, while
`hybrid-rrf` has fewer empty answers but lower judged quality among scored rows.
This is a useful diagnostic, not a stable retriever ranking.

## Reward By Budget And Policy

| Budget | Rows | Scored rewards | Mean reward | Mean quality component | Mean token cost |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 1000 | 150 | 104 | 0.520 | 0.855 | 0.397 |
| 4000 | 150 | 82 | 0.652 | 0.951 | 0.780 |

| Policy/profile | Rows | Scored rewards | Mean reward | Mean quality component | Mean token cost |
| --- | ---: | ---: | ---: | ---: | ---: |
| `adaptive-heuristic / aggressive` | 60 | 39 | 0.585 | 0.885 | 0.455 |
| `adaptive-heuristic / balanced` | 60 | 36 | 0.607 | 0.936 | 0.721 |
| `evidence-aware / conservative` | 60 | 41 | 0.603 | 0.915 | 0.599 |
| `legacy / conservative` | 60 | 35 | 0.625 | 0.917 | 0.505 |
| `score-density / conservative` | 60 | 35 | 0.464 | 0.831 | 0.552 |

The 4000-character budget has better judged quality and reward in this subset,
but it also has higher token/KV cost. This keeps the quality-efficiency
trade-off visible.

## Evidence Quality By Retriever

Ambiguous context labels are excluded from this table.

| Retriever | Rows | Sufficient | Selected chunks | Irrelevant chunks | Context quality | Evidence support |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `bm25` | 83 | 0.542 | 0.855 | 4.000 | 0.507 | 0.459 |
| `graph-bm25` | 87 | 0.529 | 0.943 | 3.943 | 0.528 | 0.496 |
| `hybrid-rrf` | 83 | 0.518 | 0.711 | 4.084 | 0.470 | 0.439 |

In this subset, graph-BM25 has the strongest context-quality and
evidence-support means, while hybrid-RRF is noisier. This is a useful
retriever-diversity diagnostic, not a stable retriever ranking.

## Multi-Seed Selector Diagnostic

The six-seed query-level split sweep used only 10 query groups, so each eval
split contains 2 query groups and 60 rows. This is too small for a stable
selector claim.

| Policy | Coverage | Reward | Quality | Oracle gap |
| --- | ---: | ---: | ---: | ---: |
| `cheapest` | 0.833 +/- 0.236 | 0.658 +/- 0.331 | 0.925 +/- 0.168 | 0.148 +/- 0.331 |
| `best_average` | 0.500 +/- 0.289 | 0.625 +/- 0.351 | 0.910 +/- 0.180 | 0.179 +/- 0.351 |
| `family_smoothed_best_average` | 0.500 +/- 0.289 | 0.625 +/- 0.351 | 0.910 +/- 0.180 | 0.179 +/- 0.351 |
| `shrinkage_smoothed_best_average` | 0.667 +/- 0.373 | 0.263 +/- 0.716 | 0.710 +/- 0.395 | 0.544 +/- 0.717 |
| `linear_reward_model` | 0.667 +/- 0.373 | 0.555 +/- 0.321 | 0.890 +/- 0.174 | 0.251 +/- 0.321 |
| `smoothed_linear_selector` | 0.583 +/- 0.186 | 0.355 +/- 0.645 | 0.792 +/- 0.311 | 0.451 +/- 0.643 |
| `oracle_logged` | 1.000 +/- 0.000 | 0.804 +/- 0.003 | 1.000 +/- 0.000 | 0.000 +/- 0.000 |

The selector sweep confirms that the offline pipeline works on retriever-diverse
rows, but it does not yet support a learned-selector claim. Coverage and reward
variance are dominated by the tiny eval splits and missing/ambiguous labels.

The context-candidate selector sweep is harsher. It increases scored reward rows
and preference pairs, but the stricter context penalty reduces selector reward
and makes simple smoothed baselines fragile on the 10-query split. This is
evidence that context labels should be treated as calibration supervision first,
not as a default selector target.

## Action Coverage

| Level | Unique | Singleton rate | Mean eval-row coverage |
| --- | ---: | ---: | ---: |
| `action_id` | 300 | 1.000 | 0.000 |
| `exact_signature` | 34 | 0.029 | 0.997 |
| `retrieval_context_family` | 15 | 0.000 | 1.000 |
| `context_policy` | 4 | 0.000 | 1.000 |
| `retriever` | 3 | 0.000 | 1.000 |

The retrieval-context family coverage is now complete across retrievers. This
is the desired precondition for later retrieval-context allocation experiments.

## Interpretation

This run strengthens the Phase 1D evidence in a narrow but important way:

```text
retrieval-only diversity run:
  closes logged-action coverage only

generation + answer-label subset:
  adds first retriever-diverse quality supervision
  creates 186 scored rewards and 1559 preferences
  verifies that BM25, graph-BM25, and hybrid-RRF rows can enter the same RLAIF pipeline

generation + answer/context-label subset:
  creates full 300-row context supervision
  exposes evidence noise and insufficiency per retriever/policy
  confirms context reward is useful but currently too harsh as a default target
```

The run also exposes two bottlenecks:

```text
generation cap too low:
  77/300 empty answers at max_completion_tokens=256

data still too small:
  only 10 query groups, so selector eval is unstable
```

## Next Step

Do not run the full 2250-row generation matrix with `MAX_COMPLETION_TOKENS=256`.
The next safe step is either:

1. rerun the same 10-query subset with a larger generation cap to reduce empty
   answers; or
2. run a 20-query subset with a larger generation cap before scaling to all 50
   queries.

Only after answer-empty rate and label coverage are acceptable should the full
retriever-diverse generation matrix be expanded.
