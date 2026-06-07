# Phase 1D Retriever-Diversity A1 MiMo V2.5 Evaluation

This report summarizes the A1-medium retriever-diverse run after answer-level labeling and the first stratified context-level labeling pass. It uses standard MiMo V2.5 labels over the 1500 generated SciFact action rows, plus 600 stratified context labels selected from the same action log.

Raw outputs remain under ignored `benchmark_results/`. The committed report records only curated counts and metrics.

## Setup

- Benchmark: SciFact sampled 50 queries.
- Retrievers: BM25, graph-BM25, hybrid-RRF.
- Policy/profile variants: legacy, evidence-aware, score-density, adaptive-balanced, adaptive-aggressive.
- Budgets: 1000 and 4000 characters.
- Generator: standard MiMo V2.5.
- Max completion tokens: 2048.
- Logged rows: 1500 action rows.
- Answer judge: MiMo / `mimo-v2.5`, `rlaif-answer-judge-v1`.
- Context judge: MiMo / `mimo-v2.5`, `rlaif-context-judge-v1`, stratified 600/1500 rows.
- Claim boundary: offline logged-candidate evaluation only; no runtime default replacement.

## Output Handling

The three Kaggle kernels completed as private notebooks:

- `codemaivanngu/tcem1-202606061632`
- `codemaivanngu/tcem2-202606061632`
- `codemaivanngu/tcem3-202606061632`

Kaggle output also exposed the cloned repo and `.venv` inside the output listing. Only root-level JSONL, manifest, and summary files were copied into the local ignored `benchmark_results/` directory. No `.venv`, `.secrets`, `.env`, or `kaggle.json` files were copied into the repo output directory.

## Answer-Label Validation

| Metric | Value |
| --- | ---: |
| Action rows | 1500 |
| Label files | 3 |
| Label rows | 1500 |
| Unique label action ids | 1500 |
| Merged labels | 1500 |
| Missing action ids | 0 |
| Unknown action ids | 0 |
| Duplicate action ids | 0 |
| Duplicate conflicts | 0 |
| Invalid JSON lines | 0 |
| Invalid JSON labels | 0 |
| Ambiguous labels | 40 |
| Error labels | 0 |
| Scored labels | 1500 |
| Clean usable labels | 1460 |

Shard-level clean usable labels:

| Shard | Rows | Clean usable |
| --- | ---: | ---: |
| 1-500 | 500 | 483 |
| 501-1000 | 500 | 489 |
| 1001-1500 | 500 | 488 |

The validation result is clean enough to rebuild answer-only RLAIF rewards. Ambiguous labels are not converted into zero-quality rows.

## Answer-Label Score Summary

| Score | N | Mean | Std | Min | Max |
| --- | ---: | ---: | ---: | ---: | ---: |
| Overall quality | 1500 | 0.890 | 0.288 | 0.000 | 1.000 |
| Answer correctness | 1499 | 0.890 | 0.300 | 0.000 | 1.000 |
| Evidence support | 1497 | 0.890 | 0.299 | 0.000 | 1.000 |
| Faithfulness | 1495 | 0.894 | 0.290 | 0.000 | 1.000 |
| Unsupported-claim penalty | 1491 | 0.108 | 0.302 | 0.000 | 1.000 |
| Conciseness | 1498 | 0.987 | 0.054 | 0.000 | 1.000 |

RAGAS correlation is not available for this run because the A1 feedback rows intentionally started as missing feedback before AI-judge labeling.

## Reward Rebuild

The answer-only reward was rebuilt with `--answer-labels`.

| Metric | Value |
| --- | ---: |
| Reward rows | 1500 |
| Scored rewards | 1460 |
| Missing-quality rewards | 40 |
| AI-judge reward rows | 1460 |
| Fallback-to-feedback rows | 40 |
| Preferences | 17026 |
| Context-policy preferences | 4087 |
| Retrieval-context preferences | 12939 |
| Missing-quality preference skips | 40 |
| Quality-guardrail skips | 93 |
| Small-delta skips | 9904 |

This answer-only reward remains the baseline. A non-default context reward candidate is evaluated below after merging the stratified context labels.

## Answer Quality By Retriever

| Retriever | Rows | Quality | Correctness | Support | Unsupported risk | Reward | Token cost | KV cost |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| BM25 | 500 | 0.899 | 0.898 | 0.899 | 0.096 | 0.617 | 0.478 | 0.605 |
| graph-BM25 | 500 | 0.911 | 0.909 | 0.909 | 0.087 | 0.642 | 0.458 | 0.559 |
| hybrid-RRF | 500 | 0.862 | 0.862 | 0.862 | 0.141 | 0.546 | 0.512 | 0.673 |

Graph-BM25 is the best retriever on mean answer quality, evidence support, unsupported-risk penalty, reward, token cost, and analytical KV cost in this answer-level A1 run. Hybrid-RRF remains noisier and more expensive under the current implementation.

## Answer Quality By Budget

| Budget | Rows | Quality | Correctness | Support | Unsupported risk | Reward | Token cost | KV cost |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1000 chars | 750 | 0.832 | 0.830 | 0.831 | 0.164 | 0.526 | 0.328 | 0.316 |
| 4000 chars | 750 | 0.948 | 0.950 | 0.949 | 0.052 | 0.676 | 0.638 | 0.908 |

The 4000-character budget strongly improves answer-level quality and support, but it also increases token and KV cost. The final selection problem remains a quality-cost trade-off rather than a pure compression problem.

## Answer Quality By Context Policy

| Policy | Rows | Quality | Correctness | Support | Unsupported risk | Reward |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| adaptive-heuristic | 600 | 0.930 | 0.929 | 0.926 | 0.067 | 0.666 |
| evidence-aware | 300 | 0.916 | 0.915 | 0.916 | 0.079 | 0.634 |
| legacy | 300 | 0.912 | 0.915 | 0.911 | 0.103 | 0.643 |
| score-density | 300 | 0.764 | 0.761 | 0.771 | 0.226 | 0.394 |

The score-density policy is weak in this A1 setting. Adaptive-heuristic and legacy/evidence-aware policies remain stronger answer-level candidates, with different cost profiles.

## Selected Retriever-Policy Signals

| Retriever / policy | Rows | Quality | Support | Unsupported risk | Reward |
| --- | ---: | ---: | ---: | ---: | ---: |
| graph-BM25 / legacy | 100 | 0.921 | 0.917 | 0.083 | 0.694 |
| graph-BM25 / adaptive-heuristic | 200 | 0.934 | 0.931 | 0.062 | 0.676 |
| graph-BM25 / evidence-aware | 100 | 0.939 | 0.935 | 0.062 | 0.668 |
| BM25 / adaptive-heuristic | 200 | 0.929 | 0.919 | 0.058 | 0.676 |
| hybrid-RRF / adaptive-heuristic | 200 | 0.926 | 0.928 | 0.080 | 0.646 |
| hybrid-RRF / score-density | 100 | 0.668 | 0.675 | 0.339 | 0.236 |

The run gives a useful answer-level signal: graph-BM25 is no longer only a coverage addition; it is competitive or better in several action cells. The strongest caution is that the oracle still mixes retrievers on held-out query groups, so a single global retriever ranking is too strong.

## Multi-Seed Held-Out Selector Sweep

The selector sweep uses six deterministic query-level splits: `1,2,3,4,5,42`. Each split trains on 40 query groups and evaluates on 10 held-out query groups.

| Policy | Coverage | Reward | Quality | Token cost | KV cost | Oracle gap |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| fixed | 0.983 +/- 0.037 | 0.716 +/- 0.040 | 0.978 +/- 0.019 | 0.668 +/- 0.015 | 0.997 +/- 0.002 | 0.099 +/- 0.040 |
| cheapest | 0.900 +/- 0.058 | 0.270 +/- 0.115 | 0.702 +/- 0.056 | 0.249 +/- 0.002 | 0.248 +/- 0.000 | 0.546 +/- 0.116 |
| best_average | 1.000 +/- 0.000 | 0.661 +/- 0.111 | 0.937 +/- 0.070 | 0.554 +/- 0.135 | 0.758 +/- 0.260 | 0.154 +/- 0.111 |
| family_smoothed_best_average | 1.000 +/- 0.000 | 0.661 +/- 0.111 | 0.937 +/- 0.070 | 0.554 +/- 0.135 | 0.758 +/- 0.260 | 0.154 +/- 0.111 |
| shrinkage_smoothed_best_average | 1.000 +/- 0.000 | 0.659 +/- 0.112 | 0.938 +/- 0.072 | 0.577 +/- 0.137 | 0.808 +/- 0.265 | 0.156 +/- 0.111 |
| linear_reward_model | 1.000 +/- 0.000 | 0.734 +/- 0.032 | 0.987 +/- 0.018 | 0.656 +/- 0.014 | 0.994 +/- 0.001 | 0.081 +/- 0.032 |
| smoothed_linear_selector | 0.967 +/- 0.047 | 0.657 +/- 0.126 | 0.931 +/- 0.076 | 0.480 +/- 0.121 | 0.659 +/- 0.240 | 0.158 +/- 0.125 |
| oracle_logged | 1.000 +/- 0.000 | 0.815 +/- 0.001 | 1.000 +/- 0.000 | 0.264 +/- 0.008 | 0.249 +/- 0.000 | 0.000 +/- 0.000 |

In this answer-only A1 sweep, `linear_reward_model` is the strongest non-oracle selector by mean reward and oracle gap while keeping full coverage. However, it pays high token/KV cost, and the fixed baseline is also strong because this small logged matrix has stable high-quality exact signatures. This is still logged-candidate offline evaluation, not online generalization.

## Seed-42 Selected Retriever Distribution

Seed 42 is not the full claim, but it is useful for inspecting allocation behavior:

- `best_average`: graph-BM25 9/10, BM25 1/10.
- `shrinkage_smoothed_best_average`: graph-BM25 10/10.
- `linear_reward_model`: graph-BM25 10/10.
- `smoothed_linear_selector`: graph-BM25 9/10, BM25 1/10.
- `oracle_logged`: BM25 5/10, graph-BM25 3/10, hybrid-RRF 2/10.

The learned and averaged selectors mostly choose graph-BM25 on this split. The oracle still selects all three retrievers, which suggests that retriever allocation should remain query-conditioned rather than collapsed into one global retriever.

## Stratified Context Labels

The planned 600-row context-label subset has now been labeled with MiMo V2.5. It covers 20 rows in each `retrieval_strategy x context_policy x adaptive_profile x budget_chars` cell.

| Metric | Value |
| --- | ---: |
| Context labels | 600 |
| Valid JSON labels | 598 |
| Clean usable labels | 548 |
| Ambiguous labels | 52 |
| Invalid JSON labels | 2 |
| Missing / unknown / duplicate action ids | 0 |
| Dropped unknown chunk ids | 1 |
| Sufficient contexts | 410 |
| Insufficient contexts | 177 |
| Sufficiency rate | 0.6985 |
| Mean selected chunks | 1.2300 |
| Mean irrelevant chunks | 3.4867 |
| Mean context quality | 0.6431 |
| Mean evidence support | 0.6567 |
| Mean minimality | 0.9020 |

Compared with the earlier 192-row context-label set, A1 has higher sufficiency, context quality, and evidence support. This is a stronger context-level signal, but it is still a stratified 600/1500 subset rather than full A1 context supervision.

## Context Evidence Quality By Retriever

| Retriever | Clean rows | Sufficiency | Context quality | Evidence support | Irrelevant chunks | Token cost | KV savings MB |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| BM25 | 182 | 0.720 | 0.647 | 0.685 | 3.544 | 687.247 | 1321.566 |
| graph-BM25 | 180 | 0.706 | 0.642 | 0.659 | 3.483 | 579.606 | 1451.406 |
| hybrid-RRF | 186 | 0.780 | 0.682 | 0.730 | 3.548 | 689.935 | 1270.842 |

This changes the interpretation of A1. Graph-BM25 remains strongest on answer-level reward/quality, but the stratified context labels do not simply confirm a global graph-BM25 win. Hybrid-RRF has the strongest mean context sufficiency, context quality, and evidence support on the labeled subset, while graph-BM25 has the lowest token cost, the largest analytical KV savings, and slightly fewer irrelevant chunks. The retriever ranking is therefore metric-dependent.

## Context Evidence Quality By Policy

| Context policy | Clean rows | Sufficiency | Context quality | Evidence support | Irrelevant chunks | Token cost |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| adaptive-heuristic | 223 | 0.700 | 0.623 | 0.669 | 3.511 | 683.852 |
| evidence-aware | 107 | 0.692 | 0.651 | 0.622 | 3.542 | 635.505 |
| legacy | 109 | 0.789 | 0.693 | 0.781 | 3.569 | 630.651 |
| score-density | 109 | 0.798 | 0.697 | 0.717 | 3.495 | 628.413 |

On this context-labeled subset, score-density and legacy look better than they did under answer-only reward. This is exactly why context-level RLAIF is useful: answer-level quality and evidence-level sufficiency can diverge.

## Context Reward Candidate 0.25

The 600 context labels were merged into a non-default context reward candidate with context quality blend 0.50, context support blend 0.50, and insufficient-context penalty 0.25.

| Metric | Value |
| --- | ---: |
| Reward rows | 1500 |
| Scored rewards | 1492 |
| Context labels used | 548 |
| Context labels fallback / invalid | 52 |
| Missing context labels | 900 |
| Preferences | 20934 |
| Context-policy preferences | 4965 |
| Retrieval-context preferences | 15969 |
| Changed reward rows vs answer-only | 464 |
| Negative deltas | 326 |
| Positive deltas | 138 |
| Mean changed delta | -0.112 |

The context candidate changes 464/1500 rows and increases preferences from 17026 to 20934. It remains non-default because it covers only 600 context-labeled rows and changes the reward target materially.

## Context-Candidate Selector Sweep

The same six query-level splits (`1,2,3,4,5,42`) were rerun on the context reward candidate.

| Policy | Coverage | Reward | Quality | Token cost | KV cost | Oracle gap |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| fixed | 1.000 +/- 0.000 | 0.573 +/- 0.103 | 0.889 +/- 0.045 | 0.482 +/- 0.210 | 0.621 +/- 0.372 | 0.240 +/- 0.103 |
| cheapest | 1.000 +/- 0.000 | 0.358 +/- 0.104 | 0.793 +/- 0.041 | 0.249 +/- 0.002 | 0.248 +/- 0.000 | 0.456 +/- 0.105 |
| best_average | 0.983 +/- 0.037 | 0.613 +/- 0.127 | 0.906 +/- 0.061 | 0.321 +/- 0.052 | 0.298 +/- 0.082 | 0.201 +/- 0.126 |
| shrinkage_smoothed_best_average | 0.983 +/- 0.037 | 0.555 +/- 0.096 | 0.885 +/- 0.045 | 0.320 +/- 0.053 | 0.310 +/- 0.079 | 0.259 +/- 0.096 |
| linear_reward_model | 1.000 +/- 0.000 | 0.669 +/- 0.081 | 0.967 +/- 0.026 | 0.623 +/- 0.017 | 0.982 +/- 0.028 | 0.145 +/- 0.082 |
| smoothed_linear_selector | 0.983 +/- 0.037 | 0.619 +/- 0.133 | 0.925 +/- 0.048 | 0.379 +/- 0.083 | 0.473 +/- 0.189 | 0.195 +/- 0.131 |
| oracle_logged | 1.000 +/- 0.000 | 0.814 +/- 0.003 | 1.000 +/- 0.000 | 0.266 +/- 0.014 | 0.273 +/- 0.035 | 0.000 +/- 0.000 |

`linear_reward_model` remains the strongest non-oracle selector by reward and oracle gap, but it still pays high token/KV cost. `smoothed_linear_selector` is cheaper, but lower reward. This is still offline logged-candidate evaluation and does not replace the runtime policy.

## Context-Candidate Selected Retriever Distribution

Counts below aggregate selected held-out query groups across the six splits.

| Policy | Selected | BM25 | graph-BM25 | hybrid-RRF |
| --- | ---: | ---: | ---: | ---: |
| cheapest | 60 | 27 | 20 | 13 |
| best_average | 60 | 42 | 17 | 1 |
| shrinkage_smoothed_best_average | 60 | 31 | 27 | 2 |
| linear_reward_model | 60 | 8 | 51 | 1 |
| smoothed_linear_selector | 60 | 38 | 19 | 3 |
| oracle_logged | 60 | 31 | 20 | 9 |

The learned linear selector strongly favors graph-BM25 under the context candidate, but the oracle still selects all three retrievers. This argues for query-conditioned retrieval-context allocation rather than one global retriever ranking.

## Context-Label Shard Validation

The 600 context rows were labeled in four Kaggle shards.

| Shard | Rows | Clean usable |
| --- | ---: | ---: |
| 1-150 | 150 | 137 |
| 151-300 | 150 | 134 |
| 301-450 | 150 | 137 |
| 451-600 | 150 | 140 |

No shard overlap, duplicate action ids, unknown action ids, or missing action ids were found. Kaggle output initially included cloned repo and `.venv` directories; local postprocess removed `.secrets`, `.venv`, `.git`, and clone directories before merge.

## Decision Gate

A1-medium passes the answer-label gate:

- generation: 1500/1500 non-empty, 0 generation errors;
- answer labels: 1500/1500 valid JSON, 0 invalid JSON, 1460 clean usable labels;
- answer-only rewards: 1460 scored rewards, 17026 preferences;
- context labels: 600 stratified rows, 548 clean usable labels, 0 missing/unknown/duplicate action ids;
- graph-BM25 has the strongest mean answer quality/reward among retrievers, while hybrid-RRF has the strongest mean context sufficiency/support on the stratified context subset;
- selector sweeps produce interpretable retriever choices and keep runtime replacement disabled.

The next bottleneck is not label availability for this subset, but calibration and coverage: context labels cover 600/1500 A1 rows, and the answer-level and context-level retriever signals disagree. A final retriever-quality ranking should remain provisional unless the remaining context labels or a larger A2 run confirm the pattern.

## Limitations

- Labels are RLAIF-style AI feedback, not human labels.
- This is logged-candidate offline evaluation, not online RL.
- Context reward remains non-default because the context-labeled subset is 600/1500 rows.
- The observed graph-BM25 advantage is answer-level; context-level evidence currently favors hybrid-RRF on sufficiency/support and graph-BM25 on efficiency/noise. This is not a final retriever benchmark claim.
