# RLAIF Context Label Summary

- Labels: `benchmark_results/rlaif/retriever_diversity_generation_mimo10_20260605T194500Z/rlaif_context_labels_mimo_v25.jsonl`

| Metric | Value |
| --- | ---: |
| label count | 300 |
| valid json count | 300 |
| invalid json count | 0 |
| ambiguous count | 47 |
| missing count | 0 |
| error count | 0 |
| scored label count | 300 |
| sufficient count | 134 |
| insufficient count | 158 |
| missing evidence count | 10 |
| dropped unknown chunk id count | 0 |
| sufficiency rate | 0.4589 |
| missing evidence rate | 0.4348 |

## Judge Counts

| Field | Value | Count |
| --- | --- | ---: |
| provider | `mimo` | 300 |
| model | `mimo-v2.5` | 300 |
| version | `rlaif-context-judge-v1` | 300 |

## Chunk Selection Statistics

| Field | N | Mean | Std | Min | Max |
| --- | ---: | ---: | ---: | ---: | ---: |
| `selected_chunk_ids` | 300 | 0.9633 | 0.7497 | 0 | 4 |
| `redundant_chunk_ids` | 300 | 0.0967 | 0.4479 | 0 | 4 |
| `irrelevant_chunk_ids` | 300 | 3.8167 | 0.9468 | 0 | 5 |

## Score Statistics

| Score | N | Mean | Std | Min | Max |
| --- | ---: | ---: | ---: | ---: | ---: |
| `context_quality_score` | 300 | 0.5047 | 0.3239 | 0.0000 | 1.0000 |
| `evidence_support_score` | 300 | 0.4355 | 0.4525 | 0.0000 | 1.0000 |
| `minimality_score` | 299 | 0.8920 | 0.2388 | 0.0000 | 1.0000 |

Invalid, ambiguous, errored, and missing labels are counted explicitly and are not treated as zero-quality context labels.
Dropped unknown chunk ids indicate judge-returned ids that were not present in the logged action row.
