# RLAIF Answer Label Summary

- Labels: `benchmark_results/rlaif/retriever_diversity_generation_mimo10_20260605T194500Z/rlaif_answer_labels_mimo_v25.jsonl`
- RAGAS feedback: `N/A`

| Metric | Value |
| --- | ---: |
| label count | 300 |
| valid json count | 299 |
| invalid json count | 1 |
| ambiguous count | 114 |
| missing count | 77 |
| error count | 0 |
| scored label count | 222 |

## Judge Counts

| Field | Value | Count |
| --- | --- | ---: |
| provider | `mimo` | 300 |
| model | `mimo-v2.5` | 300 |
| version | `rlaif-answer-judge-v1` | 300 |

## Score Statistics

| Score | N | Mean | Std | Min | Max |
| --- | ---: | ---: | ---: | ---: | ---: |
| `overall_quality` | 222 | 0.8225 | 0.3429 | 0.0000 | 1.0000 |
| `quality_score` | 222 | 0.8225 | 0.3429 | 0.0000 | 1.0000 |
| `answer_correctness` | 222 | 0.8230 | 0.3661 | 0.0000 | 1.0000 |
| `evidence_support` | 217 | 0.8171 | 0.3700 | 0.0000 | 1.0000 |
| `faithfulness` | 211 | 0.8213 | 0.3586 | 0.0000 | 1.0000 |
| `citation_faithfulness` | 211 | 0.8213 | 0.3586 | 0.0000 | 1.0000 |
| `unsupported_claim_penalty` | 219 | 0.1333 | 0.3326 | 0.0000 | 1.0000 |
| `conciseness` | 222 | 0.9450 | 0.1634 | 0.0000 | 1.0000 |

## RAGAS Correlation

- Joined pairs: 0
- Pearson overall quality vs RAGAS answer relevancy: N/A
- Pearson quality score vs RAGAS answer relevancy: N/A

Invalid, ambiguous, and missing labels are counted explicitly and are not treated as zero-quality labels.
