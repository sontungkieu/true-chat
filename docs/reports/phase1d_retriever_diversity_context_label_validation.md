# RLAIF Context Label Validation

- Actions: `benchmark_results/rlaif/retriever_diversity_generation_mimo10_20260605T194500Z/rlaif_actions.jsonl`
- Merged output: `benchmark_results/rlaif/retriever_diversity_generation_mimo10_20260605T194500Z/rlaif_context_labels_mimo_v25.jsonl`

## Counts

| Metric | Value |
| --- | ---: |
| action count | 300 |
| label file count | 2 |
| label row count | 300 |
| unique label action count | 300 |
| known label action count | 300 |
| merged label count | 300 |
| missing action count | 0 |
| unknown action count | 0 |
| blank action id count | 0 |
| duplicate label row count | 0 |
| duplicate action id count | 0 |
| duplicate conflict count | 0 |
| shard overlap action count | 0 |
| invalid json count | 0 |
| ambiguous count | 47 |
| error count | 0 |
| missing reason count | 0 |
| clean usable label count | 253 |
| dropped unknown chunk id count | 0 |

## Files

| File | Rows | Unique actions | Duplicates | Unknown | Clean usable |
| --- | ---: | ---: | ---: | ---: | ---: |
| `benchmark_results/rlaif/retriever_diversity_generation_mimo10_20260605T194500Z/context_label_shards/rlaif_context_labels_mimo_part1_1_150.jsonl` | 150 | 150 | 0 | 0 | 127 |
| `benchmark_results/rlaif/retriever_diversity_generation_mimo10_20260605T194500Z/context_label_shards/rlaif_context_labels_mimo_part2_151_300.jsonl` | 150 | 150 | 0 | 0 | 126 |

## Diagnostic Samples

- missing action ids: N/A
- unknown action ids: N/A
- duplicate action ids: N/A
- duplicate conflict action ids: N/A
- shard overlap action ids: N/A

Merge rule: for duplicate action ids, the merged output keeps the highest-priority row: clean usable labels first, then non-ambiguous/non-invalid rows, then the first row. Unknown action ids are excluded from merged output.
