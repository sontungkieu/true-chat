# Phase 1D RLAIF Context Labels MiMo50

## Summary

This report records the first real context-level RLAIF labeling run for Phase 1D.

The run labels a 50-action subset from the Phase 1D selector smoke action rows. It uses MiMo as a
context judge to identify whether the retrieved context is sufficient, which chunks are useful,
which chunks are redundant or irrelevant, and whether evidence is missing.

This is a subset audit, not a full benchmark result and not a runtime policy change.

## Inputs

```text
benchmark_results/rlaif/phase1d_selector_smoke/rlaif_actions.jsonl
```

Output:

```text
benchmark_results/rlaif/phase1d_selector_smoke/rlaif_context_labels_mimo50.jsonl
```

The job was run as two safe parallel shards to avoid concurrent writes to the same file:

```text
rlaif_actions_mimo_context_1_25.jsonl  -> rlaif_context_labels_mimo50_part1_1_25.jsonl
rlaif_actions_mimo_context_26_50.jsonl -> rlaif_context_labels_mimo50_part2_26_50.jsonl
```

The two part files were then merged and deduplicated by `action_id` in the original action order.

## Judge Setup

```text
judge_provider: mimo
judge_model: mimo-v2.5-pro
judge_version: rlaif-context-judge-v1
limit: 50
json_retries: 2
resume: true
```

## Label Summary

| Metric | Value |
| --- | ---: |
| label count | 50 |
| valid JSON | 50 |
| invalid JSON | 0 |
| errors | 0 |
| ambiguous | 4 |
| scored labels | 50 |
| sufficient | 18 |
| insufficient | 29 |
| missing evidence | 0 |
| dropped unknown chunk ids | 0 |
| sufficiency rate | 0.383 |

Non-ambiguous subset:

| Metric | Value |
| --- | ---: |
| non-ambiguous labels | 46 |
| non-ambiguous sufficient | 18 |
| non-ambiguous insufficient | 25 |
| mean context quality | 0.467 |
| mean evidence support | 0.409 |

The summary script counts invalid, ambiguous, errored, and missing labels explicitly. Ambiguous rows
are not treated as zero-quality labels. Some ambiguous rows still contain numeric diagnostic scores,
so downstream training should filter by `ambiguous=false` when it needs clean supervision.

## Chunk Selection

| Field | Mean | Min | Max |
| --- | ---: | ---: | ---: |
| selected chunks | 1.380 | 0 | 3 |
| redundant chunks | 0.120 | 0 | 2 |
| irrelevant chunks | 5.900 | 2 | 9 |

This is the first concrete signal that the context labeler is doing evidence selection rather than
only answer scoring. On average, it selects about 1-2 useful chunks and marks many retrieved chunks
as irrelevant.

## Score Summary

| Score | Mean | Std | Min | Max |
| --- | ---: | ---: | ---: | ---: |
| context quality | 0.478 | 0.416 | 0.000 | 1.000 |
| evidence support | 0.410 | 0.464 | 0.000 | 1.000 |
| minimality | 0.916 | 0.201 | 0.000 | 1.000 |

Interpretation:

- Context sufficiency is still weak on this subset: only 18 sufficient labels against 29
  insufficient labels.
- Evidence support is low on average, which means answer-level labels are not enough for future
  evidence-mask/KV work.
- Minimality is high when evidence is selected, but many rows still contain several irrelevant
  chunks.

## Research Impact

This closes an important Phase 1D infrastructure gap:

```text
answer-level RLAIF: populated on full 192-action run
context-level RLAIF: now populated on a 50-action subset
```

The next reward builder should not immediately consume these labels as default scalar reward.
First, add a non-default context-label merge path and report how many rows have clean,
non-ambiguous context supervision.

## Limitations

- This is a 50-action subset, not the full 192-action action set.
- MiMo labels are AI feedback, not human labels.
- The current selector policies still train from scalar reward rows; these context labels are not
  yet part of selector training.
- Missing evidence count is zero in this subset, but many contexts are still insufficient; the
  judge may be using insufficiency rather than the explicit missing-evidence flag.
- More retriever diversity is still needed before claiming retrieval-strategy selection.

## Next Step

Recommended next steps:

```text
1. Add rlaif-reward --context-labels as a non-default merge path.
2. Build context-clean reward reports with ambiguous=false filtering.
3. Run context labels on the remaining 142 action rows if the subset audit remains stable.
4. Only after that, add pairwise_ranker or context-aware selector features.
```
