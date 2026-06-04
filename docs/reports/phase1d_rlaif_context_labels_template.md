# Phase 1D RLAIF Context Labels Template

This report template should be filled after a real `rlaif-label-contexts` run.

## Run Metadata

- Labels file: `benchmark_results/rlaif/<run-name>/rlaif_context_labels_mimo.jsonl`
- Summary file: `benchmark_results/rlaif/<run-name>/rlaif_context_labels_mimo_summary.md`
- Judge provider/model: `mimo` / `mimo-v2.5-pro`
- Prompt version: `rlaif-context-judge-v1`
- Source actions: `benchmark_results/rlaif/<run-name>/rlaif_actions.jsonl`

## Commands

```bash
uv run --frozen rag-bench rlaif-label-contexts \
  --actions benchmark_results/rlaif/<run-name>/rlaif_actions.jsonl \
  --output benchmark_results/rlaif/<run-name>/rlaif_context_labels_mimo.jsonl \
  --judge-provider mimo \
  --judge-model mimo-v2.5-pro \
  --resume \
  --json-retries 1 \
  --max-completion-tokens 4096

uv run --frozen python scripts/summarize_rlaif_context_labels.py \
  --labels benchmark_results/rlaif/<run-name>/rlaif_context_labels_mimo.jsonl \
  --out-md benchmark_results/rlaif/<run-name>/rlaif_context_labels_mimo_summary.md \
  --out-json benchmark_results/rlaif/<run-name>/rlaif_context_labels_mimo_summary.json
```

## Required Summary

| Metric | Value |
| --- | ---: |
| label_count | TODO |
| valid_json_count | TODO |
| invalid_json_count | TODO |
| ambiguous_count | TODO |
| sufficient_count | TODO |
| missing_evidence_count | TODO |
| dropped_unknown_chunk_id_count | TODO |
| avg_selected_chunks | TODO |
| avg_redundant_chunks | TODO |
| avg_irrelevant_chunks | TODO |
| context_quality_mean | TODO |
| evidence_support_mean | TODO |
| minimality_mean | TODO |

## Evidence Selection

Summarize how many chunks are selected as useful evidence versus redundant or irrelevant context.

| Bucket | Mean | Notes |
| --- | ---: | --- |
| selected chunks | TODO | TODO |
| redundant chunks | TODO | TODO |
| irrelevant chunks | TODO | TODO |

## Examples

Add 3-5 compact examples after the full run:

| Query | Selected chunks | Redundant chunks | Irrelevant chunks | Sufficient | Rationale |
| --- | --- | --- | --- | --- | --- |
| TODO | TODO | TODO | TODO | TODO | TODO |

## Interpretation

- Context labels are judged only from logged question, optional answer, and retrieved chunks.
- Invalid, missing, ambiguous, or errored labels must stay null-quality labels, not zero-quality labels.
- Dropped unknown chunk ids measure judge chunk-id hallucination and should be reviewed before using labels as evidence masks.

## Limitations

- This is context-level RLAIF feedback, not a runtime KV pruning result.
- It should be used for context sufficiency preferences and future evidence-mask experiments only after summary coverage is acceptable.
- Do not mix live web-search action labels with reproducible BEIR benchmark claims.
