# Budget Metrics

Each query result includes a `context_budget` object:

- `policy`
- `budget_chars`
- `per_doc_budget_chars`
- `retrieved_docs`
- `kept_docs`
- `dropped_docs`
- `original_context_chars`
- `kept_context_chars`
- `compression_ratio`
- `original_context_est_tokens`
- `kept_context_est_tokens`
- `estimated_token_savings`
- `latency_s`

Aggregate rows include averages for the same core measurements. These metrics are computed even when `--skip-generation` is used, so retrieval-only matrix runs can compare context policies without spending model calls.

When generation runs, existing provider token usage remains separate from estimated context tokens. Provider `prompt_tokens` are exact only when the provider returns them.
