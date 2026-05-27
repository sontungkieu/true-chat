# Budget Metrics

Each query result includes a `context_budget` object:

- `policy`
- `budget_chars`
- `per_doc_budget_chars`
- `requested_policy`
- `requested_policy_impl`
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

When `--context-policy adaptive-heuristic` is used, each query result also includes `adaptive_budget` metadata with the selected fixed policy, selected budget, reason code, extracted features, and configured budget candidates.

Aggregate rows include averages for the same core measurements. These metrics are computed even when `--skip-generation` is used, so retrieval-only matrix runs can compare context policies without spending model calls.

Adaptive aggregate metrics are stored under `context_budget.adaptive_budget` and include:

- `adaptive_selected_policy_counts`
- `adaptive_selected_budget_counts`
- `adaptive_reason_counts`
- `avg_adaptive_query_est_tokens`
- `avg_adaptive_score_gap`
- `avg_adaptive_score_entropy`

When generation runs, existing provider token usage remains separate from estimated context tokens. Provider `prompt_tokens` are exact only when the provider returns them.
