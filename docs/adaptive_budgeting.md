# BudgetRAG Phase 1C Adaptive Budgeting

Phase 1C adds `adaptive-heuristic`, a deterministic rule-based context policy for benchmark runs.

The adaptive policy runs after retrieval and before prompt construction. It extracts lightweight query and retrieval features, chooses one existing fixed policy, applies that fixed policy, and records the decision metadata. It does not train a model, run reinforcement learning, use a bandit, inspect generated answers, or prune a live model KV cache.

## Extracted Features

- Query length in characters and estimated tokens.
- Number of retrieved candidates.
- Total, average, and maximum retrieved document text length.
- Top retrieval scores, top score gap, mean score, population score standard deviation, and score entropy.
- Count of candidates with missing or invalid scores.

Token counts use the same analytical estimate as Phase 1B: `ceil(chars / 4)`.

## Selection Rules

The current selector implementation is `deterministic-rule-v1`.

- No candidates: use `char-budget` with the medium budget.
- One document dominates the context length: if `max_doc_chars >= 2400` and `max_doc_chars >= avg_doc_chars * 2.5`, use `per-doc-budget` with the large budget and a capped per-document budget.
- All retrieval scores are missing: use `evidence-aware` with the medium budget.
- Flat or uncertain retrieval scores: if `score_gap <= max(0.1, abs(top1_score) * 0.08)` or `score_entropy >= 1.0`, use `evidence-aware` with the large budget.
- High-confidence retrieval scores: if `score_gap >= max(0.25, abs(top1_score) * 0.15)`, use `score-density` with the small budget, or the medium budget for long queries.
- Long query fallback: if `query_est_tokens >= 32`, use `evidence-aware` with the medium budget.
- Balanced fallback: use `char-budget` with the medium budget.

Default adaptive budget candidates are:

- Small: `1000` characters.
- Medium: `2000` characters.
- Large: `4000` characters.

They can be changed with `--adaptive-small-budget`, `--adaptive-medium-budget`, and `--adaptive-large-budget`.

## CLI Example

```bash
uv run rag-bench run \
  --bench scifact \
  --retrievers bm25 \
  --top-k 5 \
  --limit 10 \
  --skip-generation \
  --context-policy adaptive-heuristic \
  --adaptive-small-budget 1000 \
  --adaptive-medium-budget 2000 \
  --adaptive-large-budget 4000 \
  --kv-profile qwen2.5-14b
```

## Recorded Metadata

Each query result includes `adaptive_budget` when `--context-policy adaptive-heuristic` is used. The metadata records:

- selected fixed policy;
- selected context budget;
- selected per-document budget, when applicable;
- deterministic reason code;
- extracted adaptive features;
- configured small, medium, and large budgets.

Aggregate metrics include selected policy counts, selected budget counts, reason counts, average adaptive query tokens, average adaptive score gap, and average adaptive score entropy.

## Matrix Behavior

`scripts/run_budgetrag_matrix.py` accepts `adaptive-heuristic` in `--context-policies`. For adaptive jobs, each `--context-budgets` value is also passed as `--adaptive-medium-budget`. Small and large adaptive candidates keep their CLI defaults in the matrix helper.
