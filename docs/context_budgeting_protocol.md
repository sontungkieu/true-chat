# Context Budgeting Protocol

BudgetRAG policies run after retrieval and before prompt construction.

Inputs:

- retrieved hits with `doc_id`, title, text, score, rank, and metadata;
- the user query text;
- a global character budget;
- optional per-document character budget.

Outputs:

- a context string formatted with citation blocks such as `[doc-id]`;
- kept context items for metrics and inspection;
- compression, estimated token, dropped-document, and latency metadata.

Policies:

- `legacy`: preserves the original sequential truncation behavior.
- `char-budget`: fills a global character budget in rank order.
- `per-doc-budget`: trims each document before applying the global budget.
- `score-density`: sorts by retrieval score per estimated token.
- `sentence-trim`: keeps rank order and trims final text at simple sentence boundaries when possible.
- `evidence-aware`: keeps the backward-compatible CLI name, but its current implementation subtype is `lexical-query-aware`. It scores sentence-like spans before answer generation using query overlap, retrieval score, and title overlap.

Token counts use `ceil(chars / 4)`. These are estimates for comparison and scheduling, not provider-tokenizer counts.
