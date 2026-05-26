# BudgetRAG Phase 1B Limitations

- Context token counts are estimated with `ceil(chars / 4)`.
- KV-cache savings are analytical estimates, not measured runtime memory savings.
- No runtime KV-cache pruning is implemented in this phase.
- `evidence-aware` is query-aware, not answer-aware; it runs before generation.
- Sentence splitting uses lightweight regex and punctuation heuristics, not an NLP parser.
- Chat UI BudgetRAG controls are intentionally deferred; benchmark CLI support is the priority.
- Web search results are not crawled or cached. Titles, snippets, and URLs remain the only web context.
