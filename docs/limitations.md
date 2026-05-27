# BudgetRAG Phase 1B Limitations

- Context token counts are estimated with `ceil(chars / 4)`.
- KV-cache savings are analytical estimates, not measured runtime memory savings.
- No runtime KV-cache pruning is implemented in this phase.
- `evidence-aware` is a lexical/query-aware retention policy, not answer-aware verification; it runs before generation.
- `evidence-aware` may miss paraphrases, mishandle negation, and prefer lexical overlap even when a sentence is not citation-faithful evidence for the final answer.
- Future evidence policies may add BM25 sentence scoring, embedding sentence scoring, cross-encoder reranking, or answer-aware verification.
- Sentence splitting uses lightweight regex and punctuation heuristics, not an NLP parser.
- Chat UI BudgetRAG controls are intentionally deferred; benchmark CLI support is the priority.
- Web search results are not crawled or cached. Titles, snippets, and URLs remain the only web context.
