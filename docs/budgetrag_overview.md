# BudgetRAG Phase 1B Overview

BudgetRAG Phase 1B adds a context-budgeting layer between retrieval and prompt construction in the benchmark CLI.

The pipeline becomes:

```text
query
-> retriever
-> retrieved hits
-> context policy
-> budgeted context
-> prompt construction
-> optional generation
-> retrieval, quality, context, latency, and KV estimate metrics
```

The default policy is `legacy`, which preserves the original rank-ordered context truncation behavior. Other policies make the context budget explicit so retrieval strategies can be compared under fixed prompt-size pressure.

The current `evidence-aware` policy keeps its CLI-compatible name, but the recorded implementation subtype is `lexical-query-aware`. It ranks candidate spans before answer generation using query overlap, retrieval score, and title overlap. It is not answer-aware citation verification and does not perform semantic entailment checking.

This phase does not modify runtime model internals. KV-cache numbers are analytical estimates derived from estimated context tokens, not measured VRAM usage and not runtime KV pruning.
