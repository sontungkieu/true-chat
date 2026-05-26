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

This phase does not modify runtime model internals. KV-cache numbers are analytical estimates derived from estimated context tokens, not measured VRAM usage and not runtime KV pruning.
