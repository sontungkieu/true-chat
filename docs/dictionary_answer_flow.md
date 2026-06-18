# Dictionary Answer Flow

This note tracks the dictionary answer layers so rendering fixes do not hide backend data-shaping issues.

```mermaid
flowchart TD
    U[User query] --> R[Retriever and query planner]
    R --> C[Canonicalize dictionary redirects]
    C -->|Keep canonical definition| H[Canonical hits]
    C -->|Attach lightweight metadata| A[redirect aliases and redirect doc ids]
    H --> P[Prompt context assembly]
    A --> P
    P --> L[LLM or deterministic fallback]
    L --> S[Source payload]
    H --> S
    A --> S
    S --> UI[UI renderer]
    UI --> Cards[Top compact dictionary cards]
    UI --> Refs[Related documents table]
    UI --> Copy[Rendered-answer copy text]
```

Rules:

- Redirect-only entries such as `HEADWORD nh TARGET` are merged before prompt assembly when the target entry is also retrieved.
- The prompt keeps the canonical entry with the definition and only carries alias/ref labels as lightweight metadata.
- The source payload preserves redirect doc ids so citations to an alias can still open the canonical source.
- UI cards stay compact and should not implement semantic de-duplication as their primary responsibility.
