from __future__ import annotations

from rag_bench.context_budget import ContextBudget, apply_context_budget
from rag_bench.types import Query, RetrievalHit


SYSTEM_PROMPT = (
    "You are a concise RAG answerer. Answer using only the provided contexts. "
    "If the contexts are insufficient, say you do not know. Cite source document ids in square brackets."
)


def build_rag_messages(
    query: Query,
    hits: list[RetrievalHit],
    *,
    max_context_chars: int,
) -> list[dict[str, str]]:
    budgeted = apply_context_budget(
        hits,
        ContextBudget(policy="legacy", max_chars=max_context_chars, query=query.text),
    )
    return build_rag_messages_from_context(query, budgeted.text)


def build_rag_messages_from_context(
    query: Query,
    context: str,
) -> list[dict[str, str]]:
    user_prompt = f"Question:\n{query.text}\n\nContexts:\n{context}\n\nAnswer:"
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt},
    ]
