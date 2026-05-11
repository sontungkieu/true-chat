from __future__ import annotations

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
    context_blocks: list[str] = []
    used_chars = 0
    for hit in hits:
        title = f"{hit.title}\n" if hit.title else ""
        block = f"[{hit.doc_id}]\n{title}{hit.text}".strip()
        if not block:
            continue
        remaining = max_context_chars - used_chars
        if remaining <= 0:
            break
        if len(block) > remaining:
            block = block[:remaining].rstrip()
        context_blocks.append(block)
        used_chars += len(block)

    context = "\n\n---\n\n".join(context_blocks) if context_blocks else "No retrieved context."
    user_prompt = f"Question:\n{query.text}\n\nContexts:\n{context}\n\nAnswer:"
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt},
    ]
