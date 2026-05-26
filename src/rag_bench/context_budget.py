from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

from rag_bench.types import RetrievalHit


NO_RETRIEVED_CONTEXT = "No retrieved context."
CONTEXT_SEPARATOR = "\n\n---\n\n"


@dataclass(frozen=True)
class ContextItem:
    id: str
    text: str
    title: str = ""
    score: float | None = None
    rank: int | None = None
    source: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ContextBudget:
    policy: str = "legacy"
    max_chars: int = 12_000
    per_doc_max_chars: int | None = None
    max_docs: int | None = None
    query: str = ""
    estimate_tokens: bool = True


@dataclass
class BudgetedContext:
    items: list[ContextItem]
    text: str
    policy_name: str
    original_chars: int
    kept_chars: int
    original_est_tokens: int
    kept_est_tokens: int
    compression_ratio: float
    dropped_items: int
    latency_s: float
    metadata: dict[str, Any] = field(default_factory=dict)


def apply_context_budget(hits: list[RetrievalHit] | list[ContextItem], budget: ContextBudget) -> BudgetedContext:
    from rag_bench.context_policies import apply_context_policy

    items = hits if not hits or isinstance(hits[0], ContextItem) else retrieval_hits_to_context_items(hits)  # type: ignore[index]
    return apply_context_policy(items, budget)  # type: ignore[arg-type]


def validate_context_budget(budget: ContextBudget) -> None:
    if budget.max_chars <= 0:
        raise ValueError("context budget max_chars must be positive")
    if budget.per_doc_max_chars is not None and budget.per_doc_max_chars <= 0:
        raise ValueError("per_doc_max_chars must be positive when provided")
    if budget.max_docs is not None and budget.max_docs <= 0:
        raise ValueError("max_docs must be positive when provided")


def retrieval_hits_to_context_items(hits: list[RetrievalHit]) -> list[ContextItem]:
    items: list[ContextItem] = []
    for hit in hits:
        metadata = dict(hit.metadata)
        source_value = metadata.get("source") or metadata.get("kind")
        items.append(
            ContextItem(
                id=hit.doc_id,
                text=hit.text,
                title=hit.title,
                score=hit.score,
                rank=hit.rank,
                source=str(source_value) if source_value else None,
                metadata=metadata,
            )
        )
    return items


def context_items_to_text(items: list[ContextItem]) -> str:
    blocks = [block for item in items if (block := context_item_to_text_block(item))]
    return CONTEXT_SEPARATOR.join(blocks) if blocks else NO_RETRIEVED_CONTEXT


def context_item_to_text_block(item: ContextItem) -> str:
    title = f"{item.title}\n" if item.title else ""
    return f"[{item.id}]\n{title}{item.text}".strip()


def estimate_tokens_from_chars(chars: int) -> int:
    return int(math.ceil(max(0, chars) / 4))


def build_budgeted_context(
    *,
    original_items: list[ContextItem],
    kept_items: list[ContextItem],
    text: str,
    policy_name: str,
    latency_s: float,
    metadata: dict[str, Any] | None = None,
) -> BudgetedContext:
    original_text = context_items_to_text(original_items)
    original_chars = len(original_text)
    kept_chars = len(text)
    original_doc_ids = {item.id for item in original_items if context_item_to_text_block(item)}
    kept_doc_ids = {item.id for item in kept_items if context_item_to_text_block(item)}
    compression_ratio = kept_chars / original_chars if original_chars else 1.0
    return BudgetedContext(
        items=kept_items,
        text=text,
        policy_name=policy_name,
        original_chars=original_chars,
        kept_chars=kept_chars,
        original_est_tokens=estimate_tokens_from_chars(original_chars),
        kept_est_tokens=estimate_tokens_from_chars(kept_chars),
        compression_ratio=compression_ratio,
        dropped_items=max(0, len(original_doc_ids) - len(kept_doc_ids)),
        latency_s=latency_s,
        metadata={
            "original_item_count": len(original_items),
            "kept_item_count": len(kept_items),
            "original_doc_count": len(original_doc_ids),
            "kept_doc_count": len(kept_doc_ids),
            **(metadata or {}),
        },
    )
