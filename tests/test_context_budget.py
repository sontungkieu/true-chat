from __future__ import annotations

from rag_bench.context_budget import (
    ContextBudget,
    apply_context_budget,
    context_items_to_text,
    estimate_tokens_from_chars,
    retrieval_hits_to_context_items,
)
from rag_bench.prompts import build_rag_messages, build_rag_messages_from_context
from rag_bench.types import Query, RetrievalHit


def test_retrieval_hits_convert_to_context_items() -> None:
    hit = RetrievalHit(
        doc_id="doc-1",
        score=2.5,
        rank=1,
        title="Title",
        text="Evidence text.",
        metadata={"kind": "web", "url": "https://example.test"},
    )

    item = retrieval_hits_to_context_items([hit])[0]

    assert item.id == "doc-1"
    assert item.score == 2.5
    assert item.rank == 1
    assert item.source == "web"
    assert item.metadata["url"] == "https://example.test"


def test_context_items_to_text_preserves_citation_block_format() -> None:
    hit = RetrievalHit("web-1", 1.0, 1, "Result title", "Snippet\nURL: https://example.test")
    item = retrieval_hits_to_context_items([hit])[0]

    assert context_items_to_text([item]) == "[web-1]\nResult title\nSnippet\nURL: https://example.test"


def test_legacy_policy_matches_existing_prompt_context_behavior() -> None:
    query = Query("q1", "What is alpha?")
    hits = [
        RetrievalHit("doc-1", 1.0, 1, "Alpha", "Alpha evidence sentence."),
        RetrievalHit("doc-2", 0.5, 2, "Beta", "Beta evidence sentence."),
    ]

    budgeted = apply_context_budget(hits, ContextBudget(policy="legacy", max_chars=46, query=query.text))
    old_messages = build_rag_messages(query, hits, max_context_chars=46)
    new_messages = build_rag_messages_from_context(query, budgeted.text)

    assert budgeted.text == "[doc-1]\nAlpha\nAlpha evidence sentence.\n\n---\n\n[doc-2]"
    assert old_messages == new_messages


def test_estimate_tokens_from_chars_uses_ceil_four_chars_per_token() -> None:
    assert estimate_tokens_from_chars(0) == 0
    assert estimate_tokens_from_chars(1) == 1
    assert estimate_tokens_from_chars(8) == 2
    assert estimate_tokens_from_chars(9) == 3
