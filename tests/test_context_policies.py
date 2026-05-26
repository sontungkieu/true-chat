from __future__ import annotations

from rag_bench.context_budget import ContextBudget, apply_context_budget
from rag_bench.types import RetrievalHit


def test_char_budget_respects_global_budget() -> None:
    hits = [
        RetrievalHit("doc-1", 1.0, 1, "One", "A" * 100),
        RetrievalHit("doc-2", 0.9, 2, "Two", "B" * 100),
    ]

    budgeted = apply_context_budget(hits, ContextBudget(policy="char-budget", max_chars=50))

    assert len(budgeted.text) <= 50
    assert budgeted.policy_name == "char-budget"
    assert budgeted.dropped_items == 1


def test_per_doc_budget_trims_each_document_before_global_budget() -> None:
    hits = [
        RetrievalHit("doc-1", 1.0, 1, "", "abcdef ghijkl"),
        RetrievalHit("doc-2", 0.9, 2, "", "mnopqr stuvwx"),
    ]

    budgeted = apply_context_budget(
        hits,
        ContextBudget(policy="per-doc-budget", max_chars=80, per_doc_max_chars=6),
    )

    assert "abcdef" in budgeted.text
    assert "ghijkl" not in budgeted.text
    assert "mnopqr" in budgeted.text
    assert "stuvwx" not in budgeted.text


def test_score_density_prefers_high_score_short_document() -> None:
    hits = [
        RetrievalHit("long-low", 1.0, 1, "", "low " * 200),
        RetrievalHit("short-high", 10.0, 2, "", "high signal"),
    ]

    budgeted = apply_context_budget(hits, ContextBudget(policy="score-density", max_chars=40))

    assert "[short-high]" in budgeted.text
    assert "[long-low]" not in budgeted.text


def test_sentence_trim_uses_sentence_boundary_when_possible() -> None:
    hits = [
        RetrievalHit(
            "doc-1",
            1.0,
            1,
            "",
            "First complete sentence. Second sentence should be trimmed before it ends.",
        )
    ]

    budgeted = apply_context_budget(hits, ContextBudget(policy="sentence-trim", max_chars=48))

    assert "First complete sentence." in budgeted.text
    assert "Second sentence" not in budgeted.text


def test_evidence_aware_preserves_doc_id_and_prefers_query_overlap() -> None:
    hits = [
        RetrievalHit("doc-1", 0.5, 1, "Bananas", "Bananas are yellow. Apples are red."),
        RetrievalHit("doc-2", 0.4, 2, "Alpha", "Alpha particles appear in this evidence. unrelated tail."),
    ]

    budgeted = apply_context_budget(
        hits,
        ContextBudget(policy="evidence-aware", max_chars=90, query="alpha evidence"),
    )

    assert budgeted.policy_name == "evidence-aware"
    assert "[doc-2]" in budgeted.text
    assert "Alpha particles appear in this evidence." in budgeted.text
    assert "Bananas are yellow" not in budgeted.text


def test_empty_hits_return_no_retrieved_context() -> None:
    budgeted = apply_context_budget([], ContextBudget(policy="char-budget", max_chars=100))

    assert budgeted.text == "No retrieved context."
    assert budgeted.dropped_items == 0
    assert budgeted.compression_ratio == 1.0
