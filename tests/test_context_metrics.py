from __future__ import annotations

from rag_bench.context_budget import ContextBudget, apply_context_budget
from rag_bench.context_metrics import aggregate_context_budget_metrics, aggregate_kv_estimates, context_budget_metrics
from rag_bench.types import RetrievalHit


def test_context_budget_metrics_are_non_negative_and_bounded() -> None:
    hits = [
        RetrievalHit("doc-1", 1.0, 1, "Title", "Alpha evidence."),
        RetrievalHit("doc-2", 0.5, 2, "Other", "Beta evidence."),
    ]
    budget = ContextBudget(policy="char-budget", max_chars=45)
    budgeted = apply_context_budget(hits, budget)

    metrics = context_budget_metrics(budgeted, budget, retrieved_docs=len(hits))

    assert metrics["original_context_chars"] >= metrics["kept_context_chars"]
    assert 0 <= metrics["compression_ratio"] <= 1
    assert metrics["estimated_token_savings"] >= 0
    assert metrics["latency_s"] >= 0
    assert metrics["retrieved_docs"] == 2


def test_aggregate_context_budget_metrics_averages_expected_fields() -> None:
    rows = [
        {
            "policy": "char-budget",
            "budget_chars": 100,
            "per_doc_budget_chars": None,
            "retrieved_docs": 2,
            "kept_docs": 1,
            "dropped_docs": 1,
            "original_context_chars": 80,
            "kept_context_chars": 40,
            "compression_ratio": 0.5,
            "original_context_est_tokens": 20,
            "kept_context_est_tokens": 10,
            "estimated_token_savings": 10,
            "latency_s": 0.01,
        },
        {
            "policy": "char-budget",
            "budget_chars": 100,
            "per_doc_budget_chars": None,
            "retrieved_docs": 4,
            "kept_docs": 2,
            "dropped_docs": 2,
            "original_context_chars": 120,
            "kept_context_chars": 60,
            "compression_ratio": 0.5,
            "original_context_est_tokens": 30,
            "kept_context_est_tokens": 15,
            "estimated_token_savings": 15,
            "latency_s": 0.03,
        },
    ]

    aggregate = aggregate_context_budget_metrics(rows)

    assert aggregate["context_policy"] == "char-budget"
    assert aggregate["avg_original_context_chars"] == 100
    assert aggregate["avg_kept_context_chars"] == 50
    assert aggregate["avg_estimated_token_savings"] == 12.5
    assert aggregate["avg_context_budget_latency_s"] == 0.02


def test_aggregate_kv_estimates_ignores_missing_rows() -> None:
    aggregate = aggregate_kv_estimates(
        [
            {"profile": "generic-small", "before_mb": 8.0, "after_mb": 4.0, "savings_mb": 4.0, "savings_ratio": 0.5, "note": "n"},
            None,
            {"profile": "generic-small", "before_mb": 4.0, "after_mb": 2.0, "savings_mb": 2.0, "savings_ratio": 0.5, "note": "n"},
        ]
    )

    assert aggregate["kv_profile"] == "generic-small"
    assert aggregate["avg_estimated_kv_cache_mb_before"] == 6.0
    assert aggregate["avg_estimated_kv_cache_savings_mb"] == 3.0
