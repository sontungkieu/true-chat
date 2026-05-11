from rag_bench.metrics import exact_match, retrieval_metrics_for_query, token_f1
from rag_bench.types import Query, RetrievalHit, RetrievalResult


def test_retrieval_metrics_for_query() -> None:
    result = RetrievalResult(
        query=Query("q1", "question"),
        hits=[
            RetrievalHit(doc_id="d2", score=3.0, rank=1),
            RetrievalHit(doc_id="d1", score=2.0, rank=2),
            RetrievalHit(doc_id="d3", score=1.0, rank=3),
        ],
        latency_s=0.25,
    )

    metrics = retrieval_metrics_for_query(result, {"d1": 1, "d4": 1}, top_k=3)

    assert metrics["hit@k"] == 1.0
    assert metrics["precision@k"] == 1 / 3
    assert metrics["recall@k"] == 0.5
    assert metrics["mrr@k"] == 0.5
    assert 0.0 < metrics["ndcg@k"] < 1.0
    assert metrics["retrieval_latency_s"] == 0.25


def test_answer_metrics_return_none_without_references() -> None:
    assert exact_match("answer", []) is None
    assert token_f1("answer", []) is None


def test_answer_metrics_score_against_best_reference() -> None:
    refs = ["The capital is Paris", "Paris"]

    assert exact_match("paris", refs) == 1.0
    assert token_f1("capital paris", refs) == 0.8
