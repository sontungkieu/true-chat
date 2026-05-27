from __future__ import annotations

import math

from rag_bench.adaptive_features import extract_adaptive_budget_features
from rag_bench.context_budget import ContextItem


def test_extract_adaptive_budget_features_summarizes_query_docs_and_scores() -> None:
    features = extract_adaptive_budget_features(
        "alpha beta gamma delta",
        [
            ContextItem("doc-1", "A" * 10, score=2.0),
            ContextItem("doc-2", "B" * 20, score=1.0),
            ContextItem("doc-3", "C" * 30, score=None),
        ],
    )

    assert features.query_chars == 22
    assert features.query_est_tokens == 6
    assert features.num_candidates == 3
    assert features.total_doc_chars == 60
    assert features.avg_doc_chars == 20
    assert features.max_doc_chars == 30
    assert features.top1_score == 2.0
    assert features.top2_score == 1.0
    assert features.score_gap == 1.0
    assert features.score_mean == 1.5
    assert features.score_std == 0.5
    assert features.score_entropy is not None
    assert features.normalized_score_gap == 0.5
    assert features.normalized_score_entropy is not None
    assert features.score_confidence is not None
    assert features.missing_score_count == 1


def test_extract_adaptive_budget_features_handles_missing_and_invalid_scores() -> None:
    features = extract_adaptive_budget_features(
        "",
        [
            ContextItem("doc-1", "text", score=None),
            ContextItem("doc-2", "text", score=math.inf),
        ],
    )

    assert features.top1_score is None
    assert features.score_gap is None
    assert features.score_entropy is None
    assert features.normalized_score_gap is None
    assert features.normalized_score_entropy is None
    assert features.score_confidence is None
    assert features.missing_score_count == 2


def test_extract_adaptive_budget_features_handles_one_scored_candidate() -> None:
    features = extract_adaptive_budget_features("alpha", [ContextItem("doc-1", "a", score=3.0)])

    assert features.top1_score == 3.0
    assert features.top2_score is None
    assert features.score_gap is None
    assert features.normalized_score_gap is None
    assert features.normalized_score_entropy is None
    assert features.score_confidence is None


def test_extract_adaptive_budget_features_normalizes_negative_scores() -> None:
    features = extract_adaptive_budget_features(
        "alpha",
        [
            ContextItem("doc-1", "a", score=-1.0),
            ContextItem("doc-2", "b", score=-2.0),
        ],
    )

    assert features.top1_score == -1.0
    assert features.top2_score == -2.0
    assert features.score_gap == 1.0
    assert features.score_entropy is not None
    assert features.normalized_score_gap == 1.0
    assert features.normalized_score_entropy is not None


def test_extract_adaptive_budget_features_handles_equal_zero_scores() -> None:
    features = extract_adaptive_budget_features(
        "alpha",
        [
            ContextItem("doc-1", "a", score=0.0),
            ContextItem("doc-2", "b", score=0.0),
        ],
    )

    assert features.score_gap == 0.0
    assert features.normalized_score_gap == 0.0
    assert features.normalized_score_entropy is not None
    assert round(features.normalized_score_entropy, 6) == 1.0
    assert features.score_confidence is not None
    assert round(features.score_confidence, 6) == 0.0
