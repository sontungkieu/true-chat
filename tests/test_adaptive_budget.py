from __future__ import annotations

from rag_bench.adaptive_budget import AdaptiveBudgetConfig, select_adaptive_budget_action
from rag_bench.adaptive_features import extract_adaptive_budget_features
from rag_bench.context_budget import ContextBudget, ContextItem, apply_context_budget
from rag_bench.types import RetrievalHit


def test_adaptive_selector_uses_score_density_small_budget_for_confident_retrieval() -> None:
    features = extract_adaptive_budget_features(
        "alpha",
        [
            ContextItem("doc-1", "short signal", score=10.0),
            ContextItem("doc-2", "weak", score=1.0),
        ],
    )

    action = select_adaptive_budget_action(features)

    assert action.policy == "score-density"
    assert action.context_budget_chars == 1000
    assert action.reason == "high-confidence-retrieval"


def test_balanced_profile_uses_score_density_for_normalized_high_confidence() -> None:
    features = extract_adaptive_budget_features(
        "alpha",
        [
            ContextItem("doc-1", "short signal", score=10.0),
            ContextItem("doc-2", "weak", score=1.0),
            ContextItem("doc-3", "weaker", score=0.5),
        ],
    )

    action = select_adaptive_budget_action(features, config=AdaptiveBudgetConfig(profile="balanced"))

    assert action.policy == "score-density"
    assert action.context_budget_chars == 1000
    assert action.reason == "high-confidence-retrieval"


def test_adaptive_selector_uses_evidence_aware_large_budget_for_flat_scores() -> None:
    features = extract_adaptive_budget_features(
        "alpha beta",
        [
            ContextItem("doc-1", "a", score=1.0),
            ContextItem("doc-2", "b", score=0.98),
        ],
    )

    action = select_adaptive_budget_action(features)

    assert action.policy == "evidence-aware"
    assert action.context_budget_chars == 4000
    assert action.reason == "flat-retrieval-scores"


def test_balanced_profile_uses_large_budget_for_flat_scores() -> None:
    features = extract_adaptive_budget_features(
        "alpha beta",
        [
            ContextItem("doc-1", "a", score=1.0),
            ContextItem("doc-2", "b", score=0.99),
            ContextItem("doc-3", "c", score=0.98),
        ],
    )

    action = select_adaptive_budget_action(features, config=AdaptiveBudgetConfig(profile="balanced"))

    assert action.policy == "evidence-aware"
    assert action.context_budget_chars == 4000
    assert action.reason == "flat-retrieval-scores"


def test_aggressive_profile_uses_smaller_budget_than_conservative_when_moderate() -> None:
    features = extract_adaptive_budget_features(
        "alpha",
        [
            ContextItem("doc-1", "a", score=10.0),
            ContextItem("doc-2", "b", score=8.5),
            ContextItem("doc-3", "c", score=8.0),
        ],
    )

    conservative = select_adaptive_budget_action(features)
    aggressive = select_adaptive_budget_action(features, config=AdaptiveBudgetConfig(profile="aggressive"))

    assert conservative.context_budget_chars >= aggressive.context_budget_chars
    assert aggressive.context_budget_chars == 1000


def test_adaptive_selector_uses_per_doc_budget_for_long_document_dominance() -> None:
    features = extract_adaptive_budget_features(
        "alpha",
        [
            ContextItem("long", "A" * 3000, score=1.0),
            ContextItem("short", "B" * 10, score=0.9),
            ContextItem("tiny", "C" * 10, score=0.8),
        ],
    )

    action = select_adaptive_budget_action(features)

    assert action.policy == "per-doc-budget"
    assert action.context_budget_chars == 4000
    assert action.per_doc_budget_chars == 1000
    assert action.reason == "long-document-dominance"


def test_adaptive_context_budget_records_selected_action_metadata() -> None:
    hits = [
        RetrievalHit("long-low", 1.0, 1, "", "low " * 200),
        RetrievalHit("short-high", 10.0, 2, "", "high signal"),
    ]

    budgeted = apply_context_budget(
        hits,
        ContextBudget(
            policy="adaptive-heuristic",
            max_chars=2000,
            query="high signal",
            adaptive_small_budget=80,
            adaptive_medium_budget=120,
            adaptive_large_budget=200,
        ),
    )

    adaptive = budgeted.metadata["adaptive_budget"]
    assert budgeted.policy_name == "score-density"
    assert adaptive["requested_policy"] == "adaptive-heuristic"
    assert adaptive["profile"] == "conservative"
    assert adaptive["calibration_version"] == "phase1c2-v1"
    assert adaptive["selected_policy"] == "score-density"
    assert adaptive["selected_context_budget_chars"] == 80
    assert adaptive["reason"] == "high-confidence-retrieval"
    assert adaptive["features"]["num_candidates"] == 2
    assert "normalized_score_gap" in adaptive["features"]
    assert "normalized_score_entropy" in adaptive["features"]
    assert "score_confidence" in adaptive["features"]
    assert "[short-high]" in budgeted.text


def test_adaptive_context_budget_records_balanced_profile() -> None:
    hits = [
        RetrievalHit("doc-1", 10.0, 1, "", "high signal"),
        RetrievalHit("doc-2", 1.0, 2, "", "weak"),
    ]

    budgeted = apply_context_budget(
        hits,
        ContextBudget(policy="adaptive-heuristic", query="high signal", adaptive_profile="balanced"),
    )

    adaptive = budgeted.metadata["adaptive_budget"]
    assert adaptive["profile"] == "balanced"
    assert adaptive["selected_policy"] == "score-density"
