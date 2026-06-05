from __future__ import annotations

import pytest

from rag_bench.retrieval_context_actions import action_from_budgetrag_row
from rag_bench.rlaif_schema import (
    RetrievalContextAction,
    RlaifAnswerFeedback,
    RlaifContextFeedback,
    RlaifPreference,
    RlaifReward,
)


def test_retrieval_context_action_id_is_deterministic_and_excludes_source_run() -> None:
    first = RetrievalContextAction(
        benchmark="scifact",
        query_id="q1",
        question="What is alpha?",
        retrieval_strategy="bm25",
        top_k=5,
        context_policy="evidence-aware",
        budget_chars=2000,
        generator_model="mimo-v2.5",
        source_run_id="run-a",
    )
    second = RetrievalContextAction(
        benchmark="scifact",
        query_id="q1",
        question="What is alpha?",
        retrieval_strategy="bm25",
        top_k=5,
        context_policy="evidence-aware",
        budget_chars=2000,
        generator_model="mimo-v2.5",
        source_run_id="run-b",
    )
    different_retriever = RetrievalContextAction(
        benchmark="scifact",
        query_id="q1",
        question="What is alpha?",
        retrieval_strategy="graph-bm25",
        top_k=5,
        context_policy="evidence-aware",
        budget_chars=2000,
        generator_model="mimo-v2.5",
        source_run_id="run-a",
    )

    assert first.action_id == second.action_id
    assert first.action_id != different_retriever.action_id
    assert first.to_dict()["action_id"] == first.action_id


def test_action_from_budgetrag_row_captures_retrieval_context_dimensions() -> None:
    row = {
        "run_id": "run-1",
        "benchmark": "scifact",
        "dataset_id": "beir/scifact/test",
        "retriever": "bm25",
        "query_id": "q1",
        "question": "What is alpha?",
        "top_k": 5,
        "experiment": {
            "run_id": "run-1",
            "context_policy": "adaptive-heuristic",
            "context_budget_chars": 4000,
            "adaptive_profile": "balanced",
            "generation_model": "mimo-v2.5",
        },
        "context_budget": {
            "requested_policy": "adaptive-heuristic",
            "requested_policy_impl": "deterministic-adaptive-heuristic",
            "policy": "evidence-aware",
            "policy_impl": "lexical-evidence-aware",
            "budget_chars": 4000,
            "selected_budget_chars": 2000,
        },
        "adaptive_budget": {
            "profile": "balanced",
            "selected_policy": "evidence-aware",
            "selected_context_budget_chars": 2000,
        },
    }

    action = action_from_budgetrag_row(row)

    assert action.retrieval_strategy == "bm25"
    assert action.context_policy == "adaptive-heuristic"
    assert action.budget_chars == 4000
    assert action.adaptive_profile == "balanced"
    assert action.selected_context_policy == "evidence-aware"
    assert action.selected_budget_chars == 2000
    assert action.generator_model == "mimo-v2.5"
    assert action.metadata["dataset_id"] == "beir/scifact/test"


def test_retrieval_context_action_allows_full_context_without_budget() -> None:
    action = RetrievalContextAction(
        benchmark="scifact",
        query_id="q1",
        question="What is alpha?",
        retrieval_strategy="bm25",
        top_k=5,
        context_policy="legacy",
        budget_chars=None,
        generator_model="mimo-v2.5",
    )

    assert action.budget_chars is None
    assert action.identity_payload()["budget_chars"] is None

    with pytest.raises(ValueError, match="budget_chars"):
        RetrievalContextAction(
            benchmark="scifact",
            query_id="q1",
            question="What is alpha?",
            retrieval_strategy="bm25",
            top_k=5,
            context_policy="legacy",
            budget_chars=0,
            generator_model="mimo-v2.5",
        )


def test_answer_feedback_keeps_missing_feedback_distinct_from_zero_score() -> None:
    feedback = RlaifAnswerFeedback(
        action_id="a1",
        query_id="q1",
        provenance="missing",
        quality_score=None,
        missing_reason="no judge labels available",
    )

    assert feedback.to_dict()["quality_score"] is None
    assert feedback.missing_reason == "no judge labels available"

    with pytest.raises(ValueError, match="answer_correctness"):
        RlaifAnswerFeedback(
            action_id="a1",
            query_id="q1",
            provenance="mimo_judge",
            answer_correctness=1.5,
        )


def test_context_feedback_validates_chunk_labels_and_scores() -> None:
    feedback = RlaifContextFeedback(
        action_id="a1",
        query_id="q1",
        provenance="mimo_judge",
        sufficient=True,
        selected_chunk_ids=["doc-1", "doc-3"],
        redundant_chunk_ids=("doc-2",),
        irrelevant_chunk_ids=("doc-5",),
        missing_evidence=False,
        minimality_score=0.8,
        evidence_support_score=0.9,
        context_quality_score=0.85,
        judge_provider="mimo",
        judge_model="mimo-v2.5",
    )

    assert feedback.selected_chunk_ids == ("doc-1", "doc-3")
    assert feedback.context_quality_score == 0.85

    with pytest.raises(ValueError, match="minimality_score"):
        RlaifContextFeedback(
            action_id="a1",
            query_id="q1",
            provenance="mimo_judge",
            minimality_score=-0.1,
        )


def test_reward_from_components_uses_quality_support_and_cost_penalties() -> None:
    reward = RlaifReward.from_components(
        action_id="a1",
        query_id="q1",
        quality=0.8,
        evidence_support=0.7,
        token_cost_norm=0.4,
        latency_norm=0.3,
        kv_cost_norm=0.2,
        provenance="mimo_judge",
    )

    assert reward.reward == pytest.approx(0.625)
    assert reward.weights.quality == 0.75

    failed = RlaifReward.from_components(
        action_id="a1",
        query_id="q1",
        quality=1.0,
        error_penalty=1.0,
        unsupported_claim_penalty=1.0,
    )

    assert failed.reward < 0


def test_preference_id_is_deterministic_and_rejects_self_pair() -> None:
    preference = RlaifPreference(
        preference_type="retrieval_context_preference",
        query_id="q1",
        chosen_action_id="a-good",
        rejected_action_id="a-bad",
        reward_gap=0.12,
        quality_gap=0.08,
        efficiency_gap=-0.04,
        reason="quality_guardrail",
    )
    same = RlaifPreference(
        preference_type="retrieval_context_preference",
        query_id="q1",
        chosen_action_id="a-good",
        rejected_action_id="a-bad",
        reward_gap=0.12,
        quality_gap=0.08,
        efficiency_gap=-0.04,
        reason="quality_guardrail",
    )

    assert preference.preference_id == same.preference_id
    assert preference.to_dict()["preference_id"] == preference.preference_id

    with pytest.raises(ValueError, match="must be different"):
        RlaifPreference(
            preference_type="context_policy_preference",
            query_id="q1",
            chosen_action_id="a1",
            rejected_action_id="a1",
            reward_gap=0.0,
            quality_gap=0.0,
            efficiency_gap=0.0,
            reason="invalid",
        )
