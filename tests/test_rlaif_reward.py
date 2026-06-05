from __future__ import annotations

import json
from pathlib import Path

import pytest

from rag_bench.rlaif_reward import RlaifRewardConfig, build_rlaif_rewards


def test_rlaif_reward_builds_component_rows_and_summary(tmp_path: Path) -> None:
    actions = [
        _action("a-good", context_policy="evidence-aware", total_tokens=100, latency_s=1.0, kv_mb=10.0),
        _action("a-costly", context_policy="legacy", total_tokens=200, latency_s=2.0, kv_mb=20.0),
        _action("a-missing", query_id="q2", context_policy="legacy", total_tokens=50, latency_s=0.5, kv_mb=5.0),
    ]
    feedback = [
        _feedback("a-good", quality=0.90, faithfulness=0.80, provenance="gold"),
        _feedback("a-costly", quality=0.90, faithfulness=0.80, provenance="gold"),
        {
            "action_id": "a-missing",
            "query_id": "q2",
            "provenance": "missing",
            "quality_score": None,
            "missing_reason": "no_feedback_labels",
        },
    ]
    _write_jsonl(tmp_path / "rlaif_actions.jsonl", actions)
    _write_jsonl(tmp_path / "rlaif_feedback.jsonl", feedback)

    summary = build_rlaif_rewards(
        RlaifRewardConfig(
            actions_path=tmp_path / "rlaif_actions.jsonl",
            feedback_path=tmp_path / "rlaif_feedback.jsonl",
            output_dir=tmp_path / "out",
        )
    )

    rewards = _read_jsonl(tmp_path / "out" / "rlaif_rewards.jsonl")
    summary_md = (tmp_path / "out" / "rlaif_reward_summary.md").read_text(encoding="utf-8")
    reward_by_id = {row["action_id"]: row for row in rewards}

    assert summary["reward_count"] == 3
    assert summary["scored_reward_count"] == 2
    assert summary["missing_quality_count"] == 1
    assert summary["reward_mode_counts"] == {"gold": 2, "missing_quality": 1}
    assert reward_by_id["a-good"]["reward"] > reward_by_id["a-costly"]["reward"]
    assert reward_by_id["a-good"]["reward_components"]["quality"] == 0.9
    assert reward_by_id["a-good"]["reward_components"]["token_cost_norm"] == 0.5
    assert reward_by_id["a-missing"]["reward"] is None
    assert reward_by_id["a-missing"]["quality"] is None
    assert reward_by_id["a-missing"]["reward_mode"] == "missing_quality"
    assert "RLAIF Reward Summary" in summary_md


def test_rlaif_reward_keeps_ambiguous_feedback_out_of_preferences(tmp_path: Path) -> None:
    actions = [
        _action("a-clear", context_policy="legacy", total_tokens=100),
        _action("a-ambiguous", context_policy="evidence-aware", total_tokens=80),
    ]
    feedback = [
        _feedback("a-clear", quality=0.7, provenance="ai_judge"),
        _feedback("a-ambiguous", quality=0.9, provenance="ai_judge", ambiguous=True),
    ]
    _write_jsonl(tmp_path / "rlaif_actions.jsonl", actions)
    _write_jsonl(tmp_path / "rlaif_feedback.jsonl", feedback)

    summary = build_rlaif_rewards(
        RlaifRewardConfig(
            actions_path=tmp_path / "rlaif_actions.jsonl",
            feedback_path=tmp_path / "rlaif_feedback.jsonl",
            output_dir=tmp_path / "out",
        )
    )
    rewards = _read_jsonl(tmp_path / "out" / "rlaif_rewards.jsonl")
    preferences = _read_jsonl(tmp_path / "out" / "rlaif_preferences.jsonl")

    assert {row["reward_mode"] for row in rewards} == {"ai_judge", "ambiguous_feedback"}
    assert preferences == []
    assert summary["preference_skip_reason_counts"]["ambiguous_feedback"] == 1


def test_rlaif_reward_uses_valid_answer_labels_over_ragas_feedback(tmp_path: Path) -> None:
    actions = [_action("a1")]
    feedback = [_feedback("a1", quality=0.2, provenance="ragas")]
    answer_labels = [
        {
            "action_id": "a1",
            "query_id": "q1",
            "provenance": "ai_judge",
            "quality_score": 0.9,
            "evidence_support": 0.8,
            "faithfulness": 0.7,
            "unsupported_claim_penalty": 0.1,
            "ambiguous": False,
            "invalid_json": False,
            "error": None,
        }
    ]
    _write_jsonl(tmp_path / "rlaif_actions.jsonl", actions)
    _write_jsonl(tmp_path / "rlaif_feedback.jsonl", feedback)
    _write_jsonl(tmp_path / "rlaif_answer_labels.jsonl", answer_labels)

    summary = build_rlaif_rewards(
        RlaifRewardConfig(
            actions_path=tmp_path / "rlaif_actions.jsonl",
            feedback_path=tmp_path / "rlaif_feedback.jsonl",
            answer_labels_path=tmp_path / "rlaif_answer_labels.jsonl",
            output_dir=tmp_path / "out",
        )
    )

    reward = _read_jsonl(tmp_path / "out" / "rlaif_rewards.jsonl")[0]
    assert summary["answer_label_count"] == 1
    assert summary["answer_label_merge_counts"]["used_answer_label"] == 1
    assert reward["quality"] == 0.9
    assert reward["provenance"] == "ai_judge"
    assert reward["metadata"]["fallback_feedback"]["provenance"] == "ragas"


def test_rlaif_reward_falls_back_when_answer_label_is_invalid(tmp_path: Path) -> None:
    actions = [_action("a1")]
    feedback = [_feedback("a1", quality=0.55, provenance="ragas")]
    answer_labels = [
        {
            "action_id": "a1",
            "query_id": "q1",
            "provenance": "ai_judge",
            "quality_score": None,
            "ambiguous": True,
            "invalid_json": True,
        }
    ]
    _write_jsonl(tmp_path / "rlaif_actions.jsonl", actions)
    _write_jsonl(tmp_path / "rlaif_feedback.jsonl", feedback)
    _write_jsonl(tmp_path / "rlaif_answer_labels.jsonl", answer_labels)

    summary = build_rlaif_rewards(
        RlaifRewardConfig(
            actions_path=tmp_path / "rlaif_actions.jsonl",
            feedback_path=tmp_path / "rlaif_feedback.jsonl",
            answer_labels_path=tmp_path / "rlaif_answer_labels.jsonl",
            output_dir=tmp_path / "out",
        )
    )

    reward = _read_jsonl(tmp_path / "out" / "rlaif_rewards.jsonl")[0]
    assert summary["answer_label_merge_counts"]["invalid_answer_label"] == 1
    assert summary["answer_label_merge_counts"]["fallback_to_feedback"] == 1
    assert reward["quality"] == 0.55
    assert reward["provenance"] == "ragas"
    assert reward["metadata"]["answer_label_skip_reason"] == "invalid_json"


def test_rlaif_reward_merges_clean_context_labels_as_non_default_candidate(tmp_path: Path) -> None:
    actions = [_action("a1"), _action("a2", context_policy="evidence-aware")]
    feedback = [
        _feedback("a1", quality=0.8, faithfulness=0.8, provenance="ai_judge"),
        _feedback("a2", quality=0.8, faithfulness=0.8, provenance="ai_judge"),
    ]
    context_labels = [
        _context_label(
            "a1",
            context_quality=0.2,
            evidence_support=0.1,
            sufficient=False,
            selected=["c1"],
            irrelevant=["c2", "c3"],
        ),
        _context_label(
            "a2",
            context_quality=1.0,
            evidence_support=1.0,
            sufficient=True,
            selected=["c1", "c2"],
            irrelevant=[],
        ),
    ]
    _write_jsonl(tmp_path / "rlaif_actions.jsonl", actions)
    _write_jsonl(tmp_path / "rlaif_feedback.jsonl", feedback)
    _write_jsonl(tmp_path / "rlaif_context_labels.jsonl", context_labels)

    summary = build_rlaif_rewards(
        RlaifRewardConfig(
            actions_path=tmp_path / "rlaif_actions.jsonl",
            feedback_path=tmp_path / "rlaif_feedback.jsonl",
            context_labels_path=tmp_path / "rlaif_context_labels.jsonl",
            output_dir=tmp_path / "out",
        )
    )

    rewards = {row["action_id"]: row for row in _read_jsonl(tmp_path / "out" / "rlaif_rewards.jsonl")}
    assert summary["context_label_count"] == 2
    assert summary["context_label_merge_counts"]["used_context_label"] == 2
    assert rewards["a1"]["quality"] == 0.5
    assert rewards["a1"]["evidence_support"] == 0.45
    assert rewards["a1"]["unsupported_claim_penalty"] == 0.9
    assert rewards["a1"]["metadata"]["context_label_merge"] == "used"
    assert rewards["a1"]["metadata"]["context_label"]["irrelevant_chunk_count"] == 2
    assert rewards["a2"]["reward"] > rewards["a1"]["reward"]


def test_rlaif_reward_falls_back_when_context_label_is_ambiguous(tmp_path: Path) -> None:
    actions = [_action("a1")]
    feedback = [_feedback("a1", quality=0.55, faithfulness=0.5, provenance="ragas")]
    context_labels = [
        _context_label(
            "a1",
            context_quality=0.95,
            evidence_support=0.95,
            sufficient=True,
            ambiguous=True,
        )
    ]
    _write_jsonl(tmp_path / "rlaif_actions.jsonl", actions)
    _write_jsonl(tmp_path / "rlaif_feedback.jsonl", feedback)
    _write_jsonl(tmp_path / "rlaif_context_labels.jsonl", context_labels)

    summary = build_rlaif_rewards(
        RlaifRewardConfig(
            actions_path=tmp_path / "rlaif_actions.jsonl",
            feedback_path=tmp_path / "rlaif_feedback.jsonl",
            context_labels_path=tmp_path / "rlaif_context_labels.jsonl",
            output_dir=tmp_path / "out",
        )
    )

    reward = _read_jsonl(tmp_path / "out" / "rlaif_rewards.jsonl")[0]
    assert summary["context_label_merge_counts"]["invalid_context_label"] == 1
    assert summary["context_label_merge_counts"]["fallback_to_feedback"] == 1
    assert reward["quality"] == 0.55
    assert reward["metadata"]["context_label_skip_reason"] == "ambiguous"


def test_rlaif_reward_context_merge_weights_reduce_aggressive_penalty(tmp_path: Path) -> None:
    actions = [_action("a1")]
    feedback = [_feedback("a1", quality=0.8, faithfulness=0.8, provenance="ai_judge")]
    context_labels = [
        _context_label(
            "a1",
            context_quality=0.2,
            evidence_support=0.1,
            sufficient=False,
            selected=["c1"],
        )
    ]
    _write_jsonl(tmp_path / "rlaif_actions.jsonl", actions)
    _write_jsonl(tmp_path / "rlaif_feedback.jsonl", feedback)
    _write_jsonl(tmp_path / "rlaif_context_labels.jsonl", context_labels)

    summary = build_rlaif_rewards(
        RlaifRewardConfig(
            actions_path=tmp_path / "rlaif_actions.jsonl",
            feedback_path=tmp_path / "rlaif_feedback.jsonl",
            context_labels_path=tmp_path / "rlaif_context_labels.jsonl",
            context_quality_blend_weight=0.25,
            context_support_blend_weight=0.25,
            context_insufficient_penalty_weight=0.25,
            output_dir=tmp_path / "out",
        )
    )

    reward = _read_jsonl(tmp_path / "out" / "rlaif_rewards.jsonl")[0]
    assert summary["context_quality_blend_weight"] == 0.25
    assert summary["context_support_blend_weight"] == 0.25
    assert summary["context_insufficient_penalty_weight"] == 0.25
    assert reward["quality"] == pytest.approx(0.65)
    assert reward["evidence_support"] == pytest.approx(0.625)
    assert reward["unsupported_claim_penalty"] == pytest.approx(0.225)
    assert reward["metadata"]["context_label"]["insufficient_penalty_weight"] == 0.25


def _action(
    action_id: str,
    *,
    query_id: str = "q1",
    benchmark: str = "scifact",
    retriever: str = "bm25",
    context_policy: str = "legacy",
    budget_chars: int | None = 2000,
    top_k: int = 5,
    generator_model: str = "mimo-v2.5-pro",
    total_tokens: int = 100,
    latency_s: float = 1.0,
    kv_mb: float = 10.0,
) -> dict:
    return {
        "action_id": action_id,
        "benchmark": benchmark,
        "query_id": query_id,
        "question": "What is alpha?",
        "retrieval_strategy": retriever,
        "fusion_strategy": None,
        "top_k": top_k,
        "context_policy": context_policy,
        "budget_chars": budget_chars,
        "adaptive_profile": None,
        "selected_context_policy": context_policy,
        "selected_budget_chars": budget_chars,
        "generator_model": generator_model,
        "token_usage": {"total_tokens": total_tokens},
        "latency": {"total_latency_s": latency_s},
        "kv_estimate": {"after_mb": kv_mb},
        "generation": {"error": None},
    }


def _feedback(
    action_id: str,
    *,
    query_id: str = "q1",
    quality: float,
    faithfulness: float | None = None,
    provenance: str = "gold",
    ambiguous: bool = False,
) -> dict:
    return {
        "action_id": action_id,
        "query_id": query_id,
        "provenance": provenance,
        "quality_score": quality,
        "faithfulness": faithfulness,
        "ambiguous": ambiguous,
    }


def _context_label(
    action_id: str,
    *,
    query_id: str = "q1",
    context_quality: float,
    evidence_support: float,
    sufficient: bool,
    ambiguous: bool = False,
    selected: list[str] | None = None,
    redundant: list[str] | None = None,
    irrelevant: list[str] | None = None,
) -> dict:
    return {
        "action_id": action_id,
        "query_id": query_id,
        "judge_provider": "mimo",
        "judge_model": "mimo-v2.5-pro",
        "context_quality_score": context_quality,
        "evidence_support_score": evidence_support,
        "minimality_score": 0.8,
        "sufficient": sufficient,
        "missing_evidence": False,
        "selected_chunk_ids": selected or [],
        "redundant_chunk_ids": redundant or [],
        "irrelevant_chunk_ids": irrelevant or [],
        "ambiguous": ambiguous,
        "invalid_json": False,
        "error": None,
    }


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def _read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
