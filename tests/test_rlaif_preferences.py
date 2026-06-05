from __future__ import annotations

import json
from pathlib import Path

from rag_bench.rlaif_reward import RlaifRewardConfig, build_rlaif_rewards


def test_preferences_include_context_policy_and_retrieval_context_groups(tmp_path: Path) -> None:
    actions = [
        _action("a-bm25-legacy", retriever="bm25", context_policy="legacy", total_tokens=180),
        _action("a-bm25-evidence", retriever="bm25", context_policy="evidence-aware", total_tokens=100),
        _action("a-graph", retriever="graph-bm25", context_policy="legacy", total_tokens=120),
    ]
    feedback = [
        _feedback("a-bm25-legacy", quality=0.70),
        _feedback("a-bm25-evidence", quality=0.82),
        _feedback("a-graph", quality=0.88),
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
    preferences = _read_jsonl(tmp_path / "out" / "rlaif_preferences.jsonl")

    assert summary["preference_type_counts"]["context_policy_preference"] == 1
    assert summary["preference_type_counts"]["retrieval_context_preference"] == 3
    context_policy_preferences = [
        row for row in preferences if row["preference_type"] == "context_policy_preference"
    ]
    retrieval_context_preferences = [
        row for row in preferences if row["preference_type"] == "retrieval_context_preference"
    ]
    assert context_policy_preferences[0]["chosen_action_id"] == "a-bm25-evidence"
    assert "a-graph" in {row["chosen_action_id"] for row in retrieval_context_preferences}
    assert all(row["reason"] == "higher_reward" for row in preferences)


def test_quality_guardrail_blocks_cheaper_but_worse_preference(tmp_path: Path) -> None:
    actions = [
        _action("a-expensive-good", context_policy="legacy", total_tokens=200, latency_s=2.0, kv_mb=20.0),
        _action("a-cheap-worse", context_policy="evidence-aware", total_tokens=20, latency_s=0.2, kv_mb=2.0),
    ]
    feedback = [
        _feedback("a-expensive-good", quality=0.85),
        _feedback("a-cheap-worse", quality=0.80),
    ]
    _write_jsonl(tmp_path / "rlaif_actions.jsonl", actions)
    _write_jsonl(tmp_path / "rlaif_feedback.jsonl", feedback)

    summary = build_rlaif_rewards(
        RlaifRewardConfig(
            actions_path=tmp_path / "rlaif_actions.jsonl",
            feedback_path=tmp_path / "rlaif_feedback.jsonl",
            output_dir=tmp_path / "out",
            quality_weight=0.45,
            support_weight=0.0,
            token_weight=0.25,
            latency_weight=0.25,
            kv_weight=0.25,
            max_quality_regret=0.02,
            min_reward_delta=0.01,
        )
    )
    preferences = _read_jsonl(tmp_path / "out" / "rlaif_preferences.jsonl")
    reward_by_id = {
        row["action_id"]: row for row in _read_jsonl(tmp_path / "out" / "rlaif_rewards.jsonl")
    }

    assert reward_by_id["a-cheap-worse"]["reward"] > reward_by_id["a-expensive-good"]["reward"]
    assert preferences == []
    assert summary["preference_skip_reason_counts"]["quality_guardrail_failed"] == 2


def test_small_reward_delta_does_not_create_preferences(tmp_path: Path) -> None:
    actions = [
        _action("a-one", context_policy="legacy", total_tokens=100),
        _action("a-two", context_policy="evidence-aware", total_tokens=100),
    ]
    feedback = [
        _feedback("a-one", quality=0.80),
        _feedback("a-two", quality=0.81),
    ]
    _write_jsonl(tmp_path / "rlaif_actions.jsonl", actions)
    _write_jsonl(tmp_path / "rlaif_feedback.jsonl", feedback)

    summary = build_rlaif_rewards(
        RlaifRewardConfig(
            actions_path=tmp_path / "rlaif_actions.jsonl",
            feedback_path=tmp_path / "rlaif_feedback.jsonl",
            output_dir=tmp_path / "out",
            min_reward_delta=0.03,
        )
    )

    assert _read_jsonl(tmp_path / "out" / "rlaif_preferences.jsonl") == []
    assert summary["preference_skip_reason_counts"]["small_reward_delta"] == 2


def test_pairwise_tie_calibration_is_opt_in_and_prefers_cheaper_action(tmp_path: Path) -> None:
    actions = [
        _action("a-expensive-slightly-better", context_policy="legacy", total_tokens=200, latency_s=2.0, kv_mb=20.0),
        _action("a-cheaper-good", context_policy="evidence-aware", total_tokens=20, latency_s=0.2, kv_mb=2.0),
    ]
    feedback = [
        _feedback("a-expensive-slightly-better", quality=0.95),
        _feedback("a-cheaper-good", quality=0.90),
    ]
    _write_jsonl(tmp_path / "rlaif_actions.jsonl", actions)
    _write_jsonl(tmp_path / "rlaif_feedback.jsonl", feedback)

    default_summary = build_rlaif_rewards(
        RlaifRewardConfig(
            actions_path=tmp_path / "rlaif_actions.jsonl",
            feedback_path=tmp_path / "rlaif_feedback.jsonl",
            output_dir=tmp_path / "default",
            quality_weight=0.95,
            support_weight=0.0,
            token_weight=0.01,
            latency_weight=0.01,
            kv_weight=0.01,
            min_reward_delta=0.01,
        )
    )
    calibrated_summary = build_rlaif_rewards(
        RlaifRewardConfig(
            actions_path=tmp_path / "rlaif_actions.jsonl",
            feedback_path=tmp_path / "rlaif_feedback.jsonl",
            output_dir=tmp_path / "calibrated",
            quality_weight=0.95,
            support_weight=0.0,
            token_weight=0.01,
            latency_weight=0.01,
            kv_weight=0.01,
            min_reward_delta=0.01,
            reward_calibration="pairwise_tie_v1",
            quality_tie_threshold=0.10,
            support_tie_threshold=0.10,
            tie_break_by_efficiency=True,
        )
    )

    default_preferences = _read_jsonl(tmp_path / "default" / "rlaif_preferences.jsonl")
    calibrated_preferences = _read_jsonl(tmp_path / "calibrated" / "rlaif_preferences.jsonl")

    assert default_summary["reward_calibration"] == "none"
    assert all(row["chosen_action_id"] == "a-expensive-slightly-better" for row in default_preferences)
    assert calibrated_summary["reward_calibration"] == "pairwise_tie_v1"
    assert calibrated_summary["preference_reason_counts"]["pairwise_tie_v1_efficiency"] == 2
    assert all(row["chosen_action_id"] == "a-cheaper-good" for row in calibrated_preferences)
    assert all(row["reason"] == "pairwise_tie_v1_efficiency" for row in calibrated_preferences)
    assert calibrated_preferences[0]["reward_gap"] < 0
    assert calibrated_preferences[0]["metadata"]["quality_support_tie"] is True


def _action(
    action_id: str,
    *,
    retriever: str = "bm25",
    context_policy: str,
    total_tokens: int,
    latency_s: float = 1.0,
    kv_mb: float = 10.0,
) -> dict:
    return {
        "action_id": action_id,
        "benchmark": "scifact",
        "query_id": "q1",
        "question": "What is alpha?",
        "retrieval_strategy": retriever,
        "fusion_strategy": None,
        "top_k": 5,
        "context_policy": context_policy,
        "budget_chars": 2000,
        "adaptive_profile": None,
        "selected_context_policy": context_policy,
        "selected_budget_chars": 2000,
        "generator_model": "mimo-v2.5",
        "token_usage": {"total_tokens": total_tokens},
        "latency": {"total_latency_s": latency_s},
        "kv_estimate": {"after_mb": kv_mb},
        "generation": {"error": None},
    }


def _feedback(action_id: str, *, quality: float) -> dict:
    return {
        "action_id": action_id,
        "query_id": "q1",
        "provenance": "gold",
        "quality_score": quality,
        "faithfulness": quality,
        "ambiguous": False,
    }


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def _read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
