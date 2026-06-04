from __future__ import annotations

import json
from pathlib import Path

from rag_bench.rlaif_policy import (
    BestAverageActionPolicy,
    CheapestActionPolicy,
    FixedActionPolicy,
    OracleLoggedPolicy,
    RlaifEvalConfig,
    RlaifTrainConfig,
    evaluate_offline_selector_policies,
    train_offline_selector_policies,
)
from rag_bench.rlaif_schema import stable_record_id


def test_train_writes_offline_policy_baselines(tmp_path: Path) -> None:
    rewards_path = tmp_path / "rlaif_rewards.jsonl"
    preferences_path = tmp_path / "rlaif_preferences.jsonl"
    policy_path = tmp_path / "rlaif_policy.json"
    _write_jsonl(
        rewards_path,
        [
            _reward("q1", "bm25", "legacy", 0.60, 0.70, token=0.60),
            _reward("q1", "bm25", "evidence-aware", 0.75, 0.82, token=0.30),
            _reward("q2", "bm25", "legacy", 0.55, 0.66, token=0.62),
            _reward("q2", "bm25", "evidence-aware", 0.80, 0.86, token=0.28),
        ],
    )
    _write_jsonl(preferences_path, [{"preference_id": "pref-1"}])

    summary = train_offline_selector_policies(
        RlaifTrainConfig(
            rewards_path=rewards_path,
            preferences_path=preferences_path,
            output_path=policy_path,
        )
    )

    policy = json.loads(policy_path.read_text(encoding="utf-8"))
    assert summary["policy_count"] == 4
    assert summary["reward_count"] == 4
    assert summary["preference_count"] == 1
    assert policy["runtime_default_replacement"] is False
    assert set(policy["policies"]) == {"fixed", "cheapest", "best_average", "oracle_logged"}
    assert policy["policies"]["fixed"]["signature"]["context_policy"] == "evidence-aware"
    assert policy["policies"]["best_average"]["signatures"][0]["signature"]["context_policy"] == "evidence-aware"


def test_public_policy_names_match_artifact_keys() -> None:
    assert FixedActionPolicy.policy_type == "fixed"
    assert CheapestActionPolicy.policy_type == "cheapest"
    assert BestAverageActionPolicy.policy_type == "best_average"
    assert OracleLoggedPolicy.policy_type == "oracle_logged"


def test_eval_reports_oracle_gap_and_policy_costs(tmp_path: Path) -> None:
    rewards_path = tmp_path / "rlaif_rewards.jsonl"
    policy_path = tmp_path / "rlaif_policy.json"
    out_md = tmp_path / "rlaif_eval_summary.md"
    _write_jsonl(
        rewards_path,
            [
                _reward("q1", "bm25", "legacy", 0.55, 0.64, token=0.65, latency=0.40, kv=0.55),
                _reward("q1", "bm25", "evidence-aware", 0.78, 0.84, token=0.32, latency=0.22, kv=0.25),
                _reward("q2", "bm25", "legacy", 0.52, 0.60, token=0.70, latency=0.45, kv=0.60),
                _reward("q2", "bm25", "evidence-aware", 0.82, 0.88, token=0.30, latency=0.21, kv=0.23),
                _reward("q3", "bm25", "legacy", 0.56, 0.65, token=0.60, latency=0.38, kv=0.52),
            ],
        )
    train_offline_selector_policies(RlaifTrainConfig(rewards_path=rewards_path, output_path=policy_path))

    summary = evaluate_offline_selector_policies(
        RlaifEvalConfig(rewards_path=rewards_path, policy_path=policy_path, out_md=out_md)
    )

    oracle = summary["policy_metrics"]["oracle_logged"]
    best_average = summary["policy_metrics"]["best_average"]
    cheapest = summary["policy_metrics"]["cheapest"]
    assert oracle["oracle_gap"] == 0.0
    assert best_average["mean_reward"] == oracle["mean_reward"]
    assert cheapest["mean_token_cost"] < summary["policy_metrics"]["fixed"]["mean_token_cost"]
    assert best_average["coverage"] == 1.0
    assert out_md.read_text(encoding="utf-8").startswith("# RLAIF Offline Selector Evaluation")


def test_eval_keeps_missing_reward_separate_from_zero(tmp_path: Path) -> None:
    rewards_path = tmp_path / "rlaif_rewards.jsonl"
    policy_path = tmp_path / "rlaif_policy.json"
    _write_jsonl(
        rewards_path,
        [
            _reward("q1", "bm25", "legacy", 0.70, 0.80, token=0.50),
            _reward("q1", "bm25", "evidence-aware", None, None, token=0.10),
            _reward("q2", "bm25", "legacy", 0.65, 0.75, token=0.48),
            _reward("q2", "bm25", "evidence-aware", 0.72, 0.82, token=0.12),
        ],
    )
    train_offline_selector_policies(RlaifTrainConfig(rewards_path=rewards_path, output_path=policy_path))

    summary = evaluate_offline_selector_policies(RlaifEvalConfig(rewards_path=rewards_path, policy_path=policy_path))

    cheapest = summary["policy_metrics"]["cheapest"]
    assert cheapest["selected_count"] == 2
    assert cheapest["missing_reward_selected_count"] == 1
    assert cheapest["selection_coverage"] == 1.0
    assert cheapest["coverage"] == 0.5
    assert cheapest["mean_reward"] == 0.72


def test_oracle_gap_is_paired_on_selected_query_groups(tmp_path: Path) -> None:
    rewards_path = tmp_path / "rlaif_rewards.jsonl"
    policy_path = tmp_path / "rlaif_policy.json"
    rows = [
        _reward("q1", "bm25", "legacy", 0.10, 0.20, token=0.60),
        _reward("q1", "bm25", "evidence-aware", 0.90, 0.95, token=0.30),
        _reward("q2", "bm25", "evidence-aware", 0.80, 0.85, token=0.32),
    ]
    _write_jsonl(rewards_path, rows)
    legacy_signature = rows[0]["metadata"]["action_signature"]  # type: ignore[index]
    legacy_signature_id = rows[0]["metadata"]["action_signature_id"]  # type: ignore[index]
    evidence_signature = rows[1]["metadata"]["action_signature"]  # type: ignore[index]
    evidence_signature_id = rows[1]["metadata"]["action_signature_id"]  # type: ignore[index]
    policy_path.write_text(
        json.dumps(
            {
                "schema_version": "rlaif-policy-v1",
                "runtime_default_replacement": False,
                "policies": {
                    "fixed": {
                        "policy_type": "fixed",
                        "signature_id": legacy_signature_id,
                        "signature": legacy_signature,
                    },
                    "cheapest": {"policy_type": "cheapest"},
                    "best_average": {
                        "policy_type": "best_average",
                        "signatures": [
                            {"signature_id": legacy_signature_id, "signature": legacy_signature},
                            {"signature_id": evidence_signature_id, "signature": evidence_signature},
                        ],
                    },
                    "oracle_logged": {"policy_type": "oracle_logged"},
                },
            },
            ensure_ascii=False,
            sort_keys=True,
        ),
        encoding="utf-8",
    )

    summary = evaluate_offline_selector_policies(RlaifEvalConfig(rewards_path=rewards_path, policy_path=policy_path))

    fixed = summary["policy_metrics"]["fixed"]
    assert fixed["selected_count"] == 1
    assert fixed["mean_reward"] == 0.10
    assert fixed["oracle_gap"] == 0.80


def test_train_requires_scored_rewards(tmp_path: Path) -> None:
    rewards_path = tmp_path / "rlaif_rewards.jsonl"
    _write_jsonl(rewards_path, [_reward("q1", "bm25", "legacy", None, None)])

    try:
        train_offline_selector_policies(RlaifTrainConfig(rewards_path=rewards_path))
    except ValueError as exc:
        assert "At least one scored reward row is required" in str(exc)
    else:  # pragma: no cover - defensive assertion.
        raise AssertionError("expected ValueError")


def _reward(
    query_id: str,
    retriever: str,
    context_policy: str,
    reward: float | None,
    quality: float | None,
    *,
    token: float = 0.25,
    latency: float = 0.20,
    kv: float = 0.15,
) -> dict[str, object]:
    signature = {
        "retrieval_strategy": retriever,
        "fusion_strategy": "single",
        "top_k": 10,
        "context_policy": context_policy,
        "budget_chars": 4000 if context_policy != "legacy" else None,
        "adaptive_profile": None,
        "selected_context_policy": context_policy,
        "selected_budget_chars": 4000 if context_policy != "legacy" else None,
        "generator_model": "mimo_v25_pro",
    }
    signature_id = stable_record_id("rlaif-action-signature-v1", signature, length=12)
    return {
        "reward_id": f"reward-{query_id}-{retriever}-{context_policy}",
        "action_id": f"action-{query_id}-{retriever}-{context_policy}",
        "query_id": query_id,
        "reward": reward,
        "quality": quality,
        "token_cost_norm": token,
        "latency_norm": latency,
        "kv_cost_norm": kv,
        "metadata": {
            "query_group": {
                "benchmark": "scifact",
                "query_id": query_id,
                "top_k": 10,
                "generator_model": "mimo_v25_pro",
            },
            "action_signature": signature,
            "action_signature_id": signature_id,
        },
    }


def _write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    path.write_text(
        "\n".join(json.dumps(row, ensure_ascii=False, sort_keys=True) for row in rows) + "\n",
        encoding="utf-8",
    )
