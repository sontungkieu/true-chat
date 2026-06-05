from __future__ import annotations

import json
from pathlib import Path

from rag_bench.groq_client import GenerationResult
from rag_bench.rlaif_label_pairs import RlaifPairLabelConfig, label_rlaif_pairs


class FakeJudgeClient:
    def __init__(self, responses: list[GenerationResult]) -> None:
        self.responses = responses
        self.calls: list[list[dict[str, str]]] = []

    def generate(
        self,
        messages: list[dict[str, str]],
        *,
        model: str | None = None,
        temperature: float = 0.0,
        max_completion_tokens: int = 512,
    ) -> GenerationResult:
        self.calls.append(messages)
        if not self.responses:
            raise AssertionError("No fake response configured")
        return self.responses.pop(0)


def test_label_pairs_dry_run_writes_ambiguous_placeholder(tmp_path: Path) -> None:
    _write_fixture(tmp_path)
    output_path = tmp_path / "pair_labels.jsonl"

    summary = label_rlaif_pairs(
        RlaifPairLabelConfig(
            actions_path=tmp_path / "rlaif_actions.jsonl",
            rewards_path=tmp_path / "rlaif_rewards.jsonl",
            preferences_path=tmp_path / "rlaif_preferences.jsonl",
            output_path=output_path,
            dry_run=True,
            limit=1,
        )
    )

    label = _read_jsonl(output_path)[0]
    assert summary["processed_count"] == 1
    assert summary["ambiguous_count"] == 1
    assert label["schema_version"] == "rlaif-pairwise-label-v1"
    assert label["preference_id"] == "pref-1"
    assert label["action_a_id"] == "a-good"
    assert label["action_b_id"] == "a-cheap"
    assert label["chosen"] is None
    assert label["confidence"] is None
    assert label["metadata"]["dry_run"] is True


def test_label_pairs_retries_invalid_json_and_parses_judge_choice(tmp_path: Path) -> None:
    _write_fixture(tmp_path)
    output_path = tmp_path / "pair_labels.jsonl"
    client = FakeJudgeClient(
        [
            _generation("not json"),
            _generation(
                json.dumps(
                    {
                        "chosen": "A",
                        "tie": False,
                        "ambiguous": False,
                        "answer_quality_winner": "A",
                        "evidence_support_winner": "A",
                        "efficiency_winner": "B",
                        "quality_regret": False,
                        "unsupported_claim_risk": "B",
                        "confidence": 0.82,
                        "short_rationale": "A is better supported; B is cheaper.",
                    }
                )
            ),
        ]
    )

    summary = label_rlaif_pairs(
        RlaifPairLabelConfig(
            actions_path=tmp_path / "rlaif_actions.jsonl",
            rewards_path=tmp_path / "rlaif_rewards.jsonl",
            preferences_path=tmp_path / "rlaif_preferences.jsonl",
            output_path=output_path,
            client=client,
            json_retries=1,
        )
    )

    label = _read_jsonl(output_path)[0]
    assert summary["invalid_json_count"] == 0
    assert len(client.calls) == 2
    assert label["chosen"] == "A"
    assert label["chosen_action_id"] == "a-good"
    assert label["rejected_action_id"] == "a-cheap"
    assert label["answer_quality_winner"] == "A"
    assert label["efficiency_winner"] == "B"
    assert label["unsupported_claim_risk"] == "b"
    assert label["confidence"] == 0.82
    assert label["ambiguous"] is False
    assert label["metadata"]["json_retry_count"] == 1


def test_label_pairs_missing_context_is_ambiguous_not_zero(tmp_path: Path) -> None:
    actions, rewards, preferences = _fixture_rows()
    actions[0]["retrieved"] = []
    _write_jsonl(tmp_path / "rlaif_actions.jsonl", actions)
    _write_jsonl(tmp_path / "rlaif_rewards.jsonl", rewards)
    _write_jsonl(tmp_path / "rlaif_preferences.jsonl", preferences)
    output_path = tmp_path / "pair_labels.jsonl"

    label_rlaif_pairs(
        RlaifPairLabelConfig(
            actions_path=tmp_path / "rlaif_actions.jsonl",
            rewards_path=tmp_path / "rlaif_rewards.jsonl",
            preferences_path=tmp_path / "rlaif_preferences.jsonl",
            output_path=output_path,
            dry_run=True,
        )
    )

    label = _read_jsonl(output_path)[0]
    assert label["provenance"] == "missing"
    assert label["missing_reason"] == "missing_context"
    assert label["ambiguous"] is True
    assert label["confidence"] is None


def test_label_pairs_resume_skips_completed_preference_ids(tmp_path: Path) -> None:
    actions, rewards, preferences = _fixture_rows()
    preferences.append({**preferences[0], "preference_id": "pref-2"})
    _write_jsonl(tmp_path / "rlaif_actions.jsonl", actions)
    _write_jsonl(tmp_path / "rlaif_rewards.jsonl", rewards)
    _write_jsonl(tmp_path / "rlaif_preferences.jsonl", preferences)
    output_path = tmp_path / "pair_labels.jsonl"
    _write_jsonl(output_path, [{"preference_id": "pref-1", "schema_version": "rlaif-pairwise-label-v1"}])

    summary = label_rlaif_pairs(
        RlaifPairLabelConfig(
            actions_path=tmp_path / "rlaif_actions.jsonl",
            rewards_path=tmp_path / "rlaif_rewards.jsonl",
            preferences_path=tmp_path / "rlaif_preferences.jsonl",
            output_path=output_path,
            dry_run=True,
            resume=True,
        )
    )

    labels = _read_jsonl(output_path)
    assert summary["processed_count"] == 1
    assert summary["skipped_resume_count"] == 1
    assert [row["preference_id"] for row in labels] == ["pref-1", "pref-2"]


def test_label_pairs_stops_after_max_errors(tmp_path: Path) -> None:
    actions, rewards, preferences = _fixture_rows()
    preferences.append({**preferences[0], "preference_id": "pref-2"})
    _write_jsonl(tmp_path / "rlaif_actions.jsonl", actions)
    _write_jsonl(tmp_path / "rlaif_rewards.jsonl", rewards)
    _write_jsonl(tmp_path / "rlaif_preferences.jsonl", preferences)
    output_path = tmp_path / "pair_labels.jsonl"
    client = FakeJudgeClient([_generation("", error="judge unavailable"), _generation("{}")])

    summary = label_rlaif_pairs(
        RlaifPairLabelConfig(
            actions_path=tmp_path / "rlaif_actions.jsonl",
            rewards_path=tmp_path / "rlaif_rewards.jsonl",
            preferences_path=tmp_path / "rlaif_preferences.jsonl",
            output_path=output_path,
            client=client,
            max_errors=1,
        )
    )

    labels = _read_jsonl(output_path)
    assert summary["stopped_early"] is True
    assert summary["stop_reason"] == "max_errors"
    assert summary["processed_count"] == 1
    assert labels[0]["error"] == "judge unavailable"


def _write_fixture(tmp_path: Path) -> None:
    actions, rewards, preferences = _fixture_rows()
    _write_jsonl(tmp_path / "rlaif_actions.jsonl", actions)
    _write_jsonl(tmp_path / "rlaif_rewards.jsonl", rewards)
    _write_jsonl(tmp_path / "rlaif_preferences.jsonl", preferences)


def _fixture_rows() -> tuple[list[dict], list[dict], list[dict]]:
    actions = [
        _action("a-good", answer="The answer is supported.", total_tokens=200),
        _action("a-cheap", answer="The answer is probably supported.", total_tokens=50),
    ]
    rewards = [
        _reward("a-good", reward=0.82, quality=0.9, token_cost=1.0),
        _reward("a-cheap", reward=0.76, quality=0.75, token_cost=0.1),
    ]
    preferences = [
        {
            "preference_id": "pref-1",
            "preference_type": "retrieval_context_preference",
            "query_id": "q1",
            "chosen_action_id": "a-good",
            "rejected_action_id": "a-cheap",
            "reward_gap": 0.06,
            "quality_gap": 0.15,
            "efficiency_gap": -0.9,
            "reason": "higher_reward",
        }
    ]
    return actions, rewards, preferences


def _action(action_id: str, *, answer: str, total_tokens: int) -> dict:
    return {
        "action_id": action_id,
        "benchmark": "scifact",
        "query_id": "q1",
        "question": "Does the context support the answer?",
        "retrieval_strategy": "bm25",
        "fusion_strategy": None,
        "top_k": 5,
        "context_policy": "legacy",
        "budget_chars": 2000,
        "adaptive_profile": None,
        "selected_context_policy": "legacy",
        "selected_budget_chars": 2000,
        "generator_model": "mimo-v2.5-pro",
        "answer": answer,
        "retrieved": [
            {
                "rank": 1,
                "doc_id": f"{action_id}-doc",
                "title": "Evidence",
                "text": "The context states that the answer is supported.",
            }
        ],
        "token_usage": {"total_tokens": total_tokens},
        "latency": {"total_latency_s": 1.0},
        "kv_estimate": {"after_mb": 10.0},
    }


def _reward(action_id: str, *, reward: float, quality: float, token_cost: float) -> dict:
    return {
        "action_id": action_id,
        "query_id": "q1",
        "reward": reward,
        "quality": quality,
        "evidence_support": quality,
        "token_cost_norm": token_cost,
        "latency_norm": 0.5,
        "kv_cost_norm": 0.5,
        "unsupported_claim_penalty": 0.0,
    }


def _generation(answer: str, *, error: str | None = None) -> GenerationResult:
    return GenerationResult(
        answer=answer,
        key_alias="fake",
        attempted_aliases=["fake"],
        latency_s=0.01,
        retry_count=0,
        error=error,
    )


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text(
        "\n".join(json.dumps(row, ensure_ascii=False, sort_keys=True) for row in rows) + "\n",
        encoding="utf-8",
    )


def _read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
