from __future__ import annotations

import json
from pathlib import Path

from rag_bench.rlaif_split import RlaifSplitConfig, split_rlaif_by_query


def test_split_keeps_same_query_in_one_split_and_drops_cross_preferences(tmp_path: Path) -> None:
    rewards_path = tmp_path / "rlaif_rewards.jsonl"
    preferences_path = tmp_path / "rlaif_preferences.jsonl"
    _write_jsonl(
        rewards_path,
        [
            _reward("q1", "a-q1-fast", reward=0.5),
            _reward("q1", "a-q1-good", reward=0.7),
            _reward("q2", "a-q2-fast", reward=0.4),
            _reward("q2", "a-q2-good", reward=0.8),
        ],
    )
    _write_jsonl(
        preferences_path,
        [
            _preference("pref-q1", "a-q1-good", "a-q1-fast"),
            _preference("pref-q2", "a-q2-good", "a-q2-fast"),
            _preference("pref-cross", "a-q1-good", "a-q2-good"),
        ],
    )

    summary = split_rlaif_by_query(
        RlaifSplitConfig(
            rewards_path=rewards_path,
            preferences_path=preferences_path,
            output_dir=tmp_path / "split",
            train_ratio=0.5,
            seed=7,
        )
    )

    manifest = json.loads((tmp_path / "split" / "split_manifest.json").read_text(encoding="utf-8"))
    train_rewards = _read_jsonl(tmp_path / "split" / "train_rewards.jsonl")
    eval_rewards = _read_jsonl(tmp_path / "split" / "eval_rewards.jsonl")
    train_queries = {row["query_id"] for row in train_rewards}
    eval_queries = {row["query_id"] for row in eval_rewards}
    assert train_queries.isdisjoint(eval_queries)
    assert train_queries | eval_queries == {"q1", "q2"}
    assert summary["dropped_cross_split_preferences"] == 1
    assert manifest["dropped_cross_split_preferences"] == 1
    assert len(_read_jsonl(tmp_path / "split" / "train_preferences.jsonl")) == 1
    assert len(_read_jsonl(tmp_path / "split" / "eval_preferences.jsonl")) == 1


def test_split_seed_is_deterministic(tmp_path: Path) -> None:
    rewards_path = tmp_path / "rlaif_rewards.jsonl"
    preferences_path = tmp_path / "rlaif_preferences.jsonl"
    _write_jsonl(
        rewards_path,
        [_reward(f"q{i}", f"a-q{i}", reward=0.5) for i in range(10)],
    )
    _write_jsonl(preferences_path, [])

    split_rlaif_by_query(
        RlaifSplitConfig(rewards_path, preferences_path, tmp_path / "split-a", train_ratio=0.6, seed=42)
    )
    split_rlaif_by_query(
        RlaifSplitConfig(rewards_path, preferences_path, tmp_path / "split-b", train_ratio=0.6, seed=42)
    )

    manifest_a = json.loads((tmp_path / "split-a" / "split_manifest.json").read_text(encoding="utf-8"))
    manifest_b = json.loads((tmp_path / "split-b" / "split_manifest.json").read_text(encoding="utf-8"))
    assert manifest_a["train_queries"] == manifest_b["train_queries"]
    assert manifest_a["eval_queries"] == manifest_b["eval_queries"]
    assert manifest_a["train_query_count"] == 6
    assert manifest_a["eval_query_count"] == 4


def test_split_keeps_missing_rewards_as_regular_rows(tmp_path: Path) -> None:
    rewards_path = tmp_path / "rlaif_rewards.jsonl"
    preferences_path = tmp_path / "rlaif_preferences.jsonl"
    _write_jsonl(
        rewards_path,
        [
            _reward("q1", "a-q1", reward=None),
            _reward("q2", "a-q2", reward=0.8),
            _reward("q3", "a-q3", reward=None),
        ],
    )
    _write_jsonl(preferences_path, [_preference("pref-missing", "a-q1", "does-not-exist")])

    summary = split_rlaif_by_query(
        RlaifSplitConfig(rewards_path, preferences_path, tmp_path / "split", train_ratio=0.67, seed=1)
    )

    split_rows = _read_jsonl(tmp_path / "split" / "train_rewards.jsonl") + _read_jsonl(
        tmp_path / "split" / "eval_rewards.jsonl"
    )
    assert len(split_rows) == 3
    assert sum(1 for row in split_rows if row["reward"] is None) == 2
    assert summary["dropped_missing_action_preferences"] == 1
    assert (tmp_path / "split" / "split_summary.md").read_text(encoding="utf-8").startswith(
        "# RLAIF Held-Out Query Split"
    )


def _reward(query_id: str, action_id: str, *, reward: float | None) -> dict[str, object]:
    return {
        "action_id": action_id,
        "query_id": query_id,
        "reward": reward,
        "quality": reward,
        "metadata": {
            "query_group": {
                "benchmark": "scifact",
                "query_id": query_id,
            }
        },
    }


def _preference(preference_id: str, chosen_action_id: str, rejected_action_id: str) -> dict[str, str]:
    return {
        "preference_id": preference_id,
        "preference_type": "retrieval_context_preference",
        "chosen_action_id": chosen_action_id,
        "rejected_action_id": rejected_action_id,
        "query_id": "fixture",
    }


def _write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    path.write_text(
        "\n".join(json.dumps(row, ensure_ascii=False, sort_keys=True) for row in rows) + ("\n" if rows else ""),
        encoding="utf-8",
    )


def _read_jsonl(path: Path) -> list[dict[str, object]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
