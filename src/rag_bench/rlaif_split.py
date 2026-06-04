from __future__ import annotations

import json
import random
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

from rag_bench.io import write_json, write_jsonl


SPLIT_VERSION = "rlaif-query-split-v1"


@dataclass(frozen=True)
class RlaifSplitConfig:
    rewards_path: Path
    preferences_path: Path
    output_dir: Path
    train_ratio: float = 0.8
    seed: int = 42


def split_rlaif_by_query(config: RlaifSplitConfig) -> dict[str, Any]:
    if not config.rewards_path.is_file():
        raise ValueError(f"Rewards path does not exist: {config.rewards_path}")
    if not config.preferences_path.is_file():
        raise ValueError(f"Preferences path does not exist: {config.preferences_path}")
    if not 0.0 < config.train_ratio < 1.0:
        raise ValueError("--train-ratio must be greater than 0 and less than 1")

    rewards = _read_jsonl(config.rewards_path)
    preferences = _read_jsonl(config.preferences_path)
    if not rewards:
        raise ValueError("At least one reward row is required")

    query_groups = _group_rewards_by_query(rewards)
    train_query_keys, eval_query_keys = _split_query_keys(
        sorted(query_groups),
        train_ratio=config.train_ratio,
        seed=config.seed,
    )

    train_rewards = [row for key in sorted(train_query_keys) for row in query_groups[key]]
    eval_rewards = [row for key in sorted(eval_query_keys) for row in query_groups[key]]
    action_split = {
        str(row["action_id"]): "train"
        for row in train_rewards
        if row.get("action_id")
    }
    action_split.update(
        {
            str(row["action_id"]): "eval"
            for row in eval_rewards
            if row.get("action_id")
        }
    )

    train_preferences: list[dict[str, Any]] = []
    eval_preferences: list[dict[str, Any]] = []
    dropped_counts: Counter[str] = Counter()
    for preference in preferences:
        chosen_split = action_split.get(str(preference.get("chosen_action_id")))
        rejected_split = action_split.get(str(preference.get("rejected_action_id")))
        if chosen_split is None or rejected_split is None:
            dropped_counts["missing_action"] += 1
        elif chosen_split != rejected_split:
            dropped_counts["cross_split"] += 1
        elif chosen_split == "train":
            train_preferences.append(preference)
        else:
            eval_preferences.append(preference)

    config.output_dir.mkdir(parents=True, exist_ok=True)
    train_rewards_path = config.output_dir / "train_rewards.jsonl"
    eval_rewards_path = config.output_dir / "eval_rewards.jsonl"
    train_preferences_path = config.output_dir / "train_preferences.jsonl"
    eval_preferences_path = config.output_dir / "eval_preferences.jsonl"
    manifest_path = config.output_dir / "split_manifest.json"
    summary_path = config.output_dir / "split_summary.md"

    write_jsonl(train_rewards_path, train_rewards)
    write_jsonl(eval_rewards_path, eval_rewards)
    write_jsonl(train_preferences_path, train_preferences)
    write_jsonl(eval_preferences_path, eval_preferences)

    manifest = {
        "schema_version": SPLIT_VERSION,
        "source": {
            "rewards_path": str(config.rewards_path),
            "preferences_path": str(config.preferences_path),
        },
        "split_rule": "benchmark + query_id",
        "seed": config.seed,
        "train_ratio": config.train_ratio,
        "train_query_count": len(train_query_keys),
        "eval_query_count": len(eval_query_keys),
        "train_reward_rows": len(train_rewards),
        "eval_reward_rows": len(eval_rewards),
        "train_preferences": len(train_preferences),
        "eval_preferences": len(eval_preferences),
        "dropped_cross_split_preferences": dropped_counts["cross_split"],
        "dropped_missing_action_preferences": dropped_counts["missing_action"],
        "query_counts": {
            "total": len(query_groups),
            "train": len(train_query_keys),
            "eval": len(eval_query_keys),
        },
        "output_files": {
            "train_rewards": str(train_rewards_path),
            "eval_rewards": str(eval_rewards_path),
            "train_preferences": str(train_preferences_path),
            "eval_preferences": str(eval_preferences_path),
            "split_manifest": str(manifest_path),
            "split_summary": str(summary_path),
        },
        "train_queries": [_query_key_to_json(key) for key in sorted(train_query_keys)],
        "eval_queries": [_query_key_to_json(key) for key in sorted(eval_query_keys)],
    }
    write_json(manifest_path, manifest)
    summary_path.write_text(_render_split_summary(manifest), encoding="utf-8")
    return {
        "output_dir": str(config.output_dir),
        **{key: manifest[key] for key in _SUMMARY_KEYS},
    }


_SUMMARY_KEYS = (
    "seed",
    "train_ratio",
    "train_query_count",
    "eval_query_count",
    "train_reward_rows",
    "eval_reward_rows",
    "train_preferences",
    "eval_preferences",
    "dropped_cross_split_preferences",
    "dropped_missing_action_preferences",
)


def _split_query_keys(
    query_keys: list[tuple[str, str]],
    *,
    train_ratio: float,
    seed: int,
) -> tuple[set[tuple[str, str]], set[tuple[str, str]]]:
    shuffled = list(query_keys)
    random.Random(seed).shuffle(shuffled)
    if len(shuffled) <= 1:
        return set(shuffled), set()
    train_count = round(len(shuffled) * train_ratio)
    train_count = min(max(train_count, 1), len(shuffled) - 1)
    return set(shuffled[:train_count]), set(shuffled[train_count:])


def _group_rewards_by_query(rewards: Iterable[dict[str, Any]]) -> dict[tuple[str, str], list[dict[str, Any]]]:
    groups: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rewards:
        groups[_query_key(row)].append(row)
    return groups


def _query_key(row: dict[str, Any]) -> tuple[str, str]:
    metadata = row.get("metadata") if isinstance(row.get("metadata"), dict) else {}
    query_group = metadata.get("query_group") if isinstance(metadata.get("query_group"), dict) else {}
    benchmark = query_group.get("benchmark") or row.get("benchmark")
    query_id = query_group.get("query_id") or row.get("query_id")
    if benchmark in (None, ""):
        raise ValueError(f"Reward row missing benchmark in query_group metadata: {row.get('action_id')}")
    if query_id in (None, ""):
        raise ValueError(f"Reward row missing query_id: {row.get('action_id')}")
    return str(benchmark), str(query_id)


def _query_key_to_json(key: tuple[str, str]) -> dict[str, str]:
    return {"benchmark": key[0], "query_id": key[1]}


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            row = json.loads(stripped)
            if not isinstance(row, dict):
                raise ValueError(f"{path}:{line_no}: expected object row")
            rows.append(row)
    return rows


def _render_split_summary(manifest: dict[str, Any]) -> str:
    lines = [
        "# RLAIF Held-Out Query Split",
        "",
        f"- Rewards: `{manifest['source']['rewards_path']}`",
        f"- Preferences: `{manifest['source']['preferences_path']}`",
        f"- Split rule: `{manifest['split_rule']}`",
        f"- Train ratio: {manifest['train_ratio']}",
        f"- Seed: {manifest['seed']}",
        "",
        "| Metric | Value |",
        "| --- | ---: |",
    ]
    for key in _SUMMARY_KEYS:
        label = key.replace("_", " ")
        lines.append(f"| {label} | {manifest[key]} |")
    lines.extend(
        [
            "",
            "This split is held out by query id, not by random action rows. All actions from the same `benchmark + query_id` stay in the same split.",
        ]
    )
    return "\n".join(lines).rstrip() + "\n"
