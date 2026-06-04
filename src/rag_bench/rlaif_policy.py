from __future__ import annotations

import json
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from statistics import mean
from typing import Any, Iterable

from rag_bench.io import write_json
from rag_bench.rlaif_schema import stable_record_id


POLICY_VERSION = "rlaif-policy-v1"
POLICY_NAMES = ("fixed", "cheapest", "best_average", "oracle_logged")


@dataclass(frozen=True)
class RlaifTrainConfig:
    rewards_path: Path
    preferences_path: Path | None = None
    output_path: Path | None = None


@dataclass(frozen=True)
class RlaifEvalConfig:
    rewards_path: Path
    policy_path: Path
    out_md: Path | None = None
    split_manifest_path: Path | None = None


class FixedActionPolicy:
    policy_type = "fixed"

    def select(self, group_rewards: list[dict[str, Any]], policy: dict[str, Any]) -> dict[str, Any] | None:
        signature_id = policy["policies"]["fixed"].get("signature_id")
        candidates = [row for row in group_rewards if _signature_id(row) == signature_id]
        return _first_by_action_id(candidates)


class CheapestActionPolicy:
    policy_type = "cheapest"

    def select(self, group_rewards: list[dict[str, Any]], policy: dict[str, Any]) -> dict[str, Any] | None:
        if not group_rewards:
            return None
        return min(group_rewards, key=lambda row: (_cost_sum(row), str(row.get("action_id"))))


class BestAverageActionPolicy:
    policy_type = "best_average"

    def select(self, group_rewards: list[dict[str, Any]], policy: dict[str, Any]) -> dict[str, Any] | None:
        rank = {
            row["signature_id"]: index
            for index, row in enumerate(policy["policies"]["best_average"].get("signatures", []))
        }
        candidates = [row for row in group_rewards if _signature_id(row) in rank]
        if not candidates:
            return None
        return min(candidates, key=lambda row: (rank[_signature_id(row)], str(row.get("action_id"))))


class OracleLoggedPolicy:
    policy_type = "oracle_logged"

    def select(self, group_rewards: list[dict[str, Any]], policy: dict[str, Any]) -> dict[str, Any] | None:
        scored = [row for row in group_rewards if _is_scored(row)]
        if not scored:
            return None
        return max(scored, key=lambda row: (float(row["reward"]), str(row.get("action_id"))))


POLICY_IMPLEMENTATIONS = {
    FixedActionPolicy.policy_type: FixedActionPolicy(),
    CheapestActionPolicy.policy_type: CheapestActionPolicy(),
    BestAverageActionPolicy.policy_type: BestAverageActionPolicy(),
    OracleLoggedPolicy.policy_type: OracleLoggedPolicy(),
}


def train_offline_selector_policies(config: RlaifTrainConfig) -> dict[str, Any]:
    if not config.rewards_path.is_file():
        raise ValueError(f"Rewards path does not exist: {config.rewards_path}")
    if config.preferences_path is not None and not config.preferences_path.is_file():
        raise ValueError(f"Preferences path does not exist: {config.preferences_path}")

    rewards = _read_jsonl(config.rewards_path)
    preferences = _read_jsonl(config.preferences_path) if config.preferences_path is not None else []
    scored_rewards = [row for row in rewards if _is_scored(row)]
    if not rewards:
        raise ValueError("At least one reward row is required")
    if not scored_rewards:
        raise ValueError("At least one scored reward row is required")

    signature_stats = _signature_stats(scored_rewards)
    fixed_signature = _fixed_signature(signature_stats)
    best_average_signatures = sorted(
        signature_stats,
        key=lambda row: (-row["mean_reward"], -row["count"], row["signature_id"]),
    )
    policy = {
        "schema_version": POLICY_VERSION,
        "runtime_default_replacement": False,
        "source": {
            "rewards_path": str(config.rewards_path),
            "preferences_path": str(config.preferences_path) if config.preferences_path is not None else None,
        },
        "train_summary": {
            "reward_count": len(rewards),
            "scored_reward_count": len(scored_rewards),
            "preference_count": len(preferences),
            "query_group_count": len(_group_rewards(rewards)),
            "signature_count": len(signature_stats),
        },
        "policies": {
            "fixed": {
                "policy_type": "fixed",
                "signature_id": fixed_signature["signature_id"],
                "signature": fixed_signature["signature"],
                "selection_rule": "choose this signature when available in the query group",
            },
            "cheapest": {
                "policy_type": "cheapest",
                "selection_rule": "choose the lowest token+latency+kv normalized cost in the query group",
            },
            "best_average": {
                "policy_type": "best_average",
                "selection_rule": "choose the available signature with highest training mean reward",
                "signatures": best_average_signatures,
            },
            "oracle_logged": {
                "policy_type": "oracle_logged",
                "selection_rule": "offline upper bound; choose the logged action with highest observed reward",
            },
        },
    }
    output_path = config.output_path or (config.rewards_path.parent / "rlaif_policy.json")
    write_json(output_path, policy)
    return {
        "output_path": str(output_path),
        "policy_count": len(policy["policies"]),
        **policy["train_summary"],
        "runtime_default_replacement": False,
    }


def evaluate_offline_selector_policies(config: RlaifEvalConfig) -> dict[str, Any]:
    if not config.rewards_path.is_file():
        raise ValueError(f"Rewards path does not exist: {config.rewards_path}")
    if not config.policy_path.is_file():
        raise ValueError(f"Policy path does not exist: {config.policy_path}")

    rewards = _read_jsonl(config.rewards_path)
    policy = json.loads(config.policy_path.read_text(encoding="utf-8"))
    if policy.get("schema_version") != POLICY_VERSION:
        raise ValueError(f"Unsupported policy schema_version: {policy.get('schema_version')}")
    if config.split_manifest_path is not None and not config.split_manifest_path.is_file():
        raise ValueError(f"Split manifest path does not exist: {config.split_manifest_path}")
    split_manifest = (
        json.loads(config.split_manifest_path.read_text(encoding="utf-8"))
        if config.split_manifest_path is not None
        else None
    )

    groups = _group_rewards(rewards)
    oracle_selected = _select_by_group("oracle_logged", groups, policy)
    policy_metrics = {}
    for policy_name in POLICY_NAMES:
        selected = _select_by_group(policy_name, groups, policy)
        metrics = _policy_metrics(selected, group_count=len(groups))
        metrics["oracle_gap"] = _paired_oracle_gap(oracle_selected, selected)
        policy_metrics[policy_name] = metrics

    summary = {
        "rewards_path": str(config.rewards_path),
        "policy_path": str(config.policy_path),
        "query_group_count": len(groups),
        "policy_metrics": policy_metrics,
        "runtime_default_replacement": False,
        "held_out_query_eval": split_manifest is not None,
        "split_manifest_path": str(config.split_manifest_path) if config.split_manifest_path is not None else None,
        "split_manifest": _split_manifest_summary(split_manifest),
    }
    if config.out_md is not None:
        config.out_md.parent.mkdir(parents=True, exist_ok=True)
        config.out_md.write_text(_render_eval_markdown(summary), encoding="utf-8")
    return summary


def _select_by_group(
    policy_name: str,
    groups: dict[tuple[Any, ...], list[dict[str, Any]]],
    policy: dict[str, Any],
) -> dict[tuple[Any, ...], dict[str, Any] | None]:
    return {
        group_key: _select_reward(policy_name, group_rewards, policy)
        for group_key, group_rewards in groups.items()
    }


def _policy_metrics(
    selected_by_group: dict[tuple[Any, ...], dict[str, Any] | None],
    *,
    group_count: int,
) -> dict[str, Any]:
    selected_rows = [row for row in selected_by_group.values() if row is not None]
    scored_rows = [row for row in selected_rows if _is_scored(row)]
    distribution = Counter(_signature_id(row) for row in selected_rows)
    return {
        "query_group_count": group_count,
        "selected_count": len(selected_rows),
        "scored_selected_count": len(scored_rows),
        "missing_reward_selected_count": len(selected_rows) - len(scored_rows),
        "coverage": _ratio(len(scored_rows), group_count),
        "selection_coverage": _ratio(len(selected_rows), group_count),
        "mean_reward": _mean_field(scored_rows, "reward"),
        "mean_quality": _mean_field(scored_rows, "quality"),
        "mean_token_cost": _mean_field(selected_rows, "token_cost_norm"),
        "mean_latency": _mean_field(selected_rows, "latency_norm"),
        "mean_kv_cost": _mean_field(selected_rows, "kv_cost_norm"),
        "selected_action_distribution": dict(sorted(distribution.items())),
    }


def _select_reward(policy_name: str, group_rewards: list[dict[str, Any]], policy: dict[str, Any]) -> dict[str, Any] | None:
    policy_impl = POLICY_IMPLEMENTATIONS.get(policy_name)
    if policy_impl is None:
        raise ValueError(f"Unknown policy: {policy_name}")
    return policy_impl.select(group_rewards, policy)


def _signature_stats(scored_rewards: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in scored_rewards:
        groups[_signature_id(row)].append(row)
    stats = []
    for signature_id, rows in groups.items():
        stats.append(
            {
                "signature_id": signature_id,
                "signature": _signature(rows[0]),
                "count": len(rows),
                "mean_reward": _mean_field(rows, "reward"),
                "mean_quality": _mean_field(rows, "quality"),
                "mean_token_cost": _mean_field(rows, "token_cost_norm"),
                "mean_latency": _mean_field(rows, "latency_norm"),
                "mean_kv_cost": _mean_field(rows, "kv_cost_norm"),
            }
        )
    return stats


def _fixed_signature(signature_stats: list[dict[str, Any]]) -> dict[str, Any]:
    return sorted(
        signature_stats,
        key=lambda row: (-row["count"], -(row["mean_reward"] or 0.0), row["signature_id"]),
    )[0]


def _group_rewards(rewards: Iterable[dict[str, Any]]) -> dict[tuple[Any, ...], list[dict[str, Any]]]:
    groups: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in rewards:
        groups[_query_group_key(row)].append(row)
    return groups


def _query_group_key(row: dict[str, Any]) -> tuple[Any, ...]:
    query_group = _metadata_dict(row).get("query_group")
    if isinstance(query_group, dict):
        return (
            query_group.get("benchmark"),
            query_group.get("query_id") or row.get("query_id"),
            query_group.get("top_k"),
            query_group.get("generator_model"),
        )
    return (None, row.get("query_id"), None, None)


def _signature(row: dict[str, Any]) -> dict[str, Any]:
    signature = _metadata_dict(row).get("action_signature")
    if isinstance(signature, dict):
        return signature
    return {
        "action_id": row.get("action_id"),
        "fallback": "missing_action_signature",
    }


def _signature_id(row: dict[str, Any]) -> str:
    value = _metadata_dict(row).get("action_signature_id")
    if isinstance(value, str) and value.strip():
        return value
    return stable_record_id("rlaif-action-signature-v1", _signature(row), length=12)


def _metadata_dict(row: dict[str, Any]) -> dict[str, Any]:
    value = row.get("metadata")
    return value if isinstance(value, dict) else {}


def _is_scored(row: dict[str, Any]) -> bool:
    return row.get("reward") is not None and row.get("quality") is not None


def _cost_sum(row: dict[str, Any]) -> float:
    return sum(
        float(row.get(key) or 0.0)
        for key in ("token_cost_norm", "latency_norm", "kv_cost_norm")
    )


def _first_by_action_id(rows: list[dict[str, Any]]) -> dict[str, Any] | None:
    if not rows:
        return None
    return sorted(rows, key=lambda row: str(row.get("action_id")))[0]


def _mean_field(rows: Iterable[dict[str, Any]], field_name: str) -> float | None:
    values = [
        float(row[field_name])
        for row in rows
        if row.get(field_name) is not None
    ]
    if not values:
        return None
    return mean(values)


def _ratio(numerator: int, denominator: int) -> float:
    if denominator <= 0:
        return 0.0
    return numerator / denominator


def _paired_oracle_gap(
    oracle_selected: dict[tuple[Any, ...], dict[str, Any] | None],
    policy_selected: dict[tuple[Any, ...], dict[str, Any] | None],
) -> float | None:
    gaps = []
    for group_key, selected in policy_selected.items():
        oracle = oracle_selected.get(group_key)
        if selected is None or oracle is None:
            continue
        if not _is_scored(selected) or not _is_scored(oracle):
            continue
        gaps.append(float(oracle["reward"]) - float(selected["reward"]))
    if not gaps:
        return None
    return mean(gaps)


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


def _split_manifest_summary(manifest: dict[str, Any] | None) -> dict[str, Any] | None:
    if manifest is None:
        return None
    return {
        "schema_version": manifest.get("schema_version"),
        "split_rule": manifest.get("split_rule"),
        "seed": manifest.get("seed"),
        "train_ratio": manifest.get("train_ratio"),
        "train_query_count": manifest.get("train_query_count"),
        "eval_query_count": manifest.get("eval_query_count"),
        "train_reward_rows": manifest.get("train_reward_rows"),
        "eval_reward_rows": manifest.get("eval_reward_rows"),
        "dropped_cross_split_preferences": manifest.get("dropped_cross_split_preferences"),
    }


def _render_eval_markdown(summary: dict[str, Any]) -> str:
    lines = [
        "# RLAIF Offline Selector Evaluation",
        "",
        f"- Rewards: `{summary['rewards_path']}`",
        f"- Policy: `{summary['policy_path']}`",
        f"- Query groups: {summary['query_group_count']}",
        f"- Runtime default replacement: `{summary['runtime_default_replacement']}`",
        f"- Held-out query eval: `{summary['held_out_query_eval']}`",
    ]
    if summary["split_manifest_path"] is not None:
        lines.append(f"- Split manifest: `{summary['split_manifest_path']}`")
    lines.extend(
        [
            "",
            "| Policy | Coverage | Mean reward | Mean quality | Token cost | Latency | KV cost | Oracle gap | Selected | Missing reward |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for policy_name in POLICY_NAMES:
        metrics = summary["policy_metrics"][policy_name]
        lines.append(
            "| "
            + " | ".join(
                [
                    f"`{policy_name}`",
                    _fmt(metrics["coverage"]),
                    _fmt(metrics["mean_reward"]),
                    _fmt(metrics["mean_quality"]),
                    _fmt(metrics["mean_token_cost"]),
                    _fmt(metrics["mean_latency"]),
                    _fmt(metrics["mean_kv_cost"]),
                    _fmt(metrics["oracle_gap"]),
                    str(metrics["selected_count"]),
                    str(metrics["missing_reward_selected_count"]),
                ]
            )
            + " |"
        )
    lines.extend(["", "## Selected Action Distribution", ""])
    for policy_name in POLICY_NAMES:
        metrics = summary["policy_metrics"][policy_name]
        lines.append(f"### `{policy_name}`")
        distribution = metrics["selected_action_distribution"]
        if not distribution:
            lines.append("- N/A")
            continue
        for signature_id, count in sorted(distribution.items()):
            lines.append(f"- `{signature_id}`: {count}")
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def _fmt(value: Any) -> str:
    if value is None:
        return "N/A"
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)
