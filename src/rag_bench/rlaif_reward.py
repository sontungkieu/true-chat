from __future__ import annotations

import json
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from itertools import combinations
from pathlib import Path
from statistics import mean
from typing import Any, Iterable

from rag_bench.io import write_jsonl
from rag_bench.rlaif_schema import RlaifPreference, RlaifReward, RlaifRewardWeights


@dataclass(frozen=True)
class RlaifRewardConfig:
    actions_path: Path
    feedback_path: Path
    output_dir: Path | None = None
    quality_weight: float = 0.75
    support_weight: float = 0.10
    token_weight: float = 0.05
    latency_weight: float = 0.05
    kv_weight: float = 0.05
    error_weight: float = 1.0
    unsupported_weight: float = 1.0
    min_reward_delta: float = 0.03
    max_quality_regret: float = 0.02


def build_rlaif_rewards(config: RlaifRewardConfig) -> dict[str, Any]:
    weights = _weights_from_config(config)
    _validate_config(config)

    actions = _read_jsonl(config.actions_path)
    feedback_rows = _read_jsonl(config.feedback_path)
    output_dir = config.output_dir or config.actions_path.parent

    feedback_by_action_id, duplicate_feedback_count = _index_feedback(feedback_rows)
    scales = _CostScales.from_actions(actions)

    reward_rows: list[dict[str, Any]] = []
    invalid_rows: list[dict[str, Any]] = []
    for action in actions:
        try:
            reward = _reward_for_action(
                action=action,
                feedback=feedback_by_action_id.get(str(action.get("action_id"))),
                scales=scales,
                weights=weights,
            )
            reward_rows.append(reward.to_dict())
        except Exception as exc:  # noqa: BLE001 - keep coverage over partial datasets.
            invalid_rows.append(
                {
                    "action_id": action.get("action_id") if isinstance(action, dict) else None,
                    "query_id": action.get("query_id") if isinstance(action, dict) else None,
                    "error": str(exc),
                }
            )

    preference_rows, preference_skips = _build_preferences(
        actions=actions,
        rewards=reward_rows,
        min_reward_delta=config.min_reward_delta,
        max_quality_regret=config.max_quality_regret,
    )

    write_jsonl(output_dir / "rlaif_rewards.jsonl", reward_rows)
    write_jsonl(output_dir / "rlaif_preferences.jsonl", preference_rows)
    summary = _build_summary(
        output_dir=output_dir,
        action_count=len(actions),
        feedback_count=len(feedback_rows),
        duplicate_feedback_count=duplicate_feedback_count,
        reward_rows=reward_rows,
        preference_rows=preference_rows,
        preference_skips=preference_skips,
        invalid_rows=invalid_rows,
        weights=weights,
        config=config,
    )
    (output_dir / "rlaif_reward_summary.md").write_text(_render_summary_markdown(summary), encoding="utf-8")
    return summary


def _reward_for_action(
    *,
    action: dict[str, Any],
    feedback: dict[str, Any] | None,
    scales: _CostScales,
    weights: RlaifRewardWeights,
) -> RlaifReward:
    action_id = _required_text(action.get("action_id"), "action_id")
    query_id = _required_text(action.get("query_id"), "query_id")
    feedback = feedback or {
        "action_id": action_id,
        "query_id": query_id,
        "provenance": "missing",
        "missing_reason": "missing_feedback_row",
    }
    if feedback.get("query_id") not in (None, query_id):
        raise ValueError("feedback query_id does not match action query_id")

    quality = _score_or_none(feedback.get("quality_score"))
    provenance = _feedback_provenance(feedback.get("provenance"))
    reward_mode = _reward_mode(feedback)
    evidence_support = _evidence_support(feedback)
    token_cost_norm = scales.token_norm(action)
    latency_norm = scales.latency_norm(action)
    kv_cost_norm = scales.kv_norm(action)
    error_penalty = _error_penalty(action, feedback)
    unsupported_claim_penalty = _score_or_none(feedback.get("unsupported_claim_penalty")) or 0.0
    metadata = {
        "reward_mode": reward_mode,
        "scored": False,
        "feedback_provenance": provenance,
        "feedback_missing_reason": feedback.get("missing_reason"),
        "feedback_ambiguous": bool(feedback.get("ambiguous", False)),
        "raw_costs": scales.raw_costs(action),
    }

    if quality is None or bool(feedback.get("ambiguous", False)):
        reason = "ambiguous_feedback" if bool(feedback.get("ambiguous", False)) else "missing_quality"
        metadata["skip_reason"] = reason
        return RlaifReward(
            action_id=action_id,
            query_id=query_id,
            reward=None,
            quality=quality,
            evidence_support=evidence_support,
            token_cost_norm=token_cost_norm,
            latency_norm=latency_norm,
            kv_cost_norm=kv_cost_norm,
            error_penalty=error_penalty,
            unsupported_claim_penalty=unsupported_claim_penalty,
            weights=weights,
            provenance=provenance,
            reward_mode=reason,
            metadata=metadata,
        )

    metadata["scored"] = True
    return RlaifReward.from_components(
        action_id=action_id,
        query_id=query_id,
        quality=quality,
        evidence_support=evidence_support,
        token_cost_norm=token_cost_norm,
        latency_norm=latency_norm,
        kv_cost_norm=kv_cost_norm,
        error_penalty=error_penalty,
        unsupported_claim_penalty=unsupported_claim_penalty,
        weights=weights,
        provenance=provenance,
        reward_mode=reward_mode,
        metadata=metadata,
    )


def _build_preferences(
    *,
    actions: list[dict[str, Any]],
    rewards: list[dict[str, Any]],
    min_reward_delta: float,
    max_quality_regret: float,
) -> tuple[list[dict[str, Any]], Counter[str]]:
    action_by_id = {str(action.get("action_id")): action for action in actions if action.get("action_id")}
    reward_by_id = {str(reward.get("action_id")): reward for reward in rewards if reward.get("action_id")}
    eligible_rewards: list[dict[str, Any]] = []
    skip_counts: Counter[str] = Counter()
    for reward in rewards:
        reason = _ineligible_reward_reason(reward)
        if reason is None:
            eligible_rewards.append(reward)
        else:
            skip_counts[reason] += 1

    preferences: list[dict[str, Any]] = []
    seen_preference_ids: set[str] = set()
    specs = (
        ("context_policy_preference", _context_policy_group_key),
        ("retrieval_context_preference", _retrieval_context_group_key),
    )
    for preference_type, group_key_fn in specs:
        groups: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
        for reward in eligible_rewards:
            action = action_by_id.get(str(reward.get("action_id")))
            if not action:
                skip_counts["missing_action"] += 1
                continue
            groups[group_key_fn(action)].append(reward)

        for group_key, group_rewards in groups.items():
            if len(group_rewards) < 2:
                continue
            for left, right in combinations(sorted(group_rewards, key=lambda row: str(row["action_id"])), 2):
                preference, skip_reason = _preference_from_pair(
                    preference_type=preference_type,
                    group_key=group_key,
                    left=left,
                    right=right,
                    left_action=action_by_id[str(left["action_id"])],
                    right_action=action_by_id[str(right["action_id"])],
                    min_reward_delta=min_reward_delta,
                    max_quality_regret=max_quality_regret,
                )
                if skip_reason is not None:
                    skip_counts[skip_reason] += 1
                    continue
                if preference is None:
                    continue
                preference_id = str(preference["preference_id"])
                if preference_id in seen_preference_ids:
                    skip_counts["same_action"] += 1
                    continue
                seen_preference_ids.add(preference_id)
                preferences.append(preference)

    valid_preferences: list[dict[str, Any]] = []
    for preference in preferences:
        if preference["chosen_action_id"] in reward_by_id and preference["rejected_action_id"] in reward_by_id:
            valid_preferences.append(preference)
        else:
            skip_counts["missing_reward"] += 1
    return valid_preferences, skip_counts


def _preference_from_pair(
    *,
    preference_type: str,
    group_key: tuple[Any, ...],
    left: dict[str, Any],
    right: dict[str, Any],
    left_action: dict[str, Any],
    right_action: dict[str, Any],
    min_reward_delta: float,
    max_quality_regret: float,
) -> tuple[dict[str, Any] | None, str | None]:
    if left["action_id"] == right["action_id"]:
        return None, "same_action"
    left_reward = float(left["reward"])
    right_reward = float(right["reward"])
    reward_gap_abs = abs(left_reward - right_reward)
    if reward_gap_abs < min_reward_delta:
        return None, "small_reward_delta"
    chosen, rejected = (left, right) if left_reward > right_reward else (right, left)
    chosen_action, rejected_action = (
        (left_action, right_action) if chosen is left else (right_action, left_action)
    )
    quality_gap = float(chosen["quality"]) - float(rejected["quality"])
    if quality_gap < -max_quality_regret:
        return None, "quality_guardrail_failed"
    efficiency_gap = _efficiency_score(chosen) - _efficiency_score(rejected)
    preference = RlaifPreference(
        preference_type=preference_type,  # type: ignore[arg-type]
        query_id=str(chosen["query_id"]),
        chosen_action_id=str(chosen["action_id"]),
        rejected_action_id=str(rejected["action_id"]),
        reward_gap=round(float(chosen["reward"]) - float(rejected["reward"]), 12),
        quality_gap=round(quality_gap, 12),
        efficiency_gap=round(efficiency_gap, 12),
        reason="higher_reward",
        metadata={
            "group_key": list(group_key),
            "chosen": _action_summary(chosen_action),
            "rejected": _action_summary(rejected_action),
        },
    )
    return preference.to_dict(), None


def _ineligible_reward_reason(reward: dict[str, Any]) -> str | None:
    if reward.get("reward_mode") == "ambiguous_feedback" or reward.get("metadata", {}).get("feedback_ambiguous"):
        return "ambiguous_feedback"
    if reward.get("reward") is None or reward.get("quality") is None:
        return "missing_quality"
    return None


def _context_policy_group_key(action: dict[str, Any]) -> tuple[Any, ...]:
    return (
        action.get("benchmark"),
        action.get("query_id"),
        action.get("retrieval_strategy"),
        action.get("fusion_strategy"),
        action.get("top_k"),
        action.get("generator_model"),
    )


def _retrieval_context_group_key(action: dict[str, Any]) -> tuple[Any, ...]:
    return (
        action.get("benchmark"),
        action.get("query_id"),
        action.get("top_k"),
        action.get("generator_model"),
    )


def _efficiency_score(reward: dict[str, Any]) -> float:
    return mean(
        [
            1.0 - float(reward.get("token_cost_norm") or 0.0),
            1.0 - float(reward.get("latency_norm") or 0.0),
            1.0 - float(reward.get("kv_cost_norm") or 0.0),
        ]
    )


def _action_summary(action: dict[str, Any]) -> dict[str, Any]:
    return {
        "retrieval_strategy": action.get("retrieval_strategy"),
        "fusion_strategy": action.get("fusion_strategy"),
        "context_policy": action.get("context_policy"),
        "budget_chars": action.get("budget_chars"),
        "adaptive_profile": action.get("adaptive_profile"),
        "selected_context_policy": action.get("selected_context_policy"),
        "selected_budget_chars": action.get("selected_budget_chars"),
        "generator_model": action.get("generator_model"),
    }


def _build_summary(
    *,
    output_dir: Path,
    action_count: int,
    feedback_count: int,
    duplicate_feedback_count: int,
    reward_rows: list[dict[str, Any]],
    preference_rows: list[dict[str, Any]],
    preference_skips: Counter[str],
    invalid_rows: list[dict[str, Any]],
    weights: RlaifRewardWeights,
    config: RlaifRewardConfig,
) -> dict[str, Any]:
    scored_reward_count = sum(1 for row in reward_rows if row.get("reward") is not None)
    return {
        "output_dir": str(output_dir),
        "action_count": action_count,
        "feedback_count": feedback_count,
        "duplicate_feedback_count": duplicate_feedback_count,
        "reward_count": len(reward_rows),
        "scored_reward_count": scored_reward_count,
        "missing_quality_count": len(reward_rows) - scored_reward_count,
        "preference_count": len(preference_rows),
        "reward_mode_counts": dict(Counter(row.get("reward_mode", "unknown") for row in reward_rows)),
        "preference_type_counts": dict(Counter(row.get("preference_type", "unknown") for row in preference_rows)),
        "preference_reason_counts": dict(Counter(row.get("reason", "unknown") for row in preference_rows)),
        "preference_skip_reason_counts": dict(preference_skips),
        "invalid_row_count": len(invalid_rows),
        "invalid_rows": invalid_rows,
        "weights": asdict(weights),
        "min_reward_delta": config.min_reward_delta,
        "max_quality_regret": config.max_quality_regret,
    }


def _render_summary_markdown(summary: dict[str, Any]) -> str:
    lines = [
        "# RLAIF Reward Summary",
        "",
        f"- Output dir: `{summary['output_dir']}`",
        f"- Actions: {summary['action_count']}",
        f"- Feedback rows: {summary['feedback_count']}",
        f"- Rewards: {summary['reward_count']}",
        f"- Scored rewards: {summary['scored_reward_count']}",
        f"- Missing-quality rewards: {summary['missing_quality_count']}",
        f"- Preferences: {summary['preference_count']}",
        f"- Invalid rows: {summary['invalid_row_count']}",
        "",
        "## Reward Modes",
        "",
        "| Mode | Count |",
        "| --- | ---: |",
    ]
    for key, value in sorted(summary["reward_mode_counts"].items()):
        lines.append(f"| `{key}` | {value} |")
    lines.extend(["", "## Preference Types", "", "| Type | Count |", "| --- | ---: |"])
    for key, value in sorted(summary["preference_type_counts"].items()):
        lines.append(f"| `{key}` | {value} |")
    if not summary["preference_type_counts"]:
        lines.append("| N/A | 0 |")
    lines.extend(["", "## Preference Skips", "", "| Reason | Count |", "| --- | ---: |"])
    for key, value in sorted(summary["preference_skip_reason_counts"].items()):
        lines.append(f"| `{key}` | {value} |")
    if not summary["preference_skip_reason_counts"]:
        lines.append("| N/A | 0 |")
    lines.extend(["", "## Weights", ""])
    for key, value in sorted(summary["weights"].items()):
        lines.append(f"- `{key}`: {value}")
    if summary["invalid_rows"]:
        lines.extend(["", "## Invalid Rows", "", "| Action | Query | Error |", "| --- | --- | --- |"])
        for row in summary["invalid_rows"]:
            lines.append(f"| `{row.get('action_id')}` | `{row.get('query_id')}` | {row['error']} |")
    return "\n".join(lines) + "\n"


@dataclass(frozen=True)
class _CostScales:
    max_token_cost: float
    max_latency_cost: float
    max_kv_cost: float

    @classmethod
    def from_actions(cls, actions: Iterable[dict[str, Any]]) -> _CostScales:
        token_costs: list[float] = []
        latency_costs: list[float] = []
        kv_costs: list[float] = []
        for action in actions:
            raw_costs = cls.raw_costs(action)
            _append_positive(token_costs, raw_costs["token_cost"])
            _append_positive(latency_costs, raw_costs["latency_cost"])
            _append_positive(kv_costs, raw_costs["kv_cost"])
        return cls(
            max_token_cost=max(token_costs, default=0.0),
            max_latency_cost=max(latency_costs, default=0.0),
            max_kv_cost=max(kv_costs, default=0.0),
        )

    @staticmethod
    def raw_costs(action: dict[str, Any]) -> dict[str, float | None]:
        token_usage = _dict_or_empty(action.get("token_usage"))
        context_metrics = _dict_or_empty(action.get("context_metrics"))
        latency = _dict_or_empty(action.get("latency"))
        kv_estimate = _dict_or_empty(action.get("kv_estimate"))
        return {
            "token_cost": _first_number(
                token_usage.get("total_tokens"),
                token_usage.get("prompt_tokens"),
                token_usage.get("estimated_tokens"),
                token_usage.get("estimated_prompt_tokens_after_budget"),
                context_metrics.get("kept_context_est_tokens"),
                context_metrics.get("kept_context_chars"),
            ),
            "latency_cost": _first_number(latency.get("total_latency_s"), latency.get("answer_latency_s")),
            "kv_cost": _first_number(kv_estimate.get("after_mb"), kv_estimate.get("after_bytes")),
        }

    def token_norm(self, action: dict[str, Any]) -> float:
        return _norm(self.raw_costs(action)["token_cost"], self.max_token_cost)

    def latency_norm(self, action: dict[str, Any]) -> float:
        return _norm(self.raw_costs(action)["latency_cost"], self.max_latency_cost)

    def kv_norm(self, action: dict[str, Any]) -> float:
        return _norm(self.raw_costs(action)["kv_cost"], self.max_kv_cost)


def _weights_from_config(config: RlaifRewardConfig) -> RlaifRewardWeights:
    return RlaifRewardWeights(
        quality=config.quality_weight,
        support=config.support_weight,
        token=config.token_weight,
        latency=config.latency_weight,
        kv=config.kv_weight,
        error=config.error_weight,
        unsupported=config.unsupported_weight,
    )


def _validate_config(config: RlaifRewardConfig) -> None:
    if config.min_reward_delta < 0:
        raise ValueError("min_reward_delta must be non-negative")
    if config.max_quality_regret < 0:
        raise ValueError("max_quality_regret must be non-negative")
    if not config.actions_path.is_file():
        raise ValueError(f"Actions path does not exist: {config.actions_path}")
    if not config.feedback_path.is_file():
        raise ValueError(f"Feedback path does not exist: {config.feedback_path}")


def _index_feedback(feedback_rows: list[dict[str, Any]]) -> tuple[dict[str, dict[str, Any]], int]:
    feedback_by_action_id: dict[str, dict[str, Any]] = {}
    duplicate_count = 0
    for row in feedback_rows:
        action_id = row.get("action_id")
        if not isinstance(action_id, str) or not action_id.strip():
            continue
        if action_id in feedback_by_action_id:
            duplicate_count += 1
            continue
        feedback_by_action_id[action_id] = row
    return feedback_by_action_id, duplicate_count


def _feedback_provenance(value: Any) -> str:
    return value if value in {"gold", "ragas", "ai_judge", "mimo_judge", "heuristic", "missing"} else "missing"


def _reward_mode(feedback: dict[str, Any]) -> str:
    provenance = _feedback_provenance(feedback.get("provenance"))
    if provenance == "missing":
        return "missing_quality"
    return "ai_judge" if provenance == "mimo_judge" else provenance


def _evidence_support(feedback: dict[str, Any]) -> float:
    for value in (
        _score_or_none(feedback.get("faithfulness")),
        _score_or_none(feedback.get("answer_correctness")),
        _score_or_none(feedback.get("answer_relevancy")),
    ):
        if value is not None:
            return value
    return 0.0


def _error_penalty(action: dict[str, Any], feedback: dict[str, Any]) -> float:
    generation = _dict_or_empty(action.get("generation"))
    if generation.get("error") or feedback.get("missing_reason") == "generation_error":
        return 1.0
    return 0.0


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


def _required_text(value: Any, field_name: str) -> str:
    if isinstance(value, str) and value.strip():
        return value
    raise ValueError(f"{field_name} must be a non-empty string")


def _score_or_none(value: Any) -> float | None:
    number = _first_number(value)
    if number is None or number < 0.0 or number > 1.0:
        return None
    return number


def _first_number(*values: Any) -> float | None:
    for value in values:
        if value is None or isinstance(value, bool):
            continue
        if isinstance(value, (int, float)):
            return float(value)
        if isinstance(value, str) and value.strip():
            try:
                return float(value)
            except ValueError:
                continue
    return None


def _append_positive(values: list[float], value: float | None) -> None:
    if value is not None and value > 0:
        values.append(value)


def _norm(value: float | None, max_value: float) -> float:
    if value is None or value <= 0 or max_value <= 0:
        return 0.0
    return max(0.0, min(1.0, value / max_value))


def _dict_or_empty(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}
