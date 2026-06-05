#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


DEFAULT_GROUPS = (
    "retrieval_strategy",
    "context_policy",
    "retrieval_strategy,context_policy",
    "retrieval_strategy,context_policy,adaptive_profile",
    "budget_chars",
)


@dataclass
class GroupStats:
    group_by: str
    group_value: str
    rows: int = 0
    label_rows: int = 0
    clean_labels: int = 0
    scored_labels: int = 0
    missing_answer: int = 0
    invalid_json: int = 0
    ambiguous: int = 0
    scored_rewards: int = 0
    qualities: list[float] = field(default_factory=list)
    correctness: list[float] = field(default_factory=list)
    support: list[float] = field(default_factory=list)
    unsupported: list[float] = field(default_factory=list)
    rewards: list[float] = field(default_factory=list)
    token_costs: list[float] = field(default_factory=list)
    latency_costs: list[float] = field(default_factory=list)
    kv_costs: list[float] = field(default_factory=list)
    answer_latency_s: list[float] = field(default_factory=list)
    kept_chars: list[float] = field(default_factory=list)

    def to_row(self) -> dict[str, Any]:
        return {
            "group_by": self.group_by,
            "group": self.group_value,
            "rows": self.rows,
            "label_rows": self.label_rows,
            "clean_labels": self.clean_labels,
            "scored_labels": self.scored_labels,
            "missing_answer": self.missing_answer,
            "invalid_json": self.invalid_json,
            "ambiguous": self.ambiguous,
            "scored_rewards": self.scored_rewards,
            "quality_mean": _mean(self.qualities),
            "answer_correctness_mean": _mean(self.correctness),
            "evidence_support_mean": _mean(self.support),
            "unsupported_claim_penalty_mean": _mean(self.unsupported),
            "reward_mean": _mean(self.rewards),
            "token_cost_mean": _mean(self.token_costs),
            "latency_cost_mean": _mean(self.latency_costs),
            "kv_cost_mean": _mean(self.kv_costs),
            "answer_latency_s_mean": _mean(self.answer_latency_s),
            "kept_context_chars_mean": _mean(self.kept_chars),
        }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Analyze answer-label/reward quality by retriever, policy, and budget.",
    )
    parser.add_argument("--actions", type=Path, required=True, help="Path to rlaif_actions.jsonl.")
    parser.add_argument("--answer-labels", type=Path, required=True, help="Path to rlaif_answer_labels.jsonl.")
    parser.add_argument("--rewards", type=Path, default=None, help="Optional path to rlaif_rewards.jsonl.")
    parser.add_argument("--out-csv", type=Path, required=True)
    parser.add_argument("--out-md", type=Path, default=None)
    parser.add_argument("--out-json", type=Path, default=None)
    parser.add_argument(
        "--group-by",
        default=";".join(DEFAULT_GROUPS),
        help="Semicolon-separated group specs. Each spec can contain comma-separated action fields.",
    )
    parser.add_argument("--include-ambiguous", action="store_true")
    args = parser.parse_args(argv)

    summary = analyze_answer_quality(
        actions_path=args.actions,
        labels_path=args.answer_labels,
        rewards_path=args.rewards,
        groups=_parse_groups(args.group_by),
        include_ambiguous=args.include_ambiguous,
    )
    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    _write_csv(args.out_csv, summary["rows"])
    if args.out_json:
        args.out_json.parent.mkdir(parents=True, exist_ok=True)
        args.out_json.write_text(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if args.out_md:
        args.out_md.parent.mkdir(parents=True, exist_ok=True)
        args.out_md.write_text(render_markdown(summary), encoding="utf-8")
    print(
        json.dumps(
            {
                "actions": summary["action_count"],
                "labels": summary["label_count"],
                "group_rows": len(summary["rows"]),
                "out_csv": str(args.out_csv),
            },
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
    )
    return 0


def analyze_answer_quality(
    *,
    actions_path: Path,
    labels_path: Path,
    rewards_path: Path | None = None,
    groups: list[tuple[str, ...]] | None = None,
    include_ambiguous: bool = False,
) -> dict[str, Any]:
    actions = _index_by_action_id(_read_jsonl(actions_path))
    labels = _index_by_action_id(_read_jsonl(labels_path))
    rewards = _index_by_action_id(_read_jsonl(rewards_path)) if rewards_path else {}
    groups = groups or [tuple(group.split(",")) for group in DEFAULT_GROUPS]

    buckets: dict[tuple[str, str], GroupStats] = {}
    for action_id, action in actions.items():
        label = labels.get(action_id)
        reward = rewards.get(action_id, {})
        if label is None:
            continue
        if not include_ambiguous and bool(label.get("ambiguous")) and label.get("overall_quality") is None:
            # Keep clean scored labels by default; skip labels that are ambiguous and unscored.
            continue
        for group_fields in groups:
            group_by = ",".join(group_fields)
            group_value = " / ".join(_group_field(action, field_name) for field_name in group_fields)
            stats = buckets.setdefault((group_by, group_value), GroupStats(group_by=group_by, group_value=group_value))
            _add_row(stats, action=action, label=label, reward=reward)

    rows = [stats.to_row() for stats in sorted(buckets.values(), key=lambda item: (item.group_by, item.group_value))]
    return {
        "schema_version": "retriever-diversity-answer-quality-v1",
        "actions_path": str(actions_path),
        "answer_labels_path": str(labels_path),
        "rewards_path": str(rewards_path) if rewards_path else None,
        "action_count": len(actions),
        "label_count": len(labels),
        "reward_count": len(rewards),
        "include_ambiguous": include_ambiguous,
        "rows": rows,
    }


def render_markdown(summary: dict[str, Any]) -> str:
    lines = [
        "# Retriever-Diversity Answer Quality",
        "",
        f"- Actions: `{summary['actions_path']}`",
        f"- Answer labels: `{summary['answer_labels_path']}`",
        f"- Rewards: `{summary['rewards_path'] or 'N/A'}`",
        f"- Action rows: {summary['action_count']}",
        f"- Label rows: {summary['label_count']}",
        f"- Reward rows: {summary['reward_count']}",
        "",
        "This report groups answer-level AI-judge labels and optional rewards by retriever, context policy, retriever-policy pair, adaptive profile, and budget. Ambiguous unscored labels are excluded by default; they are counted in the source label summary, not forced to zero.",
        "",
    ]
    for group_by in _group_order(summary["rows"]):
        rows = [row for row in summary["rows"] if row["group_by"] == group_by]
        lines.extend(
            [
                f"## By `{group_by}`",
                "",
                "| Group | Rows | Scored labels | Missing answers | Quality | Correctness | Support | Unsupported | Reward | Token cost | Latency | KV cost |",
                "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
            ]
        )
        for row in rows:
            lines.append(
                "| "
                + " | ".join(
                    [
                        f"`{row['group']}`",
                        str(row["rows"]),
                        str(row["scored_labels"]),
                        str(row["missing_answer"]),
                        _fmt(row["quality_mean"]),
                        _fmt(row["answer_correctness_mean"]),
                        _fmt(row["evidence_support_mean"]),
                        _fmt(row["unsupported_claim_penalty_mean"]),
                        _fmt(row["reward_mean"]),
                        _fmt(row["token_cost_mean"]),
                        _fmt(row["latency_cost_mean"]),
                        _fmt(row["kv_cost_mean"]),
                    ]
                )
                + " |"
            )
        lines.append("")
    lines.extend(
        [
            "## Interpretation Notes",
            "",
            "- Use `scored_labels` and `missing_answer` together; high quality over a small clean subset can hide generation failures.",
            "- `unsupported_claim_penalty` is a risk score where higher is worse.",
            "- Reward and cost columns are present only when `--rewards` is provided.",
        ]
    )
    return "\n".join(lines).rstrip() + "\n"


def _add_row(stats: GroupStats, *, action: dict[str, Any], label: dict[str, Any], reward: dict[str, Any]) -> None:
    stats.rows += 1
    stats.label_rows += 1
    if not bool(label.get("ambiguous")):
        stats.clean_labels += 1
    if bool(label.get("invalid_json")):
        stats.invalid_json += 1
    if bool(label.get("ambiguous")):
        stats.ambiguous += 1
    if label.get("missing_reason") == "missing_answer":
        stats.missing_answer += 1
    for field_name, target in (
        ("overall_quality", stats.qualities),
        ("answer_correctness", stats.correctness),
        ("evidence_support", stats.support),
        ("unsupported_claim_penalty", stats.unsupported),
    ):
        value = _float_or_none(label.get(field_name))
        if value is not None:
            target.append(value)
    if label.get("overall_quality") is not None:
        stats.scored_labels += 1
    reward_value = _float_or_none(reward.get("reward"))
    if reward_value is not None:
        stats.scored_rewards += 1
        stats.rewards.append(reward_value)
    components = reward.get("reward_components") if isinstance(reward.get("reward_components"), dict) else {}
    for field_name, target in (
        ("token_cost_norm", stats.token_costs),
        ("latency_norm", stats.latency_costs),
        ("kv_cost_norm", stats.kv_costs),
    ):
        value = _float_or_none(components.get(field_name))
        if value is not None:
            target.append(value)
    latency = action.get("latency") if isinstance(action.get("latency"), dict) else {}
    answer_latency = _float_or_none(latency.get("answer_latency_s"))
    if answer_latency is not None:
        stats.answer_latency_s.append(answer_latency)
    context_metrics = action.get("context_metrics") if isinstance(action.get("context_metrics"), dict) else {}
    kept_chars = _float_or_none(context_metrics.get("kept_context_chars"))
    if kept_chars is not None:
        stats.kept_chars.append(kept_chars)


def _read_jsonl(path: Path | None) -> list[dict[str, Any]]:
    if path is None:
        return []
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            row = json.loads(stripped)
            if not isinstance(row, dict):
                raise ValueError(f"{path}:{line_no}: expected JSON object")
            rows.append(row)
    return rows


def _index_by_action_id(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    return {str(row["action_id"]): row for row in rows if row.get("action_id")}


def _parse_groups(value: str) -> list[tuple[str, ...]]:
    groups = []
    for group in value.split(";"):
        fields = tuple(field.strip() for field in group.split(",") if field.strip())
        if fields:
            groups.append(fields)
    return groups


def _group_field(action: dict[str, Any], field_name: str) -> str:
    if field_name == "retriever":
        field_name = "retrieval_strategy"
    if field_name == "selected_context_policy":
        metrics = action.get("context_metrics") if isinstance(action.get("context_metrics"), dict) else {}
        return str(action.get("selected_context_policy") or metrics.get("policy") or "unknown")
    value = action.get(field_name)
    return str(value if value not in (None, "") else "unknown")


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fieldnames = [
        "group_by",
        "group",
        "rows",
        "label_rows",
        "clean_labels",
        "scored_labels",
        "missing_answer",
        "invalid_json",
        "ambiguous",
        "scored_rewards",
        "quality_mean",
        "answer_correctness_mean",
        "evidence_support_mean",
        "unsupported_claim_penalty_mean",
        "reward_mean",
        "token_cost_mean",
        "latency_cost_mean",
        "kv_cost_mean",
        "answer_latency_s_mean",
        "kept_context_chars_mean",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: _csv_value(row.get(key)) for key in fieldnames})


def _group_order(rows: list[dict[str, Any]]) -> list[str]:
    seen = []
    for row in rows:
        value = str(row["group_by"])
        if value not in seen:
            seen.append(value)
    return seen


def _float_or_none(value: Any) -> float | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _mean(values: list[float]) -> float | None:
    return (sum(values) / len(values)) if values else None


def _fmt(value: Any) -> str:
    if value is None:
        return "N/A"
    if isinstance(value, float):
        return f"{value:.3f}"
    return str(value)


def _csv_value(value: Any) -> Any:
    if isinstance(value, float):
        return f"{value:.6f}"
    return "" if value is None else value


if __name__ == "__main__":
    raise SystemExit(main())
