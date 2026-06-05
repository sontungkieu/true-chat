#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from statistics import mean
from typing import Any


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Compare two RLAIF reward sets and summarize reward deltas.")
    parser.add_argument("--base", type=Path, required=True, help="Baseline rlaif_rewards.jsonl.")
    parser.add_argument("--candidate", type=Path, required=True, help="Candidate rlaif_rewards.jsonl.")
    parser.add_argument("--out-md", type=Path, default=None, help="Markdown output path.")
    parser.add_argument("--out-json", type=Path, default=None, help="JSON output path.")
    parser.add_argument("--changed-epsilon", type=float, default=1e-9)
    args = parser.parse_args(argv)

    summary = compare_reward_sets(base_path=args.base, candidate_path=args.candidate, changed_epsilon=args.changed_epsilon)
    out_json = args.out_json or args.candidate.with_suffix(".delta.json")
    out_md = args.out_md or args.candidate.with_suffix(".delta.md")
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_md.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    out_md.write_text(render_markdown(summary), encoding="utf-8")
    print(
        json.dumps(
            {
                "out_json": str(out_json),
                "out_md": str(out_md),
                "shared_action_count": summary["shared_action_count"],
                "changed_reward_count": summary["changed_reward_count"],
                "mean_delta_changed": summary["delta_stats_changed"]["mean"],
            },
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
    )
    return 0


def compare_reward_sets(*, base_path: Path, candidate_path: Path, changed_epsilon: float = 1e-9) -> dict[str, Any]:
    if changed_epsilon < 0:
        raise ValueError("--changed-epsilon must be non-negative")
    base_rows = _index_by_action_id(read_jsonl(base_path))
    candidate_rows = _index_by_action_id(read_jsonl(candidate_path))
    shared_ids = sorted(set(base_rows) & set(candidate_rows))
    deltas: list[float] = []
    changed_deltas: list[float] = []
    changed_by_sufficiency: Counter[str] = Counter()
    changed_by_merge: Counter[str] = Counter()
    clipped_counts: Counter[str] = Counter()
    missing_reward_counts: Counter[str] = Counter()

    for action_id in shared_ids:
        base_reward = _number_or_none(base_rows[action_id].get("reward"))
        candidate_reward = _number_or_none(candidate_rows[action_id].get("reward"))
        if base_reward is None:
            missing_reward_counts["base_missing"] += 1
            continue
        if candidate_reward is None:
            missing_reward_counts["candidate_missing"] += 1
            continue
        delta = candidate_reward - base_reward
        deltas.append(delta)
        if candidate_reward <= -1.0:
            clipped_counts["candidate_at_minus_one"] += 1
        if candidate_reward >= 1.0:
            clipped_counts["candidate_at_plus_one"] += 1
        if base_reward <= -1.0:
            clipped_counts["base_at_minus_one"] += 1
        if base_reward >= 1.0:
            clipped_counts["base_at_plus_one"] += 1
        if abs(delta) > changed_epsilon:
            changed_deltas.append(delta)
            metadata = _dict_or_empty(candidate_rows[action_id].get("metadata"))
            context_label = _dict_or_empty(metadata.get("context_label"))
            changed_by_merge[_text(metadata.get("context_label_merge"), "missing")] += 1
            changed_by_sufficiency[_text(context_label.get("sufficient"), "missing")] += 1

    return {
        "base_path": str(base_path),
        "candidate_path": str(candidate_path),
        "base_action_count": len(base_rows),
        "candidate_action_count": len(candidate_rows),
        "shared_action_count": len(shared_ids),
        "missing_reward_counts": dict(sorted(missing_reward_counts.items())),
        "changed_epsilon": changed_epsilon,
        "changed_reward_count": len(changed_deltas),
        "negative_delta_count": sum(1 for value in changed_deltas if value < 0),
        "positive_delta_count": sum(1 for value in changed_deltas if value > 0),
        "zero_delta_count": len(deltas) - len(changed_deltas),
        "delta_stats_all_scored": _stats(deltas),
        "delta_stats_changed": _stats(changed_deltas),
        "clipped_counts": dict(sorted(clipped_counts.items())),
        "changed_by_context_label_merge": dict(sorted(changed_by_merge.items())),
        "changed_by_context_sufficient": dict(sorted(changed_by_sufficiency.items())),
    }


def render_markdown(summary: dict[str, Any]) -> str:
    lines = [
        "# RLAIF Reward Delta Diagnostics",
        "",
        f"- Base: `{summary['base_path']}`",
        f"- Candidate: `{summary['candidate_path']}`",
        "",
        "## Counts",
        "",
        "| Metric | Value |",
        "| --- | ---: |",
        f"| base actions | {summary['base_action_count']} |",
        f"| candidate actions | {summary['candidate_action_count']} |",
        f"| shared actions | {summary['shared_action_count']} |",
        f"| changed rewards | {summary['changed_reward_count']} |",
        f"| negative deltas | {summary['negative_delta_count']} |",
        f"| positive deltas | {summary['positive_delta_count']} |",
        f"| zero deltas | {summary['zero_delta_count']} |",
        "",
        "## Delta Distribution",
        "",
        "| Scope | min | p25 | median | p75 | max | mean |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
        _stats_row("all scored", summary["delta_stats_all_scored"]),
        _stats_row("changed only", summary["delta_stats_changed"]),
        "",
        "## Clipped Reward Counts",
        "",
        "| Clip | Count |",
        "| --- | ---: |",
    ]
    _append_counter_rows(lines, summary["clipped_counts"])
    lines.extend(
        [
            "",
            "## Changed Rows By Context Merge",
            "",
            "| Merge | Count |",
            "| --- | ---: |",
        ]
    )
    _append_counter_rows(lines, summary["changed_by_context_label_merge"])
    lines.extend(
        [
            "",
            "## Changed Rows By Context Sufficiency",
            "",
            "| Sufficient | Count |",
            "| --- | ---: |",
        ]
    )
    _append_counter_rows(lines, summary["changed_by_context_sufficient"])
    lines.extend(
        [
            "",
            "Interpretation: large negative deltas or many candidate rewards at -1 indicate an aggressive context-label reward candidate that should remain non-default until ablated and evaluated with fuller label coverage.",
        ]
    )
    return "\n".join(lines).rstrip() + "\n"


def read_jsonl(path: Path) -> list[dict[str, Any]]:
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


def _stats(values: list[float]) -> dict[str, float | int | None]:
    if not values:
        return {"count": 0, "min": None, "p25": None, "median": None, "p75": None, "max": None, "mean": None}
    ordered = sorted(values)
    return {
        "count": len(values),
        "min": ordered[0],
        "p25": _quantile(ordered, 0.25),
        "median": _quantile(ordered, 0.50),
        "p75": _quantile(ordered, 0.75),
        "max": ordered[-1],
        "mean": mean(values),
    }


def _quantile(ordered_values: list[float], q: float) -> float:
    if len(ordered_values) == 1:
        return ordered_values[0]
    position = q * (len(ordered_values) - 1)
    lower = int(position)
    upper = min(lower + 1, len(ordered_values) - 1)
    weight = position - lower
    return ordered_values[lower] * (1.0 - weight) + ordered_values[upper] * weight


def _stats_row(label: str, stats: dict[str, Any]) -> str:
    return (
        f"| {label} | {_fmt(stats['min'])} | {_fmt(stats['p25'])} | {_fmt(stats['median'])} | "
        f"{_fmt(stats['p75'])} | {_fmt(stats['max'])} | {_fmt(stats['mean'])} |"
    )


def _append_counter_rows(lines: list[str], counts: dict[str, int]) -> None:
    if not counts:
        lines.append("| N/A | 0 |")
        return
    for key, value in counts.items():
        lines.append(f"| `{key}` | {value} |")


def _number_or_none(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int | float):
        return float(value)
    return None


def _dict_or_empty(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _text(value: Any, fallback: str) -> str:
    if isinstance(value, bool):
        return str(value).lower()
    if value is None:
        return fallback
    text = str(value).strip()
    return text or fallback


def _fmt(value: Any) -> str:
    if value is None:
        return "N/A"
    if isinstance(value, float):
        return f"{value:.3f}"
    return str(value)


if __name__ == "__main__":
    raise SystemExit(main())
