#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from statistics import mean
from typing import Any


DEFAULT_QUALITY_TIE_THRESHOLD = 0.03
DEFAULT_SUPPORT_TIE_THRESHOLD = 0.03


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Diagnose pairwise RLAIF reward calibration issues from direct judge labels.",
    )
    parser.add_argument("--labels", type=Path, required=True, help="Path to rlaif_pairwise_labels.jsonl.")
    parser.add_argument("--rewards", type=Path, required=True, help="Path to rlaif_rewards.jsonl.")
    parser.add_argument("--actions", type=Path, default=None, help="Optional path to rlaif_actions.jsonl for readable examples.")
    parser.add_argument("--out-md", type=Path, default=None, help="Markdown diagnostics output path.")
    parser.add_argument("--out-json", type=Path, default=None, help="JSON diagnostics output path.")
    parser.add_argument("--quality-tie-threshold", type=float, default=DEFAULT_QUALITY_TIE_THRESHOLD)
    parser.add_argument("--support-tie-threshold", type=float, default=DEFAULT_SUPPORT_TIE_THRESHOLD)
    parser.add_argument("--examples-limit", type=int, default=8)
    args = parser.parse_args(argv)

    summary = diagnose_pairwise_calibration(
        labels_path=args.labels,
        rewards_path=args.rewards,
        actions_path=args.actions,
        quality_tie_threshold=args.quality_tie_threshold,
        support_tie_threshold=args.support_tie_threshold,
        examples_limit=args.examples_limit,
    )
    out_json = args.out_json or args.labels.with_suffix(".calibration.json")
    out_md = args.out_md or args.labels.with_suffix(".calibration.md")
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_md.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    out_md.write_text(render_markdown(summary), encoding="utf-8")
    print(json.dumps({"out_json": str(out_json), "out_md": str(out_md), **_compact_summary(summary)}, indent=2))
    return 0


def diagnose_pairwise_calibration(
    *,
    labels_path: Path,
    rewards_path: Path,
    actions_path: Path | None = None,
    quality_tie_threshold: float = DEFAULT_QUALITY_TIE_THRESHOLD,
    support_tie_threshold: float = DEFAULT_SUPPORT_TIE_THRESHOLD,
    examples_limit: int = 8,
) -> dict[str, Any]:
    if quality_tie_threshold < 0.0:
        raise ValueError("--quality-tie-threshold must be non-negative")
    if support_tie_threshold < 0.0:
        raise ValueError("--support-tie-threshold must be non-negative")
    if examples_limit < 0:
        raise ValueError("--examples-limit must be non-negative")

    labels = read_jsonl(labels_path)
    rewards = {row["action_id"]: row for row in read_jsonl(rewards_path) if row.get("action_id")}
    actions = {row["action_id"]: row for row in read_jsonl(actions_path) if row.get("action_id")} if actions_path else {}

    valid_rows = []
    small_delta_rows = []
    cheaper_wins_when_quality_tied = []
    scalar_over_quality_disagreements = []
    query_counter: Counter[str] = Counter()
    cheaper_winner_counts: Counter[str] = Counter()

    for row in labels:
        enriched = _enrich_row(row, rewards, actions)
        if not _is_valid_decision(row):
            continue
        valid_rows.append(enriched)
        if _is_small_quality_support_delta(enriched, quality_tie_threshold, support_tie_threshold):
            small_delta_rows.append(enriched)
            if enriched["judge_chose_cheaper_action"]:
                cheaper_wins_when_quality_tied.append(enriched)
        if _is_scalar_over_quality_disagreement(enriched, quality_tie_threshold, support_tie_threshold):
            scalar_over_quality_disagreements.append(enriched)
            query_counter[_text(row.get("query_id"), "missing")] += 1
        cheaper_winner_counts[_text(enriched.get("cheaper_action_side"), "missing")] += 1

    disagreement_abs_quality_gaps = [abs(row["quality_gap_components"]) for row in scalar_over_quality_disagreements]
    disagreement_abs_support_gaps = [abs(row["support_gap_components"]) for row in scalar_over_quality_disagreements]
    suggested_quality_tie_threshold = max(disagreement_abs_quality_gaps) if disagreement_abs_quality_gaps else quality_tie_threshold
    suggested_support_tie_threshold = max(disagreement_abs_support_gaps) if disagreement_abs_support_gaps else support_tie_threshold

    return {
        "labels_path": str(labels_path),
        "rewards_path": str(rewards_path),
        "actions_path": str(actions_path) if actions_path else None,
        "quality_tie_threshold": quality_tie_threshold,
        "support_tie_threshold": support_tie_threshold,
        "label_count": len(labels),
        "valid_decision_count": len(valid_rows),
        "small_quality_delta_pairs": len(small_delta_rows),
        "cheaper_wins_when_quality_tied": len(cheaper_wins_when_quality_tied),
        "scalar_over_quality_disagreements": len(scalar_over_quality_disagreements),
        "scalar_over_quality_disagreement_rate": _ratio(len(scalar_over_quality_disagreements), len(valid_rows)),
        "cheaper_win_rate_when_quality_tied": _ratio(len(cheaper_wins_when_quality_tied), len(small_delta_rows)),
        "suggested_delta_threshold": {
            "quality": suggested_quality_tie_threshold,
            "support": suggested_support_tie_threshold,
            "source": "max_abs_gap_among_scalar_over_quality_disagreements"
            if scalar_over_quality_disagreements
            else "configured_threshold",
        },
        "query_counts_for_scalar_over_quality_disagreements": dict(sorted(query_counter.items())),
        "cheaper_action_side_counts": dict(sorted(cheaper_winner_counts.items())),
        "mean_abs_quality_gap_for_scalar_over_quality_disagreements": _mean_or_none(disagreement_abs_quality_gaps),
        "mean_abs_support_gap_for_scalar_over_quality_disagreements": _mean_or_none(disagreement_abs_support_gaps),
        "examples": [_example_row(row) for row in scalar_over_quality_disagreements[:examples_limit]],
    }


def render_markdown(summary: dict[str, Any]) -> str:
    lines = [
        "# RLAIF Pairwise Calibration Diagnostics",
        "",
        f"- Labels: `{summary['labels_path']}`",
        f"- Rewards: `{summary['rewards_path']}`",
        f"- Actions: `{summary['actions_path'] or 'N/A'}`",
        "",
        "## Thresholds",
        "",
        "| Threshold | Value |",
        "| --- | ---: |",
        f"| quality tie threshold | {_fmt(summary['quality_tie_threshold'])} |",
        f"| support tie threshold | {_fmt(summary['support_tie_threshold'])} |",
        "",
        "## Diagnostics",
        "",
        "| Metric | Value |",
        "| --- | ---: |",
    ]
    for key in (
        "label_count",
        "valid_decision_count",
        "small_quality_delta_pairs",
        "cheaper_wins_when_quality_tied",
        "scalar_over_quality_disagreements",
        "scalar_over_quality_disagreement_rate",
        "cheaper_win_rate_when_quality_tied",
        "mean_abs_quality_gap_for_scalar_over_quality_disagreements",
        "mean_abs_support_gap_for_scalar_over_quality_disagreements",
    ):
        lines.append(f"| {key.replace('_', ' ')} | {_fmt(summary[key])} |")

    suggested = summary["suggested_delta_threshold"]
    lines.extend(
        [
            "",
            "## Suggested Candidate Threshold",
            "",
            "| Field | Value |",
            "| --- | ---: |",
            f"| quality | {_fmt(suggested['quality'])} |",
            f"| support | {_fmt(suggested['support'])} |",
            f"| source | `{suggested['source']}` |",
            "",
            "## Query Counts For Scalar-Over-Quality Disagreements",
            "",
            "| Query id | Count |",
            "| --- | ---: |",
        ]
    )
    query_counts = summary["query_counts_for_scalar_over_quality_disagreements"]
    if query_counts:
        for query_id, count in query_counts.items():
            lines.append(f"| `{query_id}` | {count} |")
    else:
        lines.append("| N/A | 0 |")

    lines.extend(["", "## Examples", ""])
    examples = summary["examples"]
    if not examples:
        lines.append("No scalar-over-quality disagreement examples matched the configured thresholds.")
    else:
        lines.extend(
            [
                "| Preference | Query | A | B | Quality gap | Support gap | Cheaper side | Judge | Rationale |",
                "| --- | --- | --- | --- | ---: | ---: | --- | --- | --- |",
            ]
        )
        for row in examples:
            lines.append(
                "| "
                + " | ".join(
                    [
                        f"`{row['preference_id']}`",
                        f"`{row['query_id']}`",
                        _escape_md(row["action_a_summary"]),
                        _escape_md(row["action_b_summary"]),
                        _fmt(row["quality_gap_components"]),
                        _fmt(row["support_gap_components"]),
                        _text(row["cheaper_action_side"], "N/A"),
                        _text(row["chosen"], "N/A"),
                        _escape_md(_text(row["short_rationale"], "")),
                    ]
                )
                + " |"
            )

    lines.extend(
        [
            "",
            "Interpretation: these diagnostics identify cases where scalar reward may overvalue small quality/support differences when direct pairwise labels treat the answers as tied or acceptable and prefer lower resource cost.",
            "They are analysis-only signals and do not change reward defaults.",
        ]
    )
    return "\n".join(lines).rstrip() + "\n"


def read_jsonl(path: Path | None) -> list[dict[str, Any]]:
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
                raise ValueError(f"{path}:{line_no}: expected JSON object row")
            rows.append(row)
    return rows


def _enrich_row(row: dict[str, Any], rewards: dict[str, dict[str, Any]], actions: dict[str, dict[str, Any]]) -> dict[str, Any]:
    action_a_id = _text(row.get("action_a_id"), "")
    action_b_id = _text(row.get("action_b_id"), "")
    reward_a = rewards.get(action_a_id, {})
    reward_b = rewards.get(action_b_id, {})
    cost_a = _total_cost(reward_a)
    cost_b = _total_cost(reward_b)
    cheaper_side = None
    if cost_a is not None and cost_b is not None:
        if cost_a < cost_b:
            cheaper_side = "A"
        elif cost_b < cost_a:
            cheaper_side = "B"
        else:
            cheaper_side = "tie"
    chosen = _winner_key(row.get("chosen"))
    quality_gap = _component(reward_a, "quality") - _component(reward_b, "quality")
    support_gap = _component(reward_a, "evidence_support") - _component(reward_b, "evidence_support")
    enriched = dict(row)
    enriched.update(
        {
            "quality_gap_components": quality_gap,
            "support_gap_components": support_gap,
            "total_cost_a": cost_a,
            "total_cost_b": cost_b,
            "cheaper_action_side": cheaper_side,
            "judge_chose_cheaper_action": cheaper_side in {"A", "B"} and chosen == cheaper_side,
            "action_a_summary": _action_summary(actions.get(action_a_id, {}), reward_a),
            "action_b_summary": _action_summary(actions.get(action_b_id, {}), reward_b),
        }
    )
    return enriched


def _is_valid_decision(row: dict[str, Any]) -> bool:
    if row.get("ambiguous") or row.get("tie") or row.get("invalid_json") or row.get("missing_reason") or row.get("error"):
        return False
    return _winner_key(row.get("chosen")) in {"A", "B"}


def _is_small_quality_support_delta(row: dict[str, Any], quality_threshold: float, support_threshold: float) -> bool:
    return abs(row["quality_gap_components"]) <= quality_threshold and abs(row["support_gap_components"]) <= support_threshold


def _is_scalar_over_quality_disagreement(row: dict[str, Any], quality_threshold: float, support_threshold: float) -> bool:
    if _winner_key(row.get("chosen")) != "B":
        return False
    if not _is_small_quality_support_delta(row, quality_threshold, support_threshold):
        return False
    if _winner_key(row.get("answer_quality_winner")) not in {"tie", "missing"}:
        return False
    if _winner_key(row.get("evidence_support_winner")) not in {"tie", "missing"}:
        return False
    return bool(row.get("judge_chose_cheaper_action"))


def _component(reward: dict[str, Any], key: str) -> float:
    components = reward.get("reward_components") if isinstance(reward.get("reward_components"), dict) else {}
    value = components.get(key, reward.get(key))
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def _total_cost(reward: dict[str, Any]) -> float | None:
    keys = ("token_cost_norm", "latency_norm", "kv_cost_norm")
    values = [_component(reward, key) for key in keys]
    if not reward:
        return None
    return sum(values)


def _action_summary(action: dict[str, Any], reward: dict[str, Any]) -> str:
    parts = [
        _text(action.get("retriever") or action.get("retrieval_strategy"), "unknown"),
        _text(action.get("context_policy"), "unknown"),
    ]
    budget = action.get("budget_chars")
    if budget is not None:
        parts.append(f"budget={budget}")
    profile = action.get("adaptive_profile")
    if profile:
        parts.append(f"profile={profile}")
    if reward:
        parts.append(f"reward={_fmt(reward.get('reward'))}")
    return ", ".join(parts)


def _example_row(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "preference_id": row.get("preference_id"),
        "query_id": row.get("query_id"),
        "chosen": row.get("chosen"),
        "quality_gap_components": row.get("quality_gap_components"),
        "support_gap_components": row.get("support_gap_components"),
        "cheaper_action_side": row.get("cheaper_action_side"),
        "action_a_summary": row.get("action_a_summary"),
        "action_b_summary": row.get("action_b_summary"),
        "short_rationale": row.get("short_rationale") or row.get("rationale"),
    }


def _mean_or_none(values: list[float]) -> float | None:
    if not values:
        return None
    return mean(values)


def _ratio(numerator: int, denominator: int) -> float | None:
    if denominator <= 0:
        return None
    return numerator / denominator


def _winner_key(value: Any) -> str:
    if value is None:
        return "missing"
    text = str(value).strip()
    if not text:
        return "missing"
    if text.lower() == "tie":
        return "tie"
    upper = text.upper()
    if upper in {"A", "B"}:
        return upper
    return text


def _text(value: Any, fallback: str) -> str:
    text = str(value).strip() if value is not None else ""
    return text or fallback


def _fmt(value: Any) -> str:
    if value is None:
        return "N/A"
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)


def _escape_md(value: str) -> str:
    return value.replace("|", "\\|").replace("\n", " ")


def _compact_summary(summary: dict[str, Any]) -> dict[str, Any]:
    return {
        "label_count": summary["label_count"],
        "valid_decision_count": summary["valid_decision_count"],
        "small_quality_delta_pairs": summary["small_quality_delta_pairs"],
        "cheaper_wins_when_quality_tied": summary["cheaper_wins_when_quality_tied"],
        "scalar_over_quality_disagreements": summary["scalar_over_quality_disagreements"],
        "suggested_delta_threshold": summary["suggested_delta_threshold"],
    }


if __name__ == "__main__":
    raise SystemExit(main())
