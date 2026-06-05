#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from itertools import combinations
from pathlib import Path
from statistics import mean
from typing import Any


SCORE_FIELDS = ("context_quality_score", "evidence_support_score")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Aggregate targeted RLAIF multi-judge context-label audit results.")
    parser.add_argument("--mimo-labels", type=Path, required=True, help="Merged MiMo context labels.")
    parser.add_argument("--deepseek-labels", type=Path, nargs="*", default=[], help="DeepSeek context-label JSONL shards.")
    parser.add_argument("--groq-labels", type=Path, nargs="*", default=[], help="Optional Groq context-label JSONL shards.")
    parser.add_argument("--actions", type=Path, required=True, help="Targeted audit action/case JSONL.")
    parser.add_argument("--output-md", type=Path, required=True, help="Markdown report output.")
    parser.add_argument("--output-json", type=Path, required=True, help="JSON summary output.")
    args = parser.parse_args(argv)

    summary = aggregate_multijudge_audit(
        mimo_labels_path=args.mimo_labels,
        deepseek_label_paths=args.deepseek_labels,
        groq_label_paths=args.groq_labels,
        actions_path=args.actions,
    )
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_md.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    args.output_md.write_text(render_markdown(summary), encoding="utf-8")
    print(
        json.dumps(
            {
                "output_json": str(args.output_json),
                "output_md": str(args.output_md),
                "targeted_action_count": summary["targeted_action_count"],
                "high_disagreement_count": summary["high_disagreement_count"],
                "mimo_harsh_count": summary["mimo_harsh_count"],
                "consensus_insufficient_count": summary["consensus_insufficient_count"],
            },
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
    )
    return 0


def aggregate_multijudge_audit(
    *,
    mimo_labels_path: Path,
    deepseek_label_paths: list[Path],
    groq_label_paths: list[Path],
    actions_path: Path,
) -> dict[str, Any]:
    actions = read_jsonl(actions_path)
    targeted_ids = [str(row["action_id"]) for row in actions if row.get("action_id")]
    if not targeted_ids:
        raise ValueError(f"No action_id values found in {actions_path}")

    judge_labels = {
        "mimo": _index_best_label(read_jsonl(mimo_labels_path)),
        "deepseek": _index_best_label(_read_many(deepseek_label_paths)),
        "groq": _index_best_label(_read_many(groq_label_paths)),
    }
    judge_labels = {judge: labels for judge, labels in judge_labels.items() if labels or judge == "mimo"}

    judge_counts = {
        judge: _judge_count_summary(labels, targeted_ids)
        for judge, labels in judge_labels.items()
    }
    pairwise_agreement: dict[str, dict[str, Any]] = {}
    for left, right in combinations(judge_labels, 2):
        pairwise_agreement[f"{left}_vs_{right}"] = _sufficiency_agreement(
            judge_labels[left],
            judge_labels[right],
            targeted_ids,
        )

    score_correlations: dict[str, dict[str, Any]] = {}
    for left, right in combinations(judge_labels, 2):
        for field in SCORE_FIELDS:
            score_correlations[f"{left}_vs_{right}_{field}"] = _score_correlation(
                judge_labels[left],
                judge_labels[right],
                targeted_ids,
                field,
            )

    row_summaries = [
        _row_summary(action, {judge: labels.get(str(action.get("action_id") or "")) for judge, labels in judge_labels.items()})
        for action in actions
    ]
    high_disagreement_cases = [row for row in row_summaries if row["has_sufficiency_disagreement"]]
    mimo_harsh_cases = [row for row in row_summaries if row["mimo_harsh"]]
    consensus_insufficient_cases = [row for row in row_summaries if row["consensus_insufficient"]]

    return {
        "schema_version": "rlaif-multijudge-audit-aggregation-v1",
        "actions_path": str(actions_path),
        "mimo_labels_path": str(mimo_labels_path),
        "deepseek_label_paths": [str(path) for path in deepseek_label_paths],
        "groq_label_paths": [str(path) for path in groq_label_paths],
        "targeted_action_count": len(targeted_ids),
        "judge_counts": judge_counts,
        "sufficiency_agreement": pairwise_agreement,
        "score_correlations": score_correlations,
        "majority_vote_counts": _majority_vote_counts(row_summaries),
        "high_disagreement_count": len(high_disagreement_cases),
        "mimo_harsh_count": len(mimo_harsh_cases),
        "consensus_insufficient_count": len(consensus_insufficient_cases),
        "high_disagreement_cases": high_disagreement_cases[:20],
        "mimo_harsh_cases": mimo_harsh_cases[:20],
        "consensus_insufficient_cases": consensus_insufficient_cases[:20],
    }


def render_markdown(summary: dict[str, Any]) -> str:
    lines = [
        "# Phase 1D RLAIF Multi-Judge Targeted Audit",
        "",
        "This report aggregates secondary judge labels for a targeted RLAIF audit subset. It is an audit/confidence layer, not a reward-default replacement.",
        "",
        "## Inputs",
        "",
        f"- Actions: `{summary['actions_path']}`",
        f"- MiMo labels: `{summary['mimo_labels_path']}`",
        f"- DeepSeek labels: `{', '.join(summary['deepseek_label_paths']) or 'N/A'}`",
        f"- Groq labels: `{', '.join(summary['groq_label_paths']) or 'N/A'}`",
        "",
        "## Judge Counts",
        "",
        "| Judge | Labels | Valid | Ambiguous | Invalid JSON | Errors | Clean sufficiency |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for judge, counts in summary["judge_counts"].items():
        lines.append(
            f"| {judge} | {counts['label_count']} | {counts['valid_json_count']} | {counts['ambiguous_count']} | "
            f"{counts['invalid_json_count']} | {counts['error_count']} | {counts['clean_sufficiency_count']} |"
        )

    lines.extend(["", "## Sufficiency Agreement", "", "| Pair | Compared | Agree | Disagree | Agreement rate |", "| --- | ---: | ---: | ---: | ---: |"])
    for pair, item in summary["sufficiency_agreement"].items():
        lines.append(
            f"| {pair} | {item['compared_count']} | {item['agree_count']} | {item['disagree_count']} | {_fmt(item['agreement_rate'])} |"
        )

    lines.extend(["", "## Numeric Score Correlation", "", "| Pair / Score | N | Pearson |", "| --- | ---: | ---: |"])
    for pair, item in summary["score_correlations"].items():
        lines.append(f"| {pair} | {item['count']} | {_fmt(item['pearson'])} |")

    lines.extend(["", "## Audit Signals", "", "| Signal | Count |", "| --- | ---: |"])
    lines.append(f"| high disagreement cases | {summary['high_disagreement_count']} |")
    lines.append(f"| MiMo harsh cases | {summary['mimo_harsh_count']} |")
    lines.append(f"| consensus insufficient cases | {summary['consensus_insufficient_count']} |")
    for vote, count in summary["majority_vote_counts"].items():
        lines.append(f"| majority vote `{vote}` | {count} |")

    lines.extend(["", "## High-Disagreement Examples", "", "| Action | Query | Vote | Judge sufficiency | Selection reason |", "| --- | --- | --- | --- | --- |"])
    _append_case_rows(lines, summary["high_disagreement_cases"])
    lines.extend(["", "## MiMo-Harsh Examples", "", "| Action | Query | Vote | Judge sufficiency | Selection reason |", "| --- | --- | --- | --- | --- |"])
    _append_case_rows(lines, summary["mimo_harsh_cases"])
    lines.extend(
        [
            "",
            "Interpretation: disagreement is a low-confidence signal. The aggregation intentionally does not average judge scores or replace reward defaults. Rows with strong judge disagreement should be audited before being used as clean context supervision.",
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


def _read_many(paths: list[Path]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in paths:
        rows.extend(read_jsonl(path))
    return rows


def _index_best_label(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    output: dict[str, dict[str, Any]] = {}
    for row in rows:
        action_id = str(row.get("action_id") or "")
        if not action_id:
            continue
        current = output.get(action_id)
        if current is None or _label_priority(row) > _label_priority(current):
            output[action_id] = row
    return output


def _label_priority(row: dict[str, Any]) -> int:
    if _clean_sufficiency(row) is not None:
        return 3
    if _is_clean(row):
        return 2
    return 1


def _judge_count_summary(labels: dict[str, dict[str, Any]], targeted_ids: list[str]) -> dict[str, int]:
    targeted_labels = [labels[action_id] for action_id in targeted_ids if action_id in labels]
    return {
        "label_count": len(targeted_labels),
        "valid_json_count": sum(1 for row in targeted_labels if not row.get("invalid_json")),
        "ambiguous_count": sum(1 for row in targeted_labels if row.get("ambiguous")),
        "invalid_json_count": sum(1 for row in targeted_labels if row.get("invalid_json")),
        "error_count": sum(1 for row in targeted_labels if row.get("error")),
        "clean_sufficiency_count": sum(1 for row in targeted_labels if _clean_sufficiency(row) is not None),
    }


def _sufficiency_agreement(left: dict[str, dict[str, Any]], right: dict[str, dict[str, Any]], action_ids: list[str]) -> dict[str, Any]:
    pairs: list[tuple[bool, bool]] = []
    for action_id in action_ids:
        left_value = _clean_sufficiency(left.get(action_id, {}))
        right_value = _clean_sufficiency(right.get(action_id, {}))
        if left_value is not None and right_value is not None:
            pairs.append((left_value, right_value))
    agree = sum(1 for left_value, right_value in pairs if left_value == right_value)
    compared = len(pairs)
    return {
        "compared_count": compared,
        "agree_count": agree,
        "disagree_count": compared - agree,
        "agreement_rate": agree / compared if compared else None,
    }


def _score_correlation(
    left: dict[str, dict[str, Any]],
    right: dict[str, dict[str, Any]],
    action_ids: list[str],
    field: str,
) -> dict[str, Any]:
    xs: list[float] = []
    ys: list[float] = []
    for action_id in action_ids:
        left_row = left.get(action_id, {})
        right_row = right.get(action_id, {})
        if not _is_clean(left_row) or not _is_clean(right_row):
            continue
        x = _score(left_row.get(field))
        y = _score(right_row.get(field))
        if x is not None and y is not None:
            xs.append(x)
            ys.append(y)
    return {"count": len(xs), "pearson": _pearson(xs, ys)}


def _row_summary(action: dict[str, Any], labels_by_judge: dict[str, dict[str, Any] | None]) -> dict[str, Any]:
    action_id = str(action.get("action_id") or "")
    sufficiency = {
        judge: _clean_sufficiency(label or {})
        for judge, label in labels_by_judge.items()
    }
    clean_votes = {judge: value for judge, value in sufficiency.items() if value is not None}
    has_true = any(clean_votes.values())
    has_false = any(value is False for value in clean_votes.values())
    majority = _majority_vote(list(clean_votes.values()))
    mimo_value = sufficiency.get("mimo")
    row = {
        "action_id": action_id,
        "query_id": str(action.get("query_id") or ""),
        "question": str(action.get("question") or "")[:240],
        "selection_reason": action.get("selection_reason") or _first_selection_reason(action),
        "sufficiency_by_judge": sufficiency,
        "majority_sufficiency_vote": majority,
        "has_sufficiency_disagreement": has_true and has_false,
        "mimo_harsh": mimo_value is False and any(judge != "mimo" and value is True for judge, value in sufficiency.items()),
        "consensus_insufficient": len(clean_votes) >= 2 and all(value is False for value in clean_votes.values()),
    }
    return row


def _majority_vote(values: list[bool]) -> str:
    if not values:
        return "missing"
    true_count = sum(1 for value in values if value)
    false_count = len(values) - true_count
    if true_count > false_count:
        return "sufficient"
    if false_count > true_count:
        return "insufficient"
    return "tie"


def _majority_vote_counts(rows: list[dict[str, Any]]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for row in rows:
        key = str(row["majority_sufficiency_vote"])
        counts[key] = counts.get(key, 0) + 1
    return dict(sorted(counts.items()))


def _clean_sufficiency(row: dict[str, Any]) -> bool | None:
    if not _is_clean(row):
        return None
    value = row.get("sufficient")
    return value if isinstance(value, bool) else None


def _is_clean(row: dict[str, Any]) -> bool:
    return bool(row) and not row.get("invalid_json") and not row.get("ambiguous") and not row.get("error")


def _score(value: Any) -> float | None:
    if isinstance(value, bool) or value is None:
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if number < 0.0 or number > 1.0:
        return None
    return number


def _pearson(xs: list[float], ys: list[float]) -> float | None:
    if len(xs) < 2 or len(xs) != len(ys):
        return None
    x_mean = mean(xs)
    y_mean = mean(ys)
    numerator = sum((x - x_mean) * (y - y_mean) for x, y in zip(xs, ys, strict=True))
    x_denom = sum((x - x_mean) ** 2 for x in xs) ** 0.5
    y_denom = sum((y - y_mean) ** 2 for y in ys) ** 0.5
    if x_denom == 0.0 or y_denom == 0.0:
        return None
    return numerator / (x_denom * y_denom)


def _first_selection_reason(action: dict[str, Any]) -> str:
    audit = action.get("audit")
    if isinstance(audit, dict):
        reasons = audit.get("selection_reasons")
        if isinstance(reasons, list) and reasons:
            return str(reasons[0])
    return ""


def _append_case_rows(lines: list[str], rows: list[dict[str, Any]]) -> None:
    if not rows:
        lines.append("| N/A | N/A | N/A | N/A | N/A |")
        return
    for row in rows[:10]:
        suff = ", ".join(f"{judge}={value}" for judge, value in row["sufficiency_by_judge"].items())
        lines.append(
            f"| `{row['action_id']}` | `{row['query_id']}` | {row['majority_sufficiency_vote']} | "
            f"{suff} | {row['selection_reason']} |"
        )


def _fmt(value: Any) -> str:
    if value is None:
        return "N/A"
    if isinstance(value, float):
        return f"{value:.3f}"
    return str(value)


if __name__ == "__main__":
    raise SystemExit(main())
