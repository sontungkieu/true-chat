#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any


SCORE_FIELDS = (
    "quality_score",
    "overall_quality",
    "answer_correctness",
    "evidence_support",
    "faithfulness",
    "answer_relevancy",
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Validate and optionally merge sharded RLAIF answer labels.")
    parser.add_argument("--actions", type=Path, required=True, help="Reference rlaif_actions.jsonl.")
    parser.add_argument("--labels", type=Path, nargs="+", required=True, help="One or more answer-label JSONL shards.")
    parser.add_argument("--merged-output", type=Path, default=None, help="Optional deduped merged JSONL output.")
    parser.add_argument("--out-md", type=Path, default=None, help="Markdown validation summary output.")
    parser.add_argument("--out-json", type=Path, default=None, help="JSON validation summary output.")
    parser.add_argument("--sample-size", type=int, default=10, help="Number of ids/lines in diagnostic samples.")
    args = parser.parse_args(argv)

    summary = validate_answer_labels(
        actions_path=args.actions,
        label_paths=args.labels,
        merged_output=args.merged_output,
        sample_size=args.sample_size,
    )
    out_json = args.out_json or (args.merged_output or args.labels[0]).with_suffix(".validation.json")
    out_md = args.out_md or (args.merged_output or args.labels[0]).with_suffix(".validation.md")
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_md.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    out_md.write_text(render_markdown(summary), encoding="utf-8")
    print(
        json.dumps(
            {
                "out_json": str(out_json),
                "out_md": str(out_md),
                "merged_output": summary["merged_output"],
                "action_count": summary["action_count"],
                "merged_label_count": summary["merged_label_count"],
                "missing_action_count": summary["missing_action_count"],
                "unknown_action_count": summary["unknown_action_count"],
                "invalid_json_line_count": summary["invalid_json_line_count"],
                "duplicate_label_row_count": summary["duplicate_label_row_count"],
                "clean_usable_label_count": summary["clean_usable_label_count"],
            },
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
    )
    return 0


def validate_answer_labels(
    *,
    actions_path: Path,
    label_paths: list[Path],
    merged_output: Path | None = None,
    sample_size: int = 10,
) -> dict[str, Any]:
    if sample_size < 0:
        raise ValueError("--sample-size must be non-negative")
    actions = _read_jsonl_lenient(actions_path)
    action_order = [str(row["action_id"]) for row in actions["rows"] if row.get("action_id")]
    action_ids = set(action_order)
    if not action_ids:
        raise ValueError(f"No action_id values found in {actions_path}")
    if not label_paths:
        raise ValueError("At least one label path is required")

    file_summaries: list[dict[str, Any]] = []
    rows_by_action: dict[str, list[dict[str, Any]]] = defaultdict(list)
    file_indexes_by_action: dict[str, set[int]] = defaultdict(set)
    label_row_count = 0
    invalid_json_line_count = 0
    invalid_json_count = 0
    ambiguous_count = 0
    error_count = 0
    missing_reason_count = 0
    scored_label_count = 0
    unknown_ids: set[str] = set()
    invalid_line_samples: list[dict[str, Any]] = []

    for file_index, path in enumerate(label_paths):
        parsed = _read_jsonl_lenient(path, sample_size=sample_size)
        rows = parsed["rows"]
        invalid_json_line_count += parsed["invalid_json_line_count"]
        invalid_line_samples.extend(parsed["invalid_json_line_samples"])
        seen_in_file: set[str] = set()
        duplicate_in_file = 0
        unknown_in_file = 0
        clean_in_file = 0
        scored_in_file = 0
        for row in rows:
            label_row_count += 1
            action_id = str(row.get("action_id") or "")
            if not action_id:
                unknown_in_file += 1
                unknown_ids.add("")
                continue
            if action_id in seen_in_file:
                duplicate_in_file += 1
            seen_in_file.add(action_id)
            if action_id not in action_ids:
                unknown_in_file += 1
                unknown_ids.add(action_id)
            if row.get("invalid_json"):
                invalid_json_count += 1
            if row.get("ambiguous"):
                ambiguous_count += 1
            if row.get("error"):
                error_count += 1
            if row.get("missing_reason"):
                missing_reason_count += 1
            if _has_any_score(row):
                scored_label_count += 1
                scored_in_file += 1
            if _is_clean_usable(row):
                clean_in_file += 1
            rows_by_action[action_id].append(row)
            file_indexes_by_action[action_id].add(file_index)
        file_summaries.append(
            {
                "path": str(path),
                "row_count": len(rows),
                "invalid_json_line_count": parsed["invalid_json_line_count"],
                "unique_action_count": len(seen_in_file),
                "duplicate_within_file_count": duplicate_in_file,
                "unknown_action_count": unknown_in_file,
                "scored_label_count": scored_in_file,
                "clean_usable_label_count": clean_in_file,
            }
        )

    labeled_known_ids = set(rows_by_action) & action_ids
    duplicate_action_ids = sorted(action_id for action_id, rows in rows_by_action.items() if action_id and len(rows) > 1)
    duplicate_conflict_ids = sorted(action_id for action_id in duplicate_action_ids if _has_conflicting_rows(rows_by_action[action_id]))
    shard_overlap_ids = sorted(
        action_id for action_id, file_indexes in file_indexes_by_action.items() if action_id and len(file_indexes) > 1
    )
    missing_ids = sorted(action_ids - labeled_known_ids)
    merged_by_action = {
        action_id: _choose_best_label(rows_by_action[action_id])
        for action_id in labeled_known_ids
    }
    merged_rows = [merged_by_action[action_id] for action_id in action_order if action_id in merged_by_action]
    clean_usable_count = sum(1 for row in merged_rows if _is_clean_usable(row))
    merged_scored_count = sum(1 for row in merged_rows if _has_any_score(row))

    if merged_output is not None:
        merged_output.parent.mkdir(parents=True, exist_ok=True)
        _write_jsonl(merged_output, merged_rows)

    return {
        "schema_version": "rlaif-answer-label-validation-v1",
        "actions_path": str(actions_path),
        "label_paths": [str(path) for path in label_paths],
        "merged_output": str(merged_output) if merged_output is not None else None,
        "action_count": len(action_ids),
        "label_file_count": len(label_paths),
        "label_row_count": label_row_count,
        "unique_label_action_count": len({action_id for action_id in rows_by_action if action_id}),
        "known_label_action_count": len(labeled_known_ids),
        "merged_label_count": len(merged_rows),
        "missing_action_count": len(missing_ids),
        "unknown_action_count": len({action_id for action_id in unknown_ids if action_id}),
        "blank_action_id_count": 1 if "" in unknown_ids else 0,
        "duplicate_label_row_count": sum(max(0, len(rows) - 1) for action_id, rows in rows_by_action.items() if action_id),
        "duplicate_action_id_count": len(duplicate_action_ids),
        "duplicate_conflict_count": len(duplicate_conflict_ids),
        "shard_overlap_action_count": len(shard_overlap_ids),
        "invalid_json_line_count": invalid_json_line_count,
        "invalid_json_count": invalid_json_count,
        "ambiguous_count": ambiguous_count,
        "error_count": error_count,
        "missing_reason_count": missing_reason_count,
        "scored_label_count": scored_label_count,
        "merged_scored_label_count": merged_scored_count,
        "clean_usable_label_count": clean_usable_count,
        "file_summaries": file_summaries,
        "samples": {
            "missing_action_ids": missing_ids[:sample_size],
            "unknown_action_ids": sorted(action_id for action_id in unknown_ids if action_id)[:sample_size],
            "duplicate_action_ids": duplicate_action_ids[:sample_size],
            "duplicate_conflict_action_ids": duplicate_conflict_ids[:sample_size],
            "shard_overlap_action_ids": shard_overlap_ids[:sample_size],
            "invalid_json_lines": invalid_line_samples[:sample_size],
        },
    }


def render_markdown(summary: dict[str, Any]) -> str:
    lines = [
        "# RLAIF Answer Label Validation",
        "",
        f"- Actions: `{summary['actions_path']}`",
        f"- Merged output: `{summary['merged_output'] or 'N/A'}`",
        "",
        "## Counts",
        "",
        "| Metric | Value |",
        "| --- | ---: |",
    ]
    for key in (
        "action_count",
        "label_file_count",
        "label_row_count",
        "unique_label_action_count",
        "known_label_action_count",
        "merged_label_count",
        "missing_action_count",
        "unknown_action_count",
        "blank_action_id_count",
        "duplicate_label_row_count",
        "duplicate_action_id_count",
        "duplicate_conflict_count",
        "shard_overlap_action_count",
        "invalid_json_line_count",
        "invalid_json_count",
        "ambiguous_count",
        "error_count",
        "missing_reason_count",
        "scored_label_count",
        "merged_scored_label_count",
        "clean_usable_label_count",
    ):
        lines.append(f"| {key.replace('_', ' ')} | {summary[key]} |")

    lines.extend(
        [
            "",
            "## Files",
            "",
            "| File | Rows | Invalid lines | Unique actions | Duplicates | Unknown | Scored | Clean usable |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for item in summary["file_summaries"]:
        lines.append(
            f"| `{item['path']}` | {item['row_count']} | {item['invalid_json_line_count']} | "
            f"{item['unique_action_count']} | {item['duplicate_within_file_count']} | "
            f"{item['unknown_action_count']} | {item['scored_label_count']} | "
            f"{item['clean_usable_label_count']} |"
        )

    lines.extend(["", "## Diagnostic Samples", ""])
    for key, values in summary["samples"].items():
        if key == "invalid_json_lines":
            if not values:
                rendered = "N/A"
            else:
                rendered = "; ".join(
                    f"`{item['path']}:{item['line_no']}` {item['error']}" for item in values
                )
        else:
            rendered = ", ".join(f"`{value}`" for value in values) if values else "N/A"
        lines.append(f"- {key.replace('_', ' ')}: {rendered}")

    lines.extend(
        [
            "",
            "Merge rule: corrupted JSONL lines are skipped and counted. For duplicate action ids, the merged output keeps clean usable labels first, then valid scored labels, then non-ambiguous/non-invalid rows, then the first parseable row. Unknown action ids are excluded from merged output. Missing, ambiguous, invalid, or errored labels are never converted to score zero.",
        ]
    )
    return "\n".join(lines).rstrip() + "\n"


def _read_jsonl_lenient(path: Path, *, sample_size: int = 10) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    invalid_samples: list[dict[str, Any]] = []
    invalid_count = 0
    with path.open("r", encoding="utf-8", errors="replace") as handle:
        for line_no, line in enumerate(handle, start=1):
            stripped = line.strip("\ufeff \t\r\n\x00")
            if not stripped:
                continue
            try:
                row = json.loads(stripped)
            except json.JSONDecodeError as exc:
                invalid_count += 1
                if len(invalid_samples) < sample_size:
                    invalid_samples.append(
                        {
                            "path": str(path),
                            "line_no": line_no,
                            "error": exc.msg,
                            "preview": stripped[:160],
                        }
                    )
                continue
            if not isinstance(row, dict):
                invalid_count += 1
                if len(invalid_samples) < sample_size:
                    invalid_samples.append(
                        {
                            "path": str(path),
                            "line_no": line_no,
                            "error": "expected JSON object",
                            "preview": stripped[:160],
                        }
                    )
                continue
            rows.append(row)
    return {
        "rows": rows,
        "invalid_json_line_count": invalid_count,
        "invalid_json_line_samples": invalid_samples,
    }


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def _choose_best_label(rows: list[dict[str, Any]]) -> dict[str, Any]:
    return max(enumerate(rows), key=lambda item: (_label_priority(item[1]), -item[0]))[1]


def _label_priority(row: dict[str, Any]) -> int:
    if _is_clean_usable(row):
        return 5
    if _is_valid_scored(row):
        return 4
    if not row.get("ambiguous") and not row.get("invalid_json") and not row.get("error"):
        return 3
    if _has_any_score(row):
        return 2
    return 1


def _is_clean_usable(row: dict[str, Any]) -> bool:
    if row.get("ambiguous") or row.get("invalid_json") or row.get("error") or row.get("missing_reason"):
        return False
    return _has_any_score(row)


def _is_valid_scored(row: dict[str, Any]) -> bool:
    if row.get("invalid_json") or row.get("error") or row.get("missing_reason"):
        return False
    return _has_any_score(row)


def _has_any_score(row: dict[str, Any]) -> bool:
    return any(_score_or_none(row.get(field)) is not None for field in SCORE_FIELDS)


def _score_or_none(value: Any) -> float | None:
    if value is None:
        return None
    try:
        score = float(value)
    except (TypeError, ValueError):
        return None
    if score < 0.0 or score > 1.0:
        return None
    return score


def _has_conflicting_rows(rows: list[dict[str, Any]]) -> bool:
    encoded = {json.dumps(row, ensure_ascii=False, sort_keys=True) for row in rows}
    return len(encoded) > 1


if __name__ == "__main__":
    raise SystemExit(main())
