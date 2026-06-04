from __future__ import annotations

import json
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean
from typing import Any, Iterable

from rag_bench.io import write_jsonl
from rag_bench.retrieval_context_actions import action_from_budgetrag_row
from rag_bench.rlaif_schema import RlaifAnswerFeedback


@dataclass(frozen=True)
class RlaifBuildConfig:
    inputs: tuple[Path, ...]
    output_dir: Path | None = None
    run_name: str | None = None


def build_rlaif_dataset(config: RlaifBuildConfig) -> dict[str, Any]:
    if not config.inputs:
        raise ValueError("At least one input path is required")

    query_result_paths = discover_query_result_paths(config.inputs)
    if not query_result_paths:
        raise ValueError("No query_results.jsonl files found in input paths")

    output_dir = _resolve_output_dir(config.output_dir, config.run_name)
    actions: list[dict[str, Any]] = []
    feedback_rows: list[dict[str, Any]] = []
    invalid_rows: list[dict[str, Any]] = []

    for path in query_result_paths:
        for line_no, row, row_error in _iter_jsonl_rows(path):
            if row_error is not None:
                invalid_rows.append(
                    {
                        "path": str(path),
                        "line": line_no,
                        "query_id": None,
                        "error": row_error,
                    }
                )
                continue
            try:
                action = action_from_budgetrag_row(row)
                action_record = action.to_dict()
                action_record.update(_observation_payload(row))
                actions.append(action_record)

                feedback = answer_feedback_from_budgetrag_row(row, action_id=action.action_id)
                feedback_rows.append(feedback.to_dict())
            except Exception as exc:  # noqa: BLE001 - builder should keep coverage over partial runs.
                invalid_rows.append(
                    {
                        "path": str(path),
                        "line": line_no,
                        "query_id": row.get("query_id") if isinstance(row, dict) else None,
                        "error": str(exc),
                    }
                )

    write_jsonl(output_dir / "rlaif_actions.jsonl", actions)
    write_jsonl(output_dir / "rlaif_feedback.jsonl", feedback_rows)
    summary = _build_summary(
        output_dir=output_dir,
        input_paths=query_result_paths,
        actions=actions,
        feedback_rows=feedback_rows,
        invalid_rows=invalid_rows,
    )
    (output_dir / "rlaif_feedback_summary.md").write_text(_render_summary_markdown(summary), encoding="utf-8")
    return summary


def discover_query_result_paths(inputs: Iterable[Path]) -> list[Path]:
    paths: list[Path] = []
    for input_path in inputs:
        path = Path(input_path)
        if path.is_file():
            if path.name != "query_results.jsonl":
                raise ValueError(f"Expected query_results.jsonl file, got: {path}")
            paths.append(path)
        elif path.is_dir():
            paths.extend(sorted(path.rglob("query_results.jsonl")))
        else:
            raise ValueError(f"Input path does not exist: {path}")
    return sorted(dict.fromkeys(paths))


def answer_feedback_from_budgetrag_row(row: dict[str, Any], *, action_id: str) -> RlaifAnswerFeedback:
    query_id = _required_text(row.get("query_id"), "query_id")
    error = row.get("error")
    if error:
        return RlaifAnswerFeedback(
            action_id=action_id,
            query_id=query_id,
            provenance="missing",
            ambiguous=True,
            missing_reason="generation_error",
            metadata={
                "invalid": True,
                "error": str(error),
                "error_status_code": row.get("error_status_code"),
            },
        )

    if row.get("generation_skipped") is True:
        return RlaifAnswerFeedback(
            action_id=action_id,
            query_id=query_id,
            provenance="missing",
            missing_reason="generation_skipped",
            metadata={"invalid": False},
        )

    answer = row.get("answer")
    if not isinstance(answer, str) or not answer.strip():
        return RlaifAnswerFeedback(
            action_id=action_id,
            query_id=query_id,
            provenance="missing",
            ambiguous=True,
            missing_reason="missing_answer",
            metadata={"invalid": True},
        )

    exact_match = _score_or_none(row.get("exact_match"))
    token_f1 = _score_or_none(row.get("token_f1"))
    if exact_match is not None or token_f1 is not None:
        quality_score = token_f1 if token_f1 is not None else exact_match
        return RlaifAnswerFeedback(
            action_id=action_id,
            query_id=query_id,
            provenance="gold",
            quality_score=quality_score,
            exact_match=exact_match,
            token_f1=token_f1,
        )

    ragas_feedback = _feedback_from_ragas(row, action_id=action_id, query_id=query_id)
    if ragas_feedback is not None:
        return ragas_feedback

    judge_feedback = _feedback_from_judge(row, action_id=action_id, query_id=query_id)
    if judge_feedback is not None:
        return judge_feedback

    return RlaifAnswerFeedback(
        action_id=action_id,
        query_id=query_id,
        provenance="missing",
        missing_reason="no_feedback_labels",
        metadata={"invalid": False},
    )


def _feedback_from_ragas(row: dict[str, Any], *, action_id: str, query_id: str) -> RlaifAnswerFeedback | None:
    source = _first_dict(row.get("ragas"), row.get("ragas_metrics"), row.get("ragas_scores"))
    if not source:
        return None
    answer_relevancy = _score_or_none(_first_value(source, "answer_relevancy", "response_relevancy"))
    faithfulness = _score_or_none(source.get("faithfulness"))
    answer_correctness = _score_or_none(source.get("answer_correctness"))
    quality_values = [value for value in (answer_correctness, answer_relevancy, faithfulness) if value is not None]
    if not quality_values:
        return None
    return RlaifAnswerFeedback(
        action_id=action_id,
        query_id=query_id,
        provenance="ragas",
        quality_score=mean(quality_values),
        answer_relevancy=answer_relevancy,
        faithfulness=faithfulness,
        answer_correctness=answer_correctness,
        rationale=_optional_text(source.get("rationale")),
        metadata={"raw_metric_keys": sorted(source)},
    )


def _feedback_from_judge(row: dict[str, Any], *, action_id: str, query_id: str) -> RlaifAnswerFeedback | None:
    source = _first_dict(row.get("mimo_judge"), row.get("answer_judge"), row.get("judge"))
    if not source:
        return None
    quality_score = _score_or_none(_first_value(source, "quality_score", "score", "answer_quality"))
    answer_relevancy = _score_or_none(_first_value(source, "answer_relevancy", "relevancy"))
    faithfulness = _score_or_none(source.get("faithfulness"))
    answer_correctness = _score_or_none(source.get("answer_correctness"))
    quality_values = [value for value in (quality_score, answer_correctness, answer_relevancy, faithfulness) if value is not None]
    if not quality_values:
        return None
    provider = _optional_text(_first_value(source, "judge_provider", "provider")) or "mimo"
    return RlaifAnswerFeedback(
        action_id=action_id,
        query_id=query_id,
        provenance="mimo_judge" if provider == "mimo" else "heuristic",
        quality_score=quality_score if quality_score is not None else mean(quality_values),
        answer_relevancy=answer_relevancy,
        faithfulness=faithfulness,
        answer_correctness=answer_correctness,
        unsupported_claim_penalty=_score_or_none(source.get("unsupported_claim_penalty")),
        ambiguous=bool(source.get("ambiguous", False)),
        judge_provider=provider,
        judge_model=_optional_text(_first_value(source, "judge_model", "model")),
        rationale=_optional_text(source.get("rationale")),
        metadata={"raw_metric_keys": sorted(source)},
    )


def _observation_payload(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "retriever": row.get("retriever"),
        "answer": row.get("answer") if isinstance(row.get("answer"), str) else "",
        "retrieved": row.get("retrieved") if isinstance(row.get("retrieved"), list) else [],
        "context_metrics": _dict_or_empty(row.get("context_budget")),
        "retrieval_metrics": _dict_or_empty(row.get("retrieval_metrics")),
        "retrieval_metadata": _dict_or_empty(row.get("retrieval_metadata")),
        "kv_estimate": _dict_or_empty(row.get("kv_estimate")),
        "latency": {
            "answer_latency_s": row.get("answer_latency_s"),
            "total_latency_s": row.get("total_latency_s"),
            "scheduled_wait_s": row.get("scheduled_wait_s"),
        },
        "token_usage": {
            "estimated_tokens": row.get("estimated_tokens"),
            "prompt_tokens": row.get("prompt_tokens"),
            "completion_tokens": row.get("completion_tokens"),
            "total_tokens": row.get("total_tokens"),
            "estimated_prompt_tokens_after_budget": row.get("estimated_prompt_tokens_after_budget"),
            "estimated_prompt_tokens_saved_by_budget": row.get("estimated_prompt_tokens_saved_by_budget"),
        },
        "generation": {
            "skipped": row.get("generation_skipped"),
            "error": row.get("error"),
            "error_status_code": row.get("error_status_code"),
            "retry_count": row.get("retry_count"),
            "rate_limited": row.get("rate_limited"),
        },
    }


def _build_summary(
    *,
    output_dir: Path,
    input_paths: list[Path],
    actions: list[dict[str, Any]],
    feedback_rows: list[dict[str, Any]],
    invalid_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    provenance_counts = Counter(row.get("provenance", "unknown") for row in feedback_rows)
    missing_reason_counts = Counter(
        row.get("missing_reason")
        for row in feedback_rows
        if row.get("missing_reason")
    )
    generation_error_count = sum(
        1
        for row in feedback_rows
        if row.get("missing_reason") == "generation_error"
    )
    return {
        "output_dir": str(output_dir),
        "input_paths": [str(path) for path in input_paths],
        "input_file_count": len(input_paths),
        "action_count": len(actions),
        "unique_action_count": len({row.get("action_id") for row in actions}),
        "feedback_count": len(feedback_rows),
        "feedback_provenance_counts": dict(provenance_counts),
        "missing_reason_counts": dict(missing_reason_counts),
        "generation_error_count": generation_error_count,
        "invalid_row_count": len(invalid_rows),
        "invalid_rows": invalid_rows,
        "created_at": datetime.now(timezone.utc).isoformat(),
    }


def _render_summary_markdown(summary: dict[str, Any]) -> str:
    lines = [
        "# RLAIF Feedback Summary",
        "",
        f"- Output dir: `{summary['output_dir']}`",
        f"- Input files: {summary['input_file_count']}",
        f"- Actions: {summary['action_count']}",
        f"- Unique actions: {summary['unique_action_count']}",
        f"- Feedback rows: {summary['feedback_count']}",
        f"- Invalid rows: {summary['invalid_row_count']}",
        f"- Generation errors: {summary['generation_error_count']}",
        "",
        "## Feedback Provenance",
        "",
        "| Provenance | Count |",
        "| --- | ---: |",
    ]
    for key, value in sorted(summary["feedback_provenance_counts"].items()):
        lines.append(f"| `{key}` | {value} |")
    lines.extend(["", "## Missing Reasons", "", "| Reason | Count |", "| --- | ---: |"])
    for key, value in sorted(summary["missing_reason_counts"].items()):
        lines.append(f"| `{key}` | {value} |")
    if not summary["missing_reason_counts"]:
        lines.append("| N/A | 0 |")
    lines.extend(["", "## Inputs", ""])
    lines.extend(f"- `{path}`" for path in summary["input_paths"])
    if summary["invalid_rows"]:
        lines.extend(["", "## Invalid Rows", "", "| File | Line | Query | Error |", "| --- | ---: | --- | --- |"])
        for row in summary["invalid_rows"]:
            lines.append(
                f"| `{row['path']}` | {row['line']} | `{row.get('query_id')}` | {row['error']} |"
            )
    return "\n".join(lines) + "\n"


def _resolve_output_dir(output_dir: Path | None, run_name: str | None) -> Path:
    if output_dir is not None:
        return output_dir
    safe_name = run_name or datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return Path("benchmark_results") / "rlaif" / safe_name


def _iter_jsonl_rows(path: Path) -> Iterable[tuple[int, dict[str, Any], str | None]]:
    with path.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            try:
                row = json.loads(stripped)
            except json.JSONDecodeError as exc:
                yield line_no, {}, f"Invalid JSON: {exc.msg}"
                continue
            if not isinstance(row, dict):
                yield line_no, {}, "Expected object row"
                continue
            yield line_no, row, None


def _required_text(value: Any, field_name: str) -> str:
    text = _optional_text(value)
    if text is None:
        raise ValueError(f"{field_name} must be a non-empty string")
    return text


def _optional_text(value: Any) -> str | None:
    if isinstance(value, str) and value.strip():
        return value
    return None


def _score_or_none(value: Any) -> float | None:
    if value is None or isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        score = float(value)
    elif isinstance(value, str) and value.strip():
        try:
            score = float(value)
        except ValueError:
            return None
    else:
        return None
    if score < 0 or score > 1:
        return None
    return score


def _first_dict(*values: Any) -> dict[str, Any]:
    for value in values:
        if isinstance(value, dict):
            return value
    return {}


def _dict_or_empty(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _first_value(source: dict[str, Any], *keys: str) -> Any:
    for key in keys:
        if key in source:
            return source[key]
    return None
