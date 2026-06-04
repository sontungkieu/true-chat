from __future__ import annotations

import json
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from rag_bench.groq_client import GenerationResult
from rag_bench.io import write_json
from rag_bench.rlaif_label_answers import (
    AnswerJudgeClient,
    DEFAULT_MAX_COMPLETION_TOKENS,
    _build_judge_client,
    _completed_action_ids,
    _generation_metadata,
    _json_repair_messages,
    _optional_text,
    _parse_judge_json,
    _read_jsonl,
    _score_or_none,
)


LABEL_SCHEMA_VERSION = "rlaif-context-label-v1"
PROMPT_VERSION = "rlaif-context-judge-v1"
CONTEXT_SCORE_FIELDS = (
    "minimality_score",
    "evidence_support_score",
    "context_quality_score",
)
CHUNK_LIST_FIELDS = (
    "selected_chunk_ids",
    "redundant_chunk_ids",
    "irrelevant_chunk_ids",
)


@dataclass(frozen=True)
class RlaifContextLabelConfig:
    actions_path: Path
    output_path: Path
    judge_provider: str = "mimo"
    judge_model: str = "mimo-v2.5-pro"
    dry_run: bool = False
    resume: bool = False
    limit: int | None = None
    max_errors: int = 3
    sleep_seconds: float = 0.0
    json_retries: int = 1
    max_context_chars: int = 12_000
    max_completion_tokens: int = DEFAULT_MAX_COMPLETION_TOKENS
    temperature: float = 0.0
    groq_keys_path: Path = Path(".secrets/groq_key.env")
    env_file: Path = Path(".secrets/.env")
    api_key_var: str | None = None
    base_url: str | None = None
    timeout_s: float = 60.0
    key_tpm: int = 0
    key_rpm: int = 0
    progress_every: int = 1
    client: AnswerJudgeClient | None = None


def label_rlaif_contexts(config: RlaifContextLabelConfig) -> dict[str, Any]:
    if not config.actions_path.is_file():
        raise ValueError(f"Actions path does not exist: {config.actions_path}")
    if config.limit is not None and config.limit < 0:
        raise ValueError("--limit must be non-negative")
    if config.max_errors < 0:
        raise ValueError("--max-errors must be non-negative")
    if config.sleep_seconds < 0:
        raise ValueError("--sleep-seconds must be non-negative")
    if config.json_retries < 0:
        raise ValueError("--json-retries must be non-negative")
    if config.max_context_chars <= 0:
        raise ValueError("--max-context-chars must be positive")
    if config.max_completion_tokens <= 0:
        raise ValueError("--max-completion-tokens must be positive")
    if config.progress_every < 0:
        raise ValueError("--progress-every must be non-negative")

    actions = _read_jsonl(config.actions_path)
    completed = _completed_action_ids(config.output_path) if config.resume else set()
    processed = 0
    skipped_resume = 0
    skipped_limit = 0
    error_count = 0
    invalid_json_count = 0
    missing_input_count = 0
    ambiguous_count = 0
    stopped_early = False
    stop_reason = None

    client = None if config.dry_run else (config.client or _build_judge_client(config))
    config.output_path.parent.mkdir(parents=True, exist_ok=True)
    with config.output_path.open("a" if config.resume else "w", encoding="utf-8") as handle:
        for action in actions:
            action_id = str(action.get("action_id") or "")
            if not action_id:
                continue
            if action_id in completed:
                skipped_resume += 1
                continue
            if config.limit is not None and processed >= config.limit:
                skipped_limit += 1
                continue

            label = _label_one_action(action, config=config, client=client)
            processed += 1
            if label.get("invalid_json"):
                invalid_json_count += 1
            if label.get("missing_reason"):
                missing_input_count += 1
            if label.get("ambiguous"):
                ambiguous_count += 1
            if label.get("error"):
                error_count += 1
                if config.max_errors and error_count >= config.max_errors:
                    stopped_early = True
                    stop_reason = "max_errors"
            handle.write(json.dumps(label, ensure_ascii=False, sort_keys=True))
            handle.write("\n")
            handle.flush()

            _print_progress(
                config,
                action_index=processed + skipped_resume,
                action_count=len(actions),
                processed=processed,
                skipped_resume=skipped_resume,
                ambiguous_count=ambiguous_count,
                invalid_json_count=invalid_json_count,
                error_count=error_count,
                label=label,
            )

            if stopped_early:
                break
            if config.sleep_seconds > 0 and not config.dry_run:
                time.sleep(config.sleep_seconds)

    summary = {
        "output_path": str(config.output_path),
        "action_count": len(actions),
        "processed_count": processed,
        "skipped_resume_count": skipped_resume,
        "skipped_limit_count": skipped_limit,
        "ambiguous_count": ambiguous_count,
        "invalid_json_count": invalid_json_count,
        "missing_input_count": missing_input_count,
        "error_count": error_count,
        "stopped_early": stopped_early,
        "stop_reason": stop_reason,
        "dry_run": config.dry_run,
        "judge_provider": config.judge_provider,
        "judge_model": config.judge_model,
        "summary_path": str(config.output_path.with_suffix(".summary.json")),
    }
    write_json(config.output_path.with_suffix(".summary.json"), summary)
    config.output_path.with_suffix(".summary.md").write_text(_render_summary(summary), encoding="utf-8")
    return summary


def _label_one_action(
    action: dict[str, Any],
    *,
    config: RlaifContextLabelConfig,
    client: AnswerJudgeClient | None,
) -> dict[str, Any]:
    context, available_chunk_ids = _format_context(action, max_context_chars=config.max_context_chars)
    base = _base_label(action, config=config, available_chunk_ids=available_chunk_ids)
    if not context.strip():
        return _ambiguous_label(base, missing_reason="missing_context", rationale="No retrieved context was available.")
    if config.dry_run:
        return _ambiguous_label(
            base,
            rationale="Dry-run placeholder; no judge call was made.",
            metadata={"dry_run": True},
        )
    if client is None:
        return _ambiguous_label(base, missing_reason="missing_judge_client", rationale="No judge client was configured.")

    messages = _judge_messages(action, context=context, available_chunk_ids=available_chunk_ids)
    last_raw = ""
    last_generation: GenerationResult | None = None
    for attempt in range(config.json_retries + 1):
        result = client.generate(
            messages,
            model=config.judge_model,
            temperature=config.temperature,
            max_completion_tokens=config.max_completion_tokens,
        )
        if result.error:
            return _ambiguous_label(
                base,
                error=result.error,
                rationale="Judge request failed.",
                metadata=_generation_metadata(result, attempt=attempt),
            )
        last_raw = result.answer
        last_generation = result
        parsed = _parse_judge_json(last_raw)
        if parsed is not None:
            return _label_from_judge(
                base,
                parsed,
                raw_response=last_raw,
                generation=result,
                attempt=attempt,
                available_chunk_ids=available_chunk_ids,
            )
        messages = _json_repair_messages(messages, last_raw=last_raw)
    return _ambiguous_label(
        base,
        invalid_json=True,
        rationale="Judge returned invalid JSON.",
        metadata={
            "raw_response_preview": last_raw[:1000],
            "json_retry_count": config.json_retries,
            **(_generation_metadata(last_generation, attempt=config.json_retries) if last_generation else {}),
        },
    )


def _base_label(
    action: dict[str, Any],
    *,
    config: RlaifContextLabelConfig,
    available_chunk_ids: list[str],
) -> dict[str, Any]:
    return {
        "schema_version": LABEL_SCHEMA_VERSION,
        "action_id": str(action.get("action_id") or ""),
        "query_id": str(action.get("query_id") or ""),
        "benchmark": action.get("benchmark"),
        "provenance": "ai_judge",
        "judge_provider": config.judge_provider,
        "judge_model": config.judge_model,
        "judge_version": PROMPT_VERSION,
        "prompt_version": PROMPT_VERSION,
        "available_chunk_ids": available_chunk_ids,
        "ambiguous": False,
        "invalid_json": False,
        "missing_reason": None,
        "error": None,
    }


def _ambiguous_label(
    base: dict[str, Any],
    *,
    missing_reason: str | None = None,
    invalid_json: bool = False,
    error: str | None = None,
    rationale: str,
    metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    label = {
        **base,
        "sufficient": None,
        "missing_evidence": None,
        **{field: [] for field in CHUNK_LIST_FIELDS},
        **{field: None for field in CONTEXT_SCORE_FIELDS},
        "short_rationale": rationale,
        "rationale": rationale,
        "ambiguous": True,
        "invalid_json": invalid_json,
        "missing_reason": missing_reason,
        "error": error,
        "metadata": metadata or {},
    }
    if missing_reason:
        label["provenance"] = "missing"
    return label


def _label_from_judge(
    base: dict[str, Any],
    parsed: dict[str, Any],
    *,
    raw_response: str,
    generation: GenerationResult,
    attempt: int,
    available_chunk_ids: list[str],
) -> dict[str, Any]:
    scores = {field: _score_or_none(parsed.get(field)) for field in CONTEXT_SCORE_FIELDS}
    selected, dropped_selected = _known_chunk_ids(parsed.get("selected_chunk_ids"), available_chunk_ids)
    redundant, dropped_redundant = _known_chunk_ids(parsed.get("redundant_chunk_ids"), available_chunk_ids)
    irrelevant, dropped_irrelevant = _known_chunk_ids(parsed.get("irrelevant_chunk_ids"), available_chunk_ids)
    dropped_unknown = {
        "selected_chunk_ids": dropped_selected,
        "redundant_chunk_ids": dropped_redundant,
        "irrelevant_chunk_ids": dropped_irrelevant,
    }
    scored_values = [value for value in scores.values() if value is not None]
    ambiguous = bool(parsed.get("ambiguous", False)) or not scored_values
    return {
        **base,
        "sufficient": _bool_or_none(parsed.get("sufficient")),
        "missing_evidence": _bool_or_none(parsed.get("missing_evidence")),
        "selected_chunk_ids": selected,
        "redundant_chunk_ids": redundant,
        "irrelevant_chunk_ids": irrelevant,
        **scores,
        "short_rationale": _optional_text(parsed.get("short_rationale")),
        "rationale": _optional_text(parsed.get("short_rationale")),
        "ambiguous": ambiguous,
        "invalid_json": False,
        "metadata": {
            "raw_response_preview": raw_response[:1000],
            "json_retry_count": attempt,
            "dropped_unknown_chunk_ids": dropped_unknown,
            **_generation_metadata(generation, attempt=attempt),
        },
    }


def _judge_messages(
    action: dict[str, Any],
    *,
    context: str,
    available_chunk_ids: list[str],
) -> list[dict[str, str]]:
    question = str(action.get("question") or "").strip()
    answer = str(action.get("answer") or "").strip()
    answer_section = f"\nLogged answer, if any:\n{answer}\n" if answer else ""
    return [
        {
            "role": "system",
            "content": (
                "You are an offline RLAIF context judge for a RAG benchmark. "
                "Judge only from the provided question, optional logged answer, and retrieved context chunks. "
                "Do not browse and do not use external knowledge. Do not reveal reasoning. "
                "Return only one compact valid JSON object."
            ),
        },
        {
            "role": "user",
            "content": (
                "Evaluate whether the retrieved chunks provide sufficient, minimal evidence for the question.\n\n"
                f"Question:\n{question}\n"
                f"{answer_section}\n"
                f"Available chunk ids:\n{json.dumps(available_chunk_ids, ensure_ascii=False)}\n\n"
                f"Retrieved chunks:\n{context}\n\n"
                "Return exactly one minified JSON object. It must start with `{` and end with `}`. "
                "Use only these keys: sufficient, selected_chunk_ids, redundant_chunk_ids, "
                "irrelevant_chunk_ids, missing_evidence, minimality_score, evidence_support_score, "
                "context_quality_score, ambiguous, short_rationale. "
                "Chunk id arrays must contain only ids from Available chunk ids. "
                "Scores are numbers from 0 to 1 or null. "
                "minimality_score=1 means no obvious redundant context; "
                "evidence_support_score=1 means the selected chunks strongly support the answer/question; "
                "context_quality_score is the overall context quality. "
                "short_rationale must be one short sentence. No markdown. No extra keys."
            ),
        },
    ]


def _format_context(action: dict[str, Any], *, max_context_chars: int) -> tuple[str, list[str]]:
    retrieved = action.get("retrieved")
    if not isinstance(retrieved, list):
        return "", []
    parts: list[str] = []
    available_chunk_ids: list[str] = []
    seen: dict[str, int] = {}
    remaining = max_context_chars
    for index, item in enumerate(retrieved, start=1):
        if remaining <= 0:
            break
        if not isinstance(item, dict):
            continue
        text = str(item.get("text") or "").strip()
        if not text:
            continue
        chunk_id = _stable_chunk_id(item, index=index, seen=seen)
        title = str(item.get("title") or "")
        rank = item.get("rank")
        header = f"[chunk_id {chunk_id} rank {rank}] {title}".strip()
        chunk = f"{header}\n{text}\n"
        clipped = chunk[:remaining]
        if not clipped.strip():
            break
        parts.append(clipped)
        available_chunk_ids.append(chunk_id)
        remaining -= len(clipped)
    return "\n".join(parts), available_chunk_ids


def _stable_chunk_id(item: dict[str, Any], *, index: int, seen: dict[str, int]) -> str:
    raw = str(item.get("doc_id") or item.get("id") or "").strip()
    if not raw:
        rank = item.get("rank")
        raw = f"rank:{rank}" if rank is not None else f"chunk:{index}"
    count = seen.get(raw, 0) + 1
    seen[raw] = count
    if count == 1:
        return raw
    return f"{raw}#{count}"


def _known_chunk_ids(value: Any, available_chunk_ids: list[str]) -> tuple[list[str], list[str]]:
    if not isinstance(value, list):
        return [], []
    allowed = set(available_chunk_ids)
    result: list[str] = []
    dropped: list[str] = []
    seen: set[str] = set()
    for item in value:
        chunk_id = str(item).strip()
        if not chunk_id:
            continue
        if chunk_id not in allowed:
            dropped.append(chunk_id)
            continue
        if chunk_id in seen:
            continue
        seen.add(chunk_id)
        result.append(chunk_id)
    return result, dropped


def _bool_or_none(value: Any) -> bool | None:
    if isinstance(value, bool):
        return value
    if value is None:
        return None
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in {"true", "yes", "1"}:
            return True
        if lowered in {"false", "no", "0"}:
            return False
    return None


def _print_progress(
    config: RlaifContextLabelConfig,
    *,
    action_index: int,
    action_count: int,
    processed: int,
    skipped_resume: int,
    ambiguous_count: int,
    invalid_json_count: int,
    error_count: int,
    label: dict[str, Any],
) -> None:
    if config.progress_every <= 0:
        return
    if processed % config.progress_every != 0:
        return
    status = "ok"
    if label.get("error"):
        status = "error"
    elif label.get("invalid_json"):
        status = "invalid_json"
    elif label.get("missing_reason"):
        status = f"missing:{label['missing_reason']}"
    elif label.get("ambiguous"):
        status = "ambiguous"
    print(
        (
            f"[rlaif-label-contexts] {action_index}/{action_count} "
            f"processed={processed} skipped_resume={skipped_resume} "
            f"ambiguous={ambiguous_count} invalid_json={invalid_json_count} "
            f"errors={error_count} status={status} action_id={label.get('action_id')}"
        ),
        file=sys.stderr,
        flush=True,
    )


def _render_summary(summary: dict[str, Any]) -> str:
    return "\n".join(
        [
            "# RLAIF Context Label Summary",
            "",
            f"- Output: `{summary['output_path']}`",
            f"- Actions read: {summary['action_count']}",
            f"- Processed: {summary['processed_count']}",
            f"- Skipped by resume: {summary['skipped_resume_count']}",
            f"- Skipped by limit: {summary['skipped_limit_count']}",
            f"- Ambiguous labels: {summary['ambiguous_count']}",
            f"- Invalid JSON: {summary['invalid_json_count']}",
            f"- Missing input: {summary['missing_input_count']}",
            f"- Judge errors: {summary['error_count']}",
            f"- Stopped early: {summary['stopped_early']}",
            f"- Stop reason: {summary['stop_reason']}",
            f"- Dry run: {summary['dry_run']}",
            f"- Judge provider: `{summary['judge_provider']}`",
            f"- Judge model: `{summary['judge_model']}`",
            "",
        ]
    )
