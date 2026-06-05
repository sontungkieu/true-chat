from __future__ import annotations

import json
import re
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from statistics import mean
from typing import Any, Protocol

from rag_bench.chat_service import DEFAULT_MIMO_BASE_URL
from rag_bench.groq_client import GenerationResult, OpenAICompatibleClient, RoundRobinGroqClient
from rag_bench.io import write_json
from rag_bench.secrets import ApiKey, load_env_api_key, load_groq_keys


LABEL_SCHEMA_VERSION = "rlaif-answer-label-v1"
PROMPT_VERSION = "rlaif-answer-judge-v1"
DEFAULT_MAX_COMPLETION_TOKENS = 4096
DEFAULT_MIMO_JUDGE_MODEL = "mimo-v2.5"
SCORE_FIELDS = (
    "answer_correctness",
    "evidence_support",
    "unsupported_claim_penalty",
    "refusal_correctness",
    "citation_faithfulness",
    "conciseness",
    "overall_quality",
)


class AnswerJudgeClient(Protocol):
    def generate(
        self,
        messages: list[dict[str, str]],
        *,
        model: str | None = None,
        temperature: float = 0.0,
        max_completion_tokens: int = 512,
    ) -> GenerationResult:
        ...


@dataclass(frozen=True)
class RlaifAnswerLabelConfig:
    actions_path: Path
    output_path: Path
    judge_provider: str = "mimo"
    judge_model: str = DEFAULT_MIMO_JUDGE_MODEL
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


def label_rlaif_answers(config: RlaifAnswerLabelConfig) -> dict[str, Any]:
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
    config: RlaifAnswerLabelConfig,
    client: AnswerJudgeClient | None,
) -> dict[str, Any]:
    base = _base_label(action, config=config)
    answer = str(action.get("answer") or "").strip()
    context = _format_context(action, max_context_chars=config.max_context_chars)
    if not answer:
        return _ambiguous_label(base, missing_reason="missing_answer", rationale="No answer text was available.")
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

    messages = _judge_messages(action, answer=answer, context=context)
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
            return _label_from_judge(base, parsed, raw_response=last_raw, generation=result, attempt=attempt)
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


def _base_label(action: dict[str, Any], *, config: RlaifAnswerLabelConfig) -> dict[str, Any]:
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
        **{field: None for field in SCORE_FIELDS},
        "quality_score": None,
        "faithfulness": None,
        "answer_relevancy": None,
        "answer_correctness": None,
        "unsupported_claim_penalty": None,
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
) -> dict[str, Any]:
    scores = {field: _score_or_none(parsed.get(field)) for field in SCORE_FIELDS}
    quality_values = [
        value
        for value in (
            scores["overall_quality"],
            scores["answer_correctness"],
            scores["evidence_support"],
            scores["citation_faithfulness"],
        )
        if value is not None
    ]
    quality_score = scores["overall_quality"] if scores["overall_quality"] is not None else (
        mean(quality_values) if quality_values else None
    )
    unsupported = scores["unsupported_claim_penalty"]
    label = {
        **base,
        **scores,
        "quality_score": quality_score,
        "answer_relevancy": quality_score,
        "answer_correctness": scores["answer_correctness"],
        "faithfulness": scores["citation_faithfulness"],
        "unsupported_claim_penalty": unsupported,
        "short_rationale": _optional_text(parsed.get("short_rationale")),
        "rationale": _optional_text(parsed.get("short_rationale")),
        "ambiguous": bool(parsed.get("ambiguous", False)) or quality_score is None,
        "invalid_json": False,
        "metadata": {
            "raw_response_preview": raw_response[:1000],
            "json_retry_count": attempt,
            **_generation_metadata(generation, attempt=attempt),
        },
    }
    return label


def _judge_messages(action: dict[str, Any], *, answer: str, context: str) -> list[dict[str, str]]:
    question = str(action.get("question") or "").strip()
    return [
        {
            "role": "system",
            "content": (
                "You are an offline RLAIF answer judge for a RAG benchmark. "
                "Judge only from the provided question, answer, and retrieved context. "
                "Do not browse, do not use external knowledge, and do not reward unsupported claims. "
                "Do not reveal reasoning. Return only one compact valid JSON object."
            ),
        },
        {
            "role": "user",
            "content": (
                "Evaluate the answer against the retrieved context.\n\n"
                f"Question:\n{question}\n\n"
                f"Answer:\n{answer}\n\n"
                f"Retrieved context:\n{context}\n\n"
                "Return exactly one minified JSON object. It must start with `{` and end with `}`. "
                "Use only these keys: "
                "answer_correctness, evidence_support, unsupported_claim_penalty, refusal_correctness, "
                "citation_faithfulness, conciseness, overall_quality, ambiguous, short_rationale. "
                "Scores are numbers from 0 to 1 or null. unsupported_claim_penalty=1 means severe unsupported claims. "
                "refusal_correctness is null unless the answer is a refusal/abstention. "
                "short_rationale must be one short sentence. No markdown. No extra keys."
            ),
        },
    ]


def _format_context(action: dict[str, Any], *, max_context_chars: int) -> str:
    retrieved = action.get("retrieved")
    if not isinstance(retrieved, list):
        return ""
    parts: list[str] = []
    remaining = max_context_chars
    for item in retrieved:
        if remaining <= 0:
            break
        if not isinstance(item, dict):
            continue
        doc_id = str(item.get("doc_id") or "")
        title = str(item.get("title") or "")
        rank = item.get("rank")
        text = str(item.get("text") or "").strip()
        if not text:
            continue
        header = f"[rank {rank} doc_id {doc_id}] {title}".strip()
        chunk = f"{header}\n{text}\n"
        parts.append(chunk[:remaining])
        remaining -= len(parts[-1])
    return "\n".join(parts)


def _parse_judge_json(raw: str) -> dict[str, Any] | None:
    text = raw.strip()
    if not text:
        return None
    candidates = _json_candidates(text)
    for candidate in candidates:
        parsed = _loads_json_object(candidate)
        if parsed is not None:
            return parsed
    return None


def _json_candidates(text: str) -> list[str]:
    text = text.strip().removeprefix("\ufeff").strip()
    candidates = [text]
    fence_matches = re.findall(r"```(?:json|JSON)?\s*(.*?)\s*```", text, flags=re.DOTALL)
    candidates.extend(match.strip() for match in fence_matches if match.strip())
    extracted = _extract_first_json_object(text)
    if extracted is not None:
        candidates.append(extracted)
    # MiMo and reasoning models sometimes prefix terse commentary before the object.
    decoder = json.JSONDecoder()
    for index, char in enumerate(text):
        if char != "{":
            continue
        try:
            _, end = decoder.raw_decode(text[index:])
        except json.JSONDecodeError:
            continue
        candidates.append(text[index : index + end])
        break

    deduped: list[str] = []
    seen: set[str] = set()
    for candidate in candidates:
        normalized = candidate.strip()
        if normalized and normalized not in seen:
            seen.add(normalized)
            deduped.append(normalized)
    return deduped


def _extract_first_json_object(text: str) -> str | None:
    start = text.find("{")
    if start < 0:
        return None
    depth = 0
    in_string = False
    escaped = False
    for index in range(start, len(text)):
        char = text[index]
        if in_string:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                in_string = False
            continue
        if char == '"':
            in_string = True
        elif char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                return text[start : index + 1]
    return None


def _loads_json_object(text: str) -> dict[str, Any] | None:
    for candidate in (text, _strip_json_trailing_commas(text)):
        try:
            parsed = json.loads(candidate)
        except json.JSONDecodeError:
            continue
        if isinstance(parsed, dict):
            return parsed
        if isinstance(parsed, list) and len(parsed) == 1 and isinstance(parsed[0], dict):
            return parsed[0]
    return None


def _strip_json_trailing_commas(text: str) -> str:
    return re.sub(r",\s*([}\]])", r"\1", text)


def _json_repair_messages(messages: list[dict[str, str]], *, last_raw: str) -> list[dict[str, str]]:
    failure = "empty content" if not last_raw.strip() else "invalid JSON"
    return [
        *messages,
        {"role": "assistant", "content": last_raw[:2000]},
        {
            "role": "user",
            "content": (
                f"Repair your previous {failure}. Return only one compact JSON object now. "
                "It must start with `{` and end with `}`. No markdown, no prose, no hidden analysis."
            ),
        },
    ]


def _print_progress(
    config: RlaifAnswerLabelConfig,
    *,
    action_index: int,
    action_count: int,
    processed: int,
    skipped_resume: int,
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
            f"[rlaif-label-answers] {action_index}/{action_count} "
            f"processed={processed} skipped_resume={skipped_resume} "
            f"invalid_json={invalid_json_count} errors={error_count} status={status} "
            f"action_id={label.get('action_id')}"
        ),
        file=sys.stderr,
        flush=True,
    )


def _build_judge_client(config: RlaifAnswerLabelConfig) -> AnswerJudgeClient:
    provider = config.judge_provider.strip().lower()
    if provider == "groq":
        return RoundRobinGroqClient(
            keys=load_groq_keys(config.groq_keys_path),
            model=config.judge_model,
            max_retries=1,
            timeout_s=config.timeout_s,
            key_tokens_per_minute=config.key_tpm,
            key_requests_per_minute=config.key_rpm,
            provider_name="Groq",
        )
    env_file = config.env_file
    if provider == "mimo":
        api_key_var = config.api_key_var or "MIMO_API_KEY"
        base_url = config.base_url or DEFAULT_MIMO_BASE_URL
        token_parameter = "max_tokens"
        provider_name = "MiMo"
    elif provider == "deepseek":
        api_key_var = config.api_key_var or "DEEPSEEK_API_KEY"
        base_url = config.base_url or "https://api.deepseek.com/v1"
        token_parameter = "max_tokens"
        provider_name = "DeepSeek"
    else:
        raise ValueError("--judge-provider must be one of: mimo, groq, deepseek")
    api_key = load_env_api_key(env_file, api_key_var, alias=provider)
    return RoundRobinGroqClient(
        keys=[api_key],
        model=config.judge_model,
        max_retries=1,
        timeout_s=config.timeout_s,
        key_tokens_per_minute=config.key_tpm,
        key_requests_per_minute=config.key_rpm,
        provider_name=provider_name,
        completion_token_parameter="max_tokens",
        client_factory=lambda key, timeout: OpenAICompatibleClient(
            api_key=key.value,
            base_url=base_url,
            timeout_s=timeout,
            token_parameter=token_parameter,
        ),
    )


def _completed_action_ids(output_path: Path) -> set[str]:
    if not output_path.is_file():
        return set()
    completed = set()
    with output_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(row, dict) and row.get("action_id"):
                completed.add(str(row["action_id"]))
    return completed


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


def _score_or_none(value: Any) -> float | None:
    if value is None:
        return None
    try:
        score = float(value)
    except (TypeError, ValueError):
        return None
    if score < 0.0:
        return 0.0
    if score > 1.0:
        return 1.0
    return score


def _optional_text(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _generation_metadata(result: GenerationResult, *, attempt: int) -> dict[str, Any]:
    return {
        "attempt": attempt,
        "key_alias": result.key_alias,
        "attempted_aliases": result.attempted_aliases,
        "latency_s": result.latency_s,
        "retry_count": result.retry_count,
        "prompt_tokens": result.prompt_tokens,
        "completion_tokens": result.completion_tokens,
        "total_tokens": result.total_tokens,
        "estimated_tokens": result.estimated_tokens,
        "scheduled_wait_s": result.scheduled_wait_s,
        "rate_limited": result.rate_limited,
        "error_status_code": result.error_status_code,
    }


def _render_summary(summary: dict[str, Any]) -> str:
    lines = [
        "# RLAIF Answer Label Summary",
        "",
        f"- Output: `{summary['output_path']}`",
        f"- Judge: `{summary['judge_provider']}` / `{summary['judge_model']}`",
        f"- Dry run: `{summary['dry_run']}`",
        f"- Stopped early: `{summary['stopped_early']}`",
        "",
        "| Metric | Value |",
        "| --- | ---: |",
    ]
    for key in (
        "action_count",
        "processed_count",
        "skipped_resume_count",
        "skipped_limit_count",
        "invalid_json_count",
        "missing_input_count",
        "error_count",
    ):
        lines.append(f"| {key.replace('_', ' ')} | {summary[key]} |")
    return "\n".join(lines).rstrip() + "\n"
