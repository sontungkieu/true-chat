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
    DEFAULT_MIMO_JUDGE_MODEL,
    _build_judge_client,
    _generation_metadata,
    _json_repair_messages,
    _optional_text,
    _parse_judge_json,
    _read_jsonl,
    _score_or_none,
)
from rag_bench.rlaif_label_contexts import _format_context
from rag_bench.rlaif_schema import stable_record_id


LABEL_SCHEMA_VERSION = "rlaif-pairwise-label-v1"
PROMPT_VERSION = "rlaif-pairwise-judge-v1"
WINNER_VALUES = {"A", "B", "tie", None}
RISK_VALUES = {"a", "b", "both", "neither", "unknown", None}


@dataclass(frozen=True)
class RlaifPairLabelConfig:
    actions_path: Path
    rewards_path: Path
    preferences_path: Path
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


def label_rlaif_pairs(config: RlaifPairLabelConfig) -> dict[str, Any]:
    _validate_config(config)
    actions = _read_jsonl(config.actions_path)
    rewards = _read_jsonl(config.rewards_path)
    preferences = _read_jsonl(config.preferences_path)
    action_by_id = _index_by_id(actions, "action_id")
    reward_by_id = _index_by_id(rewards, "action_id")
    completed = _completed_preference_ids(config.output_path) if config.resume else set()

    processed = 0
    skipped_resume = 0
    skipped_limit = 0
    invalid_json_count = 0
    missing_input_count = 0
    ambiguous_count = 0
    tie_count = 0
    error_count = 0
    stopped_early = False
    stop_reason = None

    client = None if config.dry_run else (config.client or _build_judge_client(config))
    config.output_path.parent.mkdir(parents=True, exist_ok=True)
    with config.output_path.open("a" if config.resume else "w", encoding="utf-8") as handle:
        for preference in preferences:
            preference_id = str(preference.get("preference_id") or "")
            if not preference_id:
                continue
            if preference_id in completed:
                skipped_resume += 1
                continue
            if config.limit is not None and processed >= config.limit:
                skipped_limit += 1
                continue

            label = _label_one_preference(
                preference,
                action_by_id=action_by_id,
                reward_by_id=reward_by_id,
                config=config,
                client=client,
            )
            processed += 1
            if label.get("invalid_json"):
                invalid_json_count += 1
            if label.get("missing_reason"):
                missing_input_count += 1
            if label.get("ambiguous"):
                ambiguous_count += 1
            if label.get("tie"):
                tie_count += 1
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
                preference_index=processed + skipped_resume,
                preference_count=len(preferences),
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
        "reward_count": len(rewards),
        "preference_count": len(preferences),
        "processed_count": processed,
        "skipped_resume_count": skipped_resume,
        "skipped_limit_count": skipped_limit,
        "ambiguous_count": ambiguous_count,
        "tie_count": tie_count,
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


def _validate_config(config: RlaifPairLabelConfig) -> None:
    for path, label in (
        (config.actions_path, "Actions path"),
        (config.rewards_path, "Rewards path"),
        (config.preferences_path, "Preferences path"),
    ):
        if not path.is_file():
            raise ValueError(f"{label} does not exist: {path}")
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


def _label_one_preference(
    preference: dict[str, Any],
    *,
    action_by_id: dict[str, dict[str, Any]],
    reward_by_id: dict[str, dict[str, Any]],
    config: RlaifPairLabelConfig,
    client: AnswerJudgeClient | None,
) -> dict[str, Any]:
    action_a_id = str(preference.get("chosen_action_id") or "")
    action_b_id = str(preference.get("rejected_action_id") or "")
    action_a = action_by_id.get(action_a_id)
    action_b = action_by_id.get(action_b_id)
    reward_a = reward_by_id.get(action_a_id)
    reward_b = reward_by_id.get(action_b_id)
    base = _base_label(preference, config=config, action_a_id=action_a_id, action_b_id=action_b_id)
    missing_reason = _missing_pair_reason(action_a, action_b, reward_a, reward_b)
    if missing_reason is not None:
        return _ambiguous_label(base, missing_reason=missing_reason, rationale=f"Missing pair data: {missing_reason}.")

    assert action_a is not None
    assert action_b is not None
    assert reward_a is not None
    assert reward_b is not None
    context_a, _ = _format_context(action_a, max_context_chars=config.max_context_chars // 2)
    context_b, _ = _format_context(action_b, max_context_chars=config.max_context_chars // 2)
    answer_a = str(action_a.get("answer") or "").strip()
    answer_b = str(action_b.get("answer") or "").strip()
    if not answer_a or not answer_b:
        return _ambiguous_label(base, missing_reason="missing_answer", rationale="One or both action answers were missing.")
    if not context_a.strip() or not context_b.strip():
        return _ambiguous_label(base, missing_reason="missing_context", rationale="One or both action contexts were missing.")
    if config.dry_run:
        return _ambiguous_label(
            base,
            rationale="Dry-run placeholder; no judge call was made.",
            metadata={"dry_run": True},
        )
    if client is None:
        return _ambiguous_label(base, missing_reason="missing_judge_client", rationale="No judge client was configured.")

    messages = _judge_messages(
        preference=preference,
        action_a=action_a,
        action_b=action_b,
        reward_a=reward_a,
        reward_b=reward_b,
        answer_a=answer_a,
        answer_b=answer_b,
        context_a=context_a,
        context_b=context_b,
    )
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


def _base_label(
    preference: dict[str, Any],
    *,
    config: RlaifPairLabelConfig,
    action_a_id: str,
    action_b_id: str,
) -> dict[str, Any]:
    preference_id = str(preference.get("preference_id") or "")
    query_id = str(preference.get("query_id") or "")
    return {
        "schema_version": LABEL_SCHEMA_VERSION,
        "pair_label_id": stable_record_id(
            "rlaif-pairwise-label-v1",
            {
                "preference_id": preference_id,
                "action_a_id": action_a_id,
                "action_b_id": action_b_id,
                "judge_provider": config.judge_provider,
                "judge_model": config.judge_model,
            },
        ),
        "preference_id": preference_id,
        "preference_type": preference.get("preference_type"),
        "query_id": query_id,
        "provenance": "ai_judge",
        "judge_provider": config.judge_provider,
        "judge_model": config.judge_model,
        "judge_version": PROMPT_VERSION,
        "prompt_version": PROMPT_VERSION,
        "action_a_id": action_a_id,
        "action_b_id": action_b_id,
        "reward_preference_chosen_action_id": action_a_id,
        "reward_preference_rejected_action_id": action_b_id,
        "reward_gap": preference.get("reward_gap"),
        "quality_gap": preference.get("quality_gap"),
        "efficiency_gap": preference.get("efficiency_gap"),
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
        "chosen": None,
        "chosen_action_id": None,
        "rejected_action_id": None,
        "tie": False,
        "answer_quality_winner": None,
        "evidence_support_winner": None,
        "efficiency_winner": None,
        "quality_regret": None,
        "unsupported_claim_risk": None,
        "confidence": None,
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
    chosen = _winner_or_none(parsed.get("chosen"))
    tie = bool(parsed.get("tie", False)) or chosen == "tie"
    if tie:
        chosen = None
    confidence = _score_or_none(parsed.get("confidence"))
    ambiguous = bool(parsed.get("ambiguous", False)) or (not tie and chosen is None)
    chosen_action_id = _chosen_action_id(base, chosen)
    rejected_action_id = _rejected_action_id(base, chosen)
    return {
        **base,
        "chosen": chosen,
        "chosen_action_id": chosen_action_id,
        "rejected_action_id": rejected_action_id,
        "tie": tie,
        "answer_quality_winner": _winner_or_none(parsed.get("answer_quality_winner")),
        "evidence_support_winner": _winner_or_none(parsed.get("evidence_support_winner")),
        "efficiency_winner": _winner_or_none(parsed.get("efficiency_winner")),
        "quality_regret": _bool_or_none(parsed.get("quality_regret")),
        "unsupported_claim_risk": _risk_or_none(parsed.get("unsupported_claim_risk")),
        "confidence": confidence,
        "short_rationale": _optional_text(parsed.get("short_rationale")),
        "rationale": _optional_text(parsed.get("short_rationale")),
        "ambiguous": ambiguous,
        "invalid_json": False,
        "metadata": {
            "raw_response_preview": raw_response[:1000],
            "json_retry_count": attempt,
            **_generation_metadata(generation, attempt=attempt),
        },
    }


def _judge_messages(
    *,
    preference: dict[str, Any],
    action_a: dict[str, Any],
    action_b: dict[str, Any],
    reward_a: dict[str, Any],
    reward_b: dict[str, Any],
    answer_a: str,
    answer_b: str,
    context_a: str,
    context_b: str,
) -> list[dict[str, str]]:
    question = str(action_a.get("question") or action_b.get("question") or "").strip()
    return [
        {
            "role": "system",
            "content": (
                "You are an offline pairwise RLAIF judge for BudgetRAG. "
                "Compare two retrieval-context actions for the same query using only the provided question, "
                "answers, retrieved contexts, and resource-cost metadata. Do not browse or use external knowledge. "
                "Prefer correctness, evidence support, and lower unsupported-claim risk first. "
                "If both are similarly correct and supported, prefer lower token, latency, and KV cost. "
                "Never prefer a cheaper action if it loses important evidence or becomes unsupported. "
                "Return only one compact valid JSON object."
            ),
        },
        {
            "role": "user",
            "content": (
                "Compare Action A and Action B for the same BudgetRAG query.\n\n"
                f"Question:\n{question}\n\n"
                f"Reward-derived preference before judging:\n{_preference_summary(preference)}\n\n"
                f"Action A:\n{_action_summary(action_a, reward_a)}\nAnswer A:\n{answer_a}\nContext A:\n{context_a}\n\n"
                f"Action B:\n{_action_summary(action_b, reward_b)}\nAnswer B:\n{answer_b}\nContext B:\n{context_b}\n\n"
                "Return exactly one minified JSON object. It must start with `{` and end with `}`. "
                "Use only these keys: chosen, tie, ambiguous, answer_quality_winner, evidence_support_winner, "
                "efficiency_winner, quality_regret, unsupported_claim_risk, confidence, short_rationale. "
                "chosen is \"A\", \"B\", or null. tie is true only when neither action is meaningfully better. "
                "Winner fields are \"A\", \"B\", \"tie\", or null. unsupported_claim_risk is "
                "\"A\", \"B\", \"both\", \"neither\", or \"unknown\". confidence is a number from 0 to 1 or null. "
                "short_rationale must be one short sentence. No markdown. No extra keys."
            ),
        },
    ]


def _action_summary(action: dict[str, Any], reward: dict[str, Any]) -> str:
    payload = {
        "action_id": action.get("action_id"),
        "retrieval_strategy": action.get("retrieval_strategy"),
        "fusion_strategy": action.get("fusion_strategy"),
        "top_k": action.get("top_k"),
        "context_policy": action.get("context_policy"),
        "budget_chars": action.get("budget_chars"),
        "adaptive_profile": action.get("adaptive_profile"),
        "selected_context_policy": action.get("selected_context_policy"),
        "selected_budget_chars": action.get("selected_budget_chars"),
        "generator_model": action.get("generator_model"),
        "costs": {
            "tokens": _token_count(action),
            "latency_s": _latency_s(action),
            "kv_mb": _kv_mb(action),
        },
        "logged_reward": {
            "reward": reward.get("reward"),
            "quality": reward.get("quality"),
            "evidence_support": reward.get("evidence_support"),
            "token_cost_norm": reward.get("token_cost_norm"),
            "latency_norm": reward.get("latency_norm"),
            "kv_cost_norm": reward.get("kv_cost_norm"),
            "unsupported_claim_penalty": reward.get("unsupported_claim_penalty"),
        },
    }
    return json.dumps(payload, ensure_ascii=False, sort_keys=True)


def _preference_summary(preference: dict[str, Any]) -> str:
    payload = {
        "preference_id": preference.get("preference_id"),
        "preference_type": preference.get("preference_type"),
        "chosen_action_id": preference.get("chosen_action_id"),
        "rejected_action_id": preference.get("rejected_action_id"),
        "reason": preference.get("reason"),
        "reward_gap": preference.get("reward_gap"),
        "quality_gap": preference.get("quality_gap"),
        "efficiency_gap": preference.get("efficiency_gap"),
    }
    return json.dumps(payload, ensure_ascii=False, sort_keys=True)


def _missing_pair_reason(
    action_a: dict[str, Any] | None,
    action_b: dict[str, Any] | None,
    reward_a: dict[str, Any] | None,
    reward_b: dict[str, Any] | None,
) -> str | None:
    missing = []
    if action_a is None:
        missing.append("action_a")
    if action_b is None:
        missing.append("action_b")
    if reward_a is None:
        missing.append("reward_a")
    if reward_b is None:
        missing.append("reward_b")
    return ",".join(missing) if missing else None


def _chosen_action_id(base: dict[str, Any], chosen: str | None) -> str | None:
    if chosen == "A":
        return str(base["action_a_id"])
    if chosen == "B":
        return str(base["action_b_id"])
    return None


def _rejected_action_id(base: dict[str, Any], chosen: str | None) -> str | None:
    if chosen == "A":
        return str(base["action_b_id"])
    if chosen == "B":
        return str(base["action_a_id"])
    return None


def _winner_or_none(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    normalized = text.upper() if text.lower() != "tie" else "tie"
    return normalized if normalized in WINNER_VALUES else None


def _risk_or_none(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip().lower()
    return text if text in RISK_VALUES else None


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


def _token_count(action: dict[str, Any]) -> float | None:
    token_usage = action.get("token_usage")
    if not isinstance(token_usage, dict):
        return None
    return _float_or_none(token_usage.get("total_tokens"))


def _latency_s(action: dict[str, Any]) -> float | None:
    latency = action.get("latency")
    if not isinstance(latency, dict):
        return None
    return _float_or_none(latency.get("total_latency_s") or latency.get("latency_s"))


def _kv_mb(action: dict[str, Any]) -> float | None:
    kv_estimate = action.get("kv_estimate")
    if not isinstance(kv_estimate, dict):
        return None
    return _float_or_none(kv_estimate.get("after_mb") or kv_estimate.get("kv_cache_mb"))


def _float_or_none(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _index_by_id(rows: list[dict[str, Any]], field: str) -> dict[str, dict[str, Any]]:
    output: dict[str, dict[str, Any]] = {}
    for row in rows:
        value = row.get(field)
        if isinstance(value, str) and value:
            output[value] = row
    return output


def _completed_preference_ids(output_path: Path) -> set[str]:
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
            if isinstance(row, dict) and row.get("preference_id"):
                completed.add(str(row["preference_id"]))
    return completed


def _print_progress(
    config: RlaifPairLabelConfig,
    *,
    preference_index: int,
    preference_count: int,
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
    elif label.get("tie"):
        status = "tie"
    elif label.get("ambiguous"):
        status = "ambiguous"
    print(
        (
            f"[rlaif-label-pairs] {preference_index}/{preference_count} "
            f"processed={processed} skipped_resume={skipped_resume} ambiguous={ambiguous_count} "
            f"invalid_json={invalid_json_count} errors={error_count} status={status} "
            f"preference_id={label.get('preference_id')}"
        ),
        file=sys.stderr,
        flush=True,
    )


def _render_summary(summary: dict[str, Any]) -> str:
    return "\n".join(
        [
            "# RLAIF Pairwise Label Summary",
            "",
            f"- Output: `{summary['output_path']}`",
            f"- Actions read: {summary['action_count']}",
            f"- Rewards read: {summary['reward_count']}",
            f"- Preferences read: {summary['preference_count']}",
            f"- Processed: {summary['processed_count']}",
            f"- Skipped by resume: {summary['skipped_resume_count']}",
            f"- Skipped by limit: {summary['skipped_limit_count']}",
            f"- Ambiguous labels: {summary['ambiguous_count']}",
            f"- Tie labels: {summary['tie_count']}",
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
