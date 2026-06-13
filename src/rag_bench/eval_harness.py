from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol, Sequence

from rag_bench.benchmarks import load_benchmark
from rag_bench.chat_service import (
    ChatGenerationClient,
    ChatProxyConfig,
    ChatServiceResult,
    RagChatService,
    _build_llm,
    _build_retrievers,
    _default_retriever,
    _load_dictionary,
    _load_structured_evidence_index,
)
from rag_bench.groq_client import GenerationResult, OpenAICompatibleClient, RoundRobinGroqClient
from rag_bench.privacy import (
    BackendKind,
    ConversationPrivacyState,
    DataTier,
    PrivateBackendPolicy,
    PrivacyRouteError,
    classify_backend,
    data_tier_for_hit,
    enforce_privacy_route,
    normalize_data_tier,
    safe_source_payload,
)
from rag_bench.secrets import ApiKey, load_env_api_key, load_groq_keys
from rag_bench.types import RetrievalHit


DEFAULT_RAG_EVAL_OUTPUT_ROOT = Path("eval_results/rag_eval")
DEFAULT_DEEPSEEK_BASE_URL = "https://api.deepseek.com/v1"
DEFAULT_GROQ_BASE_URL = "https://api.groq.com/openai/v1"
DEFAULT_MIMO_BASE_URL = "https://token-plan-sgp.xiaomimimo.com/v1"
DEFAULT_OPENAI_BASE_URL = "https://api.openai.com/v1"


@dataclass(frozen=True)
class RagEvalItem:
    eval_id: str
    query: str
    mode: str = "dictionary"
    data_tier: str = "public"
    expected_intent: str | None = None
    expected_doc_ids: list[str] = field(default_factory=list)
    expected_structured_doc_types: list[str] = field(default_factory=list)
    expected_schema_gaps: list[str] = field(default_factory=list)
    forbidden_schema_gaps: list[str] = field(default_factory=list)
    should_answer: bool = True
    should_have_citations: bool = True
    notes: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_mapping(cls, row: dict[str, Any]) -> "RagEvalItem":
        eval_id = str(row.get("eval_id") or "").strip()
        query = str(row.get("query") or "").strip()
        if not eval_id:
            raise ValueError("RAG eval item requires eval_id")
        if not query:
            raise ValueError(f"RAG eval item {eval_id} requires query")
        return cls(
            eval_id=eval_id,
            query=query,
            mode=str(row.get("mode") or "dictionary").strip() or "dictionary",
            data_tier=normalize_data_tier(row.get("data_tier"), missing=DataTier.PUBLIC).value,
            expected_intent=_optional_str(row.get("expected_intent")),
            expected_doc_ids=_string_list(row.get("expected_doc_ids")),
            expected_structured_doc_types=_string_list(row.get("expected_structured_doc_types")),
            expected_schema_gaps=_string_list(row.get("expected_schema_gaps")),
            forbidden_schema_gaps=_string_list(row.get("forbidden_schema_gaps")),
            should_answer=bool(row.get("should_answer", True)),
            should_have_citations=bool(row.get("should_have_citations", True)),
            notes=_optional_str(row.get("notes")),
            metadata=dict(row.get("metadata") or {}),
        )

    def to_mapping(self) -> dict[str, Any]:
        return {
            "eval_id": self.eval_id,
            "query": self.query,
            "mode": self.mode,
            "data_tier": self.data_tier,
            "expected_intent": self.expected_intent,
            "expected_doc_ids": list(self.expected_doc_ids),
            "expected_structured_doc_types": list(self.expected_structured_doc_types),
            "expected_schema_gaps": list(self.expected_schema_gaps),
            "forbidden_schema_gaps": list(self.forbidden_schema_gaps),
            "should_answer": self.should_answer,
            "should_have_citations": self.should_have_citations,
            "notes": self.notes,
            "metadata": dict(self.metadata),
        }


@dataclass(frozen=True)
class RagEvalResult:
    eval_id: str
    query: str
    data_tier: str
    generator_provider: str
    generator_model: str | None
    judge_provider: str | None
    judge_model: str | None
    answer: str
    query_plan: dict[str, Any]
    retrieved_doc_ids: list[str]
    privacy: dict[str, Any]
    heuristic_scores: dict[str, Any]
    judge_scores: dict[str, Any] | None
    judge_skipped: bool
    judge_skip_reason: str | None
    expected: dict[str, Any] = field(default_factory=dict)
    judge_error: str | None = None

    def to_mapping(self, *, include_private_outputs: bool = False) -> dict[str, Any]:
        return sanitize_eval_result_for_write(self, include_private_eval_text=include_private_outputs)


def sanitize_eval_result_for_write(
    result: RagEvalResult,
    *,
    include_private_eval_text: bool = False,
) -> dict[str, Any]:
    payload = {
        "eval_id": result.eval_id,
        "query": result.query,
        "data_tier": result.data_tier,
        "generator_provider": result.generator_provider,
        "generator_model": result.generator_model,
        "judge_provider": result.judge_provider,
        "judge_model": result.judge_model,
        "answer": result.answer,
        "query_plan": result.query_plan,
        "retrieved_doc_ids": list(result.retrieved_doc_ids),
        "privacy": result.privacy,
        "heuristic_scores": result.heuristic_scores,
        "judge_scores": result.judge_scores,
        "judge_skipped": result.judge_skipped,
        "judge_skip_reason": result.judge_skip_reason,
        "expected": result.expected,
        "judge_error": result.judge_error,
        "judge_error_redacted": False,
    }
    if result.data_tier != DataTier.PRIVATE.value or include_private_eval_text:
        return payload
    payload["query"] = "[REDACTED_PRIVATE]"
    payload["answer"] = "[REDACTED_PRIVATE]"
    payload["query_plan"] = _sanitize_private_query_plan(result.query_plan)
    payload["privacy"] = _sanitize_private_privacy(result.privacy)
    payload["judge_scores"] = _sanitize_private_judge_scores(result.judge_scores)
    if result.judge_error:
        payload["judge_error"] = "redacted_private_judge_error"
        payload["judge_error_redacted"] = True
    return payload


@dataclass(frozen=True)
class RagEvalConfig:
    eval_set: Path
    out_dir: Path | None = None
    generator_provider: str = "local"
    generator_model: str | None = "heuristic-local"
    generator_backend_id: str | None = None
    generator_backend_kind: str | None = "local_process"
    generator_trusted_private_backends: tuple[str, ...] = ()
    generator_trusted_private_models: tuple[str, ...] = ()
    judge_provider: str | None = None
    judge_model: str | None = None
    judge_backend_id: str | None = None
    judge_backend_kind: str | None = None
    judge_trusted_private_backends: tuple[str, ...] = ()
    judge_trusted_private_models: tuple[str, ...] = ()
    allow_external_judge_public: bool = False
    allow_external_judge_semi_private: bool = False
    disable_llm_judge: bool = True
    judge_max_completion_tokens: int = 2048
    include_private_outputs: bool = False
    chat_config: ChatProxyConfig = field(default_factory=ChatProxyConfig)


class RagEvalJudgeClient(Protocol):
    def generate(
        self,
        messages: list[dict[str, str]],
        *,
        model: str | None = None,
        temperature: float = 0.0,
        max_completion_tokens: int = 512,
    ) -> GenerationResult: ...


@dataclass
class HeuristicGeneratorClient:
    provider: str = "local"
    model: str = "heuristic-local"
    key_usage_counts: dict[str, int] = field(default_factory=dict)

    def generate(
        self,
        messages: list[dict[str, str]],
        *,
        model: str | None = None,
        temperature: float = 0.0,
        max_completion_tokens: int = 512,
    ) -> GenerationResult:
        prompt = "\n\n".join(message.get("content", "") for message in messages)
        citations = _extract_context_citations(prompt)
        answer = "Heuristic local answer"
        if citations:
            answer = f"{answer} [{citations[0]}]"
        return GenerationResult(
            answer=answer,
            key_alias=self.provider,
            attempted_aliases=[self.provider],
            latency_s=0.0,
            retry_count=0,
            prompt_tokens=None,
            completion_tokens=None,
            total_tokens=None,
            estimated_tokens=None,
        )

    def rate_limit_snapshot(self) -> dict[str, dict[str, float | int | str]]:
        return {}


def load_rag_eval_items(path: Path) -> list[RagEvalItem]:
    items: list[RagEvalItem] = []
    with path.open(encoding="utf-8") as file_obj:
        for line_number, line in enumerate(file_obj, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_number}: invalid JSON") from exc
            if not isinstance(row, dict):
                raise ValueError(f"{path}:{line_number}: expected JSON object")
            items.append(RagEvalItem.from_mapping(row))
    return items


def run_rag_eval(
    config: RagEvalConfig,
    *,
    service: RagChatService | None = None,
    judge_client: RagEvalJudgeClient | None = None,
) -> dict[str, Any]:
    items = load_rag_eval_items(config.eval_set)
    out_dir = config.out_dir or _timestamped_output_dir()
    out_dir.mkdir(parents=True, exist_ok=True)
    service = service or build_eval_chat_service(config)
    judge_client = judge_client if judge_client is not None else _build_default_judge_client(config)
    results: list[RagEvalResult] = []
    failures: list[RagEvalResult] = []
    results_path = out_dir / "results.jsonl"
    failures_path = out_dir / "failures.jsonl"
    with results_path.open("w", encoding="utf-8") as results_file, failures_path.open("w", encoding="utf-8") as failures_file:
        for item in items:
            result = evaluate_rag_item(item, config, service=service, judge_client=judge_client)
            results.append(result)
            result_row = sanitize_eval_result_for_write(
                result,
                include_private_eval_text=config.include_private_outputs,
            )
            results_file.write(json.dumps(result_row, ensure_ascii=False) + "\n")
            if _is_failure(result):
                failures.append(result)
                failures_file.write(json.dumps(result_row, ensure_ascii=False) + "\n")
    summary = summarize_rag_eval(results, config=config)
    summary_path = out_dir / "summary.md"
    summary_path.write_text(summary, encoding="utf-8")
    return {
        "output_dir": str(out_dir),
        "results_path": str(results_path),
        "summary_path": str(summary_path),
        "failures_path": str(failures_path),
        "item_count": len(results),
        "failure_count": len(failures),
        "judge_called_count": sum(1 for result in results if not result.judge_skipped),
    }


def build_eval_chat_service(config: RagEvalConfig) -> RagChatService:
    chat_config = _eval_chat_config(config)
    llm = _build_eval_generator(config, chat_config)
    benchmark = load_benchmark(chat_config.bench, limit=None, allow_large=chat_config.allow_large_bench)
    dictionary = _load_dictionary(chat_config)
    retrievers = _build_retrievers(chat_config, benchmark, llm=llm, dictionary=dictionary)
    retriever = _default_retriever(chat_config, retrievers)
    return RagChatService(
        config=chat_config,
        benchmark=benchmark,
        retriever=retriever,
        llm=llm,
        retrievers=retrievers,
        dictionary_status=dictionary.status,
        structured_evidence_index=_load_structured_evidence_index(chat_config),
    )


def evaluate_rag_item(
    item: RagEvalItem,
    config: RagEvalConfig,
    *,
    service: RagChatService,
    judge_client: RagEvalJudgeClient | None = None,
) -> RagEvalResult:
    answer = ""
    query_plan: dict[str, Any] = {}
    retrieved_doc_ids: list[str] = []
    privacy: dict[str, Any] = {}
    hits: list[RetrievalHit] = []
    generation_error: str | None = None
    try:
        service_result = service.answer(
            [{"role": "user", "content": _mode_query(item), "data_tier": item.data_tier}],
            request_model=config.generator_model,
            response_mode=item.mode,
            session_id=f"rag-eval-{item.eval_id}",
            reset_privacy=True,
        )
        answer = _answer_from_service_result(service_result)
        query_plan = _query_plan_from_service_result(service_result)
        retrieved_doc_ids = [hit.doc_id for hit in service_result.hits]
        privacy = dict(service_result.response.get("privacy") or {})
        hits = list(service_result.hits)
    except PrivacyRouteError as exc:
        generation_error = exc.decision.reason
        privacy = exc.decision.to_payload()

    heuristic_scores = compute_heuristic_scores(
        item,
        answer=answer,
        query_plan=query_plan,
        retrieved_doc_ids=retrieved_doc_ids,
        privacy=privacy,
        generation_error=generation_error,
    )
    judge_scores: dict[str, Any] | None = None
    judge_skipped = True
    judge_skip_reason: str | None = None
    judge_error: str | None = None
    if generation_error is not None:
        judge_skip_reason = "generator_blocked_by_privacy"
    else:
        allowed, judge_skip_reason = _judge_allowed(item, config, hits)
        if allowed:
            if judge_client is None:
                judge_skip_reason = "judge_client_not_configured"
            else:
                try:
                    judge_messages = build_judge_messages(item, answer=answer, query_plan=query_plan, hits=hits, config=config)
                    judge_generation = judge_client.generate(
                        judge_messages,
                        model=config.judge_model,
                        temperature=0.0,
                        max_completion_tokens=config.judge_max_completion_tokens,
                    )
                    if judge_generation.error:
                        raise ValueError(f"judge generation failed: {judge_generation.error}")
                    judge_scores = _parse_judge_json(judge_generation.answer)
                    judge_skipped = False
                    judge_skip_reason = None
                except Exception as exc:  # noqa: BLE001 - judge providers vary; keep eval safe.
                    judge_error = _safe_error_text(exc)
                    judge_skip_reason = "judge_error"

    return RagEvalResult(
        eval_id=item.eval_id,
        query=item.query,
        data_tier=item.data_tier,
        generator_provider=config.generator_provider,
        generator_model=config.generator_model,
        judge_provider=config.judge_provider,
        judge_model=config.judge_model,
        answer=answer,
        query_plan=query_plan,
        retrieved_doc_ids=retrieved_doc_ids,
        privacy=privacy,
        heuristic_scores=heuristic_scores,
        judge_scores=judge_scores,
        judge_skipped=judge_skipped,
        judge_skip_reason=judge_skip_reason,
        expected={
            "expected_intent": item.expected_intent,
            "expected_doc_ids": list(item.expected_doc_ids),
            "expected_structured_doc_types": list(item.expected_structured_doc_types),
            "expected_schema_gaps": list(item.expected_schema_gaps),
            "forbidden_schema_gaps": list(item.forbidden_schema_gaps),
        },
        judge_error=judge_error,
    )


def compute_heuristic_scores(
    item: RagEvalItem,
    *,
    answer: str,
    query_plan: dict[str, Any],
    retrieved_doc_ids: Sequence[str],
    privacy: dict[str, Any],
    generation_error: str | None = None,
) -> dict[str, Any]:
    actual_intent = query_plan.get("intent")
    actual_gaps = set(_string_list(query_plan.get("schema_gaps")))
    retrieved = set(retrieved_doc_ids)
    structured = query_plan.get("structured_evidence") if isinstance(query_plan.get("structured_evidence"), dict) else {}
    matched_types = set(_string_list((structured or {}).get("matched_doc_types")))
    scores = {
        "intent_match": _expected_bool(item.expected_intent is None, actual_intent == item.expected_intent),
        "expected_docs_retrieved": _expected_bool(not item.expected_doc_ids, set(item.expected_doc_ids).issubset(retrieved)),
        "schema_gap_expected": _expected_bool(not item.expected_schema_gaps, set(item.expected_schema_gaps).issubset(actual_gaps)),
        "schema_gap_forbidden": _expected_bool(not item.forbidden_schema_gaps, not (set(item.forbidden_schema_gaps) & actual_gaps)),
        "citation_present": _expected_bool(not item.should_have_citations, _has_citation(answer, retrieved_doc_ids)),
        "structured_evidence_used": _expected_bool(
            not item.expected_structured_doc_types,
            set(item.expected_structured_doc_types).issubset(matched_types),
        ),
        "privacy_external_blocked": _expected_bool(
            not bool(item.metadata.get("expect_external_judge_blocked")),
            bool(privacy.get("external_blocked")) or generation_error is not None,
        ),
        "generator_blocked": generation_error is not None,
    }
    scores["all_required_passed"] = all(
        value is not False
        for key, value in scores.items()
        if key not in {"generator_blocked"}
    )
    return scores


def build_judge_messages(
    item: RagEvalItem,
    *,
    answer: str,
    query_plan: dict[str, Any],
    hits: Sequence[RetrievalHit],
    config: RagEvalConfig,
) -> list[dict[str, str]]:
    backend = classify_backend(
        provider=config.judge_provider,
        model=config.judge_model,
        backend_id=config.judge_backend_id or config.judge_provider,
        backend_kind=config.judge_backend_kind,
    )
    include_private_text = backend.kind in {BackendKind.LOCAL_PROCESS, BackendKind.SELF_HOSTED_PRIVATE, BackendKind.PRIVATE_LAN, BackendKind.PRIVATE_VPC}
    sources = [
        safe_source_payload(hit, include_private_text=include_private_text)
        for hit in hits
    ]
    payload = {
        "query": item.query,
        "answer": answer,
        "retrieved_source_ids": [hit.doc_id for hit in hits],
        "retrieved_sources": sources,
        "query_plan": query_plan,
        "expected": item.to_mapping(),
    }
    rubric = {
        "answer_correctness": "0.0 to 1.0",
        "groundedness": "0.0 to 1.0",
        "citation_support": "0.0 to 1.0",
        "missing_evidence_behavior": "0.0 to 1.0",
        "planner_success": "0.0 to 1.0",
        "privacy_safety": "0.0 to 1.0",
        "overall": "0.0 to 1.0",
        "issues": ["short issue strings"],
        "verdict": "pass|partial|fail",
    }
    return [
        {
            "role": "system",
            "content": (
                "You are a strict RAG evaluation judge. Use only the supplied query, answer, retrieved sources, "
                "planner metadata, and expected checks. Do not browse. Return one strict JSON object only."
            ),
        },
        {
            "role": "user",
            "content": (
                "Evaluate the RAG answer using this JSON rubric:\n"
                f"{json.dumps(rubric, ensure_ascii=False)}\n\n"
                "Evaluation payload:\n"
                f"{json.dumps(payload, ensure_ascii=False)}"
            ),
        },
    ]


def summarize_rag_eval(results: Sequence[RagEvalResult], *, config: RagEvalConfig) -> str:
    total = len(results)
    judge_called = sum(1 for result in results if not result.judge_skipped)
    skipped = total - judge_called
    failures = [result for result in results if _is_failure(result)]
    skip_reasons: dict[str, int] = {}
    for result in results:
        if result.judge_skipped:
            reason = result.judge_skip_reason or "unknown"
            skip_reasons[reason] = skip_reasons.get(reason, 0) + 1
    heuristic_keys = sorted({key for result in results for key in result.heuristic_scores if key != "all_required_passed"})
    lines = [
        "# RAG Generator/Judge Eval Summary",
        "",
        "## Setup",
        "",
        f"- items: {total}",
        f"- generator: `{config.generator_provider}` / `{config.generator_model}`",
        f"- judge: `{config.judge_provider}` / `{config.judge_model}`",
        f"- llm judge disabled: `{config.disable_llm_judge}`",
        f"- allow external judge public: `{config.allow_external_judge_public}`",
        f"- allow external judge semi-private: `{config.allow_external_judge_semi_private}`",
        f"- judge calls: {judge_called}",
        f"- judge skipped: {skipped}",
        f"- failures: {len(failures)}",
        "",
        "## Heuristic Pass Counts",
        "",
        "| check | pass | fail | n/a |",
        "| --- | ---: | ---: | ---: |",
    ]
    for key in heuristic_keys:
        values = [result.heuristic_scores.get(key) for result in results]
        lines.append(
            f"| `{key}` | {sum(value is True for value in values)} | "
            f"{sum(value is False for value in values)} | {sum(value is None for value in values)} |"
        )
    lines.extend(["", "## Judge Skip Reasons", "", "| reason | count |", "| --- | ---: |"])
    for reason, count in sorted(skip_reasons.items()):
        lines.append(f"| `{reason}` | {count} |")
    lines.extend(["", "## Limitations", "", "- AI judge scores are optional and policy-gated.", "- Heuristic checks are deterministic smoke checks, not human labels."])
    return "\n".join(lines) + "\n"


def _judge_allowed(item: RagEvalItem, config: RagEvalConfig, hits: Sequence[RetrievalHit]) -> tuple[bool, str | None]:
    if config.disable_llm_judge:
        return False, "llm_judge_disabled"
    if not config.judge_provider or not config.judge_model:
        return False, "judge_not_configured"
    backend = classify_backend(
        provider=config.judge_provider,
        model=config.judge_model,
        backend_id=config.judge_backend_id or config.judge_provider,
        backend_kind=config.judge_backend_kind,
    )
    item_tier = normalize_data_tier(item.data_tier)
    if backend.kind == BackendKind.EXTERNAL_SAAS and item_tier == DataTier.PUBLIC and not config.allow_external_judge_public:
        return False, "public_external_judge_disabled"
    state = ConversationPrivacyState(session_id=f"rag-eval-judge-{item.eval_id}", max_seen_tier=item_tier)
    policy = PrivateBackendPolicy.from_values(
        trusted_private_backends=config.judge_trusted_private_backends,
        trusted_private_models=config.judge_trusted_private_models,
    )
    decision = enforce_privacy_route(
        backend.provider,
        backend.model,
        state,
        hits,
        allow_external_semi_private=config.allow_external_judge_semi_private,
        private_backend_policy=policy,
        backend_id=backend.backend_id,
        backend_kind=backend.kind,
        user_message_tier=item_tier,
    )
    if not decision.provider_allowed:
        return False, decision.reason
    return True, None


def _eval_chat_config(config: RagEvalConfig) -> ChatProxyConfig:
    chat_config = config.chat_config
    generator_model = config.generator_model or chat_config.model
    generator_provider = str(config.generator_provider or "").strip().lower()
    mimo_models = chat_config.mimo_models
    if generator_provider == "mimo" and generator_model not in mimo_models:
        mimo_models = (generator_model, *mimo_models) if generator_model else mimo_models
    trusted_backends = config.generator_trusted_private_backends or (
        (config.generator_backend_id,) if config.generator_backend_id else ()
    )
    return ChatProxyConfig(
        **{
            **chat_config.__dict__,
            "model": generator_model,
            "mimo_enabled": bool(chat_config.mimo_enabled or generator_provider == "mimo"),
            "mimo_models": mimo_models,
            "backend_id": config.generator_backend_id or chat_config.backend_id,
            "backend_kind": config.generator_backend_kind or chat_config.backend_kind,
            "trusted_private_backends": trusted_backends or chat_config.trusted_private_backends,
            "trusted_private_models": config.generator_trusted_private_models or chat_config.trusted_private_models,
        }
    )


def _build_eval_generator(config: RagEvalConfig, chat_config: ChatProxyConfig) -> ChatGenerationClient:
    provider = str(config.generator_provider or "").strip().lower()
    if provider in {"local", "heuristic", "mock", "local_small"}:
        return HeuristicGeneratorClient(provider=config.generator_provider, model=config.generator_model or "heuristic-local")
    if provider in {"mimo", "deepseek", "openai"}:
        return _build_openai_compatible_eval_generator(config, chat_config, provider=provider)
    keys = load_groq_keys(chat_config.groq_keys_path)
    return _build_llm(chat_config, keys)


def _build_openai_compatible_eval_generator(
    config: RagEvalConfig,
    chat_config: ChatProxyConfig,
    *,
    provider: str,
) -> ChatGenerationClient:
    env_name, base_url = _judge_env_and_base_url(provider)
    import os

    api_key = os.getenv(env_name)
    if api_key:
        key = ApiKey(alias=provider, value=api_key)
    elif provider == "mimo":
        key = load_env_api_key(chat_config.mimo_env_file, chat_config.mimo_api_key_var, alias=provider)
    else:
        raise RuntimeError(f"{env_name} is required for {provider} eval generation")
    return RoundRobinGroqClient(
        keys=[key],
        model=config.generator_model,
        max_retries=chat_config.max_retries,
        key_tokens_per_minute=0,
        key_requests_per_minute=0,
        client_factory=lambda key, timeout: OpenAICompatibleClient(
            api_key=key.value,
            base_url=base_url,
            timeout_s=timeout,
            token_parameter="max_tokens",
        ),
        provider_name=provider,
        completion_token_parameter="max_tokens",
    )


def _build_default_judge_client(config: RagEvalConfig) -> RagEvalJudgeClient | None:
    if config.disable_llm_judge or not config.judge_provider or not config.judge_model:
        return None
    provider = str(config.judge_provider or "").strip().lower()
    env_name, base_url = _judge_env_and_base_url(provider)
    import os

    api_key = os.getenv(env_name)
    if not api_key:
        return None
    return RoundRobinGroqClient(
        keys=[ApiKey(alias=provider, value=api_key)],
        model=config.judge_model,
        max_retries=1,
        key_tokens_per_minute=0,
        key_requests_per_minute=0,
        client_factory=lambda key, timeout: OpenAICompatibleClient(
            api_key=key.value,
            base_url=base_url,
            timeout_s=timeout,
            token_parameter="max_tokens",
        ),
        provider_name=provider,
        completion_token_parameter="max_tokens",
    )


def _judge_env_and_base_url(provider: str) -> tuple[str, str]:
    if provider == "mimo":
        return "MIMO_API_KEY", DEFAULT_MIMO_BASE_URL
    if provider == "deepseek":
        return "DS_API_KEY", DEFAULT_DEEPSEEK_BASE_URL
    if provider == "groq":
        return "GROQ_API_KEY", DEFAULT_GROQ_BASE_URL
    if provider == "openai":
        return "OPENAI_API_KEY", DEFAULT_OPENAI_BASE_URL
    return f"{provider.upper()}_API_KEY", DEFAULT_OPENAI_BASE_URL


def _mode_query(item: RagEvalItem) -> str:
    if item.mode == "dictionary" and not item.query.strip().startswith("/dict"):
        return f"/dict {item.query}"
    return item.query


def _answer_from_service_result(result: ChatServiceResult) -> str:
    choices = result.response.get("choices") or []
    if choices and isinstance(choices[0], dict):
        message = choices[0].get("message") or {}
        return str(message.get("content") or "")
    return str(result.generation.answer or "")


def _query_plan_from_service_result(result: ChatServiceResult) -> dict[str, Any]:
    if isinstance(result.response.get("query_plan"), dict):
        return dict(result.response["query_plan"])
    metadata = result.retrieval_metadata or result.response.get("rag", {}).get("retrieval_metadata") or {}
    plan = metadata.get("query_plan") if isinstance(metadata, dict) else None
    return dict(plan) if isinstance(plan, dict) else {}


def _parse_judge_json(text: str) -> dict[str, Any]:
    text = text.strip()
    if not text:
        raise ValueError("judge returned empty response")
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", text, flags=re.DOTALL)
        if not match:
            raise
        parsed = json.loads(match.group(0))
    if not isinstance(parsed, dict):
        raise ValueError("judge response must be a JSON object")
    return parsed


def _has_citation(answer: str, retrieved_doc_ids: Sequence[str]) -> bool:
    if not answer.strip():
        return False
    if re.search(r"\[[^\]]+\]", answer):
        return True
    return any(doc_id and doc_id in answer for doc_id in retrieved_doc_ids)


def _expected_bool(not_applicable: bool, value: bool) -> bool | None:
    return None if not_applicable else bool(value)


def _is_failure(result: RagEvalResult) -> bool:
    if result.heuristic_scores.get("all_required_passed") is False:
        return True
    if result.judge_error:
        return True
    if result.judge_scores and str(result.judge_scores.get("verdict") or "").lower() == "fail":
        return True
    return False


def _sanitize_private_query_plan(query_plan: dict[str, Any]) -> dict[str, Any]:
    structured = query_plan.get("structured_evidence") if isinstance(query_plan.get("structured_evidence"), dict) else {}
    sanitized: dict[str, Any] = {}
    for key in ("intent", "confidence", "answer_style"):
        if key in query_plan:
            sanitized[key] = query_plan[key]
    sanitized["schema_gaps"] = _string_list(query_plan.get("schema_gaps"))
    if structured:
        sanitized["structured_evidence"] = {
            "enabled": structured.get("enabled"),
            "matched_doc_types": _string_list(structured.get("matched_doc_types")),
            "matched_doc_count": int(structured.get("matched_doc_count") or 0),
        }
    return sanitized


def _sanitize_private_privacy(privacy: dict[str, Any]) -> dict[str, Any]:
    safe_keys = {
        "session_taint",
        "turn_tier",
        "provider_requested",
        "model_requested",
        "provider_selected",
        "model_selected",
        "backend_id",
        "backend_kind",
        "provider_allowed",
        "external_blocked",
        "reason",
        "redaction_required",
    }
    sanitized = {key: privacy.get(key) for key in safe_keys if key in privacy}
    state = privacy.get("state")
    if isinstance(state, dict):
        sanitized["state"] = {
            key: state.get(key)
            for key in (
                "session_id",
                "session_taint",
                "max_seen_tier",
                "private_seen",
                "external_blocked",
                "last_turn_tier",
                "reason",
            )
            if key in state
        }
    return sanitized


def _sanitize_private_judge_scores(judge_scores: dict[str, Any] | None) -> dict[str, Any] | None:
    if judge_scores is None:
        return None
    sanitized: dict[str, Any] = {}
    for key, value in judge_scores.items():
        if key == "issues":
            sanitized[key] = ["redacted_private_judge_issues"] if value else []
        elif key == "verdict":
            verdict = str(value or "").strip().lower()
            sanitized[key] = verdict if verdict in {"pass", "partial", "fail"} else "unknown"
        elif isinstance(value, (int, float, bool)) or value is None:
            sanitized[key] = value
        elif isinstance(value, list) and key.endswith("_codes"):
            sanitized[key] = [str(item) for item in value if str(item).strip()]
        else:
            sanitized[key] = "redacted_private_judge_field"
    return sanitized


def _extract_context_citations(prompt: str) -> list[str]:
    ids: list[str] = []
    for match in re.finditer(r"^\[([^\]]+)\]\s*$", prompt, flags=re.MULTILINE):
        doc_id = match.group(1).strip()
        if doc_id and doc_id not in ids:
            ids.append(doc_id)
    return ids


def _timestamped_output_dir() -> Path:
    return DEFAULT_RAG_EVAL_OUTPUT_ROOT / time.strftime("%Y%m%d-%H%M%S")


def _string_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [value.strip()] if value.strip() else []
    if isinstance(value, Sequence):
        return [str(item).strip() for item in value if str(item).strip()]
    return [str(value).strip()] if str(value).strip() else []


def _optional_str(value: Any) -> str | None:
    text = str(value or "").strip()
    return text or None


def _safe_error_text(exc: Exception) -> str:
    text = str(exc)
    return text[:240] if text else exc.__class__.__name__
