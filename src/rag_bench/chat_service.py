from __future__ import annotations

import json
import math
import re
import time
import unicodedata
import uuid
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Callable, Protocol, Sequence

from rag_bench.benchmarks import load_benchmark
from rag_bench.dictionary import (
    DEFAULT_DICTIONARY_ARTIFACT,
    DEFAULT_DICTIONARY_LETTERS,
    DEFAULT_DICTIONARY_SOURCE_DIR,
    DictionaryLoadResult,
    load_dictionary_documents,
    normalize_spaces,
)
from rag_bench.dictionary_agent_tools import dictionary_tool_plan_payload, render_dictionary_tool_plan_prompt
from rag_bench.dictionary_query_planner import (
    DictionaryQueryIntent,
    DictionaryQueryPlan,
    annotate_and_rank_dictionary_hits,
    dictionary_plan_prompt_instructions,
    merge_planned_dictionary_results,
    plan_dictionary_query,
)
from rag_bench.groq_client import FallbackChatClient, GenerationResult, OpenAICompatibleClient, RoundRobinGroqClient
from rag_bench.prompts import RESPONSE_FORMAT_GUIDANCE, SYSTEM_PROMPT
from rag_bench.privacy import (
    BackendDescriptor,
    BackendKind,
    ConversationPrivacyState,
    DataTier,
    PrivateBackendPolicy,
    PrivacyDecision,
    PrivacyRouteError,
    classify_backend,
    data_tier_for_hit,
    enforce_privacy_route,
    include_private_source_text_from_env,
    max_data_tier,
    normalize_data_tier,
    safe_source_payload,
)
from rag_bench.retriever_registry import create_retriever, get_retriever_spec, normalize_retriever_id
from rag_bench.retrievers import EmptyCorpusRetriever, Retriever
from rag_bench.secrets import ApiKey, load_env_api_key_chain, load_groq_keys
from rag_bench.structured_evidence import (
    StructuredEvidenceIndex,
    load_structured_evidence_jsonl,
    load_structured_evidence_markdown,
)
from rag_bench.types import BenchmarkData, Query, RetrievalHit, RetrievalResult


DEFAULT_PROXY_MODEL_ID = "rag-scifact-bm25"
DEFAULT_CHAT_MODEL = "qwen/qwen3-32b"
DEFAULT_CHAT_MODELS = (DEFAULT_CHAT_MODEL, "llama-3.1-8b-instant")
DEFAULT_MIMO_BASE_URL = "https://token-plan-sgp.xiaomimimo.com/v1"
DEFAULT_MIMO_PAYG_BASE_URL = "https://api.xiaomimimo.com/v1"
DEFAULT_MIMO_MODELS = ("mimo-v2.5-pro", "mimo-v2.5")
DEFAULT_MIMO_AUTH_HEADER = "both"
DEFAULT_CHAT_RETRIEVERS = (
    "bm25",
    "tfidf",
    "keyword-match",
    "agent",
    "graph-bm25",
    "dictionary-graph",
    "image-digits",
)
MIN_RETRIEVAL_DISPLAY_SCORE = 5e-4
DICTIONARY_LIST_FALLBACK_MIN_HITS = 8
CONTEXT_SEPARATOR = "\n\n---\n\n"
ALIAS_EDGE_MIN_CONFIDENCE = 0.5
PROMPT_SECTION_SCHEMA_VERSION = "prompt_sections_v1"
AGENT_TOOL_SCHEMA_VERSION = "agent_retrieval_tools_v1"
AGENT_TOOL_ALLOWLIST = (
    "dictionary.lookup",
    "text.multi_query",
    "text.bm25",
    "text.graph_bm25",
    "text.keyword",
)


class ChatGenerationClient(Protocol):
    key_usage_counts: Any

    def generate(
        self,
        messages: list[dict[str, str]],
        *,
        model: str | None = None,
        temperature: float = 0.0,
        max_completion_tokens: int = 512,
    ) -> GenerationResult: ...

    def rate_limit_snapshot(self) -> dict[str, dict[str, float | int | str]]: ...


@dataclass(frozen=True)
class ChatProxyConfig:
    bench: str = "scifact"
    retriever: str = "bm25"
    top_k: int = 3
    groq_keys_path: Path = Path(".secrets/groq_key.env")
    model: str = DEFAULT_CHAT_MODEL
    model_id: str = DEFAULT_PROXY_MODEL_ID
    available_models: tuple[str, ...] = DEFAULT_CHAT_MODELS
    mimo_enabled: bool = False
    mimo_env_file: Path = Path(".secrets/.env")
    mimo_api_key_var: str = "MIMO_API_KEY"
    mimo_base_url: str = DEFAULT_MIMO_BASE_URL
    mimo_payg_base_url: str = DEFAULT_MIMO_PAYG_BASE_URL
    mimo_auth_header: str = DEFAULT_MIMO_AUTH_HEADER
    mimo_models: tuple[str, ...] = DEFAULT_MIMO_MODELS
    mimo_key_tokens_per_minute: int = 0
    mimo_key_requests_per_minute: int = 0
    available_retrievers: tuple[str, ...] = DEFAULT_CHAT_RETRIEVERS
    vector_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    max_retries: int = 2
    max_completion_tokens: int = 4096
    temperature: float = 0.0
    max_context_chars: int = 2500
    allow_large_bench: bool = False
    key_tokens_per_minute: int = 6000
    key_requests_per_minute: int = 30
    rate_limit_scope: str = "per-key"
    history_messages: int = 6
    image_top_k: int = 5
    dictionary_artifact: Path | None = DEFAULT_DICTIONARY_ARTIFACT
    dictionary_source_dir: Path | None = DEFAULT_DICTIONARY_SOURCE_DIR
    dictionary_letters: tuple[str, ...] = DEFAULT_DICTIONARY_LETTERS
    dictionary_top_k: int = 5
    dictionary_required: bool = False
    enable_dictionary_query_planner: bool = True
    enable_alias_extractive_answer: bool = True
    enable_structured_evidence: bool = False
    structured_evidence_jsonl: Path | None = None
    structured_evidence_md: Path | None = None
    allow_external_semi_private: bool = False
    backend_id: str | None = None
    backend_kind: str | None = None
    backend_base_url: str | None = None
    trusted_private_backends: tuple[str, ...] = ()
    trusted_private_models: tuple[str, ...] = ()
    backend_model_allowlist: dict[str, tuple[str, ...]] = field(default_factory=dict)
    trusted_local_models: tuple[str, ...] = ()
    include_private_source_text: bool = field(default_factory=include_private_source_text_from_env)


@dataclass(frozen=True)
class RetrievalScoreControls:
    min_score: float | None = None
    max_score: float | None = None
    sort_by_score: bool = False

    @property
    def active(self) -> bool:
        return self.min_score is not None or self.max_score is not None or self.sort_by_score

    @property
    def has_score_range(self) -> bool:
        return self.min_score is not None or self.max_score is not None


@dataclass(frozen=True)
class PromptSection:
    section_id: str
    title: str
    content: str
    enabled: bool = True


@dataclass
class ChatServiceResult:
    response: dict[str, Any]
    generation: GenerationResult
    hits: list[RetrievalHit]
    retrieval_latency_s: float
    retrieval_metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class AliasEvidence:
    aliases: list[str]
    source_doc_ids: list[str]
    evidence_count: int
    has_explicit_alias_evidence: bool


@dataclass
class RagChatService:
    config: ChatProxyConfig
    benchmark: BenchmarkData
    retriever: Retriever
    llm: ChatGenerationClient
    started_at_s: float = field(default_factory=time.time)
    retrievers: dict[str, Retriever] = field(default_factory=dict)
    dictionary_status: dict[str, Any] = field(default_factory=dict)
    structured_evidence_index: StructuredEvidenceIndex | None = None
    privacy_states: dict[str, ConversationPrivacyState] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.retrievers:
            self.retrievers = {self.retriever.name: self.retriever}

    @classmethod
    def from_config(
        cls,
        config: ChatProxyConfig,
        *,
        benchmark_loader: Callable[..., BenchmarkData] = load_benchmark,
        llm_factory: Callable[[list[ApiKey]], ChatGenerationClient] | None = None,
    ) -> "RagChatService":
        keys = load_groq_keys(config.groq_keys_path)
        benchmark = benchmark_loader(config.bench, limit=None, allow_large=config.allow_large_bench)
        llm = llm_factory(keys) if llm_factory is not None else _build_llm(config, keys)
        dictionary = _load_dictionary(config)
        retrievers = _build_retrievers(config, benchmark, llm=llm, dictionary=dictionary)
        retriever = _default_retriever(config, retrievers)
        structured_evidence_index = _load_structured_evidence_index(config)
        return cls(
            config=config,
            benchmark=benchmark,
            retriever=retriever,
            llm=llm,
            retrievers=retrievers,
            dictionary_status=dictionary.status,
            structured_evidence_index=structured_evidence_index,
        )

    def answer(
        self,
        messages: list[dict[str, Any]],
        *,
        request_model: str | None = None,
        request_retriever: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
        top_k: int | None = None,
        image_top_k: int | None = None,
        response_mode: str | None = None,
        image_rewrite: bool | None = None,
        language: str | None = None,
        memory: bool | None = None,
        score_min: float | None = None,
        score_max: float | None = None,
        sort_by_score: bool | None = None,
        session_id: str | None = None,
        privacy_state: dict[str, Any] | None = None,
        reset_privacy: bool = False,
    ) -> ChatServiceResult:
        response_model, generation_model = self.resolve_request_model(request_model)
        backend = self._backend_for_model(generation_model)
        session_state = self._privacy_state_for_request(
            session_id=session_id,
            privacy_state=privacy_state,
            reset=reset_privacy,
            messages=messages,
        )
        user_message_tier = _messages_data_tier(messages)
        question = last_user_text(messages)
        response_language = _normalize_response_language(language)
        score_controls = _normalize_retrieval_score_controls(score_min, score_max, sort_by_score)
        use_memory = True if memory is None else bool(memory)
        history_messages = self.config.history_messages if use_memory else 0
        command = parse_chat_command(question)
        mode = _normalize_response_mode(response_mode)
        if command and command[0] == "img":
            mode = "image"
            question = command[1] or "digit image"
        elif command and command[0] == "dict":
            mode = "dictionary"
            question = command[1] or question

        if mode == "image":
            image_query, rewrite_metadata = self._image_query(
                question,
                generation_model,
                image_rewrite=self._privacy_allowed_image_rewrite(
                    image_rewrite=image_rewrite,
                    backend=backend,
                    session_state=session_state,
                    user_message_tier=user_message_tier,
                ),
            )
            retriever = self.resolve_request_retriever("image-digits")
            request_image_top_k = _clamp_top_k(image_top_k if image_top_k is not None else top_k, fallback=self.config.image_top_k)
            retrieval = retriever.search(Query(query_id="chat-img", text=image_query), request_image_top_k)
            retrieval, score_filter_metadata = _apply_retrieval_score_controls(
                retrieval,
                score_controls,
                max_hits=request_image_top_k,
            )
            privacy_decision = self._record_no_generation_privacy(
                session_state,
                retrieval.hits,
                backend=backend,
                user_message_tier=user_message_tier,
            )
            if image_rewrite and not rewrite_metadata.get("image_query_rewrite"):
                rewrite_metadata.setdefault("image_query_rewrite_privacy_blocked", True)
            generation = GenerationResult(
                answer=_format_image_answer(image_query, retrieval.hits, language=response_language),
                key_alias=None,
                attempted_aliases=[],
                latency_s=0.0,
                retry_count=0,
                prompt_tokens=0,
                completion_tokens=0,
                total_tokens=0,
                estimated_tokens=0,
            )
            retrieval_metadata = {
                **retrieval.metadata,
                "command": "/img",
                "response_mode": "image",
                "raw_query": last_user_text(messages),
                "image_query": image_query,
                "image_top_k": request_image_top_k,
                "language": response_language,
                "memory": use_memory,
                **score_filter_metadata,
                **rewrite_metadata,
            }
            response = self._build_response(
                answer=generation.answer,
                generation=generation,
                hits=retrieval.hits,
                retrieval_latency_s=retrieval.latency_s,
                retrieval_metadata=retrieval_metadata,
                retriever=retriever,
                top_k=request_image_top_k,
                response_model=response_model,
                generation_model=generation_model,
                score_controls=score_controls,
                privacy_decision=privacy_decision,
                session_state=session_state,
            )
            return ChatServiceResult(
                response=response,
                generation=generation,
                hits=retrieval.hits,
                retrieval_latency_s=retrieval.latency_s,
                retrieval_metadata=retrieval_metadata,
            )

        if mode == "dictionary":
            retriever = self.resolve_request_retriever("dictionary-graph")
            request_top_k = _clamp_top_k(top_k, fallback=self.config.dictionary_top_k)
            query_plan = plan_dictionary_query(question) if self.config.enable_dictionary_query_planner else None
            dictionary_hit_limit = _dictionary_list_query_hit_limit(query_plan, request_top_k)
            retrieval = retriever.search(Query(query_id="chat-dict", text=question), dictionary_hit_limit)
            structured_result = None
            if query_plan is not None and self.structured_evidence_index is not None:
                structured_result = self.structured_evidence_index.search(
                    question,
                    intent=query_plan.intent.value,
                    terms=query_plan.target_terms,
                    top_k=request_top_k,
                )
                if structured_result.hits:
                    query_plan = query_plan.with_structured_evidence(structured_result.to_metadata())
                    retrieval.hits = merge_planned_dictionary_results(retrieval.hits, [structured_result.hits])
                else:
                    query_plan = query_plan.with_structured_evidence(structured_result.to_metadata())
            if query_plan is not None:
                extra_results = _planned_dictionary_extra_results(
                    retriever,
                    query_plan,
                    original_query=question,
                    request_top_k=dictionary_hit_limit,
                    query_id_prefix="chat-dict-plan",
                )
                if extra_results:
                    retrieval.hits = merge_planned_dictionary_results(retrieval.hits, extra_results)
            if query_plan is not None:
                retrieval.hits = annotate_and_rank_dictionary_hits(retrieval.hits, query_plan, max_hits=dictionary_hit_limit)
            retrieval, score_filter_metadata = _apply_retrieval_score_controls(
                retrieval,
                score_controls,
                max_hits=dictionary_hit_limit,
            )
            alias_evidence = (
                extract_alias_evidence_from_hits(retrieval.hits, target_terms=query_plan.target_terms)
                if query_plan is not None and query_plan.intent == DictionaryQueryIntent.ALIAS
                else None
            )
            redirect_preserve_terms = _dictionary_redirect_preserve_terms(query_plan)
            retrieval.hits = _canonicalize_dictionary_redirect_hits(
                retrieval.hits,
                preserve_headword_terms=redirect_preserve_terms,
            )
            retrieval_metadata = {
                **retrieval.metadata,
                "command": "/dict" if command and command[0] == "dict" else None,
                "response_mode": "dictionary",
                "raw_query": last_user_text(messages),
                "dictionary_status": self.dictionary_status,
                "language": response_language,
                "memory": use_memory,
                **score_filter_metadata,
            }
            if structured_result is not None:
                retrieval_metadata["structured_evidence"] = structured_result.to_metadata()
            if query_plan is not None:
                query_plan_payload = query_plan.to_payload()
                if query_plan.intent == DictionaryQueryIntent.ALIAS:
                    alias_summary = _alias_evidence_summary(retrieval.hits, alias_evidence=alias_evidence)
                    if not self.config.enable_alias_extractive_answer:
                        alias_answer_mode = "llm_prompt"
                    elif alias_evidence is not None and alias_evidence.has_explicit_alias_evidence:
                        alias_answer_mode = "deterministic_extractive"
                    else:
                        alias_answer_mode = "deterministic_no_alias"
                    alias_summary["alias_answer_mode"] = alias_answer_mode
                    query_plan_payload.update(
                        {
                            "alias_evidence_count": alias_summary["alias_evidence_count"],
                            "alias_evidence_doc_count": alias_summary["alias_evidence_doc_count"],
                            "alias_answer_mode": alias_answer_mode,
                        }
                    )
                    retrieval_metadata["alias_evidence"] = alias_summary
                    retrieval_metadata["alias_answer_style_used"] = query_plan.answer_style
                    retrieval_metadata["alias_answer_mode"] = alias_answer_mode
                    if alias_answer_mode.startswith("deterministic_"):
                        retrieval_metadata["generator_provider"] = "deterministic_alias"
                retrieval_metadata["query_plan"] = query_plan_payload
                retrieval_metadata["dictionary_tool_plan"] = dictionary_tool_plan_payload(
                    query_plan,
                    original_query=question,
                )
            if retrieval.hits:
                privacy_decision = self._enforce_generation_privacy(
                    backend=backend,
                    session_state=session_state,
                    hits=retrieval.hits,
                    user_message_tier=user_message_tier,
                )
                if (
                    self.config.enable_alias_extractive_answer
                    and query_plan is not None
                    and query_plan.intent == DictionaryQueryIntent.ALIAS
                    and alias_evidence is not None
                ):
                    answer = _format_alias_answer(alias_evidence, retrieval.hits, language=language)
                    generation = GenerationResult(
                        answer=answer,
                        key_alias="deterministic_alias",
                        attempted_aliases=[],
                        latency_s=0.0,
                        retry_count=0,
                        prompt_tokens=0,
                        completion_tokens=0,
                        total_tokens=0,
                        estimated_tokens=0,
                    )
                else:
                    prompt_sections = build_dictionary_rag_prompt_sections(
                        messages,
                        retrieval.hits,
                        query=question,
                        max_context_chars=self.config.max_context_chars,
                        history_messages=history_messages,
                        query_plan=query_plan,
                    )
                    retrieval_metadata["prompt_sections"] = _prompt_sections_metadata(prompt_sections)
                    prompt_messages = _build_dictionary_rag_messages_from_sections(
                        prompt_sections,
                        language=response_language,
                    )
                    generation = self.llm.generate(
                        prompt_messages,
                        model=generation_model,
                        temperature=self.config.temperature if temperature is None else temperature,
                        max_completion_tokens=self.config.max_completion_tokens if max_tokens is None else max_tokens,
                    )
                    if generation.error:
                        raise RuntimeError(generation.error)
                    grounded_redirect_answer = _format_dictionary_redirect_lookup_fallback_answer(
                        question,
                        retrieval.hits,
                        retrieval_metadata,
                        language=response_language,
                    )
                    grounded_category_answer = _format_dictionary_category_fallback_answer(
                        question,
                        retrieval.hits,
                        retrieval_metadata,
                        language=response_language,
                    )
                    grounded_plural_phrase_answer = _format_dictionary_plural_phrase_list_fallback_answer(
                        question,
                        retrieval.hits,
                        retrieval_metadata,
                        language=response_language,
                    )
                    if grounded_redirect_answer:
                        generation = replace(generation, answer=grounded_redirect_answer)
                        retrieval_metadata["dictionary_redirect_lookup_fallback"] = True
                    elif grounded_category_answer:
                        generation = replace(generation, answer=grounded_category_answer)
                        retrieval_metadata["dictionary_category_fallback"] = True
                    elif grounded_plural_phrase_answer:
                        generation = replace(generation, answer=grounded_plural_phrase_answer)
                        retrieval_metadata["dictionary_plural_phrase_list_fallback"] = True
                    if _looks_like_grounding_refusal(generation.answer) and any(
                        _hit_has_direct_dictionary_match(hit) for hit in retrieval.hits
                    ):
                        grounded_redirect_answer = _format_dictionary_redirect_lookup_fallback_answer(
                            question,
                            retrieval.hits,
                            retrieval_metadata,
                            language=response_language,
                        )
                        grounded_occurrence_answer = grounded_redirect_answer or _format_dictionary_occurrence_fallback_answer(
                            question,
                            retrieval.hits,
                            retrieval_metadata,
                            language=response_language,
                        )
                        if grounded_occurrence_answer:
                            generation = replace(generation, answer=grounded_occurrence_answer)
                            retrieval_metadata["dictionary_refusal_fallback"] = True
                            retrieval_metadata["dictionary_direct_refusal_fallback"] = True
                            if grounded_redirect_answer:
                                retrieval_metadata["dictionary_redirect_lookup_fallback"] = True
                    answer = _format_dictionary_answer(retrieval.hits, generation.answer)
            else:
                privacy_decision = self._record_no_generation_privacy(
                    session_state,
                    retrieval.hits,
                    backend=backend,
                    user_message_tier=user_message_tier,
                )
                generation = GenerationResult(
                    answer="",
                    key_alias=None,
                    attempted_aliases=[],
                    latency_s=0.0,
                    retry_count=0,
                    prompt_tokens=0,
                    completion_tokens=0,
                    total_tokens=0,
                    estimated_tokens=0,
                )
                answer = _localized_no_dictionary_answer(question, response_language)
            response = self._build_response(
                answer=answer,
                generation=generation,
                hits=retrieval.hits,
                retrieval_latency_s=retrieval.latency_s,
                retrieval_metadata=retrieval_metadata,
                retriever=retriever,
                top_k=request_top_k,
                response_model=response_model,
                generation_model=generation_model,
                score_controls=score_controls,
                privacy_decision=privacy_decision,
                session_state=session_state,
            )
            return ChatServiceResult(
                response=response,
                generation=generation,
                hits=retrieval.hits,
                retrieval_latency_s=retrieval.latency_s,
                retrieval_metadata=retrieval_metadata,
            )

        retriever = self.resolve_text_request_retriever(request_retriever)
        retriever, retriever_privacy_metadata = self._safe_text_retriever_for_privacy(
            retriever,
            backend=backend,
            session_state=session_state,
            user_message_tier=user_message_tier,
        )
        request_top_k = _clamp_top_k(top_k, fallback=self.config.top_k)
        if retriever.name == "agent":
            if self._llm_tool_blocked(
                backend=backend,
                session_state=session_state,
                user_message_tier=user_message_tier,
            ):
                retrieval = retriever.search(Query(query_id="chat-agent", text=question), request_top_k)
                retrieval.metadata.update(
                    {
                        "agent_mode": True,
                        "agent_schema": AGENT_TOOL_SCHEMA_VERSION,
                        "agent_llm_calls": 0,
                        "agent_planner_privacy_blocked": True,
                        "agent_planner_fallback": "privacy_blocked",
                    }
                )
            else:
                retrieval = self._agent_search(
                    retriever,
                    question,
                    request_top_k,
                    generation_model=generation_model,
                )
        elif retriever.name == "keyword-match":
            if self._llm_tool_blocked(
                backend=backend,
                session_state=session_state,
                user_message_tier=user_message_tier,
            ):
                retrieval = retriever.search(Query(query_id="chat-keyword", text=question), request_top_k)
                retrieval.metadata.update(
                    {
                        "keyword_llm_calls": 0,
                        "keyword_llm_privacy_blocked": True,
                        "keyword_query_variants": [question],
                    }
                )
            else:
                retrieval = self._keyword_search(
                    retriever,
                    question,
                    request_top_k,
                    generation_model=generation_model,
                )
        else:
            retrieval = retriever.search(Query(query_id="chat", text=question), request_top_k)
        retrieval, score_filter_metadata = _apply_retrieval_score_controls(
            retrieval,
            score_controls,
            max_hits=request_top_k,
        )
        dictionary_fallback = (
            None
            if isinstance(retrieval.metadata, dict) and retrieval.metadata.get("agent_dictionary_metadata")
            else self._text_dictionary_fallback(
                question,
                top_k=request_top_k,
                primary_retriever=retriever,
                primary_retrieval=retrieval,
            )
        )
        dictionary_score_filter_metadata: dict[str, Any] = {}
        if dictionary_fallback is not None:
            dictionary_fallback_hit_limit = _dictionary_list_query_hit_limit(
                dictionary_fallback.metadata.get("query_plan") if isinstance(dictionary_fallback.metadata, dict) else None,
                request_top_k,
            )
            dictionary_fallback, dictionary_score_filter_metadata = _apply_retrieval_score_controls(
                dictionary_fallback,
                score_controls,
                max_hits=dictionary_fallback_hit_limit,
            )
            if not dictionary_fallback.hits:
                dictionary_fallback = None
        prompt_hit_limit = request_top_k
        if dictionary_fallback is not None:
            prompt_hit_limit = _dictionary_list_query_hit_limit(
                dictionary_fallback.metadata.get("query_plan") if isinstance(dictionary_fallback.metadata, dict) else None,
                request_top_k,
            )
        prompt_hits = _merge_text_and_dictionary_hits(
            retrieval.hits,
            dictionary_fallback.hits if dictionary_fallback else [],
            max_hits=prompt_hit_limit,
            preserve_dictionary_redirect_terms=_dictionary_redirect_preserve_terms(
                dictionary_fallback.metadata.get("query_plan")
                if dictionary_fallback is not None and isinstance(dictionary_fallback.metadata, dict)
                else None
            ),
        )
        privacy_decision = self._enforce_generation_privacy(
            backend=backend,
            session_state=session_state,
            hits=prompt_hits,
            user_message_tier=user_message_tier,
        )
        dictionary_prompt_metadata = (
            dictionary_fallback.metadata
            if dictionary_fallback is not None
            else retrieval.metadata.get("agent_dictionary_metadata")
            if isinstance(retrieval.metadata, dict)
            else None
        )
        prompt_sections = build_chat_rag_prompt_sections(
            messages,
            prompt_hits,
            max_context_chars=self.config.max_context_chars,
            history_messages=history_messages,
            dictionary_fallback_metadata=dictionary_prompt_metadata,
        )
        prompt_messages = _build_chat_rag_messages_from_sections(prompt_sections, language=response_language)
        generation = self.llm.generate(
            prompt_messages,
            model=generation_model,
            temperature=self.config.temperature if temperature is None else temperature,
            max_completion_tokens=self.config.max_completion_tokens if max_tokens is None else max_tokens,
        )
        if generation.error:
            raise RuntimeError(generation.error)
        dictionary_refusal_fallback_used = False
        dictionary_internal_query_leak_fallback_used = False
        dictionary_category_fallback_used = False
        dictionary_redirect_lookup_fallback_used = False
        if dictionary_fallback is not None:
            grounded_redirect_answer = _format_dictionary_redirect_lookup_fallback_answer(
                question,
                dictionary_fallback.hits,
                dictionary_fallback.metadata,
                language=response_language,
            )
            grounded_category_answer = _format_dictionary_category_fallback_answer(
                question,
                dictionary_fallback.hits,
                dictionary_fallback.metadata,
                language=response_language,
            )
            grounded_plural_phrase_answer = _format_dictionary_plural_phrase_list_fallback_answer(
                question,
                dictionary_fallback.hits,
                dictionary_fallback.metadata,
                language=response_language,
            )
            if grounded_redirect_answer:
                generation = replace(generation, answer=grounded_redirect_answer)
                dictionary_redirect_lookup_fallback_used = True
            elif grounded_category_answer:
                generation = replace(generation, answer=grounded_category_answer)
                dictionary_category_fallback_used = True
            elif grounded_plural_phrase_answer:
                generation = replace(generation, answer=grounded_plural_phrase_answer)
                retrieval.metadata["dictionary_plural_phrase_list_fallback"] = True
        if (
            dictionary_fallback is not None
            and not dictionary_redirect_lookup_fallback_used
            and _looks_like_grounding_refusal(generation.answer)
        ):
            grounded_redirect_answer = _format_dictionary_redirect_lookup_fallback_answer(
                question,
                dictionary_fallback.hits,
                dictionary_fallback.metadata,
                language=response_language,
            )
            grounded_occurrence_answer = grounded_redirect_answer or _format_dictionary_occurrence_fallback_answer(
                question,
                dictionary_fallback.hits,
                dictionary_fallback.metadata,
                language=response_language,
            )
            if grounded_occurrence_answer:
                generation = replace(generation, answer=grounded_occurrence_answer)
                dictionary_refusal_fallback_used = True
                if grounded_redirect_answer:
                    retrieval.metadata["dictionary_redirect_lookup_fallback"] = True
        if (
            dictionary_fallback is not None
            and not dictionary_refusal_fallback_used
            and _looks_like_internal_query_leak(generation.answer)
        ):
            grounded_disambiguation_answer = _format_dictionary_disambiguation_fallback_answer(
                question,
                dictionary_fallback.hits,
                dictionary_fallback.metadata,
                language=response_language,
            )
            if grounded_disambiguation_answer:
                generation = replace(generation, answer=grounded_disambiguation_answer)
                dictionary_internal_query_leak_fallback_used = True

        combined_hits = list(prompt_hits)
        retrieval_metadata = {**retriever_privacy_metadata, **retrieval.metadata, **score_filter_metadata}
        retrieval_metadata["prompt_sections"] = _prompt_sections_metadata(prompt_sections)
        if dictionary_fallback is not None:
            retrieval_metadata.update(
                {
                    "dictionary_fallback": True,
                    "dictionary_fallback_latency_s": dictionary_fallback.latency_s,
                    "dictionary_fallback_count": len(dictionary_fallback.hits),
                    "dictionary_fallback_metadata": dictionary_fallback.metadata,
                }
            )
            if dictionary_refusal_fallback_used:
                retrieval_metadata["dictionary_refusal_fallback"] = True
            if dictionary_internal_query_leak_fallback_used:
                retrieval_metadata["dictionary_internal_query_leak_fallback"] = True
            if dictionary_category_fallback_used:
                retrieval_metadata["dictionary_category_fallback"] = True
            if dictionary_redirect_lookup_fallback_used:
                retrieval_metadata["dictionary_redirect_lookup_fallback"] = True
            if dictionary_score_filter_metadata:
                retrieval_metadata["dictionary_fallback_score_filter"] = dictionary_score_filter_metadata["score_filter"]
        if mode == "text_image":
            image_query, image_query_metadata = self._image_query(
                f"Question: {question}\nAnswer: {generation.answer}",
                generation_model,
                image_rewrite=True if image_rewrite is None else image_rewrite,
            )
            image_retriever = self.resolve_request_retriever("image-digits")
            request_image_top_k = _clamp_top_k(image_top_k, fallback=self.config.image_top_k)
            image_retrieval = image_retriever.search(Query(query_id="chat-img", text=image_query), request_image_top_k)
            image_retrieval, image_score_filter_metadata = _apply_retrieval_score_controls(
                image_retrieval,
                score_controls,
                max_hits=request_image_top_k,
            )
            combined_hits.extend(image_retrieval.hits)
            retrieval_metadata.update(
                {
                    "response_mode": "text_image",
                    "image_retriever": image_retriever.name,
                    "image_retrieval_latency_s": image_retrieval.latency_s,
                    "image_top_k": request_image_top_k,
                    "image_query": image_query,
                    "image_retrieval_metadata": image_retrieval.metadata,
                    "language": response_language,
                    "memory": use_memory,
                    **image_query_metadata,
                }
            )
            if image_score_filter_metadata:
                retrieval_metadata["image_score_filter"] = image_score_filter_metadata["score_filter"]
        else:
            retrieval_metadata.setdefault("response_mode", "text")
        retrieval_metadata.setdefault("language", response_language)
        retrieval_metadata.setdefault("memory", use_memory)

        response = self._build_response(
            answer=generation.answer,
            generation=generation,
            hits=combined_hits,
            retrieval_latency_s=retrieval.latency_s,
            retrieval_metadata=retrieval_metadata,
            retriever=retriever,
            top_k=request_top_k,
            response_model=response_model,
            generation_model=generation_model,
            score_controls=score_controls,
            privacy_decision=privacy_decision,
            session_state=session_state,
        )
        return ChatServiceResult(
            response=response,
            generation=generation,
            hits=combined_hits,
            retrieval_latency_s=retrieval.latency_s,
            retrieval_metadata=retrieval_metadata,
        )

    def _build_response(
        self,
        *,
        answer: str,
        generation: GenerationResult,
        hits: list[RetrievalHit],
        retrieval_latency_s: float,
        retrieval_metadata: dict[str, Any] | None = None,
        retriever: Retriever | None = None,
        top_k: int | None = None,
        response_model: str | None = None,
        generation_model: str | None = None,
        score_controls: RetrievalScoreControls | None = None,
        privacy_decision: PrivacyDecision | None = None,
        session_state: ConversationPrivacyState | None = None,
    ) -> dict[str, Any]:
        created = int(time.time())
        completion_id = f"chatcmpl-{uuid.uuid4().hex}"
        response_model = response_model or self.config.model_id
        generation_model = generation_model or self.config.model
        retriever = retriever or self.retriever
        privacy_payload = (
            privacy_decision.to_payload(session_state=session_state)
            if privacy_decision is not None
            else ({"state": session_state.to_payload()} if session_state is not None else {})
        )
        return {
            "id": completion_id,
            "object": "chat.completion",
            "created": created,
            "model": response_model,
            "privacy": privacy_payload,
            **({"query_plan": retrieval_metadata["query_plan"]} if isinstance(retrieval_metadata, dict) and "query_plan" in retrieval_metadata else {}),
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": answer},
                    "finish_reason": "stop",
                }
            ],
            "usage": {
                "prompt_tokens": generation.prompt_tokens,
                "completion_tokens": generation.completion_tokens,
                "total_tokens": generation.total_tokens,
                "estimated_tokens": generation.estimated_tokens,
            },
            "rag": {
                "benchmark": self.benchmark.name,
                "dataset_id": self.benchmark.dataset_id,
                "retriever": retriever.name,
                "generation_model": generation_model,
                "top_k": _clamp_top_k(top_k, fallback=self.config.top_k),
                "retrieval_latency_s": retrieval_latency_s,
                "retrieval_metadata": retrieval_metadata or {},
                "retrieved": [
                    _hit_source_payload(hit, include_private_text=self.config.include_private_source_text)
                    for hit in _filter_retrieved_for_display(
                        hits,
                        answer,
                        include_score_filtered=bool(score_controls and score_controls.has_score_range),
                    )
                ],
                "key_alias": generation.key_alias,
                "attempted_aliases": generation.attempted_aliases,
                "rejected_aliases": generation.rejected_aliases,
                "retry_count": generation.retry_count,
                "scheduled_wait_s": generation.scheduled_wait_s,
                "output_tokens_per_s": generation.output_tokens_per_s,
                "rate_limited": generation.rate_limited,
                "key_rate_limits": self.llm.rate_limit_snapshot(),
                "privacy": privacy_payload,
            },
        }

    def _text_dictionary_fallback(
        self,
        question: str,
        *,
        top_k: int,
        primary_retriever: Retriever,
        primary_retrieval: RetrievalResult,
    ) -> RetrievalResult | None:
        if primary_retriever.name == "dictionary-graph":
            return None
        dictionary_retriever = self.retrievers.get("dictionary-graph")
        if dictionary_retriever is None:
            return None
        if not _looks_like_dictionary_text_query(question):
            return None
        request_top_k = _clamp_top_k(top_k, fallback=self.config.dictionary_top_k)
        query_plan = plan_dictionary_query(question) if self.config.enable_dictionary_query_planner else None
        dictionary_hit_limit = _dictionary_list_query_hit_limit(query_plan, request_top_k)
        retrieval = dictionary_retriever.search(Query(query_id="chat-dict-fallback", text=question), dictionary_hit_limit)
        if query_plan is not None:
            extra_results = _planned_dictionary_extra_results(
                dictionary_retriever,
                query_plan,
                original_query=question,
                request_top_k=dictionary_hit_limit,
                query_id_prefix="chat-dict-fallback-plan",
            )
            if extra_results:
                retrieval.hits = merge_planned_dictionary_results(retrieval.hits, extra_results)
            retrieval.hits = annotate_and_rank_dictionary_hits(retrieval.hits, query_plan, max_hits=dictionary_hit_limit)
        primary_top_score = max((hit.score for hit in primary_retrieval.hits), default=0.0)
        allow_lexical_mentions = primary_top_score <= 0 and bool(query_plan and query_plan.target_terms)
        hits = [
            hit
            for hit in retrieval.hits
            if _strong_dictionary_text_fallback_hit(hit, allow_lexical=allow_lexical_mentions)
        ]
        if not hits:
            return None
        has_direct_dictionary_hit = any(float(hit.metadata.get("dictionary_direct_score") or 0.0) > 0 for hit in hits)
        if primary_top_score > 0 and not has_direct_dictionary_hit:
            return None
        metadata = dict(retrieval.metadata)
        if query_plan is not None:
            metadata["query_plan"] = query_plan.to_payload()
            metadata["dictionary_tool_plan"] = dictionary_tool_plan_payload(query_plan, original_query=question)
        hits = _canonicalize_dictionary_redirect_hits(
            hits,
            preserve_headword_terms=_dictionary_redirect_preserve_terms(query_plan),
        )
        return RetrievalResult(query=retrieval.query, hits=hits, latency_s=retrieval.latency_s, metadata=metadata)

    def lookup_dictionary(
        self,
        term: str,
        *,
        top_k: int | None = None,
        score_min: float | None = None,
        score_max: float | None = None,
        sort_by_score: bool | None = None,
    ) -> dict[str, Any]:
        query = str(term or "").strip()
        if not query:
            raise ValueError("term must not be empty")
        retriever = self.resolve_request_retriever("dictionary-graph")
        score_controls = _normalize_retrieval_score_controls(score_min, score_max, sort_by_score)
        request_top_k = _clamp_top_k(top_k, fallback=1)
        query_plan = plan_dictionary_query(query) if self.config.enable_dictionary_query_planner else None
        retrieval = retriever.search(Query(query_id="dictionary-lookup", text=query), request_top_k)
        if query_plan is not None:
            extra_results = _planned_dictionary_extra_results(
                retriever,
                query_plan,
                original_query=query,
                request_top_k=request_top_k,
                query_id_prefix="dictionary-lookup-plan",
            )
            if extra_results:
                retrieval.hits = merge_planned_dictionary_results(retrieval.hits, extra_results)
        if query_plan is not None:
            retrieval.hits = annotate_and_rank_dictionary_hits(retrieval.hits, query_plan, max_hits=request_top_k)
        retrieval, score_filter_metadata = _apply_retrieval_score_controls(
            retrieval,
            score_controls,
            max_hits=request_top_k,
        )
        hits = [hit for hit in retrieval.hits if hit.score > 0 or score_controls.has_score_range]
        retrieval_metadata = {**retrieval.metadata, **score_filter_metadata}
        if query_plan is not None:
            retrieval_metadata["query_plan"] = query_plan.to_payload()
            retrieval_metadata["dictionary_tool_plan"] = dictionary_tool_plan_payload(query_plan, original_query=query)
        return {
            "object": "dictionary.lookup",
            "query": query,
            "retriever": retriever.name,
            "top_k": request_top_k,
            "retrieval_latency_s": retrieval.latency_s,
            "retrieval_metadata": retrieval_metadata,
            "dictionary": self.dictionary_status,
            "retrieved": [_hit_source_payload(hit) for hit in hits],
        }

    def available_model_ids(self) -> tuple[str, ...]:
        mimo_models = self.config.mimo_models if self.config.mimo_enabled else ()
        return _dedupe_preserve_order((self.config.model_id, self.config.model, *self.config.available_models, *mimo_models))

    def available_generation_models(self) -> tuple[str, ...]:
        mimo_models = self.config.mimo_models if self.config.mimo_enabled else ()
        backend_models = tuple(model for models in self.config.backend_model_allowlist.values() for model in models)
        return _dedupe_preserve_order(
            (
                self.config.model,
                *self.config.available_models,
                *mimo_models,
                *self.config.trusted_private_models,
                *self.config.trusted_local_models,
                *backend_models,
            )
        )

    def available_retriever_ids(self) -> tuple[str, ...]:
        return tuple(self.retrievers.keys())

    def resolve_request_model(self, request_model: str | None) -> tuple[str, str]:
        if request_model is None or request_model == "" or request_model == self.config.model_id:
            return self.config.model_id, self.config.model
        if not isinstance(request_model, str):
            raise ValueError("model must be a string")
        if request_model in self.available_generation_models():
            return request_model, request_model
        allowed = ", ".join((self.config.model_id, *self.available_generation_models()))
        raise ValueError(f"Unknown model '{request_model}'. Use one of: {allowed}.")

    def resolve_request_retriever(self, request_retriever: str | None) -> Retriever:
        if request_retriever is None or request_retriever == "":
            return self.retriever
        if not isinstance(request_retriever, str):
            raise ValueError("retriever must be a string")
        retriever_id = normalize_retriever_id(request_retriever)
        try:
            return self.retrievers[retriever_id]
        except KeyError as exc:
            allowed = ", ".join(self.available_retriever_ids())
            raise ValueError(f"Unknown retriever '{request_retriever}'. Use one of: {allowed}.") from exc

    def resolve_text_request_retriever(self, request_retriever: str | None) -> Retriever:
        retriever = self.resolve_request_retriever(request_retriever)
        if retriever.name != "image-digits":
            return retriever
        if self.retriever.name != "image-digits":
            return self.retriever
        for candidate in self.retrievers.values():
            if candidate.name != "image-digits":
                return candidate
        raise ValueError("Text mode requires a non-image retriever.")

    def _privacy_state_for_request(
        self,
        *,
        session_id: str | None,
        privacy_state: dict[str, Any] | None,
        reset: bool,
        messages: list[dict[str, Any]],
    ) -> ConversationPrivacyState:
        requested_session = str(session_id or (privacy_state or {}).get("session_id") or "__default__")
        if reset:
            previous = self.privacy_states.get(requested_session)
            if previous and previous.private_seen and _messages_have_history(messages):
                decision = PrivacyDecision(
                    effective_tier=previous.max_seen_tier,
                    provider_allowed=False,
                    selected_provider=None,
                    selected_model=None,
                    backend_id=None,
                    backend_kind=BackendKind.UNKNOWN,
                    external_blocked=True,
                    reason="reset_privacy_requires_clean_new_session",
                    redaction_required=True,
                    provider_requested="",
                    model_requested=None,
                )
                previous.update(previous.max_seen_tier, external_blocked=True, reason=decision.reason)
                raise PrivacyRouteError(decision)
            state = ConversationPrivacyState(session_id=requested_session)
            self.privacy_states[requested_session] = state
            return state
        if requested_session in self.privacy_states:
            return self.privacy_states[requested_session]
        state = ConversationPrivacyState.from_mapping(privacy_state, session_id=requested_session)
        self.privacy_states[requested_session] = state
        return state

    def _backend_for_model(self, model: str | None) -> BackendDescriptor:
        model_id = str(model or "")
        provider = "mimo" if self.config.mimo_enabled and model_id in set(self.config.mimo_models) else "groq"
        backend_id = self.config.backend_id or provider
        return classify_backend(
            provider=provider,
            model=model_id,
            backend_id=backend_id,
            backend_kind=self.config.backend_kind,
            base_url=self.config.backend_base_url or (self.config.mimo_base_url if provider == "mimo" else None),
        )

    def _private_backend_policy(self) -> PrivateBackendPolicy:
        return PrivateBackendPolicy.from_values(
            trusted_private_backends=self.config.trusted_private_backends,
            trusted_private_models=self.config.trusted_private_models,
            backend_model_allowlist=self.config.backend_model_allowlist,
            trusted_local_models=self.config.trusted_local_models,
        )

    def _enforce_generation_privacy(
        self,
        *,
        backend: BackendDescriptor,
        session_state: ConversationPrivacyState,
        hits: list[RetrievalHit],
        user_message_tier: DataTier,
    ) -> PrivacyDecision:
        decision = enforce_privacy_route(
            backend.provider,
            backend.model,
            session_state,
            hits,
            allow_external_semi_private=self.config.allow_external_semi_private,
            private_backend_policy=self._private_backend_policy(),
            backend_id=backend.backend_id,
            backend_kind=backend.kind,
            base_url=backend.base_url,
            user_message_tier=user_message_tier,
        )
        if not decision.provider_allowed:
            raise PrivacyRouteError(decision)
        return decision

    def _record_no_generation_privacy(
        self,
        session_state: ConversationPrivacyState,
        hits: list[RetrievalHit],
        *,
        backend: BackendDescriptor,
        user_message_tier: DataTier,
    ) -> PrivacyDecision:
        effective_tier = max_data_tier(session_state.max_seen_tier, user_message_tier, *(data_tier_for_hit(hit) for hit in hits))
        session_state.update(effective_tier, external_blocked=False, reason="no_llm_generation")
        return PrivacyDecision(
            effective_tier=effective_tier,
            provider_allowed=True,
            selected_provider=None,
            selected_model=None,
            backend_id=backend.backend_id,
            backend_kind=backend.kind,
            external_blocked=False,
            reason="no_llm_generation",
            redaction_required=effective_tier == DataTier.PRIVATE,
            provider_requested=backend.provider,
            model_requested=backend.model,
        )

    def _llm_tool_blocked(
        self,
        *,
        backend: BackendDescriptor,
        session_state: ConversationPrivacyState,
        user_message_tier: DataTier,
    ) -> bool:
        decision = enforce_privacy_route(
            backend.provider,
            backend.model,
            session_state,
            (),
            allow_external_semi_private=self.config.allow_external_semi_private,
            private_backend_policy=self._private_backend_policy(),
            backend_id=backend.backend_id,
            backend_kind=backend.kind,
            base_url=backend.base_url,
            user_message_tier=user_message_tier,
        )
        return not decision.provider_allowed

    def _privacy_allowed_image_rewrite(
        self,
        *,
        image_rewrite: bool | None,
        backend: BackendDescriptor,
        session_state: ConversationPrivacyState,
        user_message_tier: DataTier,
    ) -> bool:
        if not image_rewrite:
            return False
        return not self._llm_tool_blocked(
            backend=backend,
            session_state=session_state,
            user_message_tier=user_message_tier,
        )

    def _safe_text_retriever_for_privacy(
        self,
        retriever: Retriever,
        *,
        backend: BackendDescriptor,
        session_state: ConversationPrivacyState,
        user_message_tier: DataTier,
    ) -> tuple[Retriever, dict[str, Any]]:
        if retriever.name == "agent":
            return retriever, {}
        spec = get_retriever_spec(retriever.name)
        if not spec.uses_llm:
            return retriever, {}
        decision = enforce_privacy_route(
            backend.provider,
            backend.model,
            session_state,
            (),
            allow_external_semi_private=self.config.allow_external_semi_private,
            private_backend_policy=self._private_backend_policy(),
            backend_id=backend.backend_id,
            backend_kind=backend.kind,
            base_url=backend.base_url,
            user_message_tier=user_message_tier,
        )
        if decision.provider_allowed:
            return retriever, {}
        fallback = self.retrievers.get("bm25")
        if fallback is None or fallback.name == retriever.name:
            raise PrivacyRouteError(decision)
        return fallback, {
            "retriever_privacy_fallback": True,
            "requested_retriever": retriever.name,
            "selected_retriever": fallback.name,
            "privacy_reason": decision.reason,
        }

    def _image_query(self, text: str, generation_model: str, *, image_rewrite: bool | None) -> tuple[str, dict[str, Any]]:
        query = _strip_command_prefix(text) or "digit image"
        should_rewrite = bool(image_rewrite)
        if not should_rewrite:
            return query, {"image_query_rewrite": False, "image_query_original": query}

        generation = self.llm.generate(
            [
                {
                    "role": "system",
                    "content": (
                        "Rewrite the user request as one concise image search query. "
                        "This demo image index contains handwritten digit images with labels 0-9. "
                        "Return only the query text, no JSON and no explanation."
                    ),
                },
                {"role": "user", "content": query},
            ],
            model=generation_model,
            temperature=0.0,
            max_completion_tokens=32,
        )
        rewritten = _clean_image_query(generation.answer)
        if generation.error or not rewritten:
            return query, {
                "image_query_rewrite": True,
                "image_query_original": query,
                "image_query_error": generation.error,
                "image_query_fallback": True,
            }
        return rewritten, {
            "image_query_rewrite": True,
            "image_query_original": query,
            "image_query_model": generation_model,
            "image_query_key_alias": generation.key_alias,
            "image_query_retry_count": generation.retry_count,
            "image_query_prompt_tokens": generation.prompt_tokens,
            "image_query_completion_tokens": generation.completion_tokens,
            "image_query_total_tokens": generation.total_tokens,
        }

    def _keyword_search(
        self,
        retriever: Retriever,
        question: str,
        top_k: int,
        *,
        generation_model: str,
    ) -> RetrievalResult:
        started = time.perf_counter()
        variants, metadata = _keyword_query_variants(self.llm, question, model=generation_model)
        search_texts = list(variants) if variants else [question]
        candidate_k = max(top_k * 5, top_k, 20)
        results = [
            retriever.search(Query(query_id="chat-keyword", text=variant), candidate_k)
            for variant in search_texts
        ]
        hits = _merge_positive_keyword_hits(results, top_k=top_k)
        if not hits and variants:
            fallback = retriever.search(Query(query_id="chat-keyword", text=question), top_k)
            hits = [hit for hit in fallback.hits if hit.score > 0]
            metadata["keyword_fallback_to_original"] = True
        metadata["keyword_query_variants"] = search_texts
        return RetrievalResult(
            query=Query(query_id="chat", text=question),
            hits=hits,
            latency_s=time.perf_counter() - started,
            metadata=metadata,
        )

    def _agent_search(
        self,
        retriever: Retriever,
        question: str,
        top_k: int,
        *,
        generation_model: str,
    ) -> RetrievalResult:
        started = time.perf_counter()
        available_tools = _available_agent_tools(self.retrievers)
        generation = self.llm.generate(
            _agent_planner_messages(question, available_tools),
            model=generation_model,
            temperature=0.0,
            max_completion_tokens=256,
        )
        calls, parse_error = _parse_agent_tool_calls(generation.answer, available_tools=available_tools, fallback_query=question)
        repair_generation: GenerationResult | None = None
        repair_parse_error: str | None = None
        if parse_error and not calls:
            repair_generation = self.llm.generate(
                _agent_planner_json_retry_messages(
                    question,
                    available_tools,
                    invalid_output=str(generation.answer or ""),
                ),
                model=generation_model,
                temperature=0.0,
                max_completion_tokens=128,
            )
            calls, repair_parse_error = _parse_agent_tool_calls(
                repair_generation.answer,
                available_tools=available_tools,
                fallback_query=question,
            )
        tool_name_generation: GenerationResult | None = None
        tool_name_parse_error: str | None = None
        if (repair_parse_error or parse_error) and not calls:
            tool_name_generation = self.llm.generate(
                _agent_planner_tool_name_retry_messages(question, available_tools),
                model=generation_model,
                temperature=0.0,
                max_completion_tokens=48,
            )
            calls, tool_name_parse_error = _parse_agent_tool_name_calls(
                tool_name_generation.answer,
                available_tools=available_tools,
                question=question,
            )
        calls, repair_reason = _repair_agent_tool_calls(calls, question=question, available_tools=available_tools)
        metadata: dict[str, Any] = {
            "agent_mode": True,
            "agent_schema": AGENT_TOOL_SCHEMA_VERSION,
            "agent_llm_calls": 1 + int(repair_generation is not None) + int(tool_name_generation is not None),
            "agent_planner_model": generation_model,
            "agent_planner_key_alias": generation.key_alias,
            "agent_planner_retry_count": generation.retry_count,
            "agent_planner_prompt_tokens": generation.prompt_tokens,
            "agent_planner_completion_tokens": generation.completion_tokens,
            "agent_planner_total_tokens": generation.total_tokens,
            "agent_planner_error": generation.error,
        }
        if parse_error:
            metadata["agent_planner_parse_error"] = parse_error
        if repair_generation is not None:
            metadata["agent_planner_json_retry"] = True
            metadata["agent_planner_json_retry_key_alias"] = repair_generation.key_alias
            metadata["agent_planner_json_retry_error"] = repair_generation.error
            metadata["agent_planner_json_retry_prompt_tokens"] = repair_generation.prompt_tokens
            metadata["agent_planner_json_retry_completion_tokens"] = repair_generation.completion_tokens
            metadata["agent_planner_json_retry_total_tokens"] = repair_generation.total_tokens
            if repair_parse_error:
                metadata["agent_planner_json_retry_parse_error"] = repair_parse_error
        if tool_name_generation is not None:
            metadata["agent_planner_tool_name_retry"] = True
            metadata["agent_planner_tool_name_retry_key_alias"] = tool_name_generation.key_alias
            metadata["agent_planner_tool_name_retry_error"] = tool_name_generation.error
            metadata["agent_planner_tool_name_retry_prompt_tokens"] = tool_name_generation.prompt_tokens
            metadata["agent_planner_tool_name_retry_completion_tokens"] = tool_name_generation.completion_tokens
            metadata["agent_planner_tool_name_retry_total_tokens"] = tool_name_generation.total_tokens
            if tool_name_parse_error:
                metadata["agent_planner_tool_name_retry_parse_error"] = tool_name_parse_error
        if repair_reason:
            metadata["agent_planner_repair"] = repair_reason
        if not calls:
            fallback = retriever.search(Query(query_id="chat-agent", text=question), top_k)
            fallback.metadata.update(
                {
                    **metadata,
                    "agent_planner_fallback": "empty_or_invalid_plan",
                    "agent_tool_calls": [],
                    "agent_tool_call_count": 0,
                }
            )
            fallback.latency_s += time.perf_counter() - started
            return fallback

        results: list[RetrievalResult] = []
        tool_payloads: list[dict[str, Any]] = []
        dictionary_metadata: dict[str, Any] | None = None
        for index, call in enumerate(calls, 1):
            tool_name = call["name"]
            tool_query = call["query"]
            result = self._execute_agent_retrieval_tool(tool_name, tool_query, top_k, index=index)
            payload = {
                "name": tool_name,
                "query": tool_query,
                "result_count": len(result.hits) if result is not None else 0,
            }
            if result is None:
                payload["skipped"] = True
                tool_payloads.append(payload)
                continue
            results.append(result)
            if tool_name == "dictionary.lookup" and dictionary_metadata is None:
                dictionary_metadata = result.metadata
            tool_payloads.append(payload)
        hits = _merge_agent_tool_hits(results, top_k=top_k)
        metadata.update(
            {
                "agent_tool_calls": tool_payloads,
                "agent_tool_call_count": len([payload for payload in tool_payloads if not payload.get("skipped")]),
                "agent_available_tools": list(available_tools),
            }
        )
        if dictionary_metadata is not None:
            metadata["agent_dictionary_metadata"] = dictionary_metadata
        return RetrievalResult(
            query=Query(query_id="chat-agent", text=question),
            hits=hits,
            latency_s=time.perf_counter() - started,
            metadata=metadata,
        )

    def _execute_agent_retrieval_tool(
        self,
        tool_name: str,
        tool_query: str,
        top_k: int,
        *,
        index: int,
    ) -> RetrievalResult | None:
        query = Query(query_id=f"chat-agent-tool-{index}", text=tool_query)
        if tool_name == "dictionary.lookup":
            dictionary_retriever = self.retrievers.get("dictionary-graph")
            if dictionary_retriever is None:
                return None
            retrieval = dictionary_retriever.search(query, top_k)
            query_plan = plan_dictionary_query(tool_query) if self.config.enable_dictionary_query_planner else None
            if query_plan is not None:
                extra_results = _planned_dictionary_extra_results(
                    dictionary_retriever,
                    query_plan,
                    original_query=tool_query,
                    request_top_k=top_k,
                    query_id_prefix="chat-agent-dict-plan",
                )
                if extra_results:
                    retrieval.hits = merge_planned_dictionary_results(retrieval.hits, extra_results)
                retrieval.hits = annotate_and_rank_dictionary_hits(retrieval.hits, query_plan, max_hits=top_k)
                retrieval.metadata = {
                    **retrieval.metadata,
                    "query_plan": query_plan.to_payload(),
                    "dictionary_tool_plan": dictionary_tool_plan_payload(query_plan, original_query=tool_query),
                }
            retrieval.metadata["agent_tool_name"] = tool_name
            return retrieval
        retriever_name = {
            "text.multi_query": "multi-query" if "multi-query" in self.retrievers else "agent",
            "text.bm25": "bm25",
            "text.graph_bm25": "graph-bm25",
            "text.keyword": "keyword-match",
        }.get(tool_name)
        if not retriever_name:
            return None
        tool_retriever = self.retrievers.get(retriever_name)
        if tool_retriever is None:
            return None
        retrieval = tool_retriever.search(query, top_k)
        retrieval.metadata["agent_tool_name"] = tool_name
        return retrieval


@dataclass
class ModelRoutedChatClient:
    default_client: ChatGenerationClient
    routes: dict[str, ChatGenerationClient]

    @property
    def key_usage_counts(self) -> dict[str, int]:
        counts: dict[str, int] = {}
        for client in self._clients():
            for alias, count in dict(getattr(client, "key_usage_counts", {}) or {}).items():
                counts[alias] = counts.get(alias, 0) + int(count)
        return counts

    def generate(
        self,
        messages: list[dict[str, str]],
        *,
        model: str | None = None,
        temperature: float = 0.0,
        max_completion_tokens: int = 512,
    ) -> GenerationResult:
        client = self.routes.get(model or "") or self.default_client
        return client.generate(
            messages,
            model=model,
            temperature=temperature,
            max_completion_tokens=max_completion_tokens,
        )

    def rate_limit_snapshot(self) -> dict[str, dict[str, float | int | str]]:
        snapshot: dict[str, dict[str, float | int | str]] = {}
        for client in self._clients():
            for alias, details in client.rate_limit_snapshot().items():
                snapshot[alias] = details
        return snapshot

    def _clients(self) -> list[ChatGenerationClient]:
        clients = [self.default_client]
        for client in self.routes.values():
            if not any(client is existing for existing in clients):
                clients.append(client)
        return clients


def _keyword_query_variants(
    llm: ChatGenerationClient,
    question: str,
    *,
    model: str,
    max_keywords: int = 5,
) -> tuple[tuple[str, ...], dict[str, Any]]:
    generation = llm.generate(
        [
            {
                "role": "system",
                "content": (
                    f"Extract up to {max_keywords} keyword or keyphrase search queries for a scientific keyword index. "
                    "Return only a JSON array of strings. Include a mix of short identifiers and longer phrases. "
                    "Preserve scientific identifiers exactly, such as BH1, BH2, Bcl-2, Bax, or gene names. "
                    "Do not translate identifiers and do not answer the question."
                ),
            },
            {"role": "user", "content": question},
        ],
        model=model,
        temperature=0.0,
        max_completion_tokens=96,
    )
    variants = _parse_string_array(str(generation.answer or ""), limit=max_keywords)
    if generation.error:
        variants = ()
    metadata = {
        "keyword_llm_calls": 1,
        "keyword_llm_model": model,
        "keyword_llm_key_alias": generation.key_alias,
        "keyword_llm_attempted_aliases": list(generation.attempted_aliases or []),
        "keyword_llm_rejected_aliases": list(generation.rejected_aliases or []),
        "keyword_llm_retry_count": generation.retry_count,
        "keyword_llm_prompt_tokens": generation.prompt_tokens,
        "keyword_llm_completion_tokens": generation.completion_tokens,
        "keyword_llm_total_tokens": generation.total_tokens,
        "keyword_llm_error": generation.error,
    }
    return variants, metadata


def _available_agent_tools(retrievers: dict[str, Retriever]) -> tuple[str, ...]:
    tools: list[str] = []
    if "dictionary-graph" in retrievers:
        tools.append("dictionary.lookup")
    if "agent" in retrievers or "multi-query" in retrievers:
        tools.append("text.multi_query")
    if "bm25" in retrievers:
        tools.append("text.bm25")
    if "graph-bm25" in retrievers:
        tools.append("text.graph_bm25")
    if "keyword-match" in retrievers:
        tools.append("text.keyword")
    return tuple(tool for tool in AGENT_TOOL_ALLOWLIST if tool in tools)


def _agent_planner_messages(question: str, available_tools: Sequence[str]) -> list[dict[str, str]]:
    tool_lines = "\n".join(f"- {tool}" for tool in available_tools)
    return [
        {
            "role": "system",
            "content": (
                "You are a retrieval planner. Choose retrieval tools for the user's question, then stop. "
                "Do not answer the user. Return only valid JSON with this shape: "
                '{"tool_calls":[{"name":"dictionary.lookup","query":"exact search text"}]}.\n'
                "Rules:\n"
                "- Use only tool names from the available list.\n"
                "- Use at most 4 tool calls.\n"
                "- Preserve acronyms, casing, digits, Vietnamese diacritics, hyphens, and Roman suffixes exactly.\n"
                "- For dictionary/domain terms, abbreviations, aliases, definitions, occurrences, or type/list queries, prefer dictionary.lookup.\n"
                "- For broader text evidence, add text.multi_query or another text tool when available.\n"
                "- Never invent evidence or source text.\n\n"
                f"Available tools:\n{tool_lines}"
            ),
        },
        {
            "role": "user",
            "content": (
                f"User question:\n{question}\n\n"
                "Choose retrieval tools for this question. Return only the JSON object. "
                "Do not answer the question."
            ),
        },
    ]


def _agent_planner_json_retry_messages(
    question: str,
    available_tools: Sequence[str],
    *,
    invalid_output: str,
) -> list[dict[str, str]]:
    tool_lines = "\n".join(f"- {tool}" for tool in available_tools)
    return [
        {
            "role": "system",
            "content": (
                "Return only valid JSON for retrieval tool calls. Do not answer the user. "
                "No markdown. No explanation. Required schema: "
                '{"tool_calls":[{"name":"dictionary.lookup","query":"search text"}]}.\n'
                f"Available tools:\n{tool_lines}"
            ),
        },
        {
            "role": "user",
            "content": (
                f"User question:\n{question}\n\n"
                "Previous invalid planner output:\n"
                f"{invalid_output[:600]}\n\n"
                "Now return only the JSON tool_calls object."
            ),
        },
    ]


def _agent_planner_tool_name_retry_messages(question: str, available_tools: Sequence[str]) -> list[dict[str, str]]:
    tool_list = ", ".join(available_tools)
    return [
        {
            "role": "system",
            "content": (
                "Choose retrieval tools only. Do not answer the question. "
                "Return only tool names separated by commas. No JSON, no markdown, no explanation. "
                f"Available tool names: {tool_list}."
            ),
        },
        {
            "role": "user",
            "content": (
                f"User question:\n{question}\n\n"
                "Choose retrieval tools for this question. Return only comma-separated tool names from the available list. "
                "Do not answer the question."
            ),
        },
    ]


def _parse_agent_tool_calls(
    text: str,
    *,
    available_tools: Sequence[str],
    fallback_query: str,
    limit: int = 4,
) -> tuple[list[dict[str, str]], str | None]:
    available = set(available_tools)
    stripped = _strip_code_fence(str(text or "").strip())
    candidates = [stripped]
    start_object = stripped.find("{")
    end_object = stripped.rfind("}")
    if start_object >= 0 and end_object > start_object:
        candidates.append(stripped[start_object : end_object + 1])
    start_list = stripped.find("[")
    end_list = stripped.rfind("]")
    if start_list >= 0 and end_list > start_list:
        candidates.append(stripped[start_list : end_list + 1])
    parse_error: str | None = None
    for candidate in candidates:
        if not candidate:
            continue
        try:
            parsed = json.loads(candidate)
        except json.JSONDecodeError as exc:
            parse_error = f"json_parse_error:{exc.msg}"
            continue
        raw_calls = parsed.get("tool_calls") if isinstance(parsed, dict) else parsed
        if not isinstance(raw_calls, list):
            parse_error = "tool_calls_not_list"
            continue
        calls: list[dict[str, str]] = []
        seen: set[tuple[str, str]] = set()
        for raw_call in raw_calls:
            if not isinstance(raw_call, dict):
                continue
            name = str(raw_call.get("name") or raw_call.get("tool") or "").strip()
            query = str(raw_call.get("query") or raw_call.get("input") or fallback_query).strip()
            if name not in available or name not in AGENT_TOOL_ALLOWLIST:
                continue
            if not query:
                query = fallback_query
            query = re.sub(r"\s+", " ", query).strip()[:180]
            key = (name, query)
            if key in seen:
                continue
            seen.add(key)
            calls.append({"name": name, "query": query})
            if len(calls) >= limit:
                break
        return calls, None if calls else "no_valid_tool_calls"
    return [], parse_error or "no_json_plan"


def _parse_agent_tool_name_calls(
    text: str,
    *,
    available_tools: Sequence[str],
    question: str,
    limit: int = 4,
) -> tuple[list[dict[str, str]], str | None]:
    available = set(available_tools)
    lowered = str(text or "").lower()
    calls: list[dict[str, str]] = []
    seen: set[str] = set()
    for tool_name in AGENT_TOOL_ALLOWLIST:
        if tool_name not in available or tool_name in seen:
            continue
        if tool_name.lower() in lowered:
            query = _agent_default_query_for_tool(tool_name, question)
            calls.append({"name": tool_name, "query": query})
            seen.add(tool_name)
            if len(calls) >= limit:
                break
    if calls:
        return calls, None
    return [], "no_tool_name_found"


def _agent_default_query_for_tool(tool_name: str, question: str) -> str:
    if tool_name == "dictionary.lookup" and _looks_like_dictionary_text_query(question):
        query_plan = plan_dictionary_query(question)
        target = next((str(term).strip() for term in query_plan.target_terms if str(term).strip()), "")
        if target:
            return target
    return re.sub(r"\s+", " ", question).strip()[:180]


def _repair_agent_tool_calls(
    calls: list[dict[str, str]],
    *,
    question: str,
    available_tools: Sequence[str],
) -> tuple[list[dict[str, str]], str | None]:
    available = set(available_tools)
    repaired = list(calls)
    seen = {(call["name"], call["query"]) for call in repaired}
    if "dictionary.lookup" in available and _looks_like_dictionary_text_query(question):
        has_dictionary = any(call["name"] == "dictionary.lookup" for call in repaired)
        if not has_dictionary:
            target = _agent_default_query_for_tool("dictionary.lookup", question)
            key = ("dictionary.lookup", target)
            if key not in seen:
                repaired.insert(0, {"name": "dictionary.lookup", "query": target})
                return repaired[:4], "added_dictionary_lookup_for_dictionary_query"
    if not repaired:
        for tool_name in ("text.multi_query", "text.bm25", "text.graph_bm25", "text.keyword"):
            if tool_name in available:
                return [{"name": tool_name, "query": question}], "fallback_text_tool"
    return repaired[:4], None


def _merge_agent_tool_hits(results: list[RetrievalResult], *, top_k: int, rrf_k: int = 60) -> list[RetrievalHit]:
    scores: dict[str, float] = {}
    best_rank: dict[str, int] = {}
    hits_by_doc_id: dict[str, RetrievalHit] = {}
    tool_names_by_doc_id: dict[str, list[str]] = {}
    for result in results:
        tool_name = str(result.metadata.get("agent_tool_name") or result.metadata.get("tool_name") or "")
        for hit in result.hits:
            if hit.score <= 0 and data_tier_for_hit(hit) == DataTier.PUBLIC:
                continue
            scores[hit.doc_id] = scores.get(hit.doc_id, 0.0) + max(float(hit.score), 0.0) + 1.0 / (rrf_k + hit.rank)
            best_rank[hit.doc_id] = min(best_rank.get(hit.doc_id, hit.rank), hit.rank)
            hits_by_doc_id.setdefault(hit.doc_id, hit)
            if tool_name:
                names = tool_names_by_doc_id.setdefault(hit.doc_id, [])
                if tool_name not in names:
                    names.append(tool_name)
    ranked = sorted(scores, key=lambda doc_id: (-scores[doc_id], best_rank[doc_id], doc_id))
    hits: list[RetrievalHit] = []
    for rank, doc_id in enumerate(ranked[:top_k], 1):
        hit = hits_by_doc_id[doc_id]
        metadata = dict(hit.metadata or {})
        if tool_names_by_doc_id.get(doc_id):
            metadata["agent_tool_names"] = tool_names_by_doc_id[doc_id]
        hits.append(
            RetrievalHit(
                doc_id=doc_id,
                score=scores[doc_id],
                rank=rank,
                title=hit.title,
                text=hit.text,
                metadata=metadata,
                data_tier=hit.data_tier,
                doc_type=hit.doc_type,
                source_id=hit.source_id,
                allowed_llm=hit.allowed_llm,
                allowed_embedding=hit.allowed_embedding,
                redaction_policy=hit.redaction_policy,
            )
        )
    return hits


def _merge_positive_keyword_hits(results: list[RetrievalResult], *, top_k: int) -> list[RetrievalHit]:
    scores: dict[str, float] = {}
    best_rank: dict[str, int] = {}
    hits_by_doc_id: dict[str, RetrievalHit] = {}
    for result in results:
        for hit in result.hits:
            if hit.score <= 0:
                continue
            scores[hit.doc_id] = scores.get(hit.doc_id, 0.0) + float(hit.score) + 1.0 / (60 + hit.rank)
            best_rank[hit.doc_id] = min(best_rank.get(hit.doc_id, hit.rank), hit.rank)
            hits_by_doc_id.setdefault(hit.doc_id, hit)
    ranked = sorted(scores, key=lambda doc_id: (-scores[doc_id], best_rank[doc_id], doc_id))
    return [
        RetrievalHit(
            doc_id=doc_id,
            score=scores[doc_id],
            rank=rank,
            title=hits_by_doc_id[doc_id].title,
            text=hits_by_doc_id[doc_id].text,
            metadata=hits_by_doc_id[doc_id].metadata,
            data_tier=hits_by_doc_id[doc_id].data_tier,
            doc_type=hits_by_doc_id[doc_id].doc_type,
            source_id=hits_by_doc_id[doc_id].source_id,
            allowed_llm=hits_by_doc_id[doc_id].allowed_llm,
            allowed_embedding=hits_by_doc_id[doc_id].allowed_embedding,
            redaction_policy=hits_by_doc_id[doc_id].redaction_policy,
        )
        for rank, doc_id in enumerate(ranked[:top_k], 1)
    ]


def _merge_text_and_dictionary_hits(
    primary_hits: list[RetrievalHit],
    dictionary_hits: list[RetrievalHit],
    *,
    max_hits: int | None = None,
    preserve_dictionary_redirect_terms: Sequence[str] = (),
) -> list[RetrievalHit]:
    dictionary_hits = _canonicalize_dictionary_redirect_hits(
        dictionary_hits,
        preserve_headword_terms=preserve_dictionary_redirect_terms,
    )
    if not dictionary_hits:
        hits = _canonicalize_dictionary_redirect_hits(primary_hits)
        if max_hits is not None:
            hits = hits[: _clamp_top_k(max_hits, fallback=max_hits)]
        return [
            replace(hit, rank=rank)
            for rank, hit in enumerate(hits, 1)
        ]
    primary_candidates = [hit for hit in primary_hits if hit.score > MIN_RETRIEVAL_DISPLAY_SCORE]
    merged: list[RetrievalHit] = []
    seen: set[str] = set()
    for hit in (*dictionary_hits, *primary_candidates):
        if hit.doc_id in seen:
            continue
        seen.add(hit.doc_id)
        merged.append(hit)
        if max_hits is not None and len(merged) >= _clamp_top_k(max_hits, fallback=max_hits):
            break
    merged = _canonicalize_dictionary_redirect_hits(
        merged,
        preserve_headword_terms=preserve_dictionary_redirect_terms,
    )
    if max_hits is not None:
        merged = merged[: _clamp_top_k(max_hits, fallback=max_hits)]
    return [
        RetrievalHit(
            doc_id=hit.doc_id,
            score=hit.score,
            rank=rank,
            title=hit.title,
            text=hit.text,
            metadata=hit.metadata,
            data_tier=hit.data_tier,
            doc_type=hit.doc_type,
            source_id=hit.source_id,
            allowed_llm=hit.allowed_llm,
            allowed_embedding=hit.allowed_embedding,
            redaction_policy=hit.redaction_policy,
        )
        for rank, hit in enumerate(merged, 1)
    ]


def _looks_like_dictionary_text_query(text: str) -> bool:
    query = _strip_command_prefix(text)
    if not query or len(query) > 96:
        return False
    tokens = re.findall(r"[\wĐđ]+", query, flags=re.UNICODE)
    if not (1 <= len(tokens) <= 8):
        return False
    if "?" in query and len(tokens) > 5:
        return False
    return True


def _strong_dictionary_text_fallback_hit(hit: RetrievalHit, *, allow_lexical: bool = False) -> bool:
    if hit.score <= 0:
        return False
    metadata = hit.metadata or {}
    mode = str(metadata.get("dictionary_match_mode") or "")
    direct_score = float(metadata.get("dictionary_direct_score") or 0.0)
    graph_score = float(metadata.get("dictionary_graph_score") or 0.0)
    has_highlights = bool(metadata.get("query_highlights"))
    return (
        mode in {"strict", "folded"}
        or mode == "roman_sibling"
        or direct_score > 0
        or (mode == "graph" and graph_score >= 0.35)
        or (allow_lexical and mode == "lexical" and (has_highlights or hit.score >= 0.25))
    )


def _normalize_retrieval_score_controls(
    min_score: float | None,
    max_score: float | None,
    sort_by_score: bool | None,
) -> RetrievalScoreControls:
    normalized_min = _normalize_optional_score(min_score, "score_min")
    normalized_max = _normalize_optional_score(max_score, "score_max")
    if normalized_min is not None and normalized_max is not None and normalized_min > normalized_max:
        raise ValueError("score_min must be less than or equal to score_max")
    return RetrievalScoreControls(
        min_score=normalized_min,
        max_score=normalized_max,
        sort_by_score=bool(sort_by_score),
    )


def _normalize_optional_score(value: float | None, name: str) -> float | None:
    if value is None:
        return None
    score = float(value)
    if not math.isfinite(score):
        raise ValueError(f"{name} must be a finite number")
    return score


def _apply_retrieval_score_controls(
    retrieval: RetrievalResult,
    controls: RetrievalScoreControls,
    *,
    max_hits: int | None = None,
) -> tuple[RetrievalResult, dict[str, Any]]:
    if not controls.active:
        return retrieval, {}
    filtered = [
        hit
        for hit in retrieval.hits
        if (controls.min_score is None or hit.score >= controls.min_score)
        and (controls.max_score is None or hit.score <= controls.max_score)
    ]
    sort_strategy = "input_order"
    if controls.sort_by_score:
        filtered = sorted(filtered, key=lambda hit: (-hit.score, hit.rank, hit.doc_id))
        sort_strategy = "score_desc"
    if max_hits is not None:
        filtered = filtered[: _clamp_top_k(max_hits, fallback=max_hits)]
    reranked = [
        replace(hit, rank=index)
        for index, hit in enumerate(filtered, start=1)
    ]
    metadata = {
        "score_filter": {
            "min_score": controls.min_score,
            "max_score": controls.max_score,
            "sort_by_score": controls.sort_by_score,
            "sort_strategy": sort_strategy,
            "input_count": len(retrieval.hits),
            "output_count": len(reranked),
        }
    }
    return (
        RetrievalResult(
            query=retrieval.query,
            hits=reranked,
            latency_s=retrieval.latency_s,
            metadata=retrieval.metadata,
        ),
        metadata,
    )


def _filter_retrieved_for_display(
    hits: list[RetrievalHit],
    answer: str,
    *,
    include_score_filtered: bool = False,
) -> list[RetrievalHit]:
    cited_doc_ids = _cited_doc_ids(answer)
    filtered = [
        hit
        for hit in hits
        if include_score_filtered or hit.score > MIN_RETRIEVAL_DISPLAY_SCORE or hit.doc_id in cited_doc_ids or _hit_is_image(hit)
    ]
    return _canonicalize_dictionary_redirect_hits(filtered)


def _cited_doc_ids(answer: str) -> set[str]:
    return {match.group(1).strip() for match in re.finditer(r"\[([^\[\]]+)\]", answer or "")}


def _hit_is_image(hit: RetrievalHit) -> bool:
    return bool(hit.metadata.get("image_data_url") or hit.metadata.get("image_url") or hit.metadata.get("kind") == "image")


def _parse_string_array(text: str, *, limit: int) -> tuple[str, ...]:
    stripped = _strip_code_fence(text.strip())
    candidates = [stripped]
    start = stripped.find("[")
    end = stripped.rfind("]")
    if start >= 0 and end > start:
        candidates.append(stripped[start : end + 1])
    for candidate in candidates:
        try:
            parsed = json.loads(candidate)
        except json.JSONDecodeError:
            continue
        if isinstance(parsed, list):
            return _dedupe_strings([str(item) for item in parsed if isinstance(item, str)])[:limit]
    lines = [
        re.sub(r"^[-*\d.)\s]+", "", line).strip(" \"'")
        for line in stripped.splitlines()
        if line.strip()
    ]
    return _dedupe_strings(lines)[:limit]


def _strip_code_fence(text: str) -> str:
    if text.startswith("```"):
        lines = text.splitlines()
        if len(lines) >= 2 and lines[-1].strip() == "```":
            return "\n".join(lines[1:-1]).strip()
    return text


def _dedupe_strings(values: list[str]) -> tuple[str, ...]:
    seen: set[str] = set()
    result: list[str] = []
    for value in values:
        normalized = " ".join(value.split())
        key = normalized.lower()
        if normalized and key not in seen:
            seen.add(key)
            result.append(normalized)
    return tuple(result)


def _build_llm(config: ChatProxyConfig, keys: list[ApiKey]) -> ChatGenerationClient:
    groq_client = RoundRobinGroqClient(
        keys=keys,
        model=config.model,
        max_retries=config.max_retries,
        key_tokens_per_minute=config.key_tokens_per_minute,
        key_requests_per_minute=config.key_requests_per_minute,
        rate_limit_scope=config.rate_limit_scope,
        provider_name="Groq",
    )
    if not config.mimo_enabled:
        return groq_client

    mimo_keys = load_env_api_key_chain(
        config.mimo_env_file,
        config.mimo_api_key_var,
        primary_alias="mimo",
        fallback_variables=(("MIMO_API_KEY_PAYG", "mimo_payg"),),
    )
    mimo_primary = _build_mimo_client(config, [mimo_keys[0]])
    mimo_client: ChatGenerationClient = mimo_primary
    if len(mimo_keys) > 1:
        mimo_client = FallbackChatClient(primary=mimo_primary, fallback=_build_mimo_client(config, mimo_keys[1:]))
    return ModelRoutedChatClient(
        default_client=groq_client,
        routes={model: mimo_client for model in config.mimo_models},
    )


def _build_mimo_client(config: ChatProxyConfig, keys: list[ApiKey]) -> RoundRobinGroqClient:
    return RoundRobinGroqClient(
        keys=keys,
        model=config.mimo_models[0] if config.mimo_models else "mimo-v2.5-pro",
        max_retries=config.max_retries,
        key_tokens_per_minute=config.mimo_key_tokens_per_minute,
        key_requests_per_minute=config.mimo_key_requests_per_minute,
        rate_limit_scope="per-key",
        client_factory=lambda key, timeout: OpenAICompatibleClient(
            api_key=key.value,
            base_url=_mimo_base_url_for_key(config, key),
            timeout_s=timeout,
            token_parameter="max_tokens",
            auth_header=config.mimo_auth_header,
        ),
        provider_name="MiMo",
        completion_token_parameter="max_tokens",
    )


def _mimo_base_url_for_key(config: ChatProxyConfig, key: ApiKey) -> str:
    if key.alias == "mimo_payg" or key.value.startswith("sk-"):
        return config.mimo_payg_base_url
    return config.mimo_base_url


def _build_retrievers(
    config: ChatProxyConfig,
    benchmark: BenchmarkData,
    *,
    llm: ChatGenerationClient | None = None,
    dictionary: DictionaryLoadResult | None = None,
) -> dict[str, Retriever]:
    retrievers: dict[str, Retriever] = {}
    for name in _dedupe_normalized_retriever_ids((config.retriever, *config.available_retrievers)):
        spec = get_retriever_spec(name)
        documents = dictionary.documents if spec.category == "dictionary" and dictionary is not None else benchmark.documents
        if not documents and spec.category in {"text", "keyword"}:
            retriever = EmptyCorpusRetriever(name=spec.id)
        else:
            retriever = create_retriever(
                name,
                vector_model=config.vector_model,
                query_expander=llm,
                query_model=config.model,
            )
        retriever.build(documents)
        retrievers[retriever.name] = retriever
    return retrievers


def _load_dictionary(config: ChatProxyConfig) -> DictionaryLoadResult:
    return load_dictionary_documents(
        artifact_dir=config.dictionary_artifact,
        source_dir=config.dictionary_source_dir,
        letters=config.dictionary_letters,
        required=config.dictionary_required,
    )


def _load_structured_evidence_index(config: ChatProxyConfig) -> StructuredEvidenceIndex | None:
    if not config.enable_structured_evidence:
        return None
    docs = []
    if config.structured_evidence_jsonl is not None:
        docs.extend(load_structured_evidence_jsonl(config.structured_evidence_jsonl))
    if config.structured_evidence_md is not None:
        docs.extend(load_structured_evidence_markdown(config.structured_evidence_md))
    return StructuredEvidenceIndex(docs)


def _default_retriever(config: ChatProxyConfig, retrievers: dict[str, Retriever]) -> Retriever:
    retriever_id = normalize_retriever_id(config.retriever)
    try:
        return retrievers[retriever_id]
    except KeyError as exc:
        allowed = ", ".join(retrievers)
        raise ValueError(f"Default retriever '{config.retriever}' was not built. Built retrievers: {allowed}.") from exc


def _dedupe_preserve_order(values: tuple[str, ...]) -> tuple[str, ...]:
    seen: set[str] = set()
    result: list[str] = []
    for value in values:
        if value and value not in seen:
            seen.add(value)
            result.append(value)
    return tuple(result)


def _dedupe_normalized_retriever_ids(values: tuple[str, ...]) -> tuple[str, ...]:
    seen: set[str] = set()
    result: list[str] = []
    for value in values:
        if not value:
            continue
        retriever_id = normalize_retriever_id(value)
        if retriever_id and retriever_id not in seen:
            seen.add(retriever_id)
            result.append(retriever_id)
    return tuple(result)


def _clamp_top_k(value: int | None, *, fallback: int) -> int:
    if value is None:
        return max(1, fallback)
    return min(20, max(1, int(value)))


def _dictionary_list_query_hit_limit(query_plan: Any, request_top_k: int) -> int:
    request_limit = _clamp_top_k(request_top_k, fallback=request_top_k)
    if _is_plural_type_category_query_plan(query_plan) or _is_plural_phrase_list_query_plan(query_plan):
        return max(request_limit, DICTIONARY_LIST_FALLBACK_MIN_HITS)
    return request_limit


def _planned_dictionary_extra_results(
    retriever: Retriever,
    query_plan: DictionaryQueryPlan,
    *,
    original_query: str,
    request_top_k: int,
    query_id_prefix: str,
) -> list[list[RetrievalHit]]:
    extra_results: list[list[RetrievalHit]] = []
    normalized_original = original_query.strip().lower()
    for index, term in enumerate(query_plan.target_terms[:3], 1):
        if not term or term.strip().lower() == normalized_original:
            continue
        extra_results.append(
            retriever.search(Query(query_id=f"{query_id_prefix}-{index}", text=term), request_top_k).hits
        )
        if _is_plural_type_category_query_plan(query_plan):
            prefix_search = getattr(retriever, "prefix_headword_search", None)
            if callable(prefix_search):
                prefix_top_k = max(request_top_k, min(80, request_top_k * 8))
                extra_results.append(
                    prefix_search(
                        Query(query_id=f"{query_id_prefix}-{index}-prefix", text=term),
                        prefix_top_k,
                    ).hits
                )
    return extra_results


def build_chat_rag_messages(
    messages: list[dict[str, Any]],
    hits: list[RetrievalHit],
    *,
    max_context_chars: int,
    history_messages: int,
    language: str | None = None,
    dictionary_fallback_metadata: dict[str, Any] | None = None,
) -> list[dict[str, str]]:
    sections = build_chat_rag_prompt_sections(
        messages,
        hits,
        max_context_chars=max_context_chars,
        history_messages=history_messages,
        dictionary_fallback_metadata=dictionary_fallback_metadata,
    )
    return _build_chat_rag_messages_from_sections(sections, language=language)


def build_chat_rag_prompt_sections(
    messages: list[dict[str, Any]],
    hits: list[RetrievalHit],
    *,
    max_context_chars: int,
    history_messages: int,
    dictionary_fallback_metadata: dict[str, Any] | None = None,
) -> list[PromptSection]:
    question = last_user_text(messages)
    context = _format_context(hits, max_context_chars=max_context_chars)
    history = _format_history(messages[:-1], history_messages=history_messages)
    dictionary_instruction = _text_mode_dictionary_fallback_instruction(dictionary_fallback_metadata)
    return [
        PromptSection("conversation_history", "Recent conversation", history),
        PromptSection("user_question", "Question", question),
        PromptSection("retrieved_contexts", "Retrieved contexts", context),
        PromptSection(
            "dictionary_fallback_guidance",
            "Dictionary fallback guidance",
            dictionary_instruction,
            enabled=bool(dictionary_instruction.strip()),
        ),
        PromptSection("answer_contract", "Answer", "Answer:"),
    ]


def _build_chat_rag_messages_from_sections(sections: Sequence[PromptSection], *, language: str | None = None) -> list[dict[str, str]]:
    language_instruction = _language_instruction(language)
    return [
        {"role": "system", "content": _join_prompt_parts(SYSTEM_PROMPT, language_instruction)},
        {"role": "user", "content": _render_prompt_sections(sections)},
    ]


def _render_prompt_sections(sections: Sequence[PromptSection]) -> str:
    rendered: list[str] = []
    for section in sections:
        if not section.enabled:
            continue
        content = section.content.strip()
        if not content:
            continue
        rendered.append(f"### {section.section_id}: {section.title}\n{content}")
    return "\n\n".join(rendered).strip()


def _prompt_sections_metadata(sections: Sequence[PromptSection]) -> dict[str, Any]:
    return {
        "schema": PROMPT_SECTION_SCHEMA_VERSION,
        "sections": [
            {
                "id": section.section_id,
                "title": section.title,
                "enabled": section.enabled,
                "chars": len(section.content),
            }
            for section in sections
        ],
    }


def _text_mode_dictionary_fallback_instruction(metadata: dict[str, Any] | None) -> str:
    if not metadata:
        return ""
    query_plan = metadata.get("query_plan") if isinstance(metadata, dict) else None
    if not isinstance(query_plan, dict):
        return ""
    target_terms = [str(term).strip() for term in query_plan.get("target_terms") or [] if str(term).strip()]
    target = ", ".join(target_terms) if target_terms else "the target term"
    guidance = (
        "Dictionary fallback guidance:\n"
        f"- Target term(s): {target}.\n"
        "- Preserve the target term(s) exactly as written above; do not change letters, digits, diacritics, casing, "
        "hyphens, or Roman-numeral suffixes.\n"
        "- Do not merge the target with a nearby but different acronym. If retrieved entries mention a near-match "
        "instead, say it is a near-match rather than treating it as the target.\n"
        "- If a retrieved dictionary entry directly matches the target term, do not answer that the target was not found. "
        "Summarize the cited entry and state only unsupported details as missing.\n"
        "- Some retrieved contexts are local dictionary entries. For short acronym or term-only questions, "
        "distinguish a formal definition/alias from occurrence evidence.\n"
        "- If the target term is mentioned in retrieved dictionary entries but is not explicitly defined, "
        "do not answer that the context is unusable. State that no formal definition or expansion is shown in the "
        "retrieved entries, then say it appears in the explanation/body of the cited entry or entries.\n"
        "- Never refer to internal planned searches, fallback retrieval, source rank, or multiple retrieved entries as "
        "\"first/second/third questions\". If several dictionary entries match one short acronym, present them as "
        "possible dictionary entries or senses for the same user question.\n"
        "- If a retrieved entry is only a short cross-reference such as \"HEADWORD nh TARGET\", \"HEADWORD xem TARGET\", "
        "or \"HEADWORD đồng nghĩa với TARGET\", treat it as an alias/redirect to TARGET. Do not list both entries as "
        "separate definitions when the target entry is also retrieved; merge the alias into the target item.\n"
        "- Do not infer an expansion, alias, or meaning unless the retrieved entries explicitly support it.\n\n"
    )
    if _is_plural_type_category_query_plan(query_plan):
        guidance += (
            "Category/list-query guard:\n"
            "- The user is asking for types/categories. List only retrieved dictionary entries or typed relations that "
            "directly name supported types of the target term.\n"
            "- If the retrieved entries do not contain a complete classification, say the list is incomplete.\n"
            "- Do not add common examples, safety advice, or public/general-world categories from outside the retrieved context.\n\n"
        )
    tool_prompt = render_dictionary_tool_plan_prompt(
        query_plan,
        original_query=str(query_plan.get("query") or target or ""),
    )
    if tool_prompt:
        guidance += f"{tool_prompt}\n\n"
    return guidance


def build_dictionary_rag_messages(
    messages: list[dict[str, Any]],
    hits: list[RetrievalHit],
    *,
    query: str,
    max_context_chars: int,
    history_messages: int,
    language: str | None = None,
    query_plan: DictionaryQueryPlan | None = None,
) -> list[dict[str, str]]:
    sections = build_dictionary_rag_prompt_sections(
        messages,
        hits,
        query=query,
        max_context_chars=max_context_chars,
        history_messages=history_messages,
        query_plan=query_plan,
    )
    return _build_dictionary_rag_messages_from_sections(sections, language=language)


def build_dictionary_rag_prompt_sections(
    messages: list[dict[str, Any]],
    hits: list[RetrievalHit],
    *,
    query: str,
    max_context_chars: int,
    history_messages: int,
    query_plan: DictionaryQueryPlan | None = None,
) -> list[PromptSection]:
    context = _format_context(hits, max_context_chars=max_context_chars)
    history = _format_history(messages[:-1], history_messages=history_messages)
    plan_instruction = dictionary_plan_prompt_instructions(query_plan) if query_plan is not None else ""
    plan_summary = ""
    if query_plan is not None:
        target_terms = ", ".join(query_plan.target_terms) if query_plan.target_terms else "not detected"
        alias_evidence_summary = _alias_prompt_evidence_summary(hits, query_plan)
        final_instruction = _dictionary_final_instruction(query_plan)
        plan_summary = (
            f"Detected dictionary task: {query_plan.intent.value}\n"
            f"Target terms: {target_terms}\n"
            f"Answer style: {query_plan.answer_style}\n\n"
            f"Task instructions:\n{plan_instruction}\n\n"
            f"{render_dictionary_tool_plan_prompt(query_plan, original_query=query)}\n\n"
            f"{alias_evidence_summary}"
        )
    else:
        final_instruction = _dictionary_final_instruction(None)
    return [
        PromptSection("conversation_history", "Recent conversation", history),
        PromptSection("dictionary_question", "Dictionary question", query),
        PromptSection("retrieved_dictionary_entries", "Retrieved dictionary entries", context),
        PromptSection(
            "dictionary_task_plan",
            "Dictionary task plan",
            plan_summary,
            enabled=bool(plan_summary.strip()),
        ),
        PromptSection("answer_contract", "Answer contract", final_instruction),
    ]


def _build_dictionary_rag_messages_from_sections(
    sections: Sequence[PromptSection],
    *,
    language: str | None = None,
) -> list[dict[str, str]]:
    response_language = _normalize_response_language(language)
    language_instruction = _language_instruction(response_language)
    return [
        {
            "role": "system",
            "content": _join_prompt_parts(
                "You are a careful military dictionary assistant. Use the retrieved local dictionary entries first. "
                "Keep target abbreviations, letters, digits, casing, Roman-numeral suffixes, and Vietnamese diacritics intact. "
                "Do not rewrite one acronym into a nearby acronym. Cite sources as [entry-id].",
                language_instruction,
                RESPONSE_FORMAT_GUIDANCE,
            ),
        },
        {"role": "user", "content": _render_prompt_sections(sections)},
    ]


def _dictionary_final_instruction(query_plan: DictionaryQueryPlan | None) -> str:
    if query_plan is not None and query_plan.intent == DictionaryQueryIntent.ALIAS:
        return (
            "Answer the alias/name question directly in the required response language. "
            "Start with supported alternate names only when alias evidence is present. "
            "If no alias evidence is present, state that no supported alias/tên gọi khác was found in the retrieved sources. "
            "Do not turn the answer into a long definition. Cite dictionary entries with their ids in square brackets. "
            "When listing several entries or senses, use one numbered list and indent each entry's detail bullets under that numbered item. "
            "Do not invent content not supported by the retrieved dictionary entries."
        )
    return (
        "Explain the term in the required response language. Cite dictionary entries with their ids in square brackets. "
        "If a retrieved entry directly matches the target term, do not say the target was not found; summarize the cited entry and state only unsupported details as missing. "
        "If a retrieved entry is only a short cross-reference such as `HEADWORD nh TARGET`, `HEADWORD xem TARGET`, or `HEADWORD đồng nghĩa với TARGET`, merge it with the target entry instead of listing it as a separate definition. "
        "When listing several entries or senses, use one numbered list and indent each entry's detail bullets under that numbered item. "
        "Do not invent content not supported by the retrieved dictionary entries."
    )


def extract_alias_evidence_from_hits(
    hits: Sequence[RetrievalHit],
    *,
    target_terms: Sequence[str] = (),
) -> AliasEvidence:
    aliases: list[str] = []
    source_doc_ids: list[str] = []
    seen_aliases: set[str] = set()
    seen_doc_ids: set[str] = set()
    target_keys = {_alias_dedupe_key(term) for term in target_terms if _alias_dedupe_key(term)}
    for hit in hits:
        metadata = hit.metadata or {}
        if target_keys and not _hit_matches_alias_target(hit, target_keys):
            continue
        hit_aliases = _explicit_alias_values_from_metadata(metadata)
        accepted_for_hit = False
        for alias in hit_aliases:
            normalized = _normalize_alias_value(alias)
            if not normalized:
                continue
            alias_key = _alias_dedupe_key(normalized)
            if alias_key in seen_aliases:
                accepted_for_hit = True
                continue
            seen_aliases.add(alias_key)
            aliases.append(normalized)
            accepted_for_hit = True
        if accepted_for_hit and hit.doc_id not in seen_doc_ids:
            seen_doc_ids.add(hit.doc_id)
            source_doc_ids.append(hit.doc_id)
    return AliasEvidence(
        aliases=aliases,
        source_doc_ids=source_doc_ids,
        evidence_count=len(aliases),
        has_explicit_alias_evidence=bool(aliases),
    )


def _explicit_alias_values_from_metadata(metadata: dict[str, Any]) -> list[str]:
    aliases = _metadata_text_values(metadata.get("aliases"))
    aliases.extend(_metadata_text_values(metadata.get("alias_labels")))
    aliases.extend(_metadata_text_values(metadata.get("explicit_aliases")))
    aliases.extend(_alias_values_from_graph_path(metadata))
    aliases.extend(_alias_values_from_graph_edges(metadata))
    aliases.extend(_alias_values_from_single_edge_metadata(metadata))
    return aliases


def _alias_values_from_graph_edges(metadata: dict[str, Any]) -> list[str]:
    aliases: list[str] = []
    for key in ("dictionary_graph_edges", "graph_edges", "alias_edges"):
        value = metadata.get(key)
        if not isinstance(value, list):
            continue
        for edge in value:
            if not isinstance(edge, dict) or not _edge_is_supported_alias(edge):
                continue
            label = _normalize_alias_value(str(edge.get("target_label") or edge.get("label") or ""))
            if label:
                aliases.append(label)
    return aliases


def _alias_values_from_single_edge_metadata(metadata: dict[str, Any]) -> list[str]:
    edge_type = str(metadata.get("edge_type") or metadata.get("dictionary_relation") or "").strip()
    if edge_type != "has_alias":
        return []
    edge = {
        "type": edge_type,
        "target_label": metadata.get("target_label") or metadata.get("label") or metadata.get("alias_label"),
        "confidence": metadata.get("confidence"),
    }
    if not _edge_is_supported_alias(edge):
        return []
    label = _normalize_alias_value(str(edge.get("target_label") or ""))
    return [label] if label else []


def _edge_is_supported_alias(edge: dict[str, Any]) -> bool:
    edge_type = str(edge.get("type") or edge.get("edge_type") or edge.get("relation") or "").strip()
    if edge_type != "has_alias":
        return False
    label = _normalize_alias_value(str(edge.get("target_label") or edge.get("label") or edge.get("alias_label") or ""))
    if not label:
        return False
    confidence = edge.get("confidence")
    return confidence is None or _safe_float(confidence, default=ALIAS_EDGE_MIN_CONFIDENCE) >= ALIAS_EDGE_MIN_CONFIDENCE


def _alias_values_from_graph_path(metadata: dict[str, Any]) -> list[str]:
    path = metadata.get("dictionary_graph_path")
    if not isinstance(path, list):
        return []
    aliases: list[str] = []
    relation_is_alias = False
    for item in path:
        if not isinstance(item, dict):
            continue
        item_type = str(item.get("type") or "").strip().lower()
        item_label = str(item.get("label") or item.get("id") or "").strip()
        if item_type == "relation":
            relation_is_alias = item_label == "has_alias" or str(item.get("id") or "").strip() == "has_alias"
            continue
        if relation_is_alias and item_type == "alias" and item_label:
            aliases.append(item_label)
            relation_is_alias = False
    return aliases


def _hit_matches_alias_target(hit: RetrievalHit, target_keys: set[str]) -> bool:
    metadata = hit.metadata or {}
    candidates = [
        str(metadata.get("headword") or ""),
        str(hit.title or ""),
        str(hit.doc_id or ""),
    ]
    for item in metadata.get("dictionary_graph_path", []):
        if isinstance(item, dict) and str(item.get("type") or "").strip().lower() == "entry":
            candidates.append(str(item.get("label") or item.get("id") or ""))
    return any(_alias_dedupe_key(candidate) in target_keys for candidate in candidates if candidate)


def _normalize_alias_value(value: str) -> str:
    value = re.sub(r"\s+", " ", str(value or "")).strip()
    return value.strip(" ;,")


def _alias_dedupe_key(value: str) -> str:
    normalized = unicodedata.normalize("NFKD", _normalize_alias_value(value))
    without_marks = "".join(ch for ch in normalized if not unicodedata.combining(ch))
    return without_marks.casefold()


def _metadata_text_values(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [value] if value.strip() else []
    if isinstance(value, (list, tuple, set)):
        return [str(item) for item in value if str(item).strip()]
    return []


def _alias_evidence_summary(
    hits: Sequence[RetrievalHit],
    *,
    alias_evidence: AliasEvidence | None = None,
) -> dict[str, Any]:
    alias_evidence = alias_evidence or extract_alias_evidence_from_hits(hits)
    evidence_hits = []
    alias_doc_ids = set(alias_evidence.source_doc_ids)
    for hit in hits:
        if hit.doc_id not in alias_doc_ids:
            continue
        metadata = hit.metadata or {}
        evidence_hits.append(
            {
                "doc_id": hit.doc_id,
                "rank": hit.rank,
                "alias_evidence_count": len(_explicit_alias_values_from_metadata(metadata)) or 1,
                "query_plan_role": metadata.get("query_plan_role"),
            }
        )
    return {
        "has_alias_evidence": alias_evidence.has_explicit_alias_evidence,
        "has_explicit_alias_evidence": alias_evidence.has_explicit_alias_evidence,
        "alias_evidence_count": alias_evidence.evidence_count,
        "alias_evidence_hit_count": len(evidence_hits),
        "alias_evidence_doc_count": len(alias_evidence.source_doc_ids),
        "alias_evidence_doc_ids": list(alias_evidence.source_doc_ids),
        "evidence_hits": evidence_hits,
    }


def _alias_prompt_evidence_summary(hits: list[RetrievalHit], query_plan: DictionaryQueryPlan) -> str:
    if query_plan.intent != DictionaryQueryIntent.ALIAS:
        return ""
    alias_evidence = extract_alias_evidence_from_hits(hits, target_terms=query_plan.target_terms)
    summary = _alias_evidence_summary(hits, alias_evidence=alias_evidence)
    if alias_evidence.has_explicit_alias_evidence:
        alias_lines = "\n".join(
            f"- {alias} {_format_source_citations(alias_evidence.source_doc_ids)}" for alias in alias_evidence.aliases
        )
        return (
            "Explicit alias evidence:\n"
            f"{alias_lines}\n"
            f"- Alias evidence hit count: {summary['alias_evidence_hit_count']}\n"
            f"- Alias evidence marker count: {summary['alias_evidence_count']}\n"
            "- Answer only from the explicit alias evidence block; do not infer aliases from related terms, concepts, categories, or see-also links.\n\n"
        )
    return (
        "Alias evidence summary:\n"
        "- Alias evidence hit count: 0\n"
        "- No retrieved hit is marked as alias evidence.\n"
        "- State cautiously that no explicitly marked alias/tên gọi khác was found in the retrieved sources; do not infer aliases from related terms, concepts, categories, or see-also links.\n\n"
    )


def _safe_int(value: Any) -> int:
    if isinstance(value, bool):
        return int(value)
    try:
        return max(0, int(value))
    except (TypeError, ValueError):
        return 0


def _safe_float(value: Any, *, default: float) -> float:
    if value is None:
        return default
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return default
    if not math.isfinite(parsed):
        return default
    return parsed


def _messages_data_tier(messages: list[dict[str, Any]]) -> DataTier:
    tiers: list[DataTier] = []
    for message in messages:
        if not isinstance(message, dict):
            continue
        if message.get("data_tier") not in (None, ""):
            tiers.append(normalize_data_tier(message.get("data_tier")))
        metadata = message.get("metadata")
        if isinstance(metadata, dict) and metadata.get("data_tier") not in (None, ""):
            tiers.append(normalize_data_tier(metadata.get("data_tier")))
        attachments = message.get("attachments")
        if isinstance(attachments, list):
            for attachment in attachments:
                if isinstance(attachment, dict) and attachment.get("data_tier") not in (None, ""):
                    tiers.append(normalize_data_tier(attachment.get("data_tier")))
    return max_data_tier(*tiers)


def last_user_text(messages: list[dict[str, Any]]) -> str:
    for message in reversed(messages):
        if message.get("role") == "user":
            content = message.get("content", "")
            text = _content_to_text(content)
            if text.strip():
                return text.strip()
    raise ValueError("At least one user message with text content is required.")


def parse_chat_command(text: str) -> tuple[str, str] | None:
    stripped = text.strip()
    if not stripped.startswith("/"):
        return None
    command, _, remainder = stripped.partition(" ")
    normalized = command.lower()
    if normalized in {"/img", "/image"}:
        return "img", remainder.strip()
    if normalized in {"/dict", "/dictionary", "/tu-dien", "/từ-điển"}:
        return "dict", remainder.strip()
    return None


def _normalize_response_mode(value: str | None) -> str:
    if value is None or value == "":
        return "text"
    if not isinstance(value, str):
        raise ValueError("response_mode must be a string")
    normalized = value.strip().lower().replace("+", "_").replace("-", "_")
    if normalized in {"text", "rag", "chat"}:
        return "text"
    if normalized in {"image", "images", "img"}:
        return "image"
    if normalized in {"text_image", "text_images", "text_and_image", "text_and_images", "mixed"}:
        return "text_image"
    if normalized in {"dictionary", "dict", "tu_dien", "từ_điển"}:
        return "dictionary"
    raise ValueError("response_mode must be one of: text, image, text_image, dictionary")


def _normalize_response_language(value: str | None) -> str:
    if value is None:
        return "en"
    if not isinstance(value, str):
        raise ValueError("language must be a string")
    normalized = value.strip().lower().replace("_", "-")
    if normalized in {"vi", "vi-vn", "vietnamese", "tieng-viet", "tiếng-việt"}:
        return "vi"
    if normalized in {"en", "en-us", "en-gb", "english"}:
        return "en"
    if normalized in {"", "system", "auto"}:
        return "en"
    raise ValueError("language must be one of: en, vi")


def _language_instruction(language: str | None) -> str:
    normalized = _normalize_response_language(language)
    if normalized == "vi":
        return (
            "Required response language: Vietnamese. "
            "Answer only in Vietnamese, regardless of the user's input language, unless quoting source text."
        )
    return (
        "Required response language: English. "
        "Answer only in English, regardless of the user's input language, unless quoting source text."
    )


def _join_prompt_parts(*parts: str) -> str:
    return "\n\n".join(part.strip() for part in parts if part and part.strip())


def _strip_command_prefix(text: str) -> str:
    command = parse_chat_command(text)
    if command and command[0] in {"img", "dict"}:
        return command[1].strip()
    return text.strip()


def _clean_image_query(text: str) -> str:
    cleaned = text.strip()
    if not cleaned:
        return ""
    if cleaned.startswith("```"):
        lines = cleaned.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip().startswith("```"):
            lines = lines[:-1]
        cleaned = "\n".join(lines).strip()
    try:
        parsed = json.loads(cleaned)
    except json.JSONDecodeError:
        parsed = None
    if isinstance(parsed, str):
        cleaned = parsed
    elif isinstance(parsed, list):
        first = next((item for item in parsed if isinstance(item, str) and item.strip()), "")
        cleaned = first
    cleaned = cleaned.strip().strip("\"'")
    lines = [line.strip("-* \t") for line in cleaned.splitlines() if line.strip()]
    return (lines[0] if lines else cleaned)[:180].strip()


def _format_context(hits: list[RetrievalHit], *, max_context_chars: int) -> str:
    raw_blocks: list[str] = []
    for hit in hits:
        title = f"{hit.title}\n" if hit.title else ""
        redirect_target = _dictionary_redirect_target(hit)
        redirect_note = (
            f"\nAlias/cross-reference note: this entry redirects to {redirect_target}; "
            "do not list it as a separate definition if the target entry is also retrieved."
            if redirect_target
            else ""
        )
        redirect_aliases = _dictionary_redirect_aliases_for_hit(hit)
        merged_redirect_note = (
            "\nMerged redirect aliases for this canonical entry: "
            + "; ".join(redirect_aliases)
            + ". Treat these as cross-references to this entry, not separate definitions."
            if redirect_aliases
            else ""
        )
        block = f"[{hit.doc_id}]\n{title}{hit.text}{redirect_note}{merged_redirect_note}".strip()
        if block:
            raw_blocks.append(block)
    if not raw_blocks:
        return "No retrieved context."
    if max_context_chars <= 0:
        return CONTEXT_SEPARATOR.join(block.splitlines()[0] for block in raw_blocks)
    separator_budget = len(CONTEXT_SEPARATOR) * max(0, len(raw_blocks) - 1)
    text_budget = max(1, max_context_chars - separator_budget)
    base_budget = max(1, text_budget // len(raw_blocks))
    extra_budget = text_budget % len(raw_blocks)
    carried_budget = 0
    context_blocks: list[str] = []
    for index, block in enumerate(raw_blocks):
        block_budget = base_budget + (1 if index < extra_budget else 0) + carried_budget
        formatted = _truncate_context_block(block, max_chars=block_budget)
        context_blocks.append(formatted)
        carried_budget = max(0, block_budget - len(formatted))
    return CONTEXT_SEPARATOR.join(context_blocks)


def _truncate_context_block(block: str, *, max_chars: int) -> str:
    if len(block) <= max_chars:
        return block
    lines = block.splitlines()
    header = "\n".join(lines[:2]).strip() if len(lines) >= 2 else lines[0].strip()
    if max_chars <= len(header):
        return lines[0].strip() if lines else block[:max_chars].rstrip()
    text_budget = max_chars - len(header) - 1
    body = "\n".join(lines[2:] if len(lines) >= 2 else lines[1:]).strip()
    return f"{header}\n{body[:text_budget].rstrip()}".strip()


def _format_image_answer(query: str, hits: list[RetrievalHit], *, language: str | None = None) -> str:
    response_language = _normalize_response_language(language)
    if not hits:
        if response_language == "vi":
            return f"Không tìm thấy kết quả ảnh cho '{query}'."
        return f"No image results found for '{query}'."
    if response_language == "vi":
        return f"Tìm thấy {len(hits)} kết quả ảnh cho '{query}'."
    return f"Found {len(hits)} image result(s) for '{query}'."


def _format_alias_answer(alias_evidence: AliasEvidence, hits: Sequence[RetrievalHit], *, language: str | None = None) -> str:
    response_language = "vi" if language is None else _normalize_response_language(language)
    if alias_evidence.has_explicit_alias_evidence:
        aliases = "; ".join(alias_evidence.aliases)
        citations = _format_source_citations(alias_evidence.source_doc_ids)
        if response_language == "vi":
            return f"Theo các nguồn được truy hồi, tên gọi khác/alias được ghi nhận trong nguồn là: {aliases}. {citations}".strip()
        return f"Supported alternate names/aliases recorded in the retrieved sources: {aliases}. {citations}".strip()
    fallback_doc_ids = [hit.doc_id for hit in hits]
    citations = _format_source_citations(fallback_doc_ids)
    if response_language == "vi":
        return (
            "Không tìm thấy tên gọi khác/alias được đánh dấu rõ ràng trong các nguồn đã truy hồi. "
            f"Các nguồn hiện có chỉ cung cấp định nghĩa hoặc thông tin liên quan. {citations}"
        ).strip()
    return (
        "No supported alternate name/alias was found in the retrieved sources. "
        f"The available sources only provide definitions or related information. {citations}"
    ).strip()


def _looks_like_grounding_refusal(answer: str) -> bool:
    cleaned = str(answer or "").strip()
    if not cleaned:
        return True
    if re.search(r"\[[^\]]+\]", cleaned):
        return False
    folded = _fold_prompt_text(cleaned)
    refusal_markers = (
        "i do not know",
        "i don't know",
        "cannot answer",
        "can't answer",
        "not enough information",
        "insufficient context",
        "khong the tra loi",
        "khong du thong tin",
        "khong co du thong tin",
        "khong biet",
        "khong tim thay dinh nghia",
        "khong tim thay thong tin",
        "khong co dinh nghia",
        "khong co thong tin chinh xac",
        "khong co dinh nghia chinh thuc",
        "khong the xac dinh",
        "dua tren cac van ban duoc cung cap",
        "dua tren ngu canh duoc cung cap",
        "no definition was found",
        "no exact definition",
        "no precise information",
        "not found in the provided",
    )
    return any(marker in folded for marker in refusal_markers)


def _looks_like_internal_query_leak(answer: str) -> bool:
    folded = _fold_prompt_text(answer)
    leak_markers = (
        "cau hoi dau tien",
        "cau hoi thu nhat",
        "cau hoi thu hai",
        "cau hoi thu ba",
        "first question",
        "second question",
        "third question",
        "query 1",
        "query 2",
        "query 3",
        "internal query",
        "planned search",
    )
    return any(marker in folded for marker in leak_markers)


def _answer_markdown_section_is_empty(answer: str) -> bool:
    source = str(answer or "")
    match = re.search(
        r"(?im)^\s{0,3}#{1,6}\s*(?:câu\s*trả\s*lời|cau\s*tra\s*loi|answer)\s*:?\s*$",
        source,
    )
    if not match:
        return False
    next_heading = re.search(r"(?im)^\s{0,3}#{1,6}\s+\S.*$", source[match.end() :])
    body = source[match.end() : match.end() + next_heading.start()] if next_heading else source[match.end() :]
    cleaned_lines = []
    for line in body.splitlines():
        stripped = re.sub(r"^\s*(?:[-*]|\d+[.)])\s*", "", line).strip()
        if stripped:
            cleaned_lines.append(stripped)
    return not cleaned_lines


def _format_dictionary_empty_answer_section_fallback(
    question: str,
    hits: Sequence[RetrievalHit],
    metadata: dict[str, Any] | None,
    *,
    language: str | None = None,
) -> str:
    return (
        _format_dictionary_category_fallback_answer(question, hits, metadata, language=language)
        or _format_dictionary_plural_phrase_list_fallback_answer(question, hits, metadata, language=language)
        or _format_dictionary_redirect_lookup_fallback_answer(question, hits, metadata, language=language)
        or _format_dictionary_occurrence_fallback_answer(question, hits, metadata, language=language)
    )


def _fold_prompt_text(text: str) -> str:
    normalized = unicodedata.normalize("NFD", text)
    stripped = "".join(ch for ch in normalized if unicodedata.category(ch) != "Mn")
    stripped = stripped.replace("đ", "d").replace("Đ", "D")
    return re.sub(r"\s+", " ", stripped).strip().lower()


def _format_dictionary_category_fallback_answer(
    question: str,
    hits: Sequence[RetrievalHit],
    metadata: dict[str, Any] | None,
    *,
    language: str | None = None,
) -> str:
    query_plan = metadata.get("query_plan") if isinstance(metadata, dict) else None
    if not _is_plural_type_category_query_plan(query_plan):
        return ""
    target_terms = _target_terms_from_query_plan(query_plan)
    if not target_terms:
        return ""
    response_language = _normalize_response_language(language)
    target = target_terms[0]
    type_hits = _collapse_dictionary_redirect_hits(_dictionary_direct_type_hits(hits, target_terms))
    base_hits = _dictionary_base_category_hits(hits, target_terms)
    citations = _format_source_citations([hit.doc_id for hit in (*type_hits, *base_hits)])
    if not type_hits:
        base_titles = _format_source_title_citations(base_hits) if base_hits else ""
        if response_language == "vi":
            if base_titles:
                return (
                    f"Tôi thấy mục từ nền liên quan đến “{target}” trong từ điển: {base_titles}. "
                    "Các nguồn truy hồi hiện chưa cung cấp một danh sách loại/mẫu cụ thể đủ chắc để liệt kê. "
                    f"{citations}"
                ).strip()
            return (
                f"Các nguồn truy hồi hiện chưa cung cấp danh sách loại/mẫu cụ thể cho “{target}”."
            )
        if base_titles:
            return (
                f"I found the base dictionary entry related to “{target}”: {base_titles}. "
                "However, the retrieved sources do not provide a sufficiently grounded list of specific types. "
                f"{citations}"
            ).strip()
        return (
            f"The retrieved sources do not provide a grounded list of specific types for “{target}”."
        )

    listed_hits = type_hits[:8]
    if response_language == "vi":
        lines = [
            f"Các mục từ truy hồi hỗ trợ một số loại “{target}” sau "
            "(đây không phải bảng phân loại đầy đủ):"
        ]
        for index, hit in enumerate(listed_hits, 1):
            title = str(hit.title or hit.doc_id).strip()
            aliases = _dictionary_redirect_aliases_for_hit(hit)
            alias_suffix = f" (còn được trỏ tới bởi: {', '.join(aliases)})" if aliases else ""
            lines.append(f"{index}. **{title}**{alias_suffix} [{hit.doc_id}]")
        if len(type_hits) > len(listed_hits):
            lines.append(f"Còn {len(type_hits) - len(listed_hits)} mục phù hợp trực tiếp khác trong phần nguồn liên quan.")
        return "\n\n".join(lines).strip()

    lines = [
        f"The retrieved entries support the following types of “{target}” "
        "(not a complete taxonomy):"
    ]
    for index, hit in enumerate(listed_hits, 1):
        title = str(hit.title or hit.doc_id).strip()
        aliases = _dictionary_redirect_aliases_for_hit(hit)
        alias_suffix = f" (also referenced by: {', '.join(aliases)})" if aliases else ""
        lines.append(f"{index}. **{title}**{alias_suffix} [{hit.doc_id}]")
    if len(type_hits) > len(listed_hits):
        lines.append(f"{len(type_hits) - len(listed_hits)} additional direct matches are available in the related sources.")
    return "\n\n".join(lines).strip()


def _format_dictionary_plural_phrase_list_fallback_answer(
    question: str,
    hits: Sequence[RetrievalHit],
    metadata: dict[str, Any] | None,
    *,
    language: str | None = None,
) -> str:
    query_plan = metadata.get("query_plan") if isinstance(metadata, dict) else None
    if not _is_plural_phrase_list_query_plan(query_plan):
        return ""
    target_terms = _target_terms_from_query_plan(query_plan)
    if not target_terms:
        return ""
    response_language = _normalize_response_language(language)
    target = target_terms[0]
    phrase_hits = _collapse_dictionary_redirect_hits(_dictionary_headword_prefix_hits(hits, target_terms))
    if len(phrase_hits) < 2:
        return ""
    listed_hits = phrase_hits[:8]
    if response_language == "vi":
        lines = [
            f"Các mục từ truy hồi bắt đầu bằng “{target}” gồm:"
        ]
        for index, hit in enumerate(listed_hits, 1):
            title = str(hit.title or hit.doc_id).strip()
            aliases = _dictionary_redirect_aliases_for_hit(hit)
            alias_suffix = f" (còn được trỏ tới bởi: {', '.join(aliases)})" if aliases else ""
            lines.append(f"{index}. **{title}**{alias_suffix} [{hit.doc_id}]")
        if len(phrase_hits) > len(listed_hits):
            lines.append(f"Còn {len(phrase_hits) - len(listed_hits)} mục phù hợp trực tiếp khác trong phần nguồn liên quan.")
        return "\n\n".join(lines).strip()

    lines = [
        f"The retrieved entries that start with “{target}” are:"
    ]
    for index, hit in enumerate(listed_hits, 1):
        title = str(hit.title or hit.doc_id).strip()
        aliases = _dictionary_redirect_aliases_for_hit(hit)
        alias_suffix = f" (also referenced by: {', '.join(aliases)})" if aliases else ""
        lines.append(f"{index}. **{title}**{alias_suffix} [{hit.doc_id}]")
    if len(phrase_hits) > len(listed_hits):
        lines.append(f"{len(phrase_hits) - len(listed_hits)} additional direct matches are available in the related sources.")
    return "\n\n".join(lines).strip()


def _format_dictionary_redirect_lookup_fallback_answer(
    question: str,
    hits: Sequence[RetrievalHit],
    metadata: dict[str, Any] | None,
    *,
    language: str | None = None,
) -> str:
    query_plan = metadata.get("query_plan") if isinstance(metadata, dict) else None
    target_terms = _target_terms_from_query_plan(query_plan)
    target_keys = {_fold_prompt_text(term) for term in target_terms if _fold_prompt_text(term)}
    redirect_hit = next(
        (
            hit
            for hit in hits
            if _dictionary_redirect_target(hit)
            and (
                bool((hit.metadata or {}).get("dictionary_preserved_redirect"))
                or not target_keys
                or _dictionary_hit_headword_key(hit) in target_keys
            )
        ),
        None,
    )
    if redirect_hit is None:
        return ""
    response_language = _normalize_response_language(language)
    redirect_target = _dictionary_redirect_target(redirect_hit)
    target_hit = _dictionary_redirect_target_hit(redirect_hit, hits)
    alias_title = str(redirect_hit.title or (redirect_hit.metadata or {}).get("headword") or redirect_hit.doc_id).strip()
    alias_citation = _format_source_citations([redirect_hit.doc_id])
    target_title = str((target_hit.title if target_hit else "") or redirect_target).strip()
    target_citation = _format_source_citations([target_hit.doc_id] if target_hit else [])
    target_summary = _dictionary_hit_lead_summary(target_hit) if target_hit is not None else ""
    if response_language == "vi":
        parts = [
            f"“{alias_title}” là mục từ tham chiếu trong từ điển: mục này trỏ tới **{target_title}**."
        ]
        if target_summary:
            parts.append(f"Nội dung nên đọc theo mục được trỏ tới: {target_summary}")
        citations = ", ".join(part for part in (alias_citation, target_citation) if part)
        if citations:
            parts.append(citations)
        return " ".join(parts).strip()
    parts = [
        f"“{alias_title}” is a dictionary cross-reference: it points to **{target_title}**."
    ]
    if target_summary:
        parts.append(f"Read the meaning through the referenced entry: {target_summary}")
    citations = ", ".join(part for part in (alias_citation, target_citation) if part)
    if citations:
        parts.append(citations)
    return " ".join(parts).strip()


def _is_plural_type_category_query_plan(query_plan: Any) -> bool:
    if isinstance(query_plan, DictionaryQueryPlan):
        return (
            query_plan.intent == DictionaryQueryIntent.CATEGORY
            and str(query_plan.normalization.get("target_layer") or "") == "plural_type_lookup_wrapper"
        )
    if not isinstance(query_plan, dict):
        return False
    normalization = query_plan.get("normalization")
    return (
        str(query_plan.get("intent") or "") == DictionaryQueryIntent.CATEGORY.value
        and isinstance(normalization, dict)
        and str(normalization.get("target_layer") or "") == "plural_type_lookup_wrapper"
    )


def _is_plural_phrase_list_query_plan(query_plan: Any) -> bool:
    if isinstance(query_plan, DictionaryQueryPlan):
        normalization = query_plan.normalization
        query = query_plan.query
        target_terms = query_plan.target_terms
    elif isinstance(query_plan, dict):
        normalization = query_plan.get("normalization") if isinstance(query_plan.get("normalization"), dict) else {}
        query = str(query_plan.get("query") or "")
        target_terms = [str(term) for term in query_plan.get("target_terms") or []]
    else:
        return False
    if str(normalization.get("target_layer") or "") != "regex_lookup_wrapper":
        return False
    if not target_terms:
        return False
    folded_query = _fold_prompt_text(query)
    return bool(re.match(r"^(?:cac|nhung)\s+\S+", folded_query))


def _target_terms_from_query_plan(query_plan: Any) -> list[str]:
    if isinstance(query_plan, DictionaryQueryPlan):
        return [str(term).strip() for term in query_plan.target_terms if str(term).strip()]
    if not isinstance(query_plan, dict):
        return []
    return [str(term).strip() for term in query_plan.get("target_terms") or [] if str(term).strip()]


def _dictionary_direct_type_hits(
    hits: Sequence[RetrievalHit],
    target_terms: Sequence[str],
) -> list[RetrievalHit]:
    target_keys = {_fold_prompt_text(term) for term in target_terms if _fold_prompt_text(term)}
    result: list[RetrievalHit] = []
    seen: set[str] = set()
    for hit in hits:
        doc_id = str(hit.doc_id or "").strip()
        if not doc_id or doc_id in seen:
            continue
        headword_key = _dictionary_hit_headword_key(hit)
        if not headword_key:
            continue
        if any(target and headword_key.startswith(f"{target} ") for target in target_keys):
            seen.add(doc_id)
            result.append(hit)
    return result


def _dictionary_base_category_hits(
    hits: Sequence[RetrievalHit],
    target_terms: Sequence[str],
) -> list[RetrievalHit]:
    target_keys = {_fold_prompt_text(term) for term in target_terms if _fold_prompt_text(term)}
    result: list[RetrievalHit] = []
    seen: set[str] = set()
    for hit in hits:
        doc_id = str(hit.doc_id or "").strip()
        if not doc_id or doc_id in seen:
            continue
        if _dictionary_hit_headword_key(hit) in target_keys:
            seen.add(doc_id)
            result.append(hit)
    return result


def _dictionary_headword_prefix_hits(
    hits: Sequence[RetrievalHit],
    target_terms: Sequence[str],
) -> list[RetrievalHit]:
    target_keys = {_fold_prompt_text(term) for term in target_terms if _fold_prompt_text(term)}
    result: list[RetrievalHit] = []
    seen: set[str] = set()
    for hit in hits:
        doc_id = str(hit.doc_id or "").strip()
        if not doc_id or doc_id in seen:
            continue
        headword_key = _dictionary_hit_headword_key(hit)
        if not headword_key:
            continue
        if any(target and headword_key.startswith(f"{target} ") for target in target_keys):
            seen.add(doc_id)
            result.append(hit)
    return result


def _collapse_dictionary_redirect_hits(
    hits: Sequence[RetrievalHit],
    *,
    preserve_headword_terms: Sequence[str] = (),
) -> list[RetrievalHit]:
    preserve_headword_keys = {_fold_prompt_text(term) for term in preserve_headword_terms if _fold_prompt_text(term)}
    by_headword_key = {_dictionary_hit_headword_key(hit): hit for hit in hits if _dictionary_hit_headword_key(hit)}
    aliases_by_target_doc_id: dict[str, list[str]] = {}
    alias_doc_ids_by_target_doc_id: dict[str, list[str]] = {}
    collapsed: list[RetrievalHit] = []
    for hit in hits:
        redirect_target = _dictionary_redirect_target(hit)
        redirect_target_key = _fold_prompt_text(redirect_target) if redirect_target else ""
        target_hit = by_headword_key.get(redirect_target_key)
        if target_hit is not None and target_hit.doc_id != hit.doc_id:
            if (
                (hit.metadata or {}).get("dictionary_preserved_redirect")
                or _dictionary_hit_headword_key(hit) in preserve_headword_keys
            ):
                metadata = dict(hit.metadata or {})
                metadata["dictionary_preserved_redirect"] = True
                metadata["dictionary_redirect_target"] = redirect_target
                metadata["dictionary_redirect_target_doc_id"] = target_hit.doc_id
                metadata["dictionary_redirect_target_title"] = str(target_hit.title or target_hit.doc_id).strip()
                collapsed.append(replace(hit, metadata=metadata))
                continue
            alias_title = str(hit.title or (hit.metadata or {}).get("headword") or hit.doc_id).strip()
            if alias_title:
                aliases = aliases_by_target_doc_id.setdefault(target_hit.doc_id, [])
                if alias_title not in aliases:
                    aliases.append(alias_title)
            alias_doc_ids = alias_doc_ids_by_target_doc_id.setdefault(target_hit.doc_id, [])
            if hit.doc_id not in alias_doc_ids:
                alias_doc_ids.append(hit.doc_id)
            continue
        collapsed.append(hit)
    if not aliases_by_target_doc_id:
        return collapsed
    updated: list[RetrievalHit] = []
    for hit in collapsed:
        aliases = aliases_by_target_doc_id.get(hit.doc_id)
        if not aliases:
            updated.append(hit)
            continue
        metadata = dict(hit.metadata or {})
        existing_aliases = [str(alias) for alias in metadata.get("dictionary_redirect_aliases") or [] if str(alias).strip()]
        for alias in aliases:
            if alias not in existing_aliases:
                existing_aliases.append(alias)
        metadata["dictionary_redirect_aliases"] = existing_aliases
        alias_doc_ids = alias_doc_ids_by_target_doc_id.get(hit.doc_id) or []
        existing_alias_doc_ids = [
            str(doc_id) for doc_id in metadata.get("dictionary_redirect_doc_ids") or [] if str(doc_id).strip()
        ]
        for doc_id in alias_doc_ids:
            if doc_id not in existing_alias_doc_ids:
                existing_alias_doc_ids.append(doc_id)
        if existing_alias_doc_ids:
            metadata["dictionary_redirect_doc_ids"] = existing_alias_doc_ids
        updated.append(replace(hit, metadata=metadata))
    return updated


def _canonicalize_dictionary_redirect_hits(
    hits: Sequence[RetrievalHit],
    *,
    preserve_headword_terms: Sequence[str] = (),
) -> list[RetrievalHit]:
    preserve_headword_keys = {_fold_prompt_text(term) for term in preserve_headword_terms if _fold_prompt_text(term)}
    collapsed = _collapse_dictionary_redirect_hits(hits, preserve_headword_terms=preserve_headword_terms)
    if preserve_headword_keys:
        collapsed = sorted(
            collapsed,
            key=lambda hit: (
                0
                if (hit.metadata or {}).get("dictionary_preserved_redirect")
                and _dictionary_hit_headword_key(hit) in preserve_headword_keys
                else 1
            ),
        )
    return [replace(hit, rank=index) for index, hit in enumerate(collapsed, 1)]


def _dictionary_redirect_preserve_terms(query_plan: Any) -> list[str]:
    if query_plan is None:
        return []
    if isinstance(query_plan, DictionaryQueryPlan):
        if _is_plural_type_category_query_plan(query_plan) or _is_plural_phrase_list_query_plan(query_plan):
            return []
        return [str(term).strip() for term in query_plan.target_terms if str(term).strip()]
    if isinstance(query_plan, dict):
        if _is_plural_type_category_query_plan(query_plan) or _is_plural_phrase_list_query_plan(query_plan):
            return []
        return [str(term).strip() for term in query_plan.get("target_terms") or [] if str(term).strip()]
    return []


def _dictionary_redirect_aliases_for_hit(hit: RetrievalHit) -> list[str]:
    metadata = hit.metadata or {}
    return [str(alias).strip() for alias in metadata.get("dictionary_redirect_aliases") or [] if str(alias).strip()]


def _dictionary_redirect_target_hit(hit: RetrievalHit, hits: Sequence[RetrievalHit]) -> RetrievalHit | None:
    metadata = hit.metadata or {}
    target_doc_id = str(metadata.get("dictionary_redirect_target_doc_id") or "").strip()
    if target_doc_id:
        found = next((candidate for candidate in hits if candidate.doc_id == target_doc_id), None)
        if found is not None:
            return found
    target_key = _fold_prompt_text(_dictionary_redirect_target(hit))
    if not target_key:
        return None
    return next((candidate for candidate in hits if _dictionary_hit_headword_key(candidate) == target_key), None)


def _dictionary_hit_headword_key(hit: RetrievalHit) -> str:
    metadata = hit.metadata or {}
    headword = str(metadata.get("headword") or hit.title or "").strip()
    return _fold_prompt_text(headword)


def _dictionary_redirect_target(hit: RetrievalHit) -> str:
    metadata = hit.metadata or {}
    if str(metadata.get("kind") or "") != "dictionary" and not str(metadata.get("headword") or "").strip():
        return ""
    raw_text = normalize_spaces(str(metadata.get("raw_docx_text") or hit.text or ""))
    if not raw_text:
        return ""
    title = normalize_spaces(str(metadata.get("headword") or hit.title or ""))
    body = raw_text
    if title and body.casefold().startswith(title.casefold()):
        body = body[len(title) :].strip(" \t\n\r,:;.-–—")
    if not body or len(body) > 160:
        return ""
    match = re.match(
        r"(?is)^(?:nh|x\.?|xem(?:\s+thêm)?|đồng\s+nghĩa(?:\s+(?:với|là))?|dong\s+nghia(?:\s+(?:voi|la))?)\s+(.+?)\s*[.;,]*$",
        body,
    )
    if not match:
        return ""
    target = normalize_spaces(match.group(1)).strip(" .;,")
    if not target or len(target) > 100:
        return ""
    return target


def _format_dictionary_occurrence_fallback_answer(
    question: str,
    hits: Sequence[RetrievalHit],
    metadata: dict[str, Any] | None,
    *,
    language: str | None = None,
) -> str:
    if not hits:
        return ""
    response_language = _normalize_response_language(language)
    query_plan = metadata.get("query_plan") if isinstance(metadata, dict) else None
    target_terms = (
        [str(term).strip() for term in query_plan.get("target_terms") or [] if str(term).strip()]
        if isinstance(query_plan, dict)
        else []
    )
    target = target_terms[0] if target_terms else _strip_command_prefix(question).strip()
    direct_hits = [hit for hit in hits if _hit_has_direct_dictionary_match(hit)]
    direct_citations = _format_source_citations([hit.doc_id for hit in direct_hits])
    direct_titles = _format_source_titles(direct_hits)
    occurrence_hits = [
        hit
        for hit in hits
        if not _hit_has_direct_dictionary_match(hit) and _hit_contains_dictionary_target_text(hit, target_terms or [target])
    ]
    if response_language == "vi":
        if direct_hits:
            direct_summary = _dictionary_hit_lead_summary(direct_hits[0])
            if direct_summary:
                return (
                    f"“{target}” khớp trực tiếp với mục từ {direct_titles}. "
                    f"Nội dung chính của mục này: {direct_summary} {direct_citations}"
                ).strip()
            return (
                f"“{target}” khớp trực tiếp với mục từ {direct_titles}. "
                f"Có thể đọc câu hỏi theo mục từ này; các nguồn liên quan khác chỉ nên dùng làm ngữ cảnh, không dùng để đổi nghĩa của mục khớp trực tiếp. {direct_citations}"
            ).strip()
        if not occurrence_hits:
            return (
                f"Trong các mục từ điển được truy hồi, chưa thấy định nghĩa hoặc phần xuất hiện trực tiếp đủ chắc cho “{target}”."
            )
        citations = _format_source_citations([hit.doc_id for hit in occurrence_hits])
        titles = _format_source_title_citations(occurrence_hits)
        return (
            f"Trong các mục từ điển được truy hồi, chưa thấy định nghĩa hoặc phần mở rộng chính thức cho “{target}”. "
            f"Tuy nhiên “{target}” xuất hiện trong phần giải thích/nội dung của: {titles}. "
            f"Vì vậy chỉ có thể xác nhận sự xuất hiện/ngữ cảnh được trích dẫn, không suy rộng nghĩa viết tắt nếu nguồn không nêu rõ. {citations}"
        ).strip()
    if direct_hits:
        direct_summary = _dictionary_hit_lead_summary(direct_hits[0])
        if direct_summary:
            return (
                f"“{target}” directly matches the dictionary entry {direct_titles}. "
                f"The entry says: {direct_summary} {direct_citations}"
            ).strip()
        return (
            f"“{target}” directly matches the dictionary entry {direct_titles}. "
            f"Read the question through that entry; other related sources should only provide context, not change the direct-match meaning. {direct_citations}"
        ).strip()
    if not occurrence_hits:
        return (
            f"The retrieved dictionary entries do not show a formal definition or directly citable occurrence for “{target}”."
        )
    citations = _format_source_citations([hit.doc_id for hit in occurrence_hits])
    titles = _format_source_title_citations(occurrence_hits)
    return (
        f"The retrieved dictionary entries do not show a formal definition or expansion for “{target}”. "
        f"They do show that it appears in the explanation/body of: {titles}. "
        f"I can confirm only that cited occurrence/context, not infer an unstated abbreviation meaning. {citations}"
    ).strip()


def _format_dictionary_disambiguation_fallback_answer(
    question: str,
    hits: Sequence[RetrievalHit],
    metadata: dict[str, Any] | None,
    *,
    language: str | None = None,
) -> str:
    if not hits:
        return ""
    response_language = _normalize_response_language(language)
    query_plan = metadata.get("query_plan") if isinstance(metadata, dict) else None
    target_terms = (
        [str(term).strip() for term in query_plan.get("target_terms") or [] if str(term).strip()]
        if isinstance(query_plan, dict)
        else []
    )
    target = target_terms[0] if target_terms else _strip_command_prefix(question).strip()
    cited_hits = list(hits)
    citations = _format_source_citations([hit.doc_id for hit in cited_hits])
    entries = _format_source_title_citations(cited_hits)
    if response_language == "vi":
        return (
            f"“{target}” là một truy vấn ngắn có thể khớp nhiều mục từ trong các nguồn đã truy hồi. "
            f"Các mục phù hợp gồm: {entries}. "
            f"Nếu cần một nghĩa duy nhất, cần thêm ngữ cảnh; tôi không coi các mục này là các câu hỏi riêng biệt. {citations}"
        ).strip()
    return (
        f"“{target}” is a short query that can match multiple retrieved dictionary entries. "
        f"Supported entries include: {entries}. "
        f"A single sense requires more context; I do not treat these entries as separate user questions. {citations}"
    ).strip()


def _hit_has_direct_dictionary_match(hit: RetrievalHit) -> bool:
    metadata = hit.metadata or {}
    mode = str(metadata.get("dictionary_match_mode") or "")
    direct_score = float(metadata.get("dictionary_direct_score") or 0.0)
    return mode in {"strict", "folded"} or direct_score >= 1.0


def _hit_contains_dictionary_target_text(hit: RetrievalHit, target_terms: Sequence[str]) -> bool:
    target_keys = [_fold_prompt_text(term) for term in target_terms if _fold_prompt_text(term)]
    if not target_keys:
        return False
    metadata = hit.metadata or {}
    highlight_text = " ".join(str(item) for item in metadata.get("query_highlights") or [])
    source_text = " ".join(
        str(value or "")
        for value in (
            metadata.get("headword"),
            hit.title,
            metadata.get("raw_docx_text"),
            hit.text,
            highlight_text,
        )
    )
    folded_source = _fold_prompt_text(source_text)
    return any(target and target in folded_source for target in target_keys)


def _format_source_title_citations(hits: Sequence[RetrievalHit]) -> str:
    parts: list[str] = []
    seen: set[str] = set()
    for hit in hits:
        doc_id = str(hit.doc_id or "").strip()
        if not doc_id or doc_id in seen:
            continue
        seen.add(doc_id)
        title = str(hit.title or doc_id).strip()
        parts.append(f"{title} [{doc_id}]")
    return "; ".join(parts)


def _format_source_titles(hits: Sequence[RetrievalHit]) -> str:
    parts: list[str] = []
    seen: set[str] = set()
    for hit in hits:
        doc_id = str(hit.doc_id or "").strip()
        if not doc_id or doc_id in seen:
            continue
        seen.add(doc_id)
        parts.append(str(hit.title or doc_id).strip())
    return "; ".join(parts)


def _format_source_citations(doc_ids: Sequence[str]) -> str:
    seen: set[str] = set()
    citations: list[str] = []
    for doc_id in doc_ids:
        doc_id = str(doc_id or "").strip()
        if not doc_id or doc_id in seen:
            continue
        seen.add(doc_id)
        citations.append(f"[{doc_id}]")
    return ", ".join(citations)


def _dictionary_hit_lead_summary(hit: RetrievalHit, *, max_chars: int = 320) -> str:
    raw = str(hit.metadata.get("raw_docx_text") or hit.text or "").strip()
    if not raw:
        return ""
    text = re.sub(r"\s+", " ", raw)
    title = str(hit.title or hit.metadata.get("headword") or "").strip()
    if title:
        title_pattern = re.escape(re.sub(r"\s+", " ", title))
        text = re.sub(rf"^{title_pattern}\s*[,;:.-]?\s*", "", text, count=1, flags=re.IGNORECASE)
    if not text:
        return ""
    sentence_match = re.search(r"^(.{40,}?[.!?。])(?:\s|$)", text)
    summary = sentence_match.group(1) if sentence_match else text
    if len(summary) > max_chars:
        summary = summary[:max_chars].rstrip(" ,;:") + "..."
    return summary.strip()


def _format_dictionary_answer(hits: list[RetrievalHit], explanation: str) -> str:
    first = hits[0]
    raw = str(first.metadata.get("raw_docx_text") or first.text or "").strip()
    header = f"Mục từ gốc [{first.doc_id}]:"
    parts = [header]
    if raw:
        parts.append(raw)
    cleaned = (explanation or "").strip()
    if cleaned:
        parts.append("Giải thích:\n" + cleaned)
    return "\n\n".join(parts)


def _format_no_dictionary_answer(query: str) -> str:
    return f"Không tìm thấy mục từ phù hợp trong từ điển cho: {query}"


def _localized_no_dictionary_answer(query: str, language: str | None) -> str:
    if _normalize_response_language(language) == "vi":
        return f"Không tìm thấy mục từ phù hợp trong từ điển cho: {query}"
    return f"No matching dictionary entry was found for: {query}"


def _flatten_hit_metadata(metadata: dict[str, Any]) -> dict[str, Any]:
    allowed_keys = {
        "kind",
        "image_data_url",
        "image_url",
        "label",
        "dataset",
        "width",
        "height",
        "schema_version",
        "headword",
        "raw_docx_text",
        "rich_blocks",
        "source",
        "query_highlights",
        "dictionary_redirect_aliases",
        "dictionary_redirect_doc_ids",
        "dictionary_preserved_redirect",
        "dictionary_redirect_target",
        "dictionary_redirect_target_doc_id",
        "dictionary_redirect_target_title",
    }
    return {key: value for key, value in metadata.items() if key in allowed_keys}


def _hit_source_payload(hit: RetrievalHit, *, include_private_text: bool = False) -> dict[str, Any]:
    tier = data_tier_for_hit(hit)
    flattened = _flatten_hit_metadata(hit.metadata)
    if tier == DataTier.PRIVATE:
        flattened = {
            key: value
            for key, value in flattened.items()
            if key
            not in {
                "raw_docx_text",
                "rich_blocks",
                "source",
                "dictionary_evidence_text",
            }
        }
    payload = safe_source_payload(hit, include_private_text=include_private_text, extra=flattened)
    payload["data_tier"] = tier.value
    return payload


def _format_history(messages: list[dict[str, Any]], *, history_messages: int) -> str:
    if history_messages <= 0:
        return "No prior conversation."
    selected = messages[-history_messages:]
    lines: list[str] = []
    for message in selected:
        role = str(message.get("role", "user"))
        if role not in {"system", "user", "assistant"}:
            continue
        content = _content_to_text(message.get("content", "")).strip()
        if content:
            lines.append(f"{role}: {content}")
    return "\n".join(lines) if lines else "No prior conversation."


def _messages_have_history(messages: list[dict[str, Any]]) -> bool:
    if len(messages) > 1:
        return True
    return any(str(message.get("role", "")).strip().lower() in {"assistant", "system"} for message in messages)


def _content_to_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, dict) and item.get("type") == "text":
                parts.append(str(item.get("text", "")))
        return "\n".join(part for part in parts if part)
    return str(content) if content is not None else ""
