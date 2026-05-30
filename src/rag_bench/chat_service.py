from __future__ import annotations

import json
import math
import re
import time
import uuid
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Callable, Protocol

from rag_bench.benchmarks import load_benchmark
from rag_bench.dictionary import (
    DEFAULT_DICTIONARY_ARTIFACT,
    DEFAULT_DICTIONARY_LETTERS,
    DEFAULT_DICTIONARY_SOURCE_DIR,
    DictionaryLoadResult,
    load_dictionary_documents,
)
from rag_bench.groq_client import GenerationResult, OpenAICompatibleClient, RoundRobinGroqClient
from rag_bench.prompts import SYSTEM_PROMPT
from rag_bench.retriever_registry import create_retriever, get_retriever_spec, normalize_retriever_id
from rag_bench.retrievers import Retriever
from rag_bench.secrets import ApiKey, load_env_api_key, load_groq_keys
from rag_bench.types import BenchmarkData, Query, RetrievalHit, RetrievalResult


DEFAULT_PROXY_MODEL_ID = "rag-scifact-bm25"
DEFAULT_CHAT_MODEL = "qwen/qwen3-32b"
DEFAULT_CHAT_MODELS = (DEFAULT_CHAT_MODEL, "llama-3.1-8b-instant")
DEFAULT_MIMO_BASE_URL = "https://token-plan-sgp.xiaomimimo.com/v1"
DEFAULT_MIMO_MODELS = ("mimo-v2.5-pro", "mimo-v2.5")
DEFAULT_CHAT_RETRIEVERS = (
    "bm25",
    "tfidf",
    "keyword-match",
    "multi-query",
    "graph-bm25",
    "dictionary-graph",
    "image-digits",
)
MIN_RETRIEVAL_DISPLAY_SCORE = 5e-4
CONTEXT_SEPARATOR = "\n\n---\n\n"


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


@dataclass
class ChatServiceResult:
    response: dict[str, Any]
    generation: GenerationResult
    hits: list[RetrievalHit]
    retrieval_latency_s: float
    retrieval_metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class RagChatService:
    config: ChatProxyConfig
    benchmark: BenchmarkData
    retriever: Retriever
    llm: ChatGenerationClient
    started_at_s: float = field(default_factory=time.time)
    retrievers: dict[str, Retriever] = field(default_factory=dict)
    dictionary_status: dict[str, Any] = field(default_factory=dict)

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
        return cls(
            config=config,
            benchmark=benchmark,
            retriever=retriever,
            llm=llm,
            retrievers=retrievers,
            dictionary_status=dictionary.status,
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
    ) -> ChatServiceResult:
        response_model, generation_model = self.resolve_request_model(request_model)
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
            image_query, rewrite_metadata = self._image_query(question, generation_model, image_rewrite=image_rewrite)
            retriever = self.resolve_request_retriever("image-digits")
            request_image_top_k = _clamp_top_k(image_top_k if image_top_k is not None else top_k, fallback=self.config.image_top_k)
            retrieval = retriever.search(Query(query_id="chat-img", text=image_query), request_image_top_k)
            retrieval, score_filter_metadata = _apply_retrieval_score_controls(
                retrieval,
                score_controls,
                max_hits=request_image_top_k,
            )
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
            retrieval = retriever.search(Query(query_id="chat-dict", text=question), request_top_k)
            retrieval, score_filter_metadata = _apply_retrieval_score_controls(
                retrieval,
                score_controls,
                max_hits=request_top_k,
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
            if retrieval.hits:
                prompt_messages = build_dictionary_rag_messages(
                    messages,
                    retrieval.hits,
                    query=question,
                    max_context_chars=self.config.max_context_chars,
                    history_messages=history_messages,
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
                answer = _format_dictionary_answer(retrieval.hits, generation.answer)
            else:
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
            )
            return ChatServiceResult(
                response=response,
                generation=generation,
                hits=retrieval.hits,
                retrieval_latency_s=retrieval.latency_s,
                retrieval_metadata=retrieval_metadata,
            )

        retriever = self.resolve_text_request_retriever(request_retriever)
        request_top_k = _clamp_top_k(top_k, fallback=self.config.top_k)
        if retriever.name == "keyword-match":
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
        dictionary_fallback = self._text_dictionary_fallback(
            question,
            top_k=request_top_k,
            primary_retriever=retriever,
            primary_retrieval=retrieval,
        )
        dictionary_score_filter_metadata: dict[str, Any] = {}
        if dictionary_fallback is not None:
            dictionary_fallback, dictionary_score_filter_metadata = _apply_retrieval_score_controls(
                dictionary_fallback,
                score_controls,
                max_hits=request_top_k,
            )
            if not dictionary_fallback.hits:
                dictionary_fallback = None
        prompt_hits = _merge_text_and_dictionary_hits(
            retrieval.hits,
            dictionary_fallback.hits if dictionary_fallback else [],
            max_hits=request_top_k,
        )
        prompt_messages = build_chat_rag_messages(
            messages,
            prompt_hits,
            max_context_chars=self.config.max_context_chars,
            history_messages=history_messages,
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

        combined_hits = list(prompt_hits)
        retrieval_metadata = {**retrieval.metadata, **score_filter_metadata}
        if dictionary_fallback is not None:
            retrieval_metadata.update(
                {
                    "dictionary_fallback": True,
                    "dictionary_fallback_latency_s": dictionary_fallback.latency_s,
                    "dictionary_fallback_count": len(dictionary_fallback.hits),
                    "dictionary_fallback_metadata": dictionary_fallback.metadata,
                }
            )
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
    ) -> dict[str, Any]:
        created = int(time.time())
        completion_id = f"chatcmpl-{uuid.uuid4().hex}"
        response_model = response_model or self.config.model_id
        generation_model = generation_model or self.config.model
        retriever = retriever or self.retriever
        return {
            "id": completion_id,
            "object": "chat.completion",
            "created": created,
            "model": response_model,
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
                    _hit_source_payload(hit)
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
        retrieval = dictionary_retriever.search(Query(query_id="chat-dict-fallback", text=question), request_top_k)
        hits = [hit for hit in retrieval.hits if _strong_dictionary_text_fallback_hit(hit)]
        if not hits:
            return None
        primary_top_score = max((hit.score for hit in primary_retrieval.hits), default=0.0)
        has_direct_dictionary_hit = any(float(hit.metadata.get("dictionary_direct_score") or 0.0) > 0 for hit in hits)
        if primary_top_score > 0 and not has_direct_dictionary_hit:
            return None
        return RetrievalResult(query=retrieval.query, hits=hits, latency_s=retrieval.latency_s, metadata=retrieval.metadata)

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
        retrieval = retriever.search(Query(query_id="dictionary-lookup", text=query), request_top_k)
        retrieval, score_filter_metadata = _apply_retrieval_score_controls(
            retrieval,
            score_controls,
            max_hits=request_top_k,
        )
        hits = [hit for hit in retrieval.hits if hit.score > 0 or score_controls.has_score_range]
        return {
            "object": "dictionary.lookup",
            "query": query,
            "retriever": retriever.name,
            "top_k": request_top_k,
            "retrieval_latency_s": retrieval.latency_s,
            "retrieval_metadata": {**retrieval.metadata, **score_filter_metadata},
            "dictionary": self.dictionary_status,
            "retrieved": [_hit_source_payload(hit) for hit in hits],
        }

    def available_model_ids(self) -> tuple[str, ...]:
        mimo_models = self.config.mimo_models if self.config.mimo_enabled else ()
        return _dedupe_preserve_order((self.config.model_id, self.config.model, *self.config.available_models, *mimo_models))

    def available_generation_models(self) -> tuple[str, ...]:
        mimo_models = self.config.mimo_models if self.config.mimo_enabled else ()
        return _dedupe_preserve_order((self.config.model, *self.config.available_models, *mimo_models))

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
        )
        for rank, doc_id in enumerate(ranked[:top_k], 1)
    ]


def _merge_text_and_dictionary_hits(
    primary_hits: list[RetrievalHit],
    dictionary_hits: list[RetrievalHit],
    *,
    max_hits: int | None = None,
) -> list[RetrievalHit]:
    if not dictionary_hits:
        hits = list(primary_hits)
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
    return [
        RetrievalHit(
            doc_id=hit.doc_id,
            score=hit.score,
            rank=rank,
            title=hit.title,
            text=hit.text,
            metadata=hit.metadata,
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


def _strong_dictionary_text_fallback_hit(hit: RetrievalHit) -> bool:
    if hit.score <= 0:
        return False
    metadata = hit.metadata or {}
    mode = str(metadata.get("dictionary_match_mode") or "")
    direct_score = float(metadata.get("dictionary_direct_score") or 0.0)
    graph_score = float(metadata.get("dictionary_graph_score") or 0.0)
    return mode in {"strict", "folded"} or direct_score > 0 or (mode == "graph" and graph_score >= 0.35)


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
    if controls.sort_by_score:
        filtered = sorted(filtered, key=lambda hit: (-hit.score, hit.rank, hit.doc_id))
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
    return [
        hit
        for hit in hits
        if include_score_filtered or hit.score > MIN_RETRIEVAL_DISPLAY_SCORE or hit.doc_id in cited_doc_ids or _hit_is_image(hit)
    ]


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

    mimo_key = load_env_api_key(config.mimo_env_file, config.mimo_api_key_var, alias="mimo")
    mimo_client = RoundRobinGroqClient(
        keys=[mimo_key],
        model=config.mimo_models[0] if config.mimo_models else "mimo-v2.5-pro",
        max_retries=config.max_retries,
        key_tokens_per_minute=config.mimo_key_tokens_per_minute,
        key_requests_per_minute=config.mimo_key_requests_per_minute,
        rate_limit_scope="per-key",
        client_factory=lambda key, timeout: OpenAICompatibleClient(
            api_key=key.value,
            base_url=config.mimo_base_url,
            timeout_s=timeout,
            token_parameter="max_tokens",
        ),
        provider_name="MiMo",
        completion_token_parameter="max_tokens",
    )
    return ModelRoutedChatClient(
        default_client=groq_client,
        routes={model: mimo_client for model in config.mimo_models},
    )


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
        retriever = create_retriever(
            name,
            vector_model=config.vector_model,
            query_expander=llm,
            query_model=config.model,
        )
        documents = dictionary.documents if spec.category == "dictionary" and dictionary is not None else benchmark.documents
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
    return min(50, max(1, int(value)))


def build_chat_rag_messages(
    messages: list[dict[str, Any]],
    hits: list[RetrievalHit],
    *,
    max_context_chars: int,
    history_messages: int,
    language: str | None = None,
) -> list[dict[str, str]]:
    question = last_user_text(messages)
    context = _format_context(hits, max_context_chars=max_context_chars)
    history = _format_history(messages[:-1], history_messages=history_messages)
    language_instruction = _language_instruction(language)
    user_prompt = (
        f"Recent conversation:\n{history}\n\n"
        f"Question:\n{question}\n\n"
        f"Retrieved contexts:\n{context}\n\n"
        "Answer:"
    )
    return [
        {"role": "system", "content": _join_prompt_parts(SYSTEM_PROMPT, language_instruction)},
        {"role": "user", "content": user_prompt},
    ]


def build_dictionary_rag_messages(
    messages: list[dict[str, Any]],
    hits: list[RetrievalHit],
    *,
    query: str,
    max_context_chars: int,
    history_messages: int,
    language: str | None = None,
) -> list[dict[str, str]]:
    context = _format_context(hits, max_context_chars=max_context_chars)
    history = _format_history(messages[:-1], history_messages=history_messages)
    response_language = _normalize_response_language(language)
    language_instruction = _language_instruction(response_language)
    user_prompt = (
        f"Recent conversation:\n{history}\n\n"
        f"Dictionary question:\n{query}\n\n"
        f"Retrieved dictionary entries:\n{context}\n\n"
        "Explain the term in the required response language. Cite dictionary entries with their ids in square brackets. "
        "Do not invent content not supported by the retrieved dictionary entries."
    )
    return [
        {
            "role": "system",
            "content": _join_prompt_parts(
                "You are a careful military dictionary assistant. Use the retrieved local dictionary entries first. "
                "Keep abbreviations, casing, and Vietnamese diacritics intact. Cite sources as [entry-id].",
                language_instruction,
            ),
        },
        {"role": "user", "content": user_prompt},
    ]


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
        block = f"[{hit.doc_id}]\n{title}{hit.text}".strip()
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
    }
    return {key: value for key, value in metadata.items() if key in allowed_keys}


def _hit_source_payload(hit: RetrievalHit) -> dict[str, Any]:
    return {
        "doc_id": hit.doc_id,
        "rank": hit.rank,
        "score": hit.score,
        "title": hit.title,
        "text": hit.text,
        "metadata": hit.metadata,
        **_flatten_hit_metadata(hit.metadata),
    }


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
