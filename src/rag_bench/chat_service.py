from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Protocol

from rag_bench.benchmarks import load_benchmark
from rag_bench.groq_client import GenerationResult, RoundRobinGroqClient
from rag_bench.prompts import SYSTEM_PROMPT
from rag_bench.retriever_registry import create_retriever, normalize_retriever_id
from rag_bench.retrievers import Retriever
from rag_bench.secrets import ApiKey, load_groq_keys
from rag_bench.types import BenchmarkData, Query, RetrievalHit


DEFAULT_PROXY_MODEL_ID = "rag-scifact-bm25"
DEFAULT_CHAT_MODELS = ("llama-3.1-8b-instant", "qwen/qwen3-32b")
DEFAULT_CHAT_RETRIEVERS = ("bm25", "tfidf", "keyword-match", "multi-query")


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
    model: str = "llama-3.1-8b-instant"
    model_id: str = DEFAULT_PROXY_MODEL_ID
    available_models: tuple[str, ...] = DEFAULT_CHAT_MODELS
    available_retrievers: tuple[str, ...] = DEFAULT_CHAT_RETRIEVERS
    vector_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    max_retries: int = 2
    max_completion_tokens: int = 128
    temperature: float = 0.0
    max_context_chars: int = 2500
    allow_large_bench: bool = False
    key_tokens_per_minute: int = 6000
    key_requests_per_minute: int = 30
    rate_limit_scope: str = "per-key"
    history_messages: int = 6


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
        retrievers = _build_retrievers(config, benchmark, llm=llm)
        retriever = _default_retriever(config, retrievers)
        return cls(config=config, benchmark=benchmark, retriever=retriever, llm=llm, retrievers=retrievers)

    def answer(
        self,
        messages: list[dict[str, Any]],
        *,
        request_model: str | None = None,
        request_retriever: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> ChatServiceResult:
        response_model, generation_model = self.resolve_request_model(request_model)
        retriever = self.resolve_request_retriever(request_retriever)

        question = last_user_text(messages)
        retrieval = retriever.search(Query(query_id="chat", text=question), self.config.top_k)
        prompt_messages = build_chat_rag_messages(
            messages,
            retrieval.hits,
            max_context_chars=self.config.max_context_chars,
            history_messages=self.config.history_messages,
        )
        generation = self.llm.generate(
            prompt_messages,
            model=generation_model,
            temperature=self.config.temperature if temperature is None else temperature,
            max_completion_tokens=self.config.max_completion_tokens if max_tokens is None else max_tokens,
        )
        if generation.error:
            raise RuntimeError(generation.error)

        response = self._build_response(
            answer=generation.answer,
            generation=generation,
            hits=retrieval.hits,
            retrieval_latency_s=retrieval.latency_s,
            retrieval_metadata=retrieval.metadata,
            retriever=retriever,
            response_model=response_model,
            generation_model=generation_model,
        )
        return ChatServiceResult(
            response=response,
            generation=generation,
            hits=retrieval.hits,
            retrieval_latency_s=retrieval.latency_s,
            retrieval_metadata=retrieval.metadata,
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
        response_model: str | None = None,
        generation_model: str | None = None,
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
                "top_k": self.config.top_k,
                "retrieval_latency_s": retrieval_latency_s,
                "retrieval_metadata": retrieval_metadata or {},
                "retrieved": [
                    {
                        "doc_id": hit.doc_id,
                        "rank": hit.rank,
                        "score": hit.score,
                        "title": hit.title,
                        "text": hit.text,
                    }
                    for hit in hits
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

    def available_model_ids(self) -> tuple[str, ...]:
        return _dedupe_preserve_order((self.config.model_id, self.config.model, *self.config.available_models))

    def available_generation_models(self) -> tuple[str, ...]:
        return _dedupe_preserve_order((self.config.model, *self.config.available_models))

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


def _build_llm(config: ChatProxyConfig, keys: list[ApiKey]) -> RoundRobinGroqClient:
    return RoundRobinGroqClient(
        keys=keys,
        model=config.model,
        max_retries=config.max_retries,
        key_tokens_per_minute=config.key_tokens_per_minute,
        key_requests_per_minute=config.key_requests_per_minute,
        rate_limit_scope=config.rate_limit_scope,
    )


def _build_retrievers(
    config: ChatProxyConfig,
    benchmark: BenchmarkData,
    *,
    llm: ChatGenerationClient | None = None,
) -> dict[str, Retriever]:
    retrievers: dict[str, Retriever] = {}
    for name in _dedupe_normalized_retriever_ids((config.retriever, *config.available_retrievers)):
        retriever = create_retriever(
            name,
            vector_model=config.vector_model,
            query_expander=llm,
            query_model=config.model,
        )
        retriever.build(benchmark.documents)
        retrievers[retriever.name] = retriever
    return retrievers


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


def build_chat_rag_messages(
    messages: list[dict[str, Any]],
    hits: list[RetrievalHit],
    *,
    max_context_chars: int,
    history_messages: int,
) -> list[dict[str, str]]:
    question = last_user_text(messages)
    context = _format_context(hits, max_context_chars=max_context_chars)
    history = _format_history(messages[:-1], history_messages=history_messages)
    user_prompt = (
        f"Recent conversation:\n{history}\n\n"
        f"Question:\n{question}\n\n"
        f"Retrieved contexts:\n{context}\n\n"
        "Answer:"
    )
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
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


def _format_context(hits: list[RetrievalHit], *, max_context_chars: int) -> str:
    context_blocks: list[str] = []
    used_chars = 0
    for hit in hits:
        title = f"{hit.title}\n" if hit.title else ""
        block = f"[{hit.doc_id}]\n{title}{hit.text}".strip()
        if not block:
            continue
        remaining = max_context_chars - used_chars
        if remaining <= 0:
            break
        if len(block) > remaining:
            block = block[:remaining].rstrip()
        context_blocks.append(block)
        used_chars += len(block)
    return "\n\n---\n\n".join(context_blocks) if context_blocks else "No retrieved context."


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
