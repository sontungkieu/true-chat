from __future__ import annotations

import json
import re
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
from rag_bench.types import BenchmarkData, Query, RetrievalHit, RetrievalResult


DEFAULT_PROXY_MODEL_ID = "rag-scifact-bm25"
DEFAULT_CHAT_MODELS = ("llama-3.1-8b-instant", "qwen/qwen3-32b")
DEFAULT_CHAT_RETRIEVERS = ("bm25", "tfidf", "keyword-match", "multi-query", "image-digits")


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
    image_top_k: int = 5


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
        top_k: int | None = None,
        image_top_k: int | None = None,
        response_mode: str | None = None,
        image_rewrite: bool | None = None,
    ) -> ChatServiceResult:
        response_model, generation_model = self.resolve_request_model(request_model)
        question = last_user_text(messages)
        command = parse_chat_command(question)
        mode = _normalize_response_mode(response_mode)
        if command and command[0] == "img":
            mode = "image"
            question = command[1] or "digit image"

        if mode == "image":
            image_query, rewrite_metadata = self._image_query(question, generation_model, image_rewrite=image_rewrite)
            retriever = self.resolve_request_retriever("image-digits")
            request_image_top_k = _clamp_top_k(image_top_k if image_top_k is not None else top_k, fallback=self.config.image_top_k)
            retrieval = retriever.search(Query(query_id="chat-img", text=image_query), request_image_top_k)
            generation = GenerationResult(
                answer=_format_image_answer(image_query, retrieval.hits),
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

        combined_hits = list(retrieval.hits)
        retrieval_metadata = dict(retrieval.metadata)
        if mode == "text_image":
            image_query, image_query_metadata = self._image_query(
                f"Question: {question}\nAnswer: {generation.answer}",
                generation_model,
                image_rewrite=True if image_rewrite is None else image_rewrite,
            )
            image_retriever = self.resolve_request_retriever("image-digits")
            request_image_top_k = _clamp_top_k(image_top_k, fallback=self.config.image_top_k)
            image_retrieval = image_retriever.search(Query(query_id="chat-img", text=image_query), request_image_top_k)
            combined_hits.extend(image_retrieval.hits)
            retrieval_metadata.update(
                {
                    "response_mode": "text_image",
                    "image_retriever": image_retriever.name,
                    "image_retrieval_latency_s": image_retrieval.latency_s,
                    "image_top_k": request_image_top_k,
                    "image_query": image_query,
                    "image_retrieval_metadata": image_retrieval.metadata,
                    **image_query_metadata,
                }
            )
        else:
            retrieval_metadata.setdefault("response_mode", "text")

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
                    {
                        "doc_id": hit.doc_id,
                        "rank": hit.rank,
                        "score": hit.score,
                        "title": hit.title,
                        "text": hit.text,
                        "metadata": hit.metadata,
                        **_flatten_hit_metadata(hit.metadata),
                    }
                    for hit in _filter_retrieved_for_display(hits, answer)
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


def _filter_retrieved_for_display(hits: list[RetrievalHit], answer: str) -> list[RetrievalHit]:
    cited_doc_ids = _cited_doc_ids(answer)
    return [
        hit
        for hit in hits
        if hit.score > 0 or hit.doc_id in cited_doc_ids or _hit_is_image(hit)
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


def parse_chat_command(text: str) -> tuple[str, str] | None:
    stripped = text.strip()
    if not stripped.startswith("/"):
        return None
    command, _, remainder = stripped.partition(" ")
    normalized = command.lower()
    if normalized in {"/img", "/image"}:
        return "img", remainder.strip()
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
    raise ValueError("response_mode must be one of: text, image, text_image")


def _strip_command_prefix(text: str) -> str:
    command = parse_chat_command(text)
    if command and command[0] == "img":
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


def _format_image_answer(query: str, hits: list[RetrievalHit]) -> str:
    if not hits:
        return f"No image results found for '{query}'."
    return f"Found {len(hits)} image result(s) for '{query}'."


def _flatten_hit_metadata(metadata: dict[str, Any]) -> dict[str, Any]:
    allowed_keys = {"kind", "image_data_url", "image_url", "label", "dataset", "width", "height"}
    return {key: value for key, value in metadata.items() if key in allowed_keys}


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
