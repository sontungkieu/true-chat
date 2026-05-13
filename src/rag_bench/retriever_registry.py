from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Literal

from rag_bench.retrievers import (
    BM25Retriever,
    HybridRrfRetriever,
    ImageDigitsRetriever,
    KeywordMatchRetriever,
    LlmMultiQueryRetriever,
    LlmQueryRewriteRetriever,
    MultiQueryRetriever,
    QueryExpansionClient,
    Retriever,
    TfidfRetriever,
    VectorRerankRetriever,
    VectorRetriever,
)


RetrieverCategory = Literal["text", "image", "keyword", "dictionary"]
RetrieverFactory = Callable[["RetrieverFactoryContext"], Retriever]
RETRIEVER_CATEGORIES: tuple[RetrieverCategory, ...] = ("text", "image", "keyword", "dictionary")


@dataclass(frozen=True)
class RetrieverFactoryContext:
    vector_model: str
    query_expander: QueryExpansionClient | None = None
    query_model: str | None = None


@dataclass(frozen=True)
class RetrieverSpec:
    id: str
    label: str
    description: str
    requires_extra: str | None
    factory: RetrieverFactory
    category: RetrieverCategory = "text"
    uses_llm: bool = False


_RETRIEVER_SPECS: tuple[RetrieverSpec, ...] = (
    RetrieverSpec(
        id="bm25",
        label="BM25",
        description="Lexical BM25 baseline over document titles and text.",
        requires_extra=None,
        factory=lambda _context: BM25Retriever(),
        category="text",
    ),
    RetrieverSpec(
        id="tfidf",
        label="TF-IDF",
        description="Sparse TF-IDF cosine baseline over document titles and text.",
        requires_extra=None,
        factory=lambda _context: TfidfRetriever(),
        category="text",
    ),
    RetrieverSpec(
        id="keyword-match",
        label="Keyword Match",
        description="Deterministic exact keyword and phrase matching over document titles and text.",
        requires_extra=None,
        factory=lambda _context: KeywordMatchRetriever(),
        category="keyword",
    ),
    RetrieverSpec(
        id="multi-query",
        label="Multi-query",
        description="Deterministic BM25 multi-query expansion with reciprocal-rank fusion.",
        requires_extra=None,
        factory=lambda _context: MultiQueryRetriever(),
        category="text",
    ),
    RetrieverSpec(
        id="llm-query-rewrite",
        label="LLM Query Rewrite",
        description="Groq rewrites the user question into one search query, then BM25 retrieves with RRF.",
        requires_extra=None,
        factory=lambda context: LlmQueryRewriteRetriever(
            query_expander=_require_query_expander("llm-query-rewrite", context),
            query_model=context.query_model,
        ),
        category="text",
        uses_llm=True,
    ),
    RetrieverSpec(
        id="llm-multi-query",
        label="LLM Multi-query",
        description="Groq generates multiple search queries, then BM25 retrieves and merges by RRF.",
        requires_extra=None,
        factory=lambda context: LlmMultiQueryRetriever(
            query_expander=_require_query_expander("llm-multi-query", context),
            query_model=context.query_model,
        ),
        category="text",
        uses_llm=True,
    ),
    RetrieverSpec(
        id="image-digits",
        label="Image Digits",
        description="Lightweight image search over the bundled scikit-learn handwritten digits sample dataset.",
        requires_extra=None,
        factory=lambda _context: ImageDigitsRetriever(),
        category="image",
    ),
    RetrieverSpec(
        id="vector",
        label="Vector",
        description="Dense sentence-transformer embeddings with FAISS cosine/IP search.",
        requires_extra="vector",
        factory=lambda context: VectorRetriever(model_name=context.vector_model),
        category="text",
    ),
    RetrieverSpec(
        id="hybrid-rrf",
        label="Hybrid RRF",
        description="BM25 plus vector retrieval merged by reciprocal-rank fusion.",
        requires_extra="vector",
        factory=lambda context: HybridRrfRetriever(vector_model=context.vector_model),
        category="text",
    ),
    RetrieverSpec(
        id="vector-rerank",
        label="Vector Rerank",
        description="Vector candidate retrieval reranked by normalized BM25 lexical scores.",
        requires_extra="vector",
        factory=lambda context: VectorRerankRetriever(vector_model=context.vector_model),
        category="text",
    ),
)

_RETRIEVERS_BY_ID = {spec.id: spec for spec in _RETRIEVER_SPECS}
_ALIASES = {
    "lexical": "bm25",
    "keyword": "keyword-match",
    "find": "keyword-match",
    "dense": "vector",
    "hybrid": "hybrid-rrf",
    "rrf": "hybrid-rrf",
    "rerank": "vector-rerank",
    "multi-query-bm25": "multi-query",
    "query-rewrite": "llm-query-rewrite",
    "llm-rewrite": "llm-query-rewrite",
    "img": "image-digits",
    "image": "image-digits",
    "digits": "image-digits",
}


def list_retrievers() -> tuple[RetrieverSpec, ...]:
    return _RETRIEVER_SPECS


def list_retriever_ids() -> tuple[str, ...]:
    return tuple(spec.id for spec in _RETRIEVER_SPECS)


def normalize_retriever_id(value: str) -> str:
    normalized = value.strip().lower()
    return _ALIASES.get(normalized, normalized)


def get_retriever_spec(name: str) -> RetrieverSpec:
    retriever_id = normalize_retriever_id(name)
    try:
        return _RETRIEVERS_BY_ID[retriever_id]
    except KeyError as exc:
        choices = ", ".join(list_retriever_ids())
        raise ValueError(f"Unknown retriever: {name}. Choices: {choices}") from exc


def retriever_uses_llm(name: str) -> bool:
    return get_retriever_spec(name).uses_llm


def create_retriever(
    name: str,
    *,
    vector_model: str,
    query_expander: QueryExpansionClient | None = None,
    query_model: str | None = None,
) -> Retriever:
    spec = get_retriever_spec(name)
    retriever = spec.factory(
        RetrieverFactoryContext(
            vector_model=vector_model,
            query_expander=query_expander,
            query_model=query_model,
        )
    )
    if retriever.name != spec.id:
        raise RuntimeError(f"Retriever factory for '{spec.id}' returned '{retriever.name}'")
    return retriever


def _require_query_expander(name: str, context: RetrieverFactoryContext) -> QueryExpansionClient:
    if context.query_expander is None:
        raise ValueError(f"Retriever '{name}' requires a Groq query expansion client")
    return context.query_expander
