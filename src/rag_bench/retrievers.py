from __future__ import annotations

import json
import re
import time
from urllib.parse import quote
from dataclasses import dataclass
from typing import Any, Protocol

import numpy as np

from rag_bench.types import Document, RetrievalHit, RetrievalResult, Query


TOKEN_RE = re.compile(r"\w+", re.UNICODE)
STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "by",
    "for",
    "from",
    "has",
    "have",
    "in",
    "is",
    "it",
    "of",
    "on",
    "or",
    "that",
    "the",
    "to",
    "was",
    "were",
    "with",
    "bang",
    "bằng",
    "explain",
    "giai",
    "giải",
    "thich",
    "thích",
    "tieng",
    "tiếng",
    "viet",
    "việt",
}
DIGIT_WORDS = {
    "zero": 0,
    "one": 1,
    "two": 2,
    "three": 3,
    "four": 4,
    "five": 5,
    "six": 6,
    "seven": 7,
    "eight": 8,
    "nine": 9,
}


class Retriever(Protocol):
    name: str
    build_time_s: float

    def build(self, documents: list[Document]) -> None: ...

    def search(self, query: Query, top_k: int) -> RetrievalResult: ...


class QueryExpansionClient(Protocol):
    def generate(
        self,
        messages: list[dict[str, str]],
        *,
        model: str | None = None,
        temperature: float = 0.0,
        max_completion_tokens: int = 512,
    ) -> Any: ...


@dataclass
class BM25Retriever:
    name: str = "bm25"
    build_time_s: float = 0.0

    def build(self, documents: list[Document]) -> None:
        from rank_bm25 import BM25Okapi

        started = time.perf_counter()
        self._documents = list(documents)
        tokenized = [_tokenize(doc.display_text) for doc in self._documents]
        self._index = BM25Okapi(tokenized)
        self.build_time_s = time.perf_counter() - started

    def search(self, query: Query, top_k: int) -> RetrievalResult:
        started = time.perf_counter()
        scores = self._index.get_scores(_tokenize(query.text))
        ranked = _rank_scores(scores, top_k)
        hits = [_hit_from_doc(self._documents[index], float(scores[index]), rank) for rank, index in enumerate(ranked, 1)]
        return RetrievalResult(query=query, hits=hits, latency_s=time.perf_counter() - started)


@dataclass
class TfidfRetriever:
    name: str = "tfidf"
    build_time_s: float = 0.0

    def build(self, documents: list[Document]) -> None:
        from sklearn.feature_extraction.text import TfidfVectorizer

        started = time.perf_counter()
        self._documents = list(documents)
        self._vectorizer = TfidfVectorizer(lowercase=True, stop_words="english")
        self._matrix = self._vectorizer.fit_transform([doc.display_text for doc in self._documents])
        self.build_time_s = time.perf_counter() - started

    def search(self, query: Query, top_k: int) -> RetrievalResult:
        started = time.perf_counter()
        query_vector = self._vectorizer.transform([query.text])
        scores = (self._matrix @ query_vector.T).toarray().ravel()
        ranked = _rank_scores(scores, top_k)
        hits = [_hit_from_doc(self._documents[index], float(scores[index]), rank) for rank, index in enumerate(ranked, 1)]
        return RetrievalResult(query=query, hits=hits, latency_s=time.perf_counter() - started)


@dataclass
class KeywordMatchRetriever:
    name: str = "keyword-match"
    build_time_s: float = 0.0

    def build(self, documents: list[Document]) -> None:
        started = time.perf_counter()
        self._documents = list(documents)
        self._token_counts = [_token_counts(doc.display_text) for doc in self._documents]
        self._lower_texts = [doc.display_text.lower() for doc in self._documents]
        self.build_time_s = time.perf_counter() - started

    def search(self, query: Query, top_k: int) -> RetrievalResult:
        started = time.perf_counter()
        query_tokens = _content_tokens(query.text)
        query_phrase = query.text.strip().lower()
        scores = np.zeros(len(self._documents), dtype=np.float32)
        for index, doc_tokens in enumerate(self._token_counts):
            matched_terms = 0
            matched_frequency = 0
            for token in query_tokens:
                frequency = doc_tokens.get(token, 0)
                if frequency:
                    matched_terms += 1
                    matched_frequency += frequency
            score = float(matched_terms * 2 + min(matched_frequency, 8))
            if query_phrase and query_phrase in self._lower_texts[index]:
                score += 5.0
            scores[index] = score
        ranked = _rank_scores(scores, top_k)
        hits = [_hit_from_doc(self._documents[index], float(scores[index]), rank) for rank, index in enumerate(ranked, 1)]
        return RetrievalResult(query=query, hits=hits, latency_s=time.perf_counter() - started)


@dataclass
class VectorRetriever:
    model_name: str = "sentence-transformers/all-MiniLM-L6-v2"
    encoder: object | None = None
    use_faiss: bool = True
    name: str = "vector"
    build_time_s: float = 0.0

    def build(self, documents: list[Document]) -> None:
        started = time.perf_counter()
        self._documents = list(documents)
        self._encoder = self.encoder or _load_sentence_transformer(self.model_name)
        embeddings = _as_float32(self._encoder.encode([doc.display_text for doc in self._documents]))
        embeddings = _l2_normalize(embeddings)
        self._embeddings = embeddings
        self._faiss_index = None

        if self.use_faiss:
            try:
                import faiss
            except ImportError as exc:
                raise RuntimeError(
                    "Vector retrieval requires faiss-cpu. Install with: uv sync --extra vector"
                ) from exc
            index = faiss.IndexFlatIP(embeddings.shape[1])
            index.add(embeddings)
            self._faiss_index = index

        self.build_time_s = time.perf_counter() - started

    def search(self, query: Query, top_k: int) -> RetrievalResult:
        started = time.perf_counter()
        query_embedding = _as_float32(self._encoder.encode([query.text]))
        query_embedding = _l2_normalize(query_embedding)
        if self._faiss_index is not None:
            scores, indexes = self._faiss_index.search(query_embedding, min(top_k, len(self._documents)))
            pairs = [(int(index), float(score)) for index, score in zip(indexes[0], scores[0], strict=False) if index >= 0]
        else:
            scores = self._embeddings @ query_embedding[0]
            ranked = _rank_scores(scores, top_k)
            pairs = [(int(index), float(scores[index])) for index in ranked]
        hits = [_hit_from_doc(self._documents[index], score, rank) for rank, (index, score) in enumerate(pairs, 1)]
        return RetrievalResult(query=query, hits=hits, latency_s=time.perf_counter() - started)


@dataclass
class HybridRrfRetriever:
    vector_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    vector_encoder: object | None = None
    use_faiss: bool = True
    rrf_k: int = 60
    candidate_multiplier: int = 20
    min_candidates: int = 50
    name: str = "hybrid-rrf"
    build_time_s: float = 0.0

    def build(self, documents: list[Document]) -> None:
        started = time.perf_counter()
        self._bm25 = BM25Retriever()
        self._vector = VectorRetriever(
            model_name=self.vector_model,
            encoder=self.vector_encoder,
            use_faiss=self.use_faiss,
        )
        self._bm25.build(documents)
        self._vector.build(documents)
        self.build_time_s = time.perf_counter() - started

    def search(self, query: Query, top_k: int) -> RetrievalResult:
        started = time.perf_counter()
        candidate_k = _candidate_k(top_k, self.min_candidates, self.candidate_multiplier)
        results = [
            self._bm25.search(query, candidate_k),
            self._vector.search(query, candidate_k),
        ]
        hits = _rrf_merge(results, top_k=top_k, rrf_k=self.rrf_k)
        return RetrievalResult(query=query, hits=hits, latency_s=time.perf_counter() - started)


@dataclass
class VectorRerankRetriever:
    vector_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    vector_encoder: object | None = None
    use_faiss: bool = True
    candidate_multiplier: int = 20
    min_candidates: int = 50
    lexical_weight: float = 0.7
    vector_weight: float = 0.3
    name: str = "vector-rerank"
    build_time_s: float = 0.0

    def build(self, documents: list[Document]) -> None:
        from rank_bm25 import BM25Okapi

        started = time.perf_counter()
        self._documents = list(documents)
        self._doc_index_by_id = {doc.doc_id: index for index, doc in enumerate(self._documents)}
        self._vector = VectorRetriever(
            model_name=self.vector_model,
            encoder=self.vector_encoder,
            use_faiss=self.use_faiss,
        )
        self._vector.build(self._documents)
        self._bm25 = BM25Okapi([_tokenize(doc.display_text) for doc in self._documents])
        self.build_time_s = time.perf_counter() - started

    def search(self, query: Query, top_k: int) -> RetrievalResult:
        started = time.perf_counter()
        candidate_k = _candidate_k(top_k, self.min_candidates, self.candidate_multiplier)
        vector_result = self._vector.search(query, candidate_k)
        bm25_scores = self._bm25.get_scores(_tokenize(query.text))
        candidate_indexes = [self._doc_index_by_id[hit.doc_id] for hit in vector_result.hits]
        lexical_scores = np.asarray([bm25_scores[index] for index in candidate_indexes], dtype=np.float32)
        vector_scores = np.asarray([hit.score for hit in vector_result.hits], dtype=np.float32)
        combined_scores = (
            self.lexical_weight * _normalize_vector(lexical_scores)
            + self.vector_weight * _normalize_vector(vector_scores)
        )
        pairs = sorted(
            zip(vector_result.hits, combined_scores, strict=False),
            key=lambda pair: (-float(pair[1]), pair[0].rank, pair[0].doc_id),
        )
        hits = [
            RetrievalHit(
                doc_id=hit.doc_id,
                score=float(score),
                rank=rank,
                title=hit.title,
                text=hit.text,
            )
            for rank, (hit, score) in enumerate(pairs[:top_k], 1)
        ]
        return RetrievalResult(query=query, hits=hits, latency_s=time.perf_counter() - started)


@dataclass
class MultiQueryRetriever:
    rrf_k: int = 60
    candidate_multiplier: int = 20
    min_candidates: int = 50
    name: str = "multi-query"
    build_time_s: float = 0.0

    def build(self, documents: list[Document]) -> None:
        started = time.perf_counter()
        self._bm25 = BM25Retriever()
        self._bm25.build(documents)
        self.build_time_s = time.perf_counter() - started

    def search(self, query: Query, top_k: int) -> RetrievalResult:
        started = time.perf_counter()
        candidate_k = _candidate_k(top_k, self.min_candidates, self.candidate_multiplier)
        results = [
            self._bm25.search(Query(query_id=query.query_id, text=variant), candidate_k)
            for variant in _query_variants(query.text)
        ]
        hits = _rrf_merge(results, top_k=top_k, rrf_k=self.rrf_k)
        return RetrievalResult(query=query, hits=hits, latency_s=time.perf_counter() - started)


@dataclass
class LlmQueryRewriteRetriever:
    query_expander: QueryExpansionClient
    query_model: str | None = None
    max_query_tokens: int = 96
    rrf_k: int = 60
    candidate_multiplier: int = 20
    min_candidates: int = 50
    name: str = "llm-query-rewrite"
    build_time_s: float = 0.0

    def build(self, documents: list[Document]) -> None:
        started = time.perf_counter()
        self._bm25 = BM25Retriever()
        self._bm25.build(documents)
        self.build_time_s = time.perf_counter() - started

    def search(self, query: Query, top_k: int) -> RetrievalResult:
        started = time.perf_counter()
        candidate_k = _candidate_k(top_k, self.min_candidates, self.candidate_multiplier)
        variants, metadata = _llm_query_variants(
            self.query_expander,
            query.text,
            mode="rewrite",
            max_queries=1,
            model=self.query_model,
            max_completion_tokens=self.max_query_tokens,
        )
        search_texts = _dedupe_nonempty([query.text, *variants])
        results = [
            self._bm25.search(Query(query_id=query.query_id, text=variant), candidate_k)
            for variant in search_texts
        ]
        hits = _rrf_merge(results, top_k=top_k, rrf_k=self.rrf_k)
        metadata["query_variants"] = list(search_texts)
        return RetrievalResult(
            query=query,
            hits=hits,
            latency_s=time.perf_counter() - started,
            metadata=metadata,
        )


@dataclass
class LlmMultiQueryRetriever:
    query_expander: QueryExpansionClient
    query_model: str | None = None
    max_query_tokens: int = 160
    max_queries: int = 4
    rrf_k: int = 60
    candidate_multiplier: int = 20
    min_candidates: int = 50
    name: str = "llm-multi-query"
    build_time_s: float = 0.0

    def build(self, documents: list[Document]) -> None:
        started = time.perf_counter()
        self._bm25 = BM25Retriever()
        self._bm25.build(documents)
        self.build_time_s = time.perf_counter() - started

    def search(self, query: Query, top_k: int) -> RetrievalResult:
        started = time.perf_counter()
        candidate_k = _candidate_k(top_k, self.min_candidates, self.candidate_multiplier)
        variants, metadata = _llm_query_variants(
            self.query_expander,
            query.text,
            mode="multi",
            max_queries=self.max_queries,
            model=self.query_model,
            max_completion_tokens=self.max_query_tokens,
        )
        search_texts = _dedupe_nonempty([query.text, *variants])
        results = [
            self._bm25.search(Query(query_id=query.query_id, text=variant), candidate_k)
            for variant in search_texts
        ]
        hits = _rrf_merge(results, top_k=top_k, rrf_k=self.rrf_k)
        metadata["query_variants"] = list(search_texts)
        return RetrievalResult(
            query=query,
            hits=hits,
            latency_s=time.perf_counter() - started,
            metadata=metadata,
        )


@dataclass
class ImageDigitsRetriever:
    name: str = "image-digits"
    build_time_s: float = 0.0

    def build(self, documents: list[Document]) -> None:
        from sklearn.datasets import load_digits

        started = time.perf_counter()
        digits = load_digits()
        self._items: list[dict[str, Any]] = []
        for index, (image, label) in enumerate(zip(digits.images, digits.target, strict=False)):
            label_int = int(label)
            label_word = _digit_word(label_int)
            title = f"Handwritten digit {label_int}"
            text = (
                f"Handwritten digit image from the scikit-learn digits sample dataset. "
                f"Label: {label_int} ({label_word}). "
                f"Keywords: image picture photo digit number handwritten {label_int} {label_word}."
            )
            display_text = f"{title}\n{text}"
            self._items.append(
                {
                    "doc_id": f"skdigits-{index:04d}",
                    "label": label_int,
                    "title": title,
                    "text": text,
                    "tokens": _token_counts(display_text),
                    "lower_text": display_text.lower(),
                    "image_data_url": _digit_svg_data_url(image),
                    "width": int(image.shape[1]),
                    "height": int(image.shape[0]),
                    "dataset": "sklearn-digits",
                }
            )
        self.build_time_s = time.perf_counter() - started

    def search(self, query: Query, top_k: int) -> RetrievalResult:
        started = time.perf_counter()
        query_tokens = _content_tokens(query.text)
        requested_digit = _requested_digit(query_tokens)
        query_phrase = query.text.strip().lower()
        scores = np.zeros(len(self._items), dtype=np.float32)
        for index, item in enumerate(self._items):
            score = 0.0
            for token in query_tokens:
                score += min(int(item["tokens"].get(token, 0)), 4)
            if requested_digit is not None:
                score += 100.0 if item["label"] == requested_digit else -1.0
            if query_phrase and query_phrase in item["lower_text"]:
                score += 6.0
            scores[index] = score
        ranked = _rank_scores(scores, top_k)
        hits = []
        for rank, index in enumerate(ranked, 1):
            item = self._items[index]
            hits.append(
                RetrievalHit(
                    doc_id=item["doc_id"],
                    score=float(scores[index]),
                    rank=rank,
                    title=item["title"],
                    text=item["text"],
                    metadata={
                        "kind": "image",
                        "image_data_url": item["image_data_url"],
                        "label": item["label"],
                        "dataset": item["dataset"],
                        "width": item["width"],
                        "height": item["height"],
                    },
                )
            )
        return RetrievalResult(
            query=query,
            hits=hits,
            latency_s=time.perf_counter() - started,
            metadata={
                "kind": "image",
                "dataset": "sklearn-digits",
                "query": query.text,
                "requested_label": requested_digit,
            },
        )


def create_retriever(name: str, *, vector_model: str) -> Retriever:
    from rag_bench.retriever_registry import create_retriever as registry_create_retriever

    return registry_create_retriever(name, vector_model=vector_model)


def _tokenize(text: str) -> list[str]:
    return [match.group(0).lower() for match in TOKEN_RE.finditer(text)]


def _content_tokens(text: str) -> list[str]:
    tokens = [token for token in _tokenize(text) if token not in STOPWORDS]
    return tokens or _tokenize(text)


def _token_counts(text: str) -> dict[str, int]:
    counts: dict[str, int] = {}
    for token in _tokenize(text):
        counts[token] = counts.get(token, 0) + 1
    return counts


def _rank_scores(scores: np.ndarray, top_k: int) -> list[int]:
    if top_k <= 0:
        return []
    indexes = np.arange(len(scores))
    order = np.lexsort((indexes, -scores))
    return [int(index) for index in order[: min(top_k, len(scores))]]


def _hit_from_doc(doc: Document, score: float, rank: int) -> RetrievalHit:
    return RetrievalHit(doc_id=doc.doc_id, score=score, rank=rank, title=doc.title, text=doc.text)


def _candidate_k(top_k: int, min_candidates: int, candidate_multiplier: int) -> int:
    if top_k <= 0:
        return 0
    return max(top_k, min_candidates, top_k * candidate_multiplier)


def _rrf_merge(results: list[RetrievalResult], *, top_k: int, rrf_k: int) -> list[RetrievalHit]:
    scores: dict[str, float] = {}
    best_rank: dict[str, int] = {}
    hits_by_doc_id: dict[str, RetrievalHit] = {}
    for result in results:
        for hit in result.hits:
            scores[hit.doc_id] = scores.get(hit.doc_id, 0.0) + 1.0 / (rrf_k + hit.rank)
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


def _normalize_vector(values: np.ndarray) -> np.ndarray:
    if values.size == 0:
        return values
    minimum = float(values.min())
    maximum = float(values.max())
    if maximum == minimum:
        return np.ones_like(values, dtype=np.float32) if maximum > 0 else np.zeros_like(values, dtype=np.float32)
    return (values - minimum) / (maximum - minimum)


def _query_variants(text: str) -> tuple[str, ...]:
    original = text.strip()
    tokens = _content_tokens(text)
    variants = [original]
    keyword_query = " ".join(tokens)
    if keyword_query and keyword_query.lower() != original.lower():
        variants.append(keyword_query)
    if len(tokens) >= 4:
        midpoint = max(2, len(tokens) // 2)
        variants.append(" ".join(tokens[:midpoint]))
        variants.append(" ".join(tokens[midpoint:]))
    return _dedupe_nonempty(variants)


def _llm_query_variants(
    query_expander: QueryExpansionClient,
    text: str,
    *,
    mode: str,
    max_queries: int,
    model: str | None,
    max_completion_tokens: int,
) -> tuple[tuple[str, ...], dict[str, Any]]:
    started = time.perf_counter()
    if mode == "rewrite":
        instruction = (
            "Rewrite the user question as one concise search query for a scientific retrieval system. "
            "Return only a JSON array with exactly one string."
        )
    else:
        instruction = (
            f"Generate up to {max_queries} diverse concise search queries for a scientific retrieval system. "
            "Return only a JSON array of strings. Do not answer the question."
        )
    generation = query_expander.generate(
        [
            {"role": "system", "content": instruction},
            {"role": "user", "content": text},
        ],
        model=model,
        temperature=0.0,
        max_completion_tokens=max_completion_tokens,
    )
    variants = _parse_query_array(str(getattr(generation, "answer", "")), limit=max_queries)
    if getattr(generation, "error", None):
        variants = ()
    metadata = {
        "retrieval_llm_calls": 1,
        "retrieval_llm_latency_s": float(getattr(generation, "latency_s", time.perf_counter() - started) or 0.0),
        "retrieval_llm_key_alias": getattr(generation, "key_alias", None),
        "retrieval_llm_attempted_aliases": list(getattr(generation, "attempted_aliases", []) or []),
        "retrieval_llm_rejected_aliases": list(getattr(generation, "rejected_aliases", []) or []),
        "retrieval_llm_retry_count": int(getattr(generation, "retry_count", 0) or 0),
        "retrieval_llm_scheduled_wait_s": float(getattr(generation, "scheduled_wait_s", 0.0) or 0.0),
        "retrieval_llm_prompt_tokens": getattr(generation, "prompt_tokens", None),
        "retrieval_llm_completion_tokens": getattr(generation, "completion_tokens", None),
        "retrieval_llm_total_tokens": getattr(generation, "total_tokens", None),
        "retrieval_llm_estimated_tokens": getattr(generation, "estimated_tokens", None),
        "retrieval_llm_error": getattr(generation, "error", None),
        "retrieval_llm_error_count": 1 if getattr(generation, "error", None) else 0,
    }
    return variants, metadata


def _parse_query_array(text: str, *, limit: int) -> tuple[str, ...]:
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
            return _dedupe_nonempty([str(item) for item in parsed if isinstance(item, str)])[:limit]
    lines = [
        re.sub(r"^[-*\d.)\s]+", "", line).strip(" \"'")
        for line in stripped.splitlines()
        if line.strip()
    ]
    return _dedupe_nonempty(lines)[:limit]


def _strip_code_fence(text: str) -> str:
    if text.startswith("```"):
        lines = text.splitlines()
        if len(lines) >= 2 and lines[-1].strip() == "```":
            return "\n".join(lines[1:-1]).strip()
    return text


def _dedupe_nonempty(values: list[str]) -> tuple[str, ...]:
    seen: set[str] = set()
    result: list[str] = []
    for value in values:
        normalized = " ".join(value.split())
        key = normalized.lower()
        if normalized and key not in seen:
            seen.add(key)
            result.append(normalized)
    return tuple(result)


def _requested_digit(tokens: list[str]) -> int | None:
    for token in tokens:
        if token.isdigit() and len(token) == 1:
            value = int(token)
            if 0 <= value <= 9:
                return value
        if token in DIGIT_WORDS:
            return DIGIT_WORDS[token]
    return None


def _digit_word(value: int) -> str:
    for word, number in DIGIT_WORDS.items():
        if number == value:
            return word
    return str(value)


def _digit_svg_data_url(image: np.ndarray) -> str:
    max_value = float(np.max(image)) or 1.0
    rects = ['<rect width="8" height="8" fill="#ffffff"/>']
    for y, row in enumerate(image):
        for x, value in enumerate(row):
            if float(value) <= 0.0:
                continue
            shade = 255 - int((float(value) / max_value) * 235)
            color = f"#{shade:02x}{shade:02x}{shade:02x}"
            rects.append(f'<rect x="{x}" y="{y}" width="1" height="1" fill="{color}"/>')
    svg = (
        '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 8 8" '
        'shape-rendering="crispEdges">'
        + "".join(rects)
        + "</svg>"
    )
    return "data:image/svg+xml," + quote(svg, safe="")


def _load_sentence_transformer(model_name: str) -> object:
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError as exc:
        raise RuntimeError(
            "Vector retrieval requires sentence-transformers. Install with: uv sync --extra vector"
        ) from exc
    return SentenceTransformer(model_name)


def _as_float32(values: object) -> np.ndarray:
    array = np.asarray(values, dtype=np.float32)
    if array.ndim != 2:
        raise ValueError("Encoder must return a 2D embedding array")
    return array


def _l2_normalize(matrix: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms[norms == 0.0] = 1.0
    return matrix / norms
