from __future__ import annotations

import re
import time
from dataclasses import dataclass
from typing import Protocol

import numpy as np

from rag_bench.types import Document, RetrievalHit, RetrievalResult, Query


TOKEN_RE = re.compile(r"[A-Za-z0-9_]+")


class Retriever(Protocol):
    name: str
    build_time_s: float

    def build(self, documents: list[Document]) -> None: ...

    def search(self, query: Query, top_k: int) -> RetrievalResult: ...


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


def create_retriever(name: str, *, vector_model: str) -> Retriever:
    normalized = name.strip().lower()
    if normalized in {"bm25", "lexical"}:
        return BM25Retriever()
    if normalized == "tfidf":
        return TfidfRetriever()
    if normalized in {"vector", "dense"}:
        return VectorRetriever(model_name=vector_model)
    raise ValueError(f"Unknown retriever: {name}")


def _tokenize(text: str) -> list[str]:
    return [match.group(0).lower() for match in TOKEN_RE.finditer(text)]


def _rank_scores(scores: np.ndarray, top_k: int) -> list[int]:
    if top_k <= 0:
        return []
    indexes = np.arange(len(scores))
    order = np.lexsort((indexes, -scores))
    return [int(index) for index in order[: min(top_k, len(scores))]]


def _hit_from_doc(doc: Document, score: float, rank: int) -> RetrievalHit:
    return RetrievalHit(doc_id=doc.doc_id, score=score, rank=rank, title=doc.title, text=doc.text)


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
