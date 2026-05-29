from __future__ import annotations

import json
import re
import time
import unicodedata
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
class GraphBm25Retriever:
    seed_multiplier: int = 10
    min_seed_candidates: int = 20
    candidate_multiplier: int = 30
    min_candidates: int = 80
    max_doc_terms: int = 40
    max_expansion_terms: int = 48
    max_doc_freq_ratio: float = 0.15
    lexical_weight: float = 0.65
    graph_weight: float = 0.35
    rrf_k: int = 60
    name: str = "graph-bm25"
    build_time_s: float = 0.0

    def build(self, documents: list[Document]) -> None:
        from rank_bm25 import BM25Okapi

        started = time.perf_counter()
        self._documents = list(documents)
        self._bm25 = BM25Okapi([_tokenize(doc.display_text) for doc in self._documents])
        self._doc_graph_terms: list[tuple[str, ...]] = []
        self._term_to_doc_indexes: dict[str, tuple[int, ...]] = {}
        self._term_idf: dict[str, float] = {}

        doc_freq: dict[str, int] = {}
        for doc in self._documents:
            counts = _graph_token_counts(doc.display_text)
            for term in counts:
                doc_freq[term] = doc_freq.get(term, 0) + 1

        document_count = len(self._documents)
        if document_count == 0:
            self.build_time_s = time.perf_counter() - started
            return

        max_doc_freq = max(2, int(document_count * self.max_doc_freq_ratio))
        graph_terms = {
            term
            for term, frequency in doc_freq.items()
            if 2 <= frequency <= max_doc_freq
        }
        self._term_idf = {
            term: float(np.log((1.0 + document_count) / (1.0 + doc_freq[term])) + 1.0)
            for term in graph_terms
        }

        postings: dict[str, list[int]] = {term: [] for term in graph_terms}
        for doc_index, doc in enumerate(self._documents):
            counts = _graph_token_counts(doc.display_text)
            ranked_terms = sorted(
                (term for term in counts if term in graph_terms),
                key=lambda term: (-counts[term] * self._term_idf[term], term),
            )
            selected_terms = tuple(ranked_terms[: self.max_doc_terms])
            self._doc_graph_terms.append(selected_terms)
            for term in selected_terms:
                postings[term].append(doc_index)

        self._term_to_doc_indexes = {
            term: tuple(indexes)
            for term, indexes in postings.items()
            if len(indexes) >= 2
        }
        self.build_time_s = time.perf_counter() - started

    def search(self, query: Query, top_k: int) -> RetrievalResult:
        started = time.perf_counter()
        if top_k <= 0:
            return RetrievalResult(query=query, hits=[], latency_s=time.perf_counter() - started)

        bm25_scores = np.asarray(self._bm25.get_scores(_tokenize(query.text)), dtype=np.float32)
        seed_k = min(
            len(self._documents),
            _candidate_k(top_k, self.min_seed_candidates, self.seed_multiplier),
        )
        seed_indexes = _rank_scores(bm25_scores, seed_k)
        query_terms = {
            term for term in _content_tokens(query.text)
            if term in self._term_to_doc_indexes
        }
        expansion_terms = self._expansion_terms(seed_indexes, query_terms)
        graph_scores = self._graph_scores(seed_indexes, query_terms, expansion_terms)

        candidate_indexes = set(seed_indexes[: _candidate_k(top_k, self.min_candidates, self.candidate_multiplier)])
        for term in expansion_terms:
            candidate_indexes.update(self._term_to_doc_indexes.get(term, ()))
        if not candidate_indexes:
            candidate_indexes.update(seed_indexes[:top_k])

        ranked = self._rank_combined(candidate_indexes, bm25_scores, graph_scores, top_k)
        hits = [
            RetrievalHit(
                doc_id=self._documents[index].doc_id,
                score=float(score),
                rank=rank,
                title=self._documents[index].title,
                text=self._documents[index].text,
                metadata={
                    "bm25_score": float(bm25_scores[index]),
                    "graph_score": float(graph_scores[index]),
                },
            )
            for rank, (index, score) in enumerate(ranked, 1)
        ]
        return RetrievalResult(
            query=query,
            hits=hits,
            latency_s=time.perf_counter() - started,
            metadata={
                "seed_count": len(seed_indexes),
                "graph_candidate_count": len(candidate_indexes),
                "graph_expansion_terms": list(expansion_terms),
                "graph_query_terms": sorted(query_terms),
                "lexical_weight": self.lexical_weight,
                "graph_weight": self.graph_weight,
            },
        )

    def _expansion_terms(self, seed_indexes: list[int], query_terms: set[str]) -> tuple[str, ...]:
        terms: list[str] = []
        seen: set[str] = set()

        def add(term: str) -> None:
            if term not in seen and term in self._term_to_doc_indexes:
                seen.add(term)
                terms.append(term)

        for term in sorted(query_terms, key=lambda value: (-self._term_idf.get(value, 0.0), value)):
            add(term)
        for index in seed_indexes[: max(1, min(10, len(seed_indexes)))]:
            for term in self._doc_graph_terms[index]:
                add(term)
                if len(terms) >= self.max_expansion_terms:
                    return tuple(terms)
        return tuple(terms[: self.max_expansion_terms])

    def _graph_scores(
        self,
        seed_indexes: list[int],
        query_terms: set[str],
        expansion_terms: tuple[str, ...],
    ) -> np.ndarray:
        scores = np.zeros(len(self._documents), dtype=np.float32)
        expansion_set = set(expansion_terms)
        for seed_rank, seed_index in enumerate(seed_indexes, 1):
            seed_terms = set(self._doc_graph_terms[seed_index])
            relevant_terms = seed_terms & expansion_set
            if not relevant_terms:
                continue
            seed_weight = 1.0 / (self.rrf_k + seed_rank)
            for term in relevant_terms:
                term_weight = self._term_idf.get(term, 1.0) * (2.0 if term in query_terms else 1.0)
                for doc_index in self._term_to_doc_indexes.get(term, ()):
                    scores[doc_index] += seed_weight * term_weight
        return scores

    def _rank_combined(
        self,
        candidate_indexes: set[int],
        bm25_scores: np.ndarray,
        graph_scores: np.ndarray,
        top_k: int,
    ) -> list[tuple[int, float]]:
        indexes = sorted(candidate_indexes)
        if not indexes:
            return []
        lexical = _normalize_vector(np.asarray([bm25_scores[index] for index in indexes], dtype=np.float32))
        graph = _normalize_vector(np.asarray([graph_scores[index] for index in indexes], dtype=np.float32))
        combined = self.lexical_weight * lexical + self.graph_weight * graph
        rows = [
            (index, float(score), float(lexical_score), float(graph_score))
            for index, score, lexical_score, graph_score in zip(indexes, combined, lexical, graph, strict=False)
        ]
        rows.sort(key=lambda row: (-row[1], -row[2], -row[3], self._documents[row[0]].doc_id))
        return [(index, score) for index, score, _lexical, _graph in rows[:top_k]]


@dataclass
class DictionaryGraphRetriever:
    rrf_k: int = 60
    candidate_multiplier: int = 20
    min_candidates: int = 50
    name: str = "dictionary-graph"
    build_time_s: float = 0.0

    def build(self, documents: list[Document]) -> None:
        started = time.perf_counter()
        self._documents = list(documents)
        self._headword_indexes: dict[str, list[int]] = {}
        self._alias_indexes: dict[str, list[int]] = {}
        self._concept_indexes: dict[str, list[int]] = {}
        self._abbreviation_scores: dict[str, dict[int, float]] = {}
        self._folded_texts: list[str] = []
        for index, doc in enumerate(self._documents):
            headword = str(doc.metadata.get("headword") or doc.title or "")
            headword_key = _dictionary_query_key(headword)
            for key in _dictionary_query_keys(headword):
                self._headword_indexes.setdefault(key, []).append(index)
            alias_keys: list[str] = []
            for alias in _metadata_text_values(doc.metadata.get("aliases")):
                for alias_key in _dictionary_query_keys(alias):
                    alias_keys.append(alias_key)
                    self._alias_indexes.setdefault(alias_key, []).append(index)
            abbreviation_key = _dictionary_abbreviation_key(headword)
            if abbreviation_key and _has_abbreviation_evidence(doc, abbreviation_key, alias_keys):
                self._abbreviation_scores.setdefault(abbreviation_key, {})[index] = _abbreviation_score(
                    abbreviation_key,
                    headword_key,
                    alias_keys,
                )
            for concept in _metadata_text_values(doc.metadata.get("concepts")):
                for concept_key in _dictionary_query_keys(concept):
                    self._concept_indexes.setdefault(concept_key, []).append(index)
            self._folded_texts.append(_dictionary_document_key(doc))
        self._bm25 = BM25Retriever()
        self._graph = GraphBm25Retriever()
        if self._documents:
            self._bm25.build(self._documents)
            self._graph.build(self._documents)
        self.build_time_s = time.perf_counter() - started

    def search(self, query: Query, top_k: int) -> RetrievalResult:
        started = time.perf_counter()
        if top_k <= 0 or not self._documents:
            return RetrievalResult(
                query=query,
                hits=[],
                latency_s=time.perf_counter() - started,
                metadata={"kind": "dictionary", "entry_count": len(getattr(self, "_documents", []))},
            )

        candidate_k = _candidate_k(top_k, self.min_candidates, self.candidate_multiplier)
        lexical = self._bm25.search(query, candidate_k)
        graph = self._graph.search(query, candidate_k)
        direct = self._direct_search(query, candidate_k)
        hits = _dictionary_merge(lexical, graph, direct, query=query, top_k=top_k, rrf_k=self.rrf_k)
        return RetrievalResult(
            query=query,
            hits=hits,
            latency_s=time.perf_counter() - started,
            metadata={
                "kind": "dictionary",
                "entry_count": len(self._documents),
                "lexical_latency_s": lexical.latency_s,
                "graph_latency_s": graph.latency_s,
                "direct_candidate_count": len(direct.hits),
                "graph_metadata": graph.metadata,
            },
        )

    def _direct_search(self, query: Query, top_k: int) -> RetrievalResult:
        started = time.perf_counter()
        query_keys = _dictionary_query_keys(query.text)
        if not query_keys:
            return RetrievalResult(query=query, hits=[], latency_s=time.perf_counter() - started)

        index_scores: dict[int, float] = {}
        highlight_terms = _dictionary_highlight_terms(query.text)
        for query_key in query_keys:
            query_token_count = len(query_key.split())
            if query_token_count >= 2:
                phrase = f" {query_key} "
                phrase_score = 1.1 + min(query_token_count, 6) * 0.05
                for index, folded_text in enumerate(self._folded_texts):
                    if phrase in f" {folded_text} ":
                        index_scores[index] = max(index_scores.get(index, 0.0), phrase_score)
            elif len(query_key) >= 3:
                phrase = f" {query_key} "
                for index, folded_text in enumerate(self._folded_texts):
                    if phrase in f" {folded_text} ":
                        index_scores[index] = max(index_scores.get(index, 0.0), 0.45)

            for index in self._headword_indexes.get(query_key, []):
                index_scores[index] = max(index_scores.get(index, 0.0), 1.0)
            for index, score in self._abbreviation_scores.get(query_key, {}).items():
                index_scores[index] = max(index_scores.get(index, 0.0), score)
            for index in self._alias_indexes.get(query_key, []):
                index_scores[index] = max(index_scores.get(index, 0.0), 0.8)
            for index in self._concept_indexes.get(query_key, []):
                index_scores[index] = max(index_scores.get(index, 0.0), 0.55)

            if len(query_key) >= 3:
                for headword_key, indexes in self._headword_indexes.items():
                    if headword_key == query_key:
                        continue
                    if query_key in headword_key or headword_key in query_key:
                        for index in indexes:
                            index_scores[index] = max(index_scores.get(index, 0.0), 0.7)

        ranked = sorted(index_scores, key=lambda index: (-index_scores[index], self._documents[index].doc_id))
        hits = []
        for rank, index in enumerate(ranked[:top_k], 1):
            doc = self._documents[index]
            metadata = dict(doc.metadata)
            metadata["dictionary_direct_score"] = index_scores[index]
            if highlight_terms:
                metadata["query_highlights"] = list(highlight_terms)
            hits.append(
                RetrievalHit(
                    doc_id=doc.doc_id,
                    score=index_scores[index],
                    rank=rank,
                    title=doc.title,
                    text=doc.text,
                    metadata=metadata,
                )
            )
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


def _graph_token_counts(text: str) -> dict[str, int]:
    counts: dict[str, int] = {}
    for token in _content_tokens(text):
        if _is_graph_token(token):
            counts[token] = counts.get(token, 0) + 1
    return counts


def _is_graph_token(token: str) -> bool:
    return len(token) >= 3 or any(character.isdigit() for character in token)


def _rank_scores(scores: np.ndarray, top_k: int) -> list[int]:
    if top_k <= 0:
        return []
    indexes = np.arange(len(scores))
    order = np.lexsort((indexes, -scores))
    return [int(index) for index in order[: min(top_k, len(scores))]]


def _hit_from_doc(doc: Document, score: float, rank: int) -> RetrievalHit:
    return RetrievalHit(doc_id=doc.doc_id, score=score, rank=rank, title=doc.title, text=doc.text, metadata=dict(doc.metadata))


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


def _dictionary_merge(
    lexical: RetrievalResult,
    graph: RetrievalResult,
    direct: RetrievalResult,
    *,
    query: Query,
    top_k: int,
    rrf_k: int,
) -> list[RetrievalHit]:
    query_keys = _dictionary_query_keys(query.text)
    scores: dict[str, float] = {}
    best_rank: dict[str, int] = {}
    hits_by_doc_id: dict[str, RetrievalHit] = {}
    for weight, result in ((1.0, lexical), (0.75, graph), (1.25, direct)):
        for hit in result.hits:
            direct_score = float(hit.metadata.get("dictionary_direct_score") or 0.0)
            scores[hit.doc_id] = scores.get(hit.doc_id, 0.0) + weight / (rrf_k + hit.rank) + direct_score
            best_rank[hit.doc_id] = min(best_rank.get(hit.doc_id, hit.rank), hit.rank)
            existing = hits_by_doc_id.setdefault(hit.doc_id, hit)
            if existing is not hit and hit.metadata:
                existing.metadata.update(hit.metadata)
    if query_keys:
        for doc_id, hit in hits_by_doc_id.items():
            headword = str(hit.metadata.get("headword") or hit.title or "")
            headword_keys = _dictionary_query_keys(headword)
            if set(headword_keys) & set(query_keys):
                scores[doc_id] = scores.get(doc_id, 0.0) + 1.0
            elif any(
                _dictionary_headword_partial_match(query_key, headword_key)
                for query_key in query_keys
                for headword_key in headword_keys
            ):
                scores[doc_id] = scores.get(doc_id, 0.0) + 0.35
    ranked = sorted(scores, key=lambda doc_id: (-scores[doc_id], best_rank.get(doc_id, 9999), doc_id))
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


def _dictionary_query_key(text: str) -> str:
    keys = _dictionary_query_keys(text)
    return keys[0] if keys else ""


def _dictionary_query_keys(text: str) -> list[str]:
    normalized = re.sub(r"\s+", " ", text).strip().lower()
    normalized = re.sub(r"^/(dict|dictionary|tu-dien|từ-điển)\s+", "", normalized)
    segments = [segment.strip() for segment in re.split(r"[,;]+", normalized) if segment.strip()]
    keys: list[str] = []
    for segment in segments or [normalized]:
        keys.extend(_dictionary_key_variants(segment))
    return list(_dedupe_nonempty(keys))


def _dictionary_highlight_terms(text: str) -> tuple[str, ...]:
    normalized = re.sub(r"\s+", " ", text).strip()
    normalized = re.sub(r"^/(dict|dictionary|tu-dien|từ-điển)\s+", "", normalized, flags=re.IGNORECASE)
    terms = [segment.strip() for segment in re.split(r"[,;]+", normalized) if segment.strip()]
    terms = [term for term in terms if len(_dictionary_fold_text(term)) >= 2]
    return _dedupe_nonempty(terms)


def _dictionary_headword_partial_match(query_key: str, headword_key: str) -> bool:
    if not query_key or not headword_key:
        return False
    query_tokens = query_key.split()
    headword_tokens = headword_key.split()
    if query_key in headword_key:
        return True
    if headword_key in query_key:
        return len(headword_tokens) >= 2 or len(query_tokens) == 1
    return False


def _dictionary_document_key(doc: Document) -> str:
    parts = [
        doc.title,
        doc.text,
        str(doc.metadata.get("headword") or ""),
        str(doc.metadata.get("raw_docx_text") or ""),
        " ".join(_metadata_text_values(doc.metadata.get("aliases"))),
        " ".join(_metadata_text_values(doc.metadata.get("concepts"))),
    ]
    folded = _dictionary_fold_text(" ".join(part for part in parts if part))
    compact = folded.replace(" ", "")
    return f"{folded} {compact}" if compact and compact != folded else folded


def _dictionary_abbreviation_key(text: str) -> str:
    tokens = _dictionary_fold_text(text).split()
    abbreviation = "".join(token[0] for token in tokens if any(character.isalpha() for character in token))
    return abbreviation if len(abbreviation) >= 2 else ""


def _has_abbreviation_evidence(doc: Document, abbreviation_key: str, alias_keys: list[str]) -> bool:
    if abbreviation_key in alias_keys:
        return True
    if len(abbreviation_key) < 2:
        return False
    pattern = re.compile(rf"(?<![A-Za-z0-9]){re.escape(abbreviation_key)}(?![A-Za-z0-9])", re.IGNORECASE)
    return any(
        pattern.search(str(value or ""))
        for value in (
            doc.text,
            doc.metadata.get("raw_docx_text"),
        )
    )


def _abbreviation_score(abbreviation_key: str, headword_key: str, alias_keys: list[str]) -> float:
    if abbreviation_key in alias_keys and headword_key in alias_keys:
        return 0.95
    if abbreviation_key in alias_keys:
        return 0.75
    return 0.6


def _metadata_text_values(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    return [str(item) for item in value if item is not None and str(item).strip()]


def _dictionary_fold_text(text: str) -> str:
    lowered = text.lower().replace("đ", "dd")
    folded = "".join(
        char for char in unicodedata.normalize("NFD", lowered) if unicodedata.category(char) != "Mn"
    )
    folded = re.sub(r"[^a-z0-9]+", " ", folded)
    return re.sub(r"\s+", " ", folded).strip()


def _dictionary_key_variants(text: str) -> list[str]:
    folded = _dictionary_fold_text(text)
    variants = [folded] if folded else []
    compact = folded.replace(" ", "")
    if compact and compact != folded:
        tokens = folded.split()
        # Hyphenated transliterations like "hê-xô-gen" should match the canonical
        # entry "HEXOGEN", but broad multi-word terms such as "pháo binh" should
        # keep their normal spaced key.
        if len(tokens) >= 2 and all(len(token) <= 3 for token in tokens):
            variants.append(compact)
    return _dedupe_nonempty(variants)


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
