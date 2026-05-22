from __future__ import annotations

import builtins
import sys

import numpy as np
import pytest

from rag_bench.retriever_registry import (
    RETRIEVER_CATEGORIES,
    create_retriever,
    get_retriever_spec,
    list_retriever_ids,
    list_retrievers,
    normalize_retriever_id,
)
from rag_bench.retrievers import BM25Retriever, DictionaryGraphRetriever, GraphBm25Retriever, ImageDigitsRetriever, VectorRetriever
from rag_bench.types import Document


class FakeEncoder:
    def encode(self, texts: list[str]) -> np.ndarray:
        return np.ones((len(texts), 2), dtype=np.float32)


def test_registry_lists_working_retrieval_strategies() -> None:
    specs = list_retrievers()

    assert list_retriever_ids() == (
        "bm25",
        "tfidf",
        "keyword-match",
        "multi-query",
        "graph-bm25",
        "llm-query-rewrite",
        "llm-multi-query",
        "image-digits",
        "dictionary-graph",
        "vector",
        "hybrid-rrf",
        "vector-rerank",
    )
    assert RETRIEVER_CATEGORIES == ("text", "image", "keyword", "dictionary")
    assert [spec.id for spec in specs] == list(list_retriever_ids())
    assert {spec.category for spec in specs} == {"text", "keyword", "image", "dictionary"}
    assert get_retriever_spec("vector").requires_extra == "vector"
    assert get_retriever_spec("llm-multi-query").uses_llm is True


def test_registry_normalizes_aliases_and_creates_retrievers() -> None:
    assert normalize_retriever_id(" lexical ") == "bm25"
    assert normalize_retriever_id("Dense") == "vector"
    assert normalize_retriever_id("find") == "keyword-match"
    assert normalize_retriever_id("graph") == "graph-bm25"
    assert normalize_retriever_id("img") == "image-digits"
    assert normalize_retriever_id("dict") == "dictionary-graph"
    assert normalize_retriever_id("rerank") == "vector-rerank"

    assert isinstance(create_retriever("lexical", vector_model="unused"), BM25Retriever)
    assert isinstance(create_retriever("graph-rag", vector_model="unused"), GraphBm25Retriever)
    assert isinstance(create_retriever("img", vector_model="unused"), ImageDigitsRetriever)
    assert isinstance(create_retriever("dictionary", vector_model="unused"), DictionaryGraphRetriever)
    vector = create_retriever("dense", vector_model="fake-model")

    assert isinstance(vector, VectorRetriever)
    assert vector.model_name == "fake-model"


def test_unknown_retriever_fails_with_choices() -> None:
    with pytest.raises(ValueError, match="Unknown retriever: missing"):
        get_retriever_spec("missing")


def test_vector_retriever_missing_faiss_error_mentions_vector_extra(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delitem(sys.modules, "faiss", raising=False)
    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name == "faiss":
            raise ImportError("missing faiss")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    retriever = VectorRetriever(encoder=FakeEncoder(), use_faiss=True)

    with pytest.raises(RuntimeError, match="uv sync --extra vector"):
        retriever.build([Document(doc_id="doc-1", text="alpha beta")])
