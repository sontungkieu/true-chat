from __future__ import annotations

import numpy as np

from rag_bench.retrievers import BM25Retriever, VectorRetriever
from rag_bench.types import Document, Query


class FakeEncoder:
    vocab = ("cat", "dog", "banana")

    def encode(self, texts: list[str]) -> np.ndarray:
        vectors = []
        for text in texts:
            lower = text.lower()
            vectors.append([float(lower.count(token)) for token in self.vocab])
        return np.asarray(vectors, dtype=np.float32)


def test_bm25_retriever_returns_relevant_document_first() -> None:
    docs = [
        Document(doc_id="cat-doc", title="Cats", text="Cats purr and chase toys."),
        Document(doc_id="banana-doc", title="Bananas", text="Bananas are yellow fruit."),
    ]
    retriever = BM25Retriever()
    retriever.build(docs)

    result = retriever.search(Query("q1", "cat purr"), top_k=2)

    assert result.hits[0].doc_id == "cat-doc"
    assert result.hits[0].rank == 1


def test_vector_retriever_returns_relevant_document_first_with_fake_encoder() -> None:
    docs = [
        Document(doc_id="dog-doc", title="Dogs", text="Dogs fetch sticks."),
        Document(doc_id="banana-doc", title="Bananas", text="Bananas are yellow fruit."),
    ]
    retriever = VectorRetriever(encoder=FakeEncoder(), use_faiss=False)
    retriever.build(docs)

    result = retriever.search(Query("q1", "yellow banana"), top_k=2)

    assert result.hits[0].doc_id == "banana-doc"
    assert result.hits[0].rank == 1
