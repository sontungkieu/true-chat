from __future__ import annotations

import numpy as np

from rag_bench.groq_client import GenerationResult
from rag_bench.retrievers import (
    BM25Retriever,
    DictionaryGraphRetriever,
    GraphBm25Retriever,
    HybridRrfRetriever,
    ImageDigitsRetriever,
    KeywordMatchRetriever,
    LlmMultiQueryRetriever,
    MultiQueryRetriever,
    VectorRerankRetriever,
    VectorRetriever,
)
from rag_bench.types import Document, Query


class FakeEncoder:
    vocab = ("cat", "dog", "banana")

    def encode(self, texts: list[str]) -> np.ndarray:
        vectors = []
        for text in texts:
            lower = text.lower()
            vectors.append([float(lower.count(token)) for token in self.vocab])
        return np.asarray(vectors, dtype=np.float32)


class FakeQueryExpander:
    def generate(self, *_args, **_kwargs) -> GenerationResult:
        return GenerationResult(
            answer='["cat purr", "feline purr"]',
            key_alias="alias-a",
            attempted_aliases=["alias-a"],
            latency_s=0.01,
            retry_count=0,
            prompt_tokens=10,
            completion_tokens=6,
            total_tokens=16,
            estimated_tokens=20,
        )


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


def test_keyword_match_retriever_returns_exact_keyword_match_first() -> None:
    docs = [
        Document(doc_id="cat-doc", title="Cats", text="Cats purr and chase toys."),
        Document(doc_id="banana-doc", title="Bananas", text="Bananas are yellow fruit."),
    ]
    retriever = KeywordMatchRetriever()
    retriever.build(docs)

    result = retriever.search(Query("q1", "yellow banana"), top_k=2)

    assert result.hits[0].doc_id == "banana-doc"
    assert result.hits[0].rank == 1


def test_image_digits_retriever_returns_lightweight_image_metadata() -> None:
    retriever = ImageDigitsRetriever()
    retriever.build([])

    result = retriever.search(Query("q1", "digit 7 image"), top_k=3)

    assert result.hits[0].metadata["kind"] == "image"
    assert result.hits[0].metadata["label"] == 7
    assert result.hits[0].metadata["image_data_url"].startswith("data:image/svg+xml,")
    assert result.metadata["dataset"] == "sklearn-digits"


def test_multi_query_retriever_returns_relevant_document_first() -> None:
    docs = [
        Document(doc_id="cat-doc", title="Cats", text="Cats purr and chase toys."),
        Document(doc_id="banana-doc", title="Bananas", text="Bananas are yellow fruit."),
    ]
    retriever = MultiQueryRetriever()
    retriever.build(docs)

    result = retriever.search(Query("q1", "Which animal is known to purr and chase toys?"), top_k=2)

    assert result.hits[0].doc_id == "cat-doc"
    assert result.hits[0].rank == 1


def test_multi_query_keeps_scientific_token_from_vietnamese_instruction() -> None:
    docs = [
        Document(
            doc_id="bcl2-doc",
            title="BH1 and BH2 domains of Bcl-2",
            text="BH1 and BH2 are conserved Bcl-2 domains required for apoptosis inhibition.",
        ),
        Document(
            doc_id="noise-doc",
            title="Instruction fragments",
            text="vi t ng th ch b vi t ng th ch b vi t ng th ch b unrelated clinical trial",
        ),
    ]
    retriever = MultiQueryRetriever()
    retriever.build(docs)

    result = retriever.search(Query("q1", "giải thích BH1 bằng tiếng Việt"), top_k=2)

    assert result.hits[0].doc_id == "bcl2-doc"
    assert result.hits[0].rank == 1


def test_graph_bm25_expands_from_seed_document_neighbors() -> None:
    docs = [
        Document(
            doc_id="alpha-seed",
            title="Alpha seed",
            text="Alpha seed shares bridge kinase pathway terms.",
        ),
        Document(
            doc_id="second-hop",
            title="Second hop",
            text="Bridge kinase pathway evidence gives the downstream answer.",
        ),
        Document(
            doc_id="noise",
            title="Noise",
            text="Unrelated banana document.",
        ),
    ]
    retriever = GraphBm25Retriever()
    retriever.build(docs)

    result = retriever.search(Query("q1", "alpha"), top_k=2)

    assert [hit.doc_id for hit in result.hits] == ["alpha-seed", "second-hop"]
    assert result.metadata["graph_candidate_count"] >= 2
    assert {"bridge", "kinase", "pathway"}.intersection(result.metadata["graph_expansion_terms"])
    assert result.hits[1].metadata["graph_score"] > 0.0


def test_dictionary_graph_retriever_preserves_dictionary_metadata() -> None:
    docs = [
        Document(
            doc_id="A-0001",
            title="AMONIT",
            text="AMONIT, thuốc nổ phá từ amoni nitrat.",
            metadata={"kind": "dictionary", "headword": "AMONIT", "rich_blocks": [{"type": "paragraph"}]},
        ),
        Document(doc_id="B-0001", title="B-72", text="B-72, tổ hợp tên lửa chống tăng."),
    ]
    retriever = DictionaryGraphRetriever()
    retriever.build(docs)

    result = retriever.search(Query("q1", "AMONIT"), top_k=2)

    assert result.hits[0].doc_id == "A-0001"
    assert result.hits[0].metadata["kind"] == "dictionary"
    assert result.hits[0].metadata["rich_blocks"] == [{"type": "paragraph"}]
    assert result.metadata["kind"] == "dictionary"


def test_dictionary_graph_retriever_matches_accent_folded_headword() -> None:
    docs = [
        Document(
            doc_id="base:H-0011",
            title="HEXOGEN",
            text="HEXOGEN, thuốc nổ mạnh dùng trong kỹ thuật quân sự.",
            metadata={"kind": "dictionary", "headword": "HEXOGEN"},
        ),
        Document(
            doc_id="base:T-0217",
            title="TRẠM NỔ",
            text="TRẠM NỔ, phần tử của mạch nổ có thể chứa têtryl, hêxôgen.",
            metadata={"kind": "dictionary", "headword": "TRẠM NỔ"},
        ),
        Document(
            doc_id="base:A-0002",
            title="AMONIT",
            text="AMONIT, hỗn hợp có các chất nổ có thể là hêxôgen, TNT.",
            metadata={"kind": "dictionary", "headword": "AMONIT"},
        ),
    ]
    retriever = DictionaryGraphRetriever()
    retriever.build(docs)

    plain = retriever.search(Query("q1", "hexogen"), top_k=3)
    accented = retriever.search(Query("q2", "hêxôgen"), top_k=3)
    hyphenated = retriever.search(Query("q3", "hê-xô-gen"), top_k=3)
    ascii_hyphenated = retriever.search(Query("q4", "he-xo-gen"), top_k=3)

    assert plain.hits[0].doc_id == "base:H-0011"
    assert accented.hits[0].doc_id == "base:H-0011"
    assert hyphenated.hits[0].doc_id == "base:H-0011"
    assert ascii_hyphenated.hits[0].doc_id == "base:H-0011"
    assert accented.metadata["direct_candidate_count"] >= 3


def test_dictionary_graph_retriever_matches_abbreviation_alias_to_headword() -> None:
    docs = [
        Document(
            doc_id="base:T-0130",
            title="THƯỚC PB-74",
            text="THƯỚC PB-74, khí tài tính toán của pháo binh.",
            metadata={"kind": "dictionary", "headword": "THƯỚC PB-74"},
        ),
        Document(
            doc_id="base:P-0023",
            title="PHÁO BINH",
            text="PHÁO BINH, lực lượng tác chiến. Ở Việt Nam, PB ra đời sớm.",
            metadata={
                "kind": "dictionary",
                "headword": "PHÁO BINH",
                "aliases": ["PB", "Pháo binh"],
                "concepts": ["lực lượng tác chiến"],
            },
        ),
        Document(
            doc_id="base:P-0025",
            title="PHÁO BINH BIÊN CHẾ",
            text="PHÁO BINH BIÊN CHẾ, gọi chung các phân đội pháo binh.",
            metadata={"kind": "dictionary", "headword": "PHÁO BINH BIÊN CHẾ", "aliases": ["PBBC"]},
        ),
    ]
    retriever = DictionaryGraphRetriever()
    retriever.build(docs)

    pb = retriever.search(Query("q1", "pb"), top_k=3)
    pbbc = retriever.search(Query("q2", "pbbc"), top_k=3)

    assert pb.hits[0].doc_id == "base:P-0023"
    assert pbbc.hits[0].doc_id == "base:P-0025"


def test_llm_multi_query_retriever_records_retrieval_llm_metadata() -> None:
    docs = [
        Document(doc_id="cat-doc", title="Cats", text="Cats purr and chase toys."),
        Document(doc_id="banana-doc", title="Bananas", text="Bananas are yellow fruit."),
    ]
    retriever = LlmMultiQueryRetriever(query_expander=FakeQueryExpander())
    retriever.build(docs)

    result = retriever.search(Query("q1", "Which animal purrs?"), top_k=2)

    assert result.hits[0].doc_id == "cat-doc"
    assert result.metadata["retrieval_llm_calls"] == 1
    assert result.metadata["retrieval_llm_key_alias"] == "alias-a"
    assert result.metadata["query_variants"] == ["Which animal purrs?", "cat purr", "feline purr"]


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


def test_hybrid_and_rerank_retrievers_work_with_fake_encoder() -> None:
    docs = [
        Document(doc_id="dog-doc", title="Dogs", text="Dogs fetch sticks."),
        Document(doc_id="banana-doc", title="Bananas", text="Bananas are yellow fruit."),
    ]
    hybrid = HybridRrfRetriever(vector_encoder=FakeEncoder(), use_faiss=False)
    rerank = VectorRerankRetriever(vector_encoder=FakeEncoder(), use_faiss=False)
    hybrid.build(docs)
    rerank.build(docs)

    hybrid_result = hybrid.search(Query("q1", "yellow banana"), top_k=2)
    rerank_result = rerank.search(Query("q1", "yellow banana"), top_k=2)

    assert hybrid_result.hits[0].doc_id == "banana-doc"
    assert rerank_result.hits[0].doc_id == "banana-doc"
