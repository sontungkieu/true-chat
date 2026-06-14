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


def test_dictionary_graph_retriever_expands_typed_graph_paths() -> None:
    docs = [
        Document(
            doc_id="base:A-0002",
            title="AMONIT",
            text="AMONIT, hỗn hợp của amoni nitrat.",
            metadata={
                "kind": "dictionary",
                "headword": "AMONIT",
                "dictionary_graph_edges": [
                    {
                        "source": "base:A-0002",
                        "target": "concept:thuoc no",
                        "type": "has_concept",
                        "source_entry_id": "base:A-0002",
                        "target_label": "thuốc nổ",
                        "target_type": "concept",
                        "evidence_text": "AMONIT là thuốc nổ phá",
                        "confidence": 0.9,
                    }
                ],
            },
        ),
        Document(
            doc_id="base:N-0001",
            title="NỔ",
            text="NỔ, biến đổi rất nhanh sinh công.",
            metadata={
                "kind": "dictionary",
                "headword": "NỔ",
                "concepts": ["thuốc nổ"],
                "dictionary_graph_edges": [
                    {
                        "source": "base:N-0001",
                        "target": "concept:thuoc no",
                        "type": "has_concept",
                        "source_entry_id": "base:N-0001",
                        "target_label": "thuốc nổ",
                        "target_type": "concept",
                        "evidence_text": "quá trình nổ của thuốc nổ",
                        "confidence": 0.9,
                    }
                ],
            },
        ),
    ]
    retriever = DictionaryGraphRetriever()
    retriever.build(docs)

    result = retriever.search(Query("q1", "AMONIT"), top_k=2)

    assert result.hits[0].doc_id == "base:A-0002"
    graph_hit = next(hit for hit in result.hits if hit.doc_id == "base:N-0001")
    assert graph_hit.metadata["dictionary_match_mode"] == "graph"
    assert graph_hit.metadata["dictionary_relation"] == "has_concept"
    assert graph_hit.metadata["dictionary_evidence_text"] == "AMONIT là thuốc nổ phá"
    assert [item["label"] for item in graph_hit.metadata["dictionary_graph_path"]] == [
        "AMONIT",
        "has_concept",
        "thuốc nổ",
        "NỔ",
    ]
    assert result.metadata["typed_graph_candidate_count"] >= 1


def test_dictionary_graph_retriever_uses_typed_relation_for_concept_query() -> None:
    docs = [
        Document(
            doc_id="graph-hit",
            title="NỔ",
            text="NỔ, biến đổi rất nhanh sinh công.",
            metadata={
                "kind": "dictionary",
                "headword": "NỔ",
                "concepts": ["thuốc nổ"],
                "dictionary_graph_edges": [
                    {
                        "source": "graph-hit",
                        "target": "concept:thuoc no",
                        "type": "has_concept",
                        "source_entry_id": "graph-hit",
                        "target_label": "thuốc nổ",
                        "target_type": "concept",
                        "confidence": 0.95,
                    }
                ],
            },
        ),
        Document(
            doc_id="text-only",
            title="MỤC YẾU",
            text="MỤC YẾU, chỉ nhắc thuốc nổ như một ví dụ phụ.",
            metadata={"kind": "dictionary", "headword": "MỤC YẾU"},
        ),
    ]
    retriever = DictionaryGraphRetriever()
    retriever.build(docs)

    result = retriever.search(Query("q1", "thuốc nổ"), top_k=2)

    assert result.hits[0].doc_id == "graph-hit"
    assert result.hits[0].metadata["dictionary_relation"] == "has_concept"
    assert result.hits[0].metadata["dictionary_graph_path_text"].startswith("thuốc nổ")


def test_dictionary_graph_retriever_keeps_related_to_below_exact_match() -> None:
    docs = [
        Document(
            doc_id="alpha",
            title="ALPHA",
            text="ALPHA, mục liên quan đến beta.",
            metadata={
                "kind": "dictionary",
                "headword": "ALPHA",
                "dictionary_graph_edges": [
                    {
                        "source": "alpha",
                        "target": "beta",
                        "type": "related_to",
                        "source_entry_id": "alpha",
                        "target_label": "BETA",
                        "target_type": "entry",
                        "confidence": 1.0,
                    }
                ],
            },
        ),
        Document(
            doc_id="beta",
            title="BETA",
            text="BETA, mục chính xác.",
            metadata={"kind": "dictionary", "headword": "BETA"},
        ),
    ]
    retriever = DictionaryGraphRetriever()
    retriever.build(docs)

    result = retriever.search(Query("q1", "BETA"), top_k=2)

    assert result.hits[0].doc_id == "beta"
    assert result.hits[0].metadata["dictionary_match_mode"] == "strict"


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
            doc_id="base:A-0001",
            title="ALIAS ONLY",
            text="ALIAS ONLY, synthetic entry that exposes PB as a secondary alias.",
            metadata={"kind": "dictionary", "headword": "ALIAS ONLY", "aliases": ["PB"]},
        ),
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
    assert pb.hits[0].metadata["dictionary_direct_score"] > pb.hits[1].metadata["dictionary_direct_score"]
    assert pbbc.hits[0].doc_id == "base:P-0025"


def test_dictionary_graph_retriever_prefers_exact_phrase_mentions_over_generic_headwords() -> None:
    docs = [
        Document(
            doc_id="base:P-0001",
            title="PHÁO",
            text="PHÁO, vũ khí bắn đạn theo đường đạn cong hoặc thẳng.",
            metadata={"kind": "dictionary", "headword": "PHÁO"},
        ),
        Document(
            doc_id="base:N-0001",
            title="NGÀY TRUYỀN THỐNG PHÁO BINH",
            text="NGÀY TRUYỀN THỐNG PHÁO BINH, ngày kỷ niệm trận đánh ở pháo đài Xuân Canh.",
            metadata={"kind": "dictionary", "headword": "NGÀY TRUYỀN THỐNG PHÁO BINH"},
        ),
        Document(
            doc_id="base:L-0001",
            title="LÁNG",
            text="LÁNG, địa danh có pháo đài Láng trong lịch sử lực lượng pháo binh.",
            metadata={"kind": "dictionary", "headword": "LÁNG"},
        ),
    ]
    retriever = DictionaryGraphRetriever()
    retriever.build(docs)

    xuan_canh = retriever.search(Query("q1", "Pháo đài Xuân Canh"), top_k=3)
    two_places = retriever.search(Query("q2", "pháo đài Láng, pháo đài Xuân Tảo"), top_k=3)

    assert xuan_canh.hits[0].doc_id == "base:N-0001"
    assert xuan_canh.hits[0].metadata["query_highlights"] == ["Pháo đài Xuân Canh"]
    assert two_places.hits[0].doc_id == "base:L-0001"
    assert "PHÁO" not in [hit.title for hit in xuan_canh.hits[:1]]
    assert two_places.hits[0].metadata["query_highlights"] == ["pháo đài Láng", "pháo đài Xuân Tảo"]


def test_dictionary_graph_retriever_keeps_stroked_d_distinct_from_plain_d() -> None:
    docs = [
        Document(
            doc_id="fort",
            title="PHÁO ĐÀI",
            text="PHÁO ĐÀI, công sự kiên cố dùng trong phòng thủ.",
            metadata={"kind": "dictionary", "headword": "PHÁO ĐÀI"},
        ),
        Document(
            doc_id="long-cannon",
            title="PHÁO DÀI",
            text="PHÁO DÀI, cách nói về nòng pháo dài.",
            metadata={"kind": "dictionary", "headword": "PHÁO DÀI"},
        ),
    ]
    retriever = DictionaryGraphRetriever()
    retriever.build(docs)

    fort = retriever.search(Query("q1", "pháo đài"), top_k=2)
    long_cannon = retriever.search(Query("q2", "pháo dài"), top_k=2)

    assert fort.hits[0].doc_id == "fort"
    assert fort.hits[0].metadata["query_highlights"] == ["pháo đài"]
    assert fort.hits[1].doc_id != "long-cannon" or "dictionary_direct_score" not in fort.hits[1].metadata
    assert long_cannon.hits[0].doc_id == "long-cannon"
    assert long_cannon.hits[0].metadata["query_highlights"] == ["pháo dài"]
    assert long_cannon.hits[1].doc_id != "fort" or "dictionary_direct_score" not in long_cannon.hits[1].metadata


def test_dictionary_graph_retriever_keeps_tone_distinct_for_vietnamese_headwords() -> None:
    docs = [
        Document(
            doc_id="japan",
            title="NHẬT",
            text="NHẬT, cách gọi tắt Nhật Bản trong một số ngữ cảnh.",
            metadata={"kind": "dictionary", "headword": "NHẬT"},
        ),
        Document(
            doc_id="first",
            title="NHẤT",
            text="NHẤT, thứ nhất hoặc mức cao nhất trong một thang phân loại.",
            metadata={"kind": "dictionary", "headword": "NHẤT"},
        ),
    ]
    retriever = DictionaryGraphRetriever()
    retriever.build(docs)

    japan = retriever.search(Query("q1", "nhật"), top_k=2)
    first = retriever.search(Query("q2", "nhất"), top_k=2)

    assert japan.hits[0].doc_id == "japan"
    assert japan.hits[0].metadata["dictionary_match_mode"] == "strict"
    assert japan.hits[0].metadata["query_highlights"] == ["nhật"]
    assert japan.hits[1].doc_id != "first" or "dictionary_direct_score" not in japan.hits[1].metadata
    assert first.hits[0].doc_id == "first"
    assert first.hits[0].metadata["dictionary_match_mode"] == "strict"
    assert first.hits[0].metadata["query_highlights"] == ["nhất"]
    assert first.hits[1].doc_id != "japan" or "dictionary_direct_score" not in first.hits[1].metadata


def test_dictionary_graph_retriever_uses_strict_tone_for_partial_vietnamese_headwords() -> None:
    docs = [
        Document(
            doc_id="japan",
            title="NHẬT BẢN",
            text="NHẬT BẢN, quốc gia ở Đông Á.",
            metadata={"kind": "dictionary", "headword": "NHẬT BẢN"},
        ),
        Document(
            doc_id="first",
            title="“BA NHẤT\"",
            text="BA NHẤT, phong trào thi đua trong huấn luyện.",
            metadata={"kind": "dictionary", "headword": "“BA NHẤT\""},
        ),
    ]
    retriever = DictionaryGraphRetriever()
    retriever.build(docs)

    first = retriever.search(Query("q1", "nhất"), top_k=2)
    japan = retriever.search(Query("q2", "nhật"), top_k=2)

    assert first.hits[0].doc_id == "first"
    assert first.hits[0].metadata["dictionary_match_mode"] == "strict"
    assert first.hits[0].metadata["query_highlights"] == ["nhất"]
    assert first.hits[1].doc_id != "japan" or "dictionary_direct_score" not in first.hits[1].metadata
    assert japan.hits[0].doc_id == "japan"
    assert japan.hits[0].metadata["dictionary_match_mode"] == "strict"
    assert japan.hits[0].metadata["query_highlights"] == ["nhật"]
    assert japan.hits[1].doc_id != "first" or "dictionary_direct_score" not in japan.hits[1].metadata


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
