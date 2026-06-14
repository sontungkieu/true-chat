from __future__ import annotations

from rag_bench.dictionary_query_planner import (
    DEFAULT_NORMALIZATION_ADAPTER,
    DictionaryNormalizationAdapter,
    DictionaryQueryIntent,
    annotate_and_rank_dictionary_hits,
    dictionary_plan_prompt_instructions,
    dictionary_lookup_normalization_candidates,
    plan_dictionary_query,
)
from rag_bench.types import RetrievalHit


def test_dictionary_query_planner_detects_vietnamese_and_english_intents() -> None:
    cases = {
        "TERM_A là gì": DictionaryQueryIntent.DEFINITION,
        "giải thích TERM_A": DictionaryQueryIntent.DEFINITION,
        "explain TERM_A": DictionaryQueryIntent.DEFINITION,
        "tên khác của TERM_A": DictionaryQueryIntent.ALIAS,
        "TERM_A thuộc nhóm nào": DictionaryQueryIntent.CATEGORY,
        "so sánh TERM_A và TERM_B": DictionaryQueryIntent.COMPARISON,
        "TERM_A khác TERM_B như thế nào": DictionaryQueryIntent.COMPARISON,
        "TERM_A liên quan gì đến TERM_B": DictionaryQueryIntent.RELATION,
        "TERM_A dùng để làm gì": DictionaryQueryIntent.USAGE,
        "TERM_A yêu cầu gì": DictionaryQueryIntent.REQUIREMENT,
        "quy trình thực hiện TERM_A": DictionaryQueryIntent.PROCEDURE,
        "ngoại lệ của TERM_A": DictionaryQueryIntent.EXCEPTION,
        "trường hợp này áp dụng TERM_A không": DictionaryQueryIntent.RULE_APPLICATION,
        "what is TERM_A": DictionaryQueryIntent.DEFINITION,
        "alias of TERM_A": DictionaryQueryIntent.ALIAS,
        "category of TERM_A": DictionaryQueryIntent.CATEGORY,
        "difference between TERM_A and TERM_B": DictionaryQueryIntent.COMPARISON,
        "how is TERM_A related to TERM_B": DictionaryQueryIntent.RELATION,
        "what is TERM_A used for": DictionaryQueryIntent.USAGE,
        "what does TERM_A require": DictionaryQueryIntent.REQUIREMENT,
        "procedure for TERM_A": DictionaryQueryIntent.PROCEDURE,
        "exception for TERM_A": DictionaryQueryIntent.EXCEPTION,
    }

    for query, expected in cases.items():
        assert plan_dictionary_query(query).intent == expected


def test_definition_plan_strips_question_noise_for_short_acronyms() -> None:
    cases = {
        "PB": [
            "PB",
            "P B",
            "PB là gì?",
            "PB la gi?",
            "pblagi",
            "PB viết tắt cho gì?",
            "PB viet tat cho gi?",
            "PB là viết tắt của gì?",
            "PB la viet tat cua gi?",
            "pbviettatcuagi",
            "what does PB stand for?",
            "giải thích PB",
            "giai thich PB",
            "giaithichpb",
            "explain PB",
        ],
        "CVHL": [
            "CVHL",
            "CVHL là gì?",
            "CVHL la gi?",
            "giải thích CVHL",
            "giai thich CVHL",
            "CVHL nghĩa là gì?",
            "cho tôi biết CVHL là gì",
            "cvhlnghialagi",
        ],
        "SPG9": ["SPG9 là gì?", "meaning of SPG9", "what does SPG9 mean?"],
        "ĐKZ": ["ĐKZ là gì?", "giải nghĩa ĐKZ"],
        "KHCN": [
            "KHCN",
            "K H C N",
            "KHCN là gì?",
            "KHCN xuất hiện ở đâu?",
            "KHCN xuat hien o dau?",
            "K H C N xuat hien o dau?",
            "khcnxuathienodau",
        ],
        "CTCC": [
            "CTCC",
            "C T C C",
            "CTCC là gì?",
            "CTCC xuất hiện ở đâu?",
            "ctccxuathienodau",
        ],
        "QSPB": ["QSPB", "Q S P B", "QSPB là gì?", "qspblagi"],
    }

    for expected_target, queries in cases.items():
        for query in queries:
            plan = plan_dictionary_query(query)

            assert plan.intent == DictionaryQueryIntent.DEFINITION
            assert plan.target_terms == [expected_target]


def test_definition_plan_strips_plural_lookup_wrapper_for_short_phrases() -> None:
    cases = {
        "các pháo đài": "pháo đài",
        "những pháo đài": "pháo đài",
        "cac phao dai": "phao dai",
        "nhung phao dai": "phao dai",
    }

    for query, expected_target in cases.items():
        plan = plan_dictionary_query(query)

        assert plan.intent == DictionaryQueryIntent.DEFINITION
        assert plan.target_terms == [expected_target]
        assert plan.normalization["target_changed"] is True


def test_short_lookup_normalization_generalizes_across_unseen_targets() -> None:
    target_queries = {
        "AB": [
            "AB là gì?",
            "A B là gì?",
            "AB nghĩa là gì?",
            "AB viết tắt cho gì?",
            "AB xuất hiện ở đâu?",
            "vui lòng giải thích AB",
            "ablaviettatcuagi",
        ],
        "XYZ": [
            "XYZ là gì?",
            "X Y Z xuất hiện ở đâu?",
            "giải nghĩa XYZ",
            "what does XYZ stand for?",
            "xyzxuathienodau",
            "giaithichxyz",
        ],
        "T7K": [
            "T7K là gì?",
            "T 7 K nghĩa là gì?",
            "what does T7K mean?",
            "t7knghialagi",
        ],
    }

    for expected_target, queries in target_queries.items():
        for query in queries:
            plan = plan_dictionary_query(query)

            assert plan.intent == DictionaryQueryIntent.DEFINITION
            assert plan.target_terms == [expected_target]
            assert plan.normalization["target_count"] == 1


def test_compact_acronym_planner_rerank_is_stable_across_score_noise() -> None:
    plan = plan_dictionary_query("QSPB")
    hits = [
        RetrievalHit(
            doc_id="base:H-0001",
            score=5.0,
            rank=1,
            title="SYNTHETIC H",
            text="Synthetic text with a compact/spaced acronym mention.",
            metadata={"dictionary_direct_score": 0.86, "dictionary_match_mode": "lexical"},
        ),
        RetrievalHit(
            doc_id="base:C-0001",
            score=1.0,
            rank=2,
            title="SYNTHETIC C",
            text="Synthetic text with the same compact/spaced acronym evidence.",
            metadata={"dictionary_direct_score": 0.86, "dictionary_match_mode": "lexical"},
        ),
    ]

    ranked = annotate_and_rank_dictionary_hits(hits, plan, max_hits=2)

    assert ranked[0].doc_id == "base:C-0001"


def test_short_lookup_normalization_does_not_extract_acronym_from_arbitrary_sentence() -> None:
    plan = plan_dictionary_query("nội dung XYZ trong tài liệu này")

    assert plan.target_terms != ["XYZ"]
    candidates = dictionary_lookup_normalization_candidates("nội dung XYZ trong tài liệu này")
    assert not any(row["target"] == "XYZ" and row["layer"] == "short_acronym_lookup_noise" for row in candidates)


def test_short_lookup_normalization_records_layer_candidates() -> None:
    compact = plan_dictionary_query("xyzxuathienodau")
    noisy = plan_dictionary_query("vui lòng giải thích XYZ")

    assert compact.target_terms == ["XYZ"]
    assert compact.normalization["target_layer"] == "compact_lookup_affix"
    assert noisy.target_terms == ["XYZ"]
    assert noisy.normalization["target_layer"] == "short_acronym_lookup_noise"

    candidates = dictionary_lookup_normalization_candidates("vui lòng giải thích XYZ")
    assert any(
        row["adapter"] == "generic"
        and row["layer"] == "short_acronym_lookup_noise"
        and row["target"] == "XYZ"
        and row["changed"] is True
        for row in candidates
    )


def test_short_lookup_normalization_accepts_pluggable_adapter_memory() -> None:
    default_plan = plan_dictionary_query("please define XYZ")
    adapter = DictionaryNormalizationAdapter(
        name="toy-domain",
        lookup_noise_tokens=frozenset({*DEFAULT_NORMALIZATION_ADAPTER.lookup_noise_tokens, "please"}),
        compact_lookup_prefixes=DEFAULT_NORMALIZATION_ADAPTER.compact_lookup_prefixes,
        compact_lookup_suffixes=DEFAULT_NORMALIZATION_ADAPTER.compact_lookup_suffixes,
    )
    adapted_plan = plan_dictionary_query("please define XYZ", normalization_adapter=adapter)

    assert default_plan.target_terms != ["XYZ"]
    assert adapted_plan.target_terms == ["XYZ"]
    assert adapted_plan.normalization["target_adapter"] == "toy-domain"
    assert adapted_plan.normalization["target_layer"] == "short_acronym_lookup_noise"


def test_comparison_plan_has_both_terms_and_structured_answer_style() -> None:
    plan = plan_dictionary_query("so sánh TERM_A và TERM_B")

    assert plan.intent == DictionaryQueryIntent.COMPARISON
    assert plan.require_comparison is True
    assert plan.target_terms == ["TERM_A", "TERM_B"]
    assert "comparison" in plan.answer_style


def test_relation_plan_prefers_typed_edges_before_weak_related_to() -> None:
    plan = plan_dictionary_query("TERM_A liên quan gì đến TERM_B")

    assert plan.intent == DictionaryQueryIntent.RELATION
    assert plan.max_graph_hops >= 1
    assert plan.preferred_edge_types
    assert "is_a" in plan.preferred_edge_types
    assert "related_to" in plan.preferred_edge_types
    assert plan.preferred_edge_types.index("is_a") < plan.preferred_edge_types.index("related_to")


def test_alias_plan_uses_alias_direct_style_and_requires_alias_evidence() -> None:
    queries = [
        "TERM_A còn gọi là gì",
        "tên khác của TERM_A là gì",
        "alias of TERM_A",
        "synonym of TERM_A",
    ]

    for query in queries:
        plan = plan_dictionary_query(query)

        assert plan.intent == DictionaryQueryIntent.ALIAS
        assert "has_alias" in plan.preferred_edge_types
        assert plan.answer_style == "alias_direct"
        assert plan.alias_requested is True
        assert plan.requires_alias_evidence is True
        assert plan.to_payload()["requires_alias_evidence"] is True


def test_alias_prompt_instructions_are_direct_and_evidence_bound() -> None:
    plan = plan_dictionary_query("TERM_A còn gọi là gì")
    instructions = dictionary_plan_prompt_instructions(plan)

    assert "Answer with supported alternate names first." in instructions
    assert "Use only retrieved alias evidence" in instructions
    assert "Do not treat related terms, concepts, categories, or see-also references as aliases." in instructions
    assert "If no alias evidence is present" in instructions
    assert "Cite source ids." in instructions


def test_procedure_plan_marks_schema_gap_and_prompt_blocks_hallucinated_steps() -> None:
    plan = plan_dictionary_query("quy trình xử lý TERM_A là gì")
    instructions = dictionary_plan_prompt_instructions(plan)

    assert plan.intent == DictionaryQueryIntent.PROCEDURE
    assert "procedure_schema_not_implemented" in plan.schema_gaps
    assert "Do not invent steps" in instructions
    assert "procedural/rule/case evidence is not present" in instructions


def test_procedure_plan_clears_schema_gap_when_structured_evidence_exists() -> None:
    plan = plan_dictionary_query("quy trình xử lý TERM_A là gì")
    plan = plan.with_structured_evidence(
        {"enabled": True, "matched_doc_types": ["procedure"], "matched_doc_count": 1}
    )
    instructions = dictionary_plan_prompt_instructions(plan)

    assert plan.intent == DictionaryQueryIntent.PROCEDURE
    assert "procedure_schema_not_implemented" not in plan.schema_gaps
    assert plan.structured_evidence["matched_doc_count"] == 1
    assert plan.answer_style == "procedure_grounded"
    assert "Present steps only if they are supported" in instructions


def test_procedure_gap_remains_when_structured_evidence_has_no_relevant_hit() -> None:
    plan = plan_dictionary_query("quy trình xử lý TERM_A là gì")
    plan = plan.with_structured_evidence({"enabled": True, "matched_doc_types": [], "matched_doc_count": 0})
    instructions = dictionary_plan_prompt_instructions(plan)

    assert plan.intent == DictionaryQueryIntent.PROCEDURE
    assert "procedure_schema_not_implemented" in plan.schema_gaps
    assert "Do not invent steps" in instructions
    assert "procedural/rule/case evidence is not present" in instructions


def test_rule_gap_remains_when_structured_evidence_has_no_relevant_hit() -> None:
    plan = plan_dictionary_query("trường hợp này áp dụng TERM_A không")
    plan = plan.with_structured_evidence({"enabled": True, "matched_doc_types": [], "matched_doc_count": 0})
    instructions = dictionary_plan_prompt_instructions(plan)

    assert plan.intent == DictionaryQueryIntent.RULE_APPLICATION
    assert "rule_schema_not_implemented" in plan.schema_gaps
    assert "Do not invent steps, rules, exceptions, or cases." in instructions


def test_case_gap_remains_when_structured_evidence_has_no_relevant_hit() -> None:
    plan = plan_dictionary_query("case tương tự cho TERM_A là gì")
    plan = plan.with_structured_evidence({"enabled": True, "matched_doc_types": [], "matched_doc_count": 0})
    instructions = dictionary_plan_prompt_instructions(plan)

    assert plan.intent == DictionaryQueryIntent.CASE_BASED
    assert "case_schema_not_implemented" in plan.schema_gaps
    assert "Do not fabricate rules, procedures, exceptions, or cases" in instructions


def test_case_plan_uses_case_evidence_as_example_not_universal_rule() -> None:
    plan = plan_dictionary_query("case này của TERM_A")
    plan = plan.with_structured_evidence({"enabled": True, "matched_doc_types": ["case"], "matched_doc_count": 1})
    instructions = dictionary_plan_prompt_instructions(plan)

    assert plan.intent == DictionaryQueryIntent.CASE_BASED
    assert "case_schema_not_implemented" not in plan.schema_gaps
    assert "Use cases only as examples or evidence, not universal rules." in instructions


def test_planner_preserves_vietnamese_tone_marks_for_corner_case_terms() -> None:
    japan = plan_dictionary_query("nhật là gì")
    fort_compare = plan_dictionary_query("so sánh pháo đài và pháo dài")

    assert japan.target_terms == ["nhật"]
    assert fort_compare.target_terms == ["pháo đài", "pháo dài"]


def test_planner_rerank_uses_strict_vietnamese_targets_before_folded_matches() -> None:
    plan = plan_dictionary_query("nhật là gì")
    hits = [
        RetrievalHit(
            doc_id="first",
            score=1.0,
            rank=1,
            title="NHẤT",
            text="NHẤT synthetic context.",
            metadata={"headword": "NHẤT", "dictionary_match_mode": "folded", "data_tier": "public"},
            data_tier="public",
        ),
        RetrievalHit(
            doc_id="japan",
            score=0.95,
            rank=2,
            title="NHẬT",
            text="NHẬT synthetic context.",
            metadata={"headword": "NHẬT", "dictionary_match_mode": "strict", "data_tier": "public"},
            data_tier="public",
        ),
    ]

    ranked = annotate_and_rank_dictionary_hits(hits, plan, max_hits=2)

    assert ranked[0].doc_id == "japan"
    assert ranked[0].metadata["query_plan_role"] == "primary_term"


def test_alias_evidence_metadata_counts_only_alias_evidence() -> None:
    plan = plan_dictionary_query("TERM_A còn gọi là gì")
    hits = [
        RetrievalHit(
            doc_id="alias",
            score=1.0,
            rank=1,
            title="TERM_A",
            text="Synthetic alias entry.",
            metadata={
                "headword": "TERM_A",
                "aliases": ["TERM_A_ALT"],
                "data_tier": "public",
            },
            data_tier="public",
        ),
        RetrievalHit(
            doc_id="related",
            score=1.0,
            rank=2,
            title="TERM_RELATED",
            text="Synthetic related entry.",
            metadata={
                "headword": "TERM_RELATED",
                "aliases": ["RELATED_ALIAS_X"],
                "dictionary_relation": "related_to",
                "data_tier": "public",
            },
            data_tier="public",
        ),
        RetrievalHit(
            doc_id="category",
            score=1.0,
            rank=3,
            title="TERM_CATEGORY",
            text="Synthetic category entry.",
            metadata={
                "headword": "TERM_CATEGORY",
                "dictionary_relation": "in_category",
                "data_tier": "public",
            },
            data_tier="public",
        ),
    ]

    ranked = annotate_and_rank_dictionary_hits(hits, plan, max_hits=3)
    by_id = {hit.doc_id: hit for hit in ranked}

    assert by_id["alias"].metadata["has_alias_evidence"] is True
    assert by_id["alias"].metadata["alias_evidence_count"] == 1
    assert by_id["alias"].metadata["query_plan_role"] == "alias_evidence"
    assert by_id["related"].metadata["has_alias_evidence"] is False
    assert by_id["related"].metadata["alias_evidence_count"] == 0
    assert by_id["category"].metadata["has_alias_evidence"] is False
    assert by_id["category"].metadata["alias_evidence_count"] == 0


def test_alias_evidence_metadata_counts_has_alias_graph_edges_only() -> None:
    plan = plan_dictionary_query("TERM_A còn gọi là gì")
    hits = [
        RetrievalHit(
            doc_id="alias-edge",
            score=1.0,
            rank=1,
            title="TERM_A",
            text="Synthetic alias edge entry.",
            metadata={
                "headword": "TERM_A",
                "dictionary_graph_edges": [
                    {"type": "has_alias", "target_label": "ALIAS_A", "confidence": 0.95},
                    {"type": "related_to", "target_label": "RELATED_A", "confidence": 0.99},
                    {"type": "in_category", "target_label": "CATEGORY_A", "confidence": 0.99},
                ],
                "data_tier": "public",
            },
            data_tier="public",
        ),
        RetrievalHit(
            doc_id="weak-alias-edge",
            score=0.9,
            rank=2,
            title="TERM_A",
            text="Synthetic weak alias edge entry.",
            metadata={
                "headword": "TERM_A",
                "dictionary_graph_edges": [
                    {"type": "has_alias", "target_label": "WEAK_ALIAS", "confidence": 0.01},
                ],
                "data_tier": "public",
            },
            data_tier="public",
        ),
    ]

    ranked = annotate_and_rank_dictionary_hits(hits, plan, max_hits=2)
    by_id = {hit.doc_id: hit for hit in ranked}

    assert by_id["alias-edge"].metadata["has_alias_evidence"] is True
    assert by_id["alias-edge"].metadata["alias_evidence_count"] == 1
    assert by_id["weak-alias-edge"].metadata["has_alias_evidence"] is False
    assert by_id["weak-alias-edge"].metadata["alias_evidence_count"] == 0
