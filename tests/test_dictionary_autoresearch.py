from __future__ import annotations

import json
from pathlib import Path

import pytest

from rag_bench.cli import main
from rag_bench.dictionary import load_dictionary_documents
from rag_bench.dictionary_autoresearch import (
    AutoresearchCase,
    DictionaryAutoresearchConfig,
    build_coordinator_decision_records,
    build_coordinator_decisions_markdown,
    confirmed_failures,
    generate_autoresearch_cases,
    generate_dictionary_answer,
    judge_answer_truth,
    judge_retrieval,
    parse_answer_judgement,
    read_jsonl,
    red_generate_cases,
    run_dictionary_autoresearch,
    validate_autoresearch_config,
)
from rag_bench.groq_client import GenerationResult
from rag_bench.retrievers import DictionaryGraphRetriever
from rag_bench.types import Query, RetrievalHit


class FakeAutoresearchClient:
    def __init__(
        self,
        judge_answer: str = '{"verdict":"pass","category":"answer","reason":"ok"}',
        *,
        generation_answer: str = "PHÁO BINH là binh chủng theo nguồn [base:P-0001].",
        judge_answers: list[str] | None = None,
    ) -> None:
        self.judge_answer = judge_answer
        self.generation_answer = generation_answer
        self.judge_answers = list(judge_answers or [])
        self.calls: list[list[dict[str, str]]] = []

    def generate(self, messages, *, model=None, temperature=0.0, max_completion_tokens=512):
        self.calls.append(messages)
        if "Return JSON only" in messages[0]["content"]:
            if self.judge_answers:
                answer = self.judge_answers.pop(0)
            else:
                answer = self.judge_answer
            return GenerationResult(
                answer=answer,
                key_alias="fake",
                attempted_aliases=["fake"],
                latency_s=0.0,
                retry_count=0,
            )
        return GenerationResult(
            answer=self.generation_answer,
            key_alias="fake",
            attempted_aliases=["fake"],
            latency_s=0.0,
            retry_count=0,
        )


def test_autoresearch_jsonl_outputs_round_trip(tmp_path: Path) -> None:
    artifact = _write_autoresearch_artifact(tmp_path / "artifact")
    output_root = tmp_path / "runs"

    summary = run_dictionary_autoresearch(
        DictionaryAutoresearchConfig(
            artifact_dir=artifact,
            source_dir=None,
            output_root=output_root,
            run_name="smoke",
            limit=5,
            rounds=1,
            dry_run_model=True,
        )
    )

    run_dir = Path(summary["output_dir"])
    cases = read_jsonl(run_dir / "cases.jsonl")
    rounds = read_jsonl(run_dir / "rounds.jsonl")

    assert cases
    assert rounds
    assert (run_dir / "failures.jsonl").is_file()
    assert (run_dir / "coordinator_decisions.jsonl").is_file()
    assert (run_dir / "coordinator_decisions.md").is_file()
    assert (run_dir / "codex_session.md").is_file()
    assert (run_dir / "codex_tasks.md").is_file()
    assert "Dictionary Autoresearch Summary" in (run_dir / "summary.md").read_text(encoding="utf-8")
    assert "No confirmed failures" in (run_dir / "coordinator_decisions.md").read_text(encoding="utf-8")


def test_autoresearch_resume_skips_completed_evaluations(tmp_path: Path) -> None:
    artifact = _write_autoresearch_artifact(tmp_path / "artifact")
    output_root = tmp_path / "runs"
    dictionary = load_dictionary_documents(artifact_dir=artifact, source_dir=None, required=True)
    cases = generate_autoresearch_cases(dictionary, limit=3)
    run_dir = output_root / "resume-smoke"
    run_dir.mkdir(parents=True)
    _write_jsonl(run_dir / "cases.jsonl", [case.__dict__ for case in cases])
    _write_jsonl(
        run_dir / "rounds.jsonl",
        [
            {
                "round_index": 1,
                "case_id": cases[0].id,
                "query": cases[0].query,
                "expected_doc_id": cases[0].expected_doc_id,
                "expected_title": cases[0].expected_title,
                "retrieval": {"passed": True, "expected_rank": 1},
                "answer": {"status": "skipped"},
            }
        ],
    )

    summary = run_dictionary_autoresearch(
        DictionaryAutoresearchConfig(
            artifact_dir=artifact,
            source_dir=None,
            output_root=output_root,
            run_name="resume-smoke",
            limit=3,
            rounds=1,
            dry_run_model=True,
            resume=True,
        )
    )

    rows = read_jsonl(run_dir / "rounds.jsonl")
    assert summary["evaluation_count"] == 3
    assert len(rows) == 3
    assert rows[0]["case_id"] == cases[0].id
    assert len({(row["round_index"], row["case_id"]) for row in rows}) == 3
    assert (run_dir / "summary.md").is_file()


def test_red_generator_builds_seed_and_fixture_queries(tmp_path: Path) -> None:
    artifact = _write_autoresearch_artifact(tmp_path / "artifact")
    dictionary = load_dictionary_documents(artifact_dir=artifact, source_dir=None, required=True)

    cases = generate_autoresearch_cases(dictionary, limit=20)
    queries = {case.query for case in cases}

    assert "hexogen" in queries
    assert "pb" in queries
    assert "pbbc" in queries
    assert any(case.kind == "generated-abbreviation" for case in cases)


def test_red_generator_requires_abbreviation_evidence(tmp_path: Path) -> None:
    artifact = _write_autoresearch_artifact(tmp_path / "artifact")
    dictionary = load_dictionary_documents(artifact_dir=artifact, source_dir=None, required=True)

    cases = red_generate_cases(dictionary)
    abbreviation_queries = {case.query for case in cases if case.kind == "generated-abbreviation"}

    assert "ÁS" not in abbreviation_queries
    assert "ẨK" not in abbreviation_queries
    assert "ASkk" in abbreviation_queries
    assert "ÂK" in abbreviation_queries


def test_red_generator_adds_abbreviation_collision_context_cases(tmp_path: Path) -> None:
    artifact = _write_autoresearch_artifact(tmp_path / "artifact")
    dictionary = load_dictionary_documents(artifact_dir=artifact, source_dir=None, required=True)

    cases = red_generate_cases(dictionary)
    collision_cases = [
        case
        for case in cases
        if case.kind == "adversarial-abbreviation-context" and case.metadata.get("abbreviation_key") == "bdc"
    ]

    assert collision_cases
    assert {case.expected_title for case in collision_cases} >= {"BÀN ĐẾ CỐI", "BÙI ĐÌNH CƯ"}
    assert all(case.metadata.get("required_rank") == 1 for case in collision_cases)


def test_red_generator_chooses_distinctive_definition_phrases(tmp_path: Path) -> None:
    artifact = _write_autoresearch_artifact(tmp_path / "artifact")
    dictionary = load_dictionary_documents(artifact_dir=artifact, source_dir=None, required=True)

    cases = red_generate_cases(dictionary)
    b72_phrases = [
        case.query
        for case in cases
        if case.expected_title == "B-72" and case.kind == "definition-phrase"
    ]

    assert b72_phrases
    assert "tổ hợp tên lửa chống" not in b72_phrases
    assert any("9K" in phrase or "Maliutca" in phrase for phrase in b72_phrases)


def test_feedback_run_dirs_prioritize_adaptive_cases(tmp_path: Path) -> None:
    artifact = _write_autoresearch_artifact(tmp_path / "artifact")
    feedback = tmp_path / "feedback"
    feedback.mkdir()
    _write_jsonl(
        feedback / "failures.jsonl",
        [
            {
                "id": "failure-1",
                "category": "retrieval",
                "expected_doc_id": "base:P-0002",
                "expected_title": "PHÁO BINH BIÊN CHẾ",
                "queries": ["PBBC tricky"],
            }
        ],
    )
    _write_jsonl(
        feedback / "rounds.jsonl",
        [
            {
                "round_index": 1,
                "case_id": "case-a",
                "query": "độ ẩm bị trượt",
                "expected_doc_id": "base:A-0011",
                "expected_title": "ẨM KẾ",
                "retrieval": {"passed": True, "expected_rank": 4},
                "answer": {"status": "pass"},
            },
            {
                "round_index": 1,
                "case_id": "case-b",
                "query": "khí tài khí tượng",
                "expected_doc_id": "base:A-0011",
                "expected_title": "ẨM KẾ",
                "retrieval": {"passed": True, "expected_rank": 1},
                "answer": {"status": "fail"},
                "failure_category": "answer",
                "failure_reason": "empty answer",
            },
        ],
    )
    dictionary = load_dictionary_documents(artifact_dir=artifact, source_dir=None, required=True)

    cases = generate_autoresearch_cases(dictionary, limit=60, feedback_run_dirs=(feedback,))

    assert any(case.kind == "feedback-confirmed-retrieval" and case.query == "PBBC tricky" for case in cases)
    assert any(case.kind == "feedback-near-miss" and case.query == "độ ẩm bị trượt" for case in cases)
    assert any(case.kind == "feedback-candidate-answer" and case.query == "khí tài khí tượng" for case in cases)
    assert any(case.kind == "feedback-distinctive-phrase" and case.expected_title == "ẨM KẾ" for case in cases)


def test_feedback_rejected_decisions_are_not_replayed(tmp_path: Path) -> None:
    artifact = _write_autoresearch_artifact(tmp_path / "artifact")
    feedback = tmp_path / "feedback"
    feedback.mkdir()
    _write_jsonl(
        feedback / "failures.jsonl",
        [
            {
                "id": "failure-rejected",
                "category": "retrieval",
                "expected_doc_id": "base:P-0002",
                "expected_title": "PHÁO BINH BIÊN CHẾ",
                "queries": ["synthetic rejected alias"],
            }
        ],
    )
    _write_jsonl(
        feedback / "coordinator_decisions.jsonl",
        [{"failure_id": "failure-rejected", "status": "rejected"}],
    )
    dictionary = load_dictionary_documents(artifact_dir=artifact, source_dir=None, required=True)

    cases = generate_autoresearch_cases(dictionary, limit=20, feedback_run_dirs=(feedback,))

    assert not any(case.query == "synthetic rejected alias" for case in cases)


def test_retrieval_judge_distinguishes_hit_and_miss(tmp_path: Path) -> None:
    artifact = _write_autoresearch_artifact(tmp_path / "artifact")
    dictionary = load_dictionary_documents(artifact_dir=artifact, source_dir=None, required=True)
    retriever = DictionaryGraphRetriever()
    retriever.build(dictionary.documents)
    hit_case = next(case for case in generate_autoresearch_cases(dictionary, limit=10) if case.query == "pb")
    miss_case = AutoresearchCase(
        id="case-miss",
        query="pháo binh",
        expected_doc_id="missing",
        expected_title="MISSING",
        reference_snippet="missing",
        kind="fixture",
        source="test",
    )

    hit_result = retriever.search(Query("hit", hit_case.query), top_k=5)
    miss_result = retriever.search(Query("miss", miss_case.query), top_k=5)

    assert judge_retrieval(hit_case, hit_result.hits).passed is True
    miss = judge_retrieval(miss_case, miss_result.hits)
    assert miss.passed is False
    assert miss.category == "retrieval"


def test_retrieval_judge_requires_top_rank_for_abbreviation_cases() -> None:
    case = AutoresearchCase(
        id="case-bdc",
        query="BĐC",
        expected_doc_id="base:B-0011",
        expected_title="BÀN ĐẾ CỐI",
        reference_snippet="BÀN ĐẾ CỐI, bộ phận của cối.",
        kind="generated-abbreviation",
        source="test",
    )
    hits = [
        RetrievalHit(doc_id="base:B-0158", score=1.0, rank=1, title="BÙI ĐÌNH CƯ", text="", metadata={}),
        RetrievalHit(doc_id="base:B-0011", score=0.9, rank=2, title="BÀN ĐẾ CỐI", text="", metadata={}),
    ]

    strict = judge_retrieval(case, hits)
    soft = judge_retrieval(case, hits, strict_acronym_rank=False)

    assert strict.passed is False
    assert "required top #1" in strict.reason
    assert soft.passed is True


def test_answer_judge_wrapper_uses_json_verdict() -> None:
    client = FakeAutoresearchClient('{"verdict":"fail","category":"citation","reason":"missing citation"}')
    judgement = judge_answer_truth(
        AutoresearchCase(
            id="case-1",
            query="pb",
            expected_doc_id="base:P-0001",
            expected_title="PHÁO BINH",
            reference_snippet="PHÁO BINH, binh chủng.",
            kind="fixture",
            source="test",
        ),
        "Pháo binh là binh chủng.",
        [],
        config=DictionaryAutoresearchConfig(dry_run_model=False),
        generation_client=client,
    )

    assert judgement.passed is False
    assert judgement.category == "citation"
    assert parse_answer_judgement("not json").status == "ambiguous"


def test_autoresearch_dictionary_answer_uses_user_facing_formatter() -> None:
    case = AutoresearchCase(
        id="case-amonit",
        query="AMONIT",
        expected_doc_id="base:A-0002",
        expected_title="AMONIT",
        reference_snippet="AMONIT, thuốc nổ phá.",
        kind="exact-headword",
        source="test",
    )
    hit = RetrievalHit(
        doc_id="base:A-0002",
        score=1.0,
        rank=1,
        title="AMONIT",
        text="AMONIT, thuốc nổ phá.",
        metadata={"raw_docx_text": "AMONIT, thuốc nổ phá."},
    )
    answer = generate_dictionary_answer(
        case,
        [hit],
        config=DictionaryAutoresearchConfig(dry_run_model=False),
        generation_client=FakeAutoresearchClient(generation_answer=""),
    )

    assert answer.startswith("Mục từ gốc [base:A-0002]:")
    assert "AMONIT, thuốc nổ phá." in answer
    assert "Giải thích:" not in answer


def test_answer_judge_accepts_raw_entry_and_rejects_wrong_primary_entry() -> None:
    case = AutoresearchCase(
        id="case-amonit",
        query="AMONIT",
        expected_doc_id="base:A-0002",
        expected_title="AMONIT",
        reference_snippet="AMONIT, thuốc nổ phá.",
        kind="exact-headword",
        source="test",
    )
    correct = judge_answer_truth(
        case,
        "Mục từ gốc [base:A-0002]:\n\nAMONIT, thuốc nổ phá.",
        [],
        config=DictionaryAutoresearchConfig(dry_run_model=False),
        generation_client=FakeAutoresearchClient('{"verdict":"fail","category":"answer","reason":"should not call"}'),
    )
    wrong = judge_answer_truth(
        case,
        "Mục từ gốc [base:A-0007]:\n\nÁP KẾ, khí tài đo áp suất.",
        [],
        config=DictionaryAutoresearchConfig(dry_run_model=False),
        generation_client=FakeAutoresearchClient('{"verdict":"pass","category":"answer","reason":"should not call"}'),
    )

    assert correct.passed is True
    assert correct.json_retry_count == 0
    assert wrong.passed is False
    assert "expected base:A-0002" in wrong.reason


def test_answer_judge_retries_until_valid_json() -> None:
    client = FakeAutoresearchClient(
        judge_answers=[
            "not json",
            '{"verdict":"pass","category":"answer","reason":"valid on retry"}',
        ]
    )
    judgement = judge_answer_truth(
        AutoresearchCase(
            id="case-1",
            query="pb",
            expected_doc_id="base:P-0001",
            expected_title="PHÁO BINH",
            reference_snippet="PHÁO BINH, binh chủng.",
            kind="fixture",
            source="test",
        ),
        "Pháo binh là binh chủng.",
        [],
        config=DictionaryAutoresearchConfig(dry_run_model=False, judge_json_retries=2),
        generation_client=client,
    )

    assert judgement.passed is True
    assert judgement.json_retry_count == 1
    assert len(client.calls) == 2
    assert "previous response was not valid JSON" in client.calls[1][-1]["content"]


def test_privacy_guard_blocks_cloud_provider_for_private_source() -> None:
    with pytest.raises(ValueError, match="requires --provider local"):
        validate_autoresearch_config(
            DictionaryAutoresearchConfig(source_classification="private", provider="mimo", dry_run_model=True)
        )

    with pytest.raises(ValueError, match="requires --trusted-model"):
        validate_autoresearch_config(
            DictionaryAutoresearchConfig(source_classification="private", provider="local", model="local-model")
        )

    validate_autoresearch_config(
        DictionaryAutoresearchConfig(
            source_classification="private",
            provider="local",
            model="local-model",
            trusted_models=("local-model",),
            dry_run_model=True,
        )
    )
    validate_autoresearch_config(DictionaryAutoresearchConfig(source_classification="semi-private", provider="mimo"))


def test_confirmed_failures_require_repeated_or_similar_cases() -> None:
    case = AutoresearchCase(
        id="case-1",
        query="pb",
        expected_doc_id="base:P-0001",
        expected_title="PHÁO BINH",
        reference_snippet="PHÁO BINH, binh chủng.",
        kind="abbreviation",
        source="test",
    )
    evaluation = _failed_evaluation(case.id, case.query)

    assert confirmed_failures([(case, evaluation)], confirmations=2) == []
    failures = confirmed_failures([(case, evaluation), (case, evaluation)], confirmations=2)
    assert len(failures) == 1
    assert failures[0].category == "retrieval"
    assert "src/rag_bench/retrievers.py" in failures[0].suggested_files


def test_coordinator_decisions_start_pending_for_confirmed_failures() -> None:
    case = AutoresearchCase(
        id="case-1",
        query="pb",
        expected_doc_id="base:P-0001",
        expected_title="PHÁO BINH",
        reference_snippet="PHÁO BINH, binh chủng.",
        kind="abbreviation",
        source="test",
    )
    failures = confirmed_failures(
        [(case, _failed_evaluation(case.id, case.query)), (case, _failed_evaluation(case.id, case.query))],
        confirmations=2,
    )

    decisions = build_coordinator_decision_records(run_dir=Path("runs/autoresearch/test"), failures=failures)
    markdown = build_coordinator_decisions_markdown(decisions)

    assert len(decisions) == 1
    assert decisions[0].status == "pending"
    assert "accepted" in decisions[0].status_options
    assert "failures.jsonl" in decisions[0].evidence_files[0]
    assert "Codex coordinator owns this file" in markdown
    assert decisions[0].failure_id in markdown


def test_autoresearch_cli_dry_run_smoke(tmp_path: Path) -> None:
    artifact = _write_autoresearch_artifact(tmp_path / "artifact")
    output_root = tmp_path / "runs"

    code = main(
        [
            "autoresearch-dictionary",
            "--artifact-dir",
            str(artifact),
            "--source-dir",
            str(tmp_path / "missing-source"),
            "--output-root",
            str(output_root),
            "--run-name",
            "cli-smoke",
            "--limit",
            "5",
            "--rounds",
            "1",
            "--dry-run-model",
            "--quiet",
        ]
    )

    assert code == 0
    assert (output_root / "cli-smoke" / "cases.jsonl").is_file()


def _failed_evaluation(case_id: str, query: str):
    from rag_bench.dictionary_autoresearch import CaseEvaluation

    return CaseEvaluation(
        round_index=1,
        case_id=case_id,
        query=query,
        expected_doc_id="base:P-0001",
        expected_title="PHÁO BINH",
        retrieval={
            "passed": False,
            "category": "retrieval",
            "reason": "expected dictionary entry missing from top-k",
            "expected_rank": None,
            "observed_top_ids": [],
            "observed_top_titles": [],
        },
        answer={"status": "skipped"},
        failure_category="retrieval",
        failure_reason="expected dictionary entry missing from top-k",
    )


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text("\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + "\n", encoding="utf-8")


def _write_autoresearch_artifact(path: Path) -> Path:
    path.mkdir()
    entries = [
        ("base:H-0011", "H", "HEXOGEN", "HEXOGEN, thuốc nổ mạnh dùng trong kỹ thuật quân sự."),
        ("base:P-0001", "P", "PHÁO BINH", "PHÁO BINH, binh chủng hỏa lực của lục quân."),
        ("base:P-0002", "P", "PHÁO BINH BIÊN CHẾ", "PHÁO BINH BIÊN CHẾ, pháo binh thuộc biên chế đơn vị."),
        ("base:P-0003", "P", "PHÁO ĐÀI", "PHÁO ĐÀI, công trình phòng thủ có hỏa lực pháo."),
        ("base:P-0004", "P", "PHÁO DÀI", "PHÁO DÀI, mục giả để kiểm tra phân biệt dấu."),
        ("base:N-0001", "N", "NHẬT", "NHẬT, mặt trời hoặc ngày trong một số thuật ngữ."),
        ("base:N-0002", "N", "NHẤT", "NHẤT, thứ nhất hoặc mức cao nhất."),
        ("base:B-0011", "B", "BÀN ĐẾ CỐI", "BÀN ĐẾ CỐI, bộ phận của cối dùng làm bệ tì. BĐC có thể tháo rời súng để mang vác."),
        ("base:B-0158", "B", "BÙI ĐÌNH CƯ", "BÙI ĐÌNH CƯ (Bùi Văn Mười), Ah LLVTND. BĐC dũng cảm dùng đèn pin chiếu sáng mục tiêu."),
        ("base:A-0008", "A", "ÁP SUẤT", "ÁP SUẤT không khí, đại lượng vật lí. ASkk tại mỗi điểm bằng trọng lượng cột không khí."),
        ("base:A-0011", "A", "ẨM KẾ", "ẨM KẾ, khí tài khí tượng thường dùng để đo độ ẩm không khí. Trong ÂK sử dụng các chất liệu nhạy."),
        ("base:B-0001", "B", "B-72", "B-72, tổ hợp tên lửa chống tăng có điều khiển kiểu 9K-11 biệt danh Maliutca của Liên Xô."),
        ("base:T-0196", "T", "TỔ HỢP TÊN LỬA CHỐNG TĂNG", "TỔ HỢP TÊN LỬA CHỐNG TĂNG, hệ thống vũ khí tên lửa dùng để diệt xe tăng."),
    ]
    (path / "rich_entries.jsonl").write_text(
        "\n".join(
            json.dumps(
                {
                    "id": doc_id,
                    "letter": letter,
                    "source_file": f"{letter}.docx",
                    "paragraph_index": index,
                    "headword": title,
                    "plain_text": text,
                    "raw_docx_text": text,
                    "rich_blocks": [{"type": "paragraph", "runs": [{"text": text}]}],
                    "schema_version": 2,
                },
                ensure_ascii=False,
            )
            for index, (doc_id, letter, title, text) in enumerate(entries, start=1)
        )
        + "\n",
        encoding="utf-8",
    )
    (path / "nodes.jsonl").write_text(
        "\n".join(
            [
                json.dumps({"id": "alias:pb", "type": "alias", "label": "PB"}, ensure_ascii=False),
                json.dumps({"id": "alias:pbbc", "type": "alias", "label": "PBBC"}, ensure_ascii=False),
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    (path / "edges.jsonl").write_text(
        "\n".join(
            [
                json.dumps({"source": "base:P-0001", "target": "alias:pb", "type": "has_alias", "source_entry_id": "base:P-0001"}, ensure_ascii=False),
                json.dumps(
                    {
                        "source": "base:P-0002",
                        "target": "alias:pbbc",
                        "type": "has_alias",
                        "source_entry_id": "base:P-0002",
                    },
                    ensure_ascii=False,
                ),
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    return path
