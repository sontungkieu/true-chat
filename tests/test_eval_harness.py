from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

from rag_bench.chat_service import ChatProxyConfig, RagChatService
from rag_bench.eval_harness import (
    RagEvalConfig,
    RagEvalItem,
    compute_heuristic_scores,
    evaluate_rag_item,
    load_rag_eval_items,
    run_rag_eval,
)
from rag_bench.groq_client import GenerationResult
from rag_bench.structured_evidence import StructuredEvidenceDoc, StructuredEvidenceIndex
from rag_bench.types import BenchmarkData, Query, RetrievalHit, RetrievalResult


@dataclass
class FakeJudge:
    calls: int = 0
    messages: list[list[dict[str, str]]] = field(default_factory=list)

    def generate(self, messages, *, model=None, temperature=0.0, max_completion_tokens=512):
        self.calls += 1
        self.messages.append(messages)
        return GenerationResult(
            answer=json.dumps(
                {
                    "answer_correctness": 1.0,
                    "groundedness": 1.0,
                    "citation_support": 1.0,
                    "missing_evidence_behavior": 1.0,
                    "planner_success": 1.0,
                    "privacy_safety": 1.0,
                    "overall": 1.0,
                    "issues": [],
                    "verdict": "pass",
                }
            ),
            key_alias="judge",
            attempted_aliases=["judge"],
            latency_s=0.01,
            retry_count=0,
        )


class FakeEvalService:
    def answer(self, messages, **kwargs):
        from rag_bench.chat_service import ChatServiceResult

        tier = messages[-1].get("data_tier", "public")
        query = messages[-1]["content"]
        has_procedure = "TERM_Z" not in query
        doc_id = "PROC_A" if has_procedure else "DICT_Z"
        hit = RetrievalHit(
            doc_id=doc_id,
            score=1.0,
            rank=1,
            title=doc_id,
            text=f"{tier} synthetic evidence for {doc_id}",
            metadata={
                "data_tier": tier,
                "structured_evidence": has_procedure,
                "structured_doc_type": "procedure" if has_procedure else None,
            },
            data_tier=tier,
            doc_type="procedure" if has_procedure else "dictionary",
        )
        query_plan = {
            "intent": "procedure",
            "schema_gaps": [] if has_procedure else ["procedure_schema_not_implemented"],
            "structured_evidence": {
                "matched_doc_types": ["procedure"] if has_procedure else [],
                "matched_doc_count": 1 if has_procedure else 0,
            },
        }
        response = {
            "choices": [{"message": {"content": f"Synthetic answer [{doc_id}]"}}],
            "query_plan": query_plan,
            "rag": {
                "retrieved": [{"doc_id": doc_id, "metadata": hit.metadata}],
                "retrieval_metadata": {"query_plan": query_plan},
            },
            "privacy": {
                "session_taint": tier,
                "turn_tier": tier,
                "external_blocked": False,
                "provider_allowed": True,
            },
        }
        return ChatServiceResult(
            response=response,
            generation=GenerationResult(
                answer=f"Synthetic answer [{doc_id}]",
                key_alias="generator",
                attempted_aliases=["generator"],
                latency_s=0.01,
                retry_count=0,
            ),
            hits=[hit],
            retrieval_latency_s=0.01,
            retrieval_metadata={"query_plan": query_plan},
        )


def _write_eval_set(path: Path, rows: list[dict]) -> Path:
    path.write_text("\n".join(json.dumps(row) for row in rows), encoding="utf-8")
    return path


def _config(tmp_path: Path, eval_set: Path, **overrides) -> RagEvalConfig:
    values = {
        "eval_set": eval_set,
        "out_dir": tmp_path / "out",
        "generator_provider": "local_small",
        "generator_model": "small-generator",
        "judge_provider": None,
        "judge_model": None,
        "disable_llm_judge": True,
    }
    values.update(overrides)
    return RagEvalConfig(**values)


def test_rag_eval_item_jsonl_loading(tmp_path: Path) -> None:
    path = _write_eval_set(
        tmp_path / "eval.jsonl",
        [
            {
                "eval_id": "proc_public_001",
                "query": "quy trình xử lý TERM_A là gì",
                "mode": "dictionary",
                "data_tier": "public",
                "expected_intent": "procedure",
                "expected_doc_ids": ["PROC_A"],
                "forbidden_schema_gaps": ["procedure_schema_not_implemented"],
            }
        ],
    )

    items = load_rag_eval_items(path)

    assert len(items) == 1
    assert items[0].eval_id == "proc_public_001"
    assert items[0].expected_doc_ids == ["PROC_A"]
    assert items[0].data_tier == "public"


def test_heuristic_only_eval_writes_outputs(tmp_path: Path) -> None:
    eval_set = _write_eval_set(
        tmp_path / "eval.jsonl",
        [
            {
                "eval_id": "proc_public_001",
                "query": "quy trình xử lý TERM_A là gì",
                "expected_intent": "procedure",
                "expected_doc_ids": ["PROC_A"],
                "expected_structured_doc_types": ["procedure"],
                "forbidden_schema_gaps": ["procedure_schema_not_implemented"],
            }
        ],
    )

    summary = run_rag_eval(_config(tmp_path, eval_set), service=FakeEvalService())

    assert summary["item_count"] == 1
    assert summary["judge_called_count"] == 0
    assert Path(summary["results_path"]).exists()
    assert Path(summary["summary_path"]).read_text(encoding="utf-8").startswith("# RAG Generator/Judge Eval Summary")
    result = json.loads(Path(summary["results_path"]).read_text(encoding="utf-8").splitlines()[0])
    assert result["judge_skipped"] is True
    assert result["heuristic_scores"]["all_required_passed"] is True


def test_public_external_judge_allowed_when_configured(tmp_path: Path) -> None:
    eval_set = _write_eval_set(tmp_path / "eval.jsonl", [{"eval_id": "public", "query": "TERM_A", "data_tier": "public"}])
    judge = FakeJudge()

    result = evaluate_rag_item(
        load_rag_eval_items(eval_set)[0],
        _config(
            tmp_path,
            eval_set,
            judge_provider="mimo",
            judge_model="mimo-v2.5",
            judge_backend_kind="external_saas",
            allow_external_judge_public=True,
            disable_llm_judge=False,
        ),
        service=FakeEvalService(),
        judge_client=judge,
    )

    assert judge.calls == 1
    assert result.judge_skipped is False
    assert result.judge_scores["verdict"] == "pass"


def test_semi_private_external_judge_blocked_by_default(tmp_path: Path) -> None:
    eval_set = _write_eval_set(tmp_path / "eval.jsonl", [{"eval_id": "semi", "query": "TERM_A", "data_tier": "semi_private"}])
    judge = FakeJudge()

    result = evaluate_rag_item(
        load_rag_eval_items(eval_set)[0],
        _config(
            tmp_path,
            eval_set,
            judge_provider="mimo",
            judge_model="mimo-v2.5",
            judge_backend_kind="external_saas",
            allow_external_judge_public=True,
            disable_llm_judge=False,
        ),
        service=FakeEvalService(),
        judge_client=judge,
    )

    assert judge.calls == 0
    assert result.judge_skipped is True
    assert "semi_private" in result.judge_skip_reason


def test_semi_private_external_judge_allowed_when_configured(tmp_path: Path) -> None:
    eval_set = _write_eval_set(tmp_path / "eval.jsonl", [{"eval_id": "semi", "query": "TERM_A", "data_tier": "semi_private"}])
    judge = FakeJudge()

    result = evaluate_rag_item(
        load_rag_eval_items(eval_set)[0],
        _config(
            tmp_path,
            eval_set,
            judge_provider="mimo",
            judge_model="mimo-v2.5",
            judge_backend_kind="external_saas",
            allow_external_judge_public=True,
            allow_external_judge_semi_private=True,
            disable_llm_judge=False,
        ),
        service=FakeEvalService(),
        judge_client=judge,
    )

    assert judge.calls == 1
    assert result.judge_skipped is False


def test_private_external_judge_blocked_without_call(tmp_path: Path) -> None:
    eval_set = _write_eval_set(tmp_path / "eval.jsonl", [{"eval_id": "private", "query": "TERM_PRIVATE", "data_tier": "private"}])
    judge = FakeJudge()

    result = evaluate_rag_item(
        load_rag_eval_items(eval_set)[0],
        _config(
            tmp_path,
            eval_set,
            judge_provider="deepseek",
            judge_model="deepseek-v4-flash",
            judge_backend_kind="external_saas",
            allow_external_judge_public=True,
            allow_external_judge_semi_private=True,
            disable_llm_judge=False,
        ),
        service=FakeEvalService(),
        judge_client=judge,
    )

    assert judge.calls == 0
    assert result.judge_skipped is True
    assert result.judge_skip_reason == "private_taint_blocks_external_saas_backend"
    assert judge.messages == []


def test_private_trusted_judge_allowed(tmp_path: Path) -> None:
    eval_set = _write_eval_set(tmp_path / "eval.jsonl", [{"eval_id": "private", "query": "TERM_PRIVATE", "data_tier": "private"}])
    judge = FakeJudge()

    result = evaluate_rag_item(
        load_rag_eval_items(eval_set)[0],
        _config(
            tmp_path,
            eval_set,
            judge_provider="local",
            judge_model="trusted-judge",
            judge_backend_id="private_judge",
            judge_backend_kind="self_hosted_private",
            judge_trusted_private_backends=("private_judge",),
            judge_trusted_private_models=("trusted-judge",),
            disable_llm_judge=False,
        ),
        service=FakeEvalService(),
        judge_client=judge,
    )

    assert judge.calls == 1
    assert result.judge_skipped is False


def test_generator_and_judge_config_are_independent(tmp_path: Path) -> None:
    eval_set = _write_eval_set(tmp_path / "eval.jsonl", [{"eval_id": "public", "query": "TERM_A", "data_tier": "public"}])
    judge = FakeJudge()

    result = evaluate_rag_item(
        load_rag_eval_items(eval_set)[0],
        _config(
            tmp_path,
            eval_set,
            generator_provider="local_small",
            generator_model="tiny-generator",
            judge_provider="mimo",
            judge_model="mimo-v2.5",
            judge_backend_kind="external_saas",
            allow_external_judge_public=True,
            disable_llm_judge=False,
        ),
        service=FakeEvalService(),
        judge_client=judge,
    )

    assert result.generator_provider == "local_small"
    assert result.generator_model == "tiny-generator"
    assert result.judge_provider == "mimo"
    assert result.judge_model == "mimo-v2.5"
    assert judge.calls == 1


def test_private_external_block_does_not_serialize_judge_request(tmp_path: Path) -> None:
    eval_set = _write_eval_set(tmp_path / "eval.jsonl", [{"eval_id": "private", "query": "TERM_PRIVATE", "data_tier": "private"}])
    judge = FakeJudge()

    evaluate_rag_item(
        load_rag_eval_items(eval_set)[0],
        _config(
            tmp_path,
            eval_set,
            judge_provider="mimo",
            judge_model="mimo-v2.5",
            judge_backend_kind="external_saas",
            allow_external_judge_public=True,
            disable_llm_judge=False,
        ),
        service=FakeEvalService(),
        judge_client=judge,
    )

    assert judge.calls == 0
    assert judge.messages == []


class PublicDictionaryRetriever:
    name = "dictionary-graph"
    build_time_s = 0.0

    def search(self, query: Query, top_k: int) -> RetrievalResult:
        return RetrievalResult(
            query=query,
            hits=[
                RetrievalHit(
                    doc_id="DICT_A",
                    score=1.0,
                    rank=1,
                    title="TERM_A",
                    text="TERM_A synthetic public dictionary entry.",
                    metadata={"data_tier": "public", "headword": "TERM_A"},
                    data_tier="public",
                    doc_type="dictionary",
                )
            ],
            latency_s=0.01,
            metadata={"kind": "dictionary"},
        )


class CitationLLM:
    key_usage_counts = {}

    def generate(self, messages, *, model=None, temperature=0.0, max_completion_tokens=512):
        return GenerationResult(
            answer="Synthetic grounded answer [PROC_A]",
            key_alias="local",
            attempted_aliases=["local"],
            latency_s=0.01,
            retry_count=0,
        )

    def rate_limit_snapshot(self):
        return {}


def _real_public_service(structured_index: StructuredEvidenceIndex) -> RagChatService:
    retriever = PublicDictionaryRetriever()
    return RagChatService(
        config=ChatProxyConfig(model="small-generator", model_id="eval-test", dictionary_top_k=3),
        benchmark=BenchmarkData(name="fixture", dataset_id="fixture/test", queries=[], documents=[], qrels={}),
        retriever=retriever,
        llm=CitationLLM(),
        retrievers={"dictionary-graph": retriever},
        structured_evidence_index=structured_index,
    )


def test_public_structured_evidence_e2e_smoke(tmp_path: Path) -> None:
    eval_set = _write_eval_set(tmp_path / "eval.jsonl", [{"eval_id": "proc", "query": "quy trình xử lý TERM_A là gì"}])
    item = RagEvalItem(
        eval_id="proc",
        query="quy trình xử lý TERM_A là gì",
        expected_intent="procedure",
        expected_doc_ids=["PROC_A"],
        expected_structured_doc_types=["procedure"],
        forbidden_schema_gaps=["procedure_schema_not_implemented"],
    )
    service = _real_public_service(
        StructuredEvidenceIndex(
            [
                StructuredEvidenceDoc.from_mapping(
                    {
                        "doc_id": "PROC_A",
                        "doc_type": "procedure",
                        "data_tier": "public",
                        "linked_terms": ["TERM_A"],
                        "steps": ["STEP_A1"],
                    }
                )
            ]
        )
    )

    result = evaluate_rag_item(item, _config(tmp_path, eval_set), service=service)

    assert result.heuristic_scores["intent_match"] is True
    assert result.heuristic_scores["expected_docs_retrieved"] is True
    assert result.heuristic_scores["schema_gap_forbidden"] is True
    assert result.heuristic_scores["structured_evidence_used"] is True


def test_missing_evidence_smoke_keeps_schema_gap(tmp_path: Path) -> None:
    eval_set = _write_eval_set(tmp_path / "eval.jsonl", [{"eval_id": "gap", "query": "quy trình xử lý TERM_Z là gì"}])
    item = RagEvalItem(
        eval_id="gap",
        query="quy trình xử lý TERM_Z là gì",
        expected_intent="procedure",
        expected_schema_gaps=["procedure_schema_not_implemented"],
    )
    service = _real_public_service(StructuredEvidenceIndex([]))

    result = evaluate_rag_item(item, _config(tmp_path, eval_set), service=service)

    assert result.heuristic_scores["intent_match"] is True
    assert result.heuristic_scores["schema_gap_expected"] is True


def test_heuristic_metrics_detect_expected_fields() -> None:
    scores = compute_heuristic_scores(
        RagEvalItem(
            eval_id="proc",
            query="TERM_A",
            expected_intent="procedure",
            expected_doc_ids=["PROC_A"],
            expected_structured_doc_types=["procedure"],
            forbidden_schema_gaps=["procedure_schema_not_implemented"],
        ),
        answer="Answer [PROC_A]",
        query_plan={
            "intent": "procedure",
            "schema_gaps": [],
            "structured_evidence": {"matched_doc_types": ["procedure"], "matched_doc_count": 1},
        },
        retrieved_doc_ids=["PROC_A"],
        privacy={"external_blocked": False},
    )

    assert scores["all_required_passed"] is True
