from __future__ import annotations

import json
from pathlib import Path

from rag_bench.retrieval_context_actions import action_from_budgetrag_row
from rag_bench.rlaif_build import RlaifBuildConfig, answer_feedback_from_budgetrag_row, build_rlaif_dataset


def test_rlaif_build_writes_actions_feedback_and_summary(tmp_path: Path) -> None:
    run_dir = tmp_path / "matrix" / "run-a"
    rows = [
        _query_result_row(
            run_id="run-a",
            retriever="bm25",
            query_id="q1",
            answer="Alpha is supported.",
            exact_match=0.0,
            token_f1=0.75,
        ),
        _query_result_row(
            run_id="run-b",
            retriever="graph-bm25",
            query_id="q1",
            answer="",
            generation_skipped=True,
            exact_match=None,
            token_f1=None,
        ),
        _query_result_row(
            run_id="run-c",
            retriever="bm25",
            query_id="q2",
            answer="",
            error="rate limit",
            error_status_code=429,
            exact_match=None,
            token_f1=None,
        ),
    ]
    _write_jsonl(run_dir / "query_results.jsonl", rows)

    summary = build_rlaif_dataset(
        RlaifBuildConfig(inputs=(tmp_path / "matrix",), output_dir=tmp_path / "rlaif")
    )

    actions = _read_jsonl(tmp_path / "rlaif" / "rlaif_actions.jsonl")
    feedback = _read_jsonl(tmp_path / "rlaif" / "rlaif_feedback.jsonl")
    summary_md = (tmp_path / "rlaif" / "rlaif_feedback_summary.md").read_text(encoding="utf-8")

    assert summary["action_count"] == 3
    assert summary["feedback_count"] == 3
    assert summary["invalid_row_count"] == 0
    assert summary["feedback_provenance_counts"] == {"gold": 1, "missing": 2}
    assert summary["missing_reason_counts"] == {"generation_skipped": 1, "generation_error": 1}
    assert summary["generation_error_count"] == 1
    assert len(actions) == 3
    assert len(feedback) == 3
    assert actions[0]["action_id"] == action_from_budgetrag_row(rows[0]).action_id
    assert actions[0]["answer"] == "Alpha is supported."
    assert actions[0]["retriever"] == "bm25"
    assert actions[0]["context_metrics"]["kept_context_chars"] == 500
    assert actions[0]["latency"]["answer_latency_s"] == 0.2
    assert actions[0]["token_usage"]["prompt_tokens"] == 100
    assert actions[0]["kv_estimate"]["after_mb"] == 2.5
    assert feedback[0]["provenance"] == "gold"
    assert feedback[0]["quality_score"] == 0.75
    assert feedback[1]["provenance"] == "missing"
    assert feedback[1]["missing_reason"] == "generation_skipped"
    assert feedback[1]["quality_score"] is None
    assert feedback[2]["missing_reason"] == "generation_error"
    assert feedback[2]["ambiguous"] is True
    assert feedback[2]["metadata"]["invalid"] is True
    assert "Feedback Provenance" in summary_md


def test_rlaif_build_records_invalid_rows_without_stopping(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    rows = [
        _query_result_row(run_id="run-a", retriever="bm25", query_id="q1"),
        {"query_id": "bad-row", "benchmark": "scifact"},
    ]
    query_results_path = run_dir / "query_results.jsonl"
    _write_jsonl(query_results_path, rows)
    with query_results_path.open("a", encoding="utf-8") as handle:
        handle.write("{not-json}\n")

    summary = build_rlaif_dataset(
        RlaifBuildConfig(inputs=(run_dir / "query_results.jsonl",), output_dir=tmp_path / "out")
    )
    actions = _read_jsonl(tmp_path / "out" / "rlaif_actions.jsonl")
    feedback = _read_jsonl(tmp_path / "out" / "rlaif_feedback.jsonl")

    assert summary["action_count"] == 1
    assert summary["feedback_count"] == 1
    assert summary["invalid_row_count"] == 2
    assert summary["invalid_rows"][0]["query_id"] == "bad-row"
    assert summary["invalid_rows"][1]["error"].startswith("Invalid JSON")
    assert actions[0]["query_id"] == "q1"
    assert feedback[0]["query_id"] == "q1"


def test_answer_feedback_uses_ragas_then_ai_judge_when_gold_is_absent() -> None:
    ragas_row = _query_result_row(
        run_id="run-a",
        retriever="bm25",
        query_id="q1",
        exact_match=None,
        token_f1=None,
        ragas={"answer_relevancy": 0.6, "faithfulness": 0.8, "answer_correctness": 0.7},
    )
    mimo_row = _query_result_row(
        run_id="run-b",
        retriever="bm25",
        query_id="q2",
        exact_match=None,
        token_f1=None,
        mimo_judge={"score": 0.9, "judge_provider": "mimo", "judge_model": "mimo-v2.5"},
    )
    deepseek_row = _query_result_row(
        run_id="run-c",
        retriever="bm25",
        query_id="q3",
        exact_match=None,
        token_f1=None,
    )
    deepseek_row["answer_judge"] = {
        "quality_score": 0.8,
        "judge_provider": "deepseek",
        "judge_model": "deepseek-r1",
    }

    ragas_feedback = answer_feedback_from_budgetrag_row(
        ragas_row,
        action_id=action_from_budgetrag_row(ragas_row).action_id,
    )
    mimo_feedback = answer_feedback_from_budgetrag_row(
        mimo_row,
        action_id=action_from_budgetrag_row(mimo_row).action_id,
    )
    deepseek_feedback = answer_feedback_from_budgetrag_row(
        deepseek_row,
        action_id=action_from_budgetrag_row(deepseek_row).action_id,
    )

    assert ragas_feedback.provenance == "ragas"
    assert ragas_feedback.quality_score == 0.7
    assert mimo_feedback.provenance == "ai_judge"
    assert mimo_feedback.quality_score == 0.9
    assert mimo_feedback.judge_provider == "mimo"
    assert mimo_feedback.judge_model == "mimo-v2.5"
    assert deepseek_feedback.provenance == "ai_judge"
    assert deepseek_feedback.judge_provider == "deepseek"
    assert deepseek_feedback.judge_model == "deepseek-r1"
    assert deepseek_feedback.quality_score == 0.8


def test_multiple_runs_same_query_create_multiple_action_rows(tmp_path: Path) -> None:
    run_a = tmp_path / "matrix" / "bm25"
    run_b = tmp_path / "matrix" / "graph"
    _write_jsonl(
        run_a / "query_results.jsonl",
        [_query_result_row(run_id="run-a", retriever="bm25", query_id="q1", context_policy="evidence-aware")],
    )
    _write_jsonl(
        run_b / "query_results.jsonl",
        [_query_result_row(run_id="run-b", retriever="graph-bm25", query_id="q1", context_policy="evidence-aware")],
    )

    summary = build_rlaif_dataset(
        RlaifBuildConfig(inputs=(tmp_path / "matrix",), output_dir=tmp_path / "out")
    )
    actions = _read_jsonl(tmp_path / "out" / "rlaif_actions.jsonl")

    assert summary["action_count"] == 2
    assert len({row["action_id"] for row in actions}) == 2
    assert {row["retrieval_strategy"] for row in actions} == {"bm25", "graph-bm25"}


def test_legacy_full_context_row_without_budget_still_builds(tmp_path: Path) -> None:
    run_dir = tmp_path / "legacy"
    row = _query_result_row(
        run_id="run-legacy",
        retriever="bm25",
        query_id="q1",
        context_policy="legacy",
    )
    row["experiment"].pop("context_budget_chars")
    row["context_budget"].pop("budget_chars")
    row["context_budget"].pop("selected_budget_chars")

    _write_jsonl(run_dir / "query_results.jsonl", [row])

    summary = build_rlaif_dataset(
        RlaifBuildConfig(inputs=(run_dir,), output_dir=tmp_path / "out")
    )
    actions = _read_jsonl(tmp_path / "out" / "rlaif_actions.jsonl")

    assert summary["action_count"] == 1
    assert summary["invalid_row_count"] == 0
    assert actions[0]["context_policy"] == "legacy"
    assert actions[0]["budget_chars"] is None
    assert actions[0]["selected_budget_chars"] is None
    assert action_from_budgetrag_row(row).budget_chars is None


def _query_result_row(
    *,
    run_id: str,
    retriever: str,
    query_id: str,
    answer: str = "Alpha is supported.",
    benchmark: str = "scifact",
    context_policy: str = "evidence-aware",
    generation_model: str = "mimo-v2.5",
    exact_match: float | None = 0.0,
    token_f1: float | None = 0.5,
    generation_skipped: bool = False,
    error: str | None = None,
    error_status_code: int | None = None,
    ragas: dict | None = None,
    mimo_judge: dict | None = None,
) -> dict:
    row = {
        "run_id": run_id,
        "benchmark": benchmark,
        "dataset_id": "beir/scifact/test",
        "retriever": retriever,
        "query_id": query_id,
        "question": "What is alpha?",
        "top_k": 5,
        "experiment": {
            "run_id": run_id,
            "benchmark": benchmark,
            "dataset_id": "beir/scifact/test",
            "retriever": retriever,
            "top_k": 5,
            "context_policy": context_policy,
            "context_budget_chars": 2000,
            "generation_model": generation_model,
        },
        "retrieved": [
            {
                "doc_id": "doc-1",
                "rank": 1,
                "score": 2.0,
                "title": "Alpha",
                "text": "Alpha evidence.",
            }
        ],
        "retrieval_metrics": {"hit@k": 1.0, "ndcg@k": 1.0},
        "retrieval_metadata": {"fusion_strategy": None},
        "context_budget": {
            "requested_policy": context_policy,
            "requested_policy_impl": "lexical-query-aware",
            "policy": context_policy,
            "policy_impl": "lexical-query-aware",
            "budget_chars": 2000,
            "selected_budget_chars": 2000,
            "kept_context_chars": 500,
            "kept_context_est_tokens": 125,
        },
        "kv_estimate": {"profile": "generic-small", "after_mb": 2.5},
        "estimated_prompt_tokens_after_budget": 125,
        "estimated_prompt_tokens_saved_by_budget": 75,
        "generation_skipped": generation_skipped,
        "answer": answer,
        "answer_latency_s": 0.2,
        "total_latency_s": 0.4,
        "scheduled_wait_s": 0.0,
        "estimated_tokens": 128,
        "prompt_tokens": 100,
        "completion_tokens": 28,
        "total_tokens": 128,
        "retry_count": 0,
        "rate_limited": False,
        "error": error,
        "error_status_code": error_status_code,
        "exact_match": exact_match,
        "token_f1": token_f1,
    }
    if ragas is not None:
        row["ragas"] = ragas
    if mimo_judge is not None:
        row["mimo_judge"] = mimo_judge
    return row


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def _read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
