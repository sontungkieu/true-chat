from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

from rag_bench.groq_client import GenerationResult
from rag_bench.runner import RunConfig, _evaluate_ragas_by_retriever, run_benchmark
from rag_bench.types import BenchmarkData, Document, Query


class FakeLLM:
    def __init__(self) -> None:
        self.key_usage_counts: Counter[str] = Counter()

    def generate(self, *_args, **_kwargs) -> GenerationResult:
        self.key_usage_counts["a"] += 1
        return GenerationResult(
            answer="Cats purr [cat-doc]",
            key_alias="a",
            attempted_aliases=["a"],
            latency_s=0.01,
            retry_count=0,
            prompt_tokens=10,
            completion_tokens=4,
            total_tokens=14,
            estimated_tokens=20,
            output_tokens_per_s=400.0,
        )

    def rate_limit_snapshot(self) -> dict[str, dict[str, int]]:
        return {"a": {"tokens_used": 20, "requests_used": 1}}


class FailingLLM:
    def __init__(self) -> None:
        self.key_usage_counts: Counter[str] = Counter()

    def generate(self, *_args, **_kwargs) -> GenerationResult:
        self.key_usage_counts["a"] += 1
        return GenerationResult(
            answer="",
            key_alias=None,
            attempted_aliases=["a"],
            latency_s=0.01,
            retry_count=0,
            error="status=429 FakeRateLimitError: rate limited",
            error_status_code=429,
            rate_limited=True,
            estimated_tokens=20,
        )

    def rate_limit_snapshot(self) -> dict[str, dict[str, int]]:
        return {"a": {"tokens_used": 40, "requests_used": 2}}


def test_run_benchmark_writes_outputs_with_mocked_llm(tmp_path: Path) -> None:
    key_path = tmp_path / "groq.env"
    key_path.write_text("a=gsk_secret\n", encoding="utf-8")

    def fake_loader(_bench: str, *, limit: int | None, allow_large: bool) -> BenchmarkData:
        assert limit == 1
        assert allow_large is False
        return BenchmarkData(
            name="fixture",
            dataset_id="fixture/test",
            queries=[Query("q1", "what animal purrs?", ("cats purr",))],
            documents=[
                Document("cat-doc", "Cats purr and chase toys.", "Cats"),
                Document("banana-doc", "Bananas are yellow fruit.", "Bananas"),
            ],
            qrels={"q1": {"cat-doc": 1}},
        )

    config = RunConfig(
        bench="scifact",
        retrievers=("lexical",),
        top_k=1,
        limit=1,
        output_dir=tmp_path / "runs",
        groq_keys_path=key_path,
        model="test-model",
        vector_model="fake-vector-model",
        max_retries=0,
        max_completion_tokens=32,
        temperature=0.0,
        max_context_chars=1000,
        allow_large_bench=False,
        ragas=False,
        ragas_limit=None,
        max_consecutive_errors=3,
        skip_generation=False,
        sleep_between_queries_s=0.0,
        key_tokens_per_minute=6000,
        key_requests_per_minute=30,
        rate_limit_scope="per-key",
    )

    summary = run_benchmark(
        config,
        benchmark_loader=fake_loader,
        groq_client_factory=lambda _keys: FakeLLM(),
    )

    run_dir = Path(summary["output_dir"])
    assert (run_dir / "query_results.jsonl").exists()
    assert (run_dir / "metrics.json").exists()
    assert (run_dir / "metrics.csv").exists()
    assert summary["aggregates"][0]["retriever"] == "bm25"
    assert summary["aggregates"][0]["retrieval"]["hit@k"] == 1.0
    assert summary["aggregates"][0]["generation"]["avg_output_tokens_per_s"] == 400.0
    assert summary["key_usage_counts"] == {"a": 1}
    assert summary["key_rate_limits"]["a"]["requests_used"] == 1


def test_run_benchmark_stops_after_consecutive_generation_errors(tmp_path: Path) -> None:
    key_path = tmp_path / "groq.env"
    key_path.write_text("a=gsk_secret\n", encoding="utf-8")

    def fake_loader(_bench: str, *, limit: int | None, allow_large: bool) -> BenchmarkData:
        return BenchmarkData(
            name="fixture",
            dataset_id="fixture/test",
            queries=[
                Query("q1", "what animal purrs?"),
                Query("q2", "what fruit is yellow?"),
                Query("q3", "what animal fetches?"),
            ],
            documents=[
                Document("cat-doc", "Cats purr and chase toys.", "Cats"),
                Document("banana-doc", "Bananas are yellow fruit.", "Bananas"),
            ],
            qrels={"q1": {"cat-doc": 1}, "q2": {"banana-doc": 1}, "q3": {"cat-doc": 1}},
        )

    config = RunConfig(
        bench="scifact",
        retrievers=("bm25",),
        top_k=1,
        limit=3,
        output_dir=tmp_path / "runs",
        groq_keys_path=key_path,
        model="test-model",
        vector_model="fake-vector-model",
        max_retries=0,
        max_completion_tokens=32,
        temperature=0.0,
        max_context_chars=1000,
        allow_large_bench=False,
        ragas=False,
        ragas_limit=None,
        max_consecutive_errors=2,
        skip_generation=False,
        sleep_between_queries_s=0.0,
        key_tokens_per_minute=6000,
        key_requests_per_minute=30,
        rate_limit_scope="per-key",
    )

    summary = run_benchmark(
        config,
        benchmark_loader=fake_loader,
        groq_client_factory=lambda _keys: FailingLLM(),
    )

    assert summary["stopped_early"] is True
    assert "2 consecutive generation errors" in summary["stop_reason"]
    assert summary["aggregates"][0]["query_count"] == 2
    assert summary["key_usage_counts"] == {"a": 2}


def test_run_benchmark_can_skip_generation_without_groq_keys(tmp_path: Path) -> None:
    missing_key_path = tmp_path / "missing.env"

    def fake_loader(_bench: str, *, limit: int | None, allow_large: bool) -> BenchmarkData:
        return BenchmarkData(
            name="fixture",
            dataset_id="fixture/test",
            queries=[Query("q1", "what animal purrs?")],
            documents=[
                Document("cat-doc", "Cats purr and chase toys.", "Cats"),
                Document("banana-doc", "Bananas are yellow fruit.", "Bananas"),
            ],
            qrels={"q1": {"cat-doc": 1}},
        )

    config = RunConfig(
        bench="scifact",
        retrievers=("bm25",),
        top_k=1,
        limit=1,
        output_dir=tmp_path / "runs",
        groq_keys_path=missing_key_path,
        model="test-model",
        vector_model="fake-vector-model",
        max_retries=0,
        max_completion_tokens=32,
        temperature=0.0,
        max_context_chars=1000,
        allow_large_bench=False,
        ragas=False,
        ragas_limit=None,
        max_consecutive_errors=1,
        skip_generation=True,
        sleep_between_queries_s=0.0,
        key_tokens_per_minute=6000,
        key_requests_per_minute=30,
        rate_limit_scope="per-key",
    )

    summary = run_benchmark(config, benchmark_loader=fake_loader)

    assert summary["key_usage_counts"] == {}
    assert summary["key_rate_limits"] == {}
    assert summary["aggregates"][0]["retriever"] == "bm25"
    assert summary["aggregates"][0]["query_count"] == 1
    assert summary["experiment"]["bench"] == "scifact"
    assert summary["experiment"]["context_policy"] == "legacy"
    assert summary["experiment"]["context_budget_chars"] == 1000
    assert summary["experiment"]["skip_generation"] is True
    assert summary["aggregates"][0]["context_budget"]["context_policy"] == "legacy"
    assert summary["aggregates"][0]["experiment"]["retriever"] == "bm25"
    assert summary["aggregates"][0]["experiment"]["kv_profile"] == "generic-small"
    assert summary["aggregates"][0]["kv_estimate"]["kv_profile"] == "generic-small"
    assert summary["aggregates"][0]["generation"] == {"skipped": True, "generation_count": 0}
    rows = [
        json.loads(line)
        for line in (Path(summary["output_dir"]) / "query_results.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert rows[0]["context_budget"]["policy"] == "legacy"
    assert rows[0]["context_budget"]["retrieved_docs"] == 1
    assert rows[0]["experiment"]["bench"] == "scifact"
    assert rows[0]["experiment"]["retriever"] == "bm25"
    assert rows[0]["kv_estimate"]["profile"] == "generic-small"
    assert rows[0]["estimated_prompt_tokens_saved_by_budget"] >= 0


def test_run_benchmark_records_evidence_aware_experiment_impl(tmp_path: Path) -> None:
    def fake_loader(_bench: str, *, limit: int | None, allow_large: bool) -> BenchmarkData:
        del limit, allow_large
        return BenchmarkData(
            name="fixture",
            dataset_id="fixture/test",
            queries=[Query("q1", "alpha evidence")],
            documents=[
                Document("doc-1", "Alpha evidence sentence. Other text.", "Alpha"),
                Document("doc-2", "Unrelated text.", "Other"),
            ],
            qrels={"q1": {"doc-1": 1}},
        )

    config = RunConfig(
        bench="scifact",
        retrievers=("bm25",),
        top_k=2,
        limit=1,
        output_dir=tmp_path / "runs",
        groq_keys_path=tmp_path / "missing.env",
        model="test-model",
        vector_model="fake-vector-model",
        max_retries=0,
        max_completion_tokens=32,
        temperature=0.0,
        max_context_chars=1000,
        allow_large_bench=False,
        ragas=False,
        ragas_limit=None,
        max_consecutive_errors=1,
        skip_generation=True,
        sleep_between_queries_s=0.0,
        key_tokens_per_minute=6000,
        key_requests_per_minute=30,
        rate_limit_scope="per-key",
        context_policy="evidence-aware",
        context_budget_chars=100,
        kv_profile="qwen2.5-14b",
    )

    summary = run_benchmark(config, benchmark_loader=fake_loader)
    rows = [
        json.loads(line)
        for line in (Path(summary["output_dir"]) / "query_results.jsonl").read_text(encoding="utf-8").splitlines()
    ]

    assert summary["experiment"]["context_policy"] == "evidence-aware"
    assert summary["experiment"]["context_policy_impl"] == "lexical-query-aware"
    assert summary["aggregates"][0]["context_budget"]["context_policy_impl"] == "lexical-query-aware"
    assert rows[0]["context_budget"]["policy_impl"] == "lexical-query-aware"
    assert rows[0]["experiment"]["kv_profile"] == "qwen2.5-14b"


def test_run_benchmark_uses_groq_for_llm_retriever_when_generation_is_skipped(tmp_path: Path) -> None:
    key_path = tmp_path / "groq.env"
    key_path.write_text("a=gsk_secret\n", encoding="utf-8")

    def fake_loader(_bench: str, *, limit: int | None, allow_large: bool) -> BenchmarkData:
        return BenchmarkData(
            name="fixture",
            dataset_id="fixture/test",
            queries=[Query("q1", "what animal purrs?")],
            documents=[
                Document("cat-doc", "Cats purr and chase toys.", "Cats"),
                Document("banana-doc", "Bananas are yellow fruit.", "Bananas"),
            ],
            qrels={"q1": {"cat-doc": 1}},
        )

    config = RunConfig(
        bench="scifact",
        retrievers=("llm-multi-query",),
        top_k=1,
        limit=1,
        output_dir=tmp_path / "runs",
        groq_keys_path=key_path,
        model="test-model",
        vector_model="fake-vector-model",
        max_retries=0,
        max_completion_tokens=32,
        temperature=0.0,
        max_context_chars=1000,
        allow_large_bench=False,
        ragas=False,
        ragas_limit=None,
        max_consecutive_errors=1,
        skip_generation=True,
        sleep_between_queries_s=0.0,
        key_tokens_per_minute=6000,
        key_requests_per_minute=30,
        rate_limit_scope="per-key",
    )

    summary = run_benchmark(
        config,
        benchmark_loader=fake_loader,
        groq_client_factory=lambda _keys: FakeLLM(),
    )

    assert summary["key_usage_counts"] == {"a": 1}
    assert summary["aggregates"][0]["retriever"] == "llm-multi-query"
    assert summary["aggregates"][0]["retrieval"]["retrieval_llm_calls"] == 1.0
    assert summary["aggregates"][0]["generation"] == {"skipped": True, "generation_count": 0}


def test_ragas_evaluation_is_grouped_by_retriever() -> None:
    seen: list[tuple[list[str], int | None]] = []

    def fake_evaluator(rows, *, keys, model, limit):
        del keys, model
        seen.append(([row["query_id"] for row in rows], limit))
        return {"sample_count": min(len(rows), limit or len(rows)), "error_count": 0, "metrics": {"faithfulness": 1.0}}

    summary = _evaluate_ragas_by_retriever(
        [
            {"retriever": "bm25", "query_id": "b1", "generation_skipped": False},
            {"retriever": "bm25", "query_id": "b2", "generation_skipped": False},
            {"retriever": "vector", "query_id": "v1", "generation_skipped": False},
            {"retriever": "vector", "query_id": "v2", "generation_skipped": True},
        ],
        retrievers=["bm25", "vector"],
        keys=[],
        model="test-model",
        limit=1,
        evaluator=fake_evaluator,
    )

    assert seen == [(["b1", "b2"], 1), (["v1"], 1)]
    assert summary["mode"] == "by_retriever"
    assert summary["sample_count"] == 2
    assert set(summary["by_retriever"]) == {"bm25", "vector"}
