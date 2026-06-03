from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

from rag_bench.groq_client import GenerationResult
from rag_bench.hotpotqa_cached_eval import HotpotqaCachedEvalConfig


ROOT = Path(__file__).resolve().parents[1]


def _load_script(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


retry_script = _load_script(
    "run_hotpotqa_retry_failed_rows",
    ROOT / "scripts" / "run_hotpotqa_retry_failed_rows.py",
)


class FakeRetryClient:
    def generate(self, *_args, **_kwargs) -> GenerationResult:
        return GenerationResult(
            answer="cats purr",
            key_alias="primary",
            attempted_aliases=["primary"],
            latency_s=0.01,
            retry_count=0,
            prompt_tokens=10,
            completion_tokens=3,
            total_tokens=13,
            estimated_tokens=20,
        )


def test_retry_failed_rows_merges_success_and_retried_rows(tmp_path: Path) -> None:
    original = tmp_path / "original"
    original.mkdir()
    base_row = {
        "experiment": {"action_id": "legacy__4000", "context_policy": "legacy", "context_budget_chars": 4000},
        "run_id": "source",
        "benchmark": "hotpotqa",
        "dataset_id": "beir/hotpotqa/test",
        "retriever": "bm25",
        "action_id": "legacy__4000",
        "context_policy": "legacy",
        "context_budget_chars": 4000,
        "adaptive_profile": None,
        "question": "What do cats do?",
        "reference_answers": ["cats purr"],
        "top_k": 1,
        "retrieved": [{"doc_id": "cat-doc", "rank": 1, "score": 1.0, "title": "Cats", "text": "Cats purr."}],
        "retrieval_metrics": {"recall@k": 1.0, "ndcg@k": 1.0, "retrieval_latency_s": 0.0},
        "retrieval_metadata": {},
        "context_budget": {"kept_context_chars": 10, "kept_context_est_tokens": 3, "context_compression_ratio": 1.0},
        "kv_estimate": None,
        "generation_skipped": False,
        "generation": {"provider": "groq", "model": "qwen/qwen3-32b", "model_role": "stronger-baseline"},
    }
    success_row = {
        **base_row,
        "query_id": "q1",
        "answer": "cats purr",
        "answer_latency_s": 0.01,
        "key_alias": "primary",
        "error": None,
        "error_status_code": None,
        "exact_match": 1.0,
        "token_f1": 1.0,
    }
    failed_row = {
        **base_row,
        "query_id": "q2",
        "answer": "",
        "answer_latency_s": 0.01,
        "key_alias": None,
        "error": "status=429 RateLimitError",
        "error_status_code": 429,
        "exact_match": 0.0,
        "token_f1": 0.0,
    }
    (original / "query_results.jsonl").write_text(
        "\n".join(json.dumps(row) for row in [success_row, failed_row]) + "\n",
        encoding="utf-8",
    )
    (original / "retrieval_cache.jsonl").write_text("{}\n", encoding="utf-8")
    (original / "metrics.json").write_text(
        json.dumps(
            {
                "run_id": "source",
                "benchmark": {"query_count": 2, "reference_join_count": 2},
                "actions": [{"action_id": "legacy__4000"}],
            }
        ),
        encoding="utf-8",
    )

    summary = retry_script.retry_failed_rows(
        original_run_dir=original,
        output_dir=tmp_path / "out",
        run_name="retry",
        config=HotpotqaCachedEvalConfig(
            provider="groq",
            model="qwen/qwen3-32b",
            model_role="stronger-baseline",
            skip_ragas=True,
        ),
        failed_status_code=429,
        max_failed_rows=None,
        include_non_status_errors=False,
        run_ragas=False,
        llm_factory=FakeRetryClient,
    )

    run_dir = Path(summary["output_dir"])
    rows = [json.loads(line) for line in (run_dir / "query_results.jsonl").read_text(encoding="utf-8").splitlines()]

    assert summary["retry"]["candidate_count"] == 1
    assert summary["retry"]["retry_success_count"] == 1
    assert [row["query_id"] for row in rows] == ["q1", "q2"]
    assert rows[0]["answer"] == "cats purr"
    assert rows[1]["answer"] == "cats purr"
    assert rows[1]["error"] is None
    assert (run_dir / "retry_rows.jsonl").exists()
