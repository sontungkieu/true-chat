from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

from rag_bench.groq_client import GenerationResult
from rag_bench.hotpotqa_cached_eval import (
    HotpotqaCachedEvalConfig,
    build_groq_client,
    build_reference_lookup_from_records,
    normalize_question,
    run_hotpotqa_cached_eval,
    select_ragas_sample_rows,
)
from rag_bench.ragas_eval import _row_to_ragas_sample
from rag_bench.types import BenchmarkData, Document, Query


class FakeMimo:
    def __init__(self) -> None:
        self.key_usage_counts: Counter[str] = Counter()

    def generate(self, *_args, **_kwargs) -> GenerationResult:
        self.key_usage_counts["mimo"] += 1
        return GenerationResult(
            answer="cats purr",
            key_alias="mimo",
            attempted_aliases=["mimo"],
            latency_s=0.01,
            retry_count=0,
            prompt_tokens=20,
            completion_tokens=4,
            total_tokens=24,
            estimated_tokens=30,
        )


def test_hotpotqa_reference_lookup_normalizes_questions() -> None:
    lookup = build_reference_lookup_from_records(
        [
            {"question": "Who wrote The Book?", "answer": "Ada"},
            {"question": "Who wrote the book", "answer": "duplicate"},
            {"question": "", "answer": "ignored"},
        ]
    )

    assert lookup[normalize_question("who wrote the book")] == "Ada"
    assert len(lookup) == 1


def test_ragas_sample_preserves_reference_answers() -> None:
    sample = _row_to_ragas_sample(
        {
            "question": "What do cats do?",
            "answer": "cats purr",
            "reference_answers": ["cats purr"],
            "retrieved": [{"title": "Cats", "text": "Cats purr loudly."}],
        }
    )

    assert sample["reference"] == "cats purr"
    assert sample["ground_truth"] == "cats purr"


def test_ragas_sampling_is_deterministic_and_per_action() -> None:
    rows = []
    for action_id in ("a", "b"):
        for index in range(8):
            rows.append(
                {
                    "action_id": action_id,
                    "query_id": str(index),
                    "answer": "answer",
                    "generation_skipped": False,
                    "error": None,
                }
            )
    rows.append({"action_id": "a", "query_id": "bad", "answer": "", "generation_skipped": False})

    selected_once = select_ragas_sample_rows(rows, samples_per_action=5, seed=7)
    selected_twice = select_ragas_sample_rows(rows, samples_per_action=5, seed=7)

    assert selected_once == selected_twice
    assert Counter(row["action_id"] for row in selected_once) == {"a": 5, "b": 5}
    assert all(row["query_id"] != "bad" for row in selected_once)


def test_hotpotqa_groq_provider_selects_single_key_alias(tmp_path: Path) -> None:
    key_file = tmp_path / "groq_key.env"
    key_file.write_text("primary=secret-one\nbackup=secret-two\n", encoding="utf-8")

    config = HotpotqaCachedEvalConfig(
        provider="groq",
        model="qwen/qwen3-32b",
        model_role="stronger-baseline",
        groq_keys_path=key_file,
        groq_key_alias="primary",
        key_tokens_per_minute=6000,
        key_requests_per_minute=20,
    )

    client = build_groq_client(config)

    assert client.provider_name == "Groq"
    assert client.model == "qwen/qwen3-32b"
    assert [key.alias for key in client.keys] == ["primary"]
    assert client.scheduler.tokens_per_minute == 6000
    assert client.scheduler.requests_per_minute == 20


def test_cached_eval_writes_outputs_and_answer_accuracy(tmp_path: Path) -> None:
    def fake_loader(_bench: str, *, limit: int | None, allow_large: bool) -> BenchmarkData:
        assert limit == 2
        assert allow_large is True
        return BenchmarkData(
            name="hotpotqa",
            dataset_id="beir/hotpotqa/test",
            queries=[
                Query("q1", "What do cats do?"),
                Query("q2", "What fruit is yellow?"),
            ],
            documents=[
                Document("cat-doc", "Cats purr loudly.", "Cats"),
                Document("banana-doc", "Bananas are yellow.", "Bananas"),
            ],
            qrels={"q1": {"cat-doc": 1}, "q2": {"banana-doc": 1}},
        )

    config = HotpotqaCachedEvalConfig(
        limit=2,
        top_k=1,
        output_dir=tmp_path,
        run_name="fixture",
        policies=("legacy", "adaptive-heuristic"),
        context_budgets=(200,),
        adaptive_profiles=("balanced",),
        skip_ragas=True,
        max_completion_tokens=32,
        max_context_chars=1000,
    )

    summary = run_hotpotqa_cached_eval(
        config,
        benchmark_loader=fake_loader,
        reference_lookup_loader=lambda: {
            normalize_question("What do cats do?"): "cats purr",
            normalize_question("What fruit is yellow?"): "bananas are yellow",
        },
        llm_factory=FakeMimo,
    )

    run_dir = Path(summary["output_dir"])
    rows = [json.loads(line) for line in (run_dir / "query_results.jsonl").read_text(encoding="utf-8").splitlines()]
    assert (run_dir / "retrieval_cache.jsonl").exists()
    assert (run_dir / "hotpotqa_summary.csv").exists()
    assert (run_dir / "hotpotqa_summary.md").exists()
    assert len(rows) == 4
    assert summary["benchmark"]["reference_join_count"] == 2
    assert summary["summary_rows"][0]["exact_match"] == 0.5
    assert summary["summary_rows"][0]["token_f1"] >= 0.5
