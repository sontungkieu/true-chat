from __future__ import annotations

import time
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

from rag_bench.benchmarks import load_benchmark
from rag_bench.groq_client import RoundRobinGroqClient
from rag_bench.io import write_csv, write_json, write_jsonl
from rag_bench.metrics import (
    aggregate_generation,
    aggregate_metric_dicts,
    exact_match,
    retrieval_metrics_for_query,
    token_f1,
)
from rag_bench.prompts import build_rag_messages
from rag_bench.retriever_registry import create_retriever, retriever_uses_llm
from rag_bench.secrets import ApiKey, load_groq_keys
from rag_bench.types import BenchmarkData


@dataclass(frozen=True)
class RunConfig:
    bench: str
    retrievers: tuple[str, ...]
    top_k: int
    limit: int | None
    output_dir: Path
    groq_keys_path: Path
    model: str
    vector_model: str
    max_retries: int
    max_completion_tokens: int
    temperature: float
    max_context_chars: int
    allow_large_bench: bool
    ragas: bool
    ragas_limit: int | None
    max_consecutive_errors: int
    skip_generation: bool
    sleep_between_queries_s: float
    key_tokens_per_minute: int
    key_requests_per_minute: int
    rate_limit_scope: str


def run_benchmark(
    config: RunConfig,
    *,
    benchmark_loader: Callable[..., BenchmarkData] = load_benchmark,
    groq_client_factory: Callable[[list[ApiKey]], RoundRobinGroqClient] | None = None,
) -> dict[str, Any]:
    if config.skip_generation and config.ragas:
        raise ValueError("skip_generation cannot be combined with ragas")

    started = time.perf_counter()
    run_id = _run_id(config)
    run_dir = config.output_dir / run_id
    uses_retrieval_llm = any(retriever_uses_llm(name) for name in config.retrievers)
    keys = [] if config.skip_generation and not config.ragas and not uses_retrieval_llm else load_groq_keys(config.groq_keys_path)
    llm = None
    if not config.skip_generation or uses_retrieval_llm:
        llm = groq_client_factory(keys) if groq_client_factory is not None else _build_groq_client(config, keys)
    data = benchmark_loader(config.bench, limit=config.limit, allow_large=config.allow_large_bench)

    all_rows: list[dict[str, Any]] = []
    aggregate_rows: list[dict[str, Any]] = []
    consecutive_generation_errors = 0
    stop_reason: str | None = None

    for retriever_name in config.retrievers:
        if stop_reason is not None:
            break
        retriever = create_retriever(
            retriever_name,
            vector_model=config.vector_model,
            query_expander=llm if retriever_uses_llm(retriever_name) else None,
            query_model=config.model,
        )
        retriever.build(data.documents)

        retrieval_metric_rows: list[dict[str, float]] = []
        generation_rows: list[dict[str, Any]] = []
        for query_index, query in enumerate(data.queries):
            if stop_reason is not None:
                break
            query_started = time.perf_counter()
            retrieval = retriever.search(query, config.top_k)
            per_query_metrics = retrieval_metrics_for_query(
                retrieval,
                data.qrels.get(query.query_id, {}),
                top_k=config.top_k,
            )
            per_query_metrics.update(_retrieval_ops_metrics(retrieval.metadata))
            retrieval_metric_rows.append(per_query_metrics)

            generation = None
            if not config.skip_generation and llm is not None:
                generation = llm.generate(
                    build_rag_messages(query, retrieval.hits, max_context_chars=config.max_context_chars),
                    temperature=config.temperature,
                    max_completion_tokens=config.max_completion_tokens,
                )
            answer = generation.answer if generation is not None else ""
            answer_em = exact_match(answer, query.reference_answers)
            answer_f1 = token_f1(answer, query.reference_answers)
            row = {
                "run_id": run_id,
                "benchmark": data.name,
                "dataset_id": data.dataset_id,
                "retriever": retriever.name,
                "query_id": query.query_id,
                "question": query.text,
                "top_k": config.top_k,
                "retrieved": [
                    {
                        "doc_id": hit.doc_id,
                        "rank": hit.rank,
                        "score": hit.score,
                        "title": hit.title,
                        "text": hit.text,
                    }
                    for hit in retrieval.hits
                ],
                "retrieval_metrics": per_query_metrics,
                "retrieval_metadata": retrieval.metadata,
                "generation_skipped": generation is None,
                "answer": answer,
                "answer_latency_s": generation.latency_s if generation is not None else None,
                "total_latency_s": time.perf_counter() - query_started,
                "key_alias": generation.key_alias if generation is not None else None,
                "attempted_aliases": generation.attempted_aliases if generation is not None else [],
                "rejected_aliases": generation.rejected_aliases if generation is not None else [],
                "retry_count": generation.retry_count if generation is not None else 0,
                "estimated_tokens": generation.estimated_tokens if generation is not None else None,
                "scheduled_wait_s": generation.scheduled_wait_s if generation is not None else 0.0,
                "prompt_tokens": generation.prompt_tokens if generation is not None else None,
                "completion_tokens": generation.completion_tokens if generation is not None else None,
                "total_tokens": generation.total_tokens if generation is not None else None,
                "output_tokens_per_s": generation.output_tokens_per_s if generation is not None else None,
                "error": generation.error if generation is not None else None,
                "error_status_code": generation.error_status_code if generation is not None else None,
                "rate_limited": generation.rate_limited if generation is not None else False,
                "exact_match": answer_em,
                "token_f1": answer_f1,
            }
            all_rows.append(row)
            if generation is None:
                consecutive_generation_errors = 0
            else:
                generation_rows.append(row)

            if generation is not None and generation.error:
                consecutive_generation_errors += 1
                if (
                    config.max_consecutive_errors > 0
                    and consecutive_generation_errors >= config.max_consecutive_errors
                ):
                    stop_reason = (
                        f"stopped after {consecutive_generation_errors} consecutive generation errors; "
                        f"last error: {generation.error}"
                    )
            else:
                consecutive_generation_errors = 0
            if (
                generation is not None
                and config.sleep_between_queries_s > 0
                and stop_reason is None
                and query_index < len(data.queries) - 1
            ):
                time.sleep(config.sleep_between_queries_s)

        aggregate = {
            "run_id": run_id,
            "benchmark": data.name,
            "dataset_id": data.dataset_id,
            "retriever": retriever.name,
            "query_count": len(retrieval_metric_rows),
            "document_count": len(data.documents),
            "top_k": config.top_k,
            "index_build_time_s": retriever.build_time_s,
            "retrieval": aggregate_metric_dicts(retrieval_metric_rows),
            "generation": (
                {"skipped": True, "generation_count": 0}
                if config.skip_generation
                else aggregate_generation(generation_rows)
            ),
        }
        aggregate_rows.append(aggregate)

    ragas_summary = None
    if config.ragas:
        from rag_bench.ragas_eval import evaluate_rows_with_ragas, filter_available_ragas_keys

        ragas_preflight = filter_available_ragas_keys(keys, model=config.model)
        ragas_summary = _evaluate_ragas_by_retriever(
            all_rows,
            retrievers=[aggregate["retriever"] for aggregate in aggregate_rows],
            keys=ragas_preflight.keys,
            model=config.model,
            limit=config.ragas_limit,
            evaluator=evaluate_rows_with_ragas,
        )
        ragas_summary["preflight_disabled_aliases"] = ragas_preflight.disabled_aliases
        ragas_summary["preflight_errors"] = ragas_preflight.errors

    summary = {
        "run_id": run_id,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "config": _serializable_config(config),
        "benchmark": {
            "name": data.name,
            "dataset_id": data.dataset_id,
            "query_count": len(data.queries),
            "document_count": len(data.documents),
        },
        "aggregates": aggregate_rows,
        "key_usage_counts": dict(Counter(llm.key_usage_counts)) if llm is not None else {},
        "key_rate_limits": llm.rate_limit_snapshot() if llm is not None else {},
        "ragas": ragas_summary,
        "stopped_early": stop_reason is not None,
        "stop_reason": stop_reason,
        "elapsed_s": time.perf_counter() - started,
        "output_dir": str(run_dir),
    }

    write_jsonl(run_dir / "query_results.jsonl", all_rows)
    write_json(run_dir / "metrics.json", summary)
    write_csv(run_dir / "metrics.csv", aggregate_rows)
    return summary


def _build_groq_client(config: RunConfig, keys: list[ApiKey]) -> RoundRobinGroqClient:
    return RoundRobinGroqClient(
        keys=keys,
        model=config.model,
        max_retries=config.max_retries,
        key_tokens_per_minute=config.key_tokens_per_minute,
        key_requests_per_minute=config.key_requests_per_minute,
        rate_limit_scope=config.rate_limit_scope,
    )


def _run_id(config: RunConfig) -> str:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    retrievers = "-".join(config.retrievers)
    return f"{stamp}_{config.bench}_{retrievers}"


def _serializable_config(config: RunConfig) -> dict[str, Any]:
    output = dict(config.__dict__)
    output["output_dir"] = str(config.output_dir)
    output["groq_keys_path"] = str(config.groq_keys_path)
    output["retrievers"] = list(config.retrievers)
    return output


def _retrieval_ops_metrics(metadata: dict[str, Any]) -> dict[str, float]:
    numeric_keys = [
        "retrieval_llm_calls",
        "retrieval_llm_latency_s",
        "retrieval_llm_retry_count",
        "retrieval_llm_scheduled_wait_s",
        "retrieval_llm_prompt_tokens",
        "retrieval_llm_completion_tokens",
        "retrieval_llm_total_tokens",
        "retrieval_llm_estimated_tokens",
        "retrieval_llm_error_count",
    ]
    metrics: dict[str, float] = {}
    for key in numeric_keys:
        value = metadata.get(key)
        if value is not None:
            metrics[key] = float(value)
    return metrics


def _evaluate_ragas_by_retriever(
    rows: list[dict[str, Any]],
    *,
    retrievers: list[str],
    keys: list[ApiKey],
    model: str,
    limit: int | None,
    evaluator: Callable[..., dict[str, Any]],
) -> dict[str, Any]:
    by_retriever: dict[str, Any] = {}
    for retriever in retrievers:
        retriever_rows = [
            row
            for row in rows
            if row.get("retriever") == retriever and not row.get("generation_skipped")
        ]
        by_retriever[retriever] = evaluator(
            retriever_rows,
            keys=keys,
            model=model,
            limit=limit,
        )
    return {
        "mode": "by_retriever",
        "ragas_limit_per_retriever": limit,
        "sample_count": sum(summary.get("sample_count", 0) for summary in by_retriever.values()),
        "error_count": sum(summary.get("error_count", 0) for summary in by_retriever.values()),
        "by_retriever": by_retriever,
    }
