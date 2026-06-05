from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import string
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean
from typing import Any, Callable, Iterable

from rag_bench.benchmarks import load_benchmark
from rag_bench.context_budget import ContextBudget, apply_context_budget, estimate_tokens_from_chars
from rag_bench.context_metrics import aggregate_context_budget_metrics, aggregate_kv_estimates, context_budget_metrics
from rag_bench.groq_client import OpenAICompatibleClient, RoundRobinGroqClient, estimate_requested_tokens
from rag_bench.io import write_json, write_jsonl
from rag_bench.kv_estimator import estimate_kv_cache_savings
from rag_bench.metrics import aggregate_generation, aggregate_metric_dicts, exact_match, retrieval_metrics_for_query, token_f1
from rag_bench.prompts import build_rag_messages_from_context
from rag_bench.retriever_registry import create_retriever
from rag_bench.runner import DEFAULT_MIMO_BASE_URL
from rag_bench.secrets import ApiKey, SecretFormatError, load_env_api_key, load_groq_keys
from rag_bench.types import BenchmarkData, Query, RetrievalHit, RetrievalResult


DEFAULT_BENCH = "hotpotqa"
DEFAULT_DATASET_ID = "beir/hotpotqa/test"
DEFAULT_MODEL = "mimo-v2.5"
DEFAULT_MODEL_ROLE = "long-context-judge-generator"
DEFAULT_POLICIES = ("legacy", "evidence-aware", "adaptive-heuristic")
DEFAULT_BUDGETS = (4000, 8000, 16000, 32000)
DEFAULT_ADAPTIVE_PROFILES = ("balanced", "aggressive")
DEFAULT_REFERENCE_CONFIGS = ("fullwiki", "distractor")
DEFAULT_REFERENCE_SPLITS = ("validation",)
DEFAULT_OUTPUT_DIR = Path("benchmark_results/budgetrag/phase1c3_hotpotqa_kaggle")
DEFAULT_PROVIDER = "mimo"


@dataclass(frozen=True)
class ActionSpec:
    context_policy: str
    context_budget_chars: int
    adaptive_profile: str | None = None

    @property
    def action_id(self) -> str:
        profile = f"__{self.adaptive_profile}" if self.adaptive_profile else ""
        return f"{self.context_policy}{profile}__{self.context_budget_chars}"


@dataclass(frozen=True)
class HotpotqaCachedEvalConfig:
    limit: int = 50
    top_k: int = 10
    output_dir: Path = DEFAULT_OUTPUT_DIR
    run_name: str | None = None
    policies: tuple[str, ...] = DEFAULT_POLICIES
    context_budgets: tuple[int, ...] = DEFAULT_BUDGETS
    adaptive_profiles: tuple[str, ...] = DEFAULT_ADAPTIVE_PROFILES
    max_action_rows: int | None = None
    provider: str = DEFAULT_PROVIDER
    model: str = DEFAULT_MODEL
    model_role: str = DEFAULT_MODEL_ROLE
    max_completion_tokens: int = 512
    temperature: float = 0.0
    max_retries: int = 2
    max_context_chars: int = 36_000
    per_doc_budget_chars: int | None = None
    kv_profile: str = "qwen2.5-14b"
    disable_kv_estimate: bool = False
    adaptive_small_budget: int = 1000
    adaptive_large_budget: int = 4000
    adaptive_per_doc_budget_chars: int = 800
    mimo_env_file: Path = Path(".secrets/.env")
    mimo_api_key_var: str = "MIMO_API_KEY"
    mimo_base_url: str = DEFAULT_MIMO_BASE_URL
    groq_keys_path: Path = Path(".secrets/groq_key.env")
    groq_key_alias: str | None = None
    key_tokens_per_minute: int = 0
    key_requests_per_minute: int = 0
    skip_generation: bool = False
    skip_ragas: bool = False
    ragas_model: str = DEFAULT_MODEL
    ragas_samples_per_action: int = 5
    ragas_seed: int = 20260529
    reference_configs: tuple[str, ...] = DEFAULT_REFERENCE_CONFIGS
    reference_splits: tuple[str, ...] = DEFAULT_REFERENCE_SPLITS


def run_hotpotqa_cached_eval(
    config: HotpotqaCachedEvalConfig,
    *,
    benchmark_loader: Callable[..., BenchmarkData] = load_benchmark,
    reference_lookup_loader: Callable[[], dict[str, str]] | None = None,
    llm_factory: Callable[[], RoundRobinGroqClient] | None = None,
    ragas_evaluator: Callable[..., dict[str, Any]] | None = None,
) -> dict[str, Any]:
    started = time.perf_counter()
    run_id = config.run_name or datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ_hotpotqa_kaggle")
    run_dir = config.output_dir / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    data = benchmark_loader(DEFAULT_BENCH, limit=config.limit, allow_large=True)
    reference_lookup = (
        reference_lookup_loader()
        if reference_lookup_loader is not None
        else load_hotpotqa_reference_lookup(config.reference_configs, config.reference_splits)
    )
    references_by_query_id = _references_by_query_id(data, reference_lookup)

    actions = build_action_specs(
        policies=config.policies,
        budgets=config.context_budgets,
        adaptive_profiles=config.adaptive_profiles,
    )
    if config.max_action_rows is not None:
        actions = actions[: max(0, config.max_action_rows)]

    retrieval_cache = build_retrieval_cache(data, top_k=config.top_k)
    write_jsonl(run_dir / "retrieval_cache.jsonl", retrieval_cache)

    llm = None if config.skip_generation else (llm_factory() if llm_factory is not None else build_generation_client(config))
    created_at = datetime.now(timezone.utc).isoformat()
    all_rows: list[dict[str, Any]] = []
    aggregate_rows: list[dict[str, Any]] = []
    for action in actions:
        action_rows, aggregate = run_action_from_cache(
            config=config,
            action=action,
            retrieval_cache=retrieval_cache,
            references_by_query_id=references_by_query_id,
            llm=llm,
            run_id=run_id,
            created_at=created_at,
        )
        all_rows.extend(action_rows)
        aggregate_rows.append(aggregate)

    ragas_summary = {"skipped": True, "reason": "disabled"}
    ragas_rows: list[dict[str, Any]] = []
    if not config.skip_ragas and all_rows:
        evaluator = ragas_evaluator or evaluate_ragas_with_mimo
        ragas_summary = evaluator(
            all_rows,
            samples_per_action=config.ragas_samples_per_action,
            seed=config.ragas_seed,
            model=config.ragas_model,
            mimo_env_file=config.mimo_env_file,
            mimo_api_key_var=config.mimo_api_key_var,
            mimo_base_url=config.mimo_base_url,
        )
        ragas_rows = list(ragas_summary.get("per_sample_rows", []))
        ragas_summary = {key: value for key, value in ragas_summary.items() if key != "per_sample_rows"}

    summary_rows = build_summary_rows(aggregate_rows, ragas_rows)
    summary = {
        "run_id": run_id,
        "created_at": created_at,
        "elapsed_s": time.perf_counter() - started,
        "config": _serializable_config(config),
        "benchmark": {
            "name": data.name,
            "dataset_id": data.dataset_id,
            "query_count": len(data.queries),
            "document_count": len(data.documents),
            "reference_join_count": sum(1 for query in data.queries if references_by_query_id.get(query.query_id)),
        },
        "actions": [asdict(action) | {"action_id": action.action_id} for action in actions],
        "aggregates": aggregate_rows,
        "ragas": ragas_summary,
        "summary_rows": summary_rows,
        "output_dir": str(run_dir),
    }

    write_jsonl(run_dir / "query_results.jsonl", all_rows)
    write_json(run_dir / "metrics.json", summary)
    write_csv_rows(run_dir / "hotpotqa_summary.csv", summary_rows)
    (run_dir / "hotpotqa_summary.md").write_text(render_markdown_summary(summary_rows, summary), encoding="utf-8")
    write_csv_rows(run_dir / "ragas_per_sample.csv", ragas_rows)
    return summary


def build_retrieval_cache(data: BenchmarkData, *, top_k: int) -> list[dict[str, Any]]:
    retriever = create_retriever("bm25", vector_model="sentence-transformers/all-MiniLM-L6-v2")
    retriever.build(data.documents)
    rows: list[dict[str, Any]] = []
    for query in data.queries:
        retrieval = retriever.search(query, top_k)
        rows.append(
            {
                "benchmark": data.name,
                "dataset_id": data.dataset_id,
                "retriever": retriever.name,
                "query_id": query.query_id,
                "question": query.text,
                "top_k": top_k,
                "retrieved": [_hit_to_dict(hit) for hit in retrieval.hits],
                "retrieval_metrics": retrieval_metrics_for_query(
                    retrieval,
                    data.qrels.get(query.query_id, {}),
                    top_k=top_k,
                ),
                "retrieval_metadata": retrieval.metadata,
            }
        )
    return rows


def run_action_from_cache(
    *,
    config: HotpotqaCachedEvalConfig,
    action: ActionSpec,
    retrieval_cache: list[dict[str, Any]],
    references_by_query_id: dict[str, tuple[str, ...]],
    llm: RoundRobinGroqClient | None,
    run_id: str,
    created_at: str,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    retrieval_metric_rows: list[dict[str, float]] = []
    context_budget_rows: list[dict[str, Any]] = []
    kv_estimate_rows: list[dict[str, Any] | None] = []
    generation_rows: list[dict[str, Any]] = []

    for cached in retrieval_cache:
        query = Query(
            query_id=str(cached["query_id"]),
            text=str(cached["question"]),
            reference_answers=references_by_query_id.get(str(cached["query_id"]), ()),
        )
        hits = [_hit_from_dict(hit) for hit in cached.get("retrieved", [])]
        retrieval_metric_rows.append(dict(cached.get("retrieval_metrics") or {}))
        context_budget = _action_context_budget(config, action, query_text=query.text)
        budgeted_context = apply_context_budget(hits, context_budget)
        context_metrics = context_budget_metrics(budgeted_context, context_budget, retrieved_docs=len(hits))
        prompt_context = _prompt_context_with_safety_ceiling(
            budgeted_context.text,
            max_context_chars=config.max_context_chars,
            context_budget_chars=action.context_budget_chars,
        )
        if prompt_context != budgeted_context.text:
            context_metrics["metadata"] = {
                **context_metrics.get("metadata", {}),
                "prompt_safety_truncated": True,
                "prompt_safety_max_context_chars": config.max_context_chars,
            }
        kv_estimate = None
        if not config.disable_kv_estimate:
            kv_estimate = estimate_kv_cache_savings(
                before_tokens=budgeted_context.original_est_tokens,
                after_tokens=min(budgeted_context.kept_est_tokens, budgeted_context.original_est_tokens),
                profile=config.kv_profile,
            )
        context_budget_rows.append(context_metrics)
        kv_estimate_rows.append(kv_estimate)

        messages = build_rag_messages_from_context(query, prompt_context)
        estimated_request_tokens = estimate_requested_tokens(
            messages,
            max_completion_tokens=config.max_completion_tokens,
        )
        estimated_prompt_tokens = max(1, estimated_request_tokens - max(0, config.max_completion_tokens))
        generation = None
        if llm is not None:
            generation = llm.generate(
                messages,
                temperature=config.temperature,
                max_completion_tokens=config.max_completion_tokens,
            )
        answer = generation.answer if generation is not None else ""
        estimated_completion_tokens = (
            generation.completion_tokens
            if generation is not None and generation.completion_tokens is not None
            else estimate_tokens_from_chars(len(answer))
        )
        generation_detail = (
            {
                "provider": config.provider,
                "model": config.model,
                "model_role": config.model_role,
                "max_completion_tokens": config.max_completion_tokens,
                "latency_s": generation.latency_s,
                "prompt_tokens": generation.prompt_tokens,
                "completion_tokens": generation.completion_tokens,
                "total_tokens": generation.total_tokens,
                "token_usage_is_estimated": generation.prompt_tokens is None or generation.completion_tokens is None,
                "estimated_prompt_tokens": estimated_prompt_tokens,
                "estimated_completion_tokens": estimated_completion_tokens,
                "estimated_request_tokens": estimated_request_tokens,
                "answer_length_chars": len(answer),
                "answer_length_est_tokens": estimate_tokens_from_chars(len(answer)),
                "error": generation.error,
                "error_status_code": generation.error_status_code,
            }
            if generation is not None
            else None
        )
        row = {
            "experiment": _experiment_metadata(config, action, run_id=run_id, created_at=created_at),
            "run_id": run_id,
            "benchmark": DEFAULT_BENCH,
            "dataset_id": DEFAULT_DATASET_ID,
            "retriever": "bm25",
            "action_id": action.action_id,
            "context_policy": action.context_policy,
            "context_budget_chars": action.context_budget_chars,
            "adaptive_profile": action.adaptive_profile,
            "query_id": query.query_id,
            "question": query.text,
            "reference_answers": list(query.reference_answers),
            "top_k": config.top_k,
            "retrieved": [_hit_to_dict(hit) for hit in hits],
            "retrieval_metrics": cached.get("retrieval_metrics") or {},
            "retrieval_metadata": cached.get("retrieval_metadata") or {},
            "context_budget": context_metrics,
            "adaptive_budget": context_metrics.get("metadata", {}).get("adaptive_budget"),
            "kv_estimate": kv_estimate,
            "estimated_prompt_tokens_after_budget": budgeted_context.kept_est_tokens,
            "estimated_prompt_tokens_saved_by_budget": max(
                0,
                budgeted_context.original_est_tokens - budgeted_context.kept_est_tokens,
            ),
            "estimated_prompt_tokens": estimated_prompt_tokens,
            "estimated_completion_tokens": estimated_completion_tokens if generation is not None else None,
            "answer_length_chars": len(answer) if generation is not None else None,
            "answer_length_est_tokens": estimate_tokens_from_chars(len(answer)) if generation is not None else None,
            "generation_skipped": generation is None,
            "generation": generation_detail,
            "answer": answer,
            "answer_latency_s": generation.latency_s if generation is not None else None,
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
            "exact_match": exact_match(answer, query.reference_answers),
            "token_f1": token_f1(answer, query.reference_answers),
        }
        rows.append(row)
        if generation is not None:
            generation_rows.append(row)

    aggregate = {
        "experiment": _experiment_metadata(config, action, run_id=run_id, created_at=created_at),
        "run_id": run_id,
        "benchmark": DEFAULT_BENCH,
        "dataset_id": DEFAULT_DATASET_ID,
        "retriever": "bm25",
        "action_id": action.action_id,
        "query_count": len(rows),
        "top_k": config.top_k,
        "retrieval": aggregate_metric_dicts(retrieval_metric_rows),
        "context_budget": aggregate_context_budget_metrics(context_budget_rows),
        "kv_estimate": aggregate_kv_estimates(kv_estimate_rows),
        "generation": (
            {"skipped": True, "generation_count": 0}
            if config.skip_generation
            else aggregate_generation(generation_rows)
        ),
        "reference_join_count": sum(1 for row in rows if row.get("reference_answers")),
    }
    return rows, aggregate


def build_action_specs(
    *,
    policies: Iterable[str],
    budgets: Iterable[int],
    adaptive_profiles: Iterable[str],
) -> list[ActionSpec]:
    actions: list[ActionSpec] = []
    for policy in policies:
        for budget in budgets:
            if policy == "adaptive-heuristic":
                for profile in adaptive_profiles:
                    actions.append(ActionSpec(policy, int(budget), profile))
            else:
                actions.append(ActionSpec(policy, int(budget), None))
    return actions


def load_hotpotqa_reference_lookup(
    configs: Iterable[str] = DEFAULT_REFERENCE_CONFIGS,
    splits: Iterable[str] = DEFAULT_REFERENCE_SPLITS,
) -> dict[str, str]:
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise RuntimeError("HotpotQA reference loading requires datasets. Install with --extra ragas.") from exc

    lookup: dict[str, str] = {}
    errors: list[str] = []
    for config_name in configs:
        for split in splits:
            try:
                dataset = load_dataset("hotpotqa/hotpot_qa", config_name, split=split)
            except Exception as exc:  # noqa: BLE001 - HF loaders expose several exception classes.
                errors.append(f"{config_name}/{split}: {exc}")
                continue
            lookup.update(build_reference_lookup_from_records(dataset))
            if lookup:
                return lookup
    if errors:
        raise RuntimeError("Could not load HotpotQA references: " + "; ".join(errors[:3]))
    return lookup


def build_reference_lookup_from_records(records: Iterable[dict[str, Any]]) -> dict[str, str]:
    lookup: dict[str, str] = {}
    for record in records:
        question = str(record.get("question") or "").strip()
        answer = record.get("answer")
        if not question or answer is None:
            continue
        answer_text = str(answer).strip()
        if not answer_text:
            continue
        lookup.setdefault(normalize_question(question), answer_text)
    return lookup


def normalize_question(text: str) -> str:
    table = str.maketrans("", "", string.punctuation)
    return " ".join(text.lower().translate(table).split())


def select_ragas_sample_rows(rows: list[dict[str, Any]], *, samples_per_action: int, seed: int) -> list[dict[str, Any]]:
    selected: list[dict[str, Any]] = []
    by_action: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        if row.get("generation_skipped") or row.get("error") or not str(row.get("answer") or "").strip():
            continue
        by_action.setdefault(str(row.get("action_id")), []).append(row)
    for action_id, action_rows in sorted(by_action.items()):
        ranked = sorted(
            action_rows,
            key=lambda row: _stable_sample_key(seed, action_id, str(row.get("query_id"))),
        )
        selected.extend(ranked[: max(0, samples_per_action)])
    return selected


def evaluate_ragas_with_mimo(
    rows: list[dict[str, Any]],
    *,
    samples_per_action: int,
    seed: int,
    model: str,
    mimo_env_file: Path,
    mimo_api_key_var: str,
    mimo_base_url: str,
) -> dict[str, Any]:
    selected = select_ragas_sample_rows(rows, samples_per_action=samples_per_action, seed=seed)
    if not selected:
        return {
            "skipped": False,
            "sample_count": 0,
            "samples_per_action": samples_per_action,
            "judge_provider": "mimo",
            "judge_model": model,
            "error": "no valid generated rows for RAGAS",
            "per_sample_rows": [],
        }

    try:
        from datasets import Dataset
        from langchain_openai import ChatOpenAI
        from ragas import evaluate
        from ragas.metrics import AnswerCorrectness, Faithfulness, ResponseRelevancy
        from rag_bench.ragas_eval import _load_ragas_embeddings, _row_to_ragas_sample
    except ImportError as exc:
        raise RuntimeError("MiMo RAGAS requires --extra vector --extra ragas dependencies.") from exc

    api_key = os.environ.get(mimo_api_key_var)
    if not api_key:
        api_key = load_env_api_key(mimo_env_file, mimo_api_key_var, alias="mimo").value
    llm = ChatOpenAI(
        api_key=api_key,
        base_url=mimo_base_url,
        model=model,
        temperature=0.0,
        timeout=120,
    )
    embeddings = _load_ragas_embeddings()
    metrics = [ResponseRelevancy(strictness=1), Faithfulness(max_retries=1)]
    if all(row.get("reference_answers") for row in selected):
        metrics.append(AnswerCorrectness(max_retries=1))

    samples = [_row_to_ragas_sample(_row_for_ragas(row)) for row in selected]
    result = evaluate(
        Dataset.from_list(samples),
        metrics=metrics,
        llm=llm,
        embeddings=embeddings,
        show_progress=True,
        batch_size=2,
    )
    frame = result.to_pandas()
    metric_rows = frame.to_dict(orient="records") if hasattr(frame, "to_dict") else []
    per_sample_rows: list[dict[str, Any]] = []
    for row, metric_row in zip(selected, metric_rows, strict=False):
        output = {
            "action_id": row.get("action_id"),
            "context_policy": row.get("context_policy"),
            "context_budget_chars": row.get("context_budget_chars"),
            "adaptive_profile": row.get("adaptive_profile") or "",
            "query_id": row.get("query_id"),
        }
        for key, value in metric_row.items():
            if _is_number(value):
                output[str(key)] = float(value)
        per_sample_rows.append(output)

    return {
        "skipped": False,
        "judge_provider": "mimo",
        "judge_model": model,
        "samples_per_action": samples_per_action,
        "sample_count": len(per_sample_rows),
        "metrics": _average_metrics(per_sample_rows),
        "by_action": _average_ragas_by_action(per_sample_rows),
        "per_sample_rows": per_sample_rows,
    }


def build_summary_rows(aggregates: list[dict[str, Any]], ragas_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    ragas_by_action = _average_ragas_by_action(ragas_rows)
    rows: list[dict[str, Any]] = []
    for aggregate in aggregates:
        experiment = aggregate.get("experiment") or {}
        context = aggregate.get("context_budget") or {}
        retrieval = aggregate.get("retrieval") or {}
        generation = aggregate.get("generation") or {}
        kv = aggregate.get("kv_estimate") or {}
        action_id = str(aggregate.get("action_id") or experiment.get("action_id") or "")
        ragas = ragas_by_action.get(action_id, {})
        rows.append(
            {
                "action_id": action_id,
                "context_policy": experiment.get("context_policy"),
                "context_budget_chars": experiment.get("context_budget_chars"),
                "adaptive_profile": experiment.get("adaptive_profile") or "",
                "query_count": aggregate.get("query_count"),
                "reference_join_count": aggregate.get("reference_join_count"),
                "recall@10": retrieval.get("recall@k"),
                "ndcg@10": retrieval.get("ndcg@k"),
                "avg_kept_context_chars": context.get("avg_kept_context_chars"),
                "avg_kept_context_est_tokens": context.get("avg_kept_context_est_tokens"),
                "avg_context_compression_ratio": context.get("avg_context_compression_ratio"),
                "avg_estimated_kv_cache_savings_mb": kv.get("avg_estimated_kv_cache_savings_mb"),
                "avg_generation_latency_s": generation.get("avg_answer_latency_s"),
                "exact_match": generation.get("avg_exact_match"),
                "token_f1": generation.get("avg_token_f1"),
                "ragas_sample_count": ragas.get("sample_count", 0),
                "ragas_answer_relevancy": ragas.get("answer_relevancy"),
                "ragas_faithfulness": ragas.get("faithfulness"),
                "ragas_answer_correctness": ragas.get("answer_correctness"),
            }
        )
    return rows


def render_markdown_summary(rows: list[dict[str, Any]], summary: dict[str, Any]) -> str:
    lines = [
        "# HotpotQA Kaggle BudgetRAG Eval",
        "",
        f"Run: `{summary.get('run_id')}`",
        f"Queries: {summary.get('benchmark', {}).get('query_count')}",
        f"Reference joins: {summary.get('benchmark', {}).get('reference_join_count')}",
        "",
        "| action | policy | profile | budget | recall@10 | nDCG@10 | EM | F1 | RAGAS rel | RAGAS faith |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            "| {action_id} | {policy} | {profile} | {budget} | {recall} | {ndcg} | {em} | {f1} | {rel} | {faith} |".format(
                action_id=row.get("action_id"),
                policy=row.get("context_policy"),
                profile=row.get("adaptive_profile") or "-",
                budget=row.get("context_budget_chars"),
                recall=_fmt(row.get("recall@10")),
                ndcg=_fmt(row.get("ndcg@10")),
                em=_fmt(row.get("exact_match")),
                f1=_fmt(row.get("token_f1")),
                rel=_fmt(row.get("ragas_answer_relevancy")),
                faith=_fmt(row.get("ragas_faithfulness")),
            )
        )
    lines.extend(
        [
            "",
            "Notes:",
            "- Sampled HotpotQA eval; do not present as full test-set benchmark.",
            "- RAGAS uses MiMo judge with one generation per metric call; reliability comes from samples per action.",
        ]
    )
    return "\n".join(lines) + "\n"


def build_mimo_client(config: HotpotqaCachedEvalConfig) -> RoundRobinGroqClient:
    env_value = os.environ.get(config.mimo_api_key_var)
    key = (
        ApiKey(alias="mimo", value=env_value)
        if env_value
        else load_env_api_key(config.mimo_env_file, config.mimo_api_key_var, alias="mimo")
    )
    return RoundRobinGroqClient(
        keys=[key],
        model=config.model,
        max_retries=config.max_retries,
        key_tokens_per_minute=config.key_tokens_per_minute,
        key_requests_per_minute=config.key_requests_per_minute,
        rate_limit_scope="per-key",
        client_factory=lambda api_key, timeout: OpenAICompatibleClient(
            api_key=api_key.value,
            base_url=config.mimo_base_url,
            timeout_s=timeout,
            token_parameter="max_tokens",
        ),
        provider_name="MiMo",
        completion_token_parameter="max_tokens",
    )


def build_groq_client(config: HotpotqaCachedEvalConfig) -> RoundRobinGroqClient:
    return RoundRobinGroqClient(
        keys=_load_selected_groq_keys(config.groq_keys_path, config.groq_key_alias),
        model=config.model,
        max_retries=config.max_retries,
        key_tokens_per_minute=config.key_tokens_per_minute,
        key_requests_per_minute=config.key_requests_per_minute,
        rate_limit_scope="per-key",
        provider_name="Groq",
    )


def build_generation_client(config: HotpotqaCachedEvalConfig) -> RoundRobinGroqClient:
    provider = config.provider.strip().lower()
    if provider == "mimo":
        return build_mimo_client(config)
    if provider == "groq":
        return build_groq_client(config)
    raise ValueError(f"Unsupported HotpotQA generation provider: {config.provider}")


def _load_selected_groq_keys(path: Path, alias: str | None) -> list[ApiKey]:
    keys = load_groq_keys(path)
    if alias is None:
        return keys
    selected = [key for key in keys if key.alias == alias]
    if not selected:
        raise SecretFormatError(f"Groq key alias was not found in {path}: {alias}")
    return selected


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run cached HotpotQA BudgetRAG eval with one BM25 build.")
    parser.add_argument("--limit", type=int, default=50)
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--run-name", default=None)
    parser.add_argument("--context-policies", default=",".join(DEFAULT_POLICIES))
    parser.add_argument("--context-budgets", default=",".join(str(value) for value in DEFAULT_BUDGETS))
    parser.add_argument("--adaptive-profiles", default=",".join(DEFAULT_ADAPTIVE_PROFILES))
    parser.add_argument("--max-action-rows", type=int, default=None)
    parser.add_argument("--provider", choices=("mimo", "groq"), default=DEFAULT_PROVIDER)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--model-role", default=DEFAULT_MODEL_ROLE)
    parser.add_argument("--max-completion-tokens", type=int, default=512)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max-retries", type=int, default=2)
    parser.add_argument("--max-context-chars", type=int, default=36_000)
    parser.add_argument("--per-doc-budget-chars", type=int, default=None)
    parser.add_argument("--kv-profile", default="qwen2.5-14b")
    parser.add_argument("--disable-kv-estimate", action="store_true")
    parser.add_argument("--adaptive-small-budget", type=int, default=1000)
    parser.add_argument("--adaptive-large-budget", type=int, default=4000)
    parser.add_argument("--adaptive-per-doc-budget-chars", type=int, default=800)
    parser.add_argument("--mimo-env-file", type=Path, default=Path(".secrets/.env"))
    parser.add_argument("--mimo-api-key-var", default="MIMO_API_KEY")
    parser.add_argument("--mimo-base-url", default=DEFAULT_MIMO_BASE_URL)
    parser.add_argument("--groq-keys-path", type=Path, default=Path(".secrets/groq_key.env"))
    parser.add_argument("--groq-key-alias", default=None)
    parser.add_argument("--key-tpm", type=int, default=0)
    parser.add_argument("--key-rpm", type=int, default=0)
    parser.add_argument("--skip-generation", action="store_true")
    parser.add_argument("--skip-ragas", action="store_true")
    parser.add_argument("--ragas-model", default=DEFAULT_MODEL)
    parser.add_argument("--ragas-samples-per-action", type=int, default=5)
    parser.add_argument("--ragas-seed", type=int, default=20260529)
    parser.add_argument("--reference-configs", default=",".join(DEFAULT_REFERENCE_CONFIGS))
    parser.add_argument("--reference-splits", default=",".join(DEFAULT_REFERENCE_SPLITS))
    return parser.parse_args(argv)


def config_from_args(args: argparse.Namespace) -> HotpotqaCachedEvalConfig:
    return HotpotqaCachedEvalConfig(
        limit=args.limit,
        top_k=args.top_k,
        output_dir=args.output_dir,
        run_name=args.run_name,
        policies=tuple(_split_csv(args.context_policies)),
        context_budgets=tuple(int(value) for value in _split_csv(args.context_budgets)),
        adaptive_profiles=tuple(_split_csv(args.adaptive_profiles)),
        max_action_rows=args.max_action_rows,
        provider=args.provider,
        model=args.model,
        model_role=args.model_role,
        max_completion_tokens=args.max_completion_tokens,
        temperature=args.temperature,
        max_retries=args.max_retries,
        max_context_chars=args.max_context_chars,
        per_doc_budget_chars=args.per_doc_budget_chars,
        kv_profile=args.kv_profile,
        disable_kv_estimate=args.disable_kv_estimate,
        adaptive_small_budget=args.adaptive_small_budget,
        adaptive_large_budget=args.adaptive_large_budget,
        adaptive_per_doc_budget_chars=args.adaptive_per_doc_budget_chars,
        mimo_env_file=args.mimo_env_file,
        mimo_api_key_var=args.mimo_api_key_var,
        mimo_base_url=args.mimo_base_url,
        groq_keys_path=args.groq_keys_path,
        groq_key_alias=args.groq_key_alias,
        key_tokens_per_minute=args.key_tpm,
        key_requests_per_minute=args.key_rpm,
        skip_generation=args.skip_generation,
        skip_ragas=args.skip_ragas,
        ragas_model=args.ragas_model,
        ragas_samples_per_action=args.ragas_samples_per_action,
        ragas_seed=args.ragas_seed,
        reference_configs=tuple(_split_csv(args.reference_configs)),
        reference_splits=tuple(_split_csv(args.reference_splits)),
    )


def main(argv: list[str] | None = None) -> int:
    summary = run_hotpotqa_cached_eval(config_from_args(parse_args(argv)))
    print(json.dumps({"output_dir": summary["output_dir"], "actions": len(summary["actions"])}, indent=2))
    return 0


def _references_by_query_id(data: BenchmarkData, lookup: dict[str, str]) -> dict[str, tuple[str, ...]]:
    references: dict[str, tuple[str, ...]] = {}
    for query in data.queries:
        answer = lookup.get(normalize_question(query.text))
        references[query.query_id] = (answer,) if answer else ()
    return references


def _action_context_budget(config: HotpotqaCachedEvalConfig, action: ActionSpec, *, query_text: str) -> ContextBudget:
    return ContextBudget(
        policy=action.context_policy,
        max_chars=action.context_budget_chars,
        per_doc_max_chars=config.per_doc_budget_chars,
        query=query_text,
        adaptive_small_budget=config.adaptive_small_budget,
        adaptive_medium_budget=action.context_budget_chars,
        adaptive_large_budget=config.adaptive_large_budget,
        adaptive_profile=action.adaptive_profile or "balanced",
        adaptive_per_doc_budget_chars=config.adaptive_per_doc_budget_chars,
    )


def _experiment_metadata(
    config: HotpotqaCachedEvalConfig,
    action: ActionSpec,
    *,
    run_id: str,
    created_at: str,
) -> dict[str, Any]:
    return {
        "run_id": run_id,
        "created_at": created_at,
        "bench": DEFAULT_BENCH,
        "benchmark": DEFAULT_BENCH,
        "dataset_id": DEFAULT_DATASET_ID,
        "retriever": "bm25",
        "top_k": config.top_k,
        "action_id": action.action_id,
        "context_policy": action.context_policy,
        "context_budget_chars": action.context_budget_chars,
        "adaptive_profile": action.adaptive_profile,
        "skip_generation": config.skip_generation,
        "generation_provider": None if config.skip_generation else config.provider,
        "generation_model": None if config.skip_generation else config.model,
        "generation_model_role": None if config.skip_generation else config.model_role,
        "kv_profile": None if config.disable_kv_estimate else config.kv_profile,
    }


def _prompt_context_with_safety_ceiling(
    context: str,
    *,
    max_context_chars: int,
    context_budget_chars: int | None,
) -> str:
    if context_budget_chars is None or len(context) <= max_context_chars:
        return context
    return context[:max_context_chars].rstrip()


def _row_for_ragas(row: dict[str, Any]) -> dict[str, Any]:
    reference_answers = row.get("reference_answers") or []
    output = dict(row)
    output["reference"] = reference_answers[0] if reference_answers else ""
    return output


def _average_ragas_by_action(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    by_action: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        by_action.setdefault(str(row.get("action_id")), []).append(row)
    output: dict[str, dict[str, Any]] = {}
    for action_id, action_rows in by_action.items():
        output[action_id] = {"sample_count": len(action_rows), **_average_metrics(action_rows)}
    return output


def _average_metrics(rows: list[dict[str, Any]]) -> dict[str, float]:
    values: dict[str, list[float]] = {}
    metadata_keys = {"action_id", "context_policy", "adaptive_profile", "query_id", "context_budget_chars"}
    for row in rows:
        for key, value in row.items():
            if key in metadata_keys:
                continue
            if _is_number(value):
                values.setdefault(key, []).append(float(value))
    return {key: mean(metric_values) for key, metric_values in values.items() if metric_values}


def _stable_sample_key(seed: int, action_id: str, query_id: str) -> str:
    return hashlib.sha256(f"{seed}:{action_id}:{query_id}".encode("utf-8")).hexdigest()


def _hit_from_dict(data: dict[str, Any]) -> RetrievalHit:
    return RetrievalHit(
        doc_id=str(data.get("doc_id") or ""),
        rank=int(data.get("rank") or 0),
        score=float(data.get("score") or 0.0),
        title=str(data.get("title") or ""),
        text=str(data.get("text") or ""),
        metadata=dict(data.get("metadata") or {}),
    )


def _hit_to_dict(hit: RetrievalHit) -> dict[str, Any]:
    return {
        "doc_id": hit.doc_id,
        "rank": hit.rank,
        "score": hit.score,
        "title": hit.title,
        "text": hit.text,
        "metadata": hit.metadata,
    }


def _serializable_config(config: HotpotqaCachedEvalConfig) -> dict[str, Any]:
    data = asdict(config)
    data["output_dir"] = str(config.output_dir)
    data["mimo_env_file"] = str(config.mimo_env_file)
    data["groq_keys_path"] = str(config.groq_keys_path)
    data["policies"] = list(config.policies)
    data["context_budgets"] = list(config.context_budgets)
    data["adaptive_profiles"] = list(config.adaptive_profiles)
    data["reference_configs"] = list(config.reference_configs)
    data["reference_splits"] = list(config.reference_splits)
    return data


def write_csv_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = sorted({key for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _split_csv(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def _fmt(value: Any) -> str:
    return f"{float(value):.3f}" if _is_number(value) else "-"


def _is_number(value: Any) -> bool:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return False
    return math.isfinite(number)
