#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import shutil
import time
from collections import defaultdict
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

from rag_bench.hotpotqa_cached_eval import (
    ActionSpec,
    DEFAULT_MIMO_BASE_URL,
    DEFAULT_OUTPUT_DIR,
    HotpotqaCachedEvalConfig,
    build_generation_client,
    build_summary_rows,
    evaluate_ragas_with_mimo,
    render_markdown_summary,
    run_action_from_cache,
    write_csv_rows,
)
from rag_bench.io import write_json, write_jsonl
from rag_bench.kv_estimator import estimate_kv_cache_savings
from rag_bench.metrics import aggregate_generation, aggregate_metric_dicts
from rag_bench.context_metrics import aggregate_context_budget_metrics, aggregate_kv_estimates
from rag_bench.groq_client import RoundRobinGroqClient


def retry_failed_rows(
    *,
    original_run_dir: Path,
    output_dir: Path,
    run_name: str | None,
    config: HotpotqaCachedEvalConfig,
    failed_status_code: int | None,
    max_failed_rows: int | None,
    include_non_status_errors: bool,
    run_ragas: bool,
    resume: bool = True,
    checkpoint_every: int = 1,
    progress_every: int = 1,
    llm_factory: Callable[[], RoundRobinGroqClient] | None = None,
) -> dict[str, Any]:
    started = time.perf_counter()
    original_summary = _read_json(original_run_dir / "metrics.json")
    original_rows = _read_jsonl(original_run_dir / "query_results.jsonl")
    run_id = run_name or datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ_hotpotqa_retry")
    run_dir = output_dir / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    failed_rows = [
        row
        for row in original_rows
        if _is_retry_candidate(
            row,
            failed_status_code=failed_status_code,
            include_non_status_errors=include_non_status_errors,
        )
    ]
    if max_failed_rows is not None:
        failed_rows = failed_rows[: max(0, max_failed_rows)]
    failed_keys = {_row_key(row) for row in failed_rows}
    kept_rows = [row for row in original_rows if _row_key(row) not in failed_keys]

    llm = llm_factory() if llm_factory is not None else build_generation_client(config)
    created_at = datetime.now(timezone.utc).isoformat()
    partial_path = run_dir / "retry_rows.partial.jsonl"
    progress_path = run_dir / "retry_progress.json"
    if partial_path.exists() and not resume:
        partial_path.unlink()
    retry_rows = _read_partial_retry_rows(partial_path, failed_keys) if resume else []
    previous_retry_count = len(retry_rows)
    completed_keys = {_row_key(row) for row in retry_rows}
    pending_rows = [row for row in failed_rows if _row_key(row) not in completed_keys]

    _write_retry_progress(
        progress_path,
        run_id=run_id,
        candidate_count=len(failed_rows),
        previous_retry_count=previous_retry_count,
        pending_count=len(pending_rows),
        retry_rows=retry_rows,
        elapsed_s=time.perf_counter() - started,
        status="running",
    )

    for index, row in enumerate(pending_rows, start=1):
        action = _action_from_row(row)
        action_rows, _aggregate = run_action_from_cache(
            config=config,
            action=action,
            retrieval_cache=[_cached_row_from_query_result(row)],
            references_by_query_id={
                str(row.get("query_id")): tuple(str(answer) for answer in row.get("reference_answers") or [])
            },
            llm=llm,
            run_id=run_id,
            created_at=created_at,
        )
        retry_rows.extend(action_rows)
        if checkpoint_every > 0 and index % checkpoint_every == 0:
            write_jsonl(partial_path, retry_rows)
            _write_retry_progress(
                progress_path,
                run_id=run_id,
                candidate_count=len(failed_rows),
                previous_retry_count=previous_retry_count,
                pending_count=max(0, len(pending_rows) - index),
                retry_rows=retry_rows,
                elapsed_s=time.perf_counter() - started,
                status="running",
            )
        if progress_every > 0 and (index % progress_every == 0 or index == len(pending_rows)):
            latest = action_rows[-1] if action_rows else {}
            print(
                _format_progress_line(
                    index=index,
                    total=len(pending_rows),
                    previous_retry_count=previous_retry_count,
                    candidate_count=len(failed_rows),
                    row=latest or row,
                    elapsed_s=time.perf_counter() - started,
                ),
                flush=True,
            )

    if checkpoint_every > 0:
        write_jsonl(partial_path, retry_rows)
        _write_retry_progress(
            progress_path,
            run_id=run_id,
            candidate_count=len(failed_rows),
            previous_retry_count=previous_retry_count,
            pending_count=0,
            retry_rows=retry_rows,
            elapsed_s=time.perf_counter() - started,
            status="finalizing",
        )

    merged_rows = _sort_rows([*kept_rows, *retry_rows], original_rows)
    aggregate_rows = _aggregate_rows_by_action(merged_rows, original_summary)
    retry_aggregates = _aggregate_rows_by_action(retry_rows, original_summary)

    ragas_summary = {"skipped": True, "reason": "disabled"}
    ragas_rows: list[dict[str, Any]] = []
    if run_ragas and merged_rows:
        ragas_summary = evaluate_ragas_with_mimo(
            merged_rows,
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
        "mode": "retry_failed_rows",
        "source_run": {
            "run_id": original_summary.get("run_id"),
            "output_dir": str(original_run_dir),
        },
        "retry": {
            "candidate_count": len(failed_rows),
            "retried_count": len(retry_rows),
            "previous_retry_count": previous_retry_count,
            "new_retry_count": max(0, len(retry_rows) - previous_retry_count),
            "pending_count": 0,
            "retry_success_count": sum(1 for row in retry_rows if not row.get("error")),
            "retry_error_count": sum(1 for row in retry_rows if row.get("error")),
            "failed_status_code": failed_status_code,
            "include_non_status_errors": include_non_status_errors,
            "max_failed_rows": max_failed_rows,
            "resume": resume,
            "checkpoint_every": checkpoint_every,
            "progress_every": progress_every,
            "partial_path": str(partial_path),
            "progress_path": str(progress_path),
        },
        "config": _serializable_retry_config(config),
        "benchmark": original_summary.get("benchmark", {}),
        "actions": original_summary.get("actions", []),
        "aggregates": aggregate_rows,
        "retry_aggregates": retry_aggregates,
        "ragas": ragas_summary,
        "summary_rows": summary_rows,
        "output_dir": str(run_dir),
    }

    _copy_if_exists(original_run_dir / "retrieval_cache.jsonl", run_dir / "retrieval_cache.jsonl")
    write_jsonl(run_dir / "query_results.jsonl", merged_rows)
    write_jsonl(run_dir / "retry_rows.jsonl", retry_rows)
    write_json(run_dir / "metrics.json", summary)
    write_csv_rows(run_dir / "hotpotqa_summary.csv", summary_rows)
    write_csv_rows(run_dir / "ragas_per_sample.csv", ragas_rows)
    (run_dir / "hotpotqa_summary.md").write_text(render_markdown_summary(summary_rows, summary), encoding="utf-8")
    _write_retry_progress(
        progress_path,
        run_id=run_id,
        candidate_count=len(failed_rows),
        previous_retry_count=previous_retry_count,
        pending_count=0,
        retry_rows=retry_rows,
        elapsed_s=summary["elapsed_s"],
        status="complete",
    )
    return summary


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Retry failed HotpotQA cached eval rows without rebuilding BM25.")
    parser.add_argument("--original-run-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--run-name", default=None)
    parser.add_argument("--failed-status-code", type=int, default=429)
    parser.add_argument("--all-error-rows", action="store_true")
    parser.add_argument("--max-failed-rows", type=int, default=None)
    parser.add_argument("--provider", choices=("mimo", "groq"), default="groq")
    parser.add_argument("--model", default="qwen/qwen3-32b")
    parser.add_argument("--model-role", default="stronger-baseline")
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
    parser.add_argument("--groq-keys-path", type=Path, default=Path(".secrets/groq_key.env"))
    parser.add_argument("--groq-key-alias", default=None)
    parser.add_argument("--key-tpm", type=int, default=5000)
    parser.add_argument("--key-rpm", type=int, default=3)
    parser.add_argument("--mimo-env-file", type=Path, default=Path(".secrets/.env"))
    parser.add_argument("--mimo-api-key-var", default="MIMO_API_KEY")
    parser.add_argument("--mimo-base-url", default=DEFAULT_MIMO_BASE_URL)
    parser.add_argument("--run-ragas", action="store_true")
    parser.add_argument("--ragas-model", default="mimo-v2.5")
    parser.add_argument("--ragas-samples-per-action", type=int, default=1)
    parser.add_argument("--ragas-seed", type=int, default=20260529)
    parser.add_argument("--no-resume", action="store_true")
    parser.add_argument("--checkpoint-every", type=int, default=1)
    parser.add_argument("--progress-every", type=int, default=1)
    return parser.parse_args(argv)


def config_from_args(args: argparse.Namespace) -> HotpotqaCachedEvalConfig:
    return HotpotqaCachedEvalConfig(
        output_dir=args.output_dir,
        run_name=args.run_name,
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
        groq_keys_path=args.groq_keys_path,
        groq_key_alias=args.groq_key_alias,
        key_tokens_per_minute=args.key_tpm,
        key_requests_per_minute=args.key_rpm,
        mimo_env_file=args.mimo_env_file,
        mimo_api_key_var=args.mimo_api_key_var,
        mimo_base_url=args.mimo_base_url,
        ragas_model=args.ragas_model,
        ragas_samples_per_action=args.ragas_samples_per_action,
        ragas_seed=args.ragas_seed,
    )


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    summary = retry_failed_rows(
        original_run_dir=args.original_run_dir,
        output_dir=args.output_dir,
        run_name=args.run_name,
        config=config_from_args(args),
        failed_status_code=None if args.all_error_rows else args.failed_status_code,
        max_failed_rows=args.max_failed_rows,
        include_non_status_errors=args.all_error_rows,
        run_ragas=args.run_ragas,
        resume=not args.no_resume,
        checkpoint_every=max(0, args.checkpoint_every),
        progress_every=max(0, args.progress_every),
    )
    print(json.dumps({"output_dir": summary["output_dir"], "retry": summary["retry"]}, indent=2))
    return 0


def _is_retry_candidate(
    row: dict[str, Any],
    *,
    failed_status_code: int | None,
    include_non_status_errors: bool,
) -> bool:
    if not row.get("error"):
        return False
    if include_non_status_errors:
        return True
    return row.get("error_status_code") == failed_status_code


def _row_key(row: dict[str, Any]) -> tuple[str, str]:
    return (str(row.get("action_id") or ""), str(row.get("query_id") or ""))


def _rows_by_action(rows: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row.get("action_id") or "")].append(row)
    return dict(grouped)


def _cached_row_from_query_result(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "benchmark": row.get("benchmark"),
        "dataset_id": row.get("dataset_id"),
        "retriever": row.get("retriever"),
        "query_id": row.get("query_id"),
        "question": row.get("question"),
        "top_k": row.get("top_k"),
        "retrieved": row.get("retrieved") or [],
        "retrieval_metrics": row.get("retrieval_metrics") or {},
        "retrieval_metadata": row.get("retrieval_metadata") or {},
    }


def _action_from_row(row: dict[str, Any]) -> ActionSpec:
    return ActionSpec(
        context_policy=str(row.get("context_policy") or ""),
        context_budget_chars=int(row.get("context_budget_chars") or 0),
        adaptive_profile=row.get("adaptive_profile") or None,
    )


def _sort_rows(rows: list[dict[str, Any]], original_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    order = {_row_key(row): index for index, row in enumerate(original_rows)}
    return sorted(rows, key=lambda row: order.get(_row_key(row), len(order)))


def _aggregate_rows_by_action(rows: list[dict[str, Any]], original_summary: dict[str, Any]) -> list[dict[str, Any]]:
    rows_by_action = _rows_by_action(rows)
    output: list[dict[str, Any]] = []
    action_ids = [str(action.get("action_id")) for action in original_summary.get("actions") or []]
    for action_id in action_ids:
        action_rows = rows_by_action.get(action_id, [])
        if not action_rows:
            continue
        first = action_rows[0]
        retrieval_rows = [dict(row.get("retrieval_metrics") or {}) for row in action_rows]
        context_rows = [dict(row.get("context_budget") or {}) for row in action_rows]
        kv_rows = [row.get("kv_estimate") for row in action_rows]
        generation_rows = [row for row in action_rows if not row.get("generation_skipped")]
        output.append(
            {
                "experiment": first.get("experiment") or {},
                "run_id": first.get("run_id"),
                "benchmark": first.get("benchmark"),
                "dataset_id": first.get("dataset_id"),
                "retriever": first.get("retriever"),
                "action_id": action_id,
                "query_count": len(action_rows),
                "top_k": first.get("top_k"),
                "retrieval": aggregate_metric_dicts(retrieval_rows),
                "context_budget": aggregate_context_budget_metrics(context_rows),
                "kv_estimate": aggregate_kv_estimates(kv_rows),
                "generation": aggregate_generation(generation_rows),
                "reference_join_count": sum(1 for row in action_rows if row.get("reference_answers")),
            }
        )
    return output


def _serializable_retry_config(config: HotpotqaCachedEvalConfig) -> dict[str, Any]:
    data = asdict(config)
    for key in ("output_dir", "mimo_env_file", "groq_keys_path"):
        data[key] = str(data[key])
    return data


def _read_partial_retry_rows(path: Path, failed_keys: set[tuple[str, str]]) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows = _read_jsonl(path)
    return [row for row in rows if _row_key(row) in failed_keys]


def _write_retry_progress(
    path: Path,
    *,
    run_id: str,
    candidate_count: int,
    previous_retry_count: int,
    pending_count: int,
    retry_rows: list[dict[str, Any]],
    elapsed_s: float,
    status: str,
) -> None:
    retry_success_count = sum(1 for row in retry_rows if not row.get("error"))
    retry_error_count = sum(1 for row in retry_rows if row.get("error"))
    write_json(
        path,
        {
            "run_id": run_id,
            "status": status,
            "candidate_count": candidate_count,
            "previous_retry_count": previous_retry_count,
            "retried_count": len(retry_rows),
            "new_retry_count": max(0, len(retry_rows) - previous_retry_count),
            "pending_count": pending_count,
            "retry_success_count": retry_success_count,
            "retry_error_count": retry_error_count,
            "elapsed_s": elapsed_s,
            "updated_at": datetime.now(timezone.utc).isoformat(),
        },
    )


def _format_progress_line(
    *,
    index: int,
    total: int,
    previous_retry_count: int,
    candidate_count: int,
    row: dict[str, Any],
    elapsed_s: float,
) -> str:
    done = previous_retry_count + index
    status = "error" if row.get("error") else "ok"
    query_id = str(row.get("query_id") or "")
    action_id = str(row.get("action_id") or "")
    return (
        f"retry {done}/{candidate_count} pending-batch {index}/{total} "
        f"status={status} action={action_id} query_id={query_id} elapsed_s={elapsed_s:.1f}"
    )


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _copy_if_exists(src: Path, dst: Path) -> None:
    if src.exists():
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)


if __name__ == "__main__":
    raise SystemExit(main())
