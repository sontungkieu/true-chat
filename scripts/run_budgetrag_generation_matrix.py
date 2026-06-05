from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

from rag_bench.adaptive_budget import ADAPTIVE_PROFILES
from rag_bench.generation_models import (
    DEFAULT_MODEL_CONFIG_PATH,
    GenerationModelConfig,
    load_generation_model_configs,
    select_generation_model_configs,
)
from rag_bench.secrets import SecretFormatError, load_env_values, load_groq_keys


@dataclass(frozen=True)
class GenerationMatrixJob:
    retriever: str
    model: GenerationModelConfig
    policy: str
    budget: int
    adaptive_profile: str | None
    output_dir: Path
    command: list[str]


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    retrievers = _split_csv(args.retrievers)
    policies = _split_csv(args.context_policies)
    budgets = [int(value) for value in _split_csv(args.context_budgets)]
    adaptive_profiles = _split_csv(args.adaptive_profiles)
    requested_models = _split_csv(args.models)
    _validate_matrix_inputs(retrievers, policies, budgets, adaptive_profiles, requested_models)
    if args.job_timeout_s < 0:
        raise SystemExit("--job-timeout-s must be non-negative")

    model_configs = load_generation_model_configs(args.model_config)
    selected_models, skipped_models = select_generation_model_configs(model_configs, requested_models)
    available_models, unavailable_models = _filter_available_models(selected_models, args)
    skipped_models.extend(unavailable_models)

    matrix_dir = args.output_dir / args.run_name if args.run_name else args.output_dir
    jobs = build_generation_matrix_jobs(
        args,
        retrievers=retrievers,
        models=available_models,
        policies=policies,
        budgets=budgets,
        adaptive_profiles=adaptive_profiles,
        matrix_dir=matrix_dir,
    )
    manifest = _manifest(
        args,
        retrievers=retrievers,
        policies=policies,
        budgets=budgets,
        adaptive_profiles=adaptive_profiles,
        requested_models=requested_models,
        available_models=available_models,
        skipped_models=skipped_models,
        jobs=jobs,
    )
    if args.dry_run:
        for job in jobs:
            print(" ".join(job.command), flush=True)
        print(json.dumps(manifest, indent=2), flush=True)
        return 0

    matrix_dir.mkdir(parents=True, exist_ok=True)
    failures: list[dict[str, object]] = []
    skipped_existing: list[dict[str, object]] = []
    for job in jobs:
        if not args.rerun_existing and _job_has_completed_metrics(job.output_dir):
            skipped = _job_manifest_entry(job)
            skipped["reason"] = "existing-metrics"
            skipped_existing.append(skipped)
            print(f"SKIP existing {job.output_dir}", flush=True)
            continue
        job.output_dir.mkdir(parents=True, exist_ok=True)
        print(" ".join(job.command), flush=True)
        try:
            completed = subprocess.run(
                job.command,
                check=False,
                timeout=args.job_timeout_s if args.job_timeout_s > 0 else None,
            )
            returncode: int | str = completed.returncode
        except subprocess.TimeoutExpired:
            returncode = "timeout"
        if returncode:
            failure = _job_manifest_entry(job)
            failure["returncode"] = returncode
            failure["timed_out"] = returncode == "timeout"
            failures.append(failure)
            if not args.continue_on_error:
                manifest["failures"] = failures
                manifest["skipped_existing"] = skipped_existing
                (matrix_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
                raise SystemExit(124 if returncode == "timeout" else int(returncode))
    manifest["failures"] = failures
    manifest["skipped_existing"] = skipped_existing
    (matrix_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return 1 if failures and not args.continue_on_error else 0


def build_generation_matrix_jobs(
    args: argparse.Namespace,
    *,
    retrievers: list[str],
    models: list[GenerationModelConfig],
    policies: list[str],
    budgets: list[int],
    adaptive_profiles: list[str],
    matrix_dir: Path,
) -> list[GenerationMatrixJob]:
    jobs: list[GenerationMatrixJob] = []
    for retriever in retrievers:
        for model in models:
            for policy in policies:
                for budget in budgets:
                    profiles = adaptive_profiles if policy == "adaptive-heuristic" else [None]
                    for adaptive_profile in profiles:
                        output_dir = matrix_dir / _job_slug(args.bench, retriever, model.model_id, policy, budget, adaptive_profile)
                        command = _build_run_command(
                            args,
                            retriever=retriever,
                            model=model,
                            policy=policy,
                            budget=budget,
                            adaptive_profile=adaptive_profile,
                            output_dir=output_dir,
                        )
                        jobs.append(
                            GenerationMatrixJob(
                                retriever=retriever,
                                model=model,
                                policy=policy,
                                budget=budget,
                                adaptive_profile=adaptive_profile,
                                output_dir=output_dir,
                                command=command,
                            )
                        )
    return jobs


def _build_run_command(
    args: argparse.Namespace,
    *,
    retriever: str,
    model: GenerationModelConfig,
    policy: str,
    budget: int,
    adaptive_profile: str | None,
    output_dir: Path,
) -> list[str]:
    command = [
        sys.executable,
        "-m",
        "rag_bench",
        "run",
        "--bench",
        args.bench,
        "--retrievers",
        retriever,
        "--top-k",
        str(args.top_k),
        "--limit",
        str(args.limit),
        "--output-dir",
        str(output_dir),
        "--context-policy",
        policy,
        "--context-budget-chars",
        str(budget),
        "--max-context-chars",
        str(args.max_context_chars),
        "--max-completion-tokens",
        str(args.max_completion_tokens),
        "--max-consecutive-errors",
        str(args.max_consecutive_errors),
        "--temperature",
        str(args.temperature),
        "--max-retries",
        str(args.max_retries),
        "--rate-limit-scope",
        args.rate_limit_scope,
        "--model",
        model.model,
        "--generation-provider",
        model.provider,
        "--generation-model-role",
        model.role,
        "--vector-model",
        args.vector_model,
        "--kv-profile",
        args.kv_profile,
    ]
    if model.provider == "groq":
        command.extend(["--groq-keys-path", str(args.groq_keys_path)])
        command.extend(["--key-tpm", str(args.key_tpm), "--key-rpm", str(args.key_rpm)])
    elif model.provider == "mimo":
        api_key_var = model.api_key_env or args.mimo_api_key_var
        command.extend(
            [
                "--mimo-env-file",
                str(args.mimo_env_file),
                "--mimo-api-key-var",
                api_key_var,
                "--mimo-base-url",
                _resolve_mimo_base_url(model, args),
                "--key-tpm",
                str(args.mimo_key_tpm),
                "--key-rpm",
                str(args.mimo_key_rpm),
            ]
        )
    if args.per_doc_budget_chars is not None:
        command.extend(["--per-doc-budget-chars", str(args.per_doc_budget_chars)])
    if policy == "adaptive-heuristic":
        command.extend(["--adaptive-medium-budget", str(budget)])
        if adaptive_profile is not None:
            command.extend(["--adaptive-profile", adaptive_profile])
    if args.disable_kv_estimate:
        command.append("--disable-kv-estimate")
    if args.allow_large_bench:
        command.append("--allow-large-bench")
    return command


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run BudgetRAG generation experiments across providers and models.")
    parser.add_argument("--bench", default="scifact")
    parser.add_argument("--limit", type=int, default=20)
    parser.add_argument("--retrievers", default="bm25")
    parser.add_argument("--models", default="groq_llama8b,groq_qwen32b,mimo_v25_pro")
    parser.add_argument("--model-config", type=Path, default=DEFAULT_MODEL_CONFIG_PATH)
    parser.add_argument("--context-policies", default="legacy,evidence-aware,adaptive-heuristic")
    parser.add_argument("--context-budgets", default="1000,2000,4000")
    parser.add_argument("--adaptive-profiles", default="balanced")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--output-dir", type=Path, default=Path("benchmark_results/budgetrag"))
    parser.add_argument("--run-name", default=None, help="Optional generation matrix run name used as a subdirectory.")
    parser.add_argument("--dry-run", action="store_true", help="Print commands and manifest without running benchmarks.")
    parser.add_argument("--continue-on-error", action="store_true", help="Continue running the matrix after a failed job.")
    parser.add_argument(
        "--job-timeout-s",
        type=int,
        default=0,
        help="Maximum seconds for each child rag-bench run. Defaults to 0, which disables the timeout.",
    )
    parser.add_argument(
        "--rerun-existing",
        action="store_true",
        help="Rerun jobs even when their output directory already contains completed metrics.",
    )
    parser.add_argument("--max-context-chars", type=int, default=12_000)
    parser.add_argument("--per-doc-budget-chars", type=int, default=None)
    parser.add_argument("--max-completion-tokens", type=int, default=256)
    parser.add_argument("--max-consecutive-errors", type=int, default=1)
    parser.add_argument("--max-retries", type=int, default=2)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--vector-model", default="sentence-transformers/all-MiniLM-L6-v2")
    parser.add_argument("--kv-profile", default="generic-small")
    parser.add_argument("--disable-kv-estimate", action="store_true")
    parser.add_argument("--allow-large-bench", action="store_true")
    parser.add_argument("--groq-keys-path", type=Path, default=Path(".secrets/groq_key.env"))
    parser.add_argument("--key-tpm", type=int, default=6000)
    parser.add_argument("--key-rpm", type=int, default=30)
    parser.add_argument("--rate-limit-scope", choices=("per-key", "shared"), default="per-key")
    parser.add_argument("--mimo-env-file", type=Path, default=Path(".secrets/.env"))
    parser.add_argument("--mimo-api-key-var", default="MIMO_API_KEY")
    parser.add_argument("--mimo-base-url", default="https://token-plan-sgp.xiaomimimo.com/v1")
    parser.add_argument("--mimo-key-tpm", type=int, default=0)
    parser.add_argument("--mimo-key-rpm", type=int, default=0)
    return parser


def _validate_matrix_inputs(
    retrievers: list[str],
    policies: list[str],
    budgets: list[int],
    adaptive_profiles: list[str],
    requested_models: list[str],
) -> None:
    if not retrievers:
        raise SystemExit("--retrievers must include at least one value")
    if not requested_models:
        raise SystemExit("--models must include at least one value")
    if not policies:
        raise SystemExit("--context-policies must include at least one value")
    if not budgets or any(value <= 0 for value in budgets):
        raise SystemExit("--context-budgets must include positive integers")
    if not adaptive_profiles:
        raise SystemExit("--adaptive-profiles must include at least one value")
    unknown_profiles = sorted(set(adaptive_profiles) - set(ADAPTIVE_PROFILES))
    if unknown_profiles:
        allowed = ", ".join(ADAPTIVE_PROFILES)
        unknown = ", ".join(unknown_profiles)
        raise SystemExit(f"--adaptive-profiles contains unknown values: {unknown}. Expected one of: {allowed}")


def _filter_available_models(
    models: list[GenerationModelConfig],
    args: argparse.Namespace,
) -> tuple[list[GenerationModelConfig], list[dict[str, str]]]:
    available: list[GenerationModelConfig] = []
    skipped: list[dict[str, str]] = []
    for model in models:
        reason = _model_skip_reason(model, args)
        if reason:
            skipped.append(
                {
                    "model_id": model.model_id,
                    "provider": model.provider,
                    "model": model.model,
                    "role": model.role,
                    "reason": reason,
                }
            )
            continue
        available.append(model)
    return available, skipped


def _model_skip_reason(model: GenerationModelConfig, args: argparse.Namespace) -> str | None:
    if model.provider == "groq":
        try:
            load_groq_keys(args.groq_keys_path)
        except SecretFormatError as exc:
            return f"groq-keys-unavailable:{exc}"
        return None
    if model.provider == "mimo":
        api_key_var = model.api_key_env or args.mimo_api_key_var
        if _has_env_value(api_key_var, args.mimo_env_file):
            return None
        return f"{api_key_var}-not-configured"
    return f"unsupported-provider:{model.provider}"


def _has_env_value(variable: str, env_file: Path) -> bool:
    if os.environ.get(variable):
        return True
    try:
        values = load_env_values(env_file)
    except SecretFormatError:
        return False
    return bool(values.get(variable))


def _resolve_mimo_base_url(model: GenerationModelConfig, args: argparse.Namespace) -> str:
    if model.base_url_env and os.environ.get(model.base_url_env):
        return str(os.environ[model.base_url_env])
    if model.base_url:
        return model.base_url
    return args.mimo_base_url


def _manifest(
    args: argparse.Namespace,
    *,
    retrievers: list[str],
    policies: list[str],
    budgets: list[int],
    adaptive_profiles: list[str],
    requested_models: list[str],
    available_models: list[GenerationModelConfig],
    skipped_models: list[dict[str, str]],
    jobs: list[GenerationMatrixJob],
) -> dict[str, object]:
    return {
        "bench": args.bench,
        "limit": args.limit,
        "retrievers": retrievers,
        "requested_models": requested_models,
        "available_models": [asdict(model) for model in available_models],
        "skipped_models": skipped_models,
        "context_policies": policies,
        "context_budgets": budgets,
        "adaptive_profiles": adaptive_profiles,
        "generation_note": (
            "MiMo models are skipped when their API key is not configured. Raw generation outputs stay under "
            "benchmark_results/budgetrag and should not be committed."
        ),
        "top_k": args.top_k,
        "kv_profile": args.kv_profile,
        "run_name": args.run_name,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "commands": [
            _job_manifest_entry(job)
            for job in jobs
        ],
    }


def _job_manifest_entry(job: GenerationMatrixJob) -> dict[str, object]:
    return {
        "retriever": job.retriever,
        "model_id": job.model.model_id,
        "provider": job.model.provider,
        "model": job.model.model,
        "model_role": job.model.role,
        "context_policy": job.policy,
        "adaptive_profile": job.adaptive_profile,
        "context_budget_chars": job.budget,
        "output_dir": str(job.output_dir),
        "command": job.command,
    }


def _job_has_completed_metrics(output_dir: Path) -> bool:
    return any(output_dir.glob("*/metrics.json"))


def _job_slug(
    bench: str,
    retriever: str,
    model_id: str,
    policy: str,
    budget: int,
    adaptive_profile: str | None = None,
) -> str:
    profile = f"__{adaptive_profile}" if adaptive_profile else ""
    raw = f"{bench}__{retriever}__{model_id}__{policy}{profile}__budget{budget}"
    return "".join(char if char.isalnum() or char in "._-" else "-" for char in raw)


def _split_csv(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


if __name__ == "__main__":
    raise SystemExit(main())
