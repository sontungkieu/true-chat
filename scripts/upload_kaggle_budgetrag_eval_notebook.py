#!/usr/bin/env python3
from __future__ import annotations

import argparse
import base64
import json
import shutil
import sys
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from upload_kaggle_rag_proxy_notebook import (
    code_cell,
    has_tracked_changes,
    load_kaggle_credential,
    local_head_commit,
    markdown_cell,
    run_kaggle,
    slugify,
    write_kaggle_config,
)


DEFAULT_ACCOUNT = "codemaivanngu"
DEFAULT_CREDENTIALS_PATH = Path("/home/tung/all-kaggle.json")
DEFAULT_REPO_URL = "https://github.com/sontungkieu/true-chat.git"
DEFAULT_OUTPUT_ROOT = Path("benchmark_results/budgetrag/phase1c3_hotpotqa_kaggle")
DEFAULT_POLL_TIMEOUT_S = 12 * 60 * 60
DEFAULT_PROVIDER = "mimo"
DEFAULT_MODEL = "mimo-v2.5"
DEFAULT_MODEL_ROLE = "long-context-judge-generator"
DEFAULT_RAGAS_MODEL = "mimo-v2.5"
DEFAULT_CONTEXT_POLICIES = "legacy,evidence-aware,adaptive-heuristic"
DEFAULT_CONTEXT_BUDGETS = "4000,8000,16000,32000"
DEFAULT_ADAPTIVE_PROFILES = "balanced,aggressive"


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    repo_root = Path(args.repo_root).resolve()
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M")
    run_name = args.run_name or f"{timestamp}_hotpotqa_kaggle"
    expected_commit = args.expected_commit or local_head_commit(repo_root)
    if not args.allow_dirty and has_tracked_changes(repo_root):
        raise SystemExit(
            "Working tree has tracked changes. Commit and push first, or rerun with --allow-dirty for a deliberate dry run."
        )

    credential = load_kaggle_credential(Path(args.credentials).expanduser(), args.account)
    slug = args.slug or slugify(f"budgetrag-hotpotqa-eval-{args.account}-{timestamp}")
    title = args.title or f"BudgetRAG HotpotQA Eval {args.account} {timestamp}"
    kernel_id = f"{credential.username}/{slug}"
    local_output_dir = Path(args.local_output_dir or (DEFAULT_OUTPUT_ROOT / run_name)).resolve()

    cleanup: tempfile.TemporaryDirectory[str] | None
    if args.keep_staging_dir:
        staging_dir = Path(args.keep_staging_dir).expanduser().resolve()
        if staging_dir.exists():
            shutil.rmtree(staging_dir)
        staging_dir.mkdir(parents=True)
        cleanup = None
    else:
        cleanup = tempfile.TemporaryDirectory(prefix="true-chat-hotpotqa-kaggle-")
        staging_dir = Path(cleanup.name)

    try:
        mimo_env_b64 = (
            read_mimo_env_b64(
                repo_root,
                args.mimo_env_file,
                api_key_var=args.mimo_api_key_var,
            )
            if args.embed_mimo_env or args.provider == "mimo" or not args.skip_ragas
            else None
        )
        groq_env_b64 = (
            read_groq_env_b64(
                repo_root,
                args.groq_keys_file,
                key_alias=args.groq_key_alias,
            )
            if args.provider == "groq" or args.embed_groq_key
            else None
        )
        write_staging_files(
            staging_dir,
            kernel_id=kernel_id,
            title=title,
            repo_url=args.repo_url,
            repo_ref=args.repo_ref,
            expected_commit=expected_commit,
            run_name=run_name,
            limit=args.limit,
            top_k=args.top_k,
            max_action_rows=args.max_action_rows,
            context_policies=args.context_policies,
            context_budgets=args.context_budgets,
            adaptive_profiles=args.adaptive_profiles,
            provider=args.provider,
            model=args.model,
            model_role=args.model_role,
            key_tpm=args.key_tpm,
            key_rpm=args.key_rpm,
            ragas_model=args.ragas_model,
            ragas_samples_per_action=args.ragas_samples_per_action,
            mimo_secret_name=args.mimo_secret_name,
            mimo_env_b64=mimo_env_b64,
            groq_key_alias=args.groq_key_alias,
            groq_env_b64=groq_env_b64,
            skip_ragas=args.skip_ragas,
        )
        if args.no_push:
            print(f"Rendered Kaggle notebook staging at: {staging_dir}")
            print(f"Kernel id would be: {kernel_id}")
            return 0

        with tempfile.TemporaryDirectory(prefix="kaggle-config-") as kaggle_config_dir:
            write_kaggle_config(Path(kaggle_config_dir), credential)
            run_kaggle(["kernels", "push", "-p", str(staging_dir)], kaggle_config_dir)
            status = wait_for_kernel(
                kernel_id,
                kaggle_config_dir,
                poll_interval_s=args.poll_interval_s,
                poll_timeout_s=args.poll_timeout_s,
                no_wait=args.no_wait,
            )
            if args.no_wait:
                print(f"Uploaded Kaggle notebook: {kernel_id}")
                return 0
            if status and is_success_status(status):
                local_output_dir.mkdir(parents=True, exist_ok=True)
                run_kaggle(["kernels", "output", kernel_id, "-p", str(local_output_dir)], kaggle_config_dir)
                print(f"Downloaded Kaggle outputs to: {local_output_dir}")
            elif status:
                raise SystemExit(f"Kaggle kernel did not complete successfully:\n{status.strip()}")
        print(f"Uploaded Kaggle notebook: {kernel_id}")
        return 0
    finally:
        if cleanup is not None:
            cleanup.cleanup()


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Upload a private Kaggle notebook for HotpotQA BudgetRAG eval.")
    parser.add_argument("--account", default=DEFAULT_ACCOUNT)
    parser.add_argument("--credentials", default=str(DEFAULT_CREDENTIALS_PATH))
    parser.add_argument("--repo-root", default=".")
    parser.add_argument("--repo-url", default=DEFAULT_REPO_URL)
    parser.add_argument("--repo-ref", default="main")
    parser.add_argument("--expected-commit", default=None)
    parser.add_argument("--allow-dirty", action="store_true")
    parser.add_argument("--title", default=None)
    parser.add_argument("--slug", default=None)
    parser.add_argument("--run-name", default=None)
    parser.add_argument("--limit", type=int, default=50)
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--max-action-rows", type=int, default=None)
    parser.add_argument("--context-policies", default=DEFAULT_CONTEXT_POLICIES)
    parser.add_argument("--context-budgets", default=DEFAULT_CONTEXT_BUDGETS)
    parser.add_argument("--adaptive-profiles", default=DEFAULT_ADAPTIVE_PROFILES)
    parser.add_argument("--provider", choices=("mimo", "groq"), default=DEFAULT_PROVIDER)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--model-role", default=DEFAULT_MODEL_ROLE)
    parser.add_argument("--key-tpm", type=int, default=0)
    parser.add_argument("--key-rpm", type=int, default=0)
    parser.add_argument("--ragas-model", default=DEFAULT_RAGAS_MODEL)
    parser.add_argument("--ragas-samples-per-action", type=int, default=5)
    parser.add_argument("--mimo-secret-name", default="MIMO_API_KEY")
    parser.add_argument("--embed-mimo-env", action="store_true")
    parser.add_argument("--mimo-env-file", default=".secrets/.env")
    parser.add_argument("--mimo-api-key-var", default="MIMO_API_KEY")
    parser.add_argument("--embed-groq-key", action="store_true")
    parser.add_argument("--groq-keys-file", default=".secrets/groq_key.env")
    parser.add_argument("--groq-key-alias", default=None)
    parser.add_argument("--skip-ragas", action="store_true")
    parser.add_argument("--local-output-dir", default=None)
    parser.add_argument("--poll-interval-s", type=int, default=60)
    parser.add_argument("--poll-timeout-s", type=int, default=DEFAULT_POLL_TIMEOUT_S)
    parser.add_argument("--no-wait", action="store_true")
    parser.add_argument("--keep-staging-dir", default=None)
    parser.add_argument("--no-push", action="store_true")
    return parser.parse_args(argv)


def write_staging_files(
    staging_dir: Path,
    *,
    kernel_id: str,
    title: str,
    repo_url: str,
    repo_ref: str,
    expected_commit: str,
    run_name: str,
    limit: int,
    top_k: int,
    max_action_rows: int | None,
    context_policies: str = DEFAULT_CONTEXT_POLICIES,
    context_budgets: str = DEFAULT_CONTEXT_BUDGETS,
    adaptive_profiles: str = DEFAULT_ADAPTIVE_PROFILES,
    provider: str = DEFAULT_PROVIDER,
    model: str = DEFAULT_MODEL,
    model_role: str = DEFAULT_MODEL_ROLE,
    key_tpm: int = 0,
    key_rpm: int = 0,
    ragas_model: str = DEFAULT_RAGAS_MODEL,
    ragas_samples_per_action: int,
    mimo_secret_name: str,
    mimo_env_b64: str | None = None,
    groq_key_alias: str | None = None,
    groq_env_b64: str | None = None,
    skip_ragas: bool,
) -> None:
    staging_dir.mkdir(parents=True, exist_ok=True)
    notebook_name = "hotpotqa_budgetrag_eval.ipynb"
    notebook = build_notebook(
        repo_url=repo_url,
        repo_ref=repo_ref,
        expected_commit=expected_commit,
        run_name=run_name,
        limit=limit,
        top_k=top_k,
        max_action_rows=max_action_rows,
        context_policies=context_policies,
        context_budgets=context_budgets,
        adaptive_profiles=adaptive_profiles,
        provider=provider,
        model=model,
        model_role=model_role,
        key_tpm=key_tpm,
        key_rpm=key_rpm,
        ragas_model=ragas_model,
        ragas_samples_per_action=ragas_samples_per_action,
        mimo_secret_name=mimo_secret_name,
        mimo_env_b64=mimo_env_b64,
        groq_key_alias=groq_key_alias,
        groq_env_b64=groq_env_b64,
        skip_ragas=skip_ragas,
    )
    (staging_dir / notebook_name).write_text(json.dumps(notebook, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    metadata = {
        "id": kernel_id,
        "title": title,
        "code_file": notebook_name,
        "language": "python",
        "kernel_type": "notebook",
        "is_private": "true",
        "enable_gpu": "false",
        "enable_tpu": "false",
        "enable_internet": "true",
        "dataset_sources": [],
        "competition_sources": [],
        "kernel_sources": [],
    }
    (staging_dir / "kernel-metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def build_notebook(
    *,
    repo_url: str,
    repo_ref: str,
    expected_commit: str,
    run_name: str,
    limit: int,
    top_k: int,
    max_action_rows: int | None,
    context_policies: str,
    context_budgets: str,
    adaptive_profiles: str,
    provider: str,
    model: str,
    model_role: str,
    key_tpm: int,
    key_rpm: int,
    ragas_model: str,
    ragas_samples_per_action: int,
    mimo_secret_name: str,
    mimo_env_b64: str | None,
    groq_key_alias: str | None,
    groq_env_b64: str | None,
    skip_ragas: bool,
) -> dict[str, Any]:
    command = [
        "uv",
        "run",
        "--frozen",
        "--extra",
        "vector",
        "--extra",
        "ragas",
        "python",
        "scripts/run_hotpotqa_cached_budgetrag_eval.py",
        "--limit",
        str(limit),
        "--top-k",
        str(top_k),
        "--output-dir",
        "/kaggle/working/phase1c3_hotpotqa_kaggle",
        "--run-name",
        run_name,
        "--context-policies",
        context_policies,
        "--context-budgets",
        context_budgets,
        "--adaptive-profiles",
        adaptive_profiles,
        "--provider",
        provider,
        "--model",
        model,
        "--model-role",
        model_role,
        "--key-tpm",
        str(key_tpm),
        "--key-rpm",
        str(key_rpm),
        "--ragas-model",
        ragas_model,
        "--ragas-samples-per-action",
        str(ragas_samples_per_action),
    ]
    if provider == "groq":
        command.extend(["--groq-keys-path", ".secrets/groq_key.env"])
        if groq_key_alias:
            command.extend(["--groq-key-alias", groq_key_alias])
    if max_action_rows is not None:
        command.extend(["--max-action-rows", str(max_action_rows)])
    if skip_ragas:
        command.append("--skip-ragas")
    cells = [
        markdown_cell(
            "# BudgetRAG HotpotQA Kaggle Eval\n\n"
            "This notebook clones True Chat, runs a cached HotpotQA BudgetRAG sampled eval, and leaves outputs under `/kaggle/working`.",
            cell_id="intro",
        ),
        code_cell(
            "from pathlib import Path\n"
            "import os, subprocess\n\n"
            f"REPO_URL = {repo_url!r}\n"
            f"REPO_REF = {repo_ref!r}\n"
            f"EXPECTED_COMMIT = {expected_commit!r}\n"
            f"PROVIDER = {provider!r}\n"
            f"MIMO_SECRET_NAME = {mimo_secret_name!r}\n"
            f"MIMO_ENV_B64 = {mimo_env_b64!r}\n"
            f"GROQ_ENV_B64 = {groq_env_b64!r}\n"
            f"RUN_COMMAND = {command!r}\n"
            "WORKDIR = Path('/kaggle/working')\n"
            "REPO_DIR = WORKDIR / 'true-chat'\n"
            "print('Repo:', REPO_URL, 'ref:', REPO_REF)\n"
            "print('Expected commit:', EXPECTED_COMMIT)\n"
            "print('Run command:', ' '.join(RUN_COMMAND))\n",
            cell_id="config",
        ),
        code_cell(
            "subprocess.run(['python', '-m', 'pip', 'install', '-q', 'uv'], check=True)\n"
            "if REPO_DIR.exists():\n"
            "    subprocess.run(['rm', '-rf', str(REPO_DIR)], check=True)\n"
            "subprocess.run(['git', 'clone', '--branch', REPO_REF, REPO_URL, str(REPO_DIR)], check=True)\n"
            "actual_commit = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=REPO_DIR, text=True).strip()\n"
            "print('Actual commit:', actual_commit)\n"
            "if actual_commit != EXPECTED_COMMIT:\n"
            "    raise RuntimeError(f'Commit mismatch: expected {EXPECTED_COMMIT}, got {actual_commit}. Push local commit before running this notebook.')\n",
            cell_id="clone-repo",
        ),
        code_cell(
            "os.chdir(REPO_DIR)\n"
            "subprocess.run(['uv', 'sync', '--frozen', '--extra', 'vector', '--extra', 'ragas', '--no-dev'], check=True)\n"
            "print('uv environment synced')\n",
            cell_id="sync-deps",
        ),
        code_cell(
            "secrets_dir = REPO_DIR / '.secrets'\n"
            "secrets_dir.mkdir(exist_ok=True)\n"
            "needs_mimo_key = PROVIDER == 'mimo' or '--skip-ragas' not in RUN_COMMAND\n"
            "if PROVIDER == 'groq':\n"
            "    if not GROQ_ENV_B64:\n"
            "        raise RuntimeError('Groq HotpotQA eval requires --embed-groq-key or a notebook-specific key writer.')\n"
            "    import base64\n"
            "    groq_env_text = base64.b64decode(GROQ_ENV_B64).decode('utf-8')\n"
            "    active_lines = [line for line in groq_env_text.splitlines() if line.strip() and not line.strip().startswith('#')]\n"
            "    if len(active_lines) != 1 or '=' not in active_lines[0]:\n"
            "        raise RuntimeError('Embedded Groq env must contain exactly one alias=value key line.')\n"
            "    (secrets_dir / 'groq_key.env').write_text(groq_env_text.rstrip() + '\\n')\n"
            "    print('Wrote embedded Groq key env to .secrets/groq_key.env without printing it')\n"
            "if MIMO_ENV_B64:\n"
            "    import base64\n"
            "    mimo_env_text = base64.b64decode(MIMO_ENV_B64).decode('utf-8')\n"
            "    if 'MIMO_API_KEY=' not in mimo_env_text:\n"
            "        raise RuntimeError('Embedded MiMo env is missing MIMO_API_KEY.')\n"
            "    (secrets_dir / '.env').write_text(mimo_env_text.rstrip() + '\\n')\n"
            "    print('Wrote embedded MiMo env to .secrets/.env without printing it')\n"
            "elif needs_mimo_key:\n"
            "    raise RuntimeError('MiMo key is required for this run and must be injected before upload; Kaggle secrets are not used.')\n"
            "else:\n"
            "    print('MiMo key not required for this run')\n",
            cell_id="write-mimo-secret",
        ),
        code_cell(
            "env = {**os.environ, 'PYTHONUNBUFFERED': '1'}\n"
            "subprocess.run(RUN_COMMAND, cwd=REPO_DIR, env=env, check=True)\n"
            "print('Eval completed')\n"
            "subprocess.run(['find', '/kaggle/working/phase1c3_hotpotqa_kaggle', '-maxdepth', '3', '-type', 'f', '-print'], check=False)\n",
            cell_id="run-eval",
        ),
    ]
    return {
        "cells": cells,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "pygments_lexer": "ipython3"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }


def read_mimo_env_b64(repo_root: Path, value: str | Path, *, api_key_var: str) -> str:
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = repo_root / path
    if not path.exists():
        raise SystemExit(f"MiMo env file not found: {path}")
    values = parse_env_file(path)
    api_key = values.get(api_key_var, "").strip()
    if not api_key:
        raise SystemExit(f"MiMo env file is missing {api_key_var}: {path}")
    lines = [f"MIMO_API_KEY={api_key}"]
    if values.get("MIMO_BASE_URL", "").strip():
        lines.append(f"MIMO_BASE_URL={values['MIMO_BASE_URL'].strip()}")
    return base64.b64encode(("\n".join(lines) + "\n").encode("utf-8")).decode("ascii")


def read_groq_env_b64(repo_root: Path, value: str | Path, *, key_alias: str | None) -> str:
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = repo_root / path
    if not path.exists():
        raise SystemExit(f"Groq key file not found: {path}")
    values = parse_env_file(path)
    if key_alias:
        api_key = values.get(key_alias, "").strip()
        if not api_key:
            raise SystemExit(f"Groq key file is missing alias {key_alias}: {path}")
        selected = {key_alias: api_key}
    else:
        selected = {alias: value for alias, value in values.items() if value.strip()}
        if not selected:
            raise SystemExit(f"Groq key file has no active key entries: {path}")
        if len(selected) > 1:
            first_alias = next(iter(selected))
            selected = {first_alias: selected[first_alias]}
    alias, api_key = next(iter(selected.items()))
    return base64.b64encode((f"{alias}={api_key.strip()}\n").encode("utf-8")).decode("ascii")


def parse_env_file(path: Path) -> dict[str, str]:
    values: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or "=" not in stripped:
            continue
        key, value = stripped.split("=", 1)
        values[key.strip()] = value.strip().strip("\"'")
    return values


def wait_for_kernel(
    kernel_id: str,
    kaggle_config_dir: str,
    *,
    poll_interval_s: int,
    poll_timeout_s: int,
    no_wait: bool,
) -> str:
    status = run_kaggle(["kernels", "status", kernel_id], kaggle_config_dir, capture=True)
    print(status.strip())
    if no_wait or is_terminal_status(status):
        return status
    deadline = time.time() + poll_timeout_s
    while time.time() < deadline:
        time.sleep(max(1, poll_interval_s))
        status = run_kaggle(["kernels", "status", kernel_id], kaggle_config_dir, capture=True)
        print(status.strip())
        if is_terminal_status(status):
            return status
    raise SystemExit(f"Timed out waiting for Kaggle kernel after {poll_timeout_s}s.")


def is_terminal_status(status: str) -> bool:
    text = status.lower()
    return any(marker in text for marker in ("complete", "error", "failed", "failure", "canceled", "cancelled"))


def is_success_status(status: str) -> bool:
    text = status.lower()
    return "complete" in text and not any(marker in text for marker in ("error", "failed", "failure"))


if __name__ == "__main__":
    raise SystemExit(main())
