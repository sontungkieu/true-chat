#!/usr/bin/env python3
from __future__ import annotations

import argparse
import base64
import json
import os
import re
import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


DEFAULT_ACCOUNT = "codemaivanngu"
DEFAULT_CREDENTIALS_PATH = Path(".secrets/all-kaggle.json")
DEFAULT_GROQ_KEYS_PATH = Path(".secrets/groq_key.env")
DEFAULT_MIMO_ENV_PATH = Path(".secrets/.env")
DEFAULT_UPLOAD_REGISTRY_PATH = Path(".secrets/kaggle_notebooks.jsonl")
DEFAULT_REPO_URL = "https://github.com/sontungkieu/true-chat.git"
DEFAULT_HOSTNAME = "https://chatpb.ccat.io.vn"
DEFAULT_PROXY_STARTUP_TIMEOUT_S = 900
DEFAULT_MIMO_MODELS = "mimo-v2.5"
DEFAULT_DICTIONARY_ARTIFACT = Path("runs/pb_dictionary_base_supp2021_prod_graph")
DEFAULT_AVAILABLE_RETRIEVERS = "bm25,tfidf,keyword-match,agent,graph-bm25,dictionary-graph,image-digits"
DEFAULT_SERVE_BENCH = "scifact"
DEFAULT_DICTIONARY_SERVE_BENCH = "none"
DEFAULT_SERVE_RETRIEVER = "bm25"
DEFAULT_DICTIONARY_SERVE_RETRIEVER = "dictionary-graph"
DEFAULT_SERVE_MODEL_ID = "rag-scifact-bm25"
DEFAULT_DICTIONARY_SERVE_MODEL_ID = "rag-dictionary-graph"
TOKEN_ENV_NAMES = (
    "CLOUDFLARE_TUNNEL_TOKEN",
    "CF_TUNNEL_TOKEN",
    "CLOUDFLARED_TOKEN",
    "TUNNEL_TOKEN",
)


@dataclass(frozen=True)
class KaggleCredential:
    username: str
    key: str


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    repo_root = Path(args.repo_root).resolve()
    registry_path = resolve_repo_path(repo_root, args.registry)

    if args.list_uploads:
        list_upload_registry(registry_path)
        return 0

    credentials_path = Path(args.credentials).expanduser()
    if not credentials_path.is_absolute():
        credentials_path = repo_root / credentials_path

    credential = load_kaggle_credential(credentials_path, args.account)

    delete_targets = list(args.delete_upload or [])
    if args.delete_all_uploads:
        delete_targets.extend(active_registry_kernel_ids(registry_path))
    if delete_targets:
        with tempfile.TemporaryDirectory(prefix="kaggle-config-") as kaggle_config_dir:
            write_kaggle_config(Path(kaggle_config_dir), credential)
            for kernel_id in dict.fromkeys(delete_targets):
                run_kaggle(["kernels", "delete", "-y", kernel_id], kaggle_config_dir)
                mark_registry_deleted(registry_path, kernel_id)
                print(f"Deleted Kaggle notebook: {kernel_id}")
        return 0

    cloudflare_token = resolve_cloudflare_token(args, repo_root)
    if args.proxy_startup_timeout_s <= 0:
        raise SystemExit("--proxy-startup-timeout-s must be positive.")
    expected_commit = args.expected_commit or local_head_commit(repo_root)
    if not args.allow_dirty and has_tracked_changes(repo_root):
        raise SystemExit(
            "Working tree has tracked changes. Commit them first so EXPECTED_COMMIT represents local code, "
            "or rerun with --allow-dirty."
        )

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M")
    slug = args.slug or slugify(f"true-chat-rag-proxy-{args.account}-{timestamp}")
    title = args.title or (slug if args.slug else f"True Chat RAG Proxy {args.account} {timestamp}")
    kernel_id = f"{credential.username}/{slug}"
    groq_key_env_b64 = read_groq_key_env_b64(repo_root, args.groq_keys_file) if args.embed_groq_keys else None
    mimo_env_b64 = read_mimo_env_b64(repo_root, args.mimo_env_file) if args.embed_mimo_env else None
    enable_mimo = bool(args.enable_mimo or mimo_env_b64)
    available_retrievers = args.available_retrievers
    if available_retrievers is None and args.dictionary_dataset_source:
        available_retrievers = DEFAULT_AVAILABLE_RETRIEVERS
    serve_retriever = args.serve_retriever or (
        DEFAULT_DICTIONARY_SERVE_RETRIEVER if args.dictionary_dataset_source else DEFAULT_SERVE_RETRIEVER
    )
    serve_bench = args.serve_bench or (
        DEFAULT_DICTIONARY_SERVE_BENCH if args.dictionary_dataset_source else DEFAULT_SERVE_BENCH
    )
    serve_model_id = args.serve_model_id or (
        DEFAULT_DICTIONARY_SERVE_MODEL_ID if args.dictionary_dataset_source else DEFAULT_SERVE_MODEL_ID
    )
    dataset_sources = dedupe_nonempty([*args.dataset_source, args.dictionary_dataset_source])

    if args.keep_staging_dir:
        staging_dir = Path(args.keep_staging_dir).expanduser().resolve()
        if staging_dir.exists():
            shutil.rmtree(staging_dir)
        staging_dir.mkdir(parents=True)
        cleanup = None
    else:
        cleanup = tempfile.TemporaryDirectory(prefix="true-chat-kaggle-")
        staging_dir = Path(cleanup.name)

    try:
        write_staging_files(
            staging_dir,
            kernel_id=kernel_id,
            title=title,
            repo_url=args.repo_url,
            repo_ref=args.repo_ref,
            expected_commit=expected_commit,
            cloudflare_token=cloudflare_token,
            hostname=args.hostname,
            proxy_startup_timeout_s=args.proxy_startup_timeout_s,
            groq_key_env_b64=groq_key_env_b64,
            mimo_env_b64=mimo_env_b64,
            dataset_sources=dataset_sources,
            dictionary_dataset_source=args.dictionary_dataset_source,
            dictionary_artifact=args.dictionary_artifact,
            dictionary_required=args.dictionary_required,
            available_retrievers=available_retrievers,
            serve_bench=serve_bench,
            serve_retriever=serve_retriever,
            serve_model_id=serve_model_id,
            allow_external_semi_private=args.allow_external_semi_private,
            enable_mimo=enable_mimo,
            mimo_models=args.mimo_models,
        )
        if args.no_push:
            print(f"Rendered Kaggle notebook staging at: {staging_dir}")
            print(f"Kernel id would be: {kernel_id}")
            print("Cloudflare token and embedded provider secrets were injected into the staged notebook but were not printed.")
            return 0

        with tempfile.TemporaryDirectory(prefix="kaggle-config-") as kaggle_config_dir:
            write_kaggle_config(Path(kaggle_config_dir), credential)
            run_kaggle(["kernels", "push", "-p", str(staging_dir)], kaggle_config_dir)
            status = run_kaggle(["kernels", "status", kernel_id], kaggle_config_dir, capture=True)
        append_upload_registry(
            registry_path,
            {
                "created_at": datetime.now(timezone.utc).isoformat(),
                "account": args.account,
                "kernel_id": kernel_id,
                "title": title,
                "repo_url": args.repo_url,
                "repo_ref": args.repo_ref,
                "expected_commit": expected_commit,
                "hostname": args.hostname,
                "proxy_startup_timeout_s": args.proxy_startup_timeout_s,
                "embedded_groq_keys": bool(groq_key_env_b64),
                "embedded_mimo_env": bool(mimo_env_b64),
                "dataset_sources": dataset_sources,
                "dictionary_dataset_source": args.dictionary_dataset_source,
                "dictionary_artifact": args.dictionary_artifact,
                "dictionary_required": bool(args.dictionary_required),
                "available_retrievers": available_retrievers,
                "serve_bench": serve_bench,
                "serve_retriever": serve_retriever,
                "serve_model_id": serve_model_id,
                "allow_external_semi_private": bool(args.allow_external_semi_private),
                "enable_mimo": enable_mimo,
                "mimo_models": args.mimo_models if enable_mimo else "",
            },
        )
        print(f"Uploaded Kaggle notebook: {kernel_id}")
        if status:
            print(status.strip())
        return 0
    finally:
        if cleanup is not None:
            cleanup.cleanup()


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create and upload a Kaggle notebook that runs the True Chat RAG proxy through a Cloudflare named tunnel.",
    )
    parser.add_argument("--account", default=DEFAULT_ACCOUNT, help="Kaggle account key in the credentials file.")
    parser.add_argument("--credentials", default=str(DEFAULT_CREDENTIALS_PATH), help="Path to all-kaggle credential JSON/JSONL.")
    parser.add_argument("--registry", default=str(DEFAULT_UPLOAD_REGISTRY_PATH), help="Local JSONL registry of uploaded Kaggle notebooks.")
    parser.add_argument("--repo-root", default=".", help="Local repo root used to read the expected commit.")
    parser.add_argument("--repo-url", default=DEFAULT_REPO_URL, help="Git URL cloned by the Kaggle notebook.")
    parser.add_argument("--repo-ref", default="main", help="Git branch/tag/commit checked out by the Kaggle notebook.")
    parser.add_argument("--expected-commit", default=None, help="Commit hash the notebook must see after clone. Defaults to local HEAD.")
    parser.add_argument("--allow-dirty", action="store_true", help="Allow upload even when tracked local files are modified.")
    parser.add_argument("--hostname", default=DEFAULT_HOSTNAME, help="Public Cloudflare hostname to print in the notebook.")
    parser.add_argument(
        "--proxy-startup-timeout-s",
        type=int,
        default=DEFAULT_PROXY_STARTUP_TIMEOUT_S,
        help="Seconds the Kaggle notebook waits for rag-bench serve to become healthy.",
    )
    parser.add_argument("--title", default=None, help="Kaggle notebook title.")
    parser.add_argument("--slug", default=None, help="Kaggle notebook slug. Defaults to a timestamped slug.")
    parser.add_argument("--cloudflare-token", default=None, help="Cloudflare tunnel token. Prefer env/file to avoid shell history.")
    parser.add_argument("--cloudflare-token-file", default=None, help="File containing the Cloudflare tunnel token.")
    parser.add_argument(
        "--dataset-source",
        action="append",
        default=[],
        help="Kaggle dataset source slug to attach to the notebook metadata, e.g. owner/dataset.",
    )
    parser.add_argument(
        "--dictionary-dataset-source",
        default=None,
        help="Kaggle dataset slug containing the dictionary runtime artifact. Also added to dataset_sources.",
    )
    parser.add_argument(
        "--dictionary-artifact",
        default=str(DEFAULT_DICTIONARY_ARTIFACT),
        help="Repo-relative path where the notebook copies the dictionary artifact before serving.",
    )
    parser.add_argument(
        "--dictionary-required",
        action="store_true",
        help="Pass --dictionary-required so Kaggle startup fails if the dictionary artifact is missing.",
    )
    parser.add_argument(
        "--available-retrievers",
        default=None,
        help=f"Comma-separated retriever ids exposed by the UI. For full dictionary deploys use: {DEFAULT_AVAILABLE_RETRIEVERS}",
    )
    parser.add_argument(
        "--serve-bench",
        default=None,
        help=(
            "Benchmark corpus loaded by rag-bench serve. Defaults to scifact normally, "
            "or none for dictionary runtime deploys."
        ),
    )
    parser.add_argument(
        "--serve-retriever",
        default=None,
        help=(
            "Default retriever passed to rag-bench serve. Defaults to bm25 normally, "
            "or dictionary-graph when --dictionary-dataset-source is attached."
        ),
    )
    parser.add_argument(
        "--serve-model-id",
        default=None,
        help="Model id exposed by the proxy. Defaults to rag-dictionary-graph for dictionary runtime deploys.",
    )
    parser.add_argument(
        "--allow-external-semi-private",
        action="store_true",
        help="Pass --allow-external-semi-private to the Kaggle proxy for approved semi-private external generation.",
    )
    parser.add_argument(
        "--embed-groq-keys",
        action="store_true",
        help="Embed .secrets/groq_key.env into the generated private notebook instead of using Kaggle Secrets.",
    )
    parser.add_argument("--groq-keys-file", default=str(DEFAULT_GROQ_KEYS_PATH), help="Groq key env file embedded when --embed-groq-keys is used.")
    parser.add_argument("--enable-mimo", action="store_true", help="Pass --enable-mimo to expose MiMo models in the Kaggle proxy.")
    parser.add_argument("--mimo-models", default=DEFAULT_MIMO_MODELS, help="Comma-separated MiMo model ids passed to rag-bench serve.")
    parser.add_argument(
        "--embed-mimo-env",
        action="store_true",
        help="Embed MIMO_API_KEY and optional MIMO_BASE_URL from --mimo-env-file into the generated private notebook.",
    )
    parser.add_argument("--mimo-env-file", default=str(DEFAULT_MIMO_ENV_PATH), help="Env file read when --embed-mimo-env is used.")
    parser.add_argument(
        "--env-file",
        default=".secrets/.env",
        help="Optional env file searched for CLOUDFLARE_TUNNEL_TOKEN/CF_TUNNEL_TOKEN/TUNNEL_TOKEN.",
    )
    parser.add_argument("--keep-staging-dir", default=None, help="Keep generated notebook/metadata in this directory.")
    parser.add_argument("--no-push", action="store_true", help="Render staging files but do not call kaggle kernels push.")
    parser.add_argument("--list-uploads", action="store_true", help="List notebook ids recorded in the local registry.")
    parser.add_argument("--delete-upload", action="append", default=[], help="Delete one recorded Kaggle notebook id, e.g. owner/slug.")
    parser.add_argument("--delete-all-uploads", action="store_true", help="Delete all non-deleted notebook ids in the local registry.")
    return parser.parse_args(argv)


def load_kaggle_credential(path: Path, account: str) -> KaggleCredential:
    if not path.exists():
        raise SystemExit(f"Kaggle credential file not found: {path}")
    records = load_json_or_jsonl(path)
    for record in records:
        credential = credential_from_record(record, account)
        if credential is not None:
            return credential
    raise SystemExit(f"No Kaggle credential found for account '{account}' in {path}")


def load_json_or_jsonl(path: Path) -> list[Any]:
    text = path.read_text(encoding="utf-8")
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        records: list[Any] = []
        for line in text.splitlines():
            stripped = line.strip()
            if not stripped or not stripped.startswith(("{", "[")):
                continue
            records.append(json.loads(stripped))
        if not records:
            raise SystemExit(f"Unsupported credential file format: {path}")
        return records
    if isinstance(parsed, list):
        return parsed
    if isinstance(parsed, dict):
        return [parsed]
    raise SystemExit(f"Unsupported credential file format: {path}")


def credential_from_record(record: Any, account: str) -> KaggleCredential | None:
    if not isinstance(record, dict):
        return None
    if account in record and isinstance(record[account], dict):
        return credential_from_mapping(record[account], fallback_username=account)
    names = {
        str(record.get("account", "")),
        str(record.get("name", "")),
        str(record.get("owner", "")),
        str(record.get("username", "")),
        str(record.get("user", "")),
    }
    if account in names:
        return credential_from_mapping(record, fallback_username=account)
    return None


def credential_from_mapping(mapping: dict[str, Any], *, fallback_username: str) -> KaggleCredential:
    username = str(mapping.get("username") or mapping.get("user") or mapping.get("owner") or fallback_username).strip()
    key = str(mapping.get("key") or mapping.get("api_key") or mapping.get("token") or "").strip()
    if not username or not key:
        raise SystemExit("Kaggle credential entry is missing username/key fields.")
    return KaggleCredential(username=username, key=key)


def resolve_cloudflare_token(args: argparse.Namespace, repo_root: Path) -> str:
    if args.cloudflare_token:
        return args.cloudflare_token.strip()
    if args.cloudflare_token_file:
        token_path = Path(args.cloudflare_token_file).expanduser()
        if not token_path.is_absolute():
            token_path = repo_root / token_path
        if token_path.exists():
            return token_path.read_text(encoding="utf-8").strip()
    for name in TOKEN_ENV_NAMES:
        value = os.getenv(name)
        if value:
            return value.strip()
    env_file = Path(args.env_file).expanduser()
    if not env_file.is_absolute():
        env_file = repo_root / env_file
    if env_file.exists():
        values = parse_env_file(env_file)
        for name in TOKEN_ENV_NAMES:
            if values.get(name):
                return values[name].strip()
    raise SystemExit(
        "Cloudflare tunnel token is required. Provide --cloudflare-token-file, "
        "--cloudflare-token, or CLOUDFLARE_TUNNEL_TOKEN."
    )


def parse_env_file(path: Path) -> dict[str, str]:
    values: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or "=" not in stripped:
            continue
        key, value = stripped.split("=", 1)
        key = key.strip()
        if key.startswith("export "):
            key = key[len("export ") :].strip()
        values[key] = value.strip().strip("\"'")
    return values


def resolve_repo_path(repo_root: Path, value: str | Path) -> Path:
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = repo_root / path
    return path


def read_groq_key_env_b64(repo_root: Path, value: str | Path) -> str:
    path = resolve_repo_path(repo_root, value)
    if not path.exists():
        raise SystemExit(f"Groq key env file not found: {path}")
    text = path.read_text(encoding="utf-8").strip()
    if not text:
        raise SystemExit(f"Groq key env file is empty: {path}")
    return base64.b64encode((text + "\n").encode("utf-8")).decode("ascii")


def read_mimo_env_b64(repo_root: Path, value: str | Path) -> str:
    path = resolve_repo_path(repo_root, value)
    if not path.exists():
        raise SystemExit(f"MiMo env file not found: {path}")
    values = parse_env_file(path)
    mimo_key = values.get("MIMO_API_KEY", "").strip()
    if not mimo_key:
        raise SystemExit(f"MIMO_API_KEY was not found in {path}")
    lines = [f"MIMO_API_KEY={mimo_key}"]
    mimo_base_url = values.get("MIMO_BASE_URL", "").strip()
    if mimo_base_url:
        lines.append(f"MIMO_BASE_URL={mimo_base_url}")
    return base64.b64encode(("\n".join(lines) + "\n").encode("utf-8")).decode("ascii")


def dedupe_nonempty(values: list[str | None]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for value in values:
        if value is None:
            continue
        normalized = str(value).strip()
        if not normalized or normalized in seen:
            continue
        seen.add(normalized)
        result.append(normalized)
    return result


def local_head_commit(repo_root: Path) -> str:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=repo_root,
        check=True,
        text=True,
        stdout=subprocess.PIPE,
    )
    return result.stdout.strip()


def has_tracked_changes(repo_root: Path) -> bool:
    result = subprocess.run(
        ["git", "status", "--porcelain", "--untracked-files=no"],
        cwd=repo_root,
        check=True,
        text=True,
        stdout=subprocess.PIPE,
    )
    return bool(result.stdout.strip())


def append_upload_registry(path: Path, record: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")


def read_upload_registry(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    records: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        record = json.loads(stripped)
        if isinstance(record, dict):
            records.append(record)
    return records


def list_upload_registry(path: Path) -> None:
    records = read_upload_registry(path)
    if not records:
        print(f"No Kaggle notebook uploads recorded at {path}")
        return
    for record in records:
        deleted = record.get("deleted_at")
        status = "deleted" if deleted else "active"
        created = str(record.get("created_at", "-"))
        kernel_id = str(record.get("kernel_id", "-"))
        commit = str(record.get("expected_commit", "-"))[:12]
        secret_modes = []
        secret_modes.append("embedded-groq" if record.get("embedded_groq_keys") else "kaggle-groq")
        if record.get("enable_mimo"):
            secret_modes.append("embedded-mimo" if record.get("embedded_mimo_env") else "kaggle-mimo")
        datasets = ",".join(str(item) for item in record.get("dataset_sources", []) if item) or "-"
        print(f"{status:7} {created} {kernel_id} commit={commit} secrets={'+'.join(secret_modes)} datasets={datasets}")


def active_registry_kernel_ids(path: Path) -> list[str]:
    return [
        str(record["kernel_id"])
        for record in read_upload_registry(path)
        if record.get("kernel_id") and not record.get("deleted_at")
    ]


def mark_registry_deleted(path: Path, kernel_id: str) -> None:
    records = read_upload_registry(path)
    now = datetime.now(timezone.utc).isoformat()
    changed = False
    for record in records:
        if record.get("kernel_id") == kernel_id and not record.get("deleted_at"):
            record["deleted_at"] = now
            changed = True
    if not changed:
        records.append({"kernel_id": kernel_id, "deleted_at": now, "created_at": now, "note": "deleted outside registry"})
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n" for record in records),
        encoding="utf-8",
    )


def write_staging_files(
    staging_dir: Path,
    *,
    kernel_id: str,
    title: str,
    repo_url: str,
    repo_ref: str,
    expected_commit: str,
    cloudflare_token: str,
    hostname: str,
    proxy_startup_timeout_s: int,
    groq_key_env_b64: str | None,
    mimo_env_b64: str | None = None,
    dataset_sources: list[str] | None = None,
    dictionary_dataset_source: str | None = None,
    dictionary_artifact: str | None = None,
    dictionary_required: bool = False,
    available_retrievers: str | None = None,
    serve_bench: str | None = None,
    serve_retriever: str | None = None,
    serve_model_id: str | None = None,
    allow_external_semi_private: bool = False,
    enable_mimo: bool = False,
    mimo_models: str = DEFAULT_MIMO_MODELS,
) -> None:
    notebook_name = "true_chat_rag_proxy_kaggle.ipynb"
    dataset_sources = dedupe_nonempty([*(dataset_sources or []), dictionary_dataset_source])
    serve_bench = serve_bench or (DEFAULT_DICTIONARY_SERVE_BENCH if dictionary_dataset_source else DEFAULT_SERVE_BENCH)
    serve_retriever = serve_retriever or (
        DEFAULT_DICTIONARY_SERVE_RETRIEVER if dictionary_dataset_source else DEFAULT_SERVE_RETRIEVER
    )
    serve_model_id = serve_model_id or (
        DEFAULT_DICTIONARY_SERVE_MODEL_ID if dictionary_dataset_source else DEFAULT_SERVE_MODEL_ID
    )
    (staging_dir / notebook_name).write_text(
        json.dumps(
            build_notebook(
                repo_url=repo_url,
                repo_ref=repo_ref,
                expected_commit=expected_commit,
                cloudflare_token=cloudflare_token,
                hostname=hostname,
                proxy_startup_timeout_s=proxy_startup_timeout_s,
                groq_key_env_b64=groq_key_env_b64,
                mimo_env_b64=mimo_env_b64,
                dictionary_dataset_source=dictionary_dataset_source,
                dictionary_artifact=dictionary_artifact,
                dictionary_required=dictionary_required,
                available_retrievers=available_retrievers,
                serve_bench=serve_bench,
                serve_retriever=serve_retriever,
                serve_model_id=serve_model_id,
                allow_external_semi_private=allow_external_semi_private,
                enable_mimo=enable_mimo,
                mimo_models=mimo_models,
            ),
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    metadata = {
        "id": kernel_id,
        "title": title,
        "code_file": notebook_name,
        "language": "python",
        "kernel_type": "notebook",
        "is_private": True,
        "enable_gpu": False,
        "enable_tpu": False,
        "enable_internet": True,
        "dataset_sources": dataset_sources,
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
    cloudflare_token: str,
    hostname: str,
    proxy_startup_timeout_s: int = DEFAULT_PROXY_STARTUP_TIMEOUT_S,
    groq_key_env_b64: str | None = None,
    mimo_env_b64: str | None = None,
    dictionary_dataset_source: str | None = None,
    dictionary_artifact: str | None = None,
    dictionary_required: bool = False,
    available_retrievers: str | None = None,
    serve_bench: str = DEFAULT_SERVE_BENCH,
    serve_retriever: str = DEFAULT_SERVE_RETRIEVER,
    serve_model_id: str = DEFAULT_SERVE_MODEL_ID,
    allow_external_semi_private: bool = False,
    enable_mimo: bool = False,
    mimo_models: str = DEFAULT_MIMO_MODELS,
) -> dict[str, Any]:
    cells = [
        markdown_cell(
            "# True Chat RAG Proxy on Kaggle\n\n"
            "This notebook clones the repo, starts the FastAPI RAG proxy, and connects it to a Cloudflare named tunnel.",
            cell_id="intro",
        ),
        code_cell(
            "from pathlib import Path\n"
            "import os, subprocess, time, urllib.request\n\n"
            f"REPO_URL = {repo_url!r}\n"
            f"REPO_REF = {repo_ref!r}\n"
            f"EXPECTED_COMMIT = {expected_commit!r}\n"
            f"PUBLIC_HOSTNAME = {hostname!r}\n"
            f"PROXY_STARTUP_TIMEOUT_S = {proxy_startup_timeout_s!r}\n"
            "WORKDIR = Path('/kaggle/working')\n"
            "REPO_DIR = WORKDIR / 'true-chat'\n"
            "print('Repo:', REPO_URL, 'ref:', REPO_REF)\n"
            "print('Expected commit:', EXPECTED_COMMIT)\n"
            "print('Public URL:', PUBLIC_HOSTNAME)\n"
            "print('Proxy startup timeout:', PROXY_STARTUP_TIMEOUT_S, 'seconds')\n",
            cell_id="config",
        ),
        code_cell(
            "subprocess.run(['python', '-m', 'pip', 'install', '-q', 'uv'], check=True)\n"
            "if REPO_DIR.exists():\n"
            "    subprocess.run(['rm', '-rf', str(REPO_DIR)], check=True)\n"
            "subprocess.run(['git', 'clone', '--branch', REPO_REF, REPO_URL, str(REPO_DIR)], check=True)\n"
            "os.chdir(REPO_DIR)\n"
            "actual_commit = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=REPO_DIR, text=True).strip()\n"
            "print('Cloned to', REPO_DIR)\n"
            "print('Actual commit:', actual_commit)\n"
            "if actual_commit != EXPECTED_COMMIT:\n"
            "    raise RuntimeError(f'Commit mismatch: expected {EXPECTED_COMMIT}, got {actual_commit}. Push local commit before running this notebook.')\n",
            cell_id="clone-repo",
        ),
        code_cell(
            "subprocess.run(['uv', 'sync', '--frozen', '--no-dev'], cwd=REPO_DIR, check=True)\n"
            "print('uv environment synced')\n",
            cell_id="sync-deps",
        ),
    ]
    if dictionary_dataset_source and dictionary_artifact:
        cells.append(
            code_cell(
                "import shutil\n"
                f"DICTIONARY_DATASET_SOURCE = {dictionary_dataset_source!r}\n"
                f"DICTIONARY_ARTIFACT_REL = {dictionary_artifact!r}\n"
                "input_root = Path('/kaggle/input')\n"
                "dataset_slug = DICTIONARY_DATASET_SOURCE.split('/')[-1]\n"
                "target_artifact = REPO_DIR / DICTIONARY_ARTIFACT_REL\n"
                "candidate_roots = []\n"
                "preferred_root = input_root / dataset_slug\n"
                "if preferred_root.exists():\n"
                "    candidate_roots.append(preferred_root)\n"
                "if input_root.exists():\n"
                "    for root in sorted(input_root.iterdir()):\n"
                "        if root.is_dir() and root not in candidate_roots:\n"
                "            candidate_roots.append(root)\n"
                "source_artifact = None\n"
                "required_files = ('entries.jsonl', 'manifest.json')\n"
                "for root in candidate_roots:\n"
                "    exact = root / DICTIONARY_ARTIFACT_REL\n"
                "    if exact.exists() and all((exact / name).exists() for name in required_files):\n"
                "        source_artifact = exact\n"
                "        break\n"
                "    if all((root / name).exists() for name in required_files):\n"
                "        source_artifact = root\n"
                "        break\n"
                "    for manifest_path in root.rglob('manifest.json'):\n"
                "        parent = manifest_path.parent\n"
                "        if all((parent / name).exists() for name in required_files):\n"
                "            source_artifact = parent\n"
                "            break\n"
                "    if source_artifact:\n"
                "        break\n"
                "if source_artifact is None:\n"
                "    available = [str(path) for path in candidate_roots]\n"
                "    raise RuntimeError(f'Dictionary artifact not found in Kaggle inputs for {DICTIONARY_DATASET_SOURCE}; roots={available}')\n"
                "if target_artifact.exists():\n"
                "    shutil.rmtree(target_artifact)\n"
                "target_artifact.parent.mkdir(parents=True, exist_ok=True)\n"
                "shutil.copytree(source_artifact, target_artifact)\n"
                "print('Copied dictionary artifact:', source_artifact, '->', target_artifact)\n",
                cell_id="copy-dictionary-artifact",
            )
        )
    if groq_key_env_b64:
        cells.append(
            code_cell(
                "import base64\n"
                "secrets_dir = REPO_DIR / '.secrets'\n"
                "secrets_dir.mkdir(exist_ok=True)\n"
                f"GROQ_KEY_ENV_B64 = {groq_key_env_b64!r}\n"
                "(secrets_dir / 'groq_key.env').write_text(base64.b64decode(GROQ_KEY_ENV_B64).decode('utf-8'))\n"
                "print('Wrote .secrets/groq_key.env from embedded notebook payload')\n",
                cell_id="write-embedded-groq-keys",
            )
        )
    else:
        cells.append(
            code_cell(
                "secrets_dir = REPO_DIR / '.secrets'\n"
                "secrets_dir.mkdir(exist_ok=True)\n"
                "try:\n"
                "    from kaggle_secrets import UserSecretsClient\n"
                "    kaggle_secrets = UserSecretsClient()\n"
                "    groq_env = None\n"
                "    for secret_name in ('GROQ_KEY_ENV', 'GROQ_KEYS_ENV'):\n"
                "        try:\n"
                "            groq_env = kaggle_secrets.get_secret(secret_name)\n"
                "            if groq_env:\n"
                "                break\n"
                "        except Exception:\n"
                "            pass\n"
                "    if not groq_env:\n"
                "        groq_key = kaggle_secrets.get_secret('GROQ_API_KEY')\n"
                "        groq_env = 'kaggle=' + groq_key\n"
                "    (secrets_dir / 'groq_key.env').write_text(groq_env.strip() + '\\n')\n"
                "    print('Wrote .secrets/groq_key.env from Kaggle secrets')\n"
                "except Exception as exc:\n"
                "    raise RuntimeError('Add Kaggle secret GROQ_KEY_ENV with alias=value lines, or GROQ_API_KEY for one key.') from exc\n",
                cell_id="write-kaggle-groq-keys",
            )
        )
    if enable_mimo and mimo_env_b64:
        cells.append(
            code_cell(
                "import base64\n"
                "secrets_dir = REPO_DIR / '.secrets'\n"
                "secrets_dir.mkdir(exist_ok=True)\n"
                f"MIMO_ENV_B64 = {mimo_env_b64!r}\n"
                "(secrets_dir / '.env').write_text(base64.b64decode(MIMO_ENV_B64).decode('utf-8'))\n"
                "print('Wrote .secrets/.env for MiMo from embedded notebook payload')\n",
                cell_id="write-embedded-mimo-env",
            )
        )
    elif enable_mimo:
        cells.append(
            code_cell(
                "secrets_dir = REPO_DIR / '.secrets'\n"
                "secrets_dir.mkdir(exist_ok=True)\n"
                "try:\n"
                "    from kaggle_secrets import UserSecretsClient\n"
                "    kaggle_secrets = UserSecretsClient()\n"
                "    mimo_key = kaggle_secrets.get_secret('MIMO_API_KEY')\n"
                "    lines = ['MIMO_API_KEY=' + mimo_key]\n"
                "    try:\n"
                "        mimo_base_url = kaggle_secrets.get_secret('MIMO_BASE_URL')\n"
                "        if mimo_base_url:\n"
                "            lines.append('MIMO_BASE_URL=' + mimo_base_url)\n"
                "    except Exception:\n"
                "        pass\n"
                "    (secrets_dir / '.env').write_text('\\n'.join(lines) + '\\n')\n"
                "    print('Wrote .secrets/.env for MiMo from Kaggle secrets')\n"
                "except Exception as exc:\n"
                "    raise RuntimeError('Add Kaggle secret MIMO_API_KEY, and optionally MIMO_BASE_URL, or use --embed-mimo-env for a private throwaway notebook.') from exc\n",
                cell_id="write-kaggle-mimo-env",
            )
        )
    cells.extend(
        [
            code_cell(
                "cloudflared = WORKDIR / 'cloudflared'\n"
                "if not cloudflared.exists():\n"
                "    url = 'https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64'\n"
                "    urllib.request.urlretrieve(url, cloudflared)\n"
                "    cloudflared.chmod(0o755)\n"
                "print('cloudflared ready:', cloudflared)\n",
                cell_id="download-cloudflared",
            ),
            code_cell(
                f"CLOUDFLARE_TUNNEL_TOKEN = {cloudflare_token!r}\n"
                "assert CLOUDFLARE_TUNNEL_TOKEN and CLOUDFLARE_TUNNEL_TOKEN != 'REPLACE_ME'\n"
                "proxy_log_path = WORKDIR / 'rag-proxy.log'\n"
                "def print_proxy_log_tail(lines=120):\n"
                "    if not proxy_log_path.exists():\n"
                "        print('Proxy log does not exist yet:', proxy_log_path)\n"
                "        return\n"
                "    text = proxy_log_path.read_text(errors='replace')\n"
                "    tail = '\\n'.join(text.splitlines()[-lines:])\n"
                "    print(f'--- tail {proxy_log_path} ---')\n"
                "    print(tail or '(empty)')\n"
                "    print('--- end proxy log tail ---')\n"
                f"SERVE_BENCH = {serve_bench!r}\n"
                f"SERVE_RETRIEVER = {serve_retriever!r}\n"
                f"SERVE_MODEL_ID = {serve_model_id!r}\n"
                "proxy_cmd = [\n"
                "    'uv', 'run', '--frozen', '--no-sync', 'rag-bench', 'serve',\n"
                "    '--host', '0.0.0.0', '--port', '8000',\n"
                "    '--bench', SERVE_BENCH, '--retriever', SERVE_RETRIEVER, '--top-k', '3', '--image-top-k', '5',\n"
                "    '--model', 'qwen/qwen3-32b', '--model-id', SERVE_MODEL_ID,\n"
                "    '--max-context-chars', '2500', '--max-completion-tokens', '4096',\n"
                "    '--key-tpm', '6000', '--key-rpm', '30', '--rate-limit-scope', 'per-key',\n"
                "]\n"
                f"AVAILABLE_RETRIEVERS = {available_retrievers!r}\n"
                f"DICTIONARY_ARTIFACT = {dictionary_artifact!r}\n"
                f"DICTIONARY_REQUIRED = {dictionary_required!r}\n"
                f"ALLOW_EXTERNAL_SEMI_PRIVATE = {allow_external_semi_private!r}\n"
                f"ENABLE_MIMO = {enable_mimo!r}\n"
                f"MIMO_MODELS = {mimo_models!r}\n"
                "if AVAILABLE_RETRIEVERS:\n"
                "    proxy_cmd.extend(['--available-retrievers', AVAILABLE_RETRIEVERS])\n"
                "if DICTIONARY_ARTIFACT:\n"
                "    proxy_cmd.extend(['--dictionary-artifact', DICTIONARY_ARTIFACT])\n"
                "if DICTIONARY_REQUIRED:\n"
                "    proxy_cmd.append('--dictionary-required')\n"
                "if ALLOW_EXTERNAL_SEMI_PRIVATE:\n"
                "    proxy_cmd.append('--allow-external-semi-private')\n"
                "if ENABLE_MIMO:\n"
                "    proxy_cmd.append('--enable-mimo')\n"
                "    proxy_cmd.extend(['--mimo-models', MIMO_MODELS])\n"
                "proxy_env = {\n"
                "    **os.environ,\n"
                "    'PYTHONUNBUFFERED': '1',\n"
                "    'TRUE_CHAT_EXPECTED_COMMIT': EXPECTED_COMMIT,\n"
                "    'TRUE_CHAT_ACTUAL_COMMIT': actual_commit,\n"
                "}\n"
                "proxy_log = open(proxy_log_path, 'w', buffering=1)\n"
                "proxy = subprocess.Popen(proxy_cmd, cwd=REPO_DIR, env=proxy_env, stdout=proxy_log, stderr=subprocess.STDOUT, text=True)\n"
                "deadline = time.time() + PROXY_STARTUP_TIMEOUT_S\n"
                "last_report = 0.0\n"
                "while time.time() < deadline:\n"
                "    exit_code = proxy.poll()\n"
                "    if exit_code is not None:\n"
                "        proxy_log.close()\n"
                "        print_proxy_log_tail(200)\n"
                "        raise RuntimeError(f'RAG proxy exited before becoming healthy with code {exit_code}')\n"
                "    try:\n"
                "        urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=2).read()\n"
                "        break\n"
                "    except Exception as exc:\n"
                "        now = time.time()\n"
                "        if now - last_report >= 30:\n"
                "            print(f'Waiting for RAG proxy health check: {int(deadline - now)}s left; last error: {type(exc).__name__}: {exc}')\n"
                "            print_proxy_log_tail(40)\n"
                "            last_report = now\n"
                "        time.sleep(2)\n"
                "else:\n"
                "    proxy.terminate()\n"
                "    proxy_log.close()\n"
                "    print_proxy_log_tail(200)\n"
                "    raise RuntimeError(f'RAG proxy did not become healthy within {PROXY_STARTUP_TIMEOUT_S}s')\n"
                "print('RAG proxy healthy at http://127.0.0.1:8000')\n"
                "print('Starting Cloudflare tunnel. Open:', PUBLIC_HOSTNAME)\n"
                "subprocess.run([str(cloudflared), 'tunnel', '--no-autoupdate', 'run', '--token', CLOUDFLARE_TUNNEL_TOKEN], check=True)\n",
                cell_id="serve-and-tunnel",
            ),
        ]
    )
    return {
        "cells": cells,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "pygments_lexer": "ipython3"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }


def markdown_cell(source: str, *, cell_id: str) -> dict[str, Any]:
    return {"cell_type": "markdown", "id": cell_id, "metadata": {}, "source": source.splitlines(keepends=True)}


def code_cell(source: str, *, cell_id: str) -> dict[str, Any]:
    return {
        "cell_type": "code",
        "id": cell_id,
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": source.splitlines(keepends=True),
    }


def write_kaggle_config(directory: Path, credential: KaggleCredential) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / "kaggle.json"
    path.write_text(
        json.dumps({"username": credential.username, "key": credential.key}) + "\n",
        encoding="utf-8",
    )
    path.chmod(0o600)


def run_kaggle(args: list[str], kaggle_config_dir: str, *, capture: bool = False) -> str:
    if shutil.which("kaggle"):
        cmd = ["kaggle", *args]
    elif shutil.which("uvx"):
        cmd = ["uvx", "--from", "kaggle", "kaggle", *args]
    else:
        raise SystemExit("Neither kaggle nor uvx is available on PATH.")
    env = {**os.environ, "KAGGLE_CONFIG_DIR": kaggle_config_dir}
    result = subprocess.run(
        cmd,
        check=True,
        env=env,
        text=True,
        stdout=subprocess.PIPE if capture else None,
        stderr=subprocess.STDOUT if capture else None,
    )
    return result.stdout if capture else ""


def slugify(value: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", value.lower()).strip("-")
    return slug[:63].strip("-") or "true-chat-rag-proxy"


if __name__ == "__main__":
    raise SystemExit(main())
