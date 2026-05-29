#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import shutil
import sys
import time
import urllib.error
import urllib.request
from collections import Counter
from dataclasses import dataclass
from html import escape
from pathlib import Path
from typing import Any

from rag_bench.dictionary import (
    DICTIONARY_SCHEMA_VERSION,
    DEFAULT_DICTIONARY_SOURCE_DIR,
    DictionaryEntry as Entry,
    extract_headword as parse_extract_headword,
    load_dictionary_entries,
    normalize_spaces as dictionary_normalize_spaces,
    parse_dictionary_docx,
    sha256_file as dictionary_sha256_file,
    slugify as dictionary_slugify,
    source_file_manifest as dictionary_source_file_manifest,
    strip_accents as dictionary_strip_accents,
)
from rag_bench.dictionary_graph import (
    DEFAULT_ONTOLOGY_PATH,
    GRAPH_SCHEMA_VERSION,
    PROMPT_VERSION,
    finalize_dictionary_graph,
    load_ontology,
    write_quality_report,
    write_sqlite_store,
)

DEFAULT_SOURCE_DIR = DEFAULT_DICTIONARY_SOURCE_DIR
DEFAULT_MIMO_BASE_URL = "https://token-plan-sgp.xiaomimimo.com/v1"
DEFAULT_MIMO_MODEL = "mimo-v2.5"
DEFAULT_GROQ_MODEL = "llama-3.1-8b-instant"
DEFAULT_LOCAL_MODEL = "local"
PRIVATE_SOURCE_MARKERS = {"private", "secret", "classified", "top-secret", "top_secret", "tuyet-mat", "tuyệt-mật"}

CATEGORIES = (
    "vũ khí/trang bị",
    "đạn dược/thuốc nổ",
    "khí tài đo đạc/trinh sát",
    "bản đồ/địa hình",
    "chỉ huy/huấn luyện",
    "bảo đảm kỹ thuật",
    "khái niệm tác chiến",
    "tổ chức/lực lượng",
    "khí tượng/địa vật",
    "khác",
)
RELATIONS = (
    "is_a",
    "part_of",
    "component_of",
    "used_for",
    "measures",
    "controls",
    "fires",
    "supports",
    "located_in",
    "requires",
    "see_also",
    "related_to",
)


@dataclass(frozen=True)
class ProviderResult:
    answer: str
    error: str | None
    key_alias: str | None
    attempted_aliases: list[str]
    rejected_aliases: list[str]
    retry_count: int
    prompt_tokens: int | None
    completion_tokens: int | None
    total_tokens: int | None
    estimated_tokens: int | None
    scheduled_wait_s: float | None


@dataclass(frozen=True)
class SourceSpec:
    name: str
    path: Path
    letters: list[str]
    namespace_ids: bool
    classification: str = "public"


@dataclass
class ProgressTracker:
    stage: str
    total_items: int
    item_name: str
    total_entries: int | None = None
    enabled: bool = True
    started_s: float = 0.0

    def __post_init__(self) -> None:
        self.started_s = time.monotonic()

    def update(
        self,
        *,
        done_items: int,
        batch_position: str | None = None,
        entries_done: int | None = None,
        note: str | None = None,
    ) -> None:
        if not self.enabled or self.total_items <= 0:
            return
        elapsed_s = max(0.0, time.monotonic() - self.started_s)
        percent = min(100.0, 100.0 * done_items / self.total_items)
        eta_s = elapsed_s / done_items * (self.total_items - done_items) if done_items > 0 else None
        parts = [f"[{self.stage}]"]
        if batch_position:
            parts.append(batch_position)
        parts.append(f"{self.item_name} {done_items}/{self.total_items}")
        if self.total_entries is not None and entries_done is not None:
            parts.append(f"entries {entries_done}/{self.total_entries}")
        parts.append(f"{percent:.1f}%")
        parts.append(f"elapsed {format_duration(elapsed_s)}")
        if eta_s is not None:
            parts.append(f"eta {format_duration(eta_s)}")
        if note:
            parts.append(note)
        print(" | ".join(parts), file=sys.stderr, flush=True)


class OpenAICompatibleClient:
    def __init__(
        self,
        *,
        api_key: str,
        base_url: str,
        auth_header: str,
        timeout_s: float,
        sleep_between_calls_s: float,
        extra_body: dict[str, Any] | None = None,
        key_alias: str = "openai-compatible",
    ) -> None:
        self.api_key = api_key
        self.base_url = base_url.rstrip("/")
        self.auth_header = auth_header
        self.timeout_s = timeout_s
        self.sleep_between_calls_s = sleep_between_calls_s
        self.extra_body = extra_body or {}
        self.key_alias = key_alias
        self.call_count = 0

    def generate(
        self,
        messages: list[dict[str, str]],
        *,
        model: str,
        temperature: float,
        max_completion_tokens: int,
    ) -> ProviderResult:
        if self.call_count and self.sleep_between_calls_s > 0:
            time.sleep(self.sleep_between_calls_s)
        self.call_count += 1

        payload = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_completion_tokens,
            "stream": False,
        }
        payload.update(self.extra_body)
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        headers = {
            "Content-Type": "application/json",
            "Accept": "application/json",
        }
        if self.auth_header in {"authorization", "both"}:
            headers["Authorization"] = f"Bearer {self.api_key}"
        if self.auth_header in {"api-key", "both"}:
            headers["api-key"] = self.api_key

        request = urllib.request.Request(
            f"{self.base_url}/chat/completions",
            data=body,
            headers=headers,
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=self.timeout_s) as response:
                parsed = json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            response_text = exc.read().decode("utf-8", errors="replace")
            return ProviderResult(
                answer="",
                error=f"status={exc.code} HTTPError: {_redact_secret(response_text)}",
                key_alias=self.key_alias,
                attempted_aliases=[self.key_alias],
                rejected_aliases=[self.key_alias] if exc.code in {401, 403} else [],
                retry_count=0,
                prompt_tokens=None,
                completion_tokens=None,
                total_tokens=None,
                estimated_tokens=estimate_requested_tokens(messages, max_completion_tokens),
                scheduled_wait_s=0.0,
            )
        except Exception as exc:  # noqa: BLE001 - urllib raises several transport exceptions.
            return ProviderResult(
                answer="",
                error=f"{exc.__class__.__name__}: {_redact_secret(str(exc))}",
                key_alias=self.key_alias,
                attempted_aliases=[self.key_alias],
                rejected_aliases=[],
                retry_count=0,
                prompt_tokens=None,
                completion_tokens=None,
                total_tokens=None,
                estimated_tokens=estimate_requested_tokens(messages, max_completion_tokens),
                scheduled_wait_s=0.0,
            )

        usage = parsed.get("usage") or {}
        choices = parsed.get("choices") or []
        answer = ""
        if choices:
            message = choices[0].get("message") or {}
            answer = extract_message_text(message)
        return ProviderResult(
            answer=answer,
            error=None,
            key_alias=self.key_alias,
            attempted_aliases=[self.key_alias],
            rejected_aliases=[],
            retry_count=0,
            prompt_tokens=usage.get("prompt_tokens"),
            completion_tokens=usage.get("completion_tokens"),
            total_tokens=usage.get("total_tokens"),
            estimated_tokens=estimate_requested_tokens(messages, max_completion_tokens),
            scheduled_wait_s=0.0,
        )


class GroqProviderClient:
    def __init__(
        self,
        *,
        keys_file: Path,
        model: str,
        max_retries: int,
        key_tpm: int,
        key_rpm: int,
        rate_limit_scope: str,
    ) -> None:
        from rag_bench.groq_client import RoundRobinGroqClient
        from rag_bench.secrets import load_groq_keys

        keys = load_groq_keys(keys_file)
        self.client = RoundRobinGroqClient(
            keys=keys,
            model=model,
            max_retries=max_retries,
            key_tokens_per_minute=key_tpm,
            key_requests_per_minute=key_rpm,
            rate_limit_scope=rate_limit_scope,
        )

    def generate(
        self,
        messages: list[dict[str, str]],
        *,
        model: str,
        temperature: float,
        max_completion_tokens: int,
    ) -> ProviderResult:
        result = self.client.generate(
            messages,
            model=model,
            temperature=temperature,
            max_completion_tokens=max_completion_tokens,
        )
        return ProviderResult(
            answer=result.answer,
            error=result.error,
            key_alias=result.key_alias,
            attempted_aliases=result.attempted_aliases,
            rejected_aliases=result.rejected_aliases,
            retry_count=result.retry_count,
            prompt_tokens=result.prompt_tokens,
            completion_tokens=result.completion_tokens,
            total_tokens=result.total_tokens,
            estimated_tokens=result.estimated_tokens,
            scheduled_wait_s=result.scheduled_wait_s,
        )


def main() -> int:
    args = parse_args()
    source_specs = parse_source_specs(args)
    letters = unique_letters(source_specs)
    model = args.model or _default_model_for_provider(args.provider)
    enforce_private_source_policy(args, source_specs, model)
    run_dir = resolve_run_dir(args, letters, source_specs)
    raw_dir = run_dir / "raw_batches"
    raw_dir.mkdir(parents=True, exist_ok=True)
    ontology = load_ontology(args.ontology_path)

    entries = load_entries(source_specs)
    if args.limit_entries is not None:
        entries = entries[: args.limit_entries]
    if not entries:
        raise SystemExit("No dictionary entries found for the selected letters.")

    config = {
        "schema_version": GRAPH_SCHEMA_VERSION,
        "dictionary_schema_version": DICTIONARY_SCHEMA_VERSION,
        "ontology_path": str(args.ontology_path),
        "ontology_version": ontology.ontology_version,
        "prompt_version": PROMPT_VERSION,
        "provider": args.provider,
        "model": model,
        "source_dir": str(args.source_dir),
        "source_sets": [
            {
                "name": spec.name,
                "path": str(spec.path),
                "letters": spec.letters,
                "namespace_ids": spec.namespace_ids,
                "classification": spec.classification,
            }
            for spec in source_specs
        ],
        "letters": letters,
        "batch_size": args.batch_size,
        "entry_char_limit": args.entry_char_limit,
        "max_completion_tokens": args.max_completion_tokens,
        "micro_max_completion_tokens": args.micro_max_completion_tokens,
        "temperature": args.temperature,
        "repair": not args.no_repair,
        "fallback": not args.no_fallback,
        "quality_pass": args.quality_pass,
        "force_reextract": args.force_reextract,
        "trusted_models": list(args.trusted_model),
        "model_trust": "trusted" if is_trusted_private_model(args, model) else "untrusted",
        "validate_only": args.validate_only,
        "export_only": args.export_only,
        "sqlite_path": str(args.sqlite_path) if args.sqlite_path else "dictionary_graph.sqlite",
        "progress": args.progress,
        "source_files": source_file_manifest(source_specs),
    }
    write_json(run_dir / "run_config.json", config)
    write_entries(run_dir / "entries.jsonl", entries)

    total_batches = math.ceil(len(entries) / args.batch_size)
    batch_numbers = list(range(1, total_batches + 1))
    pending = find_pending_batches(
        raw_dir,
        entries,
        args.batch_size,
        batch_numbers,
        args.resume and not args.force_reextract,
        model=model,
        prompt_version=PROMPT_VERSION,
    )

    print_json(
        {
            "run_dir": str(run_dir),
            "provider": args.provider,
            "model": model,
            "entries": len(entries),
            "total_batches": total_batches,
            "pending_batches": len(pending),
        }
    )

    client = None
    if not args.validate_only and not args.export_only:
        client = build_provider_client(args, model)
        generate_batches(
            client=client,
            raw_dir=raw_dir,
            entries=entries,
            batch_numbers=pending,
            batch_size=args.batch_size,
            entry_char_limit=args.entry_char_limit,
            model=model,
            temperature=args.temperature,
            max_completion_tokens=args.max_completion_tokens,
            total_expected_batches=total_batches,
            progress=args.progress,
        )

        failures, missing_ids = validate_raw_batches(raw_dir, entries, args.batch_size, batch_numbers)
        if not args.no_repair and failures:
            repair_bad_batches(
                client=client,
                raw_dir=raw_dir,
                entries=entries,
                batch_numbers=[failure["batch"] for failure in failures if isinstance(failure.get("batch"), int)],
                batch_size=args.batch_size,
                entry_char_limit=args.repair_entry_char_limit,
                model=model,
                temperature=args.temperature,
                max_completion_tokens=args.repair_max_completion_tokens,
                total_expected_batches=total_batches,
                progress=args.progress,
            )

        failures, missing_ids = validate_raw_batches(raw_dir, entries, args.batch_size, batch_numbers)
        if not args.no_repair and missing_ids:
            micro_repair_entries(
                client=client,
                raw_dir=raw_dir,
                entries_by_id={entry.id: entry for entry in entries},
                missing_ids=missing_ids,
                entry_char_limit=args.micro_entry_char_limit,
                model=model,
                temperature=args.temperature,
                max_completion_tokens=args.micro_max_completion_tokens,
                fallback=not args.no_fallback,
                progress=args.progress,
            )

        failures, missing_ids = validate_raw_batches(raw_dir, entries, args.batch_size, batch_numbers)
        if not args.no_fallback and (failures or missing_ids):
            apply_fallbacks(raw_dir, entries, args.batch_size, failures, missing_ids)

    graph = build_graph(raw_dir, entries, args.batch_size, batch_numbers)
    graph = finalize_dictionary_graph(
        graph=graph,
        entries=entries,
        ontology=ontology,
        extractor=args.provider,
        prompt_version=PROMPT_VERSION,
    )
    if client is not None and args.quality_pass != "none":
        graph = run_quality_pass(
            client=client,
            run_dir=run_dir,
            graph=graph,
            entries=entries,
            ontology=ontology,
            provider=args.provider,
            model=model,
            temperature=args.temperature,
            quality_pass=args.quality_pass,
            max_completion_tokens=args.repair_max_completion_tokens,
            progress=args.progress,
        )
    progress_log(args.progress, "[export] writing graph artifacts")
    write_graph_artifacts(
        run_dir=run_dir,
        graph=graph,
        entries=entries,
        config=config,
        provider=args.provider,
        model=model,
        total_batches=total_batches,
        sqlite_path=args.sqlite_path,
    )
    if args.graphml:
        write_graphml(run_dir / "graph.graphml", graph["nodes"], graph["edges"])
    if args.visualize:
        write_visualization(run_dir / "graph_visualization.html", graph["nodes"], graph["edges"], graph["manifest"])
    progress_log(args.progress, "[export] done")

    print("FINAL " + json.dumps(graph["manifest"], ensure_ascii=False), flush=True)
    return 0 if not graph["manifest"]["partial"] else 2


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a reproducible dictionary knowledge graph from DOCX entries.")
    parser.add_argument("--source-dir", type=Path, default=DEFAULT_SOURCE_DIR)
    parser.add_argument("--letters", default="A,B,C", help="Comma-separated root DOCX letters, e.g. A,B,C,D.")
    parser.add_argument(
        "--source-set",
        action="append",
        default=[],
        metavar="NAME=PATH|LETTERS",
        help=(
            "Repeatable multi-source input. Example: "
            "'base=data/semi_private/File Từ điển PB_2021|A,B,C' "
            "'supp2021=data/semi_private/File Từ điển PB_2021/01. Mục từ Bổ sung 2021|B,C'. "
            "When set, entry ids are namespaced as NAME:LETTER-0001."
        ),
    )
    parser.add_argument("--out-dir", type=Path, default=Path("runs"))
    parser.add_argument("--run-name", help="Stable run directory name. Defaults to timestamped provider/letters name.")
    parser.add_argument("--provider", choices=("mimo", "groq", "local"), default="mimo")
    parser.add_argument("--model", help="LLM model id. Defaults to provider-specific value.")
    parser.add_argument(
        "--trusted-model",
        action="append",
        default=[],
        help=(
            "Model id allowed to process private source sets. Private inputs require "
            "--provider local and the selected --model to appear in this allowlist."
        ),
    )
    parser.add_argument("--batch-size", type=int, default=3)
    parser.add_argument("--entry-char-limit", type=int, default=500)
    parser.add_argument("--limit-entries", type=int)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max-completion-tokens", type=int, default=1100)
    parser.add_argument("--repair-max-completion-tokens", type=int, default=700)
    parser.add_argument("--micro-max-completion-tokens", type=int, default=260)
    parser.add_argument("--repair-entry-char-limit", type=int, default=420)
    parser.add_argument("--micro-entry-char-limit", type=int, default=360)
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--force-reextract", action="store_true", help="Ignore valid raw batch cache and call the provider again.")
    parser.add_argument("--validate-only", action="store_true", help="Validate existing raw/artifact data without provider calls.")
    parser.add_argument("--export-only", action="store_true", help="Rebuild artifacts from existing raw batches without provider calls.")
    parser.add_argument("--quality-pass", choices=("none", "weak", "all"), default="weak")
    parser.add_argument("--ontology-path", type=Path, default=DEFAULT_ONTOLOGY_PATH)
    parser.add_argument("--sqlite-path", type=Path, default=None, help="SQLite output path. Defaults to <run-dir>/dictionary_graph.sqlite.")
    parser.add_argument("--no-repair", action="store_true", help="Skip LLM repair for invalid JSON or missing entries.")
    parser.add_argument("--no-fallback", action="store_true", help="Do not insert local minimal fallback entries.")
    parser.add_argument("--visualize", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--graphml", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--progress", action=argparse.BooleanOptionalAction, default=True)

    parser.add_argument("--api-env-file", type=Path, default=Path(".secrets/.env"))
    parser.add_argument("--api-key-var", default="MIMO_API_KEY")
    parser.add_argument("--base-url", default=None)
    parser.add_argument("--auth-header", choices=("authorization", "api-key", "both", "none"), default="both")
    parser.add_argument("--timeout-s", type=float, default=120.0)
    parser.add_argument("--sleep-between-calls-s", type=float, default=0.0)

    parser.add_argument("--groq-keys-file", type=Path, default=Path(".secrets/groq_key.env"))
    parser.add_argument("--max-retries", type=int, default=2)
    parser.add_argument("--key-tpm", type=int, default=6000)
    parser.add_argument("--key-rpm", type=int, default=30)
    parser.add_argument("--rate-limit-scope", choices=("per-key", "shared"), default="per-key")
    args = parser.parse_args()
    if args.validate_only and args.export_only:
        parser.error("--validate-only and --export-only are mutually exclusive")
    return args


def parse_letters(value: str) -> list[str]:
    letters = [part.strip() for part in value.split(",") if part.strip()]
    if not letters:
        raise SystemExit("--letters must contain at least one value")
    return letters


def parse_source_specs(args: argparse.Namespace) -> list[SourceSpec]:
    if not args.source_set:
        return [
            SourceSpec(
                name="base",
                path=args.source_dir,
                letters=parse_letters(args.letters),
                namespace_ids=False,
                classification=classify_source_path(args.source_dir),
            )
        ]
    specs = [parse_source_set(value) for value in args.source_set]
    names = [spec.name for spec in specs]
    duplicate_names = sorted(name for name, count in Counter(names).items() if count > 1)
    if duplicate_names:
        raise SystemExit(f"Duplicate --source-set names: {', '.join(duplicate_names)}")
    return specs


def parse_source_set(value: str) -> SourceSpec:
    if "=" not in value or "|" not in value:
        raise SystemExit("--source-set must use format NAME=PATH|LETTERS or NAME=PATH|LETTERS|CLASSIFICATION")
    name, rest = value.split("=", 1)
    parts = rest.split("|")
    if len(parts) not in {2, 3}:
        raise SystemExit("--source-set must use format NAME=PATH|LETTERS or NAME=PATH|LETTERS|CLASSIFICATION")
    path_text, letters_text = parts[0], parts[1]
    explicit_classification = parts[2] if len(parts) == 3 else None
    name = slugify(name.strip())
    if not name:
        raise SystemExit("--source-set name cannot be empty")
    path = Path(path_text.strip())
    letters = parse_letters(letters_text)
    classification = normalize_source_classification(explicit_classification or classify_source_path(path))
    return SourceSpec(name=name, path=path, letters=letters, namespace_ids=True, classification=classification)


def normalize_source_classification(value: str) -> str:
    normalized = slugify(value.strip()).replace("_", "-")
    if normalized in {"public", "semi-private", "semiprivate", "internal"}:
        return "public" if normalized == "public" else "semi-private"
    if normalized in PRIVATE_SOURCE_MARKERS:
        return "private"
    raise SystemExit("--source-set CLASSIFICATION must be one of: public, semi-private, private")


def classify_source_path(path: Path) -> str:
    parts = {slugify(part) for part in path.parts}
    return "private" if parts & PRIVATE_SOURCE_MARKERS else "public"


def _default_model_for_provider(provider: str) -> str:
    if provider == "mimo":
        return DEFAULT_MIMO_MODEL
    if provider == "groq":
        return DEFAULT_GROQ_MODEL
    return DEFAULT_LOCAL_MODEL


def enforce_private_source_policy(args: argparse.Namespace, source_specs: list[SourceSpec], model: str) -> None:
    private_specs = [spec for spec in source_specs if spec.classification == "private"]
    if not private_specs or args.validate_only or args.export_only:
        return
    if is_trusted_private_model(args, model):
        return
    names = ", ".join(spec.name for spec in private_specs)
    raise SystemExit(
        "Refusing to process private source set(s) with an untrusted or non-local model: "
        f"{names}. Use --provider local --model {model!r} --trusted-model {model!r} "
        "with a local OpenAI-compatible endpoint, or use --export-only/--validate-only for local artifact work."
    )


def is_trusted_private_model(args: argparse.Namespace, model: str) -> bool:
    return args.provider == "local" and model in set(args.trusted_model or [])


def unique_letters(source_specs: list[SourceSpec]) -> list[str]:
    letters: list[str] = []
    for spec in source_specs:
        for letter in spec.letters:
            if letter not in letters:
                letters.append(letter)
    return letters


def resolve_run_dir(args: argparse.Namespace, letters: list[str], source_specs: list[SourceSpec]) -> Path:
    if args.run_name:
        return args.out_dir / args.run_name
    stamp = time.strftime("%Y%m%dT%H%M%S")
    if args.source_set:
        source_slug = "-".join(spec.name for spec in source_specs)
    else:
        source_slug = "".join(slugify(letter) for letter in letters).lower()
    return args.out_dir / f"pb_dictionary_{source_slug}_{args.provider}_graph_{stamp}"


def build_provider_client(args: argparse.Namespace, model: str) -> Any:
    if args.provider == "groq":
        return GroqProviderClient(
            keys_file=args.groq_keys_file,
            model=model,
            max_retries=args.max_retries,
            key_tpm=args.key_tpm,
            key_rpm=args.key_rpm,
            rate_limit_scope=args.rate_limit_scope,
        )
    if args.provider == "local":
        base_url = args.base_url or "http://127.0.0.1:8000/v1"
        env_values = load_env_file(args.api_env_file)
        api_key = env_values.get(args.api_key_var) or "local"
        return OpenAICompatibleClient(
            api_key=api_key,
            base_url=base_url,
            auth_header=args.auth_header if args.auth_header != "both" else "none",
            timeout_s=args.timeout_s,
            sleep_between_calls_s=args.sleep_between_calls_s,
            extra_body={"enable_thinking": False},
            key_alias="local",
        )
    env_values = load_env_file(args.api_env_file)
    api_key = env_values.get(args.api_key_var)
    if not api_key:
        raise SystemExit(f"{args.api_key_var} was not found in {args.api_env_file}")
    base_url = args.base_url or env_values.get("MIMO_BASE_URL") or DEFAULT_MIMO_BASE_URL
    return OpenAICompatibleClient(
        api_key=api_key,
        base_url=base_url,
        auth_header=args.auth_header,
        timeout_s=args.timeout_s,
        sleep_between_calls_s=args.sleep_between_calls_s,
        extra_body={"enable_thinking": False},
        key_alias="mimo",
    )


def load_env_file(path: Path) -> dict[str, str]:
    if not path.exists():
        return {}
    values: dict[str, str] = {}
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[len("export ") :].strip()
        if "=" not in line:
            continue
        key, value = line.split("=", 1)
        values[key.strip()] = strip_env_quotes(value.strip())
    return values


def strip_env_quotes(value: str) -> str:
    if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
        return value[1:-1]
    return value


def load_entries(source_specs: list[SourceSpec]) -> list[Entry]:
    entries: list[Entry] = []
    seen: set[str] = set()
    for spec in source_specs:
        spec_entries = load_dictionary_entries(
            spec.path,
            spec.letters,
            source_set=spec.name if spec.namespace_ids else None,
            id_prefix=spec.name if spec.namespace_ids else None,
        )
        for entry in spec_entries:
            if entry.id in seen:
                raise SystemExit(f"Duplicate dictionary entry id after source-set namespacing: {entry.id}")
            seen.add(entry.id)
            entries.append(entry)
    return entries


def paragraph_rows(path: Path) -> list[dict[str, Any]]:
    return parse_dictionary_docx(path)


def extract_headword(row: dict[str, Any]) -> str:
    return parse_extract_headword(row)


def find_pending_batches(
    raw_dir: Path,
    entries: list[Entry],
    batch_size: int,
    batch_numbers: list[int],
    resume: bool,
    *,
    model: str,
    prompt_version: str,
) -> list[int]:
    if not resume:
        return batch_numbers
    pending: list[int] = []
    entries_by_id = {entry.id: entry for entry in entries}
    for batch_number in batch_numbers:
        raw_path = batch_path(raw_dir, batch_number)
        input_ids = [entry.id for entry in batch_for_number(entries, batch_size, batch_number)]
        if not raw_path.exists():
            pending.append(batch_number)
            continue
        error = raw_parse_error(raw_path)
        if error is not None:
            pending.append(batch_number)
            continue
        raw = read_raw(raw_path)
        expected_cache_key = batch_cache_key(
            batch_for_number(entries, batch_size, batch_number),
            batch_size=batch_size,
            model=model,
            prompt_version=prompt_version,
        )
        if raw.get("cache_key") != expected_cache_key:
            pending.append(batch_number)
            continue
        parsed = parse_json_object(raw.get("answer") or "")
        parsed_ids = {
            normalize_spaces(str(item.get("id", "")))
            for item in parsed.get("entries", [])
            if isinstance(item, dict)
        }
        if any(entry_id not in parsed_ids or entry_id not in entries_by_id for entry_id in input_ids):
            pending.append(batch_number)
    return pending


def generate_batches(
    *,
    client: Any,
    raw_dir: Path,
    entries: list[Entry],
    batch_numbers: list[int],
    batch_size: int,
    entry_char_limit: int,
    model: str,
    temperature: float,
    max_completion_tokens: int,
    total_expected_batches: int,
    progress: bool,
) -> None:
    system = graph_system_prompt()
    tracker = ProgressTracker(
        stage="generate",
        total_items=len(batch_numbers),
        item_name="pending",
        total_entries=sum(len(batch_for_number(entries, batch_size, batch_number)) for batch_number in batch_numbers),
        enabled=progress,
    )
    entries_done = 0
    for run_index, batch_number in enumerate(batch_numbers, start=1):
        batch = batch_for_number(entries, batch_size, batch_number)
        parse_error: str | None
        payload = [
            {"id": entry.id, "headword": entry.headword, "text": entry.text[:entry_char_limit]}
            for entry in batch
        ]
        result = client.generate(
            [
                {"role": "system", "content": system},
                {"role": "user", "content": json.dumps({"entries": payload}, ensure_ascii=False)},
            ],
            model=model,
            temperature=temperature,
            max_completion_tokens=max_completion_tokens,
        )
        raw = provider_result_to_raw(batch_number, batch, result)
        add_batch_cache_metadata(raw, batch, batch_size=batch_size, model=model, prompt_version=PROMPT_VERSION)
        raw["generated_at"] = iso_now()
        write_json(batch_path(raw_dir, batch_number), raw)
        parse_error = None if result.error else raw_parse_error(batch_path(raw_dir, batch_number))
        print_json(
            {
                "batch": batch_number,
                "run_index": run_index,
                "remaining_after": len(batch_numbers) - run_index,
                "input_ids": [entry.id for entry in batch],
                "key_alias": result.key_alias,
                "prompt_tokens": result.prompt_tokens,
                "completion_tokens": result.completion_tokens,
                "scheduled_wait_s": result.scheduled_wait_s,
                "error": result.error,
                "parse_error": parse_error,
            }
        )
        entries_done += len(batch)
        note = "error" if result.error else ("parse-error" if parse_error else None)
        tracker.update(
            done_items=run_index,
            batch_position=f"batch {batch_number}/{total_expected_batches}",
            entries_done=entries_done,
            note=note,
        )


def repair_bad_batches(
    *,
    client: Any,
    raw_dir: Path,
    entries: list[Entry],
    batch_numbers: list[int],
    batch_size: int,
    entry_char_limit: int,
    model: str,
    temperature: float,
    max_completion_tokens: int,
    total_expected_batches: int,
    progress: bool,
) -> None:
    system = repair_system_prompt()
    repair_batches = sorted(set(batch_numbers))
    tracker = ProgressTracker(
        stage="repair",
        total_items=len(repair_batches),
        item_name="batch",
        total_entries=sum(len(batch_for_number(entries, batch_size, batch_number)) for batch_number in repair_batches),
        enabled=progress,
    )
    entries_done = 0
    for repair_index, batch_number in enumerate(repair_batches, start=1):
        raw_path = batch_path(raw_dir, batch_number)
        backup_raw(raw_path)
        batch = batch_for_number(entries, batch_size, batch_number)
        payload = [
            {"id": entry.id, "headword": entry.headword, "text": entry.text[:entry_char_limit]}
            for entry in batch
        ]
        result = client.generate(
            [
                {"role": "system", "content": system},
                {"role": "user", "content": json.dumps({"entries": payload}, ensure_ascii=False)},
            ],
            model=model,
            temperature=temperature,
            max_completion_tokens=max_completion_tokens,
        )
        raw = provider_result_to_raw(batch_number, batch, result)
        add_batch_cache_metadata(raw, batch, batch_size=batch_size, model=model, prompt_version=PROMPT_VERSION)
        raw["repair_attempt"] = True
        raw["repaired_at"] = iso_now()
        write_json(raw_path, raw)
        parse_error = None if result.error else raw_parse_error(raw_path)
        print_json(
            {
                "repair_batch": batch_number,
                "key_alias": result.key_alias,
                "error": result.error,
                "parse_error_after": parse_error,
            }
        )
        entries_done += len(batch)
        note = "error" if result.error else ("parse-error" if parse_error else None)
        tracker.update(
            done_items=repair_index,
            batch_position=f"batch {batch_number}/{total_expected_batches}",
            entries_done=entries_done,
            note=note,
        )


def micro_repair_entries(
    *,
    client: Any,
    raw_dir: Path,
    entries_by_id: dict[str, Entry],
    missing_ids: list[str],
    entry_char_limit: int,
    model: str,
    temperature: float,
    max_completion_tokens: int,
    fallback: bool,
    progress: bool,
) -> None:
    system = micro_system_prompt()
    grouped: dict[int, list[Entry]] = {}
    for entry_id in missing_ids:
        entry = entries_by_id[entry_id]
        match = re.search(r"-(\d+)$", entry.id)
        batch_number = int(match.group(1)) if match else 0
        # This batch number is only valid for one-letter files, so compute from raw ids below if needed.
        grouped.setdefault(batch_number, []).append(entry)

    # Regroup by raw batch membership to avoid relying on local indexes across letters.
    grouped = {}
    for path in raw_dir.glob("batch_*.json"):
        raw = read_raw(path)
        batch_number = int(raw.get("batch"))
        ids = set(raw.get("input_ids", []))
        selected = [entries_by_id[entry_id] for entry_id in missing_ids if entry_id in ids]
        if selected:
            grouped[batch_number] = selected

    tracker = ProgressTracker(
        stage="micro-repair",
        total_items=sum(len(batch_entries) for batch_entries in grouped.values()),
        item_name="entry",
        total_entries=sum(len(batch_entries) for batch_entries in grouped.values()),
        enabled=progress,
    )
    entries_done = 0
    for batch_number, batch_entries in sorted(grouped.items()):
        raw_path = batch_path(raw_dir, batch_number)
        backup_raw(raw_path)
        raw = read_raw(raw_path)
        try:
            parsed = parse_json_object(raw.get("answer") or '{"entries":[]}')
        except Exception:
            parsed = {"entries": []}
        output_entries = [item for item in parsed.get("entries", []) if isinstance(item, dict)]
        micro_failures: list[dict[str, str]] = []
        token_totals: Counter[str] = Counter()
        aliases: list[str | None] = []
        for entry in batch_entries:
            result = client.generate(
                [
                    {"role": "system", "content": system},
                    {
                        "role": "user",
                        "content": json.dumps(
                            {"id": entry.id, "headword": entry.headword, "text": entry.text[:entry_char_limit]},
                            ensure_ascii=False,
                        ),
                    },
                ],
                model=model,
                temperature=temperature,
                max_completion_tokens=max_completion_tokens,
            )
            aliases.append(result.key_alias)
            add_usage(token_totals, result)
            item: dict[str, Any]
            try:
                item = parse_json_object(result.answer) if result.error is None else {}
                if normalize_spaces(str(item.get("id", ""))) != entry.id:
                    raise ValueError("missing or mismatched id")
            except Exception as exc:
                if not fallback:
                    micro_failures.append({"id": entry.id, "error": result.error or str(exc)})
                    continue
                item = fallback_item(entry)
                micro_failures.append({"id": entry.id, "error": result.error or str(exc), "fallback": "local"})
            output_entries = [existing for existing in output_entries if existing.get("id") != entry.id]
            output_entries.append(item)
            entries_done += 1
            tracker.update(
                done_items=entries_done,
                batch_position=f"batch {batch_number}",
                entries_done=entries_done,
                note="fallback" if micro_failures and micro_failures[-1].get("id") == entry.id else None,
            )

        raw["answer"] = json.dumps({"entries": output_entries}, ensure_ascii=False, separators=(",", ":"))
        raw["error"] = None
        raw["key_alias"] = next((alias for alias in reversed(aliases) if alias), raw.get("key_alias"))
        raw["attempted_aliases"] = [alias for alias in aliases if alias]
        raw["prompt_tokens"] = token_totals.get("prompt_tokens") or raw.get("prompt_tokens")
        raw["completion_tokens"] = token_totals.get("completion_tokens") or raw.get("completion_tokens")
        raw["total_tokens"] = token_totals.get("total_tokens") or raw.get("total_tokens")
        raw["estimated_tokens"] = token_totals.get("estimated_tokens") or raw.get("estimated_tokens")
        raw["micro_repair"] = True
        raw["micro_failures"] = micro_failures
        raw["repaired_at"] = iso_now()
        write_json(raw_path, raw)
        print_json(
            {
                "micro_repair_batch": batch_number,
                "entries": [entry.id for entry in batch_entries],
                "micro_failures": len(micro_failures),
                "parse_error_after": raw_parse_error(raw_path),
            }
        )


def apply_fallbacks(
    raw_dir: Path,
    entries: list[Entry],
    batch_size: int,
    failures: list[dict[str, Any]],
    missing_ids: list[str],
) -> None:
    entries_by_id = {entry.id: entry for entry in entries}
    missing = set(missing_ids)
    failed_batches = {failure["batch"] for failure in failures if isinstance(failure.get("batch"), int)}
    for batch_number in sorted(failed_batches):
        raw_path = batch_path(raw_dir, batch_number)
        backup_raw(raw_path)
        batch = batch_for_number(entries, batch_size, batch_number)
        raw = read_raw(raw_path) if raw_path.exists() else {"batch": batch_number, "input_ids": [e.id for e in batch]}
        raw["answer"] = json.dumps({"entries": [fallback_item(entry) for entry in batch]}, ensure_ascii=False)
        raw["error"] = None
        raw["fallback_batch"] = True
        raw["fallback_reason"] = "raw batch could not be parsed"
        raw["fallback_at"] = iso_now()
        write_json(raw_path, raw)
    for raw_path in raw_dir.glob("batch_*.json"):
        raw = read_raw(raw_path)
        ids = set(raw.get("input_ids", []))
        selected = [entries_by_id[entry_id] for entry_id in missing if entry_id in ids]
        if not selected:
            continue
        backup_raw(raw_path)
        parsed = parse_json_object(raw.get("answer") or '{"entries":[]}')
        output_entries = [item for item in parsed.get("entries", []) if isinstance(item, dict)]
        for entry in selected:
            output_entries = [item for item in output_entries if item.get("id") != entry.id]
            output_entries.append(fallback_item(entry))
        raw["answer"] = json.dumps({"entries": output_entries}, ensure_ascii=False)
        raw["fallback_entries"] = [{"id": entry.id, "reason": "missing from LLM JSON"} for entry in selected]
        raw["fallback_at"] = iso_now()
        write_json(raw_path, raw)


def validate_raw_batches(
    raw_dir: Path,
    entries: list[Entry],
    batch_size: int,
    batch_numbers: list[int],
) -> tuple[list[dict[str, Any]], list[str]]:
    failures: list[dict[str, Any]] = []
    parsed_ids: set[str] = set()
    expected_ids = {entry.id for entry in entries}
    for batch_number in batch_numbers:
        raw_path = batch_path(raw_dir, batch_number)
        if not raw_path.exists():
            failures.append({"batch": batch_number, "input_ids": [], "error": "missing raw batch", "raw_path": str(raw_path)})
            continue
        raw = read_raw(raw_path)
        input_ids = [str(value) for value in raw.get("input_ids", [])]
        if raw.get("error"):
            failures.append({"batch": batch_number, "input_ids": input_ids, "error": raw.get("error"), "raw_path": str(raw_path)})
            continue
        try:
            parsed = parse_json_object(str(raw.get("answer") or ""))
        except Exception as exc:
            failures.append({"batch": batch_number, "input_ids": input_ids, "error": str(exc), "raw_path": str(raw_path)})
            continue
        for item in parsed.get("entries", []):
            if isinstance(item, dict):
                entry_id = normalize_spaces(str(item.get("id", "")))
                if entry_id in expected_ids:
                    parsed_ids.add(entry_id)
    missing = sorted(expected_ids - parsed_ids)
    return failures, missing


def build_graph(
    raw_dir: Path,
    entries: list[Entry],
    batch_size: int,
    batch_numbers: list[int],
) -> dict[str, Any]:
    entry_nodes: dict[str, dict[str, Any]] = {
        entry.id: {
            "id": entry.id,
            "type": "entry",
            "label": entry.headword,
            "letter": entry.letter,
            "source_file": entry.source_file,
            "paragraph_index": entry.paragraph_index,
            "graph_status": "source_only",
        }
        for entry in entries
    }
    concept_nodes: dict[str, dict[str, Any]] = {}
    edges: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    batch_rows: list[dict[str, Any]] = []
    processed_ids: set[str] = set()
    fallback_ids: set[str] = set()
    category_counter: Counter[str] = Counter()
    relation_counter: Counter[str] = Counter()
    key_usage: Counter[str] = Counter()
    token_totals: Counter[str] = Counter()

    for batch_number in batch_numbers:
        raw_path = batch_path(raw_dir, batch_number)
        if not raw_path.exists():
            failures.append({"batch": batch_number, "error": "missing raw batch", "raw_path": str(raw_path)})
            continue
        raw = read_raw(raw_path)
        if raw.get("key_alias"):
            key_usage[str(raw["key_alias"])] += 1
        for field in ("prompt_tokens", "completion_tokens", "total_tokens", "estimated_tokens"):
            if isinstance(raw.get(field), int):
                token_totals[field] += raw[field]
        batch_rows.append(batch_row(raw))
        for item in raw.get("fallback_entries", []) or []:
            if isinstance(item, dict) and item.get("id"):
                fallback_ids.add(str(item["id"]))
        for item in raw.get("micro_failures", []) or []:
            if isinstance(item, dict) and item.get("fallback") == "local" and item.get("id"):
                fallback_ids.add(str(item["id"]))
        if raw.get("fallback_batch"):
            fallback_ids.update(str(value) for value in raw.get("input_ids", []))
        if raw.get("error"):
            failures.append({"batch": batch_number, "input_ids": raw.get("input_ids", []), "error": raw.get("error"), "raw_path": str(raw_path)})
            continue
        try:
            parsed = parse_json_object(str(raw.get("answer") or ""))
        except Exception as exc:
            failures.append({"batch": batch_number, "input_ids": raw.get("input_ids", []), "error": str(exc), "raw_path": str(raw_path)})
            continue
        for item in parsed.get("entries", []):
            if not isinstance(item, dict):
                continue
            entry_id = normalize_spaces(str(item.get("id", "")))
            if entry_id not in entry_nodes:
                continue
            processed_ids.add(entry_id)
            category = normalize_category(str(item.get("category") or ""))
            entry_nodes[entry_id]["category"] = category
            entry_nodes[entry_id]["graph_status"] = "fallback" if entry_id in fallback_ids else "llm_parsed"
            category_counter[category] += 1
            aliases = [normalize_spaces(str(value)) for value in item.get("aliases", []) if normalize_spaces(str(value))]
            concepts = [normalize_spaces(str(value)) for value in item.get("concepts", []) if normalize_spaces(str(value))]
            entry_nodes[entry_id]["aliases"] = aliases[:6]
            entry_nodes[entry_id]["concepts"] = concepts[:6]
            for concept in concepts[:6]:
                cid = concept_id(concept)
                concept_nodes.setdefault(cid, {"id": cid, "type": "concept", "label": concept})
                edges.append({"source": entry_id, "target": cid, "type": "has_concept", "weight": 1.0})
            for relation_item in item.get("relations", [])[:4]:
                if not isinstance(relation_item, dict):
                    continue
                target = normalize_spaces(str(relation_item.get("target") or ""))
                if not target:
                    continue
                relation = normalize_relation(str(relation_item.get("relation") or "related_to"))
                target_id = concept_id(target)
                if normalize_spaces(str(relation_item.get("target_type") or "concept")) == "entry":
                    wanted = strip_accents(target)
                    for node in entry_nodes.values():
                        if strip_accents(str(node["label"])) == wanted:
                            target_id = str(node["id"])
                            break
                if target_id.startswith("concept:"):
                    concept_nodes.setdefault(target_id, {"id": target_id, "type": "concept", "label": target})
                evidence = normalize_spaces(str(relation_item.get("evidence") or ""))[:140]
                confidence = relation_item.get("confidence")
                if not isinstance(confidence, (int, float)):
                    confidence = 0.72 if evidence else 0.4
                edges.append(
                    {
                        "source": entry_id,
                        "target": target_id,
                        "type": relation,
                        "weight": 2.0,
                        "evidence_text": evidence,
                        "source_entry_id": entry_id,
                        "confidence": confidence,
                    }
                )
                relation_counter[relation] += 1

    nodes = list(entry_nodes.values()) + list(concept_nodes.values())
    missing_ids = sorted(set(entry_nodes) - processed_ids)
    manifest = {
        "name": "dictionary_graph",
        "schema_version": DICTIONARY_SCHEMA_VERSION,
        "created_at": iso_now(),
        "source_entry_count": len(entries),
        "rich_entry_count": sum(1 for entry in entries if entry.rich_blocks),
        "raw_batch_count": len(batch_rows),
        "total_expected_batches": len(batch_numbers),
        "parsed_entry_count": len(processed_ids),
        "missing_entry_count": len(missing_ids),
        "missing_entries": missing_ids,
        "fallback_entry_count": len(fallback_ids),
        "fallback_entries": sorted(fallback_ids),
        "node_count": len(nodes),
        "entry_node_count": len(entry_nodes),
        "concept_node_count": len(concept_nodes),
        "edge_count": len(edges),
        "failure_count": len(failures),
        "key_usage_counts": dict(key_usage),
        "token_totals": dict(token_totals),
        "relation_counts": dict(relation_counter.most_common()),
        "top_categories": dict(category_counter.most_common()),
        "partial": bool(failures) or bool(missing_ids),
    }
    return {
        "nodes": nodes,
        "edges": edges,
        "batches": batch_rows,
        "failures": failures,
        "manifest": manifest,
    }


def write_graph_artifacts(
    *,
    run_dir: Path,
    graph: dict[str, Any],
    entries: list[Entry],
    config: dict[str, Any],
    provider: str,
    model: str,
    total_batches: int,
    sqlite_path: Path | None = None,
) -> None:
    manifest = graph["manifest"]
    manifest.update(
        {
            "provider": provider,
            "model": model,
            "output_dir": str(run_dir),
            "letters": config["letters"],
            "source_files": config["source_files"],
            "batch_size": config["batch_size"],
            "entry_char_limit": config["entry_char_limit"],
            "total_expected_batches": total_batches,
            "sqlite_path": str(sqlite_path or (run_dir / "dictionary_graph.sqlite")),
        }
    )
    write_jsonl(run_dir / "nodes.jsonl", graph["nodes"])
    write_jsonl(run_dir / "edges.jsonl", graph["edges"])
    write_jsonl(run_dir / "batches.jsonl", graph["batches"])
    write_entries(run_dir / "rich_entries.jsonl", entries)
    write_jsonl(run_dir / "validation_errors.jsonl", graph.get("validation_errors", []))
    if graph["failures"]:
        write_jsonl(run_dir / "failures.jsonl", graph["failures"])
    else:
        (run_dir / "failures.jsonl").unlink(missing_ok=True)
    write_json(run_dir / "manifest.json", manifest)
    write_summary(run_dir / "graph_summary.md", manifest)
    write_quality_report(
        run_dir / "graph_quality_report.md",
        manifest=manifest,
        metrics=graph.get("quality_metrics") or manifest.get("quality_metrics") or {},
        errors=graph.get("validation_errors", []),
    )
    write_sqlite_store(
        sqlite_path or (run_dir / "dictionary_graph.sqlite"),
        entries=entries,
        nodes=graph["nodes"],
        edges=graph["edges"],
        batches=graph.get("batches", []),
        validation_errors=graph.get("validation_errors", []),
        manifest=manifest,
    )


def run_quality_pass(
    *,
    client: Any,
    run_dir: Path,
    graph: dict[str, Any],
    entries: list[Entry],
    ontology: Any,
    provider: str,
    model: str,
    temperature: float,
    quality_pass: str,
    max_completion_tokens: int,
    progress: bool,
) -> dict[str, Any]:
    selected = select_quality_edges(graph.get("edges", []), quality_pass=quality_pass, weak_threshold=0.6)
    if not selected:
        graph["manifest"]["quality_pass_edge_count"] = 0
        return graph

    quality_dir = run_dir / "quality_batches"
    quality_dir.mkdir(parents=True, exist_ok=True)
    entries_by_id = {entry.id: entry for entry in entries}
    updates: list[dict[str, Any]] = []
    tracker = ProgressTracker(stage="quality", total_items=len(selected), item_name="edge", enabled=progress)
    for edge_index, edge in enumerate(selected, start=1):
        raw_path = quality_dir / f"quality_{edge_index:04d}.json"
        if raw_path.is_file():
            raw = read_raw(raw_path)
            parsed = raw.get("parsed") if isinstance(raw.get("parsed"), dict) else None
            if parsed:
                updates.append(parsed)
                tracker.update(done_items=edge_index, note="cached")
                continue
        entry = entries_by_id.get(str(edge.get("source_entry_id")))
        payload = {
            "edge": {
                "edge_id": edge.get("edge_id"),
                "source": edge.get("source"),
                "target": edge.get("target"),
                "type": edge.get("type"),
                "evidence_text": edge.get("evidence_text"),
                "confidence": edge.get("confidence"),
            },
            "source_entry": {
                "id": entry.id,
                "headword": entry.headword,
                "text": (entry.raw_docx_text or entry.text)[:900],
            }
            if entry
            else None,
        }
        result = client.generate(
            [
                {"role": "system", "content": quality_system_prompt()},
                {"role": "user", "content": json.dumps(payload, ensure_ascii=False)},
            ],
            model=model,
            temperature=temperature,
            max_completion_tokens=max_completion_tokens,
        )
        parsed: dict[str, Any] | None = None
        parse_error: str | None = None
        if result.error is None:
            try:
                parsed = parse_json_object(result.answer)
                parsed["edge_id"] = edge.get("edge_id")
                updates.append(parsed)
            except Exception as exc:  # noqa: BLE001 - quality pass should be auditable and non-fatal.
                parse_error = str(exc)
        raw = provider_result_to_raw(edge_index, [], result)
        raw.update(
            {
                "edge_id": edge.get("edge_id"),
                "quality_pass": quality_pass,
                "input_edge": edge,
                "parsed": parsed,
                "parse_error": parse_error,
                "generated_at": iso_now(),
            }
        )
        write_json(raw_path, raw)
        tracker.update(done_items=edge_index, note="error" if result.error or parse_error else None)

    updated_graph = apply_quality_updates(graph, updates)
    updated_graph = finalize_dictionary_graph(
        graph=updated_graph,
        entries=entries,
        ontology=ontology,
        extractor=provider,
        prompt_version=PROMPT_VERSION,
    )
    updated_graph["manifest"]["quality_pass"] = quality_pass
    updated_graph["manifest"]["quality_pass_edge_count"] = len(selected)
    updated_graph["manifest"]["quality_update_count"] = len(updates)
    return updated_graph


def select_quality_edges(edges: list[dict[str, Any]], *, quality_pass: str, weak_threshold: float) -> list[dict[str, Any]]:
    if quality_pass == "none":
        return []
    selected: list[dict[str, Any]] = []
    for edge in edges:
        edge_type = str(edge.get("type") or "")
        if edge_type in {"has_alias", "has_concept", "in_category"}:
            continue
        if quality_pass == "all":
            selected.append(edge)
            continue
        confidence = float(edge.get("confidence", 0.0)) if isinstance(edge.get("confidence"), (int, float)) else 0.0
        if confidence < weak_threshold or not normalize_spaces(str(edge.get("evidence_text") or "")):
            selected.append(edge)
    return selected


def apply_quality_updates(graph: dict[str, Any], updates: list[dict[str, Any]]) -> dict[str, Any]:
    by_id = {str(update.get("edge_id")): update for update in updates if update.get("edge_id")}
    if not by_id:
        return graph
    edges: list[dict[str, Any]] = []
    dropped: list[dict[str, Any]] = []
    for edge in graph.get("edges", []):
        update = by_id.get(str(edge.get("edge_id")))
        if update is None:
            edges.append(edge)
            continue
        if update.get("keep") is False:
            dropped.append({"edge_id": edge.get("edge_id"), "reason": update.get("reason")})
            continue
        merged = dict(edge)
        for field in ("evidence_text", "confidence"):
            if update.get(field) not in (None, ""):
                merged[field] = update[field]
        metadata = dict(merged.get("metadata") or {})
        metadata["quality_pass"] = {key: value for key, value in update.items() if key not in {"edge_id", "keep"}}
        merged["metadata"] = metadata
        edges.append(merged)
    updated = dict(graph)
    updated["edges"] = edges
    updated["quality_dropped_edges"] = dropped
    manifest = dict(graph.get("manifest") or {})
    manifest["quality_dropped_edge_count"] = len(dropped)
    updated["manifest"] = manifest
    return updated


def quality_system_prompt() -> str:
    return (
        "Bạn là critic cho graph từ điển. Final answer chỉ là JSON hợp lệ bắt đầu bằng {.\n"
        'Schema: {"keep":true|false,"confidence":0.0-1.0,"evidence_text":"trích dẫn ngắn từ source_entry",'
        '"reason":"lý do ngắn"}.\n'
        "Giữ edge nếu quan hệ được chứng minh trực tiếp bởi mục từ. Bỏ edge nếu evidence không hỗ trợ relation."
    )


def write_visualization(path: Path, nodes: list[dict[str, Any]], edges: list[dict[str, Any]], manifest: dict[str, Any]) -> None:
    compact_nodes = [
        {
            "id": node.get("id"),
            "label": node.get("label") or node.get("id"),
            "type": node.get("type"),
            "category": node.get("category") or ("concept" if node.get("type") == "concept" else "uncategorized"),
            "letter": node.get("letter"),
            "status": node.get("graph_status"),
            "aliases": node.get("aliases", []),
            "concepts": node.get("concepts", []),
        }
        for node in nodes
    ]
    compact_edges = [
        {
            "source": edge.get("source"),
            "target": edge.get("target"),
            "type": edge.get("type"),
            "evidence": edge.get("evidence_text") or edge.get("evidence", ""),
            "source_entry_id": edge.get("source_entry_id"),
            "confidence": edge.get("confidence"),
        }
        for edge in edges
    ]
    categories = sorted({node["category"] for node in compact_nodes if node.get("type") == "entry"})
    category_options = "\n".join(f'<option value="{escape(category)}">{escape(category)}</option>' for category in categories)
    relations = sorted({str(edge.get("type")) for edge in compact_edges if edge.get("type")})
    relation_options = "\n".join(f'<option value="{escape(relation)}">{escape(relation)}</option>' for relation in relations)
    data = json.dumps({"nodes": compact_nodes, "edges": compact_edges, "manifest": manifest}, ensure_ascii=False)
    data = data.replace("</script", "<\\/script")
    html = f"""<!doctype html>
<html lang="vi">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>Dictionary Graph</title>
  <style>
    * {{ box-sizing: border-box; }}
    body {{ margin: 0; font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; color: #151515; background: #f7f7f4; }}
    .app {{ min-height: 100vh; display: grid; grid-template-columns: minmax(280px, 360px) 1fr; }}
    aside {{ padding: 18px; background: #fff; border-right: 1px solid #deded8; overflow: auto; }}
    main {{ position: relative; min-width: 0; }}
    h1 {{ margin: 0 0 8px; font-size: 20px; }}
    .sub {{ color: #666a73; font-size: 13px; line-height: 1.45; margin-bottom: 16px; }}
    .stats {{ display: grid; grid-template-columns: 1fr 1fr; gap: 8px; margin-bottom: 14px; }}
    .stat {{ padding: 10px; border: 1px solid #deded8; border-radius: 8px; background: #fbfbf9; }}
    .stat b {{ display: block; font-size: 20px; }}
    .stat span, label {{ color: #666a73; font-size: 12px; }}
    label {{ display: block; margin: 12px 0 6px; }}
    input, select, button {{ width: 100%; border: 1px solid #deded8; border-radius: 8px; padding: 10px; font: inherit; background: #fff; }}
    button {{ margin-top: 12px; color: white; background: #111; border-color: #111; cursor: pointer; }}
    #graph {{ width: 100%; height: 100vh; display: block; background: radial-gradient(circle at 50% 35%, #fff 0, #f7f7f4 55%, #eee 100%); }}
    .details {{ position: absolute; top: 16px; right: 16px; width: min(360px, calc(100vw - 32px)); max-height: calc(100vh - 32px); overflow: auto; background: rgba(255,255,255,.94); border: 1px solid #deded8; border-radius: 12px; padding: 14px; box-shadow: 0 16px 50px rgba(0,0,0,.12); }}
    .details h2 {{ font-size: 16px; margin: 0 0 8px; overflow-wrap: anywhere; }}
    .meta {{ color: #666a73; font-size: 12px; line-height: 1.45; }}
    .edge-list {{ margin-top: 12px; display: grid; gap: 8px; }}
    .edge-row {{ border: 1px solid #e2e2dc; border-radius: 8px; padding: 8px; font-size: 12px; line-height: 1.4; }}
    .evidence {{ color: #4b5563; margin-top: 4px; overflow-wrap: anywhere; }}
    .chips {{ display: flex; flex-wrap: wrap; gap: 6px; margin: 10px 0; }}
    .chip {{ border-radius: 999px; background: #f0f0ec; padding: 5px 8px; font-size: 12px; }}
    .hidden {{ display: none; }}
    @media (max-width: 820px) {{ .app {{ grid-template-columns: 1fr; }} aside {{ border-right: 0; border-bottom: 1px solid #deded8; max-height: 44vh; }} #graph {{ height: 70vh; }} .details {{ position: fixed; top: auto; left: 12px; right: 12px; bottom: 12px; width: auto; max-height: 45vh; }} }}
  </style>
</head>
<body>
<div class="app">
  <aside>
    <h1>Dictionary Graph</h1>
    <div class="sub">Provider: {escape(str(manifest.get("provider")))} · Model: {escape(str(manifest.get("model")))} · Letters: {escape(",".join(manifest.get("letters", [])))}</div>
    <div class="stats">
      <div class="stat"><b>{manifest.get("entry_node_count")}</b><span>entries</span></div>
      <div class="stat"><b>{manifest.get("concept_node_count")}</b><span>concepts</span></div>
      <div class="stat"><b>{manifest.get("edge_count")}</b><span>edges</span></div>
      <div class="stat"><b>{manifest.get("failure_count")}</b><span>failures</span></div>
    </div>
    <label for="query">Search</label><input id="query" placeholder="term, id, concept..." />
    <label for="category">Category</label><select id="category"><option value="">All</option>{category_options}</select>
    <label for="relation">Relation</label><select id="relation"><option value="">All</option>{relation_options}</select>
    <label for="confidence">Minimum confidence</label><input id="confidence" type="number" min="0" max="1" step="0.05" value="0" />
    <label for="limit">Node limit</label><input id="limit" type="number" min="50" step="50" value="600" />
    <button id="apply">Apply</button>
  </aside>
  <main><canvas id="graph"></canvas><section id="details" class="details hidden"></section></main>
</div>
<script id="graph-data" type="application/json">{data}</script>
<script>
const raw = JSON.parse(document.getElementById('graph-data').textContent);
const allNodes = raw.nodes, allEdges = raw.edges;
const byId = new Map(allNodes.map(n => [n.id, n]));
let nodes = [], edges = [], selected = null, transform = {{x: 0, y: 0, scale: 1}};
const canvas = document.getElementById('graph'), ctx = canvas.getContext('2d'), details = document.getElementById('details');
let width = 0, height = 0, dpr = 1;
function resize() {{ const r = canvas.getBoundingClientRect(); width = r.width; height = r.height; dpr = window.devicePixelRatio || 1; canvas.width = width*dpr; canvas.height = height*dpr; ctx.setTransform(dpr,0,0,dpr,0,0); draw(); }}
window.addEventListener('resize', resize);
function color(n) {{ if (selected && selected.id === n.id) return '#c72c25'; if (n.type === 'concept') return '#d6a800'; return {{'vũ khí/trang bị':'#228b22','đạn dược/thuốc nổ':'#c72c25','khí tài đo đạc/trinh sát':'#3b82f6','bản đồ/địa hình':'#8b5cf6','chỉ huy/huấn luyện':'#0f766e','bảo đảm kỹ thuật':'#f97316','khái niệm tác chiến':'#111827','tổ chức/lực lượng':'#7c3aed','khí tượng/địa vật':'#0891b2','khác':'#6b7280'}}[n.category] || '#777'; }}
function activeEdges() {{ const rel = document.getElementById('relation').value, minConf = Number(document.getElementById('confidence').value) || 0; return allEdges.filter(e => (!rel || e.type === rel) && (Number(e.confidence ?? 1) >= minConf)); }}
function apply() {{ const q = document.getElementById('query').value.trim().toLowerCase(), cat = document.getElementById('category').value, limit = Number(document.getElementById('limit').value) || 600, edgePool = activeEdges(); const degree = new Map(); edgePool.forEach(e => {{ degree.set(e.source, (degree.get(e.source)||0)+1); degree.set(e.target, (degree.get(e.target)||0)+1); }}); let seed = allNodes.filter(n => (!cat || n.type !== 'entry' || n.category === cat) && (!q || [n.id,n.label,n.category,...(n.aliases||[]),...(n.concepts||[])].join(' ').toLowerCase().includes(q))); seed.sort((a,b)=>(degree.get(b.id)||0)-(degree.get(a.id)||0)); const live = new Set(seed.slice(0,limit).map(n=>n.id)); edgePool.forEach(e => {{ if (live.has(e.source)||live.has(e.target)) {{ live.add(e.source); live.add(e.target); }} }}); nodes = allNodes.filter(n => live.has(n.id)).slice(0, Math.max(limit, live.size)); const live2 = new Set(nodes.map(n=>n.id)); edges = edgePool.filter(e => live2.has(e.source)&&live2.has(e.target)); init(); simulate(170); }}
function init() {{ nodes.forEach((n,i)=>{{ if (!Number.isFinite(n.x)) {{ const a=i*2.399963, r=45+Math.sqrt(i)*12; n.x=width/2+Math.cos(a)*r; n.y=height/2+Math.sin(a)*r; n.vx=0; n.vy=0; }} }}); }}
function simulate(iter) {{ let t=0; const pairs = edges.map(e=>[byId.get(e.source),byId.get(e.target)]).filter(p=>p[0]&&p[1]); function step() {{ for (let k=0;k<3;k++) {{ t++; const a=Math.max(.02,1-t/iter); nodes.forEach(n=>{{ n.vx+=(width/2-n.x)*.0008*a; n.vy+=(height/2-n.y)*.0008*a; }}); for (const [x,y] of pairs) {{ const dx=y.x-x.x, dy=y.y-x.y, d=Math.hypot(dx,dy)||1, f=(d-95)*.006*a; x.vx+=dx/d*f; x.vy+=dy/d*f; y.vx-=dx/d*f; y.vy-=dy/d*f; }} for (let i=0;i<nodes.length;i++) for (let j=i+1;j<Math.min(nodes.length,i+55);j++) {{ const x=nodes[i], y=nodes[j], dx=x.x-y.x, dy=x.y-y.y, d2=dx*dx+dy*dy+.01; if (d2<12000) {{ const f=34/d2*a; x.vx+=dx*f; x.vy+=dy*f; y.vx-=dx*f; y.vy-=dy*f; }} }} nodes.forEach(n=>{{ n.vx*=.86; n.vy*=.86; n.x+=n.vx; n.y+=n.vy; }}); }} draw(); if(t<iter) requestAnimationFrame(step); }} requestAnimationFrame(step); }}
function draw() {{ ctx.clearRect(0,0,width,height); ctx.save(); ctx.translate(transform.x,transform.y); ctx.scale(transform.scale,transform.scale); ctx.strokeStyle='rgba(80,80,80,.16)'; ctx.lineWidth=1/transform.scale; edges.forEach(e=>{{ const a=byId.get(e.source), b=byId.get(e.target); if(!a||!b) return; ctx.beginPath(); ctx.moveTo(a.x,a.y); ctx.lineTo(b.x,b.y); ctx.stroke(); }}); nodes.forEach(n=>{{ ctx.beginPath(); ctx.fillStyle=color(n); ctx.arc(n.x,n.y,n.type==='entry'?5.4:3.7,0,Math.PI*2); ctx.fill(); }}); if(nodes.length<420||transform.scale>.8) {{ ctx.font=`${{11/transform.scale}}px Inter,sans-serif`; ctx.fillStyle='rgba(20,20,20,.78)'; nodes.forEach(n=>{{ if(n.type==='concept'&&nodes.length>340) return; ctx.fillText(String(n.label||n.id).slice(0,40), n.x+7, n.y+4/transform.scale); }}); }} ctx.restore(); }}
function screenToWorld(x,y) {{ return {{x:(x-transform.x)/transform.scale,y:(y-transform.y)/transform.scale}}; }}
canvas.addEventListener('click', ev=>{{ const r=canvas.getBoundingClientRect(), p=screenToWorld(ev.clientX-r.left,ev.clientY-r.top); let hit=null,best=14/transform.scale; nodes.forEach(n=>{{ const d=Math.hypot(n.x-p.x,n.y-p.y); if(d<best){{best=d; hit=n;}} }}); if(!hit){{selected=null;details.classList.add('hidden');draw();return;}} selected=hit; const related = edges.filter(e => e.source === hit.id || e.target === hit.id).slice(0,16); details.classList.remove('hidden'); details.innerHTML=`<h2>${{esc(hit.label||hit.id)}}</h2><div class="meta">${{esc(hit.id)}} · ${{esc(hit.type)}} · ${{esc(hit.category||'')}}</div><div class="chips">${{[...(hit.aliases||[]),...(hit.concepts||[])].slice(0,10).map(x=>`<span class="chip">${{esc(x)}}</span>`).join('')}}</div><div class="edge-list">${{related.map(e=>`<div class="edge-row"><b>${{esc(e.type)}}</b> ${{esc(e.source)}} → ${{esc(e.target)}}<div class="meta">entry ${{esc(e.source_entry_id||'')}} · confidence ${{esc(e.confidence??'')}}</div><div class="evidence">${{esc(e.evidence||'')}}</div></div>`).join('')}}</div>`; draw(); }});
canvas.addEventListener('wheel', ev=>{{ ev.preventDefault(); const r=canvas.getBoundingClientRect(), mx=ev.clientX-r.left, my=ev.clientY-r.top, before=screenToWorld(mx,my), f=ev.deltaY<0?1.1:.9; transform.scale=Math.max(.25,Math.min(4,transform.scale*f)); transform.x=mx-before.x*transform.scale; transform.y=my-before.y*transform.scale; draw(); }},{{passive:false}});
let pan=null; canvas.addEventListener('pointerdown',ev=>{{pan={{x:ev.clientX,y:ev.clientY,tx:transform.x,ty:transform.y}};}}); canvas.addEventListener('pointermove',ev=>{{if(!pan)return; transform.x=pan.tx+ev.clientX-pan.x; transform.y=pan.ty+ev.clientY-pan.y; draw();}}); canvas.addEventListener('pointerup',()=>pan=null);
function esc(v) {{ return String(v??'').replace(/[&<>"']/g,c=>({{'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#039;'}}[c])); }}
document.getElementById('apply').addEventListener('click', apply); document.getElementById('query').addEventListener('keydown', e=>{{if(e.key==='Enter')apply();}}); document.getElementById('relation').addEventListener('change', apply); document.getElementById('confidence').addEventListener('change', apply);
resize(); apply();
</script>
</body>
</html>"""
    path.write_text(html, encoding="utf-8")


def write_graphml(path: Path, nodes: list[dict[str, Any]], edges: list[dict[str, Any]]) -> None:
    lines = [
        '<?xml version="1.0" encoding="UTF-8"?>',
        '<graphml xmlns="http://graphml.graphdrawing.org/xmlns">',
        '  <key id="label" for="node" attr.name="label" attr.type="string"/>',
        '  <key id="type" for="node" attr.name="type" attr.type="string"/>',
        '  <key id="category" for="node" attr.name="category" attr.type="string"/>',
        '  <key id="edge_type" for="edge" attr.name="type" attr.type="string"/>',
        '  <key id="evidence" for="edge" attr.name="evidence" attr.type="string"/>',
        '  <graph id="G" edgedefault="directed">',
    ]
    for node in nodes:
        node_id = xml_escape(str(node["id"]))
        lines.append(f'    <node id="{node_id}">')
        for key in ("label", "type", "category"):
            value = node.get(key)
            if value is not None:
                lines.append(f'      <data key="{key}">{xml_escape(str(value))}</data>')
        lines.append("    </node>")
    for index, edge in enumerate(edges):
        source = xml_escape(str(edge["source"]))
        target = xml_escape(str(edge["target"]))
        lines.append(f'    <edge id="e{index}" source="{source}" target="{target}">')
        lines.append(f'      <data key="edge_type">{xml_escape(str(edge.get("type", "")))}</data>')
        evidence = edge.get("evidence_text") or edge.get("evidence")
        if evidence:
            lines.append(f'      <data key="evidence">{xml_escape(str(evidence))}</data>')
        lines.append("    </edge>")
    lines.extend(["  </graph>", "</graphml>"])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_summary(path: Path, manifest: dict[str, Any]) -> None:
    lines = [
        "# Dictionary Graph Build",
        "",
        f"- Output: `{manifest.get('output_dir')}`",
        f"- Provider: `{manifest.get('provider')}`",
        f"- Model: `{manifest.get('model')}`",
        f"- Letters: `{','.join(manifest.get('letters', []))}`",
        f"- Source entries: {manifest.get('source_entry_count')}",
        f"- Parsed entries: {manifest.get('parsed_entry_count')}",
        f"- Fallback entries: {manifest.get('fallback_entry_count')}",
        f"- Nodes: {manifest.get('node_count')}",
        f"- Edges: {manifest.get('edge_count')}",
        f"- Failures: {manifest.get('failure_count')}",
        f"- Validation errors: {manifest.get('validation_error_count')}",
        f"- SQLite: `{manifest.get('sqlite_path')}`",
        "",
        "## Quality Metrics",
        "",
    ]
    for key, value in (manifest.get("quality_metrics") or {}).items():
        if isinstance(value, (str, int, float)) or value is None:
            lines.append(f"- `{key}`: {value}")
    lines.extend([
        "",
        "## Relation Counts",
        "",
    ])
    for relation, count in (manifest.get("relation_counts") or {}).items():
        lines.append(f"- `{relation}`: {count}")
    lines.extend(["", "## Category Counts", ""])
    for category, count in (manifest.get("top_categories") or {}).items():
        lines.append(f"- `{category}`: {count}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def graph_system_prompt() -> str:
    return (
        "Trích xuất graph. Final answer PHẢI bắt đầu bằng ký tự { và chỉ chứa JSON hợp lệ.\n"
        'Schema: {"entries":[{"id":"...","category":"...","aliases":["..."],'
        '"concepts":["..."],"relations":[{"target":"...","relation":"...","target_type":"entry|concept",'
        '"evidence":"trích dẫn ngắn từ mục từ","confidence":0.0-1.0}]}]}\n'
        f"category: {' | '.join(CATEGORIES)}.\n"
        f"relation: {' | '.join(RELATIONS)}.\n"
        "Mỗi entry tối đa 3 concepts, tối đa 1 relation. Không chắc thì relations=[]. "
        f"prompt_version={PROMPT_VERSION}. Giữ nguyên mọi id input."
    )


def repair_system_prompt() -> str:
    return (
        "Final answer PHẢI bắt đầu bằng { và chỉ chứa JSON hợp lệ.\n"
        'Schema: {"entries":[{"id":"...","category":"...","aliases":[],"concepts":[],"relations":[]}]}\n'
        f"category: {' | '.join(CATEGORIES)}.\n"
        "Mỗi entry tối đa 1 concept. relations luôn [] nếu không chắc. Giữ nguyên mọi id input."
    )


def micro_system_prompt() -> str:
    return (
        "Final answer PHẢI bắt đầu bằng { và chỉ chứa JSON object hợp lệ cho đúng 1 mục từ.\n"
        'Schema: {"id":"...","category":"...","aliases":[],"concepts":[],"relations":[]}\n'
        f"category: {' | '.join(CATEGORIES)}.\n"
        "Tối đa 1 concept. relations luôn []. Giữ nguyên id."
    )


def batch_for_number(entries: list[Entry], batch_size: int, batch_number: int) -> list[Entry]:
    offset = (batch_number - 1) * batch_size
    return entries[offset : offset + batch_size]


def batch_path(raw_dir: Path, batch_number: int) -> Path:
    return raw_dir / f"batch_{batch_number:04d}.json"


def provider_result_to_raw(batch_number: int, batch: list[Entry], result: ProviderResult) -> dict[str, Any]:
    return {
        "batch": batch_number,
        "input_ids": [entry.id for entry in batch],
        "answer": result.answer,
        "error": result.error,
        "key_alias": result.key_alias,
        "attempted_aliases": result.attempted_aliases,
        "rejected_aliases": result.rejected_aliases,
        "retry_count": result.retry_count,
        "prompt_tokens": result.prompt_tokens,
        "completion_tokens": result.completion_tokens,
        "total_tokens": result.total_tokens,
        "estimated_tokens": result.estimated_tokens,
        "scheduled_wait_s": result.scheduled_wait_s,
    }


def add_batch_cache_metadata(
    raw: dict[str, Any],
    batch: list[Entry],
    *,
    batch_size: int,
    model: str,
    prompt_version: str,
) -> None:
    raw["model"] = model
    raw["prompt_version"] = prompt_version
    raw["batch_size"] = batch_size
    raw["source_entry_hashes"] = {entry.id: entry_content_hash(entry) for entry in batch}
    raw["cache_key"] = batch_cache_key(batch, batch_size=batch_size, model=model, prompt_version=prompt_version)


def batch_cache_key(batch: list[Entry], *, batch_size: int, model: str, prompt_version: str) -> str:
    payload = {
        "batch_size": batch_size,
        "model": model,
        "prompt_version": prompt_version,
        "entries": [{"id": entry.id, "hash": entry_content_hash(entry)} for entry in batch],
    }
    body = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(body).hexdigest()


def entry_content_hash(entry: Entry) -> str:
    body = json.dumps(
        {
            "id": entry.id,
            "headword": entry.headword,
            "plain_text": entry.plain_text or entry.text,
            "raw_docx_text": entry.raw_docx_text or entry.text,
            "source_file": entry.source_file,
            "paragraph_index": entry.paragraph_index,
        },
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(body).hexdigest()


def backup_raw(path: Path) -> None:
    if not path.exists():
        return
    backup_dir = path.parent / "repair_backups"
    backup_dir.mkdir(exist_ok=True)
    shutil.copy2(path, backup_dir / f"{path.stem}_{int(time.time())}.json")


def read_raw(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def raw_parse_error(path: Path) -> str | None:
    try:
        raw = read_raw(path)
        if raw.get("error"):
            return str(raw["error"])
        parse_json_object(str(raw.get("answer") or ""))
    except Exception as exc:  # noqa: BLE001 - validation should report any parse issue.
        return str(exc)
    return None


def parse_json_object(text: str) -> dict[str, Any]:
    stripped = strip_code_fence(text)
    candidates = [stripped]
    start = stripped.find("{")
    end = stripped.rfind("}")
    if start >= 0 and end > start:
        candidates.append(stripped[start : end + 1])
    last_error: Exception | None = None
    for candidate in candidates:
        try:
            parsed = json.loads(candidate)
        except json.JSONDecodeError as exc:
            last_error = exc
            try:
                parsed, _end = json.JSONDecoder().raw_decode(candidate.lstrip())
            except json.JSONDecodeError as raw_exc:
                last_error = raw_exc
                continue
        if isinstance(parsed, dict):
            return parsed
    raise ValueError(f"could not parse JSON: {last_error}")


def strip_code_fence(text: str) -> str:
    text = text.strip()
    if text.startswith("```"):
        lines = text.splitlines()
        if len(lines) >= 2 and lines[-1].strip() == "```":
            return "\n".join(lines[1:-1]).strip()
    return text


def fallback_item(entry: Entry) -> dict[str, Any]:
    return {
        "id": entry.id,
        "category": "khác",
        "aliases": [entry.headword],
        "concepts": [entry.headword],
        "relations": [],
    }


def normalize_category(value: str) -> str:
    value = normalize_spaces(value)
    if value in CATEGORIES:
        return value
    low = strip_accents(value)
    if any(token in low for token in ("dan", "thuoc no", "dau dan")):
        return "đạn dược/thuốc nổ"
    if any(token in low for token in ("vu khi", "trang bi", "sung", "phao", "ten lua", "khi tai")):
        return "vũ khí/trang bị"
    if any(token in low for token in ("ban do", "dia hinh")):
        return "bản đồ/địa hình"
    if any(token in low for token in ("chi huy", "huan luyen", "dien tap")):
        return "chỉ huy/huấn luyện"
    if any(token in low for token in ("bao dam", "bao duong")):
        return "bảo đảm kỹ thuật"
    if any(token in low for token in ("khi tuong", "do am", "ap suat")):
        return "khí tượng/địa vật"
    return "khác"


def normalize_relation(value: str) -> str:
    relation = normalize_spaces(value)
    aliases = {
        "is_related_to": "related_to",
        "related": "related_to",
        "uses": "used_for",
        "use_for": "used_for",
        "measure": "measures",
        "control": "controls",
        "fire": "fires",
        "support": "supports",
        "located": "located_in",
        "require": "requires",
    }
    relation = aliases.get(relation, relation)
    return relation if relation in RELATIONS else "related_to"


def concept_id(label: str) -> str:
    return "concept:" + strip_accents(label)


def strip_accents(text: str) -> str:
    return dictionary_strip_accents(text)


def normalize_spaces(text: str) -> str:
    return dictionary_normalize_spaces(text)


def slugify(value: str) -> str:
    return dictionary_slugify(value)


def source_file_manifest(source_specs: list[SourceSpec]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for spec in source_specs:
        rows.extend(dictionary_source_file_manifest(spec.path, spec.letters, source_set=spec.name if spec.namespace_ids else None))
    return rows


def sha256_file(path: Path) -> str:
    return dictionary_sha256_file(path)


def write_entries(path: Path, entries: list[Entry]) -> None:
    write_jsonl(path, [entry.__dict__ for entry in entries])


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file_obj:
        for row in rows:
            file_obj.write(json.dumps(row, ensure_ascii=False) + "\n")


def print_json(data: Any) -> None:
    print(json.dumps(data, ensure_ascii=False), flush=True)


def progress_log(enabled: bool, message: str) -> None:
    if enabled:
        print(message, file=sys.stderr, flush=True)


def format_duration(seconds: float) -> str:
    total_seconds = max(0, int(round(seconds)))
    hours, remainder = divmod(total_seconds, 3600)
    minutes, secs = divmod(remainder, 60)
    if hours:
        return f"{hours}h{minutes:02d}m{secs:02d}s"
    if minutes:
        return f"{minutes}m{secs:02d}s"
    return f"{secs}s"


def iso_now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S%z")


def add_usage(counter: Counter[str], result: ProviderResult) -> None:
    for field in ("prompt_tokens", "completion_tokens", "total_tokens", "estimated_tokens"):
        value = getattr(result, field)
        if isinstance(value, int):
            counter[field] += value


def batch_row(raw: dict[str, Any]) -> dict[str, Any]:
    return {
        "batch": raw.get("batch"),
        "input_ids": raw.get("input_ids", []),
        "key_alias": raw.get("key_alias"),
        "retry_count": raw.get("retry_count"),
        "prompt_tokens": raw.get("prompt_tokens"),
        "completion_tokens": raw.get("completion_tokens"),
        "total_tokens": raw.get("total_tokens"),
        "estimated_tokens": raw.get("estimated_tokens"),
        "scheduled_wait_s": raw.get("scheduled_wait_s"),
        "error": raw.get("error"),
        "model": raw.get("model"),
        "prompt_version": raw.get("prompt_version"),
        "cache_key": raw.get("cache_key"),
        "repair_attempt": raw.get("repair_attempt", False),
        "micro_repair": raw.get("micro_repair", False),
        "fallback_batch": raw.get("fallback_batch", False),
        "fallback_entries": raw.get("fallback_entries", []),
    }


def estimate_requested_tokens(messages: list[dict[str, str]], max_completion_tokens: int) -> int:
    chars = sum(len(message.get("content", "")) for message in messages)
    return max(1, (chars + 3) // 4 + 4 * len(messages)) + max(0, max_completion_tokens)


def extract_message_text(message: dict[str, Any]) -> str:
    content = message.get("content")
    if isinstance(content, str) and content:
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, dict):
                text = item.get("text")
                if isinstance(text, str):
                    parts.append(text)
            elif isinstance(item, str):
                parts.append(item)
        if parts:
            return "\n".join(parts)
    return ""


def xml_escape(value: str) -> str:
    return escape(value, quote=True)


def _redact_secret(value: str) -> str:
    value = re.sub(r"gsk_[A-Za-z0-9_-]+", "gsk_***REDACTED***", value)
    value = re.sub(r"tp-[A-Za-z0-9*_-]+", "tp-***REDACTED***", value)
    return value


if __name__ == "__main__":
    raise SystemExit(main())
