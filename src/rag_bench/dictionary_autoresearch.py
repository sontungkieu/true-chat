from __future__ import annotations

import hashlib
import json
import re
import sys
import time
import unicodedata
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Protocol

from rag_bench.chat_service import DEFAULT_MIMO_BASE_URL, build_dictionary_rag_messages, format_dictionary_answer
from rag_bench.dictionary import DEFAULT_DICTIONARY_SOURCE_DIR, DictionaryLoadResult, load_dictionary_documents
from rag_bench.groq_client import GenerationResult, OpenAICompatibleClient, RoundRobinGroqClient
from rag_bench.retrievers import DictionaryGraphRetriever
from rag_bench.secrets import ApiKey, load_env_api_key, load_env_values
from rag_bench.types import Query, RetrievalHit


DEFAULT_AUTORESEARCH_ARTIFACT = Path("runs/pb_dictionary_base_supp2021_prod_graph")
DEFAULT_AUTORESEARCH_OUTPUT_DIR = Path("runs/dictionary_autoresearch")
DEFAULT_AUTORESEARCH_LETTERS = (
    "A",
    "B",
    "C",
    "D",
    "Đ",
    "F",
    "G",
    "H",
    "K",
    "L",
    "M",
    "N",
    "O",
    "P",
    "Q",
    "R",
    "S",
    "T",
    "U",
    "V",
    "X",
    "Y",
)
FAILURE_CATEGORIES = {"retrieval", "answer", "citation", "ui/render", "ambiguous"}


class AutoresearchGenerationClient(Protocol):
    def generate(
        self,
        messages: list[dict[str, str]],
        *,
        model: str | None = None,
        temperature: float = 0.0,
        max_completion_tokens: int = 512,
    ) -> GenerationResult: ...


@dataclass(frozen=True)
class DictionaryAutoresearchConfig:
    artifact_dir: Path | None = DEFAULT_AUTORESEARCH_ARTIFACT
    source_dir: Path | None = DEFAULT_DICTIONARY_SOURCE_DIR
    letters: tuple[str, ...] = DEFAULT_AUTORESEARCH_LETTERS
    output_root: Path = DEFAULT_AUTORESEARCH_OUTPUT_DIR
    run_name: str | None = None
    rounds: int = 1
    limit: int = 20
    top_k: int = 5
    max_context_chars: int = 2500
    max_completion_tokens: int = 512
    source_classification: str = "semi-private"
    provider: str = "mimo"
    model: str = "mimo-v2.5-pro"
    dry_run_model: bool = False
    trusted_models: tuple[str, ...] = ()
    mimo_env_file: Path = Path(".secrets/.env")
    mimo_api_key_var: str = "MIMO_API_KEY"
    mimo_base_url: str = DEFAULT_MIMO_BASE_URL
    local_env_file: Path | None = None
    local_api_key_var: str = "LOCAL_API_KEY"
    local_base_url: str = "http://127.0.0.1:8000/v1"
    confirmations: int = 2
    judge_json_retries: int = 2
    strict_acronym_rank: bool = True
    progress: bool = False
    feedback_run_dirs: tuple[Path, ...] = ()
    resume: bool = False


@dataclass(frozen=True)
class AutoresearchCase:
    id: str
    query: str
    expected_doc_id: str | None
    expected_title: str | None
    reference_snippet: str
    kind: str
    source: str
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class RetrievalJudgement:
    passed: bool
    category: str
    reason: str
    expected_rank: int | None
    observed_top_ids: list[str]
    observed_top_titles: list[str]


@dataclass(frozen=True)
class AnswerJudgement:
    status: str
    passed: bool | None
    category: str
    reason: str
    raw: str = ""
    json_retry_count: int = 0


@dataclass(frozen=True)
class CaseEvaluation:
    round_index: int
    case_id: str
    query: str
    expected_doc_id: str | None
    expected_title: str | None
    retrieval: dict[str, Any]
    answer: dict[str, Any]
    failure_category: str | None = None
    failure_reason: str | None = None


@dataclass(frozen=True)
class FailureRecord:
    id: str
    category: str
    expected_doc_id: str | None
    expected_title: str | None
    kind: str
    count: int
    case_ids: list[str]
    queries: list[str]
    reason: str
    suggested_files: list[str]
    acceptance_tests: list[str]


@dataclass(frozen=True)
class CoordinatorDecision:
    id: str
    failure_id: str
    status: str
    status_options: list[str]
    category: str
    expected_title: str | None
    queries: list[str]
    evidence_files: list[str]
    verification_checklist: list[str]
    suggested_files: list[str]
    acceptance_tests: list[str]
    decision_reason: str = ""
    next_action: str = "verify before assigning or editing"
    coordinator_notes: str = ""


def run_dictionary_autoresearch(
    config: DictionaryAutoresearchConfig,
    *,
    generation_client: AutoresearchGenerationClient | None = None,
) -> dict[str, Any]:
    validate_autoresearch_config(config)
    started = time.perf_counter()
    dictionary = load_dictionary_documents(
        artifact_dir=config.artifact_dir,
        source_dir=config.source_dir,
        letters=config.letters,
        required=True,
    )
    retriever = DictionaryGraphRetriever()
    retriever.build(dictionary.documents)
    run_dir = _run_dir(config)
    cases_path = run_dir / "cases.jsonl"
    rounds_path = run_dir / "rounds.jsonl"
    if config.resume:
        if not run_dir.is_dir():
            raise ValueError(f"--resume requires an existing run directory: {run_dir}")
        if not cases_path.is_file():
            raise ValueError(f"--resume requires existing cases.jsonl in {run_dir}")
        if (run_dir / "summary.md").is_file():
            raise ValueError(f"--resume refused finalized run directory: {run_dir}")
        cases = [_case_from_jsonl_row(row) for row in read_jsonl(cases_path)]
        rounds_path.touch(exist_ok=True)
        evaluations = [_evaluation_from_jsonl_row(row) for row in read_jsonl(rounds_path)]
        completed_pairs = {(evaluation.round_index, evaluation.case_id) for evaluation in evaluations}
        _emit_progress(
            config,
            (
                f"resumed run_dir={run_dir} cases={len(cases)} rounds={config.rounds} "
                f"completed={len(completed_pairs)} model_calls={'off' if config.dry_run_model else 'on'}"
            ),
        )
    else:
        cases = generate_autoresearch_cases(
            dictionary,
            limit=config.limit,
            feedback_run_dirs=config.feedback_run_dirs,
        )
        run_dir.mkdir(parents=True, exist_ok=False)
        _write_jsonl(cases_path, [asdict(case) for case in cases])
        rounds_path.write_text("", encoding="utf-8")
        evaluations = []
        completed_pairs = set()
        _emit_progress(
            config,
            f"started run_dir={run_dir} cases={len(cases)} rounds={config.rounds} model_calls={'off' if config.dry_run_model else 'on'}",
        )
    _emit_progress(
        config,
        f"pending evaluations={max(0, config.rounds * len(cases) - len(completed_pairs))}",
    )

    client = generation_client
    has_pending = any((round_index, case.id) not in completed_pairs for round_index in range(1, config.rounds + 1) for case in cases)
    if client is None and not config.dry_run_model and has_pending:
        _emit_progress(config, f"building {config.provider} generation client model={config.model}")
        client = build_autoresearch_generation_client(config)

    case_by_id = {case.id: case for case in cases}
    observations: list[tuple[AutoresearchCase, CaseEvaluation]] = [
        (case_by_id[evaluation.case_id], evaluation)
        for evaluation in evaluations
        if evaluation.failure_category and evaluation.case_id in case_by_id
    ]
    for round_index in range(1, config.rounds + 1):
        _emit_progress(config, f"round {round_index}/{config.rounds} started")
        for case_index, case in enumerate(cases, start=1):
            if (round_index, case.id) in completed_pairs:
                continue
            _emit_progress(
                config,
                f"round {round_index}/{config.rounds} case {case_index}/{len(cases)} {case.id} kind={case.kind} started",
            )
            evaluation = evaluate_autoresearch_case(
                case,
                retriever,
                config=config,
                generation_client=client,
                round_index=round_index,
            )
            evaluations.append(evaluation)
            _append_jsonl(rounds_path, asdict(evaluation))
            if evaluation.failure_category:
                observations.append((case, evaluation))
            completed_pairs.add((round_index, case.id))
            _emit_progress(
                config,
                (
                    f"round {round_index}/{config.rounds} case {case_index}/{len(cases)} {case.id} "
                    f"retrieval={evaluation.retrieval.get('passed')} answer={evaluation.answer.get('status')} "
                    f"failure={evaluation.failure_category or 'none'}"
                ),
            )
        _emit_progress(config, f"round {round_index}/{config.rounds} completed")

    failures = confirmed_failures(observations, confirmations=config.confirmations)
    _write_jsonl(run_dir / "failures.jsonl", [asdict(failure) for failure in failures])
    decisions = build_coordinator_decision_records(run_dir=run_dir, failures=failures)
    _write_jsonl(run_dir / "coordinator_decisions.jsonl", [asdict(decision) for decision in decisions])
    (run_dir / "coordinator_decisions.md").write_text(
        build_coordinator_decisions_markdown(decisions),
        encoding="utf-8",
    )
    summary = build_autoresearch_summary(
        config=config,
        dictionary=dictionary,
        cases=cases,
        evaluations=evaluations,
        failures=failures,
        elapsed_s=time.perf_counter() - started,
    )
    (run_dir / "summary.md").write_text(summary, encoding="utf-8")
    session = build_codex_session(config=config, run_dir=run_dir, failures=failures, summary=summary)
    (run_dir / "codex_session.md").write_text(session, encoding="utf-8")
    (run_dir / "codex_tasks.md").write_text(build_codex_tasks(failures), encoding="utf-8")
    return {
        "output_dir": str(run_dir),
        "case_count": len(cases),
        "evaluation_count": len(evaluations),
        "failure_count": len(failures),
        "candidate_failure_count": len(observations),
        "decision_count": len(decisions),
        "elapsed_s": time.perf_counter() - started,
    }


def validate_autoresearch_config(config: DictionaryAutoresearchConfig) -> None:
    if config.rounds <= 0:
        raise ValueError("--rounds must be positive")
    if config.limit <= 0:
        raise ValueError("--limit must be positive")
    if config.top_k <= 0:
        raise ValueError("--top-k must be positive")
    if config.max_context_chars <= 0:
        raise ValueError("--max-context-chars must be positive")
    if config.max_completion_tokens <= 0:
        raise ValueError("--max-completion-tokens must be positive")
    if config.confirmations <= 0:
        raise ValueError("--confirmations must be positive")
    if config.judge_json_retries < 0:
        raise ValueError("--judge-json-retries must be non-negative")
    if config.resume and not config.run_name:
        raise ValueError("--resume requires --run-name")
    if config.source_classification not in {"semi-private", "private"}:
        raise ValueError("--source-classification must be semi-private or private")
    if config.provider not in {"mimo", "local"}:
        raise ValueError("--provider must be mimo or local")
    if config.source_classification == "private":
        if config.provider != "local":
            raise ValueError("Private dictionary autoresearch requires --provider local")
        if config.model not in set(config.trusted_models):
            raise ValueError("Private dictionary autoresearch requires --trusted-model for the selected local model")


def generate_autoresearch_cases(
    dictionary: DictionaryLoadResult,
    *,
    limit: int,
    feedback_run_dirs: tuple[Path, ...] = (),
) -> list[AutoresearchCase]:
    cases: list[AutoresearchCase] = []
    for case in semantic_seed_cases(dictionary):
        cases.append(case)
        if len(cases) >= limit:
            return cases
    for case in adaptive_feedback_cases(dictionary, feedback_run_dirs=feedback_run_dirs):
        if any(existing.id == case.id for existing in cases):
            continue
        cases.append(case)
        if len(cases) >= limit:
            return cases
    for case in red_generate_cases(dictionary):
        if any(existing.id == case.id for existing in cases):
            continue
        cases.append(case)
        if len(cases) >= limit:
            break
    return cases


def semantic_seed_cases(dictionary: DictionaryLoadResult) -> list[AutoresearchCase]:
    seed_specs = [
        ("hexogen", "HEXOGEN", "semantic_corner_cases.md", "accent-folded-headword"),
        ("hêxôgen", "HEXOGEN", "semantic_corner_cases.md", "accent-folded-headword"),
        ("hê-xô-gen", "HEXOGEN", "semantic_corner_cases.md", "accent-folded-headword"),
        ("he-xo-gen", "HEXOGEN", "semantic_corner_cases.md", "accent-folded-headword"),
        ("pb", "PHÁO BINH", "semantic_corner_cases.md", "abbreviation-alias"),
        ("pbbc", "PHÁO BINH BIÊN CHẾ", "semantic_corner_cases.md", "abbreviation-alias"),
        ("pháo đài Láng", "PHÁO ĐÀI LÁNG", "semantic_corner_cases.md", "place-phrase"),
        ("Pháo đài Xuân Canh", "PHÁO ĐÀI XUÂN CANH", "semantic_corner_cases.md", "place-phrase"),
        ("nhật", "NHẬT", "semantic_corner_cases.md", "diacritic-strict"),
        ("nhất", "NHẤT", "semantic_corner_cases.md", "diacritic-strict"),
    ]
    by_title = _documents_by_title(dictionary)
    cases: list[AutoresearchCase] = []
    for query, title, source, kind in seed_specs:
        doc = by_title.get(_title_key(title))
        if doc is None:
            continue
        metadata: dict[str, Any] | None = None
        if "abbreviation" in kind:
            metadata = {"required_rank": 1, "abbreviation_key": _folded_alnum_key(query)}
        cases.append(_case_from_doc(query, doc, kind=kind, source=source, metadata=metadata))
    return cases


def red_generate_cases(dictionary: DictionaryLoadResult) -> list[AutoresearchCase]:
    cases: list[AutoresearchCase] = []
    corpus_texts = [_normalized_phrase_text(doc.text) for doc in dictionary.documents]
    abbreviation_counts = _abbreviation_evidence_key_counts(dictionary)
    cases.extend(_abbreviation_adversarial_cases(dictionary, corpus_texts=corpus_texts))
    for doc in dictionary.documents:
        title = str(doc.title or doc.metadata.get("headword") or "").strip()
        if not title:
            continue
        cases.append(_case_from_doc(title, doc, kind="exact-headword", source="red-local-index"))
        for abbreviation in _evidenced_abbreviations(doc, title):
            abbreviation_key = _folded_alnum_key(abbreviation)
            metadata: dict[str, Any] = {"abbreviation_key": abbreviation_key}
            if abbreviation_counts.get(abbreviation_key, 0) <= 1:
                metadata["required_rank"] = 1
            else:
                metadata["ambiguous_abbreviation"] = True
            cases.append(
                _case_from_doc(
                    abbreviation,
                    doc,
                    kind="generated-abbreviation",
                    source="red-evidenced-alias",
                    metadata=metadata,
                )
            )
        if re.search(r"\d", title):
            cases.append(_case_from_doc(title, doc, kind="numeric-headword", source="red-local-index"))
        phrase = _definition_phrase(doc.text, title, corpus_texts=corpus_texts)
        if phrase:
            cases.append(_case_from_doc(phrase, doc, kind="definition-phrase", source="red-local-index"))
    return _dedupe_cases(cases)


def _abbreviation_adversarial_cases(
    dictionary: DictionaryLoadResult,
    *,
    corpus_texts: list[str],
) -> list[AutoresearchCase]:
    evidence_by_key: dict[str, list[tuple[str, Any, str]]] = defaultdict(list)
    for doc in dictionary.documents:
        title = str(doc.title or doc.metadata.get("headword") or "").strip()
        if not title:
            continue
        for abbreviation in _evidenced_abbreviations(doc, title):
            key = _folded_alnum_key(abbreviation)
            if key:
                evidence_by_key[key].append((abbreviation, doc, title))

    cases: list[AutoresearchCase] = []
    for key, items in sorted(evidence_by_key.items(), key=lambda item: (-len(item[1]), item[0])):
        docs_by_id: dict[str, tuple[str, Any, str]] = {}
        for abbreviation, doc, title in items:
            docs_by_id.setdefault(str(doc.doc_id), (abbreviation, doc, title))
        unique_items = list(docs_by_id.values())
        if len(unique_items) < 2:
            continue
        competitor_titles = [str(doc.title or title) for _abbreviation, doc, title in unique_items[:8]]
        for abbreviation, doc, title in unique_items[:6]:
            context = _abbreviation_context_query(doc.text, abbreviation, title, corpus_texts=corpus_texts)
            if context:
                cases.append(
                    _case_from_doc(
                        context,
                        doc,
                        kind="adversarial-abbreviation-context",
                        source="red-abbreviation-collision",
                        metadata={
                            "required_rank": 1,
                            "abbreviation_key": key,
                            "collision_titles": competitor_titles,
                        },
                    )
                )
            primary_items = [
                item
                for item in unique_items
                if _is_primary_abbreviation_entry(item[0], item[1], item[2])
            ]
            if len(primary_items) == 1 and primary_items[0][1].doc_id == doc.doc_id:
                cases.append(
                    _case_from_doc(
                        abbreviation,
                        doc,
                        kind="adversarial-abbreviation-primary",
                        source="red-abbreviation-collision",
                        metadata={
                            "required_rank": 1,
                            "abbreviation_key": key,
                            "collision_titles": competitor_titles,
                        },
                    )
                )
    return _dedupe_cases(cases)


def _abbreviation_evidence_key_counts(dictionary: DictionaryLoadResult) -> dict[str, int]:
    doc_ids_by_key: dict[str, set[str]] = defaultdict(set)
    for doc in dictionary.documents:
        title = str(doc.title or doc.metadata.get("headword") or "").strip()
        if not title:
            continue
        for abbreviation in _evidenced_abbreviations(doc, title):
            key = _folded_alnum_key(abbreviation)
            if key:
                doc_ids_by_key[key].add(str(doc.doc_id))
    return {key: len(doc_ids) for key, doc_ids in doc_ids_by_key.items()}


def adaptive_feedback_cases(
    dictionary: DictionaryLoadResult,
    *,
    feedback_run_dirs: tuple[Path, ...],
) -> list[AutoresearchCase]:
    if not feedback_run_dirs:
        return []
    docs_by_id = {doc.doc_id: doc for doc in dictionary.documents}
    docs_by_title = _documents_by_title(dictionary)
    corpus_texts = [_normalized_phrase_text(doc.text) for doc in dictionary.documents]
    cases: list[AutoresearchCase] = []
    for run_dir in feedback_run_dirs:
        if not run_dir.exists():
            continue
        decision_status = _feedback_decision_status(run_dir)
        for failure in _feedback_jsonl(run_dir / "failures.jsonl"):
            if decision_status.get(str(failure.get("id"))) == "rejected":
                continue
            doc = _feedback_expected_doc(failure, docs_by_id=docs_by_id, docs_by_title=docs_by_title)
            if doc is None:
                continue
            for query in _dedupe_strings([str(item) for item in failure.get("queries", []) if item]):
                cases.append(_case_from_doc(query, doc, kind=f"feedback-confirmed-{failure.get('category') or 'failure'}", source=str(run_dir)))
            cases.extend(_feedback_doc_variants(doc, source=str(run_dir), corpus_texts=corpus_texts))
        for row in _feedback_jsonl(run_dir / "rounds.jsonl"):
            doc = _feedback_expected_doc(row, docs_by_id=docs_by_id, docs_by_title=docs_by_title)
            if doc is None:
                continue
            failure_category = str(row.get("failure_category") or "")
            query = str(row.get("query") or "").strip()
            if failure_category and query:
                for query_variant in _feedback_query_variants(query):
                    cases.append(
                        _case_from_doc(
                            query_variant,
                            doc,
                            kind=f"feedback-candidate-{failure_category}",
                            source=str(run_dir),
                            metadata=_feedback_case_metadata(row, query_variant),
                        )
                    )
                if failure_category in {"answer", "citation"}:
                    cases.extend(_feedback_doc_variants(doc, source=str(run_dir), corpus_texts=corpus_texts))
            rank = _feedback_expected_rank(row)
            if rank is not None and 1 < rank <= 5 and query:
                for query_variant in _feedback_query_variants(query):
                    cases.append(
                        _case_from_doc(
                            query_variant,
                            doc,
                            kind="feedback-near-miss",
                            source=str(run_dir),
                            metadata=_feedback_case_metadata(row, query_variant),
                        )
                    )
    return _dedupe_cases(cases)


def evaluate_autoresearch_case(
    case: AutoresearchCase,
    retriever: DictionaryGraphRetriever,
    *,
    config: DictionaryAutoresearchConfig,
    generation_client: AutoresearchGenerationClient | None,
    round_index: int,
) -> CaseEvaluation:
    retrieval = retriever.search(Query(query_id=case.id, text=case.query), top_k=config.top_k)
    retrieval_judgement = judge_retrieval(
        case,
        retrieval.hits,
        strict_acronym_rank=config.strict_acronym_rank,
    )
    answer_payload: dict[str, Any] = {"status": "skipped", "reason": "retrieval failed or dry run"}
    failure_category: str | None = None
    failure_reason: str | None = None
    if not retrieval_judgement.passed:
        failure_category = retrieval_judgement.category
        failure_reason = retrieval_judgement.reason
    elif generation_client is None or config.dry_run_model:
        answer_payload = {"status": "skipped", "reason": "dry-run-model"}
    else:
        answer = generate_dictionary_answer(case, retrieval.hits, config=config, generation_client=generation_client)
        answer_judgement = judge_answer_truth(
            case,
            answer,
            retrieval.hits,
            config=config,
            generation_client=generation_client,
        )
        answer_payload = {"status": answer_judgement.status, "reason": answer_judgement.reason, "raw": answer_judgement.raw}
        answer_payload["json_retry_count"] = answer_judgement.json_retry_count
        if answer_judgement.passed is False:
            failure_category = answer_judgement.category
            failure_reason = answer_judgement.reason
    return CaseEvaluation(
        round_index=round_index,
        case_id=case.id,
        query=case.query,
        expected_doc_id=case.expected_doc_id,
        expected_title=case.expected_title,
        retrieval=asdict(retrieval_judgement),
        answer=answer_payload,
        failure_category=failure_category,
        failure_reason=failure_reason,
    )


def judge_retrieval(
    case: AutoresearchCase,
    hits: list[RetrievalHit],
    *,
    strict_acronym_rank: bool = True,
) -> RetrievalJudgement:
    top_ids = [hit.doc_id for hit in hits]
    top_titles = [hit.title or "" for hit in hits]
    if not case.expected_doc_id and not case.expected_title:
        return RetrievalJudgement(False, "ambiguous", "case has no expected document", None, top_ids, top_titles)
    expected_rank: int | None = None
    for index, hit in enumerate(hits, start=1):
        if case.expected_doc_id and hit.doc_id == case.expected_doc_id:
            expected_rank = index
            break
        if case.expected_title and _title_key(hit.title or "") == _title_key(case.expected_title):
            expected_rank = index
            break
    if expected_rank is None:
        return RetrievalJudgement(False, "retrieval", "expected dictionary entry missing from top-k", None, top_ids, top_titles)
    required_rank = _required_retrieval_rank(case, strict_acronym_rank=strict_acronym_rank)
    if required_rank is not None and expected_rank > required_rank:
        return RetrievalJudgement(
            False,
            "retrieval",
            f"expected entry ranked #{expected_rank}, required top #{required_rank}",
            expected_rank,
            top_ids,
            top_titles,
        )
    if expected_rank > max(3, min(5, len(hits))):
        return RetrievalJudgement(False, "retrieval", f"expected entry ranked too low: #{expected_rank}", expected_rank, top_ids, top_titles)
    return RetrievalJudgement(True, "", "expected entry found", expected_rank, top_ids, top_titles)


def generate_dictionary_answer(
    case: AutoresearchCase,
    hits: list[RetrievalHit],
    *,
    config: DictionaryAutoresearchConfig,
    generation_client: AutoresearchGenerationClient,
) -> str:
    messages = build_dictionary_rag_messages(
        [{"role": "user", "content": case.query}],
        hits,
        query=case.query,
        max_context_chars=config.max_context_chars,
        history_messages=0,
        language="vi",
    )
    result = generation_client.generate(
        messages,
        model=config.model,
        temperature=0.0,
        max_completion_tokens=config.max_completion_tokens,
    )
    if result.error:
        raise RuntimeError(result.error)
    return format_dictionary_answer(hits, result.answer)


def judge_answer_truth(
    case: AutoresearchCase,
    answer: str,
    hits: list[RetrievalHit],
    *,
    config: DictionaryAutoresearchConfig,
    generation_client: AutoresearchGenerationClient,
) -> AnswerJudgement:
    deterministic = _deterministic_raw_entry_judgement(case, answer)
    if deterministic is not None:
        return deterministic
    source_snippets = "\n\n".join(
        f"[{hit.doc_id}] {hit.title or ''}\n{hit.text[:900]}" for hit in hits[: config.top_k]
    )
    base_messages = [
        {
            "role": "system",
            "content": (
                "You are a strict judge for a Vietnamese military dictionary RAG system. "
                "Return JSON only with keys: verdict, category, reason. Do not wrap it in Markdown. "
                "verdict must be pass, fail, or ambiguous. "
                "category must be answer, citation, ui/render, or ambiguous."
            ),
        },
        {
            "role": "user",
            "content": (
                f"Query: {case.query}\n"
                f"Expected entry id: {case.expected_doc_id}\n"
                f"Expected title: {case.expected_title}\n"
                f"Reference snippet:\n{case.reference_snippet}\n\n"
                f"Model answer:\n{answer}\n\n"
                f"Retrieved sources:\n{source_snippets}\n\n"
                "Judge whether the answer contradicts, omits, or invents facts relative to the reference."
            ),
        },
    ]
    messages = base_messages
    last_raw = ""
    attempts = config.judge_json_retries + 1
    for attempt_index in range(attempts):
        result = generation_client.generate(
            messages,
            model=config.model,
            temperature=0.0,
            max_completion_tokens=512,
        )
        if result.error:
            return AnswerJudgement("ambiguous", None, "ambiguous", result.error, raw=last_raw, json_retry_count=attempt_index)
        last_raw = result.answer
        parsed = _parse_json_object(result.answer)
        if parsed:
            return _answer_judgement_from_parsed(parsed, raw=result.answer, json_retry_count=attempt_index)
        messages = _retry_json_judge_messages(base_messages, previous_answer=result.answer)
    return AnswerJudgement(
        "ambiguous",
        None,
        "ambiguous",
        f"judge did not return valid JSON after {attempts} attempts",
        raw=last_raw,
        json_retry_count=max(0, attempts - 1),
    )


def parse_answer_judgement(text: str) -> AnswerJudgement:
    parsed = _parse_json_object(text)
    if not parsed:
        return AnswerJudgement("ambiguous", None, "ambiguous", "judge did not return valid JSON", raw=text)
    return _answer_judgement_from_parsed(parsed, raw=text, json_retry_count=0)


def _deterministic_raw_entry_judgement(case: AutoresearchCase, answer: str) -> AnswerJudgement | None:
    clean = str(answer or "").strip()
    if not clean:
        return AnswerJudgement("fail", False, "answer", "model answer is empty")
    if "Giải thích:" in clean:
        return None
    primary_doc_id = _formatted_primary_doc_id(clean)
    if not primary_doc_id:
        return None
    expected_doc_id = case.expected_doc_id or ""
    if expected_doc_id and primary_doc_id != expected_doc_id:
        return AnswerJudgement(
            "fail",
            False,
            "answer",
            f"formatted raw entry uses {primary_doc_id}, expected {expected_doc_id}",
        )
    if expected_doc_id and f"[{expected_doc_id}]" in clean:
        return AnswerJudgement("pass", True, "answer", "formatted raw dictionary entry includes expected source id")
    expected_title = str(case.expected_title or "").strip()
    if expected_title and expected_title.casefold() in clean.casefold():
        return AnswerJudgement("pass", True, "answer", "formatted raw dictionary entry includes expected title")
    return None


def _formatted_primary_doc_id(answer: str) -> str | None:
    match = re.search(r"^(?:Mục từ gốc|Original entry)\s+\[([^\]]+)\]:", answer.strip(), flags=re.IGNORECASE)
    return match.group(1).strip() if match else None


def _answer_judgement_from_parsed(parsed: dict[str, Any], *, raw: str, json_retry_count: int) -> AnswerJudgement:
    verdict = str(parsed.get("verdict") or "").strip().lower()
    category = str(parsed.get("category") or "answer").strip().lower()
    reason = str(parsed.get("reason") or "").strip() or "no reason supplied"
    if category not in FAILURE_CATEGORIES:
        category = "answer"
    if verdict == "pass":
        return AnswerJudgement("pass", True, category, reason, raw=raw, json_retry_count=json_retry_count)
    if verdict == "fail":
        return AnswerJudgement("fail", False, category, reason, raw=raw, json_retry_count=json_retry_count)
    return AnswerJudgement("ambiguous", None, "ambiguous", reason, raw=raw, json_retry_count=json_retry_count)


def confirmed_failures(
    observations: list[tuple[AutoresearchCase, CaseEvaluation]],
    *,
    confirmations: int,
) -> list[FailureRecord]:
    groups: dict[tuple[str, str, str], list[tuple[AutoresearchCase, CaseEvaluation]]] = defaultdict(list)
    for case, evaluation in observations:
        category = evaluation.failure_category or "ambiguous"
        expected = case.expected_doc_id or case.expected_title or case.kind
        groups[(category, expected, case.kind)].append((case, evaluation))
    failures: list[FailureRecord] = []
    for (category, expected, kind), items in sorted(groups.items()):
        unique_queries = _dedupe_strings([case.query for case, _evaluation in items])
        if len(items) < confirmations and len(unique_queries) < confirmations:
            continue
        first_case, first_eval = items[0]
        failure_id = "failure-" + _stable_hash("|".join([category, expected, kind]))[:12]
        failures.append(
            FailureRecord(
                id=failure_id,
                category=category,
                expected_doc_id=first_case.expected_doc_id,
                expected_title=first_case.expected_title,
                kind=kind,
                count=len(items),
                case_ids=_dedupe_strings([case.id for case, _evaluation in items]),
                queries=unique_queries,
                reason=first_eval.failure_reason or "confirmed autoresearch failure",
                suggested_files=suggested_files_for_failure(category),
                acceptance_tests=acceptance_tests_for_failure(category, unique_queries),
            )
        )
    return failures


def build_autoresearch_summary(
    *,
    config: DictionaryAutoresearchConfig,
    dictionary: DictionaryLoadResult,
    cases: list[AutoresearchCase],
    evaluations: list[CaseEvaluation],
    failures: list[FailureRecord],
    elapsed_s: float,
) -> str:
    failed = [evaluation for evaluation in evaluations if evaluation.failure_category]
    category_counts = Counter(evaluation.failure_category or "pass" for evaluation in evaluations)
    lines = [
        "# Dictionary Autoresearch Summary",
        "",
        f"- Created: `{datetime.now(timezone.utc).isoformat()}`",
        f"- Source classification: `{config.source_classification}`",
        f"- Provider: `{config.provider}`",
        f"- Model calls: `{'disabled' if config.dry_run_model else 'enabled'}`",
        f"- Entries: `{len(dictionary.documents)}`",
        f"- Cases: `{len(cases)}`",
        f"- Evaluations: `{len(evaluations)}`",
        f"- Candidate failures: `{len(failed)}`",
        f"- Confirmed failures: `{len(failures)}`",
        f"- Elapsed seconds: `{elapsed_s:.2f}`",
        "",
        "## Counts",
        "",
        "| Category | Count |",
        "| --- | ---: |",
    ]
    for category, count in sorted(category_counts.items()):
        lines.append(f"| {category} | {count} |")
    lines.extend(["", "## Confirmed Failure Clusters", ""])
    if not failures:
        lines.append("No confirmed failure clusters yet.")
    else:
        for failure in failures:
            lines.append(f"- `{failure.id}` `{failure.category}` `{failure.expected_title or failure.expected_doc_id}`: {failure.reason}")
    lines.append("")
    return "\n".join(lines)


def build_codex_session(
    *,
    config: DictionaryAutoresearchConfig,
    run_dir: Path,
    failures: list[FailureRecord],
    summary: str,
) -> str:
    lines = [
        "# Codex Coordinator Session",
        "",
        "Use this file as persistent context for one coordinator Codex session.",
        "",
        "## Run",
        "",
        f"- Run dir: `{run_dir}`",
        f"- Branch intent: `feat/dictionary-autoresearch-selfplay`",
        f"- Source classification: `{config.source_classification}`",
        f"- Provider policy: `{config.provider}`; private data requires trusted local model.",
        "",
        "## Summary",
        "",
        summary,
        "",
        "## Worker Protocol",
        "",
        "- Give each worker one bounded task from `codex_tasks.md`.",
        "- Review and update `coordinator_decisions.jsonl` before deciding to fix or reject a failure.",
        "- Tell workers not to revert unrelated edits.",
        "- Coordinator reviews diff, runs tests, then reruns autoresearch.",
        "- Do not paste secrets or API keys into worker prompts.",
    ]
    if failures:
        lines.extend(["", "## Active Failures", ""])
        for failure in failures:
            lines.append(f"- `{failure.id}`: {failure.category}; queries={failure.queries}; files={failure.suggested_files}")
    return "\n".join(lines) + "\n"


def build_codex_tasks(failures: list[FailureRecord]) -> str:
    if not failures:
        return "# Codex Worker Tasks\n\nNo confirmed tasks yet. Rerun with more rounds or lower confirmations if needed.\n"
    lines = ["# Codex Worker Tasks", ""]
    for failure in failures:
        lines.extend(
            [
                f"## {failure.id}",
                "",
                f"- Category: `{failure.category}`",
                f"- Expected: `{failure.expected_title or failure.expected_doc_id}`",
                f"- Queries: {', '.join(f'`{query}`' for query in failure.queries)}",
                f"- Suggested files: {', '.join(f'`{path}`' for path in failure.suggested_files)}",
                f"- Reason: {failure.reason}",
                "- Acceptance:",
            ]
        )
        lines.extend(f"  - {test}" for test in failure.acceptance_tests)
        lines.append("")
    return "\n".join(lines)


def build_coordinator_decision_records(*, run_dir: Path, failures: list[FailureRecord]) -> list[CoordinatorDecision]:
    evidence_files = [
        str(run_dir / "failures.jsonl"),
        str(run_dir / "rounds.jsonl"),
        str(run_dir / "cases.jsonl"),
        str(run_dir / "summary.md"),
    ]
    return [
        CoordinatorDecision(
            id="decision-" + failure.id.removeprefix("failure-"),
            failure_id=failure.id,
            status="pending",
            status_options=["pending", "accepted", "rejected", "needs_more_evidence", "converted_to_test"],
            category=failure.category,
            expected_title=failure.expected_title,
            queries=failure.queries,
            evidence_files=evidence_files,
            verification_checklist=verification_checklist_for_failure(failure),
            suggested_files=failure.suggested_files,
            acceptance_tests=failure.acceptance_tests,
        )
        for failure in failures
    ]


def build_coordinator_decisions_markdown(decisions: list[CoordinatorDecision]) -> str:
    lines = [
        "# Coordinator Decisions",
        "",
        "Codex coordinator owns this file. Red/Blue agents only produce evidence; they do not decide fixes.",
        "",
        "Allowed statuses: `pending`, `accepted`, `rejected`, `needs_more_evidence`, `converted_to_test`.",
        "",
    ]
    if not decisions:
        lines.append("No confirmed failures need coordinator decisions in this run.")
        return "\n".join(lines) + "\n"
    for decision in decisions:
        lines.extend(
            [
                f"## {decision.id}",
                "",
                f"- Failure: `{decision.failure_id}`",
                f"- Status: `{decision.status}`",
                f"- Category: `{decision.category}`",
                f"- Expected: `{decision.expected_title or 'N/A'}`",
                f"- Queries: {', '.join(f'`{query}`' for query in decision.queries)}",
                "- Verification checklist:",
            ]
        )
        lines.extend(f"  - {item}" for item in decision.verification_checklist)
        lines.extend(
            [
                "- Decision reason: ",
                "- Next action: verify before assigning or editing",
                "",
            ]
        )
    return "\n".join(lines)


def verification_checklist_for_failure(failure: FailureRecord) -> list[str]:
    return [
        "Reproduce the failure from `rounds.jsonl` against the current branch.",
        "Confirm the expected entry/reference snippet is unambiguous.",
        "Check whether existing seed semantic corner cases still pass.",
        f"Decide whether `{failure.category}` is a code bug, test-only case, or ambiguous data issue.",
        "Only create or assign a worker task after this decision is no longer pending.",
    ]


def build_autoresearch_generation_client(config: DictionaryAutoresearchConfig) -> RoundRobinGroqClient:
    if config.provider == "mimo":
        key = load_env_api_key(config.mimo_env_file, config.mimo_api_key_var, alias="mimo")
        base_url = config.mimo_base_url
    else:
        key = _local_api_key(config)
        base_url = config.local_base_url
    return RoundRobinGroqClient(
        keys=[key],
        model=config.model,
        max_retries=1,
        key_tokens_per_minute=0,
        key_requests_per_minute=0,
        rate_limit_scope="per-key",
        client_factory=lambda api_key, timeout: OpenAICompatibleClient(
            api_key=api_key.value,
            base_url=base_url,
            timeout_s=timeout,
            token_parameter="max_tokens",
        ),
        provider_name="MiMo" if config.provider == "mimo" else "Local",
        completion_token_parameter="max_tokens",
    )


def suggested_files_for_failure(category: str) -> list[str]:
    if category == "retrieval":
        return ["src/rag_bench/retrievers.py", "tests/test_retrievers.py", "semantic_corner_cases.md"]
    if category in {"answer", "citation"}:
        return ["src/rag_bench/chat_service.py", "tests/test_chat_service.py", "semantic_corner_cases.md"]
    if category == "ui/render":
        return ["ui/chat.html", "tests/test_api.py", "semantic_corner_cases.md"]
    return ["semantic_corner_cases.md"]


def acceptance_tests_for_failure(category: str, queries: list[str]) -> list[str]:
    query_text = ", ".join(queries[:3])
    if category == "retrieval":
        return [
            f"Add a retriever regression covering: {query_text}",
            "Run `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 uv run --locked pytest -q tests/test_retrievers.py`.",
        ]
    if category in {"answer", "citation"}:
        return [
            f"Add a chat-service regression covering: {query_text}",
            "Run `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 uv run --locked pytest -q tests/test_chat_service.py`.",
        ]
    return ["Run the focused test added for this failure cluster."]


def _run_dir(config: DictionaryAutoresearchConfig) -> Path:
    run_name = config.run_name or datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return config.output_root / run_name


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as file_obj:
        for row in rows:
            file_obj.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def _append_jsonl(path: Path, row: dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as file_obj:
        file_obj.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
        file_obj.flush()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as file_obj:
        for line in file_obj:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def _case_from_jsonl_row(row: dict[str, Any]) -> AutoresearchCase:
    metadata = row.get("metadata")
    return AutoresearchCase(
        id=str(row.get("id") or ""),
        query=str(row.get("query") or ""),
        expected_doc_id=_optional_string(row.get("expected_doc_id")),
        expected_title=_optional_string(row.get("expected_title")),
        reference_snippet=str(row.get("reference_snippet") or ""),
        kind=str(row.get("kind") or "unknown"),
        source=str(row.get("source") or "resume"),
        metadata=metadata if isinstance(metadata, dict) else {},
    )


def _evaluation_from_jsonl_row(row: dict[str, Any]) -> CaseEvaluation:
    retrieval = row.get("retrieval")
    answer = row.get("answer")
    return CaseEvaluation(
        round_index=int(row.get("round_index") or 0),
        case_id=str(row.get("case_id") or ""),
        query=str(row.get("query") or ""),
        expected_doc_id=_optional_string(row.get("expected_doc_id")),
        expected_title=_optional_string(row.get("expected_title")),
        retrieval=retrieval if isinstance(retrieval, dict) else {},
        answer=answer if isinstance(answer, dict) else {},
        failure_category=_optional_string(row.get("failure_category")),
        failure_reason=_optional_string(row.get("failure_reason")),
    )


def _optional_string(value: Any) -> str | None:
    return value if isinstance(value, str) else None


def _case_from_doc(
    query: str,
    doc: Any,
    *,
    kind: str,
    source: str,
    metadata: dict[str, Any] | None = None,
) -> AutoresearchCase:
    title = str(doc.title or doc.metadata.get("headword") or doc.doc_id)
    reference = _reference_snippet(doc.text, title)
    case_id = "case-" + _stable_hash("|".join([query, doc.doc_id, kind]))[:14]
    case_metadata = {
        "source_file": doc.metadata.get("source_file"),
        "source_set": doc.metadata.get("source_set"),
    }
    if metadata:
        case_metadata.update(metadata)
    return AutoresearchCase(
        id=case_id,
        query=query,
        expected_doc_id=doc.doc_id,
        expected_title=title,
        reference_snippet=reference,
        kind=kind,
        source=source,
        metadata=case_metadata,
    )


def _documents_by_title(dictionary: DictionaryLoadResult) -> dict[str, Any]:
   result: dict[str, Any] = {}
   for doc in dictionary.documents:
       title = str(doc.title or doc.metadata.get("headword") or "")
       result.setdefault(_title_key(title), doc)
   return result


def _feedback_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as file_obj:
        for line in file_obj:
            if not line.strip():
                continue
            try:
                parsed = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(parsed, dict):
                rows.append(parsed)
    return rows


def _feedback_decision_status(run_dir: Path) -> dict[str, str]:
    result: dict[str, str] = {}
    for row in _feedback_jsonl(run_dir / "coordinator_decisions.jsonl"):
        failure_id = str(row.get("failure_id") or "")
        status = str(row.get("status") or "")
        if failure_id and status:
            result[failure_id] = status
    return result


def _feedback_expected_doc(
    row: dict[str, Any],
    *,
    docs_by_id: dict[str, Any],
    docs_by_title: dict[str, Any],
) -> Any | None:
    doc_id = row.get("expected_doc_id")
    if isinstance(doc_id, str) and doc_id in docs_by_id:
        return docs_by_id[doc_id]
    title = row.get("expected_title")
    if isinstance(title, str):
        return docs_by_title.get(_title_key(title))
    return None


def _feedback_expected_rank(row: dict[str, Any]) -> int | None:
    retrieval = row.get("retrieval")
    if not isinstance(retrieval, dict):
        return None
    rank = retrieval.get("expected_rank")
    return rank if isinstance(rank, int) else None


def _feedback_doc_variants(doc: Any, *, source: str, corpus_texts: list[str]) -> list[AutoresearchCase]:
    title = str(doc.title or doc.metadata.get("headword") or doc.doc_id)
    queries: list[tuple[str, str]] = [(title, "feedback-exact-title")]
    for abbreviation in _evidenced_abbreviations(doc, title)[:3]:
        queries.append((abbreviation, "feedback-evidenced-alias"))
    for phrase in _distinctive_phrases(doc.text, title, corpus_texts=corpus_texts, max_phrases=3)[:3]:
        queries.append((phrase, "feedback-distinctive-phrase"))
    return [_case_from_doc(query, doc, kind=kind, source=source) for query, kind in queries if query]


def _feedback_query_variants(query: str) -> list[str]:
    clean = " ".join(str(query or "").split())
    if not clean:
        return []
    variants = [clean]
    if _short_abbreviation_query(clean, kind="feedback"):
        variants.extend(
            [
                clean.upper(),
                f"{clean} là gì",
                f"tra {clean}",
                f"{clean} trong từ điển pháo binh",
            ]
        )
    elif len(clean.split()) <= 4:
        variants.extend([f"{clean} là gì", f"tra {clean}"])
    return _dedupe_strings(variants)


def _feedback_case_metadata(row: dict[str, Any], query: str) -> dict[str, Any]:
    metadata: dict[str, Any] = {}
    if _short_abbreviation_query(query, kind="feedback"):
        metadata["required_rank"] = 1
        metadata["abbreviation_key"] = _folded_alnum_key(query.split()[0])
    rank = _feedback_expected_rank(row)
    if rank is not None:
        metadata["previous_expected_rank"] = rank
    return metadata


def _title_key(value: str) -> str:
    return " ".join(str(value or "").casefold().split())


def _evidenced_abbreviations(doc: Any, title: str) -> list[str]:
    title_tokens = {_abbreviation_key(token) for token in _abbreviation_tokens(title)}
    title_initial_key = _title_initial_key(title)
    candidates: list[str] = []
    for alias in _metadata_strings(doc.metadata.get("aliases")):
        if _looks_like_abbreviation(alias, title_tokens=title_tokens):
            candidates.append(alias.strip())
    for token in _abbreviation_tokens(doc.text):
        if _looks_like_text_abbreviation_for_title(
            token,
            title_tokens=title_tokens,
            title_initial_key=title_initial_key,
        ):
            candidates.append(token.strip())
    return _dedupe_strings(candidates)


def _abbreviation_context_query(
    text: str,
    abbreviation: str,
    title: str,
    *,
    corpus_texts: list[str],
) -> str:
    token = str(abbreviation or "").strip()
    if not token:
        return ""
    pattern = re.compile(rf"(?<![\wÀ-ỹĐđ]){re.escape(token)}(?![\wÀ-ỹĐđ])", flags=re.IGNORECASE | re.UNICODE)
    match = pattern.search(str(text or ""))
    if match:
        words_before = re.findall(r"[\wÀ-ỹĐđ]+", text[max(0, match.start() - 80) : match.start()], flags=re.UNICODE)
        words_after = re.findall(r"[\wÀ-ỹĐđ]+", text[match.end() : match.end() + 120], flags=re.UNICODE)
        context_words = [word for word in [*words_before[-2:], *words_after[:5]] if _phrase_has_distinctive_token(word)]
        if not context_words:
            context_words = [word for word in words_after[:5] if len(word) >= 4]
        if context_words:
            return " ".join([token, *context_words[:4]])
    for phrase in _distinctive_phrases(text, title, corpus_texts=corpus_texts, max_phrases=2):
        phrase_words = phrase.split()
        if phrase_words:
            return " ".join([token, *phrase_words[:4]])
    return ""


def _is_primary_abbreviation_entry(abbreviation: str, doc: Any, title: str) -> bool:
    key = _folded_alnum_key(abbreviation)
    if not key:
        return False
    title_initial_key = _title_initial_key(title)
    aliases = {_folded_alnum_key(alias) for alias in _metadata_strings(doc.metadata.get("aliases"))}
    return key == title_initial_key and key in aliases


def _metadata_strings(value: Any) -> list[str]:
    if isinstance(value, str):
        return [value]
    if isinstance(value, list):
        return [item for item in value if isinstance(item, str)]
    return []


def _abbreviation_tokens(text: str) -> list[str]:
    return re.findall(r"(?<![\wÀ-ỹĐđ])[\wÀ-ỹĐđ]+(?:[-./][\wÀ-ỹĐđ]+)*(?![\wÀ-ỹĐđ])", str(text or ""), flags=re.UNICODE)


def _looks_like_abbreviation(value: str, *, title_tokens: set[str]) -> bool:
    token = value.strip(".,;:()[]{}\"'“”‘’")
    if " " in token:
        return False
    key = _abbreviation_key(token)
    if not key or key in title_tokens:
        return False
    alnum_chars = [char for char in token if char.isalnum()]
    if not (2 <= len(alnum_chars) <= 12):
        return False
    if all(char.isdigit() for char in alnum_chars):
        return False
    upper_count = sum(1 for char in token if char.isupper())
    digit_count = sum(1 for char in token if char.isdigit())
    return upper_count + digit_count >= 2


def _looks_like_text_abbreviation_for_title(
    value: str,
    *,
    title_tokens: set[str],
    title_initial_key: str,
) -> bool:
    if not title_initial_key:
        return False
    if not _looks_like_abbreviation(value, title_tokens=title_tokens):
        return False
    key = _folded_alnum_key(value)
    return key == title_initial_key or key.startswith(title_initial_key)


def _required_retrieval_rank(case: AutoresearchCase, *, strict_acronym_rank: bool) -> int | None:
    explicit = case.metadata.get("required_rank")
    if isinstance(explicit, int) and explicit > 0:
        return explicit
    if isinstance(explicit, str) and explicit.isdigit() and int(explicit) > 0:
        return int(explicit)
    if strict_acronym_rank and _short_abbreviation_query(case.query, kind=case.kind, metadata=case.metadata):
        return 1
    return None


def _short_abbreviation_query(
    query: str,
    *,
    kind: str = "",
    metadata: dict[str, Any] | None = None,
) -> bool:
    metadata = metadata or {}
    if metadata.get("ambiguous_abbreviation"):
        return False
    if metadata.get("abbreviation_key"):
        return True
    if "abbreviation" in str(kind or ""):
        return True
    token = str(query or "").strip().split(maxsplit=1)[0].strip(".,;:()[]{}\"'“”‘’")
    key = _folded_alnum_key(token)
    if not (2 <= len(key) <= 8):
        return False
    if not any(char.isalpha() for char in key):
        return False
    if any(char.isupper() or char.isdigit() for char in token):
        return True
    return len(token) <= 3 and token.isascii()


def _abbreviation_key(value: str) -> str:
    return "".join(char.casefold() for char in value if char.isalnum())


def _title_initial_key(title: str) -> str:
    tokens = re.findall(r"[\wÀ-ỹĐđ]+", title, flags=re.UNICODE)
    letters = [token[0] for token in tokens if token and token[0].isalpha()]
    if len(letters) < 2:
        return ""
    return _folded_alnum_key("".join(letters))


def _folded_alnum_key(value: str) -> str:
    normalized = unicodedata.normalize("NFD", value.replace("Đ", "D").replace("đ", "d"))
    return "".join(char.casefold() for char in normalized if char.isalnum() and unicodedata.category(char) != "Mn")


def _definition_phrase(text: str, title: str, *, corpus_texts: list[str] | None = None) -> str:
    phrases = _distinctive_phrases(text, title, corpus_texts=corpus_texts, max_phrases=1)
    if phrases:
        return phrases[0]
    return ""


def _distinctive_phrases(
    text: str,
    title: str,
    *,
    corpus_texts: list[str] | None = None,
    max_phrases: int = 3,
) -> list[str]:
    body = str(text or "")
    if title and body.casefold().startswith(title.casefold()):
        body = body[len(title) :]
    body = re.sub(r"^[,.;:\-\s]+", "", body)
    words = re.findall(r"[\wÀ-ỹĐđ]+", body, flags=re.UNICODE)
    if len(words) < 3:
        return []
    phrases: list[str] = []
    if corpus_texts:
        max_start = min(24, len(words))
        for start in range(max_start):
            for window in range(min(14, len(words) - start), 4, -1):
                phrase = " ".join(words[start : start + window])
                if len(phrase) < 16 or not _phrase_has_distinctive_token(phrase):
                    continue
                if _phrase_document_count(phrase, corpus_texts) <= 1:
                    phrases.append(phrase)
                    if len(phrases) >= max_phrases:
                        return phrases
    phrase = " ".join(words[: min(7, len(words))])
    if len(phrase) >= 16 and _phrase_has_distinctive_token(phrase):
        phrases.append(phrase)
    return _dedupe_strings(phrases)[:max_phrases]


def _phrase_has_distinctive_token(phrase: str) -> bool:
    stopwords = {"của", "các", "một", "trong", "được", "dùng", "kiểu", "gồm", "với", "cho", "khi", "từ", "lên", "mà", "có", "là", "và"}
    for token in re.findall(r"[\wÀ-ỹĐđ]+", phrase, flags=re.UNICODE):
        folded = token.casefold()
        if any(char.isdigit() for char in token):
            return True
        if len(folded) >= 6 and folded not in stopwords:
            return True
    return False


def _phrase_document_count(phrase: str, corpus_texts: list[str]) -> int:
    normalized = _normalized_phrase_text(phrase)
    if not normalized:
        return 0
    return sum(1 for text in corpus_texts if normalized in text)


def _normalized_phrase_text(value: str) -> str:
    return " ".join(re.findall(r"[\wÀ-ỹĐđ]+", str(value or ""), flags=re.UNICODE)).casefold()


def _reference_snippet(text: str, title: str) -> str:
    clean = " ".join(str(text or "").split())
    if not clean:
        return title
    return clean[:700]


def _dedupe_cases(cases: list[AutoresearchCase]) -> list[AutoresearchCase]:
    seen: set[tuple[str, str | None]] = set()
    result: list[AutoresearchCase] = []
    for case in cases:
        key = (_title_key(case.query), case.expected_doc_id)
        if key in seen:
            continue
        seen.add(key)
        result.append(case)
    return result


def _dedupe_strings(values: list[str]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for value in values:
        normalized = " ".join(str(value).split())
        key = normalized.casefold()
        if normalized and key not in seen:
            seen.add(key)
            result.append(normalized)
    return result


def _stable_hash(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _parse_json_object(text: str) -> dict[str, Any] | None:
    stripped = text.strip()
    candidates = [stripped]
    start = stripped.find("{")
    end = stripped.rfind("}")
    if start >= 0 and end > start:
        candidates.append(stripped[start : end + 1])
    for candidate in candidates:
        try:
            parsed = json.loads(candidate)
        except json.JSONDecodeError:
            continue
        if isinstance(parsed, dict):
            return parsed
    return None


def _retry_json_judge_messages(
    base_messages: list[dict[str, str]],
    *,
    previous_answer: str,
) -> list[dict[str, str]]:
    return [
        *base_messages,
        {"role": "assistant", "content": previous_answer[:2000]},
        {
            "role": "user",
            "content": (
                "Your previous response was not valid JSON. Return only one JSON object now, with exactly "
                'these keys: "verdict", "category", "reason". No Markdown, no prose.'
            ),
        },
    ]


def _emit_progress(config: DictionaryAutoresearchConfig, message: str) -> None:
    if not config.progress:
        return
    timestamp = datetime.now(timezone.utc).strftime("%H:%M:%S")
    print(f"[autoresearch {timestamp}] {message}", file=sys.stderr, flush=True)


def _local_api_key(config: DictionaryAutoresearchConfig) -> ApiKey:
    if config.local_env_file:
        values = load_env_values(config.local_env_file)
        value = values.get(config.local_api_key_var) or "local"
        return ApiKey(alias="local", value=value)
    return ApiKey(alias="local", value="local")
