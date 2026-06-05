from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import Any

from rag_bench.secrets import ApiKey


@dataclass(frozen=True)
class RagasKeyPreflight:
    keys: list[ApiKey]
    disabled_aliases: list[str]
    errors: list[str]


def filter_available_ragas_keys(keys: list[ApiKey], *, model: str) -> RagasKeyPreflight:
    if not keys:
        return RagasKeyPreflight(keys=[], disabled_aliases=[], errors=["No Groq keys configured."])

    try:
        from groq import Groq
    except ImportError as exc:
        raise RuntimeError("RAGAS Groq key preflight requires the groq package") from exc

    available: list[ApiKey] = []
    disabled_aliases: list[str] = []
    errors: list[str] = []
    for key in keys:
        try:
            client = Groq(api_key=key.value, max_retries=0, timeout=20.0)
            client.chat.completions.create(
                messages=[{"role": "user", "content": "Return ok."}],
                model=model,
                temperature=0.0,
                max_completion_tokens=1,
            )
            available.append(key)
        except Exception as exc:  # noqa: BLE001 - SDK exception classes vary by version.
            if _is_key_or_account_unavailable(exc):
                disabled_aliases.append(key.alias)
                errors.append(f"{key.alias}: {_safe_error(exc)}")
            else:
                available.append(key)
                errors.append(f"{key.alias}: preflight warning: {_safe_error(exc)}")
    return RagasKeyPreflight(keys=available, disabled_aliases=disabled_aliases, errors=errors[:10])


def evaluate_rows_with_ragas(
    rows: list[dict[str, Any]],
    *,
    keys: list[ApiKey],
    model: str,
    limit: int | None,
) -> dict[str, Any]:
    """Run optional RAGAS evaluation one row at a time so keys can rotate by sample."""

    try:
        from datasets import Dataset
        from langchain_groq import ChatGroq
        from ragas import evaluate
    except ImportError as exc:
        raise RuntimeError(
            "RAGAS evaluation requires optional dependencies. Install with: uv sync --extra ragas"
        ) from exc

    metrics = _load_ragas_metrics()
    embeddings = _load_ragas_embeddings()
    selected_rows = rows[:limit] if limit is not None else rows
    if not selected_rows:
        return {"sample_count": 0, "metrics": {}, "key_usage_counts": {}}
    if not keys:
        return {
            "sample_count": len(selected_rows),
            "metrics": {},
            "key_usage_counts": {},
            "error_count": len(selected_rows),
            "errors": ["No available Groq keys for RAGAS evaluation."],
        }

    key_index = 0
    key_counts: Counter[str] = Counter()
    errors: list[str] = []
    disabled_aliases: set[str] = set()
    metric_summary: dict[str, float] = {}

    samples = [_row_to_ragas_sample(row) for row in selected_rows]
    while True:
        available_keys = [key for key in keys if key.alias not in disabled_aliases]
        if not available_keys:
            errors.append("No available Groq keys remain for RAGAS evaluation.")
            break
        key = available_keys[key_index % len(available_keys)]
        key_index += 1
        key_counts[key.alias] += len(samples)
        try:
            llm = ChatGroq(api_key=key.value, model=model, temperature=0.0)
        except TypeError:
            llm = ChatGroq(groq_api_key=key.value, model_name=model, temperature=0.0)
        try:
            result = evaluate(
                Dataset.from_list(samples),
                metrics=metrics,
                llm=llm,
                embeddings=embeddings,
                show_progress=True,
                batch_size=4,
            )
            metric_summary = _result_to_dict(result)
            break
        except Exception as exc:  # noqa: BLE001 - RAGAS exceptions vary by version.
            if _is_key_or_account_unavailable(exc):
                disabled_aliases.add(key.alias)
                errors.append(f"{key.alias}: {_safe_error(exc)}")
                continue
            errors.append(f"{exc.__class__.__name__}: {exc}")
            break

    return {
        "sample_count": len(selected_rows),
        "metrics": metric_summary,
        "key_usage_counts": dict(key_counts),
        "error_count": len(errors),
        "errors": errors[:10],
        "disabled_aliases": sorted(disabled_aliases),
    }


def _load_ragas_metrics() -> list[Any]:
    try:
        from ragas.metrics import answer_relevancy, context_precision, context_recall, faithfulness

        return [faithfulness, answer_relevancy, context_precision, context_recall]
    except ImportError:
        from ragas.metrics import ContextPrecision, ContextRecall, Faithfulness, ResponseRelevancy

        return [Faithfulness(), ResponseRelevancy(), ContextPrecision(), ContextRecall()]


def _load_ragas_embeddings() -> Any:
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError as exc:
        raise RuntimeError(
            "RAGAS evaluation requires local embedding dependencies. "
            "Install with: uv sync --extra vector --extra ragas"
        ) from exc
    return _SentenceTransformerEmbeddings("sentence-transformers/all-MiniLM-L6-v2", SentenceTransformer)


class _SentenceTransformerEmbeddings:
    def __init__(self, model_name: str, sentence_transformer_cls: Any) -> None:
        self._model = sentence_transformer_cls(model_name)

    def embed_query(self, text: str) -> list[float]:
        return self._model.encode(text, normalize_embeddings=True).tolist()

    def embed_text(self, text: str) -> list[float]:
        return self.embed_query(text)

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return self._model.encode(texts, normalize_embeddings=True).tolist()

    async def aembed_query(self, text: str) -> list[float]:
        return self.embed_query(text)

    async def aembed_text(self, text: str) -> list[float]:
        return self.embed_text(text)

    async def aembed_documents(self, texts: list[str]) -> list[list[float]]:
        return self.embed_documents(texts)


def _row_to_ragas_sample(row: dict[str, Any]) -> dict[str, Any]:
    contexts = []
    for hit in row.get("retrieved", []):
        title = hit.get("title", "")
        text = hit.get("text", "")
        context = f"{title}\n{text}".strip()
        if context:
            contexts.append(context)
    contexts = [context for context in contexts if context]
    reference_answers = row.get("reference_answers") or []
    reference = str(row.get("reference") or (reference_answers[0] if reference_answers else ""))
    return {
        "question": row.get("question", ""),
        "answer": row.get("answer", ""),
        "contexts": contexts,
        "ground_truth": reference,
        "user_input": row.get("question", ""),
        "response": row.get("answer", ""),
        "retrieved_contexts": contexts,
        "reference": reference,
    }


def _result_to_dict(result: Any) -> dict[str, float]:
    if hasattr(result, "to_pandas"):
        frame = result.to_pandas()
        if not frame.empty:
            output: dict[str, float] = {}
            for column in frame.columns:
                values = [float(value) for value in frame[column].tolist() if _is_number(value)]
                if values:
                    output[str(column)] = sum(values) / len(values)
            return output
    if isinstance(result, dict):
        return {str(key): float(value) for key, value in result.items() if _is_number(value)}
    return {}


def _average_metric_rows(rows: list[dict[str, float]]) -> dict[str, float]:
    values: dict[str, list[float]] = {}
    for row in rows:
        for key, value in row.items():
            values.setdefault(key, []).append(value)
    return {key: sum(metric_values) / len(metric_values) for key, metric_values in values.items() if metric_values}


def _is_number(value: object) -> bool:
    try:
        float(value)
    except (TypeError, ValueError):
        return False
    return True


def _is_key_or_account_unavailable(exc: Exception) -> bool:
    status_code = getattr(exc, "status_code", None)
    message = _safe_error(exc).lower()
    if "organization_restricted" in message:
        return True
    if status_code in {401, 403}:
        return True
    if status_code == 400 and any(token in message for token in ("invalid api key", "unauthorized", "forbidden")):
        return True
    return False


def _safe_error(exc: Exception) -> str:
    return f"{exc.__class__.__name__}: {exc}"
