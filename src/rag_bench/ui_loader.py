from __future__ import annotations

import json
import os
from functools import lru_cache
from pathlib import Path


def render_chat_page(
    model_id: str,
    available_models: tuple[str, ...] | None = None,
    available_retrievers: tuple[str, ...] | None = None,
    default_retriever: str = "bm25",
) -> str:
    model_json = json.dumps(model_id)
    available_models_json = json.dumps(list(available_models or (model_id,)))
    available_retrievers_json = json.dumps(list(available_retrievers or (default_retriever,)))
    default_retriever_json = json.dumps(default_retriever)
    model_label = _escape_html(model_id)
    return (
        _load_template()
        .replace("__MODEL_JSON__", model_json)
        .replace("__AVAILABLE_MODELS_JSON__", available_models_json)
        .replace("__AVAILABLE_RETRIEVERS_JSON__", available_retrievers_json)
        .replace("__DEFAULT_RETRIEVER_JSON__", default_retriever_json)
        .replace("__MODEL_LABEL__", model_label)
    )


@lru_cache(maxsize=1)
def _load_template() -> str:
    for path in _candidate_template_paths():
        if path.is_file():
            return path.read_text(encoding="utf-8")
    candidates = ", ".join(str(path) for path in _candidate_template_paths())
    raise RuntimeError(f"chat UI template not found; checked: {candidates}")


def _candidate_template_paths() -> tuple[Path, ...]:
    override = os.getenv("RAG_BENCH_UI_TEMPLATE")
    repo_root = Path(__file__).resolve().parents[2]
    paths = [repo_root / "ui" / "chat.html", Path.cwd() / "ui" / "chat.html"]
    if override:
        paths.insert(0, Path(override))
    return tuple(paths)


def _escape_html(value: str) -> str:
    return (
        value.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
        .replace("'", "&#x27;")
    )
