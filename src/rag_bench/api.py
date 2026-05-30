from __future__ import annotations

import json
import math
import os
import time
import uuid
from dataclasses import dataclass
from typing import Any

from fastapi import Depends, FastAPI, Header, HTTPException, Request, status
from fastapi.responses import HTMLResponse, StreamingResponse

from rag_bench.chat_service import DEFAULT_PROXY_MODEL_ID, RagChatService
from rag_bench.ui_loader import render_chat_page


@dataclass(frozen=True)
class ApiSettings:
    api_key: str | None = None


def create_app(service: RagChatService, *, api_key: str | None = None) -> FastAPI:
    settings = ApiSettings(api_key=api_key or os.getenv("RAG_PROXY_API_KEY") or None)
    app = FastAPI(title="True Chat RAG Proxy", version="0.1.0")
    app.state.service = service
    app.state.settings = settings

    @app.get("/", response_class=HTMLResponse)
    def chat_page() -> HTMLResponse:
        active_service: RagChatService = app.state.service
        return HTMLResponse(
            render_chat_page(
                active_service.config.model_id,
                active_service.available_generation_models(),
                active_service.available_retriever_ids(),
                active_service.retriever.name,
                _runtime_commit(),
            )
        )

    @app.get("/health")
    def health() -> dict[str, Any]:
        expected_commit = _expected_commit()
        actual_commit = _actual_commit()
        return {
            "status": "ok",
            "model": service.config.model_id,
            "generation_model": service.config.model,
            "available_models": service.available_model_ids(),
            "available_generation_models": service.available_generation_models(),
            "benchmark": service.benchmark.name,
            "retriever": service.retriever.name,
            "available_retrievers": service.available_retriever_ids(),
            "dictionary": getattr(service, "dictionary_status", {}),
            "version": {
                "expected_commit": expected_commit,
                "actual_commit": actual_commit,
                "commit_matches_expected": (
                    actual_commit == expected_commit
                    if actual_commit is not None and expected_commit is not None
                    else None
                ),
            },
        }

    @app.get("/v1/models", dependencies=[Depends(_require_bearer)])
    def models(request: Request) -> dict[str, Any]:
        active_service: RagChatService = request.app.state.service
        return {
            "object": "list",
            "data": [
                {
                    "id": model_id,
                    "object": "model",
                    "created": int(active_service.started_at_s),
                    "owned_by": "true-chat",
                }
                for model_id in active_service.available_model_ids()
            ],
        }

    @app.post("/v1/chat/completions", dependencies=[Depends(_require_bearer)])
    def chat_completions(payload: dict[str, Any], request: Request) -> Any:
        active_service: RagChatService = request.app.state.service
        messages = _validate_messages(payload.get("messages"))
        try:
            result = active_service.answer(
                messages,
                request_model=payload.get("model"),
                request_retriever=payload.get("retriever", payload.get("search_algorithm")),
                temperature=_optional_float(payload.get("temperature"), "temperature"),
                max_tokens=_optional_int(
                    payload.get("max_tokens", payload.get("max_completion_tokens")),
                    "max_tokens",
                ),
                top_k=_optional_int(payload.get("top_k"), "top_k"),
                image_top_k=_optional_int(payload.get("image_top_k", payload.get("k_img")), "image_top_k"),
                response_mode=payload.get("response_mode", payload.get("mode")),
                image_rewrite=_optional_bool(
                    payload.get("image_rewrite", payload.get("rewrite_image_query")),
                    "image_rewrite",
                ),
                language=_optional_language(
                    payload.get("language", payload.get("response_language", payload.get("locale"))),
                ),
                memory=_optional_bool(
                    payload.get("memory", payload.get("use_memory", payload.get("chat_memory"))),
                    "memory",
                ),
                score_min=_optional_float(
                    payload.get("score_min", payload.get("min_score", payload.get("retrieval_min_score"))),
                    "score_min",
                ),
                score_max=_optional_float(
                    payload.get("score_max", payload.get("max_score", payload.get("retrieval_max_score"))),
                    "score_max",
                ),
                sort_by_score=_optional_bool(
                    payload.get("sort_by_score", payload.get("retrieval_sort_by_score")),
                    "sort_by_score",
                ),
            )
        except ValueError as exc:
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc
        except RuntimeError as exc:
            raise HTTPException(status_code=status.HTTP_502_BAD_GATEWAY, detail=str(exc)) from exc

        if payload.get("stream") is True:
            return StreamingResponse(
                _stream_chat_completion(result.response),
                media_type="text/event-stream",
            )
        return result.response

    @app.post("/v1/dictionary/lookup", dependencies=[Depends(_require_bearer)])
    def dictionary_lookup(payload: dict[str, Any], request: Request) -> dict[str, Any]:
        active_service: RagChatService = request.app.state.service
        term = payload.get("term", payload.get("query"))
        if not isinstance(term, str):
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="term must be a string")
        try:
            return active_service.lookup_dictionary(
                term,
                top_k=_optional_int(payload.get("top_k"), "top_k"),
                score_min=_optional_float(
                    payload.get("score_min", payload.get("min_score", payload.get("retrieval_min_score"))),
                    "score_min",
                ),
                score_max=_optional_float(
                    payload.get("score_max", payload.get("max_score", payload.get("retrieval_max_score"))),
                    "score_max",
                ),
                sort_by_score=_optional_bool(
                    payload.get("sort_by_score", payload.get("retrieval_sort_by_score")),
                    "sort_by_score",
                ),
            )
        except ValueError as exc:
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc

    return app


def _expected_commit() -> str | None:
    return os.getenv("TRUE_CHAT_EXPECTED_COMMIT") or None


def _actual_commit() -> str | None:
    return os.getenv("TRUE_CHAT_ACTUAL_COMMIT") or None


def _runtime_commit() -> str | None:
    return _actual_commit() or _expected_commit()


def _validate_messages(value: Any) -> list[dict[str, Any]]:
    if not isinstance(value, list):
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="messages must be a list")
    for index, message in enumerate(value):
        if not isinstance(message, dict):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"messages[{index}] must be an object",
            )
    return value


def _optional_int(value: Any, name: str) -> int | None:
    if value is None:
        return None
    if isinstance(value, bool):
        raise ValueError(f"{name} must be an integer")
    try:
        parsed = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be an integer") from exc
    if parsed <= 0:
        raise ValueError(f"{name} must be positive")
    return parsed


def _optional_float(value: Any, name: str) -> float | None:
    if value is None:
        return None
    if isinstance(value, bool):
        raise ValueError(f"{name} must be a number")
    try:
        parsed = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a number") from exc
    if not math.isfinite(parsed):
        raise ValueError(f"{name} must be a finite number")
    return parsed


def _optional_bool(value: Any, name: str) -> bool | None:
    if value is None:
        return None
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"1", "true", "yes", "on"}:
            return True
        if normalized in {"0", "false", "no", "off"}:
            return False
    raise ValueError(f"{name} must be a boolean")


def _optional_language(value: Any) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str):
        raise ValueError("language must be a string")
    normalized = value.strip().lower().replace("_", "-")
    if normalized in {"vi", "vi-vn", "vietnamese", "tieng-viet", "tiếng-việt"}:
        return "vi"
    if normalized in {"en", "en-us", "en-gb", "english"}:
        return "en"
    if normalized in {"", "system", "auto"}:
        return None
    raise ValueError("language must be one of: en, vi")


def _require_bearer(
    request: Request,
    authorization: str | None = Header(default=None),
) -> None:
    settings: ApiSettings = request.app.state.settings
    if not settings.api_key:
        return
    expected = f"Bearer {settings.api_key}"
    if authorization != expected:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid API key")


def _stream_chat_completion(response: dict[str, Any]) -> Any:
    completion_id = response.get("id") or f"chatcmpl-{uuid.uuid4().hex}"
    created = response.get("created") or int(time.time())
    model = response.get("model") or DEFAULT_PROXY_MODEL_ID
    content = response["choices"][0]["message"]["content"]
    rag = response.get("rag")
    chunks = [
        {
            "id": completion_id,
            "object": "chat.completion.chunk",
            "created": created,
            "model": model,
            "choices": [{"index": 0, "delta": {"role": "assistant"}, "finish_reason": None}],
        },
        {
            "id": completion_id,
            "object": "chat.completion.chunk",
            "created": created,
            "model": model,
            "choices": [{"index": 0, "delta": {"content": content}, "finish_reason": None}],
            "rag": rag,
        },
        {
            "id": completion_id,
            "object": "chat.completion.chunk",
            "created": created,
            "model": model,
            "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
            "answer": content,
            "rag": rag,
        },
    ]
    for chunk in chunks:
        yield f"data: {json.dumps(chunk, ensure_ascii=False)}\n\n"
    yield "data: [DONE]\n\n"
