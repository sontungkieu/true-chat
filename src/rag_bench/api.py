from __future__ import annotations

import json
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
            )
        )

    @app.get("/health")
    def health() -> dict[str, Any]:
        return {
            "status": "ok",
            "model": service.config.model_id,
            "generation_model": service.config.model,
            "available_models": service.available_model_ids(),
            "available_generation_models": service.available_generation_models(),
            "benchmark": service.benchmark.name,
            "retriever": service.retriever.name,
            "available_retrievers": service.available_retriever_ids(),
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

    return app


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
        return float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a number") from exc


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
