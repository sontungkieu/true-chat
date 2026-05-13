from __future__ import annotations

from dataclasses import dataclass

from rag_bench.api import create_app
from rag_bench.chat_service import ChatProxyConfig, RagChatService


@dataclass(frozen=True)
class ServeConfig:
    host: str
    port: int
    api_key: str | None
    chat: ChatProxyConfig


def serve_proxy(config: ServeConfig) -> None:
    try:
        import uvicorn
    except ImportError as exc:
        raise RuntimeError("Serving requires uvicorn. Install pinned dependencies with: uv sync --frozen") from exc

    service = RagChatService.from_config(config.chat)
    app = create_app(service, api_key=config.api_key)
    uvicorn.run(app, host=config.host, port=config.port)
