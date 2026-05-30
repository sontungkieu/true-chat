from __future__ import annotations

from dataclasses import dataclass

from fastapi.testclient import TestClient

from rag_bench.api import create_app
from rag_bench.chat_service import ChatProxyConfig, ChatServiceResult
from rag_bench.groq_client import GenerationResult
from rag_bench.types import BenchmarkData, RetrievalHit


@dataclass
class FakeRetriever:
    name: str = "bm25"


class FakeService:
    def __init__(self) -> None:
        self.config = ChatProxyConfig(model_id="rag-test")
        self.benchmark = BenchmarkData(name="fixture", dataset_id="fixture/test", queries=[], documents=[], qrels={})
        self.retriever = FakeRetriever()
        self.started_at_s = 123.0
        self.seen_messages = None
        self.seen_model = None
        self.seen_retriever = None
        self.seen_temperature = None
        self.seen_max_tokens = None
        self.seen_top_k = None
        self.seen_image_top_k = None
        self.seen_response_mode = None
        self.seen_image_rewrite = None
        self.seen_language = None
        self.seen_memory = None
        self.seen_score_min = None
        self.seen_score_max = None
        self.seen_sort_by_score = None

    def answer(
        self,
        messages,
        *,
        request_model=None,
        request_retriever=None,
        temperature=None,
        max_tokens=None,
        top_k=None,
        image_top_k=None,
        response_mode=None,
        image_rewrite=None,
        language=None,
        memory=None,
        score_min=None,
        score_max=None,
        sort_by_score=None,
    ):
        self.seen_messages = messages
        self.seen_model = request_model
        self.seen_retriever = request_retriever
        self.seen_temperature = temperature
        self.seen_max_tokens = max_tokens
        self.seen_top_k = top_k
        self.seen_image_top_k = image_top_k
        self.seen_response_mode = response_mode
        self.seen_image_rewrite = image_rewrite
        self.seen_language = language
        self.seen_memory = memory
        self.seen_score_min = score_min
        self.seen_score_max = score_max
        self.seen_sort_by_score = sort_by_score
        response = {
            "id": "chatcmpl-test",
            "object": "chat.completion",
            "created": 123,
            "model": request_model or self.config.model_id,
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": "hello [doc-1]"},
                    "finish_reason": "stop",
                }
            ],
            "usage": {
                "prompt_tokens": 3,
                "completion_tokens": 2,
                "total_tokens": 5,
                "estimated_tokens": 7,
            },
            "rag": {
                "retriever": request_retriever or "bm25",
                "retrieved": [{"doc_id": "doc-1", "rank": 1, "text": "Document text"}],
                "rejected_aliases": ["bad-key"],
                "output_tokens_per_s": 200.0,
            },
        }
        return ChatServiceResult(
            response=response,
            generation=GenerationResult(
                answer="hello [doc-1]",
                key_alias="a",
                attempted_aliases=["a"],
                latency_s=0.01,
                retry_count=0,
            ),
            hits=[RetrievalHit("doc-1", 1.0, 1)],
            retrieval_latency_s=0.01,
        )

    def available_model_ids(self) -> tuple[str, ...]:
        return ("rag-test", "llama-3.1-8b-instant", "qwen/qwen3-32b")

    def available_generation_models(self) -> tuple[str, ...]:
        return ("llama-3.1-8b-instant", "qwen/qwen3-32b")

    def available_retriever_ids(self) -> tuple[str, ...]:
        return ("bm25", "tfidf")

    def lookup_dictionary(self, term, *, top_k=None, score_min=None, score_max=None, sort_by_score=None):
        return {
            "object": "dictionary.lookup",
            "query": term,
            "retriever": "dictionary-graph",
            "top_k": top_k or 1,
            "retrieval_metadata": {
                "score_filter": {
                    "min_score": score_min,
                    "max_score": score_max,
                    "sort_by_score": bool(sort_by_score),
                }
            },
            "retrieved": [
                {
                    "doc_id": "base:D-0001",
                    "rank": 1,
                    "score": 1.0,
                    "title": "ĐKZ",
                    "text": "ĐKZ entry",
                    "metadata": {"kind": "dictionary", "query_highlights": [term]},
                    "kind": "dictionary",
                    "query_highlights": [term],
                }
            ],
        }


def test_health_and_models() -> None:
    service = FakeService()
    client = TestClient(create_app(service))

    assert client.get("/health").json()["status"] == "ok"
    models = client.get("/v1/models").json()

    assert models["object"] == "list"
    assert models["data"][0]["id"] == "rag-test"
    assert [model["id"] for model in models["data"]] == [
        "rag-test",
        "llama-3.1-8b-instant",
        "qwen/qwen3-32b",
    ]
    health = client.get("/health").json()
    assert health["available_generation_models"] == ["llama-3.1-8b-instant", "qwen/qwen3-32b"]
    assert health["available_retrievers"] == ["bm25", "tfidf"]
    assert health["version"]["commit_matches_expected"] is None


def test_health_reports_runtime_commit_match(monkeypatch) -> None:
    monkeypatch.setenv("TRUE_CHAT_EXPECTED_COMMIT", "abc123")
    monkeypatch.setenv("TRUE_CHAT_ACTUAL_COMMIT", "abc123")
    client = TestClient(create_app(FakeService()))

    health = client.get("/health").json()

    assert health["version"] == {
        "expected_commit": "abc123",
        "actual_commit": "abc123",
        "commit_matches_expected": True,
    }


def test_chat_page() -> None:
    client = TestClient(create_app(FakeService()))

    response = client.get("/")

    assert response.status_code == 200
    assert "text/html" in response.headers["content-type"]
    assert "RAG Chat" in response.text
    assert "rag-test" in response.text
    assert "How can I help you today?" in response.text
    assert "New Chat" in response.text
    assert "doc-panel" in response.text
    assert "openDocument" in response.text
    assert "grid-template-columns: minmax(0, 1fr) 36px" in response.text
    assert "word-break: break-word" in response.text
    assert "sidebar-collapsed" in response.text
    assert "deleteConversation" in response.text
    assert "conversation-delete" in response.text
    assert "renameConversation" in response.text
    assert "conversation-actions" in response.text
    assert "conversation-action" in response.text
    assert "Rename chat" in response.text
    assert "editUserMessage" in response.text
    assert "submitEditedUserMessage" in response.text
    assert "cancelEdit" in response.text
    assert "cancelEditMode" in response.text
    assert "themeMode" in response.text
    assert "modelChoice" in response.text
    assert "searchChoice" in response.text
    assert "Search" in response.text
    assert "TF-IDF" in response.text
    assert "Graph BM25" in response.text
    assert "Qwen3 32B" in response.text
    assert "qwen/qwen3-32b" in response.text
    assert 'const DEFAULT_CHAT_MODEL = "qwen/qwen3-32b"' in response.text
    assert 'const DEFAULT_LANGUAGE = "vi"' in response.text
    assert 'const DEFAULT_RESPONSE_MODE = "dictionary"' in response.text
    assert "const DEFAULT_MEMORY_ENABLED = false" in response.text
    assert "const DEFAULT_DICTIONARY_CROSS_REF = true" in response.text
    assert "const DEFAULT_MAX_TOKENS = 4096" in response.text
    assert "SETTINGS_SCHEMA_VERSION = 2" in response.text
    assert "runtimeVersion" in response.text
    assert "const APP_VERSION = \"\"" in response.text
    assert "__APP_VERSION_JSON__" not in response.text
    assert "MiMo V2.5 Pro" in response.text
    assert "modelSelector" not in response.text
    assert "activeTitle" not in response.text
    assert "model-selector" not in response.text
    assert "toggleModelMenu" not in response.text
    assert "responseModeTool" in response.text
    assert "imageRewriteTool" in response.text
    assert "menu-chip" in response.text
    assert "chip-divider" in response.text
    assert "chip-caret" in response.text
    assert "normalizedTextRetrieverChoices" in response.text
    assert 'retriever !== "image-digits" && retriever !== "dictionary-graph"' in response.text
    assert "Dictionary" in response.text
    assert "Từ điển" in response.text
    assert "isDictionaryCommand" in response.text
    assert 'response_mode: requestOptions.response_mode' in response.text
    assert '"dictionary-graph"' in response.text
    assert "renderRichBlocks" in response.text
    assert "rich-run" in response.text
    assert "renderDictionaryAnswer" in response.text
    assert "dictionary-inline" in response.text
    assert "dictionary-inline-list" in response.text
    assert "dictionary-relevance \" + relevance" in response.text
    assert "dictionaryRelevance" in response.text
    assert "Khớp" in response.text
    assert "Liên quan" in response.text
    assert "dictionaryCrossRef" in response.text
    assert "/v1/dictionary/lookup" in response.text
    assert "openDictionaryCrossReference" in response.text
    assert "clickedDictionaryTerm" in response.text
    assert "xref-term" in response.text
    assert "docTrail" in response.text
    assert "renderDocumentTrail" in response.text
    assert "openDocumentFromTrail" in response.text
    assert "if (options.fromCrossRef) {" in response.text
    assert "options.fromCrossRef && selectedSource && !sameDocumentSource" not in response.text
    assert "referenceTrail" in response.text
    assert "backToPreviousEntry" in response.text
    assert "dictionaryCrossRefHint" in response.text
    assert "query-highlight" in response.text
    assert "sourceHighlightTerms" in response.text
    assert "dictionaryMatchMode" in response.text
    assert "strictHighlights" in response.text
    assert "strictForMatchWithMap" in response.text
    assert "dictionaryGraphPath" in response.text
    assert "renderDictionaryGraphPath" in response.text
    assert "dictionaryRelationLabel" in response.text
    assert "dictionary-graph-path" in response.text
    assert "isHighlightBoundary" in response.text
    assert '? "dd"' in response.text
    assert "dictionarySourceLabel" in response.text
    assert "Từ điển PB 2021" in response.text
    assert "Bổ sung 2021" in response.text
    assert "dictionaryInlineSources" in response.text
    assert "isDictionaryDisplaySource" in response.text
    assert "dictionaryDisplayText" in response.text
    assert "MAX_INLINE_DICTIONARY_SOURCES" in response.text
    assert "dictionaryAnswerParts" in response.text
    assert "positionComposerToolMenu" in response.text
    assert "--composer-menu-left" in response.text
    assert '--composer-menu-bottom' in response.text
    assert 'closest("#composerToolMenu")' in response.text
    assert "imageTopK" in response.text
    assert "user-request-meta" in response.text
    assert "captureRequestOptions" in response.text
    assert "formatUserRequestMeta" in response.text
    assert '(mode === "image" || mode === "text_image") && request.image_rewrite' in response.text
    assert "image_top_k: requestOptions.image_top_k" in response.text
    assert "composerToolMenu" in response.text
    assert "imageSource" in response.text
    assert "imageLightbox" in response.text
    assert "renderImageGrid" in response.text
    assert "response_mode: requestOptions.response_mode" in response.text
    assert "startChatSwipe" in response.text
    assert "touchstart" in response.text
    assert "__AVAILABLE_RETRIEVERS_JSON__" not in response.text
    assert "__DEFAULT_RETRIEVER_JSON__" not in response.text
    assert "__AVAILABLE_MODELS_JSON__" not in response.text
    assert "languageMode" in response.text
    assert "Tiếng Việt" in response.text
    assert "devMode" in response.text
    assert "settings.devMode" in response.text
    assert "memoryMode" in response.text
    assert "settings.memory" in response.text
    assert "memory: requestOptions.memory" in response.text
    assert "compactStateForStorage" in response.text
    assert "compactSourceForStorage" in response.text
    assert "shouldRenderStreamingDelta" in response.text
    assert "updatePendingAssistant(content, rag)" in response.text
    assert "exportChatHistory" in response.text
    assert "importChatHistory" in response.text
    assert "normalizeImportedHistory" in response.text
    assert "settingsForExport" in response.text
    assert "delete exported.apiKey" in response.text
    assert "editAssistantFeedback" in response.text
    assert "feedback-note" in response.text
    assert "feedbackPrompt" in response.text
    assert "Đang sửa câu hỏi" in response.text
    assert "message-footer" in response.text
    assert "think-details" in response.text
    assert "splitThinkContent" in response.text
    assert "renderTextWithCitations" in response.text
    assert "sourceLookupKeys" in response.text
    assert "sourceInfo.source_entry_id" in response.text
    assert 'text.split(":").pop().trim()' in response.text
    assert "isDictionaryHeaderCitation" in response.text
    assert "renderMarkdownBlocks" in response.text
    assert "markdown-content" in response.text
    assert "citation-ref" in response.text
    assert "ragDetailsOpen" in response.text
    assert "ragDetailsKey" in response.text
    assert "min-width: 1.7em" in response.text
    assert "font-size: 0.74em" in response.text
    assert "Citations and related documents" in response.text
    assert "Trích dẫn và tài liệu liên quan" in response.text
    assert "No final answer returned" in response.text
    assert "fontScale" in response.text
    assert "--font-scale" in response.text
    assert "Font size" in response.text
    assert 'max="200"' in response.text
    assert 'data-theme="colorful"' in response.text
    assert "#228B22" in response.text
    assert "fetchAssistantJson" in response.text
    assert "Model returned an empty answer" in response.text
    assert 'data-app="rag-chat"' in response.text
    assert "viewport-fit=cover" in response.text
    assert "--safe-bottom" in response.text
    assert "syncViewportHeight" in response.text
    assert 'fetch("/v1/chat/completions"' in response.text
    assert "stream: true" in response.text
    assert 'buffer.split("\\n\\n")' in response.text
    assert 'buffer.split("\\\\n\\\\n")' not in response.text
    assert "parseSseEvent" in response.text
    assert "localStorage" in response.text
    assert 'id="sourceTopK"' in response.text
    assert "score_min: requestOptions.score_min" in response.text
    assert "sort_by_score: requestOptions.sort_by_score" in response.text
    assert "function showDebugSourceMetadata()" in response.text
    assert 'id="dictionaryXrefPopover"' in response.text
    assert "renderDictionaryCrossReferencePopup" in response.text
    assert "body: JSON.stringify({ term, top_k: topK })" in response.text
    assert "keyboard-open" in response.text
    assert "--keyboard-inset" in response.text
    assert "updateComposerReservedHeight" in response.text
    assert ".settings[open] .settings-body" in response.text


def test_chat_page_includes_runtime_commit(monkeypatch) -> None:
    monkeypatch.setenv("TRUE_CHAT_ACTUAL_COMMIT", "abc123def456")
    client = TestClient(create_app(FakeService()))

    response = client.get("/")

    assert response.status_code == 200
    assert "runtimeVersionValue" in response.text
    assert 'const APP_VERSION = "abc123def456"' in response.text


def test_chat_completion_non_stream() -> None:
    service = FakeService()
    client = TestClient(create_app(service))

    response = client.post(
        "/v1/chat/completions",
        json={
            "model": "rag-test",
            "retriever": "tfidf",
            "messages": [{"role": "user", "content": "hi"}],
            "temperature": 0.1,
            "max_tokens": 32,
            "top_k": 5,
            "image_top_k": 4,
            "response_mode": "text_image",
            "image_rewrite": True,
            "language": "vi",
            "memory": False,
            "score_min": 0.25,
            "score_max": 2.5,
            "sort_by_score": True,
        },
    )

    assert response.status_code == 200
    body = response.json()
    assert body["object"] == "chat.completion"
    assert body["model"] == "rag-test"
    assert body["rag"]["retriever"] == "tfidf"
    assert body["choices"][0]["message"]["content"] == "hello [doc-1]"
    assert service.seen_model == "rag-test"
    assert service.seen_retriever == "tfidf"
    assert service.seen_temperature == 0.1
    assert service.seen_max_tokens == 32
    assert service.seen_top_k == 5
    assert service.seen_image_top_k == 4
    assert service.seen_response_mode == "text_image"
    assert service.seen_image_rewrite is True
    assert service.seen_language == "vi"
    assert service.seen_memory is False
    assert service.seen_score_min == 0.25
    assert service.seen_score_max == 2.5
    assert service.seen_sort_by_score is True


def test_chat_completion_accepts_qwen_model_choice() -> None:
    service = FakeService()
    client = TestClient(create_app(service))

    response = client.post(
        "/v1/chat/completions",
        json={
            "model": "qwen/qwen3-32b",
            "messages": [{"role": "user", "content": "hi"}],
        },
    )

    assert response.status_code == 200
    assert response.json()["model"] == "qwen/qwen3-32b"
    assert service.seen_model == "qwen/qwen3-32b"


def test_chat_completion_rejects_unknown_language() -> None:
    service = FakeService()
    client = TestClient(create_app(service))

    response = client.post(
        "/v1/chat/completions",
        json={"messages": [{"role": "user", "content": "hi"}], "language": "fr"},
    )

    assert response.status_code == 400
    assert response.json()["detail"] == "language must be one of: en, vi"


def test_chat_completion_stream() -> None:
    client = TestClient(create_app(FakeService()))

    response = client.post(
        "/v1/chat/completions",
        json={"messages": [{"role": "user", "content": "hi"}], "stream": True},
    )

    assert response.status_code == 200
    assert "text/event-stream" in response.headers["content-type"]
    assert "chat.completion.chunk" in response.text
    assert '"doc_id": "doc-1"' in response.text
    assert '"text": "Document text"' in response.text
    assert '"answer": "hello [doc-1]"' in response.text
    assert response.text.count('"rag"') >= 2
    assert '"rejected_aliases": ["bad-key"]' in response.text
    assert '"output_tokens_per_s": 200.0' in response.text
    assert "data: [DONE]" in response.text


def test_chat_completion_rejects_malformed_messages() -> None:
    client = TestClient(create_app(FakeService()))

    response = client.post("/v1/chat/completions", json={"messages": ["bad"]})

    assert response.status_code == 400
    assert response.json()["detail"] == "messages[0] must be an object"


def test_dictionary_lookup_endpoint() -> None:
    client = TestClient(create_app(FakeService()))

    response = client.post("/v1/dictionary/lookup", json={"term": "ĐKZ", "top_k": 1})

    assert response.status_code == 200
    data = response.json()
    assert data["object"] == "dictionary.lookup"
    assert data["query"] == "ĐKZ"
    assert data["retrieved"][0]["title"] == "ĐKZ"


def test_dictionary_lookup_rejects_bad_term() -> None:
    client = TestClient(create_app(FakeService()))

    response = client.post("/v1/dictionary/lookup", json={"term": 123})

    assert response.status_code == 400
    assert response.json()["detail"] == "term must be a string"


def test_optional_auth() -> None:
    client = TestClient(create_app(FakeService(), api_key="secret"))

    assert client.get("/v1/models").status_code == 401
    assert client.get("/v1/models", headers={"Authorization": "Bearer secret"}).status_code == 200
    assert client.post("/v1/dictionary/lookup", json={"term": "ĐKZ"}).status_code == 401
    assert (
        client.post(
            "/v1/dictionary/lookup",
            headers={"Authorization": "Bearer secret"},
            json={"term": "ĐKZ"},
        ).status_code
        == 200
    )
    assert client.post("/v1/dictionary/lookup", json={"term": "ĐKZ"}).status_code == 401
    assert (
        client.post(
            "/v1/dictionary/lookup",
            headers={"Authorization": "Bearer secret"},
            json={"term": "ĐKZ"},
        ).status_code
        == 200
    )
