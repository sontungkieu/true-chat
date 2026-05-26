from __future__ import annotations

from dataclasses import dataclass

from rag_bench.chat_service import (
    ChatProxyConfig,
    ModelRoutedChatClient,
    RagChatService,
    last_user_text,
    parse_chat_command,
)
from rag_bench.groq_client import GenerationResult
from rag_bench.types import BenchmarkData, Document, Query, RetrievalHit, RetrievalResult
from rag_bench.web_search import WebSearchResult


@dataclass
class FakeRetriever:
    name: str = "bm25"
    build_time_s: float = 0.0

    def search(self, query: Query, top_k: int) -> RetrievalResult:
        assert query.text == "What do cats do?"
        assert top_k == 2
        return RetrievalResult(
            query=query,
            hits=[
                RetrievalHit(
                    doc_id="cat-doc",
                    score=1.5,
                    rank=1,
                    title="Cats",
                    text="Cats purr and chase toys.",
                )
            ],
            latency_s=0.02,
        )


@dataclass
class FakeImageRetriever:
    name: str = "image-digits"
    build_time_s: float = 0.0
    seen_top_k: int | None = None

    def search(self, query: Query, top_k: int) -> RetrievalResult:
        assert query.text == "digit 7"
        self.seen_top_k = top_k
        return RetrievalResult(
            query=query,
            hits=[
                RetrievalHit(
                    doc_id="image-7",
                    score=101.0,
                    rank=1,
                    title="Digit 7",
                    text="A handwritten digit 7 image.",
                    metadata={
                        "kind": "image",
                        "image_data_url": "data:image/svg+xml,%3Csvg%3E%3C/svg%3E",
                        "label": 7,
                        "dataset": "fixture-images",
                    },
                )
            ],
            latency_s=0.01,
            metadata={"kind": "image", "dataset": "fixture-images"},
        )


@dataclass
class FakeKeywordRetriever:
    name: str = "keyword-match"
    build_time_s: float = 0.0
    seen_queries: list[str] | None = None

    def search(self, query: Query, top_k: int) -> RetrievalResult:
        if self.seen_queries is None:
            self.seen_queries = []
        self.seen_queries.append(query.text)
        score = 4.0 if "BH1" in query.text else 0.0
        return RetrievalResult(
            query=query,
            hits=[
                RetrievalHit(
                    doc_id="bcl2-doc",
                    score=score,
                    rank=1,
                    title="BH1 and BH2 domains of Bcl-2",
                    text="BH1 and BH2 domains of Bcl-2 are required for apoptosis inhibition.",
                ),
                RetrievalHit(
                    doc_id="noise-doc",
                    score=0.0,
                    rank=2,
                    title="Unrelated",
                    text="Unrelated document.",
                ),
            ],
            latency_s=0.01,
        )


@dataclass
class FakeDictionaryRetriever:
    name: str = "dictionary-graph"
    build_time_s: float = 0.0
    seen_query: str | None = None

    def search(self, query: Query, top_k: int) -> RetrievalResult:
        self.seen_query = query.text
        assert top_k == 3
        return RetrievalResult(
            query=query,
            hits=[
                RetrievalHit(
                    doc_id="A-0001",
                    score=1.0,
                    rank=1,
                    title="AMONIT",
                    text="AMONIT, thuốc nổ phá.",
                    metadata={
                        "kind": "dictionary",
                        "headword": "AMONIT",
                        "raw_docx_text": "AMONIT, thuốc nổ phá.",
                        "rich_blocks": [{"type": "paragraph", "runs": [{"text": "AMONIT", "bold": True}]}],
                    },
                )
            ],
            latency_s=0.01,
            metadata={"kind": "dictionary"},
        )


class FakeLLM:
    def __init__(self, alias: str = "alias-a") -> None:
        self.alias = alias
        self.key_usage_counts = {"alias-a": 1}
        self.messages: list[dict[str, str]] = []
        self.model: str | None = None
        self.temperature: float | None = None
        self.max_completion_tokens: int | None = None

    def generate(
        self,
        messages: list[dict[str, str]],
        *,
        model: str | None = None,
        temperature: float = 0.0,
        max_completion_tokens: int = 512,
    ) -> GenerationResult:
        self.messages = messages
        self.model = model
        self.temperature = temperature
        self.max_completion_tokens = max_completion_tokens
        return GenerationResult(
            answer="Cats purr [cat-doc]",
            key_alias=self.alias,
            attempted_aliases=[self.alias],
            latency_s=0.03,
            retry_count=0,
            prompt_tokens=20,
            completion_tokens=5,
            total_tokens=25,
            estimated_tokens=30,
            output_tokens_per_s=166.7,
        )

    def rate_limit_snapshot(self) -> dict[str, dict[str, int]]:
        return {"alias-a": {"tokens_used": 30, "requests_used": 1}}


class FakeImageRewriteLLM(FakeLLM):
    def generate(
        self,
        messages: list[dict[str, str]],
        *,
        model: str | None = None,
        temperature: float = 0.0,
        max_completion_tokens: int = 512,
    ) -> GenerationResult:
        self.messages = messages
        self.model = model
        self.temperature = temperature
        self.max_completion_tokens = max_completion_tokens
        return GenerationResult(
            answer="digit 7",
            key_alias="alias-img",
            attempted_aliases=["alias-img"],
            latency_s=0.01,
            retry_count=0,
            prompt_tokens=10,
            completion_tokens=2,
            total_tokens=12,
            estimated_tokens=12,
        )


class FakeKeywordLLM(FakeLLM):
    def __init__(self) -> None:
        super().__init__()
        self.calls = 0

    def generate(
        self,
        messages: list[dict[str, str]],
        *,
        model: str | None = None,
        temperature: float = 0.0,
        max_completion_tokens: int = 512,
    ) -> GenerationResult:
        self.calls += 1
        self.messages = messages
        self.model = model
        self.temperature = temperature
        self.max_completion_tokens = max_completion_tokens
        if self.calls == 1:
            answer = '["BH1", "BH1 Bcl-2", "BH1 domain apoptosis"]'
            completion_tokens = 12
        else:
            answer = "BH1 is a Bcl-2 domain [bcl2-doc]."
            completion_tokens = 8
        return GenerationResult(
            answer=answer,
            key_alias="alias-keyword",
            attempted_aliases=["alias-keyword"],
            latency_s=0.01,
            retry_count=0,
            prompt_tokens=15,
            completion_tokens=completion_tokens,
            total_tokens=15 + completion_tokens,
            estimated_tokens=30,
        )


class FakeWebSearchClient:
    def __init__(self) -> None:
        self.query: str | None = None
        self.limit: int | None = None

    def search(self, query: str, *, limit: int) -> list[WebSearchResult]:
        self.query = query
        self.limit = limit
        return [
            WebSearchResult(
                title="True Chat release notes",
                url="https://example.test/true-chat",
                snippet="True Chat added web search for internship demos.",
            )
        ]


def test_rag_chat_service_answers_with_retrieved_context_and_history() -> None:
    llm = FakeLLM()
    service = RagChatService(
        config=ChatProxyConfig(top_k=2, model_id="rag-test", max_completion_tokens=64, temperature=0.2),
        benchmark=BenchmarkData(
            name="fixture",
            dataset_id="fixture/test",
            queries=[],
            documents=[Document("cat-doc", "Cats purr and chase toys.", "Cats")],
            qrels={},
        ),
        retriever=FakeRetriever(),
        llm=llm,
    )

    result = service.answer(
        [
            {"role": "user", "content": "Previous question"},
            {"role": "assistant", "content": "Previous answer"},
            {"role": "user", "content": "What do cats do?"},
        ]
    )

    assert result.response["model"] == "rag-test"
    assert result.response["choices"][0]["message"]["content"] == "Cats purr [cat-doc]"
    assert result.response["usage"]["total_tokens"] == 25
    assert result.response["rag"]["retrieved"][0]["doc_id"] == "cat-doc"
    assert result.response["rag"]["retrieved"][0]["text"] == "Cats purr and chase toys."
    assert result.response["rag"]["key_alias"] == "alias-a"
    assert result.response["rag"]["rejected_aliases"] == []
    assert result.response["rag"]["output_tokens_per_s"] == 166.7
    assert result.response["rag"]["generation_model"] == "llama-3.1-8b-instant"
    assert llm.model == "llama-3.1-8b-instant"
    assert llm.temperature == 0.2
    assert llm.max_completion_tokens == 64
    assert "Required response language: English" in llm.messages[0]["content"]
    user_prompt = llm.messages[1]["content"]
    assert "Previous question" in user_prompt
    assert "Cats purr and chase toys." in user_prompt
    assert "What do cats do?" in user_prompt


def test_rag_chat_service_can_disable_memory_history() -> None:
    llm = FakeLLM()
    service = RagChatService(
        config=ChatProxyConfig(top_k=2, model_id="rag-test"),
        benchmark=BenchmarkData(
            name="fixture",
            dataset_id="fixture/test",
            queries=[],
            documents=[Document("cat-doc", "Cats purr and chase toys.", "Cats")],
            qrels={},
        ),
        retriever=FakeRetriever(),
        llm=llm,
    )

    result = service.answer(
        [
            {"role": "user", "content": "Previous question"},
            {"role": "assistant", "content": "Previous answer"},
            {"role": "user", "content": "What do cats do?"},
        ],
        memory=False,
    )

    user_prompt = llm.messages[1]["content"]
    assert result.response["rag"]["retrieval_metadata"]["memory"] is False
    assert "No prior conversation." in user_prompt
    assert "Previous question" not in user_prompt
    assert "Previous answer" not in user_prompt
    assert "What do cats do?" in user_prompt


def test_rag_chat_service_forces_selected_response_language() -> None:
    llm = FakeLLM()
    service = RagChatService(
        config=ChatProxyConfig(top_k=2, model_id="rag-test"),
        benchmark=BenchmarkData(
            name="fixture",
            dataset_id="fixture/test",
            queries=[],
            documents=[Document("cat-doc", "Cats purr and chase toys.", "Cats")],
            qrels={},
        ),
        retriever=FakeRetriever(),
        llm=llm,
    )

    result = service.answer(
        [{"role": "user", "content": "What do cats do?"}],
        language="vi",
    )

    assert result.response["rag"]["retrieval_metadata"]["language"] == "vi"
    assert "Required response language: Vietnamese" in llm.messages[0]["content"]
    assert "Answer only in Vietnamese" in llm.messages[0]["content"]


def test_rag_chat_service_can_switch_to_qwen_model() -> None:
    llm = FakeLLM()
    service = RagChatService(
        config=ChatProxyConfig(top_k=2, model_id="rag-test", image_top_k=5),
        benchmark=BenchmarkData(
            name="fixture",
            dataset_id="fixture/test",
            queries=[],
            documents=[Document("cat-doc", "Cats purr and chase toys.", "Cats")],
            qrels={},
        ),
        retriever=FakeRetriever(),
        llm=llm,
    )

    result = service.answer(
        [{"role": "user", "content": "What do cats do?"}],
        request_model="qwen/qwen3-32b",
    )

    assert result.response["model"] == "qwen/qwen3-32b"
    assert result.response["rag"]["generation_model"] == "qwen/qwen3-32b"
    assert llm.model == "qwen/qwen3-32b"


def test_model_routed_chat_client_routes_mimo_models() -> None:
    groq = FakeLLM(alias="groq-a")
    mimo = FakeLLM(alias="mimo")
    router = ModelRoutedChatClient(default_client=groq, routes={"mimo-v2.5-pro": mimo})

    result = router.generate([{"role": "user", "content": "hello"}], model="mimo-v2.5-pro")
    fallback = router.generate([{"role": "user", "content": "hello"}], model="qwen/qwen3-32b")

    assert result.key_alias == "mimo"
    assert mimo.model == "mimo-v2.5-pro"
    assert fallback.key_alias == "groq-a"
    assert groq.model == "qwen/qwen3-32b"


def test_available_models_include_mimo_only_when_enabled() -> None:
    service = RagChatService(
        config=ChatProxyConfig(top_k=2, model_id="rag-test", mimo_enabled=True),
        benchmark=BenchmarkData(
            name="fixture",
            dataset_id="fixture/test",
            queries=[],
            documents=[],
            qrels={},
        ),
        retriever=FakeRetriever(),
        llm=FakeLLM(),
    )

    assert "mimo-v2.5-pro" in service.available_generation_models()


def test_rag_chat_service_resolves_retriever_alias() -> None:
    service = RagChatService(
        config=ChatProxyConfig(top_k=2, model_id="rag-test"),
        benchmark=BenchmarkData(
            name="fixture",
            dataset_id="fixture/test",
            queries=[],
            documents=[Document("cat-doc", "Cats purr and chase toys.", "Cats")],
            qrels={},
        ),
        retriever=FakeRetriever(),
        llm=FakeLLM(),
    )

    assert service.resolve_request_retriever("lexical").name == "bm25"


def test_text_mode_ignores_image_retriever_request() -> None:
    llm = FakeLLM()
    text_retriever = FakeRetriever()
    image_retriever = FakeImageRetriever()
    service = RagChatService(
        config=ChatProxyConfig(top_k=2, model_id="rag-test", image_enabled=True),
        benchmark=BenchmarkData(
            name="fixture",
            dataset_id="fixture/test",
            queries=[],
            documents=[Document("cat-doc", "Cats purr and chase toys.", "Cats")],
            qrels={},
        ),
        retriever=text_retriever,
        llm=llm,
        retrievers={"bm25": text_retriever, "image-digits": image_retriever},
    )

    result = service.answer(
        [{"role": "user", "content": "What do cats do?"}],
        request_retriever="image-digits",
        response_mode="text",
    )

    assert result.response["rag"]["retriever"] == "bm25"
    assert result.response["rag"]["retrieved"][0]["doc_id"] == "cat-doc"
    assert image_retriever.seen_top_k is None


def test_web_command_uses_web_search_results_as_rag_context() -> None:
    llm = FakeLLM()
    web = FakeWebSearchClient()
    service = RagChatService(
        config=ChatProxyConfig(
            top_k=2,
            web_search_top_k=3,
            web_search_privilege_key="search-secret",
            model_id="rag-test",
        ),
        benchmark=BenchmarkData(
            name="fixture",
            dataset_id="fixture/test",
            queries=[],
            documents=[],
            qrels={},
        ),
        retriever=FakeRetriever(),
        llm=llm,
        web_search_client=web,
    )

    result = service.answer(
        [{"role": "user", "content": "/web true chat internship"}],
        web_search_key="search-secret",
        language="vi",
    )

    assert web.query == "true chat internship"
    assert web.limit == 3
    assert result.response["rag"]["retriever"] == "web-search"
    assert result.response["rag"]["retrieval_metadata"]["response_mode"] == "web"
    assert result.response["rag"]["retrieval_metadata"]["web_result_count"] == 1
    assert result.response["rag"]["retrieved"][0]["doc_id"] == "web-1"
    assert result.response["rag"]["retrieved"][0]["url"] == "https://example.test/true-chat"
    assert "Web search results" in llm.messages[1]["content"]
    assert "True Chat added web search" in llm.messages[1]["content"]
    assert "Required response language: Vietnamese" in llm.messages[0]["content"]


def test_web_search_requires_privilege_key() -> None:
    service = RagChatService(
        config=ChatProxyConfig(
            top_k=2,
            web_search_top_k=3,
            web_search_privilege_key="search-secret",
            model_id="rag-test",
        ),
        benchmark=BenchmarkData(
            name="fixture",
            dataset_id="fixture/test",
            queries=[],
            documents=[],
            qrels={},
        ),
        retriever=FakeRetriever(),
        llm=FakeLLM(),
        web_search_client=FakeWebSearchClient(),
    )

    try:
        service.answer([{"role": "user", "content": "/web true chat internship"}], web_search_key="wrong")
    except ValueError as exc:
        assert "Invalid web search privilege key" in str(exc)
    else:
        raise AssertionError("web search should reject an invalid privilege key")


def test_web_search_requires_configured_privilege_key() -> None:
    service = RagChatService(
        config=ChatProxyConfig(top_k=2, web_search_top_k=3, model_id="rag-test"),
        benchmark=BenchmarkData(
            name="fixture",
            dataset_id="fixture/test",
            queries=[],
            documents=[],
            qrels={},
        ),
        retriever=FakeRetriever(),
        llm=FakeLLM(),
        web_search_client=FakeWebSearchClient(),
    )

    try:
        service.answer([{"role": "user", "content": "/web true chat internship"}], web_search_key="search-secret")
    except ValueError as exc:
        assert "Web search privilege key is not configured" in str(exc)
    else:
        raise AssertionError("web search should require a configured server-side key")


def test_image_and_dictionary_modes_are_disabled_by_default() -> None:
    service = RagChatService(
        config=ChatProxyConfig(top_k=2, model_id="rag-test"),
        benchmark=BenchmarkData(
            name="fixture",
            dataset_id="fixture/test",
            queries=[],
            documents=[],
            qrels={},
        ),
        retriever=FakeRetriever(),
        llm=FakeLLM(),
        retrievers={"bm25": FakeRetriever()},
    )

    try:
        service.answer([{"role": "user", "content": "/img digit 7"}])
    except ValueError as exc:
        assert "Image mode is disabled" in str(exc)
    else:
        raise AssertionError("image mode should be disabled by default")

    try:
        service.answer([{"role": "user", "content": "/dict AMONIT"}])
    except ValueError as exc:
        assert "Dictionary mode is disabled" in str(exc)
    else:
        raise AssertionError("dictionary mode should be disabled by default")


def test_img_command_routes_to_image_retriever_without_llm_generation() -> None:
    llm = FakeLLM()
    text_retriever = FakeRetriever()
    image_retriever = FakeImageRetriever()
    service = RagChatService(
        config=ChatProxyConfig(top_k=2, model_id="rag-test", image_enabled=True),
        benchmark=BenchmarkData(
            name="fixture",
            dataset_id="fixture/test",
            queries=[],
            documents=[Document("cat-doc", "Cats purr and chase toys.", "Cats")],
            qrels={},
        ),
        retriever=text_retriever,
        llm=llm,
        retrievers={"bm25": text_retriever, "image-digits": image_retriever},
    )

    result = service.answer([{"role": "user", "content": "/img digit 7"}])

    assert result.response["choices"][0]["message"]["content"] == "Found 1 image result(s) for 'digit 7'."
    assert result.response["rag"]["retriever"] == "image-digits"
    assert result.response["rag"]["retrieval_metadata"]["command"] == "/img"
    assert result.response["rag"]["retrieval_metadata"]["image_top_k"] == 5
    assert image_retriever.seen_top_k == 5
    assert result.response["rag"]["retrieved"][0]["kind"] == "image"
    assert result.response["rag"]["retrieved"][0]["image_data_url"].startswith("data:image/svg+xml")
    assert llm.messages == []


def test_img_command_accepts_request_top_k_override() -> None:
    text_retriever = FakeRetriever()
    image_retriever = FakeImageRetriever()
    service = RagChatService(
        config=ChatProxyConfig(top_k=2, model_id="rag-test", image_enabled=True, image_top_k=5),
        benchmark=BenchmarkData(
            name="fixture",
            dataset_id="fixture/test",
            queries=[],
            documents=[],
            qrels={},
        ),
        retriever=text_retriever,
        llm=FakeLLM(),
        retrievers={"bm25": text_retriever, "image-digits": image_retriever},
    )

    result = service.answer([{"role": "user", "content": "/img digit 7"}], image_top_k=4)

    assert result.response["rag"]["retrieval_metadata"]["image_top_k"] == 4
    assert image_retriever.seen_top_k == 4


def test_image_mode_can_rewrite_query_with_selected_model() -> None:
    text_retriever = FakeRetriever()
    image_retriever = FakeImageRetriever()
    llm = FakeImageRewriteLLM()
    service = RagChatService(
        config=ChatProxyConfig(top_k=2, model_id="rag-test", image_enabled=True, image_top_k=5),
        benchmark=BenchmarkData(
            name="fixture",
            dataset_id="fixture/test",
            queries=[],
            documents=[],
            qrels={},
        ),
        retriever=text_retriever,
        llm=llm,
        retrievers={"bm25": text_retriever, "image-digits": image_retriever},
    )

    result = service.answer(
        [{"role": "user", "content": "show me seven"}],
        response_mode="image",
        image_rewrite=True,
        request_model="qwen/qwen3-32b",
    )

    assert result.response["model"] == "qwen/qwen3-32b"
    assert result.response["rag"]["retrieval_metadata"]["image_query"] == "digit 7"
    assert result.response["rag"]["retrieval_metadata"]["image_query_rewrite"] is True
    assert result.response["rag"]["retrieval_metadata"]["image_query_key_alias"] == "alias-img"
    assert llm.model == "qwen/qwen3-32b"


def test_text_image_mode_appends_image_results_after_text_retrieval() -> None:
    text_retriever = FakeRetriever()
    image_retriever = FakeImageRetriever()
    llm = FakeImageRewriteLLM()
    service = RagChatService(
        config=ChatProxyConfig(top_k=2, model_id="rag-test", image_enabled=True, image_top_k=5),
        benchmark=BenchmarkData(
            name="fixture",
            dataset_id="fixture/test",
            queries=[],
            documents=[],
            qrels={},
        ),
        retriever=text_retriever,
        llm=llm,
        retrievers={"bm25": text_retriever, "image-digits": image_retriever},
    )

    result = service.answer(
        [{"role": "user", "content": "What do cats do?"}],
        response_mode="text_image",
        image_top_k=3,
    )

    assert result.response["choices"][0]["message"]["content"] == "digit 7"
    assert [hit["doc_id"] for hit in result.response["rag"]["retrieved"]] == ["cat-doc", "image-7"]
    assert result.response["rag"]["retrieval_metadata"]["response_mode"] == "text_image"
    assert result.response["rag"]["retrieval_metadata"]["image_top_k"] == 3
    assert image_retriever.seen_top_k == 3


def test_keyword_match_uses_llm_keywords_before_search() -> None:
    keyword_retriever = FakeKeywordRetriever()
    llm = FakeKeywordLLM()
    service = RagChatService(
        config=ChatProxyConfig(top_k=2, model_id="rag-test"),
        benchmark=BenchmarkData(
            name="fixture",
            dataset_id="fixture/test",
            queries=[],
            documents=[],
            qrels={},
        ),
        retriever=keyword_retriever,
        llm=llm,
        retrievers={"keyword-match": keyword_retriever},
    )

    result = service.answer(
        [{"role": "user", "content": "giải thích BH1 bằng tiếng Việt"}],
        request_retriever="keyword-match",
        request_model="qwen/qwen3-32b",
    )

    assert keyword_retriever.seen_queries == ["BH1", "BH1 Bcl-2", "BH1 domain apoptosis"]
    assert result.response["choices"][0]["message"]["content"] == "BH1 is a Bcl-2 domain [bcl2-doc]."
    assert result.response["rag"]["retriever"] == "keyword-match"
    assert result.response["rag"]["retrieval_metadata"]["keyword_llm_calls"] == 1
    assert result.response["rag"]["retrieval_metadata"]["keyword_query_variants"] == [
        "BH1",
        "BH1 Bcl-2",
        "BH1 domain apoptosis",
    ]
    assert [source["doc_id"] for source in result.response["rag"]["retrieved"]] == ["bcl2-doc"]
    assert llm.calls == 2
    assert llm.model == "qwen/qwen3-32b"


def test_dict_command_routes_to_dictionary_retriever_with_rich_metadata() -> None:
    dictionary_retriever = FakeDictionaryRetriever()
    llm = FakeLLM()
    service = RagChatService(
        config=ChatProxyConfig(top_k=2, dictionary_enabled=True, dictionary_top_k=3, model_id="rag-test"),
        benchmark=BenchmarkData(
            name="fixture",
            dataset_id="fixture/test",
            queries=[],
            documents=[],
            qrels={},
        ),
        retriever=dictionary_retriever,
        llm=llm,
        retrievers={"dictionary-graph": dictionary_retriever},
        dictionary_status={"source": "artifact", "entry_count": 1},
    )

    result = service.answer([{"role": "user", "content": "/dict AMONIT"}], language="vi")

    assert dictionary_retriever.seen_query == "AMONIT"
    assert result.response["rag"]["retriever"] == "dictionary-graph"
    assert result.response["rag"]["retrieval_metadata"]["response_mode"] == "dictionary"
    assert result.response["rag"]["retrieval_metadata"]["language"] == "vi"
    assert "Required response language: Vietnamese" in llm.messages[0]["content"]
    assert result.response["rag"]["retrieval_metadata"]["dictionary_status"]["entry_count"] == 1
    assert result.response["rag"]["retrieved"][0]["rich_blocks"][0]["runs"][0]["bold"] is True
    assert result.response["choices"][0]["message"]["content"].startswith("Mục từ gốc [A-0001]:")

    lookup = service.lookup_dictionary("ĐKZ", top_k=3)

    assert dictionary_retriever.seen_query == "ĐKZ"
    assert lookup["object"] == "dictionary.lookup"
    assert lookup["retriever"] == "dictionary-graph"
    assert lookup["retrieved"][0]["rich_blocks"][0]["runs"][0]["bold"] is True


def test_uncited_zero_score_sources_are_hidden_but_cited_zero_score_sources_remain() -> None:
    class LowScoreRetriever(FakeRetriever):
        def search(self, query: Query, top_k: int) -> RetrievalResult:
            return RetrievalResult(
                query=query,
                hits=[
                    RetrievalHit(doc_id="cited-low", score=0.0, rank=1, title="Cited", text="Cited low score."),
                    RetrievalHit(doc_id="uncited-low", score=0.0, rank=2, title="Uncited", text="Uncited low score."),
                ],
                latency_s=0.01,
            )

    class CitingLLM(FakeLLM):
        def generate(self, *args, **kwargs) -> GenerationResult:
            result = super().generate(*args, **kwargs)
            result.answer = "The answer cites one low-score source [cited-low]."
            return result

    retriever = LowScoreRetriever()
    service = RagChatService(
        config=ChatProxyConfig(top_k=2, model_id="rag-test"),
        benchmark=BenchmarkData(
            name="fixture",
            dataset_id="fixture/test",
            queries=[],
            documents=[],
            qrels={},
        ),
        retriever=retriever,
        llm=CitingLLM(),
    )

    result = service.answer([{"role": "user", "content": "What do cats do?"}])

    assert [source["doc_id"] for source in result.response["rag"]["retrieved"]] == ["cited-low"]


def test_last_user_text_supports_openai_text_parts() -> None:
    assert (
        last_user_text(
            [
                {"role": "assistant", "content": "ignored"},
                {"role": "user", "content": [{"type": "text", "text": "hello"}]},
            ]
        )
        == "hello"
    )


def test_parse_chat_command_supports_img_alias() -> None:
    assert parse_chat_command("/web latest RAG news") == ("web", "latest RAG news")
    assert parse_chat_command("/search latest RAG news") == ("web", "latest RAG news")
    assert parse_chat_command("/img digit 3") == ("img", "digit 3")
    assert parse_chat_command("/image cats") == ("img", "cats")
    assert parse_chat_command("/dict AMONIT") == ("dict", "AMONIT")
    assert parse_chat_command("plain text") is None
