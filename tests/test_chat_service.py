from __future__ import annotations

from dataclasses import dataclass

import pytest

from rag_bench.chat_service import (
    ChatProxyConfig,
    ModelRoutedChatClient,
    RagChatService,
    _build_mimo_client,
    _mimo_base_url_for_key,
    _format_context,
    build_dictionary_rag_messages,
    extract_alias_evidence_from_hits,
    last_user_text,
    parse_chat_command,
)
from rag_bench.groq_client import GenerationResult
from rag_bench.dictionary_query_planner import plan_dictionary_query
from rag_bench.privacy import PrivacyRouteError
from rag_bench.secrets import ApiKey
from rag_bench.structured_evidence import StructuredEvidenceDoc, StructuredEvidenceIndex
from rag_bench.types import BenchmarkData, Document, Query, RetrievalHit, RetrievalResult


PUBLIC_METADATA = {"data_tier": "public", "doc_type": "synthetic"}


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
                    metadata=PUBLIC_METADATA,
                    data_tier="public",
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
                        "data_tier": "public",
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
                    metadata=PUBLIC_METADATA,
                    data_tier="public",
                ),
                RetrievalHit(
                    doc_id="noise-doc",
                    score=0.0,
                    rank=2,
                    title="Unrelated",
                    text="Unrelated document.",
                    metadata=PUBLIC_METADATA,
                    data_tier="public",
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
                        "data_tier": "semi_private",
                        "kind": "dictionary",
                        "headword": "AMONIT",
                        "raw_docx_text": "AMONIT, thuốc nổ phá.",
                        "rich_blocks": [{"type": "paragraph", "runs": [{"text": "AMONIT", "bold": True}]}],
                    },
                    data_tier="semi_private",
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


class CountingLLM(FakeLLM):
    def __init__(self) -> None:
        super().__init__()
        self.calls = 0

    def generate(self, *args, **kwargs) -> GenerationResult:
        self.calls += 1
        return super().generate(*args, **kwargs)


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
    assert result.response["rag"]["generation_model"] == "qwen/qwen3-32b"
    assert llm.model == "qwen/qwen3-32b"
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


def test_mimo_chat_client_uses_configured_auth_header() -> None:
    client = _build_mimo_client(
        ChatProxyConfig(mimo_auth_header="both"),
        [ApiKey(alias="mimo", value="test-mimo-key")],
    )

    openai_client = client.client_factory(ApiKey(alias="mimo", value="test-mimo-key"), 30)

    assert openai_client.chat.completions.auth_header == "both"


def test_mimo_chat_client_routes_payg_key_to_payg_base_url() -> None:
    config = ChatProxyConfig(
        mimo_base_url="https://token-plan-sgp.xiaomimimo.com/v1",
        mimo_payg_base_url="https://api.xiaomimimo.com/v1",
    )

    assert _mimo_base_url_for_key(config, ApiKey(alias="mimo", value="tp-token-plan")) == config.mimo_base_url
    assert _mimo_base_url_for_key(config, ApiKey(alias="mimo_payg", value="sk-payg")) == config.mimo_payg_base_url
    assert _mimo_base_url_for_key(config, ApiKey(alias="mimo", value="sk-payg")) == config.mimo_payg_base_url


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
        config=ChatProxyConfig(top_k=2, model_id="rag-test"),
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


def test_img_command_routes_to_image_retriever_without_llm_generation() -> None:
    llm = FakeLLM()
    text_retriever = FakeRetriever()
    image_retriever = FakeImageRetriever()
    service = RagChatService(
        config=ChatProxyConfig(top_k=2, model_id="rag-test"),
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
        config=ChatProxyConfig(top_k=2, model_id="rag-test", image_top_k=5),
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
        config=ChatProxyConfig(top_k=2, model_id="rag-test", image_top_k=5),
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
        config=ChatProxyConfig(top_k=2, model_id="rag-test", image_top_k=5),
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
        config=ChatProxyConfig(
            top_k=2,
            dictionary_top_k=3,
            model_id="rag-test",
            allow_external_semi_private=True,
        ),
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


def test_dictionary_lookup_uses_normalized_lookup_target_for_abbreviation_questions() -> None:
    class RecordingDictionaryRetriever:
        name = "dictionary-graph"
        build_time_s = 0.0

        def __init__(self) -> None:
            self.queries: list[str] = []

        def search(self, query: Query, top_k: int) -> RetrievalResult:
            self.queries.append(query.text)
            hits = [
                RetrievalHit(
                    doc_id="noise",
                    score=0.02,
                    rank=1,
                    title="Noise",
                    text="Synthetic weak lexical noise.",
                    metadata={"data_tier": "public", "kind": "dictionary"},
                    data_tier="public",
                )
            ]
            if query.text in {"PB", "XYZ"}:
                hits = [
                    RetrievalHit(
                        doc_id=f"dict:{query.text}",
                        score=1.2,
                        rank=1,
                        title=f"Synthetic {query.text}",
                        text=f"Synthetic {query.text} dictionary entry.",
                        metadata={
                            "data_tier": "semi_private",
                            "kind": "dictionary",
                            "headword": query.text,
                            "aliases": [query.text],
                            "dictionary_match_mode": "strict",
                        },
                        data_tier="semi_private",
                    )
                ]
            return RetrievalResult(query=query, hits=hits, latency_s=0.01, metadata={"kind": "dictionary"})

    dictionary_retriever = RecordingDictionaryRetriever()
    service = RagChatService(
        config=ChatProxyConfig(
            top_k=2,
            dictionary_top_k=3,
            model_id="rag-test",
            allow_external_semi_private=True,
        ),
        benchmark=BenchmarkData(name="fixture", dataset_id="fixture/test", queries=[], documents=[], qrels={}),
        retriever=dictionary_retriever,
        llm=CountingLLM(),
        retrievers={"dictionary-graph": dictionary_retriever},
        dictionary_status={"source": "artifact", "entry_count": 1},
    )

    lookup = service.lookup_dictionary("PB viết tắt cho gì?", top_k=3)

    assert dictionary_retriever.queries == ["PB viết tắt cho gì?", "PB"]
    assert lookup["retrieval_metadata"]["query_plan"]["target_terms"] == ["PB"]
    assert lookup["retrieved"][0]["doc_id"] == "dict:PB"

    dictionary_retriever.queries.clear()
    lookup = service.lookup_dictionary("vui lòng giải thích XYZ", top_k=3)

    assert dictionary_retriever.queries == ["vui lòng giải thích XYZ", "XYZ"]
    assert lookup["retrieval_metadata"]["query_plan"]["target_terms"] == ["XYZ"]
    assert lookup["retrieved"][0]["doc_id"] == "dict:XYZ"


def test_dictionary_mode_exposes_safe_query_plan_metadata_and_prompt_instructions() -> None:
    dictionary_retriever = FakeDictionaryRetriever()
    llm = CountingLLM()
    service = RagChatService(
        config=ChatProxyConfig(
            top_k=2,
            dictionary_top_k=3,
            model_id="rag-test",
            allow_external_semi_private=True,
        ),
        benchmark=BenchmarkData(name="fixture", dataset_id="fixture/test", queries=[], documents=[], qrels={}),
        retriever=dictionary_retriever,
        llm=llm,
        retrievers={"dictionary-graph": dictionary_retriever},
        dictionary_status={"source": "artifact", "entry_count": 1},
    )

    result = service.answer(
        [{"role": "user", "content": "/dict so sánh TERM_A và TERM_B"}],
        language="vi",
    )

    assert result.response["query_plan"]["intent"] == "comparison"
    assert result.response["query_plan"]["schema_gaps"] == []
    assert result.response["rag"]["retrieval_metadata"]["query_plan"]["target_terms"] == ["TERM_A", "TERM_B"]
    assert result.response["rag"]["retrieved"][0]["metadata"]["query_plan_intent"] == "comparison"
    assert "Compare only using the retrieved sources." in llm.messages[1]["content"]
    assert llm.calls == 1


def test_dictionary_mode_normalizes_short_acronym_definition_queries() -> None:
    class RecordingDictionaryRetriever(FakeDictionaryRetriever):
        def __init__(self) -> None:
            self.queries: list[str] = []

        def search(self, query: Query, top_k: int) -> RetrievalResult:
            self.queries.append(query.text)
            return super().search(query, top_k)

    dictionary_retriever = RecordingDictionaryRetriever()
    service = RagChatService(
        config=ChatProxyConfig(
            top_k=2,
            dictionary_top_k=3,
            model_id="rag-test",
            allow_external_semi_private=True,
        ),
        benchmark=BenchmarkData(name="fixture", dataset_id="fixture/test", queries=[], documents=[], qrels={}),
        retriever=dictionary_retriever,
        llm=CountingLLM(),
        retrievers={"dictionary-graph": dictionary_retriever},
        dictionary_status={"source": "artifact", "entry_count": 1},
    )

    result = service.answer([{"role": "user", "content": "/dict PB là gì?"}], language="vi")

    assert result.response["query_plan"]["intent"] == "definition"
    assert result.response["query_plan"]["target_terms"] == ["PB"]
    assert dictionary_retriever.queries == ["PB là gì?", "PB"]

    dictionary_retriever.queries.clear()
    result = service.answer([{"role": "user", "content": "/dict PB viết tắt cho gì?"}], language="vi")

    assert result.response["query_plan"]["intent"] == "definition"
    assert result.response["query_plan"]["target_terms"] == ["PB"]
    assert dictionary_retriever.queries == ["PB viết tắt cho gì?", "PB"]

    dictionary_retriever.queries.clear()
    result = service.answer([{"role": "user", "content": "/dict CVHL nghĩa là gì?"}], language="vi")

    assert result.response["query_plan"]["intent"] == "definition"
    assert result.response["query_plan"]["target_terms"] == ["CVHL"]
    assert dictionary_retriever.queries == ["CVHL nghĩa là gì?", "CVHL"]

    dictionary_retriever.queries.clear()
    result = service.answer([{"role": "user", "content": "/dict KHCN xuất hiện ở đâu?"}], language="vi")

    assert result.response["query_plan"]["intent"] == "definition"
    assert result.response["query_plan"]["target_terms"] == ["KHCN"]
    assert dictionary_retriever.queries == ["KHCN xuất hiện ở đâu?", "KHCN"]

    dictionary_retriever.queries.clear()
    result = service.answer([{"role": "user", "content": "/dict CTCC xuất hiện ở đâu?"}], language="vi")

    assert result.response["query_plan"]["intent"] == "definition"
    assert result.response["query_plan"]["target_terms"] == ["CTCC"]
    assert dictionary_retriever.queries == ["CTCC xuất hiện ở đâu?", "CTCC"]


def test_extract_alias_evidence_from_explicit_metadata() -> None:
    evidence = extract_alias_evidence_from_hits(
        [
            RetrievalHit(
                doc_id="TERM_A_ENTRY",
                score=1.0,
                rank=1,
                title="TERM_A",
                text="Synthetic alias entry.",
                metadata={"aliases": ["ALIAS_A", "ALIAS_B"], "has_alias_evidence": True},
            )
        ]
    )

    assert evidence.aliases == ["ALIAS_A", "ALIAS_B"]
    assert evidence.source_doc_ids == ["TERM_A_ENTRY"]
    assert evidence.evidence_count == 2
    assert evidence.has_explicit_alias_evidence is True


def test_extract_alias_evidence_ignores_related_category_and_concepts() -> None:
    evidence = extract_alias_evidence_from_hits(
        [
            RetrievalHit(
                doc_id="RELATED",
                score=1.0,
                rank=1,
                title="RELATED_X",
                text="Synthetic related entry.",
                metadata={"edge_type": "related_to", "label": "RELATED_X", "dictionary_relation": "related_to"},
            ),
            RetrievalHit(
                doc_id="CATEGORY",
                score=1.0,
                rank=2,
                title="CATEGORY_X",
                text="Synthetic category entry.",
                metadata={"edge_type": "in_category", "label": "CATEGORY_X", "dictionary_relation": "in_category"},
            ),
            RetrievalHit(
                doc_id="CONCEPT",
                score=1.0,
                rank=3,
                title="CONCEPT_X",
                text="Synthetic concept entry.",
                metadata={"concepts": ["CONCEPT_X"]},
            ),
        ]
    )

    assert evidence.aliases == []
    assert evidence.source_doc_ids == []
    assert evidence.evidence_count == 0
    assert evidence.has_explicit_alias_evidence is False


def test_extract_alias_evidence_dedupes_stably() -> None:
    evidence = extract_alias_evidence_from_hits(
        [
            RetrievalHit(
                doc_id="TERM_A_ENTRY",
                score=1.0,
                rank=1,
                title="TERM_A",
                text="Synthetic alias entry.",
                metadata={"aliases": ["ALIAS_A", "alias_a", "ALIAS_B"]},
            )
        ]
    )

    assert evidence.aliases == ["ALIAS_A", "ALIAS_B"]
    assert evidence.source_doc_ids == ["TERM_A_ENTRY"]
    assert evidence.evidence_count == 2


def test_extract_alias_evidence_from_dictionary_graph_edges() -> None:
    evidence = extract_alias_evidence_from_hits(
        [
            RetrievalHit(
                doc_id="TERM_A_ENTRY",
                score=1.0,
                rank=1,
                title="TERM_A",
                text="Synthetic graph edge entry.",
                metadata={
                    "headword": "TERM_A",
                    "dictionary_graph_edges": [
                        {"type": "has_alias", "target_label": "ALIAS_A", "confidence": 0.95},
                    ],
                },
            )
        ],
        target_terms=["TERM_A"],
    )

    assert evidence.aliases == ["ALIAS_A"]
    assert evidence.source_doc_ids == ["TERM_A_ENTRY"]
    assert evidence.evidence_count == 1


def test_extract_alias_evidence_from_single_has_alias_edge_metadata() -> None:
    evidence = extract_alias_evidence_from_hits(
        [
            RetrievalHit(
                doc_id="TERM_A_ENTRY",
                score=1.0,
                rank=1,
                title="TERM_A",
                text="Synthetic single edge entry.",
                metadata={
                    "headword": "TERM_A",
                    "edge_type": "has_alias",
                    "target_label": "ALIAS_A",
                },
            )
        ],
        target_terms=["TERM_A"],
    )

    assert evidence.aliases == ["ALIAS_A"]
    assert evidence.source_doc_ids == ["TERM_A_ENTRY"]
    assert evidence.evidence_count == 1


@pytest.mark.parametrize("edge_type", ["related_to", "see_also", "has_concept", "in_category", "is_a"])
def test_extract_alias_evidence_rejects_non_alias_graph_edge_types(edge_type: str) -> None:
    evidence = extract_alias_evidence_from_hits(
        [
            RetrievalHit(
                doc_id="TERM_A_ENTRY",
                score=1.0,
                rank=1,
                title="TERM_A",
                text="Synthetic non-alias edge entry.",
                metadata={
                    "headword": "TERM_A",
                    "graph_edges": [
                        {"type": edge_type, "target_label": "NOT_ALIAS", "confidence": 0.99},
                    ],
                },
            )
        ],
        target_terms=["TERM_A"],
    )

    assert evidence.aliases == []
    assert evidence.source_doc_ids == []
    assert evidence.evidence_count == 0


def test_extract_alias_evidence_rejects_low_confidence_has_alias_edge() -> None:
    evidence = extract_alias_evidence_from_hits(
        [
            RetrievalHit(
                doc_id="TERM_A_ENTRY",
                score=1.0,
                rank=1,
                title="TERM_A",
                text="Synthetic weak alias edge entry.",
                metadata={
                    "headword": "TERM_A",
                    "dictionary_graph_edges": [
                        {"type": "has_alias", "target_label": "WEAK_ALIAS", "confidence": 0.01},
                    ],
                },
            )
        ],
        target_terms=["TERM_A"],
    )

    assert evidence.aliases == []
    assert evidence.source_doc_ids == []
    assert evidence.evidence_count == 0


def test_extract_alias_evidence_filters_aliases_by_target_term() -> None:
    evidence = extract_alias_evidence_from_hits(
        [
            RetrievalHit(
                doc_id="TERM_A_ENTRY",
                score=1.0,
                rank=1,
                title="TERM_A",
                text="Synthetic alias entry.",
                metadata={"headword": "TERM_A", "aliases": ["ALIAS_A"]},
            ),
            RetrievalHit(
                doc_id="TERM_B_RELATED",
                score=0.9,
                rank=2,
                title="TERM_B",
                text="Synthetic related entry.",
                metadata={"headword": "TERM_B", "aliases": ["RELATED_ALIAS_B"], "dictionary_relation": "related_to"},
            ),
        ],
        target_terms=["TERM_A"],
    )

    assert evidence.aliases == ["ALIAS_A"]
    assert evidence.source_doc_ids == ["TERM_A_ENTRY"]
    assert evidence.evidence_count == 1


def test_extract_alias_evidence_from_has_alias_graph_path() -> None:
    evidence = extract_alias_evidence_from_hits(
        [
            RetrievalHit(
                doc_id="TERM_A_ENTRY",
                score=1.0,
                rank=1,
                title="TERM_A",
                text="Synthetic graph alias entry.",
                metadata={
                    "dictionary_relation": "has_alias",
                    "dictionary_graph_path": [
                        {"type": "entry", "id": "TERM_A_ENTRY", "label": "TERM_A"},
                        {"type": "relation", "id": "has_alias", "label": "has_alias"},
                        {"type": "alias", "id": "alias:term-a-alt", "label": "ALIAS_A"},
                    ],
                },
            )
        ]
    )

    assert evidence.aliases == ["ALIAS_A"]
    assert evidence.source_doc_ids == ["TERM_A_ENTRY"]
    assert evidence.evidence_count == 1


def test_dictionary_alias_prompt_includes_explicit_alias_block_when_used() -> None:
    messages = [{"role": "user", "content": "/dict TERM_A còn gọi là gì"}]
    plan = plan_dictionary_query("TERM_A còn gọi là gì")
    prompt_messages = build_dictionary_rag_messages(
        messages,
        [
            RetrievalHit(
                doc_id="TERM_A_ENTRY",
                score=1.0,
                rank=1,
                title="TERM_A",
                text="Synthetic alias entry.",
                metadata={"aliases": ["ALIAS_A"]},
            )
        ],
        query="TERM_A còn gọi là gì",
        max_context_chars=1000,
        history_messages=0,
        language="vi",
        query_plan=plan,
    )
    prompt = prompt_messages[1]["content"]

    assert "Explicit alias evidence:" in prompt
    assert "- ALIAS_A [TERM_A_ENTRY]" in prompt
    assert "Answer only from the explicit alias evidence block" in prompt
    assert "Do not treat related terms, concepts, categories, or see-also references as aliases." in prompt
    assert "Explain the term in the required response language" not in prompt


def test_dictionary_alias_mode_uses_direct_prompt_and_alias_metadata() -> None:
    class AliasDictionaryRetriever:
        name = "dictionary-graph"
        build_time_s = 0.0

        def search(self, query: Query, top_k: int) -> RetrievalResult:
            return RetrievalResult(
                query=query,
                hits=[
                    RetrievalHit(
                        doc_id="TERM_A_ENTRY",
                        score=1.2,
                        rank=1,
                        title="TERM_A",
                        text="TERM_A synthetic dictionary entry with explicit alternate name evidence.",
                        metadata={
                            "data_tier": "public",
                            "kind": "dictionary",
                            "headword": "TERM_A",
                            "aliases": ["TERM_A_ALT"],
                            "dictionary_match_mode": "strict",
                        },
                        data_tier="public",
                        doc_type="dictionary",
                    ),
                    RetrievalHit(
                        doc_id="TERM_A_RELATED",
                        score=1.0,
                        rank=2,
                        title="TERM_A_RELATED",
                        text="Synthetic related entry.",
                        metadata={
                            "data_tier": "public",
                            "kind": "dictionary",
                            "headword": "TERM_A_RELATED",
                            "aliases": ["RELATED_ALIAS_X"],
                            "dictionary_relation": "related_to",
                        },
                        data_tier="public",
                        doc_type="dictionary",
                    ),
                ],
                latency_s=0.01,
                metadata={"kind": "dictionary"},
            )

    retriever = AliasDictionaryRetriever()
    llm = CountingLLM()
    service = RagChatService(
        config=ChatProxyConfig(top_k=2, dictionary_top_k=3, model_id="rag-test"),
        benchmark=BenchmarkData(name="fixture", dataset_id="fixture/test", queries=[], documents=[], qrels={}),
        retriever=retriever,
        llm=llm,
        retrievers={"dictionary-graph": retriever},
    )

    result = service.answer([{"role": "user", "content": "/dict TERM_A còn gọi là gì"}])
    alias_metadata = result.response["rag"]["retrieval_metadata"]["alias_evidence"]
    retrieved_by_id = {source["doc_id"]: source for source in result.response["rag"]["retrieved"]}
    answer = result.response["choices"][0]["message"]["content"]

    assert llm.calls == 0
    assert "TERM_A_ALT" in answer
    assert "RELATED_ALIAS_X" not in answer
    assert "[TERM_A_ENTRY]" in answer
    assert "được ghi nhận trong nguồn" in answer
    assert "tên chính thức" not in answer
    assert "chắc chắn" not in answer
    assert "always called" not in answer
    assert result.response["query_plan"]["intent"] == "alias"
    assert result.response["query_plan"]["answer_style"] == "alias_direct"
    assert result.response["query_plan"]["requires_alias_evidence"] is True
    assert result.response["query_plan"]["alias_answer_mode"] == "deterministic_extractive"
    assert result.response["query_plan"]["alias_evidence_count"] == 1
    assert result.response["query_plan"]["alias_evidence_doc_count"] == 1
    assert result.response["rag"]["key_alias"] == "deterministic_alias"
    assert result.response["rag"]["retrieval_metadata"]["generator_provider"] == "deterministic_alias"
    assert alias_metadata["has_alias_evidence"] is True
    assert alias_metadata["has_explicit_alias_evidence"] is True
    assert alias_metadata["alias_evidence_count"] == 1
    assert alias_metadata["alias_evidence_doc_count"] == 1
    assert alias_metadata["alias_answer_mode"] == "deterministic_extractive"
    assert alias_metadata["alias_evidence_doc_ids"] == ["TERM_A_ENTRY"]
    assert retrieved_by_id["TERM_A_ENTRY"]["metadata"]["has_alias_evidence"] is True
    assert retrieved_by_id["TERM_A_ENTRY"]["metadata"]["query_plan_role"] == "alias_evidence"
    assert retrieved_by_id["TERM_A_RELATED"]["metadata"]["has_alias_evidence"] is False
    assert retrieved_by_id["TERM_A_RELATED"]["metadata"]["alias_evidence_count"] == 0


def test_dictionary_alias_mode_extracts_pb_like_has_alias_edge_only() -> None:
    class GraphEdgeAliasDictionaryRetriever:
        name = "dictionary-graph"
        build_time_s = 0.0

        def search(self, query: Query, top_k: int) -> RetrievalResult:
            return RetrievalResult(
                query=query,
                hits=[
                    RetrievalHit(
                        doc_id="TERM_A_ENTRY",
                        score=1.2,
                        rank=1,
                        title="TERM_A",
                        text="TERM_A synthetic graph edge entry.",
                        metadata={
                            "data_tier": "public",
                            "kind": "dictionary",
                            "headword": "TERM_A",
                            "dictionary_graph_edges": [
                                {"type": "has_alias", "target_label": "ALIAS_A", "confidence": 0.95},
                                {"type": "related_to", "target_label": "RELATED_A", "confidence": 0.99},
                                {"type": "in_category", "target_label": "CATEGORY_A", "confidence": 0.99},
                            ],
                            "dictionary_match_mode": "strict",
                        },
                        data_tier="public",
                        doc_type="dictionary",
                    ),
                ],
                latency_s=0.01,
                metadata={"kind": "dictionary"},
            )

    retriever = GraphEdgeAliasDictionaryRetriever()
    llm = CountingLLM()
    service = RagChatService(
        config=ChatProxyConfig(top_k=2, dictionary_top_k=3, model_id="rag-test"),
        benchmark=BenchmarkData(name="fixture", dataset_id="fixture/test", queries=[], documents=[], qrels={}),
        retriever=retriever,
        llm=llm,
        retrievers={"dictionary-graph": retriever},
    )

    result = service.answer([{"role": "user", "content": "/dict TERM_A còn gọi là gì"}])
    answer = result.response["choices"][0]["message"]["content"]

    assert llm.calls == 0
    assert "ALIAS_A" in answer
    assert "RELATED_A" not in answer
    assert "CATEGORY_A" not in answer
    assert result.response["query_plan"]["intent"] == "alias"
    assert result.response["query_plan"]["alias_answer_mode"] == "deterministic_extractive"
    assert result.response["rag"]["retrieval_metadata"]["alias_evidence"]["alias_evidence_count"] == 1


def test_dictionary_alias_extractive_answer_can_be_disabled() -> None:
    class AliasDictionaryRetriever:
        name = "dictionary-graph"
        build_time_s = 0.0

        def search(self, query: Query, top_k: int) -> RetrievalResult:
            return RetrievalResult(
                query=query,
                hits=[
                    RetrievalHit(
                        doc_id="TERM_A_ENTRY",
                        score=1.2,
                        rank=1,
                        title="TERM_A",
                        text="TERM_A synthetic dictionary entry.",
                        metadata={
                            "data_tier": "public",
                            "kind": "dictionary",
                            "headword": "TERM_A",
                            "aliases": ["ALIAS_A"],
                            "dictionary_match_mode": "strict",
                        },
                        data_tier="public",
                        doc_type="dictionary",
                    ),
                ],
                latency_s=0.01,
                metadata={"kind": "dictionary"},
            )

    retriever = AliasDictionaryRetriever()
    llm = CountingLLM()
    service = RagChatService(
        config=ChatProxyConfig(
            top_k=2,
            dictionary_top_k=3,
            model_id="rag-test",
            enable_alias_extractive_answer=False,
        ),
        benchmark=BenchmarkData(name="fixture", dataset_id="fixture/test", queries=[], documents=[], qrels={}),
        retriever=retriever,
        llm=llm,
        retrievers={"dictionary-graph": retriever},
    )

    result = service.answer([{"role": "user", "content": "/dict TERM_A còn gọi là gì"}])

    assert llm.calls == 1
    assert result.response["query_plan"]["intent"] == "alias"
    assert result.response["query_plan"]["alias_answer_mode"] == "llm_prompt"
    assert result.response["rag"]["retrieval_metadata"]["alias_answer_mode"] == "llm_prompt"
    assert result.response["rag"]["key_alias"] == "alias-a"
    assert "Explicit alias evidence:" in llm.messages[1]["content"]


@pytest.mark.parametrize(
    ("query", "expected_intent"),
    [
        ("/dict TERM_A là gì", "definition"),
        ("/dict so sánh TERM_A và TERM_B", "comparison"),
        ("/dict quy trình TERM_A", "procedure"),
        ("/dict trường hợp này áp dụng TERM_A không", "rule_application"),
        ("/dict case tương tự cho TERM_A là gì", "case_based"),
    ],
)
def test_dictionary_alias_extractive_answer_is_alias_intent_only(query: str, expected_intent: str) -> None:
    class AliasMetadataDictionaryRetriever:
        name = "dictionary-graph"
        build_time_s = 0.0

        def search(self, query: Query, top_k: int) -> RetrievalResult:
            return RetrievalResult(
                query=query,
                hits=[
                    RetrievalHit(
                        doc_id="TERM_A_ENTRY",
                        score=1.2,
                        rank=1,
                        title="TERM_A",
                        text="TERM_A synthetic dictionary entry.",
                        metadata={
                            "data_tier": "public",
                            "kind": "dictionary",
                            "headword": "TERM_A",
                            "aliases": ["ALIAS_A"],
                            "dictionary_graph_edges": [
                                {"type": "has_alias", "target_label": "ALIAS_A", "confidence": 0.95},
                            ],
                            "dictionary_match_mode": "strict",
                        },
                        data_tier="public",
                        doc_type="dictionary",
                    ),
                    RetrievalHit(
                        doc_id="TERM_B_ENTRY",
                        score=1.0,
                        rank=2,
                        title="TERM_B",
                        text="TERM_B synthetic dictionary entry.",
                        metadata={"data_tier": "public", "kind": "dictionary", "headword": "TERM_B"},
                        data_tier="public",
                        doc_type="dictionary",
                    ),
                ],
                latency_s=0.01,
                metadata={"kind": "dictionary"},
            )

    retriever = AliasMetadataDictionaryRetriever()
    llm = CountingLLM()
    service = RagChatService(
        config=ChatProxyConfig(top_k=2, dictionary_top_k=3, model_id="rag-test"),
        benchmark=BenchmarkData(name="fixture", dataset_id="fixture/test", queries=[], documents=[], qrels={}),
        retriever=retriever,
        llm=llm,
        retrievers={"dictionary-graph": retriever},
    )

    result = service.answer([{"role": "user", "content": query}])

    assert llm.calls == 1
    assert result.response["query_plan"]["intent"] == expected_intent
    assert "alias_answer_mode" not in result.response["query_plan"]
    assert "alias_answer_mode" not in result.response["rag"]["retrieval_metadata"]
    assert result.response["rag"]["key_alias"] == "alias-a"


def test_dictionary_alias_mode_marks_missing_alias_evidence() -> None:
    class NoAliasDictionaryRetriever:
        name = "dictionary-graph"
        build_time_s = 0.0

        def search(self, query: Query, top_k: int) -> RetrievalResult:
            return RetrievalResult(
                query=query,
                hits=[
                    RetrievalHit(
                        doc_id="TERM_A_ENTRY",
                        score=1.2,
                        rank=1,
                        title="TERM_A",
                        text="TERM_A synthetic definition only.",
                        metadata={
                            "data_tier": "public",
                            "kind": "dictionary",
                            "headword": "TERM_A",
                            "dictionary_match_mode": "strict",
                        },
                        data_tier="public",
                        doc_type="dictionary",
                    ),
                    RetrievalHit(
                        doc_id="TERM_A_CATEGORY",
                        score=1.0,
                        rank=2,
                        title="TERM_A_CATEGORY",
                        text="Synthetic category entry.",
                        metadata={
                            "data_tier": "public",
                            "kind": "dictionary",
                            "headword": "TERM_A_CATEGORY",
                            "dictionary_relation": "in_category",
                        },
                        data_tier="public",
                        doc_type="dictionary",
                    ),
                ],
                latency_s=0.01,
                metadata={"kind": "dictionary"},
            )

    retriever = NoAliasDictionaryRetriever()
    llm = CountingLLM()
    service = RagChatService(
        config=ChatProxyConfig(top_k=2, dictionary_top_k=3, model_id="rag-test"),
        benchmark=BenchmarkData(name="fixture", dataset_id="fixture/test", queries=[], documents=[], qrels={}),
        retriever=retriever,
        llm=llm,
        retrievers={"dictionary-graph": retriever},
    )

    result = service.answer([{"role": "user", "content": "/dict tên khác của TERM_A là gì"}])
    alias_metadata = result.response["rag"]["retrieval_metadata"]["alias_evidence"]
    answer = result.response["choices"][0]["message"]["content"]

    assert llm.calls == 0
    assert result.response["query_plan"]["intent"] == "alias"
    assert result.response["query_plan"]["alias_answer_mode"] == "deterministic_no_alias"
    assert result.response["query_plan"]["alias_evidence_count"] == 0
    assert result.response["query_plan"]["alias_evidence_doc_count"] == 0
    assert alias_metadata["has_alias_evidence"] is False
    assert alias_metadata["has_explicit_alias_evidence"] is False
    assert alias_metadata["alias_evidence_count"] == 0
    assert alias_metadata["alias_answer_mode"] == "deterministic_no_alias"
    assert alias_metadata["alias_evidence_doc_ids"] == []
    assert "Không tìm thấy tên gọi khác/alias được đánh dấu rõ ràng" in answer
    assert "không có alias" not in answer.lower()
    assert "không tồn tại tên gọi khác" not in answer.lower()
    assert "there is no alias" not in answer.lower()
    assert "CATEGORY_X" not in answer
    assert "RELATED_X" not in answer
    assert result.response["rag"]["retrieved"][0]["metadata"]["has_alias_evidence"] is False


def test_dictionary_planner_public_path_preserves_external_generation() -> None:
    class PublicDictionaryRetriever(FakeDictionaryRetriever):
        def search(self, query: Query, top_k: int) -> RetrievalResult:
            result = super().search(query, top_k)
            hit = result.hits[0]
            result.hits[0] = RetrievalHit(
                doc_id=hit.doc_id,
                score=hit.score,
                rank=hit.rank,
                title=hit.title,
                text=hit.text,
                metadata={**hit.metadata, "data_tier": "public"},
                data_tier="public",
                doc_type="dictionary",
            )
            return result

    dictionary_retriever = PublicDictionaryRetriever()
    llm = CountingLLM()
    service = RagChatService(
        config=ChatProxyConfig(top_k=2, dictionary_top_k=3, model_id="rag-test"),
        benchmark=BenchmarkData(name="fixture", dataset_id="fixture/test", queries=[], documents=[], qrels={}),
        retriever=dictionary_retriever,
        llm=llm,
        retrievers={"dictionary-graph": dictionary_retriever},
    )

    result = service.answer([{"role": "user", "content": "/dict TERM_A là gì"}])

    assert result.response["query_plan"]["intent"] == "definition"
    assert result.response["privacy"]["provider_allowed"] is True
    assert llm.calls == 1


def test_dictionary_planner_does_not_bypass_private_taint_guard() -> None:
    class PrivateDictionaryRetriever(FakeDictionaryRetriever):
        def search(self, query: Query, top_k: int) -> RetrievalResult:
            result = super().search(query, top_k)
            hit = result.hits[0]
            result.hits[0] = RetrievalHit(
                doc_id=hit.doc_id,
                score=hit.score,
                rank=hit.rank,
                title=hit.title,
                text="private synthetic dictionary context",
                metadata={**hit.metadata, "data_tier": "private", "raw_docx_text": "private synthetic dictionary context"},
                data_tier="private",
                doc_type="dictionary",
            )
            return result

    dictionary_retriever = PrivateDictionaryRetriever()
    llm = CountingLLM()
    service = RagChatService(
        config=ChatProxyConfig(top_k=2, dictionary_top_k=3, model_id="rag-test"),
        benchmark=BenchmarkData(name="fixture", dataset_id="fixture/test", queries=[], documents=[], qrels={}),
        retriever=dictionary_retriever,
        llm=llm,
        retrievers={"dictionary-graph": dictionary_retriever},
    )

    with pytest.raises(PrivacyRouteError) as error:
        service.answer([{"role": "user", "content": "/dict TERM_A là gì"}], session_id="private-dict")

    assert error.value.decision.reason == "private_taint_blocks_external_saas_backend"
    assert llm.calls == 0


def test_dictionary_alias_mode_private_hit_still_blocks_external_generation() -> None:
    class PrivateAliasDictionaryRetriever(FakeDictionaryRetriever):
        def search(self, query: Query, top_k: int) -> RetrievalResult:
            return RetrievalResult(
                query=query,
                hits=[
                    RetrievalHit(
                        doc_id="PRIVATE_ALIAS",
                        score=1.2,
                        rank=1,
                        title="TERM_PRIVATE",
                        text="private synthetic alias context",
                        metadata={
                            "data_tier": "private",
                            "kind": "dictionary",
                            "headword": "TERM_PRIVATE",
                            "aliases": ["TERM_PRIVATE_ALT"],
                            "raw_docx_text": "private synthetic alias context",
                        },
                        data_tier="private",
                        doc_type="dictionary",
                    )
                ],
                latency_s=0.01,
                metadata={"kind": "dictionary"},
            )

    retriever = PrivateAliasDictionaryRetriever()
    llm = CountingLLM()
    service = RagChatService(
        config=ChatProxyConfig(top_k=2, dictionary_top_k=3, model_id="rag-test"),
        benchmark=BenchmarkData(name="fixture", dataset_id="fixture/test", queries=[], documents=[], qrels={}),
        retriever=retriever,
        llm=llm,
        retrievers={"dictionary-graph": retriever},
    )

    with pytest.raises(PrivacyRouteError) as error:
        service.answer([{"role": "user", "content": "/dict TERM_PRIVATE còn gọi là gì"}], session_id="private-alias")

    assert error.value.decision.reason == "private_taint_blocks_external_saas_backend"
    assert llm.calls == 0


def test_eval_style_private_user_tier_blocks_external_generation_even_with_public_hits() -> None:
    class PublicDictionaryRetriever(FakeDictionaryRetriever):
        def search(self, query: Query, top_k: int) -> RetrievalResult:
            result = super().search(query, top_k)
            hit = result.hits[0]
            result.hits[0] = RetrievalHit(
                doc_id=hit.doc_id,
                score=hit.score,
                rank=hit.rank,
                title=hit.title,
                text=hit.text,
                metadata={**hit.metadata, "data_tier": "public"},
                data_tier="public",
                doc_type="dictionary",
            )
            return result

    retriever = PublicDictionaryRetriever()
    llm = CountingLLM()
    service = RagChatService(
        config=ChatProxyConfig(top_k=2, dictionary_top_k=3, model_id="rag-test"),
        benchmark=BenchmarkData(name="fixture", dataset_id="fixture/test", queries=[], documents=[], qrels={}),
        retriever=retriever,
        llm=llm,
        retrievers={"dictionary-graph": retriever},
    )

    with pytest.raises(PrivacyRouteError) as error:
        service.answer(
            [{"role": "user", "content": "/dict TERM_A là gì", "data_tier": "private"}],
            session_id="eval-private-item",
        )

    assert error.value.decision.reason == "private_taint_blocks_external_saas_backend"
    assert llm.calls == 0


def test_dictionary_mode_uses_public_structured_procedure_evidence() -> None:
    class PublicDictionaryRetriever(FakeDictionaryRetriever):
        def search(self, query: Query, top_k: int) -> RetrievalResult:
            result = super().search(query, top_k)
            hit = result.hits[0]
            result.hits[0] = RetrievalHit(
                doc_id=hit.doc_id,
                score=hit.score,
                rank=hit.rank,
                title=hit.title,
                text=hit.text,
                metadata={**hit.metadata, "data_tier": "public"},
                data_tier="public",
                doc_type="dictionary",
            )
            return result

    structured_index = StructuredEvidenceIndex(
        [
            StructuredEvidenceDoc.from_mapping(
                {
                    "doc_id": "PROC_X",
                    "doc_type": "procedure",
                    "title": "Procedure X",
                    "data_tier": "public",
                    "linked_terms": ["TERM_A"],
                    "steps": ["STEP_1", "STEP_2"],
                }
            )
        ]
    )
    llm = CountingLLM()
    service = RagChatService(
        config=ChatProxyConfig(top_k=2, dictionary_top_k=3, model_id="rag-test"),
        benchmark=BenchmarkData(name="fixture", dataset_id="fixture/test", queries=[], documents=[], qrels={}),
        retriever=PublicDictionaryRetriever(),
        llm=llm,
        retrievers={"dictionary-graph": PublicDictionaryRetriever()},
        structured_evidence_index=structured_index,
    )

    result = service.answer([{"role": "user", "content": "/dict quy trình xử lý TERM_A là gì"}])

    assert result.response["query_plan"]["intent"] == "procedure"
    assert result.response["query_plan"]["schema_gaps"] == []
    assert result.response["query_plan"]["structured_evidence"]["matched_doc_count"] == 1
    assert result.response["rag"]["retrieval_metadata"]["structured_evidence"]["matched_doc_types"] == ["procedure"]
    assert "Present steps only if they are supported" in llm.messages[1]["content"]
    assert result.response["privacy"]["provider_allowed"] is True
    assert llm.calls == 1


def test_dictionary_mode_uses_rule_evidence_with_safe_counts() -> None:
    class PublicDictionaryRetriever(FakeDictionaryRetriever):
        def search(self, query: Query, top_k: int) -> RetrievalResult:
            result = super().search(query, top_k)
            hit = result.hits[0]
            result.hits[0] = RetrievalHit(
                doc_id=hit.doc_id,
                score=hit.score,
                rank=hit.rank,
                title=hit.title,
                text=hit.text,
                metadata={**hit.metadata, "data_tier": "public"},
                data_tier="public",
                doc_type="dictionary",
            )
            return result

    structured_index = StructuredEvidenceIndex(
        [
            StructuredEvidenceDoc.from_mapping(
                {
                    "doc_id": "RULE_X",
                    "doc_type": "rule",
                    "title": "Rule X",
                    "data_tier": "public",
                    "linked_terms": ["TERM_A"],
                    "conditions": ["CONDITION_A"],
                    "exceptions": ["EXCEPTION_B"],
                }
            )
        ]
    )
    llm = CountingLLM()
    retriever = PublicDictionaryRetriever()
    service = RagChatService(
        config=ChatProxyConfig(top_k=2, dictionary_top_k=3, model_id="rag-test"),
        benchmark=BenchmarkData(name="fixture", dataset_id="fixture/test", queries=[], documents=[], qrels={}),
        retriever=retriever,
        llm=llm,
        retrievers={"dictionary-graph": retriever},
        structured_evidence_index=structured_index,
    )

    result = service.answer([{"role": "user", "content": "/dict trường hợp này áp dụng TERM_A không"}])
    structured_sources = [
        source for source in result.response["rag"]["retrieved"]
        if source["metadata"].get("structured_evidence")
    ]

    assert result.response["query_plan"]["intent"] == "rule_application"
    assert result.response["query_plan"]["schema_gaps"] == []
    assert "Identify conditions and exceptions from retrieved rule sources." in llm.messages[1]["content"]
    assert structured_sources[0]["metadata"]["condition_count"] == 1
    assert structured_sources[0]["metadata"]["exception_count"] == 1


def test_unrelated_structured_procedure_does_not_clear_gap_or_enter_prompt() -> None:
    class PublicDictionaryRetriever(FakeDictionaryRetriever):
        def search(self, query: Query, top_k: int) -> RetrievalResult:
            result = super().search(query, top_k)
            hit = result.hits[0]
            result.hits[0] = RetrievalHit(
                doc_id=hit.doc_id,
                score=hit.score,
                rank=hit.rank,
                title=hit.title,
                text=hit.text,
                metadata={**hit.metadata, "data_tier": "public"},
                data_tier="public",
                doc_type="dictionary",
            )
            return result

    structured_index = StructuredEvidenceIndex(
        [
            StructuredEvidenceDoc.from_mapping(
                {
                    "doc_id": "PROC_B",
                    "doc_type": "procedure",
                    "data_tier": "public",
                    "linked_terms": ["TERM_B"],
                    "steps": ["STEP_B1"],
                }
            )
        ]
    )
    llm = CountingLLM()
    retriever = PublicDictionaryRetriever()
    service = RagChatService(
        config=ChatProxyConfig(top_k=2, dictionary_top_k=3, model_id="rag-test"),
        benchmark=BenchmarkData(name="fixture", dataset_id="fixture/test", queries=[], documents=[], qrels={}),
        retriever=retriever,
        llm=llm,
        retrievers={"dictionary-graph": retriever},
        structured_evidence_index=structured_index,
    )

    result = service.answer([{"role": "user", "content": "/dict quy trình xử lý TERM_A là gì"}])

    assert "procedure_schema_not_implemented" in result.response["query_plan"]["schema_gaps"]
    assert result.response["query_plan"]["structured_evidence"]["matched_doc_count"] == 0
    assert result.response["rag"]["retrieval_metadata"]["structured_evidence"]["matched_doc_count"] == 0
    assert "Do not invent steps" in llm.messages[1]["content"]
    assert "STEP_B1" not in llm.messages[1]["content"]


def test_unrelated_structured_rule_does_not_clear_gap_or_expose_conditions() -> None:
    class PublicDictionaryRetriever(FakeDictionaryRetriever):
        def search(self, query: Query, top_k: int) -> RetrievalResult:
            result = super().search(query, top_k)
            hit = result.hits[0]
            result.hits[0] = RetrievalHit(
                doc_id=hit.doc_id,
                score=hit.score,
                rank=hit.rank,
                title=hit.title,
                text=hit.text,
                metadata={**hit.metadata, "data_tier": "public"},
                data_tier="public",
                doc_type="dictionary",
            )
            return result

    structured_index = StructuredEvidenceIndex(
        [
            StructuredEvidenceDoc.from_mapping(
                {
                    "doc_id": "RULE_B",
                    "doc_type": "rule",
                    "data_tier": "public",
                    "linked_terms": ["TERM_B"],
                    "conditions": ["COND_B"],
                }
            )
        ]
    )
    llm = CountingLLM()
    retriever = PublicDictionaryRetriever()
    service = RagChatService(
        config=ChatProxyConfig(top_k=2, dictionary_top_k=3, model_id="rag-test"),
        benchmark=BenchmarkData(name="fixture", dataset_id="fixture/test", queries=[], documents=[], qrels={}),
        retriever=retriever,
        llm=llm,
        retrievers={"dictionary-graph": retriever},
        structured_evidence_index=structured_index,
    )

    result = service.answer([{"role": "user", "content": "/dict trường hợp này áp dụng TERM_A không"}])

    assert "rule_schema_not_implemented" in result.response["query_plan"]["schema_gaps"]
    assert result.response["query_plan"]["structured_evidence"]["matched_doc_count"] == 0
    assert "Do not invent steps, rules, exceptions, or cases." in llm.messages[1]["content"]
    assert "COND_B" not in llm.messages[1]["content"]


def test_unrelated_structured_case_does_not_clear_gap_or_enter_prompt() -> None:
    class PublicDictionaryRetriever(FakeDictionaryRetriever):
        def search(self, query: Query, top_k: int) -> RetrievalResult:
            result = super().search(query, top_k)
            hit = result.hits[0]
            result.hits[0] = RetrievalHit(
                doc_id=hit.doc_id,
                score=hit.score,
                rank=hit.rank,
                title=hit.title,
                text=hit.text,
                metadata={**hit.metadata, "data_tier": "public"},
                data_tier="public",
                doc_type="dictionary",
            )
            return result

    structured_index = StructuredEvidenceIndex(
        [
            StructuredEvidenceDoc.from_mapping(
                {
                    "doc_id": "CASE_B",
                    "doc_type": "case",
                    "data_tier": "public",
                    "linked_terms": ["TERM_B"],
                    "situation": "SITUATION_B",
                    "outcome": "OUTCOME_B",
                }
            )
        ]
    )
    llm = CountingLLM()
    retriever = PublicDictionaryRetriever()
    service = RagChatService(
        config=ChatProxyConfig(top_k=2, dictionary_top_k=3, model_id="rag-test"),
        benchmark=BenchmarkData(name="fixture", dataset_id="fixture/test", queries=[], documents=[], qrels={}),
        retriever=retriever,
        llm=llm,
        retrievers={"dictionary-graph": retriever},
        structured_evidence_index=structured_index,
    )

    result = service.answer([{"role": "user", "content": "/dict case tương tự cho TERM_A là gì"}])

    assert result.response["query_plan"]["intent"] == "case_based"
    assert "case_schema_not_implemented" in result.response["query_plan"]["schema_gaps"]
    assert result.response["query_plan"]["structured_evidence"]["matched_doc_count"] == 0
    assert "SITUATION_B" not in llm.messages[1]["content"]
    assert "OUTCOME_B" not in llm.messages[1]["content"]


def test_private_structured_evidence_blocks_external_generation() -> None:
    class PublicDictionaryRetriever(FakeDictionaryRetriever):
        def search(self, query: Query, top_k: int) -> RetrievalResult:
            result = super().search(query, top_k)
            hit = result.hits[0]
            result.hits[0] = RetrievalHit(
                doc_id=hit.doc_id,
                score=hit.score,
                rank=hit.rank,
                title=hit.title,
                text=hit.text,
                metadata={**hit.metadata, "data_tier": "public"},
                data_tier="public",
                doc_type="dictionary",
            )
            return result

    structured_index = StructuredEvidenceIndex(
        [
            StructuredEvidenceDoc.from_mapping(
                {
                    "doc_id": "PROC_SECRET",
                    "doc_type": "procedure",
                    "linked_terms": ["TERM_A"],
                    "steps": ["SECRET_STEP"],
                }
            )
        ]
    )
    retriever = PublicDictionaryRetriever()
    llm = CountingLLM()
    service = RagChatService(
        config=ChatProxyConfig(top_k=2, dictionary_top_k=3, model_id="rag-test"),
        benchmark=BenchmarkData(name="fixture", dataset_id="fixture/test", queries=[], documents=[], qrels={}),
        retriever=retriever,
        llm=llm,
        retrievers={"dictionary-graph": retriever},
        structured_evidence_index=structured_index,
    )

    with pytest.raises(PrivacyRouteError) as error:
        service.answer(
            [{"role": "user", "content": "/dict quy trình xử lý TERM_A là gì"}],
            session_id="private-structured",
        )

    assert error.value.decision.reason == "private_taint_blocks_external_saas_backend"
    assert service.privacy_states["private-structured"].max_seen_tier.value == "private"
    assert llm.calls == 0


def test_text_mode_adds_dictionary_fallback_for_short_terms() -> None:
    class WeakTextRetriever:
        name = "bm25"
        build_time_s = 0.0

        def search(self, query: Query, top_k: int) -> RetrievalResult:
            assert query.text == "pháo binh"
            return RetrievalResult(
                query=query,
                hits=[
                    RetrievalHit(
                        doc_id="bench-noise",
                        score=0.0,
                        rank=1,
                        title="Noise",
                        text="No useful context.",
                        metadata=PUBLIC_METADATA,
                        data_tier="public",
                    )
                ],
                latency_s=0.02,
            )

    class DictionaryFallbackRetriever:
        name = "dictionary-graph"
        build_time_s = 0.0

        def search(self, query: Query, top_k: int) -> RetrievalResult:
            assert query.text == "pháo binh"
            assert top_k == 2
            return RetrievalResult(
                query=query,
                hits=[
                    RetrievalHit(
                        doc_id="base:P-0023",
                        score=2.0,
                        rank=1,
                        title="PHÁO BINH",
                        text="PHÁO BINH, lực lượng tác chiến.",
                        metadata={
                            "data_tier": "semi_private",
                            "kind": "dictionary",
                            "headword": "PHÁO BINH",
                            "dictionary_direct_score": 1.2,
                            "dictionary_match_mode": "strict",
                        },
                        data_tier="semi_private",
                    )
                ],
                latency_s=0.01,
                metadata={"kind": "dictionary"},
            )

    text_retriever = WeakTextRetriever()
    dictionary_retriever = DictionaryFallbackRetriever()
    llm = FakeLLM()
    service = RagChatService(
        config=ChatProxyConfig(
            top_k=2,
            dictionary_top_k=3,
            model_id="rag-test",
            allow_external_semi_private=True,
        ),
        benchmark=BenchmarkData(name="fixture", dataset_id="fixture/test", queries=[], documents=[], qrels={}),
        retriever=text_retriever,
        llm=llm,
        retrievers={"bm25": text_retriever, "dictionary-graph": dictionary_retriever},
        dictionary_status={"source": "artifact", "entry_count": 1},
    )

    result = service.answer([{"role": "user", "content": "pháo binh"}], response_mode="text")

    assert result.response["rag"]["retriever"] == "bm25"
    assert result.response["rag"]["retrieval_metadata"]["dictionary_fallback"] is True
    assert result.response["rag"]["retrieved"][0]["doc_id"] == "base:P-0023"
    assert "PHÁO BINH, lực lượng tác chiến." in llm.messages[1]["content"]


def test_text_mode_dictionary_fallback_uses_normalized_lookup_target_for_mentions() -> None:
    class WeakTextRetriever:
        name = "bm25"
        build_time_s = 0.0

        def search(self, query: Query, top_k: int) -> RetrievalResult:
            assert query.text in {
                "PB viết tắt cho gì?",
                "pbviettatcuagi",
                "KHCN xuất hiện ở đâu?",
                "khcnxuathienodau",
                "CTCC xuất hiện ở đâu?",
                "ctccxuathienodau",
                "xyzxuathienodau",
            }
            return RetrievalResult(
                query=query,
                hits=[
                    RetrievalHit(
                        doc_id="bench-noise",
                        score=0.0,
                        rank=1,
                        title="Noise",
                        text="No useful context.",
                        metadata=PUBLIC_METADATA,
                        data_tier="public",
                    )
                ],
                latency_s=0.02,
            )

    class DictionaryFallbackRetriever:
        name = "dictionary-graph"
        build_time_s = 0.0

        def __init__(self) -> None:
            self.queries: list[str] = []

        def search(self, query: Query, top_k: int) -> RetrievalResult:
            self.queries.append(query.text)
            hits = []
            if query.text in {"PB", "KHCN", "CTCC", "XYZ"}:
                hits = [
                    RetrievalHit(
                        doc_id="dict-mention",
                        score=0.72,
                        rank=1,
                        title="Synthetic related entry",
                        text=f"Synthetic entry mentioning {query.text}.",
                        metadata={
                            "data_tier": "semi_private",
                            "kind": "dictionary",
                            "dictionary_match_mode": "lexical",
                            "query_highlights": [query.text],
                        },
                        data_tier="semi_private",
                    )
                ]
            return RetrievalResult(query=query, hits=hits, latency_s=0.01, metadata={"kind": "dictionary"})

    text_retriever = WeakTextRetriever()
    dictionary_retriever = DictionaryFallbackRetriever()
    llm = FakeLLM()
    service = RagChatService(
        config=ChatProxyConfig(
            top_k=2,
            dictionary_top_k=3,
            model_id="rag-test",
            allow_external_semi_private=True,
        ),
        benchmark=BenchmarkData(name="fixture", dataset_id="fixture/test", queries=[], documents=[], qrels={}),
        retriever=text_retriever,
        llm=llm,
        retrievers={"bm25": text_retriever, "dictionary-graph": dictionary_retriever},
        dictionary_status={"source": "artifact", "entry_count": 1},
    )

    result = service.answer([{"role": "user", "content": "KHCN xuất hiện ở đâu?"}], response_mode="text")

    assert dictionary_retriever.queries == ["KHCN xuất hiện ở đâu?", "KHCN"]
    assert result.response["rag"]["retrieval_metadata"]["dictionary_fallback"] is True
    assert result.response["rag"]["retrieval_metadata"]["dictionary_fallback_metadata"]["query_plan"]["target_terms"] == ["KHCN"]
    assert result.response["rag"]["retrieved"][0]["doc_id"] == "dict-mention"
    assert "Synthetic entry mentioning KHCN." in llm.messages[1]["content"]

    dictionary_retriever.queries.clear()
    result = service.answer([{"role": "user", "content": "PB viết tắt cho gì?"}], response_mode="text")

    assert dictionary_retriever.queries == ["PB viết tắt cho gì?", "PB"]
    assert result.response["rag"]["retrieval_metadata"]["dictionary_fallback"] is True
    assert result.response["rag"]["retrieval_metadata"]["dictionary_fallback_metadata"]["query_plan"]["target_terms"] == ["PB"]
    assert "Synthetic entry mentioning PB." in llm.messages[-1]["content"]

    dictionary_retriever.queries.clear()
    result = service.answer([{"role": "user", "content": "khcnxuathienodau"}], response_mode="text")

    assert dictionary_retriever.queries == ["khcnxuathienodau", "KHCN"]
    assert result.response["rag"]["retrieval_metadata"]["dictionary_fallback"] is True
    assert result.response["rag"]["retrieval_metadata"]["dictionary_fallback_metadata"]["query_plan"]["target_terms"] == ["KHCN"]

    dictionary_retriever.queries.clear()
    result = service.answer([{"role": "user", "content": "CTCC xuất hiện ở đâu?"}], response_mode="text")

    assert dictionary_retriever.queries == ["CTCC xuất hiện ở đâu?", "CTCC"]
    assert result.response["rag"]["retrieval_metadata"]["dictionary_fallback"] is True
    assert result.response["rag"]["retrieval_metadata"]["dictionary_fallback_metadata"]["query_plan"]["target_terms"] == ["CTCC"]
    assert "Synthetic entry mentioning CTCC." in llm.messages[-1]["content"]

    dictionary_retriever.queries.clear()
    result = service.answer([{"role": "user", "content": "ctccxuathienodau"}], response_mode="text")

    assert dictionary_retriever.queries == ["ctccxuathienodau", "CTCC"]
    assert result.response["rag"]["retrieval_metadata"]["dictionary_fallback"] is True
    assert result.response["rag"]["retrieval_metadata"]["dictionary_fallback_metadata"]["query_plan"]["target_terms"] == ["CTCC"]

    dictionary_retriever.queries.clear()
    result = service.answer([{"role": "user", "content": "pbviettatcuagi"}], response_mode="text")

    assert dictionary_retriever.queries == ["pbviettatcuagi", "PB"]
    assert result.response["rag"]["retrieval_metadata"]["dictionary_fallback"] is True
    assert result.response["rag"]["retrieval_metadata"]["dictionary_fallback_metadata"]["query_plan"]["target_terms"] == ["PB"]

    dictionary_retriever.queries.clear()
    result = service.answer([{"role": "user", "content": "xyzxuathienodau"}], response_mode="text")

    assert dictionary_retriever.queries == ["xyzxuathienodau", "XYZ"]
    assert result.response["rag"]["retrieval_metadata"]["dictionary_fallback"] is True
    assert result.response["rag"]["retrieval_metadata"]["dictionary_fallback_metadata"]["query_plan"]["target_terms"] == ["XYZ"]


def test_text_dictionary_fallback_caps_total_sources_and_drops_tiny_benchmark_hits() -> None:
    class BenchmarkRetriever:
        name = "bm25"
        build_time_s = 0.0

        def search(self, query: Query, top_k: int) -> RetrievalResult:
            assert top_k == 6
            hits = [
                RetrievalHit(
                    doc_id=f"bench-good-{index}",
                    score=0.25,
                    rank=index,
                    title=f"Good {index}",
                    text=f"Useful benchmark {index}.",
                    metadata=PUBLIC_METADATA,
                    data_tier="public",
                )
                for index in range(1, 4)
            ]
            hits.extend(
                RetrievalHit(
                    doc_id=f"bench-tiny-{index}",
                    score=0.0001,
                    rank=rank,
                    title=f"Tiny {index}",
                    text=f"Tiny benchmark {index}.",
                    metadata=PUBLIC_METADATA,
                    data_tier="public",
                )
                for rank, index in enumerate(range(1, 4), start=4)
            )
            return RetrievalResult(query=query, hits=hits, latency_s=0.02)

    class DictionaryFallbackRetriever:
        name = "dictionary-graph"
        build_time_s = 0.0

        def search(self, query: Query, top_k: int) -> RetrievalResult:
            assert top_k == 6
            return RetrievalResult(
                query=query,
                hits=[
                    RetrievalHit(
                        doc_id=f"dict-{index}",
                        score=2.0 - index * 0.01,
                        rank=index,
                        title=f"Dictionary {index}",
                        text=f"Dictionary entry {index}.",
                        metadata={
                            "data_tier": "semi_private",
                            "kind": "dictionary",
                            "dictionary_direct_score": 1.0,
                            "dictionary_match_mode": "strict",
                        },
                        data_tier="semi_private",
                    )
                    for index in range(1, 9)
                ],
                latency_s=0.01,
                metadata={"kind": "dictionary"},
            )

    text_retriever = BenchmarkRetriever()
    dictionary_retriever = DictionaryFallbackRetriever()
    llm = FakeLLM()
    service = RagChatService(
        config=ChatProxyConfig(
            top_k=6,
            dictionary_top_k=5,
            model_id="rag-test",
            allow_external_semi_private=True,
        ),
        benchmark=BenchmarkData(name="fixture", dataset_id="fixture/test", queries=[], documents=[], qrels={}),
        retriever=text_retriever,
        llm=llm,
        retrievers={"bm25": text_retriever, "dictionary-graph": dictionary_retriever},
        dictionary_status={"source": "artifact", "entry_count": 5},
    )

    result = service.answer([{"role": "user", "content": "pháo đài"}], response_mode="text", top_k=6)
    doc_ids = [source["doc_id"] for source in result.response["rag"]["retrieved"]]

    assert len(doc_ids) == 6
    assert doc_ids == ["dict-1", "dict-2", "dict-3", "dict-4", "dict-5", "dict-6"]
    assert "Tiny benchmark" not in llm.messages[1]["content"]


def test_format_context_distributes_budget_across_all_hits() -> None:
    hits = [
        RetrievalHit(
            doc_id=f"doc-{index}",
            score=1.0,
            rank=index,
            title=f"Title {index}",
            text=f"Important context {index}. " + ("x" * 700),
            metadata=PUBLIC_METADATA,
            data_tier="public",
        )
        for index in range(1, 7)
    ]

    context = _format_context(hits, max_context_chars=900)

    for index in range(1, 7):
        assert f"[doc-{index}]" in context
        assert f"Title {index}" in context
    assert len(context) <= 900
    assert not context.endswith("[")


def test_uncited_zero_score_sources_are_hidden_but_cited_zero_score_sources_remain() -> None:
    class LowScoreRetriever(FakeRetriever):
        def search(self, query: Query, top_k: int) -> RetrievalResult:
            return RetrievalResult(
                query=query,
                hits=[
                    RetrievalHit(
                        doc_id="cited-low",
                        score=0.0,
                        rank=1,
                        title="Cited",
                        text="Cited low score.",
                        metadata=PUBLIC_METADATA,
                        data_tier="public",
                    ),
                    RetrievalHit(
                        doc_id="uncited-low",
                        score=0.0,
                        rank=2,
                        title="Uncited",
                        text="Uncited low score.",
                        metadata=PUBLIC_METADATA,
                        data_tier="public",
                    ),
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


def test_score_controls_filter_sort_prompt_and_display_sources() -> None:
    class MixedScoreRetriever(FakeRetriever):
        def search(self, query: Query, top_k: int) -> RetrievalResult:
            assert top_k == 4
            return RetrievalResult(
                query=query,
                hits=[
                    RetrievalHit(
                        doc_id="low",
                        score=0.2,
                        rank=1,
                        title="Low",
                        text="Low confidence.",
                        metadata=PUBLIC_METADATA,
                        data_tier="public",
                    ),
                    RetrievalHit(
                        doc_id="high",
                        score=2.0,
                        rank=2,
                        title="High",
                        text="High confidence.",
                        metadata=PUBLIC_METADATA,
                        data_tier="public",
                    ),
                    RetrievalHit(
                        doc_id="mid",
                        score=1.0,
                        rank=3,
                        title="Mid",
                        text="Mid confidence.",
                        metadata=PUBLIC_METADATA,
                        data_tier="public",
                    ),
                    RetrievalHit(
                        doc_id="too-high",
                        score=9.0,
                        rank=4,
                        title="Too high",
                        text="Outlier.",
                        metadata=PUBLIC_METADATA,
                        data_tier="public",
                    ),
                ],
                latency_s=0.01,
            )

    class CitingFilteredLLM(FakeLLM):
        def generate(self, *args, **kwargs) -> GenerationResult:
            result = super().generate(*args, **kwargs)
            result.answer = "Filtered answer [high]."
            return result

    retriever = MixedScoreRetriever()
    llm = CitingFilteredLLM()
    service = RagChatService(
        config=ChatProxyConfig(top_k=4, model_id="rag-test"),
        benchmark=BenchmarkData(
            name="fixture",
            dataset_id="fixture/test",
            queries=[],
            documents=[],
            qrels={},
        ),
        retriever=retriever,
        llm=llm,
    )

    result = service.answer(
        [{"role": "user", "content": "What do cats do?"}],
        top_k=4,
        score_min=0.5,
        score_max=2.5,
        sort_by_score=True,
    )

    prompt = llm.messages[1]["content"]
    assert prompt.index("[high]") < prompt.index("[mid]")
    assert "[low]" not in prompt
    assert "[too-high]" not in prompt
    assert [source["doc_id"] for source in result.response["rag"]["retrieved"]] == ["high", "mid"]
    assert [source["rank"] for source in result.response["rag"]["retrieved"]] == [1, 2]
    assert result.response["rag"]["retrieval_metadata"]["score_filter"] == {
        "min_score": 0.5,
        "max_score": 2.5,
        "sort_by_score": True,
        "input_count": 4,
        "output_count": 2,
    }


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
    assert parse_chat_command("/img digit 3") == ("img", "digit 3")
    assert parse_chat_command("/image cats") == ("img", "cats")
    assert parse_chat_command("/dict AMONIT") == ("dict", "AMONIT")
    assert parse_chat_command("plain text") is None


def test_service_from_config_can_run_with_no_benchmark_corpus(tmp_path) -> None:
    keys_path = tmp_path / "groq.env"
    keys_path.write_text("test=gsk_test\n", encoding="utf-8")

    service = RagChatService.from_config(
        ChatProxyConfig(
            bench="none",
            retriever="bm25",
            available_retrievers=("bm25",),
            groq_keys_path=keys_path,
            dictionary_artifact=None,
            dictionary_source_dir=None,
        ),
        llm_factory=lambda _keys: FakeLLM(),
    )

    retrieval = service.retriever.search(Query("q", "anything"), top_k=3)

    assert service.benchmark.name == "none"
    assert service.benchmark.dataset_id == "none/empty"
    assert retrieval.hits == []
    assert retrieval.metadata["empty_corpus"] is True
