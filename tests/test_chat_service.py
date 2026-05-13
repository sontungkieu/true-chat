from __future__ import annotations

from dataclasses import dataclass

from rag_bench.chat_service import ChatProxyConfig, RagChatService, last_user_text, parse_chat_command
from rag_bench.groq_client import GenerationResult
from rag_bench.types import BenchmarkData, Document, Query, RetrievalHit, RetrievalResult


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


class FakeLLM:
    def __init__(self) -> None:
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
            key_alias="alias-a",
            attempted_aliases=["alias-a"],
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
    user_prompt = llm.messages[1]["content"]
    assert "Previous question" in user_prompt
    assert "Cats purr and chase toys." in user_prompt
    assert "What do cats do?" in user_prompt


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
    assert parse_chat_command("plain text") is None
