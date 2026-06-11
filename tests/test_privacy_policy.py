from __future__ import annotations

from dataclasses import dataclass

import pytest

from rag_bench.chat_service import ChatProxyConfig, RagChatService, _hit_source_payload
from rag_bench.groq_client import GenerationResult
from rag_bench.privacy import (
    ConversationPrivacyState,
    DataTier,
    PrivacyRouteError,
    max_data_tier,
    normalize_data_tier,
)
from rag_bench.types import BenchmarkData, Query, RetrievalHit, RetrievalResult


@dataclass
class TieredRetriever:
    tier: str = "public"
    name: str = "bm25"
    build_time_s: float = 0.0
    calls: int = 0

    def search(self, query: Query, top_k: int) -> RetrievalResult:
        self.calls += 1
        return RetrievalResult(
            query=query,
            hits=[
                RetrievalHit(
                    doc_id=f"{self.tier}-doc",
                    score=1.0,
                    rank=1,
                    title=f"{self.tier} title",
                    text=f"{self.tier} synthetic context",
                    metadata={
                        "data_tier": self.tier,
                        "doc_type": "synthetic",
                        "raw_docx_text": "sensitive synthetic raw text",
                        "rich_blocks": [{"text": "sensitive synthetic rich text"}],
                        "dictionary_evidence_text": "sensitive synthetic evidence",
                    },
                    data_tier=self.tier,
                    doc_type="synthetic",
                )
            ],
            latency_s=0.01,
        )


class FakeLLM:
    def __init__(self) -> None:
        self.calls = 0
        self.messages = []
        self.key_usage_counts = {}

    def generate(self, messages, *, model=None, temperature=0.0, max_completion_tokens=512):
        self.calls += 1
        self.messages = messages
        return GenerationResult(
            answer="synthetic answer [public-doc]",
            key_alias="fake",
            attempted_aliases=["fake"],
            latency_s=0.01,
            retry_count=0,
            prompt_tokens=10,
            completion_tokens=5,
            total_tokens=15,
            estimated_tokens=15,
        )

    def rate_limit_snapshot(self):
        return {}


def _service(retriever, *, allow_external_semi_private=False) -> tuple[RagChatService, FakeLLM]:
    llm = FakeLLM()
    service = RagChatService(
        config=ChatProxyConfig(
            model_id="rag-test",
            top_k=1,
            trusted_local_models=("local-safe",),
            allow_external_semi_private=allow_external_semi_private,
        ),
        benchmark=BenchmarkData(name="fixture", dataset_id="fixture/test", queries=[], documents=[], qrels={}),
        retriever=retriever,
        llm=llm,
        retrievers={retriever.name: retriever},
    )
    return service, llm


def test_data_tier_ordering_is_conservative_for_unknown_values() -> None:
    assert max_data_tier(DataTier.PUBLIC, DataTier.PRIVATE) == DataTier.PRIVATE
    assert max_data_tier("public", "semi_private") == DataTier.SEMI_PRIVATE
    assert normalize_data_tier("malformed-tier") == DataTier.PRIVATE


def test_private_hit_blocks_external_provider_before_generation() -> None:
    retriever = TieredRetriever(tier="private")
    service, llm = _service(retriever)

    with pytest.raises(PrivacyRouteError) as error:
        service.answer([{"role": "user", "content": "question"}], session_id="chat-a")

    assert llm.calls == 0
    assert error.value.decision.reason == "session_taint_private_requires_local_provider"
    assert service.privacy_states["chat-a"].max_seen_tier == DataTier.PRIVATE
    assert service.privacy_states["chat-a"].private_seen is True


def test_session_taint_persists_when_history_is_disabled() -> None:
    retriever = TieredRetriever(tier="private")
    service, llm = _service(retriever)

    first = service.answer(
        [{"role": "user", "content": "private question"}],
        request_model="local-safe",
        session_id="chat-tainted",
    )

    assert first.response["privacy"]["state"]["session_taint"] == "private"
    assert llm.calls == 1

    retriever.tier = "semi_private"
    with pytest.raises(PrivacyRouteError):
        service.answer(
            [{"role": "user", "content": "later non-private question"}],
            session_id="chat-tainted",
            memory=False,
        )

    assert llm.calls == 1
    assert service.privacy_states["chat-tainted"].max_seen_tier == DataTier.PRIVATE


def test_new_session_resets_privacy_taint() -> None:
    retriever = TieredRetriever(tier="private")
    service, llm = _service(retriever)

    with pytest.raises(PrivacyRouteError):
        service.answer([{"role": "user", "content": "private question"}], session_id="session-a")

    retriever.tier = "public"
    result = service.answer([{"role": "user", "content": "public question"}], session_id="session-b")

    assert result.response["privacy"]["state"]["session_taint"] == "public"
    assert llm.calls == 1


def test_public_external_provider_still_works() -> None:
    service, llm = _service(TieredRetriever(tier="public"))

    result = service.answer([{"role": "user", "content": "public question"}], session_id="public-chat")

    assert result.response["choices"][0]["message"]["content"]
    assert result.response["privacy"]["state"]["session_taint"] == "public"
    assert llm.calls == 1


def test_semi_private_external_policy_is_configurable() -> None:
    blocked_service, blocked_llm = _service(TieredRetriever(tier="semi_private"))
    with pytest.raises(PrivacyRouteError):
        blocked_service.answer([{"role": "user", "content": "semi private question"}], session_id="semi-a")
    assert blocked_llm.calls == 0

    allowed_service, allowed_llm = _service(TieredRetriever(tier="semi_private"), allow_external_semi_private=True)
    result = allowed_service.answer([{"role": "user", "content": "semi private question"}], session_id="semi-b")

    assert result.response["privacy"]["state"]["session_taint"] == "semi_private"
    assert allowed_llm.calls == 1


def test_private_source_payload_is_redacted_by_default() -> None:
    hit = TieredRetriever(tier="private").search(Query(query_id="q", text="q"), 1).hits[0]

    payload = _hit_source_payload(hit)

    assert payload["redacted"] is True
    assert payload["text"] is None
    assert payload["title"] is None
    assert payload["metadata"].get("raw_docx_text") is None
    assert payload["metadata"].get("rich_blocks") is None
    assert payload["metadata"].get("dictionary_evidence_text") is None
    assert "sensitive synthetic" not in str(payload)


def test_external_llm_rewrite_retriever_falls_back_before_call_in_private_session() -> None:
    llm_retriever = TieredRetriever(tier="public", name="llm-query-rewrite")
    bm25_retriever = TieredRetriever(tier="public", name="bm25")
    service, llm = _service(llm_retriever)
    service.retrievers = {"llm-query-rewrite": llm_retriever, "bm25": bm25_retriever}
    service.privacy_states["chat-private"] = ConversationPrivacyState(
        session_id="chat-private",
        max_seen_tier=DataTier.PRIVATE,
        private_seen=True,
    )

    with pytest.raises(PrivacyRouteError):
        service.answer(
            [{"role": "user", "content": "public followup"}],
            request_retriever="llm-query-rewrite",
            session_id="chat-private",
        )

    assert llm_retriever.calls == 0
    assert bm25_retriever.calls == 1
    assert llm.calls == 0
