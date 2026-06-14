from __future__ import annotations

from dataclasses import dataclass, field

from rag_bench.chat_service import ChatServiceResult
from rag_bench.groq_client import GenerationResult
from rag_bench.pipeline_eval import PipelineEvalRequest, ProductionChatPipelineAdapter, summarize_pipeline_outputs
from rag_bench.types import RetrievalHit


@dataclass
class RecordingService:
    calls: list[dict] = field(default_factory=list)

    def answer(self, messages, **kwargs):
        self.calls.append({"messages": messages, "kwargs": kwargs})
        hit = RetrievalHit(
            doc_id="DOC_A",
            score=1.0,
            rank=1,
            title="Synthetic title",
            text="Synthetic evidence.",
        )
        query_plan = {"intent": "definition", "schema_gaps": []}
        response = {
            "choices": [{"message": {"content": "Synthetic answer [DOC_A]"}}],
            "query_plan": query_plan,
            "privacy": {"session_taint": "semi_private", "provider_allowed": True},
            "rag": {"retrieval_metadata": {"query_plan": query_plan}},
        }
        return ChatServiceResult(
            response=response,
            generation=GenerationResult(
                answer="Synthetic answer [DOC_A]",
                key_alias="generator",
                attempted_aliases=["generator"],
                latency_s=0.01,
                retry_count=0,
            ),
            hits=[hit],
            retrieval_latency_s=0.01,
            retrieval_metadata={"query_plan": query_plan, "response_mode": "dictionary"},
        )


def test_production_chat_pipeline_adapter_preserves_eval_request_contract() -> None:
    service = RecordingService()
    adapter = ProductionChatPipelineAdapter(service=service, request_model="mimo-v2.5")

    output = adapter.evaluate(
        PipelineEvalRequest(
            eval_id="case-1",
            query="/dict TERM_A",
            mode="dictionary",
            data_tier="semi_private",
            request_retriever="dictionary-graph",
            top_k=5,
        )
    )

    assert output.ok is True
    assert output.answer == "Synthetic answer [DOC_A]"
    assert output.retrieved_doc_ids == ["DOC_A"]
    assert output.query_plan["intent"] == "definition"
    assert output.retrieval_metadata["response_mode"] == "dictionary"
    call = service.calls[0]
    assert call["messages"] == [{"role": "user", "content": "/dict TERM_A", "data_tier": "semi_private"}]
    assert call["kwargs"]["request_model"] == "mimo-v2.5"
    assert call["kwargs"]["request_retriever"] == "dictionary-graph"
    assert call["kwargs"]["response_mode"] == "dictionary"
    assert call["kwargs"]["top_k"] == 5
    assert call["kwargs"]["reset_privacy"] is True
    assert call["kwargs"]["memory"] is False


def test_summarize_pipeline_outputs_counts_errors_and_hits() -> None:
    service = RecordingService()
    adapter = ProductionChatPipelineAdapter(service=service)
    ok_output = adapter.evaluate(PipelineEvalRequest(eval_id="ok", query="hello"))
    error_output = ok_output.__class__(
        eval_id="error",
        answer="",
        hits=[],
        generation=None,
        query_plan={},
        retrieval_metadata={},
        privacy={},
        response={},
        error="blocked",
    )

    summary = summarize_pipeline_outputs([ok_output, error_output])

    assert summary == {
        "pipeline_count": 2,
        "pipeline_error_count": 1,
        "pipeline_ok_count": 1,
        "with_retrieved_hits_count": 1,
    }
