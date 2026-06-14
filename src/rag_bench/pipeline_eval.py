from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol, Sequence

from rag_bench.chat_service import ChatServiceResult, RagChatService
from rag_bench.groq_client import GenerationResult
from rag_bench.privacy import PrivacyRouteError
from rag_bench.types import RetrievalHit


@dataclass(frozen=True)
class PipelineEvalRequest:
    eval_id: str
    query: str
    mode: str = "text"
    data_tier: str = "public"
    session_id: str | None = None
    request_model: str | None = None
    request_retriever: str | None = None
    top_k: int | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class PipelineEvalOutput:
    eval_id: str
    answer: str
    hits: list[RetrievalHit]
    generation: GenerationResult | None
    query_plan: dict[str, Any]
    retrieval_metadata: dict[str, Any]
    privacy: dict[str, Any]
    response: dict[str, Any]
    error: str | None = None

    @property
    def ok(self) -> bool:
        return self.error is None

    @property
    def retrieved_doc_ids(self) -> list[str]:
        return [hit.doc_id for hit in self.hits]


class PipelineEvalAdapter(Protocol):
    pipeline_id: str

    def evaluate(self, request: PipelineEvalRequest) -> PipelineEvalOutput: ...


@dataclass
class ProductionChatPipelineAdapter:
    service: RagChatService
    request_model: str | None = None
    pipeline_id: str = "production_chat"

    def evaluate(self, request: PipelineEvalRequest) -> PipelineEvalOutput:
        session_id = request.session_id or f"pipeline-eval-{request.eval_id}"
        try:
            result = self.service.answer(
                [{"role": "user", "content": request.query, "data_tier": request.data_tier}],
                request_model=request.request_model or self.request_model,
                request_retriever=request.request_retriever,
                top_k=request.top_k,
                response_mode=request.mode,
                session_id=session_id,
                reset_privacy=True,
                memory=False,
            )
        except PrivacyRouteError as exc:
            return PipelineEvalOutput(
                eval_id=request.eval_id,
                answer="",
                hits=[],
                generation=None,
                query_plan={},
                retrieval_metadata={},
                privacy=exc.decision.to_payload(),
                response={},
                error=exc.decision.reason,
            )
        return PipelineEvalOutput(
            eval_id=request.eval_id,
            answer=_answer_from_service_result(result),
            hits=list(result.hits),
            generation=result.generation,
            query_plan=_query_plan_from_service_result(result),
            retrieval_metadata=dict(result.retrieval_metadata),
            privacy=dict(result.response.get("privacy") or {}),
            response=result.response,
            error=result.generation.error,
        )


def summarize_pipeline_outputs(outputs: Sequence[PipelineEvalOutput]) -> dict[str, Any]:
    total = len(outputs)
    error_count = sum(output.error is not None for output in outputs)
    hit_count = sum(bool(output.hits) for output in outputs)
    return {
        "pipeline_count": total,
        "pipeline_error_count": error_count,
        "pipeline_ok_count": total - error_count,
        "with_retrieved_hits_count": hit_count,
    }


def _answer_from_service_result(result: ChatServiceResult) -> str:
    choices = result.response.get("choices") or []
    if choices and isinstance(choices[0], dict):
        message = choices[0].get("message") or {}
        return str(message.get("content") or "")
    return str(result.generation.answer or "")


def _query_plan_from_service_result(result: ChatServiceResult) -> dict[str, Any]:
    if isinstance(result.response.get("query_plan"), dict):
        return dict(result.response["query_plan"])
    metadata = result.retrieval_metadata or result.response.get("rag", {}).get("retrieval_metadata") or {}
    plan = metadata.get("query_plan") if isinstance(metadata, dict) else None
    return dict(plan) if isinstance(plan, dict) else {}
