from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from typing import Any, Literal


FeedbackProvenance = Literal["gold", "ragas", "ai_judge", "mimo_judge", "heuristic", "missing"]
PreferenceType = Literal[
    "context_policy_preference",
    "retrieval_context_preference",
    "context_sufficiency_preference",
]

VALID_PROVENANCE = {"gold", "ragas", "ai_judge", "mimo_judge", "heuristic", "missing"}
VALID_PREFERENCE_TYPES = {
    "context_policy_preference",
    "retrieval_context_preference",
    "context_sufficiency_preference",
}


@dataclass(frozen=True)
class RetrievalContextAction:
    benchmark: str
    query_id: str
    question: str
    retrieval_strategy: str
    top_k: int
    context_policy: str
    budget_chars: int | None
    fusion_strategy: str | None = None
    adaptive_profile: str | None = None
    selected_context_policy: str | None = None
    selected_budget_chars: int | None = None
    generator_model: str | None = None
    source_run_id: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        _require_text("benchmark", self.benchmark)
        _require_text("query_id", self.query_id)
        _require_text("question", self.question)
        _require_text("retrieval_strategy", self.retrieval_strategy)
        _require_text("context_policy", self.context_policy)
        _require_positive_int("top_k", self.top_k)
        if self.budget_chars is not None:
            _require_positive_int("budget_chars", self.budget_chars)
        if self.selected_budget_chars is not None:
            _require_positive_int("selected_budget_chars", self.selected_budget_chars)

    @property
    def action_id(self) -> str:
        return stable_record_id("rlaif-action-v1", self.identity_payload())

    def identity_payload(self) -> dict[str, Any]:
        return {
            "benchmark": self.benchmark,
            "query_id": self.query_id,
            "retrieval_strategy": self.retrieval_strategy,
            "fusion_strategy": self.fusion_strategy,
            "top_k": self.top_k,
            "context_policy": self.context_policy,
            "budget_chars": self.budget_chars,
            "adaptive_profile": self.adaptive_profile,
            "selected_context_policy": self.selected_context_policy,
            "selected_budget_chars": self.selected_budget_chars,
            "generator_model": self.generator_model,
        }

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["action_id"] = self.action_id
        return data


@dataclass(frozen=True)
class RlaifAnswerFeedback:
    action_id: str
    query_id: str
    provenance: FeedbackProvenance
    quality_score: float | None = None
    exact_match: float | None = None
    token_f1: float | None = None
    answer_relevancy: float | None = None
    faithfulness: float | None = None
    answer_correctness: float | None = None
    unsupported_claim_penalty: float | None = None
    ambiguous: bool = False
    missing_reason: str | None = None
    judge_provider: str | None = None
    judge_model: str | None = None
    rationale: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        _require_text("action_id", self.action_id)
        _require_text("query_id", self.query_id)
        _require_provenance(self.provenance)
        for field_name in (
            "quality_score",
            "exact_match",
            "token_f1",
            "answer_relevancy",
            "faithfulness",
            "answer_correctness",
            "unsupported_claim_penalty",
        ):
            _require_score_or_none(field_name, getattr(self, field_name))

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class RlaifContextFeedback:
    action_id: str
    query_id: str
    provenance: FeedbackProvenance
    sufficient: bool | None = None
    selected_chunk_ids: tuple[str, ...] = ()
    redundant_chunk_ids: tuple[str, ...] = ()
    irrelevant_chunk_ids: tuple[str, ...] = ()
    missing_evidence: bool | None = None
    minimality_score: float | None = None
    evidence_support_score: float | None = None
    context_quality_score: float | None = None
    ambiguous: bool = False
    missing_reason: str | None = None
    judge_provider: str | None = None
    judge_model: str | None = None
    rationale: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        _require_text("action_id", self.action_id)
        _require_text("query_id", self.query_id)
        _require_provenance(self.provenance)
        object.__setattr__(
            self,
            "selected_chunk_ids",
            _tuple_of_text("selected_chunk_ids", self.selected_chunk_ids),
        )
        object.__setattr__(
            self,
            "redundant_chunk_ids",
            _tuple_of_text("redundant_chunk_ids", self.redundant_chunk_ids),
        )
        object.__setattr__(
            self,
            "irrelevant_chunk_ids",
            _tuple_of_text("irrelevant_chunk_ids", self.irrelevant_chunk_ids),
        )
        for field_name in ("minimality_score", "evidence_support_score", "context_quality_score"):
            _require_score_or_none(field_name, getattr(self, field_name))

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class RlaifRewardWeights:
    quality: float = 0.75
    support: float = 0.10
    token: float = 0.05
    latency: float = 0.05
    kv: float = 0.05
    error: float = 1.0
    unsupported: float = 1.0

    def __post_init__(self) -> None:
        for field_name, value in asdict(self).items():
            if value < 0:
                raise ValueError(f"{field_name} weight must be non-negative")


@dataclass(frozen=True)
class RlaifReward:
    action_id: str
    query_id: str
    reward: float
    quality: float
    evidence_support: float
    token_cost_norm: float
    latency_norm: float
    kv_cost_norm: float
    error_penalty: float
    unsupported_claim_penalty: float
    weights: RlaifRewardWeights = field(default_factory=RlaifRewardWeights)
    provenance: FeedbackProvenance = "heuristic"
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        _require_text("action_id", self.action_id)
        _require_text("query_id", self.query_id)
        _require_provenance(self.provenance)
        _require_bounded("reward", self.reward, -1.0, 1.0)
        for field_name in (
            "quality",
            "evidence_support",
            "token_cost_norm",
            "latency_norm",
            "kv_cost_norm",
            "error_penalty",
            "unsupported_claim_penalty",
        ):
            _require_score_or_none(field_name, getattr(self, field_name))

    @classmethod
    def from_components(
        cls,
        *,
        action_id: str,
        query_id: str,
        quality: float,
        evidence_support: float = 0.0,
        token_cost_norm: float = 0.0,
        latency_norm: float = 0.0,
        kv_cost_norm: float = 0.0,
        error_penalty: float = 0.0,
        unsupported_claim_penalty: float = 0.0,
        weights: RlaifRewardWeights | None = None,
        provenance: FeedbackProvenance = "heuristic",
        metadata: dict[str, Any] | None = None,
    ) -> RlaifReward:
        active_weights = weights or RlaifRewardWeights()
        raw_reward = (
            active_weights.quality * quality
            + active_weights.support * evidence_support
            - active_weights.token * token_cost_norm
            - active_weights.latency * latency_norm
            - active_weights.kv * kv_cost_norm
            - active_weights.error * error_penalty
            - active_weights.unsupported * unsupported_claim_penalty
        )
        return cls(
            action_id=action_id,
            query_id=query_id,
            reward=_clamp(raw_reward, -1.0, 1.0),
            quality=quality,
            evidence_support=evidence_support,
            token_cost_norm=token_cost_norm,
            latency_norm=latency_norm,
            kv_cost_norm=kv_cost_norm,
            error_penalty=error_penalty,
            unsupported_claim_penalty=unsupported_claim_penalty,
            weights=active_weights,
            provenance=provenance,
            metadata=metadata or {},
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class RlaifPreference:
    preference_type: PreferenceType
    query_id: str
    chosen_action_id: str
    rejected_action_id: str
    reward_gap: float
    quality_gap: float
    efficiency_gap: float
    reason: str
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        _require_preference_type(self.preference_type)
        _require_text("query_id", self.query_id)
        _require_text("chosen_action_id", self.chosen_action_id)
        _require_text("rejected_action_id", self.rejected_action_id)
        _require_text("reason", self.reason)
        if self.chosen_action_id == self.rejected_action_id:
            raise ValueError("chosen_action_id and rejected_action_id must be different")

    @property
    def preference_id(self) -> str:
        return stable_record_id(
            "rlaif-preference-v1",
            {
                "preference_type": self.preference_type,
                "query_id": self.query_id,
                "chosen_action_id": self.chosen_action_id,
                "rejected_action_id": self.rejected_action_id,
                "reason": self.reason,
            },
        )

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["preference_id"] = self.preference_id
        return data


def stable_record_id(prefix: str, payload: dict[str, Any], *, length: int = 16) -> str:
    _require_text("prefix", prefix)
    encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    digest = hashlib.sha256(encoded).hexdigest()[:length]
    return f"{prefix}-{digest}"


def _require_text(field_name: str, value: str) -> None:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field_name} must be a non-empty string")


def _require_positive_int(field_name: str, value: int) -> None:
    if not isinstance(value, int) or value <= 0:
        raise ValueError(f"{field_name} must be a positive integer")


def _require_score_or_none(field_name: str, value: float | None) -> None:
    if value is None:
        return
    _require_bounded(field_name, float(value), 0.0, 1.0)


def _require_bounded(field_name: str, value: float, lower: float, upper: float) -> None:
    if value < lower or value > upper:
        raise ValueError(f"{field_name} must be between {lower} and {upper}")


def _require_provenance(value: str) -> None:
    if value not in VALID_PROVENANCE:
        allowed = ", ".join(sorted(VALID_PROVENANCE))
        raise ValueError(f"provenance must be one of: {allowed}")


def _require_preference_type(value: str) -> None:
    if value not in VALID_PREFERENCE_TYPES:
        allowed = ", ".join(sorted(VALID_PREFERENCE_TYPES))
        raise ValueError(f"preference_type must be one of: {allowed}")


def _tuple_of_text(field_name: str, values: tuple[str, ...] | list[str]) -> tuple[str, ...]:
    if not isinstance(values, (tuple, list)):
        raise ValueError(f"{field_name} must be a tuple or list of strings")
    output = tuple(values)
    for value in output:
        _require_text(field_name, value)
    return output


def _clamp(value: float, lower: float, upper: float) -> float:
    return max(lower, min(upper, value))
