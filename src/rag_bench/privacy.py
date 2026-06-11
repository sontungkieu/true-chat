from __future__ import annotations

import os
import re
from dataclasses import dataclass
from enum import Enum
from typing import Any, Sequence
from urllib.parse import urlparse


class DataTier(str, Enum):
    PUBLIC = "public"
    SEMI_PRIVATE = "semi_private"
    PRIVATE = "private"


class BackendKind(str, Enum):
    LOCAL_PROCESS = "local_process"
    SELF_HOSTED_PRIVATE = "self_hosted_private"
    PRIVATE_LAN = "private_lan"
    PRIVATE_VPC = "private_vpc"
    EXTERNAL_SAAS = "external_saas"
    UNKNOWN = "unknown"


PRIVATE_CAPABLE_BACKEND_KINDS = {
    BackendKind.LOCAL_PROCESS,
    BackendKind.SELF_HOSTED_PRIVATE,
    BackendKind.PRIVATE_LAN,
    BackendKind.PRIVATE_VPC,
}
_TIER_ORDER = {
    DataTier.PUBLIC: 0,
    DataTier.SEMI_PRIVATE: 1,
    DataTier.PRIVATE: 2,
}
_PRIVATE_PATH_MARKERS = {"private", "secret", "classified", "top-secret", "top_secret", "tuyet-mat", "tuyệt-mật"}
_SEMI_PRIVATE_PATH_MARKERS = {"semi-private", "semi_private", "semiprivate"}
_LOCAL_PROVIDERS = {"local", "localhost", "vllm", "ollama", "llama.cpp", "llamacpp"}
_EXTERNAL_PROVIDERS = {"groq", "mimo", "openai", "anthropic", "deepseek", "google", "gemini"}
_SENSITIVE_SOURCE_KEYS = {
    "text",
    "plain_text",
    "raw_docx_text",
    "rich_blocks",
    "evidence_text",
    "dictionary_evidence_text",
    "messages",
    "history",
    "prompt",
    "context",
}


def normalize_data_tier(value: Any, *, missing: DataTier = DataTier.PUBLIC) -> DataTier:
    if isinstance(value, DataTier):
        return value
    if value is None or value == "":
        return missing
    normalized = str(value).strip().lower().replace("-", "_").replace(" ", "_")
    if normalized in {"public", "open"}:
        return DataTier.PUBLIC
    if normalized in {"semi_private", "semiprivate", "semi", "restricted", "internal"}:
        return DataTier.SEMI_PRIVATE
    if normalized in {"private", "secret", "classified", "top_secret", "topsecret", "tuyet_mat", "tuyệt_mật"}:
        return DataTier.PRIVATE
    return DataTier.PRIVATE


def max_data_tier(*tiers: Any) -> DataTier:
    highest = DataTier.PUBLIC
    for tier in tiers:
        normalized = normalize_data_tier(tier)
        if _TIER_ORDER[normalized] > _TIER_ORDER[highest]:
            highest = normalized
    return highest


def is_private_tier(tier: Any) -> bool:
    return normalize_data_tier(tier) == DataTier.PRIVATE


def is_local_provider(provider: str | None, model: str | None = None) -> bool:
    return classify_backend(provider=provider, model=model).kind == BackendKind.LOCAL_PROCESS


def is_external_provider(provider: str | None, model: str | None = None) -> bool:
    return classify_backend(provider=provider, model=model).kind == BackendKind.EXTERNAL_SAAS


def normalize_backend_kind(value: Any) -> BackendKind:
    if isinstance(value, BackendKind):
        return value
    normalized = str(value or "").strip().lower().replace("-", "_").replace(" ", "_")
    aliases = {
        "local": BackendKind.LOCAL_PROCESS,
        "localhost": BackendKind.LOCAL_PROCESS,
        "local_process": BackendKind.LOCAL_PROCESS,
        "self_hosted": BackendKind.SELF_HOSTED_PRIVATE,
        "self_hosted_private": BackendKind.SELF_HOSTED_PRIVATE,
        "private_lan": BackendKind.PRIVATE_LAN,
        "lan": BackendKind.PRIVATE_LAN,
        "private_vpc": BackendKind.PRIVATE_VPC,
        "vpc": BackendKind.PRIVATE_VPC,
        "external": BackendKind.EXTERNAL_SAAS,
        "external_saas": BackendKind.EXTERNAL_SAAS,
        "saas": BackendKind.EXTERNAL_SAAS,
        "unknown": BackendKind.UNKNOWN,
        "": BackendKind.UNKNOWN,
    }
    return aliases.get(normalized, BackendKind.UNKNOWN)


@dataclass(frozen=True)
class BackendDescriptor:
    backend_id: str
    provider: str
    kind: BackendKind
    base_url: str | None = None
    model: str | None = None


@dataclass(frozen=True)
class PrivateBackendPolicy:
    trusted_private_backends: set[str]
    trusted_private_models: set[str]
    backend_model_allowlist: dict[str, set[str]]

    @classmethod
    def from_values(
        cls,
        *,
        trusted_private_backends: Sequence[str] | set[str] = (),
        trusted_private_models: Sequence[str] | set[str] = (),
        backend_model_allowlist: dict[str, Sequence[str] | set[str]] | None = None,
        trusted_local_models: Sequence[str] | set[str] = (),
    ) -> "PrivateBackendPolicy":
        return cls(
            trusted_private_backends=_normalized_set(trusted_private_backends),
            trusted_private_models=_normalized_set((*trusted_private_models, *trusted_local_models)),
            backend_model_allowlist={
                str(backend_id).strip(): _normalized_set(models)
                for backend_id, models in (backend_model_allowlist or {}).items()
                if str(backend_id).strip()
            },
        )

    def is_private_allowed(self, backend: BackendDescriptor) -> bool:
        backend_id = str(backend.backend_id or "").strip()
        model = str(backend.model or "").strip()
        if not backend_id or backend_id not in self.trusted_private_backends:
            return False
        if backend.kind not in PRIVATE_CAPABLE_BACKEND_KINDS:
            return False
        allowed_for_backend = self.backend_model_allowlist.get(backend_id)
        if allowed_for_backend is not None:
            return model in allowed_for_backend
        return bool(model and model in self.trusted_private_models)

    def model_is_trusted_somewhere(self, model: str | None) -> bool:
        model_id = str(model or "").strip()
        if not model_id:
            return False
        return model_id in self.trusted_private_models or any(
            model_id in models for models in self.backend_model_allowlist.values()
        )


def classify_backend(
    *,
    provider: str | None,
    model: str | None = None,
    backend_id: str | None = None,
    backend_kind: BackendKind | str | None = None,
    base_url: str | None = None,
) -> BackendDescriptor:
    provider_key = str(provider or "").strip().lower()
    resolved_backend_id = str(backend_id or provider_key or "unknown").strip()
    explicit_kind = normalize_backend_kind(backend_kind)
    if explicit_kind != BackendKind.UNKNOWN:
        kind = explicit_kind
    elif provider_key in _EXTERNAL_PROVIDERS:
        kind = BackendKind.EXTERNAL_SAAS
    elif provider_key in _LOCAL_PROVIDERS:
        kind = BackendKind.LOCAL_PROCESS
    elif _is_localhost_url(base_url):
        kind = BackendKind.LOCAL_PROCESS
    else:
        kind = BackendKind.UNKNOWN
    return BackendDescriptor(
        backend_id=resolved_backend_id,
        provider=provider_key or str(provider or ""),
        kind=kind,
        base_url=base_url,
        model=model,
    )


def infer_data_tier_from_path(value: Any, *, default: DataTier = DataTier.PUBLIC) -> DataTier:
    text = str(value or "")
    if not text:
        return default
    path_text = text.replace("\\", "/")
    components = [component.strip().lower() for component in re.split(r"[\\/]+", path_text) if component.strip()]
    if "/data/private/" in f"/{path_text.strip('/')}/" or any(component in _PRIVATE_PATH_MARKERS for component in components):
        return DataTier.PRIVATE
    if "/data/semi_private/" in f"/{path_text.strip('/')}/" or any(
        component in _SEMI_PRIVATE_PATH_MARKERS for component in components
    ):
        return DataTier.SEMI_PRIVATE
    return default


def infer_data_tier_from_metadata(metadata: dict[str, Any] | None, *, default: DataTier = DataTier.PUBLIC) -> DataTier:
    metadata = metadata or {}
    explicit = metadata.get("data_tier")
    if explicit not in (None, ""):
        return normalize_data_tier(explicit)
    for key in ("source_file", "source_path", "path", "artifact_dir", "source_dir"):
        tier = infer_data_tier_from_path(metadata.get(key), default=default)
        if tier != default:
            return tier
    source = metadata.get("source")
    if isinstance(source, dict):
        nested = infer_data_tier_from_metadata(source, default=default)
        if nested != default:
            return nested
    kind = str(metadata.get("kind") or metadata.get("doc_type") or "").strip().lower()
    if kind == "dictionary":
        return default if default != DataTier.PUBLIC else DataTier.SEMI_PRIVATE
    return default


def privacy_fields_from_metadata(
    metadata: dict[str, Any] | None,
    *,
    default_tier: DataTier = DataTier.PUBLIC,
    doc_type: str | None = None,
    source_id: str | None = None,
) -> dict[str, Any]:
    metadata = metadata or {}
    tier = infer_data_tier_from_metadata(metadata, default=default_tier)
    resolved_doc_type = doc_type or str(metadata.get("doc_type") or metadata.get("kind") or "") or None
    resolved_source_id = source_id or str(
        metadata.get("source_id") or metadata.get("source_set") or metadata.get("dataset") or ""
    ) or None
    return {
        "data_tier": tier.value,
        "doc_type": resolved_doc_type,
        "source_id": resolved_source_id,
        "allowed_llm": _string_list_or_none(metadata.get("allowed_llm")),
        "allowed_embedding": _string_list_or_none(metadata.get("allowed_embedding")),
        "redaction_policy": str(metadata.get("redaction_policy") or "") or None,
    }


def include_private_source_text_from_env() -> bool:
    value = os.getenv("RAG_BENCH_INCLUDE_PRIVATE_SOURCE_TEXT", "")
    return value.strip().lower() in {"1", "true", "yes", "on"}


@dataclass
class ConversationPrivacyState:
    session_id: str
    max_seen_tier: DataTier = DataTier.PUBLIC
    private_seen: bool = False
    external_blocked: bool = False
    last_turn_tier: DataTier = DataTier.PUBLIC
    reason: str | None = None

    @classmethod
    def from_mapping(cls, value: dict[str, Any] | None, *, session_id: str) -> "ConversationPrivacyState":
        if not isinstance(value, dict):
            return cls(session_id=session_id)
        max_seen = normalize_data_tier(value.get("max_seen_tier") or value.get("session_taint"))
        last_turn = normalize_data_tier(value.get("last_turn_tier"), missing=DataTier.PUBLIC)
        return cls(
            session_id=str(value.get("session_id") or session_id),
            max_seen_tier=max_seen,
            private_seen=bool(value.get("private_seen") or max_seen == DataTier.PRIVATE),
            external_blocked=bool(value.get("external_blocked")),
            last_turn_tier=last_turn,
            reason=str(value.get("reason") or "") or None,
        )

    def update(self, turn_tier: DataTier, *, external_blocked: bool = False, reason: str | None = None) -> None:
        self.last_turn_tier = turn_tier
        self.max_seen_tier = max_data_tier(self.max_seen_tier, turn_tier)
        self.private_seen = self.max_seen_tier == DataTier.PRIVATE
        self.external_blocked = bool(external_blocked)
        self.reason = reason

    def to_payload(self) -> dict[str, Any]:
        return {
            "session_id": self.session_id,
            "session_taint": self.max_seen_tier.value,
            "max_seen_tier": self.max_seen_tier.value,
            "private_seen": self.private_seen,
            "external_blocked": self.external_blocked,
            "last_turn_tier": self.last_turn_tier.value,
            "reason": self.reason,
        }


@dataclass(frozen=True)
class PrivacyDecision:
    effective_tier: DataTier
    provider_allowed: bool
    selected_provider: str | None
    selected_model: str | None
    backend_id: str | None
    backend_kind: BackendKind
    external_blocked: bool
    reason: str
    redaction_required: bool
    provider_requested: str
    model_requested: str | None

    def to_payload(self, *, session_state: ConversationPrivacyState | None = None) -> dict[str, Any]:
        payload = {
            "session_taint": session_state.max_seen_tier.value if session_state else self.effective_tier.value,
            "turn_tier": self.effective_tier.value,
            "provider_requested": self.provider_requested,
            "model_requested": self.model_requested,
            "provider_selected": self.selected_provider,
            "model_selected": self.selected_model,
            "backend_id": self.backend_id,
            "backend_kind": self.backend_kind.value,
            "provider_allowed": self.provider_allowed,
            "external_blocked": self.external_blocked,
            "reason": self.reason,
            "redaction_required": self.redaction_required,
        }
        if session_state is not None:
            payload["state"] = session_state.to_payload()
        return payload


class PrivacyRouteError(RuntimeError):
    def __init__(self, decision: PrivacyDecision) -> None:
        self.decision = decision
        super().__init__(decision.reason)


def enforce_privacy_route(
    requested_provider: str,
    requested_model: str | None,
    session_state: ConversationPrivacyState,
    retrieved_hits: Sequence[Any],
    *,
    allow_external_semi_private: bool,
    private_backend_policy: PrivateBackendPolicy | None = None,
    backend_id: str | None = None,
    backend_kind: BackendKind | str | None = None,
    base_url: str | None = None,
    trusted_local_models: set[str] | None = None,
    user_message_tier: Any = None,
    attached_file_tier: Any = None,
    memory_tier: Any = None,
    tool_output_tier: Any = None,
) -> PrivacyDecision:
    hit_tier = max_data_tier(*(data_tier_for_hit(hit) for hit in retrieved_hits))
    effective_tier = max_data_tier(
        session_state.max_seen_tier,
        user_message_tier,
        hit_tier,
        attached_file_tier,
        memory_tier,
        tool_output_tier,
    )
    policy = private_backend_policy or PrivateBackendPolicy.from_values(trusted_local_models=trusted_local_models or set())
    backend = classify_backend(
        provider=requested_provider,
        model=requested_model,
        backend_id=backend_id,
        backend_kind=backend_kind,
        base_url=base_url,
    )
    provider_is_external = backend.kind == BackendKind.EXTERNAL_SAAS
    backend_is_private_capable = backend.kind in PRIVATE_CAPABLE_BACKEND_KINDS
    private_backend_allowed = policy.is_private_allowed(backend)
    reason = "public_allows_requested_provider"
    provider_allowed = True
    external_blocked = False
    selected_provider: str | None = requested_provider
    selected_model: str | None = requested_model

    if effective_tier == DataTier.SEMI_PRIVATE and provider_is_external and not allow_external_semi_private:
        provider_allowed = False
        external_blocked = True
        selected_provider = None
        selected_model = None
        reason = "semi_private_requires_private_backend_or_approved_external_saas"
    elif effective_tier == DataTier.PRIVATE:
        if provider_is_external:
            provider_allowed = False
            external_blocked = True
            selected_provider = None
            selected_model = None
            reason = "private_taint_blocks_external_saas_backend"
        elif not backend_is_private_capable:
            provider_allowed = False
            external_blocked = True
            selected_provider = None
            selected_model = None
            reason = "private_taint_requires_private_capable_backend"
        elif not private_backend_allowed:
            provider_allowed = False
            external_blocked = True
            selected_provider = None
            selected_model = None
            if policy.model_is_trusted_somewhere(requested_model):
                reason = "private_taint_requires_trusted_private_backend"
            else:
                reason = "private_taint_requires_trusted_private_model"
        else:
            reason = "private_uses_trusted_private_backend"
    elif effective_tier == DataTier.SEMI_PRIVATE:
        reason = (
            "semi_private_external_saas_allowed_by_config"
            if provider_is_external
            else "semi_private_uses_private_or_local_backend"
        )

    session_state.update(effective_tier, external_blocked=external_blocked, reason=reason)
    return PrivacyDecision(
        effective_tier=effective_tier,
        provider_allowed=provider_allowed,
        selected_provider=selected_provider,
        selected_model=selected_model,
        backend_id=backend.backend_id,
        backend_kind=backend.kind,
        external_blocked=external_blocked,
        reason=reason,
        redaction_required=effective_tier == DataTier.PRIVATE,
        provider_requested=requested_provider,
        model_requested=requested_model,
    )


def data_tier_for_hit(hit: Any) -> DataTier:
    explicit = getattr(hit, "data_tier", None)
    metadata = getattr(hit, "metadata", None) if not isinstance(hit, dict) else hit.get("metadata")
    if explicit not in (None, ""):
        return normalize_data_tier(explicit)
    if isinstance(hit, dict):
        for key in ("data_tier", "tier"):
            if hit.get(key) not in (None, ""):
                return normalize_data_tier(hit.get(key))
    if isinstance(metadata, dict):
        return infer_data_tier_from_metadata(metadata, default=DataTier.PRIVATE)
    return DataTier.PRIVATE


def safe_source_payload(
    hit: Any,
    *,
    include_private_text: bool = False,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    tier = data_tier_for_hit(hit)
    metadata = dict(getattr(hit, "metadata", {}) or {})
    doc_id = getattr(hit, "doc_id", None)
    rank = getattr(hit, "rank", None)
    score = getattr(hit, "score", None)
    title = getattr(hit, "title", "")
    text = getattr(hit, "text", "")
    data_tier = tier.value
    base = {
        "doc_id": doc_id,
        "rank": rank,
        "score": score,
        "title": title,
        "text": text,
        "data_tier": data_tier,
        "doc_type": getattr(hit, "doc_type", None) or metadata.get("doc_type") or metadata.get("kind"),
        "source_id": getattr(hit, "source_id", None) or metadata.get("source_id") or metadata.get("source_set"),
        "redacted": False,
        "metadata": metadata,
    }
    if extra:
        base.update(extra)
    if tier != DataTier.PRIVATE or include_private_text:
        return base
    safe_metadata = _redact_metadata(metadata)
    return {
        **base,
        "title": None,
        "text": None,
        "metadata": safe_metadata,
        "redacted": True,
        "text_redaction_reason": "private_source_payload_redacted",
    }


def redact_for_log(obj: Any, effective_tier: Any) -> Any:
    if normalize_data_tier(effective_tier) != DataTier.PRIVATE:
        return obj
    if isinstance(obj, str):
        return "[REDACTED_PRIVATE]"
    if isinstance(obj, dict):
        return {
            key: ("[REDACTED_PRIVATE]" if str(key) in _SENSITIVE_SOURCE_KEYS else redact_for_log(value, effective_tier))
            for key, value in obj.items()
        }
    if isinstance(obj, list):
        return [redact_for_log(item, effective_tier) for item in obj]
    return obj


def _redact_metadata(metadata: dict[str, Any]) -> dict[str, Any]:
    safe: dict[str, Any] = {}
    for key, value in metadata.items():
        key_text = str(key)
        if key_text in _SENSITIVE_SOURCE_KEYS:
            safe[key_text] = None
        elif key_text in {"kind", "doc_type", "data_tier", "source_id", "source_set", "schema_version", "letter"}:
            safe[key_text] = value
        elif key_text.startswith("dictionary_") and key_text not in {"dictionary_evidence_text"}:
            safe[key_text] = _redact_metadata_value(value)
    safe["data_tier"] = DataTier.PRIVATE.value
    return safe


def _redact_metadata_value(value: Any) -> Any:
    if isinstance(value, str):
        return value[:160]
    if isinstance(value, (int, float, bool)) or value is None:
        return value
    if isinstance(value, list):
        return [
            _redact_metadata_value(item)
            for item in value
            if not isinstance(item, dict) or "label" not in item
        ][:16]
    return None


def _string_list_or_none(value: Any) -> list[str] | None:
    if not isinstance(value, list):
        return None
    items = [str(item).strip() for item in value if str(item).strip()]
    return items or None


def _normalized_set(values: Sequence[str] | set[str]) -> set[str]:
    return {str(value).strip() for value in values if str(value).strip()}


def _is_localhost_url(value: str | None) -> bool:
    if not value:
        return False
    parsed = urlparse(str(value))
    host = (parsed.hostname or "").strip().lower()
    return host in {"localhost", "127.0.0.1", "::1"}
