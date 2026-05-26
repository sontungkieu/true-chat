from __future__ import annotations

from typing import Any


KV_ESTIMATE_NOTE = "analytical estimate from estimated context tokens, not measured runtime memory"

KV_MODEL_PROFILES: dict[str, dict[str, int | str]] = {
    "generic-small": {
        "layers": 32,
        "heads": 32,
        "head_dim": 128,
        "dtype_bytes": 2,
    },
    "qwen2.5-14b": {
        "layers": 48,
        "heads": 40,
        "head_dim": 128,
        "dtype_bytes": 2,
        "note": "approximate profile for analytical comparison only",
    },
}


def estimate_kv_cache_bytes(sequence_length: int, *, profile: str = "generic-small") -> int:
    if sequence_length < 0:
        raise ValueError("sequence_length must be non-negative")
    model_profile = _profile(profile)
    return int(
        2
        * int(model_profile["layers"])
        * int(model_profile["heads"])
        * int(model_profile["head_dim"])
        * sequence_length
        * int(model_profile["dtype_bytes"])
    )


def estimate_kv_cache_mb(sequence_length: int, *, profile: str = "generic-small") -> float:
    return estimate_kv_cache_bytes(sequence_length, profile=profile) / (1024 * 1024)


def estimate_kv_cache_savings(
    *,
    before_tokens: int,
    after_tokens: int,
    profile: str = "generic-small",
) -> dict[str, Any]:
    if before_tokens < 0 or after_tokens < 0:
        raise ValueError("before_tokens and after_tokens must be non-negative")
    if after_tokens > before_tokens:
        raise ValueError("after_tokens must be less than or equal to before_tokens")
    before_mb = estimate_kv_cache_mb(before_tokens, profile=profile)
    after_mb = estimate_kv_cache_mb(after_tokens, profile=profile)
    savings_mb = before_mb - after_mb
    savings_ratio = savings_mb / before_mb if before_mb else 0.0
    profile_note = _profile(profile).get("note")
    note = KV_ESTIMATE_NOTE if not profile_note else f"{KV_ESTIMATE_NOTE}; {profile_note}"
    return {
        "profile": profile,
        "before_mb": before_mb,
        "after_mb": after_mb,
        "savings_mb": savings_mb,
        "savings_ratio": savings_ratio,
        "note": note,
    }


def _profile(name: str) -> dict[str, int | str]:
    try:
        return KV_MODEL_PROFILES[name]
    except KeyError as exc:
        allowed = ", ".join(sorted(KV_MODEL_PROFILES))
        raise ValueError(f"Unknown KV profile '{name}'. Expected one of: {allowed}") from exc
