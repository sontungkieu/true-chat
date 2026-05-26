from __future__ import annotations

import pytest

from rag_bench.kv_estimator import estimate_kv_cache_bytes, estimate_kv_cache_savings


def test_kv_cache_bytes_grow_linearly_with_sequence_length() -> None:
    one = estimate_kv_cache_bytes(1, profile="generic-small")
    two = estimate_kv_cache_bytes(2, profile="generic-small")

    assert two == one * 2


def test_kv_cache_savings_are_non_negative_when_after_is_smaller() -> None:
    estimate = estimate_kv_cache_savings(before_tokens=100, after_tokens=40, profile="qwen2.5-14b")

    assert estimate["before_mb"] > estimate["after_mb"]
    assert estimate["savings_mb"] > 0
    assert 0 < estimate["savings_ratio"] < 1
    assert "analytical estimate" in estimate["note"]


def test_kv_cache_estimator_rejects_invalid_lengths() -> None:
    with pytest.raises(ValueError, match="sequence_length"):
        estimate_kv_cache_bytes(-1)
    with pytest.raises(ValueError, match="after_tokens"):
        estimate_kv_cache_savings(before_tokens=10, after_tokens=11)


def test_kv_cache_estimator_rejects_invalid_profile() -> None:
    with pytest.raises(ValueError, match="Unknown KV profile"):
        estimate_kv_cache_bytes(10, profile="missing")
