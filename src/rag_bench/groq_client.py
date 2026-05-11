from __future__ import annotations

import re
import time
from collections import Counter
from dataclasses import dataclass, field
from typing import Any, Callable, Iterable

from rag_bench.secrets import ApiKey


@dataclass
class GenerationResult:
    answer: str
    key_alias: str | None
    attempted_aliases: list[str]
    latency_s: float
    retry_count: int
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    total_tokens: int | None = None
    error: str | None = None
    error_status_code: int | None = None
    rate_limited: bool = False
    estimated_tokens: int | None = None
    scheduled_wait_s: float = 0.0


@dataclass
class _UsageEvent:
    at_s: float
    tokens: int
    requests: int = 1


@dataclass
class _Reservation:
    key_alias: str
    bucket_id: str
    event: _UsageEvent
    estimated_tokens: int
    waited_s: float


@dataclass
class RateLimitScheduler:
    keys: list[ApiKey]
    tokens_per_minute: int = 6000
    requests_per_minute: int = 30
    scope: str = "per-key"
    window_s: float = 60.0
    sleep_fn: Callable[[float], None] = time.sleep
    time_fn: Callable[[], float] = time.monotonic
    _next_index: int = 0
    _events_by_bucket: dict[str, list[_UsageEvent]] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.scope not in {"per-key", "shared"}:
            raise ValueError("RateLimitScheduler scope must be 'per-key' or 'shared'")
        for key in self.keys:
            self._events_by_bucket.setdefault(self._bucket_id(key), [])

    def acquire(self, estimated_tokens: int) -> _Reservation:
        waited_s = 0.0
        estimated_tokens = max(1, estimated_tokens)
        while True:
            now_s = self.time_fn()
            self._prune(now_s)
            candidate = self._find_available_key(now_s, estimated_tokens)
            if candidate is not None:
                key_index, key = candidate
                self._next_index = key_index + 1
                bucket_id = self._bucket_id(key)
                event = _UsageEvent(at_s=now_s, tokens=estimated_tokens)
                self._events_by_bucket.setdefault(bucket_id, []).append(event)
                return _Reservation(
                    key_alias=key.alias,
                    bucket_id=bucket_id,
                    event=event,
                    estimated_tokens=estimated_tokens,
                    waited_s=waited_s,
                )

            sleep_s = self._next_available_delay(now_s, estimated_tokens)
            if sleep_s <= 0:
                sleep_s = 0.01
            self.sleep_fn(sleep_s)
            waited_s += sleep_s

    def commit(self, reservation: _Reservation, actual_tokens: int | None) -> None:
        if actual_tokens is None:
            return
        reservation.event.tokens = max(reservation.estimated_tokens, actual_tokens)

    def snapshot(self) -> dict[str, dict[str, float | int | str]]:
        now_s = self.time_fn()
        self._prune(now_s)
        snapshot: dict[str, dict[str, float | int | str]] = {}
        for bucket_id, events in self._events_by_bucket.items():
            snapshot[bucket_id] = {
                "scope": self.scope,
                "window_s": self.window_s,
                "tokens_per_minute": self.tokens_per_minute,
                "requests_per_minute": self.requests_per_minute,
                "tokens_used": sum(event.tokens for event in events),
                "requests_used": sum(event.requests for event in events),
            }
        return snapshot

    def _find_available_key(self, now_s: float, estimated_tokens: int) -> tuple[int, ApiKey] | None:
        for offset in range(len(self.keys)):
            index = (self._next_index + offset) % len(self.keys)
            key = self.keys[index]
            if self._can_admit(self._bucket_id(key), estimated_tokens):
                return index, key
        return None

    def _can_admit(self, bucket_id: str, estimated_tokens: int) -> bool:
        events = self._events_by_bucket.setdefault(bucket_id, [])
        used_tokens = sum(event.tokens for event in events)
        used_requests = sum(event.requests for event in events)
        if self.requests_per_minute > 0 and used_requests + 1 > self.requests_per_minute:
            return False
        if self.tokens_per_minute > 0 and used_tokens + estimated_tokens > self.tokens_per_minute:
            # Let a single oversized request through when the bucket is empty; waiting cannot make it fit.
            return used_tokens == 0 and not events
        return True

    def _next_available_delay(self, now_s: float, estimated_tokens: int) -> float:
        delays: list[float] = []
        for key in self.keys:
            bucket_id = self._bucket_id(key)
            events = self._events_by_bucket.setdefault(bucket_id, [])
            if not events:
                return 0.0
            if self.requests_per_minute > 0 and sum(event.requests for event in events) + 1 > self.requests_per_minute:
                delays.append(max(0.0, self.window_s - (now_s - events[0].at_s)))
            if self.tokens_per_minute > 0 and sum(event.tokens for event in events) + estimated_tokens > self.tokens_per_minute:
                delays.append(max(0.0, self.window_s - (now_s - events[0].at_s)))
        return min(delays) if delays else 0.0

    def _prune(self, now_s: float) -> None:
        for bucket_id, events in self._events_by_bucket.items():
            self._events_by_bucket[bucket_id] = [
                event for event in events if now_s - event.at_s < self.window_s
            ]

    def _bucket_id(self, key: ApiKey) -> str:
        return "shared" if self.scope == "shared" else key.alias


@dataclass
class RoundRobinGroqClient:
    keys: list[ApiKey]
    model: str
    max_retries: int = 2
    timeout_s: float = 60.0
    key_tokens_per_minute: int = 6000
    key_requests_per_minute: int = 30
    rate_limit_scope: str = "per-key"
    client_factory: Callable[[ApiKey, float], Any] | None = None
    sleep_fn: Callable[[float], None] = time.sleep
    time_fn: Callable[[], float] = time.monotonic
    _next_index: int = 0
    key_usage_counts: Counter[str] = field(default_factory=Counter)
    scheduler: RateLimitScheduler = field(init=False)

    def __post_init__(self) -> None:
        if not self.keys:
            raise ValueError("RoundRobinGroqClient requires at least one API key")
        self.scheduler = RateLimitScheduler(
            keys=self.keys,
            tokens_per_minute=self.key_tokens_per_minute,
            requests_per_minute=self.key_requests_per_minute,
            scope=self.rate_limit_scope,
            sleep_fn=self.sleep_fn,
            time_fn=self.time_fn,
        )

    def generate(
        self,
        messages: Iterable[dict[str, str]],
        *,
        temperature: float = 0.0,
        max_completion_tokens: int = 512,
    ) -> GenerationResult:
        started = time.perf_counter()
        attempted_aliases: list[str] = []
        last_error: str | None = None
        attempts_allowed = max(1, self.max_retries + 1)
        scheduled_wait_s = 0.0

        messages_list = list(messages)
        estimated_tokens = estimate_requested_tokens(messages_list, max_completion_tokens=max_completion_tokens)

        for attempt in range(attempts_allowed):
            reservation = self.scheduler.acquire(estimated_tokens)
            scheduled_wait_s += reservation.waited_s
            key = self._key_by_alias(reservation.key_alias)
            attempted_aliases.append(key.alias)
            self.key_usage_counts[key.alias] += 1
            client = self._build_client(key)
            try:
                response = client.chat.completions.create(
                    messages=messages_list,
                    model=self.model,
                    temperature=temperature,
                    max_completion_tokens=max_completion_tokens,
                )
                answer = response.choices[0].message.content or ""
                prompt_tokens, completion_tokens, total_tokens = _extract_usage(response)
                self.scheduler.commit(reservation, total_tokens)
                return GenerationResult(
                    answer=answer,
                    key_alias=key.alias,
                    attempted_aliases=attempted_aliases,
                    latency_s=time.perf_counter() - started,
                    retry_count=len(attempted_aliases) - 1,
                    prompt_tokens=prompt_tokens,
                    completion_tokens=completion_tokens,
                    total_tokens=total_tokens,
                    estimated_tokens=estimated_tokens,
                    scheduled_wait_s=scheduled_wait_s,
                )
            except Exception as exc:  # noqa: BLE001 - SDK exception classes vary by version.
                last_error = _safe_error(exc)
                if not _is_retryable(exc) or attempt == attempts_allowed - 1:
                    return GenerationResult(
                        answer="",
                        key_alias=None,
                        attempted_aliases=attempted_aliases,
                        latency_s=time.perf_counter() - started,
                        retry_count=len(attempted_aliases) - 1,
                        error=last_error,
                        error_status_code=_status_code(exc),
                        rate_limited=_is_rate_limit(exc),
                        estimated_tokens=estimated_tokens,
                        scheduled_wait_s=scheduled_wait_s,
                    )
                delay = _retry_delay(exc, attempt)
                if delay > 0:
                    self.sleep_fn(delay)

        return GenerationResult(
            answer="",
            key_alias=None,
            attempted_aliases=attempted_aliases,
            latency_s=time.perf_counter() - started,
            retry_count=max(0, len(attempted_aliases) - 1),
            error=last_error or "unknown Groq error",
            estimated_tokens=estimated_tokens,
            scheduled_wait_s=scheduled_wait_s,
        )

    def _key_by_alias(self, alias: str) -> ApiKey:
        for key in self.keys:
            if key.alias == alias:
                return key
        raise KeyError(f"Unknown key alias: {alias}")

    def rate_limit_snapshot(self) -> dict[str, dict[str, float | int | str]]:
        return self.scheduler.snapshot()

    def _build_client(self, key: ApiKey) -> Any:
        if self.client_factory is not None:
            return self.client_factory(key, self.timeout_s)
        from groq import Groq

        return Groq(api_key=key.value, timeout=self.timeout_s, max_retries=0)


def _extract_usage(response: Any) -> tuple[int | None, int | None, int | None]:
    usage = getattr(response, "usage", None)
    if usage is None:
        return None, None, None
    return (
        getattr(usage, "prompt_tokens", None),
        getattr(usage, "completion_tokens", None),
        getattr(usage, "total_tokens", None),
    )


def estimate_requested_tokens(
    messages: Iterable[dict[str, str]],
    *,
    max_completion_tokens: int,
) -> int:
    messages_list = list(messages)
    content_chars = sum(len(message.get("content", "")) for message in messages_list)
    role_overhead = 4 * len(messages_list)
    estimated_prompt_tokens = max(1, (content_chars + 3) // 4 + role_overhead)
    return estimated_prompt_tokens + max(0, max_completion_tokens)


def _is_retryable(exc: Exception) -> bool:
    if _is_rate_limit(exc):
        return True
    status_code = _status_code(exc)
    if isinstance(status_code, int) and status_code >= 500:
        return True
    name = exc.__class__.__name__.lower()
    retryable_names = ("rate", "timeout", "connection", "temporar", "server")
    return any(token in name for token in retryable_names)


def _is_rate_limit(exc: Exception) -> bool:
    return _status_code(exc) == 429


def _status_code(exc: Exception) -> int | None:
    status_code = getattr(exc, "status_code", None)
    if isinstance(status_code, int):
        return status_code
    response = getattr(exc, "response", None)
    response_status = getattr(response, "status_code", None)
    return response_status if isinstance(response_status, int) else None


def _retry_delay(exc: Exception, attempt: int) -> float:
    if _is_rate_limit(exc):
        return 0.0
    retry_after = getattr(exc, "retry_after", None)
    if retry_after is not None:
        try:
            return min(float(retry_after), 30.0)
        except (TypeError, ValueError):
            pass
    return min(2.0**attempt, 10.0)


def _safe_error(exc: Exception) -> str:
    status_code = _status_code(exc)
    prefix = f"status={status_code} " if status_code is not None else ""
    message = f"{prefix}{exc.__class__.__name__}: {exc}"
    return re.sub(r"gsk_[A-Za-z0-9_-]+", "gsk_***REDACTED***", message)
