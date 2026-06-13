from __future__ import annotations

from dataclasses import dataclass

from rag_bench.groq_client import FallbackChatClient, RoundRobinGroqClient
from rag_bench.secrets import ApiKey


class FakeRateLimitError(Exception):
    status_code = 429


class FakeOrganizationRestrictedError(Exception):
    status_code = 400

    def __str__(self) -> str:
        return (
            "Error code: 400 - {'error': {'message': 'Organization has been restricted.', "
            "'code': 'organization_restricted'}}"
        )


@dataclass
class FakeUsage:
    prompt_tokens: int = 10
    completion_tokens: int = 5
    total_tokens: int = 15


@dataclass
class FakeMessage:
    content: str


@dataclass
class FakeChoice:
    message: FakeMessage


@dataclass
class FakeResponse:
    choices: list[FakeChoice]
    usage: FakeUsage


class FakeCompletions:
    def __init__(self, alias: str, failures: dict[str, int | list[Exception]]) -> None:
        self.alias = alias
        self.failures = failures

    def create(self, **_: object) -> FakeResponse:
        failure = self.failures.get(self.alias, 0)
        if isinstance(failure, list) and failure:
            raise failure.pop(0)
        if isinstance(failure, int) and failure > 0:
            self.failures[self.alias] = failure - 1
            raise FakeRateLimitError("rate limited")
        return FakeResponse(choices=[FakeChoice(FakeMessage(f"answer from {self.alias}"))], usage=FakeUsage())


class FakeChat:
    def __init__(self, alias: str, failures: dict[str, int | list[Exception]]) -> None:
        self.completions = FakeCompletions(alias, failures)


class FakeClient:
    def __init__(self, alias: str, failures: dict[str, int | list[Exception]]) -> None:
        self.chat = FakeChat(alias, failures)


class FakeClock:
    def __init__(self) -> None:
        self.now_s = 0.0
        self.sleeps: list[float] = []

    def now(self) -> float:
        return self.now_s

    def sleep(self, seconds: float) -> None:
        self.sleeps.append(seconds)
        self.now_s += seconds


def test_round_robin_rotates_keys_between_calls() -> None:
    keys = [ApiKey("a", "secret-a"), ApiKey("b", "secret-b")]

    client = RoundRobinGroqClient(
        keys=keys,
        model="test-model",
        client_factory=lambda key, _timeout: FakeClient(key.alias, {}),
        sleep_fn=lambda _seconds: None,
    )

    first = client.generate([{"role": "user", "content": "q1"}])
    second = client.generate([{"role": "user", "content": "q2"}])

    assert first.key_alias == "a"
    assert second.key_alias == "b"
    assert first.output_tokens_per_s is not None
    assert first.output_tokens_per_s > 0
    assert dict(client.key_usage_counts) == {"a": 1, "b": 1}


def test_round_robin_rotates_on_retry() -> None:
    keys = [ApiKey("a", "secret-a"), ApiKey("b", "secret-b")]
    failures = {"a": 1}
    sleeps: list[float] = []
    client = RoundRobinGroqClient(
        keys=keys,
        model="test-model",
        max_retries=1,
        client_factory=lambda key, _timeout: FakeClient(key.alias, failures),
        sleep_fn=sleeps.append,
    )

    result = client.generate([{"role": "user", "content": "q"}])

    assert result.answer == "answer from b"
    assert result.key_alias == "b"
    assert result.attempted_aliases == ["a", "b"]
    assert result.rejected_aliases == []
    assert result.retry_count == 1
    assert result.rate_limited is False
    assert sleeps == []
    assert dict(client.key_usage_counts) == {"a": 1, "b": 1}


def test_round_robin_reports_rate_limit_after_retry_budget() -> None:
    keys = [ApiKey("a", "secret-a"), ApiKey("b", "secret-b")]
    failures = {"a": 1, "b": 1}
    client = RoundRobinGroqClient(
        keys=keys,
        model="test-model",
        max_retries=1,
        client_factory=lambda key, _timeout: FakeClient(key.alias, failures),
        sleep_fn=lambda _seconds: None,
    )

    result = client.generate([{"role": "user", "content": "q"}])

    assert result.answer == ""
    assert result.key_alias is None
    assert result.attempted_aliases == ["a", "b"]
    assert result.error_status_code == 429
    assert result.rate_limited is True


def test_fallback_chat_client_uses_fallback_on_rate_limit() -> None:
    primary = RoundRobinGroqClient(
        keys=[ApiKey("mimo", "primary-secret")],
        model="test-model",
        max_retries=0,
        client_factory=lambda key, _timeout: FakeClient(key.alias, {"mimo": 1}),
        sleep_fn=lambda _seconds: None,
    )
    fallback = RoundRobinGroqClient(
        keys=[ApiKey("mimo_payg", "payg-secret")],
        model="test-model",
        max_retries=0,
        client_factory=lambda key, _timeout: FakeClient(key.alias, {}),
        sleep_fn=lambda _seconds: None,
    )
    client = FallbackChatClient(primary=primary, fallback=fallback)

    result = client.generate([{"role": "user", "content": "q"}])

    assert result.answer == "answer from mimo_payg"
    assert result.key_alias == "mimo_payg"
    assert result.attempted_aliases == ["mimo", "mimo_payg"]
    assert result.retry_count == 1
    assert result.rate_limited is False
    assert dict(client.key_usage_counts) == {"mimo": 1, "mimo_payg": 1}


def test_fallback_chat_client_does_not_use_payg_when_primary_succeeds() -> None:
    primary = RoundRobinGroqClient(
        keys=[ApiKey("mimo", "primary-secret")],
        model="test-model",
        max_retries=0,
        client_factory=lambda key, _timeout: FakeClient(key.alias, {}),
        sleep_fn=lambda _seconds: None,
    )
    fallback = RoundRobinGroqClient(
        keys=[ApiKey("mimo_payg", "payg-secret")],
        model="test-model",
        max_retries=0,
        client_factory=lambda key, _timeout: FakeClient(key.alias, {}),
        sleep_fn=lambda _seconds: None,
    )
    client = FallbackChatClient(primary=primary, fallback=fallback)

    result = client.generate([{"role": "user", "content": "q"}])

    assert result.answer == "answer from mimo"
    assert result.key_alias == "mimo"
    assert result.attempted_aliases == ["mimo"]
    assert dict(client.key_usage_counts) == {"mimo": 1}


def test_fallback_chat_client_uses_fallback_when_primary_key_unavailable() -> None:
    primary = RoundRobinGroqClient(
        keys=[ApiKey("mimo", "primary-secret")],
        model="test-model",
        max_retries=0,
        client_factory=lambda key, _timeout: FakeClient(key.alias, {"mimo": [FakeOrganizationRestrictedError()]}),
        sleep_fn=lambda _seconds: None,
    )
    fallback = RoundRobinGroqClient(
        keys=[ApiKey("mimo_payg", "payg-secret")],
        model="test-model",
        max_retries=0,
        client_factory=lambda key, _timeout: FakeClient(key.alias, {}),
        sleep_fn=lambda _seconds: None,
    )
    client = FallbackChatClient(primary=primary, fallback=fallback)

    result = client.generate([{"role": "user", "content": "q"}])

    assert result.answer == "answer from mimo_payg"
    assert result.key_alias == "mimo_payg"
    assert result.attempted_aliases == ["mimo", "mimo_payg"]
    assert result.rejected_aliases == ["mimo"]


def test_round_robin_disables_restricted_key_and_tries_next_key() -> None:
    keys = [ApiKey("a", "secret-a"), ApiKey("b", "secret-b")]
    failures: dict[str, int | list[Exception]] = {"a": [FakeOrganizationRestrictedError()]}
    client = RoundRobinGroqClient(
        keys=keys,
        model="test-model",
        max_retries=0,
        client_factory=lambda key, _timeout: FakeClient(key.alias, failures),
        sleep_fn=lambda _seconds: None,
    )

    result = client.generate([{"role": "user", "content": "q"}])

    assert result.answer == "answer from b"
    assert result.key_alias == "b"
    assert result.attempted_aliases == ["a", "b"]
    assert result.rejected_aliases == ["a"]
    assert result.retry_count == 1
    assert client.scheduler.disabled_aliases() == ["a"]


def test_round_robin_reports_when_all_keys_are_restricted() -> None:
    keys = [ApiKey("a", "secret-a"), ApiKey("b", "secret-b")]
    failures: dict[str, int | list[Exception]] = {
        "a": [FakeOrganizationRestrictedError()],
        "b": [FakeOrganizationRestrictedError()],
    }
    client = RoundRobinGroqClient(
        keys=keys,
        model="test-model",
        max_retries=0,
        client_factory=lambda key, _timeout: FakeClient(key.alias, failures),
        sleep_fn=lambda _seconds: None,
    )

    result = client.generate([{"role": "user", "content": "q"}])

    assert result.answer == ""
    assert result.key_alias is None
    assert result.attempted_aliases == ["a", "b"]
    assert result.rejected_aliases == ["a", "b"]
    assert result.error_status_code == 400
    assert result.rate_limited is False
    assert "organization_restricted" in result.error
    assert "disabling aliases: a, b" in result.error


def test_scheduler_waits_when_all_per_key_token_buckets_are_full() -> None:
    keys = [ApiKey("a", "secret-a"), ApiKey("b", "secret-b")]
    clock = FakeClock()
    client = RoundRobinGroqClient(
        keys=keys,
        model="test-model",
        max_retries=0,
        key_tokens_per_minute=100,
        key_requests_per_minute=0,
        client_factory=lambda key, _timeout: FakeClient(key.alias, {}),
        sleep_fn=clock.sleep,
        time_fn=clock.now,
    )

    first = client.generate([{"role": "user", "content": "x"}], max_completion_tokens=60)
    second = client.generate([{"role": "user", "content": "x"}], max_completion_tokens=60)
    third = client.generate([{"role": "user", "content": "x"}], max_completion_tokens=60)

    assert first.key_alias == "a"
    assert second.key_alias == "b"
    assert third.key_alias == "a"
    assert third.scheduled_wait_s == 60.0
    assert clock.sleeps == [60.0]
    assert client.rate_limit_snapshot()["a"]["requests_used"] == 1


def test_scheduler_shared_scope_uses_one_bucket_for_all_keys() -> None:
    keys = [ApiKey("a", "secret-a"), ApiKey("b", "secret-b")]
    clock = FakeClock()
    client = RoundRobinGroqClient(
        keys=keys,
        model="test-model",
        max_retries=0,
        key_tokens_per_minute=100,
        key_requests_per_minute=0,
        rate_limit_scope="shared",
        client_factory=lambda key, _timeout: FakeClient(key.alias, {}),
        sleep_fn=clock.sleep,
        time_fn=clock.now,
    )

    first = client.generate([{"role": "user", "content": "x"}], max_completion_tokens=60)
    second = client.generate([{"role": "user", "content": "x"}], max_completion_tokens=60)

    assert first.key_alias == "a"
    assert second.key_alias == "b"
    assert second.scheduled_wait_s == 60.0
    assert clock.sleeps == [60.0]
    assert list(client.rate_limit_snapshot()) == ["shared"]
