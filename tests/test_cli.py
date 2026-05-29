from __future__ import annotations

from pathlib import Path

from rag_bench import cli


def test_cli_run_smoke_with_mocked_runner(monkeypatch, tmp_path: Path, capsys) -> None:
    seen = {}

    def fake_run_benchmark(config):
        seen["config"] = config
        return {
            "run_id": "run-1",
            "output_dir": str(tmp_path / "runs" / "run-1"),
            "stopped_early": False,
            "stop_reason": None,
        }

    monkeypatch.setattr(cli, "run_benchmark", fake_run_benchmark)

    exit_code = cli.main(
        [
            "run",
            "--bench",
            "scifact",
            "--retrievers",
            "bm25,vector",
            "--top-k",
            "5",
            "--limit",
            "10",
            "--output-dir",
            str(tmp_path / "runs"),
        ]
    )

    captured = capsys.readouterr()
    assert exit_code == 0
    assert '"run_id": "run-1"' in captured.out
    assert seen["config"].retrievers == ("bm25", "vector")
    assert seen["config"].top_k == 5
    assert seen["config"].limit == 10
    assert seen["config"].max_consecutive_errors == 3
    assert seen["config"].skip_generation is False
    assert seen["config"].sleep_between_queries_s == 0.0
    assert seen["config"].key_tokens_per_minute == 6000
    assert seen["config"].key_requests_per_minute == 30
    assert seen["config"].rate_limit_scope == "per-key"


def test_cli_serve_smoke_with_mocked_server(monkeypatch) -> None:
    seen = {}

    def fake_serve_proxy(config):
        seen["config"] = config

    monkeypatch.setattr(cli, "serve_proxy", fake_serve_proxy)

    exit_code = cli.main(
        [
            "serve",
            "--host",
            "0.0.0.0",
            "--port",
            "9000",
            "--api-key",
            "dev-secret",
            "--bench",
            "scifact",
            "--retriever",
            "bm25",
            "--available-retrievers",
            "bm25,keyword-match,multi-query",
            "--top-k",
            "4",
            "--max-context-chars",
            "2000",
            "--image-top-k",
            "7",
            "--dictionary-artifact",
            "runs/dict",
            "--dictionary-source-dir",
            "data/dict",
            "--dictionary-letters",
            "A,B",
            "--dictionary-top-k",
            "6",
            "--dictionary-required",
            "--max-completion-tokens",
            "96",
            "--enable-mimo",
            "--mimo-env-file",
            ".secrets/.env",
            "--mimo-api-key-var",
            "MIMO_API_KEY",
            "--mimo-base-url",
            "https://token-plan-sgp.xiaomimimo.com/v1",
            "--mimo-models",
            "mimo-v2.5-pro,mimo-v2.5",
            "--mimo-key-tpm",
            "0",
            "--mimo-key-rpm",
            "0",
            "--key-tpm",
            "5000",
            "--key-rpm",
            "20",
            "--rate-limit-scope",
            "shared",
        ]
    )

    assert exit_code == 0
    assert seen["config"].host == "0.0.0.0"
    assert seen["config"].port == 9000
    assert seen["config"].api_key == "dev-secret"
    assert seen["config"].chat.bench == "scifact"
    assert seen["config"].chat.retriever == "bm25"
    assert seen["config"].chat.model == "qwen/qwen3-32b"
    assert seen["config"].chat.available_retrievers == ("bm25", "keyword-match", "multi-query")
    assert seen["config"].chat.top_k == 4
    assert seen["config"].chat.max_context_chars == 2000
    assert seen["config"].chat.image_top_k == 7
    assert seen["config"].chat.dictionary_artifact == Path("runs/dict")
    assert seen["config"].chat.dictionary_source_dir == Path("data/dict")
    assert seen["config"].chat.dictionary_letters == ("A", "B")
    assert seen["config"].chat.dictionary_top_k == 6
    assert seen["config"].chat.dictionary_required is True
    assert seen["config"].chat.max_completion_tokens == 96
    assert seen["config"].chat.mimo_enabled is True
    assert seen["config"].chat.mimo_env_file == Path(".secrets/.env")
    assert seen["config"].chat.mimo_api_key_var == "MIMO_API_KEY"
    assert seen["config"].chat.mimo_base_url == "https://token-plan-sgp.xiaomimimo.com/v1"
    assert seen["config"].chat.mimo_models == ("mimo-v2.5-pro", "mimo-v2.5")
    assert seen["config"].chat.mimo_key_tokens_per_minute == 0
    assert seen["config"].chat.mimo_key_requests_per_minute == 0
    assert "mimo-v2.5-pro" in seen["config"].chat.available_models
    assert seen["config"].chat.available_models[0] == "qwen/qwen3-32b"
    assert seen["config"].chat.key_tokens_per_minute == 5000
    assert seen["config"].chat.key_requests_per_minute == 20
    assert seen["config"].chat.rate_limit_scope == "shared"


def test_cli_serve_defaults_use_qwen_and_long_completion(monkeypatch) -> None:
    seen = {}

    def fake_serve_proxy(config):
        seen["config"] = config

    monkeypatch.setattr(cli, "serve_proxy", fake_serve_proxy)

    exit_code = cli.main(["serve"])

    assert exit_code == 0
    assert seen["config"].chat.model == "qwen/qwen3-32b"
    assert seen["config"].chat.max_completion_tokens == 4096
    assert seen["config"].chat.available_models[0] == "qwen/qwen3-32b"
