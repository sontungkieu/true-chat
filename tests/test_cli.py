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
            "--no-dictionary-query-planner",
            "--enable-structured-evidence",
            "--structured-evidence-jsonl",
            "fixtures/structured.jsonl",
            "--structured-evidence-md",
            "fixtures/structured.md",
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
            "--private-backend",
            "office_llm_server",
            "--private-backend-kind",
            "self_hosted_private",
            "--private-backend-base-url",
            "http://10.0.0.5:8000/v1",
            "--trusted-private-models",
            "qwen2.5-32b",
            "--private-backend-model",
            "office_llm_server:qwen2.5-32b,llama-3.1-70b",
            "--trusted-local-models",
            "legacy-local",
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
    assert seen["config"].chat.enable_dictionary_query_planner is False
    assert seen["config"].chat.enable_structured_evidence is True
    assert seen["config"].chat.structured_evidence_jsonl == Path("fixtures/structured.jsonl")
    assert seen["config"].chat.structured_evidence_md == Path("fixtures/structured.md")
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
    assert seen["config"].chat.backend_id == "office_llm_server"
    assert seen["config"].chat.backend_kind == "self_hosted_private"
    assert seen["config"].chat.backend_base_url == "http://10.0.0.5:8000/v1"
    assert seen["config"].chat.trusted_private_backends == ("office_llm_server",)
    assert seen["config"].chat.trusted_private_models == ("qwen2.5-32b", "legacy-local")
    assert seen["config"].chat.backend_model_allowlist == {
        "office_llm_server": ("qwen2.5-32b", "llama-3.1-70b")
    }
    assert seen["config"].chat.trusted_local_models == ("legacy-local",)


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


def test_cli_model_bench_smoke_with_mocked_runner(monkeypatch, tmp_path: Path, capsys) -> None:
    seen = {}

    def fake_run_model_benchmark(config):
        seen["config"] = config
        return {
            "run_id": "bench-1",
            "output_dir": str(tmp_path / "runs" / "bench-1"),
            "scenario_count": 1,
            "request_count": 2,
        }

    monkeypatch.setattr(cli, "run_model_benchmark", fake_run_model_benchmark)

    exit_code = cli.main(
        [
            "model-bench",
            "--model",
            "Qwen/Qwen2.5-7B-Instruct",
            "--preset",
            "all",
            "--tensor-parallel-size",
            "auto",
            "--concurrency",
            "1,4,16",
            "--requests-per-scenario",
            "3",
            "--warmup-requests",
            "1",
            "--output-dir",
            str(tmp_path / "model_bench"),
            "--max-model-len",
            "8192",
            "--max-output-tokens",
            "128",
            "--vllm-arg=--dtype",
            "--vllm-arg",
            "auto",
        ]
    )

    captured = capsys.readouterr()
    assert exit_code == 0
    assert '"run_id": "bench-1"' in captured.out
    assert seen["config"].model == "Qwen/Qwen2.5-7B-Instruct"
    assert seen["config"].preset == "all"
    assert seen["config"].tensor_parallel_size == "auto"
    assert seen["config"].concurrency == (1, 4, 16)
    assert seen["config"].requests_per_scenario == 3
    assert seen["config"].max_model_len == 8192
    assert seen["config"].max_output_tokens == 128
    assert seen["config"].vllm_args == ("--dtype", "auto")


def test_cli_model_bench_accepts_existing_endpoint(monkeypatch) -> None:
    seen = {}

    def fake_run_model_benchmark(config):
        seen["config"] = config
        return {"run_id": "bench-1", "output_dir": "runs/model_bench/bench-1", "scenario_count": 1, "request_count": 1}

    monkeypatch.setattr(cli, "run_model_benchmark", fake_run_model_benchmark)

    exit_code = cli.main(
        [
            "model-bench",
            "--endpoint",
            "http://127.0.0.1:8000/v1",
            "--served-model-name",
            "my-model",
            "--preset",
            "standard",
            "--no-stream",
        ]
    )

    assert exit_code == 0
    assert seen["config"].endpoint == "http://127.0.0.1:8000/v1"
    assert seen["config"].served_model_name == "my-model"
    assert seen["config"].stream is False


def test_cli_model_bench_rejects_invalid_numeric_args(capsys) -> None:
    exit_code = cli.main(["model-bench", "--model", "m", "--requests-per-scenario", "0"])

    captured = capsys.readouterr()
    assert exit_code == 2
    assert "--requests-per-scenario must be positive" in captured.err
