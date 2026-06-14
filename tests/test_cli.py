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
            "--mimo-payg-base-url",
            "https://api.xiaomimimo.com/v1",
            "--mimo-auth-header",
            "both",
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
    assert seen["config"].chat.mimo_payg_base_url == "https://api.xiaomimimo.com/v1"
    assert seen["config"].chat.mimo_auth_header == "both"
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


def test_cli_serve_accepts_none_benchmark_with_dictionary_model_id(monkeypatch) -> None:
    seen = {}

    def fake_serve_proxy(config):
        seen["config"] = config

    monkeypatch.setattr(cli, "serve_proxy", fake_serve_proxy)

    exit_code = cli.main(
        [
            "serve",
            "--bench",
            "none",
            "--retriever",
            "dictionary-graph",
            "--model-id",
            "rag-dictionary-graph",
            "--dictionary-artifact",
            "runs/dict",
        ]
    )

    assert exit_code == 0
    assert seen["config"].chat.bench == "none"
    assert seen["config"].chat.retriever == "dictionary-graph"
    assert seen["config"].chat.model_id == "rag-dictionary-graph"
    assert seen["config"].chat.dictionary_artifact == Path("runs/dict")


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


def test_cli_eval_rag_smoke_with_mocked_runner(monkeypatch, tmp_path: Path, capsys) -> None:
    seen = {}
    eval_set = tmp_path / "eval.jsonl"
    eval_set.write_text('{"eval_id":"one","query":"TERM_A"}\n', encoding="utf-8")

    def fake_run_rag_eval(config):
        seen["config"] = config
        return {
            "output_dir": str(tmp_path / "eval-out"),
            "results_path": str(tmp_path / "eval-out" / "results.jsonl"),
            "summary_path": str(tmp_path / "eval-out" / "summary.md"),
            "failures_path": str(tmp_path / "eval-out" / "failures.jsonl"),
            "item_count": 1,
            "failure_count": 0,
            "judge_called_count": 0,
        }

    monkeypatch.setattr(cli, "run_rag_eval", fake_run_rag_eval)

    exit_code = cli.main(
        [
            "eval-rag",
            "--eval-set",
            str(eval_set),
            "--out-dir",
            str(tmp_path / "eval-out"),
            "--structured-evidence-jsonl",
            str(tmp_path / "structured.jsonl"),
            "--generator-provider",
            "local_small",
            "--generator-model",
            "tiny-generator",
            "--generator-backend-id",
            "local_eval",
            "--generator-backend-kind",
            "local_process",
            "--generator-trusted-private-backend",
            "local_eval",
            "--generator-trusted-private-model",
            "tiny-generator",
            "--allow-external-semi-private",
            "--judge-provider",
            "mimo",
            "--judge-model",
            "mimo-v2.5",
            "--judge-backend-kind",
            "external_saas",
            "--allow-external-judge-public",
            "--enable-llm-judge",
            "--judge-max-completion-tokens",
            "3072",
            "--include-private-eval-text",
        ]
    )

    captured = capsys.readouterr()
    assert exit_code == 0
    assert '"item_count": 1' in captured.out
    assert seen["config"].generator_provider == "local_small"
    assert seen["config"].generator_model == "tiny-generator"
    assert seen["config"].generator_backend_id == "local_eval"
    assert seen["config"].generator_backend_kind == "local_process"
    assert seen["config"].chat_config.allow_external_semi_private is True
    assert seen["config"].judge_provider == "mimo"
    assert seen["config"].judge_model == "mimo-v2.5"
    assert seen["config"].judge_backend_kind == "external_saas"
    assert seen["config"].allow_external_judge_public is True
    assert seen["config"].disable_llm_judge is False
    assert seen["config"].judge_max_completion_tokens == 3072
    assert seen["config"].include_private_outputs is True
    assert seen["config"].chat_config.enable_structured_evidence is True
    assert seen["config"].chat_config.mimo_payg_base_url == "https://api.xiaomimimo.com/v1"
    assert seen["config"].chat_config.mimo_auth_header == "both"


def test_cli_eval_rag_accepts_redacted_smoke_fixture_benchmark(monkeypatch, tmp_path: Path, capsys) -> None:
    seen = {}

    def fake_run_rag_eval(config):
        seen["config"] = config
        return {
            "output_dir": str(tmp_path / "eval-out"),
            "results_path": str(tmp_path / "eval-out" / "results.jsonl"),
            "summary_path": str(tmp_path / "eval-out" / "summary.md"),
            "failures_path": str(tmp_path / "eval-out" / "failures.jsonl"),
            "item_count": 15,
            "failure_count": 0,
            "judge_called_count": 0,
        }

    monkeypatch.setattr(cli, "run_rag_eval", fake_run_rag_eval)

    fixture_dir = Path("tests/fixtures/rag_eval_smoke")
    pb_artifact = Path("runs/pb_dictionary_base_supp2021_prod_graph")
    exit_code = cli.main(
        [
            "eval-rag",
            "--bench",
            "fixture",
            "--eval-set",
            str(fixture_dir / "eval_public_smoke.jsonl"),
            "--out-dir",
            str(tmp_path / "eval-out"),
            "--dictionary-artifact",
            str(pb_artifact),
            "--dictionary-source-dir",
            str(fixture_dir / "no_source"),
            "--dictionary-letters",
            "T",
            "--dictionary-required",
            "--structured-evidence-jsonl",
            str(fixture_dir / "structured_evidence_public.jsonl"),
            "--generator-provider",
            "local",
            "--generator-model",
            "heuristic-local",
            "--disable-llm-judge",
        ]
    )

    captured = capsys.readouterr()
    assert exit_code == 0
    assert '"item_count": 15' in captured.out
    assert seen["config"].chat_config.bench == "fixture"
    assert seen["config"].chat_config.dictionary_artifact == pb_artifact
    assert seen["config"].chat_config.dictionary_required is True
    assert seen["config"].chat_config.structured_evidence_jsonl == fixture_dir / "structured_evidence_public.jsonl"
    assert seen["config"].disable_llm_judge is True


def test_cli_eval_rag_external_generator_defaults_to_external_saas(monkeypatch, tmp_path: Path) -> None:
    seen = {}
    eval_set = tmp_path / "eval.jsonl"
    eval_set.write_text('{"eval_id":"one","query":"TERM_A","data_tier":"semi_private"}\n', encoding="utf-8")

    def fake_run_rag_eval(config):
        seen["config"] = config
        return {
            "output_dir": str(tmp_path / "eval-out"),
            "results_path": str(tmp_path / "eval-out" / "results.jsonl"),
            "summary_path": str(tmp_path / "eval-out" / "summary.md"),
            "failures_path": str(tmp_path / "eval-out" / "failures.jsonl"),
            "item_count": 1,
            "failure_count": 0,
            "judge_called_count": 0,
        }

    monkeypatch.setattr(cli, "run_rag_eval", fake_run_rag_eval)

    assert cli.main(
        [
            "eval-rag",
            "--eval-set",
            str(eval_set),
            "--generator-provider",
            "deepseek",
            "--generator-model",
            "deepseek-v4-flash",
            "--allow-external-semi-private",
        ]
    ) == 0

    assert seen["config"].generator_provider == "deepseek"
    assert seen["config"].generator_backend_kind == "external_saas"
    assert seen["config"].chat_config.allow_external_semi_private is True
