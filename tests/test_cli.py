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
            "--context-policy",
            "evidence-aware",
            "--context-budget-chars",
            "1000",
            "--per-doc-budget-chars",
            "250",
            "--kv-profile",
            "qwen2.5-14b",
            "--adaptive-small-budget",
            "800",
            "--adaptive-medium-budget",
            "1600",
            "--adaptive-large-budget",
            "3200",
            "--adaptive-profile",
            "balanced",
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
    assert seen["config"].context_policy == "evidence-aware"
    assert seen["config"].context_budget_chars == 1000
    assert seen["config"].per_doc_budget_chars == 250
    assert seen["config"].record_context_metrics is True
    assert seen["config"].kv_profile == "qwen2.5-14b"
    assert seen["config"].disable_kv_estimate is False
    assert seen["config"].adaptive_small_budget == 800
    assert seen["config"].adaptive_medium_budget == 1600
    assert seen["config"].adaptive_large_budget == 3200
    assert seen["config"].adaptive_profile == "balanced"


def test_cli_run_rejects_invalid_context_budget(capsys) -> None:
    exit_code = cli.main(["run", "--context-budget-chars", "0"])

    captured = capsys.readouterr()
    assert exit_code == 2
    assert "--context-budget-chars must be positive" in captured.err


def test_cli_run_rejects_invalid_adaptive_budget(capsys) -> None:
    exit_code = cli.main(["run", "--context-policy", "adaptive-heuristic", "--adaptive-medium-budget", "0"])

    captured = capsys.readouterr()
    assert exit_code == 2
    assert "--adaptive-medium-budget must be positive" in captured.err


def test_cli_run_rejects_invalid_adaptive_profile(capsys) -> None:
    exit_code = cli.main(["run", "--context-policy", "adaptive-heuristic", "--adaptive-profile", "unknown"])

    captured = capsys.readouterr()
    assert exit_code == 2
    assert "invalid choice" in captured.err


def test_cli_rlaif_build_smoke_with_mocked_builder(monkeypatch, tmp_path: Path, capsys) -> None:
    seen = {}

    def fake_build_rlaif_dataset(config):
        seen["config"] = config
        return {
            "output_dir": str(tmp_path / "rlaif"),
            "action_count": 2,
            "feedback_count": 2,
            "invalid_row_count": 0,
            "feedback_provenance_counts": {"gold": 1, "missing": 1},
            "missing_reason_counts": {"generation_skipped": 1},
        }

    monkeypatch.setattr(cli, "build_rlaif_dataset", fake_build_rlaif_dataset)

    exit_code = cli.main(
        [
            "rlaif-build",
            "--inputs",
            str(tmp_path / "matrix"),
            "--output-dir",
            str(tmp_path / "out"),
        ]
    )

    captured = capsys.readouterr()
    assert exit_code == 0
    assert seen["config"].inputs == (tmp_path / "matrix",)
    assert seen["config"].output_dir == tmp_path / "out"
    assert '"action_count": 2' in captured.out
    assert '"gold": 1' in captured.out


def test_cli_rlaif_reward_smoke_with_mocked_builder(monkeypatch, tmp_path: Path, capsys) -> None:
    seen = {}

    def fake_build_rlaif_rewards(config):
        seen["config"] = config
        return {
            "output_dir": str(tmp_path / "rlaif"),
            "reward_count": 3,
            "scored_reward_count": 2,
            "preference_count": 1,
            "reward_mode_counts": {"gold": 2, "missing_quality": 1},
            "answer_label_count": 1,
            "answer_label_merge_counts": {"used_answer_label": 1},
            "preference_type_counts": {"context_policy_preference": 1},
            "preference_skip_reason_counts": {"missing_quality": 1},
            "invalid_row_count": 0,
        }

    monkeypatch.setattr(cli, "build_rlaif_rewards", fake_build_rlaif_rewards)

    exit_code = cli.main(
        [
            "rlaif-reward",
            "--actions",
            str(tmp_path / "rlaif_actions.jsonl"),
            "--feedback",
            str(tmp_path / "rlaif_feedback.jsonl"),
            "--answer-labels",
            str(tmp_path / "rlaif_answer_labels.jsonl"),
            "--output-dir",
            str(tmp_path / "out"),
            "--quality-weight",
            "0.7",
            "--min-reward-delta",
            "0.04",
        ]
    )

    captured = capsys.readouterr()
    assert exit_code == 0
    assert seen["config"].actions_path == tmp_path / "rlaif_actions.jsonl"
    assert seen["config"].feedback_path == tmp_path / "rlaif_feedback.jsonl"
    assert seen["config"].answer_labels_path == tmp_path / "rlaif_answer_labels.jsonl"
    assert seen["config"].output_dir == tmp_path / "out"
    assert seen["config"].quality_weight == 0.7
    assert seen["config"].min_reward_delta == 0.04
    assert '"reward_count": 3' in captured.out
    assert '"used_answer_label": 1' in captured.out
    assert '"context_policy_preference": 1' in captured.out


def test_cli_rlaif_reward_rejects_negative_weight(capsys) -> None:
    exit_code = cli.main(
        [
            "rlaif-reward",
            "--actions",
            "rlaif_actions.jsonl",
            "--feedback",
            "rlaif_feedback.jsonl",
            "--quality-weight",
            "-0.1",
        ]
    )

    captured = capsys.readouterr()
    assert exit_code == 2
    assert "--quality-weight must be non-negative" in captured.err


def test_cli_rlaif_split_smoke_with_mocked_splitter(monkeypatch, tmp_path: Path, capsys) -> None:
    seen = {}

    def fake_split_rlaif_by_query(config):
        seen["config"] = config
        return {
            "output_dir": str(tmp_path / "split"),
            "seed": 42,
            "train_ratio": 0.8,
            "train_query_count": 8,
            "eval_query_count": 2,
            "train_reward_rows": 80,
            "eval_reward_rows": 20,
            "train_preferences": 30,
            "eval_preferences": 5,
            "dropped_cross_split_preferences": 4,
            "dropped_missing_action_preferences": 1,
        }

    monkeypatch.setattr(cli, "split_rlaif_by_query", fake_split_rlaif_by_query)

    exit_code = cli.main(
        [
            "rlaif-split",
            "--rewards",
            str(tmp_path / "rlaif_rewards.jsonl"),
            "--preferences",
            str(tmp_path / "rlaif_preferences.jsonl"),
            "--output-dir",
            str(tmp_path / "split"),
            "--train-ratio",
            "0.8",
            "--seed",
            "42",
        ]
    )

    captured = capsys.readouterr()
    assert exit_code == 0
    assert seen["config"].rewards_path == tmp_path / "rlaif_rewards.jsonl"
    assert seen["config"].preferences_path == tmp_path / "rlaif_preferences.jsonl"
    assert seen["config"].output_dir == tmp_path / "split"
    assert seen["config"].train_ratio == 0.8
    assert seen["config"].seed == 42
    assert '"dropped_cross_split_preferences": 4' in captured.out


def test_cli_rlaif_split_rejects_invalid_train_ratio(capsys) -> None:
    exit_code = cli.main(
        [
            "rlaif-split",
            "--rewards",
            "rlaif_rewards.jsonl",
            "--preferences",
            "rlaif_preferences.jsonl",
            "--output-dir",
            "split",
            "--train-ratio",
            "1.0",
        ]
    )

    captured = capsys.readouterr()
    assert exit_code == 2
    assert "--train-ratio must be greater than 0 and less than 1" in captured.err


def test_cli_rlaif_label_answers_smoke_with_mocked_labeler(monkeypatch, tmp_path: Path, capsys) -> None:
    seen = {}

    def fake_label_rlaif_answers(config):
        seen["config"] = config
        return {
            "output_path": str(tmp_path / "labels.jsonl"),
            "action_count": 10,
            "processed_count": 3,
            "skipped_resume_count": 1,
            "skipped_limit_count": 0,
            "invalid_json_count": 0,
            "missing_input_count": 0,
            "error_count": 0,
            "stopped_early": False,
            "stop_reason": None,
            "dry_run": True,
            "judge_provider": "mimo",
            "judge_model": "mimo-v2.5-pro",
        }

    monkeypatch.setattr(cli, "label_rlaif_answers", fake_label_rlaif_answers)

    exit_code = cli.main(
        [
            "rlaif-label-answers",
            "--actions",
            str(tmp_path / "rlaif_actions.jsonl"),
            "--output",
            str(tmp_path / "labels.jsonl"),
            "--judge-provider",
            "mimo",
            "--judge-model",
            "mimo-v2.5-pro",
            "--dry-run",
            "--resume",
            "--limit",
            "3",
            "--max-errors",
            "2",
            "--sleep-seconds",
            "0.5",
            "--json-retries",
            "2",
            "--max-context-chars",
            "9000",
        ]
    )

    captured = capsys.readouterr()
    assert exit_code == 0
    assert seen["config"].actions_path == tmp_path / "rlaif_actions.jsonl"
    assert seen["config"].output_path == tmp_path / "labels.jsonl"
    assert seen["config"].dry_run is True
    assert seen["config"].resume is True
    assert seen["config"].limit == 3
    assert seen["config"].max_errors == 2
    assert seen["config"].sleep_seconds == 0.5
    assert seen["config"].json_retries == 2
    assert seen["config"].max_context_chars == 9000
    assert '"processed_count": 3' in captured.out


def test_cli_rlaif_label_answers_rejects_negative_limit(capsys) -> None:
    exit_code = cli.main(
        [
            "rlaif-label-answers",
            "--actions",
            "rlaif_actions.jsonl",
            "--output",
            "labels.jsonl",
            "--limit",
            "-1",
        ]
    )

    captured = capsys.readouterr()
    assert exit_code == 2
    assert "--limit must be non-negative" in captured.err


def test_cli_rlaif_label_contexts_smoke_with_mocked_labeler(monkeypatch, tmp_path: Path, capsys) -> None:
    seen = {}

    def fake_label_rlaif_contexts(config):
        seen["config"] = config
        return {
            "output_path": str(tmp_path / "context_labels.jsonl"),
            "action_count": 10,
            "processed_count": 3,
            "skipped_resume_count": 1,
            "skipped_limit_count": 0,
            "ambiguous_count": 2,
            "invalid_json_count": 0,
            "missing_input_count": 0,
            "error_count": 0,
            "stopped_early": False,
            "stop_reason": None,
            "dry_run": True,
            "judge_provider": "mimo",
            "judge_model": "mimo-v2.5-pro",
        }

    monkeypatch.setattr(cli, "label_rlaif_contexts", fake_label_rlaif_contexts)

    exit_code = cli.main(
        [
            "rlaif-label-contexts",
            "--actions",
            str(tmp_path / "rlaif_actions.jsonl"),
            "--output",
            str(tmp_path / "context_labels.jsonl"),
            "--judge-provider",
            "mimo",
            "--judge-model",
            "mimo-v2.5-pro",
            "--dry-run",
            "--resume",
            "--limit",
            "3",
            "--max-errors",
            "2",
            "--sleep-seconds",
            "0.5",
            "--json-retries",
            "2",
            "--max-context-chars",
            "9000",
        ]
    )

    captured = capsys.readouterr()
    assert exit_code == 0
    assert seen["config"].actions_path == tmp_path / "rlaif_actions.jsonl"
    assert seen["config"].output_path == tmp_path / "context_labels.jsonl"
    assert seen["config"].dry_run is True
    assert seen["config"].resume is True
    assert seen["config"].limit == 3
    assert seen["config"].max_errors == 2
    assert seen["config"].sleep_seconds == 0.5
    assert seen["config"].json_retries == 2
    assert seen["config"].max_context_chars == 9000
    assert '"ambiguous_count": 2' in captured.out


def test_cli_rlaif_label_contexts_rejects_negative_limit(capsys) -> None:
    exit_code = cli.main(
        [
            "rlaif-label-contexts",
            "--actions",
            "rlaif_actions.jsonl",
            "--output",
            "context_labels.jsonl",
            "--limit",
            "-1",
        ]
    )

    captured = capsys.readouterr()
    assert exit_code == 2
    assert "--limit must be non-negative" in captured.err


def test_cli_rlaif_label_pairs_smoke_with_mocked_labeler(monkeypatch, tmp_path: Path, capsys) -> None:
    seen = {}

    def fake_label_rlaif_pairs(config):
        seen["config"] = config
        return {
            "output_path": str(tmp_path / "pair_labels.jsonl"),
            "action_count": 10,
            "reward_count": 10,
            "preference_count": 6,
            "processed_count": 3,
            "skipped_resume_count": 1,
            "skipped_limit_count": 0,
            "ambiguous_count": 2,
            "tie_count": 1,
            "invalid_json_count": 0,
            "missing_input_count": 0,
            "error_count": 0,
            "stopped_early": False,
            "stop_reason": None,
            "dry_run": True,
            "judge_provider": "mimo",
            "judge_model": "mimo-v2.5-pro",
        }

    monkeypatch.setattr(cli, "label_rlaif_pairs", fake_label_rlaif_pairs)

    exit_code = cli.main(
        [
            "rlaif-label-pairs",
            "--actions",
            str(tmp_path / "rlaif_actions.jsonl"),
            "--rewards",
            str(tmp_path / "rlaif_rewards.jsonl"),
            "--preferences",
            str(tmp_path / "rlaif_preferences.jsonl"),
            "--output",
            str(tmp_path / "pair_labels.jsonl"),
            "--judge-provider",
            "mimo",
            "--judge-model",
            "mimo-v2.5-pro",
            "--dry-run",
            "--resume",
            "--limit",
            "3",
            "--max-errors",
            "2",
            "--sleep-seconds",
            "0.5",
            "--json-retries",
            "2",
            "--max-context-chars",
            "9000",
        ]
    )

    captured = capsys.readouterr()
    assert exit_code == 0
    assert seen["config"].actions_path == tmp_path / "rlaif_actions.jsonl"
    assert seen["config"].rewards_path == tmp_path / "rlaif_rewards.jsonl"
    assert seen["config"].preferences_path == tmp_path / "rlaif_preferences.jsonl"
    assert seen["config"].output_path == tmp_path / "pair_labels.jsonl"
    assert seen["config"].dry_run is True
    assert seen["config"].resume is True
    assert seen["config"].limit == 3
    assert seen["config"].max_errors == 2
    assert seen["config"].sleep_seconds == 0.5
    assert seen["config"].json_retries == 2
    assert seen["config"].max_context_chars == 9000
    assert '"tie_count": 1' in captured.out


def test_cli_rlaif_label_pairs_rejects_negative_limit(capsys) -> None:
    exit_code = cli.main(
        [
            "rlaif-label-pairs",
            "--actions",
            "rlaif_actions.jsonl",
            "--rewards",
            "rlaif_rewards.jsonl",
            "--preferences",
            "rlaif_preferences.jsonl",
            "--output",
            "pair_labels.jsonl",
            "--limit",
            "-1",
        ]
    )

    captured = capsys.readouterr()
    assert exit_code == 2
    assert "--limit must be non-negative" in captured.err


def test_cli_rlaif_train_smoke_with_mocked_trainer(monkeypatch, tmp_path: Path, capsys) -> None:
    seen = {}

    def fake_train_offline_selector_policies(config):
        seen["config"] = config
        return {
            "output_path": str(tmp_path / "rlaif_policy.json"),
            "policy_count": 4,
            "reward_count": 8,
            "scored_reward_count": 7,
            "preference_count": 3,
            "query_group_count": 2,
            "signature_count": 4,
            "runtime_default_replacement": False,
        }

    monkeypatch.setattr(cli, "train_offline_selector_policies", fake_train_offline_selector_policies)

    exit_code = cli.main(
        [
            "rlaif-train",
            "--rewards",
            str(tmp_path / "rlaif_rewards.jsonl"),
            "--preferences",
            str(tmp_path / "rlaif_preferences.jsonl"),
            "--output",
            str(tmp_path / "rlaif_policy.json"),
        ]
    )

    captured = capsys.readouterr()
    assert exit_code == 0
    assert seen["config"].rewards_path == tmp_path / "rlaif_rewards.jsonl"
    assert seen["config"].preferences_path == tmp_path / "rlaif_preferences.jsonl"
    assert seen["config"].output_path == tmp_path / "rlaif_policy.json"
    assert '"policy_count": 4' in captured.out
    assert '"runtime_default_replacement": false' in captured.out


def test_cli_rlaif_eval_smoke_with_mocked_evaluator(monkeypatch, tmp_path: Path, capsys) -> None:
    seen = {}

    def fake_evaluate_offline_selector_policies(config):
        seen["config"] = config
        return {
            "query_group_count": 2,
            "policy_metrics": {
                "fixed": {"coverage": 1.0, "mean_reward": 0.5},
                "cheapest": {"coverage": 1.0, "mean_reward": 0.6},
                "best_average": {"coverage": 1.0, "mean_reward": 0.7},
                "oracle_logged": {"coverage": 1.0, "mean_reward": 0.8},
            },
            "runtime_default_replacement": False,
            "held_out_query_eval": True,
            "split_manifest_path": str(tmp_path / "split_manifest.json"),
        }

    monkeypatch.setattr(cli, "evaluate_offline_selector_policies", fake_evaluate_offline_selector_policies)

    exit_code = cli.main(
        [
            "rlaif-eval",
            "--rewards",
            str(tmp_path / "rlaif_rewards.jsonl"),
            "--policy",
            str(tmp_path / "rlaif_policy.json"),
            "--out-md",
            str(tmp_path / "rlaif_eval_summary.md"),
            "--split-manifest",
            str(tmp_path / "split_manifest.json"),
        ]
    )

    captured = capsys.readouterr()
    assert exit_code == 0
    assert seen["config"].rewards_path == tmp_path / "rlaif_rewards.jsonl"
    assert seen["config"].policy_path == tmp_path / "rlaif_policy.json"
    assert seen["config"].out_md == tmp_path / "rlaif_eval_summary.md"
    assert seen["config"].split_manifest_path == tmp_path / "split_manifest.json"
    assert '"query_group_count": 2' in captured.out
    assert '"best_average"' in captured.out
    assert '"held_out_query_eval": true' in captured.out


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
            "--web-search-top-k",
            "8",
            "--web-search-timeout",
            "4.5",
            "--web-search-privilege-key",
            "search-secret",
            "--enable-image",
            "--image-top-k",
            "7",
            "--enable-dictionary",
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
    assert seen["config"].chat.available_retrievers == ("bm25", "keyword-match", "multi-query")
    assert seen["config"].chat.top_k == 4
    assert seen["config"].chat.max_context_chars == 2000
    assert seen["config"].chat.web_search_enabled is True
    assert seen["config"].chat.web_search_top_k == 8
    assert seen["config"].chat.web_search_timeout_s == 4.5
    assert seen["config"].chat.web_search_privilege_key == "search-secret"
    assert seen["config"].chat.image_enabled is True
    assert seen["config"].chat.image_top_k == 7
    assert seen["config"].chat.dictionary_enabled is True
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
    assert seen["config"].chat.key_tokens_per_minute == 5000
    assert seen["config"].chat.key_requests_per_minute == 20
    assert seen["config"].chat.rate_limit_scope == "shared"
