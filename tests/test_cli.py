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
