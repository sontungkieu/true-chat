from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def _load_script(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


generation_matrix = _load_script(
    "run_budgetrag_generation_matrix",
    ROOT / "scripts" / "run_budgetrag_generation_matrix.py",
)


def test_generation_matrix_dry_run_records_models_and_skips_missing_mimo(tmp_path: Path, capsys) -> None:
    groq_keys = tmp_path / "groq.env"
    groq_keys.write_text("a=gsk_secret\n", encoding="utf-8")
    output_dir = tmp_path / "budgetrag"

    exit_code = generation_matrix.main(
        [
            "--bench",
            "scifact",
            "--limit",
            "3",
            "--retrievers",
            "bm25",
            "--models",
            "groq_llama8b,mimo_v25",
            "--context-policies",
            "legacy,evidence-aware,adaptive-heuristic",
            "--context-budgets",
            "1000",
            "--adaptive-profiles",
            "balanced",
            "--top-k",
            "3",
            "--max-completion-tokens",
            "128",
            "--groq-keys-path",
            str(groq_keys),
            "--mimo-env-file",
            str(tmp_path / "missing.env"),
            "--run-name",
            "phase1c3_dry_run",
            "--output-dir",
            str(output_dir),
            "--dry-run",
        ]
    )

    captured = capsys.readouterr()
    assert exit_code == 0
    assert "--generation-provider groq" in captured.out
    assert "--generation-model-role fast-small-baseline" in captured.out
    assert "--rate-limit-scope per-key" in captured.out
    assert "--adaptive-profile balanced" in captured.out
    assert "mimo_v25" in captured.out
    assert "MIMO_API_KEY-not-configured" in captured.out
    assert "gsk_secret" not in captured.out
    assert not output_dir.exists()


def test_generation_matrix_manifest_includes_mimo_when_env_file_is_present(tmp_path: Path) -> None:
    groq_keys = tmp_path / "groq.env"
    groq_keys.write_text("a=gsk_secret\n", encoding="utf-8")
    mimo_env = tmp_path / ".env"
    mimo_env.write_text("MIMO_API_KEY=mimo_secret\n", encoding="utf-8")
    args = generation_matrix._build_parser().parse_args(
        [
            "--models",
            "mimo_v25",
            "--context-policies",
            "adaptive-heuristic",
            "--context-budgets",
            "2000",
            "--adaptive-profiles",
            "aggressive",
            "--groq-keys-path",
            str(groq_keys),
            "--mimo-env-file",
            str(mimo_env),
        ]
    )
    configs = generation_matrix.load_generation_model_configs(args.model_config)
    selected, skipped = generation_matrix.select_generation_model_configs(configs, ["mimo_v25"])
    available, unavailable = generation_matrix._filter_available_models(selected, args)

    jobs = generation_matrix.build_generation_matrix_jobs(
        args,
        retrievers=["bm25"],
        models=available,
        policies=["adaptive-heuristic"],
        budgets=[2000],
        adaptive_profiles=["aggressive"],
        matrix_dir=tmp_path / "matrix",
    )

    assert skipped == []
    assert unavailable == []
    assert len(jobs) == 1
    assert "--generation-provider" in jobs[0].command
    assert "mimo" in jobs[0].command
    assert "--mimo-api-key-var" in jobs[0].command
    assert "mimo_secret" not in json.dumps(jobs[0].command)


def test_generation_matrix_skips_existing_completed_jobs(tmp_path: Path, monkeypatch, capsys) -> None:
    groq_keys = tmp_path / "groq.env"
    groq_keys.write_text("a=gsk_secret\n", encoding="utf-8")
    output_dir = tmp_path / "budgetrag"
    run_name = "resume"
    job_dir = (
        output_dir
        / run_name
        / generation_matrix._job_slug("scifact", "bm25", "groq_llama8b", "legacy", 1000)
        / "20260101T000000Z_scifact_bm25"
    )
    job_dir.mkdir(parents=True)
    (job_dir / "metrics.json").write_text("{}", encoding="utf-8")
    calls: list[list[str]] = []

    def fake_run(command, check=False):  # noqa: ANN001 - subprocess-compatible test double.
        calls.append(command)
        return type("Completed", (), {"returncode": 0})()

    monkeypatch.setattr(generation_matrix.subprocess, "run", fake_run)

    exit_code = generation_matrix.main(
        [
            "--bench",
            "scifact",
            "--limit",
            "3",
            "--retrievers",
            "bm25",
            "--models",
            "groq_llama8b",
            "--context-policies",
            "legacy",
            "--context-budgets",
            "1000",
            "--adaptive-profiles",
            "balanced",
            "--groq-keys-path",
            str(groq_keys),
            "--run-name",
            run_name,
            "--output-dir",
            str(output_dir),
        ]
    )

    captured = capsys.readouterr()
    manifest = json.loads((output_dir / run_name / "manifest.json").read_text(encoding="utf-8"))
    assert exit_code == 0
    assert calls == []
    assert "SKIP existing" in captured.out
    assert manifest["skipped_existing"][0]["reason"] == "existing-metrics"


def test_generation_matrix_records_job_timeout(tmp_path: Path, monkeypatch) -> None:
    groq_keys = tmp_path / "groq.env"
    groq_keys.write_text("a=gsk_secret\n", encoding="utf-8")
    output_dir = tmp_path / "budgetrag"

    def fake_run(command, check=False, timeout=None):  # noqa: ANN001 - subprocess-compatible test double.
        raise generation_matrix.subprocess.TimeoutExpired(command, timeout)

    monkeypatch.setattr(generation_matrix.subprocess, "run", fake_run)

    exit_code = generation_matrix.main(
        [
            "--bench",
            "scifact",
            "--limit",
            "3",
            "--retrievers",
            "bm25",
            "--models",
            "groq_llama8b",
            "--context-policies",
            "legacy",
            "--context-budgets",
            "1000",
            "--adaptive-profiles",
            "balanced",
            "--groq-keys-path",
            str(groq_keys),
            "--run-name",
            "timeouts",
            "--output-dir",
            str(output_dir),
            "--job-timeout-s",
            "5",
            "--continue-on-error",
        ]
    )

    manifest = json.loads((output_dir / "timeouts" / "manifest.json").read_text(encoding="utf-8"))
    assert exit_code == 0
    assert manifest["failures"][0]["returncode"] == "timeout"
    assert manifest["failures"][0]["timed_out"] is True
