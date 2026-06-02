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


upload_eval = _load_script(
    "upload_kaggle_budgetrag_eval_notebook",
    ROOT / "scripts" / "upload_kaggle_budgetrag_eval_notebook.py",
)


def test_kaggle_eval_staging_metadata_is_private_and_secret_free(tmp_path: Path) -> None:
    upload_eval.write_staging_files(
        tmp_path,
        kernel_id="codemaivanngu/test-hotpotqa",
        title="HotpotQA Test",
        repo_url="https://example.com/repo.git",
        repo_ref="main",
        expected_commit="abc123",
        run_name="fixture",
        limit=5,
        top_k=10,
        max_action_rows=2,
        ragas_samples_per_action=1,
        mimo_secret_name="MIMO_API_KEY",
        skip_ragas=False,
    )

    metadata = json.loads((tmp_path / "kernel-metadata.json").read_text(encoding="utf-8"))
    notebook_text = (tmp_path / "hotpotqa_budgetrag_eval.ipynb").read_text(encoding="utf-8")

    assert metadata["is_private"] == "true"
    assert metadata["enable_internet"] == "true"
    assert metadata["enable_gpu"] == "false"
    assert "MIMO_API_KEY" in notebook_text
    assert "UserSecretsClient" not in notebook_text
    assert "mimo_secret_value" not in notebook_text
    assert "--max-action-rows" in notebook_text


def test_kaggle_eval_can_embed_minimal_mimo_env_without_raw_secret(tmp_path: Path) -> None:
    env_file = tmp_path / ".env"
    env_file.write_text("MIMO_API_KEY=secret-value\nOTHER_SECRET=do-not-copy\n", encoding="utf-8")
    mimo_env_b64 = upload_eval.read_mimo_env_b64(tmp_path, env_file, api_key_var="MIMO_API_KEY")

    upload_eval.write_staging_files(
        tmp_path / "staging",
        kernel_id="codemaivanngu/test-hotpotqa",
        title="HotpotQA Test",
        repo_url="https://example.com/repo.git",
        repo_ref="main",
        expected_commit="abc123",
        run_name="fixture",
        limit=5,
        top_k=10,
        max_action_rows=2,
        ragas_samples_per_action=1,
        mimo_secret_name="MIMO_API_KEY",
        mimo_env_b64=mimo_env_b64,
        skip_ragas=False,
    )

    notebook_text = (tmp_path / "staging" / "hotpotqa_budgetrag_eval.ipynb").read_text(encoding="utf-8")
    metadata_text = (tmp_path / "staging" / "kernel-metadata.json").read_text(encoding="utf-8")

    assert "MIMO_ENV_B64" in notebook_text
    assert "secret-value" not in notebook_text
    assert "OTHER_SECRET" not in notebook_text
    assert "secret-value" not in metadata_text


def test_kaggle_eval_can_embed_single_groq_key_without_raw_secret(tmp_path: Path) -> None:
    env_file = tmp_path / "groq_key.env"
    env_file.write_text("primary=groq-secret\nbackup=do-not-copy\n", encoding="utf-8")
    groq_env_b64 = upload_eval.read_groq_env_b64(tmp_path, env_file, key_alias="primary")
    mimo_env_file = tmp_path / ".env"
    mimo_env_file.write_text("MIMO_API_KEY=hidden-judge-key\n", encoding="utf-8")
    mimo_env_b64 = upload_eval.read_mimo_env_b64(tmp_path, mimo_env_file, api_key_var="MIMO_API_KEY")

    upload_eval.write_staging_files(
        tmp_path / "staging",
        kernel_id="codemaivanngu/test-hotpotqa-groq",
        title="HotpotQA Groq Test",
        repo_url="https://example.com/repo.git",
        repo_ref="main",
        expected_commit="abc123",
        run_name="fixture",
        limit=5,
        top_k=10,
        max_action_rows=2,
        provider="groq",
        model="qwen/qwen3-32b",
        model_role="stronger-baseline",
        key_tpm=6000,
        key_rpm=20,
        ragas_model="mimo-v2.5-pro",
        ragas_samples_per_action=1,
        mimo_secret_name="MIMO_API_KEY",
        mimo_env_b64=mimo_env_b64,
        groq_key_alias="primary",
        groq_env_b64=groq_env_b64,
        skip_ragas=False,
    )

    notebook_text = (tmp_path / "staging" / "hotpotqa_budgetrag_eval.ipynb").read_text(encoding="utf-8")
    metadata_text = (tmp_path / "staging" / "kernel-metadata.json").read_text(encoding="utf-8")

    assert "GROQ_ENV_B64" in notebook_text
    assert "UserSecretsClient" not in notebook_text
    assert "--provider" in notebook_text
    assert "groq" in notebook_text
    assert "qwen/qwen3-32b" in notebook_text
    assert "--ragas-model" in notebook_text
    assert "mimo-v2.5-pro" in notebook_text
    assert "--groq-key-alias" in notebook_text
    assert "primary" in notebook_text
    assert "groq-secret" not in notebook_text
    assert "hidden-judge-key" not in notebook_text
    assert "do-not-copy" not in notebook_text
    assert "groq-secret" not in metadata_text
    assert "hidden-judge-key" not in metadata_text
