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
    assert "mimo_secret_value" not in notebook_text
    assert "--max-action-rows" in notebook_text
