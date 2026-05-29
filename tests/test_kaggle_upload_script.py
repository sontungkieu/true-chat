from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path


SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts" / "upload_kaggle_rag_proxy_notebook.py"
SPEC = importlib.util.spec_from_file_location("upload_kaggle_rag_proxy_notebook", SCRIPT_PATH)
assert SPEC and SPEC.loader
upload_script = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = upload_script
SPEC.loader.exec_module(upload_script)


def test_load_kaggle_credential_supports_jsonl(tmp_path: Path) -> None:
    path = tmp_path / "all-kaggle.json"
    path.write_text(
        "\n".join(
            [
                json.dumps({"account": "other", "username": "other", "key": "x"}),
                json.dumps({"account": "codemaivanngu", "username": "codemaivanngu", "key": "secret-key"}),
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    credential = upload_script.load_kaggle_credential(path, "codemaivanngu")

    assert credential.username == "codemaivanngu"
    assert credential.key == "secret-key"


def test_build_notebook_checks_expected_commit_and_injects_tunnel_token() -> None:
    notebook = upload_script.build_notebook(
        repo_url="https://github.com/example/repo.git",
        repo_ref="main",
        expected_commit="abc123",
        cloudflare_token="cf-token",
        hostname="https://chat.example.com",
    )
    source = "\n".join("".join(cell["source"]) for cell in notebook["cells"])

    assert "EXPECTED_COMMIT = 'abc123'" in source
    assert "actual_commit != EXPECTED_COMMIT" in source
    assert "CLOUDFLARE_TUNNEL_TOKEN = 'cf-token'" in source
    assert "PROXY_STARTUP_TIMEOUT_S = 900" in source
    assert "uv', 'sync', '--frozen', '--no-dev'" in source
    assert "uv', 'run', '--frozen', '--no-sync'" in source
    assert "'--model', 'qwen/qwen3-32b'" in source
    assert "'--max-completion-tokens', '4096'" in source
    assert "print_proxy_log_tail" in source
    assert "LOCAL_PATCH" not in source
    assert "UserSecretsClient" in source
    assert all(cell.get("id") for cell in notebook["cells"])


def test_build_notebook_can_embed_groq_key_env_payload() -> None:
    groq_payload = upload_script.base64.b64encode(b"primary=gsk_test\n").decode("ascii")

    notebook = upload_script.build_notebook(
        repo_url="https://github.com/example/repo.git",
        repo_ref="main",
        expected_commit="abc123",
        cloudflare_token="cf-token",
        hostname="https://chat.example.com",
        groq_key_env_b64=groq_payload,
    )
    source = "\n".join("".join(cell["source"]) for cell in notebook["cells"])

    assert f"GROQ_KEY_ENV_B64 = '{groq_payload}'" in source
    assert "base64.b64decode" in source
    assert "UserSecretsClient" not in source


def test_write_staging_files_creates_private_kernel_metadata(tmp_path: Path) -> None:
    upload_script.write_staging_files(
        tmp_path,
        kernel_id="codemaivanngu/true-chat-test",
        title="True Chat Test",
        repo_url="https://github.com/example/repo.git",
        repo_ref="main",
        expected_commit="abc123",
        cloudflare_token="cf-token",
        hostname="https://chat.example.com",
        proxy_startup_timeout_s=900,
        groq_key_env_b64=None,
    )

    metadata = json.loads((tmp_path / "kernel-metadata.json").read_text(encoding="utf-8"))
    notebook = json.loads((tmp_path / "true_chat_rag_proxy_kaggle.ipynb").read_text(encoding="utf-8"))

    assert metadata["id"] == "codemaivanngu/true-chat-test"
    assert metadata["is_private"] is True
    assert metadata["enable_internet"] is True
    assert notebook["nbformat"] == 4


def test_upload_registry_marks_deleted_records(tmp_path: Path) -> None:
    registry = tmp_path / "uploads.jsonl"
    upload_script.append_upload_registry(
        registry,
        {
            "created_at": "2026-05-13T00:00:00+00:00",
            "kernel_id": "codemaivanngu/true-chat-test",
            "expected_commit": "abc123",
            "embedded_groq_keys": True,
        },
    )

    assert upload_script.active_registry_kernel_ids(registry) == ["codemaivanngu/true-chat-test"]

    upload_script.mark_registry_deleted(registry, "codemaivanngu/true-chat-test")
    records = upload_script.read_upload_registry(registry)

    assert records[0]["deleted_at"]
    assert upload_script.active_registry_kernel_ids(registry) == []
