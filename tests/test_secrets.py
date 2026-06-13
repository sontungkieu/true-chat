from pathlib import Path

import pytest

from rag_bench.secrets import SecretFormatError, load_env_api_key, load_env_api_key_chain, load_groq_keys


def test_load_groq_keys_parses_aliases_without_exposing_values(tmp_path: Path) -> None:
    path = tmp_path / "groq.env"
    path.write_text(
        """
        # ignored
        primary='gsk_secret_one'
        export secondary="gsk_secret_two"
        """,
        encoding="utf-8",
    )

    keys = load_groq_keys(path)

    assert [key.alias for key in keys] == ["primary", "secondary"]
    assert [key.value for key in keys] == ["gsk_secret_one", "gsk_secret_two"]


def test_load_groq_keys_rejects_empty_file(tmp_path: Path) -> None:
    path = tmp_path / "groq.env"
    path.write_text("# no keys\n", encoding="utf-8")

    with pytest.raises(SecretFormatError, match="No Groq API keys"):
        load_groq_keys(path)


def test_load_groq_keys_rejects_malformed_line_without_secret_leak(tmp_path: Path) -> None:
    path = tmp_path / "groq.env"
    path.write_text("primary=gsk_secret_one\nnot-an-assignment\n", encoding="utf-8")

    with pytest.raises(SecretFormatError) as exc_info:
        load_groq_keys(path)

    assert "gsk_secret_one" not in str(exc_info.value)
    assert "line 2" in str(exc_info.value)


def test_load_env_api_key_reads_named_variable_without_exposing_value(tmp_path: Path) -> None:
    path = tmp_path / ".env"
    path.write_text("OTHER=value\nexport MIMO_API_KEY='secret-mimo'\n", encoding="utf-8")

    key = load_env_api_key(path, "MIMO_API_KEY", alias="mimo")

    assert key.alias == "mimo"
    assert key.value == "secret-mimo"


def test_load_env_api_key_fails_clearly_when_missing(tmp_path: Path) -> None:
    path = tmp_path / ".env"
    path.write_text("OTHER=value\n", encoding="utf-8")

    with pytest.raises(SecretFormatError, match="MIMO_API_KEY was not found"):
        load_env_api_key(path, "MIMO_API_KEY", alias="mimo")


def test_load_env_api_key_chain_keeps_primary_before_payg_fallback(tmp_path: Path) -> None:
    path = tmp_path / ".env"
    path.write_text(
        "MIMO_API_KEY=primary-secret\nMIMO_API_KEY_PAYG=payg-secret\n",
        encoding="utf-8",
    )

    keys = load_env_api_key_chain(
        path,
        "MIMO_API_KEY",
        primary_alias="mimo",
        fallback_variables=(("MIMO_API_KEY_PAYG", "mimo_payg"),),
    )

    assert [key.alias for key in keys] == ["mimo", "mimo_payg"]
    assert [key.value for key in keys] == ["primary-secret", "payg-secret"]


def test_load_env_api_key_chain_uses_process_env_when_file_missing(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setenv("MIMO_API_KEY", "primary-from-env")
    monkeypatch.setenv("MIMO_API_KEY_PAYG", "payg-from-env")

    keys = load_env_api_key_chain(
        tmp_path / "missing.env",
        "MIMO_API_KEY",
        primary_alias="mimo",
        fallback_variables=(("MIMO_API_KEY_PAYG", "mimo_payg"),),
    )

    assert [key.alias for key in keys] == ["mimo", "mimo_payg"]
    assert [key.value for key in keys] == ["primary-from-env", "payg-from-env"]
