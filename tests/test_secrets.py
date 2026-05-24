from pathlib import Path

import pytest

from rag_bench.secrets import SecretFormatError, load_env_api_key, load_groq_keys


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
