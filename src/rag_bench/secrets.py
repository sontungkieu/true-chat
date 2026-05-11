from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


class SecretFormatError(ValueError):
    """Raised when a Groq key env file is missing or malformed."""


@dataclass(frozen=True)
class ApiKey:
    alias: str
    value: str


def load_groq_keys(path: str | Path) -> list[ApiKey]:
    """Load alias=value Groq keys without exposing secret values in errors."""

    key_path = Path(path)
    if not key_path.exists():
        raise SecretFormatError(f"Groq key file does not exist: {key_path}")
    if not key_path.is_file():
        raise SecretFormatError(f"Groq key path is not a file: {key_path}")

    keys: list[ApiKey] = []
    aliases: set[str] = set()
    for line_no, raw_line in enumerate(key_path.read_text(encoding="utf-8").splitlines(), start=1):
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[len("export ") :].strip()
        if "=" not in line:
            raise SecretFormatError(f"Invalid Groq key line {line_no}: expected alias=value")

        alias, value = line.split("=", 1)
        alias = alias.strip()
        value = _strip_env_quotes(value.strip())
        if not alias:
            raise SecretFormatError(f"Invalid Groq key line {line_no}: empty alias")
        if not alias.replace("_", "").replace("-", "").isalnum():
            raise SecretFormatError(
                f"Invalid Groq key line {line_no}: alias must contain only letters, numbers, '_' or '-'"
            )
        if alias in aliases:
            raise SecretFormatError(f"Duplicate Groq key alias on line {line_no}: {alias}")
        if not value:
            raise SecretFormatError(f"Invalid Groq key line {line_no}: empty value for alias {alias}")

        aliases.add(alias)
        keys.append(ApiKey(alias=alias, value=value))

    if not keys:
        raise SecretFormatError(f"No Groq API keys found in {key_path}")
    return keys


def _strip_env_quotes(value: str) -> str:
    if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
        return value[1:-1]
    return value
