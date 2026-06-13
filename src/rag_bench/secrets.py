from __future__ import annotations

import os
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


def load_env_values(path: str | Path) -> dict[str, str]:
    """Load simple KEY=value env files without expanding or exposing values."""

    env_path = Path(path)
    if not env_path.exists():
        raise SecretFormatError(f"Env file does not exist: {env_path}")
    if not env_path.is_file():
        raise SecretFormatError(f"Env path is not a file: {env_path}")

    values: dict[str, str] = {}
    for line_no, raw_line in enumerate(env_path.read_text(encoding="utf-8").splitlines(), start=1):
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[len("export ") :].strip()
        if "=" not in line:
            raise SecretFormatError(f"Invalid env line {line_no}: expected key=value")
        key, value = line.split("=", 1)
        key = key.strip()
        if not key:
            raise SecretFormatError(f"Invalid env line {line_no}: empty key")
        values[key] = _strip_env_quotes(value.strip())
    return values


def load_env_api_key(path: str | Path, variable: str, *, alias: str) -> ApiKey:
    """Load one API key from an env file and return a log-safe alias."""

    variable = variable.strip()
    if not variable:
        raise SecretFormatError("API key variable name must not be empty")
    values = load_env_values(path)
    value = values.get(variable)
    if not value:
        raise SecretFormatError(f"{variable} was not found in {Path(path)}")
    return ApiKey(alias=alias, value=value)


def load_env_api_key_chain(
    path: str | Path,
    primary_variable: str,
    *,
    primary_alias: str,
    fallback_variables: tuple[tuple[str, str], ...] = (),
) -> list[ApiKey]:
    """Load an ordered API key chain from process env and an env file.

    Process environment values take precedence over the env file. Secret values are
    never included in errors; callers use aliases for logging/metadata.
    """

    primary_variable = primary_variable.strip()
    if not primary_variable:
        raise SecretFormatError("API key variable name must not be empty")
    variables = ((primary_variable, primary_alias), *fallback_variables)
    env_path = Path(path)
    values: dict[str, str] = {}
    if env_path.exists():
        values = load_env_values(env_path)

    keys: list[ApiKey] = []
    seen_aliases: set[str] = set()
    seen_variables: set[str] = set()
    for variable, alias in variables:
        variable = variable.strip()
        alias = alias.strip()
        if not variable or not alias:
            raise SecretFormatError("API key variable and alias names must not be empty")
        if variable in seen_variables:
            continue
        seen_variables.add(variable)
        value = os.getenv(variable) or values.get(variable)
        if not value:
            continue
        if alias in seen_aliases:
            raise SecretFormatError(f"Duplicate API key alias: {alias}")
        seen_aliases.add(alias)
        keys.append(ApiKey(alias=alias, value=value))

    if keys:
        return keys
    if not env_path.exists():
        raise SecretFormatError(f"Env file does not exist: {env_path}")
    raise SecretFormatError(f"{primary_variable} was not found in {env_path}")


def _strip_env_quotes(value: str) -> str:
    if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
        return value[1:-1]
    return value
