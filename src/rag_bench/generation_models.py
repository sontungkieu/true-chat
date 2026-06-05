from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


DEFAULT_MODEL_CONFIG_PATH = Path("configs/budgetrag_models.json")


@dataclass(frozen=True)
class GenerationModelConfig:
    model_id: str
    provider: str
    model: str
    role: str
    enabled: bool = True
    base_url: str | None = None
    base_url_env: str | None = None
    api_key_env: str | None = None


def load_generation_model_configs(path: str | Path = DEFAULT_MODEL_CONFIG_PATH) -> dict[str, GenerationModelConfig]:
    config_path = Path(path)
    raw = json.loads(config_path.read_text(encoding="utf-8"))
    models = raw.get("models")
    if not isinstance(models, dict):
        raise ValueError(f"Model config must contain a 'models' object: {config_path}")
    parsed: dict[str, GenerationModelConfig] = {}
    for model_id, model_config in models.items():
        if not isinstance(model_id, str) or not model_id.strip():
            raise ValueError("Model ids must be non-empty strings")
        if not isinstance(model_config, dict):
            raise ValueError(f"Model config for {model_id} must be an object")
        parsed[model_id] = _parse_generation_model_config(model_id, model_config)
    return parsed


def select_generation_model_configs(
    configs: dict[str, GenerationModelConfig],
    requested_ids: list[str],
) -> tuple[list[GenerationModelConfig], list[dict[str, str]]]:
    selected: list[GenerationModelConfig] = []
    skipped: list[dict[str, str]] = []
    seen: set[str] = set()
    for model_id in requested_ids:
        if model_id in seen:
            continue
        seen.add(model_id)
        config = configs.get(model_id)
        if config is None:
            skipped.append({"model_id": model_id, "reason": "model-config-not-found"})
            continue
        if not config.enabled:
            skipped.append({"model_id": model_id, "reason": "model-disabled"})
            continue
        selected.append(config)
    return selected, skipped


def _parse_generation_model_config(model_id: str, raw: dict[str, Any]) -> GenerationModelConfig:
    provider = _required_string(raw, "provider", model_id).lower()
    if provider not in {"groq", "mimo"}:
        raise ValueError(f"Model {model_id} has unsupported provider '{provider}'")
    return GenerationModelConfig(
        model_id=model_id,
        provider=provider,
        model=_required_string(raw, "model", model_id),
        role=_required_string(raw, "role", model_id),
        enabled=bool(raw.get("enabled", True)),
        base_url=_optional_string(raw, "base_url"),
        base_url_env=_optional_string(raw, "base_url_env"),
        api_key_env=_optional_string(raw, "api_key_env"),
    )


def _required_string(raw: dict[str, Any], key: str, model_id: str) -> str:
    value = raw.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"Model {model_id} must define non-empty '{key}'")
    return value.strip()


def _optional_string(raw: dict[str, Any], key: str) -> str | None:
    value = raw.get(key)
    if value is None:
        return None
    if not isinstance(value, str):
        raise ValueError(f"Optional field '{key}' must be a string")
    value = value.strip()
    return value or None
