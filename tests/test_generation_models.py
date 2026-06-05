from __future__ import annotations

import json
from pathlib import Path

import pytest

from rag_bench.generation_models import load_generation_model_configs, select_generation_model_configs


def test_load_generation_model_configs_parses_enabled_models(tmp_path: Path) -> None:
    config_path = tmp_path / "models.json"
    config_path.write_text(
        json.dumps(
            {
                "models": {
                    "groq_test": {
                        "provider": "groq",
                        "model": "llama-test",
                        "role": "fast-small-baseline",
                    },
                    "mimo_test": {
                        "provider": "mimo",
                        "model": "mimo-test",
                        "role": "long-context-upper-bound",
                        "enabled": False,
                        "api_key_env": "MIMO_API_KEY",
                    },
                }
            }
        ),
        encoding="utf-8",
    )

    configs = load_generation_model_configs(config_path)

    assert configs["groq_test"].provider == "groq"
    assert configs["groq_test"].enabled is True
    assert configs["mimo_test"].enabled is False
    assert configs["mimo_test"].api_key_env == "MIMO_API_KEY"


def test_select_generation_model_configs_skips_disabled_and_unknown(tmp_path: Path) -> None:
    config_path = tmp_path / "models.json"
    config_path.write_text(
        json.dumps(
            {
                "models": {
                    "enabled": {"provider": "groq", "model": "a", "role": "role-a"},
                    "disabled": {"provider": "groq", "model": "b", "role": "role-b", "enabled": False},
                }
            }
        ),
        encoding="utf-8",
    )
    configs = load_generation_model_configs(config_path)

    selected, skipped = select_generation_model_configs(configs, ["enabled", "disabled", "missing"])

    assert [model.model_id for model in selected] == ["enabled"]
    assert skipped == [
        {"model_id": "disabled", "reason": "model-disabled"},
        {"model_id": "missing", "reason": "model-config-not-found"},
    ]


def test_load_generation_model_configs_rejects_unknown_provider(tmp_path: Path) -> None:
    config_path = tmp_path / "models.json"
    config_path.write_text(
        json.dumps({"models": {"bad": {"provider": "other", "model": "x", "role": "role"}}}),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="unsupported provider"):
        load_generation_model_configs(config_path)
