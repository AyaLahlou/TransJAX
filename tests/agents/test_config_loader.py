"""
Tests for transjax.agents.utils.config_loader.

All tests must pass without a config.yaml present and without an API key.
"""

import tempfile
from pathlib import Path

import pytest

from transjax.agents.utils.config_loader import (
    load_config,
    get_llm_config,
    get_agent_config,
    _DEFAULT_LLM_CONFIG,
)


def test_load_config_returns_dict():
    """load_config() must return a dict (using the bundled default)."""
    config = load_config()
    assert isinstance(config, dict)


def test_load_config_missing_file_returns_defaults():
    """load_config() with a non-existent path must not raise; returns defaults."""
    config = load_config(Path("/non/existent/path/config.yaml"))
    assert isinstance(config, dict)
    assert "llm" in config


def test_get_llm_config_has_required_keys():
    """get_llm_config() must return a dict with the four required keys."""
    llm = get_llm_config()
    for key in ("model", "temperature", "max_tokens", "timeout"):
        assert key in llm, f"Missing key '{key}' in LLM config"


def test_get_llm_config_model_is_string():
    """The 'model' field must be a non-empty string."""
    llm = get_llm_config()
    assert isinstance(llm["model"], str)
    assert len(llm["model"]) > 0


def test_get_llm_config_temperature_in_range():
    """Temperature must be a float in [0.0, 2.0]."""
    llm = get_llm_config()
    assert isinstance(llm["temperature"], (int, float))
    assert 0.0 <= llm["temperature"] <= 2.0


def test_get_agent_config_missing_returns_empty():
    """get_agent_config for an unknown agent must return an empty dict."""
    cfg = get_agent_config("nonexistent_agent_xyz")
    assert isinstance(cfg, dict)
    assert cfg == {}


def test_load_config_from_custom_yaml():
    """load_config must read a caller-supplied YAML file."""
    yaml_content = "llm:\n  model: test-model\n  temperature: 0.5\n"
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".yaml", delete=False
    ) as f:
        f.write(yaml_content)
        tmp_path = Path(f.name)

    try:
        config = load_config(tmp_path)
        assert config["llm"]["model"] == "test-model"
        assert config["llm"]["temperature"] == 0.5
    finally:
        tmp_path.unlink(missing_ok=True)


def test_get_llm_config_with_custom_config():
    """get_llm_config accepts a pre-loaded config dict."""
    custom_config = {"llm": {"model": "my-model", "temperature": 1.0, "max_tokens": 1000, "timeout": 60}}
    llm = get_llm_config(custom_config)
    assert llm["model"] == "my-model"
