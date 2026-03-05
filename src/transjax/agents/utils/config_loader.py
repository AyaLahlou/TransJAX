"""Configuration loader for transjax agents."""

from pathlib import Path
from typing import Any, Dict, Optional

import yaml

try:
    from importlib.resources import files as _res_files
    def _default_config_path() -> Path:
        ref = _res_files("transjax.agents.utils").joinpath("default_config.yaml")
        return Path(str(ref))
except Exception:
    # Fallback for Python < 3.9 or packaging edge-cases
    def _default_config_path() -> Path:  # type: ignore[misc]
        return Path(__file__).parent / "default_config.yaml"

_DEFAULT_LLM_CONFIG: Dict[str, Any] = {
    "model": "claude-sonnet-4-5",
    "temperature": 0.0,
    "max_tokens": 48000,
    "timeout": 600,
}


def load_config(config_path: Optional[Path] = None) -> Dict[str, Any]:
    """
    Load configuration from a YAML file.

    Args:
        config_path: Path to config file.  If *None*, the package-bundled
                     ``default_config.yaml`` is used.

    Returns:
        Configuration dictionary.  Falls back to a minimal dict with LLM
        defaults if no file can be found, rather than raising.
    """
    if config_path is None:
        config_path = _default_config_path()

    if not Path(config_path).exists():
        return {"llm": _DEFAULT_LLM_CONFIG, "agents": {}}

    with open(config_path, "r") as f:
        config = yaml.safe_load(f) or {}

    return config


def get_llm_config(config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """
    Return LLM configuration dict.

    Args:
        config: Full config dict.  If *None*, loads from default path.

    Returns:
        LLM configuration dictionary with keys: model, temperature, max_tokens, timeout.
    """
    if config is None:
        config = load_config()

    return config.get("llm", _DEFAULT_LLM_CONFIG)


def get_agent_config(agent_name: str, config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """
    Return configuration for a specific agent.

    Args:
        agent_name: Name of the agent (e.g. ``"translator"``).
        config: Full config dict.  If *None*, loads from default path.

    Returns:
        Agent-specific configuration dictionary (may be empty).
    """
    if config is None:
        config = load_config()

    agents_config = config.get("agents", {})
    return agents_config.get(agent_name, {})
