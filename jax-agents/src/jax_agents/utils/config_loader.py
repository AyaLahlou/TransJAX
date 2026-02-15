"""Configuration loader for jax-agents."""

import yaml
from pathlib import Path
from typing import Dict, Any, Optional

# More robust root detection: find the directory containing 'jax-agents' or 'src'
def _find_default_config() -> Path:
    current = Path(__file__).resolve()
    # Iterate upwards to find the project root containing config.yaml
    for parent in current.parents:
        candidate = parent / "config.yaml"
        if candidate.exists():
            return candidate
    # Fallback to the original logic if not found during traversal
    return current.parent.parent.parent.parent / "config.yaml"

DEFAULT_CONFIG_PATH = _find_default_config()

def load_config(config_path: Optional[Path] = None) -> Dict[str, Any]:
    """
    Load configuration from YAML file.
    
    Args:
        config_path: Path to config file. If None, uses default config.yaml
        
    Returns:
        Configuration dictionary
    """
    path = Path(config_path) if config_path else DEFAULT_CONFIG_PATH
    
    if not path.exists():
        raise FileNotFoundError(f"Config file not found at: {path.absolute()}")
    
    # Use Path.read_text for cleaner I/O
    try:
        data = yaml.safe_load(path.read_text())
        return data if data is not None else {}
    except yaml.YAMLError as e:
        raise ValueError(f"Error parsing YAML config: {e}")

def get_llm_config(config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """
    Get LLM configuration with sensible defaults.
    """
    if config is None:
        try:
            config = load_config()
        except FileNotFoundError:
            config = {}
    
    return config.get("llm", {
        "model": "claude-sonnet-4-5",
        "temperature": 0.0,
        "max_tokens": 48000,
        "timeout": 600,
    })

def get_agent_config(agent_name: str, config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """
    Get configuration for a specific agent.
    """
    if config is None:
        try:
            config = load_config()
        except FileNotFoundError:
            config = {}
            
    agents_config = config.get("agents", {})
    return agents_config.get(agent_name, {})