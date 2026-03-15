"""Configuration management for the fighting detection system."""

import yaml
from pathlib import Path
from typing import Any


class Config:
    """Hierarchical configuration object that supports attribute access."""

    def __init__(self, config_dict: dict):
        for key, value in config_dict.items():
            if isinstance(value, dict):
                setattr(self, key, Config(value))
            elif isinstance(value, list):
                setattr(self, key, [
                    Config(item) if isinstance(item, dict) else item
                    for item in value
                ])
            else:
                setattr(self, key, value)

    def to_dict(self) -> dict:
        """Convert config back to a dictionary."""
        result = {}
        for key, value in self.__dict__.items():
            if isinstance(value, Config):
                result[key] = value.to_dict()
            elif isinstance(value, list):
                result[key] = [
                    item.to_dict() if isinstance(item, Config) else item
                    for item in value
                ]
            else:
                result[key] = value
        return result

    def get(self, key: str, default: Any = None) -> Any:
        """Get a config value with a default fallback."""
        return getattr(self, key, default)

    def __repr__(self) -> str:
        return f"Config({self.to_dict()})"


def load_config(config_path: str) -> Config:
    """Load configuration from a YAML file.

    Args:
        config_path: Path to the YAML configuration file.

    Returns:
        Config object with hierarchical attribute access.
    """
    config_path = Path(config_path)
    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")

    with open(config_path, "r") as f:
        config_dict = yaml.safe_load(f)

    return Config(config_dict)


def merge_configs(base: Config, override: dict) -> Config:
    """Merge an override dictionary into a base config.

    Args:
        base: Base configuration.
        override: Dictionary of values to override.

    Returns:
        New Config with merged values.
    """
    base_dict = base.to_dict()
    _deep_update(base_dict, override)
    return Config(base_dict)


def _deep_update(base: dict, update: dict) -> dict:
    """Recursively update a dictionary."""
    for key, value in update.items():
        if key in base and isinstance(base[key], dict) and isinstance(value, dict):
            _deep_update(base[key], value)
        else:
            base[key] = value
    return base
