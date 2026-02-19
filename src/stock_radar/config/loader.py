"""YAML configuration loader with environment variable interpolation."""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import cast

import yaml

from stock_radar.config.settings import AppSettings

ENV_VAR_PATTERN = re.compile(r"\$\{(\w+)\}")

_DEFAULT_CONFIG_PATH = Path(__file__).resolve().parents[3] / "config" / "default.yaml"


def _interpolate_env_vars(value: str) -> str:
    """Replace ${ENV_VAR} placeholders with environment variable values.

    Args:
        value: String potentially containing ${ENV_VAR} patterns.

    Returns:
        String with environment variables substituted.

    Raises:
        ValueError: If a referenced environment variable is not set.
    """

    def _replace(match: re.Match) -> str:
        var_name = match.group(1)
        env_value = os.environ.get(var_name)
        if env_value is None:
            raise ValueError(
                f"Environment variable '{var_name}' is not set. "
                f"Check your .env file or environment."
            )
        return env_value

    return ENV_VAR_PATTERN.sub(_replace, value)


def _walk_and_interpolate(obj: object) -> object:
    """Recursively walk a parsed YAML structure and interpolate env vars in strings.

    Args:
        obj: Any YAML-parsed object (dict, list, str, int, etc.).

    Returns:
        The same structure with all string values interpolated.
    """
    if isinstance(obj, dict):
        return {key: _walk_and_interpolate(val) for key, val in obj.items()}
    if isinstance(obj, list):
        return [_walk_and_interpolate(item) for item in obj]
    if isinstance(obj, str):
        return _interpolate_env_vars(obj)
    return obj


def load_config(path: Path | None = None) -> dict:
    """Load and return configuration from a YAML file.

    Environment variable placeholders (``${VAR_NAME}``) in string values are
    replaced with actual environment variable values.

    Args:
        path: Path to the YAML config file. Defaults to ``config/default.yaml``
            relative to the project root.

    Returns:
        Parsed configuration dictionary with env vars interpolated.

    Raises:
        FileNotFoundError: If the config file does not exist.
        ValueError: If a referenced environment variable is not set.
    """
    config_path = path or _DEFAULT_CONFIG_PATH

    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(config_path) as f:
        raw = yaml.safe_load(f) or {}

    return cast(dict, _walk_and_interpolate(raw))


def load_settings(path: Path | None = None) -> AppSettings:
    """Load application settings from a YAML config file.

    Convenience wrapper around :func:`load_config` that returns a
    validated :class:`AppSettings` instance.

    Args:
        path: Path to the YAML config file.  Defaults to
            ``config/default.yaml`` relative to the project root.

    Returns:
        Populated AppSettings instance.
    """
    config = load_config(path)
    return AppSettings(**config)
