"""Tests for the YAML configuration loader."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pytest
import yaml

from stock_radar.config.loader import load_config, load_settings


@pytest.fixture()
def config_file(tmp_path: Path) -> Path:
    """Write a minimal YAML config file with env var placeholders."""
    content = {
        "api_keys": {
            "alpha_vantage": "${ALPHA_VANTAGE_API_KEY}",
            "finnhub": "${FINNHUB_API_KEY}",
            "anthropic": "${ANTHROPIC_API_KEY}",
        },
        "sec_edgar": {"user_agent_email": "${SEC_EDGAR_EMAIL}"},
        "cache": {"db_path": "data/test.db"},
    }
    path = tmp_path / "test_config.yaml"
    path.write_text(yaml.dump(content))
    return path


MOCK_ENV = {
    "ALPHA_VANTAGE_API_KEY": "av-key-123",
    "FINNHUB_API_KEY": "fh-key-456",
    "ANTHROPIC_API_KEY": "ant-key-789",
    "OPENAI_API_KEY": "oai-key-012",
    "SEC_EDGAR_EMAIL": "test@example.com",
}


class TestLoadConfig:
    """Tests for load_config()."""

    def test_loads_yaml_and_interpolates_env_vars(self, config_file: Path) -> None:
        with patch.dict("os.environ", MOCK_ENV):
            config = load_config(path=config_file)
        assert config["api_keys"]["alpha_vantage"] == "av-key-123"
        assert config["api_keys"]["finnhub"] == "fh-key-456"
        assert config["sec_edgar"]["user_agent_email"] == "test@example.com"

    def test_non_env_var_values_unchanged(self, config_file: Path) -> None:
        with patch.dict("os.environ", MOCK_ENV):
            config = load_config(path=config_file)
        assert config["cache"]["db_path"] == "data/test.db"

    def test_missing_env_var_raises_value_error(self, config_file: Path) -> None:
        # Omit FINNHUB_API_KEY
        env = {k: v for k, v in MOCK_ENV.items() if k != "FINNHUB_API_KEY"}
        with (
            patch.dict("os.environ", env, clear=True),
            pytest.raises(ValueError, match="FINNHUB_API_KEY"),
        ):
            load_config(path=config_file)

    def test_missing_file_raises_file_not_found(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError):
            load_config(path=tmp_path / "nonexistent.yaml")

    def test_empty_yaml_returns_empty_dict(self, tmp_path: Path) -> None:
        empty_file = tmp_path / "empty.yaml"
        empty_file.write_text("")
        result = load_config(path=empty_file)
        assert result == {}

    def test_nested_list_values_interpolated(self, tmp_path: Path) -> None:
        content = {"items": ["${ALPHA_VANTAGE_API_KEY}", "static-value"]}
        path = tmp_path / "list_config.yaml"
        path.write_text(yaml.dump(content))
        with patch.dict("os.environ", MOCK_ENV):
            config = load_config(path=path)
        assert config["items"][0] == "av-key-123"
        assert config["items"][1] == "static-value"

    def test_numeric_values_passed_through(self, tmp_path: Path) -> None:
        content = {"timeout": 30, "rate_limit": 5}
        path = tmp_path / "numeric.yaml"
        path.write_text(yaml.dump(content))
        config = load_config(path=path)
        assert config["timeout"] == 30
        assert config["rate_limit"] == 5

    def test_uses_default_path_when_none_given(self) -> None:
        """load_config() with no args should load the real default.yaml."""
        with patch.dict("os.environ", MOCK_ENV):
            config = load_config()
        assert "api_keys" in config
        assert "cache" in config


class TestLoadSettings:
    """Tests for load_settings() dotenv integration."""

    def test_loads_env_from_dotenv_file(self, tmp_path: Path) -> None:
        """load_settings() should auto-load .env before interpolating."""
        env_file = tmp_path / ".env"
        env_file.write_text(
            "AV_KEY=from-dotenv\nFH_KEY=fh\nANT_KEY=ant\nEDGAR_EMAIL=e@e.com\n"
        )

        config_content = {
            "api_keys": {
                "alpha_vantage": "${AV_KEY}",
                "finnhub": "${FH_KEY}",
                "anthropic": "${ANT_KEY}",
            },
            "sec_edgar": {"user_agent_email": "${EDGAR_EMAIL}"},
        }
        config_file = tmp_path / "config.yaml"
        config_file.write_text(yaml.dump(config_content))

        with (
            patch("stock_radar.config.loader._DEFAULT_ENV_PATH", env_file),
            patch("stock_radar.config.loader._DEFAULT_CONFIG_PATH", config_file),
            patch.dict("os.environ", {}, clear=True),
        ):
            settings = load_settings()
        assert settings.api_keys.alpha_vantage == "from-dotenv"

    def test_env_vars_override_dotenv(self, tmp_path: Path) -> None:
        """Explicit env vars should take precedence over .env file values."""
        env_file = tmp_path / ".env"
        env_file.write_text(
            "AV_KEY=from-dotenv\nFH_KEY=fh\nANT_KEY=ant\nEDGAR_EMAIL=e@e.com\n"
        )

        config_content = {
            "api_keys": {
                "alpha_vantage": "${AV_KEY}",
                "finnhub": "${FH_KEY}",
                "anthropic": "${ANT_KEY}",
            },
            "sec_edgar": {"user_agent_email": "${EDGAR_EMAIL}"},
        }
        config_file = tmp_path / "config.yaml"
        config_file.write_text(yaml.dump(config_content))

        with (
            patch("stock_radar.config.loader._DEFAULT_ENV_PATH", env_file),
            patch("stock_radar.config.loader._DEFAULT_CONFIG_PATH", config_file),
            patch.dict("os.environ", {"AV_KEY": "from-env"}, clear=True),
        ):
            settings = load_settings()
        assert settings.api_keys.alpha_vantage == "from-env"
