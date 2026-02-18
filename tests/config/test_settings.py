"""Tests for Pydantic settings models."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from stock_radar.config.settings import (
    ApiKeys,
    AppSettings,
    CacheSettings,
    EarningsLinguistSettings,
    OllamaSettings,
    ScoringSettings,
    SecEdgarSettings,
)


class TestApiKeys:
    """Tests for ApiKeys model."""

    def test_valid_construction(self) -> None:
        keys = ApiKeys(alpha_vantage="av-key", finnhub="fh-key")
        assert keys.alpha_vantage == "av-key"
        assert keys.finnhub == "fh-key"
        assert keys.anthropic == ""  # default

    def test_missing_required_field_raises(self) -> None:
        with pytest.raises(ValidationError):
            ApiKeys(finnhub="fh-key")  # type: ignore[call-arg]

    def test_anthropic_default_is_empty_string(self) -> None:
        keys = ApiKeys(alpha_vantage="av", finnhub="fh")
        assert keys.anthropic == ""


class TestOllamaSettings:
    """Tests for OllamaSettings model."""

    def test_defaults(self) -> None:
        settings = OllamaSettings()
        assert settings.host == "http://localhost:11434"
        assert settings.default_model == "llama3.1:8b"
        assert settings.large_model == "llama3.1:70b"
        assert settings.timeout_seconds == 30

    def test_override_values(self) -> None:
        settings = OllamaSettings(host="http://192.168.1.100:11434", timeout_seconds=60)
        assert settings.host == "http://192.168.1.100:11434"
        assert settings.timeout_seconds == 60


class TestCacheSettings:
    """Tests for CacheSettings model."""

    def test_defaults(self) -> None:
        settings = CacheSettings()
        assert settings.db_path == "data/stock_radar.db"
        assert settings.chroma_path == "data/chroma_data"

    def test_override_path(self) -> None:
        settings = CacheSettings(db_path="/tmp/test.db")
        assert settings.db_path == "/tmp/test.db"


class TestSecEdgarSettings:
    """Tests for SecEdgarSettings model."""

    def test_valid_construction(self) -> None:
        settings = SecEdgarSettings(user_agent_email="test@example.com")
        assert settings.user_agent_email == "test@example.com"

    def test_missing_email_raises(self) -> None:
        with pytest.raises(ValidationError):
            SecEdgarSettings()  # type: ignore[call-arg]


class TestEarningsLinguistSettings:
    """Tests for EarningsLinguistSettings model."""

    def test_defaults(self) -> None:
        settings = EarningsLinguistSettings()
        assert settings.enabled is True
        assert settings.default_horizon_days == 5
        assert settings.escalation_confidence_threshold == 0.3
        assert settings.escalation_transcript_length == 24000
        assert settings.ollama_model is None
        assert settings.temperature == 0.1
        assert settings.max_tokens == 4096

    def test_horizon_days_must_be_positive(self) -> None:
        with pytest.raises(ValidationError):
            EarningsLinguistSettings(default_horizon_days=0)

    def test_confidence_threshold_bounds(self) -> None:
        with pytest.raises(ValidationError):
            EarningsLinguistSettings(escalation_confidence_threshold=1.5)
        with pytest.raises(ValidationError):
            EarningsLinguistSettings(escalation_confidence_threshold=-0.1)

    def test_temperature_bounds(self) -> None:
        with pytest.raises(ValidationError):
            EarningsLinguistSettings(temperature=2.5)

    def test_max_tokens_must_be_positive(self) -> None:
        with pytest.raises(ValidationError):
            EarningsLinguistSettings(max_tokens=0)


class TestScoringSettings:
    """Tests for ScoringSettings model."""

    def test_defaults(self) -> None:
        settings = ScoringSettings()
        assert settings.horizon_buffer_days == 1
        assert settings.lookback_days == 365

    def test_lookback_days_must_be_positive(self) -> None:
        with pytest.raises(ValidationError):
            ScoringSettings(lookback_days=0)

    def test_buffer_days_allows_zero(self) -> None:
        settings = ScoringSettings(horizon_buffer_days=0)
        assert settings.horizon_buffer_days == 0


class TestAppSettings:
    """Tests for AppSettings (top-level) model."""

    def _minimal_config(self) -> dict:
        return {
            "api_keys": {"alpha_vantage": "av", "finnhub": "fh"},
            "sec_edgar": {"user_agent_email": "test@example.com"},
        }

    def test_valid_construction_with_required_fields(self) -> None:
        settings = AppSettings(**self._minimal_config())
        assert settings.api_keys.alpha_vantage == "av"
        assert settings.sec_edgar.user_agent_email == "test@example.com"

    def test_optional_sections_use_defaults(self) -> None:
        settings = AppSettings(**self._minimal_config())
        assert settings.ollama.host == "http://localhost:11434"
        assert settings.cache.db_path == "data/stock_radar.db"
        assert settings.predictions.db_path == "data/predictions.db"
        assert settings.agents.earnings_linguist.enabled is True
        assert settings.scoring.lookback_days == 365

    def test_missing_api_keys_raises(self) -> None:
        with pytest.raises(ValidationError):
            AppSettings(sec_edgar={"user_agent_email": "test@example.com"})  # type: ignore[call-arg]

    def test_missing_sec_edgar_raises(self) -> None:
        with pytest.raises(ValidationError):
            AppSettings(api_keys={"alpha_vantage": "av", "finnhub": "fh"})  # type: ignore[call-arg]

    def test_nested_override(self) -> None:
        config = self._minimal_config()
        config["ollama"] = {"default_model": "llama3.2:3b", "timeout_seconds": 60}
        settings = AppSettings(**config)
        assert settings.ollama.default_model == "llama3.2:3b"
        assert settings.ollama.timeout_seconds == 60
