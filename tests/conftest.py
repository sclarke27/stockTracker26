"""Shared pytest fixtures for Stock Radar tests."""

from __future__ import annotations

import pytest


@pytest.fixture()
def sample_config() -> dict:
    """Provide a minimal config dict for testing without env vars."""
    return {
        "api_keys": {
            "alpha_vantage": "test-av-key",
            "finnhub": "test-fh-key",
            "anthropic": "test-anthropic-key",
        },
        "ollama": {
            "host": "http://localhost:11434",
            "default_model": "llama3.1:8b",
            "large_model": "llama3.1:70b",
            "timeout_seconds": 30,
        },
        "cache": {
            "db_path": "data/test.db",
            "chroma_path": "data/test_chroma",
        },
        "sec_edgar": {
            "user_agent_email": "test@example.com",
        },
        "predictions": {
            "db_path": "data/test_predictions.db",
        },
        "agents": {
            "earnings_linguist": {
                "enabled": True,
                "default_horizon_days": 5,
                "escalation_confidence_threshold": 0.3,
                "escalation_transcript_length": 24000,
                "anthropic_model": "claude-sonnet-4-20250514",
                "temperature": 0.1,
                "max_tokens": 4096,
            },
        },
    }
