"""Tests for LLM client factory functions."""

from __future__ import annotations

from stock_radar.config.settings import OllamaSettings
from stock_radar.llm.anthropic_client import AnthropicClient
from stock_radar.llm.factory import (
    create_anthropic_client,
    create_ollama_client,
    create_openai_client,
)
from stock_radar.llm.ollama_client import OllamaClient
from stock_radar.llm.openai_client import OpenAiClient


class TestCreateOllamaClient:
    """Tests for create_ollama_client factory."""

    def test_creates_with_default_model(self) -> None:
        """Uses settings.default_model when no model override."""
        settings = OllamaSettings()
        client = create_ollama_client(settings)
        assert isinstance(client, OllamaClient)
        assert client._model == "llama3.1:8b"
        assert client._host == "http://localhost:11434"

    def test_creates_with_model_override(self) -> None:
        """Uses the provided model instead of settings default."""
        settings = OllamaSettings()
        client = create_ollama_client(settings, model="llama3.1:70b")
        assert isinstance(client, OllamaClient)
        assert client._model == "llama3.1:70b"


class TestCreateAnthropicClient:
    """Tests for create_anthropic_client factory."""

    def test_creates_with_default_model(self) -> None:
        """Uses DEFAULT_MODEL when no model override."""
        client = create_anthropic_client(api_key="test-key")
        assert isinstance(client, AnthropicClient)
        assert client._model == "claude-sonnet-4-20250514"

    def test_creates_with_model_override(self) -> None:
        """Uses the provided model."""
        client = create_anthropic_client(api_key="test-key", model="claude-opus-4-20250514")
        assert isinstance(client, AnthropicClient)
        assert client._model == "claude-opus-4-20250514"


class TestCreateOpenAiClient:
    """Tests for create_openai_client factory."""

    def test_creates_with_default_model(self) -> None:
        """Uses DEFAULT_MODEL (gpt-4o) when no model override."""
        client = create_openai_client(api_key="test-key")
        assert isinstance(client, OpenAiClient)
        assert client._model == "gpt-4o"

    def test_creates_with_model_override(self) -> None:
        """Uses the provided model."""
        client = create_openai_client(api_key="test-key", model="o3")
        assert isinstance(client, OpenAiClient)
        assert client._model == "o3"
