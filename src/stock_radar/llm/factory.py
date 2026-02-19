"""Factory functions for creating LLM clients from configuration."""

from __future__ import annotations

from stock_radar.config.settings import OllamaSettings
from stock_radar.llm.anthropic_client import AnthropicClient
from stock_radar.llm.ollama_client import OllamaClient
from stock_radar.llm.openai_client import OpenAiClient


def create_ollama_client(
    settings: OllamaSettings,
    model: str | None = None,
) -> OllamaClient:
    """Create an Ollama client from application settings.

    Args:
        settings: Ollama configuration settings.
        model: Override model name. Uses settings.default_model if None.

    Returns:
        Configured OllamaClient instance.
    """
    return OllamaClient(
        host=settings.host,
        model=model or settings.default_model,
        timeout_seconds=settings.timeout_seconds,
    )


def create_anthropic_client(
    api_key: str,
    model: str | None = None,
) -> AnthropicClient:
    """Create an Anthropic client.

    Args:
        api_key: Anthropic API key.
        model: Override model name. Uses default if None.

    Returns:
        Configured AnthropicClient instance.
    """
    if model:
        return AnthropicClient(api_key=api_key, model=model)
    return AnthropicClient(api_key=api_key)


def create_openai_client(
    api_key: str,
    model: str | None = None,
) -> OpenAiClient:
    """Create an OpenAI client.

    Args:
        api_key: OpenAI API key.
        model: Override model name. Uses default (gpt-4o) if None.

    Returns:
        Configured OpenAiClient instance.
    """
    if model:
        return OpenAiClient(api_key=api_key, model=model)
    return OpenAiClient(api_key=api_key)
