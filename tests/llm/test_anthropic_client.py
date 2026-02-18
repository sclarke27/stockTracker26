"""Tests for the Anthropic Claude API client."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import anthropic
import httpx
import pytest

from stock_radar.llm.exceptions import LlmConnectionError, LlmError, LlmTimeoutError
from stock_radar.llm.models import LlmMessage, LlmRequest, LlmResponse, LlmUsage

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_MODULE_PATH = "stock_radar.llm.anthropic_client"


def _make_mock_response(
    text: str = "response text",
    model: str = "claude-sonnet-4-20250514",
    input_tokens: int = 100,
    output_tokens: int = 50,
) -> MagicMock:
    """Build a mock Anthropic SDK response object."""
    mock_response = MagicMock()
    mock_response.content = [MagicMock(text=text)]
    mock_response.model = model
    mock_response.usage = MagicMock(input_tokens=input_tokens, output_tokens=output_tokens)
    return mock_response


def _make_mock_sdk_client(response: MagicMock | None = None) -> AsyncMock:
    """Build a mock ``anthropic.AsyncAnthropic`` instance."""
    mock_sdk_client = AsyncMock()
    if response is not None:
        mock_sdk_client.messages.create = AsyncMock(return_value=response)
    return mock_sdk_client


def _make_request(
    messages: list[LlmMessage] | None = None,
    temperature: float = 0.1,
    max_tokens: int = 4096,
) -> LlmRequest:
    """Build an ``LlmRequest`` with sensible defaults."""
    if messages is None:
        messages = [LlmMessage(role="user", content="Hello")]
    return LlmRequest(messages=messages, temperature=temperature, max_tokens=max_tokens)


def _dummy_httpx_request() -> httpx.Request:
    """Build a minimal httpx.Request for constructing SDK exceptions."""
    return httpx.Request("POST", "https://api.anthropic.com/v1/messages")


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestAnthropicClientGenerate:
    """Tests for AnthropicClient.generate."""

    async def test_generate_success(self) -> None:
        """Successful generation returns an LlmResponse with correct fields."""
        mock_response = _make_mock_response(
            text="Hello from Claude",
            model="claude-sonnet-4-20250514",
            input_tokens=42,
            output_tokens=17,
        )
        mock_sdk = _make_mock_sdk_client(mock_response)

        with patch(f"{_MODULE_PATH}.anthropic.AsyncAnthropic", return_value=mock_sdk):
            from stock_radar.llm.anthropic_client import AnthropicClient

            client = AnthropicClient(api_key="test-key")
            result = await client.generate(_make_request())

        assert isinstance(result, LlmResponse)
        assert result.content == "Hello from Claude"
        assert result.model == "claude-sonnet-4-20250514"
        assert result.usage == LlmUsage(prompt_tokens=42, completion_tokens=17, total_tokens=59)
        assert result.duration_ms >= 0

    async def test_generate_extracts_system_message(self) -> None:
        """System-role message is extracted into the ``system`` kwarg."""
        mock_response = _make_mock_response()
        mock_sdk = _make_mock_sdk_client(mock_response)

        messages = [
            LlmMessage(role="system", content="You are a stock analyst."),
            LlmMessage(role="user", content="Analyze AAPL."),
        ]

        with patch(f"{_MODULE_PATH}.anthropic.AsyncAnthropic", return_value=mock_sdk):
            from stock_radar.llm.anthropic_client import AnthropicClient

            client = AnthropicClient(api_key="test-key")
            await client.generate(_make_request(messages=messages))

        call_kwargs = mock_sdk.messages.create.call_args.kwargs
        assert call_kwargs["system"] == "You are a stock analyst."
        assert call_kwargs["messages"] == [{"role": "user", "content": "Analyze AAPL."}]

    async def test_generate_no_system_message(self) -> None:
        """When no system message is present, ``system`` kwarg is omitted."""
        mock_response = _make_mock_response()
        mock_sdk = _make_mock_sdk_client(mock_response)

        messages = [LlmMessage(role="user", content="Hello")]

        with patch(f"{_MODULE_PATH}.anthropic.AsyncAnthropic", return_value=mock_sdk):
            from stock_radar.llm.anthropic_client import AnthropicClient

            client = AnthropicClient(api_key="test-key")
            await client.generate(_make_request(messages=messages))

        call_kwargs = mock_sdk.messages.create.call_args.kwargs
        assert "system" not in call_kwargs

    async def test_generate_passes_temperature_and_max_tokens(self) -> None:
        """Temperature and max_tokens are forwarded to the SDK."""
        mock_response = _make_mock_response()
        mock_sdk = _make_mock_sdk_client(mock_response)

        with patch(f"{_MODULE_PATH}.anthropic.AsyncAnthropic", return_value=mock_sdk):
            from stock_radar.llm.anthropic_client import AnthropicClient

            client = AnthropicClient(api_key="test-key")
            await client.generate(_make_request(temperature=0.7, max_tokens=2048))

        call_kwargs = mock_sdk.messages.create.call_args.kwargs
        assert call_kwargs["temperature"] == 0.7
        assert call_kwargs["max_tokens"] == 2048

    async def test_generate_uses_correct_model(self) -> None:
        """Model name from the constructor is forwarded to the SDK."""
        mock_response = _make_mock_response(model="claude-opus-4-20250514")
        mock_sdk = _make_mock_sdk_client(mock_response)

        with patch(f"{_MODULE_PATH}.anthropic.AsyncAnthropic", return_value=mock_sdk):
            from stock_radar.llm.anthropic_client import AnthropicClient

            client = AnthropicClient(api_key="test-key", model="claude-opus-4-20250514")
            result = await client.generate(_make_request())

        call_kwargs = mock_sdk.messages.create.call_args.kwargs
        assert call_kwargs["model"] == "claude-opus-4-20250514"
        assert result.model == "claude-opus-4-20250514"

    async def test_generate_api_error(self) -> None:
        """``anthropic.APIError`` is translated to ``LlmError``."""
        mock_sdk = _make_mock_sdk_client()
        mock_sdk.messages.create = AsyncMock(
            side_effect=anthropic.APIError(
                message="Internal server error",
                request=_dummy_httpx_request(),
                body=None,
            )
        )

        with patch(f"{_MODULE_PATH}.anthropic.AsyncAnthropic", return_value=mock_sdk):
            from stock_radar.llm.anthropic_client import AnthropicClient

            client = AnthropicClient(api_key="test-key")
            with pytest.raises(LlmError, match="Anthropic API error"):
                await client.generate(_make_request())

    async def test_generate_timeout(self) -> None:
        """``anthropic.APITimeoutError`` is translated to ``LlmTimeoutError``."""
        mock_sdk = _make_mock_sdk_client()
        mock_sdk.messages.create = AsyncMock(
            side_effect=anthropic.APITimeoutError(request=_dummy_httpx_request())
        )

        with patch(f"{_MODULE_PATH}.anthropic.AsyncAnthropic", return_value=mock_sdk):
            from stock_radar.llm.anthropic_client import AnthropicClient

            client = AnthropicClient(api_key="test-key")
            with pytest.raises(LlmTimeoutError, match="timed out"):
                await client.generate(_make_request())

    async def test_generate_connection_error(self) -> None:
        """``anthropic.APIConnectionError`` is translated to ``LlmConnectionError``."""
        mock_sdk = _make_mock_sdk_client()
        mock_sdk.messages.create = AsyncMock(
            side_effect=anthropic.APIConnectionError(request=_dummy_httpx_request())
        )

        with patch(f"{_MODULE_PATH}.anthropic.AsyncAnthropic", return_value=mock_sdk):
            from stock_radar.llm.anthropic_client import AnthropicClient

            client = AnthropicClient(api_key="test-key")
            with pytest.raises(LlmConnectionError, match="Cannot connect"):
                await client.generate(_make_request())
