"""Tests for the OpenAI API client."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import openai
import pytest

from stock_radar.llm.exceptions import LlmConnectionError, LlmError, LlmTimeoutError
from stock_radar.llm.models import LlmMessage, LlmRequest, LlmResponse, LlmUsage

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_MODULE_PATH = "stock_radar.llm.openai_client"


def _make_mock_response(
    text: str = "response text",
    model: str = "gpt-4o",
    prompt_tokens: int = 100,
    completion_tokens: int = 50,
) -> MagicMock:
    """Build a mock OpenAI SDK chat completion response object."""
    mock_choice = MagicMock()
    mock_choice.message.content = text

    mock_response = MagicMock()
    mock_response.choices = [mock_choice]
    mock_response.model = model
    mock_response.usage = MagicMock(
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        total_tokens=prompt_tokens + completion_tokens,
    )
    return mock_response


def _make_mock_sdk_client(response: MagicMock | None = None) -> AsyncMock:
    """Build a mock ``openai.AsyncOpenAI`` instance."""
    mock_sdk_client = AsyncMock()
    if response is not None:
        mock_sdk_client.chat.completions.create = AsyncMock(return_value=response)
    return mock_sdk_client


def _make_request(
    messages: list[LlmMessage] | None = None,
    temperature: float = 0.1,
    max_tokens: int = 4096,
    response_format: str = "json",
) -> LlmRequest:
    """Build an ``LlmRequest`` with sensible defaults."""
    if messages is None:
        messages = [LlmMessage(role="user", content="Hello")]
    return LlmRequest(
        messages=messages,
        temperature=temperature,
        max_tokens=max_tokens,
        response_format=response_format,
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestOpenAiClientGenerate:
    """Tests for OpenAiClient.generate."""

    async def test_generate_success(self) -> None:
        """Successful generation returns an LlmResponse with correct fields."""
        mock_response = _make_mock_response(
            text="Hello from GPT",
            model="gpt-4o",
            prompt_tokens=42,
            completion_tokens=17,
        )
        mock_sdk = _make_mock_sdk_client(mock_response)

        with patch(f"{_MODULE_PATH}.openai.AsyncOpenAI", return_value=mock_sdk):
            from stock_radar.llm.openai_client import OpenAiClient

            client = OpenAiClient(api_key="test-key")
            result = await client.generate(_make_request())

        assert isinstance(result, LlmResponse)
        assert result.content == "Hello from GPT"
        assert result.model == "gpt-4o"
        assert result.usage == LlmUsage(prompt_tokens=42, completion_tokens=17, total_tokens=59)
        assert result.duration_ms >= 0

    async def test_generate_passes_messages(self) -> None:
        """All messages including system are forwarded in the messages list."""
        mock_response = _make_mock_response()
        mock_sdk = _make_mock_sdk_client(mock_response)

        messages = [
            LlmMessage(role="system", content="You are a stock analyst."),
            LlmMessage(role="user", content="Analyze AAPL."),
        ]

        with patch(f"{_MODULE_PATH}.openai.AsyncOpenAI", return_value=mock_sdk):
            from stock_radar.llm.openai_client import OpenAiClient

            client = OpenAiClient(api_key="test-key")
            await client.generate(_make_request(messages=messages))

        call_kwargs = mock_sdk.chat.completions.create.call_args.kwargs
        assert call_kwargs["messages"] == [
            {"role": "system", "content": "You are a stock analyst."},
            {"role": "user", "content": "Analyze AAPL."},
        ]

    async def test_generate_passes_temperature_and_max_tokens(self) -> None:
        """Temperature and max_tokens are forwarded to the SDK."""
        mock_response = _make_mock_response()
        mock_sdk = _make_mock_sdk_client(mock_response)

        with patch(f"{_MODULE_PATH}.openai.AsyncOpenAI", return_value=mock_sdk):
            from stock_radar.llm.openai_client import OpenAiClient

            client = OpenAiClient(api_key="test-key")
            await client.generate(_make_request(temperature=0.7, max_tokens=2048))

        call_kwargs = mock_sdk.chat.completions.create.call_args.kwargs
        assert call_kwargs["temperature"] == 0.7
        assert call_kwargs["max_tokens"] == 2048

    async def test_generate_uses_correct_model(self) -> None:
        """Model name from the constructor is forwarded to the SDK."""
        mock_response = _make_mock_response(model="o3")
        mock_sdk = _make_mock_sdk_client(mock_response)

        with patch(f"{_MODULE_PATH}.openai.AsyncOpenAI", return_value=mock_sdk):
            from stock_radar.llm.openai_client import OpenAiClient

            client = OpenAiClient(api_key="test-key", model="o3")
            result = await client.generate(_make_request())

        call_kwargs = mock_sdk.chat.completions.create.call_args.kwargs
        assert call_kwargs["model"] == "o3"
        assert result.model == "o3"

    async def test_generate_json_response_format(self) -> None:
        """JSON response format sets response_format type to json_object."""
        mock_response = _make_mock_response()
        mock_sdk = _make_mock_sdk_client(mock_response)

        with patch(f"{_MODULE_PATH}.openai.AsyncOpenAI", return_value=mock_sdk):
            from stock_radar.llm.openai_client import OpenAiClient

            client = OpenAiClient(api_key="test-key")
            await client.generate(_make_request(response_format="json"))

        call_kwargs = mock_sdk.chat.completions.create.call_args.kwargs
        assert call_kwargs["response_format"] == {"type": "json_object"}

    async def test_generate_text_response_format(self) -> None:
        """Text response format omits response_format parameter."""
        mock_response = _make_mock_response()
        mock_sdk = _make_mock_sdk_client(mock_response)

        with patch(f"{_MODULE_PATH}.openai.AsyncOpenAI", return_value=mock_sdk):
            from stock_radar.llm.openai_client import OpenAiClient

            client = OpenAiClient(api_key="test-key")
            await client.generate(_make_request(response_format="text"))

        call_kwargs = mock_sdk.chat.completions.create.call_args.kwargs
        assert "response_format" not in call_kwargs

    async def test_generate_api_error(self) -> None:
        """``openai.APIError`` is translated to ``LlmError``."""
        mock_sdk = _make_mock_sdk_client()
        mock_sdk.chat.completions.create = AsyncMock(
            side_effect=openai.APIError(
                message="Internal server error",
                request=MagicMock(),
                body=None,
            )
        )

        with patch(f"{_MODULE_PATH}.openai.AsyncOpenAI", return_value=mock_sdk):
            from stock_radar.llm.openai_client import OpenAiClient

            client = OpenAiClient(api_key="test-key")
            with pytest.raises(LlmError, match="OpenAI API error"):
                await client.generate(_make_request())

    async def test_generate_timeout(self) -> None:
        """``openai.APITimeoutError`` is translated to ``LlmTimeoutError``."""
        mock_sdk = _make_mock_sdk_client()
        mock_sdk.chat.completions.create = AsyncMock(
            side_effect=openai.APITimeoutError(request=MagicMock())
        )

        with patch(f"{_MODULE_PATH}.openai.AsyncOpenAI", return_value=mock_sdk):
            from stock_radar.llm.openai_client import OpenAiClient

            client = OpenAiClient(api_key="test-key")
            with pytest.raises(LlmTimeoutError, match="timed out"):
                await client.generate(_make_request())

    async def test_generate_connection_error(self) -> None:
        """``openai.APIConnectionError`` is translated to ``LlmConnectionError``."""
        mock_sdk = _make_mock_sdk_client()
        mock_sdk.chat.completions.create = AsyncMock(
            side_effect=openai.APIConnectionError(request=MagicMock())
        )

        with patch(f"{_MODULE_PATH}.openai.AsyncOpenAI", return_value=mock_sdk):
            from stock_radar.llm.openai_client import OpenAiClient

            client = OpenAiClient(api_key="test-key")
            with pytest.raises(LlmConnectionError, match="Cannot connect"):
                await client.generate(_make_request())

    async def test_generate_empty_choices(self) -> None:
        """Empty choices list returns empty content string."""
        mock_response = MagicMock()
        mock_response.choices = []
        mock_response.model = "gpt-4o"
        mock_response.usage = MagicMock(prompt_tokens=10, completion_tokens=0, total_tokens=10)
        mock_sdk = _make_mock_sdk_client(mock_response)

        with patch(f"{_MODULE_PATH}.openai.AsyncOpenAI", return_value=mock_sdk):
            from stock_radar.llm.openai_client import OpenAiClient

            client = OpenAiClient(api_key="test-key")
            result = await client.generate(_make_request())

        assert result.content == ""
