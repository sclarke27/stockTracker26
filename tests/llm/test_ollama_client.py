"""Tests for the Ollama LLM client."""

from __future__ import annotations

import json

import httpx
import pytest
import respx

from stock_radar.llm.exceptions import LlmConnectionError, LlmError, LlmTimeoutError
from stock_radar.llm.models import LlmMessage, LlmRequest, LlmResponse
from stock_radar.llm.ollama_client import OllamaClient

OLLAMA_HOST = "http://localhost:11434"
OLLAMA_CHAT_URL = f"{OLLAMA_HOST}/api/chat"
DEFAULT_MODEL = "qwen3:32b"


@pytest.fixture()
def client() -> OllamaClient:
    """Create an OllamaClient pointed at the default test host."""
    return OllamaClient(host=OLLAMA_HOST, model=DEFAULT_MODEL, timeout_seconds=30)


SAMPLE_OLLAMA_RESPONSE = {
    "message": {"role": "assistant", "content": "hello"},
    "model": "qwen3:32b",
    "eval_count": 50,
    "prompt_eval_count": 100,
    "total_duration": 1_500_000_000,
}


def _make_request(
    messages: list[LlmMessage] | None = None,
    temperature: float = 0.1,
    max_tokens: int = 4096,
    response_format: str = "json",
) -> LlmRequest:
    """Build an LlmRequest with sensible defaults for testing."""
    if messages is None:
        messages = [LlmMessage(role="user", content="Say hello")]
    return LlmRequest(
        messages=messages,
        temperature=temperature,
        max_tokens=max_tokens,
        response_format=response_format,
    )


class TestOllamaClientGenerate:
    """Tests for OllamaClient.generate()."""

    @respx.mock
    async def test_generate_success(self, client: OllamaClient) -> None:
        """Successful Ollama response is mapped to a correct LlmResponse."""
        respx.post(OLLAMA_CHAT_URL).mock(
            return_value=httpx.Response(200, json=SAMPLE_OLLAMA_RESPONSE)
        )

        result = await client.generate(_make_request())

        assert isinstance(result, LlmResponse)
        assert result.content == "hello"
        assert result.model == "qwen3:32b"
        assert result.usage.prompt_tokens == 100
        assert result.usage.completion_tokens == 50
        assert result.usage.total_tokens == 150

    @respx.mock
    async def test_generate_json_format(self, client: OllamaClient) -> None:
        """When response_format is 'json', the POST body includes format='json'."""
        route = respx.post(OLLAMA_CHAT_URL).mock(
            return_value=httpx.Response(200, json=SAMPLE_OLLAMA_RESPONSE)
        )

        await client.generate(_make_request(response_format="json"))

        body = json.loads(route.calls[0].request.content)
        assert body["format"] == "json"

    @respx.mock
    async def test_generate_text_format(self, client: OllamaClient) -> None:
        """When response_format is 'text', no 'format' key appears in the body."""
        route = respx.post(OLLAMA_CHAT_URL).mock(
            return_value=httpx.Response(200, json=SAMPLE_OLLAMA_RESPONSE)
        )

        await client.generate(_make_request(response_format="text"))

        body = json.loads(route.calls[0].request.content)
        assert "format" not in body

    @respx.mock
    async def test_generate_maps_messages(self, client: OllamaClient) -> None:
        """Messages are correctly mapped from LlmMessage to Ollama format."""
        route = respx.post(OLLAMA_CHAT_URL).mock(
            return_value=httpx.Response(200, json=SAMPLE_OLLAMA_RESPONSE)
        )

        messages = [
            LlmMessage(role="system", content="You are a helpful assistant."),
            LlmMessage(role="user", content="What is 2+2?"),
        ]
        await client.generate(_make_request(messages=messages))

        body = json.loads(route.calls[0].request.content)
        assert body["messages"] == [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "What is 2+2?"},
        ]

    @respx.mock
    async def test_generate_passes_options(self, client: OllamaClient) -> None:
        """Temperature and max_tokens are passed in the options dict."""
        route = respx.post(OLLAMA_CHAT_URL).mock(
            return_value=httpx.Response(200, json=SAMPLE_OLLAMA_RESPONSE)
        )

        await client.generate(_make_request(temperature=0.7, max_tokens=2048))

        body = json.loads(route.calls[0].request.content)
        assert body["options"]["temperature"] == pytest.approx(0.7)
        assert body["options"]["num_predict"] == 2048

    @respx.mock
    async def test_generate_timeout(self, client: OllamaClient) -> None:
        """A timeout from httpx is raised as LlmTimeoutError."""
        respx.post(OLLAMA_CHAT_URL).mock(side_effect=httpx.ReadTimeout("timeout"))

        with pytest.raises(LlmTimeoutError, match="timed out"):
            await client.generate(_make_request())

    @respx.mock
    async def test_generate_connection_error(self, client: OllamaClient) -> None:
        """A connection error from httpx is raised as LlmConnectionError."""
        respx.post(OLLAMA_CHAT_URL).mock(side_effect=httpx.ConnectError("refused"))

        with pytest.raises(LlmConnectionError, match="Cannot connect"):
            await client.generate(_make_request())

    @respx.mock
    async def test_generate_http_error(self, client: OllamaClient) -> None:
        """A 500 HTTP response is raised as LlmError."""
        respx.post(OLLAMA_CHAT_URL).mock(
            return_value=httpx.Response(500, text="Internal Server Error")
        )

        with pytest.raises(LlmError, match="500"):
            await client.generate(_make_request())

    @respx.mock
    async def test_generate_uses_correct_model(self, client: OllamaClient) -> None:
        """The model name from __init__ is sent in the request body."""
        route = respx.post(OLLAMA_CHAT_URL).mock(
            return_value=httpx.Response(200, json=SAMPLE_OLLAMA_RESPONSE)
        )

        await client.generate(_make_request())

        body = json.loads(route.calls[0].request.content)
        assert body["model"] == "qwen3:32b"

    @respx.mock
    async def test_generate_calculates_duration_ms(self, client: OllamaClient) -> None:
        """total_duration in nanoseconds is correctly converted to milliseconds."""
        respx.post(OLLAMA_CHAT_URL).mock(
            return_value=httpx.Response(200, json=SAMPLE_OLLAMA_RESPONSE)
        )

        result = await client.generate(_make_request())

        # 1_500_000_000 ns == 1500.0 ms
        assert result.duration_ms == pytest.approx(1500.0)
