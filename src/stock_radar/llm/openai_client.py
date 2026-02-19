"""OpenAI API client."""

from __future__ import annotations

import time

import openai
from loguru import logger

from stock_radar.llm.base import LlmClient
from stock_radar.llm.exceptions import LlmConnectionError, LlmError, LlmTimeoutError
from stock_radar.llm.models import LlmRequest, LlmResponse, LlmUsage

DEFAULT_MODEL = "gpt-4o"


class OpenAiClient(LlmClient):
    """Async client for the OpenAI Chat Completions API.

    Wraps the official ``openai`` SDK.  Unlike the Anthropic API, OpenAI
    accepts system messages inline in the ``messages`` list, so no
    message splitting is required.

    Args:
        api_key: OpenAI API key.
        model: Model identifier (default: gpt-4o).
    """

    def __init__(self, api_key: str, model: str = DEFAULT_MODEL) -> None:
        self._client = openai.AsyncOpenAI(api_key=api_key)
        self._model = model

    async def generate(self, request: LlmRequest) -> LlmResponse:
        """Send a request to the OpenAI API and return the response.

        Maps ``LlmRequest`` to the OpenAI Chat Completions format.
        When ``response_format`` is ``"json"``, sets
        ``response_format={"type": "json_object"}`` to enable JSON mode.

        Args:
            request: The LLM request payload containing messages and parameters.

        Returns:
            An ``LlmResponse`` with the generated content, model info,
            token usage, and wall-clock duration.

        Raises:
            LlmTimeoutError: If the API call times out.
            LlmConnectionError: If the API is unreachable.
            LlmError: For all other API-level errors.
        """
        messages = [{"role": m.role, "content": m.content} for m in request.messages]

        logger.debug(
            "OpenAI request | model={} messages={}",
            self._model,
            len(messages),
        )

        start_time = time.monotonic()

        try:
            kwargs: dict = {
                "model": self._model,
                "max_tokens": request.max_tokens,
                "messages": messages,
                "temperature": request.temperature,
            }
            if request.response_format == "json":
                kwargs["response_format"] = {"type": "json_object"}

            response = await self._client.chat.completions.create(**kwargs)
        except openai.APITimeoutError as exc:
            raise LlmTimeoutError(f"OpenAI API timed out: {exc}") from exc
        except openai.APIConnectionError as exc:
            raise LlmConnectionError(f"Cannot connect to OpenAI API: {exc}") from exc
        except openai.APIError as exc:
            raise LlmError(f"OpenAI API error: {exc}") from exc

        elapsed_ms = (time.monotonic() - start_time) * 1000

        content = ""
        if response.choices:
            content = response.choices[0].message.content or ""

        return LlmResponse(
            content=content,
            model=response.model,
            usage=LlmUsage(
                prompt_tokens=response.usage.prompt_tokens,
                completion_tokens=response.usage.completion_tokens,
                total_tokens=response.usage.total_tokens,
            ),
            duration_ms=elapsed_ms,
        )
