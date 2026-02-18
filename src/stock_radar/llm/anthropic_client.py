"""Anthropic Claude API client."""

from __future__ import annotations

import time

import anthropic
from loguru import logger

from stock_radar.llm.base import LlmClient
from stock_radar.llm.exceptions import LlmConnectionError, LlmError, LlmTimeoutError
from stock_radar.llm.models import LlmRequest, LlmResponse, LlmUsage

DEFAULT_MODEL = "claude-sonnet-4-20250514"


class AnthropicClient(LlmClient):
    """Async client for the Anthropic Claude API.

    Wraps the official ``anthropic`` SDK.  Extracts system messages from
    the messages list since Anthropic requires them as a separate
    ``system`` parameter rather than inline in the conversation.

    Args:
        api_key: Anthropic API key.
        model: Model identifier (default: claude-sonnet-4-20250514).
    """

    def __init__(self, api_key: str, model: str = DEFAULT_MODEL) -> None:
        self._client = anthropic.AsyncAnthropic(api_key=api_key)
        self._model = model

    async def generate(self, request: LlmRequest) -> LlmResponse:
        """Send a request to the Claude API and return the response.

        Extracts any system-role message from the messages list and
        passes it via the dedicated ``system`` parameter.  All other
        messages are forwarded as the ``messages`` list.

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
        system_content, conversation_messages = self._split_messages(request)

        logger.debug(
            "Anthropic request | model={} messages={} has_system={}",
            self._model,
            len(conversation_messages),
            bool(system_content),
        )

        start_time = time.monotonic()

        try:
            kwargs: dict = {
                "model": self._model,
                "max_tokens": request.max_tokens,
                "messages": conversation_messages,
                "temperature": request.temperature,
            }
            if system_content:
                kwargs["system"] = system_content

            response = await self._client.messages.create(**kwargs)
        except anthropic.APITimeoutError as exc:
            raise LlmTimeoutError(f"Anthropic API timed out: {exc}") from exc
        except anthropic.APIConnectionError as exc:
            raise LlmConnectionError(f"Cannot connect to Anthropic API: {exc}") from exc
        except anthropic.APIError as exc:
            raise LlmError(f"Anthropic API error: {exc}") from exc

        elapsed_ms = (time.monotonic() - start_time) * 1000
        content = response.content[0].text if response.content else ""

        return LlmResponse(
            content=content,
            model=response.model,
            usage=LlmUsage(
                prompt_tokens=response.usage.input_tokens,
                completion_tokens=response.usage.output_tokens,
                total_tokens=response.usage.input_tokens + response.usage.output_tokens,
            ),
            duration_ms=elapsed_ms,
        )

    @staticmethod
    def _split_messages(request: LlmRequest) -> tuple[str, list[dict[str, str]]]:
        """Separate system content from conversation messages.

        The Anthropic API expects system instructions as a top-level
        parameter, not as a message in the conversation list.

        Args:
            request: The LLM request to process.

        Returns:
            A tuple of ``(system_content, conversation_messages)`` where
            ``system_content`` is the extracted system prompt (empty string
            if none) and ``conversation_messages`` is the remaining list
            formatted for the SDK.
        """
        system_content = ""
        conversation_messages: list[dict[str, str]] = []

        for msg in request.messages:
            if msg.role == "system":
                system_content = msg.content
            else:
                conversation_messages.append({"role": msg.role, "content": msg.content})

        return system_content, conversation_messages
