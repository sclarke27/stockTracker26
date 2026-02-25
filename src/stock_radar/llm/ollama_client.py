"""Ollama LLM client using the /api/chat HTTP endpoint."""

from __future__ import annotations

import httpx
from loguru import logger

from stock_radar.llm.base import LlmClient
from stock_radar.llm.exceptions import LlmConnectionError, LlmError, LlmTimeoutError
from stock_radar.llm.models import LlmRequest, LlmResponse, LlmUsage


class OllamaClient(LlmClient):
    """Async client for Ollama's /api/chat endpoint.

    Uses httpx to POST to Ollama's HTTP API. Supports JSON-constrained
    output via the ``format`` parameter.

    Args:
        host: Ollama server URL (e.g. ``"http://localhost:11434"``).
        model: Model name (e.g. ``"qwen3:32b"``).
        timeout_seconds: Maximum time to wait for inference.
    """

    def __init__(
        self,
        host: str,
        model: str,
        timeout_seconds: int = 30,
    ) -> None:
        self._host = host.rstrip("/")
        self._model = model
        self._timeout = timeout_seconds

    async def generate(self, request: LlmRequest) -> LlmResponse:
        """Send a chat request to Ollama and return the response.

        Maps ``LlmRequest`` to the Ollama ``/api/chat`` format:

        - ``messages``: list of ``{"role": str, "content": str}``
        - ``model``: model name
        - ``stream``: ``false`` (we want the complete response)
        - ``format``: ``"json"`` when ``response_format`` is ``"json"``
        - ``options``: ``{"temperature": float, "num_predict": int}``

        Args:
            request: The LLM request payload containing messages and parameters.

        Returns:
            The raw LLM response including content and usage stats.

        Raises:
            LlmTimeoutError: If inference exceeds the configured timeout.
            LlmConnectionError: If the Ollama server is unreachable.
            LlmError: On any other HTTP error.
        """
        url = f"{self._host}/api/chat"

        body: dict = {
            "model": self._model,
            "messages": [{"role": m.role, "content": m.content} for m in request.messages],
            "stream": False,
            "options": {
                "temperature": request.temperature,
                "num_predict": request.max_tokens,
            },
        }

        if request.response_format == "json":
            body["format"] = "json"

        logger.debug(
            "Ollama request: model={model} messages={count} format={fmt}",
            model=self._model,
            count=len(request.messages),
            fmt=request.response_format,
        )

        try:
            async with httpx.AsyncClient() as client:
                response = await client.post(
                    url,
                    json=body,
                    timeout=self._timeout,
                )
                response.raise_for_status()
        except httpx.TimeoutException as exc:
            raise LlmTimeoutError(f"Ollama inference timed out after {self._timeout}s") from exc
        except httpx.ConnectError as exc:
            raise LlmConnectionError(f"Cannot connect to Ollama at {self._host}: {exc}") from exc
        except httpx.HTTPStatusError as exc:
            raise LlmError(
                f"Ollama HTTP {exc.response.status_code}: " f"{exc.response.text[:200]}"
            ) from exc

        data = response.json()
        message = data.get("message", {})

        # total_duration is in nanoseconds; convert to milliseconds
        total_duration_ns = data.get("total_duration", 0)
        duration_ms = total_duration_ns / 1_000_000

        return LlmResponse(
            content=message.get("content", ""),
            model=data.get("model", self._model),
            usage=LlmUsage(
                prompt_tokens=data.get("prompt_eval_count", 0),
                completion_tokens=data.get("eval_count", 0),
                total_tokens=(data.get("prompt_eval_count", 0) + data.get("eval_count", 0)),
            ),
            duration_ms=duration_ms,
        )
