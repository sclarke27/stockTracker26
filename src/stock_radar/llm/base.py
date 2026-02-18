"""Abstract base class for LLM clients."""

from __future__ import annotations

import json
import re
from abc import ABC, abstractmethod
from typing import TypeVar

from pydantic import BaseModel, ValidationError

from stock_radar.llm.exceptions import LlmParseError
from stock_radar.llm.models import LlmRequest, LlmResponse

T = TypeVar("T", bound=BaseModel)


class LlmClient(ABC):
    """Abstract LLM client that all providers implement.

    Subclasses must implement ``generate()`` to handle the actual inference
    call. The ``generate_structured()`` method is a concrete convenience that
    calls ``generate()``, extracts JSON from the raw response, and validates
    it against a Pydantic model.
    """

    @abstractmethod
    async def generate(self, request: LlmRequest) -> LlmResponse:
        """Send messages to the LLM and return the raw response.

        Args:
            request: The LLM request payload containing messages and parameters.

        Returns:
            The raw LLM response including content and usage stats.
        """

    async def generate_structured(
        self,
        request: LlmRequest,
        response_model: type[T],
    ) -> T:
        """Generate a response and parse it into a Pydantic model.

        Calls ``generate()``, extracts JSON from the response content
        (handling markdown code blocks), and validates against
        ``response_model``.

        Args:
            request: The LLM request payload.
            response_model: Pydantic model class to validate against.

        Returns:
            Validated instance of response_model.

        Raises:
            LlmParseError: If the response cannot be parsed or validated.
        """
        response = await self.generate(request)
        return self._parse_response(response.content, response_model)

    @staticmethod
    def _extract_json(text: str) -> str:
        """Extract JSON from text, handling markdown code blocks.

        Tries in order:
            1. JSON within ``json ... `` or `` ... `` code blocks.
            2. First ``{ ... }`` block in the text.
            3. The raw text as-is (stripped).

        Args:
            text: Raw text potentially containing JSON.

        Returns:
            The extracted JSON string.
        """
        # Try markdown code block (with or without language tag)
        match = re.search(r"```(?:json)?\s*\n?(.*?)\n?```", text, re.DOTALL)
        if match:
            return match.group(1).strip()

        # Try to find a JSON object
        match = re.search(r"\{.*\}", text, re.DOTALL)
        if match:
            return match.group(0)

        return text.strip()

    @staticmethod
    def _parse_response(raw_content: str, response_model: type[T]) -> T:
        """Parse raw LLM text into a Pydantic model.

        Args:
            raw_content: Raw text from the LLM response.
            response_model: Pydantic model class to validate against.

        Returns:
            Validated model instance.

        Raises:
            LlmParseError: On JSON decode or Pydantic validation failure.
        """
        json_str = LlmClient._extract_json(raw_content)

        try:
            data = json.loads(json_str)
        except json.JSONDecodeError as exc:
            raise LlmParseError(
                f"Failed to parse JSON from LLM response: {exc}",
                raw_response=raw_content,
            ) from exc

        try:
            return response_model.model_validate(data)
        except ValidationError as exc:
            raise LlmParseError(
                f"LLM response failed validation: {exc}",
                raw_response=raw_content,
            ) from exc
