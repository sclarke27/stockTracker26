"""Exception hierarchy for LLM client operations."""

from __future__ import annotations


class LlmError(Exception):
    """Base exception for all LLM client errors."""


class LlmTimeoutError(LlmError):
    """Raised when an LLM inference call exceeds the timeout."""


class LlmConnectionError(LlmError):
    """Raised when the LLM service is unreachable."""


class LlmParseError(LlmError):
    """Raised when the LLM response cannot be parsed into the expected model.

    Attributes:
        raw_response: The raw text that failed to parse.
    """

    def __init__(self, message: str, raw_response: str = "") -> None:
        super().__init__(message)
        self.raw_response = raw_response
