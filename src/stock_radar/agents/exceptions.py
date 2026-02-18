"""Exception hierarchy for analysis agents."""

from __future__ import annotations


class AgentError(Exception):
    """Base exception for agent execution errors."""


class EscalationError(AgentError):
    """Raised when LLM escalation fails (e.g. no Anthropic key configured)."""


class TranscriptNotFoundError(AgentError):
    """Raised when an earnings transcript is not available."""
