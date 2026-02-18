"""Pydantic models for LLM request/response contracts."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class LlmMessage(BaseModel):
    """A single message in a conversation.

    Attributes:
        role: The role of the message sender (system, user, or assistant).
        content: The text content of the message.
    """

    role: Literal["system", "user", "assistant"] = Field(description="Message role")
    content: str = Field(description="Message content")


class LlmUsage(BaseModel):
    """Token usage statistics from an LLM call.

    Attributes:
        prompt_tokens: Number of input tokens consumed.
        completion_tokens: Number of output tokens generated.
        total_tokens: Total tokens consumed (prompt + completion).
    """

    prompt_tokens: int = Field(default=0, ge=0, description="Input tokens consumed")
    completion_tokens: int = Field(default=0, ge=0, description="Output tokens generated")
    total_tokens: int = Field(default=0, ge=0, description="Total tokens consumed")


class LlmRequest(BaseModel):
    """Request payload for an LLM generation call.

    Attributes:
        messages: Ordered conversation messages to send to the model.
        temperature: Sampling temperature controlling randomness (0.0-2.0).
        max_tokens: Maximum number of tokens to generate.
        response_format: Expected response format (json or text).
    """

    messages: list[LlmMessage] = Field(description="Conversation messages")
    temperature: float = Field(default=0.1, ge=0.0, le=2.0, description="Sampling temperature")
    max_tokens: int = Field(default=4096, gt=0, description="Maximum tokens to generate")
    response_format: Literal["json", "text"] = Field(
        default="json", description="Expected response format"
    )


class LlmResponse(BaseModel):
    """Response from an LLM generation call.

    Attributes:
        content: The generated text content.
        model: Identifier of the model that produced this response.
        usage: Token usage statistics for the call.
        duration_ms: Wall-clock inference time in milliseconds.
    """

    content: str = Field(description="Generated text content")
    model: str = Field(description="Model identifier that produced this response")
    usage: LlmUsage = Field(description="Token usage statistics")
    duration_ms: float = Field(ge=0.0, description="Wall-clock inference time in milliseconds")
