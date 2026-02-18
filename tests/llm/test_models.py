"""Tests for LLM request/response Pydantic models."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from stock_radar.llm.models import LlmMessage, LlmRequest, LlmResponse, LlmUsage


class TestLlmMessage:
    """Tests for LlmMessage model."""

    @pytest.mark.parametrize("role", ["system", "user", "assistant"])
    def test_construction_with_valid_roles(self, role: str) -> None:
        """LlmMessage accepts system, user, and assistant roles."""
        msg = LlmMessage(role=role, content="Hello")
        assert msg.role == role
        assert msg.content == "Hello"

    def test_rejects_invalid_role(self) -> None:
        """LlmMessage raises ValidationError for an unrecognized role."""
        with pytest.raises(ValidationError, match="role"):
            LlmMessage(role="tool", content="Hello")


class TestLlmUsage:
    """Tests for LlmUsage model."""

    def test_defaults_to_zero(self) -> None:
        """LlmUsage fields default to 0 when not provided."""
        usage = LlmUsage()
        assert usage.prompt_tokens == 0
        assert usage.completion_tokens == 0
        assert usage.total_tokens == 0

    def test_rejects_negative_tokens(self) -> None:
        """LlmUsage raises ValidationError for negative token counts."""
        with pytest.raises(ValidationError, match="prompt_tokens"):
            LlmUsage(prompt_tokens=-1)


class TestLlmRequest:
    """Tests for LlmRequest model."""

    def test_defaults(self) -> None:
        """LlmRequest provides sensible defaults for optional fields."""
        messages = [LlmMessage(role="user", content="Hi")]
        request = LlmRequest(messages=messages)
        assert request.temperature == pytest.approx(0.1)
        assert request.max_tokens == 4096
        assert request.response_format == "json"

    def test_temperature_lower_bound(self) -> None:
        """LlmRequest rejects temperature below 0.0."""
        with pytest.raises(ValidationError, match="temperature"):
            LlmRequest(
                messages=[LlmMessage(role="user", content="Hi")],
                temperature=-0.1,
            )

    def test_temperature_upper_bound(self) -> None:
        """LlmRequest rejects temperature above 2.0."""
        with pytest.raises(ValidationError, match="temperature"):
            LlmRequest(
                messages=[LlmMessage(role="user", content="Hi")],
                temperature=2.1,
            )

    def test_max_tokens_must_be_positive(self) -> None:
        """LlmRequest rejects max_tokens of 0 or negative."""
        with pytest.raises(ValidationError, match="max_tokens"):
            LlmRequest(
                messages=[LlmMessage(role="user", content="Hi")],
                max_tokens=0,
            )


class TestLlmResponse:
    """Tests for LlmResponse model."""

    def test_construction(self) -> None:
        """LlmResponse can be constructed with all required fields."""
        usage = LlmUsage(prompt_tokens=10, completion_tokens=20, total_tokens=30)
        response = LlmResponse(
            content='{"answer": 42}',
            model="llama3.1:8b",
            usage=usage,
            duration_ms=150.5,
        )
        assert response.content == '{"answer": 42}'
        assert response.model == "llama3.1:8b"
        assert response.usage.total_tokens == 30
        assert response.duration_ms == pytest.approx(150.5)

    def test_serialization_round_trip(self) -> None:
        """LlmResponse survives JSON serialization and deserialization."""
        usage = LlmUsage(prompt_tokens=10, completion_tokens=20, total_tokens=30)
        original = LlmResponse(
            content='{"answer": 42}',
            model="llama3.1:8b",
            usage=usage,
            duration_ms=150.5,
        )
        json_str = original.model_dump_json()
        restored = LlmResponse.model_validate_json(json_str)
        assert restored == original
