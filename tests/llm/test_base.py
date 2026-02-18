"""Tests for the abstract LLM client base class."""

from __future__ import annotations

import pytest
from pydantic import BaseModel, Field

from stock_radar.llm.base import LlmClient
from stock_radar.llm.exceptions import LlmParseError
from stock_radar.llm.models import LlmRequest, LlmResponse, LlmUsage

# ---------------------------------------------------------------------------
# Test helper: a minimal Pydantic model used as a response_model target
# ---------------------------------------------------------------------------


class _SampleOutput(BaseModel):
    """A trivial model for testing structured output parsing."""

    answer: int = Field(description="The answer")
    reasoning: str = Field(description="Explanation")


# ---------------------------------------------------------------------------
# Test helper: a concrete subclass of the abstract LlmClient
# ---------------------------------------------------------------------------


class _FakeLlmClient(LlmClient):
    """Concrete LlmClient that returns a canned response for testing."""

    def __init__(self, canned_content: str) -> None:
        self._canned_content = canned_content

    async def generate(self, request: LlmRequest) -> LlmResponse:
        """Return a canned LlmResponse."""
        return LlmResponse(
            content=self._canned_content,
            model="fake-model",
            usage=LlmUsage(),
            duration_ms=1.0,
        )


# ---------------------------------------------------------------------------
# Tests for _extract_json
# ---------------------------------------------------------------------------


class TestExtractJson:
    """Tests for LlmClient._extract_json static method."""

    def test_extracts_from_json_code_block(self) -> None:
        """Extracts JSON from a ```json ... ``` markdown code block."""
        text = 'Some preamble\n```json\n{"key": "value"}\n```\nSome postamble'
        assert LlmClient._extract_json(text) == '{"key": "value"}'

    def test_extracts_from_plain_code_block(self) -> None:
        """Extracts JSON from a ``` ... ``` code block without a language tag."""
        text = 'Here is the output:\n```\n{"key": "value"}\n```'
        assert LlmClient._extract_json(text) == '{"key": "value"}'

    def test_extracts_bare_json_object(self) -> None:
        """Extracts the first { ... } block when no code fence is present."""
        text = 'The result is {"answer": 42, "reasoning": "because"} end.'
        result = LlmClient._extract_json(text)
        assert result == '{"answer": 42, "reasoning": "because"}'

    def test_returns_raw_text_when_no_json_found(self) -> None:
        """Returns stripped raw text when nothing looks like JSON."""
        text = "  no json here  "
        assert LlmClient._extract_json(text) == "no json here"


# ---------------------------------------------------------------------------
# Tests for _parse_response
# ---------------------------------------------------------------------------


class TestParseResponse:
    """Tests for LlmClient._parse_response static method."""

    def test_parses_valid_json_into_model(self) -> None:
        """Parses well-formed JSON into the target Pydantic model."""
        raw = '{"answer": 42, "reasoning": "it is the answer"}'
        result = LlmClient._parse_response(raw, _SampleOutput)
        assert result.answer == 42
        assert result.reasoning == "it is the answer"

    def test_raises_parse_error_on_invalid_json(self) -> None:
        """Raises LlmParseError when the text is not valid JSON."""
        with pytest.raises(LlmParseError, match="Failed to parse JSON") as exc_info:
            LlmClient._parse_response("not json {{{", _SampleOutput)
        assert exc_info.value.raw_response == "not json {{{"

    def test_raises_parse_error_on_validation_failure(self) -> None:
        """Raises LlmParseError when JSON is valid but fails model validation."""
        raw = '{"answer": "not_an_int", "reasoning": "oops"}'
        with pytest.raises(LlmParseError, match="failed validation") as exc_info:
            LlmClient._parse_response(raw, _SampleOutput)
        assert exc_info.value.raw_response == raw


# ---------------------------------------------------------------------------
# Tests for generate_structured (integration with generate)
# ---------------------------------------------------------------------------


class TestGenerateStructured:
    """Tests for LlmClient.generate_structured concrete method."""

    async def test_calls_generate_and_parses_result(self) -> None:
        """generate_structured delegates to generate() and parses the output."""
        canned = '{"answer": 7, "reasoning": "lucky number"}'
        client = _FakeLlmClient(canned_content=canned)
        request = LlmRequest(
            messages=[],
            temperature=0.0,
        )
        result = await client.generate_structured(request, _SampleOutput)
        assert isinstance(result, _SampleOutput)
        assert result.answer == 7
        assert result.reasoning == "lucky number"

    async def test_handles_markdown_wrapped_response(self) -> None:
        """generate_structured unwraps markdown code blocks before parsing."""
        canned = '```json\n{"answer": 99, "reasoning": "wrapped"}\n```'
        client = _FakeLlmClient(canned_content=canned)
        request = LlmRequest(messages=[])
        result = await client.generate_structured(request, _SampleOutput)
        assert result.answer == 99

    async def test_raises_parse_error_on_bad_response(self) -> None:
        """generate_structured raises LlmParseError when generate returns garbage."""
        client = _FakeLlmClient(canned_content="totally not json")
        request = LlmRequest(messages=[])
        with pytest.raises(LlmParseError):
            await client.generate_structured(request, _SampleOutput)


# ---------------------------------------------------------------------------
# Tests for LlmParseError
# ---------------------------------------------------------------------------


class TestLlmParseError:
    """Tests for the LlmParseError exception."""

    def test_stores_raw_response(self) -> None:
        """LlmParseError preserves the raw_response that caused the failure."""
        err = LlmParseError("bad parse", raw_response="raw text here")
        assert str(err) == "bad parse"
        assert err.raw_response == "raw text here"

    def test_raw_response_defaults_to_empty(self) -> None:
        """LlmParseError.raw_response defaults to an empty string."""
        err = LlmParseError("bad parse")
        assert err.raw_response == ""
