"""Tests for Earnings Linguist prompt construction."""

from __future__ import annotations

from stock_radar.agents.earnings_linguist.prompt import (
    build_messages,
    build_system_prompt,
    build_user_prompt,
)


class TestBuildSystemPrompt:
    def test_includes_json_schema(self) -> None:
        """System prompt includes the EarningsAnalysis JSON schema."""
        prompt = build_system_prompt()
        assert "overall_sentiment" in prompt
        assert "sentiment_indicators" in prompt
        assert "reasoning_summary" in prompt

    def test_includes_instructions(self) -> None:
        """System prompt includes analysis instructions."""
        prompt = build_system_prompt()
        assert "hedging" in prompt.lower()
        assert "confidence" in prompt.lower()


class TestBuildUserPrompt:
    def test_includes_ticker_and_transcript(self) -> None:
        prompt = build_user_prompt(
            ticker="AAPL",
            transcript="Tim Cook: Revenue grew 10%.",
        )
        assert "AAPL" in prompt
        assert "Tim Cook: Revenue grew 10%." in prompt

    def test_includes_prior_transcript_when_provided(self) -> None:
        prompt = build_user_prompt(
            ticker="AAPL",
            transcript="Current quarter.",
            prior_transcript="Prior quarter.",
        )
        assert "Prior quarter." in prompt
        assert "prior" in prompt.lower() or "previous" in prompt.lower()

    def test_no_prior_section_when_none(self) -> None:
        prompt = build_user_prompt(
            ticker="AAPL",
            transcript="Current quarter only.",
        )
        assert "previous quarter" not in prompt.lower() or "prior quarter" not in prompt.lower()


class TestBuildMessages:
    def test_returns_system_and_user_messages(self) -> None:
        messages = build_messages(
            ticker="MSFT",
            transcript="Satya Nadella: Cloud growth strong.",
        )
        assert len(messages) == 2
        assert messages[0].role == "system"
        assert messages[1].role == "user"
        assert "MSFT" in messages[1].content
