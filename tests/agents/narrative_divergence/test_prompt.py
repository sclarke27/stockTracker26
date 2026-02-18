"""Tests for Narrative vs Price Divergence prompt builders."""

from __future__ import annotations

from stock_radar.agents.narrative_divergence.prompt import (
    build_messages,
    build_system_prompt,
    build_user_prompt,
)
from stock_radar.llm.models import LlmMessage


class TestBuildSystemPrompt:
    """Tests for build_system_prompt()."""

    def test_returns_string(self) -> None:
        prompt = build_system_prompt()
        assert isinstance(prompt, str)
        assert len(prompt) > 100

    def test_contains_schema(self) -> None:
        """System prompt must embed the NarrativeAnalysis JSON schema."""
        prompt = build_system_prompt()
        # Schema includes field names from NarrativeAnalysis
        assert "divergence_detected" in prompt
        assert "divergence_strength" in prompt
        assert "contagion" not in prompt.lower()  # not the contagion agent

    def test_contains_analysis_instructions(self) -> None:
        prompt = build_system_prompt()
        assert "sentiment" in prompt.lower()
        assert "price" in prompt.lower()

    def test_contains_json_output_instruction(self) -> None:
        prompt = build_system_prompt()
        assert "json" in prompt.lower()


class TestBuildUserPrompt:
    """Tests for build_user_prompt()."""

    def test_returns_string(self) -> None:
        prompt = build_user_prompt(
            ticker="AAPL",
            sentiment_score=0.45,
            article_count=12,
            average_sentiment_label="Somewhat-Bullish",
            price_return_30d=-0.08,
            price_return_7d=-0.03,
            top_articles=[],
        )
        assert isinstance(prompt, str)

    def test_contains_ticker(self) -> None:
        prompt = build_user_prompt(
            ticker="TSLA",
            sentiment_score=0.6,
            article_count=20,
            average_sentiment_label="Bullish",
            price_return_30d=0.05,
            price_return_7d=0.02,
            top_articles=[],
        )
        assert "TSLA" in prompt

    def test_contains_sentiment_score(self) -> None:
        prompt = build_user_prompt(
            ticker="AAPL",
            sentiment_score=0.72,
            article_count=8,
            average_sentiment_label="Somewhat-Bullish",
            price_return_30d=-0.05,
            price_return_7d=-0.01,
            top_articles=[],
        )
        assert "0.72" in prompt

    def test_contains_price_returns(self) -> None:
        prompt = build_user_prompt(
            ticker="AAPL",
            sentiment_score=0.4,
            article_count=5,
            average_sentiment_label="Neutral",
            price_return_30d=-0.12,
            price_return_7d=0.03,
            top_articles=[],
        )
        assert "-0.12" in prompt or "-12" in prompt or "12" in prompt

    def test_includes_article_titles_when_present(self) -> None:
        articles = [
            {"title": "Apple reports record revenue", "sentiment_score": 0.8},
        ]
        prompt = build_user_prompt(
            ticker="AAPL",
            sentiment_score=0.7,
            article_count=1,
            average_sentiment_label="Bullish",
            price_return_30d=-0.05,
            price_return_7d=-0.02,
            top_articles=articles,
        )
        assert "Apple reports record revenue" in prompt

    def test_no_articles_still_produces_valid_prompt(self) -> None:
        prompt = build_user_prompt(
            ticker="AAPL",
            sentiment_score=0.3,
            article_count=0,
            average_sentiment_label="Neutral",
            price_return_30d=0.0,
            price_return_7d=0.0,
            top_articles=[],
        )
        assert "AAPL" in prompt
        assert len(prompt) > 50


class TestBuildMessages:
    """Tests for build_messages()."""

    def test_returns_two_messages(self) -> None:
        messages = build_messages(
            ticker="AAPL",
            sentiment_score=0.4,
            article_count=10,
            average_sentiment_label="Somewhat-Bullish",
            price_return_30d=-0.05,
            price_return_7d=-0.01,
            top_articles=[],
        )
        assert len(messages) == 2

    def test_first_message_is_system(self) -> None:
        messages = build_messages(
            ticker="AAPL",
            sentiment_score=0.4,
            article_count=10,
            average_sentiment_label="Somewhat-Bullish",
            price_return_30d=-0.05,
            price_return_7d=-0.01,
            top_articles=[],
        )
        assert isinstance(messages[0], LlmMessage)
        assert messages[0].role == "system"

    def test_second_message_is_user(self) -> None:
        messages = build_messages(
            ticker="AAPL",
            sentiment_score=0.4,
            article_count=10,
            average_sentiment_label="Somewhat-Bullish",
            price_return_30d=-0.05,
            price_return_7d=-0.01,
            top_articles=[],
        )
        assert messages[1].role == "user"
        assert "AAPL" in messages[1].content
