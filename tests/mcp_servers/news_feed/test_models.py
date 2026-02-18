"""Tests for news feed Pydantic models."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from stock_radar.models.news_feed import (
    NewsArticle,
    NewsResponse,
    SentimentBreakdown,
    SentimentSummaryResponse,
    TickerSentiment,
    TopicRelevance,
)


class TestTopicRelevance:
    """Tests for TopicRelevance model."""

    def test_valid_construction(self) -> None:
        topic = TopicRelevance(topic="Earnings", relevance_score=0.85)
        assert topic.topic == "Earnings"
        assert topic.relevance_score == 0.85

    def test_missing_topic_raises(self) -> None:
        with pytest.raises(ValidationError):
            TopicRelevance(relevance_score=0.5)  # type: ignore[call-arg]

    def test_missing_relevance_score_raises(self) -> None:
        with pytest.raises(ValidationError):
            TopicRelevance(topic="Technology")  # type: ignore[call-arg]

    def test_serialization_roundtrip(self) -> None:
        topic = TopicRelevance(topic="IPO", relevance_score=0.6)
        restored = TopicRelevance.model_validate_json(topic.model_dump_json())
        assert restored == topic


class TestTickerSentiment:
    """Tests for TickerSentiment model."""

    def test_valid_construction(self) -> None:
        ts = TickerSentiment(
            ticker="AAPL",
            relevance_score=0.9,
            sentiment_score=0.35,
            sentiment_label="Somewhat-Bullish",
        )
        assert ts.ticker == "AAPL"
        assert ts.relevance_score == 0.9
        assert ts.sentiment_score == 0.35
        assert ts.sentiment_label == "Somewhat-Bullish"

    def test_missing_ticker_raises(self) -> None:
        with pytest.raises(ValidationError):
            TickerSentiment(  # type: ignore[call-arg]
                relevance_score=0.5,
                sentiment_score=0.1,
                sentiment_label="Neutral",
            )

    def test_serialization_roundtrip(self) -> None:
        ts = TickerSentiment(
            ticker="MSFT",
            relevance_score=0.7,
            sentiment_score=-0.2,
            sentiment_label="Somewhat-Bearish",
        )
        restored = TickerSentiment.model_validate_json(ts.model_dump_json())
        assert restored == ts


class TestNewsArticle:
    """Tests for NewsArticle model."""

    def test_valid_construction_minimal(self) -> None:
        article = NewsArticle(
            title="Apple Reports Record Q4",
            url="https://example.com/article",
            time_published="20240115T120000",
            summary="Apple beats estimates.",
            source="Bloomberg",
            source_domain="bloomberg.com",
            overall_sentiment_score=0.45,
            overall_sentiment_label="Bullish",
        )
        assert article.title == "Apple Reports Record Q4"
        assert article.authors == []
        assert article.topics == []
        assert article.ticker_sentiment == []

    def test_valid_construction_full(self) -> None:
        article = NewsArticle(
            title="Market Update",
            url="https://news.com/update",
            time_published="20240115T093000",
            authors=["Jane Doe", "John Smith"],
            summary="Markets rallied today.",
            source="Reuters",
            source_domain="reuters.com",
            topics=[TopicRelevance(topic="Financial Markets", relevance_score=0.95)],
            overall_sentiment_score=0.25,
            overall_sentiment_label="Somewhat-Bullish",
            ticker_sentiment=[
                TickerSentiment(
                    ticker="SPY",
                    relevance_score=0.8,
                    sentiment_score=0.25,
                    sentiment_label="Somewhat-Bullish",
                )
            ],
        )
        assert len(article.authors) == 2
        assert len(article.topics) == 1
        assert len(article.ticker_sentiment) == 1

    def test_missing_required_field_raises(self) -> None:
        with pytest.raises(ValidationError):
            NewsArticle(  # type: ignore[call-arg]
                url="https://example.com",
                time_published="20240115T120000",
                summary="Missing title.",
                source="Test",
                source_domain="test.com",
                overall_sentiment_score=0.0,
                overall_sentiment_label="Neutral",
            )

    def test_serialization_roundtrip(self) -> None:
        article = NewsArticle(
            title="Test Article",
            url="https://test.com/1",
            time_published="20240115T080000",
            summary="A test article.",
            source="TestSource",
            source_domain="testsource.com",
            overall_sentiment_score=-0.1,
            overall_sentiment_label="Neutral",
        )
        restored = NewsArticle.model_validate_json(article.model_dump_json())
        assert restored == article


class TestNewsResponse:
    """Tests for NewsResponse model."""

    def test_valid_with_ticker(self) -> None:
        response = NewsResponse(
            ticker="AAPL",
            articles=[],
            total_fetched=0,
            source="alpha_vantage",
        )
        assert response.ticker == "AAPL"
        assert response.query is None
        assert response.source == "alpha_vantage"

    def test_valid_with_query(self) -> None:
        response = NewsResponse(
            query="artificial intelligence",
            articles=[],
            total_fetched=0,
            source="rss",
        )
        assert response.ticker is None
        assert response.query == "artificial intelligence"

    def test_ticker_and_query_both_none(self) -> None:
        """Both ticker and query may be None — valid for some use cases."""
        response = NewsResponse(articles=[], total_fetched=0, source="alpha_vantage")
        assert response.ticker is None
        assert response.query is None

    def test_serialization_roundtrip(self) -> None:
        response = NewsResponse(
            ticker="TSLA",
            articles=[],
            total_fetched=0,
            source="alpha_vantage",
        )
        restored = NewsResponse.model_validate_json(response.model_dump_json())
        assert restored == response


class TestSentimentBreakdown:
    """Tests for SentimentBreakdown model."""

    def test_valid_construction(self) -> None:
        bd = SentimentBreakdown(bullish=10, bearish=3, neutral=7)
        assert bd.bullish == 10
        assert bd.bearish == 3
        assert bd.neutral == 7

    def test_missing_field_raises(self) -> None:
        with pytest.raises(ValidationError):
            SentimentBreakdown(bullish=5, bearish=2)  # type: ignore[call-arg]


class TestSentimentSummaryResponse:
    """Tests for SentimentSummaryResponse model."""

    def test_valid_construction(self) -> None:
        summary = SentimentSummaryResponse(
            ticker="NVDA",
            article_count=50,
            average_sentiment_score=0.32,
            average_sentiment_label="Somewhat-Bullish",
            breakdown=SentimentBreakdown(bullish=30, bearish=8, neutral=12),
            top_topics=[TopicRelevance(topic="Technology", relevance_score=0.9)],
        )
        assert summary.ticker == "NVDA"
        assert summary.time_from is None
        assert summary.time_to is None
        assert summary.article_count == 50
        assert summary.average_sentiment_score == 0.32
        assert len(summary.top_topics) == 1

    def test_valid_with_time_window(self) -> None:
        summary = SentimentSummaryResponse(
            ticker="MSFT",
            time_from="20240101T000000",
            time_to="20240131T235959",
            article_count=100,
            average_sentiment_score=-0.05,
            average_sentiment_label="Neutral",
            breakdown=SentimentBreakdown(bullish=40, bearish=45, neutral=15),
            top_topics=[],
        )
        assert summary.time_from == "20240101T000000"
        assert summary.time_to == "20240131T235959"

    def test_missing_required_field_raises(self) -> None:
        with pytest.raises(ValidationError):
            SentimentSummaryResponse(  # type: ignore[call-arg]
                ticker="AAPL",
                article_count=10,
                # missing average_sentiment_score, average_sentiment_label, breakdown, top_topics
            )

    def test_serialization_roundtrip(self) -> None:
        summary = SentimentSummaryResponse(
            ticker="GOOG",
            article_count=25,
            average_sentiment_score=0.15,
            average_sentiment_label="Somewhat-Bullish",
            breakdown=SentimentBreakdown(bullish=15, bearish=5, neutral=5),
            top_topics=[TopicRelevance(topic="Earnings", relevance_score=0.75)],
        )
        restored = SentimentSummaryResponse.model_validate_json(summary.model_dump_json())
        assert restored == summary
