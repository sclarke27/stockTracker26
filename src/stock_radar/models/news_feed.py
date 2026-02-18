"""Pydantic models for news feed MCP server inputs and outputs."""

from __future__ import annotations

from pydantic import BaseModel, Field


class TopicRelevance(BaseModel):
    """A topic and its relevance to a news article or sentiment summary."""

    topic: str = Field(description="Topic label (e.g. 'Earnings', 'Technology')")
    relevance_score: float = Field(description="Relevance score from 0.0 to 1.0")


class TickerSentiment(BaseModel):
    """Sentiment data for a specific ticker mentioned in a news article."""

    ticker: str = Field(description="Stock ticker symbol")
    relevance_score: float = Field(
        description="How relevant this article is to the ticker (0.0–1.0)"
    )
    sentiment_score: float = Field(
        description="Sentiment score from -1.0 (bearish) to 1.0 (bullish)"
    )
    sentiment_label: str = Field(description="Human-readable label (e.g. 'Somewhat-Bullish')")


class NewsArticle(BaseModel):
    """A single news article with optional sentiment analysis."""

    title: str = Field(description="Article headline")
    url: str = Field(description="Full URL to the article")
    time_published: str = Field(
        description="Publication timestamp (AV format YYYYMMDDTHHMMSS or RSS RFC 822)"
    )
    authors: list[str] = Field(default_factory=list, description="List of author names")
    summary: str = Field(description="Article summary or description")
    source: str = Field(description="Publishing outlet name")
    source_domain: str = Field(description="Domain of the publishing outlet")
    topics: list[TopicRelevance] = Field(
        default_factory=list, description="Topics associated with this article"
    )
    overall_sentiment_score: float = Field(
        description="Overall sentiment score from -1.0 (bearish) to 1.0 (bullish)"
    )
    overall_sentiment_label: str = Field(description="Human-readable overall sentiment label")
    ticker_sentiment: list[TickerSentiment] = Field(
        default_factory=list, description="Per-ticker sentiment breakdowns"
    )


class NewsResponse(BaseModel):
    """Response for news retrieval tools."""

    ticker: str | None = Field(default=None, description="Ticker filter used, if any")
    query: str | None = Field(default=None, description="Search query used, if any")
    articles: list[NewsArticle] = Field(description="Fetched news articles")
    total_fetched: int = Field(description="Total number of articles returned")
    source: str = Field(description="Data source identifier ('alpha_vantage' or 'rss')")


class SentimentBreakdown(BaseModel):
    """Bullish/bearish/neutral article counts for a sentiment summary."""

    bullish: int = Field(description="Number of bullish articles")
    bearish: int = Field(description="Number of bearish articles")
    neutral: int = Field(description="Number of neutral articles")


class SentimentSummaryResponse(BaseModel):
    """Aggregated sentiment summary for a ticker over a time window."""

    ticker: str = Field(description="Stock ticker symbol")
    time_from: str | None = Field(
        default=None, description="Start of the time window (if specified)"
    )
    time_to: str | None = Field(default=None, description="End of the time window (if specified)")
    article_count: int = Field(description="Total number of articles analyzed")
    average_sentiment_score: float = Field(description="Mean sentiment score across all articles")
    average_sentiment_label: str = Field(
        description="Human-readable label for the average sentiment"
    )
    breakdown: SentimentBreakdown = Field(description="Bullish/bearish/neutral counts")
    top_topics: list[TopicRelevance] = Field(description="Most relevant topics across all articles")
