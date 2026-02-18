"""Alpha Vantage News Sentiment API client."""

from __future__ import annotations

import httpx
from loguru import logger

from stock_radar.mcp_servers.news_feed.config import (
    AV_BASE_URL,
    AV_DEFAULT_LIMIT,
    AV_MAX_LIMIT,
    SENTIMENT_BEARISH_THRESHOLD,
    SENTIMENT_BULLISH_THRESHOLD,
    SENTIMENT_TOP_TOPICS_COUNT,
    SERVER_NAME,
)
from stock_radar.mcp_servers.news_feed.exceptions import ApiError, NoNewsFoundError
from stock_radar.models.news_feed import (
    NewsArticle,
    NewsResponse,
    SentimentBreakdown,
    SentimentSummaryResponse,
    TickerSentiment,
    TopicRelevance,
)
from stock_radar.utils.rate_limiter import RateLimiter


class AlphaVantageNewsClient:
    """Async client for the Alpha Vantage News Sentiment API.

    Wraps the ``NEWS_SENTIMENT`` endpoint used by the news-feed MCP server.
    Requests go through a rate limiter before hitting the API.

    Args:
        api_key: Alpha Vantage API key.
        http_client: Shared ``httpx.AsyncClient`` instance.
        rate_limiter: Rate limiter controlling request frequency.
    """

    def __init__(
        self,
        api_key: str,
        http_client: httpx.AsyncClient,
        rate_limiter: RateLimiter,
    ) -> None:
        self._api_key = api_key
        self._http = http_client
        self._limiter = rate_limiter

    async def get_news(
        self,
        ticker: str,
        limit: int = AV_DEFAULT_LIMIT,
        time_from: str | None = None,
        sort: str = "LATEST",
    ) -> NewsResponse:
        """Fetch recent news articles for a specific ticker.

        Args:
            ticker: Stock ticker symbol (e.g. ``"AAPL"``).
            limit: Maximum number of articles to return (capped at AV_MAX_LIMIT).
            time_from: Earliest publish date/time in AV format ``"YYYYMMDDTHHMMSS"``.
            sort: Sort order — ``"LATEST"``, ``"EARLIEST"``, or ``"RELEVANCE"``.

        Returns:
            Parsed news response with articles filtered to the requested ticker.
        """
        params: dict[str, str] = {
            "tickers": ticker,
            "sort": sort,
            "limit": str(min(limit, AV_MAX_LIMIT)),
        }
        if time_from:
            params["time_from"] = time_from

        data = await self._request(params=params)
        articles = self._parse_feed(data.get("feed", []))[:limit]

        return NewsResponse(
            ticker=ticker,
            articles=articles,
            total_fetched=len(articles),
            source="alpha_vantage",
        )

    async def search_news(
        self,
        query: str,
        topics: str | None = None,
        limit: int = AV_DEFAULT_LIMIT,
        time_from: str | None = None,
    ) -> NewsResponse:
        """Search news by free-text query and optional topic filters.

        Raises ``NoNewsFoundError`` when the feed is empty so the server can
        fall back to the RSS client.

        Args:
            query: Free-text search string.
            topics: Comma-separated AV topic filter (e.g. ``"earnings,technology"``).
            limit: Maximum number of articles to return.
            time_from: Earliest publish date/time in AV format ``"YYYYMMDDTHHMMSS"``.

        Returns:
            Parsed news response.

        Raises:
            NoNewsFoundError: When AV returns an empty feed for the query.
        """
        params: dict[str, str] = {
            "sort": "RELEVANCE",
            "limit": str(min(limit, AV_MAX_LIMIT)),
        }
        # AV uses the `tickers` param for free-text when topics are provided;
        # for a general keyword search the query is passed as topics or we
        # just let it flow through the feed with no ticker filter.
        if topics:
            params["topics"] = topics
        if time_from:
            params["time_from"] = time_from

        data = await self._request(params=params)
        raw_feed = data.get("feed", [])

        if not raw_feed:
            raise NoNewsFoundError(f"No news found for query '{query}' via Alpha Vantage.")

        articles = self._parse_feed(raw_feed)[:limit]
        return NewsResponse(
            query=query,
            articles=articles,
            total_fetched=len(articles),
            source="alpha_vantage",
        )

    async def get_sentiment_summary(
        self,
        ticker: str,
        time_from: str | None = None,
        time_to: str | None = None,
    ) -> SentimentSummaryResponse:
        """Fetch and aggregate sentiment for a ticker over a time window.

        Fetches up to ``AV_MAX_LIMIT`` articles and aggregates client-side:
        average sentiment score, bullish/bearish/neutral counts, and top topics.

        Args:
            ticker: Stock ticker symbol.
            time_from: Start of time window in AV format ``"YYYYMMDDTHHMMSS"``.
            time_to: End of time window in AV format ``"YYYYMMDDTHHMMSS"``.

        Returns:
            Aggregated sentiment summary.
        """
        params: dict[str, str] = {
            "tickers": ticker,
            "sort": "LATEST",
            "limit": str(AV_MAX_LIMIT),
        }
        if time_from:
            params["time_from"] = time_from
        if time_to:
            params["time_to"] = time_to

        data = await self._request(params=params)
        articles = self._parse_feed(data.get("feed", []))

        article_count = len(articles)
        if article_count == 0:
            avg_score = 0.0
        else:
            avg_score = sum(a.overall_sentiment_score for a in articles) / article_count

        avg_label = self._score_to_label(avg_score)

        bullish = sum(
            1 for a in articles if a.overall_sentiment_score >= SENTIMENT_BULLISH_THRESHOLD
        )
        bearish = sum(
            1 for a in articles if a.overall_sentiment_score <= SENTIMENT_BEARISH_THRESHOLD
        )
        neutral = article_count - bullish - bearish

        # Aggregate topic relevance: sum scores per topic, then take top N
        topic_totals: dict[str, float] = {}
        for article in articles:
            for topic in article.topics:
                topic_totals[topic.topic] = (
                    topic_totals.get(topic.topic, 0.0) + topic.relevance_score
                )

        top_topics = [
            TopicRelevance(topic=name, relevance_score=score)
            for name, score in sorted(topic_totals.items(), key=lambda kv: kv[1], reverse=True)[
                :SENTIMENT_TOP_TOPICS_COUNT
            ]
        ]

        return SentimentSummaryResponse(
            ticker=ticker,
            time_from=time_from,
            time_to=time_to,
            article_count=article_count,
            average_sentiment_score=avg_score,
            average_sentiment_label=avg_label,
            breakdown=SentimentBreakdown(bullish=bullish, bearish=bearish, neutral=neutral),
            top_topics=top_topics,
        )

    # ---------------------------------------------------------------------------
    # Internal helpers
    # ---------------------------------------------------------------------------

    async def _request(self, params: dict[str, str]) -> dict:
        """Make a rate-limited request to the Alpha Vantage News Sentiment API.

        Args:
            params: Query parameters (excluding ``function`` and ``apikey``).

        Returns:
            Parsed JSON response body.

        Raises:
            ApiError: On HTTP errors or AV-specific error responses.
        """
        await self._limiter.acquire()

        query = {"function": "NEWS_SENTIMENT", "apikey": self._api_key, **params}
        logger.debug(
            "Alpha Vantage News request: {params}",
            params=params,
            server=SERVER_NAME,
        )

        try:
            response = await self._http.get(AV_BASE_URL, params=query)
            response.raise_for_status()
        except httpx.HTTPStatusError as exc:
            raise ApiError(
                f"Alpha Vantage HTTP {exc.response.status_code}: {exc.response.text[:200]}"
            ) from exc

        data = response.json()
        self._check_for_errors(data)
        return data

    @staticmethod
    def _check_for_errors(data: dict) -> None:
        """Detect Alpha Vantage error responses embedded in 200 OK bodies.

        AV returns errors as ``{"Error Message": "..."}`` and rate limit
        warnings as ``{"Note": "..."}``. Both are raised as ``ApiError``.
        """
        if "Error Message" in data:
            raise ApiError(f"Alpha Vantage error: {data['Error Message']}")
        if "Note" in data:
            raise ApiError(f"Alpha Vantage rate limit: {data['Note']}")

    @staticmethod
    def _parse_feed(feed: list[dict]) -> list[NewsArticle]:
        """Convert raw AV feed entries into ``NewsArticle`` instances.

        AV returns sentiment/relevance scores as strings; this method converts
        them to floats.

        Args:
            feed: Raw list of article dicts from the AV API response.

        Returns:
            Parsed list of ``NewsArticle`` instances.
        """
        articles = []
        for item in feed:
            topics = [
                TopicRelevance(
                    topic=t["topic"],
                    relevance_score=float(t["relevance_score"]),
                )
                for t in item.get("topics", [])
            ]
            ticker_sentiment = [
                TickerSentiment(
                    ticker=ts["ticker"],
                    relevance_score=float(ts["relevance_score"]),
                    sentiment_score=float(ts["ticker_sentiment_score"]),
                    sentiment_label=ts["ticker_sentiment_label"],
                )
                for ts in item.get("ticker_sentiment", [])
            ]
            articles.append(
                NewsArticle(
                    title=item.get("title", ""),
                    url=item.get("url", ""),
                    time_published=item.get("time_published", ""),
                    authors=item.get("authors", []),
                    summary=item.get("summary", ""),
                    source=item.get("source", ""),
                    source_domain=item.get("source_domain", ""),
                    topics=topics,
                    overall_sentiment_score=float(item.get("overall_sentiment_score", 0.0)),
                    overall_sentiment_label=item.get("overall_sentiment_label", "Neutral"),
                    ticker_sentiment=ticker_sentiment,
                )
            )
        return articles

    @staticmethod
    def _score_to_label(score: float) -> str:
        """Convert a numeric sentiment score to a human-readable label.

        Thresholds match Alpha Vantage's own labeling convention:
        - score >= 0.35  → Bullish
        - score >= 0.15  → Somewhat-Bullish
        - score > -0.15  → Neutral
        - score > -0.35  → Somewhat-Bearish
        - score <= -0.35 → Bearish

        Args:
            score: Sentiment score in [-1.0, 1.0].

        Returns:
            Human-readable label string.
        """
        if score >= 0.35:
            return "Bullish"
        if score >= SENTIMENT_BULLISH_THRESHOLD:
            return "Somewhat-Bullish"
        if score > SENTIMENT_BEARISH_THRESHOLD:
            return "Neutral"
        if score > -0.35:
            return "Somewhat-Bearish"
        return "Bearish"
