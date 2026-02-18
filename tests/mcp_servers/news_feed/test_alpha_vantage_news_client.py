"""Tests for the Alpha Vantage news sentiment client."""

from __future__ import annotations

import httpx
import pytest
import respx

from stock_radar.mcp_servers.news_feed.clients.alpha_vantage_news import (
    AlphaVantageNewsClient,
)
from stock_radar.mcp_servers.news_feed.config import AV_BASE_URL
from stock_radar.mcp_servers.news_feed.exceptions import ApiError, NoNewsFoundError
from stock_radar.utils.rate_limiter import RateLimiter


@pytest.fixture()
def rate_limiter() -> RateLimiter:
    return RateLimiter(requests_per_minute=100, requests_per_day=10_000)


@pytest.fixture()
def av_news_client(rate_limiter: RateLimiter) -> AlphaVantageNewsClient:
    http_client = httpx.AsyncClient()
    return AlphaVantageNewsClient(
        api_key="test-key",
        http_client=http_client,
        rate_limiter=rate_limiter,
    )


# --- Sample API responses ---

SAMPLE_NEWS_RESPONSE = {
    "items": "2",
    "sentiment_score_definition": "x <= -0.35: Bearish; ...",
    "relevance_score_definition": "0 < x <= 1, ...",
    "feed": [
        {
            "title": "Apple Reports Record Quarterly Revenue",
            "url": "https://example.com/apple-revenue",
            "time_published": "20240115T120000",
            "authors": ["Jane Doe"],
            "summary": "Apple exceeded analyst expectations in Q4.",
            "banner_image": "https://example.com/img.jpg",
            "source": "Bloomberg",
            "category_within_source": "Markets",
            "source_domain": "bloomberg.com",
            "topics": [
                {"topic": "Earnings", "relevance_score": "0.8750"},
                {"topic": "Technology", "relevance_score": "0.6000"},
            ],
            "overall_sentiment_score": 0.45,
            "overall_sentiment_label": "Bullish",
            "ticker_sentiment": [
                {
                    "ticker": "AAPL",
                    "relevance_score": "0.9500",
                    "ticker_sentiment_score": "0.3500",
                    "ticker_sentiment_label": "Somewhat-Bullish",
                }
            ],
        },
        {
            "title": "Tech Sector Faces Headwinds",
            "url": "https://example.com/tech-headwinds",
            "time_published": "20240114T090000",
            "authors": [],
            "summary": "Rising interest rates pressure tech valuations.",
            "banner_image": "",
            "source": "Reuters",
            "category_within_source": "Technology",
            "source_domain": "reuters.com",
            "topics": [
                {"topic": "Financial Markets", "relevance_score": "0.9000"},
            ],
            "overall_sentiment_score": -0.25,
            "overall_sentiment_label": "Somewhat-Bearish",
            "ticker_sentiment": [
                {
                    "ticker": "AAPL",
                    "relevance_score": "0.7000",
                    "ticker_sentiment_score": "-0.2000",
                    "ticker_sentiment_label": "Somewhat-Bearish",
                }
            ],
        },
    ],
}

SAMPLE_EMPTY_NEWS_RESPONSE = {
    "items": "0",
    "feed": [],
}

SAMPLE_LARGE_NEWS_RESPONSE = {
    "items": "3",
    "feed": [
        {
            "title": f"Article {i}",
            "url": f"https://example.com/{i}",
            "time_published": "20240115T120000",
            "authors": [],
            "summary": f"Summary {i}",
            "banner_image": "",
            "source": "TestSource",
            "category_within_source": "Markets",
            "source_domain": "testsource.com",
            "topics": [{"topic": "Technology", "relevance_score": str(0.5 + i * 0.1)}],
            "overall_sentiment_score": 0.1 * i,
            "overall_sentiment_label": "Neutral",
            "ticker_sentiment": [],
        }
        for i in range(3)
    ],
}


class TestGetNews:
    """Tests for AlphaVantageNewsClient.get_news()."""

    @respx.mock
    async def test_parses_response_correctly(self, av_news_client: AlphaVantageNewsClient) -> None:
        respx.get(AV_BASE_URL).mock(return_value=httpx.Response(200, json=SAMPLE_NEWS_RESPONSE))
        result = await av_news_client.get_news("AAPL")
        assert result.ticker == "AAPL"
        assert result.source == "alpha_vantage"
        assert result.total_fetched == 2
        assert len(result.articles) == 2

    @respx.mock
    async def test_article_fields_parsed(self, av_news_client: AlphaVantageNewsClient) -> None:
        respx.get(AV_BASE_URL).mock(return_value=httpx.Response(200, json=SAMPLE_NEWS_RESPONSE))
        result = await av_news_client.get_news("AAPL")
        article = result.articles[0]
        assert article.title == "Apple Reports Record Quarterly Revenue"
        assert article.url == "https://example.com/apple-revenue"
        assert article.time_published == "20240115T120000"
        assert article.authors == ["Jane Doe"]
        assert article.source == "Bloomberg"
        assert article.source_domain == "bloomberg.com"
        assert article.overall_sentiment_score == 0.45
        assert article.overall_sentiment_label == "Bullish"

    @respx.mock
    async def test_topic_relevance_scores_converted_to_float(
        self, av_news_client: AlphaVantageNewsClient
    ) -> None:
        respx.get(AV_BASE_URL).mock(return_value=httpx.Response(200, json=SAMPLE_NEWS_RESPONSE))
        result = await av_news_client.get_news("AAPL")
        topics = result.articles[0].topics
        assert len(topics) == 2
        assert topics[0].topic == "Earnings"
        assert topics[0].relevance_score == 0.875
        assert isinstance(topics[0].relevance_score, float)

    @respx.mock
    async def test_ticker_sentiment_scores_converted_to_float(
        self, av_news_client: AlphaVantageNewsClient
    ) -> None:
        respx.get(AV_BASE_URL).mock(return_value=httpx.Response(200, json=SAMPLE_NEWS_RESPONSE))
        result = await av_news_client.get_news("AAPL")
        ts = result.articles[0].ticker_sentiment[0]
        assert ts.ticker == "AAPL"
        assert ts.relevance_score == 0.95
        assert ts.sentiment_score == 0.35
        assert isinstance(ts.relevance_score, float)

    @respx.mock
    async def test_limit_respected(self, av_news_client: AlphaVantageNewsClient) -> None:
        respx.get(AV_BASE_URL).mock(
            return_value=httpx.Response(200, json=SAMPLE_LARGE_NEWS_RESPONSE)
        )
        result = await av_news_client.get_news("AAPL", limit=2)
        assert len(result.articles) == 2

    @respx.mock
    async def test_empty_feed_returns_empty_list(
        self, av_news_client: AlphaVantageNewsClient
    ) -> None:
        respx.get(AV_BASE_URL).mock(
            return_value=httpx.Response(200, json=SAMPLE_EMPTY_NEWS_RESPONSE)
        )
        result = await av_news_client.get_news("UNKNOWN")
        assert result.articles == []
        assert result.total_fetched == 0


class TestSearchNews:
    """Tests for AlphaVantageNewsClient.search_news()."""

    @respx.mock
    async def test_parses_response_correctly(self, av_news_client: AlphaVantageNewsClient) -> None:
        respx.get(AV_BASE_URL).mock(return_value=httpx.Response(200, json=SAMPLE_NEWS_RESPONSE))
        result = await av_news_client.search_news("artificial intelligence")
        assert result.query == "artificial intelligence"
        assert result.source == "alpha_vantage"
        assert result.total_fetched == 2

    @respx.mock
    async def test_empty_feed_raises_no_news_found(
        self, av_news_client: AlphaVantageNewsClient
    ) -> None:
        respx.get(AV_BASE_URL).mock(
            return_value=httpx.Response(200, json=SAMPLE_EMPTY_NEWS_RESPONSE)
        )
        with pytest.raises(NoNewsFoundError):
            await av_news_client.search_news("zzznotreal")


class TestGetSentimentSummary:
    """Tests for AlphaVantageNewsClient.get_sentiment_summary()."""

    @respx.mock
    async def test_aggregate_counts(self, av_news_client: AlphaVantageNewsClient) -> None:
        respx.get(AV_BASE_URL).mock(return_value=httpx.Response(200, json=SAMPLE_NEWS_RESPONSE))
        result = await av_news_client.get_sentiment_summary("AAPL")
        assert result.ticker == "AAPL"
        assert result.article_count == 2
        # One bullish (0.45), one somewhat-bearish (-0.25)
        assert result.breakdown.bullish == 1
        assert result.breakdown.bearish == 1
        assert result.breakdown.neutral == 0

    @respx.mock
    async def test_average_sentiment_score(self, av_news_client: AlphaVantageNewsClient) -> None:
        respx.get(AV_BASE_URL).mock(return_value=httpx.Response(200, json=SAMPLE_NEWS_RESPONSE))
        result = await av_news_client.get_sentiment_summary("AAPL")
        expected_avg = (0.45 + (-0.25)) / 2
        assert abs(result.average_sentiment_score - expected_avg) < 1e-9

    @respx.mock
    async def test_average_sentiment_label_bullish(
        self, av_news_client: AlphaVantageNewsClient
    ) -> None:
        respx.get(AV_BASE_URL).mock(return_value=httpx.Response(200, json=SAMPLE_NEWS_RESPONSE))
        result = await av_news_client.get_sentiment_summary("AAPL")
        # avg = 0.1, which is between -0.15 and 0.15 → Neutral
        assert result.average_sentiment_label == "Neutral"

    @respx.mock
    async def test_top_topics_aggregated(self, av_news_client: AlphaVantageNewsClient) -> None:
        respx.get(AV_BASE_URL).mock(return_value=httpx.Response(200, json=SAMPLE_NEWS_RESPONSE))
        result = await av_news_client.get_sentiment_summary("AAPL")
        topic_names = [t.topic for t in result.top_topics]
        assert "Earnings" in topic_names
        assert "Financial Markets" in topic_names

    @respx.mock
    async def test_time_from_passed_through(self, av_news_client: AlphaVantageNewsClient) -> None:
        respx.get(AV_BASE_URL).mock(return_value=httpx.Response(200, json=SAMPLE_NEWS_RESPONSE))
        result = await av_news_client.get_sentiment_summary("AAPL", time_from="20240101T000000")
        assert result.time_from == "20240101T000000"


class TestErrorHandling:
    """Tests for Alpha Vantage news client error scenarios."""

    @respx.mock
    async def test_api_error_message(self, av_news_client: AlphaVantageNewsClient) -> None:
        respx.get(AV_BASE_URL).mock(
            return_value=httpx.Response(200, json={"Error Message": "Invalid API call."})
        )
        with pytest.raises(ApiError, match="Invalid API call"):
            await av_news_client.get_news("AAPL")

    @respx.mock
    async def test_rate_limit_note(self, av_news_client: AlphaVantageNewsClient) -> None:
        respx.get(AV_BASE_URL).mock(
            return_value=httpx.Response(
                200,
                json={
                    "Note": (
                        "Thank you for using Alpha Vantage! "
                        "Our standard API rate limit is 5 requests per minute."
                    )
                },
            )
        )
        with pytest.raises(ApiError, match="rate limit"):
            await av_news_client.get_news("AAPL")

    @respx.mock
    async def test_http_server_error(self, av_news_client: AlphaVantageNewsClient) -> None:
        respx.get(AV_BASE_URL).mock(return_value=httpx.Response(500, text="Internal Server Error"))
        with pytest.raises(ApiError):
            await av_news_client.get_news("AAPL")
