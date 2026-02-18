"""Integration tests for the news-feed MCP server."""

from __future__ import annotations

import json
from unittest.mock import patch

import httpx
import respx
from fastmcp import Client

from stock_radar.mcp_servers.news_feed.config import AV_BASE_URL, RSS_BASE_URL
from stock_radar.mcp_servers.news_feed.server import create_server
from tests.mcp_servers.conftest import get_tool_text
from tests.mcp_servers.news_feed.test_alpha_vantage_news_client import (
    SAMPLE_EMPTY_NEWS_RESPONSE,
    SAMPLE_NEWS_RESPONSE,
)
from tests.mcp_servers.news_feed.test_rss_client import SAMPLE_RSS_XML

MOCK_ENV = {
    "ALPHA_VANTAGE_API_KEY": "test-av-key",
    "FINNHUB_API_KEY": "test-fh-key",
    "ANTHROPIC_API_KEY": "test-anthropic-key",
    "SEC_EDGAR_EMAIL": "test@example.com",
}


class TestGetNews:
    """Integration tests for the get_news tool."""

    @respx.mock
    async def test_returns_valid_json(self, tmp_db: str) -> None:
        respx.get(AV_BASE_URL).mock(return_value=httpx.Response(200, json=SAMPLE_NEWS_RESPONSE))
        with (
            patch.dict("os.environ", MOCK_ENV),
            patch(
                "stock_radar.mcp_servers.news_feed.server._get_db_path",
                return_value=tmp_db,
            ),
        ):
            async with Client(create_server()) as client:
                result = await client.call_tool("get_news", {"ticker": "AAPL"})
        data = json.loads(get_tool_text(result))
        assert isinstance(data, dict)

    @respx.mock
    async def test_ticker_in_response(self, tmp_db: str) -> None:
        respx.get(AV_BASE_URL).mock(return_value=httpx.Response(200, json=SAMPLE_NEWS_RESPONSE))
        with (
            patch.dict("os.environ", MOCK_ENV),
            patch(
                "stock_radar.mcp_servers.news_feed.server._get_db_path",
                return_value=tmp_db,
            ),
        ):
            async with Client(create_server()) as client:
                result = await client.call_tool("get_news", {"ticker": "AAPL"})
        data = json.loads(get_tool_text(result))
        assert data["ticker"] == "AAPL"
        assert data["source"] == "alpha_vantage"
        assert len(data["articles"]) == 2

    @respx.mock
    async def test_cache_hit_on_second_call(self, tmp_db: str) -> None:
        """Second call with the same ticker should hit cache, not the API."""
        route = respx.get(AV_BASE_URL).mock(
            return_value=httpx.Response(200, json=SAMPLE_NEWS_RESPONSE)
        )
        with (
            patch.dict("os.environ", MOCK_ENV),
            patch(
                "stock_radar.mcp_servers.news_feed.server._get_db_path",
                return_value=tmp_db,
            ),
        ):
            async with Client(create_server()) as client:
                await client.call_tool("get_news", {"ticker": "AAPL"})
                await client.call_tool("get_news", {"ticker": "AAPL"})
        assert route.call_count == 1


class TestSearchNews:
    """Integration tests for the search_news tool."""

    @respx.mock
    async def test_av_result_returned(self, tmp_db: str) -> None:
        respx.get(AV_BASE_URL).mock(return_value=httpx.Response(200, json=SAMPLE_NEWS_RESPONSE))
        with (
            patch.dict("os.environ", MOCK_ENV),
            patch(
                "stock_radar.mcp_servers.news_feed.server._get_db_path",
                return_value=tmp_db,
            ),
        ):
            async with Client(create_server()) as client:
                result = await client.call_tool("search_news", {"query": "artificial intelligence"})
        data = json.loads(get_tool_text(result))
        assert data["source"] == "alpha_vantage"
        assert data["total_fetched"] == 2

    @respx.mock
    async def test_rss_fallback_on_empty_av(self, tmp_db: str) -> None:
        """When AV returns an empty feed, the tool falls back to RSS."""
        respx.get(AV_BASE_URL).mock(
            return_value=httpx.Response(200, json=SAMPLE_EMPTY_NEWS_RESPONSE)
        )
        respx.get(RSS_BASE_URL).mock(return_value=httpx.Response(200, text=SAMPLE_RSS_XML))
        with (
            patch.dict("os.environ", MOCK_ENV),
            patch(
                "stock_radar.mcp_servers.news_feed.server._get_db_path",
                return_value=tmp_db,
            ),
        ):
            async with Client(create_server()) as client:
                result = await client.call_tool("search_news", {"query": "apple"})
        data = json.loads(get_tool_text(result))
        assert data["source"] == "rss"
        assert data["total_fetched"] == 2

    @respx.mock
    async def test_cache_hit_on_second_call(self, tmp_db: str) -> None:
        route = respx.get(AV_BASE_URL).mock(
            return_value=httpx.Response(200, json=SAMPLE_NEWS_RESPONSE)
        )
        with (
            patch.dict("os.environ", MOCK_ENV),
            patch(
                "stock_radar.mcp_servers.news_feed.server._get_db_path",
                return_value=tmp_db,
            ),
        ):
            async with Client(create_server()) as client:
                await client.call_tool("search_news", {"query": "ai"})
                await client.call_tool("search_news", {"query": "ai"})
        assert route.call_count == 1


class TestGetSentimentSummary:
    """Integration tests for the get_sentiment_summary tool."""

    @respx.mock
    async def test_returns_valid_json(self, tmp_db: str) -> None:
        respx.get(AV_BASE_URL).mock(return_value=httpx.Response(200, json=SAMPLE_NEWS_RESPONSE))
        with (
            patch.dict("os.environ", MOCK_ENV),
            patch(
                "stock_radar.mcp_servers.news_feed.server._get_db_path",
                return_value=tmp_db,
            ),
        ):
            async with Client(create_server()) as client:
                result = await client.call_tool("get_sentiment_summary", {"ticker": "AAPL"})
        data = json.loads(get_tool_text(result))
        assert isinstance(data, dict)

    @respx.mock
    async def test_response_fields(self, tmp_db: str) -> None:
        respx.get(AV_BASE_URL).mock(return_value=httpx.Response(200, json=SAMPLE_NEWS_RESPONSE))
        with (
            patch.dict("os.environ", MOCK_ENV),
            patch(
                "stock_radar.mcp_servers.news_feed.server._get_db_path",
                return_value=tmp_db,
            ),
        ):
            async with Client(create_server()) as client:
                result = await client.call_tool("get_sentiment_summary", {"ticker": "AAPL"})
        data = json.loads(get_tool_text(result))
        assert data["ticker"] == "AAPL"
        assert "article_count" in data
        assert "average_sentiment_score" in data
        assert "breakdown" in data
        assert "top_topics" in data

    @respx.mock
    async def test_cache_hit_on_second_call(self, tmp_db: str) -> None:
        route = respx.get(AV_BASE_URL).mock(
            return_value=httpx.Response(200, json=SAMPLE_NEWS_RESPONSE)
        )
        with (
            patch.dict("os.environ", MOCK_ENV),
            patch(
                "stock_radar.mcp_servers.news_feed.server._get_db_path",
                return_value=tmp_db,
            ),
        ):
            async with Client(create_server()) as client:
                await client.call_tool("get_sentiment_summary", {"ticker": "AAPL"})
                await client.call_tool("get_sentiment_summary", {"ticker": "AAPL"})
        assert route.call_count == 1
