"""Integration tests for the market-data MCP server."""

from __future__ import annotations

import json
from unittest.mock import patch

import httpx
import respx
from fastmcp import Client

from stock_radar.mcp_servers.market_data.server import create_server
from tests.mcp_servers.conftest import get_tool_text
from tests.mcp_servers.market_data.test_alpha_vantage_client import (
    SAMPLE_DAILY_RESPONSE,
    SAMPLE_OVERVIEW_RESPONSE,
    SAMPLE_QUOTE_RESPONSE,
    SAMPLE_SEARCH_RESPONSE,
)
from tests.mcp_servers.market_data.test_finnhub_client import (
    SAMPLE_TRANSCRIPT_RESPONSE,
)

AV_URL = "https://www.alphavantage.co/query"
FH_URL = "https://finnhub.io/api/v1/stock/transcript"

MOCK_ENV = {
    "ALPHA_VANTAGE_API_KEY": "test-av-key",
    "FINNHUB_API_KEY": "test-fh-key",
    "ANTHROPIC_API_KEY": "test-anthropic-key",
    "OPENAI_API_KEY": "test-openai-key",
    "SEC_EDGAR_EMAIL": "test@example.com",
}


class TestServerTools:
    """Integration tests using the FastMCP test client."""

    @respx.mock
    async def test_get_price_history(self, tmp_db: str) -> None:
        respx.get(AV_URL).mock(return_value=httpx.Response(200, json=SAMPLE_DAILY_RESPONSE))
        with (
            patch.dict("os.environ", MOCK_ENV),
            patch(
                "stock_radar.mcp_servers.market_data.server._get_db_path",
                return_value=tmp_db,
            ),
        ):
            async with Client(create_server()) as client:
                result = await client.call_tool(
                    "get_price_history",
                    {"ticker": "AAPL"},
                )
        data = json.loads(get_tool_text(result))
        assert data["ticker"] == "AAPL"
        assert len(data["bars"]) == 2
        assert data["bars"][0]["close"] == 153.5

    @respx.mock
    async def test_get_quote(self, tmp_db: str) -> None:
        respx.get(AV_URL).mock(return_value=httpx.Response(200, json=SAMPLE_QUOTE_RESPONSE))
        with (
            patch.dict("os.environ", MOCK_ENV),
            patch(
                "stock_radar.mcp_servers.market_data.server._get_db_path",
                return_value=tmp_db,
            ),
        ):
            async with Client(create_server()) as client:
                result = await client.call_tool(
                    "get_quote",
                    {"ticker": "MSFT"},
                )
        data = json.loads(get_tool_text(result))
        assert data["ticker"] == "MSFT"
        assert data["price"] == 420.50

    @respx.mock
    async def test_get_company_info(self, tmp_db: str) -> None:
        respx.get(AV_URL).mock(return_value=httpx.Response(200, json=SAMPLE_OVERVIEW_RESPONSE))
        with (
            patch.dict("os.environ", MOCK_ENV),
            patch(
                "stock_radar.mcp_servers.market_data.server._get_db_path",
                return_value=tmp_db,
            ),
        ):
            async with Client(create_server()) as client:
                result = await client.call_tool(
                    "get_company_info",
                    {"ticker": "AAPL"},
                )
        data = json.loads(get_tool_text(result))
        assert data["ticker"] == "AAPL"
        assert data["name"] == "Apple Inc"

    @respx.mock
    async def test_search_tickers(self, tmp_db: str) -> None:
        respx.get(AV_URL).mock(return_value=httpx.Response(200, json=SAMPLE_SEARCH_RESPONSE))
        with (
            patch.dict("os.environ", MOCK_ENV),
            patch(
                "stock_radar.mcp_servers.market_data.server._get_db_path",
                return_value=tmp_db,
            ),
        ):
            async with Client(create_server()) as client:
                result = await client.call_tool(
                    "search_tickers",
                    {"keywords": "apple"},
                )
        data = json.loads(get_tool_text(result))
        assert len(data["matches"]) == 2
        assert data["matches"][0]["symbol"] == "AAPL"

    @respx.mock
    async def test_get_earnings_transcript(self, tmp_db: str) -> None:
        respx.get(FH_URL).mock(return_value=httpx.Response(200, json=SAMPLE_TRANSCRIPT_RESPONSE))
        with (
            patch.dict("os.environ", MOCK_ENV),
            patch(
                "stock_radar.mcp_servers.market_data.server._get_db_path",
                return_value=tmp_db,
            ),
        ):
            async with Client(create_server()) as client:
                result = await client.call_tool(
                    "get_earnings_transcript",
                    {"ticker": "AAPL", "quarter": 4, "year": 2024},
                )
        data = json.loads(get_tool_text(result))
        assert data["ticker"] == "AAPL"
        assert "Tim Cook" in data["content"]

    @respx.mock
    async def test_cache_hit_on_second_call(self, tmp_db: str) -> None:
        """Second call to the same tool should hit the cache, not the API."""
        route = respx.get(AV_URL).mock(return_value=httpx.Response(200, json=SAMPLE_QUOTE_RESPONSE))
        with (
            patch.dict("os.environ", MOCK_ENV),
            patch(
                "stock_radar.mcp_servers.market_data.server._get_db_path",
                return_value=tmp_db,
            ),
        ):
            async with Client(create_server()) as client:
                await client.call_tool("get_quote", {"ticker": "MSFT"})
                await client.call_tool("get_quote", {"ticker": "MSFT"})
        # The API should have been called only once.
        assert route.call_count == 1

    @respx.mock
    async def test_tool_returns_valid_json(self, tmp_db: str) -> None:
        respx.get(AV_URL).mock(return_value=httpx.Response(200, json=SAMPLE_QUOTE_RESPONSE))
        with (
            patch.dict("os.environ", MOCK_ENV),
            patch(
                "stock_radar.mcp_servers.market_data.server._get_db_path",
                return_value=tmp_db,
            ),
        ):
            async with Client(create_server()) as client:
                result = await client.call_tool(
                    "get_quote",
                    {"ticker": "MSFT"},
                )
        parsed = json.loads(get_tool_text(result))
        assert isinstance(parsed, dict)
