"""Integration tests for the sec-edgar MCP server."""

from __future__ import annotations

import json
from unittest.mock import patch

import httpx
import respx
from fastmcp import Client

from stock_radar.mcp_servers.sec_edgar.config import SEC_EFTS_URL, SEC_TICKERS_URL
from stock_radar.mcp_servers.sec_edgar.server import create_server
from tests.mcp_servers.conftest import get_tool_text
from tests.mcp_servers.sec_edgar.test_edgar_client import (
    SAMPLE_COMPANY_TICKERS,
    SAMPLE_EFTS_RESPONSE,
    SAMPLE_FILING_HTML,
    SAMPLE_FORM4_XML,
    SAMPLE_SUBMISSIONS,
)

MOCK_ENV = {
    "SEC_EDGAR_EMAIL": "test@example.com",
    "ALPHA_VANTAGE_API_KEY": "unused",
    "FINNHUB_API_KEY": "unused",
    "ANTHROPIC_API_KEY": "unused",
}


def _mock_sec_routes() -> None:
    """Set up respx routes for SEC EDGAR endpoints."""
    respx.get(SEC_TICKERS_URL).mock(return_value=httpx.Response(200, json=SAMPLE_COMPANY_TICKERS))
    respx.get(url__startswith="https://data.sec.gov/submissions/").mock(
        return_value=httpx.Response(200, json=SAMPLE_SUBMISSIONS)
    )


class TestServerTools:
    """Integration tests using the FastMCP test client."""

    @respx.mock
    async def test_get_filings(self, tmp_db: str) -> None:
        _mock_sec_routes()
        with (
            patch.dict("os.environ", MOCK_ENV),
            patch(
                "stock_radar.mcp_servers.sec_edgar.server._get_db_path",
                return_value=tmp_db,
            ),
        ):
            async with Client(create_server()) as client:
                result = await client.call_tool(
                    "get_filings",
                    {"ticker": "AAPL"},
                )
        data = json.loads(get_tool_text(result))
        assert data["ticker"] == "AAPL"
        assert data["company_name"] == "Apple Inc."
        assert len(data["filings"]) == 4

    @respx.mock
    async def test_get_filings_with_form_filter(self, tmp_db: str) -> None:
        _mock_sec_routes()
        with (
            patch.dict("os.environ", MOCK_ENV),
            patch(
                "stock_radar.mcp_servers.sec_edgar.server._get_db_path",
                return_value=tmp_db,
            ),
        ):
            async with Client(create_server()) as client:
                result = await client.call_tool(
                    "get_filings",
                    {"ticker": "AAPL", "form_types": ["10-K"]},
                )
        data = json.loads(get_tool_text(result))
        assert len(data["filings"]) == 1
        assert data["filings"][0]["form_type"] == "10-K"

    @respx.mock
    async def test_get_filing_text(self, tmp_db: str) -> None:
        _mock_sec_routes()
        respx.get(url__startswith="https://www.sec.gov/Archives/").mock(
            return_value=httpx.Response(200, text=SAMPLE_FILING_HTML)
        )
        with (
            patch.dict("os.environ", MOCK_ENV),
            patch(
                "stock_radar.mcp_servers.sec_edgar.server._get_db_path",
                return_value=tmp_db,
            ),
        ):
            async with Client(create_server()) as client:
                result = await client.call_tool(
                    "get_filing_text",
                    {
                        "ticker": "AAPL",
                        "accession_number": "0000320193-25-000001",
                    },
                )
        data = json.loads(get_tool_text(result))
        assert data["ticker"] == "AAPL"
        assert data["form_type"] == "8-K"
        assert "definitive agreement" in data["content"]
        assert data["truncated"] is False

    @respx.mock
    async def test_get_insider_transactions(self, tmp_db: str) -> None:
        _mock_sec_routes()
        respx.get(url__startswith="https://www.sec.gov/Archives/").mock(
            return_value=httpx.Response(200, text=SAMPLE_FORM4_XML)
        )
        with (
            patch.dict("os.environ", MOCK_ENV),
            patch(
                "stock_radar.mcp_servers.sec_edgar.server._get_db_path",
                return_value=tmp_db,
            ),
        ):
            async with Client(create_server()) as client:
                result = await client.call_tool(
                    "get_insider_transactions",
                    {"ticker": "AAPL"},
                )
        data = json.loads(get_tool_text(result))
        assert data["ticker"] == "AAPL"
        assert data["company_name"] == "Apple Inc."
        assert len(data["transactions"]) >= 1
        txn = data["transactions"][0]
        assert txn["owner_name"] == "Cook Timothy D"
        assert txn["transaction_code"] == "S"

    @respx.mock
    async def test_search_filings(self, tmp_db: str) -> None:
        respx.post(SEC_EFTS_URL).mock(return_value=httpx.Response(200, json=SAMPLE_EFTS_RESPONSE))
        with (
            patch.dict("os.environ", MOCK_ENV),
            patch(
                "stock_radar.mcp_servers.sec_edgar.server._get_db_path",
                return_value=tmp_db,
            ),
        ):
            async with Client(create_server()) as client:
                result = await client.call_tool(
                    "search_filings",
                    {"query": "artificial intelligence"},
                )
        data = json.loads(get_tool_text(result))
        assert data["query"] == "artificial intelligence"
        assert data["total_hits"] == 2
        assert len(data["hits"]) == 2

    @respx.mock
    async def test_cache_hit_on_second_call(self, tmp_db: str) -> None:
        """Second call to the same tool should hit the cache, not the API."""
        _mock_sec_routes()
        route = respx.get(url__startswith="https://data.sec.gov/submissions/").mock(
            return_value=httpx.Response(200, json=SAMPLE_SUBMISSIONS)
        )
        with (
            patch.dict("os.environ", MOCK_ENV),
            patch(
                "stock_radar.mcp_servers.sec_edgar.server._get_db_path",
                return_value=tmp_db,
            ),
        ):
            async with Client(create_server()) as client:
                await client.call_tool("get_filings", {"ticker": "AAPL"})
                await client.call_tool("get_filings", {"ticker": "AAPL"})
        # Submissions endpoint should have been called only once (tickers is separate).
        assert route.call_count == 1

    @respx.mock
    async def test_tool_returns_valid_json(self, tmp_db: str) -> None:
        _mock_sec_routes()
        with (
            patch.dict("os.environ", MOCK_ENV),
            patch(
                "stock_radar.mcp_servers.sec_edgar.server._get_db_path",
                return_value=tmp_db,
            ),
        ):
            async with Client(create_server()) as client:
                result = await client.call_tool(
                    "get_filings",
                    {"ticker": "AAPL"},
                )
        parsed = json.loads(get_tool_text(result))
        assert isinstance(parsed, dict)
