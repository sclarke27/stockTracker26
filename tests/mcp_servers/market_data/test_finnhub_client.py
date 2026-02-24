"""Tests for the Finnhub API client."""

from __future__ import annotations

import httpx
import pytest
import respx

from stock_radar.mcp_servers.market_data.clients.finnhub import FinnhubClient
from stock_radar.mcp_servers.market_data.config import FINNHUB_BASE_URL
from stock_radar.mcp_servers.market_data.exceptions import ApiError, TickerNotFoundError


@pytest.fixture()
def fh_client() -> FinnhubClient:
    http_client = httpx.AsyncClient()
    return FinnhubClient(api_key="test-key", http_client=http_client)


SAMPLE_TRANSCRIPT_RESPONSE = {
    "symbol": "AAPL",
    "transcript": [
        {
            "name": "Tim Cook",
            "speech": [
                "Good afternoon, everyone.",
                "Welcome to Apple's Q4 2024 earnings call.",
            ],
        },
        {
            "name": "Luca Maestri",
            "speech": [
                "Thank you, Tim.",
                "Revenue for the quarter was $89.5 billion.",
            ],
        },
    ],
}


class TestGetEarningsTranscript:
    """Tests for FinnhubClient.get_earnings_transcript()."""

    @respx.mock
    async def test_parses_response_correctly(self, fh_client: FinnhubClient) -> None:
        respx.get(f"{FINNHUB_BASE_URL}/stock/transcript").mock(
            return_value=httpx.Response(200, json=SAMPLE_TRANSCRIPT_RESPONSE)
        )
        result = await fh_client.get_earnings_transcript("AAPL", quarter=4, year=2024)
        assert result.ticker == "AAPL"
        assert result.quarter == 4
        assert result.year == 2024
        assert "Tim Cook" in result.content
        assert "Good afternoon" in result.content
        assert "Luca Maestri" in result.content
        assert "$89.5 billion" in result.content

    @respx.mock
    async def test_concatenates_speakers(self, fh_client: FinnhubClient) -> None:
        respx.get(f"{FINNHUB_BASE_URL}/stock/transcript").mock(
            return_value=httpx.Response(200, json=SAMPLE_TRANSCRIPT_RESPONSE)
        )
        result = await fh_client.get_earnings_transcript("AAPL", quarter=4, year=2024)
        # Each speaker's text should be separated.
        assert "Tim Cook:" in result.content
        assert "Luca Maestri:" in result.content

    @respx.mock
    async def test_empty_transcript_raises(self, fh_client: FinnhubClient) -> None:
        respx.get(f"{FINNHUB_BASE_URL}/stock/transcript").mock(
            return_value=httpx.Response(
                200,
                json={"symbol": "XYZ", "transcript": []},
            )
        )
        with pytest.raises(TickerNotFoundError):
            await fh_client.get_earnings_transcript("XYZ", quarter=1, year=2025)

    @respx.mock
    async def test_missing_transcript_key_raises(self, fh_client: FinnhubClient) -> None:
        respx.get(f"{FINNHUB_BASE_URL}/stock/transcript").mock(
            return_value=httpx.Response(200, json={})
        )
        with pytest.raises(TickerNotFoundError):
            await fh_client.get_earnings_transcript("XYZ", quarter=1, year=2025)

    @respx.mock
    async def test_redirect_raises_ticker_not_found(self, fh_client: FinnhubClient) -> None:
        """Finnhub returns 302 → '/' when a transcript doesn't exist."""
        respx.get(f"{FINNHUB_BASE_URL}/stock/transcript").mock(
            return_value=httpx.Response(
                302,
                headers={"Location": "/"},
                text='<a href="/">Found</a>.',
            )
        )
        with pytest.raises(TickerNotFoundError, match="302 redirect"):
            await fh_client.get_earnings_transcript("AAPL", quarter=4, year=2025)

    @respx.mock
    async def test_http_error(self, fh_client: FinnhubClient) -> None:
        respx.get(f"{FINNHUB_BASE_URL}/stock/transcript").mock(
            return_value=httpx.Response(500, text="Server Error")
        )
        with pytest.raises(ApiError):
            await fh_client.get_earnings_transcript("AAPL", quarter=4, year=2024)
