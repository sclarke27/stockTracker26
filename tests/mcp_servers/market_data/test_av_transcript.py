"""Tests for Alpha Vantage earnings transcript retrieval."""

from __future__ import annotations

import httpx
import pytest
import respx

from stock_radar.mcp_servers.market_data.clients.alpha_vantage import AlphaVantageClient
from stock_radar.mcp_servers.market_data.config import AV_BASE_URL
from stock_radar.mcp_servers.market_data.exceptions import ApiError, TickerNotFoundError
from stock_radar.utils.rate_limiter import RateLimiter


@pytest.fixture()
def rate_limiter() -> RateLimiter:
    return RateLimiter(requests_per_minute=100, requests_per_day=10_000)


@pytest.fixture()
def av_client(rate_limiter: RateLimiter) -> AlphaVantageClient:
    http_client = httpx.AsyncClient()
    return AlphaVantageClient(
        api_key="test-key",
        http_client=http_client,
        rate_limiter=rate_limiter,
    )


SAMPLE_TRANSCRIPT_RESPONSE = {
    "symbol": "AAPL",
    "quarter": 4,
    "year": 2024,
    "date": "2024-10-31",
    "content": (
        "Operator: Good afternoon, everyone. Welcome to Apple's Q4 2024 "
        "earnings call.\n\n"
        "Tim Cook: Thank you, operator. We had an outstanding quarter with "
        "revenue of $89.5 billion."
    ),
}


class TestGetEarningsTranscript:
    """Tests for AlphaVantageClient.get_earnings_transcript()."""

    @respx.mock
    async def test_parses_response_correctly(self, av_client: AlphaVantageClient) -> None:
        respx.get(AV_BASE_URL).mock(
            return_value=httpx.Response(200, json=SAMPLE_TRANSCRIPT_RESPONSE)
        )
        result = await av_client.get_earnings_transcript("AAPL", quarter=4, year=2024)
        assert result.ticker == "AAPL"
        assert result.quarter == 4
        assert result.year == 2024
        assert result.date == "2024-10-31"
        assert "Tim Cook" in result.content
        assert "$89.5 billion" in result.content

    @respx.mock
    async def test_empty_content_raises(self, av_client: AlphaVantageClient) -> None:
        """Empty transcript content should raise TickerNotFoundError."""
        response = {**SAMPLE_TRANSCRIPT_RESPONSE, "content": ""}
        respx.get(AV_BASE_URL).mock(return_value=httpx.Response(200, json=response))
        with pytest.raises(TickerNotFoundError, match="Empty transcript"):
            await av_client.get_earnings_transcript("AAPL", quarter=4, year=2024)

    @respx.mock
    async def test_missing_content_key_raises(self, av_client: AlphaVantageClient) -> None:
        """Response without 'content' key should raise TickerNotFoundError."""
        response = {"symbol": "AAPL", "quarter": 4, "year": 2024}
        respx.get(AV_BASE_URL).mock(return_value=httpx.Response(200, json=response))
        with pytest.raises(TickerNotFoundError):
            await av_client.get_earnings_transcript("AAPL", quarter=4, year=2024)

    @respx.mock
    async def test_api_error_propagates(self, av_client: AlphaVantageClient) -> None:
        respx.get(AV_BASE_URL).mock(
            return_value=httpx.Response(200, json={"Error Message": "Invalid API call."})
        )
        with pytest.raises(ApiError, match="Invalid API call"):
            await av_client.get_earnings_transcript("AAPL", quarter=4, year=2024)

    @respx.mock
    async def test_http_error_propagates(self, av_client: AlphaVantageClient) -> None:
        respx.get(AV_BASE_URL).mock(return_value=httpx.Response(500, text="Internal Server Error"))
        with pytest.raises(ApiError):
            await av_client.get_earnings_transcript("AAPL", quarter=4, year=2024)
