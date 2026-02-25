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
    "quarter": "2024Q4",
    "transcript": [
        {
            "speaker": "Operator",
            "title": "Conference Call Operator",
            "content": ("Good afternoon, everyone. Welcome to Apple's Q4 2024 " "earnings call."),
            "sentiment": "neutral",
        },
        {
            "speaker": "Tim Cook",
            "title": "Chief Executive Officer",
            "content": (
                "Thank you, operator. We had an outstanding quarter with "
                "revenue of $89.5 billion."
            ),
            "sentiment": "positive",
        },
    ],
}


class TestGetEarningsTranscript:
    """Tests for AlphaVantageClient.get_earnings_transcript()."""

    @respx.mock
    async def test_parses_transcript_segments(self, av_client: AlphaVantageClient) -> None:
        respx.get(AV_BASE_URL).mock(
            return_value=httpx.Response(200, json=SAMPLE_TRANSCRIPT_RESPONSE)
        )
        result = await av_client.get_earnings_transcript("AAPL", quarter=4, year=2024)
        assert result.ticker == "AAPL"
        assert result.quarter == 4
        assert result.year == 2024
        assert "Tim Cook" in result.content
        assert "$89.5 billion" in result.content
        assert len(result.segments) == 2
        assert result.segments[0].speaker == "Operator"
        assert result.segments[1].sentiment == "positive"

    @respx.mock
    async def test_empty_transcript_array_raises(self, av_client: AlphaVantageClient) -> None:
        """Empty transcript array should raise TickerNotFoundError."""
        response = {"symbol": "AAPL", "quarter": "2024Q4", "transcript": []}
        respx.get(AV_BASE_URL).mock(return_value=httpx.Response(200, json=response))
        with pytest.raises(TickerNotFoundError, match="Empty transcript"):
            await av_client.get_earnings_transcript("AAPL", quarter=4, year=2024)

    @respx.mock
    async def test_missing_transcript_key_raises(self, av_client: AlphaVantageClient) -> None:
        """Response without 'transcript' key should raise TickerNotFoundError."""
        response = {"symbol": "AAPL", "quarter": "2024Q4"}
        respx.get(AV_BASE_URL).mock(return_value=httpx.Response(200, json=response))
        with pytest.raises(TickerNotFoundError):
            await av_client.get_earnings_transcript("AAPL", quarter=4, year=2024)

    @respx.mock
    async def test_segments_with_empty_content_filtered(
        self, av_client: AlphaVantageClient
    ) -> None:
        """Segments with empty content strings should be filtered out."""
        response = {
            "symbol": "AAPL",
            "quarter": "2024Q4",
            "transcript": [
                {"speaker": "A", "title": "T", "content": "", "sentiment": ""},
                {"speaker": "B", "title": "T", "content": "Actual content.", "sentiment": ""},
            ],
        }
        respx.get(AV_BASE_URL).mock(return_value=httpx.Response(200, json=response))
        result = await av_client.get_earnings_transcript("AAPL", quarter=4, year=2024)
        assert len(result.segments) == 1
        assert result.segments[0].speaker == "B"

    @respx.mock
    async def test_flat_content_fallback(self, av_client: AlphaVantageClient) -> None:
        """Legacy flat 'content' string should still be parsed."""
        response = {
            "symbol": "AAPL",
            "quarter": 4,
            "year": 2024,
            "date": "2024-10-31",
            "content": "Full transcript as flat string.",
        }
        respx.get(AV_BASE_URL).mock(return_value=httpx.Response(200, json=response))
        result = await av_client.get_earnings_transcript("AAPL", quarter=4, year=2024)
        assert "Full transcript as flat string." in result.content
        assert len(result.segments) == 1

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
