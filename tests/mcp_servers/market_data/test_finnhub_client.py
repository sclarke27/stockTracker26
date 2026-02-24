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


SAMPLE_TRANSCRIPT_LIST = {
    "symbol": "AAPL",
    "transcripts": [
        {"id": "AAPL_2024_4", "quarter": 4, "year": 2024, "title": "Q4 2024"},
        {"id": "AAPL_2024_3", "quarter": 3, "year": 2024, "title": "Q3 2024"},
    ],
}

SAMPLE_TRANSCRIPT = {
    "symbol": "AAPL",
    "id": "AAPL_2024_4",
    "time": "2024-10-31",
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


def _mock_two_step(
    list_response: dict = SAMPLE_TRANSCRIPT_LIST,
    transcript_response: dict = SAMPLE_TRANSCRIPT,
) -> None:
    """Set up respx mocks for the two-step transcript flow."""
    respx.get(f"{FINNHUB_BASE_URL}/stock/transcripts/list").mock(
        return_value=httpx.Response(200, json=list_response)
    )
    respx.get(f"{FINNHUB_BASE_URL}/stock/transcripts").mock(
        return_value=httpx.Response(200, json=transcript_response)
    )


class TestGetEarningsTranscript:
    """Tests for FinnhubClient.get_earnings_transcript()."""

    @respx.mock
    async def test_parses_response_correctly(self, fh_client: FinnhubClient) -> None:
        _mock_two_step()
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
        _mock_two_step()
        result = await fh_client.get_earnings_transcript("AAPL", quarter=4, year=2024)
        assert "Tim Cook:" in result.content
        assert "Luca Maestri:" in result.content

    @respx.mock
    async def test_quarter_not_in_list_raises(self, fh_client: FinnhubClient) -> None:
        """Requesting a quarter not in the listing raises TickerNotFoundError."""
        respx.get(f"{FINNHUB_BASE_URL}/stock/transcripts/list").mock(
            return_value=httpx.Response(200, json=SAMPLE_TRANSCRIPT_LIST)
        )
        with pytest.raises(TickerNotFoundError, match="Q1 2025"):
            await fh_client.get_earnings_transcript("AAPL", quarter=1, year=2025)

    @respx.mock
    async def test_empty_transcript_list_raises(self, fh_client: FinnhubClient) -> None:
        respx.get(f"{FINNHUB_BASE_URL}/stock/transcripts/list").mock(
            return_value=httpx.Response(200, json={"symbol": "XYZ", "transcripts": []})
        )
        with pytest.raises(TickerNotFoundError):
            await fh_client.get_earnings_transcript("XYZ", quarter=1, year=2025)

    @respx.mock
    async def test_empty_transcript_content_raises(self, fh_client: FinnhubClient) -> None:
        """Transcript ID found but content is empty."""
        empty_transcript = {**SAMPLE_TRANSCRIPT, "transcript": []}
        _mock_two_step(transcript_response=empty_transcript)
        with pytest.raises(TickerNotFoundError, match="Empty transcript"):
            await fh_client.get_earnings_transcript("AAPL", quarter=4, year=2024)

    @respx.mock
    async def test_403_raises_api_error(self, fh_client: FinnhubClient) -> None:
        """Premium-only endpoints return 403."""
        respx.get(f"{FINNHUB_BASE_URL}/stock/transcripts/list").mock(
            return_value=httpx.Response(
                403, json={"error": "You don't have access to this resource."}
            )
        )
        with pytest.raises(ApiError, match="premium plan"):
            await fh_client.get_earnings_transcript("AAPL", quarter=4, year=2024)

    @respx.mock
    async def test_redirect_raises_ticker_not_found(self, fh_client: FinnhubClient) -> None:
        """Deprecated endpoints redirect to '/'."""
        respx.get(f"{FINNHUB_BASE_URL}/stock/transcripts/list").mock(
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
        respx.get(f"{FINNHUB_BASE_URL}/stock/transcripts/list").mock(
            return_value=httpx.Response(500, text="Server Error")
        )
        with pytest.raises(ApiError):
            await fh_client.get_earnings_transcript("AAPL", quarter=4, year=2024)
