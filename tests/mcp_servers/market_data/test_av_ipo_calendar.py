"""Tests for Alpha Vantage IPO calendar retrieval."""

from __future__ import annotations

import httpx
import pytest
import respx

from stock_radar.mcp_servers.market_data.clients.alpha_vantage import AlphaVantageClient
from stock_radar.mcp_servers.market_data.config import AV_BASE_URL
from stock_radar.mcp_servers.market_data.exceptions import ApiError
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


SAMPLE_IPO_CSV = (
    "symbol,name,ipoDate,priceRangeLow,priceRangeHigh,currency,exchange\r\n"
    "ACME,Acme Corp,2026-03-15,18.00,20.00,USD,NASDAQ\r\n"
    "WIDG,Widget Inc,2026-03-20,10.50,12.50,USD,NYSE\r\n"
)

SAMPLE_IPO_CSV_EMPTY = "symbol,name,ipoDate,priceRangeLow,priceRangeHigh,currency,exchange\r\n"

SAMPLE_IPO_CSV_ZEROS = (
    "symbol,name,ipoDate,priceRangeLow,priceRangeHigh,currency,exchange\r\n"
    "KTWOR,K2 Capital Acquisition Corporation Rights,2026-02-25,0,0,USD,NASDAQ\r\n"
)


class TestGetIpoCalendar:
    """Tests for AlphaVantageClient.get_ipo_calendar()."""

    @respx.mock
    async def test_parses_csv_correctly(self, av_client: AlphaVantageClient) -> None:
        respx.get(AV_BASE_URL).mock(return_value=httpx.Response(200, text=SAMPLE_IPO_CSV))
        result = await av_client.get_ipo_calendar()
        assert len(result.entries) == 2
        assert result.entries[0].symbol == "ACME"
        assert result.entries[0].name == "Acme Corp"
        assert result.entries[0].ipo_date == "2026-03-15"
        assert result.entries[0].price_range_low == 18.0
        assert result.entries[0].price_range_high == 20.0
        assert result.entries[0].currency == "USD"
        assert result.entries[0].exchange == "NASDAQ"
        assert result.entries[1].symbol == "WIDG"
        assert result.entries[1].exchange == "NYSE"

    @respx.mock
    async def test_empty_csv_returns_empty_list(self, av_client: AlphaVantageClient) -> None:
        respx.get(AV_BASE_URL).mock(return_value=httpx.Response(200, text=SAMPLE_IPO_CSV_EMPTY))
        result = await av_client.get_ipo_calendar()
        assert result.entries == []

    @respx.mock
    async def test_zero_prices_parsed(self, av_client: AlphaVantageClient) -> None:
        """IPOs with price 0 (unpriced SPACs, etc.) should parse correctly."""
        respx.get(AV_BASE_URL).mock(return_value=httpx.Response(200, text=SAMPLE_IPO_CSV_ZEROS))
        result = await av_client.get_ipo_calendar()
        assert len(result.entries) == 1
        assert result.entries[0].price_range_low == 0.0
        assert result.entries[0].price_range_high == 0.0

    @respx.mock
    async def test_http_error_propagates(self, av_client: AlphaVantageClient) -> None:
        respx.get(AV_BASE_URL).mock(return_value=httpx.Response(500, text="Internal Server Error"))
        with pytest.raises(ApiError):
            await av_client.get_ipo_calendar()
