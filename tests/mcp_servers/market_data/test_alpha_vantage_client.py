"""Tests for the Alpha Vantage API client."""

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


# --- Sample API responses ---

SAMPLE_DAILY_RESPONSE = {
    "Meta Data": {
        "1. Information": "Daily Prices (open, high, low, close) and Volumes",
        "2. Symbol": "AAPL",
        "3. Last Refreshed": "2025-01-15",
        "4. Output Size": "Compact",
        "5. Time Zone": "US/Eastern",
    },
    "Time Series (Daily)": {
        "2025-01-15": {
            "1. open": "150.00",
            "2. high": "155.00",
            "3. low": "149.00",
            "4. close": "153.50",
            "5. volume": "1000000",
        },
        "2025-01-14": {
            "1. open": "148.00",
            "2. high": "151.00",
            "3. low": "147.00",
            "4. close": "150.00",
            "5. volume": "900000",
        },
    },
}

SAMPLE_QUOTE_RESPONSE = {
    "Global Quote": {
        "01. symbol": "MSFT",
        "02. open": "416.00",
        "03. high": "421.00",
        "04. low": "414.50",
        "05. price": "420.50",
        "06. volume": "25000000",
        "07. latest trading day": "2025-01-15",
        "08. previous close": "415.25",
        "09. change": "5.25",
        "10. change percent": "1.2640%",
    }
}

SAMPLE_OVERVIEW_RESPONSE = {
    "Symbol": "AAPL",
    "Name": "Apple Inc",
    "Description": "Apple designs consumer electronics.",
    "Sector": "TECHNOLOGY",
    "Industry": "ELECTRONIC COMPUTERS",
    "MarketCapitalization": "3000000000000",
    "PERatio": "28.5",
    "EPS": "6.42",
    "DividendYield": "0.0055",
    "52WeekHigh": "199.62",
    "52WeekLow": "164.08",
}

SAMPLE_SEARCH_RESPONSE = {
    "bestMatches": [
        {
            "1. symbol": "AAPL",
            "2. name": "Apple Inc",
            "3. type": "Equity",
            "4. region": "United States",
            "5. marketOpen": "09:30",
            "6. marketClose": "16:00",
            "7. timezone": "UTC-04",
            "8. currency": "USD",
            "9. matchScore": "1.0000",
        },
        {
            "1. symbol": "APLE",
            "2. name": "Apple Hospitality REIT Inc",
            "3. type": "Equity",
            "4. region": "United States",
            "5. marketOpen": "09:30",
            "6. marketClose": "16:00",
            "7. timezone": "UTC-04",
            "8. currency": "USD",
            "9. matchScore": "0.5714",
        },
    ]
}


class TestGetDailyPrices:
    """Tests for AlphaVantageClient.get_daily_prices()."""

    @respx.mock
    async def test_parses_response_correctly(self, av_client: AlphaVantageClient) -> None:
        respx.get(AV_BASE_URL).mock(return_value=httpx.Response(200, json=SAMPLE_DAILY_RESPONSE))
        result = await av_client.get_daily_prices("AAPL")
        assert result.ticker == "AAPL"
        assert result.last_refreshed == "2025-01-15"
        assert len(result.bars) == 2
        assert result.bars[0].date == "2025-01-15"
        assert result.bars[0].open == 150.0
        assert result.bars[0].close == 153.5
        assert result.bars[0].volume == 1_000_000

    @respx.mock
    async def test_empty_time_series(self, av_client: AlphaVantageClient) -> None:
        response = {
            "Meta Data": {
                "2. Symbol": "XYZ",
                "3. Last Refreshed": "2025-01-15",
            },
            "Time Series (Daily)": {},
        }
        respx.get(AV_BASE_URL).mock(return_value=httpx.Response(200, json=response))
        result = await av_client.get_daily_prices("XYZ")
        assert result.bars == []


class TestGetQuote:
    """Tests for AlphaVantageClient.get_quote()."""

    @respx.mock
    async def test_parses_response_correctly(self, av_client: AlphaVantageClient) -> None:
        respx.get(AV_BASE_URL).mock(return_value=httpx.Response(200, json=SAMPLE_QUOTE_RESPONSE))
        result = await av_client.get_quote("MSFT")
        assert result.ticker == "MSFT"
        assert result.price == 420.50
        assert result.change == 5.25
        assert result.change_percent == "1.2640%"
        assert result.volume == 25_000_000
        assert result.previous_close == 415.25
        assert result.open == 416.0
        assert result.high == 421.0
        assert result.low == 414.5

    @respx.mock
    async def test_empty_quote_raises(self, av_client: AlphaVantageClient) -> None:
        respx.get(AV_BASE_URL).mock(return_value=httpx.Response(200, json={"Global Quote": {}}))
        with pytest.raises(TickerNotFoundError):
            await av_client.get_quote("INVALID")


class TestGetCompanyOverview:
    """Tests for AlphaVantageClient.get_company_overview()."""

    @respx.mock
    async def test_parses_response_correctly(self, av_client: AlphaVantageClient) -> None:
        respx.get(AV_BASE_URL).mock(return_value=httpx.Response(200, json=SAMPLE_OVERVIEW_RESPONSE))
        result = await av_client.get_company_overview("AAPL")
        assert result.ticker == "AAPL"
        assert result.name == "Apple Inc"
        assert result.sector == "TECHNOLOGY"
        assert result.industry == "ELECTRONIC COMPUTERS"
        assert result.market_cap == "3000000000000"
        assert result.pe_ratio == "28.5"
        assert result.eps == "6.42"

    @respx.mock
    async def test_unknown_ticker_raises(self, av_client: AlphaVantageClient) -> None:
        # AV returns an empty dict (or just {}) for unknown tickers.
        respx.get(AV_BASE_URL).mock(return_value=httpx.Response(200, json={}))
        with pytest.raises(TickerNotFoundError):
            await av_client.get_company_overview("ZZZZZ")


class TestSearchTickers:
    """Tests for AlphaVantageClient.search_tickers()."""

    @respx.mock
    async def test_parses_response_correctly(self, av_client: AlphaVantageClient) -> None:
        respx.get(AV_BASE_URL).mock(return_value=httpx.Response(200, json=SAMPLE_SEARCH_RESPONSE))
        result = await av_client.search_tickers("apple")
        assert len(result.matches) == 2
        assert result.matches[0].symbol == "AAPL"
        assert result.matches[0].name == "Apple Inc"
        assert result.matches[0].currency == "USD"
        assert result.matches[1].symbol == "APLE"

    @respx.mock
    async def test_no_matches(self, av_client: AlphaVantageClient) -> None:
        respx.get(AV_BASE_URL).mock(return_value=httpx.Response(200, json={"bestMatches": []}))
        result = await av_client.search_tickers("zzzzznotarealcompany")
        assert result.matches == []


class TestErrorHandling:
    """Tests for Alpha Vantage error scenarios."""

    @respx.mock
    async def test_api_error_message(self, av_client: AlphaVantageClient) -> None:
        respx.get(AV_BASE_URL).mock(
            return_value=httpx.Response(
                200,
                json={"Error Message": "Invalid API call."},
            )
        )
        with pytest.raises(ApiError, match="Invalid API call"):
            await av_client.get_quote("AAPL")

    @respx.mock
    async def test_rate_limit_note(self, av_client: AlphaVantageClient) -> None:
        respx.get(AV_BASE_URL).mock(
            return_value=httpx.Response(
                200,
                json={
                    "Note": "Thank you for using Alpha Vantage! "
                    "Our standard API rate limit is 5 requests per minute."
                },
            )
        )
        with pytest.raises(ApiError, match="rate limit"):
            await av_client.get_quote("AAPL")

    @respx.mock
    async def test_rate_limit_information(self, av_client: AlphaVantageClient) -> None:
        respx.get(AV_BASE_URL).mock(
            return_value=httpx.Response(
                200,
                json={
                    "Information": "We have detected your API key and our standard "
                    "API rate limit is 25 requests per day."
                },
            )
        )
        with pytest.raises(ApiError, match="rate limit"):
            await av_client.get_quote("AAPL")

    @respx.mock
    async def test_http_server_error(self, av_client: AlphaVantageClient) -> None:
        respx.get(AV_BASE_URL).mock(return_value=httpx.Response(500, text="Internal Server Error"))
        with pytest.raises(ApiError):
            await av_client.get_quote("AAPL")
