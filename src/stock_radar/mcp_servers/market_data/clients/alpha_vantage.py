"""Alpha Vantage API client for market data retrieval."""

from __future__ import annotations

import httpx
from loguru import logger

from stock_radar.mcp_servers.market_data.config import AV_BASE_URL, SERVER_NAME
from stock_radar.mcp_servers.market_data.exceptions import ApiError, TickerNotFoundError
from stock_radar.models.market_data import (
    CompanyInfoResponse,
    OHLCVBar,
    PriceHistoryResponse,
    QuoteResponse,
    TickerMatch,
    TickerSearchResponse,
)
from stock_radar.utils.rate_limiter import RateLimiter


class AlphaVantageClient:
    """Async client for the Alpha Vantage REST API.

    Wraps all Alpha Vantage endpoints used by the market-data MCP server.
    Requests go through a rate limiter before hitting the API.

    Args:
        api_key: Alpha Vantage API key.
        http_client: Shared ``httpx.AsyncClient`` instance.
        rate_limiter: Rate limiter controlling request frequency.
    """

    def __init__(
        self,
        api_key: str,
        http_client: httpx.AsyncClient,
        rate_limiter: RateLimiter,
    ) -> None:
        self._api_key = api_key
        self._http = http_client
        self._limiter = rate_limiter

    async def get_daily_prices(
        self,
        ticker: str,
        outputsize: str = "compact",
    ) -> PriceHistoryResponse:
        """Fetch daily OHLCV price history for a ticker.

        Args:
            ticker: Stock ticker symbol (e.g. ``"AAPL"``).
            outputsize: ``"compact"`` for last 100 days, ``"full"`` for
                20+ years of history.

        Returns:
            Parsed price history with OHLCV bars.
        """
        data = await self._request(
            function="TIME_SERIES_DAILY",
            params={"symbol": ticker, "outputsize": outputsize},
        )

        meta = data.get("Meta Data", {})
        time_series = data.get("Time Series (Daily)", {})

        bars = [
            OHLCVBar(
                date=date,
                open=float(values["1. open"]),
                high=float(values["2. high"]),
                low=float(values["3. low"]),
                close=float(values["4. close"]),
                volume=int(values["5. volume"]),
            )
            for date, values in sorted(time_series.items(), reverse=True)
        ]

        return PriceHistoryResponse(
            ticker=meta.get("2. Symbol", ticker),
            bars=bars,
            last_refreshed=meta.get("3. Last Refreshed", ""),
        )

    async def get_quote(self, ticker: str) -> QuoteResponse:
        """Fetch the current quote for a ticker.

        Args:
            ticker: Stock ticker symbol.

        Returns:
            Parsed quote data.

        Raises:
            TickerNotFoundError: If the quote is empty (invalid ticker).
        """
        data = await self._request(
            function="GLOBAL_QUOTE",
            params={"symbol": ticker},
        )

        quote = data.get("Global Quote", {})
        if not quote:
            raise TickerNotFoundError(f"No quote data for ticker '{ticker}'.")

        return QuoteResponse(
            ticker=quote["01. symbol"],
            open=float(quote["02. open"]),
            high=float(quote["03. high"]),
            low=float(quote["04. low"]),
            price=float(quote["05. price"]),
            volume=int(quote["06. volume"]),
            latest_trading_day=quote["07. latest trading day"],
            previous_close=float(quote["08. previous close"]),
            change=float(quote["09. change"]),
            change_percent=quote["10. change percent"],
        )

    async def get_company_overview(self, ticker: str) -> CompanyInfoResponse:
        """Fetch company fundamentals and metadata.

        Args:
            ticker: Stock ticker symbol.

        Returns:
            Parsed company info.

        Raises:
            TickerNotFoundError: If the overview is empty (invalid ticker).
        """
        data = await self._request(
            function="OVERVIEW",
            params={"symbol": ticker},
        )

        if not data or "Symbol" not in data:
            raise TickerNotFoundError(f"No company info for ticker '{ticker}'.")

        return CompanyInfoResponse(
            ticker=data["Symbol"],
            name=data.get("Name", ""),
            description=data.get("Description", ""),
            sector=data.get("Sector", ""),
            industry=data.get("Industry", ""),
            market_cap=data.get("MarketCapitalization", ""),
            pe_ratio=data.get("PERatio", ""),
            eps=data.get("EPS", ""),
            dividend_yield=data.get("DividendYield", ""),
            fifty_two_week_high=data.get("52WeekHigh", ""),
            fifty_two_week_low=data.get("52WeekLow", ""),
        )

    async def search_tickers(self, keywords: str) -> TickerSearchResponse:
        """Search for ticker symbols by company name or keywords.

        Args:
            keywords: Search query string.

        Returns:
            Matching tickers.
        """
        data = await self._request(
            function="SYMBOL_SEARCH",
            params={"keywords": keywords},
        )

        matches = [
            TickerMatch(
                symbol=m["1. symbol"],
                name=m["2. name"],
                type=m["3. type"],
                region=m["4. region"],
                currency=m["8. currency"],
            )
            for m in data.get("bestMatches", [])
        ]

        return TickerSearchResponse(matches=matches)

    async def _request(self, function: str, params: dict[str, str]) -> dict:
        """Make a rate-limited request to the Alpha Vantage API.

        Args:
            function: AV function name (e.g. ``"GLOBAL_QUOTE"``).
            params: Additional query parameters.

        Returns:
            Parsed JSON response.

        Raises:
            ApiError: On HTTP errors or AV-specific error responses.
        """
        await self._limiter.acquire()

        query = {"function": function, "apikey": self._api_key, **params}
        logger.debug(
            "Alpha Vantage request: {function} {params}",
            function=function,
            params=params,
            server=SERVER_NAME,
        )

        try:
            response = await self._http.get(AV_BASE_URL, params=query)
            response.raise_for_status()
        except httpx.HTTPStatusError as exc:
            raise ApiError(
                f"Alpha Vantage HTTP {exc.response.status_code}: " f"{exc.response.text[:200]}"
            ) from exc

        data = response.json()
        self._check_for_errors(data)
        return data

    @staticmethod
    def _check_for_errors(data: dict) -> None:
        """Detect Alpha Vantage error responses embedded in 200 OK bodies.

        AV returns errors as ``{"Error Message": "..."}`` and rate limit
        warnings as ``{"Note": "..."}``. Both are raised as ``ApiError``.
        """
        if "Error Message" in data:
            raise ApiError(f"Alpha Vantage error: {data['Error Message']}")
        if "Note" in data:
            raise ApiError(f"Alpha Vantage rate limit: {data['Note']}")
