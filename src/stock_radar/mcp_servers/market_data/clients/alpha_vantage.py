"""Alpha Vantage API client for market data retrieval."""

from __future__ import annotations

import csv
import io

import httpx
from loguru import logger

from stock_radar.mcp_servers.market_data.config import AV_BASE_URL, SERVER_NAME
from stock_radar.mcp_servers.market_data.exceptions import ApiError, TickerNotFoundError
from stock_radar.models.market_data import (
    CompanyInfoResponse,
    EarningsTranscriptResponse,
    IPOCalendarResponse,
    IPOEntry,
    OHLCVBar,
    PriceHistoryResponse,
    QuoteResponse,
    TickerMatch,
    TickerSearchResponse,
    TranscriptSegment,
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

    async def get_earnings_transcript(
        self,
        ticker: str,
        quarter: int,
        year: int,
    ) -> EarningsTranscriptResponse:
        """Fetch an earnings call transcript for a specific quarter.

        The Alpha Vantage API returns a ``transcript`` array of speaker
        segments, each with ``speaker``, ``title``, ``content``, and
        ``sentiment`` fields.  We concatenate these into a single text
        string and also preserve the structured segments.

        Args:
            ticker: Stock ticker symbol (e.g. ``"AAPL"``).
            quarter: Fiscal quarter (1-4).
            year: Fiscal year.

        Returns:
            Parsed transcript response.

        Raises:
            TickerNotFoundError: If no transcript content is available.
        """
        av_quarter = f"{year}Q{quarter}"
        data = await self._request(
            function="EARNINGS_CALL_TRANSCRIPT",
            params={"symbol": ticker, "quarter": av_quarter},
        )

        segments = self._parse_transcript_segments(data)
        if not segments:
            raise TickerNotFoundError(f"Empty transcript for {ticker} Q{quarter} {year}.")

        content = "\n\n".join(f"{seg.speaker} ({seg.title}): {seg.content}" for seg in segments)

        # AV returns quarter as "2024Q4"; extract the trailing digit.
        raw_quarter = data.get("quarter", av_quarter)
        if isinstance(raw_quarter, str) and "Q" in raw_quarter:
            parsed_quarter = int(raw_quarter[-1])
        else:
            parsed_quarter = int(raw_quarter) if raw_quarter else quarter

        return EarningsTranscriptResponse(
            ticker=data.get("symbol", ticker),
            quarter=parsed_quarter,
            year=year,
            content=content,
            segments=segments,
        )

    @staticmethod
    def _parse_transcript_segments(data: dict) -> list[TranscriptSegment]:
        """Parse transcript segments from an AV API response.

        Handles both the array-of-segments format (``transcript`` key)
        and a legacy flat-string format (``content`` key) for
        backwards compatibility.
        """
        raw_segments = data.get("transcript", [])
        if raw_segments:
            return [
                TranscriptSegment(
                    speaker=seg.get("speaker", ""),
                    title=seg.get("title", ""),
                    content=seg.get("content", ""),
                    sentiment=seg.get("sentiment", ""),
                )
                for seg in raw_segments
                if seg.get("content")
            ]

        # Fall back to flat content string (legacy / alternative format).
        flat_content = data.get("content", "")
        if flat_content:
            return [
                TranscriptSegment(
                    speaker="Unknown",
                    title="",
                    content=flat_content,
                )
            ]

        return []

    async def get_ipo_calendar(self) -> IPOCalendarResponse:
        """Fetch the upcoming IPO calendar.

        Returns:
            Parsed IPO calendar with upcoming listings.
        """
        raw = await self._request_csv(function="IPO_CALENDAR")

        reader = csv.DictReader(io.StringIO(raw))
        entries = [
            IPOEntry(
                symbol=row["symbol"],
                name=row["name"],
                ipo_date=row["ipoDate"],
                price_range_low=float(row["priceRangeLow"]),
                price_range_high=float(row["priceRangeHigh"]),
                currency=row["currency"],
                exchange=row["exchange"],
            )
            for row in reader
        ]

        return IPOCalendarResponse(entries=entries)

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

    async def _request_csv(self, function: str, params: dict[str, str] | None = None) -> str:
        """Make a rate-limited request expecting a CSV response.

        Args:
            function: AV function name (e.g. ``"IPO_CALENDAR"``).
            params: Additional query parameters.

        Returns:
            Raw CSV text.

        Raises:
            ApiError: On HTTP errors or AV-specific error responses.
        """
        await self._limiter.acquire()

        query = {"function": function, "apikey": self._api_key, **(params or {})}
        logger.debug(
            "Alpha Vantage CSV request: {function}",
            function=function,
            server=SERVER_NAME,
        )

        try:
            response = await self._http.get(AV_BASE_URL, params=query)
            response.raise_for_status()
        except httpx.HTTPStatusError as exc:
            raise ApiError(
                f"Alpha Vantage HTTP {exc.response.status_code}: " f"{exc.response.text[:200]}"
            ) from exc

        text = response.text

        # AV may return JSON error bodies even for CSV endpoints.
        if text.startswith("{"):
            import json

            data = json.loads(text)
            self._check_for_errors(data)

        return text

    @staticmethod
    def _check_for_errors(data: dict) -> None:
        """Detect Alpha Vantage error responses embedded in 200 OK bodies.

        AV returns errors as ``{"Error Message": "..."}`` and rate limit
        warnings as ``{"Note": "..."}``. Both are raised as ``ApiError``.
        """
        if "Error Message" in data:
            raise ApiError(f"Alpha Vantage error: {data['Error Message']}")
        if "Information" in data:
            raise ApiError(f"Alpha Vantage rate limit: {data['Information']}")
        if "Note" in data:
            raise ApiError(f"Alpha Vantage rate limit: {data['Note']}")
