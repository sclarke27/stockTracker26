"""Pydantic models for market data MCP server inputs and outputs."""

from __future__ import annotations

from pydantic import BaseModel, Field

# === Response Models ===


class OHLCVBar(BaseModel):
    """Single OHLCV price bar for a trading day."""

    date: str = Field(description="Trading date (YYYY-MM-DD)")
    open: float = Field(description="Opening price")
    high: float = Field(description="High price")
    low: float = Field(description="Low price")
    close: float = Field(description="Closing price")
    volume: int = Field(description="Trading volume")


class PriceHistoryResponse(BaseModel):
    """Daily OHLCV price history for a ticker."""

    ticker: str = Field(description="Stock ticker symbol")
    bars: list[OHLCVBar] = Field(description="Daily OHLCV bars, most recent first")
    last_refreshed: str = Field(description="Last data refresh timestamp from API")


class QuoteResponse(BaseModel):
    """Current quote data for a ticker."""

    ticker: str = Field(description="Stock ticker symbol")
    price: float = Field(description="Current/last traded price")
    change: float = Field(description="Price change from previous close")
    change_percent: str = Field(description="Price change percentage (e.g. '1.26%')")
    volume: int = Field(description="Current trading volume")
    latest_trading_day: str = Field(description="Most recent trading date")
    previous_close: float = Field(description="Previous session close price")
    open: float = Field(description="Opening price")
    high: float = Field(description="Day high")
    low: float = Field(description="Day low")


class CompanyInfoResponse(BaseModel):
    """Company fundamentals and metadata."""

    ticker: str = Field(description="Stock ticker symbol")
    name: str = Field(description="Company name")
    description: str = Field(description="Company description")
    sector: str = Field(description="Business sector")
    industry: str = Field(description="Business industry")
    market_cap: str = Field(description="Market capitalization")
    pe_ratio: str = Field(description="Price-to-earnings ratio")
    eps: str = Field(description="Earnings per share")
    dividend_yield: str = Field(description="Dividend yield")
    fifty_two_week_high: str = Field(description="52-week high price")
    fifty_two_week_low: str = Field(description="52-week low price")


class TickerMatch(BaseModel):
    """Single result from a ticker symbol search."""

    symbol: str = Field(description="Ticker symbol")
    name: str = Field(description="Company name")
    type: str = Field(description="Security type (e.g. Equity, ETF)")
    region: str = Field(description="Market region")
    currency: str = Field(description="Trading currency")


class TickerSearchResponse(BaseModel):
    """Results from a ticker symbol search."""

    matches: list[TickerMatch] = Field(description="Matching tickers")


class TranscriptSegment(BaseModel):
    """Single speaker segment from an earnings call transcript."""

    speaker: str = Field(description="Speaker name")
    title: str = Field(description="Speaker title/role")
    content: str = Field(description="Spoken content")
    sentiment: str = Field(default="", description="Sentiment label for this segment")


class EarningsTranscriptResponse(BaseModel):
    """Earnings call transcript for a specific quarter."""

    ticker: str = Field(description="Stock ticker symbol")
    quarter: int = Field(description="Fiscal quarter (1-4)")
    year: int = Field(description="Fiscal year")
    content: str = Field(description="Full transcript text (concatenated from segments)")
    segments: list[TranscriptSegment] = Field(
        default_factory=list,
        description="Individual speaker segments with sentiment",
    )


class IPOEntry(BaseModel):
    """Single upcoming IPO listing."""

    symbol: str = Field(description="Ticker symbol")
    name: str = Field(description="Company name")
    ipo_date: str = Field(description="Expected IPO date (YYYY-MM-DD)")
    price_range_low: float = Field(description="Low end of expected price range")
    price_range_high: float = Field(description="High end of expected price range")
    currency: str = Field(description="Trading currency (e.g. USD)")
    exchange: str = Field(description="Listing exchange (e.g. NASDAQ, NYSE)")


class IPOCalendarResponse(BaseModel):
    """Upcoming IPO calendar."""

    entries: list[IPOEntry] = Field(description="Upcoming IPO listings")
