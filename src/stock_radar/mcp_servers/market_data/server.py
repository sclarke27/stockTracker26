"""FastMCP server assembly for market data tools."""

from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import dataclass

import httpx
from fastmcp import Context, FastMCP
from loguru import logger

from stock_radar.mcp_servers.market_data.clients.alpha_vantage import AlphaVantageClient
from stock_radar.mcp_servers.market_data.clients.finnhub import FinnhubClient
from stock_radar.mcp_servers.market_data.config import (
    AV_RATE_LIMIT_PER_DAY,
    AV_RATE_LIMIT_PER_MINUTE,
    CACHE_TTL_COMPANY_INFO,
    CACHE_TTL_PRICE_HISTORY,
    CACHE_TTL_QUOTE,
    CACHE_TTL_TICKER_SEARCH,
    CACHE_TTL_TRANSCRIPT,
    SERVER_NAME,
)
from stock_radar.utils.cache import Cache
from stock_radar.utils.logging import setup_logging
from stock_radar.utils.rate_limiter import RateLimiter


@dataclass
class ServerDeps:
    """Shared dependencies initialized during server lifespan."""

    av_client: AlphaVantageClient
    fh_client: FinnhubClient
    cache: Cache
    http_client: httpx.AsyncClient


def _get_db_path() -> str:
    """Return the cache database path from config.

    Extracted as a function so tests can patch it with a temporary path.
    """
    from stock_radar.config.loader import load_config
    from stock_radar.config.settings import AppSettings

    config = load_config()
    settings = AppSettings(**config)
    return settings.cache.db_path


def _get_api_keys() -> tuple[str, str]:
    """Return (alpha_vantage_key, finnhub_key) from config.

    Extracted as a function so tests can patch environment variables.
    """
    from stock_radar.config.loader import load_config
    from stock_radar.config.settings import AppSettings

    config = load_config()
    settings = AppSettings(**config)
    return settings.api_keys.alpha_vantage, settings.api_keys.finnhub


@asynccontextmanager
async def lifespan(server: FastMCP) -> AsyncIterator[ServerDeps]:
    """Initialize and tear down server dependencies.

    Creates the HTTP client, rate limiter, cache, and API clients.
    All are available to tools via ``ctx.fastmcp._lifespan_result``.
    """
    setup_logging()

    db_path = _get_db_path()
    av_key, fh_key = _get_api_keys()

    http_client = httpx.AsyncClient(timeout=30.0)
    rate_limiter = RateLimiter(
        requests_per_minute=AV_RATE_LIMIT_PER_MINUTE,
        requests_per_day=AV_RATE_LIMIT_PER_DAY,
    )
    cache = Cache(db_path=db_path)
    await cache.initialize()

    av_client = AlphaVantageClient(
        api_key=av_key,
        http_client=http_client,
        rate_limiter=rate_limiter,
    )
    fh_client = FinnhubClient(api_key=fh_key, http_client=http_client)

    logger.info("Market data MCP server started", server=SERVER_NAME)

    deps = ServerDeps(
        av_client=av_client,
        fh_client=fh_client,
        cache=cache,
        http_client=http_client,
    )
    try:
        yield deps
    finally:
        await cache.close()
        await http_client.aclose()
        logger.info("Market data MCP server stopped", server=SERVER_NAME)


def _deps(ctx: Context) -> ServerDeps:
    """Extract server dependencies from the tool context."""
    return ctx.fastmcp._lifespan_result


# ---------------------------------------------------------------------------
# Tool functions (plain async functions, registered on a server via
# ``create_server()`` or the module-level ``mcp`` instance).
# ---------------------------------------------------------------------------


async def get_price_history(
    ctx: Context,
    ticker: str,
    outputsize: str = "compact",
) -> str:
    """Get daily OHLCV price history for a stock.

    Args:
        ticker: Stock ticker symbol (e.g. AAPL, MSFT).
        outputsize: 'compact' for last 100 days, 'full' for 20+ years.
    """
    deps = _deps(ctx)
    cache_key = Cache.make_key("price_history", ticker=ticker, outputsize=outputsize)

    cached = await deps.cache.get(cache_key)
    if cached:
        logger.info("Cache hit: price_history", ticker=ticker, server=SERVER_NAME)
        return cached

    result = await deps.av_client.get_daily_prices(ticker, outputsize)
    serialized = result.model_dump_json()
    await deps.cache.set(cache_key, serialized, ttl=CACHE_TTL_PRICE_HISTORY)
    return serialized


async def get_quote(ctx: Context, ticker: str) -> str:
    """Get the current quote for a stock.

    Args:
        ticker: Stock ticker symbol (e.g. AAPL, MSFT).
    """
    deps = _deps(ctx)
    cache_key = Cache.make_key("quote", ticker=ticker)

    cached = await deps.cache.get(cache_key)
    if cached:
        logger.info("Cache hit: quote", ticker=ticker, server=SERVER_NAME)
        return cached

    result = await deps.av_client.get_quote(ticker)
    serialized = result.model_dump_json()
    await deps.cache.set(cache_key, serialized, ttl=CACHE_TTL_QUOTE)
    return serialized


async def get_company_info(ctx: Context, ticker: str) -> str:
    """Get company fundamentals and metadata.

    Args:
        ticker: Stock ticker symbol (e.g. AAPL, MSFT).
    """
    deps = _deps(ctx)
    cache_key = Cache.make_key("company_info", ticker=ticker)

    cached = await deps.cache.get(cache_key)
    if cached:
        logger.info("Cache hit: company_info", ticker=ticker, server=SERVER_NAME)
        return cached

    result = await deps.av_client.get_company_overview(ticker)
    serialized = result.model_dump_json()
    await deps.cache.set(cache_key, serialized, ttl=CACHE_TTL_COMPANY_INFO)
    return serialized


async def search_tickers(ctx: Context, keywords: str) -> str:
    """Search for stock ticker symbols by company name or keywords.

    Args:
        keywords: Search query (e.g. 'apple', 'microsoft').
    """
    deps = _deps(ctx)
    cache_key = Cache.make_key("ticker_search", keywords=keywords)

    cached = await deps.cache.get(cache_key)
    if cached:
        logger.info("Cache hit: ticker_search", keywords=keywords, server=SERVER_NAME)
        return cached

    result = await deps.av_client.search_tickers(keywords)
    serialized = result.model_dump_json()
    await deps.cache.set(cache_key, serialized, ttl=CACHE_TTL_TICKER_SEARCH)
    return serialized


async def get_earnings_transcript(
    ctx: Context,
    ticker: str,
    quarter: int,
    year: int,
) -> str:
    """Get an earnings call transcript for a specific quarter.

    Args:
        ticker: Stock ticker symbol (e.g. AAPL, MSFT).
        quarter: Fiscal quarter (1-4).
        year: Fiscal year (e.g. 2024).
    """
    deps = _deps(ctx)
    cache_key = Cache.make_key("transcript", ticker=ticker, quarter=str(quarter), year=str(year))

    cached = await deps.cache.get(cache_key)
    if cached:
        logger.info("Cache hit: transcript", ticker=ticker, server=SERVER_NAME)
        return cached

    result = await deps.fh_client.get_earnings_transcript(ticker, quarter, year)
    serialized = result.model_dump_json()
    await deps.cache.set(cache_key, serialized, ttl=CACHE_TTL_TRANSCRIPT)
    return serialized


# All tool functions in registration order.
TOOLS = [
    get_price_history,
    get_quote,
    get_company_info,
    search_tickers,
    get_earnings_transcript,
]


def create_server(name: str = "market-data") -> FastMCP:
    """Create a new FastMCP server instance with all tools registered.

    Returns a fresh instance each time — useful for testing where each
    test needs its own isolated server with independent lifespan state.
    """
    server = FastMCP(name, lifespan=lifespan)
    for tool_fn in TOOLS:
        server.tool()(tool_fn)
    return server


# Module-level server instance for production use.
mcp = create_server()
