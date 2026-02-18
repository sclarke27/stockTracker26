"""FastMCP server assembly for news feed tools."""

from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import dataclass

import httpx
from fastmcp import Context, FastMCP
from loguru import logger

from stock_radar.mcp_servers.news_feed.clients.alpha_vantage_news import (
    AlphaVantageNewsClient,
)
from stock_radar.mcp_servers.news_feed.clients.rss import RssNewsClient
from stock_radar.mcp_servers.news_feed.config import (
    AV_RATE_LIMIT_PER_DAY,
    AV_RATE_LIMIT_PER_MINUTE,
    CACHE_TTL_NEWS,
    CACHE_TTL_SENTIMENT_SUMMARY,
    SERVER_NAME,
)
from stock_radar.mcp_servers.news_feed.exceptions import NoNewsFoundError
from stock_radar.utils.cache import Cache
from stock_radar.utils.logging import setup_logging
from stock_radar.utils.rate_limiter import RateLimiter


@dataclass
class ServerDeps:
    """Shared dependencies initialized during server lifespan."""

    av_client: AlphaVantageNewsClient
    rss_client: RssNewsClient
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


def _get_api_key() -> str:
    """Return the Alpha Vantage API key from config.

    Extracted as a function so tests can patch environment variables.
    """
    from stock_radar.config.loader import load_config
    from stock_radar.config.settings import AppSettings

    config = load_config()
    settings = AppSettings(**config)
    return settings.api_keys.alpha_vantage


@asynccontextmanager
async def lifespan(server: FastMCP) -> AsyncIterator[ServerDeps]:
    """Initialize and tear down server dependencies.

    Creates the HTTP client, rate limiter, cache, and API clients.
    All are available to tools via ``ctx.fastmcp._lifespan_result``.
    """
    setup_logging()

    db_path = _get_db_path()
    av_key = _get_api_key()

    http_client = httpx.AsyncClient(timeout=30.0)
    rate_limiter = RateLimiter(
        requests_per_minute=AV_RATE_LIMIT_PER_MINUTE,
        requests_per_day=AV_RATE_LIMIT_PER_DAY,
    )
    cache = Cache(db_path=db_path)
    await cache.initialize()

    av_client = AlphaVantageNewsClient(
        api_key=av_key,
        http_client=http_client,
        rate_limiter=rate_limiter,
    )
    rss_client = RssNewsClient(http_client=http_client)

    logger.info("News feed MCP server started", server=SERVER_NAME)

    deps = ServerDeps(
        av_client=av_client,
        rss_client=rss_client,
        cache=cache,
        http_client=http_client,
    )
    try:
        yield deps
    finally:
        await cache.close()
        await http_client.aclose()
        logger.info("News feed MCP server stopped", server=SERVER_NAME)


def _deps(ctx: Context) -> ServerDeps:
    """Extract server dependencies from the tool context."""
    return ctx.fastmcp._lifespan_result  # type: ignore[attr-defined,return-value]


# ---------------------------------------------------------------------------
# Tool functions
# ---------------------------------------------------------------------------


async def get_news(
    ctx: Context,
    ticker: str,
    limit: int = 50,
    time_from: str | None = None,
    sort: str = "LATEST",
) -> str:
    """Get recent news articles for a stock ticker.

    Args:
        ticker: Stock ticker symbol (e.g. AAPL, MSFT).
        limit: Maximum number of articles to return (default 50).
        time_from: Earliest publish date in format YYYYMMDDTHHMMSS.
        sort: Sort order — LATEST, EARLIEST, or RELEVANCE.
    """
    deps = _deps(ctx)
    cache_key = Cache.make_key(
        "news",
        ticker=ticker,
        limit=str(limit),
        time_from=time_from or "",
        sort=sort,
    )

    cached = await deps.cache.get(cache_key)
    if cached:
        logger.info("Cache hit: news", ticker=ticker, server=SERVER_NAME)
        return cached

    result = await deps.av_client.get_news(ticker, limit=limit, time_from=time_from, sort=sort)
    serialized = result.model_dump_json()
    await deps.cache.set(cache_key, serialized, ttl=CACHE_TTL_NEWS)
    return serialized


async def search_news(
    ctx: Context,
    query: str,
    topics: str | None = None,
    limit: int = 50,
    time_from: str | None = None,
) -> str:
    """Search for news articles by keyword query with RSS fallback.

    Queries Alpha Vantage first; falls back to Google News RSS when AV
    returns no results for the query.

    Args:
        query: Free-text search string (e.g. 'artificial intelligence chips').
        topics: Comma-separated AV topic filter (e.g. 'earnings,technology').
        limit: Maximum number of articles to return (default 50).
        time_from: Earliest publish date in format YYYYMMDDTHHMMSS.
    """
    deps = _deps(ctx)
    cache_key = Cache.make_key(
        "search_news",
        query=query,
        topics=topics or "",
        limit=str(limit),
        time_from=time_from or "",
    )

    cached = await deps.cache.get(cache_key)
    if cached:
        logger.info("Cache hit: search_news", query=query, server=SERVER_NAME)
        return cached

    try:
        result = await deps.av_client.search_news(
            query, topics=topics, limit=limit, time_from=time_from
        )
    except NoNewsFoundError:
        logger.info(
            "AV returned no news for query '{query}', falling back to RSS",
            query=query,
            server=SERVER_NAME,
        )
        result = await deps.rss_client.search_news(query, limit=limit)

    serialized = result.model_dump_json()
    await deps.cache.set(cache_key, serialized, ttl=CACHE_TTL_NEWS)
    return serialized


async def get_sentiment_summary(
    ctx: Context,
    ticker: str,
    time_from: str | None = None,
    time_to: str | None = None,
) -> str:
    """Get an aggregated sentiment summary for a ticker.

    Fetches up to 1000 articles and aggregates average sentiment score,
    bullish/bearish/neutral counts, and top topics.

    Args:
        ticker: Stock ticker symbol (e.g. AAPL, MSFT).
        time_from: Start of analysis window in format YYYYMMDDTHHMMSS.
        time_to: End of analysis window in format YYYYMMDDTHHMMSS.
    """
    deps = _deps(ctx)
    cache_key = Cache.make_key(
        "sentiment_summary",
        ticker=ticker,
        time_from=time_from or "",
        time_to=time_to or "",
    )

    cached = await deps.cache.get(cache_key)
    if cached:
        logger.info("Cache hit: sentiment_summary", ticker=ticker, server=SERVER_NAME)
        return cached

    result = await deps.av_client.get_sentiment_summary(
        ticker, time_from=time_from, time_to=time_to
    )
    serialized = result.model_dump_json()
    await deps.cache.set(cache_key, serialized, ttl=CACHE_TTL_SENTIMENT_SUMMARY)
    return serialized


# All tool functions in registration order.
TOOLS = [
    get_news,
    search_news,
    get_sentiment_summary,
]


def create_server(name: str = "news-feed") -> FastMCP:
    """Create a new FastMCP server instance with all tools registered.

    Returns a fresh instance each time — useful for testing where each
    test needs its own isolated server with independent lifespan state.
    """
    server = FastMCP(name, lifespan=lifespan)
    for tool_fn in TOOLS:
        server.tool()(tool_fn)  # type: ignore[arg-type]
    return server


# Module-level server instance for production use.
mcp = create_server()
