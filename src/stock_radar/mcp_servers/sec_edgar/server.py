"""FastMCP server assembly for SEC EDGAR tools."""

from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import dataclass

import httpx
from fastmcp import Context, FastMCP
from loguru import logger

from stock_radar.mcp_servers.sec_edgar.clients.edgar import EdgarClient
from stock_radar.mcp_servers.sec_edgar.config import (
    CACHE_TTL_FILING_SEARCH,
    CACHE_TTL_FILING_TEXT,
    CACHE_TTL_FILINGS,
    CACHE_TTL_INSIDER_TRANSACTIONS,
    DEFAULT_MAX_FILING_TEXT_LENGTH,
    SEC_RATE_LIMIT_PER_SECOND,
    SEC_USER_AGENT_TEMPLATE,
    SERVER_NAME,
)
from stock_radar.utils.cache import Cache
from stock_radar.utils.logging import setup_logging
from stock_radar.utils.rate_limiter import RateLimiter


@dataclass
class ServerDeps:
    """Shared dependencies initialized during server lifespan."""

    edgar_client: EdgarClient
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


def _get_user_agent() -> str:
    """Return the SEC EDGAR User-Agent string from config.

    Extracted as a function so tests can patch it.
    """
    from stock_radar.config.loader import load_config
    from stock_radar.config.settings import AppSettings

    config = load_config()
    settings = AppSettings(**config)
    return SEC_USER_AGENT_TEMPLATE.format(email=settings.sec_edgar.user_agent_email)


@asynccontextmanager
async def lifespan(server: FastMCP) -> AsyncIterator[ServerDeps]:
    """Initialize and tear down server dependencies.

    Creates the HTTP client, rate limiter, cache, and EDGAR client.
    All are available to tools via ``ctx.fastmcp._lifespan_result``.
    """
    setup_logging()

    db_path = _get_db_path()
    user_agent = _get_user_agent()

    http_client = httpx.AsyncClient(timeout=30.0)
    rate_limiter = RateLimiter(
        requests_per_minute=600,
        requests_per_day=50_000,
        requests_per_second=SEC_RATE_LIMIT_PER_SECOND,
    )
    cache = Cache(db_path=db_path)
    await cache.initialize()

    edgar_client = EdgarClient(
        http_client=http_client,
        rate_limiter=rate_limiter,
        user_agent=user_agent,
    )

    logger.info("SEC EDGAR MCP server started", server=SERVER_NAME)

    deps = ServerDeps(
        edgar_client=edgar_client,
        cache=cache,
        http_client=http_client,
    )
    try:
        yield deps
    finally:
        await cache.close()
        await http_client.aclose()
        logger.info("SEC EDGAR MCP server stopped", server=SERVER_NAME)


def _deps(ctx: Context) -> ServerDeps:
    """Extract server dependencies from the tool context."""
    return ctx.fastmcp._lifespan_result


# ---------------------------------------------------------------------------
# Tool functions (plain async functions, registered on a server via
# ``create_server()`` or the module-level ``mcp`` instance).
# ---------------------------------------------------------------------------


async def get_filings(
    ctx: Context,
    ticker: str,
    form_types: list[str] | None = None,
    limit: int = 50,
) -> str:
    """Get recent SEC filings for a company.

    Args:
        ticker: Stock ticker symbol (e.g. AAPL, MSFT).
        form_types: Filter by form types (e.g. ['8-K', '10-K']).
        limit: Maximum number of filings to return (default 50).
    """
    deps = _deps(ctx)
    cache_key = Cache.make_key(
        "filings", ticker=ticker, form_types=str(form_types), limit=str(limit)
    )

    cached = await deps.cache.get(cache_key)
    if cached:
        logger.info("Cache hit: filings", ticker=ticker, server=SERVER_NAME)
        return cached

    result = await deps.edgar_client.get_filings(ticker, form_types, limit)
    serialized = result.model_dump_json()
    await deps.cache.set(cache_key, serialized, ttl=CACHE_TTL_FILINGS)
    return serialized


async def get_filing_text(
    ctx: Context,
    ticker: str,
    accession_number: str,
    max_length: int = DEFAULT_MAX_FILING_TEXT_LENGTH,
) -> str:
    """Get the full text content of a specific SEC filing.

    Args:
        ticker: Stock ticker symbol (e.g. AAPL, MSFT).
        accession_number: SEC accession number (e.g. '0000320193-25-000001').
        max_length: Maximum characters to return (default 100000).
    """
    deps = _deps(ctx)
    cache_key = Cache.make_key(
        "filing_text",
        ticker=ticker,
        accession_number=accession_number,
        max_length=str(max_length),
    )

    cached = await deps.cache.get(cache_key)
    if cached:
        logger.info(
            "Cache hit: filing_text",
            ticker=ticker,
            accession_number=accession_number,
            server=SERVER_NAME,
        )
        return cached

    result = await deps.edgar_client.get_filing_text(ticker, accession_number, max_length)
    serialized = result.model_dump_json()
    await deps.cache.set(cache_key, serialized, ttl=CACHE_TTL_FILING_TEXT)
    return serialized


async def get_insider_transactions(
    ctx: Context,
    ticker: str,
    limit: int = 20,
) -> str:
    """Get insider transactions for a company from Form 3/4/5 filings.

    Args:
        ticker: Stock ticker symbol (e.g. AAPL, MSFT).
        limit: Maximum number of insider filings to parse (default 20).
    """
    deps = _deps(ctx)
    cache_key = Cache.make_key("insider_transactions", ticker=ticker, limit=str(limit))

    cached = await deps.cache.get(cache_key)
    if cached:
        logger.info("Cache hit: insider_transactions", ticker=ticker, server=SERVER_NAME)
        return cached

    result = await deps.edgar_client.get_insider_transactions(ticker, limit)
    serialized = result.model_dump_json()
    await deps.cache.set(cache_key, serialized, ttl=CACHE_TTL_INSIDER_TRANSACTIONS)
    return serialized


async def search_filings(
    ctx: Context,
    query: str,
    form_types: list[str] | None = None,
    date_from: str | None = None,
    date_to: str | None = None,
) -> str:
    """Search EDGAR filings using full-text search.

    Args:
        query: Search query text.
        form_types: Filter by form types (e.g. ['8-K', '10-K']).
        date_from: Start date (YYYY-MM-DD).
        date_to: End date (YYYY-MM-DD).
    """
    deps = _deps(ctx)
    cache_key = Cache.make_key(
        "filing_search",
        query=query,
        form_types=str(form_types),
        date_from=str(date_from),
        date_to=str(date_to),
    )

    cached = await deps.cache.get(cache_key)
    if cached:
        logger.info("Cache hit: filing_search", query=query, server=SERVER_NAME)
        return cached

    result = await deps.edgar_client.search_filings(query, form_types, date_from, date_to)
    serialized = result.model_dump_json()
    await deps.cache.set(cache_key, serialized, ttl=CACHE_TTL_FILING_SEARCH)
    return serialized


# All tool functions in registration order.
TOOLS = [
    get_filings,
    get_filing_text,
    get_insider_transactions,
    search_filings,
]


def create_server(name: str = "sec-edgar") -> FastMCP:
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
