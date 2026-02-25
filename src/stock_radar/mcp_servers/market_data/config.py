"""Constants for the market data MCP server."""

from __future__ import annotations

# Cache TTLs (seconds). None = never expires.
CACHE_TTL_QUOTE = 900  # 15 minutes
CACHE_TTL_PRICE_HISTORY = 86_400  # 24 hours
CACHE_TTL_COMPANY_INFO = 604_800  # 7 days
CACHE_TTL_TRANSCRIPT: int | None = None  # Never expires
CACHE_TTL_TICKER_SEARCH = 86_400  # 24 hours
CACHE_TTL_IPO_CALENDAR = 86_400  # 24 hours

# Alpha Vantage rate limits (Premium tier: 75 req/min)
AV_RATE_LIMIT_PER_SECOND = 1
AV_RATE_LIMIT_PER_MINUTE = 75
AV_RATE_LIMIT_PER_DAY: int | None = None  # Unlimited on Premium

# Alpha Vantage base URL
AV_BASE_URL = "https://www.alphavantage.co/query"

# Server identity for logging
SERVER_NAME = "market-data-mcp"
