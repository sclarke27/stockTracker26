"""Exceptions for the market data MCP server."""

from __future__ import annotations


class MarketDataError(Exception):
    """Base exception for market data server errors."""


class RateLimitExceededError(MarketDataError):
    """Raised when the API rate limit is exhausted for the day."""


class ApiError(MarketDataError):
    """Raised when an external API returns an error response."""


class CacheError(MarketDataError):
    """Raised when a cache operation fails."""


class TickerNotFoundError(MarketDataError):
    """Raised when a ticker symbol is not recognized by the API."""
