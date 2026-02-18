"""Exceptions for the news feed MCP server."""

from __future__ import annotations


class NewsFeedError(Exception):
    """Base exception for news feed server errors."""


class ApiError(NewsFeedError):
    """Raised when an external API returns an error response."""


class NoNewsFoundError(NewsFeedError):
    """Raised when a news query returns no results; signals RSS fallback."""
