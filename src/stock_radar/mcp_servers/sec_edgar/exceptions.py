"""Exceptions for the SEC EDGAR MCP server."""

from __future__ import annotations


class SecEdgarError(Exception):
    """Base exception for SEC EDGAR server errors."""


class CikNotFoundError(SecEdgarError):
    """Raised when a ticker cannot be resolved to a CIK number."""


class FilingNotFoundError(SecEdgarError):
    """Raised when a specific filing cannot be found or accessed."""


class ApiError(SecEdgarError):
    """Raised when the SEC EDGAR API returns an error response."""
