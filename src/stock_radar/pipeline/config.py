"""Constants for the data ingestion pipeline."""

from __future__ import annotations

from pathlib import Path

# Tool lists per coverage tier — names must match MCP server tool registrations.
DEEP_MARKET_TOOLS = [
    "get_price_history",
    "get_quote",
    "get_company_info",
    "get_earnings_transcript",
]
DEEP_EDGAR_TOOLS = [
    "get_filings",
    "get_insider_transactions",
]
LIGHT_MARKET_TOOLS = [
    "get_quote",
]
LIGHT_EDGAR_TOOLS = [
    "get_filings",
]

# Default watchlist location (relative to project root).
DEFAULT_WATCHLIST_PATH = Path(__file__).resolve().parents[3] / "config" / "watchlist.yaml"

# Tier names for structured logging and result tagging.
TIER_DEEP = "deep"
TIER_LIGHT = "light"
