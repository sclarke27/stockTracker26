"""Watchlist configuration loader for the ingestion pipeline."""

from __future__ import annotations

from pathlib import Path

import yaml
from pydantic import BaseModel, Field

from stock_radar.pipeline.config import DEFAULT_WATCHLIST_PATH


class WatchlistTicker(BaseModel):
    """A single ticker entry in the watchlist."""

    symbol: str = Field(description="Stock ticker symbol (e.g. AAPL)")
    name: str = Field(default="", description="Company name (optional)")


class Watchlist(BaseModel):
    """Categorized watchlist of tickers by coverage tier."""

    deep: list[WatchlistTicker] = Field(description="Deep-coverage tickers (full data)")
    light: list[WatchlistTicker] = Field(
        default_factory=list,
        description="Light-coverage tickers (quotes + filings only)",
    )


def load_watchlist(path: Path | None = None) -> Watchlist:
    """Load watchlist from a YAML file.

    Args:
        path: Path to the YAML file. Defaults to config/watchlist.yaml.

    Returns:
        Parsed Watchlist with deep and light tickers.

    Raises:
        FileNotFoundError: If the watchlist file does not exist.
    """
    resolved = path if path is not None else DEFAULT_WATCHLIST_PATH

    if not resolved.exists():
        raise FileNotFoundError(f"Watchlist file not found: {resolved}")

    with open(resolved, encoding="utf-8") as fh:
        data: dict = yaml.safe_load(fh)

    return Watchlist(**data)
