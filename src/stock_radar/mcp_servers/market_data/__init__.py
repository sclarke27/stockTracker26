"""Market data MCP server — Alpha Vantage prices, fundamentals, Finnhub transcripts."""

from __future__ import annotations

from stock_radar.mcp_servers.market_data.server import mcp

__all__ = ["mcp", "main"]


def main() -> None:
    """Run the market data MCP server via stdio transport."""
    mcp.run()
