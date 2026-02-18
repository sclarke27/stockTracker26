"""News feed MCP server — Alpha Vantage news sentiment, RSS feeds."""

from __future__ import annotations

from stock_radar.mcp_servers.news_feed.server import mcp

__all__ = ["mcp", "main"]


def main() -> None:
    """Run the news feed MCP server via stdio transport."""
    mcp.run()
