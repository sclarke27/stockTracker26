"""SEC EDGAR MCP server — filings, insider transactions, full-text search."""

from __future__ import annotations

from stock_radar.mcp_servers.sec_edgar.server import mcp

__all__ = ["mcp", "main"]


def main() -> None:
    """Run the SEC EDGAR MCP server via stdio transport."""
    mcp.run()
