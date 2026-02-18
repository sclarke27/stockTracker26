"""Predictions database MCP server — logging and scoring."""

from __future__ import annotations

from stock_radar.mcp_servers.predictions_db.server import mcp

__all__ = ["mcp", "main"]


def main() -> None:
    """Run the predictions database MCP server via stdio transport."""
    mcp.run()
