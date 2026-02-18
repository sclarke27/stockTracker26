"""Dashboard MCP server — read-only HTTP API for the frontend."""

from stock_radar.mcp_servers.dashboard.config import DEFAULT_HOST, DEFAULT_PORT
from stock_radar.mcp_servers.dashboard.server import create_server
from stock_radar.utils.logging import setup_logging


def main() -> None:
    """CLI entry point for the dashboard MCP HTTP server."""
    setup_logging()
    server = create_server()
    server.run(transport="streamable-http", host=DEFAULT_HOST, port=DEFAULT_PORT)


__all__ = ["create_server", "main"]
