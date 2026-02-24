"""Dashboard MCP server — read-only HTTP API for the frontend."""

from __future__ import annotations

import uvicorn
from starlette.middleware import Middleware
from starlette.middleware.cors import CORSMiddleware

from stock_radar.mcp_servers.dashboard.config import DEFAULT_HOST, DEFAULT_PORT
from stock_radar.mcp_servers.dashboard.server import create_server
from stock_radar.utils.logging import setup_logging

CORS_MIDDLEWARE = [
    Middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["*"],
        allow_headers=["*"],
        expose_headers=["mcp-session-id"],
    ),
]


def main() -> None:
    """CLI entry point for the dashboard MCP HTTP server."""
    setup_logging()
    server = create_server()
    app = server.http_app(middleware=CORS_MIDDLEWARE)
    uvicorn.run(app, host=DEFAULT_HOST, port=DEFAULT_PORT)


__all__ = ["create_server", "main"]
