"""Vector store MCP server — ChromaDB embeddings."""

from __future__ import annotations

from stock_radar.mcp_servers.vector_store.server import mcp

__all__ = ["mcp", "main"]


def main() -> None:
    """Run the vector store MCP server via stdio transport."""
    mcp.run()
