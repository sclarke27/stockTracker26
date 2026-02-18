"""Shared fixtures for MCP server integration tests."""

from __future__ import annotations

from pathlib import Path

import pytest


def get_tool_text(result: object) -> str:
    """Extract the text content from a FastMCP CallToolResult.

    Every MCP tool in the project returns JSON as the first text content
    element.  This helper avoids repeating ``result.content[0].text``
    across every test module.
    """
    return result.content[0].text


@pytest.fixture()
def tmp_db(tmp_path: Path) -> str:
    """Provide a path for a temporary SQLite database.

    Works for both cache-backed servers (market-data, sec-edgar) and the
    predictions-db server.  Each test gets its own ``tmp_path`` so databases
    are always isolated.
    """
    return str(tmp_path / "test.db")
