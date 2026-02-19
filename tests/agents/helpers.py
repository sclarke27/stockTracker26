"""Shared test helpers for agent tests."""

from __future__ import annotations

from unittest.mock import MagicMock

from mcp.types import TextContent


def mock_tool_response(data: str) -> MagicMock:
    """Create a mock MCP tool call response containing JSON text.

    Args:
        data: JSON string to wrap in a TextContent-shaped mock.

    Returns:
        MagicMock with ``content[0].text == data``.
    """
    return MagicMock(content=[MagicMock(spec=TextContent, text=data)])
