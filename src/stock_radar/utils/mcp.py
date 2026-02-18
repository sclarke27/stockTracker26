"""Helpers for working with FastMCP tool call results."""

from __future__ import annotations

from fastmcp.client.client import CallToolResult
from mcp.types import TextContent


def get_tool_text(result: CallToolResult) -> str:
    """Extract the text payload from a FastMCP ``CallToolResult``.

    Every tool in this project returns exactly one ``TextContent`` element.
    This helper centralises the extraction and asserts the content type so
    static analysers see a plain ``str`` rather than the broad union type
    that FastMCP exposes.

    Args:
        result: The ``CallToolResult`` returned by ``Client.call_tool()``.

    Returns:
        The text string from the first content element.

    Raises:
        AssertionError: If the first content element is not ``TextContent``.
    """
    content = result.content[0]
    assert isinstance(content, TextContent), f"Expected TextContent, got {type(content).__name__}"
    return content.text
