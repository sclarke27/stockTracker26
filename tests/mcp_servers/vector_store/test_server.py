"""Integration tests for the vector-store MCP server."""

from __future__ import annotations

import json
from unittest.mock import patch

import chromadb
import pytest
from fastmcp import Client
from fastmcp.exceptions import ToolError

from stock_radar.mcp_servers.vector_store.server import create_server
from tests.mcp_servers.conftest import get_tool_text

MOCK_ENV = {
    "ALPHA_VANTAGE_API_KEY": "unused",
    "ANTHROPIC_API_KEY": "unused",
    "OPENAI_API_KEY": "unused",
    "SEC_EDGAR_EMAIL": "test@example.com",
}


def _fresh_client() -> chromadb.ClientAPI:
    """Create an isolated EphemeralClient for test use.

    ChromaDB EphemeralClient instances in the same process share state.
    Purging all existing collections ensures each test starts from a
    clean slate without requiring different Settings (which would conflict
    with other test modules using the default EphemeralClient).
    """
    client = chromadb.EphemeralClient()
    for col in client.list_collections():
        client.delete_collection(col.name)
    return client


async def _store_document(client: Client, **overrides: object) -> dict:
    """Store a document with sensible defaults, returning the parsed response.

    Any keyword argument overrides the default value.
    """
    args: dict = {
        "document_type": "transcript",
        "document_id": "AAPL_2024_Q4",
        "content": (
            "Apple Q4 earnings call. Revenue exceeded expectations" " with strong iPhone sales."
        ),
        "ticker": "AAPL",
    }
    args.update(overrides)
    result = await client.call_tool("store_embedding", args)
    return json.loads(get_tool_text(result))


# ---------------------------------------------------------------------------
# store_embedding
# ---------------------------------------------------------------------------


class TestStoreEmbedding:
    """Tests for the store_embedding tool."""

    async def test_store_returns_id_and_collection(self) -> None:
        """Store a transcript, verify response has correct fields."""
        with (
            patch.dict("os.environ", MOCK_ENV),
            patch(
                "stock_radar.mcp_servers.vector_store.server._create_client",
                return_value=_fresh_client(),
            ),
        ):
            async with Client(create_server()) as client:
                data = await _store_document(client)

        assert data["document_id"] == "AAPL_2024_Q4"
        assert data["document_type"] == "transcript"
        assert data["collection_name"] == "transcripts"

    async def test_store_all_document_types(self) -> None:
        """Store one of each type; each succeeds and returns correct collection."""
        expected = {
            "transcript": "transcripts",
            "filing": "filings",
            "reasoning": "reasoning",
        }
        with (
            patch.dict("os.environ", MOCK_ENV),
            patch(
                "stock_radar.mcp_servers.vector_store.server._create_client",
                return_value=_fresh_client(),
            ),
        ):
            async with Client(create_server()) as client:
                for doc_type, collection_name in expected.items():
                    data = await _store_document(
                        client,
                        document_type=doc_type,
                        document_id=f"DOC_{doc_type}",
                        content=f"Content for {doc_type} document.",
                    )
                    assert data["document_type"] == doc_type
                    assert data["collection_name"] == collection_name

    async def test_store_invalid_document_type(self) -> None:
        """Invalid document_type raises ToolError."""
        with (
            patch.dict("os.environ", MOCK_ENV),
            patch(
                "stock_radar.mcp_servers.vector_store.server._create_client",
                return_value=_fresh_client(),
            ),
        ):
            async with Client(create_server()) as client:
                with pytest.raises(ToolError, match="Invalid document_type"):
                    await _store_document(client, document_type="invalid")


# ---------------------------------------------------------------------------
# search_similar
# ---------------------------------------------------------------------------


class TestSearchSimilar:
    """Tests for the search_similar tool."""

    async def test_store_and_search_roundtrip(self) -> None:
        """Store 2 transcript docs, search returns results with content and metadata."""
        with (
            patch.dict("os.environ", MOCK_ENV),
            patch(
                "stock_radar.mcp_servers.vector_store.server._create_client",
                return_value=_fresh_client(),
            ),
        ):
            async with Client(create_server()) as client:
                await _store_document(
                    client,
                    document_id="AAPL_2024_Q4",
                    content="Apple Q4 earnings call. Revenue exceeded expectations.",
                )
                await _store_document(
                    client,
                    document_id="MSFT_2024_Q4",
                    content="Microsoft Q4 earnings call. Cloud growth accelerated.",
                    ticker="MSFT",
                )
                result = await client.call_tool(
                    "search_similar",
                    {"document_type": "transcript", "query": "earnings call"},
                )

        data = json.loads(get_tool_text(result))
        assert data["query"] == "earnings call"
        assert data["document_type"] == "transcript"
        assert len(data["results"]) == 2
        for doc in data["results"]:
            assert "document_id" in doc
            assert "content" in doc
            assert "metadata" in doc
            assert "distance" in doc
            assert doc["distance"] >= 0.0

    async def test_search_empty_collection(self) -> None:
        """Search before storing anything returns empty results list."""
        with (
            patch.dict("os.environ", MOCK_ENV),
            patch(
                "stock_radar.mcp_servers.vector_store.server._create_client",
                return_value=_fresh_client(),
            ),
        ):
            async with Client(create_server()) as client:
                result = await client.call_tool(
                    "search_similar",
                    {"document_type": "transcript", "query": "anything at all"},
                )

        data = json.loads(get_tool_text(result))
        assert data["results"] == []

    async def test_search_with_ticker_filter(self) -> None:
        """Store AAPL and MSFT docs, search with ticker=AAPL returns only AAPL."""
        with (
            patch.dict("os.environ", MOCK_ENV),
            patch(
                "stock_radar.mcp_servers.vector_store.server._create_client",
                return_value=_fresh_client(),
            ),
        ):
            async with Client(create_server()) as client:
                await _store_document(
                    client,
                    document_id="AAPL_2024_Q4",
                    content="Apple Q4 earnings call results.",
                    ticker="AAPL",
                )
                await _store_document(
                    client,
                    document_id="MSFT_2024_Q4",
                    content="Microsoft Q4 earnings call results.",
                    ticker="MSFT",
                )
                result = await client.call_tool(
                    "search_similar",
                    {
                        "document_type": "transcript",
                        "query": "quarterly earnings",
                        "ticker": "AAPL",
                    },
                )

        data = json.loads(get_tool_text(result))
        assert len(data["results"]) == 1
        assert data["results"][0]["document_id"] == "AAPL_2024_Q4"
        assert data["results"][0]["metadata"]["ticker"] == "AAPL"

    async def test_search_invalid_document_type(self) -> None:
        """Invalid document_type raises ToolError."""
        with (
            patch.dict("os.environ", MOCK_ENV),
            patch(
                "stock_radar.mcp_servers.vector_store.server._create_client",
                return_value=_fresh_client(),
            ),
        ):
            async with Client(create_server()) as client:
                with pytest.raises(ToolError, match="Invalid document_type"):
                    await client.call_tool(
                        "search_similar",
                        {"document_type": "invalid", "query": "test"},
                    )


# ---------------------------------------------------------------------------
# get_document
# ---------------------------------------------------------------------------


class TestGetDocument:
    """Tests for the get_document tool."""

    async def test_store_and_get(self) -> None:
        """Store then get, verify content present and found=True."""
        with (
            patch.dict("os.environ", MOCK_ENV),
            patch(
                "stock_radar.mcp_servers.vector_store.server._create_client",
                return_value=_fresh_client(),
            ),
        ):
            async with Client(create_server()) as client:
                await _store_document(
                    client,
                    document_id="AAPL_2024_Q4",
                    content="Apple Q4 earnings call content.",
                )
                result = await client.call_tool(
                    "get_document",
                    {"document_type": "transcript", "document_id": "AAPL_2024_Q4"},
                )

        data = json.loads(get_tool_text(result))
        assert data["found"] is True
        assert data["document_id"] == "AAPL_2024_Q4"
        assert data["document_type"] == "transcript"
        assert data["content"] == "Apple Q4 earnings call content."
        assert data["metadata"]["ticker"] == "AAPL"

    async def test_get_nonexistent(self) -> None:
        """Get unknown ID returns found=False, content=None, metadata=None."""
        with (
            patch.dict("os.environ", MOCK_ENV),
            patch(
                "stock_radar.mcp_servers.vector_store.server._create_client",
                return_value=_fresh_client(),
            ),
        ):
            async with Client(create_server()) as client:
                result = await client.call_tool(
                    "get_document",
                    {"document_type": "transcript", "document_id": "DOES_NOT_EXIST"},
                )

        data = json.loads(get_tool_text(result))
        assert data["found"] is False
        assert data["document_id"] == "DOES_NOT_EXIST"
        assert data["content"] is None
        assert data["metadata"] is None

    async def test_get_invalid_document_type(self) -> None:
        """Invalid document_type raises ToolError."""
        with (
            patch.dict("os.environ", MOCK_ENV),
            patch(
                "stock_radar.mcp_servers.vector_store.server._create_client",
                return_value=_fresh_client(),
            ),
        ):
            async with Client(create_server()) as client:
                with pytest.raises(ToolError, match="Invalid document_type"):
                    await client.call_tool(
                        "get_document",
                        {"document_type": "invalid", "document_id": "doc1"},
                    )
