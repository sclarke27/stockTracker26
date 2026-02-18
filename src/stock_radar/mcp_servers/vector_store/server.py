"""FastMCP server assembly for vector store tools."""

from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import dataclass

import chromadb
from fastmcp import Context, FastMCP
from loguru import logger

from stock_radar.mcp_servers.vector_store.config import (
    DEFAULT_N_RESULTS,
    DOCUMENT_TYPE_TO_COLLECTION,
    SERVER_NAME,
    VALID_DOCUMENT_TYPES,
)
from stock_radar.mcp_servers.vector_store.store import VectorStore
from stock_radar.models.vector_store import (
    GetDocumentResponse,
    SearchSimilarResponse,
    SimilarDocument,
    StoreEmbeddingResponse,
)
from stock_radar.utils.logging import setup_logging


@dataclass
class ServerDeps:
    """Shared dependencies initialized during server lifespan."""

    store: VectorStore


def _get_chroma_path() -> str:
    """Return ChromaDB storage path from config.

    Extracted as a function so tests can patch it with a temporary path.
    """
    from stock_radar.config.loader import load_config
    from stock_radar.config.settings import AppSettings

    config = load_config()
    settings = AppSettings(**config)
    return settings.cache.chroma_path


def _create_client(chroma_path: str) -> chromadb.ClientAPI:
    """Create ChromaDB PersistentClient.

    Extracted as a function so tests can patch it with an EphemeralClient.
    """
    return chromadb.PersistentClient(path=chroma_path)


@asynccontextmanager
async def lifespan(server: FastMCP) -> AsyncIterator[ServerDeps]:
    """Initialize and tear down server dependencies.

    Creates the VectorStore backed by ChromaDB.
    Available to tools via ``ctx.fastmcp._lifespan_result``.
    """
    setup_logging()

    chroma_path = _get_chroma_path()
    client = _create_client(chroma_path)

    store = VectorStore(client=client)
    store.initialize()

    logger.info("Vector store MCP server started", server=SERVER_NAME)

    deps = ServerDeps(store=store)
    try:
        yield deps
    finally:
        store.close()
        logger.info("Vector store MCP server stopped", server=SERVER_NAME)


def _deps(ctx: Context) -> ServerDeps:
    """Extract server dependencies from the tool context."""
    return ctx.fastmcp._lifespan_result


def _validate_document_type(document_type: str) -> str:
    """Validate document_type and return the corresponding collection name.

    Args:
        document_type: The document type to validate.

    Returns:
        The ChromaDB collection name for the given document type.

    Raises:
        ValueError: If the document type is not one of the valid types.
    """
    if document_type not in VALID_DOCUMENT_TYPES:
        raise ValueError(
            f"Invalid document_type: {document_type!r}. "
            f"Must be one of: {sorted(VALID_DOCUMENT_TYPES)}"
        )
    return DOCUMENT_TYPE_TO_COLLECTION[document_type]


# ---------------------------------------------------------------------------
# Tool functions
# ---------------------------------------------------------------------------


async def store_embedding(
    ctx: Context,
    document_type: str,
    document_id: str,
    content: str,
    ticker: str,
    metadata: dict[str, str] | None = None,
) -> str:
    """Store a document embedding in the vector store.

    Args:
        document_type: Type of document ('transcript', 'filing', or 'reasoning').
        document_id: Unique document identifier.
        content: Full text content to embed and store.
        ticker: Stock ticker symbol.
        metadata: Optional additional metadata.
    """
    deps = _deps(ctx)
    collection_name = _validate_document_type(document_type)

    full_metadata: dict[str, str] = {"ticker": ticker, "document_type": document_type}
    if metadata:
        full_metadata.update(metadata)

    deps.store.upsert(
        collection_name=collection_name,
        document_id=document_id,
        content=content,
        metadata=full_metadata,
    )

    response = StoreEmbeddingResponse(
        document_id=document_id,
        document_type=document_type,
        collection_name=collection_name,
    )
    return response.model_dump_json()


async def search_similar(
    ctx: Context,
    document_type: str,
    query: str,
    n_results: int = DEFAULT_N_RESULTS,
    ticker: str | None = None,
) -> str:
    """Search for documents similar to a query text.

    Args:
        document_type: Type of documents to search.
        query: Text to find similar documents for.
        n_results: Maximum number of results (default 5).
        ticker: Optional ticker filter.
    """
    deps = _deps(ctx)
    collection_name = _validate_document_type(document_type)

    where = {"ticker": ticker} if ticker is not None else None

    raw_results = deps.store.query(
        collection_name=collection_name,
        query_text=query,
        n_results=n_results,
        where=where,
    )

    results = [
        SimilarDocument(
            document_id=r["id"],
            content=r["content"],
            metadata=r["metadata"],
            distance=r["distance"],
        )
        for r in raw_results
    ]

    response = SearchSimilarResponse(
        query=query,
        document_type=document_type,
        results=results,
    )
    return response.model_dump_json()


async def get_document(ctx: Context, document_type: str, document_id: str) -> str:
    """Retrieve a specific document by its ID.

    Args:
        document_type: Type of document.
        document_id: The document ID to retrieve.
    """
    deps = _deps(ctx)
    collection_name = _validate_document_type(document_type)

    result = deps.store.get(collection_name=collection_name, document_id=document_id)

    if result is None:
        response = GetDocumentResponse(
            document_id=document_id,
            document_type=document_type,
            content=None,
            metadata=None,
            found=False,
        )
    else:
        response = GetDocumentResponse(
            document_id=document_id,
            document_type=document_type,
            content=result["content"],
            metadata=result["metadata"],
            found=True,
        )

    return response.model_dump_json()


# All tool functions in registration order.
TOOLS = [store_embedding, search_similar, get_document]


def create_server(name: str = "vector-store") -> FastMCP:
    """Create a new FastMCP server instance with all tools registered.

    Returns a fresh instance each time -- useful for testing where each
    test needs its own isolated server with independent lifespan state.
    """
    server = FastMCP(name, lifespan=lifespan)
    for tool_fn in TOOLS:
        server.tool()(tool_fn)
    return server


# Module-level server instance for production use.
mcp = create_server()
