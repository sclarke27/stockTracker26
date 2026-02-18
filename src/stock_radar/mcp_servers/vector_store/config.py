"""Constants for the vector store MCP server."""

from __future__ import annotations

# Server identity for logging.
SERVER_NAME = "vector-store-mcp"

# Default number of results for similarity search.
DEFAULT_N_RESULTS = 5

# Mapping from tool-level document type to ChromaDB collection name.
DOCUMENT_TYPE_TO_COLLECTION: dict[str, str] = {
    "transcript": "transcripts",
    "filing": "filings",
    "reasoning": "reasoning",
}

# Valid document types (for O(1) validation).
VALID_DOCUMENT_TYPES = frozenset(DOCUMENT_TYPE_TO_COLLECTION.keys())

# Collection names (for initialization).
COLLECTION_NAMES = frozenset(DOCUMENT_TYPE_TO_COLLECTION.values())

# ChromaDB distance metric applied to all collections.
DISTANCE_METRIC = "cosine"
