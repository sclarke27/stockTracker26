"""Pydantic models for the vector store MCP server."""

from __future__ import annotations

from pydantic import BaseModel, Field


class StoreEmbeddingResponse(BaseModel):
    """Response returned after storing a document embedding.

    Confirms that a document was successfully embedded and stored in the
    appropriate ChromaDB collection.
    """

    document_id: str = Field(description="ID of the stored document")
    document_type: str = Field(description="Type of document (transcript, filing, reasoning)")
    collection_name: str = Field(description="ChromaDB collection the document was stored in")


class SimilarDocument(BaseModel):
    """A single document result from similarity search.

    Represents one matching document returned by a cosine similarity query,
    including its full content, metadata, and distance from the query vector.
    """

    document_id: str = Field(description="Unique document identifier")
    content: str = Field(description="Full text content of the document")
    metadata: dict[str, str] = Field(description="Document metadata key-value pairs")
    distance: float = Field(
        ge=0.0,
        description="Cosine distance from query (0.0 = identical, 2.0 = opposite)",
    )


class SearchSimilarResponse(BaseModel):
    """Response returned from a similarity search.

    Contains the original query context and a list of matching documents
    ranked by cosine similarity (closest first).
    """

    query: str = Field(description="The original query text")
    document_type: str = Field(description="Type of documents searched")
    results: list[SimilarDocument] = Field(description="Documents ranked by similarity")


class GetDocumentResponse(BaseModel):
    """Response returned when fetching a specific document by ID.

    When the document exists, ``found`` is True and ``content``/``metadata``
    are populated.  When it does not exist, ``found`` is False and both
    ``content`` and ``metadata`` are None.
    """

    document_id: str = Field(description="The requested document ID")
    document_type: str = Field(description="Type of document")
    content: str | None = Field(default=None, description="Full text content, or None if not found")
    metadata: dict[str, str] | None = Field(
        default=None, description="Document metadata, or None if not found"
    )
    found: bool = Field(description="Whether the document was found in the collection")
