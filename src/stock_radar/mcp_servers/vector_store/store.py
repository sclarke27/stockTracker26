"""ChromaDB wrapper managing document collections for the vector store."""

from __future__ import annotations

from typing import Any

import chromadb
from loguru import logger

from stock_radar.mcp_servers.vector_store.config import (
    COLLECTION_NAMES,
    DISTANCE_METRIC,
    SERVER_NAME,
)

_log = logger.bind(server=SERVER_NAME)


class VectorStore:
    """ChromaDB wrapper managing document collections.

    Provides upsert, query, and get operations against named collections.
    Each collection uses cosine distance.  When no explicit embedding
    function is supplied the default ChromaDB all-MiniLM-L6-v2 model is
    used; tests may inject a lightweight alternative to avoid model
    downloads.

    Args:
        client: A ChromaDB ClientAPI instance (PersistentClient for
            production, EphemeralClient for testing).
        embedding_function: Optional embedding function passed to every
            collection.  When *None* ChromaDB uses its built-in default.
    """

    def __init__(
        self,
        client: chromadb.ClientAPI,
        embedding_function: chromadb.EmbeddingFunction[Any] | None = None,
    ) -> None:
        self._client = client
        self._embedding_function = embedding_function
        self._collections: dict[str, chromadb.Collection] = {}

    def initialize(self) -> None:
        """Create or retrieve all required collections.

        Safe to call multiple times (get_or_create_collection is idempotent).
        """
        kwargs: dict[str, Any] = {
            "metadata": {"hnsw:space": DISTANCE_METRIC},
        }
        if self._embedding_function is not None:
            kwargs["embedding_function"] = self._embedding_function

        for name in COLLECTION_NAMES:
            self._collections[name] = self._client.get_or_create_collection(
                name=name,
                **kwargs,
            )
        _log.info(
            "Vector store initialized with {n} collections",
            n=len(self._collections),
        )

    def _get_collection(self, collection_name: str) -> chromadb.Collection:
        """Look up a collection by name.

        Args:
            collection_name: Name of the ChromaDB collection.

        Returns:
            The ChromaDB Collection object.

        Raises:
            ValueError: If the collection name is not one of the initialized
                collections.
        """
        collection = self._collections.get(collection_name)
        if collection is None:
            raise ValueError(
                f"Unknown collection: {collection_name!r}. "
                f"Valid collections: {sorted(self._collections.keys())}"
            )
        return collection

    def upsert(
        self,
        collection_name: str,
        document_id: str,
        content: str,
        metadata: dict[str, str],
    ) -> None:
        """Add or update a document in a collection.

        Uses the configured embedding function to generate the vector.
        If a document with the same ID already exists, it is overwritten.

        Args:
            collection_name: Target collection name.
            document_id: Unique document identifier (deterministic).
            content: Full text content to embed and store.
            metadata: Key-value metadata pairs attached to the document.
        """
        collection = self._get_collection(collection_name)
        collection.upsert(
            ids=[document_id],
            documents=[content],
            metadatas=[metadata],
        )
        _log.debug(
            "Upserted document {doc_id} into {collection}",
            doc_id=document_id,
            collection=collection_name,
        )

    def query(
        self,
        collection_name: str,
        query_text: str,
        n_results: int,
        where: dict[str, str] | None = None,
    ) -> list[dict[str, Any]]:
        """Search for similar documents by text.

        Args:
            collection_name: Collection to search.
            query_text: Text to find similar documents for.
            n_results: Maximum number of results to return.
            where: Optional metadata filter (e.g. ``{"ticker": "AAPL"}``).

        Returns:
            List of dicts with keys: id, content, metadata, distance.
            Ordered by ascending distance (most similar first).
            Returns empty list if collection is empty.
        """
        collection = self._get_collection(collection_name)

        # ChromaDB raises if n_results > collection size.  Clamp to avoid.
        count = collection.count()
        if count == 0:
            return []
        effective_n = min(n_results, count)

        kwargs: dict[str, Any] = {
            "query_texts": [query_text],
            "n_results": effective_n,
            "include": ["documents", "metadatas", "distances"],
        }
        if where is not None:
            kwargs["where"] = where

        raw = collection.query(**kwargs)

        # ChromaDB returns parallel lists inside single-element outer lists.
        ids: list[str] = raw["ids"][0]
        documents: list[str] = raw["documents"][0]
        metadatas: list[dict[str, Any]] = raw["metadatas"][0]
        distances: list[float] = raw["distances"][0]

        results: list[dict[str, Any]] = []
        for i in range(len(ids)):
            results.append(
                {
                    "id": ids[i],
                    "content": documents[i],
                    "metadata": metadatas[i],
                    "distance": distances[i],
                }
            )
        return results

    def get(self, collection_name: str, document_id: str) -> dict[str, Any] | None:
        """Fetch a single document by its ID.

        Args:
            collection_name: Collection to look in.
            document_id: The document ID to retrieve.

        Returns:
            Dict with keys id, content, metadata if found, or *None* if
            not found.
        """
        collection = self._get_collection(collection_name)
        raw = collection.get(
            ids=[document_id],
            include=["documents", "metadatas"],
        )

        if not raw["ids"]:
            return None

        return {
            "id": raw["ids"][0],
            "content": raw["documents"][0],
            "metadata": raw["metadatas"][0],
        }

    def close(self) -> None:
        """Clean up resources.

        ChromaDB's PersistentClient does not require explicit close,
        but this method keeps pattern consistency with other stores.
        """
        self._collections.clear()
        _log.info("Vector store closed")
