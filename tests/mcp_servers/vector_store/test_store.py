"""Tests for the vector store ChromaDB wrapper."""

from __future__ import annotations

import chromadb
import chromadb.api
import pytest

from stock_radar.mcp_servers.vector_store.store import VectorStore


class _FixedEmbeddingFunction(chromadb.EmbeddingFunction[list[str]]):
    """Deterministic embedding function for tests.

    Produces a simple bag-of-characters vector so that semantically
    distinct documents still yield different embeddings without
    requiring a neural model download.
    """

    _DIMENSION = 64

    def __init__(self) -> None:
        pass

    def name(self) -> str:
        """Return function name for ChromaDB registration."""
        return "fixed_test_embedding"

    def get_config(self) -> dict:
        """Return configuration for ChromaDB serialization."""
        return {}

    def __call__(self, input: list[str]) -> list[list[float]]:  # noqa: A002
        results: list[list[float]] = []
        for text in input:
            vec = [0.0] * self._DIMENSION
            for char in text.lower():
                vec[ord(char) % self._DIMENSION] += 1.0
            # L2-normalise so cosine distance is meaningful.
            magnitude = sum(v * v for v in vec) ** 0.5
            if magnitude > 0:
                vec = [v / magnitude for v in vec]
            results.append(vec)
        return results


@pytest.fixture()
def store() -> VectorStore:
    """Provide an initialized VectorStore backed by an in-memory EphemeralClient."""
    client = chromadb.EphemeralClient()
    # Purge any leftover collections from previous tests sharing the
    # same in-process EphemeralClient backend.
    for col in client.list_collections():
        client.delete_collection(col.name)
    s = VectorStore(client=client, embedding_function=_FixedEmbeddingFunction())
    s.initialize()
    return s


# ---------------------------------------------------------------------------
# Initialization
# ---------------------------------------------------------------------------


class TestStoreInitialize:
    """Tests for collection initialization."""

    def test_creates_all_collections(self, store: VectorStore) -> None:
        """After initialize(), all 3 expected collections exist."""
        assert set(store._collections.keys()) == {"transcripts", "filings", "reasoning"}

    def test_idempotent(self, store: VectorStore) -> None:
        """Calling initialize() twice does not raise or duplicate collections."""
        store.initialize()
        assert set(store._collections.keys()) == {"transcripts", "filings", "reasoning"}


# ---------------------------------------------------------------------------
# Upsert
# ---------------------------------------------------------------------------


class TestStoreUpsert:
    """Tests for document upsert operations."""

    def test_upsert_and_get(self, store: VectorStore) -> None:
        """Upserted document can be retrieved with matching content and metadata."""
        store.upsert(
            collection_name="transcripts",
            document_id="AAPL_2024_Q4",
            content="Apple reported strong revenue growth in Q4 2024.",
            metadata={"ticker": "AAPL", "quarter": "Q4", "year": "2024"},
        )
        result = store.get(collection_name="transcripts", document_id="AAPL_2024_Q4")
        assert result is not None
        assert result["id"] == "AAPL_2024_Q4"
        assert result["content"] == "Apple reported strong revenue growth in Q4 2024."
        assert result["metadata"]["ticker"] == "AAPL"
        assert result["metadata"]["quarter"] == "Q4"
        assert result["metadata"]["year"] == "2024"

    def test_upsert_is_idempotent(self, store: VectorStore) -> None:
        """Upserting same ID twice overwrites with the latest content."""
        store.upsert(
            collection_name="filings",
            document_id="MSFT_10K_2024",
            content="Original filing content.",
            metadata={"ticker": "MSFT", "form_type": "10-K"},
        )
        store.upsert(
            collection_name="filings",
            document_id="MSFT_10K_2024",
            content="Updated filing content with amendments.",
            metadata={"ticker": "MSFT", "form_type": "10-K/A"},
        )
        result = store.get(collection_name="filings", document_id="MSFT_10K_2024")
        assert result is not None
        assert result["content"] == "Updated filing content with amendments."
        assert result["metadata"]["form_type"] == "10-K/A"

    def test_upsert_invalid_collection_raises(self, store: VectorStore) -> None:
        """Upserting to an unknown collection raises ValueError."""
        with pytest.raises(ValueError, match="Unknown collection"):
            store.upsert(
                collection_name="bogus",
                document_id="doc1",
                content="Some content.",
                metadata={"ticker": "AAPL"},
            )


# ---------------------------------------------------------------------------
# Query
# ---------------------------------------------------------------------------


class TestStoreQuery:
    """Tests for similarity search operations."""

    def test_query_returns_similar_documents(self, store: VectorStore) -> None:
        """Query returns documents ranked by similarity; best match is first."""
        store.upsert(
            collection_name="transcripts",
            document_id="earnings_doc",
            content="Apple earnings revenue growth Q4 results strong performance",
            metadata={"ticker": "AAPL", "type": "earnings"},
        )
        store.upsert(
            collection_name="transcripts",
            document_id="sec_doc",
            content="SEC filing annual report 10-K regulatory compliance disclosure",
            metadata={"ticker": "AAPL", "type": "filing"},
        )
        store.upsert(
            collection_name="transcripts",
            document_id="weather_doc",
            content="Weather forecast for tomorrow sunny with clouds and wind",
            metadata={"ticker": "N/A", "type": "weather"},
        )
        results = store.query(
            collection_name="transcripts",
            query_text="Apple quarterly earnings results",
            n_results=3,
        )
        assert len(results) > 0
        assert results[0]["id"] == "earnings_doc"
        # Verify result structure.
        assert "content" in results[0]
        assert "metadata" in results[0]
        assert "distance" in results[0]

    def test_query_empty_collection(self, store: VectorStore) -> None:
        """Querying an empty collection returns an empty list."""
        results = store.query(
            collection_name="transcripts",
            query_text="anything at all",
            n_results=5,
        )
        assert results == []

    def test_query_with_metadata_filter(self, store: VectorStore) -> None:
        """Metadata where-filter limits results to matching documents only."""
        store.upsert(
            collection_name="transcripts",
            document_id="aapl_earnings",
            content="Apple reported strong Q4 revenue growth.",
            metadata={"ticker": "AAPL"},
        )
        store.upsert(
            collection_name="transcripts",
            document_id="msft_earnings",
            content="Microsoft reported strong Q4 revenue growth.",
            metadata={"ticker": "MSFT"},
        )
        results = store.query(
            collection_name="transcripts",
            query_text="quarterly revenue growth",
            n_results=10,
            where={"ticker": "AAPL"},
        )
        assert len(results) == 1
        assert results[0]["id"] == "aapl_earnings"
        assert results[0]["metadata"]["ticker"] == "AAPL"

    def test_query_n_results_limit(self, store: VectorStore) -> None:
        """n_results caps the number of returned documents."""
        for i in range(5):
            store.upsert(
                collection_name="reasoning",
                document_id=f"doc_{i}",
                content=f"Reasoning document number {i} about stock analysis.",
                metadata={"ticker": "AAPL", "index": str(i)},
            )
        results = store.query(
            collection_name="reasoning",
            query_text="stock analysis reasoning",
            n_results=2,
        )
        assert len(results) == 2

    def test_query_n_results_exceeds_count(self, store: VectorStore) -> None:
        """Requesting more results than exist returns all without error."""
        store.upsert(
            collection_name="filings",
            document_id="doc_a",
            content="First filing document about Tesla.",
            metadata={"ticker": "TSLA"},
        )
        store.upsert(
            collection_name="filings",
            document_id="doc_b",
            content="Second filing document about Tesla.",
            metadata={"ticker": "TSLA"},
        )
        results = store.query(
            collection_name="filings",
            query_text="Tesla filing",
            n_results=10,
        )
        assert len(results) == 2

    def test_query_invalid_collection_raises(self, store: VectorStore) -> None:
        """Querying an unknown collection raises ValueError."""
        with pytest.raises(ValueError, match="Unknown collection"):
            store.query(
                collection_name="bogus",
                query_text="anything",
                n_results=5,
            )


# ---------------------------------------------------------------------------
# Get
# ---------------------------------------------------------------------------


class TestStoreGet:
    """Tests for direct document retrieval by ID."""

    def test_get_existing_document(self, store: VectorStore) -> None:
        """Getting an existing document returns its content and metadata."""
        store.upsert(
            collection_name="reasoning",
            document_id="reason_001",
            content="Bullish signal detected based on earnings language.",
            metadata={"ticker": "NVDA", "agent": "earnings_linguist"},
        )
        result = store.get(collection_name="reasoning", document_id="reason_001")
        assert result is not None
        assert result["id"] == "reason_001"
        assert result["content"] == "Bullish signal detected based on earnings language."
        assert result["metadata"]["ticker"] == "NVDA"
        assert result["metadata"]["agent"] == "earnings_linguist"

    def test_get_nonexistent_document(self, store: VectorStore) -> None:
        """Getting a non-existent ID returns None."""
        result = store.get(collection_name="transcripts", document_id="does_not_exist")
        assert result is None

    def test_get_invalid_collection_raises(self, store: VectorStore) -> None:
        """Getting from an unknown collection raises ValueError."""
        with pytest.raises(ValueError, match="Unknown collection"):
            store.get(collection_name="bogus", document_id="doc1")
