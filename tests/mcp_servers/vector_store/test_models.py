"""Tests for vector store Pydantic models."""

from __future__ import annotations

import json

import pytest
from pydantic import ValidationError

from stock_radar.models.vector_store import (
    GetDocumentResponse,
    SearchSimilarResponse,
    SimilarDocument,
    StoreEmbeddingResponse,
)


class TestStoreEmbeddingResponse:
    """Tests for the StoreEmbeddingResponse model."""

    def test_construction(self) -> None:
        resp = StoreEmbeddingResponse(
            document_id="AAPL_2024_Q4",
            document_type="transcript",
            collection_name="transcripts",
        )
        assert resp.document_id == "AAPL_2024_Q4"
        assert resp.document_type == "transcript"
        assert resp.collection_name == "transcripts"

    def test_serialization_roundtrip(self) -> None:
        resp = StoreEmbeddingResponse(
            document_id="MSFT_2025_Q1",
            document_type="filing",
            collection_name="filings",
        )
        data = json.loads(resp.model_dump_json())
        restored = StoreEmbeddingResponse(**data)
        assert restored == resp


class TestSimilarDocument:
    """Tests for the SimilarDocument model."""

    def test_construction(self) -> None:
        doc = SimilarDocument(
            document_id="AAPL_2024_Q4",
            content="Apple reported strong revenue growth in Q4...",
            metadata={"ticker": "AAPL"},
            distance=0.42,
        )
        assert doc.document_id == "AAPL_2024_Q4"
        assert doc.content == "Apple reported strong revenue growth in Q4..."
        assert doc.metadata == {"ticker": "AAPL"}
        assert doc.distance == 0.42

    def test_distance_lower_bound(self) -> None:
        with pytest.raises(ValidationError):
            SimilarDocument(
                document_id="AAPL_2024_Q4",
                content="Some content.",
                metadata={"ticker": "AAPL"},
                distance=-0.1,
            )

    def test_serialization_roundtrip(self) -> None:
        doc = SimilarDocument(
            document_id="GOOG_2025_Q1",
            content="Google Cloud revenue exceeded expectations.",
            metadata={"ticker": "GOOG", "quarter": "Q1"},
            distance=0.15,
        )
        data = json.loads(doc.model_dump_json())
        restored = SimilarDocument(**data)
        assert restored == doc


class TestSearchSimilarResponse:
    """Tests for the SearchSimilarResponse model."""

    def test_construction_with_results(self) -> None:
        result = SimilarDocument(
            document_id="AAPL_2024_Q4",
            content="Apple Q4 earnings transcript excerpt.",
            metadata={"ticker": "AAPL", "quarter": "Q4"},
            distance=0.25,
        )
        resp = SearchSimilarResponse(
            query="earnings",
            document_type="transcript",
            results=[result],
        )
        assert resp.query == "earnings"
        assert resp.document_type == "transcript"
        assert len(resp.results) == 1
        assert resp.results[0].document_id == "AAPL_2024_Q4"

    def test_empty_results(self) -> None:
        resp = SearchSimilarResponse(
            query="nonexistent topic",
            document_type="filing",
            results=[],
        )
        assert resp.results == []

    def test_serialization_roundtrip(self) -> None:
        result = SimilarDocument(
            document_id="NVDA_2025_Q2",
            content="NVIDIA reported record data center revenue.",
            metadata={"ticker": "NVDA", "quarter": "Q2"},
            distance=0.33,
        )
        resp = SearchSimilarResponse(
            query="data center growth",
            document_type="transcript",
            results=[result],
        )
        data = json.loads(resp.model_dump_json())
        restored = SearchSimilarResponse(**data)
        assert restored == resp


class TestGetDocumentResponse:
    """Tests for the GetDocumentResponse model."""

    def test_construction_found(self) -> None:
        resp = GetDocumentResponse(
            document_id="AAPL_2024_Q4",
            document_type="transcript",
            content="Full transcript content for Apple Q4 2024.",
            metadata={"ticker": "AAPL", "quarter": "Q4", "year": "2024"},
            found=True,
        )
        assert resp.document_id == "AAPL_2024_Q4"
        assert resp.document_type == "transcript"
        assert resp.content == "Full transcript content for Apple Q4 2024."
        assert resp.metadata == {"ticker": "AAPL", "quarter": "Q4", "year": "2024"}
        assert resp.found is True

    def test_construction_not_found(self) -> None:
        resp = GetDocumentResponse(
            document_id="FAKE_9999_Q9",
            document_type="transcript",
            found=False,
        )
        assert resp.document_id == "FAKE_9999_Q9"
        assert resp.document_type == "transcript"
        assert resp.content is None
        assert resp.metadata is None
        assert resp.found is False

    def test_serialization_roundtrip(self) -> None:
        resp = GetDocumentResponse(
            document_id="TSLA_2025_Q1",
            document_type="filing",
            content="Tesla 10-K filing content.",
            metadata={"ticker": "TSLA", "form_type": "10-K"},
            found=True,
        )
        data = json.loads(resp.model_dump_json())
        restored = GetDocumentResponse(**data)
        assert restored == resp
