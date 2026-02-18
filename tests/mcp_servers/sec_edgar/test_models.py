"""Tests for SEC EDGAR Pydantic models."""

from __future__ import annotations

import json

import pytest
from pydantic import ValidationError

from stock_radar.models.sec_edgar import (
    Filing,
    FilingSearchHit,
    FilingSearchResponse,
    FilingsResponse,
    FilingTextResponse,
    InsiderTransaction,
    InsiderTransactionsResponse,
)


class TestFiling:
    """Tests for the Filing model."""

    def test_construction(self) -> None:
        filing = Filing(
            accession_number="0000320193-25-000001",
            form_type="8-K",
            filing_date="2025-01-15",
            primary_document="doc1.htm",
            description="Current Report",
        )
        assert filing.accession_number == "0000320193-25-000001"
        assert filing.form_type == "8-K"

    def test_missing_required_field_raises(self) -> None:
        with pytest.raises(ValidationError):
            Filing(
                accession_number="0000320193-25-000001",
                form_type="8-K",
                # missing filing_date, primary_document, description
            )

    def test_serialization_roundtrip(self) -> None:
        filing = Filing(
            accession_number="0000320193-25-000001",
            form_type="10-K",
            filing_date="2024-12-15",
            primary_document="annual.htm",
            description="Annual Report",
        )
        data = json.loads(filing.model_dump_json())
        restored = Filing(**data)
        assert restored == filing


class TestFilingsResponse:
    """Tests for the FilingsResponse model."""

    def test_with_filings(self) -> None:
        resp = FilingsResponse(
            ticker="AAPL",
            cik="0000320193",
            company_name="Apple Inc.",
            filings=[
                Filing(
                    accession_number="0000320193-25-000001",
                    form_type="8-K",
                    filing_date="2025-01-15",
                    primary_document="doc1.htm",
                    description="Current Report",
                ),
            ],
        )
        assert resp.ticker == "AAPL"
        assert len(resp.filings) == 1

    def test_empty_filings(self) -> None:
        resp = FilingsResponse(
            ticker="XYZ",
            cik="0000000000",
            company_name="Unknown Corp",
            filings=[],
        )
        assert resp.filings == []

    def test_serialization_roundtrip(self) -> None:
        resp = FilingsResponse(
            ticker="MSFT",
            cik="0000789019",
            company_name="MICROSOFT CORP",
            filings=[
                Filing(
                    accession_number="0000789019-25-000001",
                    form_type="10-Q",
                    filing_date="2025-01-10",
                    primary_document="quarterly.htm",
                    description="Quarterly Report",
                ),
            ],
        )
        data = json.loads(resp.model_dump_json())
        restored = FilingsResponse(**data)
        assert restored == resp


class TestFilingTextResponse:
    """Tests for the FilingTextResponse model."""

    def test_not_truncated(self) -> None:
        resp = FilingTextResponse(
            ticker="AAPL",
            accession_number="0000320193-25-000001",
            form_type="8-K",
            filing_date="2025-01-15",
            content="Full filing text here.",
            truncated=False,
        )
        assert resp.truncated is False
        assert resp.content == "Full filing text here."

    def test_truncated(self) -> None:
        resp = FilingTextResponse(
            ticker="AAPL",
            accession_number="0000320193-24-000003",
            form_type="10-K",
            filing_date="2024-12-15",
            content="Truncated content...",
            truncated=True,
        )
        assert resp.truncated is True

    def test_missing_required_field_raises(self) -> None:
        with pytest.raises(ValidationError):
            FilingTextResponse(
                ticker="AAPL",
                accession_number="0000320193-25-000001",
                # missing form_type, filing_date, content, truncated
            )


class TestInsiderTransaction:
    """Tests for the InsiderTransaction model."""

    def test_construction_with_price(self) -> None:
        txn = InsiderTransaction(
            owner_name="Cook Timothy D",
            owner_title="Chief Executive Officer",
            transaction_date="2025-01-10",
            transaction_code="S",
            shares=50000.0,
            price_per_share=185.50,
            shares_owned_after=3000000.0,
        )
        assert txn.transaction_code == "S"
        assert txn.price_per_share == 185.50

    def test_price_per_share_none(self) -> None:
        txn = InsiderTransaction(
            owner_name="Doe Jane",
            owner_title="Director",
            transaction_date="2025-01-05",
            transaction_code="A",
            shares=10000.0,
            price_per_share=None,
            shares_owned_after=50000.0,
        )
        assert txn.price_per_share is None

    def test_serialization_roundtrip(self) -> None:
        txn = InsiderTransaction(
            owner_name="Smith John",
            owner_title="CFO",
            transaction_date="2025-01-08",
            transaction_code="P",
            shares=1000.0,
            price_per_share=150.00,
            shares_owned_after=25000.0,
        )
        data = json.loads(txn.model_dump_json())
        restored = InsiderTransaction(**data)
        assert restored == txn


class TestInsiderTransactionsResponse:
    """Tests for the InsiderTransactionsResponse model."""

    def test_with_transactions(self) -> None:
        resp = InsiderTransactionsResponse(
            ticker="AAPL",
            cik="0000320193",
            company_name="Apple Inc.",
            transactions=[
                InsiderTransaction(
                    owner_name="Cook Timothy D",
                    owner_title="CEO",
                    transaction_date="2025-01-10",
                    transaction_code="S",
                    shares=50000.0,
                    price_per_share=185.50,
                    shares_owned_after=3000000.0,
                ),
            ],
        )
        assert len(resp.transactions) == 1

    def test_empty_transactions(self) -> None:
        resp = InsiderTransactionsResponse(
            ticker="XYZ",
            cik="0000000000",
            company_name="Unknown Corp",
            transactions=[],
        )
        assert resp.transactions == []


class TestFilingSearchHit:
    """Tests for the FilingSearchHit model."""

    def test_construction(self) -> None:
        hit = FilingSearchHit(
            accession_number="0000320193-25-000001",
            form_type="8-K",
            filing_date="2025-01-15",
            company_name="Apple Inc.",
            cik="0000320193",
            description="Current Report",
        )
        assert hit.company_name == "Apple Inc."


class TestFilingSearchResponse:
    """Tests for the FilingSearchResponse model."""

    def test_with_hits(self) -> None:
        resp = FilingSearchResponse(
            query="artificial intelligence",
            total_hits=42,
            hits=[
                FilingSearchHit(
                    accession_number="0000320193-25-000001",
                    form_type="8-K",
                    filing_date="2025-01-15",
                    company_name="Apple Inc.",
                    cik="0000320193",
                    description="Current Report",
                ),
            ],
        )
        assert resp.total_hits == 42
        assert len(resp.hits) == 1

    def test_zero_hits(self) -> None:
        resp = FilingSearchResponse(
            query="xyznonexistent",
            total_hits=0,
            hits=[],
        )
        assert resp.hits == []

    def test_serialization_roundtrip(self) -> None:
        resp = FilingSearchResponse(
            query="revenue growth",
            total_hits=5,
            hits=[
                FilingSearchHit(
                    accession_number="0000789019-25-000001",
                    form_type="10-Q",
                    filing_date="2025-01-10",
                    company_name="MICROSOFT CORP",
                    cik="0000789019",
                    description="Quarterly Report",
                ),
            ],
        )
        data = json.loads(resp.model_dump_json())
        restored = FilingSearchResponse(**data)
        assert restored == resp
