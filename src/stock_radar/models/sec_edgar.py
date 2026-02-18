"""Pydantic models for SEC EDGAR MCP server inputs and outputs."""

from __future__ import annotations

from pydantic import BaseModel, Field

# === Response Models ===


class Filing(BaseModel):
    """Metadata for a single SEC filing."""

    accession_number: str = Field(description="SEC accession number (e.g. 0000320193-25-000001)")
    form_type: str = Field(description="SEC form type (e.g. 8-K, 10-K, 10-Q)")
    filing_date: str = Field(description="Filing date (YYYY-MM-DD)")
    primary_document: str = Field(description="Primary document filename")
    description: str = Field(description="Filing description")


class FilingsResponse(BaseModel):
    """Company filings list response."""

    ticker: str = Field(description="Stock ticker symbol")
    cik: str = Field(description="SEC Central Index Key (10-digit, zero-padded)")
    company_name: str = Field(description="Company name from SEC")
    filings: list[Filing] = Field(description="List of filings, most recent first")


class FilingTextResponse(BaseModel):
    """Full text content of a filing."""

    ticker: str = Field(description="Stock ticker symbol")
    accession_number: str = Field(description="SEC accession number")
    form_type: str = Field(description="SEC form type")
    filing_date: str = Field(description="Filing date (YYYY-MM-DD)")
    content: str = Field(description="Filing text content (HTML stripped)")
    truncated: bool = Field(description="True if content was truncated to max_length")


class InsiderTransaction(BaseModel):
    """Single insider transaction parsed from SEC Form 4 XML."""

    owner_name: str = Field(description="Name of the insider")
    owner_title: str = Field(description="Title/relationship (e.g. CEO, Director)")
    transaction_date: str = Field(description="Transaction date (YYYY-MM-DD)")
    transaction_code: str = Field(description="Transaction code (P=purchase, S=sale, A=grant)")
    shares: float = Field(description="Number of shares transacted")
    price_per_share: float | None = Field(description="Price per share, or null for grants")
    shares_owned_after: float = Field(description="Total shares owned after transaction")


class InsiderTransactionsResponse(BaseModel):
    """Aggregated insider transactions for a company."""

    ticker: str = Field(description="Stock ticker symbol")
    cik: str = Field(description="SEC Central Index Key")
    company_name: str = Field(description="Company name from SEC")
    transactions: list[InsiderTransaction] = Field(
        description="Insider transactions, most recent first"
    )


class FilingSearchHit(BaseModel):
    """Single result from EDGAR full-text search."""

    accession_number: str = Field(description="SEC accession number")
    form_type: str = Field(description="SEC form type")
    filing_date: str = Field(description="Filing date (YYYY-MM-DD)")
    company_name: str = Field(description="Company name")
    cik: str = Field(description="SEC Central Index Key")
    description: str = Field(description="Filing description or matched context")


class FilingSearchResponse(BaseModel):
    """EDGAR full-text search results."""

    query: str = Field(description="Original search query")
    total_hits: int = Field(description="Total number of matching filings")
    hits: list[FilingSearchHit] = Field(description="Search result entries")
