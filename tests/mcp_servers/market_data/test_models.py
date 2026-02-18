"""Tests for market data Pydantic models."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from stock_radar.models.market_data import (
    CompanyInfoResponse,
    EarningsTranscriptResponse,
    OHLCVBar,
    PriceHistoryResponse,
    QuoteResponse,
    TickerMatch,
    TickerSearchResponse,
)


class TestOHLCVBar:
    """Tests for the OHLCVBar model."""

    def test_valid_bar(self) -> None:
        bar = OHLCVBar(
            date="2025-01-15",
            open=150.0,
            high=155.0,
            low=149.0,
            close=153.5,
            volume=1_000_000,
        )
        assert bar.date == "2025-01-15"
        assert bar.open == 150.0
        assert bar.high == 155.0
        assert bar.low == 149.0
        assert bar.close == 153.5
        assert bar.volume == 1_000_000

    def test_missing_required_field(self) -> None:
        with pytest.raises(ValidationError):
            OHLCVBar(
                date="2025-01-15",
                open=150.0,
                high=155.0,
                low=149.0,
                # missing close
                volume=1_000_000,
            )

    def test_serialization_roundtrip(self) -> None:
        bar = OHLCVBar(
            date="2025-01-15",
            open=150.0,
            high=155.0,
            low=149.0,
            close=153.5,
            volume=1_000_000,
        )
        json_str = bar.model_dump_json()
        restored = OHLCVBar.model_validate_json(json_str)
        assert restored == bar


class TestPriceHistoryResponse:
    """Tests for the PriceHistoryResponse model."""

    def test_valid_response(self) -> None:
        bars = [
            OHLCVBar(
                date="2025-01-15",
                open=150.0,
                high=155.0,
                low=149.0,
                close=153.5,
                volume=1_000_000,
            ),
            OHLCVBar(
                date="2025-01-14",
                open=148.0,
                high=151.0,
                low=147.0,
                close=150.0,
                volume=900_000,
            ),
        ]
        response = PriceHistoryResponse(
            ticker="AAPL",
            bars=bars,
            last_refreshed="2025-01-15",
        )
        assert response.ticker == "AAPL"
        assert len(response.bars) == 2
        assert response.last_refreshed == "2025-01-15"

    def test_empty_bars_list(self) -> None:
        response = PriceHistoryResponse(
            ticker="AAPL",
            bars=[],
            last_refreshed="2025-01-15",
        )
        assert response.bars == []

    def test_serialization_roundtrip(self) -> None:
        response = PriceHistoryResponse(
            ticker="AAPL",
            bars=[
                OHLCVBar(
                    date="2025-01-15",
                    open=150.0,
                    high=155.0,
                    low=149.0,
                    close=153.5,
                    volume=1_000_000,
                ),
            ],
            last_refreshed="2025-01-15",
        )
        json_str = response.model_dump_json()
        restored = PriceHistoryResponse.model_validate_json(json_str)
        assert restored == response


class TestQuoteResponse:
    """Tests for the QuoteResponse model."""

    def test_valid_quote(self) -> None:
        quote = QuoteResponse(
            ticker="MSFT",
            price=420.50,
            change=5.25,
            change_percent="1.26%",
            volume=25_000_000,
            latest_trading_day="2025-01-15",
            previous_close=415.25,
            open=416.00,
            high=421.00,
            low=414.50,
        )
        assert quote.ticker == "MSFT"
        assert quote.price == 420.50
        assert quote.change_percent == "1.26%"

    def test_missing_required_field(self) -> None:
        with pytest.raises(ValidationError):
            QuoteResponse(
                ticker="MSFT",
                price=420.50,
                # missing other required fields
            )


class TestCompanyInfoResponse:
    """Tests for the CompanyInfoResponse model."""

    def test_valid_company_info(self) -> None:
        info = CompanyInfoResponse(
            ticker="AAPL",
            name="Apple Inc",
            description="Apple designs consumer electronics.",
            sector="Technology",
            industry="Consumer Electronics",
            market_cap="3000000000000",
            pe_ratio="28.5",
            eps="6.42",
            dividend_yield="0.55",
            fifty_two_week_high="199.62",
            fifty_two_week_low="164.08",
        )
        assert info.ticker == "AAPL"
        assert info.name == "Apple Inc"
        assert info.sector == "Technology"

    def test_serialization_roundtrip(self) -> None:
        info = CompanyInfoResponse(
            ticker="AAPL",
            name="Apple Inc",
            description="Apple designs consumer electronics.",
            sector="Technology",
            industry="Consumer Electronics",
            market_cap="3000000000000",
            pe_ratio="28.5",
            eps="6.42",
            dividend_yield="0.55",
            fifty_two_week_high="199.62",
            fifty_two_week_low="164.08",
        )
        json_str = info.model_dump_json()
        restored = CompanyInfoResponse.model_validate_json(json_str)
        assert restored == info


class TestTickerSearch:
    """Tests for TickerMatch and TickerSearchResponse models."""

    def test_valid_match(self) -> None:
        match = TickerMatch(
            symbol="AAPL",
            name="Apple Inc",
            type="Equity",
            region="United States",
            currency="USD",
        )
        assert match.symbol == "AAPL"
        assert match.currency == "USD"

    def test_search_response_with_matches(self) -> None:
        matches = [
            TickerMatch(
                symbol="AAPL",
                name="Apple Inc",
                type="Equity",
                region="United States",
                currency="USD",
            ),
            TickerMatch(
                symbol="APLE",
                name="Apple Hospitality REIT",
                type="Equity",
                region="United States",
                currency="USD",
            ),
        ]
        response = TickerSearchResponse(matches=matches)
        assert len(response.matches) == 2
        assert response.matches[0].symbol == "AAPL"

    def test_empty_search_results(self) -> None:
        response = TickerSearchResponse(matches=[])
        assert response.matches == []


class TestEarningsTranscriptResponse:
    """Tests for the EarningsTranscriptResponse model."""

    def test_valid_transcript(self) -> None:
        transcript = EarningsTranscriptResponse(
            ticker="AAPL",
            quarter=4,
            year=2024,
            date="2024-10-31",
            content="Good afternoon, everyone. Welcome to Apple's...",
        )
        assert transcript.ticker == "AAPL"
        assert transcript.quarter == 4
        assert transcript.year == 2024
        assert "Welcome to Apple" in transcript.content

    def test_missing_required_field(self) -> None:
        with pytest.raises(ValidationError):
            EarningsTranscriptResponse(
                ticker="AAPL",
                quarter=4,
                # missing year, date, content
            )

    def test_serialization_roundtrip(self) -> None:
        transcript = EarningsTranscriptResponse(
            ticker="AAPL",
            quarter=4,
            year=2024,
            date="2024-10-31",
            content="Transcript text here.",
        )
        json_str = transcript.model_dump_json()
        restored = EarningsTranscriptResponse.model_validate_json(json_str)
        assert restored == transcript
