"""Tests for SEC Filing Pattern Analyzer agent models."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from stock_radar.agents.sec_filing_analyzer.models import (
    FilingPattern,
    InsiderSummary,
    SecFilingAnalysis,
    SecFilingInput,
)


def _sample_input(**overrides) -> SecFilingInput:
    defaults = {
        "ticker": "TSLA",
        "quarter": 2,
        "year": 2024,
        "recent_filings": [
            {"form_type": "8-K", "filed_at": "2024-06-15", "description": "Material event"},
        ],
        "insider_transactions": [
            {
                "insider_name": "Elon Musk",
                "transaction_type": "S",
                "shares": 100000,
                "date": "2024-06-10",
            }
        ],
        "filing_count": 1,
        "insider_transaction_count": 1,
        "lookback_days": 90,
    }
    defaults.update(overrides)
    return SecFilingInput(**defaults)


def _sample_analysis(**overrides) -> SecFilingAnalysis:
    defaults = {
        "patterns_detected": [
            FilingPattern(
                pattern_type="insider_selling_cluster",
                description="CEO sold large block of shares.",
                significance="HIGH",
                filing_dates=["2024-06-10"],
            )
        ],
        "insider_summary": InsiderSummary(
            net_shares_acquired=-100000.0,
            total_transactions=1,
            unique_insiders=1,
            largest_transaction_shares=100000.0,
        ),
        "insider_sentiment": "BEARISH",
        "direction": "BEARISH",
        "confidence": 0.72,
        "risk_flags": ["Large insider sale by CEO"],
        "key_findings": ["CEO sold 100K shares"],
        "horizon_days": 15,
        "reasoning_summary": "Significant insider selling is a bearish signal.",
    }
    defaults.update(overrides)
    return SecFilingAnalysis(**defaults)


class TestSecFilingInput:
    """Tests for SecFilingInput model."""

    def test_valid_construction(self) -> None:
        inp = _sample_input()
        assert inp.ticker == "TSLA"
        assert inp.filing_count == 1
        assert inp.insider_transaction_count == 1
        assert inp.lookback_days == 90

    def test_empty_filings_valid(self) -> None:
        inp = _sample_input(recent_filings=[], filing_count=0)
        assert inp.recent_filings == []

    def test_empty_insider_transactions_valid(self) -> None:
        inp = _sample_input(insider_transactions=[], insider_transaction_count=0)
        assert inp.insider_transactions == []

    def test_missing_ticker_raises(self) -> None:
        with pytest.raises(ValidationError):
            SecFilingInput(
                quarter=1,
                year=2024,
                recent_filings=[],
                insider_transactions=[],
                filing_count=0,
                insider_transaction_count=0,
                lookback_days=90,
            )  # type: ignore[call-arg]

    def test_multiple_filings(self) -> None:
        filings = [
            {"form_type": "8-K", "filed_at": "2024-06-01", "description": "Event 1"},
            {"form_type": "10-Q", "filed_at": "2024-05-15", "description": "Quarterly report"},
        ]
        inp = _sample_input(recent_filings=filings, filing_count=2)
        assert len(inp.recent_filings) == 2


class TestFilingPattern:
    """Tests for FilingPattern model."""

    def test_valid_construction(self) -> None:
        pattern = FilingPattern(
            pattern_type="insider_buying_cluster",
            description="Multiple insiders bought shares.",
            significance="HIGH",
            filing_dates=["2024-06-01", "2024-06-02"],
        )
        assert pattern.pattern_type == "insider_buying_cluster"
        assert pattern.significance == "HIGH"

    def test_all_pattern_types_accepted(self) -> None:
        valid_types = [
            "insider_buying_cluster",
            "insider_selling_cluster",
            "unusual_8k_frequency",
            "s1_amendment",
            "late_filing",
            "executive_departure",
        ]
        for pt in valid_types:
            pattern = FilingPattern(
                pattern_type=pt,  # type: ignore[arg-type]
                description="test",
                significance="LOW",
                filing_dates=[],
            )
            assert pattern.pattern_type == pt

    def test_invalid_pattern_type_raises(self) -> None:
        with pytest.raises(ValidationError):
            FilingPattern(
                pattern_type="unknown_pattern",  # type: ignore[arg-type]
                description="test",
                significance="LOW",
                filing_dates=[],
            )

    def test_all_significance_levels_accepted(self) -> None:
        for sig in ("HIGH", "MEDIUM", "LOW"):
            pattern = FilingPattern(
                pattern_type="late_filing",
                description="test",
                significance=sig,  # type: ignore[arg-type]
                filing_dates=[],
            )
            assert pattern.significance == sig


class TestInsiderSummary:
    """Tests for InsiderSummary model."""

    def test_net_buying_positive(self) -> None:
        summary = InsiderSummary(
            net_shares_acquired=50000.0,
            total_transactions=3,
            unique_insiders=2,
            largest_transaction_shares=30000.0,
        )
        assert summary.net_shares_acquired > 0

    def test_net_selling_negative(self) -> None:
        summary = InsiderSummary(
            net_shares_acquired=-200000.0,
            total_transactions=5,
            unique_insiders=3,
            largest_transaction_shares=100000.0,
        )
        assert summary.net_shares_acquired < 0


class TestSecFilingAnalysis:
    """Tests for SecFilingAnalysis model (LLM structured output)."""

    def test_valid_construction(self) -> None:
        analysis = _sample_analysis()
        assert analysis.direction == "BEARISH"
        assert analysis.confidence == 0.72
        assert len(analysis.patterns_detected) == 1

    def test_direction_literal_enforced(self) -> None:
        with pytest.raises(ValidationError):
            _sample_analysis(direction="SIDEWAYS")  # type: ignore[arg-type]

    def test_all_directions_accepted(self) -> None:
        for direction in ("BULLISH", "BEARISH", "NEUTRAL"):
            analysis = _sample_analysis(direction=direction)
            assert analysis.direction == direction

    def test_insider_sentiment_literal_enforced(self) -> None:
        with pytest.raises(ValidationError):
            _sample_analysis(insider_sentiment="MIXED")  # type: ignore[arg-type]

    def test_confidence_bounds(self) -> None:
        with pytest.raises(ValidationError):
            _sample_analysis(confidence=1.5)
        with pytest.raises(ValidationError):
            _sample_analysis(confidence=-0.1)

    def test_empty_patterns_valid(self) -> None:
        analysis = _sample_analysis(patterns_detected=[], direction="NEUTRAL", confidence=0.5)
        assert analysis.patterns_detected == []

    def test_risk_flags_can_be_empty(self) -> None:
        analysis = _sample_analysis(risk_flags=[])
        assert analysis.risk_flags == []
