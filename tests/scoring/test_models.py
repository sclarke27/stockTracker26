"""Tests for scoring loop Pydantic models."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from stock_radar.scoring.models import ScoringOutcome, ScoringResult


class TestScoringOutcome:
    """Tests for ScoringOutcome model."""

    def test_defaults(self) -> None:
        """Default values are correct for optional fields."""
        outcome = ScoringOutcome(prediction_id="abc-123", ticker="AAPL")
        assert outcome.status is None
        assert outcome.return_pct is None
        assert outcome.skipped is False
        assert outcome.skip_reason is None

    def test_scored_outcome(self) -> None:
        """All fields accept valid values for a scored prediction."""
        outcome = ScoringOutcome(
            prediction_id="abc-123",
            ticker="AAPL",
            status="CORRECT",
            return_pct=3.45,
        )
        assert outcome.status == "CORRECT"
        assert outcome.return_pct == 3.45
        assert outcome.skipped is False

    def test_skipped_outcome(self) -> None:
        """Skipped outcome has reason populated."""
        outcome = ScoringOutcome(
            prediction_id="abc-123",
            ticker="AAPL",
            skipped=True,
            skip_reason="price_data_unavailable",
        )
        assert outcome.skipped is True
        assert outcome.skip_reason == "price_data_unavailable"

    def test_negative_return_pct(self) -> None:
        """Negative return_pct is valid."""
        outcome = ScoringOutcome(
            prediction_id="abc-123",
            ticker="AAPL",
            status="INCORRECT",
            return_pct=-5.67,
        )
        assert outcome.return_pct == -5.67


class TestScoringResult:
    """Tests for ScoringResult model."""

    def test_empty_outcomes(self) -> None:
        """Empty outcomes list is valid."""
        result = ScoringResult(
            started_at="2026-01-15T10:00:00+00:00",
            completed_at="2026-01-15T10:00:05+00:00",
            duration_seconds=5.0,
            predictions_found=0,
            predictions_scored=0,
            predictions_skipped=0,
            outcomes=[],
        )
        assert result.outcomes == []
        assert result.predictions_found == 0

    def test_duration_non_negative(self) -> None:
        """duration_seconds must be >= 0."""
        with pytest.raises(ValidationError):
            ScoringResult(
                started_at="2026-01-15T10:00:00+00:00",
                completed_at="2026-01-15T10:00:05+00:00",
                duration_seconds=-1.0,
                predictions_found=0,
                predictions_scored=0,
                predictions_skipped=0,
                outcomes=[],
            )

    def test_predictions_found_non_negative(self) -> None:
        """predictions_found must be >= 0."""
        with pytest.raises(ValidationError):
            ScoringResult(
                started_at="2026-01-15T10:00:00+00:00",
                completed_at="2026-01-15T10:00:05+00:00",
                duration_seconds=5.0,
                predictions_found=-1,
                predictions_scored=0,
                predictions_skipped=0,
                outcomes=[],
            )

    def test_with_outcomes(self) -> None:
        """Result with multiple outcomes is valid."""
        outcomes = [
            ScoringOutcome(prediction_id="a", ticker="AAPL", status="CORRECT", return_pct=2.5),
            ScoringOutcome(
                prediction_id="b",
                ticker="MSFT",
                skipped=True,
                skip_reason="price_data_unavailable",
            ),
        ]
        result = ScoringResult(
            started_at="2026-01-15T10:00:00+00:00",
            completed_at="2026-01-15T10:00:05+00:00",
            duration_seconds=5.0,
            predictions_found=2,
            predictions_scored=1,
            predictions_skipped=1,
            outcomes=outcomes,
        )
        assert result.predictions_found == 2
        assert result.predictions_scored == 1
        assert result.predictions_skipped == 1
        assert len(result.outcomes) == 2
