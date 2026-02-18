"""Tests for predictions DB Pydantic models."""

from __future__ import annotations

import json

import pytest
from pydantic import ValidationError

from stock_radar.models.predictions import (
    AgentAccuracyResponse,
    AgentStats,
    LogPredictionResponse,
    PredictionHistoryResponse,
    PredictionRecord,
    ScorePredictionResponse,
)


class TestPredictionRecord:
    """Tests for the PredictionRecord model."""

    def test_construction_unscored(self) -> None:
        record = PredictionRecord(
            id="pred-001",
            ticker="AAPL",
            agent_name="earnings_linguist",
            signal_type="earnings_sentiment",
            direction="BULLISH",
            confidence=0.85,
            reasoning="Strong revenue guidance language in Q4 transcript.",
            prediction_date="2025-01-15",
            horizon_days=30,
            created_at="2025-01-15T10:30:00Z",
        )
        assert record.id == "pred-001"
        assert record.ticker == "AAPL"
        assert record.direction == "BULLISH"
        assert record.confidence == 0.85
        assert record.scored_at is None
        assert record.actual_price_close is None
        assert record.actual_price_at_horizon is None
        assert record.return_pct is None
        assert record.status is None

    def test_construction_scored(self) -> None:
        record = PredictionRecord(
            id="pred-002",
            ticker="MSFT",
            agent_name="narrative_divergence",
            signal_type="price_narrative_gap",
            direction="BEARISH",
            confidence=0.72,
            reasoning="Negative narrative trend diverges from recent price action.",
            prediction_date="2025-01-10",
            horizon_days=14,
            created_at="2025-01-10T08:00:00Z",
            scored_at="2025-01-24T08:00:00Z",
            actual_price_close=420.50,
            actual_price_at_horizon=405.30,
            return_pct=-3.61,
            status="CORRECT",
        )
        assert record.status == "CORRECT"
        assert record.scored_at == "2025-01-24T08:00:00Z"
        assert record.actual_price_close == 420.50
        assert record.actual_price_at_horizon == 405.30
        assert record.return_pct == -3.61

    def test_missing_required_field_raises(self) -> None:
        with pytest.raises(ValidationError):
            PredictionRecord(
                id="pred-003",
                ticker="GOOG",
                # missing agent_name, signal_type, direction, confidence,
                # reasoning, prediction_date, horizon_days, created_at
            )

    def test_confidence_bounds(self) -> None:
        with pytest.raises(ValidationError):
            PredictionRecord(
                id="pred-004",
                ticker="TSLA",
                agent_name="earnings_linguist",
                signal_type="earnings_sentiment",
                direction="BULLISH",
                confidence=1.5,
                reasoning="Overconfident prediction.",
                prediction_date="2025-02-01",
                horizon_days=7,
                created_at="2025-02-01T12:00:00Z",
            )
        with pytest.raises(ValidationError):
            PredictionRecord(
                id="pred-005",
                ticker="TSLA",
                agent_name="earnings_linguist",
                signal_type="earnings_sentiment",
                direction="BEARISH",
                confidence=-0.1,
                reasoning="Negative confidence makes no sense.",
                prediction_date="2025-02-01",
                horizon_days=7,
                created_at="2025-02-01T12:00:00Z",
            )

    def test_invalid_direction_raises(self) -> None:
        with pytest.raises(ValidationError):
            PredictionRecord(
                id="pred-006",
                ticker="NVDA",
                agent_name="earnings_linguist",
                signal_type="earnings_sentiment",
                direction="SIDEWAYS",
                confidence=0.5,
                reasoning="Invalid direction value.",
                prediction_date="2025-02-01",
                horizon_days=7,
                created_at="2025-02-01T12:00:00Z",
            )

    def test_serialization_roundtrip(self) -> None:
        record = PredictionRecord(
            id="pred-007",
            ticker="AMZN",
            agent_name="sec_filing_analyzer",
            signal_type="filing_anomaly",
            direction="NEUTRAL",
            confidence=0.45,
            reasoning="No significant signals detected in recent 10-K.",
            prediction_date="2025-01-20",
            horizon_days=60,
            created_at="2025-01-20T14:00:00Z",
            scored_at="2025-03-21T14:00:00Z",
            actual_price_close=190.00,
            actual_price_at_horizon=192.50,
            return_pct=1.32,
            status="PARTIAL",
        )
        data = json.loads(record.model_dump_json())
        restored = PredictionRecord(**data)
        assert restored == record


class TestLogPredictionResponse:
    """Tests for the LogPredictionResponse model."""

    def test_construction(self) -> None:
        resp = LogPredictionResponse(
            prediction_id="pred-001",
            created_at="2025-01-15T10:30:00Z",
        )
        assert resp.prediction_id == "pred-001"
        assert resp.created_at == "2025-01-15T10:30:00Z"

    def test_serialization_roundtrip(self) -> None:
        resp = LogPredictionResponse(
            prediction_id="pred-002",
            created_at="2025-01-16T09:00:00Z",
        )
        data = json.loads(resp.model_dump_json())
        restored = LogPredictionResponse(**data)
        assert restored == resp


class TestScorePredictionResponse:
    """Tests for the ScorePredictionResponse model."""

    def test_construction(self) -> None:
        resp = ScorePredictionResponse(
            prediction_id="pred-001",
            status="CORRECT",
            return_pct=5.23,
            direction="BULLISH",
            confidence=0.85,
        )
        assert resp.prediction_id == "pred-001"
        assert resp.status == "CORRECT"
        assert resp.return_pct == 5.23

    def test_invalid_status_raises(self) -> None:
        with pytest.raises(ValidationError):
            ScorePredictionResponse(
                prediction_id="pred-002",
                status="MAYBE",
                return_pct=1.0,
                direction="BEARISH",
                confidence=0.6,
            )

    def test_serialization_roundtrip(self) -> None:
        resp = ScorePredictionResponse(
            prediction_id="pred-003",
            status="INCORRECT",
            return_pct=-2.15,
            direction="BULLISH",
            confidence=0.70,
        )
        data = json.loads(resp.model_dump_json())
        restored = ScorePredictionResponse(**data)
        assert restored == resp


class TestPredictionHistoryResponse:
    """Tests for the PredictionHistoryResponse model."""

    def test_with_predictions(self) -> None:
        prediction = PredictionRecord(
            id="pred-001",
            ticker="AAPL",
            agent_name="earnings_linguist",
            signal_type="earnings_sentiment",
            direction="BULLISH",
            confidence=0.85,
            reasoning="Strong guidance language.",
            prediction_date="2025-01-15",
            horizon_days=30,
            created_at="2025-01-15T10:30:00Z",
        )
        resp = PredictionHistoryResponse(
            predictions=[prediction],
            total_count=1,
        )
        assert len(resp.predictions) == 1
        assert resp.total_count == 1

    def test_empty_predictions(self) -> None:
        resp = PredictionHistoryResponse(
            predictions=[],
            total_count=0,
        )
        assert resp.predictions == []
        assert resp.total_count == 0

    def test_serialization_roundtrip(self) -> None:
        prediction = PredictionRecord(
            id="pred-010",
            ticker="GOOG",
            agent_name="cross_sector_mapper",
            signal_type="contagion_risk",
            direction="BEARISH",
            confidence=0.60,
            reasoning="Semiconductor supply chain disruption spreading.",
            prediction_date="2025-02-01",
            horizon_days=21,
            created_at="2025-02-01T11:00:00Z",
        )
        resp = PredictionHistoryResponse(
            predictions=[prediction],
            total_count=1,
        )
        data = json.loads(resp.model_dump_json())
        restored = PredictionHistoryResponse(**data)
        assert restored == resp


class TestAgentStats:
    """Tests for the AgentStats model."""

    def test_construction(self) -> None:
        stats = AgentStats(
            agent_name="earnings_linguist",
            signal_type="earnings_sentiment",
            total=100,
            scored=80,
            correct=52,
            accuracy_pct=65.0,
            avg_confidence=0.73,
            avg_return_when_correct=4.25,
        )
        assert stats.agent_name == "earnings_linguist"
        assert stats.total == 100
        assert stats.scored == 80
        assert stats.correct == 52
        assert stats.accuracy_pct == 65.0
        assert stats.avg_confidence == 0.73
        assert stats.avg_return_when_correct == 4.25

    def test_serialization_roundtrip(self) -> None:
        stats = AgentStats(
            agent_name="sec_filing_analyzer",
            signal_type="filing_anomaly",
            total=50,
            scored=30,
            correct=18,
            accuracy_pct=60.0,
            avg_confidence=0.68,
            avg_return_when_correct=3.10,
        )
        data = json.loads(stats.model_dump_json())
        restored = AgentStats(**data)
        assert restored == stats


class TestAgentAccuracyResponse:
    """Tests for the AgentAccuracyResponse model."""

    def test_with_stats(self) -> None:
        stats = AgentStats(
            agent_name="earnings_linguist",
            signal_type="earnings_sentiment",
            total=100,
            scored=80,
            correct=52,
            accuracy_pct=65.0,
            avg_confidence=0.73,
            avg_return_when_correct=4.25,
        )
        resp = AgentAccuracyResponse(agent_stats=[stats])
        assert len(resp.agent_stats) == 1
        assert resp.agent_stats[0].agent_name == "earnings_linguist"

    def test_empty_stats(self) -> None:
        resp = AgentAccuracyResponse(agent_stats=[])
        assert resp.agent_stats == []
