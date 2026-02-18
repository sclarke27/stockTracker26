"""Tests for shared agent models."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from stock_radar.agents.models import AgentInput, AgentOutput, AnalysisResult


class TestAgentInput:
    def test_construction(self) -> None:
        inp = AgentInput(ticker="AAPL", quarter=4, year=2024)
        assert inp.ticker == "AAPL"
        assert inp.quarter == 4
        assert inp.year == 2024

    def test_quarter_bounds(self) -> None:
        with pytest.raises(ValidationError):
            AgentInput(ticker="AAPL", quarter=0, year=2024)
        with pytest.raises(ValidationError):
            AgentInput(ticker="AAPL", quarter=5, year=2024)


class TestAnalysisResult:
    def test_construction(self) -> None:
        result = AnalysisResult(
            ticker="AAPL",
            direction="BULLISH",
            confidence=0.85,
            reasoning="Strong forward guidance.",
            horizon_days=5,
            model_used="llama3.1:8b",
        )
        assert result.direction == "BULLISH"
        assert result.escalated is False

    def test_confidence_bounds(self) -> None:
        with pytest.raises(ValidationError):
            AnalysisResult(
                ticker="X",
                direction="BULLISH",
                confidence=1.5,
                reasoning="r",
                horizon_days=5,
                model_used="m",
            )

    def test_invalid_direction(self) -> None:
        with pytest.raises(ValidationError):
            AnalysisResult(
                ticker="X",
                direction="UP",
                confidence=0.5,
                reasoning="r",
                horizon_days=5,
                model_used="m",
            )


class TestAgentOutput:
    def test_construction(self) -> None:
        result = AnalysisResult(
            ticker="AAPL",
            direction="BEARISH",
            confidence=0.7,
            reasoning="Risk language detected.",
            horizon_days=10,
            model_used="claude-sonnet-4-20250514",
            escalated=True,
        )
        output = AgentOutput(prediction_id="abc-123", result=result)
        assert output.prediction_id == "abc-123"
        assert output.similar_past_reasoning == []
